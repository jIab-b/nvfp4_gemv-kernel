"""
Tensor-core implementation using Py CuTe DSL for nvfp4 GEMV on Blackwell.
Implements batched scaled GEMV: C[m,l] = alpha * scale_a * A[m,k,l] @ (scale_b * B[1,k,l].T) + beta * C[m,l]
Uses cutlass_dsl to generate CUDA kernel with UMMA (tcgen05.mma mxf4nvf4.block_scale), TMEM, TMA.
"""

from __future__ import annotations

import os
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
import cutlass_dsl as dsl  # nvidia-cutlass-dsl

SUB_CUTLASS_DEBUG = os.getenv("SUB_CUTLASS_DEBUG", "1") not in ("", "0", "false", "False", "FALSE")

# Tile constants (adapted for tcgen05.mma mxf4nvf4: 128x64x64 supported)
TILE_M = 128
TILE_N = 64  # Padded for alignment (actual N=1 for GEMV, compute only first col)
TILE_K = 64
WARP_SIZE = 32
COMPUTE_WARPS = 2  # For UMMA issue + epilogue
TMA_WARPS = 2
BLOCK_WARPS = COMPUTE_WARPS + TMA_WARPS
THREADS_PER_BLOCK = BLOCK_WARPS * WARP_SIZE

# Element types
ElementA = dsl.Element("nv_float4_t<float_e2m1_t>")  # nvfp4 E2M1 (storage uint8_t nibbles)
ElementB = ElementA  # Symmetric
ElementScale = dsl.Element("float8_e4m3fn_t")  # fp8 E4M3 scales
ElementAccum = dsl.Element("float")  # Internal accum in TMEM
ElementC = dsl.Element("half_t")  # Output fp16

CPP_SOURCE = """
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_cutlass_cuda(
    torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

# DSL Kernel Definition
kernel_dsl = dsl.DSL(
    name="batched_scaled_gemv_cutlass",
    arch="sm100",  # Blackwell
    # MMA Atom: tcgen05.mma cta_group::1 kind::mxf4nvf4 block_scale scale_vec::2X (fp8 every 16 nvfp4)
    mma_atom=dsl.MmaAtom(
        kind="mxf4nvf4",  # nvfp4 inputs
        cta_group=1,  # Single SM
        block_scale=True,
        scale_vec=2,  # scale_vec::2X (every 16 elements, matches sf_vec_size=16)
        shape=(TILE_M, TILE_N, TILE_K),  # MxNxK
        a_type=ElementA,
        b_type=ElementB,
        accum_type=ElementAccum,
        a_layout=dsl.Layout("row_major_sw128", k_major=True),  # K-major 128B swizzle
        b_layout=dsl.Layout("col_major_sw128", k_major=True),  # For B^T in GEMV
        transpose_a=False,
        transpose_b=True  # B is [1,k] -> [k,1] col-major
    ),
    # TMEM Allocation: 128 columns (for 128x64 fp32 accum ~32KB)
    tmem_alloc=dsl.TmemAlloc(
        columns=128,
        mode="cta_group_1",  # Single CTA/SM
        allocator="Allocator1Sm"  # cute::TMEM::Allocator1Sm
    ),
    # TMA Loads: Async from GMEM -> SMEM for A/B/SFA/SFB tiles
    tma_loads=[
        dsl.TmaLoad(
            src="gmem",
            dst="smem",
            shape=(TILE_M, TILE_K),
            element=ElementA,
            layout="row_major_sw128",
            async=True,
            barrier="mbarrier"  # Sync with mbarrier
        ),
        dsl.TmaLoad(
            src="gmem",
            dst="smem",
            shape=(TILE_N, TILE_K),  # Padded N
            element=ElementB,
            layout="col_major_sw128",
            async=True
        ),
        dsl.TmaLoad(
            src="gmem",
            dst="smem",
            shape=(TILE_M, TILE_K // 16),  # Scales every 16 nvfp4
            element=ElementScale,
            layout="row_major",  # Scales simpler layout
            async=True
        ),
        dsl.TmaLoad(
            src="gmem",
            dst="smem",
            shape=(TILE_N, TILE_K // 16),
            element=ElementScale,
            layout="col_major",
            async=True
        )
    ],
    # Epilogue: Load TMEM -> RMEM (tcgen05.ld 32x32b.x1), alpha*A*B + beta*C, reduce N=1, store GMEM
    epilogue=dsl.Epilogue(
        load_atom="SM100_TMEM_LOAD_32dp32b1x",  # tcgen05.ld.sync.aligned.32x32b.x1
        ops=[
            dsl.Op("linear_combination", alpha=1.0, beta="param"),  # alpha * accum + beta * C
            dsl.Op("reduce_n", dim=1, mode="sum")  # GEMV: Sum padded N to vector (N=1)
        ],
        output_type=ElementC,
        store_async=True
    ),
    # Sync: mbarrier for TMA/UMMA
    sync="mbarrier",
    # Threads: 128 (4 warps: 2 TMA, 2 compute/epi)
    threads=THREADS_PER_BLOCK,
    # Pipeline: 4 stages (TMA + UMMA)
    stages=4
)

# Generate CUDA source from DSL
cuda_source = kernel_dsl.generate_kernel()

if SUB_CUTLASS_DEBUG:
    print("Generated CUDA source length:", len(cuda_source))
    print("TMEM columns:", kernel_dsl.tmem_alloc.columns)
    print("MMA shape:", kernel_dsl.mma_atom.shape)

# Compile with load_inline
module = load_inline(
    name="batched_scaled_gemv_cutlass",
    cpp_sources=CPP_SOURCE,
    cuda_sources=cuda_source,
    functions=["batched_scaled_gemv_cutlass_cuda"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a"  # Blackwell B200
    ],
    with_cuda=True,
    verbose=SUB_CUTLASS_DEBUG
)

def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device


    # View as elements (nvfp4 storage uint8, scales int8)
    a_view = a.view(torch.uint8).to(device=device)  # For DSL input
    b_view = b.view(torch.uint8).to(device=device)
    sfa_view = sfa_ref.view(torch.int8).to(device=device)
    sfb_view = sfb_ref.view(torch.int8).to(device=device)


    # Call generated kernel (handles batch L internally via grid)
    result = module.batched_scaled_gemv_cutlass_cuda(
        a_view, b_view, sfa_view, sfb_view, c
    )

    if SUB_CUTLASS_DEBUG:
        print("[sub_cutlass] Kernel completed")

    return result
