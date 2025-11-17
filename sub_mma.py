"""Tensor-core implementation scaffold for nvfp4 GEMV on Blackwell.

Implements the first three steps of plan.md:
1. PyTorch extension + kernel skeleton in sub_mma.py.
2. Explicit tile/launch sizing constants for tcgen05 usage.
3. Tensor Memory allocation helpers and stub usage inside the CUDA kernel.
"""

from __future__ import annotations

import os
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

SUB_MMA_DEBUG = os.getenv("SUB_MMA_DEBUG", "1") not in ("", "0", "false", "False", "FALSE")

TILE_M = 128
TILE_N = 64  # Logical surrogate dimension for tcgen05.mma
TILE_K = 64
WARP_SIZE = 32
COMPUTE_WARPS = 2
TMA_WARPS = 2
BLOCK_WARPS = COMPUTE_WARPS + TMA_WARPS
THREADS_PER_BLOCK = BLOCK_WARPS * WARP_SIZE

CPP_SOURCE = """
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_mma_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c);
"""

cuda_source = f"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

constexpr int kTileM = {TILE_M};
constexpr int kTileN = {TILE_N};
constexpr int kTileK = {TILE_K};
constexpr int kWarpSize = {WARP_SIZE};
constexpr int kComputeWarps = {COMPUTE_WARPS};
constexpr int kTmaWarps = {TMA_WARPS};
constexpr int kBlockWarps = {BLOCK_WARPS};
constexpr int kThreadsPerBlock = {THREADS_PER_BLOCK};
constexpr bool kEnableDebug = {str(SUB_MMA_DEBUG).lower()};
constexpr bool kStubMode = true;  // flip to false once MMA path lands

constexpr int kTmemColsA = 64;
constexpr int kTmemColsB = 64;
constexpr int kTmemColsScale = 32;
constexpr int kTmemColsAccumulator = 64;
constexpr int kDebugBytesPerLane = 4;

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

template <int Columns>
__device__ __forceinline__ void tmem_alloc_cols(unsigned &handle, bool warp_active) {{
    if (!warp_active) {{
        return;
    }}
    unsigned lane = threadIdx.x & (kWarpSize - 1);
    unsigned smem_ptr = 0;
    if (lane == 0) {{
        smem_ptr = __cvta_generic_to_shared(&handle);
    }}
    smem_ptr = __shfl_sync(0xffffffff, smem_ptr, 0);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(smem_ptr), "n"(Columns));
}}

template <int Columns>
__device__ __forceinline__ void tmem_dealloc_cols(unsigned handle, bool warp_active) {{
    if (!warp_active) {{
        return;
    }}
    unsigned lane = threadIdx.x & (kWarpSize - 1);
    unsigned tmem_base = (lane == 0) ? handle : 0u;
    tmem_base = __shfl_sync(0xffffffff, tmem_base, 0);
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :
        : "r"(tmem_base), "n"(Columns));
}}

__device__ __forceinline__ void tmem_relinquish() {{
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}}

__global__ void gemv_nvfp4_mma_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L,
    int N_rows
) {{
    (void)b;
    (void)sfa;
    (void)sfb;
    (void)c;
    (void)N_rows;

    const uint8_t* base_a = reinterpret_cast<const uint8_t*>(a);
    const size_t K_half = static_cast<size_t>(K) / 2;
    const size_t batch_stride_a = static_cast<size_t>(M) * K_half;
    const uint8_t* batch_a = base_a + static_cast<size_t>(blockIdx.y) * batch_stride_a;

    extern __shared__ uint8_t shared_bytes[];
    uint32_t* debug_stage = reinterpret_cast<uint32_t*>(shared_bytes);

    __shared__ unsigned tmem_a_base;
    __shared__ unsigned tmem_b_base;
    __shared__ unsigned tmem_sfa_base;
    __shared__ unsigned tmem_sfb_base;
    __shared__ unsigned tmem_d_base;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const bool warp0 = threadIdx.x < kWarpSize;
    const int tile_m_start = blockIdx.x * kTileM;
    const int tile_row = tile_m_start + lane;

    if (warp0) {{
        tmem_alloc_cols<kTmemColsA>(tmem_a_base, true);
        tmem_alloc_cols<kTmemColsB>(tmem_b_base, true);
        tmem_alloc_cols<kTmemColsScale>(tmem_sfa_base, true);
        tmem_alloc_cols<kTmemColsScale>(tmem_sfb_base, true);
        tmem_alloc_cols<kTmemColsAccumulator>(tmem_d_base, true);
    }}
    __syncthreads();

    if (kEnableDebug && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {{
        printf("[sub_mma] stub CTA launched M=%d K=%d L=%d, TMEM bases A=%u B=%u SFA=%u SFB=%u D=%u\\n",
               M, K, L,
               tmem_a_base, tmem_b_base,
               tmem_sfa_base, tmem_sfb_base, tmem_d_base);
    }}

    if (kEnableDebug) {{
        if (warp0) {{
            uint32_t gmem_word = 0;
            if (tile_row < M && K_half > 0) {{
                const uint8_t* row_ptr = batch_a + static_cast<size_t>(tile_row) * K_half;
                uint32_t assembled = 0;
                int bytes_to_copy = static_cast<int>(K_half);
                bytes_to_copy = bytes_to_copy < kDebugBytesPerLane ? bytes_to_copy : kDebugBytesPerLane;
                #pragma unroll
                for (int i = 0; i < kDebugBytesPerLane; ++i) {{
                    uint32_t byte_val = (i < bytes_to_copy) ? static_cast<uint32_t>(row_ptr[i]) : 0u;
                    assembled |= (byte_val << (8 * i));
                }}
                gmem_word = assembled;
            }}
            debug_stage[lane] = gmem_word;
        }}
        __syncthreads();

        if (warp0) {{
            uint32_t smem_word = debug_stage[lane];
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {{%1}};"
                :
                : "r"(tmem_a_base), "r"(smem_word));
            asm volatile("tcgen05.wait::st.sync.aligned;");
            uint32_t tmem_word = 0;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x1.b32 {{%0}}, [%1];"
                : "=r"(tmem_word)
                : "r"(tmem_a_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if (tile_row < M && smem_word != tmem_word) {{
                printf("[sub_mma][error] TMEM mismatch lane=%d smem=0x%08x tmem=0x%08x\\n",
                       lane, smem_word, tmem_word);
                asm volatile("trap;");
            }}
            if (lane == 0) {{
                printf("[sub_mma] debug TMEM roundtrip ok for CTA (%d,%d) rows [%d,%d)\\n",
                       blockIdx.x, blockIdx.y, tile_m_start, tile_m_start + kWarpSize);
            }}
        }}
        __syncthreads();
    }}

    __syncthreads();

    if constexpr (kStubMode) {{
        if (warp0) {{
            tmem_dealloc_cols<kTmemColsA>(tmem_a_base, true);
            tmem_dealloc_cols<kTmemColsB>(tmem_b_base, true);
            tmem_dealloc_cols<kTmemColsScale>(tmem_sfa_base, true);
            tmem_dealloc_cols<kTmemColsScale>(tmem_sfb_base, true);
            tmem_dealloc_cols<kTmemColsAccumulator>(tmem_d_base, true);
            tmem_relinquish();
        }}
        __syncthreads();
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {{
            printf("[sub_mma] kernel stub hit (no computation performed).\\n");
        }}
        asm volatile("trap;");
        return;
    }}

    // Future implementation will go here (TMA loads, tcgen05.mma, reductions).
}}

torch::Tensor batched_scaled_gemv_mma_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c
) {{
    TORCH_CHECK(a.is_cuda(), "input a must be CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "input b must be CUDA tensor");
    TORCH_CHECK(sfa.is_cuda(), "input sfa must be CUDA tensor");
    TORCH_CHECK(sfb.is_cuda(), "input sfb must be CUDA tensor");
    TORCH_CHECK(c.is_cuda(), "output c must be CUDA tensor");

    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    dim3 grid((M + kTileM - 1) / kTileM, L);
    dim3 block(kThreadsPerBlock);
    size_t shared_bytes = static_cast<size_t>(kTileM) * sizeof(float);  // placeholder for reductions

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    gemv_nvfp4_mma_kernel<<<grid, block, shared_bytes>>>(
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        M,
        K,
        L,
        N_rows);

    return c;
}}
"""

module = load_inline(
    name="batched_scaled_gemv_mma",
    cpp_sources=CPP_SOURCE,
    cuda_sources=cuda_source,
    functions=["batched_scaled_gemv_mma_cuda"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a"
    ],
    with_cuda=True,
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device

    a_i8 = a.view(torch.int8)
    b_i8 = b.view(torch.int8)
    sfa_i8 = sfa_ref.to(device=device, non_blocking=True).view(torch.int8)
    sfb_i8 = sfb_ref.to(device=device, non_blocking=True).view(torch.int8)

    return module.batched_scaled_gemv_mma_cuda(
        a_i8,
        b_i8,
        sfa_i8,
        sfb_i8,
        c,
    )
