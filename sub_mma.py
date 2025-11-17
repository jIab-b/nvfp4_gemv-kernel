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
constexpr int kTileKBytes = kTileK / 2;
constexpr int kScalePerRow = kTileK / 16;
constexpr int kSmemTileBytes = kTileM * kTileKBytes;
constexpr int kSmemScaleBytes = kTileM * kScalePerRow;

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

__device__ __forceinline__ uint64_t encode_field(uint32_t value) {{
    return (static_cast<uint64_t>(value & 0x3FFFF) >> 4) & 0x3FFFULL;
}}

__device__ __forceinline__ uint64_t encode_smem_descriptor(uint32_t start_addr, uint32_t leading_bytes, uint32_t stride_bytes) {{
    uint64_t desc = 0;
    desc |= encode_field(start_addr);
    desc |= encode_field(leading_bytes) << 16;
    desc |= encode_field(stride_bytes) << 32;
    desc |= (1ull << 46);
    desc |= (static_cast<uint64_t>(0xb0) << 53);
    // swizzle bits left as zero (no swizzle)
    return desc;
}}

__device__ __forceinline__ void mbarrier_init(uint64_t* ptr, unsigned count) {{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(ptr), "r"(count));
}}

__device__ __forceinline__ void mbarrier_wait(uint64_t* ptr, unsigned parity) {{
    unsigned ready = 0;
    do {{
        asm volatile(
            "mbarrier.try_wait.parity.shared::cta.b64 %0, [%1], %2;"
            : "=r"(ready)
            : "r"(ptr), "r"(parity));
    }} while (!ready);
}}

__device__ __forceinline__ void tensor_cp_128x256(unsigned tmem_base, uint64_t desc, uint64_t* mbar_ptr, unsigned& parity) {{
    asm volatile(
        "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
        :
        : "r"(tmem_base), "l"(desc));
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :
        : "r"(mbar_ptr));
    mbarrier_wait(mbar_ptr, parity);
    parity ^= 1u;
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
    const uint8_t* base_b = reinterpret_cast<const uint8_t*>(b);
    const uint8_t* base_sfa = reinterpret_cast<const uint8_t*>(sfa);
    const uint8_t* base_sfb = reinterpret_cast<const uint8_t*>(sfb);
    const size_t K_half = static_cast<size_t>(K) / 2;
    const size_t K_sf = static_cast<size_t>(K) / 16;
    const size_t batch_stride_a = static_cast<size_t>(M) * K_half;
    const size_t batch_stride_b = static_cast<size_t>(N_rows) * K_half;
    const size_t batch_stride_sfa = static_cast<size_t>(M) * K_sf;
    const size_t batch_stride_sfb = static_cast<size_t>(N_rows) * K_sf;
    const uint8_t* batch_a = base_a + static_cast<size_t>(blockIdx.y) * batch_stride_a;
    const uint8_t* batch_b = base_b + static_cast<size_t>(blockIdx.y) * batch_stride_b;
    const uint8_t* batch_sfa = base_sfa + static_cast<size_t>(blockIdx.y) * batch_stride_sfa;
    const uint8_t* batch_sfb = base_sfb + static_cast<size_t>(blockIdx.y) * batch_stride_sfb;

    extern __shared__ __align__(16) uint8_t shared_bytes[];
    uint8_t* smem_a = shared_bytes;
    uint8_t* smem_b = smem_a + kSmemTileBytes;
    uint8_t* smem_sfa = smem_b + kSmemTileBytes;
    uint8_t* smem_sfb = smem_sfa + kSmemScaleBytes;
    uint8_t* smem_after_scales = smem_sfb + kSmemScaleBytes;
    uint64_t* tma_mbar = reinterpret_cast<uint64_t*>(smem_after_scales);

    __shared__ unsigned tmem_a_base;
    __shared__ unsigned tmem_b_base;
    __shared__ unsigned tmem_sfa_base;
    __shared__ unsigned tmem_sfb_base;
    __shared__ unsigned tmem_d_base;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const bool warp0 = threadIdx.x < kWarpSize;
    const int tile_m_start = blockIdx.x * kTileM;

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

    if (threadIdx.x == 0) {{
        mbarrier_init(tma_mbar, 1);
    }}
    __syncthreads();

    const int total_k_tiles = (K + kTileK - 1) / kTileK;
    const int guarded_rows = N_rows > 0 ? N_rows : 1;
    unsigned cp_parity = 0;

    for (int kt = 0; kt < total_k_tiles; ++kt) {{
        const int k_byte_offset = kt * kTileKBytes;
        const int k_scale_offset = kt * kScalePerRow;

        for (int idx = threadIdx.x; idx < kSmemTileBytes; idx += blockDim.x) {{
            int local_row = idx / kTileKBytes;
            int local_byte = idx % kTileKBytes;
            size_t global_row = tile_m_start + local_row;
            size_t global_byte = static_cast<size_t>(k_byte_offset) + local_byte;
            uint8_t value = 0;
            if (global_row < static_cast<size_t>(M) && global_byte < K_half) {{
                value = batch_a[global_row * K_half + global_byte];
            }}
            smem_a[idx] = value;
        }}

        for (int idx = threadIdx.x; idx < kSmemTileBytes; idx += blockDim.x) {{
            int local_row = idx / kTileKBytes;
            int local_byte = idx % kTileKBytes;
            size_t global_row = local_row % guarded_rows;
            size_t global_byte = static_cast<size_t>(k_byte_offset) + local_byte;
            uint8_t value = 0;
            if (global_row < static_cast<size_t>(N_rows) && global_byte < K_half) {{
                value = batch_b[global_row * K_half + global_byte];
            }}
            smem_b[idx] = value;
        }}

        for (int idx = threadIdx.x; idx < kSmemScaleBytes; idx += blockDim.x) {{
            int local_row = idx / kScalePerRow;
            int local_byte = idx % kScalePerRow;
            size_t global_row = tile_m_start + local_row;
            size_t global_byte = static_cast<size_t>(k_scale_offset) + local_byte;
            uint8_t value = 0;
            if (global_row < static_cast<size_t>(M) && global_byte < K_sf) {{
                value = batch_sfa[global_row * K_sf + global_byte];
            }}
            smem_sfa[idx] = value;
        }}

        for (int idx = threadIdx.x; idx < kSmemScaleBytes; idx += blockDim.x) {{
            int local_row = idx / kScalePerRow;
            int local_byte = idx % kScalePerRow;
            size_t global_row = local_row % guarded_rows;
            size_t global_byte = static_cast<size_t>(k_scale_offset) + local_byte;
            uint8_t value = 0;
            if (global_row < static_cast<size_t>(N_rows) && global_byte < K_sf) {{
                value = batch_sfb[global_row * K_sf + global_byte];
            }}
            smem_sfb[idx] = value;
        }}

        __syncthreads();

        if (warp0) {{
            uint32_t smem_addr_a = __cvta_generic_to_shared(smem_a);
            uint32_t smem_addr_b = __cvta_generic_to_shared(smem_b);
            uint32_t smem_addr_sfa = __cvta_generic_to_shared(smem_sfa);
            uint32_t smem_addr_sfb = __cvta_generic_to_shared(smem_sfb);
            uint64_t desc_a = encode_smem_descriptor(smem_addr_a, kTileKBytes, kTileKBytes);
            uint64_t desc_b = encode_smem_descriptor(smem_addr_b, kTileKBytes, kTileKBytes);
            uint64_t desc_sfa = encode_smem_descriptor(smem_addr_sfa, kScalePerRow, kScalePerRow);
            uint64_t desc_sfb = encode_smem_descriptor(smem_addr_sfb, kScalePerRow, kScalePerRow);
            tensor_cp_128x256(tmem_a_base, desc_a, tma_mbar, cp_parity);
            tensor_cp_128x256(tmem_b_base, desc_b, tma_mbar, cp_parity);
            tensor_cp_128x256(tmem_sfa_base, desc_sfa, tma_mbar, cp_parity);
            tensor_cp_128x256(tmem_sfb_base, desc_sfb, tma_mbar, cp_parity);
        }}

        __syncthreads();

        if (kEnableDebug && warp0 && lane == 0) {{
            printf("[sub_mma] cp tile %d complete for CTA (%d,%d)\\n", kt, blockIdx.x, blockIdx.y);
        }}
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

    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    dim3 grid((M + kTileM - 1) / kTileM, L);
    dim3 block(kThreadsPerBlock);
    size_t shared_bytes =
        static_cast<size_t>(kSmemTileBytes) * 2 +
        static_cast<size_t>(kSmemScaleBytes) * 2 +
        sizeof(uint64_t);

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
