import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


cpp_source = r'''
#include <torch/extension.h>

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
'''


cuda_source = r'''
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

__device__ __forceinline__ float half_raw_to_float(const __half_raw& raw) {
    return __half2float(__ushort_as_half(raw.x));
}

__device__ __forceinline__ float decode_fp4(uint8_t nibble) {
    __nv_fp4_storage_t storage = static_cast<__nv_fp4_storage_t>(nibble & 0xF);
    __half_raw raw = __nv_cvt_fp4_to_halfraw(storage, __NV_E2M1);
    return half_raw_to_float(raw);
}

__device__ __forceinline__ float decode_fp8(int8_t byte) {
    __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(byte);
    __half_raw raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
    return half_raw_to_float(raw);
}


// CUDA-core reference using the same launch geometry as sub_tma: grid(m_tiles, k_tiles, L), block(128)
// Each block handles one M-tile (128 rows) and one K-tile (64 cols) for a given L slice.
// We compute a partial sum over that K-tile and atomic add to output.

constexpr int M_TILE = 128;
constexpr int K_TILE = 64;

__global__ void gemv_ref_tile_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L,
    size_t stride_a_bytes,
    size_t stride_sfa_bytes,
    size_t stride_b_bytes,
    size_t stride_sfb_bytes
) {
    int tile_m = blockIdx.x;
    int tile_k = blockIdx.y;
    int tile_l = blockIdx.z;

    int m_base = tile_m * M_TILE;
    int k_base = tile_k * K_TILE;
    if (m_base >= M || k_base >= K || tile_l >= L) return;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5; // 0..3

    // Pointers to start of tile
    const uint8_t* g_a_tile   = reinterpret_cast<const uint8_t*>(a)   + tile_l * stride_a_bytes   + (m_base * K + k_base) / 2;
    const uint8_t* g_sfa_tile = reinterpret_cast<const uint8_t*>(sfa) + tile_l * stride_sfa_bytes + m_base * (K / 16) + k_base / 16;
    const uint8_t* g_b_tile   = reinterpret_cast<const uint8_t*>(b)   + tile_l * stride_b_bytes   + k_base / 2;
    const uint8_t* g_sfb_tile = reinterpret_cast<const uint8_t*>(sfb) + tile_l * stride_sfb_bytes + k_base / 16;

    // Each warp handles a subset of rows; we’ll map warp 0..3 across 128 rows
    for (int row = warp * 32 + lane; row < M_TILE; row += 128) {
        int global_row = m_base + row;
        if (global_row >= M) continue;

        float acc = 0.f;
        const uint8_t* row_a = g_a_tile + row * (K_TILE / 2);
        const uint8_t* row_sfa = g_sfa_tile + row * (K_TILE / 16);

        // Iterate over this K tile (64 elements) sequentially
        for (int kk = 0; kk < K_TILE; ++kk) {
            int byte_idx = kk >> 1;
            uint8_t a_byte = row_a[byte_idx];
            uint8_t b_byte = g_b_tile[byte_idx];

            uint8_t a_nib = (kk & 1) ? (a_byte >> 4) : (a_byte & 0xF);
            uint8_t b_nib = (kk & 1) ? (b_byte >> 4) : (b_byte & 0xF);

            float a_val = decode_fp4(a_nib);
            float b_val = decode_fp4(b_nib);

            int sf_idx = kk >> 4;
            float sf_a = decode_fp8(static_cast<int8_t>(row_sfa[sf_idx]));
            float sf_b = decode_fp8(static_cast<int8_t>(g_sfb_tile[sf_idx]));

            acc += (a_val * sf_a) * (b_val * sf_b);
        }

        // Atomic accumulate into C (same as tcgen05 path)
        atomicAdd(reinterpret_cast<half*>(&c[global_row * L + tile_l]), __float2half(acc));
    }
}


torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int m_tiles = (M + M_TILE - 1) / M_TILE;
    int k_tiles = (K + K_TILE - 1) / K_TILE;

    dim3 grid(m_tiles, k_tiles, L);
    dim3 block(128, 1, 1); // match sub_tma geometry

    size_t stride_a_bytes   = static_cast<size_t>(M) * (K / 2);
    size_t stride_sfa_bytes = static_cast<size_t>(M) * (K / 16);
    size_t stride_b_bytes   = (K / 2);
    size_t stride_sfb_bytes = (K / 16);

    gemv_ref_tile_kernel<<<grid, block>>>(
        reinterpret_cast<const int8_t*>(a.data_ptr()),
        reinterpret_cast<const int8_t*>(b.data_ptr()),
        reinterpret_cast<const int8_t*>(sfa.data_ptr()),
        reinterpret_cast<const int8_t*>(sfb.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, K, L,
        stride_a_bytes,
        stride_sfa_bytes,
        stride_b_bytes,
        stride_sfb_bytes
    );

    return c;
}
'''


module = load_inline(
    name='batched_scaled_gemv_ref_tiled',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batched_scaled_gemv_cuda'],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-std=c++17',
        '-gencode=arch=compute_100a,code=sm_100a'
    ],
    with_cuda=True,
    verbose=False
)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device

    c.zero_()

    a_i8 = a.view(torch.int8)
    b_i8 = b.view(torch.int8)
    sfa_i8 = sfa_ref.to(device=device, non_blocking=True).view(torch.int8)
    sfb_i8 = sfb_ref.to(device=device, non_blocking=True).view(torch.int8)

    return module.batched_scaled_gemv_cuda(a_i8, b_i8, sfa_i8, sfb_i8, c)
