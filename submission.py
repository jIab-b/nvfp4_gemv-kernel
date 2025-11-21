import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_source = """
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = """
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

__global__ void gemv_nvfp4_scalar_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L,
    int N_rows
) {
    // Scalar CUDA core design: process ROWS_PER_CTA rows, reduce full K dimension
    #define THREADS_PER_CTA 128
    #define ROWS_PER_CTA 128
    #define K_TILE 128

    int m_block = blockIdx.x;
    int l = blockIdx.y;
    int tid = threadIdx.x;

    int rows_this_tile = min(ROWS_PER_CTA, M - m_block * ROWS_PER_CTA);
    if (rows_this_tile <= 0 || l >= L) return;

    const int K_sf = K / 16;
    const int K_half = K / 2;
    const size_t batch_stride_a = static_cast<size_t>(M) * K_half;
    const size_t batch_stride_b = static_cast<size_t>(N_rows) * K_half;
    const size_t batch_stride_sfa = static_cast<size_t>(M) * K_sf;
    const size_t batch_stride_sfb = static_cast<size_t>(N_rows) * K_sf;

    const uint8_t* base_a = reinterpret_cast<const uint8_t*>(a);
    const uint8_t* base_b = reinterpret_cast<const uint8_t*>(b);
    const uint8_t* base_sfa = reinterpret_cast<const uint8_t*>(sfa);
    const uint8_t* base_sfb = reinterpret_cast<const uint8_t*>(sfb);

    const uint8_t* batch_a = base_a + l * batch_stride_a;
    const uint8_t* batch_b = base_b + l * batch_stride_b;
    const uint8_t* batch_sfa = base_sfa + l * batch_stride_sfa;
    const uint8_t* batch_sfb = base_sfb + l * batch_stride_sfb;

    const uint8_t* row_a = batch_a + static_cast<size_t>(m_block * ROWS_PER_CTA) * K_half;
    const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m_block * ROWS_PER_CTA) * K_sf;

    // Double-buffered shared memory for pipelined loads
    __shared__ uint8_t vector_smem[2][K_TILE / 2];
    __shared__ int8_t vector_scale_smem[2][K_TILE / 16];
    __shared__ uint8_t matrix_smem[2][ROWS_PER_CTA * (K_TILE / 2)];
    __shared__ int8_t matrix_scale_smem[2][ROWS_PER_CTA * (K_TILE / 16)];

    // Each thread accumulates one row (ROWS_PER_CTA = 128 threads for 128 rows)
    float acc = 0.0f;

    // Pipelined loop over K tiles with double buffering
    int num_k_tiles = (K + K_TILE - 1) / K_TILE;

    for (int k_block = 0; k_block < num_k_tiles; ++k_block) {
        int buffer_idx = k_block & 1;  // Ping-pong: 0, 1, 0, 1...
        int k_start = k_block * K_TILE;
        int tile_elems = min(K_TILE, K - k_start);
        if (tile_elems <= 0) continue;

        // Load current tile into buffer_idx
        // Load vector (all threads help)
        for (int i = tid; i < (tile_elems + 1) / 2; i += THREADS_PER_CTA) {
            int byte_idx = k_start / 2 + i;
            if (byte_idx < batch_stride_b) {
                vector_smem[buffer_idx][i] = batch_b[byte_idx];
            }
        }

        // Load vector scales
        for (int i = tid; i < (tile_elems + 15) / 16; i += THREADS_PER_CTA) {
            int sf_idx = k_start / 16 + i;
            if (sf_idx < K_sf) {
                vector_scale_smem[buffer_idx][i] = batch_sfb[sf_idx];
            }
        }

        // Load matrix (each thread handles one row if tid < rows_this_tile)
        for (int row = tid; row < rows_this_tile; row += THREADS_PER_CTA) {
            const uint8_t* src = row_a + row * K_half + k_start / 2;
            uint8_t* dst = matrix_smem[buffer_idx] + row * (K_TILE / 2);
            for (int byte = 0; byte < (tile_elems + 1) / 2; ++byte) {
                dst[byte] = (k_start / 2 + byte < K_half) ? src[byte] : 0;
            }
        }

        // Load matrix scales
        for (int row = tid; row < rows_this_tile; row += THREADS_PER_CTA) {
            const uint8_t* src = row_sfa + row * K_sf + k_start / 16;
            int8_t* dst = matrix_scale_smem[buffer_idx] + row * (K_TILE / 16);
            for (int sf = 0; sf < (tile_elems + 15) / 16; ++sf) {
                dst[sf] = (k_start / 16 + sf < K_sf) ? src[sf] : 0;
            }
        }

        __syncthreads();

        // Compute current tile using current buffer
        // Each thread accumulates its assigned row
        int row = tid;
        if (row < rows_this_tile) {
            for (int kb = 0; kb < (tile_elems + 1) / 2; ++kb) {
                // Load FP4 nibble pairs from packed bytes
                uint8_t matrix_byte = matrix_smem[buffer_idx][row * (K_TILE / 2) + kb];
                uint8_t vector_byte = vector_smem[buffer_idx][kb];

                // Decode FP4 values (2 per byte)
                float a_lo = decode_fp4(matrix_byte & 0xF);
                float a_hi = decode_fp4(matrix_byte >> 4);
                float b_lo = decode_fp4(vector_byte & 0xF);
                float b_hi = decode_fp4(vector_byte >> 4);

                // FP8 scales (one per 16 FP4 elements = 8 bytes)
                int scale_idx = kb / 8;
                float sfa = decode_fp8(matrix_scale_smem[buffer_idx][row * (K_TILE / 16) + scale_idx]);
                float sfb = decode_fp8(vector_scale_smem[buffer_idx][scale_idx]);

                // Apply scales
                float scaled_a_lo = a_lo * sfa;
                float scaled_b_lo = b_lo * sfb;
                float scaled_a_hi = a_hi * sfa;
                float scaled_b_hi = b_hi * sfb;

                // FMA: acc = (a * sfa) * (b * sfb) + acc
                // Using PTX: fma.rn.f32 d, a, b, c; where d = a * b + c
                asm volatile("fma.rn.f32 %0, %1, %2, %0;"
                    : "+f"(acc) : "f"(scaled_a_lo), "f"(scaled_b_lo));
                asm volatile("fma.rn.f32 %0, %1, %2, %0;"
                    : "+f"(acc) : "f"(scaled_a_hi), "f"(scaled_b_hi));
            }
        }
    }

    __syncthreads();

    // Store result to output
    // Each thread with tid < rows_this_tile stores its accumulated value
    if (tid < rows_this_tile) {
        int m = m_block * ROWS_PER_CTA + tid;
        size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
        c[c_idx] = __float2half(acc);
    }
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    dim3 grid((M + 128 - 1) / 128, L);  // 128 = ROWS_PER_CTA
    dim3 block(128);  // THREADS_PER_CTA

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    gemv_nvfp4_scalar_kernel<<<grid, block>>>(
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        M, K, L,
        N_rows
    );

    return c;
}
"""

module = load_inline(
    name='batched_scaled_gemv_tc',
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

THREADS_PER_CTA = 128

ROWS_PER_CTA = 128

K_TILE = 128

def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device

    a_i8 = a.view(torch.int8)
    b_i8 = b.view(torch.int8)
    sfa_i8 = sfa_ref.to(device=device, non_blocking=True).view(torch.int8)
    sfb_i8 = sfb_ref.to(device=device, non_blocking=True).view(torch.int8)

    return module.batched_scaled_gemv_cuda(
        a_i8,
        b_i8,
        sfa_i8,
        sfb_i8,
        c
    )
