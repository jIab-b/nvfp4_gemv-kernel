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

// Inline PTX helpers for cp.async operations
__device__ __forceinline__ void cp_async_cg_shared_global_16B(
    void* __restrict__ dst_smem,
    const void* __restrict__ src_global
) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem))), "l"(src_global));
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
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
    // Optimized async pipeline design with cp.async and vectorized loads
    #define THREADS_PER_CTA 256
    #define ROWS_PER_CTA 128
    #define K_TILE 256

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

    // Double-buffered shared memory for async pipelined loads
    __shared__ uint8_t vector_smem[2][K_TILE / 2];
    __shared__ int8_t vector_scale_smem[2][K_TILE / 16];
    __shared__ uint8_t matrix_smem[2][ROWS_PER_CTA * (K_TILE / 2)];
    __shared__ int8_t matrix_scale_smem[2][ROWS_PER_CTA * (K_TILE / 16)];

    // Accumulators with ILP
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int num_k_tiles = (K + K_TILE - 1) / K_TILE;

    // Helper lambda to issue async loads for a given k_tile into buffer_idx
    auto issue_loads = [&](int k_tile, int buffer_idx) {
        int k_start = k_tile * K_TILE;
        int tile_elems = min(K_TILE, K - k_start);
        if (tile_elems <= 0) return;

        const int vector_bytes = (tile_elems + 1) / 2;
        const int vector_scales = (tile_elems + 15) / 16;
        const int matrix_bytes_per_row = (tile_elems + 1) / 2;
        const int matrix_scales_per_row = (tile_elems + 15) / 16;

        // Phase 2: Vectorized loads using uint4 (16 bytes at a time) with cp.async
        // Load vector data using cp.async (16-byte aligned chunks)
        for (int i = tid * 16; i < vector_bytes; i += THREADS_PER_CTA * 16) {
            if (i + 16 <= vector_bytes && (k_start / 2 + i + 16) <= batch_stride_b) {
                const void* src = batch_b + k_start / 2 + i;
                void* dst = &vector_smem[buffer_idx][i];
                cp_async_cg_shared_global_16B(dst, src);
            } else {
                // Handle non-aligned tail with smaller transfers
                for (int j = i; j < min(i + 16, vector_bytes); ++j) {
                    int byte_idx = k_start / 2 + j;
                    if (byte_idx < batch_stride_b) {
                        vector_smem[buffer_idx][j] = batch_b[byte_idx];
                    }
                }
            }
        }

        // Load vector scales
        for (int i = tid; i < vector_scales; i += THREADS_PER_CTA) {
            int sf_idx = k_start / 16 + i;
            if (sf_idx < K_sf) {
                vector_scale_smem[buffer_idx][i] = batch_sfb[sf_idx];
            }
        }

        // Load matrix data with vectorized access
        int row = tid;
        while (row < rows_this_tile) {
            const uint8_t* src_row = row_a + row * K_half + k_start / 2;
            uint8_t* dst_row = matrix_smem[buffer_idx] + row * (K_TILE / 2);

            // Use uint4 for vectorized 16-byte loads where possible
            int bytes_to_load = matrix_bytes_per_row;
            int byte_offset = 0;

            // Load 16-byte chunks using cp.async
            while (byte_offset + 16 <= bytes_to_load && (k_start / 2 + byte_offset + 16) <= K_half) {
                const void* src = src_row + byte_offset;
                void* dst = dst_row + byte_offset;
                cp_async_cg_shared_global_16B(dst, src);
                byte_offset += 16;
            }

            // Handle tail bytes
            while (byte_offset < bytes_to_load) {
                dst_row[byte_offset] = (k_start / 2 + byte_offset < K_half) ? src_row[byte_offset] : 0;
                byte_offset++;
            }

            row += THREADS_PER_CTA;
        }

        // Load matrix scales
        row = tid;
        while (row < rows_this_tile) {
            const uint8_t* src_sf = row_sfa + row * K_sf + k_start / 16;
            int8_t* dst_sf = matrix_scale_smem[buffer_idx] + row * (K_TILE / 16);
            for (int sf = 0; sf < matrix_scales_per_row; ++sf) {
                dst_sf[sf] = (k_start / 16 + sf < K_sf) ? src_sf[sf] : 0;
            }
            row += THREADS_PER_CTA;
        }
    };

    // Compute function for a given buffer
    auto compute_tile = [&](int buffer_idx, int tile_elems) {
        int row = tid;
        if (row < rows_this_tile) {
            const int tile_bytes = (tile_elems + 1) / 2;

            // Preload scales to reduce redundant decoding
            for (int kb = 0; kb < tile_bytes; kb += 4) {
                // Load scale for this block (scales repeat every 16 elements = 8 bytes)
                int scale_idx0 = kb / 8;
                int scale_idx1 = (kb + 1) / 8;
                int scale_idx2 = (kb + 2) / 8;
                int scale_idx3 = (kb + 3) / 8;

                float sfa0 = decode_fp8(matrix_scale_smem[buffer_idx][row * (K_TILE / 16) + scale_idx0]);
                float sfb0 = decode_fp8(vector_scale_smem[buffer_idx][scale_idx0]);
                float sfa1 = (scale_idx1 != scale_idx0) ? decode_fp8(matrix_scale_smem[buffer_idx][row * (K_TILE / 16) + scale_idx1]) : sfa0;
                float sfb1 = (scale_idx1 != scale_idx0) ? decode_fp8(vector_scale_smem[buffer_idx][scale_idx1]) : sfb0;
                float sfa2 = (scale_idx2 != scale_idx1) ? decode_fp8(matrix_scale_smem[buffer_idx][row * (K_TILE / 16) + scale_idx2]) : sfa1;
                float sfb2 = (scale_idx2 != scale_idx1) ? decode_fp8(vector_scale_smem[buffer_idx][scale_idx2]) : sfb1;
                float sfa3 = (scale_idx3 != scale_idx2) ? decode_fp8(matrix_scale_smem[buffer_idx][row * (K_TILE / 16) + scale_idx3]) : sfa2;
                float sfb3 = (scale_idx3 != scale_idx2) ? decode_fp8(vector_scale_smem[buffer_idx][scale_idx3]) : sfb2;

                // Process 4 bytes in parallel to 4 accumulators
                if (kb < tile_bytes) {
                    uint8_t matrix_byte = matrix_smem[buffer_idx][row * (K_TILE / 2) + kb];
                    uint8_t vector_byte = vector_smem[buffer_idx][kb];
                    float a_lo = decode_fp4(matrix_byte & 0xF);
                    float a_hi = decode_fp4(matrix_byte >> 4);
                    float b_lo = decode_fp4(vector_byte & 0xF);
                    float b_hi = decode_fp4(vector_byte >> 4);
                    acc0 += a_lo * sfa0 * b_lo * sfb0;
                    acc0 += a_hi * sfa0 * b_hi * sfb0;
                }

                if (kb + 1 < tile_bytes) {
                    uint8_t matrix_byte = matrix_smem[buffer_idx][row * (K_TILE / 2) + kb + 1];
                    uint8_t vector_byte = vector_smem[buffer_idx][kb + 1];
                    float a_lo = decode_fp4(matrix_byte & 0xF);
                    float a_hi = decode_fp4(matrix_byte >> 4);
                    float b_lo = decode_fp4(vector_byte & 0xF);
                    float b_hi = decode_fp4(vector_byte >> 4);
                    acc1 += a_lo * sfa1 * b_lo * sfb1;
                    acc1 += a_hi * sfa1 * b_hi * sfb1;
                }

                if (kb + 2 < tile_bytes) {
                    uint8_t matrix_byte = matrix_smem[buffer_idx][row * (K_TILE / 2) + kb + 2];
                    uint8_t vector_byte = vector_smem[buffer_idx][kb + 2];
                    float a_lo = decode_fp4(matrix_byte & 0xF);
                    float a_hi = decode_fp4(matrix_byte >> 4);
                    float b_lo = decode_fp4(vector_byte & 0xF);
                    float b_hi = decode_fp4(vector_byte >> 4);
                    acc2 += a_lo * sfa2 * b_lo * sfb2;
                    acc2 += a_hi * sfa2 * b_hi * sfb2;
                }

                if (kb + 3 < tile_bytes) {
                    uint8_t matrix_byte = matrix_smem[buffer_idx][row * (K_TILE / 2) + kb + 3];
                    uint8_t vector_byte = vector_smem[buffer_idx][kb + 3];
                    float a_lo = decode_fp4(matrix_byte & 0xF);
                    float a_hi = decode_fp4(matrix_byte >> 4);
                    float b_lo = decode_fp4(vector_byte & 0xF);
                    float b_hi = decode_fp4(vector_byte >> 4);
                    acc3 += a_lo * sfa3 * b_lo * sfb3;
                    acc3 += a_hi * sfa3 * b_hi * sfb3;
                }
            }
        }
    };

    // SOFTWARE PIPELINING: Prologue - issue loads for first tile
    if (num_k_tiles > 0) {
        issue_loads(0, 0);
        cp_async_commit_group();
    }

    // Main loop: overlap compute(k) with load(k+1)
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int curr_buffer = k_tile & 1;
        int next_buffer = (k_tile + 1) & 1;

        // Issue loads for NEXT tile (if exists)
        if (k_tile + 1 < num_k_tiles) {
            issue_loads(k_tile + 1, next_buffer);
            cp_async_commit_group();
        }

        // Wait for CURRENT tile's loads to complete
        // cp.async.wait_group<1> means: wait until only 1 most recent group is pending
        // This allows overlap: we wait for k_tile while k_tile+1 is loading
        if (k_tile + 1 < num_k_tiles) {
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();  // Last iteration: wait for all
        }
        __syncthreads();

        // Compute on CURRENT tile while NEXT tile loads in background
        int tile_elems = min(K_TILE, K - k_tile * K_TILE);
        compute_tile(curr_buffer, tile_elems);

        __syncthreads();
    }

    // Store result to output
    if (tid < rows_this_tile) {
        int m = m_block * ROWS_PER_CTA + tid;
        size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
        float acc = acc0 + acc1 + acc2 + acc3;
        c[c_idx] = __float2half(acc);
    }
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    dim3 grid((M + 128 - 1) / 128, L);  // 128 = ROWS_PER_CTA
    dim3 block(256);  // THREADS_PER_CTA = 256

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

THREADS_PER_CTA = 256

ROWS_PER_CTA = 128

K_TILE = 256

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
