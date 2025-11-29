import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


cpp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// ============================================================================
// ======================== CONFIGURATION CONSTANTS ===========================
// ============================================================================
#define BLOCK_SIZE 128               // 4 warps per block
#define ROWS_PER_BLOCK 4             // one warp per row
#define K_TILE_BYTES 4096            // bytes of B per tile (4096 bytes = 512 sf)
#define SCALES_PER_TILE (K_TILE_BYTES / 8)
#define NUM_BUFFERS 2

// ============================================================================
// ======================== TYPE CONVERSION HELPERS ===========================
// ============================================================================
__device__ __forceinline__ __half2 decode_fp4x2(uint8_t byte) {
    __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(
        static_cast<__nv_fp4x2_storage_t>(byte),
        __NV_E2M1
    );
    return *reinterpret_cast<__half2*>(&raw);
}

__device__ __forceinline__ float decode_fp8(int8_t byte) {
    __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(byte);
    __half_raw raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
    return __half2float(__ushort_as_half(raw.x));
}

// ============================================================================
// ================ SCALED DOT PRODUCT FOR 4 PACKED BYTES =====================
// ============================================================================
__device__ __forceinline__ __half2 dot_scaled_4bytes(
    uchar4 a4,
    uchar4 b4,
    __half2 scale_h2
) {
    __half2 acc0 = __hmul2(decode_fp4x2(a4.x), __hmul2(decode_fp4x2(b4.x), scale_h2));
    __half2 acc1 = __hmul2(decode_fp4x2(a4.y), __hmul2(decode_fp4x2(b4.y), scale_h2));
    acc0 = __hfma2(decode_fp4x2(a4.z), __hmul2(decode_fp4x2(b4.z), scale_h2), acc0);
    acc1 = __hfma2(decode_fp4x2(a4.w), __hmul2(decode_fp4x2(b4.w), scale_h2), acc1);
    return __hadd2(acc0, acc1);
}

// ============================================================================
// ==================== TILE COMPUTE USING SHARED B ===========================
// ============================================================================
__device__ __forceinline__ float compute_tile_shared(
    const uint8_t* row_a_base,
    const uint8_t* sh_b,
    const uint8_t* row_sfa_base,
    const uint8_t* sh_sfb,
    int tile_sf,
    int tid_in_row
) {
    float acc0 = 0.0f, acc1 = 0.0f;
    const int THREADS_PER_ROW = BLOCK_SIZE / ROWS_PER_BLOCK; // 32
    const int STRIDE = THREADS_PER_ROW * 2;

    for (int sf_base = tid_in_row * 2; sf_base < tile_sf; sf_base += STRIDE) {
        // Chain 0
        if (sf_base < tile_sf) {
            float scale0 = decode_fp8(static_cast<int8_t>(__ldg(&row_sfa_base[sf_base]))) *
                           decode_fp8(static_cast<int8_t>(sh_sfb[sf_base]));
            __half2 scale_h2_0 = __halves2half2(__float2half(scale0), __float2half(scale0));
            int byte_base0 = sf_base << 3;
            uchar4 a4_0 = *reinterpret_cast<const uchar4*>(&row_a_base[byte_base0]);
            uchar4 b4_0 = *reinterpret_cast<const uchar4*>(&sh_b[byte_base0]);
            uchar4 a4_1 = *reinterpret_cast<const uchar4*>(&row_a_base[byte_base0 + 4]);
            uchar4 b4_1 = *reinterpret_cast<const uchar4*>(&sh_b[byte_base0 + 4]);

            __half2 res0_0 = dot_scaled_4bytes(a4_0, b4_0, scale_h2_0);
            __half2 res0_1 = dot_scaled_4bytes(a4_1, b4_1, scale_h2_0);
            float2 f0_0 = __half22float2(res0_0);
            float2 f0_1 = __half22float2(res0_1);
            acc0 += f0_0.x + f0_0.y + f0_1.x + f0_1.y;
        }

        // Chain 1
        int sf_next = sf_base + 1;
        if (sf_next < tile_sf) {
            float scale1 = decode_fp8(static_cast<int8_t>(__ldg(&row_sfa_base[sf_next]))) *
                           decode_fp8(static_cast<int8_t>(sh_sfb[sf_next]));
            __half2 scale_h2_1 = __halves2half2(__float2half(scale1), __float2half(scale1));
            int byte_base1 = sf_next << 3;
            uchar4 a4_2 = *reinterpret_cast<const uchar4*>(&row_a_base[byte_base1]);
            uchar4 b4_2 = *reinterpret_cast<const uchar4*>(&sh_b[byte_base1]);
            uchar4 a4_3 = *reinterpret_cast<const uchar4*>(&row_a_base[byte_base1 + 4]);
            uchar4 b4_3 = *reinterpret_cast<const uchar4*>(&sh_b[byte_base1 + 4]);

            __half2 res1_0 = dot_scaled_4bytes(a4_2, b4_2, scale_h2_1);
            __half2 res1_1 = dot_scaled_4bytes(a4_3, b4_3, scale_h2_1);
            float2 f1_0 = __half22float2(res1_0);
            float2 f1_1 = __half22float2(res1_1);
            acc1 += f1_0.x + f1_0.y + f1_1.x + f1_1.y;
        }
    }
    return acc0 + acc1;
}

// ============================================================================
// ========================== MAIN KERNEL FUNCTION ============================
// ============================================================================
__global__ void gemv_nvfp4_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L,
    int N_rows
) {
    int m_base = blockIdx.x * ROWS_PER_BLOCK;
    int l = blockIdx.y;
    int tid = threadIdx.x;
    int row_in_block = tid / (BLOCK_SIZE / ROWS_PER_BLOCK);
    int tid_in_row = tid % (BLOCK_SIZE / ROWS_PER_BLOCK);

    if (m_base >= M || l >= L) return;

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

    int m = m_base + row_in_block;
    if (m >= M) return;

    const uint8_t* row_a = batch_a + static_cast<size_t>(m) * K_half;
    const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m) * K_sf;

    // Shared buffers: B and sfb only
    __shared__ uint8_t sh_b[NUM_BUFFERS][K_TILE_BYTES];
    __shared__ uint8_t sh_sfb[NUM_BUFFERS][SCALES_PER_TILE];

    float acc = 0.0f;
    int tile_count = K_half / K_TILE_BYTES;
    int remainder_bytes = K_half - tile_count * K_TILE_BYTES;
    int remainder_sf = K_sf - tile_count * SCALES_PER_TILE;

// ====================================================================
// Async copy helpers (vectorized 16B)
// ====================================================================
#define ASYNC_COPY_16(dst, src) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src))

    auto issue_tile = [&](int buf_idx, int tile_idx) {
        int byte_base = tile_idx * K_TILE_BYTES;
        int sf_base = tile_idx * SCALES_PER_TILE;

        // Copy B
        for (int i = tid * 16; i < K_TILE_BYTES; i += BLOCK_SIZE * 16) {
            const void* gptr = batch_b + byte_base + i;
            uint32_t shptr = __cvta_generic_to_shared(&sh_b[buf_idx][i]);
            ASYNC_COPY_16(shptr, gptr);
        }
        // Copy sfb
        for (int i = tid * 16; i < SCALES_PER_TILE; i += BLOCK_SIZE * 16) {
            const void* gptr = batch_sfb + sf_base + i;
            uint32_t shptr = __cvta_generic_to_shared(&sh_sfb[buf_idx][i]);
            ASYNC_COPY_16(shptr, gptr);
        }
        asm volatile("cp.async.commit_group;");
    };

    auto issue_remainder = [&](int buf_idx) {
        int byte_base = tile_count * K_TILE_BYTES;
        int sf_base = tile_count * SCALES_PER_TILE;

        int rem_bytes = remainder_bytes;
        int rem_scales = remainder_sf;
        // Copy remaining B
        for (int i = tid * 16; i < rem_bytes; i += BLOCK_SIZE * 16) {
            const void* gptr = batch_b + byte_base + i;
            uint32_t shptr = __cvta_generic_to_shared(&sh_b[buf_idx][i]);
            ASYNC_COPY_16(shptr, gptr);
        }
        // Copy remaining sfb
        for (int i = tid * 16; i < rem_scales; i += BLOCK_SIZE * 16) {
            const void* gptr = batch_sfb + sf_base + i;
            uint32_t shptr = __cvta_generic_to_shared(&sh_sfb[buf_idx][i]);
            ASYNC_COPY_16(shptr, gptr);
        }
        asm volatile("cp.async.commit_group;");
    };

    int buf = 0;

    if (tile_count > 0) {
        issue_tile(0, 0);
        asm volatile("cp.async.wait_group 0;" : : : "memory");
        __syncthreads();

        for (int tile = 0; tile < tile_count; ++tile) {
            if (tile + 1 < tile_count) {
                issue_tile(buf ^ 1, tile + 1);
            } else if (remainder_sf > 0) {
                issue_remainder(buf ^ 1);
            }

            // Compute this tile
            acc += compute_tile_shared(
                row_a + tile * K_TILE_BYTES,
                sh_b[buf],
                row_sfa + tile * SCALES_PER_TILE,
                sh_sfb[buf],
                SCALES_PER_TILE,
                tid_in_row
            );

            if (tile + 1 < tile_count || remainder_sf > 0) {
                asm volatile("cp.async.wait_group 0;" : : : "memory");
                __syncthreads();
                buf ^= 1;
            }
        }
    }

    // Remainder
    if (remainder_sf > 0) {
        if (tile_count > 0) {
            // Remainder already staged
            acc += compute_tile_shared(
                row_a + tile_count * K_TILE_BYTES,
                sh_b[buf],
                row_sfa + tile_count * SCALES_PER_TILE,
                sh_sfb[buf],
                remainder_sf,
                tid_in_row
            );
        } else {
            // No tiles staged: fall back to direct global loads for remainder
            acc += compute_tile_shared(
                row_a,
                batch_b,
                row_sfa,
                batch_sfb,
                K_sf,
                tid_in_row
            );
        }
    }

#undef ASYNC_COPY_16

    // Reduction within each row (one warp per row)
    const int lane = tid_in_row;
    float row_sum = acc;
    row_sum += __shfl_down_sync(0xffffffff, row_sum, 16);
    row_sum += __shfl_down_sync(0xffffffff, row_sum, 8);
    row_sum += __shfl_down_sync(0xffffffff, row_sum, 4);
    row_sum += __shfl_down_sync(0xffffffff, row_sum, 2);
    row_sum += __shfl_down_sync(0xffffffff, row_sum, 1);

    if (lane == 0) {
        size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
        c[c_idx] = __float2half(row_sum);
    }
}

// ============================================================================
// ========================== HOST WRAPPER FUNCTION ===========================
// ============================================================================
torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    int grid_m = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(grid_m, L);
    dim3 block(BLOCK_SIZE);

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    gemv_nvfp4_kernel<<<grid, block>>>(
        a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr,
        M, K, L, N_rows
    );
    return c;
}

"""

module = load_inline(
    name='batched_scaled_gemv_async_v3',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batched_scaled_gemv_cuda'],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-std=c++17',
        '-gencode=arch=compute_100a,code=sm_100a',
        '-maxrregcount=64'
    ],
    with_cuda=True,
    verbose=False
)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    return module.batched_scaled_gemv_cuda(
        a,
        b,
        sfa_ref,
        sfb_ref,
        c
    )
