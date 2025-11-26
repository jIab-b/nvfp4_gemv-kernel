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
#define M_TILE 16
#define BLOCK_SIZE 32
#define K_TILE_PER_CTA 2048
#define K_TILE 1024 // Sub-tile for double buffering within CTA's K chunk
#define SCALES_PER_TILE (K_TILE / 16)
#define BYTES_PER_TILE (K_TILE / 2)
#define NUM_BUFFERS 2
#define WARP_SIZE 32

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
    uint32_t a4,
    uint32_t b4,
    __half2 scale_h2
) {
    // Extract bytes using bfe.u32 for clearer semantics and same performance as prmt
    uint32_t b_byte0, b_byte1, b_byte2, b_byte3;
    uint32_t a_byte0, a_byte1, a_byte2, a_byte3;

    // Extract 4 bytes from b4 using bit field extract
    asm("bfe.u32 %0, %1, 0, 8;"  : "=r"(b_byte0) : "r"(b4));
    asm("bfe.u32 %0, %1, 8, 8;"  : "=r"(b_byte1) : "r"(b4));
    asm("bfe.u32 %0, %1, 16, 8;" : "=r"(b_byte2) : "r"(b4));
    asm("bfe.u32 %0, %1, 24, 8;" : "=r"(b_byte3) : "r"(b4));

    // Extract 4 bytes from a4
    asm("bfe.u32 %0, %1, 0, 8;"  : "=r"(a_byte0) : "r"(a4));
    asm("bfe.u32 %0, %1, 8, 8;"  : "=r"(a_byte1) : "r"(a4));
    asm("bfe.u32 %0, %1, 16, 8;" : "=r"(a_byte2) : "r"(a4));
    asm("bfe.u32 %0, %1, 24, 8;" : "=r"(a_byte3) : "r"(a4));

    // Inline b_scaled computation to minimize register live ranges
    __half2 acc = __hmul2(decode_fp4x2(a_byte0), __hmul2(decode_fp4x2(b_byte0), scale_h2));
    acc = __hfma2(decode_fp4x2(a_byte1), __hmul2(decode_fp4x2(b_byte1), scale_h2), acc);
    acc = __hfma2(decode_fp4x2(a_byte2), __hmul2(decode_fp4x2(b_byte2), scale_h2), acc);
    acc = __hfma2(decode_fp4x2(a_byte3), __hmul2(decode_fp4x2(b_byte3), scale_h2), acc);

    return acc;
}

// ============================================================================
// ==================== TILE COMPUTE DEVICE FUNCTION ==========================
// ============================================================================
__device__ __forceinline__ float compute_tile(
    const uint8_t* sh_a,
    const uint8_t* sh_b,
    const uint8_t* sh_sfa,
    const uint8_t* sh_sfb,
    int tid
) {
    float acc = 0.0f;

#pragma unroll 8
    for (int sf = tid; sf < SCALES_PER_TILE; sf += BLOCK_SIZE) {
        float scale = decode_fp8(static_cast<int8_t>(sh_sfa[sf])) *
                      decode_fp8(static_cast<int8_t>(sh_sfb[sf]));
        __half2 scale_h2 = __half2half2(__float2half(scale));

        int byte_base = sf << 3;  // sf * 8 using bit shift

        uint32_t a4_0 = *reinterpret_cast<const uint32_t*>(&sh_a[byte_base]);
        uint32_t b4_0 = *reinterpret_cast<const uint32_t*>(&sh_b[byte_base]);
        uint32_t a4_1 = *reinterpret_cast<const uint32_t*>(&sh_a[byte_base + 4]);
        uint32_t b4_1 = *reinterpret_cast<const uint32_t*>(&sh_b[byte_base + 4]);

        __half2 acc_h2_0 = dot_scaled_4bytes(a4_0, b4_0, scale_h2);
        __half2 acc_h2_1 = dot_scaled_4bytes(a4_1, b4_1, scale_h2);

        // Combine and accumulate in one step
        float2 f0 = __half22float2(acc_h2_0);
        float2 f1 = __half22float2(acc_h2_1);
        acc += f0.x + f0.y + f1.x + f1.y;
    }

    return acc;
}

// ============================================================================
// ==================== REMAINDER COMPUTE DEVICE FUNCTION =====================
// ============================================================================
__device__ __forceinline__ float compute_remainder(
    const uint8_t* row_a,
    const uint8_t* batch_b,
    const uint8_t* row_sfa,
    const uint8_t* batch_sfb,
    int remainder_sf_start,
    int K_sf,
    int tid
) {
    float acc = 0.0f;

#pragma unroll 4
    for (int sf = remainder_sf_start + tid; sf < K_sf; sf += BLOCK_SIZE) {
        float scale = decode_fp8(static_cast<int8_t>(__ldg(&row_sfa[sf]))) *
                      decode_fp8(static_cast<int8_t>(__ldg(&batch_sfb[sf])));
        __half2 scale_h2 = __half2half2(__float2half(scale));

        int byte_base = sf << 3;  // sf * 8 using bit shift

        uint32_t a4_0 = __ldg(reinterpret_cast<const uint32_t*>(&row_a[byte_base]));
        uint32_t b4_0 = __ldg(reinterpret_cast<const uint32_t*>(&batch_b[byte_base]));
        uint32_t a4_1 = __ldg(reinterpret_cast<const uint32_t*>(&row_a[byte_base + 4]));
        uint32_t b4_1 = __ldg(reinterpret_cast<const uint32_t*>(&batch_b[byte_base + 4]));

        __half2 acc_h2_0 = dot_scaled_4bytes(a4_0, b4_0, scale_h2);
        __half2 acc_h2_1 = dot_scaled_4bytes(a4_1, b4_1, scale_h2);

        // Combine and accumulate in one step
        float2 f0 = __half22float2(acc_h2_0);
        float2 f1 = __half22float2(acc_h2_1);
        acc += f0.x + f0.y + f1.x + f1.y;
    }

    return acc;
}

// ============================================================================
// ============ REMAINDER COMPUTE FROM SHARED MEMORY ==========================
// ============================================================================
__device__ __forceinline__ float compute_remainder_smem(
    const uint8_t* sh_a,
    const uint8_t* sh_b,
    const uint8_t* sh_sfa,
    const uint8_t* sh_sfb,
    int remainder_scales,
    int tid
) {
    float acc = 0.0f;

#pragma unroll 4
    for (int sf = tid; sf < remainder_scales; sf += BLOCK_SIZE) {
        float scale = decode_fp8(static_cast<int8_t>(sh_sfa[sf])) *
                      decode_fp8(static_cast<int8_t>(sh_sfb[sf]));
        __half2 scale_h2 = __half2half2(__float2half(scale));

        int byte_base = sf << 3;  // sf * 8 using bit shift

        uint32_t a4_0 = *reinterpret_cast<const uint32_t*>(&sh_a[byte_base]);
        uint32_t b4_0 = *reinterpret_cast<const uint32_t*>(&sh_b[byte_base]);
        uint32_t a4_1 = *reinterpret_cast<const uint32_t*>(&sh_a[byte_base + 4]);
        uint32_t b4_1 = *reinterpret_cast<const uint32_t*>(&sh_b[byte_base + 4]);

        __half2 acc_h2_0 = dot_scaled_4bytes(a4_0, b4_0, scale_h2);
        __half2 acc_h2_1 = dot_scaled_4bytes(a4_1, b4_1, scale_h2);

        // Combine and accumulate in one step
        float2 f0 = __half22float2(acc_h2_0);
        float2 f1 = __half22float2(acc_h2_1);
        acc += f0.x + f0.y + f1.x + f1.y;
    }

    return acc;
}

// ============================================================================
// ==================== M_TILE COMPUTE DEVICE FUNCTION ========================
// ============================================================================
__device__ __forceinline__ void compute_m_tile(
    const uint8_t* sh_a,
    const uint8_t* sh_b,
    const uint8_t* sh_sfa,
    const uint8_t* sh_sfb,
    float* acc_array,
    int m_count,
    int tid
) {
#pragma unroll 4
    for (int sf = tid; sf < SCALES_PER_TILE; sf += BLOCK_SIZE) {
        float sfb_val = decode_fp8(static_cast<int8_t>(sh_sfb[sf]));
        __half2 scale_b_h2 = __half2half2(__float2half(sfb_val));

        int byte_base = sf << 3;  // sf * 8
        uint32_t b4_0 = *reinterpret_cast<const uint32_t*>(&sh_b[byte_base]);
        uint32_t b4_1 = *reinterpret_cast<const uint32_t*>(&sh_b[byte_base + 4]);

        // Process each m in M_TILE
#pragma unroll
        for (int m_local = 0; m_local < m_count; ++m_local) {
            int a_offset = m_local * BYTES_PER_TILE;
            int sfa_offset = m_local * SCALES_PER_TILE;

            float sfa_val = decode_fp8(static_cast<int8_t>(sh_sfa[sfa_offset + sf]));
            __half2 scale_h2 = __hmul2(scale_b_h2, __half2half2(__float2half(sfa_val)));

            uint32_t a4_0 = *reinterpret_cast<const uint32_t*>(&sh_a[a_offset + byte_base]);
            uint32_t a4_1 = *reinterpret_cast<const uint32_t*>(&sh_a[a_offset + byte_base + 4]);

            __half2 acc_h2_0 = dot_scaled_4bytes(a4_0, b4_0, scale_h2);
            __half2 acc_h2_1 = dot_scaled_4bytes(a4_1, b4_1, scale_h2);

            float2 f0 = __half22float2(acc_h2_0);
            float2 f1 = __half22float2(acc_h2_1);
            acc_array[m_local] += f0.x + f0.y + f1.x + f1.y;
        }
    }
}

// ============================================================================
// ========================== PARTIAL GEMV KERNEL =============================
// ============================================================================
__global__ void gemv_nvfp4_partial_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    float* __restrict__ partial,
    int M, int K, int L,
    int N_rows,
    int K_SPLITS
) {
    int m_tile_idx = blockIdx.x;
    int l = blockIdx.y;
    int k_split = blockIdx.z;
    int tid = threadIdx.x;

    int m_start = m_tile_idx * M_TILE;
    int m_count = min(M_TILE, M - m_start);

    if (m_start >= M || l >= L || k_split >= K_SPLITS) return;

// ============================================================================
// ===================== PER-CTA BASE POINTER SETUP ===========================
// ============================================================================
    const int K_sf = K / 16;
    const int K_half = K / 2;

    // Compute K range for this split
    int k_start = k_split * K_TILE_PER_CTA;
    int k_end = min(k_start + K_TILE_PER_CTA, K);
    int k_this_split = k_end - k_start;

    if (k_this_split <= 0) return;

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

    // Offset to this CTA's K chunk
    int k_byte_offset = k_start / 2;
    int k_sf_offset = k_start / 16;

// ============================================================================
// ===================== SHARED MEMORY ALLOCATION =============================
// ============================================================================
    __shared__ uint8_t sh_a[NUM_BUFFERS][M_TILE * BYTES_PER_TILE];
    __shared__ uint8_t sh_b[NUM_BUFFERS][BYTES_PER_TILE];
    __shared__ uint8_t sh_sfa[NUM_BUFFERS][M_TILE * SCALES_PER_TILE];
    __shared__ uint8_t sh_sfb[NUM_BUFFERS][SCALES_PER_TILE];

    // Per-thread accumulators for M_TILE outputs
    float acc[M_TILE];
#pragma unroll
    for (int i = 0; i < M_TILE; ++i) {
        acc[i] = 0.0f;
    }

    int tile_count = k_this_split / K_TILE;
    int remainder_start = tile_count * K_TILE;

// ============================================================================
// ===================== ASYNC COPY MACROS ====================================
// ============================================================================
#define ASYNC_COPY_16(dst, src) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src))

#define ASYNC_COPY_4(dst, src) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(src))

    auto issue_tile_async = [&](int b_idx, int tile_idx_in_split) {
        const int base_k_in_split = tile_idx_in_split * K_TILE;
        const int base_byte_in_split = base_k_in_split >> 1;
        const int base_sf_in_split = base_k_in_split >> 4;

        const uint32_t sh_a_base = __cvta_generic_to_shared(&sh_a[b_idx][0]);
        const uint32_t sh_b_base = __cvta_generic_to_shared(&sh_b[b_idx][0]);
        const uint32_t sh_sfa_base = __cvta_generic_to_shared(&sh_sfa[b_idx][0]);
        const uint32_t sh_sfb_base = __cvta_generic_to_shared(&sh_sfb[b_idx][0]);

        // Copy B vector (shared across M_TILE)
        for (int i = tid * 16; i < BYTES_PER_TILE; i += BLOCK_SIZE * 16) {
            ASYNC_COPY_16(sh_b_base + i, batch_b + k_byte_offset + base_byte_in_split + i);
        }
        for (int i = tid * 4; i < SCALES_PER_TILE; i += BLOCK_SIZE * 4) {
            ASYNC_COPY_4(sh_sfb_base + i, batch_sfb + k_sf_offset + base_sf_in_split + i);
        }

        // Copy M_TILE rows of A
        for (int m_local = 0; m_local < m_count; ++m_local) {
            int m_global = m_start + m_local;
            const uint8_t* row_a = batch_a + static_cast<size_t>(m_global) * K_half;
            const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m_global) * K_sf;

            int a_smem_offset = m_local * BYTES_PER_TILE;
            int sfa_smem_offset = m_local * SCALES_PER_TILE;

            for (int i = tid * 16; i < BYTES_PER_TILE; i += BLOCK_SIZE * 16) {
                ASYNC_COPY_16(sh_a_base + a_smem_offset + i, row_a + k_byte_offset + base_byte_in_split + i);
            }
            for (int i = tid * 4; i < SCALES_PER_TILE; i += BLOCK_SIZE * 4) {
                ASYNC_COPY_4(sh_sfa_base + sfa_smem_offset + i, row_sfa + k_sf_offset + base_sf_in_split + i);
            }
        }
        asm volatile("cp.async.commit_group;");
    };

    auto issue_remainder_async = [&](int b_idx, int rem_k_start_in_split, int k_this_split) {
        const int base_byte_in_split = rem_k_start_in_split >> 1;
        const int base_sf_in_split = rem_k_start_in_split >> 4;

        const uint32_t sh_a_base = __cvta_generic_to_shared(&sh_a[b_idx][0]);
        const uint32_t sh_b_base = __cvta_generic_to_shared(&sh_b[b_idx][0]);
        const uint32_t sh_sfa_base = __cvta_generic_to_shared(&sh_sfa[b_idx][0]);
        const uint32_t sh_sfb_base = __cvta_generic_to_shared(&sh_sfb[b_idx][0]);

        int remainder_bytes = (k_this_split - rem_k_start_in_split) >> 1;
        int remainder_scales = (k_this_split - rem_k_start_in_split) >> 4;

        // Copy B remainder
        for (int i = tid * 16; i < remainder_bytes; i += BLOCK_SIZE * 16) {
            ASYNC_COPY_16(sh_b_base + i, batch_b + k_byte_offset + base_byte_in_split + i);
        }
        for (int i = tid * 4; i < remainder_scales; i += BLOCK_SIZE * 4) {
            ASYNC_COPY_4(sh_sfb_base + i, batch_sfb + k_sf_offset + base_sf_in_split + i);
        }

        // Copy A remainder for M_TILE rows
        for (int m_local = 0; m_local < m_count; ++m_local) {
            int m_global = m_start + m_local;
            const uint8_t* row_a = batch_a + static_cast<size_t>(m_global) * K_half;
            const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m_global) * K_sf;

            int a_smem_offset = m_local * BYTES_PER_TILE;
            int sfa_smem_offset = m_local * SCALES_PER_TILE;

            for (int i = tid * 16; i < remainder_bytes; i += BLOCK_SIZE * 16) {
                ASYNC_COPY_16(sh_a_base + a_smem_offset + i, row_a + k_byte_offset + base_byte_in_split + i);
            }
            for (int i = tid * 4; i < remainder_scales; i += BLOCK_SIZE * 4) {
                ASYNC_COPY_4(sh_sfa_base + sfa_smem_offset + i, row_sfa + k_sf_offset + base_sf_in_split + i);
            }
        }
        asm volatile("cp.async.commit_group;");
    };

// ============================================================================
// ===================== DOUBLE-BUFFERED MAIN LOOP ============================
// ============================================================================
    bool has_remainder = (remainder_start < k_this_split);
    int buf = 0;

    if (tile_count > 0) {
        issue_tile_async(0, 0);
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        for (int tile = 0; tile < tile_count; ++tile) {
            if (tile + 1 < tile_count) {
                issue_tile_async(buf ^ 1, tile + 1);
            } else if (has_remainder) {
                issue_remainder_async(buf ^ 1, remainder_start, k_this_split);
            }

            compute_m_tile(
                sh_a[buf],
                sh_b[buf],
                sh_sfa[buf],
                sh_sfb[buf],
                acc,
                m_count,
                tid
            );

            if (tile + 1 < tile_count || has_remainder) {
                asm volatile("cp.async.wait_group 0;");
                __syncthreads();
                buf ^= 1;
            }
        }
    }

// ============================================================================
// ===================== REMAINDER PROCESSING =================================
// ============================================================================
    if (has_remainder && tile_count > 0) {
        // Remainder was prefetched - use from shared memory
        // For simplicity, compute remainder tile by tile for each m
        int remainder_scales = (k_this_split - remainder_start) >> 4;
#pragma unroll
        for (int m_local = 0; m_local < m_count; ++m_local) {
            int a_offset = m_local * BYTES_PER_TILE;
            int sfa_offset = m_local * SCALES_PER_TILE;
            acc[m_local] += compute_remainder_smem(
                &sh_a[buf][a_offset],
                sh_b[buf],
                &sh_sfa[buf][sfa_offset],
                sh_sfb[buf],
                remainder_scales,
                tid
            );
        }
    }

#undef ASYNC_COPY_16
#undef ASYNC_COPY_4

// ============================================================================
// ===================== BLOCK-LEVEL REDUCTION FOR M_TILE ====================
// ============================================================================
    __shared__ float smem_reduce[M_TILE][WARP_SIZE];

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = BLOCK_SIZE >> 5;

    // Reduce each of M_TILE accumulators across the block
#pragma unroll
    for (int m_local = 0; m_local < m_count; ++m_local) {
        float warp_sum = acc[m_local];

        // Warp reduction
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 16);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 8);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 4);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 2);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 1);

        if (lane == 0) {
            smem_reduce[m_local][warp_id] = warp_sum;
        }
    }

    __syncthreads();

    // Final reduction across warps and write output
    if (warp_id == 0) {
#pragma unroll
        for (int m_local = 0; m_local < m_count; ++m_local) {
            float block_sum = (lane < num_warps) ? smem_reduce[m_local][lane] : 0.0f;

            block_sum += __shfl_down_sync(0xffffffff, block_sum, 16);
            block_sum += __shfl_down_sync(0xffffffff, block_sum, 8);
            block_sum += __shfl_down_sync(0xffffffff, block_sum, 4);
            block_sum += __shfl_down_sync(0xffffffff, block_sum, 2);
            block_sum += __shfl_down_sync(0xffffffff, block_sum, 1);

            if (lane == 0) {
                int m_global = m_start + m_local;
                size_t partial_idx = static_cast<size_t>(m_global) +
                                    static_cast<size_t>(l) * M +
                                    static_cast<size_t>(k_split) * M * L;
                partial[partial_idx] = block_sum;
            }
        }
    }
}

// ============================================================================
// ========================== K REDUCTION KERNEL ==============================
// ============================================================================
__global__ void reduce_k_kernel(
    const float* __restrict__ partial,
    half* __restrict__ c,
    int M, int L, int K_SPLITS
) {
    int m = blockIdx.x;
    int l = blockIdx.y;
    int tid = threadIdx.x;

    if (m >= M || l >= L) return;

    size_t base_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;

    float sum = 0.0f;
    for (int k_split = tid; k_split < K_SPLITS; k_split += BLOCK_SIZE) {
        size_t partial_idx = base_idx + static_cast<size_t>(k_split) * M * L;
        sum += partial[partial_idx];
    }

    // Warp reduction
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    __shared__ float smem[WARP_SIZE];
    int warp_id = tid >> 5;
    int lane = tid & 31;

    if (lane == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < (BLOCK_SIZE >> 5)) ? smem[lane] : 0.0f;
        block_sum += __shfl_down_sync(0xffffffff, block_sum, 16);
        block_sum += __shfl_down_sync(0xffffffff, block_sum, 8);
        block_sum += __shfl_down_sync(0xffffffff, block_sum, 4);
        block_sum += __shfl_down_sync(0xffffffff, block_sum, 2);
        block_sum += __shfl_down_sync(0xffffffff, block_sum, 1);

        if (lane == 0) {
            c[base_idx] = __float2half(block_sum);
        }
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

    int K_SPLITS = (K + K_TILE_PER_CTA - 1) / K_TILE_PER_CTA;

    // Allocate partial results buffer
    auto partial = torch::empty({K_SPLITS, L, M}, torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));

    // Launch partial GEMV kernel
    dim3 grid_partial((M + M_TILE - 1) / M_TILE, L, K_SPLITS);
    dim3 block(BLOCK_SIZE);

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* partial_ptr = partial.data_ptr<float>();
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    gemv_nvfp4_partial_kernel<<<grid_partial, block>>>(
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        partial_ptr,
        M, K, L,
        N_rows,
        K_SPLITS
    );

    // Launch reduction kernel
    dim3 grid_reduce(M, L);
    reduce_k_kernel<<<grid_reduce, block>>>(
        partial_ptr,
        c_ptr,
        M, L, K_SPLITS
    );

    return c;
}

"""

module = load_inline(
    name='batched_scaled_gemv',
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
    return module.batched_scaled_gemv_cuda(
        a,
        b,
        sfa_ref,
        sfb_ref,
        c
    )
