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
#define BLOCK_SIZE 32
#define K_TILE 2560
#define SCALES_PER_TILE (K_TILE / 16)
#define BYTES_PER_TILE (K_TILE / 2)
#define NUM_BUFFERS 2

// ============================================================================
// ======================== TYPE CONVERSION HELPERS ===========================
// ============================================================================
__device__ __forceinline__ float half_raw_to_float(const __half_raw& raw) {
    return __half2float(__ushort_as_half(raw.x));
}

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
    return half_raw_to_float(raw);
}

// ============================================================================
// ================ SCALED DOT PRODUCT FOR 4 PACKED BYTES =====================
// ============================================================================
__device__ __forceinline__ __half2 dot_scaled_4bytes(
    uint32_t a4,
    uint32_t b4,
    __half2 scale_h2
) {
    // Extract bytes using prmt.b32 for 50% reduction in ALU overhead
    uint32_t b_byte0, b_byte1, b_byte2, b_byte3;
    uint32_t a_byte0, a_byte1, a_byte2, a_byte3;

    // Extract 4 bytes from b4 using PTX prmt (replicate byte to all positions, select byte 0-3)
    asm("prmt.b32 %0, %1, 0, 0x4440;" : "=r"(b_byte0) : "r"(b4));
    asm("prmt.b32 %0, %1, 0, 0x4441;" : "=r"(b_byte1) : "r"(b4));
    asm("prmt.b32 %0, %1, 0, 0x4442;" : "=r"(b_byte2) : "r"(b4));
    asm("prmt.b32 %0, %1, 0, 0x4443;" : "=r"(b_byte3) : "r"(b4));

    // Extract 4 bytes from a4
    asm("prmt.b32 %0, %1, 0, 0x4440;" : "=r"(a_byte0) : "r"(a4));
    asm("prmt.b32 %0, %1, 0, 0x4441;" : "=r"(a_byte1) : "r"(a4));
    asm("prmt.b32 %0, %1, 0, 0x4442;" : "=r"(a_byte2) : "r"(a4));
    asm("prmt.b32 %0, %1, 0, 0x4443;" : "=r"(a_byte3) : "r"(a4));

    __half2 b0_scaled = __hmul2(decode_fp4x2(b_byte0), scale_h2);
    __half2 b1_scaled = __hmul2(decode_fp4x2(b_byte1), scale_h2);
    __half2 b2_scaled = __hmul2(decode_fp4x2(b_byte2), scale_h2);
    __half2 b3_scaled = __hmul2(decode_fp4x2(b_byte3), scale_h2);

    __half2 acc = __hmul2(decode_fp4x2(a_byte0), b0_scaled);
    acc = __hfma2(decode_fp4x2(a_byte1), b1_scaled, acc);
    acc = __hfma2(decode_fp4x2(a_byte2), b2_scaled, acc);
    acc = __hfma2(decode_fp4x2(a_byte3), b3_scaled, acc);

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

        int byte_base = sf * 8;

        uint32_t a4_0 = *reinterpret_cast<const uint32_t*>(&sh_a[byte_base]);
        uint32_t b4_0 = *reinterpret_cast<const uint32_t*>(&sh_b[byte_base]);
        uint32_t a4_1 = *reinterpret_cast<const uint32_t*>(&sh_a[byte_base + 4]);
        uint32_t b4_1 = *reinterpret_cast<const uint32_t*>(&sh_b[byte_base + 4]);

        __half2 acc_h2 = dot_scaled_4bytes(a4_0, b4_0, scale_h2);
        __half2 acc_h2_1 = dot_scaled_4bytes(a4_1, b4_1, scale_h2);
        acc_h2 = __hadd2(acc_h2, acc_h2_1);

        float2 f = __half22float2(acc_h2);
        acc += f.x + f.y;
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

        int byte_base = sf * 8;

        uint32_t a4_0 = __ldg(reinterpret_cast<const uint32_t*>(&row_a[byte_base]));
        uint32_t b4_0 = __ldg(reinterpret_cast<const uint32_t*>(&batch_b[byte_base]));
        uint32_t a4_1 = __ldg(reinterpret_cast<const uint32_t*>(&row_a[byte_base + 4]));
        uint32_t b4_1 = __ldg(reinterpret_cast<const uint32_t*>(&batch_b[byte_base + 4]));

        __half2 acc_h2 = dot_scaled_4bytes(a4_0, b4_0, scale_h2);
        __half2 acc_h2_1 = dot_scaled_4bytes(a4_1, b4_1, scale_h2);
        acc_h2 = __hadd2(acc_h2, acc_h2_1);

        float2 f = __half22float2(acc_h2);
        acc += f.x + f.y;
    }

    return acc;
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
    int m = blockIdx.x;
    int l = blockIdx.y;
    int tid = threadIdx.x;

    if (m >= M || l >= L) return;

// ============================================================================
// ===================== PER-CTA BASE POINTER SETUP ===========================
// ============================================================================
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

    const uint8_t* row_a = batch_a + static_cast<size_t>(m) * K_half;
    const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m) * K_sf;

// ============================================================================
// ===================== SHARED MEMORY ALLOCATION =============================
// ============================================================================
    __shared__ float smem_acc[BLOCK_SIZE];
    __shared__ uint8_t sh_a[NUM_BUFFERS][BYTES_PER_TILE];
    __shared__ uint8_t sh_b[NUM_BUFFERS][BYTES_PER_TILE];
    __shared__ uint8_t sh_sfa[NUM_BUFFERS][SCALES_PER_TILE];
    __shared__ uint8_t sh_sfb[NUM_BUFFERS][SCALES_PER_TILE];

    float acc = 0.0f;
    int tile_count = K / K_TILE;
    int remainder_start = tile_count * K_TILE;

// ============================================================================
// ===================== ASYNC COPY MACROS ====================================
// ============================================================================
#define ASYNC_COPY_16(dst, src) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src))

#define ASYNC_COPY_4(dst, src) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(src))

    auto issue_tile_async = [&](int b_idx, int tile_idx) {
        int base_k = tile_idx * K_TILE;
        int base_byte = base_k / 2;
        int base_sf = base_k / 16;
        uint32_t sh_a_base = __cvta_generic_to_shared(&sh_a[b_idx][0]);
        uint32_t sh_b_base = __cvta_generic_to_shared(&sh_b[b_idx][0]);
        uint32_t sh_sfa_base = __cvta_generic_to_shared(&sh_sfa[b_idx][0]);
        uint32_t sh_sfb_base = __cvta_generic_to_shared(&sh_sfb[b_idx][0]);

#pragma unroll 2
        for (int i = tid * 16; i < BYTES_PER_TILE; i += BLOCK_SIZE * 16) {
            ASYNC_COPY_16(sh_a_base + i, row_a + base_byte + i);
            ASYNC_COPY_16(sh_b_base + i, batch_b + base_byte + i);
        }
#pragma unroll
        for (int i = tid * 4; i < SCALES_PER_TILE; i += BLOCK_SIZE * 4) {
            ASYNC_COPY_4(sh_sfa_base + i, row_sfa + base_sf + i);
            ASYNC_COPY_4(sh_sfb_base + i, batch_sfb + base_sf + i);
        }
        asm volatile("cp.async.commit_group;");
    };

// ============================================================================
// ===================== DOUBLE-BUFFERED MAIN LOOP ============================
// ============================================================================
    if (tile_count > 0) {
        int buf = 0;
        issue_tile_async(0, 0);
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        for (int tile = 0; tile < tile_count; ++tile) {
            if (tile + 1 < tile_count) {
                issue_tile_async(buf ^ 1, tile + 1);
            }

            acc += compute_tile(
                sh_a[buf],
                sh_b[buf],
                sh_sfa[buf],
                sh_sfb[buf],
                tid
            );

            if (tile + 1 < tile_count) {
                asm volatile("cp.async.wait_group 0;");
                __syncthreads();
                buf ^= 1;
            }
        }
    }

// ============================================================================
// ===================== REMAINDER PROCESSING =================================
// ============================================================================
    int remainder_sf_start = remainder_start / 16;
    if (remainder_sf_start < K_sf) {
        acc += compute_remainder(
            row_a,
            batch_b,
            row_sfa,
            batch_sfb,
            remainder_sf_start,
            K_sf,
            tid
        );
    }

#undef ASYNC_COPY_16
#undef ASYNC_COPY_4

// ============================================================================
// ===================== WARP-LEVEL REDUCTION =================================
// ============================================================================
    float warp_sum = acc;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }

// ============================================================================
// ===================== BLOCK-LEVEL REDUCTION ================================
// ============================================================================
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) smem_acc[warp_id] = warp_sum;

    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < (blockDim.x >> 5)) ? smem_acc[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }

// ============================================================================
// ===================== FINAL OUTPUT WRITE ===================================
// ============================================================================
        if (lane == 0) {
            size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
            c[c_idx] = __float2half(block_sum);
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

    dim3 grid(M, L);
    dim3 block(BLOCK_SIZE);

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    gemv_nvfp4_kernel<<<grid, block>>>(
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
