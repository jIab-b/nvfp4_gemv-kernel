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

// ============================================================================
// ======================== INITIALIZATION HELPERS ===========================
// ============================================================================
// ======================== INITIALIZATION HELPERS ===========================
// ============================================================================
// Tunable CTA size (must be a multiple of 32)
#define BLOCK_SIZE 64
// K elements per tile (must be multiple of 16 for scale alignment)
#define K_TILE 1024
#define SCALES_PER_TILE (K_TILE / 16)
#define BYTES_PER_TILE (K_TILE / 2)

__device__ __forceinline__ float half_raw_to_float(const __half_raw& raw) {
    return __half2float(__ushort_as_half(raw.x));
}

// Decode two FP4 (E2M1) values packed in one byte into a half2
__device__ __forceinline__ __half2 decode_fp4x2(uint8_t byte) {
    __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(
        static_cast<__nv_fp4x2_storage_t>(byte),
        __NV_E2M1
    );
    return *reinterpret_cast<__half2*>(&raw);
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



__global__ void gemv_nvfp4_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L,
    int N_rows
) {

    // Parallelize: blockIdx.x -> M, blockIdx.y -> L, threads -> K reduction
    int m = blockIdx.x;
    int l = blockIdx.y;
    int tid = threadIdx.x;

    if (m >= M || l >= L) return;

// ============================================================================
// ===================== PER-CTA BASE POINTER SETUP ==========================
// ============================================================================
// ===================== PER-CTA BASE POINTER SETUP ==========================
// ============================================================================



    const int K_sf = K / 16;  // 16 elements per scale factor
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

    __shared__ float smem_acc[BLOCK_SIZE];
    __shared__ uint8_t sh_a[2][BYTES_PER_TILE];
    __shared__ uint8_t sh_b[2][BYTES_PER_TILE];
    __shared__ uint8_t sh_sfa[2][SCALES_PER_TILE];
    __shared__ uint8_t sh_sfb[2][SCALES_PER_TILE];
    float acc = 0.0f;

// ============================================================================
// ===================== MEMORY LOAD AND MAIN COMPUTE ========================
// ============================================================================
// ===================== MEMORY LOAD AND MAIN COMPUTE ========================
// ============================================================================

    int tile_count = K / K_TILE;
    int remainder_start = tile_count * K_TILE;
    int buf = 0;

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

        for (int i = tid * 16; i < BYTES_PER_TILE; i += BLOCK_SIZE * 16) {
            ASYNC_COPY_16(sh_a_base + i, row_a + base_byte + i);
            ASYNC_COPY_16(sh_b_base + i, batch_b + base_byte + i);
        }
        for (int i = tid * 4; i < SCALES_PER_TILE; i += BLOCK_SIZE * 4) {
            ASYNC_COPY_4(sh_sfa_base + i, row_sfa + base_sf + i);
            ASYNC_COPY_4(sh_sfb_base + i, batch_sfb + base_sf + i);
        }
        asm volatile("cp.async.commit_group;");
    };

    if (tile_count > 0) {
        issue_tile_async(0, 0);
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        for (int tile = 0; tile < tile_count; ++tile) {
            if (tile + 1 < tile_count) {
                issue_tile_async(buf ^ 1, tile + 1);
            }

            for (int sf = tid; sf < SCALES_PER_TILE; sf += BLOCK_SIZE) {
                float scale = decode_fp8(static_cast<int8_t>(sh_sfa[buf][sf])) *
                              decode_fp8(static_cast<int8_t>(sh_sfb[buf][sf]));
                __half scale_h = __float2half(scale);
                __half2 scale_h2 = __halves2half2(scale_h, scale_h);

                int byte_base = sf * 8;
#pragma unroll
                for (int bb = 0; bb < 8; ++bb) {
                    __half2 a2 = decode_fp4x2(sh_a[buf][byte_base + bb]);
                    __half2 b2 = decode_fp4x2(sh_b[buf][byte_base + bb]);
                    __half2 prod = __hmul2(__hmul2(a2, b2), scale_h2);
                    float2 f = __half22float2(prod);
                    acc += f.x + f.y;
                }
            }

            if (tile + 1 < tile_count) {
                asm volatile("cp.async.wait_group 0;");
                __syncthreads();
                buf ^= 1;
            }
        }
    }


    // remainder (less than K_TILE elements)
    int remainder_sf_start = remainder_start / 16;
    for (int sf = remainder_sf_start + tid; sf < K_sf; sf += BLOCK_SIZE) {
        float scale = decode_fp8(static_cast<int8_t>(row_sfa[sf])) *
                      decode_fp8(static_cast<int8_t>(batch_sfb[sf]));
        __half scale_h = __float2half(scale);
        __half2 scale_h2 = __halves2half2(scale_h, scale_h);

        int byte_base = sf * 8;
#pragma unroll
        for (int b_byte = 0; b_byte < 8; ++b_byte) {
            uint8_t a_pack = row_a[byte_base + b_byte];
            uint8_t b_pack = batch_b[byte_base + b_byte];
            __half2 a2 = decode_fp4x2(a_pack);
            __half2 b2 = decode_fp4x2(b_pack);
            __half2 prod = __hmul2(__hmul2(a2, b2), scale_h2);
            float2 f = __half22float2(prod);
            acc += f.x + f.y;
        }
    }

#undef ASYNC_COPY_16
#undef ASYNC_COPY_4

    smem_acc[tid] = acc;
    __syncthreads();

// ============================================================================
// ================== ACCUMULATION AND BLOCK REDUCTION =======================
// ============================================================================
// ================== ACCUMULATION AND BLOCK REDUCTION =======================
// ============================================================================
    // Phase 1: warp-level reduce in registers (no barriers).
    float warp_sum = acc;
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }

    // Phase 2: one partial per warp to shared, one barrier, final warp reduce.
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) smem_acc[warp_id] = warp_sum;

    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < (blockDim.x >> 5)) ? smem_acc[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
            c[c_idx] = __float2half(block_sum);
        }
    }



}



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
