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
// current perf order 64 -> 128 -> 256 -> 32
#define BLOCK_SIZE 64

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
    float acc = 0.0f;


// ============================================================================
// ===================== MEMORY LOAD AND MAIN COMPUTE ========================
// ============================================================================
// ===================== MEMORY LOAD AND MAIN COMPUTE ========================
// ============================================================================
    // Each thread processes K/128 elements (stride == blockDim.x)

    for (int k_block = tid; k_block < K_sf; k_block += BLOCK_SIZE) {

        // Load scale factors (FP8 E4M3 -> float) and combine once
        float scale = decode_fp8(static_cast<int8_t>(row_sfa[k_block])) *
                      decode_fp8(static_cast<int8_t>(batch_sfb[k_block]));
        __half scale_h = __float2half(scale);
        __half2 scale_h2 = __halves2half2(scale_h, scale_h);

        // Process 16 FP4 elements = 8 packed bytes
        int byte_base = k_block * 8;  // 8 bytes per 16 fp4 vals
#pragma unroll
        for (int b_byte = 0; b_byte < 8; ++b_byte) {
            int global_byte = byte_base + b_byte;
            int k0 = (k_block << 4) + (b_byte << 1);
            if (k0 >= K) break;  // guard partial tail

            uint8_t a_pack = row_a[global_byte];
            uint8_t b_pack = batch_b[global_byte];

            // Fast pairwise decode: two fp4 -> half2 in one instruction
            __half2 a2 = decode_fp4x2(a_pack);
            __half2 b2 = decode_fp4x2(b_pack);

            __half2 prod = __hmul2(__hmul2(a2, b2), scale_h2);
            float2 f = __half22float2(prod);
            acc += f.x + f.y;
        }

    }



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
