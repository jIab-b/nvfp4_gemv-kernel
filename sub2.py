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

    TILE_K = 256;
    // 2 fp4 / uint8
    __shared__ uint8 sh_a_pack[2][TILE_K/2];  __shared__uint8 sh_b_pack[2][TILE_K/2];

    //  2 fp4 in one byte -> decodes to 2 fp16, one fp8 -> deocde to one fp16 
    __shared__ half2 sh_decode_a[2][TILE_K/2]; __shared__ half2 sh_decode_a[2][TILE_K/2];
    __shared__ half sh_scale_a[TILE_K/16]; __shared__ half sh_scale_b[TILE_K/16];

    __shared__ float smem_acc[128];
    float acc = 0.0f;



// ============================================================================
// ===================== MEMORY LOAD AND MAIN COMPUTE ========================
// ============================================================================
// ===================== MEMORY LOAD AND MAIN COMPUTE ========================
// ============================================================================
    // Each thread processes K/32 elements (stride == blockDim.x)

    for (int k_block = tid; k_block < K_sf; k_block += 128) {

        // Load scale factors (FP8 E4M3 -> float)

        float scale_a = decode_fp8(static_cast<int8_t>(row_sfa[k_block]));

        float scale_b = decode_fp8(static_cast<int8_t>(batch_sfb[k_block]));



        // Process 16 FP4 elements
        for (int i = 0; i < 16; i++) {
            int k = k_block * 16 + i;
            if (k >= K) break;

            int byte_idx = k / 2;
            uint8_t a_byte = row_a[byte_idx];
            uint8_t b_byte = batch_b[byte_idx];

            // k = odd, take last 4 bits, k even, take first 4 bits
            uint8_t a_nib = (k & 1) ? (a_byte >> 4) : (a_byte & 0xF);
            uint8_t b_nib = (k & 1) ? (b_byte >> 4) : (b_byte & 0xF);

            float a_val = decode_fp4(a_nib);
            float b_val = decode_fp4(b_nib);

            acc += (a_val * scale_a) * (b_val * scale_b);

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
    dim3 block(128);

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



