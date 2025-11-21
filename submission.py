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



// Direct PTX inline assembly for tcgen05.mma with block scaling

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



    __shared__ float smem_acc[32];



    float acc = 0.0f;



    // Each thread processes K/32 elements

    for (int k_block = tid; k_block < K_sf; k_block += 32) {

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



            uint8_t a_nib = (k & 1) ? (a_byte >> 4) : (a_byte & 0xF);

            uint8_t b_nib = (k & 1) ? (b_byte >> 4) : (b_byte & 0xF);



            float a_val = decode_fp4(a_nib);

            float b_val = decode_fp4(b_nib);



            acc += (a_val * scale_a) * (b_val * scale_b);

        }

    }



    smem_acc[tid] = acc;

    __syncthreads();



    // Reduce

    for (int s = 16; s > 0; s >>= 1) {

        if (tid < s) smem_acc[tid] += smem_acc[tid + s];

        __syncthreads();

    }



    if (tid == 0) {

        size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;

        c[c_idx] = __float2half(smem_acc[0]);

    }

}



torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {

    int M = a.size(0);

    int K = a.size(1) * 2;

    int L = a.size(2);

    int N_rows = b.size(0);



    

    dim3 grid(M, L);

    dim3 block(32);



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

        '-gencode=arch=compute_100,code=sm_100',

        '-gencode=arch=compute_110,code=sm_110'

    ],

    with_cuda=True,

    verbose=False

)



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