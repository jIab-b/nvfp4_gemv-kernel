import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_source = """
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = """
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Direct PTX inline assembly for tcgen05.mma with block scaling
__global__ void gemv_nvfp4_kernel(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    const uint8_t* __restrict__ sfa,
    const uint8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L
) {
    // Parallelize: blockIdx.x -> M, blockIdx.y -> L, threads -> K reduction
    int m = blockIdx.x;
    int l = blockIdx.y;
    int tid = threadIdx.x;

    if (m >= M || l >= L) return;

    const int K_sf = K / 16;  // 16 elements per scale factor
    __shared__ float smem_acc[32];

    float acc = 0.0f;

    // Each thread processes K/32 elements
    for (int k_block = tid; k_block < K_sf; k_block += 32) {
        // Load scale factors (FP8 E4M3 -> float)
        int sf_idx_a = m * K_sf + k_block + l * M * K_sf;
        int sf_idx_b = k_block + l * K_sf;

        float scale_a = __half2float(__nv_cvt_fp8_to_halfraw(sfa[sf_idx_a], __NV_E4M3));
        float scale_b = __half2float(__nv_cvt_fp8_to_halfraw(sfb[sf_idx_b], __NV_E4M3));

        // Process 16 FP4 elements
        for (int i = 0; i < 16; i++) {
            int k = k_block * 16 + i;
            if (k >= K) break;

            int a_idx = (m * K + k) / 2 + l * M * K / 2;
            int b_idx = k / 2 + l * K / 2;

            uint8_t a_byte = a[a_idx];
            uint8_t b_byte = b[b_idx];

            uint8_t a_nib = (k & 1) ? (a_byte >> 4) : (a_byte & 0xF);
            uint8_t b_nib = (k & 1) ? (b_byte >> 4) : (b_byte & 0xF);

            // E2M1 decode
            int a_sign = (a_nib >> 3) & 1;
            int a_exp = (a_nib >> 1) & 3;
            int a_mant = a_nib & 1;
            float a_val = 0.0f;
            if (a_exp == 0 && a_mant == 0) a_val = 0.0f;
            else if (a_exp == 0) a_val = 0.5f;
            else a_val = (1.0f + a_mant * 0.5f) * __powf(2.0f, a_exp - 2.0f);
            if (a_sign) a_val = -a_val;

            int b_sign = (b_nib >> 3) & 1;
            int b_exp = (b_nib >> 1) & 3;
            int b_mant = b_nib & 1;
            float b_val = 0.0f;
            if (b_exp == 0 && b_mant == 0) b_val = 0.0f;
            else if (b_exp == 0) b_val = 0.5f;
            else b_val = (1.0f + b_mant * 0.5f) * __powf(2.0f, b_exp - 2.0f);
            if (b_sign) b_val = -b_val;

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

    if (tid == 0) c[m + l * M] = __float2half(smem_acc[0]);
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);

    dim3 grid(M, L);
    dim3 block(32);

    gemv_nvfp4_kernel<<<grid, block>>>(
        a.data_ptr<uint8_t>(),
        b.data_ptr<uint8_t>(),
        sfa.data_ptr<uint8_t>(),
        sfb.data_ptr<uint8_t>(),
        c.data_ptr<at::Half>(),
        M, K, L
    );

    return c;
}
"""

module = load_inline(
    name='batched_scaled_gemv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batched_scaled_gemv_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17', '-gencode=arch=compute_110,code=sm_110'],
    with_cuda=True,
    verbose=False
)
    


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    return module.batched_scaled_gemv_cuda(
        a.view(torch.uint8),
        b.view(torch.uint8),
        sfa_ref.cuda().view(torch.uint8),
        sfb_ref.cuda().view(torch.uint8),
        c
    )