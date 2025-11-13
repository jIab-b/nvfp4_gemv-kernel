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

__global__ void gemv_nvfp4_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L
) {
    int tile_row = blockIdx.x * 16;
    int batch = blockIdx.y;
    int lane = threadIdx.x;

    if (tile_row >= M || batch >= L) return;

    const int K_tile = 64;
    const int num_tiles = K / K_tile;

    extern __shared__ uint8_t smem[];
    __shared__ uint32_t tmem_buf[32 * 32];

    void* d_tmem;
    asm volatile("tcgen05.alloc tma.u32.shared::cluster [%0], %1;" : "=r"(d_tmem) : "r"(32));

    asm volatile("tcgen05.set.zero.b32 [%0];" :: "r"(d_tmem));

    for (int t = 0; t < num_tiles; ++t) {
        const uint8_t* tile_a = reinterpret_cast<const uint8_t*>(a + (batch * M * (K/2)) + (tile_row * (K/2)) + t * (K_tile/2));
        const uint8_t* tile_b = reinterpret_cast<const uint8_t*>(b + (batch * (K/2)) + t * (K_tile/2));
        const int8_t* tile_sfa = sfa + batch * M * (K/16) + tile_row * (K/16) + t * 4;
        const int8_t* tile_sfb = sfb + batch * (K/16) + t * 4;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            reinterpret_cast<uint4*>(&smem[0])[lane + i * 8] = reinterpret_cast<const uint4*>(tile_a)[lane + i * 8];
            reinterpret_cast<uint4*>(&smem[2048])[lane + i * 8] = reinterpret_cast<const uint4*>(tile_b)[lane + i * 8];
        }

        __syncthreads();

    asm volatile(
        "tcgen05.mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 [%0], [%1], [%2], [%3], {%4, %5}, [%6], {%7, %8}, enable_d_zero;\n"
        :
        : "r"(d_tmem),
          "r"(&smem[0]),
          "r"(&smem[2048]),
          "r"(tmem_buf),
          "r"(0), "r"(lane),
          "r"(tmem_buf + 64),
          "r"(0), "r"(lane)
    );
    }

    float acc_fp32;
    asm volatile("tcgen05.ldmatrix.sync.aligned.m16n8.row.f32 [%0], [%1];" : "=f"(acc_fp32) : "r"(d_tmem));
    half acc_fp16 = __float2half(acc_fp32);

    if (lane < 16 && tile_row + lane < M) {
        c[(tile_row + lane) + batch * M] = acc_fp16;
    }

    asm volatile("tcgen05.dealloc tma.u32.shared::cluster [%0];" :: "r"(d_tmem));
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);

    dim3 grid(M / 16, L);
    dim3 block(32);

    gemv_nvfp4_kernel<<<grid, block, 4096>>>(
        reinterpret_cast<const int8_t*>(a.data_ptr()),
        reinterpret_cast<const int8_t*>(b.data_ptr()),
        reinterpret_cast<const int8_t*>(sfa.data_ptr()),
        reinterpret_cast<const int8_t*>(sfb.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, K, L);

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
        '-gencode=arch=compute_110,code=sm_110',
        '-gencode=arch=compute_100,code=sm_100'
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
