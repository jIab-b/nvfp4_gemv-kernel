import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# C++ stub to expose the CUDA launcher
cpp_src = """
#include <torch/extension.h>

torch::Tensor cuda_nvfp4_gemm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor C);
"""

# CUDA implementation with corrected stride math
# Key fix: params.k = k//2 (packed bytes), so:
#   - Data offset: row * params.k (not params.k / 2)
#   - Scale offset: row * (params.k / 8) (not params.k / 16)
cuda_src = """
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cstdio>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

static constexpr int K_WORKERS = 16;
static constexpr int M_TILE = 8;
static constexpr int N_TILE = 4;

struct Gemm_params {
    using index_t = uint64_t;

    int b, m, n, k, real_k;

    void *__restrict__ a_ptr;
    void *__restrict__ b_ptr;
    void *__restrict__ sfa_ptr;
    void *__restrict__ sfb_ptr;
    void *__restrict__ c_ptr;


    index_t a_row_stride;
    index_t b_row_stride;
    index_t sfa_row_stride;
    index_t sfb_row_stride;
    index_t c_row_stride;
};
__device__ __forceinline__ void load_row_block(
    const __nv_fp4x2_e2m1* row_ptr,
    const uint16_t*        row_scale_ptr,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[2],
    uint16_t &sfa_regs)
{
    uint64_t row_addr = reinterpret_cast<uint64_t>(row_ptr + elem_base);
    uint64_t scale_addr = reinterpret_cast<uint64_t>(row_scale_ptr + block_base);

    asm volatile(
        "ld.global.u64.v2 {%0, %1}, [%2];\\n"
        : "=l"(a_regs[0]), "=l"(a_regs[1])
        : "l"(row_addr)
    );

    asm volatile(
        "ld.global.u16 %0, [%1];\\n"
        : "=h"(sfa_regs)
        : "l"(scale_addr)
    );
}

__device__ __forceinline__ void load_col_block(
    const __nv_fp4x2_e2m1* col_ptr,
    const uint16_t*        col_scale_ptr,
    int                    elem_base,
    int                    block_base,
    uint64_t (&b_regs)[2],
    uint16_t &sfb_regs)
{
    uint64_t col_addr = reinterpret_cast<uint64_t>(col_ptr + elem_base);
    uint64_t scale_addr = reinterpret_cast<uint64_t>(col_scale_ptr + block_base);

    asm volatile(
        "ld.global.u64.v2 {%0, %1}, [%2];\\n"
        : "=l"(b_regs[0]), "=l"(b_regs[1])
        : "l"(col_addr)
    );

    asm volatile(
        "ld.global.u16 %0, [%1];\\n"
        : "=h"(sfb_regs)
        : "l"(scale_addr)
    );
}

__device__ __forceinline__ float block_scaled_fma_16x2fp4(
    const uint64_t (&a_regs)[2],
    const uint64_t (&b_regs)[2],
    uint16_t       sfa_regs,
    uint16_t       sfb_regs)
{
    uint32_t const* a_regs_packed = reinterpret_cast<uint32_t const*>(&a_regs);
    uint32_t const* b_regs_packed = reinterpret_cast<uint32_t const*>(&b_regs);

    float out_f32;

    asm volatile(
        "{\\n"
        ".reg .b8 a0_0, a0_1, a0_2, a0_3;\\n"
        ".reg .b8 a0_4, a0_5, a0_6, a0_7;\\n"
        ".reg .b8 b0_0, b0_1, b0_2, b0_3;\\n"
        ".reg .b8 b0_4, b0_5, b0_6, b0_7;\\n"

        ".reg .f16x2 sfa_f16x2, sfb_f16x2, sf_f16x2;\\n"
        ".reg .f16x2 scale0_f16x2, scale1_f16x2;\\n"
        ".reg .f16x2 accum_total, accum_group;\\n"

        ".reg .f16x2 cvt_0_0, cvt_0_1, cvt_0_2, cvt_0_3;\\n"
        ".reg .f16x2 cvt_0_4, cvt_0_5, cvt_0_6, cvt_0_7;\\n"
        ".reg .f16x2 cvt_1_0, cvt_1_1, cvt_1_2, cvt_1_3;\\n"
        ".reg .f16x2 cvt_1_4, cvt_1_5, cvt_1_6, cvt_1_7;\\n"

        ".reg .f16 lane0, lane1, result_f16;\\n"
        ".reg .f32 result_f32;\\n"

        "cvt.rn.f16x2.e4m3x2 sfa_f16x2, %5;\\n"
        "cvt.rn.f16x2.e4m3x2 sfb_f16x2, %6;\\n"
        "mul.rn.f16x2 sf_f16x2, sfa_f16x2, sfb_f16x2;\\n"
        "mov.b32 {lane0, lane1}, sf_f16x2;\\n"
        "mov.b32 scale0_f16x2, {lane0, lane0};\\n"
        "mov.b32 scale1_f16x2, {lane1, lane1};\\n"

        "mov.b32 accum_total, 0;\\n"

        "mov.b32 {a0_0, a0_1, a0_2, a0_3}, %1;\\n"
        "mov.b32 {a0_4, a0_5, a0_6, a0_7}, %2;\\n"
        "mov.b32 {b0_0, b0_1, b0_2, b0_3}, %3;\\n"
        "mov.b32 {b0_4, b0_5, b0_6, b0_7}, %4;\\n"

        "cvt.rn.f16x2.e2m1x2 cvt_0_0, a0_0;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_0, b0_0;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_1, a0_1;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_1, b0_1;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_2, a0_2;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_2, b0_2;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_3, a0_3;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_3, b0_3;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_4, a0_4;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_4, b0_4;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_5, a0_5;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_5, b0_5;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_6, a0_6;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_6, b0_6;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_7, a0_7;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_7, b0_7;\\n"

        "mov.b32 accum_group, 0;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_0, cvt_1_0, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_1, cvt_1_1, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_2, cvt_1_2, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_3, cvt_1_3, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_4, cvt_1_4, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_5, cvt_1_5, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_6, cvt_1_6, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_7, cvt_1_7, accum_group;\\n"
        "mul.rn.f16x2 accum_group, scale0_f16x2, accum_group;\\n"
        "add.rn.f16x2 accum_total, accum_total, accum_group;\\n"

        "mov.b32 {a0_0, a0_1, a0_2, a0_3}, %7;\\n"
        "mov.b32 {a0_4, a0_5, a0_6, a0_7}, %8;\\n"
        "mov.b32 {b0_0, b0_1, b0_2, b0_3}, %9;\\n"
        "mov.b32 {b0_4, b0_5, b0_6, b0_7}, %10;\\n"

        "cvt.rn.f16x2.e2m1x2 cvt_0_0, a0_0;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_0, b0_0;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_1, a0_1;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_1, b0_1;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_2, a0_2;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_2, b0_2;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_3, a0_3;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_3, b0_3;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_4, a0_4;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_4, b0_4;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_5, a0_5;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_5, b0_5;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_6, a0_6;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_6, b0_6;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_7, a0_7;\\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_7, b0_7;\\n"

        "mov.b32 accum_group, 0;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_0, cvt_1_0, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_1, cvt_1_1, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_2, cvt_1_2, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_3, cvt_1_3, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_4, cvt_1_4, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_5, cvt_1_5, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_6, cvt_1_6, accum_group;\\n"
        "fma.rn.f16x2 accum_group, cvt_0_7, cvt_1_7, accum_group;\\n"
        "mul.rn.f16x2 accum_group, scale1_f16x2, accum_group;\\n"
        "add.rn.f16x2 accum_total, accum_total, accum_group;\\n"

        "mov.b32 {lane0, lane1}, accum_total;\\n"
        "add.rn.f16 result_f16, lane0, lane1;\\n"
        "cvt.f32.f16 result_f32, result_f16;\\n"
        "mov.b32 %0, result_f32;\\n"
        "}\\n"
        : "=f"(out_f32)
        : "r"(a_regs_packed[0]), "r"(a_regs_packed[1]),
          "r"(b_regs_packed[0]), "r"(b_regs_packed[1]),
          "h"(sfa_regs), "h"(sfb_regs),
          "r"(a_regs_packed[2]), "r"(a_regs_packed[3]),
          "r"(b_regs_packed[2]), "r"(b_regs_packed[3])
        : "memory"
    );

    return out_f32;
}

template <int M_TILE, int K_WORKERS, int N_TILE>
__global__ void __launch_bounds__(M_TILE * K_WORKERS)
gemm_kernel(const __grid_constant__ Gemm_params params)
{
    const int tid  = threadIdx.x;
    const int m_idx = tid / K_WORKERS;
    const int k_lane = tid % K_WORKERS;

    const int batch = blockIdx.z;
    const int row   = blockIdx.y * M_TILE + m_idx;
    const int n_tile = blockIdx.x;

    if (row >= params.m) {
        return;
    }

    const int col_start = n_tile * N_TILE;
    if (col_start >= params.n) {
        return;
    }


    // Use actual tensor strides (like the working GEMV does)
    const __nv_fp4x2_e2m1* rowA = static_cast<const __nv_fp4x2_e2m1*>(params.a_ptr) + row * params.a_row_stride;
    const __nv_fp8_e4m3* rowSFA = static_cast<const __nv_fp8_e4m3*>(params.sfa_ptr) + row * params.sfa_row_stride;

    // Scale pointer: offset in FP8 units first, then cast to uint16_t*
    const uint16_t* rowSFA_u16 = reinterpret_cast<const uint16_t*>(rowSFA);


    const __nv_fp4x2_e2m1* colB_ptrs[N_TILE];
    const uint16_t* colS_ptrs[N_TILE];
    bool col_active[N_TILE];

    #pragma unroll
    for (int ci = 0; ci < N_TILE; ++ci) {
        int col = col_start + ci;
        if (col < params.n) {
            col_active[ci] = true;

            // Use actual tensor stride for B
            colB_ptrs[ci] = static_cast<const __nv_fp4x2_e2m1*>(params.b_ptr) + col * params.b_row_stride;

            // Scale pointer: offset in FP8 units first, then cast to uint16_t*
            const __nv_fp8_e4m3* colSFA = static_cast<const __nv_fp8_e4m3*>(params.sfb_ptr) + col * params.sfb_row_stride;
            colS_ptrs[ci] = reinterpret_cast<const uint16_t*>(colSFA);
        } else {
            col_active[ci] = false;
            colB_ptrs[ci] = nullptr;
            colS_ptrs[ci] = nullptr;
        }
    }

    float accum[N_TILE] = {0.f};

    const int bytes_per_iter = 16; // 16 bytes = 32 FP4 values
    const int iters = params.k / (K_WORKERS * bytes_per_iter);

    #pragma unroll 4
    for (int iter = 0; iter < iters; ++iter) {
        int block_base = iter * K_WORKERS + k_lane;
        int elem_base = block_base * bytes_per_iter;

        // scale_block_base: offset into scale array (in uint16_t units)
        // Each uint16 contains 2 FP8 scales covering 32 FP4 values (16 bytes)
        int scale_block_base = block_base;

        uint64_t a_regs[2];
        uint16_t sfa_reg;
        load_row_block(rowA, rowSFA_u16, elem_base, scale_block_base, a_regs, sfa_reg);

        #pragma unroll
        for (int ci = 0; ci < N_TILE; ++ci) {
            if (!col_active[ci]) {
                continue;
            }
            uint64_t b_regs[2];
            uint16_t sfb_reg;
            load_col_block(colB_ptrs[ci], colS_ptrs[ci], elem_base, scale_block_base, b_regs, sfb_reg);
            float result = block_scaled_fma_16x2fp4(a_regs, b_regs, sfa_reg, sfb_reg);
            accum[ci] += result;
        }
    }

    __half* row_out = static_cast<__half*>(params.c_ptr) + row * params.c_row_stride;

    #pragma unroll
    for (int ci = 0; ci < N_TILE; ++ci) {
        if (!col_active[ci]) {
            continue;
        }
        float value = accum[ci];
        for (int offset = K_WORKERS / 2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xFFFF'FFFF, value, offset, K_WORKERS);
        }
        if (k_lane == 0) {
            int col = col_start + ci;
            __half* out_ptr = row_out + col;
            out_ptr[0] = __float2half(value);
        }
    }
}

torch::Tensor cuda_nvfp4_gemm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor C)
{
    const auto a_sizes = A.sizes();
    const int M = static_cast<int>(a_sizes[0]);
    const int K = static_cast<int>(a_sizes[1]);  // This is k//2 (packed bytes)

    const int N = static_cast<int>(B.size(0));

    dim3 block_dim(M_TILE * K_WORKERS, 1, 1);
    dim3 grid_dim(
        (N + N_TILE - 1) / N_TILE,
        (M + M_TILE - 1) / M_TILE,
        1);

    Gemm_params params;
    params.m = M;
    params.n = N;
    params.k = K;  // k//2 in packed bytes
    params.a_ptr = A.data_ptr();
    params.b_ptr = B.data_ptr();
    params.sfa_ptr = SFA.data_ptr();
    params.sfb_ptr = SFB.data_ptr();
    params.c_ptr = C.data_ptr();

    params.a_row_stride = A.stride(0);
    params.b_row_stride = B.stride(0);
    params.sfa_row_stride = SFA.stride(0);
    params.sfb_row_stride = SFB.stride(0);
    params.c_row_stride = C.stride(0);

    gemm_kernel<M_TILE, K_WORKERS, N_TILE><<<grid_dim, block_dim, 0>>>(params);
    return C;
}
"""

nvfp4_gemm_module = load_inline(
    name="nvfp4_blockscaled_gemm_claude",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["cuda_nvfp4_gemm"],
    extra_cuda_cflags=[
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--ptxas-options=--gpu-name=sm_100a",
        "-O3",
        "-w",
        "-maxrregcount=64",
        "--use_fast_math",
        "-allow-unsupported-compiler",
    ],
    extra_ldflags=["-lcuda", "-lcublas"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    return nvfp4_gemm_module.cuda_nvfp4_gemm(data[0], data[1], data[2], data[3], data[6])
