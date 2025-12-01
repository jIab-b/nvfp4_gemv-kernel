import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# C++ stub to expose the CUDA launcher
cpp_src = """
#include <torch/extension.h>

torch::Tensor cuda_nvfp4_gemm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor SFA,
    torch::Tensor SFB);
"""

# CUDA implementation that mirrors the helpers from 1.py but adapts them for GEMM
cuda_src = """
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

struct GemmParams {
    using index_t = uint64_t;

    int m;
    int n;
    int k;
    int batches;

    void* a_ptr;
    void* b_ptr;
    void* sfa_ptr;
    void* sfb_ptr;
    void* o_ptr;

    index_t a_batch_stride;
    index_t b_batch_stride;
    index_t sfa_batch_stride;
    index_t sfb_batch_stride;
    index_t c_batch_stride;

    index_t a_row_stride;
    index_t b_row_stride;
    index_t sfa_row_stride;
    index_t sfb_row_stride;
    index_t c_row_stride;
    index_t c_col_stride;
};

static constexpr int THREADS_PER_ROW = 16;
static constexpr int ROWS_PER_BLOCK = 8;
static constexpr int COLS_PER_GROUP = 4;

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
        // 8 bytes of A and B at a time (reused for upper half)
        ".reg .b8 a0_0, a0_1, a0_2, a0_3;\\n"
        ".reg .b8 a0_4, a0_5, a0_6, a0_7;\\n"
        ".reg .b8 b0_0, b0_1, b0_2, b0_3;\\n"
        ".reg .b8 b0_4, b0_5, b0_6, b0_7;\\n"

        // scales and accumulators
        ".reg .f16x2 sfa_f16x2, sfb_f16x2, sf_f16x2;\\n"
        ".reg .f16x2 scale0_f16x2, scale1_f16x2;\\n"
        ".reg .f16x2 accum_total, accum_group;\\n"

        // converted fp4 -> f16x2 (only 8 per vector kept live)
        ".reg .f16x2 cvt_0_0, cvt_0_1, cvt_0_2, cvt_0_3;\\n"
        ".reg .f16x2 cvt_0_4, cvt_0_5, cvt_0_6, cvt_0_7;\\n"
        ".reg .f16x2 cvt_1_0, cvt_1_1, cvt_1_2, cvt_1_3;\\n"
        ".reg .f16x2 cvt_1_4, cvt_1_5, cvt_1_6, cvt_1_7;\\n"

        ".reg .f16 lane0, lane1, result_f16;\\n"
        ".reg .f32 result_f32;\\n"

        // scales
        "cvt.rn.f16x2.e4m3x2 sfa_f16x2, %5;\\n"
        "cvt.rn.f16x2.e4m3x2 sfb_f16x2, %6;\\n"
        "mul.rn.f16x2 sf_f16x2, sfa_f16x2, sfb_f16x2;\\n"
        "mov.b32 {lane0, lane1}, sf_f16x2;\\n"
        "mov.b32 scale0_f16x2, {lane0, lane0};\\n"
        "mov.b32 scale1_f16x2, {lane1, lane1};\\n"

        "mov.b32 accum_total, 0;\\n"

        // First 8×(2×FP4)
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

        // Second 8×(2×FP4) - reuse same data with scale1
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

template <int ROWS_PER_BLOCK, int THREADS_PER_ROW, int COLS_PER_GROUP>
__global__ void __launch_bounds__(ROWS_PER_BLOCK * THREADS_PER_ROW)
gemm_kernel(const __grid_constant__ GemmParams params)
{
    const int tid  = threadIdx.x;
    const int warp = tid / THREADS_PER_ROW;
    const int lane = tid % THREADS_PER_ROW;

    const int batch = blockIdx.z;
    const int row   = blockIdx.y * ROWS_PER_BLOCK + warp;
    const int col_tile = blockIdx.x;

    if (row >= params.m || batch >= params.batches) {
        return;
    }

    const int col_start = col_tile * COLS_PER_GROUP;
    if (col_start >= params.n) {
        return;
    }

    const size_t A_batch_base   = static_cast<size_t>(batch) * params.a_batch_stride;
    const size_t SFA_batch_base = static_cast<size_t>(batch) * params.sfa_batch_stride;
    const size_t B_batch_base   = static_cast<size_t>(batch) * params.b_batch_stride;
    const size_t SFB_batch_base = static_cast<size_t>(batch) * params.sfb_batch_stride;
    const size_t C_batch_base   = static_cast<size_t>(batch) * params.c_batch_stride;

    const __nv_fp4x2_e2m1* a_matrix = static_cast<const __nv_fp4x2_e2m1*>(params.a_ptr);
    const __nv_fp8_e4m3*   sfa_tensor = static_cast<const __nv_fp8_e4m3*>(params.sfa_ptr);
    const __nv_fp4x2_e2m1* b_matrix = static_cast<const __nv_fp4x2_e2m1*>(params.b_ptr);
    const __nv_fp8_e4m3*   sfb_tensor = static_cast<const __nv_fp8_e4m3*>(params.sfb_ptr);
    __half* c_matrix = static_cast<__half*>(params.o_ptr);

    const __nv_fp4x2_e2m1* rowA = a_matrix + A_batch_base + row * params.a_row_stride;
    const uint16_t* rowS = reinterpret_cast<const uint16_t*>(
        sfa_tensor + SFA_batch_base + row * params.sfa_row_stride);

    const __nv_fp4x2_e2m1* colB_ptrs[COLS_PER_GROUP];
    const uint16_t* colS_ptrs[COLS_PER_GROUP];
    bool col_active[COLS_PER_GROUP];

    #pragma unroll
    for (int ci = 0; ci < COLS_PER_GROUP; ++ci) {
        int col = col_start + ci;
        if (col < params.n) {
            col_active[ci] = true;
            colB_ptrs[ci] = b_matrix + B_batch_base + col * params.b_row_stride;
            colS_ptrs[ci] = reinterpret_cast<const uint16_t*>(
                sfb_tensor + SFB_batch_base + col * params.sfb_row_stride);
        } else {
            col_active[ci] = false;
            colB_ptrs[ci] = nullptr;
            colS_ptrs[ci] = nullptr;
        }
    }

    float accum[COLS_PER_GROUP] = {0.f};

    const int iters = params.k / (THREADS_PER_ROW * 16);

    #pragma unroll 4
    for (int iter = 0; iter < iters; ++iter) {
        int block_base = iter * THREADS_PER_ROW + lane;
        int elem_base = block_base * 16;

        uint64_t a_regs[2];
        uint16_t sfa_reg;
        load_row_block(rowA, rowS, elem_base, block_base, a_regs, sfa_reg);

        #pragma unroll
        for (int ci = 0; ci < COLS_PER_GROUP; ++ci) {
            if (!col_active[ci]) {
                continue;
            }
            uint64_t b_regs[2];
            uint16_t sfb_reg;
            load_col_block(colB_ptrs[ci], colS_ptrs[ci], elem_base, block_base, b_regs, sfb_reg);
            accum[ci] += block_scaled_fma_16x2fp4(a_regs, b_regs, sfa_reg, sfb_reg);
        }
    }

    __half* row_out = c_matrix + C_batch_base + row * params.c_row_stride;

    #pragma unroll
    for (int ci = 0; ci < COLS_PER_GROUP; ++ci) {
        if (!col_active[ci]) {
            continue;
        }
        float value = accum[ci];
        for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xFFFF'FFFF, value, offset, THREADS_PER_ROW);
        }
        if (lane == 0) {
            int col = col_start + ci;
            __half* out_ptr = row_out + col * params.c_col_stride;
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
    const int K = static_cast<int>(a_sizes[1]);
    const int L = static_cast<int>(a_sizes[2]);

    const int N = static_cast<int>(B.size(0));

    GemmParams params{};
    params.m = M;
    params.n = N;
    params.k = K;
    params.batches = L;

    params.a_ptr = A.data_ptr();
    params.b_ptr = B.data_ptr();
    params.sfa_ptr = SFA.data_ptr();
    params.sfb_ptr = SFB.data_ptr();
    params.o_ptr = C.data_ptr();

    params.a_batch_stride = A.stride(2);
    params.b_batch_stride = B.stride(2);
    params.sfa_batch_stride = SFA.stride(2);
    params.sfb_batch_stride = SFB.stride(2);
    params.c_batch_stride = C.stride(2);

    params.a_row_stride = A.stride(0);
    params.b_row_stride = B.stride(0);
    params.sfa_row_stride = SFA.stride(0);
    params.sfb_row_stride = SFB.stride(0);
    params.c_row_stride = C.stride(0);
    params.c_col_stride = C.stride(1);

    dim3 block_dim(ROWS_PER_BLOCK * THREADS_PER_ROW, 1, 1);
    dim3 grid_dim(
        (params.n + COLS_PER_GROUP - 1) / COLS_PER_GROUP,
        (params.m + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK,
        params.batches);

    gemm_kernel<ROWS_PER_BLOCK, THREADS_PER_ROW, COLS_PER_GROUP><<<grid_dim, block_dim, 0>>>(params);
    return C;
}
"""

nvfp4_gemm_module = load_inline(
    name="nvfp4_blockscaled_gemm",
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
