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

# CUDA implementation that mirrors the helpers from 1.py but adapts them for GEMM
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

// Global flag to limit debug output to first kernel launch only
__device__ int g_debug_done = 0;

struct Gemm_params {
    using index_t = uint64_t;
    int m, n, k, batches;
    const __nv_fp4x2_e2m1* __restrict__ a_ptr;
    const __nv_fp4x2_e2m1* __restrict__ b_ptr;
    const __nv_fp8_e4m3* __restrict__ sfa_ptr;
    const __nv_fp8_e4m3* __restrict__ sfb_ptr;
    __half* __restrict__ c_ptr;
    index_t a_batch_stride;
    index_t b_batch_stride;
    index_t row_stride;
    index_t sfa_batch_stride;
    index_t sfb_batch_stride;
    index_t sf_row_stride;
    index_t c_batch_stride;
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

    if (row >= params.m || batch >= params.batches) {
        return;
    }

    const int col_start = n_tile * N_TILE;
    if (col_start >= params.n) {
        return;
    }

    const size_t A_batch_base   = static_cast<size_t>(batch) * params.a_batch_stride;
    const size_t SFA_batch_base = static_cast<size_t>(batch) * params.sfa_batch_stride;
    const size_t B_batch_base   = static_cast<size_t>(batch) * params.b_batch_stride;
    const size_t SFB_batch_base = static_cast<size_t>(batch) * params.sfb_batch_stride;
    const size_t C_batch_base   = static_cast<size_t>(batch) * params.c_batch_stride;

    const __nv_fp4x2_e2m1* rowA = params.a_ptr + A_batch_base + row * params.row_stride;
    const uint16_t* rowS = reinterpret_cast<const uint16_t*>(
        params.sfa_ptr + SFA_batch_base + row * params.sf_row_stride);

    const __nv_fp4x2_e2m1* colB_ptrs[N_TILE];
    const uint16_t* colS_ptrs[N_TILE];
    bool col_active[N_TILE];

    #pragma unroll
    for (int ci = 0; ci < N_TILE; ++ci) {
        int col = col_start + ci;
        if (col < params.n) {
            col_active[ci] = true;
            colB_ptrs[ci] = params.b_ptr + B_batch_base + col * params.row_stride;
            colS_ptrs[ci] = reinterpret_cast<const uint16_t*>(
                params.sfb_ptr + SFB_batch_base + col * params.sf_row_stride);
        } else {
            col_active[ci] = false;
            colB_ptrs[ci] = nullptr;
            colS_ptrs[ci] = nullptr;
        }
    }

    float accum[N_TILE] = {0.f};

    const int bytes_per_iter = 16; // 2 uint64 fp4 = 16 byes / 32 fp4 vals
    const int iters = params.k / (K_WORKERS * bytes_per_iter);

    // Debug print - sample 3 different M x N tiles, first kernel launch only
    // Tile 0: (blockIdx.y=0, blockIdx.x=0) - first tile
    // Tile 1: (blockIdx.y=params.m/(2*M_TILE), blockIdx.x=params.n/(2*N_TILE)) - middle tile
    // Tile 2: (blockIdx.y=(params.m/M_TILE)-1, blockIdx.x=(params.n/N_TILE)-1) - last tile
    const int mid_y = (params.m / M_TILE) / 2;
    const int mid_x = (params.n / N_TILE) / 2;
    const int last_y = (params.m / M_TILE) - 1;
    const int last_x = (params.n / N_TILE) - 1;

    const bool is_debug_tile = (blockIdx.z == 0) && (m_idx == 0) && (k_lane == 0) && (
        (blockIdx.y == 0 && blockIdx.x == 0) ||
        (blockIdx.y == mid_y && blockIdx.x == mid_x) ||
        (blockIdx.y == last_y && blockIdx.x == last_x)
    );
    const bool debug = is_debug_tile && (atomicAdd(&g_debug_done, 0) < 3);
    if (debug) {
        printf("=== GEMM PARAMS DEBUG [tile blockIdx.y=%d, blockIdx.x=%d] ===\\n",
               (int)blockIdx.y, (int)blockIdx.x);
        printf("params.m=%d params.n=%d params.k=%d params.batches=%d\\n",
               params.m, params.n, params.k, params.batches);
        printf("params.a_ptr=%p params.b_ptr=%p params.c_ptr=%p\\n",
               (void*)params.a_ptr, (void*)params.b_ptr, (void*)params.c_ptr);
        printf("params.sfa_ptr=%p params.sfb_ptr=%p\\n",
               (void*)params.sfa_ptr, (void*)params.sfb_ptr);
        printf("params.a_batch_stride=%llu params.b_batch_stride=%llu\\n",
               (unsigned long long)params.a_batch_stride, (unsigned long long)params.b_batch_stride);
        printf("params.row_stride=%llu params.sf_row_stride=%llu\\n",
               (unsigned long long)params.row_stride, (unsigned long long)params.sf_row_stride);
        printf("params.sfa_batch_stride=%llu params.sfb_batch_stride=%llu\\n",
               (unsigned long long)params.sfa_batch_stride, (unsigned long long)params.sfb_batch_stride);
        printf("params.c_batch_stride=%llu\\n",
               (unsigned long long)params.c_batch_stride);
        printf("--- Computed values ---\\n");
        printf("batch=%d row=%d n_tile=%d col_start=%d iters=%d\\n",
               batch, row, n_tile, col_start, iters);
        printf("A_batch_base=%llu SFA_batch_base=%llu\\n", (unsigned long long)A_batch_base, (unsigned long long)SFA_batch_base);
        printf("B_batch_base=%llu SFB_batch_base=%llu\\n", (unsigned long long)B_batch_base, (unsigned long long)SFB_batch_base);
        printf("C_batch_base=%llu\\n", (unsigned long long)C_batch_base);
        printf("rowA offset from a_ptr: %llu\\n", (unsigned long long)(rowA - params.a_ptr));
        printf("rowS offset from sfa_ptr: %llu\\n", (unsigned long long)(rowS - reinterpret_cast<const uint16_t*>(params.sfa_ptr)));
        printf("colB_ptrs[0] offset from b_ptr: %llu (col_active[0]=%d)\\n",
               (unsigned long long)(col_active[0] ? (colB_ptrs[0] - params.b_ptr) : 0), (int)col_active[0]);
        printf("colS_ptrs[0] offset from sfb_ptr: %llu\\n",
               (unsigned long long)(col_active[0] ? (colS_ptrs[0] - reinterpret_cast<const uint16_t*>(params.sfb_ptr)) : 0));
        printf("bytes_per_iter=%d K_WORKERS=%d\\n", bytes_per_iter, K_WORKERS);
        printf("--- Column pointer details ---\\n");
        for (int ci = 0; ci < N_TILE; ++ci) {
            if (col_active[ci]) {
                printf("col[%d]: col=%d colB_ptr=%p colS_ptr=%p\\n",
                       ci, col_start + ci, (void*)colB_ptrs[ci], (void*)colS_ptrs[ci]);
                printf("  colB offset=%llu (expected col*row_stride=%llu)\\n",
                       (unsigned long long)(colB_ptrs[ci] - params.b_ptr),
                       (unsigned long long)((col_start + ci) * params.row_stride));
            }
        }
        printf("========================\\n");
    }

    #pragma unroll 4
    for (int iter = 0; iter < iters; ++iter) {
        int block_base = iter * K_WORKERS + k_lane;
        int elem_base = block_base * bytes_per_iter;

        uint64_t a_regs[2];
        uint16_t sfa_reg;
        load_row_block(rowA, rowS, elem_base, block_base, a_regs, sfa_reg);

        #pragma unroll
        for (int ci = 0; ci < N_TILE; ++ci) {
            if (!col_active[ci]) {
                continue;
            }
            uint64_t b_regs[2];
            uint16_t sfb_reg;
            load_col_block(colB_ptrs[ci], colS_ptrs[ci], elem_base, block_base, b_regs, sfb_reg);
            float result = block_scaled_fma_16x2fp4(a_regs, b_regs, sfa_reg, sfb_reg);

            // Debug: print loaded values and result for first iteration, first column
            if (debug && iter == 0 && ci == 0) {
                printf("=== COMPUTE DEBUG (iter=0, col=0) ===\\n");
                printf("elem_base=%d block_base=%d\\n", elem_base, block_base);
                printf("a_regs[0]=0x%016llx a_regs[1]=0x%016llx\\n",
                       (unsigned long long)a_regs[0], (unsigned long long)a_regs[1]);
                printf("b_regs[0]=0x%016llx b_regs[1]=0x%016llx\\n",
                       (unsigned long long)b_regs[0], (unsigned long long)b_regs[1]);
                printf("sfa_reg=0x%04x sfb_reg=0x%04x\\n", sfa_reg, sfb_reg);
                // Decode scales as two fp8 values
                uint8_t sfa_lo = sfa_reg & 0xFF;
                uint8_t sfa_hi = (sfa_reg >> 8) & 0xFF;
                uint8_t sfb_lo = sfb_reg & 0xFF;
                uint8_t sfb_hi = (sfb_reg >> 8) & 0xFF;
                printf("sfa bytes: lo=0x%02x hi=0x%02x, sfb bytes: lo=0x%02x hi=0x%02x\\n",
                       sfa_lo, sfa_hi, sfb_lo, sfb_hi);
                printf("block_scaled_fma result = %f\\n", result);

                // === EXPECTED VALUE COMPUTATION ===
                // FP4 E2M1 table: 4-bit value -> float
                // Format: 1 sign bit, 2 exp bits, 1 mantissa bit
                // Values: 0=0, 1=0.5, 2=1, 3=1.5, 4=2, 5=3, 6=4, 7=6
                //         8=-0, 9=-0.5, 10=-1, 11=-1.5, 12=-2, 13=-3, 14=-4, 15=-6
                const float fp4_lut[16] = {
                    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
                };

                // Decode FP8 E4M3 scales to float
                // E4M3: 1 sign, 4 exp (bias=7), 3 mantissa
                auto fp8_e4m3_to_float = [](uint8_t val) -> float {
                    uint8_t sign = (val >> 7) & 0x1;
                    uint8_t exp = (val >> 3) & 0xF;
                    uint8_t mant = val & 0x7;
                    float mantissa = 1.0f + mant / 8.0f;
                    float result;
                    if (exp == 0) {
                        result = (mant / 8.0f) * powf(2.0f, -6.0f); // subnormal
                    } else if (exp == 15 && mant == 7) {
                        result = nanf(""); // NaN
                    } else {
                        result = mantissa * powf(2.0f, (float)exp - 7.0f);
                    }
                    return sign ? -result : result;
                };

                float scale_a0 = fp8_e4m3_to_float(sfa_lo);
                float scale_a1 = fp8_e4m3_to_float(sfa_hi);
                float scale_b0 = fp8_e4m3_to_float(sfb_lo);
                float scale_b1 = fp8_e4m3_to_float(sfb_hi);

                printf("--- EXPECTED COMPUTATION ---\\n");
                printf("scale_a0=%f scale_a1=%f scale_b0=%f scale_b1=%f\\n",
                       scale_a0, scale_a1, scale_b0, scale_b1);
                printf("combined_scale0=%f combined_scale1=%f\\n",
                       scale_a0 * scale_b0, scale_a1 * scale_b1);

                // Decode first 16 FP4 pairs (from a_regs[0], b_regs[0])
                const uint8_t* a_bytes = reinterpret_cast<const uint8_t*>(&a_regs[0]);
                const uint8_t* b_bytes = reinterpret_cast<const uint8_t*>(&b_regs[0]);

                float dot0 = 0.0f;
                printf("First 16 FP4 pairs (a_regs[0] x b_regs[0]):\\n");
                for (int i = 0; i < 8; i++) {
                    uint8_t a_byte = a_bytes[i];
                    uint8_t b_byte = b_bytes[i];
                    // Each byte has 2 FP4 values: lo nibble and hi nibble
                    uint8_t a_lo = a_byte & 0xF;
                    uint8_t a_hi = (a_byte >> 4) & 0xF;
                    uint8_t b_lo = b_byte & 0xF;
                    uint8_t b_hi = (b_byte >> 4) & 0xF;
                    float a0 = fp4_lut[a_lo], a1 = fp4_lut[a_hi];
                    float b0 = fp4_lut[b_lo], b1 = fp4_lut[b_hi];
                    dot0 += a0 * b0 + a1 * b1;
                    if (i < 4) {
                        printf("  byte[%d]: a=0x%02x (%.1f,%.1f) b=0x%02x (%.1f,%.1f) products=(%.2f,%.2f)\\n",
                               i, a_byte, a0, a1, b_byte, b0, b1, a0*b0, a1*b1);
                    }
                }
                printf("  ... (4 more bytes)\\n");
                printf("  raw_dot0 = %f, scaled_dot0 = %f\\n", dot0, dot0 * scale_a0 * scale_b0);

                // Decode second 16 FP4 pairs (from a_regs[1], b_regs[1])
                const uint8_t* a_bytes1 = reinterpret_cast<const uint8_t*>(&a_regs[1]);
                const uint8_t* b_bytes1 = reinterpret_cast<const uint8_t*>(&b_regs[1]);

                float dot1 = 0.0f;
                printf("Second 16 FP4 pairs (a_regs[1] x b_regs[1]):\\n");
                for (int i = 0; i < 8; i++) {
                    uint8_t a_byte = a_bytes1[i];
                    uint8_t b_byte = b_bytes1[i];
                    uint8_t a_lo = a_byte & 0xF;
                    uint8_t a_hi = (a_byte >> 4) & 0xF;
                    uint8_t b_lo = b_byte & 0xF;
                    uint8_t b_hi = (b_byte >> 4) & 0xF;
                    float a0 = fp4_lut[a_lo], a1 = fp4_lut[a_hi];
                    float b0 = fp4_lut[b_lo], b1 = fp4_lut[b_hi];
                    dot1 += a0 * b0 + a1 * b1;
                    if (i < 4) {
                        printf("  byte[%d]: a=0x%02x (%.1f,%.1f) b=0x%02x (%.1f,%.1f) products=(%.2f,%.2f)\\n",
                               i, a_byte, a0, a1, b_byte, b0, b1, a0*b0, a1*b1);
                    }
                }
                printf("  ... (4 more bytes)\\n");
                printf("  raw_dot1 = %f, scaled_dot1 = %f\\n", dot1, dot1 * scale_a1 * scale_b1);

                float expected_result = dot0 * scale_a0 * scale_b0 + dot1 * scale_a1 * scale_b1;
                printf("EXPECTED total = %f, ACTUAL = %f, DIFF = %f\\n",
                       expected_result, result, result - expected_result);
                printf("====================================\\n");
            }

            accum[ci] += result;
        }
    }

    __half* row_out = params.c_ptr + C_batch_base + row * params.n;

    // === DEBUG: Verify per-k_lane partial sums before reduction ===
    // IMPORTANT: All 16 k_lanes must participate in shuffle (collective op)
    const bool is_debug_tile_all_lanes = (blockIdx.z == 0) && (m_idx == 0) && (
        (blockIdx.y == 0 && blockIdx.x == 0) ||
        (blockIdx.y == mid_y && blockIdx.x == mid_x) ||
        (blockIdx.y == last_y && blockIdx.x == last_x)
    ) && (atomicAdd(&g_debug_done, 0) < 3);

    if (is_debug_tile_all_lanes && col_active[0]) {
        float my_partial = accum[0];

        // Collect all 16 partial sums to k_lane 0 using shuffle
        // ALL 16 k_lanes execute this together, mask=0xFFFF for lanes 0-15
        float all_partials[16];
        for (int i = 0; i < 16; i++) {
            all_partials[i] = __shfl_sync(0xFFFF, my_partial, i, K_WORKERS);
        }

        // Only k_lane 0 prints
        if (k_lane == 0) {
            printf("\\n=== PER-K_LANE PARTIAL SUMS (before reduction) ===\\n");
            float sum_of_partials = 0.0f;
            for (int i = 0; i < 16; i++) {
                printf("k_lane[%2d] partial = %f\\n", i, all_partials[i]);
                sum_of_partials += all_partials[i];
            }
            printf("SUM of all partials = %f\\n", sum_of_partials);
            printf("=================================================\\n\\n");
        }
    }

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

    // === DEBUG: Verify full dot product for row=0, col=0 after reduction ===
    // Only one thread (the one that wrote the result) does this check
    if (debug && k_lane == 0 && col_active[0]) {
        // We need to recompute the ENTIRE dot product for row 0, col 0 manually
        // to compare against what we just wrote

        const float fp4_lut[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };

        auto fp8_e4m3_to_float = [](uint8_t val) -> float {
            uint8_t sign = (val >> 7) & 0x1;
            uint8_t exp = (val >> 3) & 0xF;
            uint8_t mant = val & 0x7;
            float mantissa = 1.0f + mant / 8.0f;
            float result;
            if (exp == 0) {
                result = (mant / 8.0f) * powf(2.0f, -6.0f);
            } else if (exp == 15 && mant == 7) {
                result = nanf("");
            } else {
                result = mantissa * powf(2.0f, (float)exp - 7.0f);
            }
            return sign ? -result : result;
        };

        printf("\\n=== FULL DOT PRODUCT VERIFICATION (row=0, col=0) ===\\n");

        // Get the value we wrote
        float written_value = __half2float(row_out[col_start]);
        printf("Written output value: %f\\n", written_value);

        // Recompute the full dot product from scratch
        const uint8_t* a_data = reinterpret_cast<const uint8_t*>(rowA);
        const uint8_t* b_data = reinterpret_cast<const uint8_t*>(colB_ptrs[0]);
        const uint8_t* sfa_data = reinterpret_cast<const uint8_t*>(rowS);
        const uint8_t* sfb_data = reinterpret_cast<const uint8_t*>(colS_ptrs[0]);

        float total_sum = 0.0f;
        int num_scale_blocks = params.k / 16;  // 16 bytes = 32 FP4 = 2 scale factors (16 each)

        printf("num_scale_blocks=%d (params.k=%d)\\n", num_scale_blocks, params.k);

        // Sample 6 random block indices to print
        int sample_blocks[6] = {0, 1, num_scale_blocks/4, num_scale_blocks/2, num_scale_blocks*3/4, num_scale_blocks-1};

        for (int blk = 0; blk < num_scale_blocks; blk++) {
            // Each block: 16 bytes of A, 16 bytes of B, 2 scale factors each
            int byte_offset = blk * 16;
            int scale_offset = blk * 2;  // 2 FP8 scales per 32 FP4 elements

            // Load scales
            uint8_t sfa_lo = sfa_data[scale_offset];
            uint8_t sfa_hi = sfa_data[scale_offset + 1];
            uint8_t sfb_lo = sfb_data[scale_offset];
            uint8_t sfb_hi = sfb_data[scale_offset + 1];

            float scale_a0 = fp8_e4m3_to_float(sfa_lo);
            float scale_a1 = fp8_e4m3_to_float(sfa_hi);
            float scale_b0 = fp8_e4m3_to_float(sfb_lo);
            float scale_b1 = fp8_e4m3_to_float(sfb_hi);

            // Compute dot product for first 8 bytes (16 FP4 pairs) with scale0
            float dot0 = 0.0f;
            for (int i = 0; i < 8; i++) {
                uint8_t a_byte = a_data[byte_offset + i];
                uint8_t b_byte = b_data[byte_offset + i];
                uint8_t a_lo = a_byte & 0xF;
                uint8_t a_hi = (a_byte >> 4) & 0xF;
                uint8_t b_lo = b_byte & 0xF;
                uint8_t b_hi = (b_byte >> 4) & 0xF;
                dot0 += fp4_lut[a_lo] * fp4_lut[b_lo] + fp4_lut[a_hi] * fp4_lut[b_hi];
            }

            // Compute dot product for next 8 bytes (16 FP4 pairs) with scale1
            float dot1 = 0.0f;
            for (int i = 0; i < 8; i++) {
                uint8_t a_byte = a_data[byte_offset + 8 + i];
                uint8_t b_byte = b_data[byte_offset + 8 + i];
                uint8_t a_lo = a_byte & 0xF;
                uint8_t a_hi = (a_byte >> 4) & 0xF;
                uint8_t b_lo = b_byte & 0xF;
                uint8_t b_hi = (b_byte >> 4) & 0xF;
                dot1 += fp4_lut[a_lo] * fp4_lut[b_lo] + fp4_lut[a_hi] * fp4_lut[b_hi];
            }

            float block_result = dot0 * scale_a0 * scale_b0 + dot1 * scale_a1 * scale_b1;
            total_sum += block_result;

            // Print sample blocks
            bool is_sample = false;
            for (int s = 0; s < 6; s++) {
                if (blk == sample_blocks[s]) {
                    is_sample = true;
                    break;
                }
            }
            if (is_sample) {
                printf("Block[%d]: byte_off=%d, scale_off=%d\\n", blk, byte_offset, scale_offset);
                printf("  scales: sfa=(0x%02x,0x%02x)->(%f,%f) sfb=(0x%02x,0x%02x)->(%f,%f)\\n",
                       sfa_lo, sfa_hi, scale_a0, scale_a1, sfb_lo, sfb_hi, scale_b0, scale_b1);
                printf("  dot0=%f dot1=%f block_result=%f running_sum=%f\\n",
                       dot0, dot1, block_result, total_sum);
            }
        }

        printf("\\nFINAL: expected_sum=%f, written_value=%f, diff=%f\\n",
               total_sum, written_value, written_value - total_sum);
        printf("==============================================\\n\\n");

        // Increment debug counter (allow up to 3 tiles to print)
        atomicAdd(&g_debug_done, 1);
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

    dim3 block_dim(M_TILE * K_WORKERS, 1, 1);
    dim3 grid_dim(
        (N + N_TILE - 1) / N_TILE,
        (M + M_TILE - 1) / M_TILE,
        L);

    Gemm_params params;
    params.m = M;
    params.n = N;
    params.k = K;
    params.batches = L;
    params.a_ptr = reinterpret_cast<const __nv_fp4x2_e2m1*>(A.data_ptr());
    params.b_ptr = reinterpret_cast<const __nv_fp4x2_e2m1*>(B.data_ptr());
    params.sfa_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(SFA.data_ptr());
    params.sfb_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(SFB.data_ptr());
    params.c_ptr = reinterpret_cast<__half*>(C.data_ptr());
    params.a_batch_stride = A.stride(2);
    params.b_batch_stride = B.stride(2);
    params.row_stride = A.stride(0);
    params.sfa_batch_stride = SFA.stride(2);
    params.sfb_batch_stride = SFB.stride(2);
    params.sf_row_stride = SFA.stride(0);
    params.c_batch_stride = C.stride(2);

    gemm_kernel<M_TILE, K_WORKERS, N_TILE><<<grid_dim, block_dim, 0>>>(params);
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
    #print(f"A.size(): {data[0].size()}, B.size(): {data[1].size()}, C.size(): {data[6].size()}, SFA.size(): {data[2].size()}, SFB.size(): {data[3].size()}")
    #print(f"A.stride(): {data[0].stride()}, B.stride(): {data[1].stride()}, C.stride(): {data[6].stride()}, SFA.stride(): {data[2].stride()}, SFB.stride(): {data[3].stride()}")
    return nvfp4_gemm_module.cuda_nvfp4_gemm(data[0], data[1], data[2], data[3], data[6])
