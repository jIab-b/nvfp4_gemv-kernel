import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


cpp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// Same block layout as sub_sync_v4: one producer warp + 3 consumer warps
#define MAX_K 16384
#define MAX_K_HALF (MAX_K / 2)
#define MAX_K_SF (MAX_K / 16)
#define BLOCK_SIZE 128
#define ROWS_PER_BLOCK 3
#define PRODUCER_THREADS 32

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
    return __half2float(__ushort_as_half(raw.x));
}

__device__ __forceinline__ __half2 dot_scaled_4bytes(
    uchar4 a4,
    uchar4 b4,
    __half2 scale_h2
) {
    __half2 acc0 = __hmul2(decode_fp4x2(a4.x), __hmul2(decode_fp4x2(b4.x), scale_h2));
    __half2 acc1 = __hmul2(decode_fp4x2(a4.y), __hmul2(decode_fp4x2(b4.y), scale_h2));
    acc0 = __hfma2(decode_fp4x2(a4.z), __hmul2(decode_fp4x2(b4.z), scale_h2), acc0);
    acc1 = __hfma2(decode_fp4x2(a4.w), __hmul2(decode_fp4x2(b4.w), scale_h2), acc1);
    return __hadd2(acc0, acc1);
}

// Consumers keep the same control flow and branching cost as v4 but always take the
// global-memory path (b_in_smem = false, sfb_in_smem = false). This isolates the
// overhead of the branch/logic while guaranteeing correctness.
__device__ __forceinline__ float compute_dual_chain(
    const uint8_t* row_a,
    const uint8_t* batch_b,
    const uint8_t* row_sfa,
    const uint8_t* batch_sfb,
    const uint8_t* sh_b,
    const uint8_t* sh_sfb,
    const uint32_t* b_ready,
    const uint32_t* sfb_ready,
    int K_sf,
    int tid_in_row
) {
    float acc0 = 0.0f, acc1 = 0.0f;
    const int CONSUMER_THREADS = BLOCK_SIZE - PRODUCER_THREADS;
    const int THREADS_PER_ROW = CONSUMER_THREADS / ROWS_PER_BLOCK;
    const int STRIDE = THREADS_PER_ROW * 2;

#pragma unroll 2
    for (int sf_base = tid_in_row * 2; sf_base < K_sf; sf_base += STRIDE) {
        // Chain 0
        if (sf_base < K_sf) {
            bool sfb_ready_flag = sfb_ready[sf_base >> 5] & (1u << (sf_base & 31));
            bool sfb_in_smem = sfb_ready_flag;  // keep branch structure
            float sfa_val = decode_fp8(static_cast<int8_t>(__ldg(&row_sfa[sf_base])));
            float sfb_val = sfb_in_smem ?
                           decode_fp8(static_cast<int8_t>(sh_sfb[sf_base])) :
                           decode_fp8(static_cast<int8_t>(__ldg(&batch_sfb[sf_base])));
            // Force correctness: always use global value for math
            sfb_val = decode_fp8(static_cast<int8_t>(__ldg(&batch_sfb[sf_base])));

            float scale0 = sfa_val * sfb_val;
            __half2 scale_h2_0 = __halves2half2(__float2half(scale0), __float2half(scale0));

            int byte_base0 = sf_base << 3;
            uint32_t chunk_idx0 = byte_base0 >> 4;
            bool b_in_smem = b_ready[chunk_idx0 >> 5] & (1u << (chunk_idx0 & 31));

            uchar4 a4_0 = *reinterpret_cast<const uchar4*>(&row_a[byte_base0]);
            uchar4 b4_0 = b_in_smem ?
                         *reinterpret_cast<const uchar4*>(&sh_b[byte_base0]) :
                         *reinterpret_cast<const uchar4*>(&batch_b[byte_base0]);
            uchar4 a4_1 = *reinterpret_cast<const uchar4*>(&row_a[byte_base0 + 4]);
            uchar4 b4_1 = b_in_smem ?
                         *reinterpret_cast<const uchar4*>(&sh_b[byte_base0 + 4]) :
                         *reinterpret_cast<const uchar4*>(&batch_b[byte_base0 + 4]);
            // Force correctness: recompute from global
            b4_0 = *reinterpret_cast<const uchar4*>(&batch_b[byte_base0]);
            b4_1 = *reinterpret_cast<const uchar4*>(&batch_b[byte_base0 + 4]);

            __half2 res0_0 = dot_scaled_4bytes(a4_0, b4_0, scale_h2_0);
            __half2 res0_1 = dot_scaled_4bytes(a4_1, b4_1, scale_h2_0);

            float2 f0_0 = __half22float2(res0_0);
            float2 f0_1 = __half22float2(res0_1);
            acc0 += f0_0.x + f0_0.y + f0_1.x + f0_1.y;
        }

        // Chain 1
        int sf_next = sf_base + 1;
        if (sf_next < K_sf) {
            bool sfb_ready_flag = sfb_ready[sf_next >> 5] & (1u << (sf_next & 31));
            bool sfb_in_smem = sfb_ready_flag;
            float sfa_val = decode_fp8(static_cast<int8_t>(__ldg(&row_sfa[sf_next])));
            float sfb_val = sfb_in_smem ?
                           decode_fp8(static_cast<int8_t>(sh_sfb[sf_next])) :
                           decode_fp8(static_cast<int8_t>(__ldg(&batch_sfb[sf_next])));
            sfb_val = decode_fp8(static_cast<int8_t>(__ldg(&batch_sfb[sf_next])));

            float scale1 = sfa_val * sfb_val;
            __half2 scale_h2_1 = __halves2half2(__float2half(scale1), __float2half(scale1));

            int byte_base1 = sf_next << 3;
            uint32_t chunk_idx1 = byte_base1 >> 4;
            bool b_in_smem = b_ready[chunk_idx1 >> 5] & (1u << (chunk_idx1 & 31));

            uchar4 a4_2 = *reinterpret_cast<const uchar4*>(&row_a[byte_base1]);
            uchar4 b4_2 = b_in_smem ?
                         *reinterpret_cast<const uchar4*>(&sh_b[byte_base1]) :
                         *reinterpret_cast<const uchar4*>(&batch_b[byte_base1]);
            uchar4 a4_3 = *reinterpret_cast<const uchar4*>(&row_a[byte_base1 + 4]);
            uchar4 b4_3 = b_in_smem ?
                         *reinterpret_cast<const uchar4*>(&sh_b[byte_base1 + 4]) :
                         *reinterpret_cast<const uchar4*>(&batch_b[byte_base1 + 4]);
            b4_2 = *reinterpret_cast<const uchar4*>(&batch_b[byte_base1]);
            b4_3 = *reinterpret_cast<const uchar4*>(&batch_b[byte_base1 + 4]);

            __half2 res1_0 = dot_scaled_4bytes(a4_2, b4_2, scale_h2_1);
            __half2 res1_1 = dot_scaled_4bytes(a4_3, b4_3, scale_h2_1);

            float2 f1_0 = __half22float2(res1_0);
            float2 f1_1 = __half22float2(res1_1);
            acc1 += f1_0.x + f1_0.y + f1_1.x + f1_1.y;
        }
    }

    return acc0 + acc1;
}

// Producer still executes, but consumers ignore the staged data; this preserves
// the same instruction overhead and launch geometry for apples-to-apples timing.
__device__ __forceinline__ void load_b_to_smem(
    int tid,
    const uint8_t* batch_b,
    const uint8_t* batch_sfb,
    uint8_t* sh_b,
    uint8_t* sh_sfb,
    uint32_t* b_ready,
    uint32_t* sfb_ready,
    int K_half,
    int K_sf
) {
    for (int i = tid * 16; i < K_half; i += PRODUCER_THREADS * 16) {
        *reinterpret_cast<uint4*>(&sh_b[i]) =
            *reinterpret_cast<const uint4*>(&batch_b[i]);
        __threadfence_block();
        uint32_t chunk_idx = i >> 4;
        atomicOr(&b_ready[chunk_idx >> 5], (1u << (chunk_idx & 31)));
    }

    for (int i = tid * 16; i < K_sf; i += PRODUCER_THREADS * 16) {
        *reinterpret_cast<uint4*>(&sh_sfb[i]) =
            *reinterpret_cast<const uint4*>(&batch_sfb[i]);
        __threadfence_block();
        for (int j = 0; j < 16 && (i + j) < K_sf; ++j) {
            uint32_t idx = i + j;
            atomicOr(&sfb_ready[idx >> 5], (1u << (idx & 31)));
        }
    }
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
    int m_base = blockIdx.x * ROWS_PER_BLOCK;
    int l = blockIdx.y;
    int tid = threadIdx.x;

    if (m_base >= M || l >= L) return;

    const int consumer_tid = tid - PRODUCER_THREADS;
    const int CONSUMER_THREADS = BLOCK_SIZE - PRODUCER_THREADS;  // 96
    const int THREADS_PER_ROW = CONSUMER_THREADS / ROWS_PER_BLOCK; // 32
    int row_in_block = consumer_tid / THREADS_PER_ROW;
    int tid_in_row = consumer_tid % THREADS_PER_ROW;

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

    int m = m_base + row_in_block;
    if (tid >= PRODUCER_THREADS && m >= M) return;

    const uint8_t* row_a = batch_a + static_cast<size_t>(m) * K_half;
    const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m) * K_sf;

    __shared__ uint8_t sh_b[MAX_K_HALF];
    __shared__ uint8_t sh_sfb[MAX_K_SF];
    __shared__ uint32_t b_ready[16];
    __shared__ uint32_t sfb_ready[32];

    if (tid < 16) {
        b_ready[tid] = 0;
        sfb_ready[tid] = 0;
    }
    if (tid >= 16 && tid < 32) {
        sfb_ready[tid] = 0;
    }

    if (tid < PRODUCER_THREADS) {
        load_b_to_smem(tid, batch_b, batch_sfb, sh_b, sh_sfb, b_ready, sfb_ready, K_half, K_sf);
    }

    float acc = 0.0f;
    if (tid >= PRODUCER_THREADS) {
        acc = compute_dual_chain(
            row_a,
            batch_b,
            row_sfa,
            batch_sfb,
            sh_b,
            sh_sfb,
            b_ready,
            sfb_ready,
            K_sf,
            tid_in_row
        );
    }

    if (tid >= PRODUCER_THREADS) {
        const int lane = tid_in_row;
        float row_sum = acc;

        row_sum += __shfl_down_sync(0xffffffff, row_sum, 16);
        row_sum += __shfl_down_sync(0xffffffff, row_sum, 8);
        row_sum += __shfl_down_sync(0xffffffff, row_sum, 4);
        row_sum += __shfl_down_sync(0xffffffff, row_sum, 2);
        row_sum += __shfl_down_sync(0xffffffff, row_sum, 1);

        if (lane == 0) {
            size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
            c[c_idx] = __float2half(row_sum);
        }
    }
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    int grid_m = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(grid_m, L);
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
'''

module = load_inline(
    name='batched_scaled_gemv_test',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batched_scaled_gemv_cuda'],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-std=c++17',
        '-gencode=arch=compute_100a,code=sm_100a',
        '-maxrregcount=64'
    ],
    with_cuda=True,
    verbose=False
)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    return module.batched_scaled_gemv_cuda(
        a,
        b,
        sfa_ref,
        sfb_ref,
        c
    )
