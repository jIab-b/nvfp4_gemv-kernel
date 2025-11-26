import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


cpp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

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
#define BLOCK_SIZE 128
#define K_TILE 4096
#define M_TILE 64
#define SCALES_PER_TILE (K_TILE / 16)
#define BYTES_PER_TILE (K_TILE / 2)

__device__ __forceinline__ float half_raw_to_float(const __half_raw& raw) {
    return __half2float(__ushort_as_half(raw.x));
}

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
    return half_raw_to_float(raw);
}

// ============================================================================
// ======================== PARTIAL DOT PRODUCT KERNEL =======================
// ============================================================================
__global__ void gemv_partial_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    float* __restrict__ partials,
    int M, int K, int L, int N_rows, int num_k_tiles
) {
    int m_tile_idx = blockIdx.x;
    int k_idx = blockIdx.y;
    int l = blockIdx.z;
    int tid = threadIdx.x;

    if (l >= L) return;

// ============================================================================
// ===================== PER-CTA BASE POINTER SETUP ==========================
// ============================================================================
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

    int m_start = m_tile_idx * M_TILE;
    int m_end = min(m_start + M_TILE, M);
    int m_count = m_end - m_start;

    int k_start = k_idx * K_TILE;
    int k_end = min(k_start + K_TILE, K);
    int k_count = k_end - k_start;
    int sf_count = k_count / 16;
    int byte_count = k_count / 2;

    int base_byte = k_start / 2;
    int base_sf = k_start / 16;

    __shared__ float smem_acc[M_TILE][BLOCK_SIZE];
    __shared__ uint8_t sh_a[M_TILE][BYTES_PER_TILE];
    __shared__ uint8_t sh_b[BYTES_PER_TILE];
    __shared__ uint8_t sh_sfa[M_TILE][SCALES_PER_TILE];
    __shared__ uint8_t sh_sfb[SCALES_PER_TILE];

    float acc[M_TILE];
    for (int mi = 0; mi < M_TILE; ++mi) acc[mi] = 0.0f;

// ============================================================================
// ===================== MEMORY LOAD AND MAIN COMPUTE ========================
// ============================================================================
#define ASYNC_COPY_16(dst, src) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src))

#define ASYNC_COPY_4(dst, src) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(src))

    uint32_t sh_b_base = __cvta_generic_to_shared(&sh_b[0]);
    uint32_t sh_sfb_base = __cvta_generic_to_shared(&sh_sfb[0]);

    for (int i = tid * 16; i < byte_count; i += BLOCK_SIZE * 16) {
        ASYNC_COPY_16(sh_b_base + i, batch_b + base_byte + i);
    }
    for (int i = tid * 4; i < sf_count; i += BLOCK_SIZE * 4) {
        ASYNC_COPY_4(sh_sfb_base + i, batch_sfb + base_sf + i);
    }

    for (int mi = 0; mi < m_count; ++mi) {
        int m = m_start + mi;
        const uint8_t* row_a = batch_a + static_cast<size_t>(m) * K_half;
        const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m) * K_sf;

        uint32_t sh_a_base = __cvta_generic_to_shared(&sh_a[mi][0]);
        uint32_t sh_sfa_base = __cvta_generic_to_shared(&sh_sfa[mi][0]);

        for (int i = tid * 16; i < byte_count; i += BLOCK_SIZE * 16) {
            ASYNC_COPY_16(sh_a_base + i, row_a + base_byte + i);
        }
        for (int i = tid * 4; i < sf_count; i += BLOCK_SIZE * 4) {
            ASYNC_COPY_4(sh_sfa_base + i, row_sfa + base_sf + i);
        }
    }

    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    for (int sf = tid; sf < sf_count; sf += BLOCK_SIZE) {
        float scale_b = decode_fp8(static_cast<int8_t>(sh_sfb[sf]));
        int byte_base = sf * 8;

#pragma unroll
        for (int bb = 0; bb < 8; ++bb) {
            __half2 b2 = decode_fp4x2(sh_b[byte_base + bb]);

            for (int mi = 0; mi < m_count; ++mi) {
                float scale = decode_fp8(static_cast<int8_t>(sh_sfa[mi][sf])) * scale_b;
                __half scale_h = __float2half(scale);
                __half2 scale_h2 = __halves2half2(scale_h, scale_h);

                __half2 a2 = decode_fp4x2(sh_a[mi][byte_base + bb]);
                __half2 prod = __hmul2(__hmul2(a2, b2), scale_h2);
                float2 f = __half22float2(prod);
                acc[mi] += f.x + f.y;
            }
        }
    }

#undef ASYNC_COPY_16
#undef ASYNC_COPY_4

// ============================================================================
// ================== ACCUMULATION AND BLOCK REDUCTION =======================
// ============================================================================
    for (int mi = 0; mi < m_count; ++mi) {
        smem_acc[mi][tid] = acc[mi];
    }
    __syncthreads();

    for (int mi = 0; mi < m_count; ++mi) {
        float warp_sum = acc[mi];
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }

        int warp_id = tid >> 5;
        int lane = tid & 31;
        if (lane == 0) smem_acc[mi][warp_id] = warp_sum;
    }

    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;

    if (warp_id == 0) {
        for (int mi = 0; mi < m_count; ++mi) {
            float block_sum = (lane < (blockDim.x >> 5)) ? smem_acc[mi][lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
            }
            if (lane == 0) {
                int m = m_start + mi;
                size_t partial_idx = static_cast<size_t>(m) * num_k_tiles * L + 
                                     static_cast<size_t>(k_idx) * L + 
                                     static_cast<size_t>(l);
                partials[partial_idx] = block_sum;
            }
        }
    }
}

// ============================================================================
// ======================== REDUCTION KERNEL =================================
// ============================================================================
__global__ void gemv_reduce_kernel(
    const float* __restrict__ partials,
    half* __restrict__ c,
    int M, int L, int num_k_tiles
) {
    int m = blockIdx.x;
    int l = blockIdx.y;

    if (m >= M || l >= L) return;

    float sum = 0.0f;
    for (int k_idx = 0; k_idx < num_k_tiles; ++k_idx) {
        size_t partial_idx = static_cast<size_t>(m) * num_k_tiles * L + 
                             static_cast<size_t>(k_idx) * L + 
                             static_cast<size_t>(l);
        sum += partials[partial_idx];
    }

    size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
    c[c_idx] = __float2half(sum);
}

// ============================================================================
// ======================== HOST ENTRY POINT =================================
// ============================================================================
torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    int num_k_tiles = (K + K_TILE - 1) / K_TILE;
    int num_m_tiles = (M + M_TILE - 1) / M_TILE;

    torch::Tensor partials = torch::zeros({M, num_k_tiles, L}, a.options().dtype(torch::kFloat32));

    dim3 grid1(num_m_tiles, num_k_tiles, L);
    dim3 block1(BLOCK_SIZE);

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());
    auto* partials_ptr = reinterpret_cast<float*>(partials.data_ptr());

    gemv_partial_kernel<<<grid1, block1>>>(
        a_ptr, b_ptr, sfa_ptr, sfb_ptr, partials_ptr,
        M, K, L, N_rows, num_k_tiles
    );

    dim3 grid2(M, L);
    dim3 block2(1);

    gemv_reduce_kernel<<<grid2, block2>>>(
        partials_ptr, c_ptr, M, L, num_k_tiles
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
    return module.batched_scaled_gemv_cuda(a, b, sfa_ref, sfb_ref, c)
