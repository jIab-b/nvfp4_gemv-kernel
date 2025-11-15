import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

USE_PTX_MMA = 0  # 0 = Manual, 1 = PTX matmul only

cpp_source = """
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#if !USE_PTX_MMA
#include <cuda_fp4.h>

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
#else
#include <cuda/ptx>
using namespace cuda::ptx;

constexpr int BLOCK_K = 64;

constexpr uint32_t encode_mxf4nvf4_idesc(int m, int n, int k, bool scale_is_ue8m0) {
    constexpr uint32_t sparsity = 0u;
    constexpr uint32_t scale_b_id = 0u;
    constexpr uint32_t scale_a_id = 0u;
    constexpr uint32_t atype = 1u;
    constexpr uint32_t btype = 1u;
    constexpr uint32_t negate_a = 0u;
    constexpr uint32_t negate_b = 0u;
    constexpr uint32_t transpose_a = 0u;
    constexpr uint32_t transpose_b = 0u;
    const uint32_t n_field = static_cast<uint32_t>(n >> 3);
    const uint32_t m_field = static_cast<uint32_t>(m >> 7);
    const uint32_t scale_type = scale_is_ue8m0 ? 1u : 0u;
    const uint32_t k_field = (k == 96) ? 1u : 0u;

    uint32_t desc = 0;
    desc |= (sparsity & 0x1u) << 2;
    desc |= (scale_b_id & 0x3u) << 4;
    desc |= (atype & 0x7u) << 7;
    desc |= (btype & 0x3u) << 10;
    desc |= (negate_a & 0x1u) << 13;
    desc |= (negate_b & 0x1u) << 14;
    desc |= (transpose_a & 0x1u) << 15;
    desc |= (transpose_b & 0x1u) << 16;
    desc |= (n_field & 0x3Fu) << 17;
    desc |= (scale_type & 0x1u) << 23;
    desc |= (m_field & 0x3u) << 27;
    desc |= (scale_a_id & 0x3u) << 29;
    desc |= (k_field & 0x1u) << 31;
    return desc;
}

__device__ __forceinline__ uint64_t pack_smem_desc(uint32_t base, uint32_t ldm_bytes, uint32_t stride_bytes) {
    uint64_t desc = 0;
    uint64_t ldm = static_cast<uint64_t>(ldm_bytes & 0xFFFFull);
    uint64_t stride = static_cast<uint64_t>((stride_bytes >> 4) & 0x3FFFull);
    uint64_t base_field = static_cast<uint64_t>((base >> 4) & 0xFFFFull);
    desc |= ldm;
    desc |= (stride << 16);
    desc |= (base_field << 30);
    return desc;
}

__device__ __forceinline__ void ld_tmem(float (&frag)[8], uint32_t src) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 { %0, %1, %2, %3, %4, %5, %6, %7 }, [%8];\\n"
        "tcgen05.wait::ld.sync.aligned;\\n"
        : "=f"(frag[0]), "=f"(frag[1]), "=f"(frag[2]), "=f"(frag[3]),
          "=f"(frag[4]), "=f"(frag[5]), "=f"(frag[6]), "=f"(frag[7])
        : "r"(src)
        : "memory");
}
#endif

__global__ void gemv_nvfp4_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L,
    int N_rows
) {
    int m = blockIdx.x;
    int l = blockIdx.y;
    int tid = threadIdx.x;

    if (m >= M || l >= L) return;

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

    const uint8_t* row_a = batch_a + static_cast<size_t>(m) * K_half;
    const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m) * K_sf;

    __shared__ float smem_acc[32];

    float acc = 0.0f;

#if USE_PTX_MMA
    __shared__ uint8_t smem_a[128 * BLOCK_K / 2];
    __shared__ uint8_t smem_b[8 * BLOCK_K / 2];
    __shared__ uint8_t smem_scale_a[128 * (BLOCK_K / 16)];
    __shared__ uint8_t smem_scale_b[8 * (BLOCK_K / 16)];
    __shared__ uint32_t shared_handles[3];
    __shared__ unsigned long long mbar_state;

    const int tiles_k = K / BLOCK_K;
    uint32_t d_tmem_cols = ((128 * 8 + 31) / 32) * 32;
    uint32_t scale_cols = ((BLOCK_K / 16) + 31) / 32 * 32;

    if (tid == 0) {
        tcgen05_alloc(cta_group_1, &shared_handles[0], d_tmem_cols);
        tcgen05_alloc(cta_group_1, &shared_handles[1], scale_cols);
        tcgen05_alloc(cta_group_1, &shared_handles[2], scale_cols);
    }
    __syncthreads();

    uint32_t d_tmem = shared_handles[0];
    uint32_t scaleA_tmem = shared_handles[1];
    uint32_t scaleB_tmem = shared_handles[2];

    uint32_t smem_base_a = __cvta_generic_to_shared(smem_a);
    uint32_t smem_base_b = __cvta_generic_to_shared(smem_b);
    uint32_t smem_base_sfa = __cvta_generic_to_shared(smem_scale_a);
    uint32_t smem_base_sfb = __cvta_generic_to_shared(smem_scale_b);
    uint64_t mbar_addr = static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_state));

    uint32_t idesc = encode_mxf4nvf4_idesc(128, 8, BLOCK_K, false);

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" : : "l"(mbar_addr), "r"(1) : "memory");
    }
    __syncthreads();

    for (uint32_t k_tile = 0; k_tile < tiles_k; ++k_tile) {
        for (int idx = tid; idx < 128 * (BLOCK_K / 2); idx += 32) {
            int row = idx / (BLOCK_K / 2);
            int col = idx % (BLOCK_K / 2);
            if (row == m) {
                smem_a[row * (BLOCK_K/2) + col] = row_a[k_tile * (BLOCK_K/2) + col];
            } else {
                smem_a[row * (BLOCK_K/2) + col] = 0;
            }
        }
        for (int idx = tid; idx < 8 * (BLOCK_K / 2); idx += 32) {
            int row = idx / (BLOCK_K / 2);
            int col = idx % (BLOCK_K / 2);
            smem_b[idx] = batch_b[k_tile * (BLOCK_K/2) + col];
        }
        for (int idx = tid; idx < 128 * (BLOCK_K / 16); idx += 32) {
            int row = idx / (BLOCK_K / 16);
            if (row == m) {
                smem_scale_a[idx] = row_sfa[k_tile * (BLOCK_K/16) + (idx % (BLOCK_K/16))];
            } else {
                smem_scale_a[idx] = 0;
            }
        }
        for (int idx = tid; idx < 8 * (BLOCK_K / 16); idx += 32) {
            smem_scale_b[idx] = batch_sfb[k_tile * (BLOCK_K/16) + (idx % (BLOCK_K/16))];
        }
        __syncthreads();

        uint64_t a_desc = pack_smem_desc(smem_base_a, BLOCK_K / 2, BLOCK_K / 2);
        uint64_t b_desc = pack_smem_desc(smem_base_b, BLOCK_K / 2, BLOCK_K / 2);
        uint64_t scaleA_desc = pack_smem_desc(smem_base_sfa, BLOCK_K / 16, BLOCK_K / 16);
        uint64_t scaleB_desc = pack_smem_desc(smem_base_sfb, BLOCK_K / 16, BLOCK_K / 16);

        tcgen05_cp_64x128b_warpx2_01_23(cta_group_1, scaleA_tmem, scaleA_desc);
        tcgen05_cp_64x128b_warpx2_01_23(cta_group_1, scaleB_tmem, scaleB_desc);
        __syncthreads();

        asm volatile(
            "{ .reg .pred p; setp.ne.b32 p, %6, %6;\\n"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\\n}"
            : : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
                "r"(scaleA_tmem), "r"(scaleB_tmem), "r"(0) : "memory");
        __syncthreads();
    }

    if (tid == 0) {
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\\n" : : "l"(mbar_addr) : "memory");
    }

    while (true) {
        unsigned ready = 0;
        asm volatile(
            "{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\\n selp.b32 %0, 1, 0, p;\\n }\\n"
            : "=r"(ready) : "l"(mbar_addr) : "memory");
        if (ready) break;
    }

    asm volatile("tcgen05.fence::after_thread_sync;\\n" ::: "memory");
    __syncthreads();

    if (tid == 0) {
        float frag[8];
        ld_tmem(frag, d_tmem);
        acc = frag[m % 8];
        tcgen05_dealloc(cta_group_1, scaleB_tmem, scale_cols);
        tcgen05_dealloc(cta_group_1, scaleA_tmem, scale_cols);
        tcgen05_dealloc(cta_group_1, d_tmem, d_tmem_cols);
    }
#else
    for (int k_block = tid; k_block < K_sf; k_block += 32) {
        float scale_a = decode_fp8(static_cast<int8_t>(row_sfa[k_block]));
        float scale_b = decode_fp8(static_cast<int8_t>(batch_sfb[k_block]));

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
#endif

    smem_acc[tid] = acc;
    __syncthreads();

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
        a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr,
        M, K, L, N_rows
    );

    return c;
}
"""

compile_flags = [
    '-O3', '--use_fast_math', '-std=c++17',
    f'-DUSE_PTX_MMA={USE_PTX_MMA}',
    '-gencode=arch=compute_100a,code=sm_100a' if USE_PTX_MMA else '-gencode=arch=compute_100,code=sm_100'
]

module = load_inline(
    name='batched_scaled_gemv_modular',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batched_scaled_gemv_cuda'],
    extra_cuda_cflags=compile_flags,
    with_cuda=True,
    verbose=True
)

def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device
    a_i8 = a.to(device=device, copy=False).view(torch.int8)
    b_i8 = b.to(device=device, copy=False).view(torch.int8)
    sfa_i8 = sfa_ref.to(device=device, non_blocking=True).view(torch.int8)
    sfb_i8 = sfb_ref.to(device=device, non_blocking=True).view(torch.int8)
    return module.batched_scaled_gemv_cuda(a_i8, b_i8, sfa_i8, sfb_i8, c)
