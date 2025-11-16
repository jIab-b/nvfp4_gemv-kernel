import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_source = """
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
    unsigned dst_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;"
        :
        : "r"(dst_addr), "l"(src));
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    __shared__ unsigned tmem_debug_addr;
#endif
    extern __shared__ uint8_t shared_bytes[];
    uint8_t* smem_a = shared_bytes;
    uint8_t* smem_b = shared_bytes + K_half;

    const uint8_t* row_a_data = row_a;
    const uint8_t* batch_b_data = batch_b;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    for (int offset = tid * 16; offset < K_half; offset += blockDim.x * 16) {
        cp_async_16(smem_a + offset, row_a + offset);
        cp_async_16(smem_b + offset, batch_b + offset);
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    row_a_data = smem_a;
    batch_b_data = smem_b;

#else
    for (int offset = tid * 16; offset < K_half; offset += blockDim.x * 16) {
        int remaining = K_half - offset;
        int bytes = remaining >= 16 ? 16 : remaining;
        for (int i = 0; i < bytes; ++i) {
            smem_a[offset + i] = row_a[offset + i];
            smem_b[offset + i] = batch_b[offset + i];
        }
    }
    __syncthreads();
    row_a_data = smem_a;
    batch_b_data = smem_b;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // Stage B: TMEM allocation scaffold (only CTA (0,0) runs to avoid perf impact).
    if (m == 0 && l == 0) {
        unsigned smem_ptr = 0;
        if (tid == 0) {
            smem_ptr = __cvta_generic_to_shared(&tmem_debug_addr);
        }
        smem_ptr = __shfl_sync(0xffffffff, smem_ptr, 0);
        if (tid < 32) {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                :
                : "r"(smem_ptr));
        }
        __syncthreads();
        unsigned tmem_base = tmem_debug_addr;
        if (tid < 32) {
            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
                :
                : "r"(tmem_base));
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        }
        __syncthreads();
    }
#endif

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
            uint8_t a_byte = row_a_data[byte_idx];
            uint8_t b_byte = batch_b_data[byte_idx];

            uint8_t a_nib = (k & 1) ? (a_byte >> 4) : (a_byte & 0xF);
            uint8_t b_nib = (k & 1) ? (b_byte >> 4) : (b_byte & 0xF);

            float a_val = decode_fp4(a_nib) * scale_a;
            float b_val = decode_fp4(b_nib) * scale_b;


            asm volatile("fma.rn.f32 %0, %1, %2, %0;"
                         : "+f"(acc)
                         : "f"(a_val), "f"(b_val));
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
    int K_half = K / 2;
    int L = a.size(2);
    int N_rows = b.size(0);
    
    dim3 grid(M, L);
    dim3 block(32);
    size_t shared_bytes = static_cast<size_t>(K_half) * 2;

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    gemv_nvfp4_kernel<<<grid, block, shared_bytes>>>(
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
