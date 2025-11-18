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

enum class MmaKind {

  Tf32 = 0,

  F16 = 1,

  I8 = 2,

  F8F6F4 = 3,

  Mxf8F6F4 = 4,

  Mxf4 = 5,

  Mxf4Nvf4 = 6

};

/*

make_shared_desc: Creates the 64-bit shared memory descriptor for matrices in SMEM.

It encodes address, leading dimension byte offset (LBO), stride byte offset (SBO), mode, and swizzle.

Used in tcgen05.mma/ld/st to describe the source/dest in shared memory.

*/

__device__ uint64_t make_shared_desc(void* smem_ptr, uint32_t leading_byte, uint32_t stride_byte, uint32_t swizzle = 0, bool k_major = true) {

    uint64_t desc = 0;

    uintptr_t addr = reinterpret_cast<uintptr_t>(smem_ptr);

    desc |= (addr >> 4) & 0x3FFFULL;  // Bits 0-13: (addr >> 4) & 0x3FFF

    desc |= (static_cast<uint64_t>(leading_byte & 0xFFFFF) << 14);  // Bits 14-33: LBO & 0xFFFFF

    desc |= (static_cast<uint64_t>(stride_byte & 0xFFFFF) << 34);  // Bits 34-53: SBO & 0xFFFFF

    desc |= (static_cast<uint64_t>(!k_major) << 54);  // Bit 54: 1 if not K-major (M/N-major)

    desc |= (static_cast<uint64_t>(swizzle & 0x7) << 55);  // Bits 55-57: swizzle & 0x7

    return desc;

}

/*

make_instr_desc: Creates the 32-bit instruction descriptor for tcgen05.mma.

Specifies kind, log shapes (log2(size/8) for M,N,K), transposes, scale vec size log.

For .kind::mxf4nvf4 as per Table 44.

*/

__device__ MmaConfig get_default_mma_config() {

    // Default for our task: FP4 e2m1 data, FP8 e4m3 scales, FP32 internal accum (output to FP16 via store)

    return {MmaKind::Mxf4Nvf4, 128, 32, 128, false, false, 1};  // 128(M)x32(N)x128(K), no trans, 2X scale

}

struct SharedDescParams {

    void* smem_ptr;

    uint32_t leading_byte;

    uint32_t stride_byte;

    uint32_t swizzle;

    bool k_major;

};

__device__ SharedDescParams get_default_shared_params(void* ptr, uint32_t leading, uint32_t stride) {

    // Default swizzle=0, k_major=true for our task's layouts

    return {ptr, leading, stride, 0, true};

}

__global__ void gemv_nvfp4_tc_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L,
    int N_rows
) {
    // Block setup: process BM rows per block, tile over K
    #define THREADS_PER_CTA 128
    #define ROWS_PER_CTA 128
    #define K_TILE 128
    #define MIN_N 32

    int m_block = blockIdx.x;
    int l = blockIdx.y;
    int tid = threadIdx.x;

    if (m_block * ROWS_PER_CTA >= M || l >= L) return;

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

    const uint8_t* row_a = batch_a + static_cast<size_t>(m_block * ROWS_PER_CTA) * K_half;
    const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m_block * ROWS_PER_CTA) * K_sf;

    // Allocate TMEM for accumulator (BM x 1, but padded to MIN_N)
    // Use CuTe to allocate
    // auto tmem_alloc = tcgen05.alloc(BM, MIN_N); // Removed CuTe

    // Clear accumulator
    // Inline PTX for tcgen05.clear or equivalent
    // (Assume clear via zero load or PTX)

    // Loop over K tiles
    for (int k_block = 0; k_block < (K + K_TILE - 1) / K_TILE; ++k_block) {
        int k_start = k_block * K_TILE;
        int k_end = min(k_start + K_TILE, K);

        // Load vector stretch to SMEM (persist for all M in block)
        __shared__ float vector_smem[K_TILE];
        __shared__ float vector_scale_smem[K_TILE / 16];

        // Collaborative load
        for (int i = tid; i < k_end - k_start; i += THREADS_PER_CTA) {
            int k = k_start + i;
            int byte_idx = k / 2;
            uint8_t b_byte = batch_b[byte_idx];
            uint8_t b_nib = (k & 1) ? (b_byte >> 4) : (b_byte & 0xF);
            vector_smem[i] = decode_fp4(b_nib);
        }
        for (int i = tid; i < (k_end - k_start) / 16; i += THREADS_PER_CTA) {
            int k_sf = k_start / 16 + i;
            vector_scale_smem[i] = decode_fp8(static_cast<int8_t>(batch_sfb[k_sf]));
        }
        __syncthreads();

        // Load to TMEM for vector, replicate to MIN_N columns with stride 0 descriptor
        uint64_t vector_desc = 0;  // Construct descriptor
        // Set address, shape K_TILE x MIN_N, stride for N=0 to replicate
        // (Bit packing as per ISA)
        // ... (set bits for address, leading dim, stride=0 for N)

        // Load vector to TMEM using tcgen05.ld
        // Inline PTX
        asm volatile("tcgen05.ld.cta_group::1.kind::e2m1 taddr, desc;" : : "r"(vector_desc), "r"(vector_desc) );  // Load with replication

        // Load scales to TMEM, replicate
        // Similar for scale TMEM

        // Load matrix tile for this m_block
        __shared__ float matrix_smem[ROWS_PER_CTA * K_TILE];
        __shared__ float matrix_scale_smem[ROWS_PER_CTA * (K_TILE / 16)];

        // Collaborative load for matrix and scales
        for (int i = tid; i < ROWS_PER_CTA * (k_end - k_start); i += THREADS_PER_CTA) {
            int m_local = i / (k_end - k_start);
            int k_local = i % (k_end - k_start);
            int k = k_start + k_local;
            int byte_idx = k / 2;
            uint8_t a_byte = row_a[m_local * K_half + byte_idx];
            uint8_t a_nib = (k & 1) ? (a_byte >> 4) : (a_byte & 0xF);
            matrix_smem[m_local * K_TILE + k_local] = decode_fp4(a_nib);
        }
        for (int i = tid; i < ROWS_PER_CTA * (k_end - k_start) / 16; i += THREADS_PER_CTA) {
            int m_local = i / ((k_end - k_start) / 16);
            int k_sf_local = i % ((k_end - k_start) / 16);
            int k_sf = k_start / 16 + k_sf_local;
            matrix_scale_smem[m_local * (K_TILE / 16) + k_sf_local] = decode_fp8(static_cast<int8_t>(row_sfa[m_local * K_sf + k_sf]));
        }
        __syncthreads();

        // Load to TMEM for matrix
        uint64_t matrix_desc = 0;  // Construct descriptor for ROWS_PER_CTA x K_TILE

        // Load to TMEM
        asm volatile("tcgen05.ld.cta_group::1.kind::e2m1 taddr, desc;" : : "r"(matrix_desc), "r"(matrix_desc) );

        // Load matrix scales to TMEM

        // Execute MMA
        uint32_t instr_desc = 0;  // Construct instruction descriptor for mxf4nvf4, scale_vec::2X, shapes
        
        // Set bits for kind, shapes, etc.

        asm volatile("tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec_size::2X [%0], %1, %2, %3, %4;"
            : : "r"(vector_desc), "r"(matrix_desc), "r"(instr_desc), "r"(params) );

        __syncthreads();  // Wait for MMA
    }

    // Store from TMEM to c (take first column since replicated)
    // Use tcgen05.cp to registers, then store

    // For each m in 0 to ROWS_PER_CTA
    for (int i = tid; i < ROWS_PER_CTA; i += THREADS_PER_CTA) {
        int m = m_block * ROWS_PER_CTA + i;
        if (m < M) {
            size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
            c[c_idx] = /* extract from TMEM */ ;
        }
    }
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);
    
    dim3 grid((M + ROWS_PER_CTA - 1) / ROWS_PER_CTA, L);
    dim3 block(THREADS_PER_CTA);

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    gemv_nvfp4_tc_kernel<<<grid, block>>>(
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
    name='batched_scaled_gemv_tc',
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

THREADS_PER_CTA = 128

ROWS_PER_CTA = 128

K_TILE = 128

MIN_N = 32

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
