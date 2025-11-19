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

struct MmaConfig {

    MmaKind kind;

    int m_size;

    int n_size;

    int k_size;

    bool trans_a;

    bool trans_b;

    int scale_vec_size_log;  // 0:1X, 1:2X, 2:4X

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

__device__ uint32_t device_ctz(uint32_t x) {
    uint32_t result = 0;
    while ((x & 1) == 0 && result < 32) {
        x >>= 1;
        result++;
    }
    return result;
}

__device__ uint32_t make_instr_desc(const MmaConfig& config) {

    uint32_t desc = 0;

    desc |= (static_cast<uint32_t>(config.kind) & 0xF);  // Bits 0-3: kind

    uint32_t m_log = device_ctz(config.m_size / 8);  // log2(m/8)

    desc |= (m_log & 0xF) << 4;  // Bits 4-7

    uint32_t n_log = device_ctz(config.n_size / 8);

    desc |= (n_log & 0xF) << 8;  // Bits 8-11

    uint32_t k_log = device_ctz(config.k_size / 8);

    desc |= (k_log & 0x7) << 12;  // Bits 12-14 (3 bits?)

    desc |= (config.trans_a ? 1U : 0) << 15;  // Bit 15

    desc |= (config.trans_b ? 1U : 0) << 16;  // Bit 16

    desc |= (static_cast<uint32_t>(config.scale_vec_size_log & 0x3) << 17);  // Bits 17-18? Assume 2 bits

    return desc;

}

struct SharedDescParams {

    void* smem_ptr;

    uint32_t leading_byte;

    uint32_t stride_byte;

    uint32_t swizzle;

    bool k_major;

};

__device__ SharedDescParams get_default_shared_params(void* smem_ptr, uint32_t leading_byte, uint32_t stride_byte, uint32_t swizzle = 0, bool k_major = true) {

    SharedDescParams params;

    params.smem_ptr = smem_ptr;

    params.leading_byte = leading_byte;

    params.stride_byte = stride_byte;

    params.swizzle = swizzle;

    params.k_major = k_major;

    return params;

}

__device__ const MmaConfig mma_config = {
    MmaKind::Mxf4Nvf4,  // kind
    128,                // m_size (ROWS_PER_CTA)
    32,                 // n_size (MIN_N)
    128,                // k_size (K_TILE)
    false,              // trans_a
    false,              // trans_b
    1                   // scale_vec_size_log (2X scaling)
};

/*

make_zero_mask_desc: Creates the 64-bit zero-column mask descriptor.

Specifies non-zero mask (0-255) and shift amount (0-31) to generate mask for zeroing B columns.

Used optionally in tcgen05.mma to force zeros in B columns.

*/

struct ZeroMaskConfig {

    uint32_t non_zero_mask;  // 0-255

    uint32_t shift;          // 0-31

};

__device__ uint64_t make_zero_mask_desc(const ZeroMaskConfig& config) {

    uint64_t desc = 0;

    desc |= (static_cast<uint64_t>(config.non_zero_mask & 0xFF) << 0);   // Bits 0-7: mask

    desc |= (static_cast<uint64_t>(config.shift & 0x1F) << 8);           // Bits 8-12: shift

    return desc;

}

__global__ void __cluster_dims__(4, 1, 1) gemv_nvfp4_tc_kernel(
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
    int lane_id = tid & 0x1F;
    int warp_id = tid >> 5;

    int rows_this_tile = min(ROWS_PER_CTA, M - m_block * ROWS_PER_CTA);
    if (rows_this_tile <= 0 || l >= L) return;

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

    // Shared memory for packed data and scales
    __shared__ uint8_t vector_smem[K_TILE / 2];
    __shared__ int8_t vector_scale_smem[K_TILE / 16];
    __shared__ uint8_t matrix_smem[ROWS_PER_CTA * (K_TILE / 2)];
    __shared__ int8_t matrix_scale_smem[ROWS_PER_CTA * (K_TILE / 16)];
    __shared__ float accum_shared[ROWS_PER_CTA * MIN_N];

    // TMEM allocation (nCols=32 for 128x32 e2m1, 32 columns minimum)
    __shared__ uint32_t accum_taddr_smem;
    __shared__ uint32_t vector_taddr_smem;
    __shared__ uint32_t vector_scale_taddr_smem;
    __shared__ uint32_t matrix_taddr_smem;
    __shared__ uint32_t matrix_scale_taddr_smem;

    // Fix: Use correct tcgen05.alloc syntax
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" : : "r"((uint32_t)(uintptr_t)&accum_taddr_smem));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" : : "r"((uint32_t)(uintptr_t)&vector_taddr_smem));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" : : "r"((uint32_t)(uintptr_t)&vector_scale_taddr_smem));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" : : "r"((uint32_t)(uintptr_t)&matrix_taddr_smem));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" : : "r"((uint32_t)(uintptr_t)&matrix_scale_taddr_smem));

    uint32_t accum_taddr = accum_taddr_smem;
    uint32_t vector_taddr = vector_taddr_smem;
    uint32_t vector_scale_taddr = vector_scale_taddr_smem;
    uint32_t matrix_taddr = matrix_taddr_smem;
    uint32_t matrix_scale_taddr = matrix_scale_taddr_smem;

    // Zero initialize accumulators in TMEM
    __shared__ float zero_smem[32];  // 32 floats for zero initialization
    for (int i = tid; i < 32; i += THREADS_PER_CTA) {
        zero_smem[i] = 0.0f;
    }
    __syncthreads();

    // Create descriptor for zero initialization
    SharedDescParams zero_params = get_default_shared_params(zero_smem, 0, 0);
    uint64_t zero_desc = make_shared_desc(zero_params.smem_ptr, zero_params.leading_byte, zero_params.stride_byte, zero_params.swizzle, zero_params.k_major);

    // Fix: Use correct format for zero initialization - single format specifier
    asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" : : "r"(accum_taddr), "l"(zero_desc));
    asm volatile("tcgen05.wait::st.sync.aligned;");

    // Loop over K tiles
    for (int k_block = 0; k_block < (K + K_TILE - 1) / K_TILE; ++k_block) {
        int k_start = k_block * K_TILE;
        int tile_elems = min(K_TILE, K - k_start);
        if (tile_elems <= 0) continue;

        // Load vector packed
        for (int i = tid; i < (tile_elems + 1) / 2; i += THREADS_PER_CTA) {  // +1 for odd
            int byte_idx = k_start / 2 + i;
            if (byte_idx < batch_stride_b) vector_smem[i] = batch_b[byte_idx];
        }

        // Load vector scales
        for (int i = tid; i < (tile_elems + 15) / 16; i += THREADS_PER_CTA) {
            int sf_idx = k_start / 16 + i;
            if (sf_idx < K_sf) vector_scale_smem[i] = batch_sfb[sf_idx];
        }
        __syncthreads();

        // Vector desc with replication (leading=0, stride=0 for broadcast across rows)
        SharedDescParams vector_params = get_default_shared_params(vector_smem, 0, 0);
        uint64_t vector_desc = make_shared_desc(vector_params.smem_ptr, vector_params.leading_byte, vector_params.stride_byte, vector_params.swizzle, vector_params.k_major);

        SharedDescParams vscale_params = get_default_shared_params(vector_scale_smem, 0, 0);
        uint64_t vscale_desc = make_shared_desc(vscale_params.smem_ptr, vscale_params.leading_byte, vscale_params.stride_byte, vscale_params.swizzle, vscale_params.k_major);

        asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" : : "r"(vector_taddr), "l"(vector_desc));
        asm volatile("tcgen05.wait::st.sync.aligned;");
        asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" : : "r"(vector_scale_taddr), "l"(vscale_desc));
        asm volatile("tcgen05.wait::st.sync.aligned;");

        // Load matrix packed - each thread copies rows/bytes
        for (int row = tid; row < rows_this_tile; row += THREADS_PER_CTA) {
            const uint8_t* src = row_a + row * K_half + k_start / 2;
            uint8_t* dst = matrix_smem + row * (K_TILE / 2);
            for (int byte = 0; byte < (tile_elems + 1) / 2; ++byte) {
                dst[byte] = (k_start / 2 + byte < K_half) ? src[byte] : 0;
            }
        }

        for (int row = tid; row < rows_this_tile; row += THREADS_PER_CTA) {
            const uint8_t* src = row_sfa + row * K_sf + k_start / 16;
            int8_t* dst = matrix_scale_smem + row * (K_TILE / 16);
            for (int sf = 0; sf < (tile_elems + 15) / 16; ++sf) {
                dst[sf] = (k_start / 16 + sf < K_sf) ? src[sf] : 0;
            }
        }
        __syncthreads();

        // Matrix desc - leading=(K_TILE/2), stride=(K_TILE/2)
        SharedDescParams matrix_params = get_default_shared_params(matrix_smem, K_TILE / 2, K_TILE / 2);
        uint64_t matrix_desc = make_shared_desc(matrix_params.smem_ptr, matrix_params.leading_byte, matrix_params.stride_byte, matrix_params.swizzle, matrix_params.k_major);
        asm volatile("tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [%0], %1;" : : "r"(matrix_taddr), "l"(matrix_desc));
        asm volatile("tcgen05.wait::st.sync.aligned;");

        // Matrix scales desc (leading = K_TILE / 16, stride=1)
        SharedDescParams mscale_params = get_default_shared_params(matrix_scale_smem, K_TILE / 16, 1);
        uint64_t mscale_desc = make_shared_desc(mscale_params.smem_ptr, mscale_params.leading_byte, mscale_params.stride_byte, mscale_params.swizzle, mscale_params.k_major);
        // Fix: Use correct format for matrix scales (FP8) - single format specifier
        asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" : : "r"(matrix_scale_taddr), "l"(mscale_desc));
        asm volatile("tcgen05.wait::st.sync.aligned;");

        // Execute MMA
        uint32_t instr_desc = make_instr_desc(mma_config);
        uint32_t enable_input_d = (k_block > 0) ? 1 : 0;
        asm volatile("{\\n"
                     "  .reg .pred p;\\n"
                     "  setp.ne.b32 p, %6, 0;\\n"
                     "  tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], p;\\n"
                     "}\\n"
            : : "r"(accum_taddr), "l"(matrix_desc), "l"(vector_desc), "r"(instr_desc), "r"(matrix_scale_taddr), "r"(vector_scale_taddr), "r"(enable_input_d));
        __threadfence();
    }

    // Load from TMEM to registers and sum for GEMV
    // For 128x32 accumulation, load and sum values
    if (warp_id == 0) {
        int warp_tiles = (rows_this_tile + 31) / 32;
        for (int tile = 0; tile < warp_tiles; ++tile) {
            int row = tile * 32 + lane_id;
            if (row < rows_this_tile) {
                float sum = 0.0f;
                for (int n = 0; n < MIN_N; n++) {
                    float temp_val;
                    uint32_t elem_addr = accum_taddr + (row * MIN_N + n) * 4;
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                        : "=f"(temp_val) : "r"(elem_addr));
                    sum += temp_val;
                }
                accum_shared[row * MIN_N] = sum;
            }
        }
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    __syncthreads();

    // Store to c
    for (int i = tid; i < rows_this_tile; i += THREADS_PER_CTA) {
        int m = m_block * ROWS_PER_CTA + i;
        size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
        c[c_idx] = __float2half(accum_shared[i * MIN_N]);
    }

    // Dealloc TMEM
    // Fix: Use correct tcgen05.dealloc syntax
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(accum_taddr));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(vector_taddr));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(vector_scale_taddr));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(matrix_taddr));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(matrix_scale_taddr));
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
