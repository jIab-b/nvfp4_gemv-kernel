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

constexpr int TILE_M = 128;
constexpr int K_TILE = 64;
constexpr int K_TILE_BYTES = K_TILE / 2;
constexpr int K_TILE_SF = K_TILE / 16;

// Helper to compute SMEM descriptor per Table 40
// Base address, LBO, SBO are in bytes and shifted right by 4 before packing
__device__ __forceinline__ unsigned long long make_smem_descriptor(
    unsigned base_addr,    // byte address in shared memory
    unsigned lbo_bytes,    // leading-dimension byte offset
    unsigned sbo_bytes     // stride dimension byte offset
) {
    // matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4
    unsigned base_encoded = (base_addr & 0x3FFFF) >> 4;
    unsigned lbo_encoded = (lbo_bytes & 0x3FFFF) >> 4;
    unsigned sbo_encoded = (sbo_bytes & 0x3FFFF) >> 4;

    unsigned long long desc = 0;
    desc |= ((unsigned long long)base_encoded << 0);   // bits 0-13
    desc |= ((unsigned long long)lbo_encoded << 16);   // bits 16-29
    desc |= ((unsigned long long)sbo_encoded << 32);   // bits 32-45
    desc |= (1ULL << 46);                              // bits 46-48: 0b001
    // bits 49-51: 0 (matrix base offset, canonical layout)
    // bit 52: 0 (relative mode)
    // bits 53-60: 0 (fixed constant)
    // bits 61-63: 0 (no swizzle)
    return desc;
}

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


__global__ void gemv_nvfp4_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L,
    int N_rows
) {
    int tile_m = blockIdx.x;
    int l = blockIdx.y;
    int tid = threadIdx.x;

    int row_start = tile_m * TILE_M;
    if (row_start >= M || l >= L) {
        return;
    }

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

    int rows_in_tile = min(TILE_M, M - row_start);

    // Padding: pad each SFA/SFB row to 16 bytes for descriptor alignment
    constexpr int SFA_ROW_PADDED = 16;  // K_TILE_SF=4, padded to 16 for alignment
    constexpr int SFB_ROW_PADDED = 16;  // K_TILE_SF=4, padded to 16 for alignment

    extern __shared__ uint8_t shared_bytes[];
    uint8_t* smem_a = shared_bytes;
    uint8_t* smem_sfa = smem_a + TILE_M * K_TILE_BYTES;
    uint8_t* smem_b = smem_sfa + TILE_M * SFA_ROW_PADDED;
    uint8_t* smem_sfb = smem_b + K_TILE_BYTES;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    __shared__ unsigned tmem_sfa_addr;
    __shared__ unsigned tmem_sfb_addr;
    __shared__ unsigned long long desc_sfa;
    __shared__ unsigned long long desc_sfb;
#endif

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        int chunk_bytes = K_TILE_BYTES;
        int chunk_sf_bytes = K_TILE_SF;
        int a_chunk_offset = k0 / 2;
        int sfa_chunk_offset = k0 / 16;

        for (int idx = tid; idx < rows_in_tile * chunk_bytes; idx += blockDim.x) {
            int row = idx / chunk_bytes;
            int byte = idx % chunk_bytes;
            const uint8_t* src = batch_a + static_cast<size_t>(row_start + row) * K_half;
            smem_a[row * chunk_bytes + byte] = src[a_chunk_offset + byte];
        }

        for (int idx = tid; idx < rows_in_tile * chunk_sf_bytes; idx += blockDim.x) {
            int row = idx / chunk_sf_bytes;
            int byte = idx % chunk_sf_bytes;
            const uint8_t* src = batch_sfa + static_cast<size_t>(row_start + row) * K_sf;
            smem_sfa[row * SFA_ROW_PADDED + byte] = src[sfa_chunk_offset + byte];  // Use padded stride
        }

        for (int idx = tid; idx < chunk_bytes; idx += blockDim.x) {
            smem_b[idx] = batch_b[a_chunk_offset + idx];
        }

        for (int idx = tid; idx < chunk_sf_bytes; idx += blockDim.x) {
            smem_sfb[idx] = batch_sfb[sfa_chunk_offset + idx];
        }

        __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        if (tid == 0) {
            // Stage D2: SMEM descriptor computation and verification
            // Compute SMEM descriptors per Table 40

            // SFA descriptor
            // SFA tile layout: TILE_M rows × SFA_ROW_PADDED (16) bytes per row
            // LBO (leading-dimension byte offset) = stride between rows = SFA_ROW_PADDED
            // SBO (stride byte offset) = stride between K chunks = TILE_M * SFA_ROW_PADDED
            unsigned sfa_base_smem = __cvta_generic_to_shared(smem_sfa);
            desc_sfa = make_smem_descriptor(
                sfa_base_smem,
                SFA_ROW_PADDED,                // LBO: byte stride between rows
                TILE_M * SFA_ROW_PADDED        // SBO: stride for next K tile
            );

            // SFB descriptor
            // SFB is K_TILE_SF (4) bytes, padded to 16 for alignment
            unsigned sfb_base_smem = __cvta_generic_to_shared(smem_sfb);
            desc_sfb = make_smem_descriptor(
                sfb_base_smem,
                K_TILE_SF,                     // LBO: actual data size
                K_TILE_SF                      // SBO: stride
            );

            // Store TMEM allocation addresses
            tmem_sfa_addr = sfa_base_smem;
            tmem_sfb_addr = sfb_base_smem;
        }
        __syncthreads();

        // Stage D2 Verification: Validate descriptor computation and padded layout
        // This ensures that the descriptor math is correct before wiring up tcgen05.cp/ld
        if (tid == 0) {
            bool desc_valid = true;

            // Verify descriptor bit fields
            // Extract fields from desc_sfa per Table 40
            unsigned sfa_base_extracted = (desc_sfa >> 0) & 0x3FFF;      // bits 0-13
            unsigned sfa_lbo_extracted = (desc_sfa >> 16) & 0x3FFF;      // bits 16-29
            unsigned sfa_sbo_extracted = (desc_sfa >> 32) & 0x3FFF;      // bits 32-45
            unsigned sfa_fixed = (desc_sfa >> 46) & 0x7;                 // bits 46-48, should be 0b001

            // Expected values after encoding (value >> 4)
            // LBO = SFA_ROW_PADDED = 16, encodes as 16 >> 4 = 1
            // SBO = TILE_M * SFA_ROW_PADDED = 2048, encodes as 2048 >> 4 = 128
            unsigned expected_sfa_lbo = (SFA_ROW_PADDED >> 4);
            unsigned expected_sfa_sbo = (TILE_M * SFA_ROW_PADDED) >> 4;

            if (sfa_lbo_extracted != expected_sfa_lbo ||
                sfa_sbo_extracted != expected_sfa_sbo ||
                sfa_fixed != 0b001) {
                desc_valid = false;
            }

            // Verify SFB descriptor
            unsigned sfb_lbo_extracted = (desc_sfb >> 16) & 0x3FFF;
            unsigned sfb_sbo_extracted = (desc_sfb >> 32) & 0x3FFF;
            unsigned sfb_fixed = (desc_sfb >> 46) & 0x7;

            unsigned expected_sfb_lbo = (K_TILE_SF >> 4);
            unsigned expected_sfb_sbo = (K_TILE_SF >> 4);

            if (sfb_lbo_extracted != expected_sfb_lbo ||
                sfb_sbo_extracted != expected_sfb_sbo ||
                sfb_fixed != 0b001) {
                desc_valid = false;
            }

            if (!desc_valid) {
                asm volatile("trap;");
            }
        }
        __syncthreads();

        // Verify padded layout preserves original SFA/SFB data
        // Check first few rows to ensure padding doesn't corrupt actual data
        if (tid < rows_in_tile && rows_in_tile > 0) {
            bool data_valid = true;

            // Verify SFA data matches original
            for (int byte = 0; byte < chunk_sf_bytes; ++byte) {
                uint8_t original = batch_sfa[static_cast<size_t>(row_start + tid) * K_sf + sfa_chunk_offset + byte];
                uint8_t padded = smem_sfa[tid * SFA_ROW_PADDED + byte];
                if (original != padded) {
                    data_valid = false;
                    break;
                }
            }

            // Verify SFB data matches original (same for all rows, but check consistency)
            if (tid == 0) {
                for (int byte = 0; byte < chunk_sf_bytes; ++byte) {
                    uint8_t original = batch_sfb[sfa_chunk_offset + byte];
                    uint8_t loaded = smem_sfb[byte];
                    if (original != loaded) {
                        data_valid = false;
                        break;
                    }
                }
            }

            if (!data_valid) {
                asm volatile("trap;");
            }
        }
        __syncthreads();

#endif

        if (tid < rows_in_tile) {
            const uint8_t* row_a_tile = smem_a + tid * chunk_bytes;
            const uint8_t* row_sfa_tile = smem_sfa + tid * SFA_ROW_PADDED;  // Use padded stride

            for (int sf = 0; sf < chunk_sf_bytes; ++sf) {
                float scale_a = decode_fp8(static_cast<int8_t>(row_sfa_tile[sf]));
                float scale_b = decode_fp8(static_cast<int8_t>(smem_sfb[sf]));
                for (int nib = 0; nib < 16; ++nib) {
                    int local_elem = sf * 16 + nib;
                    int global_k = k0 + local_elem;
                    if (global_k >= K) {
                        break;
                    }
                    int byte_idx = local_elem / 2;
                    bool high = (local_elem & 1);
                    uint8_t a_byte = row_a_tile[byte_idx];
                    uint8_t b_byte = smem_b[byte_idx];
                    uint8_t a_nib = high ? (a_byte >> 4) : (a_byte & 0xF);
                    uint8_t b_nib = high ? (b_byte >> 4) : (b_byte & 0xF);
                    float a_val = decode_fp4(a_nib);
                    float b_val = decode_fp4(b_nib);
                    acc += (a_val * scale_a) * (b_val * scale_b);
                }
            }
        }

        __syncthreads();
    }

    if (tid < rows_in_tile) {
        size_t c_idx = static_cast<size_t>(row_start + tid) + static_cast<size_t>(l) * M;
        c[c_idx] = __float2half(acc);
    }
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);
    
    dim3 grid((M + TILE_M - 1) / TILE_M, L);
    dim3 block(TILE_M);
    // Shared memory: A tiles + padded SFA tiles + B tiles + padded SFB tiles
    // A: TILE_M × K_TILE_BYTES
    // SFA: TILE_M × 16 (padded from K_TILE_SF=4)
    // B: K_TILE_BYTES
    // SFB: 16 (padded from K_TILE_SF=4)
    constexpr int SFA_ROW_PADDED = 16;
    constexpr int SFB_ROW_PADDED = 16;
    size_t shared_bytes = static_cast<size_t>(TILE_M) * (K_TILE_BYTES + SFA_ROW_PADDED)
                        + (K_TILE_BYTES + SFB_ROW_PADDED);

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
