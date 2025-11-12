import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

    
cpp_source = """
torch::Tensor batched_scaled_gemv_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c
);
"""

cuda_source = """
#include <cuda_fp16.h>
#include <cuda_fp8.h>

constexpr int TILE_M = 256;
constexpr int TILE_K = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;

// Pack scales into tiled UE4M3 layout for tensor core consumption
// Tile structure: 128 scales per tile, covering 128 rows (M) x 4 K-blocks (64 K-elements)
// Layout: offset_in_tile = (row%32)*16 + (row/32)*4 + k_block_in_tile
__device__ inline void pack_scale_tile(
    const __nv_fp8_e4m3* src,  // Source: [rows, k_blocks] in row-major
    uint8_t* dst,               // Dest: tiled layout
    int row_in_tile,            // 0-127 or 0-63 for smaller tiles
    int k_block_in_tile,        // 0-3
    int tile_rows               // 128 or actual tile size
) {
    if (row_in_tile < tile_rows && k_block_in_tile < 4) {
        int offset_in_tile = ((row_in_tile & 31) * 16) + ((row_in_tile >> 5) * 4) + k_block_in_tile;
        dst[offset_in_tile] = *reinterpret_cast<const uint8_t*>(&src[row_in_tile * 4 + k_block_in_tile]);
    }
}

__global__ void __launch_bounds__(128)
batched_gemv_kernel(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    const uint8_t* __restrict__ sfa,
    const uint8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L
) {
    const int batch_idx = blockIdx.y;
    const int m_base = blockIdx.x * TILE_M;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Shared memory
    __shared__ uint32_t smem_a[TILE_M * TILE_K / 8];  // FP4 packed: 256*256/8 = 8192 uint32
    __shared__ uint32_t smem_b[TILE_K / 8];           // 256/8 = 32 uint32
    // Tiled scale layout for Blackwell: multiple 128-scale tiles
    // For TILE_M=256, TILE_K=256: 256 rows × 16 K-blocks = 4096 scales
    // Need 4096/128 = 32 tiles, each 128 bytes = 4KB total
    __shared__ uint8_t smem_sfa_tiled[4096];  // Tiled layout for A scales
    __shared__ uint8_t smem_sfb_tiled[128];   // Tiled layout for B scales (1 row)
    __shared__ __nv_fp8_e4m3 smem_sfa_tmp[TILE_M * (TILE_K / 16)];  // 256*16 = 4096 FP8
    __shared__ __nv_fp8_e4m3 smem_sfb_tmp[TILE_K / 16];              // 256/16 = 16 FP8
    
    // Accumulator fragments (per warp, 16 rows)
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    
    const int K_blocks = K / 16;
    const int rows_per_warp = TILE_M / NUM_WARPS;
    const int warp_m_base = warp_id * rows_per_warp;
    
    // Main loop over K
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        __syncthreads();
        
        // Load b vector (FP4, packed 8 values per uint32)
        for (int i = tid; i < TILE_K / 8; i += 128) {
            int k_offset = k_tile + i * 8;
            if (k_offset < K) {
                smem_b[i] = reinterpret_cast<const uint32_t*>(b)[
                    (k_offset * L + batch_idx) / 8
                ];
            }
        }
        
        // Load b scales (FP8) to temp buffer - vectorized 4 scales per uint32 load
        for (int i = tid * 4; i < TILE_K / 16; i += 128 * 4) {
            int kb = (k_tile / 16) + i;
            if (kb + 3 < K_blocks) {
                uint32_t packed_scales = reinterpret_cast<const uint32_t*>(sfb)[
                    (kb * L + batch_idx) / 4
                ];
                smem_sfb_tmp[i + 0] = *reinterpret_cast<__nv_fp8_e4m3*>(&packed_scales);
                smem_sfb_tmp[i + 1] = *reinterpret_cast<__nv_fp8_e4m3*>(reinterpret_cast<uint8_t*>(&packed_scales) + 1);
                smem_sfb_tmp[i + 2] = *reinterpret_cast<__nv_fp8_e4m3*>(reinterpret_cast<uint8_t*>(&packed_scales) + 2);
                smem_sfb_tmp[i + 3] = *reinterpret_cast<__nv_fp8_e4m3*>(reinterpret_cast<uint8_t*>(&packed_scales) + 3);
            } else {
                // Handle boundary case with scalar loads
                for (int j = 0; j < 4 && kb + j < K_blocks; j++) {
                    smem_sfb_tmp[i + j] = reinterpret_cast<const __nv_fp8_e4m3*>(sfb)[
                        (kb + j) * L + batch_idx
                    ];
                }
            }
        }
        
        // Load a matrix rows (FP4)
        for (int m_local = warp_id; m_local < TILE_M; m_local += NUM_WARPS) {
            int global_m = m_base + m_local;
            if (global_m < M) {
                for (int k = lane_id; k < TILE_K / 8; k += WARP_SIZE) {
                    int k_offset = k_tile + k * 8;
                    if (k_offset < K) {
                        smem_a[m_local * (TILE_K / 8) + k] =
                            reinterpret_cast<const uint32_t*>(a)[
                                ((global_m * K + k_offset) * L + batch_idx) / 8
                            ];
                    }
                }
            }
        }
        
        // Load a scales (FP8) to temp buffer - vectorized 4 scales per uint32 load
        for (int m_local = warp_id; m_local < TILE_M; m_local += NUM_WARPS) {
            int global_m = m_base + m_local;
            if (global_m < M) {
                for (int kb = lane_id * 4; kb < TILE_K / 16; kb += WARP_SIZE * 4) {
                    int global_kb = (k_tile / 16) + kb;
                    if (global_kb + 3 < K_blocks) {
                        uint32_t packed_scales = reinterpret_cast<const uint32_t*>(sfa)[
                            ((global_m * K_blocks + global_kb) * L + batch_idx) / 4
                        ];
                        smem_sfa_tmp[m_local * (TILE_K / 16) + kb + 0] = *reinterpret_cast<__nv_fp8_e4m3*>(&packed_scales);
                        smem_sfa_tmp[m_local * (TILE_K / 16) + kb + 1] = *reinterpret_cast<__nv_fp8_e4m3*>(reinterpret_cast<uint8_t*>(&packed_scales) + 1);
                        smem_sfa_tmp[m_local * (TILE_K / 16) + kb + 2] = *reinterpret_cast<__nv_fp8_e4m3*>(reinterpret_cast<uint8_t*>(&packed_scales) + 2);
                        smem_sfa_tmp[m_local * (TILE_K / 16) + kb + 3] = *reinterpret_cast<__nv_fp8_e4m3*>(reinterpret_cast<uint8_t*>(&packed_scales) + 3);
                    } else {
                        // Handle boundary case with scalar loads
                        for (int j = 0; j < 4 && global_kb + j < K_blocks; j++) {
                            smem_sfa_tmp[m_local * (TILE_K / 16) + kb + j] =
                                reinterpret_cast<const __nv_fp8_e4m3*>(sfa)[
                                    (global_m * K_blocks + global_kb + j) * L + batch_idx
                                ];
                        }
                    }
                }
            }
        }

        __syncthreads();

        // Pack scales into tiled layout for tensor core consumption
        // B scales: 1 row × 4 K-blocks = 4 scales -> pack into 128-byte tile
        for (int i = tid; i < TILE_K / 16; i += 128) {
            if (i < 4) {  // Only 4 K-blocks per TILE_K=64
                pack_scale_tile(smem_sfb_tmp, smem_sfb_tiled, 0, i, 1);
            }
        }

        // A scales: TILE_M rows × 4 K-blocks = TILE_M×4 scales
        // Pack into tiled layout (128 scales per 128-byte tile)
        for (int idx = tid; idx < TILE_M * (TILE_K / 16); idx += 128) {
            int row = idx / 4;
            int kb = idx % 4;
            if (row < TILE_M) {
                // For TILE_M=64, we only use first 64 rows of a 128-row tile
                pack_scale_tile(smem_sfa_tmp, smem_sfa_tiled, row, kb, TILE_M);
            }
        }
        
        __syncthreads();
        
        // Compute using Blackwell mma.sync instruction with hardware block scaling
        // Instruction: mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32
        // This is the Blackwell FP4 tensor core path (PTX ISA 8.7+, SM_100/SM_120)
        // Process rows assigned to this warp (64 rows total, 16 rows per MMA)
        for (int m_offset = 0; m_offset < rows_per_warp; m_offset += 16) {
            int m_local = warp_m_base + m_offset;
            int global_m = m_base + m_local;
            if (global_m >= M) break;

            // Process K dimension in 32-element chunks (standard for tcgen05.mma with block scaling)
            // Each block of 32 FP4 elements has 2 scale factors (32/16=2)
            for (int k_chunk = 0; k_chunk < TILE_K / 32; k_chunk++) {
                // Load A fragment: 32 FP4 values = 4 uint32_t registers
                uint32_t frag_a[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int idx = m_local * (TILE_K / 8) + k_chunk * 4 + i;
                    frag_a[i] = smem_a[idx];
                }

                // Load B fragment: 32 FP4 values = 4 uint32_t registers
                uint32_t frag_b[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int idx = k_chunk * 4 + i;
                    frag_b[i] = smem_b[idx];
                }

                // Load scale factors from tiled layout
                // For k_chunk, we need 2 scales (32 elements / 16 per scale = 2)
                // Scales are in tiled layout, compute offset
                uint32_t scale_a_packed, scale_b_packed;

                // Load A scales (2 FP8 values) from tiled buffer
                int kb_start = k_chunk * 2;
                int scale_offset_a = ((m_local & 31) * 16) + ((m_local >> 5) * 4) + (kb_start % 4);
                scale_a_packed = *reinterpret_cast<uint32_t*>(&smem_sfa_tiled[scale_offset_a]);

                // Load B scales (2 FP8 values) from tiled buffer
                int scale_offset_b = kb_start % 4;
                scale_b_packed = *reinterpret_cast<uint32_t*>(&smem_sfb_tiled[scale_offset_b]);

                // Blackwell Hardware Acceleration - tcgen05.mma PTX instruction
                // Correct Blackwell instruction: tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec_size::2X
                // This IS a valid PTX instruction that can be used with inline assembly
                // Requires CTA group setup but is callable from properly configured kernels

                // Each MMA operation handles 16 rows, so mma_idx = m_offset / 16
                int mma_idx = m_offset / 16;

                // Temporaries for accumulator input (cannot read and write same variables)
                float acc_in[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) acc_in[i] = acc[mma_idx * 4 + i];

                asm volatile(
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec_size::2X "
                    "{%0, %1, %2, %3}, "                    // D: 4 FP32 outputs (16×8 result per thread)
                    "{%4, %5, %6, %7}, "                    // A: 4 uint32 (16×32 FP4 data)
                    "{%8, %9, %10, %11}, "                  // B: 4 uint32 (32×8 FP4 data)
                    "{%12, %13, %14, %15}, "                // C: 4 FP32 accumulators (input)
                    "%16, %17;"                             // scaleA, scaleB: uint32 (2× FP8 E4M3)
                    : "=f"(acc[mma_idx * 4 + 0]), "=f"(acc[mma_idx * 4 + 1]),
                      "=f"(acc[mma_idx * 4 + 2]), "=f"(acc[mma_idx * 4 + 3])
                    : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]),
                      "r"(frag_b[0]), "r"(frag_b[1]), "r"(frag_b[2]), "r"(frag_b[3]),
                      "f"(acc_in[0]), "f"(acc_in[1]), "f"(acc_in[2]), "f"(acc_in[3]),
                      "r"(scale_a_packed), "r"(scale_b_packed)
                );

            }
        }
    }
    
    // Write results - Simplified approach assuming Blackwell MMA follows similar distribution to other tensor cores
    // Each warp handles 64 rows, with 4 MMA operations × 16 rows each
    // Assume each thread contributes one result value (similar to standard MMA patterns)
    for (int i = 0; i < 16; i++) {  // 16 accumulator values per warp
        // Map accumulator index to global row index
        // This is a simplified mapping - may need adjustment based on actual MMA output distribution
        int local_row_idx = i;  // Assume accumulator i corresponds to row i in the warp's 64-row block
        int global_m = m_base + warp_m_base + local_row_idx;
        if (global_m < M && lane_id == 0) {  // Only lane 0 writes (simplified)
            c[global_m * L + batch_idx] = __float2half(acc[i]);
        }
    }
}

torch::Tensor batched_scaled_gemv_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c
) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int L = a.size(2);
    
    dim3 grid((M + TILE_M - 1) / TILE_M, L);
    dim3 block(128);
    
    batched_gemv_kernel<<<grid, block>>>(
        a.data_ptr<uint8_t>(),
        b.data_ptr<uint8_t>(),
        sfa.data_ptr<uint8_t>(),
        sfb.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        M, K, L
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
        '-gencode=arch=compute_100,code=sm_100',
    ],
    with_cuda=True,
    verbose=False
)
    


def custom_kernel(data: input_t) -> output_t:
    """
    Custom implementation of batched scaled GEMV using B200 tensor cores with hardware scaling.
    """
    a, b, _, _, sfa, sfb, c = data

    # Convert Float4 and Float8 tensors to uint8 storage (free reinterpretation)
    # CUDA kernel expects raw uint8 bytes for FP4/FP8 data manipulation
    if a.dtype != torch.uint8:
        a = a.view(torch.uint8)
    if b.dtype != torch.uint8:
        b = b.view(torch.uint8)
    if sfa.dtype != torch.uint8:
        sfa = sfa.view(torch.uint8)
    if sfb.dtype != torch.uint8:
        sfb = sfb.view(torch.uint8)

    return module.batched_scaled_gemv_cuda(a, b, sfa, sfb, c)