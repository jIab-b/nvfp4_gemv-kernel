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

constexpr int TILE_M = 64;
constexpr int TILE_K = 64;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;

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
    __shared__ uint32_t smem_a[TILE_M * TILE_K / 8];  // FP4 packed
    __shared__ uint32_t smem_b[TILE_K / 8];
    __shared__ __nv_fp8_e4m3 smem_sfa[TILE_M * (TILE_K / 16)];
    __shared__ __nv_fp8_e4m3 smem_sfb[TILE_K / 16];
    
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
                    (k_offset / 8) * L + batch_idx
                ];
            }
        }
        
        // Load b scales (FP8)
        for (int i = tid; i < TILE_K / 16; i += 128) {
            int kb = (k_tile / 16) + i;
            if (kb < K_blocks) {
                smem_sfb[i] = reinterpret_cast<const __nv_fp8_e4m3*>(sfb)[
                    kb * L + batch_idx
                ];
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
                                (global_m * K + k_offset) / 8 * L + batch_idx
                            ];
                    }
                }
            }
        }
        
        // Load a scales (FP8)
        for (int m_local = warp_id; m_local < TILE_M; m_local += NUM_WARPS) {
            int global_m = m_base + m_local;
            if (global_m < M) {
                for (int kb = lane_id; kb < TILE_K / 16; kb += WARP_SIZE) {
                    int global_kb = (k_tile / 16) + kb;
                    if (global_kb < K_blocks) {
                        smem_sfa[m_local * (TILE_K / 16) + kb] = 
                            reinterpret_cast<const __nv_fp8_e4m3*>(sfa)[
                                (global_m * K_blocks + global_kb) * L + batch_idx
                            ];
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Compute using wgmma instruction
        // Process rows assigned to this warp
        for (int m_offset = 0; m_offset < rows_per_warp; m_offset++) {
            int m_local = warp_m_base + m_offset;
            int global_m = m_base + m_local;
            if (global_m >= M) break;
            
            // Load fragments for wgmma
            uint32_t frag_a[2];  // 16 FP4 values
            uint32_t frag_b[2];
            __nv_fp8_e4m3 scale_a[4];
            __nv_fp8_e4m3 scale_b[4];
            
            // Process in blocks of 64 (TILE_K)
            // Using wgmma.mma_async.m16n8k64.f32.fp4.fp4
            for (int kb = 0; kb < TILE_K / 64; kb++) {
                // Load 64 elements (16 uint32 for FP4)
                if (lane_id < 16) {
                    int idx = kb * 16 + lane_id;
                    frag_a[lane_id % 2] = smem_a[m_local * (TILE_K / 8) + idx / 2];
                    frag_b[lane_id % 2] = smem_b[idx / 2];
                }
                
                // Load 4 scales (64 / 16 = 4)
                if (lane_id < 4) {
                    scale_a[lane_id] = smem_sfa[m_local * (TILE_K / 16) + kb * 4 + lane_id];
                    scale_b[lane_id] = smem_sfb[kb * 4 + lane_id];
                }
                
                // Execute tensor core with hardware scaling
                // This is the key B200 instruction
                asm volatile(
                    "wgmma.mma_async.sync.aligned.m16n8k64.f32.e2m1.e2m1 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5}, "
                    "{%6, %7}, "
                    "p, %8, %9, %10, %11;"
                    : "+f"(acc[m_offset * 4 + 0]),
                      "+f"(acc[m_offset * 4 + 1]),
                      "+f"(acc[m_offset * 4 + 2]),
                      "+f"(acc[m_offset * 4 + 3])
                    : "r"(frag_a[0]), "r"(frag_a[1]),
                      "r"(frag_b[0]), "r"(frag_b[1]),
                      "r"(*reinterpret_cast<uint32_t*>(&scale_a[0])),
                      "r"(*reinterpret_cast<uint32_t*>(&scale_a[2])),
                      "r"(*reinterpret_cast<uint32_t*>(&scale_b[0])),
                      "r"(*reinterpret_cast<uint32_t*>(&scale_b[2]))
                );
            }
        }
    }
    
    // Write results
    for (int m_offset = 0; m_offset < rows_per_warp; m_offset++) {
        int global_m = m_base + warp_m_base + m_offset;
        if (global_m < M && lane_id == 0) {
            c[global_m * L + batch_idx] = __float2half(acc[m_offset * 4]);
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
    a, b, sfa, sfb, c = data
    return module.batched_scaled_gemv_cuda(a, b, sfa, sfb, c)