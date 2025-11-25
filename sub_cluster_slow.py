import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


cpp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 64
#define K_TILE 4096
#define SCALES_PER_TILE (K_TILE / 16)
#define BYTES_PER_TILE (K_TILE / 2)
#define CLUSTER_SIZE 16

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

extern "C" __global__ __cluster_dims__(CLUSTER_SIZE, 1, 1) void gemv_nvfp4_kernel(
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

    // Grid is padded in x to multiples of CLUSTER_SIZE; keep padded CTAs alive
    // through the cluster rendezvous, then let them exit cleanly.
    bool active = (m < M) && (l < L);

    unsigned int cluster_rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cluster_rank));

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

    __shared__ float smem_acc[BLOCK_SIZE];
    __shared__ uint8_t sh_a[2][BYTES_PER_TILE];
    __shared__ uint8_t sh_b[2][BYTES_PER_TILE];
    __shared__ uint8_t sh_sfa[2][SCALES_PER_TILE];
    __shared__ uint8_t sh_sfb[2][SCALES_PER_TILE];
    
    __shared__ __align__(8) uint64_t mbar_b[2];
    __shared__ __align__(8) uint64_t mbar_sfb[2];
    
    float acc = 0.0f;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(&mbar_b[0])), "r"(0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(&mbar_b[1])), "r"(0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(&mbar_sfb[0])), "r"(0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(&mbar_sfb[1])), "r"(0));
    }
    __syncthreads();
    
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");

    if (!active) return;

    const uint8_t* row_a = batch_a + static_cast<size_t>(m) * K_half;
    const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m) * K_sf;

    int tile_count = K / K_TILE;
    int remainder_start = tile_count * K_TILE;
    int buf = 0;
    int parity[2] = {0, 0};

#define ASYNC_COPY_16(dst, src) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src))

#define ASYNC_COPY_4(dst, src) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(src))

    auto issue_tile_a_sfa = [&](int b_idx, int tile_idx) {
        int base_k = tile_idx * K_TILE;
        int base_byte = base_k / 2;
        int base_sf = base_k / 16;
        uint32_t sh_a_base = __cvta_generic_to_shared(&sh_a[b_idx][0]);
        uint32_t sh_sfa_base = __cvta_generic_to_shared(&sh_sfa[b_idx][0]);

        for (int i = tid * 16; i < BYTES_PER_TILE; i += BLOCK_SIZE * 16) {
            ASYNC_COPY_16(sh_a_base + i, row_a + base_byte + i);
        }
        for (int i = tid * 4; i < SCALES_PER_TILE; i += BLOCK_SIZE * 4) {
            ASYNC_COPY_4(sh_sfa_base + i, row_sfa + base_sf + i);
        }
        asm volatile("cp.async.commit_group;");
    };

    auto issue_tile_b_sfb = [&](int b_idx, int tile_idx) {
        int base_k = tile_idx * K_TILE;
        int base_byte = base_k / 2;
        int base_sf = base_k / 16;
        
        uint32_t sh_b_base = __cvta_generic_to_shared(&sh_b[b_idx][0]);
        uint32_t sh_sfb_base = __cvta_generic_to_shared(&sh_sfb[b_idx][0]);
        uint32_t mbar_b_addr = __cvta_generic_to_shared(&mbar_b[b_idx]);
        uint32_t mbar_sfb_addr = __cvta_generic_to_shared(&mbar_sfb[b_idx]);
        
        if (tid == 0) {
            asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                :: "r"(mbar_b_addr), "r"((unsigned)BYTES_PER_TILE));
            asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                :: "r"(mbar_sfb_addr), "r"((unsigned)SCALES_PER_TILE));
        }
        
        asm volatile("barrier.cluster.arrive;");
        asm volatile("barrier.cluster.wait;");
        
        if (cluster_rank == 0 && tid == 0) {
            uint16_t ctaMask = 0xFFFF;
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
                "[%0], [%1], %2, [%3], %4;"
                :: "r"(sh_b_base), "l"(batch_b + base_byte), 
                   "r"((unsigned)BYTES_PER_TILE), "r"(mbar_b_addr), "h"(ctaMask)
            );
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
                "[%0], [%1], %2, [%3], %4;"
                :: "r"(sh_sfb_base), "l"(batch_sfb + base_sf), 
                   "r"((unsigned)SCALES_PER_TILE), "r"(mbar_sfb_addr), "h"(ctaMask)
            );
        }
    };

    auto wait_mbar = [&](int b_idx, int par) {
        uint32_t mbar_b_addr = __cvta_generic_to_shared(&mbar_b[b_idx]);
        uint32_t mbar_sfb_addr = __cvta_generic_to_shared(&mbar_sfb[b_idx]);
        
        if (tid == 0) {
            uint32_t done = 0;
            while (!done) {
                asm volatile(
                    "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                    : "=r"(done) : "r"(mbar_b_addr), "r"(par)
                );
            }
            done = 0;
            while (!done) {
                asm volatile(
                    "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                    : "=r"(done) : "r"(mbar_sfb_addr), "r"(par)
                );
            }
        }
        __syncthreads();
    };

    if (tile_count > 0) {
        issue_tile_a_sfa(0, 0);
        issue_tile_b_sfb(0, 0);
        
        asm volatile("cp.async.wait_group 0;");
        wait_mbar(0, parity[0]);
        parity[0] ^= 1;
        __syncthreads();

        for (int tile = 0; tile < tile_count; ++tile) {
            int next_buf = buf ^ 1;
            
            if (tile + 1 < tile_count) {
                issue_tile_a_sfa(next_buf, tile + 1);
                issue_tile_b_sfb(next_buf, tile + 1);
            }

            for (int sf = tid; sf < SCALES_PER_TILE; sf += BLOCK_SIZE) {
                float scale = decode_fp8(static_cast<int8_t>(sh_sfa[buf][sf])) *
                              decode_fp8(static_cast<int8_t>(sh_sfb[buf][sf]));
                __half scale_h = __float2half(scale);
                __half2 scale_h2 = __halves2half2(scale_h, scale_h);

                int byte_base = sf * 8;
#pragma unroll
                for (int bb = 0; bb < 8; ++bb) {
                    __half2 a2 = decode_fp4x2(sh_a[buf][byte_base + bb]);
                    __half2 b2 = decode_fp4x2(sh_b[buf][byte_base + bb]);
                    __half2 prod = __hmul2(__hmul2(a2, b2), scale_h2);
                    float2 f = __half22float2(prod);
                    acc += f.x + f.y;
                }
            }

            if (tile + 1 < tile_count) {
                asm volatile("cp.async.wait_group 0;");
                wait_mbar(next_buf, parity[next_buf]);
                parity[next_buf] ^= 1;
                __syncthreads();
                buf = next_buf;
            }
        }
    }

    int remainder_sf_start = remainder_start / 16;
    for (int sf = remainder_sf_start + tid; sf < K_sf; sf += BLOCK_SIZE) {
        float scale = decode_fp8(static_cast<int8_t>(row_sfa[sf])) *
                      decode_fp8(static_cast<int8_t>(batch_sfb[sf]));
        __half scale_h = __float2half(scale);
        __half2 scale_h2 = __halves2half2(scale_h, scale_h);

        int byte_base = sf * 8;
#pragma unroll
        for (int b_byte = 0; b_byte < 8; ++b_byte) {
            uint8_t a_pack = row_a[byte_base + b_byte];
            uint8_t b_pack = batch_b[byte_base + b_byte];
            __half2 a2 = decode_fp4x2(a_pack);
            __half2 b2 = decode_fp4x2(b_pack);
            __half2 prod = __hmul2(__hmul2(a2, b2), scale_h2);
            float2 f = __half22float2(prod);
            acc += f.x + f.y;
        }
    }

#undef ASYNC_COPY_16
#undef ASYNC_COPY_4

    smem_acc[tid] = acc;
    __syncthreads();

    float warp_sum = acc;
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }

    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) smem_acc[warp_id] = warp_sum;

    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < (blockDim.x >> 5)) ? smem_acc[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
            c[c_idx] = __float2half(block_sum);
        }
    }
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    int padded_M = ((M + CLUSTER_SIZE - 1) / CLUSTER_SIZE) * CLUSTER_SIZE;
    dim3 grid(padded_M, L);
    dim3 block(BLOCK_SIZE);

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    // Allow the explicitly requested 16-CTA clusters on architectures that support it.
    // Without this opt-in, the runtime rejects the launch with cudaErrorInvalidClusterSize.
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            gemv_nvfp4_kernel,
            cudaFuncAttributeNonPortableClusterSizeAllowed,
            1
        );
        attr_set = true;
    }

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = 0;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, gemv_nvfp4_kernel,
        a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr,
        M, K, L, N_rows);

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
    return module.batched_scaled_gemv_cuda(
        a,
        b,
        sfa_ref,
        sfb_ref,
        c
    )
