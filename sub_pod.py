import os
import torch
from torch.utils.cpp_extension import load_inline

BLOCK_M = 128
DEBUG_STAGE_COUNT = 6
STAGE_LIMIT = int(os.getenv("SUB_TEST_MAX_STAGE", "0"))

cpp_source = """
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c, int tiles_m, int stage_limit);
"""

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda/ptx>
#include <ATen/cuda/CUDAContext.h>
#include <cstdio>

using namespace cuda::ptx;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 8;
constexpr int BLOCK_K = 64;
constexpr int DEBUG_STAGE_COUNT = 6;
constexpr unsigned long long DEBUG_TIMEOUT_CYCLES = 500000000ull;

enum DebugStage : int {
    DBG_ALLOC = 0,
    DBG_COPY = 1,
    DBG_MMA = 2,
    DBG_COMMIT = 3,
    DBG_WAIT = 4,
    DBG_STORE = 5,
};

__device__ __forceinline__ const char* stage_name(int stage) {
    switch (stage) {
        case DBG_ALLOC: return "ALLOC";
        case DBG_COPY: return "COPY";
        case DBG_MMA: return "MMA";
        case DBG_COMMIT: return "COMMIT";
        case DBG_WAIT: return "WAIT";
        case DBG_STORE: return "STORE";
        default: return "UNKNOWN";
    }
}

__device__ __forceinline__ void stage_checkpoint(
    int stage_limit,
    int stage,
    int batch,
    int tile,
    int tile_k) {
    __syncthreads();
    if (threadIdx.x == 0) {
        printf("[sub_test] stage %d (%s) batch=%d tile=%d tile_k=%d\n",
               stage, stage_name(stage), batch, tile, tile_k);
    }
    __syncthreads();
    if (stage == stage_limit) {
        if (threadIdx.x == 0) {
            printf("[sub_test] stopping at stage %d (%s)\n", stage, stage_name(stage));
        }
        __threadfence();
        asm volatile("trap;\n");
    }
}

__device__ __forceinline__ void timeout_trap(
    const char* tag,
    int batch,
    int tile,
    unsigned long long elapsed) {
    if (threadIdx.x == 0) {
        printf("[sub_test] timeout at %s batch=%d tile=%d elapsed=%llu\n",
               tag, batch, tile, elapsed);
    }
    __threadfence();
    asm volatile("trap;\n");
}

__host__ __device__ __forceinline__ constexpr uint32_t encode_mxf4nvf4_idesc(int m, int n, int k, bool scale_is_ue8m0) {
    constexpr uint32_t sparsity = 0u;
    constexpr uint32_t scale_b_id = 0u;
    constexpr uint32_t scale_a_id = 0u;
    constexpr uint32_t atype = 1u;  // e2m1 nvfp4
    constexpr uint32_t btype = 1u;  // e2m1 nvfp4
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

constexpr uint32_t MMA_IDESC = encode_mxf4nvf4_idesc(BLOCK_M, BLOCK_N, BLOCK_K, false);

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
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 { %0, %1, %2, %3, %4, %5, %6, %7 }, [%8];\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        : "=f"(frag[0]), "=f"(frag[1]), "=f"(frag[2]), "=f"(frag[3]),
          "=f"(frag[4]), "=f"(frag[5]), "=f"(frag[6]), "=f"(frag[7])
        : "r"(src)
        : "memory");
}

extern "C" __global__ void gemv_tcgen05_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M,
    int K_bytes,
    int L,
    int N_rows,
    int tiles_m_total,
    int stage_limit)
{
    const int batch = blockIdx.y;
    const int tile_idx = blockIdx.x;
    const int tile_m = tile_idx * BLOCK_M;
    const int lane = threadIdx.x;

    if (batch >= L || tile_m >= M) {
        return;
    }

    const int K = K_bytes * 2;
    const int K_sf = (K + 15) / 16;
    const int tiles_k = (K + BLOCK_K - 1) / BLOCK_K;

    const size_t stride_a_batch = static_cast<size_t>(M) * K_bytes;
    const size_t stride_b_batch = static_cast<size_t>(N_rows) * K_bytes;
    const size_t stride_sfa_batch = static_cast<size_t>(M) * K_sf;
    const size_t stride_sfb_batch = static_cast<size_t>(N_rows) * K_sf;

    const uint8_t* batch_a = reinterpret_cast<const uint8_t*>(a) + batch * stride_a_batch;
    const uint8_t* batch_b = reinterpret_cast<const uint8_t*>(b) + batch * stride_b_batch;
    const uint8_t* batch_sfa = reinterpret_cast<const uint8_t*>(sfa) + batch * stride_sfa_batch;
    const uint8_t* batch_sfb = reinterpret_cast<const uint8_t*>(sfb) + batch * stride_sfb_batch;

    __shared__ uint8_t smem_a[BLOCK_M * (BLOCK_K / 2)];
    __shared__ uint8_t smem_b[BLOCK_N * (BLOCK_K / 2)];
    __shared__ uint8_t smem_scale_a[BLOCK_M * (BLOCK_K / 16)];
    __shared__ uint8_t smem_scale_b[BLOCK_N * (BLOCK_K / 16)];
    __shared__ uint32_t tmem_handles[3];
    __shared__ unsigned long long mbar_state;

    const uint32_t accum_cols = ((BLOCK_M * BLOCK_N + 31) / 32) * 32;
    const uint32_t scale_cols = ((BLOCK_K / 16) + 31) / 32 * 32;

    uint32_t d_tmem = 0;
    uint32_t scale_a_tmem = 0;
    uint32_t scale_b_tmem = 0;

    __syncthreads();
    if (lane == 0) {
        tcgen05_alloc(cta_group_1, &tmem_handles[0], accum_cols);
        tcgen05_alloc(cta_group_1, &tmem_handles[1], scale_cols);
        tcgen05_alloc(cta_group_1, &tmem_handles[2], scale_cols);
        unsigned int arrival = 1u;
        unsigned long long* dst = &mbar_state;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "l"(__cvta_generic_to_shared(dst)), "r"(arrival) : "memory");
    }
    __syncthreads();

    stage_checkpoint(stage_limit, DBG_ALLOC, batch, tile_idx, 0);

    d_tmem = tmem_handles[0];
    scale_a_tmem = tmem_handles[1];
    scale_b_tmem = tmem_handles[2];

    const uint32_t smem_base_a = __cvta_generic_to_shared(smem_a);
    const uint32_t smem_base_b = __cvta_generic_to_shared(smem_b);
    const uint32_t smem_base_scale_a = __cvta_generic_to_shared(smem_scale_a);
    const uint32_t smem_base_scale_b = __cvta_generic_to_shared(smem_scale_b);
    const uint64_t mbar_addr = static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_state));

    for (int tile_k = 0; tile_k < tiles_k; ++tile_k) {
        const int byte_base = tile_k * (BLOCK_K / 2);
        const int scale_base = tile_k * (BLOCK_K / 16);

        for (int idx = lane; idx < BLOCK_M * (BLOCK_K / 2); idx += blockDim.x) {
            int row = idx / (BLOCK_K / 2);
            int col = idx % (BLOCK_K / 2);
            int global_row = tile_m + row;
            int global_byte = byte_base + col;
            uint8_t val = 0;
            if (global_row < M && global_byte < K_bytes) {
                val = batch_a[static_cast<size_t>(global_row) * K_bytes + global_byte];
            }
            smem_a[idx] = val;
        }

        for (int idx = lane; idx < BLOCK_N * (BLOCK_K / 2); idx += blockDim.x) {
            int row = idx / (BLOCK_K / 2);
            int col = idx % (BLOCK_K / 2);
            int global_row = row;
            int global_byte = byte_base + col;
            uint8_t val = 0;
            if (global_row < N_rows && global_byte < K_bytes) {
                val = batch_b[static_cast<size_t>(global_row) * K_bytes + global_byte];
            }
            smem_b[idx] = val;
        }

        for (int idx = lane; idx < BLOCK_M * (BLOCK_K / 16); idx += blockDim.x) {
            int row = idx / (BLOCK_K / 16);
            int col = idx % (BLOCK_K / 16);
            int global_row = tile_m + row;
            int global_sf = scale_base + col;
            uint8_t val = 0;
            if (global_row < M && global_sf < K_sf) {
                val = batch_sfa[static_cast<size_t>(global_row) * K_sf + global_sf];
            }
            smem_scale_a[idx] = val;
        }

        for (int idx = lane; idx < BLOCK_N * (BLOCK_K / 16); idx += blockDim.x) {
            int row = idx / (BLOCK_K / 16);
            int col = idx % (BLOCK_K / 16);
            int global_row = row;
            int global_sf = scale_base + col;
            uint8_t val = 0;
            if (global_row < N_rows && global_sf < K_sf) {
                val = batch_sfb[static_cast<size_t>(global_row) * K_sf + global_sf];
            }
            smem_scale_b[idx] = val;
        }

        __syncthreads();
        if (tile_k == 0) {
            stage_checkpoint(stage_limit, DBG_COPY, batch, tile_idx, tile_k);
        }

        const uint64_t a_desc = pack_smem_desc(smem_base_a, BLOCK_K / 2, BLOCK_K / 2);
        const uint64_t b_desc = pack_smem_desc(smem_base_b, BLOCK_K / 2, BLOCK_K / 2);
        const uint64_t scale_a_desc = pack_smem_desc(smem_base_scale_a, BLOCK_K / 16, BLOCK_K / 16);
        const uint64_t scale_b_desc = pack_smem_desc(smem_base_scale_b, BLOCK_K / 16, BLOCK_K / 16);

        tcgen05_cp_64x128b_warpx2_01_23(cta_group_1, scale_a_tmem, scale_a_desc);
        tcgen05_cp_64x128b_warpx2_01_23(cta_group_1, scale_b_tmem, scale_b_desc);

        __syncthreads();

        int enable_input = tile_k > 0;
        asm volatile(
            "{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n}"
            :
            : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(MMA_IDESC),
              "r"(scale_a_tmem), "r"(scale_b_tmem), "r"(enable_input)
            : "memory");

        __syncthreads();
        if (tile_k == 0) {
            stage_checkpoint(stage_limit, DBG_MMA, batch, tile_idx, tile_k);
        }
    }

    if (lane == 0) {
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
            :
            : "l"(mbar_addr)
            : "memory");
    }

    stage_checkpoint(stage_limit, DBG_COMMIT, batch, tile_idx, 0);

    const unsigned long long wait_start = clock64();
    while (true) {
        unsigned ready = 0;
        asm volatile(
            "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n"
            "selp.b32 %0, 1, 0, p;\n}\n"
            : "=r"(ready)
            : "l"(mbar_addr)
            : "memory");
        if (ready) {
            break;
        }
        unsigned long long elapsed = clock64() - wait_start;
        if (elapsed > DEBUG_TIMEOUT_CYCLES) {
            timeout_trap("mbarrier.wait", batch, tile_idx, elapsed);
        }
    }

    stage_checkpoint(stage_limit, DBG_WAIT, batch, tile_idx, 0);

    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    __syncthreads();

    const size_t c_batch_offset = static_cast<size_t>(batch) * M;
    if (lane < BLOCK_M) {
        int global_row = tile_m + lane;
        float frag[8];
        ld_tmem(frag, d_tmem);
        if (global_row < M) {
            half value = __float2half(frag[0]);
            c[c_batch_offset + global_row] = value;
        }
    }

    __syncthreads();

    stage_checkpoint(stage_limit, DBG_STORE, batch, tile_idx, 0);

    if (lane == 0) {
        tcgen05_dealloc(cta_group_1, scale_b_tmem, scale_cols);
        tcgen05_dealloc(cta_group_1, scale_a_tmem, scale_cols);
        tcgen05_dealloc(cta_group_1, d_tmem, accum_cols);
    }
}

torch::Tensor batched_scaled_gemv_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c,
    int tiles_m,
    int stage_limit) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && sfa.is_cuda() && sfb.is_cuda(), "All tensors must be CUDA resident");
    TORCH_CHECK(a.scalar_type() == torch::kInt8, "Tensor 'a' must be viewed as int8");
    TORCH_CHECK(b.scalar_type() == torch::kInt8, "Tensor 'b' must be viewed as int8");
    TORCH_CHECK(sfa.scalar_type() == torch::kInt8 && sfb.scalar_type() == torch::kInt8, "Scale tensors must be int8 views");
    TORCH_CHECK(c.scalar_type() == at::kHalf, "Output tensor must be fp16");

    int M = static_cast<int>(a.size(0));
    int K_bytes = static_cast<int>(a.size(1));
    int L = static_cast<int>(a.size(2));
    int N_rows = static_cast<int>(b.size(0));
    int tiles_m_expected = (M + BLOCK_M - 1) / BLOCK_M;
    TORCH_CHECK(tiles_m == tiles_m_expected, "tiles_m mismatch with M dimension");

    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, L);
    dim3 block(BLOCK_M);
    gemv_tcgen05_kernel<<<grid, block>>>(
        reinterpret_cast<const int8_t*>(a.data_ptr<int8_t>()),
        reinterpret_cast<const int8_t*>(b.data_ptr<int8_t>()),
        reinterpret_cast<const int8_t*>(sfa.data_ptr<int8_t>()),
        reinterpret_cast<const int8_t*>(sfb.data_ptr<int8_t>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        M,
        K_bytes,
        L,
        N_rows,
        tiles_m,
        stage_limit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
"""

module = load_inline(
    name="batched_scaled_gemv_subtest_ptx",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["batched_scaled_gemv_cuda"],
    # extra_cuda_cflags=[
    #     "-O3",
    #     "--use_fast_math",
    #     "-std=c++17",
    #     "-Xptxas", "-O3",
    #     "-gencode=arch=compute_100a,code=sm_100a",
    # ],


    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-std=c++17',
        '-maxrregcount=64',
        '--ftz=true',
        '-prec-div=false',
        '-Xptxas', '-O3',
        '-Xptxas', '-v',
        '-DNVFP4_DEBUG=1',
        '-gencode=arch=compute_100a,code=sm_100a',
    ],
    with_cuda=True,
    verbose=False
)


def custom_kernel(data):
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device

    a_i8 = a.to(device=device, copy=False).contiguous().view(torch.int8)
    b_i8 = b.to(device=device, copy=False).contiguous().view(torch.int8)
    sfa_i8 = sfa_ref.to(device=device, non_blocking=True).contiguous().view(torch.int8)
    sfb_i8 = sfb_ref.to(device=device, non_blocking=True).contiguous().view(torch.int8)
    tiles_m = (int(a_i8.shape[0]) + BLOCK_M - 1) // BLOCK_M

    module.batched_scaled_gemv_cuda(
        a_i8,
        b_i8,
        sfa_i8,
        sfb_i8,
        c,
        tiles_m,
        STAGE_LIMIT,
    )

    return c
