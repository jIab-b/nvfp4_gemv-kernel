import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_source = """
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda/ptx>

using namespace cuda::ptx;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 8;
constexpr int BLOCK_K = 64;

__device__ __forceinline__ uint64_t pack_smem_desc(uint32_t base, uint32_t ldm_bytes, uint32_t stride_bytes) {
    // Descriptor layout documented in PTX ISA: bits [0:15] = leading dimension bytes,
    // bits [16:29] = stride/16B, bits [30:47] = base>>4, bits [48:63] swizzle (zero for linear row-major).
    uint64_t desc = 0;
    uint64_t ldm = static_cast<uint64_t>(ldm_bytes & 0xFFFFull);
    uint64_t stride = static_cast<uint64_t>((stride_bytes >> 4) & 0x3FFFull);
    uint64_t base_field = static_cast<uint64_t>((base >> 4) & 0xFFFFull);
    desc |= ldm;
    desc |= (stride << 16);
    desc |= (base_field << 30);
    return desc;
}

__device__ __forceinline__ uint32_t alloc_tmem(uint32_t cols) {
    uint32_t handle = 0;
    tcgen05_alloc(cta_group_1{}, &handle, cols);
    return handle;
}

__device__ __forceinline__ void dealloc_tmem(uint32_t handle, uint32_t cols) {
    tcgen05_dealloc(cta_group_1{}, handle, cols);
}

__device__ __forceinline__ void cp_to_tmem(uint32_t dst, uint64_t s_desc) {
    tcgen05_cp_64x128b_warpx2_01_23(cta_group_1{}, dst, s_desc);
}

__device__ __forceinline__ void ld_tmem(float (&frag)[8], uint32_t src) {
    asm volatile(
        "tcgen05.ldmatrix.sync.aligned.m16n8.row.f32 { %0, %1, %2, %3, %4, %5, %6, %7 }, [%8];\n"
        : "=f"(frag[0]), "=f"(frag[1]), "=f"(frag[2]), "=f"(frag[3]),
          "=f"(frag[4]), "=f"(frag[5]), "=f"(frag[6]), "=f"(frag[7])
        : "r"(src)
        : "memory");
}

__global__ void gemv_ptx_kernel(
    const uint8_t* __restrict__ a,
    long long sa0,
    long long sa1,
    long long sa2,
    const uint8_t* __restrict__ b,
    long long sb1,
    long long sb2,
    const int8_t* __restrict__ sfa,
    long long ssfa0,
    long long ssfa1,
    long long ssfa2,
    const int8_t* __restrict__ sfb,
    long long ssfb1,
    long long ssfb2,
    at::Half* __restrict__ c,
    long long sc0,
    long long sc2,
    int M,
    int K_bytes,
    int L)
{
    const int tile_row = blockIdx.x * BLOCK_M;
    const int batch = blockIdx.y;
    if (tile_row >= M || batch >= L)
        return;

    __shared__ uint8_t smem_a[BLOCK_M * BLOCK_K / 2];
    __shared__ uint8_t smem_b[BLOCK_N * BLOCK_K / 2];
    __shared__ uint8_t smem_scale_a[BLOCK_M * (BLOCK_K / 16)];
    __shared__ uint8_t smem_scale_b[BLOCK_N * (BLOCK_K / 16)];
    __shared__ uint32_t shared_handles[3];

    const int lane = threadIdx.x;
    const int threads = blockDim.x;

    const uint8_t* tile_a_global = a + tile_row * sa0 + batch * sa2;
    const uint8_t* tile_b_global = b + batch * sb2;
    const int8_t* tile_sfa_global = sfa + tile_row * ssfa0 + batch * ssfa2;
    const int8_t* tile_sfb_global = sfb + batch * ssfb2;

    const int K = K_bytes * 2;
    const int tiles_k = K / BLOCK_K;

    uint32_t d_tmem_cols = ((BLOCK_M * BLOCK_N + 31) / 32) * 32;
    uint32_t scale_cols = ((BLOCK_K / 16) + 31) / 32 * 32;

    if (lane == 0) {
        shared_handles[0] = alloc_tmem(d_tmem_cols);
        shared_handles[1] = alloc_tmem(scale_cols);
        shared_handles[2] = alloc_tmem(scale_cols);
    }
    __syncthreads();

    const uint32_t d_tmem = shared_handles[0];
    const uint32_t scaleA_tmem = shared_handles[1];
    const uint32_t scaleB_tmem = shared_handles[2];

    uint32_t smem_base_a = __cvta_generic_to_shared(smem_a);
    uint32_t smem_base_b = __cvta_generic_to_shared(smem_b);
    uint32_t smem_base_sfa = __cvta_generic_to_shared(smem_scale_a);
    uint32_t smem_base_sfb = __cvta_generic_to_shared(smem_scale_b);

    uint64_t idesc = 0x0000000000000000ull;

    auto mma_op = [&](uint32_t k_tile) {
        for (int idx = lane; idx < BLOCK_M * BLOCK_K / 2; idx += threads) {
            int row = idx / (BLOCK_K / 2);
            int col = idx % (BLOCK_K / 2);
            smem_a[idx] = tile_a_global[row * sa0 + (k_tile * (BLOCK_K / 2)) + col * sa1];
        }
        for (int idx = lane; idx < BLOCK_N * BLOCK_K / 2; idx += threads) {
            int row = idx / (BLOCK_K / 2);
            int col = idx % (BLOCK_K / 2);
            smem_b[idx] = tile_b_global[row * sb1 + (k_tile * (BLOCK_K / 2)) + col * sb1];
        }
        for (int idx = lane; idx < BLOCK_M * (BLOCK_K / 16); idx += threads) {
            smem_scale_a[idx] = tile_sfa_global[idx + k_tile * (BLOCK_M * (BLOCK_K / 16))];
        }
        for (int idx = lane; idx < BLOCK_N * (BLOCK_K / 16); idx += threads) {
            smem_scale_b[idx] = tile_sfb_global[idx + k_tile * (BLOCK_N * (BLOCK_K / 16))];
        }
        __syncthreads();

        uint64_t a_desc = pack_smem_desc(smem_base_a, BLOCK_K / 2, BLOCK_K / 2);
        uint64_t b_desc = pack_smem_desc(smem_base_b, BLOCK_K / 2, BLOCK_K / 2);
        uint64_t scaleA_desc = pack_smem_desc(smem_base_sfa, BLOCK_K / 16, BLOCK_K / 16);
        uint64_t scaleB_desc = pack_smem_desc(smem_base_sfb, BLOCK_K / 16, BLOCK_K / 16);

        cp_to_tmem(scaleA_tmem, scaleA_desc);
        cp_to_tmem(scaleB_tmem, scaleB_desc);
        cp_to_tmem(d_tmem, a_desc);
        cp_to_tmem(d_tmem, b_desc);

        __syncthreads();

        asm volatile(
            "{ .reg .pred p; setp.ne.b32 p, %6, %6;\n"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n}"
            :
            : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
              "r"(scaleA_tmem), "r"(scaleB_tmem), "r"(0)
            : "memory");
        __syncthreads();
    };

    for (uint32_t k_tile = 0; k_tile < tiles_k; ++k_tile) {
        mma_op(k_tile);
    }

    if (lane < BLOCK_M) {
        float frag[8];
        ld_tmem(frag, d_tmem);
        float value = frag[0];
        at::Half* out_ptr = c + (tile_row + lane) * sc0 + batch * sc2;
        *out_ptr = __float2half(value);
    }

    __syncthreads();
    if (lane == 0) {
        dealloc_tmem(scaleB_tmem, scale_cols);
        dealloc_tmem(scaleA_tmem, scale_cols);
        dealloc_tmem(d_tmem, d_tmem_cols);
    }
}

torch::Tensor batched_scaled_gemv_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c)
{
    const int M = a.size(0);
    const int K_bytes = a.size(1);
    const int L = a.size(2);

    auto a_strides = a.strides();
    auto b_strides = b.strides();
    auto sfa_strides = sfa.strides();
    auto sfb_strides = sfb.strides();
    auto c_strides = c.strides();

    dim3 block(BLOCK_M);
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, L);

    gemv_ptx_kernel<<<grid, block>>>(
        reinterpret_cast<const uint8_t*>(a.data_ptr<int8_t>()),
        a_strides[0], a_strides[1], a_strides[2],
        reinterpret_cast<const uint8_t*>(b.data_ptr<int8_t>()),
        b_strides[1], b_strides[2],
        sfa.data_ptr<int8_t>(),
        sfa_strides[0], sfa_strides[1], sfa_strides[2],
        sfb.data_ptr<int8_t>(),
        sfb_strides[1], sfb_strides[2],
        c.data_ptr<at::Half>(),
        c_strides[0], c_strides[2],
        M,
        K_bytes,
        L);

    return c;
}
"""

module = load_inline(
    name="batched_scaled_gemv",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["batched_scaled_gemv_cuda"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-gencode=arch=compute_100,code=sm_100",
        "-gencode=arch=compute_110,code=sm_110"
    ],
    with_cuda=True,
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device

    a_i8 = a.to(device=device, copy=False).contiguous().view(torch.int8)
    b_i8 = b.to(device=device, copy=False).contiguous().view(torch.int8)
    sfa_i8 = sfa_ref.to(device=device, non_blocking=True).contiguous().view(torch.int8)
    sfb_i8 = sfb_ref.to(device=device, non_blocking=True).contiguous().view(torch.int8)

    module.batched_scaled_gemv_cuda(
        a_i8,
        b_i8,
        sfa_i8,
        sfb_i8,
        c,
    )

    return c
