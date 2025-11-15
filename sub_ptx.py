import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_ptx(torch::Tensor a,
                                      torch::Tensor b,
                                      torch::Tensor sfa,
                                      torch::Tensor sfb,
                                      torch::Tensor c);
"""

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda/ptx>

using namespace cuda::ptx;

namespace {

struct SharedStorage {
    __align__(16) uint8_t a_tile[128 * 64 / 2];
    __align__(16) uint8_t b_tile[8 * 64 / 2];
    __align__(16) uint8_t sfa_tile[128 * 64 / 16];
    __align__(16) uint8_t sfb_tile[8 * 64 / 16];
    __align__(16) uint32_t tmem_addr;
};

__device__ __forceinline__ uint64_t matrix_desc_encode(uint32_t byte_addr) {
    return static_cast<uint64_t>((byte_addr & 0x3FFFFu) >> 4);
}

__device__ uint64_t make_smem_desc(uint32_t base, uint32_t lbo, uint32_t sbo, int swizzle) {
    uint64_t desc = 0;
    desc |= matrix_desc_encode(base) & 0x3FFFu;
    desc |= (matrix_desc_encode(lbo) & 0x3FFFu) << 16;
    desc |= (matrix_desc_encode(sbo) & 0x3FFFu) << 32;
    desc |= (uint64_t)0b001 << 46;
    desc |= (uint64_t)0b000 << 49;
    desc |= (uint64_t)0xB0 << 53;  // Guess from Table 40, to be validated on hardware.
    desc |= (uint64_t)(swizzle & 0x7) << 61;
    return desc;
}

__device__ uint32_t make_idesc(int m_tile, int n_tile, bool scale_is_e4m3) {
    uint32_t idesc = 0;
    idesc |= (1u & 0x7u) << 7;  // atype = nvfp4
    idesc |= (1u & 0x3u) << 10; // btype = nvfp4 (assumed)
    idesc |= static_cast<uint32_t>((n_tile >> 3) & 0x3Fu) << 17;
    idesc |= (scale_is_e4m3 ? 0u : 1u) << 23;
    idesc |= static_cast<uint32_t>((m_tile >> 7) & 0x3u) << 27;
    return idesc;
}

__global__ void gemv_nvfp4_ptx_kernel(const int8_t* __restrict__ a,
                                      const int8_t* __restrict__ b,
                                      const int8_t* __restrict__ sfa,
                                      const int8_t* __restrict__ sfb,
                                      half* __restrict__ c,
                                      int M, int K, int L) {
#if __CUDA_ARCH__ < 1000
    asm("trap;"); // Running on non-Blackwell hardware is not supported.
    return;
#endif
    extern __shared__ SharedStorage smem[];
    SharedStorage& storage = smem[0];

    constexpr int Mtile = 128;
    constexpr int Ntile = 8;
    constexpr int Ktile = 64;

    const int tile_m = blockIdx.x * Mtile;
    const int tile_n = blockIdx.y * Ntile;
    if (tile_m >= M || tile_n >= L) {
        return;
    }

    auto copy_tile = [&](const uint8_t* src_global, uint8_t* dst, size_t bytes_per_vec) {
        for (size_t idx = threadIdx.x; idx < bytes_per_vec; idx += blockDim.x) {
            dst[idx] = src_global[idx];
        }
    };

    const size_t slice_bytes_a = static_cast<size_t>(M) * (K / 2);
    const size_t slice_bytes_sfa = static_cast<size_t>(M) * (K / 16);

    for (int n = 0; n < Ntile && (tile_n + n) < L; ++n) {
        const int global_n = tile_n + n;
        const uint8_t* src_b = reinterpret_cast<const uint8_t*>(b) + static_cast<size_t>(global_n) * (K / 2);
        const uint8_t* src_sfb = reinterpret_cast<const uint8_t*>(sfb) + static_cast<size_t>(global_n) * (K / 16);
        copy_tile(src_b, storage.b_tile + n * (Ktile / 2), Ktile / 2);
        copy_tile(src_sfb, storage.sfb_tile + n * (Ktile / 16), Ktile / 16);
    }

    for (int m = 0; m < Mtile && (tile_m + m) < M; ++m) {
        const int global_m = tile_m + m;
        const uint8_t* src_a = reinterpret_cast<const uint8_t*>(a) +
                               tile_n * slice_bytes_a +
                               static_cast<size_t>(global_m) * (K / 2);
        const uint8_t* src_sfa = reinterpret_cast<const uint8_t*>(sfa) +
                                 tile_n * slice_bytes_sfa +
                                 static_cast<size_t>(global_m) * (K / 16);
        copy_tile(src_a, storage.a_tile + m * (Ktile / 2), Ktile / 2);
        copy_tile(src_sfa, storage.sfa_tile + m * (Ktile / 16), Ktile / 16);
    }
    __syncthreads();

    // Build shared-memory descriptors (best-effort guesses; verify on hardware).
    const uint32_t a_base = __cvta_generic_to_shared(storage.a_tile);
    const uint32_t b_base = __cvta_generic_to_shared(storage.b_tile);
    const uint32_t sfa_base = __cvta_generic_to_shared(storage.sfa_tile);
    const uint32_t sfb_base = __cvta_generic_to_shared(storage.sfb_tile);

    const uint64_t a_desc = make_smem_desc(a_base, Ktile / 2, (Ktile / 2) * 8, 2);
    const uint64_t b_desc = make_smem_desc(b_base, Ktile / 2, (Ktile / 2) * 8, 2);
    const uint64_t sfa_desc = make_smem_desc(sfa_base, Ktile / 16, (Ktile / 16) * 8, 0);
    const uint64_t sfb_desc = make_smem_desc(sfb_base, Ktile / 16, (Ktile / 16) * 8, 0);

    const uint32_t idesc = make_idesc(Mtile, Ntile, /*scale_is_e4m3=*/true);

    if (threadIdx.x == 0) {
        const uint32_t cols = 512;
        tcgen05_alloc(cta_group_1, &storage.tmem_addr, cols);
    }
    __syncthreads();

    const uint32_t tmem_acc = storage.tmem_addr;
    const uint32_t tmem_scale_a = tmem_acc + 256;  // Column offsets guessed.
    const uint32_t tmem_scale_b = tmem_acc + 384;

    if (threadIdx.x == 0) {
        tcgen05_cp_4x256b(cta_group_1, tmem_scale_a, sfa_desc);
        tcgen05_cp_4x256b(cta_group_1, tmem_scale_b, sfb_desc);
        tcgen05_mma_block_scale_vec_2x(kind_mxf4nvf4_t{}, cta_group_1,
                                       tmem_acc, a_desc, b_desc, idesc,
                                       tmem_scale_a, tmem_scale_b, false);
    }
    __syncthreads();

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    const int rows_per_warp = 32;

    uint32_t ld_regs[1];

    for (int n_offset = 0; n_offset < Ntile; ++n_offset) {
        const int global_n = tile_n + n_offset;
        if (global_n >= L) break;

        int global_row = tile_m + warp * rows_per_warp + lane;
        if (global_row >= M) continue;

        // Best-guess TMEM addressing: combine lane bits with column.
        const uint32_t column = (tmem_acc & 0xFFFFu) + n_offset;
        const uint32_t lane_bits = static_cast<uint32_t>((warp * rows_per_warp + lane) & 0x7Fu) << 16;
        // EXPECTED FAILURE: TMEM addressing model is assumed; hardware may require different lane encoding.
        const uint32_t taddr = (tmem_acc & 0xFFFF0000u) | lane_bits | (column & 0xFFFFu);
        tcgen05_ld_32x32b(ld_regs, taddr);

        half out_val = __float2half(__uint_as_float(ld_regs[0]));
        size_t c_idx = static_cast<size_t>(global_n) * M + global_row;
        c[c_idx] = out_val;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        tcgen05_dealloc(cta_group_1, tmem_acc, 512);
        tcgen05_relinquish_alloc_permit(cta_group_1);
    }
}

} // namespace

torch::Tensor batched_scaled_gemv_ptx(torch::Tensor a,
                                      torch::Tensor b,
                                      torch::Tensor sfa,
                                      torch::Tensor sfb,
                                      torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);

    dim3 grid((M + 127) / 128, (L + 7) / 8);
    dim3 block(128);
    size_t smem_bytes = sizeof(SharedStorage);

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    gemv_nvfp4_ptx_kernel<<<grid, block, smem_bytes>>>(
        a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, M, K, L);
    return c;
}
"""

module = load_inline(
    name="batched_scaled_gemv_ptx",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["batched_scaled_gemv_ptx"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
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

    return module.batched_scaled_gemv_ptx(
        a_i8,
        b_i8,
        sfa_i8,
        sfb_i8,
        c
    )
