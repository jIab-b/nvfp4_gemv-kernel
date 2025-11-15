import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/mma.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

namespace {

struct GemvParams {
  int m;
  int k;
  int batches;
};

struct SharedStorage {
  alignas(128) cute::ArrayEngine<cutlass::half_t, 128 * 64> a_tile;
  alignas(128) cute::ArrayEngine<cutlass::half_t, 64 * 64> b_tile;
  alignas(16) cute::uint32_t tmem_base;

  CUTE_DEVICE auto tensor_a() {
    return make_tensor(make_smem_ptr(a_tile.begin()), make_layout(make_shape(_128{}, _64{})));
  }

  CUTE_DEVICE auto tensor_b() {
    return make_tensor(make_smem_ptr(b_tile.begin()), make_layout(make_shape(_64{}, _64{})));
  }
};

__device__ __forceinline__ cutlass::half_t decode_fp4_elem(uint8_t nibble) {
  __nv_fp4_storage_t storage = static_cast<__nv_fp4_storage_t>(nibble & 0xF);
  __half_raw raw = __nv_cvt_fp4_to_halfraw(storage, __NV_E2M1);
  return __ushort_as_half(raw.x);
}

__device__ __forceinline__ float decode_fp8_scale(int8_t byte) {
  __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(byte);
  __half_raw raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
  return __half2float(__ushort_as_half(raw.x));
}

__global__ void cute_batched_gemv_kernel(const uint8_t* __restrict__ a,
                                         const uint8_t* __restrict__ b,
                                         const int8_t* __restrict__ sfa,
                                         const int8_t* __restrict__ sfb,
                                         cutlass::half_t* __restrict__ c,
                                         GemvParams params,
                                         int lda_k_packed,
                                         int ldb_k_packed,
                                         int ldsf_k) {
#if __CUDA_ARCH__ < 1000
  return;
#else
  extern __shared__ SharedStorage shared_storage[];
  SharedStorage& storage = shared_storage[0];

  const int tile_m = blockIdx.x * 128;
  const int batch_idx = blockIdx.y;

  if (tile_m >= params.m || batch_idx >= params.batches) {
    return;
  }

  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;

  const uint8_t* batch_a = a + static_cast<std::size_t>(batch_idx) * params.m * lda_k_packed;
  const uint8_t* batch_b = b + static_cast<std::size_t>(batch_idx) * ldb_k_packed;
  const int8_t* batch_sfa = sfa + static_cast<std::size_t>(batch_idx) * params.m * ldsf_k;
  const int8_t* batch_sfb = sfb + static_cast<std::size_t>(batch_idx) * ldsf_k;

  auto tmem_allocator = make_tmem_allocator();
  bool elect_one_warp = (warp == 0);

  if (elect_one_warp && lane == 0) {
    storage.tmem_base = tmem_allocator.alloc_columns(TmemAllocator::Sm100TmemCapacityColumns);
  }
  __syncthreads();

  constexpr int K_TILE = 64;
  int mma_iters = params.k / K_TILE;

  Tensor tCsA = storage.tensor_a();
  Tensor tCsB = storage.tensor_b();

  auto tiled_mma = make_tiled_mma(SM100_MMA_F16F16_SS<cutlass::half_t, cutlass::half_t, float,
                                                      128, 64, UMMA::Major::K, UMMA::Major::K>{});

  auto mma_shape = make_shape(tile_size<0>(tiled_mma), tile_size<1>(tiled_mma), Int<K_TILE>{});
  Tensor tCtAcc = make_tensor(tmem_allocator.make_tmem_ptr(storage.tmem_base), mma_shape);

  for (int iter = 0; iter < mma_iters; ++iter) {
    int k0 = iter * K_TILE;

    for (int idx = tid; idx < 128 * K_TILE; idx += blockDim.x) {
      int row = idx / K_TILE;
      int col = idx % K_TILE;
      int g_row = tile_m + row;
      if (g_row >= params.m) {
        continue;
      }
      int k = k0 + col;
      int pack = k / 2;
      int nibble = k & 1;
      uint8_t byte = batch_a[(static_cast<std::size_t>(g_row) * lda_k_packed) + pack];
      uint8_t fp4 = nibble ? (byte >> 4) : (byte & 0xF);
      float scale = decode_fp8_scale(batch_sfa[g_row * ldsf_k + (k / 16)]);
      cutlass::half_t value = __float2half(__half2float(decode_fp4_elem(fp4)) * scale);
      tCsA(row, col) = value;
    }

    for (int idx = tid; idx < 64 * K_TILE; idx += blockDim.x) {
      int row = idx / K_TILE;
      int col = idx % K_TILE;
      int k = k0 + col;
      if (row > 0 || k >= params.k) {
        continue;
      }
      int pack = k / 2;
      int nibble = k & 1;
      uint8_t byte = batch_b[pack];
      uint8_t fp4 = nibble ? (byte >> 4) : (byte & 0xF);
      float scale = decode_fp8_scale(batch_sfb[k / 16]);
      cutlass::half_t value = __float2half(__half2float(decode_fp4_elem(fp4)) * scale);
      tCsB(row, col) = value;
    }
    __syncthreads();

    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAs = thr_mma.partition_A(tCsA);
    auto tBs = thr_mma.partition_B(tCsB);
    auto tAcc = thr_mma.partition_C(tCtAcc);
    gemm(thr_mma, tAcc, tAs, tBs, tAcc);
    __syncthreads();
  }

  auto tmem_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  auto thr_copy = tmem_copy.get_slice(threadIdx.x);
  auto tAcc = thr_copy.partition_S(tCtAcc);

  for (int frag = 0; frag < size(tAcc); ++frag) {
    float value = tAcc[frag];
    int row_offset = frag;
    int g_row = tile_m + row_offset;
    if (g_row < params.m && lane == 0 && warp == 0) {
      c[batch_idx * params.m + g_row] = __float2half(value);
    }
  }

  __syncthreads();
  if (elect_one_warp && lane == 0) {
    tmem_allocator.free(storage.tmem_base, TmemAllocator::Sm100TmemCapacityColumns);
  }
#endif
}

} // anonymous namespace

torch::Tensor batched_scaled_gemv_cute(torch::Tensor a,
                                       torch::Tensor b,
                                       torch::Tensor sfa,
                                       torch::Tensor sfb,
                                       torch::Tensor c) {
  TORCH_CHECK(a.is_cuda(), "a must be CUDA");
  TORCH_CHECK(b.is_cuda(), "b must be CUDA");
  TORCH_CHECK(sfa.is_cuda(), "sfa must be CUDA");
  TORCH_CHECK(sfb.is_cuda(), "sfb must be CUDA");
  TORCH_CHECK(c.is_cuda(), "c must be CUDA");
  TORCH_CHECK(a.dtype() == torch::kUInt8, "a must be uint8 view");
  TORCH_CHECK(b.dtype() == torch::kUInt8, "b must be uint8 view");
  TORCH_CHECK(sfa.dtype() == torch::kInt8, "sfa must be int8 view");
  TORCH_CHECK(sfb.dtype() == torch::kInt8, "sfb must be int8 view");

  GemvParams params;
  params.m = static_cast<int>(a.size(0));
  params.k = static_cast<int>(a.size(1) * 2);
  params.batches = static_cast<int>(a.size(2));

  dim3 grid((params.m + 127) / 128, params.batches);
  dim3 block(128);

  auto cuda_stream = at::cuda::getCurrentCUDAStream();

  cute_batched_gemv_kernel<<<grid, block, sizeof(SharedStorage), cuda_stream>>>(
      a.data_ptr<uint8_t>(),
      b.data_ptr<uint8_t>(),
      sfa.data_ptr<int8_t>(),
      sfb.data_ptr<int8_t>(),
      reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()),
      params,
      static_cast<int>(a.size(1)),
      static_cast<int>(b.size(1)),
      static_cast<int>(sfa.size(1)));

  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batched_scaled_gemv_cute", &batched_scaled_gemv_cute, "CuTe-based NVFP4 GEMV");
}
"""

_module = load_inline(
    name="nvfp4_cute_kernel",
    sources=[_src],
    functions=["batched_scaled_gemv_cute"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        "-gencode=arch=compute_100a,code=sm_100a",
    ],
    with_cuda=True,
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    a, b, _, _, sfa_permuted, sfb_permuted, c = data
    device = a.device

    a_u8 = a.contiguous().view(torch.uint8)
    b_u8 = b.contiguous().view(torch.uint8)
    sfa_i8 = sfa_permuted.to(device=device, non_blocking=True).view(torch.int8)
    sfb_i8 = sfb_permuted.to(device=device, non_blocking=True).view(torch.int8)
    c = c.contiguous()

    _module.batched_scaled_gemv_cute(
        a_u8,
        b_u8,
        sfa_i8,
        sfb_i8,
        c,
    )
    return c
