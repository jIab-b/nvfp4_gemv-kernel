import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_source = """
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kBlockThreads = kWarpSize * kWarpsPerBlock;
constexpr int kValuesPerScale = 16;
constexpr int kBytesPerScale = kValuesPerScale / 2; // two fp4 per byte

__device__ __forceinline__ float half_raw_to_float(const __half_raw& raw) {
    return __half2float(__ushort_as_half(raw.x));
}

__device__ __forceinline__ float decode_fp4(uint8_t nibble) {
    __half_raw h_raw = __nv_cvt_fp4_to_halfraw(
        static_cast<__nv_fp4_storage_t>(nibble & 0xF),
        __NV_E2M1);
    return half_raw_to_float(h_raw);
}

__device__ __forceinline__ float decode_fp8(uint8_t byte) {
    __half_raw h_raw = __nv_cvt_fp8_to_halfraw(
        static_cast<__nv_fp8_storage_t>(byte),
        __NV_E4M3);
    return half_raw_to_float(h_raw);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffffu;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void batched_scaled_gemv_kernel(
    const uint8_t* __restrict__ A,     // [M, K/2, L]
    const uint8_t* __restrict__ B,     // [1, K/2, L]
    const uint8_t* __restrict__ SFA,   // [M, K/16, L]
    const uint8_t* __restrict__ SFB,   // [1, K/16, L]
    half* __restrict__ C,              // [M, 1, L]
    int M,
    int K_half,
    int K_groups,
    int batch_stride_A,
    int batch_stride_B,
    int batch_stride_SFA,
    int batch_stride_SFB,
    int batch_stride_C) {

    int batch = blockIdx.z;
    int warp_id = threadIdx.x / kWarpSize;
    int lane = threadIdx.x % kWarpSize;
    int rows_per_block = kWarpsPerBlock;
    int row = blockIdx.x * rows_per_block + warp_id;
    if (row >= M) {
        return;
    }

    const uint8_t* batch_A = A + static_cast<size_t>(batch) * batch_stride_A;
    const uint8_t* batch_B = B + static_cast<size_t>(batch) * batch_stride_B;
    const uint8_t* batch_SFA = SFA + static_cast<size_t>(batch) * batch_stride_SFA;
    const uint8_t* batch_SFB = SFB + static_cast<size_t>(batch) * batch_stride_SFB;
    half* batch_C = C + static_cast<size_t>(batch) * batch_stride_C;

    const uint8_t* row_A = batch_A + static_cast<size_t>(row) * K_half;
    const uint8_t* row_SFA = batch_SFA + static_cast<size_t>(row) * K_groups;

    float accum = 0.f;

    for (int group = lane; group < K_groups; group += kWarpSize) {
        float scale_a = decode_fp8(row_SFA[group]);
        float scale_b = decode_fp8(batch_SFB[group]);
        float scale = scale_a * scale_b;

        const uint8_t* vec_A = row_A + group * kBytesPerScale;
        const uint8_t* vec_B = batch_B + group * kBytesPerScale;

#pragma unroll
        for (int idx = 0; idx < kBytesPerScale; ++idx) {
            uint8_t a_byte = vec_A[idx];
            uint8_t b_byte = vec_B[idx];

            float a_lo = decode_fp4(a_byte & 0xF);
            float b_lo = decode_fp4(b_byte & 0xF);
            accum += a_lo * b_lo * scale;

            float a_hi = decode_fp4(a_byte >> 4);
            float b_hi = decode_fp4(b_byte >> 4);
            accum += a_hi * b_hi * scale;
        }
    }

    float reduced = warp_reduce_sum(accum);
    if (lane == 0) {
        batch_C[row] = __float2half(reduced);
    }
}

} // anonymous namespace

torch::Tensor batched_scaled_gemv_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && sfa.is_cuda() && sfb.is_cuda(),
                "All tensors must be CUDA.");
    TORCH_CHECK(a.scalar_type() == torch::kUInt8, "A must be uint8 view.");
    TORCH_CHECK(b.scalar_type() == torch::kUInt8, "B must be uint8 view.");
    TORCH_CHECK(sfa.scalar_type() == torch::kUInt8, "SFA must be uint8 view.");
    TORCH_CHECK(sfb.scalar_type() == torch::kUInt8, "SFB must be uint8 view.");
    TORCH_CHECK(c.scalar_type() == at::kHalf, "Output must be fp16.");

    int64_t M = a.size(0);
    int64_t K_half = a.size(1);
    int64_t L = a.size(2);
    int64_t K = K_half * 2;
    TORCH_CHECK(b.size(0) == 1, "This kernel expects N=1 for GEMV.");
    TORCH_CHECK(b.size(1) == K_half && b.size(2) == L, "B shape mismatch.");
    TORCH_CHECK(sfa.size(0) == M && sfa.size(1) * kValuesPerScale == K && sfa.size(2) == L,
                "Scale tensor A mismatch.");
    TORCH_CHECK(sfb.size(0) == 1 && sfb.size(1) * kValuesPerScale == K && sfb.size(2) == L,
                "Scale tensor B mismatch.");
    TORCH_CHECK(c.size(0) == M && c.size(1) == 1 && c.size(2) == L,
                "Output tensor shape mismatch.");

    int K_groups = static_cast<int>(K / kValuesPerScale);
    int batch_stride_A = static_cast<int>(M * K_half);
    int batch_stride_B = static_cast<int>(K_half);
    int batch_stride_SFA = static_cast<int>(M * K_groups);
    int batch_stride_SFB = static_cast<int>(K_groups);
    int batch_stride_C = static_cast<int>(M);

    const dim3 block(kBlockThreads);
    const dim3 grid(
        static_cast<unsigned int>((M + kWarpsPerBlock - 1) / kWarpsPerBlock),
        1,
        static_cast<unsigned int>(L));

    int shared_mem = 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    batched_scaled_gemv_kernel<<<grid, block, shared_mem, stream>>>(
        a.data_ptr<uint8_t>(),
        b.data_ptr<uint8_t>(),
        sfa.data_ptr<uint8_t>(),
        sfb.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        static_cast<int>(M),
        static_cast<int>(K_half),
        K_groups,
        batch_stride_A,
        batch_stride_B,
        batch_stride_SFA,
        batch_stride_SFB,
        batch_stride_C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
)"""

module = load_inline(
    name="batched_scaled_gemv_cutlass",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["batched_scaled_gemv_cuda"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_100,code=sm_100"
    ],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device

    a_u8 = a.view(torch.uint8)
    b_u8 = b.view(torch.uint8)
    sfa_u8 = sfa_ref.to(device=device, non_blocking=True).view(torch.uint8)
    sfb_u8 = sfb_ref.to(device=device, non_blocking=True).view(torch.uint8)

    return module.batched_scaled_gemv_cuda(a_u8, b_u8, sfa_u8, sfb_u8, c)
