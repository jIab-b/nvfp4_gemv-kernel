import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_source = """
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = """
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>

using namespace cute;

// Define NVFP4 blockscaled layout for B200/SM100
using ElementA = cutlass::nvfp4_t;  // NVFP4 E2M1 format
using ElementB = cutlass::nvfp4_t;
using ElementSF = cutlass::float_e4m3_t;  // FP8 E4M3FNUZ scale factors
using ElementC = cutlass::half_t;  // FP16 output
using ElementAccumulator = float;  // FP32 accumulation

// Layouts - K-major means column-major for row vectors
using LayoutA = cutlass::layout::ColumnMajor;  // K-major = column-major
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// Define the blockscaled layout for NVFP4
// Scale factors are per 16 elements along K dimension
constexpr int kBlockSize = 16;

// CollectiveBuilder for SM100 Blackwell architecture
using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,  // B200 GPU architecture
    cutlass::arch::OpClassTensorOp,  // Use Tensor Cores
    ElementA,
    LayoutA,
    cutlass::gemm::collective::ScaleType::Blockscale,  // NVFP4 blockscale
    ElementB,
    LayoutB,
    cutlass::gemm::collective::ScaleType::Blockscale,
    ElementC,
    LayoutC,
    ElementAccumulator,
    cute::Shape<_128, _128, _64>,  // Tile shape: M, N, K
    cute::Shape<_1, _1, _1>,  // Cluster shape
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename cutlass::arch::Sm100TensorMemory))
    >
>::CollectiveOp;

// Define the GEMM kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,  // Problem shape: M, N, K, L (batch)
    CollectiveMainloop,
    cutlass::epilogue::collective::DefaultEpilogue<
        cutlass::gemm::TagToStrideC_t<LayoutC>,
        cutlass::gemm::TagToStrideC_t<LayoutC>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 1, ElementAccumulator, ElementAccumulator
        >
    >
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Kernel launcher function
cudaError_t run_batched_fp4_matmul(
    int M, int K, int L,
    const ElementA* A,        // M x K x L in K-major
    const ElementSF* sfa,     // M x (K/16) x L scale factors
    const ElementB* B,        // 1 x K x L in K-major
    const ElementSF* sfb,     // 1 x (K/16) x L scale factors
    ElementC* C,              // M x 1 x L output
    cudaStream_t stream = 0
) {
    // Define problem size for each batch
    typename Gemm::Arguments arguments;

    // For batched matrix-vector multiply:
    // We treat this as batched GEMM with N=1
    arguments.problem_shape = {M, 1, K, L};  // M x 1 x K, batch L
    arguments.mainloop.ptr_A = A;
    arguments.mainloop.ptr_B = B;
    arguments.mainloop.ptr_scale_A = sfa;
    arguments.mainloop.ptr_scale_B = sfb;
    arguments.epilogue.ptr_C = nullptr;  // No C matrix (beta=0)
    arguments.epilogue.ptr_D = C;
    arguments.epilogue.alpha = 1.0f;
    arguments.epilogue.beta = 0.0f;

    // Set strides for K-major layout
    // K-major means elements are contiguous along K dimension
    arguments.mainloop.dA = {1, K};      // stride along M, K
    arguments.mainloop.dB = {1, K};      // B is vector, stride along K
    arguments.mainloop.dSFA = {1, K/16}; // scale factor stride
    arguments.mainloop.dSFB = {1, K/16};
    arguments.epilogue.dC = {1, M};      // output stride
    arguments.epilogue.dD = {1, M};

    // Batch strides (distance between batches)
    arguments.mainloop.batch_stride_A = M * K;
    arguments.mainloop.batch_stride_B = K;  // B is 1xKxL
    arguments.mainloop.batch_stride_SFA = M * (K/16);
    arguments.mainloop.batch_stride_SFB = K/16;
    arguments.epilogue.batch_stride_D = M;

    // Initialize and run
    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    cutlass::Status status = gemm_op.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    status = gemm_op.run(stream);

    if (workspace) cudaFree(workspace);

    return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);

    // Cast to CUTLASS types
    auto A_ptr = reinterpret_cast<const ElementA*>(a.data_ptr<uint8_t>());
    auto B_ptr = reinterpret_cast<const ElementB*>(b.data_ptr<uint8_t>());
    auto sfa_ptr = reinterpret_cast<const ElementSF*>(sfa.data_ptr<uint8_t>());
    auto sfb_ptr = reinterpret_cast<const ElementSF*>(sfb.data_ptr<uint8_t>());
    auto C_ptr = reinterpret_cast<ElementC*>(c.data_ptr<at::Half>());

    cudaError_t status = run_batched_fp4_matmul(M, K, L, A_ptr, sfa_ptr, B_ptr, sfb_ptr, C_ptr);
    if (status != cudaSuccess) {
        // Handle error - in real implementation you'd want proper error handling
        TORCH_CHECK(false, "CUTLASS kernel failed");
    }

    return c;
}
"""

module = load_inline(
    name='batched_scaled_gemv_cutlass',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batched_scaled_gemv_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17', '-gencode=arch=compute_100,code=sm_100'],
    extra_cflags=['-O3'],
    extra_ldflags=[],
    with_cuda=True,
    verbose=False
)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    return module.batched_scaled_gemv_cuda(
        a.view(torch.uint8),
        b.view(torch.uint8),
        sfa_ref.cuda().view(torch.uint8),
        sfb_ref.cuda().view(torch.uint8),
        c
    )
