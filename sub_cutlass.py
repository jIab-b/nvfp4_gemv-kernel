import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_cutlass(torch::Tensor a,
                                          torch::Tensor b,
                                          torch::Tensor sfa,
                                          torch::Tensor sfb,
                                          torch::Tensor c);
"""

cuda_source = r"""
#include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementScale = typename ElementA::ScaleFactorType;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementOutput = cutlass::half_t;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::ColumnMajor;
using LayoutDTag = cutlass::layout::ColumnMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 8;

using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using MmaTileShape = cute::Shape<cute::_128, cute::_64, cute::_64>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementOutput, LayoutCTag, AlignmentC,
    ElementOutput, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm100BlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm100BlkScaledConfig;

torch::Tensor batched_scaled_gemv_cutlass(torch::Tensor a,
                                          torch::Tensor b,
                                          torch::Tensor sfa,
                                          torch::Tensor sfb,
                                          torch::Tensor c) {
    TORCH_CHECK(a.is_cuda(), "Inputs must be CUDA tensors");

    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N = 1;

    using DataA = typename ElementA::DataType;
    using DataB = typename ElementB::DataType;

    auto* ptr_a = reinterpret_cast<DataA const*>(a.data_ptr<int8_t>());
    auto* ptr_b = reinterpret_cast<DataB const*>(b.data_ptr<int8_t>());
    auto* ptr_sfa = reinterpret_cast<ElementScale const*>(sfa.data_ptr<int8_t>());
    auto* ptr_sfb = reinterpret_cast<ElementScale const*>(sfb.data_ptr<int8_t>());
    auto* ptr_c = reinterpret_cast<ElementOutput*>(c.data_ptr<at::Half>());

    auto shape_A = cute::make_shape(M, K, L);
    auto shape_B = cute::make_shape(N, K, L);
    auto shape_C = cute::make_shape(M, N, L);

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_C);
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_C);

    LayoutSFA layout_SFA = Sm100BlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, L));
    LayoutSFB layout_SFB = Sm100BlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, L));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, L},
        {
            ptr_a, stride_A,
            ptr_b, stride_B,
            ptr_sfa, layout_SFA,
            ptr_sfb, layout_SFB
        },
        {
            {1.0f, 0.0f},
            ptr_c, stride_C,
            ptr_c, stride_D
        }
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM failed");
    return c;
}
"""

module = load_inline(
    name="batched_scaled_gemv_cutlass",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["batched_scaled_gemv_cutlass"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-I/usr/local/cuda/include",
        "-I/opt/cutlass/4.3.0/include",
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_110,code=sm_110",
    ],
    with_cuda=True,
    verbose=False
)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device

    a_nv = a.to(device=device, non_blocking=True).view(torch.int8)
    b_nv = b.to(device=device, non_blocking=True).view(torch.int8)
    sfa_nv = sfa_ref.to(device=device, non_blocking=True).view(torch.int8)
    sfb_nv = sfb_ref.to(device=device, non_blocking=True).view(torch.int8)

    return module.batched_scaled_gemv_cutlass(
        a_nv,
        b_nv,
        sfa_nv,
        sfb_nv,
        c
    )
