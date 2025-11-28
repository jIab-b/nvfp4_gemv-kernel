import torch
from task import input_t, output_t

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import tcgen05


# ----------------------------
# Kernel configuration
# ----------------------------
mma_tiler_mnk = (128, 8, 64)  # M, N, K tile for tcgen05 mxf4nvf4
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16
sf_vec_size = 16  # 16 fp4 elems share one scale (4 scales per K=64 tile)
threads_per_cta = 128  # one warpgroup


def ceil_div(a, b):
    return (a + b - 1) // b


@cute.kernel
def kernel_tcgen(
    mA_mkl: cute.Tensor,
    mB_nkl: cute.Tensor,
    mSFA_mkl: cute.Tensor,
    mSFB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
):
    # Block and thread indices
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    # Local tiles from GMEM
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None, None)
    )

    # K tiles to process
    k_tile_cnt = gA_mkl.layout[3].shape

    # ----------------------------
    # Allocate SMEM (static)
    # Layouts chosen to match tcgen05 SMEM descriptors: A row-major 32B stride, B 32B stride
    # SFA/SFB use 16B stride per scale (4 scales per tile)
    # ----------------------------
    smem_a_ptr = cute.arch.alloc_smem(ab_dtype, mma_tiler_mnk[0] * mma_tiler_mnk[2])
    smem_b_ptr = cute.arch.alloc_smem(ab_dtype, mma_tiler_mnk[1] * mma_tiler_mnk[2])
    # scales: 4 scales per tile, each replicated at 16B stride → 64 bytes per row
    smem_sfa_ptr = cute.arch.alloc_smem(sf_dtype, mma_tiler_mnk[0] * 4 * 16)
    smem_sfb_ptr = cute.arch.alloc_smem(sf_dtype, mma_tiler_mnk[1] * 4 * 16)
    smem_acc_ptr = cute.arch.alloc_smem(cutlass.Float32, mma_tiler_mnk[0] * mma_tiler_mnk[1])

    # SMEM tensors with explicit layouts
    sA = cute.make_tensor(
        smem_a_ptr,
        cute.make_layout((mma_tiler_mnk[0], mma_tiler_mnk[2]), stride=(mma_tiler_mnk[2], 1)),
    )
    sB = cute.make_tensor(
        smem_b_ptr,
        cute.make_layout((mma_tiler_mnk[1], mma_tiler_mnk[2]), stride=(mma_tiler_mnk[2], 1)),
    )
    # 16B stride per scale entry (4 scales => 64 bytes per row)
    sSFA = cute.make_tensor(
        smem_sfa_ptr,
        cute.make_layout((mma_tiler_mnk[0], 4), stride=(64, 16)),
    )
    sSFB = cute.make_tensor(
        smem_sfb_ptr,
        cute.make_layout((mma_tiler_mnk[1], 4), stride=(64, 16)),
    )
    sAcc = cute.make_tensor(
        smem_acc_ptr,
        cute.make_layout((mma_tiler_mnk[0], mma_tiler_mnk[1]), stride=(mma_tiler_mnk[1], 1)),
    )

    # Tiled MMA op (block-scaled nvf4)
    mma_op = tcgen05.MmaMXF4NVF4Op(
        sf_dtype,
        mma_tiler_mnk,
        tcgen05.mma.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)
    thr_mma = tiled_mma.get_slice(tidx)

    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCsAcc = thr_mma.partition_C(sAcc)

    for kt in range(k_tile_cnt):
        # Cooperative load whole tiles into SMEM
        gA_tile = gA_mkl[:, :, bidx, kt, bidz]
        gSFA_tile = gSFA_mkl[:, :, bidx, kt, bidz]
        gB_tile = gB_nkl[:, :, bidy, kt, bidz]
        gSFB_tile = gSFB_nkl[:, :, bidy, kt, bidz]

        cute.copy(gA_tile, sA)
        cute.copy(gB_tile, sB)
        cute.copy(gSFA_tile, sSFA)
        cute.copy(gSFB_tile, sSFB)

        cute.arch.sync_threads()

        tiled_mma.set(tcgen05.Field.ACCUMULATE, kt != 0)
        cute.gemm(tiled_mma, tCsAcc, tCsA, tCsB, tCsAcc)

    cute.arch.sync_threads()

    # Write column 0 of accumulator to C (GEMV)
    if tidx < mma_tiler_mnk[0]:
        tCgC = gC_mnl[tidx, 0, bidx, bidy, bidz]
        tCgC.store(sAcc[tidx, 0].to(c_dtype))


@cute.jit
def my_kernel(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, problem_size):
    m, _, k, l = problem_size
    a_tensor = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (m, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
        ),
    )
    n_pad = mma_tiler_mnk[1]
    b_tensor = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (n_pad, cute.assume(k, 32), l),
            stride=(0, 1, cute.assume(n_pad * k, 32)),
        ),
    )
    c_tensor = cute.make_tensor(
        c_ptr, cute.make_layout((cute.assume(m, 32), 1, l), stride=(1, 1, m))
    )

    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, sf_vec_size)
    sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, sf_vec_size)
    sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

    grid = (ceil_div(c_tensor.shape[0], mma_tiler_mnk[0]), 1, c_tensor.shape[2])

    kernel_tcgen(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(1, 1, 1),
    )


_compiled_kernel = None


def compile_kernel():
    global _compiled_kernel
    if _compiled_kernel is not None:
        return _compiled_kernel

    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

    _compiled_kernel = cute.compile(
        my_kernel, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0)
    )
    return _compiled_kernel


def custom_kernel(data: input_t) -> output_t:
    a, b, _, _, sfa_perm, sfb_perm, c = data
    compiled = compile_kernel()

    m, k_half, l = a.shape
    k = k_half * 2

    a_ptr = make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, sfa_perm.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, sfb_perm.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    compiled(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, 1, k, l))
    return c
