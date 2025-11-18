"""
Batched persistent GEMV kernel for NVIDIA Blackwell (SM100) with NVFP4 quantization.
Persistent scheduling with vector storage in TMEM, reused across M-tiles per K-block.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils


class GemvConfig:
    """Configuration for persistent batched GEMV kernel."""

    def __init__(self, mma_tiler_m=128, block_k=64, sf_vec_size=16, num_ab_stages=2):
        self.mma_tiler_m = mma_tiler_m
        self.block_k = block_k
        self.sf_vec_size = sf_vec_size
        self.num_ab_stages = num_ab_stages
        self.acc_dtype = cutlass.Float32
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.c_dtype = cutlass.Float16
        self.cta_group = tcgen05.CtaGroup.ONE


_config = GemvConfig()


@cute.kernel
def gemv_persistent_kernel(a, b, sfa, sfb, c):
    """Persistent GEMV kernel with vector storage in TMEM."""

    # Wrap tensors (MLIR context available inside @cute.kernel)
    a_tensor = cute.make_tensor(a)
    b_tensor = cute.make_tensor(b)
    sfa_tensor = cute.make_tensor(sfa)
    sfb_tensor = cute.make_tensor(sfb)
    c_tensor = cute.make_tensor(c)

    config = _config

    # Create tiled MMA (MLIR operations here, context available)
    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        ab_dtype=config.ab_dtype,
        a_major_mode="K",
        b_major_mode="K",
        sf_dtype=config.sf_dtype,
        sf_vec_size=config.sf_vec_size,
        cta_group=config.cta_group,
        mma_inst_shape_mn=(config.mma_tiler_m, 1),
    )

    M, K, L = a_tensor.shape
    m_tiles = (M + config.mma_tiler_m - 1) // config.mma_tiler_m
    k_tiles = (K + config.block_k - 1) // config.block_k

    thread_idx = cute.arch.thread_idx()
    block_idx = cute.arch.block_idx()
    warp_idx = thread_idx // 32

    # Allocate SMEM for A and SFA staging
    sA_size = config.mma_tiler_m * config.block_k * config.num_ab_stages
    sSFA_size = config.mma_tiler_m * (config.block_k // config.sf_vec_size) * config.num_ab_stages

    sA_ptr = cute.arch.alloc_smem(config.ab_dtype, sA_size)
    sSFA_ptr = cute.arch.alloc_smem(config.sf_dtype, sSFA_size)

    # Allocate TMEM for persistent vector b, scales sfb, and accumulator
    tmem_addr_buffer = cute.arch.alloc_smem(cutlass.Int64, 4)
    total_tmem_cols = config.block_k + (config.block_k // config.sf_vec_size) + config.mma_tiler_m
    cute.arch.alloc_tmem(num_columns=total_tmem_cols, smem_ptr_to_write_address=tmem_addr_buffer)

    # Retrieve TMEM pointers
    b_tmem_ptr = cute.arch.retrieve_tmem_ptr(config.ab_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_addr_buffer)
    sfb_tmem_ptr = cute.arch.retrieve_tmem_ptr(
        config.sf_dtype, alignment=8,
        ptr_to_buffer_holding_addr=cute.recast_ptr(tmem_addr_buffer + config.block_k)
    )
    accum_tmem_ptr = cute.arch.retrieve_tmem_ptr(
        config.acc_dtype, alignment=16,
        ptr_to_buffer_holding_addr=cute.recast_ptr(
            tmem_addr_buffer + config.block_k + (config.block_k // config.sf_vec_size)
        )
    )

    # Make TMEM tensors
    b_tmem_layout = cute.make_layout(cute.make_shape(config.block_k), cute.make_stride(1))
    b_tmem_tensor = cute.make_tensor(b_tmem_ptr, b_tmem_layout)

    sfb_tmem_layout = cute.make_layout(
        cute.make_shape(config.block_k // config.sf_vec_size),
        cute.make_stride(1)
    )
    sfb_tmem_tensor = cute.make_tensor(sfb_tmem_ptr, sfb_tmem_layout)

    accum_tmem_layout = cute.make_layout(cute.make_shape(config.mma_tiler_m), cute.make_stride(1))
    accum_tmem_tensor = cute.make_tensor(accum_tmem_ptr, accum_tmem_layout)

    # Initialize accumulator
    for i in range(config.mma_tiler_m):
        accum_tmem_tensor[i] = config.acc_dtype(0.0)

    barrier_id = cute.arch.barrier(barrier_id=1)

    # === MAIN K-LOOP ===
    for k_tile_idx in range(k_tiles):
        k_offset = k_tile_idx * config.block_k
        k_size = min(config.block_k, K - k_offset)

        # TMA/DMA Warp: Load persistent vector b and scales to TMEM
        if warp_idx == 1:
            for lane_idx in range(32):
                if lane_idx < k_size and lane_idx < config.block_k:
                    b_tmem_tensor[lane_idx] = b_tensor[k_offset + lane_idx, 0]

            sfb_idx_start = k_offset // config.sf_vec_size
            sfb_idx_end = (k_offset + k_size + config.sf_vec_size - 1) // config.sf_vec_size
            for lane_idx in range(32):
                if lane_idx < (sfb_idx_end - sfb_idx_start) and lane_idx < (config.block_k // config.sf_vec_size):
                    sfb_tmem_tensor[lane_idx] = sfb_tensor[sfb_idx_start + lane_idx, 0]

        cute.arch.barrier_arrive(barrier_id)
        cute.arch.barrier(barrier_id)

        # === PERSISTENT M-LOOP ===
        m_block_idx = block_idx[0]

        if m_block_idx < m_tiles:
            m_offset = m_block_idx * config.mma_tiler_m
            m_size = min(config.mma_tiler_m, M - m_offset)

            # Load A and SFA to SMEM
            if warp_idx == 1:
                for m_idx in range(m_size):
                    for k_idx in range(k_size):
                        a_val = a_tensor[m_offset + m_idx, k_offset + k_idx, 0]
                        sA_ptr_offset = (m_idx * config.block_k + k_idx)
                        sA_ptr[sA_ptr_offset] = a_val

                sfa_idx_start = k_offset // config.sf_vec_size
                sfa_idx_end = (k_offset + k_size + config.sf_vec_size - 1) // config.sf_vec_size
                for m_idx in range(m_size):
                    for sfa_idx in range(sfa_idx_end - sfa_idx_start):
                        sfa_val = sfa_tensor[m_offset + m_idx, sfa_idx_start + sfa_idx, 0]
                        sSFA_ptr_offset = (m_idx * (config.block_k // config.sf_vec_size) + sfa_idx)
                        sSFA_ptr[sSFA_ptr_offset] = sfa_val

            cute.arch.barrier_arrive(barrier_id)
            cute.arch.barrier(barrier_id)

            # MMA Warp: Execute block-scaled matrix-vector multiply
            if warp_idx == 0:
                a_smem_tensor = cute.make_tensor(
                    sA_ptr,
                    cute.make_layout(
                        cute.make_shape(config.mma_tiler_m, config.block_k),
                        cute.make_stride(config.block_k, 1)
                    )
                )
                sfa_smem_tensor = cute.make_tensor(
                    sSFA_ptr,
                    cute.make_layout(
                        cute.make_shape(config.mma_tiler_m, config.block_k // config.sf_vec_size),
                        cute.make_stride(config.block_k // config.sf_vec_size, 1)
                    )
                )

                thr_mma = tiled_mma.get_slice(thread_idx)
                a_partition = thr_mma.partition_A(a_smem_tensor)
                sfa_partition = thr_mma.partition_A(sfa_smem_tensor)

                tiled_mma.set(tcgen05.Field.SFA, sfa_partition)
                tiled_mma.set(tcgen05.Field.SFB, sfb_tmem_tensor)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, (k_tile_idx > 0))

                cute.gemm(tiled_mma, accum_tmem_tensor, a_partition, b_tmem_tensor, accum_tmem_tensor)

            cute.arch.barrier_arrive(barrier_id)
            cute.arch.barrier(barrier_id)

        # === EPILOGUE: Store accumulated results ===
        if m_block_idx < m_tiles and k_tile_idx == k_tiles - 1:
            m_offset = m_block_idx * config.mma_tiler_m
            m_size = min(config.mma_tiler_m, M - m_offset)

            if warp_idx < 2:
                for m_idx in range(m_size):
                    c_val = accum_tmem_tensor[m_idx]
                    c_tensor[m_offset + m_idx, 0, 0] = c_val

    # Cleanup
    cute.arch.dealloc_tmem(b_tmem_ptr, num_columns=total_tmem_cols)


def custom_kernel(data) -> torch.Tensor:
    """Launch persistent GEMV kernel."""
    a, b, sfa, sfb, _, _, c = data

    gemv_persistent_kernel(a, b, sfa, sfb, c).launch(
        grid=(
            (a.shape[0] + 128 - 1) // 128,
            1,
            a.shape[2],
        ),
        block=(128, 1, 1),
    )

    return c
