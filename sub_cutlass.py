"""
Batched persistent GEMV kernel with persistent vector storage in TMEM.
Optimized for NVIDIA Blackwell (SM100) with NVFP4 quantization + block-scaled factors.

Problem: c[M, 1, L] = a[M, K, L] @ b[1, K, L] with scale factors
  - a, b: NVFP4 (e2m1), K-major layout
  - sfa, sfb: FP8 (e4m3fnuz), 1 scale per 16 values
  - Output c: FP16

Key optimization: Vector b[K, L] and sfb scales stay in TMEM,
reused across all M-tile computations.
"""

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass._mlir.dialects import llvm
from typing import Tuple
import math


class GemvPersistentConfig:
    """Configuration for persistent batched GEMV kernel."""

    def __init__(
        self,
        mma_tiler_m: int = 128,        # MMA tile M dimension
        block_k: int = 64,              # K dimension tile
        sf_vec_size: int = 16,          # Scale factor vector size (NVFP4 uses 16)
        cluster_shape_m: int = 1,       # Cluster M dimension
        cluster_shape_n: int = 1,       # Cluster N dimension (always 1 for GEMV)
        num_ab_stages: int = 2,         # Pipeline stages for A/B loads
        num_acc_stages: int = 1,        # Pipeline stages for accumulators
        smem_capacity_bytes: int = 98304,  # SM100 shared memory
    ):
        self.mma_tiler_m = mma_tiler_m
        self.block_k = block_k
        self.sf_vec_size = sf_vec_size
        self.cluster_shape_m = cluster_shape_m
        self.cluster_shape_n = cluster_shape_n
        self.num_ab_stages = num_ab_stages
        self.num_acc_stages = num_acc_stages
        self.smem_capacity = smem_capacity_bytes

        # Derived settings
        self.acc_dtype = cutlass.Float32
        self.ab_dtype = cutlass.Float4E2M1FN  # NVFP4
        self.sf_dtype = cutlass.Float8E4M3FN   # FP8 for scales
        self.c_dtype = cutlass.Float16
        self.cta_group = tcgen05.CtaGroup.ONE

        # Warp IDs for specialization
        self.tma_warp_id = 5
        self.mma_warp_id = 4
        self.epi_warp_ids = (0, 1, 2, 3)


def _compute_tile_scheduler_params(m_tiles: int, k_tiles: int, cluster_m: int, max_active_clusters: int):
    """Compute persistent tile scheduler parameters."""
    # Total tiles to schedule
    total_tiles = m_tiles * k_tiles

    # Cluster-level scheduling
    clusters_m = (m_tiles + cluster_m - 1) // cluster_m
    ctas_per_cluster = cluster_m

    return {
        'total_m_tiles': m_tiles,
        'total_k_tiles': k_tiles,
        'clusters_m': clusters_m,
        'ctas_per_cluster': ctas_per_cluster,
        'total_tiles': total_tiles,
    }


def _compute_smem_layout_a(config: GemvPersistentConfig, mma_tiler: Tuple[int, int, int]):
    """Layout for matrix A in SMEM: (M_tile, K_tile, stages)."""
    m, k, stages = mma_tiler[0], mma_tiler[2], config.num_ab_stages
    # Swizzle-friendly layout for bank conflict avoidance
    return cute.make_layout(
        cute.make_shape(m, k, stages),
        cute.make_stride(k, 1, m * k)  # K-major, stage-contiguous
    )


def _compute_smem_layout_sfa(config: GemvPersistentConfig, mma_tiler: Tuple[int, int, int]):
    """Layout for scale factors SFA in SMEM: (M_tile, K_tile÷16, stages)."""
    m, k_tiles, stages = mma_tiler[0], mma_tiler[2] // config.sf_vec_size, config.num_ab_stages
    return cute.make_layout(
        cute.make_shape(m, k_tiles, stages),
        cute.make_stride(k_tiles, 1, m * k_tiles)
    )


def _make_tiled_mma(config: GemvPersistentConfig, mma_shape_mnk: Tuple[int, int, int]):
    """Create tiled MMA descriptor for block-scaled NVFP4."""
    return cutlass.cute.nvgpu.sm100_utils.make_blockscaled_trivial_tiled_mma(
        ab_dtype=config.ab_dtype,
        a_major_mode="K",  # K-major for A
        b_major_mode="K",  # K-major for B (vector)
        sf_dtype=config.sf_dtype,
        sf_vec_size=config.sf_vec_size,
        cta_group=config.cta_group,
        mma_inst_shape_mn=(mma_shape_mnk[0], mma_shape_mnk[1]),
    )


class GemvPersistentKernel:
    """Persistent batched GEMV kernel with vector storage in TMEM."""

    def __init__(self, config: GemvPersistentConfig):
        self.config = config
        self.mma_tiler = (config.mma_tiler_m, 1, config.block_k)
        self.tiled_mma = _make_tiled_mma(config, self.mma_tiler)
        self.smem_layout_a = _compute_smem_layout_a(config, self.mma_tiler)
        self.smem_layout_sfa = _compute_smem_layout_sfa(config, self.mma_tiler)

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,       # (M, K, L)
        b_tensor: cute.Tensor,       # (1, K, L)
        sfa_tensor: cute.Tensor,     # (M, K÷16, L)
        sfb_tensor: cute.Tensor,     # (1, K÷16, L)
        c_tensor: cute.Tensor,       # (M, 1, L) output
        max_active_clusters: cutlass.Constexpr = 12,
        stream: cuda.CUstream = None,
    ):
        """
        Persistent kernel: processes M tiles with persistent vector b in TMEM.

        Loop structure:
        for k_tile_idx in range(K // BLOCK_K):
            load b[k_tile] ’ TMEM_b (persistent)
            load sfb[k_tile] ’ TMEM_sfb (persistent)

            for m_tile_idx in persistent_scheduler:
                load a[m_tile, k_tile] ’ SMEM
                load sfa[m_tile, k_tile] ’ SMEM
                mma: c[m_tile] += a[m_tile] @ b[k_tile] (from TMEM)

            store c[all m_tiles] ’ GMEM
        """

        config = self.config
        M, K, L = a_tensor.shape
        m_tiles = (M + config.mma_tiler_m - 1) // config.mma_tiler_m
        k_tiles = (K + config.block_k - 1) // config.block_k

        # === INITIALIZATION ===
        thread_idx = cute.thread_id()
        block_idx = cute.block_id()
        cluster_idx = cute.cluster_id()

        # Tile scheduler state
        sched_params = _compute_tile_scheduler_params(
            m_tiles, k_tiles, config.cluster_shape_m, int(max_active_clusters)
        )
        total_m_tiles = sched_params['total_m_tiles']
        total_k_tiles = sched_params['total_k_tiles']

        # Allocate SMEM for staging A and SFA
        a_smem = cute.allocate_smem(self.smem_layout_a)
        sfa_smem = cute.allocate_smem(self.smem_layout_sfa)

        # TMEM for persistent vector b and its scales (allocated once)
        b_tmem_shape = (config.block_k,)
        sfb_tmem_shape = (config.block_k // config.sf_vec_size,)
        b_tmem = cute.allocate_tmem_tensor(b_tmem_shape)
        sfb_tmem = cute.allocate_tmem_tensor(sfb_tmem_shape)

        # TMEM for accumulator (reused across K tiles for each M tile)
        accum_tmem = cute.allocate_tmem_tensor((config.mma_tiler_m,))
        cute.clear(accum_tmem)

        # === SHARED STATE FOR WARP SPECIALIZATION ===
        b_ready_barrier = cute.create_barrier(initial_count=0)
        mma_done_barrier = cute.create_barrier(initial_count=0)

        # === MAIN PERSISTENT TILE LOOP ===
        for k_tile_idx in range(total_k_tiles):
            k_offset = k_tile_idx * config.block_k
            k_size = min(config.block_k, K - k_offset)

            # === TMA WARP: Load persistent vector b and scales ===
            if thread_idx == config.tma_warp_id:
                # Load b[k_offset : k_offset+k_size, :] ’ TMEM
                b_tile = cute.local_tile(b_tensor, (slice(k_offset, k_offset + k_size), slice(None)))
                cute.copy(b_tile, b_tmem)

                # Load sfb scales [k_offset÷16 : ..., :] ’ TMEM
                sfb_idx_start = k_offset // config.sf_vec_size
                sfb_idx_end = (k_offset + k_size + config.sf_vec_size - 1) // config.sf_vec_size
                sfb_tile = cute.local_tile(
                    sfb_tensor,
                    (slice(sfb_idx_start, sfb_idx_end), slice(None))
                )
                cute.copy(sfb_tile, sfb_tmem)

                # Signal MMA warp that vector is ready
                cute.mbarrier_arrive(b_ready_barrier)

            # === PERSISTENT TILE SCHEDULER: Process M tiles ===
            for persistent_idx in cute.persistent_tile_scheduler(total_m_tiles, max_active_clusters):
                m_tile_idx = persistent_idx
                m_offset = m_tile_idx * config.mma_tiler_m
                m_size = min(config.mma_tiler_m, M - m_offset)

                # === TMA WARP (parallel): Load A and SFA for this M tile ===
                if thread_idx == config.tma_warp_id:
                    # Load a[m_offset : m_offset+m_size, k_offset : k_offset+k_size, :] ’ SMEM
                    a_tile = cute.local_tile(
                        a_tensor,
                        (slice(m_offset, m_offset + m_size),
                         slice(k_offset, k_offset + k_size),
                         slice(None))
                    )
                    cute.copy(a_tile, a_smem[:, :, 0])

                    # Load SFA scales
                    sfa_idx_start = k_offset // config.sf_vec_size
                    sfa_idx_end = (k_offset + k_size + config.sf_vec_size - 1) // config.sf_vec_size
                    sfa_tile = cute.local_tile(
                        sfa_tensor,
                        (slice(m_offset, m_offset + m_size),
                         slice(sfa_idx_start, sfa_idx_end),
                         slice(None))
                    )
                    cute.copy(sfa_tile, sfa_smem[:, :, 0])

                cute.cta_sync_barrier()  # All threads wait for SMEM loads

                # === MMA WARP: Execute tensor core operations ===
                if thread_idx == config.mma_warp_id:
                    cute.mbarrier_wait(b_ready_barrier)  # Wait for persistent vector

                    # Partition for MMA
                    a_part = self.tiled_mma.partition_A(a_smem[:, :, 0])
                    b_part_tmem = b_tmem  # Already in TMEM
                    sfa_part = self.tiled_mma.partition_A(sfa_smem[:, :, 0])
                    sfb_part = sfb_tmem

                    # Execute block-scaled MMA
                    cute.tcgen05_mma_block_scale_vec_2x(
                        kind=cutlass.cute.nvgpu.tcgen05.kind_mxf4nvf4_t,
                        d_tmem=accum_tmem,
                        a_desc=a_part,           # Matrix A from SMEM
                        b_desc=b_part_tmem,      # Vector B from TMEM (persistent)
                        scale_A_tmem=sfa_part,
                        scale_B_tmem=sfb_part,
                        enable_input_d=(k_tile_idx > 0)  # Accumulate over K tiles
                    )

                    cute.mbarrier_arrive(mma_done_barrier)

                cute.cta_sync_barrier()

            # === EPILOGUE WARP: Store results after processing all M tiles for this K tile ===
            if thread_idx in config.epi_warp_ids:
                cute.mbarrier_wait(mma_done_barrier)

                # Load from TMEM to registers
                accum_regs = cute.make_register_tensor(shape=(config.mma_tiler_m,))
                cute.copy(accum_tmem, accum_regs)

                # For each M tile, store result
                for m_tile_idx in range(m_tiles):
                    m_offset = m_tile_idx * config.mma_tiler_m
                    m_size = min(config.mma_tiler_m, M - m_offset)

                    # Store c[m_offset : m_offset+m_size, 0, :]  accum_regs[0:m_size]
                    c_tile = cute.local_tile(
                        c_tensor,
                        (slice(m_offset, m_offset + m_size), slice(None), slice(None))
                    )
                    cute.copy(accum_regs[0:m_size], c_tile)

        # === CLEANUP ===
        cute.dealloc_tmem(b_tmem)
        cute.dealloc_tmem(sfb_tmem)
        cute.dealloc_tmem(accum_tmem)


def custom_kernel(data) -> torch.Tensor:
    """
    Interface matching submission.py signature.

    Args:
        data: Tuple of (a, b, sfa, sfb, _, _, c)
              - a: (M, K, L) NVFP4
              - b: (1, K, L) NVFP4
              - sfa: (M, K÷16, L) FP8
              - sfb: (1, K÷16, L) FP8
              - c: (M, 1, L) FP16 (output)

    Returns:
        c: Accumulated result (M, 1, L) FP16
    """
    from task import input_t, output_t

    a, b, sfa, sfb, _, _, c = data
    device = a.device

    # Create kernel configuration
    M, K, L = a.shape
    config = GemvPersistentConfig(
        mma_tiler_m=128,
        block_k=64,
        sf_vec_size=16,
        cluster_shape_m=1,
        cluster_shape_n=1,
        num_ab_stages=2,
    )

    # Create kernel instance
    kernel = GemvPersistentKernel(config)

    # Wrap tensors for CUTE
    a_cute = cute.make_tensor(a)
    b_cute = cute.make_tensor(b)
    sfa_cute = cute.make_tensor(sfa.to(device=device, non_blocking=True))
    sfb_cute = cute.make_tensor(sfb.to(device=device, non_blocking=True))
    c_cute = cute.make_tensor(c)

    # Get CUDA stream
    stream = cuda.CUstream()

    # Launch kernel
    kernel(
        a_cute,
        b_cute,
        sfa_cute,
        sfb_cute,
        c_cute,
        max_active_clusters=cutlass.const_expr(12),
        stream=stream
    )

    return c
