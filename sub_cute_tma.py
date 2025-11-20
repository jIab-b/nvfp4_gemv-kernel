from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack

from task import input_t, output_t


ROWS_PER_CTA = 128
COLS_PER_CTA = 64
VEC_TILE = 8
SCALE_GROUP = 16
K_TILE = 64


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@cute.struct
class GemvBarriers:
    tma_a: cute.struct.MemRange[cutlass.Int32, 1]
    tma_b: cute.struct.MemRange[cutlass.Int32, 1]
    cp: cute.struct.MemRange[cutlass.Int32, 1]
    mma: cute.struct.MemRange[cutlass.Int32, 1]


class CuteGEMVKernel:
    """CuTe/TMA aware GEMV kernel sketch for Blackwell tcgen05."""

    def __init__(
        self,
        rows_per_cta: int = ROWS_PER_CTA,
        cols_per_cta: int = COLS_PER_CTA,
        vec_tile: int = VEC_TILE,
        scale_group: int = SCALE_GROUP,
    ) -> None:
        self.rows_per_cta = rows_per_cta
        self.cols_per_cta = cols_per_cta
        self.vec_tile = vec_tile
        self.scale_group = scale_group
        self.k_tile = K_TILE
        self.shared_storage = GemvBarriers

        self.mma_op = tcgen05.MmaMXF4NVF4Op(
            sf_dtype=cutlass.Float8E4M3FN,
            instruction_shape=(self.rows_per_cta, self.vec_tile, self.k_tile),
            cta_group=tcgen05.CtaGroup.ONE,
            a_src=tcgen05.OperandSource.SMEM,
        )


    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        sfa: torch.Tensor,
        sfb: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:

        m = int(a.size(0))
        k = int(a.size(1)) * 2
        l = int(a.size(2))

        a_cute = from_dlpack(a.view(torch.int8))
        b_cute = from_dlpack(b.view(torch.int8))
        sfa_cute = from_dlpack(sfa.view(torch.int8))
        sfb_cute = from_dlpack(sfb.view(torch.int8))

        c_cute = from_dlpack(c.view(torch.float16))

        self._launch(
            a_cute,
            b_cute,
            sfa_cute,
            sfb_cute,
            c_cute,
            m,
            k,
            l,
        )


        return c

    @cute.jit
    def _launch(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        m: int,
        k: int,
        l: int,
    ):
        grid_x = _ceil_div(m, self.rows_per_cta)
        grid_y = max(1, l)
        block = [self.rows_per_cta, 1, 1]

        self.kernel(
            a,
            b,
            sfa,
            sfb,
            c,
            m,
            k,
            grid_x,
            grid_y,
            rows_per_cta=self.rows_per_cta,
            cols_per_cta=self.cols_per_cta,
            vec_tile=self.vec_tile,
            scale_group=self.scale_group,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=block,
        )

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        m,
        k,
        grid_x,
        grid_y,
        rows_per_cta: int,
        cols_per_cta: int,
        vec_tile: int,
        scale_group: int,
    ):
        tidx = cute.arch.thread_idx()[0]
        block_x, block_y, _ = cute.arch.block_idx()
        row = block_x * rows_per_cta + tidx

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage, 64)
        mbar_tma_a = storage.tma_a.data_ptr()
        mbar_tma_b = storage.tma_b.data_ptr()
        mbar_cp = storage.cp.data_ptr()
        mbar_mma = storage.mma.data_ptr()

        if tidx == 0:
            cute.arch.mbarrier_init(mbar_tma_a, 1)
            cute.arch.mbarrier_init(mbar_tma_b, 1)
            cute.arch.mbarrier_init(mbar_cp, 1)
            cute.arch.mbarrier_init(mbar_mma, 1)
        cute.arch.mbarrier_init_fence()

        # Allocate SMEM buffers for A, B, and scale factors
        # Double-buffered: 2 sets of (matrix_A, vector_B, scales_A, scales_B)
        smem_a_0 = smem.allocate(shape=(rows_per_cta, self.k_tile // 2), dtype=cutlass.Int8)
        smem_a_1 = smem.allocate(shape=(rows_per_cta, self.k_tile // 2), dtype=cutlass.Int8)
        smem_b_0 = smem.allocate(shape=(self.k_tile // 2,), dtype=cutlass.Int8)
        smem_b_1 = smem.allocate(shape=(self.k_tile // 2,), dtype=cutlass.Int8)
        smem_sfa_0 = smem.allocate(shape=(rows_per_cta, self.k_tile // 16), dtype=cutlass.Int8)
        smem_sfa_1 = smem.allocate(shape=(rows_per_cta, self.k_tile // 16), dtype=cutlass.Int8)
        smem_sfb_0 = smem.allocate(shape=(self.k_tile // 16,), dtype=cutlass.Int8)
        smem_sfb_1 = smem.allocate(shape=(self.k_tile // 16,), dtype=cutlass.Int8)

        # Allocate TMEM for accumulator and operands
        # Accumulator: [128, 8] float16 (128 rows, 8 columns for GEMV N-dimension)
        accum_tmem = utils.TmemAllocator().allocate(
            shape=(rows_per_cta, cols_per_cta),
            dtype=cutlass.Float16,
        )

        # TMEM for operands during computation
        matrix_a_tmem = utils.TmemAllocator().allocate(
            shape=(rows_per_cta, self.k_tile),
            dtype=cutlass.Float16,
        )
        vector_b_tmem = utils.TmemAllocator().allocate(
            shape=(self.k_tile,),
            dtype=cutlass.Float16,
        )
        matrix_sfa_tmem = utils.TmemAllocator().allocate(
            shape=(rows_per_cta, self.k_tile // 16),
            dtype=cutlass.Float8E4M3FN,
        )
        vector_sfb_tmem = utils.TmemAllocator().allocate(
            shape=(self.k_tile // 16,),
            dtype=cutlass.Float8E4M3FN,
        )

        # Zero-initialize accumulator in TMEM
        if tidx == 0:
            for i in cute.range(rows_per_cta):
                for j in cute.range(cols_per_cta):
                    accum_tmem[i, j] = 0.0

        num_k_tiles = cute._ceil_div(k, self.k_tile)
        load_phase = 0
        cp_phase = 0

        for tile_idx in cute.range(num_k_tiles):
            # Select ping-pong buffers based on phase
            smem_a_curr = smem_a_0 if load_phase == 0 else smem_a_1
            smem_b_curr = smem_b_0 if load_phase == 0 else smem_b_1
            smem_sfa_curr = smem_sfa_0 if load_phase == 0 else smem_sfa_1
            smem_sfb_curr = smem_sfb_0 if load_phase == 0 else smem_sfb_1

            # Stage 1: TMA Load GMEM -> SMEM (async, signaled via mbarrier)
            # Thread 0 issues TMA load commands for matrix A, vector B, and scales
            if tidx == 0:
                # Compute tile coordinates
                k_tile_offset = tile_idx * self.k_tile

                # TMA load matrix A: [row:row+ROWS_PER_CTA, k_tile_offset:k_tile_offset+K_TILE] -> SMEM
                # a layout is [M, K/2, L], we load [ROWS_PER_CTA, K_TILE/2] per iteration
                # Implicit TMA load signaled via mbarrier
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_tma_a, 1, self.k_tile * 2
                )

                # TMA load vector B: [k_tile_offset:k_tile_offset+K_TILE] -> SMEM
                # b layout is [N, K/2, L], we load a [K_TILE/2] segment per iteration
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_tma_b, 1, self.k_tile * 2
                )

            # Wait for TMA loads to complete (all threads synchronize)
            cute.arch.mbarrier_wait(mbar_tma_a, load_phase)
            cute.arch.mbarrier_wait(mbar_tma_b, load_phase)

            # Stage 2: tcgen05.cp - SMEM -> TMEM copy with FP4 decompression
            # Create SMEM descriptors (abstract layout info for tcgen05.cp)
            smem_desc_a = cute.make_smem_desc(smem_a_curr)
            smem_desc_b = cute.make_smem_desc(smem_b_curr)
            smem_desc_sfa = cute.make_smem_desc(smem_sfa_curr)
            smem_desc_sfb = cute.make_smem_desc(smem_sfb_curr)

            # Thread 0 initiates copies to TMEM with decompression
            if tidx == 0:
                # Copy matrix A (FP4->FP8 decompression) SMEM -> TMEM
                # tcgen05.cp handles the bit-unpacking: 2 FP4 nibbles -> 1 FP8 byte
                tcgen05.copy(smem_desc_a, matrix_a_tmem)

                # Copy vector B (FP4->FP8) SMEM -> TMEM
                tcgen05.copy(smem_desc_b, vector_b_tmem)

                # Copy scale factors (already FP8) SMEM -> TMEM
                tcgen05.copy(smem_desc_sfa, matrix_sfa_tmem)
                tcgen05.copy(smem_desc_sfb, vector_sfb_tmem)

                # Signal copy completion
                tcgen05.commit(mbar_cp)

            # Wait for all copies to complete
            cute.arch.mbarrier_wait(mbar_cp, cp_phase)

            # Stage 3: tcgen05.mma - Block-scaled multiply-accumulate in TMEM
            # Create instruction descriptor for MMA (encodes shape, datatype, scale info)
            instr_desc = 0  # Descriptor bits encoding M=128, N=8, K=64, mxf4nvf4, scale_vec::2X

            # Wait for prior MMA to complete (important for pipelined accumulation)
            cute.arch.mbarrier_wait(mbar_mma, cp_phase)

            # Only accumulate after first K-tile (first tile initializes, rest accumulate)
            p_enable = 1 if tile_idx > 0 else 0

            # Thread 0 executes the block-scaled MMA
            if tidx == 0:
                # tcgen05.mma: A[128×64] × B[64×8] -> Accum[128×8]
                # Inputs: matrix_a_tmem, vector_b_tmem (FP8 decompressed data)
                # Scales: matrix_sfa_tmem, vector_sfb_tmem (FP8 scale factors)
                # Block-scaling: dequant(A[i,j]) = unquant(A_fp4[i,j]) * scale_a[i//16, j//16]
                tcgen05.mma(
                    accum=accum_tmem,
                    a=matrix_a_tmem,
                    b=vector_b_tmem,
                    sfa=matrix_sfa_tmem,
                    sfb=vector_sfb_tmem,
                    instr_desc=instr_desc,
                    p_enable=p_enable,
                    cta_group=tcgen05.CtaGroup.ONE,
                )

                # Signal MMA completion for next iteration
                tcgen05.commit(mbar_mma)

            # Toggle phases for double-buffering
            load_phase = 1 - load_phase
            cp_phase = 1 - cp_phase

        # Stage 4: TMEM -> Registers -> Global Memory
        # Load accumulated results from TMEM and reduce to GEMV output
        if row < m:
            lane_id = cute.arch.lane_idx()
            warp_id = tidx >> 5

            # Process results per thread: each thread owns one row
            if tidx < rows_per_cta:
                sum_val = 0.0

                # Sum across the 8 columns (N-dimension reduction for GEMV)
                for col in cute.range(cols_per_cta):
                    # tcgen05.ld.sync: load from TMEM with synchronization (32x32b F16 format)
                    val = accum_tmem[tidx, col]
                    sum_val = sum_val + val

                # Store reduced result to global output [M, 1, L]
                c[row, 0, block_y] = sum_val

            # Deallocate TMEM resources
            utils.TmemAllocator().deallocate(accum_tmem)
            utils.TmemAllocator().deallocate(matrix_a_tmem)
            utils.TmemAllocator().deallocate(vector_b_tmem)
            utils.TmemAllocator().deallocate(matrix_sfa_tmem)
            utils.TmemAllocator().deallocate(vector_sfb_tmem)


cute_kernel = CuteGEMVKernel()


def jit_warmup(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    return cute_kernel(a, b, sfa_ref, sfb_ref, c)

# Warmup with dummy inputs to precompile the CuTe kernel
_dummy_m, _dummy_k, _dummy_l = 128, 64, 1
_dummy_a = torch.zeros((_dummy_m, _dummy_k, _dummy_l), dtype=torch.uint8, device='cuda')
_dummy_b = torch.zeros((128, _dummy_k, _dummy_l), dtype=torch.uint8, device='cuda')
_dummy_sfa = torch.zeros((_dummy_m, _dummy_k // 16, _dummy_l), dtype=torch.int8, device='cuda')
_dummy_sfb = torch.zeros((128, _dummy_k // 16, _dummy_l), dtype=torch.int8, device='cuda')
_dummy_c = torch.zeros((_dummy_m, 1, _dummy_l), dtype=torch.float16, device='cuda')
_dummy_data = (_dummy_a, _dummy_b, _dummy_sfa, _dummy_sfb, None, None, _dummy_c)
jit_warmup(_dummy_data)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    return cute_kernel(a, b, sfa_ref, sfb_ref, c)
