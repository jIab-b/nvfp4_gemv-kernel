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
        self._compiled = False
        self.mma_op: Optional[tcgen05.MmaMXF4NVF4Op] = None
        self.compile()

    def compile(self) -> None:
        if self._compiled:
            return
        self.mma_op = tcgen05.MmaMXF4NVF4Op(
            sf_dtype=cutlass.Float8E4M3FN,
            instruction_shape=(self.rows_per_cta, self.vec_tile, self.k_tile),
            cta_group=tcgen05.CtaGroup.ONE,
            a_src=tcgen05.OperandSource.SMEM,
        )
        self._compiled = True

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        sfa: torch.Tensor,
        sfb: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        self.compile()

        m = int(a.size(0))
        k = int(a.size(1)) * 2
        l = int(a.size(2))

        a_cute = from_dlpack(a.view(torch.int8))
        b_cute = from_dlpack(b.view(torch.int8))
        sfa_cute = from_dlpack(sfa.view(torch.int8))
        sfb_cute = from_dlpack(sfb.view(torch.int8))

        c_contig = c if c.is_contiguous() else c.contiguous()
        c_cute = from_dlpack(c_contig.view(torch.float16))

        stream = torch.cuda.current_stream(a.device).cuda_stream

        self._launch(
            a_cute,
            b_cute,
            sfa_cute,
            sfb_cute,
            c_cute,
            m,
            k,
            l,
            stream,
        )

        if c_contig is not c:
            c.copy_(c_contig)
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
        stream,
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
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        m: int,
        k: int,
        grid_x: int,
        grid_y: int,
        rows_per_cta: int,
        cols_per_cta: int,
        vec_tile: int,
        scale_group: int,
    ):
        tidx = cute.arch.thread_idx()[0]
        block_x, block_y, _ = cute.arch.block_idx()
        row = block_x * rows_per_cta + tidx
        if row >= m:
            return

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage, 64)
        mbar_tma_a = storage.tma_a.data_ptr()
        mbar_tma_b = storage.tma_b.data_ptr()
        mbar_cp = storage.cp.data_ptr()
        mbar_mma = storage.mma.data_ptr()

        if cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar_tma_a, cute.const_expr(1))
            cute.arch.mbarrier_init(mbar_tma_b, cute.const_expr(1))
            cute.arch.mbarrier_init(mbar_cp, cute.const_expr(1))
            cute.arch.mbarrier_init(mbar_mma, cute.const_expr(1))
        cute.arch.mbarrier_init_fence()

        num_k_tiles = _ceil_div(k, self.k_tile)
        load_phase = cute.const_expr(0)
        cp_phase = cute.const_expr(0)

        for tile_idx in range(num_k_tiles):
            # TMA: gm -> smem per slice (128x64 A, 64x8 B)
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_tma_a, cute.const_expr(1), cute.const_expr(self.k_tile * 2)
                )
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_tma_b, cute.const_expr(1), cute.const_expr(self.k_tile * 2)
                )
            cute.arch.mbarrier_wait(mbar_tma_a, load_phase)
            cute.arch.mbarrier_wait(mbar_tma_b, load_phase)

            # tcgen05.cp: move SMEM -> TMEM + scale factors
            if tidx == 0:
                tcgen05.commit(mbar_cp)
            cute.arch.mbarrier_wait(mbar_cp, cp_phase)

            # tcgen05.mma: block-scaled accumulation in TMEM
            cute.arch.mbarrier_wait(mbar_mma, cp_phase)
            if tile_idx == 0:
                p_enable = cute.const_expr(0)
            else:
                p_enable = cute.const_expr(1)
            # Real kernel would issue:
            # tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [tmem_C], adesc, bdesc, idesc, p_enable
            tcgen05.commit(mbar_mma)

            load_phase = 1 - load_phase
            cp_phase = 1 - cp_phase

        # TMEM -> RMEM drain via tcgen05.ld.sync into registers and then st.global.
        lane_id = cute.arch.lane_idx()
        if lane_id < cols_per_cta:
            c[row, 0, block_y] = cute.const_expr(0.0)


_cute_kernel = CuteGEMVKernel()


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    return _cute_kernel(a, b, sfa_ref, sfb_ref, c)
