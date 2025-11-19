from typing import Optional

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from task import input_t, output_t


ROWS_PER_CTA = 128
COLS_PER_CTA = 64
VEC_TILE = 8
SCALE_GROUP = 16


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class CuteGEMVKernel:
    """Lightweight CuTe wrapper for the NVFP4 GEMV blueprint."""

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
        self._compiled = False
        self.compile()

    def compile(self) -> None:
        """Compile the CuTe kernel once (host/device builders live inside the class)."""
        if self._compiled:
            return
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

        device = a.device
        m = int(a.size(0))
        k = int(a.size(1)) * 2
        l = int(a.size(2))

        a_view = a.view(torch.int8)
        b_view = b.view(torch.int8)
        sfa_view = sfa.view(torch.int8)
        sfb_view = sfb.view(torch.int8)

        a_cute = from_dlpack(a_view)
        b_cute = from_dlpack(b_view)
        sfa_cute = from_dlpack(sfa_view)
        sfb_cute = from_dlpack(sfb_view)

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

        grid_x = _ceil_div(m, self.rows_per_cta); grid_y = l
        block = [self.rows_per_cta, 1, 1]

        self.kernel(
            a,
            b,
            sfa,
            sfb,
            c,
            m,
            k,
            l,
            self.rows_per_cta,
            self.cols_per_cta,
            self.vec_tile,
            self.scale_group,
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
        l: int,
        rows_per_cta: int,
        cols_per_cta: int,
        vec_tile: int,
        scale_group: int,
    ):
        """CuTe kernel stub that mirrors the planned pipeline structure."""
        tidx = cute.arch.thread_idx()[0]
        block_x, block_y, _ = cute.arch.block_idx()
        row = block_x * rows_per_cta + tidx
        if row >= m:
            return

        # === Placeholder: TMA loads would iterate over k-slices here ===
        # (we will hook mbarrier coordination once the real atoms live inside TMEM)

        # === Placeholder: tcgen05.cp would copy A/B/scale to TMEM ===
        # === Placeholder: tcgen05.mma would accumulate in TMEM ===
        # For now we simply zero the first batch column for visibility.
        if block_y == 0 and l > 0:
            c[row, 0, 0] = 0.0


_cute_kernel = CuteGEMVKernel()


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    return _cute_kernel(a, b, sfa_ref, sfb_ref, c)
