import torch

from task import input_t, output_t


class CuteGEMVKernel:
    """Placeholder CuTe-based GEMV kernel wrapper.

    This object owns the tiling configuration and will eventually host the
    CuTe/TMA compiled kernel. For now it just records the parameters and allows
    us to plug in the real implementation later without touching the call site.
    """

    def __init__(
        self,
        rows_per_cta: int = 128,
        cols_per_cta: int = 64,
        vec_tile: int = 8,
        scale_group: int = 16,
    ) -> None:
        self.rows_per_cta = rows_per_cta
        self.cols_per_cta = cols_per_cta
        self.vec_tile = vec_tile
        self.scale_group = scale_group
        self._compiled = False
        self.compile()

    def compile(self) -> None:
        """Stub compile hook for the eventual CuTe/TMA kernel."""

        if self._compiled:
            return
        # Future work: materialize CuTe tensors, call the @cute.jit kernel once,
        # and keep the resulting module handle for reuse.
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
        # Placeholder path: just return the original output buffer unchanged.
        return c


_cute_kernel = CuteGEMVKernel()


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    return _cute_kernel(a, b, sfa_ref, sfb_ref, c)
