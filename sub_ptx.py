"""
Queued work: tcgen05 GEMV kernel.
Currently no fallback CUDA-core kernel is provided; this file deliberately
raises to avoid pretending we have an implementation.
"""

from task import input_t, output_t

# Plan (PTX-ready) retained for reference
PTX_PLAN = r"""
// tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
// K_TILE=64, M_TILE=128, N_TILE=8 (min legal N)
// .scale_vec::4X == one scale per 16 FP4 elements (alias .block16)
// All tcgen05 ops share .cta_group::1; one warp issues them.
"""


def custom_kernel(data: input_t) -> output_t:
    raise RuntimeError(
        "tcgen05 PTX kernel not implemented in this stub. "
        "Provide a hardware path or wire to an external binary."
    )
