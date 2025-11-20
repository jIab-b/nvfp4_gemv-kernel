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
        n_rows = int(b.shape[0])
        groups_per_batch = _ceil_div(n_rows, self.vec_tile)

        grid_x = _ceil_div(m, self.rows_per_cta)
        grid_y = max(1, l * max(1, groups_per_batch))
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
            n_rows,
            groups_per_batch,
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
        l,
        n_rows,
        groups_per_batch,
        grid_x,
        grid_y,
        rows_per_cta: int,
        cols_per_cta: int,
        vec_tile: int,
        scale_group: int,
    ):
        # Block / thread coordinates
        block_m, block_batch, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # Decode batch and vector group (K-major batches are contiguous in 'l')
        batch_id = block_batch // groups_per_batch
        vec_group = block_batch % groups_per_batch
        if batch_id >= l:
            return

        vec_base = vec_group * vec_tile
        vec_count = min(vec_tile, max(0, n_rows - vec_base))
        if vec_count == 0:
            return

        # How many valid rows remain for this tile
        global_row0 = block_m * rows_per_cta
        rows_left = max(0, m - global_row0)
        rows_this_tile = min(rows_per_cta, rows_left)
        if rows_this_tile == 0:
            return

        # K spans full columns of the matrix (nvfp4 packs 2 values per byte)
        k_bytes_per_row = k // 2
        k_scale_per_row = k // scale_group
        k_tiles = _ceil_div(k, K_TILE)

        #
        # GMEM tensor views (already K-major)
        #
        a_g = a
        b_g = b
        sfa_g = sfa
        sfb_g = sfb
        c_g = c

        #
        # Shared memory allocations (double buffer for A/B + scales)
        #
        bytes_a_stage = rows_per_cta * (K_TILE // 2)
        bytes_sfa_stage = rows_per_cta * (K_TILE // scale_group)
        bytes_b_stage = vec_tile * (K_TILE // 2)
        bytes_sfb_stage = vec_tile * (K_TILE // scale_group)

        smem_a_ptr = cute.arch.alloc_smem(cutlass.Int8, 2 * bytes_a_stage)
        smem_b_ptr = cute.arch.alloc_smem(cutlass.Int8, 2 * bytes_b_stage)
        smem_sfa_ptr = cute.arch.alloc_smem(cutlass.Int8, 2 * bytes_sfa_stage)
        smem_sfb_ptr = cute.arch.alloc_smem(cutlass.Int8, 2 * bytes_sfb_stage)

        smem_layout_a = cute.make_layout((rows_per_cta, K_TILE // 2), (K_TILE // 2, 1))
        smem_layout_b = cute.make_layout((vec_tile, K_TILE // 2), (K_TILE // 2, 1))
        smem_layout_sfa = cute.make_layout((rows_per_cta, K_TILE // scale_group), (K_TILE // scale_group, 1))
        smem_layout_sfb = cute.make_layout((vec_tile, K_TILE // scale_group), (K_TILE // scale_group, 1))

        #
        # TMEM allocations: accumulator (128x8) and scale tiles
        #
        acc_tmem_ptr = cute.arch.alloc_tmem(num_columns=vec_tile, smem_ptr_to_write_address=None)
        acc_tmem = cute.make_tensor(acc_tmem_ptr, cute.make_layout((rows_per_cta, vec_tile), (vec_tile, 1)))

        sfa_tmem_ptr = cute.arch.alloc_tmem(num_columns=K_TILE // scale_group, smem_ptr_to_write_address=None)
        sfb_tmem_ptr = cute.arch.alloc_tmem(num_columns=K_TILE // scale_group, smem_ptr_to_write_address=None)
        sfa_tmem = cute.make_tensor(sfa_tmem_ptr, cute.make_layout((rows_per_cta, K_TILE // scale_group), (K_TILE // scale_group, 1)))
        sfb_tmem = cute.make_tensor(sfb_tmem_ptr, cute.make_layout((vec_tile, K_TILE // scale_group), (K_TILE // scale_group, 1)))

        # Zero accumulator TMEM
        cute.copy(cute.full_like(acc_tmem, 0), acc_tmem)

        # Loop over K in 64-wide tiles
        for kt in range(k_tiles):
            stage = kt & 1
            k_byte_offset = kt * (K_TILE // 2)
            k_scale_offset = kt * (K_TILE // scale_group)

            # SMEM tensor views for this stage
            a_smem = cute.make_tensor(
                smem_a_ptr + stage * bytes_a_stage,
                smem_layout_a,
            )
            b_smem = cute.make_tensor(
                smem_b_ptr + stage * bytes_b_stage,
                smem_layout_b,
            )
            sfa_smem = cute.make_tensor(
                smem_sfa_ptr + stage * bytes_sfa_stage,
                smem_layout_sfa,
            )
            sfb_smem = cute.make_tensor(
                smem_sfb_ptr + stage * bytes_sfb_stage,
                smem_layout_sfb,
            )

            # Clear B/SFB rows to avoid garbage in unused columns
            cute.copy(cute.full_like(b_smem, 0), b_smem)
            cute.copy(cute.full_like(sfb_smem, 0), sfb_smem)

            #
            # Cooperative GMEM -> SMEM copies (manual, simple)
            #
            # Copy A rows
            if tidx < rows_this_tile:
                a_tile = cute.local_tile(
                    a_g,
                    cute.make_tile(1, K_TILE // 2, 1),
                    cute.make_coord(global_row0 + tidx, k_byte_offset, batch_id),
                )
                cute.copy(a_tile, a_smem[tidx : tidx + 1, :])

                sfa_tile = cute.local_tile(
                    sfa_g,
                    cute.make_tile(1, K_TILE // scale_group, 1),
                    cute.make_coord(global_row0 + tidx, k_scale_offset, batch_id),
                )
                cute.copy(sfa_tile, sfa_smem[tidx : tidx + 1, :])

            # Copy B rows (vector tile)
            for v in range(vec_count):
                idx = vec_base + v
                b_tile = cute.local_tile(
                    b_g,
                    cute.make_tile(1, K_TILE // 2, 1),
                    cute.make_coord(idx, k_byte_offset, batch_id),
                )
                cute.copy(b_tile, b_smem[v : v + 1, :])

                sfb_tile = cute.local_tile(
                    sfb_g,
                    cute.make_tile(1, K_TILE // scale_group, 1),
                    cute.make_coord(idx, k_scale_offset, batch_id),
                )
                cute.copy(sfb_tile, sfb_smem[v : v + 1, :])

            cute.arch.sync_threads()

            #
            # Move scales to TMEM (SMEM -> TMEM copy)
            #
            cute.copy(sfa_smem, sfa_tmem)
            cute.copy(sfb_smem, sfb_tmem)

            cute.arch.sync_threads()

            #
            # MMA: A/B from SMEM, scales from TMEM, accumulate into TMEM
            #
            mma_atom = cute.make_mma_atom(self.mma_op)
            mma_atom.set(tcgen05.Field.ACCUMULATE, kt > 0)
            mma_atom.set(tcgen05.Field.SFA, sfa_tmem.iterator)
            mma_atom.set(tcgen05.Field.SFB, sfb_tmem.iterator)

            # Tile A/B to match MMA expected shapes: [M, K] and [K, N]
            tCrA = cute.local_tile(
                a_smem,
                mma_atom.tv_layout_A,
                cute.make_coord(0, 0),
            )
            tCrB = cute.local_tile(
                b_smem,
                mma_atom.tv_layout_B,
                cute.make_coord(0, 0),
            )

            cute.gemm(mma_atom, acc_tmem, tCrA, tCrB, acc_tmem)

            cute.arch.sync_threads()

        #
        # Epilogue: TMEM -> registers -> global
        #
        # Load first column (or all vec_count) from TMEM
        # Use TMEM load op to registers
        acc_regs = cute.make_fragment_like(acc_tmem)
        cute.copy(acc_tmem, acc_regs)

        # Store to global C (only first column used)
        if tidx < rows_this_tile:
            c_tile = cute.local_tile(
                c_g,
                cute.make_tile(1, 1, 1),
                cute.make_coord(global_row0 + tidx, 0, batch_id),
            )
            # Reduce over vec_tile columns into a single scalar
            acc_sum = acc_regs[tidx, 0]
            for v in range(1, vec_count):
                acc_sum = acc_sum + acc_regs[tidx, v]
            acc_sum = cute.cast(cutlass.Float16, acc_sum)
            c_tile[0, 0, 0] = acc_sum

        cute.arch.sync_threads()

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
