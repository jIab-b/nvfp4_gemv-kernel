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
        k_super_blocks = _ceil_div(k, self.k_tile * self.vec_tile)
        grid_y = max(1, l * k_super_blocks)
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
            k_super_blocks,
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
        k_super_blocks,
    ):
        # Treat tiling constants as compile-time literals for SMEM alloc / loops
        rows_per_cta = ROWS_PER_CTA
        vec_tile = VEC_TILE
        scale_group = SCALE_GROUP

        # Block / thread coordinates
        block_m, block_batch, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # Decode batch and K super-tile (512-wide) index
        batch_id = block_batch // k_super_blocks
        super_k = block_batch - batch_id * k_super_blocks

        # How many valid rows remain for this tile
        global_row0 = block_m * rows_per_cta
        rows_left = max(0, m - global_row0)
        rows_this_tile = min(rows_per_cta, rows_left)

        # K spans full columns of the matrix (nvfp4 packs 2 values per byte)
        k_bytes_per_row = k // 2
        k_scale_per_row = k // scale_group
        k_byte_base = super_k * (K_TILE // 2) * vec_tile
        k_scale_base = super_k * (K_TILE // scale_group) * vec_tile

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
        bytes_a_stage = ROWS_PER_CTA * (K_TILE // 2)
        bytes_sfa_stage = ROWS_PER_CTA * (K_TILE // scale_group)
        bytes_b_stage = VEC_TILE * (K_TILE // 2)
        bytes_sfb_stage = VEC_TILE * (K_TILE // scale_group)

        smem_a_ptr = cute.arch.alloc_smem(cutlass.Int8, 2 * bytes_a_stage)
        smem_b_ptr = cute.arch.alloc_smem(cutlass.Int8, 2 * bytes_b_stage)
        smem_sfa_ptr = cute.arch.alloc_smem(cutlass.Int8, 2 * bytes_sfa_stage)
        smem_sfb_ptr = cute.arch.alloc_smem(cutlass.Int8, 2 * bytes_sfb_stage)

        smem_layout_a = cute.make_layout((rows_per_cta, K_TILE // 2), stride=(K_TILE // 2, 1))
        smem_layout_b = cute.make_layout((vec_tile, K_TILE // 2), stride=(K_TILE // 2, 1))
        smem_layout_sfa = cute.make_layout((rows_per_cta, K_TILE // scale_group), stride=(K_TILE // scale_group, 1))
        smem_layout_sfb = cute.make_layout((vec_tile, K_TILE // scale_group), stride=(K_TILE // scale_group, 1))

        #
        # TMEM allocations: accumulator (128x8) and scale tiles
        #
        # Allocate small SMEM buffers to receive TMEM addresses
        tmem_addr_acc = cute.arch.alloc_smem(cutlass.Int32, 2)
        tmem_addr_sfa = cute.arch.alloc_smem(cutlass.Int32, 2)
        tmem_addr_sfb = cute.arch.alloc_smem(cutlass.Int32, 2)

        cute.arch.alloc_tmem(num_columns=32, smem_ptr_to_write_address=tmem_addr_acc)
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Int8, 16, tmem_addr_acc)
        acc_tmem = cute.make_tensor(acc_tmem_ptr, cute.make_layout((rows_per_cta, VEC_TILE), stride=(VEC_TILE, 1)))

        cute.arch.alloc_tmem(num_columns=32, smem_ptr_to_write_address=tmem_addr_sfa)
        sfa_tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Int8, 16, tmem_addr_sfa)
        cute.arch.alloc_tmem(num_columns=32, smem_ptr_to_write_address=tmem_addr_sfb)
        sfb_tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Int8, 16, tmem_addr_sfb)
        sfa_tmem = cute.make_tensor(sfa_tmem_ptr, cute.make_layout((rows_per_cta, K_TILE // scale_group), stride=(K_TILE // scale_group, 1)))
        sfb_tmem = cute.make_tensor(sfb_tmem_ptr, cute.make_layout((vec_tile, K_TILE // scale_group), stride=(K_TILE // scale_group, 1)))

        # Zero accumulator TMEM
        acc_zero = cute.make_fragment_like(acc_tmem)
        cute.copy(cute.full_like(acc_zero, 0), acc_zero)
        cute.copy(acc_zero, acc_tmem)

        # Process the eight 64-wide slices within this 512-wide super tile
        for col in cutlass.range_constexpr(VEC_TILE):
            stage = col & 1
            k_byte_offset = k_byte_base + col * (K_TILE // 2)
            k_scale_offset = k_scale_base + col * (K_TILE // scale_group)
            col_valid = (k_byte_offset * 2) < k  # still within total K

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
            cute.copy(cute.full_like(a_smem, 0), a_smem)
            cute.copy(cute.full_like(sfa_smem, 0), sfa_smem)

            #
            # Cooperative GMEM -> SMEM copies (manual, simple)
            #
            # Copy A rows
            if tidx < rows_this_tile and col_valid:
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

            # Copy B rows (single vector, eight K-slices become columns)
            if col_valid:
                b_tile = cute.local_tile(
                    b_g,
                    cute.make_tile(1, K_TILE // 2, 1),
                    cute.make_coord(0, k_byte_offset, batch_id),
                )
                cute.copy(b_tile, b_smem[col : col + 1, :])

                sfb_tile = cute.local_tile(
                    sfb_g,
                    cute.make_tile(1, K_TILE // scale_group, 1),
                    cute.make_coord(0, k_scale_offset, batch_id),
                )
                cute.copy(sfb_tile, sfb_smem[col : col + 1, :])

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
            mma_atom.set(tcgen05.Field.ACCUMULATE, col > 0)
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
        # Load accumulator from TMEM to registers
        acc_regs = cute.make_fragment_like(acc_tmem)
        cute.copy(acc_tmem, acc_regs)

        # Store to global C (only first column used)
        if tidx < rows_this_tile:
            c_tile = cute.local_tile(
                c_g,
                cute.make_tile(1, 1, 1),
                cute.make_coord(global_row0 + tidx, 0, batch_id),
            )
            # Reduce over all vec_tile columns into a single scalar
            acc_sum = acc_regs[tidx, 0]
            for v in range(1, VEC_TILE):
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
