# PTX Implementation Plan for NVFP4 Block-Scaled GEMV

This document outlines the design required to replace the current scalar fallback with a full Tensor Memory (TMEM) + tcgen05 implementation. Every step happens inside the `cuda_source` string in `sub_ptx.py`, so we rely solely on inline PTX and CUDA built-ins.

## 1. Kernel Goals

1. Load packed NVFP4 inputs (`A`, `B`) and FP8 scale vectors (`SFA`, `SFB`) from global memory.
2. Stage tiles into shared memory (SRAM) in the descriptor layouts expected by tcgen05.
3. Allocate TMEM columns for accumulators and scale vectors using `tcgen05.alloc`.
4. Copy staged tiles into TMEM with `tcgen05.cp`.
5. Launch `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X` so Tensor Cores apply block scales and compute the GEMV.
6. Drain the TMEM accumulator with `tcgen05.ldmatrix` (or `tcgen05.ld`) and write results to FP16 output.
7. Deallocate TMEM columns and loop across K / batch tiles.

## 2. Data Layout and Tiling

- **Tile Shape:** Use the standard 1-CTA shape for NVFP4 block scaling: `BLOCK_M = 128`, `BLOCK_N = 8`, `BLOCK_K = 64`. GEMV forces `N=1`, so we can treat the RHS as an 8-column padded tile.
- **Scale Vector Size:** `.scale_vec::4X` (one FP8 scale per 16 FP4 elements) per the NVFP4 spec.
- **Shared Memory Buffers:**
  - `smem_a`: `BLOCK_M × BLOCK_K / 2` bytes (FP4 packed).
  - `smem_b`: `BLOCK_N × BLOCK_K / 2` bytes.
  - `smem_scale_a`: `rep_m × rep_k × 32 × 4` bytes, where `rep_m = BLOCK_M / 128` and `rep_k = BLOCK_K / 64`.
  - `smem_scale_b`: similar sizing for `BLOCK_N` tiles (pad to 128 rows because descriptors expect that height).

## 3. TMEM Management

Instructions to use:

```
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], nCols;
tcgen05.dealloc.cta_group::1.sync.aligned.b32 dst, nCols;
```

- Allocate three regions per CTA:
  1. `d_tmem` for accumulators (`BLOCK_M × BLOCK_N` / 32 columns rounded up).
  2. `scaleA_tmem` for the scale vector tile.
  3. `scaleB_tmem` for the RHS scale tile.
- Only lane 0 executes `alloc`/`dealloc`; broadcast via shared memory.
- Column counts must be multiples of 32.

## 4. Shared-Memory Descriptors

We need `a_desc`, `b_desc`, and `idesc` values that describe the shared-memory tiles. Each descriptor is 64 bits with the following fields (from PTX ISA):

```
bits 0:15   -> leading dimension in bytes
bits 16:29  -> stride between minor tiles (in 16B units)
bits 30:47  -> base pointer (shared address >> 4)
bits 48:63  -> swizzle / layout flags
```

Implementation plan:
1. Convert shared addresses with `__cvta_generic_to_shared` (32-bit).
2. Pack the descriptor using inline bitfield ops (`shl.b64`, `or.b64`).
3. Store descriptors in registers for later PTX instructions.
4. Build `idesc` by encoding MMA shape (m=128, n=8, k=64) and data types (NVFP4) according to Table 54 in PTX ISA.

## 5. Copy Tiles into TMEM

Instruction family:

```
tcgen05.cp.cta_group::1.128x256b       [taddr], s_desc;
tcgen05.cp.cta_group::1.64x128b.warpx2 [taddr], s_desc;
```

Steps:
1. After shared memory is populated, prepare the `s_desc` (shared descriptor) required by `tcgen05.cp`.
2. Issue `tcgen05.cp` per operand: once for A, once for B, and once for each scale tile.
3. Use `bar.sync` to ensure copies finish before launching MMA.

## 6. Tensor-Core MMA

Instruction:

```
tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
    [d_tmem],  a_desc,  b_desc, idesc,
    [scaleA_tmem], [scaleB_tmem], PRED_enable_input_d;
```

Implementation details:
1. Set predicate `PRED_enable_input_d = 0` (no pre-existing accumulator).
2. Provide `[scaleA_tmem]` and `[scaleB_tmem]` addresses from allocations.
3. For GEMV we only need the base operation (no collector usage). If we pipeline multiple mmaps per CTA, use `.collector::a::fill/use` modifiers following the PTX doc.
4. Loop over K tiles: after each `tcgen05.mma`, update descriptors / shared memory pointers for the next tile.

## 7. Reading Results from TMEM

Instruction:

```
tcgen05.ldmatrix.sync.aligned.m16n8.row.f32 {dst regs}, [d_tmem];
```

Plan:
1. After finishing the K loop, use `tcgen05.ldmatrix` (or `tcgen05.ld` with `.m64n16`) to pull the accumulator fragment for each 16×8 block.
2. Convert each FP32 value to FP16 with `cvt.rn.f16.f32` and write to the output buffer `c`. Since GEMV only needs N=1, we take the first column.
3. Once all fragments are written, deallocate `d_tmem` (and scale TMEM) to free columns.

## 8. Control Flow per CTA

Pseudo-structure:

```
for (batch in 0..L)
  for (mTile in 0..M/BLOCK_M)
    alloc TMEM regions
    acc_zero
    for (kTile in 0..K/BLOCK_K)
      global->shared copies (A, B, scales)
      descriptor pack
      tcgen05.cp (A/B/scales)
      tcgen05.mma ... block_scale
    tcgen05.ldmatrix -> registers -> output fp16
    tcgen05.dealloc
```

Barriers (`bar.sync 0`) guard transitions between loads, TMEM copies, and MMA launches.

## 9. Error Handling / Debugging Hooks

- Wrap every `asm volatile` block with `#if __CUDA_ARCH__ >= 1000` guard.
- Optionally emit `asm volatile("{ .reg .pred PTX_P; setp.ne.b32 PTX_P, %0, %0; }")` placeholders before TMEM ops for better disassembly alignment.
- Provide `assert(K % 64 == 0)` etc. in host code so we never launch unsupported configurations.

## 10. Next Steps

1. Implement shared-memory loaders and descriptor packers in `cuda_source`.
2. Add inline PTX helpers for TMEM alloc/dealloc, cp, mma, ldmatrix.
3. Validate with small fake tiles by compiling locally (`nvcc -arch=sm_110`).
4. Integrate into `custom_kernel` and rerun `popcorn-cli submit`.

