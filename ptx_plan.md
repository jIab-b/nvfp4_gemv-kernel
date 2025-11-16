# PTX Bring-Up Plan (Updated)

## Target Tile Shape
- CTA processes `(tile_M=128, tile_N=64)` with `K` looped in `tile_K=64` chunks.
- For GEMV (`N=1`), treat `tile_N` as 64 to match tcgen05 descriptors; we’ll only materialize column 0 in the epilogue.

## Stage Outline
1. **Stage A** — GMEM→SMEM (done): `cp.async` double-buffered tiles for A/B and scale tensors.
2. **Stage B** — TMEM allocation scaffold (done): `tcgen05.alloc/dealloc` once per CTA.
3. **Stage C** — Scale-factor staging (done for SMEM): copy `sfa/sfb` tiles into shared memory.
4. **Stage D** — Block-scale vector path (new sub-steps):
   - **D1: Tile restructure** (CTA per 128×64 tile, `K` looped in 64-chunk). Scalar math still used.
   - **D2: TMEM copy verification** (`tcgen05.cp` from SMEM to TMEM, `tcgen05.ld` back to verify). [CURRENT]
   - **D3: tcgen05.mma smoke test** (one MMA per chunk; read TMEM accumulator and compare to scalar result).
5. **Stage E** — Epilogue: move TMEM accumulator directly to registers, apply `alpha/beta`, write GMEM.

## Stage D Sub-Steps in Detail
- **D1: Tile restructure**
  - Launch grid `(ceil(M/128), L)` with `blockDim=128`.
  - For each CTA: iterate `K` in `64`-wide chunks. Inside the chunk, cooperate to load `A_tile[128][64/2]`, `B_tile[64/2]`, `SFA_tile[128][4]`, `SFB_tile[4]` into SMEM.
  - Keep the existing scalar GEMV loop but reindex it to use `tile_M` and `tile_K`.
  - *Verification:* Run the full test suite—results must match the baseline since math hasn’t changed.

- **D2: TMEM copy verification**
  - After staging each chunk, allocate TMEM columns once per CTA (`tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`).
  - Build SMEM descriptors for the staged `SFA_tile` and `SFB_tile` (see Table 40). Issue `tcgen05.cp.cta_group::1.128x256b` (or appropriate shape) to copy those tiles into TMEM columns.
  - Immediately issue `tcgen05.ld.32x32b` on the same TMEM addresses, copy the data into registers, and compare against the original SMEM buffers. Trap if any mismatch occurs.
  - Deallocate the TMEM columns afterwards (`tcgen05.dealloc`, `tcgen05.relinquish_alloc_permit`).
  - *References:* PTX ISA §9.7.16.7.1 (alloc/dealloc), §9.7.16.9.2 (`tcgen05.cp`), §9.7.16.7.3 (`tcgen05.ld`).
  - *Verification:* Test suite passes unless TMEM copies misbehave—in which case the CTA hits the trap immediately.

- **D3: tcgen05.mma smoke test**
  - For each chunk, use the staged SMEM tiles and TMEM scales to issue `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale` with `tile_M=128`, `tile_N=64`, `tile_K=64` (IDESC per Table 44).
  - Immediately load the TMEM accumulator via `tcgen05.ld` and compare to the scalar result for that chunk. Keep the scalar path active in parallel for A/B testing.
  - *Verification:* For small test inputs (M=128, K=64, L=1) the tcgen05 output should match the scalar result exactly.

## Testing Strategy
- After D1: run the full harness; results must match baseline (no PTX yet).
- After D2: run harness; expect either success or a `trap` if TMEM copies are wrong.
- After D3: run small tests (M=128, K=64) to compare scalar vs. tcgen05 per chunk; once parity is proven, remove the scalar redundancy chunk-by-chunk.
