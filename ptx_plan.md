# PTX Bring-Up Plan for `sub_ptx.py`

## Target Pipeline (GMEM → SMEM → TMEM → Registers → GMEM)
- **Mainloop tiles**: target `M_tile=128`, `N_tile=64` (for GEMV we’ll only use the first column), `K_tile=64`. These align with tcgen05 shape options (Table 39 in `ptx_docs/ptx_isa.txt`).
- **Data stages & instructions**:
  1. **GMEM → SMEM**: `tma.load.tensor.{1d,2d}.shared::cluster.global` (PTX §9.7.16.6). Requires TMA descriptors + `mbarrier` sync (§9.7.16.5).
  2. **SMEM → TMEM (MMA)**:
     - SMEM descriptors per Table 40 in `smem_nv4_desc.txt`.
     - `tcgen05.cp.async` for block-scale vectors (PTX §9.7.16.7.2).
     - `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale` (PTX §9.7.16.4) with IDESC from Table 44.
  3. **TMEM → Registers**: `tcgen05.ld.32x32b*` (PTX §9.7.16.7.3).
  4. **Registers → GMEM**: standard CUDA stores or `st.global` (no new PTX).

## Incremental Replacement Strategy
1. **Baseline sanity (already done)**
   - Keep scalar CUDA loops; only PTX inlined is `fma.rn.f32` (ensures inline PTX tooling works).

2. **Stage A: GMEM→SMEM with TMA (no tensor cores yet)**
   - Introduce shared-memory staging buffers for A/B tiles (128×64, 64×64).
   - Emit TMA descriptor setup on host (C++) or inline constant arrays referencing `ptx_docs/ptx_isa.txt` §9.7.16.6.
   - In kernel: replace manual loads with `tma.load.tensor` + `mbarrier.arrive/try_wait` loops (see CuTe tutorial 02 for sequence). Still run the old scalar loop on the SMEM tile to verify correctness.
   - Debug focus: descriptor bitfields (Table 40), swizzle alignment, mbarrier parity.

3. **Stage B: TMEM allocation scaffolding**
   - Re-introduce `tcgen05.alloc/dealloc/relinquish_alloc_permit` once per tile CTA (PTX §9.7.16.7.1).
   - Reserve minimal columns (32) and verify CTAs don’t deadlock (warp-synchronous issue). No MMA yet; just test allocator+deallocator to ensure they run without stalling (limit CTAs per SM temporarily if needed).

4. **Stage C: Block-scale vector path**
   - TMA-load `sfa/sfb` tiles into SMEM (same stage as A).
   - Emit `tcgen05.cp.async` to copy those SMEM tiles into TMEM columns (`PTX §9.7.16.7.2`). Still keep scalar math for the dot product so we can verify the TMEM copies land where we expect (e.g., read TMEM back with `tcgen05.ld` and compare).

5. **Stage D: Replace scalar inner loop with `tcgen05.mma`**
   - Build the instruction descriptor (IDESC) per Table 44 (`smem_nv4_desc.txt`) for `.kind::mxf4nvf4.block_scale`. Confirm bitfields by cross-checking CUTLASS 72b example.
   - Emit `tcgen05.mma` once per K tile, pointing to the SMEM descriptors and TMEM accumulator pointer.
   - For verification, immediately read TMEM back with `tcgen05.ld` and run the existing reduction, so only the compute stage changes.

6. **Stage E: Full TMEM epilogue**
   - After all K tiles, replace the shared-memory reduction with a proper TMEM→register load using `tcgen05.ld.32x32b`. Map warp/lane IDs to `(M_tile, N_tile)` rows.
   - Apply `alpha/beta` scaling in registers (same math as before) and store to GMEM.

7. **Stage F: GEMV-specific optimizations**
   - Exploit the fact N=1 by trimming the descriptor to only issue the first column, or fuse L dimension as N.
   - Once consistent, consider `tma.store` for the epilogue, persistent CTAs, etc.

## Debug/Validation Checklist per Stage
- **Stage A**: compare SMEM tile contents against CPU copy. Use conditional compilation to dump SMEM via printf for small shapes.
- **Stage B**: keep CTA count low while testing allocation to avoid starving; check for deadlocks/timeouts.
- **Stage C**: read TMEM scale-factor columns back via `tcgen05.ld` and compare to original scale data.
- **Stage D**: verify MMA outputs tile-wise by comparing TMEM contents vs. scalar results for small shapes.
- **Stage E**: ensure TMEM addressing matches lane layout; we may need to derive the column/lane mapping from PTX docs.

## References
- `ptx_docs/ptx_isa.txt`: §9.7.16 (TMEM/TMA/TCGEN05), §9.7.16.7 (alloc/cp/ld instructions).
- `ptx_docs/smem_nv4_desc.txt`: Table 40 (SMEM descriptor bitfields), Table 44 (IDESC).
- CuTe tutorials (`cute/02_mma_tma_sm100.cu`, `cute/03_mma_tma_multicast_sm100.cu`) for sequencing TMEM alloc + MMA.
- CUTLASS example `cutlass_examples/72b_blackwell_nvfp4_nvfp4_gemm.cu` for block-scaled layout and dtype tags.

