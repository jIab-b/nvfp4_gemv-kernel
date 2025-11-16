# PTX Bring-Up Plan (Comprehensive)

## Kernel Targets
- **CTA tile shape**: `tile_M = 128`, `tile_N = 64` (for GEMV we materialize only column 0), `tile_K = 64`.
- **Layouts**: SMEM tiles (A, B, SFA, SFB) staged with `cp.async`, padded/aligned for descriptor compatibility, TMEM used for accumulators and block scale vectors.

## Stage Overview
| Stage | Status | Covered in `sub_ptx.py` | Notes |
|-------|--------|-------------------------|-------|
| Stage A – GMEM→SMEM via `cp.async` | ✅ | Yes | A/B/sfa/sfb loaded into shared memory with async copies and tail handling. |
| Stage B – TMEM allocator scaffold | ✅ | Yes (CTA (0,0) only) | `tcgen05.alloc/dealloc` exercised; alloc addresses unused so far. |
| Stage C – Scale-factor staging | ✅ | Yes | sfa/sfb staged in SMEM, verified against GMEM. |
| Stage D – Block-scale vector path | ⏳ | Partially | Need to reorganize into tcgen tile, issue `tcgen05.cp/ld`, then `tcgen05.mma`. |
| Stage E – Epilogue (TMEM→register→GMEM) | ⏳ | No | Once MMA runs, load TMEM accumulators and write results. |

## Stage D Breakdown
1. **D1: Tile Restructure** *(in `sub_mma.py` sandbox)*
   - CTA tiles of 128 rows loop over `K` in 64-chunk increments.
   - Shared-memory buffers sized per chunk (A: 64 nvfp4 elements per row; SFA padded to 16 bytes per row).
   - Scalar math still used; tests validate tiling.
2. **D2: TMEM copy verification** *(needs port to `sub_ptx.py`)*
   - For each chunk:
     - Allocate TMEM columns (e.g., 32) per CTA.
     - Build SMEM descriptors (base address, LBO, SBO) for SFA/SFB tiles; obey swizzle/alignment rules.
     - Issue `tcgen05.cp.cta_group::1` to copy SFA/SFB from SMEM to TMEM.
     - Immediately `tcgen05.ld` the same columns and compare against SMEM (trap on mismatch).
     - Deallocate TMEM columns.
3. **D3: tcgen05.mma smoke test**
   - Replace the scalar accumulation per tile with one `tcgen05.mma` invocation per K chunk.
   - After MMA, `tcgen05.ld` the TMEM accumulator and compare to the scalar result for that chunk.
   - Keep scalar path (under a guard) until parity is confirmed.

## Stage E – Epilogue
- Once MMA parity is proven, remove the scalar fallback.
- After all K chunks, load TMEM accumulator fragments via `tcgen05.ld`, apply `alpha/beta`, and store to GMEM (optionally with vectorized stores).
- Release TMEM columns at CTA exit.

## TODO Summary
1. **Port the tiling/padding descriptor work from `sub_mma.py` into `sub_ptx.py`.**
2. **Implement Stage D2**: real TMEM copies via `tcgen05.cp`/`tcgen05.ld` with descriptor verification per chunk.
3. **Implement Stage D3**: invoke `tcgen05.mma` on each chunk and compare with scalar output until validated.
4. **Implement Stage E**: full TMEM epilogue, replacing the shared-memory reduction.

Throughout, keep `submission.py` stable for benchmarking; use `sub_mma.py` (or a branch) to prototype before porting into `sub_ptx.py`.
