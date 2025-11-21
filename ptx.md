# Blackwell tcgen05 GEMV Plan (v2)

Goal: Faster batched FP4 GEMV on B200 using tcgen05 with fully async, double‑buffered pipeline; stay within PTX shape rules and avoid CTA-wide barriers.

## Shapes, tiles, grid
- MMA kind: `.kind::mxf4nvf4.block_scale.scale_vec::4X` (K=64).
- Legal shape table ⇒ choose `M_TILE=128`, `N_TILE=8`, `K_TILE=64` with `.cta_group::1`.
- Grid: `(ceil(M/128), L)`. For L>1 we saturate 192 SMs; L=1 large case gives 56 CTAs (acceptable without K-split).
- Block: 128 threads (4 warps). Warp 0 is the sole producer/consumer of SMEM/TMEM; other warps may stay idle or help GMEM→SMEM only if we add a barrier (kept off for simplicity).

## TMEM allocations (once per CTA, power-of-two nCols≥32)
```ptx
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [sm_taddr_a], 128;   // A tile (128 cols)
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [sm_taddr_sfa], 32;  // scale A
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [sm_taddr_sfb], 32;  // scale B
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [sm_taddr_d], 128;   // accum D
```
One warp reads taddrs from shared. `tcgen05.relinquish_alloc_permit` at kernel end; `tcgen05.dealloc` for each taddr.

## SMEM ping‑pong buffers
- Two SMEM slots for A (128×64 FP4 ≈ 4 KB each), B (64×8 FP4 ≈ 256 B), sfa/sfb (≈512 B total). ≈5 KB/slot ×2 ≈10 KB → fits easily.
- No `__syncthreads` because a single warp issues all GMEM→SMEM copies and later consumes them.

## Per-K-tile loop (double buffered)
Pseudo PTX for the controlling warp:
```ptx
// Prologue: load tile 0 into buf=0
cp.async.bulk.shared::cta.global.L2::128B [sm_a[0]],   [gA + m0 + k0], 4096, [mbar_g2s];
cp.async.bulk.shared::cta.global.L2::128B [sm_b[0]],   [gB + k0],       256,  [mbar_g2s];
cp.async.bulk.shared::cta.global            [sm_sfa[0]],[gSFA + m0 + k0s],512,[mbar_g2s];
cp.async.bulk.shared::cta.global            [sm_sfb[0]],[gSFB + k0s],    64, [mbar_g2s];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0;

loop over k tiles:
  if (has_next) {
    cp.async.bulk... -> buffer next   // same 4 copies, overlap with compute
    cp.async.bulk.commit_group;
  }

  // SMEM→TMEM for current buffer
  tcgen05.cp.cta_group::1.128x256b [taddr_a], sdesc_a(buf);
  tcgen05.cp.cta_group::1.4x256b   [taddr_sfa], sdesc_sfa(buf);
  tcgen05.cp.cta_group::1.4x256b   [taddr_sfb], sdesc_sfb(buf);

  // MMA
  tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
      [taddr_d], [taddr_a], bdesc(buf), idesc, [taddr_sfa], [taddr_sfb], 1;

  // Completion and ordering
  tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbar_tc];
  mbarrier.try_wait.parity.b64 %p, [mbar_tc], %phase;
  @!%p bra NOT_DONE;
NOT_DONE:
  tcgen05.fence::after_thread_sync;   // makes D visible to tcgen05.ld/st
  xor.b32 %phase, %phase, 1;

  // Writeback (option B: TMEM→SMEM→GMEM to bypass lane restriction)
  tcgen05.st [sm_d_tmp], [taddr_d];   // shape matching 128x8
  // cooperative ld.shared + st.global.f16 for column 0

  // Wait GMEM→SMEM next if issued
  cp.async.bulk.wait_group 0;
  swap current/next buffers;
end loop
```

## Writeback detail
- TMEM access restriction: warp 0 owns lanes 0–31; to keep it simple, first store D to SMEM (`tcgen05.st` or `tcgen05.cp` to SMEM descriptor), then all warps can `ld.shared` and write column 0 to GMEM with regular stores. (If preferred, have warp 0 directly read its lane slice and store; both are legal.)

## Instruction/qualifier checklist
- Allocation: `tcgen05.alloc`, `tcgen05.dealloc`, `tcgen05.relinquish_alloc_permit`.
- Data move SMEM→TMEM: `tcgen05.cp.cta_group::1.{128x256b|4x256b}`.
- Compute: `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X`.
- Completion: `tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64`.
- Ordering before TMEM read/write: `tcgen05.fence::after_thread_sync`.
- TMEM I/O: `tcgen05.st` / `tcgen05.ld` (lane restrictions observed) or via SMEM staging.
- GMEM→SMEM: `cp.async.bulk.shared::cta.global.L2::128B` + `commit_group` + `wait_group`.
- Same `.cta_group` for all tcgen05 ops; explicit `.scale_vec::4X` because default is illegal for `mxf4nvf4`.

## Why no `__syncthreads`
- Single producer/consumer warp handles SMEM traffic and tcgen05 ops; ordering is enforced by cp.async wait and mbarrier/fence. If multi‑warp GMEM loaders are added, insert one `bar.sync` after `wait_group` to protect SMEM before `tcgen05.cp`.

## Launch/occupancy notes
- Block=128 threads. SMEM ~10 KB / CTA → multiple CTAs/SM; TMEM columns=128 keeps alloc legal. No padding beyond N=8; unused columns masked at writeback.
- L>1 cases fill SMs well; L=1 large case under-utilizes SMs but keeps simplicity and avoids K-split reduction.

## Next steps
1) Implement single-warp version first (no barriers) to validate correctness.  
2) Add optional multi-warp GMEM loaders guarded by one barrier to boost bandwidth.  
3) Tune bdesc/idesc construction and try fusing TMEM→SMEM writeback into the same warp for less traffic.  
4) Profile mbarrier spin cost; consider relaxed waiting if needed.
