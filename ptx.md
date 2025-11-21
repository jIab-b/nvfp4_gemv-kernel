# Blackwell tcgen05 Tensor Core GEMV Implementation Plan

## Problem Statement

Optimize batched scaled FP4 GEMV kernel for NVIDIA B200 using 5th-gen Tensor Cores (tcgen05).

**Current performance:** 105 µs (CUDA cores, software FP4 decode)
**Target:** 8.6 µs (speed-of-light)
**Gap:** 12× slower

**Operation:** `C[M×1×L] = (A[M×K×L] * scale_A[M×K/16×L]) × (B[1×K×L] * scale_B[1×K/16×L])`

Where:
- A, B: FP4 E2M1 (2 elements per byte)
- scale_A, scale_B: FP8 E4M3 (1 scale per 16 FP4 elements)
- C: FP16 output

## Hardware Architecture

### B200 Resources
- **192 SMs** × 4 Tensor Cores = **768 Tensor Cores total**
- Each Tensor Core: 1024 FP4 ops/cycle @ 1.5 GHz
- Peak FP4: **1.18 PFLOPS**
- Memory bandwidth: **1.8 TB/s HBM3e**

### Pipeline
```
GMEM (HBM3e) → SMEM (128KB/SM) → TMEM (tensor buffer) → Tensor Cores
     ↓ TMA          ↓ tcgen05.cp        ↓ tcgen05.mma
  (LSU DMA)      (optional decomp)   (4 per SM, FP4 native)
```

## Tensor Core Constraints

### Block Dimensions
- **M**: 32, 64, 128, 256 (flexible)
- **N**: **Minimum 8-16** ← **Problem: GEMV has N=1**
- **K**: 64, 96, 128 for `.kind::mxf4nvf4`

### Workaround for N=1
Pad B from [K×1] to [K×16], compute full matrix product, extract column 0.

**Waste:** 93.75% (15/16 columns unused)
**Trade-off:** Still faster than software decode overhead

### Block Scaling Configuration

**Your data:** K/16 scale factors per row
- `.kind::mxf4nvf4` with `.scale_vec::4X` or `.block16`
- Block size: 16 FP4 elements per scale
- For K=64 tile: 4 scale factors needed
- For K=16384: 16384/16 = 1024 scales total = 256 tiles × 4 scales/tile ✓

**Scale factor layout in TMEM:**
- scale_A: [M × 4] per K=64 tile (UE8M0 or UE4M3 format)
- scale_B: [4 × N] per K=64 tile

## Implementation Strategy

### Tiling Scheme

**Per CTA:**
- M_TILE = 256 rows of A
- N_TILE = 16 (padded, use only column 0)
- K_TILE = 64 (matches 4× scale factor grouping)

**For M=7168, K=16384, L=1:**
- M tiles: 7168 / 256 = 28
- K tiles: 16384 / 64 = 256
- Total MMA operations: 28 × 256 = 7,168 tensor core calls

**Grid config:**
- Grid: (28, 1) for L=1, or (28, L) for batched
- Block: 128 threads (warp-synchronized for tcgen05 instructions)

### Memory Allocation

**TMEM allocations per CTA:**
```ptx
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [taddr_a], 256;    // A matrix
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [taddr_sfa], 32;   // scale_A
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [taddr_sfb], 32;   // scale_B
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [taddr_d], 256;    // accumulator
```

**SMEM staging:**
- A tile: 256 rows × 64 K × 0.5 bytes = 8 KB
- B tile: 64 K × 16 N × 0.5 bytes = 512 bytes
- scale_A tile: 256 × 4 × 1 byte = 1 KB
- scale_B tile: 4 × 16 × 1 byte = 64 bytes
- **Total per tile: ~9.6 KB** (fits comfortably in 128KB SMEM)

## PTX Instruction Sequence

### Phase 1: Setup (once per CTA)

```ptx
.reg .b32 %taddr_a, %taddr_sfa, %taddr_sfb, %taddr_d;
.reg .b64 %desc_b, %desc_a_smem, %desc_sfa_smem, %desc_sfb_smem, %idesc;
.reg .pred %p;
.reg .b32 %phase;

// Allocate tensor memory
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_taddr_a], 256;
ld.shared.b32 %taddr_a, [smem_taddr_a];

tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_taddr_sfa], 32;
ld.shared.b32 %taddr_sfa, [smem_taddr_sfa];

tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_taddr_sfb], 32;
ld.shared.b32 %taddr_sfb, [smem_taddr_sfb];

tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_taddr_d], 256;
ld.shared.b32 %taddr_d, [smem_taddr_d];

// Initialize mbarrier for synchronization
mbarrier.init.shared.b64 [mbar_gmem], 1;
mbarrier.init.shared.b64 [mbar_compute], 1;
mov.b32 %phase, 0;
```

### Phase 2: Main Loop (per K-tile)

```ptx
// Outer loop: K dimension (256 iterations for K=16384)
mov.b32 %k_tile, 0;

loop_k_start:

  // ========== STAGE 1: GMEM → SMEM ==========

  // Calculate global memory offsets
  // m_base = blockIdx.x * 256
  // k_offset = %k_tile * 64

  // Copy A tile: [256 × 64] FP4 = 8 KB
  cp.async.bulk.shared::cta.global.L2::128B
      [smem_a], [gbl_a_base + m_offset + k_offset], 8192, [mbar_gmem];

  // Copy B tile: [64 × 16] FP4 = 512 bytes (B replicated 16 times)
  cp.async.bulk.shared::cta.global.L2::128B
      [smem_b], [gbl_b_base + k_offset], 512, [mbar_gmem];

  // Copy scale_A tile: [256 × 4] FP8 = 1 KB
  cp.async.bulk.shared::cta.global
      [smem_sfa], [gbl_sfa_base + m_offset + k_scale_offset], 1024, [mbar_gmem];

  // Copy scale_B tile: [4 × 16] FP8 = 64 bytes (replicated)
  cp.async.bulk.shared::cta.global
      [smem_sfb], [gbl_sfb_base + k_scale_offset], 64, [mbar_gmem];

  // Commit and wait for GMEM → SMEM
  cp.async.bulk.commit_group;
  cp.async.bulk.wait_group 0;
  __syncthreads();


  // ========== STAGE 2: Create Matrix Descriptors ==========

  // Create descriptor for B in SMEM (K=64, N=16)
  // Note: Descriptor creation is typically via CUDA API,
  //       but can be done in PTX with movmatrix pseudo-instruction
  //       or by directly constructing the 64-bit descriptor value

  // Simplified: assume desc_b constructed with proper swizzling


  // ========== STAGE 3: SMEM → TMEM ==========

  // Copy A from SMEM to TMEM
  tcgen05.cp.cta_group::1.128x256b [%taddr_a], %desc_a_smem;

  // Copy scale factors to TMEM
  tcgen05.cp.cta_group::1.4x256b [%taddr_sfa], %desc_sfa_smem;
  tcgen05.cp.cta_group::1.4x256b [%taddr_sfb], %desc_sfb_smem;


  // ========== STAGE 4: TENSOR CORE MMA ==========

  // THE MONEY SHOT: Hardware FP4 decode + scaling + FMA
  tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
      [%taddr_d],         // D = output accumulator (256 × 16) FP32
      [%taddr_a],         // A in TMEM (256 × 64) FP4
      %desc_b,            // B descriptor (64 × 16) FP4 in SMEM
      %idesc,             // Instruction descriptor (matrix shapes/types)
      [%taddr_sfa],       // Scale A (256 × 4) FP8 in TMEM
      [%taddr_sfb],       // Scale B (4 × 16) FP8 in TMEM
      1;                  // enable_input_d = 1 (accumulate into D)

  // Commit MMA and synchronize
  tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbar_compute];

  mbarrier.try_wait.parity.b64 %p, [mbar_compute], %phase;
  @!%p bra wait_mma;
wait_mma:
  mbarrier.try_wait.parity.b64 %p, [mbar_compute], %phase;
  @!%p bra wait_mma;

  // Flip phase for next iteration
  xor.b32 %phase, %phase, 1;

  // Next K tile
  add.s32 %k_tile, %k_tile, 1;
  setp.lt.s32 %p_continue, %k_tile, 256;  // 256 K-tiles for K=16384
  @%p_continue bra loop_k_start;

loop_k_done:
```

### Phase 3: Writeback (once per CTA)

```ptx
  // ========== STAGE 5: TMEM → REGISTERS → GMEM ==========

  // Read column 0 of D from TMEM (discard columns 1-15)
  // For 256 rows, need to load in chunks based on register constraints

  // Row-by-row extraction (could be optimized with vectorized loads)
  mov.b32 %row, 0;
writeback_loop:

  // Load row from TMEM (returns FP32 values for all 16 columns)
  tcgen05.ld.row [%d0, %d1, ..., %d15], [%taddr_d], %row;

  // Convert column 0 from FP32 to FP16
  cvt.rn.f16.f32 %h_out, %d0;

  // Store to global memory
  // output_addr = gbl_c_base + (blockIdx.x * 256 + %row) * sizeof(half)
  st.global.f16 [output_addr], %h_out;

  add.s32 %row, %row, 1;
  setp.lt.s32 %p_row, %row, 256;
  @%p_row bra writeback_loop;


  // ========== STAGE 6: Cleanup ==========

  // Deallocate tensor memory
  tcgen05.dealloc.cta_group::1.sync.aligned.b32 %taddr_a, 256;
  tcgen05.dealloc.cta_group::1.sync.aligned.b32 %taddr_sfa, 32;
  tcgen05.dealloc.cta_group::1.sync.aligned.b32 %taddr_sfb, 32;
  tcgen05.dealloc.cta_group::1.sync.aligned.b32 %taddr_d, 256;

  tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
```

## Key Optimizations

### 1. Software Pipelining
Overlap GMEM→SMEM for tile k+1 with SMEM→TMEM→MMA for tile k:

```ptx
// Prologue: load tile 0
load_tile_to_smem(0);

loop:
  if (k_tile + 1 < num_k_tiles):
    load_tile_to_smem(k_tile + 1);  // Async, overlaps with compute

  wait_for_smem(k_tile);
  copy_smem_to_tmem(k_tile);
  tcgen05.mma(...);
  tcgen05.commit(...);
  wait_for_compute();

  k_tile++;
```

### 2. Double Buffering SMEM
Use 2× SMEM buffers to enable full overlap:
- Buffer A: 2 × 8 KB = 16 KB
- Buffer B: 2 × 512 B = 1 KB
- Total: ~20 KB (still fits in 128KB SMEM)

### 3. Instruction Descriptor Optimization
Pre-compute `idesc` encoding matrix shapes once:
- M=256, N=16, K=64
- Element types: A=E2M1, B=E2M1, D=FP32
- Scale types: UE8M0 or UE4M3

### 4. Batched Processing (L > 1)
For L=8 case:
- Grid: (28, 8) = 224 blocks
- Each block handles independent batch element
- No cross-batch communication needed

## Performance Analysis

### Compute Bound
**FP4 operations per output element:**
- 2 × K = 2 × 16384 = 32,768 FP4 ops

**Total for M=7168:**
- 7168 × 32,768 = 235M FP4 ops

**Tensor core throughput:**
- 768 cores × 1024 ops/cycle × 1.5 GHz = 1.18 PFLOPS
- Theoretical time: 235M / 1.18T = **0.2 µs** (if 100% utilization)

### Memory Bound
**Data to load:**
- A: 7168 × 16384 × 0.5 B = 58.7 MB
- B: 16384 × 0.5 B × 16 (replicated) = 0.13 MB
- scale_A: 7168 × 1024 × 1 B = 7.3 MB
- scale_B: 1024 × 1 B × 16 (replicated) = 0.016 MB
- **Total: 66.1 MB**

**Memory time:**
- 66.1 MB / 1.8 TB/s = **36.7 µs** (if 100% bandwidth)

### Expected Performance
- Memory bound: 36.7 µs theoretical
- Current target: 8.6 µs
- Implies need for **4.3× bandwidth utilization over naive calculation**

**Gap explanation:**
- Speed-of-light calculation likely uses optimized layouts (no B replication)
- Better cache locality (L2 hits on scale factors)
- Reduced SMEM staging overhead

**Realistic target with N=16 padding:**
- 36.7 µs × 0.7 (achievable bandwidth) = **~25 µs**
- Still **2.4× improvement over current 105 µs**

## Implementation Path

1. **Phase 1:** Basic tcgen05 kernel (single K-tile, no pipelining)
   - Validate correctness
   - Measure baseline

2. **Phase 2:** K-tiling loop with synchronization
   - Add K-dimension loop
   - Verify accumulation works correctly

3. **Phase 3:** GMEM→SMEM pipelining
   - Add cp.async.bulk with mbarriers
   - Overlap memory transfers

4. **Phase 4:** Double buffering
   - Overlap load(k+1) with compute(k)
   - Maximize SM utilization

5. **Phase 5:** Tuning
   - Experiment with M_TILE (128 vs 256)
   - Try different K_TILE values (64 vs 96 vs 128)
   - Profile with NCU

## Risks and Mitigations

### Risk 1: Descriptor Construction Complexity
Creating proper matrix descriptors in PTX is complex.

**Mitigation:** Use CUDA C++ API (`cudaTensorMapEncode`) to create descriptors on host, pass as kernel parameters.

### Risk 2: N=16 Padding Overhead
93.75% wasted compute significantly hurts performance.

**Mitigation:**
- Accept 2-3× improvement over current (still valuable)
- Future: explore custom N=1 tensor core mode if available in newer architectures

### Risk 3: TMEM Allocation Limits
Unknown TMEM capacity per SM.

**Mitigation:**
- Use smaller M_TILE if allocation fails
- Spill to SMEM if TMEM exhausted

### Risk 4: Scale Factor Layout Mismatch
Your scales might not match expected TMEM layout.

**Mitigation:**
- Pre-process scales on CPU/GPU to match required format
- Use transpose or swizzle operations in SMEM before TMEM copy

## Success Metrics

**Minimum viable:** 50 µs (2× improvement)
**Good:** 25 µs (4× improvement, ~80% of memory-bound limit)
**Excellent:** 15 µs (7× improvement, approaching speed-of-light)

**Current:** 105 µs
**Target:** 8.6 µs (requires eliminating N-padding waste - may need architectural changes)
