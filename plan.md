# sub_mma.py Implementation Plan

## Target Kernel Overview
- **Goal**: Replace the scalar FMA GEMV in `sub_ptx.py` with a Tensor-Memory-backed Blackwell kernel in `sub_mma.py` that saturates `tcgen05.mma` throughput for the M/K/L shapes in `task.yml` while keeping tensor formats `(nvfp4, fp8)` intact.
- **CTA Tile**: 128 rows (M) × 64 cols (N surrogate) × 64 K depth, mapping naturally to tcgen05-supported shapes and to the divisibility constraints of every provided test/benchmark.
- **Work decomposition**: Each CTA handles one `(M_tile, L_slice)` pair; `cta_group::1` controls two warps (64 threads) for MMA issue, while extra warps manage TMA copies and reduction to the final GEMV vector.
- **Data movement**: Global → Tensor Memory via `cp.async.bulk.tensor.*` (TMA) for A, B, SFA, SFB tiles; accumulator fragments reside in Tensor Memory (`d_tmem`) then spilled to shared for epilogue.
- **Debug strategy**: Instrument with `printf` checkpoints (guarded on `blockIdx`/`threadIdx`) and `asm("trap;")`/`assert`-style errors for mismatched descriptor setup. Remote CLI prints STDOUT/STDERR, so each sanity phase logs short lines only when debug flag is set.

## Instruction Inventory
| Instruction | Purpose |
|-------------|---------|
| `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32` / `tcgen05.dealloc...` / `tcgen05.relinquish_alloc_permit...` | Reserve/free 32-column Tensor Memory chunks per CTA before MMA launch.
| `cp.async.bulk.tensor.<rank>.shared::cta.global.mbarrier::complete_tx::bytes` | Stream multi-dimensional tiles for A/B/SF from global memory into Tensor Memory (through shared pointer indirection) while overlapping with compute.
| `cp.async.commit_group` / `cp.async.wait_group` | Synchronize asynchronous copies feeding Tensor Memory so MMA doesn’t see stale data.
| `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X` | Main Tensor Core compute path for nvfp4 inputs + fp8(E4M3) scales; writes accumulators into Tensor Memory.
| `tcgen05.store.d.shared::cta` (or equivalent `stmatrix` helper) | Move accumulator fragments from Tensor Memory to shared or global memory before reduction to GEMV outputs.
| `mbarrier.init` / `mbarrier.arrive.expect_tx` (optional) | Coordinate producer/consumer stages for TMA copies among warps.
| `printf` (device) & `asm("trap")` | Debugging/logging via remote CLI.

## Final Kernel Shape
1. **Prologue**
   - Load CTA parameters; allocate Tensor Memory columns for A, B, SFA, SFB, and accumulators.
   - Initialize mbarriers and shared-state for pointer descriptors.
2. **TMA Stage**
   - Use `cp.async.bulk.tensor` to pull first K-slice of A/B plus scale tiles into Tensor Memory. Maintain double-buffer descriptors for overlapping copies.
3. **Compute Stage**
   - Cooperative warp group issues `tcgen05.mma...block_scale` with descriptors referencing Tensor Memory tiles and scale pointers. Iterate across all needed K_tiles per CTA, ping-ponging between TMEM buffers.
4. **Epilogue / Reduction**
   - Once entire K dimension processed, spill accumulators to shared memory, reduce across the internal 64 “N” columns to a scalar per row (because GEMV target), convert to FP16, and store to output `c`.
5. **Debug Hooks**
   - `printf` verifying descriptor addresses and comparing sample results vs reference for early tiles.
   - Guarded traps if TMEM copy mismatches expected data length.

## Step-by-Step Implementation Plan
1. **Scaffold `sub_mma.py`**
   - Clone host binding logic from `sub_ptx.py` but rename module/function, add debug flag plumbing, and stub CUDA kernel that currently just `assert(false)`.
2. **Define Kernel Launch Parameters**
   - Encode CTA tile sizes (128×64×64), block dimensions (at least 128 threads to cover mma + TMA warps), grid mapping (M/128 × L grid), and shared-memory budgeting for reductions/logging.
3. **Tensor Memory Setup**
   - Inside CUDA code, reserve TMEM columns via `tcgen05.alloc...`, derive base addresses for A, B, SF_A, SF_B, and for the accumulator tile; add helper inline asm wrappers for readability.
4. **Descriptor + TMA Copy Helpers**
   - Build 4D tensor map descriptors for A/B/SF that match K-major layout; add wrappers that call `cp.async.bulk.tensor` with byte strides. Implement double-buffer bookkeeping and `cp.async.commit/wait` groups.
5. **Implement Main MMA Loop**
   - Emit inline PTX for `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X`, wiring descriptors + scale TMEM pointers. Iterate over K_tiles; in each iteration wait for copies, issue MMA, kick off next copy.
6. **Epilogue Reduction**
   - Convert TMEM accumulator fragments to shared memory, sum across synthetic N dimension to obtain GEMV output, cast to FP16, and write to `c`. Add masking for any leftover columns if future configs demand it.
7. **Debugging + Validation Hooks**
   - Integrate `printf` statements enabled via macro/env flag; add `trap` for descriptor mismatches. Document remote CLI expectations (stdout/stderr capture) at top of file.
8. **Host Integration & Testing**
   - Update `custom_kernel` to select between reference and new kernel; run provided tests via remote CLI harness, capture perf/ correctness logs, iterate as needed.

