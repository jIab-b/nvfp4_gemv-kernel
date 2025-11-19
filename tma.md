# TMA-Based NVFP4 GEMV: Implementation Blueprint

This document lays out, in detail, how to rearchitect the kernel in `submission.py`
so it leverages the Blackwell TMA + tcgen05 pipelines without wasting TensorCore
cycles on redundant columns.

## 0. High-Level Goals

* Treat GEMV as a skinny GEMM so every tcgen05.mma multiplies useful data.
* Stage operands through the GMEM → SMEM → TMEM hierarchy using TMA and
  tcgen05.cp, removing per-thread decode loops.
* Process **N = 8** real vectors at a time (the smallest legal N). Each column of
  the B tile corresponds to a different RHS vector so no work is wasted.
* Double-buffer descriptors so copies, MMAs, and TMEM loads overlap; use
  mbarriers instead of coarse `__syncthreads()`.

## 1. Work Partitioning & File Structure

* **Files**: mirror `submission.py` structure. New CUDA source lives inside
  `submission.py` so the entry points stay `batched_scaled_gemv_cuda()` and
  `custom_kernel()` in Python.
* **CTA layout**: `gridDim.x = ceil_div(M, 128)` rows per tile. `gridDim.y = ceil_div(L * N_vectors, N_tile)` so each CTA receives eight vectors (N_tile=8) pulled from the batch dimension. `gridDim.z = 1`.
* **Threads**: warpgroup of 128 threads. Only a single “loader” thread issues the
  TMA / tcgen05 instructions; the rest idle unless they participate in any FMAs
  outside the TensorCores.

## 2. Data Fragmentation

* **Matrix A**: reshape input to `[M_tiles, 128, K]`. For each tile and for each
  `k0` chunk of 64 FP4 elements (32 packed bytes) we copy a `128×64` slab.
* **Vectors B**: gather eight independent vectors per CTA. For batch index `l`
  and vector group `g`, build a matrix `B_tile` of shape `64×8` (64 from K, 8 from
  vectors). Each column is an actual RHS vector segment.
* **Scale factors**: follow the same tiling—`sfa` produces a `128×(64/16)` FP8
  slab per A tile, `sfb` produces `(64/16)×8` per vector tile.
* **Padding**: if `K % 64 != 0`, zero the remaining FP4 bytes and scales so the
  last chunk still fits the canonical K=64 requirement.

## 3. GMEM → SMEM via TMA

* For each operand we create a TMA descriptor (see CUTLASS `cute::make_tma_desc`).
  The `A` descriptor encodes a 128×64 tile with k-major layout; the `B`
  descriptor encodes a 64×8 tile with vector columns stored contiguously.
* In the kernel:
  ```ptx
  // TMA loads, launched by thread 0 of the CTA
  tma.load.mbarrier::complete_tx::bytes [mbarA], [tma_desc_A], [smem_ptr_A], A_coords;
  tma.load.mbarrier::complete_tx::bytes [mbarB], [tma_desc_B], [smem_ptr_B], B_coords;
  ```
  Each load uses its own mbarrier object in shared memory. No `__syncthreads()`;
  consumers call `mbarrier.try_wait.parity` before hitting tcgen05.cp.

## 4. SMEM → TMEM via tcgen05.cp

* Once the TMA barriers signal completion, issue the two tensor-memory copies:
  ```ptx
  // A operand copy (with FP4→FP8 decompression)
  tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [tmem_A], adesc;

  // B operand copy (vector tile)
  tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [tmem_B], bdesc;
  ```
  After each cp, immediately launch
  ```ptx
  tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbar_cp];
  ```
  so the consumer can wait on the same barrier before issuing `tcgen05.mma`.

## 5. MMA and Accumulation

* Instruction descriptor (`idesc`): encode `.kind::mxf4nvf4`, `.scale_vec::2X`,
  `M=128`, `N=8`, `K=64`, no transpose. Precompute the 32-bit descriptor on the
  host or via device helper.
* Launch the MMA once A/B TMEM tiles are ready:
  ```ptx
  tcgen05.mma.cta_group::1.kind::mxf4nvf4
      .block_scale.scale_vec::2X
      [tmem_C], adesc, bdesc, idesc, p_enable;
  ```
  where `p_enable` keeps D zero for the first K slice and true otherwise. No
  registers are touched; accumulation stays in TMEM.
* Loop over all `k0` slices; reuse the same TMEM accumulator address so the
  tensor core keeps adding into the 128×8 tile.

## 6. TMEM → Registers → Global

* After the K-loop, drain the accumulator with loads, eight columns at a time,
  per warp:
  ```ptx
  tcgen05.ld.sync.aligned.32x32b.x4.b32 {{r0, r1, r2, r3}}, [tmem_C_lane];
  tcgen05.wait::ld.sync.aligned;
  ```
* Each lane now owns 32 partial sums for one vector column. Optionally reduce
  along N (if the client still wants a single column) or scatter directly to the
  eight RHS outputs:
  ```ptx
  st.global.cs.f16 [c_ptr + vec_id * M + row], cvt.rn.f16.f32(r0);
  ```

## 7. Double Buffering & Synchronization Strategy

* Maintain two TMEM slots for `A`/`B` (ping-pong). While MMA consumes slot 0,
  schedule the next TMA load into slot 1.
* Barrier sequence per tile:
  1. `tma.load` signals `mbarA/mbarB`.
  2. `tcgen05.cp` waits on those barriers, copies into TMEM, then signals
     `mbar_cp`.
  3. `tcgen05.mma` waits on `mbar_cp`, runs, and immediately issues
     `tcgen05.commit` so the next stage can proceed.
* No `__syncthreads()`; all ordering is handled by mbarriers plus the implicit
  tcgen05 pipelines.

## 8. Python Side Changes

* Shuffle the vector dimension before launching the kernel. Group RHS vectors in
  batches of eight so each CTA sees eight contiguous vectors. Update
  `custom_kernel` to reshape `b` and `sfb` accordingly and to reshape the output
  back to `(M, 1, L)`.
* Keep the `torch.int8` views (NVFP4 packs) but pass auxiliary tensors that map
  CTA indices to vector groups so the kernel can compute `B_coords` for the TMA
  descriptors.

## 9. Outstanding Questions / TODOs

1. Finalize descriptor helpers (`make_shared_desc`, `make_instr_desc`) so they
   emit the exact bit patterns required for `.kind::mxf4nvf4`.
2. Decide whether to adopt `.cta_group::2` once the single-CTA pipeline is
   stable; that would double M to 256 and require `.warpx2` multicast.
3. Benchmark the TMEM load/store shapes (`tcgen05.ld .32x32b.x4` vs `.16x64b.x8`)
   to maximize register packing before the FP32→FP16 conversion.

With this blueprint implemented inside `submission.py`, the kernel will finally
drive tcgen05 at something close to its intended throughput: every TensorCore
operation multiplies real data, memory movement is fully asynchronous, and the
vector padding overhead disappears.
