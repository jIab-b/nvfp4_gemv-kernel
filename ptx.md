# PTX Kernel Implementation Plan

This file tracks the end-to-end plan for replacing `submission.py` with a PTX-heavy kernel that matches the same high-level structure (module = `load_inline`, `custom_kernel(...)` launcher), while driving Blackwell tensor cores through `tcgen05` instructions.

## Host Scaffold (Python / C++ glue)

1. **Keep `load_inline` layout**  
   - `cpp_source`: declare `torch::Tensor batched_scaled_gemv_cuda(...)`.  
   - `cuda_source`: keep one `.cu` translation unit compiled with `-gencode=sm_100` (and optionally `sm_110`).  
   - Export `module = load_inline(...)` exactly like the current submission and keep `custom_kernel` signature so evaluation harness stays unchanged.

2. **Host-visible arguments**  
   - Convert fp4/fp8 tensors to `torch.int8` alias (as in current code).  
   - Pass raw pointers + sizes into the kernel launch.  
   - Launch grid = `(M / 128, ceil_div(L, 8))`, blockDim = `128` threads (one warpgroup).  
   - Provide `N_rows` (always 1) but keep parameter for future generality.

## Device Kernel Skeleton

```cuda
extern "C" __global__
void gemv_nvfp4_kernel(int8_t const* a, int8_t const* b,
                       int8_t const* sfa, int8_t const* sfb,
                       half* c,
                       int M, int K, int L, int N_rows) {
    // CTA coordinates -> tile origin
    // Shared memory struct -> swizzled tiles, mbarrier state, TMEM base pointer
    // Pipeline over K in 64-element slices
}
```

Inside the kernel, arrange four warps:

| Warp | Responsibility |
|------|----------------|
| 0    | Control plane: TMEM alloc/free, issuing `tcgen05.cp`, `tcgen05.mma`, `tcgen05.commit`, and final `tcgen05.ld`. |
| 1    | TMA / `cp.async.bulk.tensor` loads for `A` + `sfa`. |
| 2    | TMA loads for broadcast `B` + `sfb`. |
| 3    | Epilogue + global stores (cooperates with warp 0 for TMEM drains). |

## PTX Instruction Checklist

| Instruction | Purpose | Implementation Notes |
|-------------|---------|----------------------|
| `cp.async.bulk.tensor.{1d,2d}` (or `cp.async.ca.shared.global` as fallback) | DRAM → SMEM staging for A/B tiles and scale factors. | Configure tensor map descriptors for `(Mtile,Ktile)` slices, double-buffer SMEM using stage index `(k_tile & 1)`. |
| `mbarrier.init`, `mbarrier.arrive.expect_tx`, `mbarrier.test.wait` (or CUTLASS helpers) | Synchronize SMEM reuse vs asynchronous TMEM ops. | One barrier per CTA stored in shared memory; warp 0 initializes, all warps wait before overwriting tiles. |
| `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32` | Allocate TMEM columns for accumulators + scale vectors. | Warp 0 executes once per CTA; pass number of columns (≥ Ntile × (Mtile/32)). Store returned 32-bit base in shared memory. |
| `tcgen05.cp.cta_group::1.128x256b.{dst_fmt,src_fmt}` | Move swizzled SMEM tiles into TMEM. | Issue after SMEM tile ready; pipeline with `tcgen05.mma`. Provide separate descriptors for A data and scale vectors (scale_tmem). |
| `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X` | Perform UMMA on nvfp4 inputs with per-16 scaling, accumulating in TMEM. | Control warp iterates over `Ktile/16` groups, toggles `ScaleOut::Zero/One`, sets `disable_output_lane` mask for padded L. |
| `tcgen05.commit.cta_group::1` | Fence asynchronous MMA before overwriting SMEM stage. | Called by warp 0 once per K-slice after issuing all `tcgen05.mma`; paired with `mbarrier` wait by other warps. |
| `tcgen05.ld.sync.aligned.32x32b.xN.b32` | TMEM → registers for epilogue. | Warp 3 (plus warp 0 if needed) loads each accumulator block, obeying lane ownership (warp i accesses lanes `i*32..`). Immediately follow with `tcgen05.wait::ld`. |
| `tcgen05.wait::ld.sync.aligned` | Ensure TMEM loads complete before using registers. | One per warp after finishing `tcgen05.ld` sequence. |
| `tcgen05.dealloc.cta_group::1.sync.aligned.b32` + `tcgen05.relinquish_alloc_permit` | Return TMEM columns once CTA finished. | Warp 0 handles cleanup before exiting kernel. |
| `setmaxnreg.{dec,inc}.sync.aligned.u32` | Dynamically lend registers from helper warps to the control warp during MMA/epilogue phases if register pressure spikes. | Issue with identical immediates across the warpgroup and surround with `_syncthreads()` per ISA requirements. |

## Execution Flow

1. **CTA setup**  
   - Compute tile origins (`m0`, `n0`), guard `m0 >= M` or `n0 >= L`.  
   - Declare `SharedStorage` struct: swizzled buffers for `A/B`, arrays for `sfa/sfb`, barrier storage, TMEM base pointer.
2. **TMEM allocation**  
   - Warp 0 calls `tcgen05.alloc` with column count = `(Ntile * (Mtile / 32) * 2)` to cover accumulators + scale vectors.  
   - Optionally call `setmaxnreg.dec` on helper warps before entering the mainloop.
3. **Mainloop per `Ktile` (64 elements)**  
   - Warps 1–2 issue `cp.async`(/TMA) for the next stage; commit with `cp.async.commit_group`.  
   - `cp.async.wait_group 0` (or `mbarrier` wait) before handing descriptors to `tcgen05.cp`.  
   - Warp 0 launches `tcgen05.cp` for `A`, `tcgen05.cp` for `scale_A`, plus analogous copies for `B`/`scale_B` if storing them in TMEM.  
   - Once both operands resident, warp 0 iterates `tcgen05.mma` across the four 16-element scale groups; after first iteration switch to accumulation.  
   - Conclude slice with `tcgen05.commit` + `mbarrier.arrive` so other warps know TMEM consumed the current SMEM buffers.
4. **Epilogue**  
   - Warps 3/0 cooperatively execute `tcgen05.ld`/`tcgen05.wait::ld` to drain TMEM accumulators into registers.  
   - Convert to fp16, scale/mask invalid `n` columns, and store to global memory via vectorized `st.global.cs` (e.g., `float2`).  
   - Final `_syncthreads()`, then warp 0 calls `tcgen05.dealloc` and `tcgen05.relinquish_alloc_permit`.
5. **Host result**  
   - `custom_kernel` returns original `c` tensor, consistent with current submission.

