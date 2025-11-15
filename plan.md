1. Python wrapper skeleton
   - In `submission.py`, keep a single `custom_kernel(data: input_t) -> output_t` that calls a `torch.utils.cpp_extension.load_inline` module exactly once (module cached globally).
   - Prepare tensors: ensure `a/b/sfa/sfb` are contiguous on device, cast to `torch.uint8` views for raw nvfp4/fp8 data, keep `c` as fp16 destination.

2. Inline extension layout
   - Use a single `src` string passed both as C++ and CUDA source; include only CUDA 13.0 system headers (`torch/extension.h`, `<cuda_fp16.h>`, `<cuda_fp4.h>`, `<cuda_fp8.h>`, `<cute/*>`, `<cutlass/*>` etc.).
   - Expose `torch::Tensor batched_scaled_gemv_cute(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);` and register with `TORCH_LIBRARY`/`PYBIND11_MODULE`.

3. Type aliases and layouts
   - Inside the src string, set `using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;` and same for `ElementB`.
   - Define fp8 scale-factor element as `cutlass::float_ue4m3_t` (matches fp8(e4m3fnuz)).
   - Use `cute::Stride`/`cute::Layout` objects to build GMEM tensors reflecting `(M,K,L)` in K-major order and the `(K//16)` scale-factor grids.

4. Shared storage & TMA descriptors
   - Create a templated `SharedStorage` struct (following `cute/02_mma_tma_sm100.cu`) holding SMEM tiles for A/B plus TMA + MMA barriers and TMEM base pointer.
   - Implement helper routines to allocate TMEM columns per CTA and to set up `cute::tmem::async_copy` barriers as in the tutorials.

5. Tiled MMA definition
   - Choose an `mma_tiler = make_shape(Int<M_TILE>, Int<N_TILE>, Int<K_TILE>)` with `M_TILE` dividing `M` (likely `_128`), `N_TILE = _64` (since GEMV) but keep hardware-supported `N>=64` by fusing `L` or replicating columns; document how we reinterpret GEMV as GEMM with `N=64` but only produce column 0.
   - Instantiate `TiledMMA = decltype(make_tiled_mma(cute::SM100_64x128x16_F16F16F32{}))` or appropriate block-scaled variant for nvfp4.
   - Ensure cluster shape matches occupancy goals (probably `Shape<_1,_1,_1>` to start).

6. Mainloop structure
   - Partition GMEM tensors with `local_tile` to obtain per-CTA views, kick off TMA loads of A/B tiles into SMEM double buffers, wait on producer barrier, and issue `cute::gemm` with `tiled_mma` writing accumulators to TMEM.
   - Keep a loop over `K` tiles; for each iteration issue asynchronous copy for next tile while current tile computes (standard CuTe pipeline).

7. Epilogue with scale factors
   - After MMA, load the TMEM accumulator fragments with `cute::copy(tiled_mma, ...)` into registers.
   - Load the `c` vector tile from GMEM (fp16), cast to float, and blend with accumulator (alpha=1, beta=0 unless we need to accumulate onto existing `c`).
   - Down-convert to `half` and store back to GMEM using `cute::copy` (or manual store) only for the first column to realize GEMV.

8. Kernel launcher
   - Write a `__global__` kernel `cute_batched_gemv_kernel( … )` that instantiates the layouts, shared storage, mma tiler, etc., then calls the GEMM routine described above.
   - Choose grid dimensions `(grid_m = M / TILE_M, grid_l = L)` and `blockDim = 128` threads (matching the MMA atom requirement).

9. Host binding
   - Implement `torch::Tensor batched_scaled_gemv_cute(...)` that validates tensor shapes/dtypes, computes `M,K,L`, launches the kernel with `cudaStream_t` from the input tensors, and returns `c`.
   - Register the function through pybind; in Python, call `module.batched_scaled_gemv_cute(...)` inside `custom_kernel` and return the updated `c` tensor.

10. Compilation flags & testing
   - Pass `extra_cuda_cflags=['-O3','-std=c++20','--use_fast_math','-gencode=arch=compute_100a,code=sm_100a']` to `load_inline`.
   - After integrating, run `python task.py` (or provided tests) to confirm correctness and measure throughput.
