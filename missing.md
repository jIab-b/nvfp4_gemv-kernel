# Missing Artifacts for a Complete tcgen05 Kernel

1. **Shared-memory layout details**
   - Exact byte offsets for every CTA tile (`A`, `B`, `SFA`, `SFB`) after swizzling, including padding requirements and which warp owns which slice.
   - Known-good 64-bit SMEM descriptors (base/LBO/SBO/swizzle bits) for those tiles so we’re not guessing at Table 40 encodings.

2. **Instruction descriptor (idesc) constants**
   - Packed 32-bit values for `.kind::mxf4nvf4.block_scale.scale_vec::4X` covering our tile shapes (`Mtile=128`, `Ntile=8`, `K=64`), scale vector type (ue4m3 vs ue8m0), and scale-data IDs.
   - Clarification of the `atype/btype` encodings for nvfp4 (E2M1) and any non-zero negate/transpose bits.

3. **TMEM allocation and addressing plan**
   - Number of columns each CTA must reserve for accumulators and scale vectors.
   - Mapping from warp ID / lane to TMEM lane/column addresses so that `tcgen05.ld` loads the correct accumulator fragment.
   - TMEM circular-buffer strategy (if any) to overlap MMA and epilogue.

4. **Synchronization choreography**
   - Required sequence of `tcgen05.commit`, `mbarrier`, and/or `tcgen05.wait::*` calls around `tcgen05.cp`/`tcgen05.mma` so the asynchronous pipeline makes forward progress.
   - Whether we need `tcgen05.fence::*` or TMA barriers between cp/mma stages.

5. **Global↔SMEM staging specifics**
   - Do we use TMA (`cp.async.bulk.tensor`) or cooperative loads? If TMA, we need the tensor-map descriptors (extents/strides/swizzle masks) for each operand.
   - Cluster/CTA-pair configuration (cta_group::1 vs ::2) and any multicast masks if we decide to use 2-SM launches.

6. **Validation hooks**
   - Either a reference CUTLASS/CuTe kernel or SASS dump that we can match against, or access to runtime checks (`--g-tensor-memory-access-check`) plus instructions on interpreting TMEM fault reports.
   - Target performance/benchmark expectations so we can confirm once the PTX path is correct.

Without this information, we’re still guessing at descriptors, TMEM addresses, and synchronization, which is why the current kernel deadlocks. Once these blanks are filled in, we can wire the PTX path end-to-end with confidence. 
