cuda_source = R"CUDA(
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

constexpr int TILE_M = 256;
constexpr int TILE_K = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;

// Shared: staging for A/B tiles and scale vectors (packed NVFP4 & FP8)
extern "C" __global__
void __launch_bounds__(128)
batched_gemv_kernel(
    const uint8_t* __restrict__ a,     // M x K x L (nvfp4 e2m1, K-major)
    const uint8_t* __restrict__ b,     // 1 x K x L (nvfp4 e2m1, K-major)
    const uint8_t* __restrict__ sfa,   // M x (K/16) x L (fp8 e4m3fnuz)
    const uint8_t* __restrict__ sfb,   // 1 x (K/16) x L (fp8 e4m3fnuz)
    half* __restrict__ c,              // M x 1 x L (fp16)
    int M, int K, int L)
{
    const int batch = blockIdx.y;      // L
    const int m_base = blockIdx.x * TILE_M;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // --- SMEM staging (NVFP4 packed data + FP8 scales) ---
    extern __shared__ uint8_t smem[];
    uint8_t* smem_a  = smem;                                          // size: TILE_M * TILE_K / 2 bytes (4-bit)
    uint8_t* smem_b  = smem_a + (TILE_M * TILE_K) / 2;                // size: TILE_K / 2
    uint8_t* smem_sa = smem_b + (TILE_K / 2);                         // scales for A, size: TILE_M*(TILE_K/16)
    uint8_t* smem_sb = smem_sa + (TILE_M * (TILE_K / 16));            // scales for B, size: (TILE_K/16)

    // --- Load a/b/scales from GMEM → SMEM (regular ld/st, elided for brevity) ---
    // NOTE: keep your coalesced loads; just stage into smem_* buffers as you did.
    //       Ensure they are in the packed layout you plan to CP into TMEM.

    // Dummy zero-out accum landing buffer in SMEM (optional)
    for (int i = tid; i < TILE_M; i += blockDim.x) {
        if (m_base + i < M) {
            reinterpret_cast<half*>(smem)[i] = __float2half(0.f);
        }
    }
    __syncthreads();

    // === Begin TCGen05 path (TMEM allocation → copies → MMA → readback) ===

    // TMEM “addresses” / descriptors (64-bit each)
    unsigned long long tmem_d = 0;   // destination tile (accumulator/output in TMEM)
    unsigned long long tmem_a = 0;   // A tile in TMEM (NVFP4-packed)
    unsigned long long tmem_b = 0;   // B tile in TMEM (NVFP4-packed)
    unsigned long long tmem_sa = 0;  // scale A vectors in TMEM (FP8)
    unsigned long long tmem_sb = 0;  // scale B vectors in TMEM (FP8)
    unsigned long long idesc  = 0;   // indirection/shape descriptor if needed

    // --- Allocate destination columns in TMEM for D ---
    // (Pick a column count matching your m16n8k32-style microtile mapping; 1..n)
    const int d_cols = 1;
    asm volatile(
        "tcgen05.alloc.cta_group.sync.aligned.b32 [%0], %1;\n"
        : "=l"(tmem_d)
        : "r"(d_cols)
    );

    // --- Copy A/B (NVFP4 packed) and scale vectors (FP8 E4M3) from SMEM → TMEM ---
    // Choose shapes & format pairs matching your packing (examples shown).
    // A (NVFP4): src_fmt .b4x16_p64  → dst_fmt .b8x16
    asm volatile(
        // multicast/broadcast flags omitted; adjust shapes for your tile
        "tcgen05.cp.cta_group.shape::128x128b.dst_fmt::.b8x16.src_fmt::.b4x16_p64 [%0], %1;\n"
        : "=l"(tmem_a) : "r"(smem_a)
    );
    // B (NVFP4)
    asm volatile(
        "tcgen05.cp.cta_group.shape::128x128b.dst_fmt::.b8x16.src_fmt::.b4x16_p64 [%0], %1;\n"
        : "=l"(tmem_b) : "r"(smem_b)
    );
    // Scales A (FP8 E4M3)
    asm volatile(
        "tcgen05.cp.cta_group.shape::128x128b.dst_fmt::.b8x16.src_fmt::.b8x16 [%0], %1;\n"
        : "=l"(tmem_sa) : "r"(smem_sa)
    );
    // Scales B (FP8 E4M3)
    asm volatile(
        "tcgen05.cp.cta_group.shape::128x128b.dst_fmt::.b8x16.src_fmt::.b8x16 [%0], %1;\n"
        : "=l"(tmem_sb) : "r"(smem_sb)
    );

    // If you need an indirection descriptor for layout/stride, build it here.
    // For the minimal GEMV microkernel, you can often pass a null idesc (0) or a basic one:
    idesc = 0ull;

    // --- MMA: NVFP4 with per-block FP8 scales, scale_vec::2X (E2M1 * E2M1 → F32 accumulate) ---
    // D is written to TMEM ([tmem_d]); A/B/scales are TMEM descriptors.
    // enable_input_d = 0 (fresh) or 1 (accumulate)
    const int enable_input_d = 0;
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X "
        "[%0], %1, %2, %3, [%4], [%5], %6;\n"
        :
        : "l"(tmem_d), "l"(tmem_a), "l"(tmem_b), "l"(idesc),
          "l"(tmem_sa), "l"(tmem_sb), "r"(enable_input_d)
    );

    // (Loop over K tiles / blocks as needed: re-issue cp + mma per chunk)

    // --- Read back D from TMEM → registers → GMEM ---
    // Example: load 32b lanes from TMEM and write out as fp16
    float d_frag = 0.0f;
    asm volatile(
        "tcgen05.ld.sync.aligned.shape::16x64b.x1.b32 %0, [%1];\n"
        : "=f"(d_frag) : "l"(tmem_d)
    );

    if (lane_id == 0) {
        int m_row = m_base + warp_id * (TILE_M / NUM_WARPS); // example mapping; map properly in your tiler
        if (m_row < M) {
            c[m_row * L + batch] = __float2half(d_frag);
        }
    }

    // --- Optionally deallocate (good hygiene in larger kernels) ---
    asm volatile(
        "tcgen05.dealloc.cta_group.sync.aligned.b32 %0, %1;\n" :: "l"(tmem_d), "r"(d_cols)
    );
}

// C++ shim
extern "C" __host__ __device__
torch::Tensor batched_scaled_gemv_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c
) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int L = a.size(2);

    dim3 grid((M + TILE_M - 1) / TILE_M, L);
    dim3 block(128);

    // SMEM size: A + B + SA + SB (see pointers above)
    size_t smem_bytes = (TILE_M * TILE_K)/2 + (TILE_K/2) + (TILE_M * (TILE_K/16)) + (TILE_K/16);

    batched_gemv_kernel<<<grid, block, smem_bytes>>>(
        a.data_ptr<uint8_t>(),
        b.data_ptr<uint8_t>(),
        sfa.data_ptr<uint8_t>(),
        sfb.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        M, K, L
    );
    return c;
}
)CUDA";



module = load_inline(
    name='batched_scaled_gemv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batched_scaled_gemv_cuda'],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-std=c++17',
        '-gencode=arch=compute_110,code=sm_110',
    ],
    with_cuda=True,
    verbose=False
)
    


def custom_kernel(data: input_t) -> output_t:
    """
    Custom implementation of batched scaled GEMV using B200 tensor cores with hardware scaling.
    """
    a, b, _, _, sfa, sfb, c = data

    # Convert Float4 and Float8 tensors to uint8 storage (free reinterpretation)
    # CUDA kernel expects raw uint8 bytes for FP4/FP8 data manipulation
    if a.dtype != torch.uint8:
        a = a.view(torch.uint8)
    if b.dtype != torch.uint8:
        b = b.view(torch.uint8)
    if sfa.dtype != torch.uint8:
        sfa = sfa.view(torch.uint8)
    if sfb.dtype != torch.uint8:
        sfb = sfb.view(torch.uint8)

    return module.batched_scaled_gemv_cuda(a, b, sfa, sfb, c)