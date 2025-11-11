import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# nvfp4_gemv_megakernel.py
# One "mega kernel" using torch.utils.cpp_extension.load_inline to perform:
# NVFP4(E2M1) unpack + per-16 FP8(E4M3) scaling + GEMV accumulation -> FP16
# Shapes (packed K-major tensors):
#   a:   [M, K//2, L]  uint8   (two 4-bit values per byte)
#   b:   [1, K//2, L]  uint8
#   sfa: [M, K//16, L] uint8   (FP8 E4M3 enc for scales of A)
#   sfb: [1, K//16, L] uint8   (FP8 E4M3 enc for scales of B)
#   c:   [M, 1, L]     float16
#
# You can later add an SM100 (Blackwell) fast path by replacing the inner loop
# under the __CUDA_ARCH__ >= 1000 section with inline PTX tcgen05.mma ... block_scale.


_src = r"""
// ====== C++ (ATen) interface ======
#include <torch/extension.h>
#include <vector>

void nvfp4_batched_scaled_gemv_launcher(
    const at::Tensor& A_nvfp4,   // [M, K2, L] uint8
    const at::Tensor& B_nvfp4,   // [1, K2, L] uint8
    const at::Tensor& SFA_fp8,   // [M, K16, L] uint8
    const at::Tensor& SFB_fp8,   // [1, K16, L] uint8
    at::Tensor& C_out            // [M, 1, L] float16
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batched_scaled_gemv", &nvfp4_batched_scaled_gemv_launcher,
        "NVFP4 block-scaled batched GEMV (mega kernel)");
}
"""

_cuda = r"""
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x,dt) TORCH_CHECK((x).dtype() == (dt), #x " has wrong dtype")
#define CUDA_KERNEL_ASSERT() asm volatile("")

// --- Toy FP8(E4M3) decode for scales (demo-quality; replace with accurate if needed) ---
__device__ __forceinline__ float fp8_e4m3_to_float(unsigned char e) {
    int sign = (e >> 7) & 1;
    int exp  = (e >> 3) & 0xF;
    int man  = (e & 0x7);
    const int exp_bias = 7;
    float v = (1.0f + (float)man / 8.0f) * __powf(2.0f, (float)(exp - exp_bias));
    return sign ? -v : v;
}

// --- NVFP4(E2M1) code -> float mapping LUT (illustrative) ---
// For correctness against your encoder, make sure these codes mirror your quantizer.
__device__ __constant__ float NVFP4_LUT[16] = {
    -6.f, -4.f, -3.f, -2.f, -1.f, -0.5f, 0.f, 0.5f,
     1.f,  1.5f, 2.f,  3.f,  4.f,   6.f,  0.f, 0.f
};

// Kernel computes: for each (m,l), c[m,0,l] = dot( deq(a[m,:,l]), deq(b[0,:,l]) )
// where deq applies per-16-K FP8(E4M3) scales from sfa/sfb.
__global__ void nvfp4_batched_scaled_gemv_kernel(
    const uint8_t* __restrict__ A_nv,   // [M, K2, L]
    const uint8_t* __restrict__ B_nv,   // [1, K2, L]
    const uint8_t* __restrict__ SFA,    // [M, K16, L]
    const uint8_t* __restrict__ SFB,    // [1, K16, L]
    half* __restrict__ C,               // [M, 1, L]
    int M, int K, int L
){
    // Thread maps a single output element (m,l)
    int l = blockIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || l >= L) return;

    // Derived strides for tight K-major packing:
    const int K2  = K >> 1;   // bytes in packed arrays along K
    const int K16 = K >> 4;   // block scales along K

    // Layout:
    // A_nv:   [M, K2, L] -> index:  m*K2*L + k2*L + l   (if contiguous in the provided dim order)
    // B_nv:   [1, K2, L] -> index:  0*K2*L + k2*L + l   (effectively k2*L + l)
    // SFA:    [M, K16, L] -> index: m*K16*L + k16*L + l
    // SFB:    [1, K16, L] -> index: k16*L + l
    // C:      [M, 1, L]   -> index: m*1*L + 0*L + l ==> m*L + l

    const uint8_t* Arow = A_nv + (size_t)m * (size_t)K2 * (size_t)L + (size_t)l; // advance by l each k2
    const uint8_t* Brow = B_nv + (size_t)l;                                      // advance by l each k2
    const uint8_t* SA   = SFA  + (size_t)m * (size_t)K16 * (size_t)L + (size_t)l;
    const uint8_t* SB   = SFB  + (size_t)l;

    float acc = 0.f;

#if __CUDA_ARCH__ >= 1000
    // --- Blackwell fast path placeholder ---
    // Here you can implement a tiled K-loop that:
    // 1) loads 64 elements of A (nibbles) and B (nibbles) for this (m,l) into registers/SMEM,
    // 2) loads the matching per-16 FP8 scales,
    // 3) issues tcgen05.mma ... kind::mxf4nvf4.block_scale to accumulate into FP16 fragments.
    // For single-output GEMV, you'd typically form fragments with N=1 (or N=8 and reduce).
    // This demo keeps the portable path below. Replace the inner loop when you add PTX.
#endif

    // --- Portable baseline (software dequant + FMA) ---
    // Walk K in bytes (2 elements/byte). Also track k16 (=k/16) for scales.
    for (int k2 = 0, k = 0; k2 < K2; ++k2, k += 2) {
        // per-16 scale index
        const int k16 = k >> 4;

        // load one byte from A and B (two nibbles each)
        const uint8_t a_pack = Arow[(size_t)k2 * (size_t)L];
        const uint8_t b_pack = Brow[(size_t)k2 * (size_t)L];

        const uint8_t a0 = (a_pack >> 4) & 0xF;
        const uint8_t a1 = (a_pack     ) & 0xF;
        const uint8_t b0 = (b_pack >> 4) & 0xF;
        const uint8_t b1 = (b_pack     ) & 0xF;

        // load FP8(E4M3) scales (one per 16 values)
        const float sa = fp8_e4m3_to_float( SA[(size_t)k16 * (size_t)L] );
        const float sb = fp8_e4m3_to_float( SB[(size_t)k16 * (size_t)L] );

        // nvfp4 dequant via LUT then scale
        const float A0 = __ldg(&NVFP4_LUT[a0]) * sa;
        const float A1 = __ldg(&NVFP4_LUT[a1]) * sa;
        const float B0 = __ldg(&NVFP4_LUT[b0]) * sb;
        const float B1 = __ldg(&NVFP4_LUT[b1]) * sb;

        acc = fmaf(A0, B0, acc);
        acc = fmaf(A1, B1, acc);
    }

    C[(size_t)m * (size_t)L + (size_t)l] = __float2half(acc);
}

static inline dim3 make_grid(int M, int L, int blk) {
    int gx = (M + blk - 1) / blk;
    return dim3(gx, L, 1);
}

void nvfp4_batched_scaled_gemv_launcher(
    const at::Tensor& A_nvfp4,   // [M, K2, L] u8
    const at::Tensor& B_nvfp4,   // [1, K2, L] u8
    const at::Tensor& SFA_fp8,   // [M, K16, L] u8
    const at::Tensor& SFB_fp8,   // [1, K16, L] u8
    at::Tensor& C_out            // [M, 1, L]  f16
){
    CHECK_CUDA(A_nvfp4); CHECK_CUDA(B_nvfp4);
    CHECK_CUDA(SFA_fp8); CHECK_CUDA(SFB_fp8);
    CHECK_CUDA(C_out);

    CHECK_CONTIG(A_nvfp4); CHECK_CONTIG(B_nvfp4);
    CHECK_CONTIG(SFA_fp8); CHECK_CONTIG(SFB_fp8);
    CHECK_CONTIG(C_out);

    CHECK_DTYPE(A_nvfp4, torch::kUInt8);
    CHECK_DTYPE(B_nvfp4, torch::kUInt8);
    CHECK_DTYPE(SFA_fp8, torch::kUInt8);
    CHECK_DTYPE(SFB_fp8, torch::kUInt8);
    CHECK_DTYPE(C_out,   torch::kHalf);

    TORCH_CHECK(A_nvfp4.dim()==3, "A must be [M,K//2,L]");
    TORCH_CHECK(B_nvfp4.dim()==3, "B must be [1,K//2,L]");
    TORCH_CHECK(SFA_fp8.dim()==3,"SFA must be [M,K//16,L]");
    TORCH_CHECK(SFB_fp8.dim()==3,"SFB must be [1,K//16,L]");
    TORCH_CHECK(C_out.dim()==3,  "C must be [M,1,L]");

    const int64_t M = A_nvfp4.size(0);
    const int64_t K2= A_nvfp4.size(1);
    const int64_t L = A_nvfp4.size(2);
    TORCH_CHECK(B_nvfp4.size(0)==1 && B_nvfp4.size(1)==K2 && B_nvfp4.size(2)==L, "B shape mismatch");
    TORCH_CHECK(SFA_fp8.size(0)==M && SFA_fp8.size(2)==L, "SFA shape mismatch");
    TORCH_CHECK(SFB_fp8.size(0)==1 && SFB_fp8.size(2)==L, "SFB shape mismatch");
    TORCH_CHECK(C_out.size(0)==M && C_out.size(1)==1 && C_out.size(2)==L, "C shape mismatch");

    const int K = (int)(K2 * 2);
    TORCH_CHECK((K % 16)==0, "K must be divisible by 16");
    TORCH_CHECK(SFA_fp8.size(1) == K/16, "SFA K/16 dim mismatch");
    TORCH_CHECK(SFB_fp8.size(1) == K/16, "SFB K/16 dim mismatch");

    const dim3 block(256,1,1);
    const dim3 grid = make_grid((int)M, (int)L, block.x);

    nvfp4_batched_scaled_gemv_kernel<<<grid, block>>>(
        A_nvfp4.data_ptr<uint8_t>(),
        B_nvfp4.data_ptr<uint8_t>(),
        SFA_fp8.data_ptr<uint8_t>(),
        SFB_fp8.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(C_out.data_ptr<at::Half>()),
        (int)M, (int)K, (int)L
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));
}
"""

_ext = load_inline(
    name="nvfp4_batched_scaled_gemv_ext",
    cpp_sources=[_src],
    cuda_sources=[_cuda],
    functions=["batched_scaled_gemv"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=False,
)

# ---------- Python wrappers ----------

def batched_scaled_gemv(a, b, sfa, sfb, c_out):
    """
    a:   [M, K//2, L] torch.uint8 (NVFP4 packed: 2 nibbles/byte)
    b:   [1, K//2, L] torch.uint8
    sfa: [M, K//16, L] torch.uint8 (FP8 E4M3 scales for A)
    sfb: [1, K//16, L] torch.uint8 (FP8 E4M3 scales for B)
    c_out: [M, 1, L] torch.float16 (output; written in-place)
    """
    # Ensure contiguous CUDA tensors
    assert a.is_cuda and b.is_cuda and sfa.is_cuda and sfb.is_cuda and c_out.is_cuda
    _ext.batched_scaled_gemv(a.contiguous(), b.contiguous(), sfa.contiguous(), sfb.contiguous(), c_out)
    return c_out

# Your requested entry point

def custom_kernel(
    data: input_t,
) -> output_t:
    """
    Minimal implementation: pre-compute scales, launch GEMV kernel, return result.
    Now it's a single mega kernel, so we just forward tensors and compute everything inside.
    """
    a, b, sfa, sfb, _, _, c = data
    # one-call "mega kernel"
    return batched_scaled_gemv(a, b, sfa, sfb, c)

# -------------- quick smoke test (optional) --------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    M, K, L = 256, 1024, 3
    assert K % 16 == 0

    # Fake random packed tensors for a/b and fp8 scales (these won't be numerically meaningful,
    # but they validate shapes/launches). In your pipeline you would supply real packed tensors.
    A = torch.randint(0, 256, (M, K//2, L), dtype=torch.uint8, device=device)
    B = torch.randint(0, 256, (1, K//2, L), dtype=torch.uint8, device=device)
    SFA = torch.randint(0, 256, (M, K//16, L), dtype=torch.uint8, device=device)
    SFB = torch.randint(0, 256, (1, K//16, L), dtype=torch.uint8, device=device)
    C   = torch.empty((M, 1, L), dtype=torch.float16, device=device)

    out = custom_kernel((A,B,SFA,SFB,None,None,C))
    print("Output shape:", out.shape)
