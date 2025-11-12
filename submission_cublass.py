import torch
from torch.utils.cpp_extension import load_inline

src_cpp = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <vector>
#include <stdexcept>
#include <sstream>

// CUDA 13 narrow-precision headers (available in 12.9+/13.0 toolkits)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>   // __nv_fp8_e4m3, etc.
#include <cuda_fp4.h>   // __nv_fp4_e2m1, conversions

// Error helpers
#define CUDA_CHECK(err) do { cudaError_t e = (err); if (e != cudaSuccess) { \
  std::stringstream ss; ss << "CUDA error " << cudaGetErrorName(e) << ": " << cudaGetErrorString(e); \
  throw std::runtime_error(ss.str()); }} while(0)

#define CUBLASLT_CHECK(st) do { cublasStatus_t s = (st); if (s != CUBLAS_STATUS_SUCCESS) { \
  std::stringstream ss; ss << "cublasLt error code " << int(s); throw std::runtime_error(ss.str()); }} while(0)

static inline int ceil_div_int(int a, int b){ return (a + b - 1) / b; }

// Pack sfa/sfb (outer x k16) -> cuBLASLt tiled scale layout (UE4M3) for VEC16:
// Each 128x4 "tile" of scale factors is laid out with index mapping:
//   offset_in_tile = (outer%32)*16 + (outer/32)*4 + inner  where outer in [0,127], inner in [0,3]
// Tiles are arranged row-major by (sf_outer, sf_inner): base = (sf_inner + sf_outer*sf_inner_dim)*128
// where sf_outer = outer//128, sf_inner = (k16_index)//4 and sf_inner_dim = ceil((K/16)/4).
// See cuBLASLt docs 1D Block Scaling Factors Layout. (Ref) 
// https://docs.nvidia.com/cuda/cublas/index.html 3.1.5.2.2
__global__ void pack_scales_ue4m3_tiled_kernel(
    const uint8_t* __restrict__ in, // [outer_dim, k16]
    uint8_t* __restrict__ out,      // tiled buffer
    int outer_dim,                  // = M for A, = N for B (N=1 here)
    int k16                         // = K/16
){
    int outer = blockIdx.y * blockDim.y + threadIdx.y; // row in outer dimension
    int inner = blockIdx.x * blockDim.x + threadIdx.x; // k/16 index
    if (outer >= outer_dim || inner >= k16) return;

    int tile_outer = outer >> 7;           // /128
    int tile_inner = inner >> 2;           // /4
    int in_tile_outer = outer & 127;       // %128
    int in_tile_inner = inner & 3;         // %4

    int sf_inner_dim = (k16 + 3) >> 2;     // ceil(k16/4)

    // within 128-element tile
    int offset_in_tile = ( (in_tile_outer & 31) * 16 ) + ((in_tile_outer >> 5) * 4) + in_tile_inner;
    int tile_idx = tile_inner + tile_outer * sf_inner_dim;

    out[tile_idx * 128 + offset_in_tile] = in[outer * k16 + inner];
}

static void pack_scales_ue4m3_tiled(
    const at::Tensor& in_2d, // uint8: [outer_dim, k16]
    at::Tensor& out_tiled    // uint8: size = 128 * ceil(k16/4) * ceil(outer_dim/128)
){
    TORCH_CHECK(in_2d.dtype() == at::kByte, "scale input must be uint8 (fp8 e4m3fnuz/UE4M3 storage)");
    TORCH_CHECK(out_tiled.dtype() == at::kByte, "tiled scale buffer must be uint8");
    TORCH_CHECK(in_2d.is_cuda() && out_tiled.is_cuda(), "tensors must be CUDA");
    int outer_dim = in_2d.size(0);
    int k16 = in_2d.size(1);

    const int bx = 32, by = 4;
    dim3 block(bx, by);
    dim3 grid(ceil_div_int(k16, bx), ceil_div_int(outer_dim, by));

    // Zero-fill the whole tiled buffer (cuBLAS docs: tiles beyond bounds should be zero)
    CUDA_CHECK(cudaMemsetAsync(out_tiled.data_ptr(), 0, out_tiled.numel() * sizeof(uint8_t), at::cuda::getCurrentCUDAStream()));

    pack_scales_ue4m3_tiled_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        in_2d.data_ptr<uint8_t>(),
        out_tiled.data_ptr<uint8_t>(),
        outer_dim, k16
    );
    CUDA_CHECK(cudaGetLastError());
}

// Create a cublasLtHandle with RAII
struct LtHandle {
    cublasLtHandle_t h{nullptr};
    LtHandle(){ CUBLASLT_CHECK(cublasLtCreate(&h)); }
    ~LtHandle(){ if (h) cublasLtDestroy(h); }
};

// Run one TN matmul for batch l: D(Mx1) = A(MxK) * B(Kx1), A/B in NVFP4 (e2m1), scales in UE4M3 (vec16)
static void run_fp4_vec16_tn_matmul_one(
    cublasLtHandle_t lt,
    int M, int K,
    const void* A, int64_t lda_elems,
    const void* B, int64_t ldb_elems,
    void* D, int64_t ldd_elems,
    const void* sfa_tiled, size_t sfa_tiled_bytes,
    const void* sfb_tiled, size_t sfb_tiled_bytes
){
    // cuBLASLt wants TN for block-scaled FP4 on Blackwell (compute 10.x+), computeType=FP32, D=FP16
    // Ref: cuBLAS 12.8+ docs, Narrow precision usage, FP4 block scaling requirements.
    cublasOperation_t transa = CUBLAS_OP_T; // TN path
    cublasOperation_t transb = CUBLAS_OP_N;

    // Descriptor (compute type FP32)
    cublasLtMatmulDesc_t matmul_desc=nullptr;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Set A/B scale modes to vector UE4M3 (per-16 K)
    // enum cublasLtMatmulMatrixScale_t has VEC16_UE4M3 for FP4 (UE4M3 scales)
    cublasLtMatmulMatrixScale_t vec16_ue4m3 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec16_ue4m3, sizeof(vec16_ue4m3)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec16_ue4m3, sizeof(vec16_ue4m3)));

    // Provide scale pointers (device). Layout is the tiled layout we packed.
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sfa_tiled, sizeof(sfa_tiled)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sfb_tiled, sizeof(sfb_tiled)));

    // Matrix layouts: A/B = NVFP4 (e2m1), D = FP16
    cublasLtMatrixLayout_t a_desc=nullptr, b_desc=nullptr, d_desc=nullptr;
    // cuBLASLt uses "rows, cols, ld" in elements; we choose ROW-major order with LD = cols (K or 1).
    // We'll rely on default order ROW (unless explicitly set), which is supported for narrow types.
    // A: MxK stored K-major (row-major); pass lda = K
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, M, K, lda_elems));
    // B: Kx1 stored K-major; pass ldb = K
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, K, 1, ldb_elems));
    // D: Mx1 fp16; ldd = 1
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16F, M, 1, ldd_elems));

    // Alpha/Beta in FP32 compute
    float alpha = 1.0f, beta = 0.0f;

    // Heuristic selection (no extra workspace to keep things simple/portable)
    cublasLtMatmulPreference_t pref=nullptr;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t workspace_size = 0; // can be grown if desired
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heur{};
    int returned = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, matmul_desc, a_desc, b_desc, d_desc, d_desc, pref, 1, &heur, &returned));
    if (returned == 0) {
        // Fall back to direct matmul without heuristic (rare)
        CUBLASLT_CHECK(cublasLtMatmul(lt, matmul_desc, &alpha,
            A, a_desc,
            B, b_desc,
            &beta,
            D, d_desc,
            D, d_desc,
            nullptr, nullptr, 0, at::cuda::getCurrentCUDAStream()));
    } else {
        CUBLASLT_CHECK(cublasLtMatmul(lt, matmul_desc, &alpha,
            A, a_desc,
            B, b_desc,
            &beta,
            D, d_desc,
            D, d_desc,
            &heur.algo, nullptr, 0, at::cuda::getCurrentCUDAStream()));
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(d_desc);
    cublasLtMatmulDescDestroy(matmul_desc);
}

// Main entry: (a,b,sfa,sfb,c)
at::Tensor batched_scaled_gemv_cuda(
    const at::Tensor& a,   // [M,K,L] NVFP4(e2m1) packed (uint8 storage)
    const at::Tensor& b,   // [1,K,L] NVFP4(e2m1) packed (uint8 storage)
    const at::Tensor& sfa, // [M,K//16,L] fp8 e4m3fnuz -> stored as uint8
    const at::Tensor& sfb, // [1,K//16,L] fp8 e4m3fnuz -> stored as uint8
    const at::Tensor& c    // [M,1,L] fp16 (output buffer or placeholder)
){
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && sfa.is_cuda() && sfb.is_cuda() && c.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(a.scalar_type() == at::kByte && b.scalar_type() == at::kByte, "a,b must be packed NVFP4 (uint8) storage");
    TORCH_CHECK(sfa.scalar_type() == at::kByte && sfb.scalar_type() == at::kByte, "sfa,sfb must be uint8 storage (UE4M3)");
    TORCH_CHECK(c.scalar_type() == at::kHalf, "c must be fp16");

    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t L = a.size(2);
    TORCH_CHECK(b.size(0) == 1 && b.size(1) == K && b.size(2) == L, "b must be [1,K,L]");
    TORCH_CHECK(sfa.size(0) == M && sfa.size(1) == K/16 && sfa.size(2) == L, "sfa must be [M,K//16,L]");
    TORCH_CHECK(sfb.size(0) == 1 && sfb.size(1) == K/16 && sfb.size(2) == L, "sfb must be [1,K//16,L]");
    TORCH_CHECK(c.size(0) == M && c.size(1) == 1 && c.size(2) == L, "c must be [M,1,L]");
    TORCH_CHECK((K % 64)==0, "K must be divisible by 64");

    // Quick runtime arch check
    int device = a.get_device();
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    TORCH_CHECK(prop.major >= 10, "This kernel requires Blackwell SM100+ (compute capability 10.x). Got sm_", prop.major, ".", prop.minor);

    // Create Lt handle once per call
    LtHandle lth;

    // For each l, we pack sfa[:, :, l] and sfb[0, :, l] into the tiled UE4M3 layout and issue one TN matmul
    int k16 = (int)(K/16);
    int sf_inner_dim = ceil_div_int(k16, 4);

    // Tiled scale buffer sizes (bytes)
    // A-scales: tiles = ceil(M/128) * ceil(k16/4); each tile 128 bytes (UE4M3)
    int tiles_A_outer = ceil_div_int((int)M, 128);
    int tiles_AB_inner = sf_inner_dim;
    size_t sfa_tiled_bytes = size_t(tiles_A_outer) * size_t(tiles_AB_inner) * 128;

    // B-scales: outer=N=1 -> ceil(1/128)=1 tile in outer dimension; inner same
    int tiles_B_outer = 1;
    size_t sfb_tiled_bytes = size_t(tiles_B_outer) * size_t(tiles_AB_inner) * 128;

    // Allocate per-call buffers for tiled scales reused across l (we repack in-place)
    auto options_u8 = at::TensorOptions().dtype(at::kByte).device(a.device());
    at::Tensor sfa_tiled_buf = at::empty({(long)(sfa_tiled_bytes)}, options_u8);
    at::Tensor sfb_tiled_buf = at::empty({(long)(sfb_tiled_bytes)}, options_u8);

    // Leading dimensions in elements (row-major):
    int64_t lda = K; // A is [M,K]
    int64_t ldb = K; // B is [K,1]
    int64_t ldd = 1; // D is [M,1]

    // Strides between batch planes (in elements of their logical types)
    // But we don't use batched matmul for scales (no A/B scale batch stride attribute), so loop over L.
    // NVFP4 is 4 bits => 2 elems per byte. Advance raw pointer by (M*K)/2 bytes between batch planes.
    size_t a_batch_bytes = (size_t)(M*K/2);
    size_t b_batch_bytes = (size_t)(K/2);
    size_t d_batch_bytes = (size_t)(M) * sizeof(at::Half);

    const uint8_t* a_base = a.data_ptr<uint8_t>();
    const uint8_t* b_base = b.data_ptr<uint8_t>();
    at::Half* c_base = (at::Half*)c.data_ptr<at::Half>();

    for (int64_t l = 0; l < L; ++l) {
        // Slices for this batch
        at::Tensor sfa_l = sfa.select(2, l).contiguous();         // [M, K/16] u8
        at::Tensor sfb_l = sfb.select(2, l).contiguous().view({1, k16}); // [1, K/16] u8

        // Pack into tiled UE4M3 layout expected by cuBLASLt (A outer=M, B outer=N=1)
        // A-scales
        {
            at::Tensor sfa2d = sfa_l; // already [M, k16]
            // view as 2D (guard shapes)
            TORCH_CHECK(sfa2d.size(0)==M && sfa2d.size(1)==k16);
            pack_scales_ue4m3_tiled(sfa2d, sfa_tiled_buf);
        }
        // B-scales
        {
            at::Tensor sfb2d = sfb_l; // [1, k16]
            TORCH_CHECK(sfb2d.size(0)==1 && sfb2d.size(1)==k16);
            pack_scales_ue4m3_tiled(sfb2d, sfb_tiled_buf);
        }

        const void* A_ptr = (const void*)(a_base + l * a_batch_bytes);
        const void* B_ptr = (const void*)(b_base + l * b_batch_bytes);
        void* D_ptr = (void*)((uint8_t*)c_base + l * d_batch_bytes);

        run_fp4_vec16_tn_matmul_one(
            lth.h,
            (int)M, (int)K,
            A_ptr, lda,
            B_ptr, ldb,
            D_ptr, ldd,
            sfa_tiled_buf.data_ptr(), sfa_tiled_bytes,
            sfb_tiled_buf.data_ptr(), sfb_tiled_bytes
        );
    }

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batched_scaled_gemv_cuda", &batched_scaled_gemv_cuda,
        "B200 NVFP4(e2m1) batched GEMV with per-16K UE4M3 scale (cuBLASLt TN path)");
}
'''

src_cuda = r'''
// CUDA kernels are in the C++ TU above for simplicity (we compiled with NVCC already).
// This file is intentionally empty to allow load_inline to pass a CUDA source as well.
'''

module = load_inline(
    name="b200_nvfp4_gemv",
    cpp_sources=[src_cpp],
    cuda_sources=[src_cuda],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3", "-std=c++17",
        "-gencode=arch=compute_100,code=sm_100",    # Blackwell SM100 (B200)
        "-Xptxas=-v"
    ],
    extra_ldflags=["-lcublasLt","-lcublas"],
    verbose=False
)

def custom_kernel(data):
    """
    Custom implementation of batched scaled GEMV using B200 tensor cores with hardware scaling.
    Expects:
      a: [M,K,L] uint8 storage of nvfp4(e2m1) packed 2-per-byte, K-major
      b: [1,K,L] uint8 storage of nvfp4(e2m1) packed 2-per-byte, K-major
      sfa: [M,K//16,L] uint8 storage of fp8(e4m3fnuz) (UE4M3)
      sfb: [1,K//16,L] uint8 storage of fp8(e4m3fnuz) (UE4M3)
      c: [M,1,L] torch.float16
    """
    a, b, sfa, sfb, _, _, c = data
    # Light sanity conversions: if user passed float8_e4m3fnuz tensors, reinterpret storage as u8
    if sfa.dtype != torch.uint8 and str(sfa.dtype).endswith("float8_e4m3fnuz"):
        sfa = sfa.view(torch.uint8)
    if sfb.dtype != torch.uint8 and str(sfb.dtype).endswith("float8_e4m3fnuz"):
        sfb = sfb.view(torch.uint8)
    # a/b might be provided as an nvfp4 wrapper dtype in some stacks; require uint8 storage
    if a.dtype != torch.uint8:
        a = a.view(torch.uint8)
    if b.dtype != torch.uint8:
        b = b.view(torch.uint8)

    return module.batched_scaled_gemv_cuda(a, b, sfa, sfb, c)