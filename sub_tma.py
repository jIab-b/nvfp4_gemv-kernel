import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


ROWS_PER_CTA = 128
COLS_PER_CTA = 64
VEC_TILE = 8
SCALE_GROUP = 16


cpp_source = """
#include <torch/extension.h>

torch::Tensor batched_scaled_gemv_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c,
    int64_t n_rows);
"""


cuda_source = f"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#define ROWS_PER_CTA {ROWS_PER_CTA}
#define COLS_PER_CTA {COLS_PER_CTA}
#define VEC_TILE {VEC_TILE}
#define SCALE_GROUP {SCALE_GROUP}
#define ENABLE_TMA_PTX 0

__device__ __forceinline__ float half_raw_to_float(const __half_raw& raw) {{
    return __half2float(__ushort_as_half(raw.x));
}}

__device__ __forceinline__ float decode_fp4(uint8_t nibble) {{
    __nv_fp4_storage_t storage = static_cast<__nv_fp4_storage_t>(nibble & 0xF);
    __half_raw raw = __nv_cvt_fp4_to_halfraw(storage, __NV_E2M1);
    return half_raw_to_float(raw);
}}

__device__ __forceinline__ float decode_fp8(int8_t byte) {{
    __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(byte);
    __half_raw raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
    return half_raw_to_float(raw);
}}

#if ENABLE_TMA_PTX
__device__ __forceinline__ void issue_tma_load(uint64_t mbar, const void* desc, void* smem) {{
    asm volatile(
        "tma.load.mbarrier::complete_tx::bytes [%0], [%1], [%2];\n"
        :
        : "l"(mbar), "r"(desc), "r"(smem)
        : "memory");
}}

__device__ __forceinline__ void issue_tc_cp(uint64_t tmem_addr, uint64_t desc) {{
    asm volatile(
        "tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [%0], %1;\n"
        :
        : "r"(tmem_addr), "l"(desc)
        : "memory");
}}

__device__ __forceinline__ void issue_tc_mma(uint64_t tmem_d, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc, uint32_t enable_d) {{
    asm volatile(
        "{{\\n"
        "  .reg .pred p;\\n"
        "  setp.ne.b32 p, %4, 0;\\n"
        "  tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, p;\\n"
        "}}\\n"
        :
        : "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_d)
        : "memory");
}}
#else
__device__ __forceinline__ void issue_tma_load(uint64_t, const void*, void*) {{}}
__device__ __forceinline__ void issue_tc_cp(uint64_t, uint64_t) {{}}
__device__ __forceinline__ void issue_tc_mma(uint64_t, uint64_t, uint64_t, uint32_t, uint32_t) {{}}
#endif

extern "C" __global__ void gemv_nvfp4_tma_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M,
    int K,
    int L,
    int N_rows,
    int groups_per_batch) {{

    if (groups_per_batch <= 0) return;

    const int block_m = blockIdx.x;
    const int batch_group = blockIdx.y;
    const int batch_id = batch_group / groups_per_batch;
    const int vec_group = batch_group % groups_per_batch;

    if (batch_id >= L) return;

    const int vec_base = vec_group * VEC_TILE;
    const int vec_count = max(0, min(VEC_TILE, N_rows - vec_base));
    if (vec_count <= 0) return;

    const int tid = threadIdx.x;
    const int global_row = block_m * ROWS_PER_CTA + tid;
    const int rows_this_block = min(ROWS_PER_CTA, M - block_m * ROWS_PER_CTA);

    const int K_half = K / 2;
    const int K_scale = K / SCALE_GROUP;

    const size_t batch_stride_a = static_cast<size_t>(M) * K_half;
    const size_t batch_stride_sfa = static_cast<size_t>(M) * K_scale;
    const size_t batch_stride_b = static_cast<size_t>(N_rows) * K_half;
    const size_t batch_stride_sfb = static_cast<size_t>(N_rows) * K_scale;

    const uint8_t* base_a = reinterpret_cast<const uint8_t*>(a);
    const uint8_t* base_sfa = reinterpret_cast<const uint8_t*>(sfa);
    const uint8_t* base_b = reinterpret_cast<const uint8_t*>(b);
    const uint8_t* base_sfb = reinterpret_cast<const uint8_t*>(sfb);

    const uint8_t* slab_a = base_a + batch_id * batch_stride_a;
    const uint8_t* slab_sfa = base_sfa + batch_id * batch_stride_sfa;
    const uint8_t* slab_b = base_b + batch_id * batch_stride_b;
    const uint8_t* slab_sfb = base_sfb + batch_id * batch_stride_sfb;

    __shared__ uint8_t smem_matrix_bytes[ROWS_PER_CTA * (COLS_PER_CTA / 2)];
    __shared__ uint8_t smem_matrix_scales[ROWS_PER_CTA * (COLS_PER_CTA / SCALE_GROUP)];
    __shared__ uint8_t smem_vector_bytes[VEC_TILE][COLS_PER_CTA / 2];
    __shared__ uint8_t smem_vector_scales[VEC_TILE][COLS_PER_CTA / SCALE_GROUP];
    __shared__ float smem_vector_values[VEC_TILE][COLS_PER_CTA];

    float acc[VEC_TILE];
    #pragma unroll
    for (int v = 0; v < VEC_TILE; ++v) {{
        acc[v] = 0.0f;
    }}

    for (int k0 = 0; k0 < K; k0 += COLS_PER_CTA) {{
        const int tile_elems = min(COLS_PER_CTA, K - k0);
        const int tile_bytes = (tile_elems + 1) >> 1;
        const int tile_scales = (tile_elems + SCALE_GROUP - 1) / SCALE_GROUP;

        if (tid < rows_this_block) {{
            const uint8_t* row_a = slab_a + static_cast<size_t>(global_row) * K_half + (k0 >> 1);
            const uint8_t* row_sfa = slab_sfa + static_cast<size_t>(global_row) * K_scale + k0 / SCALE_GROUP;
            uint8_t* dst_bytes = smem_matrix_bytes + tid * (COLS_PER_CTA / 2);
            uint8_t* dst_scales = smem_matrix_scales + tid * (COLS_PER_CTA / SCALE_GROUP);

            #pragma unroll
            for (int i = 0; i < COLS_PER_CTA / 2; ++i) {{
                dst_bytes[i] = (i < tile_bytes) ? row_a[i] : 0;
            }}

            #pragma unroll
            for (int i = 0; i < COLS_PER_CTA / SCALE_GROUP; ++i) {{
                dst_scales[i] = (i < tile_scales) ? row_sfa[i] : 0;
            }}
        }}

        for (int vec_lane = tid; vec_lane < tile_bytes * max(1, vec_count); vec_lane += blockDim.x) {{
            const int col = vec_lane / tile_bytes;
            const int byte_idx = vec_lane % tile_bytes;
            if (col < vec_count) {{
                const int vec_id = vec_base + col;
                const uint8_t* vec_ptr = slab_b + static_cast<size_t>(vec_id) * K_half + (k0 >> 1);
                smem_vector_bytes[col][byte_idx] = vec_ptr[byte_idx];
            }}
        }}

        for (int vec_lane = tid; vec_lane < tile_scales * max(1, vec_count); vec_lane += blockDim.x) {{
            const int col = vec_lane / tile_scales;
            const int scale_idx = vec_lane % tile_scales;
            if (col < vec_count) {{
                const int vec_id = vec_base + col;
                const uint8_t* vec_scale_ptr = slab_sfb + static_cast<size_t>(vec_id) * K_scale + k0 / SCALE_GROUP;
                smem_vector_scales[col][scale_idx] = vec_scale_ptr[scale_idx];
            }}
        }}

        __syncthreads();

        for (int col = 0; col < vec_count; ++col) {{
            for (int elem = tid; elem < tile_elems; elem += blockDim.x) {{
                const int byte_idx = elem >> 1;
                const bool high = elem & 1;
                uint8_t packed = smem_vector_bytes[col][byte_idx];
                uint8_t nibble = high ? (packed >> 4) : (packed & 0xF);
                const int scale_idx = elem / SCALE_GROUP;
                float scale = decode_fp8(static_cast<int8_t>(smem_vector_scales[col][scale_idx]));
                smem_vector_values[col][elem] = decode_fp4(nibble) * scale;
            }}
        }}

        __syncthreads();

        if (tid < rows_this_block && global_row < M) {{
            const uint8_t* row_bytes = smem_matrix_bytes + tid * (COLS_PER_CTA / 2);
            const uint8_t* row_scales = smem_matrix_scales + tid * (COLS_PER_CTA / SCALE_GROUP);

            for (int elem = 0; elem < tile_elems; ++elem) {{
                const int byte_idx = elem >> 1;
                const bool high = elem & 1;
                uint8_t packed = row_bytes[byte_idx];
                uint8_t nibble = high ? (packed >> 4) : (packed & 0xF);
                const int scale_idx = elem / SCALE_GROUP;
                float scale = decode_fp8(static_cast<int8_t>(row_scales[scale_idx]));
                float a_val = decode_fp4(nibble) * scale;
                #pragma unroll
                for (int col = 0; col < vec_count; ++col) {{
                    acc[col] += a_val * smem_vector_values[col][elem];
                }}
            }}
        }}

        __syncthreads();
    }}

    if (tid < rows_this_block && global_row < M) {{
        const size_t stride_n = static_cast<size_t>(M);
        const size_t stride_l = static_cast<size_t>(M) * static_cast<size_t>(N_rows);

        for (int col = 0; col < vec_count; ++col) {{
            const int vec_id = vec_base + col;
            half* out_ptr = c + static_cast<size_t>(global_row)
                               + static_cast<size_t>(vec_id) * stride_n
                               + static_cast<size_t>(batch_id) * stride_l;
            *out_ptr = __float2half(acc[col]);
        }}
    }}
}}

torch::Tensor batched_scaled_gemv_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c,
    int64_t n_rows) {{

    const int M = a.size(0);
    const int K = a.size(1) * 2;
    const int L = a.size(2);
    const int N_rows = static_cast<int>(n_rows);

    const int groups_per_batch = (N_rows + VEC_TILE - 1) / VEC_TILE;
    dim3 grid((M + ROWS_PER_CTA - 1) / ROWS_PER_CTA,
              max(1, L * max(1, groups_per_batch)),
              1);
    dim3 block(ROWS_PER_CTA, 1, 1);

    const int8_t* a_ptr = a.data_ptr<int8_t>();
    const int8_t* b_ptr = b.data_ptr<int8_t>();
    const int8_t* sfa_ptr = sfa.data_ptr<int8_t>();
    const int8_t* sfb_ptr = sfb.data_ptr<int8_t>();
    half* c_ptr = c.data_ptr<half>();

    gemv_nvfp4_tma_kernel<<<grid, block>>>(
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        M,
        K,
        L,
        N_rows,
        max(1, groups_per_batch));

    return c;
}}
"""


module = load_inline(
    name="batched_scaled_gemv_tma",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["batched_scaled_gemv_cuda"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
    ],
    with_cuda=True,
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device

    a_i8 = a.to(device=device, non_blocking=True).view(torch.int8)
    b_i8 = b.to(device=device, non_blocking=True).view(torch.int8)
    sfa_i8 = sfa_ref.to(device=device, non_blocking=True).view(torch.int8)
    sfb_i8 = sfb_ref.to(device=device, non_blocking=True).view(torch.int8)

    return module.batched_scaled_gemv_cuda(
        a_i8,
        b_i8,
        sfa_i8,
        sfb_i8,
        c,
        b.size(0),
    )
