import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// ============================================================================
// ======================== INITIALIZATION HELPERS ===========================
// ============================================================================
__device__ __forceinline__ half decode_fp4_half(uint8_t nibble) {
    __nv_fp4_storage_t storage = static_cast<__nv_fp4_storage_t>(nibble & 0xF);
    __half_raw raw = __nv_cvt_fp4_to_halfraw(storage, __NV_E2M1);
    half h; reinterpret_cast<__half_raw&>(h) = raw; return h;
}

__device__ __forceinline__ half decode_fp8_half(int8_t byte) {
    __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(byte);
    __half_raw raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
    half h; reinterpret_cast<__half_raw&>(h) = raw; return h;
}

// Only cp.async uses inline PTX for now; other PTX-level tuning can be added later.
__device__ __forceinline__ void cp_async_16(void* dst, const void* src, int valid_bytes) {
    // cp.async expects a 32-bit shared address, so convert explicitly.
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    if (valid_bytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(src));
    } else {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_addr), "l"(src), "r"(valid_bytes));
    }
}

__global__ void gemv_nvfp4_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L,
    int N_rows
) {
    // blockIdx.x -> row (m), blockIdx.y -> batch (l)
    const int m = blockIdx.x;
    const int l = blockIdx.y;
    const int tid = threadIdx.x;
    if (m >= M || l >= L) return;

    // =========================================================================
    // Perf hooks to add later (commented for now):
    // - cp.async.bulk / prefetch
    // - ldmatrix / tensor-core mma
    // - warp-specialized roles
    // =========================================================================

    const int K_half = K / 2;          // packed FP4 bytes per row
    const int K_sf   = K / 16;         // scale factors per row
    const size_t batch_stride_a   = static_cast<size_t>(M) * K_half;
    const size_t batch_stride_b   = static_cast<size_t>(N_rows) * K_half;
    const size_t batch_stride_sfa = static_cast<size_t>(M) * K_sf;
    const size_t batch_stride_sfb = static_cast<size_t>(N_rows) * K_sf;

    const uint8_t* base_a   = reinterpret_cast<const uint8_t*>(a);
    const uint8_t* base_b   = reinterpret_cast<const uint8_t*>(b);
    const uint8_t* base_sfa = reinterpret_cast<const uint8_t*>(sfa);
    const uint8_t* base_sfb = reinterpret_cast<const uint8_t*>(sfb);

    const uint8_t* batch_a   = base_a + l * batch_stride_a;
    const uint8_t* batch_b   = base_b + l * batch_stride_b;
    const uint8_t* batch_sfa = base_sfa + l * batch_stride_sfa;
    const uint8_t* batch_sfb = base_sfb + l * batch_stride_sfb;

    const uint8_t* row_a   = batch_a + static_cast<size_t>(m) * K_half;
    const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m) * K_sf;

    constexpr int TILE_K = 1024;                       // elements
    constexpr int BYTES_PER_TILE = TILE_K / 2;        // packed FP4 bytes
    constexpr int SCALES_PER_TILE = TILE_K / 16;      // FP8 scale bytes

    __shared__ uint8_t sh_a[2][BYTES_PER_TILE];
    __shared__ uint8_t sh_b[2][BYTES_PER_TILE];
    __shared__ half2   sh_dec_a[2][BYTES_PER_TILE];
    __shared__ half2   sh_dec_b[2][BYTES_PER_TILE];
    __shared__ half    sh_scale_a[2][SCALES_PER_TILE];
    __shared__ half    sh_scale_b[2][SCALES_PER_TILE];
    __shared__ float   smem_acc[128];

    float acc0 = 0.0f, acc1 = 0.0f;
    int stage = 0;
    int tile = 0;

    // Prefetch the first tile
    {
        const int elems = min(TILE_K, K);
        const int bytes = (elems + 1) >> 1;
        for (int byte = tid * 16; byte < bytes; byte += blockDim.x * 16) {
            const int remaining = bytes - byte;
            const int cp_bytes = remaining >= 16 ? 16 : remaining;
            cp_async_16(sh_a[stage] + byte, row_a + byte, cp_bytes);
            cp_async_16(sh_b[stage] + byte, batch_b + byte, cp_bytes);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    while (true) {
        // Wait for current stage to land
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        const int elems_this_tile   = min(TILE_K, K - tile);
        const int bytes_this_tile   = (elems_this_tile + 1) >> 1;
        const int scales_this_tile  = (elems_this_tile + 15) >> 4;

        // Decode scales for this tile
        if (tid < scales_this_tile) {
            sh_scale_a[stage][tid] = decode_fp8_half(static_cast<int8_t>(row_sfa[(tile >> 4) + tid]));
            sh_scale_b[stage][tid] = decode_fp8_half(static_cast<int8_t>(batch_sfb[(tile >> 4) + tid]));
        }

        // Launch prefetch for the next tile (overlaps with compute)
        const int next_tile = tile + TILE_K;
        const int next_stage = stage ^ 1;
        if (next_tile < K) {
            const int next_elems = min(TILE_K, K - next_tile);
            const int next_bytes = (next_elems + 1) >> 1;
            for (int byte = tid * 16; byte < next_bytes; byte += blockDim.x * 16) {
                const int remaining = next_bytes - byte;
                const int cp_bytes = remaining >= 16 ? 16 : remaining;
                cp_async_16(sh_a[next_stage] + byte, row_a + (next_tile >> 1) + byte, cp_bytes);
                cp_async_16(sh_b[next_stage] + byte, batch_b + (next_tile >> 1) + byte, cp_bytes);
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }

        // Decode packed FP4 -> half2 once per byte
        for (int byte = tid; byte < bytes_this_tile; byte += blockDim.x) {
            uint8_t a_byte = sh_a[stage][byte];
            uint8_t b_byte = sh_b[stage][byte];
            half2 a2 = __halves2half2(decode_fp4_half(a_byte & 0xF), decode_fp4_half(a_byte >> 4));
            half2 b2 = __halves2half2(decode_fp4_half(b_byte & 0xF), decode_fp4_half(b_byte >> 4));
            sh_dec_a[stage][byte] = a2;
            sh_dec_b[stage][byte] = b2;
        }

        __syncthreads();

        // Compute with decoded data
        for (int byte = tid; byte < bytes_this_tile; byte += blockDim.x) {
            const int elem_base = byte * 2;
            const int sf = elem_base >> 4;  // /16
            half2 sa = __halves2half2(sh_scale_a[stage][sf], sh_scale_a[stage][sf]);
            half2 sb = __halves2half2(sh_scale_b[stage][sf], sh_scale_b[stage][sf]);
            half2 prod = __hmul2(__hmul2(sh_dec_a[stage][byte], sa), __hmul2(sh_dec_b[stage][byte], sb));
            acc0 += __low2float(prod);
            acc1 += __high2float(prod);
        }

        if (next_tile >= K) break;
        tile = next_tile;
        stage = next_stage;
    }

    float acc = acc0 + acc1;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Block-level reduction
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) smem_acc[warp_id] = acc;

    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < (blockDim.x >> 5)) ? smem_acc[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
            c[c_idx] = __float2half(block_sum);
        }
    }
}


torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    dim3 grid(M, L);
    dim3 block(128);

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    gemv_nvfp4_kernel<<<grid, block>>>(
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        M, K, L,
        N_rows
    );

    return c;
}

"""

module = load_inline(
    name='batched_scaled_gemv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batched_scaled_gemv_cuda'],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-std=c++17',
        '-gencode=arch=compute_100a,code=sm_100a'
    ],
    with_cuda=True,
    verbose=False
)

def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, sfa_per, sfb_per, c = data
    #device = a.device
    return module.batched_scaled_gemv_cuda(
        a,
        b,
        sfa_ref,
        sfb_ref,
        c
    )
