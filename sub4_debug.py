import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t, TestSpec

cpp_source = """
#include <torch/extension.h>
torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdio>

__device__ __forceinline__ float half_raw_to_float(const __half_raw& raw) {
    return __half2float(__ushort_as_half(raw.x));
}

__device__ __forceinline__ float decode_fp4(uint8_t nibble) {
    __nv_fp4_storage_t storage = static_cast<__nv_fp4_storage_t>(nibble & 0xF);
    __half_raw raw = __nv_cvt_fp4_to_halfraw(storage, __NV_E2M1);
    return half_raw_to_float(raw);
}

__device__ __forceinline__ float decode_fp8(int8_t byte) {
    __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(byte);
    __half_raw raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
    return half_raw_to_float(raw);
}

enum class MmaKind {
  Tf32 = 0,
  F16 = 1,
  I8 = 2,
  F8F6F4 = 3,
  Mxf8F6F4 = 4,
  Mxf4 = 5,
  Mxf4Nvf4 = 6
};

struct MmaConfig {
    MmaKind kind;
    int m_size;
    int n_size;
    int k_size;
    bool trans_a;
    bool trans_b;
    int scale_vec_size_log;
};

__device__ uint64_t make_shared_desc(void* smem_ptr, uint32_t leading_byte, uint32_t stride_byte, uint32_t swizzle = 0, bool k_major = true) {
    uint64_t desc = 0;
    uintptr_t addr = reinterpret_cast<uintptr_t>(smem_ptr);
    desc |= (addr >> 4) & 0x3FFFULL;
    desc |= (static_cast<uint64_t>(leading_byte & 0xFFFFF) << 14);
    desc |= (static_cast<uint64_t>(stride_byte & 0xFFFFF) << 34);
    desc |= (static_cast<uint64_t>(!k_major) << 54);
    desc |= (static_cast<uint64_t>(swizzle & 0x7) << 55);
    return desc;
}

__device__ uint32_t device_ctz(uint32_t x) {
    uint32_t result = 0;
    while ((x & 1) == 0 && result < 32) {
        x >>= 1;
        result++;
    }
    return result;
}

__device__ uint32_t make_instr_desc(const MmaConfig& config) {
    uint32_t desc = 0;
    desc |= (static_cast<uint32_t>(config.kind) & 0xF);
    uint32_t m_log = device_ctz(config.m_size / 8);
    desc |= (m_log & 0xF) << 4;
    uint32_t n_log = device_ctz(config.n_size / 8);
    desc |= (n_log & 0xF) << 8;
    uint32_t k_log = device_ctz(config.k_size / 8);
    desc |= (k_log & 0x7) << 12;
    desc |= (config.trans_a ? 1U : 0) << 15;
    desc |= (config.trans_b ? 1U : 0) << 16;
    desc |= (static_cast<uint32_t>(config.scale_vec_size_log & 0x3) << 17);
    return desc;
}

struct SharedDescParams {
    void* smem_ptr;
    uint32_t leading_byte;
    uint32_t stride_byte;
    uint32_t swizzle;
    bool k_major;
};

__device__ SharedDescParams get_default_shared_params(void* smem_ptr, uint32_t leading_byte, uint32_t stride_byte, uint32_t swizzle = 0, bool k_major = true) {
    SharedDescParams params;
    params.smem_ptr = smem_ptr;
    params.leading_byte = leading_byte;
    params.stride_byte = stride_byte;
    params.swizzle = swizzle;
    params.k_major = k_major;
    return params;
}

__device__ const MmaConfig mma_config = {
    MmaKind::Mxf4Nvf4,
    128,
    32,
    128,
    false,
    false,
    1
};

struct ZeroMaskConfig {
    uint32_t non_zero_mask;
    uint32_t shift_amount;
};

__device__ uint64_t make_zero_mask_desc(const ZeroMaskConfig& config) {
    uint64_t desc = 0;
    desc |= (static_cast<uint64_t>(config.non_zero_mask & 0xFF));
    desc |= (static_cast<uint64_t>(config.shift_amount & 0x1F) << 8);
    return desc;
}

__device__ __forceinline__ bool debug_gate(int m_block, int l, int tid) {
    return (m_block == 0 && l == 0 && tid == 0);
}

__device__ __forceinline__ void debug_emit_stage(const char* tag, int stage, uint8_t payload, int extra = -1) {
    char c = static_cast<char>('A' + (stage & 0x1F));
    if (extra >= 0) {
        printf("DBG %s %c 0x%02x %u extra=%d\\n", tag, c, payload, static_cast<unsigned int>(payload), extra);
    } else {
        printf("DBG %s %c 0x%02x %u\\n", tag, c, payload, static_cast<unsigned int>(payload));
    }
}

__device__ __forceinline__ void debug_emit_float(const char* tag, float value, int extra = -1) {
    if (extra >= 0) {
        printf("DBG %s %.6f extra=%d\\n", tag, value, extra);
    } else {
        printf("DBG %s %.6f\\n", tag, value);
    }
}

__global__ void gemv_nvfp4_tc_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L,
    int N_rows
) {
    #define THREADS_PER_CTA 128
    #define ROWS_PER_CTA 128
    #define K_TILE 128
    #define MIN_N 32

    int m_block = blockIdx.x;
    int l = blockIdx.y;
    int tid = threadIdx.x;

    int rows_this_tile = min(ROWS_PER_CTA, M - m_block * ROWS_PER_CTA);
    if (rows_this_tile <= 0 || l >= L) return;

    bool dbg_lane = debug_gate(m_block, l, tid);
    if (dbg_lane) {
        printf("DBG_start rows=%d K=%d L=%d N=%d block=(%d,%d) tid=%d\\n", rows_this_tile, K, L, N_rows, m_block, l, tid);
    }

    const int K_sf = K / 16;
    const int K_half = K / 2;
    const size_t batch_stride_a = static_cast<size_t>(M) * K_half;
    const size_t batch_stride_b = static_cast<size_t>(N_rows) * K_half;
    const size_t batch_stride_sfa = static_cast<size_t>(M) * K_sf;
    const size_t batch_stride_sfb = static_cast<size_t>(N_rows) * K_sf;

    const uint8_t* base_a = reinterpret_cast<const uint8_t*>(a);
    const uint8_t* base_b = reinterpret_cast<const uint8_t*>(b);
    const uint8_t* base_sfa = reinterpret_cast<const uint8_t*>(sfa);
    const uint8_t* base_sfb = reinterpret_cast<const uint8_t*>(sfb);

    const uint8_t* batch_a = base_a + l * batch_stride_a;
    const uint8_t* batch_b = base_b + l * batch_stride_b;
    const uint8_t* batch_sfa = base_sfa + l * batch_stride_sfa;
    const uint8_t* batch_sfb = base_sfb + l * batch_stride_sfb;

    const uint8_t* row_a = batch_a + static_cast<size_t>(m_block * ROWS_PER_CTA) * K_half;
    const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m_block * ROWS_PER_CTA) * K_sf;

    __shared__ uint8_t vector_smem[K_TILE / 2];
    __shared__ int8_t vector_scale_smem[K_TILE / 16];
    __shared__ uint8_t matrix_smem[ROWS_PER_CTA * (K_TILE / 2)];
    __shared__ int8_t matrix_scale_smem[ROWS_PER_CTA * (K_TILE / 16)];
    __shared__ float accum_shared[ROWS_PER_CTA * MIN_N];

    __shared__ uint32_t accum_taddr_smem;
    __shared__ uint32_t vector_taddr_smem;
    __shared__ uint32_t vector_scale_taddr_smem;
    __shared__ uint32_t matrix_taddr_smem;
    __shared__ uint32_t matrix_scale_taddr_smem;

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" : : "r"((uint32_t)(uintptr_t)&accum_taddr_smem));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" : : "r"((uint32_t)(uintptr_t)&vector_taddr_smem));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" : : "r"((uint32_t)(uintptr_t)&vector_scale_taddr_smem));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" : : "r"((uint32_t)(uintptr_t)&matrix_taddr_smem));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" : : "r"((uint32_t)(uintptr_t)&matrix_scale_taddr_smem));

    uint32_t accum_taddr = accum_taddr_smem;
    uint32_t vector_taddr = vector_taddr_smem;
    uint32_t vector_scale_taddr = vector_scale_taddr_smem;
    uint32_t matrix_taddr = matrix_taddr_smem;
    uint32_t matrix_scale_taddr = matrix_scale_taddr_smem;

    __shared__ float zero_smem[32];
    for (int i = tid; i < 32; i += THREADS_PER_CTA) {
        zero_smem[i] = 0.0f;
    }
    __syncthreads();

    SharedDescParams zero_params = get_default_shared_params(zero_smem, 0, 0);
    uint64_t zero_desc = make_shared_desc(zero_params.smem_ptr, zero_params.leading_byte, zero_params.stride_byte, zero_params.swizzle, zero_params.k_major);

    asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" : : "r"(accum_taddr), "l"(zero_desc));
    asm volatile("tcgen05.wait::st.sync.aligned;");

    if (dbg_lane) {
        uint8_t sample = reinterpret_cast<uint8_t*>(zero_smem)[0];
        debug_emit_stage("zero_copy", 0, sample, 0);
    }

    for (int k_block = 0; k_block < (K + K_TILE - 1) / K_TILE; ++k_block) {
        int k_start = k_block * K_TILE;
        int tile_elems = min(K_TILE, K - k_start);
        if (tile_elems <= 0) continue;

        for (int i = tid; i < (tile_elems + 1) / 2; i += THREADS_PER_CTA) {
            int byte_idx = k_start / 2 + i;
            if (byte_idx < batch_stride_b) vector_smem[i] = batch_b[byte_idx];
        }

        for (int i = tid; i < (tile_elems + 15) / 16; i += THREADS_PER_CTA) {
            int sf_idx = k_start / 16 + i;
            if (sf_idx < K_sf) vector_scale_smem[i] = batch_sfb[sf_idx];
        }
        __syncthreads();

        if (dbg_lane) {
            debug_emit_stage("vector_bytes", 1, vector_smem[0], k_block);
            debug_emit_stage("vector_scales", 2, static_cast<uint8_t>(vector_scale_smem[0]), k_block);
        }

        SharedDescParams vector_params = get_default_shared_params(vector_smem, 0, 0);
        uint64_t vector_desc = make_shared_desc(vector_params.smem_ptr, vector_params.leading_byte, vector_params.stride_byte, vector_params.swizzle, vector_params.k_major);

        asm volatile("tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [%0], %1;" : : "r"(vector_taddr), "l"(vector_desc));
        asm volatile("tcgen05.wait::st.sync.aligned;");

        SharedDescParams vscale_params = get_default_shared_params(vector_scale_smem, 0, 0);
        uint64_t vscale_desc = make_shared_desc(vscale_params.smem_ptr, vscale_params.leading_byte, vscale_params.stride_byte, vscale_params.swizzle, vscale_params.k_major);
        asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" : : "r"(vector_scale_taddr), "l"(vscale_desc));
        asm volatile("tcgen05.wait::st.sync.aligned;");

        for (int i = tid; i < ROWS_PER_CTA * ((tile_elems + 1) / 2); i += THREADS_PER_CTA) {
            int m_local = i / ((tile_elems + 1) / 2);
            int byte_local = i % ((tile_elems + 1) / 2);
            if (m_local < rows_this_tile) {
                int byte_idx = k_start / 2 + byte_local;
                matrix_smem[i] = row_a[m_local * K_half + byte_idx];
            } else {
                matrix_smem[i] = 0;
            }
        }

        for (int i = tid; i < ROWS_PER_CTA * ((tile_elems + 15) / 16); i += THREADS_PER_CTA) {
            int m_local = i / ((tile_elems + 15) / 16);
            int sf_local = i % ((tile_elems + 15) / 16);
            if (m_local < rows_this_tile) {
                int sf_idx = k_start / 16 + sf_local;
                matrix_scale_smem[i] = row_sfa[m_local * K_sf + sf_idx];
            } else {
                matrix_scale_smem[i] = 0;
            }
        }
        __syncthreads();

        if (dbg_lane) {
            debug_emit_stage("matrix_bytes", 3, matrix_smem[0], k_block);
            debug_emit_stage("matrix_scales", 4, static_cast<uint8_t>(matrix_scale_smem[0]), k_block);
        }

        SharedDescParams matrix_params = get_default_shared_params(matrix_smem, K_TILE / 2, 1);
        uint64_t matrix_desc = make_shared_desc(matrix_params.smem_ptr, matrix_params.leading_byte, matrix_params.stride_byte, matrix_params.swizzle, matrix_params.k_major);

        asm volatile("tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [%0], %1;" : : "r"(matrix_taddr), "l"(matrix_desc));
        asm volatile("tcgen05.wait::st.sync.aligned;");

        SharedDescParams mscale_params = get_default_shared_params(matrix_scale_smem, K_TILE / 16, 1);
        uint64_t mscale_desc = make_shared_desc(mscale_params.smem_ptr, mscale_params.leading_byte, mscale_params.stride_byte, mscale_params.swizzle, mscale_params.k_major);
        asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" : : "r"(matrix_scale_taddr), "l"(mscale_desc));
        asm volatile("tcgen05.wait::st.sync.aligned;");

        uint32_t instr_desc = make_instr_desc(mma_config);
        uint32_t enable_input_d = (k_block > 0) ? 1 : 0;
        asm volatile("{\\n"
                     "  .reg .pred p;\\n"
                     "  setp.ne.b32 p, %6, 0;\\n"
                     "  tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], p;\\n"
                     "}\\n"
            : : "r"(accum_taddr), "l"(matrix_desc), "l"(vector_desc), "r"(instr_desc), "r"(matrix_scale_taddr), "r"(vector_scale_taddr), "r"(enable_input_d));
        __threadfence();

        if (dbg_lane) {
            debug_emit_float("mma_tile", static_cast<float>(k_block), k_block);
        }
    }

    for (int i = tid; i < rows_this_tile; i += THREADS_PER_CTA) {
        float sum = 0.0f;
        for (int n = 0; n < MIN_N; n++) {
            float temp_val;
            uint32_t elem_addr = accum_taddr + (i * MIN_N + n) * 4;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                : "=f"(temp_val) : "r"(elem_addr));
            sum += temp_val;
        }
        accum_shared[i * MIN_N] = sum;
        if (dbg_lane && i < 4) {
            debug_emit_float("accum_sum", sum, i);
        }
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    __syncthreads();

    for (int i = tid; i < rows_this_tile; i += THREADS_PER_CTA) {
        int m = m_block * ROWS_PER_CTA + i;
        size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
        c[c_idx] = __float2half(accum_shared[i * MIN_N]);
        if (dbg_lane && i < 4) {
            debug_emit_float("store_value", accum_shared[i * MIN_N], i);
        }
    }

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(accum_taddr));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(vector_taddr));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(vector_scale_taddr));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(matrix_taddr));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(matrix_scale_taddr));
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    dim3 grid((M + ROWS_PER_CTA - 1) / ROWS_PER_CTA, L);
    dim3 block(THREADS_PER_CTA);

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    gemv_nvfp4_tc_kernel<<<grid, block>>>(
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
    name="batched_scaled_gemv_tc_debug",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["batched_scaled_gemv_cuda"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a"
    ],
    with_cuda=True,
    verbose=False
)

THREADS_PER_CTA = 128
ROWS_PER_CTA = 128
K_TILE = 128
MIN_N = 32

def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device
    a_i8 = a.view(torch.int8)
    b_i8 = b.view(torch.int8)
    sfa_i8 = sfa_ref.to(device=device, non_blocking=True).view(torch.int8)
    sfb_i8 = sfb_ref.to(device=device, non_blocking=True).view(torch.int8)
    return module.batched_scaled_gemv_cuda(
        a_i8,
        b_i8,
        sfa_i8,
        sfb_i8,
        c
    )

def generate_debug_case(m: int = 128, k: int = 256, l: int = 1) -> input_t:
    device = torch.device("cuda")
    torch.manual_seed(0)
    a_bytes = torch.arange(m * (k // 2) * l, dtype=torch.uint8, device=device) % 16
    a = a_bytes.reshape(m, k // 2, l).clone().view(torch.float4_e2m1fn_x2)
    b_bytes = torch.arange((k // 2) * l, dtype=torch.uint8, device=device) % 16
    b = b_bytes.reshape(1, k // 2, l).clone().view(torch.float4_e2m1fn_x2)
    k_sf = k // 16
    sfa_vals = torch.arange(m * k_sf * l, dtype=torch.int8, device="cpu") % 16
    sfb_vals = torch.arange(k_sf * l, dtype=torch.int8, device="cpu") % 16
    sfa = sfa_vals.reshape(m, k_sf, l).clone().to(dtype=torch.float8_e4m3fn)
    sfb = sfb_vals.reshape(1, k_sf, l).clone().to(dtype=torch.float8_e4m3fn)
    sfa_perm = sfa.to(device=device)
    sfb_perm = sfb.to(device=device)
    c = torch.zeros((m, 1, l), dtype=torch.float16, device=device)
    return (a, b, sfa, sfb, sfa_perm, sfb_perm, c)

def _print_tensor(name: str, tensor: torch.Tensor, limit: int = 8) -> None:
    flat = tensor.detach().flatten()
    if flat.numel() == 0:
        values = []
    else:
        values = flat[:limit].to(torch.float32).cpu().tolist()
    print(f"[sub4_debug] {name} shape={tuple(tensor.shape)} dtype={tensor.dtype} sample={values}", flush=True)

def run_debug_pipeline(spec: TestSpec | None = None) -> output_t:
    if spec is None:
        spec = {"m": 128, "k": 256, "l": 1, "seed": 0}
    data = generate_debug_case(spec["m"], spec["k"], spec["l"])
    a, b, sfa, sfb, _, _, c = data
    _print_tensor("a_bytes", a.view(torch.uint8))
    _print_tensor("b_bytes", b.view(torch.uint8))
    _print_tensor("sfa_bytes", sfa.view(torch.uint8))
    _print_tensor("sfb_bytes", sfb.view(torch.uint8))
    out = custom_kernel(data)
    torch.cuda.synchronize()
    _print_tensor("c_out", out)
    return out

if __name__ == "__main__":
    run_debug_pipeline()

