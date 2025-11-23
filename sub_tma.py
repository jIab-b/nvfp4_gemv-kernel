import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


cpp_source = r'''
#include <torch/extension.h>

torch::Tensor batched_scaled_gemv_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c);
'''


cuda_source = r'''
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#ifndef DEBUG_GEMV_PIPELINE
#define DEBUG_GEMV_PIPELINE 1
#endif

// ============================================================================
// Constants
// ============================================================================
constexpr int M_TILE = 128;   // rows per CTA
constexpr int N_TILE = 8;     // padded N (GEMV → keep column 0)
constexpr int K_TILE = 64;    // K per tile (matches tcgen05 requirement)


// ============================================================================
// Helpers
// ============================================================================
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr,
                                                   uint32_t leading_byte_offset,
                                                   uint32_t stride_byte_offset,
                                                   int swizzle_mode = 0) {
    auto encode = [](uint32_t x) -> uint64_t { return (x & 0x3FFFF) >> 4; };
    uint64_t desc = 0;
    desc |= encode(smem_addr);
    desc |= (encode(leading_byte_offset) << 16);
    desc |= (encode(stride_byte_offset)  << 32);
    desc |= (1ULL << 46);
    desc |= (0xB0ULL << 53);
    desc |= (static_cast<uint64_t>(swizzle_mode) << 61);
    return desc;
}

__device__ __forceinline__ uint32_t make_idesc_mxf4nvf4(int M, int N, int K,
                                                        bool scale_type_ue4m3 = false,
                                                        int scale_factor_a_id = 0,
                                                        int scale_factor_b_id = 0) {
    uint32_t idesc = 0;
    idesc |= ((scale_factor_b_id & 0x3) << 4);
    idesc |= (1 << 7);   // atype = E2M1
    idesc |= (1 << 10);  // btype = E2M1
    idesc |= (((N >> 3) & 0x3F) << 17);
    idesc |= ((scale_type_ue4m3 ? 0 : 1) << 23);
    idesc |= (((M >> 7) & 0x3) << 27);
    idesc |= ((scale_factor_a_id & 0x3) << 29);
    return idesc;
}

__device__ __forceinline__ float decode_fp4_nibble(uint8_t nibble) {
    __nv_fp4_storage_t storage = static_cast<__nv_fp4_storage_t>(nibble & 0xF);
    __half_raw raw = __nv_cvt_fp4_to_halfraw(storage, __NV_E2M1);
    return __half2float(__ushort_as_half(raw.x));
}

__device__ __forceinline__ float decode_fp8_byte(int8_t byte) {
    __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(static_cast<uint8_t>(byte));
    __half_raw raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
    return __half2float(__ushort_as_half(raw.x));
}

// ============================================================================
// cp.async helpers (GMEM -> SMEM)
// ============================================================================
template<int BYTES>
__device__ __forceinline__ void cp_async_cg(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"(smem_u32), "l"(gmem_ptr), "n"(BYTES));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;" ::);
}


// ============================================================================
// Kernel
// ============================================================================
__global__ void gemv_tcgen05_kernel(const int8_t* a,
                                    const int8_t* b,
                                    const int8_t* sfa,
                                    const int8_t* sfb,
                                    half* c,
                                    int M, int K, int L) {
    // ---------------------- CTA indices ----------------------
    const int tile_m = blockIdx.x;
    const int tile_k = blockIdx.y;
    const int tile_l = blockIdx.z;
    const int m_base = tile_m * M_TILE;
    const int k_base = tile_k * K_TILE;
    if (m_base >= M || k_base >= K || tile_l >= L) return;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x & 31;

    // ---------------------- Shared memory ----------------------
    __shared__ alignas(128) int8_t sm_a[4096];      // 128 x 64 fp4 packed
    __shared__ alignas(128) int8_t sm_sfa[2048];    // 128 rows x 16B stride
    __shared__ alignas(128) int8_t sm_b[256];       // padded B
    __shared__ alignas(128) int8_t sm_sfb[512];
    __shared__ alignas(128) float  sm_d[1024];      // 128 x 8 accum

    __shared__ uint32_t sm_taddr_a, sm_taddr_sfa, sm_taddr_sfb, sm_taddr_d;
    __shared__ uint64_t mbar_mma;

    // ---------------------- Global base pointers ----------------------
    const size_t stride_a   = static_cast<size_t>(M) * (K / 2);
    const size_t stride_sfa = static_cast<size_t>(M) * (K / 16);
    const size_t stride_b   = (K / 2);
    const size_t stride_sfb = (K / 16);

    const int8_t* g_a_tile   = a   + tile_l * stride_a   + (m_base * K + k_base) / 2;
    const int8_t* g_sfa_tile = sfa + tile_l * stride_sfa + m_base * (K / 16) + k_base / 16;
    const int8_t* g_b_tile   = b   + tile_l * stride_b   + k_base / 2;
    const int8_t* g_sfb_tile = sfb + tile_l * stride_sfb + k_base / 16;



    // =====================================================================
    // Stage 1: GMEM -> SMEM using cp.async
    // =====================================================================
    // A tile: 4096B. Use warp0 to issue cp.async in 16B chunks (256 copies).
    if (warpId == 0) {
        int offset = laneId * 16;
        for (; offset < 4096; offset += 32 * 16) {
            cp_async_cg<16>(sm_a + offset, g_a_tile + offset);
        }
    }
#if DEBUG_GEMV_PIPELINE
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("DBG after cp.async A loop\n");
    }
#endif

    // sfa tile: 2048B, use 4B cp.async to keep src aligned. Map offset->row explicitly.
    if (warpId == 0) {
        int offset = laneId * 16;            // offset in smem bytes
        for (; offset < 2048; offset += 32 * 16) {
            int row = offset / 16;           // target row in smem
            cp_async_cg<4>(sm_sfa + offset, g_sfa_tile + row * 4);
        }
    }
#if DEBUG_GEMV_PIPELINE
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("DBG after cp.async sfa loop\n");
    }
#endif

    // Commit and wait for all async copies to complete.
    if (warpId == 0 && laneId == 0) cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

#if DEBUG_GEMV_PIPELINE
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("DBG cp.async done: sm_a[0]=%d sm_a[1]=%d sm_sfa[0]=%d sm_sfa[1]=%d\n",
               sm_a[0], sm_a[1], sm_sfa[0], sm_sfa[1]);
    }
#endif

    // Post-process sfa in SMEM to clear sign bits (UE4M3 requirement)
    for (int i = warpId * 32 + laneId; i < 128; i += 128) {
        int* row = reinterpret_cast<int*>(sm_sfa + i * 16);
        int val = row[0] & 0x7F7F7F7F;
        row[0] = val;
    }

    // B + sfb: tiny; load with warp0
    if (warpId == 0) {
        if (laneId < 8) { // 32B packed B
            reinterpret_cast<int*>(sm_b)[laneId] = reinterpret_cast<const int*>(g_b_tile)[laneId];
        }
        if (laneId == 0) {
            int val = *reinterpret_cast<const int*>(g_sfb_tile);
            val &= 0x7F7F7F7F;
            for (int k = 0; k < 4; ++k) {
                int8_t s = (val >> (k * 8)) & 0xFF;
                uint64_t s8 = 0x0101010101010101ULL * static_cast<uint8_t>(s);
                *reinterpret_cast<uint64_t*>(sm_sfb + k * 16) = s8;
            }
        }
        // pad B tail with zeros
        if (laneId < 64) reinterpret_cast<int*>(sm_b)[8 + laneId] = 0;
    }

    __syncthreads();

    // =====================================================================
    // Stage 2: TMEM allocation + descriptors
    // =====================================================================
    uint32_t addr_a   = __cvta_generic_to_shared(sm_a);
    uint32_t addr_b   = __cvta_generic_to_shared(sm_b);
    uint32_t addr_sfa = __cvta_generic_to_shared(sm_sfa);
    uint32_t addr_sfb = __cvta_generic_to_shared(sm_sfb);

    uint64_t sdesc_a   = make_smem_desc(addr_a, 32, 32);
    uint64_t sdesc_b   = make_smem_desc(addr_b, 32, 32);
    uint64_t sdesc_sfa = make_smem_desc(addr_sfa, 16, 16);
    uint64_t sdesc_sfb = make_smem_desc(addr_sfb, 16, 16);

    uint32_t idesc = make_idesc_mxf4nvf4(M_TILE, N_TILE, K_TILE, true, 0, 0);

    if (warpId == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;\n"
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%1], 128;\n"
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%2], 32;\n"
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%3], 32;\n"
            :: "l"(&sm_taddr_a), "l"(&sm_taddr_d), "l"(&sm_taddr_sfa), "l"(&sm_taddr_sfb)
        );
        if (laneId == 0) {
            asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" :: "l"(&mbar_mma));
        }
    }
    __syncthreads();

    uint32_t taddr_a = 0, taddr_sfa = 0, taddr_sfb = 0, taddr_d = 0;
    if (warpId == 0) {
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_a)   : "r"((uint32_t)__cvta_generic_to_shared(&sm_taddr_a)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_sfa) : "r"((uint32_t)__cvta_generic_to_shared(&sm_taddr_sfa)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_sfb) : "r"((uint32_t)__cvta_generic_to_shared(&sm_taddr_sfb)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_d)   : "r"((uint32_t)__cvta_generic_to_shared(&sm_taddr_d)));
    }
    __syncthreads();

    // =====================================================================
    // Stage 3: SMEM -> TMEM, MMA, Commit
    // =====================================================================
    if (warpId == 0) {
        asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;\n" :: "r"(taddr_a), "l"(sdesc_a));
        asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;\n" :: "r"(taddr_sfa), "l"(sdesc_sfa));
        asm volatile("tcgen05.cp.cta_group::1.4x256b   [%0], %1;\n"  :: "r"(taddr_sfb), "l"(sdesc_sfb));

#if DEBUG_GEMV_PIPELINE
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && laneId == 0) {
            printf("DBG after tc.cp\n");
        }
#endif

        asm volatile(
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
            "[%0], [%1], %2, %3, [%4], [%5], 0;\n"
            :: "r"(taddr_d), "r"(taddr_a), "l"(sdesc_b), "r"(idesc), "r"(taddr_sfa), "r"(taddr_sfb)
        );
#if DEBUG_GEMV_PIPELINE
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && laneId == 0) {
            printf("DBG after tc.mma\n");
        }
#endif
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n" :: "l"(&mbar_mma));

#if DEBUG_GEMV_PIPELINE
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && laneId == 0) {
            printf("DBG cp->mma issued: taddr_a=%u taddr_sfa=%u taddr_sfb=%u taddr_d=%u idesc=%u\n",
                   taddr_a, taddr_sfa, taddr_sfb, taddr_d, idesc);
        }
#endif
    }

    asm volatile("{.reg .pred p; Lw%=: mbarrier.try_wait.parity.b64 p, [%0], 1; @!p bra Lw%=;}\n" :: "l"(&mbar_mma));
    asm volatile("tcgen05.fence::after_thread_sync;\n");

    // =====================================================================
    // Stage 4: TMEM -> SMEM, reduction, dealloc
    // =====================================================================
    if (warpId == 0) {
        uint32_t sm_d_ptr = __cvta_generic_to_shared(sm_d) + threadIdx.x * 4;
        asm volatile(
            ".reg .b32 r<32>;\n"
            "tcgen05.ld.sync.aligned.16x256b.x8.b32 {r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,"
            "r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31}, [%1];\n"
            "st.shared.b32 [%0+0], r0;\n"
            "st.shared.b32 [%0+128], r1;\n"
            "st.shared.b32 [%0+256], r2;\n"
            "st.shared.b32 [%0+384], r3;\n"
            "st.shared.b32 [%0+512], r4;\n"
            "st.shared.b32 [%0+640], r5;\n"
            "st.shared.b32 [%0+768], r6;\n"
            "st.shared.b32 [%0+896], r7;\n"
            "st.shared.b32 [%0+1024], r8;\n"
            "st.shared.b32 [%0+1152], r9;\n"
            "st.shared.b32 [%0+1280], r10;\n"
            "st.shared.b32 [%0+1408], r11;\n"
            "st.shared.b32 [%0+1536], r12;\n"
            "st.shared.b32 [%0+1664], r13;\n"
            "st.shared.b32 [%0+1792], r14;\n"
            "st.shared.b32 [%0+1920], r15;\n"
            "st.shared.b32 [%0+2048], r16;\n"
            "st.shared.b32 [%0+2176], r17;\n"
            "st.shared.b32 [%0+2304], r18;\n"
            "st.shared.b32 [%0+2432], r19;\n"
            "st.shared.b32 [%0+2560], r20;\n"
            "st.shared.b32 [%0+2688], r21;\n"
            "st.shared.b32 [%0+2816], r22;\n"
            "st.shared.b32 [%0+2944], r23;\n"
            "st.shared.b32 [%0+3072], r24;\n"
            "st.shared.b32 [%0+3200], r25;\n"
            "st.shared.b32 [%0+3328], r26;\n"
            "st.shared.b32 [%0+3456], r27;\n"
            "st.shared.b32 [%0+3584], r28;\n"
            "st.shared.b32 [%0+3712], r29;\n"
            "st.shared.b32 [%0+3840], r30;\n"
            "st.shared.b32 [%0+3968], r31;\n"
            :: "r"(sm_d_ptr), "r"(taddr_d)
        );
#if DEBUG_GEMV_PIPELINE
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && laneId == 0) {
            printf("DBG after tc.ld\n");
        }
#endif
    }
    __syncthreads();

#if DEBUG_GEMV_PIPELINE
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("DBG sm_d head: %f %f %f %f | tail %f %f\n",
               sm_d[0], sm_d[1], sm_d[2], sm_d[3], sm_d[30], sm_d[31]);
    }
#endif

    // Accumulate to global
    for (int i = warpId * 32 + laneId; i < M_TILE; i += 128) {
        int g_row = m_base + i;
        if (g_row < M) {
            int stripe = i >> 5;
            int lane = i & 31;
            int sm_idx = lane + stripe * N_TILE * 32;
            half val = __float2half(sm_d[sm_idx]);
            atomicAdd(reinterpret_cast<half*>(&c[g_row * L + tile_l]), val);
        }
    }

    // Dealloc (warp3)
    if (warpId == 3) {
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_a)   : "r"((uint32_t)__cvta_generic_to_shared(&sm_taddr_a)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_sfa) : "r"((uint32_t)__cvta_generic_to_shared(&sm_taddr_sfa)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_sfb) : "r"((uint32_t)__cvta_generic_to_shared(&sm_taddr_sfb)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_d)   : "r"((uint32_t)__cvta_generic_to_shared(&sm_taddr_d)));

        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\n" :: "r"(taddr_a));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"  :: "r"(taddr_sfa));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"  :: "r"(taddr_sfb));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\n" :: "r"(taddr_d));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
#if DEBUG_GEMV_PIPELINE
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && laneId == 0) {
            printf("DBG after tc.dealloc\n");
        }
#endif
    }
}


torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a,
                                       torch::Tensor b,
                                       torch::Tensor sfa,
                                       torch::Tensor sfb,
                                       torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);

    int m_tiles = (M + M_TILE - 1) / M_TILE;
    int k_tiles = (K + K_TILE - 1) / K_TILE;
    dim3 grid(m_tiles, k_tiles, L);
    dim3 block(128, 1, 1);

    gemv_tcgen05_kernel<<<grid, block>>>(reinterpret_cast<int8_t*>(a.data_ptr()),
                                         reinterpret_cast<int8_t*>(b.data_ptr()),
                                         reinterpret_cast<int8_t*>(sfa.data_ptr()),
                                         reinterpret_cast<int8_t*>(sfb.data_ptr()),
                                         reinterpret_cast<half*>(c.data_ptr()),
                                         M, K, L);
    return c;
}
'''


module = load_inline(
    name="gemv_tcgen05_tma",
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
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device

    c.zero_()

    a_i8 = a.view(torch.int8)
    b_i8 = b.view(torch.int8)
    sfa_i8 = sfa_ref.to(device=device, non_blocking=True).view(torch.int8)
    sfb_i8 = sfb_ref.to(device=device, non_blocking=True).view(torch.int8)

    return module.batched_scaled_gemv_cuda(a_i8, b_i8, sfa_i8, sfb_i8, c)
