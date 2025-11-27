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

// ============================================================================
// ======================== CONFIGURATION CONSTANTS ===========================
// ============================================================================
// tcgen05.mma constraints for .kind::mxf4nvf4:
//   M = 128 (fixed), N >= 8, K = 64 per MMA (dense)
//   Block scaling with .scale_vec::4X = 4 scales per row (block size 16)

#define M_TILE 128              // Rows per CTA (fixed by tcgen05 for mxf4)
#define N_TILE 8                // Minimum N for tcgen05 (B is broadcast)
#define K_TILE 64               // K per MMA operation for mxf4 (dense)
#define BLOCK_SIZE 128          // 4 warps = 1 warpgroup
#define NUM_BUFFERS 2           // Double buffering

// Derived constants
#define SCALES_PER_K (K_TILE / 16)      // 4 scales per K_TILE
#define BYTES_PER_K (K_TILE / 2)        // 32 bytes per K_TILE for FP4
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// SMEM sizes per buffer
#define SMEM_A_SIZE (M_TILE * BYTES_PER_K)          // 128 * 32 = 4096
#define SMEM_B_SIZE (N_TILE * BYTES_PER_K)          // 8 * 32 = 256
#define SMEM_SFA_SIZE (M_TILE * 16)                 // 128 * 16B stride = 2048
#define SMEM_SFB_SIZE 512                           // Scale B with padding

// ============================================================================
// ======================== TYPE CONVERSION HELPERS ===========================
// ============================================================================
__device__ __forceinline__ __half2 decode_fp4x2(uint8_t byte) {
    __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(
        static_cast<__nv_fp4x2_storage_t>(byte),
        __NV_E2M1
    );
    return *reinterpret_cast<__half2*>(&raw);
}

__device__ __forceinline__ float decode_fp8(int8_t byte) {
    __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(byte);
    __half_raw raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
    return __half2float(__ushort_as_half(raw.x));
}

// ============================================================================
// ======================== DESCRIPTOR HELPERS ================================
// ============================================================================
__device__ __forceinline__ uint64_t make_smem_desc(
    uint32_t smem_addr,
    uint32_t leading_byte_offset,
    uint32_t stride_byte_offset,
    int swizzle_mode = 0
) {
    // Shared memory descriptor layout (64-bit):
    // Bits 0-13:   matrix start address >> 4
    // Bits 16-29:  leading dimension byte offset >> 4
    // Bits 32-45:  stride dimension byte offset >> 4
    // Bits 46-48:  fixed 0b001
    // Bits 53-60:  fixed 0xB0
    // Bits 61-63:  swizzle mode
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

__device__ __forceinline__ uint32_t make_idesc_mxf4nvf4(
    int M, int N, int K,
    bool scale_type_ue4m3 = false,
    int scale_factor_a_id = 0,
    int scale_factor_b_id = 0
) {
    // Instruction descriptor for .kind::mxf4nvf4 (32-bit):
    // Bits 4-5:   scale factor B ID
    // Bits 7-9:   atype (E2M1 = 1)
    // Bits 10-11: btype (E2M1 = 1)
    // Bits 17-22: N >> 3
    // Bits 23:    scale type (0 = UE4M3, 1 = UE8M0)
    // Bits 27-28: M >> 7
    // Bits 29-30: scale factor A ID
    uint32_t idesc = 0;
    idesc |= ((scale_factor_b_id & 0x3) << 4);
    idesc |= (1 << 7);   // atype = E2M1
    idesc |= (1 << 10);  // btype = E2M1
    idesc |= (((N >> 3) & 0x3F) << 17);
    idesc |= ((scale_type_ue4m3 ? 0 : 1) << 23);  // UE4M3=0, UE8M0=1
    idesc |= (((M >> 7) & 0x3) << 27);
    idesc |= ((scale_factor_a_id & 0x3) << 29);
    return idesc;
}

// ============================================================================
// ================ SCALED DOT PRODUCT FOR 4 PACKED BYTES =====================
// ============================================================================
__device__ __forceinline__ __half2 dot_scaled_4bytes(
    uint32_t a4,
    uint32_t b4,
    __half2 scale_h2
) {
    // Extract bytes using bfe.u32
    uint32_t b_byte0, b_byte1, b_byte2, b_byte3;
    uint32_t a_byte0, a_byte1, a_byte2, a_byte3;

    asm("bfe.u32 %0, %1, 0, 8;"  : "=r"(b_byte0) : "r"(b4));
    asm("bfe.u32 %0, %1, 8, 8;"  : "=r"(b_byte1) : "r"(b4));
    asm("bfe.u32 %0, %1, 16, 8;" : "=r"(b_byte2) : "r"(b4));
    asm("bfe.u32 %0, %1, 24, 8;" : "=r"(b_byte3) : "r"(b4));

    asm("bfe.u32 %0, %1, 0, 8;"  : "=r"(a_byte0) : "r"(a4));
    asm("bfe.u32 %0, %1, 8, 8;"  : "=r"(a_byte1) : "r"(a4));
    asm("bfe.u32 %0, %1, 16, 8;" : "=r"(a_byte2) : "r"(a4));
    asm("bfe.u32 %0, %1, 24, 8;" : "=r"(a_byte3) : "r"(a4));

    // Compute scaled dot product
    __half2 acc = __hmul2(decode_fp4x2(a_byte0), __hmul2(decode_fp4x2(b_byte0), scale_h2));
    acc = __hfma2(decode_fp4x2(a_byte1), __hmul2(decode_fp4x2(b_byte1), scale_h2), acc);
    acc = __hfma2(decode_fp4x2(a_byte2), __hmul2(decode_fp4x2(b_byte2), scale_h2), acc);
    acc = __hfma2(decode_fp4x2(a_byte3), __hmul2(decode_fp4x2(b_byte3), scale_h2), acc);

    return acc;
}

// ============================================================================
// ============ REMAINDER COMPUTE FROM SHARED MEMORY (SOFTWARE) ===============
// ============================================================================
__device__ __forceinline__ void compute_remainder_smem(
    const uint8_t* sh_a,
    const uint8_t* sh_b,
    const uint8_t* sh_sfa,
    const uint8_t* sh_sfb,
    int remainder_scales,
    int m_count,
    int tid,
    float* remainder_acc
) {
    // Software FP4 dot product for remainder K elements that don't fill K_TILE=64
    // This handles cases where K % 64 != 0
    // Each thread processes multiple (m, sf) pairs in parallel

    // Parallelize across M rows - each thread handles subset of rows
    for (int m_local = tid; m_local < m_count; m_local += BLOCK_SIZE) {
        float acc = 0.0f;

        // Process all scale factors for this M row (serial per thread)
        for (int sf = 0; sf < remainder_scales; ++sf) {
            // Load and decode scale factors
            float sfa_val = decode_fp8(static_cast<int8_t>(sh_sfa[m_local * 16 + sf]));
            float sfb_val = decode_fp8(static_cast<int8_t>(sh_sfb[sf]));
            float scale = sfa_val * sfb_val;
            __half2 scale_h2 = __half2half2(__float2half(scale));

            int byte_base = sf << 3;  // sf * 8
            int a_offset = m_local * BYTES_PER_K + byte_base;

            uint32_t a4_0 = *reinterpret_cast<const uint32_t*>(&sh_a[a_offset]);
            uint32_t b4_0 = *reinterpret_cast<const uint32_t*>(&sh_b[byte_base]);
            uint32_t a4_1 = *reinterpret_cast<const uint32_t*>(&sh_a[a_offset + 4]);
            uint32_t b4_1 = *reinterpret_cast<const uint32_t*>(&sh_b[byte_base + 4]);

            __half2 acc_h2_0 = dot_scaled_4bytes(a4_0, b4_0, scale_h2);
            __half2 acc_h2_1 = dot_scaled_4bytes(a4_1, b4_1, scale_h2);

            float2 f0 = __half22float2(acc_h2_0);
            float2 f1 = __half22float2(acc_h2_1);
            acc += f0.x + f0.y + f1.x + f1.y;
        }

        remainder_acc[m_local] = acc;
    }
}

// ============================================================================
// ===================== CP.ASYNC HELPER MACROS ===============================
// ============================================================================
#define ASYNC_COPY_16(dst, src) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src))

#define ASYNC_COPY_4(dst, src) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(src))

// ============================================================================
// ===================== SMEM LAYOUT POST-PROCESSING ==========================
// ============================================================================
__device__ __forceinline__ void postprocess_sfa_smem(
    int8_t* sm_sfa,
    int tid
) {
    // Clear sign bits and replicate scale factors across 16B stride
    // Each row needs 4 scale bytes replicated to fill 16B for tcgen05.cp layout
    for (int i = tid; i < M_TILE; i += BLOCK_SIZE) {
        int* row = reinterpret_cast<int*>(sm_sfa + i * 16);
        int val = row[0] & 0x7F7F7F7F;  // Clear FP8 sign bits
        row[0] = val;
        row[1] = val;
        row[2] = val;
        row[3] = val;
    }
}

__device__ __forceinline__ void prepare_b_smem(
    int8_t* sm_b,
    int8_t* sm_sfb,
    const int8_t* g_b,
    const int8_t* g_sfb,
    int warpId,
    int laneId
) {
    // Load B vector and broadcast to N_TILE rows
    // B is 1 x K_TILE/2 = 32 bytes, replicate to 8 rows
    if (warpId == 0) {
        // Load 32 bytes of B data
        if (laneId < 8) {
            reinterpret_cast<int*>(sm_b)[laneId] = reinterpret_cast<const int*>(g_b)[laneId];
        }
        __syncwarp();

        // Replicate B to all 8 rows (each row is 32 bytes = 8 ints)
        for (int row = 1; row < N_TILE; ++row) {
            if (laneId < 8) {
                reinterpret_cast<int*>(sm_b)[row * 8 + laneId] =
                    reinterpret_cast<int*>(sm_b)[laneId];
            }
        }

        // Load and process sfb: 4 bytes, replicate with 16B stride
        if (laneId == 0) {
            int val = *reinterpret_cast<const int*>(g_sfb);
            val &= 0x7F7F7F7F;  // Clear sign bits

            // Fill scale factors with proper layout for tcgen05
            // Replicate each scale factor individually across 16B (matches working sub_tma.py)
            for (int k = 0; k < 4; ++k) {
                int8_t s = (val >> (k * 8)) & 0xFF;
                uint64_t s8 = 0x0101010101010101ULL * static_cast<uint8_t>(s);
                *reinterpret_cast<uint64_t*>(sm_sfb + k * 16 + 0) = s8;
                *reinterpret_cast<uint64_t*>(sm_sfb + k * 16 + 8) = s8;
            }
        }
    }
}

// ============================================================================
// ===================== TCGEN05 MMA TILE EXECUTION ===========================
// ============================================================================
__device__ __forceinline__ void execute_tcgen05_mma_tile(
    uint32_t taddr_a,
    uint32_t taddr_d,
    uint32_t taddr_sfa,
    uint32_t taddr_sfb,
    uint64_t sdesc_a,
    uint64_t sdesc_b,
    uint64_t sdesc_sfa,
    uint64_t sdesc_sfb,
    uint32_t idesc,
    uint64_t* mbar,
    int warpId,
    int laneId,
    bool accumulate,
    int parity
) {
    // Execute tcgen05 MMA pipeline:
    // 1. Copy A and scales from SMEM to TMEM
    // 2. Issue MMA instruction
    // 3. Commit and wait on mbarrier

    if (warpId == 0) {
        // SMEM -> TMEM copy for A matrix (128 x 32 bytes)
        asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;\n"
            :: "r"(taddr_a), "l"(sdesc_a));

        // SMEM -> TMEM copy for scale factors
        asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;\n"
            :: "r"(taddr_sfa), "l"(sdesc_sfa));
        asm volatile("tcgen05.cp.cta_group::1.4x256b [%0], %1;\n"
            :: "r"(taddr_sfb), "l"(sdesc_sfb));

        // Issue tcgen05.mma with block scaling
        // D = A * B with scale_A and scale_B
        // enable_input_d = accumulate (1 to add to existing D, 0 to overwrite)
        // NOTE: Last parameter MUST be a literal constant (0 or 1), not a register
        if (accumulate) {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
                "[%0], [%1], %2, %3, [%4], [%5], 1;\n"
                :: "r"(taddr_d), "r"(taddr_a), "l"(sdesc_b), "r"(idesc),
                   "r"(taddr_sfa), "r"(taddr_sfb)
            );
        } else {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
                "[%0], [%1], %2, %3, [%4], [%5], 0;\n"
                :: "r"(taddr_d), "r"(taddr_a), "l"(sdesc_b), "r"(idesc),
                   "r"(taddr_sfa), "r"(taddr_sfb)
            );
        }

        // Commit MMA and signal mbarrier
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
            :: "l"(mbar));
    }

    // All threads wait for MMA completion
    // After mbarrier.init (phase=0), first arrive transitions to phase=1
    // Subsequent arrives alternate phases: 1->2(parity 0)->3(parity 1)->...
    // We wait on the parity we're transitioning TO
    asm volatile(
        "{.reg .pred p; Lw%=: mbarrier.try_wait.parity.b64 p, [%0], %1; @!p bra Lw%=;}\n"
        :: "l"(mbar), "r"(parity)
    );
    asm volatile("tcgen05.fence::after_thread_sync;\n");
    __syncthreads();

    // Reinitialize mbarrier for next iteration
    if (warpId == 0 && laneId == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" :: "l"(mbar));
    }
    __syncthreads();
}

// ============================================================================
// ===================== TCGEN05 LOAD RESULTS FROM TMEM =======================
// ============================================================================
__device__ __forceinline__ void load_tmem_results(
    uint32_t taddr_d,
    float* sm_d,
    int tid
) {
    // Load D matrix from TMEM to SMEM
    // D is M_TILE x N_TILE = 128 x 8 floats = 4096 bytes
    // tcgen05.ld loads in warp-collective manner

    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    if (warpId == 0) {
        uint32_t sm_d_ptr = __cvta_generic_to_shared(sm_d) + laneId * 4;
        asm volatile(
            ".reg .b32 r<32>;\n"
            "tcgen05.ld.sync.aligned.16x256b.x8.b32 "
            "{r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,"
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
    }
}

// ============================================================================
// ========================== MAIN KERNEL FUNCTION ============================
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE)
gemv_tcgen05_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L
) {
    const int tile_m = blockIdx.x;
    const int tile_l = blockIdx.y;
    const int m_base = tile_m * M_TILE;
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;

    // Early exit for out-of-bounds CTAs
    if (m_base >= M || tile_l >= L) return;

    const int m_count = min(M_TILE, M - m_base);

// ============================================================================
// ===================== DIMENSION CALCULATIONS ===============================
// ============================================================================
    const int K_sf = K / 16;
    const int K_half = K / 2;
    const int tile_count = K / K_TILE;
    const int remainder_k = K % K_TILE;
    const int remainder_sf = remainder_k / 16;

// ============================================================================
// ===================== GLOBAL MEMORY POINTERS ===============================
// ============================================================================
    // A is [M, K/2, L] packed FP4, row-major
    // Global A offset = tile_l * (M * K/2) + (m_base * K + k_offset) / 2
    const size_t batch_stride_a = static_cast<size_t>(M) * K_half;
    const size_t batch_stride_sfa = static_cast<size_t>(M) * K_sf;
    const size_t batch_stride_b = K_half;
    const size_t batch_stride_sfb = K_sf;

    // Base pointers for this batch - k_offset added per tile
    const int8_t* g_a_batch = a + tile_l * batch_stride_a;
    const int8_t* g_sfa_batch = sfa + tile_l * batch_stride_sfa;
    const int8_t* g_b_batch = b + tile_l * batch_stride_b;
    const int8_t* g_sfb_batch = sfb + tile_l * batch_stride_sfb;

// ============================================================================
// ===================== SHARED MEMORY ALLOCATION =============================
// ============================================================================
    // Double-buffered shared memory for A and scales
    __shared__ alignas(128) int8_t sm_a[NUM_BUFFERS][SMEM_A_SIZE];
    __shared__ alignas(128) int8_t sm_sfa[NUM_BUFFERS][SMEM_SFA_SIZE];

    // Single buffer for B (loaded synchronously per K tile)
    __shared__ alignas(128) int8_t sm_b[SMEM_B_SIZE];
    __shared__ alignas(128) int8_t sm_sfb[SMEM_SFB_SIZE];

    // D matrix output buffer
    __shared__ alignas(128) float sm_d[M_TILE * N_TILE];

    // Remainder accumulator in shared memory (for K remainder software fallback)
    __shared__ alignas(128) float sm_remainder_acc[M_TILE];

    // TMEM addresses (written by tcgen05.alloc)
    __shared__ uint32_t sm_taddr_a, sm_taddr_d, sm_taddr_sfa, sm_taddr_sfb;

    // Mbarrier for MMA synchronization
    __shared__ uint64_t mbar_mma;

// ============================================================================
// ===================== TMEM ALLOCATION (WARP 0) =============================
// ============================================================================
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

// ============================================================================
// ===================== LOAD TMEM ADDRESSES ==================================
// ============================================================================
    uint32_t taddr_a, taddr_d, taddr_sfa, taddr_sfb;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_a)   : "r"((uint32_t)__cvta_generic_to_shared(&sm_taddr_a)));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_d)   : "r"((uint32_t)__cvta_generic_to_shared(&sm_taddr_d)));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_sfa) : "r"((uint32_t)__cvta_generic_to_shared(&sm_taddr_sfa)));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_sfb) : "r"((uint32_t)__cvta_generic_to_shared(&sm_taddr_sfb)));

// ============================================================================
// ===================== BUILD SMEM DESCRIPTORS ===============================
// ============================================================================
    uint32_t addr_b   = __cvta_generic_to_shared(sm_b);
    uint64_t sdesc_b  = make_smem_desc(addr_b, 32, 32);

    // Instruction descriptor for mxf4nvf4
    uint32_t idesc = make_idesc_mxf4nvf4(M_TILE, N_TILE, K_TILE, true, 0, 0);

// ============================================================================
// ===================== SMEM DESCRIPTORS FOR SFB ==============================
// ============================================================================
    uint32_t addr_sfb = __cvta_generic_to_shared(sm_sfb);
    uint64_t sdesc_sfb = make_smem_desc(addr_sfb, 16, 16);

// ============================================================================
// ===================== ASYNC COPY LAMBDA FOR A/B TILES =======================
// ============================================================================
    // Load A tile and scales for a K tile
    // A layout: each row is contiguous K elements, so for row r at k_offset:
    //   global addr = g_a_batch + (m_base + r) * K_half + k_byte_offset
    auto issue_a_tile_async = [&](int buf, int k_tile_idx) {
        const int k_offset = k_tile_idx * K_TILE;
        const int k_byte_offset = k_offset / 2;  // 32 bytes per K_TILE
        const int k_sf_offset = k_offset / 16;   // 4 scales per K_TILE

        uint32_t sh_a_base = __cvta_generic_to_shared(&sm_a[buf][0]);
        uint32_t sh_sfa_base = __cvta_generic_to_shared(&sm_sfa[buf][0]);

        // Load A: M_TILE rows x BYTES_PER_K (32) bytes each
        // Global A is row-major: row r -> g_a_batch + (m_base + r) * K_half + k_byte_offset
        for (int row = tid; row < M_TILE; row += BLOCK_SIZE) {
            const int8_t* g_row = g_a_batch + (m_base + row) * K_half + k_byte_offset;
            uint32_t sh_row = sh_a_base + row * BYTES_PER_K;
            // Load 32 bytes in two 16B chunks
            ASYNC_COPY_16(sh_row, g_row);
            ASYNC_COPY_16(sh_row + 16, g_row + 16);
        }

        // Load sfa: M_TILE rows x 4 bytes each, with 16B stride in SMEM
        // Global sfa is row-major: row r -> g_sfa_batch + (m_base + r) * K_sf + k_sf_offset
        for (int row = tid; row < M_TILE; row += BLOCK_SIZE) {
            uint32_t dst = sh_sfa_base + row * 16;
            const int8_t* src = g_sfa_batch + (m_base + row) * K_sf + k_sf_offset;
            ASYNC_COPY_4(dst, src);
        }

        asm volatile("cp.async.commit_group;");
    };

    // Load B tile and scales for a K tile (B changes per K tile!)
    auto issue_b_tile = [&](int k_tile_idx, int warpId, int laneId) {
        const int k_offset = k_tile_idx * K_TILE;
        const int k_byte_offset = k_offset / 2;
        const int k_sf_offset = k_offset / 16;

        const int8_t* g_b_tile = g_b_batch + k_byte_offset;
        const int8_t* g_sfb_tile = g_sfb_batch + k_sf_offset;

        prepare_b_smem(sm_b, sm_sfb, g_b_tile, g_sfb_tile, warpId, laneId);
    };

    auto issue_remainder_async = [&](int buf, int k_start) {
        const int k_byte_offset = k_start / 2;
        const int k_sf_offset = k_start / 16;

        uint32_t sh_a_base = __cvta_generic_to_shared(&sm_a[buf][0]);
        uint32_t sh_sfa_base = __cvta_generic_to_shared(&sm_sfa[buf][0]);

        int remainder_bytes = remainder_k / 2;

        // Load remainder A data - row by row with correct addressing
        for (int row = tid; row < M_TILE; row += BLOCK_SIZE) {
            const int8_t* g_row = g_a_batch + (m_base + row) * K_half + k_byte_offset;
            uint32_t sh_row = sh_a_base + row * BYTES_PER_K;
            // Only load as many bytes as needed
            for (int off = 0; off < remainder_bytes && off < BYTES_PER_K; off += 16) {
                ASYNC_COPY_16(sh_row + off, g_row + off);
            }
        }

        // Load remainder sfa
        for (int row = tid; row < M_TILE; row += BLOCK_SIZE) {
            uint32_t dst = sh_sfa_base + row * 16;
            const int8_t* src = g_sfa_batch + (m_base + row) * K_sf + k_sf_offset;
            ASYNC_COPY_4(dst, src);
        }

        asm volatile("cp.async.commit_group;");
    };

    // Load remainder B
    auto issue_remainder_b = [&](int k_start, int warpId, int laneId) {
        const int k_byte_offset = k_start / 2;
        const int k_sf_offset = k_start / 16;

        const int8_t* g_b_tile = g_b_batch + k_byte_offset;
        const int8_t* g_sfb_tile = g_sfb_batch + k_sf_offset;

        prepare_b_smem(sm_b, sm_sfb, g_b_tile, g_sfb_tile, warpId, laneId);
    };

// ============================================================================
// ===================== DOUBLE-BUFFERED MAIN LOOP ============================
// ============================================================================
    int buf = 0;
    bool has_remainder = (remainder_k > 0);

    // Initialize remainder accumulators in shared memory
    for (int i = tid; i < M_TILE; i += BLOCK_SIZE) {
        sm_remainder_acc[i] = 0.0f;
    }
    __syncthreads();

    // mbarrier.init sets phase=0. After arrive::one, phase becomes 1.
    // Since we reinitialize mbarrier each iteration, always wait on parity=1.
    const int mbar_parity = 1;

    if (tile_count > 0) {
        // Issue first tile load (A and B)
        issue_a_tile_async(0, 0);
        issue_b_tile(0, warpId, laneId);
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Post-process scale factors
        postprocess_sfa_smem(sm_sfa[0], tid);
        __syncthreads();

        for (int tile = 0; tile < tile_count; ++tile) {
            // ================================================================
            // Prefetch next tile or remainder (A only, B loaded after sync)
            // ================================================================
            if (tile + 1 < tile_count) {
                issue_a_tile_async(buf ^ 1, tile + 1);
            } else if (has_remainder) {
                issue_remainder_async(buf ^ 1, tile_count * K_TILE);
            }

            // ================================================================
            // Build descriptors for current buffer
            // ================================================================
            uint32_t addr_a = __cvta_generic_to_shared(&sm_a[buf][0]);
            uint32_t addr_sfa = __cvta_generic_to_shared(&sm_sfa[buf][0]);
            uint64_t sdesc_a = make_smem_desc(addr_a, 32, 32);
            uint64_t sdesc_sfa = make_smem_desc(addr_sfa, 16, 16);

            // ================================================================
            // Execute tcgen05 MMA for this tile
            // ================================================================
            bool accumulate = (tile > 0);  // First tile overwrites, rest accumulate
            execute_tcgen05_mma_tile(
                taddr_a, taddr_d, taddr_sfa, taddr_sfb,
                sdesc_a, sdesc_b, sdesc_sfa, sdesc_sfb,
                idesc, &mbar_mma, warpId, laneId, accumulate, mbar_parity
            );

            // ================================================================
            // Wait for prefetch and switch buffers
            // ================================================================
            if (tile + 1 < tile_count || has_remainder) {
                asm volatile("cp.async.wait_group 0;");
                __syncthreads();

                // Post-process scale factors for next buffer
                postprocess_sfa_smem(sm_sfa[buf ^ 1], tid);

                // Load B for next tile (B changes per K tile!)
                if (tile + 1 < tile_count) {
                    issue_b_tile(tile + 1, warpId, laneId);
                } else if (has_remainder) {
                    issue_remainder_b(tile_count * K_TILE, warpId, laneId);
                }
                __syncthreads();

                buf ^= 1;
            }
        }
    }

// ============================================================================
// ===================== REMAINDER PROCESSING (SOFTWARE) ======================
// ============================================================================
    if (has_remainder) {
        if (tile_count == 0) {
            // No full tiles - load remainder directly (both A and B)
            issue_remainder_async(0, 0);
            issue_remainder_b(0, warpId, laneId);
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            postprocess_sfa_smem(sm_sfa[0], tid);
            __syncthreads();
            buf = 0;
        }

        // Software fallback for remainder (K % 64 elements)
        // tcgen05.mma requires K=64, so we compute remainder on CUDA cores
        // Parallelizes across M rows, handles M remainder automatically via m_count
        compute_remainder_smem(
            reinterpret_cast<uint8_t*>(sm_a[buf]),
            reinterpret_cast<uint8_t*>(sm_b),
            reinterpret_cast<uint8_t*>(sm_sfa[buf]),
            reinterpret_cast<uint8_t*>(sm_sfb),
            remainder_sf,
            m_count,
            tid,
            sm_remainder_acc
        );
        __syncthreads();
    }

// ============================================================================
// ===================== LOAD RESULTS FROM TMEM ===============================
// ============================================================================
    if (tile_count > 0) {
        load_tmem_results(taddr_d, sm_d, tid);
    }
    __syncthreads();

// ============================================================================
// ===================== REDUCE AND WRITE OUTPUT ==============================
// ============================================================================
    // Each thread handles some M rows
    // D is stored in stripes: 32 lanes x N_TILE columns per stripe
    for (int m_local = tid; m_local < m_count; m_local += BLOCK_SIZE) {
        int g_row = m_base + m_local;

        float val = 0.0f;

        // Get result from TMEM (if we had full tiles)
        if (tile_count > 0) {
            int stripe = m_local >> 5;      // m_local / 32
            int lane = m_local & 31;        // m_local % 32
            int sm_idx = lane + stripe * N_TILE * 32;
            val = sm_d[sm_idx];
        }

        // Add remainder contribution from shared memory
        val += sm_remainder_acc[m_local];

        // Write output
        c[g_row * L + tile_l] = __float2half(val);
    }

// ============================================================================
// ===================== TMEM DEALLOCATION (WARP 3) ===========================
// ============================================================================
    __syncthreads();
    if (warpId == 3) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\n" :: "r"(taddr_a));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"  :: "r"(taddr_sfa));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"  :: "r"(taddr_sfb));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\n" :: "r"(taddr_d));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }
}

#undef ASYNC_COPY_16
#undef ASYNC_COPY_4

// ============================================================================
// ========================== HOST WRAPPER FUNCTION ===========================
// ============================================================================
torch::Tensor batched_scaled_gemv_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c
) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);

    // Grid: one CTA per (M_TILE rows, batch)
    // Each CTA processes all K for its M_TILE rows
    int m_tiles = (M + M_TILE - 1) / M_TILE;

    dim3 grid(m_tiles, L);
    dim3 block(BLOCK_SIZE);

    gemv_tcgen05_kernel<<<grid, block>>>(
        reinterpret_cast<const int8_t*>(a.data_ptr()),
        reinterpret_cast<const int8_t*>(b.data_ptr()),
        reinterpret_cast<const int8_t*>(sfa.data_ptr()),
        reinterpret_cast<const int8_t*>(sfb.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, K, L
    );

    return c;
}
'''


module = load_inline(
    name='batched_scaled_gemv_tcgen05_v2',
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
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device

    # Convert to int8 view for the kernel
    a_i8 = a.view(torch.int8)
    b_i8 = b.view(torch.int8)
    sfa_i8 = sfa_ref.to(device=device, non_blocking=True).view(torch.int8)
    sfb_i8 = sfb_ref.to(device=device, non_blocking=True).view(torch.int8)

    return module.batched_scaled_gemv_cuda(a_i8, b_i8, sfa_i8, sfb_i8, c)
