import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

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

__device__ __forceinline__ float smem_decode_fp4(const int8_t* smem, int row_stride_bytes, int row, int k) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(smem);
    int byte_idx = row * row_stride_bytes + (k >> 1);
    uint8_t packed = bytes[byte_idx];
    uint8_t nib = (k & 1) ? (packed >> 4) : (packed & 0xF);
    return decode_fp4_nibble(nib);
}

#ifndef DEBUG_GEMV_PIPELINE
#define DEBUG_GEMV_PIPELINE 1
#endif


// Kernel launch parameters
constexpr int M_TILE = 128;
constexpr int N_TILE = 8;    // padded N
constexpr int K_TILE = 64;

// Descriptor construction helpers
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr, uint32_t leading_byte_offset,
                                                     uint32_t stride_byte_offset, int swizzle_mode = 0) {
    // Shared memory descriptor layout per PTX ISA 9.7.16.4.1
    auto encode = [](uint32_t x) -> uint64_t { return (x & 0x3FFFF) >> 4; };

    uint64_t desc = 0;
    desc |= encode(smem_addr);                          // bits 0-13
    desc |= (encode(leading_byte_offset) << 16);       // bits 16-29
    desc |= (encode(stride_byte_offset) << 32);        // bits 32-45
    desc |= (1ULL << 46);                              // bits 46-48 = 0b001
    desc |= (0ULL << 49);                              // bits 49-51 = 0 (base offset)
    desc |= (0ULL << 52);                              // bit 52 = 0 (relative mode)
    desc |= (0xB0ULL << 53);                           // bits 53-60 fixed constant 0b1011_0000
    desc |= (((uint64_t)swizzle_mode) << 61);         // bits 61-63

    return desc;
}

__device__ __forceinline__ uint32_t make_idesc_mxf4nvf4(int M, int N, int K,
                                                          bool scale_type_ue4m3 = false,
                                                          int scale_factor_a_id = 0,
                                                          int scale_factor_b_id = 0) {
    // Instruction descriptor for .kind::mxf4nvf4 per PTX ISA Table 44
    uint32_t idesc = 0;
    idesc |= (0 << 0);                          // bits 0-1: reserved
    idesc |= (0 << 2);                          // bit 2: dense
    idesc |= (0 << 3);                          // bit 3: reserved
    idesc |= ((scale_factor_b_id & 0x3) << 4);  // bits 4-5: B scale ID
    idesc |= (0 << 6);                          // bit 6: reserved
    idesc |= (1 << 7);                          // bits 7-9: atype=E2M1
    idesc |= (1 << 10);                         // bits 10-11: btype=E2M1
    idesc |= (0 << 12);                         // bit 12: reserved
    idesc |= (0 << 13);                         // bit 13: no negate A
    idesc |= (0 << 14);                         // bit 14: no negate B
    idesc |= (0 << 15);                         // bit 15: no transpose A
    idesc |= (0 << 16);                         // bit 16: no transpose B
    idesc |= (((N >> 3) & 0x3F) << 17);         // bits 17-22: N>>3
    idesc |= ((scale_type_ue4m3 ? 0 : 1) << 23); // bit 23: scale type (0=UE4M3, 1=UE8M0)
    idesc |= (0 << 24);                         // bits 24-26: reserved
    idesc |= (((M >> 7) & 0x3) << 27);          // bits 27-28: M>>7
    idesc |= ((scale_factor_a_id & 0x3) << 29); // bits 29-30: A scale ID
    idesc |= (0 << 31);                         // bit 31: K=64 dense
    
    return idesc;
}

// Inline-PTX tcgen05 kernel with warp-cooperative staging/cp/st, single-lane MMA
__global__ void gemv_tcgen05_kernel(const int8_t* a,
                                    const int8_t* b,
                                    const int8_t* sfa,
                                    const int8_t* sfb,
                                    half* c,
                                    int M, int K, int L) {
    int tile_m = blockIdx.x;
    int tile_k = blockIdx.y;
    int tile_l = blockIdx.z;
    int m_base = tile_m * M_TILE;
    int k_base = tile_k * K_TILE;
    if (m_base >= M || k_base >= K || tile_l >= L) return;

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x & 31;

    // SMEM buffers
    __shared__ alignas(128) int8_t sm_a[4096];    // 128x32B (128x64 FP4)
    __shared__ alignas(128) int8_t sm_b[256];     // 256B buffer for B
    // sfa: Need 16B stride. 128 rows * 16B = 2048B.
    __shared__ alignas(128) int8_t sm_sfa[2048];  
// sfb: Need 16B stride. 4 rows * 16B = 64B. 512B is plenty.
__shared__ alignas(128) int8_t sm_sfb[512];   
__shared__ alignas(128) float sm_d[1024];     // 128x8 accum in fp32 (4096B)

    __shared__ uint32_t sm_taddr_a, sm_taddr_sfa, sm_taddr_sfb, sm_taddr_d;
    __shared__ uint64_t mbar_mma;

    // Strides per batch
    size_t stride_a = static_cast<size_t>(M) * (K / 2);
    size_t stride_sfa = static_cast<size_t>(M) * (K / 16);
    size_t stride_b = (K / 2);
    size_t stride_sfb = (K / 16);

    const int8_t* g_a_tile = a + tile_l * stride_a + (m_base * K + k_base) / 2;
    const int8_t* g_b_tile = b + tile_l * stride_b + k_base / 2;
    const int8_t* g_sfa_tile = sfa + tile_l * stride_sfa + m_base * (K / 16) + k_base / 16;
    const int8_t* g_sfb_tile = sfb + tile_l * stride_sfb + k_base / 16;

    // ====================================================================================================
    // 1. Memory Loading Pipeline (GMEM -> SMEM -> TMEM)
    // ====================================================================================================
    // Warp-coop GMEM->SMEM loads
    // A: 1024B per warp (4 warps * 1024 = 4096)
    for (int i = laneId * 4; i < 1024; i += 32 * 4) {
        int idx = warpId * 1024 + i;
        if (idx + 4 <= 4096)
            *reinterpret_cast<int*>(sm_a + idx) = *reinterpret_cast<const int*>(g_a_tile + idx);
    }
    
    // B: Input is 1xK (64 elems) = 32 bytes.
    if (warpId == 0) {
        for (int i = laneId * 4; i < 32; i += 32 * 4) {
            if (i + 4 <= 32)
                *reinterpret_cast<int*>(sm_b + i) = *reinterpret_cast<const int*>(g_b_tile + i);
        }
        for (int i = laneId * 4 + 32; i < 256; i += 32 * 4) {
             *reinterpret_cast<int*>(sm_b + i) = 0;
        }
    }
    
    // sfa: 128 rows. Each row is 4 bytes.
    // We must scatter to 16-byte stride for alignment requirements.
    // We use 4 warps to load.
    // Total 128 rows.
    for (int i = warpId * 32 + laneId; i < 128; i += 128) {
        // Load 4 bytes
        int val = *reinterpret_cast<const int*>(g_sfa_tile + i * 4);
        // Convert e4m3fnuz -> ue4m3 in-place (clear sign bit). Keeps value for non-negative scales.
        val &= 0x7F7F7F7F;
        // Store to sm_sfa with stride 16
        *reinterpret_cast<int*>(sm_sfa + i * 16) = val;
        // Zero pad the rest of the 16B stride? Not strictly necessary if we don't read it, 
        // but for safety/cleanliness we can. tcgen05.cp reads 32B, so overlap happens.
        // Overlap is fine.
    }
    
    // sfb: Input is 1x4 bytes (4 bytes total).
    // We need 4 rows. Stride 16.
    // Row 0: 8 copies of S0.
    // Row 1: 8 copies of S1.
    // Row 2: 8 copies of S2.
    // Row 3: 8 copies of S3.
    if (warpId == 0 && laneId == 0) {
        int val = *reinterpret_cast<const int*>(g_sfb_tile);
        // Convert e4m3fnuz -> ue4m3 for B scales.
        val &= 0x7F7F7F7F;
        for(int k=0; k<4; ++k) {
            int8_t s = (val >> (k*8)) & 0xFF;
            // Create 8 copies of s (64 bits)
            uint64_t s8 = 0x0101010101010101ULL * (uint8_t)s;
            *reinterpret_cast<uint64_t*>(sm_sfb + k * 16) = s8;
        }
    }

    __syncthreads(); // SMEM ready



    // ====================================================================================================
    // 2. Initialization & Allocation
    // ====================================================================================================
    // Descriptors
    uint32_t addr_a = __cvta_generic_to_shared(sm_a);
    uint32_t addr_b = __cvta_generic_to_shared(sm_b);
    uint32_t addr_sfa = __cvta_generic_to_shared(sm_sfa);
    uint32_t addr_sfb = __cvta_generic_to_shared(sm_sfb);
    
    // SMEM descriptors: leading and stride are per-row byte strides.
    uint64_t sdesc_a   = make_smem_desc(addr_a, 32, 32);
    uint64_t sdesc_b   = make_smem_desc(addr_b, 32, 32);   // single row, pitch 32B
    uint64_t sdesc_sfa = make_smem_desc(addr_sfa, 16, 16); // 128 rows, 16B pitch
    uint64_t sdesc_sfb = make_smem_desc(addr_sfb, 16, 16); // 4 rows, 16B pitch


    // Pass true for UE4M3 (e4m3fnuz input)
    uint32_t idesc = make_idesc_mxf4nvf4(M_TILE, N_TILE, K_TILE, true, 0, 0);

    // TMEM alloc
    if (warpId == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;\\n"
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%1], 128;\\n"
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%2], 32;\\n"
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%3], 32;\\n"
            :: "l"(&sm_taddr_a), "l"(&sm_taddr_d), "l"(&sm_taddr_sfa), "l"(&sm_taddr_sfb)
        );
        if (laneId == 0) {
            asm volatile("mbarrier.init.shared.b64 [%0], 1;\\n" :: "l"(&mbar_mma));
        }
    }
    __syncthreads();

    // Only warp 0 needs the TMEM handles; keep other warps out to avoid handle scope violations.
    uint32_t taddr_a = 0, taddr_sfa = 0, taddr_sfb = 0, taddr_d = 0;
    if (warpId == 0) {
        uint32_t sm_taddr_a_ptr = __cvta_generic_to_shared(&sm_taddr_a);
        uint32_t sm_taddr_sfa_ptr = __cvta_generic_to_shared(&sm_taddr_sfa);
        uint32_t sm_taddr_sfb_ptr = __cvta_generic_to_shared(&sm_taddr_sfb);
        uint32_t sm_taddr_d_ptr = __cvta_generic_to_shared(&sm_taddr_d);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_a)   : "r"(sm_taddr_a_ptr));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_sfa) : "r"(sm_taddr_sfa_ptr));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_sfb) : "r"(sm_taddr_sfb_ptr));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_d)   : "r"(sm_taddr_d_ptr));

    }
    __syncthreads(); // keep other warps aligned with warp 0 after handle loads

    // SMEM -> TMEM (warp 0 handles tc core cp)
    __syncthreads(); // ensure SMEM filled before cp
    if (warpId == 0) {
        // A: 4096B -> .128x256b
        asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;\\n" :: "r"(taddr_a), "l"(sdesc_a));
        // sfa: copy 2048B via .128x128b shape.
        asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;\\n" :: "r"(taddr_sfa), "l"(sdesc_sfa));
        // sfb: Use .4x256b. Stride 16.
        // tiny, leave on warp0
        asm volatile("tcgen05.cp.cta_group::1.4x256b   [%0], %1;\\n" :: "r"(taddr_sfb), "l"(sdesc_sfb));
    }



    // ====================================================================================================
    // 3. Matrix Multiply & Accumulate (MMA + Reduction)
    // ====================================================================================================
    // MMA single warp
    if (warpId == 0) {
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
            "[%0], [%1], %2, %3, [%4], [%5], 0;\\n"
            :: "r"(taddr_d), "r"(taddr_a), "l"(sdesc_b), "r"(idesc),
               "r"(taddr_sfa), "r"(taddr_sfb)
        );
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\\n" :: "l"(&mbar_mma));

    }

    // wait for MMA
    asm volatile(
        "{.reg .pred p; Lw%=: mbarrier.try_wait.parity.b64 p, [%0], 1; @!p bra Lw%=;}\\n"
        :: "l"(&mbar_mma));
    asm volatile("tcgen05.fence::after_thread_sync;\\n");



    // TMEM -> SMEM writeback (warp 0 only; TMEM is warp-scoped)
    // Use tcgen05.ld to registers, then st.shared
    if (warpId == 0) {
        uint32_t sm_d_ptr = __cvta_generic_to_shared(sm_d) + threadIdx.x * 4;
        asm volatile(
            ".reg .b32 r<32>;\\n"
            "tcgen05.ld.sync.aligned.16x256b.x8.b32 "
            "{r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,"
            "r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31}, [%1];\\n"
            "st.shared.b32 [%0+0], r0;\\n"
            "st.shared.b32 [%0+128], r1;\\n"
            "st.shared.b32 [%0+256], r2;\\n"
            "st.shared.b32 [%0+384], r3;\\n"
            "st.shared.b32 [%0+512], r4;\\n"
            "st.shared.b32 [%0+640], r5;\\n"
            "st.shared.b32 [%0+768], r6;\\n"
            "st.shared.b32 [%0+896], r7;\\n"
            "st.shared.b32 [%0+1024], r8;\\n"
            "st.shared.b32 [%0+1152], r9;\\n"
            "st.shared.b32 [%0+1280], r10;\\n"
            "st.shared.b32 [%0+1408], r11;\\n"
            "st.shared.b32 [%0+1536], r12;\\n"
            "st.shared.b32 [%0+1664], r13;\\n"
            "st.shared.b32 [%0+1792], r14;\\n"
            "st.shared.b32 [%0+1920], r15;\\n"
            "st.shared.b32 [%0+2048], r16;\\n"
            "st.shared.b32 [%0+2176], r17;\\n"
            "st.shared.b32 [%0+2304], r18;\\n"
            "st.shared.b32 [%0+2432], r19;\\n"
            "st.shared.b32 [%0+2560], r20;\\n"
            "st.shared.b32 [%0+2688], r21;\\n"
            "st.shared.b32 [%0+2816], r22;\\n"
            "st.shared.b32 [%0+2944], r23;\\n"
            "st.shared.b32 [%0+3072], r24;\\n"
            "st.shared.b32 [%0+3200], r25;\\n"
            "st.shared.b32 [%0+3328], r26;\\n"
            "st.shared.b32 [%0+3456], r27;\\n"
            "st.shared.b32 [%0+3584], r28;\\n"
            "st.shared.b32 [%0+3712], r29;\\n"
            "st.shared.b32 [%0+3840], r30;\\n"
            "st.shared.b32 [%0+3968], r31;\\n"
            :: "r"(sm_d_ptr), "r"(taddr_d)
        );
    }
    __syncthreads();

#if DEBUG_GEMV_PIPELINE
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        float d0 = sm_d[0];
        float d1 = sm_d[N_TILE];
        printf("DEBUG tmemory out tile(%d,%d,%d): D0=%f D1=%f\\n",
               tile_m, tile_k, tile_l, d0, d1);
        printf("DEBUG sm_d slice: 0:%f 1:%f 2:%f 3:%f 4:%f 5:%f 6:%f 7:%f | 32:%f 33:%f 34:%f 35:%f 36:%f 37:%f 38:%f 39:%f\\n",
               sm_d[0], sm_d[1], sm_d[2], sm_d[3], sm_d[4], sm_d[5], sm_d[6], sm_d[7],
               sm_d[32], sm_d[33], sm_d[34], sm_d[35], sm_d[36], sm_d[37], sm_d[38], sm_d[39]);
        for (int preview_row = 0; preview_row < 8; ++preview_row) {
            int stripe = preview_row >> 5;
            int lane = preview_row & 31;
            int sm_idx = lane + stripe * N_TILE * 32;
            printf("DEBUG accum preview row=%d stripe=%d lane=%d sm_idx=%d val=%f\\n",
                   preview_row, stripe, lane, sm_idx, sm_d[sm_idx]);
        }
    }
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warpId == 0 && laneId < 4) {
        for (int stripe = 0; stripe < 4; ++stripe) {
            int sm_idx = laneId + stripe * N_TILE * 32;
            printf("DEBUG lane stripe lane=%d stripe=%d sm_idx=%d val=%f\\n",
                   laneId, stripe, sm_idx, sm_d[sm_idx]);
        }
    }
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warpId == 0 && laneId == 0) {
        constexpr int row_stride_fp4 = K_TILE / 2;
        for (int test_row = 0; test_row < 4; ++test_row) {
            float ref = 0.f;
            for (int kk = 0; kk < K_TILE; ++kk) {
                float aval = smem_decode_fp4(sm_a, row_stride_fp4, test_row, kk);
                float bval = smem_decode_fp4(sm_b, row_stride_fp4, 0, kk);
                int scale_idx = kk >> 4;
                float sfa = decode_fp8_byte(sm_sfa[test_row * 16 + scale_idx]);
                float sfb = decode_fp8_byte(sm_sfb[scale_idx * 16]);
                ref += (aval * sfa) * (bval * sfb);
            }
            int stripe = test_row >> 5;
            int lane = test_row & 31;
            int sm_idx = lane + stripe * N_TILE * 32;
            float mma_val = sm_d[sm_idx];
            printf("DEBUG ref row=%d ref=%f mma=%f diff=%f sm_idx=%d\\n",
                   test_row, ref, mma_val, ref - mma_val, sm_idx);
        }
        for (int sf_row = 0; sf_row < 4; ++sf_row) {
            printf("DEBUG sfa row=%d bytes=%d %d %d %d\\n",
                   sf_row,
                   static_cast<int>(sm_sfa[sf_row * 16 + 0]),
                   static_cast<int>(sm_sfa[sf_row * 16 + 1]),
                   static_cast<int>(sm_sfa[sf_row * 16 + 2]),
                   static_cast<int>(sm_sfa[sf_row * 16 + 3]));
        }
        for (int sf = 0; sf < 4; ++sf) {
            printf("DEBUG sfb idx=%d byte=%d\\n",
                   sf,
                   static_cast<int>(sm_sfb[sf * 16]));
        }
        for (int kk = 0; kk < 8; ++kk) {
            float bval = smem_decode_fp4(sm_b, row_stride_fp4, 0, kk);
            printf("DEBUG b row0 k=%d val=%f\\n", kk, bval);
        }
    }
#endif

    // Accumulate partial: tmemory D layout is 16 x 64 (rows x cols) for M=128,N=8
    // Row groups of 16 are interleaved across columns blocks of 8.
    // For global row i: group = i>>4 selects which 16-row block, row16 = i&15.
    // Column 0 for that row is at sm_d index = row16*64 + group*N_TILE.
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

    // Use an otherwise idle warp (warp3) to handle TMEM dealloc/relinquish so warp0 can exit earlier.
    if (warpId == 3) {
        // Reload TMEM handles from shared for this warp.
        uint32_t sm_taddr_a_ptr = __cvta_generic_to_shared(&sm_taddr_a);
        uint32_t sm_taddr_sfa_ptr = __cvta_generic_to_shared(&sm_taddr_sfa);
        uint32_t sm_taddr_sfb_ptr = __cvta_generic_to_shared(&sm_taddr_sfb);
        uint32_t sm_taddr_d_ptr = __cvta_generic_to_shared(&sm_taddr_d);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_a)   : "r"(sm_taddr_a_ptr));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_sfa) : "r"(sm_taddr_sfa_ptr));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_sfb) : "r"(sm_taddr_sfb_ptr));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_d)   : "r"(sm_taddr_d_ptr));

        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\\n" :: "r"(taddr_a));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\\n"  :: "r"(taddr_sfa));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\\n"  :: "r"(taddr_sfb));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\\n" :: "r"(taddr_d));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\\n");
    }
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    int m_tiles = (M + M_TILE - 1) / M_TILE;
    int k_tiles = (K + K_TILE - 1) / K_TILE;
    dim3 grid(m_tiles, k_tiles, L);
    dim3 block(128, 1, 1);  // 4 warps; warp0 owns tcgen05

    gemv_tcgen05_kernel<<<grid, block>>>(reinterpret_cast<int8_t*>(a.data_ptr()),
                                         reinterpret_cast<int8_t*>(b.data_ptr()),
                                         reinterpret_cast<int8_t*>(sfa.data_ptr()),
                                         reinterpret_cast<int8_t*>(sfb.data_ptr()),
                                         reinterpret_cast<half*>(c.data_ptr()),
                                         M, K, L);

    return c;
}
"""

module = load_inline(
    name="gemv_tcgen05",
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


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device

    # Ensure output starts at zero to avoid race with multi-tile accumulation.
    c.zero_()

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
