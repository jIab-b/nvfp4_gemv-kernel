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
    desc |= (0ULL << 53);                              // bits 53-60 = 0
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
    __shared__ alignas(128) int8_t sm_sfa[1024];  // 128x4B (512B used)
    __shared__ alignas(128) int8_t sm_sfb[512];   // 4x8B (32B used)
    __shared__ alignas(128) half sm_d[1024];      // 128x8 fp16

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

    // Warp-coop GMEM->SMEM loads
    // A: 1024B per warp (4 warps * 1024 = 4096)
    for (int i = laneId * 4; i < 1024; i += 32 * 4) {
        int idx = warpId * 1024 + i;
        if (idx + 4 <= 4096)
            *reinterpret_cast<int*>(sm_a + idx) = *reinterpret_cast<const int*>(g_a_tile + idx);
    }
    
    // B: Input is 1xK (64 elems) = 32 bytes.
    // Load 32 bytes valid, zero the rest of sm_b (256B) to avoid OOB reads
    if (warpId == 0) {
        for (int i = laneId * 4; i < 32; i += 32 * 4) {
            if (i + 4 <= 32)
                *reinterpret_cast<int*>(sm_b + i) = *reinterpret_cast<const int*>(g_b_tile + i);
        }
        for (int i = laneId * 4 + 32; i < 256; i += 32 * 4) {
             *reinterpret_cast<int*>(sm_b + i) = 0;
        }
    }
    
    // sfa: 512B total (128 rows * 4 bytes).
    for (int i = laneId * 4; i < 128; i += 32 * 4) {
        int idx = warpId * 128 + i;
        if (idx + 4 <= 512)
            *reinterpret_cast<int*>(sm_sfa + idx) = *reinterpret_cast<const int*>(g_sfa_tile + idx);
    }
    // Zero padding in sm_sfa (1024B)
    if (warpId == 0) {
        for (int i = laneId * 4 + 512; i < 1024; i += 32 * 4) {
            *reinterpret_cast<int*>(sm_sfa + i) = 0;
        }
    }
    
    // sfb: Input is 1x4 bytes (4 bytes). 
    // MMA expects 4xN (4x8 = 32 bytes).
    // Load 4 bytes valid, zero the rest of sm_sfb
    if (warpId == 0) {
        for (int i = laneId * 4; i < 4; i += 32 * 4) {
            if (i + 4 <= 4)
                *reinterpret_cast<int*>(sm_sfb + i) = *reinterpret_cast<const int*>(g_sfb_tile + i);
        }
        for (int i = laneId * 4 + 4; i < 512; i += 32 * 4) {
            *reinterpret_cast<int*>(sm_sfb + i) = 0;
        }
    }

    __syncthreads(); // SMEM ready

    // Descriptors
    uint32_t addr_a = __cvta_generic_to_shared(sm_a);
    uint32_t addr_b = __cvta_generic_to_shared(sm_b);
    uint32_t addr_sfa = __cvta_generic_to_shared(sm_sfa);
    uint32_t addr_sfb = __cvta_generic_to_shared(sm_sfb);
    
    uint64_t sdesc_a = make_smem_desc(addr_a, 32, 4096);
    uint64_t sdesc_b = make_smem_desc(addr_b, 4, 256);
    uint64_t sdesc_sfa = make_smem_desc(addr_sfa, 4, 512); // 4 bytes leading dim
    uint64_t sdesc_sfb = make_smem_desc(addr_sfb, 8, 512); // 8 bytes leading dim for N=8 (4x8 matrix)

    // Pass true for UE4M3 (e4m3fnuz input)
    uint32_t idesc = make_idesc_mxf4nvf4(M_TILE, N_TILE, K_TILE, true, 0, 0);

    // TMEM alloc
    if (warpId == 0 && laneId == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;\\n"
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%1], 32;\\n"
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%2], 32;\\n"
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%3], 128;\\n"
            :: "l"(&sm_taddr_a), "l"(&sm_taddr_sfa), "l"(&sm_taddr_sfb), "l"(&sm_taddr_d)
        );
        asm volatile("mbarrier.init.shared.b64 [%0], 1;\\n" :: "l"(&mbar_mma));
    }
    __syncthreads();

    uint32_t taddr_a, taddr_sfa, taddr_sfb, taddr_d;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_a) : "l"(&sm_taddr_a));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_sfa) : "l"(&sm_taddr_sfa));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_sfb) : "l"(&sm_taddr_sfb));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr_d) : "l"(&sm_taddr_d));

    // SMEM -> TMEM (all warps issue their stripe)
    // A: 4096B -> .128x256b (128 rows * 32B)
    asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;\\n" :: "r"(taddr_a), "l"(sdesc_a));
    // sfa: 512B -> .16x256b (16 rows * 32B = 512B)
    asm volatile("tcgen05.cp.cta_group::1.16x256b  [%0], %1;\\n" :: "r"(taddr_sfa), "l"(sdesc_sfa));
    // sfb: 32B needed. .4x128b copies 64B. Safe since we zeroed.
    asm volatile("tcgen05.cp.cta_group::1.4x128b  [%0], %1;\\n" :: "r"(taddr_sfb), "l"(sdesc_sfb));

    // MMA single lane
    if (warpId == 0 && laneId == 0) {
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
        "{.reg .pred p; Lw%=: mbarrier.try_wait.parity.b64 p, [%0], 0; @!p bra Lw%=;}\\n"
        :: "l"(&mbar_mma));
    asm volatile("tcgen05.fence::after_thread_sync;\\n");

    // TMEM -> SMEM writeback (all warps)
    asm volatile("tcgen05.st.sync.aligned.16x256b.x8 [%0], %1;\\n" :: "l"(sm_d), "r"(taddr_d));
    __syncthreads();

    // SMEM -> GMEM stores
    // First K tile zeroes output to avoid stale data.
    if (tile_k == 0) {
        for (int i = warpId * 32 + laneId; i < M_TILE; i += 128) {
            int g_row = m_base + i;
            if (g_row < M) {
                c[g_row * L + tile_l] = __float2half(0.0f);
            }
        }
    }
    __syncthreads();

    // Accumulate partial
    for (int i = warpId * 32 + laneId; i < M_TILE; i += 128) {
        int g_row = m_base + i;
        if (g_row < M) {
            half val = sm_d[i * N_TILE]; // column 0
            atomicAdd(reinterpret_cast<half*>(&c[g_row * L + tile_l]), val);
        }
    }

    if (warpId == 0 && laneId == 0) {
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
