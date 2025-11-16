import torch
from typing import TypedDict, TypeVar

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)

class TestSpec(TypedDict):
    m: int
    k: int
    l: int
    seed: int

sf_vec_size = 16

def ceil_div(a, b):
    return (a + b - 1) // b

def generate_input(m: int, k: int, l: int, seed: int):
    torch.manual_seed(seed)
    n = 1
    n_padded_128 = 128
    
    a_ref = torch.randint(
        0, 2, (l, m, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    b_ref = torch.randint(
        0, 2, (l, n_padded_128, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b_ref = b_ref.view(torch.float4_e2m1fn_x2)
    
    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(1, 2, 0)
    
    def create_scale_factor_tensors(l, mn, sf_k):
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)
        ref_f8_random_int = torch.randint(1, 3, ref_shape, dtype=torch.int8, device='cuda')
        ref_f8_torch_tensor = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
        ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)
        
        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )
        
        mma_permute_order = (3, 4, 1, 5, 2, 0)
        rand_int_tensor = torch.randint(0, 2, mma_shape, dtype=torch.int8, device='cuda')
        reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(*mma_permute_order)
        
        i_idx = torch.arange(mn, device='cuda')
        j_idx = torch.arange(sf_k, device='cuda')
        b_idx = torch.arange(l, device='cuda')
        
        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')
        
        mm = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4 = (i_grid % 128) // atom_m[0]
        kk = j_grid // atom_k
        kk4 = j_grid % atom_k
        
        reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]
        
        return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor
    
    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb_ref_cpu, sfb_permuted = create_scale_factor_tensors(l, n_padded_128, sf_k)
    
    return (a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, sfa_permuted, sfb_permuted, c_ref)

def get_test_data():
    m, k, l, seed = 512, 512, 1, 42
    return generate_input(m=m, k=k, l=l, seed=seed)

