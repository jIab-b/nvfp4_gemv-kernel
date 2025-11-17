"""
Modal sync utility: Uploads submission_ref.py and dependencies to Modal workspace volume.
Useful for setting up remote profiling with NCU.

Usage:
    modal run submission_modal_debug.py::sync_to_workspace
"""

import os
import sys

try:
    import modal
except ImportError:
    print("Error: modal not installed. Install with: pip install modal")
    sys.exit(1)

app = modal.App("nvfp4-sync-workspace")

workspace_volume = modal.Volume.from_name("workspace", create_if_missing=True)

_TASK_PY_CONTENT = '''import torch
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
'''

_SUBMISSION_REF_PY_CONTENT = '''import torch
from task import input_t, output_t

sf_vec_size = 16

def ceil_div(a, b):
    return (a + b - 1) // b

def to_blocked(input_matrix):
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()

def custom_kernel(data: input_t) -> output_t:
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    _, _, l = c_ref.shape
    for l_idx in range(l):
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        res = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b.cuda(),
            bias=None,
            out_dtype=torch.float16,
        )
        c_ref[:, 0, l_idx] = res[:, 0]
    return c_ref
'''

_RUN_KERNEL_PY = '''import sys
sys.path.insert(0, "/workspace/files")
from task import get_test_data
from submission_ref import custom_kernel
print("Loading test data...")
test_data = get_test_data()
print("Running kernel...")
result = custom_kernel(test_data)
print("✓ Kernel completed successfully!")
'''

image_pytorch = modal.Image.from_registry(
    "pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel",
)

@app.function(
    gpu="B200",
    volumes={"/workspace": workspace_volume},
    image=image_pytorch,
)
def shell_with_gpu():
    import os
    os.system("bash")

@app.function(
    volumes={"/workspace": workspace_volume},
)
def sync_to_workspace():
    from pathlib import Path
    
    files_dir = Path("/workspace/files")
    files_dir.mkdir(parents=True, exist_ok=True)
    
    task_file = files_dir / "task.py"
    with open(task_file, "w") as f:
        f.write(_TASK_PY_CONTENT)
    print(f"✓ Synced task.py to /workspace/files/task.py")
    
    ref_file = files_dir / "submission_ref.py"
    with open(ref_file, "w") as f:
        f.write(_SUBMISSION_REF_PY_CONTENT)
    print(f"✓ Synced submission_ref.py to /workspace/files/submission_ref.py")
    
    run_script = files_dir / "run_kernel.py"
    with open(run_script, "w") as f:
        f.write(_RUN_KERNEL_PY)
    print(f"✓ Synced run_kernel.py to /workspace/files/run_kernel.py")
    
    workspace_volume.commit()
    
    print()
    print("=" * 80)
    print("✓ All files synced to Modal workspace volume!")
    print("=" * 80)
    print()
    print("Location: /workspace/files/")
    print("Files:")
    print("  - task.py (test data generation)")
    print("  - submission_ref.py (CUTLASS kernel with torch._scaled_mm)")
    print("  - run_kernel.py (simple runner script)")
    print()
    print("To profile with NCU on a GPU run:")
    print("  modal run -q submission_modal_debug.py::profile_with_ncu")
    print()
    print("Or manually in a GPU container:")
    print("  ncu --set full python /workspace/files/run_kernel.py")
    print()
