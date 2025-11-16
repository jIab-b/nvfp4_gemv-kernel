"""
PyTorch torch.compile version with PTX debugging enabled.

This version wraps the custom_kernel with torch.compile to capture:
1. PTX intermediate code via TRITON_KERNEL_DUMP=1
2. CUTLASS-invoking code via TORCH_COMPILE_DEBUG=1

To use this script with PTX debugging:

    # Set environment variables before running
    export TORCH_COMPILE_DEBUG=1
    export TRITON_KERNEL_DUMP=1
    export TRITON_DUMP_DIR=/tmp/triton_dumps
    export CUDA_CACHE_DISABLE=1

    python submission_torch_compile.py

    # After running, you can inspect:
    # 1. /tmp/torchinductor/ - Contains generated code and IR
    # 2. /tmp/triton_dumps/ - Contains PTX files from Triton kernels

To extract PTX from compiled binaries (if using extensions):
    cuobjdump -ptx your_extension.so

For NCU profiling with line information:
    # When running with ncu, add -lineinfo flag:
    ncu --set full python submission_torch_compile.py
"""

import os

# ============================================================================
# CRITICAL: Set environment variables at MODULE IMPORT TIME
# This ensures debugging is enabled even when imported by other scripts
# ============================================================================
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TRITON_KERNEL_DUMP'] = '1'
os.environ['TRITON_DUMP_DIR'] = '/tmp/triton_dumps'
os.environ['CUDA_CACHE_DISABLE'] = '1'
os.environ['TORCH_LOGS'] = 'dynamo,aot,inductor'

# Create the dump directory if it doesn't exist
import pathlib
pathlib.Path('/tmp/triton_dumps').mkdir(parents=True, exist_ok=True)
pathlib.Path('/tmp/torchinductor').mkdir(parents=True, exist_ok=True)

import torch
from task import input_t, output_t


# Kernel configuration parameters
sf_vec_size = 16


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


# Helper function to convert scale factor tensor to blocked format
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape

    # Please ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def custom_kernel_impl(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMV.
    Core logic without torch.compile wrapping.
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data

    # Get dimensions from MxNxL layout
    _, _, l = c_ref.shape

    # Call torch._scaled_mm to compute the GEMV result
    for l_idx in range(l):
        # Convert the scale factor tensor to blocked format
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        # (m, k) @ (n, k).T -> (m, n)
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


# Wrap with torch.compile for CUTLASS/Inductor kernel generation and debugging
# The backend='inductor' specifically targets the Inductor backend which may use CUTLASS
custom_kernel_compiled = torch.compile(
    custom_kernel_impl,
    backend='inductor',
    mode='reduce-overhead',  # Can also use 'default' or 'max-autotune'
)


def custom_kernel(data: input_t) -> output_t:
    """
    PyTorch reference implementation with torch.compile wrapper.

    This version compiles the kernel code, triggering:
    - Inductor to potentially generate CUTLASS kernels for torch._scaled_mm ops
    - PTX generation that can be inspected in debug directories
    - SASS instruction generation on the GPU
    """
    return custom_kernel_compiled(data)


if __name__ == "__main__":
    print("=" * 80)
    print("Debug environment variables already set at module import:")
    print("=" * 80)
    print(f"  TORCH_COMPILE_DEBUG={os.environ.get('TORCH_COMPILE_DEBUG')}")
    print(f"  TRITON_KERNEL_DUMP={os.environ.get('TRITON_KERNEL_DUMP')}")
    print(f"  TRITON_DUMP_DIR={os.environ.get('TRITON_DUMP_DIR')}")
    print(f"  CUDA_CACHE_DISABLE={os.environ.get('CUDA_CACHE_DISABLE')}")
    print(f"  TORCH_LOGS={os.environ.get('TORCH_LOGS')}")
    print()

    # Example usage (if task module is available)
    try:
        from task import get_test_data
        print("Executing kernel with torch.compile debugging enabled...")
        print()
        test_data = get_test_data()
        result = custom_kernel(test_data)
        print()
        print("✓ Kernel execution completed successfully!")
        print()
        print("=" * 80)
        print("Debug outputs saved to:")
        print("=" * 80)
        print("  - /tmp/torchinductor/ - Generated code and IR files")
        print("  - /tmp/triton_dumps/ - PTX files from Triton kernels")
        print()
        print("To inspect PTX files:")
        print("  ls -la /tmp/triton_dumps/")
        print("  cat /tmp/triton_dumps/*.ptx | head -100")
    except ImportError as e:
        print(f"Note: Could not import task module ({e})")
        print("Import this module and use custom_kernel() function directly.")
        print()
        print("After running, inspect:")
        print("  - /tmp/torchinductor/ for generated code and IR")
        print("  - /tmp/triton_dumps/ for PTX files from Triton")
