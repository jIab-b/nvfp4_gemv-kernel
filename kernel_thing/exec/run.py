"""
Kernel Execution - Load and run compiled CUDA kernels
"""

import torch
from torch.utils.cpp_extension import load_inline
from typing import Tuple, Any, Optional
from dataclasses import dataclass
import re
import tempfile
import os


@dataclass
class RunResult:
    """Result of running a kernel"""
    success: bool
    output: Optional[torch.Tensor]
    error_msg: Optional[str]


def extract_kernel_name(source: str) -> Optional[str]:
    """Extract __global__ kernel function name from source"""
    pattern = r'__global__\s+void\s+(\w+)\s*\('
    match = re.search(pattern, source)
    return match.group(1) if match else None


def extract_kernel_params(source: str) -> list:
    """Extract kernel parameter types"""
    pattern = r'__global__\s+void\s+\w+\s*\(([^)]*)\)'
    match = re.search(pattern, source)
    if not match:
        return []

    params_str = match.group(1)
    params = []
    for param in params_str.split(','):
        param = param.strip()
        if param:
            # Get type (everything except last word)
            parts = param.rsplit(None, 1)
            if parts:
                params.append(parts[0].strip())
    return params


def make_wrapper(source: str, kernel_name: str, dtype: str = "int32") -> str:
    """
    Generate torch extension wrapper for a kernel.

    Creates a run_kernel() function that launches the CUDA kernel.
    """
    # Map dtype to C++ type
    dtype_map = {
        "fp16": "at::Half",
        "fp32": "float",
        "int32": "int32_t",
        "uint8": "uint8_t",
    }
    cpp_type = dtype_map.get(dtype, "int32_t")

    wrapper = f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

torch::Tensor run_kernel(torch::Tensor A, torch::Tensor B, torch::Tensor C) {{
    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);

    // Launch config - one block per row
    dim3 grid(M, batch, 1);
    dim3 block(256, 1, 1);

    auto A_ptr = A.data_ptr<{cpp_type}>();
    auto B_ptr = B.data_ptr<{cpp_type}>();
    auto C_ptr = C.data_ptr<{cpp_type}>();

    {kernel_name}<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, K, batch);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {{
        throw std::runtime_error(cudaGetErrorString(err));
    }}

    return C;
}}
"""
    return source + wrapper


def load_and_run(
    source: str,
    inputs: Tuple[torch.Tensor, ...],
    sm: int = 75,
    dtype: str = "int32"
) -> RunResult:
    """
    Compile source with torch extension and run kernel.

    This uses JIT compilation via load_inline, which has its own caching.
    Use for quick testing; for training, use cubin-based execution.

    Args:
        source: CUDA source code with __global__ kernel
        inputs: Tuple of (A, B, C) tensors
        sm: Target SM version
        dtype: Data type string

    Returns:
        RunResult with output tensor or error
    """
    try:
        kernel_name = extract_kernel_name(source)
        if kernel_name is None:
            return RunResult(
                success=False,
                output=None,
                error_msg="No __global__ kernel found in source"
            )

        wrapped = make_wrapper(source, kernel_name, dtype)

        # Unique name based on source hash
        import hashlib
        name = f"kernel_{hashlib.md5(source.encode()).hexdigest()[:8]}"

        module = load_inline(
            name=name,
            cpp_sources=[""],
            cuda_sources=[wrapped],
            functions=["run_kernel"],
            verbose=False,
            extra_cuda_cflags=["-O3", f"-arch=sm_{sm}"]
        )

        A, B, C = inputs[0], inputs[1], inputs[2]

        torch.cuda.synchronize()
        output = module.run_kernel(A, B, C)
        torch.cuda.synchronize()

        return RunResult(success=True, output=output, error_msg=None)

    except Exception as e:
        return RunResult(success=False, output=None, error_msg=str(e))


def run_kernel(
    cubin_path: str,
    inputs: Tuple[torch.Tensor, ...],
    kernel_name: str,
    grid: Tuple[int, int, int],
    block: Tuple[int, int, int]
) -> RunResult:
    """
    Run kernel from pre-compiled cubin.

    Uses CUDA driver API for direct cubin loading.
    Faster than JIT for repeated runs.

    Args:
        cubin_path: Path to .cubin file
        inputs: Input tensors
        kernel_name: Name of kernel function
        grid: Grid dimensions
        block: Block dimensions

    Returns:
        RunResult with output tensor or error
    """
    try:
        import cupy as cp
        from cupy.cuda import Module

        # Load cubin
        with open(cubin_path, 'rb') as f:
            cubin = f.read()

        module = Module()
        module.load(cubin)
        kernel = module.get_function(kernel_name)

        A, B, C = inputs[0], inputs[1], inputs[2]
        M, K = A.shape[1], A.shape[2]
        batch = A.shape[0]

        # Launch kernel
        kernel(
            grid, block,
            (A.data_ptr(), B.data_ptr(), C.data_ptr(), M, K, batch)
        )

        cp.cuda.runtime.deviceSynchronize()

        return RunResult(success=True, output=C, error_msg=None)

    except ImportError:
        # Fallback: cupy not available, use load_inline approach
        return RunResult(
            success=False,
            output=None,
            error_msg="cupy not installed, cannot load cubin directly"
        )
    except Exception as e:
        return RunResult(success=False, output=None, error_msg=str(e))
