"""
Benchmarking - Time kernel execution
"""

import torch
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from task import TaskSpec


@dataclass
class BenchResult:
    """Benchmark timing results"""
    mean_us: float
    min_us: float
    max_us: float
    std_us: float
    runs: int
    gflops: Optional[float] = None


def benchmark(
    kernel_fn: Callable,
    inputs: Tuple[torch.Tensor, ...],
    warmup: int = 5,
    runs: int = 20,
    flops: Optional[int] = None
) -> BenchResult:
    """
    Benchmark a kernel function.

    Args:
        kernel_fn: Callable that takes inputs and runs kernel
        inputs: Input tensors
        warmup: Number of warmup iterations
        runs: Number of timed runs
        flops: Total FLOPs for GFLOPS calculation

    Returns:
        BenchResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        kernel_fn(inputs)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        kernel_fn(inputs)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / max(1, len(times) - 1)
    std = variance ** 0.5

    gflops = None
    if flops is not None:
        gflops = flops / (mean * 1e-6) / 1e9

    return BenchResult(
        mean_us=mean,
        min_us=min(times),
        max_us=max(times),
        std_us=std,
        runs=len(times),
        gflops=gflops
    )


def benchmark_reference(
    spec: TaskSpec,
    warmup: int = 5,
    runs: int = 20
) -> BenchResult:
    """
    Benchmark the reference implementation for a task.

    Args:
        spec: Task specification
        warmup: Warmup iterations
        runs: Timed runs

    Returns:
        BenchResult for reference implementation
    """
    inputs = spec.generate_input()
    A, B = inputs[0], inputs[1]

    def ref_fn(_):
        return spec.reference(A, B)

    return benchmark(ref_fn, inputs, warmup, runs, spec.flops())


def benchmark_kernel_source(
    source: str,
    spec: TaskSpec,
    sm: int = 75,
    warmup: int = 5,
    runs: int = 20
) -> Tuple[Optional[BenchResult], Optional[str]]:
    """
    Benchmark a kernel from source code.

    Args:
        source: CUDA source code
        spec: Task specification
        sm: Target SM version
        warmup: Warmup iterations
        runs: Timed runs

    Returns:
        (BenchResult, None) on success, (None, error_msg) on failure
    """
    from .run import load_and_run, extract_kernel_name, make_wrapper
    from torch.utils.cpp_extension import load_inline
    import hashlib

    try:
        kernel_name = extract_kernel_name(source)
        if kernel_name is None:
            return None, "No kernel found"

        # Infer dtype from spec
        inputs = spec.generate_input()
        C = inputs[2]
        if C.dtype == torch.int32:
            dtype = "int32"
        elif C.dtype == torch.float16:
            dtype = "fp16"
        elif C.dtype == torch.float32:
            dtype = "fp32"
        else:
            dtype = "int32"

        wrapped = make_wrapper(source, kernel_name, dtype)
        name = f"bench_{hashlib.md5(source.encode()).hexdigest()[:8]}"

        module = load_inline(
            name=name,
            cpp_sources=[""],
            cuda_sources=[wrapped],
            functions=["run_kernel"],
            verbose=False,
            extra_cuda_cflags=["-O3", f"-arch=sm_{sm}"]
        )

        def kernel_fn(inp):
            return module.run_kernel(inp[0], inp[1], inp[2])

        result = benchmark(kernel_fn, inputs, warmup, runs, spec.flops())
        return result, None

    except Exception as e:
        return None, str(e)
