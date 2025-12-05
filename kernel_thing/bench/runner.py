#!/usr/bin/env python3
"""
Simple Kernel Test/Bench Runner

No frills. Compile, run, check, time.

Usage:
    python runner.py test submission.py                    # correctness check
    python runner.py bench submission.py                   # benchmark
    python runner.py both submission.py                    # both
    python runner.py test submission.py --task u8gemv      # specify task
"""

import sys
import os
import time
import math
import argparse
import importlib.util
from pathlib import Path
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass

import torch

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from task import TaskSpec, U8GemvSpec, TESTS, BENCHMARKS


@dataclass
class RunResult:
    spec: TaskSpec
    passed: bool
    message: str
    time_us: Optional[float] = None


@dataclass
class BenchStats:
    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    runs: int
    gflops: float


def load_submission(path: str) -> Callable:
    """Load custom_kernel from submission file"""
    spec = importlib.util.spec_from_file_location("submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "custom_kernel"):
        raise ValueError(f"No custom_kernel function in {path}")

    return module.custom_kernel


def run_test(kernel: Callable, spec: TaskSpec) -> RunResult:
    """Run single test case"""
    try:
        inputs = spec.generate_input()
        A, B, C = inputs[:3]  # First two are inputs, third is output buffer
        C_ref = spec.reference(A, B)

        torch.cuda.synchronize()
        C_got = kernel(inputs)
        torch.cuda.synchronize()

        passed, msg = spec.check(C_got, C_ref)
        return RunResult(spec=spec, passed=passed, message=msg)

    except Exception as e:
        return RunResult(spec=spec, passed=False, message=str(e))


def run_bench(kernel: Callable, spec: TaskSpec,
              warmup: int = 5, runs: int = 50) -> Tuple[bool, BenchStats]:
    """Run benchmark for single spec"""
    import dataclasses

    # First check correctness
    inputs = spec.generate_input()
    A, B, C = inputs[:3]
    C_ref = spec.reference(A, B)

    torch.cuda.synchronize()
    C_got = kernel(inputs)
    torch.cuda.synchronize()

    passed, msg = spec.check(C_got, C_ref)
    if not passed:
        return False, msg

    # Warmup
    for _ in range(warmup):
        inputs = spec.generate_input()
        kernel(inputs)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for i in range(runs):
        # Create new spec with different seed for each run
        run_spec = dataclasses.replace(spec, seed=spec.seed + i)
        inputs = run_spec.generate_input()

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        kernel(inputs)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    # Stats
    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
    std = math.sqrt(variance)

    # GFLOPS from task spec
    flops = spec.flops()
    gflops = flops / (mean * 1e-6) / 1e9  # us -> s, then /1e9 for G

    stats = BenchStats(
        mean_us=mean,
        std_us=std,
        min_us=min(times),
        max_us=max(times),
        runs=len(times),
        gflops=gflops
    )

    return True, stats


def test_all(kernel: Callable, cases: List[TaskSpec] = None) -> bool:
    """Run all test cases"""
    cases = cases or TESTS
    print(f"Running {len(cases)} test cases...")
    print("-" * 60)

    all_passed = True
    for spec in cases:
        result = run_test(kernel, spec)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status}  {spec}  {result.message}")
        if not result.passed:
            all_passed = False

    print("-" * 60)
    print(f"{'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def bench_all(kernel: Callable, cases: List[TaskSpec] = None) -> bool:
    """Run all benchmark cases"""
    cases = cases or BENCHMARKS
    print(f"Running {len(cases)} benchmark cases...")
    print("-" * 60)

    all_passed = True
    for spec in cases:
        ok, result = run_bench(kernel, spec)
        if not ok:
            print(f"✗ FAIL  {spec}  {result}")
            all_passed = False
        else:
            print(f"✓ {spec}")
            print(f"    {result.mean_us:.2f} ± {result.std_us:.2f} us  "
                  f"(min={result.min_us:.2f}, max={result.max_us:.2f})  "
                  f"{result.gflops:.1f} GFLOPS")

    print("-" * 60)
    return all_passed


def main():
    if len(sys.argv) < 3:
        print("Usage: python runner.py <test|bench|both> <submission.py>")
        sys.exit(1)

    mode = sys.argv[1]
    submission_path = sys.argv[2]

    if not Path(submission_path).exists():
        print(f"File not found: {submission_path}")
        sys.exit(1)

    print(f"Loading {submission_path}...")
    try:
        kernel = load_submission(submission_path)
    except Exception as e:
        print(f"Failed to load submission: {e}")
        sys.exit(1)

    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"SM version: {torch.cuda.get_device_capability()}")
    print()

    if mode == "test":
        ok = test_all(kernel)
        sys.exit(0 if ok else 1)

    elif mode == "bench":
        ok = bench_all(kernel)
        sys.exit(0 if ok else 1)

    elif mode == "both":
        print("=== TESTING ===")
        test_ok = test_all(kernel)
        print()
        print("=== BENCHMARKING ===")
        bench_ok = bench_all(kernel)
        sys.exit(0 if test_ok and bench_ok else 1)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
