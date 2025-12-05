#!/usr/bin/env python3
"""
Neural Net Training Pipeline for CUDA Kernel Generation

Flow:
1. Task spec → NN input (tokenized)
2. NN generates builder commands (tokens)
3. Builder commands → CUDA source
4. CUDA source → compile
5. If compile OK → run tests
6. If tests OK → benchmark
7. Update PTX DB with results
8. Return reward signal

This is the full loop for RL training.
"""

import sys
import os
import re
import tempfile
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import IntEnum

sys.path.insert(0, str(Path(__file__).parent.parent))

from task import TaskSpec, U8GemvSpec


class ResultStatus(IntEnum):
    """Pipeline result status"""
    COMPILE_FAIL = 0
    RUN_FAIL = 1
    WRONG_OUTPUT = 2
    CORRECT_SLOW = 3
    CORRECT_FAST = 4


@dataclass
class PipelineResult:
    """Result of running the full pipeline"""
    status: ResultStatus
    compile_ok: bool = False
    run_ok: bool = False
    correct: bool = False
    time_us: Optional[float] = None
    baseline_us: Optional[float] = None
    speedup: Optional[float] = None
    error_msg: str = ""
    failed_instructions: List[str] = field(default_factory=list)

    @property
    def reward(self) -> float:
        """Compute reward signal for RL"""
        if not self.compile_ok:
            return -1.0  # Compile fail is bad
        if not self.run_ok:
            return -0.5  # Runtime error
        if not self.correct:
            return 0.0   # Wrong output, but at least it ran
        # Correct output - reward based on speed
        if self.speedup is None:
            return 0.5   # Correct but no perf data
        if self.speedup >= 1.0:
            return 0.5 + min(self.speedup - 1.0, 2.0)  # Up to 2.5 for 3x speedup
        else:
            return 0.5 * self.speedup  # Slower than baseline


@dataclass
class TaskInput:
    """Input to the neural net"""
    task_type: str           # "gemv", "gemm", etc.
    m: int
    k: int
    n: int = 1
    batch: int = 1
    dtype: str = "fp16"
    target_sm: int = 75

    def to_tokens(self) -> List[str]:
        """Convert to token sequence for NN input"""
        return [
            f"<task:{self.task_type}>",
            f"<m:{self.m}>",
            f"<k:{self.k}>",
            f"<n:{self.n}>",
            f"<batch:{self.batch}>",
            f"<dtype:{self.dtype}>",
            f"<sm:{self.target_sm}>",
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "m": self.m, "k": self.k, "n": self.n,
            "batch": self.batch,
            "dtype": self.dtype,
            "target_sm": self.target_sm,
        }


class Pipeline:
    """
    Full pipeline: NN output → compile → test → bench → reward

    Usage:
        pipe = Pipeline(target_sm=75)

        # From builder commands
        result = pipe.run_from_commands(commands, task_input)

        # From raw CUDA source
        result = pipe.run_from_source(cuda_source, task_input)

        # From TaskSpec (new task module)
        result = pipe.run_from_spec(cuda_source, task_spec)

        # Get reward for RL
        reward = result.reward
    """

    def __init__(self, target_sm: int = 75, db_path: Optional[str] = None):
        self.target_sm = target_sm
        self.db_path = db_path

        # Lazy imports
        self._db = None
        self._torch = None

    @property
    def db(self):
        if self._db is None:
            from ptx_db import PTXDB
            self._db = PTXDB(
                load_path=self.db_path if self.db_path and Path(self.db_path).exists() else None,
                bootstrap=False
            )
        return self._db

    @property
    def torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def run_from_commands(self, commands: List[str], task: TaskInput) -> PipelineResult:
        """
        Run pipeline from builder commands

        Args:
            commands: List of builder command strings, e.g.:
                ["cb.include('cuda_fp16.h', system=True)",
                 "cb.func_begin('kernel', ...)",
                 ...]
            task: Task specification

        Returns:
            PipelineResult with status and metrics
        """
        # Execute commands to get CUDA source
        try:
            cuda_source = self._commands_to_source(commands)
        except Exception as e:
            return PipelineResult(
                status=ResultStatus.COMPILE_FAIL,
                error_msg=f"Command execution failed: {e}"
            )

        return self.run_from_source(cuda_source, task)

    def run_from_spec(self, cuda_source: str, spec: TaskSpec) -> PipelineResult:
        """
        Run pipeline using a TaskSpec from the task module.

        This is the preferred method for new code - integrates with
        the task module's input generation and correctness checking.

        Args:
            cuda_source: Raw CUDA/C++ source code
            spec: TaskSpec instance (e.g., U8GemvSpec)

        Returns:
            PipelineResult with status and metrics
        """
        result = PipelineResult(status=ResultStatus.COMPILE_FAIL)

        # Step 1: Compile
        compile_ok, compile_msg, failed_instrs = self._compile(cuda_source)
        result.failed_instructions = failed_instrs

        if not compile_ok:
            result.error_msg = compile_msg
            self._update_db_compile_fail(failed_instrs)
            return result

        result.compile_ok = True
        self._update_db_compile_ok(cuda_source)

        # Step 2: Run using TaskSpec
        try:
            run_ok, output = self._run_with_spec(cuda_source, spec)
        except Exception as e:
            result.status = ResultStatus.RUN_FAIL
            result.error_msg = str(e)
            return result

        if not run_ok:
            result.status = ResultStatus.RUN_FAIL
            result.error_msg = "Kernel execution failed"
            return result

        result.run_ok = True

        # Step 3: Check correctness using TaskSpec
        inputs = spec.generate_input()
        A, B = inputs[0], inputs[1]
        ref = spec.reference(A, B)
        correct, check_msg = spec.check(output, ref)

        if not correct:
            result.status = ResultStatus.WRONG_OUTPUT
            result.error_msg = check_msg
            return result

        result.correct = True

        # Step 4: Benchmark
        time_us, baseline_us = self._benchmark_with_spec(cuda_source, spec)
        result.time_us = time_us
        result.baseline_us = baseline_us

        if time_us and baseline_us:
            result.speedup = baseline_us / time_us
            result.status = ResultStatus.CORRECT_FAST if result.speedup >= 1.0 else ResultStatus.CORRECT_SLOW
        else:
            result.status = ResultStatus.CORRECT_SLOW

        return result

    def _run_with_spec(self, cuda_source: str, spec: TaskSpec) -> Tuple[bool, Any]:
        """Run kernel using TaskSpec for input generation"""
        try:
            from torch.utils.cpp_extension import load_inline

            # Detect dtype from spec
            dtype = self._infer_dtype_from_spec(spec)

            # Create a minimal TaskInput for the wrapper
            task = TaskInput(
                task_type="gemv",
                m=getattr(spec, 'm', 1024),
                k=getattr(spec, 'k', 1024),
                n=1,
                batch=getattr(spec, 'batch', 1),
                dtype=dtype,
                target_sm=self.target_sm
            )

            wrapped = self._wrap_for_torch(cuda_source, task)

            module = load_inline(
                name=f"kernel_{id(cuda_source)}",
                cpp_sources=[""],
                cuda_sources=[wrapped],
                functions=["run_kernel"],
                verbose=False,
                extra_cuda_cflags=["-O3", f"-arch=sm_{self.target_sm}"]
            )

            # Generate input from spec
            inputs = spec.generate_input()
            A, B, C = inputs[0], inputs[1], inputs[2]

            self.torch.cuda.synchronize()
            output = module.run_kernel(A, B, C)
            self.torch.cuda.synchronize()

            return True, output

        except Exception as e:
            return False, None

    def _benchmark_with_spec(self, cuda_source: str, spec: TaskSpec,
                              warmup: int = 5, runs: int = 20) -> Tuple[Optional[float], Optional[float]]:
        """Benchmark using TaskSpec"""
        try:
            from torch.utils.cpp_extension import load_inline
            import dataclasses

            dtype = self._infer_dtype_from_spec(spec)
            task = TaskInput(
                task_type="gemv",
                m=getattr(spec, 'm', 1024),
                k=getattr(spec, 'k', 1024),
                n=1,
                batch=getattr(spec, 'batch', 1),
                dtype=dtype,
                target_sm=self.target_sm
            )

            wrapped = self._wrap_for_torch(cuda_source, task)

            module = load_inline(
                name=f"bench_{id(cuda_source)}",
                cpp_sources=[""],
                cuda_sources=[wrapped],
                functions=["run_kernel"],
                verbose=False,
                extra_cuda_cflags=["-O3", f"-arch=sm_{self.target_sm}"]
            )

            inputs = spec.generate_input()
            A, B, C = inputs[0], inputs[1], inputs[2]

            # Warmup
            for _ in range(warmup):
                module.run_kernel(A, B, C)
            self.torch.cuda.synchronize()

            # Benchmark kernel
            times = []
            for _ in range(runs):
                start = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)

                start.record()
                module.run_kernel(A, B, C)
                end.record()

                self.torch.cuda.synchronize()
                times.append(start.elapsed_time(end) * 1000)

            kernel_us = sum(times) / len(times)

            # Benchmark reference (from TaskSpec)
            times = []
            for _ in range(runs):
                start = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)

                start.record()
                spec.reference(A, B)
                end.record()

                self.torch.cuda.synchronize()
                times.append(start.elapsed_time(end) * 1000)

            baseline_us = sum(times) / len(times)

            return kernel_us, baseline_us

        except Exception:
            return None, None

    def _infer_dtype_from_spec(self, spec: TaskSpec) -> str:
        """Infer dtype string from TaskSpec"""
        # Check if spec has explicit dtype
        if hasattr(spec, 'dtype'):
            return spec.dtype

        # Infer from output type
        try:
            inputs = spec.generate_input()
            if len(inputs) >= 3:
                C = inputs[2]
                if C.dtype == self.torch.float16:
                    return "fp16"
                elif C.dtype == self.torch.float32:
                    return "fp32"
                elif C.dtype == self.torch.int32:
                    return "int32"
                elif C.dtype == self.torch.uint8:
                    return "uint8"
        except:
            pass

        return "fp32"  # default

    def run_from_source(self, cuda_source: str, task: TaskInput) -> PipelineResult:
        """
        Run pipeline from CUDA source

        Args:
            cuda_source: Raw CUDA/C++ source code
            task: Task specification

        Returns:
            PipelineResult with status and metrics
        """
        result = PipelineResult(status=ResultStatus.COMPILE_FAIL)

        # Step 1: Compile
        compile_ok, compile_msg, failed_instrs = self._compile(cuda_source)
        result.failed_instructions = failed_instrs

        if not compile_ok:
            result.error_msg = compile_msg
            self._update_db_compile_fail(failed_instrs)
            return result

        result.compile_ok = True
        self._update_db_compile_ok(cuda_source)

        # Step 2: Run
        run_ok, run_msg, output = self._run(cuda_source, task)

        if not run_ok:
            result.status = ResultStatus.RUN_FAIL
            result.error_msg = run_msg
            return result

        result.run_ok = True

        # Step 3: Check correctness
        correct, check_msg = self._check(output, task)

        if not correct:
            result.status = ResultStatus.WRONG_OUTPUT
            result.error_msg = check_msg
            return result

        result.correct = True

        # Step 4: Benchmark
        time_us, baseline_us = self._benchmark(cuda_source, task)
        result.time_us = time_us
        result.baseline_us = baseline_us

        if time_us and baseline_us:
            result.speedup = baseline_us / time_us
            result.status = ResultStatus.CORRECT_FAST if result.speedup >= 1.0 else ResultStatus.CORRECT_SLOW
        else:
            result.status = ResultStatus.CORRECT_SLOW

        return result

    def _commands_to_source(self, commands: List[str]) -> str:
        """Execute builder commands and return CUDA source"""
        # Import builder
        from builder import StructuredCudaBuilder as CudaBuilder
        from cuda_ast import (
            TypeRef, Parameter as Param, StructField as Field, LaunchBounds
        )

        # Create builder context
        cb = CudaBuilder()

        # Execute each command
        for cmd in commands:
            # Commands are like "cb.include('cuda.h', system=True)"
            # We need to execute them with cb in scope
            exec(cmd, {
                "cb": cb,
                "CudaBuilder": CudaBuilder,
                "TypeRef": TypeRef,
                "Param": Param,
                "Field": Field,
                "LaunchBounds": LaunchBounds,
            })

        # Build and return source
        return cb.build_source()

    def _compile(self, cuda_source: str) -> Tuple[bool, str, List[str]]:
        """
        Compile CUDA source (with caching)

        Returns: (success, message, failed_instructions)
        """
        from compile_and_learn import compile_cuda_source, parse_nvcc_errors

        success, result = compile_cuda_source(cuda_source, self.target_sm, use_cache=True)

        if success:
            return True, "OK", []
        else:
            # Parse errors to find bad instructions
            failed = parse_nvcc_errors(result)
            return False, result[:500], list(set(failed))

    def _run(self, cuda_source: str, task: TaskInput) -> Tuple[bool, str, Any]:
        """
        Run the kernel

        Returns: (success, message, output_tensors)
        """
        try:
            from torch.utils.cpp_extension import load_inline

            # Wrap source with entry point
            wrapped = self._wrap_for_torch(cuda_source, task)

            # Compile
            module = load_inline(
                name=f"kernel_{id(cuda_source)}",
                cpp_sources=[""],  # Header only
                cuda_sources=[wrapped],
                functions=["run_kernel"],
                verbose=False,
                extra_cuda_cflags=["-O3", f"-arch=sm_{self.target_sm}"]
            )

            # Generate input
            A, B, C = self._generate_input(task)

            # Run
            self.torch.cuda.synchronize()
            output = module.run_kernel(A, B, C)
            self.torch.cuda.synchronize()

            return True, "OK", output

        except Exception as e:
            return False, str(e), None

    def _check(self, output: Any, task: TaskInput) -> Tuple[bool, str]:
        """Check output correctness"""
        if output is None:
            return False, "No output"

        try:
            A, B, C = self._generate_input(task)
            expected = self.torch.bmm(A, B)

            diff = (output - expected).abs()
            max_diff = diff.max().item()

            # FP16 tolerance
            if max_diff < 0.1:
                return True, f"max_diff={max_diff:.4f}"
            else:
                return False, f"max_diff={max_diff:.4f} > 0.1"

        except Exception as e:
            return False, str(e)

    def _benchmark(self, cuda_source: str, task: TaskInput,
                   warmup: int = 5, runs: int = 20) -> Tuple[Optional[float], Optional[float]]:
        """
        Benchmark kernel

        Returns: (kernel_time_us, baseline_time_us)
        """
        try:
            from torch.utils.cpp_extension import load_inline

            wrapped = self._wrap_for_torch(cuda_source, task)

            module = load_inline(
                name=f"bench_{id(cuda_source)}",
                cpp_sources=[""],
                cuda_sources=[wrapped],
                functions=["run_kernel"],
                verbose=False,
                extra_cuda_cflags=["-O3", f"-arch=sm_{self.target_sm}"]
            )

            A, B, C = self._generate_input(task)

            # Warmup
            for _ in range(warmup):
                module.run_kernel(A, B, C)
            self.torch.cuda.synchronize()

            # Benchmark kernel
            times = []
            for _ in range(runs):
                start = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)

                start.record()
                module.run_kernel(A, B, C)
                end.record()

                self.torch.cuda.synchronize()
                times.append(start.elapsed_time(end) * 1000)  # ms -> us

            kernel_us = sum(times) / len(times)

            # Benchmark baseline (torch.bmm)
            times = []
            for _ in range(runs):
                start = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)

                start.record()
                self.torch.bmm(A, B)
                end.record()

                self.torch.cuda.synchronize()
                times.append(start.elapsed_time(end) * 1000)

            baseline_us = sum(times) / len(times)

            return kernel_us, baseline_us

        except Exception as e:
            return None, None

    def _generate_input(self, task: TaskInput):
        """Generate input tensors for task"""
        self.torch.manual_seed(42)
        device = "cuda"
        dtype = self.torch.float16 if task.dtype == "fp16" else self.torch.float32

        A = self.torch.randn(task.batch, task.m, task.k, dtype=dtype, device=device)
        B = self.torch.randn(task.batch, task.k, task.n, dtype=dtype, device=device)
        C = self.torch.zeros(task.batch, task.m, task.n, dtype=dtype, device=device)

        return A, B, C

    def _extract_kernel_info(self, cuda_source: str) -> Tuple[Optional[str], List[str]]:
        """
        Extract kernel function name and parameter types from CUDA source.

        Returns: (kernel_name, param_types) or (None, []) if not found
        """
        # Match __global__ void kernel_name(params...)
        pattern = r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)'
        match = re.search(pattern, cuda_source)

        if not match:
            return None, []

        kernel_name = match.group(1)
        params_str = match.group(2)

        # Parse parameter types (simplified)
        param_types = []
        for param in params_str.split(','):
            param = param.strip()
            if param:
                # Extract type (everything except last word which is the name)
                parts = param.rsplit(None, 1)
                if len(parts) >= 1:
                    param_types.append(parts[0].strip())

        return kernel_name, param_types

    def _wrap_for_torch(self, cuda_source: str, task: TaskInput) -> str:
        """Wrap CUDA source with torch extension entry point"""

        kernel_name, param_types = self._extract_kernel_info(cuda_source)

        if kernel_name is None:
            # Fallback - just add a stub
            return cuda_source + """

#include <torch/extension.h>

torch::Tensor run_kernel(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // No kernel found in source
    return C;
}
"""

        # Determine grid/block dimensions based on task
        # For GEMV: one block per row, threads handle K reduction
        block_x = 256  # threads per block
        grid_x = f"A.size(1)"  # M rows
        grid_y = f"A.size(0)"  # batch

        # Generate wrapper that calls the kernel
        wrapper = f"""

#include <torch/extension.h>
#include <cuda_runtime.h>

torch::Tensor run_kernel(torch::Tensor A, torch::Tensor B, torch::Tensor C) {{
    // Get tensor dimensions
    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    // Launch configuration
    dim3 grid({grid_x}, {grid_y}, 1);
    dim3 block({block_x}, 1, 1);

    // Get raw pointers
    auto A_ptr = A.data_ptr<{self._torch_dtype_to_cpp(task.dtype)}>();
    auto B_ptr = B.data_ptr<{self._torch_dtype_to_cpp(task.dtype)}>();
    auto C_ptr = C.data_ptr<{self._torch_dtype_to_cpp(task.dtype)}>();

    // Launch kernel
    {kernel_name}<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, K, batch);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {{
        throw std::runtime_error(cudaGetErrorString(err));
    }}

    return C;
}}
"""
        return cuda_source + wrapper

    def _torch_dtype_to_cpp(self, dtype: str) -> str:
        """Convert dtype string to C++ type"""
        mapping = {
            "fp16": "at::Half",
            "fp32": "float",
            "int32": "int32_t",
            "uint8": "uint8_t",
        }
        return mapping.get(dtype, "float")

    def _update_db_compile_fail(self, failed_instrs: List[str]):
        """Update DB with compile failures"""
        from ptx_db import TestStatus
        for instr in failed_instrs:
            self.db.record_result(instr, self.target_sm, TestStatus.COMPILE_FAIL)
        if self.db_path:
            self.db.save(self.db_path)

    def _update_db_compile_ok(self, cuda_source: str):
        """Update DB with compile successes"""
        from ptx_db import TestStatus
        import re

        # Extract PTX instructions from source
        asm_pattern = r'asm\s*(?:volatile)?\s*\(\s*"([^"]+)"'
        for match in re.finditer(asm_pattern, cuda_source, re.DOTALL):
            asm_body = match.group(1)
            for line in asm_body.replace('\\n', '\n').split('\n'):
                line = line.strip().rstrip(';')
                if line and not line.startswith('//') and not line.startswith('.'):
                    parts = line.split()
                    if parts:
                        self.db.record_result(parts[0], self.target_sm, TestStatus.COMPILE_OK)

        if self.db_path:
            self.db.save(self.db_path)

    def save_db(self):
        """Explicitly save DB"""
        if self.db_path:
            self.db.save(self.db_path)


# =========================================================
# Example usage for NN training
# =========================================================

def example_training_step():
    """Example of how NN training would use this"""

    # 1. Define task
    task = TaskInput(
        task_type="gemv",
        m=1024,
        k=1024,
        n=1,
        batch=1,
        dtype="fp16",
        target_sm=75
    )

    # 2. NN generates builder commands (this would be from the model)
    commands = [
        "cb.include('cuda_fp16.h', system=True)",
        "cb.func_begin('gemv_kernel', TypeRef('void'), params=[], qualifier='__global__')",
        "cb.stmt('// kernel body here')",
        "cb.func_end()",
    ]

    # 3. Run pipeline
    pipe = Pipeline(target_sm=75, db_path="learned.json")
    result = pipe.run_from_commands(commands, task)

    # 4. Get reward for RL
    reward = result.reward

    print(f"Status: {result.status.name}")
    print(f"Reward: {reward}")
    print(f"Compile: {result.compile_ok}, Run: {result.run_ok}, Correct: {result.correct}")
    if result.time_us:
        print(f"Time: {result.time_us:.2f} us, Speedup: {result.speedup:.2f}x")

    return reward


if __name__ == "__main__":
    print("Pipeline module - import and use Pipeline class")
    print()
    print("Example:")
    print("  from pipeline import Pipeline, TaskInput")
    print("  pipe = Pipeline(target_sm=75)")
    print("  result = pipe.run_from_source(cuda_code, task)")
    print("  reward = result.reward")
