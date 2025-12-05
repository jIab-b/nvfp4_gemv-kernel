"""
uint8 Unscaled GEMV Task

C = A @ B where:
  A: [batch, M, K] uint8
  B: [batch, K, 1] uint8
  C: [batch, M, 1] int32 (accumulated to avoid overflow)

This is the simplest possible GEMV - no scales, no fancy types.
Good for testing on SM 75+ (Turing and later).
"""

from dataclasses import dataclass
from typing import Tuple

import torch

from . import TaskSpec


@dataclass
class U8GemvSpec(TaskSpec):
    """uint8 GEMV specification"""
    m: int = 1024
    k: int = 1024
    batch: int = 1
    seed: int = 42

    def generate_input(self, device: str = "cuda") -> Tuple[torch.Tensor, ...]:
        """
        Generate random uint8 input tensors.

        Returns: (A, B, C) where C is pre-allocated output buffer
        """
        torch.manual_seed(self.seed)

        # uint8 inputs (0-255)
        A = torch.randint(0, 256, (self.batch, self.m, self.k),
                          dtype=torch.uint8, device=device)
        B = torch.randint(0, 256, (self.batch, self.k, 1),
                          dtype=torch.uint8, device=device)

        # int32 output to avoid overflow (max value = 255 * 255 * K)
        C = torch.zeros(self.batch, self.m, 1, dtype=torch.int32, device=device)

        return A, B, C

    def reference(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Reference implementation using PyTorch"""
        # Cast to int32 for accumulation, then bmm
        A_i32 = A.to(torch.int32)
        B_i32 = B.to(torch.int32)
        return torch.bmm(A_i32, B_i32)

    def check(self, output: torch.Tensor, ref: torch.Tensor,
              rtol: float = 0.0, atol: float = 0.0) -> Tuple[bool, str]:
        """
        Check correctness - for int32, we expect exact match.
        """
        if output.shape != ref.shape:
            return False, f"Shape mismatch: {output.shape} vs {ref.shape}"

        if output.dtype != ref.dtype:
            # Cast if needed
            output = output.to(ref.dtype)

        diff = torch.abs(output - ref)
        num_bad = (diff > atol).sum().item()

        if num_bad > 0:
            worst_idx = torch.argmax(diff.flatten())
            worst_diff = diff.flatten()[worst_idx].item()
            got_val = output.flatten()[worst_idx].item()
            ref_val = ref.flatten()[worst_idx].item()
            return False, f"{num_bad} mismatches, worst: got={got_val} ref={ref_val} diff={worst_diff}"

        return True, "OK"

    def flops(self) -> int:
        """2 ops per multiply-add"""
        return 2 * self.m * self.k * self.batch

    def __str__(self) -> str:
        return f"m:{self.m};k:{self.k};batch:{self.batch};seed:{self.seed}"


# Test cases - small for quick iteration
TESTS = [
    U8GemvSpec(m=64, k=64, batch=1),
    U8GemvSpec(m=128, k=128, batch=1),
    U8GemvSpec(m=256, k=256, batch=1),
    U8GemvSpec(m=512, k=512, batch=1),
    U8GemvSpec(m=1024, k=1024, batch=1),
    U8GemvSpec(m=256, k=1024, batch=2),
    U8GemvSpec(m=1024, k=256, batch=2),
]

# Benchmark cases - match original task_gemv.yml sizes where reasonable
BENCHMARKS = [
    U8GemvSpec(m=4096, k=4096, batch=1),
    U8GemvSpec(m=7168, k=2048, batch=4),
    U8GemvSpec(m=4096, k=7168, batch=8),
    U8GemvSpec(m=7168, k=16384, batch=1),
]


if __name__ == "__main__":
    print("uint8 GEMV Task - Sanity Check")
    print("=" * 50)

    for spec in TESTS[:3]:
        A, B, C = spec.generate_input()
        C_ref = spec.reference(A, B)

        print(f"{spec}")
        print(f"  A: {A.shape} {A.dtype}")
        print(f"  B: {B.shape} {B.dtype}")
        print(f"  C: {C_ref.shape} {C_ref.dtype}")
        print(f"  Max output value: {C_ref.max().item()}")
        print(f"  FLOPs: {spec.flops():,}")
        print()
