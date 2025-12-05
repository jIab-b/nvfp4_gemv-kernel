"""
Task Definitions for Kernel Benchmarking

Each task defines:
- Input generation
- Reference implementation
- Correctness checking
- Test cases and benchmarks
"""

from dataclasses import dataclass
from typing import Tuple, List, Any
from abc import ABC, abstractmethod

import torch


@dataclass
class TaskSpec(ABC):
    """Base task specification"""
    seed: int = 42

    @abstractmethod
    def generate_input(self, device: str = "cuda") -> Tuple[torch.Tensor, ...]:
        """Generate input tensors for this task"""
        pass

    @abstractmethod
    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Reference implementation"""
        pass

    @abstractmethod
    def check(self, output: torch.Tensor, ref: torch.Tensor) -> Tuple[bool, str]:
        """Check correctness"""
        pass

    @abstractmethod
    def flops(self) -> int:
        """Return FLOPs for this task (for GFLOPS calculation)"""
        pass

    def __str__(self) -> str:
        """String representation for logging"""
        fields = [f"{k}:{v}" for k, v in self.__dict__.items()]
        return ";".join(fields)

    @classmethod
    def from_str(cls, s: str) -> "TaskSpec":
        """Parse from string representation"""
        parts = {}
        for part in s.split(";"):
            if ":" in part:
                key, val = part.split(":", 1)
                try:
                    parts[key.strip()] = int(val.strip())
                except ValueError:
                    parts[key.strip()] = val.strip()
        return cls(**parts)


# Import task implementations
from .u8gemv import U8GemvSpec, TESTS, BENCHMARKS

__all__ = [
    "TaskSpec",
    "U8GemvSpec",
    "TESTS",
    "BENCHMARKS",
]
