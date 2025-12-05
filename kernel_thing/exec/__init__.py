"""
Execution Layer - Compile, Run, Benchmark CUDA kernels

Provides cached compilation and fast kernel execution for RL training.

- cuda_ast.py: CUDA AST node definitions
- ptx_ast.py: PTX AST node definitions
- ptx_db.py: SM-specific PTX instruction cache
- compile.py: No torch dependency, pure nvcc wrapper
- run.py: Requires torch for kernel execution
- bench.py: Requires torch for timing
"""

# AST modules
from .cuda_ast import (
    CudaNode, CudaModule, Include, Define, Pragma, Comment, RawCode, BlankLine,
    TypeRef, Variable, Parameter, StructField, UsingDecl, Struct,
    LaunchBounds, FunctionDecl, Function, Statement, VarDecl, Return,
    If, For, While, Block, InlineAsm, StaticAssert, Constexpr, Lambda,
    FunctionQualifier, StorageClass
)
from .ptx_ast import (
    PTXModule, Instruction, Directive, Label, RegisterDecl, SharedDecl,
    RegisterOp, ImmediateOp, VectorOp, MemoryOp, SymbolOp, Operand,
    Statement as PTXStatement
)

# Compile module (no torch required)
from .compile import compile_cuda, compile_source, get_cache, clear_cache, get_sm_version

__all__ = [
    # CUDA AST
    "CudaNode", "CudaModule", "Include", "Define", "Pragma", "Comment", "RawCode", "BlankLine",
    "TypeRef", "Variable", "Parameter", "StructField", "UsingDecl", "Struct",
    "LaunchBounds", "FunctionDecl", "Function", "Statement", "VarDecl", "Return",
    "If", "For", "While", "Block", "InlineAsm", "StaticAssert", "Constexpr", "Lambda",
    "FunctionQualifier", "StorageClass",
    # PTX AST
    "PTXModule", "Instruction", "Directive", "Label", "RegisterDecl", "SharedDecl",
    "RegisterOp", "ImmediateOp", "VectorOp", "MemoryOp", "SymbolOp", "Operand", "PTXStatement",
    # Compile (no torch)
    "compile_cuda",
    "compile_source",
    "get_cache",
    "clear_cache",
    "get_sm_version",
]


def __getattr__(name):
    """Lazy imports for torch-dependent modules"""
    if name in ("load_and_run", "run_kernel"):
        from . import run
        return getattr(run, name)
    elif name in ("benchmark", "benchmark_reference", "benchmark_kernel_source"):
        from . import bench
        return getattr(bench, name)
    raise AttributeError(f"module 'exec' has no attribute '{name}'")
