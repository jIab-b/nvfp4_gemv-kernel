"""
CUDA C++ Abstract Syntax Tree

Structured representation of CUDA/C++ source code for programmatic manipulation.
Parses includes, structs, functions (device/global/host), templates, and control flow.

Design Goals:
- Lossless roundtrip: parse -> AST -> emit produces identical output
- Expose decision points: cache hints, unroll factors, launch bounds
- Structured builder commands rather than verbatim text blobs
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Any
from enum import Enum


# =============================================================================
# ENUMS FOR CUDA-SPECIFIC ATTRIBUTES
# =============================================================================

class FunctionQualifier(Enum):
    NONE = ""
    DEVICE = "__device__"
    GLOBAL = "__global__"
    HOST = "__host__"
    HOST_DEVICE = "__host__ __device__"


class StorageClass(Enum):
    NONE = ""
    STATIC = "static"
    EXTERN = "extern"
    CONSTEXPR = "constexpr"
    INLINE = "inline"


# =============================================================================
# COMPILATION FLAGS
# =============================================================================

@dataclass
class CompilationFlags:
    """CUDA/NVCC compilation flags and options."""
    maxrregcount: Optional[int] = None  # -maxrregcount N
    arch: Optional[str] = None  # -arch sm_90a
    use_fast_math: bool = False  # --use_fast_math
    ftz: Optional[bool] = None  # --ftz=true/false (flush to zero)
    prec_div: Optional[bool] = None  # --prec-div=true/false
    prec_sqrt: Optional[bool] = None  # --prec-sqrt=true/false
    fmad: Optional[bool] = None  # --fmad=true/false
    extra_flags: List[str] = field(default_factory=list)  # arbitrary extra flags

    def to_nvcc_args(self) -> List[str]:
        """Convert to NVCC command-line arguments."""
        args = []
        if self.maxrregcount is not None:
            args.append(f"-maxrregcount={self.maxrregcount}")
        if self.arch:
            args.append(f"-arch={self.arch}")
        if self.use_fast_math:
            args.append("--use_fast_math")
        if self.ftz is not None:
            args.append(f"--ftz={'true' if self.ftz else 'false'}")
        if self.prec_div is not None:
            args.append(f"--prec-div={'true' if self.prec_div else 'false'}")
        if self.prec_sqrt is not None:
            args.append(f"--prec-sqrt={'true' if self.prec_sqrt else 'false'}")
        if self.fmad is not None:
            args.append(f"--fmad={'true' if self.fmad else 'false'}")
        args.extend(self.extra_flags)
        return args

    def to_builder(self) -> str:
        """Generate builder command."""
        parts = []
        if self.maxrregcount is not None:
            parts.append(f"maxrregcount={self.maxrregcount}")
        if self.arch:
            parts.append(f'arch="{self.arch}"')
        if self.use_fast_math:
            parts.append("use_fast_math=True")
        if self.ftz is not None:
            parts.append(f"ftz={self.ftz}")
        if self.prec_div is not None:
            parts.append(f"prec_div={self.prec_div}")
        if self.prec_sqrt is not None:
            parts.append(f"prec_sqrt={self.prec_sqrt}")
        if self.fmad is not None:
            parts.append(f"fmad={self.fmad}")
        if self.extra_flags:
            parts.append(f"extra_flags={repr(self.extra_flags)}")
        if parts:
            return f"cb.flags({', '.join(parts)})"
        return ""


# =============================================================================
# BASE AST NODE
# =============================================================================

@dataclass
class CudaNode:
    """Base class for all CUDA AST nodes."""

    def emit(self) -> str:
        raise NotImplementedError

    def to_builder(self) -> str:
        raise NotImplementedError


# =============================================================================
# PREPROCESSOR DIRECTIVES
# =============================================================================

@dataclass
class Include(CudaNode):
    """#include directive"""
    path: str
    is_system: bool = False  # <...> vs "..."

    def emit(self) -> str:
        if self.is_system:
            return f"#include <{self.path}>"
        return f'#include "{self.path}"'

    def to_builder(self) -> str:
        return f'cb.include("{self.path}", system={self.is_system})'


@dataclass
class Define(CudaNode):
    """#define directive"""
    name: str
    value: Optional[str] = None
    params: Optional[List[str]] = None  # For function-like macros

    def emit(self) -> str:
        if self.params is not None:
            params_str = ", ".join(self.params)
            if self.value:
                return f"#define {self.name}({params_str}) {self.value}"
            return f"#define {self.name}({params_str})"
        if self.value:
            return f"#define {self.name} {self.value}"
        return f"#define {self.name}"

    def to_builder(self) -> str:
        if self.params is not None:
            return f'cb.define("{self.name}", value={repr(self.value)}, params={repr(self.params)})'
        if self.value:
            return f'cb.define("{self.name}", value={repr(self.value)})'
        return f'cb.define("{self.name}")'


@dataclass
class Pragma(CudaNode):
    """#pragma directive"""
    content: str

    def emit(self) -> str:
        return f"#pragma {self.content}"

    def to_builder(self) -> str:
        return f'cb.pragma("{self.content}")'


@dataclass
class PreprocessorIf(CudaNode):
    """#if/#ifdef/#ifndef/#elif/#else/#endif block"""
    condition: str
    directive: str  # "if", "ifdef", "ifndef", "elif", "else"
    body: List[CudaNode] = field(default_factory=list)
    else_branch: Optional[PreprocessorIf] = None

    def emit(self) -> str:
        if self.directive == "else":
            lines = ["#else"]
        elif self.directive in ("ifdef", "ifndef"):
            lines = [f"#{self.directive} {self.condition}"]
        else:
            lines = [f"#{self.directive} {self.condition}"]

        for node in self.body:
            lines.append(node.emit())

        if self.else_branch:
            lines.append(self.else_branch.emit())
        else:
            lines.append("#endif")

        return "\n".join(lines)

    def to_builder(self) -> str:
        # Simplified - complex preprocessor blocks may need raw handling
        return f'cb.raw({repr(self.emit())})'


# =============================================================================
# COMMENTS AND RAW TEXT
# =============================================================================

@dataclass
class Comment(CudaNode):
    """C/C++ comment"""
    text: str
    is_multiline: bool = False

    def emit(self) -> str:
        if self.is_multiline:
            return f"/*{self.text}*/"
        return f"// {self.text}"

    def to_builder(self) -> str:
        if self.is_multiline:
            return f'cb.comment({repr(self.text)}, multiline=True)'
        return f'cb.comment({repr(self.text)})'


@dataclass
class RawCode(CudaNode):
    """Unparsed raw code - fallback for complex constructs"""
    text: str

    def emit(self) -> str:
        return self.text

    def to_builder(self) -> str:
        # Use triple quotes for readability
        escaped = self.text.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
        return f'cb.raw("""{escaped}""")'


@dataclass
class BlankLine(CudaNode):
    """Empty line for formatting"""
    count: int = 1

    def emit(self) -> str:
        return "\n" * (self.count - 1)  # -1 because join adds one

    def to_builder(self) -> str:
        if self.count == 1:
            return 'cb.blank()'
        return f'cb.blank({self.count})'


# =============================================================================
# TYPE SYSTEM
# =============================================================================

@dataclass
class TypeRef(CudaNode):
    """Type reference (simple or templated)"""
    name: str
    template_args: Optional[List[Union[str, TypeRef]]] = None
    is_const: bool = False
    is_volatile: bool = False
    is_pointer: bool = False
    is_reference: bool = False
    array_dims: Optional[List[str]] = None  # e.g., ["16", ""] for [16][]

    def emit(self) -> str:
        parts = []
        if self.is_const:
            parts.append("const")
        if self.is_volatile:
            parts.append("volatile")

        type_str = self.name
        if self.template_args:
            args = []
            for arg in self.template_args:
                if isinstance(arg, TypeRef):
                    args.append(arg.emit())
                else:
                    args.append(str(arg))
            type_str += f"<{', '.join(args)}>"

        parts.append(type_str)

        if self.is_pointer:
            parts.append("*")
        if self.is_reference:
            parts.append("&")

        result = " ".join(parts)
        if self.is_pointer or self.is_reference:
            result = result.replace(" *", "*").replace(" &", "&")

        if self.array_dims:
            for dim in self.array_dims:
                result += f"[{dim}]"

        return result

    def to_builder(self) -> str:
        parts = [f'"{self.name}"']
        if self.template_args:
            args_str = repr([a if isinstance(a, str) else a.emit() for a in self.template_args])
            parts.append(f"template_args={args_str}")
        if self.is_const:
            parts.append("const=True")
        if self.is_pointer:
            parts.append("pointer=True")
        if self.is_reference:
            parts.append("reference=True")
        if self.array_dims:
            parts.append(f"array={repr(self.array_dims)}")
        return f'TypeRef({", ".join(parts)})'


# =============================================================================
# VARIABLES AND PARAMETERS
# =============================================================================

@dataclass
class Variable(CudaNode):
    """Variable declaration"""
    name: str
    type: TypeRef
    initializer: Optional[str] = None
    storage: StorageClass = StorageClass.NONE
    attributes: List[str] = field(default_factory=list)  # __restrict__, etc.

    def emit(self) -> str:
        parts = []
        if self.storage != StorageClass.NONE:
            parts.append(self.storage.value)
        parts.append(self.type.emit())

        # Handle __restrict__ and similar
        if self.attributes:
            parts.append(" ".join(self.attributes))

        parts.append(self.name)

        result = " ".join(parts)
        if self.initializer:
            result += f" = {self.initializer}"
        return result

    def to_builder(self) -> str:
        parts = [f'"{self.name}"', self.type.to_builder()]
        if self.initializer:
            parts.append(f'init={repr(self.initializer)}')
        if self.storage != StorageClass.NONE:
            parts.append(f'storage="{self.storage.value}"')
        if self.attributes:
            parts.append(f'attrs={repr(self.attributes)}')
        return f'Variable({", ".join(parts)})'


@dataclass
class Parameter(CudaNode):
    """Function parameter"""
    name: str
    type: TypeRef
    default: Optional[str] = None

    def emit(self) -> str:
        result = f"{self.type.emit()} {self.name}"
        if self.default:
            result += f" = {self.default}"
        return result

    def to_builder(self) -> str:
        parts = [f'"{self.name}"', self.type.to_builder()]
        if self.default:
            parts.append(f'default={repr(self.default)}')
        return f'Param({", ".join(parts)})'


# =============================================================================
# STRUCT/CLASS DEFINITIONS
# =============================================================================

@dataclass
class StructField(CudaNode):
    """Field in a struct/class"""
    name: str
    type: TypeRef
    initializer: Optional[str] = None

    def emit(self) -> str:
        result = f"    {self.type.emit()} {self.name}"
        if self.initializer:
            result += f" = {self.initializer}"
        return result + ";"

    def to_builder(self) -> str:
        parts = [f'"{self.name}"', self.type.to_builder()]
        if self.initializer:
            parts.append(f'init={repr(self.initializer)}')
        return f'Field({", ".join(parts)})'


@dataclass
class UsingDecl(CudaNode):
    """using declaration inside struct/class"""
    name: str
    type_expr: str

    def emit(self) -> str:
        return f"    using {self.name} = {self.type_expr};"

    def to_builder(self) -> str:
        return f'cb.using("{self.name}", "{self.type_expr}")'


@dataclass
class Struct(CudaNode):
    """Struct or class definition"""
    name: str
    members: List[Union[StructField, UsingDecl, RawCode]] = field(default_factory=list)
    is_class: bool = False
    template_params: Optional[List[str]] = None

    def emit(self) -> str:
        lines = []

        if self.template_params:
            lines.append(f"template<{', '.join(self.template_params)}>")

        keyword = "class" if self.is_class else "struct"
        lines.append(f"{keyword} {self.name} {{")

        for member in self.members:
            lines.append(member.emit())

        lines.append("};")
        return "\n".join(lines)

    def to_builder(self) -> str:
        lines = []
        template_str = f", template={repr(self.template_params)}" if self.template_params else ""
        class_str = ", is_class=True" if self.is_class else ""
        lines.append(f'cb.struct_begin("{self.name}"{template_str}{class_str})')
        for member in self.members:
            if isinstance(member, StructField):
                lines.append(f'    cb.field({member.to_builder()})')
            elif isinstance(member, UsingDecl):
                lines.append(f'    {member.to_builder()}')
            else:
                lines.append(f'    cb.raw({repr(member.emit())})')
        lines.append('cb.struct_end()')
        return "\n".join(lines)


# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

@dataclass
class LaunchBounds(CudaNode):
    """__launch_bounds__ attribute - supports expressions like ROWS_PER_BLOCK * 32"""
    max_threads: Union[int, str]  # Can be int or expression string
    min_blocks: Optional[Union[int, str]] = None

    def emit(self) -> str:
        if self.min_blocks is not None:
            return f"__launch_bounds__({self.max_threads}, {self.min_blocks})"
        return f"__launch_bounds__({self.max_threads})"

    def to_builder(self) -> str:
        max_repr = self.max_threads if isinstance(self.max_threads, int) else repr(self.max_threads)
        if self.min_blocks is not None:
            min_repr = self.min_blocks if isinstance(self.min_blocks, int) else repr(self.min_blocks)
            return f'LaunchBounds({max_repr}, {min_repr})'
        return f'LaunchBounds({max_repr})'


@dataclass
class FunctionDecl(CudaNode):
    """Function declaration (forward declaration)"""
    name: str
    return_type: TypeRef
    params: List[Parameter] = field(default_factory=list)
    qualifier: FunctionQualifier = FunctionQualifier.NONE
    is_inline: bool = False
    is_forceinline: bool = False
    storage: StorageClass = StorageClass.NONE
    template_params: Optional[List[str]] = None

    def emit(self) -> str:
        parts = []

        if self.template_params:
            parts.append(f"template<{', '.join(self.template_params)}>")
            parts.append("\n")

        if self.storage != StorageClass.NONE:
            parts.append(self.storage.value)

        if self.qualifier != FunctionQualifier.NONE:
            parts.append(self.qualifier.value)

        if self.is_forceinline:
            parts.append("__forceinline__")
        elif self.is_inline:
            parts.append("inline")

        parts.append(self.return_type.emit())

        params_str = ", ".join(p.emit() for p in self.params)
        parts.append(f"{self.name}({params_str})")

        return " ".join(parts) + ";"

    def to_builder(self) -> str:
        parts = [f'"{self.name}"', self.return_type.to_builder()]
        if self.params:
            params_str = "[" + ", ".join(p.to_builder() for p in self.params) + "]"
            parts.append(f"params={params_str}")
        if self.qualifier != FunctionQualifier.NONE:
            parts.append(f'qualifier="{self.qualifier.value}"')
        return f'cb.func_decl({", ".join(parts)})'


@dataclass
class Function(CudaNode):
    """Complete function definition"""
    name: str
    return_type: TypeRef
    params: List[Parameter] = field(default_factory=list)
    body: List[CudaNode] = field(default_factory=list)
    qualifier: FunctionQualifier = FunctionQualifier.NONE
    is_inline: bool = False
    is_forceinline: bool = False
    storage: StorageClass = StorageClass.NONE
    template_params: Optional[List[str]] = None
    launch_bounds: Optional[LaunchBounds] = None
    grid_constant_params: List[str] = field(default_factory=list)  # params with __grid_constant__

    def emit(self) -> str:
        lines = []

        if self.template_params:
            lines.append(f"template<{', '.join(self.template_params)}>")

        sig_parts = []

        if self.storage != StorageClass.NONE:
            sig_parts.append(self.storage.value)

        if self.qualifier != FunctionQualifier.NONE:
            sig_parts.append(self.qualifier.value)

        if self.launch_bounds:
            sig_parts.append(self.launch_bounds.emit())

        if self.is_forceinline:
            sig_parts.append("__forceinline__")
        elif self.is_inline:
            sig_parts.append("inline")

        sig_parts.append(self.return_type.emit())

        # Handle __grid_constant__ params
        param_strs = []
        for p in self.params:
            p_str = p.emit()
            if p.name in self.grid_constant_params:
                p_str = "const __grid_constant__ " + p_str.lstrip("const ")
            param_strs.append(p_str)

        params_str = ", ".join(param_strs) if param_strs else ""

        sig = " ".join(sig_parts)
        lines.append(f"{sig} {self.name}({params_str})")
        lines.append("{")

        for node in self.body:
            body_lines = node.emit().split('\n')
            for line in body_lines:
                if line.strip():
                    lines.append(f"    {line}")
                else:
                    lines.append("")

        lines.append("}")
        return "\n".join(lines)

    def to_builder(self) -> str:
        lines = []

        # Build function header
        parts = [f'"{self.name}"', self.return_type.to_builder()]

        if self.params:
            params_str = "[" + ", ".join(p.to_builder() for p in self.params) + "]"
            parts.append(f"params={params_str}")

        if self.qualifier != FunctionQualifier.NONE:
            parts.append(f'qualifier="{self.qualifier.value}"')

        if self.is_forceinline:
            parts.append("forceinline=True")
        elif self.is_inline:
            parts.append("inline=True")

        if self.template_params:
            parts.append(f"template={repr(self.template_params)}")

        if self.launch_bounds:
            parts.append(f"launch_bounds={self.launch_bounds.to_builder()}")

        lines.append(f'cb.func_begin({", ".join(parts)})')

        # Body
        for node in self.body:
            node_builder = node.to_builder()
            for line in node_builder.split('\n'):
                lines.append(f"    {line}")

        lines.append("cb.func_end()")
        return "\n".join(lines)


# =============================================================================
# STATEMENTS
# =============================================================================

@dataclass
class Statement(CudaNode):
    """Simple statement (expression + semicolon)"""
    expr: str

    def emit(self) -> str:
        return f"{self.expr};"

    def to_builder(self) -> str:
        return f'cb.stmt({repr(self.expr)})'


@dataclass
class VarDecl(CudaNode):
    """Variable declaration statement"""
    var: Variable

    def emit(self) -> str:
        return f"{self.var.emit()};"

    def to_builder(self) -> str:
        return f'cb.var({self.var.to_builder()})'


@dataclass
class Return(CudaNode):
    """Return statement"""
    value: Optional[str] = None

    def emit(self) -> str:
        if self.value:
            return f"return {self.value};"
        return "return;"

    def to_builder(self) -> str:
        if self.value:
            return f'cb.ret({repr(self.value)})'
        return 'cb.ret()'


# =============================================================================
# CONTROL FLOW
# =============================================================================

@dataclass
class If(CudaNode):
    """If statement"""
    condition: str
    then_body: List[CudaNode] = field(default_factory=list)
    else_body: Optional[List[CudaNode]] = None
    is_constexpr: bool = False

    def emit(self) -> str:
        constexpr = " constexpr" if self.is_constexpr else ""
        lines = [f"if{constexpr} ({self.condition}) {{"]

        for node in self.then_body:
            for line in node.emit().split('\n'):
                lines.append(f"    {line}")

        if self.else_body:
            lines.append("} else {")
            for node in self.else_body:
                for line in node.emit().split('\n'):
                    lines.append(f"    {line}")

        lines.append("}")
        return "\n".join(lines)

    def to_builder(self) -> str:
        constexpr = ", constexpr=True" if self.is_constexpr else ""
        lines = [f'cb.if_begin({repr(self.condition)}{constexpr})']
        for node in self.then_body:
            lines.append(f"    {node.to_builder()}")
        if self.else_body:
            lines.append('cb.else_begin()')
            for node in self.else_body:
                lines.append(f"    {node.to_builder()}")
        lines.append('cb.if_end()')
        return "\n".join(lines)


@dataclass
class For(CudaNode):
    """For loop"""
    init: str
    condition: str
    increment: str
    body: List[CudaNode] = field(default_factory=list)
    pragma_unroll: Optional[int] = None  # #pragma unroll N (or -1 for just #pragma unroll)

    def emit(self) -> str:
        lines = []
        if self.pragma_unroll is not None:
            if self.pragma_unroll == -1:
                lines.append("#pragma unroll")
            else:
                lines.append(f"#pragma unroll {self.pragma_unroll}")

        lines.append(f"for ({self.init}; {self.condition}; {self.increment}) {{")

        for node in self.body:
            for line in node.emit().split('\n'):
                lines.append(f"    {line}")

        lines.append("}")
        return "\n".join(lines)

    def to_builder(self) -> str:
        unroll_str = ""
        if self.pragma_unroll is not None:
            if self.pragma_unroll == -1:
                unroll_str = ", unroll=True"
            else:
                unroll_str = f", unroll={self.pragma_unroll}"

        lines = [f'cb.for_begin({repr(self.init)}, {repr(self.condition)}, {repr(self.increment)}{unroll_str})']
        for node in self.body:
            lines.append(f"    {node.to_builder()}")
        lines.append('cb.for_end()')
        return "\n".join(lines)


@dataclass
class While(CudaNode):
    """While loop"""
    condition: str
    body: List[CudaNode] = field(default_factory=list)

    def emit(self) -> str:
        lines = [f"while ({self.condition}) {{"]
        for node in self.body:
            for line in node.emit().split('\n'):
                lines.append(f"    {line}")
        lines.append("}")
        return "\n".join(lines)

    def to_builder(self) -> str:
        lines = [f'cb.while_begin({repr(self.condition)})']
        for node in self.body:
            lines.append(f"    {node.to_builder()}")
        lines.append('cb.while_end()')
        return "\n".join(lines)


@dataclass
class Block(CudaNode):
    """Standalone scoped block { ... }"""
    body: List[CudaNode] = field(default_factory=list)

    def emit(self) -> str:
        lines = ["{"]
        for node in self.body:
            for line in node.emit().split('\n'):
                lines.append(f"    {line}")
        lines.append("}")
        return "\n".join(lines)

    def to_builder(self) -> str:
        lines = ['cb.block_begin()']
        for node in self.body:
            lines.append(f"    {node.to_builder()}")
        lines.append('cb.block_end()')
        return "\n".join(lines)


# =============================================================================
# INLINE ASM (connects to PTX AST)
# =============================================================================

@dataclass
class InlineAsm(CudaNode):
    """Inline assembly block - wraps PTX AST"""
    ptx_lines: List[str]  # Raw PTX lines
    outputs: List[Tuple[str, str]] = field(default_factory=list)  # (constraint, var)
    inputs: List[Tuple[str, str]] = field(default_factory=list)
    clobbers: List[str] = field(default_factory=list)
    is_volatile: bool = True
    ptx_ast: Optional[Any] = None  # PTXModule from ptx_ast.py

    def emit(self) -> str:
        # Escape PTX for C string
        ptx_str = "\\n".join(self.ptx_lines)
        ptx_str = ptx_str.replace('"', '\\"')

        parts = [f'asm {"volatile" if self.is_volatile else ""}(\n        "{ptx_str}"']

        if self.outputs:
            parts.append('\n        : ' + ', '.join(f'"{c}"({v})' for c, v in self.outputs))
        elif self.inputs or self.clobbers:
            parts.append('\n        :')

        if self.inputs:
            parts.append('\n        : ' + ', '.join(f'"{c}"({v})' for c, v in self.inputs))
        elif self.clobbers:
            parts.append('\n        :')

        if self.clobbers:
            parts.append('\n        : ' + ', '.join(f'"{c}"' for c in self.clobbers))

        parts.append('\n    )')
        return ''.join(parts)

    def to_builder(self) -> str:
        # If we have a PTX AST, use it for structured output
        if self.ptx_ast:
            lines = ["# --- inline asm block ---"]
            lines.append("b = ASTBuilder()")
            for stmt in self.ptx_ast.statements:
                lines.append(stmt.to_builder())
            lines.append(f"cb.asm(b.build(), outputs={repr(self.outputs)}, inputs={repr(self.inputs)}, clobbers={repr(self.clobbers)})")
            return "\n".join(lines)

        # Fallback to raw PTX
        return f'cb.asm_raw({repr(self.ptx_lines)}, outputs={repr(self.outputs)}, inputs={repr(self.inputs)}, clobbers={repr(self.clobbers)})'


# =============================================================================
# CONSTEXPR AND STATIC ASSERT
# =============================================================================

@dataclass
class StaticAssert(CudaNode):
    """static_assert statement"""
    condition: str
    message: Optional[str] = None

    def emit(self) -> str:
        if self.message:
            return f'static_assert({self.condition}, "{self.message}");'
        return f'static_assert({self.condition});'

    def to_builder(self) -> str:
        if self.message:
            return f'cb.static_assert({repr(self.condition)}, {repr(self.message)})'
        return f'cb.static_assert({repr(self.condition)})'


@dataclass
class Constexpr(CudaNode):
    """constexpr variable/constant"""
    name: str
    type: TypeRef
    value: str
    storage: StorageClass = StorageClass.NONE

    def emit(self) -> str:
        storage = f"{self.storage.value} " if self.storage != StorageClass.NONE else ""
        return f"{storage}constexpr {self.type.emit()} {self.name} = {self.value};"

    def to_builder(self) -> str:
        storage = f', storage="{self.storage.value}"' if self.storage != StorageClass.NONE else ""
        return f'cb.constexpr("{self.name}", {self.type.to_builder()}, {repr(self.value)}{storage})'


# =============================================================================
# LAMBDA EXPRESSIONS
# =============================================================================

@dataclass
class Lambda(CudaNode):
    """Lambda expression"""
    capture: str = ""  # &, =, specific captures
    params: List[Parameter] = field(default_factory=list)
    body: List[CudaNode] = field(default_factory=list)
    return_type: Optional[TypeRef] = None

    def emit(self) -> str:
        parts = [f"[{self.capture}]"]

        if self.params:
            params_str = ", ".join(p.emit() for p in self.params)
            parts.append(f"({params_str})")
        elif self.return_type:
            parts.append("()")

        if self.return_type:
            parts.append(f" -> {self.return_type.emit()}")

        parts.append(" {\n")
        for node in self.body:
            for line in node.emit().split('\n'):
                parts.append(f"    {line}\n")
        parts.append("}")

        return "".join(parts)

    def to_builder(self) -> str:
        lines = [f'cb.lambda_begin({repr(self.capture)})']
        for node in self.body:
            lines.append(f"    {node.to_builder()}")
        lines.append('cb.lambda_end()')
        return "\n".join(lines)


# =============================================================================
# CUDA MODULE (TOP-LEVEL)
# =============================================================================

@dataclass
class CudaModule(CudaNode):
    """Complete CUDA source file"""
    nodes: List[CudaNode] = field(default_factory=list)
    flags: Optional[CompilationFlags] = None

    def emit(self) -> str:
        return "\n".join(node.emit() for node in self.nodes)

    def to_builder(self) -> str:
        lines = [
            "from cuda_ast import *",
            "from builder import CudaBuilder, ASTBuilder, reg, imm, vec, mem, sym",
            "",
            "cb = CudaBuilder()",
        ]
        # Emit flags first if present
        if self.flags:
            flags_cmd = self.flags.to_builder()
            if flags_cmd:
                lines.append(flags_cmd)
        for node in self.nodes:
            node_builder = node.to_builder()
            for line in node_builder.split('\n'):
                lines.append(line)
        lines.append("")
        lines.append("cuda_source = cb.build()")
        return "\n".join(lines)

    def find_functions(self, name: Optional[str] = None) -> List[Function]:
        """Find all functions, optionally filtered by name."""
        funcs = [n for n in self.nodes if isinstance(n, Function)]
        if name:
            funcs = [f for f in funcs if f.name == name]
        return funcs

    def find_kernels(self) -> List[Function]:
        """Find all __global__ kernel functions."""
        return [n for n in self.nodes if isinstance(n, Function)
                and n.qualifier == FunctionQualifier.GLOBAL]

    def find_structs(self, name: Optional[str] = None) -> List[Struct]:
        """Find all struct definitions."""
        structs = [n for n in self.nodes if isinstance(n, Struct)]
        if name:
            structs = [s for s in structs if s.name == name]
        return structs


# =============================================================================
# PARSER
# =============================================================================

class CudaParser:
    """
    Parser for CUDA C++ source code into AST.

    Uses a combination of regex and recursive descent parsing.
    Falls back to RawCode for constructs it can't handle.
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.nodes: List[CudaNode] = []

    def parse(self) -> CudaModule:
        """Parse the entire source into a CudaModule."""
        while self.pos < len(self.source):
            self._skip_whitespace()
            if self.pos >= len(self.source):
                break

            node = self._parse_top_level()
            if node:
                self.nodes.append(node)

        return CudaModule(nodes=self.nodes)

    def _skip_whitespace(self, preserve_newlines: bool = False):
        """Skip whitespace, optionally counting newlines."""
        newline_count = 0
        while self.pos < len(self.source) and self.source[self.pos] in ' \t\n\r':
            if self.source[self.pos] == '\n':
                newline_count += 1
            self.pos += 1

        # Add blank line nodes for multiple newlines
        if preserve_newlines and newline_count > 1:
            return BlankLine(count=newline_count - 1)
        return None

    def _peek(self, n: int = 1) -> str:
        """Peek at next n characters."""
        return self.source[self.pos:self.pos + n]

    def _match(self, pattern: str) -> Optional[re.Match]:
        """Try to match a regex at current position."""
        m = re.match(pattern, self.source[self.pos:], re.MULTILINE)
        return m

    def _consume_match(self, pattern: str) -> Optional[re.Match]:
        """Match and consume a regex."""
        m = self._match(pattern)
        if m:
            self.pos += m.end()
        return m

    def _parse_top_level(self) -> Optional[CudaNode]:
        """Parse a top-level construct."""

        # Skip leading whitespace and track blank lines
        blank = self._skip_whitespace(preserve_newlines=True)
        if blank:
            return blank

        if self.pos >= len(self.source):
            return None

        # Comment
        if self._peek(2) == '//':
            return self._parse_line_comment()
        if self._peek(2) == '/*':
            return self._parse_block_comment()

        # Preprocessor
        if self._peek(1) == '#':
            return self._parse_preprocessor()

        # Template
        if self._match(r'template\s*<'):
            return self._parse_template_decl()

        # Struct/class
        if self._match(r'(struct|class)\s+\w+'):
            return self._parse_struct()

        # Static constexpr variable or constexpr function
        # Check if it's a constexpr variable (has = or []) or function (has ())
        if self._match(r'(?:static\s+)?constexpr\s+\w+\s+\w+\s*(?:\[\s*\])?\s*='):
            # It's a constexpr variable or array
            return self._parse_constexpr()

        # Function (device/global/host/regular/constexpr)
        if self._match(r'(__device__|__global__|__host__|static|inline|__forceinline__|constexpr|[a-zA-Z_]\w*(?:\s*::\s*\w+)*\s+\w+\s*\()'):
            return self._parse_function_or_decl()

        # Macro blocks like TORCH_LIBRARY(name, m) { ... }
        if self._match(r'[A-Z_][A-Z0-9_]*\s*\([^)]*\)\s*\{'):
            return self._parse_macro_block()

        # Using statement at top level
        if self._match(r'using\s+\w+'):
            return self._parse_using()

        # Fallback: read until next top-level construct
        return self._parse_raw_until_toplevel()

    def _parse_line_comment(self) -> Comment:
        """Parse // comment"""
        self.pos += 2  # skip //
        end = self.source.find('\n', self.pos)
        if end == -1:
            end = len(self.source)
        text = self.source[self.pos:end].strip()
        self.pos = end
        return Comment(text=text)

    def _parse_block_comment(self) -> Comment:
        """Parse /* ... */ comment"""
        self.pos += 2  # skip /*
        end = self.source.find('*/', self.pos)
        if end == -1:
            end = len(self.source)
            text = self.source[self.pos:]
        else:
            text = self.source[self.pos:end]
            self.pos = end + 2
        return Comment(text=text, is_multiline=True)

    def _parse_preprocessor(self) -> CudaNode:
        """Parse preprocessor directive."""
        start = self.pos
        self.pos += 1  # skip #

        # Get directive name
        m = self._match(r'\s*(\w+)')
        if not m:
            return self._read_line_as_raw(start)

        directive = m.group(1)
        self.pos += m.end()

        if directive == 'include':
            return self._parse_include()
        elif directive == 'define':
            return self._parse_define()
        elif directive == 'pragma':
            return self._parse_pragma()
        elif directive in ('if', 'ifdef', 'ifndef', 'elif', 'else', 'endif'):
            # Complex - return raw for now
            return self._read_line_as_raw(start)
        else:
            return self._read_line_as_raw(start)

    def _parse_include(self) -> Include:
        """Parse #include directive."""
        self._skip_whitespace()

        if self._peek(1) == '<':
            self.pos += 1
            end = self.source.find('>', self.pos)
            if end == -1:
                end = self.source.find('\n', self.pos)
            path = self.source[self.pos:end]
            self.pos = end + 1
            return Include(path=path, is_system=True)
        elif self._peek(1) == '"':
            self.pos += 1
            end = self.source.find('"', self.pos)
            if end == -1:
                end = self.source.find('\n', self.pos)
            path = self.source[self.pos:end]
            self.pos = end + 1
            return Include(path=path, is_system=False)

        # Malformed
        return Include(path="", is_system=False)

    def _parse_define(self) -> Define:
        """Parse #define directive."""
        self._skip_whitespace()

        # Get name
        m = self._match(r'(\w+)')
        if not m:
            return Define(name="")

        name = m.group(1)
        self.pos += m.end()

        # Check for function-like macro
        params = None
        if self._peek(1) == '(':
            self.pos += 1
            end = self.source.find(')', self.pos)
            if end != -1:
                params_str = self.source[self.pos:end]
                params = [p.strip() for p in params_str.split(',') if p.strip()]
                self.pos = end + 1

        # Get value (rest of line, handling continuation)
        value_parts = []
        while True:
            # Skip spaces but not newline
            while self.pos < len(self.source) and self.source[self.pos] in ' \t':
                self.pos += 1

            # Read until end of line
            end = self.source.find('\n', self.pos)
            if end == -1:
                end = len(self.source)

            line = self.source[self.pos:end]

            # Check for line continuation
            if line.rstrip().endswith('\\'):
                value_parts.append(line.rstrip()[:-1])
                self.pos = end + 1
            else:
                value_parts.append(line)
                self.pos = end
                break

        value = ' '.join(value_parts).strip() or None
        return Define(name=name, value=value, params=params)

    def _parse_pragma(self) -> Pragma:
        """Parse #pragma directive."""
        end = self.source.find('\n', self.pos)
        if end == -1:
            end = len(self.source)
        content = self.source[self.pos:end].strip()
        self.pos = end
        return Pragma(content=content)

    def _parse_struct(self) -> Struct:
        """Parse struct/class definition."""
        start = self.pos

        # Match struct/class name
        m = self._match(r'(struct|class)\s+(\w+)\s*\{')
        if not m:
            return self._read_until_semicolon_as_raw(start)

        is_class = m.group(1) == 'class'
        name = m.group(2)
        self.pos += m.end()

        members = []

        # Parse members until }
        while self.pos < len(self.source):
            self._skip_whitespace()

            if self._peek(1) == '}':
                self.pos += 1
                break

            # Using declaration
            if self._match(r'using\s+\w+\s*='):
                members.append(self._parse_using())
                continue

            # Try to parse as field(s) - may return multiple for "int a, b, c;"
            fields = self._parse_struct_field()
            if fields:
                members.extend(fields)
            else:
                # Skip line
                end = self.source.find('\n', self.pos)
                if end == -1:
                    break
                members.append(RawCode(text=self.source[self.pos:end].strip()))
                self.pos = end

        # Skip trailing ;
        self._skip_whitespace()
        if self._peek(1) == ';':
            self.pos += 1

        return Struct(name=name, members=members, is_class=is_class)

    def _parse_struct_field(self) -> Optional[List[StructField]]:
        """Parse struct field declaration(s). Returns list for multi-var decls like 'int a, b, c;'"""
        start = self.pos

        # Match: type name1[, name2, ...] [= value];
        # First try multi-variable: type name, name, name;
        m = self._match(r'\s*([a-zA-Z_][\w:*&<>\s]*?)\s+(\w+(?:\s*,\s*\w+)*)\s*;')
        if m:
            type_str = m.group(1).strip()
            names_str = m.group(2)
            names = [n.strip() for n in names_str.split(',')]
            self.pos += m.end()
            return [StructField(name=n, type=TypeRef(name=type_str)) for n in names]

        # Try single variable with optional initializer: type name = value;
        m = self._match(r'\s*([a-zA-Z_][\w:*&<>,\s]*?)\s+(\w+)\s*(=\s*[^;]+)?;')
        if not m:
            return None

        type_str = m.group(1).strip()
        name = m.group(2)
        init = m.group(3)
        if init:
            init = init.strip()[1:].strip()  # Remove '='

        self.pos += m.end()

        return [StructField(
            name=name,
            type=TypeRef(name=type_str),
            initializer=init
        )]

    def _parse_using(self) -> UsingDecl:
        """Parse using declaration."""
        m = self._consume_match(r'using\s+(\w+)\s*=\s*([^;]+);')
        if m:
            return UsingDecl(name=m.group(1), type_expr=m.group(2).strip())

        # Malformed
        return UsingDecl(name="", type_expr="")

    def _parse_constexpr(self) -> Constexpr:
        """Parse [static] constexpr declaration, including arrays."""
        start = self.pos

        # Parse storage and constexpr keyword
        storage = StorageClass.NONE
        if self._match(r'static\s+'):
            self._consume_match(r'static\s+')
            storage = StorageClass.STATIC
        self._consume_match(r'constexpr\s+')

        # Parse type name
        m = self._match(r'(\w+)\s+(\w+)\s*(\[\s*\])?\s*=')
        if not m:
            return self._read_until_semicolon_as_raw(start)

        type_str = m.group(1)
        name = m.group(2)
        is_array = bool(m.group(3))
        self.pos += m.end()

        # Parse value - could be simple value or { ... } initializer
        self._skip_whitespace()
        if self._peek(1) == '{':
            # Array initializer - read until matching }
            value = self._parse_balanced_braces()
        else:
            # Simple value - read until ;
            end = self.source.find(';', self.pos)
            if end == -1:
                end = len(self.source)
            value = self.source[self.pos:end].strip()
            self.pos = end

        # Skip semicolon
        if self.pos < len(self.source) and self.source[self.pos] == ';':
            self.pos += 1

        return Constexpr(
            name=name,
            type=TypeRef(name=type_str + ("[]" if is_array else "")),
            value=value,
            storage=storage
        )

    def _parse_macro_block(self) -> Function:
        """Parse macro blocks like TORCH_LIBRARY(name, m) { ... }"""
        start = self.pos

        # Match: MACRO_NAME(args) {
        m = self._consume_match(r'([A-Z_][A-Z0-9_]*)\s*\(([^)]*)\)\s*\{')
        if not m:
            return self._read_until_brace_or_semicolon_as_raw(start)

        macro_name = m.group(1)
        args = m.group(2).strip()

        # Parse the body
        body = self._parse_function_body()

        # Return as a function-like node
        return Function(
            name=macro_name,
            return_type=TypeRef(name=""),
            params=[Parameter(name=args, type=TypeRef(name=""))],
            body=body,
            qualifier=FunctionQualifier.NONE
        )

    def _parse_template_decl(self) -> CudaNode:
        """Parse template<...> followed by struct/class/function."""
        start = self.pos

        # Get template params
        m = self._consume_match(r'template\s*<([^>]*)>\s*')
        if not m:
            return self._read_line_as_raw(start)

        template_params = [p.strip() for p in m.group(1).split(',')]

        # What follows?
        if self._match(r'(struct|class)\s+'):
            struct = self._parse_struct()
            struct.template_params = template_params
            return struct
        else:
            # Function
            func = self._parse_function_or_decl()
            if isinstance(func, Function):
                func.template_params = template_params
            return func

    def _parse_function_or_decl(self) -> CudaNode:
        """Parse function definition or declaration."""
        start = self.pos

        # Parse qualifiers
        qualifier = FunctionQualifier.NONE
        is_inline = False
        is_forceinline = False
        storage = StorageClass.NONE
        launch_bounds = None

        while True:
            self._skip_whitespace()

            if self._match(r'__global__\s*'):
                m = self._consume_match(r'__global__\s*')
                qualifier = FunctionQualifier.GLOBAL
            elif self._match(r'__device__\s*'):
                m = self._consume_match(r'__device__\s*')
                qualifier = FunctionQualifier.DEVICE
            elif self._match(r'__host__\s*'):
                m = self._consume_match(r'__host__\s*')
                if qualifier == FunctionQualifier.DEVICE:
                    qualifier = FunctionQualifier.HOST_DEVICE
                else:
                    qualifier = FunctionQualifier.HOST
            elif self._match(r'__forceinline__\s*'):
                self._consume_match(r'__forceinline__\s*')
                is_forceinline = True
            elif self._match(r'inline\s+'):
                self._consume_match(r'inline\s+')
                is_inline = True
            elif self._match(r'static\s+'):
                self._consume_match(r'static\s+')
                storage = StorageClass.STATIC
            elif self._match(r'constexpr\s+'):
                self._consume_match(r'constexpr\s+')
                storage = StorageClass.CONSTEXPR
            elif self._match(r'__launch_bounds__\s*\('):
                lb = self._parse_launch_bounds()
                if lb:
                    launch_bounds = lb
            else:
                break

        # Parse return type - might be followed by __launch_bounds__ before function name
        # Pattern: return_type [__launch_bounds__(...)] func_name(params)
        return_type_str = None
        func_name = None

        # First, try simple: type name( - type can be namespaced (foo::bar)
        m = self._match(r'([\w:]+)\s+(\w+)\s*\(')
        if m and m.group(2) != '__launch_bounds__':
            return_type_str = m.group(1)
            func_name = m.group(2)
            self.pos += m.end()
        else:
            # Try: type __launch_bounds__(...) name(
            # Use non-greedy match for content inside parens
            m = self._match(r'([\w:]+)\s+__launch_bounds__\s*\(')
            if m:
                return_type_str = m.group(1)
                self.pos += m.end()
                # Parse launch bounds args
                launch_bounds = self._parse_launch_bounds_args()
                self._skip_whitespace()
                # Now match function name
                m2 = self._match(r'(\w+)\s*\(')
                if m2:
                    func_name = m2.group(1)
                    self.pos += m2.end()

        if not return_type_str or not func_name:
            return self._read_until_brace_or_semicolon_as_raw(start)

        # Parse parameters
        params = self._parse_params()

        self._skip_whitespace()

        # Is it a declaration (;) or definition ({)?
        if self._peek(1) == ';':
            self.pos += 1
            # Check if this looks like a constructor-call variable declaration
            # e.g., "dim3 grid(params.m / 16, 1, params.b);"
            # vs a real function declaration "void func(int x, float y);"
            # Heuristic: if no qualifier and params look like expressions (have operators/dots),
            # it's likely a variable declaration
            if qualifier == FunctionQualifier.NONE and not is_inline and not is_forceinline:
                is_var_decl = False
                if not params:
                    is_var_decl = False  # empty parens could be default constructor
                else:
                    # Check if any param looks like an expression rather than "type name"
                    for p in params:
                        # If param type contains operators, dots, or is just a number, it's an expression
                        type_name = p.type.name if p.type else ""
                        if any(c in type_name for c in '.+-*/') or type_name.isdigit():
                            is_var_decl = True
                            break
                        # If name looks like a number or expression
                        if p.name.isdigit() or any(c in p.name for c in '.+-*/'):
                            is_var_decl = True
                            break

                if is_var_decl:
                    # Reconstruct as statement: "type name(args);"
                    # params have been parsed as "type name" pairs but they're really just expressions
                    # Reconstruct by combining type and name back into expressions
                    args = []
                    for p in params:
                        if p.type and p.type.name and p.type.name != "void":
                            # Type and name were split, recombine
                            args.append(f"{p.type.name} {p.name}".strip())
                        else:
                            args.append(p.name)
                    stmt_text = f"{return_type_str} {func_name}({', '.join(args)})"
                    return Statement(expr=stmt_text)

            return FunctionDecl(
                name=func_name,
                return_type=TypeRef(name=return_type_str),
                params=params,
                qualifier=qualifier,
                is_inline=is_inline,
                is_forceinline=is_forceinline,
                storage=storage
            )

        if self._peek(1) != '{':
            return self._read_until_brace_or_semicolon_as_raw(start)

        self.pos += 1  # skip {

        # Parse body
        body = self._parse_function_body()

        return Function(
            name=func_name,
            return_type=TypeRef(name=return_type_str),
            params=params,
            body=body,
            qualifier=qualifier,
            is_inline=is_inline,
            is_forceinline=is_forceinline,
            storage=storage,
            launch_bounds=launch_bounds
        )

    def _parse_launch_bounds(self) -> Optional[LaunchBounds]:
        """Parse __launch_bounds__(max, min) - supports expressions"""
        m = self._consume_match(r'__launch_bounds__\s*\(')
        if not m:
            return None
        return self._parse_launch_bounds_args()

    def _parse_launch_bounds_args(self) -> Optional[LaunchBounds]:
        """Parse the arguments inside __launch_bounds__(...) - called after '(' is consumed"""
        # Parse arguments (expressions until , or ))
        args = []
        current_arg = ""
        depth = 1

        while self.pos < len(self.source) and depth > 0:
            c = self.source[self.pos]
            if c == '(':
                depth += 1
                current_arg += c
            elif c == ')':
                depth -= 1
                if depth == 0:
                    if current_arg.strip():
                        args.append(current_arg.strip())
                    self.pos += 1
                    break
                current_arg += c
            elif c == ',' and depth == 1:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += c
            self.pos += 1

        self._skip_whitespace()

        if not args:
            return None

        # Try to convert to int if possible, otherwise keep as string expression
        def parse_arg(s):
            try:
                return int(s)
            except ValueError:
                return s

        max_threads = parse_arg(args[0])
        min_blocks = parse_arg(args[1]) if len(args) > 1 else None

        return LaunchBounds(max_threads=max_threads, min_blocks=min_blocks)

    def _parse_params(self) -> List[Parameter]:
        """Parse function parameters."""
        params = []
        depth = 1  # Already past opening (
        param_start = self.pos

        while self.pos < len(self.source) and depth > 0:
            c = self.source[self.pos]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    # End of params
                    param_text = self.source[param_start:self.pos].strip()
                    if param_text:
                        params.extend(self._parse_param_list(param_text))
                    self.pos += 1
                    break
            elif c == ',' and depth == 1:
                param_text = self.source[param_start:self.pos].strip()
                if param_text:
                    params.extend(self._parse_param_list(param_text))
                self.pos += 1
                param_start = self.pos
                continue
            self.pos += 1

        return params

    def _parse_param_list(self, text: str) -> List[Parameter]:
        """Parse a single parameter or comma-separated list."""
        params = []

        # Simple pattern: [const] [__grid_constant__] type name [= default]
        # This handles most cases
        parts = text.split(',')
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check for default value
            default = None
            if '=' in part:
                part, default = part.rsplit('=', 1)
                part = part.strip()
                default = default.strip()

            # Extract name (last word)
            tokens = part.split()
            if tokens:
                name = tokens[-1]
                type_str = ' '.join(tokens[:-1]) if len(tokens) > 1 else 'void'
                params.append(Parameter(
                    name=name,
                    type=TypeRef(name=type_str),
                    default=default
                ))

        return params

    def _parse_function_body(self) -> List[CudaNode]:
        """Parse function body until matching }"""
        body = []
        depth = 1

        while self.pos < len(self.source) and depth > 0:
            self._skip_whitespace()

            if self.pos >= len(self.source):
                break

            c = self.source[self.pos]

            if c == '}':
                depth -= 1
                if depth == 0:
                    self.pos += 1
                    break
                # Nested block end
                body.append(RawCode(text='}'))
                self.pos += 1
                continue

            if c == '{':
                # Standalone scoped block - _parse_block_body handles depth internally
                self.pos += 1
                inner_body = self._parse_block_body()
                body.append(Block(body=inner_body))
                continue

            # Comment
            if self._peek(2) == '//':
                body.append(self._parse_line_comment())
                continue
            if self._peek(2) == '/*':
                body.append(self._parse_block_comment())
                continue

            # Inline asm - various forms: asm volatile(...), asm(...), asm __volatile__(...)
            if self._match(r'asm\s*(?:volatile|__volatile__)?\s*\('):
                body.append(self._parse_inline_asm())
                continue

            # Control flow - check if constexpr first
            if self._match(r'if\s+constexpr\s*\(') or self._match(r'if\s*\('):
                body.append(self._parse_if())
                continue
            if self._match(r'for\s*\('):
                body.append(self._parse_for())
                continue
            if self._match(r'while\s*\('):
                body.append(self._parse_while())
                continue

            # Return
            if self._match(r'return\s*'):
                body.append(self._parse_return())
                continue

            # #pragma
            if self._peek(1) == '#':
                # Check if it's #pragma unroll followed by for
                if self._match(r'#\s*pragma\s+unroll'):
                    pragma = self._parse_pragma_unroll()
                    if pragma:
                        body.append(pragma)
                        continue
                # Other pragmas
                body.append(self._parse_preprocessor())
                continue

            # Statement (read until semicolon or brace)
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)

        return body

    def _parse_inline_asm(self) -> InlineAsm:
        """Parse asm volatile(...) block."""
        start = self.pos

        m = self._consume_match(r'asm\s*(?:volatile|__volatile__)?\s*\(')
        if not m:
            return InlineAsm(ptx_lines=[])

        # Collect PTX string parts
        ptx_parts = []

        while self.pos < len(self.source):
            self._skip_whitespace()

            # String literal
            if self._peek(1) == '"':
                self.pos += 1
                string_end = self.pos
                while string_end < len(self.source):
                    if self.source[string_end] == '\\' and string_end + 1 < len(self.source):
                        string_end += 2
                    elif self.source[string_end] == '"':
                        break
                    else:
                        string_end += 1
                ptx_parts.append(self.source[self.pos:string_end])
                self.pos = string_end + 1
                continue

            # End of PTX string part, check for : or )
            if self._peek(1) in ':)':
                break

            self.pos += 1

        # Combine and normalize PTX
        ptx_raw = "".join(ptx_parts)
        ptx_lines = ptx_raw.replace('\\n', '\n').replace('\\t', '\t').split('\n')
        ptx_lines = [l.strip() for l in ptx_lines if l.strip()]

        # Parse output/input/clobber sections
        outputs = []
        inputs = []
        clobbers = []
        section = 0

        while self.pos < len(self.source) and self.source[self.pos] != ')':
            if self.source[self.pos] == ':':
                section += 1
                self.pos += 1

                # Find section end
                section_start = self.pos
                paren_depth = 0
                while self.pos < len(self.source):
                    c = self.source[self.pos]
                    if c == '(':
                        paren_depth += 1
                    elif c == ')':
                        if paren_depth == 0:
                            break
                        paren_depth -= 1
                    elif c == ':' and paren_depth == 0:
                        break
                    self.pos += 1

                content = self.source[section_start:self.pos].strip()

                if section == 1:
                    outputs = re.findall(r'"([^"]+)"\s*\(\s*([^)]+)\s*\)', content)
                elif section == 2:
                    inputs = re.findall(r'"([^"]+)"\s*\(\s*([^)]+)\s*\)', content)
                elif section == 3:
                    clobbers = re.findall(r'"([^"]+)"', content)
                continue

            self.pos += 1

        # Skip closing paren and semicolon
        if self.pos < len(self.source) and self.source[self.pos] == ')':
            self.pos += 1
        self._skip_whitespace()
        if self.pos < len(self.source) and self.source[self.pos] == ';':
            self.pos += 1

        return InlineAsm(
            ptx_lines=ptx_lines,
            outputs=list(outputs),
            inputs=list(inputs),
            clobbers=clobbers
        )

    def _parse_if(self) -> If:
        """Parse if statement."""
        # Check for constexpr
        is_constexpr = bool(self._match(r'if\s+constexpr\s*\('))

        m = self._consume_match(r'if\s*(?:constexpr\s*)?\(')
        if not m:
            return If(condition="")

        # Parse condition
        condition = self._parse_balanced_parens()

        self._skip_whitespace()

        # Parse then body
        then_body = []
        if self._peek(1) == '{':
            self.pos += 1
            then_body = self._parse_block_body()
        else:
            # Single statement
            stmt = self._parse_statement()
            if stmt:
                then_body = [stmt]

        # Check for else
        self._skip_whitespace()
        else_body = None
        if self._match(r'else\s*'):
            self._consume_match(r'else\s*')
            # Check for else if (including else if constexpr)
            if self._match(r'if\s+constexpr\s*\(') or self._match(r'if\s*\('):
                # else if / else if constexpr
                else_if = self._parse_if()
                else_body = [else_if]
            elif self._peek(1) == '{':
                self.pos += 1
                else_body = self._parse_block_body()
            else:
                stmt = self._parse_statement()
                if stmt:
                    else_body = [stmt]

        return If(
            condition=condition,
            then_body=then_body,
            else_body=else_body,
            is_constexpr=is_constexpr
        )

    def _parse_for(self) -> For:
        """Parse for loop."""
        m = self._consume_match(r'for\s*\(')
        if not m:
            return For(init="", condition="", increment="")

        # Parse init; condition; increment
        parts = []
        depth = 1
        current = ""

        while self.pos < len(self.source) and depth > 0:
            c = self.source[self.pos]
            if c == '(':
                depth += 1
                current += c
            elif c == ')':
                depth -= 1
                if depth == 0:
                    parts.append(current.strip())
                    self.pos += 1
                    break
                current += c
            elif c == ';' and depth == 1:
                parts.append(current.strip())
                current = ""
            else:
                current += c
            self.pos += 1

        init = parts[0] if len(parts) > 0 else ""
        condition = parts[1] if len(parts) > 1 else ""
        increment = parts[2] if len(parts) > 2 else ""

        self._skip_whitespace()

        # Parse body
        body = []
        if self._peek(1) == '{':
            self.pos += 1
            body = self._parse_block_body()
        else:
            stmt = self._parse_statement()
            if stmt:
                body = [stmt]

        return For(init=init, condition=condition, increment=increment, body=body)

    def _parse_while(self) -> While:
        """Parse while loop."""
        m = self._consume_match(r'while\s*\(')
        if not m:
            return While(condition="")

        condition = self._parse_balanced_parens()

        self._skip_whitespace()

        body = []
        if self._peek(1) == '{':
            self.pos += 1
            body = self._parse_block_body()
        else:
            stmt = self._parse_statement()
            if stmt:
                body = [stmt]

        return While(condition=condition, body=body)

    def _parse_return(self) -> Return:
        """Parse return statement."""
        m = self._consume_match(r'return\s*')
        if not m:
            return Return()

        if self._peek(1) == ';':
            self.pos += 1
            return Return()

        # Read until semicolon
        end = self.source.find(';', self.pos)
        if end == -1:
            end = len(self.source)
        value = self.source[self.pos:end].strip()
        self.pos = end + 1

        return Return(value=value)

    def _parse_pragma_unroll(self) -> Optional[For]:
        """Parse #pragma unroll [N] followed by for loop."""
        m = self._consume_match(r'#\s*pragma\s+unroll\s*(\d*)\s*\n?')
        if not m:
            return None

        unroll_count = int(m.group(1)) if m.group(1) else -1

        self._skip_whitespace()

        if self._match(r'for\s*\('):
            for_loop = self._parse_for()
            for_loop.pragma_unroll = unroll_count
            return for_loop

        return None

    def _parse_block_body(self) -> List[CudaNode]:
        """Parse { ... } block body."""
        body = []
        depth = 1

        while self.pos < len(self.source) and depth > 0:
            self._skip_whitespace()

            if self.pos >= len(self.source):
                break

            c = self.source[self.pos]

            if c == '}':
                depth -= 1
                if depth == 0:
                    self.pos += 1
                    break
                self.pos += 1
                continue

            if c == '{':
                depth += 1
                self.pos += 1
                continue

            # Parse statement or nested construct
            # Inline asm
            if self._match(r'asm\s*(?:volatile|__volatile__)?\s*\('):
                body.append(self._parse_inline_asm())
            # Check if constexpr first
            elif self._match(r'if\s+constexpr\s*\(') or self._match(r'if\s*\('):
                body.append(self._parse_if())
            elif self._match(r'for\s*\('):
                body.append(self._parse_for())
            elif self._match(r'while\s*\('):
                body.append(self._parse_while())
            elif self._match(r'return\s*'):
                body.append(self._parse_return())
            elif self._peek(2) == '//':
                body.append(self._parse_line_comment())
            elif self._peek(1) == '#':
                if self._match(r'#\s*pragma\s+unroll'):
                    pragma = self._parse_pragma_unroll()
                    if pragma:
                        body.append(pragma)
                else:
                    body.append(self._parse_preprocessor())
            else:
                stmt = self._parse_statement()
                if stmt:
                    body.append(stmt)

        return body

    def _parse_statement(self) -> Optional[CudaNode]:
        """Parse a single statement."""
        start = self.pos

        # Read until semicolon, considering nested braces/parens
        depth_brace = 0
        depth_paren = 0

        while self.pos < len(self.source):
            c = self.source[self.pos]

            if c == '{':
                depth_brace += 1
            elif c == '}':
                if depth_brace == 0:
                    break
                depth_brace -= 1
            elif c == '(':
                depth_paren += 1
            elif c == ')':
                depth_paren -= 1
            elif c == ';' and depth_brace == 0 and depth_paren == 0:
                text = self.source[start:self.pos].strip()
                self.pos += 1
                if text:
                    return Statement(expr=text)
                return None

            self.pos += 1

        # Didn't find semicolon
        if self.pos > start:
            text = self.source[start:self.pos].strip()
            if text:
                return RawCode(text=text)
        return None

    def _parse_balanced_parens(self) -> str:
        """Parse content until balanced parentheses."""
        depth = 1
        start = self.pos

        while self.pos < len(self.source) and depth > 0:
            c = self.source[self.pos]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    result = self.source[start:self.pos]
                    self.pos += 1
                    return result
            self.pos += 1

        return self.source[start:self.pos]

    def _parse_balanced_braces(self) -> str:
        """Parse content inside { ... } including the braces."""
        if self._peek(1) != '{':
            return ""

        start = self.pos
        self.pos += 1  # skip opening {
        depth = 1

        while self.pos < len(self.source) and depth > 0:
            c = self.source[self.pos]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            self.pos += 1

        return self.source[start:self.pos]

    def _read_line_as_raw(self, start: int) -> RawCode:
        """Read rest of line as raw code."""
        end = self.source.find('\n', self.pos)
        if end == -1:
            end = len(self.source)
        text = self.source[start:end].strip()
        self.pos = end
        return RawCode(text=text)

    def _read_until_semicolon_as_raw(self, start: int) -> RawCode:
        """Read until semicolon as raw code."""
        end = self.source.find(';', self.pos)
        if end == -1:
            end = len(self.source)
        else:
            end += 1
        text = self.source[start:end].strip()
        self.pos = end
        return RawCode(text=text)

    def _read_until_brace_or_semicolon_as_raw(self, start: int) -> RawCode:
        """Read until { or ; as raw code."""
        brace = self.source.find('{', self.pos)
        semi = self.source.find(';', self.pos)

        if brace == -1 and semi == -1:
            end = len(self.source)
        elif brace == -1:
            end = semi + 1
        elif semi == -1:
            end = brace
        else:
            end = min(brace, semi + 1) if semi < brace else brace

        text = self.source[start:end].strip()
        self.pos = end
        return RawCode(text=text)

    def _read_until_toplevel(self) -> RawCode:
        """Read raw until next top-level construct."""
        start = self.pos

        while self.pos < len(self.source):
            # Check for start of new top-level
            self._skip_whitespace()

            if self._peek(1) == '#':
                break
            if self._match(r'(struct|class|template|__device__|__global__|__host__|static|inline)\s'):
                break
            if self._match(r'[a-zA-Z_]\w*\s+\w+\s*\('):
                break

            # Skip to next line
            end = self.source.find('\n', self.pos)
            if end == -1:
                self.pos = len(self.source)
                break
            self.pos = end + 1

        text = self.source[start:self.pos].strip()
        return RawCode(text=text) if text else None

    def _parse_raw_until_toplevel(self) -> Optional[RawCode]:
        """Parse raw code until next recognizable top-level."""
        return self._read_until_toplevel()


# =============================================================================
# PUBLIC API
# =============================================================================

def parse_cuda(source: str) -> CudaModule:
    """Parse CUDA source into AST."""
    parser = CudaParser(source)
    return parser.parse()


def emit_cuda(ast: CudaModule) -> str:
    """Emit CUDA source from AST."""
    return ast.emit()


if __name__ == "__main__":
    # Test with a simple example
    test_source = '''
#include <cuda.h>
#include "myheader.h"

struct Params {
    int x;
    float y;
};

__device__ __forceinline__ void helper(int a, int b) {
    int c = a + b;
    return;
}

__global__ void __launch_bounds__(128, 8) kernel(Params params) {
    int tid = threadIdx.x;

    if (tid < 32) {
        // Do something
        helper(tid, 1);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        params.x += i;
    }
}
'''

    ast = parse_cuda(test_source)
    print("=== Parsed AST ===")
    for node in ast.nodes:
        print(f"  {type(node).__name__}: {node.emit()[:60]}...")

    print("\n=== Reconstructed ===")
    print(ast.emit())

    print("\n=== Builder Commands ===")
    print(ast.to_builder()[:1000] + "...")
