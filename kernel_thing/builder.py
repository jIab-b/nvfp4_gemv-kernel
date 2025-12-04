"""
PTX AST Builder

Provides a fluent API for constructing PTX AST nodes programmatically.
This is the inverse of to_builder() - executing builder commands reconstructs the AST.

Usage:
    from builder import ASTBuilder, reg, imm, vec, mem, sym

    b = ASTBuilder()
    b.reg("b8", ["a0", "a1", "a2", "a3"])
    b.instr("mov", ["b32"], [vec(reg("a0"), reg("a1")), reg("%0")])
    b.instr("st", ["global", "b32"], [mem(reg("%rd0")), reg("a0")])
    ast = b.build()

    print(ast.emit())
"""

from typing import List, Optional, Union, Tuple, Any
from ptx_ast import (
    PTXModule, Instruction, Directive, Label, Comment, RawLine, Line, Block,
    RegisterDecl, SharedDecl, RegisterOp, ImmediateOp, VectorOp, MemoryOp, SymbolOp,
    Operand, Statement, CudaSource, CudaCode, InlineAsmBlock, escape_ptx, emit_ptx
)
from cuda_ast import (
    CudaModule, CudaNode, Include, Define, Pragma, Comment as CudaComment,
    RawCode, BlankLine, TypeRef, Variable, Parameter, StructField, UsingDecl,
    Struct, LaunchBounds, FunctionDecl, Function, Statement as CudaStatement,
    VarDecl, Return, If, For, While, InlineAsm, StaticAssert, Constexpr, Lambda,
    FunctionQualifier, StorageClass, CompilationFlags
)


# =============================================================================
# OPERAND HELPER FUNCTIONS
# =============================================================================

def reg(name: str) -> RegisterOp:
    """Create a register operand."""
    return RegisterOp(name=name)


def imm(value: str) -> ImmediateOp:
    """Create an immediate operand."""
    return ImmediateOp(value=str(value))


def vec(*elements: Operand) -> VectorOp:
    """Create a vector operand: {a0, a1, a2, a3}"""
    return VectorOp(elements=list(elements))


def mem(base: Operand, offset: Optional[Operand] = None, offset_op: str = "+") -> MemoryOp:
    """Create a memory operand: [%rd0] or [%rd0+16]"""
    return MemoryOp(base=base, offset=offset, offset_op=offset_op)


def sym(name: str) -> SymbolOp:
    """Create a symbol operand."""
    return SymbolOp(name=name)


# =============================================================================
# AST BUILDER
# =============================================================================

class ASTBuilder:
    """Fluent builder for constructing PTX AST."""

    def __init__(self):
        self._statements: List[Statement] = []
        self._block_stack: List[List[Statement]] = []

    def instr(self, mnemonic: str, modifiers: List[str], operands: List[Operand],
              pred: Optional[str] = None) -> "ASTBuilder":
        """Add an instruction."""
        instr = Instruction(
            mnemonic=mnemonic,
            modifiers=modifiers,
            operands=operands,
            predicate=pred
        )
        self._current_list().append(instr)
        return self

    def directive(self, text: str) -> "ASTBuilder":
        """Add a raw directive."""
        self._current_list().append(Directive(text=text))
        return self

    def label(self, name: str) -> "ASTBuilder":
        """Add a label."""
        self._current_list().append(Label(name=name))
        return self

    def comment(self, text: str) -> "ASTBuilder":
        """Add a comment."""
        self._current_list().append(Comment(text=text))
        return self

    def raw(self, text: str) -> "ASTBuilder":
        """Add a raw unparsed line."""
        self._current_list().append(RawLine(text=text))
        return self

    def line(self, statements: List[Union[Instruction, Directive]],
             separator: str = " ") -> "ASTBuilder":
        """Add multiple statements on one line."""
        self._current_list().append(Line(statements=statements, separator=separator))
        return self

    def reg(self, dtype: str, names: List[str]) -> "ASTBuilder":
        """Add a register declaration: .reg .dtype name1, name2, ...;"""
        self._current_list().append(RegisterDecl(dtype=dtype, names=names))
        return self

    def shared(self, dtype: str, name: str, size: int,
               align: Optional[int] = None) -> "ASTBuilder":
        """Add shared memory declaration: .shared .align N .dtype name[size];"""
        self._current_list().append(SharedDecl(dtype=dtype, name=name, size=size, align=align))
        return self

    def block_open(self) -> "ASTBuilder":
        """Start a new block scope."""
        self._block_stack.append([])
        return self

    def block_close(self) -> "ASTBuilder":
        """Close current block scope."""
        if not self._block_stack:
            raise ValueError("No block to close")
        block_statements = self._block_stack.pop()
        self._current_list().append(Block(statements=block_statements))
        return self

    def _current_list(self) -> List[Statement]:
        """Get the current statement list (either top-level or inside a block)."""
        if self._block_stack:
            return self._block_stack[-1]
        return self._statements

    def build(self) -> PTXModule:
        """Build and return the PTX AST module."""
        if self._block_stack:
            raise ValueError(f"Unclosed blocks: {len(self._block_stack)}")
        return PTXModule(statements=self._statements)

    def clear(self) -> "ASTBuilder":
        """Clear all statements and start fresh."""
        self._statements = []
        self._block_stack = []
        return self


# =============================================================================
# CONVENIENCE: INSTRUCTION BUILDER
# =============================================================================

class InstrBuilder:
    """Helper for building a single instruction with method chaining."""

    def __init__(self, mnemonic: str):
        self._mnemonic = mnemonic
        self._modifiers: List[str] = []
        self._operands: List[Operand] = []
        self._predicate: Optional[str] = None

    def mod(self, *mods: str) -> "InstrBuilder":
        """Add modifiers."""
        self._modifiers.extend(mods)
        return self

    def ops(self, *operands: Operand) -> "InstrBuilder":
        """Add operands."""
        self._operands.extend(operands)
        return self

    def pred(self, predicate: str) -> "InstrBuilder":
        """Set predicate."""
        self._predicate = predicate
        return self

    def build(self) -> Instruction:
        """Build the instruction."""
        return Instruction(
            mnemonic=self._mnemonic,
            modifiers=self._modifiers,
            operands=self._operands,
            predicate=self._predicate
        )


def instr(mnemonic: str) -> InstrBuilder:
    """Start building an instruction with method chaining."""
    return InstrBuilder(mnemonic)


# =============================================================================
# CUDA BUILDER (LEGACY - for backward compatibility)
# =============================================================================

class CudaBuilder:
    """
    Legacy builder for constructing CUDA source files.
    Interleaves C++/CUDA code with inline PTX asm blocks.

    For new code, prefer StructuredCudaBuilder which creates proper AST nodes.
    """

    def __init__(self):
        self._segments: List[Union[CudaCode, InlineAsmBlock]] = []

    def cuda(self, code: str) -> "CudaBuilder":
        """Add a C++/CUDA code segment."""
        self._segments.append(CudaCode(text=code))
        return self

    def asm(self, ast: PTXModule,
            outputs: List[Tuple[str, str]] = None,
            inputs: List[Tuple[str, str]] = None,
            clobbers: List[str] = None) -> "CudaBuilder":
        """Add an inline asm block from a PTX AST."""
        block = InlineAsmBlock(
            ptx_raw="",  # Not used for builder-constructed blocks
            ptx_normalized=ast.emit(),
            outputs=outputs or [],
            inputs=inputs or [],
            clobbers=clobbers or [],
            start_pos=0,  # Not relevant for builder
            end_pos=0,
            ast=ast
        )
        self._segments.append(block)
        return self

    def build(self) -> str:
        """Build and return the complete CUDA source as a string."""
        return CudaSource(segments=self._segments).emit()

    def build_source(self) -> CudaSource:
        """Build and return the CudaSource object."""
        return CudaSource(segments=self._segments)


# =============================================================================
# STRUCTURED CUDA BUILDER
# =============================================================================

class StructuredCudaBuilder:
    """
    Structured builder for constructing CUDA AST programmatically.

    Creates proper AST nodes instead of raw text blobs, enabling:
    - Structured manipulation of CUDA constructs
    - Semantic builder commands in output
    - Modification of decision points (cache hints, unroll factors, etc.)

    Usage:
        cb = StructuredCudaBuilder()
        cb.include("cuda.h", system=True)
        cb.include("myheader.h")

        cb.struct_begin("Params")
        cb.field("x", "int")
        cb.field("y", "float")
        cb.struct_end()

        cb.func_begin("kernel", "void", qualifier="__global__",
                      params=[("Params", "p")], launch_bounds=(128, 8))
        cb.stmt("int tid = threadIdx.x")
        cb.func_end()

        ast = cb.build()
        print(ast.emit())
    """

    def __init__(self):
        self._nodes: List[CudaNode] = []
        self._stack: List[Tuple[str, List[CudaNode]]] = []  # (context_type, nodes)
        self._flags: Optional[CompilationFlags] = None

    def _current_list(self) -> List[CudaNode]:
        """Get current node list (top-level or inside a construct)."""
        if self._stack:
            return self._stack[-1][1]
        return self._nodes

    # --- Preprocessor ---

    def include(self, path: str, system: bool = False) -> "StructuredCudaBuilder":
        """Add #include directive."""
        self._current_list().append(Include(path=path, is_system=system))
        return self

    def define(self, name: str, value: Optional[str] = None,
               params: Optional[List[str]] = None) -> "StructuredCudaBuilder":
        """Add #define directive."""
        self._current_list().append(Define(name=name, value=value, params=params))
        return self

    def pragma(self, content: str) -> "StructuredCudaBuilder":
        """Add #pragma directive."""
        self._current_list().append(Pragma(content=content))
        return self

    # --- Comments and Raw ---

    def comment(self, text: str, multiline: bool = False) -> "StructuredCudaBuilder":
        """Add a comment."""
        self._current_list().append(CudaComment(text=text, is_multiline=multiline))
        return self

    def raw(self, text: str) -> "StructuredCudaBuilder":
        """Add raw code (fallback for unparseable constructs)."""
        self._current_list().append(RawCode(text=text))
        return self

    def blank(self, count: int = 1) -> "StructuredCudaBuilder":
        """Add blank line(s)."""
        self._current_list().append(BlankLine(count=count))
        return self

    # --- Compilation Flags ---

    def flags(self,
              maxrregcount: Optional[int] = None,
              arch: Optional[str] = None,
              use_fast_math: bool = False,
              ftz: Optional[bool] = None,
              prec_div: Optional[bool] = None,
              prec_sqrt: Optional[bool] = None,
              fmad: Optional[bool] = None,
              extra_flags: Optional[List[str]] = None) -> "StructuredCudaBuilder":
        """Set CUDA compilation flags (maxrregcount, arch, fast_math, etc.)."""
        self._flags = CompilationFlags(
            maxrregcount=maxrregcount,
            arch=arch,
            use_fast_math=use_fast_math,
            ftz=ftz,
            prec_div=prec_div,
            prec_sqrt=prec_sqrt,
            fmad=fmad,
            extra_flags=extra_flags or []
        )
        return self

    # --- Struct/Class ---

    def struct_begin(self, name: str, template: Optional[List[str]] = None,
                     is_class: bool = False) -> "StructuredCudaBuilder":
        """Begin struct/class definition."""
        self._stack.append(("struct", []))
        self._stack[-1] = ("struct", [], name, template, is_class)  # Store metadata
        return self

    def field(self, name: str, type_name: str,
              init: Optional[str] = None) -> "StructuredCudaBuilder":
        """Add field to current struct."""
        if not self._stack or self._stack[-1][0] != "struct":
            raise ValueError("field() must be inside struct_begin/struct_end")
        self._stack[-1][1].append(StructField(
            name=name,
            type=TypeRef(name=type_name),
            initializer=init
        ))
        return self

    def using(self, name: str, type_expr: str) -> "StructuredCudaBuilder":
        """Add using declaration."""
        if self._stack and self._stack[-1][0] == "struct":
            self._stack[-1][1].append(UsingDecl(name=name, type_expr=type_expr))
        else:
            self._current_list().append(RawCode(text=f"using {name} = {type_expr};"))
        return self

    def struct_end(self) -> "StructuredCudaBuilder":
        """End struct/class definition."""
        if not self._stack or self._stack[-1][0] != "struct":
            raise ValueError("struct_end() without matching struct_begin()")
        ctx = self._stack.pop()
        members = ctx[1]
        name = ctx[2] if len(ctx) > 2 else "Unknown"
        template = ctx[3] if len(ctx) > 3 else None
        is_class = ctx[4] if len(ctx) > 4 else False

        self._current_list().append(Struct(
            name=name,
            members=members,
            is_class=is_class,
            template_params=template
        ))
        return self

    # --- Constexpr ---

    def constexpr(self, name: str, type_name: str, value: str,
                  storage: str = "static") -> "StructuredCudaBuilder":
        """Add constexpr declaration."""
        self._current_list().append(Constexpr(
            name=name,
            type=TypeRef(name=type_name),
            value=value,
            storage=StorageClass.STATIC if storage == "static" else StorageClass.NONE
        ))
        return self

    # --- Functions ---

    def func_begin(self, name: str, return_type: str,
                   params: Optional[List[Tuple[str, str]]] = None,
                   qualifier: Optional[str] = None,
                   template: Optional[List[str]] = None,
                   launch_bounds: Optional[Tuple[int, ...]] = None,
                   forceinline: bool = False,
                   inline: bool = False,
                   storage: Optional[str] = None) -> "StructuredCudaBuilder":
        """Begin function definition."""
        self._stack.append(("func", [], {
            "name": name,
            "return_type": return_type,
            "params": params or [],
            "qualifier": qualifier,
            "template": template,
            "launch_bounds": launch_bounds,
            "forceinline": forceinline,
            "inline": inline,
            "storage": storage
        }))
        return self

    def func_end(self) -> "StructuredCudaBuilder":
        """End function definition."""
        if not self._stack or self._stack[-1][0] != "func":
            raise ValueError("func_end() without matching func_begin()")
        ctx = self._stack.pop()
        body = ctx[1]
        meta = ctx[2]

        # Convert params
        params = [Parameter(name=n, type=TypeRef(name=t)) for t, n in meta["params"]]

        # Parse qualifier
        qual_map = {
            "__device__": FunctionQualifier.DEVICE,
            "__global__": FunctionQualifier.GLOBAL,
            "__host__": FunctionQualifier.HOST,
            "__host__ __device__": FunctionQualifier.HOST_DEVICE,
        }
        qualifier = qual_map.get(meta["qualifier"], FunctionQualifier.NONE)

        # Parse launch bounds
        lb = None
        if meta["launch_bounds"]:
            if len(meta["launch_bounds"]) >= 2:
                lb = LaunchBounds(meta["launch_bounds"][0], meta["launch_bounds"][1])
            else:
                lb = LaunchBounds(meta["launch_bounds"][0])

        # Parse storage
        storage_map = {
            "static": StorageClass.STATIC,
            "extern": StorageClass.EXTERN,
            "inline": StorageClass.INLINE,
        }
        storage = storage_map.get(meta.get("storage"), StorageClass.NONE)

        self._current_list().append(Function(
            name=meta["name"],
            return_type=TypeRef(name=meta["return_type"]),
            params=params,
            body=body,
            qualifier=qualifier,
            is_inline=meta["inline"],
            is_forceinline=meta["forceinline"],
            storage=storage,
            template_params=meta["template"],
            launch_bounds=lb
        ))
        return self

    # --- Statements ---

    def stmt(self, expr: str) -> "StructuredCudaBuilder":
        """Add statement (expression + semicolon)."""
        self._current_list().append(CudaStatement(expr=expr))
        return self

    def ret(self, value: Optional[str] = None) -> "StructuredCudaBuilder":
        """Add return statement."""
        self._current_list().append(Return(value=value))
        return self

    # --- Control Flow ---

    def if_begin(self, condition: str, constexpr: bool = False) -> "StructuredCudaBuilder":
        """Begin if statement."""
        self._stack.append(("if", [], {"condition": condition, "constexpr": constexpr, "else": None}))
        return self

    def else_begin(self) -> "StructuredCudaBuilder":
        """Begin else branch."""
        if not self._stack or self._stack[-1][0] != "if":
            raise ValueError("else_begin() without matching if_begin()")
        # Save then body and start else body
        then_body = self._stack[-1][1]
        self._stack[-1] = ("if_else", [], {**self._stack[-1][2], "then": then_body})
        return self

    def if_end(self) -> "StructuredCudaBuilder":
        """End if statement."""
        if not self._stack or self._stack[-1][0] not in ("if", "if_else"):
            raise ValueError("if_end() without matching if_begin()")
        ctx = self._stack.pop()
        meta = ctx[2]

        if ctx[0] == "if_else":
            then_body = meta["then"]
            else_body = ctx[1]
        else:
            then_body = ctx[1]
            else_body = None

        self._current_list().append(If(
            condition=meta["condition"],
            then_body=then_body,
            else_body=else_body,
            is_constexpr=meta["constexpr"]
        ))
        return self

    def for_begin(self, init: str, condition: str, increment: str,
                  unroll: Union[bool, int, None] = None) -> "StructuredCudaBuilder":
        """Begin for loop."""
        pragma_unroll = None
        if unroll is True:
            pragma_unroll = -1
        elif isinstance(unroll, int):
            pragma_unroll = unroll

        self._stack.append(("for", [], {
            "init": init,
            "condition": condition,
            "increment": increment,
            "unroll": pragma_unroll
        }))
        return self

    def for_end(self) -> "StructuredCudaBuilder":
        """End for loop."""
        if not self._stack or self._stack[-1][0] != "for":
            raise ValueError("for_end() without matching for_begin()")
        ctx = self._stack.pop()
        body = ctx[1]
        meta = ctx[2]

        self._current_list().append(For(
            init=meta["init"],
            condition=meta["condition"],
            increment=meta["increment"],
            body=body,
            pragma_unroll=meta["unroll"]
        ))
        return self

    def while_begin(self, condition: str) -> "StructuredCudaBuilder":
        """Begin while loop."""
        self._stack.append(("while", [], {"condition": condition}))
        return self

    def while_end(self) -> "StructuredCudaBuilder":
        """End while loop."""
        if not self._stack or self._stack[-1][0] != "while":
            raise ValueError("while_end() without matching while_begin()")
        ctx = self._stack.pop()
        body = ctx[1]
        meta = ctx[2]

        self._current_list().append(While(
            condition=meta["condition"],
            body=body
        ))
        return self

    # --- Inline ASM ---

    def asm(self, ptx_ast: PTXModule,
            outputs: Optional[List[Tuple[str, str]]] = None,
            inputs: Optional[List[Tuple[str, str]]] = None,
            clobbers: Optional[List[str]] = None) -> "StructuredCudaBuilder":
        """Add inline asm block from PTX AST."""
        ptx_lines = ptx_ast.emit().split('\n')
        self._current_list().append(InlineAsm(
            ptx_lines=ptx_lines,
            outputs=outputs or [],
            inputs=inputs or [],
            clobbers=clobbers or [],
            ptx_ast=ptx_ast
        ))
        return self

    def asm_raw(self, ptx_lines: List[str],
                outputs: Optional[List[Tuple[str, str]]] = None,
                inputs: Optional[List[Tuple[str, str]]] = None,
                clobbers: Optional[List[str]] = None) -> "StructuredCudaBuilder":
        """Add inline asm block from raw PTX lines."""
        self._current_list().append(InlineAsm(
            ptx_lines=ptx_lines,
            outputs=outputs or [],
            inputs=inputs or [],
            clobbers=clobbers or []
        ))
        return self

    # --- Build ---

    def build(self) -> CudaModule:
        """Build and return the CudaModule AST."""
        if self._stack:
            raise ValueError(f"Unclosed constructs: {[s[0] for s in self._stack]}")
        return CudaModule(nodes=self._nodes, flags=self._flags)

    def build_source(self) -> str:
        """Build and return the CUDA source as a string."""
        return self.build().emit()

    def clear(self) -> "StructuredCudaBuilder":
        """Clear all nodes and start fresh."""
        self._nodes = []
        self._stack = []
        self._flags = None
        return self


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test basic builder usage
    b = ASTBuilder()
    b.reg("b8", ["a0", "a1", "a2", "a3"])
    b.reg("b32", ["%r0", "%r1"])
    b.instr("mov", ["b32"], [vec(reg("a0"), reg("a1"), reg("a2"), reg("a3")), reg("%r0")])
    b.instr("st", ["global", "b32"], [mem(reg("%rd0")), reg("a0")])
    b.instr("bra", [], [sym("$LOOP")], pred="!%p0")

    ast = b.build()
    print("=== Built AST ===")
    print(ast.emit())
    print()

    # Test roundtrip: AST -> builder commands -> execute -> AST
    print("=== Builder Commands ===")
    print(ast.to_builder_commands())
    print()

    # Test InstrBuilder
    print("=== InstrBuilder Test ===")
    i = instr("mma").mod("sync", "aligned", "m16n8k32").ops(
        vec(reg("d0"), reg("d1"), reg("d2"), reg("d3")),
        vec(reg("a0"), reg("a1")),
        reg("b0"),
        vec(reg("c0"), reg("c1"), reg("c2"), reg("c3"))
    ).build()
    print(i.emit())
