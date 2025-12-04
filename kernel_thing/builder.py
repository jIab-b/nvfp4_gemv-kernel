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

from typing import List, Optional, Union
from ptx_ast import (
    PTXModule, Instruction, Directive, Label, Comment, RawLine, Line, Block,
    RegisterDecl, SharedDecl, RegisterOp, ImmediateOp, VectorOp, MemoryOp, SymbolOp,
    Operand, Statement
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
