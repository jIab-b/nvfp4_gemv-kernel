"""
PTX Abstract Syntax Tree

Lossless parsing and reconstruction of PTX assembly.
Handles standalone PTX files and inline asm in CUDA/Python sources.

Design:
    Layer 0 (Normalize): Convert escaped strings → clean PTX with real newlines
    Layer 1 (Parse): Line-by-line parsing into AST
    Layer 2 (Emit): AST → clean PTX or escaped C string

Usage:
    # Deconstruct any source file
    doc = deconstruct(source)

    # Modify AST
    for block in doc.asm_blocks:
        for instr in block.ast.find_instructions('ld'):
            instr.modifiers.append('cs')

    # Reconstruct
    new_source = reconstruct(doc)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple


# =============================================================================
# LAYER 0: NORMALIZATION
# =============================================================================

def normalize_ptx(s: str) -> str:
    """Convert C-escaped PTX string to clean PTX with real newlines."""
    s = s.replace('\\n', '\n')
    s = s.replace('\\t', '\t')
    s = s.replace('\\r', '\r')
    s = s.replace('\\"', '"')
    s = s.replace('\\\\', '\\')
    # Remove line continuation backslashes (from C macro style)
    s = s.replace('\\\n', '\n')
    return s


def escape_ptx(s: str) -> str:
    """Convert clean PTX back to C-escaped string for inline asm."""
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\t', '\\t')
    s = s.replace('\n', '\\n')
    return s


# =============================================================================
# AST NODE TYPES
# =============================================================================

@dataclass
class RegisterOp:
    """Register: %r0, %rd5, a0, cvt_a0"""
    name: str
    def emit(self) -> str: return self.name
    def to_builder(self) -> str: return f'reg("{self.name}")'


@dataclass
class ImmediateOp:
    """Immediate: 0, 16, 0xffffffff"""
    value: str
    def emit(self) -> str: return self.value
    def to_builder(self) -> str: return f'imm("{self.value}")'


@dataclass
class VectorOp:
    """Vector: {a0, a1, a2, a3}"""
    elements: List[Operand]
    def emit(self) -> str:
        return "{" + ", ".join(e.emit() for e in self.elements) + "}"
    def to_builder(self) -> str:
        args = ", ".join(e.to_builder() for e in self.elements)
        return f'vec({args})'


@dataclass
class MemoryOp:
    """Memory: [%rd0], [%rd0+16]"""
    base: Operand
    offset: Optional[Operand] = None
    offset_op: str = "+"
    def emit(self) -> str:
        if self.offset:
            return f"[{self.base.emit()}{self.offset_op}{self.offset.emit()}]"
        return f"[{self.base.emit()}]"
    def to_builder(self) -> str:
        if self.offset:
            return f'mem({self.base.to_builder()}, {self.offset.to_builder()}, "{self.offset_op}")'
        return f'mem({self.base.to_builder()})'


@dataclass
class SymbolOp:
    """Symbol/label: $LOOP, param_name"""
    name: str
    def emit(self) -> str: return self.name
    def to_builder(self) -> str: return f'sym("{self.name}")'


Operand = Union[RegisterOp, ImmediateOp, VectorOp, MemoryOp, SymbolOp]


@dataclass
class Instruction:
    """PTX instruction: mov.b32 {a0, a1}, %r0;"""
    mnemonic: str
    modifiers: List[str] = field(default_factory=list)
    operands: List[Operand] = field(default_factory=list)
    predicate: Optional[str] = None

    def emit(self) -> str:
        parts = []
        if self.predicate:
            parts.append(f"@{self.predicate} ")
        if self.modifiers:
            parts.append(f"{self.mnemonic}.{'.'.join(self.modifiers)}")
        else:
            parts.append(self.mnemonic)
        if self.operands:
            parts.append(" " + ", ".join(op.emit() for op in self.operands))
        return "".join(parts) + ";"

    def to_builder(self) -> str:
        mods = repr(self.modifiers)
        ops = "[" + ", ".join(op.to_builder() for op in self.operands) + "]"
        pred = f', pred="{self.predicate}"' if self.predicate else ""
        return f'b.instr("{self.mnemonic}", {mods}, {ops}{pred})'

    def to_constructor(self) -> str:
        """Return code that constructs this instruction object directly."""
        mods = repr(self.modifiers)
        ops = "[" + ", ".join(op.to_builder() for op in self.operands) + "]"
        pred = f', predicate="{self.predicate}"' if self.predicate else ""
        return f'Instruction("{self.mnemonic}", {mods}, {ops}{pred})'


@dataclass
class Directive:
    """PTX directive: .reg .b8 a0, a1;"""
    text: str
    def emit(self) -> str: return self.text
    def to_builder(self) -> str: return f'b.directive({repr(self.text)})'


@dataclass
class Label:
    """Label: $LOOP:"""
    name: str
    def emit(self) -> str: return f"{self.name}:"
    def to_builder(self) -> str: return f'b.label("{self.name}")'


@dataclass
class Comment:
    """Comment: // text"""
    text: str
    def emit(self) -> str: return f"// {self.text}"
    def to_builder(self) -> str: return f'b.comment({repr(self.text)})'


@dataclass
class RawLine:
    """Unparseable line - preserve as-is"""
    text: str
    def emit(self) -> str: return self.text
    def to_builder(self) -> str: return f'b.raw({repr(self.text)})'


@dataclass
class Line:
    """Multiple statements on one line (common in inline asm)"""
    statements: List[Union[Instruction, Directive]]
    separator: str = " "

    def emit(self) -> str:
        return self.separator.join(s.emit() for s in self.statements)

    def to_builder(self) -> str:
        # Use to_constructor() for statements to create objects, not builder calls
        constructors = []
        for s in self.statements:
            if hasattr(s, 'to_constructor'):
                constructors.append(s.to_constructor())
            else:
                # Fallback for Directive (just use Directive constructor)
                constructors.append(f'Directive({repr(s.text)})')
        return f'b.line([{", ".join(constructors)}])'


@dataclass
class Block:
    """Block scope: { ... }"""
    statements: List[Statement] = field(default_factory=list)

    def emit(self) -> str:
        inner = "\n".join(s.emit() for s in self.statements)
        return "{\n" + inner + "\n}" if inner else "{}"

    def to_builder(self) -> str:
        lines = ["b.block_open()"]
        for s in self.statements:
            lines.append(s.to_builder())
        lines.append("b.block_close()")
        return "\n".join(lines)


@dataclass
class RegisterDecl:
    """Parsed register declaration: .reg .b8 a0, a1, a2;"""
    dtype: str
    names: List[str]

    def emit(self) -> str:
        return f".reg .{self.dtype} {', '.join(self.names)};"

    def to_builder(self) -> str:
        return f'b.reg("{self.dtype}", {repr(self.names)})'


@dataclass
class SharedDecl:
    """Parsed shared memory declaration: .shared .align 16 .b8 smem[4096];"""
    dtype: str
    name: str
    size: int
    align: Optional[int] = None

    def emit(self) -> str:
        align_str = f".align {self.align} " if self.align else ""
        return f".shared {align_str}.{self.dtype} {self.name}[{self.size}];"

    def to_builder(self) -> str:
        align = f", align={self.align}" if self.align else ""
        return f'b.shared("{self.dtype}", "{self.name}", {self.size}{align})'


Statement = Union[Instruction, Directive, Label, Comment, RawLine, Line, Block, RegisterDecl, SharedDecl]


@dataclass
class PTXModule:
    """Root AST node"""
    statements: List[Statement] = field(default_factory=list)

    def emit(self) -> str:
        return "\n".join(s.emit() for s in self.statements)

    def _iter_all(self):
        """Recursively iterate all statements including nested ones."""
        for s in self.statements:
            yield s
            if isinstance(s, Line):
                yield from s.statements
            elif isinstance(s, Block):
                yield from s.statements

    def find_instructions(self, mnemonic: str = None) -> List[Instruction]:
        return [s for s in self._iter_all()
                if isinstance(s, Instruction) and (mnemonic is None or s.mnemonic == mnemonic)]

    def find_directives(self, name: str = None) -> List[Directive]:
        return [s for s in self._iter_all()
                if isinstance(s, Directive) and (name is None or s.text.startswith(name))]

    def find_register_decls(self) -> List[RegisterDecl]:
        return [s for s in self._iter_all() if isinstance(s, RegisterDecl)]

    def to_builder_commands(self) -> str:
        """Generate Python code that reconstructs this AST via builder API."""
        lines = [
            "from builder import ASTBuilder, reg, imm, vec, mem, sym",
            "from ptx_ast import Instruction, Directive",
            "",
            "b = ASTBuilder()",
        ]
        for s in self.statements:
            lines.append(s.to_builder())
        lines.append("")
        lines.append("ast = b.build()")
        return "\n".join(lines)


# =============================================================================
# LAYER 1: PARSER
# =============================================================================

def parse_ptx(source: str) -> PTXModule:
    """Parse normalized PTX into AST."""
    statements = []
    for line in source.split('\n'):
        stmt = _parse_line(line)
        if stmt:
            statements.append(stmt)
    return PTXModule(statements=statements)


def _parse_line(line: str) -> Optional[Statement]:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith('//'):
        return Comment(text=stripped[2:].strip())
    if re.match(r'^[\$\w]+:$', stripped):
        return Label(name=stripped[:-1])
    if stripped.startswith('.'):
        return _parse_directive(stripped)
    if stripped == '{':
        return RawLine(text='{')
    if stripped == '}':
        return RawLine(text='}')
    return _parse_instruction_line(stripped)


def _parse_directive(text: str) -> Statement:
    """Parse directive, extracting structure where possible."""
    # .reg .dtype name1, name2, ...;
    reg_match = re.match(r'^\.reg\s+\.(\w+)\s+(.+);?$', text)
    if reg_match:
        dtype = reg_match.group(1)
        names_str = reg_match.group(2).rstrip(';')
        names = [n.strip() for n in names_str.split(',')]
        return RegisterDecl(dtype=dtype, names=names)

    # .shared .align N .dtype name[size];
    shared_match = re.match(r'^\.shared\s+(?:\.align\s+(\d+)\s+)?\.(\w+)\s+(\w+)\[(\d+)\];?$', text)
    if shared_match:
        align = int(shared_match.group(1)) if shared_match.group(1) else None
        dtype = shared_match.group(2)
        name = shared_match.group(3)
        size = int(shared_match.group(4))
        return SharedDecl(dtype=dtype, name=name, size=size, align=align)

    # Fallback to raw directive
    return Directive(text=text)


def _parse_instruction_line(line: str) -> Statement:
    """Parse line with one or more instructions."""
    parts = _split_on_semicolons(line)

    if len(parts) == 0:
        return RawLine(text=line)

    if len(parts) == 1:
        instr = _parse_single_instruction(parts[0])
        return instr if instr else RawLine(text=line)

    # Multiple instructions per line
    parsed = []
    for part in parts:
        instr = _parse_single_instruction(part)
        if instr:
            parsed.append(instr)
        else:
            # If any part fails to parse, fall back to raw
            return RawLine(text=line)

    return Line(statements=parsed, separator=" ")


def _split_on_semicolons(line: str) -> List[str]:
    parts, current, depth = [], "", 0
    for c in line:
        if c == '{': depth += 1
        elif c == '}': depth -= 1
        elif c == ';' and depth == 0:
            if current.strip():
                parts.append(current.strip())
            current = ""
            continue
        current += c
    if current.strip():
        parts.append(current.strip())
    return parts


def _parse_single_instruction(text: str) -> Optional[Instruction]:
    text = text.strip().rstrip(';')
    if not text:
        return None

    predicate = None
    if text.startswith('@'):
        match = re.match(r'^@(!?%?\w+)\s+', text)
        if match:
            predicate = match.group(1)
            text = text[match.end():]

    parts = text.split(None, 1)
    if not parts:
        return None

    mnem_parts = parts[0].split('.')
    mnemonic = mnem_parts[0]
    modifiers = mnem_parts[1:] if len(mnem_parts) > 1 else []
    operands = _parse_operands(parts[1] if len(parts) > 1 else "")

    return Instruction(mnemonic=mnemonic, modifiers=modifiers, operands=operands, predicate=predicate)


def _parse_operands(s: str) -> List[Operand]:
    s = s.strip()
    if not s:
        return []

    tokens, current, depth = [], "", 0
    for c in s:
        if c in '{[': depth += 1
        elif c in '}]': depth -= 1
        elif c == ',' and depth == 0:
            if current.strip():
                tokens.append(current.strip())
            current = ""
            continue
        current += c
    if current.strip():
        tokens.append(current.strip())

    return [_parse_operand(t) for t in tokens]


def _parse_operand(s: str) -> Operand:
    s = s.strip()

    if s.startswith('{') and s.endswith('}'):
        elements = [_parse_operand(x.strip()) for x in s[1:-1].split(',')]
        return VectorOp(elements=elements)

    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        for op in ['+', '-']:
            if op in inner:
                parts = inner.split(op, 1)
                return MemoryOp(base=_parse_operand(parts[0].strip()),
                               offset=_parse_operand(parts[1].strip()), offset_op=op)
        return MemoryOp(base=_parse_operand(inner))

    if re.match(r'^-?\d+\.?\d*[fFdD]?$', s) or re.match(r'^0x[0-9a-fA-F]+$', s):
        return ImmediateOp(value=s)

    if s.startswith('$'):
        return SymbolOp(name=s)

    return RegisterOp(name=s)


# =============================================================================
# LAYER 2: EMIT
# =============================================================================

def emit_ptx(ast: PTXModule) -> str:
    return ast.emit()


def emit_c_string(ast: PTXModule) -> str:
    return escape_ptx(ast.emit())


# =============================================================================
# INLINE ASM EXTRACTION
# =============================================================================

@dataclass
class InlineAsmBlock:
    """An asm volatile(...) block from CUDA source."""
    ptx_raw: str
    ptx_normalized: str
    outputs: List[Tuple[str, str]]
    inputs: List[Tuple[str, str]]
    clobbers: List[str]
    start_pos: int
    end_pos: int
    ast: Optional[PTXModule] = None


def extract_inline_asm(cuda_source: str) -> List[InlineAsmBlock]:
    """Extract all inline asm blocks from CUDA source."""
    blocks = []
    for match in re.finditer(r'asm\s+(?:volatile|__volatile__)\s*\(', cuda_source):
        result = _parse_asm_block(cuda_source, match.start())
        if result:
            blocks.append(result)
    return blocks


def _parse_asm_block(source: str, start: int) -> Optional[InlineAsmBlock]:
    paren_match = re.match(r'asm\s+(?:volatile|__volatile__)\s*\(', source[start:])
    if not paren_match:
        return None

    pos = start + paren_match.end()
    strings = []

    while pos < len(source):
        while pos < len(source) and source[pos] in ' \t\n\r':
            pos += 1
        if pos >= len(source):
            break
        if source[pos:pos+2] == '//':
            pos = source.find('\n', pos)
            if pos == -1: break
            pos += 1
            continue
        if source[pos] == '"':
            end = pos + 1
            while end < len(source):
                if source[end] == '\\' and end + 1 < len(source):
                    end += 2
                elif source[end] == '"':
                    break
                else:
                    end += 1
            if end < len(source):
                strings.append(source[pos+1:end])
                pos = end + 1
            else:
                break
            continue
        if source[pos] in ':)':
            break
        pos += 1

    ptx_raw = "".join(strings)
    ptx_normalized = normalize_ptx(ptx_raw)

    outputs, inputs, clobbers = [], [], []
    section = 0

    while pos < len(source) and source[pos] != ')':
        while pos < len(source) and source[pos] in ' \t\n\r':
            pos += 1
        if pos >= len(source):
            break
        if source[pos] == ':':
            section += 1
            pos += 1
            section_end, paren_depth = pos, 0
            while section_end < len(source):
                c = source[section_end]
                if c == '(': paren_depth += 1
                elif c == ')':
                    if paren_depth == 0: break
                    paren_depth -= 1
                elif c == ':' and paren_depth == 0:
                    break
                section_end += 1
            content = source[pos:section_end].strip()
            if section == 1:
                outputs = re.findall(r'"([^"]+)"\s*\(\s*([^)]+)\s*\)', content)
            elif section == 2:
                inputs = re.findall(r'"([^"]+)"\s*\(\s*([^)]+)\s*\)', content)
            elif section == 3:
                clobbers = re.findall(r'"([^"]+)"', content)
            pos = section_end
            continue
        if source[pos] == ')':
            break
        pos += 1

    end_pos = pos + 1 if pos < len(source) and source[pos] == ')' else pos
    if end_pos < len(source) and source[end_pos] == ';':
        end_pos += 1

    block = InlineAsmBlock(ptx_raw=ptx_raw, ptx_normalized=ptx_normalized,
                           outputs=outputs, inputs=inputs, clobbers=clobbers,
                           start_pos=start, end_pos=end_pos)
    block.ast = parse_ptx(ptx_normalized)
    return block


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

@dataclass
class DeconstructedSource:
    """Result of deconstructing a source file."""
    original: str
    source_type: str
    asm_blocks: List[InlineAsmBlock] = field(default_factory=list)
    ptx_ast: Optional[PTXModule] = None
    cuda_var_name: Optional[str] = None


def deconstruct(source: str, source_type: str = "auto") -> DeconstructedSource:
    """Deconstruct source into components."""
    if source_type == "auto":
        if 'import ' in source or 'def ' in source:
            source_type = "python"
        elif source.strip().startswith('.version') or source.strip().startswith('.target'):
            source_type = "ptx"
        elif '__global__' in source or 'asm volatile' in source:
            source_type = "cuda"
        else:
            source_type = "ptx"

    result = DeconstructedSource(original=source, source_type=source_type)

    if source_type == "ptx":
        result.ptx_ast = parse_ptx(source)
    elif source_type == "cuda":
        result.asm_blocks = extract_inline_asm(source)
    elif source_type == "python":
        for match in re.finditer(r'(\w+)\s*=\s*r?"""(.*?)"""', source, re.DOTALL):
            var_name, content = match.group(1), match.group(2)
            if '__global__' in content or 'asm volatile' in content or '__device__' in content:
                result.cuda_var_name = var_name
                result.asm_blocks = extract_inline_asm(content)
                break

    return result


def reconstruct(doc: DeconstructedSource) -> str:
    """Reconstruct source from deconstructed components."""
    if doc.source_type == "ptx" and doc.ptx_ast:
        return emit_ptx(doc.ptx_ast)

    if doc.source_type == "cuda" and doc.asm_blocks:
        return _reconstruct_with_asm_blocks(doc.original, doc.asm_blocks)

    if doc.source_type == "python" and doc.asm_blocks and doc.cuda_var_name:
        match = re.search(rf'({re.escape(doc.cuda_var_name)}\s*=\s*r?)"""(.*?)"""',
                         doc.original, re.DOTALL)
        if match:
            cuda_source = match.group(2)
            new_cuda = _reconstruct_with_asm_blocks(cuda_source, doc.asm_blocks)
            return doc.original[:match.start(2)] + new_cuda + doc.original[match.end(2):]

    return doc.original


def _reconstruct_with_asm_blocks(source: str, blocks: List[InlineAsmBlock]) -> str:
    for block in sorted(blocks, key=lambda b: b.start_pos, reverse=True):
        if block.ast:
            new_ptx = escape_ptx(emit_ptx(block.ast))
            new_asm = _build_asm_block(new_ptx, block.outputs, block.inputs, block.clobbers)
            source = source[:block.start_pos] + new_asm + source[block.end_pos:]
    return source


def _build_asm_block(ptx: str, outputs: List, inputs: List, clobbers: List) -> str:
    parts = [f'asm volatile(\n        "{ptx}"']
    if outputs:
        parts.append('\n        : ' + ', '.join(f'"{c}"({v})' for c, v in outputs))
    elif inputs or clobbers:
        parts.append('\n        :')
    if inputs:
        parts.append('\n        : ' + ', '.join(f'"{c}"({v})' for c, v in inputs))
    elif clobbers:
        parts.append('\n        :')
    if clobbers:
        parts.append('\n        : ' + ', '.join(f'"{c}"' for c in clobbers))
    parts.append('\n    )')
    return ''.join(parts)


def deconstruct_file(path: str) -> DeconstructedSource:
    """Deconstruct a file."""
    with open(path) as f:
        source = f.read()
    ext = path.rsplit('.', 1)[-1] if '.' in path else ''
    source_type = {"py": "python", "ptx": "ptx", "cu": "cuda", "cuh": "cuda"}.get(ext, "auto")
    return deconstruct(source, source_type)


if __name__ == "__main__":
    test_ptx = r"{\n.reg .b8 a0;\nmov.b32 a0, %0;\n}"
    print("Raw:", repr(test_ptx))
    normalized = normalize_ptx(test_ptx)
    print("Normalized:", repr(normalized))
    ast = parse_ptx(normalized)
    print(f"Statements: {len(ast.statements)}")
    for s in ast.statements:
        print(f"  {type(s).__name__}: {s.emit()}")
