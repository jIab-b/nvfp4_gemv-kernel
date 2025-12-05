"""
PTX Actions - Create ptx_ast nodes from PTX grammar actions

Given a PTXNodeType and values, create the corresponding ptx_ast node.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from ptx_ast import (
    Instruction, Directive, Label, RegisterDecl, SharedDecl,
    RegisterOp, ImmediateOp, VectorOp, MemoryOp, SymbolOp,
    Operand
)
from gen.ptx_grammar import PTXNodeType, PTXValueType, PTX_SPECS, get_ptx_spec


def create_ptx_node(node_type: PTXNodeType, values: Dict[str, Any]) -> Optional[Any]:
    """
    Create a ptx_ast node from a PTXNodeType and values.

    Args:
        node_type: The PTX node type to create
        values: Dictionary mapping operand names to values

    Returns:
        The created ptx_ast node, or None for PTX_END
    """
    if node_type == PTXNodeType.PTX_END:
        return None

    creator = PTX_CREATORS.get(node_type)
    if creator is None:
        raise ValueError(f"No creator for PTX node type: {node_type}")

    return creator(values)


def _parse_operand(val: Any) -> Operand:
    """Convert a value to a ptx_ast Operand."""
    if isinstance(val, (RegisterOp, ImmediateOp, VectorOp, MemoryOp, SymbolOp)):
        return val

    s = str(val)

    # Vector: {a, b, c, d}
    if s.startswith('{') and s.endswith('}'):
        inner = s[1:-1]
        parts = [p.strip() for p in inner.split(',')]
        return VectorOp(elements=[_parse_operand(p) for p in parts])

    # Memory: [addr] or [addr+offset]
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        if '+' in inner:
            parts = inner.split('+', 1)
            return MemoryOp(
                base=_parse_operand(parts[0].strip()),
                offset=_parse_operand(parts[1].strip()),
                offset_op='+'
            )
        elif '-' in inner:
            parts = inner.split('-', 1)
            return MemoryOp(
                base=_parse_operand(parts[0].strip()),
                offset=_parse_operand(parts[1].strip()),
                offset_op='-'
            )
        return MemoryOp(base=_parse_operand(inner))

    # Label: $LOOP
    if s.startswith('$'):
        return SymbolOp(name=s)

    # Immediate: numbers
    if s.lstrip('-').replace('.', '').isdigit() or s.startswith('0x'):
        return ImmediateOp(value=s)

    # Register: everything else
    return RegisterOp(name=s)


def _parse_modifiers(mods: Any) -> List[str]:
    """Parse modifiers from values."""
    if mods is None:
        return []
    if isinstance(mods, str):
        # Split on dots, filter empties
        return [m for m in mods.split('.') if m]
    if isinstance(mods, (list, tuple)):
        return list(mods)
    return []


# =============================================================================
# PTX NODE CREATORS
# =============================================================================

def _create_reg_decl(values: Dict[str, Any]) -> RegisterDecl:
    dtype = values.get("dtype", ".b32").lstrip('.')
    names = values.get("names", [])
    if isinstance(names, str):
        names = [n.strip() for n in names.split(',')]
    return RegisterDecl(dtype=dtype, names=names)


def _create_shared_decl(values: Dict[str, Any]) -> SharedDecl:
    dtype = values.get("dtype", ".b8").lstrip('.')
    name = values.get("name", "smem")
    size = int(values.get("size", 0))
    align = values.get("align")
    return SharedDecl(dtype=dtype, name=name, size=size, align=int(align) if align else None)


def _create_mov(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b32")
    mods = [dtype.lstrip('.')] if dtype else []
    return Instruction(
        mnemonic="mov",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("src", "0")),
        ]
    )


def _create_ld_global(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".u32")
    extra_mods = _parse_modifiers(values.get("modifiers"))
    mods = extra_mods + [dtype.lstrip('.')] if dtype else extra_mods
    return Instruction(
        mnemonic="ld.global",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("addr", "[%rd0]")),
        ]
    )


def _create_ld_shared(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".u32")
    extra_mods = _parse_modifiers(values.get("modifiers"))
    mods = extra_mods + [dtype.lstrip('.')] if dtype else extra_mods
    return Instruction(
        mnemonic="ld.shared",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("addr", "[smem]")),
        ]
    )


def _create_ld_param(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".u64")
    return Instruction(
        mnemonic="ld.param",
        modifiers=[dtype.lstrip('.')] if dtype else [],
        operands=[
            _parse_operand(values.get("dst", "%rd0")),
            _parse_operand(values.get("param", "[param0]")),
        ]
    )


def _create_st_global(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".u32")
    extra_mods = _parse_modifiers(values.get("modifiers"))
    mods = extra_mods + [dtype.lstrip('.')] if dtype else extra_mods
    return Instruction(
        mnemonic="st.global",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("addr", "[%rd0]")),
            _parse_operand(values.get("src", "%r0")),
        ]
    )


def _create_st_shared(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".u32")
    extra_mods = _parse_modifiers(values.get("modifiers"))
    mods = extra_mods + [dtype.lstrip('.')] if dtype else extra_mods
    return Instruction(
        mnemonic="st.shared",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("addr", "[smem]")),
            _parse_operand(values.get("src", "%r0")),
        ]
    )


def _create_add(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".s32")
    return Instruction(
        mnemonic="add",
        modifiers=[dtype.lstrip('.')] if dtype else [],
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("a", "%r1")),
            _parse_operand(values.get("b", "%r2")),
        ]
    )


def _create_sub(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".s32")
    return Instruction(
        mnemonic="sub",
        modifiers=[dtype.lstrip('.')] if dtype else [],
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("a", "%r1")),
            _parse_operand(values.get("b", "%r2")),
        ]
    )


def _create_mul(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".s32")
    extra_mods = _parse_modifiers(values.get("modifiers", ".lo"))
    mods = extra_mods + [dtype.lstrip('.')] if dtype else extra_mods
    return Instruction(
        mnemonic="mul",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("a", "%r1")),
            _parse_operand(values.get("b", "%r2")),
        ]
    )


def _create_mad(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".s32")
    extra_mods = _parse_modifiers(values.get("modifiers", ".lo"))
    mods = extra_mods + [dtype.lstrip('.')] if dtype else extra_mods
    return Instruction(
        mnemonic="mad",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("a", "%r1")),
            _parse_operand(values.get("b", "%r2")),
            _parse_operand(values.get("c", "%r3")),
        ]
    )


def _create_shl(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b32")
    return Instruction(
        mnemonic="shl",
        modifiers=[dtype.lstrip('.')] if dtype else [],
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("src", "%r1")),
            _parse_operand(values.get("amt", "1")),
        ]
    )


def _create_shr(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b32")
    return Instruction(
        mnemonic="shr",
        modifiers=[dtype.lstrip('.')] if dtype else [],
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("src", "%r1")),
            _parse_operand(values.get("amt", "1")),
        ]
    )


def _create_add_f(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".f32")
    extra_mods = _parse_modifiers(values.get("modifiers", ".rn"))
    mods = extra_mods + [dtype.lstrip('.')] if dtype else extra_mods
    return Instruction(
        mnemonic="add",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%f0")),
            _parse_operand(values.get("a", "%f1")),
            _parse_operand(values.get("b", "%f2")),
        ]
    )


def _create_mul_f(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".f32")
    extra_mods = _parse_modifiers(values.get("modifiers", ".rn"))
    mods = extra_mods + [dtype.lstrip('.')] if dtype else extra_mods
    return Instruction(
        mnemonic="mul",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%f0")),
            _parse_operand(values.get("a", "%f1")),
            _parse_operand(values.get("b", "%f2")),
        ]
    )


def _create_sub_f(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".f32")
    extra_mods = _parse_modifiers(values.get("modifiers", ".rn"))
    mods = extra_mods + [dtype.lstrip('.')] if dtype else extra_mods
    return Instruction(
        mnemonic="sub",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%f0")),
            _parse_operand(values.get("a", "%f1")),
            _parse_operand(values.get("b", "%f2")),
        ]
    )


def _create_div_f(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".f32")
    extra_mods = _parse_modifiers(values.get("modifiers", ".rn"))
    mods = extra_mods + [dtype.lstrip('.')] if dtype else extra_mods
    return Instruction(
        mnemonic="div",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%f0")),
            _parse_operand(values.get("a", "%f1")),
            _parse_operand(values.get("b", "%f2")),
        ]
    )


def _create_fma(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".f32")
    extra_mods = _parse_modifiers(values.get("modifiers", ".rn"))
    mods = extra_mods + [dtype.lstrip('.')] if dtype else extra_mods
    return Instruction(
        mnemonic="fma",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%f0")),
            _parse_operand(values.get("a", "%f1")),
            _parse_operand(values.get("b", "%f2")),
            _parse_operand(values.get("c", "%f3")),
        ]
    )


def _create_cvt(values: Dict[str, Any]) -> Instruction:
    dst_dtype = values.get("dst_dtype", ".f16")
    src_dtype = values.get("src_dtype", ".f32")
    extra_mods = _parse_modifiers(values.get("modifiers", ".rn"))
    mods = extra_mods + [dst_dtype.lstrip('.'), src_dtype.lstrip('.')]
    return Instruction(
        mnemonic="cvt",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%h0")),
            _parse_operand(values.get("src", "%f0")),
        ]
    )


def _create_and(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b32")
    return Instruction(
        mnemonic="and",
        modifiers=[dtype.lstrip('.')] if dtype else [],
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("a", "%r1")),
            _parse_operand(values.get("b", "%r2")),
        ]
    )


def _create_or(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b32")
    return Instruction(
        mnemonic="or",
        modifiers=[dtype.lstrip('.')] if dtype else [],
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("a", "%r1")),
            _parse_operand(values.get("b", "%r2")),
        ]
    )


def _create_xor(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b32")
    return Instruction(
        mnemonic="xor",
        modifiers=[dtype.lstrip('.')] if dtype else [],
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("a", "%r1")),
            _parse_operand(values.get("b", "%r2")),
        ]
    )


def _create_not(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b32")
    return Instruction(
        mnemonic="not",
        modifiers=[dtype.lstrip('.')] if dtype else [],
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("src", "%r1")),
        ]
    )


def _create_setp(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".s32")
    cmp_op = values.get("cmp", ".lt")
    mods = [cmp_op.lstrip('.'), dtype.lstrip('.')]
    return Instruction(
        mnemonic="setp",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("pred", "%p0")),
            _parse_operand(values.get("a", "%r0")),
            _parse_operand(values.get("b", "%r1")),
        ]
    )


def _create_selp(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b32")
    return Instruction(
        mnemonic="selp",
        modifiers=[dtype.lstrip('.')] if dtype else [],
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("a", "%r1")),
            _parse_operand(values.get("b", "%r2")),
            _parse_operand(values.get("pred", "%p0")),
        ]
    )


def _create_bra(values: Dict[str, Any]) -> Instruction:
    uni = values.get("uniform", False)
    mods = ["uni"] if uni else []
    return Instruction(
        mnemonic="bra",
        modifiers=mods,
        operands=[_parse_operand(values.get("label", "$LOOP"))]
    )


def _create_label(values: Dict[str, Any]) -> Label:
    return Label(name=values.get("name", "$LABEL"))


def _create_bar(values: Dict[str, Any]) -> Instruction:
    return Instruction(
        mnemonic="bar",
        modifiers=["sync"],
        operands=[_parse_operand(values.get("barrier_id", "0"))]
    )


def _create_membar(values: Dict[str, Any]) -> Instruction:
    scope = values.get("scope", ".cta")
    return Instruction(
        mnemonic="membar",
        modifiers=[scope.lstrip('.')],
        operands=[]
    )


def _create_shfl(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b32")
    mode = values.get("mode", ".bfly")
    mods = ["sync", mode.lstrip('.'), dtype.lstrip('.')]
    return Instruction(
        mnemonic="shfl",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("src", "%r1")),
            _parse_operand(values.get("lane", "0")),
            _parse_operand(values.get("mask", "0xffffffff")),
        ]
    )


def _create_redux(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".s32")
    op = values.get("op", ".add")
    mods = ["sync", op.lstrip('.'), dtype.lstrip('.')]
    return Instruction(
        mnemonic="redux",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("src", "%r1")),
            _parse_operand(values.get("mask", "0xffffffff")),
        ]
    )


def _create_mma(values: Dict[str, Any]) -> Instruction:
    shape = values.get("shape", ".m16n8k16")
    d_dtype = values.get("d_dtype", ".f16")
    a_dtype = values.get("a_dtype", ".f16")
    layout_a = values.get("layout_a", ".row")
    layout_b = values.get("layout_b", ".col")
    mods = ["sync", "aligned", shape.lstrip('.'), layout_a.lstrip('.'), layout_b.lstrip('.'),
            d_dtype.lstrip('.'), a_dtype.lstrip('.')]
    return Instruction(
        mnemonic="mma",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("d", "{%r0, %r1}")),
            _parse_operand(values.get("a", "{%r2, %r3}")),
            _parse_operand(values.get("b", "{%r4}")),
            _parse_operand(values.get("c", "{%r0, %r1}")),
        ]
    )


def _create_ldmatrix(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b16")
    shape = values.get("shape", ".m8n8")
    count = values.get("count", ".x4")
    trans = values.get("trans", False)
    mods = ["sync", "aligned", shape.lstrip('.'), count.lstrip('.'), dtype.lstrip('.')]
    if trans:
        mods.insert(4, "trans")
    return Instruction(
        mnemonic="ldmatrix",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "{%r0, %r1, %r2, %r3}")),
            _parse_operand(values.get("addr", "[smem]")),
        ]
    )


def _create_stmatrix(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b16")
    shape = values.get("shape", ".m8n8")
    count = values.get("count", ".x4")
    mods = ["sync", "aligned", shape.lstrip('.'), count.lstrip('.'), dtype.lstrip('.')]
    return Instruction(
        mnemonic="stmatrix",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("addr", "[smem]")),
            _parse_operand(values.get("src", "{%r0, %r1, %r2, %r3}")),
        ]
    )


def _create_vote(values: Dict[str, Any]) -> Instruction:
    mode = values.get("mode", ".all")
    mods = ["sync", mode.lstrip('.'), "b32"]
    return Instruction(
        mnemonic="vote",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("pred", "%p0")),
            _parse_operand(values.get("mask", "0xffffffff")),
        ]
    )


def _create_match(values: Dict[str, Any]) -> Instruction:
    mode = values.get("mode", ".any")
    dtype = values.get("dtype", ".b32")
    mods = ["sync", mode.lstrip('.'), dtype.lstrip('.')]
    return Instruction(
        mnemonic="match",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("membermask", "%r0")),
            _parse_operand(values.get("src", "%r1")),
            _parse_operand(values.get("mask", "0xffffffff")),
        ]
    )


def _create_dp2a(values: Dict[str, Any]) -> Instruction:
    d_dtype = values.get("d_dtype", ".s32")
    a_dtype = values.get("a_dtype", ".s32")
    mods = [d_dtype.lstrip('.'), a_dtype.lstrip('.')]
    return Instruction(
        mnemonic="dp2a",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("a", "%r1")),
            _parse_operand(values.get("b", "%r2")),
            _parse_operand(values.get("c", "%r3")),
        ]
    )


def _create_prmt(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b32")
    return Instruction(
        mnemonic="prmt",
        modifiers=[dtype.lstrip('.')] if dtype else [],
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("a", "%r1")),
            _parse_operand(values.get("b", "%r2")),
            _parse_operand(values.get("sel", "0x3210")),
        ]
    )


def _create_lop3(values: Dict[str, Any]) -> Instruction:
    dtype = values.get("dtype", ".b32")
    return Instruction(
        mnemonic="lop3",
        modifiers=[dtype.lstrip('.')] if dtype else [],
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("a", "%r1")),
            _parse_operand(values.get("b", "%r2")),
            _parse_operand(values.get("c", "%r3")),
            _parse_operand(values.get("lut", "0x96")),  # XOR by default
        ]
    )


def _create_dp4a(values: Dict[str, Any]) -> Instruction:
    d_dtype = values.get("d_dtype", ".s32")
    a_dtype = values.get("a_dtype", ".s32")
    mods = [d_dtype.lstrip('.'), a_dtype.lstrip('.')]
    return Instruction(
        mnemonic="dp4a",
        modifiers=mods,
        operands=[
            _parse_operand(values.get("dst", "%r0")),
            _parse_operand(values.get("a", "%r1")),
            _parse_operand(values.get("b", "%r2")),
            _parse_operand(values.get("c", "%r3")),
        ]
    )


# Map PTXNodeType to creator function
PTX_CREATORS = {
    PTXNodeType.PTX_REG_DECL: _create_reg_decl,
    PTXNodeType.PTX_SHARED_DECL: _create_shared_decl,
    PTXNodeType.PTX_MOV: _create_mov,
    PTXNodeType.PTX_LD_GLOBAL: _create_ld_global,
    PTXNodeType.PTX_LD_SHARED: _create_ld_shared,
    PTXNodeType.PTX_LD_PARAM: _create_ld_param,
    PTXNodeType.PTX_ST_GLOBAL: _create_st_global,
    PTXNodeType.PTX_ST_SHARED: _create_st_shared,
    PTXNodeType.PTX_ADD: _create_add,
    PTXNodeType.PTX_SUB: _create_sub,
    PTXNodeType.PTX_MUL: _create_mul,
    PTXNodeType.PTX_MAD: _create_mad,
    PTXNodeType.PTX_SHL: _create_shl,
    PTXNodeType.PTX_SHR: _create_shr,
    PTXNodeType.PTX_ADD_F: _create_add_f,
    PTXNodeType.PTX_SUB_F: _create_sub_f,
    PTXNodeType.PTX_MUL_F: _create_mul_f,
    PTXNodeType.PTX_DIV_F: _create_div_f,
    PTXNodeType.PTX_FMA: _create_fma,
    PTXNodeType.PTX_CVT: _create_cvt,
    PTXNodeType.PTX_AND: _create_and,
    PTXNodeType.PTX_OR: _create_or,
    PTXNodeType.PTX_XOR: _create_xor,
    PTXNodeType.PTX_NOT: _create_not,
    PTXNodeType.PTX_SETP: _create_setp,
    PTXNodeType.PTX_SELP: _create_selp,
    PTXNodeType.PTX_BRA: _create_bra,
    PTXNodeType.PTX_LABEL: _create_label,
    PTXNodeType.PTX_BAR: _create_bar,
    PTXNodeType.PTX_MEMBAR: _create_membar,
    PTXNodeType.PTX_SHFL: _create_shfl,
    PTXNodeType.PTX_VOTE: _create_vote,
    PTXNodeType.PTX_MATCH: _create_match,
    PTXNodeType.PTX_REDUX: _create_redux,
    PTXNodeType.PTX_MMA: _create_mma,
    PTXNodeType.PTX_LDMATRIX: _create_ldmatrix,
    PTXNodeType.PTX_STMATRIX: _create_stmatrix,
    PTXNodeType.PTX_PRMT: _create_prmt,
    PTXNodeType.PTX_LOP3: _create_lop3,
    PTXNodeType.PTX_DP4A: _create_dp4a,
    PTXNodeType.PTX_DP2A: _create_dp2a,
}


def get_ptx_value_names(node_type: PTXNodeType) -> List[str]:
    """Get the operand names needed for a PTX node type."""
    spec = get_ptx_spec(node_type)
    return [op.name for op in spec.operands]


def get_ptx_value_types(node_type: PTXNodeType) -> List[PTXValueType]:
    """Get the operand types needed for a PTX node type."""
    spec = get_ptx_spec(node_type)
    return [op.value_type for op in spec.operands]
