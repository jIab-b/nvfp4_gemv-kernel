"""
PTX Grammar - Node types for PTX instructions

When inside an INLINE_ASM block, the NN can emit these PTX-specific node types.
Each maps to a ptx_ast node.

Design:
- PTX instructions are organized by category (load, store, arith, etc.)
- Each instruction type specifies what operands it needs
- Register allocation is tracked by builder_state
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple


class PTXNodeType(IntEnum):
    """
    PTX instruction/directive types the NN can emit inside INLINE_ASM.
    """
    # Special
    PTX_END = 0  # Done with PTX block

    # Directives (must come first in asm block)
    PTX_REG_DECL = auto()      # .reg .b32 r0, r1, r2
    PTX_SHARED_DECL = auto()   # .shared .b8 smem[4096]

    # Data movement
    PTX_MOV = auto()           # mov.b32 dst, src
    PTX_LD_GLOBAL = auto()     # ld.global.u32 dst, [addr]
    PTX_LD_SHARED = auto()     # ld.shared.u32 dst, [addr]
    PTX_LD_PARAM = auto()      # ld.param.u64 dst, [param]
    PTX_ST_GLOBAL = auto()     # st.global.u32 [addr], src
    PTX_ST_SHARED = auto()     # st.shared.u32 [addr], src

    # Integer arithmetic
    PTX_ADD = auto()           # add.s32 dst, a, b
    PTX_SUB = auto()           # sub.s32 dst, a, b
    PTX_MUL = auto()           # mul.lo.s32 dst, a, b
    PTX_MAD = auto()           # mad.lo.s32 dst, a, b, c
    PTX_SHL = auto()           # shl.b32 dst, src, amt
    PTX_SHR = auto()           # shr.b32 dst, src, amt

    # Float arithmetic
    PTX_ADD_F = auto()         # add.rn.f32 dst, a, b
    PTX_SUB_F = auto()         # sub.rn.f32 dst, a, b
    PTX_MUL_F = auto()         # mul.rn.f32 dst, a, b
    PTX_FMA = auto()           # fma.rn.f32 dst, a, b, c
    PTX_DIV_F = auto()         # div.rn.f32 dst, a, b

    # Conversions
    PTX_CVT = auto()           # cvt.rn.f16.f32 dst, src

    # Bitwise
    PTX_AND = auto()           # and.b32 dst, a, b
    PTX_OR = auto()            # or.b32 dst, a, b
    PTX_XOR = auto()           # xor.b32 dst, a, b
    PTX_NOT = auto()           # not.b32 dst, src

    # Comparison / predicates
    PTX_SETP = auto()          # setp.lt.s32 p, a, b
    PTX_SELP = auto()          # selp.b32 dst, a, b, p

    # Control flow
    PTX_BRA = auto()           # bra label
    PTX_LABEL = auto()         # $LOOP:

    # Synchronization
    PTX_BAR = auto()           # bar.sync 0
    PTX_MEMBAR = auto()        # membar.gl

    # Warp-level
    PTX_SHFL = auto()          # shfl.sync.bfly.b32 dst, src, lane, mask
    PTX_VOTE = auto()          # vote.sync.all.pred p, q, mask
    PTX_MATCH = auto()         # match.any.sync.b32 dst, src, mask
    PTX_REDUX = auto()         # redux.sync.add.s32 dst, src, mask

    # Tensor cores
    PTX_MMA = auto()           # mma.sync.aligned.m16n8k16...
    PTX_LDMATRIX = auto()      # ldmatrix.sync.aligned.m8n8...
    PTX_STMATRIX = auto()      # stmatrix.sync.aligned...

    # Special
    PTX_PRMT = auto()          # prmt.b32 dst, a, b, sel
    PTX_LOP3 = auto()          # lop3.b32 dst, a, b, c, lut
    PTX_DP4A = auto()          # dp4a.s32.s32 dst, a, b, c
    PTX_DP2A = auto()          # dp2a.lo.s32.s32 dst, a, b, c


NUM_PTX_NODE_TYPES = len(PTXNodeType)


class PTXValueType(IntEnum):
    """Types of values PTX nodes need."""
    NONE = 0
    REGISTER = auto()      # %r0, %rd5, a0
    IMMEDIATE = auto()     # 0, 16, 0xffffffff
    MEMORY = auto()        # [%rd0], [%rd0+16]
    LABEL = auto()         # $LOOP
    DTYPE = auto()         # .b32, .f16, .u64
    MODIFIER = auto()      # .rn, .sync, .lo, .hi
    PREDICATE = auto()     # %p0, !%p1
    VECTOR = auto()        # {a0, a1, a2, a3}


# PTX data types
PTX_DTYPES = [
    ".b8", ".b16", ".b32", ".b64",
    ".u8", ".u16", ".u32", ".u64",
    ".s8", ".s16", ".s32", ".s64",
    ".f16", ".f16x2", ".f32", ".f64",
    ".bf16", ".bf16x2", ".tf32",
    ".e4m3", ".e4m3x2", ".e5m2", ".e5m2x2",
    ".e2m1", ".e2m1x2",  # FP4
    ".pred",
]

# Common modifiers
PTX_MODIFIERS = [
    # Rounding
    ".rn", ".rz", ".rm", ".rp",
    # Memory hints
    ".cs", ".lu", ".cv", ".cg",
    ".L1::no_allocate", ".L1::evict_last", ".L1::evict_first",
    ".L2::128B", ".L2::256B", ".L2::evict_first", ".L2::evict_last",
    # Sync
    ".sync", ".aligned",
    # Mul/mad
    ".lo", ".hi", ".wide",
    # Vector
    ".v2", ".v4",
    # Branch
    ".uni",
    # Comparison
    ".eq", ".ne", ".lt", ".le", ".gt", ".ge",
    ".equ", ".neu", ".ltu", ".leu", ".gtu", ".geu",  # unsigned
    # Approx
    ".approx", ".ftz",
]


@dataclass
class PTXSlotInfo:
    """Describes a slot for PTX operands."""
    name: str
    value_type: PTXValueType
    required: bool = True
    is_list: bool = False  # For vector operands


@dataclass
class PTXNodeSpec:
    """Specification for a PTX node type."""
    node_type: PTXNodeType
    mnemonic: str              # Base PTX mnemonic
    required_modifiers: List[str] = field(default_factory=list)
    optional_modifiers: List[str] = field(default_factory=list)
    operands: List[PTXSlotInfo] = field(default_factory=list)
    description: str = ""


# =============================================================================
# PTX NODE SPECIFICATIONS
# =============================================================================

PTX_SPECS: Dict[PTXNodeType, PTXNodeSpec] = {
    PTXNodeType.PTX_END: PTXNodeSpec(
        node_type=PTXNodeType.PTX_END,
        mnemonic="",
        description="End PTX block"
    ),

    # --- Directives ---
    PTXNodeType.PTX_REG_DECL: PTXNodeSpec(
        node_type=PTXNodeType.PTX_REG_DECL,
        mnemonic=".reg",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("names", PTXValueType.REGISTER, is_list=True),
        ],
        description="Declare registers"
    ),

    PTXNodeType.PTX_SHARED_DECL: PTXNodeSpec(
        node_type=PTXNodeType.PTX_SHARED_DECL,
        mnemonic=".shared",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("name", PTXValueType.REGISTER),
            PTXSlotInfo("size", PTXValueType.IMMEDIATE),
        ],
        description="Declare shared memory"
    ),

    # --- Data movement ---
    PTXNodeType.PTX_MOV: PTXNodeSpec(
        node_type=PTXNodeType.PTX_MOV,
        mnemonic="mov",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("src", PTXValueType.REGISTER),  # or immediate
        ],
        description="Move data"
    ),

    PTXNodeType.PTX_LD_GLOBAL: PTXNodeSpec(
        node_type=PTXNodeType.PTX_LD_GLOBAL,
        mnemonic="ld.global",
        optional_modifiers=[".cs", ".lu", ".cv", ".L1::no_allocate", ".L2::128B", ".v2", ".v4"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("addr", PTXValueType.MEMORY),
        ],
        description="Load from global memory"
    ),

    PTXNodeType.PTX_LD_SHARED: PTXNodeSpec(
        node_type=PTXNodeType.PTX_LD_SHARED,
        mnemonic="ld.shared",
        optional_modifiers=[".v2", ".v4"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("addr", PTXValueType.MEMORY),
        ],
        description="Load from shared memory"
    ),

    PTXNodeType.PTX_LD_PARAM: PTXNodeSpec(
        node_type=PTXNodeType.PTX_LD_PARAM,
        mnemonic="ld.param",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("param", PTXValueType.MEMORY),
        ],
        description="Load kernel parameter"
    ),

    PTXNodeType.PTX_ST_GLOBAL: PTXNodeSpec(
        node_type=PTXNodeType.PTX_ST_GLOBAL,
        mnemonic="st.global",
        optional_modifiers=[".cs", ".v2", ".v4"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("addr", PTXValueType.MEMORY),
            PTXSlotInfo("src", PTXValueType.REGISTER),
        ],
        description="Store to global memory"
    ),

    PTXNodeType.PTX_ST_SHARED: PTXNodeSpec(
        node_type=PTXNodeType.PTX_ST_SHARED,
        mnemonic="st.shared",
        optional_modifiers=[".v2", ".v4"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("addr", PTXValueType.MEMORY),
            PTXSlotInfo("src", PTXValueType.REGISTER),
        ],
        description="Store to shared memory"
    ),

    # --- Integer arithmetic ---
    PTXNodeType.PTX_ADD: PTXNodeSpec(
        node_type=PTXNodeType.PTX_ADD,
        mnemonic="add",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
        ],
        description="Integer add"
    ),

    PTXNodeType.PTX_SUB: PTXNodeSpec(
        node_type=PTXNodeType.PTX_SUB,
        mnemonic="sub",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
        ],
        description="Integer subtract"
    ),

    PTXNodeType.PTX_MUL: PTXNodeSpec(
        node_type=PTXNodeType.PTX_MUL,
        mnemonic="mul",
        optional_modifiers=[".lo", ".hi", ".wide"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
        ],
        description="Integer multiply"
    ),

    PTXNodeType.PTX_MAD: PTXNodeSpec(
        node_type=PTXNodeType.PTX_MAD,
        mnemonic="mad",
        optional_modifiers=[".lo", ".hi", ".wide"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
            PTXSlotInfo("c", PTXValueType.REGISTER),
        ],
        description="Integer multiply-add"
    ),

    PTXNodeType.PTX_SHL: PTXNodeSpec(
        node_type=PTXNodeType.PTX_SHL,
        mnemonic="shl",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("src", PTXValueType.REGISTER),
            PTXSlotInfo("amt", PTXValueType.IMMEDIATE),
        ],
        description="Shift left"
    ),

    PTXNodeType.PTX_SHR: PTXNodeSpec(
        node_type=PTXNodeType.PTX_SHR,
        mnemonic="shr",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("src", PTXValueType.REGISTER),
            PTXSlotInfo("amt", PTXValueType.IMMEDIATE),
        ],
        description="Shift right"
    ),

    # --- Float arithmetic ---
    PTXNodeType.PTX_ADD_F: PTXNodeSpec(
        node_type=PTXNodeType.PTX_ADD_F,
        mnemonic="add",
        optional_modifiers=[".rn", ".rz", ".ftz"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
        ],
        description="Float add"
    ),

    PTXNodeType.PTX_MUL_F: PTXNodeSpec(
        node_type=PTXNodeType.PTX_MUL_F,
        mnemonic="mul",
        optional_modifiers=[".rn", ".rz", ".ftz"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
        ],
        description="Float multiply"
    ),

    PTXNodeType.PTX_FMA: PTXNodeSpec(
        node_type=PTXNodeType.PTX_FMA,
        mnemonic="fma",
        optional_modifiers=[".rn", ".rz", ".ftz"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
            PTXSlotInfo("c", PTXValueType.REGISTER),
        ],
        description="Float fused multiply-add"
    ),

    # --- Conversions ---
    PTXNodeType.PTX_CVT: PTXNodeSpec(
        node_type=PTXNodeType.PTX_CVT,
        mnemonic="cvt",
        optional_modifiers=[".rn", ".rz", ".sat"],
        operands=[
            PTXSlotInfo("dst_dtype", PTXValueType.DTYPE),
            PTXSlotInfo("src_dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("src", PTXValueType.REGISTER),
        ],
        description="Type conversion"
    ),

    # --- Bitwise ---
    PTXNodeType.PTX_AND: PTXNodeSpec(
        node_type=PTXNodeType.PTX_AND,
        mnemonic="and",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
        ],
        description="Bitwise AND"
    ),

    PTXNodeType.PTX_OR: PTXNodeSpec(
        node_type=PTXNodeType.PTX_OR,
        mnemonic="or",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
        ],
        description="Bitwise OR"
    ),

    PTXNodeType.PTX_XOR: PTXNodeSpec(
        node_type=PTXNodeType.PTX_XOR,
        mnemonic="xor",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
        ],
        description="Bitwise XOR"
    ),

    # --- Comparison ---
    PTXNodeType.PTX_SETP: PTXNodeSpec(
        node_type=PTXNodeType.PTX_SETP,
        mnemonic="setp",
        optional_modifiers=[".eq", ".ne", ".lt", ".le", ".gt", ".ge"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("pred", PTXValueType.PREDICATE),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
        ],
        description="Set predicate"
    ),

    PTXNodeType.PTX_SELP: PTXNodeSpec(
        node_type=PTXNodeType.PTX_SELP,
        mnemonic="selp",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
            PTXSlotInfo("pred", PTXValueType.PREDICATE),
        ],
        description="Select based on predicate"
    ),

    # --- Control flow ---
    PTXNodeType.PTX_BRA: PTXNodeSpec(
        node_type=PTXNodeType.PTX_BRA,
        mnemonic="bra",
        optional_modifiers=[".uni"],
        operands=[
            PTXSlotInfo("label", PTXValueType.LABEL),
        ],
        description="Branch"
    ),

    PTXNodeType.PTX_LABEL: PTXNodeSpec(
        node_type=PTXNodeType.PTX_LABEL,
        mnemonic="",  # Labels are special
        operands=[
            PTXSlotInfo("name", PTXValueType.LABEL),
        ],
        description="Label definition"
    ),

    # --- Synchronization ---
    PTXNodeType.PTX_BAR: PTXNodeSpec(
        node_type=PTXNodeType.PTX_BAR,
        mnemonic="bar",
        required_modifiers=[".sync"],
        operands=[
            PTXSlotInfo("barrier_id", PTXValueType.IMMEDIATE),
        ],
        description="Barrier synchronization"
    ),

    PTXNodeType.PTX_MEMBAR: PTXNodeSpec(
        node_type=PTXNodeType.PTX_MEMBAR,
        mnemonic="membar",
        optional_modifiers=[".cta", ".gl", ".sys"],
        operands=[],
        description="Memory barrier"
    ),

    # --- Warp-level ---
    PTXNodeType.PTX_SHFL: PTXNodeSpec(
        node_type=PTXNodeType.PTX_SHFL,
        mnemonic="shfl",
        required_modifiers=[".sync"],
        optional_modifiers=[".bfly", ".up", ".down", ".idx"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("src", PTXValueType.REGISTER),
            PTXSlotInfo("lane", PTXValueType.REGISTER),
            PTXSlotInfo("mask", PTXValueType.IMMEDIATE),
        ],
        description="Warp shuffle"
    ),

    PTXNodeType.PTX_REDUX: PTXNodeSpec(
        node_type=PTXNodeType.PTX_REDUX,
        mnemonic="redux",
        required_modifiers=[".sync"],
        optional_modifiers=[".add", ".min", ".max", ".and", ".or", ".xor"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("src", PTXValueType.REGISTER),
            PTXSlotInfo("mask", PTXValueType.IMMEDIATE),
        ],
        description="Warp reduction"
    ),

    # --- Tensor cores ---
    PTXNodeType.PTX_MMA: PTXNodeSpec(
        node_type=PTXNodeType.PTX_MMA,
        mnemonic="mma",
        required_modifiers=[".sync", ".aligned"],
        optional_modifiers=[".m16n8k16", ".m16n8k8", ".m8n8k4", ".row", ".col"],
        operands=[
            PTXSlotInfo("d_dtype", PTXValueType.DTYPE),
            PTXSlotInfo("a_dtype", PTXValueType.DTYPE),
            PTXSlotInfo("d", PTXValueType.VECTOR),
            PTXSlotInfo("a", PTXValueType.VECTOR),
            PTXSlotInfo("b", PTXValueType.VECTOR),
            PTXSlotInfo("c", PTXValueType.VECTOR),
        ],
        description="Matrix multiply-accumulate"
    ),

    PTXNodeType.PTX_LDMATRIX: PTXNodeSpec(
        node_type=PTXNodeType.PTX_LDMATRIX,
        mnemonic="ldmatrix",
        required_modifiers=[".sync", ".aligned"],
        optional_modifiers=[".m8n8", ".x1", ".x2", ".x4", ".trans"],
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.VECTOR),
            PTXSlotInfo("addr", PTXValueType.MEMORY),
        ],
        description="Load matrix from shared memory"
    ),

    # --- Special ops ---
    PTXNodeType.PTX_PRMT: PTXNodeSpec(
        node_type=PTXNodeType.PTX_PRMT,
        mnemonic="prmt",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
            PTXSlotInfo("sel", PTXValueType.IMMEDIATE),
        ],
        description="Byte permute"
    ),

    PTXNodeType.PTX_LOP3: PTXNodeSpec(
        node_type=PTXNodeType.PTX_LOP3,
        mnemonic="lop3",
        operands=[
            PTXSlotInfo("dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
            PTXSlotInfo("c", PTXValueType.REGISTER),
            PTXSlotInfo("lut", PTXValueType.IMMEDIATE),
        ],
        description="Three-input logic op"
    ),

    PTXNodeType.PTX_DP4A: PTXNodeSpec(
        node_type=PTXNodeType.PTX_DP4A,
        mnemonic="dp4a",
        operands=[
            PTXSlotInfo("d_dtype", PTXValueType.DTYPE),
            PTXSlotInfo("a_dtype", PTXValueType.DTYPE),
            PTXSlotInfo("dst", PTXValueType.REGISTER),
            PTXSlotInfo("a", PTXValueType.REGISTER),
            PTXSlotInfo("b", PTXValueType.REGISTER),
            PTXSlotInfo("c", PTXValueType.REGISTER),
        ],
        description="4-element dot product accumulate"
    ),
}


# What PTX nodes are valid inside an INLINE_ASM block
PTX_BODY = {nt for nt in PTXNodeType if nt != PTXNodeType.PTX_END} | {PTXNodeType.PTX_END}


def get_ptx_spec(node_type: PTXNodeType) -> PTXNodeSpec:
    """Get specification for a PTX node type."""
    return PTX_SPECS.get(node_type, PTX_SPECS[PTXNodeType.PTX_END])


def get_valid_ptx_nodes() -> Set[PTXNodeType]:
    """Get all valid PTX node types."""
    return PTX_BODY
