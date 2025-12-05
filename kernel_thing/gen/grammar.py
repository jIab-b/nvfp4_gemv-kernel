"""
CUDA AST Grammar for Neural Network Generation

Defines what nodes can follow what, and what values are valid.
This is derived directly from cuda_ast.py - no hardcoded kernel knowledge.

The grammar is:
- NodeType: what kind of AST node
- Slot: where in the parent to attach (e.g., function body, loop body)
- Value: for leaf nodes, what value (e.g., identifier name, integer literal)
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from exec.cuda_ast import (
    CudaNode, CudaModule, Include, Define, Pragma, Comment, RawCode, BlankLine,
    TypeRef, Variable, Parameter, StructField, UsingDecl, Struct,
    LaunchBounds, FunctionDecl, Function, Statement, VarDecl, Return,
    If, For, While, Block, InlineAsm, StaticAssert, Constexpr, Lambda,
    FunctionQualifier, StorageClass
)


class NodeType(IntEnum):
    """
    All AST node types the NN can emit.

    This is the complete set from cuda_ast.py.
    """
    # Special
    END = 0  # Done building current scope

    # Module level
    INCLUDE = auto()
    DEFINE = auto()
    PRAGMA = auto()
    COMMENT = auto()
    RAW_CODE = auto()
    BLANK_LINE = auto()
    STRUCT = auto()
    FUNCTION = auto()

    # Type system
    TYPE_REF = auto()

    # Inside struct
    STRUCT_FIELD = auto()
    USING_DECL = auto()

    # Inside function
    STATEMENT = auto()
    VAR_DECL = auto()
    RETURN = auto()
    IF = auto()
    FOR = auto()
    WHILE = auto()
    BLOCK = auto()
    INLINE_ASM = auto()
    STATIC_ASSERT = auto()
    CONSTEXPR = auto()

    # Function attributes
    LAUNCH_BOUNDS = auto()
    PARAM = auto()


NUM_NODE_TYPES = len(NodeType)


class ValueType(IntEnum):
    """
    Types of values the NN must provide for nodes.
    """
    NONE = 0           # Node needs no value
    IDENTIFIER = auto()  # Variable/function name
    TYPE_NAME = auto()   # Type name (int, float, etc.)
    INTEGER = auto()     # Integer literal
    STRING = auto()      # String literal
    EXPRESSION = auto()  # C++ expression (most flexible)
    QUALIFIER = auto()   # Function qualifier (__global__, etc.)
    STORAGE = auto()     # Storage class (static, etc.)
    BOOLEAN = auto()     # True/False


@dataclass
class SlotInfo:
    """Describes a slot where child nodes can be added."""
    name: str
    accepts: Set[NodeType]  # What node types can go here
    is_list: bool = True    # Can hold multiple nodes
    required: bool = False  # Must have at least one


@dataclass
class NodeSpec:
    """Specification for a node type."""
    node_type: NodeType
    value_types: List[ValueType]  # What values this node needs
    value_names: List[str]        # Names for the values
    slots: List[SlotInfo]         # Child slots
    valid_parents: Set[NodeType]  # Where this node can appear


# =============================================================================
# GRAMMAR DEFINITION
# =============================================================================

# What nodes can appear at module (top) level
MODULE_LEVEL = {
    NodeType.INCLUDE, NodeType.DEFINE, NodeType.PRAGMA,
    NodeType.COMMENT, NodeType.RAW_CODE, NodeType.BLANK_LINE,
    NodeType.STRUCT, NodeType.FUNCTION
}

# What nodes can appear in function body
FUNCTION_BODY = {
    NodeType.STATEMENT, NodeType.VAR_DECL, NodeType.RETURN,
    NodeType.IF, NodeType.FOR, NodeType.WHILE, NodeType.BLOCK,
    NodeType.INLINE_ASM, NodeType.STATIC_ASSERT, NodeType.CONSTEXPR,
    NodeType.COMMENT, NodeType.RAW_CODE, NodeType.BLANK_LINE,
    NodeType.END
}

# What nodes can appear in struct body
STRUCT_BODY = {
    NodeType.STRUCT_FIELD, NodeType.USING_DECL,
    NodeType.COMMENT, NodeType.RAW_CODE,
    NodeType.END
}

# What nodes can appear in control flow body (if/for/while/block)
CONTROL_BODY = FUNCTION_BODY


# Node specifications
NODE_SPECS: Dict[NodeType, NodeSpec] = {
    NodeType.END: NodeSpec(
        node_type=NodeType.END,
        value_types=[],
        value_names=[],
        slots=[],
        valid_parents=MODULE_LEVEL | FUNCTION_BODY | STRUCT_BODY
    ),

    NodeType.INCLUDE: NodeSpec(
        node_type=NodeType.INCLUDE,
        value_types=[ValueType.STRING, ValueType.BOOLEAN],
        value_names=["path", "is_system"],
        slots=[],
        valid_parents={NodeType.END}  # Only at module level
    ),

    NodeType.DEFINE: NodeSpec(
        node_type=NodeType.DEFINE,
        value_types=[ValueType.IDENTIFIER, ValueType.EXPRESSION],
        value_names=["name", "value"],
        slots=[],
        valid_parents={NodeType.END}
    ),

    NodeType.PRAGMA: NodeSpec(
        node_type=NodeType.PRAGMA,
        value_types=[ValueType.STRING],
        value_names=["content"],
        slots=[],
        valid_parents={NodeType.END} | FUNCTION_BODY
    ),

    NodeType.COMMENT: NodeSpec(
        node_type=NodeType.COMMENT,
        value_types=[ValueType.STRING],
        value_names=["text"],
        slots=[],
        valid_parents={NodeType.END} | FUNCTION_BODY | STRUCT_BODY
    ),

    NodeType.RAW_CODE: NodeSpec(
        node_type=NodeType.RAW_CODE,
        value_types=[ValueType.STRING],
        value_names=["text"],
        slots=[],
        valid_parents={NodeType.END} | FUNCTION_BODY | STRUCT_BODY
    ),

    NodeType.BLANK_LINE: NodeSpec(
        node_type=NodeType.BLANK_LINE,
        value_types=[],
        value_names=[],
        slots=[],
        valid_parents={NodeType.END} | FUNCTION_BODY
    ),

    NodeType.STRUCT: NodeSpec(
        node_type=NodeType.STRUCT,
        value_types=[ValueType.IDENTIFIER],
        value_names=["name"],
        slots=[
            SlotInfo("members", STRUCT_BODY, is_list=True, required=False)
        ],
        valid_parents={NodeType.END}
    ),

    NodeType.FUNCTION: NodeSpec(
        node_type=NodeType.FUNCTION,
        value_types=[ValueType.IDENTIFIER, ValueType.TYPE_NAME, ValueType.QUALIFIER],
        value_names=["name", "return_type", "qualifier"],
        slots=[
            SlotInfo("params", {NodeType.PARAM}, is_list=True, required=False),
            SlotInfo("body", FUNCTION_BODY, is_list=True, required=False)
        ],
        valid_parents={NodeType.END}
    ),

    NodeType.PARAM: NodeSpec(
        node_type=NodeType.PARAM,
        value_types=[ValueType.IDENTIFIER, ValueType.TYPE_NAME],
        value_names=["name", "type"],
        slots=[],
        valid_parents={NodeType.FUNCTION}
    ),

    NodeType.STRUCT_FIELD: NodeSpec(
        node_type=NodeType.STRUCT_FIELD,
        value_types=[ValueType.IDENTIFIER, ValueType.TYPE_NAME],
        value_names=["name", "type"],
        slots=[],
        valid_parents={NodeType.STRUCT}
    ),

    NodeType.USING_DECL: NodeSpec(
        node_type=NodeType.USING_DECL,
        value_types=[ValueType.IDENTIFIER, ValueType.TYPE_NAME],
        value_names=["name", "type_expr"],
        slots=[],
        valid_parents={NodeType.STRUCT}
    ),

    NodeType.STATEMENT: NodeSpec(
        node_type=NodeType.STATEMENT,
        value_types=[ValueType.EXPRESSION],
        value_names=["expr"],
        slots=[],
        valid_parents=FUNCTION_BODY
    ),

    NodeType.VAR_DECL: NodeSpec(
        node_type=NodeType.VAR_DECL,
        value_types=[ValueType.IDENTIFIER, ValueType.TYPE_NAME, ValueType.EXPRESSION],
        value_names=["name", "type", "initializer"],
        slots=[],
        valid_parents=FUNCTION_BODY
    ),

    NodeType.RETURN: NodeSpec(
        node_type=NodeType.RETURN,
        value_types=[ValueType.EXPRESSION],
        value_names=["value"],
        slots=[],
        valid_parents=FUNCTION_BODY
    ),

    NodeType.IF: NodeSpec(
        node_type=NodeType.IF,
        value_types=[ValueType.EXPRESSION],
        value_names=["condition"],
        slots=[
            SlotInfo("then_body", CONTROL_BODY, is_list=True, required=False),
            SlotInfo("else_body", CONTROL_BODY, is_list=True, required=False)
        ],
        valid_parents=FUNCTION_BODY
    ),

    NodeType.FOR: NodeSpec(
        node_type=NodeType.FOR,
        value_types=[ValueType.EXPRESSION, ValueType.EXPRESSION, ValueType.EXPRESSION],
        value_names=["init", "condition", "increment"],
        slots=[
            SlotInfo("body", CONTROL_BODY, is_list=True, required=False)
        ],
        valid_parents=FUNCTION_BODY
    ),

    NodeType.WHILE: NodeSpec(
        node_type=NodeType.WHILE,
        value_types=[ValueType.EXPRESSION],
        value_names=["condition"],
        slots=[
            SlotInfo("body", CONTROL_BODY, is_list=True, required=False)
        ],
        valid_parents=FUNCTION_BODY
    ),

    NodeType.BLOCK: NodeSpec(
        node_type=NodeType.BLOCK,
        value_types=[],
        value_names=[],
        slots=[
            SlotInfo("body", CONTROL_BODY, is_list=True, required=False)
        ],
        valid_parents=FUNCTION_BODY
    ),

    NodeType.INLINE_ASM: NodeSpec(
        node_type=NodeType.INLINE_ASM,
        value_types=[],  # No values - PTX instructions added as children
        value_names=[],
        slots=[
            SlotInfo("ptx_body", set(), is_list=True, required=False)  # Accepts PTX nodes (handled specially)
        ],
        valid_parents=FUNCTION_BODY
    ),

    NodeType.STATIC_ASSERT: NodeSpec(
        node_type=NodeType.STATIC_ASSERT,
        value_types=[ValueType.EXPRESSION, ValueType.STRING],
        value_names=["condition", "message"],
        slots=[],
        valid_parents=FUNCTION_BODY
    ),

    NodeType.CONSTEXPR: NodeSpec(
        node_type=NodeType.CONSTEXPR,
        value_types=[ValueType.IDENTIFIER, ValueType.TYPE_NAME, ValueType.EXPRESSION],
        value_names=["name", "type", "value"],
        slots=[],
        valid_parents=FUNCTION_BODY
    ),

    NodeType.LAUNCH_BOUNDS: NodeSpec(
        node_type=NodeType.LAUNCH_BOUNDS,
        value_types=[ValueType.INTEGER, ValueType.INTEGER],
        value_names=["max_threads", "min_blocks"],
        slots=[],
        valid_parents={NodeType.FUNCTION}
    ),
}


def get_valid_next_nodes(parent_type: Optional[NodeType], slot_name: str = "body") -> Set[NodeType]:
    """
    Get valid node types that can be added given current context.

    Args:
        parent_type: Type of parent node (None for module level)
        slot_name: Which slot we're adding to

    Returns:
        Set of valid NodeType values
    """
    if parent_type is None:
        return MODULE_LEVEL | {NodeType.END}

    spec = NODE_SPECS.get(parent_type)
    if spec is None:
        return {NodeType.END}

    for slot in spec.slots:
        if slot.name == slot_name:
            return slot.accepts | {NodeType.END}

    return {NodeType.END}


def get_node_spec(node_type: NodeType) -> NodeSpec:
    """Get specification for a node type."""
    return NODE_SPECS.get(node_type, NODE_SPECS[NodeType.END])
