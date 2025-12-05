"""
AST Actions - Instantiate cuda_ast nodes from grammar actions

Given a NodeType and values, create the corresponding cuda_ast node.
No hardcoded values - everything comes from the NN's outputs.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).parent.parent))

from cuda_ast import (
    CudaNode, CudaModule, Include, Define, Pragma, Comment, RawCode, BlankLine,
    TypeRef, Variable, Parameter, StructField, UsingDecl, Struct,
    LaunchBounds, FunctionDecl, Function, Statement, VarDecl, Return,
    If, For, While, Block, InlineAsm, StaticAssert, Constexpr, Lambda,
    FunctionQualifier, StorageClass, Variable
)
from gen.grammar import NodeType, ValueType, NODE_SPECS, get_node_spec


def create_node(node_type: NodeType, values: Dict[str, Any]) -> Optional[CudaNode]:
    """
    Create a cuda_ast node from a NodeType and values.

    Args:
        node_type: The type of node to create
        values: Dictionary mapping value names to their values

    Returns:
        The created CudaNode, or None for END
    """
    if node_type == NodeType.END:
        return None

    creator = NODE_CREATORS.get(node_type)
    if creator is None:
        raise ValueError(f"No creator for node type: {node_type}")

    return creator(values)


# =============================================================================
# NODE CREATORS
# Each takes a values dict and returns the corresponding cuda_ast node
# =============================================================================

def _create_include(values: Dict[str, Any]) -> Include:
    return Include(
        path=values.get("path", ""),
        is_system=values.get("is_system", False)
    )


def _create_define(values: Dict[str, Any]) -> Define:
    return Define(
        name=values.get("name", ""),
        value=values.get("value")
    )


def _create_pragma(values: Dict[str, Any]) -> Pragma:
    return Pragma(content=values.get("content", ""))


def _create_comment(values: Dict[str, Any]) -> Comment:
    return Comment(text=values.get("text", ""))


def _create_raw_code(values: Dict[str, Any]) -> RawCode:
    return RawCode(text=values.get("text", ""))


def _create_blank_line(values: Dict[str, Any]) -> BlankLine:
    return BlankLine()


def _create_struct(values: Dict[str, Any]) -> Struct:
    return Struct(
        name=values.get("name", ""),
        members=[]  # Members added via slots
    )


def _create_function(values: Dict[str, Any]) -> Function:
    # Parse qualifier
    qualifier_str = values.get("qualifier", "")
    qualifier = None
    if qualifier_str:
        try:
            qualifier = FunctionQualifier(qualifier_str)
        except ValueError:
            pass

    return Function(
        name=values.get("name", ""),
        return_type=TypeRef(values.get("return_type", "void")),
        params=[],  # Params added via slots
        body=[],    # Body added via slots
        qualifier=qualifier,
        launch_bounds=None  # Set separately if needed
    )


def _create_param(values: Dict[str, Any]) -> Parameter:
    return Parameter(
        name=values.get("name", ""),
        type=TypeRef(values.get("type", "int"))
    )


def _create_struct_field(values: Dict[str, Any]) -> StructField:
    return StructField(
        name=values.get("name", ""),
        type=TypeRef(values.get("type", "int"))
    )


def _create_using_decl(values: Dict[str, Any]) -> UsingDecl:
    return UsingDecl(
        name=values.get("name", ""),
        type_expr=values.get("type_expr", "")
    )


def _create_statement(values: Dict[str, Any]) -> Statement:
    return Statement(expr=values.get("expr", ""))


def _create_var_decl(values: Dict[str, Any]) -> VarDecl:
    return VarDecl(
        var=Variable(
            name=values.get("name", ""),
            type=TypeRef(values.get("type", "int")),
            initializer=values.get("initializer")
        )
    )


def _create_return(values: Dict[str, Any]) -> Return:
    return Return(value=values.get("value"))


def _create_if(values: Dict[str, Any]) -> If:
    return If(
        condition=values.get("condition", "true"),
        then_body=[],  # Added via slots
        else_body=[]   # Added via slots
    )


def _create_for(values: Dict[str, Any]) -> For:
    return For(
        init=values.get("init", ""),
        condition=values.get("condition", ""),
        increment=values.get("increment", ""),
        body=[]  # Added via slots
    )


def _create_while(values: Dict[str, Any]) -> While:
    return While(
        condition=values.get("condition", "true"),
        body=[]  # Added via slots
    )


def _create_block(values: Dict[str, Any]) -> Block:
    return Block(body=[])  # Added via slots


def _create_inline_asm(values: Dict[str, Any]) -> InlineAsm:
    ptx = values.get("ptx", "")
    return InlineAsm(ptx_lines=[ptx] if ptx else [])


def _create_static_assert(values: Dict[str, Any]) -> StaticAssert:
    return StaticAssert(
        condition=values.get("condition", "true"),
        message=values.get("message", "")
    )


def _create_constexpr(values: Dict[str, Any]) -> Constexpr:
    return Constexpr(
        name=values.get("name", ""),
        type=TypeRef(values.get("type", "int")),
        value=values.get("value", "0")
    )


def _create_launch_bounds(values: Dict[str, Any]) -> LaunchBounds:
    return LaunchBounds(
        max_threads=values.get("max_threads", 256),
        min_blocks=values.get("min_blocks")
    )


# Map NodeType to creator function
NODE_CREATORS = {
    NodeType.INCLUDE: _create_include,
    NodeType.DEFINE: _create_define,
    NodeType.PRAGMA: _create_pragma,
    NodeType.COMMENT: _create_comment,
    NodeType.RAW_CODE: _create_raw_code,
    NodeType.BLANK_LINE: _create_blank_line,
    NodeType.STRUCT: _create_struct,
    NodeType.FUNCTION: _create_function,
    NodeType.PARAM: _create_param,
    NodeType.STRUCT_FIELD: _create_struct_field,
    NodeType.USING_DECL: _create_using_decl,
    NodeType.STATEMENT: _create_statement,
    NodeType.VAR_DECL: _create_var_decl,
    NodeType.RETURN: _create_return,
    NodeType.IF: _create_if,
    NodeType.FOR: _create_for,
    NodeType.WHILE: _create_while,
    NodeType.BLOCK: _create_block,
    NodeType.INLINE_ASM: _create_inline_asm,
    NodeType.STATIC_ASSERT: _create_static_assert,
    NodeType.CONSTEXPR: _create_constexpr,
    NodeType.LAUNCH_BOUNDS: _create_launch_bounds,
}


def get_value_names(node_type: NodeType) -> List[str]:
    """Get the value names needed for a node type."""
    spec = get_node_spec(node_type)
    return spec.value_names


def get_value_types(node_type: NodeType) -> List[ValueType]:
    """Get the value types needed for a node type."""
    spec = get_node_spec(node_type)
    return spec.value_types
