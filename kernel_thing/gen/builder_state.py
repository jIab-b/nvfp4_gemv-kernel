"""
Builder State - Tracks partial AST during generation

Manages:
- The AST being built (CudaModule with nodes)
- Context stack (what scope we're in)
- Valid actions at current position
- Encoding for NN input
- PTX mode: when inside INLINE_ASM, emits PTX instructions
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import IntEnum

sys.path.insert(0, str(Path(__file__).parent.parent))

from cuda_ast import (
    CudaNode, CudaModule, Function, Struct, If, For, While, Block,
    Parameter, StructField, InlineAsm
)
from ptx_ast import (
    PTXModule, Instruction, Directive, Label, RegisterDecl, SharedDecl,
    Statement as PTXStatement
)
from gen.grammar import (
    NodeType, ValueType, NUM_NODE_TYPES, NODE_SPECS,
    get_valid_next_nodes, get_node_spec,
    MODULE_LEVEL, FUNCTION_BODY, STRUCT_BODY, CONTROL_BODY
)
from gen.ast_actions import create_node
from gen.ptx_grammar import (
    PTXNodeType, PTXValueType, NUM_PTX_NODE_TYPES, PTX_SPECS,
    get_ptx_spec, get_valid_ptx_nodes, PTX_BODY
)
from gen.ptx_actions import create_ptx_node


class SlotType(IntEnum):
    """What kind of slot we're filling."""
    MODULE = 0       # Top-level module
    FUNCTION_BODY = 1
    FUNCTION_PARAMS = 2
    STRUCT_MEMBERS = 3
    IF_THEN = 4
    IF_ELSE = 5
    FOR_BODY = 6
    WHILE_BODY = 7
    BLOCK_BODY = 8
    PTX_BODY = 9     # Inside inline asm


@dataclass
class ContextFrame:
    """A frame in the context stack."""
    node: CudaNode           # The node we're inside
    node_type: NodeType      # Type of that node
    slot_type: SlotType      # Which slot we're filling
    slot_name: str           # Name of the slot


@dataclass
class PTXContext:
    """PTX-specific state when inside INLINE_ASM."""
    # The InlineAsm node we're building
    asm_node: InlineAsm

    # PTX statements being built
    ptx_statements: List[PTXStatement] = field(default_factory=list)

    # Register allocation tracking
    reg_counters: Dict[str, int] = field(default_factory=lambda: {
        "b8": 0, "b16": 0, "b32": 0, "b64": 0,
        "f16": 0, "f16x2": 0, "f32": 0, "f64": 0,
        "pred": 0,
    })

    # Declared registers (dtype -> list of names)
    declared_regs: Dict[str, List[str]] = field(default_factory=dict)

    # Labels defined
    labels: Set[str] = field(default_factory=set)

    def alloc_reg(self, dtype: str) -> str:
        """Allocate a new register of given type."""
        dtype = dtype.lstrip('.')
        base = dtype[0] if dtype != "pred" else "p"
        if dtype.startswith("f"):
            base = "f" if dtype in ("f32", "f64") else "h"
        elif dtype == "b64" or dtype == "u64" or dtype == "s64":
            base = "rd"
        elif dtype == "b32" or dtype == "u32" or dtype == "s32":
            base = "r"

        idx = self.reg_counters.get(dtype, 0)
        self.reg_counters[dtype] = idx + 1
        name = f"%{base}{idx}"

        if dtype not in self.declared_regs:
            self.declared_regs[dtype] = []
        self.declared_regs[dtype].append(name)

        return name

    def get_ptx_lines(self) -> List[str]:
        """Get PTX as list of lines for InlineAsm."""
        lines = []
        for stmt in self.ptx_statements:
            lines.append(stmt.emit())
        return lines


@dataclass
class BuilderState:
    """
    Tracks state of AST generation.

    The NN sees:
    - Current context (what scope we're in)
    - Valid next actions
    - Some representation of what's been built

    Actions:
    - (node_type, values) to add a CUDA node
    - (ptx_node_type, values) to add a PTX node (when in PTX mode)
    - END to close current scope
    """

    # The module being built
    module: CudaModule = field(default_factory=lambda: CudaModule(nodes=[]))

    # Stack of contexts (what scopes we're in)
    context_stack: List[ContextFrame] = field(default_factory=list)

    # PTX context (when inside INLINE_ASM)
    ptx_context: Optional[PTXContext] = None

    # History of actions taken (for encoding)
    # Each entry is (is_ptx, node_type_value, values_dict)
    action_history: List[Tuple[bool, int, Dict[str, Any]]] = field(default_factory=list)

    # Is generation complete?
    done: bool = False

    def in_ptx_mode(self) -> bool:
        """Are we currently inside an INLINE_ASM block?"""
        return self.ptx_context is not None

    def get_valid_node_types(self) -> Set[NodeType]:
        """Get CUDA node types valid at current position (not PTX mode)."""
        if self.done or self.in_ptx_mode():
            return set()

        if not self.context_stack:
            # At module level
            return MODULE_LEVEL | {NodeType.END}

        frame = self.context_stack[-1]

        # Get valid nodes for this slot type
        if frame.slot_type == SlotType.MODULE:
            return MODULE_LEVEL | {NodeType.END}
        elif frame.slot_type == SlotType.FUNCTION_BODY:
            return FUNCTION_BODY
        elif frame.slot_type == SlotType.FUNCTION_PARAMS:
            return {NodeType.PARAM, NodeType.END}
        elif frame.slot_type == SlotType.STRUCT_MEMBERS:
            return STRUCT_BODY
        elif frame.slot_type in (SlotType.IF_THEN, SlotType.IF_ELSE,
                                  SlotType.FOR_BODY, SlotType.WHILE_BODY,
                                  SlotType.BLOCK_BODY):
            return CONTROL_BODY

        return {NodeType.END}

    def get_valid_ptx_types(self) -> Set[PTXNodeType]:
        """Get PTX node types valid at current position (in PTX mode)."""
        if not self.in_ptx_mode():
            return set()
        return PTX_BODY

    def add_node(self, node_type: NodeType, values: Dict[str, Any]) -> bool:
        """
        Add a CUDA node to the AST.

        Args:
            node_type: Type of CUDA node to add
            values: Values for the node

        Returns:
            True if successful, False if invalid
        """
        if self.done or self.in_ptx_mode():
            return False

        # Check if this node type is valid
        valid = self.get_valid_node_types()
        if node_type not in valid:
            return False

        # Handle END - close current scope
        if node_type == NodeType.END:
            return self._close_scope()

        # Create the node
        node = create_node(node_type, values)
        if node is None:
            return False

        # Add to appropriate place
        if not self.context_stack:
            # Module level
            self.module.nodes.append(node)
        else:
            frame = self.context_stack[-1]
            self._add_to_slot(frame, node)

        # Record action
        self.action_history.append((False, node_type.value, values))

        # If this node has slots, push context for first slot
        self._maybe_push_context(node, node_type)

        # Special: if INLINE_ASM, enter PTX mode
        if node_type == NodeType.INLINE_ASM:
            self.ptx_context = PTXContext(asm_node=node)

        return True

    def add_ptx_node(self, ptx_type: PTXNodeType, values: Dict[str, Any]) -> bool:
        """
        Add a PTX instruction/directive to current INLINE_ASM block.

        Args:
            ptx_type: Type of PTX node to add
            values: Values for the node (registers, immediates, etc.)

        Returns:
            True if successful, False if invalid
        """
        if not self.in_ptx_mode():
            return False

        # Handle PTX_END - close PTX block
        if ptx_type == PTXNodeType.PTX_END:
            return self._close_ptx_block()

        # Create the PTX node
        ptx_node = create_ptx_node(ptx_type, values)
        if ptx_node is None:
            return False

        # Add to PTX context
        self.ptx_context.ptx_statements.append(ptx_node)

        # Track labels
        if isinstance(ptx_node, Label):
            self.ptx_context.labels.add(ptx_node.name)

        # Track register declarations
        if isinstance(ptx_node, RegisterDecl):
            dtype = ptx_node.dtype
            if dtype not in self.ptx_context.declared_regs:
                self.ptx_context.declared_regs[dtype] = []
            self.ptx_context.declared_regs[dtype].extend(ptx_node.names)

        # Record action
        self.action_history.append((True, ptx_type.value, values))

        return True

    def _close_ptx_block(self) -> bool:
        """Close PTX block and finalize InlineAsm node."""
        if not self.in_ptx_mode():
            return False

        # Get PTX lines
        ptx_lines = self.ptx_context.get_ptx_lines()

        # Update the InlineAsm node
        self.ptx_context.asm_node.ptx_lines = ptx_lines

        # Record action
        self.action_history.append((True, PTXNodeType.PTX_END.value, {}))

        # Exit PTX mode
        self.ptx_context = None

        return True

    def _add_to_slot(self, frame: ContextFrame, node: CudaNode) -> None:
        """Add a node to the current slot."""
        parent = frame.node

        if frame.slot_type == SlotType.FUNCTION_BODY:
            parent.body.append(node)
        elif frame.slot_type == SlotType.FUNCTION_PARAMS:
            parent.params.append(node)
        elif frame.slot_type == SlotType.STRUCT_MEMBERS:
            parent.members.append(node)
        elif frame.slot_type == SlotType.IF_THEN:
            parent.then_body.append(node)
        elif frame.slot_type == SlotType.IF_ELSE:
            parent.else_body.append(node)
        elif frame.slot_type in (SlotType.FOR_BODY, SlotType.WHILE_BODY, SlotType.BLOCK_BODY):
            parent.body.append(node)

    def _maybe_push_context(self, node: CudaNode, node_type: NodeType) -> None:
        """Push context if this node has slots to fill."""
        spec = get_node_spec(node_type)

        # INLINE_ASM is special - don't push context, we enter PTX mode instead
        if node_type == NodeType.INLINE_ASM:
            return

        if not spec.slots:
            return

        # Push context for first slot
        first_slot = spec.slots[0]
        slot_type = self._slot_name_to_type(node_type, first_slot.name)

        self.context_stack.append(ContextFrame(
            node=node,
            node_type=node_type,
            slot_type=slot_type,
            slot_name=first_slot.name
        ))

    def _slot_name_to_type(self, node_type: NodeType, slot_name: str) -> SlotType:
        """Convert slot name to SlotType."""
        if node_type == NodeType.FUNCTION:
            if slot_name == "params":
                return SlotType.FUNCTION_PARAMS
            return SlotType.FUNCTION_BODY
        elif node_type == NodeType.STRUCT:
            return SlotType.STRUCT_MEMBERS
        elif node_type == NodeType.IF:
            if slot_name == "then_body":
                return SlotType.IF_THEN
            return SlotType.IF_ELSE
        elif node_type == NodeType.FOR:
            return SlotType.FOR_BODY
        elif node_type == NodeType.WHILE:
            return SlotType.WHILE_BODY
        elif node_type == NodeType.BLOCK:
            return SlotType.BLOCK_BODY

        return SlotType.MODULE

    def _close_scope(self) -> bool:
        """Close current scope (END action)."""
        if not self.context_stack:
            # Closing module level = done
            self.done = True
            self.action_history.append((False, NodeType.END.value, {}))
            return True

        frame = self.context_stack[-1]
        spec = get_node_spec(frame.node_type)

        # Find current slot index
        current_idx = 0
        for i, slot in enumerate(spec.slots):
            if slot.name == frame.slot_name:
                current_idx = i
                break

        # Check if there's a next slot
        if current_idx + 1 < len(spec.slots):
            # Move to next slot
            next_slot = spec.slots[current_idx + 1]
            self.context_stack[-1] = ContextFrame(
                node=frame.node,
                node_type=frame.node_type,
                slot_type=self._slot_name_to_type(frame.node_type, next_slot.name),
                slot_name=next_slot.name
            )
        else:
            # Pop this context
            self.context_stack.pop()

        self.action_history.append((False, NodeType.END.value, {}))
        return True

    def get_context_depth(self) -> int:
        """Get nesting depth."""
        return len(self.context_stack)

    def get_current_context_type(self) -> Optional[NodeType]:
        """Get type of current context node."""
        if not self.context_stack:
            return None
        return self.context_stack[-1].node_type

    def emit(self) -> str:
        """Emit the built AST as CUDA source."""
        return self.module.emit()

    def encode_state(self) -> Dict[str, Any]:
        """
        Encode state for NN input.

        Returns dict with:
        - context_depth: int
        - current_context: NodeType or -1
        - in_ptx_mode: bool
        - valid_mask: bool[NUM_NODE_TYPES] - which CUDA node types are valid
        - valid_ptx_mask: bool[NUM_PTX_NODE_TYPES] - which PTX node types are valid
        - recent_actions: list of recent (is_ptx, node_type, has_values)
        - ptx_reg_counts: dict of register counts (when in PTX mode)
        """
        valid = self.get_valid_node_types()
        valid_mask = [nt in valid for nt in NodeType]

        valid_ptx = self.get_valid_ptx_types()
        valid_ptx_mask = [pt in valid_ptx for pt in PTXNodeType]

        # Last N actions
        recent = [(is_ptx, nt, len(vals) > 0)
                  for is_ptx, nt, vals in self.action_history[-16:]]

        # PTX register counts
        ptx_reg_counts = {}
        if self.ptx_context:
            ptx_reg_counts = dict(self.ptx_context.reg_counters)

        return {
            "context_depth": self.get_context_depth(),
            "current_context": (self.context_stack[-1].node_type.value
                               if self.context_stack else -1),
            "current_slot": (self.context_stack[-1].slot_type.value
                            if self.context_stack else -1),
            "in_ptx_mode": self.in_ptx_mode(),
            "valid_mask": valid_mask,
            "valid_ptx_mask": valid_ptx_mask,
            "recent_actions": recent,
            "num_actions": len(self.action_history),
            "ptx_reg_counts": ptx_reg_counts,
            "ptx_num_instrs": len(self.ptx_context.ptx_statements) if self.ptx_context else 0,
            "done": self.done,
        }

    def clone(self) -> "BuilderState":
        """Create a deep copy of this state."""
        import copy
        return copy.deepcopy(self)


def test_builder():
    """Test the builder state with PTX."""
    state = BuilderState()

    # Add an include
    assert state.add_node(NodeType.INCLUDE, {"path": "cuda_runtime.h", "is_system": True})

    # Add a function
    assert state.add_node(NodeType.FUNCTION, {
        "name": "my_kernel",
        "return_type": "void",
        "qualifier": "__global__"
    })

    # Add a parameter
    assert state.add_node(NodeType.PARAM, {"name": "x", "type": "float*"})
    assert state.add_node(NodeType.END, {})  # close params

    # Add inline asm
    assert state.add_node(NodeType.INLINE_ASM, {})
    assert state.in_ptx_mode()

    # Add PTX instructions
    assert state.add_ptx_node(PTXNodeType.PTX_REG_DECL, {"dtype": ".b32", "names": "r0, r1, r2"})
    assert state.add_ptx_node(PTXNodeType.PTX_MOV, {"dtype": ".b32", "dst": "%r0", "src": "0"})
    assert state.add_ptx_node(PTXNodeType.PTX_ADD, {"dtype": ".s32", "dst": "%r2", "a": "%r0", "b": "%r1"})
    assert state.add_ptx_node(PTXNodeType.PTX_BAR, {"barrier_id": "0"})

    # Close PTX block
    assert state.add_ptx_node(PTXNodeType.PTX_END, {})
    assert not state.in_ptx_mode()

    # Close function body
    assert state.add_node(NodeType.END, {})

    # Close module
    assert state.add_node(NodeType.END, {})
    assert state.done

    # Emit and print
    code = state.emit()
    print("Generated code with PTX:")
    print(code)
    print()
    print("State encoding:")
    enc = state.encode_state()
    print(f"  in_ptx_mode: {enc['in_ptx_mode']}")
    print(f"  num_actions: {enc['num_actions']}")
    print(f"  done: {enc['done']}")

    return state


def test_ptx_only():
    """Test just the PTX generation."""
    state = BuilderState()

    # Minimal wrapper
    state.add_node(NodeType.FUNCTION, {"name": "k", "return_type": "void", "qualifier": "__global__"})
    state.add_node(NodeType.END, {})  # params

    # Enter PTX mode
    state.add_node(NodeType.INLINE_ASM, {})

    # Build a small reduction
    state.add_ptx_node(PTXNodeType.PTX_REG_DECL, {"dtype": ".b32", "names": "r0, r1"})
    state.add_ptx_node(PTXNodeType.PTX_REG_DECL, {"dtype": ".pred", "names": "p0"})
    state.add_ptx_node(PTXNodeType.PTX_LD_GLOBAL, {"dtype": ".u32", "dst": "%r0", "addr": "[%rd0]"})
    state.add_ptx_node(PTXNodeType.PTX_SHFL, {
        "dtype": ".b32", "mode": ".bfly",
        "dst": "%r1", "src": "%r0", "lane": "16", "mask": "0xffffffff"
    })
    state.add_ptx_node(PTXNodeType.PTX_ADD, {"dtype": ".s32", "dst": "%r0", "a": "%r0", "b": "%r1"})
    state.add_ptx_node(PTXNodeType.PTX_END, {})

    state.add_node(NodeType.END, {})  # body
    state.add_node(NodeType.END, {})  # module

    print("PTX kernel:")
    print(state.emit())


if __name__ == "__main__":
    test_builder()
    print("\n" + "="*50 + "\n")
    test_ptx_only()
