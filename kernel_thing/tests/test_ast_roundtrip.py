"""
Test that NN grammar system can build complete ASTs and emit valid code.

Tests:
1. Manual action sequences produce expected code
2. BuilderState + ast_actions roundtrip works
3. PTX mode entry/exit works correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gen.grammar import NodeType, get_node_spec
from gen.ptx_grammar import PTXNodeType
from gen.builder_state import BuilderState
from gen.ast_actions import create_node
from gen.ptx_actions import create_ptx_node


def test_minimal_kernel():
    """Build minimal __global__ void k() {} kernel."""
    state = BuilderState()

    # Add function
    state.add_node(NodeType.FUNCTION, {
        "name": "k",
        "return_type": "void",
        "qualifier": "__global__"
    })
    # End function body
    state.add_node(NodeType.END, {})

    code = state.emit()
    assert "__global__" in code
    assert "void" in code
    assert "k(" in code
    print("✓ test_minimal_kernel")
    return code


def test_kernel_with_params():
    """Build kernel with parameters."""
    state = BuilderState()

    state.add_node(NodeType.FUNCTION, {
        "name": "add_kernel",
        "return_type": "void",
        "qualifier": "__global__"
    })
    state.add_node(NodeType.PARAM, {
        "name": "a",
        "type": "float*"
    })
    state.add_node(NodeType.PARAM, {
        "name": "b",
        "type": "float*"
    })
    state.add_node(NodeType.PARAM, {
        "name": "n",
        "type": "int"
    })
    state.add_node(NodeType.END, {})  # end params, start body

    # Add a variable
    state.add_node(NodeType.VAR_DECL, {
        "name": "idx",
        "type": "int",
        "init": "threadIdx.x + blockIdx.x * blockDim.x"
    })

    state.add_node(NodeType.END, {})  # end function

    code = state.emit()
    assert "add_kernel" in code
    assert "float*" in code or "float *" in code
    assert "int n" in code
    assert "idx" in code
    print("✓ test_kernel_with_params")
    return code


def test_inline_asm_ptx():
    """Build kernel with inline PTX assembly."""
    state = BuilderState()

    state.add_node(NodeType.FUNCTION, {
        "name": "ptx_kernel",
        "return_type": "void",
        "qualifier": "__global__"
    })
    state.add_node(NodeType.END, {})  # end params

    # Enter inline asm (enters PTX mode)
    state.add_node(NodeType.INLINE_ASM, {})

    # Now in PTX mode - add PTX instructions
    assert state.in_ptx_mode(), "Should be in PTX mode after INLINE_ASM"

    state.add_ptx_node(PTXNodeType.PTX_REG_DECL, {
        "dtype": ".b32",
        "names": ["%r0", "%r1"]
    })

    state.add_ptx_node(PTXNodeType.PTX_MOV, {
        "dtype": ".b32",
        "dst": "%r0",
        "src": "0"
    })

    state.add_ptx_node(PTXNodeType.PTX_ADD, {
        "dtype": ".s32",
        "dst": "%r0",
        "a": "%r0",
        "b": "%r1"
    })

    # Exit PTX mode
    state.add_ptx_node(PTXNodeType.PTX_END, {})
    assert not state.in_ptx_mode(), "Should exit PTX mode after PTX_END"

    state.add_node(NodeType.END, {})  # end function

    code = state.emit()
    assert "asm" in code.lower() or "asm" in code
    assert "mov" in code
    assert "add" in code
    print("✓ test_inline_asm_ptx")
    return code


def test_for_loop():
    """Build kernel with for loop."""
    state = BuilderState()

    state.add_node(NodeType.FUNCTION, {
        "name": "loop_kernel",
        "return_type": "void",
        "qualifier": "__global__"
    })
    state.add_node(NodeType.END, {})

    state.add_node(NodeType.FOR, {
        "init": "int i = 0",
        "condition": "i < 10",
        "increment": "i++"
    })

    state.add_node(NodeType.STATEMENT, {
        "code": "x += i"
    })

    state.add_node(NodeType.END, {})  # end for
    state.add_node(NodeType.END, {})  # end function

    code = state.emit()
    assert "for" in code
    assert "i < 10" in code or "i<10" in code
    print("✓ test_for_loop")
    return code


def test_if_statement():
    """Build kernel with if statement."""
    state = BuilderState()

    state.add_node(NodeType.FUNCTION, {
        "name": "branch_kernel",
        "return_type": "void",
        "qualifier": "__global__"
    })
    state.add_node(NodeType.END, {})

    state.add_node(NodeType.IF, {
        "condition": "threadIdx.x == 0"
    })
    state.add_node(NodeType.STATEMENT, {"code": "result = 1"})
    state.add_node(NodeType.END, {})  # end if
    state.add_node(NodeType.END, {})  # end function

    code = state.emit()
    assert "if" in code
    assert "threadIdx.x" in code
    print("✓ test_if_statement")
    return code


def test_complex_ptx_sequence():
    """Test more complex PTX instruction sequence."""
    state = BuilderState()

    state.add_node(NodeType.FUNCTION, {
        "name": "mma_kernel",
        "return_type": "void",
        "qualifier": "__global__"
    })
    state.add_node(NodeType.END, {})

    state.add_node(NodeType.INLINE_ASM, {})

    # Register declarations
    state.add_ptx_node(PTXNodeType.PTX_REG_DECL, {
        "dtype": ".b32",
        "names": ["%r0", "%r1", "%r2", "%r3"]
    })

    # Load from global
    state.add_ptx_node(PTXNodeType.PTX_LD_GLOBAL, {
        "dtype": ".u32",
        "dst": "%r0",
        "addr": "[%rd0]"
    })

    # Shuffle
    state.add_ptx_node(PTXNodeType.PTX_SHFL, {
        "dtype": ".b32",
        "mode": ".bfly",
        "dst": "%r1",
        "src": "%r0",
        "lane": "16",
        "mask": "0xffffffff"
    })

    # Add
    state.add_ptx_node(PTXNodeType.PTX_ADD, {
        "dtype": ".s32",
        "dst": "%r0",
        "a": "%r0",
        "b": "%r1"
    })

    # Store to global
    state.add_ptx_node(PTXNodeType.PTX_ST_GLOBAL, {
        "dtype": ".u32",
        "addr": "[%rd1]",
        "src": "%r0"
    })

    state.add_ptx_node(PTXNodeType.PTX_END, {})
    state.add_node(NodeType.END, {})

    code = state.emit()
    assert "ld.global" in code
    assert "shfl" in code
    assert "st.global" in code
    print("✓ test_complex_ptx_sequence")
    return code


def test_valid_actions_mask():
    """Test that valid action masks work correctly."""
    state = BuilderState()

    # At start, should allow top-level items
    valid = state.get_valid_node_types()
    assert NodeType.FUNCTION in valid
    assert NodeType.INCLUDE in valid

    # After starting function, params should be valid
    state.add_node(NodeType.FUNCTION, {
        "name": "test",
        "return_type": "void",
        "qualifier": "__global__"
    })

    valid = state.get_valid_node_types()
    assert NodeType.PARAM in valid
    assert NodeType.END in valid

    print("✓ test_valid_actions_mask")


def test_ptx_valid_actions():
    """Test PTX valid action masks."""
    state = BuilderState()

    state.add_node(NodeType.FUNCTION, {
        "name": "test",
        "return_type": "void",
        "qualifier": "__global__"
    })
    state.add_node(NodeType.END, {})
    state.add_node(NodeType.INLINE_ASM, {})

    # In PTX mode, should have PTX actions available
    valid_ptx = state.get_valid_ptx_types()
    assert PTXNodeType.PTX_REG_DECL in valid_ptx
    assert PTXNodeType.PTX_END in valid_ptx
    assert PTXNodeType.PTX_ADD in valid_ptx

    print("✓ test_ptx_valid_actions")


def run_all():
    """Run all tests."""
    print("\n=== AST Roundtrip Tests ===\n")

    code1 = test_minimal_kernel()
    code2 = test_kernel_with_params()
    code3 = test_inline_asm_ptx()
    code4 = test_for_loop()
    code5 = test_if_statement()
    code6 = test_complex_ptx_sequence()
    test_valid_actions_mask()
    test_ptx_valid_actions()

    print("\n=== All tests passed ===\n")

    # Print sample outputs
    print("Sample outputs:\n")
    print("--- Minimal kernel ---")
    print(code1)
    print("\n--- PTX kernel ---")
    print(code3)
    print("\n--- Complex PTX ---")
    print(code6)


if __name__ == "__main__":
    run_all()
