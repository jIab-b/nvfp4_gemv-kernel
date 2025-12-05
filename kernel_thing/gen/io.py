"""
I/O for action sequences - serialize/deserialize and parse CUDA source.

Two main functions:
1. serialize/deserialize: Actions <-> JSON for storage and LLM interface
2. parse_cuda: CUDA source -> action sequence for training data

Action format:
    List[Tuple[NodeType | PTXNodeType, Dict[str, Any]]]

JSON format:
    [{"type": "FUNCTION", "values": {"name": "kernel", ...}}, ...]
"""

import json
import re
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

from gen.grammar import NodeType, ValueType, get_node_spec
from gen.ptx_grammar import PTXNodeType, PTXValueType, get_ptx_spec


# =============================================================================
# SERIALIZATION: Actions <-> JSON
# =============================================================================

Action = Tuple[Union[NodeType, PTXNodeType], Dict[str, Any]]
ActionSequence = List[Action]


def actions_to_json(actions: ActionSequence) -> str:
    """Convert action sequence to JSON string."""
    json_actions = []
    for node_type, values in actions:
        # PTXNodeType names already start with PTX_, CUDA NodeType don't
        # Use the enum name directly
        type_name = node_type.name

        json_actions.append({
            "type": type_name,
            "values": values
        })

    return json.dumps(json_actions, indent=2)


def actions_from_json(json_str: str) -> ActionSequence:
    """Parse JSON string to action sequence."""
    data = json.loads(json_str)
    actions = []

    for item in data:
        type_name = item["type"]
        values = item.get("values", {})

        # PTXNodeType names start with PTX_, CUDA NodeType don't
        if type_name.startswith("PTX_"):
            node_type = PTXNodeType[type_name]
        else:
            node_type = NodeType[type_name]

        actions.append((node_type, values))

    return actions


def save_actions(actions: ActionSequence, path: Union[str, Path]) -> None:
    """Save action sequence to JSON file."""
    with open(path, 'w') as f:
        f.write(actions_to_json(actions))


def load_actions(path: Union[str, Path]) -> ActionSequence:
    """Load action sequence from JSON file."""
    with open(path, 'r') as f:
        return actions_from_json(f.read())


# =============================================================================
# PARSING: CUDA Source -> Actions
# =============================================================================

# Regex patterns for CUDA parsing
RE_INCLUDE = re.compile(r'#include\s*([<"])([^>"]+)[>"]')
RE_DEFINE = re.compile(r'#define\s+(\w+)(?:\s+(.*))?')
RE_PRAGMA = re.compile(r'#pragma\s+(.*)')
RE_COMMENT_LINE = re.compile(r'//\s*(.*)')
RE_COMMENT_BLOCK = re.compile(r'/\*(.+?)\*/', re.DOTALL)
RE_STRUCT = re.compile(r'struct\s+(\w+)\s*\{')
RE_FUNCTION = re.compile(
    r'(?:(__global__|__device__|__host__|__host__\s+__device__)\s+)?'
    r'(?:(__forceinline__|inline)\s+)?'
    r'(\w+(?:\s*[*&])?)\s+'  # return type
    r'(\w+)\s*\('  # function name
)
RE_FOR = re.compile(r'for\s*\(\s*([^;]*)\s*;\s*([^;]*)\s*;\s*([^)]*)\s*\)')
RE_WHILE = re.compile(r'while\s*\(\s*([^)]+)\s*\)')
RE_IF = re.compile(r'if\s*\(\s*([^)]+)\s*\)')
RE_VAR_DECL = re.compile(r'(\w+(?:\s*[*&<>,:]+\s*\w*)*)\s+(\w+)\s*(?:=\s*([^;]+))?\s*;')
RE_ASM_START = re.compile(r'asm\s*(?:volatile)?\s*\(')
RE_PARAM = re.compile(r'(\w+(?:\s*[*&<>,]+\s*\w*)*)\s+(\w+)')

# PTX instruction patterns
RE_PTX_REG_DECL = re.compile(r'\.reg\s+(\.\w+)\s+([^;]+);')
RE_PTX_SHARED_DECL = re.compile(r'\.shared\s+(\.\w+)\s+(\w+)\[(\d+)\];')
RE_PTX_INSTR = re.compile(r'(\w+)(\.[.\w]+)*\s+([^;]+);')
RE_PTX_LABEL = re.compile(r'\$(\w+):')


def parse_cuda(source: str) -> ActionSequence:
    """
    Parse CUDA source code into action sequence.

    This is a simplified parser that handles common patterns.
    Complex C++ constructs may not parse perfectly.

    Returns action sequence that can be replayed through BuilderState.
    """
    actions: ActionSequence = []
    lines = source.split('\n')

    # State tracking
    in_function = False
    in_asm = False
    brace_depth = 0
    asm_content = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines at top level
        if not line:
            if in_function and not in_asm:
                actions.append((NodeType.BLANK_LINE, {}))
            i += 1
            continue

        # Inside asm block - collect PTX
        if in_asm:
            # Extract quoted content from this line
            quotes = re.findall(r'"([^"]*)"', line)
            for q in quotes:
                asm_content.append(q)

            # Check for end of asm block (look for closing paren)
            if ');' in line:
                # End of asm block
                asm_text = '\n'.join(asm_content)
                _parse_ptx_block(asm_text, actions)
                actions.append((PTXNodeType.PTX_END, {}))
                in_asm = False
                asm_content = []
            i += 1
            continue

        # Check for asm start
        if RE_ASM_START.search(line):
            actions.append((NodeType.INLINE_ASM, {}))
            in_asm = True
            asm_content = []
            # Extract any quoted content on same line
            quote_match = re.search(r'"([^"]*)"', line)
            if quote_match:
                asm_content.append(quote_match.group(1))
            # Check if ends on same line
            if ');' in line and line.rfind(');') > line.find('asm'):
                # Collect all quoted strings on this line
                all_quotes = re.findall(r'"([^"]*)"', line)
                asm_text = '\n'.join(all_quotes)
                _parse_ptx_block(asm_text, actions)
                actions.append((PTXNodeType.PTX_END, {}))
                in_asm = False
                asm_content = []
            i += 1
            continue

        # Preprocessor directives
        if line.startswith('#include'):
            match = RE_INCLUDE.match(line)
            if match:
                is_system = match.group(1) == '<'
                actions.append((NodeType.INCLUDE, {
                    "path": match.group(2),
                    "is_system": is_system
                }))
            i += 1
            continue

        if line.startswith('#define'):
            match = RE_DEFINE.match(line)
            if match:
                actions.append((NodeType.DEFINE, {
                    "name": match.group(1),
                    "value": match.group(2) or ""
                }))
            i += 1
            continue

        if line.startswith('#pragma'):
            match = RE_PRAGMA.match(line)
            if match:
                actions.append((NodeType.PRAGMA, {"content": match.group(1)}))
            i += 1
            continue

        # Comments
        if line.startswith('//'):
            match = RE_COMMENT_LINE.match(line)
            if match:
                actions.append((NodeType.COMMENT, {"text": match.group(1)}))
            i += 1
            continue

        if line.startswith('/*'):
            # Multi-line comment - find end
            comment_text = line[2:]
            while '*/' not in comment_text and i < len(lines) - 1:
                i += 1
                comment_text += '\n' + lines[i]
            comment_text = comment_text.replace('*/', '').strip()
            actions.append((NodeType.COMMENT, {"text": comment_text}))
            i += 1
            continue

        # Struct
        struct_match = RE_STRUCT.match(line)
        if struct_match:
            actions.append((NodeType.STRUCT, {"name": struct_match.group(1)}))
            brace_depth = 1
            # Parse struct body until closing brace
            i += 1
            while i < len(lines) and brace_depth > 0:
                sline = lines[i].strip()
                if '{' in sline:
                    brace_depth += sline.count('{')
                if '}' in sline:
                    brace_depth -= sline.count('}')
                if brace_depth > 0 and sline and not sline.startswith('//'):
                    # Try to parse as field
                    field_match = RE_VAR_DECL.match(sline)
                    if field_match:
                        actions.append((NodeType.STRUCT_FIELD, {
                            "type": field_match.group(1).strip(),
                            "name": field_match.group(2)
                        }))
                i += 1
            actions.append((NodeType.END, {}))  # End struct
            continue

        # Function
        func_match = RE_FUNCTION.match(line)
        if func_match and '(' in line:
            qualifier = func_match.group(1) or ""
            return_type = func_match.group(3).strip()
            name = func_match.group(4)

            actions.append((NodeType.FUNCTION, {
                "name": name,
                "return_type": return_type,
                "qualifier": qualifier
            }))

            # Parse parameters
            param_start = line.find('(') + 1
            param_end = line.find(')')
            if param_end == -1:
                # Multi-line params
                param_text = line[param_start:]
                while ')' not in param_text and i < len(lines) - 1:
                    i += 1
                    param_text += ' ' + lines[i].strip()
                param_end = param_text.find(')')
                param_text = param_text[:param_end]
            else:
                param_text = line[param_start:param_end]

            # Parse each param
            if param_text.strip():
                params = param_text.split(',')
                for param in params:
                    param = param.strip()
                    if param:
                        pmatch = RE_PARAM.match(param)
                        if pmatch:
                            actions.append((NodeType.PARAM, {
                                "type": pmatch.group(1).strip(),
                                "name": pmatch.group(2)
                            }))

            actions.append((NodeType.END, {}))  # End params, start body

            # Track braces for function body
            if '{' in line:
                in_function = True
                brace_depth = line.count('{') - line.count('}')
            i += 1
            continue

        # Track braces
        if in_function:
            open_braces = line.count('{')
            close_braces = line.count('}')
            brace_depth += open_braces - close_braces

            if brace_depth <= 0:
                actions.append((NodeType.END, {}))  # End function
                in_function = False
                brace_depth = 0
                i += 1
                continue

            # For loop
            for_match = RE_FOR.search(line)
            if for_match:
                actions.append((NodeType.FOR, {
                    "init": for_match.group(1).strip(),
                    "condition": for_match.group(2).strip(),
                    "increment": for_match.group(3).strip()
                }))
                if '{' not in line:
                    # Single-line for body
                    i += 1
                    if i < len(lines):
                        body_line = lines[i].strip().rstrip(';')
                        if body_line:
                            actions.append((NodeType.STATEMENT, {"code": body_line}))
                    actions.append((NodeType.END, {}))
                i += 1
                continue

            # While loop
            while_match = RE_WHILE.match(line)
            if while_match:
                actions.append((NodeType.WHILE, {
                    "condition": while_match.group(1).strip()
                }))
                i += 1
                continue

            # If statement
            if_match = RE_IF.match(line)
            if if_match:
                actions.append((NodeType.IF, {
                    "condition": if_match.group(1).strip()
                }))
                i += 1
                continue

            # Variable declaration
            var_match = RE_VAR_DECL.match(line)
            if var_match and not line.startswith('return'):
                actions.append((NodeType.VAR_DECL, {
                    "type": var_match.group(1).strip(),
                    "name": var_match.group(2),
                    "initializer": var_match.group(3).strip() if var_match.group(3) else ""
                }))
                i += 1
                continue

            # Return statement
            if line.startswith('return'):
                value = line[6:].strip().rstrip(';')
                actions.append((NodeType.RETURN, {"value": value}))
                i += 1
                continue

            # Generic statement
            if line and not line.startswith('{') and not line.startswith('}'):
                code = line.rstrip(';')
                if code:
                    actions.append((NodeType.STATEMENT, {"code": code}))

        i += 1

    return actions


def _parse_ptx_block(ptx_text: str, actions: ActionSequence) -> None:
    """Parse PTX assembly text and append actions."""
    # Handle both actual newlines and escaped \n sequences
    # Replace escaped sequences with actual newlines first
    ptx_text = ptx_text.replace('\\n', '\n').replace('\\t', '\t')
    lines = ptx_text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Register declaration
        reg_match = RE_PTX_REG_DECL.match(line)
        if reg_match:
            dtype = reg_match.group(1)
            names = [n.strip() for n in reg_match.group(2).split(',')]
            actions.append((PTXNodeType.PTX_REG_DECL, {
                "dtype": dtype,
                "names": names
            }))
            continue

        # Shared memory declaration
        shared_match = RE_PTX_SHARED_DECL.match(line)
        if shared_match:
            actions.append((PTXNodeType.PTX_SHARED_DECL, {
                "dtype": shared_match.group(1),
                "name": shared_match.group(2),
                "size": shared_match.group(3)
            }))
            continue

        # Label
        label_match = RE_PTX_LABEL.match(line)
        if label_match:
            actions.append((PTXNodeType.PTX_LABEL, {
                "name": label_match.group(1)
            }))
            continue

        # Generic instruction
        instr_match = RE_PTX_INSTR.match(line)
        if instr_match:
            mnemonic = instr_match.group(1)
            modifiers = instr_match.group(2) or ""
            operands_str = instr_match.group(3).strip()
            operands = [op.strip() for op in operands_str.split(',')]

            # Map mnemonic to PTXNodeType
            node_type = _map_ptx_mnemonic(mnemonic, modifiers)
            if node_type:
                values = _build_ptx_values(node_type, modifiers, operands)
                actions.append((node_type, values))


def _map_ptx_mnemonic(mnemonic: str, modifiers: str) -> Optional[PTXNodeType]:
    """Map PTX mnemonic to node type."""
    mnemonic = mnemonic.lower()

    # Data movement
    if mnemonic == 'mov':
        return PTXNodeType.PTX_MOV
    if mnemonic == 'ld':
        if '.global' in modifiers:
            return PTXNodeType.PTX_LD_GLOBAL
        if '.shared' in modifiers:
            return PTXNodeType.PTX_LD_SHARED
        if '.param' in modifiers:
            return PTXNodeType.PTX_LD_PARAM
    if mnemonic == 'st':
        if '.global' in modifiers:
            return PTXNodeType.PTX_ST_GLOBAL
        if '.shared' in modifiers:
            return PTXNodeType.PTX_ST_SHARED

    # Integer arithmetic
    if mnemonic == 'add' and '.f' not in modifiers:
        return PTXNodeType.PTX_ADD
    if mnemonic == 'sub' and '.f' not in modifiers:
        return PTXNodeType.PTX_SUB
    if mnemonic == 'mul' and '.f' not in modifiers:
        return PTXNodeType.PTX_MUL
    if mnemonic == 'mad':
        return PTXNodeType.PTX_MAD
    if mnemonic == 'shl':
        return PTXNodeType.PTX_SHL
    if mnemonic == 'shr':
        return PTXNodeType.PTX_SHR

    # Float arithmetic
    if mnemonic == 'add' and '.f' in modifiers:
        return PTXNodeType.PTX_ADD_F
    if mnemonic == 'sub' and '.f' in modifiers:
        return PTXNodeType.PTX_SUB_F
    if mnemonic == 'mul' and '.f' in modifiers:
        return PTXNodeType.PTX_MUL_F
    if mnemonic == 'fma':
        return PTXNodeType.PTX_FMA
    if mnemonic == 'div' and '.f' in modifiers:
        return PTXNodeType.PTX_DIV_F

    # Conversion
    if mnemonic == 'cvt':
        return PTXNodeType.PTX_CVT

    # Bitwise
    if mnemonic == 'and':
        return PTXNodeType.PTX_AND
    if mnemonic == 'or':
        return PTXNodeType.PTX_OR
    if mnemonic == 'xor':
        return PTXNodeType.PTX_XOR
    if mnemonic == 'not':
        return PTXNodeType.PTX_NOT

    # Comparison
    if mnemonic == 'setp':
        return PTXNodeType.PTX_SETP
    if mnemonic == 'selp':
        return PTXNodeType.PTX_SELP

    # Control flow
    if mnemonic == 'bra':
        return PTXNodeType.PTX_BRA

    # Synchronization
    if mnemonic == 'bar':
        return PTXNodeType.PTX_BAR
    if mnemonic == 'membar':
        return PTXNodeType.PTX_MEMBAR

    # Warp-level
    if mnemonic == 'shfl':
        return PTXNodeType.PTX_SHFL
    if mnemonic == 'vote':
        return PTXNodeType.PTX_VOTE
    if mnemonic == 'match':
        return PTXNodeType.PTX_MATCH
    if mnemonic == 'redux':
        return PTXNodeType.PTX_REDUX

    # Tensor core
    if mnemonic == 'mma':
        return PTXNodeType.PTX_MMA
    if mnemonic == 'ldmatrix':
        return PTXNodeType.PTX_LDMATRIX
    if mnemonic == 'stmatrix':
        return PTXNodeType.PTX_STMATRIX

    # Special
    if mnemonic == 'prmt':
        return PTXNodeType.PTX_PRMT
    if mnemonic == 'lop3':
        return PTXNodeType.PTX_LOP3
    if mnemonic == 'dp4a':
        return PTXNodeType.PTX_DP4A
    if mnemonic == 'dp2a':
        return PTXNodeType.PTX_DP2A

    return None


def _build_ptx_values(node_type: PTXNodeType, modifiers: str, operands: List[str]) -> Dict[str, Any]:
    """Build values dict from parsed PTX instruction."""
    spec = get_ptx_spec(node_type)
    values = {}

    # Extract dtype from modifiers
    dtype_match = re.search(r'\.([bsuf]\d+|f16x2|bf16x2|e\d+m\d+x?\d*|pred)', modifiers)
    if dtype_match:
        values['dtype'] = '.' + dtype_match.group(1)

    # Map operands to spec
    op_idx = 0
    for slot in spec.operands:
        if slot.name == 'dtype':
            continue  # Already handled
        if op_idx < len(operands):
            values[slot.name] = operands[op_idx]
            op_idx += 1

    # Store full modifiers for reconstruction
    if modifiers:
        values['modifiers'] = modifiers

    return values


# =============================================================================
# UTILITY
# =============================================================================

def parse_example_file(path: Union[str, Path]) -> ActionSequence:
    """
    Parse an example file (like examples/1.py) and extract CUDA source.

    Example files contain CUDA source in triple-quoted strings.
    """
    with open(path, 'r') as f:
        content = f.read()

    # Find CUDA source in triple-quoted strings
    # Look for gemv_cuda = r\"\"\" or similar patterns
    cuda_match = re.search(r'[a-z_]+_cuda\s*=\s*r?"""(.+?)"""', content, re.DOTALL)
    if cuda_match:
        return parse_cuda(cuda_match.group(1))

    # Try single quotes
    cuda_match = re.search(r"[a-z_]+_cuda\s*=\s*r?'''(.+?)'''", content, re.DOTALL)
    if cuda_match:
        return parse_cuda(cuda_match.group(1))

    return []


def replay_actions(actions: ActionSequence) -> str:
    """
    Replay action sequence through BuilderState to get CUDA source.

    Useful for testing roundtrip.
    """
    from gen.builder_state import BuilderState

    state = BuilderState()
    for node_type, values in actions:
        if isinstance(node_type, PTXNodeType):
            state.add_ptx_node(node_type, values)
        else:
            state.add_node(node_type, values)

    return state.emit()
