#!/usr/bin/env python3
"""
Test roundtrip: Original PTX -> AST -> builder commands -> execute -> AST -> PTX

Verifies that:
1. Parsing and emitting PTX is lossless for the PTX content itself
2. Builder commands can recreate the exact same AST
"""

import sys
from pathlib import Path

from ptx_ast import deconstruct_file, parse_ptx, emit_ptx
from builder import ASTBuilder, reg, imm, vec, mem, sym


def test_ptx_roundtrip(doc):
    """Test that PTX parse -> emit is lossless."""
    errors = []

    if doc.ptx_ast:
        original = doc.ptx_ast.emit()
        reparsed = parse_ptx(original)
        re_emitted = emit_ptx(reparsed)
        if original != re_emitted:
            errors.append(f"Main PTX: emit mismatch")

    for i, block in enumerate(doc.asm_blocks):
        if block.ast:
            original = block.ast.emit()
            reparsed = parse_ptx(original)
            re_emitted = emit_ptx(reparsed)
            if original != re_emitted:
                errors.append(f"Block {i}: emit mismatch")

    return errors


def test_builder_roundtrip(doc):
    """Test that builder commands recreate identical AST."""
    errors = []

    for i, block in enumerate(doc.asm_blocks):
        if block.ast:
            # Get original PTX
            original_ptx = block.ast.emit()

            # Get builder commands
            commands = block.ast.to_builder_commands()

            # Execute builder commands
            local_vars = {
                'ASTBuilder': ASTBuilder,
                'reg': reg, 'imm': imm, 'vec': vec, 'mem': mem, 'sym': sym
            }
            try:
                exec(commands, local_vars)
                rebuilt_ast = local_vars['ast']
                rebuilt_ptx = rebuilt_ast.emit()

                if original_ptx != rebuilt_ptx:
                    errors.append(f"Block {i}: builder mismatch")
                    errors.append(f"  Original: {original_ptx[:100]}...")
                    errors.append(f"  Rebuilt:  {rebuilt_ptx[:100]}...")
            except Exception as e:
                errors.append(f"Block {i}: exec failed: {e}")

    return errors


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else list(Path("examples").glob("*.py"))

    all_errors = []

    for file_path in files:
        print(f"\nTesting {file_path}...")
        doc = deconstruct_file(str(file_path))

        # Test PTX roundtrip
        ptx_errors = test_ptx_roundtrip(doc)
        if ptx_errors:
            print(f"  PTX roundtrip errors:")
            for e in ptx_errors:
                print(f"    - {e}")
            all_errors.extend(ptx_errors)
        else:
            print(f"  ✓ PTX roundtrip OK ({len(doc.asm_blocks)} blocks)")

        # Test builder roundtrip
        builder_errors = test_builder_roundtrip(doc)
        if builder_errors:
            print(f"  Builder roundtrip errors:")
            for e in builder_errors:
                print(f"    - {e}")
            all_errors.extend(builder_errors)
        else:
            print(f"  ✓ Builder roundtrip OK")

    print(f"\n{'='*50}")
    if all_errors:
        print(f"FAILED: {len(all_errors)} errors")
        return 1
    else:
        print("ALL TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
