#!/usr/bin/env python3
"""
PTX Reconstruction Pipeline

Batch processes source files:
1. Deconstructs each file to extract inline asm blocks
2. Parses PTX to AST
3. Optionally generates builder commands (.txt) showing API calls
4. Reconstructs source files from AST
5. Verifies roundtrip fidelity

Usage:
    python reconstruct.py examples/*.py --output reconstructed/ --commands
    python reconstruct.py examples/2.py --verify
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

from ptx_ast import deconstruct_file, reconstruct, DeconstructedSource


def process_file(input_path: str, output_dir: str,
                 generate_commands: bool = False,
                 verify: bool = False,
                 structured: bool = False) -> Tuple[bool, str]:
    """
    Process a single file.

    Returns:
        (success, message)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # Deconstruct
    try:
        doc = deconstruct_file(str(input_path))
    except Exception as e:
        return False, f"Failed to deconstruct: {e}"

    # Check if we found any ASM blocks
    if doc.source_type in ("cuda", "python") and not doc.asm_blocks:
        return True, f"No asm blocks found (type: {doc.source_type})"

    if doc.source_type == "ptx" and not doc.ptx_ast:
        return False, "Failed to parse PTX"

    # Reconstruct
    try:
        reconstructed = reconstruct(doc)
    except Exception as e:
        return False, f"Failed to reconstruct: {e}"

    # Verify if requested
    if verify:
        if reconstructed == doc.original:
            return True, "Roundtrip verified: identical"
        else:
            # Find differences
            diff_info = _find_diff(doc.original, reconstructed)
            return False, f"Roundtrip mismatch: {diff_info}"

    # Write output
    output_path = output_dir / input_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(reconstructed)

    # Generate builder commands if requested
    if generate_commands:
        commands_path = output_path.with_suffix('.commands.txt')
        commands = _generate_all_commands(doc, structured=structured)
        with open(commands_path, 'w') as f:
            f.write(commands)

    # Summary
    stats = _get_stats(doc)
    return True, f"Written to {output_path} ({stats})"


def _generate_all_commands(doc: DeconstructedSource, structured: bool = False) -> str:
    """Generate builder commands for all AST blocks in the document."""
    lines = [
        f"# Builder commands for {doc.source_type} source",
        f"# Automatically generated - execute to recreate AST/CUDA",
        f"# Mode: {'structured' if structured else 'legacy'}",
        "",
    ]

    if doc.ptx_ast:
        lines.append("# === Main PTX AST ===")
        lines.append(doc.ptx_ast.to_builder_commands())
        lines.append("")

    # Prefer full CUDA source if available (includes both code and asm)
    if doc.cuda_source:
        lines.append("# === Full CUDA Source ===")
        lines.append(doc.cuda_source.to_builder_commands(structured=structured))
        lines.append("")
    elif doc.asm_blocks:
        # Fallback to individual asm blocks
        for i, block in enumerate(doc.asm_blocks):
            lines.append(f"# === ASM Block {i} ===")
            if block.ast:
                lines.append(block.ast.to_builder_commands())
            else:
                lines.append("# (no AST)")
            lines.append("")

    return "\n".join(lines)


def _get_stats(doc: DeconstructedSource) -> str:
    """Get statistics string for a document."""
    if doc.ptx_ast:
        return f"{len(doc.ptx_ast.statements)} statements"

    if doc.cuda_source:
        n_cuda = len(doc.cuda_source.find_cuda_code())
        n_asm = len(doc.cuda_source.find_asm_blocks())
        total_stmts = sum(len(b.ast.statements) if b.ast else 0
                         for b in doc.cuda_source.find_asm_blocks())
        return f"{n_cuda} cuda segments, {n_asm} asm blocks, {total_stmts} ptx stmts"

    if doc.asm_blocks:
        total_stmts = sum(len(b.ast.statements) if b.ast else 0 for b in doc.asm_blocks)
        return f"{len(doc.asm_blocks)} blocks, {total_stmts} statements"

    return "no content"


def _find_diff(original: str, reconstructed: str) -> str:
    """Find first difference between two strings."""
    orig_lines = original.split('\n')
    recon_lines = reconstructed.split('\n')

    for i, (o, r) in enumerate(zip(orig_lines, recon_lines)):
        if o != r:
            return f"line {i+1}: {repr(o[:50])} vs {repr(r[:50])}"

    if len(orig_lines) != len(recon_lines):
        return f"line count: {len(orig_lines)} vs {len(recon_lines)}"

    return "content differs"


def main():
    parser = argparse.ArgumentParser(description="PTX Reconstruction Pipeline")
    parser.add_argument("files", nargs="+", help="Input files to process")
    parser.add_argument("-o", "--output", default="reconstructed",
                        help="Output directory (default: reconstructed)")
    parser.add_argument("-c", "--commands", action="store_true",
                        help="Generate .commands.txt with builder API calls")
    parser.add_argument("-v", "--verify", action="store_true",
                        help="Verify roundtrip fidelity without writing files")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only print errors")
    parser.add_argument("-s", "--structured", action="store_true",
                        help="Use structured CUDA AST for command generation")

    args = parser.parse_args()

    success_count = 0
    fail_count = 0

    for file_path in args.files:
        success, msg = process_file(
            file_path,
            args.output,
            generate_commands=args.commands,
            verify=args.verify,
            structured=args.structured
        )

        if success:
            success_count += 1
            if not args.quiet:
                print(f"✓ {file_path}: {msg}")
        else:
            fail_count += 1
            print(f"✗ {file_path}: {msg}", file=sys.stderr)

    if not args.quiet:
        print(f"\nProcessed: {success_count} success, {fail_count} failed")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
