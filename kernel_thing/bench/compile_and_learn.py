#!/usr/bin/env python3
"""
Compile CUDA code and update PTX DB based on results

This is the feedback loop:
1. Generate CUDA code (from builder commands)
2. Try to compile (with caching)
3. If fail -> record in DB which instructions failed
4. If success -> run tests
5. Record results in DB

Usage:
    python compile_and_learn.py <cuda_source.cu> [--sm 75] [--db ptx_db.json]
"""

import sys
import os
import re
import json
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from ptx_db import PTXDB, TestStatus


# =============================================================================
# Compile Cache
# =============================================================================

@dataclass
class CompileResult:
    """Cached compilation result"""
    success: bool
    output_path: Optional[str]  # Path to cubin if success
    error_msg: Optional[str]    # Error message if fail
    sm: int
    source_hash: str


class CompileCache:
    """
    Cache for compilation results.
    Key: hash(source_code + sm_version)
    Value: CompileResult
    """

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent / ".compile_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.index_path = self.cache_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    @staticmethod
    def compute_hash(source: str, sm: int) -> str:
        """Compute cache key from source code and SM version"""
        content = f"sm{sm}:{source}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, source: str, sm: int) -> Optional[CompileResult]:
        """Look up cached result"""
        key = self.compute_hash(source, sm)
        if key not in self._index:
            return None

        entry = self._index[key]

        # Validate cubin still exists if it was a success
        if entry["success"] and entry["output_path"]:
            if not Path(entry["output_path"]).exists():
                del self._index[key]
                self._save_index()
                return None

        return CompileResult(
            success=entry["success"],
            output_path=entry["output_path"],
            error_msg=entry["error_msg"],
            sm=entry["sm"],
            source_hash=key
        )

    def put(self, source: str, sm: int, result: CompileResult):
        """Store compilation result"""
        key = self.compute_hash(source, sm)
        self._index[key] = asdict(result)
        self._save_index()

    def clear(self):
        """Clear the cache"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._index = {}

    def stats(self) -> Dict[str, int]:
        """Return cache statistics"""
        total = len(self._index)
        success = sum(1 for e in self._index.values() if e["success"])
        return {
            "total": total,
            "success": success,
            "fail": total - success
        }


# Global cache instance
_compile_cache: Optional[CompileCache] = None


def get_compile_cache() -> CompileCache:
    """Get or create global compile cache"""
    global _compile_cache
    if _compile_cache is None:
        _compile_cache = CompileCache()
    return _compile_cache


def get_sm_version() -> int:
    """Get SM version of current GPU"""
    try:
        import torch
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    except:
        return 75  # Default to Turing


def extract_ptx_instructions(cuda_source: str) -> List[str]:
    """Extract PTX instructions from inline asm blocks"""
    instructions = []

    # Find asm blocks
    asm_pattern = r'asm\s*(?:volatile|__volatile__)?\s*\(\s*"([^"]+)"'
    for match in re.finditer(asm_pattern, cuda_source, re.DOTALL):
        asm_body = match.group(1)
        # Parse PTX lines
        for line in asm_body.replace('\\n', '\n').split('\n'):
            line = line.strip().rstrip(';')
            if not line or line.startswith('//') or line.startswith('.'):
                if line.startswith('.reg'):
                    instructions.append(line.split()[0] + line.split()[1] if len(line.split()) > 1 else line)
                continue
            # Extract opcode
            parts = line.split()
            if parts:
                instructions.append(parts[0])

    return list(set(instructions))


def compile_cuda(source_path: str, sm: int, output_path: Optional[str] = None,
                 use_cache: bool = True) -> Tuple[bool, str]:
    """
    Try to compile CUDA source

    Args:
        source_path: Path to .cu file
        sm: Target SM version
        output_path: Where to put cubin (auto-generated if None)
        use_cache: Whether to use compile cache

    Returns: (success, error_message_or_output_path)
    """
    # Read source for caching
    with open(source_path) as f:
        source = f.read()

    cache = get_compile_cache() if use_cache else None

    # Check cache
    if cache:
        cached = cache.get(source, sm)
        if cached is not None:
            if cached.success:
                return True, cached.output_path
            else:
                return False, cached.error_msg

    # Determine output path
    if output_path is None:
        if cache:
            # Use cache dir for persistent storage
            source_hash = CompileCache.compute_hash(source, sm)
            output_path = str(cache.cache_dir / f"{source_hash}.cubin")
        else:
            output_path = tempfile.mktemp(suffix=".cubin")

    cmd = [
        "nvcc",
        "-cubin",
        f"-arch=sm_{sm}",
        "-O3",
        "--use_fast_math",
        "-o", output_path,
        source_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            # Cache success
            if cache:
                cache.put(source, sm, CompileResult(
                    success=True,
                    output_path=output_path,
                    error_msg=None,
                    sm=sm,
                    source_hash=CompileCache.compute_hash(source, sm)
                ))
            return True, output_path
        else:
            # Cache failure
            if cache:
                cache.put(source, sm, CompileResult(
                    success=False,
                    output_path=None,
                    error_msg=result.stderr,
                    sm=sm,
                    source_hash=CompileCache.compute_hash(source, sm)
                ))
            return False, result.stderr

    except subprocess.TimeoutExpired:
        return False, "Compilation timeout"
    except Exception as e:
        return False, str(e)


def compile_cuda_source(source: str, sm: int, use_cache: bool = True) -> Tuple[bool, str]:
    """
    Compile CUDA source code directly (not from file)

    Args:
        source: CUDA source code as string
        sm: Target SM version
        use_cache: Whether to use compile cache

    Returns: (success, error_message_or_output_path)
    """
    cache = get_compile_cache() if use_cache else None

    # Check cache first
    if cache:
        cached = cache.get(source, sm)
        if cached is not None:
            if cached.success:
                return True, cached.output_path
            else:
                return False, cached.error_msg

    # Write to temp file and compile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(source)
        temp_path = f.name

    try:
        return compile_cuda(temp_path, sm, use_cache=use_cache)
    finally:
        os.unlink(temp_path)


def parse_nvcc_errors(stderr: str) -> List[str]:
    """Parse nvcc error output to find problematic instructions"""
    bad_instructions = []

    # Common patterns for PTX errors
    patterns = [
        r"'([a-z]+\.[a-z0-9.::]+)'\s+is not a valid",
        r"unknown instruction\s+'([^']+)'",
        r"feature not supported.*'([^']+)'",
        r"requires sm_(\d+)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, stderr, re.IGNORECASE):
            bad_instructions.append(match.group(1))

    return bad_instructions


def learn_from_compile(db: PTXDB, source_path: str, sm: int) -> Tuple[bool, str]:
    """
    Compile and update DB based on results

    Returns: (success, message)
    """
    # Read source
    with open(source_path) as f:
        source = f.read()

    # Extract PTX instructions
    instructions = extract_ptx_instructions(source)
    print(f"Found {len(instructions)} PTX instruction patterns")

    # Try to compile
    success, result = compile_cuda(source_path, sm)

    if success:
        # All instructions worked!
        for instr in instructions:
            db.record_result(instr, sm, TestStatus.COMPILE_OK)
        return True, f"Compiled successfully for SM {sm}"

    else:
        # Parse errors
        bad_instrs = parse_nvcc_errors(result)

        if bad_instrs:
            for instr in bad_instrs:
                db.record_result(instr, sm, TestStatus.COMPILE_FAIL)
                print(f"  FAIL: {instr}")

        # Record others as untested (we don't know if they would work alone)
        for instr in instructions:
            if instr not in bad_instrs:
                # Don't overwrite existing status
                entry = db.get(instr)
                if not entry or entry.get_status(sm) == TestStatus.UNTESTED:
                    pass  # Leave as untested

        return False, result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compile CUDA and learn from results")
    parser.add_argument("source", nargs="?", help="CUDA source file")
    parser.add_argument("--sm", type=int, default=None, help="Target SM version")
    parser.add_argument("--db", type=str, default="ptx_learned.json", help="PTX DB file")
    parser.add_argument("--no-cache", action="store_true", help="Disable compile cache")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache stats and exit")
    parser.add_argument("--clear-cache", action="store_true", help="Clear compile cache")
    args = parser.parse_args()

    cache = get_compile_cache()

    # Handle cache commands
    if args.clear_cache:
        cache.clear()
        print("Compile cache cleared")
        return

    if args.cache_stats:
        stats = cache.stats()
        print(f"Compile cache stats:")
        print(f"  Total entries: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Fail: {stats['fail']}")
        return

    if not args.source:
        parser.error("source file required (unless using --cache-stats or --clear-cache)")

    sm = args.sm or get_sm_version()
    print(f"Target SM: {sm}")
    print(f"Compile cache: {'disabled' if args.no_cache else 'enabled'}")

    # Load or create DB (blank slate - no bootstrap)
    db = PTXDB(load_path=args.db if Path(args.db).exists() else None, bootstrap=False)
    print(f"DB has {len(db)} instruction entries")

    # Check if cached first (informational)
    if not args.no_cache:
        with open(args.source) as f:
            source = f.read()
        cached = cache.get(source, sm)
        if cached:
            print(f"(cached: {'success' if cached.success else 'fail'})")

    # Compile and learn
    success, msg = learn_from_compile(db, args.source, sm)

    if success:
        print(f"✓ {msg}")
    else:
        print(f"✗ Compile failed")
        print(msg[:500] if len(msg) > 500 else msg)

    # Save DB
    db.save(args.db)
    print(f"Saved DB to {args.db} ({len(db)} entries)")


if __name__ == "__main__":
    main()
