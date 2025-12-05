"""
CUDA Compilation with Caching

Hash-based cache: hash(source + sm) → CompileResult
"""

import os
import json
import hashlib
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any


@dataclass
class CompileResult:
    """Cached compilation result"""
    success: bool
    cubin_path: Optional[str]
    error_msg: Optional[str]
    sm: int
    source_hash: str


class CompileCache:
    """
    Persistent compile cache.

    Stores cubin files and index in .compile_cache/
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / ".compile_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.index_path = self.cache_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()
        self._hits = 0
        self._misses = 0

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
            json.dump(self._index, f)

    @staticmethod
    def compute_hash(source: str, sm: int) -> str:
        content = f"sm{sm}:{source}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, source: str, sm: int) -> Optional[CompileResult]:
        key = self.compute_hash(source, sm)
        if key not in self._index:
            self._misses += 1
            return None

        entry = self._index[key]

        # Validate cubin exists
        if entry["success"] and entry["cubin_path"]:
            if not Path(entry["cubin_path"]).exists():
                del self._index[key]
                self._save_index()
                self._misses += 1
                return None

        self._hits += 1
        return CompileResult(
            success=entry["success"],
            cubin_path=entry["cubin_path"],
            error_msg=entry["error_msg"],
            sm=entry["sm"],
            source_hash=key
        )

    def put(self, source: str, sm: int, result: CompileResult):
        key = self.compute_hash(source, sm)
        self._index[key] = asdict(result)
        self._save_index()

    def clear(self):
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._index = {}
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, int]:
        total = len(self._index)
        success = sum(1 for e in self._index.values() if e["success"])
        return {
            "total": total,
            "success": success,
            "fail": total - success,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses)
        }


# Global cache
_cache: Optional[CompileCache] = None


def get_cache() -> CompileCache:
    global _cache
    if _cache is None:
        _cache = CompileCache()
    return _cache


def clear_cache():
    get_cache().clear()


def get_sm_version() -> int:
    """Detect GPU SM version"""
    try:
        import torch
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    except:
        return 75


def compile_cuda(source_path: str, sm: int, use_cache: bool = True) -> CompileResult:
    """
    Compile .cu file to cubin.

    Args:
        source_path: Path to CUDA source file
        sm: Target SM version (e.g., 75 for Turing)
        use_cache: Whether to use compile cache

    Returns:
        CompileResult with success status and cubin path or error
    """
    with open(source_path) as f:
        source = f.read()

    return compile_source(source, sm, use_cache)


def compile_source(source: str, sm: int, use_cache: bool = True) -> CompileResult:
    """
    Compile CUDA source string to cubin.

    Args:
        source: CUDA source code
        sm: Target SM version
        use_cache: Whether to use compile cache

    Returns:
        CompileResult with success status and cubin path or error
    """
    cache = get_cache() if use_cache else None

    # Check cache
    if cache:
        cached = cache.get(source, sm)
        if cached is not None:
            return cached

    # Determine output path
    source_hash = CompileCache.compute_hash(source, sm)
    if cache:
        cubin_path = str(cache.cache_dir / f"{source_hash}.cubin")
    else:
        cubin_path = tempfile.mktemp(suffix=".cubin")

    # Write source to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(source)
        temp_path = f.name

    try:
        cmd = [
            "nvcc",
            "-cubin",
            f"-arch=sm_{sm}",
            "-O3",
            "--use_fast_math",
            "-o", cubin_path,
            temp_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            compile_result = CompileResult(
                success=True,
                cubin_path=cubin_path,
                error_msg=None,
                sm=sm,
                source_hash=source_hash
            )
        else:
            compile_result = CompileResult(
                success=False,
                cubin_path=None,
                error_msg=result.stderr[:1000],
                sm=sm,
                source_hash=source_hash
            )

        if cache:
            cache.put(source, sm, compile_result)

        return compile_result

    except subprocess.TimeoutExpired:
        return CompileResult(
            success=False,
            cubin_path=None,
            error_msg="Compilation timeout",
            sm=sm,
            source_hash=source_hash
        )
    except Exception as e:
        return CompileResult(
            success=False,
            cubin_path=None,
            error_msg=str(e),
            sm=sm,
            source_hash=source_hash
        )
    finally:
        os.unlink(temp_path)
