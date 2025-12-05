"""
PTX Instruction Database - NN-Consultable during training/generation

The NN queries this DB to:
1. Get valid instructions for target SM
2. Check if a generated instruction will work
3. Get alternatives when an instruction isn't available
4. Record compile/run results to learn what works

Bootstrap from our examples, grows through training feedback.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import IntEnum
import json
import re


class TestStatus(IntEnum):
    """Status of an instruction on a particular SM"""
    UNTESTED = 0
    COMPILE_FAIL = 1
    COMPILE_OK = 2
    RUN_FAIL = 3
    RUN_OK = 4


@dataclass
class InstructionEntry:
    """Single PTX instruction pattern"""
    opcode: str                    # Base opcode: "ld.global", "fma.rn.f16x2"
    modifiers: Tuple[str, ...]     # Additional modifiers: (".v4", ".u64")
    min_sm: int = 50               # Minimum SM version (theoretical)
    status: Dict[int, TestStatus] = field(default_factory=dict)  # sm -> tested status
    perf_score: Dict[int, float] = field(default_factory=dict)   # sm -> relative perf
    alternatives: List[str] = field(default_factory=list)        # fallback patterns
    description: str = ""          # Human-readable description

    @property
    def canonical(self) -> str:
        """Full instruction string"""
        return self.opcode + "".join(self.modifiers)

    def is_available(self, target_sm: int) -> bool:
        """Check if instruction is available on target SM"""
        return self.min_sm <= target_sm

    def get_status(self, target_sm: int) -> TestStatus:
        """Get tested status for SM"""
        return self.status.get(target_sm, TestStatus.UNTESTED)

    def to_dict(self) -> dict:
        """Serialize for JSON storage"""
        return {
            "opcode": self.opcode,
            "modifiers": list(self.modifiers),
            "min_sm": self.min_sm,
            "status": {str(k): v.value for k, v in self.status.items()},
            "perf_score": {str(k): v for k, v in self.perf_score.items()},
            "alternatives": self.alternatives,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InstructionEntry":
        """Deserialize from JSON"""
        return cls(
            opcode=d["opcode"],
            modifiers=tuple(d["modifiers"]),
            min_sm=d["min_sm"],
            status={int(k): TestStatus(v) for k, v in d.get("status", {}).items()},
            perf_score={int(k): v for k, v in d.get("perf_score", {}).items()},
            alternatives=d.get("alternatives", []),
            description=d.get("description", ""),
        )


class PTXDB:
    """
    PTX Instruction Database

    Usage:
        db = PTXDB()

        # Check what's available
        ops = db.available_for(75)  # SM 7.5

        # Validate an instruction
        ok, alt = db.check("ld.global.L2::128B", 75)
        if not ok and alt:
            use(alt)  # fallback

        # Record training results
        db.record_result("fma.rn.f16x2", 75, TestStatus.RUN_OK, perf=1.0)

        # Save learned knowledge
        db.save("ptx_db.json")
    """

    def __init__(self, load_path: Optional[str] = None, bootstrap: bool = False):
        self.instructions: Dict[str, InstructionEntry] = {}
        if bootstrap:
            self._bootstrap()  # Only bootstrap if explicitly requested
        if load_path:
            self.load(load_path)

    def _add(self, opcode: str, modifiers: Tuple[str, ...], min_sm: int,
             desc: str = "", alternatives: List[str] = None):
        """Add instruction to database"""
        key = opcode + "".join(modifiers)
        self.instructions[key] = InstructionEntry(
            opcode=opcode,
            modifiers=modifiers,
            min_sm=min_sm,
            description=desc,
            alternatives=alternatives or [],
        )

    def _bootstrap(self):
        """Initialize with known instructions from our examples"""

        # =========================================================
        # UNIVERSAL (SM 50+ / Maxwell)
        # =========================================================

        # Basic loads
        self._add("ld.global", (), 50, "Global memory load")
        self._add("ld.global", (".u8",), 50, "Load unsigned 8-bit")
        self._add("ld.global", (".u16",), 50, "Load unsigned 16-bit")
        self._add("ld.global", (".u32",), 50, "Load unsigned 32-bit")
        self._add("ld.global", (".u64",), 50, "Load unsigned 64-bit")
        self._add("ld.global", (".s8",), 50, "Load signed 8-bit")
        self._add("ld.global", (".s16",), 50, "Load signed 16-bit")
        self._add("ld.global", (".s32",), 50, "Load signed 32-bit")
        self._add("ld.global", (".s64",), 50, "Load signed 64-bit")
        self._add("ld.global", (".f32",), 50, "Load float32")
        self._add("ld.global", (".f64",), 50, "Load float64")

        # Vector loads
        self._add("ld.global", (".v2", ".u32"), 50, "Load 2x u32")
        self._add("ld.global", (".v4", ".u32"), 50, "Load 4x u32")
        self._add("ld.global", (".v2", ".u64"), 50, "Load 2x u64")
        self._add("ld.global", (".v4", ".u64"), 50, "Load 4x u64")
        self._add("ld.global", (".v2", ".f32"), 50, "Load 2x f32")
        self._add("ld.global", (".v4", ".f32"), 50, "Load 4x f32")

        # Cache streaming hints (basic)
        self._add("ld.global", (".cs",), 50, "Cache streaming (don't cache in L1)",
                  alternatives=["ld.global"])
        self._add("ld.global", (".lu",), 50, "Last use (evict after read)",
                  alternatives=["ld.global"])
        self._add("ld.global", (".cv",), 50, "Cache volatile",
                  alternatives=["ld.global"])

        # Combined cache + type
        self._add("ld.global", (".cs", ".u64"), 50, "Cache streaming u64")
        self._add("ld.global", (".cs", ".v2", ".u64"), 50, "Cache streaming 2x u64")
        self._add("ld.global", (".lu", ".u16"), 50, "Last use u16")

        # Stores
        self._add("st.global", (), 50, "Global memory store")
        self._add("st.global", (".u32",), 50, "Store u32")
        self._add("st.global", (".u64",), 50, "Store u64")
        self._add("st.global", (".f32",), 50, "Store f32")
        self._add("st.global", (".v2", ".u32"), 50, "Store 2x u32")
        self._add("st.global", (".v4", ".u32"), 50, "Store 4x u32")

        # Shared memory
        self._add("ld.shared", (), 50, "Shared memory load")
        self._add("ld.shared", (".u32",), 50, "Shared load u32")
        self._add("ld.shared", (".f32",), 50, "Shared load f32")
        self._add("st.shared", (), 50, "Shared memory store")
        self._add("st.shared", (".u32",), 50, "Shared store u32")
        self._add("st.shared", (".f32",), 50, "Shared store f32")

        # Data movement
        self._add("mov", (".b16",), 50, "Move 16-bit")
        self._add("mov", (".b32",), 50, "Move 32-bit")
        self._add("mov", (".b64",), 50, "Move 64-bit")
        self._add("mov", (".u16",), 50, "Move u16")
        self._add("mov", (".u32",), 50, "Move u32")
        self._add("mov", (".u64",), 50, "Move u64")
        self._add("mov", (".f32",), 50, "Move f32")
        self._add("mov", (".f64",), 50, "Move f64")

        # Integer arithmetic
        self._add("add", (".s32",), 50, "Add signed 32-bit")
        self._add("add", (".u32",), 50, "Add unsigned 32-bit")
        self._add("add", (".s64",), 50, "Add signed 64-bit")
        self._add("add", (".u64",), 50, "Add unsigned 64-bit")
        self._add("sub", (".s32",), 50, "Sub signed 32-bit")
        self._add("mul", (".lo", ".s32"), 50, "Multiply low 32-bit")
        self._add("mul", (".lo", ".u32"), 50, "Multiply low u32")
        self._add("mul", (".hi", ".s32"), 50, "Multiply high 32-bit")
        self._add("mad", (".lo", ".s32"), 50, "Multiply-add low 32-bit")

        # Float32 arithmetic
        self._add("add", (".f32",), 50, "Add f32")
        self._add("add", (".rn", ".f32"), 50, "Add f32 round nearest")
        self._add("sub", (".f32",), 50, "Sub f32")
        self._add("mul", (".f32",), 50, "Mul f32")
        self._add("mul", (".rn", ".f32"), 50, "Mul f32 round nearest")
        self._add("fma", (".rn", ".f32"), 50, "FMA f32")
        self._add("div", (".rn", ".f32"), 50, "Div f32")
        self._add("sqrt", (".rn", ".f32"), 50, "Sqrt f32")
        self._add("rsqrt", (".approx", ".f32"), 50, "Reciprocal sqrt approx")
        self._add("rcp", (".approx", ".f32"), 50, "Reciprocal approx")

        # Comparisons
        self._add("setp", (".eq", ".s32"), 50, "Set predicate equal s32")
        self._add("setp", (".ne", ".s32"), 50, "Set predicate not equal s32")
        self._add("setp", (".lt", ".s32"), 50, "Set predicate less than s32")
        self._add("setp", (".le", ".s32"), 50, "Set predicate less equal s32")
        self._add("setp", (".gt", ".s32"), 50, "Set predicate greater s32")
        self._add("setp", (".ge", ".s32"), 50, "Set predicate greater equal s32")
        self._add("setp", (".eq", ".f32"), 50, "Set predicate equal f32")
        self._add("setp", (".lt", ".f32"), 50, "Set predicate less than f32")

        # Bitwise
        self._add("and", (".b32",), 50, "Bitwise AND 32-bit")
        self._add("or", (".b32",), 50, "Bitwise OR 32-bit")
        self._add("xor", (".b32",), 50, "Bitwise XOR 32-bit")
        self._add("not", (".b32",), 50, "Bitwise NOT 32-bit")
        self._add("shl", (".b32",), 50, "Shift left 32-bit")
        self._add("shr", (".b32",), 50, "Shift right 32-bit")
        self._add("shr", (".u32",), 50, "Shift right unsigned 32-bit")

        # Control flow
        self._add("bra", (), 50, "Branch")
        self._add("bra", (".uni",), 50, "Branch uniform")
        self._add("ret", (), 50, "Return")
        self._add("exit", (), 50, "Exit")
        self._add("bar", (".sync",), 50, "Barrier sync")

        # Register declarations
        self._add(".reg", (".b8",), 50, "Declare 8-bit register")
        self._add(".reg", (".b16",), 50, "Declare 16-bit register")
        self._add(".reg", (".b32",), 50, "Declare 32-bit register")
        self._add(".reg", (".b64",), 50, "Declare 64-bit register")
        self._add(".reg", (".f32",), 50, "Declare f32 register")
        self._add(".reg", (".f64",), 50, "Declare f64 register")
        self._add(".reg", (".pred",), 50, "Declare predicate register")

        # =========================================================
        # FP16 (SM 53+ / Maxwell GM20x)
        # =========================================================

        self._add(".reg", (".f16",), 53, "Declare f16 register")
        self._add(".reg", (".f16x2",), 53, "Declare f16x2 register")

        self._add("add", (".f16",), 53, "Add f16")
        self._add("add", (".rn", ".f16"), 53, "Add f16 round nearest")
        self._add("add", (".f16x2",), 53, "Add f16x2 (packed)")
        self._add("add", (".rn", ".f16x2"), 53, "Add f16x2 round nearest")

        self._add("sub", (".f16",), 53, "Sub f16")
        self._add("sub", (".rn", ".f16x2"), 53, "Sub f16x2")

        self._add("mul", (".f16",), 53, "Mul f16")
        self._add("mul", (".rn", ".f16"), 53, "Mul f16 round nearest")
        self._add("mul", (".f16x2",), 53, "Mul f16x2 (packed)")
        self._add("mul", (".rn", ".f16x2"), 53, "Mul f16x2 round nearest")

        self._add("fma", (".rn", ".f16"), 53, "FMA f16")
        self._add("fma", (".rn", ".f16x2"), 53, "FMA f16x2 (packed)")

        # Conversions f16 <-> f32
        self._add("cvt", (".f16", ".f32"), 53, "Convert f32 to f16")
        self._add("cvt", (".rn", ".f16", ".f32"), 53, "Convert f32 to f16 round nearest")
        self._add("cvt", (".f32", ".f16"), 53, "Convert f16 to f32")
        self._add("cvt", (".rn", ".f32", ".f16"), 53, "Convert f16 to f32")  # technically no-op rounding

        # Packed f16x2 conversions
        self._add("cvt", (".rn", ".f16x2", ".f32"), 53, "Convert f32 to f16x2")

        # =========================================================
        # VOLTA (SM 70)
        # =========================================================

        # Tensor core MMA (basic)
        self._add("mma", (".sync", ".aligned", ".m8n8k4", ".f16", ".f16"), 70,
                  "Tensor core MMA 8x8x4 f16")

        # =========================================================
        # TURING (SM 75)
        # =========================================================

        # Integer tensor cores
        self._add("mma", (".sync", ".aligned", ".m8n8k16", ".s8"), 75,
                  "Tensor core MMA 8x8x16 int8")
        self._add("mma", (".sync", ".aligned", ".m16n8k16", ".s8"), 75,
                  "Tensor core MMA 16x8x16 int8")

        # =========================================================
        # AMPERE (SM 80+)
        # =========================================================

        # Advanced cache hints
        self._add("ld.global", (".L1::no_allocate",), 80,
                  "Load bypass L1 cache",
                  alternatives=["ld.global.cs", "ld.global"])
        self._add("ld.global", (".L1::evict_last",), 80,
                  "Load L1 evict last",
                  alternatives=["ld.global.lu", "ld.global"])
        self._add("ld.global", (".L1::evict_first",), 80,
                  "Load L1 evict first",
                  alternatives=["ld.global.cs", "ld.global"])
        self._add("ld.global", (".L2::128B",), 80,
                  "Load with L2 128B sector hint",
                  alternatives=["ld.global.cs", "ld.global"])
        self._add("ld.global", (".L2::256B",), 80,
                  "Load with L2 256B sector hint",
                  alternatives=["ld.global.cs", "ld.global"])
        self._add("ld.global", (".L2::evict_first",), 80,
                  "Load L2 evict first",
                  alternatives=["ld.global.cs", "ld.global"])
        self._add("ld.global", (".L2::evict_last",), 80,
                  "Load L2 evict last",
                  alternatives=["ld.global.lu", "ld.global"])

        # Combined Ampere cache hints
        self._add("ld.global", (".L1::no_allocate", ".L2::evict_first", ".L2::256B", ".v4", ".u64"), 80,
                  "Ampere optimized vector load",
                  alternatives=["ld.global.cs.v4.u64", "ld.global.v4.u64"])
        self._add("ld.global", (".L1::evict_last", ".L2::evict_last", ".v4", ".u64"), 80,
                  "Ampere evict-last vector load",
                  alternatives=["ld.global.lu.v4.u64", "ld.global.v4.u64"])
        self._add("ld.global", (".L1::no_allocate", ".v2", ".u16"), 80,
                  "Ampere bypass L1 load 2x u16",
                  alternatives=["ld.global.cs.v2.u16", "ld.global.v2.u16"])
        self._add("ld.global", (".L1::evict_last", ".v2", ".u16"), 80,
                  "Ampere evict-last load 2x u16",
                  alternatives=["ld.global.lu.v2.u16", "ld.global.v2.u16"])

        # Ampere MMA shapes
        self._add("mma", (".sync", ".aligned", ".m16n8k16", ".f16", ".f16"), 80,
                  "Tensor core MMA 16x8x16 f16")
        self._add("mma", (".sync", ".aligned", ".m16n8k8", ".tf32"), 80,
                  "Tensor core MMA TF32")

        # BF16
        self._add(".reg", (".bf16",), 80, "Declare bf16 register")
        self._add(".reg", (".bf16x2",), 80, "Declare bf16x2 register")
        self._add("cvt", (".rn", ".bf16", ".f32"), 80, "Convert f32 to bf16")
        self._add("cvt", (".f32", ".bf16"), 80, "Convert bf16 to f32")

        # =========================================================
        # ADA LOVELACE (SM 89) - FP8
        # =========================================================

        # FP8 e4m3
        self._add(".reg", (".e4m3",), 89, "Declare FP8 e4m3 register")
        self._add("cvt", (".rn", ".f16x2", ".e4m3x2"), 89,
                  "Convert FP8 e4m3x2 to f16x2")
        self._add("cvt", (".rn", ".e4m3x2", ".f16x2"), 89,
                  "Convert f16x2 to FP8 e4m3x2")

        # FP8 e5m2
        self._add(".reg", (".e5m2",), 89, "Declare FP8 e5m2 register")
        self._add("cvt", (".rn", ".f16x2", ".e5m2x2"), 89,
                  "Convert FP8 e5m2x2 to f16x2")

        # =========================================================
        # HOPPER/BLACKWELL (SM 90+) - FP4
        # =========================================================

        # FP4 e2m1
        self._add(".reg", (".e2m1",), 89, "Declare FP4 e2m1 register")  # Available in Ada
        self._add("cvt", (".rn", ".f16x2", ".e2m1x2"), 89,
                  "Convert FP4 e2m1x2 to f16x2")

        # Hopper MMA
        self._add("mma", (".sync", ".aligned", ".m16n8k32", ".f16", ".e4m3"), 90,
                  "Tensor core MMA with FP8 inputs")

    # =========================================================
    # QUERY INTERFACE (for NN)
    # =========================================================

    def available_for(self, target_sm: int) -> List[str]:
        """Get all instruction patterns available on target SM"""
        return [k for k, v in self.instructions.items() if v.min_sm <= target_sm]

    def get(self, instr: str) -> Optional[InstructionEntry]:
        """Get instruction entry by canonical name"""
        return self.instructions.get(instr)

    def check(self, instr: str, target_sm: int) -> Tuple[bool, Optional[str], TestStatus]:
        """
        Check if instruction works on target SM.

        Returns:
            (True, None, status) - instruction available or unknown (try it!)
            (False, alternative, status) - known to fail, here's alternative
            (False, None, status) - known to fail, no alternative

        Philosophy: Unknown = try it, learn from compile/run result
        """
        entry = self.instructions.get(instr)

        if not entry:
            # Unknown instruction - TRY IT, we'll learn from the result
            return True, None, TestStatus.UNTESTED

        # Check if we have test results for this SM
        status = entry.get_status(target_sm)

        if status == TestStatus.UNTESTED:
            # Never tested on this SM - try it
            return True, None, status

        if status in (TestStatus.COMPILE_OK, TestStatus.RUN_OK):
            # Known to work
            return True, None, status

        if status in (TestStatus.COMPILE_FAIL, TestStatus.RUN_FAIL):
            # Known to fail - suggest alternative if available
            for alt in entry.alternatives:
                alt_entry = self.instructions.get(alt)
                if alt_entry:
                    alt_status = alt_entry.get_status(target_sm)
                    if alt_status in (TestStatus.COMPILE_OK, TestStatus.RUN_OK, TestStatus.UNTESTED):
                        return False, alt, status
            return False, None, status

        # Fallback - try it
        return True, None, status

    def _extract_base_opcode(self, instr: str) -> str:
        """Extract base opcode from full instruction string"""
        # Handle things like "ld.global.L2::128B.v4.u64" -> "ld.global"
        parts = instr.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
        return instr

    def parse_ptx_line(self, line: str) -> Optional[str]:
        """
        Parse a PTX line and return the canonical instruction pattern.
        Returns None if not a recognizable instruction.
        """
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("."):
            if line.startswith(".reg"):
                # Register declaration
                m = re.match(r'\.reg\s+(\.\w+)', line)
                if m:
                    return ".reg" + m.group(1)
            return None

        # Match instruction: opcode.mod.mod.mod operands
        m = re.match(r'(\w+(?:\.\w+)*)\s+', line)
        if m:
            return m.group(1)
        return None

    def check_ptx_lines(self, lines: List[str], target_sm: int) -> List[Tuple[str, bool, Optional[str]]]:
        """
        Check multiple PTX lines.

        Returns list of (instruction, ok, alternative) tuples
        """
        results = []
        for line in lines:
            instr = self.parse_ptx_line(line)
            if instr:
                ok, alt = self.check(instr, target_sm)
                results.append((instr, ok, alt))
        return results

    # =========================================================
    # LEARNING INTERFACE (record training feedback)
    # =========================================================

    def record_result(self, instr: str, sm: int, status: TestStatus, perf: float = 1.0):
        """
        Record compilation/runtime result for an instruction.
        Called during training when we get feedback.
        """
        if instr not in self.instructions:
            # Learn new instruction from experience
            base = self._extract_base_opcode(instr)
            self.instructions[instr] = InstructionEntry(
                opcode=base,
                modifiers=tuple(instr.replace(base, "").split(".")[1:]) if base != instr else (),
                min_sm=sm if status in (TestStatus.COMPILE_OK, TestStatus.RUN_OK) else 999,
                description="Learned from training",
            )

        entry = self.instructions[instr]
        entry.status[sm] = status

        if status == TestStatus.RUN_OK:
            entry.perf_score[sm] = perf
            # Update min_sm if we found it works on lower SM than expected
            if sm < entry.min_sm:
                entry.min_sm = sm
        elif status == TestStatus.COMPILE_FAIL:
            # If we thought it worked but it doesn't, update
            if sm >= entry.min_sm:
                # Find next higher SM where it might work
                pass  # Keep min_sm, just record the failure

    def record_batch(self, results: List[Tuple[str, int, TestStatus, float]]):
        """Record multiple results at once"""
        for instr, sm, status, perf in results:
            self.record_result(instr, sm, status, perf)

    # =========================================================
    # EMBEDDING INTERFACE (for NN training)
    # =========================================================

    def get_vocab(self) -> List[str]:
        """Get vocabulary of all known instructions"""
        return sorted(self.instructions.keys())

    def get_embedding_data(self) -> Dict:
        """
        Return structured data for NN embedding layer.

        This can be used to:
        1. Build instruction vocabulary
        2. Create SM-conditional embeddings
        3. Learn instruction relationships
        """
        vocab = self.get_vocab()
        return {
            "vocab": vocab,
            "vocab_size": len(vocab),
            "min_sm": {k: v.min_sm for k, v in self.instructions.items()},
            "alternatives": {k: v.alternatives for k, v in self.instructions.items()},
            "categories": self._categorize_instructions(),
        }

    def _categorize_instructions(self) -> Dict[str, List[str]]:
        """Group instructions by category for structured learning"""
        categories = {
            "load": [],
            "store": [],
            "arithmetic": [],
            "fma": [],
            "convert": [],
            "move": [],
            "compare": [],
            "control": [],
            "tensor": [],
            "register": [],
            "other": [],
        }

        for key in self.instructions:
            if key.startswith("ld."):
                categories["load"].append(key)
            elif key.startswith("st."):
                categories["store"].append(key)
            elif key.startswith(("add", "sub", "mul", "div")):
                categories["arithmetic"].append(key)
            elif key.startswith("fma"):
                categories["fma"].append(key)
            elif key.startswith("cvt"):
                categories["convert"].append(key)
            elif key.startswith("mov"):
                categories["move"].append(key)
            elif key.startswith("setp"):
                categories["compare"].append(key)
            elif key.startswith(("bra", "ret", "exit", "bar")):
                categories["control"].append(key)
            elif key.startswith("mma"):
                categories["tensor"].append(key)
            elif key.startswith(".reg"):
                categories["register"].append(key)
            else:
                categories["other"].append(key)

        return categories

    # =========================================================
    # PERSISTENCE
    # =========================================================

    def save(self, path: str):
        """Save database to JSON file"""
        data = {
            "version": 1,
            "instructions": {k: v.to_dict() for k, v in self.instructions.items()}
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load database from JSON file (merges with bootstrap)"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            for key, entry_dict in data.get("instructions", {}).items():
                if key in self.instructions:
                    # Merge learned data into existing entry
                    existing = self.instructions[key]
                    learned = InstructionEntry.from_dict(entry_dict)
                    existing.status.update(learned.status)
                    existing.perf_score.update(learned.perf_score)
                    if learned.min_sm < existing.min_sm:
                        existing.min_sm = learned.min_sm
                else:
                    # New instruction learned during training
                    self.instructions[key] = InstructionEntry.from_dict(entry_dict)
        except FileNotFoundError:
            pass  # No saved data yet

    # =========================================================
    # UTILITIES
    # =========================================================

    def summary(self, target_sm: Optional[int] = None) -> str:
        """Get human-readable summary"""
        lines = ["PTX Instruction Database"]
        lines.append(f"Total instructions: {len(self.instructions)}")

        if target_sm:
            avail = self.available_for(target_sm)
            lines.append(f"Available for SM {target_sm}: {len(avail)}")

        # By SM tier
        sm_tiers = [50, 53, 70, 75, 80, 89, 90]
        for sm in sm_tiers:
            count = sum(1 for v in self.instructions.values() if v.min_sm == sm)
            if count:
                lines.append(f"  SM {sm}+: {count} instructions")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.instructions)

    def __contains__(self, item: str) -> bool:
        return item in self.instructions


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    # Test 1: Blank slate DB (learns from experience)
    print("=== Blank Slate DB ===")
    db = PTXDB(bootstrap=False)
    print(f"Starting with {len(db)} instructions (should be 0)")

    # Unknown instruction - should say "try it"
    ok, alt, status = db.check("ld.global.v4.u64", 75)
    print(f"Unknown instr: ok={ok}, status={status.name}")
    assert ok == True and status == TestStatus.UNTESTED

    # Learn from compile success
    db.record_result("ld.global.v4.u64", 75, TestStatus.COMPILE_OK)
    ok, alt, status = db.check("ld.global.v4.u64", 75)
    print(f"After compile OK: ok={ok}, status={status.name}")
    assert ok == True and status == TestStatus.COMPILE_OK

    # Learn from compile fail
    db.record_result("ld.global.L2::128B", 75, TestStatus.COMPILE_FAIL)
    ok, alt, status = db.check("ld.global.L2::128B", 75)
    print(f"After compile FAIL: ok={ok}, status={status.name}")
    assert ok == False and status == TestStatus.COMPILE_FAIL

    print(f"DB now has {len(db)} learned instructions")
    print()

    # Test 2: Bootstrapped DB (has prior knowledge)
    print("=== Bootstrapped DB ===")
    db2 = PTXDB(bootstrap=True)
    print(f"Bootstrapped with {len(db2)} instructions")

    # Check known instruction
    ok, alt, status = db2.check("fma.rn.f16x2", 75)
    print(f"fma.rn.f16x2 on SM 75: ok={ok}, status={status.name}")

    # Check instruction that needs higher SM
    ok, alt, status = db2.check("cvt.rn.f16x2.e2m1x2", 75)
    print(f"cvt.rn.f16x2.e2m1x2 on SM 75: ok={ok}, alt={alt}, status={status.name}")

    print()
    print("=== All tests passed ===")
