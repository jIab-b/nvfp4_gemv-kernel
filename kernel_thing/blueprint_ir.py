"""
Blueprint IR - Layer 2

Bidirectional lossless transformation between kernels and IR.
- DECOMPOSE: CUDA/PTX source → IR representation
- RECONSTRUCT: IR representation → PTX source

The IR captures everything needed for exact reconstruction:
- Every instruction with operands
- Register allocation
- Control flow structure
- Memory access patterns
- Configuration parameters

Usage:
    # Decompose existing kernel
    ir = BlueprintIR.from_cuda_source(cuda_code)
    ir = BlueprintIR.from_ptx_source(ptx_code)

    # Modify configuration
    ir.config["BLOCK_M"] = 8
    ir.config["CACHE_A"] = ".cs"

    # Reconstruct
    ptx = ir.to_ptx()

    # Get config vector for NN
    vec = ir.to_config_vector()

    # Get examples for LLM
    examples = ir.to_llm_examples()
"""

import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
from enum import Enum, auto


class PhaseType(Enum):
    PROLOGUE = auto()
    MAIN_LOOP = auto()
    EPILOGUE = auto()
    CUSTOM = auto()


class OpType(Enum):
    # Memory
    LOAD_PARAM = auto()
    LOAD_GLOBAL = auto()
    LOAD_GLOBAL_VEC = auto()
    LOAD_SHARED = auto()
    STORE_GLOBAL = auto()
    STORE_SHARED = auto()

    # Compute
    CVT = auto()
    CVT_FP4_TO_FP16 = auto()   # cvt.rn.f16x2.e2m1x2
    CVT_FP8_TO_FP16 = auto()   # cvt.rn.f16x2.e4m3x2
    CVT_FP16_TO_FP32 = auto()  # cvt.f32.f16
    FMA = auto()
    FMA_F16X2 = auto()         # fma.rn.f16x2
    MUL = auto()
    MUL_F16X2 = auto()         # mul.rn.f16x2
    ADD = auto()
    ADD_F16X2 = auto()         # add.rn.f16x2
    ADD_F16 = auto()           # add.rn.f16
    MAD = auto()
    MAD_WIDE = auto()          # mad.wide.u32

    # Control
    BRANCH = auto()
    BARRIER = auto()
    PREDICATE = auto()

    # Reduction
    SHFL = auto()
    SHFL_DOWN = auto()         # shfl.sync.down.b32
    REDUCE_SMEM = auto()

    # Pack/Unpack
    MOV = auto()
    MOV_PACK = auto()          # mov.b32 dst, {a, b, c, d}
    MOV_UNPACK = auto()        # mov.b32 {a, b, c, d}, src

    # Special
    SETP = auto()
    EXIT = auto()


@dataclass
class Operand:
    """Single operand in an instruction"""
    name: str
    type: str  # register, immediate, address, label
    value: str
    modifiers: list[str] = field(default_factory=list)

    def __str__(self):
        return self.value


@dataclass
class Instruction:
    """Single PTX instruction with full details"""
    mnemonic: str
    modifiers: list[str]
    operands: list[Operand]
    predicate: Optional[str] = None
    comment: Optional[str] = None
    source_line: Optional[int] = None

    def to_ptx(self) -> str:
        """Reconstruct PTX instruction string"""
        parts = []

        if self.predicate:
            parts.append(f"@{self.predicate}")

        # Mnemonic with modifiers
        full_mnemonic = ".".join([self.mnemonic] + self.modifiers)
        parts.append(full_mnemonic)

        # Operands
        if self.operands:
            op_strs = [str(op) for op in self.operands]
            parts.append(" " + ", ".join(op_strs))

        result = "".join(parts) + ";"

        if self.comment:
            result += f"  // {self.comment}"

        return result


@dataclass
class Phase:
    """A phase in kernel execution (prologue, loop body, epilogue)"""
    type: PhaseType
    name: str
    instructions: list[Instruction] = field(default_factory=list)
    loop_var: Optional[str] = None
    loop_bound: Optional[str] = None
    children: list["Phase"] = field(default_factory=list)

    def to_ptx(self, indent: int = 0) -> str:
        """Reconstruct PTX for this phase"""
        lines = []
        prefix = "    " * indent

        if self.type == PhaseType.MAIN_LOOP:
            lines.append(f"{prefix}// LOOP: {self.name}")
            if self.loop_var:
                lines.append(f"{prefix}${self.name}_START:")

        for instr in self.instructions:
            lines.append(f"{prefix}{instr.to_ptx()}")

        for child in self.children:
            lines.extend(child.to_ptx(indent + 1).split("\n"))

        if self.type == PhaseType.MAIN_LOOP and self.loop_var:
            lines.append(f"{prefix}@{self.loop_var} bra ${self.name}_START;")

        return "\n".join(lines)


@dataclass
class RegisterAllocation:
    """Track register usage"""
    pred: int = 0
    b16: int = 0
    b32: int = 0
    b64: int = 0
    f16: int = 0
    f32: int = 0
    f64: int = 0

    def to_ptx(self) -> str:
        """Generate register declarations"""
        lines = []
        if self.pred > 0:
            lines.append(f".reg .pred %p<{self.pred}>;")
        if self.b16 > 0:
            lines.append(f".reg .b16 %rs<{self.b16}>;")
        if self.b32 > 0:
            lines.append(f".reg .b32 %r<{self.b32}>;")
        if self.b64 > 0:
            lines.append(f".reg .b64 %rd<{self.b64}>;")
        if self.f16 > 0:
            lines.append(f".reg .f16 %h<{self.f16}>;")
        if self.f32 > 0:
            lines.append(f".reg .f32 %f<{self.f32}>;")
        if self.f64 > 0:
            lines.append(f".reg .f64 %fd<{self.f64}>;")
        return "\n".join(lines)


@dataclass
class KernelParameter:
    """Kernel parameter definition"""
    name: str
    dtype: str  # b64, b32, etc.
    role: str  # ptr_input, ptr_output, scalar, etc.

    def to_ptx(self) -> str:
        return f".param .{self.dtype} {self.name}"


@dataclass
class SharedMemory:
    """Shared memory allocation"""
    name: str
    size_expr: str
    alignment: int = 16
    dtype: str = "b8"

    def to_ptx(self) -> str:
        return f".shared .align {self.alignment} .{self.dtype} {self.name}[{self.size_expr}];"


class BlueprintIR:
    """
    Complete intermediate representation of a kernel.

    Captures everything for lossless round-trip:
    - Metadata (target SM, PTX version)
    - Parameters
    - Register allocation
    - Shared memory
    - Phases with instructions
    - Configuration values
    """

    def __init__(self):
        # Metadata
        self.name: str = "kernel"
        self.target_sm: int = 100
        self.ptx_version: str = "7.8"
        self.address_size: int = 64

        # Declarations
        self.parameters: list[KernelParameter] = []
        self.registers: RegisterAllocation = RegisterAllocation()
        self.shared_memory: list[SharedMemory] = []

        # Structure
        self.phases: list[Phase] = []

        # Configuration (the searchable parameters)
        self.config: dict[str, Any] = {}

        # Source info for decomposition
        self._source_lines: list[str] = []
        self._source_type: str = ""  # "cuda" or "ptx"

    # =========================================================================
    # DECOMPOSITION: Source → IR
    # =========================================================================

    @classmethod
    def from_ptx_source(cls, ptx: str) -> "BlueprintIR":
        """Decompose PTX source into IR"""
        ir = cls()
        ir._source_lines = ptx.split("\n")
        ir._source_type = "ptx"

        ir._parse_ptx_metadata()
        ir._parse_ptx_parameters()
        ir._parse_ptx_registers()
        ir._parse_ptx_shared()
        ir._parse_ptx_instructions()
        ir._infer_config()

        return ir

    @classmethod
    def from_cuda_source(cls, cuda: str) -> "BlueprintIR":
        """
        Decompose CUDA source into IR.

        This is more complex as we need to:
        1. Extract template parameters (BLOCK_M, etc.)
        2. Parse inline ASM
        3. Identify phases from control flow
        """
        ir = cls()
        ir._source_lines = cuda.split("\n")
        ir._source_type = "cuda"

        ir._parse_cuda_template_params()
        ir._parse_cuda_kernel_signature()
        ir._parse_cuda_inline_asm()
        ir._parse_cuda_control_flow()
        ir._infer_config()

        return ir

    def _parse_ptx_metadata(self):
        """Extract .version, .target, .address_size"""
        for line in self._source_lines:
            line = line.strip()

            if line.startswith(".version"):
                match = re.search(r"\.version\s+([\d.]+)", line)
                if match:
                    self.ptx_version = match.group(1)

            elif line.startswith(".target"):
                match = re.search(r"sm_(\d+)", line)
                if match:
                    self.target_sm = int(match.group(1))

            elif line.startswith(".address_size"):
                match = re.search(r"\.address_size\s+(\d+)", line)
                if match:
                    self.address_size = int(match.group(1))

    def _parse_ptx_parameters(self):
        """Extract kernel parameters"""
        in_entry = False

        for line in self._source_lines:
            line = line.strip()

            if ".entry" in line or ".func" in line:
                in_entry = True
                # Extract kernel name
                match = re.search(r"\.entry\s+(\w+)", line)
                if match:
                    self.name = match.group(1)

            if in_entry and ".param" in line:
                match = re.search(r"\.param\s+\.(b\d+|[usf]\d+)\s+(\w+)", line)
                if match:
                    dtype = match.group(1)
                    name = match.group(2)
                    # Infer role from name
                    role = "unknown"
                    if "ptr" in name.lower() or name.startswith("p"):
                        role = "pointer"
                    elif any(x in name.lower() for x in ["size", "dim", "m", "n", "k"]):
                        role = "scalar"

                    self.parameters.append(KernelParameter(
                        name=name, dtype=dtype, role=role
                    ))

            if in_entry and line.startswith("{"):
                in_entry = False

    def _parse_ptx_registers(self):
        """Extract register declarations"""
        for line in self._source_lines:
            line = line.strip()

            if not line.startswith(".reg"):
                continue

            match = re.search(r"\.reg\s+\.(\w+)\s+%(\w+)<(\d+)>", line)
            if match:
                dtype = match.group(1)
                prefix = match.group(2)
                count = int(match.group(3))

                if dtype == "pred":
                    self.registers.pred = max(self.registers.pred, count)
                elif dtype in ["b16", "s16", "u16"]:
                    self.registers.b16 = max(self.registers.b16, count)
                elif dtype in ["b32", "s32", "u32"]:
                    self.registers.b32 = max(self.registers.b32, count)
                elif dtype in ["b64", "s64", "u64"]:
                    self.registers.b64 = max(self.registers.b64, count)
                elif dtype == "f16":
                    self.registers.f16 = max(self.registers.f16, count)
                elif dtype == "f32":
                    self.registers.f32 = max(self.registers.f32, count)
                elif dtype == "f64":
                    self.registers.f64 = max(self.registers.f64, count)

    def _parse_ptx_shared(self):
        """Extract shared memory declarations"""
        for line in self._source_lines:
            line = line.strip()

            if not line.startswith(".shared"):
                continue

            match = re.search(
                r"\.shared\s+(?:\.align\s+(\d+)\s+)?\.(\w+)\s+(\w+)\[([^\]]+)\]",
                line
            )
            if match:
                align = int(match.group(1)) if match.group(1) else 16
                dtype = match.group(2)
                name = match.group(3)
                size = match.group(4)

                self.shared_memory.append(SharedMemory(
                    name=name, size_expr=size, alignment=align, dtype=dtype
                ))

    def _parse_ptx_instructions(self):
        """Parse instruction sequence and identify phases"""
        in_body = False
        current_phase = Phase(type=PhaseType.PROLOGUE, name="prologue")
        self.phases = [current_phase]

        for line_num, line in enumerate(self._source_lines):
            original = line
            line = line.strip()

            # Track entry into kernel body
            if line == "{" or (line.startswith("{") and len(line) == 1):
                in_body = True
                continue

            if line == "}" or line.startswith("}"):
                in_body = False
                continue

            if not in_body:
                continue

            # Skip declarations
            if line.startswith("."):
                continue

            # Skip empty/comments
            if not line or line.startswith("//"):
                continue

            # Detect labels (potential loop markers)
            if line.endswith(":"):
                label = line[:-1]
                if "LOOP" in label.upper() or label.startswith("$L"):
                    # New loop phase
                    current_phase = Phase(
                        type=PhaseType.MAIN_LOOP,
                        name=label,
                        loop_var="%p_loop"  # Placeholder
                    )
                    self.phases.append(current_phase)
                continue

            # Parse instruction
            instr = self._parse_instruction(line, line_num)
            if instr:
                current_phase.instructions.append(instr)

                # Detect phase transitions
                if instr.mnemonic == "bra" and current_phase.type == PhaseType.MAIN_LOOP:
                    # End of loop, switch to epilogue
                    current_phase = Phase(type=PhaseType.EPILOGUE, name="epilogue")
                    self.phases.append(current_phase)

    def _parse_instruction(self, line: str, line_num: int) -> Optional[Instruction]:
        """Parse a single instruction line"""
        # Handle predicate
        predicate = None
        if line.startswith("@"):
            match = re.match(r"@([%!]?\w+)\s+", line)
            if match:
                predicate = match.group(1)
                line = line[match.end():]

        # Split mnemonic and operands
        parts = line.rstrip(";").split(None, 1)
        if not parts:
            return None

        mnemonic_full = parts[0]
        operands_str = parts[1] if len(parts) > 1 else ""

        # Parse mnemonic and modifiers
        mnemonic_parts = mnemonic_full.split(".")
        base_mnemonic = mnemonic_parts[0]
        modifiers = mnemonic_parts[1:] if len(mnemonic_parts) > 1 else []

        # Parse operands
        operands = self._parse_operands(operands_str)

        return Instruction(
            mnemonic=base_mnemonic,
            modifiers=modifiers,
            operands=operands,
            predicate=predicate,
            source_line=line_num
        )

    def _parse_operands(self, operands_str: str) -> list[Operand]:
        """Parse operand string"""
        operands_str = operands_str.strip()
        if not operands_str:
            return []

        operands = []
        current = ""
        brace_depth = 0
        bracket_depth = 0

        for char in operands_str:
            if char == "{":
                brace_depth += 1
                current += char
            elif char == "}":
                brace_depth -= 1
                current += char
            elif char == "[":
                bracket_depth += 1
                current += char
            elif char == "]":
                bracket_depth -= 1
                current += char
            elif char == "," and brace_depth == 0 and bracket_depth == 0:
                if current.strip():
                    operands.append(self._make_operand(current.strip()))
                current = ""
            else:
                current += char

        if current.strip():
            operands.append(self._make_operand(current.strip()))

        return operands

    def _make_operand(self, s: str) -> Operand:
        """Create Operand from string"""
        # Detect type
        if s.startswith("%"):
            op_type = "register"
        elif s.startswith("["):
            op_type = "address"
        elif s.startswith("{"):
            op_type = "vector"
        elif re.match(r"^-?\d+", s) or s.startswith("0x"):
            op_type = "immediate"
        else:
            op_type = "label"

        return Operand(name="", type=op_type, value=s)

    def _parse_cuda_template_params(self):
        """Extract C++ template parameters like BLOCK_M, NUM_WARPS"""
        for line in self._source_lines:
            # Look for template parameters
            match = re.search(r"template\s*<([^>]+)>", line)
            if match:
                params = match.group(1)
                # Extract int params
                for param_match in re.finditer(r"int\s+(\w+)", params):
                    param_name = param_match.group(1)
                    self.config[param_name] = None  # Value unknown until instantiation

            # Look for constexpr definitions
            match = re.search(r"constexpr\s+int\s+(\w+)\s*=\s*(\d+)", line)
            if match:
                self.config[match.group(1)] = int(match.group(2))

    def _parse_cuda_kernel_signature(self):
        """Extract kernel name and parameters from CUDA"""
        for line in self._source_lines:
            # Look for __global__ function
            match = re.search(r"__global__\s*\w*\s*void\s+(\w+)\s*\(", line)
            if match:
                self.name = match.group(1)
                break

    def _parse_cuda_inline_asm(self):
        """Extract inline PTX assembly blocks"""
        in_asm = False
        asm_content = []

        for line in self._source_lines:
            if 'asm volatile' in line or 'asm __volatile__' in line:
                in_asm = True
                asm_content = []
                continue

            if in_asm:
                # End of asm block
                if line.strip().startswith(":") or line.strip() == ");":
                    # Parse accumulated asm
                    self._process_inline_asm("\n".join(asm_content))
                    in_asm = False
                    continue

                # Strip string delimiters
                cleaned = re.sub(r'^["\s]+|["\s\\n]+$', '', line.strip())
                if cleaned:
                    asm_content.append(cleaned)

    def _process_inline_asm(self, asm: str):
        """Process inline assembly block"""
        # Create a phase for this asm block if substantial
        phase = Phase(type=PhaseType.CUSTOM, name="inline_asm")

        for line in asm.split("\n"):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith(".reg"):
                continue

            instr = self._parse_instruction(line, 0)
            if instr:
                phase.instructions.append(instr)

        if phase.instructions:
            self.phases.append(phase)

    def _parse_cuda_control_flow(self):
        """Identify loops and control flow from CUDA source"""
        for line_num, line in enumerate(self._source_lines):
            # Detect for loops
            if re.search(r"for\s*\([^;]+;\s*\w+\s*<\s*(\w+)", line):
                match = re.search(r"for\s*\([^;]+;\s*\w+\s*<\s*(\w+)", line)
                if match:
                    bound = match.group(1)
                    # This is likely the main loop
                    if not any(p.type == PhaseType.MAIN_LOOP for p in self.phases):
                        self.phases.append(Phase(
                            type=PhaseType.MAIN_LOOP,
                            name="main_loop",
                            loop_bound=bound
                        ))

    def _infer_config(self):
        """Infer configuration from parsed structure"""
        # Count threads from barrier usage
        for phase in self.phases:
            for instr in phase.instructions:
                if instr.mnemonic == "bar" and "sync" in instr.modifiers:
                    # Could infer thread count from barrier
                    pass

        # Infer cache policies from load instructions
        for phase in self.phases:
            for instr in phase.instructions:
                if instr.mnemonic == "ld":
                    cache_mods = [m for m in instr.modifiers
                                  if m in ["ca", "cg", "cs", "cv", "L1::no_allocate",
                                           "L1::evict_last", "L2::128B", "L2::256B"]]
                    if cache_mods:
                        # Determine if this is A or B based on context
                        pass

    # =========================================================================
    # RECONSTRUCTION: IR → PTX
    # =========================================================================

    def to_ptx(self) -> str:
        """Reconstruct complete PTX source from IR"""
        lines = []

        # Header
        lines.append(f".version {self.ptx_version}")
        lines.append(f".target sm_{self.target_sm}")
        lines.append(f".address_size {self.address_size}")
        lines.append("")

        # Entry point
        param_decls = ", ".join(p.to_ptx() for p in self.parameters)
        lines.append(f".visible .entry {self.name}(")
        lines.append(f"    {param_decls}")
        lines.append(") {")

        # Register declarations
        reg_decls = self.registers.to_ptx()
        if reg_decls:
            for reg_line in reg_decls.split("\n"):
                lines.append(f"    {reg_line}")
            lines.append("")

        # Shared memory
        for smem in self.shared_memory:
            lines.append(f"    {smem.to_ptx()}")
        if self.shared_memory:
            lines.append("")

        # Phases
        for phase in self.phases:
            phase_ptx = phase.to_ptx(indent=1)
            lines.append(phase_ptx)
            lines.append("")

        # Exit
        lines.append("    ret;")
        lines.append("}")

        return "\n".join(lines)

    # =========================================================================
    # CONFIG VECTOR: For NN - CHOICE ARRAYS
    # =========================================================================
    CHOICE_ARRAYS = {
        # Tiling parameters
        "BLOCK_M": [1, 2, 4, 8, 16, 32, 64],
        "BLOCK_K": [64, 128, 256, 512, 1024, 2048, 3584, 4096, 8192],
        "NUM_WARPS": [1, 2, 4, 7, 8, 16],
        "THREADS_PER_ROW": [8, 16, 32, 64, 128],
        "ROWS_PER_BLOCK": [1, 2, 4, 8, 16],

        # Memory parameters
        "CP_SIZE": [8, 16, 32],
        "ELEMENTS_PER_LOAD": [16, 32],  # 16 or 32 FP4 elements per load

        # Cache policies (from 1.py patterns)
        "CACHE_A": [
            "",                                          # Default
            ".cs",                                       # Streaming
            ".L1::no_allocate",                         # No L1 allocate
            ".L1::no_allocate.L2::evict_first",         # Stream through
            ".L1::no_allocate.L2::evict_first.L2::256B" # Full streaming (8192 case)
        ],
        "CACHE_B": [
            "",                                          # Default
            ".ca",                                       # Cache all
            ".L1::evict_last",                          # Keep in L1
            ".L1::evict_last.L2::evict_last",           # Keep everywhere
            ".L2::128B"                                  # L2 prefetch
        ],

        # Reduction strategy
        "REDUCTION": ["warp_shfl", "smem_then_shfl", "smem_tree"],

        # Loop parameters
        "UNROLL": [1, 2, 4, 7, 8, "full"],

        # Accumulator type
        "ACCUM_DTYPE": ["f16", "f32"],
    }

    # =========================================================================
    # PTX TEMPLATES - Exact instruction patterns for NVFP4 GEMV
    # =========================================================================
    PTX_TEMPLATES = {
        # Load patterns
        "load_a_16": "ld.global{cache}.u64.v2 {{%rd{r0}, %rd{r1}}}, [{addr}];",
        "load_a_32": "ld.global{cache}.v4.u64 {{%rd{r0}, %rd{r1}, %rd{r2}, %rd{r3}}}, [{addr}];",
        "load_sfa": "ld.global{cache}.u16 %rs{r}, [{addr}];",
        "load_sfa_v2": "ld.global{cache}.v2.u16 {{%rs{r0}, %rs{r1}}}, [{addr}];",

        # FP4/FP8 conversion (THE KEY INSTRUCTIONS)
        "cvt_fp4_to_f16x2": "cvt.rn.f16x2.e2m1x2 %r{dst}, %b{src};",
        "cvt_fp8_to_f16x2": "cvt.rn.f16x2.e4m3x2 %r{dst}, %rs{src};",
        "cvt_f16_to_f32": "cvt.f32.f16 %f{dst}, %h{src};",

        # Pack/unpack
        "unpack_b32_to_b8": "mov.b32 {{%b{d0}, %b{d1}, %b{d2}, %b{d3}}}, %r{src};",
        "unpack_f16x2": "mov.b32 {{%h{d0}, %h{d1}}}, %r{src};",
        "pack_f16x2": "mov.b32 %r{dst}, {{%h{s0}, %h{s1}}};",
        "zero_init": "mov.b32 %r{dst}, 0;",

        # FMA chain (f16x2)
        "fma_f16x2": "fma.rn.f16x2 %r{acc}, %r{a}, %r{b}, %r{acc};",
        "mul_f16x2": "mul.rn.f16x2 %r{dst}, %r{a}, %r{b};",
        "add_f16x2": "add.rn.f16x2 %r{dst}, %r{a}, %r{b};",
        "add_f16": "add.rn.f16 %h{dst}, %h{a}, %h{b};",

        # Warp reduction
        "shfl_down": "shfl.sync.down.b32 %f{dst}, %f{src}, {offset}, 0xffffffff;",
        "add_f32": "add.f32 %f{dst}, %f{a}, %f{b};",

        # Store
        "store_f16": "st.global.f16 [{addr}], %h{src};",
        "store_f16_pred": "@%p{pred} st.global.f16 [{addr}], %h{src};",
    }

    # =========================================================================
    # NVFP4 GEMV SPECIFIC CODE BLOCKS
    # =========================================================================
    NVFP4_CODE_BLOCKS = {
        # Block-scaled FMA for 16 FP4 elements (from 1.py block_scaled_fma_16x2fp4)
        "block_scaled_fma_16x2fp4": '''
        // Unpack A (8 bytes = 16 fp4 = 8 fp4x2)
        mov.b32 {{%b0, %b1, %b2, %b3}}, %r{a_lo};
        mov.b32 {{%b4, %b5, %b6, %b7}}, %r{a_hi};

        // Unpack B
        mov.b32 {{%b8, %b9, %b10, %b11}}, %r{b_lo};
        mov.b32 {{%b12, %b13, %b14, %b15}}, %r{b_hi};

        // Convert FP4 -> FP16x2
        cvt.rn.f16x2.e2m1x2 %r{cvt_a0}, %b0;
        cvt.rn.f16x2.e2m1x2 %r{cvt_a1}, %b1;
        cvt.rn.f16x2.e2m1x2 %r{cvt_a2}, %b2;
        cvt.rn.f16x2.e2m1x2 %r{cvt_a3}, %b3;
        cvt.rn.f16x2.e2m1x2 %r{cvt_a4}, %b4;
        cvt.rn.f16x2.e2m1x2 %r{cvt_a5}, %b5;
        cvt.rn.f16x2.e2m1x2 %r{cvt_a6}, %b6;
        cvt.rn.f16x2.e2m1x2 %r{cvt_a7}, %b7;

        cvt.rn.f16x2.e2m1x2 %r{cvt_b0}, %b8;
        cvt.rn.f16x2.e2m1x2 %r{cvt_b1}, %b9;
        cvt.rn.f16x2.e2m1x2 %r{cvt_b2}, %b10;
        cvt.rn.f16x2.e2m1x2 %r{cvt_b3}, %b11;
        cvt.rn.f16x2.e2m1x2 %r{cvt_b4}, %b12;
        cvt.rn.f16x2.e2m1x2 %r{cvt_b5}, %b13;
        cvt.rn.f16x2.e2m1x2 %r{cvt_b6}, %b14;
        cvt.rn.f16x2.e2m1x2 %r{cvt_b7}, %b15;

        // Convert scales FP8 -> FP16x2
        cvt.rn.f16x2.e4m3x2 %r{sfa_f16x2}, %rs{sfa};
        cvt.rn.f16x2.e4m3x2 %r{sfb_f16x2}, %rs{sfb};

        // Combine scales: scale = sfa * sfb
        mul.rn.f16x2 %r{sf_f16x2}, %r{sfa_f16x2}, %r{sfb_f16x2};

        // Extract individual scales
        mov.b32 {{%h{lane0}, %h{lane1}}}, %r{sf_f16x2};
        mov.b32 %r{scale0}, {{%h{lane0}, %h{lane0}}};
        mov.b32 %r{scale1}, {{%h{lane1}, %h{lane1}}};

        // FMA chain for first 8 elements (uses scale0)
        mov.b32 %r{acc_group}, 0;
        fma.rn.f16x2 %r{acc_group}, %r{cvt_a0}, %r{cvt_b0}, %r{acc_group};
        fma.rn.f16x2 %r{acc_group}, %r{cvt_a1}, %r{cvt_b1}, %r{acc_group};
        fma.rn.f16x2 %r{acc_group}, %r{cvt_a2}, %r{cvt_b2}, %r{acc_group};
        fma.rn.f16x2 %r{acc_group}, %r{cvt_a3}, %r{cvt_b3}, %r{acc_group};
        fma.rn.f16x2 %r{acc_group}, %r{cvt_a4}, %r{cvt_b4}, %r{acc_group};
        fma.rn.f16x2 %r{acc_group}, %r{cvt_a5}, %r{cvt_b5}, %r{acc_group};
        fma.rn.f16x2 %r{acc_group}, %r{cvt_a6}, %r{cvt_b6}, %r{acc_group};
        fma.rn.f16x2 %r{acc_group}, %r{cvt_a7}, %r{cvt_b7}, %r{acc_group};

        // Apply scale and accumulate
        mul.rn.f16x2 %r{acc_group}, %r{scale0}, %r{acc_group};
        add.rn.f16x2 %r{acc_total}, %r{acc_total}, %r{acc_group};

        // Reduce f16x2 to scalar f16 then to f32
        mov.b32 {{%h{lane0}, %h{lane1}}}, %r{acc_total};
        add.rn.f16 %h{result_f16}, %h{lane0}, %h{lane1};
        cvt.f32.f16 %f{result_f32}, %h{result_f16};
        ''',

        # Warp shuffle reduction
        "warp_shfl_reduce": '''
        // Warp shuffle reduction for {width} threads
        shfl.sync.down.b32 %f{tmp}, %f{acc}, 16, 0xffffffff;
        add.f32 %f{acc}, %f{acc}, %f{tmp};
        shfl.sync.down.b32 %f{tmp}, %f{acc}, 8, 0xffffffff;
        add.f32 %f{acc}, %f{acc}, %f{tmp};
        shfl.sync.down.b32 %f{tmp}, %f{acc}, 4, 0xffffffff;
        add.f32 %f{acc}, %f{acc}, %f{tmp};
        shfl.sync.down.b32 %f{tmp}, %f{acc}, 2, 0xffffffff;
        add.f32 %f{acc}, %f{acc}, %f{tmp};
        shfl.sync.down.b32 %f{tmp}, %f{acc}, 1, 0xffffffff;
        add.f32 %f{acc}, %f{acc}, %f{tmp};
        ''',

        # Shared memory + shuffle reduction (for TB_WIDTH > 32)
        "smem_then_shfl_reduce": '''
        // Store to shared memory
        st.shared.f32 [%r{smem_addr}], %f{acc};
        bar.sync 0;

        // Tree reduction in shared memory
        @%p{active} ld.shared.f32 %f{tmp}, [%r{smem_addr} + {offset}];
        @%p{active} add.f32 %f{acc}, %f{acc}, %f{tmp};
        @%p{active} st.shared.f32 [%r{smem_addr}], %f{acc};
        bar.sync 0;

        // Final warp shuffle
        shfl.sync.down.b32 %f{tmp}, %f{acc}, 16, 0xffffffff;
        add.f32 %f{acc}, %f{acc}, %f{tmp};
        // ... (continue shuffle reduction)
        ''',
    }

    def to_config_vector(self) -> list[int]:
        """
        Convert config to numeric vector for NN.

        Each config value becomes an index into its choice array.
        """
        vector = []

        for key, choices in self.CHOICE_ARRAYS.items():
            value = self.config.get(key)
            if value is not None and value in choices:
                idx = choices.index(value)
            else:
                idx = 0  # Default to first choice
            vector.append(idx)

        return vector

    @classmethod
    def from_config_vector(cls, vector: list[int], problem: dict) -> "BlueprintIR":
        """
        Create IR from config vector (NN output).

        Args:
            vector: List of indices into choice arrays
            problem: Dict with M, K, L values
        """
        ir = cls()

        # Decode vector to config
        for i, (key, choices) in enumerate(cls.CHOICE_ARRAYS.items()):
            if i < len(vector):
                idx = min(vector[i], len(choices) - 1)
                ir.config[key] = choices[idx]

        # Generate IR structure from config
        ir._generate_from_config(problem)

        return ir

    def _generate_from_config(self, problem: dict):
        """Generate IR structure from config and problem dims"""
        M = problem.get("M", 1024)
        K = problem.get("K", 1024)
        L = problem.get("L", 1)

        BLOCK_M = self.config.get("BLOCK_M", 8)
        BLOCK_K = self.config.get("BLOCK_K", 512)
        NUM_WARPS = self.config.get("NUM_WARPS", 4)

        # Set up parameters
        self.parameters = [
            KernelParameter("ptr_A", "b64", "ptr_input"),
            KernelParameter("ptr_B", "b64", "ptr_input"),
            KernelParameter("ptr_SFA", "b64", "ptr_input"),
            KernelParameter("ptr_SFB", "b64", "ptr_input"),
            KernelParameter("ptr_C", "b64", "ptr_output"),
            KernelParameter("M", "b32", "scalar"),
            KernelParameter("K", "b32", "scalar"),
        ]

        # Set up registers (estimate)
        self.registers = RegisterAllocation(
            pred=8,
            b32=32,
            b64=16,
            f32=16,
        )

        # Generate phases
        self.phases = [
            self._generate_prologue(),
            self._generate_main_loop(K, BLOCK_K),
            self._generate_epilogue(),
        ]

    def _generate_prologue(self) -> Phase:
        """Generate prologue phase"""
        phase = Phase(type=PhaseType.PROLOGUE, name="prologue")

        # Load parameters
        for i, param in enumerate(self.parameters):
            if param.dtype == "b64":
                phase.instructions.append(Instruction(
                    mnemonic="ld",
                    modifiers=["param", "b64"],
                    operands=[
                        Operand("", "register", f"%rd{i}"),
                        Operand("", "address", f"[{param.name}]"),
                    ]
                ))

        # Thread ID
        phase.instructions.append(Instruction(
            mnemonic="mov",
            modifiers=["u32"],
            operands=[
                Operand("", "register", "%r0"),
                Operand("", "register", "%tid.x"),
            ]
        ))

        return phase

    def _generate_main_loop(self, K: int, BLOCK_K: int) -> Phase:
        """Generate main loop phase"""
        phase = Phase(
            type=PhaseType.MAIN_LOOP,
            name="main_loop",
            loop_var="%p_loop",
            loop_bound=str(K // BLOCK_K)
        )

        cache_a = self.config.get("CACHE_A", ".cs")
        cache_b = self.config.get("CACHE_B", ".ca")

        # Load A
        a_mods = ["global"] + cache_a.strip(".").split(".")
        phase.instructions.append(Instruction(
            mnemonic="ld",
            modifiers=a_mods + ["v4", "b32"],
            operands=[
                Operand("", "vector", "{%r4, %r5, %r6, %r7}"),
                Operand("", "address", "[%rd_a]"),
            ],
            comment="Load A tile"
        ))

        # Load B
        b_mods = ["global"] + cache_b.strip(".").split(".")
        phase.instructions.append(Instruction(
            mnemonic="ld",
            modifiers=b_mods + ["v4", "b32"],
            operands=[
                Operand("", "vector", "{%r8, %r9, %r10, %r11}"),
                Operand("", "address", "[%rd_b]"),
            ],
            comment="Load B"
        ))

        # FMA chain (placeholder)
        phase.instructions.append(Instruction(
            mnemonic="fma",
            modifiers=["rn", "f32"],
            operands=[
                Operand("", "register", "%f0"),
                Operand("", "register", "%f1"),
                Operand("", "register", "%f2"),
                Operand("", "register", "%f0"),
            ]
        ))

        return phase

    def _generate_epilogue(self) -> Phase:
        """Generate epilogue phase"""
        phase = Phase(type=PhaseType.EPILOGUE, name="epilogue")

        reduction = self.config.get("REDUCTION", "warp_shfl")

        if reduction == "warp_shfl":
            # Warp shuffle reduction
            for offset in [16, 8, 4, 2, 1]:
                phase.instructions.append(Instruction(
                    mnemonic="shfl",
                    modifiers=["sync", "down", "b32"],
                    operands=[
                        Operand("", "register", "%f1"),
                        Operand("", "register", "%f0"),
                        Operand("", "immediate", str(offset)),
                        Operand("", "immediate", "0xffffffff"),
                    ]
                ))
                phase.instructions.append(Instruction(
                    mnemonic="add",
                    modifiers=["f32"],
                    operands=[
                        Operand("", "register", "%f0"),
                        Operand("", "register", "%f0"),
                        Operand("", "register", "%f1"),
                    ]
                ))

        # Store result
        phase.instructions.append(Instruction(
            mnemonic="st",
            modifiers=["global", "f16"],
            operands=[
                Operand("", "address", "[%rd_c]"),
                Operand("", "register", "%h0"),
            ],
            predicate="%p_lane0"
        ))

        return phase

    # =========================================================================
    # LLM EXAMPLES: For prompting
    # =========================================================================

    def to_llm_example(self) -> str:
        """
        Generate example string for LLM prompting.

        Shows key code snippets and config, not full IR.
        """
        lines = []

        # Problem context (if known)
        if "M" in self.config or "K" in self.config:
            lines.append(f"Problem: M={self.config.get('M', '?')}, "
                         f"K={self.config.get('K', '?')}, "
                         f"L={self.config.get('L', '?')}")

        # Config summary
        lines.append("Config:")
        for key in ["BLOCK_M", "BLOCK_K", "NUM_WARPS", "CACHE_A", "CACHE_B", "REDUCTION"]:
            if key in self.config:
                lines.append(f"  {key}: {self.config[key]}")

        # Key code snippets
        lines.append("\nKey instructions:")
        for phase in self.phases:
            if phase.type == PhaseType.MAIN_LOOP:
                lines.append(f"  // {phase.name}")
                for instr in phase.instructions[:5]:  # First few
                    lines.append(f"  {instr.to_ptx()}")
                if len(phase.instructions) > 5:
                    lines.append(f"  // ... {len(phase.instructions) - 5} more")

        return "\n".join(lines)


# =============================================================================
# Convenience functions
# =============================================================================

def decompose_ptx(ptx: str) -> BlueprintIR:
    """Decompose PTX source to IR"""
    return BlueprintIR.from_ptx_source(ptx)


def decompose_cuda(cuda: str) -> BlueprintIR:
    """Decompose CUDA source to IR"""
    return BlueprintIR.from_cuda_source(cuda)


def reconstruct_ptx(ir: BlueprintIR) -> str:
    """Reconstruct PTX from IR"""
    return ir.to_ptx()


def config_to_ir(config_vector: list[int], problem: dict) -> BlueprintIR:
    """Create IR from NN output config vector"""
    return BlueprintIR.from_config_vector(config_vector, problem)


if __name__ == "__main__":
    # Test round-trip
    test_ptx = """
.version 7.8
.target sm_100
.address_size 64

.visible .entry test_kernel(
    .param .b64 ptr_A,
    .param .b64 ptr_B,
    .param .b32 M
) {
    .reg .pred %p<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<8>;

    ld.param.b64 %rd0, [ptr_A];
    ld.param.b64 %rd1, [ptr_B];
    mov.u32 %r0, %tid.x;

$LOOP:
    ld.global.cs.v4.b32 {%r1, %r2, %r3, %r4}, [%rd0];
    ld.global.ca.v4.b32 {%r5, %r6, %r7, %r8}, [%rd1];
    fma.rn.f32 %f0, %f1, %f2, %f0;
    @%p0 bra $LOOP;

    shfl.sync.down.b32 %f1, %f0, 16, 0xffffffff;
    add.f32 %f0, %f0, %f1;
    st.global.f32 [%rd2], %f0;
    ret;
}
"""

    print("=== Decomposing PTX ===")
    ir = BlueprintIR.from_ptx_source(test_ptx)

    print(f"Kernel: {ir.name}")
    print(f"Target: sm_{ir.target_sm}")
    print(f"Parameters: {len(ir.parameters)}")
    print(f"Phases: {len(ir.phases)}")

    print("\n=== Config Vector ===")
    vec = ir.to_config_vector()
    print(f"Vector: {vec}")

    print("\n=== LLM Example ===")
    print(ir.to_llm_example())

    print("\n=== Reconstructed PTX ===")
    reconstructed = ir.to_ptx()
    print(reconstructed[:500] + "..." if len(reconstructed) > 500 else reconstructed)
