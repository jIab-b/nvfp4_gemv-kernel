"""
PTX Syntax Checker - Layer 1

Validates PTX strings against the syntax table.
Fast (<1ms), no execution, pure string validation.

Usage:
    checker = SyntaxChecker("ptx_sm100_syntax_table.yaml")
    result = checker.check(ptx_string)
    if not result.valid:
        print(result.errors)
"""

import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time


@dataclass
class SyntaxResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def __bool__(self):
        return self.valid


@dataclass
class InstructionPattern:
    """Compiled pattern for a single instruction variant"""
    mnemonic: str
    regex: re.Pattern
    min_sm: int
    min_ptx: str
    operand_types: list[dict]


class SyntaxChecker:
    """
    Validates PTX assembly against syntax table.

    Architecture:
    1. Load YAML syntax table at init
    2. Pre-compile regex patterns for all instruction variants
    3. check() tokenizes input and matches against patterns
    """

    def __init__(self, syntax_table_path: str | Path):
        self.syntax_table_path = Path(syntax_table_path)
        self.table: dict = {}
        self.patterns: dict[str, list[InstructionPattern]] = {}
        self.directives: dict[str, re.Pattern] = {}
        self.register_patterns: dict[str, re.Pattern] = {}
        self.target_sm: int = 100

        self._load_table()
        self._compile_patterns()

    def _load_table(self):
        """Load and parse YAML syntax table"""
        with open(self.syntax_table_path) as f:
            self.table = yaml.safe_load(f)

        # Extract target SM from metadata
        metadata = self.table.get("metadata", {})
        target_gpu = metadata.get("target_gpu", {})
        self.target_sm = target_gpu.get("sm", 100)

    def _compile_patterns(self):
        """Pre-compile regex patterns for fast matching"""

        # Compile register patterns
        for reg_file in self.table.get("register_files", []):
            name = reg_file["name"]
            # Match patterns like %r0, %rd15, %f32
            self.register_patterns[name] = re.compile(
                rf"%{reg_file.get('prefix', name[0])}\d+"
            )

        # Compile directive patterns
        for directive in self.table.get("module_directives", []):
            name = directive["name"]
            # Simple pattern - can be refined per directive
            self.directives[name] = re.compile(
                rf"^\s*{re.escape(name)}\s+.*$"
            )

        # Compile instruction patterns
        for group in self.table.get("instruction_groups", []):
            for instr in group.get("instructions", []):
                mnemonic = instr["mnemonic"]
                if mnemonic not in self.patterns:
                    self.patterns[mnemonic] = []

                for template in instr.get("templates", []):
                    pattern = self._compile_instruction_pattern(
                        mnemonic, template, instr
                    )
                    if pattern:
                        self.patterns[mnemonic].append(pattern)

    def _compile_instruction_pattern(
        self,
        mnemonic: str,
        template: dict,
        instr: dict
    ) -> Optional[InstructionPattern]:
        """
        Compile a single instruction template into a regex pattern.

        Handles permutation axes to generate all valid modifier combinations.
        """
        syntax = template.get("syntax", "")
        if not syntax:
            return None

        # Get permutation axes
        axes = template.get("permutation_axes", [])

        # Build regex from syntax template
        # This is simplified - real implementation needs to handle all cases
        regex_str = self._syntax_to_regex(syntax, axes)

        try:
            regex = re.compile(regex_str, re.IGNORECASE)
        except re.error:
            return None

        return InstructionPattern(
            mnemonic=mnemonic,
            regex=regex,
            min_sm=instr.get("min_sm", 10),
            min_ptx=instr.get("min_ptx", "1.0"),
            operand_types=template.get("operands", [])
        )

    def _syntax_to_regex(self, syntax: str, axes: list[dict]) -> str:
        """
        Convert syntax template to regex pattern.

        Example:
            syntax: "ld.global{.cache}{.scope}.{type} ${dst}, [${addr}];"
            axes: [{axis: cache, values: ["", ".ca", ".cg"]}, ...]

        Returns regex that matches all valid combinations.
        """
        # Start with escaped syntax
        pattern = re.escape(syntax)

        # Replace ${name} placeholders with operand patterns
        pattern = re.sub(
            r"\\\$\\\{(\w+)\\\}",
            r"(?P<\1>[%\\w\\d\\[\\]\\+\\-]+)",
            pattern
        )

        # Handle optional modifiers {.modifier}
        pattern = re.sub(
            r"\\\{([^}]+)\\\}",
            r"(?:\1)?",
            pattern
        )

        # Handle axis value alternations
        for axis in axes:
            axis_name = axis.get("axis", "")
            values = axis.get("values", [])
            if values:
                # Create alternation group
                escaped_values = [re.escape(v) for v in values]
                alt_pattern = f"(?:{'|'.join(escaped_values)})"
                # Replace axis placeholder if present
                pattern = pattern.replace(f"${{{axis_name}}}", alt_pattern)

        # Allow flexible whitespace
        pattern = pattern.replace(r"\ ", r"\s+")
        pattern = pattern.replace(",", r"\s*,\s*")

        return f"^\\s*(?:@[%!]?\\w+\\s+)?{pattern}\\s*$"

    def check(self, ptx: str) -> SyntaxResult:
        """
        Validate PTX source against syntax table.

        Returns SyntaxResult with valid flag, errors, and timing.
        """
        start = time.perf_counter()
        errors = []
        warnings = []

        lines = ptx.strip().split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("//"):
                continue

            # Check if it's a directive
            if line.startswith("."):
                result = self._check_directive(line, line_num)
                if result:
                    errors.append(result)
                continue

            # Check if it's a label
            if line.endswith(":") or line.startswith("$"):
                continue

            # Check instruction
            result = self._check_instruction(line, line_num)
            if result:
                errors.append(result)

        elapsed = (time.perf_counter() - start) * 1000

        return SyntaxResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            latency_ms=elapsed
        )

    def _check_directive(self, line: str, line_num: int) -> Optional[str]:
        """Check if directive is valid"""
        # Extract directive name
        match = re.match(r"^\s*(\.[\w]+)", line)
        if not match:
            return f"Line {line_num}: Invalid directive format: {line}"

        directive_name = match.group(1)

        # Check if directive is known
        if directive_name not in self.directives:
            # Check data directives too
            data_directives = {d["name"] for d in self.table.get("data_directives", [])}
            if directive_name not in data_directives:
                return f"Line {line_num}: Unknown directive: {directive_name}"

        return None

    def _check_instruction(self, line: str, line_num: int) -> Optional[str]:
        """Check if instruction is valid"""
        # Remove predicate guard if present
        line_no_pred = re.sub(r"^@[%!]?\w+\s+", "", line)

        # Extract mnemonic (first word, possibly with modifiers)
        match = re.match(r"^([\w\.]+)", line_no_pred)
        if not match:
            return f"Line {line_num}: Cannot parse instruction: {line}"

        full_mnemonic = match.group(1)

        # Get base mnemonic (before first dot or full thing)
        parts = full_mnemonic.split(".")
        base_mnemonic = parts[0]

        # Some mnemonics include first modifier (e.g., "ld.global")
        if base_mnemonic in ["ld", "st", "atom", "red"]:
            if len(parts) > 1:
                base_mnemonic = f"{parts[0]}.{parts[1]}"

        # Check if mnemonic is known
        if base_mnemonic not in self.patterns:
            # Try without compound
            if parts[0] not in self.patterns:
                return f"Line {line_num}: Unknown instruction: {base_mnemonic}"
            base_mnemonic = parts[0]

        # Check against patterns
        patterns = self.patterns.get(base_mnemonic, [])

        for pattern in patterns:
            # Check SM compatibility
            if pattern.min_sm > self.target_sm:
                continue

            if pattern.regex.match(line):
                return None  # Valid!

        # No pattern matched
        return f"Line {line_num}: Invalid syntax for {base_mnemonic}: {line}"

    def check_sm_compatibility(self, ptx: str) -> list[str]:
        """Check which instructions require SM > target"""
        incompatible = []

        for line in ptx.split("\n"):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("."):
                continue

            # Extract mnemonic and check min_sm
            match = re.match(r"^(?:@[%!]?\w+\s+)?([\w\.]+)", line)
            if match:
                mnemonic = match.group(1).split(".")[0]
                for pattern in self.patterns.get(mnemonic, []):
                    if pattern.min_sm > self.target_sm:
                        incompatible.append(
                            f"{mnemonic} requires sm_{pattern.min_sm}, "
                            f"target is sm_{self.target_sm}"
                        )
                        break

        return incompatible


class PTXTokenizer:
    """
    Tokenizes PTX source into structured tokens.
    Used for detailed analysis beyond simple validation.
    """

    @dataclass
    class Token:
        type: str  # directive, label, instruction, comment, register, immediate
        value: str
        line: int
        col: int
        modifiers: list[str] = field(default_factory=list)
        operands: list[str] = field(default_factory=list)

    def __init__(self):
        self.tokens: list[PTXTokenizer.Token] = []

    def tokenize(self, ptx: str) -> list[Token]:
        """Tokenize PTX source into token list"""
        self.tokens = []

        for line_num, line in enumerate(ptx.split("\n"), 1):
            self._tokenize_line(line, line_num)

        return self.tokens

    def _tokenize_line(self, line: str, line_num: int):
        """Tokenize a single line"""
        original = line
        col = 0

        # Strip leading whitespace
        stripped = line.lstrip()
        col = len(line) - len(stripped)
        line = stripped

        if not line:
            return

        # Comment
        if line.startswith("//"):
            self.tokens.append(self.Token(
                type="comment",
                value=line,
                line=line_num,
                col=col
            ))
            return

        # Directive
        if line.startswith("."):
            match = re.match(r"(\.[\w]+)(.*)", line)
            if match:
                self.tokens.append(self.Token(
                    type="directive",
                    value=match.group(1),
                    line=line_num,
                    col=col,
                    operands=[match.group(2).strip()] if match.group(2).strip() else []
                ))
            return

        # Label
        if ":" in line and not line.startswith("@"):
            label = line.split(":")[0]
            self.tokens.append(self.Token(
                type="label",
                value=label,
                line=line_num,
                col=col
            ))
            return

        # Instruction
        self._tokenize_instruction(line, line_num, col)

    def _tokenize_instruction(self, line: str, line_num: int, col: int):
        """Tokenize an instruction line"""
        # Handle predicate
        predicate = None
        if line.startswith("@"):
            match = re.match(r"(@[%!]?\w+)\s+", line)
            if match:
                predicate = match.group(1)
                line = line[match.end():]
                col += match.end()

        # Split into mnemonic and operands
        parts = line.split(None, 1)
        if not parts:
            return

        mnemonic_full = parts[0]
        operands_str = parts[1] if len(parts) > 1 else ""

        # Parse mnemonic and modifiers
        mnemonic_parts = mnemonic_full.split(".")
        base_mnemonic = mnemonic_parts[0]
        modifiers = mnemonic_parts[1:] if len(mnemonic_parts) > 1 else []

        # Parse operands
        operands = self._parse_operands(operands_str)

        token = self.Token(
            type="instruction",
            value=base_mnemonic,
            line=line_num,
            col=col,
            modifiers=modifiers,
            operands=operands
        )

        if predicate:
            token.modifiers.insert(0, f"pred:{predicate}")

        self.tokens.append(token)

    def _parse_operands(self, operands_str: str) -> list[str]:
        """Parse operand string into list of operands"""
        # Remove trailing semicolon
        operands_str = operands_str.rstrip(";").strip()

        if not operands_str:
            return []

        # Handle braces for vector operands
        # e.g., {%r0, %r1, %r2, %r3}
        operands = []
        current = ""
        brace_depth = 0

        for char in operands_str:
            if char == "{":
                brace_depth += 1
                current += char
            elif char == "}":
                brace_depth -= 1
                current += char
            elif char == "," and brace_depth == 0:
                if current.strip():
                    operands.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            operands.append(current.strip())

        return operands


# Convenience functions
def quick_check(ptx: str, syntax_table: str = "ptx_sm100_syntax_table.yaml") -> bool:
    """Quick validation - returns True if valid"""
    checker = SyntaxChecker(syntax_table)
    return checker.check(ptx).valid


def validate_and_report(ptx: str, syntax_table: str = "ptx_sm100_syntax_table.yaml") -> str:
    """Validate and return human-readable report"""
    checker = SyntaxChecker(syntax_table)
    result = checker.check(ptx)

    if result.valid:
        return f"✓ Valid PTX ({result.latency_ms:.2f}ms)"
    else:
        report = f"✗ Invalid PTX ({result.latency_ms:.2f}ms)\n"
        report += f"  {len(result.errors)} error(s):\n"
        for error in result.errors[:10]:  # Show first 10
            report += f"    - {error}\n"
        if len(result.errors) > 10:
            report += f"    ... and {len(result.errors) - 10} more\n"
        return report


if __name__ == "__main__":
    # Test with simple PTX
    test_ptx = """
    .version 7.8
    .target sm_100
    .address_size 64

    .visible .entry test_kernel(
        .param .b64 ptr_A
    ) {
        .reg .b64 %rd<4>;
        .reg .b32 %r<4>;

        ld.param.b64 %rd0, [ptr_A];
        mov.u32 %r0, %tid.x;
        mad.wide.u32 %rd1, %r0, 16, %rd0;
        ld.global.b32 %r1, [%rd1];
        st.global.b32 [%rd1], %r1;
        ret;
    }
    """

    print(validate_and_report(test_ptx))
