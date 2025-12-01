# PTX Kernel Generator Plan

## 1. Canonical Syntax Layer
- Maintain `ptx_sm75_syntax_table.yaml` as the source of truth for mnemonics, modifiers, operands, directives, and SM/PTX compatibility.
- Build a deterministic parser/validator that loads the table, tokenizes sequences, and rejects illegal combinations before calling `ptxas`.
- Extend tables per architecture (e.g., sm_80) without touching higher layers.

## 2. Blueprint IR Layer
- Use `blueprint_ir.yaml` to describe problem-agnostic primitives: tensor roles, dimension symbols, parameter-pack schema, launch formulas, shared-memory tiles, phase graph, optional epilogues, instrumentation hooks, and search-space parameters.
- Ensure every instruction identifier referenced in the blueprint exists in the syntax spec.
- Allow workloads to bind concrete values (tensor shapes, launch choices) at runtime without editing the schema.

## 3. Generator & Checker Pipeline
- Generator consumes blueprint + workload binding → produces instruction graph honoring phase constraints and syntax IDs.
- Fast syntax checker validates tokens using the canonical table; failures return structured errors for model training.
- Successful candidates run through `ptxas` (or nvcc) for definitive assembly, followed by unit tests and Nsight profiling.

## 4. Model Training Loop
- Model proposes blueprint decisions (tiling, phase instructions, epilogue variant) rather than raw PTX text.
- Feedback signals:
  1. Syntax checker (instant rejection/penalty)
  2. `ptxas` result (compile success/failure)
  3. Unit-test correctness (compare vs. reference)
  4. Profiling metrics (reward for performance)
- Keep logs per attempt to refine instructions, update syntax table when discrepancies appear.

## 5. Extensibility Roadmap
- Add new architecture tables and blueprint overrides (shared-memory limits, available ops).
- Introduce optional dialect-like layer if richer semantics are needed; otherwise keep YAMLs lightweight.
- Integrate host-launch codegen templates referencing the parameter-pack schema.
