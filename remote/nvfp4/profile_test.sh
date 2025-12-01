#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/out_local"
mkdir -p "$OUT_DIR"

usage() {
    cat <<'EOF'
Usage: profile_test.sh [TEST_CASES_FILE]

Runs nvfp4/eval.py in profile mode under Nsight Compute.
If TEST_CASES_FILE is omitted, a default single test (m=1024,k=65536,l=1,seed=0)
is generated automatically.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 1 ]]; then
    echo "Error: accepts at most one test-case file" >&2
    usage
    exit 1
fi

cleanup_tmp_case() {
    [[ -n "${TMP_TEST_CASE:-}" && -f "$TMP_TEST_CASE" ]] && rm -f "$TMP_TEST_CASE"
}
trap cleanup_tmp_case EXIT

if [[ $# -eq 1 ]]; then
    TEST_CASE_FILE="$1"
else
    TMP_TEST_CASE="$(mktemp)"
    TEST_CASE_FILE="$TMP_TEST_CASE"
    cat > "$TEST_CASE_FILE" <<'EOF'
m:1024;k:65536;l:1;seed:0
EOF
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
REPORT_PREFIX="$OUT_DIR/nvfp4_profile_${timestamp}"

export POPCORN_FD=1
export POPCORN_NCU=1

cd "$SCRIPT_DIR"

echo "Running Nsight Compute profile..."
ncu \
    --set full \
    --target-processes all \
    --kernel-name-base demangled \
    --launch-skip 0 --launch-count 1 \
    --export "$REPORT_PREFIX" \
    --log-file "${REPORT_PREFIX}.log" \
    python eval.py profile "$TEST_CASE_FILE"

echo "\nNsight Compute report saved to ${REPORT_PREFIX}.ncu-rep"
echo "Raw log saved to ${REPORT_PREFIX}.log"
