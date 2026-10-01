#!/usr/bin/env bash
# Run the benchmarks: device work against PyTorch through py-base, then the
# CPU suite (`cpu/run.sh`) and the system suite (`system/run.sh`).
#
# First run: creates .venv, builds py_base, installs deps (~1 min).
# Subsequent runs: fast. Lake builds are incremental.
#
# This script asks Lake for the artifacts, then sets up the venv and py_base.
#
# Usage: ./run.sh [--bench <name>] [--rounds <n>]
#   --bench   torchops | vllm | all                          (default: all)
#   --rounds  timed iterations per size                      (default: 10)
#
# Args are forwarded to the Python runner; the CPU and system suites always run
# all.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_BENCH_DIR="$SCRIPT_DIR/python"
PY_BASE_DIR="$REPO_ROOT/py-base"
VENV="$PY_BASE_DIR/.venv"

# ── Python environment ────────────────────────────────────────────────────────

if [[ ! -d "$VENV" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

pip install -q maturin

echo "Building py_base ..."
(cd "$PY_BASE_DIR" && maturin develop --release -q)

pip install -q -r "$PY_BENCH_DIR/requirements-bench.txt"

# ── Artifacts ─────────────────────────────────────────────────────────────────

echo "Building artifacts ..."
(cd "$REPO_ROOT/lean" && lake build algorithmLib/artifacts)

# ── Run Python suite ──────────────────────────────────────────────────────────

python "$PY_BENCH_DIR/bench.py" "$@"

# ── Run CPU and system suites ─────────────────────────────────────────────────

"$SCRIPT_DIR/cpu/run.sh"
"$SCRIPT_DIR/system/run.sh"
