#!/usr/bin/env bash
# Run the benchmark suite: device work against PyTorch through py-base.
#
# First run: creates .venv, builds py_base, installs deps (~1 min).
# Subsequent runs: fast. Lake builds are incremental.
#
# Usage: ./run.sh [--bench <name>] [--rounds <n>]
#   --bench   torchops | vllm | all                          (default: all)
#   --rounds  timed iterations per size                      (default: 10)
#
# Args are forwarded to the Python runner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_BENCH_DIR="$SCRIPT_DIR/python"
PY_BASE_DIR="$REPO_ROOT/py-base"
VENV="$PY_BASE_DIR/.venv"
LAKE_DIR="$REPO_ROOT/lean/algorithms"
OUT_DIR="$REPO_ROOT/lean/data"
BENCHMARKS_MODULE="PythonBenchmarks"

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

# ── PythonBenchmarks artifacts ────────────────────────────────────────────────

mkdir -p "$OUT_DIR"
src="$LAKE_DIR/$BENCHMARKS_MODULE.lean"
module_out_dir="$OUT_DIR/$BENCHMARKS_MODULE"

(cd "$LAKE_DIR" && lake build "$BENCHMARKS_MODULE")
echo "Generating artifacts from $BENCHMARKS_MODULE.lean ..."
mkdir -p "$module_out_dir"
(cd "$LAKE_DIR" && lake env lean --run "$src" "$module_out_dir")

# ── Run Python suite ──────────────────────────────────────────────────────────

python "$PY_BENCH_DIR/bench.py" "$@"
