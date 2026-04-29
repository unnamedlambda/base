#!/usr/bin/env bash
# Run the py_base benchmark suite.
#
# First run: creates .venv, builds py_base, installs deps (~1 min).
# Subsequent runs: fast. Lean JSONs are regenerated only if source changed.
#
# Usage: ./run.sh [--bench <name>] [--rounds <n>]
#   --bench   csv | json | regex | strsearch | vecops | vllm | all  (default: all)
#   --rounds  timed iterations per size                      (default: 10)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../.venv"
LAKE_DIR="$SCRIPT_DIR/../../lean/algorithms"
LEAN_DIR="$LAKE_DIR"
OUT_DIR="$SCRIPT_DIR/../../lean/data"

# ── Python environment ────────────────────────────────────────────────────────

if [[ ! -d "$VENV" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

if ! python -c "import py_base" 2>/dev/null; then
    echo "Installing py_base (first time — builds Rust, may take a minute)..."
    pip install -q maturin
    (cd "$SCRIPT_DIR/.." && maturin develop -q)
fi

pip install -q -r "$SCRIPT_DIR/requirements-bench.txt"

# ── Algorithm JSON files ──────────────────────────────────────────────────────

mkdir -p "$OUT_DIR"

declare -A ALGOS=(
    ["CsvBenchAlgorithm.lean"]="csv"
    ["JsonBenchAlgorithm.lean"]="json"
    ["RegexBenchAlgorithm.lean"]="regex"
    ["StringSearchAlgorithm.lean"]="strsearch"
    ["WordCountAlgorithm.lean"]="wordcount"
    ["VecOpsBenchAlgorithm.lean"]="vecops"
    ["ClampSumBenchAlgorithm.lean"]="clampsum"
    ["PandasBenchAlgorithm.lean"]="pandas"
    ["PandasFilterBenchAlgorithm.lean"]="pandas_filter"
    ["CudaGemvPersistAlgorithm.lean"]="cuda_gemv_persist"
    ["CudaRmsNormPersistAlgorithm.lean"]="cuda_rmsnorm_persist"
    ["CudaSoftmaxPersistAlgorithm.lean"]="cuda_softmax_persist"
)

for lean_file in "${!ALGOS[@]}"; do
    name="${ALGOS[$lean_file]}"
    src="$LEAN_DIR/$lean_file"
    out="$OUT_DIR/$name.json"

    if [[ ! -f "$out" ]] || [[ "$src" -nt "$out" ]]; then
        echo "Generating $name.json ..."
        raw=$(cd "$LAKE_DIR" && lake env lean --run "$src")
        echo "$raw" | awk '/^\[/{found=1} found{print}' > "$out"
    fi
done

# ── Run benchmarks ────────────────────────────────────────────────────────────

python "$SCRIPT_DIR/bench.py" "$@"
