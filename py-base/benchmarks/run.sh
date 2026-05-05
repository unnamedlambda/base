#!/usr/bin/env bash
# Run the py_base benchmark suite.
#
# First run: creates .venv, builds py_base, installs deps (~1 min).
# Subsequent runs: fast. Lean modules are rebuilt incrementally by Lake,
# and benchmark JSONs are regenerated on each run.
#
# Usage: ./run.sh [--bench <name>] [--rounds <n>]
#   --bench   csv | json | regex | strsearch | vecops | torchops | pandas | vllm | all  (default: all)
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

pip install -q maturin

echo "Building py_base ..."
(cd "$SCRIPT_DIR/.." && maturin develop -q)

pip install -q -r "$SCRIPT_DIR/requirements-bench.txt"

# ── Algorithm JSON files ──────────────────────────────────────────────────────

mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

LEAN_FILES=(
    "CsvBenchAlgorithm.lean"
    "JsonBenchAlgorithm.lean"
    "RegexBenchAlgorithm.lean"
    "StringSearchAlgorithm.lean"
    "WordCountAlgorithm.lean"
    "VecOpsBenchAlgorithm.lean"
    "ClampSumBenchAlgorithm.lean"
    "RowDotBenchAlgorithm.lean"
    "RowAffineReduceBenchAlgorithm.lean"
    "PandasBenchAlgorithm.lean"
    "PandasFilterBenchAlgorithm.lean"
    "CudaVecAddPersistAlgorithm.lean"
    "CudaSaxpyPersistAlgorithm.lean"
    "CudaGemvPersistAlgorithm.lean"
    "CudaRmsNormPersistAlgorithm.lean"
    "CudaSoftmaxPersistAlgorithm.lean"
    "CudaDecoderLayerAlgorithm.lean"
    "CudaDecodeAttentionAlgorithm.lean"
)

for lean_file in "${LEAN_FILES[@]}"; do
    src="$LEAN_DIR/$lean_file"
    module="${lean_file%.lean}"
    module_out_dir="$OUT_DIR/$module"

    (cd "$LAKE_DIR" && lake build "$module")
    echo "Generating artifacts from $lean_file ..."
    mkdir -p "$module_out_dir"
    (cd "$LAKE_DIR" && lake env lean --run "$src" "$module_out_dir")
done

# ── Run benchmarks ────────────────────────────────────────────────────────────

python "$SCRIPT_DIR/bench.py" "$@"
