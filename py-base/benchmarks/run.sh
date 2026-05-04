#!/usr/bin/env bash
# Run the py_base benchmark suite.
#
# First run: creates .venv, builds py_base, installs deps (~1 min).
# Subsequent runs: fast. Lean JSONs are regenerated only if source changed.
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

# Each Lean file outputs [[name, config, alg], ...]; write one {name}.json per entry.
_extract_artifacts() {
    local out_dir="$1"
    python3 -c '
import sys, json, os
out_dir = sys.argv[1]
text = sys.stdin.read()
start = text.find("\n[")
data = json.loads(text[start + 1:] if start >= 0 else text.strip())
for name, config, alg in data:
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump([config, alg], f)
    print(f"  wrote {name}.json")
' "$out_dir"
}

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
    stamp="$OUT_DIR/.${lean_file%.lean}.stamp"

    if [[ ! -f "$stamp" ]] || [[ "$src" -nt "$stamp" ]]; then
        echo "Generating artifacts from $lean_file ..."
        (cd "$LAKE_DIR" && lake env lean --run "$src") | _extract_artifacts "$OUT_DIR"
        touch "$stamp"
    fi
done

# ── Run benchmarks ────────────────────────────────────────────────────────────

python "$SCRIPT_DIR/bench.py" "$@"
