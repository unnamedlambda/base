#!/usr/bin/env bash
# Generate algorithm JSON files from Lean source.
# Run this once before benchmarking: ./generate_algos.sh
# Requires: lake (Lean package manager) in PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run lake from the benchmarks/ directory where lakefile.lean lives so elan
# picks up the already-configured toolchain instead of downloading a new one.
LAKE_DIR="$SCRIPT_DIR/../../benchmarks"
LEAN_DIR="$LAKE_DIR/lean"
OUT_DIR="$SCRIPT_DIR/algos"

mkdir -p "$OUT_DIR"

declare -A ALGOS=(
    ["CsvBenchAlgorithm.lean"]="csv"
    ["JsonBenchAlgorithm.lean"]="json"
    ["RegexBenchAlgorithm.lean"]="regex"
    ["StringSearchAlgorithm.lean"]="strsearch"
    ["WordCountAlgorithm.lean"]="wordcount"
)

for lean_file in "${!ALGOS[@]}"; do
    name="${ALGOS[$lean_file]}"
    src="$LEAN_DIR/$lean_file"
    out="$OUT_DIR/$name.json"
    echo "Generating $name.json ..."
    # Extract JSON: Lean may emit warning lines before the '[' array start.
    raw=$(cd "$LAKE_DIR" && lake env lean --run "$src")
    # Strip everything before the first '[' that starts a line.
    json=$(echo "$raw" | awk '/^\[/{found=1} found{print}')
    echo "$json" > "$out"
done

echo "Done. Algorithm JSONs written to $OUT_DIR/"
