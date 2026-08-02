#!/usr/bin/env bash
# Set up everything the LZ4 benchmarks need: the Silesia corpus and a venv with
# nvCOMP (the GPU LZ4 baseline). Both land in gitignored directories.
#
#   ./setup.sh                 # corpus + venv
#   ./setup.sh --corpus-only   # skip the venv (our own bench needs only the corpus)
#
# Then:
#   cargo run --release -p lz4-ptx --bin lz4-comp-warp        # ours
#   ./baseline/.venv/bin/python baseline/bench_nvcomp_compress.py corpus/silesia_all.bin
#
# Both benchmarks print their own methodology and environment when run.

set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$BASELINE_DIR/.." && pwd)"
CORPUS_DIR="$APP_DIR/corpus"
CORPUS="$CORPUS_DIR/silesia_all.bin"
VENV="$BASELINE_DIR/.venv"
SILESIA_URL="http://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"

# ── Corpus: the 12 Silesia files concatenated, 211.9 MB ───────────────────────
if [[ ! -f "$CORPUS" ]]; then
    mkdir -p "$CORPUS_DIR/silesia"
    echo "Downloading Silesia corpus ..."
    curl -fsSL -o "$CORPUS_DIR/silesia.zip" "$SILESIA_URL"
    (cd "$CORPUS_DIR/silesia" && unzip -oq ../silesia.zip)
    cat "$CORPUS_DIR"/silesia/* > "$CORPUS"
fi
bytes=$(stat -c%s "$CORPUS")
echo "corpus: $CORPUS ($bytes bytes)"
# The shipped kernels bake in a 209_715_200-byte prefix (Lz4CompAlgorithm.corpusBytes).
if (( bytes < 209715200 )); then
    echo "ERROR: corpus is smaller than the 209715200-byte prefix the kernels bake in" >&2
    exit 1
fi

[[ "${1:-}" == "--corpus-only" ]] && exit 0

# ── nvCOMP baseline venv ──────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "Creating baseline venv ..."
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -r "$BASELINE_DIR/requirements.txt"
"$VENV/bin/python" -c "import nvidia.nvcomp" && echo "nvcomp: OK"
