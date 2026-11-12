#!/usr/bin/env bash
# The compression benchmark, from scratch, measured as lzbench measures:
# inputs, a check on the first 1 MiB, every regime on Silesia, the GPU codecs,
# then the report.
#
#   ./run.sh            everything (about 15 min)
#
# Nothing measured is reused: each invocation is one run, written to
# results/<time>-<git rev>/ with every process's wall time (times.tsv), and
# REPORT.md is rewritten from it. The corpus and the lzbench build are inputs
# and stay in data/compress/ and build/.
#
# Each step is its own process under a 9 GB memory cap with swap off (safety,
# not a benchmark condition; see ../ann/run.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=.venv/bin/python

cap() { systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 "$@"; }
cap true || { echo "systemd-run --user is not available: not running uncapped" >&2; exit 1; }

if [[ ! -x $PY ]]; then
    python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt
fi
cap python3 prepare.py
CORPUS=../../data/compress/silesia.tar
TINY=../../data/compress/tiny

export COMPRESS_RUN="$(date +%Y-%m-%d-%H%M%S)-$(git rev-parse --short HEAD)"
OUT=results/$COMPRESS_RUN
mkdir -p "$OUT"

STRICT=1
step() {
    local t0 t1 rc=0 status=ok
    t0=$(date +%s.%N)
    cap env COMPRESS_RUN="$COMPRESS_RUN" "$@" || rc=$?
    t1=$(date +%s.%N)
    [[ $rc -eq 137 ]] && status=no-fit
    [[ $status == ok && $rc -ne 0 ]] && status=failed
    printf '%s\t%s\t%.1f\n' "$*" "$status" "$(echo "$t1 - $t0" | bc)" >> "$OUT/times.tsv"
    if [[ $status != ok && $STRICT == 1 ]]; then
        echo "FAILED on the check input: $*" >&2
        exit 1
    fi
    [[ $status == failed ]] && echo "FAILED: $*" >&2
    return 0
}

# The checks: every codec level's round trip, and every GPU codec's.
step python3 cpu.py check "$TINY"
step $PY gpu.py "$TINY"
STRICT=0
for r in file 64k-all 4k; do step python3 cpu.py "$r" "$CORPUS"; done
step $PY gpu.py "$CORPUS"

python3 report.py "$COMPRESS_RUN" > "$OUT/REPORT.md"
cp "$OUT/REPORT.md" REPORT.md
echo "run $COMPRESS_RUN done; $OUT/REPORT.md and REPORT.md written" >&2
