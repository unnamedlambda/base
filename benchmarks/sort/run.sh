#!/usr/bin/env bash
# The sort and top-k benchmark, from scratch: CUB's radix sort built with the
# CUDA toolkit's nvcc, base's sort artifact written by its Lean generator,
# both run through bench.py, then the report.
#
#   ./run.sh              every engine
#   ./run.sh base         base only (the development loop)
#
# Each invocation is a run in results/<time>-<git rev>/ with every process's
# wall time (times.tsv); REPORT.md is rewritten from the newest run that ran
# every engine. Each step runs under a 9 GB memory cap with swap off (safety,
# not a benchmark condition; see ../ann/run.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$(cd ../.. && pwd)
PY="$ROOT/py-base/.venv/bin/python"   # numpy, torch and py_base (tools/check.sh builds it)
[[ -x $PY ]] || { echo "py-base is not built (tools/check.sh builds it)" >&2; exit 1; }

cap() { systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 "$@"; }
cap true || { echo "systemd-run --user is not available: not running uncapped" >&2; exit 1; }

mkdir -p build/base
cap nvcc -O3 -std=c++17 -arch=native -o build/cub_sort cub_sort.cu
( cd "$ROOT" && tools/lean_build.sh gensortbench > /dev/null ) || { echo "base's sort generator did not build" >&2; exit 1; }
cap "$ROOT/lean/.lake/build/bin/gensortbench" build/base > /dev/null

export SORT_RUN="$(date +%Y-%m-%d-%H%M%S)-$(git rev-parse --short HEAD)"
OUT=results/$SORT_RUN
mkdir -p "$OUT"

step() {
    local t0 t1 rc=0 status=ok
    t0=$(date +%s.%N)
    cap env SORT_RUN="$SORT_RUN" "$PY" "$@" || rc=$?
    t1=$(date +%s.%N)
    [[ $rc -eq 137 ]] && status=no-fit
    [[ $status == ok && $rc -ne 0 ]] && status=failed
    printf '%s\t%s\t%.1f\n' "$*" "$status" "$(echo "$t1 - $t0" | bc)" >> "$OUT/times.tsv"
    [[ $status == ok ]] || { echo "FAILED: $*" >&2; exit 1; }
}

if [[ ${1:-} == base ]]; then
    SORT_ENGINES=base; TOPK_ENGINES=base
else
    SORT_ENGINES="base cub"; TOPK_ENGINES="base torch"; touch "$OUT/full"
fi
for e in $SORT_ENGINES; do step bench.py sort "$e"; done
for e in $TOPK_ENGINES; do step bench.py topk "$e"; done

"$PY" report.py "$SORT_RUN" > "$OUT/REPORT.md"
[[ -f $OUT/full ]] && cp "$OUT/REPORT.md" REPORT.md
echo "run $SORT_RUN done: $OUT/REPORT.md" >&2
