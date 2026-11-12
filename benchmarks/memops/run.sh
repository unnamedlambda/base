#!/usr/bin/env bash
# The memory-operations benchmark, from scratch, measured as llvm-libc
# measures: its benchmark built at a pinned release with this suite's list of
# implementations, every function over Google's size distributions, then the
# report.
#
#   ./run.sh            (the build: about 10 minutes the first time; the run: a few)
#
# Nothing measured is reused: each invocation is one run, written to
# results/<time>-<git rev>/ with every process's wall time (times.tsv), and
# REPORT.md is rewritten from it. The llvm-project checkout and build are
# inputs and stay in build/.
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
cap $PY prepare.py
# base's engine is an input, as the harness is: its library (libbase, the C
# API) and its memops artifact, written by the Lean generator.
ROOT=$(cd ../.. && pwd)
( cd "$ROOT" && cap cargo build --release -q -p base ) || { echo "libbase did not build" >&2; exit 1; }
( cd "$ROOT" && tools/lean_build.sh genmemopsbench > /dev/null ) || { echo "base's memops generator did not build" >&2; exit 1; }
mkdir -p build/base
cap "$ROOT/lean/.lake/build/bin/genmemopsbench" build/base > /dev/null
export BASE_LIB="$ROOT/target/release/libbase.so" BASE_MEMOPS="$PWD/build/base/memops.cbor"

export MEMOPS_RUN="$(date +%Y-%m-%d-%H%M%S)-$(git rev-parse --short HEAD)"
OUT=results/$MEMOPS_RUN
mkdir -p "$OUT"

step() {
    local t0 t1 rc=0 status=ok
    t0=$(date +%s.%N)
    cap env MEMOPS_RUN="$MEMOPS_RUN" BASE_LIB="$BASE_LIB" BASE_MEMOPS="$BASE_MEMOPS" "$@" || rc=$?
    t1=$(date +%s.%N)
    [[ $rc -eq 137 ]] && status=no-fit
    [[ $status == ok && $rc -ne 0 ]] && status=failed
    printf '%s\t%s\t%.1f\n' "$*" "$status" "$(echo "$t1 - $t0" | bc)" >> "$OUT/times.tsv"
    if [[ $status != ok ]]; then echo "FAILED: $*" >&2; exit 1; fi
}

# The check: the implementations' checks run at start-up, then one quick pass.
step $PY bench.py '/0/'
step $PY bench.py

$PY report.py "$MEMOPS_RUN" > "$OUT/REPORT.md"
cp "$OUT/REPORT.md" REPORT.md
echo "run $MEMOPS_RUN done; $OUT/REPORT.md and REPORT.md written" >&2
