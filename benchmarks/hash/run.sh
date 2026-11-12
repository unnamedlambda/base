#!/usr/bin/env bash
# The hashing benchmark, from scratch, measured as SMHasher3 measures:
# SMHasher3 built at its pinned commit (with base's hash added to its list),
# every hash's Sanity and Speed suites, then the report.
#
#   ./run.sh            every hash in speed.HASHES (about 11 minutes)
#
# Nothing measured is reused: each invocation is one run, written to
# results/<time>-<git rev>/ with every process's wall time (times.tsv), and
# REPORT.md is rewritten from it. The SMHasher3 build is an input and stays
# in build/.
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
# base's hash is an input, as SMHasher3 is: its library (the C API) and its
# artifact, written by the Lean generator
ROOT=$(cd ../.. && pwd)
( cd "$ROOT" && cap cargo build --release -q -p base ) || { echo "libbase did not build" >&2; exit 1; }
( cd "$ROOT" && tools/lean_build.sh genhashbench > /dev/null ) || { echo "base's hash generator did not build" >&2; exit 1; }
mkdir -p build/base
cap "$ROOT/lean/.lake/build/bin/genhashbench" build/base > /dev/null
export BASE_LIB="$ROOT/target/release/libbase.so" BASE_HASH="$PWD/build/base/hash.cbor"

export HASH_RUN="$(date +%Y-%m-%d-%H%M%S)-$(git rev-parse --short HEAD)"
OUT=results/$HASH_RUN
mkdir -p "$OUT"

step() {
    local t0 t1 rc=0 status=ok
    t0=$(date +%s.%N)
    cap env HASH_RUN="$HASH_RUN" "$@" || rc=$?
    t1=$(date +%s.%N)
    [[ $rc -eq 137 ]] && status=no-fit
    [[ $status == ok && $rc -ne 0 ]] && status=failed
    printf '%s\t%s\t%.1f\n' "$*" "$status" "$(echo "$t1 - $t0" | bc)" >> "$OUT/times.tsv"
    [[ $status == failed ]] && echo "FAILED: $*" >&2
    return 0
}

for h in $($PY -c 'import speed; print(*speed.HASHES)'); do step $PY speed.py "$h"; done

$PY report.py "$HASH_RUN" > "$OUT/REPORT.md"
cp "$OUT/REPORT.md" REPORT.md
echo "run $HASH_RUN done; $OUT/REPORT.md and REPORT.md written" >&2
