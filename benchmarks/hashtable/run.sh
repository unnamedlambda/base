#!/usr/bin/env bash
# The hash table benchmark, from scratch: base's table written by its Lean
# generator, the harness built against hashbrown, every operation and size
# measured, then the report.
#
#   ./run.sh
#
# Each invocation is a run in results/<time>-<git rev>/ (ops.jsonl, every
# sample of every measure; times.tsv); REPORT.md is rewritten from it. The
# harness runs pinned to core 3 under a 9 GB memory cap with swap off.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$(cd ../.. && pwd)
cap() { systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 "$@"; }
cap true || { echo "systemd-run --user is not available: not running uncapped" >&2; exit 1; }

mkdir -p build/base
( cd "$ROOT" && tools/lean_build.sh genhashtablebench > /dev/null ) || { echo "base's hash table generator did not build" >&2; exit 1; }
cap "$ROOT/lean/.lake/build/bin/genhashtablebench" build/base > /dev/null
RUSTFLAGS="-C target-cpu=native" cap cargo build --release -q --target-dir "$ROOT/target/hashtable-bench"

RUN="$(date +%Y-%m-%d-%H%M%S)-$(git rev-parse --short HEAD)"
OUT=results/$RUN
mkdir -p "$OUT"
t0=$(date +%s.%N)
cap taskset -c 3 "$ROOT/target/hashtable-bench/release/hashtable-bench" build/base/hashtable.cbor > "$OUT/ops.jsonl"
printf 'hashtable-bench\tok\t%.1f\n' "$(echo "$(date +%s.%N) - $t0" | bc)" >> "$OUT/times.tsv"
python3 report.py "$RUN" > "$OUT/REPORT.md"
cp "$OUT/REPORT.md" REPORT.md
echo "run $RUN done: $OUT/REPORT.md" >&2
