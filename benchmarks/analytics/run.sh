#!/usr/bin/env bash
# The analytics benchmark, from scratch: ClickBench's first 10 partitions
# (checked), the columns extracted, base's engine written by its Lean
# generator, every engine's hot runs, then the report.
#
#   ./run.sh
#
# Each invocation is a run in results/<time>-<git rev>/ (queries.jsonl,
# every timed run of every query; times.tsv); REPORT.md is rewritten from
# it. Each step runs under a 9 GB memory cap with swap off.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$(cd ../.. && pwd)
PY=.venv/bin/python
cap() { systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 "$@"; }
cap true || { echo "systemd-run --user is not available: not running uncapped" >&2; exit 1; }

[[ -x $PY ]] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
[[ -x $ROOT/py-base/.venv/bin/python ]] || { echo "py-base is not built (tools/check.sh builds it)" >&2; exit 1; }
cap $PY prepare.py
mkdir -p build/base
( cd "$ROOT" && tools/lean_build.sh genanalyticsbench > /dev/null ) || { echo "base's analytics generator did not build" >&2; exit 1; }
cap "$ROOT/lean/.lake/build/bin/genanalyticsbench" build/base > /dev/null

export ANALYTICS_RUN="$(date +%Y-%m-%d-%H%M%S)-$(git rev-parse --short HEAD)"
OUT=results/$ANALYTICS_RUN
mkdir -p "$OUT"
for e in duckdb base; do
    t0=$(date +%s.%N)
    cap env ANALYTICS_RUN="$ANALYTICS_RUN" $PY bench.py "$e" || { echo "FAILED: bench.py $e" >&2; exit 1; }
    printf 'bench.py %s\tok\t%.1f\n' "$e" "$(echo "$(date +%s.%N) - $t0" | bc)" >> "$OUT/times.tsv"
done
$PY report.py "$ANALYTICS_RUN" > "$OUT/REPORT.md"
cp "$OUT/REPORT.md" REPORT.md
echo "run $ANALYTICS_RUN done: $OUT/REPORT.md" >&2
