#!/usr/bin/env bash
# The ANN benchmark, from scratch, measured as ann-benchmarks and
# big-ann-benchmarks measure: environment, sources, prepared datasets, every
# engine through every harness, then the report.
#
#   ./run.sh                    ann-benchmarks' SIFT-1M and GloVe-100 curves
#                               (curves.py), the streaming runbook on SIFT-1M
#                               (stream.py), and one query a call on SIFT-1M
#                               (latency.py)
#   ./run.sh sift-1m base       one dataset, one engine (the development loop)
#   ./run.sh t2i-1m             big-ann's out-of-distribution track, its own invocation
#   ./run.sh yfcc-1m            big-ann's filtered track (filtered.py), its own invocation
#   ./run.sh bigann-10m         BIGANN at 10M (BIG_ENGINES), its own invocation
#
# Nothing measured is reused: each invocation is one run, written to
# results/<time>-<git rev>/ with every process's wall time (times.tsv), and
# REPORT.md is rewritten from it. Downloads and prepared datasets are inputs
# and stay in data/ann/.
#
# The tiny datasets run first through every harness and engine that will run,
# and the run stops if any of them fails there: a broken harness costs a
# couple of minutes, not the full run.
#
# Each step is its own process under a 9 GB memory cap with swap off. The cap
# is safety, not a benchmark condition: it leaves this 15.5 GB machine about
# 3 GB for its desktop and system, and an uncapped run that exhausts memory
# freezes it. With swap off a runaway is killed at once rather than thrashing.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=.venv/bin/python

cap() { systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 nice -n 19 "$@"; }
cap true || { echo "systemd-run --user is not available: not running uncapped" >&2; exit 1; }

if [[ ! -x $PY ]]; then
    python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt
fi
# base's engine is an input, as the datasets are: its artifacts written by
# the Lean generator (lean/algorithms/Ann), run through py-base.
( cd ../.. && tools/lean_build.sh genannalgorithm > /dev/null ) || { echo "base's ANN generator did not build" >&2; exit 1; }
# The artifacts are rewritten only when the generator is newer than one of them.
stale=0
for s in 128_l2 100_ip 200_ip 192_l2; do
  [[ ../../lean/.lake/build/artifacts/ann_$s.cbor -nt ../../lean/.lake/build/bin/genannalgorithm ]] || stale=1
done
if (( stale )); then
  ( cd ../../lean && cap .lake/build/bin/genannalgorithm .lake/build/artifacts ) || { echo "base's ANN artifacts were not written" >&2; exit 1; }
fi
[[ -x ../../py-base/.venv/bin/python ]] || { echo "py-base is not built (tools/check.sh builds it)" >&2; exit 1; }
DATASETS="${1:-sift-1m glove-100}"
# The engines: base, and the leaders it is measured against, each the best
# of its family in the exploration this suite began with: CAGRA (the GPU
# graph, cuvs) and ScaNN (the CPU's fastest) on the curves; hnswlib (the
# best streaming recall) and ScaNN (the fastest runbook) on the stream.
ENGINES="${2:-base seg-cagra scann}"
STREAM_ENGINES="${2:-base hnswlib scann}"
# One query a call: base and the leaders there, the CPU's graph and
# quantizer (hnswlib, ScaNN), and CAGRA.
LATENCY_ENGINES="${2:-base hnswlib scann seg-cagra}"
# At 10M the same leaders.
BIG_ENGINES="${2:-base seg-cagra scann}"
# The filtered track ranks native filtering, as big-ann's does: base, Qdrant
# (the only other engine here that filters natively), and exact brute force,
# the ceiling.
FILTER_ENGINES="${2:-base qdrant exact}"
CHECKS=tiny
if [[ " $DATASETS " == *" yfcc-"* ]]; then CHECKS="$CHECKS yfcc-tiny"; fi
cap $PY prepare.py $CHECKS $DATASETS

export ANN_RUN="$(date +%Y-%m-%d-%H%M%S)-$(git rev-parse --short HEAD)"
OUT=results/$ANN_RUN
mkdir -p "$OUT"

# One process. 137 is a kill by the 9 GB memory cap, 3 an engine out of the
# card's memory (common.harness): no-fit, an outcome on a full dataset, not a
# broken harness. On a tiny dataset anything but ok stops the run.
STRICT=1
step() {
    local t0 t1 rc=0 status=ok
    t0=$(date +%s.%N)
    cap env ANN_RUN="$ANN_RUN" $PY "$@" || rc=$?
    t1=$(date +%s.%N)
    [[ $rc -eq 137 || $rc -eq 3 ]] && status=no-fit
    [[ $status == ok && $rc -ne 0 ]] && status=failed
    printf '%s\t%s\t%.1f\n' "$*" "$status" "$(echo "$t1 - $t0" | bc)" >> "$OUT/times.tsv"
    if [[ $status != ok && $STRICT == 1 ]]; then
        echo "FAILED on a tiny dataset: $*" >&2
        exit 1
    fi
    [[ $status == failed ]] && echo "FAILED: $*" >&2
    return 0
}

# Every harness for one dataset: the filtered track's through filtered.py;
# the rest through curves.py, stream.py where the dataset has a runbook, and
# latency.py on SIFT-1M (ann-benchmarks' default mode, one query a call).
dataset() {
    local ds=$1 e engines=$ENGINES
    [[ $ds == bigann-* ]] && engines=$BIG_ENGINES
    if [[ $ds == yfcc-* ]]; then
        for e in $FILTER_ENGINES; do step filtered.py "$ds" "$e"; done
    else
        for e in $engines; do step curves.py "$ds" "$e"; done
        if [[ -f ../../data/ann/$ds/stream_gt.npy ]]; then
            for e in $STREAM_ENGINES; do step stream.py "$ds" "$e"; done
        fi
        if [[ $ds == sift-1m || $ds == tiny ]]; then
            for e in $LATENCY_ENGINES; do step latency.py "$ds" "$e"; done
        fi
    fi
}

cap $PY -c "from common import idle_power; idle_power()"
for ds in $CHECKS; do dataset "$ds"; done
STRICT=0
for ds in $DATASETS; do dataset "$ds"; done
$PY report.py "$ANN_RUN" > "$OUT/REPORT.md"
# REPORT.md: each dataset from the newest run with every engine (a run naming
# its engines is a development run and stays in its own directory).
if [[ -z "${2:-}" ]]; then touch "$OUT/full"; fi
$PY report.py --latest > REPORT.md
echo "run $ANN_RUN done; $OUT/REPORT.md and REPORT.md written" >&2
