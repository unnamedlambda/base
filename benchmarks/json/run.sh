#!/usr/bin/env bash
# The JSON benchmark, from scratch: simdjson and its corpora at their pinned
# commits, the harness built against simdjson's single header and base's
# library, base's parser written by its Lean generator, its doubles checked
# against simdjson's on `numbers.py`'s hard cases, every corpus file checked
# and timed, then the report.
#
#   ./run.sh
#
# Each invocation is a run in results/<time>-<git rev>/ (parse.jsonl, every
# timed parse; times.tsv); REPORT.md is rewritten from it. The harness runs
# pinned to core 3 under a 9 GB memory cap with swap off.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$(cd ../.. && pwd)
SIMDJSON=(v5.0.5 9c003534b1be3bc4f1322bfa7a770a7ffbf247d9)   # 2026-10-14
DATA=4197c425e857f0ec38e89822fdd0bd9ea21f4daf                 # simdjson-data, 2025-11-21
FILES="twitter citm_catalog canada github_events gsoc-2018 mesh numbers"
cap() { systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 "$@"; }
cap true || { echo "systemd-run --user is not available: not running uncapped" >&2; exit 1; }

mkdir -p build/base
[[ -d build/simdjson ]] || git clone -q --depth=1 --branch "${SIMDJSON[0]}" https://github.com/simdjson/simdjson.git build/simdjson
[[ $(git -C build/simdjson rev-parse HEAD) == "${SIMDJSON[1]}" ]] || { echo "build/simdjson is not at ${SIMDJSON[0]}" >&2; exit 1; }
[[ -d build/simdjson-data ]] || git clone -q --filter=blob:none https://github.com/simdjson/simdjson-data.git build/simdjson-data
git -C build/simdjson-data checkout -q "$DATA"
( cd "$ROOT" && cap cargo build --release -q -p base ) || { echo "libbase did not build" >&2; exit 1; }
( cd "$ROOT" && tools/lean_build.sh genjsonbench > /dev/null ) || { echo "base's JSON generator did not build" >&2; exit 1; }
cap "$ROOT/lean/.lake/build/bin/genjsonbench" build/base > /dev/null
cap g++ -O3 -march=native -std=c++17 -o build/bench bench.cpp build/simdjson/singleheader/simdjson.cpp \
    -Ibuild/simdjson/singleheader -ldl

# the doubles, checked bit for bit against simdjson's before anything counts
python3 numbers.py build/numbers.json
cap build/bench "$ROOT/target/release/libbase.so" build/base/json.cbor build/numbers.json > /dev/null

RUN="$(date +%Y-%m-%d-%H%M%S)-$(git rev-parse --short HEAD)"
OUT=results/$RUN
mkdir -p "$OUT"
t0=$(date +%s.%N)
cap taskset -c 3 build/bench "$ROOT/target/release/libbase.so" build/base/json.cbor \
    $(for f in $FILES; do echo "build/simdjson-data/jsonexamples/$f.json"; done) > "$OUT/parse.jsonl"
printf 'bench\tok\t%.1f\n' "$(echo "$(date +%s.%N) - $t0" | bc)" >> "$OUT/times.tsv"
python3 report.py "$RUN" > "$OUT/REPORT.md"
cp "$OUT/REPORT.md" REPORT.md
echo "run $RUN done: $OUT/REPORT.md" >&2
