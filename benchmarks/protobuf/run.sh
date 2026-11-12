#!/usr/bin/env bash
# The protobuf benchmark, from scratch: HyperProtoBench and protobuf (C++ and
# upb) at their pinned commits, base's codec generated from each benchmark's
# .proto by its Lean generator, one harness a benchmark, every engine checked
# then timed, then the report.
#
#   ./run.sh
#
# Each invocation is a run in results/<time>-<git rev>/ (proto.jsonl, every
# timed pass; times.tsv); REPORT.md is rewritten from it. Builds and the
# harnesses run under a 9 GB memory cap with swap off, the harnesses pinned to
# core 3.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$(cd ../.. && pwd)
HPB=3ff7f5f925a7ea00fd0cab81226fa4a3ba022435         # HyperProtoBench, 2022-07-20 (archived)
PROTOBUF=(v36.2 2c74169b34066ceb8ddb6b882fcb3fb32d737a55)   # 2026-09-17
BENCHES="bench0 bench1 bench2 bench3 bench4 bench5"
cap() { systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 "$@"; }
cap true || { echo "systemd-run --user is not available: not running uncapped" >&2; exit 1; }

[[ -x .venv/bin/cmake ]] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
export PATH="$PWD/.venv/bin:$PATH"
mkdir -p build
[[ -d build/hpb ]] || git clone -q https://github.com/google/HyperProtoBench.git build/hpb
git -C build/hpb checkout -q "$HPB"
if [[ ! -d build/protobuf ]]; then
    git clone -q --depth=1 --branch "${PROTOBUF[0]}" --recurse-submodules --shallow-submodules \
        https://github.com/protocolbuffers/protobuf.git build/protobuf
fi
[[ $(git -C build/protobuf rev-parse HEAD) == "${PROTOBUF[1]}" ]] || { echo "build/protobuf is not at ${PROTOBUF[0]}" >&2; exit 1; }
I=$PWD/build/protobuf/_install
if [[ ! -x $I/bin/protoc ]]; then
    mkdir -p build/protobuf/_build
    ( cd build/protobuf/_build && cap cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF \
        -Dprotobuf_BUILD_LIBUPB=ON -Dprotobuf_ABSL_PROVIDER=module -DCMAKE_INSTALL_PREFIX="$I" .. > /dev/null \
      && cap ninja -j6 install > /dev/null )
fi
export PKG_CONFIG_PATH=$I/lib/pkgconfig
( cd "$ROOT" && cap cargo build --release -q -p base ) || { echo "libbase did not build" >&2; exit 1; }
( cd "$ROOT" && tools/lean_build.sh genprotobench > /dev/null ) || { echo "base's protobuf generator did not build" >&2; exit 1; }

# each benchmark: base's codec from its schema, protobuf's and upb's code from
# it, its top-level messages, and the harness
for b in $BENCHES; do
    G=build/gen/$b
    mkdir -p "$G" "build/base/$b"
    cap "$ROOT/lean/.lake/build/bin/genprotobench" "build/hpb/$b/benchmark.proto" "build/base/$b" > /dev/null
    "$I/bin/protoc" -I "build/hpb/$b" --cpp_out="$G" \
        --plugin=protoc-gen-upb="$I/bin/protoc-gen-upb" --upb_out="$G" \
        --plugin=protoc-gen-upb_minitable="$I/bin/protoc-gen-upb_minitable" --upb_minitable_out="$G" \
        "build/hpb/$b/benchmark.proto"
    # the generated driver, its one call protobuf has since renamed
    sed 's/Arena::CreateMessage</Arena::Create</g' "build/hpb/$b/benchmark.inc" > "$G/benchmark.inc"
    python3 tops.py "build/hpb/$b/benchmark.proto" > "$G/tops.h"
    cap gcc -O3 -march=native -I"$I/include" -I"$G" -c "$G/benchmark.upb_minitable.c" -o "$G/minitable.o"
    cap g++ -O3 -march=native -std=c++17 -I"$I/include" -I"$G" -include "$G/tops.h" bench.cpp \
        "$G/benchmark.pb.cc" "$G/minitable.o" -o "$G/bench" $(pkg-config --libs protobuf upb) -lutf8_range -ldl
done

RUN="$(date +%Y-%m-%d-%H%M%S)-$(git rev-parse --short HEAD)"
OUT=results/$RUN
mkdir -p "$OUT"
for b in $BENCHES; do
    t0=$(date +%s.%N)
    cap taskset -c 3 "build/gen/$b/bench" "$ROOT/target/release/libbase.so" "build/base/$b/proto.cbor" "$b" >> "$OUT/proto.jsonl"
    printf '%s\tok\t%.1f\n' "$b" "$(echo "$(date +%s.%N) - $t0" | bc)" >> "$OUT/times.tsv"
done
python3 report.py "$RUN" > "$OUT/REPORT.md"
cp "$OUT/REPORT.md" REPORT.md
echo "run $RUN done; $OUT/REPORT.md and REPORT.md written" >&2
