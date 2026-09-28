#!/usr/bin/env bash
# Common workloads through Base, against the same kernels in Rust: build, run,
# report.
#
#   ./run.sh [runs]      runs the benchmark `runs` times (default 3) and prints
#                        the tables
#
# Built with -C target-cpu=native into its own target directory, so the Rust
# side is LLVM's code for this machine. Everything runs pinned to core 3; the
# report states its own run-to-run noise, which is only low on an idle machine.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
RUNS="${1:-3}"
CORE=3
BIN="$ROOT/target/native/release/cpu-bench"

echo "building" >&2
(cd "$ROOT" && RUSTFLAGS="-C target-cpu=native" cargo build --release -q -p cpu-bench \
  --target-dir target/native)

RAW=$(mktemp -d)
trap 'rm -rf "$RAW"' EXIT
for r in $(seq 1 "$RUNS"); do
  echo "run $r/$RUNS" >&2
  taskset -c "$CORE" "$BIN" > "$RAW/run_$r.tsv"
done

echo "Base against Rust on common workloads"
echo
echo "  CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
echo "  rustc: $(rustc --version), -C target-cpu=native"
echo "  Cranelift: $(awk '/^name = "cranelift-codegen"$/ {getline; print $3}' "$ROOT/Cargo.lock" | tr -d '"')"
echo "  runs: $RUNS, pinned to core $CORE"
echo
python3 report.py "$RAW"
