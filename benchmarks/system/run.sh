#!/usr/bin/env bash
# The libraries a program links, at work, against the Rust crates a program
# would use: build, run, report.
#
#   ./run.sh [runs]      runs the suite `runs` times (default 3) and prints
#                        the table
#
# Built with -C target-cpu=native into its own target directory, pinned to
# core 3. The files `kv` and `wc` touch are made fresh in the temporary
# directory and removed after.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
RUNS="${1:-3}"
CORE=3
BIN="$ROOT/target/native/release/system-bench"

echo "building" >&2
(cd "$ROOT" && RUSTFLAGS="-C target-cpu=native" cargo build --release -q -p system-bench \
  --target-dir target/native)

RAW=$(mktemp -d)
trap 'rm -rf "$RAW"' EXIT
for r in $(seq 1 "$RUNS"); do
  echo "run $r/$RUNS" >&2
  taskset -c "$CORE" "$BIN" > "$RAW/run_$r.tsv"
done

echo "Base's libraries against Rust crates"
echo
echo "  CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
echo "  rustc: $(rustc --version), -C target-cpu=native"
echo "  runs: $RUNS, pinned to core $CORE"
echo
python3 - "$RAW" <<'PY'
import collections, glob, sys
best = collections.defaultdict(lambda: float("inf"))
runs = collections.defaultdict(list)
for f in sorted(glob.glob(sys.argv[1] + "/run_*.tsv")):
    for line in open(f):
        w, col, ns = line.split()
        best[w, col] = min(best[w, col], float(ns))
        runs[w, col].append(float(ns))
rows = [("ht", "table keyed by byte strings, 100k inserts and lookups", "hashbrown + foldhash"),
        ("kv", "ordered store, 100k records in scrambled order in a transaction, then a scan", "LMDB (liblmdb)"),
        ("wc", "newlines in a 32 MB file", "std::fs + memchr")]
print(f"  {'workload':<9}{'Rust':<22}{'Rust (ms)':>10}{'Base (ms)':>11}{'Base / Rust':>13}   spread")
for w, what, crate in rows:
    r, b = best[w, "rust"], best[w, "base"]
    spread = max((max(v) - min(v)) / min(v) for v in (runs[w, "rust"], runs[w, "base"]))
    print(f"  {w:<9}{crate:<22}{r/1e6:>10.2f}{b/1e6:>11.2f}{b/r:>13.2f}   {spread*100:.1f}%")
print()
for w, what, _ in rows:
    print(f"  {w}: {what}")
PY
