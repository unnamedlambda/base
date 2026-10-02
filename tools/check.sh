#!/usr/bin/env bash
#
# What has to hold before this tree is worth pushing.
#
#   tools/check.sh          everything, including the GPU tests
#   tools/check.sh --fast   skips the tests that need a device or the weights
#
# The GPU half is the strongest evidence here — the Qwen2 golden tests run the
# real 2.5 GB weights end to end — and it is the half no hosted runner can
# reach, which is why this is a local script rather than a workflow.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

# A cgroup caps the whole process tree and, decisively, disables swap: a runaway
# elaboration then gets killed instead of thrashing the machine into a state
# where even the SSH session stops responding. `ulimit -v` is the wrong tool —
# it is per-process, so N jobs multiply it, and it counts thread stacks.
#
# 10G was too high to be a guard. The cap has to sit below what the session
# needs to stay alive, or the kernel is still short when the build reaches it
# and systemd-oomd kills the whole user slice instead — which logged the user
# out rather than failing the build. 6G is under that, and the build fits.
GUARD=()
if command -v systemd-run >/dev/null 2>&1; then
  GUARD=(systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- nice -n 19)
fi
# Lake runs as many Lean processes at once as the Lean runtime has worker
# threads, every hardware thread by default whatever cores it is pinned to,
# and the Ship proofs and scans each need one to two gigabytes: three fit the
# cap. Cargo's build scripts run Lake too, so this covers them as well.
export LEAN_NUM_THREADS="${LEAN_NUM_THREADS:-3}"

# The GPU is not shareable, and the driver that holds it does not fail politely.
#
# `applications/gpt-oss/*` pin 9.48 GiB of host memory and most of the card, and
# they take an exclusive lock on this file for exactly that reason. This script
# needs a CUDA device for the Qwen2 golden tests, so it takes the same lock
# rather than starting and failing on CUDA_ERROR_OUT_OF_MEMORY several minutes
# in -- which is what happened, and what the message below is for.
ENGINE_LOCK=/tmp/gpt-oss-engine.lock
if command -v flock >/dev/null 2>&1; then
  exec 9>>"$ENGINE_LOCK" || true
  if ! flock -n 9; then
    held=$(awk '{print $1, $2}' "$ENGINE_LOCK" 2>/dev/null || echo "?")
    echo "a gpt-oss driver is running ($held) and holds the GPU."
    echo "check.sh needs a device for the golden tests. Wait for it, or stop it."
    exit 1
  fi
fi

FAILED=()
# Each step returns its own verdict explicitly. `set -e` is suspended inside a
# function called from a condition, so a step that let a failing command run on
# would report success -- which is the one failure mode a check script cannot
# have.
step() {
  local name="$1"; shift
  printf '\n\033[1m== %s\033[0m\n' "$name"
  local t0=$SECONDS
  if "$@"; then
    printf '\033[32mok\033[0m  %s (%ds)\n' "$name" "$((SECONDS - t0))"
  else
    printf '\033[31mFAILED\033[0m  %s (%ds)\n' "$name" "$((SECONDS - t0))"
    FAILED+=("$name")
  fi
}

# --- the Lean side -----------------------------------------------------------
# One invocation: lake decides staleness by hashing contents, so asking twice
# only costs the up-to-date check twice, and a failing build caches nothing.
#
# The root package's libraries reach only the library modules they import, so
# the library is built whole as well: a module nothing imports -- a proof such
# as `Host.Sound` -- is otherwise never checked.

lean_build() {
  cd "$ROOT/lean"
  local libs
  libs=$(grep -oP '^lean_lib \K\w+' lakefile.lean | tr '\n' ' ')
  "${GUARD[@]}" taskset -c 0-3 lake build algorithmLib/AlgorithmLib $libs &&
    "${GUARD[@]}" taskset -c 0-3 lake build algorithmLib/artifacts
}

# `sorry` leaves a warning rather than an error, so a file can carry one and
# still build clean. The trust scans run inside the build above and fail it when
# a claim starts resting on an open obligation; this catches the blunter case.
no_sorry() {
  local hits
  hits=$(python3 - "$ROOT" <<'PY'
import re, sys, pathlib

# `sorry` is a term, and the word also appears in prose *about* it -- several
# docstrings here explain why `0 sorry` is a weak claim. Comments come out
# first, so an honest note about the check does not trip it.
root = pathlib.Path(sys.argv[1])
block = re.compile(r"/-.*?-/", re.S)
found = []
for d in ("lean/lib", "lean/algorithms"):
    for p in sorted((root / d).rglob("*.lean")):
        if ".lake" in p.parts:
            continue
        src = p.read_text(errors="replace")
        # Replace each block comment with its own newlines so line numbers hold.
        src = block.sub(lambda m: "\n" * m.group(0).count("\n"), src)
        for n, ln in enumerate(src.splitlines(), 1):
            if re.search(r"\bsorry\b", re.sub(r"--.*", "", ln)):
                found.append(f"{p.relative_to(root)}:{n}: {ln.strip()}")
print("\n".join(found))
PY
)
  if [ -n "$hits" ]; then echo "$hits"; return 1; fi
  echo "no sorry in the library or the generators"
}

# The library imports only downward, applications import no other application,
# and no generator reaches a host proof beyond the listed exceptions. With the
# module system this is what keeps a proof edit from rebuilding the generators.
layers() {
  python3 "$ROOT/tools/layers.py"
}

# `AlgorithmLib.X86` must encode as GNU `as` does. The encoder is pure Lean, so
# one kind of host checks it for all; elsewhere the step is skipped.
x86_vs_as() {
  if [ "$(uname -m)" != x86_64 ] || ! as --version 2>/dev/null | grep -q 'GNU assembler'; then
    echo "skipped: needs an x86-64 host with GNU as"
    return 0
  fi
  cd "$ROOT/lean"
  "${GUARD[@]}" lake env lean -Dexperimental.module=true --run algorithms/Bench/X86Check.lean
}

# --- artifacts ---------------------------------------------------------------
# Regenerating and diffing is what says a generator is a function of its source
# and nothing else. It is also the check that licenses a refactor: if every
# artifact is byte-identical, the proofs and the Rust below them cannot have
# noticed the change.

artifacts_reproduce() {
  local query dir runs fresh rc=0
  cd "$ROOT/lean"
  query=$(lake query algorithmLib/artifacts) || return 1
  dir=$(sed -n 's/^dir //p' <<<"$query")
  runs=$(mktemp -d)
  fresh=$(mktemp -d)
  while read -r gen bin; do
    mkdir -p "$runs/$gen"
    "$bin" "$runs/$gen" >/dev/null || rc=1
  done < <(sed -n 's/^generator //p' <<<"$query")
  # Flattened as the target flattens them. It refuses a name two generators
  # share, so nothing here is overwritten.
  for g in "$runs"/*/; do cp -r "$g". "$fresh"/; done
  diff -rq "$dir" "$fresh" || rc=1
  [ $rc -eq 0 ] &&
    echo "$(find "$fresh" -name '*.cbor' | wc -l) artifacts reproduce byte-for-byte"
  rm -rf "$runs" "$fresh"
  return $rc
}

# --- the Rust side -----------------------------------------------------------
# `--all-targets` is the minimum: `cargo check` does not build test targets, so
# a type change can pass a plain check and fail only in the tests that use it.

rust_check() {
  cd "$ROOT"
  "${GUARD[@]}" taskset -c 0-3 cargo check -j 2 --workspace --all-targets \
    --exclude bench-scaling
}

# The benchmarks crate is never run here: it is long, it needs the device to be
# quiet to mean anything, and it proves nothing about correctness.
rust_test() {
  cd "$ROOT"
  "${GUARD[@]}" taskset -c 0-3 cargo test -j 2 --workspace \
    --exclude bench-scaling
}

rust_test_fast() {
  cd "$ROOT"
  "${GUARD[@]}" taskset -c 0-3 cargo test -j 2 --workspace \
    --exclude bench-scaling --exclude qwen2
}

# The MXFP4 expert kernels are outside the machine this project proves kernels
# in -- they unpack nibbles, and that machine holds Float32 and addresses buffers
# by element. So they ship checked rather than proven, and this is the check:
# every one of the 256 byte values against `QuantMX.fp4Val`, the contractions
# against a Float32 fold in the kernel's own order, and the activation at its
# clamp edges. It needs a device, so it runs in the full pass and not `--fast`.
#
# The module is dumped rather than taken from an artifact because the harness
# wants all four kernels under their own names, and an artifact carries one
# kernel per slot called `main`.
gptoss_kernels() {
  local ptx
  ptx=$(mktemp --suffix=.ptx)
  cd "$ROOT/lean"
  lake env lean --run "$ROOT/tools/dump_gptoss_kernels.lean" > "$ptx"
  "$ROOT/py-base/.venv/bin/python" "$ROOT/applications/gpt-oss/kernel_test.py" "$ptx"
  local rc=$?
  rm -f "$ptx"
  return $rc
}

# --- the Lean host -----------------------------------------------------------
# The third way to run an artifact, beside Rust and `py-base`: a Lean
# program that builds one and runs it in the same process. It is a separate Lake
# package, so the build above does not reach it.
#
# `lake` builds `libbase.so` itself here -- the package's `libbase` target runs
# cargo -- so this needs no ordering against the Rust steps. Running the demo is
# the point: it links against the runtime through the C ABI, and a link that
# succeeds proves nothing about whether the two agree on the ABI.

lean_host() {
  cd "$ROOT/lean/host"
  "${GUARD[@]}" lake build upcasehost || return 1
  # From a scratch directory: the demo reads and writes in its working
  # directory, and `lean/host` is a tree we do not want it writing into.
  local work
  work=$(mktemp -d)
  ( cd "$work" && "$ROOT/lean/host/.lake/build/bin/upcasehost" )
  local rc=$?
  rm -rf "$work"
  return $rc
}

# --- run ---------------------------------------------------------------------

step "lean build"          lean_build
step "no sorry"            no_sorry
step "layers"              layers
step "x86 against as"      x86_vs_as
step "artifacts reproduce" artifacts_reproduce
step "rust check"          rust_check
step "lean host"           lean_host
if [ "$FAST" = 1 ]; then
  step "rust test (fast)"  rust_test_fast
else
  step "rust test"         rust_test
  step "gpt-oss kernels"   gptoss_kernels
fi

printf '\n'
if [ ${#FAILED[@]} -eq 0 ]; then
  printf '\033[32mall checks passed\033[0m (%ds)\n' "$SECONDS"
else
  printf '\033[31m%d failed:\033[0m %s\n' "${#FAILED[@]}" "${FAILED[*]}"
  exit 1
fi
