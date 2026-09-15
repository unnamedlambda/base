#!/usr/bin/env bash
# Run every generator into a directory of your choosing, so two trees can be
# diffed. `cargo build -p lean-artifacts` writes the same tree in place; this
# writes it somewhere a comparison can survive the next build.
#
#   tools/regen-artifacts.sh <out-dir> [module ...]
#
# With no modules named, every generator runs. Naming modules runs only those,
# which is what a batch port wants.
set -euo pipefail

out=${1:?usage: regen-artifacts.sh <out-dir> [module ...]}
shift || true
want=("$@")

root=$(cd "$(dirname "$0")/.." && pwd)
algs="$root/lean/algorithms"
bin="$algs/.lake/build/bin"

# Build the generator executables first. Running a stale binary would make the
# comparison meaningless, which is exactly the failure this script exists to
# catch.
mapfile -t lines < <(cd "$algs" && lake run generators)
targets=()
for line in "${lines[@]}"; do
  set -- $line
  if [ ${#want[@]} -gt 0 ]; then
    hit=no
    for w in "${want[@]}"; do [ "$w" = "$2" ] && hit=yes; done
    [ "$hit" = yes ] || continue
  fi
  targets+=("$1")
done
(cd "$algs" && lake build "${targets[@]}" >/dev/null)

mkdir -p "$out"
while read -r exe module; do
  if [ ${#want[@]} -gt 0 ]; then
    hit=no
    for w in "${want[@]}"; do [ "$w" = "$module" ] && hit=yes; done
    [ "$hit" = yes ] || continue
  fi
  rm -rf "${out:?}/$module"
  mkdir -p "$out/$module"
  "$bin/$exe" "$out/$module" >/dev/null
  echo "$module"
done < <(cd "$algs" && lake run generators)
