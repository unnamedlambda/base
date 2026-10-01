#!/usr/bin/env bash
#
# Builds the Lean tree as fast as the machine allows without taking the desktop
# down with it.
#
#   tools/lean_build.sh [lake build arguments]
#
# Lake runs one Lean process per ready module, as many as it sees cores, and
# never asks how much memory a module needs; a few here need gigabytes. So the
# build runs in a cgroup with a hard cap and no swap: a build that reaches the
# cap is killed at once by the kernel, inside the cgroup, and the session never
# sees sustained pressure. It starts on every core; after a kill it tries again
# on fewer, and Lake keeps every module that finished, so a kill costs only the
# modules that were building.
#
# There is deliberately no soft limit (MemoryHigh). It throttles instead of
# killing, which holds the session under memory pressure, and systemd-oomd
# kills inside user@1000.service when that pressure stays above 50% for 20s —
# on 2026-09-16 it killed the session's init.scope and logged the user out.
#
# Debugging:
#   journalctl -u systemd-oomd        a session-level kill (the one to avoid)
#   journalctl --user -u 'run-*'      the build's own cgroup-OOM kills
#   the log it names on exit          which core count finished, and why
#
# Exits with Lake's status on the last attempt.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/lean"
# elan puts Lake here; a detached run (a systemd unit) does not have the
# shell's PATH.
export PATH="$HOME/.elan/bin:$PATH"

LOG="${LEAN_BUILD_LOG:-$ROOT/target/lean_build.log}"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

last=$(( $(nproc) - 1 ))
status=1
for top in "$last" 5 3 1; do
  [ "$top" -gt "$last" ] && continue
  echo "== lake build on cores 0-$top" | tee -a "$LOG"
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
    nice -n 19 taskset -c "0-$top" lake build "$@" >> "$LOG" 2>&1
  status=$?
  # The cap kills with SIGKILL, Lake itself (137) or one of its Lean
  # processes, which Lake then reports as exiting with 137; systemd then
  # stops what is left of the scope with SIGTERM (143). Anything else is the
  # build's own answer.
  if [ "$status" -ne 137 ] && [ "$status" -ne 143 ] && ! tail -n 400 "$LOG" | grep -q 'exited with code 137'; then
    break
  fi
  echo "== killed at the memory cap on cores 0-$top; fewer cores" | tee -a "$LOG"
done
grep -E '^error' -A8 "$LOG" | head -60
tail -1 "$LOG"
echo "log: $LOG (status $status)"
exit "$status"
