#!/usr/bin/env bash
#
# Builds the Lean tree as fast as the machine allows without taking the desktop
# down with it.
#
#   tools/lean_build.sh [lake build arguments]
#
# Lake runs one Lean process per ready module, as many at once as the Lean
# runtime has worker threads (`LEAN_NUM_THREADS`, by default every hardware
# thread: pinning Lake to fewer cores does not change it), and never asks how
# much memory a module needs; the Ship proofs and the scans each need one to
# two gigabytes and the largest library module over four. So the build runs in
# a cgroup with a hard cap and no swap, and a watchdog kills it short of the
# cap. The cap alone is not enough: a Lean process maps its imports' .olean
# files, and at the cap the kernel reclaims and refaults those file pages
# before it kills anything, which stalls the whole machine. The watchdog
# kills the build when its cgroup passes 5.3G or the machine has under 2.5G
# available, so the session never sees sustained pressure. The build starts
# with as many Lean processes as fit; after a kill it tries again with fewer,
# and Lake keeps every module that finished, so a kill costs only the modules
# that were building.
#
# There is deliberately no soft limit (MemoryHigh). It throttles instead of
# killing, which holds the session under memory pressure, and systemd-oomd
# kills inside user@1000.service when that pressure stays above 50% for 20s —
# on 2026-09-16 it killed the session's init.scope and logged the user out.
#
# Debugging:
#   journalctl -u systemd-oomd        a session-level kill (the one to avoid)
#   journalctl --user -u 'run-*'      the build's own cgroup-OOM kills
#   the log it names on exit          which job count finished, and why
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

# Kill the scope `$1` when it passes 5.3G or the machine runs short, and say so
# in the log; ends when the scope does.
watchdog() {
  local cg="/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice/$1.scope"
  local cur av
  for _ in $(seq 50); do [ -d "$cg" ] && break; sleep 0.1; done
  while [ -d "$cg" ]; do
    cur=$(cat "$cg/memory.current" 2>/dev/null || echo 0)
    av=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$cur" -gt $((5300 * 1024 * 1024)) ] || [ "$av" -lt $((2500 * 1024)) ]; then
      echo "== watchdog: build at $((cur / 1048576)) MiB, $((av / 1024)) MiB available; killing it" >> "$LOG"
      systemctl --user kill --signal=KILL "$1.scope" 2>/dev/null
      return
    fi
    sleep 0.5
  done
}

status=1
for jobs in 3 2 1; do
  echo "== lake build with $jobs Lean processes at once" | tee -a "$LOG"
  from=$(( $(wc -l < "$LOG") + 1 ))
  unit="lean-build-$$-$jobs"
  watchdog "$unit" &
  LEAN_NUM_THREADS=$jobs systemd-run --user --unit="$unit" --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
    nice -n 19 lake build "$@" >> "$LOG" 2>&1
  status=$?
  wait
  # The watchdog and the cap kill with SIGKILL, Lake itself (137) or one of its Lean
  # processes, which Lake then reports as exiting with 137; systemd then
  # stops what is left of the scope with SIGTERM (143). Anything else is the
  # build's own answer.
  if [ "$status" -ne 137 ] && [ "$status" -ne 143 ] && ! tail -n "+$from" "$LOG" | grep -q 'exited with code 137'; then
    break
  fi
  echo "== killed for memory with $jobs Lean processes; fewer" | tee -a "$LOG"
done
grep -E '^error' -A8 "$LOG" | head -60
tail -1 "$LOG"
echo "log: $LOG (status $status)"
exit "$status"
