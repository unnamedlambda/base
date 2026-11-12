"""REPORT.md from one run's results.

    python report.py [run]      # default: the newest run under results/

The reproduction reads SMHasher3's published results from the pinned tree
(build/smhasher3/results/raw/<hash>.txt) with the same parser.
"""

import glob
import json
import os
import platform
import subprocess
import sys

from prepare import SMHASHER
from speed import parse

HERE = os.path.dirname(os.path.abspath(__file__))
R_DIR = os.path.join(HERE, "results")


def newest_run() -> str:
    runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(R_DIR, "*", "times.tsv")))
    if not runs:
        raise SystemExit("no run with times.tsv under results/")
    return runs[-1]


def num(x, fmt):
    return "—" if x is None else format(x, fmt)


def speed_table(rows, ghz):
    lines = [
        "### Speed",
        "",
        "SMHasher3's Speed suite on one thread. Small keys: cycles per hash, averaged over 1-31-byte keys",
        f"(lower is faster). Bulk: bytes per cycle over 256 KiB keys, and GB/s at this machine's {ghz:.2f} GHz",
        "time-stamp counter. Sorted by small-key cost, the figure a hash table's lookups pay.",
        "",
        "| hash | small keys, cycles/hash | bulk, bytes/cycle | bulk GB/s | verification | sanity |",
        "|---|---:|---:|---:|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r.get("small_cycles", 1e9)):
        gbps = r["bulk_bytes_per_cycle"] * ghz if r.get("bulk_bytes_per_cycle") else None
        lines.append(
            f"| {r['hash']} | {num(r.get('small_cycles'), '.1f')} | {num(r.get('bulk_bytes_per_cycle'), '.2f')} | {num(gbps, '.1f')} | "
            f"{r.get('verify', '?')} | {'pass' if r.get('sanity_pass') else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"


def repro_table(rows):
    lines = [
        "### Reproduction: SMHasher3's published results",
        "",
        "SMHasher3's results directory at the pinned commit (`results/raw/`). The verification value is a",
        "hash over a fixed set of the hash's outputs: equal values mean the same function. Speeds are in",
        "cycles, so their ratio is the microarchitecture's difference, not the clock's.",
        "",
        "| hash | verification published | here | same | small-key cycles published | here | bulk bytes/cycle published | here |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    any_row = False
    for r in rows:
        path = os.path.join(SMHASHER, "results", "raw", f"{r['hash']}.txt")
        if not os.path.exists(path):
            continue
        p = parse(open(path).read())
        any_row = True
        same = "yes" if p.get("verify_value") == r.get("verify_value") else "NO"
        lines.append(
            f"| {r['hash']} | {p.get('verify_value', '—')} | {r.get('verify_value', '—')} | {same} | "
            f"{num(p.get('small_cycles'), '.1f')} | {num(r.get('small_cycles'), '.1f')} | "
            f"{num(p.get('bulk_bytes_per_cycle'), '.2f')} | {num(r.get('bulk_bytes_per_cycle'), '.2f')} |"
        )
    return "\n".join(lines) + "\n" if any_row else ""


def times_table(run):
    lines = ["### Run time", "", "| step | minutes | status |", "|---|---:|---|"]
    total = 0.0
    for line in open(os.path.join(R_DIR, run, "times.tsv")):
        cmd, status, secs = line.rstrip("\n").split("\t")
        total += float(secs)
        lines.append(f"| {cmd} | {float(secs) / 60:.1f} | {status} |")
    lines.append(f"| all | {total / 60:.1f} | |")
    return "\n".join(lines) + "\n"


def tsc_ghz() -> float:
    """The time-stamp counter's rate, from the kernel's calibration of it: on
    x86 the delay loop runs on the TSC, so BogoMIPS is exactly twice its rate
    in MHz (here 7186.09: 3.593 GHz, as rdtsc against the wall clock gives).
    Not `cpu MHz`, the core's current clock, nor dmesg, which needs root."""
    for line in open("/proc/cpuinfo"):
        if line.lower().startswith("bogomips"):
            return float(line.split(":")[1]) / 2e3
    return float("nan")


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else newest_run()
    cpu = subprocess.run("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-", shell=True, capture_output=True, text=True).stdout.strip()
    cc = subprocess.run("c++ --version | head -1", shell=True, capture_output=True, text=True).stdout.strip()
    print(f"# Hashing benchmark: run {run}\n")
    print(f"Machine: {cpu}; Linux {platform.release()}; {cc}.\n")
    print(times_table(run))
    p = os.path.join(R_DIR, run, "speed.jsonl")
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    if rows:
        print(speed_table(rows, tsc_ghz()))
        t = repro_table(rows)
        if t:
            print(t)

def headline():
    """The hashing suite's headline rows for the top-level report (../suite.py):
    from the newest run, each hash's small-key cycles per hash (1-31-byte keys,
    lower is better) and bulk bytes per cycle (256 KiB keys), for the hashes
    whose verification value passed. Base has no hash on this suite yet, so every
    row is the frontier's."""
    run = newest_run()
    p = os.path.join(R_DIR, run, "speed.jsonl")
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p):
        r = json.loads(line)
        if r.get("verify") != "PASS":
            continue
        for measure, key, unit, higher in (("small keys", "small_cycles", "cycles/hash", False),
                                           ("bulk", "bulk_bytes_per_cycle", "B/cycle", True)):
            if r.get(key) is not None:
                out.append(dict(suite="hash", workload="non-cryptographic hash (SMHasher3)", measure=measure,
                                unit=unit, higher=higher, engine=r["hash"], value=r[key], base=False, run=run))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--headline"]:
        for row in headline():
            print(json.dumps(row))
    else:
        main()
