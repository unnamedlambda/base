"""REPORT.md from one run's results.

    python report.py [run]      # default: the newest run under results/
"""

import glob
import json
import math
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R_DIR = os.path.join(HERE, "results")
FUNCTIONS = ("memcpy", "memmove", "memset", "bzero", "memcmp", "bcmp")
IMPLS = ("base", "llvm-libc", "glibc", "rep")


def newest_run() -> str:
    runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(R_DIR, "*", "times.tsv")))
    if not runs:
        raise SystemExit("no run with times.tsv under results/")
    return runs[-1]


def table(fn, rows):
    rows = [r for r in rows if r["function"] == fn]
    if not rows:
        return ""
    impls = [i for i in IMPLS if any(r["impl"] == i for r in rows)]
    dists = sorted({r["distribution"] for r in rows})
    at = {(r["distribution"], r["impl"]): r["ns_per_call"] for r in rows}
    lines = [f"### {fn}", "", "| distribution | " + " | ".join(f"{i} ns/call" for i in impls) + " |",
             "|---|" + "---:|" * len(impls)]
    for d in dists:
        vals = [at.get((d, i)) for i in impls]
        best = min(v for v in vals if v is not None)
        cells = [("—" if v is None else (f"**{v:.2f}**" if v == best else f"{v:.2f}")) for v in vals]
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    geo = []
    for i in impls:
        vs = [at[(d, i)] for d in dists if at.get((d, i))]
        geo.append(math.exp(sum(map(math.log, vs)) / len(vs)) if len(vs) == len(dists) else None)
    best = min(g for g in geo if g is not None)
    lines.append("| geometric mean | " + " | ".join(("—" if g is None else (f"**{g:.2f}**" if g == best else f"{g:.2f}")) for g in geo) + " |")
    return "\n".join(lines) + "\n"


def times_table(run):
    lines = ["### Run time", "", "| step | minutes | status |", "|---|---:|---|"]
    total = 0.0
    for line in open(os.path.join(R_DIR, run, "times.tsv")):
        cmd, status, secs = line.rstrip("\n").split("\t")
        total += float(secs)
        lines.append(f"| {cmd} | {float(secs) / 60:.1f} | {status} |")
    lines.append(f"| all | {total / 60:.1f} | |")
    return "\n".join(lines) + "\n"


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else newest_run()
    cpu = subprocess.run("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-", shell=True, capture_output=True, text=True).stdout.strip()
    glibc = platform.libc_ver()[1]
    cc = subprocess.run("c++ --version | head -1", shell=True, capture_output=True, text=True).stdout.strip()
    print(f"# Memory operations benchmark: run {run}\n")
    print(f"Machine: {cpu}; Linux {platform.release()}; glibc {glibc}; llvm-libc built by {cc}.\n")
    ctx = os.path.join(R_DIR, run, "context.json")
    if os.path.exists(ctx) and json.load(open(ctx)).get("cpu_scaling_enabled"):
        print("The CPU's frequency scaling was on (Google Benchmark's warning): absolute times carry the clock's state.\n")
    print(times_table(run))
    p = os.path.join(R_DIR, run, "memops.jsonl")
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    rows = [r for r in rows if not r["quick"]]
    if rows:
        print("llvm-libc's memory benchmark: calls with sizes drawn from each of Google's published fleet")
        print("distributions and random offsets, within L1. Median of 5 repetitions, nanoseconds per call (lower")
        print("is faster); the fastest per row in bold. rep: `rep movsb` / `rep stosb`.\n")
        for fn in FUNCTIONS:
            t = table(fn, rows)
            if t:
                print(t)

HERE = os.path.dirname(os.path.abspath(__file__))


def headline():
    """The memory-operations suite's headline rows for the top-level report
    (../suite.py): from the newest full run, nanoseconds per call of every
    implementation on each function and Google size distribution."""
    runs = sorted(glob.glob(os.path.join(HERE, "results", "*", "memops.jsonl")))
    out = []
    for path in reversed(runs):
        rs = [json.loads(l) for l in open(path)]
        full = [r for r in rs if not r["quick"] and r["ns_per_call"] is not None]
        if not full:
            continue
        for r in full:
            out.append(dict(suite="memops", workload=f"{r['function']}, {r['distribution']}", measure="time a call",
                            unit="ns", higher=False, engine=r["impl"], value=r["ns_per_call"], base=r["impl"] == "base",
                            run=os.path.basename(os.path.dirname(path))))
        break
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--headline"]:
        for row in headline():
            print(json.dumps(row))
    else:
        main()
