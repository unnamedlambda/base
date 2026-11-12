"""REPORT.md from one run's results.

    python report.py [run]      # one run; default: the newest under results/
    python report.py --latest   # each dataset from the newest full run that measured it

A run is results/<time>-<git rev>/ (run.sh). Energy is the GPU board's
cumulative counter over a timed window: "total" is everything it saw,
"dynamic" subtracts the GPU's idle power measured at the start of the run.
CPU package energy appears only where RAPL is readable.
"""

import glob
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict

from common import HERE

R_DIR = os.path.join(HERE, "results")
TARGETS = (0.9, 0.95, 0.99)  # 0.9 is big-ann-benchmarks' threshold
CHECKS = {"tiny", "yfcc-tiny"}


def newest_run() -> str:
    runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(R_DIR, "*", "times.tsv")))
    if not runs:
        raise SystemExit("no run with times.tsv under results/")
    return runs[-1]


def rows(run: str, harness: str, dataset: str):
    p = os.path.join(R_DIR, run, f"{harness}-{dataset}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def datasets(run: str, harness: str):
    pre = os.path.join(R_DIR, run, f"{harness}-")
    return sorted({p[len(pre) : -len(".jsonl")] for p in glob.glob(pre + "*.jsonl")} - CHECKS)


def idle(run: str):
    p = os.path.join(R_DIR, run, "idle.jsonl")
    return json.loads(open(p).readline()) if os.path.exists(p) else {}


def mj(energy, idle_w) -> str:
    """'total (dynamic)' GPU millijoules per query, or '—'."""
    if not energy or energy.get("gpu_j") is None or not energy.get("queries"):
        return "—"
    total = energy["gpu_j"] / energy["queries"] * 1e3
    if idle_w is None:
        return f"{total:.3g}"
    dyn = (energy["gpu_j"] - idle_w * energy["window_s"]) / energy["queries"] * 1e3
    return f"{total:.3g} ({max(dyn, 0):.3g})"


def num(x, fmt=",.0f"):
    return "—" if x is None else format(x, fmt)


def fastest(points, t):
    return max((r["qps"] for r in points if r.get("recall10", -1) >= t), default=None)


def curves_table(ds, rs, idle_w):
    by = defaultdict(list)
    for r in rs:
        by[r["engine"]].append(r)
    lines = [
        f"### {ds}",
        "",
        "Queries/s (batches of 1,000, host memory to host memory) at recall@10 ≥ each threshold, from each",
        "engine's sweep of its settings; the highest recall it reached; GPU mJ per query at its 0.9 point,",
        "total (dynamic); and the build: every item in one insert, with the bytes per item it added (device",
        "through its memory pool, host as resident memory, a server's included).",
        "",
        "| engine | q/s @0.9 | q/s @0.95 | q/s @0.99 | max recall | mJ/query @0.9 | build s | device B/item | host B/item |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = sorted(by, key=lambda e: -(fastest([r for r in by[e] if r["phase"] == "point"], 0.9) or 0))
    for e in order:
        pts = [r for r in by[e] if r["phase"] == "point"]
        b = next((r for r in by[e] if r["phase"] == "build"), {})
        p = next((r for r in by[e] if r["phase"] == "power"), {})
        top = max((r["recall10"] for r in pts), default=None)
        lines.append(
            f"| {e} | " + " | ".join(num(fastest(pts, t)) for t in TARGETS) + f" | {num(top, '.3f')} | {mj(p.get('energy'), idle_w)} | "
            f"{num(b.get('seconds'), '.1f')} | {num(b.get('device_bytes_per_item'))} | {num(b.get('host_bytes_per_item'))} |"
        )
    return "\n".join(lines) + "\n"


def stream_table(ds, rs):
    runs = sorted((r for r in rs if r["phase"] == "runbook"), key=lambda r: -r["mean_recall10"])
    if not runs:
        return ""
    lines = [
        f"### {ds}: streaming runbook",
        "",
        "A window of 16 k-means clusters slides across the items: each step inserts the next cluster and",
        "deletes the oldest; every 4 steps all queries are searched with one fixed setting (the cheapest",
        "reaching recall 0.9 on the first window) and scored against the exact top 10 over the live set.",
        "big-ann-benchmarks' streaming measures: mean recall over the searches, and the runbook's time.",
        "",
        "Inserts/s and deletes/s are the calls' own time; indexing s is the time spent waiting for an",
        "engine to finish indexing before each search, where it defers that work.",
        "",
        "| engine | mean recall@10 | min recall@10 | runbook s | inserts/s | deletes/s | indexing s | search q/s | setting |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in runs:
        lines.append(
            f"| {r['engine']} | {r['mean_recall10']:.3f} | {r['min_recall10']:.3f} | {r['seconds']:.1f} | "
            f"{r['insert_per_s']:,.0f} | {r['delete_per_s']:,.0f} | {r.get('indexing_seconds', 0):.1f} | "
            f"{r['search_qps']:,.0f} | {r['setting']} |"
        )
    return "\n".join(lines) + "\n"


def filtered_table(ds, rs):
    groups = defaultdict(list)
    for r in rs:
        if r["phase"] == "point":
            groups[r["engine"]].append(r)
    if not groups:
        return ""
    lines = [
        f"### {ds}: filtered search",
        "",
        "Queries/s (batch 1,000, the exact predicate check included) for the 10 nearest items satisfying each",
        "query's tags, at recall@10 ≥ each threshold, each engine filtering itself. exact is GPU brute force",
        "over the admitted items, the ceiling. \"below floor\": even the cheapest setting answers fewer than",
        "1,000 queries/s.",
        "",
        "| engine | q/s @0.9 | q/s @0.95 | q/s @0.99 | max recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for e, g in sorted(groups.items(), key=lambda kv: -(fastest(kv[1], 0.9) or 0)):
        scored = [r for r in g if "recall10" in r]
        top = num(max(r["recall10"] for r in scored), ".3f") if scored else "below floor"
        lines.append(f"| {e} | " + " | ".join(num(fastest(scored, t)) for t in TARGETS) + f" | {top} |")
    return "\n".join(lines) + "\n"


def latency_table(ds, rs):
    groups = defaultdict(list)
    for r in rs:
        if r["phase"] == "point":
            groups[r["engine"]].append(r)
    if not groups:
        return ""
    lines = [
        f"### {ds}: one query a call",
        "",
        "ann-benchmarks' default mode: each of 1,000 queries searched alone, a call each (host memory to",
        "host memory). Queries/s is one over the mean call, at recall@10 ≥ each threshold, with that",
        "point's median and 99th-percentile call.",
        "",
        "| engine | q/s @0.9 | p50 / p99 ms @0.9 | q/s @0.95 | q/s @0.99 | p50 / p99 ms @0.99 | max recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def best(g, t):
        ok = [r for r in g if r["recall10"] >= t]
        return max(ok, key=lambda r: r["qps"]) if ok else None

    def ms(r):
        return f"{r['p50_ms']:.3f} / {r['p99_ms']:.3f}" if r else "—"

    for e, g in sorted(groups.items(), key=lambda kv: -(fastest(kv[1], 0.9) or 0)):
        b9, b99 = best(g, 0.9), best(g, 0.99)
        lines.append(
            f"| {e} | {num(fastest(g, 0.9))} | {ms(b9)} | {num(fastest(g, 0.95))} | {num(fastest(g, 0.99))} | {ms(b99)} | "
            f"{num(max(r['recall10'] for r in g), '.3f')} |"
        )
    return "\n".join(lines) + "\n"


def times_table(run):
    lines = ["### Run time", "", "| step | dataset | engine | minutes | status |", "|---|---|---|---:|---|"]
    total = 0.0
    for line in open(os.path.join(R_DIR, run, "times.tsv")):
        cmd, status, secs = line.rstrip("\n").split("\t")
        f = cmd.split()
        script = f.pop(0)[: -len(".py")]
        total += float(secs)
        lines.append(f"| {script} | {(f + [''])[0]} | {' '.join(f[1:])} | {float(secs) / 60:.1f} | {status} |")
    lines.append(f"| all | | | {total / 60:.1f} | |")
    return "\n".join(lines) + "\n"


def machine():
    def sh(c):
        return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

    cpu = sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-")
    gpu = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
    return f"{cpu.strip()}; {gpu}; Linux {platform.release()}"


def latest():
    """Every dataset's sections from the newest full run (every engine;
    run.sh marks it `full`) that measured it: the standing report across
    invocations, which run.sh writes to REPORT.md."""
    runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(R_DIR, "*", "full")))
    newest = {}
    for run in runs:  # oldest first, so the newest run wins
        for harness in ("curves", "stream", "latency", "filtered"):
            for ds in datasets(run, harness):
                newest[(harness, ds)] = run
    print("# ANN benchmark\n")
    print(f"Machine: {machine()}. Each section is from the newest run that measured it (results/<run>/REPORT.md")
    print("holds that run's own report and run time). Nothing is combined across runs within a table.\n")
    for (harness, ds), run in sorted(newest.items(), key=lambda kv: (["curves", "stream", "latency", "filtered"].index(kv[0][0]), kv[0][1])):
        idle_w = idle(run).get("gpu_w")
        print(f"_run {run}_\n")
        if harness == "curves":
            print(curves_table(ds, rows(run, harness, ds), idle_w))
        elif harness == "stream":
            print(stream_table(ds, rows(run, harness, ds)))
        elif harness == "latency":
            print(latency_table(ds, rows(run, harness, ds)))
        else:
            print(filtered_table(ds, rows(run, harness, ds)))


def main():
    if sys.argv[1:] == ["--latest"]:
        return latest()
    run = sys.argv[1] if len(sys.argv) > 1 else newest_run()
    i = idle(run)
    idle_w = i.get("gpu_w")
    print(f"# ANN benchmark: run {run}\n")
    print(f"Machine: {machine()}. GPU idle power at the start of the run: " + (f"{idle_w:.1f} W." if idle_w is not None else "not measured."))
    if i.get("cpu_pkg_w") is None:
        print(
            "\nCPU package energy (RAPL) is not readable on this machine, so energy covers the GPU only. A CPU engine's"
            "\nfigures are the idle GPU beside it, not its own cost: compare energy only between GPU engines."
        )
    print()
    print(times_table(run))
    for ds in datasets(run, "curves"):
        print(curves_table(ds, rows(run, "curves", ds), idle_w))
    for ds in datasets(run, "stream"):
        print(stream_table(ds, rows(run, "stream", ds)))
    for ds in datasets(run, "latency"):
        print(latency_table(ds, rows(run, "latency", ds)))
    for ds in datasets(run, "filtered"):
        print(filtered_table(ds, rows(run, "filtered", ds)))

TRACK = {"curves": "ann-benchmarks", "stream": "big-ann streaming", "latency": "ann-benchmarks, one query a call",
         "filtered": "big-ann filtered"}


def newest():
    runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(R_DIR, "*", "full")))
    found = {}
    for run in runs:
        for harness in TRACK:
            for ds in datasets(run, harness):
                found[(harness, ds)] = run
    return found


def headline():
    """The ANN suite's headline rows for the top-level report (../suite.py).
    
    Each row is one measure of one workload for one engine, taken from the newest
    full run that measured it (as REPORT.md's sections are): the curves' queries/s
    at recall@10 0.9 and 0.99, the build's seconds, the streaming runbook's mean
    recall and seconds, one query a call's queries/s at 0.9 and its 99th
    percentile, and the filtered track's queries/s at 0.9."""
    out = []

    def row(ds, harness, measure, unit, higher, engine, value, run):
        if value is not None:
            out.append(dict(suite="ann", workload=f"{ds} ({TRACK[harness]})", measure=measure, unit=unit,
                            higher=higher, engine=engine, value=value, base=engine == "base", run=run))

    for (harness, ds), run in sorted(newest().items()):
        rs = rows(run, harness, ds)
        engines = sorted({r["engine"] for r in rs})
        for e in engines:
            pts = [r for r in rs if r["engine"] == e and r.get("phase") == "point"]
            if harness in ("curves", "filtered", "latency"):
                row(ds, harness, "q/s @ recall 0.9", "q/s", True, e, fastest(pts, 0.9), run)
            if harness in ("curves", "latency"):
                row(ds, harness, "q/s @ recall 0.99", "q/s", True, e, fastest(pts, 0.99), run)
            if harness == "curves":
                b = [r["seconds"] for r in rs if r["engine"] == e and r.get("phase") == "build"]
                row(ds, harness, "build", "s", False, e, b[0] if b else None, run)
            if harness == "latency":
                ok = [r for r in pts if r["recall10"] >= 0.9]
                row(ds, harness, "p99 call @ recall 0.9", "ms", False, e,
                    max(ok, key=lambda r: r["qps"])["p99_ms"] if ok else None, run)
            if harness == "stream":
                rb = [r for r in rs if r["engine"] == e and r.get("phase") == "runbook"]
                if rb:
                    row(ds, harness, "mean recall@10", "", True, e, rb[0]["mean_recall10"], run)
                    row(ds, harness, "runbook", "s", False, e, rb[0]["seconds"], run)
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--headline"]:
        for row in headline():
            print(json.dumps(row))
    else:
        main()
