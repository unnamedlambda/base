"""The hash table suite's report: REPORT.md for one run, and the headline rows
for the top-level report (../suite.py).

    python report.py [run]      # default: the newest run under results/
"""

import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R_DIR = os.path.join(HERE, "results")
OPS = {"build": "insert into an empty table", "hit": "lookup, present, random order",
       "miss": "lookup, absent", "zipf": "lookup, Zipfian 0.99"}


def runs():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(R_DIR, "*", "ops.jsonl")))


def rows(run):
    return [json.loads(l) for l in open(os.path.join(R_DIR, run, "ops.jsonl"))]


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else runs()[-1]
    rs = rows(run)
    engines = sorted({r["engine"] for r in rs})
    by = defaultdict(dict)
    for r in rs:
        by[(r["op"], r["n"])][r["engine"]] = r
    print(f"# Hash tables: run {run}\n")
    print("u64 keys to u64 values. Nanoseconds an operation at the median of the samples (the range is the")
    print("fastest and slowest sample's), one core; every lookup pass's answer checked across engines.\n")
    print("| operation | entries | " + " | ".join(f"{e} ns (range)" for e in engines) + " |")
    print("|---|---:|" + "---:|" * len(engines))
    for (op, n), es in by.items():
        cells = []
        for e in engines:
            r = es.get(e)
            if r:
                s = sorted(r["samples"])
                cells.append(f"{r['ns_per_op']:.2f} ({s[0] / n * 1e9:.2f}-{s[-1] / n * 1e9:.2f})")
            else:
                cells.append("—")
        print(f"| {OPS[op]} | {n:,} | " + " | ".join(cells) + " |")


def headline():
    """Every engine's nanoseconds an operation on each operation and size,
    from the newest run, with its fastest and slowest sample."""
    r_ = runs()
    if not r_:
        return []
    run = r_[-1]
    out = []
    for r in rows(run):
        s = sorted(r["samples"])
        out.append(dict(suite="hashtable", workload=f"{OPS[r['op']]}, {r['n']:,} entries", measure="time an op",
                        unit="ns", higher=False, engine=r["engine"], value=r["ns_per_op"], base=r["engine"] == "base",
                        run=run, spread=(s[-1] / r["n"] * 1e9, s[0] / r["n"] * 1e9)))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--headline"]:
        for row in headline():
            print(json.dumps(row))
    else:
        main()
