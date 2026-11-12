"""The JSON suite's report: REPORT.md for one run, and (with `--headline`)
the headline rows for the top-level report (../suite.py), as JSON lines.

    python report.py [run]      # default: the newest run under results/
"""

import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R_DIR = os.path.join(HERE, "results")


def runs():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(R_DIR, "*", "parse.jsonl")))


def rows(run):
    return [json.loads(line) for line in open(os.path.join(R_DIR, run, "parse.jsonl"))]


def gbps(r, t):
    return r["bytes"] / t / 1e9


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else runs()[-1]
    rs = rows(run)
    engines = sorted({r["engine"] for r in rs})
    by = defaultdict(dict)
    for r in rs:
        by[r["file"]][r["engine"]] = r
    print(f"# JSON parsing: run {run}\n")
    print("Text parsed a second into each engine's document (simdjson's DOM, base's tape), at the median")
    print("of the timed parses (the range is the slowest and fastest), one core. Every file's checksum over")
    print("every value is checked equal between the engines before its times count.\n")
    print("| file | bytes | " + " | ".join(f"{e} GB/s (range)" for e in engines) + " |")
    print("|---|---:|" + "---:|" * len(engines))
    for f, es in by.items():
        cells = []
        for e in engines:
            r = es.get(e)
            if r:
                s = sorted(r["samples"])
                cells.append(f"{gbps(r, r['seconds']):.2f} ({gbps(r, s[-1]):.2f}-{gbps(r, s[0]):.2f})")
            else:
                cells.append("—")
        print(f"| {f} | {next(iter(es.values()))['bytes']:,} | " + " | ".join(cells) + " |")


def headline():
    """Every engine's GB/s on each file, from the newest run, with its
    slowest and fastest parse."""
    r_ = runs()
    if not r_:
        return []
    run = r_[-1]
    out = []
    for r in rows(run):
        s = sorted(r["samples"])
        out.append(dict(suite="json", workload=f"parse {r['file']}", measure="GB/s", unit="GB/s", higher=True,
                        engine=r["engine"], value=gbps(r, r["seconds"]), base=r["engine"] == "base", run=run,
                        spread=(gbps(r, s[-1]), gbps(r, s[0]))))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--headline"]:
        for row in headline():
            print(json.dumps(row))
    else:
        main()
