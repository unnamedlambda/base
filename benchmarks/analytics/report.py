"""The analytics suite's report: REPORT.md for one run, and (with
`--headline`) the headline rows for the top-level report (../suite.py), as
JSON lines.

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
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(R_DIR, "*", "queries.jsonl")))


def rows(run):
    return [json.loads(line) for line in open(os.path.join(R_DIR, run, "queries.jsonl"))]


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else runs()[-1]
    rs = rows(run)
    engines = sorted({r["engine"] for r in rs})
    by = defaultdict(dict)
    for r in rs:
        by[r["query"]][r["engine"]] = r
    print(f"# Analytics: run {run}\n")
    print("ClickBench's numeric queries on the first 10 million rows of `hits`, hot: milliseconds at the")
    print("median of the timed runs (the range is the fastest and slowest). Base's answers are checked")
    print("against DuckDB's before its times count.\n")
    print("| query | " + " | ".join(f"{e} ms (range)" for e in engines) + " | SQL |")
    print("|---|" + "---:|" * len(engines) + "---|")
    for q, es in by.items():
        cells = []
        for e in engines:
            r = es.get(e)
            s = sorted(r["samples"]) if r else None
            cells.append(f"{r['seconds'] * 1e3:.3f} ({s[0] * 1e3:.3f}-{s[-1] * 1e3:.3f})" if r else "—")
        print(f"| {q} | " + " | ".join(cells) + f" | `{next(iter(es.values()))['sql']}` |")


def headline():
    """Every engine's milliseconds on each query, from the newest run, with
    its fastest and slowest run."""
    r_ = runs()
    if not r_:
        return []
    run = r_[-1]
    out = []
    for r in rows(run):
        s = sorted(r["samples"])
        out.append(dict(suite="analytics", workload=f"ClickBench {r['query']}, 10M rows", measure="time, hot",
                        unit="ms", higher=False, engine=r["engine"], value=r["seconds"] * 1e3,
                        base=r["engine"] == "base", run=run, spread=(s[-1] * 1e3, s[0] * 1e3)))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--headline"]:
        for row in headline():
            print(json.dumps(row))
    else:
        main()
