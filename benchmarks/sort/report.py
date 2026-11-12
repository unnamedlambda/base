"""The sort suite's report: REPORT.md for one run, and the headline rows for
the top-level report (../suite.py).

    python report.py [run]      # default: the newest run under results/
"""

import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R_DIR = os.path.join(HERE, "results")


def runs(full=False):
    marker = "full" if full else "times.tsv"
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(R_DIR, "*", marker)))


def rows(run, kind):
    p = os.path.join(R_DIR, run, f"{kind}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def spread(r):
    s = sorted(r["samples"])
    return f"{s[0] * 1e3:.3f}-{s[-1] * 1e3:.3f}"


def table(title, head, rs, key):
    by = defaultdict(dict)
    for r in rs:
        by[key(r)][r["engine"]] = r
    engines = sorted({r["engine"] for r in rs})
    out = [f"### {title}", "", f"| {head} | " + " | ".join(f"{e} Gkeys/s (ms range)" for e in engines) + " |",
           "|---|" + "---:|" * len(engines)]
    for k, es in by.items():
        out.append(f"| {k} | " + " | ".join(
            f"{es[e]['keys_per_s'] / 1e9:.2f} ({spread(es[e])})" if e in es else "—" for e in engines) + " |")
    return "\n".join(out) + "\n"


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else runs()[-1]
    print(f"# Sort and top-k: run {run}\n")
    print("Keys a second at the median of the runs, on data already on the device; the range is the fastest")
    print("and slowest run. Every answer is checked against numpy before its time counts.\n")
    print(table("sort: (u32 key, u32 value) pairs, stable", "keys, distribution", rows(run, "sort"),
                lambda r: f"{r['n']:,}, {r['dist']}"))
    print(table("top-k: the k smallest of each row", "rows x columns, k", rows(run, "topk"),
                lambda r: f"{r['rows']:,} x {r['cols']:,}, k={r['k']}"))


def headline():
    """Every engine's keys a second on each sort and top-k measure, from the
    newest run that ran every engine, with its fastest and slowest run."""
    full = runs(full=True)
    if not full:
        return []
    run = full[-1]
    out = []
    for kind, wl in (("sort", lambda r: f"sort {r['n']:,} pairs, {r['dist']}"),
                     ("topk", lambda r: f"top-{r['k']} of {r['rows']:,} x {r['cols']:,}")):
        for r in rows(run, kind):
            keys = r["n"] if kind == "sort" else r["rows"] * r["cols"]
            s = sorted(r["samples"])
            out.append(dict(suite="sort", workload=wl(r), measure="keys/s", unit="keys/s", higher=True,
                            engine=r["engine"], value=r["keys_per_s"], base=r["engine"] == "base", run=run,
                            spread=(keys / s[-1], keys / s[0])))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--headline"]:
        for row in headline():
            print(json.dumps(row))
    else:
        main()
