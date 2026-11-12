"""The protobuf suite's report: REPORT.md for one run, and (with
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
ENGINES = ("base", "protobuf-arena", "protobuf", "upb")


def runs():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(R_DIR, "*", "proto.jsonl")))


def rows(run):
    return [json.loads(line) for line in open(os.path.join(R_DIR, run, "proto.jsonl"))]


def gbps(r, t):
    return r["bytes"] / t / 1e9


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else runs()[-1]
    rs = rows(run)
    by = defaultdict(dict)
    for r in rs:
        by[(r["op"], r["bench"])][r["engine"]] = r
    print(f"# Protobuf: run {run}\n")
    print("HyperProtoBench's six benchmarks, each benchmark's top-level messages built by its own")
    print("generated Set functions and serialized by protobuf C++: the bytes of wire format parsed (into")
    print("each engine's messages) or serialized a second, at the median of the timed passes (the range")
    print("is the slowest and fastest), one core. Before any time counts, upb and protobuf round-trip")
    print("every message, and base decodes the whole corpus and encodes it back byte for byte, which")
    print("protobuf reads as the same messages.\n")
    for op in ("parse", "serialize"):
        print(f"### {op}\n")
        print("| benchmark | bytes | " + " | ".join(f"{e} GB/s (range)" for e in ENGINES) + " |")
        print("|---|---:|" + "---:|" * len(ENGINES))
        for (o, b), es in sorted(by.items()):
            if o != op:
                continue
            cells = []
            for e in ENGINES:
                r = es.get(e)
                if r:
                    s = sorted(r["samples"])
                    cells.append(f"{gbps(r, r['seconds']):.1f} ({gbps(r, s[-1]):.1f}-{gbps(r, s[0]):.1f})")
                else:
                    cells.append("—")
            print(f"| {b} | {next(iter(es.values()))['bytes']:,} | " + " | ".join(cells) + " |")
        print()


def headline():
    """Every engine's GB/s on each benchmark and operation, from the newest
    run, with its slowest and fastest pass."""
    r_ = runs()
    if not r_:
        return []
    run = r_[-1]
    out = []
    for r in rows(run):
        s = sorted(r["samples"])
        out.append(dict(suite="protobuf", workload=r["bench"], measure=r["op"], unit="GB/s", higher=True,
                        engine=r["engine"], value=gbps(r, r["seconds"]), base=r["engine"] == "base", run=run,
                        spread=[gbps(r, s[-1]), gbps(r, s[0])]))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--headline"]:
        for row in headline():
            print(json.dumps(row))
    else:
        main()
