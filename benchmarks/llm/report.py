"""The LLM suite's report: REPORT.md for one run, the headline rows for the
top-level report (../suite.py), and the offload setting run.sh measures at.

    python report.py [run]                     # default: the newest run
    python report.py --best-ncmoe <sweep.jsonl>  # llama.cpp's fastest decode
"""

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R_DIR = os.path.join(HERE, "results")


def runs():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(R_DIR, "*", "llama.jsonl")))


def llama(run):
    """(depth, prompt or decode) -> (tokens/s, stddev, ncmoe)."""
    out = {}
    for line in open(os.path.join(R_DIR, run, "llama.jsonl")):
        r = json.loads(line)
        kind = "decode" if r["n_gen"] > 0 else "prompt"
        out[(r["n_depth"], kind)] = (r["avg_ts"], r["stddev_ts"], r.get("n_cpu_moe"))
    return out


def base(run):
    """(depth, prompt or decode) -> tokens/s, from bench.py's lines."""
    out = {}
    for p in glob.glob(os.path.join(R_DIR, run, "base-*.txt")):
        d = int(re.search(r"base-(\d+)\.txt", p).group(1))
        text = open(p).read()
        for kind, pat in (("prompt", r"prompt, \d+ tokens\s+[\d.]+ s\s+([\d.]+) tok/s"),
                          ("decode", r"generate, \d+ tokens\s+[\d.]+ s\s+([\d.]+) tok/s")):
            m = re.search(pat, text)
            if m:
                out[(d, kind)] = float(m.group(1))
    return out


def best_ncmoe(sweep):
    """The setting with the most decode tokens a second, of those that ran."""
    best = None
    for line in open(sweep):
        r = json.loads(line)
        if r["n_gen"] > 0 and (best is None or r["avg_ts"] > best[0]):
            best = (r["avg_ts"], r["n_cpu_moe"])
    if best is None:
        raise SystemExit(f"no setting in {sweep} ran")
    return best[1]


def main():
    if sys.argv[1:2] == ["--best-ncmoe"]:
        print(best_ncmoe(sys.argv[2]))
        return
    run = sys.argv[1] if len(sys.argv) > 1 else runs()[-1]
    L, B = llama(run), base(run)
    print(f"# LLM decode: run {run}\n")
    print("gpt-oss-20b (MXFP4), one sequence, experts offloaded to host memory as each engine does it on")
    print("a 12 GB card. Tokens a second; llama.cpp's are llama-bench's mean over 3 runs (± its standard")
    print("deviation), at its fastest offload setting (`-ncmoe`, the layers whose experts stay on the CPU).")
    print("Base's prompt is fed a token at a time (it has no batched prefill yet), so its prompt rate is")
    print("its decode rate.\n")
    print("| context | measure | llama.cpp | base |")
    print("|---:|---|---:|---:|")
    for d, kind in sorted(set(L) | set(B)):
        l = L.get((d, kind))
        lc = f"{l[0]:.1f} ± {l[1]:.1f} (ncmoe {l[2]})" if l else "—"
        b = B.get((d, kind))
        print(f"| {d:,} | {kind} | {lc} | {f'{b:.1f}' if b else '—'} |")


def headline():
    """Tokens a second of each engine, decode and prompt, at each context."""
    r_ = runs()
    if not r_:
        return []
    run = r_[-1]
    out = []
    for (d, kind), (ts, sd, _) in llama(run).items():
        out.append(dict(suite="llm", workload=f"gpt-oss-20b {kind}, context {d:,}", measure="tokens/s",
                        unit="tok/s", higher=True, engine="llama.cpp", value=ts, base=False, run=run,
                        spread=(ts - sd, ts + sd)))
    for (d, kind), ts in base(run).items():
        out.append(dict(suite="llm", workload=f"gpt-oss-20b {kind}, context {d:,}", measure="tokens/s",
                        unit="tok/s", higher=True, engine="base", value=ts, base=True, run=run))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--headline"]:
        for row in headline():
            print(json.dumps(row))
    else:
        main()
