"""llvm-libc's memory benchmark, every function x distribution x implementation.

    python bench.py [filter]      # appends to <run dir>/memops.jsonl

One process of the harness's Google Benchmark target: for each function
(memcpy, memmove, memset, bzero, memcmp, bcmp), each of Google's published
size distributions for it, and each implementation (Implementations.cpp),
calls with sizes drawn from the distribution and random offsets, all within
L1 (the harness's design: the size distribution is the variable). REPS
repetitions; the median is kept. `filter` is a Google Benchmark regex (the
check passes one, for a quick run).
"""

import json
import os
import subprocess
import sys

from prepare import EXE

REPS = 5


def run_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", os.environ.get("MEMOPS_RUN", "adhoc"))
    os.makedirs(d, exist_ok=True)
    return d


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else "."
    quick = filt != "."
    cmd = [EXE, "--benchmark_format=json", f"--benchmark_filter={filt}",
           f"--benchmark_repetitions={1 if quick else REPS}", "--benchmark_report_aggregates_only=true"]
    if quick:
        cmd.append("--benchmark_min_time=0.01s")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-3000:])
        raise SystemExit(f"the benchmark exited {p.returncode}")
    out = json.loads(p.stdout)
    rows = []
    for b in out["benchmarks"]:
        if quick or b.get("aggregate_name") == "median":
            impl, dist = b["label"].split(",", 1)
            rows.append({
                "function": b["run_name"].split("/")[0].removeprefix("BM_").lower(),
                "distribution": dist,
                "impl": impl,
                "ns_per_call": b["cpu_time"] if b["time_unit"] == "ns" else None,
                "bytes_per_cycle": b.get("bytes_per_cycle"),
                "GBps": b.get("bytes_per_second", 0) / 1e9,
            })
    if not rows:
        raise SystemExit("the benchmark reported nothing")
    with open(os.path.join(run_dir(), "memops.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps({"quick": quick, **r}) + "\n")
    with open(os.path.join(run_dir(), "context.json"), "w") as f:
        json.dump(out["context"], f)
    print(f"{len(rows)} results", flush=True)


if __name__ == "__main__":
    main()
