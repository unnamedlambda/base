"""Recall against speed: one engine on one dataset, the way ann-benchmarks and
big-ann-benchmarks measure it.

    python curves.py <dataset> <engine>      # appends to <run dir>/curves-<dataset>.jsonl

1. build: the engine takes every item in one insert. Seconds, energy, and the
   device and host bytes it added per item.
2. sweep: the engine's settings run cheapest first. Each is run once (its
   answers scored by recall@10, tie-aware: common.recall), then timed at
   batches of BATCH queries (host memory to host memory) for at least TIMED_S.
   A sweep runs the engine's whole grid, as ann-benchmarks does, but stops at
   a setting the engine refuses, below FLOOR queries/s, or once recall has
   saturated (SATURATED, which the report prints as 1.000).
3. power: the fastest setting reaching TARGET (big-ann-benchmarks' recall
   threshold) runs again for POWER_S, long enough for the GPU's energy
   counter (it advances every 100 ms): joules per query at that point.

`tiny` checks the harness: one timed run per point, no power window. Its
numbers are not reported.
"""

import gc
import sys

import numpy as np

from common import Dataset, METER, Out, distances, harness, recall, refused, timed
from engines import device_held, make

BATCH = 1000
TIMED_S = 0.5
FLOOR = 100
SATURATED = 0.9995
TARGET = 0.9
POWER_S = 2.0


def rss(pid: int = 0) -> int:
    with open(f"/proc/{pid or 'self'}/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) * 1024
    return 0


def answer(eng, queries, setting):
    return np.concatenate([eng.search(queries[b : b + BATCH], setting, 10) for b in range(0, len(queries), BATCH)])


def timed_window(eng, queries, setting, seconds):
    """(median seconds per pass over the queries, energy over the window)."""
    times = []
    start = METER.read()
    while not times or sum(times) < seconds:
        times.append(timed(lambda: answer(eng, queries, setting))[0])
    energy = METER.since(start)
    energy["queries"] = len(queries) * len(times)
    return float(np.median(times)), energy


def main():
    name, engine = sys.argv[1], sys.argv[2]
    check = name == "tiny"
    ds = Dataset(name)
    gt, gt_dist = ds.load("gt.npy"), ds.load("gt_dist.npy")
    n, dim = ds.base.shape
    emit = Out("curves", name, engine)

    eng = make(engine, ds.metric, dim, n)
    dev0, host0 = device_held() or 0, rss()

    def build():
        eng.insert(np.ascontiguousarray(ds.base[:]), np.arange(n, dtype=np.int64))
        if hasattr(eng, "wait_indexed"):
            eng.wait_indexed()

    start = METER.read()
    secs, _ = timed(build)
    energy = METER.since(start)
    gc.collect()
    server = rss(eng.proc.pid) if hasattr(eng, "proc") else 0
    emit(
        phase="build",
        items=n,
        seconds=secs,
        energy=energy,
        device_bytes_per_item=((device_held() or 0) - dev0) / n,
        host_bytes_per_item=max(0, rss() - host0 + server) / n,
    )

    chosen = None
    for s in eng.settings_for(10):
        try:
            found = answer(eng, ds.queries, s)
        except Exception as e:  # a setting past what the engine accepts ends its sweep
            refused(f"setting {s}", e)
            break
        r = recall(found, gt, gt_dist, distances(ds.base, ds.queries, found, ds.metric))
        per_pass, energy = timed_window(eng, ds.queries, s, 0.0 if check else TIMED_S)
        qps = len(ds.queries) / per_pass
        emit(phase="point", setting=s, qps=qps, recall10=r, energy=energy)
        if r >= TARGET and (chosen is None or qps > chosen[1]):
            chosen = (s, qps, r)
        if qps < FLOOR or r >= SATURATED:
            break

    if chosen and not check:
        s, _, r = chosen
        per_pass, energy = timed_window(eng, ds.queries, s, POWER_S)
        emit(phase="power", setting=s, qps=len(ds.queries) / per_pass, recall10=r, energy=energy)


if __name__ == "__main__":
    harness(main)
