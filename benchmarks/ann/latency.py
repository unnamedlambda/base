"""Latency: one engine answering one query at a time, the way ann-benchmarks
runs by default.

    python latency.py <dataset> <engine>     # appends to <run dir>/latency-<dataset>.jsonl

The engine takes every item in one insert, then its settings run cheapest
first, as curves.py runs them; at each, QUERIES queries are searched one call
a query (host memory to host memory), each call timed. A point is the
setting's recall@10 over those queries, its queries/s (one over the mean
call), and the median and 99th-percentile call. The sweep stops below FLOOR
queries/s or once recall has saturated, as curves.py's does.

`tiny` checks the harness; its numbers are not reported.
"""

import sys
import time

import numpy as np

from common import Dataset, Out, distances, harness, recall, refused
from curves import FLOOR, SATURATED
from engines import make

QUERIES = 1000


def main():
    name, engine = sys.argv[1], sys.argv[2]
    ds = Dataset(name)
    nq = min(QUERIES, len(ds.queries))
    queries = np.ascontiguousarray(ds.queries[:nq])
    gt, gt_dist = ds.load("gt.npy")[:nq], ds.load("gt_dist.npy")[:nq]
    n, dim = ds.base.shape
    emit = Out("latency", name, engine)

    eng = make(engine, ds.metric, dim, n)
    eng.insert(np.ascontiguousarray(ds.base[:]), np.arange(n, dtype=np.int64))
    if hasattr(eng, "wait_indexed"):
        eng.wait_indexed()

    for s in eng.settings_for(10):
        try:
            eng.search(queries[:1], s, 10)  # a first call outside the timing
        except Exception as e:
            refused(f"setting {s}", e)
            break
        found = np.empty((nq, 10), np.int64)
        calls = np.empty(nq)
        for i in range(nq):
            t = time.perf_counter()
            found[i] = eng.search(queries[i : i + 1], s, 10)[0]
            calls[i] = time.perf_counter() - t
        r = recall(found, gt, gt_dist, distances(ds.base, queries, found, ds.metric))
        qps = 1.0 / float(np.mean(calls))
        emit(phase="point", setting=s, qps=qps, recall10=r,
             p50_ms=float(np.median(calls)) * 1e3, p99_ms=float(np.percentile(calls, 99)) * 1e3)
        if qps < FLOOR or r >= SATURATED:
            break


if __name__ == "__main__":
    harness(main)
