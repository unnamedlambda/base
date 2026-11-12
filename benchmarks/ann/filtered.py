"""Filtered search: the 10 nearest items that satisfy a query's predicate, the
way big-ann-benchmarks' filtered track measures it: each engine's own filter.

    python filtered.py yfcc-1m <engine>     # appends to <run dir>/filtered-yfcc-1m.jsonl

The data is the track's YFCC (prepare.py, yfcc-1m): CLIP image embeddings,
each item holding about 11 tags out of 200,386, and queries each requiring 1
or 2 tags (a conjunction), so selectivity runs from common tags down to a
handful of items in a million. The engine is given the items' tags before
its build and filters itself; its answers are checked exactly against the
predicate before they are returned, and that check is inside the timed work.

Each setting is scored by recall@10 (tie-aware: common.recall) against the
exact top 10 among the items the predicate admits, and timed at batch BATCH
for at least TIMED_S. Settings run cheapest first. A setting's first batch is
probed before the full pass, and a sweep stops at a setting the engine
refuses, below FLOOR queries/s (a slower point takes over 10 s per pass of
the 10,000 queries), or once recall has saturated (curves.SATURATED, which the
report prints as 1.000).
"""

import sys

import numpy as np

from common import Dataset, METER, Out, distances, harness, recall, refused, timed
from curves import SATURATED
from engines import make
from prepare import TagIndex

BATCH = 1000
TIMED_S = 0.5
FLOOR = 1000


def main():
    name, engine = sys.argv[1], sys.argv[2]
    ds = Dataset(name)
    n = len(ds.base)
    indptr, tags = ds.load("tags_indptr.npy"), ds.load("tags.npy")
    tix = TagIndex(indptr, tags, n)
    qptr, qtags = ds.load("query_tags_indptr.npy"), ds.load("query_tags.npy")
    required = [qtags[qptr[i] : qptr[i + 1]] for i in range(len(ds.queries))]
    gt, gt_dist = ds.load("gt_filtered.npy"), ds.load("gt_filtered_dist.npy")
    emit = Out("filtered", name, engine)

    eng = make(engine, ds.metric, ds.base.shape[1], n)
    ids = np.arange(n, dtype=np.int64)
    start = METER.read()
    secs, _ = timed(lambda: (eng.set_tags(indptr, tags), eng.insert(np.ascontiguousarray(ds.base[:]), ids),
                             getattr(eng, "wait_indexed", lambda: None)()))
    emit(phase="build", seconds=secs, energy=METER.since(start))

    def search(s, nq=len(ds.queries)):
        """The 10 answers per query, of the first nq, that pass the predicate; -1 padded."""
        out = np.full((nq, 10), -1, np.int64)
        for b in range(0, nq, BATCH):
            q = ds.queries[b : b + BATCH]
            cand = eng.search_filtered(q, s, 10, required[b : b + BATCH])
            for j in range(len(q)):
                c = cand[j]
                c = c[tix.keep(c, required[b + j])][:10]
                out[b + j, : len(c)] = c
        return out

    for s in eng.settings_for(10):
        try:
            search(s, BATCH)  # warm: first calls compile and allocate
            probe = BATCH / timed(lambda: search(s, BATCH))[0]
        except Exception as e:
            refused(f"setting {s}", e)
            break
        if probe < FLOOR:
            emit(phase="point", setting=s, qps=probe, below_floor=True)
            break
        times = []
        start = METER.read()
        while not times or sum(times) < TIMED_S:
            secs, found = timed(lambda: search(s))
            times.append(secs)
        energy = METER.since(start)
        qps = len(ds.queries) / float(np.median(times))
        r = recall(found, gt, gt_dist, distances(ds.base, ds.queries, found, ds.metric))
        emit(phase="point", setting=s, qps=qps, recall10=r, energy=energy)
        if qps < FLOOR or r >= SATURATED:
            break


if __name__ == "__main__":
    harness(main)
