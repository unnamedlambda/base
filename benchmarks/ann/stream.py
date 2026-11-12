"""Streaming: one engine runs a runbook of inserts, deletes and searches, the
way big-ann-benchmarks' streaming track measures it.

    python stream.py <dataset> <engine>      # appends to <run dir>/stream-<dataset>.jsonl

The runbook (prepare.runbook) slides a window of WINDOW k-means clusters
across the dataset: each step inserts the next cluster and deletes the oldest,
and every few steps all queries are searched and scored by recall@10 against
the exact top 10 over the live set at that moment. The distribution moves
under the index and deleted space must be reused.

The engine searches with one setting throughout, as a streaming submission
does: the cheapest that reaches TARGET on the first window. The headline is
big-ann-benchmarks': the mean recall over the runbook's searches, and the
runbook's total time. Insert, delete and search times are also given apart,
and the time an engine takes to finish indexing before a search
(`wait_indexed`, where it has one): an engine may index as it inserts and
deletes, or defer that work to the wait, and the runbook's time holds both.
"""

import sys
import time

import numpy as np

from common import Dataset, METER, Out, distances, harness, recall, timed
from curves import TARGET, answer
from engines import make
from prepare import runbook


def main():
    name, engine = sys.argv[1], sys.argv[2]
    ds = Dataset(name)
    order, bounds = ds.load("order.npy"), ds.load("bounds.npy")
    gts, gt_dists = ds.load("stream_gt.npy"), ds.load("stream_gt_dist.npy")
    n, dim = ds.base.shape
    emit = Out("stream", name, engine)
    eng = make(engine, ds.metric, dim, n)

    def rows(c):
        return np.sort(order[bounds[c] : bounds[c + 1]]).astype(np.int64)

    def score(found, i):
        return recall(found, gts[i], gt_dists[i], distances(ds.base, ds.queries, found, ds.metric))

    spent = {"insert": 0.0, "delete": 0.0, "search": 0.0}
    indexing = 0.0
    items = {"insert": 0, "delete": 0, "search": 0}
    recalls, setting, searches = [], None, 0
    start = METER.read()
    t0 = time.perf_counter()
    for op, c in runbook():
        if op == "insert":
            r = rows(c)
            vecs = np.ascontiguousarray(ds.base[r])
            secs, _ = timed(lambda: eng.insert(vecs, r))
        elif op == "delete":
            r = rows(c)
            secs, _ = timed(lambda: eng.delete(r))
        else:
            if hasattr(eng, "wait_indexed"):
                w, _ = timed(eng.wait_indexed)
                indexing += w
            if setting is None:  # the first window picks the setting, untimed
                for s in eng.settings_for(10):
                    setting = s
                    if score(answer(eng, ds.queries, s), searches) >= TARGET:
                        break
            secs, found = timed(lambda: answer(eng, ds.queries, setting))
            recalls.append(score(found, searches))
            emit(phase="search", step=searches, setting=setting, recall10=recalls[-1], qps=len(ds.queries) / secs)
            searches += 1
            r = ds.queries
        spent[op] += secs
        items[op] += len(r)
    wall = time.perf_counter() - t0
    emit(
        phase="runbook",
        setting=setting,
        mean_recall10=float(np.mean(recalls)),
        min_recall10=float(np.min(recalls)),
        seconds=wall,
        insert_per_s=items["insert"] / spent["insert"],
        delete_per_s=items["delete"] / spent["delete"],
        search_qps=items["search"] / spent["search"],
        indexing_seconds=indexing,
        energy=METER.since(start),
    )


if __name__ == "__main__":
    harness(main)
