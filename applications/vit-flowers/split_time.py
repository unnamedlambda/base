#!/usr/bin/env python3
"""Where the step's time goes, split by who performs the launch.

There is no working profiler here, so the split is measured by capturing each
class of launch as its own graph: `replayBlas` records only the contractions and
`replayRow` only the proven kernels. What each graph computes is meaningless —
the other half never ran — but what it *takes* is real, since the launches,
shapes, streams and dependence edges are the ones the full step uses.

The two do not sum to the whole: run together they overlap across the sixteen
streams. That gap is the point of the third number.
"""
import os
import sys, time, numpy as np, py_base
import entries

ART, D, N = sys.argv[1], sys.argv[2], 50
blob = np.load(D + "/blob.npy").tobytes()
art = py_base.load_artifact(ART)
art_entries = entries.entries(os.path.basename(ART).removesuffix(".json"))
base = py_base.Base(art)
ex = art_entries
base.execute_into(art_entries["main"], blob, bytearray(0))
for c in ["capture", "captureStep", "captureBlas", "captureRow"]:
    base.execute_into(ex[c], b"", bytearray(0))
base.execute_into(ex["reload"], blob, bytearray(0))


def t(fn):
    for _ in range(5):
        base.execute_into(ex[fn], b"", bytearray(0))
    s = time.perf_counter()
    for _ in range(N):
        base.execute_into(ex[fn], b"", bytearray(0))
    return (time.perf_counter() - s) / N * 1e3


fwd, step = t("replay"), t("replayStep")
blas, row = t("replayBlas"), t("replayRow")
whole = fwd + step
print(f"  contractions alone      {blas:6.2f} ms   (723 cuBLAS launches)")
print(f"  proven kernels alone    {row:6.2f} ms   (794 launches)")
print(f"  sum of the two          {blas + row:6.2f} ms")
print(f"  the whole step          {whole:6.2f} ms   (overlap saves "
      f"{blas + row - whole:.2f} ms)")
print(f"  torch.compile step        4.82 ms")
