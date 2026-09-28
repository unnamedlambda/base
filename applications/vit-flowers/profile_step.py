#!/usr/bin/env python3
"""Replay forward+backward+update under a profiler, nothing else in the timeline."""
import os
import sys, numpy as np, py_base

ART, D = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20

blob = np.load(D + "/blob.npy").tobytes()
art = py_base.load_artifact(ART)
base = py_base.Driver(art)
base.execute("main", blob)
base.execute("capture")
base.execute("captureStep")
base.execute("reload", blob)

for _ in range(5):
    base.execute("replay")
    base.execute("replayStep")
for _ in range(N):
    base.execute("replay")
    base.execute("replayStep")
