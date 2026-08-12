#!/usr/bin/env python3
"""Replay forward+backward+update under a profiler, nothing else in the timeline."""
import sys, numpy as np, py_base

ART, D = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20

blob = np.load(D + "/blob.npy").tobytes()
art = py_base.load_artifact(ART)
base = py_base.Base(art.setup)
ex = art.extras
base.execute_into(art.main, blob, bytearray(0))
base.execute_into(ex["capture"], b"", bytearray(0))
base.execute_into(ex["captureStep"], b"", bytearray(0))
base.execute_into(ex["reload"], blob, bytearray(0))

for _ in range(5):
    base.execute_into(ex["replay"], b"", bytearray(0))
    base.execute_into(ex["replayStep"], b"", bytearray(0))
for _ in range(N):
    base.execute_into(ex["replay"], b"", bytearray(0))
    base.execute_into(ex["replayStep"], b"", bytearray(0))
