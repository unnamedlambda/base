#!/usr/bin/env python3
"""Replay forward+backward+update under a profiler, nothing else in the timeline."""
import os
import sys, numpy as np, py_base
import entries

ART, D = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20

blob = np.load(D + "/blob.npy").tobytes()
art = py_base.load_artifact(ART)
art_entries = entries.entries(os.path.basename(ART).removesuffix(".json"))
base = py_base.Base(art)
ex = art_entries
base.execute_into(art_entries["main"], blob, bytearray(0))
base.execute_into(ex["capture"], b"", bytearray(0))
base.execute_into(ex["captureStep"], b"", bytearray(0))
base.execute_into(ex["reload"], blob, bytearray(0))

for _ in range(5):
    base.execute_into(ex["replay"], b"", bytearray(0))
    base.execute_into(ex["replayStep"], b"", bytearray(0))
for _ in range(N):
    base.execute_into(ex["replay"], b"", bytearray(0))
    base.execute_into(ex["replayStep"], b"", bytearray(0))
