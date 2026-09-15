#!/usr/bin/env python3
"""Where a training step's time goes. numpy + py_base only."""
import os
import sys, time, numpy as np, py_base
import entries

ART, D = sys.argv[1], sys.argv[2]
SQ, NC, N = 200, 128, 30

blob = np.load(D + "/blob.npy").tobytes()
art = py_base.load_artifact(ART)
art_entries = entries.entries(os.path.basename(ART).removesuffix(".json"))
base = py_base.Base(art)
ex = art_entries
base.execute_into(art_entries["main"], blob, bytearray(0))
base.execute_into(ex["capture"], b"", bytearray(0))
base.execute_into(ex["captureStep"], b"", bytearray(0))
base.execute_into(ex["reload"], blob, bytearray(0))

buf = bytearray(SQ * NC * 4)
seed = np.zeros((SQ, NC), np.float32).tobytes()


def t(fn, arg=b"", out=None):
    o = out if out is not None else bytearray(0)
    for _ in range(5):
        base.execute_into(fn, arg, o)
    s = time.perf_counter()
    for _ in range(N):
        base.execute_into(fn, arg, o)
    return (time.perf_counter() - s) / N * 1e3


parts = [
    ("forward  (replay)", t(ex["replay"])),
    ("bwd+sgd  (replay)", t(ex["replayStep"])),
    ("fetch logits", t(ex["fetch"], b"", buf)),
    ("upload seed", t(ex["seed"], seed)),
    ("forward  (host issue)", t(ex["run"])),
    ("backward (host issue)", t(ex["bwd"])),
    ("updates  (host issue)", t(ex["sgd"])),
    ("whole step (host issue)", t(ex["step"])),
]
for name, ms in parts:
    print(f"  {name:26s} {ms:7.2f} ms")
tot = parts[0][1] + parts[1][1] + parts[2][1] + parts[3][1]
print(f"  {'step, replayed + host CE':26s} {tot:7.2f} ms")
print(f"  {'torch.compile step':26s} {4.82:7.2f} ms")
print(f"  ratio                      {4.82/tot:7.2f}x")
