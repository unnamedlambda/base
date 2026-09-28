#!/usr/bin/env python3
"""Where a training step's time goes. numpy + py_base only."""
import os
import sys, time, numpy as np, py_base

ART, D = sys.argv[1], sys.argv[2]
SQ, NC, N = 200, 128, 30

blob = np.load(D + "/blob.npy").tobytes()
art = py_base.load_artifact(ART)
base = py_base.Driver(art)
base.execute("main", blob)
base.execute("capture")
base.execute("captureStep")
base.execute("reload", blob)

buf = bytearray(SQ * NC * 4)
seed = np.zeros((SQ, NC), np.float32).tobytes()


def t(fn, arg=b"", out=None):
    for _ in range(5):
        base.execute(fn, arg, out)
    s = time.perf_counter()
    for _ in range(N):
        base.execute(fn, arg, out)
    return (time.perf_counter() - s) / N * 1e3


parts = [
    ("forward  (replay)", t("replay")),
    ("bwd+sgd  (replay)", t("replayStep")),
    ("fetch logits", t("fetch", b"", buf)),
    ("upload seed", t("seed", seed)),
    ("forward  (host issue)", t("run")),
    ("backward (host issue)", t("bwd")),
    ("updates  (host issue)", t("sgd")),
    ("whole step (host issue)", t("step")),
]
for name, ms in parts:
    print(f"  {name:26s} {ms:7.2f} ms")
tot = parts[0][1] + parts[1][1] + parts[2][1] + parts[3][1]
print(f"  {'step, replayed + host CE':26s} {tot:7.2f} ms")
print(f"  {'torch.compile step':26s} {4.82:7.2f} ms")
print(f"  ratio                      {4.82/tot:7.2f}x")
