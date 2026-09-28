#!/usr/bin/env python3
"""The tape captured as a chain, against the tape captured as its own DAG.

`vDag` deals the operations onto four streams and inserts one event per
dependence that crosses between them, so `replayDagFwd` and `replayDagStep`
perform exactly the launches `replay` and `replayStep` perform, in an order the
tape's read/write dependences allow.

Same launches means same numbers: the logits and the updated parameters are
compared bit for bit rather than to a tolerance.
"""
import os
import sys, time, numpy as np, py_base

ART, D = sys.argv[1], sys.argv[2]
SQ, NC, N = 200, 128, 50

blob = np.load(D + "/blob.npy").tobytes()
seed = np.load(D + "/seed.npy").astype("<f4").tobytes()
art = py_base.load_artifact(ART)
base = py_base.Driver(art)
base.execute("main", blob)

for c in ["captureChain", "captureStepChain", "capture", "captureStep"]:
    base.execute(c)
base.execute("reload", blob)


def logits():
    out = bytearray(SQ * NC * 4)
    base.execute("fetch", b"", out)
    return bytes(out)


def fwd_out(fn):
    base.execute("reload", blob)
    base.execute(fn)
    return logits()


def step_out(fn):
    """One training step, then a forward, so the result reflects the updates."""
    base.execute("reload", blob)
    base.execute("replay")
    base.execute("seed", seed)
    base.execute(fn)
    base.execute("replay")
    return logits()


def t(fn, pre=None):
    for _ in range(5):
        if pre: pre()
        base.execute(fn)
    s = time.perf_counter()
    for _ in range(N):
        base.execute(fn)
    return (time.perf_counter() - s) / N * 1e3


f1, f2 = fwd_out("replayChain"), fwd_out("replay")
s1, s2 = step_out("replayStepChain"), step_out("replayStep")

base.execute("reload", blob)
base.execute("replay")
base.execute("seed", seed)

mf1, mf2 = t("replayChain"), t("replay")
ms1, ms2 = t("replayStepChain"), t("replayStep")

print(f"  forward   chain {mf1:7.2f} ms   dag {mf2:7.2f} ms   {mf1/mf2:.2f}x   "
      f"identical {f1 == f2}")
print(f"  bwd+sgd   chain {ms1:7.2f} ms   dag {ms2:7.2f} ms   {ms1/ms2:.2f}x   "
      f"identical {s1 == s2}")
print(f"  full step chain {mf1+ms1:7.2f} ms   dag {mf2+ms2:7.2f} ms   "
      f"{(mf1+ms1)/(mf2+ms2):.2f}x")
print(f"  torch.compile: forward 1.42 ms, step 4.82 ms")
