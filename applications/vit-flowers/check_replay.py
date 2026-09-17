#!/usr/bin/env python3
"""The *captured* forward against the reference, and against the eager one.

`run.py` issues the tape from the host, so it never exercises the graph the
schedule actually replays -- and a batched contraction only appears on that
path.  This runs both and reports them separately: agreement with timm says the
numbers are right, and bit-equality with the eager pass says the two ways of
issuing the same tape landed the same values.
"""
import os
import sys, numpy as np, py_base

ART, D = sys.argv[1], sys.argv[2]
SQ, NC = 200, 128
blob = np.load(D + "/blob.npy").tobytes()
ref = np.load(D + "/ref.npy")

art = py_base.load_artifact(ART)
base = py_base.Base(art)
base.execute("main", blob)


def logits():
    out = bytearray(SQ * NC * 4)
    base.execute("fetch", b"", out)
    return bytes(out)


base.execute("run")
eager = logits()

base.execute("capture")
base.execute("reload", blob)
base.execute("replay")
replayed = logits()

# One training step, then a forward: the logits then reflect the updates, so a
# digest of them covers the backward and the optimiser as well as the forward.
base.execute("captureStep")
base.execute("reload", blob)
base.execute("replay")
base.execute("seed", np.zeros((SQ, NC), np.float32).tobytes())
base.execute("replayStep")
base.execute("replay")
stepped = logits()

got = np.frombuffer(replayed, "<f4").reshape(SQ, NC)[0]
d = np.abs(got - ref)
scale = max(float(np.abs(ref).max()), 1e-30)
print(f"replayed[:4] : {got[:4]}")
print(f"ref[:4]      : {ref[:4]}")
print(f"vs timm      : max |Δ| {d.max():.3e}   rel {d.max()/scale:.3e}")
print(f"vs eager     : {'bit-identical' if eager == replayed else 'DIFFERENT'}")
sv = np.frombuffer(stepped, "<f4").reshape(SQ, NC)[0]
print(f"after 1 step : {sv[:4]}  sum {float(np.frombuffer(stepped, '<f4').sum()):.9g}")
print("RESULT       :", "OK" if d.max() / scale < 2e-4 else "MISMATCH")
