#!/usr/bin/env python3
"""Fine-tune the twelve-block ViT on the GPU, and compare the loss trajectory
against PyTorch doing the same descent.

numpy + py_base only — torch must not share the process.

Every kernel in the step comes from one `Ten` term: the forward is the model,
the backward is `Ten.backwardFrom` applied to the forward tape, and the update
is one `TOp.upd2` per parameter placed where `Ten.backwardCoT` put that
parameter's gradient.  No gradient formula and no optimiser kernel is written
here or in the generator.

The softmax and the cross-entropy gradient run on the host, on the 128 logits of
the class token — arithmetic on one row, where the label lives.  That is a
stated part of the step, not an omission: `seed` uploads the result and the
backward is derived from it.
"""
import os
import sys, struct, time
import numpy as np
import py_base

ART, D = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
LABEL = int(sys.argv[4]) if len(sys.argv) > 4 else 7
SQ, NC = 200, 128

blob = np.load(D + "/blob.npy").tobytes()
ref = np.load(D + "/losses.npy")

art = py_base.load_artifact(ART)
base = py_base.Driver(art)
base.execute("main", blob)

# Capture both graphs first.  Capturing runs its sequence once eagerly to make
# every module resident, so the step graph's capture applies two updates; the
# reload puts the parameters back before the run that is being compared.
base.execute("capture")
base.execute("captureStep")
base.execute("reload", blob)

logit_buf = bytearray(SQ * NC * 4)
seed = np.zeros((SQ, NC), np.float32)


def forward_logits():
    base.execute("replay")
    base.execute("fetch", b"", logit_buf)
    return np.frombuffer(bytes(logit_buf), "<f4")[:NC].astype(np.float64)


losses, t0 = [], time.perf_counter()
for _ in range(STEPS):
    z = forward_logits()
    p = np.exp(z - z.max())
    p /= p.sum()
    losses.append(float(-np.log(max(p[LABEL], 1e-30))))
    seed[0] = p
    seed[0, LABEL] -= 1.0
    base.execute("seed", seed.tobytes())
    base.execute("replayStep")
ms = (time.perf_counter() - t0) / STEPS * 1e3

n = min(len(losses), len(ref))
d = np.abs(np.array(losses[:n]) - ref[:n])
scale = max(float(np.abs(ref[:n]).max()), 1e-30)

print(f"steps    : {n}   label {LABEL}   lr 1e-3 SGD   {ms:.2f} ms/step")
print(f"{'step':>5}  {'ours':>10}  {'pytorch':>10}  {'|Δ|':>10}")
for i in list(range(min(5, n))) + ([None] if n > 8 else []) + list(range(max(5, n - 3), n)):
    if i is None:
        print(f"{'...':>5}")
        continue
    print(f"{i:5d}  {losses[i]:10.6f}  {ref[i]:10.6f}  {d[i]:10.3e}")

print(f"loss     : {losses[0]:.6f} -> {losses[n-1]:.6f}   "
      f"(pytorch {ref[0]:.6f} -> {ref[n-1]:.6f})")
print(f"max |Δ|  : {d.max():.3e}   rel {d.max()/scale:.3e}")
ok = bool(d.max() / scale < 1e-3 and losses[n - 1] < losses[0])
print("RESULT   :", "OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
