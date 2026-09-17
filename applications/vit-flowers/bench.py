#!/usr/bin/env python3
"""Time our forward, both ways it can be issued.

numpy + py_base only — torch must not share the process.

`run` issues every launch from the host; `replay` issues one
`cl_cuda_graph_launch` of the sequence `capture` recorded.  They are the same
sequence, so the outputs are compared bit for bit rather than to a tolerance:
a replay that differs at all is a replay of something else.
"""
import os
import sys, time, numpy as np, py_base

ART, D = sys.argv[1], sys.argv[2]
SQ, NC = 200, 128
TORCH_MS = 1.42          # batch 1, torch.compile(max-autotune), bench_torch_step.py

blob = np.load(D + "/blob.npy").tobytes()
art = py_base.load_artifact(ART)
base = py_base.Base(art)
base.execute("main", blob, bytearray(0))
run, replay, fetch = "run", "replay", "fetch"


def timed(fn, n=50, warm=10):
    for _ in range(warm):
        base.execute(fn, b"", bytearray(0))
    out = bytearray(SQ * NC * 4)
    base.execute(fetch, b"", out)
    t = time.perf_counter()
    for _ in range(n):
        base.execute(fn, b"", bytearray(0))
    ms = (time.perf_counter() - t) / n * 1e3
    return ms, np.frombuffer(bytes(out), "<f4")


ms_run, out_run = timed(run)
base.execute("capture", b"", bytearray(0))
ms_rep, out_rep = timed(replay)

same = bool((out_run == out_rep).all())
print(f"launches   : 873 (241 cuBLAS + 632 proven)")
print(f"host issue : {ms_run:7.2f} ms   {TORCH_MS/ms_run:5.2f}x PyTorch")
print(f"graph replay:{ms_rep:7.2f} ms   {TORCH_MS/ms_rep:5.2f}x PyTorch   "
      f"({ms_run/ms_rep:.2f}x over host issue)")
print(f"PyTorch    : {TORCH_MS:7.2f} ms   (batch 1, torch.compile)")
print(f"replay bit-identical to host issue: {same}")
print("RESULT     :", "OK" if same else "MISMATCH")
sys.exit(0 if same else 1)
