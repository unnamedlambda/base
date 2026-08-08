#!/usr/bin/env python3
"""Attribute the training step's time to real launches, and to graph replay.

Two questions this answers that a single wall-clock number cannot.

*Where* the time is: the `runTo{k}` entries run the first `k` launches of the
tape and nothing else, so differencing adjacent entries charges a bucket of
launches with its own microseconds.  A bucket that costs far more than its
launch count is a slow kernel, not launch overhead — which is the distinction
that decided what to optimise here and what to leave alone.

*Whether* it is launch overhead at all: `capture` records the sequence as a
CUDA graph and `replay` issues it as one driver call.  What survives replay is
device-side per-kernel cost.  Replay is checked against the launched path
first — it must be bit-identical, not close.

Run:  py-base/.venv/bin/python applications/ten-qwen/profile.py <artifact.json>
"""

import sys
import time

import numpy as np
import py_base

sys.path.insert(0, "applications/ten-qwen")
import run as R

DM, DFF = R.DM, R.DFF
ORDER = ["cos", "sin", "ones", "kc", "vc", "x", "g1", "g2",
         "wq", "wo", "w1", "w3", "w2"]

# The launch counts the artifact carries entries for, in order.
LADDER = [("runFwd", 28), ("runTo30", 30), ("runTo31", 31), ("runTo34", 34),
          ("runTo35", 35), ("runTo37", 37), ("runTo45", 45), ("runTo60", 60),
          ("runTo75", 75), ("runBlock", 90)]


def timed(base, fn, reps=100):
    for _ in range(5):
        base.execute_into(fn, b"", bytearray(0))
    t0 = time.perf_counter()
    for _ in range(reps):
        base.execute_into(fn, b"", bytearray(0))
    return (time.perf_counter() - t0) / reps * 1e6


def main():
    art = py_base.load_artifact(sys.argv[1])
    ex = art.extras
    w = R.make_weights()
    w["ones"] = np.ones(DM, dtype=np.float32)
    base = py_base.Base(art.setup)
    base.execute_into(art.main,
                      b"".join(w[k].astype("<f4").ravel().tobytes() for k in ORDER),
                      bytearray(0))
    base.execute_into(ex["uploadDOut"], np.ones(DM, dtype=np.float32).tobytes(),
                      bytearray(0))

    def gradient():
        g = bytearray(DM * DFF * 4)
        base.execute_into(ex["fetchDW2"], b"", g)
        return np.frombuffer(bytes(g), dtype="<f4").copy()

    base.execute_into(ex["runBlock"], b"", bytearray(0))
    ref = gradient()
    base.execute_into(ex["capture"], b"", bytearray(0))
    base.execute_into(ex["replay"], b"", bytearray(0))
    delta = float(np.abs(gradient() - ref).max())
    print(f"replay vs launches: max |Δ| {delta:.3e}"
          f"   {'identical' if delta == 0.0 else 'DIFFERS'}\n")

    print(f"{'launches':>13} {'cum us':>9} {'bucket us':>10} {'us/launch':>10}")
    prev_k, prev_t = 0, 0.0
    for name, k in LADDER:
        if name not in ex:
            continue
        t = timed(base, ex[name])
        dk, dt = k - prev_k, t - prev_t
        print(f"{prev_k:5d}..{k:<6d} {t:9.1f} {dt:10.1f} {dt / dk:10.2f}")
        prev_k, prev_t = k, t

    one, two, four = (timed(base, ex["replay"]), timed(base, ex["replay2"]),
                      timed(base, ex["replay4"]))
    marginal = (four - two) / 2.0
    print(f"\nlaunched step  : {prev_t:8.1f} us  ({prev_t / prev_k:.2f} us/launch)")
    print(f"replayed step  : {one:8.1f} us")
    print(f"replay marginal: {marginal:8.1f} us  ({marginal / prev_k:.3f} us/launch)")
    print(f"graph is worth : {prev_t / marginal:8.2f}x")
    return 0 if delta == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
