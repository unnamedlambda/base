#!/usr/bin/env python3
"""Check the derived backward pass against finite differences, on the GPU.

The backward is not written: `Ten.backwardFrom` derives 62 passes from the
block's 28, and `block_backward_lowers` says all of them lower to proven
stages.  What no theorem can say is whether the *rules* mean what they claim,
so this compares them against the only oracle outside the proof — the forward
pass itself, evaluated at perturbed weights.

Loss is `L = Σᵢ out[i]`, so the incoming gradient is a vector of ones.

Run:  py-base/.venv/bin/python applications/ten-qwen/gradcheck.py <artifact.json>
"""

import sys

import numpy as np
import py_base

sys.path.insert(0, "applications/ten-qwen")
import run as R

DM, NH, HD, SQ, DFF = R.DM, R.NH, R.HD, R.SQ, R.DFF
ORDER = ["cos", "sin", "ones", "kc", "vc", "x", "g1", "g2",
         "wq", "wo", "w1", "w3", "w2"]


def main():
    path = sys.argv[1]
    art = py_base.load_artifact(path)
    ex = art.extras
    w = R.make_weights()
    w["ones"] = np.ones(DM, dtype=np.float32)      # spans the widest reduction

    blob = b"".join(w[k].astype("<f4").ravel().tobytes() for k in ORDER)
    base = py_base.Base(art.setup)
    base.execute_into(art.main, blob, bytearray(0))

    dout = np.ones(DM, dtype=np.float32)
    base.execute_into(ex["uploadDOut"], dout.tobytes(), bytearray(0))

    def loss(w2: np.ndarray) -> float:
        base.execute_into(ex["uploadW2"], w2.astype("<f4").ravel().tobytes(),
                          bytearray(0))
        base.execute_into(ex["runFwd"], b"", bytearray(0))
        o = bytearray(DM * 4)
        base.execute_into(ex["fetchOut"], b"", o)
        return float(np.sum(np.frombuffer(bytes(o), dtype="<f4"), dtype=np.float64))

    # The derived gradient, from the run that also does the forward pass.
    base.execute_into(ex["uploadW2"], w["w2"].astype("<f4").ravel().tobytes(),
                      bytearray(0))
    base.execute_into(ex["runBlock"], b"", bytearray(0))
    g = bytearray(DM * DFF * 4)
    base.execute_into(ex["fetchDW2"], b"", g)
    dW2 = np.frombuffer(bytes(g), dtype="<f4").reshape(DM, DFF)

    rng = np.random.default_rng(3)
    h = np.float32(1.0 / 512.0)
    rows = rng.integers(0, DM, size=8)
    cols = rng.integers(0, DFF, size=8)
    print(f"{'w2[i][j]':>14}  {'derived':>12}  {'finite diff':>12}  {'rel':>9}")
    worst = 0.0
    for i, j in zip(rows, cols):
        base_w2 = w["w2"].copy()
        up = base_w2.copy(); up[i, j] += h
        dn = base_w2.copy(); dn[i, j] -= h
        fd = (loss(up) - loss(dn)) / (2.0 * float(h))
        got = float(dW2[i, j])
        rel = abs(got - fd) / max(abs(fd), 1e-6)
        worst = max(worst, rel)
        print(f"  [{i:4d}][{j:4d}]  {got:12.6f}  {fd:12.6f}  {rel:9.2e}")
    print(f"\nworst relative difference: {worst:.2e}")
    nz = float(np.abs(dW2).max())
    print(f"max |dW2|: {nz:.6e}")
    ok = worst < 5e-3 and nz > 0.0   # an all-zero gradient is not agreement
    print("RESULT   :", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
