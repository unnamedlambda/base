#!/usr/bin/env python3
"""Run the `Ten`-defined transformer block and check it against NumPy.

The block is written once, as a Lean term (`MlpCifar.k3Layer`): RMSNorm, Wq,
RoPE, attention against the KV cache, Wo, residual, RMSNorm, a gated
feed-forward, residual.  Twenty-eight kernels — the artifact's first 28 launches,
its forward half — each an instance of a schema proven once, and
`qwen_kernels_are_the_stages` says the PTX in this artifact is those stages' own
statements.

This is the other end: the same block in NumPy, so the numbers can be compared.

Run:  py-base/.venv/bin/python applications/ten-qwen/run.py <artifact.json>
"""

import json
import os
import struct
import sys

import numpy as np
import py_base

DM, NH, HD, SQ, DFF = 512, 8, 64, 128, 1408
EPS = np.float32(0.000001)
QHOST_LEN_OFF = 0x0080
QPTX_OFF = 0x0100


def rms_norm(g, x):
    ss = np.float32(np.sum(x.astype(np.float32) ** 2))
    return (x * np.float32(1.0) / np.sqrt(ss / np.float32(DM) + EPS) * g).astype(np.float32)


def rope(x, cos, sin):
    """Half-split rotation, row by row — the six passes the kernel runs."""
    h = HD // 2
    out = np.empty_like(x)
    out[:, :h] = x[:, :h] * cos[:h] - x[:, h:] * sin[:h]
    out[:, h:] = x[:, h:] * cos[h:] + x[:, :h] * sin[h:]
    return out.astype(np.float32)


def attn(kc, vc, q):
    sc = (q @ kc.T).astype(np.float32) * np.float32(1.0 / np.sqrt(HD))
    sc = (sc - sc.max(axis=1, keepdims=True)).astype(np.float32)   # the kernel's row max
    e = np.exp(sc).astype(np.float32)
    p = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    return (p @ vc).astype(np.float32)


def silu(v):
    return (v / (np.float32(1.0) + np.exp(-v))).astype(np.float32)


def block(w, x):
    n1 = rms_norm(w["g1"], x)
    q = rope((w["wq"] @ n1).astype(np.float32).reshape(NH, HD), w["cos"], w["sin"])
    a = attn(w["kc"], w["vc"], q).reshape(DM)
    xr = (x + (w["wo"] @ a).astype(np.float32)).astype(np.float32)
    n2 = rms_norm(w["g2"], xr)
    ff = (silu((w["w1"] @ n2).astype(np.float32)) * (w["w3"] @ n2).astype(np.float32)).astype(np.float32)
    return (xr + (w["w2"] @ ff).astype(np.float32)).astype(np.float32)


def make_weights(seed=7):
    r = np.random.default_rng(seed)
    def u(*shape):
        return (r.standard_normal(shape) * 0.02).astype(np.float32)
    ang = (np.arange(HD // 2, dtype=np.float32) + 1.0) * 0.11
    return {
        "cos": np.concatenate([np.cos(ang), np.cos(ang)]).astype(np.float32),
        "sin": np.concatenate([np.sin(ang), np.sin(ang)]).astype(np.float32),
        "ones": np.ones(SQ, dtype=np.float32),
        "kc": u(SQ, HD), "vc": u(SQ, HD), "x": u(DM),
        "g1": (1.0 + u(DM)).astype(np.float32),
        "g2": (1.0 + u(DM)).astype(np.float32),
        "wq": u(DM, DM), "wo": u(DM, DM),
        "w1": u(DFF, DM), "w3": u(DFF, DM), "w2": u(DM, DFF),
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path or not os.path.exists(path):
        sys.exit("usage: run.py <ten_qwen_block.json>")
    art = py_base.load_artifact(path)
    w = make_weights()
    w["ones"] = np.ones(DM, dtype=np.float32)   # spans the widest reduction

    # Binding order is Lean's: cos sin ones kc vc x g1 g2 wq wo w1 w3 w2.
    order = ["cos", "sin", "ones", "kc", "vc", "x", "g1", "g2",
             "wq", "wo", "w1", "w3", "w2"]
    blob = b"".join(w[k].astype("<f4").ravel().tobytes() for k in order)

    mem = bytes(json.load(open(path))["setup"]["initial_memory"])
    want = struct.unpack_from("<I", mem, QHOST_LEN_OFF)[0]
    assert len(blob) == want, f"host packing {len(blob)} vs Lean layout {want}"
    print(f"layout   : {len(blob)} bytes, matches Lean's qHostIn ({want})")

    base = py_base.Base(art.setup)
    ex = art.extras
    base.execute_into(art.main, blob, bytearray(0))
    base.execute_into(ex["runFwd"], b"", bytearray(0))

    out = bytearray(DM * 4)
    base.execute_into(ex["fetchOut"], b"", out)
    got = np.frombuffer(bytes(out), dtype="<f4")

    ref = block(w, w["x"])
    d = np.abs(got - ref)
    scale = max(float(np.max(np.abs(ref))), 1e-30)
    print(f"kernels  : 28, one launch sequence, one sync")
    print(f"out[:4]  : gpu {got[:4]}")
    print(f"           ref {ref[:4]}")
    print(f"max |Δ|  : {float(d.max()):.3e}   rel {float(d.max())/scale:.3e}")
    ok = float(d.max()) / scale < 2e-5

    # The fused schedule, against the same forward it is a schedule of.
    # `qwen_fwd_fusion_agrees` says the two denote the same memory at every
    # buffer but 33, the temporary fusion removes — so this is bit equality,
    # not a tolerance. Without it, `runFwdFused` would be an emitted function
    # nothing runs, and the theorem would be about a program nothing runs.
    base.execute_into(ex["runFwdFused"], b"", bytearray(0))
    fused_out = bytearray(DM * 4)
    base.execute_into(ex["fetchOut"], b"", fused_out)
    fused = np.frombuffer(bytes(fused_out), dtype="<f4")
    fused_ok = bool((fused == got).all())
    print(f"fused    : 27 kernels, second norm's scale and gain in one")
    print(f"           bit-identical to the 28-kernel forward: {fused_ok}")
    ok = ok and fused_ok

    print("RESULT   :", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
