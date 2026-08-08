#!/usr/bin/env python3
"""Sparse mixture-of-experts, dispatched by rebinding.

Thirty-two experts are resident; the launch sequence has two slots.  The host
runs the router, reads the row it wrote, ranks it, and writes six integers into
the bind array — the ids of the chosen experts' weights, into the entries the
slot kernels read.  No kernel is re-emitted and no weight is copied, which is
what `moe_slots_are_bound` says: the term names `MSLOT0…`, never a buffer in
the store.

The check is against NumPy at the experts the host picked, so a dispatch that
bound the wrong weights shows up as a wrong answer rather than a slower one.

Run:  py-base/.venv/bin/python applications/ten-qwen/moe.py <artifact.json>
"""

import sys
import time

import numpy as np
import py_base

MD, MDFF, NE, MUSED = 128, 256, 32, 2


def silu(v):
    return (v / (np.float32(1.0) + np.exp(-v))).astype(np.float32)


def softmax(v):
    e = np.exp((v - v.max()).astype(np.float32)).astype(np.float32)
    return (e / e.sum()).astype(np.float32)


def expert(w1, w3, w2, x):
    return (w2 @ (silu((w1 @ x).astype(np.float32))
                  * (w3 @ x).astype(np.float32)).astype(np.float32)).astype(np.float32)


def main():
    r = np.random.default_rng(11)

    def u(*shape):
        return (r.standard_normal(shape) * 0.05).astype(np.float32)

    wr = u(NE, MD)
    x = u(MD)
    ws = [(u(MDFF, MD), u(MDFF, MD), u(MD, MDFF)) for _ in range(NE)]

    art = py_base.load_artifact(sys.argv[1])
    ex = art.extras
    blob = (np.ones(NE, dtype=np.float32).tobytes() + wr.tobytes() + x.tobytes()
            + b"".join(m.astype("<f4").ravel().tobytes() for w in ws for m in w))
    base = py_base.Base(art.setup)
    base.execute_into(art.main, blob, bytearray(0))

    # 1. Route.
    base.execute_into(ex["runRouter"], b"", bytearray(0))
    g = bytearray(NE * 4)
    base.execute_into(ex["fetchGate"], b"", g)
    gate = np.frombuffer(bytes(g), dtype="<f4")

    ref_gate = softmax((wr @ x).astype(np.float32))
    print(f"router   : max |Δ| {float(np.abs(gate - ref_gate).max()):.3e}")

    # 2. Rank, and tell the device which weights the slots are.
    chosen = np.argsort(-gate)[:MUSED].astype("<u4")
    print(f"chosen   : experts {list(chosen)}  gates {gate[chosen]}")
    base.execute_into(ex["bindExperts"], chosen.tobytes(), bytearray(0))
    packed = np.zeros(32, dtype=np.float32)      # the buffer is a warp wide
    packed[:MUSED] = gate[chosen]
    base.execute_into(ex["uploadGates"], packed.tobytes(), bytearray(0))

    # 3. Run the two slots.
    base.execute_into(ex["runExperts"], b"", bytearray(0))
    o = bytearray(MD * 4)
    base.execute_into(ex["fetchOut"], b"", o)
    got = np.frombuffer(bytes(o), dtype="<f4")

    ref = np.zeros(MD, dtype=np.float32)
    for j, e in enumerate(chosen):
        ref = (ref + gate[chosen][j] * expert(*ws[e], x)).astype(np.float32)

    d = float(np.abs(got - ref).max())
    scale = max(float(np.abs(ref).max()), 1e-30)
    print(f"mixture  : max |Δ| {d:.3e}   rel {d / scale:.3e}")

    # A dispatch that ignored the binding would still produce *a* number, so
    # check it is not the number some other pair of experts gives.
    other = [e for e in range(NE) if e not in set(chosen.tolist())][:MUSED]
    alt = np.zeros(MD, dtype=np.float32)
    for j, e in enumerate(other):
        alt = (alt + gate[chosen][j] * expert(*ws[e], x)).astype(np.float32)
    apart = float(np.abs(ref - alt).max()) / scale
    print(f"distinct : a different pair differs by rel {apart:.3e}")

    # What the dispatch costs, and what it saves.  Five launches per expert
    # slot plus one accumulate, so the per-expert cost is measurable from the
    # sequence that ran and the dense reading is an extrapolation, not a guess.
    def t(fn, reps=200):
        for _ in range(5):
            base.execute_into(fn, b"", bytearray(0))
        t0 = time.perf_counter()
        for _ in range(reps):
            base.execute_into(fn, b"", bytearray(0))
        return (time.perf_counter() - t0) / reps * 1e6

    t_route = t(ex["runRouter"])
    t_bind = t(ex["bindExperts"])
    t_exp = t(ex["runExperts"])
    per_expert = t_exp / MUSED
    print()
    print(f"router   : {t_route:7.1f} us   ({len(chosen)} of {NE} scored)")
    print(f"bind     : {t_bind:7.1f} us   ({3 * MUSED} integers, no copy)")
    print(f"experts  : {t_exp:7.1f} us   ({per_expert:.1f} us per slot)")
    print(f"sparse   : {t_route + t_bind + t_exp:7.1f} us")
    print(f"dense    : {t_route + per_expert * NE:7.1f} us   "
          f"(all {NE} evaluated, {NE // MUSED}x the expert work)")

    ok = d / scale < 2e-5 and apart > 1e-2
    print("\nRESULT   :", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
