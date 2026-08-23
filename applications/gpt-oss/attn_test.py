"""One layer's attention, on the device, against the NumPy reference.

The other half of the vertical slice.  `moe_test.py` established that pointing
slots at experts computes the mixture; this establishes that the sequence
around it -- normalise, project, rotate, cache, score, soften, mix, project
back -- computes what the model defines, at positions chosen so that each of
the three things gpt-oss does differently is actually exercised:

  * **position 0**, where the row is one key long and every reduction runs
    entirely in its remainder pass;
  * **position 5**, a row shorter than a warp;
  * **position 130**, the first position past the 128-wide window, so the ring
    cache has wrapped and the entry a token overwrites is one that has already
    fallen out of its own window -- if the ring were wrong, this is where it
    stops being an off-by-one and starts being a wrong answer;
  * **position 1000**, deep enough that the window has turned over seven times.

Layer 0 is a sliding-attention layer, which is why those are the interesting
positions.  Its input is the embedding, so no earlier layer has to run.

## Two references, and why

The engine narrows activations to bf16 before they meet a bf16 weight, because
cuBLAS will not contract a mixed pair.  The check is against a reference that
narrows too -- otherwise it would be measuring a modelling decision rather than
an implementation -- and the distance to the *unnarrowed* reference is printed
beside it, so what that decision costs is visible rather than absorbed into a
tolerance.

  python applications/gpt-oss/attn_test.py <artifact.json> --bank data/gptoss-bank
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import Bank, bf16_to_f32, rms_norm  # noqa: E402

CAP = 128           # the sliding window, and so the ring cache's depth
ROPE_N = 131072     # `GptOssAttention.ROPE_N`: the published height, all of it


def to_bf16(x):
    """Nearest, ties to even -- the same rounding `rneBf16` emits."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (b >> 16) & 1
    return ((b + np.uint32(0x7FFF) + lsb) >> 16).astype(np.uint16)


def nb(x):
    """A round trip through bf16: what the device will actually contract."""
    return bf16_to_f32(to_bf16(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--positions", default="0,5,130,1000")
    args = ap.parse_args()

    import py_base

    bank = Bank(args.bank)
    m = bank.meta
    H, HD, NQ, NKV = bank.H, bank.HD, bank.NQ, bank.NKV
    GQA, QO = NQ // NKV, NQ * HD
    QKV = QO + 2 * NKV * HD
    ent = m["dense"]["layers"][0]
    assert "sliding" in ent["type"], "this slice is written for a sliding layer"
    assert m["sliding_window"] == CAP

    positions = [int(p) for p in args.positions.split(",")]
    T = max(positions) + 1

    art = py_base.load_artifact(args.artifact)
    ex = art.extras
    base = py_base.Base(art.setup)
    mem = bytes(json.load(open(args.artifact))["setup"]["initial_memory"])
    want = struct.unpack_from("<I", mem, 0x80)[0]

    # ---- the weights, exactly as the buffers want them ----
    anorm = bank.f32(ent["attn_norm"], (H,))
    qkv_w_raw = bank.dense[ent["qkv_w"]:ent["qkv_b"]]              # bf16, [QKV, H]
    qkv_b = bank.f32(ent["qkv_b"], (QKV,))
    sinks = bank.f32(ent["sinks"], (NQ,))
    o_w_raw = bank.dense[ent["o_w"]:ent["o_b"]]                    # bf16, [H, QO]
    o_b = bank.f32(ent["o_b"], (H,))
    cos = bank.f32(m["dense"]["rope_cos"], (m["max_position"], HD // 2))[:ROPE_N]
    sin = bank.f32(m["dense"]["rope_sin"], (m["max_position"], HD // 2))[:ROPE_N]

    rng = np.random.default_rng(11)
    tokens = rng.integers(0, m["vocab"], T).astype(np.int64)
    x = bf16_to_f32(np.asarray(bank.embed[tokens])).astype(np.float32)

    blob = b"".join([
        np.zeros(H, np.float32).tobytes(),                # in0: x, per step
        np.zeros(64, np.int32).tobytes(),                 # in1: meta, per step
        anorm.tobytes(), qkv_w_raw.tobytes(), qkv_b.tobytes(), sinks.tobytes(),
        o_w_raw.tobytes(), o_b.tobytes(),
        sin.astype(np.float32).tobytes(), cos.astype(np.float32).tobytes(),
        np.int32(H // 2).tobytes(), np.int32(QO // 2).tobytes(),
    ])
    assert len(blob) == want, f"host layout disagrees: packed {len(blob)}, Lean says {want}"
    print(f"  host region {len(blob)/2**20:.1f} MiB matches the layout Lean declared")

    base.execute_into(art.main, blob, bytearray(0))

    # ---- the reference, written the way the model defines it ----
    def project(row, narrow):
        h = rms_norm(row, anorm, m["rms_eps"])
        w = bf16_to_f32(qkv_w_raw.view(np.uint16).reshape(QKV, H))
        return (nb(h) if narrow else h) @ w.T + qkv_b

    def rope(t, pos):
        a, b = t[..., :HD // 2], t[..., HD // 2:]
        c, s = cos[pos], sin[pos]
        return np.concatenate([a * c - b * s, b * c + a * s], -1).astype(np.float32)

    def qkv_at(pos, narrow):
        r = project(x[pos], narrow)
        q = rope(r[:QO].reshape(NQ, HD), pos)
        k = rope(r[QO:QO + NKV * HD].reshape(NKV, HD), pos)
        v = r[QO + NKV * HD:].reshape(NKV, HD)
        return q, k, v

    kv = [qkv_at(p, True) for p in range(T)]
    kv_f = [qkv_at(p, False) for p in range(T)]

    def attend(pos, narrow):
        src = kv if narrow else kv_f
        lo = max(0, pos - CAP + 1)
        q = src[pos][0]
        K = np.stack([src[j][1] for j in range(lo, pos + 1)])     # [L, NKV, HD]
        V = np.stack([src[j][2] for j in range(lo, pos + 1)])
        att = np.einsum("hd,lhd->hl", q, np.repeat(K, GQA, 1)) / np.sqrt(HD)
        mx = np.maximum(att.max(-1), sinks)
        e = np.exp(att - mx[:, None])
        p = e / (e.sum(-1) + np.exp(sinks - mx))[:, None]
        o = np.einsum("hl,lhd->hd", p, np.repeat(V, GQA, 1)).reshape(QO)
        ow = bf16_to_f32(o_w_raw.view(np.uint16).reshape(H, QO))
        return x[pos] + ((nb(o) if narrow else o) @ ow.T + o_b)

    # ---- run every position, read back at the interesting ones ----
    def sweep(rows, readback):
        """Feed the whole sequence through, keeping what `readback` names.

        Every position runs, because the cache is what makes the next one
        mean anything; only the interesting ones are fetched.
        """
        kept = {}
        for pos in range(len(rows)):
            L = min(pos + 1, CAP)
            meta = np.zeros(64, np.int32)
            meta[1] = pos
            meta[2] = L
            meta[3] = L // 32
            meta[4] = 32 * (L // 32)
            meta[5] = L % 32
            meta[6] = pos % CAP
            # M_KVSTRIDE: the cache depth, published rather than emitted
            meta[7] = CAP * HD
            step = rows[pos].tobytes() + meta.tobytes()
            base.execute_into(ex["uploadStep"], step, bytearray(0))
            base.execute_into(ex["step"], step, bytearray(0))
            if pos in readback:
                out = bytearray(H * 4)
                base.execute_into(ex["fetchX"], b"", out)
                kept[pos] = np.frombuffer(bytes(out), np.float32).copy()
            if pos % 100 == 0:
                print(f"    position {pos}/{len(rows)-1}", end="\r", flush=True)
        return kept

    kept = sweep(x, set(positions))
    worst, worst_at = 0.0, -1
    results = []
    for pos in positions:
        got = kept[pos]
        ref = attend(pos, True)
        ref_f = attend(pos, False)
        scale = max(np.abs(ref).max(), 1e-6)
        rel = np.abs(got - ref).max() / scale
        cost = np.abs(ref - ref_f).max() / scale
        results.append((pos, min(pos + 1, CAP), rel, cost))
        if rel > worst:
            worst, worst_at = rel, pos

    print(" " * 40, end="\r")
    for pos, L, rel, cost in results:
        tag = "  (ring has wrapped)" if pos >= CAP else ""
        print(f"  {'ok  ' if rel < 2e-3 else 'FAIL'} position {pos:<5d} keys {L:<4d}"
              f" max rel {rel:.2e}   narrowing costs {cost:.2e}{tag}")

    # ---- the rotation and the packing, read straight out of the row ----
    qkv_out = bytearray(QKV * 4)
    base.execute_into(ex["fetchQkv"], b"", qkv_out)
    gq = np.frombuffer(bytes(qkv_out), np.float32)
    rq, rk, rv = kv[T - 1]
    parts = [("Q, rotated", gq[:QO], rq.reshape(QO)),
             ("K, rotated", gq[QO:QO + NKV * HD], rk.reshape(-1)),
             ("V", gq[QO + NKV * HD:], rv.reshape(-1))]
    stage_ok = True
    for name, a, b in parts:
        r = np.abs(a - b).max() / max(np.abs(b).max(), 1e-6)
        stage_ok &= r < 2e-3
        print(f"  {'ok  ' if r < 2e-3 else 'FAIL'} {name:<12s} at position {T-1}"
              f"        max rel {r:.2e}")

    # ---- the window is load-bearing, checked against the device itself ----
    #
    # Both directions, because either alone passes for the wrong reason.  An
    # implementation that ignored the cache entirely would attend to nothing
    # but the current token, agree with a reference making the same mistake,
    # and pass every check above -- so ask whether a key *inside* the window
    # changes the answer.  An implementation with no window at all would also
    # pass those checks at every position tested -- so ask whether a key
    # outside it leaves the answer alone.
    win_ok = True
    if T > CAP + 1:
        far, near = 0, T - 3          # far is 998 positions back, near is two
        probe = max(positions)
        for name, victim, want_same in [("outside the window", far, True),
                                        ("inside the window", near, False)]:
            y = x.copy()
            y[victim] = bf16_to_f32(np.asarray(bank.embed[(tokens[victim] + 7919)
                                                          % m["vocab"]]))
            again = sweep(y, {probe})[probe]
            sep = np.abs(again - kept[probe]).max() / max(np.abs(kept[probe]).max(), 1e-6)
            ok = (sep == 0.0) if want_same else (sep > 1e-3)
            win_ok &= ok
            verb = "leaves it alone" if want_same else "changes the answer"
            print(f"  {'ok  ' if ok else 'FAIL'} a different token at {victim:<5d}"
                  f" {verb:<18s} at {probe}   rel {sep:.2e}")
        print(" " * 40, end="\r")

    print()
    if worst < 2e-3 and stage_ok and win_ok:
        print("RESULT : OK")
        return 0
    print(f"RESULT : MISMATCH  (worst {worst:.2e} at position {worst_at})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
