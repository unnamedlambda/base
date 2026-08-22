"""One whole layer, on the device, against the NumPy reference.

Attention was checked apart and the mixture was checked apart.  This checks
them joined, which is a different claim: the two halves were validated against
references that each supplied the other's input, and neither says the layer
hands its own output on correctly.  This repo has twice reached "every layer
proven" with none of them applied, both times because the composition was
assumed rather than written down, so it is written down here.

What the joint runs through that neither half did:

  * the **router**, which nothing has exercised until now -- its 32-row f32
    projection of the second normalisation, and the top-4 and softmax the host
    does over it;
  * the **second residual**, and with it the fact that the mixture reads the
    hidden state attention produced rather than one a test handed it;
  * the **two entry points**, because which experts run is not known until the
    router's row has been read back.  A layer is `stepAttn`, a fetch, a bind,
    and `stepMoe` -- and that gap is exactly where a cache miss will be served.

The host's choice is checked for being load-bearing the way `moe_test.py`
checks it: a layer whose router output were ignored would still agree with a
reference that ignored it too, so the run asks whether a deliberately wrong
top-4 changes the answer.

  python applications/gpt-oss/layer_test.py <artifact.json> --bank data/gptoss-bank
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import Bank, bf16_to_f32, rms_norm, softmax, swiglu  # noqa: E402
from attn_test import to_bf16, nb, CAP, ROPE_N  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--positions", default="0,5,130")
    args = ap.parse_args()

    import py_base

    bank = Bank(args.bank)
    m = bank.meta
    H, I, HD, NQ, NKV = bank.H, bank.I, bank.HD, bank.NQ, bank.NKV
    E, K = bank.E, bank.K
    GQA, QO = NQ // NKV, NQ * HD
    QKV = QO + 2 * NKV * HD
    ent = m["dense"]["layers"][0]
    assert "sliding" in ent["type"]

    positions = [int(p) for p in args.positions.split(",")]
    T = max(positions) + 1

    art = py_base.load_artifact(args.artifact)
    ex = art.extras
    base = py_base.Base(art.setup)
    want = struct.unpack_from(
        "<I", bytes(json.load(open(args.artifact))["setup"]["initial_memory"]), 0x80)[0]

    anorm = bank.f32(ent["attn_norm"], (H,))
    qkv_w_raw = bank.dense[ent["qkv_w"]:ent["qkv_b"]]
    qkv_b = bank.f32(ent["qkv_b"], (QKV,))
    sinks = bank.f32(ent["sinks"], (NQ,))
    o_w_raw = bank.dense[ent["o_w"]:ent["o_b"]]
    o_b = bank.f32(ent["o_b"], (H,))
    mnorm = bank.f32(ent["mlp_norm"], (H,))
    rw = bank.f32(ent["router_w"], (E, H))
    rb = bank.f32(ent["router_b"], (E,))
    cos = bank.f32(m["dense"]["rope_cos"], (m["max_position"], HD // 2))[:ROPE_N]
    sin = bank.f32(m["dense"]["rope_sin"], (m["max_position"], HD // 2))[:ROPE_N]

    rng = np.random.default_rng(23)
    tokens = rng.integers(0, m["vocab"], T).astype(np.int64)
    x0 = bf16_to_f32(np.asarray(bank.embed[tokens])).astype(np.float32)

    r, rbytes = m["expert_row"], m["expert_row_bytes"]
    expert_parts = []
    for e in range(E):
        row = bank.experts[e * rbytes:(e + 1) * rbytes]
        for a, b in [(r["gu_blocks"], r["gu_scales"]), (r["gu_scales"], r["gu_bias"]),
                     (r["gu_bias"], r["dn_blocks"]), (r["dn_blocks"], r["dn_scales"]),
                     (r["dn_scales"], r["dn_bias"]), (r["dn_bias"], rbytes)]:
            expert_parts.append(row[a:b].tobytes())

    blob = b"".join([
        np.zeros(H, np.float32).tobytes(), np.zeros(64, np.int32).tobytes(),
        anorm.tobytes(), qkv_w_raw.tobytes(), qkv_b.tobytes(), sinks.tobytes(),
        o_w_raw.tobytes(), o_b.tobytes(),
        sin.astype(np.float32).tobytes(), cos.astype(np.float32).tobytes(),
        np.int32(H // 2).tobytes(), np.int32(QO // 2).tobytes(),
        mnorm.tobytes(), rw.tobytes(), rb.tobytes(),
    ] + expert_parts)
    assert len(blob) == want, f"host layout disagrees: packed {len(blob)}, Lean says {want}"
    print(f"  host region {len(blob)/2**20:.1f} MiB matches the layout Lean declared")
    assert not ex, f"this artifact should have no extras; it has {sorted(ex)}"
    print(f"  the artifact has one entry and no extras: everything a token needs "
          f"happens inside it")

    # ---- the reference: the layer, written the way the model defines it ----
    qkv_w = bf16_to_f32(qkv_w_raw.view(np.uint16).reshape(QKV, H))
    o_w = bf16_to_f32(o_w_raw.view(np.uint16).reshape(H, QO))

    def rope(t, pos):
        a, b = t[..., :HD // 2], t[..., HD // 2:]
        return np.concatenate([a * cos[pos] - b * sin[pos],
                               b * cos[pos] + a * sin[pos]], -1).astype(np.float32)

    def qkv_at(pos):
        row = nb(rms_norm(x0[pos], anorm, m["rms_eps"])) @ qkv_w.T + qkv_b
        return (rope(row[:QO].reshape(NQ, HD), pos),
                rope(row[QO:QO + NKV * HD].reshape(NKV, HD), pos),
                row[QO + NKV * HD:].reshape(NKV, HD))

    kv = [qkv_at(p) for p in range(T)]

    def layer(pos, force=None):
        lo = max(0, pos - CAP + 1)
        q = kv[pos][0]
        Kc = np.stack([kv[j][1] for j in range(lo, pos + 1)])
        Vc = np.stack([kv[j][2] for j in range(lo, pos + 1)])
        att = np.einsum("hd,lhd->hl", q, np.repeat(Kc, GQA, 1)) / np.sqrt(HD)
        mx = np.maximum(att.max(-1), sinks)
        e = np.exp(att - mx[:, None])
        p = e / (e.sum(-1) + np.exp(sinks - mx))[:, None]
        o = np.einsum("hl,lhd->hd", p, np.repeat(Vc, GQA, 1)).reshape(QO)
        x = x0[pos] + (nb(o) @ o_w.T + o_b)
        h = rms_norm(x, mnorm, m["rms_eps"])
        logits = h @ rw.T + rb
        chosen = np.argsort(-logits)[:K] if force is None else force
        gates = softmax(logits[chosen])
        acc = np.zeros(H, np.float32)
        for j, ee in enumerate(chosen):
            gu_w, gu_b, dn_w, dn_b = bank.expert(0, int(ee))
            a = h @ gu_w.T + gu_b
            act = swiglu(a[:I], a[I:], m["swiglu_alpha"], m["swiglu_limit"])
            acc += (act @ dn_w.T + dn_b) * gates[j]
        return x + acc, logits, chosen, gates

    # ---- run: attention, read the router, choose, mix ----
    def sweep(readback):
        """One call per position.  The router's top-4 and its gates are the
        device's now, so nothing is handed back in between -- which also makes
        the comparison below a stronger claim than it was: the reference picks
        its own four, and agreeing means the kernel picked the same."""
        kept = {}
        for pos in range(T):
            L = min(pos + 1, CAP)
            meta = np.zeros(64, np.int32)
            meta[1], meta[2] = pos, L
            meta[3], meta[4], meta[5] = L // 32, 32 * (L // 32), L % 32
            meta[6] = pos % CAP
            # The first call carries the weights; every later one carries only
            # the token's row and its position, because the program remembers
            # it has loaded.  That is the difference between an entry point and
            # a step in someone else's loop.
            step = (blob if pos == 0 else b"") or blob
            if pos:
                step = x0[pos].tobytes() + meta.tobytes()
            else:
                step = bytearray(blob)
                step[0:H * 4] = x0[0].tobytes()
                step[H * 4:H * 4 + 256] = meta.tobytes()
                step = bytes(step)
            out = bytearray(H * 4 + E * 4)
            base.execute_into(art.main, step, out)
            if pos in readback:
                kept[pos] = (np.frombuffer(bytes(out[:H * 4]), np.float32).copy(),
                             np.frombuffer(bytes(out[H * 4:]), np.float32).copy())
            if pos % 25 == 0:
                print(f"    position {pos}/{T-1}", end="\r", flush=True)
        return kept

    kept = sweep(set(positions))
    print(" " * 40, end="\r")

    ok_all = True
    for pos in positions:
        got, glog = kept[pos]
        ref, rlog, rchosen, rgates = layer(pos)
        s = max(np.abs(ref).max(), 1e-6)
        rel = np.abs(got - ref).max() / s
        lrel = np.abs(glog - rlog).max() / max(np.abs(rlog).max(), 1e-6)
        ok = rel < 2e-3 and lrel < 2e-3
        ok_all &= ok
        # The layer output can only match if the device chose the same four and
        # weighted them the same way; a different fourth expert moves it far
        # more than the tolerance.  So the choice is checked by the output, not
        # by reading the choice back.
        print(f"  {'ok  ' if ok else 'FAIL'} position {pos:<5d} layer out {rel:.2e}"
              f"   router row {lrel:.2e}   reference picked"
              f"  {sorted(int(c) for c in rchosen)}"
              f"  gates {np.round(np.sort(rgates)[::-1], 3)}")

    # ---- the choice is load-bearing, measured against a wrong one ----
    #
    # The device picks now, so this cannot force a binding.  Instead it asks
    # what the answer *would* have been under a different top-4: if the layer
    # were ignoring its router, the two would coincide.
    probe = positions[-1]
    wrong = np.array([1, 2, 3, 4], dtype=np.uint32)
    ref_w, _, _, _ = layer(probe, force=wrong)
    sep = np.abs(ref_w - kept[probe][0]).max() / max(np.abs(kept[probe][0]).max(), 1e-6)
    ok2 = sep > 1e-3
    print(f"  {'ok  ' if ok2 else 'FAIL'} a different top-4 would give a different answer"
          f"   {sep:.2e}   (a layer ignoring its router would score 0)")

    print()
    if ok_all and ok2:
        print("RESULT : OK")
        return 0
    print("RESULT : MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
