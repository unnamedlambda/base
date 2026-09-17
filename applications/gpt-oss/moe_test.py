"""One layer's mixture, on the device, against the NumPy reference.

The first thing here that runs through the real path -- artifact, JIT'd CLIF
host program, launches -- rather than through a test harness poking libcuda.
What it is checking is not the kernels, which `kernel_test.py` already pins to
the bit; it is the *dispatch*: that pointing four slots at four of thirty-two
resident experts, by rewriting buffer ids and nothing else, computes the
mixture the model defines.

Two checks, because one of them alone would pass for the wrong reason:

  * against the reference, which says the arithmetic is right;
  * against the *other* experts, which says the binding is right.  A dispatch
    that ignored its argument and always ran expert 0 would agree with a
    reference that made the same mistake, so the second check asks whether
    choosing differently computes differently.

  python applications/gpt-oss/moe_test.py <artifact.json> --bank data/gptoss-bank
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import Bank, mx_dequant, swiglu, softmax  # noqa: E402
import entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    import py_base

    bank = Bank(args.bank)
    m = bank.meta
    H, I, E, K = bank.H, bank.I, bank.E, bank.K
    L = args.layer

    art = py_base.load_artifact(args.artifact)
    art_entries = entries.entries(os.path.basename(args.artifact).removesuffix(".json"))
    ex = art_entries
    base = py_base.Base(art)

    # The host region Lean declared, read back rather than restated: if the
    # packing here and the layout there disagreed, this is where it shows.
    want = struct.unpack_from("<I", base.read_memory(0x80, 4))[0]

    # ---- pack: the activation, the gates, then all thirty-two experts ----
    rng = np.random.default_rng(7)
    x = (rng.standard_normal(H) * 0.5).astype(np.float32)
    gates = np.zeros(32, np.float32)
    g4 = softmax(rng.standard_normal(K).astype(np.float32))
    gates[:K] = g4
    chosen = np.array([3, 17, 28, 5], dtype=np.uint32)

    r = m["expert_row"]
    rb = m["expert_row_bytes"]
    parts = [x.tobytes(), gates.tobytes()]
    for e in range(E):
        off = (L * E + e) * rb
        row = bank.experts[off:off + rb]
        for a, b in [(r["gu_blocks"], r["gu_scales"]), (r["gu_scales"], r["gu_bias"]),
                     (r["gu_bias"], r["dn_blocks"]), (r["dn_blocks"], r["dn_scales"]),
                     (r["dn_scales"], r["dn_bias"]), (r["dn_bias"], rb)]:
            parts.append(row[a:b].tobytes())
    blob = b"".join(parts)
    assert len(blob) == want, f"host layout disagrees: packed {len(blob)}, Lean says {want}"
    print(f"  host region {len(blob)/2**20:.1f} MiB matches the layout Lean declared")

    base.execute_into(art_entries["main"], blob, bytearray(0))
    base.execute_into(ex["bindExperts"], chosen.tobytes(), bytearray(0))
    base.execute_into(ex["uploadX"], x.tobytes(), bytearray(0))
    base.execute_into(ex["uploadGates"], gates.tobytes(), bytearray(0))
    base.execute_into(ex["runExperts"], b"", bytearray(0))
    out = bytearray(H * 4)
    base.execute_into(ex["fetchOut"], b"", out)
    got = np.frombuffer(bytes(out), np.float32)

    # ---- the reference: the same four experts, mixed by the same gates ----
    def mixture(expert_ids, gs):
        acc = np.zeros(H, np.float32)
        for j, e in enumerate(expert_ids):
            gu_w, gu_b, dn_w, dn_b = bank.expert(L, int(e))
            a = x @ gu_w.T + gu_b
            act = swiglu(a[:I], a[I:], m["swiglu_alpha"], m["swiglu_limit"])
            acc += (act @ dn_w.T + dn_b) * gs[j]
        return acc

    ref = mixture(chosen, g4)
    rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-6)
    ok = rel < 2e-3
    print(f"  {'ok  ' if ok else 'FAIL'} mixture matches the reference   max rel {rel:.2e}")

    # ---- the binding is load-bearing: a different choice must differ ----
    other = np.array([11, 24, 2, 30], dtype=np.uint32)
    base.execute_into(ex["bindExperts"], other.tobytes(), bytearray(0))
    base.execute_into(ex["runExperts"], b"", bytearray(0))
    out2 = bytearray(H * 4)
    base.execute_into(ex["fetchOut"], b"", out2)
    got2 = np.frombuffer(bytes(out2), np.float32)
    ref2 = mixture(other, g4)
    rel2 = np.abs(got2 - ref2).max() / max(np.abs(ref2).max(), 1e-6)
    sep = np.abs(got - got2).max() / max(np.abs(got).max(), 1e-6)
    ok2 = rel2 < 2e-3
    ok3 = sep > 0.1
    print(f"  {'ok  ' if ok2 else 'FAIL'} a second binding matches too     max rel {rel2:.2e}")
    print(f"  {'ok  ' if ok3 else 'FAIL'} the two bindings really differ   rel sep {sep:.2f}"
          f"   (a dispatch ignoring its argument would score 0)")

    print()
    if ok and ok2 and ok3:
        print("RESULT : OK")
        return 0
    print("RESULT : MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
