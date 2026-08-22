"""Score logits the engine dumped against the NumPy reference.

This is a separate program from run.py for a reason that cost a machine freeze:
run.py pins 9.5 GiB of host memory for the expert pool, and pinned pages cannot
be swapped or reclaimed.  Running the reference beside them on a 16 GiB box
leaves the kernel nothing to page against, so it thrashes instead of failing.

    python run.py <artifact> --bank B --tokens ... --dump-logits /tmp/lg.bin
    python check.py --bank B --logits /tmp/lg.bin

The dump carries its own token ids, so the two halves cannot silently score
different prompts.
"""

import argparse
import struct
import sys

import numpy as np

from reference import Bank, forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--logits", required=True, help="file written by run.py --dump-logits")
    ap.add_argument("--rel", type=float, default=5e-2,
                    help="max acceptable relative logit error")
    args = ap.parse_args()

    with open(args.logits, "rb") as f:
        blob = f.read()
    (n,) = struct.unpack_from("<I", blob, 0)
    ids = list(np.frombuffer(blob, np.uint32, count=n, offset=4))
    rest = np.frombuffer(blob, np.float32, offset=4 + 4 * n)
    bank = Bank(args.bank)
    toks = [int(i) for i in ids]
    V, H, NL = bank.meta["vocab"], bank.H, bank.L
    ours, trace = rest[:V], None
    if rest.size >= V + 2 * NL * H:
        trace = rest[V:V + 2 * NL * H].reshape(NL, 2, H)
    print(f"  scoring {ours.size} logits at the end of a {n}-token prompt")
    print(f"  tokens: {toks}")

    # Two references, not one.  The f32 reference is the idealised model; the
    # bf16 one rounds activations where the engine's buffers do.  A device that
    # tracks bf16 far better than f32 is precision-limited, not broken -- and
    # scoring only against f32 would report that as a defect.
    scores = {}
    for acts in ("f32", "bf16"):
        print(f"  reference pass, activations {acts}")
        tr = {}
        ref = forward(bank, toks, trace=tr, acts=acts)
        if trace is not None and acts == "bf16":
            # Layer by layer, so a composition fault names the layer that made
            # it instead of only showing up as a wrong token.
            print("  residual stream, device vs reference "
                  "(a = after attention, m = after the mixture):")
            worst = None
            for l in range(NL):
                for h, tag in ((0, "a"), (1, "m")):
                    key = "attn_out" if h == 0 else "layer_out"
                    if key not in tr or l not in tr[key]:
                        continue
                    r = tr[key][l].astype(np.float64)
                    o = trace[l][h].astype(np.float64)
                    rel = float(np.abs(r - o).max()) / max(float(np.abs(r).max()), 1e-6)
                    cos = float(r @ o / max(np.linalg.norm(r) * np.linalg.norm(o), 1e-30))
                    first = worst is None and rel > 1e-2
                    if first:
                        worst = f"{l}{tag}"
                    print(f"    layer {l:>2}{tag}  max-rel {rel:.3e}   cosine {cos:.6f}"
                          f"   rms {np.sqrt((o**2).mean()):.4f} vs "
                          f"{np.sqrt((r**2).mean()):.4f}"
                          f"{'  <-- first divergence' if first else ''}")
            print(f"  first half-layer past 1e-2: {worst}" if worst is not None
                  else "  every half-layer within 1e-2 of the reference")
        k = min(len(ref), ours.size)
        r, o = ref[:k], ours[:k]
        top = np.argsort(-r)[:5]
        scores[acts] = {
            "argmax": int(r.argmax()),
            "rel": float(np.abs(r - o).max()) / max(float(np.abs(r).max()), 1e-6),
            "top": [int(t) for t in top],
            "gap": float(r[top[0]] - r[top[1]]),
        }

    oa = int(ours.argmax())
    print()
    for acts, s in scores.items():
        print(f"  vs {acts:>4}:  argmax {s['argmax']}   top-5 {s['top']}"
              f"   gap-to-2nd {s['gap']:.4f}")
        print(f"            max-rel {s['rel']:.3e}"
              f"   top-1 {'agrees' if s['argmax'] == oa else 'DIFFERS'}")
    print(f"  ours:  argmax {oa}")

    b = scores["bf16"]
    ok = b["argmax"] == oa and b["rel"] <= args.rel
    print()
    if not ok:
        print(f"RESULT : MISMATCH  (device disagrees with the bf16 reference: "
              f"max-rel {b['rel']:.3e} > {args.rel:.0e}"
              f"{', argmax differs' if b['argmax'] != oa else ''})")
        return 1
    print("RESULT : OK   device matches the bf16 reference "
          f"(max-rel {b['rel']:.3e}); the f32 idealisation differs by "
          f"{scores['f32']['rel']:.3e}, which is the cost of bf16, not a fault")
    return 0


if __name__ == "__main__":
    sys.exit(main())
