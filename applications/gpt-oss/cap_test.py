"""What the engine does when a conversation runs out of key cache.

`CAP_FULL` positions is the ceiling, and two things used to be indexed by the
raw position rather than by a slot: the rotation tables, which hold exactly
`ROPE_N = CAP_FULL` rows.  A step at position `CAP_FULL` read sine one row past
its table -- which is the first row of cosine -- and cosine one row past the
buffer, off the end of a device allocation.  The ring slots were never the
problem; a modulus cannot leave its buffer.

`dStepM` now clamps.  That is a backstop rather than a policy: the caller is
expected to refuse a turn that cannot fit, and the generation loop stops at the
cap on its own.  What the clamp buys is that overrunning is a stalled reply
instead of an out-of-bounds read.

This test says so in the one way that cannot be argued with: two steps at
different positions above the cap must produce *bit-identical* logits, because
both ran at `CAP_FULL - 1`; and a step just below the cap must differ from
them, because nothing clamped it.  A missing clamp fails the first; a clamp
that fires too early fails the second.

  python applications/gpt-oss/cap_test.py \\
      lean-artifacts/artifacts/GptOssDecode/gptoss_decode.json \\
      --bank data/gptoss-bank
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout import (CAP_FULL, D_IN_BYTES, D_OUT_BYTES, D_OUT_LOGITS, VOCAB,
                    check_layout, acquire_engine_lock)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--token", type=int, default=1428)
    args = ap.parse_args()

    acquire_engine_lock('cap_test.py')
    import py_base

    check_layout(json.load(open(args.artifact)))
    art = py_base.load_artifact(args.artifact)
    assert not art.extras, f"expected one entry and no extras; got {sorted(art.extras)}"
    base = py_base.Base(art.setup)

    paths = {}
    for off, name in ((16, "experts.bin"), (272, "dense.bin"),
                      (528, "embed.bin"), (784, "tokenizer.bin")):
        p = os.path.abspath(os.path.join(args.bank, name)).encode() + b"\0"
        assert len(p) < 256, name
        paths[off] = p

    def step(tok, pos):
        buf = bytearray(D_IN_BYTES)
        struct.pack_into("<III", buf, 0, tok, pos, 0)     # mode 0: one step
        for off, p in paths.items():
            buf[off:off + len(p)] = p
        out = bytearray(D_OUT_BYTES)
        base.execute_into(art.main, bytes(buf), out)
        b = bytes(out)
        nxt = struct.unpack_from("<I", b, 0)[0]
        return nxt, np.frombuffer(b, np.float32, count=VOCAB, offset=D_OUT_LOGITS).copy()

    print("  first call reads 12.9 GiB off disk and pins 9.5 GiB; this takes a while")
    # The cache holds whatever these steps put there, so the logits mean
    # nothing on their own.  What is being compared is one run against another.
    below = CAP_FULL - 2
    at = CAP_FULL
    far = CAP_FULL + 808

    # Warm first, so the comparisons below are not also comparing a cold cache
    # against a warm one.
    step(args.token, 0)

    id_b, lg_b = step(args.token, below)
    id_a, lg_a = step(args.token, at)
    id_f, lg_f = step(args.token, far)

    clamped_agree = np.array_equal(lg_a, lg_f)
    below_differs = not np.array_equal(lg_b, lg_a)
    print(f"  position {below:5d} (under the cap): id {id_b}")
    print(f"  position {at:5d} (at the cap)     : id {id_a}")
    print(f"  position {far:5d} (well past it)  : id {id_f}")
    print(f"  the two over the cap are bit-identical : {clamped_agree}")
    print(f"  the one under it differs               : {below_differs}")

    if not clamped_agree:
        d = float(np.abs(lg_a.astype(np.float64) - lg_f.astype(np.float64)).max())
        print(f"RESULT : FAIL  (positions {at} and {far} disagree by {d:.4e}; "
              f"the clamp is not holding, and the rotation tables are being "
              f"read past their end)")
        return 1
    if not below_differs:
        print(f"RESULT : FAIL  (position {below} matches the clamped ones, so "
              f"the clamp is firing below the cap and is throwing away "
              f"positions the cache has room for)")
        return 1
    print("RESULT : OK  (no step runs past the key cache, and none below it is clamped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
