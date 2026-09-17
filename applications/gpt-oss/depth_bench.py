"""What a token costs as the conversation gets longer.

Every other measurement in this directory is taken near position zero, where
attention reads almost nothing and the cost is the weights.  That is the
flattering end, and it is the end this application had been reporting: the
131072-context figure in the bank notes was measured with the cache *allocated*
to 131072 and some three hundred positions actually in it.

The twelve full-attention layers read the whole key cache on every token, so
the cost grows with the position, and a conversation that has run to a hundred
thousand tokens is not paying what that benchmark said.

Measuring it does not require filling the cache first.  The length the
contractions run over is `min(pos + 1, cap)`, published in the meta buffer, and
the kernels read that many keys whatever those keys happen to contain.  So a
step at position `p` costs what a step at position `p` costs, on a cold cache
or a warm one -- the numbers here are timings, and the answers that come with
them are meaningless on purpose.

  python applications/gpt-oss/depth_bench.py \\
      lean-artifacts/artifacts/GptOssDecode/gptoss_decode.json \\
      --bank data/gptoss-bank --context 131072

Measured on an RTX 3060, August 2026: 17.8 ms a token at position zero, 84.1 ms
at 131071.  The key cache accounts for a ninth of that growth, so the rest is
an attention path running at roughly a tenth of what the card can read -- which
is the finding this script exists to keep honest.

Run it alone: the pool is 9.48 GiB of pinned host memory.
"""

import argparse
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout import (CAP_DEFAULT, CAP_MAX, D_CTX, D_IN_BYTES, D_OUT_BYTES,
                    check_layout, acquire_engine_lock)  # noqa: E402

# The twelve full-attention layers each read a key cache and a value cache of
# `pos` entries, eight heads of sixty-four, two bytes an element.
NL_FULL, NKV, HD = 12, 8, 64
VRAM_GBPS = 333.0


def kv_bytes(pos):
    return NL_FULL * 2 * NKV * pos * HD * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--context", type=int, default=CAP_DEFAULT)
    ap.add_argument("--token", type=int, default=1428)
    ap.add_argument("--reps", type=int, default=24,
                    help="steps to time at each position")
    args = ap.parse_args()
    assert 0 < args.context <= CAP_MAX, f"--context must be 1..{CAP_MAX}"

    acquire_engine_lock('depth_bench.py')
    import py_base

    art = py_base.load_artifact(args.artifact)
    base = py_base.Base(art)
    check_layout(base)

    paths = {}
    for off, name in ((16, "experts.bin"), (272, "dense.bin"),
                      (528, "embed.bin"), (784, "tokenizer.bin")):
        p = os.path.abspath(os.path.join(args.bank, name)).encode() + b"\0"
        assert len(p) < 256, name
        paths[off] = p

    out = bytearray(D_OUT_BYTES)

    def step(pos):
        buf = bytearray(D_IN_BYTES)
        struct.pack_into("<III", buf, 0, args.token, pos, 0)   # mode 0: one step
        struct.pack_into("<I", buf, D_CTX, args.context)
        for off, p in paths.items():
            buf[off:off + len(p)] = p
        base.execute("main", bytes(buf), out)

    print(f"  context {args.context}; {args.reps} steps timed at each position",
          flush=True)
    print("  first call reads 12.9 GiB off disk and pins 9.5 GiB\n", flush=True)
    step(0)

    marks = [p for p in (0, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                         98304, 131071) if p < args.context]
    if args.context - 1 not in marks:
        marks.append(args.context - 1)

    print(f"  {'position':>9}  {'ms/token':>9}  {'tok/s':>7}  {'vs pos 0':>8}  "
          f"{'KV read':>9}  {'of which KV':>11}", flush=True)
    base_ms = None
    for pos in marks:
        for _ in range(4):                      # warm this position's shapes
            step(pos)
        t0 = time.perf_counter()
        for _ in range(args.reps):
            step(pos)
        ms = (time.perf_counter() - t0) / args.reps * 1e3
        if base_ms is None:
            base_ms = ms
        kb = kv_bytes(pos)
        kv_ms = kb / (VRAM_GBPS * 1e9) * 1e3
        print(f"  {pos:9d}  {ms:9.2f}  {1e3/ms:7.1f}  {ms/base_ms:7.2f}x  "
              f"{kb/2**30:8.2f}G  {kv_ms/ms*100:10.1f}%", flush=True)

    print("\n  The last column is what the key cache alone would cost at "
          f"{VRAM_GBPS:.0f} GB/s.\n  It is the part that grows; everything else "
          "is the same weights every token.\n  When it is a small fraction and "
          "the total has grown anyway, the attention\n  path is not bound by "
          "the cache it reads -- which is the case here.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
