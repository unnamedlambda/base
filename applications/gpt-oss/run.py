"""gpt-oss-20b decoding, with the whole step inside CLIF.

One call per token.  In: the token, its position, the two paths, and the
embedding row.  Out: the next token, and how many cache misses the step took.
Everything between -- twenty-four layers, the router, the expert cache, the
head -- happens in the artifact.

The expert file is pinned once at start-up, 9.48 GiB of it, and the device
holds as many slots as it turns out to have room for.  So the first call is
slow in a way none of the others are: it reads twelve gigabytes off disk.

  python applications/gpt-oss/run.py <artifact.json> --bank data/gptoss-bank \
      --prompt "The capital of France is" -n 8

Do not run this beside tools/check.sh; they both want most of the machine.
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import Bank, bf16_to_f32  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--check", action="store_true", help="compare logits to reference.py")
    ap.add_argument("--tokens", default=None, help="comma-separated ids, skips the tokenizer")
    args = ap.parse_args()

    import py_base

    bank = Bank(args.bank)
    H = bank.H
    art = py_base.load_artifact(args.artifact)
    assert not art.extras, f"expected one entry and no extras; got {sorted(art.extras)}"
    base = py_base.Base(art.setup)

    exp = os.path.abspath(os.path.join(args.bank, "experts.bin")).encode() + b"\0"
    den = os.path.abspath(os.path.join(args.bank, "dense.bin")).encode() + b"\0"
    assert len(exp) < 256 and len(den) < 256

    if args.tokens:
        ids = [int(t) for t in args.tokens.split(",")]
    else:
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download
        tok = Tokenizer.from_file(hf_hub_download("openai/gpt-oss-20b", "tokenizer.json"))
        ids = tok.encode(args.prompt, add_special_tokens=False).ids
        print(f"prompt: {args.prompt!r}  ({len(ids)} tokens)")

    def step(tok_id, pos):
        buf = bytearray(528 + H * 4)
        struct.pack_into("<II", buf, 0, tok_id, pos)
        buf[16:16 + len(exp)] = exp
        buf[272:272 + len(den)] = den
        row = bf16_to_f32(np.asarray(bank.embed[tok_id])).astype(np.float32)
        buf[528:] = row.tobytes()
        out = bytearray(8 + 201088 * 4)
        base.execute_into(art.main, bytes(buf), out)
        tok_id_out, miss = struct.unpack_from("<Ii", bytes(out), 0)
        return tok_id_out, miss, np.frombuffer(bytes(out[8:]), np.float32)

    print("  first call reads 12.9 GiB off disk and pins 9.5 GiB; this takes a while")
    t0 = time.time()
    nxt, miss, lg0 = step(ids[0], 0)
    print(f"  start-up + first token: {time.time()-t0:.1f}s   (misses {miss})")

    out_ids = []
    prev_miss = miss
    t0 = time.time()
    for pos in range(1, len(ids)):
        nxt, miss, lg = step(ids[pos], pos)
    pre = time.time() - t0
    if len(ids) > 1:
        print(f"  prompt, {len(ids)-1} more tokens: {pre:.2f}s"
              f"   {(len(ids)-1)/max(pre,1e-9):.1f} tok/s   misses {miss-prev_miss}")

    pos = len(ids)
    t0, m0 = time.time(), miss
    for _ in range(args.n):
        out_ids.append(nxt)
        nxt, miss, lg = step(nxt, pos)
        pos += 1
    dt = time.time() - t0
    print(f"  generated {args.n} tokens in {dt:.2f}s   {args.n/max(dt,1e-9):.2f} tok/s"
          f"   misses {miss-m0}  ({(miss-m0)/(args.n*24*4)*100:.1f}% of lookups)")
    print(f"  ids: {out_ids}")
    print(f"  last logits: min {lg.min():.3f}  max {lg.max():.3f}  "
          f"argmax {int(lg.argmax())}  nonzero {int((lg != 0).sum())}/{lg.size}")
    if args.check:
        from reference import forward
        ref = forward(bank, ids[:len(ids)])
        k = min(len(ref), lg.size)
        print(f"  reference argmax {int(np.argmax(ref))}  ours {int(lg.argmax())}")
        d = np.abs(ref[:k] - lg[:k]).max() / max(np.abs(ref[:k]).max(), 1e-6)
        print(f"  logit max-rel vs reference: {d:.3e}")
    if not args.tokens:
        print(f"  continuation: {tok.decode(out_ids)!r}")
    print()
    # A run that produced the same id every step, or id 0 every step, has not
    # decoded anything -- it has failed in a way that still returns.  Saying so
    # here is the difference between a test and a log.
    degenerate = len(set(out_ids)) <= 1 or all(i == 0 for i in out_ids)
    if degenerate:
        print("RESULT : DEGENERATE  (every step produced the same id; not a decode)")
        return 1
    print("RESULT : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
