"""gpt-oss-20b decoding, with the whole step inside CLIF.

One call per token.  In: the token, its position, the two paths, and the
embedding row.  Out: the next token, and how many cache misses the step took.
Everything between -- twenty-four layers, the router, the expert cache, the
head -- happens in the artifact.

The expert file is pinned once at start-up, 9.48 GiB of it, and the device
holds as many slots as it turns out to have room for.  So the first call is
slow in a way none of the others are: it reads twelve gigabytes off disk.

  python applications/gpt-oss/run.py <artifact.cbor> --bank data/gptoss-bank \
      --prompt "The capital of France is" -n 8

Do not run this beside tools/check.sh; they both want most of the machine.
Nothing memory-hungry may share this process either: the 9.48 GiB is *pinned*,
so the kernel can neither swap nor reclaim it, and a second large allocation
here freezes the box rather than failing.  That is why scoring against the
reference lives in check.py, downstream of --dump-logits.
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import Bank  # noqa: E402
from layout import (CAP_DEFAULT, CAP_MAX, D_CTX, D_IN_BYTES, D_OUT_TRACE,
                    D_OUT_BYTES, check_layout, acquire_engine_lock)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("-n", type=int, default=8)
    # No --check here on purpose.  This process pins 9.5 GiB, and pinned pages
    # cannot be swapped or reclaimed; running the NumPy reference beside them on
    # a 16 GiB box thrashes the whole machine.  Dump the logits instead and let
    # check.py score them once this process has exited.
    ap.add_argument("--twice", action="store_true",
                    help="decode the first token twice; a warm cache must not change it")
    ap.add_argument("--dump-logits", default=None,
                    help="write the prompt-end logits (f32) here for check.py")
    ap.add_argument("--tokens", default=None, help="comma-separated ids, skips the tokenizer")
    ap.add_argument("--context", type=int, default=CAP_DEFAULT,
                    help=f"positions the full-attention layers keep keys "
                         f"for (max {CAP_MAX}); every one of them is an "
                         f"expert slot not held resident")
    args = ap.parse_args()

    # before anything allocates: two of these pin 9.48 GiB each
    acquire_engine_lock('run.py')

    import py_base

    bank = Bank(args.bank)
    H, NL = bank.H, bank.L
    art = py_base.load_artifact(args.artifact)
    base = py_base.Base(art)
    check_layout(base)

    exp = os.path.abspath(os.path.join(args.bank, "experts.bin")).encode() + b"\0"
    den = os.path.abspath(os.path.join(args.bank, "dense.bin")).encode() + b"\0"
    emb = os.path.abspath(os.path.join(args.bank, "embed.bin")).encode() + b"\0"
    tkz = os.path.abspath(os.path.join(args.bank, "tokenizer.bin")).encode() + b"\0"
    assert max(len(exp), len(den), len(emb), len(tkz)) < 256

    if args.tokens:
        ids = [int(t) for t in args.tokens.split(",")]
    else:
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download
        tok = Tokenizer.from_file(hf_hub_download("openai/gpt-oss-20b", "tokenizer.json"))
        ids = tok.encode(args.prompt, add_special_tokens=False).ids
        print(f"prompt: {args.prompt!r}  ({len(ids)} tokens)")

    def step(tok_id, pos):
        # A token, a position, and three paths.  Nothing the model is made of:
        # the embedding row is gathered and widened inside the artifact.
        buf = bytearray(D_IN_BYTES)
        struct.pack_into("<II", buf, 0, tok_id, pos)          # mode 0: one step
        struct.pack_into("<I", buf, D_CTX, args.context)
        buf[16:16 + len(exp)] = exp
        buf[272:272 + len(den)] = den
        buf[528:528 + len(emb)] = emb
        buf[784:784 + len(tkz)] = tkz
        out = bytearray(D_OUT_BYTES)
        base.execute("main", bytes(buf), out)
        b = bytes(out)
        tok_id_out, miss = struct.unpack_from("<Ii", b, 0)
        lg = np.frombuffer(b, np.float32, count=201088, offset=8)
        # the residual stream after each layer, in the order they ran
        # two rows per layer: after attention, then after the mixture
        tr = np.frombuffer(b, np.float32, count=2 * NL * H,
                           offset=D_OUT_TRACE).reshape(NL, 2, H)
        return tok_id_out, miss, lg, tr

    print("  first call reads 12.9 GiB off disk and pins 9.5 GiB; this takes a while")
    t0 = time.time()
    nxt, miss, lg0, tr0 = step(ids[0], 0)
    print(f"  start-up + first token: {time.time()-t0:.1f}s   (misses {miss})")

    if args.twice:
        # The same token at the same position, once cold and once warm.  The
        # cache is the only thing that differs between the two calls, so a
        # difference here is the cache serving the wrong expert -- and this
        # catches it without a reference model to disagree with.
        nxt2, miss2, lg0b, _ = step(ids[0], 0)
        same = np.array_equal(lg0, lg0b)
        d = float(np.abs(lg0.astype(np.float64) - lg0b.astype(np.float64)).max())
        print(f"  repeat of the same token: misses {miss2-miss}   "
              f"identical {same}   max|delta| {d:.4e}   ids {nxt} vs {nxt2}")
        if not same:
            print("RESULT : NONDETERMINISTIC  (a warm cache changed the answer)")
            return 1

    out_ids = []
    prev_miss = miss
    lg, tr = lg0, tr0
    t0 = time.time()
    for pos in range(1, len(ids)):
        nxt, miss, lg, tr = step(ids[pos], pos)
    pre = time.time() - t0
    # The reference scores the prompt, so the comparison has to be against the
    # logits at the prompt's last position -- generation overwrites `lg`.
    lg_prompt, tr_prompt = lg, tr
    if len(ids) > 1:
        print(f"  prompt, {len(ids)-1} more tokens: {pre:.2f}s"
              f"   {(len(ids)-1)/max(pre,1e-9):.1f} tok/s   misses {miss-prev_miss}")

    pos = len(ids)
    t0, m0 = time.time(), miss
    for _ in range(args.n):
        out_ids.append(nxt)
        nxt, miss, lg, _ = step(nxt, pos)
        pos += 1
    dt = time.time() - t0
    print(f"  generated {args.n} tokens in {dt:.2f}s   {args.n/max(dt,1e-9):.2f} tok/s"
          f"   misses {miss-m0}  ({(miss-m0)/(args.n*24*4)*100:.1f}% of lookups)")
    print(f"  ids: {out_ids}")
    print(f"  last logits: min {lg.min():.3f}  max {lg.max():.3f}  "
          f"argmax {int(lg.argmax())}  nonzero {int((lg != 0).sum())}/{lg.size}")
    if args.dump_logits:
        with open(args.dump_logits, "wb") as f:
            f.write(struct.pack("<I", len(ids)))
            f.write(np.asarray(ids, np.uint32).tobytes())
            f.write(lg_prompt.astype(np.float32).tobytes())
            f.write(tr_prompt.astype(np.float32).tobytes())
        print(f"  prompt-end logits + {NL}x2 half-layer trace -> {args.dump_logits}"
              f"   (score with check.py)")
    if not args.tokens:
        print(f"  continuation: {tok.decode(out_ids)!r}")
    print()
    # A run that produced the same id every step, or id 0 every step, has not
    # decoded anything -- it has failed in a way that still returns.  Saying so
    # here is the difference between a test and a log.
    degenerate = all(i == 0 for i in out_ids) or (
        len(out_ids) > 1 and len(set(out_ids)) == 1)
    if degenerate:
        print("RESULT : DEGENERATE  (every step produced the same id; not a decode)")
        return 1
    print("RESULT : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
