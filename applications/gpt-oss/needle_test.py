"""A fact planted early, asked for late, past where the cache used to end.

`cap_test.py` says the key cache ends where it was asked to. It does not say
the model can *use* the depth: a cache that is allocated but mis-strided, or a
rotation table read at the wrong row, gives a model that answers fluently about
nothing in particular -- which is exactly how the inverted YaRN ramp presented,
and it took a cross-check against another implementation to find.

So this asks the one question a wrong long-context implementation cannot
answer: a code is stated in the first turn, thousands of tokens of filler
follow, and the last turn asks for it back from a position beyond 8192 -- the
depth every earlier build was fixed at. Getting it right means the keys written
at position ~100 were still there, still at the right address, and still
rotated for the position they were written at.

Progress is flushed as it goes: this run takes tens of minutes and redirecting
it to a file would otherwise buffer every line until it exited, which makes a
slow run and a hung one look identical.

  python applications/gpt-oss/needle_test.py \\
      lean-artifacts/artifacts/GptOssDecode/gptoss_decode.cbor \\
      --bank data/gptoss-bank --context 16384

The filler is prose rather than repetition on purpose: a repeated sentence is
compressible in a way attention finds too easy, and would pass on a cache that
only kept the most recent window.
"""

import argparse
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout import (CAP_MAX, D_CTX, D_INVT, D_SEED, D_LNMINP, D_STOP, D_MAXNEW,
                    D_NPRE, D_NPOST, D_PRE, D_POST, D_TEXT, D_IN_BYTES,
                    D_OUT_TEXT, D_OUT_NGEN, D_OUT_GEN, D_OUT_NTEXT,
                    D_OUT_TEXTTOK, D_OUT_BYTES, D_STARTPOS, TEXT_MAX,
                    check_layout, ln_min_p, acquire_engine_lock)  # noqa: E402

START, END, MESSAGE, RETURN = 200006, 200007, 200008, 200002
ROLE = {"system": 17360, "user": 1428, "assistant": 173781}
SYSTEM_TEXT = ("You are ChatGPT, a large language model trained by OpenAI.\n"
               "Knowledge cutoff: 2024-06\nCurrent date: 2025-06-28\n\n"
               "Reasoning: low\n\n"
               "# Valid channels: analysis, commentary, final. "
               "Channel must be included for every message.")

CODE = "47291"

# Filler that is prose and not a loop.  Thirty-odd distinct sentences, cycled
# with their index attached so no two are identical.
SENTENCES = [
    "The harbour master kept a ledger of every vessel that cleared the bar.",
    "Rain moved across the estuary in sheets that flattened the water.",
    "A surveyor's chain is sixty-six feet, which is why a furlong is ten of them.",
    "The lighthouse ran on a clockwork weight that had to be wound each dusk.",
    "Salt crusted the railings and had to be washed off before it bit in.",
    "Charts were corrected by hand from notices that arrived by post.",
    "The bell buoy could be heard from the town when the wind sat north.",
    "Coal came in by barge and left by cart, and both were weighed twice.",
    "A pilot who missed the tide waited six hours for the next one.",
    "The customs house opened at seven and closed whenever the last ship was clear.",
    "Fog signals were sounded by a compressor that shook the whole building.",
    "Rope was tarred against rot and stank of it for weeks afterwards.",
    "The tide tables were printed a year ahead and were rarely wrong.",
    "Fishing boats returned before dawn so the catch could go on the early train.",
    "A gale in the autumn took the roof off the net store and nobody replaced it.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--filler-turns", type=int, default=2,
                    help="turns of filler between planting the code and asking")
    args = ap.parse_args()
    assert 0 < args.context <= CAP_MAX

    acquire_engine_lock('needle_test.py')
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

    def turn(text, pre, post, max_new, start):
        raw = text.encode("utf-8")
        assert len(raw) < TEXT_MAX, f"{len(raw)} bytes"
        buf = bytearray(D_IN_BYTES)
        struct.pack_into("<III", buf, 0, 0, 0, 1)
        struct.pack_into("<I", buf, 12, len(raw))
        for off, p in paths.items():
            buf[off:off + len(p)] = p
        struct.pack_into("<f", buf, D_INVT, 0.0)             # greedy: reproducible
        struct.pack_into("<I", buf, D_SEED, 1)
        struct.pack_into("<f", buf, D_LNMINP, ln_min_p(0.0))
        struct.pack_into("<I", buf, D_STARTPOS, start)
        struct.pack_into("<I", buf, D_CTX, args.context)
        struct.pack_into("<II", buf, D_STOP, RETURN, max_new)
        struct.pack_into("<II", buf, D_NPRE, len(pre), len(post))
        if pre:
            struct.pack_into(f"<{len(pre)}I", buf, D_PRE, *pre)
        if post:
            struct.pack_into(f"<{len(post)}I", buf, D_POST, *post)
        buf[D_TEXT:D_TEXT + len(raw)] = raw
        out = bytearray(D_OUT_BYTES)
        base.execute("main", bytes(buf), out)
        b = bytes(out)
        n, _ = struct.unpack_from("<Ii", b, 0)
        reply = b[D_OUT_TEXT:D_OUT_TEXT + n].decode("utf-8", "replace")
        (ngen,) = struct.unpack_from("<I", b, D_OUT_NGEN)
        (ntext,) = struct.unpack_from("<I", b, D_OUT_NTEXT)
        return reply, ntext, ngen

    def filler(n_sentences, tag):
        return " ".join(f"({tag}.{i}) {SENTENCES[i % len(SENTENCES)]}"
                        for i in range(n_sentences))

    print(f"  context {args.context}; filler turns {args.filler_turns}", flush=True)
    print("  first call reads 12.9 GiB off disk and pins 9.5 GiB", flush=True)
    t0 = time.time()
    pos = 0

    _r, nt, ng = turn(SYSTEM_TEXT, [START, ROLE["system"], MESSAGE], [END], 1, 0)
    pos = 3 + nt + 1
    print(f"  system turn: {pos} in context ({time.time()-t0:.0f} s)", flush=True)

    user_pre = [START, ROLE["user"], MESSAGE]
    user_post = [END, START, ROLE["assistant"]]

    plant = (f"Please remember this for later: the access code is {CODE}. "
             f"I will ask you for it at the end. Reply with just 'noted'.\n\n"
             + filler(90, "a"))
    _r, nt, ng = turn(plant, user_pre, user_post, 24, pos)
    pos += len(user_pre) + nt + len(user_post) + ng
    print(f"  planted the code: {pos} in context", flush=True)

    for k in range(args.filler_turns):
        msg = ("Here is more background; no reply needed beyond 'ok'.\n\n"
               + filler(220, f"b{k}"))
        _r, nt, ng = turn(msg, user_pre, user_post, 16, pos)
        pos += len(user_pre) + nt + len(user_post) + ng
        print(f"  filler turn {k + 1}: {pos} in context", flush=True)

    ask = "What was the access code I gave you at the start? Answer with just the number."
    reply, nt, ng = turn(ask, user_pre, user_post, 200, pos)
    pos += len(user_pre) + nt + len(user_post) + ng

    mark = "<|channel|>final<|message|>"
    answer = reply.split(mark, 1)[1] if mark in reply else reply
    for e in ("<|return|>", "<|end|>"):
        answer = answer.split(e, 1)[0]
    answer = answer.strip()
    print(f"\n  asked at position {pos}, {time.time()-t0:.0f} s total")
    print(f"  answer: {answer!r}")

    if pos <= 8192:
        print(f"\nRESULT : INCONCLUSIVE  (the question was asked at {pos}, which "
              f"is inside the 8192 every earlier build had; raise --filler-turns)")
        return 1
    if CODE not in answer:
        print(f"\nRESULT : FAIL  (the code {CODE} was stated at position ~100 and "
              f"was not recalled at {pos}; the cache past 8192 is not being "
              f"read back correctly)")
        return 1
    print(f"\nRESULT : OK  (recalled {CODE} from position ~100 while at {pos}, "
          f"which is {pos - 8192} beyond the old fixed cache)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
