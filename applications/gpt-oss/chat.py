"""A chat turn, entirely inside the artifact.

Text goes in and text comes out. Between them -- the tokenizer, the
pre-tokenizer's split, BPE, the prefill, twenty-four layers a token, the expert
cache, the head, and the detokenizer -- nothing runs here. This driver picks
the file paths, writes the harmony template as a list of ids, and prints what
comes back.

The template is data on purpose. Harmony's control tokens are not text: BPE
over the literal `<|start|>` yields its pieces rather than the special id, so
they cannot arrive through the tokenizer, and putting one checkpoint's chat
format inside the model program would be the wrong place for it.

  python applications/gpt-oss/chat.py <artifact.json> --bank data/gptoss-bank \\
      --prompt "What is the capital of France?"

The first call reads 12.9 GiB off disk and pins 9.5 GiB. Do not run this beside
tools/check.sh, and nothing memory-hungry may share the process -- the pin
cannot be swapped or reclaimed.
"""

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout import (TEXT_MAX, TMPL_MAX, CAP_DEFAULT, CAP_MAX, D_CTX, D_INVT, D_SEED, D_LNMINP, D_STOP, D_MAXNEW,
                    D_NPRE, D_PRE,
                    D_POST, D_TEXT, D_IN_BYTES, D_OUT_TEXT, D_OUT_NGEN,
                    D_OUT_GEN, D_OUT_BYTES, check_layout, ln_min_p, acquire_engine_lock)

# Harmony's system turn, in the shape the checkpoint documents.  The channel
# list is not decoration: without it the model opens with a channel name it
# then has to invent, and greedy decoding walks straight into a loop.
# `Reasoning:` is not a knob this engine implements -- it is a line the model
# was trained to condition on, so how long it thinks is set by a string in the
# system turn and by nothing else.  There is no separate reasoning loop to add.
def system_text(effort):
    return ("You are ChatGPT, a large language model trained by OpenAI.\n"
            "Knowledge cutoff: 2024-06\n"
            "Current date: 2025-06-28\n\n"
            f"Reasoning: {effort}\n\n"
            "# Valid channels: analysis, commentary, final. "
            "Channel must be included for every message.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--reasoning", choices=("low", "medium", "high"),
                    default="medium",
                    help="how long the analysis channel runs; a line in the "
                         "system turn, which is the only place it lives")
    ap.add_argument("--system", default=None)
    ap.add_argument("--developer",
                    default="Answer the user directly and briefly.")
    ap.add_argument("-n", "--max-new", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="0 is greedy; anything else draws from the tempered "
                         "softmax by the Gumbel-max trick, on the device")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-p", type=float, default=0.02,
                    help="drop tokens below this fraction of the most "
                         "likely one's probability; 0 keeps the whole tail")
    ap.add_argument("--context", type=int, default=CAP_DEFAULT,
                    help=f"positions the full-attention layers keep keys for "
                         f"(max {CAP_MAX})")
    ap.add_argument("--raw", action="store_true",
                    help="no template: send the prompt text alone")
    args = ap.parse_args()

    import time
    # before anything allocates: two of these pin 9.48 GiB each
    acquire_engine_lock('chat.py')

    import py_base
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download

    # only to resolve the template's ids and to name the stop token; no text
    # this program sends is tokenised here
    ref = Tokenizer.from_file(hf_hub_download("openai/gpt-oss-20b", "tokenizer.json"))

    def ids(s):
        return ref.encode(s, add_special_tokens=False).ids

    if args.raw:
        pre, post, stop = [], [], 199999
    else:
        system = args.system or system_text(args.reasoning)
        pre = ids("<|start|>system<|message|>" + system
                  + "<|end|><|start|>developer<|message|># Instructions\n\n"
                  + args.developer
                  + "<|end|><|start|>user<|message|>")
        post = ids("<|end|><|start|>assistant")
        stop = ref.token_to_id("<|return|>")
        if stop is None:
            stop = 199999
    assert len(pre) <= TMPL_MAX and len(post) <= TMPL_MAX, "template too long"
    # The prompt has to leave room for the reply: the full-attention layers keep
    # `CAP_FULL` positions, and the engine stops there whatever was asked for.
    assert 0 < args.context <= CAP_MAX, f"--context must be 1..{CAP_MAX}"
    assert len(pre) + len(post) + args.max_new <= args.context, (
        f"template ({len(pre) + len(post)}) plus -n {args.max_new} exceeds the "
        f"{args.context}-position key cache")

    import json
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

    raw = args.prompt.encode("utf-8")
    assert len(raw) < TEXT_MAX, "prompt too long"
    buf = bytearray(D_IN_BYTES)
    struct.pack_into("<III", buf, 0, 0, 0, 1)          # token, position, mode 1
    struct.pack_into("<I", buf, 12, len(raw))
    for off, p in paths.items():
        buf[off:off + len(p)] = p
    inv_t = 0.0 if args.temp <= 0 else 1.0 / args.temp
    struct.pack_into("<f", buf, D_INVT, inv_t)
    struct.pack_into("<I", buf, D_SEED, args.seed & 0xFFFFFFFF)
    struct.pack_into("<f", buf, D_LNMINP, ln_min_p(args.min_p))
    struct.pack_into("<II", buf, D_STOP, stop, args.max_new)
    struct.pack_into("<I", buf, D_CTX, args.context)
    struct.pack_into("<II", buf, D_NPRE, len(pre), len(post))
    if pre:
        struct.pack_into(f"<{len(pre)}I", buf, D_PRE, *pre)
    if post:
        struct.pack_into(f"<{len(post)}I", buf, D_POST, *post)
    buf[D_TEXT:D_TEXT + len(raw)] = raw

    print(f"  prompt: {args.prompt!r}   reasoning: {args.reasoning}"
          f"   temp: {args.temp}{'' if args.temp <= 0 else f' seed {args.seed}'}")
    print(f"  template: {len(pre)} + text + {len(post)} tokens, stop {stop}")
    print("  first call reads 12.9 GiB off disk and pins 9.5 GiB; this takes a while")
    out = bytearray(D_OUT_BYTES)
    t0 = time.time()
    base.execute_into(art.main, bytes(buf), out)
    dt = time.time() - t0
    b = bytes(out)
    n, miss = struct.unpack_from("<Ii", b, 0)
    text = b[D_OUT_TEXT:D_OUT_TEXT + n].decode("utf-8", "replace")
    (ngen,) = struct.unpack_from("<I", b, D_OUT_NGEN)
    gen = list(struct.unpack_from(f"<{ngen}I", b, D_OUT_GEN))
    print(f"  turn took {dt:.1f}s   {ngen} tokens, {n} bytes   misses {miss}")
    # The ids come back beside the text so the detokenizer can be checked
    # without running the model again.
    want = ref.decode(gen, skip_special_tokens=False)
    # Compare decoded strings, not bytes against a string.  A token boundary
    # can split a multi-byte character, and the reference replaces the invalid
    # bytes with U+FFFD while this returns the true bytes -- so a byte
    # comparison fails on output that is in fact identical, and would keep
    # failing however correct the detokenizer got.
    agree = want == text
    print(f"  detokenised text matches the reference for those ids: {agree}")
    print(f"  ids: {gen}")
    print()
    print(text)
    # harmony puts the user-facing answer in the `final` channel; the rest is
    # the model's own reasoning and is not the reply
    # how much of the turn was thinking, which is what `--reasoning` moves
    amark = "<|channel|>analysis<|message|>"
    if amark in text:
        analysis = text.split(amark, 1)[1].split("<|end|>", 1)[0]
        print(f"  analysis: {len(analysis)} chars before the first <|end|>")
    mark = "<|channel|>final<|message|>"
    got_final = mark in text
    if got_final:
        final = text.split(mark, 1)[1]
        for end in ("<|return|>", "<|end|>"):
            final = final.split(end, 1)[0]
        print()
        print(f"  final channel: {final.strip()!r}")
    print()
    if n == 0 or ngen == 0:
        print("RESULT : EMPTY  (the turn produced no text)")
        return 1
    if not agree:
        print("RESULT : DETOK MISMATCH")
        print(f"  reference: {want!r}")
        return 1
    # A turn that ran out of budget inside the analysis channel has produced
    # the model's thinking and no answer.  Saying so is the difference between
    # a test and a log -- `--reasoning high` will do this at a small `-n`.
    if not got_final:
        print(f"RESULT : TRUNCATED  (no final channel in {ngen} tokens; "
              f"raise -n or lower --reasoning)")
        return 1
    print("RESULT : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
