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
from layout import (TEXT_MAX, TMPL_MAX, D_STOP, D_MAXNEW, D_NPRE, D_PRE,
                    D_POST, D_TEXT, D_IN_BYTES, D_OUT_TEXT, D_OUT_NGEN,
                    D_OUT_GEN, D_OUT_BYTES, check_layout)

# Harmony's system turn, in the shape the checkpoint documents.  The channel
# list is not decoration: without it the model opens with a channel name it
# then has to invent, and greedy decoding walks straight into a loop.
SYSTEM = ("You are ChatGPT, a large language model trained by OpenAI.\n"
          "Knowledge cutoff: 2024-06\n"
          "Current date: 2025-06-28\n\n"
          "Reasoning: medium\n\n"
          "# Valid channels: analysis, commentary, final. "
          "Channel must be included for every message.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--system", default=SYSTEM)
    ap.add_argument("--developer",
                    default="Answer the user directly and briefly.")
    ap.add_argument("-n", "--max-new", type=int, default=64)
    ap.add_argument("--raw", action="store_true",
                    help="no template: send the prompt text alone")
    args = ap.parse_args()

    import time
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
        pre = ids("<|start|>system<|message|>" + args.system
                  + "<|end|><|start|>developer<|message|># Instructions\n\n"
                  + args.developer
                  + "<|end|><|start|>user<|message|>")
        post = ids("<|end|><|start|>assistant")
        stop = ref.token_to_id("<|return|>")
        if stop is None:
            stop = 199999
    assert len(pre) <= TMPL_MAX and len(post) <= TMPL_MAX, "template too long"

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
    struct.pack_into("<II", buf, D_STOP, stop, args.max_new)
    struct.pack_into("<II", buf, D_NPRE, len(pre), len(post))
    if pre:
        struct.pack_into(f"<{len(pre)}I", buf, D_PRE, *pre)
    if post:
        struct.pack_into(f"<{len(post)}I", buf, D_POST, *post)
    buf[D_TEXT:D_TEXT + len(raw)] = raw

    print(f"  prompt: {args.prompt!r}")
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
    agree = want.encode("utf-8") == b[D_OUT_TEXT:D_OUT_TEXT + n]
    print(f"  detokenised text matches the reference for those ids: {agree}")
    print(f"  ids: {gen}")
    print()
    print(text)
    # harmony puts the user-facing answer in the `final` channel; the rest is
    # the model's own reasoning and is not the reply
    mark = "<|channel|>final<|message|>"
    if mark in text:
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
    print("RESULT : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
