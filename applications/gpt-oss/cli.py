"""gpt-oss-20b at a prompt.

Loads once — the twelve gigabytes off disk and the 9.48 GiB pin are paid at
start-up, not per turn — and then holds a conversation.

Nothing here tokenizes. The user's text goes to the artifact as bytes and comes
back as ids, which is the whole point: a client that tokenized its own history
would have to agree with the model about what a token is, and that agreement is
what a separate tokenizer cannot be held to. The only ids this file knows are
harmony's control tokens, which are constants of the chat format rather than of
the text.

  python applications/gpt-oss/cli.py \\
      lean-artifacts/artifacts/GptOssDecode/gptoss_decode.json \\
      --bank data/gptoss-bank

Then type. `/reasoning high`, `/temp 0.7`, `/new`, `/quit` change things
mid-conversation; anything else is a message.

Run it alone: the pin cannot be swapped or reclaimed, so a second large
allocation in this process takes the machine down rather than failing.
"""

import argparse
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout import (TEXT_MAX, TMPL_MAX, D_INVT, D_SEED, D_STOP, D_MAXNEW,
                    D_NPRE, D_PRE, D_POST, D_TEXT, D_IN_BYTES, D_OUT_TEXT,
                    D_OUT_NGEN, D_OUT_GEN, D_OUT_NTEXT, D_OUT_TEXTTOK,
                    D_OUT_BYTES, check_layout)

# Harmony's control tokens. Constants of the chat format, and the only ids this
# program names: everything else it handles is bytes in or ids out.
START, END, MESSAGE, CHANNEL, RETURN = 200006, 200007, 200008, 200005, 200002
ROLE = {"system": 17360, "developer": 173781, "user": 1428, "assistant": 173781}

SYSTEM_TEXT = ("You are ChatGPT, a large language model trained by OpenAI.\n"
               "Knowledge cutoff: 2024-06\n"
               "Current date: 2025-06-28\n\n"
               "Reasoning: {effort}\n\n"
               "# Valid channels: analysis, commentary, final. "
               "Channel must be included for every message.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--reasoning", choices=("low", "medium", "high"),
                    default="medium")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-n", "--max-new", type=int, default=1024)
    ap.add_argument("--show-analysis", action="store_true",
                    help="print the model's reasoning as well as its answer")
    args = ap.parse_args()

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

    state = {"reasoning": args.reasoning, "temp": args.temp, "seed": args.seed}
    history = []          # the conversation, as ids

    def turn(text, pre, post, max_new=None):
        raw = text.encode("utf-8")
        if len(raw) >= TEXT_MAX:
            return None, None, f"message too long: {len(raw)} bytes"
        if max(len(pre), len(post)) > TMPL_MAX:
            return None, None, (f"conversation too long for the buffers: "
                                f"{len(pre)} ids; /new to start over")
        buf = bytearray(D_IN_BYTES)
        struct.pack_into("<III", buf, 0, 0, 0, 1)
        struct.pack_into("<I", buf, 12, len(raw))
        for off, p in paths.items():
            buf[off:off + len(p)] = p
        inv_t = 0.0 if state["temp"] <= 0 else 1.0 / state["temp"]
        struct.pack_into("<f", buf, D_INVT, inv_t)
        struct.pack_into("<I", buf, D_SEED, state["seed"] & 0xFFFFFFFF)
        struct.pack_into("<II", buf, D_STOP, RETURN,
                         args.max_new if max_new is None else max_new)
        struct.pack_into("<II", buf, D_NPRE, len(pre), len(post))
        if pre:
            struct.pack_into(f"<{len(pre)}I", buf, D_PRE, *pre)
        if post:
            struct.pack_into(f"<{len(post)}I", buf, D_POST, *post)
        buf[D_TEXT:D_TEXT + len(raw)] = raw

        out = bytearray(D_OUT_BYTES)
        base.execute_into(art.main, bytes(buf), out)
        b = bytes(out)
        n, _miss = struct.unpack_from("<Ii", b, 0)
        reply = b[D_OUT_TEXT:D_OUT_TEXT + n].decode("utf-8", "replace")
        (ngen,) = struct.unpack_from("<I", b, D_OUT_NGEN)
        gen = list(struct.unpack_from(f"<{ngen}I", b, D_OUT_GEN))
        (ntext,) = struct.unpack_from("<I", b, D_OUT_NTEXT)
        txt = list(struct.unpack_from(f"<{ntext}I", b, D_OUT_TEXTTOK))
        return reply, (txt, gen), None

    def tokenize_only(text, role):
        """Run a turn purely to get the engine's ids for `text`.

        There is no cheaper way to ask: the tokenizer lives inside the artifact
        and the artifact's entry point is a turn. One generated token is the
        price, and it is paid once per conversation.
        """
        _r, ids_, err = turn(text, [START, ROLE[role], MESSAGE], [END],
                             max_new=1)
        return (None, err) if err else (ids_[0], None)

    def open_conversation():
        body, err = tokenize_only(
            SYSTEM_TEXT.format(effort=state["reasoning"]), "system")
        if err:
            return err
        history[:] = [START, ROLE["system"], MESSAGE] + body + [END]
        return None

    print(f"  gpt-oss-20b   reasoning {state['reasoning']}   "
          f"temp {state['temp']}   max {args.max_new}")
    print("  loading: 12.9 GiB off disk, 9.48 GiB pinned; this takes ~30 s")
    t0 = time.time()
    err = open_conversation()
    if err:
        print(f"  {err}")
        return 1
    print(f"  ready in {time.time()-t0:.0f} s.  /help for commands.\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            if cmd in ("quit", "exit", "q"):
                break
            if cmd == "new":
                open_conversation()
                print("  (conversation cleared)\n")
            elif cmd == "reasoning" and rest in ("low", "medium", "high"):
                state["reasoning"] = rest
                open_conversation()
                print(f"  reasoning {rest}; conversation cleared\n")
            elif cmd == "temp":
                try:
                    state["temp"] = float(rest)
                    print(f"  temperature {state['temp']}"
                          f"{' (greedy)' if state['temp'] <= 0 else ''}\n")
                except ValueError:
                    print("  usage: /temp 0.7\n")
            elif cmd == "seed":
                try:
                    state["seed"] = int(rest)
                    print(f"  seed {state['seed']}\n")
                except ValueError:
                    print("  usage: /seed 42\n")
            elif cmd == "analysis":
                args.show_analysis = not args.show_analysis
                print(f"  reasoning {'shown' if args.show_analysis else 'hidden'}\n")
            else:
                print("  /new  /reasoning low|medium|high  /temp X  /seed N  "
                      "/analysis  /quit\n")
            continue

        pre = history + [START, ROLE["user"], MESSAGE]
        post = [END, START, ROLE["assistant"]]
        t0 = time.time()
        reply, ids_, err = turn(line, pre, post)
        dt = time.time() - t0
        if err:
            print(f"  {err}\n")
            continue
        txt_ids, gen_ids = ids_
        history[:] = pre + txt_ids + post + gen_ids

        mark = "<|channel|>final<|message|>"
        answer = None
        if mark in reply:
            answer = reply.split(mark, 1)[1]
            for e in ("<|return|>", "<|end|>"):
                answer = answer.split(e, 1)[0]
        if args.show_analysis:
            am = "<|channel|>analysis<|message|>"
            if am in reply:
                think = reply.split(am, 1)[1].split("<|end|>", 1)[0]
                print(f"\n  [thinking] {think.strip()}")
        if answer is None:
            print(f"\ngpt> (no final channel in {len(gen_ids)} tokens; "
                  f"raise -n or lower /reasoning)")
        else:
            print(f"\ngpt> {answer.strip()}")
        print(f"     [{len(gen_ids)} tokens, {dt:.1f}s, "
              f"{len(gen_ids)/max(dt,1e-9):.0f} tok/s, "
              f"{len(history)} in context]\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
