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

Sampling is on by default at temperature 1.0, which is what this checkpoint is
meant to be run at. Greedy decoding loops on anything open-ended -- ask a
reasoning model for a poem at temperature zero and it will repeat one sentence
until the budget runs out. `/temp 0` is there when a reproducible run matters
more than a good one.

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
from layout import (TEXT_MAX, TMPL_MAX, D_INVT, D_SEED, D_LNMINP, D_STOP, D_MAXNEW,
                    D_NPRE, D_PRE, D_POST, D_TEXT, D_IN_BYTES, D_OUT_TEXT,
                    D_OUT_NGEN, D_OUT_GEN, D_OUT_NTEXT, D_OUT_TEXTTOK,
                    D_OUT_BYTES, D_STARTPOS, check_layout, ln_min_p,
                    acquire_engine_lock)

# Harmony's control tokens. Constants of the chat format, and the only ids this
# program names: everything else it handles is bytes in or ids out.
START, END, MESSAGE, CHANNEL, RETURN = 200006, 200007, 200008, 200005, 200002
# the channel a user-facing answer goes in; the others are the model thinking
FINAL = 17196
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
    # 1.0, not 0. Greedy decoding of a reasoning model walks into a loop on
    # anything open-ended -- ask it for a poem at temperature zero and the
    # analysis channel repeats one sentence until the budget runs out. The
    # checkpoint is meant to be sampled from; `/temp 0` is still there for a
    # reproducible run.
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-p", type=float, default=0.02,
                    help="drop tokens below this fraction of the most "
                         "likely one's probability; 0 keeps the whole tail")
    ap.add_argument("-n", "--max-new", type=int, default=2048)
    ap.add_argument("--show-analysis", action="store_true",
                    help="print the model's reasoning as well as its answer")
    args = ap.parse_args()

    # before anything allocates: two of these pin 9.48 GiB each
    acquire_engine_lock('cli.py')

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

    state = {"reasoning": args.reasoning, "temp": args.temp, "seed": args.seed,
             "min_p": args.min_p, "pos": 0}

    def turn(text, pre, post, max_new=None, start=0):
        raw = text.encode("utf-8")
        if len(raw) >= TEXT_MAX:
            return None, None, f"message too long: {len(raw)} bytes"
        if start + len(pre) + len(post) >= TMPL_MAX:
            return None, None, (f"conversation is {start} tokens and the key "
                                f"cache holds {TMPL_MAX}; /new to start over")
        buf = bytearray(D_IN_BYTES)
        struct.pack_into("<III", buf, 0, 0, 0, 1)
        struct.pack_into("<I", buf, 12, len(raw))
        for off, p in paths.items():
            buf[off:off + len(p)] = p
        inv_t = 0.0 if state["temp"] <= 0 else 1.0 / state["temp"]
        struct.pack_into("<f", buf, D_INVT, inv_t)
        struct.pack_into("<I", buf, D_SEED, state["seed"] & 0xFFFFFFFF)
        struct.pack_into("<f", buf, D_LNMINP, ln_min_p(state["min_p"]))
        struct.pack_into("<I", buf, D_STARTPOS, start)
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
                             max_new=1, start=0)
        return (None, err) if err else (ids_[0], None)

    def open_conversation():
        """Put the system turn at position zero and remember how long it is.

        Only the length is kept. The cache holds the conversation itself, and
        the point of `start` is that it never has to be sent again."""
        body, err = tokenize_only(
            SYSTEM_TEXT.format(effort=state["reasoning"]), "system")
        if err:
            return err
        state["pos"] = 3 + len(body) + 1
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

        pre = [START, ROLE["user"], MESSAGE]
        post = [END, START, ROLE["assistant"]]
        t0 = time.time()
        reply, ids_, err = turn(line, pre, post, start=state["pos"])
        dt = time.time() - t0
        if err:
            print(f"  {err}\n")
            continue
        txt_ids, gen_ids = ids_
        state["pos"] += len(pre) + len(txt_ids) + len(post) + len(gen_ids)

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
            # **The thinking ran over. Ask for the answer anyway.**
            #
            # Harmony puts the reply in the `final` channel, and a turn that
            # spends its whole budget in `analysis` never opens one -- so the
            # user gets the model's reasoning and no reply, which is the one
            # outcome nobody wanted. Opening the channel for it costs a
            # continuation from where it stopped, which is cheap now that a
            # turn does not replay the transcript, and the model has already
            # done the work the answer is made of.
            forced = [END, START, ROLE["assistant"], CHANNEL, FINAL, MESSAGE]
            reply2, ids2, err2 = turn("", forced, [], max_new=256,
                                      start=state["pos"])
            if err2 is None:
                _t2, gen2 = ids2
                state["pos"] += len(forced) + len(gen2)
                answer = reply2
                for e in ("<|return|>", "<|end|>"):
                    answer = answer.split(e, 1)[0]
                print(f"\ngpt> {answer.strip()}")
                print(f"     [thought for {len(gen_ids)} tokens without "
                      f"answering; the reply above was asked for directly]")
                dt += 0.0
                print(f"     [{len(gen_ids) + len(gen2)} tokens, {dt:.1f}s, "
                      f"{state['pos']} in context]\n")
                continue
            am = "<|channel|>analysis<|message|>"
            think = reply.split(am, 1)[1] if am in reply else reply
            think = think.split("<|end|>", 1)[0].strip()
            tail = think[-300:] if len(think) > 300 else think
            print(f"\ngpt> (no answer, and asking directly failed: {err2})")
            if tail:
                print(f"     ...{tail}")
        else:
            print(f"\ngpt> {answer.strip()}")
        print(f"     [{len(gen_ids)} tokens, {dt:.1f}s, "
              f"{len(gen_ids)/max(dt,1e-9):.0f} tok/s, "
              f"{state['pos']} in context]\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
