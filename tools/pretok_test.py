"""The pre-tokenizer against the checkpoint's own regex, and end to end.

Two claims, and the second is the one that matters:

  * the table scan splits text exactly where the checkpoint's regex does;
  * splitting there and then running BPE reproduces the checkpoint's own
    tokenizer, token for token.

The first can pass while the second fails -- a normalizer sitting in front of
the split will do it, which is how `normalizer: NFC` was found in Qwen2 after
the splits already agreed 50033/50033.

  python tools/pretok_test.py [--fuzz 20000]
"""

import argparse
import glob
import json
import random
import sys
import unicodedata

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import pretok  # noqa: E402

MODELS = [
    ("Qwen2-0.5B-Instruct", "models--Qwen--Qwen2-0.5B-Instruct"),
    ("gpt-oss-20b", "models--openai--gpt-oss-20b"),
]
HUB = "/home/ulam/.cache/huggingface/hub"


def byte_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def bpe(sym, rank):
    sym = list(sym)
    while len(sym) > 1:
        best, bi = None, -1
        for i in range(len(sym) - 1):
            r = rank.get((sym[i], sym[i + 1]))
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if bi < 0:
            break
        sym[bi:bi + 2] = [sym[bi] + sym[bi + 1]]
    return sym


def corpus(n_fuzz):
    here = __import__("os").path.dirname(__import__("os").path.dirname(
        __import__("os").path.abspath(__file__)))
    docs = []
    for f in ["lean/algorithms/GptOss/Attention.lean", "applications/gpt-oss/reference.py"]:
        try:
            t = open(f"{here}/{f}").read()
        except OSError:
            continue
        docs += [t[i:i + 900] for i in range(0, len(t), 900)]
    docs += ["", " ", "a", "\n\n\n", "   ", "\t\t x", "a\t+b", "  a  b  ",
             "日Z", "ZZ日z", "ᵃB", "ÀÀàà", "9́,", "á", "'''x'''",
             "def f(n):\n    if n < 2:\n        return n\n", "1000000 999 12 1 007",
             "He said IT'S fine, they'll SEE, I'd've thought",
             "emoji \U0001f642 combining é RTL אבג"]
    rnd = random.Random(5)
    alpha = list(" \t\n\rabZÉ0159.,'/+<>#日\U0001f642ᵃ́ǅ")
    docs += ["".join(rnd.choice(alpha) for _ in range(rnd.randrange(0, 40)))
             for _ in range(n_fuzz)]
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzz", type=int, default=20000)
    args = ap.parse_args()

    import regex as re
    from tokenizers import Tokenizer

    tab = pretok.class_table()
    docs = corpus(args.fuzz)
    b2u = byte_to_unicode()
    rc = 0

    for name, repo in MODELS:
        hits = glob.glob(f"{HUB}/{repo}/snapshots/*/tokenizer.json")
        if not hits:
            print(f"  skip {name}: no tokenizer.json in the hub cache")
            continue
        path = hits[0]
        d = json.load(open(path))
        alts, _o2, norm = pretok.pattern_for(path)
        rx = re.compile([e for e in (d["pre_tokenizer"].get("pretokenizers")
                                     or [d["pre_tokenizer"]])
                         if e.get("type") == "Split"][0]["pattern"]["Regex"])
        m = d["model"]
        vocab, mg = m["vocab"], m["merges"]
        rank = ({tuple(x): i for i, x in enumerate(mg)} if isinstance(mg[0], list)
                else {tuple(s.split(" ")): i for i, s in enumerate(mg)})
        tok = Tokenizer.from_file(path)

        split_ok = enc_ok = 0
        for s in docs:
            t = unicodedata.normalize(norm, s) if norm else s
            if rx.findall(t) == pretok.split(tab, alts, t):
                split_ok += 1
            got = [vocab.get(sy, -1) for c in pretok.split(tab, alts, t)
                   for sy in bpe("".join(b2u[b] for b in c.encode()), rank)]
            if got == tok.encode(s, add_special_tokens=False).ids:
                enc_ok += 1

        tables = pretok.emit_tables(alts, tab)
        n = len(docs)
        print(f"  {name}  (normalizer: {norm or 'none'})")
        print(f"    {'ok  ' if split_ok == n else 'FAIL'} splits match the regex        "
              f"{split_ok}/{n}")
        print(f"    {'ok  ' if enc_ok == n else 'FAIL'} tokenisation matches the model  "
              f"{enc_ok}/{n}")
        print(f"         tables for CLIF: {len(tables)/2**20:.2f} MiB")
        if split_ok != n or enc_ok != n:
            rc = 1

    print()
    print("RESULT : OK" if rc == 0 else "RESULT : MISMATCH")
    return rc


if __name__ == "__main__":
    sys.exit(main())
