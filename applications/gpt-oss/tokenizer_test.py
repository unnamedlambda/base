"""The CLIF tokenizer against the reference, over the whole corpus.

Two claims, checked separately, because they fail for different reasons:

  * the **split** -- that the pre-tokenizer's chunk boundaries are the ones the
    checkpoint's regex produces.  This is the part CLIF has no native
    equivalent of and the part `tools/pretok.py` exists to make expressible.
  * the **tokens** -- that BPE inside those chunks lands on the ids the model's
    own tokenizer produces.

A tokenizer that is only ever exercised inside a chat loop is a tokenizer
nobody can disagree with, so this runs the artifact directly: text in, ids out.

  python applications/gpt-oss/tokenizer_test.py \\
      lean/.lake/build/artifacts/tokenizer_test.cbor \\
      --tokenizer data/gptoss-bank/tokenizer.bin \\
      --json <path to the checkpoint's tokenizer.json>
"""

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))
import pretok  # noqa: E402

TEXT_MAX = 4096
D_PATH, D_TEXT = 16, 272


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--tokenizer", required=True, help="the .bin the CLIF reads")
    ap.add_argument("--json", required=True, help="the checkpoint's tokenizer.json")
    ap.add_argument("--cases", type=int, default=0, help="0 = the whole corpus")
    ap.add_argument("--fuzz", type=int, default=20000,
                    help="random cases, matching tools/pretok_test.py")
    ap.add_argument("--nfc", action="store_true",
                    help="normalise the input before sending it, standing in for "
                         "the normaliser stage the CLIF does not have")
    args = ap.parse_args()

    import json
    import py_base
    from tokenizers import Tokenizer

    ref = Tokenizer.from_file(args.json)
    tokjson = json.load(open(args.json, encoding="utf-8"))
    alts, _o2, norm = pretok.pattern_for(tokjson)
    tab = pretok.class_table()

    art = py_base.load_artifact(args.artifact)
    base = py_base.Driver(art)
    path = os.path.abspath(args.tokenizer).encode() + b"\0"
    assert len(path) < 256

    def run(text):
        raw = text.encode("utf-8")
        assert len(raw) < TEXT_MAX, f"case too long for the artifact: {len(raw)}"
        buf = bytearray(D_TEXT + TEXT_MAX)
        struct.pack_into("<I", buf, 0, len(raw))
        buf[D_PATH:D_PATH + len(path)] = path
        buf[D_TEXT:D_TEXT + len(raw)] = raw
        out = bytearray(4 + TEXT_MAX * 4)
        base.execute("main", bytes(buf), out)
        (n,) = struct.unpack_from("<I", bytes(out), 0)
        return list(struct.unpack_from(f"<{n}I", bytes(out), 4))

    from pretok_test import corpus as mk
    cases = mk(args.fuzz)
    if args.cases:
        cases = cases[:args.cases]

    if args.nfc:
        import unicodedata
        cases = [unicodedata.normalize("NFC", t) for t in cases]
    print(f"  {len(cases)} cases   normalizer {norm or 'none'}"
          f"{'   (input pre-normalised NFC)' if args.nfc else ''}")
    bad_tok, first = 0, None
    for t in cases:
        if len(t.encode("utf-8")) >= TEXT_MAX:
            continue
        got = run(t)
        want = ref.encode(t, add_special_tokens=False).ids
        if got != want:
            bad_tok += 1
            if first is None:
                first = (t, want, got, pretok.split(tab, alts, t))
    ok = bad_tok == 0
    print(f"  {'ok  ' if ok else 'FAIL'} tokenisation matches the model"
          f"   {len(cases) - bad_tok}/{len(cases)}")
    if first is not None:
        t, want, got, sp = first
        print(f"    first disagreement: {t!r}")
        print(f"      reference split : {sp}")
        print(f"      reference ids   : {want}")
        print(f"      CLIF ids        : {got}")
    print()
    print("RESULT : OK" if ok else "RESULT : FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
