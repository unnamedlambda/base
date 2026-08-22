"""The tokenizer binary both models load, written once.

The format `Qwen2Common`'s CLIF already reads, with the pre-tokenizer tables
appended.  Appending is safe: every existing offset is computed forward from
the header and the byte pool's length is in it, so a reader that predates this
section cannot see it.  The header's fourth word was spare and now points at
the new section, which is how a reader that *does* know about it finds it
without recomputing anything.

  [n_merges:u32][vocab_size:u32][byte_pool_size:u32][pretok_off:u32]
  byte_init : 256 x u32     byte value -> its single-byte token
  merges    : n x 12        (tok_a, tok_b, result), in rank order
  dec_off   : vocab x u32   into byte_pool
  dec_len   : vocab x u32
  byte_pool : byte_pool_size
  pretok    : the alternation table and the Unicode class table

Nothing here is model-specific: every dimension is in the header, which is why
one CLIF tokenizer serves both checkpoints.  What differs between them is the
alternation table, and that is data.
"""

import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pretok  # noqa: E402


def gpt2_byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_str_to_bytes(s, dec):
    return bytes(dec[c] for c in s if c in dec)


def convert(tok_path, out_path, class_tab=None):
    tok = json.load(open(tok_path, encoding="utf-8"))
    model = tok["model"]
    vocab = model["vocab"]
    raw = model.get("merges", [])
    vocab_size = len(vocab)
    id_to_str = {v: k for k, v in vocab.items()}
    dec = gpt2_byte_decoder()

    byte_init = np.zeros(256, dtype=np.uint32)
    for char, bv in dec.items():
        if char in vocab:
            byte_init[bv] = vocab[char]

    merges = []
    for e in raw:
        a, b = (e if isinstance(e, list) else e.split(" ", 1))[:2] if (
            isinstance(e, list) or len(e.split(" ", 1)) == 2) else (None, None)
        if a is None:
            continue
        ab = a + b
        if a in vocab and b in vocab and ab in vocab:
            merges.append((vocab[a], vocab[b], vocab[ab]))

    # Added tokens sit *outside* model["vocab"] and at higher ids -- harmony's
    # control tokens are all of them.  The decode tables have to cover them or
    # detokenising anything the model emits reads past the end of the table,
    # which is a buffer overrun rather than a wrong answer.  Their bytes are
    # their literal text, not gpt2-encoded, because nothing encoded them.
    added = {int(a["id"]): a["content"] for a in tok.get("added_tokens", [])}
    table_size = max([vocab_size] + [i + 1 for i in added])
    pool = bytearray()
    off = np.zeros(table_size, dtype=np.uint32)
    ln = np.zeros(table_size, dtype=np.uint32)
    for i in range(table_size):
        if i in added:
            b = added[i].encode("utf-8")
        else:
            b = token_str_to_bytes(id_to_str.get(i, ""), dec)
        off[i], ln[i] = len(pool), len(b)
        pool.extend(b)

    alts, _, norm = pretok.pattern_for(tok)
    tables = pretok.emit_tables(alts, class_tab)

    pretok_off = 16 + 1024 + len(merges) * 12 + table_size * 8 + len(pool)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<IIII", len(merges), table_size, len(pool), pretok_off))
        f.write(byte_init.tobytes())
        for a, b, r in merges:
            f.write(struct.pack("<III", a, b, r))
        f.write(off.tobytes())
        f.write(ln.tobytes())
        f.write(bytes(pool))
        f.write(tables)

    mb = os.path.getsize(out_path) / 2 ** 20
    print(f"  tokenizer: {len(merges)} merges, {table_size} decodable "
          f"({vocab_size} bpe + {len(added)} added), "
          f"normalizer {norm or 'none'} -> {out_path} ({mb:.1f} MiB)")
    if norm:
        print(f"  NOTE: this checkpoint specifies {norm}; the CLIF path does not "
              f"normalise yet, so inputs holding decomposed characters will differ")
    return {"n_merges": len(merges), "vocab": table_size,
            "pretok_off": pretok_off, "normalizer": norm}


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
