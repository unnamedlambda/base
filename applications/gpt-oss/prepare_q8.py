"""Q8_0 dense weights for the decode, from llama.cpp's published GGUF,
appended to the bank's `dense.bin`.

    python applications/gpt-oss/prepare_q8.py <gpt-oss-20b-MXFP4.gguf> --bank data/gptoss-bank

llama.cpp's gpt-oss-20b file stores attention and lm_head as Q8_0: blocks of
32 int8 values and one f16 scale. The decode reads those same values, so its
weights are llama.cpp's, repacked aligned: per tensor its int8 values row by
row, then its scales as f32 (exact from f16), one a block of 32.

The section starts where prepare.py's bf16 layout ends (`Bank.q8_start`)
and holds, in order: for each layer, the fused QKV (the q, k and v rows, as
dense.bin fuses them) values then scales, then o_proj's values then scales;
then lm_head's values then scales. Running it again rewrites the section.

Every tensor is checked against the bank's bf16 copy before it is written:
the same matrix, row for row, within its quantisation step.
"""

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import Bank  # noqa: E402

NL = 24
HH = 2880
QKV = 64 * 64 + 2 * 8 * 64
QO = 64 * 64
VOCAB = 201088
Q8_0 = 8  # ggml's type id
GGUF_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Gguf:
    """The tensors of a GGUF file, by name: (shape, ggml type, offset)."""

    def __init__(self, path):
        self.f = open(path, "rb")
        r = self.f.read
        magic, version, nt, nkv = struct.unpack("<IIQQ", r(24))
        if magic != 0x46554747 or version < 2:
            raise SystemExit(f"{path} is not a GGUF v2+ file")

        def string():
            (n,) = struct.unpack("<Q", r(8))
            return r(n).decode()

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                et, n = struct.unpack("<IQ", r(12))
                return [value(et) for _ in range(n)]
            fmt = GGUF_TYPES[t]
            return struct.unpack("<" + fmt, r(struct.calcsize(fmt)))[0]

        self.kv = {}
        for _ in range(nkv):
            k = string()
            (t,) = struct.unpack("<I", r(4))
            self.kv[k] = value(t)
        self.tensors = {}
        for _ in range(nt):
            name = string()
            (nd,) = struct.unpack("<I", r(4))
            shape = struct.unpack(f"<{nd}Q", r(8 * nd))
            t, off = struct.unpack("<IQ", r(12))
            self.tensors[name] = (shape, t, off)
        align = self.kv.get("general.alignment", 32)
        self.base = (self.f.tell() + align - 1) // align * align

    def q8(self, name):
        """(values int8 [rows, cols], scales f32 [rows, cols / 32])."""
        shape, t, off = self.tensors[name]
        if t != Q8_0:
            raise SystemExit(f"{name} is type {t}, not Q8_0")
        cols, rows = shape
        nb = cols // 32
        self.f.seek(self.base + off)
        raw = np.frombuffer(self.f.read(rows * nb * 34), np.uint8).reshape(rows, nb, 34)
        d = raw[:, :, :2].copy().view(np.float16).reshape(rows, nb).astype(np.float32)
        qs = raw[:, :, 2:].copy().view(np.int8).reshape(rows, cols)
        return qs, d


def check(name, qs, d, ref, chunk=8192):
    """The Q8_0 matrix is the bf16 one: every value within half its block's
    step (the quantisation), plus the f16 scale's and bf16's rounding; `chunk` rows at a time,
    so lm_head never needs more than a few of its rows in floats."""
    for r in range(0, qs.shape[0], chunk):
        q, s, f = qs[r : r + chunk], d[r : r + chunk], ref(r, min(chunk, qs.shape[0] - r))
        deq = (q.reshape(q.shape[0], -1, 32).astype(np.float32) * s[:, :, None]).reshape(q.shape)
        # half a step, the f16 scale's own rounding (times the code it
        # scales), and bf16's rounding of the reference
        step = np.repeat(s, 32, axis=1)
        # (an f16 scale is off by up to half its spacing: 2^-11 of it, or
        # 2^-25 outright where it is subnormal)
        dscale = np.maximum(step * 2.0 ** -11, 2.0 ** -25)
        tol = step * 0.5 + np.abs(q).astype(np.float32) * dscale + np.abs(f) * 2.0 ** -7 + 1e-30
        bad = np.count_nonzero(np.abs(deq - f) > tol)
        if bad:
            raise SystemExit(f"{name}: {bad} values off the bf16 matrix by more than a step (rows {r}…)")


def bf16(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--bank", required=True)
    args = ap.parse_args()
    g = Gguf(args.gguf)
    bank = Bank(args.bank)
    meta, end = bank.meta, bank.q8_start()
    path = os.path.join(args.bank, "dense.bin")
    if os.path.getsize(path) < end:
        raise SystemExit(f"{path} is shorter than prepare.py's layout")
    dense = np.memmap(path, np.uint16, "r", shape=(end // 2,))

    def ref(off, cols):
        """Rows `r … r + n` of the bf16 matrix at byte `off`, `cols` wide."""
        return lambda r, n: bf16(np.asarray(dense[off // 2 + r * cols : off // 2 + (r + n) * cols]).reshape(n, cols))

    out = open(path, "r+b")
    out.truncate(end)
    out.seek(end)
    for layer in range(NL):
        parts = [g.q8(f"blk.{layer}.attn_{x}.weight") for x in "qkv"]
        qs = np.concatenate([p[0] for p in parts])
        d = np.concatenate([p[1] for p in parts])
        lay = meta["dense"]["layers"][layer]
        check(f"layer {layer} qkv", qs, d, ref(lay["qkv_w"], HH))
        out.write(qs.tobytes())
        out.write(d.tobytes())
        qs, d = g.q8(f"blk.{layer}.attn_output.weight")
        check(f"layer {layer} o_proj", qs, d, ref(lay["o_w"], QO))
        out.write(qs.tobytes())
        out.write(d.tobytes())
        print(f"  layer {layer}: checked", flush=True)
    qs, d = g.q8("output.weight")
    check("lm_head", qs, d, ref(meta["dense"]["lm_head"], HH))
    out.write(qs.tobytes())
    out.write(d.tobytes())
    out.close()
    print("  dense.bin: Q8_0 section written")


if __name__ == "__main__":
    main()
