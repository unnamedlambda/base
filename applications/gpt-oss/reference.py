"""gpt-oss-20b in NumPy, reading the converted bank.

The reference every later stage is checked against, and the first thing that
says the *converter* is right: nothing else reads `experts.bin` until the
engine does, so if the de-interleave or the packed byte order were wrong, this
is where it shows.

Reads the bank by memory map rather than loading it.  The expert pool is 9.5
GiB against a machine with 13 free, and the point of the whole design is that
the pool does not have to be resident; a reference that insisted on residency
would not run at all here.

Deliberately literal.  Every step is written the way the model defines it, not
the way the kernels compute it -- separate rows rather than one packed QKV, a
full attention matrix rather than a windowed slice -- because a reference that
shared the implementation's shortcuts would agree with it for the wrong
reasons.  The one exception is documented at `mx_dequant`.

  python applications/gpt-oss/reference.py --bank data/gptoss-bank --prompt "..." -n 16
"""

import argparse
import json
import os
import sys

import numpy as np

FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def mx_dequant(blocks: np.ndarray, scales: np.ndarray, k: int) -> np.ndarray:
    """[rows, k/32, 16] packed codes + [rows, k/32] scale bytes -> [rows, k] f32.

    Vectorised rather than looped, which is the one place this file follows the
    implementation's shape instead of the specification's: the definition is
    per element, and doing it per element over ten million of them in NumPy is
    the difference between seconds and an hour.  The arithmetic is identical --
    a code indexes the sixteen-entry table, a scale byte is the Float32 whose
    exponent field it is.
    """
    rows = blocks.shape[0]
    b = blocks.reshape(rows, -1)
    codes = np.empty((rows, k), dtype=np.uint8)
    codes[:, 0::2] = b & 0xF
    codes[:, 1::2] = b >> 4
    sc = (scales.astype(np.uint32) << 23).view(np.float32)
    return FP4[codes] * np.repeat(sc, 32, axis=1)


def to_bf16(x):
    """Round f32 to bf16 precision, nearest-even, staying in f32 storage.

    The same rounding GptOssKernels.rneBf16 does on the device.  Used to model
    the engine's precision, never to define the model: see `acts` in forward.
    """
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    r = ((u >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


def rms_norm(x, w, eps):
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w).astype(np.float32)


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def swiglu(gate, up, alpha, limit):
    # gate is clamped above only, up on both sides -- that asymmetry is the
    # model's, not a typo.  A very negative gate would overflow exp(-alpha*g),
    # so the sigmoid is written in the branch that keeps the exponent negative.
    g = np.minimum(gate, limit)
    u = np.clip(up, -limit, limit)
    z = alpha * g
    ez = np.exp(-np.abs(z))
    sig = np.where(z >= 0, 1.0 / (1.0 + ez), ez / (1.0 + ez))
    return (g * sig) * (u + 1.0)


class Bank:
    """The converted checkpoint, mapped rather than loaded."""

    def __init__(self, path):
        self.meta = json.load(open(os.path.join(path, "bank.json")))
        m = self.meta
        self.experts = np.memmap(os.path.join(path, "experts.bin"), np.uint8, "r")
        self.dense = np.memmap(os.path.join(path, "dense.bin"), np.uint8, "r")
        self.embed = np.memmap(os.path.join(path, "embed.bin"), np.uint16, "r").reshape(
            m["vocab"], m["hidden"])
        self.H, self.I, self.E = m["hidden"], m["inter"], m["experts"]
        self.HD, self.NQ, self.NKV = m["head_dim"], m["n_q"], m["n_kv"]
        self.L, self.K = m["layers"], m["top_k"]

    def _at(self, off, dtype, shape):
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return self.dense[off:off + n].view(dtype).reshape(shape)

    def f32(self, off, shape):
        return self._at(off, np.float32, shape)

    def bf16(self, off, shape):
        return bf16_to_f32(self._at(off, np.uint16, shape))

    def expert(self, layer, e):
        """One expert's six pieces, dequantised. This is the row a cache miss
        moves, read exactly as the engine reads it."""
        m = self.meta
        rb, r = m["expert_row_bytes"], m["expert_row"]
        base = (layer * self.E + e) * rb
        row = self.experts[base:base + rb]
        H, I, nbH, nbI = self.H, self.I, self.H // 32, self.I // 32
        gu_b = row[r["gu_blocks"]:r["gu_scales"]].reshape(2 * I, nbH, 16)
        gu_s = row[r["gu_scales"]:r["gu_bias"]].reshape(2 * I, nbH)
        gu_bias = row[r["gu_bias"]:r["dn_blocks"]].view(np.float32)
        dn_b = row[r["dn_blocks"]:r["dn_scales"]].reshape(H, nbI, 16)
        dn_s = row[r["dn_scales"]:r["dn_bias"]].reshape(H, nbI)
        dn_bias = row[r["dn_bias"]:].view(np.float32)
        return (mx_dequant(gu_b, gu_s, H), gu_bias, mx_dequant(dn_b, dn_s, I), dn_bias)


def forward(bank, tokens, trace=None, acts="f32"):
    """Logits for the last token of `tokens`, computing the whole sequence.

    No KV cache: this is the reference, and recomputing is what makes it one.

    `acts` selects the activation precision at the three matmuls the engine
    feeds from bf16 buffers (qkv, o-proj, lm_head).  "f32" is the idealisation;
    "bf16" is what the released model and our engine both actually compute in.
    Comparing against both is how a precision gap is told from a defect.
    """
    narrow = to_bf16 if acts == "bf16" else (lambda t: t)
    m = bank.meta
    H, HD, NQ, NKV, K = bank.H, bank.HD, bank.NQ, bank.NKV, bank.K
    T = len(tokens)
    x = bf16_to_f32(np.asarray(bank.embed[tokens])).astype(np.float32)
    cos = bank.f32(m["dense"]["rope_cos"], (m["max_position"], HD // 2))[:T]
    sin = bank.f32(m["dense"]["rope_sin"], (m["max_position"], HD // 2))[:T]
    scale = 1.0 / np.sqrt(HD)

    for l, ent in enumerate(m["dense"]["layers"]):
        # ---- attention ----
        h = rms_norm(x, bank.f32(ent["attn_norm"], (H,)), m["rms_eps"])
        qkv_w = bank.bf16(ent["qkv_w"], (NQ * HD + 2 * NKV * HD, H))
        qkv = narrow(h) @ qkv_w.T + bank.f32(ent["qkv_b"], (NQ * HD + 2 * NKV * HD,))
        q = qkv[:, :NQ * HD].reshape(T, NQ, HD)
        k = qkv[:, NQ * HD:NQ * HD + NKV * HD].reshape(T, NKV, HD)
        v = qkv[:, NQ * HD + NKV * HD:].reshape(T, NKV, HD)

        # RoPE on the interleaved halves, tables precomputed by the converter
        def rope(t):
            a, b = t[..., :HD // 2], t[..., HD // 2:]
            c, s = cos[:, None, :], sin[:, None, :]
            return np.concatenate([a * c - b * s, b * c + a * s], -1).astype(np.float32)

        q, k = rope(q), rope(k)
        # GQA: each key head serves NQ/NKV query heads
        kx = np.repeat(k, NQ // NKV, axis=1)
        vx = np.repeat(v, NQ // NKV, axis=1)
        att = np.einsum("qhd,khd->hqk", q, kx) * scale
        causal = np.triu(np.ones((T, T), bool), 1)
        if "sliding" in ent["type"]:
            # window w means a query attends to the w positions ending at itself
            w = m["sliding_window"]
            idx = np.arange(T)
            causal |= (idx[:, None] - idx[None, :]) >= w
        att = np.where(causal[None], -np.inf, att)
        # the learned sink is an extra logit in the denominator and nowhere else
        sinks = bank.f32(ent["sinks"], (NQ,))
        mx = np.maximum(att.max(-1), sinks[:, None])
        e = np.exp(att - mx[..., None])
        denom = e.sum(-1) + np.exp(sinks[:, None] - mx)
        p = e / denom[..., None]
        o = np.einsum("hqk,khd->qhd", p, vx).reshape(T, NQ * HD)
        x = x + (narrow(o) @ bank.bf16(ent["o_w"], (H, NQ * HD)).T
                 + bank.f32(ent["o_b"], (H,)))
        if trace is not None:
            trace.setdefault("attn_out", {})[l] = x[-1].copy()

        # ---- routed experts ----
        h = rms_norm(x, bank.f32(ent["mlp_norm"], (H,)), m["rms_eps"])
        logits = h @ bank.f32(ent["router_w"], (bank.E, H)).T \
            + bank.f32(ent["router_b"], (bank.E,))
        chosen = np.argsort(-logits, axis=-1)[:, :K]
        gates = softmax(np.take_along_axis(logits, chosen, -1), -1)
        out = np.zeros_like(h)
        for e in np.unique(chosen):
            gu_w, gu_b, dn_w, dn_b = bank.expert(l, int(e))
            rows = np.where(chosen == e)
            if len(rows[0]) == 0:
                continue
            hs = h[rows[0]]
            a = hs @ gu_w.T + gu_b
            act = swiglu(a[:, :bank.I], a[:, bank.I:], m["swiglu_alpha"], m["swiglu_limit"])
            y = act @ dn_w.T + dn_b
            out[rows[0]] += y * gates[rows][:, None]
        x = x + out
        if trace is not None:
            trace.setdefault("layer_out", {})[l] = x[-1].copy()
        print(f"    layer {l+1}/{bank.L}", end="\r", flush=True)

    x = rms_norm(x, bank.f32(m["dense"]["final_norm"], (H,)), m["rms_eps"])
    # lm_head is vocab x H bf16 -- 1.08 GiB on disk, 2.3 GiB widened.  Widening it
    # whole is what makes this box swap, so it goes a slice at a time.
    xf = narrow(x[-1])
    out = np.empty(m["vocab"], np.float32)
    step = 8192
    for i in range(0, m["vocab"], step):
        n = min(step, m["vocab"] - i)
        w = bank.bf16(m["dense"]["lm_head"] + i * H * 2, (n, H))
        out[i:i + n] = w @ xf
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()

    bank = Bank(args.bank)
    from tokenizers import Tokenizer
    tok_path = args.tokenizer
    if tok_path is None:
        from huggingface_hub import hf_hub_download
        tok_path = hf_hub_download("openai/gpt-oss-20b", "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)

    ids = tok.encode(args.prompt, add_special_tokens=False).ids
    print(f"prompt: {args.prompt!r}  ({len(ids)} tokens)")
    out = []
    for i in range(args.n):
        logits = forward(bank, ids)
        nxt = int(np.argmax(logits))
        ids.append(nxt)
        out.append(nxt)
        print(f"  {i+1}/{args.n}: {tok.decode([nxt])!r}                    ")
    print(f"\ncontinuation: {tok.decode(out)!r}")
    print("RESULT : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
