"""Convert the published gpt-oss-20b checkpoint into the flat bank the engine reads.

Runs in its own process because it imports torch: a PyTorch CUDA context in the
same process makes every cuBLAS call from our runtime fail.  Nothing downstream
of this script imports torch.

## What it writes

Three files and an index, because the three have different lifetimes:

  experts.bin  the routed-expert pool, one contiguous 12.64 MiB row per
               (layer, expert).  Host-resident and pinned at run time; a decode
               step reads four rows of it per layer and a miss streams one row
               over PCIe, so a row is the unit everything here is arranged
               around.
  dense.bin    everything that stays on the GPU: attention, the norms, the
               router, the sinks, lm_head.  bf16 where a contraction reads it,
               f32 where a proven kernel does.
  embed.bin    the token embedding, alone, because it is the one large tensor
               that is *not* pinned: 11.5 KB of it is read per token, on the
               host, so it is mmap'd and left to the page cache.  Pinning it
               would cost a gigabyte of the pool's budget to no purpose.
  bank.json    offsets, shapes and the geometry, so the reader and the engine
               agree about the layout without either restating it.

## Two transforms, both declared

**De-interleaving.** The checkpoint stores the gate and up projections
interleaved by row: row 2i is the gate for intermediate i and row 2i+1 is the
up.  The kernels want the gate rows first and the up rows after, so that one
row index addresses both.  Undoing the interleave offline costs nothing at run
time.

**bf16 narrowing.** Activations reaching a dense contraction are bf16 because
cuBLAS refuses a mixed operand pair (measured, not assumed), so the rounding is
part of the pipeline rather than an artefact of the weights.  The rounding used
here -- nearest, ties to even -- is the same one the engine applies to the
activation, and `bf16_round` below is the only definition of it.

Both are on the ledger as `ConverterFidelity`: this script is checked against
the source tensors, not proven, and a kernel that is exactly right about the
wrong bytes computes the wrong model.

  python applications/gpt-oss/prepare.py --out data/gptoss-bank
"""

import argparse
import json
import os
import sys
import time

import numpy as np

MXGROUP = 32          # elements per microscaling block
BYTES_PER_BLOCK = 16  # two four-bit codes to a byte


def bf16_round(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bit patterns, nearest with ties to even.

    Truncation would be cheaper and is wrong: it biases every weight toward
    zero, and over 2880 accumulated products that bias does not cancel.
    """
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (b >> 16) & 1
    return ((b + np.uint32(0x7FFF) + lsb) >> 16).astype(np.uint16)


def to_bf16(t) -> np.ndarray:
    """A torch tensor to bf16 bit patterns, via f32 so the rounding is ours."""
    return bf16_round(t.float().numpy())


def to_f32(t) -> np.ndarray:
    return t.float().numpy().astype(np.float32)


class Writer:
    """Append-only writer that records where each region landed."""

    def __init__(self, path):
        self.f = open(path, "wb")
        self.off = 0

    def write(self, arr: np.ndarray) -> int:
        at = self.off
        b = np.ascontiguousarray(arr).tobytes()
        self.f.write(b)
        self.off += len(b)
        return at

    def close(self):
        self.f.close()
        return self.off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b",
                    help="HF repo id or a local snapshot directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None,
                    help="convert only the first N layers (for a fast check)")
    args = ap.parse_args()

    import torch  # noqa: F401  (only here; see the module docstring)
    from safetensors import safe_open

    if os.path.isdir(args.model):
        snap = args.model
    else:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(args.model, allow_patterns=["*.json"])

    cfg = json.load(open(os.path.join(snap, "config.json")))
    L = args.layers or cfg["num_hidden_layers"]
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    E = cfg["num_local_experts"]
    V = cfg["vocab_size"]
    HD = cfg["head_dim"]
    NQ = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    layer_types = cfg["layer_types"]

    index = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(name):
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(snap, shard), framework="pt")
        return handles[shard].get_tensor(name)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    print(f"gpt-oss-20b -> {args.out}   ({L} layers, {E} experts, H={H}, I={I})")

    # -- the expert pool -----------------------------------------------------
    # One row per (layer, expert), the six pieces in a fixed order.  The row is
    # the unit a cache miss transfers, so nothing in it is addressed from
    # elsewhere and its internal offsets are constants.
    gu_blk_b = 2 * I * (H // MXGROUP) * BYTES_PER_BLOCK
    gu_scl_b = 2 * I * (H // MXGROUP)
    gu_bias_b = 2 * I * 4
    dn_blk_b = H * (I // MXGROUP) * BYTES_PER_BLOCK
    dn_scl_b = H * (I // MXGROUP)
    dn_bias_b = H * 4
    row_bytes = gu_blk_b + gu_scl_b + gu_bias_b + dn_blk_b + dn_scl_b + dn_bias_b

    ew = Writer(os.path.join(args.out, "experts.bin"))
    # Interleave -> [gate rows; up rows].  The checkpoint's row 2i is the gate
    # for intermediate i and row 2i+1 the up; the kernels index both by i.
    deint = np.concatenate([np.arange(0, 2 * I, 2), np.arange(1, 2 * I, 2)])
    nan_scales = 0
    for l in range(L):
        p = f"model.layers.{l}.mlp.experts."
        gu_blocks = get(p + "gate_up_proj_blocks").numpy()   # [E, 2I, H/32, 16]
        gu_scales = get(p + "gate_up_proj_scales").numpy()   # [E, 2I, H/32]
        gu_bias = to_f32(get(p + "gate_up_proj_bias"))       # [E, 2I]
        dn_blocks = get(p + "down_proj_blocks").numpy()      # [E, H, I/32, 16]
        dn_scales = get(p + "down_proj_scales").numpy()
        dn_bias = to_f32(get(p + "down_proj_bias"))
        # A scale byte of 255 is the format's reserved NaN; the kernels decode
        # a byte by shifting it into an exponent field and do not test for it,
        # so if one ever appeared it would silently poison a row.
        nan_scales += int((gu_scales == 255).sum() + (dn_scales == 255).sum())
        for e in range(E):
            ew.write(gu_blocks[e][deint])
            ew.write(gu_scales[e][deint])
            ew.write(gu_bias[e][deint])
            ew.write(dn_blocks[e])
            ew.write(dn_scales[e])
            ew.write(dn_bias[e])
        print(f"  layer {l+1:2d}/{L} experts", end="\r", flush=True)
    experts_bytes = ew.close()
    assert experts_bytes == L * E * row_bytes, (experts_bytes, L * E * row_bytes)
    print(f"  experts.bin  {experts_bytes/2**30:6.2f} GiB "
          f"({L*E} rows x {row_bytes/2**20:.2f} MiB)          ")

    # -- everything that stays on the GPU ------------------------------------
    dw = Writer(os.path.join(args.out, "dense.bin"))
    dense = {"layers": []}
    for l in range(L):
        a = f"model.layers.{l}.self_attn."
        # q, k and v are one contraction against one activation, so they are
        # stored as one matrix: three launches become one.
        qkv_w = np.concatenate([get(a + "q_proj.weight").float().numpy(),
                                get(a + "k_proj.weight").float().numpy(),
                                get(a + "v_proj.weight").float().numpy()], axis=0)
        qkv_b = np.concatenate([to_f32(get(a + "q_proj.bias")),
                                to_f32(get(a + "k_proj.bias")),
                                to_f32(get(a + "v_proj.bias"))])
        ent = {
            "attn_norm": dw.write(to_f32(get(f"model.layers.{l}.input_layernorm.weight"))),
            "qkv_w": dw.write(bf16_round(qkv_w)),
            "qkv_b": dw.write(qkv_b),
            "sinks": dw.write(to_f32(get(a + "sinks"))),
            "o_w": dw.write(to_bf16(get(a + "o_proj.weight"))),
            "o_b": dw.write(to_f32(get(a + "o_proj.bias"))),
            "mlp_norm": dw.write(to_f32(get(f"model.layers.{l}.post_attention_layernorm.weight"))),
            # the router stays f32: it is 32x2880, it decides which experts are
            # read, and a rounding there changes which weights a token sees
            "router_w": dw.write(to_f32(get(f"model.layers.{l}.mlp.router.weight"))),
            "router_b": dw.write(to_f32(get(f"model.layers.{l}.mlp.router.bias"))),
            "type": layer_types[l],
        }
        dense["layers"].append(ent)
        print(f"  layer {l+1:2d}/{L} dense  ", end="\r", flush=True)
    dense["final_norm"] = dw.write(to_f32(get("model.norm.weight")))
    dense["lm_head"] = dw.write(to_bf16(get("lm_head.weight")))

    # YaRN rotation tables, precomputed on the host: the engine's RoPE kernel
    # reads cos and sin and knows nothing about how they were scaled.
    rs = cfg["rope_scaling"]
    d = HD
    theta = cfg["rope_theta"]
    inv = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
    orig = rs["original_max_position_embeddings"]
    factor, bfast, bslow = rs["factor"], rs["beta_fast"], rs["beta_slow"]

    def corr_dim(rot):
        return d * np.log(orig / (rot * 2 * np.pi)) / (2 * np.log(theta))

    # The ramp goes from extrapolation to interpolation, in that order.
    #
    # Low dimensions rotate fast and have seen their whole period inside the
    # original context, so they are left alone; high dimensions rotate slowly,
    # have not, and are the ones scaled by 1/factor.  Blending them the other
    # way round is not a subtle error: dimension zero comes out thirty-two
    # times too slow, every position in the model is mis-encoded, and the model
    # emits locally plausible text that does not track its own prompt.
    #
    # `truncate` is false for this checkpoint, so the correction range is not
    # rounded to integers -- it is clamped to the dimension instead.
    lo = max(corr_dim(bfast), 0.0)
    hi = min(corr_dim(bslow), float(d - 1))
    ramp = np.clip((np.arange(d // 2, dtype=np.float64) - lo) / max(hi - lo, 1e-3), 0, 1)
    inv_scaled = (inv / factor) * ramp + inv * (1 - ramp)
    attn_factor = 0.1 * np.log(factor) + 1.0
    # **Cross-check against the library's own YaRN, not against ourselves.**
    #
    # This is the one place a converter error cannot be caught downstream:
    # `reference.py` reads these tables too, so an engine that matches the
    # reference matches it on identical wrong numbers.  Inverting this blend
    # made dimension zero rotate thirty-two times too slowly, and the model
    # produced locally plausible text that did not track its own prompt --
    # through a verification that compared the engine to the reference and
    # agreed to seven parts in a thousand.
    try:
        from transformers import AutoConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        hc = AutoConfig.from_pretrained(args.model)
        want, want_af = ROPE_INIT_FUNCTIONS["yarn"](hc, device="cpu")
        ref = np.asarray(want, np.float64)
        rel = float(np.abs(np.asarray(inv_scaled, np.float64) / ref - 1.0).max())
        # their inv_freq is float32, so the floor here is rounding, not method
        assert rel < 1e-5, (
            f"rope tables disagree with transformers' yarn by {rel:.3e}: "
            f"ours[0]={inv_scaled[0]:.6g} theirs[0]={ref[0]:.6g}")
        assert abs(attn_factor / float(want_af) - 1.0) < 1e-9
        print(f"  rope: matches transformers' yarn to {rel:.1e}")
    except ImportError:
        print("  rope: transformers not importable; tables NOT cross-checked")

    maxpos = cfg["max_position_embeddings"]
    pos = np.arange(maxpos, dtype=np.float64)[:, None]
    ang = pos * inv_scaled[None, :]
    dense["rope_cos"] = dw.write((np.cos(ang) * attn_factor).astype(np.float32))
    dense["rope_sin"] = dw.write((np.sin(ang) * attn_factor).astype(np.float32))
    dense_bytes = dw.close()
    print(f"  dense.bin    {dense_bytes/2**30:6.2f} GiB                          ")

    # -- the embedding, on its own -------------------------------------------
    emb = to_bf16(get("model.embed_tokens.weight"))
    with open(os.path.join(args.out, "embed.bin"), "wb") as f:
        f.write(emb.tobytes())
    print(f"  embed.bin    {emb.nbytes/2**30:6.2f} GiB (mmap'd, never pinned)")

    meta = {
        "model": args.model, "layers": L, "hidden": H, "inter": I, "experts": E,
        "vocab": V, "head_dim": HD, "n_q": NQ, "n_kv": NKV,
        "top_k": cfg["num_experts_per_tok"], "sliding_window": cfg["sliding_window"],
        "rms_eps": cfg["rms_norm_eps"], "swiglu_limit": cfg["swiglu_limit"],
        "swiglu_alpha": 1.702, "max_position": maxpos,
        "layer_types": layer_types[:L],
        "expert_row_bytes": row_bytes,
        "expert_row": {"gu_blocks": 0, "gu_scales": gu_blk_b,
                       "gu_bias": gu_blk_b + gu_scl_b,
                       "dn_blocks": gu_blk_b + gu_scl_b + gu_bias_b,
                       "dn_scales": gu_blk_b + gu_scl_b + gu_bias_b + dn_blk_b,
                       "dn_bias": gu_blk_b + gu_scl_b + gu_bias_b + dn_blk_b + dn_scl_b},
        "dense": dense,
        "bytes": {"experts": experts_bytes, "dense": dense_bytes, "embed": int(emb.nbytes)},
        "reserved_nan_scales": nan_scales,
    }
    # ---- the tokenizer, in the binary the CLIF reads ----
    #
    # Written here rather than beside the checkpoint because it is part of the
    # bank: the engine loads one directory.  The emitter is shared with Qwen2's
    # converter -- the format carries every dimension in its header, so one
    # CLIF tokenizer serves both models and what differs between them is data.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "tools"))
    import tokbin
    # `snap` and not `args.model`: the latter may be a repo id that was resolved
    # to a snapshot directory above, and the tokenizer lives beside the weights.
    tok_json = os.path.join(snap, "tokenizer.json")
    if os.path.exists(tok_json):
        meta["tokenizer"] = tokbin.convert(
            tok_json, os.path.join(args.out, "tokenizer.bin"))
    else:
        print(f"  no tokenizer.json at {tok_json}; skipping the tokenizer binary")

    with open(os.path.join(args.out, "bank.json"), "w") as f:
        json.dump(meta, f, indent=1)

    total = experts_bytes + dense_bytes + emb.nbytes
    print(f"\n  total        {total/2**30:6.2f} GiB in {time.time()-t0:.0f}s")
    if nan_scales:
        print(f"  WARNING: {nan_scales} reserved (255) scale bytes -- the kernels "
              f"decode these to NaN and do not check for them")
    else:
        print("  no reserved (255) scale bytes: every block decodes to a finite scale")
    print("RESULT : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
