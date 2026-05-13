#!/usr/bin/env python3
"""
Download a Qwen2 0.5B checkpoint from HuggingFace and convert to flat f32 binary.
Requires only: numpy (usually pre-installed)

Usage:
    python3 tools/qwen2_convert.py [weights_out] [tokenizer_out] [safetensors_path]

Default checkpoint: Qwen/Qwen2-0.5B-Instruct
Default outputs: qwen2_instruct_weights.bin, qwen2_instruct_tokenizer.bin

Weight layout (f32 little-endian):
  embed_tokens              [vocab_size × hidden_dim]
  For each layer 0..23:
    rms_attn_weight         [hidden_dim]
    q_proj.weight           [n_heads*head_dim × hidden_dim]
    q_proj.bias             [n_heads*head_dim]
    k_proj.weight           [n_kv_heads*head_dim × hidden_dim]
    k_proj.bias             [n_kv_heads*head_dim]
    v_proj.weight           [n_kv_heads*head_dim × hidden_dim]
    v_proj.bias             [n_kv_heads*head_dim]
    o_proj.weight           [hidden_dim × n_heads*head_dim]
    rms_ffn_weight          [hidden_dim]
    gate_proj.weight        [intermediate_size × hidden_dim]
    up_proj.weight          [intermediate_size × hidden_dim]
    down_proj.weight        [hidden_dim × intermediate_size]
  rms_final_weight          [hidden_dim]
  lm_head.weight            [vocab_size × hidden_dim]
"""

import json
import os
import struct
import sys
import urllib.request

import numpy as np

MODEL_ID = os.environ.get("QWEN2_MODEL_ID", "Qwen/Qwen2-0.5B-Instruct")
HF_BASE = "https://huggingface.co"
N_LAYERS = 24
DEFAULT_OUT = "qwen2_instruct_weights.bin"
CACHE_FILE = "/tmp/qwen2_instruct_model.safetensors"


def hf_url(filename: str) -> str:
    return f"{HF_BASE}/{MODEL_ID}/resolve/main/{filename}"


def download(url: str, dest: str):
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "python3"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        chunk = 1 << 20  # 1 MB
        with open(tmp, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = done / total * 100
                    mb = done / 1e6
                    print(f"\r  {mb:.0f} MB / {total/1e6:.0f} MB  ({pct:.1f}%)", end="", flush=True)
    print()
    os.rename(tmp, dest)


def open_safetensors(path: str):
    """Return (mmap_array, header_dict, data_start_offset)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=data_start)
    return mm, header, data_start


def read_tensor(mm, header, name: str) -> np.ndarray:
    info = header[name]
    start, end = info["data_offsets"]
    dtype = info["dtype"]
    raw = mm[start:end]
    if dtype == "BF16":
        u16 = raw.view(np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32).copy()
    elif dtype == "F32":
        return raw.view(np.float32).copy()
    elif dtype == "F16":
        return raw.view(np.float16).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype {dtype} for {name}")


def convert(sf_path: str, out_path: str):
    print(f"Parsing {sf_path} ...")
    mm, header, _ = open_safetensors(sf_path)

    # Check for sharded index
    keys = [k for k in header if k != "__metadata__"]
    print(f"  {len(keys)} tensors found")

    def w(name: str) -> np.ndarray:
        return read_tensor(mm, header, name)

    print(f"Writing {out_path} ...")
    with open(out_path, "wb") as f:
        def write(arr: np.ndarray):
            f.write(arr.astype(np.float32).ravel().tobytes())

        write(w("model.embed_tokens.weight"))
        print("  embed_tokens done")

        for i in range(N_LAYERS):
            p = f"model.layers.{i}"
            write(w(f"{p}.input_layernorm.weight"))
            write(w(f"{p}.self_attn.q_proj.weight"))
            write(w(f"{p}.self_attn.q_proj.bias"))
            write(w(f"{p}.self_attn.k_proj.weight"))
            write(w(f"{p}.self_attn.k_proj.bias"))
            write(w(f"{p}.self_attn.v_proj.weight"))
            write(w(f"{p}.self_attn.v_proj.bias"))
            write(w(f"{p}.self_attn.o_proj.weight"))
            write(w(f"{p}.post_attention_layernorm.weight"))
            write(w(f"{p}.mlp.gate_proj.weight"))
            write(w(f"{p}.mlp.up_proj.weight"))
            write(w(f"{p}.mlp.down_proj.weight"))
            if (i + 1) % 6 == 0:
                print(f"  layer {i+1}/{N_LAYERS} done")

        write(w("model.norm.weight"))
        # Qwen2-0.5B ties lm_head weights to embed_tokens
        lm_key = "lm_head.weight" if "lm_head.weight" in header else "model.embed_tokens.weight"
        write(w(lm_key))
        print(f"  final norm + lm_head ({lm_key}) done")

    size_gb = os.path.getsize(out_path) / 1e9
    print(f"Done. {out_path}  ({size_gb:.2f} GB)")


TOKENIZER_CACHE = "/tmp/qwen2_instruct_tokenizer.json"
DEFAULT_TOK_OUT = "qwen2_instruct_tokenizer.bin"


def gpt2_byte_decoder() -> dict:
    """Return {unicode_char: byte_value} for the GPT-2 byte-level encoding."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_str_to_bytes(s: str, byte_dec: dict) -> bytes:
    out = bytearray()
    for c in s:
        if c in byte_dec:
            out.append(byte_dec[c])
        else:
            out.extend(c.encode("utf-8", errors="replace"))
    return bytes(out)


def convert_tokenizer(tok_path: str, out_path: str):
    with open(tok_path, encoding="utf-8") as f:
        tok = json.load(f)

    model  = tok["model"]
    vocab  = model["vocab"]           # str → int
    merges_raw = model.get("merges", [])  # ["a b", ...]
    vocab_size = len(vocab)
    id_to_str  = {v: k for k, v in vocab.items()}
    byte_dec   = gpt2_byte_decoder()

    # byte_init[256]: byte value → initial token ID
    byte_init = np.zeros(256, dtype=np.uint32)
    for char, byte_val in byte_dec.items():
        if char in vocab:
            byte_init[byte_val] = vocab[char]

    # Process merge rules; store as (tok_a, tok_b, result) in rank order
    merges = []
    for merge_str in merges_raw:
        parts = merge_str.split(" ", 1)
        if len(parts) != 2:
            continue
        a, b = parts
        ab = a + b
        if a in vocab and b in vocab and ab in vocab:
            merges.append((vocab[a], vocab[b], vocab[ab]))

    n_merges = len(merges)

    # decode table: offset + length into byte_pool for each token
    byte_pool = bytearray()
    decode_offsets = np.zeros(vocab_size, dtype=np.uint32)
    decode_lens    = np.zeros(vocab_size, dtype=np.uint32)
    for tok_id in range(vocab_size):
        s = id_to_str.get(tok_id, "")
        b = token_str_to_bytes(s, byte_dec)
        decode_offsets[tok_id] = len(byte_pool)
        decode_lens[tok_id]    = len(b)
        byte_pool.extend(b)

    byte_pool_size = len(byte_pool)

    with open(out_path, "wb") as f:
        # Header (16 bytes)
        f.write(struct.pack("<IIII", n_merges, vocab_size, byte_pool_size, 0))
        # byte_init (1024 bytes)
        f.write(byte_init.tobytes())
        # merge table: n_merges × 12 bytes
        for tok_a, tok_b, result in merges:
            f.write(struct.pack("<III", tok_a, tok_b, result))
        # decode_offsets (vocab_size × 4)
        f.write(decode_offsets.tobytes())
        # decode_lens (vocab_size × 4)
        f.write(decode_lens.tobytes())
        # byte_pool
        f.write(bytes(byte_pool))

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Tokenizer: {n_merges} merges, {vocab_size} vocab → {out_path} ({size_mb:.1f} MB)")


def main():
    weights_out  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    tok_out      = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TOK_OUT
    sf_path      = sys.argv[3] if len(sys.argv) > 3 else CACHE_FILE
    tokenizer_url = hf_url("tokenizer.json")

    if not os.path.exists(sf_path):
        download(hf_url("model.safetensors"), sf_path)
    else:
        print(f"Using cached {sf_path}")

    convert(sf_path, weights_out)

    if not os.path.exists(TOKENIZER_CACHE):
        download(tokenizer_url, TOKENIZER_CACHE)
    else:
        print(f"Using cached {TOKENIZER_CACHE}")

    convert_tokenizer(TOKENIZER_CACHE, tok_out)


if __name__ == "__main__":
    main()
