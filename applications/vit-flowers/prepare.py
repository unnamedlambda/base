#!/usr/bin/env python3
"""Pack DeiT-Tiny's weights into the Lean buffer layout, and compute the
reference forward.

Runs in its own process because it imports torch: a PyTorch CUDA context in
the same process makes every cuBLAS call from our runtime fail.
"""
import sys, json, struct, numpy as np, torch, timm

ART, D = sys.argv[1], sys.argv[2]
NL, SQ, SK, DM, NH, HD, DFF, NC, TOK = 12, 200, 224, 192, 3, 64, 768, 128, 197

m = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
sd = {k: v.detach().numpy().astype(np.float32) for k, v in m.state_dict().items()}
torch.manual_seed(0)
img = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    t = m.patch_embed(img)
    t = torch.cat([m.cls_token.expand(1, -1, -1), t], 1) + m.pos_embed
x0 = np.zeros((SQ, DM), np.float32); x0[:TOK] = t.numpy()[0]
hw, hb = sd["head.weight"][:NC].copy(), sd["head.bias"][:NC].copy()

# The padding mask (buffer 3) is not here: the artifact ships it in its own
# memory and uploads it from there, so there is nothing for this script to agree
# with about it.
bufs = [np.full(DM, 1.0/DM, np.float32), np.ones(DFF, np.float32),
        np.ones(SK, np.float32),
        x0, sd["norm.weight"], hw, sd["norm.bias"], hb]
for i in range(NL):
    p = f"blocks.{i}."
    qkv, qkvb = sd[p+"attn.qkv.weight"], sd[p+"attn.qkv.bias"]
    Wo = sd[p+"attn.proj.weight"]
    bufs += [sd[p+"norm1.weight"], sd[p+"norm1.bias"]]
    for base in (0, DM, 2*DM):                       # q, k, v: weight then bias
        bufs += [qkv[base+h*HD:base+(h+1)*HD] for h in range(NH)]
        bufs += [qkvb[base+h*HD:base+(h+1)*HD] for h in range(NH)]
    bufs += [Wo]                                     # one (DM x DM), all heads
    bufs += [sd[p+"attn.proj.bias"], sd[p+"norm2.weight"], sd[p+"norm2.bias"],
             sd[p+"mlp.fc1.weight"], sd[p+"mlp.fc1.bias"],
             sd[p+"mlp.fc2.weight"], sd[p+"mlp.fc2.bias"]]

blob = b"".join(np.ascontiguousarray(b, np.float32).tobytes() for b in bufs)
mem = bytes(json.load(open(ART))["setup"]["initial_memory"])
want = struct.unpack_from("<I", mem, 0x80)[0]
assert len(blob) == want, f"packed {len(blob)} vs Lean layout {want}"
print(f"layout   : {len(blob)} bytes, matches Lean vHostIn")

def ln(x, w, b, eps=1e-6):
    c = x - x.mean(-1, keepdims=True)
    return c / np.sqrt((c*c).mean(-1, keepdims=True) + eps) * w + b
def gelu(x): return 0.5*x*(1+np.tanh(0.7978845608028654*(x+0.044715*x**3)))

x = x0[:TOK].astype(np.float64)
for i in range(NL):
    p = f"blocks.{i}."
    qkv, qkvb = sd[p+"attn.qkv.weight"], sd[p+"attn.qkv.bias"]
    Wo, bo = sd[p+"attn.proj.weight"], sd[p+"attn.proj.bias"]
    n1 = ln(x, sd[p+"norm1.weight"], sd[p+"norm1.bias"])
    q, k, v = np.split(n1 @ qkv.T + qkvb, 3, -1)
    att = np.zeros((TOK, DM))
    for h in range(NH):
        s = h*HD
        sc = (q[:, s:s+HD] @ k[:, s:s+HD].T) / np.sqrt(HD)
        pr = np.exp(sc - sc.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
        att += (pr @ v[:, s:s+HD]) @ Wo[:, s:s+HD].T
    x = x + att + bo
    n2 = ln(x, sd[p+"norm2.weight"], sd[p+"norm2.bias"])
    x = x + gelu(n2 @ sd[p+"mlp.fc1.weight"].T + sd[p+"mlp.fc1.bias"]) \
            @ sd[p+"mlp.fc2.weight"].T + sd[p+"mlp.fc2.bias"]
ref = ln(x, sd["norm.weight"], sd["norm.bias"])[0] @ hw.T + hb

np.save(D + "/blob.npy", np.frombuffer(blob, np.uint8))
np.save(D + "/ref.npy", ref)
print(f"reference: {NL} blocks, logits[:4] {ref[:4]}")
