"""Check the Lean formulation against timm, in NumPy, before any GPU work.

Verifies three things independently:
  1. distributed output projection  == concat-then-project
  2. pad-to-224 with an additive key mask == unpadded 197
  3. tanh-GELU vs timm's exact erf-GELU, as a measured number
"""
import numpy as np, torch, timm

m = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
sd = {k: v.detach().numpy().astype(np.float64) for k, v in m.state_dict().items()}
DM, NH, HD, SQ, PAD = 192, 3, 64, 197, 224

torch.manual_seed(0)
img = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    ref_logits = m(img).numpy()[0]
    ref_tokens = m.patch_embed(img)
    ref_tokens = torch.cat([m.cls_token.expand(1, -1, -1), ref_tokens], 1) + m.pos_embed
    x0 = ref_tokens.numpy()[0].astype(np.float64)

def ln(x, w, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    c = x - mu
    return c / np.sqrt((c * c).mean(-1, keepdims=True) + eps) * w + b

def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))

def gelu_erf(x):
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2.0)))

def block(x, i, act, pad_to=None):
    p = f"blocks.{i}."
    n1 = ln(x, sd[p+"norm1.weight"], sd[p+"norm1.bias"])
    qkv = n1 @ sd[p+"attn.qkv.weight"].T + sd[p+"attn.qkv.bias"]
    q, k, v = np.split(qkv, 3, axis=-1)
    n = x.shape[0]
    if pad_to:                                    # pad keys/values, mask them off
        padw = pad_to - n
        k = np.concatenate([k, np.zeros((padw, DM))], 0)
        v = np.concatenate([v, np.zeros((padw, DM))], 0)
        mask = np.concatenate([np.zeros(n), np.full(padw, -1e30)])
    else:
        mask = np.zeros(n)
    Wo, bo = sd[p+"attn.proj.weight"], sd[p+"attn.proj.bias"]
    att = np.zeros((n, DM))
    for h in range(NH):                           # distributed output projection
        s = h * HD
        sc = (q[:, s:s+HD] @ k[:, s:s+HD].T) / np.sqrt(HD) + mask
        pr = np.exp(sc - sc.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        att += (pr @ v[:, s:s+HD]) @ Wo[:, s:s+HD].T
    x = x + att + bo
    n2 = ln(x, sd[p+"norm2.weight"], sd[p+"norm2.bias"])
    h1 = act(n2 @ sd[p+"mlp.fc1.weight"].T + sd[p+"mlp.fc1.bias"])
    return x + h1 @ sd[p+"mlp.fc2.weight"].T + sd[p+"mlp.fc2.bias"]

def run(act, pad_to=None):
    x = x0.copy()
    for i in range(12):
        x = block(x, i, act, pad_to)
    x = ln(x, sd["norm.weight"], sd["norm.bias"])
    return x[0] @ sd["head.weight"].T + sd["head.bias"]

def rel(a, b): return np.abs(a - b).max() / np.abs(b).max()

erf_unpad = run(gelu_erf)
print(f"our formulation (erf-GELU, no pad) vs timm : rel {rel(erf_unpad, ref_logits):.3e}")
print(f"padded-224 + key mask vs unpadded-197     : rel {rel(run(gelu_erf, PAD), erf_unpad):.3e}")
print(f"tanh-GELU vs erf-GELU (the approximation) : rel {rel(run(gelu_tanh), erf_unpad):.3e}")
print(f"top-1 agrees: erf {erf_unpad.argmax() == ref_logits.argmax()}, "
      f"tanh {run(gelu_tanh).argmax() == ref_logits.argmax()}")
