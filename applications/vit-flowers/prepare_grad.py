#!/usr/bin/env python3
"""PyTorch's gradients for the same forward, as the thing to be checked against.

Runs in its own process because it imports torch: a PyTorch CUDA context in the
same process makes every cuBLAS call from our runtime fail.

The loss is cross-entropy on the class token's logits, so `dL/dlogits` is
`softmax(logits) - onehot` — the seed our backward is given, saved here so both
sides are differentiating the same scalar from the same point.
"""
import sys, numpy as np, torch, timm

D, LABEL = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7
NL, SQ, SK, DM, NH, HD, NC, TOK = 12, 200, 224, 192, 3, 64, 128, 197

m = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
# Our model's GELU is the tanh form, so the reference must use it too: the two
# activations differ by 3.4e-3 in the forward, and differentiating different
# functions would make that the answer rather than the check.
for blk in m.blocks:
    blk.mlp.act = torch.nn.GELU(approximate="tanh")

zmax = []
for blk in m.blocks:
    blk.mlp.fc1.register_forward_hook(
        lambda mod, i, o: zmax.append((round(float(o.min()), 1), round(float(o.max()), 1))))
torch.manual_seed(0)
img = torch.randn(1, 3, 224, 224)

# The head is padded to NC in our layout; PyTorch's is 1000 wide, so the
# comparison is against the same slice the artifact was given.
hw = m.head.weight.detach()[:NC].clone().requires_grad_(True)
hb = m.head.bias.detach()[:NC].clone().requires_grad_(True)

t = m.patch_embed(img)
t = torch.cat([m.cls_token.expand(1, -1, -1), t], 1) + m.pos_embed
x = t
for blk in m.blocks:
    x = blk(x)
x = m.norm(x)
logits = x[0, 0] @ hw.T + hb

loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([LABEL]))
loss.backward()

# The seed, at our padded geometry: row 0 carries the class token's gradient,
# every other row is zero because nothing else feeds the loss.
p = torch.softmax(logits.detach(), 0).numpy()
seed = np.zeros((SQ, NC), np.float32)
seed[0] = p
seed[0, LABEL] -= 1.0

grads = {"norm.weight": m.norm.weight.grad, "norm.bias": m.norm.bias.grad,
         "head.weight": hw.grad, "head.bias": hb.grad}
for i, blk in enumerate(m.blocks):
    pre = f"blocks.{i}."
    for n, prm in blk.named_parameters():
        grads[pre + n] = prm.grad

np.save(D + "/seed.npy", seed)
np.savez(D + "/grads.npz", **{k: v.detach().numpy().astype(np.float32)
                              for k, v in grads.items()})
print(f"label    : {LABEL}   loss {loss.item():.6f}")
print(f"logits[:4]: {logits.detach().numpy()[:4]}")
print(f"grads    : {len(grads)} tensors, seed {seed.shape}")
print(f"fc1 pre-activation (min, max) per block: {zmax}")
