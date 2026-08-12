#!/usr/bin/env python3
"""PyTorch's loss trajectory for the same fine-tuning run, to compare against.

Runs in its own process because it imports torch.

The comparison is only meaningful if both sides descend the same function from
the same point with the same rule, so: the patch embedding is frozen (it is an
input buffer on our side, not a parameter), the activation is the tanh GELU our
model uses, the head is the 128-wide slice our layout carries, and the optimiser
is plain SGD at 1e-3 with no momentum and no weight decay — which is what
`Vit.sgdSpec` at `LR_RECIP = 1000` is.
"""
import sys, numpy as np, torch, timm

D = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
LABEL = int(sys.argv[3]) if len(sys.argv) > 3 else 7
NC, LR = 128, 1e-3

m = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
for blk in m.blocks:
    blk.mlp.act = torch.nn.GELU(approximate="tanh")

torch.manual_seed(0)
img = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    t = m.patch_embed(img)
    t = torch.cat([m.cls_token.expand(1, -1, -1), t], 1) + m.pos_embed
x0 = t.detach()          # frozen: our buffer 4

hw = m.head.weight.detach()[:NC].clone().requires_grad_(True)
hb = m.head.bias.detach()[:NC].clone().requires_grad_(True)
params = ([p for blk in m.blocks for p in blk.parameters()]
          + list(m.norm.parameters()) + [hw, hb])
for p in params:
    p.requires_grad_(True)
opt = torch.optim.SGD(params, lr=LR)

y = torch.tensor([LABEL])
losses = []
for _ in range(STEPS):
    x = x0
    for blk in m.blocks:
        x = blk(x)
    logits = m.norm(x)[0, 0] @ hw.T + hb
    loss = torch.nn.functional.cross_entropy(logits[None], y)
    losses.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()

np.save(D + "/losses.npy", np.array(losses, np.float64))
print(f"steps    : {STEPS}   label {LABEL}   lr {LR}   params {len(params)} tensors")
print(f"loss     : {losses[0]:.6f} -> {losses[-1]:.6f}")
