#!/usr/bin/env python3
"""Dump the code Inductor actually emits for the compiled model.

Separate process because it imports torch.  `TORCH_LOGS=output_code` prints the
generated Triton (GPU) or C++ (CPU) for every fused region, which is the direct
answer to what a fused implementation of this model looks like.
"""
import os, sys
os.environ.setdefault("TORCH_LOGS", "output_code")
import torch, timm

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda"
NC = 128

m = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval().to(DEV)
for blk in m.blocks:
    blk.mlp.act = torch.nn.GELU(approximate="tanh")

torch.manual_seed(0)
img = torch.randn(1, 3, 224, 224, device=DEV)
with torch.no_grad():
    t = m.patch_embed(img)
    t = torch.cat([m.cls_token.expand(1, -1, -1), t], 1) + m.pos_embed
x0 = t.detach()

hw = m.head.weight.detach()[:NC].clone().requires_grad_(True)
hb = m.head.bias.detach()[:NC].clone().requires_grad_(True)

blocks_c = torch.compile(torch.nn.Sequential(*m.blocks), mode="max-autotune")
norm_c = torch.compile(m.norm)

with torch.no_grad():
    out = norm_c(blocks_c(x0))[0, 0] @ hw.T + hb
print("logits", out.shape, file=sys.stderr)

# and the backward, which is where most of our launches are
y = torch.tensor([7], device=DEV)
for p in [p for blk in m.blocks for p in blk.parameters()] + list(m.norm.parameters()):
    p.requires_grad_(True)
loss = torch.nn.functional.cross_entropy(
    (norm_c(blocks_c(x0))[0, 0] @ hw.T + hb)[None], y)
loss.backward()
print("backward done", file=sys.stderr)
