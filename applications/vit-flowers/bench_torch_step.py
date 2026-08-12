#!/usr/bin/env python3
"""PyTorch's training-step time for exactly the setup `train.py` runs.

Runs in its own process because it imports torch.

Matched to ours: patch embedding frozen (an input buffer on our side, so
PyTorch must not backprop into it either), tanh GELU, the 128-wide head slice,
plain SGD at 1e-3, batch 1, and the cross-entropy taken on the class token.
"""
import sys, time, torch, timm

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
NC, LABEL = 128, 7
dev = "cuda"

m = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval().to(dev)
for blk in m.blocks:
    blk.mlp.act = torch.nn.GELU(approximate="tanh")

torch.manual_seed(0)
img = torch.randn(1, 3, 224, 224, device=dev)
with torch.no_grad():
    t = m.patch_embed(img)
    t = torch.cat([m.cls_token.expand(1, -1, -1), t], 1) + m.pos_embed
x0 = t.detach()

hw = m.head.weight.detach()[:NC].clone().requires_grad_(True)
hb = m.head.bias.detach()[:NC].clone().requires_grad_(True)
params = ([p for blk in m.blocks for p in blk.parameters()]
          + list(m.norm.parameters()) + [hw, hb])
for p in params:
    p.requires_grad_(True)
opt = torch.optim.SGD(params, lr=1e-3)
y = torch.tensor([LABEL], device=dev)


def step():
    x = x0
    for blk in m.blocks:
        x = blk(x)
    loss = torch.nn.functional.cross_entropy((m.norm(x)[0, 0] @ hw.T + hb)[None], y)
    opt.zero_grad()
    loss.backward()
    opt.step()


def fwd():
    with torch.no_grad():
        x = x0
        for blk in m.blocks:
            x = blk(x)
        return m.norm(x)[0, 0] @ hw.T + hb


def bench(fn, name, tag):
    for _ in range(15):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / STEPS * 1e3
    print(f"PyTorch {name:8s} {tag:10s}: {ms:7.2f} ms")
    return ms


bench(fwd, "forward", "eager")
bench(step, "step", "eager")

# The honest target: compiled.  `torch.compile` fuses elementwise chains and
# uses CUDA graphs, which is precisely what an eager baseline does not do.
try:
    blocks_c = torch.compile(torch.nn.Sequential(*m.blocks), mode="max-autotune")
    norm_c = torch.compile(m.norm)

    def fwd_c():
        with torch.no_grad():
            return norm_c(blocks_c(x0))[0, 0] @ hw.T + hb

    def step_c():
        loss = torch.nn.functional.cross_entropy(
            (norm_c(blocks_c(x0))[0, 0] @ hw.T + hb)[None], y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    bench(fwd_c, "forward", "compiled")
    bench(step_c, "step", "compiled")
except Exception as e:
    print(f"compiled: FAILED — {type(e).__name__}: {e}")
