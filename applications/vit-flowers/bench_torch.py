#!/usr/bin/env python3
"""Baseline: DeiT-Tiny under PyTorch across batch sizes, on this card.

Reported per image as well as per step, because the comparison we care about
is throughput at whatever batch each side runs best at.
"""
import torch, timm, time, sys

d = "cuda"
m = timm.create_model("deit_tiny_patch16_224", pretrained=True, num_classes=102).to(d)
lossf = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(m.parameters(), lr=1e-3)

def timed(fn, n=30):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3

print(f"PyTorch DeiT-Tiny, {torch.cuda.get_device_name(0)}, fp32")
print(f"{'batch':>6} {'fwd ms':>9} {'fwd/img':>9} {'step ms':>9} {'step/img':>9} {'img/s':>8}")
for B in [1, 8, 16, 32, 64, 128]:
    try:
        x = torch.randn(B, 3, 224, 224, device=d)
        y = torch.randint(0, 102, (B,), device=d)
        m.eval()
        with torch.no_grad(): fwd = timed(lambda: m(x))
        m.train()
        def step():
            opt.zero_grad(set_to_none=True)
            lossf(m(x), y).backward()
            opt.step()
        full = timed(step)
        print(f"{B:>6} {fwd:9.2f} {fwd/B:9.3f} {full:9.2f} {full/B:9.3f} {B/full*1e3:8.0f}")
        del x, y; torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"{B:>6}   {type(e).__name__}: out of memory"); break
