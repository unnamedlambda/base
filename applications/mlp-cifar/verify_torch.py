#!/usr/bin/env python3
"""Cross-check the proven CIFAR MLP against PyTorch — numerics and throughput.

`cargo run -p mlp-cifar --release` checks each kernel against a CPU reference
written from the Lean spec. That catches a kernel that disagrees with its
theorem, but it shares an author with the thing it checks: if I misread the
spec, the reference is wrong in the same direction and the check passes. An
independent implementation of the *model* is what closes that.

So this builds the identical network in PyTorch — same weights, same batch,
same loss — and compares:

  * the logits (forward),
  * dW1 and dW2 (backward),
  * the accuracy trajectory over a real training run,
  * time per sample.

**Bit-exactness is not expected and would be suspicious.** PyTorch's matmul
goes through cuBLAS, whose fold order NVIDIA does not specify; the proven
kernels commit to a lane-sequential sweep and a five-round butterfly. The two
agree as reals and differ in the last bits. Agreement to ~1e-6 relative is the
correct outcome; agreement to 0 would mean one of them is not doing what it
claims.

Run:  py-base/.venv/bin/python applications/mlp-cifar/verify_torch.py
"""

import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (  # noqa: E402
    BATCH, C, CLASSES, H, IN, LR, find_artifact, load_cifar,
)

import py_base  # noqa: E402


def agree(name: str, a: np.ndarray, b: np.ndarray, tol: float = 1e-5) -> None:
    """Report and check agreement, scaled to the tensor rather than per element.

    Per-element relative error is the wrong criterion for a gradient: `silu'`
    has a root, so `dW1` contains entries that are near zero by cancellation,
    and a rounding there is a huge *relative* difference and a meaningless one.
    What matters is the error against the size of the thing being computed, so
    that is what is asserted; the per-element figure is printed beside it so a
    genuine systematic disagreement is still visible.
    """
    d = np.abs(a - b)
    scale = float(np.max(np.abs(b)))
    worst_abs = float(np.max(d))
    per_elem = float(np.max(d / np.maximum(np.abs(b), 1e-6)))
    print(f"{name:<9}: {worst_abs / scale:.2e} of scale   "
          f"(worst abs {worst_abs:.2e}, scale {scale:.2e}, "
          f"worst per-element rel {per_elem:.2e})")
    assert worst_abs / scale < tol, (
        f"{name}: {worst_abs / scale:.2e} of scale exceeds {tol:.0e} — "
        "that is a disagreement, not a rounding"
    )


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("This comparison needs CUDA — both sides must run on the same card.")
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True

    path = find_artifact()
    art = py_base.load_artifact(path)
    ex = art.extras
    xtr, ytr, xte, yte = load_cifar()

    rng = np.random.default_rng(0)
    w1 = rng.uniform(-1, 1, (H, IN)).astype(np.float32) * np.float32(np.sqrt(6 / IN))
    w2 = np.zeros((C, H), dtype=np.float32)
    w2[:CLASSES] = rng.uniform(-1, 1, (CLASSES, H)) * np.sqrt(6 / H)

    blob = w1.tobytes() + w2.tobytes()
    HOST_LEN_OFF = 0x0080
    with open(path) as fh:
        mem = bytes(json.load(fh)["setup"]["initial_memory"])
    assert len(blob) == struct.unpack_from("<I", mem, HOST_LEN_OFF)[0]

    base = py_base.Base(art.setup)
    base.execute_into(art.main, blob, bytearray(0))
    nil = bytearray(0)

    # The identical network in PyTorch: no biases, silu, and the *padded* output
    # width, so the two models have the same parameter count and the padding
    # rows are exercised identically (they start at zero and get zero gradient).
    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(IN, H, bias=False)
            self.fc2 = nn.Linear(H, C, bias=False)

        def forward(self, x):
            return self.fc2(torch.nn.functional.silu(self.fc1(x)))

    net = Mlp().to(dev)
    with torch.no_grad():
        net.fc1.weight.copy_(torch.from_numpy(w1))
        net.fc2.weight.copy_(torch.from_numpy(w2))

    # `reduction="sum"` so PyTorch's gradient is the *same* quantity the proven
    # kernels accumulate; the division by BATCH lives in the learning rate on
    # both sides, exactly as `MlpCifar.LR_RECIP` documents.
    #
    # Every call below slices to `:CLASSES` first. The 22 padding units exist to
    # make the class axis a whole warp; they are not classes, and a softmax over
    # all 32 is a different distribution. Feeding the padded width to PyTorch is
    # the one way to make this comparison silently compare two different models.
    lossfn = nn.CrossEntropyLoss(reduction="sum")
    opt = torch.optim.SGD(net.parameters(), lr=LR)

    print(f"model    : {IN} → {H} → {CLASSES} (padded to {C}), batch {BATCH}, no bias")
    print(f"lr       : 1/{int(1 / LR)} on the summed gradient, both sides")
    print(f"device   : {torch.cuda.get_device_name(0)}\n")

    # ── forward ─────────────────────────────────────────────────────────────
    xb = np.ascontiguousarray(xtr[:BATCH])
    yb = ytr[:BATCH]

    def onehot(labels):
        oh = np.zeros((BATCH, C), dtype=np.float32)
        oh[np.arange(BATCH), labels] = 1.0
        return oh

    # The class axis is padded to a whole warp, and masking the 22 padding units
    # is the *host's* job: the softmax kernel sweeps all 32 and adds this bias
    # first, so -inf there is what makes it a 10-class distribution.  Forgetting
    # it trains a 32-class model that still looks plausible.
    bias = np.where(np.arange(C) < CLASSES, 0.0, -1.0e30).astype(np.float32)
    base.execute_into(ex["uploadBias"], bias.tobytes(), nil)

    base.execute_into(ex["uploadX"], xb.tobytes(), nil)
    base.execute_into(ex["uploadOneHot"], onehot(yb).tobytes(), nil)
    base.execute_into(ex["runFwd"], b"", nil)
    lbuf = bytearray(BATCH * C * 4)
    base.execute_into(ex["fetchLogits"], b"", lbuf)
    got = np.frombuffer(bytes(lbuf), dtype="<f4").reshape(BATCH, C)

    xt = torch.from_numpy(xb).to(dev)
    yt = torch.from_numpy(yb).to(dev)
    logits_t = net(xt)
    want = logits_t.detach().cpu().numpy()
    agree("logits", got, want)

    # ── backward ────────────────────────────────────────────────────────────
    # `runFwd` ends with the softmax/cross-entropy kernel, so the gradient of
    # the loss is already in the device buffer `runBwd` reads.  Checking it
    # against the host computation first, because a wrong `dlog` would show up
    # downstream as a plausible-looking gradient rather than as a failure.
    p = np.exp(got[:, :CLASSES] - got[:, :CLASSES].max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    dlog = np.zeros((BATCH, C), dtype=np.float32)
    dlog[:, :CLASSES] = p
    dlog[np.arange(BATCH), yb] -= 1.0
    dlbuf = bytearray(BATCH * C * 4)
    base.execute_into(ex["fetchDlog"], b"", dlbuf)
    agree("dlog", np.frombuffer(bytes(dlbuf), dtype="<f4").reshape(BATCH, C)[:, :CLASSES],
          dlog[:, :CLASSES])
    base.execute_into(ex["runBwd"], b"", nil)

    dw1_buf = bytearray(H * IN * 4)
    dw2_buf = bytearray(C * H * 4)
    # `runBwd` ends with the optimiser, so the gradients are still resident.
    base.execute_into(ex["fetchDw1"], b"", dw1_buf)
    base.execute_into(ex["fetchDw2"], b"", dw2_buf)
    dw1 = np.frombuffer(bytes(dw1_buf), dtype="<f4").reshape(H, IN)
    dw2 = np.frombuffer(bytes(dw2_buf), dtype="<f4").reshape(C, H)

    opt.zero_grad()
    lossfn(logits_t[:, :CLASSES], yt).backward()
    dw1_t = net.fc1.weight.grad.detach().cpu().numpy()
    dw2_t = net.fc2.weight.grad.detach().cpu().numpy()
    agree("dW1", dw1, dw1_t)
    agree("dW2", dw2, dw2_t)

    # `runBwd` ended with the optimiser, so `base` has already taken this step.
    # Take it on the PyTorch side too, or the two enter the training loop one
    # update apart and every later comparison is against a shifted baseline.
    opt.step()

    # ── train both, same data order, and compare where they land ────────────
    STEPS = 12500  # 100k samples at batch 8 — two epochs, sampled with replacement
    order = rng.integers(0, len(xtr), size=(STEPS, BATCH))

    def base_eval(n: int) -> tuple:
        n -= n % BATCH
        loss, right = 0.0, 0
        for i in range(0, n, BATCH):
            base.execute_into(ex["uploadX"], np.ascontiguousarray(xte[i:i + BATCH]).tobytes(), nil)
            base.execute_into(ex["runFwd"], b"", nil)
            base.execute_into(ex["fetchLogits"], b"", lbuf)
            lg = np.frombuffer(bytes(lbuf), dtype="<f4").reshape(BATCH, C)[:, :CLASSES]
            q = np.exp(lg - lg.max(axis=1, keepdims=True))
            q /= q.sum(axis=1, keepdims=True)
            yy = yte[i:i + BATCH]
            loss -= float(np.log(np.maximum(q[np.arange(BATCH), yy], 1e-12)).sum())
            right += int((lg.argmax(axis=1) == yy).sum())
        return loss / n, right / n, right

    @torch.no_grad()
    def torch_eval(n: int) -> tuple:
        n -= n % BATCH
        loss, right = 0.0, 0
        for i in range(0, n, BATCH):
            lg = net(torch.from_numpy(np.ascontiguousarray(xte[i:i + BATCH])).to(dev))
            yy = torch.from_numpy(yte[i:i + BATCH]).to(dev)
            loss += float(nn.functional.cross_entropy(lg[:, :CLASSES], yy, reduction="sum"))
            right += int((lg[:, :CLASSES].argmax(1) == yy).sum())
        return loss / n, right / n, right

    print(f"\ntraining both for {STEPS * BATCH} samples on the same data order")
    t0 = time.perf_counter()
    for step in order:
        base.execute_into(ex["uploadX"], np.ascontiguousarray(xtr[step]).tobytes(), nil)
        base.execute_into(ex["uploadOneHot"], onehot(ytr[step]).tobytes(), nil)
        base.execute_into(ex["runFwd"], b"", nil)
        base.execute_into(ex["runBwd"], b"", nil)
    t_base = (time.perf_counter() - t0) / (STEPS * BATCH)

    # The same model lowered to cuBLAS for its five GEMMs — identical vendor
    # kernels to PyTorch's, so this isolates everything *around* the GEMM.
    # Throughput only: it trains the weights further, which a rate does not care
    # about, so it runs after the evaluations above.
    BLAS_STEPS = 2000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in order:
        xb = torch.from_numpy(np.ascontiguousarray(xtr[step])).to(dev)
        yy = torch.from_numpy(ytr[step]).to(dev)
        opt.zero_grad(set_to_none=True)
        lossfn(net(xb)[:, :CLASSES], yy).backward()
        opt.step()
    torch.cuda.synchronize()
    t_torch = (time.perf_counter() - t0) / (STEPS * BATCH)

    lb, ab, nb = base_eval(len(xte))
    lt, at, nt = torch_eval(len(xte))
    # Full precision, because two runs landing on the same rounded number is not
    # evidence that they landed on the same number — and if they were the *same*
    # run by mistake, only the undisplayed digits would say so.
    t0 = time.perf_counter()
    for step in order[:BLAS_STEPS]:
        base.execute_into(ex["uploadX"], np.ascontiguousarray(xtr[step]).tobytes(), nil)
        base.execute_into(ex["uploadOneHot"], onehot(ytr[step]).tobytes(), nil)
        base.execute_into(ex["runFwdBlas"], b"", nil)
        base.execute_into(ex["runBwdBlas"], b"", nil)
    t_blas = (time.perf_counter() - t0) / (BLAS_STEPS * BATCH)

    print(f"\nbase, cuBLAS lowering: {t_blas * 1e6:.1f} us/sample "
          f"({t_torch / t_blas:.2f}x PyTorch, identical GEMM kernels both sides)")
    print(f"\n{'':<10} {'test loss':>14} {'correct':>9} {'test acc':>9} {'us/sample':>10}")
    print(f"{'base':<10} {lb:>14.9f} {nb:>9} {ab:>8.2%} {t_base * 1e6:>10.1f}")
    print(f"{'pytorch':<10} {lt:>14.9f} {nt:>9} {at:>8.2%} {t_torch * 1e6:>10.1f}")
    print(f"{'ratio':<10} {'':>14} {'':>9} {'':>9} {t_torch / t_base:>9.2f}x")
    assert lb != lt, (
        "identical test loss to nine digits means these are not two independent "
        "runs — check that both models are actually being trained"
    )

    # The two runs see identical data in identical order, but their arithmetic
    # differs in the last bits, and SGD amplifies that over 12500 steps. Landing
    # within a point or two is agreement; landing on the same number would not
    # be believable.
    print(f"\naccuracy gap: {abs(ab - at) * 100:.2f} points over {STEPS} steps")
    print("\nOn the speed ratio: at this size PyTorch is bound by per-op Python and")
    print("launch overhead, not by its kernels, so this compares overheads as much")
    print("as arithmetic. A torch.compile or CUDA-graph baseline would narrow it.")
    print("Both sides keep the whole step on the device: each uploads a batch and")
    print("its labels, and neither moves a gradient back across the bus.")
    assert abs(ab - at) < 0.03, (
        f"the two implementations must train to the same place: {ab:.1%} vs {at:.1%}"
    )
    print("OK — the proven kernels match PyTorch on gradients and on where they train to.")


if __name__ == "__main__":
    main()
