#!/usr/bin/env python3
"""Train a CIFAR-10 classifier on proven kernels, driven from Python.

The model is `3072 → 256 → 10`, silu activation, cross-entropy loss, plain SGD
at batch `BATCH`. Every kernel — both matvecs, the activation and its
derivative, both outer products, both optimiser steps — is an instance of a
schema proven in `lean/algorithms/MlpCifarAlgorithm.lean` and checked
bit-for-bit against its committed fold order by
`cargo run -p mlp-cifar --release`.

One thing happens on the host: the data loader, which keeps the dataset in RAM
and uploads one batch. Softmax and the cross-entropy gradient run on the device
(`MlpCifar.smKernel`, proven by `softmaxCE_stores`); the host sends one-hot
labels instead of gradients.

Run:  py-base/.venv/bin/python applications/mlp-cifar/train.py
"""

import glob
import json
import os
import struct
import sys
import time

import numpy as np
import py_base

IN = 3072
H = 256
C = 32           # output units: 10 classes padded to the warp width
CLASSES = 10
BATCH = 8        # must match `MlpCifar.B`
# Must match `MlpCifar.LR_RECIP`. The BATCH is what makes the step the *mean*
# gradient, since the gradient kernels accumulate a sum over the batch — so a
# batch-size change is not silently a learning-rate change.
LR = 1.0 / (256 * BATCH)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA = os.path.join(ROOT, "cifar-10-batches-bin")

# Everything the driver calls, so a stale artifact is caught before the run
# rather than as a `KeyError` partway through it.
NEEDED = ["uploadX", "uploadOneHot", "uploadBias", "fetchLogits", "runFwd",
          "runBwd", "fetchH"]


def find_artifact() -> str:
    pat = os.path.join(
        ROOT, "target", "*", "build", "mlp-cifar-*", "out",
        "MlpCifarAlgorithm", "mlp_cifar.json",
    )
    hits = glob.glob(pat)
    if not hits:
        sys.exit("No artifact — run `cargo build -p mlp-cifar --release` first.")
    # Select by *content*, not by mtime: several build directories persist, and
    # one built before a kernel existed looks perfectly healthy until the
    # missing key surfaces mid-run.
    for h in sorted(hits, key=os.path.getmtime, reverse=True):
        try:
            keys = set(py_base.load_artifact(h).extras.keys())
        except Exception:
            continue
        if set(NEEDED) <= keys:
            return h
    sys.exit(f"No artifact has {NEEDED} — rebuild: cargo build -p mlp-cifar --release")


def load_cifar() -> tuple:
    """The CIFAR-10 binary format: 3073 bytes per record, label then 3072 pixels.

    The whole dataset is 180 MB as `float32`, so it stays resident and a step
    uploads one `BATCH`-image slice. This is the piece to move into `base`
    next; at ImageNet scale it is a real pipeline, and here it is a slice.
    """
    if not os.path.isdir(DATA):
        sys.exit(
            f"CIFAR-10 not found at {DATA}\n"
            "  curl -L https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz | tar xz"
        )

    def read(paths):
        raw = np.concatenate([np.fromfile(p, dtype=np.uint8) for p in paths])
        rec = raw.reshape(-1, 1 + IN)
        y = rec[:, 0].astype(np.int64)
        # CIFAR stores plane-major (all R, then G, then B); the model is dense,
        # so the pixel order only has to be *consistent*, and it is.
        x = rec[:, 1:].astype(np.float32) / 255.0
        return x, y

    train = sorted(glob.glob(os.path.join(DATA, "data_batch_*.bin")))
    xtr, ytr = read(train)
    xte, yte = read([os.path.join(DATA, "test_batch.bin")])
    # Standardise with the training statistics — the usual per-channel
    # normalisation, computed once rather than per batch.
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    return (xtr - mu) / sd, ytr, (xte - mu) / sd, yte


def main() -> None:
    path = find_artifact()
    art = py_base.load_artifact(path)
    ex = art.extras
    print(f"artifact : {os.path.relpath(path, ROOT)}")

    xtr, ytr, xte, yte = load_cifar()
    print(f"data     : {len(xtr)} train / {len(xte)} test, {IN} features, {CLASSES} classes")

    rng = np.random.default_rng(0)
    # Xavier-uniform. The 22 padding units start at zero and stay there: their
    # gradient is zeroed below, so they contribute nothing and are never read.
    w1 = rng.uniform(-1, 1, (H, IN)).astype(np.float32) * np.float32(np.sqrt(6 / IN))
    w2 = np.zeros((C, H), dtype=np.float32)
    w2[:CLASSES] = rng.uniform(-1, 1, (CLASSES, H)) * np.sqrt(6 / H)

    blob = w1.tobytes() + w2.tobytes()
    # `PySetup` exposes no getters, so read the published layout out of the
    # artifact JSON. A `hasattr` guard would silently skip and read as a pass.
    HOST_LEN_OFF = 0x0080
    with open(path) as fh:
        mem = bytes(json.load(fh)["setup"]["initial_memory"])
    want = struct.unpack_from("<I", mem, HOST_LEN_OFF)[0]
    assert len(blob) == want, f"host packing {len(blob)} vs Lean's hostIn {want}"
    print(f"layout   : {len(blob)} bytes, matches Lean's hostIn")

    base = py_base.Base(art.setup)
    base.execute_into(art.main, blob, bytearray(0))

    nil = bytearray(0)
    logit_buf = bytearray(BATCH * C * 4)
    onehot = np.zeros((BATCH, C), dtype=np.float32)
    # Padding classes are masked here, once: `exp` underflows them to zero, so
    # they never enter the sum and their gradient is zero.
    bias = np.where(np.arange(C) < CLASSES, 0.0, -1.0e30).astype(np.float32)
    base.execute_into(ex["uploadBias"], bias.tobytes(), nil)

    def forward(xb: np.ndarray) -> np.ndarray:
        """`xb` is (BATCH, IN); returns (BATCH, CLASSES) logits."""
        base.execute_into(ex["uploadX"], xb.tobytes(), nil)
        base.execute_into(ex["runFwd"], b"", nil)
        base.execute_into(ex["fetchLogits"], b"", logit_buf)
        return np.frombuffer(bytes(logit_buf), dtype="<f4").reshape(BATCH, C)[:, :CLASSES]

    def softmax(lg: np.ndarray) -> np.ndarray:
        p = np.exp(lg - lg.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    def evaluate(x: np.ndarray, y: np.ndarray, n: int) -> tuple:
        # Trim to whole batches: a partial one would leave stale samples from
        # the previous upload in the tail, which would quietly inflate or
        # deflate the score rather than fail.
        n -= n % BATCH
        loss, right = 0.0, 0
        for i in range(0, n, BATCH):
            lg = forward(x[i:i + BATCH])
            p = softmax(lg)
            yb = y[i:i + BATCH]
            loss -= float(np.log(np.maximum(p[np.arange(BATCH), yb], 1e-12)).sum())
            right += int((lg.argmax(axis=1) == yb).sum())
        return loss / n, right / n

    EVAL = 2000
    l0, a0 = evaluate(xte, yte, EVAL)
    print(f"\nmodel    : {IN} → {H} → {CLASSES}, silu, cross-entropy, SGD")
    print(f"batch    : {BATCH}, lr = 1/{int(1 / LR)} on the summed gradient "
          f"(= 1/256 per sample)")
    print(f"{'seen':>8}  {'test loss':>10}  {'test acc':>9}  {'us/sample':>10}")
    print(f"{0:>8}  {l0:>10.4f}  {a0:>8.1%}  {'-':>10}")

    CHUNK = 25000  # samples between reports
    seen, t_train = 0, 0.0
    for _ in range(8):
        order = rng.permutation(len(xtr))
        t0 = time.perf_counter()
        for k in range(0, CHUNK, BATCH):
            idx = order[k % len(order):k % len(order) + BATCH]
            if len(idx) < BATCH:
                idx = order[:BATCH]
            xb = np.ascontiguousarray(xtr[idx])
            # The labels go up *before* the forward pass: softmax and the CE
            # gradient are the last kernel of `runFwd`, so `runBwd` reads a
            # gradient the forward pass already computed on the device.
            onehot[:] = 0.0
            onehot[np.arange(BATCH), ytr[idx]] = 1.0
            base.execute_into(ex["uploadOneHot"], onehot.tobytes(), nil)
            forward(xb)
            base.execute_into(ex["runBwd"], b"", nil)
        dt = time.perf_counter() - t0
        seen += CHUNK
        t_train += dt
        l, a = evaluate(xte, yte, EVAL)
        print(f"{seen:>8}  {l:>10.4f}  {a:>8.1%}  {dt / CHUNK * 1e6:>10.1f}")

    lf, af = evaluate(xte, yte, len(xte))
    print(f"\nfull test set ({len(xte) - len(xte) % BATCH}): loss {lf:.4f}, accuracy {af:.1%}")
    print(f"train time   : {t_train:.1f} s for {seen} samples "
          f"({t_train / seen * 1e6:.1f} us/sample, includes the Python loop)")
    assert lf < l0, "training must reduce the test loss"
    assert af > a0, "training must improve accuracy"
    print("OK — trained on proven kernels.")


if __name__ == "__main__":
    main()
