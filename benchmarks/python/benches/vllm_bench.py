"""
GPU kernel benchmarks: PyTorch (cuBLAS) vs py_base (PTX / cuBLAS via CLIF).

GEMV:    torch.mv(A, x) vs py_base cuBLAS SGEMV — persistent pattern.
         A is uploaded once (like torch.cuda tensor), only x transferred per call.
         Both sides time resident compute only.

RMSNorm: manual RMSNorm vs py_base PTX (256-thread block, warp shuffle reduce)
         y[i] = x[i] * w[i] / sqrt(mean(x^2) + eps), eps=1e-5

Softmax: torch.softmax vs py_base PTX (3-phase: max, exp+sum, normalize)
         Uses ex2.approx for fast approximate exp.

Decode attention: persistent batch-1 decode attention over a resident KV cache.
                  Uses batched cuBLAS GEMMs for QK / PV and PTX softmax over
                  attention scores. No RoPE or GQA yet.

PyTorch times are CUDA-synchronized and use resident GPU tensors. Each
reference is timed eagerly and under torch.compile and the faster is reported,
decided per benchmark: fusion wins on the elementwise chains and loses on the
ops that are already launch-bound.

Both sides multiply in plain float32. Inductor suggests TF32 for the GEMVs,
which is declined because the cuBLAS calls on the other side do not use it.
"""

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
import py_base

try:
    import torch
    _TORCH_OK = torch.cuda.is_available()
except ImportError:
    _TORCH_OK = False


GEMV_SIZES = [(4096, 4096), (4096, 11008), (11008, 4096)]
NORM_SIZES = [896, 2048, 4096]
SOFTMAX_SIZES = [512, 2048, 32000]
ATTN_SEQS = [128, 512, 2048]
D_MODEL = 896
D_HEAD = 64
N_HEADS = 14
SOFTMAX_INNER_ITERS = 64
ATTN_INNER_ITERS = 64


def _run_gemv(artifact_path: str, rounds: int) -> list[harness.BenchResult]:
    artifact = py_base.load_artifact(artifact_path)
    engine = py_base.Driver(artifact)
    load_alg = "main"
    prep_alg = "prep"
    infer_alg = "infer"

    results = []
    rng = np.random.default_rng(42)

    for m, n in GEMV_SIZES:
        a_np = rng.standard_normal((m, n)).astype(np.float32)
        x_np = rng.standard_normal(n).astype(np.float32)
        out = bytearray(m * 4)

        load_data = struct.pack("<QQ", m, n) + a_np.tobytes()
        engine.execute(load_alg, load_data)
        x_bytes = x_np.tobytes()

        if _TORCH_OK:
            a_t = torch.from_numpy(a_np).cuda()
            x_t = torch.from_numpy(x_np).cuda()
            torch_ms = harness.torch_median(lambda: torch.mv(a_t, x_t), rounds, label=f"GEMV ({m}x{n})")
        else:
            torch_ms = None

        engine.execute(prep_alg, x_bytes)
        engine.execute(infer_alg, b"", out)
        pybase_ms = harness.median_of(
            rounds,
            lambda: (
                engine.execute(prep_alg, x_bytes),
                harness.time_ms(lambda: engine.execute(infer_alg)),
            )[1],
        )

        if _TORCH_OK:
            engine.execute(prep_alg, x_bytes)
            engine.execute(infer_alg, b"", out)
            ref = torch.mv(a_t, x_t).cpu().numpy()
            got = np.frombuffer(bytes(out), dtype=np.float32)
            mag = max(float(np.abs(ref).max()), 1e-6)
            verified = bool(np.max(np.abs(got - ref)) / mag < 1e-5)
        else:
            verified = None

        results.append(
            harness.BenchResult(
                name=f"GEMV    ({m}×{n})",
                python_ms=torch_ms,
                pybase_ms=pybase_ms,
                verified=verified,
            )
        )

    return results


def _run_rmsnorm(artifact_path: str, rounds: int) -> list[harness.BenchResult]:
    artifact = py_base.load_artifact(artifact_path)
    engine = py_base.Driver(artifact)
    load_alg = "main"
    prep_alg = "prep"
    infer_alg = "infer"
    results = []
    rng = np.random.default_rng(7)

    for n in NORM_SIZES:
        x_np = rng.standard_normal(n).astype(np.float32)
        w_np = rng.standard_normal(n).astype(np.float32) * 0.5 + 1.0
        out = bytearray(n * 4)
        load_data = struct.pack("<Q", n) + w_np.tobytes()
        engine.execute(load_alg, load_data)
        x_bytes = x_np.tobytes()

        if _TORCH_OK:
            x_t = torch.from_numpy(x_np).cuda()
            w_t = torch.from_numpy(w_np).cuda()

            def rms(x, w):
                return x * w * torch.rsqrt(x.pow(2).mean() + 1e-5)

            torch_ms = harness.torch_median(lambda: rms(x_t, w_t), rounds, label=f"RMSNorm ({n})")
        else:
            torch_ms = None

        engine.execute(prep_alg, x_bytes)
        engine.execute(infer_alg)
        pybase_ms = harness.median_of(
            rounds,
            lambda: (
                engine.execute(prep_alg, x_bytes),
                harness.time_ms(lambda: engine.execute(infer_alg)),
            )[1],
        )

        if _TORCH_OK:
            engine.execute(prep_alg, x_bytes)
            engine.execute(infer_alg, b"", out)
            ref = rms(x_t, w_t).cpu().numpy()
            got = np.frombuffer(bytes(out), dtype=np.float32)
            mag = max(float(np.abs(ref).max()), 1e-6)
            verified = bool(np.max(np.abs(got - ref)) / mag < 1e-5)
        else:
            verified = None

        results.append(
            harness.BenchResult(
                name=f"RMSNorm ({n})",
                python_ms=torch_ms,
                pybase_ms=pybase_ms,
                verified=verified,
            )
        )
    return results


def _run_softmax(artifact_path: str, rounds: int) -> list[harness.BenchResult]:
    artifact = py_base.load_artifact(artifact_path)
    engine = py_base.Driver(artifact)
    load_alg = "main"
    prep_alg = "prep"
    infer_alg = "infer"
    results = []
    rng = np.random.default_rng(13)

    for n in SOFTMAX_SIZES:
        x_np = rng.standard_normal(n).astype(np.float32)
        out = bytearray(n * 4)
        load_data = struct.pack("<Q", n)
        engine.execute(load_alg, load_data)
        x_bytes = x_np.tobytes()

        if _TORCH_OK:
            x_t = torch.from_numpy(x_np).cuda()
            torch_ms = harness.torch_median(
                lambda: torch.softmax(x_t, dim=0),
                rounds,
                inner=SOFTMAX_INNER_ITERS,
                label=f"Softmax ({n})",
                sync_each=True,
            )
        else:
            torch_ms = None

        # Both sides now pay SOFTMAX_INNER_ITERS dispatches and the same number
        # of device round trips. The batched `stack` entry, which performed all
        # of them inside one `execute`, made this row an upper bound rather than
        # a measurement.
        def _infer_many() -> None:
            for _ in range(SOFTMAX_INNER_ITERS):
                engine.execute(infer_alg)

        engine.execute(prep_alg, x_bytes)
        _infer_many()
        pybase_ms = harness.median_of(
            rounds,
            lambda: (
                engine.execute(prep_alg, x_bytes),
                harness.time_ms(_infer_many),
            )[1],
        ) / SOFTMAX_INNER_ITERS

        if _TORCH_OK:
            engine.execute(prep_alg, x_bytes)
            engine.execute(infer_alg, b"", out)
            ref = torch.softmax(x_t, dim=0).cpu().numpy()
            got = np.frombuffer(bytes(out), dtype=np.float32)
            verified = bool(np.max(np.abs(got - ref)) < 1e-5)
        else:
            verified = None

        results.append(
            harness.BenchResult(
                name=f"Softmax ({n})",
                python_ms=torch_ms,
                pybase_ms=pybase_ms,
                verified=verified,
            )
        )
    return results


def _run_decode_attention(artifact_path: str, rounds: int) -> list[harness.BenchResult]:
    artifact = py_base.load_artifact(artifact_path)
    engine = py_base.Driver(artifact)
    load_alg = "main"
    prep_alg = "prep"
    infer_alg = "infer"
    results = []
    rng = np.random.default_rng(29)

    scale = float(D_HEAD ** -0.5)

    for seq_len in ATTN_SEQS:
        q_np = rng.standard_normal((N_HEADS, D_HEAD)).astype(np.float32) * 0.2
        k_np = rng.standard_normal((N_HEADS, seq_len, D_HEAD)).astype(np.float32) * 0.2
        v_np = rng.standard_normal((N_HEADS, seq_len, D_HEAD)).astype(np.float32) * 0.2
        out = bytearray(D_MODEL * 4)

        load_data = struct.pack("<Q", seq_len) + k_np.tobytes() + v_np.tobytes()
        engine.execute(load_alg, load_data)
        q_bytes = q_np.reshape(-1).tobytes()

        if _TORCH_OK:
            q_t = torch.from_numpy(q_np).cuda()
            k_t = torch.from_numpy(k_np).cuda()
            v_t = torch.from_numpy(v_np).cuda()

            def decode_attn():
                scores = torch.bmm(k_t, q_t.unsqueeze(2)).squeeze(2) * scale
                probs = torch.softmax(scores, dim=1)
                return torch.bmm(probs.unsqueeze(1), v_t).squeeze(1).reshape(-1)

            torch_ms = harness.torch_median(
                decode_attn,
                rounds,
                inner=ATTN_INNER_ITERS,
                label=f"Decode attn ({seq_len})",
                sync_each=True,
            )
        else:
            torch_ms = None

        # As in the softmax row: ATTN_INNER_ITERS dispatches and the same number
        # of device round trips on both sides, rather than one batched `execute`
        # against that many separate torch calls.
        def _infer_many() -> None:
            for _ in range(ATTN_INNER_ITERS):
                engine.execute(infer_alg)

        engine.execute(prep_alg, q_bytes)
        _infer_many()
        pybase_ms = harness.median_of(
            rounds,
            lambda: (
                engine.execute(prep_alg, q_bytes),
                harness.time_ms(_infer_many),
            )[1],
        ) / ATTN_INNER_ITERS

        if _TORCH_OK:
            engine.execute(prep_alg, q_bytes)
            engine.execute(infer_alg, b"", out)
            ref = decode_attn().cpu().numpy()
            got = np.frombuffer(bytes(out), dtype=np.float32)
            mag = max(float(np.abs(ref).max()), 1e-6)
            verified = bool(np.max(np.abs(got - ref)) / mag < 1e-5)
        else:
            verified = None

        results.append(
            harness.BenchResult(
                name=f"Decode  (ctx={seq_len})",
                python_ms=torch_ms,
                pybase_ms=pybase_ms,
                verified=verified,
            )
        )

    return results


def run(
    gemv: str,
    rmsnorm: str,
    softmax: str,
    decode_attn: str,
    rounds: int,
) -> list[harness.BenchResult]:
    return (
        _run_gemv(gemv, rounds)
        + _run_rmsnorm(rmsnorm, rounds)
        + _run_softmax(softmax, rounds)
        + _run_decode_attention(decode_attn, rounds)
    )
