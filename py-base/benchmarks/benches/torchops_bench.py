"""
Basic GPU tensor benchmarks: PyTorch vs py_base.

VecAdd: y = x + y
SAXPY:  y = 2*x + y

Both sides keep tensors resident on GPU and time compute only. py_base uses a
load / prep / infer split so the timed path matches the resident-tensor PyTorch
reference instead of including host-device transfers.
"""

import json
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


SIZES = [1_000_000, 10_000_000, 50_000_000]


def _time_torch(fn):
    torch.cuda.synchronize()

    def synced():
        fn()
        torch.cuda.synchronize()

    return harness.time_ms(synced)


def _load_persist(path: str):
    raw = json.loads(open(path).read())
    return (
        py_base.BaseConfig(json.dumps(raw[0])),
        py_base.Algorithm(json.dumps(raw[1])),
        py_base.Algorithm(json.dumps(raw[2])),
        py_base.Algorithm(json.dumps(raw[3])),
    )


def _run_binary(
    algo_path: str,
    rounds: int,
    label: str,
    torch_ref,
) -> list[harness.BenchResult]:
    config, load_alg, prep_alg, infer_alg = _load_persist(algo_path)
    results = []
    rng = np.random.default_rng(2026)

    for n in SIZES:
        engine = py_base.Base(config)
        engine.execute(load_alg, struct.pack("<Q", n))

        x_np = rng.standard_normal(n).astype(np.float32)
        y_np = rng.standard_normal(n).astype(np.float32)
        prep = x_np.tobytes() + y_np.tobytes()
        out = bytearray(n * 4)

        if _TORCH_OK:
            x_t = torch.from_numpy(x_np).cuda()
            y_t = torch.from_numpy(y_np).cuda()
            torch_ref(x_t, y_t)
            torch_ms = harness.median_of(
                rounds,
                lambda: _time_torch(lambda: torch_ref(x_t, y_t)),
            )
        else:
            torch_ms = None

        engine.execute(prep_alg, prep)
        engine.execute(infer_alg)
        pybase_ms = harness.median_of(
            rounds,
            lambda: (
                engine.execute(prep_alg, prep),
                harness.time_ms(lambda: engine.execute(infer_alg)),
            )[1],
        )

        if _TORCH_OK:
            engine.execute(prep_alg, prep)
            engine.execute_into(infer_alg, b"", out)
            ref = torch_ref(x_t, y_t).cpu().numpy()
            got = np.frombuffer(bytes(out), dtype=np.float32)
            mag = max(float(np.abs(ref).max()), 1e-6)
            verified = bool(np.max(np.abs(got - ref)) / mag < 1e-5)
        else:
            verified = None

        results.append(
            harness.BenchResult(
                name=f"{label} ({harness.format_count(n)})",
                python_ms=torch_ms,
                pybase_ms=pybase_ms,
                verified=verified,
            )
        )

    return results


def run(vecadd_path: str, saxpy_path: str, rounds: int) -> list[harness.BenchResult]:
    return (
        _run_binary(vecadd_path, rounds, "VecAdd ", lambda x, y: x + y)
        + _run_binary(saxpy_path, rounds, "SAXPY  ", lambda x, y: 2.0 * x + y)
    )
