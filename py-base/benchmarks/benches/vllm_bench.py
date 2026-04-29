"""
GPU GEMV benchmark: PyTorch (cuBLAS) vs py_base (cuBLAS via CLIF).

This is a resident-GPU persistent benchmark:
- A is uploaded once and stays on GPU
- x is refreshed before each timed run
- timing covers compute only
- correctness is checked with an untimed download
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


GEMV_SIZES = [(4096, 4096), (4096, 11008), (11008, 4096)]


def _time_torch(fn):
    torch.cuda.synchronize()

    def synced():
        fn()
        torch.cuda.synchronize()

    return harness.time_ms(synced)


def run(persist_path: str, rounds: int) -> list[harness.BenchResult]:
    raw = json.loads(open(persist_path).read())
    config = py_base.BaseConfig(json.dumps(raw[0]))
    load_alg = py_base.Algorithm(json.dumps(raw[1]))
    prep_alg = py_base.Algorithm(json.dumps(raw[2]))
    infer_alg = py_base.Algorithm(json.dumps(raw[3]))

    results = []
    rng = np.random.default_rng(42)

    for m, n in GEMV_SIZES:
        engine = py_base.Base(config)
        a_np = rng.standard_normal((m, n)).astype(np.float32)
        x_np = rng.standard_normal(n).astype(np.float32)
        out = bytearray(m * 4)

        load_data = struct.pack("<QQ", m, n) + a_np.tobytes()
        engine.execute(load_alg, load_data)
        x_bytes = x_np.tobytes()

        if _TORCH_OK:
            a_t = torch.from_numpy(a_np).cuda()
            x_t = torch.from_numpy(x_np).cuda()
            torch.mv(a_t, x_t)
            torch_ms = harness.median_of(
                rounds,
                lambda: _time_torch(lambda: torch.mv(a_t, x_t)),
            )
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
            engine.execute_into(infer_alg, b"", out)
            ref = torch.mv(a_t, x_t).cpu().numpy()
            got = np.frombuffer(bytes(out), dtype=np.float32)
            mag = max(float(np.abs(ref).max()), 1e-6)
            verified = bool(np.max(np.abs(got - ref)) / mag < 1e-2)
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
