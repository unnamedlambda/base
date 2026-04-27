"""
NumPy vector benchmarks: numpy vs py_base.

VecAdd:   (a + b).sum()
  numpy:   allocates intermediate f32 array for (a+b), then sums — two passes.
  py_base: 4x-unrolled f32x4 SIMD add + accumulate, horizontal reduce to f64.
           Single pass, zero allocations.

ClampSum: np.clip(a, -0.5, 0.5).sum()
  numpy:   allocates intermediate f32 array for the clipped values, then sums — two passes.
  py_base: same 4x-unrolled f32x4 structure as VecAdd; fmin/fmax replace the add.
           Accumulates in f32x4, reduces to f64 at the end. Single pass, zero allocations.

Both sides start from pre-built numpy arrays (zero-copy views for numpy;
pre-packed bytes for py_base). Only the computation is timed.
"""

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
import py_base

SIZES = [1_000_000, 10_000_000, 50_000_000]

CLAMP_LO = -0.5
CLAMP_HI =  0.5


def _run_vecadd(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    engine, alg = py_base.load(algo_path)
    out = bytearray(8)
    results = []
    rng = np.random.default_rng(42)

    for n in SIZES:
        n = (n // 16) * 16   # must be multiple of 16 for the SIMD main loop

        buf = rng.standard_normal(2 * n).astype(np.float32)
        data = bytes(buf)
        a = np.frombuffer(data, dtype=np.float32, count=n)
        b = np.frombuffer(data, dtype=np.float32, offset=n * 4)

        (a + b).sum()  # warmup
        numpy_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: (a + b).sum()
        ))

        engine.execute_into(alg, data, out)  # warmup
        pybase_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: engine.execute_into(alg, data, out)
        ))

        expected = float((a + b).sum())
        pybase_result = struct.unpack("<d", bytes(out))[0]
        mag = max(abs(expected), 1.0)
        verified = abs(pybase_result - expected) / mag < 0.01

        results.append(harness.BenchResult(
            name=f"VecAdd  ({harness.format_count(n)})",
            python_ms=numpy_ms,
            pybase_ms=pybase_ms,
            verified=verified,
        ))

    return results


def _run_clampsum(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    engine, alg = py_base.load(algo_path)
    out = bytearray(8)
    results = []
    rng = np.random.default_rng(99)

    for n in SIZES:
        a = rng.standard_normal(n).astype(np.float32)
        data = bytes(a)

        np.clip(a, CLAMP_LO, CLAMP_HI).sum()  # warmup
        numpy_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: np.clip(a, CLAMP_LO, CLAMP_HI).sum()
        ))

        engine.execute_into(alg, data, out)  # warmup
        pybase_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: engine.execute_into(alg, data, out)
        ))

        expected = float(np.clip(a, CLAMP_LO, CLAMP_HI).sum())
        pybase_result = struct.unpack("<d", bytes(out))[0]
        mag = max(abs(expected), 1.0)
        verified = abs(pybase_result - expected) / mag < 0.01

        results.append(harness.BenchResult(
            name=f"ClampSum({harness.format_count(n)})",
            python_ms=numpy_ms,
            pybase_ms=pybase_ms,
            verified=verified,
        ))

    return results


def run(vecadd_path: str, clampsum_path: str, rounds: int) -> list[harness.BenchResult]:
    return _run_vecadd(vecadd_path, rounds) + _run_clampsum(clampsum_path, rounds)
