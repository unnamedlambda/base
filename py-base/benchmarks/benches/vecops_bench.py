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

RowDot:   X @ w
  numpy:   matrix-vector multiply over a 2D f32 array and a shared weight vector.
  py_base: row-wise dot-product kernel over packed [rows, cols, X, w].

RowAffine: (X * scale + bias).sum(axis=1)
  numpy:   broadcasted affine transform followed by row-wise reduction.
  py_base: single-pass row-wise affine+reduce over packed [rows, cols, X, scale, bias].
"""

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
import py_base

SIZES = [1_000_000, 10_000_000, 50_000_000]
ROW_SHAPES = [(2048, 256), (4096, 256), (4096, 512)]

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


def _run_rowdot(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    engine, alg = py_base.load(algo_path)
    results = []
    rng = np.random.default_rng(123)

    for rows, cols in ROW_SHAPES:
        x = rng.standard_normal((rows, cols)).astype(np.float32)
        w = rng.standard_normal(cols).astype(np.float32)
        out = bytearray(rows * 4)
        data = struct.pack("<QQ", rows, cols) + x.tobytes() + w.tobytes()

        (x @ w)
        numpy_ms = harness.median_of(rounds, lambda: harness.time_ms(lambda: x @ w))

        engine.execute_into(alg, data, out)
        pybase_ms = harness.median_of(
            rounds, lambda: harness.time_ms(lambda: engine.execute_into(alg, data, out))
        )

        expected = x @ w
        pybase_result = np.frombuffer(bytes(out), dtype=np.float32)
        mag = max(float(np.abs(expected).max()), 1.0)
        verified = float(np.max(np.abs(pybase_result - expected))) / mag < 0.01

        results.append(
            harness.BenchResult(
                name=f"RowDot  ({harness.format_count(rows)}x{cols})",
                python_ms=numpy_ms,
                pybase_ms=pybase_ms,
                verified=verified,
            )
        )

    return results


def _run_row_affine_reduce(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    engine, alg = py_base.load(algo_path)
    results = []
    rng = np.random.default_rng(321)

    for rows, cols in ROW_SHAPES:
        x = rng.standard_normal((rows, cols)).astype(np.float32)
        scale = rng.standard_normal(cols).astype(np.float32)
        bias = rng.standard_normal(cols).astype(np.float32)
        out = bytearray(rows * 4)
        data = (
            struct.pack("<QQ", rows, cols)
            + x.tobytes()
            + scale.tobytes()
            + bias.tobytes()
        )

        ((x * scale + bias).sum(axis=1))
        numpy_ms = harness.median_of(
            rounds, lambda: harness.time_ms(lambda: (x * scale + bias).sum(axis=1))
        )

        engine.execute_into(alg, data, out)
        pybase_ms = harness.median_of(
            rounds, lambda: harness.time_ms(lambda: engine.execute_into(alg, data, out))
        )

        expected = (x * scale + bias).sum(axis=1)
        pybase_result = np.frombuffer(bytes(out), dtype=np.float32)
        mag = max(float(np.abs(expected).max()), 1.0)
        verified = float(np.max(np.abs(pybase_result - expected))) / mag < 0.01

        results.append(
            harness.BenchResult(
                name=f"RowAff  ({harness.format_count(rows)}x{cols})",
                python_ms=numpy_ms,
                pybase_ms=pybase_ms,
                verified=verified,
            )
        )

    return results


def run(
    vecadd_path: str,
    clampsum_path: str,
    rowdot_path: str,
    rowaffine_path: str,
    rounds: int,
) -> list[harness.BenchResult]:
    return (
        _run_vecadd(vecadd_path, rounds)
        + _run_clampsum(clampsum_path, rounds)
        + _run_rowdot(rowdot_path, rounds)
        + _run_row_affine_reduce(rowaffine_path, rounds)
    )
