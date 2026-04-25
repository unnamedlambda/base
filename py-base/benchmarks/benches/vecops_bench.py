"""
VecAdd benchmark: numpy vs py_base.

Both start from the same contiguous bytes buffer — the natural py_base convention.
Data lives in one flat allocation; no split into separate Python objects.

  numpy:   np.frombuffer views (zero-copy) → (a + b).sum()
           Two-pass: allocates a full intermediate array for (a+b), then sums it.

  py_base: passes buffer directly → 4x-unrolled SIMD f32x4 loop
           Single pass: adds and accumulates in registers, zero intermediate allocation.

Payload: [A floats: n * f32][B floats: n * f32]  (back to back, n inferred from data_len/8)
Out:     bytearray(8)  (f64 result)
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
import py_base

try:
    import numpy as np
except ImportError:
    np = None

SIZES = [1_000_000, 10_000_000, 50_000_000]


def close_enough(a: float, b: float, tol: float = 0.01) -> bool:
    if a != a or b != b:  # NaN check
        return False
    mag = max(abs(a), abs(b), 1.0)
    return abs(a - b) / mag < tol


def run(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    if np is None:
        print("numpy not installed — skipping vecops benchmark", file=sys.stderr)
        return []

    engine, alg = py_base.load(algo_path)
    results = []

    rng = np.random.default_rng(42)

    for n in SIZES:
        # Must be multiple of 16 for the SIMD main loop.
        n = (n // 16) * 16

        # Single contiguous buffer: [A floats][B floats] — the py_base convention.
        # A real program would fill this buffer directly as data is generated.
        buf = rng.standard_normal(2 * n).astype(np.float32)
        data = bytes(buf)  # pre-built once outside timing; both sides start here
        out = bytearray(8)

        a = np.frombuffer(data, dtype=np.float32, count=n)
        b = np.frombuffer(data, dtype=np.float32, offset=n * 4)

        # numpy: zero-copy views into the shared buffer, then two-pass compute
        # (allocates intermediate array for a+b, then sums it separately)
        numpy_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: (a + b).sum()
        ))

        # py_base: buffer passed directly, single-pass SIMD, no intermediate allocation
        engine.execute_into(alg, data, out)  # warmup
        pybase_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: engine.execute_into(alg, data, out)
        ))

        expected = float((a + b).sum())
        pybase_result = struct.unpack("<d", bytes(out))[0]
        verified = close_enough(expected, pybase_result)

        results.append(harness.BenchResult(
            name=f"VecAdd ({harness.format_count(n)})",
            python_ms=numpy_ms,
            pybase_ms=pybase_ms,
            verified=verified,
        ))

    return results
