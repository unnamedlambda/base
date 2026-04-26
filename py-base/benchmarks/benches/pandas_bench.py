"""
Revenue aggregation benchmark: pandas vs py_base.

Both sides pre-build their data structure once (outside timing), then time
only the analytical computation. This isolates steady-state analysis cost.

  pandas:  pre-built DataFrame; timed portion: revenue Series (alloc + multiply
           pass) → groupby(category).sum() (hash aggregation) → max()
           Two O(n) passes with one intermediate Series allocation.

  py_base: pre-built packed rows (bytes); timed portion: single linear scan
           over 12-byte rows, accumulating revenue per category in 16 f64
           accumulators, then pairwise fmax reduction.
           One O(n) pass, zero intermediate allocations.

Row layout (py_base binary format): [category: u32][price: f32][quantity: f32] = 12 bytes
N_CATS = 16  (categories 0–15, uniformly distributed)
Result:  f64 — max total revenue across all 16 categories
"""

import struct
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
import py_base

try:
    import pandas as pd
except ImportError:
    pd = None

N_CATS = 16
SIZES = [1_000_000, 5_000_000, 10_000_000]


def build_packed(categories: np.ndarray, prices: np.ndarray, quantities: np.ndarray) -> bytes:
    rows = np.zeros(len(categories), dtype=np.dtype([
        ("category", "<u4"),
        ("price",    "<f4"),
        ("quantity", "<f4"),
    ]))
    rows["category"] = categories
    rows["price"]    = prices
    rows["quantity"] = quantities
    return bytes(rows)


def run(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    if pd is None:
        print("pandas not installed — skipping pandas benchmark", file=sys.stderr)
        return []

    engine, alg = py_base.load(algo_path)
    out = bytearray(8)
    results = []
    rng = np.random.default_rng(42)

    for n in SIZES:
        categories = rng.integers(0, N_CATS, n, dtype=np.int32)
        prices     = rng.uniform(1.0, 100.0, n).astype(np.float32)
        quantities = rng.uniform(1.0, 10.0,  n).astype(np.float32)

        # Pre-build each side's native data structure once — not part of timing.
        # pandas: wraps numpy arrays (zero-copy views), O(1)
        # py_base: packs columns into contiguous rows, O(n) — same as writing a binary file
        df   = pd.DataFrame({"category": categories, "price": prices, "quantity": quantities})
        data = build_packed(categories, prices, quantities)

        # pandas: revenue Series (O(n) alloc + multiply) → groupby sum → max
        (df["price"] * df["quantity"]).groupby(df["category"]).sum().max()  # warmup
        pandas_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: (df["price"] * df["quantity"]).groupby(df["category"]).sum().max()
        ))
        expected = float((df["price"] * df["quantity"]).groupby(df["category"]).sum().max())

        # py_base: single-pass CLIF scan → max revenue
        engine.execute_into(alg, data, out)  # warmup
        pybase_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: engine.execute_into(alg, data, out)
        ))

        pybase_result = struct.unpack("<d", bytes(out))[0]
        verified = abs(pybase_result - expected) / max(abs(expected), 1.0) < 1e-3

        results.append(harness.BenchResult(
            name=f"RevAgg ({harness.format_count(n)})",
            python_ms=pandas_ms,
            pybase_ms=pybase_ms,
            verified=verified,
        ))

    return results
