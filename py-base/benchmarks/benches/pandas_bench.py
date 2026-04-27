"""
Pandas benchmarks: pandas vs py_base.

RevAgg:       max total revenue per category
  pandas:  (df["price"] * df["quantity"]).groupby(df["category"]).sum().max()
           One O(n) multiply + one O(n) groupby-sum. Two passes, one intermediate Series.

FilterRevAgg: same but only rows where price > 50.0
  pandas:  df[df["price"] > 50.0] → boolean mask (O(n)) + filtered copy,
           then revenue multiply + groupby.
           Three passes, two intermediate allocations.

  py_base (both): single linear scan, 16 f64 accumulators. FilterRevAgg adds only
           a branch — still one pass, zero extra allocations.

Row layout (py_base binary): [category: u32][price: f32][quantity: f32] = 12 bytes
N_CATS = 16. Both sides pre-build their native data structure once; only computation timed.
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
PRICE_THRESHOLD = 50.0


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


def _run_revagg(engine, alg, df, data: bytes, rounds: int, n: int) -> harness.BenchResult:
    out = bytearray(8)

    (df["price"] * df["quantity"]).groupby(df["category"]).sum().max()  # warmup
    pandas_ms = harness.median_of(rounds, lambda: harness.time_ms(
        lambda: (df["price"] * df["quantity"]).groupby(df["category"]).sum().max()
    ))
    expected = float((df["price"] * df["quantity"]).groupby(df["category"]).sum().max())

    engine.execute_into(alg, data, out)  # warmup
    pybase_ms = harness.median_of(rounds, lambda: harness.time_ms(
        lambda: engine.execute_into(alg, data, out)
    ))

    pybase_result = struct.unpack("<d", bytes(out))[0]
    verified = abs(pybase_result - expected) / max(abs(expected), 1.0) < 1e-3

    return harness.BenchResult(
        name=f"RevAgg      ({harness.format_count(n)})",
        python_ms=pandas_ms,
        pybase_ms=pybase_ms,
        verified=verified,
    )


def _run_filter_revagg(engine, alg, df, data: bytes, rounds: int, n: int) -> harness.BenchResult:
    out = bytearray(8)

    def pandas_filter():
        filtered = df[df["price"] > PRICE_THRESHOLD]
        return float((filtered["price"] * filtered["quantity"])
                     .groupby(filtered["category"]).sum().max())

    pandas_filter()  # warmup
    pandas_ms = harness.median_of(rounds, lambda: harness.time_ms(pandas_filter))
    expected = pandas_filter()

    engine.execute_into(alg, data, out)  # warmup
    pybase_ms = harness.median_of(rounds, lambda: harness.time_ms(
        lambda: engine.execute_into(alg, data, out)
    ))

    pybase_result = struct.unpack("<d", bytes(out))[0]
    verified = abs(pybase_result - expected) / max(abs(expected), 1.0) < 1e-3

    return harness.BenchResult(
        name=f"FilterRevAgg({harness.format_count(n)})",
        python_ms=pandas_ms,
        pybase_ms=pybase_ms,
        verified=verified,
    )


def run(revagg_path: str, filter_path: str, rounds: int) -> list[harness.BenchResult]:
    if pd is None:
        print("pandas not installed — skipping pandas benchmark", file=sys.stderr)
        return []

    eng_rev,    alg_rev    = py_base.load(revagg_path)
    eng_filter, alg_filter = py_base.load(filter_path)
    results = []
    rng = np.random.default_rng(42)

    for n in SIZES:
        categories = rng.integers(0, N_CATS, n, dtype=np.int32)
        prices     = rng.uniform(1.0, 100.0, n).astype(np.float32)
        quantities = rng.uniform(1.0, 10.0,  n).astype(np.float32)

        df   = pd.DataFrame({"category": categories, "price": prices, "quantity": quantities})
        data = build_packed(categories, prices, quantities)

        results.append(_run_revagg(eng_rev, alg_rev, df, data, rounds, n))
        results.append(_run_filter_revagg(eng_filter, alg_filter, df, data, rounds, n))

    return results
