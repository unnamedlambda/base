import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional


def median_of(n: int, fn: Callable[[], float]) -> float:
    times = [fn() for _ in range(n)]
    return statistics.median(times)


def time_ms(fn: Callable[[], None]) -> float:
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000.0


@dataclass
class BenchResult:
    name: str
    python_ms: Optional[float]
    pybase_ms: Optional[float]
    verified: Optional[bool]


def _fmt_ms(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.1f}ms"


def _fmt_check(v: Optional[bool]) -> str:
    if v is True:
        return "✓"
    if v is False:
        return "✗"
    return "—"


def print_table(results: list[BenchResult]) -> None:
    name_w = 20
    col_w = 12

    print()
    print(
        f"{'Benchmark':<{name_w}} {'Python':>{col_w}} {'PyO3':>{col_w}} {'Check':>6}"
    )
    print("-" * (name_w + col_w * 2 + 6 + 3))

    for r in results:
        print(
            f"{r.name:<{name_w}} {_fmt_ms(r.python_ms):>{col_w}} "
            f"{_fmt_ms(r.pybase_ms):>{col_w}} {_fmt_check(r.verified):>6}"
        )
    print()


def format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)
