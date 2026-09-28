import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional


#: Which of eager / compiled supplied each reported PyTorch time, in the order
#: the benchmarks ran. Printed under the table by `print_table`.
TORCH_MODE: list[tuple[str, str]] = []


def median_of(n: int, fn: Callable[[], float]) -> float:
    times = [fn() for _ in range(n)]
    return statistics.median(times)


def time_ms(fn: Callable[[], None]) -> float:
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000.0


def torch_median(
    fn: Callable[[], object],
    rounds: int,
    inner: int = 1,
    label: str = "",
    sync_each: bool = False,
) -> float:
    """Median CUDA-synchronised time in ms for a zero-argument torch callable.

    Each reference is timed both eagerly and under `torch.compile`, and the
    faster of the two is reported. Which one wins varies by workload rather
    than by run: compilation fuses an elementwise chain, but an op that is
    already launch-bound has nothing to fuse and can come out slower. Choosing
    per benchmark is what a PyTorch user tuning each of these would arrive at.
    Which one won is recorded in `TORCH_MODE` and printed under the table, so a
    row where compilation raised is not reported as one where it lost.

    `fn` must return its result. A discarded result is not merely wasteful:
    `torch.compile` will delete the whole computation as dead code, which
    reports a time for a kernel that never ran.

    An op too short to time on its own is repeated `inner` times and the total
    divided. The repetition is outside `fn` rather than inside it so that each
    iteration is a separate call the compiler cannot fold into its neighbour.

    `sync_each` synchronises after every one of those repetitions instead of
    once at the end. Use it wherever the other side of the comparison pays a
    device round trip per call --- an artifact entry that launches and
    synchronises does --- because otherwise this side pays one synchronisation
    against that side's `inner`, and the rows where that matters are exactly
    the short ones whose claim is about per-dispatch cost.
    """
    import torch

    def run(f: Callable[[], object]) -> float:
        def synced() -> None:
            for _ in range(inner):
                f()
                if sync_each:
                    torch.cuda.synchronize()
            if not sync_each:
                torch.cuda.synchronize()

        for _ in range(3):
            synced()
        return median_of(rounds, lambda: time_ms(synced)) / inner

    eager = run(fn)
    try:
        compiled = run(torch.compile(fn))
    except Exception as e:
        TORCH_MODE.append((label, f"eager (torch.compile raised {type(e).__name__})"))
        return eager
    if compiled < eager:
        TORCH_MODE.append((label, "compiled"))
        return compiled
    TORCH_MODE.append((label, "eager"))
    return eager


@dataclass
class BenchResult:
    name: str
    python_ms: Optional[float]
    pybase_ms: Optional[float]
    verified: Optional[bool]


def _fmt_ms(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    if v >= 10.0:
        return f"{v:.1f}ms"
    if v >= 1.0:
        return f"{v:.2f}ms"
    return f"{v:.3f}ms"


def _fmt_check(v: Optional[bool]) -> str:
    if v is True:
        return "✓"
    if v is False:
        return "✗"
    return "—"


def print_table(results: list[BenchResult], col_a: str = "Python") -> None:
    name_w = 20
    col_w = 12

    print()
    print(
        f"{'Benchmark':<{name_w}} {col_a:>{col_w}} {'PyO3':>{col_w}} {'Check':>6}"
    )
    print("-" * (name_w + col_w * 2 + 6 + 3))

    for r in results:
        print(
            f"{r.name:<{name_w}} {_fmt_ms(r.python_ms):>{col_w}} "
            f"{_fmt_ms(r.pybase_ms):>{col_w}} {_fmt_check(r.verified):>6}"
        )
    print()
    if TORCH_MODE:
        failed = [(n, m) for n, m in TORCH_MODE if "raised" in m]
        shown = failed if failed else TORCH_MODE
        print("PyTorch reference timed as:")
        for n, m in shown:
            print(f"  {n or '(unnamed)'}: {m}")
        print()


def format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)
