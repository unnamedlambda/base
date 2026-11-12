"""Dataset access, exact search and recall, shared by every ANN script.

A prepared dataset is `data/ann/<name>/` (see prepare.py). Exact search here is
the ground truth every recall is scored against: an fp32 GEMM in cupy (cuBLAS
SGEMM, never TF32) followed by a sort, checked against float64 on a sample.
cuvs's own `brute_force` is not used for it: on SIFT-1M its top 10 agree with
float64 only 89% of the time.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))  # benchmarks/: the shared meter
from meter import METER  # noqa: E402  (re-exported to the harnesses)

DATA = os.path.join(HERE, "..", "..", "data", "ann")
# Query rows per distance tile: 100 x 1M fp32 is 400 MB, its sort ~1.6 GB.
TILE = 100


def run_dir() -> str:
    """This run's result directory: results/<ANN_RUN>/, set by run.sh to
    `<time>-<git rev>`. A script started by hand writes to results/adhoc/."""
    d = os.path.join(HERE, "results", os.environ.get("ANN_RUN", "adhoc"))
    os.makedirs(d, exist_ok=True)
    return d


class Out:
    """Appends result rows to <run dir>/<harness>-<dataset>.jsonl."""

    def __init__(self, harness: str, dataset: str, engine: str):
        self.f = open(os.path.join(run_dir(), f"{harness}-{dataset}.jsonl"), "a")
        self.tag = {"dataset": dataset, "engine": engine}

    def __call__(self, **row):
        row = {**self.tag, **row}
        self.f.write(json.dumps(row, default=lambda x: list(x) if isinstance(x, tuple) else str(x)) + "\n")
        self.f.flush()
        print({k: v for k, v in row.items() if not (isinstance(v, list) and len(v) > 20)}, flush=True)


DEVICE_NO_FIT = 3  # exit status: the engine ran out of device memory


def out_of_memory(e: Exception) -> bool:
    return isinstance(e, MemoryError) or "out_of_memory" in str(e) or "cudaErrorMemoryAllocation" in str(e)


def refused(what: str, e: Exception) -> None:
    """A setting the engine does not accept, which ends its sweep. Running
    out of memory is not a refusal: it propagates to `harness`."""
    if out_of_memory(e):
        raise e
    print(f"{what} refused: {type(e).__name__}: {str(e).splitlines()[0][:300]}", flush=True)


def harness(main) -> None:
    """Run a harness's main. Running out of device memory exits
    DEVICE_NO_FIT, which run.sh records as no-fit (the engine does not fit
    this card), not as a broken harness."""
    import sys
    import traceback

    try:
        main()
    except Exception as e:
        if not out_of_memory(e):
            raise
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(DEVICE_NO_FIT)  # past CUDA teardown, which can fault after an OOM


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return time.perf_counter() - t, out




def idle_power(seconds: float = 10.0) -> None:
    """The machine at rest, before a run: mean GPU and CPU-package power over
    `seconds`, written to <run dir>/idle.jsonl. Reports subtract it to give
    the energy a workload adds (dynamic) beside what the meter saw (total)."""
    start = METER.read()
    time.sleep(seconds)
    e = METER.since(start)
    row = dict(
        window_s=e["window_s"],
        gpu_w=e["gpu_j"] / e["window_s"] if e["gpu_j"] is not None else None,
        cpu_pkg_w=e["cpu_pkg_j"] / e["window_s"] if e["cpu_pkg_j"] is not None else None,
    )
    with open(os.path.join(run_dir(), "idle.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(row, flush=True)


class Dataset:
    def __init__(self, name: str):
        self.name = name
        self.folder = os.path.join(DATA, name)
        self.base = np.load(self.path("base.npy"), mmap_mode="r")
        self.queries = np.load(self.path("queries.npy"))
        self.metric = open(self.path("metric.txt")).read().strip()

    def path(self, f: str) -> str:
        return os.path.join(self.folder, f)

    def load(self, f: str):
        return np.load(self.path(f))


def drop_cache(x) -> None:
    """Drop the page cache behind a memory-mapped array (no-op otherwise).

    A large base read through the page cache fills a memory cap with file
    pages, and reclaim pressure at a cap is what makes systemd-oomd kill the
    session; dropping them as they are used keeps the cap's pressure at zero.
    """
    path = getattr(x, "filename", None)
    if path is None and isinstance(getattr(x, "base", None), np.memmap):
        path = x.base.filename
    if path:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def to_device(x):
    """A host array (possibly memory-mapped) on the device, copied in slices."""
    import cupy as cp

    out = cp.empty(x.shape, cp.float32)
    for s in range(0, len(x), 100000):
        out[s : s + 100000] = cp.asarray(np.ascontiguousarray(x[s : s + 100000], np.float32))
        if s % 1000000 == 0:
            drop_cache(x)
    drop_cache(x)
    return out


class Exact:
    """Exact top-k over a device-resident base, restricted to an active row set."""

    def __init__(self, base, metric: str):
        import cupy as cp

        self.cp = cp
        self.metric = metric
        self.base = to_device(base)
        self.norms = (self.base * self.base).sum(1) if metric == "sqeuclidean" else None
        # A tile's distances and selection take about 24 bytes per base row
        # per query (fp32 keys, int64 indices, the partition's scratch): keep
        # a tile near 1.5 GB beside the base.
        self.tile = max(1, min(TILE, int(1.5e9 // (24 * len(self.base)))))

    def search(self, queries: np.ndarray, k: int, active=None) -> np.ndarray:
        """Rows of the `k` nearest base vectors per query, nearest first.

        `active` (host bool mask over base rows) restricts the candidates.
        """
        cp = self.cp
        penalty = None
        if active is not None:
            penalty = cp.where(cp.asarray(active), 0.0, cp.inf).astype(cp.float32)
        out = np.empty((len(queries), k), np.int32)
        for s in range(0, len(queries), self.tile):
            q = cp.asarray(queries[s : s + self.tile])
            dots = q @ self.base.T
            # Ranking key, smaller is nearer. ||q||^2 is constant per row.
            key = self.norms[None, :] - 2 * dots if self.metric == "sqeuclidean" else -dots
            if penalty is not None:
                key += penalty[None, :]
            # The k smallest, unordered (exact selection), then ordered.
            part = cp.argpartition(key, k - 1, axis=1)[:, :k]
            order = cp.argsort(cp.take_along_axis(key, part, 1), axis=1)
            out[s : s + self.tile] = cp.asnumpy(cp.take_along_axis(part, order, 1))
            del dots, key
        return out


def float64_check(base, queries, found: np.ndarray, metric: str, n: int = 20, active=None) -> float:
    """Agreement of `found` with a float64 top-k on the first `n` queries.

    The base is read once, in slices of 256 MB of float64, each scored for
    all `n` queries (never a whole n x rows x d broadcast: tens of GB).
    """
    k = found.shape[1]
    rows = np.arange(len(base)) if active is None else np.nonzero(active)[0]
    qs = queries[:n].astype(np.float64)
    best = np.full((n, 0), 0.0)
    best_rows = np.full((n, 0), 0, np.int64)
    step = max(1, (256 << 20) // (base.shape[1] * 8))
    for s in range(0, len(rows), step):
        r = rows[s : s + step]
        b = np.asarray(base[r], np.float64)
        if metric == "sqeuclidean":
            d = (b * b).sum(1)[None, :] - 2 * qs @ b.T  # ||q||^2 is constant per query
        else:
            d = -(qs @ b.T)
        best = np.concatenate([best, d], 1)
        best_rows = np.concatenate([best_rows, np.broadcast_to(r, (n, len(r)))], 1)
        keep = np.argsort(best, axis=1, kind="stable")[:, :k]
        best, best_rows = np.take_along_axis(best, keep, 1), np.take_along_axis(best_rows, keep, 1)
        del b, d
        drop_cache(base)
    hits = sum(len(np.intersect1d(best_rows[i], found[i])) for i in range(n))
    return hits / (n * k)


def distances(base, queries, ids: np.ndarray, metric: str) -> np.ndarray:
    """float32 (q, k) distance of each listed item to its query, in the
    ranking key's units (squared L2, or minus the inner product); inf where an
    id is -1. Computed in float64 from the rows themselves."""
    ids = np.asarray(ids)
    out = np.full(ids.shape, np.inf, np.float64)
    ok = ids >= 0
    qi, _ = np.nonzero(ok)
    rows, at = np.unique(ids[ok], return_inverse=True)  # each row read once, in order
    x, q = np.asarray(base[rows], np.float64)[at], queries[qi].astype(np.float64)
    out[ok] = ((x - q) ** 2).sum(1) if metric == "sqeuclidean" else -(x * q).sum(1)
    return out.astype(np.float32)


def recall(found: np.ndarray, gt: np.ndarray, gt_dist: np.ndarray, found_dist: np.ndarray) -> float:
    """recall@10 as ann-benchmarks and big-ann-benchmarks count it: a returned
    item is a hit when it is as near as the 10th true neighbour (so ties at
    that distance count). Truth may be -1 padded (a filter admitting fewer than
    10 items): such a query counts only its real neighbours."""
    real = (gt >= 0).sum(1)
    last = np.take_along_axis(gt_dist, np.maximum(real - 1, 0)[:, None], 1)
    tol = 1e-5 * np.abs(last) + 1e-6  # float32 rounding of equal distances
    hits = ((found >= 0) & (found_dist <= last + tol)).sum(1)
    return float(np.minimum(hits, real).sum() / max(1, real.sum()))
