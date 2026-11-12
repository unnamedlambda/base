"""Sorting and top-k on the GPU, measured as the field measures them: on data
already on the device, keys per second.

    python bench.py sort <engine>     # appends to <run dir>/sort.jsonl
    python bench.py topk <engine>     # appends to <run dir>/topk.jsonl

**sort**: (u32 key, u32 value) pairs, stable by key, as CUB benchmarks its
`DeviceRadixSort::SortPairs`, at each size in SIZES and each key
distribution in DISTS (uniform, Zipfian 0.99 over 2^20 keys, already sorted,
16 distinct keys). Engines: `base`, and `cub` (`cub_sort.cu`: CUB's
`SortPairs` called directly, as the CUDA toolkit ships it).

**topk**: the k smallest of each row of a float32 matrix, with their column
indices, at each (rows, columns, k) in SHAPES. Engines: `base`, and `torch`
(PyTorch's `topk`, the batched top-k production ML code calls).

Every answer is checked before its time counts: a sort's keys and values
against numpy's stable argsort, exactly; a top-k's keys against the exact k
smallest of each row, and each returned column against the key it names
(ties may pick any of the equal columns). The time is the median of REPS
runs, each ended by a device synchronisation; every run's time is kept
(`samples`), so a report can give the spread beside the median.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
BUILD = os.path.join(HERE, "build")
ART = os.path.join(BUILD, "base", "sort.cbor")
CUB = os.path.join(BUILD, "cub_sort")

SIZES = [1 << 20, 1 << 24, 1 << 26]
DISTS = ["uniform", "zipf", "sorted", "few"]
SHAPES = [(1000, 16384, 10), (1000, 16384, 100), (100, 1 << 20, 100), (10000, 1024, 32)]
REPS = 10


def run_dir() -> str:
    d = os.path.join(HERE, "results", os.environ.get("SORT_RUN", "adhoc"))
    os.makedirs(d, exist_ok=True)
    return d


def emit(kind, **row):
    with open(os.path.join(run_dir(), f"{kind}.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(row, flush=True)


def keys_for(dist, n, rng):
    if dist == "uniform":
        return rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
    if dist == "zipf":
        # Zipfian 0.99 over 2^20 ranks, each rank a fixed random key
        ranks = np.arange(1, (1 << 20) + 1, dtype=np.float64)
        p = ranks**-0.99
        p /= p.sum()
        table = rng.integers(0, 2**32, 1 << 20, dtype=np.uint64).astype(np.uint32)
        return table[rng.choice(1 << 20, n, p=p)]
    if dist == "sorted":
        return np.sort(rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32))
    if dist == "few":
        return rng.integers(0, 16, n, dtype=np.uint32) * np.uint32(0x10000001)
    raise ValueError(dist)


def times(fn, sync=lambda: None):
    """Every one of REPS timed runs, after one untimed."""
    fn()
    sync()
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return ts


class Base:
    """base's artifact (lean/algorithms/Ann/Sort.lean): its functions keep
    the data on the device between calls, and each call returns synchronised."""

    def __init__(self):
        from py_base import Driver

        self.drv = Driver(open(ART, "rb").read())
        self.f = {n: self.drv.function(n) for n in ("load", "sort", "fetch", "tload", "topk", "tfetch")}

    def call(self, name, data=b"", out=None):
        st = self.f[name](data, out)
        if st != 0:
            raise RuntimeError(f"{name} returned {st}")

    def sort(self, keys, vals):
        n = len(keys)
        self.call("load", np.array([n], np.uint64).tobytes() + keys.tobytes() + vals.tobytes())
        def run():
            self.call("sort")

        run()
        out = np.empty(2 * n, np.uint32)
        self.call("fetch", b"", out.view(np.uint8))
        return out[:n], out[n:], times(run)

    def topk(self, x, k):
        rows = x.shape[0]
        self.call("tload", np.array([x.shape[0], x.shape[1]], np.uint64).tobytes() + x.tobytes())
        kb = np.array([k], np.uint64).tobytes()

        def run():
            self.call("topk", kb)

        run()
        out = np.empty(2 * rows * k, np.uint32)
        self.call("tfetch", b"", out.view(np.uint8))
        vals = out[: rows * k].view(np.float32).reshape(rows, k)
        idx = out[rows * k :].view(np.int32).reshape(rows, k)
        return vals, idx, times(run)


class Cub:
    """`cub_sort` on the same keys (written to a file), values 0 … n-1: it
    times `SortPairs` on the device and writes the sorted pairs back."""

    def sort(self, keys, vals):
        path = os.path.join(BUILD, "keys.bin")
        keys.tofile(path)
        out = os.path.join(BUILD, "sorted.bin")
        r = subprocess.run([CUB, path, str(len(keys)), str(REPS), out], capture_output=True, text=True, check=True)
        secs = [float(x) for x in r.stdout.split()]
        got = np.fromfile(out, np.uint32)
        os.remove(path)
        os.remove(out)
        n = len(keys)
        return got[:n], got[n:], secs


class Torch:
    """PyTorch's `topk` (smallest, unsorted, as base's) on the device."""

    def __init__(self):
        import torch

        self.torch = torch

    def topk(self, x, k):
        t = self.torch
        xd = t.from_numpy(x).cuda()
        res = {}

        def run():
            res["v"], res["i"] = t.topk(xd, k, dim=1, largest=False, sorted=False)

        secs = times(run, t.cuda.synchronize)
        return res["v"].cpu().numpy(), res["i"].cpu().numpy().astype(np.int32), secs


def run_sort(engine):
    eng = {"base": Base, "cub": Cub}[engine]()
    rng = np.random.default_rng(0)
    for n in SIZES:
        for dist in DISTS:
            keys = keys_for(dist, n, rng)
            vals = np.arange(n, dtype=np.uint32)
            k, v, ts = eng.sort(keys, vals)
            p = np.argsort(keys, kind="stable")
            if not (np.array_equal(k, keys[p]) and np.array_equal(v, vals[p])):
                raise SystemExit(f"{engine}: wrong sort at n={n} {dist}")
            secs = float(np.median(ts))
            emit("sort", engine=engine, n=n, dist=dist, seconds=secs, keys_per_s=n / secs, samples=ts)


def run_topk(engine):
    eng = {"base": Base, "torch": Torch}[engine]()
    rng = np.random.default_rng(1)
    for rows, cols, k in SHAPES:
        x = rng.standard_normal((rows, cols), dtype=np.float32)
        vals, idx, ts = eng.topk(x, k)
        want = np.sort(np.partition(x, k - 1, axis=1)[:, :k], axis=1)
        named = np.take_along_axis(x, idx.astype(np.int64), axis=1)
        if not (np.array_equal(np.sort(vals, axis=1), want) and np.array_equal(np.sort(named, axis=1), want)):
            raise SystemExit(f"{engine}: wrong top-k at {rows}x{cols} k={k}")
        secs = float(np.median(ts))
        emit("topk", engine=engine, rows=rows, cols=cols, k=k, seconds=secs, keys_per_s=rows * cols / secs,
             samples=ts)


if __name__ == "__main__":
    {"sort": run_sort, "topk": run_topk}[sys.argv[1]](sys.argv[2])
