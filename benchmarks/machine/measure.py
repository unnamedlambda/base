"""This machine's ceilings, as result lines every suite's roofline reads.

    python measure.py           # writes results/<host>.jsonl (replacing it)

Measured here, never taken from a data sheet:

    flash, per drive (the NVMe at /mnt/nvme, the SATA SSD at /):
        sequential read, O_DIRECT, 4 MiB requests          bytes/s
        4 KiB random read, O_DIRECT, one at a time         seconds (latency)
        4 KiB random read, O_DIRECT, 32 threads            reads/s
        4 KiB write + fsync                                 seconds (latency)
    DRAM: one core copying 1 GiB arrays                     bytes/s
    GPU (when cupy is importable):
        device-to-device copy                               bytes/s
        host-to-device and device-to-host, pinned          bytes/s

Each figure is the median of three. Files are made fresh and removed.
"""

import json
import mmap
import os
import socket
import statistics
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVES = {"nvme": "/mnt/nvme/bench", "sata": os.path.join(HERE, "..", "..", "data")}
FILE_BYTES = 2 << 30
BLOCK = 4 << 20


def med(f, n=3):
    return statistics.median(f() for _ in range(n))


def aligned(nbytes):
    return mmap.mmap(-1, nbytes)  # anonymous maps are page-aligned, as O_DIRECT needs


def make_file(path):
    buf = aligned(BLOCK)
    buf.write(os.urandom(BLOCK))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o600)
    for i in range(FILE_BYTES // BLOCK):
        os.pwritev(fd, [buf], i * BLOCK)
    os.fsync(fd)
    os.close(fd)


def seq_read(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = aligned(BLOCK)
    t = time.perf_counter()
    for off in range(0, FILE_BYTES, BLOCK):
        os.preadv(fd, [buf], off)
    dt = time.perf_counter() - t
    os.close(fd)
    return FILE_BYTES / dt


def rand_read_latency(path, n=2000):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = aligned(4096)
    offs = np.random.default_rng(0).integers(0, FILE_BYTES // 4096, n) * 4096
    lat = []
    for off in offs:
        t = time.perf_counter()
        os.preadv(fd, [buf], int(off))
        lat.append(time.perf_counter() - t)
    os.close(fd)
    return statistics.median(lat)


def rand_read_iops(path, threads=32, per=2000):
    def worker(seed):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        buf = aligned(4096)
        for off in np.random.default_rng(seed).integers(0, FILE_BYTES // 4096, per) * 4096:
            os.preadv(fd, [buf], int(off))
        os.close(fd)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t = time.perf_counter()
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    return threads * per / (time.perf_counter() - t)


def fsync_latency(path, n=200):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    block = os.urandom(4096)
    lat = []
    for i in range(n):
        os.pwrite(fd, block, i * 4096)
        t = time.perf_counter()
        os.fsync(fd)
        lat.append(time.perf_counter() - t)
    os.close(fd)
    os.unlink(path)
    return statistics.median(lat)


def dram_copy():
    a = np.ones(1 << 27, np.float64)  # 1 GiB
    b = np.empty_like(a)

    def once():
        t = time.perf_counter()
        np.copyto(b, a)
        return 2 * a.nbytes / (time.perf_counter() - t)  # read + write

    once()
    return med(once)


def gpu():
    try:
        import cupy as cp
    except ImportError:
        return {}
    n = 1 << 28
    d1 = cp.empty(n, cp.uint8)
    d2 = cp.empty_like(d1)
    host = cp.cuda.alloc_pinned_memory(n)
    h = np.frombuffer(host, np.uint8, n)

    def timed(f):
        f()
        cp.cuda.Device().synchronize()
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        f()
        e.record()
        e.synchronize()
        return cp.cuda.get_elapsed_time(s, e) / 1e3

    return {
        "gpu_copy_bytes_per_s": 2 * n / med(lambda: timed(lambda: d2.__setitem__(slice(None), d1))),
        "pcie_h2d_bytes_per_s": n / med(lambda: timed(lambda: d1.set(h))),
        "pcie_d2h_bytes_per_s": n / med(lambda: timed(lambda: d1.get(out=h))),
    }


def main():
    rows = {}
    for drive, folder in DRIVES.items():
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, ".machine-read.bin")
        make_file(path)
        try:
            rows[f"{drive}_seq_read_bytes_per_s"] = med(lambda: seq_read(path))
            rows[f"{drive}_4k_read_latency_s"] = med(lambda: rand_read_latency(path))
            rows[f"{drive}_4k_read_iops_32"] = med(lambda: rand_read_iops(path))
        finally:
            os.unlink(path)
        rows[f"{drive}_fsync_latency_s"] = med(lambda: fsync_latency(os.path.join(folder, ".machine-fsync.bin")))
        print(drive, {k: v for k, v in rows.items() if k.startswith(drive)}, flush=True)
    rows["dram_copy_bytes_per_s"] = dram_copy()
    rows.update(gpu())
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    host = socket.gethostname()
    with open(os.path.join(out, f"{host}.jsonl"), "w") as f:
        for k, v in rows.items():
            f.write(json.dumps({"host": host, "measure": k, "value": v, "time": time.time()}) + "\n")
    for k, v in rows.items():
        print(f"{k}: {v:.4g}")


if __name__ == "__main__":
    sys.exit(main())
