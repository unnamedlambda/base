"""The GPU codecs through nvCOMP, one input per invocation.

    python gpu.py <input>        # appends to <run dir>/gpu.jsonl

Each codec compresses the whole input in 64 KiB chunks, batched in one call
(nvCOMP's own benchmark protocol and its native bitstream), then
decompresses it; the round trip must equal the input byte for byte.

Two rates per direction, each the fastest of repeated passes over at least
TIMED_S seconds (lzbench's rule, so the CPU and GPU tables read alike):
    device      input and output already in device memory
    end to end  host memory to host memory: the copy in, the codec, the
                copy out
GPU joules per GB come from the board's energy counter over the
device-resident passes (`../meter.py`).
"""

import json
import os
import sys
import time

import cupy as cp
import numpy as np
from nvidia import nvcomp

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from meter import METER  # noqa: E402

CODECS = ("LZ4", "Snappy", "Zstd", "Deflate", "GDeflate")
CHUNK = 64 * 1024
TIMED_S = 2.0  # the energy counter advances every 100 ms: windows of seconds


def run_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", os.environ.get("COMPRESS_RUN", "adhoc"))
    os.makedirs(d, exist_ok=True)
    return d


def fastest(fn, seconds: float):
    """(fastest pass in seconds, passes, energy over all passes)."""
    fn()
    cp.cuda.Device().synchronize()
    best, n, start, t_end = float("inf"), 0, METER.read(), time.perf_counter() + seconds
    while n < 3 or time.perf_counter() < t_end:
        t = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        best = min(best, time.perf_counter() - t)
        n += 1
    return best, n, METER.since(start)


def j_per_gb(energy, passes: int, nbytes: int):
    if energy.get("gpu_j") is None:
        return None
    return energy["gpu_j"] / (passes * nbytes / 1e9)


def measure(algo: str, host: np.ndarray, seconds: float) -> dict:
    codec = nvcomp.Codec(algorithm=algo, chunk_size=CHUNK)
    dev = cp.asarray(host)
    src = nvcomp.as_array(dev)
    comp = codec.encode(src)
    out = cp.asarray(codec.decode(comp)).view(cp.uint8).reshape(-1)
    if out.size != dev.size or not bool((out == dev).all()):
        raise SystemExit(f"{algo}: the round trip differs from the input")
    comp_bytes = int(comp.buffer_size)
    comp_host = cp.asnumpy(cp.asarray(comp))

    c_dev, c_n, c_e = fastest(lambda: codec.encode(src), seconds)
    d_dev, d_n, d_e = fastest(lambda: codec.decode(comp), seconds)
    c_e2e, _, _ = fastest(lambda: cp.asnumpy(cp.asarray(codec.encode(nvcomp.as_array(cp.asarray(host))))), seconds)
    d_e2e, _, _ = fastest(lambda: cp.asnumpy(cp.asarray(codec.decode(nvcomp.as_array(cp.asarray(comp_host))))), seconds)
    n = host.size
    return {
        "codec": algo,
        "chunk_bytes": CHUNK,
        "orig_bytes": n,
        "comp_bytes": comp_bytes,
        "ratio": n / comp_bytes,
        "compress_GBps_device": n / c_dev / 1e9,
        "decompress_GBps_device": n / d_dev / 1e9,
        "compress_GBps_e2e": n / c_e2e / 1e9,
        "decompress_GBps_e2e": n / d_e2e / 1e9,
        "compress_J_per_GB": j_per_gb(c_e, c_n, n),
        "decompress_J_per_GB": j_per_gb(d_e, d_n, n),
    }


def main():
    path = sys.argv[1]
    host = np.fromfile(path, dtype=np.uint8)
    name = os.path.basename(path)
    seconds = 0.0 if name == "tiny" else TIMED_S
    with open(os.path.join(run_dir(), "gpu.jsonl"), "a") as f:
        for algo in CODECS:
            row = {"input": name, "nvcomp": nvcomp.__version__, **measure(algo, host, seconds)}
            f.write(json.dumps(row) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
