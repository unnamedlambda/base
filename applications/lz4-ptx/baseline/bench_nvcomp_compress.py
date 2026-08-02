#!/usr/bin/env python3
"""nvCOMP LZ4 GPU COMPRESSION throughput — the baseline `lz4-comp-warp` is quoted against.

Throughput = input-bytes / encode-time (the standard compression basis), over the
same corpus and the same chunk sizes as our kernel's block sizes.

Data is pre-staged on the GPU and only the device-side encode is timed, so this
EXCLUDES host transfers and allocation. That is deliberately generous to nvCOMP:
our kernel's number includes buffer allocation, the H2D upload and the D2H
download (amortized over the launches in one execution).

Usage: bench_nvcomp_compress.py <corpus.bin> [chunk_bytes ...]
"""
import ctypes, os, sys, time

import numpy as np
import nvidia.nvcomp as nvcomp

_rt = ctypes.CDLL("libcudart.so.12")


def sync():
    e = _rt.cudaDeviceSynchronize()
    if e:
        raise RuntimeError(f"cudaDeviceSynchronize -> {e}")


CORPUS = sys.argv[1] if len(sys.argv) > 1 else "corpus/silesia_all.bin"
CHUNKS = [int(a) for a in sys.argv[2:]] or [32768, 65536]
N_RUNS, N_WARMUP = 30, 5

# Exactly the prefix the shipped kernels bake in (Lz4CompAlgorithm.corpusBytes),
# so both sides compress the identical bytes.
PREFIX = int(os.environ.get("LZ4_PREFIX", 209_715_200))

data = np.fromfile(CORPUS, dtype=np.uint8)[:PREFIX]
uncomp = data.nbytes
assert uncomp == PREFIX, f"corpus shorter than the {PREFIX}-byte prefix"
print(f"corpus: {CORPUS}, {uncomp/1e6:.1f} MB prefix ({uncomp} bytes)")

def encode_time(codec, d_arr, runs):
    """Best of `runs` device-side encodes, seconds."""
    for _ in range(N_WARMUP):
        codec.encode(d_arr)
    sync()
    times = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        codec.encode(d_arr)
        sync()
        times.append(time.perf_counter() - t0)
    return np.array(times)


print(f"\n--- nvCOMP LZ4 COMPRESSION (input-bytes / time), best of {N_RUNS}, device-side encode only ---")
print("`encode()` allocates its output buffer per call, so the raw number carries a fixed")
print("per-call cost. Re-timing a 1/8 prefix separates it: slope = marginal encode rate,")
print("intercept = fixed cost. Both are reported; the marginal rate is the generous one.")
print()

for chunk in CHUNKS:
    codec = nvcomp.Codec(algorithm="LZ4", uncomp_chunk_size=chunk)
    d_arr = nvcomp.as_array(data).cuda()
    sync()
    comp = codec.encode(d_arr)
    sync()
    ratio = uncomp / comp.buffer_size

    full = encode_time(codec, d_arr, N_RUNS)

    # Second point on the same line: same codec, one eighth of the input.
    small_n = uncomp // 8
    d_small = nvcomp.as_array(data[:small_n]).cuda()
    sync()
    small = encode_time(codec, d_small, N_RUNS)

    slope = (full.min() - small.min()) / (uncomp - small_n)  # s/byte
    fixed = full.min() - slope * uncomp
    g = lambda t: uncomp / t / 1e9
    print(
        f"{chunk:>6} B chunks: ratio {ratio:.3f}:1, "
        f"best {full.min()*1e3:7.2f} ms -> {g(full.min()):5.2f} GB/s, "
        f"median {np.median(full)*1e3:7.2f} ms -> {g(np.median(full)):5.2f} GB/s"
    )
    print(
        f"{'':>6}             fixed per-call cost {fixed*1e3:5.2f} ms "
        f"({100*fixed/full.min():4.1f}% of the encode) -> marginal rate {1/slope/1e9:5.2f} GB/s"
    )
