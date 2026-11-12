"""Pillow on the harness's frames, by the harness's method.

    pillow_bench.py DIR one|cores

DIR holds what `image-bench ... dump` wrote: the eight source frames and each
table's reference output for frame 0. Each plane goes through Image.resize,
which Pillow and Pillow-SIMD run as a separable convolution with an
antialiased filter; Pillow releases the GIL while it resizes, so threads run
at once. Run it under the interpreter of the Pillow being measured.
"""
import json
import os
import sys
import threading
import time

import PIL
from PIL import Image

SRC, DST, FRAMES, ROUNDS = (3840, 2160), (1920, 1080), 8, 9


def planes(w, h):
    return [(w, h, 0), (w // 2, h // 2, w * h), (w // 2, h // 2, w * h + (w // 2) * (h // 2))]


def frame_bytes(w, h):
    return w * h * 3 // 2


FILTERS = [("bilinear", Image.BILINEAR), ("box", Image.BOX), ("lanczos", Image.LANCZOS)]
# The Rust harness's name for each table, which the result lines share.
TITLE = {"bilinear": "Bilinear", "box": "2x2 average", "lanczos": "Lanczos3"}


def record(table, mode, engine, threads, rate):
    """One result line, appended to the file IMAGE_RESULTS names where it is
    set, in the Rust harness's shape."""
    path = os.environ.get("IMAGE_RESULTS")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"table": TITLE[table], "mode": mode, "engine": engine, "threads": threads,
                                "fps": rate}) + "\n")


def scale(src, k, f):
    """Frame k's three planes, resized; the images Pillow hands back."""
    base = k * frame_bytes(*SRC)
    out = []
    for (w, h, o) in planes(*SRC):
        im = Image.frombuffer("L", (w, h), src[base + o:base + o + w * h], "raw", "L", 0, 1)
        out.append(im.resize((w // 2, h // 2), f))
    return out


def distance(out, ref):
    res = []
    for im, (w, h, o) in zip(out, planes(*DST)):
        a, b = im.tobytes(), ref[o:o + w * h]
        d = [abs(x - y) for x, y in zip(a, b)]
        res.append((max(d), sum(d) / len(d)))
    return res


def fps(src, f, threads):
    """Frames per second of `threads` threads at the best of ROUNDS rounds, each
    thread scaling 16 frames from its own starting frame."""
    per = 2 * FRAMES

    def work(t, start):
        scale(src, t % FRAMES, f)
        start.wait()
        for i in range(per):
            scale(src, (t + i) % FRAMES, f)

    best = float("inf")
    for _ in range(ROUNDS):
        start = threading.Barrier(threads + 1)
        ts = [threading.Thread(target=work, args=(t, start)) for t in range(threads)]
        for t in ts:
            t.start()
        start.wait()
        t0 = time.perf_counter()
        for t in ts:
            t.join()
        best = min(best, time.perf_counter() - t0)
    return threads * per / best


def main():
    d, mode = sys.argv[1], sys.argv[2]
    src = memoryview(open(f"{d}/src.raw", "rb").read())
    name = f"Pillow {PIL.__version__}"
    for table, f in FILTERS:
        ref = open(f"{d}/ref_{table}.raw", "rb").read()
        if mode == "one":
            dist = distance(scale(src, 0, f), ref)
            cells = "   ".join(f"{p} {m:3} {a:6.3f}" for p, (m, a) in zip("YUV", dist))
            rate = fps(src, f, 1)
            print(f"  {table:9} {name:24} same pixels? {cells}   one core {rate:8.1f} frames/s")
            record(table, "one core", name, 1, rate)
        else:
            rates = [fps(src, f, n) for n in (1, 2, 4, 6)]
            print(f"  {table:9} {name:24} threads 1 2 4 6: " + " ".join(f"{r:8.1f}" for r in rates))
            for n, r in zip((1, 2, 4, 6), rates):
                record(table, "threads", name, n, r)


main()
