"""base: the engine under test, as base runs it.

The engine is an artifact `lean/algorithms/Ann/Host.lean` writes, one per
vector shape (`ann_<d>_<l2|ip>.cbor`), loaded through py-base
(`Driver(artifact).function(name)`). Its host program and kernels are base's
own: see that file for what each function does. Rows reach it in chunks of
CHUNK, so a large insert never needs a second copy of the whole dataset in
host memory.

Settings are (nprobe, R): lists probed, and candidates each list keeps for
the exact re-rank; the grid steps nprobe by about sqrt 2, R through 16 and 32,
and its deep end takes R 64 and 192 at the most lists.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
ARTIFACTS = os.path.join(ROOT, "lean", ".lake", "build", "artifacts")
PY_BASE = os.path.join(ROOT, "py-base", ".venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")

PROBES = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
CHUNK = 500_000  # rows a call carries


def driver_for(metric: str, dim: int):
    if PY_BASE not in sys.path:
        sys.path.append(PY_BASE)
    from py_base import Driver

    path = os.path.join(ARTIFACTS, f"ann_{dim}_{'l2' if metric == 'sqeuclidean' else 'ip'}.cbor")
    if not os.path.exists(path):
        raise SystemExit(f"no base engine for {dim}-dimensional {metric} vectors: {path}")
    with open(path, "rb") as f:
        return Driver(f.read())


class Base:
    def __init__(self, metric: str, dim: int, capacity: int):
        self.metric, self.dim = metric, dim
        self.drv = driver_for(metric, dim)
        self.f = {n: self.drv.function(n) for n in ("init", "insert", "delete", "commit", "search", "tags", "searchf")}
        self.call("init", np.array([capacity], np.uint64).tobytes())

    def call(self, name, data, out=None):
        status = self.f[name](data, out)
        if status != 0:
            raise RuntimeError(f"{name} returned {status}")

    def settings_for(self, k: int):
        """(nprobe, R) cheapest first, R at least k; the deep end keeps more
        candidates a list, where the last recall is."""
        return [(p, r) for p in PROBES for r in (16, 32) if k <= r] + [(PROBES[-1], 64), (PROBES[-1], 192)]

    def insert(self, vecs, ids):
        ids = np.asarray(ids, np.int64)
        for s in range(0, len(ids), CHUNK):
            v = np.ascontiguousarray(vecs[s : s + CHUNK], np.float32)
            hdr = np.array([len(v), 0], np.uint32).tobytes()
            self.call("insert", hdr + v.tobytes() + ids[s : s + CHUNK].tobytes())

    def delete(self, ids):
        ids = np.ascontiguousarray(ids, np.int64)
        self.call("delete", np.array([len(ids), 0], np.uint32).tobytes() + ids.tobytes())

    def wait_indexed(self):
        self.call("commit", b"")

    def set_tags(self, indptr, tags):
        """The items' tags (item i's at tags[indptr[i]:indptr[i+1]]); the engine
        builds its inverted index from them."""
        indptr = np.ascontiguousarray(indptr, np.int32)
        tags = np.ascontiguousarray(tags, np.int32)
        hdr = np.array([len(indptr) - 1, len(tags)], np.uint32).tobytes()
        self.call("tags", hdr + indptr.tobytes() + tags.tobytes())

    def search_filtered(self, q, s, k: int, required):
        """The k nearest items holding every tag of required[j], per query j."""
        nprobe, R = s
        q = np.ascontiguousarray(q, np.float32)
        rptr = np.zeros(len(required) + 1, np.uint32)
        rptr[1:] = np.cumsum([len(r) for r in required])
        rtags = np.concatenate([np.asarray(r, np.uint32) for r in required]) if len(required) else np.zeros(0, np.uint32)
        out = np.empty((len(q), k), np.int64)
        hdr = np.array([len(q), nprobe, R, k], np.uint32).tobytes()
        self.call("searchf", hdr + q.tobytes() + rptr.tobytes() + rtags.tobytes(), out.reshape(-1).view(np.uint8))
        return out

    def search(self, q, s, k: int):
        nprobe, R = s
        q = np.ascontiguousarray(q, np.float32)
        out = np.empty((len(q), k), np.int64)
        hdr = np.array([len(q), nprobe, R, k], np.uint32).tobytes()
        self.call("search", hdr + q.tobytes(), out.reshape(-1).view(np.uint8))
        return out
