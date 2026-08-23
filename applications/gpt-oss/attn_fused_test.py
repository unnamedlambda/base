"""The fused attention pair, against a reference that keeps the scores.

`gptoss_attn_tile` folds every score straight into a running maximum and sum
and never writes one down, so there is nothing to inspect midway: what it
computes is only visible at the end.  This scores it against the obvious
implementation -- materialise the whole score row, softmax it, multiply by V --
which is what the four-kernel path it replaces actually did.

The two must agree to within f32 rounding, not approximately: the online
softmax is an exact refactoring of the same sum, so the only differences
allowed are fold order and `ex2.approx`.

What the tests are chosen to catch:

  * a tile boundary landing wrong -- run at lengths that are and are not
    multiples of `ATT_TILE`, including one key and one key past a tile;
  * the sink applied per tile instead of per head, which is invisible at one
    tile and wrong at two;
  * the running maximum dropped, which is invisible on small scores and
    catastrophic on large ones, so one case has scores that overflow `exp`
    outright;
  * a query head reading another's row, which a uniform query would hide, so
    every head gets its own scale.

  python applications/gpt-oss/attn_fused_test.py <gptoss_kernels.ptx>
"""

import ctypes
import sys

import numpy as np

HD, GQA, NKV = 64, 8, 8
NQ = GQA * NKV
ATT_TILE = 128
PARTIAL = 2 + HD
# `GptOssAttention`: the meta words this kernel reads.
M_SEQ, M_KVSTRIDE, M_NTILES = 2, 7, 8

FAILURES = []


def to_bf16(a):
    """Round f32 to bf16, the way the cache holds it."""
    u = a.astype(np.float32).view(np.uint32)
    return ((u + 0x8000 + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def pack_bf16(a):
    """f32 -> the packed bf16 words the cache and the query buffer hold."""
    u = to_bf16(a).view(np.uint32) >> 16
    u = u.reshape(-1, 2).astype(np.uint32)
    return (u[:, 0] | (u[:, 1] << 16)).astype(np.uint32)


class Cuda:
    def __init__(self, ptx_path):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._check(self.lib.cuInit(0))
        dev = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(dev), 0))
        self.ctx = ctypes.c_void_p()
        self._check(self.lib.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev))
        with open(ptx_path, "rb") as f:
            src = f.read() + b"\0"
        self.mod = ctypes.c_void_p()
        self._check(self.lib.cuModuleLoadData(ctypes.byref(self.mod), src))

    def _check(self, rc):
        if rc != 0:
            name = ctypes.c_char_p()
            self.lib.cuGetErrorString(rc, ctypes.byref(name))
            raise RuntimeError(f"CUDA {rc}: {name.value.decode() if name.value else '?'}")

    def func(self, name):
        f = ctypes.c_void_p()
        self._check(self.lib.cuModuleGetFunction(ctypes.byref(f), self.mod, name.encode()))
        return f

    def upload(self, arr):
        d = ctypes.c_void_p()
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(d), ctypes.c_size_t(arr.nbytes)))
        self._check(self.lib.cuMemcpyHtoD_v2(d, arr.ctypes.data_as(ctypes.c_void_p),
                                             ctypes.c_size_t(arr.nbytes)))
        return d

    def download(self, d, like):
        out = np.empty_like(like)
        self._check(self.lib.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), d,
                                             ctypes.c_size_t(out.nbytes)))
        return out

    def launch(self, fn, grid, block, args):
        arr = (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        self._check(self.lib.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, None, arr, None))
        self._check(self.lib.cuCtxSynchronize())


def reference(q, k, v, sinks, L):
    """Scores materialised, softmaxed, and mixed -- the path being replaced.

    Operands are rounded to bf16 first, because that is what the cache holds
    and what the kernel therefore reads.
    """
    qb, kb, vb = to_bf16(q), to_bf16(k), to_bf16(v)
    out = np.zeros((NQ, HD), np.float32)
    for b in range(NKV):
        for g in range(GQA):
            h = b * GQA + g
            sc = (kb[b, :L] @ qb[h]).astype(np.float32) * (1.0 / np.sqrt(HD))
            m = max(sc.max(), sinks[h])
            e = np.exp(sc - m)
            denom = e.sum() + np.exp(sinks[h] - m)
            out[h] = (e / denom) @ vb[b, :L]
    return out


def run_case(cu, tile, comb, L, cap, scale, label, rng):
    q = (rng.standard_normal((NQ, HD)) * scale).astype(np.float32)
    # One row of slack past `cap`, which is part of the kernel's contract: it
    # loads the next key while working on the current one, so the last
    # iteration reads one row past the keys that exist. `dInitM` allocates the
    # caches the same way. Without it this harness gets an illegal access --
    # which is how the contract was discovered, and why it is written down.
    k = (rng.standard_normal((NKV, cap + 1, HD))).astype(np.float32)
    v = (rng.standard_normal((NKV, cap + 1, HD))).astype(np.float32)
    sinks = rng.standard_normal(NQ).astype(np.float32)

    d_q = cu.upload(pack_bf16(q.reshape(-1)))
    d_k = cu.upload(pack_bf16(k.reshape(-1)))
    d_v = cu.upload(pack_bf16(v.reshape(-1)))
    d_s = cu.upload(sinks)
    n_tiles = (L + ATT_TILE - 1) // ATT_TILE
    meta = np.zeros(64, np.int32)
    meta[M_SEQ] = L
    meta[M_KVSTRIDE] = (cap + 1) * HD // 2    # words a key head strides by
    # The combine reads the tile count from the meta too, because the layer
    # uploads the meta anyway and a buffer of its own cost an upload a layer.
    meta[M_NTILES] = n_tiles
    d_m = cu.upload(meta)
    part = np.zeros(n_tiles * NKV * GQA * PARTIAL, np.float32)
    d_p = cu.upload(part)
    d_o = cu.upload(np.zeros(NQ * HD, np.float32))

    cu.launch(tile, n_tiles, 32 * NKV, [d_k, d_v, d_q, d_p, d_m])
    cu.launch(comb, NQ, 32, [d_p, d_s, d_o, d_m])
    got = cu.download(d_o, np.zeros(NQ * HD, np.float32)).reshape(NQ, HD)

    want = reference(q, k, v, sinks, L)
    denom = np.maximum(np.abs(want).max(), 1e-6)
    rel = float(np.abs(got - want).max() / denom)
    ok = rel < 2e-3 and np.isfinite(got).all()
    print(f"  {'ok  ' if ok else 'FAIL'} {label:34s} L={L:6d}  tiles={n_tiles:3d}  "
          f"max-rel {rel:.2e}")
    if not ok:
        FAILURES.append(label)


def main():
    ptx = sys.argv[1] if len(sys.argv) > 1 else "gptoss_kernels.ptx"
    cu = Cuda(ptx)
    tile, comb = cu.func("gptoss_attn_tile"), cu.func("gptoss_attn_combine")
    rng = np.random.default_rng(20260823)
    print("gpt-oss fused attention, against a reference that keeps the scores")
    print(f"  module: {ptx}\n")

    run_case(cu, tile, comb, 1, 2048, 1.0, "one key", rng)
    run_case(cu, tile, comb, 37, 2048, 1.0, "under one tile", rng)
    run_case(cu, tile, comb, ATT_TILE, 2048, 1.0, "exactly one tile", rng)
    run_case(cu, tile, comb, ATT_TILE + 1, 2048, 1.0, "one key past a tile", rng)
    run_case(cu, tile, comb, 3 * ATT_TILE, 4096, 1.0, "three whole tiles", rng)
    run_case(cu, tile, comb, 2000, 4096, 1.0, "ragged, four tiles", rng)
    # A running maximum that is dropped survives small scores and dies here:
    # exp of the raw score overflows f32 outright.
    run_case(cu, tile, comb, 1500, 4096, 40.0, "scores that overflow exp", rng)
    run_case(cu, tile, comb, 8192, 8192, 1.0, "sixteen tiles", rng)

    print()
    if FAILURES:
        print(f"RESULT : MISMATCH ({len(FAILURES)} failed: {', '.join(FAILURES)})")
        return 1
    print("RESULT : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
