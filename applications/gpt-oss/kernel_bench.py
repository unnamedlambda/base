"""What the MXFP4 expert kernels cost, against what they could cost.

A decode step reads four experts in each of twenty-four layers and nothing else
of comparable size, so these two kernels are most of a token.  The roofline is
not a guess: an expert is 13.25 MiB of packed weight, every byte of it is read
exactly once, and the card's measured bandwidth says how long that takes.  The
gap between that and the clock is the whole subject.

This runs the PTX the generator emitted, directly, with one expert's worth of
buffers -- no artifact, no bank, no 12.9 GiB.  That is the point: the edit-test
loop for a kernel should be seconds.

  python applications/gpt-oss/kernel_bench.py \\
      lean/.lake/build/artifacts/gptoss_decode.cbor

`--csv` prints one line per kernel for recording a before and an after.
"""

import argparse
import ctypes
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout import S_GATEUP, S_DOWN, ptx_modules  # noqa: E402

H = I = 2880
G = 32
WARPS = 8
# `GptOssKernels.rowsPerWarpDown`: the down projection gives each warp several
# output rows so the staged activation is read once for all of them.
ROWS_DOWN = 4
# `GptOssKernels.rowsPerWarpGateUp`; each is a gate row and an up row.
ROWS_GU = 2
NL, TOPK = 24, 4

ROW_BYTES = H // 2
ROW_SCALES = H // G
# gate and up are one table of 2*I rows; down is I rows of H/2.  Plus scales
# and biases, which is the 13.25 MiB the plan's traffic model counts.
EXPERT_BYTES = (2 * I * ROW_BYTES + 2 * I * ROW_SCALES + 2 * I * 4
                + H * ROW_BYTES + H * ROW_SCALES + H * 4)

# Measured on this box, and the number the plan's model is built on.
VRAM_GBPS = 333.0


class Cuda:
    """The smallest driver binding that will run a PTX module and time it."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._check(self.lib.cuInit(0))
        dev = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(dev), 0))
        self.ctx = ctypes.c_void_p()
        self._check(self.lib.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev))

    def module(self, src):
        mod = ctypes.c_void_p()
        self._check(self.lib.cuModuleLoadData(ctypes.byref(mod), src.encode() + b"\0"))
        return mod

    def _check(self, rc):
        if rc != 0:
            name = ctypes.c_char_p()
            self.lib.cuGetErrorString(rc, ctypes.byref(name))
            raise RuntimeError(f"CUDA {rc}: {name.value.decode() if name.value else '?'}")

    def func(self, mod, name="main"):
        f = ctypes.c_void_p()
        self._check(self.lib.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()))
        return f

    def upload(self, arr):
        d = ctypes.c_void_p()
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(d), ctypes.c_size_t(arr.nbytes)))
        self._check(self.lib.cuMemcpyHtoD_v2(d, arr.ctypes.data_as(ctypes.c_void_p),
                                             ctypes.c_size_t(arr.nbytes)))
        return d

    def _args(self, args):
        return (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])

    def launch(self, fn, grid, block, args):
        self._check(self.lib.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, None,
                                            self._args(args), None))

    def sync(self):
        self._check(self.lib.cuCtxSynchronize())

    def time(self, fn, grid, block, args, iters=200, warmup=20):
        """Wall clock over many launches, which is what a token actually pays.

        Not a graph and not an event pair: the decode loop launches these one at
        a time from the host, so launch overhead belongs inside the number.
        """
        a = self._args(args)
        for _ in range(warmup):
            self._check(self.lib.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, None, a, None))
        self.sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            self._check(self.lib.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, None, a, None))
        self.sync()
        return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    cu = Cuda()
    import py_base
    mods = ptx_modules(py_base.Driver(py_base.load_artifact(args.artifact)))

    # One expert, filled with plausible bytes.  Values do not matter to a timing
    # -- there is no data-dependent branch in either kernel -- but the buffers
    # have to be the true size or the cache hierarchy answers a different
    # question than the one being asked.
    gu_blocks = cu.upload(rng.integers(0, 256, 2 * I * ROW_BYTES, dtype=np.uint8))
    gu_scales = cu.upload(rng.integers(120, 134, 2 * I * ROW_SCALES, dtype=np.uint8))
    gu_bias = cu.upload(rng.standard_normal(2 * I).astype(np.float32))
    dn_blocks = cu.upload(rng.integers(0, 256, H * ROW_BYTES, dtype=np.uint8))
    dn_scales = cu.upload(rng.integers(120, 134, H * ROW_SCALES, dtype=np.uint8))
    dn_bias = cu.upload(rng.standard_normal(H).astype(np.float32))
    x = cu.upload(rng.standard_normal(H).astype(np.float32))
    hid = cu.upload(rng.standard_normal(I).astype(np.float32))
    out_i = cu.upload(np.zeros(I, np.float32))
    out_h = cu.upload(np.zeros(H, np.float32))

    gu = cu.func(cu.module(mods[S_GATEUP]))
    dn = cu.func(cu.module(mods[S_DOWN]))
    block = WARPS * 32
    gu_grid = I // (WARPS * ROWS_GU)
    dn_grid = H // (WARPS * ROWS_DOWN)

    t_gu = cu.time(gu, gu_grid, block, [gu_blocks, gu_scales, gu_bias, x, out_i], args.iters)
    t_dn = cu.time(dn, dn_grid, block, [dn_blocks, dn_scales, dn_bias, hid, out_h], args.iters)

    # Each kernel reads its own weights once; that is its roofline.
    gu_bytes = 2 * I * ROW_BYTES + 2 * I * ROW_SCALES + 2 * I * 4
    dn_bytes = H * ROW_BYTES + H * ROW_SCALES + H * 4
    per_expert = t_gu + t_dn
    per_token = per_expert * NL * TOPK
    roof_expert = EXPERT_BYTES / (VRAM_GBPS * 1e9)

    if args.csv:
        print(f"gate_up,{t_gu*1e6:.1f},{gu_bytes/t_gu/1e9:.1f}")
        print(f"down,{t_dn*1e6:.1f},{dn_bytes/t_dn/1e9:.1f}")
        print(f"expert,{per_expert*1e6:.1f},{EXPERT_BYTES/per_expert/1e9:.1f}")
        return 0

    print(f"  one expert is {EXPERT_BYTES/2**20:.2f} MiB; the card reads "
          f"{VRAM_GBPS:.0f} GB/s, so the floor is {roof_expert*1e6:.1f} us/expert\n")
    for name, t, nbytes, grid in (("gate_up_swiglu", t_gu, gu_bytes, gu_grid),
                                  ("down_bias", t_dn, dn_bytes, dn_grid)):
        bw = nbytes / t / 1e9
        print(f"  {name:16s} {t*1e6:8.1f} us   {bw:6.1f} GB/s   "
              f"{bw/VRAM_GBPS*100:4.1f}% of bandwidth   {grid} CTAs")
    print()
    print(f"  per expert       {per_expert*1e6:8.1f} us   "
          f"floor {roof_expert*1e6:.1f} us   {per_expert/roof_expert:.2f}x off")
    print(f"  per token        {per_token*1e3:8.2f} ms   "
          f"({NL} layers x {TOPK} experts)   "
          f"=> {1.0/per_token:.0f} tok/s if nothing else cost anything")
    return 0


if __name__ == "__main__":
    sys.exit(main())
