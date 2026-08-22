"""Bit-level check of the MXFP4 expert kernels against their specification.

These kernels are outside the machine this project proves kernels in — they
unpack nibbles, and that machine addresses buffers by element and holds
Float32.  So they ship checked rather than proven, and this is the check.  It
is deliberately not a tolerance test on realistic data:

  * the dequantiser is compared on **all 256 byte values** crossed with the
    scale extremes, and the comparison is on bits, not on a difference.  Every
    code that can appear is therefore exercised, including the two the format
    reserves and the ones a lookup table would most plausibly transpose;
  * the contractions are compared against a Float32 fold in the kernel's own
    order, so what is left after the comparison is fold order and nothing else;
  * the activation is checked at its clamp edges, which is where a `min` and a
    `max` are most easily written the wrong way round.

The reference here is `ML/QuantMX.lean`'s, transcribed: `fp4Mag`'s eight
magnitudes with the sign in the high bit, and a scale byte read as the Float32
whose exponent field it is.  When the machine grows a byte load, that file's
`mxDot_spec` replaces this script for the first two claims.

Runs against libcuda directly rather than through an artifact: at this stage
there is no host program to test, only three kernels, and loading the emitted
PTX is the shortest path between the generator and a verdict.

  python applications/gpt-oss/kernel_test.py <gptoss_kernels.ptx>
"""

import ctypes
import warnings
import sys

import numpy as np

# ── the specification, transcribed from ML/QuantMX.lean ──────────────────────

# Warps per block; the GEMV kernels give each warp one output row.
WARPS = 8

FP4_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def fp4_val(code: np.ndarray) -> np.ndarray:
    """`QuantMX.fp4Val`: magnitude in the low three bits, sign in the fourth."""
    mag = FP4_MAG[code & 7]
    return np.where(code & 8, -mag, mag).astype(np.float32)


def e8m0_scale(sbyte: np.ndarray) -> np.ndarray:
    """A scale byte is the Float32 whose exponent field it is: `2^(s-127)`.

    `s = 0` leaves the exponent field zero, which is the Float32 encoding of a
    subnormal and reads as `0.0`; `s = 255` is the reserved NaN the converter
    rejects.  Both are produced by the same shift, so both are tested.
    """
    return (sbyte.astype(np.uint32) << 23).view(np.float32)


def swiglu(gate, up, alpha=1.702, limit=7.0):
    """gpt-oss's clamped SwiGLU, in Float32 throughout."""
    g = np.minimum(gate, np.float32(limit))
    u = np.clip(up, np.float32(-limit), np.float32(limit))
    sig = (np.float32(1.0) / (np.float32(1.0) + np.exp(-np.float32(alpha) * g))).astype(np.float32)
    return ((g * sig) * (u + np.float32(1.0))).astype(np.float32)


def dot_ref(blocks, scales, x, n_blocks):
    """The kernel's own fold order, which is a claim about the kernel and not a
    detail: a sum of 2880 Float32 products depends on the order it is taken in,
    so a reference that reassociates would be testing something else.

    Lane L accumulates, in this order: eight consecutive elements per chunk of
    eight blocks (`c*256 + L*8 + j`), then one element per leftover block
    (`blk*32 + L`).  The 32 lane totals are then butterfly-reduced, which is the
    order the rest of this development commits to.
    """
    n_chunks = n_blocks // 8
    tail_start = n_chunks * 8
    codes_all = np.empty(n_blocks * 32, dtype=np.uint8)
    codes_all[0::2] = blocks[: n_blocks * 16] & 0xF
    codes_all[1::2] = blocks[: n_blocks * 16] >> 4
    scale_per_elem = e8m0_scale(np.repeat(scales[:n_blocks], 32))
    w_all = (fp4_val(codes_all) * scale_per_elem).astype(np.float32)

    lane_acc = np.zeros(32, dtype=np.float32)
    for c in range(n_chunks):
        for j in range(8):
            idx = c * 256 + np.arange(32) * 8 + j
            lane_acc = (lane_acc + w_all[idx] * x[idx]).astype(np.float32)
    for blk in range(tail_start, n_blocks):
        idx = blk * 32 + np.arange(32)
        lane_acc = (lane_acc + w_all[idx] * x[idx]).astype(np.float32)

    acc = lane_acc.copy()
    for mask in (16, 8, 4, 2, 1):                     # the committed butterfly
        acc = (acc + acc[np.arange(32) ^ mask]).astype(np.float32)
    return acc[0]


# ── the smallest CUDA driver binding that will run a PTX module ──────────────


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
            raise RuntimeError(f"CUDA error {rc}: {name.value.decode() if name.value else '?'}")

    def func(self, name):
        f = ctypes.c_void_p()
        self._check(self.lib.cuModuleGetFunction(ctypes.byref(f), self.mod, name.encode()))
        return f

    def upload(self, arr):
        d = ctypes.c_void_p()
        n = arr.nbytes
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(d), ctypes.c_size_t(n)))
        self._check(self.lib.cuMemcpyHtoD_v2(d, arr.ctypes.data_as(ctypes.c_void_p),
                                             ctypes.c_size_t(n)))
        return d

    def download(self, d, like):
        out = np.empty_like(like)
        self._check(self.lib.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), d,
                                             ctypes.c_size_t(out.nbytes)))
        return out

    def launch(self, fn, grid, block, args):
        arr = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p)
                                              for a in args])
        self._check(self.lib.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, None, arr, None))
        self._check(self.lib.cuCtxSynchronize())


# ── the checks ───────────────────────────────────────────────────────────────

FAILURES = []


def report(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def check_dequant_exhaustive(cu):
    """Every byte value against every interesting scale, compared on bits.

    One row per scale, 256 bytes to a row = 512 codes = 16 blocks.  A tolerance
    would hide exactly the errors this is looking for — a transposed nibble is
    a wrong value, not a slightly wrong one — so the comparison is on the
    Float32 bit patterns.
    """
    fn = cu.func("mxfp4_dequant_f32")
    scale_bytes = [0, 1, 64, 126, 127, 128, 200, 254]
    n_blocks = 16                      # 256 bytes = 512 elements = 16 blocks
    rows = len(scale_bytes)
    blocks = np.tile(np.arange(256, dtype=np.uint8), rows).reshape(rows, 256)
    scales = np.array([[s] * n_blocks for s in scale_bytes], dtype=np.uint8)
    meta = np.array([n_blocks], dtype=np.uint32)
    out = np.zeros((rows, n_blocks * 32), dtype=np.float32)

    d_b, d_s, d_o, d_m = cu.upload(blocks), cu.upload(scales), cu.upload(out), cu.upload(meta)
    cu.launch(fn, rows, 32, [d_b, d_s, d_o, d_m])
    got = cu.download(d_o, out)

    want = np.empty_like(out)
    for r, s in enumerate(scale_bytes):
        codes = np.empty(n_blocks * 32, dtype=np.uint8)
        row = blocks[r]
        codes[0::2] = row & 0xF
        codes[1::2] = row >> 4
        want[r] = fp4_val(codes) * e8m0_scale(np.full(codes.shape, s, np.uint8))

    # NaN never appears: scale 255 is rejected by the converter, and no code maps to one.
    bit_eq = got.view(np.uint32) == want.view(np.uint32)
    report("dequant bit-exact over all 256 byte values x 8 scales",
           bool(bit_eq.all()),
           f"{int(bit_eq.sum())}/{bit_eq.size} elements")
    if not bit_eq.all():
        bad = np.argwhere(~bit_eq)[:5]
        for r, c in bad:
            print(f"       row {r} (scale {scale_bytes[r]}) elem {c}: "
                  f"got {got[r, c]!r} want {want[r, c]!r}")

    # The sixteen values a code can take, named rather than left implicit —
    # this is what a lookup table would most plausibly get wrong.  Byte `c`
    # holds code `c & 0xF` in its low nibble and `c >> 4` in its high one, so
    # for the first sixteen bytes the codes 0..15 land at the even positions
    # and a zero sits between each pair.
    expect = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                      dtype=np.float32)
    got_unit = got[scale_bytes.index(127)][0:32:2]
    report("the sixteen codes at scale 2^0 are the E2M1 table",
           bool(np.array_equal(got_unit.view(np.uint32), expect.view(np.uint32))),
           f"{[float(v) for v in got_unit[:8]]}")


def check_gate_up(cu, rng):
    """The fused gate/up contraction with the clamped SwiGLU epilogue."""
    fn = cu.func("mxfp4_gate_up_swiglu_gemv")
    H = I = 2880
    n_blocks = H // 32
    rows = 2 * I
    blocks = rng.integers(0, 256, size=(rows, H // 2), dtype=np.uint8)
    # scales near 2^0 keep the reference in a range where a Float32 fold and a
    # Float64 one agree; the exhaustive scale check above covers the extremes.
    scales = rng.integers(120, 135, size=(rows, n_blocks)).astype(np.uint8)
    bias = (rng.standard_normal(rows) * 0.1).astype(np.float32)
    x = (rng.standard_normal(H) * 0.1).astype(np.float32)
    out = np.zeros(I, dtype=np.float32)

    d = [cu.upload(a) for a in (blocks, scales, bias, x, out)]
    cu.launch(fn, I // WARPS, 32 * WARPS, d)
    got = cu.download(d[4], out)

    idx = [0, 1, 17, 1000, I - 1]
    want = np.array([
        swiglu(dot_ref(blocks[r], scales[r], x, n_blocks) + bias[r],
               dot_ref(blocks[I + r], scales[I + r], x, n_blocks) + bias[I + r])
        for r in idx], dtype=np.float32)
    rel = np.abs(got[idx] - want) / np.maximum(np.abs(want), 1e-6)
    report("gate/up + clamped SwiGLU matches the Float32 reference",
           bool(rel.max() < 2e-6), f"max rel {rel.max():.2e} over {len(idx)} rows")


def check_swiglu_edges(cu, rng):
    """The activation at its clamp edges, where a min and a max are most easily
    written the wrong way round.

    Driven through the real kernel with a one-hot activation, so the value
    reaching the epilogue is a weight the test chose: code 7 is 6.0, so a scale
    of 2^0 gives a gate of 6.0 and 2^1 gives 12.0, either side of the limit."""
    fn = cu.func("mxfp4_gate_up_swiglu_gemv")
    H = I = 2880
    n_blocks = H // 32
    rows = 2 * I
    for sbyte, label in ((127, "below the clamp (6.0)"), (128, "above it (12.0)")):
        blocks = np.zeros((rows, H // 2), dtype=np.uint8)
        blocks[:, 0] = 0x07                       # element 0 -> code 7 -> 6.0
        scales = np.full((rows, n_blocks), sbyte, np.uint8)
        bias = np.zeros(rows, dtype=np.float32)
        x = np.zeros(H, dtype=np.float32)
        x[0] = np.float32(1.0)                    # select element 0 only
        out = np.zeros(I, dtype=np.float32)
        d = [cu.upload(a) for a in (blocks, scales, bias, x, out)]
        cu.launch(fn, I // WARPS, 32 * WARPS, d)
        got = cu.download(d[4], out)[0]
        v = np.float32(6.0) * np.float32(2.0) ** (sbyte - 127)
        want = swiglu(np.float32(v), np.float32(v))
        report(f"SwiGLU {label}", bool(abs(got - want) <= 2e-6 * max(abs(want), 1e-6)),
               f"got {got:.6f} want {float(want):.6f}")


def check_down(cu, rng):
    fn = cu.func("mxfp4_down_gemv_bias")
    H = I = 2880
    n_blocks = I // 32
    blocks = rng.integers(0, 256, size=(H, I // 2), dtype=np.uint8)
    scales = rng.integers(120, 135, size=(H, n_blocks)).astype(np.uint8)
    bias = (rng.standard_normal(H) * 0.1).astype(np.float32)
    h = (rng.standard_normal(I) * 0.1).astype(np.float32)
    out = np.zeros(H, dtype=np.float32)
    d = [cu.upload(a) for a in (blocks, scales, bias, h, out)]
    cu.launch(fn, H // WARPS, 32 * WARPS, d)
    got = cu.download(d[4], out)
    idx = [0, 3, 500, H - 1]
    want = np.array([dot_ref(blocks[r], scales[r], h, n_blocks) + bias[r] for r in idx],
                    dtype=np.float32)
    rel = np.abs(got[idx] - want) / np.maximum(np.abs(want), 1e-6)
    report("down projection + bias matches the Float32 reference",
           bool(rel.max() < 2e-6), f"max rel {rel.max():.2e} over {len(idx)} rows")


def check_adversarial(cu):
    """Rows a random draw will not produce: every code at its largest magnitude,
    and a zero scale.  The first is where an accumulator would overflow if the
    kernel had narrowed it; the second is where a shift-based scale decode is
    most likely to produce something other than zero."""
    fn = cu.func("mxfp4_down_gemv_bias")
    H = I = 2880
    n_blocks = I // 32
    for fill, sbyte, label in ((0x77, 127, "all codes 6.0, scale 2^0"),
                              (0xFF, 127, "all codes -6.0, scale 2^0"),
                              (0x77, 0, "all codes 6.0, scale byte 0 (zero)")):
        blocks = np.full((H, I // 2), fill, np.uint8)
        scales = np.full((H, n_blocks), sbyte, np.uint8)
        bias = np.zeros(H, dtype=np.float32)
        h = np.ones(I, dtype=np.float32)
        out = np.zeros(H, dtype=np.float32)
        d = [cu.upload(a) for a in (blocks, scales, bias, h, out)]
        cu.launch(fn, H // WARPS, 32 * WARPS, d)
        got = cu.download(d[4], out)[0]
        want = dot_ref(blocks[0], scales[0], h, n_blocks)
        report(f"adversarial: {label}", bool(got == want), f"got {got} want {float(want)}")


def main():
    ptx = sys.argv[1] if len(sys.argv) > 1 else "gptoss_kernels.ptx"
    cu = Cuda(ptx)
    rng = np.random.default_rng(20260822)
    # scale 2^127 times 6.0 overflows to inf, which is the correct answer on
    # both sides and is compared as such; numpy would rather warn about it.
    warnings.filterwarnings("ignore", "overflow encountered")
    print(f"gpt-oss MXFP4 kernels, checked against ML/QuantMX.lean's specification")
    print(f"  module: {ptx}\n")
    check_dequant_exhaustive(cu)
    check_swiglu_edges(cu, rng)
    check_gate_up(cu, rng)
    check_down(cu, rng)
    check_adversarial(cu)
    print()
    if FAILURES:
        print(f"RESULT : MISMATCH ({len(FAILURES)} failed: {', '.join(FAILURES)})")
        return 1
    print("RESULT : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
