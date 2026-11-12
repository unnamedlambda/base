"""The card's measured peaks: the ceilings a search is read against.

    python roofline.py

Peaks are measured here, not taken from a data sheet: device-to-device copy
bandwidth (cupy), dp4a throughput and fp32 FMA throughput (a kernel of
independent chains on every SM).
"""

import sys

import numpy as np


BENCH = r"""
extern "C" __global__ void dp4a_peak(int* out, int iters) {
    int a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    int b = 0x01020304 ^ blockIdx.x;
    for (int i = 0; i < iters; i++) {
        a0 = __dp4a(a0, b, a0); a1 = __dp4a(a1, b, a1); a2 = __dp4a(a2, b, a2); a3 = __dp4a(a3, b, a3);
        a4 = __dp4a(a4, b, a4); a5 = __dp4a(a5, b, a5); a6 = __dp4a(a6, b, a6); a7 = __dp4a(a7, b, a7);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
extern "C" __global__ void fma_peak(float* out, int iters) {
    float a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    float b = 0.999f;
    for (int i = 0; i < iters; i++) {
        a0 = fmaf(a0, b, 1.f); a1 = fmaf(a1, b, 1.f); a2 = fmaf(a2, b, 1.f); a3 = fmaf(a3, b, 1.f);
        a4 = fmaf(a4, b, 1.f); a5 = fmaf(a5, b, 1.f); a6 = fmaf(a6, b, 1.f); a7 = fmaf(a7, b, 1.f);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
"""


def event_ms(cp, fn, reps=5):
    fn()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))
    return float(np.median(times))


def peaks(cp):
    mod = cp.RawModule(code=BENCH)
    sms = cp.cuda.Device().attributes["MultiProcessorCount"]
    blocks, threads, iters = sms * 16, 256, 4096
    out_i = cp.empty(blocks * threads, cp.int32)
    out_f = cp.empty(blocks * threads, cp.float32)
    t = event_ms(cp, lambda: mod.get_function("dp4a_peak")((blocks,), (threads,), (out_i, np.int32(iters))))
    dp4a = blocks * threads * iters * 8 / (t * 1e-3)
    t = event_ms(cp, lambda: mod.get_function("fma_peak")((blocks,), (threads,), (out_f, np.int32(iters))))
    fma = blocks * threads * iters * 8 / (t * 1e-3)
    a = cp.empty(1 << 28, cp.uint8)
    b = cp.empty_like(a)
    t = event_ms(cp, lambda: b.__setitem__(slice(None), a))
    bw = 2 * a.nbytes / (t * 1e-3)  # read + write
    return dict(sms=sms, dp4a_per_s=dp4a, fma_per_s=fma, copy_bytes_per_s=bw)


def main():
    import cupy as cp

    pk = peaks(cp)
    print(f"peaks: dp4a {pk['dp4a_per_s'] / 1e12:.2f} T/s ({pk['dp4a_per_s'] * 8 / 1e12:.1f} int8 TOPS), "
          f"fp32 FMA {pk['fma_per_s'] / 1e12:.2f} T/s ({pk['fma_per_s'] * 2 / 1e12:.1f} TFLOPS), "
          f"copy {pk['copy_bytes_per_s'] / 1e9:.0f} GB/s, {pk['sms']} SMs")


if __name__ == "__main__":
    main()
