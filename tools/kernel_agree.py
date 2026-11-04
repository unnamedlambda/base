#!/usr/bin/env python3
"""Run the kernel corpus on the GPU and compare every output with the model's.

    (cd lean && lake env lean -Dexperimental.module=true --run algorithms/Host/KernelCorpus.lean DIR)
    tools/kernel_agree.py DIR

A corpus (`Kernel.Corpus.write`: `Host/KernelCorpus.lean`, and
`GptOss/EwCorpus.lean` for the ML layer's statements as bodies) writes, per
kernel, its PTX (`Kernel.ptx`), its input buffers and the buffers the model
(`Kernel.launch`) leaves. This loads the PTX
through the CUDA driver, launches it on the same inputs and compares the
buffers byte for byte, except that a 4-byte word the model and the GPU both
hold as an f32 NaN matches: which NaN an operation makes is the hardware's,
and the semantics leaves it open. A case that states `exact` (an instruction
checked against `PtxRun`, `Host/PtxBridge.lean`) matches only bit for bit. A case that states `ulps` (one whose
kernel uses the hardware's approximations, which the model runs exactly
rounded) also matches where two finite f32 words of one sign are within that
many units in the last place.

Exits 1 on any difference.
"""
import ctypes as C
import json
import pathlib
import struct
import sys

cu = C.CDLL("libcuda.so.1")


def ck(r, what):
    if r != 0:
        s = C.c_char_p()
        cu.cuGetErrorString(r, C.byref(s))
        raise SystemExit(f"{what}: {r} {s.value}")


def is_nan(word):
    return (word & 0x7F800000) == 0x7F800000 and (word & 0x7FFFFF) != 0


def ulp_close(a, b, ulps):
    finite = lambda w: (w & 0x7F800000) != 0x7F800000
    return ulps > 0 and finite(a) and finite(b) and (a >> 31) == (b >> 31) and abs(a - b) <= ulps


def differences(got, want, ulps=0, exact=False):
    if len(got) != len(want):
        return [f"size {len(got)} vs {len(want)}"]
    out = []
    for i in range(0, len(got) - len(got) % 4, 4):
        g, w = got[i:i + 4], want[i:i + 4]
        if g != w:
            (gw,), (ww,) = struct.unpack("<I", g), struct.unpack("<I", w)
            if exact and not ulps or not (is_nan(gw) and is_nan(ww)) and not ulp_close(gw, ww, ulps):
                out.append(f"word {i // 4}: gpu {gw:#010x} model {ww:#010x}")
    tail = len(got) - len(got) % 4
    if got[tail:] != want[tail:]:
        out.append("tail bytes differ")
    return out


def main():
    root = pathlib.Path(sys.argv[1]) / "kernel_corpus"
    ck(cu.cuInit(0), "init")
    dev = C.c_int()
    ck(cu.cuDeviceGet(C.byref(dev), 0), "device")
    ctx = C.c_void_p()
    ck(cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev), "context")
    ck(cu.cuCtxSetCurrent(ctx), "current")
    failed = False
    for case in json.loads((root / "index.json").read_text()):
        d = root / case["name"]
        mod, fn = C.c_void_p(), C.c_void_p()
        ck(cu.cuModuleLoadData(C.byref(mod), (d / "kernel.ptx").read_bytes() + b"\0"), "load")
        ck(cu.cuModuleGetFunction(C.byref(fn), mod, case["name"].encode()), "function")
        bufs = []
        for i in range(case["nbufs"]):
            data = (d / f"in{i}.bin").read_bytes()
            p = C.c_uint64()
            ck(cu.cuMemAlloc_v2(C.byref(p), max(len(data), 1)), "alloc")
            ck(cu.cuMemcpyHtoD_v2(p, data, len(data)), "upload")
            bufs.append((p, len(data)))
        holders = [p for p, _ in bufs]
        for kind, v in case["scalars"]:
            holders.append({"u32": C.c_uint32, "f32": C.c_uint32, "u64": C.c_uint64}[kind](v))
        params = (C.c_void_p * len(holders))(*[C.cast(C.byref(h), C.c_void_p) for h in holders])
        ck(cu.cuLaunchKernel(fn, case["grid"], 1, 1, case["block"], 1, 1, 0, None, params, None), "launch")
        ck(cu.cuCtxSynchronize(), "sync")
        diffs = []
        for i, (p, n) in enumerate(bufs):
            got = C.create_string_buffer(n)
            ck(cu.cuMemcpyDtoH_v2(got, p, n), "download")
            diffs += [f"buffer {i} {x}" for x in differences(got.raw[:n], (d / f"out{i}.bin").read_bytes(), case.get("ulps", 0), case.get("exact", False))]
            ck(cu.cuMemFree_v2(p), "free")
        ck(cu.cuModuleUnload(mod), "unload")
        if diffs:
            failed = True
            print(f"{case['name']}: {len(diffs)} differences, first: {diffs[:3]}")
        else:
            print(f"{case['name']}: agrees")
    sys.exit(1 if failed else 0)


main()
