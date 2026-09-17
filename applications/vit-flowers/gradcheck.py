#!/usr/bin/env python3
"""Our derived backward against PyTorch autograd, parameter by parameter.

numpy + py_base only — torch must not share the process.

Nothing here writes a gradient formula.  The artifact's backward is
`Ten.backwardFrom` applied to the forward tape, and where each gradient landed
is read out of the artifact's own map rather than assumed: `VGMAP_OFF` holds one
`u32` per input, filled from `Ten.backwardCoT`.

The attention weights are compared reassembled.  Our layout splits `qkv` into
nine per-head slices and `proj` into three, because a head is the unit the model
is written in; PyTorch keeps them fused, so the check stacks ours back up rather
than comparing a shape neither side has.
"""
import os
import sys, json, struct
import numpy as np
import py_base
import entries

ART, D = sys.argv[1], sys.argv[2]
NL, SQ, SK, DM, NH, HD, DFF, NC = 12, 200, 224, 192, 3, 64, 768, 128
VBASE = 9 + 22 * NL


def pbase(i):
    return 9 + 22 * i


art = py_base.load_artifact(ART)
base = py_base.Base(art)
# The map is the last region Lean writes: VMEM_SIZE = VGMAP_OFF + 4*VBASE + 0x100.
GMAP = base.memory_size() - 0x100 - 4 * VBASE
gmap = np.frombuffer(base.read_memory(GMAP, 4 * VBASE), "<u4")

blob = np.load(D + "/blob.npy").tobytes()
seed = np.load(D + "/seed.npy")
ref = np.load(D + "/grads.npz")

art_entries = entries.entries(os.path.basename(ART).removesuffix(".json"))
base.execute_into(art_entries["main"], blob, bytearray(0))
base.execute_into(art_entries["run"], b"", bytearray(0))
base.execute_into(art_entries["seed"], seed.astype("<f4").tobytes(), bytearray(0))
base.execute_into(art_entries["bwd"], b"", bytearray(0))

fetch = art_entries["fetchAny"]


def grad(inp, n):
    """The gradient of input buffer `inp`, `n` floats, from where it landed."""
    g = int(gmap[inp])
    assert g != 0, f"input {inp} has no gradient buffer"
    out = bytearray(n * 4)
    base.execute_into(fetch, struct.pack("<II", g, n * 4), out)
    return np.frombuffer(bytes(out), "<f4").copy()


checks = []
for i in range(NL):
    p = pbase(i)
    for name, off, shape in [
        ("norm1.weight", 0, (DM,)), ("norm1.bias", 1, (DM,)),
        ("attn.proj.weight", 14, (DM, DM)), ("attn.proj.bias", 15, (DM,)),
        ("norm2.weight", 16, (DM,)), ("norm2.bias", 17, (DM,)),
        ("mlp.fc1.weight", 18, (DFF, DM)), ("mlp.fc1.bias", 19, (DFF,)),
        ("mlp.fc2.weight", 20, (DM, DFF)), ("mlp.fc2.bias", 21, (DM,)),
    ]:
        checks.append((f"blocks.{i}.{name}",
                       grad(p + off, int(np.prod(shape))).reshape(shape)))
    # q, k and v are one (DM x DM) buffer each, holding all three heads, so
    # PyTorch's stacked qkv is just the three of them end to end.
    qkv = np.concatenate([grad(p + b, DM * DM).reshape(DM, DM) for b in (2, 6, 10)], 0)
    checks.append((f"blocks.{i}.attn.qkv.weight", qkv))
    # the weights are merged but the biases stay per head, because a head's bias
    # is added while its share of the projection is sliced out.
    qkvb = np.concatenate([grad(p + b + h, HD) for b in (3, 7, 11) for h in range(NH)])
    checks.append((f"blocks.{i}.attn.qkv.bias", qkvb))


checks += [("norm.weight", grad(5, DM)), ("norm.bias", grad(7, DM)),
           ("head.weight", grad(6, NC * DM).reshape(NC, DM)),
           ("head.bias", grad(8, NC))]

rows, nz = [], 0
for name, got in checks:
    want = ref[name]
    assert got.shape == want.shape, f"{name}: {got.shape} vs {want.shape}"
    scale = max(float(np.abs(want).max()), 1e-30)
    rel = float(np.abs(got - want).max()) / scale
    gmax = float(np.abs(got).max())
    if gmax > 0:
        nz += 1
    rows.append((rel, name, gmax, float(np.abs(want).max())))

bybl = {}
for rel, name, gm, wm in rows:
    if name.startswith("blocks."):
        b = int(name.split(".")[1])
        bybl.setdefault(b, {})[name.split(".", 2)[2]] = rel
print("--- block 11, every tensor (its backward runs first) ---")
for k, v in sorted(bybl[NL - 1].items(), key=lambda kv: -kv[1]):
    print(f"    {v:9.2e}  {k}")
print("--- relative error by block (backward runs 11 -> 0) ---")
for b in range(NL - 1, -1, -1):
    d = bybl[b]
    print(f"  block {b:2d}  norm1.w {d['norm1.weight']:9.2e}  qkv.w {d['attn.qkv.weight']:9.2e}"
          f"  fc1.w {d['mlp.fc1.weight']:9.2e}  fc2.w {d['mlp.fc2.weight']:9.2e}")

rows.sort(reverse=True)
print(f"tensors  : {len(checks)} checked, {nz} with a nonzero gradient")
print("--- worst 15 ---")
for rel, name, gm, wm in rows[:15]:
    print(f"  {rel:10.3e}  {name:34s} ours {gm:.4e}  torch {wm:.4e}")
print("--- best 8 ---")
for rel, name, gm, wm in rows[-8:]:
    print(f"  {rel:10.3e}  {name:34s} ours {gm:.4e}  torch {wm:.4e}")
zeros = [n for _, n, gm, _ in rows if gm == 0.0]
print(f"--- zero gradient ({len(zeros)}) ---")
for n in zeros: print("  ", n)
worst, worst_name = rows[0][0], rows[0][1]
ok = worst < 5e-3 and nz == len(checks)
print("RESULT   :", "OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
