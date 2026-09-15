#!/usr/bin/env python3
"""Run the artifact with no torch in the process: cuDNN/cuBLAS state from a
PyTorch CUDA context in the same process is not ours to share."""
import os
import sys, numpy as np, py_base
import entries
ART, D = sys.argv[1], sys.argv[2]
SQ, NC = 200, 128
blob = np.load(D + "/blob.npy").tobytes()
ref = np.load(D + "/ref.npy")
art = py_base.load_artifact(ART)
art_entries = entries.entries(os.path.basename(ART).removesuffix(".json"))
base = py_base.Base(art)
base.execute_into(art_entries["main"], blob, bytearray(0))
base.execute_into(art_entries["run"], b"", bytearray(0))
out = bytearray(SQ*NC*4)
base.execute_into(art_entries["fetch"], b"", out)
got = np.frombuffer(bytes(out), "<f4").reshape(SQ, NC)[0]
d = np.abs(got - ref); scale = max(float(np.abs(ref).max()), 1e-30)
print(f"gpu[:4]  : {got[:4]}")
print(f"ref[:4]  : {ref[:4]}")
print(f"max |Δ|  : {d.max():.3e}   rel {d.max()/scale:.3e}")
print("RESULT   :", "OK" if d.max()/scale < 2e-4 else "MISMATCH")
