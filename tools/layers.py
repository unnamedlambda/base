#!/usr/bin/env python3
"""The Lean tree's layering, checked.

A layer of the library imports only the layers below it, and an application
directory imports the library, the shared scans and tokenizer, and itself --
never another application. A generator (a `lean_exe` root) must not reach the
host proofs: editing a proof would otherwise rebuild, relink and rerun it.

Where the tree does not meet a rule yet, the exception is listed here, so it
can only shrink: a new one fails the check, and a listed one that has been
fixed fails it too until it is removed from the list.

    tools/layers.py            check, exit 1 on a violation
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "lean"

# What each library layer may import, besides itself.
LIB_BELOW = {
    "Core": set(),
    "Vocab": {"Core"},
    "Host": {"Core", "Vocab"},
    "Surface": {"Core", "Vocab", "Host.Term"},
    "LZ4": {"Core", "Vocab"},
    "ML": {"Core", "Vocab", "Host", "Surface"},
    # What is proved about programs the surface builds: combinator lemmas and
    # their instances.
    "Proof": {"Core", "Vocab", "Host", "Surface"},
}
# The ML library's own layers, lowest first: carriers, the mathematics, the
# warp machine, the PTX lowering, the kernel schemas, the model as written
# (tensor terms, their tape and what it computes), launch sequences, and the
# lowering of models to launches with its schedules. `Assumptions` (the ledger) sits on top.
ML_ORDER = ["Num", "Math", "Machine", "Ptx", "Kernel", "Tensor", "Launch", "Model"]
# Application directories every application may import.
SHARED_APPS = {"Scan", "Tokenizer"}
# Application imports of another application, by (importer dir, imported dir).
APP_EDGES = {("Host", "Bench"), ("Bench", "Host")}
# The host proofs a generator must not reach.
PROOFS = {f"AlgorithmLib.Host.{m}" for m in
          ["Sem", "Blocks", "Frames", "Sound", "Trust", "SemCheck", "Clif", "ClifCheck", "HostIR",
           "StaticCong", "Static", "Hoare", "DevSpec", "StaticHoare", "Contracts", "Lifecycle", "LaunchSpec"]}
# Generators that still reach them, and why.
GENERATOR_EXCEPTIONS = {
    # `ML.HostBridge` states host-to-kernel facts over `HostIR`, and these
    # generators prove them beside the emission.
    "Warp.BackwardWide", "Warp.MlpCifar", "Qwen2.Algorithm", "Qwen2.OnDisk",
    "GptOss.Decode", "GptOss.Algorithm",
    # The conformance corpora and the pilots are checked against the semantics.
    "Host.Pilots", "Host.Corpus", "Host.CudaCorpus", "Host.GpuCorpus", "Host.StreamCorpus", "Host.LmdbCorpus", "Host.WindowCorpus", "Host.NativeCorpus", "Host.LocalCorpus", "Host.ThreadCorpus", "Host.DriverCorpus", "Host.CpuCorpus", "Host.SerialCorpus", "Host.UsbCorpus", "Host.WgpuCorpus",
}

IMPORT = re.compile(r"^\s*(?:public\s+)?(?:meta\s+)?import\s+(?:all\s+)?(\S+)", re.M)


def modules():
    out = {}
    for base in ("lib", "algorithms"):
        for p in (ROOT / base).rglob("*.lean"):
            if ".lake" in p.parts or p.name == "lakefile.lean":
                continue
            out[".".join(p.relative_to(ROOT / base).with_suffix("").parts)] = p
    return out


def lib_layer(m):
    parts = m.split(".")
    return parts[1] if m.startswith("AlgorithmLib.") and len(parts) > 2 else None


def allowed(m, i, mods):
    if i not in mods:
        return True
    if m.startswith("AlgorithmLib"):
        if not i.startswith("AlgorithmLib"):
            return False
        a, b = lib_layer(m), lib_layer(i)
        if a is None or a == b:
            return True
        below = LIB_BELOW.get(a, set())
        return b in below or any(i.startswith(f"AlgorithmLib.{x}") for x in below)
    if i.startswith("AlgorithmLib"):
        return True
    a, b = m.split(".")[0], i.split(".")[0]
    # a scan sits above every application it checks, a Ship proof above every
    # application it proves, an executable root above the generator it runs
    return a == b or a in ("Scan", "Ship", "Main") or b in SHARED_APPS or (a, b) in APP_EDGES


def main():
    mods = modules()
    imps = {m: set(IMPORT.findall(p.read_text())) & set(mods) for m, p in mods.items()}
    bad = [f"{m} imports {i}" for m in sorted(mods) for i in sorted(imps[m]) if not allowed(m, i, mods)]

    def ml_rank(m):
        parts = m.split(".")
        if m.startswith("AlgorithmLib.ML.") and len(parts) > 3 and parts[2] in ML_ORDER:
            return ML_ORDER.index(parts[2])
        return None
    for m in sorted(mods):
        for i in sorted(imps[m]):
            a, b = ml_rank(m), ml_rank(i)
            if a is not None and b is not None and b > a:
                bad.append(f"{m} imports {i}, a higher ML layer")

    def closure(m, acc):
        for i in imps[m]:
            if i not in acc:
                acc.add(i)
                closure(i, acc)
        return acc

    # an executable's root is a one-line `Main.<module>`; judge the module it runs
    roots = [r.removeprefix("Main.") for r in
             re.findall(r"root := `(\S+)", (ROOT / "lakefile.lean").read_text())]
    reach = {r for r in roots if closure(r, set()) & PROOFS}
    bad += [f"generator {r} reaches the host proofs" for r in sorted(reach - GENERATOR_EXCEPTIONS)]
    bad += [f"generator {r} no longer reaches the host proofs: drop its exception"
            for r in sorted(GENERATOR_EXCEPTIONS - reach)]
    for b in bad:
        print(b)
    if bad:
        return 1
    print(f"{len(mods)} modules layered; {len(roots)} generators, "
          f"{len(GENERATOR_EXCEPTIONS)} still reaching the host proofs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
