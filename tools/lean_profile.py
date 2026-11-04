#!/usr/bin/env python3
"""Where the Lean build's time goes: every module elaborated on its own with
Lean's profiler, against the oleans a build left, then summed by kind of work
and by declaration.

    tools/lean_profile.py [--jobs N] [MODULE_PREFIX ...]

Build first (tools/lean_build.sh). Profiles land in target/prof/<Module>.txt
and are kept: a second run profiles only the modules that have none, so delete
the directory to start over. The summary goes to stdout.

The trace profiler names the declaration behind each top-level cost; its
nested output for the LZ4 proofs runs to gigabytes, so only its top two levels
are kept. Run it under a memory cap: the ViT and Warp modules evaluate their
checks in the interpreter and take several gigabytes each.
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAN = ROOT / "lean"
OUT = ROOT / "target" / "prof"
CUM = re.compile(r"^\t([a-zA-Z ()\-]+?) ([\d.]+)(ms|s)$")
TOP = re.compile(r"^\[([\w.]+)\] \[([\d.]+)\] (.*)$")
DECL = re.compile(r"\b(theorem|def|abbrev|instance|ship_entry|#eval|#guard)\b\s*([\w.']*)")


def modules():
    for base, pkg in ((LEAN / "lib", LEAN / "lib"), (LEAN / "algorithms", LEAN)):
        for f in sorted(base.rglob("*.lean")):
            if ".lake" not in f.parts:
                yield ".".join(f.relative_to(base).with_suffix("").parts), f, pkg


def profile(name, path, pkg):
    out = OUT / f"{name}.txt"
    if out.exists():
        return
    cmd = ["lake", "env", "lean", "-Dexperimental.module=true", "-Dprofiler=true",
           "-Dprofiler.threshold=200", "-Dtrace.profiler=true", "-Dtrace.profiler.threshold=1000",
           str(path.relative_to(pkg))]
    p = subprocess.Popen(cmd, cwd=pkg, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    kept = [line for line in p.stdout if not re.match(r"^ {3,}\[", line)]
    p.wait()
    out.write_text("".join(kept))


def summary():
    kinds = defaultdict(float)
    decls = []
    for f in sorted(OUT.glob("*.txt")):
        lines = f.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines):
            if m := CUM.match(line):
                kinds[m.group(1)] += float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
            elif (m := TOP.match(line)) and float(m.group(2)) >= 2.0:
                what = m.group(3).replace("elaborating proof of ", "proof ")
                if m.group(1) == "Elab.command":
                    for j in range(i, min(i + 40, len(lines))):
                        if d := DECL.search(lines[j]):
                            what = f"{d.group(1)} {d.group(2)}"
                            break
                decls.append((float(m.group(2)), f.stem, what[:100]))
    print("by kind of work (s, summed over modules; 'blocked' is waiting, not work)")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {v:8.0f}  {k}")
    print("\ndeclarations taking 2 s or more")
    for t, m, what in sorted(decls, reverse=True)[:80]:
        print(f"  {t:7.1f}s  {m:44s} {what}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("prefix", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [m for m in modules() if not a.prefix or any(m[0].startswith(p) for p in a.prefix)]
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda m: profile(*m), todo))
    summary()


if __name__ == "__main__":
    sys.exit(main())
