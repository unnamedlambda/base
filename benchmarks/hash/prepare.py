"""Inputs for the hashing benchmark: SMHasher3, built.

    .venv/bin/python prepare.py      # anything missing: clone, check, build

SMHasher3 is the field's harness for non-cryptographic hashes: speed (bulk
bytes per cycle, cycles per hash for 1-31-byte keys) and quality (its test
suites), with every hash's verification value checked at start. It is cloned
at a pinned commit and built with its documented CMake build; cmake and
ninja come from the suite's virtual environment.
"""

import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
SMHASHER = os.path.join(BUILD, "smhasher3")
REPO = ("https://gitlab.com/fwojcik/smhasher3.git", "94c6395")  # main, 2026-10-08
EXE = os.path.join(SMHASHER, "build", "SMHasher3")


def smhasher() -> str:
    url, rev = REPO
    if not os.path.isdir(SMHASHER):
        os.makedirs(BUILD, exist_ok=True)
        subprocess.run(["git", "clone", "-q", url, SMHASHER], check=True)
        subprocess.run(["git", "-C", SMHASHER, "checkout", "-q", rev], check=True)
    got = subprocess.run(["git", "-C", SMHASHER, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if not got.startswith(rev):
        raise SystemExit(f"{SMHASHER} is at {got}, not the pinned {rev}")
    if not os.path.exists(EXE):
        bdir = os.path.join(SMHASHER, "build")
        os.makedirs(bdir, exist_ok=True)
        venv_bin = os.path.join(HERE, ".venv", "bin")
        env = dict(os.environ, PATH=venv_bin + os.pathsep + os.environ["PATH"])
        subprocess.run(["cmake", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", ".."], cwd=bdir, env=env, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["ninja"], cwd=bdir, env=env, check=True, stdout=subprocess.DEVNULL)
    return EXE


if __name__ == "__main__":
    print(smhasher())
