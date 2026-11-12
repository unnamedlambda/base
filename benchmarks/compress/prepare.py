"""Inputs for the compression benchmark: the corpus and the lzbench build.

    python3 prepare.py          # anything missing: download, check, unpack, build

The corpus is Silesia as lzbench publishes against it: `silesia.tar`, the 12
files of the Silesia corpus tarred (211,947,520 bytes), from the
DataCompression corpus collection, pinned by sha256. lzbench is the field's
in-memory harness: every codec compiled with one compiler and flags, each
compressing and decompressing the whole input and checking the round trip.
It is built from its pinned release with its documented build (`make`).

Inputs live in data/compress/ (the corpus) and build/ (lzbench); results never
do. Runs with the system python3: the CPU side needs nothing else.
"""

import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "compress")
BUILD = os.path.join(HERE, "build")
LZBENCH = os.path.join(BUILD, "lzbench")

CORPUS = (
    "https://raw.githubusercontent.com/DataCompression/corpus-collection/main/Silesia-Corpus/silesia.tar.gz",
    "4eedfb1a11acf1f5c63abc2845bd6c043b1ca75d11b199f9f8c719e30f132de8",  # the .tar.gz
    "a667bac80089b3d04d5496772149e632f0758974b0579d743cd8d6e5e0d63eb2",  # the .tar inside it
)
LZBENCH_REPO = ("https://github.com/inikep/lzbench", "v2.4", "f06583baaed96126db9911aed40b7b505640610f")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def corpus() -> str:
    """data/compress/silesia.tar, checked."""
    url, gz_sha, tar_sha = CORPUS
    tar = os.path.join(DATA, "silesia.tar")
    if os.path.exists(tar) and sha256(tar) == tar_sha:
        return tar
    src = os.path.join(DATA, "src", "silesia.tar.gz")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    if not (os.path.exists(src) and sha256(src) == gz_sha):
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, src + ".part")
        os.replace(src + ".part", src)
    if sha256(src) != gz_sha:
        raise SystemExit(f"{src}: sha256 is not the pinned {gz_sha}")
    with gzip.open(src) as f, open(tar + ".part", "wb") as g:
        shutil.copyfileobj(f, g, 1 << 20)
    os.replace(tar + ".part", tar)
    if sha256(tar) != tar_sha:
        raise SystemExit(f"{tar}: sha256 is not the pinned {tar_sha}")
    return tar


def lzbench() -> str:
    """build/lzbench/lzbench at the pinned release, built if missing."""
    repo, tag, rev = LZBENCH_REPO
    exe = os.path.join(LZBENCH, "lzbench")
    if not os.path.isdir(LZBENCH):
        os.makedirs(BUILD, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", tag, repo, LZBENCH], check=True)
    got = subprocess.run(["git", "-C", LZBENCH, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if got != rev:
        raise SystemExit(f"{LZBENCH} is at {got}, not the pinned {tag} {rev}")
    if not os.path.exists(exe):
        subprocess.run(["make", f"-j{os.cpu_count()}"], cwd=LZBENCH, check=True, stdout=subprocess.DEVNULL)
    return exe


def tiny() -> str:
    """data/compress/tiny: the first 1 MiB of the corpus, for the checks."""
    path = os.path.join(DATA, "tiny")
    if not os.path.exists(path):
        with open(corpus(), "rb") as f, open(path + ".part", "wb") as g:
            g.write(f.read(1 << 20))
        os.replace(path + ".part", path)
    return path


def main():
    print(corpus())
    print(tiny())
    if "--no-build" not in sys.argv:
        print(lzbench())


if __name__ == "__main__":
    main()
