"""Inputs for the memory-operations benchmark: llvm-libc's benchmark, built.

    .venv/bin/python prepare.py      # anything missing: clone, check, build

llvm-libc's memory-function benchmark (libc/benchmarks, the harness of the
automemcpy work) draws sizes from the distributions Google measured in its
fleet (libc/benchmarks/distributions) and offsets at random, so the branch
predictor cannot learn a size. llvm-project is cloned sparse at a pinned
release, Implementations.cpp replaces the harness's list of implementations
(libc/benchmarks/LibcDefaultImplementations.cpp), and the Google Benchmark
target is built with the harness's documented CMake build, by the clang of
the same release (LLVM's own prebuilt toolchain, checked by its sha256):
llvm-libc builds only with clang.
"""

import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
LLVM = os.path.join(BUILD, "llvm-project")
REPO = ("https://github.com/llvm/llvm-project.git", "llvmorg-23.1.4", "7a941dfb2877a0b8cf30323fefa85ad019e0b0f5")  # 2026-10-20
TARGET = "libc.benchmarks.memory_functions.opt_host"
OUT = os.path.join(LLVM, "build")
EXE = os.path.join(OUT, "bin", TARGET)
LIST = os.path.join(LLVM, "libc", "benchmarks", "LibcDefaultImplementations.cpp")
TOOLCHAIN = ("https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.4/LLVM-23.1.4-Linux-X64.tar.zst",
             "efdd1dbd089b99bfcfcde7b8eb1236163f2e88df5a981cca4f783ec9cb5e0378")
CLANG_DIR = os.path.join(BUILD, "LLVM-23.1.4-Linux-X64")
JOBS = 6  # the build's compile jobs: within the 9 GB cap


def run(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def checkout() -> None:
    url, tag, sha = REPO
    if not os.path.isdir(LLVM):
        os.makedirs(BUILD, exist_ok=True)
        run(["git", "clone", "-q", "--depth=1", "--filter=blob:none", "--sparse", "--branch", tag, url, LLVM])
        run(["git", "-C", LLVM, "sparse-checkout", "set", "llvm", "libc", "cmake", "third-party"])
    got = subprocess.run(["git", "-C", LLVM, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if got != sha:
        raise SystemExit(f"{LLVM} is at {got}, not {tag} ({sha})")


def toolchain() -> str:
    """The release's clang, downloaded and checked once."""
    clang = os.path.join(CLANG_DIR, "bin", "clang")
    if not os.path.exists(clang):
        url, sha = TOOLCHAIN
        tar = os.path.join(BUILD, os.path.basename(url))
        if not os.path.exists(tar):
            os.makedirs(BUILD, exist_ok=True)
            run(["curl", "-fsSL", "-o", tar + ".part", url])
            os.rename(tar + ".part", tar)
        got = subprocess.run(["sha256sum", tar], capture_output=True, text=True, check=True).stdout.split()[0]
        if got != sha:
            raise SystemExit(f"{tar}: sha256 {got}, not {sha}")
        # the release is compressed with a 1 GiB window, past zstd's default limit
        run(["tar", "-I", "zstd -d --long=31", "-xf", tar, "-C", BUILD])
        os.remove(tar)
    return clang


def build() -> str:
    checkout()
    clang = toolchain()
    mine = os.path.join(HERE, "Implementations.cpp")
    if not os.path.exists(LIST) or open(LIST).read() != open(mine).read():
        shutil.copyfile(mine, LIST)
    venv_bin = os.path.join(HERE, ".venv", "bin")
    env = dict(os.environ, PATH=venv_bin + os.pathsep + os.environ["PATH"])
    if not os.path.exists(os.path.join(OUT, "build.ninja")):
        run(["cmake", "-G", "Ninja", "-S", os.path.join(LLVM, "llvm"), "-B", OUT,
             "-DCMAKE_BUILD_TYPE=Release", "-DLLVM_ENABLE_PROJECTS=libc", "-DLIBC_INCLUDE_BENCHMARKS=ON",
             "-DLLVM_INCLUDE_TESTS=OFF", "-DLLVM_TARGETS_TO_BUILD=X86",
             f"-DCMAKE_C_COMPILER={clang}", f"-DCMAKE_CXX_COMPILER={clang}++"], env=env, stdout=subprocess.DEVNULL)
    run(["ninja", "-C", OUT, f"-j{JOBS}", TARGET], env=env, stdout=subprocess.DEVNULL)
    if not os.path.exists(EXE):
        raise SystemExit(f"the build made no {EXE}")
    return EXE


if __name__ == "__main__":
    print(build())
