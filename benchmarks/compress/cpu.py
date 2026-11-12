"""The CPU codecs through lzbench, one regime per invocation.

    python3 cpu.py <regime> <input>      # appends to <run dir>/cpu.jsonl

Regimes (REGIMES), each lzbench's own measurement at its default times (at
least 1 s of compression and 2 s of decompression per codec level, fastest
pass reported):
    check     one pass of each codec level (run on the first 1 MiB first, so a
              broken build stops the run in seconds)
    file      the whole input on one thread: lzbench's published protocol
    64k-all   64 KiB blocks compressed independently on every core: a storage
              server's block size (RocksDB, Ceph, ZFS records), aggregate rate
    4k        4 KiB blocks on one thread: memory pages and small messages

The codecs (CODECS) are the production set: LZ4,
Snappy, zstd (a negative level, the default range and 19 for archival),
zlib-ng, libdeflate and brotli at their common levels, and memcpy as the
ceiling. lzbench checks every round trip and stops on a mismatch.
"""

import json
import os
import subprocess
import sys

from prepare import LZBENCH

CODECS = "memcpy/lz4/lz4hc,9/snappy/zstd_fast,-1/zstd,1,3,6,19/zlib-ng,1,6/libdeflate,1,6,12/brotli,1,5"
REGIMES = {
    "check": ["-t0,0", "-i1,1"],  # one pass each: the round trips, before a full run
    "file": [],
    "64k-all": ["-b64", f"-T{os.cpu_count()}"],
    "4k": ["-b4"],
}


def run_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", os.environ.get("COMPRESS_RUN", "adhoc"))
    os.makedirs(d, exist_ok=True)
    return d


def parse(name: str):
    """('zstd', '1.5.7', '-3') from lzbench's 'zstd 1.5.7 -3'."""
    parts = name.split()
    codec = parts[0]
    version = parts[1] if len(parts) > 1 else ""
    level = " ".join(parts[2:])
    return codec, version, level


def main():
    regime, path = sys.argv[1], sys.argv[2]
    codecs = sys.argv[3] if len(sys.argv) > 3 else CODECS
    cmd = [os.path.join(LZBENCH, "lzbench"), "-o4", "-q", "-x", f"-e{codecs}", *REGIMES[regime], path]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        raise SystemExit(f"lzbench exited {out.returncode}")
    lines = [l for l in out.stdout.splitlines() if l.count(",") == 6 and not l.startswith("Compressor name")]
    if not lines:
        sys.stderr.write(out.stdout + out.stderr)
        raise SystemExit("lzbench printed no results")
    with open(os.path.join(run_dir(), "cpu.jsonl"), "a") as f:
        for l in lines:
            name, cs, ds, orig, comp, _, fname = l.split(",")
            codec, version, level = parse(name)
            row = {
                "regime": regime,
                "input": os.path.basename(fname),
                "codec": codec,
                "version": version,
                "level": level,
                "name": name,
                "orig_bytes": int(orig),
                "comp_bytes": int(comp),
                "ratio": int(orig) / max(1, int(comp)),
                "compress_MBps": float(cs),
                "decompress_MBps": float(ds),
                "threads": os.cpu_count() if "-T" in " ".join(REGIMES[regime]) else 1,
            }
            f.write(json.dumps(row) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
