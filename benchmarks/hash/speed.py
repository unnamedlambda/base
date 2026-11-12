"""Hash speed and sanity through SMHasher3, one hash per invocation.

    python speed.py <hash>       # appends to <run dir>/speed.jsonl

SMHasher3's Sanity and Speed suites, single-threaded as its published
results are: the verification value (a hash of a fixed set of hashes, which
must equal the value the hash's own source records), the sanity checks, the
small-key speed (cycles per hash, 1-31-byte keys, and random lengths in that
range) and the bulk speed (bytes per cycle over 256 KiB keys). Cycles are the
time-stamp counter's.
"""

import json
import os
import re
import subprocess
import sys

from prepare import EXE

HASHES = (
    "XXH3-64", "XXH3-128", "rapidhash", "rust-ahash", "rust-ahash-fb", "CRC-32C",
    "wyhash", "komihash", "gxhash", "Abseil64-llh", "CityHash-64", "FNV-1a-64",
)


def run_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", os.environ.get("HASH_RUN", "adhoc"))
    os.makedirs(d, exist_ok=True)
    return d


def parse(text: str) -> dict:
    """The figures of one SMHasher3 Sanity + Speed report."""
    out = {}
    m = re.search(r"Verification value (\S+) (0x[0-9A-Fa-f]+) \.+ (\w+)", text)
    if m:
        out.update(verify_value=m.group(2), verify=m.group(3))
    out["sanity_pass"] = "FAIL" not in text.split("[[[ Speed Tests ]]]")[0]
    m = re.search(r"Small key speed test.*?\nAverage\s+-\s+([\d.]+) cycles/hash\s*\n\s*rnd-byte keys -\s+([\d.]+)", text, re.S)
    if m:
        out.update(small_cycles=float(m.group(1)), small_rnd_cycles=float(m.group(2)))
    m = re.search(r"Bulk speed test - 262144-byte keys.*?\nAverage\s+-\s+([\d.]+) bytes/cycle -\s+([\d.]+) GiB/sec @ ([\d.]+) ghz", text, re.S)
    if m:
        out.update(bulk_bytes_per_cycle=float(m.group(1)), bulk_GiBps_at=float(m.group(2)), assumed_ghz=float(m.group(3)))
    return out


def main():
    h = sys.argv[1]
    p = subprocess.run([EXE, "--ncpu=1", "--test=Sanity,Speed", h], capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
        raise SystemExit(f"SMHasher3 exited {p.returncode}")
    row = {"hash": h, **parse(p.stdout)}
    if "bulk_bytes_per_cycle" not in row or "small_cycles" not in row:
        sys.stderr.write(p.stdout[-3000:])
        raise SystemExit("could not read the speed figures")
    with open(os.path.join(run_dir(), "speed.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(row, flush=True)


if __name__ == "__main__":
    main()
