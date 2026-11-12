"""REPORT.md from one run's results.

    python3 report.py [run]      # default: the newest run under results/

The reproduction table reads lzbench's published Silesia results from the
pinned release's own README (build/lzbench/README.md).
"""

import glob
import json
import os
import platform
import re
import subprocess
import sys

from prepare import LZBENCH

HERE = os.path.dirname(os.path.abspath(__file__))
R_DIR = os.path.join(HERE, "results")
REGIMES = {
    "file": "The whole corpus on one thread: lzbench's published protocol.",
    "64k-all": "64 KiB blocks compressed independently on every core: aggregate MB/s of input.",
    "4k": "4 KiB blocks on one thread: memory pages and small messages.",
}


def newest_run() -> str:
    runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(R_DIR, "*", "times.tsv")))
    if not runs:
        raise SystemExit("no run with times.tsv under results/")
    return runs[-1]


def load(run, name):
    p = os.path.join(R_DIR, run, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def num(x, fmt=",.0f"):
    return "—" if x is None else format(x, fmt)


def cpu_table(regime, rows):
    lines = [
        f"### CPU, {regime}",
        "",
        REGIMES[regime],
        "",
        "| codec | level | ratio | compress MB/s | decompress MB/s |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['codec']} {r['version']} | {r['level'] or '—'} | {r['ratio']:.3f} | {num(r['compress_MBps'])} | {num(r['decompress_MBps'])} |"
        )
    return "\n".join(lines) + "\n"


def gpu_table(rows):
    rows = [r for r in rows if r["input"] != "tiny"]
    if not rows:
        return ""
    lines = [
        "### GPU (nvCOMP)",
        "",
        "64 KiB chunks, batched, as nvCOMP's own benchmarks run. Device: input and output already in device",
        "memory. End to end: from host memory to host memory, the PCIe copies included. GPU joules per GB of",
        "input over the device-resident passes.",
        "",
        "| codec | ratio | compress GB/s device | decompress GB/s device | compress GB/s end to end | decompress GB/s end to end | J/GB compress | J/GB decompress |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['codec']} | {r['ratio']:.3f} | {num(r['compress_GBps_device'], '.1f')} | {num(r['decompress_GBps_device'], '.1f')} | "
            f"{num(r['compress_GBps_e2e'], '.2f')} | {num(r['decompress_GBps_e2e'], '.2f')} | {num(r.get('compress_J_per_GB'), '.2f')} | {num(r.get('decompress_J_per_GB'), '.2f')} |"
        )
    return "\n".join(lines) + "\n"


def published():
    """{(codec, level): (version, compress MB/s, decompress MB/s, size)} from lzbench's README."""
    path = os.path.join(LZBENCH, "README.md")
    out, header = {}, ""
    if not os.path.exists(path):
        return out, header
    text = open(path).read()
    m = re.search(r"The following results were obtained.*?(?=\n\n)", text, re.S)
    header = re.sub(r"\[`?([^\]`]+)`?\]\([^)]*\)", r"\1", m.group(0)).replace("\n", " ").strip() if m else ""
    for line in text.splitlines():
        m = re.match(r"\|\s*(\S+) (\S+)( [^|]*?)?\s*\|\s*([\d.]+) MB/s\s*\|\s*([\d.]+) MB/s\s*\|\s*(\d+)\s*\|", line)
        if m:
            codec, version, level = m.group(1), m.group(2), (m.group(3) or "").strip()
            if codec == "memcpy":
                level, version = "", ""
            out[(codec, level)] = (version, float(m.group(4)), float(m.group(5)), int(m.group(6)))
    return out, header


def norm_level(codec, level):
    """lzbench prints zstd_fast's level as '--1' in its README and '-1' in its output."""
    return level.replace("--", "-") if codec == "zstd_fast" else level


def repro_table(rows):
    pub, header = published()
    if not pub:
        return ""
    lines = [
        "### Reproduction: lzbench's published results",
        "",
        f"lzbench's README: {header} A codec at the same version must give the same compressed size, byte for",
        "byte (the corpus and the codec match); a later version may differ. Speed ratios are this machine",
        "against theirs.",
        "",
        "| codec | level | published version | version here | size published | size here | same size | compress × | decompress × |",
        "|---|---|---|---|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        key = next(((c, l) for (c, l) in pub if c == r["codec"] and norm_level(c, l) == r["level"]), None)
        if key is None:
            continue
        v, cs, ds, size = pub[key]
        same = "yes" if size == r["comp_bytes"] else ("no" if v == r["version"] else "newer version")
        lines.append(
            f"| {r['codec']} | {r['level'] or '—'} | {v or '—'} | {r['version'] or '—'} | {size:,} | {r['comp_bytes']:,} | {same} | "
            f"{r['compress_MBps'] / cs:.2f} | {r['decompress_MBps'] / ds:.2f} |"
        )
    return "\n".join(lines) + "\n"


def times_table(run):
    lines = ["### Run time", "", "| step | minutes | status |", "|---|---:|---|"]
    total = 0.0
    for line in open(os.path.join(R_DIR, run, "times.tsv")):
        cmd, status, secs = line.rstrip("\n").split("\t")
        total += float(secs)
        lines.append(f"| {cmd.replace('../../data/compress/', '')} | {float(secs) / 60:.1f} | {status} |")
    lines.append(f"| all | {total / 60:.1f} | |")
    return "\n".join(lines) + "\n"


def machine():
    def sh(c):
        return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

    cpu = sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-")
    gpu = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
    cc = sh("cc --version | head -1")
    return f"{cpu.strip()} ({os.cpu_count()} threads); {gpu}; Linux {platform.release()}; {cc}"


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else newest_run()
    print(f"# Compression benchmark: run {run}\n")
    print(f"Machine: {machine()}.\n")
    print("Corpus: Silesia (`silesia.tar`, 211,947,520 bytes), as lzbench publishes against it. Ratio is input")
    print("bytes over compressed bytes. MB is 10^6 bytes. CPU energy (RAPL) is not readable on this machine.\n")
    print(times_table(run))
    cpu = [r for r in load(run, "cpu.jsonl") if r["regime"] != "check"]
    for regime in REGIMES:
        rs = [r for r in cpu if r["regime"] == regime]
        if rs:
            print(cpu_table(regime, rs))
    t = gpu_table(load(run, "gpu.jsonl"))
    if t:
        print(t)
    t = repro_table([r for r in cpu if r["regime"] == "file"])
    if t:
        print(t)


if __name__ == "__main__":
    main()
