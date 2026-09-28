#!/usr/bin/env python3
"""The tables `run.sh` prints, from the runs in the directory it names.

Each line of a run is
    <workload> <column> <ns>
where column is `rust` for the Rust kernel, `clif` for Base's CLIF entry, or
`asm` for Base's entry that calls LLVM's machine code.

A column's time is its fastest over the runs. Ratios are Base / Rust: above 1
means Base is slower. 'over runs' is the ratio's range when each run is taken
on its own.
"""
import glob
import os
import sys
import textwrap
from collections import defaultdict

# Which part of the story each workload tells.
LEVEL = ["histogram", "mandel", "chase"]
SHORT = ["poly", "stream"]

# Why the CLIF is slower, for the workloads where Cranelift cannot reach
# LLVM's code. Each is read off the two compilers' machine code.
CAUSE = {
    "poly": "width: Cranelift's widest vector is 128 bits, and LLVM runs this loop at 256.",
    "stream": "no non-temporal store: LLVM writes each line without reading it first "
              "(vmovntdq); every store Cranelift has reads the line in, so a copy far past the "
              "caches moves half again as much through memory.",
}


def load(raw):
    """(workload, column) -> list over runs of ns, and the workloads in order."""
    rows = defaultdict(list)
    order = []
    for path in sorted(glob.glob(os.path.join(raw, "run_*.tsv"))):
        for line in open(path):
            w, col, t = line.rstrip("\n").split("\t")
            if w not in order:
                order.append(w)
            rows[(w, col)].append(float(t))
    return rows, order


def fmt_us(ns):
    return f"{ns / 1e3:.1f}" if ns >= 1e3 else f"{ns / 1e3:.3f}"


class Row:
    """One workload's numbers."""

    def __init__(self, rows, w):
        self.w = w
        self.rust_runs = rows[(w, "rust")]
        self.rust = min(self.rust_runs)
        self.clif_runs = rows[(w, "clif")]
        self.clif = min(self.clif_runs)
        self.asm_runs = rows.get((w, "asm"))
        self.asm = min(self.asm_runs) if self.asm_runs else None
        runs = [c / r for c, r in zip(self.clif_runs, self.rust_runs)]
        self.ratio = self.clif / self.rust
        self.lo, self.hi = min(runs), max(runs)
        self.noise = max(max(ts) / min(ts) - 1
                         for ts in [self.rust_runs, self.clif_runs] + [self.asm_runs or [1]])


def table(head, rows):
    """An aligned text table, the first column left and the rest right."""
    cols = [head] + rows
    w = [max(len(r[k]) for r in cols) for k in range(len(head))]
    line = lambda r: "  ".join(c.ljust(w[k]) if k == 0 else c.rjust(w[k]) for k, c in enumerate(r))
    print(line(head))
    print("-" * len(line(head)))
    for r in rows:
        print(line(r))
    print()


def para(text):
    print(textwrap.fill(text, 88))
    print()


def notes(items):
    for name, text in items:
        print(textwrap.fill(text, 88, initial_indent=f"  {name}: ",
                            subsequent_indent=" " * (len(name) + 4)))
    print()


def main():
    rows, order = load(sys.argv[1])
    if not rows:
        sys.exit(f"no runs in {sys.argv[1]}")
    R = {w: Row(rows, w) for w in order if w != "execute"}
    noop = min(rows[("execute", "noop")]) if ("execute", "noop") in rows else float("nan")

    para("Every workload is a kernel a Base program would run: CLIF a generator emits, "
         "compiled by Cranelift and run through Driver::execute, against the same kernel in "
         "Rust built for this machine. Every answer is checked against Rust's before "
         "anything is timed; a mismatch stops the run. Ratios are Base / Rust, above 1 Base "
         "slower; 'over runs' is the ratio's range across the runs. No average is given.")

    print("1. Where the CLIF says what LLVM's loop says")
    print()
    table(["Benchmark", "Rust", "Base CLIF", "CLIF/Rust", "over runs"],
          [[r.w, fmt_us(r.rust) + " us", fmt_us(r.clif) + " us", f"{r.ratio:.3f}",
            f"{r.lo:.3f}-{r.hi:.3f}"] for r in (R[w] for w in LEVEL if w in R)])

    print("2. Where Cranelift cannot, and LLVM's code carried in the artifact")
    print()
    sh = [R[w] for w in SHORT if w in R]
    table(["Benchmark", "Rust", "Base CLIF", "Base asm", "CLIF/Rust", "asm/Rust"],
          [[r.w, fmt_us(r.rust) + " us", fmt_us(r.clif) + " us",
            fmt_us(r.asm) + " us", f"{r.ratio:.2f}", f"{r.asm / r.rust:.2f}"] for r in sh])
    notes([(r.w, CAUSE[r.w]) for r in sh])
    para("'Base asm' is the same program calling LLVM's machine code for the kernel, which "
         "the artifact carries as bytes, maps once, and calls as it calls any function.")

    worst = max(R.values(), key=lambda r: r.noise)
    para(f"A call through Driver::execute of an entry that does nothing takes {noop:.1f} ns, "
         f"in every Base time above. Noise is the spread of a column's time between runs; "
         f"the largest here is {worst.noise * 100:.1f}%, on {worst.w}.")


if __name__ == "__main__":
    main()
