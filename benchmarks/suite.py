"""The benchmark suites, run one way and reported in one table.

    python suite.py run [suite ...]     # each suite's own invocations, in turn
    python suite.py report              # REPORT.md from every suite's newest results

A suite is a directory holding a `run.sh` that measures everything from
scratch into its own `results/`, and a `report.py` that, run with
`--headline`, prints its headline numbers as JSON lines of one shape:

    suite, workload, measure, unit, higher (is a larger value better),
    engine, value, base (is this engine base), run (where it came from)

Base may have several engines on one measure (two ways to run the same
work); its best is the one set beside the frontier. A row may carry
`spread`, the (worst, best) value over its runs: where base's and the best
other engine's spreads overlap, the comparison is within noise and counts
as neither ahead nor behind.

The report sets base beside the best other engine on every (workload,
measure): `base / best` is the factor base is better by (above 1) or worse by
(below 1), whichever direction the measure counts as better. A workload with
no base engine yet shows its frontier alone.

The suites run one at a time and never beside tools/check.sh or a loaded
model: each takes the whole machine.
"""

import json
import os
import subprocess
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# Each suite: its directory and the invocations a full measurement takes.
SUITES = {
    "ann": [[], ["t2i-1m"], ["yfcc-1m"], ["bigann-10m"]],
    "hash": [[]],
    "memops": [[]],
    "cpu": [[]],
    "image": [[]],
    "sort": [[]],
    "hashtable": [[]],
    "analytics": [[]],
    "json": [[]],
    # alone and without the memory cap (its engines keep ~10 GB resident)
    "llm": [[]],
    # run here, not yet in the table: their reports have no headline()
    "python": [[]],
    "system": [[]],
}


def headline(suite):
    """A suite's headline rows, from its own report in a process of its own
    (so the suites' modules never meet); none where it has no report or no
    results yet."""
    d = os.path.join(HERE, suite)
    if not os.path.exists(os.path.join(d, "report.py")):
        return []
    r = subprocess.run([sys.executable, "report.py", "--headline"], cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{suite}: no headline ({r.stderr.strip().splitlines()[-1:]})", file=sys.stderr)
        return []
    return [json.loads(line) for line in r.stdout.splitlines() if line.strip()]


def run(names):
    for name in names or SUITES:
        for args in SUITES[name]:
            t0 = time.time()
            r = subprocess.run([os.path.join(HERE, name, "run.sh"), *args])
            print(f"{name} {' '.join(args) or '(default)'}: exit {r.returncode}, {(time.time() - t0) / 60:.1f} min",
                  flush=True)
            if r.returncode != 0:
                raise SystemExit(r.returncode)


def fmt(v, unit):
    if v is None:
        return "—"
    if unit == "" or abs(v) < 10:
        return f"{v:.3f}" if unit == "" else f"{v:.3g} {unit}"
    return f"{v:,.0f} {unit}"


def report():
    groups = defaultdict(list)
    for suite in SUITES:
        for r in headline(suite):
            groups[(r["suite"], r["workload"], r["measure"])].append(r)
    lines = [
        "# Benchmarks",
        "",
        "Every suite's headline numbers, from each suite's newest results (its own REPORT.md has the",
        "detail and methodology). `base / best` is the factor base is ahead (above 1) or behind (below 1)",
        "of the best other engine, in the measure's own better direction.",
        "",
        "| suite | workload | measure | base | best other | base / best |",
        "|---|---|---|---:|---|---:|",
    ]
    behind = []
    for (suite, workload, measure), rs in sorted(groups.items()):
        higher, unit = rs[0]["higher"], rs[0]["unit"]
        mine = [r for r in rs if r["base"]]
        others = [r for r in rs if not r["base"]]
        pick = max if higher else min
        best = pick(others, key=lambda r: r["value"]) if others else None
        # base's best entry, where it has more than one way to run the work
        bm = pick(mine, key=lambda r: r["value"]) if mine else None
        b = bm["value"] if bm else None
        ratio = None
        if b and best and best["value"]:
            ratio = b / best["value"] if higher else best["value"] / b
        noise = False
        if bm and best and "spread" in best:
            if "spread" in bm:
                lo1, hi1 = sorted(bm["spread"])
                lo2, hi2 = sorted(best["spread"])
                noise = lo1 <= hi2 and lo2 <= hi1
        if ratio is not None and ratio < 1 and not noise:
            behind.append((suite, workload, measure, ratio))
        other = f"{best['engine']} {fmt(best['value'], unit)}" if best else "—"
        lines.append(f"| {suite} | {workload} | {measure} | {fmt(b, unit) if mine else 'not built'} | {other} | "
                     f"{f'{ratio:.2f}×' if ratio is not None else '—'}{' (within noise)' if noise else ''} |")
    lines += ["", f"Measures where base is behind: {len(behind)}."]
    for suite, workload, measure, ratio in behind:
        lines.append(f"- {suite}, {workload}, {measure}: {ratio:.2f}×")
    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    if sys.argv[1:2] == ["run"]:
        run(sys.argv[2:])
        report()
    elif sys.argv[1:2] == ["report"]:
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
