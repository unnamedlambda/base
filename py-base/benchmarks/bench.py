"""
Python benchmark suite for py_base.

Usage:
    python bench.py [--bench <name>] [--rounds <n>]

    --bench   csv | json | regex | strsearch | all  (default: all)
    --rounds  number of timed iterations per size         (default: 10)

Prerequisites:
    1. maturin develop        # build py_base
    2. ./generate_algos.sh    # generate algos/*.json from Lean source
"""

import sys
import os

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARKS_DIR)

from benches import csv_bench, json_bench, regex_bench, strsearch_bench
import harness


def print_usage():
    print(__doc__)


def main():
    bench = "all"
    rounds = 10

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--bench" and i + 1 < len(args):
            bench = args[i + 1]
            i += 2
        elif args[i] == "--rounds" and i + 1 < len(args):
            rounds = int(args[i + 1])
            i += 2
        elif args[i] in ("--help", "-h"):
            print_usage()
            return
        else:
            print(f"Unknown argument: {args[i]}", file=sys.stderr)
            print_usage()
            sys.exit(1)

    algos_dir = os.path.join(BENCHMARKS_DIR, "algos")

    def algo_path(name: str) -> str:
        path = os.path.join(algos_dir, f"{name}.json")
        if not os.path.exists(path):
            print(
                f"ERROR: {path} not found. Run ./generate_algos.sh first.",
                file=sys.stderr,
            )
            sys.exit(1)
        return path

    if bench in ("all", "csv"):
        results = csv_bench.run(algo_path("csv"), rounds)
        harness.print_table(results)

    if bench in ("all", "json"):
        results = json_bench.run(algo_path("json"), rounds)
        harness.print_table(results)

    if bench in ("all", "regex"):
        results = regex_bench.run(algo_path("regex"), rounds)
        harness.print_table(results)

    if bench in ("all", "strsearch"):
        results = strsearch_bench.run(algo_path("strsearch"), rounds)
        harness.print_table(results)

    if bench not in ("all", "csv", "json", "regex", "strsearch"):
        print(f"Unknown benchmark: {bench}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
