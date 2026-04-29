import sys
import os

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARKS_DIR)

from benches import csv_bench, json_bench, pandas_bench, regex_bench, strsearch_bench, vecops_bench, vllm_bench
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

    data_dir = os.path.normpath(os.path.join(BENCHMARKS_DIR, "..", "..", "lean", "data"))

    def algo_path(name: str) -> str:
        path = os.path.join(data_dir, f"{name}.json")
        if not os.path.exists(path):
            print(
                f"ERROR: {path} not found. Run ./run.sh first.",
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

    if bench in ("all", "vecops"):
        results = vecops_bench.run(algo_path("vecops"), algo_path("clampsum"), rounds)
        harness.print_table(results, col_a="NumPy")

    if bench in ("all", "pandas"):
        results = pandas_bench.run(algo_path("pandas"), algo_path("pandas_filter"), rounds)
        harness.print_table(results, col_a="Pandas")

    if bench in ("vllm",):
        results = vllm_bench.run(
            algo_path("cuda_gemv_persist"),
            algo_path("cuda_rmsnorm_persist"),
            algo_path("cuda_softmax_persist"),
            algo_path("cuda_decoder_layer"),
            algo_path("cuda_decode_attention"),
            rounds,
        )
        harness.print_table(results, col_a="PyTorch")

    if bench not in ("all", "csv", "json", "regex", "strsearch", "vecops", "pandas", "vllm"):
        print(f"Unknown benchmark: {bench}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
