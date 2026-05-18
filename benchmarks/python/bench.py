import sys
import os

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARKS_DIR)

from benches import csv_bench, json_bench, pandas_bench, regex_bench, strsearch_bench, torchops_bench, vecops_bench, vllm_bench, wordcount_bench
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
    generator = "PythonBenchmarks"

    def artifact_path(name: str) -> str:
        path = os.path.join(data_dir, generator, f"{name}.json")
        if not os.path.exists(path):
            print(
                f"ERROR: {path} not found. Run ./run.sh first.",
                file=sys.stderr,
            )
            sys.exit(1)
        return path

    if bench in ("all", "csv"):
        results = csv_bench.run(artifact_path("csv_algorithm"), rounds)
        harness.print_table(results)

    if bench in ("all", "json"):
        results = json_bench.run(artifact_path("json_algorithm"), rounds)
        harness.print_table(results)

    if bench in ("all", "regex"):
        results = regex_bench.run(artifact_path("regex_algorithm"), rounds)
        harness.print_table(results)

    if bench in ("all", "strsearch"):
        results = strsearch_bench.run(artifact_path("strsearch_algorithm"), rounds)
        harness.print_table(results)

    if bench in ("all", "wordcount"):
        results = wordcount_bench.run(artifact_path("wc_algorithm"), rounds)
        harness.print_table(results)

    if bench in ("all", "vecops"):
        results = vecops_bench.run(
            artifact_path("vecops_algorithm"),
            artifact_path("clamp_sum_algorithm"),
            artifact_path("row_dot_algorithm"),
            artifact_path("row_affine_reduce_algorithm"),
            rounds,
        )
        harness.print_table(results, col_a="NumPy")

    if bench in ("all", "torchops"):
        results = torchops_bench.run(
            artifact_path("cuda_vecadd_persist"),
            artifact_path("cuda_saxpy_persist"),
            rounds,
        )
        harness.print_table(results, col_a="PyTorch")

    if bench in ("all", "pandas"):
        results = pandas_bench.run(
            artifact_path("pandas_algorithm"),
            artifact_path("pandas_filter_algorithm"),
            rounds,
        )
        harness.print_table(results, col_a="Pandas")

    if bench in ("all", "vllm"):
        results = vllm_bench.run(
            artifact_path("cuda_gemv"),
            artifact_path("cuda_rmsnorm"),
            artifact_path("cuda_softmax"),
            artifact_path("cuda_decoder"),
            artifact_path("cuda_decode_attn"),
            rounds,
        )
        harness.print_table(results, col_a="PyTorch")

    if bench not in ("all", "csv", "json", "regex", "strsearch", "wordcount", "vecops", "torchops", "pandas", "vllm"):
        print(f"Unknown benchmark: {bench}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
