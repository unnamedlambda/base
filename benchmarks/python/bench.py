"""Device work against PyTorch, through py-base.

Usage: bench.py [--bench torchops|vllm|all] [--rounds N]
"""
import sys
import os

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARKS_DIR)

from benches import torchops_bench, vllm_bench
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

    data_dir = os.path.normpath(
        os.path.join(BENCHMARKS_DIR, "..", "..", "lean", ".lake", "build", "artifacts")
    )

    def artifact_path(name: str) -> str:
        path = os.path.join(data_dir, f"{name}.cbor")
        if not os.path.exists(path):
            print(
                f"ERROR: {path} not found. Run ./run.sh first.",
                file=sys.stderr,
            )
            sys.exit(1)
        return path

    if bench in ("all", "torchops"):
        results = torchops_bench.run(
            artifact_path("cuda_vecadd_persist"),
            artifact_path("cuda_saxpy_persist"),
            rounds,
        )
        harness.print_table(results, col_a="PyTorch")

    if bench in ("all", "vllm"):
        results = vllm_bench.run(
            artifact_path("cuda_gemv"),
            artifact_path("cuda_rmsnorm"),
            artifact_path("cuda_softmax"),
            artifact_path("cuda_decode_attn"),
            rounds,
        )
        harness.print_table(results, col_a="PyTorch")

    if bench not in ("all", "torchops", "vllm"):
        print(f"Unknown benchmark: {bench}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
