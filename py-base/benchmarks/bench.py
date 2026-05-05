import sys
import os

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARKS_DIR)

from benches import csv_bench, json_bench, pandas_bench, regex_bench, strsearch_bench, torchops_bench, vecops_bench, vllm_bench
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

    def algo_path(module: str, name: str) -> str:
        path = os.path.join(data_dir, module, f"{name}.json")
        if not os.path.exists(path):
            print(
                f"ERROR: {path} not found. Run ./run.sh first.",
                file=sys.stderr,
            )
            sys.exit(1)
        return path

    if bench in ("all", "csv"):
        results = csv_bench.run(algo_path("CsvBenchAlgorithm", "csv_algorithm"), rounds)
        harness.print_table(results)

    if bench in ("all", "json"):
        results = json_bench.run(algo_path("JsonBenchAlgorithm", "json_algorithm"), rounds)
        harness.print_table(results)

    if bench in ("all", "regex"):
        results = regex_bench.run(algo_path("RegexBenchAlgorithm", "regex_algorithm"), rounds)
        harness.print_table(results)

    if bench in ("all", "strsearch"):
        results = strsearch_bench.run(algo_path("StringSearchAlgorithm", "strsearch_algorithm"), rounds)
        harness.print_table(results)

    if bench in ("all", "vecops"):
        results = vecops_bench.run(
            algo_path("VecOpsBenchAlgorithm", "vecops_algorithm"),
            algo_path("ClampSumBenchAlgorithm", "clamp_sum_algorithm"),
            algo_path("RowDotBenchAlgorithm", "row_dot_algorithm"),
            algo_path("RowAffineReduceBenchAlgorithm", "row_affine_reduce_algorithm"),
            rounds,
        )
        harness.print_table(results, col_a="NumPy")

    if bench in ("all", "torchops"):
        results = torchops_bench.run(
            algo_path("CudaVecAddPersistAlgorithm", "cuda_vecadd_persist_load"),
            algo_path("CudaVecAddPersistAlgorithm", "cuda_vecadd_persist_prep"),
            algo_path("CudaVecAddPersistAlgorithm", "cuda_vecadd_persist_infer"),
            algo_path("CudaSaxpyPersistAlgorithm", "cuda_saxpy_persist_load"),
            algo_path("CudaSaxpyPersistAlgorithm", "cuda_saxpy_persist_prep"),
            algo_path("CudaSaxpyPersistAlgorithm", "cuda_saxpy_persist_infer"),
            rounds,
        )
        harness.print_table(results, col_a="PyTorch")

    if bench in ("all", "pandas"):
        results = pandas_bench.run(
            algo_path("PandasBenchAlgorithm", "pandas_algorithm"),
            algo_path("PandasFilterBenchAlgorithm", "pandas_filter_algorithm"),
            rounds,
        )
        harness.print_table(results, col_a="Pandas")

    if bench in ("all", "vllm"):
        results = vllm_bench.run(
            algo_path("CudaGemvPersistAlgorithm", "cuda_gemv_load"),
            algo_path("CudaGemvPersistAlgorithm", "cuda_gemv_prep"),
            algo_path("CudaGemvPersistAlgorithm", "cuda_gemv_infer"),
            algo_path("CudaRmsNormPersistAlgorithm", "cuda_rmsnorm_load"),
            algo_path("CudaRmsNormPersistAlgorithm", "cuda_rmsnorm_prep"),
            algo_path("CudaRmsNormPersistAlgorithm", "cuda_rmsnorm_infer"),
            algo_path("CudaSoftmaxPersistAlgorithm", "cuda_softmax_load"),
            algo_path("CudaSoftmaxPersistAlgorithm", "cuda_softmax_prep"),
            algo_path("CudaSoftmaxPersistAlgorithm", "cuda_softmax_infer"),
            algo_path("CudaSoftmaxPersistAlgorithm", "cuda_softmax_stack"),
            algo_path("CudaDecoderLayerAlgorithm", "cuda_decoder_load"),
            algo_path("CudaDecoderLayerAlgorithm", "cuda_decoder_prep"),
            algo_path("CudaDecoderLayerAlgorithm", "cuda_decoder_infer"),
            algo_path("CudaDecoderLayerAlgorithm", "cuda_decoder_stack16"),
            algo_path("CudaDecoderLayerAlgorithm", "cuda_decoder_stack32"),
            algo_path("CudaDecodeAttentionAlgorithm", "cuda_decode_attn_load"),
            algo_path("CudaDecodeAttentionAlgorithm", "cuda_decode_attn_prep"),
            algo_path("CudaDecodeAttentionAlgorithm", "cuda_decode_attn_infer"),
            algo_path("CudaDecodeAttentionAlgorithm", "cuda_decode_attn_stack"),
            rounds,
        )
        harness.print_table(results, col_a="PyTorch")

    if bench not in ("all", "csv", "json", "regex", "strsearch", "vecops", "torchops", "pandas", "vllm"):
        print(f"Unknown benchmark: {bench}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
