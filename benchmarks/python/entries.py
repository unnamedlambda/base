"""The entry points of the artifacts this directory runs.

A generator fixes each entry's function index, and the artifact carries only
the functions, so these names live here instead of travelling with it. Keep
them in step with the generator that emits each artifact.
"""

ENTRIES = {
    # PythonBenchmarks
    "clamp_sum_algorithm": {
        "main": 1,
    },
    # PythonBenchmarks
    "csv_algorithm": {
        "main": 1,
    },
    # PythonBenchmarks
    "cuda_decode_attn": {
        "infer": 5,
        "main": 1,
        "prep": 2,
        "stack": 6,
    },
    # PythonBenchmarks
    "cuda_decoder": {
        "infer": 5,
        "main": 1,
        "prep": 2,
        "stack16": 6,
        "stack32": 7,
    },
    # PythonBenchmarks
    "cuda_gemv": {
        "infer": 3,
        "main": 1,
        "prep": 2,
    },
    # PythonBenchmarks
    "cuda_rmsnorm": {
        "infer": 3,
        "main": 1,
        "prep": 2,
    },
    # PythonBenchmarks
    "cuda_saxpy_persist": {
        "infer": 3,
        "main": 1,
        "prep": 2,
    },
    # PythonBenchmarks
    "cuda_softmax": {
        "infer": 5,
        "main": 1,
        "prep": 2,
        "stack": 6,
    },
    # PythonBenchmarks
    "cuda_vecadd_persist": {
        "infer": 3,
        "main": 1,
        "prep": 2,
    },
    # PythonBenchmarks
    "json_algorithm": {
        "main": 1,
    },
    # PythonBenchmarks
    "pandas_algorithm": {
        "main": 1,
    },
    # PythonBenchmarks
    "pandas_filter_algorithm": {
        "main": 1,
    },
    # PythonBenchmarks
    "regex_algorithm": {
        "main": 1,
    },
    # PythonBenchmarks
    "row_affine_reduce_algorithm": {
        "main": 1,
    },
    # PythonBenchmarks
    "row_dot_algorithm": {
        "main": 1,
    },
    # PythonBenchmarks
    "strsearch_algorithm": {
        "main": 1,
    },
    # PythonBenchmarks
    "vecops_algorithm": {
        "main": 1,
    },
    # PythonBenchmarks
    "wc_algorithm": {
        "main": 1,
    },
}


def entries(name):
    """The entry points of one artifact, by the file name it was emitted as."""
    try:
        return ENTRIES[name]
    except KeyError:
        raise KeyError(
            f"no entry points recorded for {name!r}; this module lists: "
            + ", ".join(sorted(ENTRIES))
        ) from None
