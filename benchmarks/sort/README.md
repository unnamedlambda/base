# Sort and top-k benchmark

Sorting and top-k selection on the GPU, measured the way the field measures
them: on data already on the device, keys a second.

## Running

    ./run.sh            # every engine
    ./run.sh base       # base alone: the development loop

One command builds CUB's radix sort with the CUDA toolkit's `nvcc`
(`cub_sort.cu`) and base's artifact with its Lean generator
(`lean/algorithms/Ann/Sort.lean`), runs `bench.py` for each engine, and
writes `REPORT.md`. Each invocation is a run in `results/<time>-<git rev>/`;
every timed run's time is kept (`samples`), and the report gives the range
beside the median. It runs in py-base's venv (numpy, PyTorch, py_base).

## What is measured

- **sort**: (u32 key, u32 value) pairs, stable, at 1M, 16M and 64M keys,
  for uniform keys, Zipfian (0.99 over 2^20 keys), already sorted, and 16
  distinct keys. Engines: base (`lean/algorithms/Ann/Sort.lean`: four 8-bit
  passes, each a histogram of the tiles, a scan of it and a scatter by tile)
  and CUB's `DeviceRadixSort::SortPairs` called directly.
- **top-k**: the k smallest of each row of a float matrix, with their
  columns, in no order, at four (rows, columns, k). Engines: base (the ANN
  engine's row select and a gather) and PyTorch's `topk`.

Every answer is checked against numpy before its time counts: a sort's keys
and values exactly; a top-k's keys exactly, and each column against the key
it names.

## Not built

- CPU sorts (ips4o, vqsort, `sort_unstable`): base's engine here is the GPU's.
- cuvs `select_k` batched: its C API takes one row only.
