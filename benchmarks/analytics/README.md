# Analytics benchmark

ClickBench's numeric queries, measured the way ClickBench measures them: hot
runs (the data already loaded), on the first 10 million rows of its `hits`
table.

## Running

    ./run.sh

One command checks the first 10 of ClickBench's published parquet partitions
against `SHA256SUMS` (downloading what is missing), extracts the columns base
reads (`prepare.py`, by DuckDB), writes base's engine with its Lean generator
(`lean/algorithms/Bench/Analytics.lean`), runs each engine (`bench.py`), and
writes `REPORT.md`. Each invocation is a run in `results/<time>-<git rev>/`,
every timed run of every query kept.

## What is measured

Q1, Q2, Q3, Q6 and Q7 (ClickBench's numbering: counts, sums, an average, a
minimum and maximum, a small group-by), each run once untimed and 10 times
timed. Engines:
- **duckdb** 1.4.1: the rows in its own in-memory table, every core;
- **base**: the columns on the GPU, each query its own pipeline of kernels
  (every block folding its rows into a partial, one block folding the
  partials, no atomic operation; the group-by a warp-private histogram).

Base's answers are checked against DuckDB's, exactly (Q3's average from an
exact sum, as DuckDB's is), before its times count.

This sets a GPU beside a CPU: base's columns are in device memory, DuckDB's
in host memory. The GPU frontier is cuDF, not built here yet.

## Not built

- The queries over strings and `COUNT(DISTINCT …)` (Q4, Q5, Q8 on): base's
  engine needs a 64-bit sort or a hash table on the device for them.
- cuDF, the GPU frontier.
- Parquet decoding (B4's own row): every engine here starts from decoded
  columns.
