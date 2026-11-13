# Hash table benchmark

u64-to-u64 hash tables, measured the way the field measures them: the time an
operation takes, for each operation, at sizes from L1 to far past the last
cache.

## Running

    ./run.sh

One command writes base's table with its Lean generator
(`lean/algorithms/Bench/HashTable.lean`), builds the harness (`src/main.rs`,
its own Cargo workspace, `-C target-cpu=native`), runs it pinned to one core,
and writes `REPORT.md`. Each invocation is a run in `results/<time>-<git
rev>/`, holding every sample of every measure.

## What is measured

At 1K, 64K, 1M and 16M entries, nanoseconds an operation of:
- **build**: n distinct keys inserted into an empty table, no reserve;
- **hit**: every key looked up, in a random order;
- **miss**: n absent keys looked up;
- **zipf**: n lookups Zipfian (0.99) over the table's keys.

Engines: base (a SwissTable in CLIF: 16 control bytes probed by one vector
compare, the width hashbrown probes at on x86; Fibonacci hashing, the key
times an odd constant with the bucket and tag from the product's top bits;
tables in its arena, doubled at 7/8 full) and hashbrown 0.15 with its default hasher (foldhash). Every
lookup pass's sum and count are checked equal across the engines.

## Not built

- abseil's `flat_hash_map` and boost's `unordered_flat_map` (C++).
- String keys, erase churn, iteration.
