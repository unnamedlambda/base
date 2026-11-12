# Hashing benchmark

Non-cryptographic hashes measured the way the field measures them:
[SMHasher3](https://gitlab.com/fwojcik/smhasher3), its Sanity and Speed
suites on one thread, as its published results are. Base's engine is rapidhash v3 as base's own CLIF
(`lean/algorithms/Bench/Hash.lean`), registered in SMHasher3 as
`base-rapidhash` by `base_hash.cpp` with rapidhash's verification value, so
it is reported only if it is that function bit for bit. The hash tables are
`../hashtable`.

## Running

    ./run.sh

One command does everything, from scratch:
1. creates the virtual environment (cmake and ninja) if it is missing;
2. clones SMHasher3 at its pinned commit, adds `base_hash.cpp` to its hash
   sources and builds it with its documented CMake build, if missing;
   builds base's library and its hash artifact;
3. runs every hash in `speed.HASHES` (`speed.py`, one process each);
4. writes `REPORT.md`.

Nothing measured is reused. Each invocation is a run in
`results/<time>-<git rev>/`, holding its result lines and every process's
wall time (`times.tsv`).

## What is measured

Per hash:
- **verification value**: a hash over a fixed set of the hash's outputs,
  checked against the value its source records; equal values mean the same
  function;
- **sanity**: SMHasher3's sanity checks (determinism, appended and
  prepended zeroes);
- **small keys**: cycles per hash, averaged over 1-31-byte keys, the cost a
  hash table's lookups pay;
- **bulk**: bytes per cycle over 256 KiB keys.

Cycles are the time-stamp counter's; the report converts bulk to GB/s at
the counter's rate.

Hashes: XXH3 (64 and 128), rapidhash, aHash (its AES path and its fallback),
CRC-32C, wyhash, komihash, gxhash, Abseil's hash, CityHash64, FNV-1a-64 as
the classic baseline, and base's rapidhash. Base's is called through its C
API (input, length, output, output length; the seed rides in the output
word), which the small-key cycles include.

## Where base stands

Base's rapidhash runs at 6.5 bytes/cycle bulk against the C build's 8.8,
and 27 cycles a small key against 22: Cranelift forms a 64x64->128 product
as two multiplies where x86 has one, and every hash pays the C API's call.
The frontier is a different algorithm on each measure: gxhash's AES rounds
in bulk (23 bytes/cycle), Abseil's low-level hash on small keys (15 cycles).

**Reproduction.** SMHasher3's results directory at the pinned commit gives
every hash's published verification value and speeds; the report sets them
beside this machine's.

## Not built

- SMHasher3's quality suites (hours per hash): their published pass / fail
  is in the pinned tree's `results/README.md`.
- foldhash, which SMHasher3 does not include.
- Real key sets (ClickBench URLs and user ids, words): these come with A6,
  where the keys meet the tables.
