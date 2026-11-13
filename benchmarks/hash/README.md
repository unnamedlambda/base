# Hashing benchmark

Non-cryptographic hashes measured the way the field measures them:
[SMHasher3](https://gitlab.com/fwojcik/smhasher3), its Sanity and Speed
suites on one thread, as its published results are. Base's engines (`lean/algorithms/Bench/Hash.lean`) are
rapidhash v3 as base's own CLIF, registered in SMHasher3 as
`base-rapidhash`, and gxhash's 64-bit form as base's own AES loop
(`lean/algorithms/Bench/HashAsm.lean`, assembled by base's assembler),
registered as `base-gxhash64`; `base_hash.cpp` registers each with the
verification value of the function it implements, so it is reported only if
it is that function bit for bit. The hash tables are `../hashtable`.

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
the classic baseline, and base's rapidhash and gxhash. Base's are called through its C
API (input, length, output, output length; the seed rides in the output
word), which the small-key cycles include.

## Where base stands

Base leads in bulk: its gxhash runs at 26.5 bytes/cycle against gxhash's
own 23.9, the same AES rounds (two lanes of eight vectors a trip) with the
loop's loads folded into the rounds. On small keys the frontier is Abseil's
low-level hash (15.9 cycles); base's gxhash takes 34 and its rapidhash 27
(the C build's 22): every hash of base's pays the C API's call and the
seed's trip through the output word, and Cranelift forms rapidhash's
64x64->128 product as two multiplies where x86 has one. The small-key rows
stay as the measure of that boundary: a base function takes whole buffers
through its four-argument C API, which costs nothing beside real work (as
in `../hashtable`, where base leads) and shows where a call does a few
nanoseconds of it.

**Reproduction.** SMHasher3's results directory at the pinned commit gives
every hash's published verification value and speeds; the report sets them
beside this machine's.

## Not built

- SMHasher3's quality suites (hours per hash): their published pass / fail
  is in the pinned tree's `results/README.md`.
- foldhash, which SMHasher3 does not include.
- Real key sets (ClickBench URLs and user ids, words): these come with A6,
  where the keys meet the tables.
