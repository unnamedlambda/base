# Compression benchmark

Lossless compression measured the way the field measures it:
[lzbench](https://github.com/inikep/lzbench), every codec compiled into one
executable with the same compiler and flags, compressing and decompressing
the [Silesia corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia)
in memory and checking every round trip; and on the GPU, nvCOMP's batched
API over 64 KiB chunks, as nvCOMP's own benchmarks run it. No base engine yet.

## Running

    ./run.sh

One command does everything, from scratch:
1. creates the virtual environment (cupy and nvCOMP, for the GPU side) if
   it is missing;
2. downloads the corpus if missing and checks its pinned sha256;
3. clones lzbench at its pinned release (v2.4) and builds it with its
   documented `make`, if missing;
4. runs every codec level once on the first 1 MiB, and every GPU codec, and
   stops if any round trip fails;
5. runs every regime on the corpus, then the GPU codecs;
6. writes `REPORT.md`.

Nothing measured is reused. Each invocation is a run in
`results/<time>-<git rev>/`, holding its result lines and every process's
wall time (`times.tsv`). `report.py [run]` rebuilds a report from any run.

## What is measured

**Corpus.** `silesia.tar`: the 12 Silesia files tarred, 211,947,520 bytes,
the input lzbench publishes against.

**CPU** (`cpu.py`, lzbench's own timing: at least 1 s of compression and 2 s
of decompression per codec level, fastest pass):

| regime | what it is |
|---|---|
| `file` | the whole corpus on one thread: lzbench's published protocol |
| `64k-all` | 64 KiB blocks compressed independently on all 12 threads: a storage server's block size, aggregate rate |
| `4k` | 4 KiB blocks on one thread: memory pages and small messages |

Codecs, at their common levels: LZ4 (and LZ4HC 9), Snappy, zstd (−1, 1, 3, 6
and 19 for archival), zlib-ng (1, 6), libdeflate (1, 6, 12), brotli (1, 5), and
memcpy as the ceiling. Brotli 11 and the other archival levels below 1 MB/s
are left out: one pass over the corpus takes six minutes.

**GPU** (`gpu.py`, nvCOMP 6.0): LZ4, Snappy, zstd, Deflate and GDeflate over
64 KiB chunks, batched. Device-resident GB/s (input and output in device
memory) and end to end (host memory to host memory, PCIe included), the
ratio, and GPU joules per GB.

**Reproduction.** lzbench's README publishes its own Silesia results (lzbench
2.0.1, an EPYC 9554 at 3.1 GHz, one thread). For each codec level measured
here at the same version, the compressed size must match byte for byte; the
speed ratio is the hardware difference.

## Not built

- Small blocks with a trained zstd dictionary (lzbench has no dictionary
  mode).
- A real-payload set (JSON logs, protobuf messages,
  Parquet pages, a source tarball).
- base's own GPU LZ4.
- CPU energy: RAPL is root-only on this machine.
