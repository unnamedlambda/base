# Memory operations benchmark

`memcpy`, `memmove`, `memset`, `bzero`, `memcmp` and `bcmp`, measured the way
the field measures them: [llvm-libc's memory benchmark](https://github.com/llvm/llvm-project/tree/llvmorg-23.1.4/libc/benchmarks)
(the automemcpy harness) over the size distributions Google published from
its fleet, with base's own
implementations beside the frontier's.

## Running

    ./run.sh

One command does everything, from scratch:
1. creates the virtual environment (cmake and ninja) if it is missing;
2. clones llvm-project sparse at its pinned release (23.1.4), puts this
   suite's `Implementations.cpp` in place of the harness's list of
   implementations, and builds the harness's Google Benchmark target with its
   documented CMake build, by the release's own clang (LLVM's prebuilt
   toolchain, sha256-checked: llvm-libc builds only with clang; about 10
   minutes the first time);
3. runs a quick pass over one distribution per function, then the full
   benchmark (`bench.py`);
4. writes `REPORT.md`.

Nothing measured is reused. Each invocation is a run in
`results/<time>-<git rev>/`, holding its results and every process's wall
time (`times.tsv`).

## What is measured

For each function and each of Google's distributions for it (services A, B,
D, L, M, Q, S, U, W), calls whose sizes are drawn from the distribution and
whose offsets are random, so no size can be learned by the branch predictor;
everything stays in L1, as the harness designs it, so the distribution is the
variable. Nanoseconds per call, the median of 5 repetitions.

Implementations:
- **llvm-libc**: the harness's own, built for this CPU;
- **glibc**: the system C library, through its IFUNC selection (what a Rust
  program gets too: Rust calls the C library's functions on Linux);
- **rep**: `rep movsb` / `rep stosb` (memcpy, memset, bzero);
- **base**: base's own CLIF (`lean/algorithms/Bench/Memops.lean`), compiled by
  base's driver and called through its C API (`base/include/base.h`):
  memcpy, memset, bzero, memcmp and bcmp. Not memmove: a base function's
  input and output do not overlap.

**Where base stands, and why.** Base is behind on every measure today, for
three causes, each measured:
- *The call* (about 1 ns of the 1-2 ns base trails by on Google's sizes): a
  base function is `f(in, in_len, out, out_len)`, so the harness calls it
  through a wrapper that reorders a libc call's arguments. Called directly
  at a fixed size, base's memcpy takes what glibc's does. This is the C API's
  shape, not the code, and the suite keeps it as the measure of that
  boundary: a call that does a few nanoseconds of work is where its fixed
  cost shows.
- *Width* (about 2x at 384-4096 bytes for memcpy, memset and bzero):
  Cranelift's widest vector is 128 bits; llvm-libc and glibc copy 32 bytes an
  instruction with AVX2. A system limit, which base passes where it carries
  machine code for a body (as `../cpu` does).
- *The compare loop* (memcmp and bcmp at 384-4096 bytes): 16 bytes a trip
  with a test each trip, against the frontier's unrolled 32-byte AVX2 loops.

**Check.** Before anything is timed, every implementation is compared with a
byte loop at every size from 0 to 1024 bytes and several offsets
(`Implementations.cpp`); a mismatch stops the run.

## Not built

- Cold buffers (64 MB): the harness measures in L1 only.
- Base's machine-code bodies for the large sizes (AVX2, as `../cpu` carries
  for `stream`), and memmove.
- A published reference: the harness publishes distributions, not results.
