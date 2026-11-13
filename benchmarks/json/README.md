# JSON parsing benchmark

JSON parsed into a document, measured the way simdjson measures itself:
bytes of text a second, the parse alone, on the standard corpora.

## Running

    ./run.sh

One command fetches simdjson (v5.0.5) and its corpora (`simdjson-data`) at
their pinned commits, builds the harness (`bench.cpp`) against simdjson's
single header and base's library, writes base's parser with its Lean
generator (`lean/algorithms/Bench/Json.lean`), checks and times every file
pinned to one core, and writes `REPORT.md`. Each invocation is a run in
`results/<time>-<git rev>/`, every timed parse kept.

## What is measured

twitter, citm_catalog, canada, github_events, gsoc-2018, mesh and numbers:
each parsed 50 times after one untimed parse. Engines:
- **simdjson**: its DOM parser (`dom::parser::parse`), the text padded once
  beforehand, as its own benchmarks do;
- **base**: one pass over the text with an explicit stack, writing a tape
  (strings unescaped into a buffer; numbers as int64, uint64 or a correctly
  rounded double, by Clinger's fast path and Eisel–Lemire with fast_float's
  table of powers of five). Its loop runs in machine code it carries
  (`lean/algorithms/Bench/JsonAsm.lean`, assembled by base's own assembler)
  where the CPU has AVX2, BMI1 and LZCNT: strings and whitespace 32 bytes at
  a time, a number's digits 16 at a time; any value that code does not take
  (a number past 19 digits or for the slow path, one within 40 bytes of the
  text's end, text that is not JSON) goes back to the CLIF loop.

Before any timing, each file's checksum over every value in document order
is computed from base's tape (its `sum`) and from simdjson's DOM, and must be
equal.

## Not built

- Protobuf (A3's other half): HyperProtoBench against protobuf C++ and upb.
- Integers past 64 bits, which base refuses (simdjson's DOM types them as
  big integers, which this suite's checksum does not fold).
