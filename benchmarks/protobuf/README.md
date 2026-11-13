# Protobuf benchmark

Protocol Buffers parsed and serialized: base's codec, generated from each
benchmark's own `.proto` by base's Lean generator
(`lean/algorithms/Bench/Proto.lean`), beside protobuf C++ (with an arena and
without) and upb.

## Running

    ./run.sh

One command does everything, from scratch: clones HyperProtoBench and
protobuf v36.2 (with upb and abseil) at their pinned commits, builds protobuf
with its CMake build, generates for each of the six benchmarks base's codec
(`genprotobench <schema>`), protobuf's C++ and upb's code (`protoc`), and the
harness (`bench.cpp`, against that benchmark's generated code), then runs each
harness pinned to one core and writes `REPORT.md`. Nothing measured is reused:
each invocation is a run in `results/<time>-<git rev>/`.

## What is measured

- **Data**: [HyperProtoBench](https://github.com/google/HyperProtoBench)'s six
  benchmarks, each the message shapes of one Google service. Each benchmark's
  top-level messages are built by its own generated `Set` functions, as its
  driver builds them, and serialized by protobuf: that is the corpus.
- **Parse**: the corpus's bytes of wire format a second, into each engine's
  messages (protobuf's on an arena, protobuf's on the heap, upb's on its arena,
  base's in its arena): the median of 200 passes, one core.
- **Serialize**: the same messages back to wire format, likewise.
- **Checks**, before any time counts: upb and protobuf round-trip every
  message to the same bytes; base decodes the corpus and encodes it back byte
  for byte, and protobuf parses base's encoding as the same messages.

The one change to HyperProtoBench's code: its driver's `Arena::CreateMessage`,
which protobuf has since renamed `Arena::Create`.

## Base's codec

The generator reads proto2 (messages, nested messages and enums, `optional`
and `repeated` fields of every scalar type, strings, bytes, enums and
messages) and writes one decoder, one sizer and one writer, which read a table
of each message type's fields. A message is a record in base's arena that
keeps each field's wire value, so the encoder writes back exactly what the
wire said, in field-number order as protobuf does; strings and bytes are
copied into the arena, as protobuf and upb copy them by default. Records nest
at most 100 deep, protobuf's and upb's limit.

## Where base stands

Every corpus is nearly all string bytes, so a parse or a serialize is mostly
copies: base copies a string of 128 bytes or more with its own AVX2 loop
(`MemopsAsm.copy`) into 64-byte-aligned arena space, and asks for huge pages
under its arena once. The codec's code stays a few kilobytes however large
the schema (bench2 is 310 message types), and the sizer and writer walk a
record and the records under it without calls, a frame in the arena for each
level. Base parses every benchmark faster than the best other engine and
serializes five faster; bench1's serialize is level with protobuf's.

## Not built

- proto3, maps, oneofs, groups and unknown-field preservation (HyperProtoBench
  has none of them).
- FlatBuffers and Cap'n Proto, and prost.
