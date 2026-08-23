import AlgorithmLib.Core

/-!
# Running an artifact from Lean

`AlgorithmLib` builds a `Setup` and an `Algorithm`. Those are values, and until
now the only way to run one was to write it out and let a Rust or Python host
pick it up. This module hands the same values straight to the runtime, so the
program that *builds* an artifact can be the program that runs it.

What that buys is not speed -- the runtime does exactly what it did before --
but that the effectful part of a Lean program becomes a value. A `main` that
uses this does its work through the artifact, and its own `IO` is the three
lines below: open a runtime, run, close it.

## What a host reads back

The Rust and Python surfaces answer Arrow `RecordBatch`es. That is a second ABI,
and this one does not carry it. Results come back two ways instead:

* the out buffer `execute` answers, which is what a program writes through its
  `out_ptr`/`out_len` offsets, and
* `readMemory`, at the offsets an output schema names.

An artifact whose effects are files, sockets or the GPU needs neither: it has
already done its work by the time `execute` returns.

## Threads

A runtime is bound to the thread that created it, because the FFI entry points a
program calls find their compiled functions in a thread-local. `run` and
`withRuntime` stay on one thread, so this only matters to a caller that moves a
`Runtime` into a `Task`; `bindThread` is what makes it callable there.
-/

namespace Base

open AlgorithmLib
open Lean (Json toJson)

/-- A live runtime: a compiled program and the memory it runs in.

`raw` is the runtime's own pointer. It is private because the only safe
lifetime for it is the one `withRuntime` gives. -/
structure Runtime where
  private raw : USize

@[extern "lean_base_new"]
private opaque newRaw (setupJson : @& ByteArray) : IO USize

@[extern "lean_base_free"]
private opaque freeRaw (handle : USize) : IO Unit

@[extern "lean_base_bind_thread"]
private opaque bindThreadRaw (handle : USize) : IO Unit

@[extern "lean_base_execute"]
private opaque executeRaw (handle : USize) (algorithmJson : @& ByteArray)
    (data : @& ByteArray) (outLen : USize) : IO ByteArray

@[extern "lean_base_read_memory"]
private opaque readMemoryRaw (handle : USize) (offset len : USize) : IO ByteArray

@[extern "lean_base_memory_size"]
private opaque memorySizeRaw (handle : USize) : IO USize

private def jsonBytes (j : Json) : ByteArray := j.compress.toUTF8

/-- Compile a setup and take its memory. Release it with `Runtime.close`, or let
`withRuntime` do it. -/
def open_ (setup : Setup) : IO Runtime :=
  return ⟨← newRaw (jsonBytes (toJson setup))⟩

namespace Runtime

/-- Release a runtime. Using it afterwards is undefined, which is why this is
not what a caller should reach for first. -/
def close (rt : Runtime) : IO Unit :=
  freeRaw rt.raw

/-- Make `rt` callable from the current thread.

Only needed by a caller executing from a thread other than the one that opened
it: the compiled functions live in a thread-local as well as in the runtime,
because the FFI entry points a program calls reach them with no runtime in
hand. -/
def bindThread (rt : Runtime) : IO Unit :=
  bindThreadRaw rt.raw

/-- Run one algorithm, answering the bytes it wrote to its out buffer.

`data` is what the program reads through its `data_ptr`/`data_len` offsets and
`outLen` how much room it is given to answer; a program that uses neither passes
the defaults. -/
def execute (rt : Runtime) (algorithm : Algorithm)
    (data : ByteArray := .empty) (outLen : Nat := 0) : IO ByteArray :=
  executeRaw rt.raw (jsonBytes (toJson algorithm)) data (USize.ofNat outLen)

/-- `len` bytes of the runtime's shared memory from `offset`.

This is the offset an output column names. A range reaching past the end is an
error rather than a short answer, so a truncated read cannot be mistaken for a
result. -/
def readMemory (rt : Runtime) (offset len : Nat) : IO ByteArray :=
  readMemoryRaw rt.raw (USize.ofNat offset) (USize.ofNat len)

/-- How much shared memory the runtime holds, which bounds `readMemory`. -/
def memorySize (rt : Runtime) : IO Nat :=
  return (← memorySizeRaw rt.raw).toNat

end Runtime

/-- Open a runtime for `setup`, hand it to `f`, and close it however `f`
ends. -/
def withRuntime (setup : Setup) (f : Runtime → IO α) : IO α := do
  let rt ← open_ setup
  try f rt finally rt.close

/-- Build an artifact and run its entry point, in one process.

The whole of what a simple host does. Anything reading a result, running an
extra stage out of `extras`, or looping wants `withRuntime` instead. -/
def run (setup : Setup) (algorithm : Algorithm)
    (data : ByteArray := .empty) (outLen : Nat := 0) : IO ByteArray :=
  withRuntime setup fun rt => rt.execute algorithm data outLen

end Base
