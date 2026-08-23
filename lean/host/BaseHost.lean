import AlgorithmLib.Core
import AlgorithmLib.Layout

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
and this one does not carry it. Results come back three ways instead:

* `readField`, at the `Fld` the artifact was *built* from -- the offset and the
  width both come from the layout, so a host reads a result by naming the same
  thing the program stored it to;
* the out buffer `execute` answers, which is what a program writes through its
  `out_ptr`/`out_len` offsets, and
* `readMemory`, for an address no field describes.

An artifact whose effects are files, sockets or the GPU needs none of them: it
has already done its work by the time `execute` returns.

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

-- ---------------------------------------------------------------------------
-- Reading a result by the field it was written to
-- ---------------------------------------------------------------------------

open AlgorithmLib.Layout

/-- What a field of type `t` is, on this side of the ABI.

The artifact side has `IsScalarR`, which says how a field is loaded and stored
*in* a program. This is the same idea for the host: how the bytes at a field
come back as a Lean value. `FieldTy` has four cases and no float, so this is
total and there is nothing to fail on.

The Lean type is an `outParam`, so instance search *determines* it from the
field type rather than checking a guess. That is what lets a caller write
`← rt.readField f.size` and get a `UInt64` it can use directly, rather than a
projection out of the class that nothing else will fire on. -/
class IsHostField (t : FieldTy) (α : outParam Type) where
  decode : ByteArray → α

/-- Little-endian, which is what every store the emitter produces writes and
what `base` reads memory back as. -/
private def leNat (bs : ByteArray) : Nat :=
  bs.data.foldr (fun b acc => acc * 256 + b.toNat) 0

instance : IsHostField .u8 UInt8 where
  decode bs := bs.get! 0

instance : IsHostField .i32 UInt32 where
  decode bs := UInt32.ofNat (leNat bs)

instance : IsHostField .i64 UInt64 where
  decode bs := UInt64.ofNat (leNat bs)

instance : IsHostField (.bytes n) ByteArray where
  decode bs := bs

/-- The value at `f`, read from the runtime's memory.

This is what makes a host stop carrying offsets and widths of its own: the
`Fld` handed in is the same one the program was compiled against, so there is
no second place for the layout to be written down and no way for the two to
disagree. -/
def Runtime.readField (rt : Runtime) (f : Fld t) [inst : IsHostField t α] :
    IO α :=
  return inst.decode (← rt.readMemory f.offset t.size)

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

-- ---------------------------------------------------------------------------
-- Artifacts with more than one entry point
-- ---------------------------------------------------------------------------

/-- What a generator emits: one setup, the entry point to call first, and any
further stages by name.

The same three fields the Rust and Python hosts deserialize, so a host that
builds its artifact in-process and one that reads it off disk are looking at
the same thing. A pipeline whose stages must run in order -- load, then infer
-- is the case this exists for: `main` is the one you call first and the rest
are in `extras`. -/
structure App where
  setup : Setup
  main : Algorithm
  extras : List (String × Algorithm) := []

/-- Run a named stage out of `extras`.

An unknown name is an error rather than a silent no-op: a mistyped stage that
did nothing would look exactly like a stage that ran and had nothing to do. -/
def Runtime.executeNamed (rt : Runtime) (app : App) (name : String)
    (data : ByteArray := .empty) (outLen : Nat := 0) : IO ByteArray := do
  match app.extras.lookup name with
  | some alg => rt.execute alg data outLen
  | none =>
      let known := ", ".intercalate (app.extras.map (·.1))
      throw <| IO.userError
        s!"no stage named '{name}'; this artifact has: {known}"

/-- Open a runtime for `app`'s setup, hand it to `f`, and close it however `f`
ends. -/
def withApp (app : App) (f : Runtime → IO α) : IO α :=
  withRuntime app.setup f

/-- Run `app`'s entry point and nothing else. -/
def App.run (app : App) (data : ByteArray := .empty) (outLen : Nat := 0) :
    IO ByteArray :=
  Base.run app.setup app.main data outLen

end Base
