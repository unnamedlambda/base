import AlgorithmLib.Core
import AlgorithmLib.Layout

/-!
# Running an artifact from Lean

`AlgorithmLib` builds an `Artifact`, whose entry points are the names the
generator exported them as. The artifact is a value, and this module hands it
straight to the runtime, so the program that *builds* an artifact can be the
program that runs it — no file in between, and no Rust or Python host.

What that buys is not speed -- the runtime does exactly what it did before --
but that the effectful part of a Lean program becomes a value. A `main` that
uses this does its work through the artifact, and its own `IO` is the three
lines below: open a runtime, run, close it.

## What a host reads back

Results come back four ways:

* the status `executeStatus` answers -- one `i64`, needing no address;
* `readField`, at the `Fld` the artifact was *built* from -- the offset and the
  width both come from the layout, so a host reads a result by naming the same
  thing the program stored it to;
* the out buffer `execute` answers, which the program is handed as its fourth
  and fifth arguments and writes through, and
* `readMemory`, for an address no field describes.

An artifact whose effects are files, sockets or the GPU needs none of them: it
has already done its work by the time `execute` returns.

## Threads

`execute` may be called from any thread: it installs the runtime's compiled
functions on the calling thread first, which is how the FFI entry points a
program calls find them. Two runtimes used from one thread are fine for the
same reason. A window program is the exception — on macOS its event loop must
be on the main thread.
-/

namespace Base

open AlgorithmLib

/-- A live runtime: a compiled program and the memory it runs in.

`raw` is the runtime's own pointer. It is private because the only safe
lifetime for it is the one `withRuntime` gives. -/
structure Runtime where
  private raw : USize

@[extern "lean_base_new"]
private opaque newRaw (artifact : @& ByteArray) : IO USize

@[extern "lean_base_free"]
private opaque freeRaw (handle : USize) : IO Unit

@[extern "lean_base_execute"]
private opaque executeRaw (handle : USize) (name : @& String)
    (data : @& ByteArray) (outLen : USize) : IO (ByteArray × Int64)

@[extern "lean_base_read_memory"]
private opaque readMemoryRaw (handle : USize) (offset len : USize) : IO ByteArray

@[extern "lean_base_memory_size"]
private opaque memorySizeRaw (handle : USize) : IO USize

/-- Compile an artifact and take its memory. Release it with `Runtime.close`,
or let `withRuntime` do it. -/
def open_ (artifact : Artifact) : IO Runtime := do
  match Cbor.encode artifact with
  | .ok bytes => return ⟨← newRaw bytes⟩
  | .error e => throw <| IO.userError e

namespace Runtime

/-- Release a runtime. Using it afterwards is undefined, which is why this is
not what a caller should reach for first. -/
def close (rt : Runtime) : IO Unit :=
  freeRaw rt.raw

/-- Run the entry point exported as `name`, answering both the out buffer and
the status.

The status is what an entry point answers without a host and a program having
agreed on a place in memory to leave it: an entry whose `return` carries a
value answers with it, one whose `return` carries nothing answers `0`. -/
def executeStatus (rt : Runtime) (name : String)
    (data : ByteArray := .empty) (outLen : Nat := 0) : IO (ByteArray × Int64) :=
  executeRaw rt.raw name data (USize.ofNat outLen)

/-- Call one entry point, answering the bytes it wrote to its out buffer.

`name` is the name the artifact exports the entry point as. `data` is what the
program is handed as its input buffer and `outLen` how much room it is given to
answer; a program that uses neither passes the defaults. The status is dropped — `executeStatus` is the same call
keeping it. -/
def execute (rt : Runtime) (name : String)
    (data : ByteArray := .empty) (outLen : Nat := 0) : IO ByteArray :=
  return (← executeStatus rt name data outLen).1

/-- `len` bytes of the runtime's shared memory from `offset`.

This is how a host reads what a program left in its own memory, at an address
the generator says it writes. A range reaching past the end is an error rather
than a short answer, so a truncated read cannot be mistaken for a result. -/
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

/-- Open a runtime for `artifact`, hand it to `f`, and close it however `f`
ends. -/
def withRuntime (artifact : Artifact) (f : Runtime → IO α) : IO α := do
  let rt ← open_ artifact
  try f rt finally rt.close

/-- Build an artifact and call the entry point it exports as `name`, in one
process.

The whole of what a simple host does. Anything reading a result, calling a
further entry point, or looping wants `withRuntime` instead. -/
def run (artifact : Artifact) (name : String)
    (data : ByteArray := .empty) (outLen : Nat := 0) : IO ByteArray :=
  withRuntime artifact fun rt => rt.execute name data outLen

end Base
