import AlgorithmLib.Core.Artifact
import AlgorithmLib.Surface.Layout

/-!
# Running an artifact from Lean

`AlgorithmLib` builds an `Artifact`, whose entry points are the names the
generator exported them as. The artifact is a value, and this module hands it
straight to the driver, so the program that *builds* an artifact can be the
program that runs it — no file in between, and no Rust or Python host.

What that buys is not speed -- the driver does exactly what it did before --
but that the effectful part of a Lean program becomes a value. A `main` that
uses this does its work through the artifact, and its own `IO` is the three
lines below: open a driver, run, close it.

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

`execute` may be called from any thread: it installs the driver's compiled
functions on the calling thread first, which is how the FFI entry points a
program calls find them. Two drivers used from one thread are fine for the
same reason. A window program is the exception — on macOS its event loop must
be on the main thread.
-/

namespace Base

open AlgorithmLib

/-- A live driver: a compiled program and the memory it runs in.

`raw` is the driver's own pointer. It is private because the only safe
lifetime for it is the one `withDriver` gives. -/
structure Driver where
  private raw : USize

@[extern "lean_base_driver_load"]
private opaque newRaw (artifact : @& ByteArray) : IO USize

@[extern "lean_base_driver_free"]
private opaque freeRaw (handle : USize) : IO Unit

@[extern "lean_base_driver_execute"]
private opaque executeRaw (handle : USize) (name : @& String)
    (input : @& ByteArray) (outputLen : USize) : IO (ByteArray × Int64)

@[extern "lean_base_driver_read_memory"]
private opaque readMemoryRaw (handle : USize) (offset len : USize) : IO ByteArray

@[extern "lean_base_driver_memory_size"]
private opaque memorySizeRaw (handle : USize) : IO USize

namespace Driver

/-- Compile an artifact and take its memory. Release it with `Driver.close`,
or let `withDriver` do it. -/
def load (artifact : Artifact) : IO Driver := do
  match Cbor.encode artifact with
  | .ok bytes => return ⟨← newRaw bytes⟩
  | .error e => throw <| IO.userError e

/-- Release a driver. Using it afterwards is undefined, which is why this is
not what a caller should reach for first. -/
def close (drv : Driver) : IO Unit :=
  freeRaw drv.raw

/-- Run the entry point exported as `name`, answering both the out buffer and
the status.

The status is what an entry point answers without a host and a program having
agreed on a place in memory to leave it: an entry whose `return` carries a
value answers with it, one whose `return` carries nothing answers `0`. -/
def executeStatus (drv : Driver) (name : String)
    (input : ByteArray := .empty) (outputLen : Nat := 0) : IO (ByteArray × Int64) :=
  executeRaw drv.raw name input (USize.ofNat outputLen)

/-- Call one entry point, answering the bytes it wrote to its out buffer.

`name` is the name the artifact exports the entry point as. `input` is what the
program is handed as its input buffer and `outputLen` how much room it is given to
answer; a program that uses neither passes the defaults. The status is dropped — `executeStatus` is the same call
keeping it. -/
def execute (drv : Driver) (name : String)
    (input : ByteArray := .empty) (outputLen : Nat := 0) : IO ByteArray :=
  return (← executeStatus drv name input outputLen).1

/-- `len` bytes of the driver's shared memory from `offset`.

This is how a host reads what a program left in its own memory, at an address
the generator says it writes. A range reaching past the end is an error rather
than a short answer, so a truncated read cannot be mistaken for a result. -/
def readMemory (drv : Driver) (offset len : Nat) : IO ByteArray :=
  readMemoryRaw drv.raw (USize.ofNat offset) (USize.ofNat len)

/-- How much shared memory the driver holds, which bounds `readMemory`. -/
def memorySize (drv : Driver) : IO Nat :=
  return (← memorySizeRaw drv.raw).toNat

end Driver

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
`← drv.readField f.size` and get a `UInt64` it can use directly, rather than a
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

/-- The value at `f`, read from the driver's memory.

This is what makes a host stop carrying offsets and widths of its own: the
`Fld` handed in is the same one the program was compiled against, so there is
no second place for the layout to be written down and no way for the two to
disagree. -/
def Driver.readField (drv : Driver) (f : Fld t) [inst : IsHostField t α] :
    IO α :=
  return inst.decode (← drv.readMemory f.offset t.size)

/-- Open a driver for `artifact`, hand it to `f`, and close it however `f`
ends. -/
def withDriver (artifact : Artifact) (f : Driver → IO α) : IO α := do
  let drv ← Driver.load artifact
  try f drv finally drv.close

/-- Build an artifact and call the entry point it exports as `name`, in one
process.

The whole of what a simple host does. Anything reading a result, calling a
further entry point, or looping wants `withDriver` instead. -/
def run (artifact : Artifact) (name : String)
    (input : ByteArray := .empty) (outputLen : Nat := 0) : IO ByteArray :=
  withDriver artifact fun drv => drv.execute name input outputLen

end Base
