/-!
# base's C API, in Lean's terms

The four calls of `base/include/base.h`, and nothing built on them:

* `Driver.load`: `base_driver_load`, on an encoded artifact;
* `Driver.function`: `base_driver_function`, answered as a Lean function;
* `Driver.arenaSize`, `Driver.read`, `Driver.write`: `base_driver_arena`. A
  Lean value cannot alias foreign memory, so the arena is reached by copy:
  its size, the bytes at an offset, and bytes put at an offset. Each aligned
  8-byte word is moved with one atomic access and every other byte with one,
  as programs on other threads reach it;
* `Driver.close`: `base_driver_free`.

A function may be called from any thread, and from several at once.
-/

namespace Base

/-- A live driver: a compiled program and the arena it runs in. -/
structure Driver where
  private raw : USize

/-- A function of a driver: run on `input` and `output`, it answers `output` as
the program left it, and the program's status. `output` is written in place
when nothing else holds it, and copied first otherwise. -/
abbrev Function := ByteArray → ByteArray → IO (ByteArray × Int64)

@[extern "lean_base_driver_load"]
private opaque loadRaw (artifact : @& ByteArray) : IO USize

@[extern "lean_base_driver_function"]
private opaque functionRaw (handle : USize) (name : @& String) : IO USize

@[extern "lean_base_call"]
private opaque callRaw (fn : USize) (input : @& ByteArray) (output : ByteArray) : IO (ByteArray × Int64)

@[extern "lean_base_driver_arena_size"]
private opaque arenaSizeRaw (handle : USize) : IO USize

@[extern "lean_base_driver_read"]
private opaque readRaw (handle : USize) (offset len : USize) : IO ByteArray

@[extern "lean_base_driver_write"]
private opaque writeRaw (handle : USize) (offset : USize) (bytes : @& ByteArray) : IO Unit

@[extern "lean_base_driver_free"]
private opaque freeRaw (handle : USize) : IO Unit

namespace Driver

/-- Loads an encoded artifact; an error says why one is refused. -/
def load (artifact : ByteArray) : IO Driver :=
  return ⟨← loadRaw artifact⟩

/-- The function named `name`; an error when there is none. -/
def function (drv : Driver) (name : String) : IO Function := do
  let fn ← functionRaw drv.raw name
  return fun input output => callRaw fn input output

/-- How many bytes the arena holds. -/
def arenaSize (drv : Driver) : IO Nat :=
  return (← arenaSizeRaw drv.raw).toNat

/-- The `len` bytes of the arena from `offset`; an error past its end. -/
def read (drv : Driver) (offset len : Nat) : IO ByteArray :=
  readRaw drv.raw (USize.ofNat offset) (USize.ofNat len)

/-- Puts `bytes` into the arena at `offset`; an error past its end. -/
def write (drv : Driver) (offset : Nat) (bytes : ByteArray) : IO Unit :=
  writeRaw drv.raw (USize.ofNat offset) bytes

/-- Releases the driver. No call on it, and no thread a program of it
started, may still be running, and it is not used again. -/
def close (drv : Driver) : IO Unit :=
  freeRaw drv.raw

end Driver

end Base
