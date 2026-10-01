module
public import AlgorithmLib.Host.Ffi
meta import AlgorithmLib.Host.Ffi
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The CPU library, called directly

What each `CpuFn` does: the engine's five functions over what the operating
system answers (`base/src/ffi/cpu.rs`). The CPUs are the world's `cpus`, each
with the core and package the system reports for it; whether a thread may be
pinned is its `cpuPins`. Every call is defined for any argument, answers an
`int`, and changes nothing a program can read: which CPU a thread runs on is
the scheduler's, and only timing depends on it.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

/-- The CPU `c` names, if it is one of the world's. -/
def World.cpuAt? (w : World) (c : UInt64) : Option (Int × Int) :=
  let n := asI32 c
  if n < 0 then none else w.cpus[n.toNat]?

/-- What a CPU library call answers. -/
def cpuAnswer (f : CpuFn) (bits : List UInt64) (w : World) : Int :=
  match f, bits with
  | .count, _ => max 1 w.cpus.length
  | .core, [c] => ((w.cpuAt? c).map (·.1)).getD (-1)
  | .package, [c] => ((w.cpuAt? c).map (·.2)).getD (-1)
  | .pin, [c] => if (w.cpuAt? c).isSome && w.cpuPins then 0 else -1
  | .unpin, _ => if w.cpuPins then 0 else -1
  | _, _ => -1

/-- **What a CPU library call does**: it answers, and leaves the world as it
    was. -/
def cpuCall (f : CpuFn) (bits : List UInt64) (w : World) : Option (Option V × World) :=
  if bits.length ≠ (Ext.cpu f).sig.1.length then none
  else some (some (ofInt .i32 (cpuAnswer f bits w)), w)

end AlgorithmLib.HProg.Sem
