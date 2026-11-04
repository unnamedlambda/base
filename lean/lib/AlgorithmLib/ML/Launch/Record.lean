module
public import AlgorithmLib.Core.ClifData
meta import AlgorithmLib.Core.ClifData
import all Init.Data.Repr
@[expose] public section

/-!
# A device write, as a pipeline reads it

What one call that writes device memory is, in the terms a pipeline is matched
against: the call it is, its kernel and grid where it is a launch, and the
buffers it is handed. A program in the driver's standard layout states these
records by definition (`ML.appLaunch`); nothing reads them out of instructions.
-/

namespace AlgorithmLib.ML

/-- **What one argument of a call is**: a constant, the buffer whose handle is
    at slot `k` of the arena, or something else. -/
inductive ArgDesc where
  | const  : Int → ArgDesc
  | slot   : Int → ArgDesc
  | opaque : ArgDesc
  deriving Repr, DecidableEq, BEq

/-- One device write: the vendor function it calls (its symbol), and for a
    launch the kernel's place, how many buffers it takes, where its parameter
    array is, and its grid and block; `none` where the record does not say. -/
structure LaunchRec where
  fnName   : String
  kernelOff : Option Int
  nBufs    : Option Int
  bindOff  : Option Int
  gridX    : Option Int
  blockX   : Option Int
  /-- For a declared write (a vendor call), what each argument is. A launch
      leaves this empty: its identity is its kernel and its parameter array. -/
  args     : List ArgDesc := []
  /-- The stream the write is issued on, by which of the program's streams it
      is; `none` for the default stream, which is ordered against everything. -/
  stream   : Option Nat := none
  deriving Repr, DecidableEq, BEq

/-- **A buffer handle**: `near k`, the handle in slot `k` of the arena; `far b
    k`, slot `k` of a base `b` the program computes (a per-layer weight); a
    constant; or something else. Two `far` handles are the same buffer exactly
    when both components agree. -/
inductive BufDesc where
  | near   : Int → BufDesc
  | far    : Nat → Int → BufDesc
  | const  : Int → BufDesc
  | opaque : BufDesc
  deriving Repr, DecidableEq, BEq

/-- Forgetting the base gives the `ArgDesc` a `LaunchRec` records. -/
def BufDesc.toArg : BufDesc → ArgDesc
  | .near k   => .slot k
  | .far _ k  => .slot k
  | .const k  => .const k
  | .opaque   => .opaque

/-- **The buffers a device write is handed**: a launch's parameter array, and
    a vendor call's arguments. -/
structure OpBinds where
  /-- A launch's buffers; `none` for a vendor call. -/
  bufs : Option (List BufDesc) := none
  /-- A vendor call's arguments; empty for a launch. -/
  args : List BufDesc := []
  deriving Repr, DecidableEq, BEq

end AlgorithmLib.ML
