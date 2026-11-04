module
public import AlgorithmLib.Kernel.Run
meta import AlgorithmLib.Kernel.Run
public import AlgorithmLib.Host.DrvCodes
meta import AlgorithmLib.Host.DrvCodes
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Kernel.Law` — a launch runs the kernel it prints

The host's model does not run PTX: what a launch leaves in the buffers it
binds is the world's oracle `World.kernel`. For a kernel that is a body
printed by `Kernel.ptx`, the oracle is not free: launching that text runs the
body on every thread (`Kernel.launchOn`). That is the one assumption a claim
about a program launching such a kernel rests on, as `Cranelift` compiling a
body to what it means is for the host's own code; `tools/kernel_agree.py` is
its test.

A device pointer the host holds is the address a kernel uses for the same
bytes (`devPtr n = addrOf (.buf n) 0`), so a launch's argument words reach the
body unchanged, and the buffers it binds are the kernel's memory at their own
numbers.
-/

namespace AlgorithmLib.Kernel

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

/-- The PTX text a kernel prints, or nothing where the printer refuses it. -/
def Kernel.text (k : Kernel) : String :=
  match k.ptx with
  | .ok s => s
  | .error _ => ""

/-- A launch's argument words, as the values its body's parameters are. -/
def Kernel.argsOf (k : Kernel) (l : Launch) : List V :=
  (k.params.zip l.args).map fun (t, a) => norm t a

/-- The kernel's memory for a launch: each bound buffer at its own number. -/
def bindsMem (l : Launch) (ins : List ByteArray) (id : Nat) : ByteArray :=
  match l.binds.idxOf? id with
  | some j => ins.getD j .empty
  | none => .empty

/-- **The launch law**: launching the text `k` prints, one-dimensional, where
    the model's threads of `k`'s body run, leaves in each bound buffer what one
    such run leaves there. -/
def Kernel.Runs (oracle : Launch → List ByteArray → List ByteArray) (k : Kernel) : Prop :=
  ∀ (l : Launch) (ins : List ByteArray) (gx bx : UInt64),
    l.kernel = k.text → l.entry = k.name → l.dims = [gx, 1, 1, bx, 1, 1] →
    (∃ bufs, LaunchRuns k gx.toNat bx.toNat (k.argsOf l) (bindsMem l ins) bufs) →
    ∃ bufs, LaunchRuns k gx.toNat bx.toNat (k.argsOf l) (bindsMem l ins) bufs ∧ oracle l ins = l.binds.map bufs

end AlgorithmLib.Kernel
