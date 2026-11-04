module
@[expose] public section

/-!
# How the driver's pointers and handles are spelled

The values the CUDA driver and cuBLAS hand a program, as the model numbers
them: a device allocation's pointer, and a handle of each kind. Plain
arithmetic, kept apart from the driver's model so that what reads a call's
words — a pipeline's records among them — need not import the model.
-/

namespace AlgorithmLib.HProg.Sem

/-- Where device buffer `n` starts: the address of memory region `.buf n`,
    region `6 + n` of `2^36` bytes, which is where a kernel finds it too. An
    offset inside one never reaches the next. -/
def devPtr (n : Nat) : UInt64 := UInt64.ofNat ((6 + n) * 2 ^ 36)

/-- The buffer a device pointer falls in, and the offset into it: the region
    it decodes to, when that is a device buffer's. -/
def devPtrOf (p : UInt64) : Option (Nat × Nat) :=
  let x := p.toNat
  if 6 * 2 ^ 36 ≤ x && x < (6 + 268435449) * 2 ^ 36 then some (x / 2 ^ 36 - 6, x % 2 ^ 36)
  else none

/-- A stream, module, function, event, graph or cuBLAS handle: its kind and
    index. -/
def drvHandle (kind i : Nat) : UInt64 := UInt64.ofNat (2 ^ 56 + kind * 2 ^ 48 + i + 1)

def handleOf (kind : Nat) (h : UInt64) : Option Nat :=
  let x := h.toNat
  let lo := 2 ^ 56 + kind * 2 ^ 48
  if lo < x && x ≤ lo + 2 ^ 48 then some (x - lo - 1) else none

def kStream : Nat := 1
def kModule : Nat := 2
def kFunc : Nat := 3
def kEvent : Nat := 4
def kBlas : Nat := 5
def kGraph : Nat := 6
def kExec : Nat := 7

end AlgorithmLib.HProg.Sem
