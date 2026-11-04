module
public import AlgorithmLib.ML.Ptx.BodyLane
meta import AlgorithmLib.ML.Ptx.BodyLane
public import AlgorithmLib.Proof.LibSpecFloat
meta import AlgorithmLib.Proof.LibSpecFloat
public import AlgorithmLib.Proof.EndsRules
meta import AlgorithmLib.Proof.EndsRules
public import AlgorithmLib.Kernel.Host
meta import AlgorithmLib.Kernel.Host
public import AlgorithmLib.Kernel.Warp
meta import AlgorithmLib.Kernel.Warp
public import AlgorithmLib.Proof.LibStep
meta import AlgorithmLib.Proof.LibStep
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.BodyProof` — a statement's buffers and the words its indexes read

Buffer `b` of a statement's `nbuf` lives at `bufB nbuf b`, its word `j` at
`bufW nbuf b j`. An index reading memory reads the words `rd` gives
(`idxL`), which memory holds (`RdOk`). A store to one buffer word leaves
every other word (`apart_elems`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel

namespace AlgorithmLib.ML.BodyProof

/-- An index's value, a word of memory read by `rd` (buffer, element), none
    where some part of it does not fit 32 bits. -/
def idxL (rd : Buf → Nat → Option Nat) (cta i l : Nat) : IdxE → Option Nat
  | .laneId => if l < 2 ^ 32 then some l else none
  | .loopI => if i < 2 ^ 32 then some i else none
  | .ctaId => if cta < 2 ^ 32 then some cta else none
  | .lit n => if n < 2 ^ 31 then some n else none
  | .ldIdx b off => do rd b (← idxL rd cta i l off)
  | .add a b => do
      let x ← idxL rd cta i l a
      let y ← idxL rd cta i l b
      if x + y < 2 ^ 32 then some (x + y) else none
  | .mul a b => do
      let x ← idxL rd cta i l a
      let y ← idxL rd cta i l b
      if x * y < 2 ^ 32 then some (x * y) else none
  | .ireg _ => none

/-- A statement's buffer `b`, among `nbuf`: its base. -/
def bufB (nbuf b : Nat) : UInt64 := if h : b < nbuf ∧ b < 268435449 then addrOf (.buf ⟨b, h.2⟩) 0 else 0

/-- Where a buffer's word `j` is. -/
def bufW (nbuf b j : Nat) : UInt64 := bufB nbuf b + off (UInt32.ofNat j)

/-- Word `j` of buffer `b` in `m`, as a number. -/
def rdM (nbuf : Nat) (m : Mem) (b : Buf) (j : Nat) : Option Nat := (m.load (bufW nbuf b j) 4).map UInt64.toNat

/-- **`m` holds what `rd` reads**: every word `rd` answers is a word of one
    of the `nbuf` buffers, and `m` holds it. -/
def RdOk (nbuf : Nat) (m : Mem) (rd : Buf → Nat → Option Nat) : Prop :=
  ∀ b j n, rd b j = some n → (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) ∧ rdM nbuf m b j = some n

theorem decode_bufW {nbuf b j : Nat} (hb : b < nbuf) (hb' : b < 268435449) (hj : j < 2 ^ 32) :
    decodeAddr (bufW nbuf b j) = some (.buf ⟨b, hb'⟩, 4 * j) := by
  unfold bufW bufB
  rw [dif_pos ⟨hb, hb'⟩, decode_elem, toNat_ofNat32 hj]

theorem apart_elems {nbuf b j b' j' : Nat} (hb : b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hb' : b' < nbuf ∧ b' < 268435449 ∧ j' < 2 ^ 32) (hne : ¬(b' = b ∧ j' = j)) {m m' : Mem} {v : UInt64}
    (h : m.store (bufW nbuf b j) 4 v = some m') :
    m'.load (bufW nbuf b' j') 4 = m.load (bufW nbuf b' j') 4 := by
  refine load_store_clear h (decode_bufW hb.1 hb.2.1 hb.2.2) (decode_bufW hb'.1 hb'.2.1 hb'.2.2) ?_
  by_cases hbb : b' = b
  · subst hbb
    right
    have hjj : j' ≠ j := fun e => hne ⟨rfl, e⟩
    omega
  · left; intro e; apply hbb; cases e; rfl

end AlgorithmLib.ML.BodyProof
