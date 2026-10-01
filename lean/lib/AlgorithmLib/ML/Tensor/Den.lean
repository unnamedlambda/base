module
public import AlgorithmLib.ML.Tensor.Ten
meta import AlgorithmLib.ML.Tensor.Ten
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# What a tape operation computes

`TOp.den` is the mathematics of one operation, written from the op alone:
which addresses of which buffer it writes --- the extents come from the op,
which carries them from the term's types --- and what lands there, in the
committed fold orders (`bflyFold`, `dotStridedLane`, `denote`, `smSpec`). It
mentions no stage, no grid, no `dom`: it is what the lowering is checked
against (`Model/TenDenote`), so it must not be produced by it.
-/

namespace AlgorithmLib.ML

-- ---------------------------------------------------------------------------
-- Ownership arithmetic, once
-- ---------------------------------------------------------------------------

/-- A sample-major address is inside the rectangle. -/
theorem addr_lt {s cta B g : Nat} (hs : s < B) (hc : cta < g) : s * g + cta < B * g := by
  have h1 : s + 1 ≤ B := hs
  have h2 : (s + 1) * g ≤ B * g := Nat.mul_le_mul_right g h1
  rw [Nat.succ_mul] at h2
  omega

/-- The batched reduction's ownership, as a plain bound. -/
theorem batched_own_iff (B g a : Nat) (hg : 0 < g) :
    (∃ cta, cta < g ∧ ∃ s, s < B ∧ a = s * g + cta) ↔ a < B * g := by
  constructor
  · rintro ⟨cta, hc, s, hs, rfl⟩
    exact addr_lt hs hc
  · intro ha
    refine ⟨a % g, Nat.mod_lt _ hg, a / g, ?_, ?_⟩
    · exact (Nat.div_lt_iff_lt_mul hg).mpr ha
    · have h := Nat.div_add_mod a g
      rw [Nat.mul_comm (a / g) g]
      omega

/-- The row-segment ownership — outer products and row passes share it. -/
theorem seg_own_iff (n off w g a : Nat) (hw : off + w ≤ n) :
    (∃ cta, cta < g ∧ ∃ t, t < w ∧ cta * n + off + t = a)
      ↔ a / n < g ∧ off ≤ a % n ∧ a % n < off + w := by
  constructor
  · rintro ⟨cta, hc, t, ht, rfl⟩
    have hn : 0 < n := by omega
    have hin : off + t < n := by omega
    have hd : (cta * n + (off + t)) / n = cta := by
      rw [Nat.mul_comm cta n, Nat.mul_add_div hn, Nat.div_eq_of_lt hin]
      omega
    have hm : (cta * n + (off + t)) % n = off + t := by
      rw [Nat.mul_comm cta n, Nat.mul_add_mod, Nat.mod_eq_of_lt hin]
    constructor
    · rw [show cta * n + off + t = cta * n + (off + t) by omega, hd]; exact hc
    constructor
    · rw [show cta * n + off + t = cta * n + (off + t) by omega, hm]; omega
    · rw [show cta * n + off + t = cta * n + (off + t) by omega, hm]; omega
  · rintro ⟨hc, hlo, hhi⟩
    have hn : 0 < n := by omega
    refine ⟨a / n, hc, a % n - off, by omega, ?_⟩
    have h := Nat.div_add_mod a n
    rw [Nat.mul_comm (a / n) n]
    omega

-- ---------------------------------------------------------------------------
-- The mathematics of one operation
-- ---------------------------------------------------------------------------

/-- The committed row-times-row reduction: a `K`-trip strided accumulation in
    each lane, folded by the butterfly.  This is the number a matvec puts at
    one output element. -/
def rowDot (A B : Nat → Float32) (fA fB : Nat → Nat) (K : Nat) : Float32 :=
  bflyFold (dotStridedLane A B
    (fun i l => fA (i * 32 + l.val)) (fun i l => fB (i * 32 + l.val)) K)
    ⟨0, by decide⟩

open Classical in
/-- **What one operation does to memory** — written from the op alone.

    The output rectangle comes from the op's own extents (which `Ten.flat`
    copies out of the term's types), the addresses are explicit arithmetic,
    and the values are the committed folds.  No stage is mentioned: this is
    the statement the lowering is *checked against*, so it must not be
    produced by it. -/
noncomputable def TOp.den (op : TOp) (m : Buf → Nat → Float32) : Buf → Nat → Float32 :=
  match op with
  | .mv _ w x o b inW outW _ => fun b' a =>
      if b' = o ∧ a < b * outW then
        rowDot (m w) (m x)
          (fun t => (a % outW) * inW + t) (fun t => (a / outW) * inW + t) (inW / 32)
      else m b' a
  | .mvT _ w d o b inW outW => fun b' a =>
      if b' = o ∧ a < b * inW then
        bflyFold (dotStridedLane (m w) (m d)
          (fun i l => (i * 32 + l.val) * inW + a % inW)
          (fun i l => (a / inW) * outW + (i * 32 + l.val)) (outW / 32))
          ⟨0, by decide⟩
      else m b' a
  | .outer _ d x o b inW outW => fun b' a =>
      if b' = o ∧ a / inW < outW ∧ a % inW < inW then
        dotStridedLane (m d) (m x)
          (fun s _ => s * outW + a / inW) (fun s _ => s * inW + a % inW) b
          (laneMod a)
      else m b' a
  | .ew1 f i o g => fun b' a =>
      if b' = o ∧ a < g * 32 then denote (fun _ => m i a) f else m b' a
  | .ew2 f i j o g => fun b' a =>
      if b' = o ∧ a < g * 32 then
        denote (fun v : Fin 2 => if v.val = 0 then m i a else m j a) f
      else m b' a
  | .ew3 f i j k o g => fun b' a =>
      if b' = o ∧ a < g * 32 then
        denote (fun v : Fin 3 =>
          if v.val = 0 then m i a else if v.val = 1 then m j a else m k a) f
      else m b' a
  | .ew4 f i j k n o g => fun b' a =>
      if b' = o ∧ a < g * 32 then
        denote (fun v : Fin 4 =>
          if v.val = 0 then m i a else if v.val = 1 then m j a else
            if v.val = 2 then m k a else m n a) f
      else m b' a
  | .smce l bi oh o g => fun b' a =>
      if b' = o ∧ a < g * 32 then
        smSpec (m l) (m bi) (m oh)
          (fun ln => a / 32 * 32 + ln.val) (fun ln => ln.val) (laneMod a)
      else m b' a
  | .upd2 f i j g => fun b' a =>
      if b' = i ∧ a < g * 32 then
        denote (fun v : Fin 2 => if v.val = 0 then m i a else m j a) f
      else m b' a
  | .rowsq x o n rows => fun b' a =>
      if b' = o ∧ a < rows then
        rowDot (m x) (m x) (fun t => a * n + t) (fun t => a * n + t) (n / 32)
      else m b' a
  | .rowmax x o n rows init => fun b' a =>
      if b' = o ∧ a < rows then
        bflyFoldOp (fun p q => NumOps.max p q)
          (maxStridedLane (m x)
            (fun t l => (stride32 (.mul .ctaId (.lit n))).eval a t l) (n / 32) init)
          ⟨0, by decide⟩
      else m b' a
  | .rowdot i j o mA mB n rows => fun b' a =>
      if b' = o ∧ a < rows then
        bflyFold (dotStridedLane (m i) (m j)
          (fun t l => mA.ix.eval a t l) (fun t l => mB.ix.eval a t l) (n / 32))
          ⟨0, by decide⟩
      else m b' a
  | .rowdot4 i j k d o f mA mB mC mD n rows => fun b' a =>
      if b' = o ∧ a < rows then
        bflyFold (dotStridedLane4 (m i) (m j) (m k) (m d)
          (fun t l => mA.ix.eval a t l) (fun t l => mB.ix.eval a t l)
          (fun t l => mC.ix.eval a t l) (fun t l => mD.ix.eval a t l)
          (fun x y z => f.evalTriple x y z) (n / 32))
          ⟨0, by decide⟩
      else m b' a
  | .ziprow3 i j k o f mA mB mC n off w rows => fun b' a =>
      if b' = o ∧ a / n < rows ∧ off ≤ a % n ∧ a % n < off + w then
        f.evalTriple (m i (mA.ev (a / n) (a % n - off)))
          (m j (mB.ev (a / n) (a % n - off))) (m k (mC.ev (a / n) (a % n - off)))
      else m b' a
  | .ziprow4 i j k d o f mA mB mC mD n off w rows => fun b' a =>
      if b' = o ∧ a / n < rows ∧ off ≤ a % n ∧ a % n < off + w then
        f.evalQuad (m i (mA.ev (a / n) (a % n - off)))
          (m j (mB.ev (a / n) (a % n - off))) (m k (mC.ev (a / n) (a % n - off)))
          (m d (mD.ev (a / n) (a % n - off)))
      else m b' a
  | .ziprow i j o f mA mB n off w rows => fun b' a =>
      if b' = o ∧ a / n < rows ∧ off ≤ a % n ∧ a % n < off + w then
        f.evalPair (m i (mA.ev (a / n) (a % n - off))) (m j (mB.ev (a / n) (a % n - off)))
      else m b' a



end AlgorithmLib.ML
