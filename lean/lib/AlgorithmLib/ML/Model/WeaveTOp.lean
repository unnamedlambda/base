module
public import AlgorithmLib.ML.Model.WeaveBCast
meta import AlgorithmLib.ML.Model.WeaveBCast
public import AlgorithmLib.ML.Model.TenDenote
meta import AlgorithmLib.ML.Model.TenDenote
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # Shipped operations are broadcasted operations

  `Weave` proves laws about `BOp`, and `WeaveBCast` shows the addressing modes
  a kernel uses are reindexings.  Neither says anything about an operation a
  model actually emits, and a law that is never instantiated is a law about
  nothing.

  This module closes that: the denotation a shipped `TOp` carries — the one the
  lowering is checked against, in the committed fold orders — is exhibited as
  the denotation of a `BOp`.  The elementwise pass is the whole statement; the
  row pass is stated at its operand addresses, which is where its broadcasting
  lives.
-/

namespace AlgorithmLib.ML.Broadcast

open AlgorithmLib.ML

/-- An elementwise pass as a broadcasted operation: unit blocks, no
    reindexing, and the scalar expression as the target. -/
def ew1BOp (f : Expr 1) : BOp Float32 Float32 1 1 :=
  { front := fun p => p
    target := fun mm _ => denote (fun _ => mm 0) f }

/-- Reading an array through the identity reindexing at unit blocks is reading
    the array. -/
theorem pullFront_one_id (m : Arr α) : pullFront 1 (fun p => p) m = m := by
  funext addr
  show m (addr / 1 * 1 + addr % 1) = m addr
  rw [Nat.div_one, Nat.mul_one, Nat.mod_one, Nat.add_zero]

/-- **An emitted elementwise pass is a broadcasted operation.**

    At every address the pass owns, the value `TOp.den` commits to is the value
    the algebra's `BOp.den` computes. -/
theorem ew1_is_bop (f : Expr 1) (i o : Ref) (g : Nat)
    (m : Buf → Nat → Float32) (a : Nat) (ha : a < g * 32) :
    (TOp.ew1 f i o g).den m o a = (ew1BOp f).den (m i) a := by
  have hguard : (o = o ∧ a < g * 32) := ⟨rfl, ha⟩
  show (if o = o ∧ a < g * 32 then denote (fun _ => m i a) f else m o a) = _
  rw [if_pos hguard]
  show _ = (ew1BOp f).target
      (fun b => pullFront 1 (fun p => p) (m i) (a / 1 * 1 + b)) (a % 1)
  rw [pullFront_one_id]
  show _ = denote (fun _ => m i (a / 1 * 1 + 0)) f
  rw [Nat.div_one, Nat.mul_one, Nat.add_zero]

/-- An elementwise target reads one address, so it is confined to its block —
    the side condition every law about tiling carries, discharged for a real
    operation rather than assumed. -/
theorem ew1BOp_readsBelow (f : Expr 1) : ReadsBelow 1 (ew1BOp f).target := by
  intro m m' h
  funext _
  show denote (fun _ => m 0) f = denote (fun _ => m' 0) f
  rw [h 0 Nat.one_pos]

/-- **The composition law, at operations a model emits.**

    Two elementwise passes run in sequence are one broadcasted operation.  The
    hypotheses of `BOp.comp_den` are discharged here, so the law is applied and
    not merely stated — the failure mode this repository has hit before is a
    law that holds of nothing. -/
theorem ew1_comp_is_bop (f g : Expr 1) (x : Arr Float32) :
    ((ew1BOp f).comp (ew1BOp g)).den x = (ew1BOp g).den ((ew1BOp f).den x) :=
  BOp.comp_den (ew1BOp f) (ew1BOp g) (ew1BOp_readsBelow f) (ew1BOp_readsBelow g)
    Nat.one_pos Nat.one_pos x

/-- **A row pass reads its operands through reindexings.**

    `ziprow` decomposes an address into a row `a / n` and a column `a % n`,
    and reads each operand at that point through its `BCast`.  That address is
    the affine reindexing of `[row, column]`, read row-major in the operand's
    own shape — so the broadcasting a shipped row pass performs is the paper's
    reindexing morphism, at the addresses the kernel really touches. -/
theorem ziprow_operand_is_reindex (mode : BCast) (n off w rows a : Nat) :
    mode.ev (a / n) (a % n - off)
      = flatten [rows, bcastCols w mode]
          ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols w mode]) mode).apply
            [a / n, a % n - off]) :=
  (bcast_is_reindex mode w rows (a / n) (a % n - off)).symm

/-- The same statement for the two operands of a row pass, at the guard the
    operation owns — so nothing here is true only off the domain. -/
theorem ziprow_reads_reindexed (i j o : Ref) (f : WFExp) (mA mB : BCast)
    (n off w rows : Nat) (m : Buf → Nat → Float32) (a : Nat)
    (hrow : a / n < rows) (hlo : off ≤ a % n) (hhi : a % n < off + w) :
    (TOp.ziprow i j o f mA mB n off w rows).den m o a
      = f.evalPair
          (m i (flatten [rows, bcastCols w mA]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols w mA]) mA).apply
              [a / n, a % n - off])))
          (m j (flatten [rows, bcastCols w mB]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols w mB]) mB).apply
              [a / n, a % n - off]))) := by
  have hguard : (o = o ∧ a / n < rows ∧ off ≤ a % n ∧ a % n < off + w) :=
    ⟨rfl, hrow, hlo, hhi⟩
  show (if o = o ∧ a / n < rows ∧ off ≤ a % n ∧ a % n < off + w then
      f.evalPair (m i (mA.ev (a / n) (a % n - off))) (m j (mB.ev (a / n) (a % n - off)))
    else m o a) = _
  rw [if_pos hguard, ← ziprow_operand_is_reindex mA n off w rows a,
    ← ziprow_operand_is_reindex mB n off w rows a]

end AlgorithmLib.ML.Broadcast
