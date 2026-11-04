module
public import AlgorithmLib.ML.Model.Reorder
meta import AlgorithmLib.ML.Model.Reorder
public import AlgorithmLib.ML.Kernel.Rewrite
meta import AlgorithmLib.ML.Kernel.Rewrite
public import AlgorithmLib.ML.Num.Bits
meta import AlgorithmLib.ML.Num.Bits
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# A tape whose contractions are the vendor's

A contraction the tape hands a vendor (`Backend.vendor`) is a GEMM: each
output element the vendor's sum of its own products (`cublasSgemmResult`),
over the operands as the tape lays them out (`TOp.gemmShape`). `TOp.denV`
reads the tape that way at `F32` and every other operation as its
denotation; it keeps the frame (`denV_frame`), so the schedule computes it
as it computes the tape (`pull_tape_with`).
-/

namespace AlgorithmLib.ML

/-- A word as a single. -/
def F32.toFloat (x : F32) : Float32 := Float32.ofBits x.bits.toUInt32

/-- A single as a word. -/
def F32.ofFloat (x : Float32) : F32 := ⟨x.toBits.toUInt64⟩

/-- **A vendor contraction as a GEMM**: `C = A · B` with `C` `m × n`, its
    element `(r, c)` at `r + c · m`, `A` `m × k` and `B` `k × n` read where
    the tape lays them (`A (r, t)` at `ai r t` of buffer `A`, `B (t, c)` at
    `bi t c` of buffer `B`). -/
def TOp.gemmShape : TOp → Option (Nat × Nat × Nat × Buf × (Nat → Nat → Nat) × Buf × (Nat → Nat → Nat) × Buf)
  | .mv (.vendor _) w x o b i ow _ => some (ow, b, i, w, fun r t => r * i + t, x, fun t c => c * i + t, o)
  | .mvT (.vendor _) w d o b i ow => some (i, b, ow, w, fun r s => s * i + r, d, fun s c => c * ow + s, o)
  | .outer (.vendor _) d x o b i ow => some (i, ow, b, x, fun r s => s * i + r, d, fun s c => s * ow + c, o)
  | _ => none

theorem gemmShape_out {op : TOp} {m n k : Nat} {A B C : Buf} {ai bi : Nat → Nat → Nat}
    (h : op.gemmShape = some (m, n, k, A, ai, B, bi, C)) (batch : Nat) :
    C = (TOp.outSize batch op).1 ∧ A ∈ op.reads ∧ B ∈ op.reads := by
  cases op with
  | mv bk w x o b i ow bA =>
      cases bk with
      | proven => simp [TOp.gemmShape] at h
      | vendor v =>
          simp only [TOp.gemmShape, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨-, -, -, rfl, -, rfl, -, rfl⟩ := h
          simp [TOp.outSize, TOp.reads]
  | mvT bk w d o b i ow =>
      cases bk with
      | proven => simp [TOp.gemmShape] at h
      | vendor v =>
          simp only [TOp.gemmShape, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨-, -, -, rfl, -, rfl, -, rfl⟩ := h
          simp [TOp.outSize, TOp.reads]
  | outer bk d x o b i ow =>
      cases bk with
      | proven => simp [TOp.gemmShape] at h
      | vendor v =>
          simp only [TOp.gemmShape, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨-, -, -, rfl, -, rfl, -, rfl⟩ := h
          simp [TOp.outSize, TOp.reads]
  | _ => simp [TOp.gemmShape] at h

/-- **The tape at `F32`, its vendor contractions the vendor's**: the output
    element `(r, c)` of a contraction the vendor's sum of its products,
    every other word and every other operation as the denotations have it,
    the hardware's approximations `hw`'s. -/
noncomputable def TOp.denV (hw : Kernel.Hw) (op : TOp) (mem : Buf → Nat → F32) : Buf → Nat → F32 :=
  match op.gemmShape with
  | some (m, n, k, A, ai, B, bi, C) => fun b' a =>
      if h : b' = C ∧ a < m * n then
        have hm : 0 < m := Nat.pos_of_ne_zero fun e => by subst e; simp at h
        F32.ofFloat (cublasSgemmResult m n k (fun r t => (mem A (ai r t)).toFloat)
          (fun t c => (mem B (bi t c)).toFloat) ⟨a % m, Nat.mod_lt _ hm⟩
          ⟨a / m, (Nat.div_lt_iff_lt_mul hm).mpr (by rw [Nat.mul_comm]; exact h.2)⟩)
      else mem b' a
  | none => @TOp.den F32 (F32.ops hw) _ op mem

/-- **It keeps the frame**: a contraction writes only its output and reads
    only its operands, every other operation as its denotation does. -/
theorem denV_frame (hw : Kernel.Hw) (batch : Nat) : TOp.Frame (R := F32) batch (TOp.denV hw) := by
  refine ⟨fun op mem b hb => ?_, fun op mem mem' hr b a hb => ?_⟩
  · cases hg : op.gemmShape with
    | none =>
        simp only [TOp.denV, hg]
        exact @den_out F32 (F32.ops hw) _ batch op mem b hb
    | some t =>
        obtain ⟨m, n, k, A, ai, B, bi, C⟩ := t
        simp only [TOp.denV, hg]
        funext a
        rw [dif_neg (fun h => hb (h.1.trans (gemmShape_out hg batch).1))]
  · cases hg : op.gemmShape with
    | none =>
        simp only [TOp.denV, hg]
        exact @den_congr F32 (F32.ops hw) _ op mem mem' hr b a hb
    | some t =>
        obtain ⟨m, n, k, A, ai, B, bi, C⟩ := t
        obtain ⟨-, hA, hB⟩ := gemmShape_out hg batch
        simp only [TOp.denV, hg]
        split
        · simp only [hr A hA, hr B hB]
        · exact hb

end AlgorithmLib.ML
