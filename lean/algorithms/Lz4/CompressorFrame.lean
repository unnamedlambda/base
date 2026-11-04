module
public import Lz4.Compressor
public import AlgorithmLib.LZ4.WarpLoops
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CompressorFrame` — what the model's run of a statement keeps

The model's run (`WStmt.eval`) leaves every register a statement never sets
(`eval_frame`) and the size of its table (`eval_smem_size`).
-/

open AlgorithmLib AlgorithmLib.LZ4WarpDSL

namespace Lz4Code

/-- The warp's run leaves every register it never sets. -/
theorem eval_frame (r : String) : ∀ (s : WStmt) (f : Nat) (ws : WState), r ∉ setsOf s →
    (s.eval f ws).regs r = ws.regs r := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro f ws h; simp only [setsOf, List.mem_append, not_or] at h
      simp only [WStmt.eval]; rw [ihb f _ h.2, iha f ws h.1]
  | uif c t e iht ihe =>
      intro f ws h; simp only [setsOf, List.mem_append, not_or] at h
      by_cases hc : (ws.regs c == 1) = true
      · simp only [WStmt.eval, hc, if_true]; exact iht f ws h.1
      · simp only [WStmt.eval, hc, Bool.false_eq_true, if_false]; exact ihe f ws h.2
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws _; simp [WStmt.eval]
      | succ f ihf =>
          intro ws h
          have h' : r ∉ setsOf b := fun hm => h (List.mem_cons_of_mem _ hm)
          by_cases hc : (ws.regs c == 1) = true
          · simp only [WStmt.eval, hc, if_true]; rw [ihf _ h, ih f ws h']
          · simp only [WStmt.eval, hc, Bool.false_eq_true, if_false]
  | bin o d a b => intro f ws h; simp only [setsOf, List.mem_singleton] at h; simp [WStmt.eval, WState.setReg, h]
  | setp c d a b => intro f ws h; simp only [setsOf, List.mem_singleton] at h; simp [WStmt.eval, WState.setReg, h]
  | mov d a => intro f ws h; simp only [setsOf, List.mem_singleton] at h; simp [WStmt.eval, WState.setReg, h]
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      intro f ws h
      simp only [setsOf, List.mem_cons, List.mem_nil_iff, or_false, not_or] at h
      rw [WStmt.eval.eq_11]
      exact evalCoopWindow_regs_frame _ _ _ _ _ _ _ _ _ _ h.1 h.2.1 h.2.2
  | coopExtendStep adv p0 cand0 ml inStride endCap =>
      intro f ws h
      simp only [setsOf, List.mem_singleton] at h
      rw [WStmt.eval.eq_12]; simp [evalCoopExtendStep, WState.setReg, h]
  | _ => intro f ws _; simp [WStmt.eval, WState.stgByte]

theorem tableInsert_size (sm g : Array UInt8) (hashLog tbl searchLim ib sp upto : Nat) :
    (tableInsert sm g hashLog tbl searchLim ib sp upto).size = sm.size := by
  unfold tableInsert
  have : ∀ (L : List Nat) (acc : Array UInt8), (L.foldl (fun acc k =>
      if k ≤ upto ∧ sp + k < searchLim then
        (acc.setIfInBounds (tbl + 2 * wHash g hashLog (ib + (sp + k))) (UInt8.ofNat ((sp + k + 1) % 256))).setIfInBounds
          (tbl + 2 * wHash g hashLog (ib + (sp + k)) + 1) (UInt8.ofNat ((sp + k + 1) / 256 % 256))
      else acc) acc).size = acc.size := by
    intro L
    induction L with
    | nil => intro acc; rfl
    | cons k L ih => intro acc; simp only [List.foldl_cons]; rw [ih]; split <;> simp
  exact this _ _

/-- The warp's run leaves the size of its table. -/
theorem eval_smem_size : ∀ (s : WStmt) (f : Nat) (ws : WState), (s.eval f ws).smem.size = ws.smem.size := by
  intro s
  induction s with
  | seq a b iha ihb => intro f ws; simp only [WStmt.eval]; rw [ihb, iha]
  | uif c t e iht ihe =>
      intro f ws
      by_cases hc : (ws.regs c == 1) = true
      · simp only [WStmt.eval, hc, if_true]; exact iht f ws
      · simp only [WStmt.eval, hc, Bool.false_eq_true, if_false]; exact ihe f ws
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws; simp [WStmt.eval]
      | succ f ihf =>
          intro ws
          by_cases hc : (ws.regs c == 1) = true
          · simp only [WStmt.eval, hc, if_true]; rw [ihf, ih]
          · simp only [WStmt.eval, hc, Bool.false_eq_true, if_false]
  | coopWindow =>
      intro f ws
      rw [WStmt.eval.eq_11, evalCoopWindow_eq_go, evalCoopWindowGo_smem]
      split <;> exact tableInsert_size _ _ _ _ _ _ _ _
  | coopExtendStep => intro f ws; rw [WStmt.eval.eq_12]; rfl
  | coopCopy => intro f ws; simp [WStmt.eval]
  | _ => intro f ws; simp [WStmt.eval, WState.setReg, WState.stgByte]

end Lz4Code
