module
public import AlgorithmLib.ML.Kernel.BodyCheck
meta import AlgorithmLib.ML.Kernel.BodyCheck
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Kernel.BodyGroup` — what a statement's run reads

A static statement's run at block `cta`, from two states with the same
registers and shared memory, agreeing on every word the statement touches
there (`touchL`), ends in states that still agree: registers, shared memory
and every touched word (`run_agree`). So what a run leaves in memory depends
only on the words it touches.
-/

open AlgorithmLib AlgorithmLib.ML AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck AlgorithmLib.ML.BodyPhase

namespace AlgorithmLib.ML.BodyGroup

section Carrier
variable {R : Type} [NumOps R] [NumLit R]

/-- Two states alike in registers and shared memory, and on the words `P`
    names. -/
def Agree (P : Buf → Nat → Prop) (w w' : WSt R) : Prop :=
  w.regs = w'.regs ∧ w.smem = w'.smem ∧ ∀ b j, P b j → w.mem b j = w'.mem b j

theorem WFExp.eval_regs (e : WFExp) (w w' : WSt R) (h : w.regs = w'.regs) (l : Lane) :
    e.eval w l = e.eval w' l := by
  induction e with
  | reg r => simp only [WFExp.eval, h]
  | lit v => rfl
  | add a b iha ihb => simp only [WFExp.eval, iha, ihb]
  | mul a b iha ihb => simp only [WFExp.eval, iha, ihb]
  | neg a ih => simp only [WFExp.eval, ih]
  | inv a ih => simp only [WFExp.eval, ih]
  | exp a ih => simp only [WFExp.eval, ih]
  | ex2 a ih => simp only [WFExp.eval, ih]
  | rsqrt a ih => simp only [WFExp.eval, ih]
  | maxW a b iha ihb => simp only [WFExp.eval, iha, ihb]
  | geF a b iha ihb => simp only [WFExp.eval, iha, ihb]

omit [NumOps R] [NumLit R] in
/-- Stores from equal registers keep any word two memories agree on. -/
theorem storeFold_agree (b : Buf) (ix : Lane → Nat) (v : Lane → R) :
    ∀ (L : List Lane) (w w' : WSt R) (c : Buf) (j : Nat), w.mem c j = w'.mem c j →
      (L.foldl (fun s l => s.store1 b (ix l) (v l)) w).mem c j
        = (L.foldl (fun s l => s.store1 b (ix l) (v l)) w').mem c j := by
  intro L
  induction L with
  | nil => intro w w' c j h; exact h
  | cons l ls ih =>
    intro w w' c j h
    simp only [List.foldl_cons]
    refine ih _ _ c j ?_
    simp only [WSt.store1]
    split <;> simp_all

omit [NumOps R] [NumLit R] in
theorem storeFold_regs (b : Buf) (ix : Lane → Nat) (v : Lane → R) :
    ∀ (L : List Lane) (w : WSt R),
      (L.foldl (fun s l => s.store1 b (ix l) (v l)) w).regs = w.regs ∧
      (L.foldl (fun s l => s.store1 b (ix l) (v l)) w).smem = w.smem := by
  intro L
  induction L with
  | nil => intro w; exact ⟨rfl, rfl⟩
  | cons l ls ih => intro w; exact ih _

/-- A word no lane of block `cta` stores, the run leaves. -/
theorem run_keep (cta i : Nat) (s : EWStmt) (w : WSt R) (b : Buf) (j : Nat)
    (h : ∀ l : Lane, (b, j) ∉ touchL cta l i s) :
    ((s.elabAt cta i).run w).mem b j = w.mem b j := by
  refine run_frame_st cta (fun _ _ => 0) s i w b j (fun l hs => h l ?_)
  have := storesL_mem cta l s i b j (by rw [imOf_none]; exact hs)
  exact storeL_sub cta l s i this

/-- **A statement's run reads only the words it touches.** -/
theorem run_agree (cta : Nat) : ∀ (s : EWStmt) (i : Nat) (w w' : WSt R),
    Agree (fun b j => ∃ l : Lane, (b, j) ∈ touchL cta l i s) w w' →
    Agree (fun b j => ∃ l : Lane, (b, j) ∈ touchL cta l i s)
      ((s.elabAt cta i).run w) ((s.elabAt cta i).run w') := by
  intro s
  induction s with
  | skip => intro i w w' h; exact h
  | setR r e =>
    intro i w w' ⟨hr, hs, hm⟩
    refine ⟨?_, hs, hm⟩
    show (w.setReg r _).regs = (w'.setReg r _).regs
    simp only [WSt.setReg, hr]
    funext x
    split
    · funext l; exact WFExp.eval_regs e w w' hr l
    · rfl
  | shflXor d s m =>
    intro i w w' ⟨hr, hs, hm⟩
    refine ⟨?_, hs, hm⟩
    show (w.setReg d _).regs = (w'.setReg d _).regs
    simp only [WSt.setReg, hr]
  | loadIdx d b ix =>
    intro i w w' ⟨hr, hs, hm⟩
    refine ⟨?_, hs, hm⟩
    show (w.setReg d _).regs = (w'.setReg d _).regs
    simp only [WSt.setReg, hr]
    funext x
    split
    · funext l; exact hm b _ ⟨l, by simp [touchL]⟩
    · rfl
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i w w' ⟨hr, hs, hm⟩
    have hq : ∀ (q : Nat) (l : Lane), q < 4 → w.mem b (ix.eval cta i l + q) = w'.mem b (ix.eval cta i l + q) :=
      fun q l hq => hm b _ ⟨l, List.mem_map.mpr ⟨q, List.mem_range.mpr hq, rfl⟩⟩
    refine ⟨?_, hs, hm⟩
    show ((((w.setReg d0 _).setReg d1 _).setReg d2 _).setReg d3 _).regs
      = ((((w'.setReg d0 _).setReg d1 _).setReg d2 _).setReg d3 _).regs
    have h0 : (fun l => w.mem b (ix.eval cta i l)) = (fun l => w'.mem b (ix.eval cta i l)) :=
      funext fun l => by simpa using hq 0 l (by decide)
    have h1 : (fun l => w.mem b (ix.eval cta i l + 1)) = (fun l => w'.mem b (ix.eval cta i l + 1)) :=
      funext fun l => hq 1 l (by decide)
    have h2 : (fun l => w.mem b (ix.eval cta i l + 2)) = (fun l => w'.mem b (ix.eval cta i l + 2)) :=
      funext fun l => hq 2 l (by decide)
    have h3 : (fun l => w.mem b (ix.eval cta i l + 3)) = (fun l => w'.mem b (ix.eval cta i l + 3)) :=
      funext fun l => hq 3 l (by decide)
    simp only [WSt.setReg, hr, h0, h1, h2, h3]
  | storeLane0 b ix r =>
    intro i w w' ⟨hr, hs, hm⟩
    refine ⟨hr, hs, fun c j hc => ?_⟩
    show (w.store1 b _ (w.regs r _)).mem c j = (w'.store1 b _ (w'.regs r _)).mem c j
    simp only [WSt.store1, hr]
    split
    · rfl
    · exact hm c j hc
  | storeLane b ix r =>
    intro i w w' ⟨hr, hs, hm⟩
    have e1 := storeFold_regs b (fun l => ix.eval cta i l) (fun l => w.regs r l) (List.finRange W) w
    have e2 := storeFold_regs b (fun l => ix.eval cta i l) (fun l => w'.regs r l) (List.finRange W) w'
    refine ⟨?_, ?_, fun c j hc => ?_⟩
    · show (List.foldl _ w _).regs = (List.foldl _ w' _).regs
      rw [e1.1, e2.1, hr]
    · show (List.foldl _ w _).smem = (List.foldl _ w' _).smem
      rw [e1.2, e2.2, hs]
    · show (List.foldl _ w _).mem c j = (List.foldl _ w' _).mem c j
      have hv : (fun l => w.regs r l) = (fun l => w'.regs r l) := by rw [hr]
      show (List.foldl (fun s l => s.store1 b (ix.eval cta i l) (w.regs r l)) w (List.finRange W)).mem c j
        = (List.foldl (fun s l => s.store1 b (ix.eval cta i l) (w'.regs r l)) w' (List.finRange W)).mem c j
      rw [show (fun (s : WSt R) (l : Lane) => s.store1 b (ix.eval cta i l) (w.regs r l))
          = (fun s l => s.store1 b (ix.eval cta i l) (w'.regs r l)) from by rw [hr]]
      exact storeFold_agree b _ _ _ w w' c j (hm c j hc)
  | stSm ix r =>
    intro i w w' ⟨hr, hs, hm⟩
    refine ⟨hr, ?_, hm⟩
    show (List.foldl _ w.smem _) = (List.foldl _ w'.smem _)
    simp only [hr, hs]
  | ldSm d ix =>
    intro i w w' ⟨hr, hs, hm⟩
    refine ⟨?_, hs, hm⟩
    show (w.setReg d _).regs = (w'.setReg d _).regs
    simp only [WSt.setReg, hr, hs]
  | barrier => intro i w w' h; exact h
  | cvtIF d ix =>
    intro i w w' ⟨hr, hs, hm⟩
    refine ⟨?_, hs, hm⟩
    show (w.setReg d _).regs = (w'.setReg d _).regs
    simp only [WSt.setReg, hr]
  | seq x y ihx ihy =>
    intro i w w' h
    obtain ⟨hr, hs, hm⟩ := h
    have hx := ihx i w w' ⟨hr, hs, fun b j ⟨l, hl⟩ => hm b j ⟨l, List.mem_append_left _ hl⟩⟩
    have hy := ihy i ((x.elabAt cta i).run w) ((x.elabAt cta i).run w') ⟨hx.1, hx.2.1, fun b j ⟨l, hl⟩ => by
      by_cases hxt : ∃ l : Lane, (b, j) ∈ touchL cta l i x
      · exact hx.2.2 b j hxt
      · have hn : ∀ l : Lane, (b, j) ∉ touchL cta l i x := fun l h => hxt ⟨l, h⟩
        rw [run_keep cta i x w b j hn, run_keep cta i x w' b j hn]
        exact hm b j ⟨l, List.mem_append_right _ hl⟩⟩
    refine ⟨hy.1, hy.2.1, fun b j ⟨l, hl⟩ => ?_⟩
    show ((y.elabAt cta i).run ((x.elabAt cta i).run w)).mem b j
      = ((y.elabAt cta i).run ((x.elabAt cta i).run w')).mem b j
    by_cases hyt : ∃ l : Lane, (b, j) ∈ touchL cta l i y
    · exact hy.2.2 b j hyt
    · have hn : ∀ l : Lane, (b, j) ∉ touchL cta l i y := fun l h => hyt ⟨l, h⟩
      rw [run_keep cta i y _ b j hn, run_keep cta i y _ b j hn]
      rcases List.mem_append.mp hl with h | h
      · exact hx.2.2 b j ⟨l, h⟩
      · exact absurd ⟨l, h⟩ hyt
  | forN n body ih =>
    intro i w w' h
    have key : ∀ (L : List Nat) (w w' : WSt R),
        Agree (fun b j => ∃ l : Lane, ∃ t ∈ L, (b, j) ∈ touchL cta l t body) w w' →
        Agree (fun b j => ∃ l : Lane, ∃ t ∈ L, (b, j) ∈ touchL cta l t body)
          (L.foldl (fun s t => (body.elabAt cta t).run s) w)
          (L.foldl (fun s t => (body.elabAt cta t).run s) w') := by
      intro L
      induction L with
      | nil => intro w w' h; exact h
      | cons t ts ihL =>
        intro w w' ⟨hr, hs, hm⟩
        have ht := ih t w w' ⟨hr, hs, fun b j ⟨l, hl⟩ => hm b j ⟨l, t, List.mem_cons_self, hl⟩⟩
        have hrest := ihL ((body.elabAt cta t).run w) ((body.elabAt cta t).run w') ⟨ht.1, ht.2.1,
          fun b j ⟨l, t', ht', hl⟩ => by
            by_cases htt : ∃ l : Lane, (b, j) ∈ touchL cta l t body
            · exact ht.2.2 b j htt
            · have hn : ∀ l : Lane, (b, j) ∉ touchL cta l t body := fun l h => htt ⟨l, h⟩
              rw [run_keep cta t body w b j hn, run_keep cta t body w' b j hn]
              exact hm b j ⟨l, t', List.mem_cons_of_mem _ ht', hl⟩⟩
        refine ⟨hrest.1, hrest.2.1, fun b j ⟨l, t', ht', hl⟩ => ?_⟩
        simp only [List.foldl_cons]
        rcases List.mem_cons.mp ht' with rfl | ht'
        · by_cases hrt : ∃ l : Lane, ∃ t'' ∈ ts, (b, j) ∈ touchL cta l t'' body
          · exact hrest.2.2 b j hrt
          · have keepL : ∀ (M : List Nat) (v : WSt R), (∀ t'' ∈ M, ∀ l : Lane, (b, j) ∉ touchL cta l t'' body) →
                (M.foldl (fun s t => (body.elabAt cta t).run s) v).mem b j = v.mem b j := by
              intro M
              induction M with
              | nil => intro v _; rfl
              | cons t'' ms ihM =>
                intro v hv
                simp only [List.foldl_cons]
                rw [ihM _ (fun t3 h3 => hv t3 (List.mem_cons_of_mem _ h3)),
                  run_keep cta t'' body v b j (hv t'' List.mem_cons_self)]
            have hn : ∀ t'' ∈ ts, ∀ l : Lane, (b, j) ∉ touchL cta l t'' body :=
              fun t'' h1 l h2 => hrt ⟨l, t'', h1, h2⟩
            rw [keepL ts _ hn, keepL ts _ hn]
            exact ht.2.2 b j ⟨l, hl⟩
        · exact hrest.2.2 b j ⟨l, t', ht', hl⟩
    obtain ⟨hr, hs, hm⟩ := h
    have := key (List.range n) w w' ⟨hr, hs, fun b j ⟨l, t, ht, hl⟩ =>
      hm b j ⟨l, List.mem_flatMap.mpr ⟨t, ht, hl⟩⟩⟩
    refine ⟨this.1, this.2.1, fun b j ⟨l, hl⟩ => ?_⟩
    obtain ⟨t, ht, hl⟩ := List.mem_flatMap.mp hl
    exact this.2.2 b j ⟨l, t, ht, hl⟩
  | forM bb a body _ => intro i w w' h; exact h

end Carrier

end AlgorithmLib.ML.BodyGroup
