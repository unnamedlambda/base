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
  w.regs = w'.regs ∧ w.smem = w'.smem ∧ (∀ b j, P b j → w.mem b j = w'.mem b j) ∧ w.int = w'.int

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
      (L.foldl (fun s l => s.store1 b (ix l) (v l)) w).smem = w.smem ∧
      (L.foldl (fun s l => s.store1 b (ix l) (v l)) w).int = w.int := by
  intro L
  induction L with
  | nil => intro w; exact ⟨rfl, rfl, rfl⟩
  | cons l ls ih => intro w; exact ih _

/-- A word no lane of block `cta` stores, the run leaves. -/
theorem run_keep (cta i : Nat) (s : EWStmt) (w : WSt R) (b : Buf) (j : Nat)
    (h : ∀ l : Lane, (b, j) ∉ touchL cta l i s) :
    ((s.run w cta i)).mem b j = w.mem b j := by
  refine run_frame_st cta (fun _ _ => 0) s i w b j (fun l hs => h l ?_)
  have := storesL_mem cta l s i b j (by rw [imOf_none]; exact hs)
  exact storeL_sub cta l s i this

/-- A relation every pair of steps keeps holds between two folds. -/
theorem foldl_rel {α β : Type} (f f' : β → α → β) (Rl : β → β → Prop)
    (h : ∀ s s' a, Rl s s' → Rl (f s a) (f' s' a)) :
    ∀ (L : List α) (s s' : β), Rl s s' → Rl (L.foldl f s) (L.foldl f' s') := by
  intro L
  induction L with
  | nil => intro s s' hs; exact hs
  | cons a L ih => intro s s' hs; exact ih _ _ (h s s' a hs)

/-- **A statement's run reads only the words it touches.** -/
theorem run_agree (cta : Nat) : ∀ (s : EWStmt) (i : Nat) (w w' : WSt R),
    Agree (fun b j => ∃ l : Lane, (b, j) ∈ touchL cta l i s) w w' →
    Agree (fun b j => ∃ l : Lane, (b, j) ∈ touchL cta l i s)
      ((s.run w cta i)) ((s.run w' cta i)) := by
  intro s
  induction s with
  | skip => intro i w w' h; exact h
  | setR r e =>
    intro i w w' ⟨hr, hs, hm, hi⟩
    refine ⟨?_, hs, hm, hi⟩
    show (w.setReg r _).regs = (w'.setReg r _).regs
    simp only [WSt.setReg, hr]
    funext x
    split
    · funext l; exact WFExp.eval_regs e w w' hr l
    · rfl
  | shflXor d s m =>
    intro i w w' ⟨hr, hs, hm, hi⟩
    refine ⟨?_, hs, hm, hi⟩
    show (w.setReg d _).regs = (w'.setReg d _).regs
    simp only [WSt.setReg, hr]
  | loadIdx d b ix =>
    intro i w w' ⟨hr, hs, hm, hi⟩
    refine ⟨?_, hs, hm, hi⟩
    show (w.setReg d _).regs = (w'.setReg d _).regs
    simp only [WSt.setReg, hr]
    funext x
    split
    · funext l; exact hm b _ ⟨l, by simp [touchL]⟩
    · rfl
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i w w' ⟨hr, hs, hm, hi⟩
    have hq : ∀ (q : Nat) (l : Lane), q < 4 → w.mem b (ix.eval cta i l + q) = w'.mem b (ix.eval cta i l + q) :=
      fun q l hq => hm b _ ⟨l, List.mem_map.mpr ⟨q, List.mem_range.mpr hq, rfl⟩⟩
    refine ⟨?_, hs, hm, hi⟩
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
    intro i w w' ⟨hr, hs, hm, hi⟩
    refine ⟨hr, hs, fun c j hc => ?_, hi⟩
    show (w.store1 b _ (w.regs r _)).mem c j = (w'.store1 b _ (w'.regs r _)).mem c j
    simp only [WSt.store1, hr]
    split
    · rfl
    · exact hm c j hc
  | storeLane b ix r =>
    intro i w w' ⟨hr, hs, hm, hi⟩
    have e1 := storeFold_regs b (fun l => ix.eval cta i l) (fun l => w.regs r l) (List.finRange W) w
    have e2 := storeFold_regs b (fun l => ix.eval cta i l) (fun l => w'.regs r l) (List.finRange W) w'
    refine ⟨?_, ?_, fun c j hc => ?_, ?_⟩
    · show (List.foldl _ w _).regs = (List.foldl _ w' _).regs
      rw [e1.1, e2.1, hr]
    · show (List.foldl _ w _).smem = (List.foldl _ w' _).smem
      rw [e1.2.1, e2.2.1, hs]
    · show (List.foldl _ w _).mem c j = (List.foldl _ w' _).mem c j
      have hv : (fun l => w.regs r l) = (fun l => w'.regs r l) := by rw [hr]
      show (List.foldl (fun s l => s.store1 b (ix.eval cta i l) (w.regs r l)) w (List.finRange W)).mem c j
        = (List.foldl (fun s l => s.store1 b (ix.eval cta i l) (w'.regs r l)) w' (List.finRange W)).mem c j
      rw [show (fun (s : WSt R) (l : Lane) => s.store1 b (ix.eval cta i l) (w.regs r l))
          = (fun s l => s.store1 b (ix.eval cta i l) (w'.regs r l)) from by rw [hr]]
      exact storeFold_agree b _ _ _ w w' c j (hm c j hc)
    · show (List.foldl _ w _).int = (List.foldl _ w' _).int
      rw [e1.2.2, e2.2.2, hi]
  | stSm ix r =>
    intro i w w' ⟨hr, hs, hm, hi⟩
    refine ⟨hr, ?_, hm, hi⟩
    show (List.foldl _ w.smem _) = (List.foldl _ w'.smem _)
    simp only [hr, hs]
  | ldSm d ix =>
    intro i w w' ⟨hr, hs, hm, hi⟩
    refine ⟨?_, hs, hm, hi⟩
    show (w.setReg d _).regs = (w'.setReg d _).regs
    simp only [WSt.setReg, hr, hs]
  | barrier => intro i w w' h; exact h
  | cvtIF d ix =>
    intro i w w' ⟨hr, hs, hm, hi⟩
    refine ⟨?_, hs, hm, hi⟩
    show (w.setReg d _).regs = (w'.setReg d _).regs
    simp only [WSt.setReg, hr]
  | seq x y ihx ihy =>
    intro i w w' h
    obtain ⟨hr, hs, hm, hi⟩ := h
    have hx := ihx i w w' ⟨hr, hs, fun b j ⟨l, hl⟩ => hm b j ⟨l, List.mem_append_left _ hl⟩, hi⟩
    have hy := ihy i ((x.run w cta i)) ((x.run w' cta i)) ⟨hx.1, hx.2.1, fun b j ⟨l, hl⟩ => by
      by_cases hxt : ∃ l : Lane, (b, j) ∈ touchL cta l i x
      · exact hx.2.2.1 b j hxt
      · have hn : ∀ l : Lane, (b, j) ∉ touchL cta l i x := fun l h => hxt ⟨l, h⟩
        rw [run_keep cta i x w b j hn, run_keep cta i x w' b j hn]
        exact hm b j ⟨l, List.mem_append_right _ hl⟩, hx.2.2.2⟩
    refine ⟨hy.1, hy.2.1, fun b j ⟨l, hl⟩ => ?_, hy.2.2.2⟩
    show ((y.run ((x.run w cta i)) cta i)).mem b j
      = ((y.run ((x.run w' cta i)) cta i)).mem b j
    by_cases hyt : ∃ l : Lane, (b, j) ∈ touchL cta l i y
    · exact hy.2.2.1 b j hyt
    · have hn : ∀ l : Lane, (b, j) ∉ touchL cta l i y := fun l h => hyt ⟨l, h⟩
      rw [run_keep cta i y _ b j hn, run_keep cta i y _ b j hn]
      rcases List.mem_append.mp hl with h | h
      · exact hx.2.2.1 b j ⟨l, h⟩
      · exact absurd ⟨l, h⟩ hyt
  | forN n body ih =>
    intro i w w' h
    have key : ∀ (L : List Nat) (w w' : WSt R),
        Agree (fun b j => ∃ l : Lane, ∃ t ∈ L, (b, j) ∈ touchL cta l t body) w w' →
        Agree (fun b j => ∃ l : Lane, ∃ t ∈ L, (b, j) ∈ touchL cta l t body)
          (L.foldl (fun s t => (body.run s cta t)) w)
          (L.foldl (fun s t => (body.run s cta t)) w') := by
      intro L
      induction L with
      | nil => intro w w' h; exact h
      | cons t ts ihL =>
        intro w w' ⟨hr, hs, hm, hi⟩
        have ht := ih t w w' ⟨hr, hs, fun b j ⟨l, hl⟩ => hm b j ⟨l, t, List.mem_cons_self, hl⟩, hi⟩
        have hrest := ihL ((body.run w cta t)) ((body.run w' cta t)) ⟨ht.1, ht.2.1,
          fun b j ⟨l, t', ht', hl⟩ => by
            by_cases htt : ∃ l : Lane, (b, j) ∈ touchL cta l t body
            · exact ht.2.2.1 b j htt
            · have hn : ∀ l : Lane, (b, j) ∉ touchL cta l t body := fun l h => htt ⟨l, h⟩
              rw [run_keep cta t body w b j hn, run_keep cta t body w' b j hn]
              exact hm b j ⟨l, t', List.mem_cons_of_mem _ ht', hl⟩, ht.2.2.2⟩
        refine ⟨hrest.1, hrest.2.1, fun b j ⟨l, t', ht', hl⟩ => ?_, hrest.2.2.2⟩
        simp only [List.foldl_cons]
        rcases List.mem_cons.mp ht' with rfl | ht'
        · by_cases hrt : ∃ l : Lane, ∃ t'' ∈ ts, (b, j) ∈ touchL cta l t'' body
          · exact hrest.2.2.1 b j hrt
          · have keepL : ∀ (M : List Nat) (v : WSt R), (∀ t'' ∈ M, ∀ l : Lane, (b, j) ∉ touchL cta l t'' body) →
                (M.foldl (fun s t => (body.run s cta t)) v).mem b j = v.mem b j := by
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
            exact ht.2.2.1 b j ⟨l, hl⟩
        · exact hrest.2.2.1 b j ⟨l, t', ht', hl⟩
    obtain ⟨hr, hs, hm, hi⟩ := h
    have := key (List.range n) w w' ⟨hr, hs, fun b j ⟨l, t, ht, hl⟩ =>
      hm b j ⟨l, List.mem_flatMap.mpr ⟨t, ht, hl⟩⟩, hi⟩
    refine ⟨this.1, this.2.1, fun b j ⟨l, hl⟩ => ?_, this.2.2.2⟩
    obtain ⟨t, ht, hl⟩ := List.mem_flatMap.mp hl
    exact this.2.2.1 b j ⟨l, t, ht, hl⟩
  | forM bb a body _ => intro i w w' h; exact h
  | iif c x y ihx ihy =>
    intro i w w' h
    have hP : ∀ (z : EWStmt), (∀ b j, (∃ l : Lane, (b, j) ∈ touchL cta l i z) →
        ∃ l : Lane, (b, j) ∈ touchL cta l i (.iif c x y)) →
        Agree (fun b j => ∃ l : Lane, (b, j) ∈ touchL cta l i z) ((z.run w cta i)) ((z.run w' cta i)) →
        Agree (fun b j => ∃ l : Lane, (b, j) ∈ touchL cta l i (.iif c x y)) ((z.run w cta i)) ((z.run w' cta i)) := by
      intro z hz hza
      refine ⟨hza.1, hza.2.1, fun b j hb => ?_, hza.2.2.2⟩
      by_cases hzt : ∃ l : Lane, (b, j) ∈ touchL cta l i z
      · exact hza.2.2.1 b j hzt
      · have hn : ∀ l : Lane, (b, j) ∉ touchL cta l i z := fun l h => hzt ⟨l, h⟩
        rw [run_keep cta i z w b j hn, run_keep cta i z w' b j hn]
        exact h.2.2.1 b j hb
    simp only [EWStmt.run, h.2.2.2]
    split
    · exact hP x (fun b j ⟨l, hl⟩ => ⟨l, List.mem_append_left _ hl⟩)
        (ihx i w w' ⟨h.1, h.2.1, fun b j ⟨l, hl⟩ => h.2.2.1 b j ⟨l, List.mem_append_left _ hl⟩, h.2.2.2⟩)
    · exact hP y (fun b j ⟨l, hl⟩ => ⟨l, List.mem_append_right _ hl⟩)
        (ihy i w w' ⟨h.1, h.2.1, fun b j ⟨l, hl⟩ => h.2.2.1 b j ⟨l, List.mem_append_right _ hl⟩, h.2.2.2⟩)
  | iwhile n c cs body ih =>
    intro i w w' h
    simp only [EWStmt.run]
    refine foldl_rel _ _ (Agree (fun b j => ∃ l : Lane, (b, j) ∈ touchL cta l i (.iwhile n c cs body)))
      (fun s s' _ hss => ?_) _ w w' h
    rw [hss.2.2.2]
    split
    · have hb := ih i s s' hss
      exact ⟨hb.1, hb.2.1, hb.2.2.1, by simp only [WSt.withInt, h.2.2.2, hb.2.2.2]⟩
    · exact hss
  | locals rs body ih =>
    intro i w w' h
    have hb := ih i w w' h
    exact ⟨hb.1, hb.2.1, hb.2.2.1, by simp only [EWStmt.run, WSt.withInt, h.2.2.2, hb.2.2.2]⟩
  | _ =>
    intro i w w' h
    refine ⟨h.1, h.2.1, h.2.2.1, ?_⟩
    simp only [EWStmt.run, WSt.withInt, h.2.2.2]

end Carrier

end AlgorithmLib.ML.BodyGroup
