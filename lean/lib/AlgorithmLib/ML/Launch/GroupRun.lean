module
public import AlgorithmLib.ML.Kernel.BodyGroup
meta import AlgorithmLib.ML.Kernel.BodyGroup
public import AlgorithmLib.ML.Launch.Sequence
meta import AlgorithmLib.ML.Launch.Sequence
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Launch.GroupRun` — a group's blocks perform its members' pipeline

A group of stages over one grid, each block running the members' statements
in order, lands at every word a block touches what the members' pipeline
(`Pipeline.denote`) lands there — when no block stores a word another block
touches (`group_run`). Nothing is asked of the order blocks run in: each
block starts from the launch's memory, as a GPU's do.

The two facts it rests on: a stage's value reads only the words its
statement touches, and its own output word (`val_reads`); and a word a
block's statement does not store, its run leaves (`run_keepS`).
-/

open AlgorithmLib AlgorithmLib.ML AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck AlgorithmLib.ML.BodyPhase
open AlgorithmLib.ML.BodyGroup

namespace AlgorithmLib.ML.GroupRun

section Carrier
variable {R : Type} [NumOps R] [NumLit R]

/-- A word no lane of block `cta` stores, the run leaves. -/
theorem run_keepS (cta i : Nat) (s : EWStmt) (w : WSt R) (b : Buf) (j : Nat)
    (h : ∀ l : Lane, (b, j) ∉ storeL cta l i s) :
    ((s.elabAt cta i).run w).mem b j = w.mem b j :=
  run_frame_st cta (fun _ _ => 0) s i w b j (fun l hs => h l (storesL_mem cta l s i b j (by rw [imOf_none]; exact hs)))

/-- A stage that elaborates its statement against no integer state. -/
def Plain (S : StageSpec R) : Prop := S.iregs = (fun _ _ => 0) ∧ S.imem = (fun _ _ => 0)

theorem Plain.blk {S : StageSpec R} (h : Plain S) (cta : Nat) : S.blk cta = S.ew.elabAt cta 0 := by
  simp only [StageSpec.blk, h.1, h.2]

/-- A state with memory `m` and the registers and shared memory of `w`. -/
def withMem (w : WSt R) (m : Buf → Nat → R) : WSt R := { w with mem := m }

/-- **A stage's value reads only what its statement touches**, and its own
    output word. -/
theorem val_reads (S : StageSpec R) (hp : Plain S) (w : WSt R) (m m' : Buf → Nat → R) (cta a : Nat)
    (hd : S.dom cta a)
    (hag : ∀ b j, ((∃ l : Lane, (b, j) ∈ touchL cta l 0 S.ew) ∨ (b = S.out ∧ j = a)) → m b j = m' b j) :
    S.val m cta a = S.val m' cta a := by
  have v1 := S.value cta (withMem w m) a hd
  have v2 := S.value cta (withMem w m') a hd
  have e : S.ew.elabAt cta 0 S.iregs S.imem = S.ew.elabAt cta 0 := by simp only [hp.1, hp.2]
  rw [e] at v1 v2
  show S.val (withMem w m).mem cta a = S.val (withMem w m').mem cta a
  rw [← v1, ← v2]
  by_cases ht : ∃ l : Lane, (S.out, a) ∈ touchL cta l 0 S.ew
  · exact (run_agree cta S.ew 0 (withMem w m) (withMem w m')
      ⟨rfl, rfl, fun b j h => hag b j (Or.inl h)⟩).2.2 _ _ ht
  · have hn : ∀ l : Lane, (S.out, a) ∉ touchL cta l 0 S.ew := fun l h => ht ⟨l, h⟩
    rw [run_keep cta 0 S.ew _ _ _ hn, run_keep cta 0 S.ew _ _ _ hn]
    exact hag _ _ (Or.inr ⟨rfl, rfl⟩)

/-- **A group's blocks perform its members' pipeline.**  Stages `ss`, each
    plain and exclusive over the grid `G`; `T c` any set of words containing
    every word block `c` of any member touches; no block storing a word in
    another block's `T`. Then block `c`, running the members in order from a
    state agreeing with `M` on `T c`, leaves every word of `T c` as the
    pipeline does from `M`. -/
theorem group_run (G : Nat) (T : Nat → Buf → Nat → Prop) :
    ∀ (ss : List (StageSpec R)),
      (∀ S ∈ ss, Plain S ∧ S.Exclusive ∧ S.grid = G) →
      (∀ S ∈ ss, ∀ c (l : Lane) b j, (b, j) ∈ touchL c l 0 S.ew → T c b j) →
      (∀ S ∈ ss, ∀ c c', c < G → c' < G → c ≠ c' → ∀ (l : Lane) b j, (b, j) ∈ storeL c' l 0 S.ew → ¬ T c b j) →
      ∀ (w : Nat → WSt R) (M : Buf → Nat → R), (∀ c, c < G → ∀ b j, T c b j → (w c).mem b j = M b j) →
      ∀ c, c < G → ∀ b j, T c b j →
        (ss.foldl (fun st S => (S.blk c).run st) (w c)).mem b j = (Pipeline.mk ss).denote M b j := by
  intro ss
  induction ss with
  | nil => intro _ _ _ w M hw c hc b j hT; exact hw c hc b j hT
  | cons S rest ih =>
    intro hS hTS hAp w M hw
    have hS0 := hS S List.mem_cons_self
    obtain ⟨hp, hex, hg⟩ := hS0
    let w' : Nat → WSt R := fun c => (S.blk c).run (w c)
    have hw' : ∀ c, c < G → ∀ b j, T c b j → (w' c).mem b j = S.step M b j := by
      intro c hc b j hT
      show ((S.blk c).run (w c)).mem b j = _
      rw [hp.blk c]
      by_cases hown : b = S.out ∧ S.dom c j
      · obtain ⟨rfl, hd⟩ := hown
        have hv := S.value c (w c) j hd
        have e : S.ew.elabAt c 0 S.iregs S.imem = S.ew.elabAt c 0 := by simp only [hp.1, hp.2]
        rw [e] at hv
        rw [hv, StageSpec.step_val S hex M c j (by omega) hd]
        refine val_reads S hp (w c) _ _ c j hd (fun b' j' h => ?_)
        rcases h with ⟨l, hl⟩ | ⟨rfl, rfl⟩
        · exact hw c hc b' j' (hTS S List.mem_cons_self c l b' j' hl)
        · exact hw c hc _ _ hT
      · have hfr := S.frame c (w c) b j (by
          by_cases hb : b = S.out
          · exact Or.inr (fun hd => hown ⟨hb, hd⟩)
          · exact Or.inl hb)
        have e : S.ew.elabAt c 0 S.iregs S.imem = S.ew.elabAt c 0 := by simp only [hp.1, hp.2]
        rw [e] at hfr
        rw [hfr, hw c hc b j hT]
        by_cases hother : b = S.out ∧ ∃ c', c' < S.grid ∧ S.dom c' j
        · obtain ⟨rfl, c', hc', hd'⟩ := hother
          have hne : c' ≠ c := fun h => hown ⟨rfl, h ▸ hd'⟩
          rw [StageSpec.step_val S hex M c' j hc' hd']
          have hns : ∀ l : Lane, (S.out, j) ∉ storeL c' l 0 S.ew :=
            fun l hs => hAp S List.mem_cons_self c c' hc (by omega) (Ne.symm hne) l _ _ hs hT
          have hv := S.value c' (withMem (w c) M) j hd'
          have e' : S.ew.elabAt c' 0 S.iregs S.imem = S.ew.elabAt c' 0 := by simp only [hp.1, hp.2]
          rw [e', run_keepS c' 0 S.ew _ _ _ hns] at hv
          exact hv
        · rw [StageSpec.step, dif_neg hother]
    intro c hc b j hT
    show (rest.foldl (fun st S => (S.blk c).run st) ((S.blk c).run (w c))).mem b j
      = (Pipeline.mk rest).denote (S.step M) b j
    exact ih (fun S' h => hS S' (List.mem_cons_of_mem _ h))
      (fun S' h => hTS S' (List.mem_cons_of_mem _ h))
      (fun S' h => hAp S' (List.mem_cons_of_mem _ h)) w' (S.step M) hw' c hc b j hT

/-- **A pipeline leaves every word no member's statement stores.** -/
theorem denote_keep : ∀ (ss : List (StageSpec R)), (∀ S ∈ ss, Plain S) →
    ∀ (b : Buf) (j : Nat), (∀ S ∈ ss, ∀ c, c < S.grid → ∀ l : Lane, (b, j) ∉ storeL c l 0 S.ew) →
      ∀ m : Buf → Nat → R, (Pipeline.mk ss).denote m b j = m b j := by
  intro ss
  induction ss with
  | nil => intro _ _ _ _ _; rfl
  | cons S rest ih =>
    intro hp b j hn m
    show (Pipeline.mk rest).denote (S.step m) b j = m b j
    have hstep : S.step m b j = m b j := by
      by_cases hc : b = S.out ∧ ∃ cta, cta < S.grid ∧ S.dom cta j
      · obtain ⟨rfl, c, hc, hd⟩ := hc
        rw [StageSpec.step, dif_pos ⟨rfl, c, hc, hd⟩]
        have hex : S.val m (Exists.choose (⟨c, hc, hd⟩ : ∃ cta, cta < S.grid ∧ S.dom cta j)) j = m S.out j := by
          have hsp := (⟨c, hc, hd⟩ : ∃ cta, cta < S.grid ∧ S.dom cta j).choose_spec
          let c0 := (⟨c, hc, hd⟩ : ∃ cta, cta < S.grid ∧ S.dom cta j).choose
          let w : WSt R := ⟨fun _ _ => m S.out j, m, fun _ => m S.out j⟩
          have hv := S.value c0 w j hsp.2
          have e1 : S.ew.elabAt c0 0 S.iregs S.imem = S.ew.elabAt c0 0 := by
            simp only [(hp S List.mem_cons_self).1, (hp S List.mem_cons_self).2]
          rw [e1, run_keepS c0 0 S.ew w S.out j (hn S List.mem_cons_self c0 hsp.1)] at hv
          exact hv.symm
        exact hex
      · rw [StageSpec.step, dif_neg hc]
    rw [ih (fun S' h => hp S' (List.mem_cons_of_mem _ h)) b j (fun S' h => hn S' (List.mem_cons_of_mem _ h)) _,
      hstep]

end Carrier

end AlgorithmLib.ML.GroupRun
