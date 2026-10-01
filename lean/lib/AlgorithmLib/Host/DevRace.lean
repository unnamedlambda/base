module
public import AlgorithmLib.Host.DevTrack
public import AlgorithmLib.Host.DevSpec
meta import AlgorithmLib.Host.DevTrack
meta import AlgorithmLib.Host.DevSpec
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The default stream alone never races

A program that issues all its device work on the default stream finds every
buffer ready for it. Two facts give that: every access the tracker records is
at or below its own party's clock (`CReached.ownBelow`, of every state the
tracker reaches), and every access recorded is the default stream's
(`DefaultOnly`, kept by each operation the default stream issues).
-/

namespace AlgorithmLib.Device

open AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.DevSpec (wAcc rAcc know Ready Below)

/-- Every recorded access is at or below its own party's clock. -/
def OwnBelow (r : Race) : Prop :=
  ∀ b e, (e ∈ wAcc r b ∨ e ∈ rAcc r b) → e.2 ≤ (r.clock e.1).get e.1

/-- Every recorded access is the default stream's. -/
def DefaultOnly (r : Race) : Prop :=
  ∀ b e, (e ∈ wAcc r b ∨ e ∈ rAcc r b) → e.1 = defaultParty

theorem wAcc_setClock (r : Race) (p : Nat) (c : Clock) (b : Nat) : wAcc (r.setClock p c) b = wAcc r b := rfl
theorem rAcc_setClock (r : Race) (p : Nat) (c : Clock) (b : Nat) : rAcc (r.setClock p c) b = rAcc r b := rfl

theorem clock_joinInto_ge (r : Race) (q : Nat) (C : Clock) (x i : Nat) :
    (r.clock x).get i ≤ ((r.joinInto q C).clock x).get i := by
  simp only [Race.joinInto, Race.clock_setClock]
  split
  · next h => subst h; rw [Clock.get_join]; exact Nat.le_max_left _ _
  · exact Nat.le_refl _

theorem OwnBelow.joinInto {r : Race} (h : OwnBelow r) (q : Nat) (C : Clock) : OwnBelow (r.joinInto q C) :=
  fun b e he => Nat.le_trans (h b e he) (clock_joinInto_ge r q C e.1 e.1)

theorem DefaultOnly.joinInto {r : Race} (h : DefaultOnly r) (q : Nat) (C : Clock) :
    DefaultOnly (r.joinInto q C) := h

theorem DefaultOnly.sync {r : Race} (h : DefaultOnly r) (p : Nat) : DefaultOnly (r.sync p) := h

theorem ownBelow_empty : OwnBelow {} := by
  intro b e he; simp [wAcc, rAcc] at he

theorem defaultOnly_empty : DefaultOnly {} := by
  intro b e he; simp [wAcc, rAcc] at he

/-- What an accepted operation leaves: its own access, by its issuing clock,
    over what was there. -/
theorem op_accs {r r' : Race} {p : Nat} {rs ws : List Nat} (h : r.op p rs ws = some r') (b : Nat)
    (e : Nat × Nat) (he : e ∈ wAcc r' b ∨ e ∈ rAcc r' b) :
    (e ∈ wAcc r b ∨ e ∈ rAcc r b) ∨ e = (p, (r.issue p).2.get p) := by
  unfold Race.op at h
  obtain ⟨-, rfl⟩ := access_eq h
  simp only [wAcc, rAcc] at he ⊢
  rw [writesFold_lastW, writesFold_reads] at he
  rcases he with he | he
  · split at he
    · simp at he; exact .inr he
    · rw [(readsFold_other _ _ _).2] at he; exact .inl (.inl he)
  · split at he
    · simp at he
    · rw [readsFold_mem] at he
      rcases he with he | ⟨-, rfl⟩
      · exact .inl (.inr he)
      · exact .inr rfl

theorem op_clock {r r' : Race} {p : Nat} {rs ws : List Nat} (h : r.op p rs ws = some r') (q : Nat) :
    r'.clock q = (r.setClock p (r.issue p).2).clock q := by
  unfold Race.op at h
  obtain ⟨-, rfl⟩ := access_eq h
  simp only [Race.clock, writesFold_other, (readsFold_other _ _ _).1]
  rfl

theorem OwnBelow.op {r r' : Race} (h : OwnBelow r) {p : Nat} {rs ws : List Nat}
    (hop : r.op p rs ws = some r') : OwnBelow r' := by
  intro b e he
  rw [op_clock hop, Race.clock_setClock]
  rcases op_accs hop b e he with he | rfl
  · have := h b e he
    split
    · next hq =>
      rw [hq] at this ⊢
      refine Nat.le_trans this ?_
      simp only [Race.issue, Clock.get_bump, if_true, Clock.get_join]
      omega
    · exact this
  · simp

theorem DefaultOnly.op {r r' : Race} (h : DefaultOnly r) {rs ws : List Nat}
    (hop : r.op defaultParty rs ws = some r') : DefaultOnly r' := by
  intro b e he
  rcases op_accs hop b e he with he | rfl
  · exact h b e he
  · rfl

/-- **Every state the tracker reaches has each access below its party's
    clock.** -/
theorem CReached.ownBelow {r : Race} {evs : List Clock} (h : CReached r evs) : OwnBelow r := by
  induction h with
  | init => exact ownBelow_empty
  | op p rs ws _ hop ih => exact ih.op hop
  | record _ _ ih => exact ih
  | learnParty q p _ ih => exact ih.joinInto _ _
  | learnEvent q C _ _ ih => exact ih.joinInto _ _
  | learnTwo q p₁ p₂ _ ih => exact ih.joinInto _ _
  | learnNothing q _ ih => exact ih.joinInto _ _

/-- **The default stream finds every buffer ready** when every access is its
    own. -/
theorem ready_default {r : Race} (hO : OwnBelow r) (hD : DefaultOnly r) (b : Nat) (write : Bool) :
    Ready r defaultParty b write := by
  have key : ∀ e, (e ∈ wAcc r b ∨ e ∈ rAcc r b) → e.2 ≤ (know r defaultParty).get e.1 := by
    intro e he
    have h1 := hO b e he
    rw [hD b e he] at h1 ⊢
    simp only [know, Clock.get_join]
    omega
  exact ⟨fun e he => key e (.inl he), fun _ e he => key e (.inr he)⟩

end AlgorithmLib.Device
