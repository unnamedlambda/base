import AlgorithmLib.LZ4Plan

namespace AlgorithmLib.LZ4WarpFind

open AlgorithmLib.LZ4Plan
open AlgorithmLib.LZ4Imp

/-- Byte access, out-of-range reads 0 (matches `LZ4Plan`). -/
abbrev byte (inp : List UInt8) (i : Nat) : UInt8 := inp.getD i 0

/-- The kernel's 4-byte candidate verify (`v32 == vc`). -/
def verify4 (inp : List UInt8) (p c : Nat) : Bool :=
  (List.range 4).all fun i => byte inp (p + i) == byte inp (c + i)

/-- Byte-by-byte match extension from `ml`, capped below `endCap`.
    (The kernel's cooperative ballot extend computes this same count.) -/
def extendFrom (inp : List UInt8) (p c endCap : Nat) : Nat → Nat → Nat
  | 0, ml => ml
  | fuel + 1, ml =>
      if p + ml < endCap ∧ byte inp (p + ml) = byte inp (c + ml)
      then extendFrom inp p c endCap fuel (ml + 1)
      else ml

/-- One lane's probe: candidate must exist, precede `p`, and pass the verify. -/
def probe (inp : List UInt8) (oracle : Nat → Option Nat) (searchLim p : Nat) :
    Option Nat :=
  match oracle p with
  | some c => if p < searchLim ∧ c < p ∧ verify4 inp p c then some c else none
  | none => none

/-- Scan lanes `lane, lane+1, …` (k remaining): first verified probe wins —
    the model of `vote.ballot` + `brev`/`clz` earliest-lane selection. -/
def windowGo (inp : List UInt8) (oracle : Nat → Option Nat) (searchLim s : Nat) :
    Nat → Nat → Option (Nat × Nat)
  | 0, _ => none
  | k + 1, lane =>
      match probe inp oracle searchLim (s + lane) with
      | some c => some (s + lane, c)
      | none => windowGo inp oracle searchLim s k (lane + 1)

def window (inp : List UInt8) (oracle : Nat → Option Nat) (searchLim s : Nat) :
    Option (Nat × Nat) :=
  windowGo inp oracle searchLim s 32 0

theorem extendFrom_le (inp : List UInt8) (p c endCap : Nat) :
    ∀ fuel ml, ml ≤ extendFrom inp p c endCap fuel ml := by
  intro fuel
  induction fuel with
  | zero => intro ml; simp [extendFrom]
  | succ n ih =>
      intro ml
      unfold extendFrom
      split
      · exact Nat.le_trans (Nat.le_succ ml) (ih (ml + 1))
      · exact Nat.le_refl ml

theorem extendFrom_agree (inp : List UInt8) (p c endCap : Nat) :
    ∀ fuel ml, (∀ i, i < ml → byte inp (p + i) = byte inp (c + i)) →
      ∀ i, i < extendFrom inp p c endCap fuel ml →
        byte inp (p + i) = byte inp (c + i) := by
  intro fuel
  induction fuel with
  | zero =>
      intro ml h i hi
      exact h i (by simpa [extendFrom] using hi)
  | succ n ih =>
      intro ml h i hi
      unfold extendFrom at hi
      split at hi
      · rename_i hcond
        refine ih (ml + 1) ?_ i hi
        intro j hj
        by_cases hj' : j < ml
        · exact h j hj'
        · have hje : j = ml := by omega
          subst hje
          exact hcond.2
      · exact h i hi

theorem extendFrom_cap (inp : List UInt8) (p c endCap : Nat) :
    ∀ fuel ml, extendFrom inp p c endCap fuel ml = ml ∨
      p + extendFrom inp p c endCap fuel ml ≤ endCap := by
  intro fuel
  induction fuel with
  | zero => intro ml; left; simp [extendFrom]
  | succ n ih =>
      intro ml
      unfold extendFrom
      split
      · rename_i hcond
        rcases ih (ml + 1) with heq | hle
        · right; rw [heq]; omega
        · right; exact hle
      · left; rfl

theorem verify4_agree (inp : List UInt8) (p c : Nat) (h : verify4 inp p c = true) :
    ∀ i, i < 4 → byte inp (p + i) = byte inp (c + i) := by
  intro i hi
  simp only [verify4, List.all_eq_true, List.mem_range] at h
  exact eq_of_beq (h i hi)

theorem probe_sound (inp : List UInt8) (oracle : Nat → Option Nat)
    (searchLim p c : Nat) (h : probe inp oracle searchLim p = some c) :
    p < searchLim ∧ c < p ∧ verify4 inp p c = true := by
  unfold probe at h
  split at h
  · split at h
    · rename_i hcond
      injection h with h'
      subst h'
      exact hcond
    · cases h
  · cases h

theorem windowGo_sound (inp : List UInt8) (oracle : Nat → Option Nat)
    (searchLim s : Nat) :
    ∀ k lane p c, windowGo inp oracle searchLim s k lane = some (p, c) →
      s ≤ p ∧ p < searchLim ∧ c < p ∧ verify4 inp p c = true := by
  intro k
  induction k with
  | zero => intro lane p c h; simp [windowGo] at h
  | succ n ih =>
      intro lane p c h
      unfold windowGo at h
      split at h
      · rename_i c' hpr
        injection h with h'
        injection h' with h1 h2
        subst h1; subst h2
        obtain ⟨hsl, hcp, hv⟩ := probe_sound inp oracle searchLim _ _ hpr
        exact ⟨Nat.le_add_right s lane, hsl, hcp, hv⟩
      · exact ih (lane + 1) p c h

theorem window_sound (inp : List UInt8) (oracle : Nat → Option Nat)
    (searchLim s p c : Nat) (h : window inp oracle searchLim s = some (p, c)) :
    s ≤ p ∧ p < searchLim ∧ c < p ∧ verify4 inp p c = true :=
  windowGo_sound inp oracle searchLim s 32 0 p c h

/-- A successful probe reports exactly the oracle's suggested candidate. -/
theorem probe_oracle (inp : List UInt8) (oracle : Nat → Option Nat) (searchLim p c : Nat)
    (h : probe inp oracle searchLim p = some c) : oracle p = some c := by
  cases ho : oracle p with
  | none => simp [probe, ho] at h
  | some d =>
      simp only [probe, ho] at h
      split at h
      · exact h
      · simp at h

/-- The `window`'s reported candidate is the oracle's value at the match position. -/
theorem windowGo_oracle (inp : List UInt8) (oracle : Nat → Option Nat) (searchLim s : Nat) :
    ∀ (k lane p c : Nat), windowGo inp oracle searchLim s k lane = some (p, c) → oracle p = some c := by
  intro k
  induction k with
  | zero => intro lane p c h; simp [windowGo] at h
  | succ n ih =>
      intro lane p c h
      rw [windowGo] at h
      cases hpr : probe inp oracle searchLim (s + lane) with
      | none => rw [hpr] at h; exact ih (lane + 1) p c h
      | some c' =>
          rw [hpr] at h
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨hp, hc⟩ := h; subst hp; subst hc
          exact probe_oracle inp oracle searchLim _ _ hpr

theorem window_oracle (inp : List UInt8) (oracle : Nat → Option Nat) (searchLim s p c : Nat)
    (h : window inp oracle searchLim s = some (p, c)) : oracle p = some c :=
  windowGo_oracle inp oracle searchLim s 32 0 p c h

/-- The `window` match position lies within the 32-lane scan window. -/
theorem windowGo_lt (inp : List UInt8) (oracle : Nat → Option Nat) (searchLim s : Nat) :
    ∀ (k lane p c : Nat), windowGo inp oracle searchLim s k lane = some (p, c) → p < s + lane + k := by
  intro k
  induction k with
  | zero => intro lane p c h; simp [windowGo] at h
  | succ n ih =>
      intro lane p c h
      rw [windowGo] at h
      cases hpr : probe inp oracle searchLim (s + lane) with
      | none => rw [hpr] at h; have := ih (lane + 1) p c h; omega
      | some c' =>
          rw [hpr] at h
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨hp, _⟩ := h; omega

theorem window_lt (inp : List UInt8) (oracle : Nat → Option Nat) (searchLim s p c : Nat)
    (h : window inp oracle searchLim s = some (p, c)) : p < s + 32 := by
  have := windowGo_lt inp oracle searchLim s 32 0 p c h; omega

-- ---------------------------------------------------------------------------
-- Main theorem
-- ---------------------------------------------------------------------------

end AlgorithmLib.LZ4WarpFind
