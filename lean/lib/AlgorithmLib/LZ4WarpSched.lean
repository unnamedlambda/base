import AlgorithmLib.LZ4WarpFind

namespace AlgorithmLib.LZ4WarpSched

open AlgorithmLib.LZ4WarpFind

/-- The per-index "good byte" test: in bounds and equal (kernel `pOk`). -/
def good (inp : List UInt8) (p c endCap k : Nat) : Bool :=
  decide (p + k < endCap) && (byte inp (p + k) == byte inp (c + k))

theorem good_iff (inp : List UInt8) (p c endCap k : Nat) :
    good inp p c endCap k = true ↔
      (p + k < endCap ∧ byte inp (p + k) = byte inp (c + k)) := by
  unfold good
  rw [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]

theorem good_false_past (inp : List UInt8) (p c endCap k : Nat)
    (h : endCap ≤ p + k) : good inp p c endCap k = false := by
  unfold good
  rw [decide_eq_false (by omega : ¬ p + k < endCap), Bool.false_and]

/-- `extendFrom`'s guard, expressed through `good`. -/
theorem extendFrom_step (inp : List UInt8) (p c endCap fuel ml : Nat) :
    extendFrom inp p c endCap (fuel + 1) ml =
      (if good inp p c endCap ml = true
       then extendFrom inp p c endCap fuel (ml + 1) else ml) := by
  simp only [extendFrom]
  by_cases h : good inp p c endCap ml = true
  · rw [if_pos ((good_iff inp p c endCap ml).mp h), if_pos h]
  · rw [if_neg (fun hc => h ((good_iff inp p c endCap ml).mpr hc)), if_neg h]

/-- **Spec of the extend result**: `L` is the least index ≥ ml failing `good`. -/
def Res (inp : List UInt8) (p c endCap ml L : Nat) : Prop :=
  ml ≤ L ∧ (∀ k, ml ≤ k → k < L → good inp p c endCap k = true) ∧
    good inp p c endCap L = false

/-- `Res` exists — bounded induction on the distance to `endCap`. -/
theorem Res_exists_bounded (inp : List UInt8) (p c endCap : Nat) :
    ∀ N ml, endCap - (p + ml) ≤ N → ∃ L, Res inp p c endCap ml L := by
  intro N
  induction N with
  | zero =>
      intro ml hN
      exact ⟨ml, Nat.le_refl ml,
        fun k hk1 hk2 => absurd (Nat.lt_of_le_of_lt hk1 hk2) (Nat.lt_irrefl ml),
        good_false_past inp p c endCap ml (by omega)⟩
  | succ n ih =>
      intro ml hN
      by_cases hg : good inp p c endCap ml = true
      · have hlt : p + ml < endCap := ((good_iff inp p c endCap ml).mp hg).1
        obtain ⟨L, hL1, hL2, hL3⟩ := ih (ml + 1) (by omega)
        refine ⟨L, by omega, ?_, hL3⟩
        intro k hk1 hk2
        by_cases hkml : k = ml
        · subst hkml; exact hg
        · exact hL2 k (by omega) hk2
      · exact ⟨ml, Nat.le_refl ml,
          fun k hk1 hk2 => absurd (Nat.lt_of_le_of_lt hk1 hk2) (Nat.lt_irrefl ml),
          by rw [Bool.not_eq_true] at hg; exact hg⟩

theorem Res_exists (inp : List UInt8) (p c endCap ml : Nat) :
    ∃ L, Res inp p c endCap ml L :=
  Res_exists_bounded inp p c endCap (endCap - (p + ml)) ml (Nat.le_refl _)

/-- `extendFrom` reaches the `Res` value once fuel covers the good run. -/
theorem extendFrom_eq_Res (inp : List UInt8) (p c endCap : Nat) :
    ∀ fuel ml L, Res inp p c endCap ml L → L - ml ≤ fuel →
      extendFrom inp p c endCap fuel ml = L := by
  intro fuel
  induction fuel with
  | zero =>
      intro ml L hR hf
      obtain ⟨hle, _, _⟩ := hR
      have : ml = L := by omega
      subst this; simp [extendFrom]
  | succ n ih =>
      intro ml L hR hf
      obtain ⟨hle, hg, hbad⟩ := hR
      rw [extendFrom_step]
      by_cases hgood : good inp p c endCap ml = true
      · rw [if_pos hgood]
        have hmlL : ml < L := by
          rcases Nat.lt_or_ge ml L with h | h
          · exact h
          · have hml : ml = L := by omega
            rw [hml] at hgood; rw [hbad] at hgood; exact absurd hgood (by simp)
        exact ih (ml + 1) L ⟨by omega, fun k hk1 hk2 => hg k (by omega) hk2, hbad⟩ (by omega)
      · rw [if_neg hgood]
        rcases Nat.lt_or_ge ml L with h | h
        · exact absurd (hg ml (Nat.le_refl ml) h) (by rw [Bool.not_eq_true] at hgood; rw [hgood]; simp)
        · omega

-- ---------------------------------------------------------------------------
-- Cooperative chunk extend
-- ---------------------------------------------------------------------------

/-- Count of consecutive `good` indices from `ml`, capped at `b` — the value the
    warp's `brev`/`clz` of the mismatch ballot yields per 32-lane step. -/
def leadingGood (inp : List UInt8) (p c endCap : Nat) (ml : Nat) : Nat → Nat
  | 0 => 0
  | b + 1 => if good inp p c endCap ml = true
             then 1 + leadingGood inp p c endCap (ml + 1) b else 0

/-- `leadingGood` never runs the extend position past `endCap`: every counted
    index `ml..ml+result-1` was `good`, so `p + (last good) < endCap` and thus
    `p + ml + result ≤ endCap`.  The extend loop's invariant `p + ml ≤ endCap`
    is preserved by `ml += leadingGood`. -/
theorem leadingGood_pos_le (inp : List UInt8) (p c endCap : Nat) :
    ∀ b ml, p + ml ≤ endCap → p + ml + leadingGood inp p c endCap ml b ≤ endCap := by
  intro b
  induction b with
  | zero => intro ml hml; simpa [leadingGood] using hml
  | succ n ih =>
      intro ml hml
      unfold leadingGood
      by_cases hgood : good inp p c endCap ml = true
      · rw [if_pos hgood]
        have hlt : p + ml < endCap := by
          have := (good_iff inp p c endCap ml).mp hgood; exact this.1
        have := ih (ml + 1) (by omega)
        omega
      · rw [if_neg hgood]; simpa using hml

/-- The cooperative extend: advance 32 on a full good window, else stop at the
    first mismatch (kernel `ext`/`extFin`). -/
def extendChunk (inp : List UInt8) (p c endCap : Nat) : Nat → Nat → Nat
  | 0, ml => ml
  | fuel + 1, ml =>
      if leadingGood inp p c endCap ml 32 = 32
      then extendChunk inp p c endCap fuel (ml + 32)
      else ml + leadingGood inp p c endCap ml 32

/-- Under `Res`, the leading-good count from `ml` is `min b (L - ml)`. -/
theorem leadingGood_Res (inp : List UInt8) (p c endCap : Nat) :
    ∀ b ml L, Res inp p c endCap ml L →
      leadingGood inp p c endCap ml b = min b (L - ml) := by
  intro b
  induction b with
  | zero => intro ml L _; simp [leadingGood]
  | succ n ih =>
      intro ml L hR
      obtain ⟨hle, hg, hbad⟩ := hR
      unfold leadingGood
      by_cases hgood : good inp p c endCap ml = true
      · rw [if_pos hgood]
        have hmlL : ml < L := by
          rcases Nat.lt_or_ge ml L with h | h
          · exact h
          · have hml : ml = L := by omega
            rw [hml] at hgood; rw [hbad] at hgood; exact absurd hgood (by simp)
        rw [ih (ml + 1) L ⟨by omega, fun k hk1 hk2 => hg k (by omega) hk2, hbad⟩]
        omega
      · rw [if_neg hgood]
        have hmlL : ml = L := by
          rcases Nat.lt_or_ge ml L with h | h
          · exact absurd (hg ml (Nat.le_refl ml) h) (by rw [Bool.not_eq_true] at hgood; rw [hgood]; simp)
          · omega
        omega

/-- `extendChunk` reaches the same `Res` value, given enough (×32) fuel. -/
theorem extendChunk_eq_Res (inp : List UInt8) (p c endCap : Nat) :
    ∀ fuel ml L, Res inp p c endCap ml L → L - ml ≤ 32 * fuel →
      extendChunk inp p c endCap fuel ml = L := by
  intro fuel
  induction fuel with
  | zero =>
      intro ml L hR hf
      obtain ⟨hle, _, _⟩ := hR
      have : ml = L := by omega
      simp [extendChunk, this]
  | succ n ih =>
      intro ml L hR hf
      obtain ⟨hle, hg, hbad⟩ := hR
      unfold extendChunk
      rw [leadingGood_Res inp p c endCap 32 ml L ⟨hle, hg, hbad⟩]
      by_cases hlt : L - ml < 32
      · rw [if_neg (show ¬ min 32 (L - ml) = 32 by omega)]
        omega
      · rw [if_pos (show min 32 (L - ml) = 32 by omega)]
        exact ih (ml + 32) L ⟨by omega, fun k hk1 hk2 => hg k (by omega) hk2, hbad⟩ (by omega)

/-- The cooperative 32-wide extend computes exactly the
    model's byte-by-byte `extendFrom`, for any fuels covering the good run. -/
theorem extendChunk_eq_extendFrom (inp : List UInt8) (p c endCap fuelC fuelF ml : Nat)
    (hC : endCap - (p + ml) ≤ 32 * fuelC) (hF : endCap - (p + ml) ≤ fuelF) :
    extendChunk inp p c endCap fuelC ml = extendFrom inp p c endCap fuelF ml := by
  obtain ⟨L, hR⟩ := Res_exists inp p c endCap ml
  obtain ⟨hle, hg, hbad⟩ := hR
  -- `endCap - p` is always a bad index; the least bad `L` sits at or before it,
  -- so `L - ml ≤ endCap - (p + ml)`.
  have hEh : endCap ≤ p + (endCap - p) := by omega
  have hEbad : good inp p c endCap (endCap - p) = false := good_false_past inp p c endCap (endCap - p) hEh
  have hLml : L - ml ≤ endCap - (p + ml) := by
    rcases Nat.lt_or_ge (endCap - p) ml with hlt | hge
    · have hmlh : endCap ≤ p + ml := by omega
      have hmlbad : good inp p c endCap ml = false := good_false_past inp p c endCap ml hmlh
      have hmlL : L ≤ ml := by
        rcases Nat.lt_or_ge ml L with h | h
        · rw [hg ml (Nat.le_refl ml) h] at hmlbad; exact absurd hmlbad (by simp)
        · exact h
      omega
    · have hLE : L ≤ endCap - p := by
        rcases Nat.lt_or_ge L (endCap - p + 1) with h | h
        · omega
        · have hEL : endCap - p < L := by omega
          rw [hg (endCap - p) hge hEL] at hEbad
          exact absurd hEbad (by simp)
      omega
  rw [extendChunk_eq_Res inp p c endCap fuelC ml L ⟨hle, hg, hbad⟩ (by omega),
      extendFrom_eq_Res inp p c endCap fuelF ml L ⟨hle, hg, hbad⟩ (by omega)]

/-- Earliest lane in `[lane, lane+b)` whose predicate holds — the `vote.ballot` +
    `brev`/`clz` earliest-set-lane reduction, as a scan. -/
def firstHit (f : Nat → Bool) : Nat → Nat → Option Nat
  | 0, _ => none
  | b + 1, lane => if f lane then some lane else firstHit f b (lane + 1)

/-- `windowGo`'s short-circuit scan = compute-all-then-pick-earliest-hit. -/
theorem windowGo_eq_firstHit (inp : List UInt8) (oracle : Nat → Option Nat)
    (searchLim s : Nat) :
    ∀ k lane, windowGo inp oracle searchLim s k lane =
      (match firstHit (fun L => (probe inp oracle searchLim (s + L)).isSome) k lane with
       | some L => (probe inp oracle searchLim (s + L)).map (fun c => (s + L, c))
       | none => none) := by
  intro k
  induction k with
  | zero => intro lane; rfl
  | succ n ih =>
      intro lane
      unfold windowGo firstHit
      cases hp : probe inp oracle searchLim (s + lane) with
      | some c => simp [hp]
      | none =>
          simp only [Option.isSome_none, Bool.false_eq_true, if_false]
          exact ih (lane + 1)

/-- The cooperative window select (earliest hitting lane, candidate broadcast). -/
def coopWindow (inp : List UInt8) (oracle : Nat → Option Nat) (searchLim s : Nat) :
    Option (Nat × Nat) :=
  match firstHit (fun L => (probe inp oracle searchLim (s + L)).isSome) 32 0 with
  | some L => (probe inp oracle searchLim (s + L)).map (fun c => (s + L, c))
  | none => none

/-- Cooperative earliest-lane select = the model's
    `window` (first verified probe in lane order). -/
theorem coopWindow_eq_window (inp : List UInt8) (oracle : Nat → Option Nat)
    (searchLim s : Nat) : coopWindow inp oracle searchLim s = window inp oracle searchLim s := by
  unfold coopWindow window
  exact (windowGo_eq_firstHit inp oracle searchLim s 32 0).symm

