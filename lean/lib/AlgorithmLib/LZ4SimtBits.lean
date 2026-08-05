import AlgorithmLib.LZ4Simt
import AlgorithmLib.LZ4WarpSched

namespace AlgorithmLib.LZ4SimtBits
open AlgorithmLib.LZ4Simt

-- ── Reusable core: bit `j` of a fold that ORs in `2^(f a)` for selected `a` ─────

/-- Bit `j` of `foldl (acc ⊕= if P a then 2^(f a))` is set iff some selected
    element maps to `j`.  Both `brev32` and `ballotOf` are instances. -/
theorem testBit_foldl_or {α : Type} (P : α → Bool) (f : α → Nat) :
    ∀ (l : List α) (init j : Nat),
      (l.foldl (fun acc a => if P a then acc ||| 2 ^ (f a) else acc) init).testBit j
        = (init.testBit j || l.any (fun a => P a && decide (f a = j))) := by
  intro l
  induction l with
  | nil => intro init j; simp
  | cons a t ih =>
      intro init j
      simp only [List.foldl_cons, List.any_cons]
      rw [ih]
      have hstep : (if P a then init ||| 2 ^ (f a) else init).testBit j
          = (init.testBit j || (P a && decide (f a = j))) := by
        by_cases hP : P a
        · rw [if_pos hP, Nat.testBit_or, Nat.testBit_two_pow, hP, Bool.true_and]
        · rw [if_neg hP]
          have : P a = false := by simpa using hP
          rw [this, Bool.false_and, Bool.or_false]
      rw [hstep, Bool.or_assoc]

/-- `foldl`-of-OR stays below `2^32` when every set bit index is `< 32`. -/
theorem foldl_or_bound {α : Type} (P : α → Bool) (f : α → Nat) (hf : ∀ a, f a < 32) :
    ∀ (l : List α) (init : Nat), init < 2 ^ 32 →
      (l.foldl (fun acc a => if P a then acc ||| 2 ^ (f a) else acc) init) < 2 ^ 32 := by
  intro l
  induction l with
  | nil => intro init h; simpa using h
  | cons a t ih =>
      intro init h
      rw [List.foldl_cons]
      apply ih
      dsimp only
      by_cases hP : P a
      · rw [if_pos hP]
        have h1 : (2 : Nat) ^ (f a) < 2 ^ 32 :=
          Nat.pow_lt_pow_right (by decide : (1 : Nat) < 2) (hf a)
        exact Nat.or_lt_two_pow h h1
      · rw [if_neg hP]; exact h

/-- The `any` over `range 32` with a `31-i = j` selector picks out `i = 31-j`. -/
theorem any_range_bit (Q : Nat → Bool) (j : Nat) (hj : j < 32) :
    (List.range 32).any (fun i => Q i && decide (31 - i = j)) = Q (31 - j) := by
  cases hQ : Q (31 - j) with
  | true =>
      rw [List.any_eq_true]
      exact ⟨31 - j, List.mem_range.mpr (by omega),
        by rw [Bool.and_eq_true, decide_eq_true_eq]; exact ⟨hQ, by omega⟩⟩
  | false =>
      rw [List.any_eq_false]
      intro i hi
      rw [Bool.and_eq_true]
      rintro ⟨hQi, hdec⟩
      rw [decide_eq_true_eq] at hdec
      have hi32 := List.mem_range.mp hi
      have hij : i = 31 - j := by omega
      rw [hij, hQ] at hQi
      exact absurd hQi (by simp)

/-- The `toNat` of `brev32` is its `Nat`-level fold (it stays below `2^32`). -/
theorem brev32_toNat (x : UInt64) :
    (brev32 x).toNat
      = (List.range 32).foldl
          (fun acc i => if x.toNat.testBit i then acc ||| 2 ^ (31 - i) else acc) 0 := by
  unfold brev32
  have hbound : (List.range 32).foldl
      (fun acc i => if x.toNat.testBit i then acc ||| 2 ^ (31 - i) else acc) 0 < 2 ^ 64 :=
    Nat.lt_of_lt_of_le
      (foldl_or_bound (fun i => x.toNat.testBit i) (fun i => 31 - i)
        (fun i => by show 31 - i < 32; omega) (List.range 32) 0 (by decide))
      (Nat.pow_le_pow_right (by decide) (by decide))
  exact Nat.mod_eq_of_lt hbound

/-- **`brev` bit reversal**: bit `j` of `brev32 x` is bit `31-j` of `x` (`j<32`). -/
theorem brev32_testBit (x : UInt64) (j : Nat) (hj : j < 32) :
    (brev32 x).toNat.testBit j = x.toNat.testBit (31 - j) := by
  rw [brev32_toNat, testBit_foldl_or, Nat.zero_testBit, Bool.false_or,
      any_range_bit _ j hj]

/-- The `any` over `finRange 32` with a `l'.val = l.val` selector picks out `l`. -/
theorem any_finRange_bit (Q : Fin 32 → Bool) (l : Fin 32) :
    (List.finRange 32).any (fun l' => Q l' && decide (l'.val = l.val)) = Q l := by
  cases hQ : Q l with
  | true =>
      rw [List.any_eq_true]
      exact ⟨l, List.mem_finRange l,
        by rw [Bool.and_eq_true, decide_eq_true_eq]; exact ⟨hQ, rfl⟩⟩
  | false =>
      rw [List.any_eq_false]
      intro l' _
      rw [Bool.and_eq_true]
      rintro ⟨hQl', hdec⟩
      rw [decide_eq_true_eq] at hdec
      have hll : l' = l := Fin.ext hdec
      rw [hll, hQ] at hQl'
      exact absurd hQl' (by simp)

/-- `toNat` of `ballotOf` is its `Nat` fold (bits only in `[0,32)`). -/
theorem ballotOf_toNat (regs : String → Lane → UInt64) (p : String) :
    (ballotOf regs p).toNat
      = (List.finRange 32).foldl
          (fun acc l => if regs p l == 1 then acc ||| 2 ^ l.val else acc) 0 := by
  unfold ballotOf
  have hbound : (List.finRange 32).foldl
      (fun acc l => if regs p l == 1 then acc ||| 2 ^ l.val else acc) 0 < 2 ^ 64 :=
    Nat.lt_of_lt_of_le
      (foldl_or_bound (fun l => regs p l == 1) (fun l => l.val)
        (fun l => l.isLt) (List.finRange 32) 0 (by decide))
      (Nat.pow_le_pow_right (by decide) (by decide))
  exact Nat.mod_eq_of_lt hbound

/-- **Ballot bit structure**: bit `l` of the ballot is set iff lane `l` holds 1. -/
theorem ballotOf_testBit (regs : String → Lane → UInt64) (p : String) (l : Fin 32) :
    (ballotOf regs p).toNat.testBit l.val = (regs p l == 1) := by
  rw [ballotOf_toNat, testBit_foldl_or, Nat.zero_testBit, Bool.false_or, any_finRange_bit]

/-- Least index `≥ start` in the window `[start, start+bound)` with `g` true;
    `start+bound` if none.  With `(bound, start) = (32, 0)` this is exactly the
    lane `clz∘brev∘ballot` returns (or `32` for an empty ballot). -/
def firstSetNat (g : Nat → Bool) : Nat → Nat → Nat
  | 0, start => start
  | b + 1, start => if g start then start else firstSetNat g b (start + 1)

/-- **The clean bridge**: the least-set-index view equals `firstHit`,
    defaulting to `start+bound` (i.e. `32`) exactly when no lane hits. -/
theorem firstSetNat_eq_firstHit (g : Nat → Bool) :
    ∀ b start, firstSetNat g b start
      = (AlgorithmLib.LZ4WarpSched.firstHit g b start).getD (start + b) := by
  intro b
  induction b with
  | zero => intro start; simp [firstSetNat, AlgorithmLib.LZ4WarpSched.firstHit]
  | succ n ih =>
      intro start
      unfold firstSetNat AlgorithmLib.LZ4WarpSched.firstHit
      by_cases hg : g start = true
      · rw [if_pos hg, if_pos hg]; simp
      · rw [if_neg hg, if_neg hg, ih (start + 1)]
        congr 1; omega

/-- Specialization at the warp: `firstSetNat g 32 0 = (firstHit g 32 0).getD 32`. -/
theorem firstSetNat_warp (g : Nat → Bool) :
    firstSetNat g 32 0 = (AlgorithmLib.LZ4WarpSched.firstHit g 32 0).getD 32 := by
  have := firstSetNat_eq_firstHit g 32 0
  simpa using this

/-- `find?` depends only on the predicate's values at list elements. -/
theorem find?_congr {α : Type} {p q : α → Bool} :
    ∀ (l : List α), (∀ a ∈ l, p a = q a) → l.find? p = l.find? q := by
  intro l
  induction l with
  | nil => intro _; rfl
  | cons a t ih =>
      intro h
      rw [List.find?_cons, List.find?_cons, h a (by simp)]
      cases q a
      · simp; exact ih (fun b hb => h b (by simp [hb]))
      · simp

/-- Least index `k < 32` with `n.testBit k`; `32` if none. -/
def leastSetBit (n : Nat) : Nat := ((List.range 32).find? (fun k => n.testBit k)).getD 32

/-- The recursive scan `firstSetNat` equals the `find?` over the same window. -/
theorem firstSetNat_eq_find (g : Nat → Bool) :
    ∀ b start, firstSetNat g b start = ((List.range' start b).find? g).getD (start + b) := by
  intro b
  induction b with
  | zero => intro start; simp [firstSetNat, List.range']
  | succ n ih =>
      intro start
      rw [firstSetNat, List.range'_succ]
      by_cases hg : g start = true
      · rw [if_pos hg, List.find?_cons_of_pos hg]; simp
      · rw [if_neg hg, List.find?_cons_of_neg hg, ih (start + 1)]
        congr 1; omega

/-- **`clz∘brev` = least set bit.**  `brev` reverses the bit order, so scanning
    `clz`'s reversed range for the highest set bit of `brev x` finds the lowest
    set bit of `x`. -/
theorem clz32_brev32 (x : UInt64) :
    (clz32 (brev32 x)).toNat = leastSetBit x.toNat := by
  have hrev : (List.range 32).reverse = (List.range 32).map (fun k => 31 - k) := by decide
  have hfind : (List.range 32).reverse.find? (fun i => (brev32 x).toNat.testBit i)
      = Option.map (fun k => 31 - k) ((List.range 32).find? (fun k => x.toNat.testBit k)) := by
    rw [hrev, List.find?_map]
    congr 1
    apply find?_congr
    intro k hk
    have hk32 : k < 32 := List.mem_range.mp hk
    show (brev32 x).toNat.testBit (31 - k) = x.toNat.testBit k
    rw [brev32_testBit x (31 - k) (by omega)]
    congr 1; omega
  unfold clz32 leastSetBit
  rw [hfind]
  cases hforward : (List.range 32).find? (fun k => x.toNat.testBit k) with
  | some k0 =>
      have hk0 : k0 < 32 := List.mem_range.mp (List.mem_of_find?_eq_some hforward)
      show (UInt64.ofNat (31 - (31 - k0))).toNat = k0
      rw [show 31 - (31 - k0) = k0 from by omega]
      exact Nat.mod_eq_of_lt (by omega)
  | none =>
      show (UInt64.ofNat 32).toNat = 32
      exact Nat.mod_eq_of_lt (by decide)

/-- **Item 2b keystone**: the machine's `clz∘brev∘ballot` = the model's earliest
    hitting lane index (`firstSetNat` of the lane predicate). -/
theorem collective_select (regs : String → Lane → UInt64) (p : String) :
    (clz32 (brev32 (ballotOf regs p))).toNat
      = firstSetNat (fun k => (ballotOf regs p).toNat.testBit k) 32 0 := by
  rw [clz32_brev32, firstSetNat_eq_find]
  simp only [leastSetBit, List.range_eq_range', Nat.zero_add]

/-- **Item 2 capstone** (fully proved, axiom-clean): the machine's
    `vote.ballot → brev → clz` selection equals W2's `firstHit` of the lane
    predicate (defaulting to `32 = no lane`).  This is the single fact the RSim
    (item 4) invokes at the select site to hand off to `coopWindow_eq_window`. -/
theorem collective_select_firstHit (regs : String → Lane → UInt64) (p : String) :
    (clz32 (brev32 (ballotOf regs p))).toNat
      = (AlgorithmLib.LZ4WarpSched.firstHit
          (fun k => (ballotOf regs p).toNat.testBit k) 32 0).getD 32 := by
  rw [collective_select, firstSetNat_warp]

-- ── Concrete validation of the (now fully proved) bit bridge ──────────────────

/-- `clz∘brev∘ballot` finds the least hitting lane — single lane. -/
example :
    (clz32 (brev32 (ballotOf (fun _ l => if l.val = 5 then 1 else 0) "p"))).toNat = 5 := by
  native_decide

/-- Two hitting lanes → the earliest wins. -/
example :
    (clz32 (brev32 (ballotOf (fun _ l => if l.val = 5 ∨ l.val = 11 then 1 else 0) "p"))).toNat
      = 5 := by native_decide

/-- Lane 0 hits → index 0. -/
example :
    (clz32 (brev32 (ballotOf (fun _ l => if l.val = 0 then 1 else 0) "p"))).toNat = 0 := by
  native_decide

/-- Empty ballot → 32 (no lane), which the kernel's `ballot == 0` test routes to
    the no-match branch. -/
example : (clz32 (brev32 (ballotOf (fun _ _ => 0) "p"))).toNat = 32 := by native_decide

/-- And it agrees with the model spec on these: `firstSetNat` of the same
    predicate. -/
example :
    (clz32 (brev32 (ballotOf (fun _ l => if l.val = 5 ∨ l.val = 11 then 1 else 0) "p"))).toNat
      = firstSetNat (fun i => decide (i = 5 ∨ i = 11)) 32 0 := by native_decide

-- ── What the selected lane, and a full ballot, actually say ──────────────────

/-- Complementing a `UInt64` flips every bit below 64.  `bnot` is how the extend
    loop turns "which lanes matched" into "which lanes mismatched". -/
theorem uint64_not_testBit (x : UInt64) (k : Nat) (h : k < 64) :
    (~~~x).toNat.testBit k = !(x.toNat.testBit k) := by
  have e : ∀ y : UInt64, y.toNat.testBit k = y.toBitVec.getLsbD k := by
    intro y; simp [BitVec.getLsbD]
  rw [e, e, UInt64.toBitVec_not, BitVec.getLsbD_not]
  simp [h]

/-- If the least set bit is a real lane index, that bit is set. -/
theorem leastSetBit_testBit (n k : Nat) (h : leastSetBit n = k) (hk : k < 32) :
    n.testBit k = true := by
  unfold leastSetBit at h
  cases hf : (List.range 32).find? (fun i => n.testBit i) with
  | none => rw [hf] at h; simp at h; omega
  | some v =>
      rw [hf] at h
      simp only [Option.getD] at h
      rw [← h]
      exact List.find?_some hf

/-- …and if it is 32, no lane's bit is set. -/
theorem leastSetBit_eq_32 (n : Nat) (h : leastSetBit n = 32) :
    ∀ k, k < 32 → n.testBit k = false := by
  intro k hk
  unfold leastSetBit at h
  cases hf : (List.range 32).find? (fun i => n.testBit i) with
  | none =>
      have := List.find?_eq_none.mp hf k (by simp [List.mem_range, hk])
      simpa using this
  | some v =>
      rw [hf] at h
      simp only [Option.getD] at h
      have hv : v ∈ List.range 32 := List.mem_of_find?_eq_some hf
      rw [List.mem_range] at hv
      omega

/-- Everything strictly below the first set bit is clear.  With the scan read as
    "the first lane that failed", this says every earlier lane succeeded — which
    is what bounds the extend loop when it stops part-way through the warp. -/
theorem firstSetNat_lt_false (g : Nat → Bool) :
    ∀ b start a, firstSetNat g b start = a → ∀ k, start ≤ k → k < a → g k = false := by
  intro b
  induction b with
  | zero => intro start a h k h1 h2; rw [firstSetNat] at h; omega
  | succ m ih =>
      intro start a h k h1 h2
      rw [firstSetNat] at h
      by_cases hg : g start = true
      · rw [if_pos hg] at h; omega
      · rw [if_neg hg] at h
        rcases Nat.eq_or_lt_of_le h1 with he | hlt
        · rw [← he]
          cases hb : g start
          · rfl
          · exact absurd hb hg
        · exact ih (start + 1) a h k (by omega) h2

/-- **Every lane below the one the extend loop stopped at had its predicate
    set.**  The `bnot` makes `mis` the complement of the ballot, so a clear bit
    of `mis` is a lane that matched. -/
theorem ballot_below_clz_not (regs : String → Lane → UInt64) (p : String)
    (l : Lane) (h : l.val < (clz32 (brev32 (~~~ (ballotOf regs p)))).toNat) :
    regs p l = 1 := by
  have hzero : (~~~ (ballotOf regs p)).toNat.testBit l.val = false := by
    have hcs : (clz32 (brev32 (~~~ (ballotOf regs p)))).toNat
        = firstSetNat (fun k => (~~~ (ballotOf regs p)).toNat.testBit k) 32 0 := by
      rw [clz32_brev32, firstSetNat_eq_find]
      simp only [leastSetBit, List.range_eq_range', Nat.zero_add]
    rw [hcs] at h
    exact firstSetNat_lt_false _ 32 0 _ rfl l.val (by omega) h
  rw [uint64_not_testBit _ l.val (Nat.lt_trans l.isLt (by decide))] at hzero
  have hbit : (ballotOf regs p).toNat.testBit l.val = true := by
    cases hb : (ballotOf regs p).toNat.testBit l.val
    · rw [hb] at hzero; simp at hzero
    · rfl
  have := ballotOf_testBit regs p l
  rw [hbit] at this
  exact beq_iff_eq.mp this.symm

/-- A non-empty ballot selects a real lane: `clz∘brev` lands below 32. -/
theorem clz_brev_ballot_lt (regs : String → Lane → UInt64) (p : String)
    (h : ballotOf regs p ≠ 0) : (clz32 (brev32 (ballotOf regs p))).toNat < 32 := by
  rw [clz32_brev32]
  by_cases he : leastSetBit (ballotOf regs p).toNat = 32
  · -- no bit set below 32, but the ballot only ever has bits below 32
    exfalso
    obtain ⟨k, hk⟩ := Nat.exists_testBit_of_ne_zero
      (fun hz => h (by rw [← UInt64.toNat_inj]; simpa using hz))
    rcases Nat.lt_or_ge k 32 with hlt | hge
    · rw [leastSetBit_eq_32 _ he k hlt] at hk; exact absurd hk (by simp)
    · -- bits at or above 32 are impossible: the fold only ever sets bits `< 32`
      rw [ballotOf_toNat] at hk
      have hb := foldl_or_bound (fun l => regs p l == 1) (fun l : Lane => l.val)
        (fun l => l.isLt) (List.finRange 32) 0 (by decide)
      exact absurd (Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hb
        (Nat.pow_le_pow_right (by decide) hge))) (by rw [hk]; simp)
  · -- otherwise it is the index of a set bit, hence a lane
    unfold leastSetBit at he ⊢
    cases hf : (List.range 32).find? (fun i => (ballotOf regs p).toNat.testBit i) with
    | none => rw [hf] at he; simp at he
    | some v =>
        have hv : v ∈ List.range 32 := List.mem_of_find?_eq_some hf
        rw [List.mem_range] at hv
        exact hv

/-- **The lane the ballot selects really is a hitting lane.**

    `vote → brev → clz` picks the earliest lane whose predicate is set, and the
    kernel then `shfl`s that lane's registers to everybody.  This is what makes
    the shuffled values inherit the selected lane's guards — which is how the
    match candidate inherits `cand < posP` and `posP < searchLim`. -/
theorem ballot_select_holds (regs : String → Lane → UInt64) (p : String)
    (h : ballotOf regs p ≠ 0) :
    regs p (toLane (clz32 (brev32 (ballotOf regs p)))) = 1 := by
  have hlt := clz_brev_ballot_lt regs p h
  have hbit : (ballotOf regs p).toNat.testBit
      (clz32 (brev32 (ballotOf regs p))).toNat = true :=
    leastSetBit_testBit _ _ (clz32_brev32 _).symm hlt
  have := ballotOf_testBit regs p ⟨(clz32 (brev32 (ballotOf regs p))).toNat, hlt⟩
  rw [hbit] at this
  have hl : toLane (clz32 (brev32 (ballotOf regs p)))
      = (⟨(clz32 (brev32 (ballotOf regs p))).toNat, hlt⟩ : Lane) := by
    apply Fin.ext
    simp only [toLane]
    exact Nat.mod_eq_of_lt hlt
  rw [hl]
  exact beq_iff_eq.mp this.symm

/-- **A saturated ballot means every lane set its predicate.**

    The extend loop's continue-condition is `clz(brev(~ballot)) = 32` — "the
    first mismatching lane is past the end of the warp" — so this is what turns
    "the loop went round again" into a fact about all thirty-two lanes. -/
theorem ballot_full_of_clz_not (regs : String → Lane → UInt64) (p : String)
    (h : (clz32 (brev32 (~~~ (ballotOf regs p)))).toNat = 32) (l : Lane) :
    regs p l = 1 := by
  rw [clz32_brev32] at h
  have hnot := leastSetBit_eq_32 _ h l.val l.isLt
  rw [uint64_not_testBit _ l.val (Nat.lt_trans l.isLt (by decide))] at hnot
  have hbit : (ballotOf regs p).toNat.testBit l.val = true := by
    cases hb : (ballotOf regs p).toNat.testBit l.val
    · rw [hb] at hnot; simp at hnot
    · rfl
  have := ballotOf_testBit regs p l
  rw [hbit] at this
  exact beq_iff_eq.mp this.symm

end AlgorithmLib.LZ4SimtBits
