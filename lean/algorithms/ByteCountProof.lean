import ByteCountAlgorithm

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem

/-!
  # What one trip of the byte scanner computes

  `ByteCountAlgorithm` emits, per vector and per needle, three instructions:

      m ← icmp .eq v nVec        -- every lane against the splatted needle
      h ← vhighBits m            -- the lane mask, as sixteen bits of a scalar
      p ← popcnt h               -- how many bits are set

  That this sequence counts the matching bytes is not obvious, and it is the
  step a person writing intrinsics gets wrong: the comparison leaves each lane
  all-ones or zero, `vhighBits` keeps only each lane's *top* bit, and `popcnt`
  then counts across a 32-bit word of which only sixteen positions can be live.

  `trip_counts` says it does, for every sixteen bytes and every needle, over
  `HProgSem`'s semantics --- the semantics the shipped artifact is checked
  against, not a paraphrase of the algorithm. The domain is 2^128 inputs, so no
  test establishes it, and no `static_assert`, `constexpr` or `requires` clause
  can state it: those range over values a compiler knows, and this ranges over
  values the program will meet.

  The claim is about one trip. The loop that repeats it is ordinary
  accumulation, and nothing here says anything about it.
-/

namespace ByteCountProof

/-! ## The statement -/

/-- The three instructions the scan emits per vector, under the semantics the
    artifact is checked against. Each operation reads its operands from the
    slots in scope, so a one-element environment per step is just naming. -/
def trip (m : Mem) (xs : Array UInt64) (n : UInt64) : Option V := do
  let mask ← evalOp m #[.vec .i8x16 xs, .vec .i8x16 (Array.replicate 16 n)] (.icmp .eq 0 1)
  let bits ← evalOp m #[mask] (.vhighBits 0)
  evalOp m #[bits] (.popcnt 0)

/-- How many of `xs` are `n`. -/
def matchCount (xs : Array UInt64) (n : UInt64) : Nat := (xs.filter (· == n)).size

/-! ## Bits -/

theorem get_def (Γ : Env) (r : R) : get Γ r = Γ[r]? := rfl

/-- The fold `vhighBits` performs sets bit `k` exactly when `k` is one of the
    positions it was given. -/
theorem testBit_foldl_or (l : List Nat) (acc : Nat) (k : Nat) :
    Nat.testBit (l.foldl (fun a i => a ||| 1 <<< i) acc) k
      = (Nat.testBit acc k || l.contains k) := by
  induction l generalizing acc with
  | nil => simp
  | cons a rest ih =>
      rw [List.foldl_cons, ih]
      simp [Nat.testBit_or, Nat.shiftLeft_eq, Nat.one_mul, Nat.testBit_two_pow, eq_comm,
            Bool.or_assoc]

/-- `popcnt`'s bit test, in `Nat` terms. -/
theorem bit_test_ofNat (v i : Nat) (hi : i < 64) :
    ((UInt64.ofNat v >>> UInt64.ofNat i) &&& 1 == 1) = Nat.testBit (v % 2 ^ 64) i := by
  have h64 : i < 2 ^ 64 := Nat.lt_trans hi (by decide)
  have h1 : (1 : Nat) % 2 ^ 64 = 1 := by decide
  rw [Bool.eq_iff_iff]
  simp only [beq_iff_eq, ← UInt64.toNat_inj, UInt64.toNat_and, UInt64.toNat_shiftRight,
             UInt64.toNat_ofNat', UInt64.toNat_ofNat, Nat.mod_eq_of_lt h64,
             Nat.mod_eq_of_lt hi, h1, Nat.testBit, Nat.and_comm (1 : Nat),
             Nat.and_one_is_mod, bne_iff_ne, ne_eq]
  omega

/-- `norm .i32` is the identity on a value that already fits. -/
theorem and_mask32 (P : Nat) (hP : P < 2 ^ 32) :
    UInt64.ofNat P &&& 0xffffffff = UInt64.ofNat P := by
  have h64 : P < 2 ^ 64 := Nat.lt_trans hP (by decide)
  rw [← UInt64.toNat_inj]
  simp only [UInt64.toNat_and, UInt64.toNat_ofNat', UInt64.toNat_ofNat,
             Nat.mod_eq_of_lt h64, show (0xffffffff : Nat) = 2 ^ 32 - 1 by decide,
             Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hP]
  rw [Nat.mod_eq_of_lt (show 2 ^ 32 - 1 < 2 ^ 64 by decide),
      Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hP]

/-! ## Positions -/

/-- The positions a predicate selects, numbering from `k`. This is the list the
    emitted `vhighBits` folds over. -/
def hits {α : Type} (p : α → Bool) (l : List α) (k : Nat) : List Nat :=
  ((l.zipIdx k).filter (fun q => p q.1)).map (·.2)

theorem hits_nil {α} (p : α → Bool) (k : Nat) : hits p [] k = [] := rfl

theorem hits_cons {α} (p : α → Bool) (a : α) (l : List α) (k : Nat) :
    hits p (a :: l) k = (if p a then [k] else []) ++ hits p l (k + 1) := by
  cases h : p a <;> simp [hits, List.zipIdx_cons, h]

theorem hits_lt {α} (p : α → Bool) (l : List α) (k i : Nat) (hi : i ∈ hits p l k) :
    i < k + l.length := by
  induction l generalizing k with
  | nil => simp [hits_nil] at hi
  | cons a rest ih =>
      rw [hits_cons] at hi
      cases h : p a <;> simp [h] at hi
      · have := ih (k + 1) hi; simp only [List.length_cons]; omega
      · rcases hi with rfl | hi
        · simp only [List.length_cons]; omega
        · have := ih (k + 1) hi; simp only [List.length_cons]; omega

theorem hits_length {α} (p : α → Bool) (l : List α) (k : Nat) :
    (hits p l k).length = (l.filter p).length := by
  induction l generalizing k with
  | nil => simp [hits_nil]
  | cons a rest ih => cases h : p a <;> simp [hits_cons, h, List.filter_cons, ih]

/-- The positions are a sublist of the range they are drawn from --- in
    particular they are increasing and distinct. -/
theorem hits_sublist {α} (p : α → Bool) (l : List α) (k : Nat) :
    (hits p l k).Sublist (List.range' k l.length) := by
  induction l generalizing k with
  | nil => simp [hits_nil]
  | cons a rest ih =>
      rw [hits_cons, List.length_cons, List.range'_succ]
      cases h : p a
      · simpa using (ih (k + 1)).cons k
      · simpa using (ih (k + 1)).cons₂ k

/-- Filtering a nodup list by membership in one of its sublists returns it. -/
theorem filter_mem_of_sublist {α : Type} [DecidableEq α] :
    ∀ {s l : List α}, s.Sublist l → l.Nodup →
      l.filter (fun i => decide (i ∈ s)) = s := by
  intro s l hsub hnd
  induction hsub with
  | slnil => simp
  | cons a hsub ih =>
      rename_i l₁ l₂
      simp only [List.nodup_cons] at hnd
      have hne : a ∉ l₁ := fun h => hnd.1 (hsub.subset h)
      simp [List.filter_cons, hne, ih hnd.2]
  | cons₂ a hsub ih =>
      rename_i l₁ l₂
      simp only [List.nodup_cons] at hnd
      have hcong : ∀ i ∈ l₂, (decide (i = a) || decide (i ∈ l₁)) = decide (i ∈ l₁) := by
        intro i hi
        have hia : i ≠ a := fun h => hnd.1 (h ▸ hi)
        simp [hia]
      simp [List.filter_cons, List.filter_congr hcong, ih hnd.2]

/-- Lanes whose top bit survives the comparison are exactly the matching ones. -/
theorem filter_zipWith_eq_count (l : List UInt64) (n : UInt64) :
    (List.filter (fun v => v &&& 128 != 0)
      (List.zipWith (fun x y => if x = y then (255 : UInt64) else 0) l
        (List.replicate l.length n))).length
      = (l.filter (· == n)).length := by
  induction l with
  | nil => simp
  | cons a rest ih =>
      rw [List.length_cons, List.replicate_succ, List.zipWith_cons_cons,
          List.filter_cons, List.filter_cons]
      have h255 : ((255 : UInt64) &&& 128) = 128 := by decide
      have h0 : ((0 : UInt64) &&& 128) = 0 := by decide
      by_cases h : a = n <;> simp [h, ih, h255, h0]

theorem range32_split : List.range 32 = List.range' 0 16 ++ List.range' 16 16 := by
  rw [List.range_eq_range']
  have := List.range'_append (s := 0) (m := 16) (n := 16) (step := 1)
  simpa using this.symm

/-- Counting the set bits of the fold recovers the number of selected
    positions --- the step the SIMD idiom turns on. -/
theorem popcount_hits {α} (p : α → Bool) (l : List α) (hl : l.length = 16) :
    (List.filter (fun i =>
        UInt64.ofNat ((hits p l 0).foldl (fun a i => a ||| 1 <<< i) 0)
          >>> UInt64.ofNat i &&& 1 == 1)
      (List.range 32)).length
      = (l.filter p).length := by
  have hpred : ∀ i ∈ List.range 32,
      (UInt64.ofNat ((hits p l 0).foldl (fun a i => a ||| 1 <<< i) 0)
        >>> UInt64.ofNat i &&& 1 == 1) = decide (i ∈ hits p l 0) := by
    intro i hi
    have hi64 : i < 64 := by simp [List.mem_range] at hi; omega
    rw [bit_test_ofNat _ _ hi64, Nat.testBit_mod_two_pow, testBit_foldl_or]
    simp [hi64, List.contains_iff_mem]
  rw [List.filter_congr hpred, range32_split, List.filter_append]
  have htail : (List.range' 16 16).filter (fun i => decide (i ∈ hits p l 0)) = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro i hi hmem
    have := hits_lt p l 0 i (by simpa using hmem)
    simp [List.mem_range'] at hi
    omega
  have hhead : (List.range' 0 16).filter (fun i => decide (i ∈ hits p l 0)) = hits p l 0 := by
    apply filter_mem_of_sublist
    · simpa [hl] using hits_sublist p l 0
    · exact List.nodup_range' 1
  rw [htail, hhead, List.append_nil, hits_length]

/-! ## The theorem -/

/-- **For every sixteen bytes and every needle, one trip counts exactly the
    matching bytes.** -/
theorem trip_counts (m : Mem) (xs : Array UInt64) (n : UInt64) (hsz : xs.size = 16) :
    trip m xs n = some (.sc .i32 (UInt64.ofNat (matchCount xs n))) := by
  simp +decide [trip, evalOp, get_def, zipIntCmp, cmpInt, ClifTy.lanes, widthMask, un, norm,
        ClifTy.width, ClifTy.isInt, hsz]
  have hmsz : (Array.zipWith (fun x y => if x = y then (255:UInt64) else 0) xs
                 (Array.replicate 16 n)).size = 16 := by simp [hsz]
  have hlen : (Array.zipWith (fun x y => if x = y then (255:UInt64) else 0) xs
                 (Array.replicate 16 n)).toList.length = 16 := by
    rw [Array.length_toList, hmsz]
  -- the emitted `Array.filter`/`Array.foldl`, as a fold over a list
  have key : ∀ (a : Array (UInt64 × Nat)), a.size = 16 →
      Array.foldl (fun acc (x : UInt64 × Nat) => acc ||| 1 <<< x.snd) 0
        (Array.filter (fun (x : UInt64 × Nat) => x.fst &&& (1 : UInt64) <<< (7 : UInt64) != 0) a 0 16)
        = List.foldl (fun acc (x : UInt64 × Nat) => acc ||| 1 <<< x.snd) 0
            (a.toList.filter (fun (x : UInt64 × Nat) => x.fst &&& (1 : UInt64) <<< (7 : UInt64) != 0)) := by
    intro a h
    rw [← h, ← Array.foldl_toList, Array.toList_filter]
  have asHits : ∀ (l : List UInt64),
      List.foldl (fun acc (x : UInt64 × Nat) => acc ||| 1 <<< x.snd) 0
        ((l.zipIdx 0).filter (fun (x : UInt64 × Nat) => x.fst &&& (1 : UInt64) <<< (7 : UInt64) != 0))
      = (hits (fun v : UInt64 => v &&& (1 : UInt64) <<< (7 : UInt64) != 0) l 0).foldl (fun a i => a ||| 1 <<< i) 0 :=
    fun l => by rw [hits, List.foldl_map]
  rw [key _ (by rw [Array.size_zipIdx, hmsz]), Array.toList_zipIdx, asHits,
      popcount_hits _ _ hlen, Array.toList_zipWith, Array.toList_replicate,
      show (16 : Nat) = xs.toList.length by rw [Array.length_toList, hsz],
      show ((1 : UInt64) <<< (7 : UInt64)) = 128 from rfl, filter_zipWith_eq_count]
  rw [matchCount, ← Array.length_toList, Array.toList_filter]
  refine and_mask32 _ ?_
  have h1 : (List.filter (fun x => x == n) xs.toList).length ≤ xs.toList.length :=
    List.length_filter_le _ _
  have h2 : xs.toList.length = 16 := by rw [Array.length_toList]; exact hsz
  have h3 : (2 : Nat) ^ 32 = 4294967296 := by decide
  omega

end ByteCountProof
