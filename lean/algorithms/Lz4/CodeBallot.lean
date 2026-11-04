module
public import Lz4.CodeWarp
public import AlgorithmLib.LZ4.WarpDSL
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeBallot` — what a warp's ballot answers

Where each of a warp's 32 lanes makes a ballot of its flag at the same
collective, the laws answer every lane the mask of the flags (`maskN`,
`answerStep_ballot`), whose trailing zeros are the first lane whose flag is
set, 32 where none is (`ctz32_mask`).
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel

/-- The mask of 32 lanes' flags. -/
def maskN (h : Nat → Bool) : Nat := (List.range 32).foldl (fun acc l => if h l then acc ||| 2 ^ l else acc) 0

theorem any_range_eq (Q : Nat → Bool) {j : Nat} (hj : j < 32) :
    (List.range 32).any (fun i => Q i && decide (id i = j)) = Q j := by
  cases hQ : Q j with
  | true =>
      rw [List.any_eq_true]
      exact ⟨j, List.mem_range.mpr hj, by simp [hQ]⟩
  | false =>
      rw [List.any_eq_false]
      intro i _ hi
      simp only [Bool.and_eq_true, decide_eq_true_eq, id] at hi
      obtain ⟨hq, rfl⟩ := hi
      rw [hQ] at hq; cases hq

/-- Bit `i` of the mask is lane `i`'s flag. -/
theorem maskN_testBit (h : Nat → Bool) {i : Nat} (hi : i < 32) : (maskN h).testBit i = h i := by
  unfold maskN
  have := testBit_foldl_or h id (List.range 32) 0 i
  simp only [id] at this
  rw [this, Nat.zero_testBit, Bool.false_or]
  simpa using any_range_eq h hi

theorem fold_or_lt (h : Nat → Bool) : ∀ (L : List Nat) (acc : Nat), acc < 2 ^ 32 → (∀ l ∈ L, l < 32) →
    L.foldl (fun acc l => if h l then acc ||| 2 ^ l else acc) acc < 2 ^ 32
  | [], acc, ha, _ => ha
  | l :: L, acc, ha, hL => by
      rw [List.foldl_cons]
      refine fold_or_lt h L _ ?_ (fun x hx => hL x (by simp [hx]))
      split
      · exact Nat.or_lt_two_pow ha (Nat.pow_lt_pow_right (by decide) (hL l (by simp)))
      · exact ha

theorem maskN_lt (h : Nat → Bool) : maskN h < 2 ^ 32 :=
  fold_or_lt h _ 0 (by decide) (fun l hl => List.mem_range.mp hl)

theorem fold_u32 (h : Nat → Bool) : ∀ (L : List Nat) (acc : UInt32), (∀ l ∈ L, l < 32) →
    (L.foldl (fun acc l => if h l then acc ||| (1 <<< l.toUInt32) else acc) acc).toNat =
      L.foldl (fun acc l => if h l then acc ||| 2 ^ l else acc) acc.toNat
  | [], _, _ => rfl
  | l :: L, acc, hL => by
      rw [List.foldl_cons, List.foldl_cons, fold_u32 h L _ (fun x hx => hL x (by simp [hx]))]
      congr 1
      have hl := hL l (by simp)
      split
      · rw [UInt32.toNat_or, UInt32.toNat_shiftLeft]
        have : l.toUInt32.toNat = l := by simp [Nat.toUInt32]; omega
        rw [this, show (1 : UInt32).toNat = 1 from rfl, Nat.mod_eq_of_lt (by omega : l < 32), Nat.one_shiftLeft,
          Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by decide) hl)]
      · rfl

theorem foldlM_ok {g : UInt32 → Nat → Except String UInt32} {f : UInt32 → Nat → UInt32}
    (hg : ∀ acc l, l < 32 → g acc l = .ok (f acc l)) :
    ∀ (L : List Nat) (acc : UInt32), (∀ l ∈ L, l < 32) → L.foldlM g acc = .ok (L.foldl f acc)
  | [], _, _ => rfl
  | l :: L, acc, hL => by
      rw [List.foldlM_cons, hg acc l (hL l (by simp))]
      exact foldlM_ok hg L _ (fun x hx => hL x (by simp [hx]))

theorem asBits_boolV (b : Bool) : ((asBits (boolV b)).getD 0 != 0) = b := by cases b <;> rfl

/-- **The laws answer a warp's ballot the mask of its lanes' flags.** -/
theorem answerStep_ballot {F : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {nb k t : Nat} {h : Nat → Bool} (ht : t < 32)
    (hF : ∀ l, l < 32 → (F l)[k]? = some (.ballot, [boolV (h l)])) (args : List V) :
    answerStep F snaps smem0 (t / 32) (t % 32) (nb, #[]) ((.ballot, args), k) =
      .ok (nb, #[.val (.sc .i32 (UInt32.ofNat (maskN h)).toUInt64)]) := by
  have ht0 : t / 32 = 0 := by omega
  have hg : ∀ acc l, l < 32 → (do
      match ← laneArg F (t / 32) l k .ballot with
      | [v] => pure (if (asBits v).getD 0 != 0 then acc ||| (1 <<< l.toUInt32) else acc)
      | _ => throw "a ballot of other than one value" : Except String UInt32) =
      .ok (if h l then acc ||| (1 <<< l.toUInt32) else acc) := by
    intro acc l hl
    simp only [laneArg, ht0, Nat.mul_zero, Nat.zero_add, hF l hl, show (Intr.ballot == Intr.ballot) = true by decide,
      if_true, bind, Except.bind, pure, Except.pure, asBits_boolV]
  unfold answerStep
  dsimp only
  erw [foldlM_ok hg (List.range 32) 0 (fun l hl => List.mem_range.mp hl)]
  simp only [bind, Except.bind]
  have hb : (List.foldl (fun acc l => if h l then acc ||| (1 <<< l.toUInt32) else acc) (0 : UInt32)
      (List.range 32)) = UInt32.ofNat (maskN h) := by
    apply UInt32.toNat_inj.mp
    rw [fold_u32 h _ 0 (fun l hl => List.mem_range.mp hl), UInt32.toNat_ofNat_of_lt' (maskN_lt h)]
    rfl
  rw [hb]; rfl

theorem shr_and_one (x : UInt64) {i : Nat} (hi : i < 64) :
    ((x >>> UInt64.ofNat i) &&& 1 == 1) = x.toNat.testBit i := by
  have hs : (x >>> UInt64.ofNat i).toNat = x.toNat >>> i := by
    rw [UInt64.toNat_shiftRight, UInt64.toNat_ofNat_of_lt' (by omega : i < 2 ^ 64), Nat.mod_eq_of_lt hi]
  have : ((x >>> UInt64.ofNat i) &&& 1 == 1) = decide ((x.toNat >>> i) % 2 = 1) := by
    rw [Bool.eq_iff_iff, beq_iff_eq, decide_eq_true_iff, ← UInt64.toNat_inj, UInt64.toNat_and, hs]
    simp [Nat.and_one_is_mod]
  rw [this, Nat.testBit_eq_decide_div_mod_eq, Nat.shiftRight_eq_div_pow]

/-- **The mask's trailing zeros are the first lane whose flag is set**, 32
    where none is. -/
theorem ctz32_mask (h : Nat → Bool) : ctz32 (UInt32.ofNat (maskN h)).toUInt64 = UInt64.ofNat (firstSetNat h 32 0) := by
  have hx : (UInt32.ofNat (maskN h)).toUInt64.toNat = maskN h := by
    simp [UInt32.toNat_ofNat_of_lt' (maskN_lt h)]
  have hf : (List.range 32).find? (fun i => ((UInt32.ofNat (maskN h)).toUInt64 >>> UInt64.ofNat i) &&& 1 == 1) =
      (List.range 32).find? h :=
    find?_congr _ fun i hi => by
      have hi := List.mem_range.mp hi
      rw [shr_and_one _ (by omega), hx, maskN_testBit h hi]
  have hle : ((List.range 32).find? h).getD 32 ≤ 32 := by
    cases e : (List.range 32).find? h with
    | none => simp
    | some v => have := List.mem_range.mp (List.mem_of_find?_eq_some e); simp; omega
  unfold ctz32
  rw [hf, firstSetNat_eq_find, ← List.range_eq_range', Nat.zero_add]
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, UInt64.toNat_ofNat_of_lt' (Nat.lt_of_le_of_lt hle (by decide)), show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl,
    Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt (by omega)]

end Lz4Code
