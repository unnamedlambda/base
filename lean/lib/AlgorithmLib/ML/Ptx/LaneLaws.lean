module
public import AlgorithmLib.ML.Ptx.LaneSettle
meta import AlgorithmLib.ML.Ptx.LaneSettle
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.LaneLaws` — the laws answer a warp's collectives as the warp does

Where every lane of warp `wi` makes, at its `k`-th collective, the call the
warp's run gives it at a collective leaf (`callOne`), the laws answer each
lane what the warp's run gives it there (`ansOne`): a ballot the mask of the
lanes' flags (`lawAt_ballot`), an index shuffle its source lane's word
(`lawAt_shflIdx`), a butterfly shuffle its partner's (`lawAt_bfly`), and a
barrier the shared memory the block's stores leave by then, which is the
warp's where they agree (`lawAt_barrier`). So where each lane's collectives
are the warp's, one by one, the laws give back the warp's answers
(`warp_laws`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyLane

namespace AlgorithmLib.ML.LaneLaws

open LaneWarp LaneSettle

/-! ## A ballot's mask -/

/-- Lane `l`'s flag, `false` past the warp. -/
def flagN (p : Lane → UInt64) (l : Nat) : Bool := if h : l < 32 then p ⟨l, h⟩ != 0 else false

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

theorem fold_u64 (p : Lane → UInt64) : ∀ (L : List Lane) (m : UInt64),
    (L.foldl (fun m l => if p l ≠ 0 then m ||| ((1 : UInt64) <<< UInt64.ofNat l.val) else m) m).toNat =
      (L.map Fin.val).foldl (fun acc l => if flagN p l then acc ||| 2 ^ l else acc) m.toNat
  | [], _ => rfl
  | l :: L, m => by
      rw [List.foldl_cons, List.map_cons, List.foldl_cons, fold_u64 p L]
      congr 1
      have hl : l.val < 32 := l.isLt
      have hf : flagN p l.val = (p l != 0) := by simp [flagN, hl]
      rw [hf]
      by_cases hp : p l ≠ 0
      · rw [if_pos hp, if_pos (by simpa using hp), UInt64.toNat_or, UInt64.toNat_shiftLeft,
          UInt64.toNat_ofNat_of_lt' (by omega : l.val < 2 ^ 64), show (1 : UInt64).toNat = 1 from rfl,
          Nat.mod_eq_of_lt (by omega : l.val < 64), Nat.one_shiftLeft,
          Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by decide) (by omega : l.val < 64))]
      · rw [if_neg hp, if_neg (by simpa using hp)]

theorem foldlM_ok {g : UInt32 → Nat → Except String UInt32} {f : UInt32 → Nat → UInt32}
    (hg : ∀ acc l, l < 32 → g acc l = .ok (f acc l)) :
    ∀ (L : List Nat) (acc : UInt32), (∀ l ∈ L, l < 32) → L.foldlM g acc = .ok (L.foldl f acc)
  | [], _, _ => rfl
  | l :: L, acc, hL => by
      rw [List.foldlM_cons, hg acc l (hL l (by simp))]
      exact foldlM_ok hg L _ (fun x hx => hL x (by simp [hx]))

theorem asBits_boolV (b : Bool) : ((asBits (boolV b)).getD 0 != 0) = b := by cases b <;> rfl

theorem finRange_val : (List.finRange 32).map Fin.val = List.range 32 := by decide

theorem ballot_u32 (p : Lane → UInt64) :
    (List.foldl (fun acc l => if flagN p l then acc ||| (1 <<< l.toUInt32) else acc) (0 : UInt32) (List.range 32)) =
      (ISt.ballot p).toUInt32 := by
  apply UInt32.toNat_inj.mp
  rw [fold_u32 _ _ 0 (fun l hl => List.mem_range.mp hl), UInt64.toNat_toUInt32, ISt.ballot, fold_u64, finRange_val,
    Nat.mod_eq_of_lt ?_]
  · rfl
  · have := ballot_lt p
    rw [ISt.ballot, fold_u64, finRange_val] at this
    exact this

/-! ## The laws, collective by collective -/

section Laws
variable {hw : Hw} {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
  {cta S wi k : Nat} {i : Nat} {ws : WSt F32}

/-- **The laws answer a ballot the mask of the lanes' flags.** -/
theorem lawAt_ballot {d : Nat} {p : IExp} (l : Lane)
    (hall : ∀ l' : Lane, (calls (32 * wi + l'.val))[k]? = some (callOne cta l' (i, .ballot d p, ws))) :
    LawAt hw calls snaps smem0 (32 * wi + l.val) k (ansOne cta S l (i, .ballot d p, ws)) := by
  refine ⟨_, ((calls (32 * wi + l.val)).take k).countP (·.1 == .barrier), hall l, ?_⟩
  obtain ⟨e1, e2⟩ := lane_div wi l
  have hg : ∀ acc l', l' < 32 → (do
      match ← laneArg calls ((32 * wi + l.val) / 32) l' k .ballot with
      | [v] => pure (if (asBits v).getD 0 != 0 then acc ||| (1 <<< l'.toUInt32) else acc)
      | _ => throw "a ballot of other than one value" : Except String UInt32) =
      .ok (if flagN (fun l => p.eval ws.int cta l) l' then acc ||| (1 <<< l'.toUInt32) else acc) := by
    intro acc l' hl'
    have h := hall ⟨l', hl'⟩
    simp only [laneArg, e1, h, callOne, callLeaf, List.headD_cons, show (Intr.ballot == Intr.ballot) = true by decide,
      if_true, bind, Except.bind, pure, Except.pure, asBits_boolV, flagN, hl', dif_pos]
  simp only [callOne, callLeaf, List.headD_cons]
  unfold answerStep
  dsimp only
  erw [foldlM_ok hg (List.range 32) 0 (fun l hl => List.mem_range.mp hl)]
  simp only [bind, Except.bind]
  rw [ballot_u32]
  rfl

theorem and_mask_toUInt32 (x : UInt64) : (x &&& 0xffffffff).toUInt32 = x.toUInt32 := by
  apply UInt32.toNat_inj.mp
  simp only [UInt64.toNat_toUInt32, UInt64.toNat_and, show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl,
    Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod]

theorem and_mask_mod (x : UInt64) : (x &&& 0xffffffff).toNat % 32 = x.toNat % 32 := by
  rw [UInt64.toNat_and, show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]
  exact Nat.mod_mod_of_dvd _ (by decide)

/-- **The laws answer an index shuffle its source lane's word.** -/
theorem lawAt_shflIdx {d : Nat} {v sr : IExp} (l : Lane)
    (hall : ∀ l' : Lane, (calls (32 * wi + l'.val))[k]? = some (callOne cta l' (i, .shflIdx d v sr, ws))) :
    LawAt hw calls snaps smem0 (32 * wi + l.val) k (ansOne cta S l (i, .shflIdx d v sr, ws)) := by
  refine ⟨_, ((calls (32 * wi + l.val)).take k).countP (·.1 == .barrier), hall l, ?_⟩
  obtain ⟨e1, e2⟩ := lane_div wi l
  let P : Lane := ISt.srcLane (sr.eval ws.int cta l)
  have hP : (sr.eval ws.int cta l &&& 0xffffffff).toNat % 32 = P.val := and_mask_mod _
  have h := hall P
  simp only [callOne, callLeaf, List.headD_cons, answerStep, laneArg, e1, bind, Except.bind, pure, Except.pure]
  rw [hP, h]
  simp only [callOne, callLeaf, List.headD_cons, show (Intr.shflIdx == Intr.shflIdx) = true by decide, if_true,
    ansOne, ansLeaf, lane32, and_mask_toUInt32, s32]
  rfl

/-- **The laws answer a butterfly shuffle its partner's word.** -/
theorem lawAt_bfly {d sr m : Nat} (hm : m < 2 ^ 31) (l : Lane)
    (hall : ∀ l' : Lane, (calls (32 * wi + l'.val))[k]? = some (callOne cta l' (i, .shflXor d sr m, ws))) :
    LawAt hw calls snaps smem0 (32 * wi + l.val) k (ansOne cta S l (i, .shflXor d sr m, ws)) := by
  refine ⟨_, ((calls (32 * wi + l.val)).take k).countP (·.1 == .barrier), hall l, ?_⟩
  obtain ⟨e1, e2⟩ := lane_div wi l
  have hmn : (s32 (UInt32.ofNat m)).toNat = m := by
    simp only [s32, UInt32.toNat_toUInt64]; exact toNat_ofNat32 (by omega)
  let P : Lane := xorLane l m
  have h := hall P
  simp only [callOne, callLeaf, List.headD_cons, answerStep, laneArg, e1, e2, hmn, bind, Except.bind, pure,
    Except.pure]
  rw [show 32 * wi + (l.val ^^^ m) % 32 = 32 * wi + P.val from rfl, h]
  simp only [callOne, callLeaf, List.headD_cons, show (Intr.shflBfly == Intr.shflBfly) = true by decide, if_true,
    ansOne, ansLeaf, lane32, s32]
  rfl

/-- **The laws answer a barrier the shared memory the block's stores leave by
    then**, the warp's where they agree. -/
theorem lawAt_barrier (l : Lane)
    (hl : (calls (32 * wi + l.val))[k]? = some (callOne cta l (i, .barrier, ws)))
    (hsnap : snaps[((calls (32 * wi + l.val)).take k).countP (·.1 == .barrier)]?.getD smem0 =
      bytesOf S ws.int.smem) :
    LawAt hw calls snaps smem0 (32 * wi + l.val) k (ansOne cta S l (i, .barrier, ws)) := by
  refine ⟨_, ((calls (32 * wi + l.val)).take k).countP (·.1 == .barrier) + 1, hl, ?_⟩
  simp only [callOne, callLeaf, List.headD_cons, answerStep, pure, Except.pure, hsnap]
  rfl

end Laws

/-! ## A warp's answers -/

/-- **The laws give back the warp's answers**: where each lane of warp `wi`
    makes the warp's collectives at its leaves `J`, in order, every barrier
    answered the warp's shared memory there, each lane is answered the warp's
    answers at them. -/
theorem warp_laws {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {cta S wi : Nat} (J : List (Nat × EWStmt × WSt F32))
    (hJ : ∀ e ∈ J, ∃ i ws, (∃ d p, e = (i, .ballot d p, ws)) ∨ (∃ d v sr, e = (i, .shflIdx d v sr, ws)) ∨
      (∃ d sr m, e = (i, .shflXor d sr m, ws) ∧ m < 2 ^ 31) ∨ e = (i, .barrier, ws))
    (hc : ∀ l : Lane, calls (32 * wi + l.val) = J.map (callOne cta l))
    (hbar : ∀ (l : Lane) k i ws, J[k]? = some (i, .barrier, ws) →
      snaps[((calls (32 * wi + l.val)).take k).countP (·.1 == .barrier)]?.getD smem0 = bytesOf S ws.int.smem) :
    ∀ l : Lane, answersOf hw calls snaps smem0 (32 * wi + l.val) = .ok (J.map (ansOne cta S l)) := by
  intro l
  refine answersOf_of_get (by rw [hc l]; simp) (fun k a ha => ?_)
  rw [List.getElem?_map] at ha
  obtain ⟨e, he, rfl⟩ := Option.map_eq_some_iff.mp ha
  have hall : ∀ l' : Lane, (calls (32 * wi + l'.val))[k]? = some (callOne cta l' e) := fun l' => by
    rw [hc l', List.getElem?_map, he]; rfl
  obtain ⟨i, ws, h | h | h | h⟩ := hJ e (List.mem_of_getElem? he)
  · obtain ⟨d, p, rfl⟩ := h; exact lawAt_ballot l hall
  · obtain ⟨d, v, sr, rfl⟩ := h; exact lawAt_shflIdx l hall
  · obtain ⟨d, sr, m, rfl, hm⟩ := h; exact lawAt_bfly hm l hall
  · subst h; exact lawAt_barrier l (hall l) (hbar l k i ws he)

end AlgorithmLib.ML.LaneLaws
