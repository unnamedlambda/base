module
public import Lz4.CodeWindow
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeExtend` — a warp's lanes run an extend step with the warp

Each lane compares its byte of the match with the candidate's (`lane_extend_step`):
its flag is the model's `good` at `ml + lane`. The ballot of the flags,
complemented, has its trailing zeros at the first lane that is not good,
which is the model's `leadingGood` (`extend_lockstep`).
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel

/-- Whether lane `l`'s byte of the match is good. -/
abbrev extGood (ws : WState) (p0 cand0 ml : String) (inStride endCap l : Nat) : Bool :=
  AlgorithmLib.LZ4WarpSched.good (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (ws.regs p0).toNat
    (ws.regs cand0).toNat endCap ((ws.regs ml).toNat + l)

theorem byte_beq (a b : UInt8) : (a.toUInt64 == b.toUInt64) = (a == b) := by
  rw [Bool.eq_iff_iff, beq_iff_eq, beq_iff_eq]
  constructor
  · intro h; exact UInt8.toNat_inj.mp (by simpa using congrArg UInt64.toNat h)
  · intro h; rw [h]

set_option maxHeartbeats 1000000 in
/-- **One lane's extend step**, whatever its answer holds: it passes the ballot
    of the model's `good` at its byte, leaves memory, and sets `adv` from the
    answer. -/
theorem lane_extend_step {I : UInt64 → Prop} {ws : WState} {st st' : LSt} (hR : Rel I ws st)
    {ans : List Answer} {start : Nat} (hs : start ≤ st.w.obs.length)
    {adv p0 cand0 ml : String} {inStride endCap endCap' l : Nat} (hl : l < 32)
    (hecR : ws.regs "ecR" = UInt64.ofNat endCap) (hec1 : ws.regs "ec1" = UInt64.ofNat (endCap - 1))
    (hend : 1 ≤ endCap) (hendI : endCap ≤ inStride)
    (hcp : (ws.regs cand0).toNat < (ws.regs p0).toNat)
    (hov : (ws.regs p0).toNat + (ws.regs ml).toNat + 32 < 2 ^ 64)
    (hin : ∀ j, j < inStride → I (ws.regs "inBase" + UInt64.ofNat j))
    (hib : (ws.regs "inBase").toNat + inStride < 2 ^ 64)
    (hr : coopL ans start (addrOf .arena 0) (UInt64.ofNat l)
      (.coopExtendStep adv p0 cand0 ml inStride endCap') st = some st') :
    collCalls st'.w start = collCalls st.w start ++ [(.ballot, [boolV (extGood ws p0 cand0 ml inStride endCap l)])] ∧
      st'.w.mem = st.w.mem ∧
      st'.regs = (ws.setReg adv (ctz32 ((ansW ans (collCalls st.w start).length ^^^ 0xffffffff) &&& 0xffffffff) &&&
        0xffffffff)).regs := by
  simp only [coopL, hs, if_true] at hr
  obtain ⟨bP, hbP, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨bC, hbC, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain rfl := (Option.some.inj hr).symm
  rw [hR.regs] at hbP hbC
  have hpeN : (ws.regs p0 + (ws.regs ml + UInt64.ofNat l)).toNat = (ws.regs p0).toNat + (ws.regs ml).toNat + l := by
    rw [UInt64.toNat_add, UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show l < 2 ^ 64 by omega),
      Nat.mod_eq_of_lt (show (ws.regs ml).toNat + l < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
    omega
  have hok : ((decide (ws.regs p0 + (ws.regs ml + UInt64.ofNat l) < ws.regs "ecR")) &&
      (bP == bC)) = extGood ws p0 cand0 ml inStride endCap l := by
    have hE64 : endCap < 2 ^ 64 := by omega
    rw [hecR]
    simp only [extGood, AlgorithmLib.LZ4WarpSched.good]
    by_cases hlt : (ws.regs p0).toNat + (ws.regs ml).toNat + l < endCap
    · have hd1 : decide (ws.regs p0 + (ws.regs ml + UInt64.ofNat l) < UInt64.ofNat endCap) = true := by
        simp [UInt64.lt_iff_toNat_lt, hpeN, UInt64.toNat_ofNat_of_lt' hE64, hlt]
      have hd2 : decide ((ws.regs p0).toNat + ((ws.regs ml).toNat + l) < endCap) = true := by
        simp; omega
      rw [hd1, hd2, Bool.true_and, Bool.true_and]
      have hle : (ws.regs p0 + (ws.regs ml + UInt64.ofNat l) ≤ ws.regs "ec1") := by
        rw [hec1, UInt64.le_iff_toNat_le, hpeN, UInt64.toNat_ofNat_of_lt' (show endCap - 1 < 2 ^ 64 by omega)]; omega
      rw [if_pos hle] at hbP hbC
      -- the lane's byte
      have haP : ws.regs "inBase" + (ws.regs p0 + (ws.regs ml + UInt64.ofNat l)) =
          ws.regs "inBase" + UInt64.ofNat ((ws.regs p0).toNat + (ws.regs ml).toNat + l) := by
        rw [← hpeN, UInt64.ofNat_toNat]
      rw [haP, hR.gin _ (hin _ (by omega))] at hbP
      -- the candidate's byte
      have hsub : (ws.regs p0 + (ws.regs ml + UInt64.ofNat l) - ws.regs p0).toNat = (ws.regs ml).toNat + l := by
        rw [UInt64.toNat_sub_of_le _ _ (by rw [UInt64.le_iff_toNat_le, hpeN]; omega), hpeN]; omega
      have haC : ws.regs "inBase" + (ws.regs cand0 + (ws.regs p0 + (ws.regs ml + UInt64.ofNat l) - ws.regs p0)) =
          ws.regs "inBase" + UInt64.ofNat ((ws.regs cand0).toNat + (ws.regs ml).toNat + l) := by
        congr 1
        apply UInt64.toNat_inj.mp
        rw [UInt64.toNat_add, hsub,
          UInt64.toNat_ofNat_of_lt' (show (ws.regs cand0).toNat + (ws.regs ml).toNat + l < 2 ^ 64 by omega),
          Nat.mod_eq_of_lt (by omega)]
        omega
      rw [haC, hR.gin _ (hin _ (by omega))] at hbC
      obtain rfl := (Option.some.inj hbP).symm
      obtain rfl := (Option.some.inj hbC).symm
      have hta : ∀ x, x < inStride → (ws.regs "inBase" + UInt64.ofNat x).toNat = (ws.regs "inBase").toNat + x := by
        intro x hx
        rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show x < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
      rw [byte_beq, hta _ (by omega), hta _ (by omega)]
      simp only [AlgorithmLib.LZ4WarpFind.byte]
      rw [getD_inp (by omega), getD_inp (by omega)]
      simp only [Nat.add_assoc]
    · have hd1 : decide (ws.regs p0 + (ws.regs ml + UInt64.ofNat l) < UInt64.ofNat endCap) = false := by
        simp [UInt64.lt_iff_toNat_lt, hpeN, UInt64.toNat_ofNat_of_lt' hE64]; omega
      have hd2 : decide ((ws.regs p0).toNat + ((ws.regs ml).toNat + l) < endCap) = false := by
        simp; omega
      rw [hd1, hd2]; rfl
  refine ⟨?_, rfl, ?_⟩
  · show collCalls (callW st.w .ballot _) start = _
    rw [callW, collCalls_call hs, hR.regs, hok]; rfl
  · rw [hR.regs]; rfl

/-- **A lane's share of an extend step finishes** where the match starts before
    the end cap. -/
theorem lane_extend_some {I : UInt64 → Prop} {ws : WState} {st : LSt} (hR : Rel I ws st)
    {ans : List Answer} {start : Nat} (hs : start ≤ st.w.obs.length)
    {adv p0 cand0 ml : String} {inStride endCap endCap' l : Nat} (hl : l < 32)
    (hecR : ws.regs "ecR" = UInt64.ofNat endCap) (hec1 : ws.regs "ec1" = UInt64.ofNat (endCap - 1))
    (hend : 1 ≤ endCap) (hendI : endCap ≤ inStride)
    (hcp : (ws.regs cand0).toNat < (ws.regs p0).toNat)
    (hov : (ws.regs p0).toNat + (ws.regs ml).toNat + 32 < 2 ^ 64)
    (hin : ∀ j, j < inStride → I (ws.regs "inBase" + UInt64.ofNat j))
    (hib : (ws.regs "inBase").toNat + inStride < 2 ^ 64) (hpe : (ws.regs p0).toNat < endCap) :
    ∃ st', coopL ans start (addrOf .arena 0) (UInt64.ofNat l)
      (.coopExtendStep adv p0 cand0 ml inStride endCap') st = some st' := by
  have hpeN : (ws.regs p0 + (ws.regs ml + UInt64.ofNat l)).toNat = (ws.regs p0).toNat + (ws.regs ml).toNat + l := by
    rw [UInt64.toNat_add, UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show l < 2 ^ 64 by omega),
      Nat.mod_eq_of_lt (show (ws.regs ml).toNat + l < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
    omega
  have he1 : (ws.regs "ec1").toNat = endCap - 1 := by
    rw [hec1, UInt64.toNat_ofNat_of_lt' (show endCap - 1 < 2 ^ 64 by omega)]
  generalize hpc : (if ws.regs p0 + (ws.regs ml + UInt64.ofNat l) ≤ ws.regs "ec1" then
    ws.regs p0 + (ws.regs ml + UInt64.ofNat l) else ws.regs "ec1") = peC
  have hpc1 : peC.toNat < endCap ∧ (ws.regs p0).toNat ≤ peC.toNat := by
    rw [← hpc]
    split
    · rename_i hle; rw [UInt64.le_iff_toNat_le, hpeN, he1] at hle; rw [hpeN]; omega
    · rw [he1]; omega
  have hca : (ws.regs cand0 + (peC - ws.regs p0)).toNat < peC.toNat := by
    rw [UInt64.toNat_add, UInt64.toNat_sub_of_le _ _ (by rw [UInt64.le_iff_toNat_le]; omega),
      Nat.mod_eq_of_lt (by omega)]
    omega
  obtain ⟨bP, hbP⟩ := load_input_some hR hin (x := peC) (by omega)
  obtain ⟨bC, hbC⟩ := load_input_some hR hin (x := ws.regs cand0 + (peC - ws.regs p0)) (by omega)
  simp only [coopL, hs, if_true, hR.regs, Option.bind_eq_bind, hpc]
  exact bind_some_of hbP (bind_some_of hbC ⟨_, rfl⟩)

/-- The ballot's complement, within 32 bits, is the mask of the negated flags. -/
theorem maskN_compl (h : Nat → Bool) :
    ((UInt32.ofNat (maskN h)).toUInt64 ^^^ 0xffffffff) &&& 0xffffffff =
      (UInt32.ofNat (maskN fun k => !h k)).toUInt64 := by
  apply UInt64.toNat_inj.mp
  have h1 := maskN_lt h
  have h2 := maskN_lt (fun k => !h k)
  simp only [UInt64.toNat_and, UInt64.toNat_xor, UInt32.toNat_toUInt64, UInt32.toNat_ofNat_of_lt' h1,
    UInt32.toNat_ofNat_of_lt' h2, show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl]
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_and, Nat.testBit_xor, Nat.testBit_two_pow_sub_one]
  by_cases hi : i < 32
  · rw [maskN_testBit _ hi, maskN_testBit _ hi]; simp [hi]
  · rw [Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le h2 (Nat.pow_le_pow_right (by decide) (by omega)))]
    simp [hi]

/-- What the warp's extend step answers its ballot: the mask of the lanes'
    flags. -/
def extAns (ws : WState) (p0 cand0 ml : String) (inStride endCap : Nat) : List Answer :=
  [.val (.sc .i32 (UInt32.ofNat (maskN (extGood ws p0 cand0 ml inStride endCap))).toUInt64)]

set_option maxHeartbeats 2000000 in
/-- **A warp's lanes run an extend step with the warp**: answered the warp's
    answer (`extAns`), each lane ends holding the warp's registers, `adv` the
    model's count of leading good bytes, and has passed one more collective
    and no barrier or shared store; and the laws, from what all the lanes
    passed, give that answer back. -/
theorem extend_lockstep {I : UInt64 → Prop} {ws : WState} {sts sts' : Nat → LSt} {fin : Nat → World}
    {ans : Nat → List Answer} {start : Nat → Nat} {smem0 : ByteArray}
    {adv p0 cand0 ml : String} {inStride endCap K NB f : Nat}
    (hrun : ∀ l, l < 32 → coopL (ans l) (start l) (addrOf .arena 0) (UInt64.ofNat l)
      (.coopExtendStep adv p0 cand0 ml inStride endCap) (sts l) = some (sts' l))
    (hR : ∀ l, l < 32 → Rel I ws (sts l) ∧ start l ≤ (sts l).w.obs.length)
    (hfin : ∀ l, l < 32 → WExt (sts' l).w (fin l))
    {R : List Answer} (htape : ∀ l, l < 32 → (ans l).drop K = extAns ws p0 cand0 ml inStride endCap ++ R)
    (hK : ∀ l, l < 32 → (collCalls (sts l).w (start l)).length = K)
    (hNB : ∀ l, l < 32 → (collCalls (sts l).w (start l)).countP (·.1 == .barrier) = NB)
    (htag : ∀ l, l < 32 → ∀ x ∈ sharedStores (sts l).w (start l), x.1 < NB)
    (hecR : ws.regs "ecR" = UInt64.ofNat endCap) (hec1 : ws.regs "ec1" = UInt64.ofNat (endCap - 1))
    (hend : 1 ≤ endCap) (hendI : endCap ≤ inStride)
    (hcp : (ws.regs cand0).toNat < (ws.regs p0).toNat)
    (hov : (ws.regs p0).toNat + (ws.regs ml).toNat + 32 < 2 ^ 64)
    (hin : ∀ j, j < inStride → I (ws.regs "inBase" + UInt64.ofNat j))
    (hib : (ws.regs "inBase").toNat + inStride < 2 ^ 64) :
    (∀ l, l < 32 → Rel I ((WStmt.coopExtendStep adv p0 cand0 ml inStride endCap).eval f ws) (sts' l) ∧
      (collCalls (sts' l).w (start l)).length = K + 1 ∧
      (collCalls (sts' l).w (start l)).countP (·.1 == .barrier) = NB ∧
      ∀ x ∈ sharedStores (sts' l).w (start l), x.1 < NB) ∧
    ((WStmt.coopExtendStep adv p0 cand0 ml inStride endCap).eval f ws).smem = ws.smem ∧
    ∀ l, l < 32 → ∀ i a, (extAns ws p0 cand0 ml inStride endCap)[i]? = some a →
      LawAt (fun t => collCalls (fin t) (start t))
        (snapshots 32 smem0 (fun t => collCalls (fin t) (start t)) (fun t => sharedStores (fin t) (start t)))
        smem0 l (K + i) a := by
  have hstep := fun l (hl : l < 32) => lane_extend_step (hR l hl).1 (hR l hl).2 hl hecR hec1 hend hendI hcp hov
    hin hib (hrun l hl)
  generalize hF : (fun t => collCalls (fin t) (start t)) = F
  have hext : ∀ l, l < 32 → start l ≤ (sts' l).w.obs.length := fun l hl =>
    Nat.le_trans (hR l hl).2 (coopL_ext _ _ _ (hrun l hl)).1.len
  have hFK : ∀ l, l < 32 → (F l)[K]? = some (.ballot, [boolV (extGood ws p0 cand0 ml inStride endCap l)]) := by
    intro l hl
    obtain ⟨R, hR'⟩ := collCalls_prefix (hext l hl) (hfin l hl)
    rw [← hF]
    show (collCalls (fin l) (start l))[K]? = _
    rw [hR', (hstep l hl).1, List.append_assoc, List.getElem?_append_right (by rw [hK l hl]; exact Nat.le_refl _),
      hK l hl, Nat.sub_self]
    rfl
  generalize hg : extGood ws p0 cand0 ml inStride endCap = g at hFK
  have hMk : ∀ l, l < 32 → ansW (ans l) K = (UInt32.ofNat (maskN g)).toUInt64 := by
    intro l hl
    have h0 : (ans l)[K]? = some (.val (.sc .i32 (UInt32.ofNat (maskN g)).toUInt64)) := by
      rw [← hg]; simpa using getElem?_of_drop (i := 0) (htape l hl) rfl
    exact ansW_of h0 (by simpa using (UInt32.ofNat (maskN g)).toNat_lt)
  have hlaws : ∀ l, l < 32 → ∀ i a, (extAns ws p0 cand0 ml inStride endCap)[i]? = some a →
      LawAt F (snapshots 32 smem0 F (fun t => sharedStores (fin t) (start t))) smem0 l (K + i) a := by
    intro l hl i a hi
    rw [extAns, hg] at hi
    match i, hi with
    | 0, hi => cases hi; exact ⟨_, _, hFK l hl, answerStep_ballot hl hFK _⟩
    | _ + 1, hi => simp at hi
  have hadv : ctz32 (((UInt32.ofNat (maskN g)).toUInt64 ^^^ 0xffffffff) &&& 0xffffffff) &&& 0xffffffff =
      UInt64.ofNat (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (ws.regs p0).toNat (ws.regs cand0).toNat endCap (ws.regs ml).toNat 32) := by
    rw [maskN_compl, ctz32_mask, firstSetNat_warp, ← hg]
    rw [← AlgorithmLib.LZ4WarpSched.firstHit_leadingGood]
    have hle := firstSetNat_le (fun k => !extGood ws p0 cand0 ml inStride endCap k)
    rw [firstSetNat_warp] at hle
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_and, UInt64.toNat_ofNat_of_lt' (Nat.lt_of_le_of_lt hle (by decide)),
      show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
      Nat.mod_eq_of_lt (by omega)]
  have hev : (WStmt.coopExtendStep adv p0 cand0 ml inStride endCap).eval f ws =
      ws.setReg adv (UInt64.ofNat (AlgorithmLib.LZ4WarpSched.leadingGood
        (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (ws.regs p0).toNat (ws.regs cand0).toNat endCap
        (ws.regs ml).toNat 32)) := by
    rw [WStmt.eval.eq_12]; rfl
  refine ⟨fun l hl => ?_, by rw [hev]; rfl, hlaws⟩
  obtain ⟨hc, hmem, hregs⟩ := hstep l hl
  rw [hK l hl, hMk l hl, hadv] at hregs
  refine ⟨⟨by rw [hregs, hev], ?_, fun a ha => ?_⟩, ?_, ?_, ?_⟩
  · rw [hev, hmem]; exact (hR l hl).1.smem
  · rw [hmem, (hR l hl).1.gin a ha, hev]; rfl
  · rw [hc]; simp [hK l hl]
  · rw [hc, List.countP_append, hNB l hl]; simp [show (Intr.ballot == Intr.barrier) = false by decide]
  · intro x hx
    have hsh : sharedStores (sts' l).w (start l) = sharedStores (sts l).w (start l) := by
      have hr := hrun l hl
      simp only [coopL, (hR l hl).2, if_true] at hr
      obtain ⟨_, -, hr⟩ := Option.bind_eq_some_iff.mp hr
      obtain ⟨_, -, hr⟩ := Option.bind_eq_some_iff.mp hr
      obtain h' := (Option.some.inj hr).symm
      rw [h']
      exact sharedStores_call (hR l hl).2 (by decide) _
    rw [hsh] at hx; exact htag l hl x hx

/-- What lane `l` passes in the warp's extend step: the ballot of its flag. -/
def extCalls (ws : WState) (p0 cand0 ml : String) (inStride endCap l : Nat) : List (Intr × List V) :=
  [(.ballot, [boolV (extGood ws p0 cand0 ml inStride endCap l)])]

set_option maxHeartbeats 2000000 in
/-- **A lane's share of an extend step runs with the warp**: answered the
    warp's answer (`extAns`), the lane finishes holding the warp's registers,
    its collective the warp's for it (`extCalls`), no shared store. -/
theorem lane_extend {I : UInt64 → Prop} {ws : WState} {st : LSt} (hR : Rel I ws st)
    {ans : List Answer} {start : Nat} (hs : start ≤ st.w.obs.length)
    {adv p0 cand0 ml : String} {inStride endCap l f : Nat} (hl : l < 32)
    (hecR : ws.regs "ecR" = UInt64.ofNat endCap) (hec1 : ws.regs "ec1" = UInt64.ofNat (endCap - 1))
    (hend : 1 ≤ endCap) (hendI : endCap ≤ inStride)
    (hcp : (ws.regs cand0).toNat < (ws.regs p0).toNat)
    (hov : (ws.regs p0).toNat + (ws.regs ml).toNat + 32 < 2 ^ 64)
    (hin : ∀ j, j < inStride → I (ws.regs "inBase" + UInt64.ofNat j))
    (hib : (ws.regs "inBase").toNat + inStride < 2 ^ 64) (hpe : (ws.regs p0).toNat < endCap) {R : List Answer}
    (htape : ans.drop (collCalls st.w start).length = extAns ws p0 cand0 ml inStride endCap ++ R) :
    ∃ st', coopL ans start (addrOf .arena 0) (UInt64.ofNat l)
        (.coopExtendStep adv p0 cand0 ml inStride endCap) st = some st' ∧
      Rel I ((WStmt.coopExtendStep adv p0 cand0 ml inStride endCap).eval f ws) st' ∧
      collCalls st'.w start = collCalls st.w start ++ extCalls ws p0 cand0 ml inStride endCap l ∧
      sharedStores st'.w start = sharedStores st.w start ∧
      ans.drop (collCalls st'.w start).length = R ∧ Tracks st.w st'.w := by
  obtain ⟨st', hc⟩ := lane_extend_some (ans := ans) (adv := adv) (endCap' := endCap) hR hs hl hecR hec1 hend hendI
    hcp hov hin hib hpe
  obtain ⟨hcl, hmem, hregs⟩ := lane_extend_step hR hs hl hecR hec1 hend hendI hcp hov hin hib hc
  have h0 : ans[(collCalls st.w start).length]? =
      some (.val (.sc .i32 (UInt32.ofNat (maskN (extGood ws p0 cand0 ml inStride endCap))).toUInt64)) := by
    simpa using getElem?_of_drop (i := 0) htape rfl
  have hMk : ansW ans (collCalls st.w start).length =
      (UInt32.ofNat (maskN (extGood ws p0 cand0 ml inStride endCap))).toUInt64 :=
    ansW_of h0 (by simpa using (UInt32.ofNat (maskN (extGood ws p0 cand0 ml inStride endCap))).toNat_lt)
  have hadv : ctz32 (((UInt32.ofNat (maskN (extGood ws p0 cand0 ml inStride endCap))).toUInt64 ^^^ 0xffffffff) &&&
      0xffffffff) &&& 0xffffffff =
      UInt64.ofNat (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (ws.regs p0).toNat (ws.regs cand0).toNat endCap (ws.regs ml).toNat 32) := by
    rw [maskN_compl, ctz32_mask, firstSetNat_warp]
    rw [← AlgorithmLib.LZ4WarpSched.firstHit_leadingGood]
    have hle := firstSetNat_le (fun k => !extGood ws p0 cand0 ml inStride endCap k)
    rw [firstSetNat_warp] at hle
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_and, UInt64.toNat_ofNat_of_lt' (Nat.lt_of_le_of_lt hle (by decide)),
      show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
      Nat.mod_eq_of_lt (by omega)]
  have hev : (WStmt.coopExtendStep adv p0 cand0 ml inStride endCap).eval f ws =
      ws.setReg adv (UInt64.ofNat (AlgorithmLib.LZ4WarpSched.leadingGood
        (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (ws.regs p0).toNat (ws.regs cand0).toNat endCap
        (ws.regs ml).toNat 32)) := by
    rw [WStmt.eval.eq_12]; rfl
  rw [hMk, hadv] at hregs
  have hsh : sharedStores st'.w start = sharedStores st.w start := by
    have hr := hc
    simp only [coopL, hs, if_true] at hr
    obtain ⟨_, -, hr⟩ := Option.bind_eq_some_iff.mp hr
    obtain ⟨_, -, hr⟩ := Option.bind_eq_some_iff.mp hr
    rw [← Option.some.inj hr]
    exact sharedStores_call hs (by decide) _
  refine ⟨st', hc, ⟨by rw [hregs, hev], ?_, fun a ha => ?_⟩, hcl, hsh, ?_,
    Tracks.of_nobyte ((coopL_ext _ _ _ hc).2 rfl) fun a _ => by rw [hmem]⟩
  · rw [hev, hmem]; exact hR.smem
  · rw [hmem, hR.gin a ha, hev]; rfl
  · rw [hcl, List.length_append, ← List.drop_drop, htape]; rfl

/-- **The laws give the extend step's answer back**: where every lane has
    passed the warp's extend step as `extCalls` says after `K` collectives, the
    laws answer each lane's ballot with the warp's answer (`extAns`). -/
theorem extend_laws {ws : WState} {p0 cand0 ml : String} {inStride endCap K : Nat} {smem0 : ByteArray}
    {F : Nat → List (Intr × List V)} {S : Nat → List (Nat × UInt64 × Nat × UInt64)}
    (hF : ∀ l, l < 32 → ∃ P Q, F l = P ++ extCalls ws p0 cand0 ml inStride endCap l ++ Q ∧ P.length = K) :
    ∀ l, l < 32 → ∀ i a, (extAns ws p0 cand0 ml inStride endCap)[i]? = some a →
      LawAt F (snapshots 32 smem0 F S) smem0 l (K + i) a := by
  have hFK : ∀ l, l < 32 → (F l)[K]? = some (.ballot, [boolV (extGood ws p0 cand0 ml inStride endCap l)]) := by
    intro l hl
    obtain ⟨P, Q, hP, hK⟩ := hF l hl
    rw [hP, List.append_assoc, List.getElem?_append_right (by omega), hK, Nat.sub_self]; rfl
  intro l hl i a hi
  match i, hi with
  | 0, hi => cases hi; exact ⟨_, _, hFK l hl, answerStep_ballot hl hFK _⟩
  | _ + 1, hi => simp [extAns] at hi

end Lz4Code
