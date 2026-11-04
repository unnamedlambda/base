module
public import Lz4.CodeWarpRun
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeThread` — one thread of the compressor

A thread of the compressor (`compressor`): its block's warp, its lane, the
input and output bases of the warp's block, its share of clearing the table
(`clearL`: entries `lane, lane + 32, …`), the barrier, and the warp language's
body from registers holding those values (`codeOf_lane_coop`).
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.HProg.Hoare

/-- A lane's share of clearing the table: entries `ci` from `ci0` by 32 below
    `n`, `f` trips of fuel. -/
def clearL (sm tbl n msk : UInt64) : Nat → UInt64 → World → Option World
  | 0, ci, w => if ci < n then none else some w
  | f + 1, ci, w =>
      if ci < n then do
        let w1 ← stw w (sm + ((tbl + (ci <<< c64 1)) &&& msk)) 2 (c64 0 &&& 0xffff)
        clearL sm tbl n msk f (ci + c64 32) w1
      else some w

/-- **A statement's flags are set only by comparisons.** -/
theorem wf_flagRegs (s : WStmt) : Wf (flagRegs s) s := by
  intro r hr
  cases hc : (flagRegs s).contains r with
  | false => rfl
  | true =>
      exfalso
      have hm : r ∈ flagRegs s := by simpa using hc
      unfold flagRegs at hm
      rw [mem_dedup, List.mem_filterMap] at hm
      obtain ⟨⟨r', b⟩, hmem, hsome⟩ := hm
      dsimp only at hsome
      split at hsome
      · rename_i hcond
        obtain rfl := Option.some.inj hsome
        simp only [Bool.and_eq_true, Bool.not_eq_true'] at hcond
        have : (assigns s).contains (r', false) = true := by simpa using hr
        rw [hcond.2] at this; cases this
      · cases hsome

/-- The registers a thread's body starts from: its block's input and output
    bases, every other zero: the same in every lane. -/
def regs0 (inBase outBase : UInt64) : String → UInt64 := fun r =>
  if r = "inBase" then inBase else if r = "outBase" then outBase else 0

section
variable {cfg : Cfg} {Γ : Env} {w : World}

theorem clearL_done {sm tbl n msk ci : UInt64} {w : World} (h : ¬ ci < n) : ∀ k, clearL sm tbl n msk k ci w = some w
  | 0 => by simp [clearL, h]
  | _ + 1 => by simp [clearL, h]

/-- **The table clear's code ends as a lane's share of it.** -/
theorem wpE_clear {smS tblS laneS zS : Slot .i64} {smv tblv lv : UInt64} {nk mk : Int} {F : Nat} {wEnd : World}
    {Q : Vals Slot [] → Env → World → Prop}
    (hsm : Γ[smS]? = some (.sc .i64 smv)) (htbl : Γ[tblS]? = some (.sc .i64 tblv))
    (hlane : Γ[laneS]? = some (.sc .i64 lv)) (hz : Γ[zS]? = some (.sc .i64 (c64 0)))
    (hrun : clearL smv tblv (c64 nk) (c64 mk) F lv w = some wEnd)
    (hQ : ∀ Γb vs nE, Extends Γ Γb → Γ.size ≤ nE → Q (carriesFrom nE []) (bindAt Γb nE vs) wEnd) :
    wpE cfg (wloop1 (exitTys := []) laneS (fun ci => do pure (contIf .ult ci (← iconst64 nk), %[], ()))
      (fun ci _ => do stsh smS (← band (← iadd tblS (← ishlImm ci 1)) (← iconst64 mk)) zS; pure %[← iaddImm ci 32]))
      Q Γ w := by
  simp only [wloop1]
  let R (cs : List V) (wi : World) (k : Nat) : Prop :=
    ∃ i, cs = [.sc .i64 i] ∧ clearL smv tblv (c64 nk) (c64 mk) k i wi = some wEnd
  have hread : ∀ (Γ1 : Env) (i : UInt64), Extends (bindAt Γ Γ.size [.sc .i64 i]) Γ1 →
      Γ1[Γ.size]? = some (.sc .i64 i) := fun Γ1 i hx => hx.get (by simpa using bindAt_at Γ Γ.size [.sc .i64 i] 0)
  have hkeep : ∀ (Γ1 : Env) (cs : List V) {x : Nat} {v : V}, Extends (bindAt Γ Γ.size cs) Γ1 → Γ[x]? = some v →
      Γ1[x]? = some v := fun Γ1 cs x v hx h => hx.get (get_bindAt (Nat.le_refl _) h)
  refine wpE_wloopN (fun cs wi => ∃ k, k ≤ F ∧ R cs wi k) (fun cs wi => least (R cs wi) F)
    (fun cs wi a Γ1 w1 => Extends (bindAt Γ Γ.size cs) Γ1 ∧ ∃ i, cs = [.sc .i64 i] ∧
      a = (contIf (ty := .i64) .ult Γ.size (bindAt Γ Γ.size cs).size, %[], ()) ∧
      Γ1 = (bindAt Γ Γ.size cs).push (.sc .i64 (c64 nk)) ∧ w1 = wi ∧ ∃ k, k ≤ F ∧
        clearL smv tblv (c64 nk) (c64 mk) k i wi = some wEnd)
    (fun _ w => w = wEnd) ?_ ?_ ?_ ?_ ?_ ?_
  · exact ⟨_, by simp [Vals.slots, hlane], F, Nat.le_refl _, lv, rfl, hrun⟩
  · rintro wi cs ⟨k, hk, i, rfl, hr⟩
    refine wpE_bind (wpE_iconst64 ?_)
    exact wpE_pure ⟨Extends.push _ _, Extends.push _ _, i, rfl, rfl, rfl, rfl, k, hk, hr⟩
  · rintro cs wi a Γ1 w1 ⟨hx, i, rfl, rfl, rfl, rfl, -⟩
    exact ⟨_, _, by
      simp only [contIf]
      simp [evalOp, Sem.get, (Extends.push _ _).get (hread _ i (Extends.refl _)), zipIntCmp,
        ClifTy.isInt]; rfl⟩
  · rintro cs wi a Γ1 w1 t f ⟨hx, i, rfl, rfl, rfl, rfl, k, hk, hr⟩ hev hexit
    simp only [contIf] at hev hexit
    rw [show evalOp w1.mem ((bindAt Γ Γ.size [.sc .i64 i]).push (.sc .i64 (c64 nk)))
        (.icmp .ult Γ.size (bindAt Γ Γ.size [.sc .i64 i]).size) = some (boolV (cmpInt .ult .i64 i (c64 nk))) by
      simp [evalOp, Sem.get, (Extends.push _ _).get (hread _ i (Extends.refl _)), zipIntCmp,
        ClifTy.isInt]] at hev
    have hlt : ¬ i < c64 nk := by
      intro h
      simp [h, boolV, cmpInt] at hev
      obtain ⟨-, rfl⟩ := hev; simp at hexit
    rw [clearL_done hlt k] at hr
    exact ⟨[], rfl, Option.some.inj hr⟩
  · rintro cs wi a Γ1 wb t f ⟨hx, i, rfl, rfl, hΓ1, rfl, k, hk, hr⟩ _ hev hcont
    simp only [contIf] at hev hcont
    rw [hΓ1] at hev
    rw [show evalOp wb.mem ((bindAt Γ Γ.size [.sc .i64 i]).push (.sc .i64 (c64 nk)))
        (.icmp .ult Γ.size (bindAt Γ Γ.size [.sc .i64 i]).size) = some (boolV (cmpInt .ult .i64 i (c64 nk))) by
      simp [evalOp, Sem.get, (Extends.push _ _).get (hread _ i (Extends.refl _)), zipIntCmp,
        ClifTy.isInt]] at hev
    have hlt : i < c64 nk := by
      refine Classical.byContradiction fun h => ?_
      simp [h, boolV, cmpInt] at hev
      obtain ⟨-, rfl⟩ := hev; simp at hcont
    obtain ⟨hR0, hmin⟩ := least_spec (P := R [.sc .i64 i] wb) ⟨i, rfl, hr⟩ hk
    generalize hk0e : least (R [.sc .i64 i] wb) F = k0 at hR0 hmin
    obtain ⟨i', hi', hrun0⟩ := hR0
    obtain rfl : i' = i := by simpa using hi'.symm
    have hk0 : k0 ≤ k := hmin k ⟨i', rfl, hr⟩
    cases k0 with
    | zero => simp [clearL, hlt] at hrun0
    | succ k' =>
    simp only [clearL, hlt, if_true] at hrun0
    obtain ⟨w1, hw1, hrun1⟩ := Option.bind_eq_some_iff.mp hrun0
    have hxa := (hx.push_of (.sc t f))
    have hxb : Extends (Γ1.push (.sc t f)) (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) [.sc .i64 i']) :=
      (Extends.refl _).bindAt_of (by simp) _
    have hI : (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) [.sc .i64 i'])[Γ1.size + 1]? = some (.sc .i64 i') := by
      simpa using bindAt_at (Γ1.push (.sc t f)) (Γ1.size + 1) [.sc .i64 i'] 0
    have hkb : ∀ {x : Nat} {v : V}, Γ[x]? = some v →
        (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) [.sc .i64 i'])[x]? = some v := fun h =>
      hxb.get (hxa.get (get_bindAt (Nat.le_refl _) h))
    simp only [carriesFrom, Vals.head]
    refine wpE_bind (xishlImm hI fun vS Γ5 hx5 hS => ?_)
    refine wpE_bind (xiadd (hx5.get (hkb htbl)) hS fun vA Γ6a hx6a hA => ?_)
    refine wpE_bind (xiconst64 fun vM Γ6b hx6b hM => ?_)
    refine wpE_bind (xband64 (hx6b.get hA) hM fun vB Γ6 hx6c hB => ?_)
    have hx6 : Extends Γ5 Γ6 := hx6a.trans (hx6b.trans hx6c)
    refine wpE_bind (wpE_stsh ((hx5.trans hx6).get (hkb hsm)) hB ((hx5.trans hx6).get (hkb hz)) hw1
      fun Γ7 hx7 => ?_)
    refine wpE_bind (xiaddImm ((hx5.trans (hx6.trans hx7)).get hI) fun vI Γ8 hx8 hI8 => ?_)
    have hR2 : R [.sc .i64 (i' + c64 32)] w1 k' := ⟨_, rfl, hrun1⟩
    refine wpE_pure ⟨[.sc .i64 (i' + c64 32)], by simp [Vals.slots, hI8], ⟨k', by omega, hR2⟩, ?_⟩
    have := (least_spec hR2 (show k' ≤ F by omega)).2 k' hR2
    show least (R [.sc .i64 (i' + c64 32)] w1) F < least (R [.sc .i64 i'] wb) F
    rw [hk0e]; omega
  · rintro Γb vs w' nE hxb hle rfl
    exact hQ Γb vs nE hxb hle

theorem clearL_ext {sm tbl n msk : UInt64} : ∀ k ci (w w' : World), clearL sm tbl n msk k ci w = some w' → WExt w w'
  | 0, ci, w, w', h => by
      simp only [clearL] at h
      split at h
      · cases h
      · obtain rfl := (Option.some.inj h).symm; exact WExt.refl _
  | k + 1, ci, w, w', h => by
      simp only [clearL] at h
      split at h
      · obtain ⟨w1, h1, h2⟩ := Option.bind_eq_some_iff.mp h
        exact (stw_ext h1).trans (clearL_ext k _ w1 w' h2)
      · obtain rfl := (Option.some.inj h).symm; exact WExt.refl _

set_option maxHeartbeats 4000000 in
/-- **A thread of the compressor ends as its lane does**: its block's warp and
    lane, its share of the table clear (`clearL`), the barrier, and its lane's
    run of the body from registers holding the warp's input and output bases
    and its lane. -/
theorem wpE_thread {ans : List Answer} {start : Nat} (hA : AnswersFrom ans start cfg) (hW : Words ans)
    {numBlocks inStride outStride lenOff hashLog cta t F f : Nat} {inP outA : UInt64} {w0 w1 : World}
    {b : ByteArray} {st' : LSt}
    (ht : t < 32) (hcta : cta < numBlocks) (hc32 : cta * 32 + 32 < 2 ^ 32) (hnb : numBlocks < 2 ^ 63)
    (hs : start ≤ w0.obs.length)
    (hflags : ∀ r, r ∈ ["inBase", "outBase", "tbl", "op", "searchPos", "litAnchor"] →
      (flagRegs (compressorBodyWStmt inStride lenOff hashLog)).contains r = false)
    (hclear : clearL (addrOf .arena 0) (c64 0) (c64 (2 ^ hashLog)) (c64 ((2 * 2 ^ hashLog - 1 : Nat) : Int)) F (UInt64.ofNat t) w0 = some w1)
    (hbar : ans[(collCalls w1 start).length]? = some (.smem b))
    (hrun : laneRun (coopL ans start (addrOf .arena 0) (UInt64.ofNat t)) (addrOf .arena 0) f
      (compressorBodyWStmt inStride lenOff hashLog)
      ⟨regs0 (inP + UInt64.ofNat cta * c64 inStride)
        (inP + c64 (numBlocks * inStride + copySlack) + UInt64.ofNat cta * c64 outStride),
       { obsCall w1 (.intr .barrier) [] with mem := { (obsCall w1 (.intr .barrier) []).mem with arena := b } }⟩ =
        some st') :
    wpE cfg (compressor numBlocks inStride outStride lenOff hashLog : Prog Slot Lvl Unit) (fun _ _ w' => w' = st'.w)
      (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t).toArray w0 := by
  have hcta' : cta < 2 ^ 32 := by omega
  have ht' : t < 2 ^ 32 := by omega
  have h0 : (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t).toArray[0]? =
      some (.sc .i32 (UInt32.ofNat cta).toUInt64) := rfl
  have h1 : (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t).toArray[1]? =
      some (.sc .i32 (UInt32.ofNat t).toUInt64) := rfl
  have h2 : (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t).toArray[2]? =
      some (.sc .i32 (UInt32.ofNat 32).toUInt64) := rfl
  have h3 : (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t).toArray[3]? =
      some (.sc .i64 (addrOf .arena 0)) := rfl
  have h4 : (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t).toArray[4]? = some (.sc .i64 inP) := rfl
  have hu : ∀ n : Nat, n < 2 ^ 32 → (UInt32.ofNat n).toUInt64 &&& 0xffffffff = UInt64.ofNat n := by
    intro n hn
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_and, UInt32.toNat_toUInt64, UInt32.toNat_ofNat_of_lt' hn,
      show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hn,
      UInt64.toNat_ofNat_of_lt' (show n < 2 ^ 64 by omega)]
  generalize hG0 : (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t).toArray = G0 at h0 h1 h2 h3 h4
  unfold compressor
  refine wpE_entryParams ?_
  simp only [carriesFrom, Vals.head, Vals.snd, Vals.thd, Vals.tail, Vals.drop, List.drop]
  refine wpE_bind (xuext32 h0 fun vC G1 hx1 hC => ?_)
  refine wpE_bind (xuext32 (hx1.get h1) fun vT G2 hx2 hT => ?_)
  refine wpE_bind (xuext32 ((hx1.trans hx2).get h2) fun vN G3 hx3 hN => ?_)
  rw [hu _ hcta'] at hC; rw [hu _ ht'] at hT; rw [hu _ (by decide)] at hN
  refine wpE_bind (xmul (by liftf hC) hN fun vM G4 hx4 hM => ?_)
  refine wpE_bind (xiadd hM (by liftf hT) fun vS G5 hx5 hS => ?_)
  refine wpE_bind (xushrImm hS fun vG G6 hx6 hG => ?_)
  have hgw : (UInt64.ofNat cta * UInt64.ofNat 32 + UInt64.ofNat t) >>> c64 5 = UInt64.ofNat cta := by
    have e1 : (UInt64.ofNat cta * UInt64.ofNat 32 + UInt64.ofNat t).toNat = cta * 32 + t := by
      rw [UInt64.toNat_add, UInt64.toNat_mul]; simp only [UInt64.toNat_ofNat']; omega
    apply UInt64.toNat_inj.mp
    rw [show c64 5 = UInt64.ofNat 5 by decide, UInt64.toNat_shiftRight, e1, UInt64.toNat_ofNat',
      UInt64.toNat_ofNat', show 5 % 2 ^ 64 % 64 = 5 from rfl, Nat.shiftRight_eq_div_pow]
    omega
  rw [hgw] at hG
  refine wpE_bind (xiconst64 fun v31 G7 hx7 h31 => ?_)
  refine wpE_bind (xband64 (by liftf hT) h31 fun vL G8 hx8 hL => ?_)
  have hlv : UInt64.ofNat t &&& c64 31 = UInt64.ofNat t := by
    apply UInt64.toNat_inj.mp
    rw [show c64 31 = UInt64.ofNat (2 ^ 5 - 1) by decide, UInt64.toNat_and,
      UInt64.toNat_ofNat_of_lt' (show t < 2 ^ 64 by omega), UInt64.toNat_ofNat_of_lt' (by decide),
      Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt (by omega)]
  rw [hlv] at hL
  refine wpE_bind (xiconst64 fun vNB G9 hx9 hNB => ?_)
  have hs1 : start ≤ w1.obs.length := Nat.le_trans hs (clearL_ext _ _ _ _ hclear).len
  refine wpE_whenT (fun _ w' => w' = st'.w) rfl (by liftf hG) hNB ?_ ?_ (fun _ _ h _ _ => h)
  · intro _
    have hxT : Extends G9 (G9.push (boolV true)) := Extends.push _ _
    refine wpE_bind (xiaddImm (by liftf h4) fun vO H1 hy1 hO => ?_)
    refine wpE_bind (xiconst64 fun vIs H2 hy2 hIs => ?_)
    refine wpE_bind (xmul (by liftf hG) hIs fun vM2 H3 hy3 hM2 => ?_)
    refine wpE_bind (xiadd (by liftf h4) hM2 fun vIB H4 hy4 hIB => ?_)
    refine wpE_bind (xiconst64 fun vOs H5 hy5 hOs => ?_)
    refine wpE_bind (xmul (by liftf hG) hOs fun vM3 H6 hy6 hM3 => ?_)
    refine wpE_bind (xiadd (by liftf hO) hM3 fun vOB H7 hy7 hOB => ?_)
    refine wpE_bind (xiconst64 fun vTb H8 hy8 hTb => ?_)
    refine wpE_bind (xiconst64 fun vZ H9 hy9 hZ => ?_)
    refine wpE_bind (wpE_clear (by liftf h3) (by liftf hTb) (by liftf hL) hZ hclear fun Γb vs nE hxb hle => ?_)
    have hxc : Extends H9 (bindAt Γb nE vs) := hxb.bindAt hle vs
    refine wpE_bind (wpE_bar hA barrier rfl hs1 rfl hbar ?_)
    refine wpE_bind (wpE_val (v := .sc .i8 0) (by simp [Op'.erase, evalOp]; rfl) ?_)
    have hxN : Extends (bindAt Γb nE vs) ((bindAt Γb nE vs).push (.sc .i8 0)) := Extends.push _ _
    have hno : ((bindAt Γb nE vs).push (.sc .i8 0))[(bindAt Γb nE vs).size]? = some (.sc .i8 0) := get_push_last _ _
    have hz0 : c64 0 = 0 := by decide
    rw [hz0] at hZ hTb
    have hfl := fun r h => hflags r h
    have hEnv : EnvOk (flagRegs (compressorBodyWStmt inStride lenOff hashLog))
        { vals := [("inBase", vIB), ("outBase", vOB), ("tbl", vTb), ("op", vZ), ("searchPos", vZ),
          ("litAnchor", vZ)] }
        (regs0 (inP + UInt64.ofNat cta * c64 inStride)
          (inP + c64 (numBlocks * inStride + copySlack) + UInt64.ofNat cta * c64 outStride))
        ((bindAt Γb nE vs).push (.sc .i8 0)) := by
      have gIB : ((bindAt Γb nE vs).push (.sc .i8 0))[vIB]? = some (.sc .i64 (inP + UInt64.ofNat cta * c64 inStride)) :=
        by liftf hIB
      have gOB : ((bindAt Γb nE vs).push (.sc .i8 0))[vOB]? =
          some (.sc .i64 (inP + c64 (numBlocks * inStride + copySlack) + UInt64.ofNat cta * c64 outStride)) :=
        by liftf hOB
      have gTb : ((bindAt Γb nE vs).push (.sc .i8 0))[vTb]? = some (.sc .i64 0) := by liftf hTb
      have gL : ((bindAt Γb nE vs).push (.sc .i8 0))[vL]? = some (.sc .i64 (UInt64.ofNat t)) := by liftf hL
      have gZ : ((bindAt Γb nE vs).push (.sc .i8 0))[vZ]? = some (.sc .i64 0) := by liftf hZ
      refine ⟨fun r v hv => ?_, fun r v hv => (by simp at hv), fun r hr => ?_, fun r hr _ => ?_, fun r hr hn => ?_,
        fun r v hv => (by simp at hv), fun r v hv => ?_⟩
      · by_cases e1 : r = "inBase"
        · subst e1; simp [List.lookup] at hv; subst hv; simpa [regs0] using gIB
        by_cases e2 : r = "outBase"
        · subst e2; simp [List.lookup] at hv; subst hv; simpa [regs0] using gOB
        by_cases e3 : r = "tbl"
        · subst e3; simp [List.lookup] at hv; subst hv; simpa [regs0] using gTb
        by_cases e5 : r = "op"
        · subst e5; simp [List.lookup] at hv; subst hv; simpa [regs0] using gZ
        by_cases e6 : r = "searchPos"
        · subst e6; simp [List.lookup] at hv; subst hv; simpa [regs0] using gZ
        by_cases e7 : r = "litAnchor"
        · subst e7; simp [List.lookup] at hv; subst hv; simpa [regs0] using gZ
        rw [lookup_cons_ne e1, lookup_cons_ne e2, lookup_cons_ne e3, lookup_cons_ne e5, lookup_cons_ne e6, lookup_cons_ne e7] at hv; simp at hv
      · have h1 : r ≠ "inBase" := fun e => by subst e; rw [hfl _ (by simp)] at hr; cases hr
        have h2 : r ≠ "outBase" := fun e => by subst e; rw [hfl _ (by simp)] at hr; cases hr
        simp [regs0, h1, h2]
      · have h1 : r ≠ "inBase" := fun e => by subst e; rw [hfl _ (by simp)] at hr; cases hr
        have h2 : r ≠ "outBase" := fun e => by subst e; rw [hfl _ (by simp)] at hr; cases hr
        simp [regs0, h1, h2]
      · have h1 : r ≠ "inBase" := fun e => by subst e; simp [List.lookup] at hn
        have h2 : r ≠ "outBase" := fun e => by subst e; simp [List.lookup] at hn
        simp [regs0, h1, h2]
      · have hmem : r ∈ ["inBase", "outBase", "tbl", "op", "searchPos", "litAnchor"] := by
          by_cases e1 : r = "inBase"
          · simp [e1]
          by_cases e2 : r = "outBase"
          · simp [e2]
          by_cases e3 : r = "tbl"
          · simp [e3]
          by_cases e5 : r = "op"
          · simp [e5]
          by_cases e6 : r = "searchPos"
          · simp [e6]
          by_cases e7 : r = "litAnchor"
          · simp [e7]
          rw [lookup_cons_ne e1, lookup_cons_ne e2, lookup_cons_ne e3, lookup_cons_ne e5, lookup_cons_ne e6, lookup_cons_ne e7] at hv; simp at hv
        exact hfl r hmem
    have hSl : Slots (3 : Nat) vL (bindAt Γb nE vs).size (addrOf .arena 0) (UInt64.ofNat t)
        ((bindAt Γb nE vs).push (.sc .i8 0)) := ⟨by liftf h3, by liftf hL, hno⟩
    refine wpE_bind (codeOf_lane_coop hA hW (flagRegs (compressorBodyWStmt inStride lenOff hashLog)) 3 vL
      (bindAt Γb nE vs).size (addrOf .arena 0) (UInt64.ofNat t) (compressorBodyWStmt inStride lenOff hashLog) f
      _ _ _ st' _ (wf_flagRegs _) hEnv hSl hrun fun e' Γ' _ _ => ?_)
    exact wpE_pure rfl
  · intro hcmp
    exfalso
    have : UInt64.ofNat cta < c64 ↑numBlocks := by
      rw [c64_ofNat, UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat_of_lt' (show cta < 2 ^ 64 by omega),
        UInt64.toNat_ofNat_of_lt' (show numBlocks < 2 ^ 64 by omega)]
      exact hcta
    simp [cmpInt, this] at hcmp


end

/-- **A compressor thread ends**, at some fuel, where its lane's run of the
    body ends: from the world after its share of the clear and the barrier,
    whose shared memory is the barrier's answer. -/
theorem thread_ends (hw : Hw) {ans : List Answer} (hW : Words ans) (name : String)
    {numBlocks inStride outStride lenOff hashLog cta t F f : Nat} {inP outA : UInt64} {w0 w1 : World}
    {b : ByteArray} {st' : LSt}
    (ht : t < 32) (hcta : cta < numBlocks) (hc32 : cta * 32 + 32 < 2 ^ 32) (hnb : numBlocks < 2 ^ 63)
    (hfine : Fine (emitGo (fnBody (idParams ++ [.i64, .i64]) (compressor numBlocks inStride outStride lenOff hashLog))
      ⟨(idParams ++ [ClifTy.i64, .i64]).length, 0, [], [], [], none⟩).2)
    (hflags : ∀ r, r ∈ ["inBase","outBase","tbl","op","searchPos","litAnchor"] →
      (flagRegs (compressorBodyWStmt inStride lenOff hashLog)).contains r = false)
    (hclear : clearL (addrOf .arena 0) (c64 0) (c64 (2 ^ hashLog)) (c64 ((2 * 2 ^ hashLog - 1 : Nat) : Int)) F (UInt64.ofNat t) w0 = some w1)
    (hbar : ans[(collCalls w1 w0.obs.length).length]? = some (.smem b))
    (hrun : laneRun (coopL ans w0.obs.length (addrOf .arena 0) (UInt64.ofNat t)) (addrOf .arena 0) f
      (compressorBodyWStmt inStride lenOff hashLog)
      ⟨regs0 (inP + UInt64.ofNat cta * c64 inStride)
        (inP + c64 (numBlocks * inStride + copySlack) + UInt64.ofNat cta * c64 outStride),
       { obsCall w1 (.intr .barrier) [] with mem := { (obsCall w1 (.intr .barrier) []).mem with arena := b } }⟩
      = some st') :
    ∃ steps u, thread (codeKernel name numBlocks inStride outStride lenOff hashLog).code ans
        (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t) w0 hw steps = .ok u ∧ u = st'.w := by
  have hE := emit_ends (cfg := { env := [], locals := threadLocals ans w0.obs.length hw })
    (P0 := At (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t).toArray w0)
    (by
      rw [fnBody_of (by decide)]
      exact wpE_thread (answersFrom_thread _ _ _) hW ht hcta hc32 hnb (Nat.le_refl _) hflags hclear hbar hrun)
    hfine (fun Γ' w' h => ⟨by rw [h.1]; simp [threadArgs, idParams], h⟩)
  obtain ⟨k, Γ', u, hk, hu⟩ := hE _ _ ⟨rfl, rfl⟩
  refine ⟨k, u, ?_, hu⟩
  unfold thread
  rw [show (codeKernel name numBlocks inStride outStride lenOff hashLog).code
    = emit (compressor numBlocks inStride outStride lenOff hashLog) (idParams ++ [.i64, .i64]) from rfl, hk]

end Lz4Code
