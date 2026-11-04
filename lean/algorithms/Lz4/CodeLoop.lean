module
public import Lz4.CodeModel
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeLoop` — the body's loops emit

A loop that halts within its fuel, each trip from where it runs emitting and
keeping where it runs, emits (`Emits.uwhile`). The extend loop runs within the
match's room and halts (`extInv`, `extLoop_halts`), each step's conditions
holding (`Emits.extLoop`); a found match's branch emits its match sequence
(`Emits.found`); a trip of the main loop emits, from `LoopAt`, keeping it
(`Emits.loopBody`); the main loop halts (`loopC_halts`) and so emits
(`Emits.mainLoop`); and the body's encoding prefix emits (`Emits.prefix`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel LZ4WarpEvalBytes

namespace Lz4Code

section
variable {I : UInt64 → Prop} {okC : WStmt → WState → Prop}

/-- **A loop that halts, each trip emitting, emits**: where an invariant `P`
    holds, each trip from `P` keeps it and emits, and the loop halts within its
    fuel. -/
theorem Emits.uwhile {c : String} {body : WStmt} (P : Nat → WState → Prop)
    (hstep : ∀ f ws, P (f + 1) ws → (ws.regs c == 1) = true → P f (body.eval f ws) ∧ ∃ k, Emits I okC f body ws k) :
    ∀ f ws, P f ws → WhileHalts c body f ws → ∃ k, Emits I okC f (.uwhile c body) ws k := by
  intro f
  induction f with
  | zero => intro ws _ h; exact absurd h (by simp [WhileHalts])
  | succ f ih =>
      intro ws hP hH
      simp only [WhileHalts] at hH
      by_cases hc : (ws.regs c == 1) = true
      · rw [if_pos hc] at hH
        obtain ⟨hP', kb, hb⟩ := hstep f ws hP hc
        obtain ⟨kl, hl⟩ := ih _ hP' hH
        refine ⟨kb + kl, ?_, ?_, ?_, ?_, ?_⟩
        · have := outRun_seq hb.addrs hl.addrs hb.op hb.ob
          simpa [addrsOf, gstOf, hc] using this
        · simp only [WStmt.eval, hc, if_true]; rw [hl.op, hb.op, UInt64.ofNat_add, UInt64.add_assoc]
        · simp only [WStmt.eval, hc, if_true]; rw [hl.ob, hb.ob]
        · simp only [okBase]; intro _; exact ⟨hb.base, hl.base⟩
        · have h := hl.lt
          have h2 := hb.lt
          rw [hb.op, UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show kb < 2 ^ 64 by omega),
            Nat.mod_eq_of_lt (by omega)] at h
          omega
      · refine ⟨0, by simp [addrsOf, gstOf, hc, outRun], by simp [WStmt.eval, hc], by simp [WStmt.eval, hc], ?_,
          by have := (ws.regs "op").toNat_lt; omega⟩
        simp only [okBase]; intro h; exact absurd h hc

end

section
variable {I : UInt64 → Prop} {okC : WStmt → WState → Prop}

theorem Emits.window {f : Nat} {sp : String} {inStride searchLim hashLog tbl : Nat} (ws : WState)
    (hok : okC (.coopWindow "found" "p0" "cand0" sp inStride searchLim hashLog tbl) ws) :
    Emits I okC f (.coopWindow "found" "p0" "cand0" sp inStride searchLim hashLog tbl) ws 0 :=
  ⟨by simp [addrsOf, gstOf, outRun],
   by rw [WStmt.eval.eq_11, evalCoopWindow_regs_frame _ _ _ _ _ _ _ _ _ _ (by decide) (by decide) (by decide)]; simp,
   by rw [WStmt.eval.eq_11, evalCoopWindow_regs_frame _ _ _ _ _ _ _ _ _ _ (by decide) (by decide) (by decide)],
   by simpa [okBase] using hok, by have := (ws.regs "op").toNat_lt; omega⟩

theorem Emits.extend {f : Nat} {ml : String} {inStride endCap : Nat} (ws : WState)
    (hok : okC (.coopExtendStep "adv" "p0" "cand0" ml inStride endCap) ws) :
    Emits I okC f (.coopExtendStep "adv" "p0" "cand0" ml inStride endCap) ws 0 :=
  ⟨by simp [addrsOf, gstOf, outRun], by rw [WStmt.eval.eq_12]; simp [evalCoopExtendStep, WState.setReg],
   by rw [WStmt.eval.eq_12]; simp [evalCoopExtendStep, WState.setReg], by simpa [okBase] using hok,
   by have := (ws.regs "op").toNat_lt; omega⟩

end

/-- The input of a block: `inStride` bytes from `ib`. -/
def inputOf (ib : UInt64) (inStride : Nat) (a : UInt64) : Prop := ∃ j, j < inStride ∧ a = ib + UInt64.ofNat j

/-- What a block's warp may read: its input, and the 32 bytes after it that a
    copy's lanes read past its end (the launch's copy slack). -/
abbrev blockIn (ib : UInt64) (inStride : Nat) : UInt64 → Prop := inputOf ib (inStride + 32)

/-- Where the extend loop runs: its invariant (`extInv`), the match before the
    end cap with room for the first four bytes, and the block's input. -/
def ExtAt (ib : UInt64) (inStride : Nat) (ws : WState) : Prop :=
  extInv inStride (inStride - 5) ws ∧ ws.regs "inBase" = ib ∧ (ws.regs "p0").toNat + 4 ≤ inStride - 5 ∧
    ib.toNat < 2 ^ 40

theorem extBody_regs (inStride endCap f : Nat) (ws : WState) (r : String) (h1 : r ≠ "adv") (h2 : r ≠ "ml")
    (h3 : r ≠ "extC") : ((extBodyStmt inStride endCap).eval f ws).regs r = ws.regs r := by
  simp [extBodyStmt, wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, h1, h2, h3]

/-- **The extend loop emits nothing**, and makes only the accesses `okBase`
    allows: each step's conditions (`okLZ4`) hold where the loop runs, and it
    halts within its fuel. -/
theorem Emits.extLoop {f inStride : Nat} {ib : UInt64} (ws : WState) (hst : 16 ≤ inStride) (hst2 : inStride ≤ 65536)
    (hA : ExtAt ib inStride ws)
    (hfuel : inStride - 5 + 32 < 32 * f + ((ws.regs "p0").toNat + (ws.regs "ml").toNat)) :
    ∃ k, Emits (blockIn ib inStride) (okLZ4 (blockIn ib inStride)) f
      (.uwhile "extC" (extBodyStmt inStride (inStride - 5))) ws k := by
  refine Emits.uwhile (fun _ => ExtAt ib inStride) (fun g s hs hc => ?_) f ws hA
    (extLoop_halts inStride (inStride - 5) rfl (by omega) f ws hA.1 hfuel)
  obtain ⟨hinv, hib, hp4, hib40⟩ := hs
  refine ⟨⟨extBody_preserves_extInv inStride _ g s rfl (by omega) hinv,
    by rw [extBody_regs _ _ _ _ _ (by decide) (by decide) (by decide), hib],
    by rw [extBody_regs _ _ _ _ _ (by decide) (by decide) (by decide)]; exact hp4, hib40⟩, 0, ?_⟩
  obtain ⟨hecR, hec1, hcp, hpml⟩ := hinv
  have hok : okLZ4 (blockIn ib inStride) (.coopExtendStep "adv" "p0" "cand0" "ml" inStride (inStride - 5)) s := by
    refine ⟨hecR, hec1, by omega, by omega, hcp, by omega, fun j hj => ⟨j, by omega, by rw [hib]⟩, by rw [hib]; omega,
      by omega⟩
  have e := (Emits.extend (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := g) s hok).seq
    ((Emits.bin (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := g) .add "ml" (.reg "adv") _
      (d := "ml") (by decide) (by decide)).seq
    (Emits.setp (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := g) .eq "adv" (.imm 32) _
      (d := "extC") (by decide) (by decide)))
  simpa [extBodyStmt, wseq] using e

theorem encNib_len (n : Nat) : (AlgorithmLib.LZ4.encNib n).length ≤ n / 255 + 1 := by
  unfold AlgorithmLib.LZ4.encNib
  split
  · simp
  · simp [AlgorithmLib.LZ4.ext]; omega

set_option maxHeartbeats 4000000 in
/-- **A found match's branch emits its match sequence**, where the window
    found it: the extend loop runs to the match's end, then the match sequence
    emits its token, literals, offset and lengths, and the cursor's anchors
    move past the match. -/
theorem Emits.found {f inStride : Nat} {ib : UInt64} (s : WState) (hst : 16 ≤ inStride) (hst2 : inStride ≤ 65536)
    (hf : inStride ≤ f) (hib : s.regs "inBase" = ib) (hib40 : ib.toNat < 2 ^ 40)
    (hla : (s.regs "litAnchor").toNat ≤ (s.regs "p0").toNat) (hp0 : (s.regs "p0").toNat < inStride)
    (hbud : (s.regs "outBase").toNat + (s.regs "op").toNat + 9 * (inStride - (s.regs "litAnchor").toNat) < 2 ^ 64)
    (hcand : (s.regs "cand0").toNat < (s.regs "p0").toNat) (hp4 : (s.regs "p0").toNat + 4 ≤ inStride - 5) :
    ∃ k, Emits (blockIn ib inStride) (okLZ4 (blockIn ib inStride)) f (foundBranchStmt inStride (inStride - 5)) s k := by
  -- after the four setup statements
  generalize ht4 : (WStmt.setp .ge "extC" "ml" (.imm 0)).eval f ((WStmt.mov "ml" (.imm 4)).eval f
    ((WStmt.mov "ec1" (.imm (inStride - 5 - 1))).eval f ((WStmt.mov "ecR" (.imm (inStride - 5))).eval f s))) = t4
  have ht4' : (wseq [ .mov "ecR" (.imm (inStride - 5)), .mov "ec1" (.imm (inStride - 5 - 1)),
      .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval f s = t4 := by
    rw [← ht4]; simp only [wseq, eval_seq]
  have t4_r : ∀ r, r ≠ "ecR" → r ≠ "ec1" → r ≠ "ml" → r ≠ "extC" → t4.regs r = s.regs r := by
    intro r h1 h2 h3 h4; rw [← ht4]; simp [WStmt.eval, WState.setReg, h1, h2, h3, h4]
  have t4_ml : t4.regs "ml" = UInt64.ofNat 4 := by rw [← ht4]; simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  have hinv4 : extInv inStride (inStride - 5) t4 := by
    rw [← ht4']; exact extInv_of_foundMovs _ _ f s hcand hp4
  have hA4 : ExtAt ib inStride t4 := ⟨hinv4, by rw [t4_r _ (by decide) (by decide) (by decide) (by decide), hib],
    by rw [t4_r _ (by decide) (by decide) (by decide) (by decide)]; exact hp4, hib40⟩
  obtain ⟨kx, ex⟩ := Emits.extLoop (f := f) t4 hst hst2 hA4 (by
    rw [t4_ml, show (UInt64.ofNat 4).toNat = 4 from rfl]; omega)
  -- after the extend loop
  generalize ht5 : (WStmt.uwhile "extC" (extBodyStmt inStride (inStride - 5))).eval f t4 = t5
  have t5_r : ∀ r, r ≠ "adv" → r ≠ "ml" → r ≠ "extC" → t5.regs r = t4.regs r := fun r h1 h2 h3 => by
    rw [← ht5]; exact uwhile_extBody_frame _ _ r h1 h2 h3 f t4
  have hinv5 : extInv inStride (inStride - 5) t5 := by rw [← ht5]; exact extInv_uwhile _ _ rfl (by omega) f t4 hinv4
  have hml5 : 4 ≤ (t5.regs "ml").toNat := by
    rw [← ht5]; exact uwhile_ml_ge4 _ _ rfl (by omega) f t4 hinv4 (by rw [t4_ml]; decide)
  have s5 : ∀ r, r ≠ "adv" → r ≠ "ml" → r ≠ "extC" → r ≠ "ecR" → r ≠ "ec1" → t5.regs r = s.regs r :=
    fun r h1 h2 h3 h4 h5 => by rw [t5_r r h1 h2 h3, t4_r r h4 h5 h2 h3]
  -- off0 and litLen
  generalize ht6 : (WStmt.bin .sub "off0" "p0" (.reg "cand0")).eval f t5 = t6
  have t6_r : ∀ r, r ≠ "off0" → t6.regs r = t5.regs r := fun r h => by rw [← ht6]; exact bin_reg _ _ _ _ _ t5 f h
  generalize ht7 : (WStmt.bin .sub "litLen" "p0" (.reg "litAnchor")).eval f t6 = t7
  have t7_r : ∀ r, r ≠ "litLen" → t7.regs r = t6.regs r := fun r h => by rw [← ht7]; exact bin_reg _ _ _ _ _ t6 f h
  have q : ∀ r, r ≠ "adv" → r ≠ "ml" → r ≠ "extC" → r ≠ "ecR" → r ≠ "ec1" → r ≠ "off0" → r ≠ "litLen" →
      t7.regs r = s.regs r := fun r h1 h2 h3 h4 h5 h6 h7 => by rw [t7_r r h7, t6_r r h6, s5 r h1 h2 h3 h4 h5]
  have t7_ll : (t7.regs "litLen").toNat = (s.regs "p0").toNat - (s.regs "litAnchor").toNat := by
    rw [← ht7]; simp only [WStmt.eval, WState.setReg, if_true, WArg.eval, WOp.run]
    rw [t6_r _ (by decide), t6_r _ (by decide), s5 _ (by decide) (by decide) (by decide) (by decide) (by decide),
      s5 _ (by decide) (by decide) (by decide) (by decide) (by decide),
      UInt64.toNat_sub_of_le _ _ (by rw [UInt64.le_iff_toNat_le]; exact hla)]
  have t7_ml : t7.regs "ml" = t5.regs "ml" := by rw [t7_r _ (by decide), t6_r _ (by decide)]
  have hib7 : t7.regs "inBase" = ib := by
    rw [q _ (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hib]
  have hla7 : t7.regs "litAnchor" = s.regs "litAnchor" :=
    q _ (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have hop7 : t7.regs "op" = s.regs "op" := q _ (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have hob7 : t7.regs "outBase" = s.regs "outBase" :=
    q _ (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have hpml : (t5.regs "p0").toNat + (t5.regs "ml").toNat ≤ inStride - 5 := hinv5.2.2.2
  have hp05 : t5.regs "p0" = s.regs "p0" := s5 _ (by decide) (by decide) (by decide) (by decide) (by decide)
  rw [hp05] at hpml
  have hencb := encNib_len (t7.regs "litLen").toNat
  have hlit : (t7.regs "litLen").toNat < inStride := by omega
  -- the match sequence
  have ems := Emits.matchSeq (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := f) t7
    (by rw [t7_ml]; exact hml5) (by omega) (by rw [t7_ml]; omega) (by omega)
    (by
      rw [hib7, hla7, UInt64.toNat_add, Nat.mod_eq_of_lt (by have := (s.regs "litAnchor").toNat_lt; omega)]
      omega)
    (by
      have hk : 1 + (AlgorithmLib.LZ4.encNib (t7.regs "litLen").toNat).length < 2 ^ 32 := by omega
      have h1 : ((s.regs "op") + UInt64.ofNat (1 + (AlgorithmLib.LZ4.encNib (t7.regs "litLen").toNat).length)).toNat =
          (s.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (t7.regs "litLen").toNat).length) := by
        rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show _ < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
      have h2 : ((s.regs "outBase") + ((s.regs "op") + UInt64.ofNat (1 +
          (AlgorithmLib.LZ4.encNib (t7.regs "litLen").toNat).length))).toNat =
          (s.regs "outBase").toNat + ((s.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (t7.regs "litLen").toNat).length)) := by
        rw [UInt64.toNat_add, h1, Nat.mod_eq_of_lt (by omega)]
      rw [hob7, hop7, h2]
      omega)
    (fun i hi => ⟨(s.regs "litAnchor").toNat + i, by omega, by
      rw [hib7, hla7, show s.regs "litAnchor" = UInt64.ofNat (s.regs "litAnchor").toNat from UInt64.ofNat_toNat.symm,
        UInt64.add_assoc, ← UInt64.ofNat_add, UInt64.toNat_ofNat_of_lt' (show (s.regs "litAnchor").toNat < 2 ^ 64 from
          (s.regs "litAnchor").toNat_lt)]⟩)
    (by
      have hm := encNib_len ((t5.regs "ml").toNat - 4)
      rw [hop7, t7_ml]
      omega)
  generalize ht8 : (wEmitMatchSeq "litAnchor" "litLen" "off0" "ml").eval f t7 = t8 at ems
  -- assemble, from the back
  have e10 := (Emits.bin (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := f) .add "p0" (.reg "ml") t8
    (d := "litAnchor") (by decide) (by decide)).seq
    (Emits.mov (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := f) (.reg "litAnchor") _
      (d := "searchPos") (by decide) (by decide))
  have e9 := ems.seq (by rw [ht8]; exact e10)
  have e8 := (Emits.bin (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := f) .sub "p0" (.reg "litAnchor")
    t6 (d := "litLen") (by decide) (by decide)).seq (by rw [ht7]; exact e9)
  have e7 := (Emits.bin (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := f) .sub "p0" (.reg "cand0")
    t5 (d := "off0") (by decide) (by decide)).seq (by rw [ht6]; exact e8)
  have e6 := ex.seq (by rw [ht5]; exact e7)
  have e5 := (Emits.mov (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := f) (.imm (inStride - 5)) s
    (d := "ecR") (by decide) (by decide)).seq
    ((Emits.mov (.imm (inStride - 5 - 1)) _ (d := "ec1") (by decide) (by decide)).seq
    ((Emits.mov (.imm 4) _ (d := "ml") (by decide) (by decide)).seq
    ((Emits.setp .ge "ml" (.imm 0) _ (d := "extC") (by decide) (by decide)).seq (by rw [ht4]; exact e6))))
  exact ⟨_, by simpa only [foundBranchStmt, wseq, extBodyStmt] using e5⟩

/-- The warp's run leaves every register it never sets. -/
theorem eval_frame (r : String) : ∀ (s : WStmt) (f : Nat) (ws : WState), r ∉ setsOf s →
    (s.eval f ws).regs r = ws.regs r := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro f ws h; simp only [setsOf, List.mem_append, not_or] at h
      simp only [WStmt.eval]; rw [ihb f _ h.2, iha f ws h.1]
  | uif c t e iht ihe =>
      intro f ws h; simp only [setsOf, List.mem_append, not_or] at h
      by_cases hc : (ws.regs c == 1) = true
      · simp only [WStmt.eval, hc, if_true]; exact iht f ws h.1
      · simp only [WStmt.eval, hc, Bool.false_eq_true, if_false]; exact ihe f ws h.2
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws _; simp [WStmt.eval]
      | succ f ihf =>
          intro ws h
          have h' : r ∉ setsOf b := fun hm => h (List.mem_cons_of_mem _ hm)
          by_cases hc : (ws.regs c == 1) = true
          · simp only [WStmt.eval, hc, if_true]; rw [ihf _ h, ih f ws h']
          · simp only [WStmt.eval, hc, Bool.false_eq_true, if_false]
  | bin o d a b => intro f ws h; simp only [setsOf, List.mem_singleton] at h; simp [WStmt.eval, WState.setReg, h]
  | setp c d a b => intro f ws h; simp only [setsOf, List.mem_singleton] at h; simp [WStmt.eval, WState.setReg, h]
  | mov d a => intro f ws h; simp only [setsOf, List.mem_singleton] at h; simp [WStmt.eval, WState.setReg, h]
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      intro f ws h
      simp only [setsOf, List.mem_cons, List.mem_nil_iff, or_false, not_or] at h
      rw [WStmt.eval.eq_11]
      exact evalCoopWindow_regs_frame _ _ _ _ _ _ _ _ _ _ h.1 h.2.1 h.2.2
  | coopExtendStep adv p0 cand0 ml inStride endCap =>
      intro f ws h
      simp only [setsOf, List.mem_singleton] at h
      rw [WStmt.eval.eq_12]; simp [evalCoopExtendStep, WState.setReg, h]
  | _ => intro f ws _; simp [WStmt.eval, WState.stgByte]

theorem tableInsert_size (sm g : Array UInt8) (hashLog tbl searchLim ib sp upto : Nat) :
    (tableInsert sm g hashLog tbl searchLim ib sp upto).size = sm.size := by
  unfold tableInsert
  have : ∀ (L : List Nat) (acc : Array UInt8), (L.foldl (fun acc k =>
      if k ≤ upto ∧ sp + k < searchLim then
        (acc.setIfInBounds (tbl + 2 * wHash g hashLog (ib + (sp + k))) (UInt8.ofNat ((sp + k + 1) % 256))).setIfInBounds
          (tbl + 2 * wHash g hashLog (ib + (sp + k)) + 1) (UInt8.ofNat ((sp + k + 1) / 256 % 256))
      else acc) acc).size = acc.size := by
    intro L
    induction L with
    | nil => intro acc; rfl
    | cons k L ih => intro acc; simp only [List.foldl_cons]; rw [ih]; split <;> simp
  exact this _ _

/-- The warp's run leaves the size of its table. -/
theorem eval_smem_size : ∀ (s : WStmt) (f : Nat) (ws : WState), (s.eval f ws).smem.size = ws.smem.size := by
  intro s
  induction s with
  | seq a b iha ihb => intro f ws; simp only [WStmt.eval]; rw [ihb, iha]
  | uif c t e iht ihe =>
      intro f ws
      by_cases hc : (ws.regs c == 1) = true
      · simp only [WStmt.eval, hc, if_true]; exact iht f ws
      · simp only [WStmt.eval, hc, Bool.false_eq_true, if_false]; exact ihe f ws
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws; simp [WStmt.eval]
      | succ f ihf =>
          intro ws
          by_cases hc : (ws.regs c == 1) = true
          · simp only [WStmt.eval, hc, if_true]; rw [ihf, ih]
          · simp only [WStmt.eval, hc, Bool.false_eq_true, if_false]
  | coopWindow =>
      intro f ws
      rw [WStmt.eval.eq_11, evalCoopWindow_eq_go, evalCoopWindowGo_smem]
      split <;> exact tableInsert_size _ _ _ _ _ _ _ _
  | coopExtendStep => intro f ws; rw [WStmt.eval.eq_12]; rfl
  | coopCopy => intro f ws; simp [WStmt.eval]
  | _ => intro f ws; simp [WStmt.eval, WState.setReg, WState.stgByte]

section
variable {I : UInt64 → Prop} {okC : WStmt → WState → Prop} {f : Nat}

theorem Emits.uifT {c : String} {t e : WStmt} {ws : WState} {k : Nat} (hc : (ws.regs c == 1) = true)
    (h : Emits I okC f t ws k) : Emits I okC f (.uif c t e) ws k :=
  ⟨by simpa [addrsOf, gstOf, hc] using h.addrs, by simpa [WStmt.eval, hc] using h.op,
   by simpa [WStmt.eval, hc] using h.ob, by simpa [okBase, hc] using h.base, h.lt⟩

theorem Emits.uifF {c : String} {t e : WStmt} {ws : WState} {k : Nat} (hc : ¬ (ws.regs c == 1) = true)
    (h : Emits I okC f e ws k) : Emits I okC f (.uif c t e) ws k :=
  ⟨by simpa [addrsOf, gstOf, hc] using h.addrs, by simpa [WStmt.eval, hc] using h.op,
   by simpa [WStmt.eval, hc] using h.ob, by simpa [okBase, hc] using h.base, h.lt⟩

end

/-- Where the main loop runs: `LoopCQ`, the block's input, the table at 0
    inside shared memory. -/
def LoopAt (ib : UInt64) (inStride hashLog : Nat) (ws : WState) : Prop :=
  LoopCQ (2 ^ 64) inStride ws ∧ ws.regs "inBase" = ib ∧ ws.regs "tbl" = 0 ∧ 2 * 2 ^ hashLog ≤ ws.smem.size

set_option maxHeartbeats 4000000 in
/-- **A trip of the main loop emits**, from where the loop runs and its guard
    holds: the window, then a found match's sequence or the cursor's step past
    the window; and the loop still runs from where it ends, its search
    position past where it was. -/
theorem Emits.loopBody {f inStride hashLog : Nat} {ib : UInt64} (ws : WState) (hst : 16 ≤ inStride)
    (hst2 : inStride ≤ 65536) (hh : hashLog ≤ 32) (htb : 2 * 2 ^ hashLog + 2 < 2 ^ 36) (hf : inStride ≤ f)
    (hA : LoopAt ib inStride hashLog ws) (hg : (ws.regs "loopC" == 1) = true) :
    LoopAt ib inStride hashLog ((loopCBodyStmt inStride hashLog).eval f ws) ∧
      (ws.regs "searchPos").toNat + 1 ≤ (((loopCBodyStmt inStride hashLog).eval f ws).regs "searchPos").toNat ∧
      ∃ k, Emits (blockIn ib inStride) (okLZ4 (blockIn ib inStride)) f (loopCBodyStmt inStride hashLog) ws k := by
  obtain ⟨hQ, hib, htbl, hsz⟩ := hA
  have hsp := loopCQ_guard inStride ws hst2 (by omega) hQ hg
  have hqa := loopCBody_Qadvance inStride hashLog f ws hst2 (by omega) (by omega) (by decide) hf hQ hg
  have hib40 := hQ.1
  refine ⟨⟨hqa.1, ?_, ?_, ?_⟩, hqa.2.1, ?_⟩
  · rw [eval_frame _ _ _ _ (by simp [setsOf, loopCBodyStmt, foundBranchStmt, wEmitMatchSeq, wEmitToken, wStoreByte,
      wEmitLSIC, wseq]), hib]
  · rw [eval_frame _ _ _ _ (by simp [setsOf, loopCBodyStmt, foundBranchStmt, wEmitMatchSeq, wEmitToken, wStoreByte,
      wEmitLSIC, wseq]), htbl]
  · rw [eval_smem_size]; exact hsz
  have hok : okLZ4 (blockIn ib inStride) (.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0)
      ws := ⟨by omega, hh, fun j hj => ⟨j, by omega, by rw [hib]⟩, by omega, by omega, by rw [htbl]; rfl, by omega,
        by decide, by decide, by decide, by omega⟩
  have e1 := Emits.window (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := f) ws hok
  generalize hs1 : (WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0).eval f ws = s1
    at e1
  have hs1' : s1 = evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws := by
    rw [← hs1, WStmt.eval.eq_11]
  have hsetp : ∀ t : WState, Emits (blockIn ib inStride) (okLZ4 (blockIn ib inStride)) f
      (.setp .lt "loopC" "searchPos" (.imm (inStride - 12))) t 0 :=
    fun t => Emits.setp .lt "searchPos" (.imm (inStride - 12)) t (d := "loopC") (by decide) (by decide)
  cases hw : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
      (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) (ws.regs "searchPos").toNat with
  | none =>
      have hf0 : ¬ (s1.regs "found" == 1) = true := by
        rw [hs1', evalCoopWindow_eq_go, hw, evalCoopWindowGo_regs]; simp [WState.setReg]
      have eb := Emits.bin (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := f) .add "searchPos"
        (.imm 32) s1 (d := "searchPos") (by decide) (by decide)
      have eu := Emits.uifF (t := foundBranchStmt inStride (inStride - 5)) hf0 (show Emits _ _ f
        (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]) s1 0 by simpa [wseq] using eb)
      have e := e1.seq (by rw [hs1]; exact eu.seq (hsetp _))
      exact ⟨_, by simpa only [loopCBodyStmt, wseq] using e⟩
  | some pc =>
      obtain ⟨p, c⟩ := pc
      have hf1 : (s1.regs "found" == 1) = true := by
        rw [hs1', evalCoopWindow_eq_go, hw, evalCoopWindowGo_regs]; simp [WState.setReg]
      obtain ⟨h1, h2, h3, h4, -, -, h7, h8⟩ := loopCBody_found_cwFacts inStride hashLog ws hst2 (by omega) hQ hg p c hw
      rw [← hs1'] at h1 h2 h3 h4 h7 h8
      have hib1 : s1.regs "inBase" = ib := by
        rw [hs1', evalCoopWindow_regs_frame _ _ _ _ _ _ _ _ _ _ (by decide) (by decide) (by decide), hib]
      obtain ⟨kf, ef⟩ := Emits.found (f := f) s1 hst hst2 hf hib1 (by rw [← hib1]; exact h1) h2 h3 h4 h7 h8
      have eu := Emits.uifT (e := wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]) hf1 ef
      have e := e1.seq (by rw [hs1]; exact eu.seq (hsetp _))
      exact ⟨_, by simpa only [loopCBodyStmt, wseq] using e⟩

/-- **The main loop emits**, from where it runs with fuel for every trip. -/
theorem Emits.mainLoop {inStride hashLog : Nat} {ib : UInt64} (hst : 16 ≤ inStride) (hst2 : inStride ≤ 65536)
    (hh : hashLog ≤ 32) (htb : 2 * 2 ^ hashLog + 2 < 2 ^ 36) :
    ∀ f (ws : WState), LoopAt ib inStride hashLog ws →
      (inStride - 12) - (ws.regs "searchPos").toNat + inStride ≤ f →
      ∃ k, Emits (blockIn ib inStride) (okLZ4 (blockIn ib inStride)) f
        (.uwhile "loopC" (loopCBodyStmt inStride hashLog)) ws k := by
  intro f ws hA hm
  refine Emits.uwhile (fun g s => LoopAt ib inStride hashLog s ∧
      (inStride - 12) - (s.regs "searchPos").toNat + inStride ≤ g) (fun g s hs hc => ?_) f ws ⟨hA, hm⟩
    (loopC_halts inStride hashLog hst2 (by omega) (by omega) (by decide) f ws hA.1 hm)
  obtain ⟨hA', hm'⟩ := hs
  have hsp := loopCQ_guard inStride s hst2 (by omega) hA'.1 hc
  obtain ⟨hA2, hadv, k, e⟩ := Emits.loopBody (f := g) s hst hst2 hh htb (by omega) hA' hc
  exact ⟨⟨hA2, by omega⟩, k, e⟩

set_option maxHeartbeats 4000000 in
/-- **The body's encoding prefix emits**: from the state a block's body starts
    in, the main loop then the final literal sequence. -/
theorem Emits.prefix {inStride hashLog : Nat} {ib : UInt64} (ws : WState) (hst : 16 ≤ inStride)
    (hst2 : inStride ≤ 65536) (hh : hashLog ≤ 32) (htb : 2 * 2 ^ hashLog + 2 < 2 ^ 36)
    (hop0 : ws.regs "op" = 0) (hla0 : ws.regs "litAnchor" = 0) (hsp0 : ws.regs "searchPos" = 0)
    (hib : ws.regs "inBase" = ib) (hib40 : ib.toNat < 2 ^ 40)
    (hobN : (ws.regs "outBase").toNat + 9 * inStride < 2 ^ 64)
    (hsize : (ws.regs "outBase").toNat + 9 * inStride ≤ ws.gmem.size)
    (hobLB : (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat)
    (htbl : ws.regs "tbl" = 0) (hsz : 2 * 2 ^ hashLog ≤ ws.smem.size) :
    ∃ k, Emits (blockIn ib inStride) (okLZ4 (blockIn ib inStride)) (inStride + 34 * inStride)
      (bodyEncodePrefix inStride hashLog) ws k := by
  generalize hF : inStride + 34 * inStride = F
  have hFi : inStride ≤ F := by omega
  have hsets : ∀ r, r = "inBase" ∨ r = "tbl" ∨ r = "outBase" → r ∉ setsOf (WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)) := by
    intro r hr
    rcases hr with rfl | rfl | rfl <;>
    · simp [setsOf, loopCBodyStmt, foundBranchStmt, wEmitMatchSeq, wEmitToken, wStoreByte, wEmitLSIC, wseq]
  generalize hs1 : (WStmt.setp .lt "loopC" "searchPos" (.imm (inStride - 12))).eval F ws = s1
  have s1_r : ∀ r, r ≠ "loopC" → s1.regs r = ws.regs r := fun r h => by rw [← hs1]; exact setp_reg _ _ _ _ _ ws F h
  have hQ1 : LoopCQ (2 ^ 64) inStride s1 := by
    rw [← hs1]; exact setpLoopC_LoopCQ inStride (by omega) (by omega) ws hop0 hla0 hsp0 (by rw [hib]; exact hib40)
      hobN hsize hobLB F
  have hA1 : LoopAt ib inStride hashLog s1 := ⟨hQ1, by rw [s1_r _ (by decide), hib], by rw [s1_r _ (by decide), htbl],
    by rw [← hs1]; simpa [WStmt.eval, WState.setReg] using hsz⟩
  have hm1 : (inStride - 12) - (s1.regs "searchPos").toNat + inStride ≤ F := by
    rw [s1_r _ (by decide), hsp0]; simp; omega
  obtain ⟨kl, el⟩ := Emits.mainLoop hst hst2 hh htb F s1 hA1 hm1
  generalize hs2 : (WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F s1 = s2
  have hQ2 : LoopCQ (2 ^ 64) inStride s2 := by
    rw [← hs2]; exact loopC_loop_preservesInv inStride hashLog hst2 (by omega) (by omega) (by decide) F s1 hQ1 hm1
  have hla2 : (s2.regs "litAnchor").toNat ≤ inStride - 5 := by
    rw [← hs2]; exact loopC_litAnchor_bound inStride hashLog hst2 (by omega) (by omega) (by decide) F s1 hQ1
      (by rw [s1_r _ (by decide), hla0]; simp) hm1
  have hib2 : s2.regs "inBase" = ib := by rw [← hs2, eval_frame _ _ _ _ (hsets _ (.inl rfl)), s1_r _ (by decide), hib]
  generalize hs3 : (WStmt.mov "fLen" (.imm inStride)).eval F s2 = s3
  have s3_r : ∀ r, r ≠ "fLen" → s3.regs r = s2.regs r := fun r h => by rw [← hs3]; exact mov_reg _ _ _ s2 F h
  generalize hs4 : (WStmt.bin .sub "fLen" "fLen" (.reg "litAnchor")).eval F s3 = s4
  have s4_r : ∀ r, r ≠ "fLen" → s4.regs r = s3.regs r := fun r h => by rw [← hs4]; exact bin_reg _ _ _ _ _ s3 F h
  have q : ∀ r, r ≠ "fLen" → s4.regs r = s2.regs r := fun r h => by rw [s4_r r h, s3_r r h]
  have hfl : (s4.regs "fLen").toNat = inStride - (s2.regs "litAnchor").toNat := by
    rw [← hs4]; simp only [WStmt.eval, WState.setReg, if_true, WArg.eval, WOp.run]
    rw [← hs3]; simp only [WStmt.eval, WState.setReg, if_true, WArg.eval, show ("litAnchor" = "fLen") = False by decide,
      if_false]
    rw [UInt64.toNat_sub_of_le _ _ (by rw [UInt64.le_iff_toNat_le, UInt64.toNat_ofNat_of_lt' (show inStride < 2 ^ 64 by omega)]; omega),
      UInt64.toNat_ofNat_of_lt' (show inStride < 2 ^ 64 by omega)]
  obtain ⟨_, _, _, _, hbud32, -⟩ := hQ2
  have hencb := encNib_len (s4.regs "fLen").toNat
  have ef := Emits.finalSeq (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := F) s4 (by omega) (by omega)
    (by
      rw [q _ (by decide), q _ (by decide), hib2, UInt64.toNat_add,
        Nat.mod_eq_of_lt (by have := (s2.regs "litAnchor").toNat_lt; omega)]
      omega)
    (by
      have h1 : ((s4.regs "op") + UInt64.ofNat (1 + (AlgorithmLib.LZ4.encNib (s4.regs "fLen").toNat).length)).toNat =
          (s4.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (s4.regs "fLen").toNat).length) := by
        rw [q _ (by decide)]
        rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show _ < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
      have h2 : ((s4.regs "outBase") + ((s4.regs "op") + UInt64.ofNat (1 +
          (AlgorithmLib.LZ4.encNib (s4.regs "fLen").toNat).length))).toNat =
          (s4.regs "outBase").toNat + ((s4.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (s4.regs "fLen").toNat).length)) := by
        rw [UInt64.toNat_add, h1, q _ (by decide), q _ (by decide), Nat.mod_eq_of_lt (by omega)]
      rw [h2, q _ (by decide), q _ (by decide)]
      omega)
    (fun i hi => ⟨(s2.regs "litAnchor").toNat + i, by omega, by
      rw [q _ (by decide), q _ (by decide), hib2,
        show s2.regs "litAnchor" = UInt64.ofNat (s2.regs "litAnchor").toNat from UInt64.ofNat_toNat.symm,
        UInt64.add_assoc, ← UInt64.ofNat_add, UInt64.toNat_ofNat_of_lt' (show (s2.regs "litAnchor").toNat < 2 ^ 64 from
          (s2.regs "litAnchor").toNat_lt)]⟩)
    (by rw [q _ (by decide)]; omega)
  have e4 := (Emits.bin (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := F) .sub "fLen" (.reg "litAnchor") s3
    (d := "fLen") (by decide) (by decide)).seq (by rw [hs4]; exact ef)
  have e3 := (Emits.mov (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := F) (.imm inStride) s2
    (d := "fLen") (by decide) (by decide)).seq (by rw [hs3]; exact e4)
  have e2 := el.seq (by rw [hs2]; exact e3)
  have e1 := (Emits.setp (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride)) (f := F) .lt "searchPos"
    (.imm (inStride - 12)) ws (d := "loopC") (by decide) (by decide)).seq (by rw [hs1]; exact e2)
  exact ⟨_, by simpa only [bodyEncodePrefix, wseq, loopCBodyStmt, foundBranchStmt] using e1⟩

end Lz4Code

