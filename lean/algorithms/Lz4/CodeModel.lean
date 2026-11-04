module
public import Lz4.CodeContent
public import AlgorithmLib.LZ4.EvalValid
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeModel` — what the warp's run of the body stores, and where

`okOf` asks of the warp's run what it reads and where it stores. The part
beyond where it stores is `okBase`; where it stores is a property of its byte
stores (`gstOf`), so `okOf` follows from both (`okOf_of`). The compressor's
emit sequences store at the output cursor, byte after byte, moving it as they
go (`Emits`: a token, a length's extension bytes, the literals copied from the
input, a whole match or final literal sequence), so their byte stores are a
run of consecutive output bytes (`outRun`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel LZ4WarpEvalBytes EmitContent

open AlgorithmLib.ML.LaneWarp (NotArena Writable)

namespace Lz4Code

/-- **What the lanes need of the warp's run, beyond where it stores**: a copy's
    bytes from `I`, both runs below `2^64` with room for the lanes' strides; a
    loop out of fuel stopped; a cooperative statement's own (`okC`). -/
def okBase (I : UInt64 → Prop) (okC : WStmt → WState → Prop) : Nat → WStmt → WState → Prop
  | f, .seq a b, st => okBase I okC f a st ∧ okBase I okC f b (a.eval f st)
  | f, .uif c t e, st => if st.regs c == 1 then okBase I okC f t st else okBase I okC f e st
  | 0, .uwhile c _, st => (st.regs c == 1) = false
  | f + 1, .uwhile c b, st =>
      st.regs c == 1 → okBase I okC f b st ∧ okBase I okC f (WStmt.uwhile c b) (b.eval f st)
  | _, .coopCopy d s n, st =>
      (st.regs n).toNat + 64 < 2 ^ 64 ∧ (st.regs s).toNat + (st.regs n).toNat < 2 ^ 64 ∧
      (st.regs d).toNat + (st.regs n).toNat < 2 ^ 64 ∧ ∀ i, i < (st.regs n).toNat + 31 → I (st.regs s + UInt64.ofNat i)
  | _, s@(.coopWindow ..), st | _, s@(.coopExtendStep ..), st => okC s st
  | _, _, _ => True

/-- **The warp's run makes only the accesses `okOf` allows** where it does
    beyond its stores and every byte it stores lies outside `I` and shared
    memory and inside `O`. -/
theorem okOf_of {I O : UInt64 → Prop} {okC : WStmt → WState → Prop} :
    ∀ (s : WStmt) (f : Nat) (ws : WState), okBase I okC f s ws →
      (∀ x ∈ gstOf f s ws, ¬ I x.2.1 ∧ NotArena x.2.1 ∧ O x.2.1) → okOf I O okC f s ws := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro f ws hb ht; simp only [okBase] at hb; simp only [gstOf, List.mem_append] at ht
      simp only [okOf]
      exact ⟨iha f ws hb.1 fun x hx => ht x (.inl hx), ihb f _ hb.2 fun x hx => ht x (.inr hx)⟩
  | uif c t e iht ihe =>
      intro f ws hb ht; simp only [okBase] at hb; simp only [okOf]
      by_cases hc : (ws.regs c == 1) = true
      · simp only [hc, if_true] at hb ht ⊢; simp only [gstOf, hc, if_true] at ht; exact iht f ws hb ht
      · simp only [hc, Bool.false_eq_true, if_false] at hb ⊢
        simp only [gstOf, hc, Bool.false_eq_true, if_false] at ht; exact ihe f ws hb ht
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws hb _; simp only [okOf, okBase] at hb ⊢; exact hb
      | succ f ihf =>
          intro ws hb ht
          simp only [okOf]
          intro hc
          simp only [okBase] at hb
          simp only [gstOf, hc, if_true, List.mem_append] at ht
          exact ⟨ih f ws (hb hc).1 fun x hx => ht x (.inl hx), ihf _ (hb hc).2 fun x hx => ht x (.inr hx)⟩
  | stgB addr val =>
      intro f ws _ ht
      simp only [okOf]
      exact ht (none, ws.regs addr, ws.regs val &&& 0xff) (by simp [gstOf])
  | coopCopy dst src len =>
      intro f ws hb ht
      simp only [okBase] at hb
      simp only [okOf]
      obtain ⟨hn, hs, hd, hI⟩ := hb
      refine ⟨hn, hs, hd, hI, fun i hi => ?_⟩
      exact ht (some (i % 32), ws.regs dst + UInt64.ofNat i, _) (by
        simp only [gstOf, copyTr, List.mem_map, List.mem_range]; exact ⟨i, hi, rfl⟩)
  | coopWindow => intro f ws hb _; simp only [okOf, okBase] at hb ⊢; exact hb
  | coopExtendStep => intro f ws hb _; simp only [okOf, okBase] at hb ⊢; exact hb
  | _ => intro f ws _ _; simp [okOf]

/-- `k` bytes of output from the cursor: `outBase + op`, `+ 1`, …. -/
def outRun (ws : WState) (k : Nat) : List UInt64 :=
  (List.range k).map fun i => ws.regs "outBase" + (ws.regs "op" + UInt64.ofNat i)

/-- The addresses of the warp's byte stores. -/
abbrev addrsOf (f : Nat) (s : WStmt) (ws : WState) : List UInt64 := (gstOf f s ws).map (·.2.1)

/-- Output runs from the cursor follow one another as the cursor moves. -/
theorem outRun_seq {f : Nat} {a b : WStmt} {ws : WState} {ka kb : Nat}
    (ha : addrsOf f a ws = outRun ws ka) (hb : addrsOf f b (a.eval f ws) = outRun (a.eval f ws) kb)
    (hop : (a.eval f ws).regs "op" = ws.regs "op" + UInt64.ofNat ka)
    (hob : (a.eval f ws).regs "outBase" = ws.regs "outBase") :
    addrsOf f (.seq a b) ws = outRun ws (ka + kb) := by
  simp only [addrsOf, gstOf, List.map_append] at ha hb ⊢
  rw [ha, hb]
  simp only [outRun, List.range_add, List.map_append, List.map_map]
  congr 1
  apply List.map_congr_left
  intro i _
  simp only [Function.comp, hop, hob, UInt64.ofNat_add, UInt64.add_assoc]

/-- A statement with no byte store runs no output. -/
theorem outRun_zero (ws : WState) : outRun ws 0 = [] := rfl

theorem wStoreByte_run (val : String) (ws : WState) (f : Nat) (h1 : val ≠ "sbAddr") :
    addrsOf f (wStoreByte val) ws = outRun ws 1 := by
  simp [addrsOf, wStoreByte, wseq, gstOf, WStmt.eval, WState.setReg, outRun, h1, WArg.eval, WOp.run]

theorem lsicBody_run (n : String) (hn_c : n ≠ "c255") (hn_sb : n ≠ "sbAddr") (hn_op : n ≠ "op")
    (hn_ob : n ≠ "outBase") (ws : WState) (f : Nat) :
    addrsOf f (lsicBody n) ws = outRun ws 1 := by
  simp [addrsOf, lsicBody, wStoreByte, wseq, gstOf, WStmt.eval, WState.setReg, outRun, WArg.eval, WOp.run]

theorem lsicBody_base {I : UInt64 → Prop} {okC : WStmt → WState → Prop} (n : String) (ws : WState) (f : Nat) :
    okBase I okC f (lsicBody n) ws := by
  simp [okBase, lsicBody, wStoreByte, wseq]

theorem lsicLoop_run {I : UInt64 → Prop} {okC : WStmt → WState → Prop} (n : String)
    (hn_c : n ≠ "c255") (hn_l : n ≠ "lsicC") (hn_op : n ≠ "op")
    (hn_ob : n ≠ "outBase") (hn_sb : n ≠ "sbAddr") :
    ∀ (q : Nat) (ws : WState) (fuel : Nat),
      (ws.regs n).toNat / 255 = q →
      ws.regs "lsicC" = (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0) →
      q < fuel →
      addrsOf fuel (WStmt.uwhile "lsicC" (lsicBody n)) ws = outRun ws q ∧
        okBase I okC fuel (WStmt.uwhile "lsicC" (lsicBody n)) ws := by
  have h255 : (UInt64.ofNat 255).toNat = 255 := by decide
  intro q
  induction q with
  | zero =>
    intro ws fuel hq hlsic hfuel
    have hvlt : (ws.regs n).toNat < 255 := by omega
    have hnge : ¬ (UInt64.ofNat 255 ≤ ws.regs n) := by
      rw [UInt64.le_iff_toNat_le, h255]; omega
    have hl0 : ws.regs "lsicC" = 0 := by rw [hlsic, if_neg hnge]
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
    refine ⟨?_, ?_⟩
    · simp [addrsOf, gstOf, hl0, outRun]
    · simp only [okBase, hl0]; intro h; simp at h
  | succ q ih =>
    intro ws fuel hq hlsic hfuel
    have hvge : 255 ≤ (ws.regs n).toNat := by
      have : 1 ≤ (ws.regs n).toNat / 255 := by omega
      omega
    have hge : UInt64.ofNat 255 ≤ ws.regs n := by rw [UInt64.le_iff_toNat_le, h255]; omega
    have hl1 : ws.regs "lsicC" = 1 := by rw [hlsic, if_pos hge]
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
    obtain ⟨hsn, hsop, hsob, hslc, -⟩ := lsicBodyStep n hn_c hn_l hn_op hn_ob hn_sb ws f
    have hsnNat : (((lsicBody n).eval f ws).regs n).toNat = (ws.regs n).toNat - 255 := by
      rw [hsn, UInt64.toNat_sub_of_le _ _ hge, h255]
    have hq1 : (((lsicBody n).eval f ws).regs n).toNat / 255 = q := by rw [hsnNat]; omega
    have hlsic1 : ((lsicBody n).eval f ws).regs "lsicC"
        = (if UInt64.ofNat 255 ≤ ((lsicBody n).eval f ws).regs n then 1 else 0) := by
      rw [hslc, hsn]
    obtain ⟨ra, rb⟩ := ih ((lsicBody n).eval f ws) f hq1 hlsic1 (by omega)
    refine ⟨?_, ?_⟩
    · have := outRun_seq (f := f) (a := lsicBody n) (b := WStmt.uwhile "lsicC" (lsicBody n)) (ws := ws)
        (lsicBody_run n hn_c hn_sb hn_op hn_ob ws f) ra (by rw [hsop]; rfl) hsob
      rw [show q + 1 = 1 + q by omega]
      simp only [addrsOf, gstOf, hl1, if_true] at this ⊢
      simpa using this
    · simp only [okBase, hl1]
      intro _
      exact ⟨lsicBody_base n ws f, rb⟩

theorem wEmitLSIC_run {I : UInt64 → Prop} {okC : WStmt → WState → Prop} (n : String) (ws : WState) (fuel : Nat)
    (hn_c : n ≠ "c255") (hn_l : n ≠ "lsicC") (hn_op : n ≠ "op")
    (hn_ob : n ≠ "outBase") (hn_sb : n ≠ "sbAddr")
    (hfuel : (ws.regs n).toNat / 255 < fuel) :
    addrsOf fuel (wEmitLSIC n) ws = outRun ws (AlgorithmLib.LZ4.ext (ws.regs n).toNat).length ∧
      okBase I okC fuel (wEmitLSIC n) ws := by
  have hsetp : WStmt.eval fuel (WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255)) ws
      = ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0) := by
    simp [WStmt.eval, WArg.eval, SCmp.run]
  generalize hs0 : ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0) = s0 at hsetp
  have e_n : s0.regs n = ws.regs n := by rw [← hs0]; simp [WState.setReg, hn_l]
  have e_lc : s0.regs "lsicC" = (if UInt64.ofNat 255 ≤ s0.regs n then 1 else 0) := by
    rw [e_n, ← hs0]; simp [WState.setReg]
  have e_op : s0.regs "op" = ws.regs "op" := by rw [← hs0]; simp [WState.setReg]
  have e_ob : s0.regs "outBase" = ws.regs "outBase" := by rw [← hs0]; simp [WState.setReg]
  obtain ⟨hla, hlb⟩ := lsicLoop_run (I := I) (okC := okC) n hn_c hn_l hn_op hn_ob hn_sb
    ((ws.regs n).toNat / 255) s0 fuel (by rw [e_n]) e_lc hfuel
  obtain ⟨rn, rop, rob, -⟩ := lsicLoop n hn_c hn_l hn_op hn_ob hn_sb ((ws.regs n).toNat / 255) s0 fuel
    (by rw [e_n]) e_lc hfuel
  have hlen : (AlgorithmLib.LZ4.ext (ws.regs n).toNat).length = (ws.regs n).toNat / 255 + 1 := by simp [AlgorithmLib.LZ4.ext]
  refine ⟨?_, ?_⟩
  · show addrsOf fuel (.seq (.setp .ge "lsicC" n (.imm 255)) (.seq (.uwhile "lsicC" (lsicBody n)) (wStoreByte n))) ws = _
    have h1 : addrsOf fuel (.setp .ge "lsicC" n (.imm 255)) ws = outRun ws 0 := by simp [addrsOf, gstOf, outRun]
    have h2 := outRun_seq (f := fuel) (a := .uwhile "lsicC" (lsicBody n)) (b := wStoreByte n) (ws := s0)
      hla (wStoreByte_run n _ fuel hn_sb) (by rw [rop]) rob
    have h3 := outRun_seq (f := fuel) (a := .setp .ge "lsicC" n (.imm 255))
      (b := .seq (.uwhile "lsicC" (lsicBody n)) (wStoreByte n)) (ws := ws) h1 (by rw [hsetp]; exact h2)
      (by rw [hsetp, e_op]; simp) (by rw [hsetp, e_ob])
    rw [h3, hlen]; congr 1; omega
  · show okBase I okC fuel (.seq (.setp .ge "lsicC" n (.imm 255)) (.seq (.uwhile "lsicC" (lsicBody n)) (wStoreByte n))) ws
    simp only [okBase]
    refine ⟨trivial, ?_, ?_⟩
    · rw [hsetp]; exact hlb
    · simp [okBase, wStoreByte, wseq]

/-- **A statement emits `k` bytes**: its byte stores are the next `k` bytes of
    output from the cursor, the cursor moves by `k`, the output's base stays,
    its run makes the accesses `okBase` allows, and the cursor does not pass
    `2^64`. -/
structure Emits (I : UInt64 → Prop) (okC : WStmt → WState → Prop) (f : Nat) (s : WStmt) (ws : WState) (k : Nat) :
    Prop where
  addrs : addrsOf f s ws = outRun ws k
  op : (s.eval f ws).regs "op" = ws.regs "op" + UInt64.ofNat k
  ob : (s.eval f ws).regs "outBase" = ws.regs "outBase"
  base : okBase I okC f s ws
  lt : (ws.regs "op").toNat + k < 2 ^ 64

section
variable {I : UInt64 → Prop} {okC : WStmt → WState → Prop} {f : Nat}

theorem Emits.seq {a b : WStmt} {ws : WState} {ka kb : Nat} (ha : Emits I okC f a ws ka)
    (hb : Emits I okC f b (a.eval f ws) kb) : Emits I okC f (.seq a b) ws (ka + kb) :=
  ⟨outRun_seq ha.addrs hb.addrs ha.op ha.ob,
   by simp only [WStmt.eval]; rw [hb.op, ha.op, UInt64.ofNat_add, UInt64.add_assoc],
   by simp only [WStmt.eval]; rw [hb.ob, ha.ob], by simp only [okBase]; exact ⟨ha.base, hb.base⟩,
   by
    have h := hb.lt
    have h2 := ha.lt
    rw [ha.op, UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show ka < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)] at h
    omega⟩

theorem Emits.bin (o : WOp) {d : String} (a : String) (b : WArg) (ws : WState) (h1 : d ≠ "op")
    (h2 : d ≠ "outBase") : Emits I okC f (.bin o d a b) ws 0 :=
  ⟨by simp [addrsOf, gstOf, outRun], by simp [WStmt.eval, WState.setReg, Ne.symm h1],
   by simp [WStmt.eval, WState.setReg, Ne.symm h2], by simp [okBase], by have := (ws.regs "op").toNat_lt; omega⟩

theorem Emits.setp (c : SCmp) {d : String} (a : String) (b : WArg) (ws : WState) (h1 : d ≠ "op")
    (h2 : d ≠ "outBase") : Emits I okC f (.setp c d a b) ws 0 :=
  ⟨by simp [addrsOf, gstOf, outRun], by simp [WStmt.eval, WState.setReg, Ne.symm h1],
   by simp [WStmt.eval, WState.setReg, Ne.symm h2], by simp [okBase], by have := (ws.regs "op").toNat_lt; omega⟩

theorem Emits.mov {d : String} (b : WArg) (ws : WState) (h1 : d ≠ "op") (h2 : d ≠ "outBase") :
    Emits I okC f (.mov d b) ws 0 :=
  ⟨by simp [addrsOf, gstOf, outRun], by simp [WStmt.eval, WState.setReg, Ne.symm h1],
   by simp [WStmt.eval, WState.setReg, Ne.symm h2], by simp [okBase], by have := (ws.regs "op").toNat_lt; omega⟩

theorem Emits.skip (ws : WState) : Emits I okC f .skip ws 0 :=
  ⟨by simp [addrsOf, gstOf, outRun], by simp [WStmt.eval], by simp [WStmt.eval], by simp [okBase],
   by have := (ws.regs "op").toNat_lt; omega⟩

theorem Emits.storeByte (val : String) (ws : WState) (h1 : val ≠ "sbAddr") (h2 : val ≠ "op")
    (hop : (ws.regs "op").toNat + 1 < 2 ^ 64) : Emits I okC f (wStoreByte val) ws 1 :=
  ⟨wStoreByte_run val ws f h1, by rw [(eval_wStoreByte val ws f h1 h2).1]; rfl,
   (eval_wStoreByte val ws f h1 h2).2.1, by simp [okBase, wStoreByte, wseq], hop⟩

theorem Emits.token (litLen tokLo : String) (ws : WState) (hop : (ws.regs "op").toNat + 1 < 2 ^ 64) :
    Emits I okC f (wEmitToken litLen tokLo) ws 1 := by
  have := ((Emits.bin (I := I) (okC := okC) (f := f) .min litLen (.imm 15) ws (d := "tokHi") (by decide) (by decide)).seq
    ((Emits.bin .shl "tokHi" (.imm 4) _ (d := "tok") (by decide) (by decide)).seq
    ((Emits.bin .bor "tok" (.reg tokLo) _ (d := "tok") (by decide) (by decide)).seq
    (Emits.storeByte "tok" _ (by decide) (by decide) (by simp [WStmt.eval, WState.setReg]; exact hop)))))
  simpa [wEmitToken, wseq] using this

theorem Emits.lsic (n : String) (ws : WState)
    (hn_c : n ≠ "c255") (hn_l : n ≠ "lsicC") (hn_op : n ≠ "op") (hn_ob : n ≠ "outBase") (hn_sb : n ≠ "sbAddr")
    (hfuel : (ws.regs n).toNat / 255 < f)
    (hop : (ws.regs "op").toNat + (AlgorithmLib.LZ4.ext (ws.regs n).toNat).length < 2 ^ 64) :
    Emits I okC f (wEmitLSIC n) ws (AlgorithmLib.LZ4.ext (ws.regs n).toNat).length := by
  obtain ⟨ha, hb⟩ := wEmitLSIC_run (I := I) (okC := okC) n ws f hn_c hn_l hn_op hn_ob hn_sb hfuel
  obtain ⟨hop', hob, -⟩ := eval_wEmitLSIC n ws f hn_c hn_l hn_op hn_ob hn_sb hfuel
  exact ⟨ha, hop', hob, hb, hop⟩

/-- A length field's extension bytes, where the nibble saturates. -/
theorem Emits.uifExt (P E src : String) (ws : WState)
    (hP : ws.regs P = (if UInt64.ofNat 15 ≤ ws.regs src then 1 else 0))
    (hEc : E ≠ "c255") (hEl : E ≠ "lsicC") (hEop : E ≠ "op") (hEsb : E ≠ "sbAddr")
    (hEob : E ≠ "outBase") (hfuel : (ws.regs src).toNat / 255 < f)
    (hop : (ws.regs "op").toNat + (AlgorithmLib.LZ4.encNib (ws.regs src).toNat).length < 2 ^ 64) :
    Emits I okC f (.uif P (.seq (.bin .sub E src (.imm 15)) (wEmitLSIC E)) .skip) ws
      (AlgorithmLib.LZ4.encNib (ws.regs src).toNat).length := by
  have h15 : (UInt64.ofNat 15).toNat = 15 := by decide
  by_cases hbig : UInt64.ofNat 15 ≤ ws.regs src
  · have hP1 : ws.regs P = 1 := by rw [hP, if_pos hbig]
    have hVge : 15 ≤ (ws.regs src).toNat := by rw [UInt64.le_iff_toNat_le, h15] at hbig; exact hbig
    have hE : ((WStmt.bin WOp.sub E src (WArg.imm 15)).eval f ws).regs E = ws.regs src - UInt64.ofNat 15 := by
      simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
    have hEN : (((WStmt.bin WOp.sub E src (WArg.imm 15)).eval f ws).regs E).toNat = (ws.regs src).toNat - 15 := by
      rw [hE, UInt64.toNat_sub_of_le _ _ hbig, h15]
    have hs := (Emits.bin (I := I) (okC := okC) (f := f) .sub src (.imm 15) ws hEop hEob).seq
      (Emits.lsic E _ hEc hEl hEop hEob hEsb (by rw [hEN]; omega) (by
        have hlen : (AlgorithmLib.LZ4.encNib (ws.regs src).toNat).length =
            (AlgorithmLib.LZ4.ext ((ws.regs src).toNat - 15)).length := by
          simp [AlgorithmLib.LZ4.encNib, show ¬ (ws.regs src).toNat < 15 by omega]
        rw [hEN, ← hlen]; simpa [WStmt.eval, WState.setReg, Ne.symm hEop] using hop))
    rw [hEN] at hs
    have hlen : (AlgorithmLib.LZ4.encNib (ws.regs src).toNat).length =
        0 + (AlgorithmLib.LZ4.ext ((ws.regs src).toNat - 15)).length := by
      simp [AlgorithmLib.LZ4.encNib, show ¬ (ws.regs src).toNat < 15 by omega]
    rw [hlen]
    exact ⟨by simpa [addrsOf, gstOf, hP1] using hs.addrs, by simpa [WStmt.eval, hP1] using hs.op,
      by simpa [WStmt.eval, hP1] using hs.ob, by simpa [okBase, hP1] using hs.base, by rw [← hlen]; exact hop⟩
  · have hP0 : ws.regs P = 0 := by rw [hP, if_neg hbig]
    have hlt : (ws.regs src).toNat < 15 := by rw [UInt64.le_iff_toNat_le, h15] at hbig; omega
    have hlen : (AlgorithmLib.LZ4.encNib (ws.regs src).toNat).length = 0 := by
      simp [AlgorithmLib.LZ4.encNib, hlt]
    rw [hlen]
    have hs := Emits.skip (I := I) (okC := okC) (f := f) ws
    exact ⟨by simpa [addrsOf, gstOf, hP0] using hs.addrs, by simpa [WStmt.eval, hP0] using hs.op,
      by simpa [WStmt.eval, hP0] using hs.ob, by simp [okBase, hP0], hs.lt⟩

end

section
variable {I : UInt64 → Prop} {okC : WStmt → WState → Prop} {f : Nat}

/-- **A literal copy, then the cursor moved past it, emits its bytes**: copying
    `litLen` bytes from the input to the cursor (`cpDst` there), where they all
    lie in `I` and nothing runs past `2^64`, then advancing the cursor. -/
theorem Emits.copy2 (cpDst cpSrc litLen : String) (ws : WState)
    (hd : ws.regs cpDst = ws.regs "outBase" + ws.regs "op") (hlo : litLen ≠ "op")
    (hn : (ws.regs litLen).toNat + 64 < 2 ^ 64) (hs : (ws.regs cpSrc).toNat + (ws.regs litLen).toNat < 2 ^ 64)
    (hdn : (ws.regs cpDst).toNat + (ws.regs litLen).toNat < 2 ^ 64)
    (hI : ∀ i, i < (ws.regs litLen).toNat + 31 → I (ws.regs cpSrc + UInt64.ofNat i))
    (hop : (ws.regs "op").toNat + (ws.regs litLen).toNat < 2 ^ 64) :
    Emits I okC f (.seq (.coopCopy cpDst cpSrc litLen) (.bin .add "op" "op" (.reg litLen))) ws
      (ws.regs litLen).toNat := by
  have hc : (WStmt.coopCopy cpDst cpSrc litLen).eval f ws =
      { ws with gmem := copyGmem ws.gmem (ws.regs cpDst).toNat (ws.regs cpSrc).toNat (ws.regs litLen).toNat } := by
    simp [WStmt.eval]
  have hop1 : ((WStmt.bin .add "op" "op" (.reg litLen)).eval f ((WStmt.coopCopy cpDst cpSrc litLen).eval f ws)).regs "op" =
      ws.regs "op" + UInt64.ofNat (ws.regs litLen).toNat := by
    rw [hc]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
  have hob1 : ((WStmt.bin .add "op" "op" (.reg litLen)).eval f ((WStmt.coopCopy cpDst cpSrc litLen).eval f ws)).regs
      "outBase" = ws.regs "outBase" := by
    rw [hc]; simp [WStmt.eval, WState.setReg]
  refine ⟨?_, by rw [eval_seq]; exact hop1, by rw [eval_seq]; exact hob1, ?_, ?_⟩
  · simp only [addrsOf, gstOf, copyTr, List.map_append, List.map_map, List.map_nil, List.append_nil, outRun]
    apply List.map_congr_left
    intro i _
    simp only [Function.comp, hd, UInt64.add_assoc]
  · simp only [okBase]
    exact ⟨⟨hn, hs, hdn, hI⟩, trivial⟩
  · exact hop

/-- A literal copy, the cursor moved past it, then more. -/
theorem Emits.copy (cpDst cpSrc litLen : String) (rest : WStmt) (ws : WState) {k : Nat}
    (hd : ws.regs cpDst = ws.regs "outBase" + ws.regs "op") (hlo : litLen ≠ "op")
    (hn : (ws.regs litLen).toNat + 64 < 2 ^ 64) (hs : (ws.regs cpSrc).toNat + (ws.regs litLen).toNat < 2 ^ 64)
    (hdn : (ws.regs cpDst).toNat + (ws.regs litLen).toNat < 2 ^ 64)
    (hI : ∀ i, i < (ws.regs litLen).toNat + 31 → I (ws.regs cpSrc + UInt64.ofNat i))
    (hop : (ws.regs "op").toNat + (ws.regs litLen).toNat < 2 ^ 64)
    (hrest : Emits I okC f rest ((WStmt.bin .add "op" "op" (.reg litLen)).eval f
      ((WStmt.coopCopy cpDst cpSrc litLen).eval f ws)) k) :
    Emits I okC f (.seq (.coopCopy cpDst cpSrc litLen) (.seq (.bin .add "op" "op" (.reg litLen)) rest)) ws
      ((ws.regs litLen).toNat + k) := by
  have := (Emits.copy2 (I := I) (okC := okC) (f := f) cpDst cpSrc litLen ws hd hlo hn hs hdn hI hop).seq (b := rest)
    (by rw [eval_seq]; exact hrest)
  refine ⟨?_, ?_, ?_, ?_, this.lt⟩
  · have h := this.addrs
    simp only [addrsOf, gstOf, List.map_append, List.append_assoc, eval_seq] at h ⊢
    exact h
  · have h := this.op; simp only [eval_seq] at h ⊢; exact h
  · have h := this.ob; simp only [eval_seq] at h ⊢; exact h
  · have h := this.base; simp only [okBase, eval_seq] at h ⊢; exact ⟨h.1.1, h.1.2, h.2⟩

end

section
variable {I : UInt64 → Prop} {okC : WStmt → WState → Prop} {f : Nat}

set_option maxHeartbeats 4000000 in
/-- **A match sequence emits its bytes**: the token, the literal length's
    extension, the literals copied from the input, the two offset bytes and
    the match length's extension. -/
theorem Emits.matchSeq (ws : WState) (hml : 4 ≤ (ws.regs "ml").toNat)
    (hfuelL : (ws.regs "litLen").toNat / 255 < f) (hfuelM : ((ws.regs "ml").toNat - 4) / 255 < f)
    (hn : (ws.regs "litLen").toNat + 64 < 2 ^ 64)
    (hsrcb : (ws.regs "inBase" + ws.regs "litAnchor").toNat + (ws.regs "litLen").toNat < 2 ^ 64)
    (hdstb : (ws.regs "outBase" + (ws.regs "op" + UInt64.ofNat
      (1 + (AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length))).toNat + (ws.regs "litLen").toNat < 2 ^ 64)
    (hsrc : ∀ i, i < (ws.regs "litLen").toNat + 31 → I (ws.regs "inBase" + ws.regs "litAnchor" + UInt64.ofNat i))
    (htot : (ws.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length +
      (ws.regs "litLen").toNat + 2 + (AlgorithmLib.LZ4.encNib ((ws.regs "ml").toNat - 4)).length) < 2 ^ 64) :
    Emits I okC f (wEmitMatchSeq "litAnchor" "litLen" "off0" "ml") ws
      (1 + (AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length + (ws.regs "litLen").toNat + 2 +
        (AlgorithmLib.LZ4.encNib ((ws.regs "ml").toNat - 4)).length) := by
  have hml4 : UInt64.ofNat 4 ≤ ws.regs "ml" := by
    rw [UInt64.le_iff_toNat_le, show (UInt64.ofNat 4).toNat = 4 from by decide]; omega
  -- the states
  generalize hs1 : (WStmt.bin WOp.sub "mlm" "ml" (WArg.imm 4)).eval f ws = s1
  have s1_ll : s1.regs "litLen" = ws.regs "litLen" := by rw [← hs1]; exact bin_reg _ _ _ _ _ ws f (by decide)
  have s1_mlmN : (s1.regs "mlm").toNat = (ws.regs "ml").toNat - 4 := by
    rw [← hs1, show ((WStmt.bin WOp.sub "mlm" "ml" (WArg.imm 4)).eval f ws).regs "mlm" = ws.regs "ml" - UInt64.ofNat 4
      from by simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run],
      UInt64.toNat_sub_of_le _ _ hml4, show (UInt64.ofNat 4).toNat = 4 from by decide]
  have s1_r : ∀ r, r ≠ "mlm" → s1.regs r = ws.regs r := fun r h => by rw [← hs1]; exact bin_reg _ _ _ _ _ ws f h
  generalize hs2 : (WStmt.bin WOp.min "tokLo" "mlm" (WArg.imm 15)).eval f s1 = s2
  have s2_r : ∀ r, r ≠ "tokLo" → s2.regs r = s1.regs r := fun r h => by rw [← hs2]; exact bin_reg _ _ _ _ _ s1 f h
  generalize hs3 : (wEmitToken "litLen" "tokLo").eval f s2 = s3
  have s3_r : ∀ r, r ≠ "tokHi" → r ≠ "tok" → r ≠ "sbAddr" → r ≠ "op" → s3.regs r = s2.regs r := fun r h1 h2 h3 h4 => by
    rw [← hs3]; exact wEmitToken_reg _ _ _ s2 f h1 h2 h3 h4
  have e3 : Emits I okC f (wEmitToken "litLen" "tokLo") s2 1 := Emits.token _ _ s2 (by
    rw [s2_r _ (by decide), s1_r _ (by decide)]; omega)
  have s3_opN : (s3.regs "op").toNat = (ws.regs "op").toNat + 1 := by
    have h3 : s3.regs "op" = s2.regs "op" + UInt64.ofNat 1 := by rw [← hs3]; exact e3.op
    rw [h3, s2_r _ (by decide), s1_r _ (by decide), UInt64.toNat_add, Nat.mod_eq_of_lt (by
      rw [UInt64.toNat_ofNat_of_lt' (show 1 < 2 ^ 64 by decide)]; omega)]; rfl
  generalize hs4 : (WStmt.setp SCmp.ge "pLitBig" "litLen" (WArg.imm 15)).eval f s3 = s4
  have s4_r : ∀ r, r ≠ "pLitBig" → s4.regs r = s3.regs r := fun r h => by rw [← hs4]; exact setp_reg _ _ _ _ _ s3 f h
  have s4_ll : s4.regs "litLen" = ws.regs "litLen" := by
    rw [s4_r _ (by decide), s3_r _ (by decide) (by decide) (by decide) (by decide), s2_r _ (by decide), s1_ll]
  have s4_pLit : s4.regs "pLitBig" = (if UInt64.ofNat 15 ≤ s4.regs "litLen" then 1 else 0) := by
    rw [← hs4]; simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  have e5 := Emits.uifExt (I := I) (okC := okC) (f := f) "pLitBig" "litExtra" "litLen" s4 s4_pLit (by decide)
    (by decide) (by decide) (by decide) (by decide) (by rw [s4_ll]; exact hfuelL)
    (by rw [s4_r _ (by decide), s3_opN, s4_ll]; omega)
  rw [s4_ll] at e5
  generalize hs5 : (WStmt.uif "pLitBig" (WStmt.seq (WStmt.bin WOp.sub "litExtra" "litLen" (WArg.imm 15))
      (wEmitLSIC "litExtra")) WStmt.skip).eval f s4 = s5 at e5
  obtain ⟨-, lu_fr⟩ := eval_uifExt "pLitBig" "litExtra" "litLen" s4 f s4_pLit (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by rw [s4_ll]; exact hfuelL)
  rw [hs5] at lu_fr
  have s5_r : ∀ r, r ≠ "litExtra" → r ≠ "c255" → r ≠ "lsicC" → r ≠ "op" → r ≠ "sbAddr" → s5.regs r = s4.regs r :=
    lu_fr
  -- the cursor and the copy
  have s5_op : s5.regs "op" = ws.regs "op" + UInt64.ofNat (1 + (AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length) := by
    have h5 := e5.op; rw [hs5] at h5
    have h3 : s3.regs "op" = s2.regs "op" + UInt64.ofNat 1 := by rw [← hs3]; exact e3.op
    rw [h5, s4_r _ (by decide), h3, s2_r _ (by decide), s1_r _ (by decide), UInt64.ofNat_add, UInt64.add_assoc]
  have s5_opN : (s5.regs "op").toNat = (ws.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length) := by
    rw [s5_op, UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show _ < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
  have s5_ob : s5.regs "outBase" = ws.regs "outBase" := by
    have h5 := e5.ob; rw [hs5] at h5
    rw [h5, s4_r _ (by decide), show s3.regs "outBase" = s2.regs "outBase" by rw [← hs3]; exact e3.ob,
      s2_r _ (by decide), s1_r _ (by decide)]
  have s5_ll : s5.regs "litLen" = ws.regs "litLen" := by
    rw [s5_r _ (by decide) (by decide) (by decide) (by decide) (by decide), s4_ll]
  have s5_reg : ∀ r, r = "inBase" ∨ r = "litAnchor" ∨ r = "off0" → s5.regs r = ws.regs r := by
    intro r hr
    rcases hr with rfl | rfl | rfl <;>
    · rw [s5_r _ (by decide) (by decide) (by decide) (by decide) (by decide), s4_r _ (by decide),
        s3_r _ (by decide) (by decide) (by decide) (by decide), s2_r _ (by decide), s1_r _ (by decide)]
  have s5_mlmN : (s5.regs "mlm").toNat = (ws.regs "ml").toNat - 4 := by
    rw [s5_r _ (by decide) (by decide) (by decide) (by decide) (by decide), s4_r _ (by decide),
      s3_r _ (by decide) (by decide) (by decide) (by decide), s2_r _ (by decide), s1_mlmN]
  generalize hs6 : (WStmt.bin WOp.add "cpDst" "outBase" (WArg.reg "op")).eval f s5 = s6
  have s6_r : ∀ r, r ≠ "cpDst" → s6.regs r = s5.regs r := fun r h => by rw [← hs6]; exact bin_reg _ _ _ _ _ s5 f h
  have s6_d : s6.regs "cpDst" = s5.regs "outBase" + s5.regs "op" := by
    rw [← hs6]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
  generalize hs7 : (WStmt.bin WOp.add "cpSrc" "inBase" (WArg.reg "litAnchor")).eval f s6 = s7
  have s7_r : ∀ r, r ≠ "cpSrc" → s7.regs r = s6.regs r := fun r h => by rw [← hs7]; exact bin_reg _ _ _ _ _ s6 f h
  have s7_s : s7.regs "cpSrc" = ws.regs "inBase" + ws.regs "litAnchor" := by
    rw [← hs7]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
    rw [s6_r _ (by decide), s6_r _ (by decide), s5_reg _ (.inl rfl), s5_reg _ (.inr (.inl rfl))]
  have s7_ll : s7.regs "litLen" = ws.regs "litLen" := by rw [s7_r _ (by decide), s6_r _ (by decide), s5_ll]
  have s7_d : s7.regs "cpDst" = s7.regs "outBase" + s7.regs "op" := by
    rw [s7_r _ (by decide), s6_d, s7_r _ (by decide), s6_r _ (by decide), s7_r _ (by decide), s6_r _ (by decide)]
  -- after the copy and the cursor's move
  generalize hs9 : (WStmt.bin .add "op" "op" (.reg "litLen")).eval f
    ((WStmt.coopCopy "cpDst" "cpSrc" "litLen").eval f s7) = s9
  have s9_r : ∀ r, r ≠ "op" → s9.regs r = s7.regs r := fun r h => by
    rw [← hs9, bin_reg _ _ _ _ _ _ f h, coopCopy_reg]
  have s7_opN : (s7.regs "op").toNat = (ws.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length) := by
    rw [s7_r _ (by decide), s6_r _ (by decide), s5_opN]
  have s9_opN : (s9.regs "op").toNat = (ws.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length) + (ws.regs "litLen").toNat := by
    have h9 : s9.regs "op" = s7.regs "op" + s7.regs "litLen" := by
      rw [← hs9]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
    rw [h9, UInt64.toNat_add, s7_opN, s7_ll, Nat.mod_eq_of_lt (by omega)]
  generalize hs10 : (WStmt.bin WOp.band "offLo" "off0" (WArg.imm 255)).eval f s9 = s10
  have s10_r : ∀ r, r ≠ "offLo" → s10.regs r = s9.regs r := fun r h => by rw [← hs10]; exact bin_reg _ _ _ _ _ s9 f h
  generalize hs11 : (wStoreByte "offLo").eval f s10 = s11
  have s11_r : ∀ r, r ≠ "sbAddr" → r ≠ "op" → s11.regs r = s10.regs r := fun r h1 h2 => by
    rw [← hs11]; exact wStoreByte_reg _ _ s10 f h1 h2
  have s10_opN : (s10.regs "op").toNat = (ws.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length) + (ws.regs "litLen").toNat := by
    rw [s10_r _ (by decide), s9_opN]
  have s11_opN : (s11.regs "op").toNat = (ws.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length) + (ws.regs "litLen").toNat + 1 := by
    rw [← hs11, (eval_wStoreByte "offLo" s10 f (by decide) (by decide)).1, UInt64.toNat_add, s10_opN,
      Nat.mod_eq_of_lt (by rw [show (1 : UInt64).toNat = 1 from rfl]; omega)]; rfl
  generalize hs12 : (WStmt.bin WOp.shr "offHi" "off0" (WArg.imm 8)).eval f s11 = s12
  have s12_r : ∀ r, r ≠ "offHi" → s12.regs r = s11.regs r := fun r h => by rw [← hs12]; exact bin_reg _ _ _ _ _ s11 f h
  generalize hs13 : (WStmt.bin WOp.band "offHi" "offHi" (WArg.imm 255)).eval f s12 = s13
  have s13_r : ∀ r, r ≠ "offHi" → s13.regs r = s12.regs r := fun r h => by rw [← hs13]; exact bin_reg _ _ _ _ _ s12 f h
  generalize hs14 : (wStoreByte "offHi").eval f s13 = s14
  have s14_r : ∀ r, r ≠ "sbAddr" → r ≠ "op" → s14.regs r = s13.regs r := fun r h1 h2 => by
    rw [← hs14]; exact wStoreByte_reg _ _ s13 f h1 h2
  have s13_opN : (s13.regs "op").toNat = (ws.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length) + (ws.regs "litLen").toNat + 1 := by
    rw [s13_r _ (by decide), s12_r _ (by decide), s11_opN]
  have s14_opN : (s14.regs "op").toNat = (ws.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length) + (ws.regs "litLen").toNat + 2 := by
    rw [← hs14, (eval_wStoreByte "offHi" s13 f (by decide) (by decide)).1, UInt64.toNat_add, s13_opN,
      Nat.mod_eq_of_lt (by rw [show (1 : UInt64).toNat = 1 from rfl]; omega)]; rfl
  generalize hs15 : (WStmt.setp SCmp.ge "pMatBig" "mlm" (WArg.imm 15)).eval f s14 = s15
  have s15_r : ∀ r, r ≠ "pMatBig" → s15.regs r = s14.regs r := fun r h => by rw [← hs15]; exact setp_reg _ _ _ _ _ s14 f h
  have s15_mlmN : (s15.regs "mlm").toNat = (ws.regs "ml").toNat - 4 := by
    rw [s15_r _ (by decide), s14_r _ (by decide) (by decide), s13_r _ (by decide), s12_r _ (by decide),
      s11_r _ (by decide) (by decide), s10_r _ (by decide), s9_r _ (by decide), s7_r _ (by decide),
      s6_r _ (by decide), s5_mlmN]
  have s15_pMat : s15.regs "pMatBig" = (if UInt64.ofNat 15 ≤ s15.regs "mlm" then 1 else 0) := by
    rw [← hs15]; simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  -- the emits, from the back
  have e16 := Emits.uifExt (I := I) (okC := okC) (f := f) "pMatBig" "matExtra" "mlm" s15 s15_pMat (by decide)
    (by decide) (by decide) (by decide) (by decide) (by rw [s15_mlmN]; exact hfuelM)
    (by rw [s15_r _ (by decide), s14_opN, s15_mlmN]; omega)
  rw [s15_mlmN] at e16
  have e15 := (Emits.setp (I := I) (okC := okC) (f := f) .ge "mlm" (.imm 15) s14 (d := "pMatBig") (by decide)
    (by decide)).seq (by rw [hs15]; exact e16)
  have e14 := (Emits.storeByte (I := I) (okC := okC) (f := f) "offHi" s13 (by decide) (by decide)
    (by rw [s13_opN]; omega)).seq
    (by rw [hs14]; exact e15)
  have e13 := (Emits.bin (I := I) (okC := okC) (f := f) .band "offHi" (.imm 255) s12 (d := "offHi") (by decide)
    (by decide)).seq (by rw [hs13]; exact e14)
  have e12 := (Emits.bin (I := I) (okC := okC) (f := f) .shr "off0" (.imm 8) s11 (d := "offHi") (by decide)
    (by decide)).seq (by rw [hs12]; exact e13)
  have e11 := (Emits.storeByte (I := I) (okC := okC) (f := f) "offLo" s10 (by decide) (by decide)
    (by rw [s10_opN]; omega)).seq
    (by rw [hs11]; exact e12)
  have e10 := (Emits.bin (I := I) (okC := okC) (f := f) .band "off0" (.imm 255) s9 (d := "offLo") (by decide)
    (by decide)).seq (by rw [hs10]; exact e11)
  have e8 := Emits.copy (I := I) (okC := okC) (f := f) "cpDst" "cpSrc" "litLen" _ s7 s7_d (by decide)
    (by rw [s7_ll]; exact hn) (by rw [s7_s, s7_ll]; exact hsrcb)
    (by rw [s7_d, s7_r _ (by decide), s6_r _ (by decide), s7_r _ (by decide), s6_r _ (by decide), s5_op, s5_ob, s7_ll];
        exact hdstb)
    (fun i hi => by rw [s7_s]; exact hsrc i (by rw [← s7_ll]; exact hi)) (by rw [s7_opN, s7_ll]; omega)
    (by rw [hs9]; exact e10)
  rw [s7_ll] at e8
  have e7 := (Emits.bin (I := I) (okC := okC) (f := f) .add "inBase" (.reg "litAnchor") s6 (d := "cpSrc") (by decide)
    (by decide)).seq (by rw [hs7]; exact e8)
  have e6 := (Emits.bin (I := I) (okC := okC) (f := f) .add "outBase" (.reg "op") s5 (d := "cpDst") (by decide)
    (by decide)).seq (by rw [hs6]; exact e7)
  have e5' := e5.seq (by rw [hs5]; exact e6)
  have e4 := (Emits.setp (I := I) (okC := okC) (f := f) .ge "litLen" (.imm 15) s3 (d := "pLitBig") (by decide)
    (by decide)).seq (by rw [hs4]; exact e5')
  have e3' := e3.seq (by rw [hs3]; exact e4)
  have e2 := (Emits.bin (I := I) (okC := okC) (f := f) .min "mlm" (.imm 15) s1 (d := "tokLo") (by decide)
    (by decide)).seq (by rw [hs2]; exact e3')
  have e1 := (Emits.bin (I := I) (okC := okC) (f := f) .sub "ml" (.imm 4) ws (d := "mlm") (by decide)
    (by decide)).seq (by rw [hs1]; exact e2)
  have hk : 0 + (0 + (1 + (0 + ((AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length +
      (0 + (0 + ((ws.regs "litLen").toNat + (0 + (1 + (0 + (0 + (1 + (0 +
      (AlgorithmLib.LZ4.encNib ((ws.regs "ml").toNat - 4)).length))))))))))))) =
      1 + (AlgorithmLib.LZ4.encNib (ws.regs "litLen").toNat).length + (ws.regs "litLen").toNat + 2 +
        (AlgorithmLib.LZ4.encNib ((ws.regs "ml").toNat - 4)).length := by omega
  rw [hk] at e1
  simpa only [wEmitMatchSeq, wseq] using e1

end

section
variable {I : UInt64 → Prop} {okC : WStmt → WState → Prop} {f : Nat}

set_option maxHeartbeats 4000000 in
/-- **The final literal sequence emits its bytes**: the token, the literal
    length's extension, the literals copied from the input. -/
theorem Emits.finalSeq (ws : WState) (hfuel : (ws.regs "fLen").toNat / 255 < f)
    (hn : (ws.regs "fLen").toNat + 64 < 2 ^ 64)
    (hsrcb : (ws.regs "inBase" + ws.regs "litAnchor").toNat + (ws.regs "fLen").toNat < 2 ^ 64)
    (hdstb : (ws.regs "outBase" + (ws.regs "op" + UInt64.ofNat
      (1 + (AlgorithmLib.LZ4.encNib (ws.regs "fLen").toNat).length))).toNat + (ws.regs "fLen").toNat < 2 ^ 64)
    (hsrc : ∀ i, i < (ws.regs "fLen").toNat + 31 → I (ws.regs "inBase" + ws.regs "litAnchor" + UInt64.ofNat i))
    (htot : (ws.regs "op").toNat + (1 + (AlgorithmLib.LZ4.encNib (ws.regs "fLen").toNat).length +
      (ws.regs "fLen").toNat) < 2 ^ 64) :
    Emits I okC f (wEmitFinalSeq "litAnchor" "fLen") ws
      (1 + (AlgorithmLib.LZ4.encNib (ws.regs "fLen").toNat).length + (ws.regs "fLen").toNat) := by
  generalize hs1 : (WStmt.mov "zero" (WArg.imm 0)).eval f ws = s1
  have s1_r : ∀ r, r ≠ "zero" → s1.regs r = ws.regs r := fun r h => by rw [← hs1]; exact mov_reg _ _ _ ws f h
  generalize hs2 : (wEmitToken "fLen" "zero").eval f s1 = s2
  have s2_r : ∀ r, r ≠ "tokHi" → r ≠ "tok" → r ≠ "sbAddr" → r ≠ "op" → s2.regs r = s1.regs r := fun r h1 h2 h3 h4 => by
    rw [← hs2]; exact wEmitToken_reg _ _ _ s1 f h1 h2 h3 h4
  have e2 : Emits I okC f (wEmitToken "fLen" "zero") s1 1 := Emits.token _ _ s1 (by rw [s1_r _ (by decide)]; omega)
  have s2_opN : (s2.regs "op").toNat = (ws.regs "op").toNat + 1 := by
    have h2 : s2.regs "op" = s1.regs "op" + UInt64.ofNat 1 := by rw [← hs2]; exact e2.op
    rw [h2, s1_r _ (by decide), UInt64.toNat_add, Nat.mod_eq_of_lt (by
      rw [UInt64.toNat_ofNat_of_lt' (show 1 < 2 ^ 64 by decide)]; omega)]; rfl
  generalize hs3 : (WStmt.setp SCmp.ge "pLitBigF" "fLen" (WArg.imm 15)).eval f s2 = s3
  have s3_r : ∀ r, r ≠ "pLitBigF" → s3.regs r = s2.regs r := fun r h => by rw [← hs3]; exact setp_reg _ _ _ _ _ s2 f h
  have s3_ll : s3.regs "fLen" = ws.regs "fLen" := by
    rw [s3_r _ (by decide), s2_r _ (by decide) (by decide) (by decide) (by decide), s1_r _ (by decide)]
  have s3_p : s3.regs "pLitBigF" = (if UInt64.ofNat 15 ≤ s3.regs "fLen" then 1 else 0) := by
    rw [← hs3]; simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  have e4 := Emits.uifExt (I := I) (okC := okC) (f := f) "pLitBigF" "litExtraF" "fLen" s3 s3_p (by decide)
    (by decide) (by decide) (by decide) (by decide) (by rw [s3_ll]; exact hfuel)
    (by rw [s3_r _ (by decide), s2_opN, s3_ll]; omega)
  rw [s3_ll] at e4
  generalize hs4 : (WStmt.uif "pLitBigF" (WStmt.seq (WStmt.bin WOp.sub "litExtraF" "fLen" (WArg.imm 15))
      (wEmitLSIC "litExtraF")) WStmt.skip).eval f s3 = s4
  obtain ⟨-, lu_fr⟩ := eval_uifExt "pLitBigF" "litExtraF" "fLen" s3 f s3_p (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by rw [s3_ll]; exact hfuel)
  rw [hs4] at lu_fr
  have s4_op : s4.regs "op" = ws.regs "op" + UInt64.ofNat (1 + (AlgorithmLib.LZ4.encNib (ws.regs "fLen").toNat).length) := by
    have h4 := e4.op; rw [hs4] at h4
    have h2 : s2.regs "op" = s1.regs "op" + UInt64.ofNat 1 := by rw [← hs2]; exact e2.op
    rw [h4, s3_r _ (by decide), h2, s1_r _ (by decide), UInt64.ofNat_add, UInt64.add_assoc]
  have s4_ob : s4.regs "outBase" = ws.regs "outBase" := by
    have h4 := e4.ob; rw [hs4] at h4
    rw [h4, s3_r _ (by decide), show s2.regs "outBase" = s1.regs "outBase" by rw [← hs2]; exact e2.ob,
      s1_r _ (by decide)]
  have s4_reg : ∀ r, r = "inBase" ∨ r = "litAnchor" ∨ r = "fLen" → s4.regs r = ws.regs r := by
    intro r hr
    rcases hr with rfl | rfl | rfl <;>
    · rw [lu_fr _ (by decide) (by decide) (by decide) (by decide) (by decide), s3_r _ (by decide),
        s2_r _ (by decide) (by decide) (by decide) (by decide), s1_r _ (by decide)]
  generalize hs5 : (WStmt.bin WOp.add "cpDstF" "outBase" (WArg.reg "op")).eval f s4 = s5
  have s5_r : ∀ r, r ≠ "cpDstF" → s5.regs r = s4.regs r := fun r h => by rw [← hs5]; exact bin_reg _ _ _ _ _ s4 f h
  have s5_d : s5.regs "cpDstF" = s4.regs "outBase" + s4.regs "op" := by
    rw [← hs5]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
  generalize hs6 : (WStmt.bin WOp.add "cpSrcF" "inBase" (WArg.reg "litAnchor")).eval f s5 = s6
  have s6_r : ∀ r, r ≠ "cpSrcF" → s6.regs r = s5.regs r := fun r h => by rw [← hs6]; exact bin_reg _ _ _ _ _ s5 f h
  have s6_s : s6.regs "cpSrcF" = ws.regs "inBase" + ws.regs "litAnchor" := by
    rw [← hs6]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
    rw [s5_r _ (by decide), s5_r _ (by decide), s4_reg _ (.inl rfl), s4_reg _ (.inr (.inl rfl))]
  have s6_ll : s6.regs "fLen" = ws.regs "fLen" := by
    rw [s6_r _ (by decide), s5_r _ (by decide), s4_reg _ (.inr (.inr rfl))]
  have s6_d : s6.regs "cpDstF" = s6.regs "outBase" + s6.regs "op" := by
    rw [s6_r _ (by decide), s5_d, s6_r _ (by decide), s5_r _ (by decide), s6_r _ (by decide), s5_r _ (by decide)]
  have e7 := Emits.copy2 (I := I) (okC := okC) (f := f) "cpDstF" "cpSrcF" "fLen" s6 s6_d (by decide)
    (by rw [s6_ll]; exact hn) (by rw [s6_s, s6_ll]; exact hsrcb)
    (by rw [s6_d, s6_r _ (by decide), s5_r _ (by decide), s6_r _ (by decide), s5_r _ (by decide), s4_op, s4_ob, s6_ll];
        exact hdstb)
    (fun i hi => by rw [s6_s]; exact hsrc i (by rw [← s6_ll]; exact hi))
    (by
      rw [s6_r _ (by decide), s5_r _ (by decide), s4_op, s6_ll, UInt64.toNat_add,
        UInt64.toNat_ofNat_of_lt' (show _ < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
      omega)
  rw [s6_ll] at e7
  have e6 := (Emits.bin (I := I) (okC := okC) (f := f) .add "inBase" (.reg "litAnchor") s5 (d := "cpSrcF") (by decide)
    (by decide)).seq (by rw [hs6]; exact e7)
  have e5 := (Emits.bin (I := I) (okC := okC) (f := f) .add "outBase" (.reg "op") s4 (d := "cpDstF") (by decide)
    (by decide)).seq (by rw [hs5]; exact e6)
  have e4' := e4.seq (by rw [hs4]; exact e5)
  have e3 := (Emits.setp (I := I) (okC := okC) (f := f) .ge "fLen" (.imm 15) s2 (d := "pLitBigF") (by decide)
    (by decide)).seq (by rw [hs3]; exact e4')
  have e2' := e2.seq (by rw [hs2]; exact e3)
  have e1 := (Emits.mov (I := I) (okC := okC) (f := f) (.imm 0) ws (d := "zero") (by decide) (by decide)).seq
    (by rw [hs1]; exact e2')
  have hk : 0 + (1 + (0 + ((AlgorithmLib.LZ4.encNib (ws.regs "fLen").toNat).length + (0 + (0 +
      (ws.regs "fLen").toNat))))) =
      1 + (AlgorithmLib.LZ4.encNib (ws.regs "fLen").toNat).length + (ws.regs "fLen").toNat := by omega
  rw [hk] at e1
  simpa only [wEmitFinalSeq, wseq] using e1

end

end Lz4Code
