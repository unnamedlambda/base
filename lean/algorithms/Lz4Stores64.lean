import Lz4OpLe
import Lz4OpLe64
import Lz4Stores

set_option maxRecDepth 1500

/-!
  # `RegConfined`'s store half

  Fourteen of the sixteen store sites: the ten `sbAddr` stores of the token emit
  and final literal run, which `cursorAtSites_shipped64` bounds, and the four
  length-field stores, which sit at fixed offsets `lenOff … lenOff+3`.

  The two remaining sites are the cooperative copy's `cpDo` (pcs 165 and 251).
  They are predicated, so `ActiveAt` restricts them to the lanes that actually
  write, and bounding those needs the copy loop's counter — see the note at the
  end of this file.
-/

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt

theorem stores_la64 (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (w : Fin (WP.mk 16).numBlk) (k : Nat) (l : Lane) (r : String)
    (hr : r = "la0" ∨ r = "la1" ∨ r = "la2" ∨ r = "la3")
    (hpc : ((siter K16 k (initSt w.val inPtr outPtr gm smemB)).pc, r) ∈ storeSites K16)
    (htop : outPtr + w.val * (WP.mk 16).outStride + (WP.mk 16).lenOff + 4 < 2 ^ 64) :
    Lz4Interleave.outRegion 16 outPtr w.val
      (((siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs r l).toNat) := by
  -- the pc of an `la` store site, read off the emitted array
  have hsites : ∀ (q : Nat) (r' : String), (q, r') ∈ storeSites K16 →
      (r' = "la0" → q = 259) ∧ (r' = "la1" → q = 263) ∧ (r' = "la2" → q = 267)
        ∧ (r' = "la3" → q = 271) := by
    have hall : ∀ (x : Nat × String), x ∈ storeSites K16 →
        (x.2 = "la0" → x.1 = 259) ∧ (x.2 = "la1" → x.1 = 263) ∧ (x.2 = "la2" → x.1 = 267)
          ∧ (x.2 = "la3" → x.1 = 271) := by
      rw [shipped64_store_sites]; decide
    intro q r' hq
    exact hall (q, r') hq
  have hla := la_at_store64 w.val inPtr outPtr gm smemB k l
  have hob : (siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs "outBase" l
      = UInt64.ofNat (outPtr + w.val * (WP.mk 16).outStride) := by
    refine outBase_at_store_site64 w.val inPtr outPtr gm smemB w.isLt
      (by have h := w.isLt; have hn := numBlk3264; omega) hderive k l ?_ ?_ <;>
      · obtain ⟨e0, e1, e2, e3⟩ := hsites _ r hpc
        rcases hr with rfl | rfl | rfl | rfl
        · rw [e0 rfl]; omega
        · rw [e1 rfl]; omega
        · rw [e2 rfl]; omega
        · rw [e3 rfl]; omega
  have hlen : (WP.mk 16).lenOff = 69888 := rfl
  have hstride : (WP.mk 16).outStride = 69896 := rfl
  obtain ⟨e0, e1, e2, e3⟩ := hsites _ r hpc
  obtain ⟨q0, q1, q2, q3⟩ := hla
  refine ⟨?_, ?_⟩ <;>
    · rcases hr with rfl | rfl | rfl | rfl
      · rw [q0 (e0 rfl), hob, UInt64.toNat_add,
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          Nat.mod_eq_of_lt (by omega)]
        have hprod : (WP.mk 16).numBlk * (WP.mk 16).inStride = 209715200 := (by decide); omega
      · rw [q1 (e1 rfl), hob, UInt64.toNat_add,
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          Nat.mod_eq_of_lt (by omega)]
        have hprod : (WP.mk 16).numBlk * (WP.mk 16).inStride = 209715200 := (by decide); omega
      · rw [q2 (e2 rfl), hob, UInt64.toNat_add,
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          Nat.mod_eq_of_lt (by omega)]
        have hprod : (WP.mk 16).numBlk * (WP.mk 16).inStride = 209715200 := (by decide); omega
      · rw [q3 (e3 rfl), hob, UInt64.toNat_add,
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          Nat.mod_eq_of_lt (by omega)]
        have hprod : (WP.mk 16).numBlk * (WP.mk 16).inStride = 209715200 := (by decide); omega

theorem stores_sbAddr64 (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 16).numBlk → inPtr + w * (WP.mk 16).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 16).numBlk →
      inPtr + w * (WP.mk 16).inStride + (WP.mk 16).inStride
        ≤ outPtr + w * (WP.mk 16).outStride)
    (w : Fin (WP.mk 16).numBlk) (k : Nat) (l : Lane)
    (hpc : ((siter K16 k (initSt w.val inPtr outPtr gm smemB)).pc, "sbAddr") ∈ storeSites K16) :
    Lz4Interleave.outRegion 16 outPtr w.val
      (((siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs "sbAddr" l).toNat) := by
  have hmem : (siter K16 k (initSt w.val inPtr outPtr gm smemB)).pc ∈ sbAddrSites := by
    have hall : ∀ (x : Nat × String), x ∈ storeSites K16 → x.2 = "sbAddr" → x.1 ∈ sbAddrSites := by
      rw [shipped64_store_sites]; decide
    exact hall _ hpc rfl
  -- step 0 is the launch state, whose pc is 0
  cases k with
  | zero =>
      exfalso
      have : (siter K16 0 (initSt w.val inPtr outPtr gm smemB)).pc = 0 := rfl
      rw [this] at hmem
      exact absurd hmem (by decide)
  | succ m =>
      exact sbAddr_confined_of_cursor64 inPtr outPtr gm smemB
        (cursorAtSites_shipped64 inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj)
        hderive w m l hmem
        (by have h := htop32 w.val w.isLt
            have hl : (WP.mk 16).lenOff = 69888 := rfl
            have hi : (WP.mk 16).inStride = 65536 := rfl
            omega)

theorem stores_except_copy64 (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 16).numBlk → inPtr + w * (WP.mk 16).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 16).numBlk →
      inPtr + w * (WP.mk 16).inStride + (WP.mk 16).inStride
        ≤ outPtr + w * (WP.mk 16).outStride)
    (w : Fin (WP.mk 16).numBlk) (k : Nat) (l : Lane) (r : String)
    (hpc : ((siter K16 k (initSt w.val inPtr outPtr gm smemB)).pc, r) ∈ storeSites K16)
    (hnc : r ≠ "cpDo") :
    Lz4Interleave.outRegion 16 outPtr w.val
      (((siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs r l).toNat) := by
  have hregs : ∀ (x : Nat × String), x ∈ storeSites K16 →
      x.2 = "sbAddr" ∨ x.2 = "cpDo" ∨ x.2 = "la0" ∨ x.2 = "la1" ∨ x.2 = "la2" ∨ x.2 = "la3" := by
    rw [shipped64_store_sites]; decide
  have htop64 : outPtr + w.val * (WP.mk 16).outStride + (WP.mk 16).lenOff + 4 < 2 ^ 64 := by
    have h := htop32 w.val w.isLt
    have hl : (WP.mk 16).lenOff = 69888 := rfl
    have hi : (WP.mk 16).inStride = 65536 := rfl
    omega
  rcases hregs _ hpc with e | e | e | e | e | e
  · subst e; exact stores_sbAddr64 inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj w k l hpc
  · exact absurd e hnc
  · exact stores_la64 inPtr outPtr gm smemB hderive w k l r (Or.inl e) hpc htop64
  · exact stores_la64 inPtr outPtr gm smemB hderive w k l r (Or.inr (Or.inl e)) hpc htop64
  · exact stores_la64 inPtr outPtr gm smemB hderive w k l r (Or.inr (Or.inr (Or.inl e))) hpc htop64
  · exact stores_la64 inPtr outPtr gm smemB hderive w k l r (Or.inr (Or.inr (Or.inr e))) hpc htop64

-- ── The literal copy loop, pcs 152–168 ───────────────────────────────────────

def copyS64 : List Nat := (List.range 18).map (· + 152)

theorem copyS_closed64 : PcClosed K16 copyS64 [169] :=
  ivClosed_at K16 152 18 [169] kSize16 (by omega) (by decide)

theorem copyS_entry_lt64 : ∀ q, q < 274 → q ∉ copyS64 →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q, q' ∈ copyS64 → q' = 152 :=
  ivEntry_at K16 152 18 152 kSize16 (by decide)

theorem copyS_entry64 : ∀ q, q ∉ copyS64 →
    ∀ q', q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q → q' ∈ copyS64 → q' = 152 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact copyS_entry_lt64 q h hq q' hq' hin
  · rw [show AlgorithmLib.LZ4Simt.succsOf K16 q = [q] from by
      simp only [AlgorithmLib.LZ4Simt.succsOf,
        Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

theorem copy_no_dest64 : ∀ r ∈ ["op", "outBase", "litLen", "cpDst", "lane"],
    ∀ q ∈ ((List.range 14).map (· + 156)),
      (K16[q]?.map (fun i => destOf i != some r)) = some true := by decide

theorem copyLoop_closed64 : PcClosed K16 ((List.range 14).map (· + 156)) [169] :=
  ivClosed_at K16 156 14 [169] kSize16 (by omega) (by decide)

theorem copy_head64 (E : SState) (h152 : E.pc = 152) :
    (∀ i, i ≤ 4 → (siter K16 i E).pc = 152 + i)
    ∧ (siter K16 4 E).pc = 156
    ∧ (∀ l : Lane, (siter K16 4 E).regs "cpDst" l = E.regs "outBase" l + E.regs "op" l)
    ∧ (∀ r : String, r = "op" ∨ r = "outBase" ∨ r = "litLen" →
        (siter K16 4 E).regs r = E.regs r) := by
  have e0 : siter K16 0 E = E := rfl
  have p1 : (siter K16 1 E).pc = 153 := by
    rw [siter_succ, e0]
    exact tail_pc_bin64 E 152 (.add) "cpDst" "outBase" (SArg.reg "op") h152 (by decide)
  have p2 : (siter K16 2 E).pc = 154 := by
    rw [siter_succ]
    exact tail_pc_bin64 _ 153 (.add) "cpSrc" "inBase" (SArg.reg "litAnchor") p1 (by decide)
  have p3 : (siter K16 3 E).pc = 155 := by
    rw [siter_succ]; exact tail_pc_mov64 _ 154 "cpI" (SArg.imm 0) p2 (by decide)
  have p4 : (siter K16 4 E).pc = 156 := by
    rw [siter_succ]
    exact tail_pc_setp64 _ 155 (.lt) "cpCont" "cpI" (SArg.reg "litLen") p3 (by decide)
  refine ⟨fun i hi => ?_, p4, fun l => ?_, fun r hr => ?_⟩
  · match i, hi with
    | 0, _ => rw [e0]; exact h152
    | 1, _ => exact p1
    | 2, _ => exact p2
    | 3, _ => exact p3
    | 4, _ => exact p4
  · have f1 : (siter K16 1 E).regs "cpDst" l = E.regs "outBase" l + E.regs "op" l := by
      rw [siter_succ, e0,
        tail_val_bin64 E 152 (.add) "cpDst" "outBase" (SArg.reg "op") h152 (by decide) l]
      rfl
    have f2 : (siter K16 2 E).regs "cpDst" = (siter K16 1 E).regs "cpDst" := by
      rw [siter_succ]
      exact tail_frame64 _ 153 (.bin (.add) "cpSrc" "inBase" (SArg.reg "litAnchor")) "cpDst" p1
        (by decide) (by decide)
    have f3 : (siter K16 3 E).regs "cpDst" = (siter K16 2 E).regs "cpDst" := by
      rw [siter_succ]
      exact tail_frame64 _ 154 (.mov "cpI" (SArg.imm 0)) "cpDst" p2 (by decide) (by decide)
    have f4 : (siter K16 4 E).regs "cpDst" = (siter K16 3 E).regs "cpDst" := by
      rw [siter_succ]
      exact tail_frame64 _ 155 (.setp (.lt) "cpCont" "cpI" (SArg.reg "litLen")) "cpDst" p3
        (by decide) (by decide)
    rw [show (siter K16 4 E).regs "cpDst" l = (siter K16 4 E).regs "cpDst" l from rfl, f4, f3, f2, f1]
  · have hne : r ≠ "cpDst" ∧ r ≠ "cpSrc" ∧ r ≠ "cpI" ∧ r ≠ "cpCont" := by
      rcases hr with rfl | rfl | rfl <;> exact ⟨by decide, by decide, by decide, by decide⟩
    have f1 : (siter K16 1 E).regs r = E.regs r := by
      rw [siter_succ, e0]
      exact tail_frame64 E 152 (.bin (.add) "cpDst" "outBase" (SArg.reg "op")) r h152 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.1)
    have f2 : (siter K16 2 E).regs r = (siter K16 1 E).regs r := by
      rw [siter_succ]
      exact tail_frame64 _ 153 (.bin (.add) "cpSrc" "inBase" (SArg.reg "litAnchor")) r p1 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.2.1)
    have f3 : (siter K16 3 E).regs r = (siter K16 2 E).regs r := by
      rw [siter_succ]
      exact tail_frame64 _ 154 (.mov "cpI" (SArg.imm 0)) r p2 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.2.2.1)
    have f4 : (siter K16 4 E).regs r = (siter K16 3 E).regs r := by
      rw [siter_succ]
      exact tail_frame64 _ 155 (.setp (.lt) "cpCont" "cpI" (SArg.reg "litLen")) r p3 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.2.2.2)
    rw [f4, f3, f2, f1]

theorem tail_val_binr64 (st : SState) (q : Nat) (o : SOp) (d a b : String) (hq : st.pc = q)
    (hi : K16[q]? = some (.binr o d a b)) (l : Lane) :
    (sstep K16 st).regs d l = o.run (st.regs a l) (st.regs b l) := by
  rw [sstep, show K16[st.pc]? = some (.binr o d a b) from by rw [hq]; exact hi]
  simp only [sstepInstr, SState.setPc, SState.setReg, eq_self_iff_true, if_true]

theorem tail_pc_binr64 (st : SState) (q : Nat) (o : SOp) (d a b : String) (hq : st.pc = q)
    (hi : K16[q]? = some (.binr o d a b)) : (sstep K16 st).pc = q + 1 := by
  rw [sstep, show K16[st.pc]? = some (.binr o d a b) from by rw [hq]; exact hi]
  show st.pc + 1 = q + 1; rw [hq]

theorem copy_pred64 (E : SState) (h162 : E.pc = 162) :
    (∀ i, i ≤ 3 → (siter K16 i E).pc = 162 + i)
    ∧ (siter K16 3 E).pc = 165
    ∧ (∀ l : Lane, (siter K16 3 E).regs "cpP" l = 1 →
        E.regs "cpI" l + E.regs "lane" l < E.regs "litLen" l)
    ∧ (∀ r : String, r = "cpI" ∨ r = "lane" ∨ r = "litLen" ∨ r = "cpDst" ∨ r = "cpDo" →
        (siter K16 3 E).regs r = E.regs r)
    ∧ (∀ l : Lane, (siter K16 3 E).regs "cpJ" l = E.regs "cpI" l + E.regs "lane" l) := by
  have e0 : siter K16 0 E = E := rfl
  have p1 : (siter K16 1 E).pc = 163 := by
    rw [siter_succ, e0]; exact tail_pc_binr64 E 162 (.add) "cpJ" "cpI" "lane" h162 (by decide)
  have p2 : (siter K16 2 E).pc = 164 := by
    rw [siter_succ]
    exact tail_pc_setp64 _ 163 (.lt) "cpP" "cpJ" (SArg.reg "litLen") p1 (by decide)
  have p3 : (siter K16 3 E).pc = 165 := by
    rw [siter_succ]
    rw [sstep, show K16[(siter K16 2 E).pc]? = some (.ldgo "cpB" "cpSo" 0) from by rw [p2]; decide]
    show (siter K16 2 E).pc + 1 = 165
    rw [p2]
  have hfr : ∀ r : String, r ≠ "cpJ" → r ≠ "cpP" → r ≠ "cpB" →
      (siter K16 3 E).regs r = E.regs r := by
    intro r h1 h2 h3
    have f1 : (siter K16 1 E).regs r = E.regs r := by
      rw [siter_succ, e0]
      exact tail_frame64 E 162 (.binr (.add) "cpJ" "cpI" "lane") r h162 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h1)
    have f2 : (siter K16 2 E).regs r = (siter K16 1 E).regs r := by
      rw [siter_succ]
      exact tail_frame64 _ 163 (.setp (.lt) "cpP" "cpJ" (SArg.reg "litLen")) r p1 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h2)
    have f3 : (siter K16 3 E).regs r = (siter K16 2 E).regs r := by
      rw [siter_succ]
      exact tail_frame64 _ 164 (.ldgo "cpB" "cpSo" 0) r p2 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h3)
    rw [f3, f2, f1]
  refine ⟨fun i hi => ?_, p3, fun l hp => ?_, fun r hr => ?_, fun l => ?_⟩
  · match i, hi with
    | 0, _ => rw [e0]; exact h162
    | 1, _ => exact p1
    | 2, _ => exact p2
    | 3, _ => exact p3
  · -- `cpP` survives the load, and at 163 it is the comparison
    have fP : (siter K16 3 E).regs "cpP" l = (siter K16 2 E).regs "cpP" l := by
      rw [siter_succ]
      exact congrFun (tail_frame64 _ 164 (.ldgo "cpB" "cpSo" 0) "cpP" p2 (by decide) (by decide)) l
    have vP : (siter K16 2 E).regs "cpP" l
        = (if SCmp.run (.lt) ((siter K16 1 E).regs "cpJ" l)
            ((siter K16 1 E).get l (SArg.reg "litLen")) then 1 else 0) := by
      rw [siter_succ]
      exact tail_val_setp64 _ 163 (.lt) "cpP" "cpJ" (SArg.reg "litLen") p1 (by decide) l
    have vJ : (siter K16 1 E).regs "cpJ" l = E.regs "cpI" l + E.regs "lane" l := by
      rw [siter_succ, e0, tail_val_binr64 E 162 (.add) "cpJ" "cpI" "lane" h162 (by decide) l]
      rfl
    have vL : (siter K16 1 E).get l (SArg.reg "litLen") = E.regs "litLen" l := by
      rw [siter_succ, e0]
      show (sstep K16 E).regs "litLen" l = _
      exact congrFun (tail_frame64 E 162 (.binr (.add) "cpJ" "cpI" "lane") "litLen" h162
        (by decide) (by decide)) l
    rw [fP, vP, vJ, vL] at hp
    by_cases hlt : SCmp.run (.lt) (E.regs "cpI" l + E.regs "lane" l) (E.regs "litLen" l)
    · exact of_decide_eq_true (by simpa only [SCmp.run] using hlt)
    · rw [if_neg hlt] at hp; exact absurd hp (by decide)
  · rcases hr with rfl | rfl | rfl | rfl | rfl <;>
      exact hfr _ (by decide) (by decide) (by decide)
  · have fJ : (siter K16 3 E).regs "cpJ" l = (siter K16 1 E).regs "cpJ" l := by
      have a2 : (siter K16 2 E).regs "cpJ" = (siter K16 1 E).regs "cpJ" := by
        rw [siter_succ]
        exact tail_frame64 _ 163 (.setp (.lt) "cpP" "cpJ" (SArg.reg "litLen")) "cpJ" p1
          (by decide) (by decide)
      have a3 : (siter K16 3 E).regs "cpJ" = (siter K16 2 E).regs "cpJ" := by
        rw [siter_succ]
        exact tail_frame64 _ 164 (.ldgo "cpB" "cpSo" 0) "cpJ" p2 (by decide) (by decide)
      rw [show (siter K16 3 E).regs "cpJ" l = (siter K16 3 E).regs "cpJ" l from rfl, a3, a2]
    rw [fJ, siter_succ, e0, tail_val_binr64 E 162 (.add) "cpJ" "cpI" "lane" h162 (by decide) l]
    rfl

def preStoreS64 : List Nat := (List.range 4).map (· + 162)

theorem preStoreS_entry_lt64 : ∀ q, q < 274 → q ∉ preStoreS64 →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q, q' ∈ preStoreS64 → q' = 162 :=
  ivEntry_at K16 162 4 162 kSize16 (by decide)

theorem preStoreS_entry64 : ∀ q, q ∉ preStoreS64 →
    ∀ q', q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q → q' ∈ preStoreS64 → q' = 162 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact preStoreS_entry_lt64 q h hq q' hq' hin
  · rw [show AlgorithmLib.LZ4Simt.succsOf K16 q = [q] from by
      simp only [AlgorithmLib.LZ4Simt.succsOf,
        Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

theorem preStore_succ64 : AlgorithmLib.LZ4Simt.succsOf K16 165 = [166] := by decide

theorem tokInv_at64 (LO : Nat) (hLO : LO < 2 ^ 64) (ss : SState)
    (h0 : ss.pc ∉ tokRegion)
    (l : Lane) (k : Nat)
    (hck : ∀ j, j ≤ k → (siter K16 j ss).pc = 124 →
      AlgorithmLib.LZ4WarpDSL.MatchEntryQ (WP.mk 16).inStride LO (siter K16 j ss))
    (hq : (siter K16 k ss).pc ∈ tokS) :
    TokInv l LO (siter K16 k ss) := by
  have hmem : (siter K16 k ss).pc ∈ tokRegion := by
    have hsub : ∀ q ∈ tokS, q ∈ tokRegion := by decide
    exact hsub _ hq
  obtain ⟨j, hjk, hje, hall⟩ := region_entry K16 tokRegion 124 tokRegion_entry64 ss h0 k hmem
  have hp1 := me_pc164 (siter K16 j ss) hje
  have hp2 := me_pc264 (siter K16 j ss) hje
  have hp3 := me_pc364 (siter K16 j ss) hje
  have hp4 := me_pc464 (siter K16 j ss) hje
  have hp5 := me_pc564 (siter K16 j ss) hje
  have hshift : ∀ a : Nat, siter K16 (j + a) ss = siter K16 a (siter K16 j ss) := by
    intro a; exact siter_add K16 j a ss
  have hstep : ∀ a : Nat, a ≤ 5 → (siter K16 (j + a) ss).pc = 124 + a := by
    intro a ha
    rw [hshift a]
    match a, ha with
    | 0, _ => exact hje
    | 1, _ => exact hp1
    | 2, _ => exact hp2
    | 3, _ => exact hp3
    | 4, _ => exact hp4
    | 5, _ => exact hp5
  -- a site is at least 130, so the visit is past the prologue
  have hk5 : j + 5 ≤ k := by
    rcases Nat.lt_or_ge k (j + 5) with hlt | hge
    · have hle : k - j ≤ 5 := by omega
      have hh := hstep (k - j) hle
      rw [show j + (k - j) = k from by omega] at hh
      have h129 : 129 ≤ (siter K16 k ss).pc := by
        have hlo : ∀ q ∈ tokS, 129 ≤ q := by decide
        exact hlo _ hq
      omega
    · exact hge
  have hTok : TokInv l LO (siter K16 (j + 5) ss) := by
    rw [hshift 5]
    exact tokInv_at_entry64 l LO (siter K16 j ss) hje (hck j hjk hje)
  have hpc129 : (siter K16 (j + 5) ss).pc = 129 := by rw [hshift 5]; exact hp5 ▸ rfl
  -- no exit of the token region is passed before the visit
  have hne : ∀ i, i < k - (j + 5) → (siter K16 i (siter K16 (j + 5) ss)).pc ∉ [197, 198] := by
    intro i hi hin
    rw [← siter_add K16 (j + 5) i ss, show j + 5 + i = j + (5 + i) from by omega] at hin
    have hnext : (siter K16 (j + (5 + i) + 1) ss).pc ∈ tokRegion :=
      hall (j + (5 + i) + 1) (by omega) (by omega)
    have hs : (siter K16 (j + (5 + i) + 1) ss).pc
        ∈ AlgorithmLib.LZ4Simt.succsOf K16 (siter K16 (j + (5 + i)) ss).pc := by
      rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K16 _
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hin
    rcases hin with e | e
    · rw [e, tok_exits64.1, List.mem_singleton] at hs; rw [hs] at hnext; exact absurd hnext (by decide)
    · rw [e, tok_exits64.2, List.mem_singleton] at hs; rw [hs] at hnext; exact absurd hnext (by decide)
  have hinv := inv_on K16 (TokInv l LO) tokS [197, 198] tokS_closed64
    (fun s hsm hexs hh => tokS_hstep64 l LO hLO s hsm hexs hh) (siter K16 (j + 5) ss)
    (by rw [hpc129]; decide) hTok (k - (j + 5)) hne
  rw [← siter_add K16 (j + 5) (k - (j + 5)) ss, show j + 5 + (k - (j + 5)) = k from by omega] at hinv
  exact hinv


theorem cpDo_toNat64 (ob opv cj : UInt64) (base : Nat) (hob : ob = UInt64.ofNat base)
    (hlt : base + opv.toNat + cj.toNat < 2 ^ 64) :
    ((ob + opv) + cj).toNat = base + opv.toNat + cj.toNat := by
  have hb : ob.toNat = base := by
    rw [hob, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by omega)]
  have h1 : (ob + opv).toNat = base + opv.toNat := by
    rw [UInt64.toNat_add, hb, Nat.mod_eq_of_lt (by omega)]
  rw [UInt64.toNat_add, h1, Nat.mod_eq_of_lt (by omega)]

theorem cpDo_region64 (base LO : Nat) (ob opv cj : UInt64) (hob : ob = UInt64.ofNat base)
    (hbud : opv.toNat + cj.toNat < LO) (htop : base + LO + 4 < 2 ^ 64) :
    base ≤ ((ob + opv) + cj).toNat ∧ ((ob + opv) + cj).toNat < base + LO + 4 := by
  rw [cpDo_toNat64 ob opv cj base hob (by omega)]
  omega

theorem preStore_pin64 (ss : SState) (h0p : ss.pc ∉ preStoreS64) (k : Nat)
    (hpc : (siter K16 k ss).pc = 165) :
    ∃ j, k = j + 3 ∧ (siter K16 j ss).pc = 162 := by
  have hmem : (siter K16 k ss).pc ∈ preStoreS64 := by rw [hpc]; decide
  obtain ⟨j, hjk, hje, hall⟩ := region_entry K16 preStoreS64 162 preStoreS_entry64 ss h0p k hmem
  obtain ⟨hpcs, -, -, -, -⟩ := copy_pred64 (siter K16 j ss) hje
  refine ⟨j, ?_, hje⟩
  have hshift : ∀ a : Nat, siter K16 (j + a) ss = siter K16 a (siter K16 j ss) :=
    fun a => siter_add K16 j a ss
  rcases Nat.lt_or_ge k (j + 3) with hlt | hge
  · exfalso
    have hle : k - j ≤ 3 := by omega
    have hh := hpcs (k - j) hle
    rw [← hshift (k - j), show j + (k - j) = k from by omega] at hh
    rw [hpc] at hh; omega
  · rcases Nat.lt_or_ge (j + 3) k with hgt | hle2
    · exfalso
      have h165 : (siter K16 (j + 3) ss).pc = 165 := by
        rw [hshift 3]; exact (copy_pred64 (siter K16 j ss) hje).2.1
      have hs : (siter K16 (j + 3 + 1) ss).pc
          ∈ AlgorithmLib.LZ4Simt.succsOf K16 (siter K16 (j + 3) ss).pc := by
        rw [show (siter K16 (j + 3 + 1) ss) = sstep K16 (siter K16 (j + 3) ss) from siter_succ K16 (j+3) ss]
        exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K16 _
      rw [h165, preStore_succ64, List.mem_singleton] at hs
      have hin := hall (j + 3 + 1) (by omega) (by omega)
      rw [hs] at hin
      exact absurd hin (by decide)
    · omega

theorem copy_head_pin64 (ss : SState) (h0c : ss.pc ∉ copyS64) (k : Nat)
    (hpc : (siter K16 k ss).pc = 165) :
    ∃ j, j + 4 ≤ k ∧ (siter K16 j ss).pc = 152
      ∧ ∀ i, j ≤ i → i ≤ k → (siter K16 i ss).pc ∈ copyS64 := by
  have hmem : (siter K16 k ss).pc ∈ copyS64 := by rw [hpc]; decide
  obtain ⟨j, hjk, hje, hall⟩ := region_entry K16 copyS64 152 copyS_entry64 ss h0c k hmem
  obtain ⟨hpcs, -, -, -⟩ := copy_head64 (siter K16 j ss) hje
  refine ⟨j, ?_, hje, hall⟩
  rcases Nat.lt_or_ge k (j + 4) with hlt | hge
  · exfalso
    have hle : k - j ≤ 4 := by omega
    have hh := hpcs (k - j) hle
    rw [← siter_add K16 j (k - j) ss, show j + (k - j) = k from by omega] at hh
    rw [hpc] at hh; omega
  · exact hge

theorem copyLoop_exit_succ64 : AlgorithmLib.LZ4Simt.succsOf K16 169 = [170] := by decide

theorem copy_const64 (ss : SState) (j k : Nat) (hjk : j ≤ k)
    (h156 : (siter K16 j ss).pc = 156)
    (hall : ∀ i, j ≤ i → i ≤ k → (siter K16 i ss).pc ∈ copyS64)
    (r : String) (hr : r ∈ ["op", "outBase", "litLen", "cpDst", "lane"]) :
    (siter K16 k ss).regs r = (siter K16 j ss).regs r := by
  have hne : ∀ i, i < k - j → (siter K16 i (siter K16 j ss)).pc ∉ [169] := by
    intro i hi hin
    rw [← siter_add K16 j i ss] at hin
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hin
    have hs : (siter K16 (j + i + 1) ss).pc
        ∈ AlgorithmLib.LZ4Simt.succsOf K16 (siter K16 (j + i) ss).pc := by
      rw [show siter K16 (j + i + 1) ss = sstep K16 (siter K16 (j + i) ss) from siter_succ K16 (j + i) ss]
      exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K16 _
    rw [hin, copyLoop_exit_succ64, List.mem_singleton] at hs
    have hmem := hall (j + i + 1) (by omega) (by omega)
    rw [hs] at hmem
    exact absurd hmem (by decide)
  have hconst := AlgorithmLib.LZ4Simt.regs_const_on K16 r ((List.range 14).map (· + 156)) [169]
    copyLoop_closed64 (copy_no_dest64 r hr) (siter K16 j ss) (by rw [h156]; decide) (k - j) hne
  rw [← siter_add K16 j (k - j) ss, show j + (k - j) = k from by omega] at hconst
  exact hconst

theorem copy_budget64 (l : Lane) (LO : Nat) (st : SState) (h : TokInv l LO st)
    (hpc : st.pc = 152) :
    (st.regs "op" l).toNat + (st.regs "litLen" l).toNat + 2 ≤ LO := by
  have h1 := h.1
  rw [hpc] at h1
  have hr : tokRem 152 ((st.regs "litLen" l).toNat) ((st.regs "litExtra" l).toNat)
      ((st.regs "mlm" l).toNat) ((st.regs "matExtra" l).toNat)
      = (st.regs "litLen" l).toNat + 2 + lsicLen ((st.regs "mlm" l).toNat) := rfl
  rw [hr] at h1
  omega

theorem cpDo_confined64 (LO : Nat) (hLO : LO < 2 ^ 64) (base : Nat) (ss : SState)
    (h0t : ss.pc ∉ tokRegion) (h0c : ss.pc ∉ copyS64) (h0p : ss.pc ∉ preStoreS64)
    (l : Lane) (k : Nat)
    (hck : ∀ j, j ≤ k → (siter K16 j ss).pc = 124 →
      AlgorithmLib.LZ4WarpDSL.MatchEntryQ (WP.mk 16).inStride LO (siter K16 j ss))
    (hpc : (siter K16 k ss).pc = 165)
    (hact : (siter K16 k ss).regs "cpP" l = 1)
    (hcpdo : (siter K16 k ss).regs "cpDo" l
      = ((siter K16 k ss).regs "cpDst" l + (siter K16 k ss).regs "cpI" l)
        + (siter K16 k ss).regs "lane" l)
    (hob : (siter K16 k ss).regs "outBase" l = UInt64.ofNat base)
    (htop : base + LO + 4 < 2 ^ 64) :
    base ≤ ((siter K16 k ss).regs "cpDo" l).toNat
      ∧ ((siter K16 k ss).regs "cpDo" l).toNat < base + LO + 4 := by
  -- the predicate half
  obtain ⟨j2, hk3, h162⟩ := preStore_pin64 ss h0p k hpc
  have hE2 : siter K16 3 (siter K16 j2 ss) = siter K16 k ss := by
    rw [← siter_add K16 j2 3 ss, ← hk3]
  obtain ⟨-, -, hpred, hfr2, hJ⟩ := copy_pred64 (siter K16 j2 ss) h162
  rw [hE2] at hpred hfr2 hJ
  have hlt : (siter K16 j2 ss).regs "cpI" l + (siter K16 j2 ss).regs "lane" l
      < (siter K16 j2 ss).regs "litLen" l := hpred l hact
  -- the setup half
  obtain ⟨j1, h4k, h152, hall⟩ := copy_head_pin64 ss h0c k hpc
  obtain ⟨-, h156, hDst, hFr⟩ := copy_head64 (siter K16 j1 ss) h152
  have hE1 : siter K16 4 (siter K16 j1 ss) = siter K16 (j1 + 4) ss := (siter_add K16 j1 4 ss).symm
  rw [hE1] at h156 hDst hFr
  have hconst : ∀ r : String, r ∈ ["op", "outBase", "litLen", "cpDst", "lane"] →
      (siter K16 k ss).regs r = (siter K16 (j1 + 4) ss).regs r :=
    fun r hr => copy_const64 ss (j1 + 4) k h4k h156 (fun i h1 h2 => hall i (by omega) h2) r hr
  -- the budget
  have htok := tokInv_at64 LO hLO ss h0t l j1 (fun j hj => hck j (by omega)) (by rw [h152]; decide)
  have hbud := copy_budget64 l LO (siter K16 j1 ss) htok h152
  -- move everything to the visit
  have hopk : (siter K16 k ss).regs "op" l = (siter K16 j1 ss).regs "op" l := by
    rw [congrFun (hconst "op" (by decide)) l, congrFun (hFr "op" (Or.inl rfl)) l]
  have hllk : (siter K16 k ss).regs "litLen" l = (siter K16 j1 ss).regs "litLen" l := by
    rw [congrFun (hconst "litLen" (by decide)) l,
      congrFun (hFr "litLen" (Or.inr (Or.inr rfl))) l]
  have hobk : (siter K16 k ss).regs "outBase" l = (siter K16 j1 ss).regs "outBase" l := by
    rw [congrFun (hconst "outBase" (by decide)) l, congrFun (hFr "outBase" (Or.inr (Or.inl rfl))) l]
  have hdstk : (siter K16 k ss).regs "cpDst" l
      = (siter K16 j1 ss).regs "outBase" l + (siter K16 j1 ss).regs "op" l := by
    rw [congrFun (hconst "cpDst" (by decide)) l, hDst l]
  -- the cursor plus this lane's offset is inside the budget
  have e1 : (siter K16 k ss).regs "cpI" l = (siter K16 j2 ss).regs "cpI" l :=
    congrFun (hfr2 "cpI" (Or.inl rfl)) l
  have e2 : (siter K16 k ss).regs "lane" l = (siter K16 j2 ss).regs "lane" l :=
    congrFun (hfr2 "lane" (Or.inr (Or.inl rfl))) l
  have e3 : (siter K16 k ss).regs "litLen" l = (siter K16 j2 ss).regs "litLen" l :=
    congrFun (hfr2 "litLen" (Or.inr (Or.inr (Or.inl rfl)))) l
  have hcjU : (siter K16 k ss).regs "cpI" l + (siter K16 k ss).regs "lane" l
      < (siter K16 k ss).regs "litLen" l := by rw [e1, e2, e3]; exact hlt
  have hcjlt : ((siter K16 k ss).regs "cpI" l + (siter K16 k ss).regs "lane" l).toNat
      < ((siter K16 j1 ss).regs "litLen" l).toNat := by
    rw [← hllk]; exact UInt64.lt_iff_toNat_lt.mp hcjU
  have hfinal : (siter K16 k ss).regs "cpDo" l
      = ((siter K16 j1 ss).regs "outBase" l + (siter K16 j1 ss).regs "op" l)
        + ((siter K16 k ss).regs "cpI" l + (siter K16 k ss).regs "lane" l) := by
    rw [hcpdo, hdstk, UInt64.add_assoc]
  rw [hfinal]
  refine cpDo_region64 base LO ((siter K16 j1 ss).regs "outBase" l) ((siter K16 j1 ss).regs "op" l)
    ((siter K16 k ss).regs "cpI" l + (siter K16 k ss).regs "lane" l) (by rw [← hobk]; exact hob)
    (by omega) htop

theorem matchEntry_anywhere64 (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 16).numBlk → inPtr + w * (WP.mk 16).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 16).numBlk →
      inPtr + w * (WP.mk 16).inStride + (WP.mk 16).inStride
        ≤ outPtr + w * (WP.mk 16).outStride)
    (w : Fin (WP.mk 16).numBlk) :
    ∀ j, (siter K16 j (initSt w.val inPtr outPtr gm smemB)).pc = 124 →
      AlgorithmLib.LZ4WarpDSL.MatchEntryQ (WP.mk 16).inStride (WP.mk 16).lenOff
        (siter K16 j (initSt w.val inPtr outPtr gm smemB)) := by
  obtain ⟨n, hpc39, hpc40, hck, -, -⟩ :=
    shipped_loop_ckpt64 w.val inPtr outPtr gm smemB w.isLt
      (by have := w.isLt; have := numBlk3264; omega)
      (hib40 _ w.isLt) (htop32 _ w.isLt) (hbuf _ w.isLt) hderive (hdisj _ w.isLt)
  intro j hj
  have hmem : (siter K16 j (initSt w.val inPtr outPtr gm smemB)).pc ∈ bodyRegion := by
    rw [hj]; decide
  obtain ⟨hge, -, -⟩ :=
    body_entry_at64 (initSt w.val inPtr outPtr gm smemB) preSteps hpc39 j hmem
  have hidx : preSteps + 1 + (j - (preSteps + 1)) = j := by omega
  have hres := hck (j - (preSteps + 1)) (by rw [hidx]; exact hj)
  rw [hidx] at hres
  exact hres

theorem stores_cpDo16564 (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 16).numBlk → inPtr + w * (WP.mk 16).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 16).numBlk →
      inPtr + w * (WP.mk 16).inStride + (WP.mk 16).inStride
        ≤ outPtr + w * (WP.mk 16).outStride)
    (w : Fin (WP.mk 16).numBlk) (k : Nat) (l : Lane)
    (hpc : (siter K16 k (initSt w.val inPtr outPtr gm smemB)).pc = 165)
    (hact : (siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs "cpP" l = 1) :
    Lz4Interleave.outRegion 16 outPtr w.val
      (((siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs "cpDo" l).toNat) := by
  have h0 : (initSt w.val inPtr outPtr gm smemB).pc = 0 := rfl
  have hlen : (WP.mk 16).lenOff = 69888 := rfl
  have hstr : (WP.mk 16).inStride = 65536 := rfl
  exact cpDo_confined64 (WP.mk 16).lenOff (by decide) (outPtr + w.val * (WP.mk 16).outStride)
    (initSt w.val inPtr outPtr gm smemB)
    (by rw [h0]; decide) (by rw [h0]; decide) (by rw [h0]; decide) l k
    (fun j _ hj => matchEntry_anywhere64 inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj w j hj)
    hpc hact
    ((cpDo_at_store64 w.val inPtr outPtr gm smemB k l).1 hpc)
    (outBase_at_store_site64 w.val inPtr outPtr gm smemB w.isLt
      (by have := w.isLt; have := numBlk3264; omega) hderive k l
      (by rw [hpc]; omega) (by rw [hpc]; omega))
    (by have h := htop32 w.val w.isLt; omega)

-- ── The tail copy, pcs 238–254 (same shape, shifted by 86) ──────────────────
def copyS264 : List Nat := (List.range 18).map (· + 238)

theorem copyS2_closed64 : PcClosed K16 copyS264 [255] :=
  ivClosed_at K16 238 18 [255] kSize16 (by omega) (by decide)

theorem copyS2_entryLt64 : ∀ q, q < 274 → q ∉ copyS264 →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q, q' ∈ copyS264 → q' = 238 :=
  ivEntry_at K16 238 18 238 kSize16 (by decide)

theorem copyS2_entry64 : ∀ q, q ∉ copyS264 →
    ∀ q', q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q → q' ∈ copyS264 → q' = 238 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact copyS2_entryLt64 q h hq q' hq' hin
  · rw [show AlgorithmLib.LZ4Simt.succsOf K16 q = [q] from by
      simp only [AlgorithmLib.LZ4Simt.succsOf,
        Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq


theorem copy_no_dest264 : ∀ r ∈ ["op", "outBase", "fLen", "cpDstF", "lane"],
    ∀ q ∈ ((List.range 14).map (· + 242)),
      (K16[q]?.map (fun i => destOf i != some r)) = some true := by decide

theorem copyLoop_closed264 : PcClosed K16 ((List.range 14).map (· + 242)) [255] :=
  ivClosed_at K16 242 14 [255] kSize16 (by omega) (by decide)


theorem copy_head264 (E : SState) (h238 : E.pc = 238) :
    (∀ i, i ≤ 4 → (siter K16 i E).pc = 238 + i)
    ∧ (siter K16 4 E).pc = 242
    ∧ (∀ l : Lane, (siter K16 4 E).regs "cpDstF" l = E.regs "outBase" l + E.regs "op" l)
    ∧ (∀ r : String, r = "op" ∨ r = "outBase" ∨ r = "fLen" →
        (siter K16 4 E).regs r = E.regs r) := by
  have e0 : siter K16 0 E = E := rfl
  have p1 : (siter K16 1 E).pc = 239 := by
    rw [siter_succ, e0]
    exact tail_pc_bin64 E 238 (.add) "cpDstF" "outBase" (SArg.reg "op") h238 (by decide)
  have p2 : (siter K16 2 E).pc = 240 := by
    rw [siter_succ]
    exact tail_pc_bin64 _ 239 (.add) "cpSrcF" "inBase" (SArg.reg "litAnchor") p1 (by decide)
  have p3 : (siter K16 3 E).pc = 241 := by
    rw [siter_succ]; exact tail_pc_mov64 _ 240 "cpI" (SArg.imm 0) p2 (by decide)
  have p4 : (siter K16 4 E).pc = 242 := by
    rw [siter_succ]
    exact tail_pc_setp64 _ 241 (.lt) "cpCont" "cpI" (SArg.reg "fLen") p3 (by decide)
  refine ⟨fun i hi => ?_, p4, fun l => ?_, fun r hr => ?_⟩
  · match i, hi with
    | 0, _ => rw [e0]; exact h238
    | 1, _ => exact p1
    | 2, _ => exact p2
    | 3, _ => exact p3
    | 4, _ => exact p4
  · have f1 : (siter K16 1 E).regs "cpDstF" l = E.regs "outBase" l + E.regs "op" l := by
      rw [siter_succ, e0,
        tail_val_bin64 E 238 (.add) "cpDstF" "outBase" (SArg.reg "op") h238 (by decide) l]
      rfl
    have f2 : (siter K16 2 E).regs "cpDstF" = (siter K16 1 E).regs "cpDstF" := by
      rw [siter_succ]
      exact tail_frame64 _ 239 (.bin (.add) "cpSrcF" "inBase" (SArg.reg "litAnchor")) "cpDstF" p1
        (by decide) (by decide)
    have f3 : (siter K16 3 E).regs "cpDstF" = (siter K16 2 E).regs "cpDstF" := by
      rw [siter_succ]
      exact tail_frame64 _ 240 (.mov "cpI" (SArg.imm 0)) "cpDstF" p2 (by decide) (by decide)
    have f4 : (siter K16 4 E).regs "cpDstF" = (siter K16 3 E).regs "cpDstF" := by
      rw [siter_succ]
      exact tail_frame64 _ 241 (.setp (.lt) "cpCont" "cpI" (SArg.reg "fLen")) "cpDstF" p3
        (by decide) (by decide)
    rw [show (siter K16 4 E).regs "cpDstF" l = (siter K16 4 E).regs "cpDstF" l from rfl, f4, f3, f2, f1]
  · have hne : r ≠ "cpDstF" ∧ r ≠ "cpSrcF" ∧ r ≠ "cpI" ∧ r ≠ "cpCont" := by
      rcases hr with rfl | rfl | rfl <;> exact ⟨by decide, by decide, by decide, by decide⟩
    have f1 : (siter K16 1 E).regs r = E.regs r := by
      rw [siter_succ, e0]
      exact tail_frame64 E 238 (.bin (.add) "cpDstF" "outBase" (SArg.reg "op")) r h238 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.1)
    have f2 : (siter K16 2 E).regs r = (siter K16 1 E).regs r := by
      rw [siter_succ]
      exact tail_frame64 _ 239 (.bin (.add) "cpSrcF" "inBase" (SArg.reg "litAnchor")) r p1 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.2.1)
    have f3 : (siter K16 3 E).regs r = (siter K16 2 E).regs r := by
      rw [siter_succ]
      exact tail_frame64 _ 240 (.mov "cpI" (SArg.imm 0)) r p2 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.2.2.1)
    have f4 : (siter K16 4 E).regs r = (siter K16 3 E).regs r := by
      rw [siter_succ]
      exact tail_frame64 _ 241 (.setp (.lt) "cpCont" "cpI" (SArg.reg "fLen")) r p3 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.2.2.2)
    rw [f4, f3, f2, f1]


theorem copy_pred264 (E : SState) (h248 : E.pc = 248) :
    (∀ i, i ≤ 3 → (siter K16 i E).pc = 248 + i)
    ∧ (siter K16 3 E).pc = 251
    ∧ (∀ l : Lane, (siter K16 3 E).regs "cpP" l = 1 →
        E.regs "cpI" l + E.regs "lane" l < E.regs "fLen" l)
    ∧ (∀ r : String, r = "cpI" ∨ r = "lane" ∨ r = "fLen" ∨ r = "cpDstF" ∨ r = "cpDo" →
        (siter K16 3 E).regs r = E.regs r)
    ∧ (∀ l : Lane, (siter K16 3 E).regs "cpJ" l = E.regs "cpI" l + E.regs "lane" l) := by
  have e0 : siter K16 0 E = E := rfl
  have p1 : (siter K16 1 E).pc = 249 := by
    rw [siter_succ, e0]; exact tail_pc_binr64 E 248 (.add) "cpJ" "cpI" "lane" h248 (by decide)
  have p2 : (siter K16 2 E).pc = 250 := by
    rw [siter_succ]
    exact tail_pc_setp64 _ 249 (.lt) "cpP" "cpJ" (SArg.reg "fLen") p1 (by decide)
  have p3 : (siter K16 3 E).pc = 251 := by
    rw [siter_succ]
    rw [sstep, show K16[(siter K16 2 E).pc]? = some (.ldgo "cpB" "cpSo" 0) from by rw [p2]; decide]
    show (siter K16 2 E).pc + 1 = 251
    rw [p2]
  have hfr : ∀ r : String, r ≠ "cpJ" → r ≠ "cpP" → r ≠ "cpB" →
      (siter K16 3 E).regs r = E.regs r := by
    intro r h1 h2 h3
    have f1 : (siter K16 1 E).regs r = E.regs r := by
      rw [siter_succ, e0]
      exact tail_frame64 E 248 (.binr (.add) "cpJ" "cpI" "lane") r h248 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h1)
    have f2 : (siter K16 2 E).regs r = (siter K16 1 E).regs r := by
      rw [siter_succ]
      exact tail_frame64 _ 249 (.setp (.lt) "cpP" "cpJ" (SArg.reg "fLen")) r p1 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h2)
    have f3 : (siter K16 3 E).regs r = (siter K16 2 E).regs r := by
      rw [siter_succ]
      exact tail_frame64 _ 250 (.ldgo "cpB" "cpSo" 0) r p2 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h3)
    rw [f3, f2, f1]
  refine ⟨fun i hi => ?_, p3, fun l hp => ?_, fun r hr => ?_, fun l => ?_⟩
  · match i, hi with
    | 0, _ => rw [e0]; exact h248
    | 1, _ => exact p1
    | 2, _ => exact p2
    | 3, _ => exact p3
  · -- `cpP` survives the load, and at 249 it is the comparison
    have fP : (siter K16 3 E).regs "cpP" l = (siter K16 2 E).regs "cpP" l := by
      rw [siter_succ]
      exact congrFun (tail_frame64 _ 250 (.ldgo "cpB" "cpSo" 0) "cpP" p2 (by decide) (by decide)) l
    have vP : (siter K16 2 E).regs "cpP" l
        = (if SCmp.run (.lt) ((siter K16 1 E).regs "cpJ" l)
            ((siter K16 1 E).get l (SArg.reg "fLen")) then 1 else 0) := by
      rw [siter_succ]
      exact tail_val_setp64 _ 249 (.lt) "cpP" "cpJ" (SArg.reg "fLen") p1 (by decide) l
    have vJ : (siter K16 1 E).regs "cpJ" l = E.regs "cpI" l + E.regs "lane" l := by
      rw [siter_succ, e0, tail_val_binr64 E 248 (.add) "cpJ" "cpI" "lane" h248 (by decide) l]
      rfl
    have vL : (siter K16 1 E).get l (SArg.reg "fLen") = E.regs "fLen" l := by
      rw [siter_succ, e0]
      show (sstep K16 E).regs "fLen" l = _
      exact congrFun (tail_frame64 E 248 (.binr (.add) "cpJ" "cpI" "lane") "fLen" h248
        (by decide) (by decide)) l
    rw [fP, vP, vJ, vL] at hp
    by_cases hlt : SCmp.run (.lt) (E.regs "cpI" l + E.regs "lane" l) (E.regs "fLen" l)
    · exact of_decide_eq_true (by simpa only [SCmp.run] using hlt)
    · rw [if_neg hlt] at hp; exact absurd hp (by decide)
  · rcases hr with rfl | rfl | rfl | rfl | rfl <;>
      exact hfr _ (by decide) (by decide) (by decide)
  · have fJ : (siter K16 3 E).regs "cpJ" l = (siter K16 1 E).regs "cpJ" l := by
      have a2 : (siter K16 2 E).regs "cpJ" = (siter K16 1 E).regs "cpJ" := by
        rw [siter_succ]
        exact tail_frame64 _ 249 (.setp (.lt) "cpP" "cpJ" (SArg.reg "fLen")) "cpJ" p1
          (by decide) (by decide)
      have a3 : (siter K16 3 E).regs "cpJ" = (siter K16 2 E).regs "cpJ" := by
        rw [siter_succ]
        exact tail_frame64 _ 250 (.ldgo "cpB" "cpSo" 0) "cpJ" p2 (by decide) (by decide)
      rw [show (siter K16 3 E).regs "cpJ" l = (siter K16 3 E).regs "cpJ" l from rfl, a3, a2]
    rw [fJ, siter_succ, e0, tail_val_binr64 E 248 (.add) "cpJ" "cpI" "lane" h248 (by decide) l]
    rfl

def preStoreS264 : List Nat := (List.range 4).map (· + 248)

theorem preStoreS2_entry_lt64 : ∀ q, q < 274 → q ∉ preStoreS264 →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q, q' ∈ preStoreS264 → q' = 248 :=
  ivEntry_at K16 248 4 248 kSize16 (by decide)

theorem preStoreS2_entry64 : ∀ q, q ∉ preStoreS264 →
    ∀ q', q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q → q' ∈ preStoreS264 → q' = 248 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact preStoreS2_entry_lt64 q h hq q' hq' hin
  · rw [show AlgorithmLib.LZ4Simt.succsOf K16 q = [q] from by
      simp only [AlgorithmLib.LZ4Simt.succsOf,
        Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

theorem preStore_succ264 : AlgorithmLib.LZ4Simt.succsOf K16 251 = [252] := by decide
theorem preStore_pin264 (ss : SState) (h0p : ss.pc ∉ preStoreS264) (k : Nat)
    (hpc : (siter K16 k ss).pc = 251) :
    ∃ j, k = j + 3 ∧ (siter K16 j ss).pc = 248 := by
  have hmem : (siter K16 k ss).pc ∈ preStoreS264 := by rw [hpc]; decide
  obtain ⟨j, hjk, hje, hall⟩ := region_entry K16 preStoreS264 248 preStoreS2_entry64 ss h0p k hmem
  obtain ⟨hpcs, -, -, -, -⟩ := copy_pred264 (siter K16 j ss) hje
  refine ⟨j, ?_, hje⟩
  have hshift : ∀ a : Nat, siter K16 (j + a) ss = siter K16 a (siter K16 j ss) :=
    fun a => siter_add K16 j a ss
  rcases Nat.lt_or_ge k (j + 3) with hlt | hge
  · exfalso
    have hle : k - j ≤ 3 := by omega
    have hh := hpcs (k - j) hle
    rw [← hshift (k - j), show j + (k - j) = k from by omega] at hh
    rw [hpc] at hh; omega
  · rcases Nat.lt_or_ge (j + 3) k with hgt | hle2
    · exfalso
      have h251 : (siter K16 (j + 3) ss).pc = 251 := by
        rw [hshift 3]; exact (copy_pred264 (siter K16 j ss) hje).2.1
      have hs : (siter K16 (j + 3 + 1) ss).pc
          ∈ AlgorithmLib.LZ4Simt.succsOf K16 (siter K16 (j + 3) ss).pc := by
        rw [show (siter K16 (j + 3 + 1) ss) = sstep K16 (siter K16 (j + 3) ss) from siter_succ K16 (j+3) ss]
        exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K16 _
      rw [h251, preStore_succ264, List.mem_singleton] at hs
      have hin := hall (j + 3 + 1) (by omega) (by omega)
      rw [hs] at hin
      exact absurd hin (by decide)
    · omega

theorem copy_head2_pin264 (ss : SState) (h0c : ss.pc ∉ copyS264) (k : Nat)
    (hpc : (siter K16 k ss).pc = 251) :
    ∃ j, j + 4 ≤ k ∧ (siter K16 j ss).pc = 238
      ∧ ∀ i, j ≤ i → i ≤ k → (siter K16 i ss).pc ∈ copyS264 := by
  have hmem : (siter K16 k ss).pc ∈ copyS264 := by rw [hpc]; decide
  obtain ⟨j, hjk, hje, hall⟩ := region_entry K16 copyS264 238 copyS2_entry64 ss h0c k hmem
  obtain ⟨hpcs, -, -, -⟩ := copy_head264 (siter K16 j ss) hje
  refine ⟨j, ?_, hje, hall⟩
  rcases Nat.lt_or_ge k (j + 4) with hlt | hge
  · exfalso
    have hle : k - j ≤ 4 := by omega
    have hh := hpcs (k - j) hle
    rw [← siter_add K16 j (k - j) ss, show j + (k - j) = k from by omega] at hh
    rw [hpc] at hh; omega
  · exact hge

theorem copyLoop_exit_succ264 : AlgorithmLib.LZ4Simt.succsOf K16 255 = [256] := by decide
theorem copy_const264 (ss : SState) (j k : Nat) (hjk : j ≤ k)
    (h242 : (siter K16 j ss).pc = 242)
    (hall : ∀ i, j ≤ i → i ≤ k → (siter K16 i ss).pc ∈ copyS264)
    (r : String) (hr : r ∈ ["op", "outBase", "fLen", "cpDstF", "lane"]) :
    (siter K16 k ss).regs r = (siter K16 j ss).regs r := by
  have hne : ∀ i, i < k - j → (siter K16 i (siter K16 j ss)).pc ∉ [255] := by
    intro i hi hin
    rw [← siter_add K16 j i ss] at hin
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hin
    have hs : (siter K16 (j + i + 1) ss).pc
        ∈ AlgorithmLib.LZ4Simt.succsOf K16 (siter K16 (j + i) ss).pc := by
      rw [show siter K16 (j + i + 1) ss = sstep K16 (siter K16 (j + i) ss) from siter_succ K16 (j + i) ss]
      exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K16 _
    rw [hin, copyLoop_exit_succ264, List.mem_singleton] at hs
    have hmem := hall (j + i + 1) (by omega) (by omega)
    rw [hs] at hmem
    exact absurd hmem (by decide)
  have hconst := AlgorithmLib.LZ4Simt.regs_const_on K16 r ((List.range 14).map (· + 242)) [255]
    copyLoop_closed264 (copy_no_dest264 r hr) (siter K16 j ss) (by rw [h242]; decide) (k - j) hne
  rw [← siter_add K16 j (k - j) ss, show j + (k - j) = k from by omega] at hconst
  exact hconst

theorem cpDo_confined264 (LO : Nat) (hLO : LO < 2 ^ 64) (base : Nat) (E : SState)
    (h209 : E.pc = 209)
    (hlaN : (E.regs "litAnchor" 0).toNat ≤ 65536)
    (hlau : ∀ j : Lane, E.regs "litAnchor" j = E.regs "litAnchor" 0)
    (hopu : ∀ j : Lane, E.regs "op" j = E.regs "op" 0)
    (hbud : (E.regs "op" 0).toNat + (65536 - (E.regs "litAnchor" 0).toNat)
        + (65536 - (E.regs "litAnchor" 0).toNat) / 255 + 2 ≤ LO)
    (l : Lane) (k : Nat)
    (hpc : (siter K16 k E).pc = 251)
    (hact : (siter K16 k E).regs "cpP" l = 1)
    (hcpdo : (siter K16 k E).regs "cpDo" l
      = ((siter K16 k E).regs "cpDstF" l + (siter K16 k E).regs "cpI" l)
        + (siter K16 k E).regs "lane" l)
    (hob : (siter K16 k E).regs "outBase" l = UInt64.ofNat base)
    (htop : base + LO + 4 < 2 ^ 64) :
    base ≤ ((siter K16 k E).regs "cpDo" l).toNat
      ∧ ((siter K16 k E).regs "cpDo" l).toNat < base + LO + 4 := by
  -- the predicate half
  obtain ⟨j2, hk3, h248⟩ := preStore_pin264 E (by rw [h209]; decide) k hpc
  have hE2 : siter K16 3 (siter K16 j2 E) = siter K16 k E := by
    rw [← siter_add K16 j2 3 E, ← hk3]
  obtain ⟨-, -, hpred, hfr2, hJ⟩ := copy_pred264 (siter K16 j2 E) h248
  rw [hE2] at hpred hfr2 hJ
  have hlt : (siter K16 j2 E).regs "cpI" l + (siter K16 j2 E).regs "lane" l
      < (siter K16 j2 E).regs "fLen" l := hpred l hact
  -- the setup half
  obtain ⟨j1, h4k, h238, hall⟩ := copy_head2_pin264 E (by rw [h209]; decide) k hpc
  obtain ⟨-, h242, hDst, hFr⟩ := copy_head264 (siter K16 j1 E) h238
  have hE1 : siter K16 4 (siter K16 j1 E) = siter K16 (j1 + 4) E := (siter_add K16 j1 4 E).symm
  rw [hE1] at h242 hDst hFr
  have hconst : ∀ r : String, r ∈ ["op", "outBase", "fLen", "cpDstF", "lane"] →
      (siter K16 k E).regs r = (siter K16 (j1 + 4) E).regs r :=
    fun r hr => copy_const264 E (j1 + 4) k h4k h242 (fun i h1 h2 => hall i (by omega) h2) r hr
  -- the budget, at the copy's setup
  have hbudk := tail_copy_budget64 LO hLO E h209 hlaN hlau hopu hbud l j1 h238
  -- move everything to the visit
  have hopk : (siter K16 k E).regs "op" l = (siter K16 j1 E).regs "op" l := by
    rw [congrFun (hconst "op" (by decide)) l, congrFun (hFr "op" (Or.inl rfl)) l]
  have hflk : (siter K16 k E).regs "fLen" l = (siter K16 j1 E).regs "fLen" l := by
    rw [congrFun (hconst "fLen" (by decide)) l, congrFun (hFr "fLen" (Or.inr (Or.inr rfl))) l]
  have hobk : (siter K16 k E).regs "outBase" l = (siter K16 j1 E).regs "outBase" l := by
    rw [congrFun (hconst "outBase" (by decide)) l, congrFun (hFr "outBase" (Or.inr (Or.inl rfl))) l]
  have hdstk : (siter K16 k E).regs "cpDstF" l
      = (siter K16 j1 E).regs "outBase" l + (siter K16 j1 E).regs "op" l := by
    rw [congrFun (hconst "cpDstF" (by decide)) l, hDst l]
  have e1 : (siter K16 k E).regs "cpI" l = (siter K16 j2 E).regs "cpI" l :=
    congrFun (hfr2 "cpI" (Or.inl rfl)) l
  have e2 : (siter K16 k E).regs "lane" l = (siter K16 j2 E).regs "lane" l :=
    congrFun (hfr2 "lane" (Or.inr (Or.inl rfl))) l
  have e3 : (siter K16 k E).regs "fLen" l = (siter K16 j2 E).regs "fLen" l :=
    congrFun (hfr2 "fLen" (Or.inr (Or.inr (Or.inl rfl)))) l
  have hcjU : (siter K16 k E).regs "cpI" l + (siter K16 k E).regs "lane" l
      < (siter K16 k E).regs "fLen" l := by rw [e1, e2, e3]; exact hlt
  have hcjlt : ((siter K16 k E).regs "cpI" l + (siter K16 k E).regs "lane" l).toNat
      < ((siter K16 j1 E).regs "fLen" l).toNat := by
    rw [← hflk]; exact UInt64.lt_iff_toNat_lt.mp hcjU
  have hfinal : (siter K16 k E).regs "cpDo" l
      = ((siter K16 j1 E).regs "outBase" l + (siter K16 j1 E).regs "op" l)
        + ((siter K16 k E).regs "cpI" l + (siter K16 k E).regs "lane" l) := by
    rw [hcpdo, hdstk, UInt64.add_assoc]
  rw [hfinal]
  refine cpDo_region64 base LO ((siter K16 j1 E).regs "outBase" l) ((siter K16 j1 E).regs "op" l)
    ((siter K16 k E).regs "cpI" l + (siter K16 k E).regs "lane" l) (by rw [← hobk]; exact hob)
    (by have hprod : (WP.mk 16).numBlk * (WP.mk 16).inStride = 209715200 := (by decide); omega) htop

set_option maxHeartbeats 1000000 in
theorem stores_cpDo25164 (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 16).numBlk → inPtr + w * (WP.mk 16).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 16).numBlk →
      inPtr + w * (WP.mk 16).inStride + (WP.mk 16).inStride
        ≤ outPtr + w * (WP.mk 16).outStride)
    (w : Fin (WP.mk 16).numBlk) (k : Nat) (l : Lane)
    (hpc : (siter K16 k (initSt w.val inPtr outPtr gm smemB)).pc = 251)
    (hact : (siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs "cpP" l = 1) :
    Lz4Interleave.outRegion 16 outPtr w.val
      (((siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs "cpDo" l).toNat) := by
  obtain ⟨n, hpc39, hpc40, hck, hpc209, wE, hcE, hQE, hTE, hlaE⟩ :=
    shipped_loop_ckpt64 w.val inPtr outPtr gm smemB w.isLt
      (by have := w.isLt; have := numBlk3264; omega)
      (hib40 _ w.isLt) (htop32 _ w.isLt) (hbuf _ w.isLt) hderive (hdisj _ w.isLt)
  have hkn : preSteps + 1 + n ≤ k := by
    rcases Nat.lt_or_ge k (preSteps + 1 + n) with hlt | hge
    · exfalso
      have hst := stays_from_21064 (initSt w.val inPtr outPtr gm smemB) k (by rw [hpc]; omega)
        (preSteps + 1 + n) (by omega)
      rw [hpc209] at hst; omega
    · exact hge
  have hsplit : siter K16 (k - (preSteps + 1 + n))
      (siter K16 (preSteps + 1 + n) (initSt w.val inPtr outPtr gm smemB))
      = siter K16 k (initSt w.val inPtr outPtr gm smemB) := by
    rw [← siter_add K16 (preSteps + 1 + n) (k - (preSteps + 1 + n)),
      show preSteps + 1 + n + (k - (preSteps + 1 + n)) = k from by omega]
  have hlaC : ∀ j : Lane,
      (siter K16 (preSteps + 1 + n) (initSt w.val inPtr outPtr gm smemB)).regs "litAnchor" j
        = wE.regs "litAnchor" := fun j => hcE.reg (by decide) j
  have hopC : ∀ j : Lane,
      (siter K16 (preSteps + 1 + n) (initSt w.val inPtr outPtr gm smemB)).regs "op" j
        = wE.regs "op" := fun j => hcE.reg (by decide) j
  simp only [AlgorithmLib.LZ4WarpDSL.TightQ, AlgorithmLib.LZ4WarpDSL.tightRem] at hTE
  have hfin := cpDo_confined264 (WP.mk 16).lenOff (by decide)
    (outPtr + w.val * (WP.mk 16).outStride)
    (siter K16 (preSteps + 1 + n) (initSt w.val inPtr outPtr gm smemB)) hpc209
    (by rw [hlaC 0]; exact Nat.le_trans hlaE (by decide))
    (fun j => by rw [hlaC j, hlaC 0])
    (fun j => by rw [hopC j, hopC 0])
    (by rw [hlaC 0, hopC 0]
        rw [show (65536 : Nat) = (WP.mk 16).inStride from rfl]
        omega)
    l (k - (preSteps + 1 + n)) (by rw [hsplit]; exact hpc)
    (by rw [hsplit]; exact hact)
    (by rw [hsplit]; exact (cpDo_at_store64 w.val inPtr outPtr gm smemB k l).2 hpc)
    (by rw [hsplit]
        exact outBase_at_store_site64 w.val inPtr outPtr gm smemB w.isLt
          (by have := w.isLt; have := numBlk3264; omega) hderive k l
          (by rw [hpc]; omega) (by rw [hpc]; omega))
    (by have h := htop32 w.val w.isLt
        have hl : (WP.mk 16).lenOff = 69888 := rfl
        have hi : (WP.mk 16).inStride = 65536 := rfl
        omega)
  rw [hsplit] at hfin
  exact hfin

-- ── The load half: the kernel clamps its own search addresses ────────────────

theorem win_shapeG64 (p : Array SInstr) (S : Nat) (h : winShapeB p S = true) :
    (∀ q ∈ (List.range 51).map (· + 42), (p[q]?.map fallthroughOnlyB) = some true)
    ∧ (∀ q ∈ (List.range 47).map (· + 45),
        (p[q]?.map (fun i => destOf i != some "cap4")) = some true)
    ∧ (∀ q ∈ (List.range 46).map (· + 46),
        (p[q]?.map (fun i => destOf i != some "rp")) = some true)
    ∧ (∀ q ∈ (List.range 27).map (· + 65),
        (p[q]?.map (fun i => destOf i != some "rc")) = some true)
    ∧ p[44]? = some (SInstr.mov "cap4" (.imm (S - 4)))
    ∧ p[45]? = some (SInstr.binr .min "rp" "posP" "cap4")
    ∧ p[64]? = some (SInstr.binr .min "rc" "cand" "cap4") := by
  simp only [winShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩ := h
  exact ⟨h1, h2, h3, h4, h5, h6, h7⟩

theorem win_ftG64 (p : Array SInstr) (S : Nat) (h : winShapeB p S = true) :
    ∀ q, 42 ≤ q → q ≤ 92 → (p[q]?.map fallthroughOnlyB) = some true := by
  intro q h1 h2; exact forall_window 42 51 (win_shapeG64 p S h).1 q h1 (by omega)

theorem cap4_in_windowG64 (p : Array SInstr) (S : Nat) (h : winShapeB p S = true)
    (ss : SState) (h0 : ss.pc < 45) :
    ∀ k : Nat, 45 ≤ (siter p k ss).pc → (siter p k ss).pc ≤ 92 →
      ∀ l : Lane, (siter p k ss).regs "cap4" l = UInt64.ofNat (S - 4) :=
  const_in_region p "cap4" 45 92 (S - 4) (win_shapeG64 p S h).2.2.2.2.1
    (fun q h1 h2 => win_ftG64 p S h q (by omega) h2)
    (fun q h1 h2 => forall_window 45 47 (win_shapeG64 p S h).2.1 q h1 (by omega))
    ss h0

theorem rp_clampedG64 (p : Array SInstr) (S : Nat) (hS : S < 2 ^ 64) (h : winShapeB p S = true)
    (ss : SState) (h0 : ss.pc < 45) :
    ∀ k : Nat, 46 ≤ (siter p k ss).pc → (siter p k ss).pc ≤ 92 →
      ∀ l : Lane, ((siter p k ss).regs "rp" l).toNat ≤ S - 4 :=
  reg_prop_in_region p "rp" 46 92 (fun v => v.toNat ≤ S - 4)
    (fun q h1 h2 => win_ftG64 p S h q (by omega) h2)
    (fun q h1 h2 => forall_window 46 46 (win_shapeG64 p S h).2.2.1 q h1 (by omega))
    ss (by omega)
    (fun k hk l => by
      rw [sstep, show p[(siter p k ss).pc]?
        = some (SInstr.binr .min "rp" "posP" "cap4") from by
          rw [hk]; exact (win_shapeG64 p S h).2.2.2.2.2.1]
      simp only [sstepInstr, SState.setReg, SState.setPc, if_true]
      refine Nat.le_trans (min_run_le_right _ _) ?_
      rw [cap4_in_windowG64 p S h ss h0 k (by omega) (by omega) l]
      exact Nat.le_of_eq (AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by omega)))

theorem rc_clampedG64 (p : Array SInstr) (S : Nat) (hS : S < 2 ^ 64) (h : winShapeB p S = true)
    (ss : SState) (h0 : ss.pc < 45) :
    ∀ k : Nat, 65 ≤ (siter p k ss).pc → (siter p k ss).pc ≤ 92 →
      ∀ l : Lane, ((siter p k ss).regs "rc" l).toNat ≤ S - 4 :=
  reg_prop_in_region p "rc" 65 92 (fun v => v.toNat ≤ S - 4)
    (fun q h1 h2 => win_ftG64 p S h q (by omega) h2)
    (fun q h1 h2 => forall_window 65 27 (win_shapeG64 p S h).2.2.2.1 q h1 (by omega))
    ss (by omega)
    (fun k hk l => by
      rw [sstep, show p[(siter p k ss).pc]?
        = some (SInstr.binr .min "rc" "cand" "cap4") from by
          rw [hk]; exact (win_shapeG64 p S h).2.2.2.2.2.2]
      simp only [sstepInstr, SState.setReg, SState.setPc, if_true]
      refine Nat.le_trans (min_run_le_right _ _) ?_
      rw [cap4_in_windowG64 p S h ss h0 k (by omega) (by omega) l]
      exact Nat.le_of_eq (AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by omega)))

-- ── The shipped 32 KiB instance ─────────────────────────────────────────────


theorem cap4_in_window64 (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, 45 ≤ (siter K16 k (initSt w inPtr outPtr gm smemB)).pc →
      (siter K16 k (initSt w inPtr outPtr gm smemB)).pc ≤ 92 →
      ∀ l : Lane, (siter K16 k (initSt w inPtr outPtr gm smemB)).regs "cap4" l
        = UInt64.ofNat 65532 :=
  cap4_in_windowG64 K16 65536 winShapeB_64 (initSt w inPtr outPtr gm smemB)
    (by show (0:Nat) < 45; decide)

theorem rp_clamped64 (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, 46 ≤ (siter K16 k (initSt w inPtr outPtr gm smemB)).pc →
      (siter K16 k (initSt w inPtr outPtr gm smemB)).pc ≤ 92 →
      ∀ l : Lane, ((siter K16 k (initSt w inPtr outPtr gm smemB)).regs "rp" l).toNat ≤ 65532 :=
  rp_clampedG64 K16 65536 (by decide) winShapeB_64 (initSt w inPtr outPtr gm smemB)
    (by show (0:Nat) < 45; decide)

theorem rc_clamped64 (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, 65 ≤ (siter K16 k (initSt w inPtr outPtr gm smemB)).pc →
      (siter K16 k (initSt w inPtr outPtr gm smemB)).pc ≤ 92 →
      ∀ l : Lane, ((siter K16 k (initSt w inPtr outPtr gm smemB)).regs "rc" l).toNat ≤ 65532 :=
  rc_clampedG64 K16 65536 (by decide) winShapeB_64 (initSt w inPtr outPtr gm smemB)
    (by show (0:Nat) < 45; decide)

theorem win_ft64 : ∀ q, 42 ≤ q → q ≤ 92 → (K16[q]?.map fallthroughOnlyB) = some true :=
  win_ftG64 K16 65536 winShapeB_64

-- ── …and the 64 KiB one, which is now one `decide` ──────────────────────────


end Lz4Sites