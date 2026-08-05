import Lz4OpLe

set_option maxRecDepth 8192

/-!
  # `RegConfined`'s store half

  Fourteen of the sixteen store sites: the ten `sbAddr` stores of the token emit
  and final literal run, which `cursorAtSites_shipped` bounds, and the four
  length-field stores, which sit at fixed offsets `lenOff … lenOff+3`.

  The two remaining sites are the cooperative copy's `cpDo` (pcs 165 and 251).
  They are predicated, so `ActiveAt` restricts them to the lanes that actually
  write, and bounding those needs the copy loop's counter — see the note at the
  end of this file.
-/

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt

/-- The four length stores are at `outBase + lenOff + i`, and the region runs to
    `outBase + lenOff + 4`. -/
theorem stores_la (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (w : Fin (WP.mk 15).numBlk) (k : Nat) (l : Lane) (r : String)
    (hr : r = "la0" ∨ r = "la1" ∨ r = "la2" ∨ r = "la3")
    (hpc : ((siter K k (initSt w.val inPtr outPtr gm smemB)).pc, r) ∈ storeSites K)
    (htop : outPtr + w.val * (WP.mk 15).outStride + (WP.mk 15).lenOff + 4 < 2 ^ 64) :
    Lz4Interleave.outRegion 15 outPtr w.val
      (((siter K k (initSt w.val inPtr outPtr gm smemB)).regs r l).toNat) := by
  -- the pc of an `la` store site, read off the emitted array
  have hsites : ∀ (q : Nat) (r' : String), (q, r') ∈ storeSites K →
      (r' = "la0" → q = 259) ∧ (r' = "la1" → q = 263) ∧ (r' = "la2" → q = 267)
        ∧ (r' = "la3" → q = 271) := by
    have hall : ∀ (x : Nat × String), x ∈ storeSites K →
        (x.2 = "la0" → x.1 = 259) ∧ (x.2 = "la1" → x.1 = 263) ∧ (x.2 = "la2" → x.1 = 267)
          ∧ (x.2 = "la3" → x.1 = 271) := by
      rw [shipped32_store_sites]; decide
    intro q r' hq
    exact hall (q, r') hq
  have hla := la_at_store w.val inPtr outPtr gm smemB k l
  have hob : (siter K k (initSt w.val inPtr outPtr gm smemB)).regs "outBase" l
      = UInt64.ofNat (outPtr + w.val * (WP.mk 15).outStride) := by
    refine outBase_at_store_site w.val inPtr outPtr gm smemB w.isLt
      (by have h := w.isLt; have hn := numBlk32; omega) hderive k l ?_ ?_ <;>
      · obtain ⟨e0, e1, e2, e3⟩ := hsites _ r hpc
        rcases hr with rfl | rfl | rfl | rfl
        · rw [e0 rfl]; omega
        · rw [e1 rfl]; omega
        · rw [e2 rfl]; omega
        · rw [e3 rfl]; omega
  have hlen : (WP.mk 15).lenOff = 35072 := rfl
  have hstride : (WP.mk 15).outStride = 35080 := rfl
  obtain ⟨e0, e1, e2, e3⟩ := hsites _ r hpc
  obtain ⟨q0, q1, q2, q3⟩ := hla
  refine ⟨?_, ?_⟩ <;>
    · rcases hr with rfl | rfl | rfl | rfl
      · rw [q0 (e0 rfl), hob, UInt64.toNat_add,
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          Nat.mod_eq_of_lt (by omega)]
        omega
      · rw [q1 (e1 rfl), hob, UInt64.toNat_add,
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          Nat.mod_eq_of_lt (by omega)]
        omega
      · rw [q2 (e2 rfl), hob, UInt64.toNat_add,
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          Nat.mod_eq_of_lt (by omega)]
        omega
      · rw [q3 (e3 rfl), hob, UInt64.toNat_add,
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega),
          Nat.mod_eq_of_lt (by omega)]
        omega

/-- The ten `sbAddr` stores, from `cursorAtSites_shipped`. -/
theorem stores_sbAddr (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 15).numBlk → inPtr + w * (WP.mk 15).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 15).numBlk →
      inPtr + w * (WP.mk 15).inStride + (WP.mk 15).inStride
        ≤ outPtr + w * (WP.mk 15).outStride)
    (w : Fin (WP.mk 15).numBlk) (k : Nat) (l : Lane)
    (hpc : ((siter K k (initSt w.val inPtr outPtr gm smemB)).pc, "sbAddr") ∈ storeSites K) :
    Lz4Interleave.outRegion 15 outPtr w.val
      (((siter K k (initSt w.val inPtr outPtr gm smemB)).regs "sbAddr" l).toNat) := by
  have hmem : (siter K k (initSt w.val inPtr outPtr gm smemB)).pc ∈ sbAddrSites := by
    have hall : ∀ (x : Nat × String), x ∈ storeSites K → x.2 = "sbAddr" → x.1 ∈ sbAddrSites := by
      rw [shipped32_store_sites]; decide
    exact hall _ hpc rfl
  -- step 0 is the launch state, whose pc is 0
  cases k with
  | zero =>
      exfalso
      have : (siter K 0 (initSt w.val inPtr outPtr gm smemB)).pc = 0 := rfl
      rw [this] at hmem
      exact absurd hmem (by decide)
  | succ m =>
      exact sbAddr_confined_of_cursor inPtr outPtr gm smemB
        (cursorAtSites_shipped inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj)
        hderive w m l hmem
        (by have h := htop32 w.val w.isLt
            have hl : (WP.mk 15).lenOff = 35072 := rfl
            have hi : (WP.mk 15).inStride = 32768 := rfl
            omega)

/-- **Fourteen of the sixteen store sites are confined** — every store except the
    cooperative copy's two.

    What is left for `RegConfined.stores` is exactly `cpDo` at pcs 165 and 251.
    Those are `stgp`, so `ActiveAt`restricts them to the lanes that write, and
    for an active lane `cpI + lane < litLen`; with `cpDst = outBase + op` and the
    token budget `op + 1 + |encNib litLen| + litLen + 2 + |encNib mlm| ≤ lenOff`
    (which `MatchEntryQ` carries at the match-sequence entry) the address is in
    region.  The missing link is the copy loop's own invariant relating `cpI`,
    `cpP` and `litLen` — none of `cpI`'s bounds are stated anywhere yet. -/
theorem stores_except_copy (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 15).numBlk → inPtr + w * (WP.mk 15).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 15).numBlk →
      inPtr + w * (WP.mk 15).inStride + (WP.mk 15).inStride
        ≤ outPtr + w * (WP.mk 15).outStride)
    (w : Fin (WP.mk 15).numBlk) (k : Nat) (l : Lane) (r : String)
    (hpc : ((siter K k (initSt w.val inPtr outPtr gm smemB)).pc, r) ∈ storeSites K)
    (hnc : r ≠ "cpDo") :
    Lz4Interleave.outRegion 15 outPtr w.val
      (((siter K k (initSt w.val inPtr outPtr gm smemB)).regs r l).toNat) := by
  have hregs : ∀ (x : Nat × String), x ∈ storeSites K →
      x.2 = "sbAddr" ∨ x.2 = "cpDo" ∨ x.2 = "la0" ∨ x.2 = "la1" ∨ x.2 = "la2" ∨ x.2 = "la3" := by
    rw [shipped32_store_sites]; decide
  have htop64 : outPtr + w.val * (WP.mk 15).outStride + (WP.mk 15).lenOff + 4 < 2 ^ 64 := by
    have h := htop32 w.val w.isLt
    have hl : (WP.mk 15).lenOff = 35072 := rfl
    have hi : (WP.mk 15).inStride = 32768 := rfl
    omega
  rcases hregs _ hpc with e | e | e | e | e | e
  · subst e; exact stores_sbAddr inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj w k l hpc
  · exact absurd e hnc
  · exact stores_la inPtr outPtr gm smemB hderive w k l r (Or.inl e) hpc htop64
  · exact stores_la inPtr outPtr gm smemB hderive w k l r (Or.inr (Or.inl e)) hpc htop64
  · exact stores_la inPtr outPtr gm smemB hderive w k l r (Or.inr (Or.inr (Or.inl e))) hpc htop64
  · exact stores_la inPtr outPtr gm smemB hderive w k l r (Or.inr (Or.inr (Or.inr e))) hpc htop64

-- ── The literal copy loop, pcs 152–168 ───────────────────────────────────────

/-- `152 cpDst := outBase + op; 153 cpSrc := inBase + litAnchor; 154 cpI := 0;
     155 setp cpCont; 156 lbl; 157 braifnot cpCont → 169; 158 cpDo := cpDst + cpI;
     159 cpDo += lane; 160 cpSo := cpSrc + cpI; 161 cpSo += lane;
     162 cpJ := cpI + lane; 163 setp cpP (cpJ < litLen); 164 ldgo; 165 stgp;
     166 cpI += 32; 167 setp cpCont; 168 bra → 156`.

    The store at 165 is predicated on `cpP`, and `cpP` is exactly `cpI + lane <
    litLen` — so an ACTIVE lane's address is below `cpDst + litLen`, and no bound
    on `cpI` itself is needed. -/
def copyS : List Nat := (List.range 18).map (· + 152)

theorem copyS_entry_lt : ∀ q, q < 274 → q ∉ copyS →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K q, q' ∈ copyS → q' = 152 :=
  ivEntry_at K 152 18 152 shipped32_size (by decide)

theorem copyS_entry : ∀ q, q ∉ copyS →
    ∀ q', q' ∈ AlgorithmLib.LZ4Simt.succsOf K q → q' ∈ copyS → q' = 152 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact copyS_entry_lt q h hq q' hq' hin
  · rw [show AlgorithmLib.LZ4Simt.succsOf K q = [q] from by
      simp only [AlgorithmLib.LZ4Simt.succsOf,
        Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

/-- The four registers the bound needs are untouched inside the loop: the cursor,
    the output base, the literal length and the copy destination. -/
theorem copy_no_dest : ∀ r ∈ ["op", "outBase", "litLen", "cpDst", "lane"],
    ∀ q ∈ ((List.range 14).map (· + 156)),
      (K[q]?.map (fun i => destOf i != some r)) = some true := by decide

/-- …and the loop proper, `156–168`, is closed with the same exit. -/
theorem copyLoop_closed : PcClosed K ((List.range 14).map (· + 156)) [169] :=
  ivClosed_at K 156 14 [169] shipped32_size (by omega) (by decide)

/-- **The copy loop's head**: four straight steps from 152 set `cpDst` to the
    cursor and leave the cursor, base and literal length alone. -/
theorem copy_head (E : SState) (h152 : E.pc = 152) :
    (∀ i, i ≤ 4 → (siter K i E).pc = 152 + i)
    ∧ (siter K 4 E).pc = 156
    ∧ (∀ l : Lane, (siter K 4 E).regs "cpDst" l = E.regs "outBase" l + E.regs "op" l)
    ∧ (∀ r : String, r = "op" ∨ r = "outBase" ∨ r = "litLen" →
        (siter K 4 E).regs r = E.regs r) := by
  have e0 : siter K 0 E = E := rfl
  have p1 : (siter K 1 E).pc = 153 := by
    rw [siter_succ, e0]
    exact tail_pc_bin E 152 (.add) "cpDst" "outBase" (SArg.reg "op") h152 (by decide)
  have p2 : (siter K 2 E).pc = 154 := by
    rw [siter_succ]
    exact tail_pc_bin _ 153 (.add) "cpSrc" "inBase" (SArg.reg "litAnchor") p1 (by decide)
  have p3 : (siter K 3 E).pc = 155 := by
    rw [siter_succ]; exact tail_pc_mov _ 154 "cpI" (SArg.imm 0) p2 (by decide)
  have p4 : (siter K 4 E).pc = 156 := by
    rw [siter_succ]
    exact tail_pc_setp _ 155 (.lt) "cpCont" "cpI" (SArg.reg "litLen") p3 (by decide)
  refine ⟨fun i hi => ?_, p4, fun l => ?_, fun r hr => ?_⟩
  · match i, hi with
    | 0, _ => rw [e0]; exact h152
    | 1, _ => exact p1
    | 2, _ => exact p2
    | 3, _ => exact p3
    | 4, _ => exact p4
  · have f1 : (siter K 1 E).regs "cpDst" l = E.regs "outBase" l + E.regs "op" l := by
      rw [siter_succ, e0,
        tail_val_bin E 152 (.add) "cpDst" "outBase" (SArg.reg "op") h152 (by decide) l]
      rfl
    have f2 : (siter K 2 E).regs "cpDst" = (siter K 1 E).regs "cpDst" := by
      rw [siter_succ]
      exact tail_frame _ 153 (.bin (.add) "cpSrc" "inBase" (SArg.reg "litAnchor")) "cpDst" p1
        (by decide) (by decide)
    have f3 : (siter K 3 E).regs "cpDst" = (siter K 2 E).regs "cpDst" := by
      rw [siter_succ]
      exact tail_frame _ 154 (.mov "cpI" (SArg.imm 0)) "cpDst" p2 (by decide) (by decide)
    have f4 : (siter K 4 E).regs "cpDst" = (siter K 3 E).regs "cpDst" := by
      rw [siter_succ]
      exact tail_frame _ 155 (.setp (.lt) "cpCont" "cpI" (SArg.reg "litLen")) "cpDst" p3
        (by decide) (by decide)
    rw [show (siter K 4 E).regs "cpDst" l = (siter K 4 E).regs "cpDst" l from rfl, f4, f3, f2, f1]
  · have hne : r ≠ "cpDst" ∧ r ≠ "cpSrc" ∧ r ≠ "cpI" ∧ r ≠ "cpCont" := by
      rcases hr with rfl | rfl | rfl <;> exact ⟨by decide, by decide, by decide, by decide⟩
    have f1 : (siter K 1 E).regs r = E.regs r := by
      rw [siter_succ, e0]
      exact tail_frame E 152 (.bin (.add) "cpDst" "outBase" (SArg.reg "op")) r h152 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.1)
    have f2 : (siter K 2 E).regs r = (siter K 1 E).regs r := by
      rw [siter_succ]
      exact tail_frame _ 153 (.bin (.add) "cpSrc" "inBase" (SArg.reg "litAnchor")) r p1 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.2.1)
    have f3 : (siter K 3 E).regs r = (siter K 2 E).regs r := by
      rw [siter_succ]
      exact tail_frame _ 154 (.mov "cpI" (SArg.imm 0)) r p2 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.2.2.1)
    have f4 : (siter K 4 E).regs r = (siter K 3 E).regs r := by
      rw [siter_succ]
      exact tail_frame _ 155 (.setp (.lt) "cpCont" "cpI" (SArg.reg "litLen")) r p3 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.2.2.2)
    rw [f4, f3, f2, f1]

/-- The `binr` value helper (register-register), for `cpJ := cpI + lane`. -/
theorem tail_val_binr (st : SState) (q : Nat) (o : SOp) (d a b : String) (hq : st.pc = q)
    (hi : K[q]? = some (.binr o d a b)) (l : Lane) :
    (sstep K st).regs d l = o.run (st.regs a l) (st.regs b l) := by
  rw [sstep, show K[st.pc]? = some (.binr o d a b) from by rw [hq]; exact hi]
  simp only [sstepInstr, SState.setPc, SState.setReg, eq_self_iff_true, if_true]

theorem tail_pc_binr (st : SState) (q : Nat) (o : SOp) (d a b : String) (hq : st.pc = q)
    (hi : K[q]? = some (.binr o d a b)) : (sstep K st).pc = q + 1 := by
  rw [sstep, show K[st.pc]? = some (.binr o d a b) from by rw [hq]; exact hi]
  show st.pc + 1 = q + 1; rw [hq]

/-- **The copy store's predicate is the bound.**  Three steps from 162:
    `cpJ := cpI + lane`, `cpP := cpJ < litLen`, the load — so at the store an
    active lane satisfies `cpI + lane < litLen`, and no bound on `cpI` is needed. -/
theorem copy_pred (E : SState) (h162 : E.pc = 162) :
    (∀ i, i ≤ 3 → (siter K i E).pc = 162 + i)
    ∧ (siter K 3 E).pc = 165
    ∧ (∀ l : Lane, (siter K 3 E).regs "cpP" l = 1 →
        E.regs "cpI" l + E.regs "lane" l < E.regs "litLen" l)
    ∧ (∀ r : String, r = "cpI" ∨ r = "lane" ∨ r = "litLen" ∨ r = "cpDst" ∨ r = "cpDo" →
        (siter K 3 E).regs r = E.regs r)
    ∧ (∀ l : Lane, (siter K 3 E).regs "cpJ" l = E.regs "cpI" l + E.regs "lane" l) := by
  have e0 : siter K 0 E = E := rfl
  have p1 : (siter K 1 E).pc = 163 := by
    rw [siter_succ, e0]; exact tail_pc_binr E 162 (.add) "cpJ" "cpI" "lane" h162 (by decide)
  have p2 : (siter K 2 E).pc = 164 := by
    rw [siter_succ]
    exact tail_pc_setp _ 163 (.lt) "cpP" "cpJ" (SArg.reg "litLen") p1 (by decide)
  have p3 : (siter K 3 E).pc = 165 := by
    rw [siter_succ]
    rw [sstep, show K[(siter K 2 E).pc]? = some (.ldgo "cpB" "cpSo" 0) from by rw [p2]; decide]
    show (siter K 2 E).pc + 1 = 165
    rw [p2]
  have hfr : ∀ r : String, r ≠ "cpJ" → r ≠ "cpP" → r ≠ "cpB" →
      (siter K 3 E).regs r = E.regs r := by
    intro r h1 h2 h3
    have f1 : (siter K 1 E).regs r = E.regs r := by
      rw [siter_succ, e0]
      exact tail_frame E 162 (.binr (.add) "cpJ" "cpI" "lane") r h162 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h1)
    have f2 : (siter K 2 E).regs r = (siter K 1 E).regs r := by
      rw [siter_succ]
      exact tail_frame _ 163 (.setp (.lt) "cpP" "cpJ" (SArg.reg "litLen")) r p1 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h2)
    have f3 : (siter K 3 E).regs r = (siter K 2 E).regs r := by
      rw [siter_succ]
      exact tail_frame _ 164 (.ldgo "cpB" "cpSo" 0) r p2 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h3)
    rw [f3, f2, f1]
  refine ⟨fun i hi => ?_, p3, fun l hp => ?_, fun r hr => ?_, fun l => ?_⟩
  · match i, hi with
    | 0, _ => rw [e0]; exact h162
    | 1, _ => exact p1
    | 2, _ => exact p2
    | 3, _ => exact p3
  · -- `cpP` survives the load, and at 163 it is the comparison
    have fP : (siter K 3 E).regs "cpP" l = (siter K 2 E).regs "cpP" l := by
      rw [siter_succ]
      exact congrFun (tail_frame _ 164 (.ldgo "cpB" "cpSo" 0) "cpP" p2 (by decide) (by decide)) l
    have vP : (siter K 2 E).regs "cpP" l
        = (if SCmp.run (.lt) ((siter K 1 E).regs "cpJ" l)
            ((siter K 1 E).get l (SArg.reg "litLen")) then 1 else 0) := by
      rw [siter_succ]
      exact tail_val_setp _ 163 (.lt) "cpP" "cpJ" (SArg.reg "litLen") p1 (by decide) l
    have vJ : (siter K 1 E).regs "cpJ" l = E.regs "cpI" l + E.regs "lane" l := by
      rw [siter_succ, e0, tail_val_binr E 162 (.add) "cpJ" "cpI" "lane" h162 (by decide) l]
      rfl
    have vL : (siter K 1 E).get l (SArg.reg "litLen") = E.regs "litLen" l := by
      rw [siter_succ, e0]
      show (sstep K E).regs "litLen" l = _
      exact congrFun (tail_frame E 162 (.binr (.add) "cpJ" "cpI" "lane") "litLen" h162
        (by decide) (by decide)) l
    rw [fP, vP, vJ, vL] at hp
    by_cases hlt : SCmp.run (.lt) (E.regs "cpI" l + E.regs "lane" l) (E.regs "litLen" l)
    · exact of_decide_eq_true (by simpa only [SCmp.run] using hlt)
    · rw [if_neg hlt] at hp; exact absurd hp (by decide)
  · rcases hr with rfl | rfl | rfl | rfl | rfl <;>
      exact hfr _ (by decide) (by decide) (by decide)
  · have fJ : (siter K 3 E).regs "cpJ" l = (siter K 1 E).regs "cpJ" l := by
      have a2 : (siter K 2 E).regs "cpJ" = (siter K 1 E).regs "cpJ" := by
        rw [siter_succ]
        exact tail_frame _ 163 (.setp (.lt) "cpP" "cpJ" (SArg.reg "litLen")) "cpJ" p1
          (by decide) (by decide)
      have a3 : (siter K 3 E).regs "cpJ" = (siter K 2 E).regs "cpJ" := by
        rw [siter_succ]
        exact tail_frame _ 164 (.ldgo "cpB" "cpSo" 0) "cpJ" p2 (by decide) (by decide)
      rw [show (siter K 3 E).regs "cpJ" l = (siter K 3 E).regs "cpJ" l from rfl, a3, a2]
    rw [fJ, siter_succ, e0, tail_val_binr E 162 (.add) "cpJ" "cpI" "lane" h162 (by decide) l]
    rfl

/-- The straight stretch that ends at the copy store: `162 … 165`.  Its only
    in-edge is `161 → 162`, so a visit to the store descends from a visit to 162
    and `copy_pred` applies there. -/
def preStoreS : List Nat := (List.range 4).map (· + 162)

theorem preStoreS_entry_lt : ∀ q, q < 274 → q ∉ preStoreS →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K q, q' ∈ preStoreS → q' = 162 :=
  ivEntry_at K 162 4 162 shipped32_size (by decide)

theorem preStoreS_entry : ∀ q, q ∉ preStoreS →
    ∀ q', q' ∈ AlgorithmLib.LZ4Simt.succsOf K q → q' ∈ preStoreS → q' = 162 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact preStoreS_entry_lt q h hq q' hq' hin
  · rw [show AlgorithmLib.LZ4Simt.succsOf K q = [q] from by
      simp only [AlgorithmLib.LZ4Simt.succsOf,
        Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

/-- 165 leaves the stretch, so a visit to the store is exactly three steps past
    its entry. -/
theorem preStore_succ : AlgorithmLib.LZ4Simt.succsOf K 165 = [166] := by decide

theorem tokInv_at (LO : Nat) (hLO : LO < 2 ^ 64) (ss : SState)
    (h0 : ss.pc ∉ tokRegion)
    (l : Lane) (k : Nat)
    (hck : ∀ j, j ≤ k → (siter K j ss).pc = 124 →
      AlgorithmLib.LZ4WarpDSL.MatchEntryQ (WP.mk 15).inStride LO (siter K j ss))
    (hq : (siter K k ss).pc ∈ tokS) :
    TokInv l LO (siter K k ss) := by
  have hmem : (siter K k ss).pc ∈ tokRegion := by
    have hsub : ∀ q ∈ tokS, q ∈ tokRegion := by decide
    exact hsub _ hq
  obtain ⟨j, hjk, hje, hall⟩ := region_entry K tokRegion 124 tokRegion_entry ss h0 k hmem
  have hp1 := me_pc1 (siter K j ss) hje
  have hp2 := me_pc2 (siter K j ss) hje
  have hp3 := me_pc3 (siter K j ss) hje
  have hp4 := me_pc4 (siter K j ss) hje
  have hp5 := me_pc5 (siter K j ss) hje
  have hshift : ∀ a : Nat, siter K (j + a) ss = siter K a (siter K j ss) := by
    intro a; exact siter_add K j a ss
  have hstep : ∀ a : Nat, a ≤ 5 → (siter K (j + a) ss).pc = 124 + a := by
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
      have h129 : 129 ≤ (siter K k ss).pc := by
        have hlo : ∀ q ∈ tokS, 129 ≤ q := by decide
        exact hlo _ hq
      omega
    · exact hge
  have hTok : TokInv l LO (siter K (j + 5) ss) := by
    rw [hshift 5]
    exact tokInv_at_entry l LO (siter K j ss) hje (hck j hjk hje)
  have hpc129 : (siter K (j + 5) ss).pc = 129 := by rw [hshift 5]; exact hp5 ▸ rfl
  -- no exit of the token region is passed before the visit
  have hne : ∀ i, i < k - (j + 5) → (siter K i (siter K (j + 5) ss)).pc ∉ [197, 198] := by
    intro i hi hin
    rw [← siter_add K (j + 5) i ss, show j + 5 + i = j + (5 + i) from by omega] at hin
    have hnext : (siter K (j + (5 + i) + 1) ss).pc ∈ tokRegion :=
      hall (j + (5 + i) + 1) (by omega) (by omega)
    have hs : (siter K (j + (5 + i) + 1) ss).pc
        ∈ AlgorithmLib.LZ4Simt.succsOf K (siter K (j + (5 + i)) ss).pc := by
      rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K _
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hin
    rcases hin with e | e
    · rw [e, tok_exits.1, List.mem_singleton] at hs; rw [hs] at hnext; exact absurd hnext (by decide)
    · rw [e, tok_exits.2, List.mem_singleton] at hs; rw [hs] at hnext; exact absurd hnext (by decide)
  have hinv := inv_on K (TokInv l LO) tokS [197, 198] tokS_closed
    (fun s hsm hexs hh => tokS_hstep l LO hLO s hsm hexs hh) (siter K (j + 5) ss)
    (by rw [hpc129]; decide) hTok (k - (j + 5)) hne
  rw [← siter_add K (j + 5) (k - (j + 5)) ss, show j + 5 + (k - (j + 5)) = k from by omega] at hinv
  exact hinv


/-- **The copy store's address, in `Nat`.**  `cpDo = (cpDst + cpJ)` with
    `cpDst = outBase + op`, so the address is `outBase + op + cpJ` provided the sum
    does not wrap — which the geometry gives, since `op + cpJ < op + litLen ≤ lenOff`
    and `outBase + lenOff + 4 < 2 ^ 64`.

    Stated separately because it is the only part of the copy-store bound that is
    not composition of trace lemmas. -/
theorem cpDo_toNat (ob opv cj : UInt64) (base : Nat) (hob : ob = UInt64.ofNat base)
    (hlt : base + opv.toNat + cj.toNat < 2 ^ 64) :
    ((ob + opv) + cj).toNat = base + opv.toNat + cj.toNat := by
  have hb : ob.toNat = base := by
    rw [hob, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by omega)]
  have h1 : (ob + opv).toNat = base + opv.toNat := by
    rw [UInt64.toNat_add, hb, Nat.mod_eq_of_lt (by omega)]
  rw [UInt64.toNat_add, h1, Nat.mod_eq_of_lt (by omega)]

/-- …and the region membership that follows. -/
theorem cpDo_region (base LO : Nat) (ob opv cj : UInt64) (hob : ob = UInt64.ofNat base)
    (hbud : opv.toNat + cj.toNat < LO) (htop : base + LO + 4 < 2 ^ 64) :
    base ≤ ((ob + opv) + cj).toNat ∧ ((ob + opv) + cj).toNat < base + LO + 4 := by
  rw [cpDo_toNat ob opv cj base hob (by omega)]
  omega

/-- **Step 1 of the copy-store assembly**: a visit to the store at 165 is exactly
    three steps past a visit to 162.  Earlier indices sit at 162–164, and a later
    one would step to 166, which is outside the stretch. -/
theorem preStore_pin (ss : SState) (h0p : ss.pc ∉ preStoreS) (k : Nat)
    (hpc : (siter K k ss).pc = 165) :
    ∃ j, k = j + 3 ∧ (siter K j ss).pc = 162 := by
  have hmem : (siter K k ss).pc ∈ preStoreS := by rw [hpc]; decide
  obtain ⟨j, hjk, hje, hall⟩ := region_entry K preStoreS 162 preStoreS_entry ss h0p k hmem
  obtain ⟨hpcs, -, -, -, -⟩ := copy_pred (siter K j ss) hje
  refine ⟨j, ?_, hje⟩
  have hshift : ∀ a : Nat, siter K (j + a) ss = siter K a (siter K j ss) :=
    fun a => siter_add K j a ss
  rcases Nat.lt_or_ge k (j + 3) with hlt | hge
  · exfalso
    have hle : k - j ≤ 3 := by omega
    have hh := hpcs (k - j) hle
    rw [← hshift (k - j), show j + (k - j) = k from by omega] at hh
    rw [hpc] at hh; omega
  · rcases Nat.lt_or_ge (j + 3) k with hgt | hle2
    · exfalso
      have h165 : (siter K (j + 3) ss).pc = 165 := by
        rw [hshift 3]; exact (copy_pred (siter K j ss) hje).2.1
      have hs : (siter K (j + 3 + 1) ss).pc
          ∈ AlgorithmLib.LZ4Simt.succsOf K (siter K (j + 3) ss).pc := by
        rw [show (siter K (j + 3 + 1) ss) = sstep K (siter K (j + 3) ss) from siter_succ K (j+3) ss]
        exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K _
      rw [h165, preStore_succ, List.mem_singleton] at hs
      have hin := hall (j + 3 + 1) (by omega) (by omega)
      rw [hs] at hin
      exact absurd hin (by decide)
    · omega

/-- **Step 2 of the copy-store assembly**: a visit to the store descends from the
    loop's setup at 152, at least four steps earlier, and the whole stretch stays
    inside the copy region. -/
theorem copy_head_pin (ss : SState) (h0c : ss.pc ∉ copyS) (k : Nat)
    (hpc : (siter K k ss).pc = 165) :
    ∃ j, j + 4 ≤ k ∧ (siter K j ss).pc = 152
      ∧ ∀ i, j ≤ i → i ≤ k → (siter K i ss).pc ∈ copyS := by
  have hmem : (siter K k ss).pc ∈ copyS := by rw [hpc]; decide
  obtain ⟨j, hjk, hje, hall⟩ := region_entry K copyS 152 copyS_entry ss h0c k hmem
  obtain ⟨hpcs, -, -, -⟩ := copy_head (siter K j ss) hje
  refine ⟨j, ?_, hje, hall⟩
  rcases Nat.lt_or_ge k (j + 4) with hlt | hge
  · exfalso
    have hle : k - j ≤ 4 := by omega
    have hh := hpcs (k - j) hle
    rw [← siter_add K j (k - j) ss, show j + (k - j) = k from by omega] at hh
    rw [hpc] at hh; omega
  · exact hge

theorem copyLoop_exit_succ : AlgorithmLib.LZ4Simt.succsOf K 169 = [170] := by decide

/-- **Step 3**: the four registers the bound needs are carried unchanged from the
    loop head to the visit.  The loop's exit at 169 cannot have been passed —
    it steps to 170, outside `copyS`, which the confinement forbids. -/
theorem copy_const (ss : SState) (j k : Nat) (hjk : j ≤ k)
    (h156 : (siter K j ss).pc = 156)
    (hall : ∀ i, j ≤ i → i ≤ k → (siter K i ss).pc ∈ copyS)
    (r : String) (hr : r ∈ ["op", "outBase", "litLen", "cpDst", "lane"]) :
    (siter K k ss).regs r = (siter K j ss).regs r := by
  have hne : ∀ i, i < k - j → (siter K i (siter K j ss)).pc ∉ [169] := by
    intro i hi hin
    rw [← siter_add K j i ss] at hin
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hin
    have hs : (siter K (j + i + 1) ss).pc
        ∈ AlgorithmLib.LZ4Simt.succsOf K (siter K (j + i) ss).pc := by
      rw [show siter K (j + i + 1) ss = sstep K (siter K (j + i) ss) from siter_succ K (j + i) ss]
      exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K _
    rw [hin, copyLoop_exit_succ, List.mem_singleton] at hs
    have hmem := hall (j + i + 1) (by omega) (by omega)
    rw [hs] at hmem
    exact absurd hmem (by decide)
  have hconst := AlgorithmLib.LZ4Simt.regs_const_on K r ((List.range 14).map (· + 156)) [169]
    copyLoop_closed (copy_no_dest r hr) (siter K j ss) (by rw [h156]; decide) (k - j) hne
  rw [← siter_add K j (k - j) ss, show j + (k - j) = k from by omega] at hconst
  exact hconst

/-- **Step 4**: the budget the copy needs, read out of `TokInv` at the loop setup.
    `tokRem q = litLen + 2 + lsicLen mlm` for `149 ≤ q ≤ 170`, so the invariant at
    pc 152 already says the cursor plus the whole literal run fits. -/
theorem copy_budget (l : Lane) (LO : Nat) (st : SState) (h : TokInv l LO st)
    (hpc : st.pc = 152) :
    (st.regs "op" l).toNat + (st.regs "litLen" l).toNat + 2 ≤ LO := by
  have h1 := h.1
  rw [hpc] at h1
  have hr : tokRem 152 ((st.regs "litLen" l).toNat) ((st.regs "litExtra" l).toNat)
      ((st.regs "mlm" l).toNat) ((st.regs "matExtra" l).toNat)
      = (st.regs "litLen" l).toNat + 2 + lsicLen ((st.regs "mlm" l).toNat) := rfl
  rw [hr] at h1
  omega

/-- **The cooperative copy's store is confined.**  The composition of the four
    steps: the visit descends from 162 (predicate) and from 152 (setup), the four
    registers are carried across the loop, and `TokInv` at 152 pays for the whole
    literal run.  `cpDo = cpDst + cpJ` with `cpJ < litLen`, so no bound on `cpI`
    is ever needed. -/
theorem cpDo_confined (LO : Nat) (hLO : LO < 2 ^ 64) (base : Nat) (ss : SState)
    (h0t : ss.pc ∉ tokRegion) (h0c : ss.pc ∉ copyS) (h0p : ss.pc ∉ preStoreS)
    (l : Lane) (k : Nat)
    (hck : ∀ j, j ≤ k → (siter K j ss).pc = 124 →
      AlgorithmLib.LZ4WarpDSL.MatchEntryQ (WP.mk 15).inStride LO (siter K j ss))
    (hpc : (siter K k ss).pc = 165)
    (hact : (siter K k ss).regs "cpP" l = 1)
    (hcpdo : (siter K k ss).regs "cpDo" l
      = ((siter K k ss).regs "cpDst" l + (siter K k ss).regs "cpI" l)
        + (siter K k ss).regs "lane" l)
    (hob : (siter K k ss).regs "outBase" l = UInt64.ofNat base)
    (htop : base + LO + 4 < 2 ^ 64) :
    base ≤ ((siter K k ss).regs "cpDo" l).toNat
      ∧ ((siter K k ss).regs "cpDo" l).toNat < base + LO + 4 := by
  -- the predicate half
  obtain ⟨j2, hk3, h162⟩ := preStore_pin ss h0p k hpc
  have hE2 : siter K 3 (siter K j2 ss) = siter K k ss := by
    rw [← siter_add K j2 3 ss, ← hk3]
  obtain ⟨-, -, hpred, hfr2, hJ⟩ := copy_pred (siter K j2 ss) h162
  rw [hE2] at hpred hfr2 hJ
  have hlt : (siter K j2 ss).regs "cpI" l + (siter K j2 ss).regs "lane" l
      < (siter K j2 ss).regs "litLen" l := hpred l hact
  -- the setup half
  obtain ⟨j1, h4k, h152, hall⟩ := copy_head_pin ss h0c k hpc
  obtain ⟨-, h156, hDst, hFr⟩ := copy_head (siter K j1 ss) h152
  have hE1 : siter K 4 (siter K j1 ss) = siter K (j1 + 4) ss := (siter_add K j1 4 ss).symm
  rw [hE1] at h156 hDst hFr
  have hconst : ∀ r : String, r ∈ ["op", "outBase", "litLen", "cpDst", "lane"] →
      (siter K k ss).regs r = (siter K (j1 + 4) ss).regs r :=
    fun r hr => copy_const ss (j1 + 4) k h4k h156 (fun i h1 h2 => hall i (by omega) h2) r hr
  -- the budget
  have htok := tokInv_at LO hLO ss h0t l j1 (fun j hj => hck j (by omega)) (by rw [h152]; decide)
  have hbud := copy_budget l LO (siter K j1 ss) htok h152
  -- move everything to the visit
  have hopk : (siter K k ss).regs "op" l = (siter K j1 ss).regs "op" l := by
    rw [congrFun (hconst "op" (by decide)) l, congrFun (hFr "op" (Or.inl rfl)) l]
  have hllk : (siter K k ss).regs "litLen" l = (siter K j1 ss).regs "litLen" l := by
    rw [congrFun (hconst "litLen" (by decide)) l,
      congrFun (hFr "litLen" (Or.inr (Or.inr rfl))) l]
  have hobk : (siter K k ss).regs "outBase" l = (siter K j1 ss).regs "outBase" l := by
    rw [congrFun (hconst "outBase" (by decide)) l, congrFun (hFr "outBase" (Or.inr (Or.inl rfl))) l]
  have hdstk : (siter K k ss).regs "cpDst" l
      = (siter K j1 ss).regs "outBase" l + (siter K j1 ss).regs "op" l := by
    rw [congrFun (hconst "cpDst" (by decide)) l, hDst l]
  -- the cursor plus this lane's offset is inside the budget
  have e1 : (siter K k ss).regs "cpI" l = (siter K j2 ss).regs "cpI" l :=
    congrFun (hfr2 "cpI" (Or.inl rfl)) l
  have e2 : (siter K k ss).regs "lane" l = (siter K j2 ss).regs "lane" l :=
    congrFun (hfr2 "lane" (Or.inr (Or.inl rfl))) l
  have e3 : (siter K k ss).regs "litLen" l = (siter K j2 ss).regs "litLen" l :=
    congrFun (hfr2 "litLen" (Or.inr (Or.inr (Or.inl rfl)))) l
  have hcjU : (siter K k ss).regs "cpI" l + (siter K k ss).regs "lane" l
      < (siter K k ss).regs "litLen" l := by rw [e1, e2, e3]; exact hlt
  have hcjlt : ((siter K k ss).regs "cpI" l + (siter K k ss).regs "lane" l).toNat
      < ((siter K j1 ss).regs "litLen" l).toNat := by
    rw [← hllk]; exact UInt64.lt_iff_toNat_lt.mp hcjU
  have hfinal : (siter K k ss).regs "cpDo" l
      = ((siter K j1 ss).regs "outBase" l + (siter K j1 ss).regs "op" l)
        + ((siter K k ss).regs "cpI" l + (siter K k ss).regs "lane" l) := by
    rw [hcpdo, hdstk, UInt64.add_assoc]
  rw [hfinal]
  refine cpDo_region base LO ((siter K j1 ss).regs "outBase" l) ((siter K j1 ss).regs "op" l)
    ((siter K k ss).regs "cpI" l + (siter K k ss).regs "lane" l) (by rw [← hobk]; exact hob)
    (by omega) htop

/-- **The checkpoint, on the launch trace, at every index.**  `shipped_loop_ckpt`
    states it relative to the loop head; before that the prologue keeps the pc at
    most 39, so `pc_j = 124` forces `j` past the anchor. -/
theorem matchEntry_anywhere (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 15).numBlk → inPtr + w * (WP.mk 15).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 15).numBlk →
      inPtr + w * (WP.mk 15).inStride + (WP.mk 15).inStride
        ≤ outPtr + w * (WP.mk 15).outStride)
    (w : Fin (WP.mk 15).numBlk) :
    ∀ j, (siter K j (initSt w.val inPtr outPtr gm smemB)).pc = 124 →
      AlgorithmLib.LZ4WarpDSL.MatchEntryQ (WP.mk 15).inStride (WP.mk 15).lenOff
        (siter K j (initSt w.val inPtr outPtr gm smemB)) := by
  obtain ⟨n, hpc39, hpc40, hck, -, -⟩ :=
    shipped_loop_ckpt w.val inPtr outPtr gm smemB w.isLt
      (by have := w.isLt; have := numBlk32; omega)
      (hib40 _ w.isLt) (htop32 _ w.isLt) (hbuf _ w.isLt) hderive (hdisj _ w.isLt)
  intro j hj
  have hmem : (siter K j (initSt w.val inPtr outPtr gm smemB)).pc ∈ bodyRegion := by
    rw [hj]; decide
  obtain ⟨hge, -, -⟩ :=
    body_entry_at (initSt w.val inPtr outPtr gm smemB) preSteps hpc39 j hmem
  have hidx : preSteps + 1 + (j - (preSteps + 1)) = j := by omega
  have hres := hck (j - (preSteps + 1)) (by rw [hidx]; exact hj)
  rw [hidx] at hres
  exact hres

/-- **The body copy's store site, on the launch trace** — the fifteenth of the
    sixteen. -/
theorem stores_cpDo165 (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 15).numBlk → inPtr + w * (WP.mk 15).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 15).numBlk →
      inPtr + w * (WP.mk 15).inStride + (WP.mk 15).inStride
        ≤ outPtr + w * (WP.mk 15).outStride)
    (w : Fin (WP.mk 15).numBlk) (k : Nat) (l : Lane)
    (hpc : (siter K k (initSt w.val inPtr outPtr gm smemB)).pc = 165)
    (hact : (siter K k (initSt w.val inPtr outPtr gm smemB)).regs "cpP" l = 1) :
    Lz4Interleave.outRegion 15 outPtr w.val
      (((siter K k (initSt w.val inPtr outPtr gm smemB)).regs "cpDo" l).toNat) := by
  have h0 : (initSt w.val inPtr outPtr gm smemB).pc = 0 := rfl
  have hlen : (WP.mk 15).lenOff = 35072 := rfl
  have hstr : (WP.mk 15).inStride = 32768 := rfl
  exact cpDo_confined (WP.mk 15).lenOff (by decide) (outPtr + w.val * (WP.mk 15).outStride)
    (initSt w.val inPtr outPtr gm smemB)
    (by rw [h0]; decide) (by rw [h0]; decide) (by rw [h0]; decide) l k
    (fun j _ hj => matchEntry_anywhere inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj w j hj)
    hpc hact
    ((cpDo_at_store w.val inPtr outPtr gm smemB k l).1 hpc)
    (outBase_at_store_site w.val inPtr outPtr gm smemB w.isLt
      (by have := w.isLt; have := numBlk32; omega) hderive k l
      (by rw [hpc]; omega) (by rw [hpc]; omega))
    (by have h := htop32 w.val w.isLt; omega)

-- ── The tail copy, pcs 238–254 (same shape, shifted by 86) ──────────────────
def copyS2 : List Nat := (List.range 18).map (· + 238)

theorem copyS2_entryLt : ∀ q, q < 274 → q ∉ copyS2 →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K q, q' ∈ copyS2 → q' = 238 :=
  ivEntry_at K 238 18 238 shipped32_size (by decide)

theorem copyS2_entry : ∀ q, q ∉ copyS2 →
    ∀ q', q' ∈ AlgorithmLib.LZ4Simt.succsOf K q → q' ∈ copyS2 → q' = 238 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact copyS2_entryLt q h hq q' hq' hin
  · rw [show AlgorithmLib.LZ4Simt.succsOf K q = [q] from by
      simp only [AlgorithmLib.LZ4Simt.succsOf,
        Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq


theorem copy_no_dest2 : ∀ r ∈ ["op", "outBase", "fLen", "cpDstF", "lane"],
    ∀ q ∈ ((List.range 14).map (· + 242)),
      (K[q]?.map (fun i => destOf i != some r)) = some true := by decide

/-- …and the loop proper, `242–254`, is closed with the same exit. -/
theorem copyLoop_closed2 : PcClosed K ((List.range 14).map (· + 242)) [255] :=
  ivClosed_at K 242 14 [255] shipped32_size (by omega) (by decide)


theorem copy_head2 (E : SState) (h238 : E.pc = 238) :
    (∀ i, i ≤ 4 → (siter K i E).pc = 238 + i)
    ∧ (siter K 4 E).pc = 242
    ∧ (∀ l : Lane, (siter K 4 E).regs "cpDstF" l = E.regs "outBase" l + E.regs "op" l)
    ∧ (∀ r : String, r = "op" ∨ r = "outBase" ∨ r = "fLen" →
        (siter K 4 E).regs r = E.regs r) := by
  have e0 : siter K 0 E = E := rfl
  have p1 : (siter K 1 E).pc = 239 := by
    rw [siter_succ, e0]
    exact tail_pc_bin E 238 (.add) "cpDstF" "outBase" (SArg.reg "op") h238 (by decide)
  have p2 : (siter K 2 E).pc = 240 := by
    rw [siter_succ]
    exact tail_pc_bin _ 239 (.add) "cpSrcF" "inBase" (SArg.reg "litAnchor") p1 (by decide)
  have p3 : (siter K 3 E).pc = 241 := by
    rw [siter_succ]; exact tail_pc_mov _ 240 "cpI" (SArg.imm 0) p2 (by decide)
  have p4 : (siter K 4 E).pc = 242 := by
    rw [siter_succ]
    exact tail_pc_setp _ 241 (.lt) "cpCont" "cpI" (SArg.reg "fLen") p3 (by decide)
  refine ⟨fun i hi => ?_, p4, fun l => ?_, fun r hr => ?_⟩
  · match i, hi with
    | 0, _ => rw [e0]; exact h238
    | 1, _ => exact p1
    | 2, _ => exact p2
    | 3, _ => exact p3
    | 4, _ => exact p4
  · have f1 : (siter K 1 E).regs "cpDstF" l = E.regs "outBase" l + E.regs "op" l := by
      rw [siter_succ, e0,
        tail_val_bin E 238 (.add) "cpDstF" "outBase" (SArg.reg "op") h238 (by decide) l]
      rfl
    have f2 : (siter K 2 E).regs "cpDstF" = (siter K 1 E).regs "cpDstF" := by
      rw [siter_succ]
      exact tail_frame _ 239 (.bin (.add) "cpSrcF" "inBase" (SArg.reg "litAnchor")) "cpDstF" p1
        (by decide) (by decide)
    have f3 : (siter K 3 E).regs "cpDstF" = (siter K 2 E).regs "cpDstF" := by
      rw [siter_succ]
      exact tail_frame _ 240 (.mov "cpI" (SArg.imm 0)) "cpDstF" p2 (by decide) (by decide)
    have f4 : (siter K 4 E).regs "cpDstF" = (siter K 3 E).regs "cpDstF" := by
      rw [siter_succ]
      exact tail_frame _ 241 (.setp (.lt) "cpCont" "cpI" (SArg.reg "fLen")) "cpDstF" p3
        (by decide) (by decide)
    rw [show (siter K 4 E).regs "cpDstF" l = (siter K 4 E).regs "cpDstF" l from rfl, f4, f3, f2, f1]
  · have hne : r ≠ "cpDstF" ∧ r ≠ "cpSrcF" ∧ r ≠ "cpI" ∧ r ≠ "cpCont" := by
      rcases hr with rfl | rfl | rfl <;> exact ⟨by decide, by decide, by decide, by decide⟩
    have f1 : (siter K 1 E).regs r = E.regs r := by
      rw [siter_succ, e0]
      exact tail_frame E 238 (.bin (.add) "cpDstF" "outBase" (SArg.reg "op")) r h238 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.1)
    have f2 : (siter K 2 E).regs r = (siter K 1 E).regs r := by
      rw [siter_succ]
      exact tail_frame _ 239 (.bin (.add) "cpSrcF" "inBase" (SArg.reg "litAnchor")) r p1 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.2.1)
    have f3 : (siter K 3 E).regs r = (siter K 2 E).regs r := by
      rw [siter_succ]
      exact tail_frame _ 240 (.mov "cpI" (SArg.imm 0)) r p2 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.2.2.1)
    have f4 : (siter K 4 E).regs r = (siter K 3 E).regs r := by
      rw [siter_succ]
      exact tail_frame _ 241 (.setp (.lt) "cpCont" "cpI" (SArg.reg "fLen")) r p3 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using
          Ne.symm hne.2.2.2)
    rw [f4, f3, f2, f1]


/-- **The copy store's predicate is the bound.**  Three steps from 248:
    `cpJ := cpI + lane`, `cpP := cpJ < litLen`, the load — so at the store an
    active lane satisfies `cpI + lane < litLen`, and no bound on `cpI` is needed. -/
theorem copy_pred2 (E : SState) (h248 : E.pc = 248) :
    (∀ i, i ≤ 3 → (siter K i E).pc = 248 + i)
    ∧ (siter K 3 E).pc = 251
    ∧ (∀ l : Lane, (siter K 3 E).regs "cpP" l = 1 →
        E.regs "cpI" l + E.regs "lane" l < E.regs "fLen" l)
    ∧ (∀ r : String, r = "cpI" ∨ r = "lane" ∨ r = "fLen" ∨ r = "cpDstF" ∨ r = "cpDo" →
        (siter K 3 E).regs r = E.regs r)
    ∧ (∀ l : Lane, (siter K 3 E).regs "cpJ" l = E.regs "cpI" l + E.regs "lane" l) := by
  have e0 : siter K 0 E = E := rfl
  have p1 : (siter K 1 E).pc = 249 := by
    rw [siter_succ, e0]; exact tail_pc_binr E 248 (.add) "cpJ" "cpI" "lane" h248 (by decide)
  have p2 : (siter K 2 E).pc = 250 := by
    rw [siter_succ]
    exact tail_pc_setp _ 249 (.lt) "cpP" "cpJ" (SArg.reg "fLen") p1 (by decide)
  have p3 : (siter K 3 E).pc = 251 := by
    rw [siter_succ]
    rw [sstep, show K[(siter K 2 E).pc]? = some (.ldgo "cpB" "cpSo" 0) from by rw [p2]; decide]
    show (siter K 2 E).pc + 1 = 251
    rw [p2]
  have hfr : ∀ r : String, r ≠ "cpJ" → r ≠ "cpP" → r ≠ "cpB" →
      (siter K 3 E).regs r = E.regs r := by
    intro r h1 h2 h3
    have f1 : (siter K 1 E).regs r = E.regs r := by
      rw [siter_succ, e0]
      exact tail_frame E 248 (.binr (.add) "cpJ" "cpI" "lane") r h248 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h1)
    have f2 : (siter K 2 E).regs r = (siter K 1 E).regs r := by
      rw [siter_succ]
      exact tail_frame _ 249 (.setp (.lt) "cpP" "cpJ" (SArg.reg "fLen")) r p1 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h2)
    have f3 : (siter K 3 E).regs r = (siter K 2 E).regs r := by
      rw [siter_succ]
      exact tail_frame _ 250 (.ldgo "cpB" "cpSo" 0) r p2 (by decide)
        (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h3)
    rw [f3, f2, f1]
  refine ⟨fun i hi => ?_, p3, fun l hp => ?_, fun r hr => ?_, fun l => ?_⟩
  · match i, hi with
    | 0, _ => rw [e0]; exact h248
    | 1, _ => exact p1
    | 2, _ => exact p2
    | 3, _ => exact p3
  · -- `cpP` survives the load, and at 249 it is the comparison
    have fP : (siter K 3 E).regs "cpP" l = (siter K 2 E).regs "cpP" l := by
      rw [siter_succ]
      exact congrFun (tail_frame _ 250 (.ldgo "cpB" "cpSo" 0) "cpP" p2 (by decide) (by decide)) l
    have vP : (siter K 2 E).regs "cpP" l
        = (if SCmp.run (.lt) ((siter K 1 E).regs "cpJ" l)
            ((siter K 1 E).get l (SArg.reg "fLen")) then 1 else 0) := by
      rw [siter_succ]
      exact tail_val_setp _ 249 (.lt) "cpP" "cpJ" (SArg.reg "fLen") p1 (by decide) l
    have vJ : (siter K 1 E).regs "cpJ" l = E.regs "cpI" l + E.regs "lane" l := by
      rw [siter_succ, e0, tail_val_binr E 248 (.add) "cpJ" "cpI" "lane" h248 (by decide) l]
      rfl
    have vL : (siter K 1 E).get l (SArg.reg "fLen") = E.regs "fLen" l := by
      rw [siter_succ, e0]
      show (sstep K E).regs "fLen" l = _
      exact congrFun (tail_frame E 248 (.binr (.add) "cpJ" "cpI" "lane") "fLen" h248
        (by decide) (by decide)) l
    rw [fP, vP, vJ, vL] at hp
    by_cases hlt : SCmp.run (.lt) (E.regs "cpI" l + E.regs "lane" l) (E.regs "fLen" l)
    · exact of_decide_eq_true (by simpa only [SCmp.run] using hlt)
    · rw [if_neg hlt] at hp; exact absurd hp (by decide)
  · rcases hr with rfl | rfl | rfl | rfl | rfl <;>
      exact hfr _ (by decide) (by decide) (by decide)
  · have fJ : (siter K 3 E).regs "cpJ" l = (siter K 1 E).regs "cpJ" l := by
      have a2 : (siter K 2 E).regs "cpJ" = (siter K 1 E).regs "cpJ" := by
        rw [siter_succ]
        exact tail_frame _ 249 (.setp (.lt) "cpP" "cpJ" (SArg.reg "fLen")) "cpJ" p1
          (by decide) (by decide)
      have a3 : (siter K 3 E).regs "cpJ" = (siter K 2 E).regs "cpJ" := by
        rw [siter_succ]
        exact tail_frame _ 250 (.ldgo "cpB" "cpSo" 0) "cpJ" p2 (by decide) (by decide)
      rw [show (siter K 3 E).regs "cpJ" l = (siter K 3 E).regs "cpJ" l from rfl, a3, a2]
    rw [fJ, siter_succ, e0, tail_val_binr E 248 (.add) "cpJ" "cpI" "lane" h248 (by decide) l]
    rfl

/-- The straight stretch that ends at the copy store: `248 … 251`.  Its only
    in-edge is `247 → 248`, so a visit to the store descends from a visit to 248
    and `copy_pred2` applies there. -/
def preStoreS2 : List Nat := (List.range 4).map (· + 248)

theorem preStoreS2_entry_lt : ∀ q, q < 274 → q ∉ preStoreS2 →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K q, q' ∈ preStoreS2 → q' = 248 :=
  ivEntry_at K 248 4 248 shipped32_size (by decide)

theorem preStoreS2_entry : ∀ q, q ∉ preStoreS2 →
    ∀ q', q' ∈ AlgorithmLib.LZ4Simt.succsOf K q → q' ∈ preStoreS2 → q' = 248 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact preStoreS2_entry_lt q h hq q' hq' hin
  · rw [show AlgorithmLib.LZ4Simt.succsOf K q = [q] from by
      simp only [AlgorithmLib.LZ4Simt.succsOf,
        Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

/-- 251 leaves the stretch, so a visit to the store is exactly three steps past
    its entry. -/
theorem preStore_succ2 : AlgorithmLib.LZ4Simt.succsOf K 251 = [252] := by decide
theorem preStore_pin2 (ss : SState) (h0p : ss.pc ∉ preStoreS2) (k : Nat)
    (hpc : (siter K k ss).pc = 251) :
    ∃ j, k = j + 3 ∧ (siter K j ss).pc = 248 := by
  have hmem : (siter K k ss).pc ∈ preStoreS2 := by rw [hpc]; decide
  obtain ⟨j, hjk, hje, hall⟩ := region_entry K preStoreS2 248 preStoreS2_entry ss h0p k hmem
  obtain ⟨hpcs, -, -, -, -⟩ := copy_pred2 (siter K j ss) hje
  refine ⟨j, ?_, hje⟩
  have hshift : ∀ a : Nat, siter K (j + a) ss = siter K a (siter K j ss) :=
    fun a => siter_add K j a ss
  rcases Nat.lt_or_ge k (j + 3) with hlt | hge
  · exfalso
    have hle : k - j ≤ 3 := by omega
    have hh := hpcs (k - j) hle
    rw [← hshift (k - j), show j + (k - j) = k from by omega] at hh
    rw [hpc] at hh; omega
  · rcases Nat.lt_or_ge (j + 3) k with hgt | hle2
    · exfalso
      have h251 : (siter K (j + 3) ss).pc = 251 := by
        rw [hshift 3]; exact (copy_pred2 (siter K j ss) hje).2.1
      have hs : (siter K (j + 3 + 1) ss).pc
          ∈ AlgorithmLib.LZ4Simt.succsOf K (siter K (j + 3) ss).pc := by
        rw [show (siter K (j + 3 + 1) ss) = sstep K (siter K (j + 3) ss) from siter_succ K (j+3) ss]
        exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K _
      rw [h251, preStore_succ2, List.mem_singleton] at hs
      have hin := hall (j + 3 + 1) (by omega) (by omega)
      rw [hs] at hin
      exact absurd hin (by decide)
    · omega

theorem copy_head2_pin2 (ss : SState) (h0c : ss.pc ∉ copyS2) (k : Nat)
    (hpc : (siter K k ss).pc = 251) :
    ∃ j, j + 4 ≤ k ∧ (siter K j ss).pc = 238
      ∧ ∀ i, j ≤ i → i ≤ k → (siter K i ss).pc ∈ copyS2 := by
  have hmem : (siter K k ss).pc ∈ copyS2 := by rw [hpc]; decide
  obtain ⟨j, hjk, hje, hall⟩ := region_entry K copyS2 238 copyS2_entry ss h0c k hmem
  obtain ⟨hpcs, -, -, -⟩ := copy_head2 (siter K j ss) hje
  refine ⟨j, ?_, hje, hall⟩
  rcases Nat.lt_or_ge k (j + 4) with hlt | hge
  · exfalso
    have hle : k - j ≤ 4 := by omega
    have hh := hpcs (k - j) hle
    rw [← siter_add K j (k - j) ss, show j + (k - j) = k from by omega] at hh
    rw [hpc] at hh; omega
  · exact hge

theorem copyLoop_exit_succ2 : AlgorithmLib.LZ4Simt.succsOf K 255 = [256] := by decide
theorem copy_const2 (ss : SState) (j k : Nat) (hjk : j ≤ k)
    (h242 : (siter K j ss).pc = 242)
    (hall : ∀ i, j ≤ i → i ≤ k → (siter K i ss).pc ∈ copyS2)
    (r : String) (hr : r ∈ ["op", "outBase", "fLen", "cpDstF", "lane"]) :
    (siter K k ss).regs r = (siter K j ss).regs r := by
  have hne : ∀ i, i < k - j → (siter K i (siter K j ss)).pc ∉ [255] := by
    intro i hi hin
    rw [← siter_add K j i ss] at hin
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hin
    have hs : (siter K (j + i + 1) ss).pc
        ∈ AlgorithmLib.LZ4Simt.succsOf K (siter K (j + i) ss).pc := by
      rw [show siter K (j + i + 1) ss = sstep K (siter K (j + i) ss) from siter_succ K (j + i) ss]
      exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K _
    rw [hin, copyLoop_exit_succ2, List.mem_singleton] at hs
    have hmem := hall (j + i + 1) (by omega) (by omega)
    rw [hs] at hmem
    exact absurd hmem (by decide)
  have hconst := AlgorithmLib.LZ4Simt.regs_const_on K r ((List.range 14).map (· + 242)) [255]
    copyLoop_closed2 (copy_no_dest2 r hr) (siter K j ss) (by rw [h242]; decide) (k - j) hne
  rw [← siter_add K j (k - j) ss, show j + (k - j) = k from by omega] at hconst
  exact hconst

/-- **The tail copy's store is confined.**  The same four steps as
    `cpDo_confined`, on the final literal run: the visit descends from 248 (the
    predicate) and from 238 (the setup), the five registers are carried across
    the loop, and `tail_copy_budget` pays for the whole run.  `cpDstF = outBase +
    op` with `cpJ < fLen`, so no bound on `cpI` is needed here either. -/
theorem cpDo_confined2 (LO : Nat) (hLO : LO < 2 ^ 64) (base : Nat) (E : SState)
    (h209 : E.pc = 209)
    (hlaN : (E.regs "litAnchor" 0).toNat ≤ 32768)
    (hlau : ∀ j : Lane, E.regs "litAnchor" j = E.regs "litAnchor" 0)
    (hopu : ∀ j : Lane, E.regs "op" j = E.regs "op" 0)
    (hbud : (E.regs "op" 0).toNat + (32768 - (E.regs "litAnchor" 0).toNat)
        + (32768 - (E.regs "litAnchor" 0).toNat) / 255 + 2 ≤ LO)
    (l : Lane) (k : Nat)
    (hpc : (siter K k E).pc = 251)
    (hact : (siter K k E).regs "cpP" l = 1)
    (hcpdo : (siter K k E).regs "cpDo" l
      = ((siter K k E).regs "cpDstF" l + (siter K k E).regs "cpI" l)
        + (siter K k E).regs "lane" l)
    (hob : (siter K k E).regs "outBase" l = UInt64.ofNat base)
    (htop : base + LO + 4 < 2 ^ 64) :
    base ≤ ((siter K k E).regs "cpDo" l).toNat
      ∧ ((siter K k E).regs "cpDo" l).toNat < base + LO + 4 := by
  -- the predicate half
  obtain ⟨j2, hk3, h248⟩ := preStore_pin2 E (by rw [h209]; decide) k hpc
  have hE2 : siter K 3 (siter K j2 E) = siter K k E := by
    rw [← siter_add K j2 3 E, ← hk3]
  obtain ⟨-, -, hpred, hfr2, hJ⟩ := copy_pred2 (siter K j2 E) h248
  rw [hE2] at hpred hfr2 hJ
  have hlt : (siter K j2 E).regs "cpI" l + (siter K j2 E).regs "lane" l
      < (siter K j2 E).regs "fLen" l := hpred l hact
  -- the setup half
  obtain ⟨j1, h4k, h238, hall⟩ := copy_head2_pin2 E (by rw [h209]; decide) k hpc
  obtain ⟨-, h242, hDst, hFr⟩ := copy_head2 (siter K j1 E) h238
  have hE1 : siter K 4 (siter K j1 E) = siter K (j1 + 4) E := (siter_add K j1 4 E).symm
  rw [hE1] at h242 hDst hFr
  have hconst : ∀ r : String, r ∈ ["op", "outBase", "fLen", "cpDstF", "lane"] →
      (siter K k E).regs r = (siter K (j1 + 4) E).regs r :=
    fun r hr => copy_const2 E (j1 + 4) k h4k h242 (fun i h1 h2 => hall i (by omega) h2) r hr
  -- the budget, at the copy's setup
  have hbudk := tail_copy_budget LO hLO E h209 hlaN hlau hopu hbud l j1 h238
  -- move everything to the visit
  have hopk : (siter K k E).regs "op" l = (siter K j1 E).regs "op" l := by
    rw [congrFun (hconst "op" (by decide)) l, congrFun (hFr "op" (Or.inl rfl)) l]
  have hflk : (siter K k E).regs "fLen" l = (siter K j1 E).regs "fLen" l := by
    rw [congrFun (hconst "fLen" (by decide)) l, congrFun (hFr "fLen" (Or.inr (Or.inr rfl))) l]
  have hobk : (siter K k E).regs "outBase" l = (siter K j1 E).regs "outBase" l := by
    rw [congrFun (hconst "outBase" (by decide)) l, congrFun (hFr "outBase" (Or.inr (Or.inl rfl))) l]
  have hdstk : (siter K k E).regs "cpDstF" l
      = (siter K j1 E).regs "outBase" l + (siter K j1 E).regs "op" l := by
    rw [congrFun (hconst "cpDstF" (by decide)) l, hDst l]
  have e1 : (siter K k E).regs "cpI" l = (siter K j2 E).regs "cpI" l :=
    congrFun (hfr2 "cpI" (Or.inl rfl)) l
  have e2 : (siter K k E).regs "lane" l = (siter K j2 E).regs "lane" l :=
    congrFun (hfr2 "lane" (Or.inr (Or.inl rfl))) l
  have e3 : (siter K k E).regs "fLen" l = (siter K j2 E).regs "fLen" l :=
    congrFun (hfr2 "fLen" (Or.inr (Or.inr (Or.inl rfl)))) l
  have hcjU : (siter K k E).regs "cpI" l + (siter K k E).regs "lane" l
      < (siter K k E).regs "fLen" l := by rw [e1, e2, e3]; exact hlt
  have hcjlt : ((siter K k E).regs "cpI" l + (siter K k E).regs "lane" l).toNat
      < ((siter K j1 E).regs "fLen" l).toNat := by
    rw [← hflk]; exact UInt64.lt_iff_toNat_lt.mp hcjU
  have hfinal : (siter K k E).regs "cpDo" l
      = ((siter K j1 E).regs "outBase" l + (siter K j1 E).regs "op" l)
        + ((siter K k E).regs "cpI" l + (siter K k E).regs "lane" l) := by
    rw [hcpdo, hdstk, UInt64.add_assoc]
  rw [hfinal]
  refine cpDo_region base LO ((siter K j1 E).regs "outBase" l) ((siter K j1 E).regs "op" l)
    ((siter K k E).regs "cpI" l + (siter K k E).regs "lane" l) (by rw [← hobk]; exact hob)
    (by omega) htop

set_option maxHeartbeats 1000000 in
/-- **The tail copy's store site, on the launch trace** — the sixteenth and last.
    Same entry as `cursorAtSites_shipped`'s tail branch: the checkpoint at the
    loop exit carries the budget, lane-uniformity of `op`/`litAnchor` and the
    tight bound `litAnchor ≤ inStride`. -/
theorem stores_cpDo251 (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 15).numBlk → inPtr + w * (WP.mk 15).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 15).numBlk →
      inPtr + w * (WP.mk 15).inStride + (WP.mk 15).inStride
        ≤ outPtr + w * (WP.mk 15).outStride)
    (w : Fin (WP.mk 15).numBlk) (k : Nat) (l : Lane)
    (hpc : (siter K k (initSt w.val inPtr outPtr gm smemB)).pc = 251)
    (hact : (siter K k (initSt w.val inPtr outPtr gm smemB)).regs "cpP" l = 1) :
    Lz4Interleave.outRegion 15 outPtr w.val
      (((siter K k (initSt w.val inPtr outPtr gm smemB)).regs "cpDo" l).toNat) := by
  obtain ⟨n, hpc39, hpc40, hck, hpc209, wE, hcE, hQE, hTE, hlaE⟩ :=
    shipped_loop_ckpt w.val inPtr outPtr gm smemB w.isLt
      (by have := w.isLt; have := numBlk32; omega)
      (hib40 _ w.isLt) (htop32 _ w.isLt) (hbuf _ w.isLt) hderive (hdisj _ w.isLt)
  have hkn : preSteps + 1 + n ≤ k := by
    rcases Nat.lt_or_ge k (preSteps + 1 + n) with hlt | hge
    · exfalso
      have hst := stays_from_210 (initSt w.val inPtr outPtr gm smemB) k (by rw [hpc]; omega)
        (preSteps + 1 + n) (by omega)
      rw [hpc209] at hst; omega
    · exact hge
  have hsplit : siter K (k - (preSteps + 1 + n))
      (siter K (preSteps + 1 + n) (initSt w.val inPtr outPtr gm smemB))
      = siter K k (initSt w.val inPtr outPtr gm smemB) := by
    rw [← siter_add K (preSteps + 1 + n) (k - (preSteps + 1 + n)),
      show preSteps + 1 + n + (k - (preSteps + 1 + n)) = k from by omega]
  have hlaC : ∀ j : Lane,
      (siter K (preSteps + 1 + n) (initSt w.val inPtr outPtr gm smemB)).regs "litAnchor" j
        = wE.regs "litAnchor" := fun j => hcE.reg (by decide) j
  have hopC : ∀ j : Lane,
      (siter K (preSteps + 1 + n) (initSt w.val inPtr outPtr gm smemB)).regs "op" j
        = wE.regs "op" := fun j => hcE.reg (by decide) j
  simp only [AlgorithmLib.LZ4WarpDSL.TightQ, AlgorithmLib.LZ4WarpDSL.tightRem] at hTE
  have hfin := cpDo_confined2 (WP.mk 15).lenOff (by decide)
    (outPtr + w.val * (WP.mk 15).outStride)
    (siter K (preSteps + 1 + n) (initSt w.val inPtr outPtr gm smemB)) hpc209
    (by rw [hlaC 0]; exact Nat.le_trans hlaE (by decide))
    (fun j => by rw [hlaC j, hlaC 0])
    (fun j => by rw [hopC j, hopC 0])
    (by rw [hlaC 0, hopC 0]
        rw [show (32768 : Nat) = (WP.mk 15).inStride from rfl]
        omega)
    l (k - (preSteps + 1 + n)) (by rw [hsplit]; exact hpc)
    (by rw [hsplit]; exact hact)
    (by rw [hsplit]; exact (cpDo_at_store w.val inPtr outPtr gm smemB k l).2 hpc)
    (by rw [hsplit]
        exact outBase_at_store_site w.val inPtr outPtr gm smemB w.isLt
          (by have := w.isLt; have := numBlk32; omega) hderive k l
          (by rw [hpc]; omega) (by rw [hpc]; omega))
    (by have h := htop32 w.val w.isLt
        have hl : (WP.mk 15).lenOff = 35072 := rfl
        have hi : (WP.mk 15).inStride = 32768 := rfl
        omega)
  rw [hsplit] at hfin
  exact hfin

-- ── The load half: the kernel clamps its own search addresses ────────────────

/-- The seven checks, unpacked.  No `decide` here — `p` is a variable. -/
theorem win_shapeG (p : Array SInstr) (S : Nat) (h : winShapeB p S = true) :
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

theorem win_ftG (p : Array SInstr) (S : Nat) (h : winShapeB p S = true) :
    ∀ q, 42 ≤ q → q ≤ 92 → (p[q]?.map fallthroughOnlyB) = some true := by
  intro q h1 h2; exact forall_window 42 51 (win_shapeG p S h).1 q h1 (by omega)

/-- **`cap4 = S - 4` throughout the window**, pcs 45–92. -/
theorem cap4_in_windowG (p : Array SInstr) (S : Nat) (h : winShapeB p S = true)
    (ss : SState) (h0 : ss.pc < 45) :
    ∀ k : Nat, 45 ≤ (siter p k ss).pc → (siter p k ss).pc ≤ 92 →
      ∀ l : Lane, (siter p k ss).regs "cap4" l = UInt64.ofNat (S - 4) :=
  const_in_region p "cap4" 45 92 (S - 4) (win_shapeG p S h).2.2.2.2.1
    (fun q h1 h2 => win_ftG p S h q (by omega) h2)
    (fun q h1 h2 => forall_window 45 47 (win_shapeG p S h).2.1 q h1 (by omega))
    ss h0

/-- **Both search clamps hold at every state of the window**, at any geometry:
    `45 rp := min posP cap4` and `64 rc := min cand cap4`, so the kernel clamps
    its own read pointer and no loop invariant has to be threaded to reach it. -/
theorem rp_clampedG (p : Array SInstr) (S : Nat) (hS : S < 2 ^ 64) (h : winShapeB p S = true)
    (ss : SState) (h0 : ss.pc < 45) :
    ∀ k : Nat, 46 ≤ (siter p k ss).pc → (siter p k ss).pc ≤ 92 →
      ∀ l : Lane, ((siter p k ss).regs "rp" l).toNat ≤ S - 4 :=
  reg_prop_in_region p "rp" 46 92 (fun v => v.toNat ≤ S - 4)
    (fun q h1 h2 => win_ftG p S h q (by omega) h2)
    (fun q h1 h2 => forall_window 46 46 (win_shapeG p S h).2.2.1 q h1 (by omega))
    ss (by omega)
    (fun k hk l => by
      rw [sstep, show p[(siter p k ss).pc]?
        = some (SInstr.binr .min "rp" "posP" "cap4") from by
          rw [hk]; exact (win_shapeG p S h).2.2.2.2.2.1]
      simp only [sstepInstr, SState.setReg, SState.setPc, if_true]
      refine Nat.le_trans (min_run_le_right _ _) ?_
      rw [cap4_in_windowG p S h ss h0 k (by omega) (by omega) l]
      exact Nat.le_of_eq (AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by omega)))

theorem rc_clampedG (p : Array SInstr) (S : Nat) (hS : S < 2 ^ 64) (h : winShapeB p S = true)
    (ss : SState) (h0 : ss.pc < 45) :
    ∀ k : Nat, 65 ≤ (siter p k ss).pc → (siter p k ss).pc ≤ 92 →
      ∀ l : Lane, ((siter p k ss).regs "rc" l).toNat ≤ S - 4 :=
  reg_prop_in_region p "rc" 65 92 (fun v => v.toNat ≤ S - 4)
    (fun q h1 h2 => win_ftG p S h q (by omega) h2)
    (fun q h1 h2 => forall_window 65 27 (win_shapeG p S h).2.2.2.1 q h1 (by omega))
    ss (by omega)
    (fun k hk l => by
      rw [sstep, show p[(siter p k ss).pc]?
        = some (SInstr.binr .min "rc" "cand" "cap4") from by
          rw [hk]; exact (win_shapeG p S h).2.2.2.2.2.2]
      simp only [sstepInstr, SState.setReg, SState.setPc, if_true]
      refine Nat.le_trans (min_run_le_right _ _) ?_
      rw [cap4_in_windowG p S h ss h0 k (by omega) (by omega) l]
      exact Nat.le_of_eq (AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by omega)))

-- ── The shipped 32 KiB instance ─────────────────────────────────────────────

theorem winShapeB_32 : winShapeB K 32768 = true := by decide


theorem rp_clamped (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, 46 ≤ (siter K k (initSt w inPtr outPtr gm smemB)).pc →
      (siter K k (initSt w inPtr outPtr gm smemB)).pc ≤ 92 →
      ∀ l : Lane, ((siter K k (initSt w inPtr outPtr gm smemB)).regs "rp" l).toNat ≤ 32764 :=
  rp_clampedG K 32768 (by decide) winShapeB_32 (initSt w inPtr outPtr gm smemB)
    (by show (0:Nat) < 45; decide)

theorem rc_clamped (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, 65 ≤ (siter K k (initSt w inPtr outPtr gm smemB)).pc →
      (siter K k (initSt w inPtr outPtr gm smemB)).pc ≤ 92 →
      ∀ l : Lane, ((siter K k (initSt w inPtr outPtr gm smemB)).regs "rc" l).toNat ≤ 32764 :=
  rc_clampedG K 32768 (by decide) winShapeB_32 (initSt w inPtr outPtr gm smemB)
    (by show (0:Nat) < 45; decide)


-- ── The same window check at the 64 KiB geometry ───────────────────────────

theorem winShapeB_64 : winShapeB (WP.mk 16).kernel 65536 = true := by decide

end Lz4Sites