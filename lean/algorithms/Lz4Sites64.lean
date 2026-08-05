import Lz4Sites

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

/-!
  # The geometry-specific facts, re-decided at 64 KiB

  Everything here is the corresponding `Lz4Sites` theorem with `blkLog := 16`.
  The proofs are unchanged; only the immediates move, and every `decide` is
  re-run against the 64 KiB array.  Nothing is assumed.
-/

namespace Lz4Sites


open Algorithm
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

/-- The 64 KiB kernel, as a reducible abbreviation — mirrors `K` at 32 KiB.
    Using the raw projection instead changes how `rw`/`whnf` unfold it. -/
abbrev K16 := (WP.mk 16).kernel

theorem nbpos16 : 0 < (WP.mk 16).numBlk := by decide
theorem nb16 : (WP.mk 16).numBlk < 2 ^ 64 := by decide


theorem numBlk3264 : (WP.mk 16).numBlk = 3200 := by decide

theorem kSize16 : K16.size = 274 := by decide





theorem load_regs3264 :
    ∀ s ∈ loadSites K16, s.2.1 ∈ loadRegs ∧ s.2.2 ≤ 3 := by
  rw [shipped64_load_sites]; decide

theorem store_regs3264 :
    ∀ s ∈ storeSites K16, s.2 ∈ storeRegs := by
  rw [shipped64_store_sites]; decide

theorem unconditioned_form_is_false64 (gm : Array UInt8) (smemB : List UInt8) :
    ¬ (∀ (w : Fin (WP.mk 16).numBlk) (k : Nat) (r : String), r ∈ storeRegs →
        ∀ l : Lane,
          Lz4Interleave.outRegion 16 (WP.mk 16).totIn w.val
            (((siter K16 k (initSt w.val 0 (WP.mk 16).totIn gm smemB)).regs
              r l).toNat)) := by
  intro h
  have hw : (0 : Nat) < (WP.mk 16).numBlk := by decide
  have := h ⟨0, hw⟩ 0 "sbAddr" (by decide) 0
  -- at `k = 0` the state is the launch state and `sbAddr` is still the default 0
  have hz : ((siter K16 0 (initSt 0 0 (WP.mk 16).totIn gm smemB)).regs
      "sbAddr" (0 : Lane)).toNat = 0 := rfl
  rw [hz] at this
  obtain ⟨hge, _⟩ := this
  have : (WP.mk 16).totIn = 209715200 := by decide
  omega

theorem sbAddr_is_outBase_add_op64 (w inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (k q : Nat) (hq : q ∈ sbAddrSites)
    (hpc : (siter K16 (k + 1)
      (initSt w inPtr outPtr gm smemB)).pc = q) (l : Lane) :
    (siter K16 (k + 1) (initSt w inPtr outPtr gm smemB)).regs "sbAddr" l
      = (siter K16 (k + 1) (initSt w inPtr outPtr gm smemB)).regs "outBase" l
        + (siter K16 (k + 1) (initSt w inPtr outPtr gm smemB)).regs "op" l := by
  rw [siter_succ] at hpc ⊢
  have go : ∀ r : Nat, r ∈ sbAddrSites →
      (K16[r]?.map fallthroughOnlyB) = some true ∧
      K16[r - 1]? = some (.bin .add "sbAddr" "outBase" (.reg "op")) ∧
      0 < r := by decide
  obtain ⟨h1, h2, h3⟩ := go q hq
  exact add_above_holds_at' _ q "sbAddr" "outBase" "op" h1 h2 (by decide) (by decide) h3 _ hpc l

theorem la_at_store64 (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) (k : Nat) :
    let S := siter K16 k (initSt w inPtr outPtr gm smemB)
    ∀ l : Lane,
      (S.pc = 259 → S.regs "la0" l = S.regs "outBase" l + UInt64.ofNat (WP.mk 16).lenOff) ∧
      (S.pc = 263 → S.regs "la1" l = S.regs "outBase" l + UInt64.ofNat ((WP.mk 16).lenOff + 1)) ∧
      (S.pc = 267 → S.regs "la2" l = S.regs "outBase" l + UInt64.ofNat ((WP.mk 16).lenOff + 2)) ∧
      (S.pc = 271 → S.regs "la3" l = S.regs "outBase" l + UInt64.ofNat ((WP.mk 16).lenOff + 3)) := by
  intro S l
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  have base : ∀ (q n : Nat), q - n - 1 = 257 → n + 1 ≤ q →
      (∀ t, t < n + 1 → (K16[q - t]?.map fallthroughOnlyB) = some true) →
      (∀ t, t < n → (K16[q - t - 1]?.map
        (fun i => destOf i != some "la0")) = some true) →
      (∀ t, t < n → (K16[q - t - 1]?.map
        (fun i => destOf i != some "outBase")) = some true) →
      S.pc = q → S.regs "la0" l = S.regs "outBase" l + UInt64.ofNat 69888 := by
    intro q n h257 hn hft hd ha hpc
    exact add_imm_carried _ _ hinit "la0" "outBase" 69888 q n hft hd ha
      (by rw [h257]; decide) (by decide) hn k hpc l
  have step : ∀ (d : String) (j q n : Nat), n + 1 ≤ q →
      K16[q - n - 1]? = some (.bin .add d "la0" (.imm j)) →
      (∀ t, t < n + 1 → (K16[q - t]?.map fallthroughOnlyB) = some true) →
      (∀ t, t < n → (K16[q - t - 1]?.map
        (fun i => destOf i != some d)) = some true) →
      (∀ t, t < n → (K16[q - t - 1]?.map
        (fun i => destOf i != some "la0")) = some true) →
      d ≠ "la0" → S.pc = q → S.regs d l = S.regs "la0" l + UInt64.ofNat j := by
    intro d j q n hn hpre hft hd ha hne hpc
    exact add_imm_carried _ _ hinit d "la0" j q n hft hd ha hpre hne hn k hpc l
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_⟩
  · exact base 259 1 rfl (by omega) (by decide) (by decide) (by decide) h
  · rw [step "la1" 1 263 2 (by omega) (by decide) (by decide) (by decide) (by decide)
        (by decide) h,
      base 263 5 rfl (by omega) (by decide) (by decide) (by decide) h]
    rw [show (WP.mk 16).lenOff + 1 = 69888 + 1 from rfl, UInt64.ofNat_add,
      UInt64.add_assoc]
  · rw [step "la2" 2 267 2 (by omega) (by decide) (by decide) (by decide) (by decide)
        (by decide) h,
      base 267 9 rfl (by omega) (by decide) (by decide) (by decide) h]
    rw [show (WP.mk 16).lenOff + 2 = 69888 + 2 from rfl, UInt64.ofNat_add,
      UInt64.add_assoc]
  · rw [step "la3" 3 271 2 (by omega) (by decide) (by decide) (by decide) (by decide)
        (by decide) h,
      base 271 13 rfl (by omega) (by decide) (by decide) (by decide) h]
    rw [show (WP.mk 16).lenOff + 3 = 69888 + 3 from rfl, UInt64.ofNat_add,
      UInt64.add_assoc]

theorem cpDo_at_store64 (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (k : Nat) (l : Lane) :
    let S := siter K16 k (initSt w inPtr outPtr gm smemB)
    (S.pc = 165 → S.regs "cpDo" l = S.regs "cpDst" l + S.regs "cpI" l + S.regs "lane" l) ∧
    (S.pc = 251 → S.regs "cpDo" l = S.regs "cpDstF" l + S.regs "cpI" l + S.regs "lane" l) := by
  intro S
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  have go : ∀ (dst : String) (q : Nat), q - 5 - 2 = q - 7 →
      K16[q - 5 - 2]? = some (.binr .add "cpDo" dst "cpI") →
      K16[q - 5 - 1]? = some (.binr .add "cpDo" "cpDo" "lane") →
      (∀ t, t < 7 → (K16[q - t]?.map fallthroughOnlyB) = some true) →
      (∀ t, t < 5 → (K16[q - t - 1]?.map
        (fun i => destOf i != some "cpDo")) = some true) →
      (∀ t, t < 7 → (K16[q - t - 1]?.map
        (fun i => destOf i != some dst)) = some true) →
      (∀ t, t < 7 → (K16[q - t - 1]?.map
        (fun i => destOf i != some "cpI")) = some true) →
      (∀ t, t < 7 → (K16[q - t - 1]?.map
        (fun i => destOf i != some "lane")) = some true) →
      7 ≤ q → S.pc = q →
      S.regs "cpDo" l = S.regs dst l + S.regs "cpI" l + S.regs "lane" l := by
    intro dst q _ h1 h2 hft hnwd hnwa hnwb hnwc hq hpc
    exact binr_pair_carried _ _ hinit .add "cpDo" dst "cpI" "lane" q 5
      hft hnwd hnwa hnwb hnwc h1 h2 (by omega) k hpc l
  exact ⟨go "cpDst" 165 rfl (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by omega),
    go "cpDstF" 251 rfl (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by omega)⟩

instance loads3264 : Loads K16 where
  loadAt := load_at_site64
  storeCpDo := by rw [shipped64_store_sites]; decide
  loadSitesEq := shipped64_load_sites

theorem shape_lo64 : ∀ t, t < 25 → stepShapeB K16 272 t = true := by decide

theorem shape_hi64 : ∀ q, 272 ≤ q → stepShapeB K16 272 q = true := by
  intro q hq
  rcases Nat.lt_or_ge q 274 with h | h
  · have : q = 272 ∨ q = 273 := by omega
    rcases this with rfl | rfl <;> decide
  · rw [stepShapeB, Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)]

theorem prologue_pc_shape64 (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k, k ≤ 25 →
      (siter K16 k (initSt w inPtr outPtr gm smemB)).pc ≤ k ∨
      272 ≤ (siter K16 k (initSt w inPtr outPtr gm smemB)).pc := by
  intro k
  induction k with
  | zero => intro _; exact Or.inl (Nat.le_refl _)
  | succ m ih =>
      intro hm
      have hstep := ih (by omega)
      rw [siter_succ]
      rcases hstep with hlo | hhi
      · have hs : stepShapeB K16 272
            (siter K16 m (initSt w inPtr outPtr gm smemB)).pc = true :=
          shape_lo64 _ (by omega)
        rcases pc_next K16 272 _ _ rfl hs with e | e | e <;> omega
      · have hs : stepShapeB K16 272
            (siter K16 m (initSt w inPtr outPtr gm smemB)).pc = true :=
          shape_hi64 _ hhi
        rcases pc_next K16 272 _ _ rfl hs with e | e | e <;> omega

theorem prologue_not_at_store_site64 (w inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (k : Nat) (hk : k ≤ 25) (r : String) :
    ¬ ((siter K16 k (initSt w inPtr outPtr gm smemB)).pc, r)
        ∈ storeSites K16 := by
  intro hmem
  have hrange : ∀ s ∈ storeSites K16, 130 ≤ s.1 ∧ s.1 < 272 := by
    rw [shipped64_store_sites]; decide
  obtain ⟨h1, h2⟩ := hrange _ hmem
  rcases prologue_pc_shape64 w inPtr outPtr gm smemB k hk with e | e <;> omega

theorem noDest2064 : noDestFrom K16 "outBase" 20 = true := by
  rw [noDestFrom_eq _ _ _ (by rw [kSize16]; omega)]; decide
theorem noExit2064 : noExitBelow K16 20 = true := by
  rw [noExitBelow_eq _ _ (by rw [kSize16]; omega)]; decide
theorem nbpos64 : 0 < (WP.mk 16).numBlk := by decide

set_option maxHeartbeats 2000000 in
theorem outBase_at_store_site64 (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hw : w < (WP.mk 16).numBlk) (hw64 : w * 32 + 32 < 2 ^ 64)
    (hderive : outPtr = inPtr + ((WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (k : Nat) (l : Fin 32)
    (h130 : 130 ≤ (siter K16 k (initSt w inPtr outPtr gm smemB)).pc)
    (h272 : (siter K16 k (initSt w inPtr outPtr gm smemB)).pc < 272) :
    (siter K16 k (initSt w inPtr outPtr gm smemB)).regs "outBase" l
      = UInt64.ofNat (outPtr + w * (WP.mk 16).outStride) := by
  have hk : 26 ≤ k := by
    rcases Nat.lt_or_ge k 26 with hlt | hge
    · rcases prologue_pc_shape64 w inPtr outPtr gm smemB k (by omega) with e | e <;> omega
    · exact hge
  have hker : K16 = AlgorithmLib.LZ4WarpDSL.warpKernelDSL
      (WP.mk 16).numBlk (WP.mk 16).inStride (WP.mk 16).outStride (WP.mk 16).lenOff wHashLog :=
    rfl
  obtain ⟨hpc25, -, -, -, hob, -⟩ :=
    AlgorithmLib.LZ4WarpDSL.head25 (WP.mk 16).numBlk (WP.mk 16).inStride (WP.mk 16).outStride
      (WP.mk 16).lenOff wHashLog w inPtr outPtr gm smemB nbpos64 nb16 hw hw64 hderive
  rw [← hker, ← siter_eq_snsteps] at hpc25 hob
  have hsplit : siter K16 k (initSt w inPtr outPtr gm smemB)
      = siter K16 (k - 25)
          (siter K16 25 (initSt w inPtr outPtr gm smemB)) := by
    rw [← siter_add, show 25 + (k - 25) = k from by omega]
  have hconst := regs_const_from K16 "outBase" 20 noDest2064 noExit2064
    (siter K16 25 (initSt w inPtr outPtr gm smemB)) (by rw [hpc25]; omega) (k - 25)
  rw [hsplit, hconst.1]
  exact (show rOutBase = "outBase" from rfl) ▸ hob l

theorem sbAddr_confined_of_cursor64 (inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (h : CursorAtSites 16 inPtr outPtr gm smemB)
    (hderive : outPtr = inPtr + ((WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (w : Fin (WP.mk 16).numBlk) (k : Nat) (l : Lane)
    (hpc : (siter (WP.mk 16).kernel (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc
      ∈ sbAddrSites)
    (htop : outPtr + w.val * (WP.mk 16).outStride + (WP.mk 16).lenOff + 4 < 2 ^ 64) :
    Lz4Interleave.outRegion 16 outPtr w.val
      (((siter (WP.mk 16).kernel (k + 1)
        (initSt w.val inPtr outPtr gm smemB)).regs "sbAddr" l).toNat) := by
  obtain ⟨q, hq, hpcq⟩ : ∃ q, q ∈ sbAddrSites ∧
      (siter (WP.mk 16).kernel (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = q :=
    ⟨_, hpc, rfl⟩
  have hadd := sbAddr_is_outBase_add_op64 w.val inPtr outPtr gm smemB k q hq hpcq l
  have hob : ((siter (WP.mk 16).kernel (k + 1)
      (initSt w.val inPtr outPtr gm smemB)).regs "outBase" l).toNat
      = outPtr + w.val * (WP.mk 16).outStride := by
    have hr : ∀ q ∈ sbAddrSites, 130 ≤ q ∧ q < 272 := by decide
    rw [outBase_at_store_site64 w.val inPtr outPtr gm smemB w.isLt
      (by have h := w.isLt; have hn := numBlk3264; omega) hderive (k + 1) l
      (hr _ hpc).1 (hr _ hpc).2]
    exact UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega)
  have hop := h.opLe w k l hpc
  have hnw : ((siter (WP.mk 16).kernel (k + 1)
      (initSt w.val inPtr outPtr gm smemB)).regs "outBase" l).toNat
    + ((siter (WP.mk 16).kernel (k + 1)
      (initSt w.val inPtr outPtr gm smemB)).regs "op" l).toNat < 2 ^ 64 := by
    rw [hob]; omega
  rw [hadd, toNat_add_of_lt _ _ hnw, hob]
  have hprod : (WP.mk 16).numBlk * (WP.mk 16).inStride = 209715200 := by decide
  have hls : (WP.mk 16).lenOff = 69888 := by decide
  have hos : (WP.mk 16).outStride = 69896 := by decide
  exact ⟨by omega, by omega⟩

theorem outBase_const_after_prologue64 (st0 : AlgorithmLib.LZ4Simt.SState)
    (h0 : 39 ≤ st0.pc) (k : Nat) :
    (siter K16 k st0).regs "outBase" = st0.regs "outBase" ∧
    39 ≤ (siter K16 k st0).pc :=
  regs_const_from K16 "outBase" 39
    (by rw [noDestFrom_eq _ _ _ (by rw [kSize16]; omega)]; decide)
    (by rw [noExitBelow_eq _ _ (by rw [kSize16]; omega)]; decide) st0 h0 k

theorem inBase_const_after_prologue64 (st0 : AlgorithmLib.LZ4Simt.SState)
    (h0 : 39 ≤ st0.pc) (k : Nat) :
    (siter K16 k st0).regs "inBase" = st0.regs "inBase" ∧
    39 ≤ (siter K16 k st0).pc :=
  regs_const_from K16 "inBase" 39
    (by rw [noDestFrom_eq _ _ _ (by rw [kSize16]; omega)]; decide)
    (by rw [noExitBelow_eq _ _ (by rw [kSize16]; omega)]; decide) st0 h0 k

theorem shipped64_op_accumulates :
    ((opWriteSites K16).filter (fun s => decide (37 < s.1))).all
      (fun s => match s.2 with
                | .bin .add "op" "op" _ => true
                | _ => false) = true := by
  rw [shipped64_op_writes]; decide


theorem tailPre_closed64 : PcClosed K16 tailPre [216] :=
  tailPre_iv ▸ ivClosed_at K16 208 9 [216] kSize16 (by omega) (by decide)

theorem op_const_to_21664 (st : SState) (h0 : st.pc = 208) (k : Nat)
    (hne : ∀ j, j < k → (siter K16 j st).pc ∉ [216]) :
    (siter K16 k st).regs "op" = st.regs "op" :=
  regs_const_on K16 "op" tailPre [216] tailPre_closed64 (by decide) st
    (by rw [h0]; decide) k hne

theorem lsicFS_closed64 : PcClosed K16 lsicFS [234] :=
  lsicFS_iv ▸ ivClosed_at K16 222 13 [234] kSize16 (by omega) (by decide)


theorem shipped64_no_stg32p_lt : ∀ q : Nat, q < 274 → noWide K16[q]? = true := by
  have h : K16.toList.all (fun i => noWide (some i)) = true := by decide
  intro q hq
  have hlt : q < K16.size := by rw [kSize16]; exact hq
  rw [getElem?_pos K16 q hlt]
  exact List.all_eq_true.mp h K16[q] (by simpa using Array.getElem_mem hlt)

theorem shipped64_no_stg32p : ∀ q : Nat, noWide K16[q]? = true := by
  intro q
  rcases Nat.lt_or_ge q 274 with h | h
  · exact shipped64_no_stg32p_lt q h
  · rw [Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)]; rfl

theorem writes_at_site3264 (st : SState) (j : Nat) (h : WritesAct K16 st j) :
    ∃ r : String, (st.pc, r) ∈ storeSites K16 ∧ ∃ l : Lane,
      ActiveAt K16 st l ∧ (st.regs r l).toNat = j := by
  rw [WritesAct] at h
  have hno := shipped64_no_stg32p st.pc
  cases hp : K16[st.pc]? with
  | none => rw [hp] at h; exact absurd h not_false
  | some i =>
      rw [hp] at h
      rw [hp] at hno
      have site : ∀ (addr : String),
          (fun i => match i with
            | SInstr.stg addr _ => some addr
            | SInstr.stgp _ addr _ => some addr
            | SInstr.stg32p _ addr _ => some addr
            | _ => none) i = some addr → (st.pc, addr) ∈ storeSites K16 := by
        intro addr hf
        have : K16.toList[st.pc]? = some i := by simpa using hp
        simpa using mem_siteAux _ K16.toList 0 st.pc i addr this hf
      cases i with
      | stg addr s =>
          obtain ⟨l, hl⟩ := h
          exact ⟨addr, site addr rfl, l, by rw [ActiveAt, hp]; trivial, hl⟩
      | stgp q addr s =>
          obtain ⟨l, hq, hl⟩ := h
          exact ⟨addr, site addr rfl, l, by rw [ActiveAt, hp]; exact hq, hl⟩
      | stg32p q addr s => rw [noWide] at hno; exact absurd hno (by decide)
      | _ => exact absurd h not_false

theorem kernelConfined_of_regConfined3264 (inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (h : RegConfined 16 inPtr outPtr gm smemB) :
    Lz4Interleave.KernelConfined 16 inPtr outPtr gm smemB where
  writes := by
    intro w k j hw
    obtain ⟨r, hsite, l, hact, hl⟩ := writes_at_site3264 _ j hw
    rw [← hl]
    exact h.stores w k r hsite l hact
  reads := by
    intro w w' k j _hne hr hreg
    obtain ⟨r, off, hsite, l, hl⟩ := reads_at_site _ _ j hr
    have hoff : off ≤ 3 := (load_regs3264 _ hsite).2
    have hlt := h.loads w k r off hsite l
    rw [hl] at hlt
    obtain ⟨hge, _⟩ := hreg
    have : outPtr ≤ outPtr + w'.val * (WP.mk 16).outStride := Nat.le_add_right _ _
    omega




theorem lsic_frame64 (l : Lane) (B : Nat) (st : SState) (q' : Nat)
    (hpc' : (sstep K16 st).pc = q')
    (hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litExtraF" ∨ r = "lsicC" →
      (sstep K16 st).regs r j = st.regs r j)
    (hrem : ∀ x : Nat, lsicRem q' x ≤ lsicRem st.pc x)
    (h3' : (q' = 222 ∨ q' = 223 ∨ q' = 230) → (st.pc = 222 ∨ st.pc = 223 ∨ st.pc = 230))
    (h4' : 224 ≤ q' → q' ≤ 228 → 224 ≤ st.pc ∧ st.pc ≤ 228)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litExtraF" l (Or.inr (Or.inl rfl))]
    exact Nat.le_trans (Nat.add_le_add_left (hrem _) _) h1
  · rw [hfr "litExtraF" l (Or.inr (Or.inl rfl)), hfr "litExtraF" 0 (Or.inr (Or.inl rfl))]
    exact h2
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr rfl)), hfr "litExtraF" 0 (Or.inr (Or.inl rfl))]
    exact h3 (h3' hq)
  · rw [hpc']; intro ha hb
    rw [hfr "litExtraF" l (Or.inr (Or.inl rfl))]
    exact h4 (h4' ha hb).1 (h4' ha hb).2

theorem lsic_at22264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 222)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Lh18") := by rw [he]; decide
  have hstep : sstep K16 st = st.setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsic_frame64 l B st 223 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at22464 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 224)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.mov "c255" (SArg.imm 255)) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "c255" (fun l => st.get l (SArg.imm 255))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsic_frame64 l B st 225 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at22564 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 225)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin .add "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "sbAddr" (fun l => SOp.add.run (st.regs "outBase" l) (st.get l (SArg.reg "op")))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsic_frame64 l B st 226 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at22664 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 226)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "c255") := by rw [he]; decide
  have hstep : sstep K16 st = { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs "sbAddr") (st.regs "c255"), pc := st.pc + 1 } := by rw [sstep, hp]; rfl
  refine lsic_frame64 l B st 227 (by rw [hstep, he])
    (fun r j _ => by rw [hstep])
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at23164 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 231)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Lx19") := by rw [he]; decide
  have hstep : sstep K16 st = st.setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsic_frame64 l B st 232 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at23264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 232)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin .add "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "sbAddr" (fun l => SOp.add.run (st.regs "outBase" l) (st.get l (SArg.reg "op")))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsic_frame64 l B st 233 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at23364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 233)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "litExtraF") := by rw [he]; decide
  have hstep : sstep K16 st = { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs "sbAddr") (st.regs "litExtraF"), pc := st.pc + 1 } := by rw [sstep, hp]; rfl
  refine lsic_frame64 l B st 234 (by rw [hstep, he])
    (fun r j _ => by rw [hstep])
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at23064 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 230)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bra "Lh18") := by rw [he]; decide
  have hstep : sstep K16 st = st.setPc (sfindLabel K16 "Lh18") := by rw [sstep, hp]; rfl
  refine lsic_frame64 l B st 222 (by rw [hstep]; show sfindLabel K16 "Lh18" = 222; decide)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h


theorem lsic_at22364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 223)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.braifnot "lsicC" "Lx19") := by rw [he]; decide
  have hstep : sstep K16 st
      = st.setPc (if st.regs "lsicC" 0 == 1 then st.pc + 1 else sfindLabel K16 "Lx19") := by
    rw [sstep, hp]; rfl
  have hlbl : sfindLabel K16 "Lx19" = 231 := by decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hfr : ∀ (r : String) (j : Lane), (sstep K16 st).regs r j = st.regs r j := by
    intro r j; rw [hstep]; rfl
  rw [he] at h1
  by_cases hg : (st.regs "lsicC" 0 == 1) = true
  · have hpc' : (sstep K16 st).pc = 224 := by rw [hstep, he, if_pos hg]; rfl
    have hlx : 255 ≤ (st.regs "litExtraF" l).toNat := by
      rw [h2]; exact (h3 (Or.inr (Or.inl he))).mp hg
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr, hfr]
      have hr : lsicRem 224 ((st.regs "litExtraF" l).toNat)
          = lsicRem 223 ((st.regs "litExtraF" l).toNat) := rfl
      omega
    · rw [hfr, hfr]; exact h2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc', hfr]; intro _ _; exact hlx
  · have hpc' : (sstep K16 st).pc = 231 := by rw [hstep, if_neg hg, hlbl]; rfl
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr, hfr]
      have hr223 : lsicRem 223 ((st.regs "litExtraF" l).toNat)
          = (st.regs "litExtraF" l).toNat / 255 + 1 := rfl
      have hr231 : lsicRem 231 ((st.regs "litExtraF" l).toNat) = 1 := rfl
      omega
    · rw [hfr, hfr]; exact h2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro _ hb; omega

theorem lsic_at22764 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 227)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin .add "op" "op" (SArg.imm 1)) := by rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K16 st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 228 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K16 st).regs "op" j = st.regs "op" j + 1 := by
    intro j; rw [hstep]; rfl
  have hlx' : ∀ j : Lane, (sstep K16 st).regs "litExtraF" j = st.regs "litExtraF" j := by
    intro j; rw [hstep]; rfl
  have hlc' : ∀ j : Lane, (sstep K16 st).regs "lsicC" j = st.regs "lsicC" j := by
    intro j; rw [hstep]; rfl
  have hlx : 255 ≤ (st.regs "litExtraF" l).toNat := h4 (by omega) (by omega)
  rw [he] at h1
  have hr227 : lsicRem 227 ((st.regs "litExtraF" l).toNat)
      = (st.regs "litExtraF" l).toNat / 255 + 1 := rfl
  have hr228 : lsicRem 228 ((st.regs "litExtraF" l).toNat)
      = (st.regs "litExtraF" l).toNat / 255 := rfl
  have hopN : ((st.regs "op" l) + 1).toNat = (st.regs "op" l).toNat + 1 := by
    have hb := (st.regs "op" l).toNat_lt
    have hle : (st.regs "op" l).toNat + 1 ≤ B := by omega
    have hL := (toNat_add_ofNat_of_lt (st.regs "op" l) 1 (by omega)).1
    rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
    omega
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlx' l, hopN]; omega
  · rw [hlx' l, hlx' 0]; exact h2
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc', hlx' l]; intro _ _; exact hlx

theorem lsic_at22864 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 228)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin .sub "litExtraF" "litExtraF" (SArg.imm 255)) := by
    rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K16 st = (st.setReg "litExtraF"
      (fun l => SOp.sub.run (st.regs "litExtraF" l) (st.get l (SArg.imm 255)))).setPc
      (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hlx : 255 ≤ (st.regs "litExtraF" l).toNat := h4 (by omega) (by omega)
  have hpc' : (sstep K16 st).pc = 229 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K16 st).regs "op" j = st.regs "op" j := by
    intro j; rw [hstep]; rfl
  have hlxs : ∀ j : Lane,
      (sstep K16 st).regs "litExtraF" j = st.regs "litExtraF" j - UInt64.ofNat 255 := by
    intro j; rw [hstep]; rfl
  have hsubN : ∀ a : UInt64, 255 ≤ a.toNat → (a - UInt64.ofNat 255).toNat = a.toNat - 255 := by
    intro a ha
    rw [UInt64.toNat_sub, show ((UInt64.ofNat 255).toNat) = 255 from by decide,
      show 2 ^ 64 - 255 + a.toNat = 2 ^ 64 + (a.toNat - 255) from by
        have := a.toNat_lt; omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by have := a.toNat_lt; omega)]
  rw [he] at h1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlxs l, hsubN _ hlx]
    have hr228 : lsicRem 228 ((st.regs "litExtraF" l).toNat)
        = (st.regs "litExtraF" l).toNat / 255 := rfl
    have hr229 : lsicRem 229 ((st.regs "litExtraF" l).toNat - 255)
        = ((st.regs "litExtraF" l).toNat - 255) / 255 + 1 := rfl
    have hdiv := Nat.div_eq_sub_div (Nat.zero_lt_succ 254) hlx
    omega
  · rw [hlxs l, hlxs 0, h2]
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _ hb; omega

theorem lsic_at22964 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 229)
    (h : LsicInv l B st) : LsicInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.setp .ge "lsicC" "litExtraF" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K16 st = (st.setReg "lsicC"
      (fun l => if SCmp.ge.run (st.regs "litExtraF" l) (st.get l (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 230 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K16 st).regs "op" j = st.regs "op" j := by
    intro j; rw [hstep]; rfl
  have hlx' : ∀ j : Lane, (sstep K16 st).regs "litExtraF" j = st.regs "litExtraF" j := by
    intro j; rw [hstep]; rfl
  have hlc' : (sstep K16 st).regs "lsicC" 0
      = (if SCmp.ge.run (st.regs "litExtraF" 0) (UInt64.ofNat 255) then 1 else 0) := by
    rw [hstep]; rfl
  rw [he] at h1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlx' l]
    have hr229 : lsicRem 229 ((st.regs "litExtraF" l).toNat)
        = (st.regs "litExtraF" l).toNat / 255 + 1 := rfl
    have hr230 : lsicRem 230 ((st.regs "litExtraF" l).toNat)
        = (st.regs "litExtraF" l).toNat / 255 + 1 := rfl
    omega
  · rw [hlx' l, hlx' 0]; exact h2
  · rw [hpc', hlc', hlx' 0]; intro _
    have hiff : SCmp.ge.run (st.regs "litExtraF" 0) (UInt64.ofNat 255) = true
        ↔ 255 ≤ (st.regs "litExtraF" 0).toNat := by
      simp only [SCmp.run, decide_eq_true_eq, UInt64.le_iff_toNat_le,
        show ((UInt64.ofNat 255).toNat) = 255 from by decide]
    by_cases hc : SCmp.ge.run (st.regs "litExtraF" 0) (UInt64.ofNat 255) = true
    · rw [if_pos hc]
      exact ⟨fun _ => hiff.mp hc, fun _ => rfl⟩
    · rw [if_neg hc]
      constructor
      · intro hcon; exact absurd hcon (by decide)
      · intro hn; exact absurd (hiff.mpr hn) hc
  · rw [hpc']; intro _ hb; omega

theorem lsicFS_hstep64 (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (hs : st.pc ∈ lsicFS) (hex : st.pc ∉ [234]) (h : LsicInv l B st) :
    LsicInv l B (sstep K16 st) := by
  simp only [lsicFS, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with e | e | e | e | e | e | e | e | e | e | e | e | e
  · exact lsic_at22264 l B st e h
  · exact lsic_at22364 l B st e h
  · exact lsic_at22464 l B st e h
  · exact lsic_at22564 l B st e h
  · exact lsic_at22664 l B st e h
  · exact lsic_at22764 l B st hB e h
  · exact lsic_at22864 l B st e h
  · exact lsic_at22964 l B st e h
  · exact lsic_at23064 l B st e h
  · exact lsic_at23164 l B st e h
  · exact lsic_at23264 l B st e h
  · exact lsic_at23364 l B st e h
  · exact absurd (by simp [e]) hex

theorem lsic_op_lt64 (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (h0 : st.pc = 222) (h : LsicInv l B st) (k : Nat)
    (hne : ∀ j, j < k → (siter K16 j st).pc ∉ [234])
    (hq : (siter K16 k st).pc = 226 ∨ (siter K16 k st).pc = 233) :
    ((siter K16 k st).regs "op" l).toNat < B :=
  lsicInv_op_le l B _
    (inv_on K16 (LsicInv l B) lsicFS [234] lsicFS_closed64
      (fun s hsm hexs hh => lsicFS_hstep64 l B hB s hsm hexs hh) st (by rw [h0]; decide) h k hne) hq

theorem lsicLS_closed64 : PcClosed K16 lsicLS [148] :=
  lsicLS_iv ▸ ivClosed_at K16 136 13 [148] kSize16 (by omega) (by decide)

theorem lsicL_frame64 (l : Lane) (B : Nat) (st : SState) (q' : Nat)
    (hpc' : (sstep K16 st).pc = q')
    (hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litExtra" ∨ r = "lsicC" →
      (sstep K16 st).regs r j = st.regs r j)
    (hrem : ∀ x : Nat, lsicRemL q' x ≤ lsicRemL st.pc x)
    (h3' : (q' = 136 ∨ q' = 137 ∨ q' = 144) → (st.pc = 136 ∨ st.pc = 137 ∨ st.pc = 144))
    (h4' : 138 ≤ q' → q' ≤ 142 → 138 ≤ st.pc ∧ st.pc ≤ 142)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litExtra" l (Or.inr (Or.inl rfl))]
    exact Nat.le_trans (Nat.add_le_add_left (hrem _) _) h1
  · rw [hfr "litExtra" l (Or.inr (Or.inl rfl)), hfr "litExtra" 0 (Or.inr (Or.inl rfl))]
    exact h2
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr rfl)), hfr "litExtra" 0 (Or.inr (Or.inl rfl))]
    exact h3 (h3' hq)
  · rw [hpc']; intro ha hb
    rw [hfr "litExtra" l (Or.inr (Or.inl rfl))]
    exact h4 (h4' ha hb).1 (h4' ha hb).2

theorem lsicL_at22264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 136)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Lh8") := by rw [he]; decide
  have hstep : sstep K16 st = st.setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicL_frame64 l B st 137 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at22464 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 138)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.mov "c255" (SArg.imm 255)) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "c255" (fun l => st.get l (SArg.imm 255))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicL_frame64 l B st 139 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at22564 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 139)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin .add "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "sbAddr" (fun l => SOp.add.run (st.regs "outBase" l) (st.get l (SArg.reg "op")))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicL_frame64 l B st 140 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at22664 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 140)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "c255") := by rw [he]; decide
  have hstep : sstep K16 st = { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs "sbAddr") (st.regs "c255"), pc := st.pc + 1 } := by rw [sstep, hp]; rfl
  refine lsicL_frame64 l B st 141 (by rw [hstep, he])
    (fun r j _ => by rw [hstep])
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at23164 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 145)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Lx9") := by rw [he]; decide
  have hstep : sstep K16 st = st.setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicL_frame64 l B st 146 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at23264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 146)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin .add "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "sbAddr" (fun l => SOp.add.run (st.regs "outBase" l) (st.get l (SArg.reg "op")))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicL_frame64 l B st 147 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at23364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 147)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "litExtra") := by rw [he]; decide
  have hstep : sstep K16 st = { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs "sbAddr") (st.regs "litExtra"), pc := st.pc + 1 } := by rw [sstep, hp]; rfl
  refine lsicL_frame64 l B st 148 (by rw [hstep, he])
    (fun r j _ => by rw [hstep])
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at23064 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 144)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bra "Lh8") := by rw [he]; decide
  have hstep : sstep K16 st = st.setPc (sfindLabel K16 "Lh8") := by rw [sstep, hp]; rfl
  refine lsicL_frame64 l B st 136 (by rw [hstep]; show sfindLabel K16 "Lh8" = 136; decide)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h


theorem lsicL_at22364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 137)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.braifnot "lsicC" "Lx9") := by rw [he]; decide
  have hstep : sstep K16 st
      = st.setPc (if st.regs "lsicC" 0 == 1 then st.pc + 1 else sfindLabel K16 "Lx9") := by
    rw [sstep, hp]; rfl
  have hlbl : sfindLabel K16 "Lx9" = 145 := by decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hfr : ∀ (r : String) (j : Lane), (sstep K16 st).regs r j = st.regs r j := by
    intro r j; rw [hstep]; rfl
  rw [he] at h1
  by_cases hg : (st.regs "lsicC" 0 == 1) = true
  · have hpc' : (sstep K16 st).pc = 138 := by rw [hstep, he, if_pos hg]; rfl
    have hlx : 255 ≤ (st.regs "litExtra" l).toNat := by
      rw [h2]; exact (h3 (Or.inr (Or.inl he))).mp hg
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr, hfr]
      have hr : lsicRemL 138 ((st.regs "litExtra" l).toNat)
          = lsicRemL 137 ((st.regs "litExtra" l).toNat) := rfl
      omega
    · rw [hfr, hfr]; exact h2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc', hfr]; intro _ _; exact hlx
  · have hpc' : (sstep K16 st).pc = 145 := by rw [hstep, if_neg hg, hlbl]; rfl
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr, hfr]
      have hr223 : lsicRemL 137 ((st.regs "litExtra" l).toNat)
          = (st.regs "litExtra" l).toNat / 255 + 1 := rfl
      have hr231 : lsicRemL 145 ((st.regs "litExtra" l).toNat) = 1 := rfl
      omega
    · rw [hfr, hfr]; exact h2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro _ hb; omega

theorem lsicL_at22764 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 141)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin .add "op" "op" (SArg.imm 1)) := by rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K16 st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 142 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K16 st).regs "op" j = st.regs "op" j + 1 := by
    intro j; rw [hstep]; rfl
  have hlx' : ∀ j : Lane, (sstep K16 st).regs "litExtra" j = st.regs "litExtra" j := by
    intro j; rw [hstep]; rfl
  have hlc' : ∀ j : Lane, (sstep K16 st).regs "lsicC" j = st.regs "lsicC" j := by
    intro j; rw [hstep]; rfl
  have hlx : 255 ≤ (st.regs "litExtra" l).toNat := h4 (by omega) (by omega)
  rw [he] at h1
  have hr227 : lsicRemL 141 ((st.regs "litExtra" l).toNat)
      = (st.regs "litExtra" l).toNat / 255 + 1 := rfl
  have hr228 : lsicRemL 142 ((st.regs "litExtra" l).toNat)
      = (st.regs "litExtra" l).toNat / 255 := rfl
  have hopN : ((st.regs "op" l) + 1).toNat = (st.regs "op" l).toNat + 1 := by
    have hb := (st.regs "op" l).toNat_lt
    have hle : (st.regs "op" l).toNat + 1 ≤ B := by omega
    have hL := (toNat_add_ofNat_of_lt (st.regs "op" l) 1 (by omega)).1
    rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
    omega
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlx' l, hopN]; omega
  · rw [hlx' l, hlx' 0]; exact h2
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc', hlx' l]; intro _ _; exact hlx

theorem lsicL_at22864 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 142)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin .sub "litExtra" "litExtra" (SArg.imm 255)) := by
    rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K16 st = (st.setReg "litExtra"
      (fun l => SOp.sub.run (st.regs "litExtra" l) (st.get l (SArg.imm 255)))).setPc
      (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hlx : 255 ≤ (st.regs "litExtra" l).toNat := h4 (by omega) (by omega)
  have hpc' : (sstep K16 st).pc = 143 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K16 st).regs "op" j = st.regs "op" j := by
    intro j; rw [hstep]; rfl
  have hlxs : ∀ j : Lane,
      (sstep K16 st).regs "litExtra" j = st.regs "litExtra" j - UInt64.ofNat 255 := by
    intro j; rw [hstep]; rfl
  have hsubN : ∀ a : UInt64, 255 ≤ a.toNat → (a - UInt64.ofNat 255).toNat = a.toNat - 255 := by
    intro a ha
    rw [UInt64.toNat_sub, show ((UInt64.ofNat 255).toNat) = 255 from by decide,
      show 2 ^ 64 - 255 + a.toNat = 2 ^ 64 + (a.toNat - 255) from by
        have := a.toNat_lt; omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by have := a.toNat_lt; omega)]
  rw [he] at h1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlxs l, hsubN _ hlx]
    have hr228 : lsicRemL 142 ((st.regs "litExtra" l).toNat)
        = (st.regs "litExtra" l).toNat / 255 := rfl
    have hr229 : lsicRemL 143 ((st.regs "litExtra" l).toNat - 255)
        = ((st.regs "litExtra" l).toNat - 255) / 255 + 1 := rfl
    have hdiv := Nat.div_eq_sub_div (Nat.zero_lt_succ 254) hlx
    omega
  · rw [hlxs l, hlxs 0, h2]
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _ hb; omega

theorem lsicL_at22964 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 143)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.setp .ge "lsicC" "litExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K16 st = (st.setReg "lsicC"
      (fun l => if SCmp.ge.run (st.regs "litExtra" l) (st.get l (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 144 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K16 st).regs "op" j = st.regs "op" j := by
    intro j; rw [hstep]; rfl
  have hlx' : ∀ j : Lane, (sstep K16 st).regs "litExtra" j = st.regs "litExtra" j := by
    intro j; rw [hstep]; rfl
  have hlc' : (sstep K16 st).regs "lsicC" 0
      = (if SCmp.ge.run (st.regs "litExtra" 0) (UInt64.ofNat 255) then 1 else 0) := by
    rw [hstep]; rfl
  rw [he] at h1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlx' l]
    have hr229 : lsicRemL 143 ((st.regs "litExtra" l).toNat)
        = (st.regs "litExtra" l).toNat / 255 + 1 := rfl
    have hr230 : lsicRemL 144 ((st.regs "litExtra" l).toNat)
        = (st.regs "litExtra" l).toNat / 255 + 1 := rfl
    omega
  · rw [hlx' l, hlx' 0]; exact h2
  · rw [hpc', hlc', hlx' 0]; intro _
    have hiff : SCmp.ge.run (st.regs "litExtra" 0) (UInt64.ofNat 255) = true
        ↔ 255 ≤ (st.regs "litExtra" 0).toNat := by
      simp only [SCmp.run, decide_eq_true_eq, UInt64.le_iff_toNat_le,
        show ((UInt64.ofNat 255).toNat) = 255 from by decide]
    by_cases hc : SCmp.ge.run (st.regs "litExtra" 0) (UInt64.ofNat 255) = true
    · rw [if_pos hc]
      exact ⟨fun _ => hiff.mp hc, fun _ => rfl⟩
    · rw [if_neg hc]
      constructor
      · intro hcon; exact absurd hcon (by decide)
      · intro hn; exact absurd (hiff.mpr hn) hc
  · rw [hpc']; intro _ hb; omega

theorem lsicLS_hstep64 (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (hs : st.pc ∈ lsicLS) (hex : st.pc ∉ [148]) (h : LsicInvL l B st) :
    LsicInvL l B (sstep K16 st) := by
  simp only [lsicLS, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with e | e | e | e | e | e | e | e | e | e | e | e | e
  · exact lsicL_at22264 l B st e h
  · exact lsicL_at22364 l B st e h
  · exact lsicL_at22464 l B st e h
  · exact lsicL_at22564 l B st e h
  · exact lsicL_at22664 l B st e h
  · exact lsicL_at22764 l B st hB e h
  · exact lsicL_at22864 l B st e h
  · exact lsicL_at22964 l B st e h
  · exact lsicL_at23064 l B st e h
  · exact lsicL_at23164 l B st e h
  · exact lsicL_at23264 l B st e h
  · exact lsicL_at23364 l B st e h
  · exact absurd (by simp [e]) hex

theorem lsicL_op_lt64 (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (h0 : st.pc = 136) (h : LsicInvL l B st) (k : Nat)
    (hne : ∀ j, j < k → (siter K16 j st).pc ∉ [148])
    (hq : (siter K16 k st).pc = 140 ∨ (siter K16 k st).pc = 147) :
    ((siter K16 k st).regs "op" l).toNat < B :=
  lsicLInv_op_le l B _
    (inv_on K16 (LsicInvL l B) lsicLS [148] lsicLS_closed64
      (fun s hsm hexs hh => lsicLS_hstep64 l B hB s hsm hexs hh) st (by rw [h0]; decide) h k hne) hq


theorem lsicMS_closed64 : PcClosed K16 lsicMS [196] :=
  lsicMS_iv ▸ ivClosed_at K16 184 13 [196] kSize16 (by omega) (by decide)

theorem lsicM_frame64 (l : Lane) (B : Nat) (st : SState) (q' : Nat)
    (hpc' : (sstep K16 st).pc = q')
    (hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "matExtra" ∨ r = "lsicC" →
      (sstep K16 st).regs r j = st.regs r j)
    (hrem : ∀ x : Nat, lsicRemM q' x ≤ lsicRemM st.pc x)
    (h3' : (q' = 184 ∨ q' = 185 ∨ q' = 192) → (st.pc = 184 ∨ st.pc = 185 ∨ st.pc = 192))
    (h4' : 186 ≤ q' → q' ≤ 190 → 186 ≤ st.pc ∧ st.pc ≤ 190)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "matExtra" l (Or.inr (Or.inl rfl))]
    exact Nat.le_trans (Nat.add_le_add_left (hrem _) _) h1
  · rw [hfr "matExtra" l (Or.inr (Or.inl rfl)), hfr "matExtra" 0 (Or.inr (Or.inl rfl))]
    exact h2
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr rfl)), hfr "matExtra" 0 (Or.inr (Or.inl rfl))]
    exact h3 (h3' hq)
  · rw [hpc']; intro ha hb
    rw [hfr "matExtra" l (Or.inr (Or.inl rfl))]
    exact h4 (h4' ha hb).1 (h4' ha hb).2

theorem lsicM_at22264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 184)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Lh14") := by rw [he]; decide
  have hstep : sstep K16 st = st.setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicM_frame64 l B st 185 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at22464 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 186)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.mov "c255" (SArg.imm 255)) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "c255" (fun l => st.get l (SArg.imm 255))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicM_frame64 l B st 187 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at22564 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 187)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin .add "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "sbAddr" (fun l => SOp.add.run (st.regs "outBase" l) (st.get l (SArg.reg "op")))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicM_frame64 l B st 188 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at22664 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 188)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "c255") := by rw [he]; decide
  have hstep : sstep K16 st = { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs "sbAddr") (st.regs "c255"), pc := st.pc + 1 } := by rw [sstep, hp]; rfl
  refine lsicM_frame64 l B st 189 (by rw [hstep, he])
    (fun r j _ => by rw [hstep])
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at23164 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 193)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Lx15") := by rw [he]; decide
  have hstep : sstep K16 st = st.setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicM_frame64 l B st 194 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at23264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 194)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin .add "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "sbAddr" (fun l => SOp.add.run (st.regs "outBase" l) (st.get l (SArg.reg "op")))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicM_frame64 l B st 195 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at23364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 195)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "matExtra") := by rw [he]; decide
  have hstep : sstep K16 st = { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs "sbAddr") (st.regs "matExtra"), pc := st.pc + 1 } := by rw [sstep, hp]; rfl
  refine lsicM_frame64 l B st 196 (by rw [hstep, he])
    (fun r j _ => by rw [hstep])
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at23064 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 192)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bra "Lh14") := by rw [he]; decide
  have hstep : sstep K16 st = st.setPc (sfindLabel K16 "Lh14") := by rw [sstep, hp]; rfl
  refine lsicM_frame64 l B st 184 (by rw [hstep]; show sfindLabel K16 "Lh14" = 184; decide)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h


theorem lsicM_at22364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 185)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.braifnot "lsicC" "Lx15") := by rw [he]; decide
  have hstep : sstep K16 st
      = st.setPc (if st.regs "lsicC" 0 == 1 then st.pc + 1 else sfindLabel K16 "Lx15") := by
    rw [sstep, hp]; rfl
  have hlbl : sfindLabel K16 "Lx15" = 193 := by decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hfr : ∀ (r : String) (j : Lane), (sstep K16 st).regs r j = st.regs r j := by
    intro r j; rw [hstep]; rfl
  rw [he] at h1
  by_cases hg : (st.regs "lsicC" 0 == 1) = true
  · have hpc' : (sstep K16 st).pc = 186 := by rw [hstep, he, if_pos hg]; rfl
    have hlx : 255 ≤ (st.regs "matExtra" l).toNat := by
      rw [h2]; exact (h3 (Or.inr (Or.inl he))).mp hg
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr, hfr]
      have hr : lsicRemM 186 ((st.regs "matExtra" l).toNat)
          = lsicRemM 185 ((st.regs "matExtra" l).toNat) := rfl
      omega
    · rw [hfr, hfr]; exact h2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc', hfr]; intro _ _; exact hlx
  · have hpc' : (sstep K16 st).pc = 193 := by rw [hstep, if_neg hg, hlbl]; rfl
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr, hfr]
      have hr223 : lsicRemM 185 ((st.regs "matExtra" l).toNat)
          = (st.regs "matExtra" l).toNat / 255 + 1 := rfl
      have hr231 : lsicRemM 193 ((st.regs "matExtra" l).toNat) = 1 := rfl
      omega
    · rw [hfr, hfr]; exact h2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro _ hb; omega

theorem lsicM_at22764 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 189)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin .add "op" "op" (SArg.imm 1)) := by rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K16 st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 190 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K16 st).regs "op" j = st.regs "op" j + 1 := by
    intro j; rw [hstep]; rfl
  have hlx' : ∀ j : Lane, (sstep K16 st).regs "matExtra" j = st.regs "matExtra" j := by
    intro j; rw [hstep]; rfl
  have hlc' : ∀ j : Lane, (sstep K16 st).regs "lsicC" j = st.regs "lsicC" j := by
    intro j; rw [hstep]; rfl
  have hlx : 255 ≤ (st.regs "matExtra" l).toNat := h4 (by omega) (by omega)
  rw [he] at h1
  have hr227 : lsicRemM 189 ((st.regs "matExtra" l).toNat)
      = (st.regs "matExtra" l).toNat / 255 + 1 := rfl
  have hr228 : lsicRemM 190 ((st.regs "matExtra" l).toNat)
      = (st.regs "matExtra" l).toNat / 255 := rfl
  have hopN : ((st.regs "op" l) + 1).toNat = (st.regs "op" l).toNat + 1 := by
    have hb := (st.regs "op" l).toNat_lt
    have hle : (st.regs "op" l).toNat + 1 ≤ B := by omega
    have hL := (toNat_add_ofNat_of_lt (st.regs "op" l) 1 (by omega)).1
    rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
    omega
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlx' l, hopN]; omega
  · rw [hlx' l, hlx' 0]; exact h2
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc', hlx' l]; intro _ _; exact hlx

theorem lsicM_at22864 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 190)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin .sub "matExtra" "matExtra" (SArg.imm 255)) := by
    rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K16 st = (st.setReg "matExtra"
      (fun l => SOp.sub.run (st.regs "matExtra" l) (st.get l (SArg.imm 255)))).setPc
      (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hlx : 255 ≤ (st.regs "matExtra" l).toNat := h4 (by omega) (by omega)
  have hpc' : (sstep K16 st).pc = 191 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K16 st).regs "op" j = st.regs "op" j := by
    intro j; rw [hstep]; rfl
  have hlxs : ∀ j : Lane,
      (sstep K16 st).regs "matExtra" j = st.regs "matExtra" j - UInt64.ofNat 255 := by
    intro j; rw [hstep]; rfl
  have hsubN : ∀ a : UInt64, 255 ≤ a.toNat → (a - UInt64.ofNat 255).toNat = a.toNat - 255 := by
    intro a ha
    rw [UInt64.toNat_sub, show ((UInt64.ofNat 255).toNat) = 255 from by decide,
      show 2 ^ 64 - 255 + a.toNat = 2 ^ 64 + (a.toNat - 255) from by
        have := a.toNat_lt; omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by have := a.toNat_lt; omega)]
  rw [he] at h1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlxs l, hsubN _ hlx]
    have hr228 : lsicRemM 190 ((st.regs "matExtra" l).toNat)
        = (st.regs "matExtra" l).toNat / 255 := rfl
    have hr229 : lsicRemM 191 ((st.regs "matExtra" l).toNat - 255)
        = ((st.regs "matExtra" l).toNat - 255) / 255 + 1 := rfl
    have hdiv := Nat.div_eq_sub_div (Nat.zero_lt_succ 254) hlx
    omega
  · rw [hlxs l, hlxs 0, h2]
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _ hb; omega

theorem lsicM_at22964 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 191)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.setp .ge "lsicC" "matExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K16 st = (st.setReg "lsicC"
      (fun l => if SCmp.ge.run (st.regs "matExtra" l) (st.get l (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 192 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K16 st).regs "op" j = st.regs "op" j := by
    intro j; rw [hstep]; rfl
  have hlx' : ∀ j : Lane, (sstep K16 st).regs "matExtra" j = st.regs "matExtra" j := by
    intro j; rw [hstep]; rfl
  have hlc' : (sstep K16 st).regs "lsicC" 0
      = (if SCmp.ge.run (st.regs "matExtra" 0) (UInt64.ofNat 255) then 1 else 0) := by
    rw [hstep]; rfl
  rw [he] at h1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlx' l]
    have hr229 : lsicRemM 191 ((st.regs "matExtra" l).toNat)
        = (st.regs "matExtra" l).toNat / 255 + 1 := rfl
    have hr230 : lsicRemM 192 ((st.regs "matExtra" l).toNat)
        = (st.regs "matExtra" l).toNat / 255 + 1 := rfl
    omega
  · rw [hlx' l, hlx' 0]; exact h2
  · rw [hpc', hlc', hlx' 0]; intro _
    have hiff : SCmp.ge.run (st.regs "matExtra" 0) (UInt64.ofNat 255) = true
        ↔ 255 ≤ (st.regs "matExtra" 0).toNat := by
      simp only [SCmp.run, decide_eq_true_eq, UInt64.le_iff_toNat_le,
        show ((UInt64.ofNat 255).toNat) = 255 from by decide]
    by_cases hc : SCmp.ge.run (st.regs "matExtra" 0) (UInt64.ofNat 255) = true
    · rw [if_pos hc]
      exact ⟨fun _ => hiff.mp hc, fun _ => rfl⟩
    · rw [if_neg hc]
      constructor
      · intro hcon; exact absurd hcon (by decide)
      · intro hn; exact absurd (hiff.mpr hn) hc
  · rw [hpc']; intro _ hb; omega

theorem lsicMS_hstep64 (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (hs : st.pc ∈ lsicMS) (hex : st.pc ∉ [196]) (h : LsicInvM l B st) :
    LsicInvM l B (sstep K16 st) := by
  simp only [lsicMS, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with e | e | e | e | e | e | e | e | e | e | e | e | e
  · exact lsicM_at22264 l B st e h
  · exact lsicM_at22364 l B st e h
  · exact lsicM_at22464 l B st e h
  · exact lsicM_at22564 l B st e h
  · exact lsicM_at22664 l B st e h
  · exact lsicM_at22764 l B st hB e h
  · exact lsicM_at22864 l B st e h
  · exact lsicM_at22964 l B st e h
  · exact lsicM_at23064 l B st e h
  · exact lsicM_at23164 l B st e h
  · exact lsicM_at23264 l B st e h
  · exact lsicM_at23364 l B st e h
  · exact absurd (by simp [e]) hex

theorem lsicM_op_lt64 (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (h0 : st.pc = 184) (h : LsicInvM l B st) (k : Nat)
    (hne : ∀ j, j < k → (siter K16 j st).pc ∉ [196])
    (hq : (siter K16 k st).pc = 188 ∨ (siter K16 k st).pc = 195) :
    ((siter K16 k st).regs "op" l).toNat < B :=
  lsicMInv_op_le l B _
    (inv_on K16 (LsicInvM l B) lsicMS [196] lsicMS_closed64
      (fun s hsm hexs hh => lsicMS_hstep64 l B hB s hsm hexs hh) st (by rw [h0]; decide) h k hne) hq

theorem bodyPre_closed64 : PcClosed K16 bodyPre [130, 208] := by decide

theorem op_const_to_13064 (st : SState) (h0 : st.pc = 40) (k : Nat)
    (hne : ∀ j, j < k → (siter K16 j st).pc ∉ [130, 208]) :
    (siter K16 k st).regs "op" = st.regs "op" :=
  regs_const_on K16 "op" bodyPre [130, 208] bodyPre_closed64 (by decide) st
    (by rw [h0]; decide) k hne

theorem tokS_closed64 : PcClosed K16 tokS [197, 198] :=
  tokS_iv ▸ ivClosed_at K16 129 70 [197, 198] kSize16 (by omega) (by decide)

theorem tok_frame64 (l : Lane) (B : Nat) (st : SState) (q' : Nat)
    (hpc' : (sstep K16 st).pc = q')
    (hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm"
        ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j)
    (hrem : ∀ a b c d : Nat, tokRem q' a b c d ≤ tokRem st.pc a b c d)
    (h3 : (q' = 136 ∨ q' = 137 ∨ q' = 144) →
      (st.pc = 136 ∨ st.pc = 137 ∨ st.pc = 144))
    (h4 : 138 ≤ q' → q' ≤ 142 → 138 ≤ st.pc ∧ st.pc ≤ 142)
    (h5 : q' = 134 → st.pc = 134)
    (h6 : q' = 133 → st.pc = 133)
    (h7 : (q' = 184 ∨ q' = 185 ∨ q' = 192) →
      (st.pc = 184 ∨ st.pc = 185 ∨ st.pc = 192))
    (h8 : 186 ≤ q' → q' ≤ 190 → 186 ≤ st.pc ∧ st.pc ≤ 190)
    (h9 : q' = 182 → st.pc = 182)
    (h10 : q' = 181 → st.pc = 181)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))),
      hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    exact Nat.le_trans (Nat.add_le_add_left (hrem _ _ _ _) _) c1
  · intro r hr
    have hr8 : r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra"
        ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" := by
      rcases hr with rfl | rfl | rfl | rfl | rfl | rfl <;> simp
    rw [hfr r l hr8, hfr r 0 hr8]; exact c2 r hr
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr "litExtra" 0 (Or.inr (Or.inr (Or.inl rfl)))]; exact c3 (h3 hq)
  · rw [hpc']; intro ha hb; rw [hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl)))]
    exact c4 (h4 ha hb).1 (h4 ha hb).2
  · rw [hpc']; intro hq; rw [hfr "litLen" l (Or.inr (Or.inl rfl))]; exact c5 (h5 hq)
  · rw [hpc']; intro hq
    rw [hfr "pLitBig" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))), hfr "litLen" 0 (Or.inr (Or.inl rfl))]; exact c6 (h6 hq)
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr "matExtra" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]; exact c7 (h7 hq)
  · rw [hpc']; intro ha hb; rw [hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    exact c8 (h8 ha hb).1 (h8 ha hb).2
  · rw [hpc']; intro hq; rw [hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]; exact c9 (h9 hq)
  · rw [hpc']; intro hq
    rw [hfr "pMatBig" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl)))))))), hfr "mlm" 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]; exact c10 (h10 hq)

theorem tok_at12964 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 129)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame64 l B st 130 (by rw [sstep, hp]; show st.pc + 1 = 130; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at13064 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 130)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "tok") := by rw [he]; decide
  refine tok_frame64 l B st 131 (by rw [sstep, hp]; show st.pc + 1 = 131; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at13664 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 136)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Lh8") := by rw [he]; decide
  refine tok_frame64 l B st 137 (by rw [sstep, hp]; show st.pc + 1 = 137; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at13864 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 138)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.mov "c255" (SArg.imm 255)) := by rw [he]; decide
  refine tok_frame64 l B st 139 (by rw [sstep, hp]; show st.pc + 1 = 139; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at13964 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 139)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame64 l B st 140 (by rw [sstep, hp]; show st.pc + 1 = 140; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at14064 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 140)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "c255") := by rw [he]; decide
  refine tok_frame64 l B st 141 (by rw [sstep, hp]; show st.pc + 1 = 141; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at14464 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 144)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bra "Lh8") := by rw [he]; decide
  refine tok_frame64 l B st 136 (by rw [sstep, hp]; show sfindLabel K16 "Lh8" = 136; decide)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at14564 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 145)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Lx9") := by rw [he]; decide
  refine tok_frame64 l B st 146 (by rw [sstep, hp]; show st.pc + 1 = 146; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at14664 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 146)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame64 l B st 147 (by rw [sstep, hp]; show st.pc + 1 = 147; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at14764 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 147)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "litExtra") := by rw [he]; decide
  refine tok_frame64 l B st 148 (by rw [sstep, hp]; show st.pc + 1 = 148; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at14964 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 149)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bra "Ln7") := by rw [he]; decide
  refine tok_frame64 l B st 151 (by rw [sstep, hp]; show sfindLabel K16 "Ln7" = 151; decide)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at15064 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 150)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Le6") := by rw [he]; decide
  refine tok_frame64 l B st 151 (by rw [sstep, hp]; show st.pc + 1 = 151; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at15164 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 151)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Ln7") := by rw [he]; decide
  refine tok_frame64 l B st 152 (by rw [sstep, hp]; show st.pc + 1 = 152; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at15264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 152)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "cpDst" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame64 l B st 153 (by rw [sstep, hp]; show st.pc + 1 = 153; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at15364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 153)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "cpSrc" "inBase" (SArg.reg "litAnchor")) := by rw [he]; decide
  refine tok_frame64 l B st 154 (by rw [sstep, hp]; show st.pc + 1 = 154; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at15464 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 154)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.mov "cpI" (SArg.imm 0)) := by rw [he]; decide
  refine tok_frame64 l B st 155 (by rw [sstep, hp]; show st.pc + 1 = 155; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at15564 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 155)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.setp (.lt) "cpCont" "cpI" (SArg.reg "litLen")) := by rw [he]; decide
  refine tok_frame64 l B st 156 (by rw [sstep, hp]; show st.pc + 1 = 156; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at15664 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 156)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Ch10") := by rw [he]; decide
  refine tok_frame64 l B st 157 (by rw [sstep, hp]; show st.pc + 1 = 157; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at15864 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 158)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.binr (.add) "cpDo" "cpDst" "cpI") := by rw [he]; decide
  refine tok_frame64 l B st 159 (by rw [sstep, hp]; show st.pc + 1 = 159; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at15964 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 159)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.binr (.add) "cpDo" "cpDo" "lane") := by rw [he]; decide
  refine tok_frame64 l B st 160 (by rw [sstep, hp]; show st.pc + 1 = 160; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at16064 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 160)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.binr (.add) "cpSo" "cpSrc" "cpI") := by rw [he]; decide
  refine tok_frame64 l B st 161 (by rw [sstep, hp]; show st.pc + 1 = 161; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at16164 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 161)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.binr (.add) "cpSo" "cpSo" "lane") := by rw [he]; decide
  refine tok_frame64 l B st 162 (by rw [sstep, hp]; show st.pc + 1 = 162; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at16264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 162)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.binr (.add) "cpJ" "cpI" "lane") := by rw [he]; decide
  refine tok_frame64 l B st 163 (by rw [sstep, hp]; show st.pc + 1 = 163; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at16364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 163)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.setp (.lt) "cpP" "cpJ" (SArg.reg "litLen")) := by rw [he]; decide
  refine tok_frame64 l B st 164 (by rw [sstep, hp]; show st.pc + 1 = 164; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h


theorem tok_at16564 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 165)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stgp "cpP" "cpDo" "cpB") := by rw [he]; decide
  refine tok_frame64 l B st 166 (by rw [sstep, hp]; show st.pc + 1 = 166; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at16664 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 166)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "cpI" "cpI" (SArg.imm 32)) := by rw [he]; decide
  refine tok_frame64 l B st 167 (by rw [sstep, hp]; show st.pc + 1 = 167; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at16764 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 167)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.setp (.lt) "cpCont" "cpI" (SArg.reg "litLen")) := by rw [he]; decide
  refine tok_frame64 l B st 168 (by rw [sstep, hp]; show st.pc + 1 = 168; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at16864 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 168)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bra "Ch10") := by rw [he]; decide
  refine tok_frame64 l B st 156 (by rw [sstep, hp]; show sfindLabel K16 "Ch10" = 156; decide)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at16964 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 169)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Cx11") := by rw [he]; decide
  refine tok_frame64 l B st 170 (by rw [sstep, hp]; show st.pc + 1 = 170; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at17164 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 171)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.band) "offLo" "off0" (SArg.imm 255)) := by rw [he]; decide
  refine tok_frame64 l B st 172 (by rw [sstep, hp]; show st.pc + 1 = 172; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at17264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 172)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame64 l B st 173 (by rw [sstep, hp]; show st.pc + 1 = 173; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at17364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 173)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "offLo") := by rw [he]; decide
  refine tok_frame64 l B st 174 (by rw [sstep, hp]; show st.pc + 1 = 174; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at17564 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 175)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.shr) "offHi" "off0" (SArg.imm 8)) := by rw [he]; decide
  refine tok_frame64 l B st 176 (by rw [sstep, hp]; show st.pc + 1 = 176; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at17664 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 176)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.band) "offHi" "offHi" (SArg.imm 255)) := by rw [he]; decide
  refine tok_frame64 l B st 177 (by rw [sstep, hp]; show st.pc + 1 = 177; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at17764 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 177)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame64 l B st 178 (by rw [sstep, hp]; show st.pc + 1 = 178; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at17864 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 178)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "offHi") := by rw [he]; decide
  refine tok_frame64 l B st 179 (by rw [sstep, hp]; show st.pc + 1 = 179; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at18464 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 184)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Lh14") := by rw [he]; decide
  refine tok_frame64 l B st 185 (by rw [sstep, hp]; show st.pc + 1 = 185; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at18664 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 186)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.mov "c255" (SArg.imm 255)) := by rw [he]; decide
  refine tok_frame64 l B st 187 (by rw [sstep, hp]; show st.pc + 1 = 187; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at18764 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 187)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame64 l B st 188 (by rw [sstep, hp]; show st.pc + 1 = 188; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at18864 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 188)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "c255") := by rw [he]; decide
  refine tok_frame64 l B st 189 (by rw [sstep, hp]; show st.pc + 1 = 189; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at19264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 192)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bra "Lh14") := by rw [he]; decide
  refine tok_frame64 l B st 184 (by rw [sstep, hp]; show sfindLabel K16 "Lh14" = 184; decide)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at19364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 193)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.lbl "Lx15") := by rw [he]; decide
  refine tok_frame64 l B st 194 (by rw [sstep, hp]; show st.pc + 1 = 194; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at19464 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 194)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame64 l B st 195 (by rw [sstep, hp]; show st.pc + 1 = 195; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at19564 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 195)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.stg "sbAddr" "matExtra") := by rw [he]; decide
  refine tok_frame64 l B st 196 (by rw [sstep, hp]; show st.pc + 1 = 196; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_op164 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (q' : Nat)
    (hpc' : (sstep K16 st).pc = q')
    (hop' : ∀ j : Lane, (sstep K16 st).regs "op" j = st.regs "op" j + 1)
    (hfr : ∀ (r : String) (j : Lane), r = "litLen" ∨ r = "litExtra" ∨ r = "mlm"
        ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j)
    (hrem : ∀ a b c d : Nat, tokRem q' a b c d + 1 ≤ tokRem st.pc a b c d)
    (h3 : (q' = 136 ∨ q' = 137 ∨ q' = 144) →
      (st.pc = 136 ∨ st.pc = 137 ∨ st.pc = 144))
    (h4 : 138 ≤ q' → q' ≤ 142 → 138 ≤ st.pc ∧ st.pc ≤ 142)
    (h5 : q' = 134 → st.pc = 134) (h6 : q' = 133 → st.pc = 133)
    (h7 : (q' = 184 ∨ q' = 185 ∨ q' = 192) →
      (st.pc = 184 ∨ st.pc = 185 ∨ st.pc = 192))
    (h8 : 186 ≤ q' → q' ≤ 190 → 186 ≤ st.pc ∧ st.pc ≤ 190)
    (h9 : q' = 182 → st.pc = 182) (h10 : q' = 181 → st.pc = 181)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hr := hrem (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat
    (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat
  have hopN : ((st.regs "op" l) + 1).toNat = (st.regs "op" l).toNat + 1 := by
    have hb := (st.regs "op" l).toNat_lt
    have hL := (toNat_add_ofNat_of_lt (st.regs "op" l) 1 (by omega)).1
    rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
    omega
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hopN, hfr "litLen" l (Or.inl rfl), hfr "litExtra" l (Or.inr (Or.inl rfl)),
      hfr "mlm" l (Or.inr (Or.inr (Or.inl rfl))),
      hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
    omega
  · intro r hr2
    rcases hr2 with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hop' l, hop' 0, c2 "op" (Or.inl rfl)]
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]; exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))),
        hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl)))))
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))),
      hfr "litExtra" 0 (Or.inr (Or.inl rfl))]
    exact c3 (h3 hq)
  · rw [hpc']; intro ha hb2; rw [hfr "litExtra" l (Or.inr (Or.inl rfl))]
    exact c4 (h4 ha hb2).1 (h4 ha hb2).2
  · rw [hpc']; intro hq; rw [hfr "litLen" l (Or.inl rfl)]; exact c5 (h5 hq)
  · rw [hpc']; intro hq
    rw [hfr "pLitBig" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))),
      hfr "litLen" 0 (Or.inl rfl)]
    exact c6 (h6 hq)
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))),
      hfr "matExtra" 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
    exact c7 (h7 hq)
  · rw [hpc']; intro ha hb2
    rw [hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
    exact c8 (h8 ha hb2).1 (h8 ha hb2).2
  · rw [hpc']; intro hq; rw [hfr "mlm" l (Or.inr (Or.inr (Or.inl rfl)))]; exact c9 (h9 hq)
  · rw [hpc']; intro hq
    rw [hfr "pMatBig" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl)))))),
      hfr "mlm" 0 (Or.inr (Or.inr (Or.inl rfl)))]
    exact c10 (h10 hq)

theorem tok_at13164 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 131)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op164 l B st hB 132 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop131)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at14164 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 141)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op164 l B st hB 142 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop141)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at14864 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 148)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op164 l B st hB 149 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop148)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at17464 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 174)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op164 l B st hB 175 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop174)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at17964 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 179)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op164 l B st hB 180 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop179)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at18964 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 189)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op164 l B st hB 190 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop189)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at19664 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 196)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K16 st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op164 l B st hB 197 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop196)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at13564 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 135)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.setp (.ge) "lsicC" "litExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K16 st = (st.setReg "lsicC"
      (fun j => if SCmp.ge.run (st.regs "litExtra" j) (st.get j (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 136 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm"
      ∨ r = "matExtra" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hlc : ∀ j : Lane, (sstep K16 st).regs "lsicC" j
      = (if SCmp.ge.run (st.regs "litExtra" j) (UInt64.ofNat 255) then 1 else 0) := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)),
      hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))),
      hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
      hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    have e : tokRem 136 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat
        (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat
      = tokRem 135 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat
        (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]; exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))),
        hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hlc l, hlc 0, c2 "litExtra" (Or.inr (Or.inr (Or.inl rfl)))]
  · rw [hpc']; intro _
    rw [hlc 0, hfr "litExtra" 0 (Or.inr (Or.inr (Or.inl rfl)))]
    exact setp_ge_iff (st.regs "litExtra" 0) 255 (by decide)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at14364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 143)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.setp (.ge) "lsicC" "litExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K16 st = (st.setReg "lsicC"
      (fun j => if SCmp.ge.run (st.regs "litExtra" j) (st.get j (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 144 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hlc : ∀ j : Lane, (sstep K16 st).regs "lsicC" j
      = (if SCmp.ge.run (st.regs "litExtra" j) (UInt64.ofNat 255) then 1 else 0) := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    have e : tokRem 144 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 143 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hlc l, hlc 0, c2 "litExtra" (Or.inr (Or.inr (Or.inl rfl)))]
  · rw [hpc']; intro _
    rw [hlc 0, hfr "litExtra" 0 (Or.inr (Or.inr (Or.inl rfl)))]
    exact setp_ge_iff (st.regs "litExtra" 0) 255 (by decide)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at18364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 183)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.setp (.ge) "lsicC" "matExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K16 st = (st.setReg "lsicC"
      (fun j => if SCmp.ge.run (st.regs "matExtra" j) (st.get j (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 184 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hlc : ∀ j : Lane, (sstep K16 st).regs "lsicC" j
      = (if SCmp.ge.run (st.regs "matExtra" j) (UInt64.ofNat 255) then 1 else 0) := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    have e : tokRem 184 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 183 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hlc l, hlc 0, c2 "matExtra" (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _
    rw [hlc 0, hfr "matExtra" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    exact setp_ge_iff (st.regs "matExtra" 0) 255 (by decide)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at19164 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 191)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.setp (.ge) "lsicC" "matExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K16 st = (st.setReg "lsicC"
      (fun j => if SCmp.ge.run (st.regs "matExtra" j) (st.get j (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 192 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hlc : ∀ j : Lane, (sstep K16 st).regs "lsicC" j
      = (if SCmp.ge.run (st.regs "matExtra" j) (UInt64.ofNat 255) then 1 else 0) := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    have e : tokRem 192 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 191 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hlc l, hlc 0, c2 "matExtra" (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _
    rw [hlc 0, hfr "matExtra" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    exact setp_ge_iff (st.regs "matExtra" 0) 255 (by decide)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at13264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 132)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.setp (.ge) "pLitBig" "litLen" (SArg.imm 15)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K16 st = (st.setReg "pLitBig"
      (fun j => if SCmp.ge.run (st.regs "litLen" j) (st.get j (SArg.imm 15)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 133 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hlc : ∀ j : Lane, (sstep K16 st).regs "pLitBig" j
      = (if SCmp.ge.run (st.regs "litLen" j) (UInt64.ofNat 15) then 1 else 0) := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    have e : tokRem 133 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 132 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _
    rw [hlc 0, hfr "litLen" 0 (Or.inr (Or.inl rfl))]
    exact setp_ge_iff (st.regs "litLen" 0) 15 (by decide)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at18064 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 180)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.setp (.ge) "pMatBig" "mlm" (SArg.imm 15)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K16 st = (st.setReg "pMatBig"
      (fun j => if SCmp.ge.run (st.regs "mlm" j) (st.get j (SArg.imm 15)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 181 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hlc : ∀ j : Lane, (sstep K16 st).regs "pMatBig" j
      = (if SCmp.ge.run (st.regs "mlm" j) (UInt64.ofNat 15) then 1 else 0) := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    have e : tokRem 181 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 180 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _
    rw [hlc 0, hfr "mlm" 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
    exact setp_ge_iff (st.regs "mlm" 0) 15 (by decide)

theorem tok_at13464 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 134)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.sub) "litExtra" "litLen" (SArg.imm 15)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K16 st = (st.setReg "litExtra"
      (fun j => SOp.sub.run (st.regs "litLen" j) (st.get j (SArg.imm 15)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 135 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hD : ∀ j : Lane, (sstep K16 st).regs "litExtra" j
      = st.regs "litLen" j - UInt64.ofNat 15 := by
    intro j; rw [hstep]; rfl
  have hguard : 15 ≤ (st.regs "litLen" l).toNat := c5 he
  have hDN : ((sstep K16 st).regs "litExtra" l).toNat = (st.regs "litLen" l).toNat - 15 := by
    rw [hD l]; exact uint64_sub_toNat _ _ (by decide) hguard
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "mlm" l (Or.inr (Or.inr (Or.inl rfl))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hDN]
    exact Nat.le_trans (Nat.add_le_add_left (tokRem_sub134 _ _ _ _ hguard) _) c1
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hD l, hD 0, c2 "litLen" (Or.inr (Or.inl rfl))]
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at14264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 142)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.sub) "litExtra" "litExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K16 st = (st.setReg "litExtra"
      (fun j => SOp.sub.run (st.regs "litExtra" j) (st.get j (SArg.imm 255)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 143 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hD : ∀ j : Lane, (sstep K16 st).regs "litExtra" j
      = st.regs "litExtra" j - UInt64.ofNat 255 := by
    intro j; rw [hstep]; rfl
  have hguard : 255 ≤ (st.regs "litExtra" l).toNat := c4 (by rw [he]; omega) (by rw [he]; omega)
  have hDN : ((sstep K16 st).regs "litExtra" l).toNat = (st.regs "litExtra" l).toNat - 255 := by
    rw [hD l]; exact uint64_sub_toNat _ _ (by decide) hguard
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "mlm" l (Or.inr (Or.inr (Or.inl rfl))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hDN]
    exact Nat.le_trans (Nat.add_le_add_left (tokRem_sub142 _ _ _ _ hguard) _) c1
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hD l, hD 0, c2 "litExtra" (Or.inr (Or.inr (Or.inl rfl)))]
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at18264 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 182)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.sub) "matExtra" "mlm" (SArg.imm 15)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K16 st = (st.setReg "matExtra"
      (fun j => SOp.sub.run (st.regs "mlm" j) (st.get j (SArg.imm 15)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 183 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hD : ∀ j : Lane, (sstep K16 st).regs "matExtra" j
      = st.regs "mlm" j - UInt64.ofNat 15 := by
    intro j; rw [hstep]; rfl
  have hguard : 15 ≤ (st.regs "mlm" l).toNat := c9 he
  have hDN : ((sstep K16 st).regs "matExtra" l).toNat = (st.regs "mlm" l).toNat - 15 := by
    rw [hD l]; exact uint64_sub_toNat _ _ (by decide) hguard
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hDN]
    exact Nat.le_trans (Nat.add_le_add_left (tokRem_sub182 _ _ _ _ hguard) _) c1
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hD l, hD 0, c2 "mlm" (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at19064 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 190)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.sub) "matExtra" "matExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K16 st = (st.setReg "matExtra"
      (fun j => SOp.sub.run (st.regs "matExtra" j) (st.get j (SArg.imm 255)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 191 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hD : ∀ j : Lane, (sstep K16 st).regs "matExtra" j
      = st.regs "matExtra" j - UInt64.ofNat 255 := by
    intro j; rw [hstep]; rfl
  have hguard : 255 ≤ (st.regs "matExtra" l).toNat := c8 (by rw [he]; omega) (by rw [he]; omega)
  have hDN : ((sstep K16 st).regs "matExtra" l).toNat = (st.regs "matExtra" l).toNat - 255 := by
    rw [hD l]; exact uint64_sub_toNat _ _ (by decide) hguard
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hDN]
    exact Nat.le_trans (Nat.add_le_add_left (tokRem_sub190 _ _ _ _ hguard) _) c1
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hD l, hD 0, c2 "matExtra" (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at17064 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 170)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.bin (.add) "op" "op" (SArg.reg "litLen")) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K16 st = (st.setReg "op"
      (fun j => SOp.add.run (st.regs "op" j) (st.get j (SArg.reg "litLen")))).setPc
      (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K16 st).pc = 171 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "litLen" ∨ r = "litExtra" ∨ r = "mlm"
      ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hop : ∀ j : Lane, (sstep K16 st).regs "op" j = st.regs "op" j + st.regs "litLen" j := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  have e170 : tokRem 170 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat
      (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat
      = (st.regs "litLen" l).toNat + 2 + lsicLen (st.regs "mlm" l).toNat := rfl
  have e171 : tokRem 171 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat
      (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat
      = 2 + lsicLen (st.regs "mlm" l).toNat := rfl
  have hopN : ((st.regs "op" l) + (st.regs "litLen" l)).toNat
      = (st.regs "op" l).toNat + (st.regs "litLen" l).toNat := by
    rw [UInt64.toNat_add, Nat.mod_eq_of_lt (by omega)]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hop l, hopN, hfr "litLen" l (Or.inl rfl), hfr "litExtra" l (Or.inr (Or.inl rfl)),
      hfr "mlm" l (Or.inr (Or.inr (Or.inl rfl))),
      hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hop l, hop 0, c2 "op" (Or.inl rfl), c2 "litLen" (Or.inr (Or.inl rfl))]
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]; exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))),
        hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl)))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at15764 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 157)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.braifnot "cpCont" "Cx11") := by rw [he]; decide
  have hstep : sstep K16 st
      = st.setPc (if st.regs "cpCont" 0 == 1 then st.pc + 1 else sfindLabel K16 "Cx11") := by
    rw [sstep, hp]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm"
      ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j _; rw [hstep]; rfl
  by_cases hg : (st.regs "cpCont" 0 == 1) = true
  · exact tok_frame64 l B st 158 (by rw [hstep, he, if_pos hg]; rfl) hfr
      (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) h
  · exact tok_frame64 l B st 169
      (by rw [hstep, if_neg hg, show sfindLabel K16 "Cx11" = 169 from by decide]; rfl) hfr
      (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) h


theorem tok_at13364 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 133)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.braifnot "pLitBig" "Le6") := by rw [he]; decide
  have hstep : sstep K16 st
      = st.setPc (if st.regs "pLitBig" 0 == 1 then st.pc + 1 else sfindLabel K16 "Le6") := by
    rw [sstep, hp]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j _; rw [hstep]; rfl
  by_cases hg : (st.regs "pLitBig" 0 == 1) = true
  · obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
    have hg2 : 15 ≤ (st.regs "litLen" l).toNat := by
      rw [c2 "litLen" (Or.inr (Or.inl rfl))]; exact (c6 he).mp hg
    have hpc' : (sstep K16 st).pc = 134 := by rw [hstep, he, if_pos hg]; rfl
    rw [he] at c1
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)),
        hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      have e : tokRem 134 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 133 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
      omega
    · intro r hr
      rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
      · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
        exact c2 _ (Or.inl rfl)
      · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
        exact c2 _ (Or.inr (Or.inl rfl))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
        exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro ha hb; omega
    · rw [hpc']; intro _; rw [hfr "litLen" l (Or.inr (Or.inl rfl))]; exact hg2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro ha hb; omega
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
  · exact tok_frame64 l B st 150
      (by rw [hstep, if_neg hg, show sfindLabel K16 "Le6" = 150 from by decide]; rfl) hfr
      (by rw [he]; exact tokRem_le150) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at13764 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 137)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.braifnot "lsicC" "Lx9") := by rw [he]; decide
  have hstep : sstep K16 st
      = st.setPc (if st.regs "lsicC" 0 == 1 then st.pc + 1 else sfindLabel K16 "Lx9") := by
    rw [sstep, hp]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j _; rw [hstep]; rfl
  by_cases hg : (st.regs "lsicC" 0 == 1) = true
  · obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
    have hg2 : 255 ≤ (st.regs "litExtra" l).toNat := by
      rw [c2 "litExtra" (Or.inr (Or.inr (Or.inl rfl)))]; exact (c3 (Or.inr (Or.inl he))).mp hg
    have hpc' : (sstep K16 st).pc = 138 := by rw [hstep, he, if_pos hg]; rfl
    rw [he] at c1
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)),
        hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      have e : tokRem 138 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 137 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
      omega
    · intro r hr
      rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
      · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
        exact c2 _ (Or.inl rfl)
      · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
        exact c2 _ (Or.inr (Or.inl rfl))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
        exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro _ _; rw [hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl)))]; exact hg2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro ha hb; omega
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
  · exact tok_frame64 l B st 145
      (by rw [hstep, if_neg hg, show sfindLabel K16 "Lx9" = 145 from by decide]; rfl) hfr
      (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at18164 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 181)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.braifnot "pMatBig" "Le12") := by rw [he]; decide
  have hstep : sstep K16 st
      = st.setPc (if st.regs "pMatBig" 0 == 1 then st.pc + 1 else sfindLabel K16 "Le12") := by
    rw [sstep, hp]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j _; rw [hstep]; rfl
  by_cases hg : (st.regs "pMatBig" 0 == 1) = true
  · obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
    have hg2 : 15 ≤ (st.regs "mlm" l).toNat := by
      rw [c2 "mlm" (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]; exact (c10 he).mp hg
    have hpc' : (sstep K16 st).pc = 182 := by rw [hstep, he, if_pos hg]; rfl
    rw [he] at c1
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)),
        hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      have e : tokRem 182 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 181 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
      omega
    · intro r hr
      rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
      · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
        exact c2 _ (Or.inl rfl)
      · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
        exact c2 _ (Or.inr (Or.inl rfl))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
        exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro ha hb; omega
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro ha hb; omega
    · rw [hpc']; intro _; rw [hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]; exact hg2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
  · exact tok_frame64 l B st 198
      (by rw [hstep, if_neg hg, show sfindLabel K16 "Le12" = 198 from by decide]; rfl) hfr
      (by rw [he]; exact tokRem_le198) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at18564 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 185)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.braifnot "lsicC" "Lx15") := by rw [he]; decide
  have hstep : sstep K16 st
      = st.setPc (if st.regs "lsicC" 0 == 1 then st.pc + 1 else sfindLabel K16 "Lx15") := by
    rw [sstep, hp]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K16 st).regs r j = st.regs r j := by
    intro r j _; rw [hstep]; rfl
  by_cases hg : (st.regs "lsicC" 0 == 1) = true
  · obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
    have hg2 : 255 ≤ (st.regs "matExtra" l).toNat := by
      rw [c2 "matExtra" (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]; exact (c7 (Or.inr (Or.inl he))).mp hg
    have hpc' : (sstep K16 st).pc = 186 := by rw [hstep, he, if_pos hg]; rfl
    rw [he] at c1
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)),
        hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      have e : tokRem 186 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 185 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
      omega
    · intro r hr
      rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
      · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
        exact c2 _ (Or.inl rfl)
      · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
        exact c2 _ (Or.inr (Or.inl rfl))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
        exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro ha hb; omega
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro _ _; rw [hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]; exact hg2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
  · exact tok_frame64 l B st 193
      (by rw [hstep, if_neg hg, show sfindLabel K16 "Lx15" = 193 from by decide]; rfl) hfr
      (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) h


theorem tok_at164_g16 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 164)
    (h : TokInv l B st) : TokInv l B (sstep K16 st) := by
  have hp : K16[st.pc]? = some (.ldgo "cpB" "cpSo" 0) := by rw [he]; decide
  refine tok_frame64 l B st 165 (by rw [sstep, hp]; show st.pc + 1 = 165; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tokS_hstep64 (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (hs : st.pc ∈ tokS) (hex : st.pc ∉ [197, 198]) (h : TokInv l B st) :
    TokInv l B (sstep K16 st) := by
  simp only [tokS, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e
  · exact tok_at12964 l B st e h
  · exact tok_at13064 l B st e h
  · exact tok_at13164 l B st hB e h
  · exact tok_at13264 l B st e h
  · exact tok_at13364 l B st e h
  · exact tok_at13464 l B st e h
  · exact tok_at13564 l B st e h
  · exact tok_at13664 l B st e h
  · exact tok_at13764 l B st e h
  · exact tok_at13864 l B st e h
  · exact tok_at13964 l B st e h
  · exact tok_at14064 l B st e h
  · exact tok_at14164 l B st hB e h
  · exact tok_at14264 l B st e h
  · exact tok_at14364 l B st e h
  · exact tok_at14464 l B st e h
  · exact tok_at14564 l B st e h
  · exact tok_at14664 l B st e h
  · exact tok_at14764 l B st e h
  · exact tok_at14864 l B st hB e h
  · exact tok_at14964 l B st e h
  · exact tok_at15064 l B st e h
  · exact tok_at15164 l B st e h
  · exact tok_at15264 l B st e h
  · exact tok_at15364 l B st e h
  · exact tok_at15464 l B st e h
  · exact tok_at15564 l B st e h
  · exact tok_at15664 l B st e h
  · exact tok_at15764 l B st e h
  · exact tok_at15864 l B st e h
  · exact tok_at15964 l B st e h
  · exact tok_at16064 l B st e h
  · exact tok_at16164 l B st e h
  · exact tok_at16264 l B st e h
  · exact tok_at16364 l B st e h
  · exact tok_at164_g16 l B st e h
  · exact tok_at16564 l B st e h
  · exact tok_at16664 l B st e h
  · exact tok_at16764 l B st e h
  · exact tok_at16864 l B st e h
  · exact tok_at16964 l B st e h
  · exact tok_at17064 l B st hB e h
  · exact tok_at17164 l B st e h
  · exact tok_at17264 l B st e h
  · exact tok_at17364 l B st e h
  · exact tok_at17464 l B st hB e h
  · exact tok_at17564 l B st e h
  · exact tok_at17664 l B st e h
  · exact tok_at17764 l B st e h
  · exact tok_at17864 l B st e h
  · exact tok_at17964 l B st hB e h
  · exact tok_at18064 l B st e h
  · exact tok_at18164 l B st e h
  · exact tok_at18264 l B st e h
  · exact tok_at18364 l B st e h
  · exact tok_at18464 l B st e h
  · exact tok_at18564 l B st e h
  · exact tok_at18664 l B st e h
  · exact tok_at18764 l B st e h
  · exact tok_at18864 l B st e h
  · exact tok_at18964 l B st hB e h
  · exact tok_at19064 l B st e h
  · exact tok_at19164 l B st e h
  · exact tok_at19264 l B st e h
  · exact tok_at19364 l B st e h
  · exact tok_at19464 l B st e h
  · exact tok_at19564 l B st e h
  · exact tok_at19664 l B st hB e h
  · exact absurd (by simp [e]) hex
  · exact absurd (by simp [e]) hex

theorem tok_op_lt64 (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (h0 : st.pc = 129) (h : TokInv l B st) (k : Nat)
    (hne : ∀ j, j < k → (siter K16 j st).pc ∉ [197, 198])
    (hq : (siter K16 k st).pc = 130 ∨ (siter K16 k st).pc = 140 ∨ (siter K16 k st).pc = 147
      ∨ (siter K16 k st).pc = 173 ∨ (siter K16 k st).pc = 178 ∨ (siter K16 k st).pc = 188
      ∨ (siter K16 k st).pc = 195) :
    ((siter K16 k st).regs "op" l).toNat < B :=
  tokInv_op_lt l B _
    (inv_on K16 (TokInv l B) tokS [197, 198] tokS_closed64
      (fun s hsm hexs hh => tokS_hstep64 l B hB s hsm hexs hh) st (by rw [h0]; decide) h k hne) hq

theorem loopBodyS_closed64 : PcClosed K16 loopBodyS [207] :=
  ivClosed_at K16 42 166 [207] kSize16 (by omega) (by decide)

theorem mb_succs64 : AlgorithmLib.LZ4Simt.succsOf K16 124 = [125] := by decide

theorem mb_top64 : ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 124, 124 < q' := by decide

theorem mb_noentry_lt64 : ∀ q', q' < 203 → 124 ≤ q' →
    124 ∉ AlgorithmLib.LZ4Simt.succsOf K16 q' := by decide

theorem mb_noentry64 (hi : Nat) (hhi : hi ≤ 202) :
    ∀ q', 124 ≤ q' → q' ≤ hi → 124 ∉ AlgorithmLib.LZ4Simt.succsOf K16 q' :=
  fun q' h1 h2 => mb_noentry_lt64 q' (by omega) h1


end Lz4Sites
