import AlgorithmLib.LZ4WarpKernel
set_option maxRecDepth 4096

namespace CoopCopyModel
open AlgorithmLib.LZ4WarpDSL (copyGmem copyGmem_size copyGmem_getD_lt copyGmem_getD)

/-- One 32-lane strided-copy block at offset `i` (see `ScratchCopyCore.cpBlock`). -/
def cpBlock (g : Array UInt8) (dst src i len : Nat) : Nat → Array UInt8
  | 0 => g
  | l + 1 =>
      let acc := cpBlock g dst src i len l
      if i + l < len then acc.setIfInBounds (dst + i + l) (g.getD (src + i + l) 0) else acc

theorem cpBlock_size : ∀ (w : Nat) (g : Array UInt8) (dst src i len : Nat),
    (cpBlock g dst src i len w).size = g.size := by
  intro w
  induction w with
  | zero => intro g dst src i len; rfl
  | succ w ih =>
      intro g dst src i len
      show (if i + w < len then (cpBlock g dst src i len w).setIfInBounds (dst + i + w)
              (g.getD (src + i + w) 0) else cpBlock g dst src i len w).size = g.size
      by_cases h : i + w < len <;> simp [h, ih, Array.size_setIfInBounds]

theorem cpBlock_getD_ne : ∀ (w : Nat) (g : Array UInt8) (dst src i len j : Nat),
    (∀ l, l < w → i + l < len → dst + i + l ≠ j) →
    (cpBlock g dst src i len w).getD j 0 = g.getD j 0 := by
  intro w
  induction w with
  | zero => intro g dst src i len j _; rfl
  | succ w ih =>
      intro g dst src i len j hne
      show (if i + w < len then (cpBlock g dst src i len w).setIfInBounds (dst + i + w)
              (g.getD (src + i + w) 0) else cpBlock g dst src i len w).getD j 0 = g.getD j 0
      have ihj : (cpBlock g dst src i len w).getD j 0 = g.getD j 0 :=
        ih g dst src i len j (fun l hl hlen => hne l (by omega) hlen)
      by_cases h : i + w < len
      · rw [if_pos h]
        rw [Array.getD_eq_getD_getElem?,
            Array.getElem?_setIfInBounds_ne (hne w (by omega) h),
            ← Array.getD_eq_getD_getElem?, ihj]
      · rw [if_neg h]; exact ihj

theorem cpBlock_getD_hit : ∀ (w : Nat) (g : Array UInt8) (dst src i len l0 : Nat),
    l0 < w → i + l0 < len → dst + i + l0 < g.size →
    (cpBlock g dst src i len w).getD (dst + i + l0) 0 = g.getD (src + i + l0) 0 := by
  intro w
  induction w with
  | zero => intro g dst src i len l0 hl0 _ _; omega
  | succ w ih =>
      intro g dst src i len l0 hl0 hlen hsz
      show (if i + w < len then (cpBlock g dst src i len w).setIfInBounds (dst + i + w)
              (g.getD (src + i + w) 0) else cpBlock g dst src i len w).getD (dst + i + l0) 0
        = g.getD (src + i + l0) 0
      by_cases he : l0 = w
      · subst he
        rw [if_pos hlen]
        have hsz' : dst + i + l0 < (cpBlock g dst src i len l0).size := by
          rw [cpBlock_size]; exact hsz
        rw [Array.getD_eq_getD_getElem?,
            Array.getElem?_setIfInBounds_self_of_lt hsz',
            Option.getD_some]
      · have hl0w : l0 < w := by omega
        have ihj : (cpBlock g dst src i len w).getD (dst + i + l0) 0 = g.getD (src + i + l0) 0 :=
          ih g dst src i len l0 hl0w hlen hsz
        by_cases h : i + w < len
        · rw [if_pos h]
          rw [Array.getD_eq_getD_getElem?,
              Array.getElem?_setIfInBounds_ne (by omega : dst + i + w ≠ dst + i + l0),
              ← Array.getD_eq_getD_getElem?, ihj]
        · rw [if_neg h]; exact ihj

/-- The whole emitted copy loop, block-by-block: `fuel` iterations of `cpBlock`. -/
def cpLoop (dst src len : Nat) : Nat → Nat → Array UInt8 → Array UInt8
  | 0, _, g => g
  | fuel + 1, i, g =>
      if i < len then cpLoop dst src len fuel (i + 32) (cpBlock g dst src i len 32) else g

theorem cpLoop_size : ∀ (fuel i : Nat) (g : Array UInt8) (dst src len : Nat),
    (cpLoop dst src len fuel i g).size = g.size := by
  intro fuel
  induction fuel with
  | zero => intro i g dst src len; rfl
  | succ fuel ih =>
      intro i g dst src len
      show (if i < len then cpLoop dst src len fuel (i + 32) (cpBlock g dst src i len 32)
              else g).size = g.size
      by_cases h : i < len
      · rw [if_pos h, ih, cpBlock_size]
      · rw [if_neg h]

theorem cpLoop_spec (dst src len : Nat) (hdisj : dst + len ≤ src ∨ src + len ≤ dst) :
    ∀ (fuel i : Nat) (g : Array UInt8), dst + len ≤ g.size → len ≤ i + 32 * fuel →
      (∀ j, (j < dst + i ∨ dst + len ≤ j) →
          (cpLoop dst src len fuel i g).getD j 0 = g.getD j 0)
      ∧ (∀ k, i ≤ k → k < len →
          (cpLoop dst src len fuel i g).getD (dst + k) 0 = g.getD (src + k) 0) := by
  intro fuel
  induction fuel with
  | zero =>
      intro i g _ hfuel
      refine ⟨fun j _ => rfl, fun k hik hk => ?_⟩
      simp only [Nat.mul_zero, Nat.add_zero] at hfuel; omega
  | succ fuel ih =>
      intro i g hsize hfuel
      show (∀ j, (j < dst + i ∨ dst + len ≤ j) →
          (if i < len then cpLoop dst src len fuel (i + 32) (cpBlock g dst src i len 32)
              else g).getD j 0 = g.getD j 0)
        ∧ (∀ k, i ≤ k → k < len →
          (if i < len then cpLoop dst src len fuel (i + 32) (cpBlock g dst src i len 32)
              else g).getD (dst + k) 0 = g.getD (src + k) 0)
      by_cases hi : i < len
      · rw [if_pos hi]
        have hsize' : dst + len ≤ (cpBlock g dst src i len 32).size := by
          rw [cpBlock_size]; exact hsize
        have hfuel' : len ≤ (i + 32) + 32 * fuel := by omega
        obtain ⟨hframe', hhit'⟩ := ih (i + 32) (cpBlock g dst src i len 32) hsize' hfuel'
        constructor
        · intro j hj
          rcases hj with hjlt | hjge
          · rw [hframe' j (Or.inl (by omega))]
            exact cpBlock_getD_ne 32 g dst src i len j (fun l _ _ => by omega)
          · rw [hframe' j (Or.inr hjge)]
            exact cpBlock_getD_ne 32 g dst src i len j
              (fun l _ hlen => by omega)
        · intro k hik hk
          by_cases hk2 : k < i + 32
          · have hblk : (cpBlock g dst src i len 32).getD (dst + k) 0 = g.getD (src + k) 0 := by
              have := cpBlock_getD_hit 32 g dst src i len (k - i) (by omega) (by omega)
                (by omega)
              rw [show dst + i + (k - i) = dst + k by omega,
                  show src + i + (k - i) = src + k by omega] at this
              exact this
            rw [hframe' (dst + k) (Or.inl (by omega)), hblk]
          · rw [hhit' k (by omega) hk]
            exact cpBlock_getD_ne 32 g dst src i len (src + k)
              (fun l _ hlen => by rcases hdisj with hd | hd <;> omega)
      · rw [if_neg hi]
        exact ⟨fun j _ => rfl, fun k hik hk => by omega⟩

theorem copyGmem_getD_ge (j : Nat) : ∀ (n : Nat) (g : Array UInt8) (dst src : Nat),
    dst + n ≤ j → (copyGmem g dst src n).getD j 0 = g.getD j 0 := by
  intro n
  induction n with
  | zero => intro g dst src _; rfl
  | succ n ih =>
      intro g dst src hj
      show (copyGmem (g.setIfInBounds dst (g.getD src 0)) (dst + 1) (src + 1) n).getD j 0
        = g.getD j 0
      rw [ih (g.setIfInBounds dst (g.getD src 0)) (dst + 1) (src + 1) (by omega)]
      rw [Array.getD_eq_getD_getElem?,
          Array.getElem?_setIfInBounds_ne (by omega : dst ≠ j),
          ← Array.getD_eq_getD_getElem?]

theorem array_eq_of_getD {a b : Array UInt8} (hs : a.size = b.size)
    (h : ∀ j, a.getD j 0 = b.getD j 0) : a = b := by
  apply Array.ext hs
  intro i hi hi2
  have hj := h i
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?,
      Array.getElem?_eq_getElem hi, Array.getElem?_eq_getElem hi2] at hj
  simpa using hj

theorem cpLoop_eq_copyGmem (dst src len : Nat) (g : Array UInt8)
    (hdisj : dst + len ≤ src ∨ src + len ≤ dst) (hsize : dst + len ≤ g.size)
    (fuel : Nat) (hfuel : len ≤ 32 * fuel) :
    cpLoop dst src len fuel 0 g = copyGmem g dst src len := by
  obtain ⟨hframe, hhit⟩ :=
    cpLoop_spec dst src len hdisj fuel 0 g hsize (by simpa using hfuel)
  apply array_eq_of_getD
  · rw [cpLoop_size, copyGmem_size]
  · intro j
    by_cases hjd : j < dst
    · rw [hframe j (Or.inl (by omega)), copyGmem_getD_lt j len g dst src hjd]
    · by_cases hjl : j < dst + len
      · have hk : j - dst < len := by omega
        have hj : dst + (j - dst) = j := by omega
        rw [← hj, hhit (j - dst) (by omega) hk,
            copyGmem_getD len g dst src (j - dst) hk hsize hdisj]
      · rw [hframe j (Or.inr (by omega)), copyGmem_getD_ge j len g dst src (by omega)]

end CoopCopyModel

-- ── The machine leaf ────────────────────────────────────────────────────────

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib AlgorithmLib.LZ4Simt
open CoopCopyModel

-- ── UInt64 arithmetic helpers ──────────────────────────────────────────────

theorem uOfNat_toNat (a : Nat) (h : a < 2 ^ 64) : (UInt64.ofNat a).toNat = a := by
  show (BitVec.ofNat 64 a).toNat = a
  rw [BitVec.toNat_ofNat]; omega

/-- Unconditional (mod-2^64-compatible) collapse of two `UInt64.ofNat` sums. -/
theorem uAdd_eq (a b : Nat) : UInt64.ofNat a + UInt64.ofNat b = UInt64.ofNat (a + b) := by
  apply UInt64.eq_of_toBitVec_eq
  show BitVec.ofNat 64 a + BitVec.ofNat 64 b = BitVec.ofNat 64 (a + b)
  rw [BitVec.ofNat_add]

theorem u8_roundtrip (x : UInt8) : (UInt64.ofNat x.toNat).toUInt8 = x := by
  apply UInt8.toNat.inj
  have h := x.toNat_lt
  have h1 : (UInt64.ofNat x.toNat).toNat = x.toNat := uOfNat_toNat x.toNat (by omega)
  rw [UInt64.toNat_toUInt8, h1]; omega

-- ── `storeBytes` (SIMT, `List.finRange 32`-indexed) = `cpBlock` (Nat-indexed) ──

theorem storeBytes_take_eq (g : Array UInt8) (dst src i len : Nat)
    (pred : Lane → Bool) (addr val : Lane → UInt64)
    (hpred : ∀ l : Lane, pred l = decide (i + l.val < len))
    (haddr : ∀ l : Lane, (addr l).toNat = dst + i + l.val)
    (hval : ∀ l : Lane, (val l).toUInt8 = g.getD (src + i + l.val) 0) :
    ∀ w : Nat, w ≤ 32 →
    List.foldl (fun m (l : Lane) => if pred l then m.set! (addr l).toNat (val l).toUInt8 else m) g
      ((List.finRange 32).take w)
    = cpBlock g dst src i len w := by
  intro w
  induction w with
  | zero => intro _; rfl
  | succ w ih =>
    intro hw
    have hw32 : w < 32 := by omega
    have hwle : w ≤ 32 := by omega
    have hlt : w < (List.finRange 32).length := by rw [List.length_finRange]; exact hw32
    rw [List.take_succ]
    have hget : (List.finRange 32)[w]? = some (⟨w, hw32⟩ : Lane) := by
      rw [List.getElem?_eq_getElem hlt, List.getElem_finRange]; rfl
    rw [hget]
    simp only [Option.toList_some]
    rw [List.foldl_append, ih hwle]
    show (if pred ⟨w, hw32⟩ then
            (cpBlock g dst src i len w).set! (addr ⟨w, hw32⟩).toNat (val ⟨w, hw32⟩).toUInt8
          else cpBlock g dst src i len w) = cpBlock g dst src i len (w + 1)
    rw [hpred, haddr, hval]
    simp [cpBlock, Array.set!_eq_setIfInBounds]

theorem storeBytes_eq_cpBlock (g : Array UInt8) (dst src i len : Nat)
    (pred : Lane → Bool) (addr val : Lane → UInt64)
    (hpred : ∀ l : Lane, pred l = decide (i + l.val < len))
    (haddr : ∀ l : Lane, (addr l).toNat = dst + i + l.val)
    (hval : ∀ l : Lane, (val l).toUInt8 = g.getD (src + i + l.val) 0) :
    storeBytes g pred addr val = cpBlock g dst src i len 32 := by
  have h32 : (List.finRange 32).take 32 = List.finRange 32 :=
    List.take_of_length_le (by rw [List.length_finRange]; omega)
  have := storeBytes_take_eq g dst src i len pred addr val hpred haddr hval 32 (by omega)
  rwa [h32] at this

-- ── Per-iteration machine leaf: the 10-instruction loop body ──────────────────

/-- Scratch registers written by one `.coopCopy` loop-body iteration. -/
def coopCopyScratch : List String := ["cpDo", "cpSo", "cpJ", "cpP", "cpB", "cpI", "cpCont"]

/-- One pass of the emitted `.coopCopy` loop body: 10 straight-line instructions
    advancing the strided-copy offset `cpI` by 32, and updating `gmem` exactly as
    `cpBlock` (the pure block model) does for one 32-lane block. -/
theorem coopCopy_iter (prog : Array SInstr) (ss : SState) (dst src len : String)
    (i dst0 src0 len0 : Nat)
    (h0 : prog[ss.pc]? = some (.binr .add "cpDo" dst "cpI"))
    (h1 : prog[ss.pc + 1]? = some (.binr .add "cpDo" "cpDo" "lane"))
    (h2 : prog[ss.pc + 2]? = some (.binr .add "cpSo" src "cpI"))
    (h3 : prog[ss.pc + 3]? = some (.binr .add "cpSo" "cpSo" "lane"))
    (h4 : prog[ss.pc + 4]? = some (.binr .add "cpJ" "cpI" "lane"))
    (h5 : prog[ss.pc + 5]? = some (.setp .lt "cpP" "cpJ" (.reg len)))
    (h6 : prog[ss.pc + 6]? = some (.ldgo "cpB" "cpSo" 0))
    (h7 : prog[ss.pc + 7]? = some (.stgp "cpP" "cpDo" "cpB"))
    (h8 : prog[ss.pc + 8]? = some (.bin .add "cpI" "cpI" (.imm 32)))
    (h9 : prog[ss.pc + 9]? = some (.setp .lt "cpCont" "cpI" (.reg len)))
    (hdst : dst ∉ coopCopyScratch) (hsrc : src ∉ coopCopyScratch) (hlenn : len ∉ coopCopyScratch)
    (hcpI : ∀ l : Fin 32, ss.regs "cpI" l = UInt64.ofNat i)
    (hdstv : ∀ l : Fin 32, ss.regs dst l = UInt64.ofNat dst0)
    (hsrcv : ∀ l : Fin 32, ss.regs src l = UInt64.ofNat src0)
    (hlenv : ∀ l : Fin 32, ss.regs len l = UInt64.ofNat len0)
    (hlane : ∀ l : Fin 32, ss.regs "lane" l = UInt64.ofNat l.val)
    (hb1 : dst0 < 2 ^ 32) (hb2 : src0 < 2 ^ 32) (hb3 : len0 < 2 ^ 32) (hb4 : i ≤ len0 + 32) :
    (snsteps prog 10 ss).pc = ss.pc + 10 ∧
    (snsteps prog 10 ss).smem = ss.smem ∧
    (snsteps prog 10 ss).gmem = cpBlock ss.gmem dst0 src0 i len0 32 ∧
    (∀ l : Fin 32, (snsteps prog 10 ss).regs "cpI" l = UInt64.ofNat (i + 32)) ∧
    (∀ l : Fin 32, (snsteps prog 10 ss).regs "cpCont" l = if i + 32 < len0 then 1 else 0) ∧
    (∀ r : String, r ∉ coopCopyScratch → ∀ l : Fin 32, (snsteps prog 10 ss).regs r l = ss.regs r l) := by
  simp only [coopCopyScratch, List.mem_cons, List.mem_singleton, not_or, List.not_mem_nil,
    not_false_iff, and_true] at hdst hsrc hlenn
  obtain ⟨hd0, hd1, hd2, hd3, hd4, hd5, hd6⟩ := hdst
  obtain ⟨hs0, hs1, hs2, hs3, hs4, hs5, hs6⟩ := hsrc
  obtain ⟨hl0, hl1, hl2, hl3, hl4, hl5, hl6⟩ := hlenn
  -- Frame / pc / smem (values irrelevant, pure name-disjointness).
  have hframe : ∀ r : String, r ∉ coopCopyScratch →
      (snsteps prog 10 ss).pc = ss.pc + 10 ∧ (snsteps prog 10 ss).smem = ss.smem ∧
      (snsteps prog 10 ss).regs r = ss.regs r := by
    intro r hr
    simp only [coopCopyScratch, List.mem_cons, List.mem_singleton, not_or, List.not_mem_nil,
      not_false_iff, and_true] at hr
    obtain ⟨n0, n1, n2, n3, n4, n5, n6⟩ := hr
    refine ⟨?_, ?_, ?_⟩ <;>
      simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, sstepInstr, SState.setReg,
        SState.setPc, SState.get, SOp.run, SCmp.run, storeBytes, n0, n1, n2, n3, n4, n5, n6]
  -- Raw (pre-arithmetic) value of `cpI`.
  have hrawI : ∀ l : Fin 32, (snsteps prog 10 ss).regs "cpI" l = ss.regs "cpI" l + UInt64.ofNat 32 := by
    intro l
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, sstepInstr, SState.setReg,
      SState.setPc, SState.get, SOp.run, SCmp.run]
  -- Raw (pre-arithmetic) value of `cpCont`.
  have hrawC : ∀ l : Fin 32, (snsteps prog 10 ss).regs "cpCont" l
      = if SCmp.run .lt (ss.regs "cpI" l + UInt64.ofNat 32) (ss.regs len l) then 1 else 0 := by
    intro l
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, sstepInstr, SState.setReg,
      SState.setPc, SState.get, SOp.run, SCmp.run, hl0, hl1, hl2, hl3, hl4, hl5]
  -- Raw (pre-arithmetic) `gmem` update, as a `storeBytes` call.
  have hrawG : (snsteps prog 10 ss).gmem
      = storeBytes ss.gmem
          (fun l => decide (SCmp.run .lt (ss.regs "cpI" l + ss.regs "lane" l) (ss.regs len l)))
          (fun l => (ss.regs dst l + ss.regs "cpI" l) + ss.regs "lane" l)
          (fun l => UInt64.ofNat (ss.gmem.getD
            (((ss.regs src l + ss.regs "cpI" l) + ss.regs "lane" l).toNat) 0).toNat) := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, sstepInstr, SState.setReg,
      SState.setPc, SState.get, SOp.run, SCmp.run, storeBytes, hd0, hs0, hl0, hl1, hl2, hl3, hl4, hl5]
  refine ⟨(hframe "cpDo_unused_marker" (by simp [coopCopyScratch])).1, ?_, ?_, ?_, ?_, ?_⟩
  · -- smem: use the frame fact for any name (pc/smem don't depend on `r`).
    exact (hframe "cpDo_unused_marker" (by simp [coopCopyScratch])).2.1
  · -- gmem = cpBlock.
    rw [hrawG]
    apply storeBytes_eq_cpBlock
    · intro l
      have hlval : l.val < 32 := l.isLt
      rw [hlane, hcpI, hlenv, uAdd_eq i l.val]
      have hlt64 : i + l.val < 2 ^ 64 := by omega
      have hlen64 : len0 < 2 ^ 64 := by omega
      simp only [SCmp.run, decide_eq_decide, decide_eq_true_eq, UInt64.lt_iff_toNat_lt,
        uOfNat_toNat (i + l.val) hlt64, uOfNat_toNat len0 hlen64]
    · intro l
      have hlval : l.val < 32 := l.isLt
      rw [hdstv, hcpI, hlane, uAdd_eq dst0 i, uAdd_eq (dst0 + i) l.val]
      have h2' : dst0 + i + l.val < 2 ^ 64 := by omega
      rw [uOfNat_toNat (dst0 + i + l.val) h2']
    · intro l
      have hlval : l.val < 32 := l.isLt
      rw [hsrcv, hcpI, hlane, uAdd_eq src0 i, uAdd_eq (src0 + i) l.val]
      have h2' : src0 + i + l.val < 2 ^ 64 := by omega
      rw [uOfNat_toNat (src0 + i + l.val) h2']
      exact u8_roundtrip _
  · -- cpI = ofNat (i + 32)
    intro l; rw [hrawI, hcpI]
    exact uAdd_eq i 32
  · -- cpCont = if i+32<len0 then 1 else 0
    intro l
    rw [hrawC, hcpI, hlenv, uAdd_eq i 32]
    have h1' : i + 32 < 2 ^ 64 := by omega
    have h2' : len0 < 2 ^ 64 := by omega
    simp only [SCmp.run, decide_eq_true_eq, UInt64.lt_iff_toNat_lt,
      uOfNat_toNat (i + 32) h1', uOfNat_toNat len0 h2']
  · -- frame.
    intro r hr l
    exact congrFun (hframe r hr).2.2 l

-- ── The whole `.coopCopy` loop: induction over `fuel`, mirroring `simSL_uwhile` ─

/-- The 10-instruction body emitted for one `.coopCopy` loop iteration
    (`WStmt.emitM`'s `.coopCopy` case in `LZ4WarpDSL.lean`). -/
def coopCopyBody (dst src len : String) : List SInstr :=
  [.binr .add "cpDo" dst "cpI", .binr .add "cpDo" "cpDo" "lane",
   .binr .add "cpSo" src "cpI", .binr .add "cpSo" "cpSo" "lane",
   .binr .add "cpJ" "cpI" "lane", .setp .lt "cpP" "cpJ" (.reg len),
   .ldgo "cpB" "cpSo" 0, .stgp "cpP" "cpDo" "cpB",
   .bin .add "cpI" "cpI" (.imm 32), .setp .lt "cpCont" "cpI" (.reg len)]

theorem coopCopyBody_length (dst src len : String) : (coopCopyBody dst src len).length = 10 := rfl

/-- The whole emitted `.coopCopy` loop (`uwhileEmit "cpCont" lH lX (coopCopyBody ..)`,
    the `mov`/`setp` preamble handled separately by the caller): from a machine
    state at the loop head coupled (via uniform registers) to `dst0/src0/len0/i`,
    running to completion reaches the exit pc with `gmem` updated exactly as the
    pure block model `cpLoop` computes, and every register outside the loop's
    scratch set preserved. -/
theorem coopCopy_loop (prog : Array SInstr) (base : Nat) (lH lX dst src len : String)
    (dst0 src0 len0 : Nat)
    (hseg : SegAt prog base (uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)))
    (hlr : LabelsResolve prog base (uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)))
    (hdst : dst ∉ coopCopyScratch) (hsrc : src ∉ coopCopyScratch) (hlenn : len ∉ coopCopyScratch)
    (hb1 : dst0 < 2 ^ 32) (hb2 : src0 < 2 ^ 32) (hb3 : len0 < 2 ^ 32) :
    ∀ (fuel i : Nat) (ss : SState),
      ss.pc = base → i ≤ len0 + 32 →
      (∀ l : Fin 32, ss.regs "cpI" l = UInt64.ofNat i) →
      (∀ l : Fin 32, ss.regs "cpCont" l = if i < len0 then 1 else 0) →
      (∀ l : Fin 32, ss.regs dst l = UInt64.ofNat dst0) →
      (∀ l : Fin 32, ss.regs src l = UInt64.ofNat src0) →
      (∀ l : Fin 32, ss.regs len l = UInt64.ofNat len0) →
      (∀ l : Fin 32, ss.regs "lane" l = UInt64.ofNat l.val) →
      len0 ≤ i + 32 * fuel →
      ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
        ss'.pc = base + (uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)).length ∧
        ss'.gmem = cpLoop dst0 src0 len0 fuel i ss.gmem ∧ ss'.smem = ss.smem ∧
        (∀ r : String, r ∉ coopCopyScratch → ∀ l : Fin 32, ss'.regs r l = ss.regs r l) := by
  -- Peel the (fuel-independent) layout, mirroring `simSL_uwhile`.
  obtain ⟨hlblH, hsegA⟩ := hseg.cons
  obtain ⟨hbrn, hsegB⟩ := hsegA.cons
  have hsegBody : SegAt prog (base + 1 + 1) (coopCopyBody dst src len) := hsegB.append_left
  obtain ⟨hbra, hsegD⟩ := hsegB.append_right.cons
  obtain ⟨hlblE, _⟩ := hsegD.cons
  have hLhead : sfindLabel prog lH = base := by
    have := hlr 0 lH (by simp [uwhileEmit]); simpa using this
  have hLend : sfindLabel prog lX = base + 1 + 1 + (coopCopyBody dst src len).length + 1 :=
    hlr.cons.cons.append_right.cons 0 lX (by simp)
  -- Body instruction facts, anchored at `base + 2` (the loop-head body position).
  have hbb : base + 1 + 1 = base + 2 := by omega
  rw [hbb] at hsegBody
  have h0 : prog[base + 2]? = some (.binr .add "cpDo" dst "cpI") := hsegBody 0 (by rw [coopCopyBody_length]; omega)
  have h1 : prog[base + 2 + 1]? = some (.binr .add "cpDo" "cpDo" "lane") := hsegBody 1 (by rw [coopCopyBody_length]; omega)
  have h2 : prog[base + 2 + 2]? = some (.binr .add "cpSo" src "cpI") := hsegBody 2 (by rw [coopCopyBody_length]; omega)
  have h3 : prog[base + 2 + 3]? = some (.binr .add "cpSo" "cpSo" "lane") := hsegBody 3 (by rw [coopCopyBody_length]; omega)
  have h4 : prog[base + 2 + 4]? = some (.binr .add "cpJ" "cpI" "lane") := hsegBody 4 (by rw [coopCopyBody_length]; omega)
  have h5 : prog[base + 2 + 5]? = some (.setp .lt "cpP" "cpJ" (.reg len)) := hsegBody 5 (by rw [coopCopyBody_length]; omega)
  have h6 : prog[base + 2 + 6]? = some (.ldgo "cpB" "cpSo" 0) := hsegBody 6 (by rw [coopCopyBody_length]; omega)
  have h7 : prog[base + 2 + 7]? = some (.stgp "cpP" "cpDo" "cpB") := hsegBody 7 (by rw [coopCopyBody_length]; omega)
  have h8 : prog[base + 2 + 8]? = some (.bin .add "cpI" "cpI" (.imm 32)) := hsegBody 8 (by rw [coopCopyBody_length]; omega)
  have h9 : prog[base + 2 + 9]? = some (.setp .lt "cpCont" "cpI" (.reg len)) := hsegBody 9 (by rw [coopCopyBody_length]; omega)
  intro fuel
  induction fuel with
  | zero =>
    intro i ss hpc hib hcpI hcpCont hdstv hsrcv hlenv hlane hfuel
    have hguard : ¬ i < len0 := by
      simp only [Nat.mul_zero, Nat.add_zero] at hfuel; omega
    have hcv : ss.regs "cpCont" 0 = (0:UInt64) := by
      rw [hcpCont]; simp [hguard]
    have hlblH' : prog[ss.pc]? = some (.lbl lH) := by rw [hpc]; exact hlblH
    have s0 : sstep prog ss = ss.setPc (ss.pc + 1) := by rw [lbl_step prog ss lH hlblH']
    have hbrn'' : prog[(ss.setPc (ss.pc+1)).pc]? = some (.braifnot "cpCont" lX) := by
      simp only [SState.setPc]; rw [hpc]; exact hbrn
    have s1 : sstep prog (ss.setPc (ss.pc + 1)) = (ss.setPc (ss.pc+1)).setPc (sfindLabel prog lX) := by
      rw [braifnot_step prog _ "cpCont" lX hbrn'']
      simp only [SState.setPc]
      rw [show ss.regs "cpCont" 0 = (0:UInt64) from hcv]
      rfl
    have hLendPc : sfindLabel prog lX = base + 1 + 1 + (coopCopyBody dst src len).length + 1 := hLend
    have hlblE' : prog[((ss.setPc (ss.pc+1)).setPc (sfindLabel prog lX)).pc]? = some (.lbl lX) := by
      simp only [SState.setPc]; rw [hLendPc]; exact hlblE
    have s2 : sstep prog ((ss.setPc (ss.pc+1)).setPc (sfindLabel prog lX))
        = (((ss.setPc (ss.pc+1)).setPc (sfindLabel prog lX))).setPc
            (base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1) := by
      rw [lbl_step prog _ lX hlblE']
      simp only [SState.setPc]
      rw [hLendPc]
    refine ⟨3, (((ss.setPc (ss.pc+1)).setPc (sfindLabel prog lX))).setPc
        (base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1), ?_, ?_, ?_, ?_, ?_⟩
    · exact sreaches_trans prog 2 1 _ _ _
        (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq s0) (sreaches_one_eq s1)) (sreaches_one_eq s2)
    · show (base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1)
        = base + (uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)).length
      simp [uwhileEmit_length]; omega
    · show cpLoop dst0 src0 len0 0 i ss.gmem = ss.gmem
      simp [cpLoop]
    · rfl
    · intro r hr l; rfl
  | succ fuel ih =>
    intro i ss hpc hib hcpI hcpCont hdstv hsrcv hlenv hlane hfuel
    by_cases hb : i < len0
    · -- guard true: head, braifnot(true), 10-instr body, back to head; recurse.
      have hcv : ss.regs "cpCont" 0 = (1:UInt64) := by rw [hcpCont]; simp [hb]
      have hlblH' : prog[ss.pc]? = some (.lbl lH) := by rw [hpc]; exact hlblH
      have s0 : sstep prog ss = ss.setPc (ss.pc + 1) := by rw [lbl_step prog ss lH hlblH']
      have hbrn'' : prog[(ss.setPc (ss.pc+1)).pc]? = some (.braifnot "cpCont" lX) := by
        simp only [SState.setPc]; rw [hpc]; exact hbrn
      have s1 : sstep prog (ss.setPc (ss.pc + 1)) = (ss.setPc (ss.pc+1)).setPc (ss.pc + 2) := by
        rw [braifnot_step prog _ "cpCont" lX hbrn'']
        simp only [SState.setPc]
        rw [show ss.regs "cpCont" 0 = (1:UInt64) from hcv]
        rfl
      have hpcbody : ((ss.setPc (ss.pc+1)).setPc (ss.pc+2)).pc = base + 2 := by
        simp only [SState.setPc]; rw [hpc]
      -- 10-instruction body via `coopCopy_iter`.
      generalize hssB : (ss.setPc (ss.pc+1)).setPc (ss.pc+2) = ssB
      rw [hssB] at s1
      have hpcbodyB : ssB.pc = base + 2 := by rw [← hssB]; exact hpcbody
      have hssBsmem : ssB.smem = ss.smem := by rw [← hssB]; simp only [SState.setPc]
      have hssBgmem : ssB.gmem = ss.gmem := by rw [← hssB]; simp only [SState.setPc]
      have hssBregs : ∀ (r : String) (l : Fin 32), ssB.regs r l = ss.regs r l := by
        intro r l; rw [← hssB]; simp only [SState.setPc]
      have hiter := coopCopy_iter prog ssB dst src len i dst0 src0 len0
        (by rw [hpcbodyB]; exact h0) (by rw [hpcbodyB]; exact h1) (by rw [hpcbodyB]; exact h2)
        (by rw [hpcbodyB]; exact h3) (by rw [hpcbodyB]; exact h4) (by rw [hpcbodyB]; exact h5)
        (by rw [hpcbodyB]; exact h6) (by rw [hpcbodyB]; exact h7) (by rw [hpcbodyB]; exact h8)
        (by rw [hpcbodyB]; exact h9)
        hdst hsrc hlenn
        (fun l => by show ssB.regs "cpI" l = UInt64.ofNat i; rw [← hssB]; exact hcpI l)
        (fun l => by show ssB.regs dst l = UInt64.ofNat dst0; rw [← hssB]; exact hdstv l)
        (fun l => by show ssB.regs src l = UInt64.ofNat src0; rw [← hssB]; exact hsrcv l)
        (fun l => by show ssB.regs len l = UInt64.ofNat len0; rw [← hssB]; exact hlenv l)
        (fun l => by show ssB.regs "lane" l = UInt64.ofNat l.val; rw [← hssB]; exact hlane l)
        hb1 hb2 hb3 hib
      obtain ⟨hiterPc, hiterSmem, hiterGmem, hiterCpI, hiterCpCont, hiterFrame⟩ := hiter
      have hbra' : prog[(snsteps prog 10 ssB).pc]? = some (.bra lH) := by
        rw [hiterPc, hpcbodyB]; exact hbra
      have s2 : sstep prog (snsteps prog 10 ssB) = (snsteps prog 10 ssB).setPc (sfindLabel prog lH) := by
        rw [bra_step prog _ lH hbra']
      have s2' : sstep prog (snsteps prog 10 ssB) = (snsteps prog 10 ssB).setPc base := by
        rw [s2, hLhead]
      -- Recurse via `ih` from the post-block state, offset `i + 32`.
      have hrec := ih (i + 32) ((snsteps prog 10 ssB).setPc base)
        (by simp only [SState.setPc])
        (by omega)
        (fun l => by simp only [SState.setPc]; exact hiterCpI l)
        (fun l => by simp only [SState.setPc]; exact hiterCpCont l)
        (fun l => by
          simp only [SState.setPc]
          rw [hiterFrame dst hdst l, hssBregs]; exact hdstv l)
        (fun l => by
          simp only [SState.setPc]
          rw [hiterFrame src hsrc l, hssBregs]; exact hsrcv l)
        (fun l => by
          simp only [SState.setPc]
          rw [hiterFrame len hlenn l, hssBregs]; exact hlenv l)
        (fun l => by
          simp only [SState.setPc]
          rw [hiterFrame "lane" (by decide) l, hssBregs]; exact hlane l)
        (by
          have hfuel32 : len0 ≤ (i + 32) + 32 * fuel := by
            have : 32 * (fuel + 1) = 32 * fuel + 32 := by omega
            omega
          exact hfuel32)
      obtain ⟨n, ssf, hrf, hpcf, hgf, hsf, hff⟩ := hrec
      refine ⟨1 + 1 + 10 + 1 + n, ssf, ?_, hpcf, ?_, ?_, ?_⟩
      · exact sreaches_trans prog (1 + 1 + 10 + 1) n _ _ _
          (sreaches_trans prog (1 + 1 + 10) 1 _ _ _
            (sreaches_trans prog (1 + 1) 10 _ _ _
              (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq s0) (sreaches_one_eq s1))
              (sreaches_snsteps prog 10 ssB))
            (sreaches_one_eq s2'))
          hrf
      · rw [hgf]
        simp only [SState.setPc]
        rw [hiterGmem, hssBgmem, cpLoop, if_pos hb]
      · rw [hsf]
        simp only [SState.setPc]
        rw [hiterSmem]; exact hssBsmem
      · intro r hr l
        rw [hff r hr]
        simp only [SState.setPc]
        rw [hiterFrame r hr l]; exact hssBregs r l
    · -- guard false but fuel wasn't exhausted: exit anyway (same as `zero` case).
      have hcv : ss.regs "cpCont" 0 = (0:UInt64) := by rw [hcpCont]; simp [hb]
      have hlblH' : prog[ss.pc]? = some (.lbl lH) := by rw [hpc]; exact hlblH
      have s0 : sstep prog ss = ss.setPc (ss.pc + 1) := by rw [lbl_step prog ss lH hlblH']
      have hbrn'' : prog[(ss.setPc (ss.pc+1)).pc]? = some (.braifnot "cpCont" lX) := by
        simp only [SState.setPc]; rw [hpc]; exact hbrn
      have s1 : sstep prog (ss.setPc (ss.pc + 1)) = (ss.setPc (ss.pc+1)).setPc (sfindLabel prog lX) := by
        rw [braifnot_step prog _ "cpCont" lX hbrn'']
        simp only [SState.setPc]
        rw [show ss.regs "cpCont" 0 = (0:UInt64) from hcv]
        rfl
      have hlblE' : prog[((ss.setPc (ss.pc+1)).setPc (sfindLabel prog lX)).pc]? = some (.lbl lX) := by
        simp only [SState.setPc]; rw [hLend]; exact hlblE
      have s2 : sstep prog ((ss.setPc (ss.pc+1)).setPc (sfindLabel prog lX))
          = (((ss.setPc (ss.pc+1)).setPc (sfindLabel prog lX))).setPc
              (base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1) := by
        rw [lbl_step prog _ lX hlblE']
        simp only [SState.setPc]
        rw [hLend]
      refine ⟨3, (((ss.setPc (ss.pc+1)).setPc (sfindLabel prog lX))).setPc
          (base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1), ?_, ?_, ?_, ?_, ?_⟩
      · exact sreaches_trans prog 2 1 _ _ _
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq s0) (sreaches_one_eq s1)) (sreaches_one_eq s2)
      · show (base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1)
          = base + (uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)).length
        simp [uwhileEmit_length]; omega
      · simp only [SState.setPc]
        rw [cpLoop, if_neg hb]
      · rfl
      · intro r hr l; rfl

-- ── Top-level `Couple`-preserving leaf ───────────────────────────────────────

/-- **`Couple` leaf for `.coopCopy`**: from a `Couple`d machine/model pair at the
    segment start, the full emitted segment (2-instruction preamble +
    `uwhileEmit`-lowered strided loop) reaches a state coupled to
    `WStmt.eval fuel (.coopCopy dst src len) ws`. Disjointness/size are the
    real-kernel facts that make `.coopCopy`'s literal-region copies well-defined
    (matching `cpLoop_eq_copyGmem`'s hypotheses exactly). -/
theorem coopCopy_couple (R : List String) (dst src len lH lX : String)
    (prog : Array SInstr) (base : Nat) (ss : SState) (ws : WState) (fuel : Nat)
    (hpc : ss.pc = base)
    (hseg : SegAt prog base
      (([.mov "cpI" (.imm 0), .setp .lt "cpCont" "cpI" (.reg len)] : List SInstr)
        ++ uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)))
    (hlr : LabelsResolve prog base
      (([.mov "cpI" (.imm 0), .setp .lt "cpCont" "cpI" (.reg len)] : List SInstr)
        ++ uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)))
    (hc : Couple R ss ws)
    (hdst : dst ∈ R) (hsrc : src ∈ R) (hlenR : len ∈ R)
    (hRdisj : ∀ r ∈ R, r ∉ coopCopyScratch)
    (hlane : ∀ l : Fin 32, ss.regs "lane" l = UInt64.ofNat l.val)
    (hb1 : (ws.regs dst).toNat < 2 ^ 32) (hb2 : (ws.regs src).toNat < 2 ^ 32)
    (hb3 : (ws.regs len).toNat < 2 ^ 32)
    (hdisj : (ws.regs dst).toNat + (ws.regs len).toNat ≤ (ws.regs src).toNat
      ∨ (ws.regs src).toNat + (ws.regs len).toNat ≤ (ws.regs dst).toNat)
    (hsize : (ws.regs dst).toNat + (ws.regs len).toNat ≤ ws.gmem.size) :
    ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
      ss'.pc = base + (([.mov "cpI" (.imm 0), .setp .lt "cpCont" "cpI" (.reg len)] : List SInstr)
        ++ uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)).length ∧
      Couple R ss' (WStmt.eval fuel (.coopCopy dst src len) ws) ∧
      -- register frame (for `r ∉ coopCopyScratch`): exposes lane/inBase/tbl preservation.
      (∀ r : String, r ∉ coopCopyScratch → ∀ l : Fin 32, ss'.regs r l = ss.regs r l) := by
  subst hpc
  -- Peel the 2-instruction preamble off the segment/label facts.
  obtain ⟨h0, hsegA⟩ := hseg.cons
  obtain ⟨h1, hsegB⟩ := hsegA.cons
  have hsegLoop : SegAt prog (ss.pc + 1 + 1) (uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)) :=
    hsegB
  have hlrLoop : LabelsResolve prog (ss.pc + 1 + 1)
      (uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)) :=
    hlr.cons.cons
  have hlenI : len ≠ "cpI" := fun h => hRdisj len hlenR (h ▸ by simp [coopCopyScratch])
  -- Step 1: `mov "cpI" (.imm 0)`.
  have s0 : sstep prog ss = (ss.setReg "cpI" (fun _ => 0)).setPc (ss.pc + 1) := by
    simp only [sstep, h0, sstepInstr, SState.get]; congr 1
  -- Step 2: `setp .lt "cpCont" "cpI" (.reg len)`.
  have s1 : sstep prog ((ss.setReg "cpI" (fun _ => 0)).setPc (ss.pc + 1))
      = (((ss.setReg "cpI" (fun _ => 0)).setPc (ss.pc + 1)).setReg "cpCont"
          (fun l => if SCmp.run .lt 0 (ss.regs len l) then 1 else 0)).setPc (ss.pc + 1 + 1) := by
    simp only [sstep, SState.setPc, h1, sstepInstr, SState.get, SState.setReg, if_neg hlenI]
    congr 1
  obtain ⟨ss2, hss2⟩ : ∃ x : SState, x = (((ss.setReg "cpI" (fun _ => 0)).setPc (ss.pc + 1)).setReg
      "cpCont" (fun l => if SCmp.run .lt 0 (ss.regs len l) then 1 else 0)).setPc (ss.pc + 1 + 1) :=
    ⟨_, rfl⟩
  have hstep2 : sstep prog (sstep prog ss) = ss2 := by rw [s0, s1, hss2]
  -- Register/gmem/smem facts at `ss2`, feeding `coopCopy_loop`.
  have hpc2 : ss2.pc = ss.pc + 1 + 1 := by rw [hss2]; simp [SState.setPc]
  have hcpI2 : ∀ l : Fin 32, ss2.regs "cpI" l = UInt64.ofNat 0 := by
    intro l; rw [hss2]; simp [SState.setPc, SState.setReg]
  have hcpCont2 : ∀ l : Fin 32, ss2.regs "cpCont" l
      = if 0 < (ws.regs len).toNat then 1 else 0 := by
    intro l
    rw [hss2]
    simp [SState.setPc, SState.setReg]
    have hlv : ss.regs len l = ws.regs len := hc.reg hlenR l
    rw [hlv]
    simp [SCmp.run, UInt64.lt_iff_toNat_lt]
  -- General frame: any register other than `cpI`/`cpCont` is untouched by the preamble.
  have hgen2 : ∀ r : String, r ≠ "cpCont" → r ≠ "cpI" → ∀ l : Fin 32, ss2.regs r l = ss.regs r l := by
    intro r hrC hrI l
    rw [hss2]
    simp only [SState.setPc, SState.setReg, if_neg hrC, if_neg hrI]
  have hdst2 : ∀ l : Fin 32, ss2.regs dst l = UInt64.ofNat (ws.regs dst).toNat := by
    intro l
    rw [hgen2 dst (fun h => hRdisj dst hdst (h ▸ by simp [coopCopyScratch]))
      (fun h => hRdisj dst hdst (h ▸ by simp [coopCopyScratch])) l,
      UInt64.ofNat_toNat]
    exact hc.reg hdst l
  have hsrc2 : ∀ l : Fin 32, ss2.regs src l = UInt64.ofNat (ws.regs src).toNat := by
    intro l
    rw [hgen2 src (fun h => hRdisj src hsrc (h ▸ by simp [coopCopyScratch]))
      (fun h => hRdisj src hsrc (h ▸ by simp [coopCopyScratch])) l,
      UInt64.ofNat_toNat]
    exact hc.reg hsrc l
  have hlen2 : ∀ l : Fin 32, ss2.regs len l = UInt64.ofNat (ws.regs len).toNat := by
    intro l
    rw [hgen2 len (fun h => hRdisj len hlenR (h ▸ by simp [coopCopyScratch]))
      (fun h => hRdisj len hlenR (h ▸ by simp [coopCopyScratch])) l,
      UInt64.ofNat_toNat]
    exact hc.reg hlenR l
  have hlane2 : ∀ l : Fin 32, ss2.regs "lane" l = UInt64.ofNat l.val := by
    intro l
    rw [hgen2 "lane" (by decide) (by decide) l]
    exact hlane l
  have hgmem2 : ss2.gmem = ws.gmem := by
    rw [hss2]; simp only [SState.setPc, SState.setReg]; exact hc.gmem
  have hsmem2 : ss2.smem = ws.smem := by
    rw [hss2]; simp only [SState.setPc, SState.setReg]; exact hc.smem
  -- Fuel for `coopCopy_loop`: `len0` iterations suffice (`len0 ≤ 0 + 32*len0`).
  obtain ⟨n, ss', hreach, hpcE, hgmemE, hsmemE, hframeE⟩ :=
    coopCopy_loop prog (ss.pc + 1 + 1) lH lX dst src len
      (ws.regs dst).toNat (ws.regs src).toNat (ws.regs len).toNat
      hsegLoop hlrLoop
      (hRdisj dst hdst) (hRdisj src hsrc) (hRdisj len hlenR)
      hb1 hb2 hb3
      (ws.regs len).toNat 0 ss2 hpc2 (by omega)
      hcpI2 hcpCont2 hdst2 hsrc2 hlen2 hlane2 (by omega)
  refine ⟨n + 2, ss', ?_, ?_, ⟨?_, ?_, ?_⟩, ?_⟩
  · show SReaches prog n (sstep prog (sstep prog ss)) ss'
    rw [hstep2]; exact hreach
  · rw [hpcE]
    simp only [List.length_append, List.length_cons, List.length_nil,
      uwhileEmit_length, coopCopyBody_length]
    try omega
  · show ss'.gmem = (WStmt.eval fuel (.coopCopy dst src len) ws).gmem
    simp only [WStmt.eval]
    rw [hgmemE, hgmem2]
    exact cpLoop_eq_copyGmem (ws.regs dst).toNat (ws.regs src).toNat (ws.regs len).toNat
      ws.gmem hdisj hsize (ws.regs len).toNat (by omega)
  · show ss'.smem = (WStmt.eval fuel (.coopCopy dst src len) ws).smem
    simp only [WStmt.eval]
    rw [hsmemE, hsmem2]
  · intro r hr l
    have hrC : r ≠ "cpCont" := fun h => hRdisj r hr (h ▸ by simp [coopCopyScratch])
    have hrI : r ≠ "cpI" := fun h => hRdisj r hr (h ▸ by simp [coopCopyScratch])
    show ss'.regs r l = (WStmt.eval fuel (.coopCopy dst src len) ws).regs r
    simp only [WStmt.eval]
    rw [hframeE r (hRdisj r hr) l, hgen2 r hrC hrI l]
    exact hc.reg hr l
  · -- register frame for `r ∉ coopCopyScratch`, threaded from `coopCopy_loop` + preamble.
    intro r hr l
    have hrC : r ≠ "cpCont" := fun h => hr (h ▸ by simp [coopCopyScratch])
    have hrI : r ≠ "cpI" := fun h => hr (h ▸ by simp [coopCopyScratch])
    rw [hframeE r hr l, hgen2 r hrC hrI l]

end AlgorithmLib.LZ4WarpDSL
