import AlgorithmLib.U32Field
import AlgorithmLib.EvalValid
import AlgorithmLib.LZ4SimtRSimComp

namespace AlgorithmLib.LZ4WarpDSL

open AlgorithmLib AlgorithmLib.LZ4Simt

/-- `warpKernelDSL` agrees with `prologueInstrs` on the prologue prefix (indices
    `< 39`), *without* evaluating the emitted body.  Uses only append/prefix
    lemmas so the tail `compressorBodyEmit` is never unfolded. -/
theorem wk_prefix (nb iS oS lO hL k : Nat)
    (hk : k < (prologueInstrs nb iS oS hL).length) :
    (warpKernelDSL nb iS oS lO hL)[k]? = (prologueInstrs nb iS oS hL)[k]? := by
  have hk2 : k < (prologueInstrs nb iS oS hL ++ compressorBodyEmit iS lO hL).length := by
    rw [List.length_append]; omega
  simp only [warpKernelDSL, List.getElem?_toArray,
    List.getElem?_append_left hk, List.getElem?_append_left hk2]

/-- The prologue has 39 instructions (`loop` label at index 38). -/
theorem prologueInstrs_length (nb iS oS hL : Nat) :
    (prologueInstrs nb iS oS hL).length = 39 := by rfl

/-- `lane = tid & 31` for a warp (variant with `UInt64.ofNat 31`, matching `SOp.run`). -/
theorem u64_and31' : ∀ l : Fin 32, UInt64.ofNat l.val &&& UInt64.ofNat 31 = UInt64.ofNat l.val := by decide
/-- `tid >> 5 = 0` for a warp (variant with `UInt64.ofNat 5`, matching `SOp.run`). -/
theorem u64_shr5' : ∀ l : Fin 32, UInt64.ofNat l.val >>> UInt64.ofNat 5 = 0 := by decide

theorem u64_ofNat0 : UInt64.ofNat 0 = 0 := rfl

/-- A single prologue index fact for `warpKernelDSL`, reduced to a `prologueInstrs`
    lookup and closed by `rfl`. -/
theorem wk_idx (nb iS oS lO hL k : Nat) (i : SInstr) (hk : k < 39)
    (h : (prologueInstrs nb iS oS hL)[k]? = some i) :
    (warpKernelDSL nb iS oS lO hL)[k]? = some i := by
  rw [wk_prefix nb iS oS lO hL k (by rw [prologueInstrs_length]; omega)]; exact h

/-- `warpKernelDSL` agrees with `compressorBodyEmit` on the body (indices `≥ 39`),
    without evaluating the body: pure append indexing. -/
theorem wk_body (nb iS oS lO hL i : Nat) (hi : i < (compressorBodyEmit iS lO hL).length) :
    (warpKernelDSL nb iS oS lO hL)[39 + i]? = (compressorBodyEmit iS lO hL)[i]? := by
  have h39 : (prologueInstrs nb iS oS hL).length = 39 := prologueInstrs_length nb iS oS hL
  have hlt : 39 + i < (prologueInstrs nb iS oS hL ++ compressorBodyEmit iS lO hL).length := by
    rw [List.length_append, h39]; omega
  have hge : (prologueInstrs nb iS oS hL).length ≤ 39 + i := by rw [h39]; omega
  simp only [warpKernelDSL, List.getElem?_toArray]
  rw [List.getElem?_append_left hlt, List.getElem?_append_right hge,
    show 39 + i - (prologueInstrs nb iS oS hL).length = i from by rw [h39]; omega]

/-- Combine two adjacent `SegAt`s (contiguous layout) into one over the concatenation. -/
theorem segAt_append_intro {prog : Array SInstr} {base : Nat} {A B : List SInstr}
    (hA : SegAt prog base A) (hB : SegAt prog (base + A.length) B) : SegAt prog base (A ++ B) := by
  intro i hi
  rw [List.length_append] at hi
  by_cases h : i < A.length
  · rw [List.getElem?_append_left h]; exact hA i h
  · rw [List.getElem?_append_right (by omega)]
    have hb := hB (i - A.length) (by omega)
    rwa [show base + A.length + (i - A.length) = base + i from by omega] at hb

/-- The emitted body has 233 instructions. -/
theorem bodyLen233 (iS lO hL : Nat) : (compressorBodyEmit iS lO hL).length = 233 := rfl

/-- A body leaf of known length `len` sits at `warpKernelDSL` index `base = 39 + off`
    when it equals the body's slice there — a *small* `rfl` on the slice (never the
    whole body).  Explicit `base`/`len` so it composes with `segAt_append_intro'`. -/
theorem segAt_body_leaf (nb iS oS lO hL base off len : Nat) (leaf : List SInstr)
    (hbase : base = 39 + off) (hlen : leaf.length = len)
    (hbound : off + len ≤ (compressorBodyEmit iS lO hL).length)
    (hslice : ((compressorBodyEmit iS lO hL).drop off).take len = leaf) :
    SegAt (warpKernelDSL nb iS oS lO hL) base leaf := by
  subst hbase
  intro j hj
  rw [hlen] at hj
  rw [show 39 + off + j = 39 + (off + j) from by omega, wk_body nb iS oS lO hL (off + j) (by omega),
    ← hslice, List.getElem?_take_of_lt hj, List.getElem?_drop]

/-- Peel one instruction: a leading `a` sits at `base` when the body has it at `off`
    (a fast per-index `rfl`), and the rest follows. -/
theorem segAt_body_cons (nb iS oS lO hL base off : Nat) (a : SInstr) (rest : List SInstr)
    (hbase : base = 39 + off)
    (hhead : (compressorBodyEmit iS lO hL)[off]? = some a)
    (hrest : SegAt (warpKernelDSL nb iS oS lO hL) (base + 1) rest) :
    SegAt (warpKernelDSL nb iS oS lO hL) base (a :: rest) := by
  have hoff : off < (compressorBodyEmit iS lO hL).length := by
    rw [List.getElem?_eq_some_iff] at hhead; exact hhead.1
  intro i hi
  cases i with
  | zero => subst hbase; simp only [Nat.add_zero, List.getElem?_cons_zero]
            rw [wk_body nb iS oS lO hL off hoff]; exact hhead
  | succ k => simp only [List.getElem?_cons_succ]
              rw [show base + (k + 1) = base + 1 + k from by omega]
              exact hrest k (by simp only [List.length_cons] at hi; omega)

/-- `segAt_append_intro` with the left length supplied (so the right base is concrete). -/
theorem segAt_append_intro' {prog : Array SInstr} {base : Nat} {A B : List SInstr} (lenA : Nat)
    (hlenA : A.length = lenA) (hA : SegAt prog base A) (hB : SegAt prog (base + lenA) B) :
    SegAt prog base (A ++ B) := by subst hlenA; exact segAt_append_intro hA hB

-- ── LabelsResolve combinators (mirror the SegAt ones) ─────────────────────────

theorem labelsResolve_nil (prog : Array SInstr) (base : Nat) : LabelsResolve prog base [] := by
  intro k name hk; simp at hk

/-- Peel a leading label: it resolves here (`sfindLabel = base`), the rest follows. -/
theorem labelsResolve_cons_lbl {prog : Array SInstr} {base : Nat} {name : String} {rest : List SInstr}
    (hl : sfindLabel prog name = base) (hrest : LabelsResolve prog (base + 1) rest) :
    LabelsResolve prog base (.lbl name :: rest) := by
  intro k nm hk
  cases k with
  | zero => simp only [List.getElem?_cons_zero, Option.some.injEq, SInstr.lbl.injEq] at hk
            subst hk; simpa using hl
  | succ j => simp only [List.getElem?_cons_succ] at hk
              have := hrest j nm hk; omega

/-- Peel a leading non-label instruction. -/
theorem labelsResolve_cons_other {prog : Array SInstr} {base : Nat} {a : SInstr} {rest : List SInstr}
    (ha : ∀ n, a ≠ .lbl n) (hrest : LabelsResolve prog (base + 1) rest) :
    LabelsResolve prog base (a :: rest) := by
  intro k nm hk
  cases k with
  | zero => simp only [List.getElem?_cons_zero, Option.some.injEq] at hk; exact absurd hk (ha nm)
  | succ j => simp only [List.getElem?_cons_succ] at hk
              have := hrest j nm hk; omega

/-- The registers written by `prologueInstrs` (indices 0–37).  Every loop register
    except `{inBase,outBase,op,litAnchor,searchPos}` is *outside* this set, so the
    frame invariant hands us their uniform launch value. -/
def prologueW : List String :=
  ["inP", "outP", "tid", "ctab", "ntid", "gtid", "gwarp", "lane", "lwarp", "oob",
   "pL0", "gwD", "inOff", "outOff", "inBase", "outBase", "smem", "tblOff", "tbl",
   "ci", "z", "pcnd", "ca", "litAnchor", "searchPos", "op"]

/-- **Head slice A** (indices 0–11): compute the thread ids and pass the `oob` guard
    (not taken since `gwarp = 0 < nb`).  Exposes the raw values (`inP = 0`, `outP = iS`,
    `gwarp = lwarp = 0`, lane identity) that slice B consumes. -/
theorem headA (nb iS oS lO hL w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hnb : 0 < nb) (hnb2 : nb < 2 ^ 64) (hw : w < nb) (hw64 : w * 32 + 32 < 2 ^ 64) :
    let prog := warpKernelDSL nb iS oS lO hL
    let s0 := initSt w inPtr outPtr gm smemB
    (snsteps prog 12 s0).pc = 12 ∧
      (∀ l : Fin 32, (snsteps prog 12 s0).regs "inP" l = UInt64.ofNat inPtr) ∧
      (∀ l : Fin 32, (snsteps prog 12 s0).regs "outP" l = UInt64.ofNat outPtr) ∧
      (∀ l : Fin 32, (snsteps prog 12 s0).regs "gwarp" l = UInt64.ofNat w) ∧
      (∀ l : Fin 32, (snsteps prog 12 s0).regs "lwarp" l = 0) ∧
      (∀ l : Fin 32, (snsteps prog 12 s0).regs rLane l = UInt64.ofNat l.val) ∧
      (snsteps prog 12 s0).gmem = s0.gmem ∧
      (∀ r, r ∉ prologueW → (snsteps prog 12 s0).regs r = s0.regs r) := by
  intro prog s0
  have i0 : prog[0]? = some (.mov "inP" (.reg rInPtr)) := wk_idx nb iS oS lO hL 0 _ (by omega) rfl
  have i1 : prog[1]? = some (.mov "outP" (.reg rOutPtr)) := wk_idx nb iS oS lO hL 1 _ (by omega) rfl
  have i2 : prog[2]? = some (.mov "tid" (.reg rTidX)) := wk_idx nb iS oS lO hL 2 _ (by omega) rfl
  have i3 : prog[3]? = some (.mov "ctab" (.reg rCtaX)) := wk_idx nb iS oS lO hL 3 _ (by omega) rfl
  have i4 : prog[4]? = some (.mov "ntid" (.reg rNtidX)) := wk_idx nb iS oS lO hL 4 _ (by omega) rfl
  have i5 : prog[5]? = some (.binr .mul "gtid" "ctab" "ntid") := wk_idx nb iS oS lO hL 5 _ (by omega) rfl
  have i6 : prog[6]? = some (.binr .add "gtid" "gtid" "tid") := wk_idx nb iS oS lO hL 6 _ (by omega) rfl
  have i7 : prog[7]? = some (.bin .shr "gwarp" "gtid" (.imm 5)) := wk_idx nb iS oS lO hL 7 _ (by omega) rfl
  have i8 : prog[8]? = some (.bin .band rLane "tid" (.imm 31)) := wk_idx nb iS oS lO hL 8 _ (by omega) rfl
  have i9 : prog[9]? = some (.bin .shr "lwarp" "tid" (.imm 5)) := wk_idx nb iS oS lO hL 9 _ (by omega) rfl
  have i10 : prog[10]? = some (.setp .ge "oob" "gwarp" (.imm nb)) := wk_idx nb iS oS lO hL 10 _ (by omega) rfl
  have i11 : prog[11]? = some (.braif "oob" "OOB") := wk_idx nb iS oS lO hL 11 _ (by omega) rfl
  have hpc0 : s0.pc = 0 := rfl
  have hregs : s0.regs = initRegs w inPtr outPtr := rfl
  have hgmem0 : s0.gmem = gm := rfl
  have hg0 : s0.gmem = s0.gmem := rfl
  simp only [snsteps]
  -- step 0
  rw [mov_step prog s0 "inP" (.reg rInPtr) (by rw [hpc0]; exact i0)]
  generalize hs1 : (s0.setReg "inP" (fun l => s0.get l (.reg rInPtr))).setPc (s0.pc + 1) = s1
  have hpc1 : s1.pc = 1 := by rw [← hs1]; simp [SState.setPc, hpc0]
  have hg1 : s1.gmem = s0.gmem := by rw [← hs1]; exact hg0
  rw [mov_step prog s1 "outP" (.reg rOutPtr) (by rw [hpc1]; exact i1)]
  generalize hs2 : (s1.setReg "outP" (fun l => s1.get l (.reg rOutPtr))).setPc (s1.pc + 1) = s2
  have hpc2 : s2.pc = 2 := by rw [← hs2]; simp [SState.setPc, hpc1]
  have hg2 : s2.gmem = s0.gmem := by rw [← hs2]; exact hg1
  rw [mov_step prog s2 "tid" (.reg rTidX) (by rw [hpc2]; exact i2)]
  generalize hs3 : (s2.setReg "tid" (fun l => s2.get l (.reg rTidX))).setPc (s2.pc + 1) = s3
  have hpc3 : s3.pc = 3 := by rw [← hs3]; simp [SState.setPc, hpc2]
  have hg3 : s3.gmem = s0.gmem := by rw [← hs3]; exact hg2
  rw [mov_step prog s3 "ctab" (.reg rCtaX) (by rw [hpc3]; exact i3)]
  generalize hs4 : (s3.setReg "ctab" (fun l => s3.get l (.reg rCtaX))).setPc (s3.pc + 1) = s4
  have hpc4 : s4.pc = 4 := by rw [← hs4]; simp [SState.setPc, hpc3]
  have hg4 : s4.gmem = s0.gmem := by rw [← hs4]; exact hg3
  rw [mov_step prog s4 "ntid" (.reg rNtidX) (by rw [hpc4]; exact i4)]
  generalize hs5 : (s4.setReg "ntid" (fun l => s4.get l (.reg rNtidX))).setPc (s4.pc + 1) = s5
  have hpc5 : s5.pc = 5 := by rw [← hs5]; simp [SState.setPc, hpc4]
  have hg5 : s5.gmem = s0.gmem := by rw [← hs5]; exact hg4
  rw [binr_step prog s5 .mul "gtid" "ctab" "ntid" (by rw [hpc5]; exact i5)]
  generalize hs6 : (s5.setReg "gtid" (fun l => SOp.run .mul (s5.regs "ctab" l) (s5.regs "ntid" l))).setPc (s5.pc + 1) = s6
  have hpc6 : s6.pc = 6 := by rw [← hs6]; simp [SState.setPc, hpc5]
  have hg6 : s6.gmem = s0.gmem := by rw [← hs6]; exact hg5
  rw [binr_step prog s6 .add "gtid" "gtid" "tid" (by rw [hpc6]; exact i6)]
  generalize hs7 : (s6.setReg "gtid" (fun l => SOp.run .add (s6.regs "gtid" l) (s6.regs "tid" l))).setPc (s6.pc + 1) = s7
  have hpc7 : s7.pc = 7 := by rw [← hs7]; simp [SState.setPc, hpc6]
  have hg7 : s7.gmem = s0.gmem := by rw [← hs7]; exact hg6
  rw [bin_step prog s7 .shr "gwarp" "gtid" (.imm 5) (by rw [hpc7]; exact i7)]
  generalize hs8 : (s7.setReg "gwarp" (fun l => SOp.run .shr (s7.regs "gtid" l) (s7.get l (.imm 5)))).setPc (s7.pc + 1) = s8
  have hpc8 : s8.pc = 8 := by rw [← hs8]; simp [SState.setPc, hpc7]
  have hg8 : s8.gmem = s0.gmem := by rw [← hs8]; exact hg7
  rw [bin_step prog s8 .band rLane "tid" (.imm 31) (by rw [hpc8]; exact i8)]
  generalize hs9 : (s8.setReg rLane (fun l => SOp.run .band (s8.regs "tid" l) (s8.get l (.imm 31)))).setPc (s8.pc + 1) = s9
  have hpc9 : s9.pc = 9 := by rw [← hs9]; simp [SState.setPc, hpc8]
  have hg9 : s9.gmem = s0.gmem := by rw [← hs9]; exact hg8
  rw [bin_step prog s9 .shr "lwarp" "tid" (.imm 5) (by rw [hpc9]; exact i9)]
  generalize hs10 : (s9.setReg "lwarp" (fun l => SOp.run .shr (s9.regs "tid" l) (s9.get l (.imm 5)))).setPc (s9.pc + 1) = s10
  have hpc10 : s10.pc = 10 := by rw [← hs10]; simp [SState.setPc, hpc9]
  have hg10 : s10.gmem = s0.gmem := by rw [← hs10]; exact hg9
  rw [setp_step prog s10 .ge "oob" "gwarp" (.imm nb) (by rw [hpc10]; exact i10)]
  generalize hs11 : (s10.setReg "oob" (fun l => if SCmp.run .ge (s10.regs "gwarp" l) (s10.get l (.imm nb)) then 1 else 0)).setPc (s10.pc + 1) = s11
  have hpc11 : s11.pc = 11 := by rw [← hs11]; simp [SState.setPc, hpc10]
  have hg11 : s11.gmem = s0.gmem := by rw [← hs11]; exact hg10
  -- braif oob OOB: oob 0 = 0 since gwarp 0 = 0 < nb, so the branch is NOT taken.
  have hgwarp0 : s10.regs "gwarp" 0 = UInt64.ofNat w := by
    rw [← hs10, ← hs9, ← hs8, ← hs7, ← hs6, ← hs5, ← hs4, ← hs3, ← hs2, ← hs1]
    simp only [SState.setReg, SState.setPc, SState.get, SOp.run, hregs, initRegs,
      rTidX, rCtaX, rNtidX, rInPtr, rOutPtr, rLane, String.reduceEq, reduceIte]
    exact u64_gwarp w hw64 0
  have hnotle : ¬ (UInt64.ofNat nb ≤ UInt64.ofNat w) := by
    rw [UInt64.le_iff_toNat_le, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt nb hnb2,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt w (by omega)]
    omega
  have hoob0 : s11.regs "oob" 0 = 0 := by
    rw [← hs11]
    simp only [SState.setReg, SState.setPc, SState.get, String.reduceEq, reduceIte]
    rw [hgwarp0]
    simp only [SCmp.run, decide_eq_false hnotle]
    decide
  rw [braif_step prog s11 "oob" "OOB" (by rw [hpc11]; exact i11)]
  rw [show (if s11.regs "oob" 0 == 1 then sfindLabel prog "OOB" else s11.pc + 1) = s11.pc + 1 by
    rw [hoob0]; rfl]
  generalize hs12 : s11.setPc (s11.pc + 1) = s12
  have hpc12 : s12.pc = 12 := by rw [← hs12]; simp [SState.setPc, hpc11]
  have hg12 : s12.gmem = s0.gmem := by rw [← hs12]; exact hg11
  subst hs12 hs11 hs10 hs9 hs8 hs7 hs6 hs5 hs4 hs3 hs2 hs1
  refine ⟨hpc12, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro l
    simp only [SState.setReg, SState.setPc, SState.get, SOp.run, hregs, initRegs,
      rLane, rTidX, rCtaX, rNtidX, rInPtr, rOutPtr,
      u64_and31, u64_and31', u64_shr5, u64_shr5', u64_gwarp w hw64, UInt64.zero_add, UInt64.add_zero,
      String.reduceEq, reduceIte]
  · intro l
    simp only [SState.setReg, SState.setPc, SState.get, SOp.run, hregs, initRegs,
      rLane, rTidX, rCtaX, rNtidX, rInPtr, rOutPtr,
      u64_and31, u64_and31', u64_shr5, u64_shr5', u64_gwarp w hw64, UInt64.zero_add, UInt64.add_zero,
      String.reduceEq, reduceIte]
  · intro l
    simp only [SState.setReg, SState.setPc, SState.get, SOp.run, hregs, initRegs,
      rLane, rTidX, rCtaX, rNtidX, rInPtr, rOutPtr,
      u64_and31, u64_and31', u64_shr5, u64_shr5', u64_gwarp w hw64, UInt64.zero_add, UInt64.add_zero,
      String.reduceEq, reduceIte]
  · intro l
    simp only [SState.setReg, SState.setPc, SState.get, SOp.run, hregs, initRegs,
      rLane, rTidX, rCtaX, rNtidX, rInPtr, rOutPtr,
      u64_and31, u64_and31', u64_shr5, u64_shr5', u64_gwarp w hw64, UInt64.zero_add, UInt64.add_zero,
      String.reduceEq, reduceIte]
  · intro l
    simp only [SState.setReg, SState.setPc, SState.get, SOp.run, hregs, initRegs,
      rLane, rTidX, rCtaX, rNtidX, rInPtr, rOutPtr,
      u64_and31, u64_and31', u64_shr5, u64_shr5', u64_gwarp w hw64, UInt64.zero_add, UInt64.add_zero,
      String.reduceEq, reduceIte]
  · exact hg12
  · intro r hr
    simp only [prologueW, List.mem_cons, List.mem_singleton, not_or] at hr
    obtain ⟨a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,
      a18, a19, a20, a21, a22, a23, a24, a25, a26⟩ := hr
    simp only [SState.setReg, SState.setPc, rLane,
      a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,
      a18, a19, a20, a21, a22, a23, a24, a25, a26, reduceIte]

/-- **Head slice B** (indices 12–24): from the post-guard state `s` (whose base values
    are given as hypotheses), set the base pointers and table cursor.  All computed
    values bottom out at the hypotheses, so extraction stays shallow. -/
theorem headB (nb iS oS lO hL w inPtr outPtr : Nat) (s : SState) (hpc : s.pc = 12)
    (hinP : ∀ l : Fin 32, s.regs "inP" l = UInt64.ofNat inPtr)
    (houtP : ∀ l : Fin 32, s.regs "outP" l = UInt64.ofNat outPtr)
    (hgw : ∀ l : Fin 32, s.regs "gwarp" l = UInt64.ofNat w)
    (hlw : ∀ l : Fin 32, s.regs "lwarp" l = 0)
    (hlaneS : ∀ l : Fin 32, s.regs rLane l = UInt64.ofNat l.val) :
    let prog := warpKernelDSL nb iS oS lO hL
    (snsteps prog 13 s).pc = 25 ∧
      (∀ l : Fin 32, (snsteps prog 13 s).regs rLane l = UInt64.ofNat l.val) ∧
      (∀ l : Fin 32, (snsteps prog 13 s).regs rInBase l = UInt64.ofNat (inPtr + w * iS)) ∧
      (∀ l : Fin 32, (snsteps prog 13 s).regs rTbl l = 0) ∧
      (∀ l : Fin 32, (snsteps prog 13 s).regs rOutBase l = UInt64.ofNat (outPtr + w * oS)) ∧
      (snsteps prog 13 s).regs "ci" 0 = 0 ∧
      (snsteps prog 13 s).gmem = s.gmem ∧
      (∀ r, r ∉ prologueW → (snsteps prog 13 s).regs r = s.regs r) := by
  intro prog
  have hlaneS' : ∀ l : Fin 32, s.regs "lane" l = UInt64.ofNat l.val := hlaneS
  have hgb12 : s.gmem = s.gmem := rfl
  have i12 : prog[12]? = some (.setp .eq rPL0 rLane (.imm 0)) := wk_idx nb iS oS lO hL 12 _ (by omega) rfl
  have i13 : prog[13]? = some (.mov "gwD" (.reg "gwarp")) := wk_idx nb iS oS lO hL 13 _ (by omega) rfl
  have i14 : prog[14]? = some (.mov "inOff" (.imm iS)) := wk_idx nb iS oS lO hL 14 _ (by omega) rfl
  have i15 : prog[15]? = some (.binr .mul "inOff" "gwD" "inOff") := wk_idx nb iS oS lO hL 15 _ (by omega) rfl
  have i16 : prog[16]? = some (.mov "outOff" (.imm oS)) := wk_idx nb iS oS lO hL 16 _ (by omega) rfl
  have i17 : prog[17]? = some (.binr .mul "outOff" "gwD" "outOff") := wk_idx nb iS oS lO hL 17 _ (by omega) rfl
  have i18 : prog[18]? = some (.binr .add rInBase "inP" "inOff") := wk_idx nb iS oS lO hL 18 _ (by omega) rfl
  have i19 : prog[19]? = some (.binr .add rOutBase "outP" "outOff") := wk_idx nb iS oS lO hL 19 _ (by omega) rfl
  have i20 : prog[20]? = some (.mov "smem" (.imm 0)) := wk_idx nb iS oS lO hL 20 _ (by omega) rfl
  have i21 : prog[21]? = some (.bin .mul "tblOff" "lwarp" (.imm ((2 ^ hL) * 2))) := wk_idx nb iS oS lO hL 21 _ (by omega) rfl
  have i22 : prog[22]? = some (.binr .add rTbl "smem" "tblOff") := wk_idx nb iS oS lO hL 22 _ (by omega) rfl
  have i23 : prog[23]? = some (.mov "ci" (.reg rLane)) := wk_idx nb iS oS lO hL 23 _ (by omega) rfl
  have i24 : prog[24]? = some (.mov "z" (.imm 0)) := wk_idx nb iS oS lO hL 24 _ (by omega) rfl
  simp only [snsteps]
  rw [setp_step prog s .eq rPL0 rLane (.imm 0) (by rw [hpc]; exact i12)]
  generalize hb13 : (s.setReg rPL0 (fun l => if SCmp.run .eq (s.regs rLane l) (s.get l (.imm 0)) then 1 else 0)).setPc (s.pc + 1) = b13
  have hpcb13 : b13.pc = 13 := by rw [← hb13]; simp [SState.setPc, hpc]
  have hgb13 : b13.gmem = s.gmem := by rw [← hb13]; exact hgb12
  rw [mov_step prog b13 "gwD" (.reg "gwarp") (by rw [hpcb13]; exact i13)]
  generalize hb14 : (b13.setReg "gwD" (fun l => b13.get l (.reg "gwarp"))).setPc (b13.pc + 1) = b14
  have hpcb14 : b14.pc = 14 := by rw [← hb14]; simp [SState.setPc, hpcb13]
  have hgb14 : b14.gmem = s.gmem := by rw [← hb14]; exact hgb13
  rw [mov_step prog b14 "inOff" (.imm iS) (by rw [hpcb14]; exact i14)]
  generalize hb15 : (b14.setReg "inOff" (fun l => b14.get l (.imm iS))).setPc (b14.pc + 1) = b15
  have hpcb15 : b15.pc = 15 := by rw [← hb15]; simp [SState.setPc, hpcb14]
  have hgb15 : b15.gmem = s.gmem := by rw [← hb15]; exact hgb14
  rw [binr_step prog b15 .mul "inOff" "gwD" "inOff" (by rw [hpcb15]; exact i15)]
  generalize hb16 : (b15.setReg "inOff" (fun l => SOp.run .mul (b15.regs "gwD" l) (b15.regs "inOff" l))).setPc (b15.pc + 1) = b16
  have hpcb16 : b16.pc = 16 := by rw [← hb16]; simp [SState.setPc, hpcb15]
  have hgb16 : b16.gmem = s.gmem := by rw [← hb16]; exact hgb15
  rw [mov_step prog b16 "outOff" (.imm oS) (by rw [hpcb16]; exact i16)]
  generalize hb17 : (b16.setReg "outOff" (fun l => b16.get l (.imm oS))).setPc (b16.pc + 1) = b17
  have hpcb17 : b17.pc = 17 := by rw [← hb17]; simp [SState.setPc, hpcb16]
  have hgb17 : b17.gmem = s.gmem := by rw [← hb17]; exact hgb16
  rw [binr_step prog b17 .mul "outOff" "gwD" "outOff" (by rw [hpcb17]; exact i17)]
  generalize hb18 : (b17.setReg "outOff" (fun l => SOp.run .mul (b17.regs "gwD" l) (b17.regs "outOff" l))).setPc (b17.pc + 1) = b18
  have hpcb18 : b18.pc = 18 := by rw [← hb18]; simp [SState.setPc, hpcb17]
  have hgb18 : b18.gmem = s.gmem := by rw [← hb18]; exact hgb17
  rw [binr_step prog b18 .add rInBase "inP" "inOff" (by rw [hpcb18]; exact i18)]
  generalize hb19 : (b18.setReg rInBase (fun l => SOp.run .add (b18.regs "inP" l) (b18.regs "inOff" l))).setPc (b18.pc + 1) = b19
  have hpcb19 : b19.pc = 19 := by rw [← hb19]; simp [SState.setPc, hpcb18]
  have hgb19 : b19.gmem = s.gmem := by rw [← hb19]; exact hgb18
  rw [binr_step prog b19 .add rOutBase "outP" "outOff" (by rw [hpcb19]; exact i19)]
  generalize hb20 : (b19.setReg rOutBase (fun l => SOp.run .add (b19.regs "outP" l) (b19.regs "outOff" l))).setPc (b19.pc + 1) = b20
  have hpcb20 : b20.pc = 20 := by rw [← hb20]; simp [SState.setPc, hpcb19]
  have hgb20 : b20.gmem = s.gmem := by rw [← hb20]; exact hgb19
  rw [mov_step prog b20 "smem" (.imm 0) (by rw [hpcb20]; exact i20)]
  generalize hb21 : (b20.setReg "smem" (fun l => b20.get l (.imm 0))).setPc (b20.pc + 1) = b21
  have hpcb21 : b21.pc = 21 := by rw [← hb21]; simp [SState.setPc, hpcb20]
  have hgb21 : b21.gmem = s.gmem := by rw [← hb21]; exact hgb20
  rw [bin_step prog b21 .mul "tblOff" "lwarp" (.imm ((2 ^ hL) * 2)) (by rw [hpcb21]; exact i21)]
  generalize hb22 : (b21.setReg "tblOff" (fun l => SOp.run .mul (b21.regs "lwarp" l) (b21.get l (.imm ((2 ^ hL) * 2))))).setPc (b21.pc + 1) = b22
  have hpcb22 : b22.pc = 22 := by rw [← hb22]; simp [SState.setPc, hpcb21]
  have hgb22 : b22.gmem = s.gmem := by rw [← hb22]; exact hgb21
  rw [binr_step prog b22 .add rTbl "smem" "tblOff" (by rw [hpcb22]; exact i22)]
  generalize hb23 : (b22.setReg rTbl (fun l => SOp.run .add (b22.regs "smem" l) (b22.regs "tblOff" l))).setPc (b22.pc + 1) = b23
  have hpcb23 : b23.pc = 23 := by rw [← hb23]; simp [SState.setPc, hpcb22]
  have hgb23 : b23.gmem = s.gmem := by rw [← hb23]; exact hgb22
  rw [mov_step prog b23 "ci" (.reg rLane) (by rw [hpcb23]; exact i23)]
  generalize hb24 : (b23.setReg "ci" (fun l => b23.get l (.reg rLane))).setPc (b23.pc + 1) = b24
  have hpcb24 : b24.pc = 24 := by rw [← hb24]; simp [SState.setPc, hpcb23]
  have hgb24 : b24.gmem = s.gmem := by rw [← hb24]; exact hgb23
  rw [mov_step prog b24 "z" (.imm 0) (by rw [hpcb24]; exact i24)]
  generalize hb25 : (b24.setReg "z" (fun l => b24.get l (.imm 0))).setPc (b24.pc + 1) = b25
  have hpcb25 : b25.pc = 25 := by rw [← hb25]; simp [SState.setPc, hpcb24]
  have hgb25 : b25.gmem = s.gmem := by rw [← hb25]; exact hgb24
  subst hb25 hb24 hb23 hb22 hb21 hb20 hb19 hb18 hb17 hb16 hb15 hb14 hb13
  refine ⟨hpcb25, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro l
    simp only [SState.setReg, SState.setPc, SState.get, SOp.run, hinP, houtP, hgw, hlw,
      hlaneS, hlaneS', u64_ofNat0, Fin.val_zero, rLane, rPL0, rInBase, rOutBase, rTbl,
      u64_inBase, u64_outBase, UInt64.zero_mul, UInt64.mul_zero, UInt64.zero_add,
      UInt64.add_zero, String.reduceEq, reduceIte]
  · intro l
    simp only [SState.setReg, SState.setPc, SState.get, SOp.run, hinP, houtP, hgw, hlw,
      hlaneS, hlaneS', u64_ofNat0, Fin.val_zero, rLane, rPL0, rInBase, rOutBase, rTbl,
      u64_inBase, u64_outBase, UInt64.zero_mul, UInt64.mul_zero, UInt64.zero_add,
      UInt64.add_zero, String.reduceEq, reduceIte]
  · intro l
    simp only [SState.setReg, SState.setPc, SState.get, SOp.run, hinP, houtP, hgw, hlw,
      hlaneS, hlaneS', u64_ofNat0, Fin.val_zero, rLane, rPL0, rInBase, rOutBase, rTbl,
      u64_inBase, u64_outBase, UInt64.zero_mul, UInt64.mul_zero, UInt64.zero_add,
      UInt64.add_zero, String.reduceEq, reduceIte]
  · intro l
    simp only [SState.setReg, SState.setPc, SState.get, SOp.run, hinP, houtP, hgw, hlw,
      hlaneS, hlaneS', u64_ofNat0, Fin.val_zero, rLane, rPL0, rInBase, rOutBase, rTbl,
      u64_inBase, u64_outBase, UInt64.zero_mul, UInt64.mul_zero, UInt64.zero_add,
      UInt64.add_zero, String.reduceEq, reduceIte]
  · simp only [SState.setReg, SState.setPc, SState.get, SOp.run, hinP, houtP, hgw, hlw,
      hlaneS, hlaneS', u64_ofNat0, Fin.val_zero, rLane, rPL0, rInBase, rOutBase, rTbl,
      u64_inBase, u64_outBase, UInt64.zero_mul, UInt64.mul_zero, UInt64.zero_add,
      UInt64.add_zero, String.reduceEq, reduceIte]
  · exact hgb25
  · intro r hr
    simp only [prologueW, List.mem_cons, List.mem_singleton, not_or] at hr
    obtain ⟨a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,
      a18, a19, a20, a21, a22, a23, a24, a25, a26⟩ := hr
    simp only [SState.setReg, SState.setPc, rPL0, rInBase, rOutBase, rTbl,
      a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,
      a18, a19, a20, a21, a22, a23, a24, a25, a26, reduceIte]

/-- **Strong head simulation** (indices 0–24): compose slices A and B.  From the launch
    state the machine reaches the `loop` head (index 25) with `MachInv`'s constants
    (`lane = l`, `inBase = tbl = 0`), `outBase = iS`, `ci 0 = 0`, and a frame. -/
theorem head25 (nb iS oS lO hL w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hnb : 0 < nb) (hnb2 : nb < 2 ^ 64) (hw : w < nb) (hw64 : w * 32 + 32 < 2 ^ 64) :
    (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB)).pc = 25 ∧
      (∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB)).regs rLane l = UInt64.ofNat l.val) ∧
      (∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB)).regs rInBase l = UInt64.ofNat (inPtr + w * iS)) ∧
      (∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB)).regs rTbl l = 0) ∧
      (∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB)).regs rOutBase l = UInt64.ofNat (outPtr + w * oS)) ∧
      (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB)).regs "ci" 0 = 0 ∧
      (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB)).gmem = (initSt w inPtr outPtr gm smemB).gmem ∧
      (∀ r, r ∉ prologueW →
        (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB)).regs r = (initSt w inPtr outPtr gm smemB).regs r) := by
  obtain ⟨hpcA, hinP, houtP, hgw, hlw, hlaneA, hgmemA, hframeA⟩ :=
    headA nb iS oS lO hL w inPtr outPtr gm smemB hnb hnb2 hw hw64
  obtain ⟨hpcB, hlaneB, hinB, htbl, houtB, hciB, hgmemB, hframeB⟩ :=
    headB nb iS oS lO hL w inPtr outPtr (snsteps (warpKernelDSL nb iS oS lO hL) 12 (initSt w inPtr outPtr gm smemB))
      hpcA hinP houtP hgw hlw hlaneA
  rw [show (25 : Nat) = 12 + 13 from rfl, snsteps_add]
  exact ⟨hpcB, hlaneB, hinB, htbl, houtB, hciB, hgmemB.trans hgmemA,
    fun r hr => (hframeB r hr).trans (hframeA r hr)⟩

/-- One table-clear loop body (8 steps, branch not taken) frames every register it
    does not write: only `pcnd`/`ca`/`ci` change.  Same threading as
    `prologue_clr_loop_body_slice`, carrying the register frame instead of `ci`. -/
theorem clr_body_frame (prog : Array SInstr) (st : SState) (entries : Nat)
    (h25 : prog[st.pc]? = some (.lbl "clr"))
    (h26 : prog[st.pc + 1]? = some (.setp .ge "pcnd" "ci" (.imm entries)))
    (h27 : prog[st.pc + 2]? = some (.braif "pcnd" "clrDone"))
    (h28 : prog[st.pc + 3]? = some (.bin .shl "ca" "ci" (.imm 1)))
    (h29 : prog[st.pc + 4]? = some (.binr .add "ca" "ca" rTbl))
    (h30 : prog[st.pc + 5]? = some (.stsh "ca" "z"))
    (h31 : prog[st.pc + 6]? = some (.bin .add "ci" "ci" (.imm 32)))
    (h32 : prog[st.pc + 7]? = some (.bra "clr"))
    (hfind : sfindLabel prog "clr" = 25)
    (hpc : st.pc = 25)
    (hbrHead :
      ((if SCmp.run .ge (st.regs "ci" 0) (st.get 0 (.imm entries)) then (1 : UInt64) else 0) ==
          1) = false) :
    (snsteps prog 8 st).pc = 25 ∧
      (∀ r, r ≠ "pcnd" → r ≠ "ca" → r ≠ "ci" → (snsteps prog 8 st).regs r = st.regs r) := by
  simp only [snsteps]
  rw [lbl_step prog st "clr" h25]
  let st26 := st.setPc (st.pc + 1)
  change (snsteps prog 7 st26).pc = 25 ∧
    (∀ r, r ≠ "pcnd" → r ≠ "ca" → r ≠ "ci" → (snsteps prog 7 st26).regs r = st.regs r)
  simp only [snsteps]
  rw [setp_step prog st26 .ge "pcnd" "ci" (.imm entries)
    (by simpa [st26, SState.setPc, hpc] using h26)]
  let st27 := (st26.setReg "pcnd"
    (fun l => if SCmp.run .ge (st26.regs "ci" l) (st26.get l (.imm entries)) then 1 else 0)).setPc
      (st26.pc + 1)
  change (snsteps prog 6 st27).pc = 25 ∧
    (∀ r, r ≠ "pcnd" → r ≠ "ca" → r ≠ "ci" → (snsteps prog 6 st27).regs r = st.regs r)
  simp only [snsteps]
  rw [braif_step prog st27 "pcnd" "clrDone"
    (by simpa [st27, st26, SState.setPc, hpc] using h27)]
  have hbr : (st27.regs "pcnd" 0 == 1) = false := by
    simpa [st27, st26, SState.setReg, SState.setPc, SState.get] using hbrHead
  rw [hbr]
  let st28 := st27.setPc (st27.pc + 1)
  change (snsteps prog 5 st28).pc = 25 ∧
    (∀ r, r ≠ "pcnd" → r ≠ "ca" → r ≠ "ci" → (snsteps prog 5 st28).regs r = st.regs r)
  simp only [snsteps]
  rw [bin_step prog st28 .shl "ca" "ci" (.imm 1)
    (by simpa [st28, st27, st26, SState.setPc, hpc, hbr] using h28)]
  let st29 := (st28.setReg "ca"
    (fun l => SOp.run .shl (st28.regs "ci" l) (st28.get l (.imm 1)))).setPc (st28.pc + 1)
  change (snsteps prog 4 st29).pc = 25 ∧
    (∀ r, r ≠ "pcnd" → r ≠ "ca" → r ≠ "ci" → (snsteps prog 4 st29).regs r = st.regs r)
  simp only [snsteps]
  rw [binr_step prog st29 .add "ca" "ca" rTbl
    (by simpa [st29, st28, st27, st26, SState.setPc, hpc, hbr] using h29)]
  let st30 := (st29.setReg "ca"
    (fun l => SOp.run .add (st29.regs "ca" l) (st29.regs rTbl l))).setPc (st29.pc + 1)
  change (snsteps prog 3 st30).pc = 25 ∧
    (∀ r, r ≠ "pcnd" → r ≠ "ca" → r ≠ "ci" → (snsteps prog 3 st30).regs r = st.regs r)
  simp only [snsteps]
  rw [stsh_step prog st30 "ca" "z"
    (by simpa [st30, st29, st28, st27, st26, SState.setPc, hpc, hbr] using h30)]
  let st31 : SState :=
    { st30 with
      smem :=
        (let s0 := storeBytes st30.smem (fun _ => true) (st30.regs "ca") (st30.regs "z")
         storeBytes s0 (fun _ => true) (fun l => st30.regs "ca" l + 1) (fun l => st30.regs "z" l >>> 8))
      pc := st30.pc + 1 }
  change (snsteps prog 2 st31).pc = 25 ∧
    (∀ r, r ≠ "pcnd" → r ≠ "ca" → r ≠ "ci" → (snsteps prog 2 st31).regs r = st.regs r)
  simp only [snsteps]
  rw [bin_step prog st31 .add "ci" "ci" (.imm 32)
    (by simpa [st31, st30, st29, st28, st27, st26, SState.setPc, hpc, hbr] using h31)]
  let st32 := (st31.setReg "ci"
    (fun l => SOp.run .add (st31.regs "ci" l) (st31.get l (.imm 32)))).setPc (st31.pc + 1)
  change (snsteps prog 1 st32).pc = 25 ∧
    (∀ r, r ≠ "pcnd" → r ≠ "ca" → r ≠ "ci" → (snsteps prog 1 st32).regs r = st.regs r)
  simp only [snsteps]
  rw [bra_step prog st32 "clr"
    (by simpa [st32, st31, st30, st29, st28, st27, st26, SState.setPc, hpc, hbr] using h32)]
  refine ⟨by simp [SState.setPc, hfind], ?_⟩
  intro r hp hca hci
  simp only [st32, st31, st30, st29, st28, st27, st26, SState.setReg, SState.setPc,
    if_neg hp, if_neg hca, if_neg hci]

/-- Over all `k` table-clear iterations, every register other than the loop's own
    `pcnd`/`ca`/`ci` is framed.  Induction on `k`, taking the per-iteration branch
    and `ci` boundary values from `prologue_clear_body_iter`. -/
theorem clr_loop_frame (prog : Array SInstr) (st : SState) (hashLog entries : Nat)
    (h25 : prog[25]? = some (.lbl "clr"))
    (h26 : prog[26]? = some (.setp .ge "pcnd" "ci" (.imm entries)))
    (h27 : prog[27]? = some (.braif "pcnd" "clrDone"))
    (h28 : prog[28]? = some (.bin .shl "ca" "ci" (.imm 1)))
    (h29 : prog[29]? = some (.binr .add "ca" "ca" rTbl))
    (h30 : prog[30]? = some (.stsh "ca" "z"))
    (h31 : prog[31]? = some (.bin .add "ci" "ci" (.imm 32)))
    (h32 : prog[32]? = some (.bra "clr"))
    (hclr : sfindLabel prog "clr" = 25)
    (hEntries : entries = 2 ^ hashLog)
    (hHash : hashLog ≤ 32)
    (hstpc : st.pc = 25)
    (hstci : st.regs "ci" 0 = 0)
    (hlane : ∀ l : Fin 32, st.regs rLane l = UInt64.ofNat l.val)
    (k : Nat) (hk : k ≤ clearIters hashLog) :
    ∀ r, r ≠ "pcnd" → r ≠ "ca" → r ≠ "ci" →
      (snsteps prog (8 * k) st).regs r = st.regs r := by
  induction k with
  | zero => intro r _ _ _; simp only [Nat.mul_zero]; rfl
  | succ k ih =>
      intro r hp hca hci
      have hkprev : k ≤ clearIters hashLog := by omega
      have hklt : k < clearIters hashLog := by omega
      have ihv := ih hkprev
      have hiter := prologue_clear_body_iter prog st hashLog entries
        h25 h26 h27 h28 h29 h30 h31 h32 hclr hEntries hHash hstpc hstci hlane k hkprev
      dsimp only at hiter
      rw [show 8 * (k + 1) = 8 * k + 8 by omega, snsteps_add]
      have hbrHead : ((if SCmp.run .ge ((snsteps prog (8 * k) st).regs "ci" 0)
          ((snsteps prog (8 * k) st).get 0 (.imm entries)) then (1 : UInt64) else 0) == 1) = false := by
        rw [hiter.2.1, hEntries]
        change ((if SCmp.run .ge (UInt64.ofNat (32 * k)) (UInt64.ofNat (2 ^ hashLog)) then
            (1 : UInt64) else 0) == 1) = false
        exact clearLoop_body_branch_false hashLog k hHash hklt
      have hbody := clr_body_frame prog (snsteps prog (8 * k) st) entries
        (by rw [hiter.1]; exact h25) (by rw [hiter.1]; exact h26) (by rw [hiter.1]; exact h27)
        (by rw [hiter.1]; exact h28) (by rw [hiter.1]; exact h29) (by rw [hiter.1]; exact h30)
        (by rw [hiter.1]; exact h31) (by rw [hiter.1]; exact h32) hclr hiter.1 hbrHead
      rw [hbody.2 r hp hca hci]
      exact ihv r hp hca hci

/-- The table-clear exit (branch taken) plus the `barwarp` and the three `mov …
    (imm 0)` initializers reach the loop head at pc 38, framing every register other
    than `pcnd`/`litAnchor`/`searchPos`/`op`.  Mirrors `prologue_clr_exit_to_loop_slice`. -/
theorem clr_exit_frame (prog : Array SInstr) (st : SState) (entries : Nat)
    (h25 : prog[st.pc]? = some (.lbl "clr"))
    (h26 : prog[st.pc + 1]? = some (.setp .ge "pcnd" "ci" (.imm entries)))
    (h27 : prog[st.pc + 2]? = some (.braif "pcnd" "clrDone"))
    (h33 : prog[33]? = some (.lbl "clrDone"))
    (h34 : prog[34]? = some .barwarp)
    (h35 : prog[35]? = some (.mov rLitAnchor (.imm 0)))
    (h36 : prog[36]? = some (.mov rSearchPos (.imm 0)))
    (h37 : prog[37]? = some (.mov rOp (.imm 0)))
    (hfind : sfindLabel prog "clrDone" = 33)
    (hpc : st.pc = 25)
    (hbrHead :
      ((if SCmp.run .ge (st.regs "ci" 0) (st.get 0 (.imm entries)) then (1 : UInt64) else 0) ==
          1) = true) :
    (snsteps prog 8 st).pc = 38 ∧
      (∀ l : Fin 32, (snsteps prog 8 st).regs rLitAnchor l = 0) ∧
      (∀ l : Fin 32, (snsteps prog 8 st).regs rSearchPos l = 0) ∧
      (∀ l : Fin 32, (snsteps prog 8 st).regs rOp l = 0) ∧
      (∀ r, r ≠ "pcnd" → r ≠ rLitAnchor → r ≠ rSearchPos → r ≠ rOp →
        (snsteps prog 8 st).regs r = st.regs r) := by
  simp only [snsteps]
  rw [lbl_step prog st "clr" h25]
  let st26 := st.setPc (st.pc + 1)
  change (snsteps prog 7 st26).pc = 38 ∧
    (∀ l : Fin 32, (snsteps prog 7 st26).regs rLitAnchor l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 7 st26).regs rSearchPos l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 7 st26).regs rOp l = 0) ∧
    (∀ r, r ≠ "pcnd" → r ≠ rLitAnchor → r ≠ rSearchPos → r ≠ rOp →
      (snsteps prog 7 st26).regs r = st.regs r)
  simp only [snsteps]
  rw [setp_step prog st26 .ge "pcnd" "ci" (.imm entries)
    (by simpa [st26, SState.setPc, hpc] using h26)]
  let st27 := (st26.setReg "pcnd"
    (fun l => if SCmp.run .ge (st26.regs "ci" l) (st26.get l (.imm entries)) then 1 else 0)).setPc
      (st26.pc + 1)
  change (snsteps prog 6 st27).pc = 38 ∧
    (∀ l : Fin 32, (snsteps prog 6 st27).regs rLitAnchor l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 6 st27).regs rSearchPos l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 6 st27).regs rOp l = 0) ∧
    (∀ r, r ≠ "pcnd" → r ≠ rLitAnchor → r ≠ rSearchPos → r ≠ rOp →
      (snsteps prog 6 st27).regs r = st.regs r)
  simp only [snsteps]
  rw [braif_step prog st27 "pcnd" "clrDone"
    (by simpa [st27, st26, SState.setPc, hpc] using h27)]
  have hbr : (st27.regs "pcnd" 0 == 1) = true := by
    simpa [st27, st26, SState.setReg, SState.setPc, SState.get] using hbrHead
  rw [hbr, hfind]
  simp only [reduceIte]
  let st33 := st27.setPc 33
  change (snsteps prog 5 st33).pc = 38 ∧
    (∀ l : Fin 32, (snsteps prog 5 st33).regs rLitAnchor l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 5 st33).regs rSearchPos l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 5 st33).regs rOp l = 0) ∧
    (∀ r, r ≠ "pcnd" → r ≠ rLitAnchor → r ≠ rSearchPos → r ≠ rOp →
      (snsteps prog 5 st33).regs r = st.regs r)
  simp only [snsteps]
  rw [lbl_step prog st33 "clrDone" (by simpa [st33, SState.setPc] using h33)]
  let st34 := st33.setPc (st33.pc + 1)
  change (snsteps prog 4 st34).pc = 38 ∧
    (∀ l : Fin 32, (snsteps prog 4 st34).regs rLitAnchor l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 4 st34).regs rSearchPos l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 4 st34).regs rOp l = 0) ∧
    (∀ r, r ≠ "pcnd" → r ≠ rLitAnchor → r ≠ rSearchPos → r ≠ rOp →
      (snsteps prog 4 st34).regs r = st.regs r)
  simp only [snsteps]
  rw [barwarp_step prog st34 (by simpa [st34, st33, SState.setPc] using h34)]
  let st35 := st34.setPc (st34.pc + 1)
  change (snsteps prog 3 st35).pc = 38 ∧
    (∀ l : Fin 32, (snsteps prog 3 st35).regs rLitAnchor l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 3 st35).regs rSearchPos l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 3 st35).regs rOp l = 0) ∧
    (∀ r, r ≠ "pcnd" → r ≠ rLitAnchor → r ≠ rSearchPos → r ≠ rOp →
      (snsteps prog 3 st35).regs r = st.regs r)
  simp only [snsteps]
  rw [mov_step prog st35 rLitAnchor (.imm 0)
    (by simpa [st35, st34, st33, SState.setPc] using h35)]
  let st36 := (st35.setReg rLitAnchor (fun l => st35.get l (.imm 0))).setPc (st35.pc + 1)
  change (snsteps prog 2 st36).pc = 38 ∧
    (∀ l : Fin 32, (snsteps prog 2 st36).regs rLitAnchor l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 2 st36).regs rSearchPos l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 2 st36).regs rOp l = 0) ∧
    (∀ r, r ≠ "pcnd" → r ≠ rLitAnchor → r ≠ rSearchPos → r ≠ rOp →
      (snsteps prog 2 st36).regs r = st.regs r)
  simp only [snsteps]
  rw [mov_step prog st36 rSearchPos (.imm 0)
    (by simpa [st36, st35, st34, st33, SState.setPc] using h36)]
  let st37 := (st36.setReg rSearchPos (fun l => st36.get l (.imm 0))).setPc (st36.pc + 1)
  change (snsteps prog 1 st37).pc = 38 ∧
    (∀ l : Fin 32, (snsteps prog 1 st37).regs rLitAnchor l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 1 st37).regs rSearchPos l = 0) ∧
    (∀ l : Fin 32, (snsteps prog 1 st37).regs rOp l = 0) ∧
    (∀ r, r ≠ "pcnd" → r ≠ rLitAnchor → r ≠ rSearchPos → r ≠ rOp →
      (snsteps prog 1 st37).regs r = st.regs r)
  simp only [snsteps]
  rw [mov_step prog st37 rOp (.imm 0)
    (by simpa [st37, st36, st35, st34, st33, SState.setPc] using h37)]
  refine ⟨by simp [st37, st36, st35, st34, st33, SState.setReg, SState.setPc], ?_, ?_, ?_, ?_⟩
  · intro l
    simp only [st37, st36, st35, st34, st33, SState.setReg, SState.setPc, SState.get,
      rLitAnchor, rSearchPos, rOp, String.reduceEq, reduceIte, u64_ofNat0]
  · intro l
    simp only [st37, st36, st35, st34, st33, SState.setReg, SState.setPc, SState.get,
      rLitAnchor, rSearchPos, rOp, String.reduceEq, reduceIte, u64_ofNat0]
  · intro l
    simp only [st37, st36, st35, st34, st33, SState.setReg, SState.setPc, SState.get,
      rLitAnchor, rSearchPos, rOp, String.reduceEq, reduceIte, u64_ofNat0]
  · intro r hp hla hsp hop
    simp only [snsteps, st37, st36, st35, st34, st33, st27, st26, SState.setReg, SState.setPc,
      if_neg hp, if_neg hla, if_neg hsp, if_neg hop]

/-- **Full prologue simulation** (indices 0–37): from the launch state the machine
    runs the id setup, base-pointer setup, and the whole table-clear loop, reaching
    the main-loop head at pc 38 with every register the body roundtrip needs:
    `MachInv`'s constants (`lane = l`, `inBase = tbl = 0`), `outBase = iS`, the three
    `mov`-initialized regs `op = litAnchor = searchPos = 0` (uniform), `gmem` intact,
    and a frame for every register outside the prologue's write set. -/
theorem head38 (nb iS oS lO hL w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hnb : 0 < nb) (hnb2 : nb < 2 ^ 64) (hw : w < nb) (hw64 : w * 32 + 32 < 2 ^ 64) (hHash : hL ≤ 32) :
    (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).pc = 38 ∧
      (∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs rLane l = UInt64.ofNat l.val) ∧
      (∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs rInBase l = UInt64.ofNat (inPtr + w * iS)) ∧
      (∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs rTbl l = 0) ∧
      (∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs rOutBase l = UInt64.ofNat (outPtr + w * oS)) ∧
      (∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs rLitAnchor l = 0) ∧
      (∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs rSearchPos l = 0) ∧
      (∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs rOp l = 0) ∧
      (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).gmem = (initSt w inPtr outPtr gm smemB).gmem ∧
      (∀ r, r ∉ prologueW →
        (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs r = (initSt w inPtr outPtr gm smemB).regs r) := by
  have e25 : (warpKernelDSL nb iS oS lO hL)[25]? = some (.lbl "clr") := wk_idx nb iS oS lO hL 25 _ (by omega) rfl
  have e26 : (warpKernelDSL nb iS oS lO hL)[26]? = some (.setp .ge "pcnd" "ci" (.imm (2 ^ hL))) := wk_idx nb iS oS lO hL 26 _ (by omega) rfl
  have e27 : (warpKernelDSL nb iS oS lO hL)[27]? = some (.braif "pcnd" "clrDone") := wk_idx nb iS oS lO hL 27 _ (by omega) rfl
  have e28 : (warpKernelDSL nb iS oS lO hL)[28]? = some (.bin .shl "ca" "ci" (.imm 1)) := wk_idx nb iS oS lO hL 28 _ (by omega) rfl
  have e29 : (warpKernelDSL nb iS oS lO hL)[29]? = some (.binr .add "ca" "ca" rTbl) := wk_idx nb iS oS lO hL 29 _ (by omega) rfl
  have e30 : (warpKernelDSL nb iS oS lO hL)[30]? = some (.stsh "ca" "z") := wk_idx nb iS oS lO hL 30 _ (by omega) rfl
  have e31 : (warpKernelDSL nb iS oS lO hL)[31]? = some (.bin .add "ci" "ci" (.imm 32)) := wk_idx nb iS oS lO hL 31 _ (by omega) rfl
  have e32 : (warpKernelDSL nb iS oS lO hL)[32]? = some (.bra "clr") := wk_idx nb iS oS lO hL 32 _ (by omega) rfl
  have e33 : (warpKernelDSL nb iS oS lO hL)[33]? = some (.lbl "clrDone") := wk_idx nb iS oS lO hL 33 _ (by omega) rfl
  have e34 : (warpKernelDSL nb iS oS lO hL)[34]? = some .barwarp := wk_idx nb iS oS lO hL 34 _ (by omega) rfl
  have e35 : (warpKernelDSL nb iS oS lO hL)[35]? = some (.mov rLitAnchor (.imm 0)) := wk_idx nb iS oS lO hL 35 _ (by omega) rfl
  have e36 : (warpKernelDSL nb iS oS lO hL)[36]? = some (.mov rSearchPos (.imm 0)) := wk_idx nb iS oS lO hL 36 _ (by omega) rfl
  have e37 : (warpKernelDSL nb iS oS lO hL)[37]? = some (.mov rOp (.imm 0)) := wk_idx nb iS oS lO hL 37 _ (by omega) rfl
  have hclr : sfindLabel (warpKernelDSL nb iS oS lO hL) "clr" = 25 := by rfl
  have hfind : sfindLabel (warpKernelDSL nb iS oS lO hL) "clrDone" = 33 := by rfl
  obtain ⟨hpc25, hlane25, hinB25, htbl25, houtB25, hci25, hgmem25, hframe25⟩ :=
    head25 nb iS oS lO hL w inPtr outPtr gm smemB hnb hnb2 hw hw64
  have hiter := prologue_clear_body_iter (warpKernelDSL nb iS oS lO hL)
    (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB)) hL (2 ^ hL)
    e25 e26 e27 e28 e29 e30 e31 e32 hclr rfl hHash hpc25 hci25 hlane25 (clearIters hL) (Nat.le_refl _)
  dsimp only at hiter
  have hloopfr := clr_loop_frame (warpKernelDSL nb iS oS lO hL)
    (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB)) hL (2 ^ hL)
    e25 e26 e27 e28 e29 e30 e31 e32 hclr rfl hHash hpc25 hci25 hlane25 (clearIters hL) (Nat.le_refl _)
  have hbrHead : ((if SCmp.run .ge
      ((snsteps (warpKernelDSL nb iS oS lO hL) (8 * clearIters hL)
        (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB))).regs "ci" 0)
      ((snsteps (warpKernelDSL nb iS oS lO hL) (8 * clearIters hL)
        (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB))).get 0 (.imm (2 ^ hL)))
      then (1 : UInt64) else 0) == 1) = true := by
    rw [hiter.2.1]
    change ((if SCmp.run .ge (UInt64.ofNat (32 * clearIters hL)) (UInt64.ofNat (2 ^ hL)) then
        (1 : UInt64) else 0) == 1) = true
    exact clearLoop_exit_branch_true hL hHash
  have hexitfr := clr_exit_frame (warpKernelDSL nb iS oS lO hL)
    (snsteps (warpKernelDSL nb iS oS lO hL) (8 * clearIters hL)
      (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB))) (2 ^ hL)
    (by rw [hiter.1]; exact e25) (by rw [hiter.1]; exact e26) (by rw [hiter.1]; exact e27)
    e33 e34 e35 e36 e37 hfind hiter.1 hbrHead
  have hexitgmem := prologue_clr_exit_to_loop_slice (warpKernelDSL nb iS oS lO hL)
    (snsteps (warpKernelDSL nb iS oS lO hL) (8 * clearIters hL)
      (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB))) (2 ^ hL)
    (by rw [hiter.1]; exact e25) (by rw [hiter.1]; exact e26) (by rw [hiter.1]; exact e27)
    e33 e34 e35 e36 e37 hfind hiter.1 hbrHead hiter.2.2.1
  rw [show 25 + 8 * clearIters hL + 8 = (25 + 8 * clearIters hL) + 8 from rfl, snsteps_add, snsteps_add]
  have hframeExit : ∀ r, r ≠ "pcnd" → r ≠ rLitAnchor → r ≠ rSearchPos → r ≠ rOp →
      (snsteps (warpKernelDSL nb iS oS lO hL) 8
        (snsteps (warpKernelDSL nb iS oS lO hL) (8 * clearIters hL)
          (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB)))).regs r =
      (snsteps (warpKernelDSL nb iS oS lO hL) (8 * clearIters hL)
        (snsteps (warpKernelDSL nb iS oS lO hL) 25 (initSt w inPtr outPtr gm smemB))).regs r :=
    hexitfr.2.2.2.2
  refine ⟨hexitfr.1, ?_, ?_, ?_, ?_, hexitfr.2.1, hexitfr.2.2.1, hexitfr.2.2.2.1, ?_, ?_⟩
  · intro l
    rw [hframeExit rLane (by decide) (by decide) (by decide) (by decide),
        hloopfr rLane (by decide) (by decide) (by decide)]
    exact hlane25 l
  · intro l
    rw [hframeExit rInBase (by decide) (by decide) (by decide) (by decide),
        hloopfr rInBase (by decide) (by decide) (by decide)]
    exact hinB25 l
  · intro l
    rw [hframeExit rTbl (by decide) (by decide) (by decide) (by decide),
        hloopfr rTbl (by decide) (by decide) (by decide)]
    exact htbl25 l
  · intro l
    rw [hframeExit rOutBase (by decide) (by decide) (by decide) (by decide),
        hloopfr rOutBase (by decide) (by decide) (by decide)]
    exact houtB25 l
  · rw [hexitgmem.2.2.2.2.2, hiter.2.2.2, hgmem25]
  · intro r hr
    have hne : ∀ s, s ∈ prologueW → r ≠ s := fun s hs h => hr (h ▸ hs)
    rw [hframeExit r (hne "pcnd" (by decide)) (hne rLitAnchor (by decide))
          (hne rSearchPos (by decide)) (hne rOp (by decide)),
        hloopfr r (hne "pcnd" (by decide)) (hne "ca" (by decide)) (hne "ci" (by decide))]
    exact hframe25 r hr

/-- **Post-prologue coupled state**: stepping past the `loop` label (pc 39), the
    machine state `MachInv`-holds and `Couple`s to its own lane-0 projection over
    `loopR` (every loop register is uniform: the five prologue-set ones by value, the
    thirty scratch ones at their uniform launch value `0`).  Plus the scalar facts
    the body roundtrip consumes: `op = litAnchor = searchPos = inBase = 0`,
    `outBase = outPtr + w*oS`, `gmem` unchanged. -/
theorem prologue_couple (nb iS oS lO hL w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hnb : 0 < nb) (hnb2 : nb < 2 ^ 64) (hw : w < nb) (hw64 : w * 32 + 32 < 2 ^ 64) (hHash : hL ≤ 32) :
    (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).pc = 39 ∧
      MachInv (inPtr + w * iS) (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)) ∧
      Couple loopR (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB))
        { regs := fun r => (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).regs r 0,
          gmem := (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).gmem,
          smem := (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).smem } ∧
      (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).regs "op" 0 = 0 ∧
      (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).regs "litAnchor" 0 = 0 ∧
      (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).regs "searchPos" 0 = 0 ∧
      (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).regs "inBase" 0 = UInt64.ofNat (inPtr + w * iS) ∧
      (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).regs "outBase" 0 = UInt64.ofNat (outPtr + w * oS) ∧
      (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).gmem = gm := by
  obtain ⟨h38pc, h38lane, h38inB, h38tbl, h38outB, h38la, h38sp, h38op, h38gmem, h38frame⟩ :=
    head38 nb iS oS lO hL w inPtr outPtr gm smemB hnb hnb2 hw hw64 hHash
  have e38 : (warpKernelDSL nb iS oS lO hL)[38]? = some (.lbl "loop") := wk_idx nb iS oS lO hL 38 _ (by omega) rfl
  have hstep : snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB) =
      (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).setPc 39 := by
    rw [snsteps_add]
    show sstep (warpKernelDSL nb iS oS lO hL)
      (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)) = _
    rw [lbl_step _ _ "loop" (by rw [h38pc]; exact e38), h38pc]
  have hreg : (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).regs =
      (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs := by rw [hstep]; rfl
  have hgm : (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).gmem =
      (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).gmem := by rw [hstep]; rfl
  -- string-literal forms of head38's value facts (accepted by defeq of the reg-name defs)
  have hlaneS : ∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs "lane" l = UInt64.ofNat l.val := h38lane
  have hinBS : ∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs "inBase" l = UInt64.ofNat (inPtr + w * iS) := h38inB
  have htblS : ∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs "tbl" l = 0 := h38tbl
  have houtBS : ∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs "outBase" l = UInt64.ofNat (outPtr + w * oS) := h38outB
  have hlaS : ∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs "litAnchor" l = 0 := h38la
  have hspS : ∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs "searchPos" l = 0 := h38sp
  have hopS : ∀ l : Fin 32, (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8) (initSt w inPtr outPtr gm smemB)).regs "op" l = 0 := h38op
  refine ⟨by rw [hstep]; simp [SState.setPc], ⟨fun l => ?_, fun l => ?_, fun l => ?_⟩,
    ⟨rfl, rfl, ?_⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hreg]; exact hlaneS l
  · rw [hreg]; exact hinBS l
  · rw [hreg]; exact htblS l
  · -- Couple register uniformity over loopR
    intro r hr l
    show (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).regs r l =
      (snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1) (initSt w inPtr outPtr gm smemB)).regs r 0
    rw [hreg]
    simp only [loopR, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
      first
        | (rw [hinBS l, hinBS 0])
        | (rw [houtBS l, houtBS 0])
        | (rw [hopS l, hopS 0])
        | (rw [hlaS l, hlaS 0])
        | (rw [hspS l, hspS 0])
        | (rw [h38frame _ (by decide)]; rfl)
        | (rw [h38frame _ (by decide)]; simp [initSt, initRegs])
  · rw [hreg]; exact hopS 0
  · rw [hreg]; exact hlaS 0
  · rw [hreg]; exact hspS 0
  · rw [hreg]; exact hinBS 0
  · rw [hreg]; exact houtBS 0
  · rw [hgm, h38gmem]; rfl

/-- The LSIC extend-body emit (matches the discharged theorem's `hLdef`/`hMdef`/`hFdef`). -/
def myLsic (extra : String) : List SInstr :=
  [.mov "c255" (.imm 255)]
    ++ (([.bin .add "sbAddr" "outBase" (.reg "op")] ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
      ++ ([.bin .sub extra extra (.imm 255)] ++ [.setp .ge "lsicC" extra (.imm 255)]))

/-- The body prefix the roundtrip consumes (`compressorBodyEmit` minus the length-store
    tail), with the emitter's concrete gensym labels. -/
def bodyPrefixSeg (iS hL : Nat) : List SInstr :=
  ([.setp .lt "loopC" "searchPos" (.imm (iS - 12))] : List SInstr)
   ++ (uwhileEmit "loopC" "Lh0" "Lx1"
        (loopCBodyEmit iS hL "Le2" "Ln3" "Lh4" "Lx5" "Le6" "Ln7" "Lh8" "Lx9" "Ch10" "Cx11" "Le12" "Ln13" "Lh14" "Lx15"
          (myLsic "litExtra") (myLsic "matExtra"))
     ++ (([.mov "fLen" (.imm iS)] : List SInstr)
       ++ (([.bin .sub "fLen" "fLen" (.reg "litAnchor")] : List SInstr)
         ++ wEmitFinalSeqEmit "litAnchor" "fLen" "Le16" "Ln17" "Lh18" "Lx19" "Ch20" "Cx21" (myLsic "litExtraF"))))

/-- Peel one instruction leaf off the flattened body. -/
local macro "consPeel" o:term : tactic =>
  `(tactic| refine segAt_body_cons _ _ _ _ _ _ $o _ _ (by omega) (by rfl) ?_)
/-- Peel one multi-instruction emit leaf of length `ln` off the flattened body. -/
local macro "leafPeel" o:term "," ln:term : tactic =>
  `(tactic| refine segAt_append_intro' $ln (by rfl)
      (segAt_body_leaf _ _ _ _ _ _ $o $ln _ (by omega) (by rfl) (by rw [bodyLen233]; omega) (by rfl)) ?_)

/-- **Body layout (`SegAt`)**: the discharged theorem's segment sits at `warpKernelDSL`
    index 39.  Proved by flattening the emit structure and peeling each leaf against a
    small body slice — no monolithic (slow) comparison. -/
theorem body_segAt (nb iS oS lO hL : Nat) :
    SegAt (warpKernelDSL nb iS oS lO hL) 39 (bodyPrefixSeg iS hL) := by
  simp only [bodyPrefixSeg, uwhileEmit, loopCBodyEmit, uifEmit, foundBranchEmit,
    List.cons_append, List.append_assoc, List.nil_append]
  consPeel 0; consPeel 1; consPeel 2
  leafPeel 3, 51
  consPeel 54; consPeel 55; consPeel 56; consPeel 57; consPeel 58; consPeel 59; consPeel 60
  leafPeel 61, 20
  consPeel 81; consPeel 82; consPeel 83; consPeel 84
  leafPeel 85, 76
  consPeel 161; consPeel 162; consPeel 163; consPeel 164; consPeel 165; consPeel 166; consPeel 167
  consPeel 168; consPeel 169; consPeel 170; consPeel 171
  exact segAt_body_leaf _ _ _ _ _ _ 172 46 _ (by omega) (by rfl) (by rw [bodyLen233]; omega) (by rfl)

/-- No-label leaf proof shared by the three fixed straight-line collectives. -/
local macro "lrNoLab" : term =>
  `(term| (labelsResolve_no_labels (by intro n; simp only [coopWindowEmit, foundExtBodyEmit, coopCopyBody,
    List.mem_cons, List.mem_singleton, List.not_mem_nil, reduceCtorEq, or_false, or_self, not_false_iff])))
/-- One LabelsResolve peel: nil, a label (`sfindLabel` by `rfl`), a non-label, or one of
    the fixed no-label collective leaves (`coopWindowEmit`/`foundExtBodyEmit`/`coopCopyBody`). -/
local macro "lrStep" : tactic => `(tactic| first
  | exact labelsResolve_nil _ _
  | (refine labelsResolve_cons_lbl (by rfl) ?_)
  | (refine labelsResolve_cons_other (by intro n; exact fun h => nomatch h) ?_)
  | (refine labelsResolve_append_intro 51 (by rfl) lrNoLab ?_)
  | (refine labelsResolve_append_intro 20 (by rfl) lrNoLab ?_)
  | (refine labelsResolve_append_intro 10 (by rfl) lrNoLab ?_))

-- **Body labels (`LabelsResolve`)**: every label in the discharged theorem's segment
-- resolves (via `sfindLabel`) to its own index.  Fully flatten the emit, then peel; each
-- label's `sfindLabel` is a deep-but-fast `rfl` (the scan short-circuits — it never forces
-- the emitter monad), so the one `maxRecDepth` bump keeps the compile fast.
set_option maxRecDepth 8192 in
theorem body_labelsResolve (nb iS oS lO hL : Nat) :
    LabelsResolve (warpKernelDSL nb iS oS lO hL) 39 (bodyPrefixSeg iS hL) := by
  simp only [bodyPrefixSeg, uwhileEmit, loopCBodyEmit, uifEmit, foundBranchEmit, wEmitMatchSeqEmit,
    wEmitFinalSeqEmit, matchPreEmit, matchMidEmit, matchCoopEmit, matchOffEmit, finalPreEmit,
    finalMidEmit, finalCoopEmit, lsicThen, lsicEmit, myLsic,
    List.cons_append, List.append_assoc, List.nil_append]
  repeat lrStep

theorem labelsResolve_no_labels {prog : Array SInstr} {base : Nat} {L : List SInstr}
    (hL : ∀ n, (SInstr.lbl n) ∉ L) : LabelsResolve prog base L := by
  intro k name hk; exact absurd (List.mem_of_getElem? hk) (hL name)

-- ── Length-store tail: `[outBase+lenOff .. +3] := op` (u32 LE), then `OOB`/`ret` ──

/-- The tail as a `WStmt` (15 leaves, no labels). -/
def tailStmt (lO : Nat) : WStmt :=
  wseq
  [ .bin .add "la0" "outBase" (.imm lO),
    .bin .band "lb" "op" (.imm 255), .stgB "la0" "lb",
    .bin .add "la1" "la0" (.imm 1), .bin .shr "ls" "op" (.imm 8),
    .bin .band "lb" "ls" (.imm 255), .stgB "la1" "lb",
    .bin .add "la2" "la0" (.imm 2), .bin .shr "ls" "op" (.imm 16),
    .bin .band "lb" "ls" (.imm 255), .stgB "la2" "lb",
    .bin .add "la3" "la0" (.imm 3), .bin .shr "ls" "op" (.imm 24),
    .bin .band "lb" "ls" (.imm 255), .stgB "la3" "lb" ]

/-- The tail's machine instructions, shaped to match `simSL'_seq`'s `ea ++ eb`. -/
def tailEmit (lO : Nat) : List SInstr :=
  ([.bin .add "la0" "outBase" (.imm lO)] : List SInstr) ++
  (([.bin .band "lb" "op" (.imm 255)] : List SInstr) ++
  (([.stg "la0" "lb"] : List SInstr) ++
  (([.bin .add "la1" "la0" (.imm 1)] : List SInstr) ++
  (([.bin .shr "ls" "op" (.imm 8)] : List SInstr) ++
  (([.bin .band "lb" "ls" (.imm 255)] : List SInstr) ++
  (([.stg "la1" "lb"] : List SInstr) ++
  (([.bin .add "la2" "la0" (.imm 2)] : List SInstr) ++
  (([.bin .shr "ls" "op" (.imm 16)] : List SInstr) ++
  (([.bin .band "lb" "ls" (.imm 255)] : List SInstr) ++
  (([.stg "la2" "lb"] : List SInstr) ++
  (([.bin .add "la3" "la0" (.imm 3)] : List SInstr) ++
  (([.bin .shr "ls" "op" (.imm 24)] : List SInstr) ++
  (([.bin .band "lb" "ls" (.imm 255)] : List SInstr) ++
   ([.stg "la3" "lb"] : List SInstr))))))))))))))

theorem tailEmit_length (lO : Nat) : (tailEmit lO).length = 15 := by
  simp [tailEmit]

/-- The tail sits at `warpKernelDSL` index 257 (= 39 + 218). -/
theorem tail_segAt (nb iS oS lO hL : Nat) :
    SegAt (warpKernelDSL nb iS oS lO hL) 257 (tailEmit lO) :=
  segAt_body_leaf nb iS oS lO hL 257 218 15 _ (by omega) (tailEmit_length lO)
    (by rw [bodyLen233]; omega) (by rfl)

/-- The tail has no labels, so `LabelsResolve` is vacuous. -/
theorem tail_labelsResolve (nb iS oS lO hL : Nat) :
    LabelsResolve (warpKernelDSL nb iS oS lO hL) 257 (tailEmit lO) :=
  labelsResolve_no_labels (by
    intro n
    simp only [tailEmit, List.mem_cons, List.mem_append, List.mem_singleton,
      List.not_mem_nil, reduceCtorEq, or_false, or_self, not_false_iff])

/-- The tail simulates: composed from the `bin`/`stgB` leaves (all its registers
    are in `loopR`). -/
theorem tail_sim (ib lO : Nat) : SimSL' ib loopR (tailStmt lO) (tailEmit lO) := by
  simp only [tailStmt, tailEmit, wseq]
  refine simSL'_seq _ _ _ _ _
    (simSL'_bin (ib := ib) loopR .add "la0" "outBase" (.imm lO) (by decide) (fun n h => by cases h) (by decide)) ?_
  refine simSL'_seq _ _ _ _ _
    (simSL'_bin (ib := ib) loopR .band "lb" "op" (.imm 255) (by decide) (fun n h => by cases h) (by decide)) ?_
  refine simSL'_seq _ _ _ _ _ (simSL'_stgB (ib := ib) loopR "la0" "lb" (by decide) (by decide)) ?_
  refine simSL'_seq _ _ _ _ _
    (simSL'_bin (ib := ib) loopR .add "la1" "la0" (.imm 1) (by decide) (fun n h => by cases h) (by decide)) ?_
  refine simSL'_seq _ _ _ _ _
    (simSL'_bin (ib := ib) loopR .shr "ls" "op" (.imm 8) (by decide) (fun n h => by cases h) (by decide)) ?_
  refine simSL'_seq _ _ _ _ _
    (simSL'_bin (ib := ib) loopR .band "lb" "ls" (.imm 255) (by decide) (fun n h => by cases h) (by decide)) ?_
  refine simSL'_seq _ _ _ _ _ (simSL'_stgB (ib := ib) loopR "la1" "lb" (by decide) (by decide)) ?_
  refine simSL'_seq _ _ _ _ _
    (simSL'_bin (ib := ib) loopR .add "la2" "la0" (.imm 2) (by decide) (fun n h => by cases h) (by decide)) ?_
  refine simSL'_seq _ _ _ _ _
    (simSL'_bin (ib := ib) loopR .shr "ls" "op" (.imm 16) (by decide) (fun n h => by cases h) (by decide)) ?_
  refine simSL'_seq _ _ _ _ _
    (simSL'_bin (ib := ib) loopR .band "lb" "ls" (.imm 255) (by decide) (fun n h => by cases h) (by decide)) ?_
  refine simSL'_seq _ _ _ _ _ (simSL'_stgB (ib := ib) loopR "la2" "lb" (by decide) (by decide)) ?_
  refine simSL'_seq _ _ _ _ _
    (simSL'_bin (ib := ib) loopR .add "la3" "la0" (.imm 3) (by decide) (fun n h => by cases h) (by decide)) ?_
  refine simSL'_seq _ _ _ _ _
    (simSL'_bin (ib := ib) loopR .shr "ls" "op" (.imm 24) (by decide) (fun n h => by cases h) (by decide)) ?_
  exact simSL'_seq _ _ _ _ _
    (simSL'_bin (ib := ib) loopR .band "lb" "ls" (.imm 255) (by decide) (fun n h => by cases h) (by decide))
    (simSL'_stgB (ib := ib) loopR "la3" "lb" (by decide) (by decide))

/-- The tail's gmem effect: exactly four byte-writes at `outBase+lO .. +3`. -/
theorem tailStmt_gmem (lO : Nat) (ws : WState) (F : Nat) :
    ((tailStmt lO).eval F ws).gmem
      = ((((ws.gmem.set! (ws.regs "outBase" + UInt64.ofNat lO).toNat
              (ws.regs "op" &&& 255).toUInt8).set!
            (ws.regs "outBase" + UInt64.ofNat lO + 1).toNat
              ((ws.regs "op" >>> 8) &&& 255).toUInt8).set!
          (ws.regs "outBase" + UInt64.ofNat lO + 2).toNat
            ((ws.regs "op" >>> 16) &&& 255).toUInt8).set!
        (ws.regs "outBase" + UInt64.ofNat lO + 3).toNat
          ((ws.regs "op" >>> 24) &&& 255).toUInt8) := by
  simp [tailStmt, wseq, WStmt.eval, WState.setReg, WState.stgByte, WArg.eval, WOp.run]

/-- The tail writes ONLY the four length bytes at `outBase+lO … +3`; every other
    address is untouched.  (Generalised from `j < outBase+lO` to both sides, so it
    can carry a full write-confinement frame, not just the window below.) -/
theorem tailStmt_frame (lO : Nat) (ws : WState) (F : Nat) (j : Nat)
    (hj : j < (ws.regs "outBase").toNat + lO ∨ (ws.regs "outBase").toNat + lO + 4 ≤ j)
    (hnw : (ws.regs "outBase").toNat + lO + 3 < 2 ^ 64) :
    ((tailStmt lO).eval F ws).gmem.getD j 0 = ws.gmem.getD j 0 := by
  have hlO : (UInt64.ofNat lO).toNat = lO := by
    rw [AlgorithmLib.LZ4Ptx.toNat_ofNat_lt]; omega
  have hb : (ws.regs "outBase" + UInt64.ofNat lO).toNat = (ws.regs "outBase").toNat + lO := by
    rw [UInt64.toNat_add, hlO, Nat.mod_eq_of_lt (by omega)]
  have h1 : (ws.regs "outBase" + UInt64.ofNat lO + 1).toNat
      = (ws.regs "outBase").toNat + lO + 1 := by
    rw [UInt64.toNat_add, hb, show (1 : UInt64).toNat = 1 from rfl, Nat.mod_eq_of_lt (by omega)]
  have h2 : (ws.regs "outBase" + UInt64.ofNat lO + 2).toNat
      = (ws.regs "outBase").toNat + lO + 2 := by
    rw [UInt64.toNat_add, hb, show (2 : UInt64).toNat = 2 from rfl, Nat.mod_eq_of_lt (by omega)]
  have h3 : (ws.regs "outBase" + UInt64.ofNat lO + 3).toNat
      = (ws.regs "outBase").toNat + lO + 3 := by
    rw [UInt64.toNat_add, hb, show (3 : UInt64).toNat = 3 from rfl, Nat.mod_eq_of_lt (by omega)]
  have e0 : (ws.regs "outBase" + UInt64.ofNat lO).toNat ≠ j := by
    rcases hj with h | h <;> omega
  have e1 : (ws.regs "outBase" + UInt64.ofNat lO + 1).toNat ≠ j := by
    rcases hj with h | h <;> omega
  have e2 : (ws.regs "outBase" + UInt64.ofNat lO + 2).toNat ≠ j := by
    rcases hj with h | h <;> omega
  have e3 : (ws.regs "outBase" + UInt64.ofNat lO + 3).toNat ≠ j := by
    rcases hj with h | h <;> omega
  rw [tailStmt_gmem]
  simp only [Array.set!_eq_setIfInBounds]
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?,
      Array.getElem?_setIfInBounds_ne e3, Array.getElem?_setIfInBounds_ne e2,
      Array.getElem?_setIfInBounds_ne e1, Array.getElem?_setIfInBounds_ne e0]

theorem bodyPrefixSeg_length (iS hL : Nat) : (bodyPrefixSeg iS hL).length = 218 := by rfl

/-- The tail leaves the compressed length readable as a little-endian `u32` at
    `outBase+lO` — exactly the field the host reads to size each block. -/
theorem tailStmt_lenField (lO : Nat) (ws : WState) (F : Nat)
    (hnw : (ws.regs "outBase").toNat + lO + 3 < 2 ^ 64)
    (hsz : (ws.regs "outBase").toNat + lO + 4 ≤ ws.gmem.size)
    (hop : (ws.regs "op").toNat < 2 ^ 32) :
    AlgorithmLib.readU32LE ((tailStmt lO).eval F ws).gmem ((ws.regs "outBase").toNat + lO)
      = (ws.regs "op").toNat := by
  have hlO : (UInt64.ofNat lO).toNat = lO := by
    rw [AlgorithmLib.LZ4Ptx.toNat_ofNat_lt]; omega
  have hb : (ws.regs "outBase" + UInt64.ofNat lO).toNat = (ws.regs "outBase").toNat + lO := by
    rw [UInt64.toNat_add, hlO, Nat.mod_eq_of_lt (by omega)]
  have h1 : (ws.regs "outBase" + UInt64.ofNat lO + 1).toNat
      = (ws.regs "outBase").toNat + lO + 1 := by
    rw [UInt64.toNat_add, hb, show (1 : UInt64).toNat = 1 from rfl, Nat.mod_eq_of_lt (by omega)]
  have h2 : (ws.regs "outBase" + UInt64.ofNat lO + 2).toNat
      = (ws.regs "outBase").toNat + lO + 2 := by
    rw [UInt64.toNat_add, hb, show (2 : UInt64).toNat = 2 from rfl, Nat.mod_eq_of_lt (by omega)]
  have h3 : (ws.regs "outBase" + UInt64.ofNat lO + 3).toNat
      = (ws.regs "outBase").toNat + lO + 3 := by
    rw [UInt64.toNat_add, hb, show (3 : UInt64).toNat = 3 from rfl, Nat.mod_eq_of_lt (by omega)]
  -- the four stored bytes, as `Nat`s
  have e0 : ((ws.regs "op" &&& 255).toUInt8).toNat = (ws.regs "op").toNat % 256 := by
    rw [AlgorithmLib.and255_toUInt8, AlgorithmLib.toUInt8_toNat]
  have ek : ∀ k : Nat, k < 64 →
      (((ws.regs "op" >>> UInt64.ofNat k) &&& 255).toUInt8).toNat
        = (ws.regs "op").toNat / 2 ^ k % 256 := by
    intro k hk
    rw [AlgorithmLib.and255_toUInt8, AlgorithmLib.toUInt8_toNat,
      AlgorithmLib.shiftRight_toNat _ k hk]
  have e8 : (((ws.regs "op" >>> (8 : UInt64)) &&& 255).toUInt8).toNat
      = (ws.regs "op").toNat / 256 % 256 := ek 8 (by omega)
  have e16 : (((ws.regs "op" >>> (16 : UInt64)) &&& 255).toUInt8).toNat
      = (ws.regs "op").toNat / 65536 % 256 := ek 16 (by omega)
  have e24 : (((ws.regs "op" >>> (24 : UInt64)) &&& 255).toUInt8).toNat
      = (ws.regs "op").toNat / 16777216 % 256 := ek 24 (by omega)
  have hs0 : (ws.regs "outBase").toNat + lO < ws.gmem.size := by omega
  have hs1 : (ws.regs "outBase").toNat + lO + 1 < ws.gmem.size := by omega
  have hs2 : (ws.regs "outBase").toNat + lO + 2 < ws.gmem.size := by omega
  have hs3 : (ws.regs "outBase").toNat + lO + 3 < ws.gmem.size := by omega
  rw [AlgorithmLib.readU32LE, tailStmt_gmem]
  simp only [Array.set!_eq_setIfInBounds, Array.getD_eq_getD_getElem?, h3, h2, h1, hb,
    Array.size_setIfInBounds]
  rw [Array.getElem?_setIfInBounds_ne (by omega : (ws.regs "outBase").toNat + lO + 3
        ≠ (ws.regs "outBase").toNat + lO),
    Array.getElem?_setIfInBounds_ne (by omega : (ws.regs "outBase").toNat + lO + 2
        ≠ (ws.regs "outBase").toNat + lO),
    Array.getElem?_setIfInBounds_ne (by omega : (ws.regs "outBase").toNat + lO + 1
        ≠ (ws.regs "outBase").toNat + lO),
    Array.getElem?_setIfInBounds_self_of_lt (by first | omega | (simp only [Array.size_setIfInBounds]; omega)),
    Array.getElem?_setIfInBounds_ne (by omega : (ws.regs "outBase").toNat + lO + 3
        ≠ (ws.regs "outBase").toNat + lO + 1),
    Array.getElem?_setIfInBounds_ne (by omega : (ws.regs "outBase").toNat + lO + 2
        ≠ (ws.regs "outBase").toNat + lO + 1),
    Array.getElem?_setIfInBounds_self_of_lt (by first | omega | (simp only [Array.size_setIfInBounds]; omega)),
    Array.getElem?_setIfInBounds_ne (by omega : (ws.regs "outBase").toNat + lO + 3
        ≠ (ws.regs "outBase").toNat + lO + 2),
    Array.getElem?_setIfInBounds_self_of_lt (by first | omega | (simp only [Array.size_setIfInBounds]; omega)),
    Array.getElem?_setIfInBounds_self_of_lt (by first | omega | (simp only [Array.size_setIfInBounds]; omega))]
  simp only [Option.getD_some]
  rw [e0, e8, e16, e24]
  exact AlgorithmLib.le_reassemble _ (by omega)

end AlgorithmLib.LZ4WarpDSL
