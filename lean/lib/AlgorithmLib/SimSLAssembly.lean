import AlgorithmLib
import AlgorithmLib.LZ4WarpKernelProof
import AlgorithmLib.CoopWindowLeaf
import AlgorithmLib.CoopWindowRelaxed
import AlgorithmLib.CoopCopyLeaf
import AlgorithmLib.EmitContent

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib.LZ4Simt AlgorithmLib.LZ4WarpFind
open AlgorithmLib.LZ4Ptx (toNat_ofNat_lt u64_add_ofNat u64_sub_ofNat)

/-- Extend a coupling with two registers whose (lane-uniform) machine values match
    the sequential values — used to bring `p0`/`cand0` into `R` after `coopWindow`
    (they are excluded from `coopWindow_couple`'s `R` but established by
    `coopWindow_p0_found`/`coopWindow_cand0_found`). -/
theorem Couple.extend2 (R : List String) (ss : SState) (ws : WState) (a b : String)
    (h : Couple R ss ws) (ha : ∀ l : Fin 32, ss.regs a l = ws.regs a)
    (hb : ∀ l : Fin 32, ss.regs b l = ws.regs b) :
    Couple (a :: b :: R) ss ws := by
  refine ⟨h.1, h.2.1, ?_⟩
  intro r hr l
  simp only [List.mem_cons] at hr
  rcases hr with rfl | rfl | hr
  · exact ha l
  · exact hb l
  · exact h.2.2 r hr l

/-- Drop two registers from a coupling (weaken back to the loop-boundary `R0`). -/
theorem Couple.drop2 (R : List String) (ss : SState) (ws : WState) (a b : String)
    (h : Couple (a :: b :: R) ss ws) : Couple R ss ws :=
  ⟨h.1, h.2.1, fun r hr l => h.2.2 r (List.mem_cons_of_mem a (List.mem_cons_of_mem b hr)) l⟩

/-- Every `WStmt.eval` step preserves `gmem.size`: the only gmem writers are
    `stgByte` (`set!`), `coopCopy` (`copyGmem`), and the coop primitives — all
    size-preserving; every other constructor leaves `gmem` untouched. -/
theorem eval_gmem_size : ∀ (fuel : Nat) (s : WStmt) (st : WState),
    (WStmt.eval fuel s st).gmem.size = st.gmem.size := by
  intro fuel s st
  induction fuel, s, st using WStmt.eval.induct <;>
    simp_all [WStmt.eval, WState.setReg, WState.stgByte, WState.stshU16,
      evalCoopWindow, evalCoopWindowGo_gmem, evalCoopExtendStep,
      copyGmem_size, Array.set!_eq_setIfInBounds, Array.size_setIfInBounds]

/-- The constant per-lane machine invariants the coop `_couple` leaves require but
    plain `Couple` does not track: `lane` holds its index, `inBase`/`tbl` are 0.
    (Set once by the prologue, never rewritten by the body.) -/
def MachInv (ib : Nat) (ss : SState) : Prop :=
  (∀ l : Fin 32, ss.regs "lane" l = UInt64.ofNat l.val)
    ∧ (∀ l : Fin 32, ss.regs "inBase" l = UInt64.ofNat ib)
    ∧ (∀ l : Fin 32, ss.regs "tbl" l = 0)

/-- Every `sstepInstr` writes at most the single register `d = wtgt i` (or memory /
    pc only).  Reading it off the instruction lets us frame `MachInv` across a step
    without re-casing the semantics at every use site. -/
def wtgt : SInstr → Option String
  | .mov d _ => some d          | .bin _ d _ _ => some d
  | .binr _ d _ _ => some d     | .cvt32 d _ => some d
  | .brev d _ => some d         | .clz d _ => some d
  | .bnot d _ => some d         | .setp _ d _ _ => some d
  | .andp d _ _ => some d       | .selp d _ _ _ => some d
  | .ldgo d _ _ => some d       | .ldgop _ d _ _ => some d       | .ldsh d _ => some d
  | .vote d _ => some d         | .shfl d _ _ => some d
  | _ => none   -- stg/stgp/stg32p/stsh/stshp/barwarp/bra*/lbl/ret: no reg write

/-- A step whose write-target is not `r` preserves `regs r`. -/
theorem sstepInstr_regs_ne (prog : Array SInstr) (i : SInstr) (st : SState) (r : String)
    (hr : wtgt i ≠ some r) : (sstepInstr prog i st).regs r = st.regs r := by
  cases i <;>
    simp_all only [sstepInstr, SState.setReg, SState.setPc, wtgt, ne_eq, Option.some.injEq] <;>
    first
      | rfl
      | (funext l; rw [if_neg (fun h => hr h.symm)])

/-- `sstep` preserves `regs r` when the instruction at `pc` doesn't write `r`. -/
theorem sstep_regs_ne (prog : Array SInstr) (st : SState) (r : String)
    (hr : ∀ i, prog[st.pc]? = some i → wtgt i ≠ some r) :
    (sstep prog st).regs r = st.regs r := by
  unfold sstep
  cases hpc : prog[st.pc]? with
  | none => rfl
  | some i =>
      cases i <;>
        first
          | rfl
          | exact sstepInstr_regs_ne prog _ st r (hr _ hpc)

/-- `MachInv` is preserved by any step whose instruction writes none of
    `lane`/`inBase`/`tbl`. -/
theorem machInv_sstep (ib : Nat) (prog : Array SInstr) (st : SState) (h : MachInv ib st)
    (hne : ∀ i, prog[st.pc]? = some i →
        wtgt i ≠ some "lane" ∧ wtgt i ≠ some "inBase" ∧ wtgt i ≠ some "tbl") :
    MachInv ib (sstep prog st) := by
  obtain ⟨hla, hib, htb⟩ := h
  refine ⟨fun l => ?_, fun l => ?_, fun l => ?_⟩
  · rw [sstep_regs_ne prog st "lane" (fun i hi => (hne i hi).1)]; exact hla l
  · rw [sstep_regs_ne prog st "inBase" (fun i hi => (hne i hi).2.1)]; exact hib l
  · rw [sstep_regs_ne prog st "tbl" (fun i hi => (hne i hi).2.2)]; exact htb l

/-- Read the `k`-th instruction of a segment out of `SegAt` (as a concrete
    `prog[base+k]? = some i` fact, for feeding the coop `_couple` leaves). -/
theorem segAt_get {prog base emit} (h : SegAt prog base emit) (k : Nat) (i : SInstr)
    (hk : emit[k]? = some i) : prog[base + k]? = some i := by
  have hlt : k < emit.length := by rw [List.getElem?_eq_some_iff] at hk; exact hk.1
  rw [h k hlt, hk]

/-- Strengthened straight-line simulation carrying `MachInv` (the constant per-lane
    invariants `lane = l`, `inBase = tbl = 0`).  Same shape as `SimSL` but both the
    precondition and the reached state include `MachInv`, so it threads through the
    coop leaves (which need those invariants) and composes over the whole body. -/
def SimSL' (ib : Nat) (R : List String) (stmt : WStmt) (emit : List SInstr) : Prop :=
  ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (ws : WState) (fuel : Nat),
    ss.pc = base → SegAt prog base emit → LabelsResolve prog base emit →
    Couple R ss ws → MachInv ib ss →
    ∃ (n : Nat) (ss' : SState),
      SReaches prog n ss ss' ∧ ss'.pc = base + emit.length ∧
      Couple R ss' (stmt.eval fuel ws) ∧ MachInv ib ss'

/-- `SimSL'` for a one-instruction leaf, mirroring `simSL_single` but also carrying
    `MachInv` (which survives the single `sstep` since `i0` is write-safe). -/
theorem simSL'_single (R : List String) (stmt : WStmt) (i0 : SInstr) (T : WState → WState)
    (f : ∀ (prog : Array SInstr) (ss : SState) (ws : WState),
      prog[ss.pc]? = some i0 → Couple R ss ws →
      Couple R (sstep prog ss) (T ws) ∧ (sstep prog ss).pc = ss.pc + 1)
    (heval : ∀ (ws : WState) (fuel : Nat), stmt.eval fuel ws = T ws)
    (hsafe : wtgt i0 ≠ some "lane" ∧ wtgt i0 ≠ some "inBase" ∧ wtgt i0 ≠ some "tbl") :
    SimSL' ib R stmt [i0] := by
  intro prog base ss ws fuel hpc hseg _hlr hc hmi
  have hpci : prog[ss.pc]? = some i0 := by rw [hpc]; exact hseg.head i0 rfl
  obtain ⟨hcpl, hpc'⟩ := f prog ss ws hpci hc
  refine ⟨1, sstep prog ss, sreaches_one prog ss, by simp [hpc', hpc], ?_, ?_⟩
  · rw [heval]; exact hcpl
  · exact machInv_sstep ib prog ss hmi (fun i hi => by rw [hpci] at hi; cases hi; exact hsafe)

/-- `SimSL'` composes over `.seq` exactly like `SimSL`, threading `MachInv`. -/
theorem simSL'_seq (R : List String) (a b : WStmt) (ea eb : List SInstr)
    (sa : SimSL' ib R a ea) (sb : SimSL' ib R b eb) : SimSL' ib R (.seq a b) (ea ++ eb) := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1⟩ :=
    sa prog base ss ws fuel hpc hseg.append_left hlr.append_left hc hmi
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2⟩ :=
    sb prog (base + ea.length) ss1 (a.eval fuel ws) fuel hpc1 hseg.append_right
      hlr.append_right hc1 hmi1
  refine ⟨n1 + n2, ss2, sreaches_trans prog n1 n2 ss ss1 ss2 hr1 hr2, ?_, ?_, hmi2⟩
  · rw [hpc2, List.length_append]; omega
  · have hseval : WStmt.eval fuel (a.seq b) ws = b.eval fuel (a.eval fuel ws) := by
      simp [WStmt.eval]
    rw [hseval]; exact hc2

/-- `SimSL'` mov leaf. -/
theorem simSL'_mov (R : List String) (d : String) (a : WArg)
    (ha : ∀ n, a = .reg n → n ∈ R)
    (hd : d ≠ "lane" ∧ d ≠ "inBase" ∧ d ≠ "tbl") :
    SimSL' ib R (.mov d a) [.mov d a.toS] :=
  simSL'_single R (.mov d a) (.mov d a.toS) (fun ws => ws.setReg d (a.eval ws))
    (fun prog ss ws hpci hc => mov_sound R prog d a ss ws hpci hc ha)
    (fun ws fuel => by simp [WStmt.eval])
    (by simp only [wtgt, ne_eq, Option.some.injEq]; exact ⟨hd.1, hd.2.1, hd.2.2⟩)

/-- `SimSL'` bin leaf. -/
theorem simSL'_bin (R : List String) (o : WOp) (d a : String) (b : WArg)
    (ha : a ∈ R) (hb : ∀ n, b = .reg n → n ∈ R)
    (hd : d ≠ "lane" ∧ d ≠ "inBase" ∧ d ≠ "tbl") :
    SimSL' ib R (.bin o d a b) [.bin o.toS d a b.toS] :=
  simSL'_single R (.bin o d a b) (.bin o.toS d a b.toS)
    (fun ws => ws.setReg d (o.run (ws.regs a) (b.eval ws)))
    (fun prog ss ws hpci hc => bin_sound R prog o d a b ss ws hpci hc ha hb)
    (fun ws fuel => by simp [WStmt.eval])
    (by simp only [wtgt, ne_eq, Option.some.injEq]; exact ⟨hd.1, hd.2.1, hd.2.2⟩)

/-- `SimSL'` setp leaf. -/
theorem simSL'_setp (R : List String) (c : SCmp) (d a : String) (b : WArg)
    (ha : a ∈ R) (hb : ∀ n, b = .reg n → n ∈ R)
    (hd : d ≠ "lane" ∧ d ≠ "inBase" ∧ d ≠ "tbl") :
    SimSL' ib R (.setp c d a b) [.setp c d a b.toS] :=
  simSL'_single R (.setp c d a b) (.setp c d a b.toS)
    (fun ws => ws.setReg d (if c.run (ws.regs a) (b.eval ws) then 1 else 0))
    (fun prog ss ws hpci hc => setp_sound R prog c d a b ss ws hpci hc ha hb)
    (fun ws fuel => by simp [WStmt.eval])
    (by simp only [wtgt, ne_eq, Option.some.injEq]; exact ⟨hd.1, hd.2.1, hd.2.2⟩)

/-- Setting the pc leaves the registers untouched, so `MachInv` survives. -/
theorem machInv_setPc (ss : SState) (n : Nat) (h : MachInv ib ss) : MachInv ib (ss.setPc n) := by
  obtain ⟨hl, hi, ht⟩ := h
  exact ⟨fun l => by simpa [SState.setPc] using hl l,
         fun l => by simpa [SState.setPc] using hi l,
         fun l => by simpa [SState.setPc] using ht l⟩

/-- `SimSL'` for `uwhile`: mirrors `simSL_uwhile` with `MachInv` threaded through the
    loop.  The body preserves `MachInv` (`sb : SimSL' ib R body ebody`); the loop-control
    steps (lbl/braifnot/bra) only touch pc, so they preserve it via `machInv_setPc`. -/
theorem simSL'_uwhile (R : List String) (cond lHead lEnd : String) (body : WStmt)
    (ebody : List SInstr) (prog : Array SInstr) (base : Nat)
    (hcond : cond ∈ R) (sb : SimSL' ib R body ebody)
    (hseg : SegAt prog base (uwhileEmit cond lHead lEnd ebody))
    (hlr : LabelsResolve prog base (uwhileEmit cond lHead lEnd ebody)) :
    ∀ (fuel : Nat) (ss : SState) (ws : WState),
      ss.pc = base → Couple R ss ws → MachInv ib ss → WhileHalts cond body fuel ws →
      ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
        ss'.pc = base + (uwhileEmit cond lHead lEnd ebody).length ∧
        Couple R ss' (WStmt.eval fuel (.uwhile cond body) ws) ∧ MachInv ib ss' := by
  obtain ⟨hlblH, hsegA⟩ := hseg.cons
  obtain ⟨hbrn, hsegB⟩ := hsegA.cons
  have hsegBody : SegAt prog (base + 1 + 1) ebody := hsegB.append_left
  obtain ⟨hbra, hsegD⟩ := hsegB.append_right.cons
  obtain ⟨hlblE, _⟩ := hsegD.cons
  have hlrBody : LabelsResolve prog (base + 1 + 1) ebody := hlr.cons.cons.append_left
  have hLhead : sfindLabel prog lHead = base := by
    have := hlr 0 lHead (by simp [uwhileEmit]); simpa using this
  have hLend : sfindLabel prog lEnd = base + 1 + 1 + ebody.length + 1 :=
    hlr.cons.cons.append_right.cons 0 lEnd (by simp)
  intro fuel
  induction fuel with
  | zero => intro ss ws _ _ _ hH; simp [WhileHalts] at hH
  | succ fuel ih =>
    intro ss ws hpc hc hmi hH
    have hcv : ss.regs cond 0 = ws.regs cond := hc.reg hcond 0
    have hlblH' : prog[ss.pc]? = some (.lbl lHead) := by rw [hpc]; exact hlblH
    have sH : sstep prog ss = ss.setPc (base + 1) := by rw [lbl_step prog ss lHead hlblH', hpc]
    have hbrn' : prog[(ss.setPc (base + 1)).pc]? = some (.braifnot cond lEnd) := hbrn
    by_cases hb : (ws.regs cond == 1) = true
    · have sB0 : sstep prog (ss.setPc (base + 1)) = ss.setPc (base + 1 + 1) := by
        rw [braifnot_step prog _ cond lEnd hbrn']
        simp only [SState.setPc]
        rw [show ss.regs cond 0 = ws.regs cond from hcv, if_pos hb]
      obtain ⟨nb, ss1, hrB, hpcB, hcB, hmiB⟩ :=
        sb prog (base + 1 + 1) (ss.setPc (base + 1 + 1)) ws fuel rfl hsegBody hlrBody
          (couple_setPc hc _) (machInv_setPc ss _ hmi)
      have hbra' : prog[ss1.pc]? = some (.bra lHead) := by rw [hpcB]; exact hbra
      have sBk : sstep prog ss1 = ss1.setPc base := by rw [bra_step prog ss1 lHead hbra', hLhead]
      have hHrec : WhileHalts cond body fuel (body.eval fuel ws) := by
        rw [WhileHalts] at hH; rw [if_pos hb] at hH; exact hH
      obtain ⟨nr, ssf, hrR, hpcR, hcR, hmiR⟩ :=
        ih (ss1.setPc base) (body.eval fuel ws) rfl (couple_setPc hcB _)
          (machInv_setPc ss1 _ hmiB) hHrec
      refine ⟨1 + 1 + nb + 1 + nr, ssf,
        sreaches_trans prog (1 + 1 + nb + 1) nr _ _ _
          (sreaches_trans prog (1 + 1 + nb) 1 _ _ _
            (sreaches_trans prog (1 + 1) nb _ _ _
              (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
              hrB)
            (sreaches_one_eq sBk))
          hrR, hpcR, ?_, hmiR⟩
      have heval : WStmt.eval (fuel + 1) (.uwhile cond body) ws
          = WStmt.eval fuel (.uwhile cond body) (body.eval fuel ws) := by
        simp [WStmt.eval, hb]
      rw [heval]; exact hcR
    · have sB0 : sstep prog (ss.setPc (base + 1))
          = ss.setPc (base + 1 + 1 + ebody.length + 1) := by
        rw [braifnot_step prog _ cond lEnd hbrn']
        simp only [SState.setPc]
        rw [show ss.regs cond 0 = ws.regs cond from hcv, if_neg hb, hLend]
      have hlblE' : prog[(ss.setPc (base + 1 + 1 + ebody.length + 1)).pc]? = some (.lbl lEnd) :=
        hlblE
      have sE : sstep prog (ss.setPc (base + 1 + 1 + ebody.length + 1))
          = ss.setPc (base + 1 + 1 + ebody.length + 1 + 1) := by
        rw [lbl_step prog _ lEnd hlblE']; simp [SState.setPc]
      refine ⟨1 + 1 + 1, ss.setPc (base + 1 + 1 + ebody.length + 1 + 1),
        sreaches_trans prog (1 + 1) 1 _ _ _
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
          (sreaches_one_eq sE), ?_, ?_, machInv_setPc ss _ hmi⟩
      · show (base + 1 + 1 + ebody.length + 1 + 1)
          = base + (uwhileEmit cond lHead lEnd ebody).length
        rw [uwhileEmit_length]; omega
      · have heval : WStmt.eval (fuel + 1) (.uwhile cond body) ws = ws := by
          simp [WStmt.eval, hb]
        rw [heval]; exact couple_setPc hc _

/-- `SimSL'` stgB leaf (writes gmem, not a register — trivially `MachSafe`). -/
theorem simSL'_stgB (R : List String) (addr s : String) (haddr : addr ∈ R) (hs : s ∈ R) :
    SimSL' ib R (.stgB addr s) [.stg addr s] :=
  simSL'_single R (.stgB addr s) (.stg addr s)
    (fun ws => ws.stgByte (ws.regs addr) (ws.regs s))
    (fun prog ss ws hpci hc => stgU_sound R prog addr s ss ws hpci hc haddr hs)
    (fun ws fuel => by simp [WStmt.eval])
    (by simp only [wtgt, ne_eq]; exact ⟨by simp, by simp, by simp⟩)

/-- `SimSL'` for `wStoreByte val` — the `MachInv`-carrying analogue of
    `simSL_wStoreByte`, composed from the `bin`/`stgB` leaves. -/
theorem simSL'_wStoreByte (R : List String) (val : String)
    (hval : val ∈ R) (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R) :
    SimSL' ib R (wStoreByte val)
      ([.bin .add "sbAddr" "outBase" (.reg "op")]
        ++ ([.stg "sbAddr" val] ++ [.bin .add "op" "op" (.imm 1)])) := by
  apply simSL'_seq
  · exact simSL'_bin R .add "sbAddr" "outBase" (.reg "op") hout (fun n h => by cases h; exact hop)
      (by decide)
  · apply simSL'_seq
    · exact simSL'_stgB R "sbAddr" val hsb hval
    · exact simSL'_bin R .add "op" "op" (.imm 1) hop (fun n h => by cases h) (by decide)

/-- `SimSL'` for `wEmitToken litLen tokLo`. -/
theorem simSL'_wEmitToken (R : List String) (litLen tokLo : String)
    (hll : litLen ∈ R) (htl : tokLo ∈ R)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R)
    (htokHi : "tokHi" ∈ R) (htok : "tok" ∈ R) :
    SimSL' ib R (wEmitToken litLen tokLo)
      ([.bin .min "tokHi" litLen (.imm 15)]
        ++ ([.bin .shl "tok" "tokHi" (.imm 4)]
          ++ ([.bin .bor "tok" "tok" (.reg tokLo)]
            ++ ([.bin .add "sbAddr" "outBase" (.reg "op")]
              ++ ([.stg "sbAddr" "tok"] ++ [.bin .add "op" "op" (.imm 1)]))))) := by
  apply simSL'_seq
  · exact simSL'_bin R .min "tokHi" litLen (.imm 15) hll (fun n h => by cases h) (by decide)
  apply simSL'_seq
  · exact simSL'_bin R .shl "tok" "tokHi" (.imm 4) htokHi (fun n h => by cases h) (by decide)
  apply simSL'_seq
  · exact simSL'_bin R .bor "tok" "tok" (.reg tokLo) htok (fun n h => by cases h; exact htl)
      (by decide)
  · exact simSL'_wStoreByte R "tok" htok hout hop hsb

/-- `SimSL'` for the LSIC extend loop body (one iteration): `mov c255; wStoreByte
    c255; bin sub n; setp ge lsicC`. -/
theorem simSL'_wEmitLSIC_body (R : List String) (n : String) (hn : n ∈ R)
    (hnW : n ≠ "lane" ∧ n ≠ "inBase" ∧ n ≠ "tbl")
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R)
    (hc255 : "c255" ∈ R) (hlsicC : "lsicC" ∈ R) :
    SimSL' ib R
      (wseq [ .mov "c255" (.imm 255), wStoreByte "c255", .bin .sub n n (.imm 255),
              .setp .ge "lsicC" n (.imm 255) ])
      ([.mov "c255" (SArg.imm 255)]
        ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
            ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
          ++ ([.bin .sub n n (.imm 255)] ++ [.setp .ge "lsicC" n (.imm 255)]))) := by
  apply simSL'_seq
  · exact simSL'_mov R "c255" (.imm 255) (fun m h => by cases h) (by decide)
  apply simSL'_seq
  · exact simSL'_wStoreByte R "c255" hc255 hout hop hsb
  apply simSL'_seq
  · exact simSL'_bin R .sub n n (.imm 255) hn (fun m h => by cases h) hnW
  · exact simSL'_setp R .ge "lsicC" n (.imm 255) hn (fun m h => by cases h) (by decide)

/-- `SimSL'` for `wEmitLSIC n`.  Manually glues the `lsicC` loop (via `simSL'_uwhile`,
    which needs a `WhileHalts` witness) between the `setp` guard and trailing store —
    loops can't chain through `simSL'_seq`, so this is done by hand. -/
theorem simSL'_wEmitLSIC (R : List String) (n lH lX : String)
    (hn : n ∈ R) (hnW : n ≠ "lane" ∧ n ≠ "inBase" ∧ n ≠ "tbl")
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R)
    (hc255 : "c255" ∈ R) (hlsicC : "lsicC" ∈ R)
    (lsicBody : List SInstr)
    (hbodyDef : lsicBody =
      ([.mov "c255" (SArg.imm 255)]
        ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
            ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
          ++ ([.bin .sub n n (.imm 255)] ++ [.setp .ge "lsicC" n (.imm 255)])))) :
    ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (ws : WState) (fuel : Nat),
      ss.pc = base →
      SegAt prog base
        (([.setp .ge "lsicC" n (.imm 255)] : List SInstr)
          ++ (uwhileEmit "lsicC" lH lX lsicBody
            ++ ([.bin .add "sbAddr" "outBase" (.reg "op")]
              ++ ([.stg "sbAddr" n] ++ [.bin .add "op" "op" (.imm 1)])))) →
      LabelsResolve prog base
        (([.setp .ge "lsicC" n (.imm 255)] : List SInstr)
          ++ (uwhileEmit "lsicC" lH lX lsicBody
            ++ ([.bin .add "sbAddr" "outBase" (.reg "op")]
              ++ ([.stg "sbAddr" n] ++ [.bin .add "op" "op" (.imm 1)])))) →
      Couple R ss ws → MachInv ib ss →
      WhileHalts "lsicC"
        (wseq [ .mov "c255" (.imm 255), wStoreByte "c255", .bin .sub n n (.imm 255),
                .setp .ge "lsicC" n (.imm 255) ]) fuel
        ((WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255)).eval fuel ws) →
      ∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
        ss'.pc = base +
          (([.setp .ge "lsicC" n (.imm 255)] : List SInstr)
            ++ (uwhileEmit "lsicC" lH lX lsicBody
              ++ (([.bin .add "sbAddr" "outBase" (.reg "op")] : List SInstr)
                ++ (([.stg "sbAddr" n] : List SInstr)
                  ++ ([.bin .add "op" "op" (.imm 1)] : List SInstr))))).length ∧
        Couple R ss' ((wEmitLSIC n).eval fuel ws) ∧ MachInv ib ss' := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi hHalt
  -- Segment/label split: `[setp] ++ (uwhile ++ store)`.
  have hsegSetp : SegAt prog base [.setp .ge "lsicC" n (.imm 255)] := hseg.append_left
  have hsegRest : SegAt prog (base + 1)
      (uwhileEmit "lsicC" lH lX lsicBody
        ++ ([.bin .add "sbAddr" "outBase" (.reg "op")]
          ++ ([.stg "sbAddr" n] ++ [.bin .add "op" "op" (.imm 1)]))) := by
    have := hseg.append_right (ea := [.setp .ge "lsicC" n (.imm 255)]); simpa using this
  have hlrRest : LabelsResolve prog (base + 1)
      (uwhileEmit "lsicC" lH lX lsicBody
        ++ ([.bin .add "sbAddr" "outBase" (.reg "op")]
          ++ ([.stg "sbAddr" n] ++ [.bin .add "op" "op" (.imm 1)]))) := by
    have := hlr.append_right (ea := [.setp .ge "lsicC" n (.imm 255)]); simpa using this
  -- Step 1: the guard `setp .ge "lsicC" n 255` (single instruction leaf).
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1⟩ :=
    (simSL'_setp R .ge "lsicC" n (.imm 255) hn (fun m h => by cases h) (by decide))
      prog base ss ws fuel hpc hsegSetp hlr.append_left hc hmi
  -- Step 2: the LSIC loop, via `simSL'_uwhile`.
  have hsegLoop : SegAt prog (base + 1) (uwhileEmit "lsicC" lH lX lsicBody) := hsegRest.append_left
  have hlrLoop : LabelsResolve prog (base + 1) (uwhileEmit "lsicC" lH lX lsicBody) :=
    hlrRest.append_left
  have hbodySL : SimSL' ib R
      (wseq [ .mov "c255" (.imm 255), wStoreByte "c255", .bin .sub n n (.imm 255),
              .setp .ge "lsicC" n (.imm 255) ]) lsicBody := by
    rw [hbodyDef]; exact simSL'_wEmitLSIC_body R n hn hnW hout hop hsb hc255 hlsicC
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2⟩ :=
    simSL'_uwhile R "lsicC" lH lX _ lsicBody prog (base + 1) hlsicC hbodySL hsegLoop hlrLoop
      fuel ss1 ((WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255)).eval fuel ws)
      (by rw [hpc1]; simp) (by simpa [SArg.reg] using hc1) hmi1 hHalt
  -- Step 3: the trailing `wStoreByte n`.
  have hsegStore : SegAt prog (base + 1 + (uwhileEmit "lsicC" lH lX lsicBody).length)
      ([.bin .add "sbAddr" "outBase" (.reg "op")]
        ++ ([.stg "sbAddr" n] ++ [.bin .add "op" "op" (.imm 1)])) := hsegRest.append_right
  have hlrStore : LabelsResolve prog (base + 1 + (uwhileEmit "lsicC" lH lX lsicBody).length)
      ([.bin .add "sbAddr" "outBase" (.reg "op")]
        ++ ([.stg "sbAddr" n] ++ [.bin .add "op" "op" (.imm 1)])) := hlrRest.append_right
  obtain ⟨n3, ss3, hr3, hpc3, hc3, hmi3⟩ :=
    (simSL'_wStoreByte R n hn hout hop hsb) prog
      (base + 1 + (uwhileEmit "lsicC" lH lX lsicBody).length) ss2
      ((WStmt.uwhile "lsicC"
        (wseq [ .mov "c255" (.imm 255), wStoreByte "c255", .bin .sub n n (.imm 255),
                .setp .ge "lsicC" n (.imm 255) ])).eval fuel
        ((WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255)).eval fuel ws)) fuel
      (by rw [hpc2]) hsegStore hlrStore hc2 hmi2
  refine ⟨n1 + (n2 + n3), ss3,
    sreaches_trans prog n1 (n2 + n3) _ _ _ hr1 (sreaches_trans prog n2 n3 _ _ _ hr2 hr3),
    ?_, ?_, hmi3⟩
  · rw [hpc3]
    simp only [List.length_append, List.length_cons, List.length_nil]
    omega
  · -- assemble the eval: `wEmitLSIC n = setp ; uwhile ; wStoreByte`.
    have heval : (wEmitLSIC n).eval fuel ws
        = (wStoreByte n).eval fuel
          ((WStmt.uwhile "lsicC"
            (wseq [ .mov "c255" (.imm 255), wStoreByte "c255", .bin .sub n n (.imm 255),
                    .setp .ge "lsicC" n (.imm 255) ])).eval fuel
            ((WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255)).eval fuel ws)) := by
      simp only [wEmitLSIC, wseq, WStmt.eval]
    rw [heval]; exact hc3

-- ── `simSL_coopExtendStep`: lift `coopExtendStep_couple` to `SimSL` ──────────────

/-- The registers `coopExtendStep_couple` reads uniformly (must be in `R`) and the
    machine invariants it needs (`inBase = 0`, `lane = l`), packaged for reuse. -/
structure ExtCtx (R : List String) : Prop where
  p0 : "p0" ∈ R
  cand0 : "cand0" ∈ R
  ml : "ml" ∈ R
  ecR : "ecR" ∈ R
  ec1 : "ec1" ∈ R
  ib : "inBase" ∈ R
  adv : "adv" ∈ R

/-- `SimSL'` for the extend step.  The added coupling-side value hypotheses
    (`hecRv`/`hec1v` — `ecR`/`ec1` hold `endCap`/`endCap-1`) and no-overflow bounds
    (`hp64`/`hml64`) are what the body establishes just before the step; they let the
    per-lane `hbound`/`hbytes` reduce to the `Nat`-level extend semantics. -/
theorem simSL_coopExtendStep (R : List String) (inStride endCap : Nat)
    (hctx : ExtCtx R)
    (hendCap : endCap = inStride - 5)
    (hRdisj : ∀ r ∈ R, r ∉ ["idx", "pe", "pIn", "peC", "dfe", "caC", "peD", "aP", "caD", "aC",
               "bP", "bC", "pEqB", "pOk", "balOk", "mis", "revM"])
    -- Coupling-side (`ws`) restatements of the leaf's per-lane conditions, which the
    -- body assembly discharges from the loop invariant (`p < searchLim`, `ml` bounds,
    -- `ecR = endCap`, `ec1 = endCap-1`, `inBase = 0`).  Given here so the machine
    -- (`ss`) versions follow by `Couple`/`MachInv` substitution.
    (ws : WState)
    (hboundW : ∀ l : Fin 32,
        ((ws.regs "p0") + ((ws.regs "ml") + UInt64.ofNat l.val) < ws.regs "ecR")
          ↔ (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) < endCap)
    (hbytesW : ∀ l : Fin 32,
        (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) < endCap →
        (UInt64.ofNat (ws.gmem.getD
            (SOp.run .add (ws.regs "inBase")
              (SOp.run .min (ws.regs "p0" + (ws.regs "ml" + UInt64.ofNat l.val))
                (ws.regs "ec1"))).toNat 0).toNat
         == UInt64.ofNat (ws.gmem.getD
            (SOp.run .add (ws.regs "inBase")
              (ws.regs "cand0" + (SOp.run .sub
                (SOp.run .min (ws.regs "p0" + (ws.regs "ml" + UInt64.ofNat l.val)) (ws.regs "ec1"))
                (ws.regs "p0")))).toNat 0).toNat)
          = (byte (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                ((ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val))
             == byte (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                ((ws.regs "cand0").toNat + ((ws.regs "ml").toNat + l.val)))) :
    -- (`ws` is now an explicit arg; `SimSL'` is instantiated at exactly this `ws`.)
    ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (fuel : Nat),
      ss.pc = base → SegAt prog base (coopExtendEmit "adv") →
      LabelsResolve prog base (coopExtendEmit "adv") →
      Couple R ss ws → MachInv ib ss →
      ∃ (n : Nat) (ss' : SState),
        SReaches prog n ss ss' ∧ ss'.pc = base + (coopExtendEmit "adv").length ∧
        Couple R ss' ((WStmt.coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap).eval fuel ws)
          ∧ MachInv ib ss' := by
  intro prog base ss fuel hpc hseg _hlr hc hmi
  -- Extract the 18 instruction facts from `SegAt (coopExtendEmit "adv")`.
  have e0 := segAt_get hseg 0 _ (by rfl)
  have e1 := segAt_get hseg 1 _ (by rfl)
  have e2 := segAt_get hseg 2 _ (by rfl)
  have e3 := segAt_get hseg 3 _ (by rfl)
  have e4 := segAt_get hseg 4 _ (by rfl)
  have e5 := segAt_get hseg 5 _ (by rfl)
  have e6 := segAt_get hseg 6 _ (by rfl)
  have e7 := segAt_get hseg 7 _ (by rfl)
  have e8 := segAt_get hseg 8 _ (by rfl)
  have e9 := segAt_get hseg 9 _ (by rfl)
  have e10 := segAt_get hseg 10 _ (by rfl)
  have e11 := segAt_get hseg 11 _ (by rfl)
  have e12 := segAt_get hseg 12 _ (by rfl)
  have e13 := segAt_get hseg 13 _ (by rfl)
  have e14 := segAt_get hseg 14 _ (by rfl)
  have e15 := segAt_get hseg 15 _ (by rfl)
  have e16 := segAt_get hseg 16 _ (by rfl)
  have e17 := segAt_get hseg 17 _ (by rfl)
  -- MachInv ib survives all 18 steps via `coopExtendStep_frame` (lane/inBase/tbl are
  -- not among the primitive's scratch registers).
  have hframe : ∀ r, r ∉ ["idx", "pe", "pIn", "peC", "dfe", "caC", "peD", "aP", "caD", "aC",
      "bP", "bC", "pEqB", "pOk", "balOk", "mis", "revM", "adv"] →
      (snsteps prog 18 ss).regs r = ss.regs r := fun r hr =>
    coopExtendStep_frame prog ss base r hpc e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13 e14 e15
      e16 e17 hr
  have hmi' : MachInv ib (snsteps prog 18 ss) := by
    refine ⟨fun l => ?_, fun l => ?_, fun l => ?_⟩
    · rw [hframe "lane" (by decide)]; exact hmi.1 l
    · rw [hframe "inBase" (by decide)]; exact hmi.2.1 l
    · rw [hframe "tbl" (by decide)]; exact hmi.2.2 l
  -- Coupling via `coopExtendStep_couple`; discharge `hbound`/`hbytes` from `Couple`
  -- (uniform `p0`/`ml`/`cand0`/`ecR`/`ec1`/`inBase` = ws, gmem-eq) + `MachInv`
  -- (`lane l = l`, `inBase = 0`).
  have hbound : ∀ l : Fin 32,
      ((SOp.run .add (ss.regs "p0" l)
          (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l))) < ss.regs "ecR" l)
        ↔ (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) < endCap := by
    intro l
    rw [hc.reg hctx.p0 l, hc.reg hctx.ml l, hmi.1 l, hc.reg hctx.ecR l]
    show (ws.regs "p0") + ((ws.regs "ml") + UInt64.ofNat l.val) < ws.regs "ecR" ↔ _
    exact hboundW l
  have hbytes : ∀ l : Fin 32,
      (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) < endCap →
      (UInt64.ofNat
          (ss.gmem.getD
            (SOp.run .add (ss.regs "inBase" l)
              (SOp.run .min
                (SOp.run .add (ss.regs "p0" l)
                  (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l)))
                (ss.regs "ec1" l))).toNat 0).toNat
       ==
       UInt64.ofNat
          (ss.gmem.getD
            (SOp.run .add (ss.regs "inBase" l)
              (SOp.run .add (ss.regs "cand0" l)
                (SOp.run .sub
                  (SOp.run .min
                    (SOp.run .add (ss.regs "p0" l)
                      (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l)))
                    (ss.regs "ec1" l))
                  (ss.regs "p0" l)))).toNat 0).toNat)
        = (byte (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
              ((ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val))
           == byte (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
              ((ws.regs "cand0").toNat + ((ws.regs "ml").toNat + l.val))) := by
    intro l hlt
    rw [hc.1, hc.reg hctx.p0 l, hc.reg hctx.ml l, hmi.1 l, hc.reg hctx.cand0 l,
      hc.reg hctx.ec1 l, hc.reg hctx.ib l]
    exact hbytesW l hlt
  obtain ⟨hreach, hpc18, hcpl⟩ := coopExtendStep_couple R prog ss ws base inStride endCap hpc
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13 e14 e15 e16 e17 hc hctx.adv hRdisj
    hbound hbytes
  refine ⟨18, snsteps prog 18 ss, hreach, ?_, ?_, hmi'⟩
  · rw [hpc18]; rfl
  · have heval : WStmt.eval fuel (WStmt.coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap) ws
      = evalCoopExtendStep "adv" "p0" "cand0" "ml" inStride endCap ws := by
      simp only [WStmt.eval]
    rw [heval]; exact hcpl

-- ── `simSL_coopWindow`: lift `coopWindow_couple` to the `SimSL'` shape ───────────

/-- `SimSL'`-shaped simulation for the window step.  Threads `MachInv` (giving the
    leaf's `inBase=0`/`tbl=0`/`lane=l`) and takes the remaining loop-invariant facts
    (`searchPos` uniform `= s`, bounds) at `ws`/`s` level — discharged by the body. -/
theorem simSL_coopWindow (R : List String) (inStride searchLim hashLog s : Nat)
    (hsp0 : "searchPos" ∈ R) (hfound : "found" ∈ R) (hinbR : "inBase" ∈ R)
    (hib40 : ib < 2 ^ 40)
    (hsl : searchLim ≤ inStride - 4) (hcapb : inStride - 4 < 2 ^ 64)
    (hhl : hashLog ≤ 32) (hlen : inStride < 2 ^ 40)
    (hp64 : s + 32 < 2 ^ 64)
    (hp0R : "p0" ∉ R) (hcand0R : "cand0" ∉ R)
    (hRdisj : ∀ r ∈ R, r ∉ ["posP", "pValid", "cap4", "rp", "rpA", "b0", "b1", "b2", "b3", "v32",
        "hh", "addr", "candRaw", "cand", "rc", "rcA", "c0", "c1", "c2", "c3", "vc",
        "pNE", "pCO", "pEq", "pH1", "pH2", "pHit", "bal", "rev", "fl", "pLe",
        "pIns", "pp1"])
    (ws : WState) (hsval : (ws.regs "searchPos").toNat = s)
    (hspU : ∀ l : Fin 32, (∀ ss : SState, Couple R ss ws → ss.regs "searchPos" l = UInt64.ofNat s)) :
    ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (fuel : Nat),
      ss.pc = base →
      SegAt prog base (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride searchLim hashLog) →
      LabelsResolve prog base
        (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride searchLim hashLog) →
      Couple R ss ws → MachInv ib ss →
      ∃ (n : Nat) (ss' : SState),
        SReaches prog n ss ss' ∧
        ss'.pc = base +
          (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride searchLim hashLog).length ∧
        Couple R ss'
          ((WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0).eval
            fuel ws)
          ∧ MachInv ib ss' := by
  intro prog base ss fuel hpc hseg _hlr hc hmi
  -- Extract the 51 instruction facts.
  have e0 := segAt_get hseg 0 _ (by rfl)
  have e1 := segAt_get hseg 1 _ (by rfl)
  have e2 := segAt_get hseg 2 _ (by rfl)
  have e3 := segAt_get hseg 3 _ (by rfl)
  have e4 := segAt_get hseg 4 _ (by rfl)
  have e5 := segAt_get hseg 5 _ (by rfl)
  have e6 := segAt_get hseg 6 _ (by rfl)
  have e7 := segAt_get hseg 7 _ (by rfl)
  have e8 := segAt_get hseg 8 _ (by rfl)
  have e9 := segAt_get hseg 9 _ (by rfl)
  have e10 := segAt_get hseg 10 _ (by rfl)
  have e11 := segAt_get hseg 11 _ (by rfl)
  have e12 := segAt_get hseg 12 _ (by rfl)
  have e13 := segAt_get hseg 13 _ (by rfl)
  have e14 := segAt_get hseg 14 _ (by rfl)
  have e15 := segAt_get hseg 15 _ (by rfl)
  have e16 := segAt_get hseg 16 _ (by rfl)
  have e17 := segAt_get hseg 17 _ (by rfl)
  have e18 := segAt_get hseg 18 _ (by rfl)
  have e19 := segAt_get hseg 19 _ (by rfl)
  have e20 := segAt_get hseg 20 _ (by rfl)
  have e21 := segAt_get hseg 21 _ (by rfl)
  have e22 := segAt_get hseg 22 _ (by rfl)
  have e23 := segAt_get hseg 23 _ (by rfl)
  have e24 := segAt_get hseg 24 _ (by rfl)
  have e25 := segAt_get hseg 25 _ (by rfl)
  have e26 := segAt_get hseg 26 _ (by rfl)
  have e27 := segAt_get hseg 27 _ (by rfl)
  have e28 := segAt_get hseg 28 _ (by rfl)
  have e29 := segAt_get hseg 29 _ (by rfl)
  have e30 := segAt_get hseg 30 _ (by rfl)
  have e31 := segAt_get hseg 31 _ (by rfl)
  have e32 := segAt_get hseg 32 _ (by rfl)
  have e33 := segAt_get hseg 33 _ (by rfl)
  have e34 := segAt_get hseg 34 _ (by rfl)
  have e35 := segAt_get hseg 35 _ (by rfl)
  have e36 := segAt_get hseg 36 _ (by rfl)
  have e37 := segAt_get hseg 37 _ (by rfl)
  have e38 := segAt_get hseg 38 _ (by rfl)
  have e39 := segAt_get hseg 39 _ (by rfl)
  have e40 := segAt_get hseg 40 _ (by rfl)
  have e41 := segAt_get hseg 41 _ (by rfl)
  have e42 := segAt_get hseg 42 _ (by rfl)
  have e43 := segAt_get hseg 43 _ (by rfl)
  have e44 := segAt_get hseg 44 _ (by rfl)
  have e45 := segAt_get hseg 45 _ (by rfl)
  have e46 := segAt_get hseg 46 _ (by rfl)
  have e47 := segAt_get hseg 47 _ (by rfl)
  have e48 := segAt_get hseg 48 _ (by rfl)
  have e49 := segAt_get hseg 49 _ (by rfl)
  have e50 := segAt_get hseg 50 _ (by rfl)
  -- MachInv ib survives via `coopWindow_frame` (lane/inBase/tbl not in scratch list).
  have hframe : ∀ r, r ∉ ["posP", "pValid", "cap4", "rp", "rpA", "b0", "b1", "b2", "b3", "v32",
      "hh", "addr", "candRaw", "cand", "rc", "rcA", "c0", "c1", "c2", "c3", "vc",
      "pNE", "pCO", "pEq", "pH1", "pH2", "pHit", "bal", "rev", "fl", "pLe",
      "pIns", "pp1", "p0", "cand0", "found"] →
      (snsteps prog 51 ss).regs r = ss.regs r := fun r hr =>
    (coopWindow_frame prog ss base r searchLim (inStride - 4) (32 - hashLog) (2 ^ hashLog - 1) hpc
      e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13 e14 e15 e16 e17 e18 e19 e20 e21 e22 e23 e24 e25
      e26 e27 e28 e29 e30 e31 e32 e33 e34 e35 e36 e37 e38 e39 e40 e41 e42 e43 e44 e45 e46 e47 e48
      e49 e50 hr).1
  have hmi' : MachInv ib (snsteps prog 51 ss) := by
    refine ⟨fun l => ?_, fun l => ?_, fun l => ?_⟩
    · rw [hframe "lane" (by decide)]; exact hmi.1 l
    · rw [hframe "inBase" (by decide)]; exact hmi.2.1 l
    · rw [hframe "tbl" (by decide)]; exact hmi.2.2 l
  -- Coupling via `coopWindow_couple_relaxed`, discharging its invariants from `MachInv`+hyps.
  have hwsib : ws.regs "inBase" = UInt64.ofNat ib := by
    rw [← hc.reg hinbR 0]; exact hmi.2.1 0
  obtain ⟨hreach, hpc51, hcpl⟩ := coopWindow_couple_relaxed R prog ss ws base s inStride searchLim
    (inStride - 4) hashLog ib hpc
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13 e14 e15 e16 e17 e18 e19 e20 e21 e22 e23 e24 e25
    e26 e27 e28 e29 e30 e31 e32 e33 e34 e35 e36 e37 e38 e39 e40 e41 e42 e43 e44 e45 e46 e47 e48
    e49 e50 hc hsl rfl hcapb hhl (fun l => hmi.2.1 l) hlen (fun l => hmi.2.2 l) (fun l => hmi.1 l)
    (fun l => hspU l ss hc) hsval hwsib hp64 hp0R hcand0R hRdisj (by omega)
  refine ⟨51, snsteps prog 51 ss, hreach, ?_, ?_, hmi'⟩
  · rw [hpc51]; rfl
  · have heval : WStmt.eval fuel
        (WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0) ws
      = evalCoopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0 ws := by
      simp only [WStmt.eval]
    rw [heval]; exact hcpl

/-- **coopWindow simulates AND couples `p0`/`cand0` in the found case.**  Wraps
    `coopWindow_couple_relaxed` (the `R0`-coupling excluding `p0`/`cand0`), the machine
    frame (`MachInv`), and the item-1 lemmas `coopWindow_p0_found`/`coopWindow_cand0_found`
    (which pin the machine `p0`/`cand0` to `ofNat p`/`ofNat c`) via `Couple.extend2`. -/
theorem coopWindow_couple_found (R0 : List String) (inStride searchLim hashLog s p c : Nat)
    (hsp0 : "searchPos" ∈ R0) (hfound : "found" ∈ R0) (hinbR : "inBase" ∈ R0)
    (hib40 : ib < 2 ^ 40)
    (hsl : searchLim ≤ inStride - 4) (hcapb : inStride - 4 < 2 ^ 64)
    (hhl : hashLog ≤ 32) (hlen : inStride < 2 ^ 40)
    (hp64 : s + 32 < 2 ^ 64)
    (hp0R : "p0" ∉ R0) (hcand0R : "cand0" ∉ R0)
    (hRdisj : ∀ r ∈ R0, r ∉ ["posP", "pValid", "cap4", "rp", "rpA", "b0", "b1", "b2", "b3", "v32",
        "hh", "addr", "candRaw", "cand", "rc", "rcA", "c0", "c1", "c2", "c3", "vc",
        "pNE", "pCO", "pEq", "pH1", "pH2", "pHit", "bal", "rev", "fl", "pLe",
        "pIns", "pp1"])
    (ws : WState) (hsval : (ws.regs "searchPos").toNat = s)
    (hspU : ∀ l : Fin 32, (∀ ss : SState, Couple R0 ss ws → ss.regs "searchPos" l = UInt64.ofNat s))
    (hwin : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) searchLim s = some (p, c)) :
    ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (fuel : Nat),
      ss.pc = base →
      SegAt prog base (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride searchLim hashLog) →
      LabelsResolve prog base
        (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride searchLim hashLog) →
      Couple R0 ss ws → MachInv ib ss →
      ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
        ss'.pc = base +
          (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride searchLim hashLog).length ∧
        Couple ("p0" :: "cand0" :: R0) ss'
          (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0 ws)
          ∧ MachInv ib ss' := by
  intro prog base ss fuel hpc hseg _hlr hc hmi
  have e0 := segAt_get hseg 0 _ (by rfl)
  have e1 := segAt_get hseg 1 _ (by rfl)
  have e2 := segAt_get hseg 2 _ (by rfl)
  have e3 := segAt_get hseg 3 _ (by rfl)
  have e4 := segAt_get hseg 4 _ (by rfl)
  have e5 := segAt_get hseg 5 _ (by rfl)
  have e6 := segAt_get hseg 6 _ (by rfl)
  have e7 := segAt_get hseg 7 _ (by rfl)
  have e8 := segAt_get hseg 8 _ (by rfl)
  have e9 := segAt_get hseg 9 _ (by rfl)
  have e10 := segAt_get hseg 10 _ (by rfl)
  have e11 := segAt_get hseg 11 _ (by rfl)
  have e12 := segAt_get hseg 12 _ (by rfl)
  have e13 := segAt_get hseg 13 _ (by rfl)
  have e14 := segAt_get hseg 14 _ (by rfl)
  have e15 := segAt_get hseg 15 _ (by rfl)
  have e16 := segAt_get hseg 16 _ (by rfl)
  have e17 := segAt_get hseg 17 _ (by rfl)
  have e18 := segAt_get hseg 18 _ (by rfl)
  have e19 := segAt_get hseg 19 _ (by rfl)
  have e20 := segAt_get hseg 20 _ (by rfl)
  have e21 := segAt_get hseg 21 _ (by rfl)
  have e22 := segAt_get hseg 22 _ (by rfl)
  have e23 := segAt_get hseg 23 _ (by rfl)
  have e24 := segAt_get hseg 24 _ (by rfl)
  have e25 := segAt_get hseg 25 _ (by rfl)
  have e26 := segAt_get hseg 26 _ (by rfl)
  have e27 := segAt_get hseg 27 _ (by rfl)
  have e28 := segAt_get hseg 28 _ (by rfl)
  have e29 := segAt_get hseg 29 _ (by rfl)
  have e30 := segAt_get hseg 30 _ (by rfl)
  have e31 := segAt_get hseg 31 _ (by rfl)
  have e32 := segAt_get hseg 32 _ (by rfl)
  have e33 := segAt_get hseg 33 _ (by rfl)
  have e34 := segAt_get hseg 34 _ (by rfl)
  have e35 := segAt_get hseg 35 _ (by rfl)
  have e36 := segAt_get hseg 36 _ (by rfl)
  have e37 := segAt_get hseg 37 _ (by rfl)
  have e38 := segAt_get hseg 38 _ (by rfl)
  have e39 := segAt_get hseg 39 _ (by rfl)
  have e40 := segAt_get hseg 40 _ (by rfl)
  have e41 := segAt_get hseg 41 _ (by rfl)
  have e42 := segAt_get hseg 42 _ (by rfl)
  have e43 := segAt_get hseg 43 _ (by rfl)
  have e44 := segAt_get hseg 44 _ (by rfl)
  have e45 := segAt_get hseg 45 _ (by rfl)
  have e46 := segAt_get hseg 46 _ (by rfl)
  have e47 := segAt_get hseg 47 _ (by rfl)
  have e48 := segAt_get hseg 48 _ (by rfl)
  have e49 := segAt_get hseg 49 _ (by rfl)
  have e50 := segAt_get hseg 50 _ (by rfl)
  have hframe : ∀ r, r ∉ ["posP", "pValid", "cap4", "rp", "rpA", "b0", "b1", "b2", "b3", "v32",
      "hh", "addr", "candRaw", "cand", "rc", "rcA", "c0", "c1", "c2", "c3", "vc",
      "pNE", "pCO", "pEq", "pH1", "pH2", "pHit", "bal", "rev", "fl", "pLe",
      "pIns", "pp1", "p0", "cand0", "found"] →
      (snsteps prog 51 ss).regs r = ss.regs r := fun r hr =>
    (coopWindow_frame prog ss base r searchLim (inStride - 4) (32 - hashLog) (2 ^ hashLog - 1) hpc
      e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13 e14 e15 e16 e17 e18 e19 e20 e21 e22 e23 e24 e25 e26 e27 e28 e29 e30 e31 e32 e33 e34 e35 e36 e37 e38 e39 e40 e41 e42 e43 e44 e45 e46 e47 e48 e49 e50 hr).1
  have hmi' : MachInv ib (snsteps prog 51 ss) := by
    refine ⟨fun l => ?_, fun l => ?_, fun l => ?_⟩
    · rw [hframe "lane" (by decide)]; exact hmi.1 l
    · rw [hframe "inBase" (by decide)]; exact hmi.2.1 l
    · rw [hframe "tbl" (by decide)]; exact hmi.2.2 l
  have hwsib : ws.regs "inBase" = UInt64.ofNat ib := by
    rw [← hc.reg hinbR 0]; exact hmi.2.1 0
  have hwsibN : (ws.regs "inBase").toNat = ib := by
    rw [hwsib]; exact AlgorithmLib.LZ4Ptx.toNat_ofNat_lt ib (by omega)
  obtain ⟨hreach, hpc51, hcpl⟩ := coopWindow_couple_relaxed R0 prog ss ws base s inStride searchLim
    (inStride - 4) hashLog ib hpc
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13 e14 e15 e16 e17 e18 e19 e20 e21 e22 e23 e24 e25 e26 e27 e28 e29 e30 e31 e32 e33 e34 e35 e36 e37 e38 e39 e40 e41 e42 e43 e44 e45 e46 e47 e48 e49 e50 hc hsl rfl hcapb hhl (fun l => hmi.2.1 l) hlen (fun l => hmi.2.2 l) (fun l => hmi.1 l)
    (fun l => hspU l ss hc) hsval hwsib hp64 hp0R hcand0R hRdisj (by omega)
  have hwin_ss : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ss.gmem ib inStride)
      (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s = some (p, c) := by
    rw [hc.gmem, hc.smem, ← hwsibN]; exact hwin
  have hp0m := coopWindow_p0_found prog ss base s inStride searchLim (inStride - 4) hashLog p c ib hpc
    e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13 e14 e15 e16 e17 e18 e19 e20 e21 e22 e23 e24 e25 e26 e27 e28 e29 e30 e31 e32 e33 e34 e35 e36 e37 e38 e39 e40 e41 e42 e43 e44 e45 e46 e47 e48 e49 e50 hsl rfl hcapb hhl (fun l => hmi.2.1 l) hlen (fun l => hmi.2.2 l) (fun l => hmi.1 l)
    (fun l => hspU l ss hc) hp64 hwin_ss (by omega)
  have hcand0m := coopWindow_cand0_found prog ss base s inStride searchLim (inStride - 4) hashLog p c ib
    hpc e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13 e14 e15 e16 e17 e18 e19 e20 e21 e22 e23 e24 e25 e26 e27 e28 e29 e30 e31 e32 e33 e34 e35 e36 e37 e38 e39 e40 e41 e42 e43 e44 e45 e46 e47 e48 e49 e50 hsl rfl hcapb hhl (fun l => hmi.2.1 l) hlen (fun l => hmi.2.2 l) (fun l => hmi.1 l)
    (fun l => hspU l ss hc) hp64 hwin_ss (by omega)
  have hp0e : (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0 ws).regs
      "p0" = UInt64.ofNat p := by
    rw [evalCoopWindow_eq_go, evalCoopWindowGo_regs, hsval, hwin]; simp [WState.setReg]
  have hcand0e : (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0 ws).regs
      "cand0" = UInt64.ofNat c := by
    rw [evalCoopWindow_eq_go, evalCoopWindowGo_regs, hsval, hwin]; simp [WState.setReg]
  refine ⟨51, snsteps prog 51 ss, hreach, by rw [hpc51]; rfl, ?_, hmi'⟩
  apply Couple.extend2 R0 (snsteps prog 51 ss) _ "p0" "cand0" hcpl
  · intro l; rw [hp0m l, hp0e]
  · intro l; rw [hcand0m l, hcand0e]

-- ── `simSL_coopCopy`: lift `coopCopy_couple` (with its exposed frame) ────────────

/-- `SimSL'`-shaped simulation for a `.coopCopy dst src len` step, whose emit is
    `[mov cpI; setp cpCont] ++ uwhileEmit "cpCont" lH lX (coopCopyBody ...)`.  Uses
    the frame `coopCopy_couple` now exposes to preserve `MachInv`. -/
theorem simSL_coopCopy (R : List String) (dst src len lH lX : String)
    (hdst : dst ∈ R) (hsrc : src ∈ R) (hlenR : len ∈ R)
    (hRdisj : ∀ r ∈ R, r ∉ coopCopyScratch)
    (ws : WState)
    (hb1 : (ws.regs dst).toNat < 2 ^ 32) (hb2 : (ws.regs src).toNat < 2 ^ 32)
    (hb3 : (ws.regs len).toNat < 2 ^ 32)
    (hdisjW : (ws.regs dst).toNat + (ws.regs len).toNat ≤ (ws.regs src).toNat
      ∨ (ws.regs src).toNat + (ws.regs len).toNat ≤ (ws.regs dst).toNat)
    (hsizeW : (ws.regs dst).toNat + (ws.regs len).toNat ≤ ws.gmem.size) :
    ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (fuel : Nat),
      ss.pc = base →
      SegAt prog base
        (([.mov "cpI" (.imm 0), .setp .lt "cpCont" "cpI" (.reg len)] : List SInstr)
          ++ uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)) →
      LabelsResolve prog base
        (([.mov "cpI" (.imm 0), .setp .lt "cpCont" "cpI" (.reg len)] : List SInstr)
          ++ uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)) →
      Couple R ss ws → MachInv ib ss →
      ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
        ss'.pc = base +
          (([.mov "cpI" (.imm 0), .setp .lt "cpCont" "cpI" (.reg len)] : List SInstr)
            ++ uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)).length ∧
        Couple R ss' ((WStmt.coopCopy dst src len).eval fuel ws) ∧ MachInv ib ss' := by
  intro prog base ss fuel hpc hseg hlr hc hmi
  obtain ⟨n, ss', hreach, hpcE, hcpl, hframeE⟩ :=
    coopCopy_couple R dst src len lH lX prog base ss ws fuel hpc hseg hlr hc
      hdst hsrc hlenR hRdisj (fun l => hmi.1 l) hb1 hb2 hb3 hdisjW hsizeW
  refine ⟨n, ss', hreach, hpcE, hcpl, ?_⟩
  refine ⟨fun l => ?_, fun l => ?_, fun l => ?_⟩
  · rw [hframeE "lane" (by simp [coopCopyScratch]) l]; exact hmi.1 l
  · rw [hframeE "inBase" (by simp [coopCopyScratch]) l]; exact hmi.2.1 l
  · rw [hframeE "tbl" (by simp [coopCopyScratch]) l]; exact hmi.2.2 l

-- ── extC loop body: `coopExtendStep ; bin add ml ; setp eq extC` ─────────────────

/-- `SimSL'` for one iteration of the `extC` extend loop body.  Glues the
    `coopExtendStep` wrapper (bespoke — carries `ws`/side-conditions) with the
    `bin`/`setp` leaves by hand (coop isn't a plain `SimSL'`). -/
theorem simSL'_extCBody (R : List String) (inStride endCap : Nat)
    (hctx : ExtCtx R) (hendCap : endCap = inStride - 5)
    (hml : "ml" ∈ R) (hadv : "adv" ∈ R) (hextC : "extC" ∈ R)
    (hRdisj : ∀ r ∈ R, r ∉ ["idx", "pe", "pIn", "peC", "dfe", "caC", "peD", "aP", "caD", "aC",
               "bP", "bC", "pEqB", "pOk", "balOk", "mis", "revM"])
    (ws : WState)
    (hboundW : ∀ l : Fin 32,
        ((ws.regs "p0") + ((ws.regs "ml") + UInt64.ofNat l.val) < ws.regs "ecR")
          ↔ (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) < endCap)
    (hbytesW : ∀ l : Fin 32,
        (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) < endCap →
        (UInt64.ofNat (ws.gmem.getD
            (SOp.run .add (ws.regs "inBase")
              (SOp.run .min (ws.regs "p0" + (ws.regs "ml" + UInt64.ofNat l.val))
                (ws.regs "ec1"))).toNat 0).toNat
         == UInt64.ofNat (ws.gmem.getD
            (SOp.run .add (ws.regs "inBase")
              (ws.regs "cand0" + (SOp.run .sub
                (SOp.run .min (ws.regs "p0" + (ws.regs "ml" + UInt64.ofNat l.val)) (ws.regs "ec1"))
                (ws.regs "p0")))).toNat 0).toNat)
          = (byte (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                ((ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val))
             == byte (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                ((ws.regs "cand0").toNat + ((ws.regs "ml").toNat + l.val)))) :
    ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (fuel : Nat),
      ss.pc = base →
      SegAt prog base
        (coopExtendEmit "adv"
          ++ (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
            ++ [.setp .eq "extC" "adv" (.imm 32)])) →
      LabelsResolve prog base
        (coopExtendEmit "adv"
          ++ (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
            ++ [.setp .eq "extC" "adv" (.imm 32)])) →
      Couple R ss ws → MachInv ib ss →
      ∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
        ss'.pc = base +
          (coopExtendEmit "adv"
            ++ (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
              ++ ([.setp .eq "extC" "adv" (.imm 32)] : List SInstr))).length ∧
        Couple R ss'
          ((wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
                   .bin .add "ml" "ml" (.reg "adv"),
                   .setp .eq "extC" "adv" (.imm 32) ]).eval fuel ws)
          ∧ MachInv ib ss' := by
  intro prog base ss fuel hpc hseg hlr hc hmi
  have hsegCoop : SegAt prog base (coopExtendEmit "adv") := hseg.append_left
  have hlrCoop : LabelsResolve prog base (coopExtendEmit "adv") := hlr.append_left
  -- Step 1: the coop-extend segment (18 instrs), via the bespoke wrapper.
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1⟩ :=
    simSL_coopExtendStep R inStride endCap hctx hendCap hRdisj ws hboundW hbytesW
      prog base ss fuel hpc hsegCoop hlrCoop hc hmi
  have hcoopLen : (coopExtendEmit "adv").length = 18 := by decide
  -- Step 2/3: `bin add ml` then `setp eq extC` (both plain `SimSL'` leaves).
  have hsegRest : SegAt prog (base + (coopExtendEmit "adv").length)
      (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
        ++ [.setp .eq "extC" "adv" (.imm 32)]) := hseg.append_right
  have hlrRest : LabelsResolve prog (base + (coopExtendEmit "adv").length)
      (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
        ++ [.setp .eq "extC" "adv" (.imm 32)]) := hlr.append_right
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2⟩ :=
    (simSL'_seq R (.bin .add "ml" "ml" (.reg "adv")) (.setp .eq "extC" "adv" (.imm 32))
      _ _
      (simSL'_bin R .add "ml" "ml" (.reg "adv") hml (fun m h => by cases h; exact hadv) (by decide))
      (simSL'_setp R .eq "extC" "adv" (.imm 32) hadv (fun m h => by cases h) (by decide)))
      prog (base + (coopExtendEmit "adv").length) ss1
      (evalCoopExtendStep "adv" "p0" "cand0" "ml" inStride endCap ws) fuel
      (by rw [hpc1]) hsegRest hlrRest
      (by have : WStmt.eval fuel (WStmt.coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap) ws
            = evalCoopExtendStep "adv" "p0" "cand0" "ml" inStride endCap ws := by simp only [WStmt.eval]
          rw [← this]; exact hc1)
      hmi1
  refine ⟨n1 + n2, ss2, sreaches_trans prog n1 n2 _ _ _ hr1 hr2, ?_, ?_, hmi2⟩
  · rw [hpc2]; simp only [List.length_append, List.length_cons, List.length_nil]; omega
  · have heval : (wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
          .bin .add "ml" "ml" (.reg "adv"), .setp .eq "extC" "adv" (.imm 32) ]).eval fuel ws
        = (WStmt.seq (.bin .add "ml" "ml" (.reg "adv")) (.setp .eq "extC" "adv" (.imm 32))).eval fuel
            (evalCoopExtendStep "adv" "p0" "cand0" "ml" inStride endCap ws) := by
      simp only [wseq, WStmt.eval]
    rw [heval]; exact hc2

-- ── General custom loop induction for a state-dependent (coop) body ──────────────

/-- Invariant-carrying `BodySim`: simulates only from states satisfying `P`, and
    preserves `P` (`P (bodyStmt.eval fuel ws)`).  The form the loop induction needs
    when the body's coop side-conditions hold only under the loop invariant `P`. -/
def BodySimInv (ib : Nat) (R : List String) (P : WState → Prop) (bodyStmt : WStmt) (bodyEmit : List SInstr) :
    Prop :=
  ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (ws : WState) (fuel : Nat),
    ss.pc = base → SegAt prog base bodyEmit → LabelsResolve prog base bodyEmit →
    Couple R ss ws → MachInv ib ss → P ws →
    (∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧ ss'.pc = base + bodyEmit.length ∧
      Couple R ss' (bodyStmt.eval fuel ws) ∧ MachInv ib ss')
    ∧ P (bodyStmt.eval fuel ws)

/-- `uwhile` loop induction threading a loop invariant `P` (preserved by the body,
    required by its coop side-conditions).  The machine-level analogue of the
    invariant-threading in `emitLoop_eq`. -/
theorem simSL'_uwhileBodyInv (R : List String) (cond lHead lEnd : String) (P : WState → Prop)
    (bodyStmt : WStmt) (bodyEmit : List SInstr) (prog : Array SInstr) (base : Nat)
    (hcond : cond ∈ R) (hbody : BodySimInv ib R P bodyStmt bodyEmit)
    (hseg : SegAt prog base (uwhileEmit cond lHead lEnd bodyEmit))
    (hlr : LabelsResolve prog base (uwhileEmit cond lHead lEnd bodyEmit)) :
    ∀ (fuel : Nat) (ss : SState) (ws : WState),
      ss.pc = base → Couple R ss ws → MachInv ib ss → P ws → WhileHalts cond bodyStmt fuel ws →
      ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
        ss'.pc = base + (uwhileEmit cond lHead lEnd bodyEmit).length ∧
        Couple R ss' (WStmt.eval fuel (.uwhile cond bodyStmt) ws) ∧ MachInv ib ss' := by
  obtain ⟨hlblH, hsegA⟩ := hseg.cons
  obtain ⟨hbrn, hsegB⟩ := hsegA.cons
  have hsegBody : SegAt prog (base + 1 + 1) bodyEmit := hsegB.append_left
  obtain ⟨hbra, hsegD⟩ := hsegB.append_right.cons
  obtain ⟨hlblE, _⟩ := hsegD.cons
  have hlrBody : LabelsResolve prog (base + 1 + 1) bodyEmit := hlr.cons.cons.append_left
  have hLhead : sfindLabel prog lHead = base := by
    have := hlr 0 lHead (by simp [uwhileEmit]); simpa using this
  have hLend : sfindLabel prog lEnd = base + 1 + 1 + bodyEmit.length + 1 :=
    hlr.cons.cons.append_right.cons 0 lEnd (by simp)
  intro fuel
  induction fuel with
  | zero => intro ss ws _ _ _ _ hH; simp [WhileHalts] at hH
  | succ fuel ih =>
    intro ss ws hpc hc hmi hP hH
    have hcv : ss.regs cond 0 = ws.regs cond := hc.reg hcond 0
    have hlblH' : prog[ss.pc]? = some (.lbl lHead) := by rw [hpc]; exact hlblH
    have sH : sstep prog ss = ss.setPc (base + 1) := by rw [lbl_step prog ss lHead hlblH', hpc]
    have hbrn' : prog[(ss.setPc (base + 1)).pc]? = some (.braifnot cond lEnd) := hbrn
    by_cases hb : (ws.regs cond == 1) = true
    · have sB0 : sstep prog (ss.setPc (base + 1)) = ss.setPc (base + 1 + 1) := by
        rw [braifnot_step prog _ cond lEnd hbrn']
        simp only [SState.setPc]
        rw [show ss.regs cond 0 = ws.regs cond from hcv, if_pos hb]
      obtain ⟨⟨nb, ss1, hrB, hpcB, hcB, hmiB⟩, hPnext⟩ :=
        hbody prog (base + 1 + 1) (ss.setPc (base + 1 + 1)) ws fuel rfl hsegBody hlrBody
          (couple_setPc hc _) (machInv_setPc ss _ hmi) hP
      have hbra' : prog[ss1.pc]? = some (.bra lHead) := by rw [hpcB]; exact hbra
      have sBk : sstep prog ss1 = ss1.setPc base := by rw [bra_step prog ss1 lHead hbra', hLhead]
      have hHrec : WhileHalts cond bodyStmt fuel (bodyStmt.eval fuel ws) := by
        rw [WhileHalts] at hH; rw [if_pos hb] at hH; exact hH
      obtain ⟨nr, ssf, hrR, hpcR, hcR, hmiR⟩ :=
        ih (ss1.setPc base) (bodyStmt.eval fuel ws) rfl (couple_setPc hcB _)
          (machInv_setPc ss1 _ hmiB) hPnext hHrec
      refine ⟨1 + 1 + nb + 1 + nr, ssf,
        sreaches_trans prog (1 + 1 + nb + 1) nr _ _ _
          (sreaches_trans prog (1 + 1 + nb) 1 _ _ _
            (sreaches_trans prog (1 + 1) nb _ _ _
              (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
              hrB)
            (sreaches_one_eq sBk))
          hrR, hpcR, ?_, hmiR⟩
      have heval : WStmt.eval (fuel + 1) (.uwhile cond bodyStmt) ws
          = WStmt.eval fuel (.uwhile cond bodyStmt) (bodyStmt.eval fuel ws) := by
        simp [WStmt.eval, hb]
      rw [heval]; exact hcR
    · have sB0 : sstep prog (ss.setPc (base + 1))
          = ss.setPc (base + 1 + 1 + bodyEmit.length + 1) := by
        rw [braifnot_step prog _ cond lEnd hbrn']
        simp only [SState.setPc]
        rw [show ss.regs cond 0 = ws.regs cond from hcv, if_neg hb, hLend]
      have hlblE' : prog[(ss.setPc (base + 1 + 1 + bodyEmit.length + 1)).pc]? = some (.lbl lEnd) :=
        hlblE
      have sE : sstep prog (ss.setPc (base + 1 + 1 + bodyEmit.length + 1))
          = ss.setPc (base + 1 + 1 + bodyEmit.length + 1 + 1) := by
        rw [lbl_step prog _ lEnd hlblE']; simp [SState.setPc]
      refine ⟨1 + 1 + 1, ss.setPc (base + 1 + 1 + bodyEmit.length + 1 + 1),
        sreaches_trans prog (1 + 1) 1 _ _ _
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
          (sreaches_one_eq sE), ?_, ?_, machInv_setPc ss _ hmi⟩
      · show (base + 1 + 1 + bodyEmit.length + 1 + 1)
          = base + (uwhileEmit cond lHead lEnd bodyEmit).length
        rw [uwhileEmit_length]; omega
      · have heval : WStmt.eval (fuel + 1) (.uwhile cond bodyStmt) ws = ws := by
          simp [WStmt.eval, hb]
        rw [heval]; exact couple_setPc hc _

/-- Bespoke `uwhile` loop engine for loops whose body needs a LOWER bound on the
    (decreasing) fuel.  Threads a state invariant `Q` plus a `Nat` measure `μ` the body
    advances by ≥1; the coupling `F ≤ μ ws + fuel` is preserved and, with the loop
    guard, hands the body a fuel floor (`F ≤ μ ws + fuel + 1`).  The body is a plain
    simulation lemma taking `Q` + guard + the fuel bound explicitly — avoiding the
    antitone-in-fuel requirement of `simSL'_uwhileBodyInvF`, which the encode loop's
    `searchPos`/fuel measure cannot satisfy (fuel decrements 1 while `searchPos`
    advances ≥1, so no fixed-state antitone `P` exists). -/
theorem simSL'_measureLoop (R : List String) (cond lHead lEnd : String)
    (Q : WState → Prop) (μ : WState → Nat) (F : Nat)
    (bodyStmt : WStmt) (bodyEmit : List SInstr) (prog : Array SInstr) (base : Nat)
    (hcond : cond ∈ R)
    (hbody : ∀ (ss : SState) (ws : WState) (fuel : Nat),
      ss.pc = base + 1 + 1 → SegAt prog (base + 1 + 1) bodyEmit →
      LabelsResolve prog (base + 1 + 1) bodyEmit →
      Couple R ss ws → MachInv ib ss → Q ws → (ws.regs cond == 1) = true → F ≤ μ ws + fuel + 1 →
      (∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
        ss'.pc = base + 1 + 1 + bodyEmit.length ∧
        Couple R ss' (bodyStmt.eval fuel ws) ∧ MachInv ib ss')
      ∧ Q (bodyStmt.eval fuel ws) ∧ μ ws + 1 ≤ μ (bodyStmt.eval fuel ws))
    (hseg : SegAt prog base (uwhileEmit cond lHead lEnd bodyEmit))
    (hlr : LabelsResolve prog base (uwhileEmit cond lHead lEnd bodyEmit)) :
    ∀ (fuel : Nat) (ss : SState) (ws : WState),
      ss.pc = base → Couple R ss ws → MachInv ib ss → Q ws → F ≤ μ ws + fuel →
      WhileHalts cond bodyStmt fuel ws →
      ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
        ss'.pc = base + (uwhileEmit cond lHead lEnd bodyEmit).length ∧
        Couple R ss' (WStmt.eval fuel (.uwhile cond bodyStmt) ws) ∧ MachInv ib ss' := by
  obtain ⟨hlblH, hsegA⟩ := hseg.cons
  obtain ⟨hbrn, hsegB⟩ := hsegA.cons
  have hsegBody : SegAt prog (base + 1 + 1) bodyEmit := hsegB.append_left
  obtain ⟨hbra, hsegD⟩ := hsegB.append_right.cons
  obtain ⟨hlblE, _⟩ := hsegD.cons
  have hlrBody : LabelsResolve prog (base + 1 + 1) bodyEmit := hlr.cons.cons.append_left
  have hLhead : sfindLabel prog lHead = base := by
    have := hlr 0 lHead (by simp [uwhileEmit]); simpa using this
  have hLend : sfindLabel prog lEnd = base + 1 + 1 + bodyEmit.length + 1 :=
    hlr.cons.cons.append_right.cons 0 lEnd (by simp)
  intro fuel
  induction fuel with
  | zero => intro ss ws _ _ _ _ _ hH; simp [WhileHalts] at hH
  | succ fuel ih =>
    intro ss ws hpc hc hmi hQ hcoup hH
    have hcv : ss.regs cond 0 = ws.regs cond := hc.reg hcond 0
    have hlblH' : prog[ss.pc]? = some (.lbl lHead) := by rw [hpc]; exact hlblH
    have sH : sstep prog ss = ss.setPc (base + 1) := by rw [lbl_step prog ss lHead hlblH', hpc]
    have hbrn' : prog[(ss.setPc (base + 1)).pc]? = some (.braifnot cond lEnd) := hbrn
    by_cases hb : (ws.regs cond == 1) = true
    · have sB0 : sstep prog (ss.setPc (base + 1)) = ss.setPc (base + 1 + 1) := by
        rw [braifnot_step prog _ cond lEnd hbrn']
        simp only [SState.setPc]
        rw [show ss.regs cond 0 = ws.regs cond from hcv, if_pos hb]
      obtain ⟨⟨nb, ss1, hrB, hpcB, hcB, hmiB⟩, hQnext, hμnext⟩ :=
        hbody (ss.setPc (base + 1 + 1)) ws fuel rfl hsegBody hlrBody
          (couple_setPc hc _) (machInv_setPc ss _ hmi) hQ hb (by omega)
      have hbra' : prog[ss1.pc]? = some (.bra lHead) := by rw [hpcB]; exact hbra
      have sBk : sstep prog ss1 = ss1.setPc base := by rw [bra_step prog ss1 lHead hbra', hLhead]
      have hHrec : WhileHalts cond bodyStmt fuel (bodyStmt.eval fuel ws) := by
        rw [WhileHalts] at hH; rw [if_pos hb] at hH; exact hH
      obtain ⟨nr, ssf, hrR, hpcR, hcR, hmiR⟩ :=
        ih (ss1.setPc base) (bodyStmt.eval fuel ws) rfl (couple_setPc hcB _)
          (machInv_setPc ss1 _ hmiB) hQnext (by omega) hHrec
      refine ⟨1 + 1 + nb + 1 + nr, ssf,
        sreaches_trans prog (1 + 1 + nb + 1) nr _ _ _
          (sreaches_trans prog (1 + 1 + nb) 1 _ _ _
            (sreaches_trans prog (1 + 1) nb _ _ _
              (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
              hrB)
            (sreaches_one_eq sBk))
          hrR, hpcR, ?_, hmiR⟩
      have heval : WStmt.eval (fuel + 1) (.uwhile cond bodyStmt) ws
          = WStmt.eval fuel (.uwhile cond bodyStmt) (bodyStmt.eval fuel ws) := by
        simp [WStmt.eval, hb]
      rw [heval]; exact hcR
    · have sB0 : sstep prog (ss.setPc (base + 1))
          = ss.setPc (base + 1 + 1 + bodyEmit.length + 1) := by
        rw [braifnot_step prog _ cond lEnd hbrn']
        simp only [SState.setPc]
        rw [show ss.regs cond 0 = ws.regs cond from hcv, if_neg hb, hLend]
      have hlblE' : prog[(ss.setPc (base + 1 + 1 + bodyEmit.length + 1)).pc]? = some (.lbl lEnd) :=
        hlblE
      have sE : sstep prog (ss.setPc (base + 1 + 1 + bodyEmit.length + 1))
          = ss.setPc (base + 1 + 1 + bodyEmit.length + 1 + 1) := by
        rw [lbl_step prog _ lEnd hlblE']; simp [SState.setPc]
      refine ⟨1 + 1 + 1, ss.setPc (base + 1 + 1 + bodyEmit.length + 1 + 1),
        sreaches_trans prog (1 + 1) 1 _ _ _
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
          (sreaches_one_eq sE), ?_, ?_, machInv_setPc ss _ hmi⟩
      · show (base + 1 + 1 + bodyEmit.length + 1 + 1)
          = base + (uwhileEmit cond lHead lEnd bodyEmit).length
        rw [uwhileEmit_length]; omega
      · have heval : WStmt.eval (fuel + 1) (.uwhile cond bodyStmt) ws = ws := by
          simp [WStmt.eval, hb]
        rw [heval]; exact couple_setPc hc _

/-- The not-found branch (`searchPos += 32`) as a `SimSL'`. -/
theorem simSL'_notFoundBranch (R : List String) (hsp : "searchPos" ∈ R) :
    SimSL' ib R (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ])
      ([.bin .add "searchPos" "searchPos" (.imm 32)] : List SInstr) := by
  have h := simSL'_bin (ib := ib) R .add "searchPos" "searchPos" (.imm 32) hsp
    (fun n h => by cases h) (by decide)
  simpa [wseq] using h

/-- `coopWindow` only writes `found`/`p0`/`cand0` (and `smem`); every other register —
    `op`, `outBase`, `inBase`, `litAnchor`, `searchPos` — is preserved. -/
theorem evalCoopWindow_regs_frame (found p0 cand0 sp : String)
    (inStride searchLim hashLog tbl : Nat) (st : WState) (r : String)
    (hf : r ≠ found) (hp : r ≠ p0) (hc : r ≠ cand0) :
    (evalCoopWindow found p0 cand0 sp inStride searchLim hashLog tbl st).regs r = st.regs r := by
  rw [evalCoopWindow_eq_go, evalCoopWindowGo_regs]
  split <;> simp [WState.setReg, hf, hp, hc]

/-- `coopWindow`'s `found` register: `1` on a window hit, `0` otherwise. -/
theorem coopWindow_found_val (inStride hashLog : Nat) (ws : WState) :
    (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws).regs
        "found"
      = (match AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) (ws.regs "searchPos").toNat with
         | some _ => 1 | none => 0) := by
  rw [evalCoopWindow_eq_go, evalCoopWindowGo_regs]
  cases AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
      (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) (ws.regs "searchPos").toNat with
  | none => simp [WState.setReg]
  | some pc => obtain ⟨p, c⟩ := pc; simp [WState.setReg]

/-- **The found-branch preconditions, from the window semantics.**  When `coopWindow`
    reports a match (`found = 1`), the eval-level `window` returned `some (p, c)`, whose
    soundness (`window_sound`: `c < p` and `p < searchLim = inStride - 12`) pins
    `cand0 = c < p = p0` and `p0 + 4 ≤ inStride - 5 = endCap`.  These discharge
    `extInv_of_foundMovs`'s hypotheses inside the loopC body. -/
theorem coopWindow_found_bounds (inStride hashLog : Nat) (ws : WState)
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride)
    (hfound1 : ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
        hashLog 0 ws).regs "found") = 1) :
    ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws).regs
        "cand0").toNat
      < ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws).regs
        "p0").toNat
    ∧ ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws).regs
        "p0").toNat + 4 ≤ inStride - 5 := by
  rw [evalCoopWindow_eq_go] at hfound1 ⊢
  cases hw : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
      (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) (ws.regs "searchPos").toNat with
  | none =>
      rw [hw, evalCoopWindowGo_regs] at hfound1
      simp only [WState.setReg, if_pos] at hfound1
      exact absurd hfound1 (by decide)
  | some pc =>
      obtain ⟨p, c⟩ := pc
      obtain ⟨hsp, hpsl, hcp, _⟩ := AlgorithmLib.LZ4WarpFind.window_sound
        (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12)
        (ws.regs "searchPos").toNat p c hw
      have hpN : p < 2 ^ 64 := by omega
      have hcN : c < 2 ^ 64 := by omega
      have hp0 : (evalCoopWindowGo "found" "p0" "cand0" "searchPos" hashLog 0 (inStride - 12)
          (ws.regs "inBase").toNat           (some (p, c)) ws).regs "p0" = UInt64.ofNat p := by
        rw [evalCoopWindowGo_regs]; simp [WState.setReg]
      have hcand0 : (evalCoopWindowGo "found" "p0" "cand0" "searchPos" hashLog 0 (inStride - 12)
          (ws.regs "inBase").toNat           (some (p, c)) ws).regs "cand0" = UInt64.ofNat c := by
        rw [evalCoopWindowGo_regs]; simp [WState.setReg]
      rw [hp0, hcand0, toNat_ofNat_lt p hpN, toNat_ofNat_lt c hcN]
      exact ⟨hcp, by omega⟩

/-- Widening a byte through `UInt64.ofNat _.toNat` is injective, so a `BEq` on the
    widened values matches the `BEq` on the original bytes (the machine loads bytes
    into u64 registers; the DSL compares raw bytes). -/
theorem byteEqBridge (a b : UInt8) :
    (UInt64.ofNat a.toNat == UInt64.ofNat b.toNat) = (a == b) := by
  have ha : a.toNat < 2 ^ 64 := by have := a.toNat_lt; omega
  have hb : b.toNat < 2 ^ 64 := by have := b.toNat_lt; omega
  rw [Bool.eq_iff_iff, beq_iff_eq, beq_iff_eq, ← UInt64.toNat_inj,
    toNat_ofNat_lt a.toNat ha, toNat_ofNat_lt b.toNat hb]
  constructor
  · intro h; exact UInt8.toNat_inj.mp h
  · intro h; exact congrArg UInt8.toNat h

/-- The extend loop's invariant: `ecR`/`ec1` pin the endCap constants, the
    candidate lies strictly before the match position, and the running match
    length keeps the extend position within `endCap`.  It discharges the coop
    leaf's `hboundW`/`hbytesW` and is preserved by `ml += leadingGood`
    (`leadingGood_pos_le`). -/
def extInv (inStride endCap : Nat) : WState → Prop :=
  fun ws => ws.regs "ecR" = UInt64.ofNat endCap
    ∧ ws.regs "ec1" = UInt64.ofNat (endCap - 1)
    ∧ (ws.regs "cand0").toNat < (ws.regs "p0").toNat
    ∧ (ws.regs "p0").toNat + (ws.regs "ml").toNat ≤ endCap

/-- The four found-branch setup statements (`ecR := endCap; ec1 := endCap-1;
    ml := 4; extC := ml ≥ 0`) establish `extInv`, given the match-position facts
    `cand0 < p0` (candidate before match) and `p0 + 4 ≤ endCap` (from `window_sound`:
    `p < searchLim = inStride-12 ⟹ p+4 ≤ inStride-8 < endCap`). -/
theorem extInv_of_foundMovs (endCap : Nat) (inStride : Nat) (fuel : Nat) (ws : WState)
    (hcand : (ws.regs "cand0").toNat < (ws.regs "p0").toNat)
    (hp4 : (ws.regs "p0").toNat + 4 ≤ endCap) :
    extInv inStride endCap
      ((wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
               .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval fuel ws) := by
  have hml4 : ((wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
      .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval fuel ws).regs "ml"
      = UInt64.ofNat 4 := by simp [wseq, WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [wseq, WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  · simp [wseq, WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  · have hcand' : ((wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
        .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval fuel ws).regs "cand0"
        = ws.regs "cand0" ∧ ((wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
        .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval fuel ws).regs "p0"
        = ws.regs "p0" := by
      constructor <;> simp [wseq, WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
    rw [hcand'.1, hcand'.2]; exact hcand
  · have hp0' : ((wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
        .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval fuel ws).regs "p0"
        = ws.regs "p0" := by simp [wseq, WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
    rw [hp0', hml4, show ((UInt64.ofNat 4)).toNat = 4 from by decide]; exact hp4

/-- `BodySimInv` for one iteration of the `extC` extend-loop body, under `extInv`.
    Wraps `simSL'_extCBody`, discharging its per-lane `hboundW`/`hbytesW` side
    conditions from `extInv` (bounds ⇒ no UInt64 overflow; the machine's clamped
    load address `min(p0+ml+l, ec1)` collapses to `p0+ml+l` in-range, matching the
    DSL byte), and proving `extInv` is preserved by the body. -/
theorem bodySimInv_extCBody (R : List String) (inStride endCap : Nat)
    (hctx : ExtCtx R) (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hib40 : ib < 2 ^ 40)
    (hml : "ml" ∈ R) (hadv : "adv" ∈ R) (hextC : "extC" ∈ R)
    (hRdisj : ∀ r ∈ R, r ∉ ["idx", "pe", "pIn", "peC", "dfe", "caC", "peD", "aP", "caD", "aC",
               "bP", "bC", "pEqB", "pOk", "balOk", "mis", "revM"]) :
    BodySimInv ib R (extInv inStride endCap)
      (wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
              .bin .add "ml" "ml" (.reg "adv"),
              .setp .eq "extC" "adv" (.imm 32) ])
      (coopExtendEmit "adv"
        ++ (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
          ++ [.setp .eq "extC" "adv" (.imm 32)])) := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi hP
  obtain ⟨hecR, hec1, hcandlt, hpml⟩ := hP
  have hend40 : endCap < 2 ^ 40 := by omega
  -- The extend body's `wseq` unfolds to the `.seq`-shape `simSL'_extCBody` proves.
  have hbodyEq : (wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
        .bin .add "ml" "ml" (.reg "adv"), .setp .eq "extC" "adv" (.imm 32) ])
      = (WStmt.seq (.coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap)
          (WStmt.seq (.bin .add "ml" "ml" (.reg "adv"))
            (.setp .eq "extC" "adv" (.imm 32)))) := by rfl
  -- Machine sum `p0 + (ml + l)` as a Nat, no overflow (bounded by `endCap < 2^40`).
  have hsumN : ∀ l : Fin 32,
      (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) < endCap →
      ((ws.regs "p0") + ((ws.regs "ml") + UInt64.ofNat l.val)).toNat
        = (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) := by
    intro l hlt
    have hl32 : l.val < 32 := l.isLt
    rw [UInt64.toNat_add, UInt64.toNat_add, toNat_ofNat_lt l.val (by omega),
      Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  -- ── `hboundW`: `p0 + (ml + l) < ecR ↔ p0.toNat + (ml.toNat + l) < endCap`. ──
  have hboundW : ∀ l : Fin 32,
      ((ws.regs "p0") + ((ws.regs "ml") + UInt64.ofNat l.val) < ws.regs "ecR")
        ↔ (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) < endCap := by
    intro l
    have hl32 : l.val < 32 := l.isLt
    rw [hecR, UInt64.lt_iff_toNat_lt, toNat_ofNat_lt endCap (by omega)]
    constructor
    · intro h
      by_cases hlt : (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) < endCap
      · exact hlt
      · rw [UInt64.toNat_add, UInt64.toNat_add, toNat_ofNat_lt l.val (by omega)] at h
        omega
    · intro hlt; rw [hsumN l hlt]; exact hlt
  -- ── `hbytesW`: the machine's clamped loads match the DSL bytes, in-range. ──
  have hwsibN : (ws.regs "inBase").toNat = ib := by
    rw [← hc.reg hctx.ib 0, hmi.2.1 0]
    exact AlgorithmLib.LZ4Ptx.toNat_ofNat_lt ib (by omega)
  have hgi : ∀ j, j < inStride →
      (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride).getD j 0
        = ws.gmem.getD ((ws.regs "inBase").toNat + j) 0 := by
    intro j hj
    simp only [gmemInpAt, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range, hj,
      Option.map_some, Option.getD_some]
  have hbytesW : ∀ l : Fin 32,
      (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) < endCap →
      (UInt64.ofNat (ws.gmem.getD
          (SOp.run .add (ws.regs "inBase")
            (SOp.run .min (ws.regs "p0" + (ws.regs "ml" + UInt64.ofNat l.val))
              (ws.regs "ec1"))).toNat 0).toNat
       == UInt64.ofNat (ws.gmem.getD
          (SOp.run .add (ws.regs "inBase")
            (ws.regs "cand0" + (SOp.run .sub
              (SOp.run .min (ws.regs "p0" + (ws.regs "ml" + UInt64.ofNat l.val)) (ws.regs "ec1"))
              (ws.regs "p0")))).toNat 0).toNat)
        = (byte (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
              ((ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val))
           == byte (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
              ((ws.regs "cand0").toNat + ((ws.regs "ml").toNat + l.val))) := by
    intro l hlt
    have hl32 : l.val < 32 := l.isLt
    -- the machine sum as a UInt64.ofNat.
    have hsum : (ws.regs "p0" + (ws.regs "ml" + UInt64.ofNat l.val))
        = UInt64.ofNat ((ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val)) := by
      rw [← UInt64.toNat_inj, hsumN l hlt, toNat_ofNat_lt _ (by omega)]
    -- `min(sum, ec1) = sum` since `sum ≤ endCap-1 = ec1`.
    have hminNoClamp : SOp.run .min (ws.regs "p0" + (ws.regs "ml" + UInt64.ofNat l.val))
        (ws.regs "ec1")
        = UInt64.ofNat ((ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val)) := by
      rw [hsum, hec1]
      show (if UInt64.ofNat _ ≤ UInt64.ofNat (endCap - 1) then _ else _) = _
      rw [if_pos (by rw [UInt64.le_iff_toNat_le, toNat_ofNat_lt _ (by omega),
        toNat_ofNat_lt _ (by omega)]; omega)]
    -- first machine position = p0+ml+l.
    have hpos1 : (SOp.run .add (ws.regs "inBase")
        (SOp.run .min (ws.regs "p0" + (ws.regs "ml" + UInt64.ofNat l.val))
          (ws.regs "ec1"))).toNat
        = (ws.regs "inBase").toNat + ((ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val)) := by
      show (ws.regs "inBase" + _).toNat = _
      rw [hminNoClamp, UInt64.toNat_add, hwsibN, toNat_ofNat_lt _ (by omega),
        Nat.mod_eq_of_lt (by omega)]
    -- `min(sum,ec1) - p0 = ml + l`.
    have hsub : (SOp.run .sub
        (SOp.run .min (ws.regs "p0" + (ws.regs "ml" + UInt64.ofNat l.val)) (ws.regs "ec1"))
        (ws.regs "p0"))
        = UInt64.ofNat ((ws.regs "ml").toNat + l.val) := by
      rw [hminNoClamp]
      show (UInt64.ofNat ((ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val))
        - ws.regs "p0") = _
      rw [← UInt64.toNat_inj, UInt64.toNat_sub, toNat_ofNat_lt _ (by omega),
        toNat_ofNat_lt _ (by omega)]
      rw [show 2 ^ 64 - (ws.regs "p0").toNat + ((ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val))
            = 2 ^ 64 + ((ws.regs "ml").toNat + l.val) from by omega,
        Nat.add_mod_left, Nat.mod_eq_of_lt (by omega)]
    -- second machine position = cand0+ml+l.
    have hpos2 : (SOp.run .add (ws.regs "inBase")
        (ws.regs "cand0" + (SOp.run .sub
          (SOp.run .min (ws.regs "p0" + (ws.regs "ml" + UInt64.ofNat l.val)) (ws.regs "ec1"))
          (ws.regs "p0")))).toNat
        = (ws.regs "inBase").toNat + ((ws.regs "cand0").toNat + ((ws.regs "ml").toNat + l.val)) := by
      show (ws.regs "inBase" + _).toNat = _
      rw [hsub, UInt64.toNat_add, UInt64.toNat_add, hwsibN, toNat_ofNat_lt _ (by omega),
        Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
    rw [hpos1, hpos2]
    -- DSL bytes: `byte = getD`, and gmemInp = gmem in-range; the `UInt64.ofNat _.toNat`
    -- wrapper is injective on bytes, so `==` matches on both sides.
    show (UInt64.ofNat (ws.gmem.getD _ 0).toNat == UInt64.ofNat (ws.gmem.getD _ 0).toNat)
      = ((gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride).getD _ 0 == (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride).getD _ 0)
    rw [hgi _ (by omega), hgi _ (by omega)]
    exact byteEqBridge _ _
  -- ── simulation half, via `simSL'_extCBody`. ──
  have hsim := simSL'_extCBody R inStride endCap hctx hendCap hml hadv hextC hRdisj ws
    hboundW hbytesW prog base ss fuel hpc hseg hlr hc hmi
  refine ⟨hsim, ?_⟩
  -- ── `extInv` preserved: only `adv`/`ml`/`extC` change; `ml' = ml + leadingGood`. ──
  -- Register readout of the body's `eval`: `adv := ofNat LG`, `ml += adv`, `extC := …`.
  have hbodyReg : ∀ r, r ≠ "adv" → r ≠ "ml" → r ≠ "extC" →
      ((wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
        .bin .add "ml" "ml" (.reg "adv"), .setp .eq "extC" "adv" (.imm 32) ]).eval fuel ws).regs r
        = ws.regs r := by
    intro r hadvr hmlr hextr
    simp only [wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run, SCmp.run, WArg.eval,
      if_neg hextr, if_neg hmlr, if_neg hadvr]
  have hmlReg : ((wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
        .bin .add "ml" "ml" (.reg "adv"), .setp .eq "extC" "adv" (.imm 32) ]).eval fuel ws).regs "ml"
        = ws.regs "ml" + UInt64.ofNat
          (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32) := by
    simp [wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run, SCmp.run, WArg.eval]
  -- `p0 + (ml + LG) ≤ endCap`, so the invariant survives (`leadingGood_pos_le`).
  have hpres : (ws.regs "p0").toNat + ((ws.regs "ml").toNat
        + AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32) ≤ endCap := by
    have := AlgorithmLib.LZ4WarpSched.leadingGood_pos_le (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
      (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap 32 (ws.regs "ml").toNat hpml
    omega
  -- `ml' = ml + LG` as a Nat (no overflow: bounded by `endCap < 2^40`).
  have hmlN : (ws.regs "ml" + UInt64.ofNat
        (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32)).toNat
      = (ws.regs "ml").toNat + AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32 := by
    rw [UInt64.toNat_add, toNat_ofNat_lt _ (by omega), Nat.mod_eq_of_lt (by omega)]
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hbodyReg "ecR" (by decide) (by decide) (by decide)]; exact hecR
  · rw [hbodyReg "ec1" (by decide) (by decide) (by decide)]; exact hec1
  · rw [hbodyReg "cand0" (by decide) (by decide) (by decide),
      hbodyReg "p0" (by decide) (by decide) (by decide)]; exact hcandlt
  · rw [hbodyReg "p0" (by decide) (by decide) (by decide), hmlReg, hmlN]; exact hpres

/-- The four found-branch setup statements as one `SimSL'` (straight-line leaves). -/
theorem simSL'_foundMovs (R : List String) (endCap : Nat)
    (hecR : "ecR" ∈ R) (hec1 : "ec1" ∈ R) (hml : "ml" ∈ R) (hextC : "extC" ∈ R) :
    SimSL' ib R
      (wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
              .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ])
      (([.mov "ecR" (SArg.imm endCap)] : List SInstr)
        ++ ([.mov "ec1" (.imm (endCap - 1))]
          ++ ([.mov "ml" (.imm 4)] ++ [.setp .ge "extC" "ml" (.imm 0)]))) := by
  apply simSL'_seq
  · exact simSL'_mov R "ecR" (.imm endCap) (fun n h => by cases h) (by decide)
  apply simSL'_seq
  · exact simSL'_mov R "ec1" (.imm (endCap - 1)) (fun n h => by cases h) (by decide)
  apply simSL'_seq
  · exact simSL'_mov R "ml" (.imm 4) (fun n h => by cases h) (by decide)
  · exact simSL'_setp R .ge "extC" "ml" (.imm 0) hml (fun n h => by cases h) (by decide)

/-- `off0 := p0 - cand0 ; litLen := p0 - litAnchor` (the two computes before the
    match-sequence emit) as one `SimSL'`. -/
theorem simSL'_foundSubs (R : List String) (litStart : String)
    (hp0 : "p0" ∈ R) (hcand0 : "cand0" ∈ R) (hls : litStart ∈ R)
    (hoff0 : "off0" ∈ R) (hlitLen : "litLen" ∈ R) :
    SimSL' ib R
      (wseq [ .bin .sub "off0" "p0" (.reg "cand0"),
              .bin .sub "litLen" "p0" (.reg litStart) ])
      (([.bin .sub "off0" "p0" (.reg "cand0")] : List SInstr)
        ++ [.bin .sub "litLen" "p0" (.reg litStart)]) := by
  apply simSL'_seq
  · exact simSL'_bin R .sub "off0" "p0" (.reg "cand0") hp0 (fun n h => by cases h; exact hcand0)
      (by decide)
  · exact simSL'_bin R .sub "litLen" "p0" (.reg litStart) hp0 (fun n h => by cases h; exact hls)
      (by decide)

/-- `litAnchor := p0 + ml ; searchPos := litAnchor` (the two updates after the
    match-sequence emit) as one `SimSL'`. -/
theorem simSL'_foundUpdates (R : List String)
    (hp0 : "p0" ∈ R) (hml : "ml" ∈ R) (hla : "litAnchor" ∈ R) (hsp : "searchPos" ∈ R) :
    SimSL' ib R
      (wseq [ .bin .add "litAnchor" "p0" (.reg "ml"), .mov "searchPos" (.reg "litAnchor") ])
      (([.bin .add "litAnchor" "p0" (.reg "ml")] : List SInstr)
        ++ [.mov "searchPos" (.reg "litAnchor")]) := by
  apply simSL'_seq
  · exact simSL'_bin R .add "litAnchor" "p0" (.reg "ml") hp0 (fun n h => by cases h; exact hml)
      (by decide)
  · exact simSL'_mov R "searchPos" (.reg "litAnchor") (fun n h => by cases h; exact hla) (by decide)

/-- The extend loop's body statement (coop extend step; `ml += adv`; `extC := adv==32`). -/
def extBodyStmt (inStride endCap : Nat) : WStmt :=
  wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
         .bin .add "ml" "ml" (.reg "adv"),
         .setp .eq "extC" "adv" (.imm 32) ]

/-- The extend body preserves `extInv` (only `adv`/`ml`/`extC` change; `ml += leadingGood`
    keeps `p0 + ml ≤ endCap` by `leadingGood_pos_le`).  Extracted from
    `bodySimInv_extCBody`'s preservation half, at the pure eval level. -/
theorem extBody_preserves_extInv (inStride endCap fuel : Nat) (ws : WState)
    (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hInv : extInv inStride endCap ws) :
    extInv inStride endCap ((extBodyStmt inStride endCap).eval fuel ws) := by
  obtain ⟨hecR, hec1, hcandlt, hpml⟩ := hInv
  simp only [extBodyStmt]
  have hbodyReg : ∀ r, r ≠ "adv" → r ≠ "ml" → r ≠ "extC" →
      ((wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
        .bin .add "ml" "ml" (.reg "adv"), .setp .eq "extC" "adv" (.imm 32) ]).eval fuel ws).regs r
        = ws.regs r := by
    intro r hadvr hmlr hextr
    simp only [wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run, SCmp.run, WArg.eval,
      if_neg hextr, if_neg hmlr, if_neg hadvr]
  have hmlReg : ((wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
        .bin .add "ml" "ml" (.reg "adv"), .setp .eq "extC" "adv" (.imm 32) ]).eval fuel ws).regs "ml"
        = ws.regs "ml" + UInt64.ofNat
          (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32) := by
    simp [wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run, SCmp.run, WArg.eval]
  have hpres : (ws.regs "p0").toNat + ((ws.regs "ml").toNat
        + AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32) ≤ endCap := by
    have := AlgorithmLib.LZ4WarpSched.leadingGood_pos_le (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
      (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap 32 (ws.regs "ml").toNat hpml
    omega
  have hmlN : (ws.regs "ml" + UInt64.ofNat
        (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32)).toNat
      = (ws.regs "ml").toNat + AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32 := by
    rw [UInt64.toNat_add, toNat_ofNat_lt _ (by omega), Nat.mod_eq_of_lt (by omega)]
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hbodyReg "ecR" (by decide) (by decide) (by decide)]; exact hecR
  · rw [hbodyReg "ec1" (by decide) (by decide) (by decide)]; exact hec1
  · rw [hbodyReg "cand0" (by decide) (by decide) (by decide),
      hbodyReg "p0" (by decide) (by decide) (by decide)]; exact hcandlt
  · rw [hbodyReg "p0" (by decide) (by decide) (by decide), hmlReg, hmlN]; exact hpres

/-- The extend `uwhile` preserves `extInv` (each iteration does, via
    `extBody_preserves_extInv`).  Gives `p0 + ml ≤ endCap` on the loop result. -/
theorem extInv_uwhile (inStride endCap : Nat) (hendCap : endCap = inStride - 5)
    (hlen : inStride < 2 ^ 40) :
    ∀ (fuel : Nat) (ws : WState), extInv inStride endCap ws →
      extInv inStride endCap
        ((WStmt.uwhile "extC" (extBodyStmt inStride endCap)).eval fuel ws) := by
  intro fuel
  induction fuel with
  | zero => intro ws hInv; simpa [WStmt.eval] using hInv
  | succ fuel ih =>
      intro ws hInv
      simp only [WStmt.eval]
      by_cases hb : (ws.regs "extC" == 1) = true
      · rw [if_pos hb]
        exact ih _ (extBody_preserves_extInv inStride endCap fuel ws hendCap hlen hInv)
      · rw [if_neg hb]; exact hInv

/-- The extend body only grows `ml` (`ml += leadingGood ≥ 0`), so `4 ≤ ml` is
    preserved.  Needs `extInv` for the no-overflow bound on the new `ml`. -/
theorem extBody_ml_ge4 (inStride endCap fuel : Nat) (ws : WState)
    (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hInv : extInv inStride endCap ws) (h4 : 4 ≤ (ws.regs "ml").toNat) :
    4 ≤ (((extBodyStmt inStride endCap).eval fuel ws).regs "ml").toNat := by
  obtain ⟨_, _, _, hpml⟩ := hInv
  simp only [extBodyStmt]
  have hlg := AlgorithmLib.LZ4WarpSched.leadingGood_pos_le (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
    (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap 32 (ws.regs "ml").toNat hpml
  have hmlReg : ((wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
        .bin .add "ml" "ml" (.reg "adv"), .setp .eq "extC" "adv" (.imm 32) ]).eval fuel ws).regs "ml"
        = ws.regs "ml" + UInt64.ofNat
          (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32) := by
    simp [wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run, SCmp.run, WArg.eval]
  have hmlN : (ws.regs "ml" + UInt64.ofNat
        (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32)).toNat
      = (ws.regs "ml").toNat + AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32 := by
    rw [UInt64.toNat_add, toNat_ofNat_lt _ (by omega), Nat.mod_eq_of_lt (by omega)]
  rw [hmlReg, hmlN]; omega

/-- The extend `uwhile` preserves `4 ≤ ml` (each iteration grows `ml`). -/
theorem uwhile_ml_ge4 (inStride endCap : Nat) (hendCap : endCap = inStride - 5)
    (hlen : inStride < 2 ^ 40) :
    ∀ (fuel : Nat) (ws : WState), extInv inStride endCap ws → 4 ≤ (ws.regs "ml").toNat →
      4 ≤ (((WStmt.uwhile "extC" (extBodyStmt inStride endCap)).eval fuel ws).regs "ml").toNat := by
  intro fuel
  induction fuel with
  | zero => intro ws _ h4; simpa [WStmt.eval] using h4
  | succ fuel ih =>
      intro ws hInv h4
      simp only [WStmt.eval]
      by_cases hb : (ws.regs "extC" == 1) = true
      · rw [if_pos hb]
        exact ih _ (extBody_preserves_extInv inStride endCap fuel ws hendCap hlen hInv)
          (extBody_ml_ge4 inStride endCap fuel ws hendCap hlen hInv h4)
      · rw [if_neg hb]; exact h4

/-- The extend loop terminates: while it continues (`adv == 32`), `ml` grows by
    32 and `p0 + ml` stays `≤ endCap` (`extInv`), so the gap `endCap - (p0+ml)`
    strictly decreases by 32 each iteration.  Any `fuel` with `endCap ≤ 32*fuel +
    (p0+ml)` suffices. -/
theorem extLoop_halts (inStride endCap : Nat) (hendCap : endCap = inStride - 5)
    (hlen : inStride < 2 ^ 40) :
    ∀ (fuel : Nat) (ws : WState), extInv inStride endCap ws →
      endCap + 32 < 32 * fuel + ((ws.regs "p0").toNat + (ws.regs "ml").toNat) →
      WhileHalts "extC" (extBodyStmt inStride endCap) fuel ws := by
  intro fuel
  induction fuel with
  | zero =>
      intro ws hInv hmeas
      -- fuel 0 ⟹ measure `endCap+32 < p0+ml`, contradicting `extInv`'s `p0+ml ≤ endCap`.
      obtain ⟨_, _, _, hpml⟩ := hInv
      omega
  | succ fuel ih =>
      intro ws hInv hmeas
      simp only [WhileHalts]
      by_cases hb : (ws.regs "extC" == 1) = true
      · rw [if_pos hb]
        obtain ⟨hecR, hec1, hcandlt, hpml⟩ := hInv
        have hLGle := AlgorithmLib.LZ4WarpSched.leadingGood_pos_le (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap 32 (ws.regs "ml").toNat hpml
        -- eval readout: p0/cand0/ecR/ec1 unchanged, ml += LG.
        have hbReg : ∀ r, r ≠ "adv" → r ≠ "ml" → r ≠ "extC" →
            ((extBodyStmt inStride endCap).eval fuel ws).regs r = ws.regs r := by
          intro r ha hm he
          simp only [extBodyStmt, wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run,
            SCmp.run, WArg.eval, if_neg he, if_neg hm, if_neg ha]
        have hmlReg : ((extBodyStmt inStride endCap).eval fuel ws).regs "ml"
            = ws.regs "ml" + UInt64.ofNat
              (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32) := by
          simp [extBodyStmt, wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run, SCmp.run,
            WArg.eval]
        have hmlN : (((extBodyStmt inStride endCap).eval fuel ws).regs "ml").toNat
            = (ws.regs "ml").toNat + AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32 := by
          rw [hmlReg, UInt64.toNat_add, toNat_ofNat_lt _ (by omega), Nat.mod_eq_of_lt (by omega)]
        have hInv' : extInv inStride endCap ((extBodyStmt inStride endCap).eval fuel ws) := by
          refine ⟨?_, ?_, ?_, ?_⟩
          · rw [hbReg "ecR" (by decide) (by decide) (by decide)]; exact hecR
          · rw [hbReg "ec1" (by decide) (by decide) (by decide)]; exact hec1
          · rw [hbReg "cand0" (by decide) (by decide) (by decide),
              hbReg "p0" (by decide) (by decide) (by decide)]; exact hcandlt
          · rw [hbReg "p0" (by decide) (by decide) (by decide), hmlN]; omega
        by_cases hLG32 : AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32 = 32
        · -- continuing step: measure drops by 32.
          apply ih ((extBodyStmt inStride endCap).eval fuel ws) hInv'
          rw [hbReg "p0" (by decide) (by decide) (by decide), hmlN, hLG32]
          omega
        · -- `adv ≠ 32` ⟹ next guard `extC = 0` ⟹ halts immediately.
          have hne : (UInt64.ofNat
              (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32)
              == (32 : UInt64)) = false := by
            rw [beq_eq_false_iff_ne]
            intro hEq
            apply hLG32
            have h2 := congrArg UInt64.toNat hEq
            rw [toNat_ofNat_lt _ (by omega), show ((32 : UInt64).toNat) = 32 from by decide] at h2
            exact h2
          have hLGne : ¬ UInt64.ofNat
              (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32)
              = (32 : UInt64) := beq_eq_false_iff_ne.mp hne
          have hextC0 : ((extBodyStmt inStride endCap).eval fuel ws).regs "extC" = 0 := by
            simp [extBodyStmt, wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run,
              SCmp.run, WArg.eval, hLGne]
          -- The strict measure `endCap < 32*(fuel+1)+(p0+ml)` with `p0+ml ≤ endCap`
          -- forces `fuel ≥ 1`, so one more `WhileHalts` unfold hits `else True`.
          obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := by
            cases fuel with
            | zero => exact absurd hmeas (by omega)
            | succ f => exact ⟨f, rfl⟩
          simp only [WhileHalts, hextC0]
          rw [if_neg (by decide)]
          trivial
      · rw [if_neg hb]; trivial

/-- The whole extC `.uwhile` loop simulates: from an `extInv` state with adequate
    `fuel` (the loop halts, `extLoop_halts`), the emitted loop reaches the exit
    coupled to `eval`.  Composes `bodySimInv_extCBody` (body sim) with the
    invariant-threading loop engine `simSL'_uwhileBodyInv`, discharging `WhileHalts`
    from `extLoop_halts`. -/
theorem extLoop_sim (R : List String) (inStride endCap : Nat)
    (hctx : ExtCtx R) (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hib40 : ib < 2 ^ 40)
    (hml : "ml" ∈ R) (hadv : "adv" ∈ R) (hextC : "extC" ∈ R)
    (hRdisj : ∀ r ∈ R, r ∉ ["idx", "pe", "pIn", "peC", "dfe", "caC", "peD", "aP", "caD", "aC",
               "bP", "bC", "pEqB", "pOk", "balOk", "mis", "revM"])
    (lHead lEnd : String) (prog : Array SInstr) (base : Nat)
    (hseg : SegAt prog base (uwhileEmit "extC" lHead lEnd
      (coopExtendEmit "adv" ++ (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
        ++ [.setp .eq "extC" "adv" (.imm 32)]))))
    (hlr : LabelsResolve prog base (uwhileEmit "extC" lHead lEnd
      (coopExtendEmit "adv" ++ (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
        ++ [.setp .eq "extC" "adv" (.imm 32)])))) :
    ∀ (fuel : Nat) (ss : SState) (ws : WState),
      ss.pc = base → Couple R ss ws → MachInv ib ss → extInv inStride endCap ws →
      endCap + 32 < 32 * fuel + ((ws.regs "p0").toNat + (ws.regs "ml").toNat) →
      ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
        ss'.pc = base + (uwhileEmit "extC" lHead lEnd
          (coopExtendEmit "adv" ++ (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
            ++ [.setp .eq "extC" "adv" (.imm 32)]))).length ∧
        Couple R ss' (WStmt.eval fuel (.uwhile "extC" (extBodyStmt inStride endCap)) ws)
          ∧ MachInv ib ss' := by
  intro fuel ss ws hpc hc hmi hInv hfuel
  exact simSL'_uwhileBodyInv R "extC" lHead lEnd (extInv inStride endCap)
    (extBodyStmt inStride endCap)
    (coopExtendEmit "adv" ++ (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
      ++ [.setp .eq "extC" "adv" (.imm 32)]))
    prog base hextC
    (by unfold extBodyStmt
        exact bodySimInv_extCBody R inStride endCap hctx hendCap hlen hib40 hml hadv hextC hRdisj)
    hseg hlr fuel ss ws hpc hc hmi hInv
    (extLoop_halts inStride endCap hendCap hlen fuel ws hInv hfuel)

/-- The LSIC length-extension loop's body (`n`-register form): store a `255` byte,
    subtract 255 from `n`, recompute the guard `lsicC := n ≥ 255`. -/
def lsicBodyStmt (n : String) : WStmt :=
  wseq [ .mov "c255" (.imm 255), wStoreByte "c255", .bin .sub n n (.imm 255),
         .setp .ge "lsicC" n (.imm 255) ]

/-- The LSIC loop terminates: while it continues (`lsicC == 1`, and the guard
    invariant `lsicC=1 → n ≥ 255` holds), `n` drops by 255 each iteration, so any
    `fuel` with `n.toNat < 255 * fuel` suffices.  The guard invariant is preserved
    by the body's trailing `setp`. -/
theorem lsicLoop_halts (n : String) (hnc : n ≠ "c255") (hnop : n ≠ "op")
    (hnsb : n ≠ "sbAddr") (hnl : n ≠ "lsicC") :
    ∀ (fuel : Nat) (ws : WState),
      (ws.regs "lsicC" = 1 → 255 ≤ (ws.regs n).toNat) →
      (ws.regs n).toNat < 255 * fuel →
      WhileHalts "lsicC" (lsicBodyStmt n) fuel ws := by
  intro fuel
  induction fuel with
  | zero => intro ws _ hmeas; omega
  | succ fuel ih =>
      intro ws hguard hmeas
      simp only [WhileHalts]
      by_cases hb : (ws.regs "lsicC" == 1) = true
      · rw [if_pos hb]
        have hlsic1 : ws.regs "lsicC" = 1 := by
          have := hb; rw [beq_iff_eq] at this; exact this
        have hbig : 255 ≤ (ws.regs n).toNat := hguard hlsic1
        -- eval readout: `n' = n - 255` (UInt64).
        have hnReg : ((lsicBodyStmt n).eval fuel ws).regs n
            = ws.regs n - UInt64.ofNat 255 := by
          simp [lsicBodyStmt, wseq, WStmt.eval, wStoreByte, WState.setReg, WState.stgByte,
            WOp.run, SCmp.run, WArg.eval, hnl, hnc, hnop, hnsb, Ne.symm hnl, Ne.symm hnsb]
        have hnN : (((lsicBodyStmt n).eval fuel ws).regs n).toNat
            = (ws.regs n).toNat - 255 := by
          rw [hnReg, UInt64.toNat_sub, show ((UInt64.ofNat 255).toNat) = 255 from by decide]
          rw [show 2 ^ 64 - 255 + (ws.regs n).toNat = 2 ^ 64 + ((ws.regs n).toNat - 255) from by
            omega, Nat.add_mod_left, Nat.mod_eq_of_lt (by have := (ws.regs n).toNat_lt; omega)]
        -- new lsicC = (n' ≥ 255) as 1/0.
        have hlsicVal : ((lsicBodyStmt n).eval fuel ws).regs "lsicC"
            = (if (UInt64.ofNat 255) ≤ (ws.regs n - UInt64.ofNat 255) then 1 else 0) := by
          simp [lsicBodyStmt, wseq, WStmt.eval, wStoreByte, WState.setReg, WState.stgByte,
            WOp.run, SCmp.run, WArg.eval, hnl, hnc, hnop, hnsb, Ne.symm hnl, Ne.symm hnsb]
        -- guard invariant preserved: `lsicC=1 → n' ≥ 255`.
        have hguard' : ((lsicBodyStmt n).eval fuel ws).regs "lsicC" = 1 →
            255 ≤ (((lsicBodyStmt n).eval fuel ws).regs n).toNat := by
          intro hg1
          rw [hnN]
          rw [hlsicVal] at hg1
          by_cases hge : (UInt64.ofNat 255) ≤ (ws.regs n - UInt64.ofNat 255)
          · rw [UInt64.le_iff_toNat_le, show ((UInt64.ofNat 255).toNat) = 255 from by decide,
              UInt64.toNat_sub, show ((UInt64.ofNat 255).toNat) = 255 from by decide,
              show 2 ^ 64 - 255 + (ws.regs n).toNat = 2 ^ 64 + ((ws.regs n).toNat - 255) from by
                omega, Nat.add_mod_left, Nat.mod_eq_of_lt (by have := (ws.regs n).toNat_lt; omega)]
              at hge
            omega
          · rw [if_neg hge] at hg1; exact absurd hg1 (by decide)
        apply ih ((lsicBodyStmt n).eval fuel ws) hguard'
        rw [hnN]; omega
      · rw [if_neg hb]; trivial

/-- The LSIC loop body writes only `c255`/`sbAddr`/`op`/`n`/`lsicC` (+`gmem`); any
    other register is preserved. -/
theorem lsicBody_frame (n r : String)
    (h1 : r ≠ "c255") (h2 : r ≠ "sbAddr") (h3 : r ≠ "op") (h4 : r ≠ n) (h5 : r ≠ "lsicC")
    (fuel : Nat) (ws : WState) : ((lsicBodyStmt n).eval fuel ws).regs r = ws.regs r := by
  simp [lsicBodyStmt, wseq, WStmt.eval, wStoreByte, WState.setReg, WState.stgByte, WOp.run,
    SCmp.run, WArg.eval, h1, h2, h3, h4, h5]

/-- The LSIC `uwhile` preserves every register outside `{c255,sbAddr,op,n,lsicC}`. -/
theorem lsicUwhile_frame (n r : String)
    (h1 : r ≠ "c255") (h2 : r ≠ "sbAddr") (h3 : r ≠ "op") (h4 : r ≠ n) (h5 : r ≠ "lsicC") :
    ∀ (fuel : Nat) (ws : WState),
      ((WStmt.uwhile "lsicC" (lsicBodyStmt n)).eval fuel ws).regs r = ws.regs r := by
  intro fuel
  induction fuel with
  | zero => intro ws; simp [WStmt.eval]
  | succ fuel ih =>
    intro ws; simp only [WStmt.eval]
    by_cases hb : (ws.regs "lsicC" == 1) = true
    · rw [if_pos hb, ih, lsicBody_frame n r h1 h2 h3 h4 h5]
    · rw [if_neg hb]

/-- `wEmitLSIC n` preserves every register outside `{lsicC,c255,sbAddr,op,n}`. -/
theorem wEmitLSIC_frame (n r : String)
    (h1 : r ≠ "lsicC") (h2 : r ≠ "c255") (h3 : r ≠ "sbAddr") (h4 : r ≠ "op") (h5 : r ≠ n)
    (fuel : Nat) (ws : WState) : ((wEmitLSIC n).eval fuel ws).regs r = ws.regs r := by
  have hw : wEmitLSIC n = (WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255)).seq
      ((WStmt.uwhile "lsicC" (lsicBodyStmt n)).seq (wStoreByte n)) := by
    rw [wEmitLSIC]; rfl
  have e1 : ∀ st : WState, ((wStoreByte n).eval fuel st).regs r = st.regs r := fun st => by
    simp [wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte, WOp.run, WArg.eval, h3, h4]
  have e2 : ∀ st : WState,
      ((WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255)).eval fuel st).regs r = st.regs r := fun st => by
    simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run, h1]
  rw [hw]
  simp only [WStmt.eval.eq_2]
  rw [e1, lsicUwhile_frame n r h2 h3 h4 h5 h1, e2]

/-- The emitted instructions of `wEmitLSIC eReg` (guard setp, the LSIC loop, the
    trailing single-byte store).  Matches `simSL'_wEmitLSIC`'s segment. -/
def lsicEmit (eReg lH lX : String) (lsicBody : List SInstr) : List SInstr :=
  ([.setp .ge "lsicC" eReg (.imm 255)] : List SInstr)
    ++ (uwhileEmit "lsicC" lH lX lsicBody
      ++ (([.bin .add "sbAddr" "outBase" (.reg "op")] : List SInstr)
        ++ (([.stg "sbAddr" eReg] : List SInstr)
          ++ ([.bin .add "op" "op" (.imm 1)] : List SInstr))))

/-- The then-branch emit of the LSIC-conditional block: `bin sub eReg srcReg 15`
    followed by `wEmitLSIC eReg`. -/
def lsicThen (srcReg eReg lH lX : String) (lsicBody : List SInstr) : List SInstr :=
  ([.bin .sub eReg srcReg (.imm 15)] : List SInstr) ++ lsicEmit eReg lH lX lsicBody

theorem lsicThen_length (srcReg eReg lH lX : String) (lsicBody : List SInstr) :
    (lsicThen srcReg eReg lH lX lsicBody).length = lsicBody.length + 9 := by
  simp [lsicThen, lsicEmit, uwhileEmit_length, List.length_append, Nat.add_comm, Nat.add_left_comm]

/-- The LSIC-conditional block `.uif pcond (wseq [bin sub eReg srcReg 15; wEmitLSIC eReg]) .skip`
    simulates.  When the guard is live the extra-length `eReg = srcReg - 15` feeds a
    terminating LSIC loop (fuel bound `srcReg.toNat < 255*fuel + 15`); otherwise it's a
    no-op.  Bespoke (carries `ws` + the LSIC fuel bound) since `wEmitLSIC` isn't a plain
    `SimSL'`.  Used for both litLen/mlm big-length extensions. -/
theorem simSL'_lsicUif (R : List String) (pcond srcReg eReg lElse lEnd lH lX : String)
    (hpcR : pcond ∈ R) (heReg : eReg ∈ R) (hsrc : srcReg ∈ R)
    (heRegW : eReg ≠ "lane" ∧ eReg ≠ "inBase" ∧ eReg ≠ "tbl")
    (heC : eReg ≠ "c255") (heO : eReg ≠ "op") (heS : eReg ≠ "sbAddr") (heL : eReg ≠ "lsicC")
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R)
    (hc255 : "c255" ∈ R) (hlsicC : "lsicC" ∈ R)
    (lsicBody : List SInstr)
    (hbodyDef : lsicBody =
      ([.mov "c255" (SArg.imm 255)]
        ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
            ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
          ++ ([.bin .sub eReg eReg (.imm 255)] ++ [.setp .ge "lsicC" eReg (.imm 255)])))) :
    ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (ws : WState) (fuel : Nat),
      ss.pc = base →
      SegAt prog base (uifEmit pcond lElse lEnd (lsicThen srcReg eReg lH lX lsicBody) []) →
      LabelsResolve prog base (uifEmit pcond lElse lEnd (lsicThen srcReg eReg lH lX lsicBody) []) →
      Couple R ss ws → MachInv ib ss →
      (ws.regs pcond = 1 → 15 ≤ (ws.regs srcReg).toNat) →
      (ws.regs pcond = 1 → (ws.regs srcReg).toNat < 255 * fuel + 15) →
      ∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
        ss'.pc = base +
          (uifEmit pcond lElse lEnd (lsicThen srcReg eReg lH lX lsicBody) []).length ∧
        Couple R ss'
          ((WStmt.uif pcond
            (wseq [ .bin .sub eReg srcReg (.imm 15), wEmitLSIC eReg ]) .skip).eval fuel ws)
          ∧ MachInv ib ss' := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi hsrcG hfuel
  have hcv : ss.regs pcond 0 = ws.regs pcond := hc.reg hpcR 0
  obtain ⟨hbr, hseg1⟩ := hseg.cons
  have hbr' : prog[ss.pc]? = some (.braifnot pcond lElse) := by rw [hpc]; exact hbr
  -- the then-branch `SimSL'`-with-side-condition (bin sub ; wEmitLSIC), producing
  -- the coupled eval of `wseq [bin sub, wEmitLSIC]`.
  have thenSim : ∀ ss0, ss0.pc = base + 1 → Couple R ss0 ws → MachInv ib ss0 →
      ws.regs pcond = 1 →
      ∃ (m : Nat) (ss' : SState), SReaches prog m ss0 ss' ∧
        ss'.pc = base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length ∧
        Couple R ss'
          ((wseq [ .bin .sub eReg srcReg (.imm 15), wEmitLSIC eReg ]).eval fuel ws) ∧ MachInv ib ss' := by
    intro ss0 hpc0 hc0 hmi0 hp1
    have hsegT : SegAt prog (base + 1) (lsicThen srcReg eReg lH lX lsicBody) := by
      have := hseg1.append_left (eb := [.bra lEnd, .lbl lElse] ++ ([] ++ [.lbl lEnd]))
      simpa [uifEmit, lsicThen] using this
    have hlrT : LabelsResolve prog (base + 1) (lsicThen srcReg eReg lH lX lsicBody) := by
      have := hlr.cons.append_left (eb := [.bra lEnd, .lbl lElse] ++ ([] ++ [.lbl lEnd]))
      simpa [uifEmit, lsicThen] using this
    -- split `lsicThen = [bin sub] ++ wEmitLSIC-emit`.
    rw [lsicThen] at hsegT hlrT
    have hsegSub : SegAt prog (base + 1) [.bin .sub eReg srcReg (.imm 15)] := hsegT.append_left
    obtain ⟨na, ssA, hrA, hpcA, hcA, hmiA⟩ :=
      (simSL'_bin R .sub eReg srcReg (.imm 15) hsrc (fun m h => by cases h) heRegW)
        prog (base + 1) ss0 ws fuel hpc0 hsegSub hlrT.append_left hc0 hmi0
    have hwsA : WStmt.eval fuel (.bin .sub eReg srcReg (.imm 15)) ws
        = ws.setReg eReg (ws.regs srcReg - UInt64.ofNat 15) := by
      simp [WStmt.eval, WOp.run, WArg.eval]
    have hsegLS : SegAt prog (base + 1 + 1)
        (lsicEmit eReg lH lX lsicBody) := by
      have := hsegT.append_right (ea := [.bin .sub eReg srcReg (.imm 15)])
      simpa [lsicEmit] using this
    have hlrLS : LabelsResolve prog (base + 1 + 1) (lsicEmit eReg lH lX lsicBody) := by
      have := hlrT.append_right (ea := [.bin .sub eReg srcReg (.imm 15)])
      simpa [lsicEmit] using this
    -- fuel adequacy + guard-consistency at the post-`setp` LSIC-entry state.
    have hsrcN := (ws.regs srcReg).toNat_lt
    have hf := hfuel hp1
    have hg15 := hsrcG hp1
    have hsubN : ((ws.regs srcReg - UInt64.ofNat 15)).toNat = (ws.regs srcReg).toNat - 15 := by
      rw [UInt64.toNat_sub, show ((UInt64.ofNat 15).toNat) = 15 from by decide]
      rw [show 2 ^ 64 - 15 + (ws.regs srcReg).toNat = 2 ^ 64 + ((ws.regs srcReg).toNat - 15)
          from by omega, Nat.add_mod_left, Nat.mod_eq_of_lt (by omega)]
    have hentryReg : ((WStmt.setp SCmp.ge "lsicC" eReg (WArg.imm 255)).eval fuel
        (ws.setReg eReg (ws.regs srcReg - UInt64.ofNat 15))).regs eReg
        = ws.regs srcReg - UInt64.ofNat 15 := by
      simp [WStmt.eval, WState.setReg, SCmp.run, WArg.eval, heL, Ne.symm heL]
    have hentryLsic : ((WStmt.setp SCmp.ge "lsicC" eReg (WArg.imm 255)).eval fuel
        (ws.setReg eReg (ws.regs srcReg - UInt64.ofNat 15))).regs "lsicC" = 1 →
        255 ≤ (ws.regs srcReg - UInt64.ofNat 15).toNat := by
      intro hg
      have hval : ((WStmt.setp SCmp.ge "lsicC" eReg (WArg.imm 255)).eval fuel
          (ws.setReg eReg (ws.regs srcReg - UInt64.ofNat 15))).regs "lsicC"
          = (if UInt64.ofNat 255 ≤ (ws.regs srcReg - UInt64.ofNat 15) then 1 else 0) := by
        simp [WStmt.eval, WState.setReg, SCmp.run, WArg.eval, heL]
      rw [hval] at hg
      by_cases hge : UInt64.ofNat 255 ≤ ws.regs srcReg - UInt64.ofNat 15
      · rw [UInt64.le_iff_toNat_le] at hge
        rw [show ((UInt64.ofNat 255).toNat) = 255 from by decide] at hge; exact hge
      · rw [if_neg hge] at hg; exact absurd hg (by decide)
    obtain ⟨nb, ssB, hrB, hpcB, hcB, hmiB⟩ :=
      simSL'_wEmitLSIC R eReg lH lX heReg heRegW hout hop hsb hc255 hlsicC lsicBody hbodyDef
        prog (base + 1 + 1) ssA (ws.setReg eReg (ws.regs srcReg - UInt64.ofNat 15)) fuel
        (by rw [hpcA]; rfl) hsegLS hlrLS (by rw [hwsA] at hcA; exact hcA) hmiA
        (lsicLoop_halts eReg heC heO heS heL fuel
          ((WStmt.setp SCmp.ge "lsicC" eReg (WArg.imm 255)).eval fuel
            (ws.setReg eReg (ws.regs srcReg - UInt64.ofNat 15)))
          (fun hg => by rw [hentryReg]; exact hentryLsic hg)
          (by rw [hentryReg, hsubN]; omega))
    refine ⟨na + nb, ssB, sreaches_trans prog na nb _ _ _ hrA hrB, ?_, ?_, hmiB⟩
    · rw [hpcB, lsicThen]
      simp only [lsicEmit, List.length_append, List.length_cons, List.length_nil]; omega
    · have heval : (wseq [ .bin .sub eReg srcReg (.imm 15), wEmitLSIC eReg ]).eval fuel ws
          = (wEmitLSIC eReg).eval fuel (ws.setReg eReg (ws.regs srcReg - UInt64.ofNat 15)) := by
        have h1 : (wseq [ .bin .sub eReg srcReg (.imm 15), wEmitLSIC eReg ]).eval fuel ws
            = (wEmitLSIC eReg).eval fuel
              ((WStmt.bin .sub eReg srcReg (.imm 15)).eval fuel ws) := by
          simp only [wseq, WStmt.eval]
        rw [h1, hwsA]
      rw [heval]; exact hcB
  -- now the `uif` case split, mirroring `simSL'_uif`.
  have hLen := lsicThen_length srcReg eReg lH lX lsicBody
  by_cases hb : (ws.regs pcond == 1) = true
  · have hp1 : ws.regs pcond = 1 := by rw [beq_iff_eq] at hb; exact hb
    have s0 : sstep prog ss = ss.setPc (base + 1) := by
      rw [braifnot_step prog ss pcond lElse hbr', hcv, hpc, if_pos hb]
    obtain ⟨nt, ss1, hrT, hpcT, hcT, hmiT⟩ :=
      thenSim (ss.setPc (base + 1)) rfl (couple_setPc hc _) (machInv_setPc ss _ hmi) hp1
    obtain ⟨hbra, hseg3⟩ := hseg1.append_right.cons
    obtain ⟨hlblE, hseg4⟩ := hseg3.cons
    obtain ⟨hlblN, _⟩ := hseg4.append_right.cons
    have hlr3 := hlr.cons.append_right.cons
    have hLend : sfindLabel prog lEnd
        = base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1 + ([] : List SInstr).length :=
      hlr3.cons.append_right 0 lEnd (by simp)
    have hbra' : prog[ss1.pc]? = some (.bra lEnd) := by rw [hpcT]; exact hbra
    have s1 : sstep prog ss1
        = ss1.setPc (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1 + 0) := by
      rw [bra_step prog ss1 lEnd hbra', hLend]; simp
    have hlblN' : prog[(ss1.setPc
        (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1 + 0)).pc]?
        = some (.lbl lEnd) := by simpa using hlblN
    have s2 : sstep prog (ss1.setPc
          (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1 + 0))
        = ss1.setPc (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1 + 0 + 1) := by
      rw [lbl_step prog _ lEnd hlblN']; simp [SState.setPc]
    refine ⟨1 + nt + 1 + 1, _,
      sreaches_trans prog (1 + nt + 1) 1 _ _ _
        (sreaches_trans prog (1 + nt) 1 _ _ _
          (sreaches_trans prog 1 nt _ _ _ (sreaches_one_eq s0) hrT)
          (sreaches_one_eq s1))
        (sreaches_one_eq s2), ?_, ?_, machInv_setPc _ _ (machInv_setPc _ _ hmiT)⟩
    · show (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1 + 0 + 1)
        = base + (uifEmit pcond lElse lEnd (lsicThen srcReg eReg lH lX lsicBody) []).length
      rw [uifEmit_length]; simp; omega
    · have heval : WStmt.eval fuel
          (.uif pcond (wseq [ .bin .sub eReg srcReg (.imm 15), wEmitLSIC eReg ]) .skip) ws
          = (wseq [ .bin .sub eReg srcReg (.imm 15), wEmitLSIC eReg ]).eval fuel ws := by
        simp [WStmt.eval, hb]
      rw [heval]; exact couple_setPc (couple_setPc hcT _) _
  · -- guard 0: skip to lEnd.
    obtain ⟨hbra, hseg3⟩ := hseg1.append_right.cons
    obtain ⟨hlblE, hseg4⟩ := hseg3.cons
    obtain ⟨hlblN, _⟩ := hseg4.append_right.cons
    have hlr3 := hlr.cons.append_right.cons
    have hLelse : sfindLabel prog lElse
        = base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 := hlr3 0 lElse (by simp)
    have s0 : sstep prog ss
        = ss.setPc (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1) := by
      rw [braifnot_step prog ss pcond lElse hbr', hcv, if_neg hb, hLelse]
    have hlblE' : prog[(ss.setPc
        (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1)).pc]?
        = some (.lbl lElse) := hlblE
    have s1 : sstep prog (ss.setPc (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1))
        = ss.setPc (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1) := by
      rw [lbl_step prog _ lElse hlblE']; simp [SState.setPc]
    have hlblN' : prog[(ss.setPc
        (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1)).pc]?
        = some (.lbl lEnd) := by simpa using hlblN
    have s2 : sstep prog (ss.setPc
          (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1))
        = ss.setPc (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1 + 1) := by
      rw [lbl_step prog _ lEnd hlblN']; simp [SState.setPc]
    refine ⟨1 + 1 + 1, _,
      sreaches_trans prog (1 + 1) 1 _ _ _
        (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq s0) (sreaches_one_eq s1))
        (sreaches_one_eq s2), ?_, ?_, machInv_setPc _ _ (machInv_setPc _ _ (machInv_setPc _ _ hmi))⟩
    · show (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1 + 1)
        = base + (uifEmit pcond lElse lEnd (lsicThen srcReg eReg lH lX lsicBody) []).length
      rw [uifEmit_length]; simp; omega
    · have heval : WStmt.eval fuel
          (.uif pcond (wseq [ .bin .sub eReg srcReg (.imm 15), wEmitLSIC eReg ]) .skip) ws = ws := by
        simp [WStmt.eval, hb]
      rw [heval]; exact couple_setPc (couple_setPc (couple_setPc hc _) _) _

/-- Eval of `wEmitFinalSeq` decomposed as the right-nested `.seq` chain, so a
    simulation can peel it one statement at a time (keeps `wEmitToken`/`wEmitLSIC`
    internals folded — no kernel blow-up). -/
theorem wEmitFinalSeq_eval_seq (litStart litLen : String) (fuel : Nat) (ws : WState) :
    (wEmitFinalSeq litStart litLen).eval fuel ws
      = (WStmt.eval fuel (.bin .add "op" "op" (.reg litLen))
        (WStmt.eval fuel (.coopCopy "cpDstF" "cpSrcF" litLen)
        (WStmt.eval fuel (.bin .add "cpSrcF" "inBase" (.reg litStart))
        (WStmt.eval fuel (.bin .add "cpDstF" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBigF"
            (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
        (WStmt.eval fuel (.setp .ge "pLitBigF" litLen (.imm 15))
        (WStmt.eval fuel (wEmitToken litLen "zero")
        (WStmt.eval fuel (.mov "zero" (.imm 0)) ws)))))))) := by
  simp only [wEmitFinalSeq, wseq, WStmt.eval]

/-- The prefix of `wEmitFinalSeq` up to (and including) the `pLitBigF` guard, as a
    single `SimSL'` (all straight-line leaves): `mov zero; wEmitToken; setp`. -/
theorem simSL'_finalPrefix (R : List String) (litStart litLen : String)
    (hll : litLen ∈ R) (hzero : "zero" ∈ R) (hout : "outBase" ∈ R) (hop : "op" ∈ R)
    (hsb : "sbAddr" ∈ R) (htokHi : "tokHi" ∈ R) (htok : "tok" ∈ R) (hpb : "pLitBigF" ∈ R) :
    SimSL' ib R
      (wseq [ .mov "zero" (.imm 0), wEmitToken litLen "zero",
              .setp .ge "pLitBigF" litLen (.imm 15) ])
      (([.mov "zero" (SArg.imm 0)] : List SInstr)
        ++ (([.bin .min "tokHi" litLen (.imm 15)]
          ++ ([.bin .shl "tok" "tokHi" (.imm 4)]
            ++ ([.bin .bor "tok" "tok" (.reg "zero")]
              ++ ([.bin .add "sbAddr" "outBase" (.reg "op")]
                ++ ([.stg "sbAddr" "tok"] ++ [.bin .add "op" "op" (.imm 1)])))))
          ++ [.setp .ge "pLitBigF" litLen (.imm 15)])) := by
  apply simSL'_seq
  · exact simSL'_mov R "zero" (.imm 0) (fun n h => by cases h) (by decide)
  apply simSL'_seq
  · exact simSL'_wEmitToken R litLen "zero" hll hzero hout hop hsb htokHi htok
  · exact simSL'_setp R .ge "pLitBigF" litLen (.imm 15) hll (fun n h => by cases h) (by decide)

/-- The two address computes before the literal copy: `cpDstF := outBase+op`,
    `cpSrcF := inBase+litStart`.  A single `SimSL'`. -/
theorem simSL'_finalMid (R : List String) (litStart : String)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hib : "inBase" ∈ R) (hls : litStart ∈ R)
    (hcd : "cpDstF" ∈ R) (hcs : "cpSrcF" ∈ R) :
    SimSL' ib R
      (wseq [ .bin .add "cpDstF" "outBase" (.reg "op"),
              .bin .add "cpSrcF" "inBase" (.reg litStart) ])
      (([.bin .add "cpDstF" "outBase" (.reg "op")] : List SInstr)
        ++ [.bin .add "cpSrcF" "inBase" (.reg litStart)]) := by
  apply simSL'_seq
  · exact simSL'_bin R .add "cpDstF" "outBase" (.reg "op") hout (fun n h => by cases h; exact hop)
      (by decide)
  · exact simSL'_bin R .add "cpSrcF" "inBase" (.reg litStart) hib
      (fun n h => by cases h; exact hls) (by decide)

/-- Prefix emit segment (matches `simSL'_finalPrefix`): `mov zero; wEmitToken; setp`. -/
def finalPreEmit (litLen : String) : List SInstr :=
  ([.mov "zero" (SArg.imm 0)] : List SInstr)
    ++ (([.bin .min "tokHi" litLen (.imm 15)]
      ++ ([.bin .shl "tok" "tokHi" (.imm 4)]
        ++ ([.bin .bor "tok" "tok" (.reg "zero")]
          ++ ([.bin .add "sbAddr" "outBase" (.reg "op")]
            ++ ([.stg "sbAddr" "tok"] ++ [.bin .add "op" "op" (.imm 1)])))))
      ++ [.setp .ge "pLitBigF" litLen (.imm 15)])

/-- Copy-address computes (matches `simSL'_finalMid`). -/
def finalMidEmit (litStart : String) : List SInstr :=
  ([.bin .add "cpDstF" "outBase" (.reg "op")] : List SInstr)
    ++ [.bin .add "cpSrcF" "inBase" (.reg litStart)]

/-- The emitted `.coopCopy cpDstF cpSrcF litLen` loop (matches `simSL_coopCopy`). -/
def finalCoopEmit (litLen cpH cpX : String) : List SInstr :=
  ([.mov "cpI" (.imm 0), .setp .lt "cpCont" "cpI" (.reg litLen)] : List SInstr)
    ++ uwhileEmit "cpCont" cpH cpX (coopCopyBody "cpDstF" "cpSrcF" litLen)

/-- The concrete emit of `wEmitFinalSeq` (abstract labels for the LSIC uif/loop and
    the coopCopy loop), as the `SimSL'`-composition order:
    prefix ++ uif ++ mid ++ coopCopy ++ [bin op op litLen]. -/
def wEmitFinalSeqEmit (litStart litLen lElseF lEndF lHF lXF cpH cpX : String)
    (lsicBody : List SInstr) : List SInstr :=
  finalPreEmit litLen
  ++ (uifEmit "pLitBigF" lElseF lEndF (lsicThen litLen "litExtraF" lHF lXF lsicBody) []
    ++ (finalMidEmit litStart
      ++ (finalCoopEmit litLen cpH cpX ++ [.bin .add "op" "op" (.reg litLen)])))

/-- The `wEmitFinalSeq` eval state right after the `pLitBigF` guard `setp` (before
    the LSIC uif) — where the LSIC side-conditions are checked. -/
def finalAfterSetp (litLen : String) (fuel : Nat) (ws : WState) : WState :=
  WStmt.eval fuel (.setp .ge "pLitBigF" litLen (.imm 15))
    (WStmt.eval fuel (wEmitToken litLen "zero")
      (WStmt.eval fuel (.mov "zero" (.imm 0)) ws))

/-- The `wEmitFinalSeq` eval state at the literal `coopCopy` entry (after prefix,
    LSIC uif, and the two address computes) — where the copy layout is checked. -/
def finalCpEntry (litStart litLen : String) (fuel : Nat) (ws : WState) : WState :=
  WStmt.eval fuel (.bin .add "cpSrcF" "inBase" (.reg litStart))
    (WStmt.eval fuel (.bin .add "cpDstF" "outBase" (.reg "op"))
      (WStmt.eval fuel (.uif "pLitBigF"
          (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
        (finalAfterSetp litLen fuel ws)))

/-- The final literal-run `wEmitFinalSeq litStart litLen` simulates.  Bespoke glue of
    the straight-line prefix/mid/suffix (`SimSL'`), the `pLitBigF` LSIC block
    (`simSL'_lsicUif`), and the literal `coopCopy` (`simSL_coopCopy`).  The caller
    discharges the LSIC guard/fuel side-conditions on `finalAfterSetp` and the copy
    layout on `finalCpEntry`. -/
theorem simSL'_wEmitFinalSeq (R : List String) (litStart litLen : String)
    (lElseF lEndF lHF lXF cpH cpX : String)
    (hll : litLen ∈ R) (hls : litStart ∈ R) (hzero : "zero" ∈ R)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hib : "inBase" ∈ R) (hsb : "sbAddr" ∈ R)
    (htokHi : "tokHi" ∈ R) (htok : "tok" ∈ R) (hpb : "pLitBigF" ∈ R)
    (hle : "litExtraF" ∈ R) (hc255 : "c255" ∈ R) (hlsicC : "lsicC" ∈ R)
    (hcd : "cpDstF" ∈ R) (hcs : "cpSrcF" ∈ R)
    (hRcp : ∀ r ∈ R, r ∉ coopCopyScratch)
    (lsicBody : List SInstr)
    (hbodyDef : lsicBody =
      ([.mov "c255" (SArg.imm 255)]
        ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
            ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
          ++ ([.bin .sub "litExtraF" "litExtraF" (.imm 255)]
            ++ [.setp .ge "lsicC" "litExtraF" (.imm 255)])))) :
    ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (ws : WState) (fuel : Nat),
      ss.pc = base →
      SegAt prog base (wEmitFinalSeqEmit litStart litLen lElseF lEndF lHF lXF cpH cpX lsicBody) →
      LabelsResolve prog base
        (wEmitFinalSeqEmit litStart litLen lElseF lEndF lHF lXF cpH cpX lsicBody) →
      Couple R ss ws → MachInv ib ss →
      ((finalAfterSetp litLen fuel ws).regs "pLitBigF" = 1 →
          15 ≤ ((finalAfterSetp litLen fuel ws).regs litLen).toNat) →
      ((finalAfterSetp litLen fuel ws).regs "pLitBigF" = 1 →
          ((finalAfterSetp litLen fuel ws).regs litLen).toNat < 255 * fuel + 15) →
      ((finalCpEntry litStart litLen fuel ws).regs "cpDstF").toNat < 2 ^ 32 →
      ((finalCpEntry litStart litLen fuel ws).regs "cpSrcF").toNat < 2 ^ 32 →
      ((finalCpEntry litStart litLen fuel ws).regs litLen).toNat < 2 ^ 32 →
      (((finalCpEntry litStart litLen fuel ws).regs "cpDstF").toNat
          + ((finalCpEntry litStart litLen fuel ws).regs litLen).toNat
          ≤ ((finalCpEntry litStart litLen fuel ws).regs "cpSrcF").toNat
        ∨ ((finalCpEntry litStart litLen fuel ws).regs "cpSrcF").toNat
          + ((finalCpEntry litStart litLen fuel ws).regs litLen).toNat
          ≤ ((finalCpEntry litStart litLen fuel ws).regs "cpDstF").toNat) →
      (((finalCpEntry litStart litLen fuel ws).regs "cpDstF").toNat
          + ((finalCpEntry litStart litLen fuel ws).regs litLen).toNat
          ≤ (finalCpEntry litStart litLen fuel ws).gmem.size) →
      ∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
        ss'.pc = base +
          (wEmitFinalSeqEmit litStart litLen lElseF lEndF lHF lXF cpH cpX lsicBody).length ∧
        Couple R ss' ((wEmitFinalSeq litStart litLen).eval fuel ws) ∧ MachInv ib ss' := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi hgd hfl hb1 hb2 hb3 hdisj hsz
  rw [wEmitFinalSeqEmit] at hseg hlr
  -- Step 1: prefix, via `simSL'_finalPrefix`.
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1⟩ :=
    (simSL'_finalPrefix R litStart litLen hll hzero hout hop hsb htokHi htok hpb)
      prog base ss ws fuel hpc hseg.append_left
      hlr.append_left hc hmi
  have hEval1 : (wseq [ .mov "zero" (.imm 0), wEmitToken litLen "zero",
        .setp .ge "pLitBigF" litLen (.imm 15) ]).eval fuel ws = finalAfterSetp litLen fuel ws := by
    simp only [wseq, WStmt.eval, finalAfterSetp]
  rw [hEval1] at hc1
  have hpc1' : ss1.pc = base + (finalPreEmit litLen).length := by
    rw [finalPreEmit]; exact hpc1
  have hsegAfterPre : SegAt prog (base + (finalPreEmit litLen).length)
      (uifEmit "pLitBigF" lElseF lEndF (lsicThen litLen "litExtraF" lHF lXF lsicBody) []
        ++ (finalMidEmit litStart
          ++ (finalCoopEmit litLen cpH cpX ++ [.bin .add "op" "op" (.reg litLen)]))) := by
    exact hseg.append_right
  have hlrAfterPre : LabelsResolve prog (base + (finalPreEmit litLen).length)
      (uifEmit "pLitBigF" lElseF lEndF (lsicThen litLen "litExtraF" lHF lXF lsicBody) []
        ++ (finalMidEmit litStart
          ++ (finalCoopEmit litLen cpH cpX ++ [.bin .add "op" "op" (.reg litLen)]))) := by
    exact hlr.append_right
  -- Step 2: the `pLitBigF` LSIC uif, via `simSL'_lsicUif`.
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2⟩ :=
    simSL'_lsicUif R "pLitBigF" litLen "litExtraF" lElseF lEndF lHF lXF hpb hle hll
      ⟨by decide, by decide, by decide⟩ (by decide) (by decide) (by decide) (by decide)
      hout hop hsb hc255 hlsicC lsicBody hbodyDef
      prog (base + (finalPreEmit litLen).length) ss1 (finalAfterSetp litLen fuel ws) fuel hpc1'
      hsegAfterPre.append_left hlrAfterPre.append_left hc1 hmi1 hgd hfl
  -- eval after the uif.
  obtain ⟨uifSt, hUif⟩ : ∃ st, (WStmt.uif "pLitBigF"
      (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip).eval fuel
      (finalAfterSetp litLen fuel ws) = st := ⟨_, rfl⟩
  rw [hUif] at hc2
  have hpc2' : ss2.pc = base + (finalPreEmit litLen).length
      + (uifEmit "pLitBigF" lElseF lEndF (lsicThen litLen "litExtraF" lHF lXF lsicBody) []).length :=
    hpc2
  have hsegAfterUif : SegAt prog (base + (finalPreEmit litLen).length
      + (uifEmit "pLitBigF" lElseF lEndF (lsicThen litLen "litExtraF" lHF lXF lsicBody) []).length)
      (finalMidEmit litStart
        ++ (finalCoopEmit litLen cpH cpX ++ [.bin .add "op" "op" (.reg litLen)])) :=
    hsegAfterPre.append_right
  have hlrAfterUif : LabelsResolve prog (base + (finalPreEmit litLen).length
      + (uifEmit "pLitBigF" lElseF lEndF (lsicThen litLen "litExtraF" lHF lXF lsicBody) []).length)
      (finalMidEmit litStart
        ++ (finalCoopEmit litLen cpH cpX ++ [.bin .add "op" "op" (.reg litLen)])) :=
    hlrAfterPre.append_right
  -- Step 3: mid (two address computes), via `simSL'_finalMid`.
  obtain ⟨n3, ss3, hr3, hpc3, hc3, hmi3⟩ :=
    (simSL'_finalMid R litStart hout hop hib hls hcd hcs)
      prog _ ss2 uifSt fuel hpc2' hsegAfterUif.append_left
      hlrAfterUif.append_left hc2 hmi2
  have hEval3 : (wseq [ .bin .add "cpDstF" "outBase" (.reg "op"),
        .bin .add "cpSrcF" "inBase" (.reg litStart) ]).eval fuel uifSt
      = finalCpEntry litStart litLen fuel ws := by
    rw [finalCpEntry, hUif]; simp only [wseq, WStmt.eval]
  rw [hEval3] at hc3
  have hpc3' : ss3.pc = base + (finalPreEmit litLen).length
      + (uifEmit "pLitBigF" lElseF lEndF (lsicThen litLen "litExtraF" lHF lXF lsicBody) []).length
      + (finalMidEmit litStart).length := hpc3
  have hsegAfterMid := hsegAfterUif.append_right
  have hlrAfterMid := hlrAfterUif.append_right
  -- Step 4: coopCopy, via `simSL_coopCopy`.
  obtain ⟨n4, ss4, hr4, hpc4, hc4, hmi4⟩ :=
    simSL_coopCopy R "cpDstF" "cpSrcF" litLen cpH cpX hcd hcs hll hRcp
      (finalCpEntry litStart litLen fuel ws) hb1 hb2 hb3 hdisj hsz
      prog _ ss3 fuel hpc3'
      hsegAfterMid.append_left
      hlrAfterMid.append_left hc3 hmi3
  obtain ⟨coopSt, hCoop⟩ : ∃ st, (WStmt.coopCopy "cpDstF" "cpSrcF" litLen).eval fuel
      (finalCpEntry litStart litLen fuel ws) = st := ⟨_, rfl⟩
  rw [hCoop] at hc4
  have hpc4' : ss4.pc = base + (finalPreEmit litLen).length
      + (uifEmit "pLitBigF" lElseF lEndF (lsicThen litLen "litExtraF" lHF lXF lsicBody) []).length
      + (finalMidEmit litStart).length + (finalCoopEmit litLen cpH cpX).length := by
    rw [finalCoopEmit]; exact hpc4
  have hsegTail := hsegAfterMid.append_right
  have hlrTail := hlrAfterMid.append_right
  -- Step 5: the trailing `bin add op op litLen`.
  obtain ⟨n5, ss5, hr5, hpc5, hc5, hmi5⟩ :=
    (simSL'_bin R .add "op" "op" (.reg litLen) hop (fun n h => by cases h; exact hll) (by decide))
      prog _ ss4 coopSt fuel hpc4' hsegTail hlrTail hc4 hmi4
  -- assemble.
  refine ⟨n1 + (n2 + (n3 + (n4 + n5))), ss5,
    sreaches_trans prog n1 _ _ _ _ hr1
      (sreaches_trans prog n2 _ _ _ _ hr2
        (sreaches_trans prog n3 _ _ _ _ hr3
          (sreaches_trans prog n4 n5 _ _ _ hr4 hr5))), ?_, ?_, hmi5⟩
  · rw [hpc5, wEmitFinalSeqEmit]
    simp only [List.length_append, List.length_cons, List.length_nil, Nat.add_assoc]
  · -- eval assembly: `hc5` already couples to `(bin op).eval fuel coopSt`, and
    -- `wEmitFinalSeq_eval_seq` expresses the goal as the same nested chain.
    rw [wEmitFinalSeq_eval_seq]
    have hgoalEq : WStmt.eval fuel (.bin .add "op" "op" (.reg litLen))
        (WStmt.eval fuel (.coopCopy "cpDstF" "cpSrcF" litLen)
        (WStmt.eval fuel (.bin .add "cpSrcF" "inBase" (.reg litStart))
        (WStmt.eval fuel (.bin .add "cpDstF" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBigF"
            (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
        (WStmt.eval fuel (.setp .ge "pLitBigF" litLen (.imm 15))
        (WStmt.eval fuel (wEmitToken litLen "zero")
        (WStmt.eval fuel (.mov "zero" (.imm 0)) ws)))))))
        = WStmt.eval fuel (.bin .add "op" "op" (.reg litLen)) coopSt := by
      rw [← hCoop, finalCpEntry, finalAfterSetp]
    rw [hgoalEq]; exact hc5

/-- Emit of the `wEmitMatchSeq` prefix (sub mlm; min tokLo; wEmitToken; setp). -/
def matchPreEmit (litLen ml : String) : List SInstr :=
  ([.bin .sub "mlm" ml (.imm 4)] : List SInstr)
    ++ (([.bin .min "tokLo" "mlm" (.imm 15)] : List SInstr)
      ++ ((([.bin .min "tokHi" litLen (.imm 15)]
        ++ ([.bin .shl "tok" "tokHi" (.imm 4)]
          ++ ([.bin .bor "tok" "tok" (.reg "tokLo")]
            ++ ([.bin .add "sbAddr" "outBase" (.reg "op")]
              ++ ([.stg "sbAddr" "tok"] ++ [.bin .add "op" "op" (.imm 1)])))))
        ++ [.setp .ge "pLitBig" litLen (.imm 15)])))

/-- Emit of the two copy-address computes for the match literal copy. -/
def matchMidEmit (litStart : String) : List SInstr :=
  ([.bin .add "cpDst" "outBase" (.reg "op")] : List SInstr)
    ++ [.bin .add "cpSrc" "inBase" (.reg litStart)]

/-- Emit of the match literal `.coopCopy cpDst cpSrc litLen` loop. -/
def matchCoopEmit (litLen cpH cpX : String) : List SInstr :=
  ([.mov "cpI" (.imm 0), .setp .lt "cpCont" "cpI" (.reg litLen)] : List SInstr)
    ++ uwhileEmit "cpCont" cpH cpX (coopCopyBody "cpDst" "cpSrc" litLen)

/-- Emit of the offset-store + `pMatBig` guard block. -/
def matchOffEmit (litLen off : String) : List SInstr :=
  ([.bin .add "op" "op" (.reg litLen)] : List SInstr)
    ++ (([.bin .band "offLo" off (.imm 255)] : List SInstr)
      ++ ((([.bin .add "sbAddr" "outBase" (.reg "op")]
          ++ ([.stg "sbAddr" "offLo"] ++ [.bin .add "op" "op" (.imm 1)])) : List SInstr)
        ++ (([.bin .shr "offHi" off (.imm 8)] : List SInstr)
          ++ (([.bin .band "offHi" "offHi" (.imm 255)] : List SInstr)
            ++ ((([.bin .add "sbAddr" "outBase" (.reg "op")]
                ++ ([.stg "sbAddr" "offHi"] ++ [.bin .add "op" "op" (.imm 1)])) : List SInstr)
              ++ [.setp .ge "pMatBig" "mlm" (.imm 15)])))))

/-- Prefix of `wEmitMatchSeq` up to the `pLitBig` guard: `sub mlm; min tokLo;
    wEmitToken; setp pLitBig`.  A single `SimSL'`. -/
theorem simSL'_matchPre (R : List String) (litLen ml : String)
    (hll : litLen ∈ R) (hml : ml ∈ R) (hmlm : "mlm" ∈ R) (htokLo : "tokLo" ∈ R)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R)
    (htokHi : "tokHi" ∈ R) (htok : "tok" ∈ R) (hpb : "pLitBig" ∈ R) :
    SimSL' ib R
      (wseq [ .bin .sub "mlm" ml (.imm 4), .bin .min "tokLo" "mlm" (.imm 15),
              wEmitToken litLen "tokLo", .setp .ge "pLitBig" litLen (.imm 15) ])
      (matchPreEmit litLen ml) := by
  rw [matchPreEmit]
  apply simSL'_seq
  · exact simSL'_bin R .sub "mlm" ml (.imm 4) hml (fun n h => by cases h) (by decide)
  apply simSL'_seq
  · exact simSL'_bin R .min "tokLo" "mlm" (.imm 15) hmlm (fun n h => by cases h) (by decide)
  apply simSL'_seq
  · exact simSL'_wEmitToken R litLen "tokLo" hll htokLo hout hop hsb htokHi htok
  · exact simSL'_setp R .ge "pLitBig" litLen (.imm 15) hll (fun n h => by cases h) (by decide)

/-- The two copy-address computes for the match literal: `cpDst := outBase+op`,
    `cpSrc := inBase+litStart`. -/
theorem simSL'_matchMid (R : List String) (litStart : String)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hib : "inBase" ∈ R) (hls : litStart ∈ R)
    (hcd : "cpDst" ∈ R) (hcs : "cpSrc" ∈ R) :
    SimSL' ib R
      (wseq [ .bin .add "cpDst" "outBase" (.reg "op"),
              .bin .add "cpSrc" "inBase" (.reg litStart) ])
      (matchMidEmit litStart) := by
  rw [matchMidEmit]
  apply simSL'_seq
  · exact simSL'_bin R .add "cpDst" "outBase" (.reg "op") hout (fun n h => by cases h; exact hop)
      (by decide)
  · exact simSL'_bin R .add "cpSrc" "inBase" (.reg litStart) hib
      (fun n h => by cases h; exact hls) (by decide)

/-- The offset-store + `pMatBig` guard block: `bin op += litLen; band offLo;
    wStoreByte offLo; shr offHi; band offHi; wStoreByte offHi; setp pMatBig`. -/
theorem simSL'_matchOff (R : List String) (litLen off : String)
    (hll : litLen ∈ R) (hoff : off ∈ R) (hmlm : "mlm" ∈ R)
    (hoffLo : "offLo" ∈ R) (hoffHi : "offHi" ∈ R)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R) (hpm : "pMatBig" ∈ R) :
    SimSL' ib R
      (wseq [ .bin .add "op" "op" (.reg litLen),
              .bin .band "offLo" off (.imm 255), wStoreByte "offLo",
              .bin .shr "offHi" off (.imm 8), .bin .band "offHi" "offHi" (.imm 255),
              wStoreByte "offHi", .setp .ge "pMatBig" "mlm" (.imm 15) ])
      (matchOffEmit litLen off) := by
  rw [matchOffEmit]
  apply simSL'_seq
  · exact simSL'_bin R .add "op" "op" (.reg litLen) hop (fun n h => by cases h; exact hll) (by decide)
  apply simSL'_seq
  · exact simSL'_bin R .band "offLo" off (.imm 255) hoff (fun n h => by cases h) (by decide)
  apply simSL'_seq
  · exact simSL'_wStoreByte R "offLo" hoffLo hout hop hsb
  apply simSL'_seq
  · exact simSL'_bin R .shr "offHi" off (.imm 8) hoff (fun n h => by cases h) (by decide)
  apply simSL'_seq
  · exact simSL'_bin R .band "offHi" "offHi" (.imm 255) hoffHi (fun n h => by cases h) (by decide)
  apply simSL'_seq
  · exact simSL'_wStoreByte R "offHi" hoffHi hout hop hsb
  · exact simSL'_setp R .ge "pMatBig" "mlm" (.imm 15) hmlm (fun n h => by cases h) (by decide)

/-- The full emit of `wEmitMatchSeq`, in `SimSL'`-composition order:
    matchPre ++ uif(pLitBig) ++ matchMid ++ coopCopy ++ matchOff ++ uif(pMatBig). -/
def wEmitMatchSeqEmit (litStart litLen off ml lElseL lEndL lHL lXL cpH cpX
    lElseM lEndM lHM lXM : String) (lsicL lsicM : List SInstr) : List SInstr :=
  matchPreEmit litLen ml
  ++ (uifEmit "pLitBig" lElseL lEndL (lsicThen litLen "litExtra" lHL lXL lsicL) []
    ++ (matchMidEmit litStart
      ++ (matchCoopEmit litLen cpH cpX
        ++ (matchOffEmit litLen off
          ++ uifEmit "pMatBig" lElseM lEndM (lsicThen "mlm" "matExtra" lHM lXM lsicM) []))))

/-- Eval of `wEmitMatchSeq` as the right-nested `.seq` chain. -/
theorem wEmitMatchSeq_eval_seq (litStart litLen off ml : String) (fuel : Nat) (ws : WState) :
    (wEmitMatchSeq litStart litLen off ml).eval fuel ws
      = (WStmt.eval fuel (.uif "pMatBig"
            (wseq [ .bin .sub "matExtra" "mlm" (.imm 15), wEmitLSIC "matExtra" ]) .skip)
        (WStmt.eval fuel (.setp .ge "pMatBig" "mlm" (.imm 15))
        (WStmt.eval fuel (wStoreByte "offHi")
        (WStmt.eval fuel (.bin .band "offHi" "offHi" (.imm 255))
        (WStmt.eval fuel (.bin .shr "offHi" off (.imm 8))
        (WStmt.eval fuel (wStoreByte "offLo")
        (WStmt.eval fuel (.bin .band "offLo" off (.imm 255))
        (WStmt.eval fuel (.bin .add "op" "op" (.reg litLen))
        (WStmt.eval fuel (.coopCopy "cpDst" "cpSrc" litLen)
        (WStmt.eval fuel (.bin .add "cpSrc" "inBase" (.reg litStart))
        (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBig"
            (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
        (WStmt.eval fuel (.setp .ge "pLitBig" litLen (.imm 15))
        (WStmt.eval fuel (wEmitToken litLen "tokLo")
        (WStmt.eval fuel (.bin .min "tokLo" "mlm" (.imm 15))
        (WStmt.eval fuel (.bin .sub "mlm" ml (.imm 4)) ws)))))))))))))))) := by
  simp only [wEmitMatchSeq, wseq, WStmt.eval]

/-- State after the `pLitBig` guard `setp` (LSIC side-conditions checked here). -/
def matchAfterSetp (litLen ml : String) (fuel : Nat) (ws : WState) : WState :=
  WStmt.eval fuel (.setp .ge "pLitBig" litLen (.imm 15))
    (WStmt.eval fuel (wEmitToken litLen "tokLo")
    (WStmt.eval fuel (.bin .min "tokLo" "mlm" (.imm 15))
    (WStmt.eval fuel (.bin .sub "mlm" ml (.imm 4)) ws)))

/-- State at the match literal `coopCopy` entry (copy layout checked here). -/
def matchCpEntry (litStart litLen ml : String) (fuel : Nat) (ws : WState) : WState :=
  WStmt.eval fuel (.bin .add "cpSrc" "inBase" (.reg litStart))
    (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op"))
    (WStmt.eval fuel (.uif "pLitBig"
        (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
      (matchAfterSetp litLen ml fuel ws)))

/-- `matchAfterSetp` (sub `mlm`; min `tokLo`; `wEmitToken`; setp `pLitBig`) is
    straight-line and writes only `{mlm,tokLo,tokHi,tok,sbAddr,op,pLitBig}` (+`gmem`). -/
theorem matchAfterSetp_frame (litLen ml r : String) (fuel : Nat) (ws : WState)
    (h1 : r ≠ "mlm") (h2 : r ≠ "tokLo") (h3 : r ≠ "tokHi") (h4 : r ≠ "tok")
    (h5 : r ≠ "sbAddr") (h6 : r ≠ "op") (h7 : r ≠ "pLitBig") :
    (matchAfterSetp litLen ml fuel ws).regs r = ws.regs r := by
  simp [matchAfterSetp, wEmitToken, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte,
    WOp.run, SCmp.run, WArg.eval, h1, h2, h3, h4, h5, h6, h7]

/-- The `pLitBig` literal-LSIC `uif` on `matchAfterSetp` preserves every register
    outside the match-emit scratch set. -/
theorem matchUifOut_frame (litLen ml r : String) (fuel : Nat) (ws : WState)
    (h1 : r ≠ "mlm") (h2 : r ≠ "tokLo") (h3 : r ≠ "tokHi") (h4 : r ≠ "tok")
    (h5 : r ≠ "sbAddr") (h6 : r ≠ "op") (h7 : r ≠ "pLitBig")
    (h8 : r ≠ "litExtra") (h9 : r ≠ "lsicC") (h10 : r ≠ "c255") :
    (WStmt.eval fuel (.uif "pLitBig"
        (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
      (matchAfterSetp litLen ml fuel ws)).regs r = ws.regs r := by
  simp only [WStmt.eval]
  by_cases hg : ((matchAfterSetp litLen ml fuel ws).regs "pLitBig" == 1) = true
  · rw [if_pos hg, show (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ])
        = (WStmt.bin WOp.sub "litExtra" litLen (WArg.imm 15)).seq (wEmitLSIC "litExtra") from rfl]
    simp only [WStmt.eval.eq_2]
    rw [wEmitLSIC_frame "litExtra" r h9 h10 h5 h6 h8]
    simp only [WStmt.eval, WState.setReg, WOp.run, WArg.eval, h8]
    exact matchAfterSetp_frame litLen ml r fuel ws h1 h2 h3 h4 h5 h6 h7
  · rw [if_neg hg]
    exact matchAfterSetp_frame litLen ml r fuel ws h1 h2 h3 h4 h5 h6 h7

/-- `matchCpEntry` preserves every register outside the match-emit scratch/output set;
    in particular `inBase`, `litAnchor` (= `litStart`), `outBase`, `litLen` survive. -/
theorem matchCpEntry_frame (litStart litLen ml r : String) (fuel : Nat) (ws : WState)
    (hcs : r ≠ "cpSrc") (hcd : r ≠ "cpDst") (h1 : r ≠ "mlm") (h2 : r ≠ "tokLo")
    (h3 : r ≠ "tokHi") (h4 : r ≠ "tok") (h5 : r ≠ "sbAddr") (h6 : r ≠ "op") (h7 : r ≠ "pLitBig")
    (h8 : r ≠ "litExtra") (h9 : r ≠ "lsicC") (h10 : r ≠ "c255") :
    (matchCpEntry litStart litLen ml fuel ws).regs r = ws.regs r := by
  rw [matchCpEntry]
  have e1 : ∀ (st : WState),
      (WStmt.eval fuel (.bin .add "cpSrc" "inBase" (.reg litStart)) st).regs r = st.regs r :=
    fun st => by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcs]
  have e2 : ∀ (st : WState),
      (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op")) st).regs r = st.regs r :=
    fun st => by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcd]
  rw [e1, e2]
  exact matchUifOut_frame litLen ml r fuel ws h1 h2 h3 h4 h5 h6 h7 h8 h9 h10

/-- The literal-copy source address `cpSrc = inBase + litStart` at the copy entry. -/
theorem matchCpEntry_cpSrc (litStart litLen ml : String) (fuel : Nat) (ws : WState)
    (hcd : litStart ≠ "cpDst") (h1 : litStart ≠ "mlm") (h2 : litStart ≠ "tokLo")
    (h3 : litStart ≠ "tokHi") (h4 : litStart ≠ "tok") (h5 : litStart ≠ "sbAddr")
    (h6 : litStart ≠ "op") (h7 : litStart ≠ "pLitBig") (h8 : litStart ≠ "litExtra")
    (h9 : litStart ≠ "lsicC") (h10 : litStart ≠ "c255") :
    (matchCpEntry litStart litLen ml fuel ws).regs "cpSrc"
      = ws.regs "inBase" + ws.regs litStart := by
  have hX : ∀ (r' : String), r' ≠ "cpDst" → r' ≠ "mlm" → r' ≠ "tokLo" → r' ≠ "tokHi" →
      r' ≠ "tok" → r' ≠ "sbAddr" → r' ≠ "op" → r' ≠ "pLitBig" → r' ≠ "litExtra" →
      r' ≠ "lsicC" → r' ≠ "c255" →
      (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBig"
            (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
          (matchAfterSetp litLen ml fuel ws))).regs r' = ws.regs r' := by
    intro r' hcd' a1 a2 a3 a4 a5 a6 a7 a8 a9 a10
    rw [show (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBig"
            (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
          (matchAfterSetp litLen ml fuel ws))).regs r'
      = (WStmt.eval fuel (.uif "pLitBig"
            (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
          (matchAfterSetp litLen ml fuel ws)).regs r' from by
        simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcd']]
    exact matchUifOut_frame litLen ml r' fuel ws a1 a2 a3 a4 a5 a6 a7 a8 a9 a10
  have hib := hX "inBase" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
  have hls := hX litStart hcd h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  rw [matchCpEntry]
  generalize (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBig"
            (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
          (matchAfterSetp litLen ml fuel ws))) = X at hib hls ⊢
  simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hib, hls]

/-- The `pLitBig` guard equals `litLen ≥ 15` (as `1`/`0`). -/
theorem matchAfterSetp_pLitBig (litLen ml : String) (fuel : Nat) (ws : WState)
    (h1 : litLen ≠ "mlm") (h2 : litLen ≠ "tokLo") (h3 : litLen ≠ "tokHi") (h4 : litLen ≠ "tok")
    (h5 : litLen ≠ "sbAddr") (h6 : litLen ≠ "op") :
    (matchAfterSetp litLen ml fuel ws).regs "pLitBig"
      = (if UInt64.ofNat 15 ≤ ws.regs litLen then 1 else 0) := by
  rw [matchAfterSetp]
  have hY : (WStmt.eval fuel (wEmitToken litLen "tokLo")
      (WStmt.eval fuel (.bin .min "tokLo" "mlm" (.imm 15))
      (WStmt.eval fuel (.bin .sub "mlm" ml (.imm 4)) ws))).regs litLen = ws.regs litLen := by
    simp [wEmitToken, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte, WOp.run,
      WArg.eval, h1, h2, h3, h4, h5, h6]
  generalize hYeq : (WStmt.eval fuel (wEmitToken litLen "tokLo")
      (WStmt.eval fuel (.bin .min "tokLo" "mlm" (.imm 15))
      (WStmt.eval fuel (.bin .sub "mlm" ml (.imm 4)) ws))) = Y at hY ⊢
  simp [WStmt.eval, WState.setReg, SCmp.run, WArg.eval, hY]

/-- State after the `pMatBig` guard `setp` (2nd LSIC side-conditions checked here). -/
def matchAfterMatSetp (litStart litLen off ml : String) (fuel : Nat) (ws : WState) : WState :=
  WStmt.eval fuel (.setp .ge "pMatBig" "mlm" (.imm 15))
    (WStmt.eval fuel (wStoreByte "offHi")
    (WStmt.eval fuel (.bin .band "offHi" "offHi" (.imm 255))
    (WStmt.eval fuel (.bin .shr "offHi" off (.imm 8))
    (WStmt.eval fuel (wStoreByte "offLo")
    (WStmt.eval fuel (.bin .band "offLo" off (.imm 255))
    (WStmt.eval fuel (.bin .add "op" "op" (.reg litLen))
    (WStmt.eval fuel (.coopCopy "cpDst" "cpSrc" litLen)
      (matchCpEntry litStart litLen ml fuel ws))))))))

/-- `mlm = ml - 4` after the match-sequence prefix. -/
theorem matchAfterSetp_mlm (litLen ml : String) (fuel : Nat) (ws : WState) (hm : ml ≠ "mlm") :
    (matchAfterSetp litLen ml fuel ws).regs "mlm" = ws.regs ml - UInt64.ofNat 4 := by
  rw [matchAfterSetp]
  generalize hZeq : (WStmt.eval fuel (.bin .sub "mlm" ml (.imm 4)) ws) = Z at *
  have hZmlm : Z.regs "mlm" = ws.regs ml - UInt64.ofNat 4 := by
    rw [← hZeq]; simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hm]
  simp [WStmt.eval, wEmitToken, wStoreByte, wseq, WState.setReg, WState.stgByte, WOp.run,
    SCmp.run, WArg.eval, hZmlm]

/-- `matchCpEntry` preserves every register the two adds / literal-LSIC `uif` don't
    write, relative to `matchAfterSetp` (so e.g. `mlm` survives to the copy entry). -/
theorem matchCpEntry_over (litStart litLen ml r : String) (fuel : Nat) (ws : WState)
    (hcs : r ≠ "cpSrc") (hcd : r ≠ "cpDst") (h5 : r ≠ "sbAddr") (h6 : r ≠ "op")
    (h8 : r ≠ "litExtra") (h9 : r ≠ "lsicC") (h10 : r ≠ "c255") :
    (matchCpEntry litStart litLen ml fuel ws).regs r = (matchAfterSetp litLen ml fuel ws).regs r := by
  rw [matchCpEntry]
  have e1 : ∀ (st : WState),
      (WStmt.eval fuel (.bin .add "cpSrc" "inBase" (.reg litStart)) st).regs r = st.regs r :=
    fun st => by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcs]
  have e2 : ∀ (st : WState),
      (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op")) st).regs r = st.regs r :=
    fun st => by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcd]
  rw [e1, e2]
  simp only [WStmt.eval]
  by_cases hg : ((matchAfterSetp litLen ml fuel ws).regs "pLitBig" == 1) = true
  · rw [if_pos hg, show (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ])
        = (WStmt.bin WOp.sub "litExtra" litLen (WArg.imm 15)).seq (wEmitLSIC "litExtra") from rfl]
    simp only [WStmt.eval.eq_2]
    rw [wEmitLSIC_frame "litExtra" r h9 h10 h5 h6 h8]
    simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, h8]
  · rw [if_neg hg]

/-- The token store advances `op` by one; `outBase`/`inBase` unchanged. -/
theorem matchAfterSetp_op (litLen ml : String) (fuel : Nat) (ws : WState) :
    (matchAfterSetp litLen ml fuel ws).regs "op" = ws.regs "op" + 1 := by
  simp [matchAfterSetp, wEmitToken, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte,
    WOp.run, SCmp.run, WArg.eval]

/-- The `op` value after the literal-LSIC `uif`: `op + 1 + encNib(litLen).length`
    (token byte + the `pLitBig` continuation bytes).  Uses `EmitContent.eval_wEmitLSIC`
    for the LSIC advance; the two branches of `pLitBig` unify because
    `encNib(litLen).length = 0` when `litLen < 15`. -/
theorem matchUifOut_op (litLen ml : String) (fuel : Nat) (ws : WState)
    (hml : litLen ≠ "mlm") (htl : litLen ≠ "tokLo") (hth : litLen ≠ "tokHi") (htk : litLen ≠ "tok")
    (hsb : litLen ≠ "sbAddr") (hop : litLen ≠ "op") (hpb : litLen ≠ "pLitBig")
    (hfuel : ((ws.regs litLen).toNat - 15) / 255 < fuel) :
    (WStmt.eval fuel (.uif "pLitBig"
        (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
      (matchAfterSetp litLen ml fuel ws)).regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (LZ4.encNib (ws.regs litLen).toNat).length := by
  have hM_op := matchAfterSetp_op litLen ml fuel ws
  have hM_ll := matchAfterSetp_frame litLen ml litLen fuel ws hml htl hth htk hsb hop hpb
  have hM_pb := matchAfterSetp_pLitBig litLen ml fuel ws hml htl hth htk hsb hop
  generalize hMeq : matchAfterSetp litLen ml fuel ws = M at hM_op hM_ll hM_pb ⊢
  simp only [WStmt.eval]
  by_cases hg : UInt64.ofNat 15 ≤ ws.regs litLen
  · have hgN : 15 ≤ (ws.regs litLen).toNat := by
      rw [UInt64.le_iff_toNat_le] at hg; simpa using hg
    have hpb1 : (M.regs "pLitBig" == 1) = true := by rw [hM_pb, if_pos hg]; rfl
    rw [if_pos hpb1, show (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ])
        = (WStmt.bin WOp.sub "litExtra" litLen (WArg.imm 15)).seq (wEmitLSIC "litExtra") from rfl]
    simp only [WStmt.eval.eq_2]
    generalize hM1 : (WStmt.eval fuel (.bin .sub "litExtra" litLen (.imm 15)) M) = M1
    have hM1_le : M1.regs "litExtra" = ws.regs litLen - UInt64.ofNat 15 := by
      rw [← hM1]
      have hstep : (WStmt.eval fuel (.bin .sub "litExtra" litLen (.imm 15)) M).regs "litExtra"
          = M.regs litLen - UInt64.ofNat 15 := by
        simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval]
      rw [hstep, hM_ll]
    have hM1_leN : (M1.regs "litExtra").toNat = (ws.regs litLen).toNat - 15 := by
      rw [hM1_le, UInt64.toNat_sub, show ((UInt64.ofNat 15).toNat) = 15 from by decide,
        show 2 ^ 64 - 15 + (ws.regs litLen).toNat = 2 ^ 64 + ((ws.regs litLen).toNat - 15) from by
          omega, Nat.add_mod_left, Nat.mod_eq_of_lt (by have := (ws.regs litLen).toNat_lt; omega)]
    have hM1_op : M1.regs "op" = ws.regs "op" + 1 := by
      rw [← hM1]
      have hstep : (WStmt.eval fuel (.bin .sub "litExtra" litLen (.imm 15)) M).regs "op"
          = M.regs "op" := by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval]
      rw [hstep, hM_op]
    obtain ⟨hlsic_op, _, _⟩ := EmitContent.eval_wEmitLSIC "litExtra" M1 fuel
      (by decide) (by decide) (by decide) (by decide) (by decide) (by rw [hM1_leN]; exact hfuel)
    rw [hlsic_op, hM1_op, hM1_leN, AlgorithmLib.LZ4Imp.ext_length,
      AlgorithmLib.LZ4Imp.encNib_length, if_neg (by omega)]
  · have hgN : (ws.regs litLen).toNat < 15 := by
      rw [UInt64.le_iff_toNat_le] at hg
      simp only [show ((UInt64.ofNat 15).toNat) = 15 from by decide] at hg; omega
    have hpb0 : (M.regs "pLitBig" == 1) = false := by rw [hM_pb, if_neg hg]; rfl
    rw [if_neg (by rw [hpb0]; exact Bool.false_ne_true), hM_op,
      AlgorithmLib.LZ4Imp.encNib_length, if_pos hgN]
    simp

/-- The literal-copy destination `cpDst = outBase + op + 1 + encNib(litLen).length`
    at the copy entry (token + literal-LSIC bytes already emitted). -/
theorem matchCpEntry_cpDst (litStart litLen ml : String) (fuel : Nat) (ws : WState)
    (hml : litLen ≠ "mlm") (htl : litLen ≠ "tokLo") (hth : litLen ≠ "tokHi") (htk : litLen ≠ "tok")
    (hsb : litLen ≠ "sbAddr") (hop : litLen ≠ "op") (hpb : litLen ≠ "pLitBig")
    (hfuel : ((ws.regs litLen).toNat - 15) / 255 < fuel) :
    (matchCpEntry litStart litLen ml fuel ws).regs "cpDst"
      = ws.regs "outBase" + ws.regs "op" + 1 + UInt64.ofNat (LZ4.encNib (ws.regs litLen).toNat).length := by
  have hob : (WStmt.eval fuel (.uif "pLitBig"
      (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
    (matchAfterSetp litLen ml fuel ws)).regs "outBase" = ws.regs "outBase" :=
    matchUifOut_frame litLen ml "outBase" fuel ws (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have hop' := matchUifOut_op litLen ml fuel ws hml htl hth htk hsb hop hpb hfuel
  rw [matchCpEntry]
  generalize hUeq : (WStmt.eval fuel (.uif "pLitBig"
      (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
    (matchAfterSetp litLen ml fuel ws)) = U at hob hop'
  simp only [WStmt.eval, WState.setReg, WOp.run, WArg.eval]
  rw [hob, hop']
  ac_rfl

/-- `mlm = ml - 4` still holds after the offset-store block (nothing there writes `mlm`). -/
theorem matchAfterMatSetp_mlm (litStart litLen off ml : String) (fuel : Nat) (ws : WState)
    (hm : ml ≠ "mlm") :
    (matchAfterMatSetp litStart litLen off ml fuel ws).regs "mlm" = ws.regs ml - UInt64.ofNat 4 := by
  rw [matchAfterMatSetp]
  generalize hCeq : matchCpEntry litStart litLen ml fuel ws = C at *
  have hCmlm : C.regs "mlm" = ws.regs ml - UInt64.ofNat 4 := by
    rw [← hCeq, matchCpEntry_over litStart litLen ml "mlm" fuel ws (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide),
      matchAfterSetp_mlm litLen ml fuel ws hm]
  simp [WStmt.eval, wStoreByte, wseq, WState.setReg, WState.stgByte, WOp.run, SCmp.run,
    WArg.eval, hCmlm]

/-- The `pMatBig` guard equals `mlm ≥ 15` (`mlm = ml - 4`). -/
theorem matchAfterMatSetp_pMatBig (litStart litLen off ml : String) (fuel : Nat) (ws : WState)
    (hm : ml ≠ "mlm") :
    (matchAfterMatSetp litStart litLen off ml fuel ws).regs "pMatBig"
      = (if UInt64.ofNat 15 ≤ ws.regs ml - UInt64.ofNat 4 then 1 else 0) := by
  rw [matchAfterMatSetp]
  generalize hCeq : matchCpEntry litStart litLen ml fuel ws = C at *
  have hCmlm : C.regs "mlm" = ws.regs ml - UInt64.ofNat 4 := by
    rw [← hCeq, matchCpEntry_over litStart litLen ml "mlm" fuel ws (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide),
      matchAfterSetp_mlm litLen ml fuel ws hm]
  have hmlmChain : (WStmt.eval fuel (wStoreByte "offHi")
      (WStmt.eval fuel (.bin .band "offHi" "offHi" (.imm 255))
      (WStmt.eval fuel (.bin .shr "offHi" off (.imm 8))
      (WStmt.eval fuel (wStoreByte "offLo")
      (WStmt.eval fuel (.bin .band "offLo" off (.imm 255))
      (WStmt.eval fuel (.bin .add "op" "op" (.reg litLen))
      (WStmt.eval fuel (.coopCopy "cpDst" "cpSrc" litLen) C))))))).regs "mlm"
      = ws.regs ml - UInt64.ofNat 4 := by
    simp [WStmt.eval, wStoreByte, wseq, WState.setReg, WState.stgByte, WOp.run, WArg.eval, hCmlm]
  generalize hDeq : (WStmt.eval fuel (wStoreByte "offHi")
      (WStmt.eval fuel (.bin .band "offHi" "offHi" (.imm 255))
      (WStmt.eval fuel (.bin .shr "offHi" off (.imm 8))
      (WStmt.eval fuel (wStoreByte "offLo")
      (WStmt.eval fuel (.bin .band "offLo" off (.imm 255))
      (WStmt.eval fuel (.bin .add "op" "op" (.reg litLen))
      (WStmt.eval fuel (.coopCopy "cpDst" "cpSrc" litLen) C))))))) = D at hmlmChain ⊢
  simp [WStmt.eval, WState.setReg, SCmp.run, WArg.eval, hmlmChain]

/-- The match-sequence emit `wEmitMatchSeq litStart litLen off ml` simulates.
    Bespoke glue of the straight-line prefix/mid/off segments (`SimSL'`), the two
    LSIC blocks (`simSL'_lsicUif`, for `pLitBig`/`pMatBig`), and the literal
    `coopCopy` (`simSL_coopCopy`).  The caller discharges the LSIC guard/fuel
    side-conditions on `matchAfterSetp`/`matchAfterMatSetp` and the copy layout on
    `matchCpEntry`. -/
theorem simSL'_wEmitMatchSeq (R : List String) (litStart litLen off ml : String)
    (lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM : String)
    (hll : litLen ∈ R) (hls : litStart ∈ R) (hoff : off ∈ R) (hml : ml ∈ R)
    (hmlm : "mlm" ∈ R) (htokLo : "tokLo" ∈ R) (htokHi : "tokHi" ∈ R) (htok : "tok" ∈ R)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hib : "inBase" ∈ R) (hsb : "sbAddr" ∈ R)
    (hpb : "pLitBig" ∈ R) (hpm : "pMatBig" ∈ R)
    (hle : "litExtra" ∈ R) (hme : "matExtra" ∈ R) (hc255 : "c255" ∈ R) (hlsicC : "lsicC" ∈ R)
    (hoffLo : "offLo" ∈ R) (hoffHi : "offHi" ∈ R)
    (hcd : "cpDst" ∈ R) (hcs : "cpSrc" ∈ R)
    (hRcp : ∀ r ∈ R, r ∉ coopCopyScratch)
    (lsicL lsicM : List SInstr)
    (hLdef : lsicL =
      ([.mov "c255" (SArg.imm 255)]
        ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
            ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
          ++ ([.bin .sub "litExtra" "litExtra" (.imm 255)]
            ++ [.setp .ge "lsicC" "litExtra" (.imm 255)]))))
    (hMdef : lsicM =
      ([.mov "c255" (SArg.imm 255)]
        ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
            ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
          ++ ([.bin .sub "matExtra" "matExtra" (.imm 255)]
            ++ [.setp .ge "lsicC" "matExtra" (.imm 255)])))) :
    ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (ws : WState) (fuel : Nat),
      ss.pc = base →
      SegAt prog base
        (wEmitMatchSeqEmit litStart litLen off ml lElseL lEndL lHL lXL cpH cpX
          lElseM lEndM lHM lXM lsicL lsicM) →
      LabelsResolve prog base
        (wEmitMatchSeqEmit litStart litLen off ml lElseL lEndL lHL lXL cpH cpX
          lElseM lEndM lHM lXM lsicL lsicM) →
      Couple R ss ws → MachInv ib ss →
      -- 1st LSIC (pLitBig on litLen):
      ((matchAfterSetp litLen ml fuel ws).regs "pLitBig" = 1 →
          15 ≤ ((matchAfterSetp litLen ml fuel ws).regs litLen).toNat) →
      ((matchAfterSetp litLen ml fuel ws).regs "pLitBig" = 1 →
          ((matchAfterSetp litLen ml fuel ws).regs litLen).toNat < 255 * fuel + 15) →
      -- coopCopy layout on `matchCpEntry`:
      ((matchCpEntry litStart litLen ml fuel ws).regs "cpDst").toNat < 2 ^ 32 →
      ((matchCpEntry litStart litLen ml fuel ws).regs "cpSrc").toNat < 2 ^ 32 →
      ((matchCpEntry litStart litLen ml fuel ws).regs litLen).toNat < 2 ^ 32 →
      (((matchCpEntry litStart litLen ml fuel ws).regs "cpDst").toNat
          + ((matchCpEntry litStart litLen ml fuel ws).regs litLen).toNat
          ≤ ((matchCpEntry litStart litLen ml fuel ws).regs "cpSrc").toNat
        ∨ ((matchCpEntry litStart litLen ml fuel ws).regs "cpSrc").toNat
          + ((matchCpEntry litStart litLen ml fuel ws).regs litLen).toNat
          ≤ ((matchCpEntry litStart litLen ml fuel ws).regs "cpDst").toNat) →
      (((matchCpEntry litStart litLen ml fuel ws).regs "cpDst").toNat
          + ((matchCpEntry litStart litLen ml fuel ws).regs litLen).toNat
          ≤ (matchCpEntry litStart litLen ml fuel ws).gmem.size) →
      -- 2nd LSIC (pMatBig on mlm):
      ((matchAfterMatSetp litStart litLen off ml fuel ws).regs "pMatBig" = 1 →
          15 ≤ ((matchAfterMatSetp litStart litLen off ml fuel ws).regs "mlm").toNat) →
      ((matchAfterMatSetp litStart litLen off ml fuel ws).regs "pMatBig" = 1 →
          ((matchAfterMatSetp litStart litLen off ml fuel ws).regs "mlm").toNat < 255 * fuel + 15) →
      ∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
        ss'.pc = base +
          (wEmitMatchSeqEmit litStart litLen off ml lElseL lEndL lHL lXL cpH cpX
            lElseM lEndM lHM lXM lsicL lsicM).length ∧
        Couple R ss' ((wEmitMatchSeq litStart litLen off ml).eval fuel ws) ∧ MachInv ib ss' := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi hgdL hflL hb1 hb2 hb3 hdisj hsz hgdM hflM
  rw [wEmitMatchSeqEmit] at hseg hlr
  -- Step 1: matchPre.
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1⟩ :=
    (simSL'_matchPre R litLen ml hll hml hmlm htokLo hout hop hsb htokHi htok hpb)
      prog base ss ws fuel hpc hseg.append_left hlr.append_left hc hmi
  have hEval1 : (wseq [ .bin .sub "mlm" ml (.imm 4), .bin .min "tokLo" "mlm" (.imm 15),
        wEmitToken litLen "tokLo", .setp .ge "pLitBig" litLen (.imm 15) ]).eval fuel ws
      = matchAfterSetp litLen ml fuel ws := by
    simp only [wseq, WStmt.eval, matchAfterSetp]
  rw [hEval1] at hc1
  have hpc1' : ss1.pc = base + (matchPreEmit litLen ml).length := by
    rw [matchPreEmit]; exact hpc1
  have hsegR1 : SegAt prog (base + (matchPreEmit litLen ml).length)
      (uifEmit "pLitBig" lElseL lEndL (lsicThen litLen "litExtra" lHL lXL lsicL) []
        ++ (matchMidEmit litStart
          ++ (matchCoopEmit litLen cpH cpX
            ++ (matchOffEmit litLen off
              ++ uifEmit "pMatBig" lElseM lEndM (lsicThen "mlm" "matExtra" lHM lXM lsicM) [])))) :=
    hseg.append_right
  have hlrR1 : LabelsResolve prog (base + (matchPreEmit litLen ml).length)
      (uifEmit "pLitBig" lElseL lEndL (lsicThen litLen "litExtra" lHL lXL lsicL) []
        ++ (matchMidEmit litStart
          ++ (matchCoopEmit litLen cpH cpX
            ++ (matchOffEmit litLen off
              ++ uifEmit "pMatBig" lElseM lEndM (lsicThen "mlm" "matExtra" lHM lXM lsicM) [])))) :=
    hlr.append_right
  -- Step 2: pLitBig LSIC uif.
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2⟩ :=
    simSL'_lsicUif R "pLitBig" litLen "litExtra" lElseL lEndL lHL lXL hpb hle hll
      ⟨by decide, by decide, by decide⟩ (by decide) (by decide) (by decide) (by decide)
      hout hop hsb hc255 hlsicC lsicL hLdef
      prog (base + (matchPreEmit litLen ml).length) ss1 (matchAfterSetp litLen ml fuel ws) fuel
      hpc1' hsegR1.append_left hlrR1.append_left hc1 hmi1 hgdL hflL
  obtain ⟨uifSt1, hUif1⟩ : ∃ st, (WStmt.uif "pLitBig"
      (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip).eval fuel
      (matchAfterSetp litLen ml fuel ws) = st := ⟨_, rfl⟩
  rw [hUif1] at hc2
  have hpc2' : ss2.pc = base + (matchPreEmit litLen ml).length
      + (uifEmit "pLitBig" lElseL lEndL (lsicThen litLen "litExtra" lHL lXL lsicL) []).length :=
    hpc2
  have hsegR2 := hsegR1.append_right
  have hlrR2 := hlrR1.append_right
  -- Step 3: matchMid.
  obtain ⟨n3, ss3, hr3, hpc3, hc3, hmi3⟩ :=
    (simSL'_matchMid R litStart hout hop hib hls hcd hcs)
      prog _ ss2 uifSt1 fuel hpc2' hsegR2.append_left hlrR2.append_left hc2 hmi2
  have hEval3 : (wseq [ .bin .add "cpDst" "outBase" (.reg "op"),
        .bin .add "cpSrc" "inBase" (.reg litStart) ]).eval fuel uifSt1
      = matchCpEntry litStart litLen ml fuel ws := by
    rw [matchCpEntry, ← hUif1]; simp only [wseq, WStmt.eval]
  rw [hEval3] at hc3
  have hpc3' : ss3.pc = base + (matchPreEmit litLen ml).length
      + (uifEmit "pLitBig" lElseL lEndL (lsicThen litLen "litExtra" lHL lXL lsicL) []).length
      + (matchMidEmit litStart).length := hpc3
  have hsegR3 := hsegR2.append_right
  have hlrR3 := hlrR2.append_right
  -- Step 4: coopCopy.
  obtain ⟨n4, ss4, hr4, hpc4, hc4, hmi4⟩ :=
    simSL_coopCopy R "cpDst" "cpSrc" litLen cpH cpX hcd hcs hll hRcp
      (matchCpEntry litStart litLen ml fuel ws) hb1 hb2 hb3 hdisj hsz
      prog _ ss3 fuel hpc3' hsegR3.append_left hlrR3.append_left hc3 hmi3
  obtain ⟨coopSt, hCoop⟩ : ∃ st, (WStmt.coopCopy "cpDst" "cpSrc" litLen).eval fuel
      (matchCpEntry litStart litLen ml fuel ws) = st := ⟨_, rfl⟩
  rw [hCoop] at hc4
  have hpc4' : ss4.pc = base + (matchPreEmit litLen ml).length
      + (uifEmit "pLitBig" lElseL lEndL (lsicThen litLen "litExtra" lHL lXL lsicL) []).length
      + (matchMidEmit litStart).length + (matchCoopEmit litLen cpH cpX).length := by
    rw [matchCoopEmit]; exact hpc4
  have hsegR4 := hsegR3.append_right
  have hlrR4 := hlrR3.append_right
  -- Step 5: matchOff.
  obtain ⟨n5, ss5, hr5, hpc5, hc5, hmi5⟩ :=
    (simSL'_matchOff R litLen off hll hoff hmlm hoffLo hoffHi hout hop hsb hpm)
      prog _ ss4 coopSt fuel hpc4' hsegR4.append_left hlrR4.append_left hc4 hmi4
  have hEval5 : (wseq [ .bin .add "op" "op" (.reg litLen),
        .bin .band "offLo" off (.imm 255), wStoreByte "offLo",
        .bin .shr "offHi" off (.imm 8), .bin .band "offHi" "offHi" (.imm 255),
        wStoreByte "offHi", .setp .ge "pMatBig" "mlm" (.imm 15) ]).eval fuel coopSt
      = matchAfterMatSetp litStart litLen off ml fuel ws := by
    rw [matchAfterMatSetp, ← hCoop]; simp only [wseq, WStmt.eval]
  rw [hEval5] at hc5
  have hpc5' : ss5.pc = base + (matchPreEmit litLen ml).length
      + (uifEmit "pLitBig" lElseL lEndL (lsicThen litLen "litExtra" lHL lXL lsicL) []).length
      + (matchMidEmit litStart).length + (matchCoopEmit litLen cpH cpX).length
      + (matchOffEmit litLen off).length := hpc5
  have hsegR5 := hsegR4.append_right
  have hlrR5 := hlrR4.append_right
  -- Step 6: pMatBig LSIC uif.
  obtain ⟨n6, ss6, hr6, hpc6, hc6, hmi6⟩ :=
    simSL'_lsicUif R "pMatBig" "mlm" "matExtra" lElseM lEndM lHM lXM hpm hme hmlm
      ⟨by decide, by decide, by decide⟩ (by decide) (by decide) (by decide) (by decide)
      hout hop hsb hc255 hlsicC lsicM hMdef
      prog _ ss5 (matchAfterMatSetp litStart litLen off ml fuel ws) fuel hpc5'
      hsegR5 hlrR5 hc5 hmi5 hgdM hflM
  -- assemble.
  refine ⟨n1 + (n2 + (n3 + (n4 + (n5 + n6)))), ss6,
    sreaches_trans prog n1 _ _ _ _ hr1
      (sreaches_trans prog n2 _ _ _ _ hr2
        (sreaches_trans prog n3 _ _ _ _ hr3
          (sreaches_trans prog n4 _ _ _ _ hr4
            (sreaches_trans prog n5 n6 _ _ _ hr5 hr6)))), ?_, ?_, hmi6⟩
  · rw [hpc6, wEmitMatchSeqEmit]
    simp only [List.length_append, List.length_cons, List.length_nil, Nat.add_assoc]
  · rw [wEmitMatchSeq_eval_seq]
    rw [matchAfterMatSetp, matchCpEntry, matchAfterSetp] at hc6
    exact hc6

-- ── extC loop (custom induction; `simSL'_uwhile` can't consume a coop body) ───────

/-- The extend-loop emit body (coop extend step; `ml += adv`; `extC := adv==32`). -/
def foundExtBodyEmit : List SInstr :=
  coopExtendEmit "adv" ++ (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
    ++ [.setp .eq "extC" "adv" (.imm 32)])

/-- The full emit of the `uif found` true branch: found-setup movs, the extend
    `uwhile`, the two subs, the match-sequence emit, and the two updates. -/
def foundBranchEmit (endCap : Nat) (lHE lEE lElseL lEndL lHL lXL cpH cpX
    lElseM lEndM lHM lXM : String) (lsicL lsicM : List SInstr) : List SInstr :=
  ([.mov "ecR" (SArg.imm endCap), .mov "ec1" (.imm (endCap - 1)),
    .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0)] : List SInstr)
  ++ (uwhileEmit "extC" lHE lEE foundExtBodyEmit
    ++ (([.bin .sub "off0" "p0" (.reg "cand0"), .bin .sub "litLen" "p0" (.reg "litAnchor")]
        : List SInstr)
      ++ (wEmitMatchSeqEmit "litAnchor" "litLen" "off0" "ml" lElseL lEndL lHL lXL cpH cpX
            lElseM lEndM lHM lXM lsicL lsicM
        ++ ([.bin .add "litAnchor" "p0" (.reg "ml"), .mov "searchPos" (.reg "litAnchor")]
            : List SInstr))))

/-- State after the four found-setup movs (`ecR,ec1,ml,extC`). -/
def foundExtEntry (endCap : Nat) (fuel : Nat) (ws : WState) : WState :=
  (wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
          .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval fuel ws

/-- State after the extend `uwhile` (from `foundExtEntry`). -/
def foundExtDone (inStride endCap : Nat) (fuel : Nat) (ws : WState) : WState :=
  (WStmt.uwhile "extC" (extBodyStmt inStride endCap)).eval fuel (foundExtEntry endCap fuel ws)

/-- State at the `wEmitMatchSeq` entry (after movs, extend loop, and the two subs). -/
def foundMatchEntry (inStride endCap : Nat) (fuel : Nat) (ws : WState) : WState :=
  (wseq [ .bin .sub "off0" "p0" (.reg "cand0"),
          .bin .sub "litLen" "p0" (.reg "litAnchor") ]).eval fuel (foundExtDone inStride endCap fuel ws)

/-- The `uif found` true branch as a WStmt (matches `bodyEncodePrefix`'s inline form
    when `endCap = inStride - 5`). -/
def foundBranchStmt (inStride endCap : Nat) : WStmt :=
  wseq
  [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
    .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0),
    .uwhile "extC" (wseq
      [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
        .bin .add "ml" "ml" (.reg "adv"),
        .setp .eq "extC" "adv" (.imm 32) ]),
    .bin .sub "off0" "p0" (.reg "cand0"),
    .bin .sub "litLen" "p0" (.reg "litAnchor"),
    wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
    .bin .add "litAnchor" "p0" (.reg "ml"),
    .mov "searchPos" (.reg "litAnchor") ]

/-- `p0`/`cand0` survive the found-setup movs; `ml` becomes `4`. -/
theorem foundExtEntry_p0 (endCap fuel : Nat) (ws : WState) :
    (foundExtEntry endCap fuel ws).regs "p0" = ws.regs "p0" := by
  simp [foundExtEntry, wseq, WStmt.eval, WState.setReg, WOp.run, SCmp.run, WArg.eval]

theorem foundExtEntry_mlN (endCap fuel : Nat) (ws : WState) :
    ((foundExtEntry endCap fuel ws).regs "ml").toNat = 4 := by
  simp [foundExtEntry, wseq, WStmt.eval, WState.setReg, WOp.run, SCmp.run, WArg.eval]

/-- The extend `uwhile` only writes `adv`/`ml`/`extC`; every other register is
    preserved (the eval `evalCoopExtendStep` writes only `adv`, `bin add ml` writes
    `ml`, the trailing `setp` writes `extC`). -/
theorem uwhile_extBody_frame (inStride endCap : Nat) (r : String)
    (hadv : r ≠ "adv") (hml : r ≠ "ml") (hextC : r ≠ "extC") :
    ∀ (fuel : Nat) (ws : WState),
      ((WStmt.uwhile "extC" (extBodyStmt inStride endCap)).eval fuel ws).regs r = ws.regs r := by
  intro fuel
  induction fuel with
  | zero => intro ws; simp [WStmt.eval]
  | succ fuel ih =>
    intro ws
    simp only [WStmt.eval]
    by_cases hb : (ws.regs "extC" == 1) = true
    · rw [if_pos hb, ih]
      simp [extBodyStmt, wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run,
        SCmp.run, WArg.eval, hadv, hml, hextC]
    · rw [if_neg hb]

/-- `op`/`outBase`/`inBase`/`litAnchor`/`p0`/`cand0` survive the whole found-branch
    prefix (movs, extend loop, subs), so they equal their values at the branch entry
    at the `wEmitMatchSeq` entry state. -/
theorem foundMatchEntry_frame (inStride endCap fuel : Nat) (ws : WState) (r : String)
    (h1 : r ≠ "ecR") (h2 : r ≠ "ec1") (h3 : r ≠ "ml") (h4 : r ≠ "extC")
    (h5 : r ≠ "adv") (h6 : r ≠ "off0") (h7 : r ≠ "litLen") :
    (foundMatchEntry inStride endCap fuel ws).regs r = ws.regs r := by
  rw [foundMatchEntry, foundExtDone, foundExtEntry]
  simp only [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval, SCmp.run]
  rw [uwhile_extBody_frame inStride endCap r h5 h3 h4]
  simp [WState.setReg, WOp.run, WArg.eval, SCmp.run, h1, h2, h3, h4, h6, h7]

/-- The literal length `litLen = p0 - litAnchor` at the match-sequence entry (the two
    subs compute it from the `coopWindow`/frame-preserved `p0`/`litAnchor`). -/
theorem foundMatchEntry_litLen (inStride endCap fuel : Nat) (ws : WState) :
    (foundMatchEntry inStride endCap fuel ws).regs "litLen"
      = ws.regs "p0" - ws.regs "litAnchor" := by
  rw [foundMatchEntry, foundExtDone, foundExtEntry]
  simp only [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval, SCmp.run]
  rw [uwhile_extBody_frame inStride endCap "p0" (by decide) (by decide) (by decide),
      uwhile_extBody_frame inStride endCap "litAnchor" (by decide) (by decide) (by decide)]
  simp [WState.setReg, WOp.run, WArg.eval, SCmp.run]

/-- The match length `ml` at the match-sequence entry is bounded by `endCap` (the
    extend loop preserves `extInv`'s `p0 + ml ≤ endCap`), given the found-branch
    preconditions.  Discharges the `pMatBig` LSIC-fuel side-condition. -/
theorem foundMatchEntry_ml_bound (inStride endCap fuel : Nat) (ws : WState)
    (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hcand : (ws.regs "cand0").toNat < (ws.regs "p0").toNat)
    (hp4 : (ws.regs "p0").toNat + 4 ≤ endCap) :
    ((foundMatchEntry inStride endCap fuel ws).regs "ml").toNat ≤ endCap := by
  have hExtInv : extInv inStride endCap (foundExtDone inStride endCap fuel ws) := by
    rw [foundExtDone]
    exact extInv_uwhile inStride endCap hendCap hlen fuel (foundExtEntry endCap fuel ws)
      (extInv_of_foundMovs endCap inStride fuel ws hcand hp4)
  have hml : (foundMatchEntry inStride endCap fuel ws).regs "ml"
      = (foundExtDone inStride endCap fuel ws).regs "ml" := by
    rw [foundMatchEntry]
    simp [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval]
  rw [hml]
  obtain ⟨_, _, _, hpml⟩ := hExtInv
  omega

/-- The match length `ml` at the match-sequence entry is at least `MINMATCH = 4`:
    `foundExtEntry` sets `ml := 4` and the extend loop only grows it.  Discharges the
    `mlm = ml - 4` well-definedness for the `pMatBig` side-conditions. -/
theorem foundMatchEntry_ml_lb (inStride endCap fuel : Nat) (ws : WState)
    (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hcand : (ws.regs "cand0").toNat < (ws.regs "p0").toNat)
    (hp4 : (ws.regs "p0").toNat + 4 ≤ endCap) :
    4 ≤ ((foundMatchEntry inStride endCap fuel ws).regs "ml").toNat := by
  have hEntry4 : ((foundExtEntry endCap fuel ws).regs "ml").toNat = 4 := by
    rw [foundExtEntry]
    simp [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval, SCmp.run]
  have hml : (foundMatchEntry inStride endCap fuel ws).regs "ml"
      = (foundExtDone inStride endCap fuel ws).regs "ml" := by
    rw [foundMatchEntry]
    simp [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval]
  rw [hml, foundExtDone]
  exact uwhile_ml_ge4 inStride endCap hendCap hlen fuel (foundExtEntry endCap fuel ws)
    (extInv_of_foundMovs endCap inStride fuel ws hcand hp4) (by omega)

/-- The extend `uwhile` never touches `gmem` (the coop extend step writes only `adv`). -/
theorem uwhile_gmem_extBody (inStride endCap : Nat) :
    ∀ (fuel : Nat) (ws : WState),
      ((WStmt.uwhile "extC" (extBodyStmt inStride endCap)).eval fuel ws).gmem = ws.gmem := by
  intro fuel
  induction fuel with
  | zero => intro ws; simp [WStmt.eval]
  | succ fuel ih =>
      intro ws; simp only [WStmt.eval]
      by_cases hb : (ws.regs "extC" == 1) = true
      · rw [if_pos hb, ih]
        simp [extBodyStmt, wseq, WStmt.eval, evalCoopExtendStep, WState.setReg]
      · rw [if_neg hb]

/-- The whole found-branch prefix (movs, extend loop, subs) leaves `gmem` unchanged. -/
theorem foundMatchEntry_gmem (inStride endCap fuel : Nat) (ws : WState) :
    (foundMatchEntry inStride endCap fuel ws).gmem = ws.gmem := by
  rw [foundMatchEntry, foundExtDone, foundExtEntry]
  simp only [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval, SCmp.run]
  rw [uwhile_gmem_extBody inStride endCap]

/-- `encNib n` is never longer than `n`. -/
theorem encNib_len_le (n : Nat) : (LZ4.encNib n).length ≤ n := by
  rw [AlgorithmLib.LZ4Imp.encNib_length]
  split
  · omega
  · have := Nat.div_le_self (n - 15) 255; omega

/-- **The 9 `wEmitMatchSeq` copy-layout / LSIC-fuel side-conditions**, discharged from
    a `LoopCInv`-style hypothesis set on the (post-`coopWindow`) state `ws`.  This is the
    caller-obligation bundle that `simSL'_foundBranch` consumes. -/
theorem foundBranch_sideconds (inStride endCap fuel : Nat) (ws : WState)
    (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hib0 : (ws.regs "inBase").toNat < 2 ^ 40)
    (hla_p0 : (ws.regs "litAnchor").toNat ≤ (ws.regs "p0").toNat)
    (hp0_lt : (ws.regs "p0").toNat < inStride)
    (hbud32 : (ws.regs "outBase").toNat + (ws.regs "op").toNat
        + 9 * (inStride - (ws.regs "litAnchor").toNat) < 2 ^ 32)
    (hbudsz : (ws.regs "outBase").toNat + (ws.regs "op").toNat
        + 9 * (inStride - (ws.regs "litAnchor").toNat) ≤ ws.gmem.size)
    (hobLB : (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat)
    (hcand : (ws.regs "cand0").toNat < (ws.regs "p0").toNat)
    (hp4 : (ws.regs "p0").toNat + 4 ≤ endCap)
    (hfuel : inStride ≤ fuel) :
    let M := foundMatchEntry inStride endCap fuel ws
    ((matchAfterSetp "litLen" "ml" fuel M).regs "pLitBig" = 1 →
        15 ≤ ((matchAfterSetp "litLen" "ml" fuel M).regs "litLen").toNat)
    ∧ ((matchAfterSetp "litLen" "ml" fuel M).regs "pLitBig" = 1 →
        ((matchAfterSetp "litLen" "ml" fuel M).regs "litLen").toNat < 255 * fuel + 15)
    ∧ ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpDst").toNat < 2 ^ 32
    ∧ ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpSrc").toNat < 2 ^ 32
    ∧ ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "litLen").toNat < 2 ^ 32
    ∧ (((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpDst").toNat
          + ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "litLen").toNat
          ≤ ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpSrc").toNat
        ∨ ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpSrc").toNat
          + ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "litLen").toNat
          ≤ ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpDst").toNat)
    ∧ (((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpDst").toNat
          + ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "litLen").toNat
          ≤ (matchCpEntry "litAnchor" "litLen" "ml" fuel M).gmem.size)
    ∧ ((matchAfterMatSetp "litAnchor" "litLen" "off0" "ml" fuel M).regs "pMatBig" = 1 →
        15 ≤ ((matchAfterMatSetp "litAnchor" "litLen" "off0" "ml" fuel M).regs "mlm").toNat)
    ∧ ((matchAfterMatSetp "litAnchor" "litLen" "off0" "ml" fuel M).regs "pMatBig" = 1 →
        ((matchAfterMatSetp "litAnchor" "litLen" "off0" "ml" fuel M).regs "mlm").toNat
          < 255 * fuel + 15) := by
  intro M
  -- litLen = p0 - litAnchor at M; frame op/outBase/inBase/litAnchor to ws; ml ≤ endCap.
  have hLL : (M.regs "litLen").toNat = (ws.regs "p0").toNat - (ws.regs "litAnchor").toNat := by
    show ((foundMatchEntry inStride endCap fuel ws).regs "litLen").toNat = _
    rw [foundMatchEntry_litLen, UInt64.toNat_sub,
      show 2 ^ 64 - (ws.regs "litAnchor").toNat + (ws.regs "p0").toNat
        = 2 ^ 64 + ((ws.regs "p0").toNat - (ws.regs "litAnchor").toNat) from by
        have := (ws.regs "p0").toNat_lt; omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by have := (ws.regs "p0").toNat_lt; omega)]
  have hLLlt : (M.regs "litLen").toNat < inStride := by rw [hLL]; omega
  have hop : M.regs "op" = ws.regs "op" :=
    foundMatchEntry_frame inStride endCap fuel ws "op" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
  have hob : M.regs "outBase" = ws.regs "outBase" :=
    foundMatchEntry_frame inStride endCap fuel ws "outBase" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
  have hib : M.regs "inBase" = ws.regs "inBase" :=
    foundMatchEntry_frame inStride endCap fuel ws "inBase" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
  have hla : M.regs "litAnchor" = ws.regs "litAnchor" :=
    foundMatchEntry_frame inStride endCap fuel ws "litAnchor" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
  have hgm : M.gmem = ws.gmem := foundMatchEntry_gmem inStride endCap fuel ws
  have hmlle : (M.regs "ml").toNat ≤ endCap :=
    foundMatchEntry_ml_bound inStride endCap fuel ws hendCap hlen hcand hp4
  have hml4 : 4 ≤ (M.regs "ml").toNat :=
    foundMatchEntry_ml_lb inStride endCap fuel ws hendCap hlen hcand hp4
  -- the LSIC fuel bound: (litLen - 15)/255 < fuel.
  have hlfuel : ((M.regs "litLen").toNat - 15) / 255 < fuel := by
    have h := Nat.div_le_self ((M.regs "litLen").toNat - 15) 255
    omega
  -- litLen value at matchAfterSetp / matchCpEntry (framed) and the cpSrc/cpDst values.
  have hM_ll : (matchAfterSetp "litLen" "ml" fuel M).regs "litLen" = M.regs "litLen" :=
    matchAfterSetp_frame "litLen" "ml" "litLen" fuel M (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
  have hM_pb := matchAfterSetp_pLitBig "litLen" "ml" fuel M (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide)
  have hC_ll : (matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "litLen" = M.regs "litLen" :=
    matchCpEntry_frame "litAnchor" "litLen" "ml" "litLen" fuel M (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide)
  have hC_cs : (matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpSrc"
      = M.regs "inBase" + M.regs "litAnchor" :=
    matchCpEntry_cpSrc "litAnchor" "litLen" "ml" fuel M (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have hC_cd : (matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpDst"
      = M.regs "outBase" + M.regs "op" + 1 + UInt64.ofNat (LZ4.encNib (M.regs "litLen").toNat).length :=
    matchCpEntry_cpDst "litAnchor" "litLen" "ml" fuel M (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) hlfuel
  have hC_gm_size : (matchCpEntry "litAnchor" "litLen" "ml" fuel M).gmem.size = M.gmem.size := by
    rw [matchCpEntry, eval_gmem_size, eval_gmem_size, eval_gmem_size, matchAfterSetp,
      eval_gmem_size, eval_gmem_size, eval_gmem_size, eval_gmem_size]
  -- toNat of cpSrc and cpDst.
  have hcsN : ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpSrc").toNat
      = (ws.regs "inBase").toNat + (ws.regs "litAnchor").toNat := by
    rw [hC_cs, hib, hla, UInt64.toNat_add, Nat.mod_eq_of_lt (by omega)]
  have hencle : (LZ4.encNib (M.regs "litLen").toNat).length ≤ (M.regs "litLen").toNat :=
    encNib_len_le _
  have hencle' : (LZ4.encNib (M.regs "litLen").toNat).length
      ≤ (ws.regs "p0").toNat - (ws.regs "litAnchor").toNat := by omega
  have hencN : (LZ4.encNib (M.regs "litLen").toNat).length < 2 ^ 64 := by omega
  have hnoof : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
      + (LZ4.encNib (M.regs "litLen").toNat).length < 2 ^ 64 := by omega
  have s_op1 : (ws.regs "op" + 1).toNat = (ws.regs "op").toNat + 1 := by
    rw [UInt64.toNat_add, show (1 : UInt64).toNat = 1 from by decide,
      Nat.mod_eq_of_lt (by omega)]
  have s_op1e : (ws.regs "op" + 1
        + UInt64.ofNat (LZ4.encNib (M.regs "litLen").toNat).length).toNat
      = (ws.regs "op").toNat + 1 + (LZ4.encNib (M.regs "litLen").toNat).length := by
    rw [UInt64.toNat_add, s_op1, toNat_ofNat_lt _ hencN, Nat.mod_eq_of_lt (by omega)]
  have hcdN : ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpDst").toNat
      = (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
        + (LZ4.encNib (M.regs "litLen").toNat).length := by
    rw [hC_cd, hob, hop,
      show ws.regs "outBase" + ws.regs "op" + 1
          + UInt64.ofNat (LZ4.encNib (M.regs "litLen").toNat).length
        = ws.regs "outBase" + (ws.regs "op" + 1
          + UInt64.ofNat (LZ4.encNib (M.regs "litLen").toNat).length) from by ac_rfl,
      UInt64.toNat_add, s_op1e, Nat.mod_eq_of_lt (by omega)]
    omega
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro hpb; rw [hM_ll, hLL]
    rw [hM_pb] at hpb
    by_cases hg : UInt64.ofNat 15 ≤ M.regs "litLen"
    · rw [UInt64.le_iff_toNat_le, show ((UInt64.ofNat 15).toNat) = 15 from by decide] at hg
      rw [hLL] at hg; omega
    · rw [if_neg hg] at hpb; exact absurd hpb (by decide)
  · intro _; rw [hM_ll]; omega
  · rw [hcdN]; omega
  · rw [hcsN]; omega
  · rw [hC_ll]; omega
  · right; rw [hcdN, hcsN, hC_ll, hLL]; omega
  · rw [show (matchCpEntry "litAnchor" "litLen" "ml" fuel M).gmem.size = ws.gmem.size from by
        rw [hC_gm_size, hgm], hcdN, hC_ll]
    have := hencle; have := hLL; omega
  · intro hpm
    have hmlm := matchAfterMatSetp_mlm "litAnchor" "litLen" "off0" "ml" fuel M (by decide)
    have hpmb := matchAfterMatSetp_pMatBig "litAnchor" "litLen" "off0" "ml" fuel M (by decide)
    rw [hmlm]
    rw [hpmb] at hpm
    by_cases hg : UInt64.ofNat 15 ≤ M.regs "ml" - UInt64.ofNat 4
    · rw [UInt64.le_iff_toNat_le, show ((UInt64.ofNat 15).toNat) = 15 from by decide] at hg; exact hg
    · rw [if_neg hg] at hpm; exact absurd hpm (by decide)
  · intro _
    have hmlm := matchAfterMatSetp_mlm "litAnchor" "litLen" "off0" "ml" fuel M (by decide)
    rw [hmlm, UInt64.toNat_sub, show ((UInt64.ofNat 4).toNat) = 4 from by decide,
      show 2 ^ 64 - 4 + (M.regs "ml").toNat = 2 ^ 64 + ((M.regs "ml").toNat - 4) from by omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by have := (M.regs "ml").toNat_lt; omega)]
    omega

/-- **The found-branch simulates.**  Bespoke five-segment glue: found-setup movs
    (`simSL'_foundMovs`, establishing `extInv` via `extInv_of_foundMovs`), the extend
    `uwhile` (`extLoop_sim`), the two address subs (`simSL'_foundSubs`), the
    match-sequence emit (`simSL'_wEmitMatchSeq`), and the two updates
    (`simSL'_foundUpdates`).  The caller discharges the `extInv`/extend-fuel bounds and
    the `wEmitMatchSeq` copy-layout / LSIC-fuel side-conditions. -/
theorem simSL'_foundBranch (R : List String) (inStride endCap : Nat)
    (lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM : String)
    (hctx : ExtCtx R) (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hib40 : ib < 2 ^ 40)
    (hextC : "extC" ∈ R)
    (hRdisj : ∀ r ∈ R, r ∉ ["idx", "pe", "pIn", "peC", "dfe", "caC", "peD", "aP", "caD", "aC",
               "bP", "bC", "pEqB", "pOk", "balOk", "mis", "revM"])
    (hoff0 : "off0" ∈ R) (hlitLen : "litLen" ∈ R) (hla : "litAnchor" ∈ R) (hsp : "searchPos" ∈ R)
    (hmlm : "mlm" ∈ R) (htokLo : "tokLo" ∈ R) (htokHi : "tokHi" ∈ R) (htok : "tok" ∈ R)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R)
    (hpb : "pLitBig" ∈ R) (hpm : "pMatBig" ∈ R)
    (hle : "litExtra" ∈ R) (hme : "matExtra" ∈ R) (hc255 : "c255" ∈ R) (hlsicC : "lsicC" ∈ R)
    (hoffLo : "offLo" ∈ R) (hoffHi : "offHi" ∈ R)
    (hcd : "cpDst" ∈ R) (hcs : "cpSrc" ∈ R)
    (hRcp : ∀ r ∈ R, r ∉ coopCopyScratch)
    (lsicL lsicM : List SInstr)
    (hLdef : lsicL =
      ([.mov "c255" (SArg.imm 255)]
        ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
            ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
          ++ ([.bin .sub "litExtra" "litExtra" (.imm 255)]
            ++ [.setp .ge "lsicC" "litExtra" (.imm 255)]))))
    (hMdef : lsicM =
      ([.mov "c255" (SArg.imm 255)]
        ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
            ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
          ++ ([.bin .sub "matExtra" "matExtra" (.imm 255)]
            ++ [.setp .ge "lsicC" "matExtra" (.imm 255)])))) :
    ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (ws : WState) (fuel : Nat),
      ss.pc = base →
      SegAt prog base (foundBranchEmit endCap lHE lEE lElseL lEndL lHL lXL cpH cpX
        lElseM lEndM lHM lXM lsicL lsicM) →
      LabelsResolve prog base (foundBranchEmit endCap lHE lEE lElseL lEndL lHL lXL cpH cpX
        lElseM lEndM lHM lXM lsicL lsicM) →
      Couple R ss ws → MachInv ib ss →
      (ws.regs "cand0").toNat < (ws.regs "p0").toNat →
      (ws.regs "p0").toNat + 4 ≤ endCap →
      endCap + 32 < 32 * fuel + ((ws.regs "p0").toNat + 4) →
      -- 1st LSIC (pLitBig on litLen), on `foundMatchEntry`:
      ((matchAfterSetp "litLen" "ml" fuel (foundMatchEntry inStride endCap fuel ws)).regs "pLitBig"
          = 1 → 15 ≤ ((matchAfterSetp "litLen" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).regs "litLen").toNat) →
      ((matchAfterSetp "litLen" "ml" fuel (foundMatchEntry inStride endCap fuel ws)).regs "pLitBig"
          = 1 → ((matchAfterSetp "litLen" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).regs "litLen").toNat < 255 * fuel + 15) →
      -- coopCopy layout, on `matchCpEntry`:
      ((matchCpEntry "litAnchor" "litLen" "ml" fuel
          (foundMatchEntry inStride endCap fuel ws)).regs "cpDst").toNat < 2 ^ 32 →
      ((matchCpEntry "litAnchor" "litLen" "ml" fuel
          (foundMatchEntry inStride endCap fuel ws)).regs "cpSrc").toNat < 2 ^ 32 →
      ((matchCpEntry "litAnchor" "litLen" "ml" fuel
          (foundMatchEntry inStride endCap fuel ws)).regs "litLen").toNat < 2 ^ 32 →
      (((matchCpEntry "litAnchor" "litLen" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).regs "cpDst").toNat
          + ((matchCpEntry "litAnchor" "litLen" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).regs "litLen").toNat
          ≤ ((matchCpEntry "litAnchor" "litLen" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).regs "cpSrc").toNat
        ∨ ((matchCpEntry "litAnchor" "litLen" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).regs "cpSrc").toNat
          + ((matchCpEntry "litAnchor" "litLen" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).regs "litLen").toNat
          ≤ ((matchCpEntry "litAnchor" "litLen" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).regs "cpDst").toNat) →
      (((matchCpEntry "litAnchor" "litLen" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).regs "cpDst").toNat
          + ((matchCpEntry "litAnchor" "litLen" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).regs "litLen").toNat
          ≤ (matchCpEntry "litAnchor" "litLen" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).gmem.size) →
      -- 2nd LSIC (pMatBig on mlm), on `matchAfterMatSetp`:
      ((matchAfterMatSetp "litAnchor" "litLen" "off0" "ml" fuel
          (foundMatchEntry inStride endCap fuel ws)).regs "pMatBig" = 1 →
          15 ≤ ((matchAfterMatSetp "litAnchor" "litLen" "off0" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).regs "mlm").toNat) →
      ((matchAfterMatSetp "litAnchor" "litLen" "off0" "ml" fuel
          (foundMatchEntry inStride endCap fuel ws)).regs "pMatBig" = 1 →
          ((matchAfterMatSetp "litAnchor" "litLen" "off0" "ml" fuel
            (foundMatchEntry inStride endCap fuel ws)).regs "mlm").toNat < 255 * fuel + 15) →
      ∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
        ss'.pc = base + (foundBranchEmit endCap lHE lEE lElseL lEndL lHL lXL cpH cpX
          lElseM lEndM lHM lXM lsicL lsicM).length ∧
        Couple R ss' ((foundBranchStmt inStride endCap).eval fuel ws) ∧ MachInv ib ss' := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi hcand hp4 hfuelExt
    hgdL hflL hb1 hb2 hb3 hdisj hsz hgdM hflM
  rw [foundBranchEmit] at hseg hlr
  -- Segment decomposition of the emit.
  have hsegMovs := hseg.append_left
  have hlrMovs := hlr.append_left
  have hsegRest := hseg.append_right
  have hlrRest := hlr.append_right
  -- Step 1: found-setup movs.
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1⟩ :=
    (simSL'_foundMovs R endCap hctx.ecR hctx.ec1 hctx.ml hextC)
      prog base ss ws fuel hpc hsegMovs hlrMovs hc hmi
  have hc1' : Couple R ss1 (foundExtEntry endCap fuel ws) := hc1
  have hpc1' : ss1.pc = base + 4 := hpc1
  have hextInv : extInv inStride endCap (foundExtEntry endCap fuel ws) :=
    extInv_of_foundMovs endCap inStride fuel ws hcand (by omega)
  -- extend-fuel bound, restated on `foundExtEntry`.
  have hfuelExt' : endCap + 32 < 32 * fuel + (((foundExtEntry endCap fuel ws).regs "p0").toNat
      + ((foundExtEntry endCap fuel ws).regs "ml").toNat) := by
    rw [foundExtEntry_p0, foundExtEntry_mlN]; omega
  -- Step 2: the extend `uwhile`.
  have hsegExt : SegAt prog (base + 4)
      (uwhileEmit "extC" lHE lEE foundExtBodyEmit) := hsegRest.append_left
  have hlrExt : LabelsResolve prog (base + 4)
      (uwhileEmit "extC" lHE lEE foundExtBodyEmit) := hlrRest.append_left
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2⟩ :=
    extLoop_sim R inStride endCap hctx hendCap hlen hib40 hctx.ml hctx.adv hextC hRdisj lHE lEE
      prog (base + 4) hsegExt hlrExt fuel ss1 (foundExtEntry endCap fuel ws)
      hpc1' hc1' hmi1 hextInv hfuelExt'
  have hc2' : Couple R ss2 (foundExtDone inStride endCap fuel ws) := hc2
  have hpc2' : ss2.pc = base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length := hpc2
  -- Step 3: the two address subs.
  have hsegAfterExt := hsegRest.append_right
  have hlrAfterExt := hlrRest.append_right
  obtain ⟨n3, ss3, hr3, hpc3, hc3, hmi3⟩ :=
    (simSL'_foundSubs R "litAnchor" hctx.p0 hctx.cand0 hla hoff0 hlitLen)
      prog _ ss2 (foundExtDone inStride endCap fuel ws) fuel hpc2'
      hsegAfterExt.append_left hlrAfterExt.append_left hc2' hmi2
  have hc3' : Couple R ss3 (foundMatchEntry inStride endCap fuel ws) := hc3
  have hpc3' : ss3.pc = base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2 := hpc3
  -- Step 4: the match-sequence emit.
  have hsegAfterSubs := hsegAfterExt.append_right
  have hlrAfterSubs := hlrAfterExt.append_right
  obtain ⟨n4, ss4, hr4, hpc4, hc4, hmi4⟩ :=
    simSL'_wEmitMatchSeq R "litAnchor" "litLen" "off0" "ml"
      lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM
      hlitLen hla hoff0 hctx.ml hmlm htokLo htokHi htok hout hop hctx.ib hsb hpb hpm
      hle hme hc255 hlsicC hoffLo hoffHi hcd hcs hRcp lsicL lsicM hLdef hMdef
      prog _ ss3 (foundMatchEntry inStride endCap fuel ws) fuel hpc3'
      hsegAfterSubs.append_left hlrAfterSubs.append_left hc3' hmi3
      hgdL hflL hb1 hb2 hb3 hdisj hsz hgdM hflM
  obtain ⟨matchSt, hMatch⟩ : ∃ st, (wEmitMatchSeq "litAnchor" "litLen" "off0" "ml").eval fuel
      (foundMatchEntry inStride endCap fuel ws) = st := ⟨_, rfl⟩
  rw [hMatch] at hc4
  have hpc4' : ss4.pc = base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2
      + (wEmitMatchSeqEmit "litAnchor" "litLen" "off0" "ml" lElseL lEndL lHL lXL cpH cpX
          lElseM lEndM lHM lXM lsicL lsicM).length := hpc4
  -- Step 5: the two updates.
  have hsegAfterMatch := hsegAfterSubs.append_right
  have hlrAfterMatch := hlrAfterSubs.append_right
  obtain ⟨n5, ss5, hr5, hpc5, hc5, hmi5⟩ :=
    (simSL'_foundUpdates R hctx.p0 hctx.ml hla hsp)
      prog _ ss4 matchSt fuel hpc4' hsegAfterMatch hlrAfterMatch hc4 hmi4
  -- assemble.
  refine ⟨n1 + (n2 + (n3 + (n4 + n5))), ss5,
    sreaches_trans prog n1 _ _ _ _ hr1
      (sreaches_trans prog n2 _ _ _ _ hr2
        (sreaches_trans prog n3 _ _ _ _ hr3
          (sreaches_trans prog n4 n5 _ _ _ hr4 hr5))), ?_, ?_, hmi5⟩
  · rw [hpc5, foundBranchEmit]
    simp only [List.length_append, List.length_cons, List.length_nil]; omega
  · -- eval assembly: `hc5` couples to `(updates).eval fuel matchSt`; the goal unfolds
    -- to the same nested chain.
    have hgoalEq : (foundBranchStmt inStride endCap).eval fuel ws
        = (wseq [ .bin .add "litAnchor" "p0" (.reg "ml"),
                  .mov "searchPos" (.reg "litAnchor") ]).eval fuel matchSt := by
      rw [← hMatch]
      simp only [foundBranchStmt, foundMatchEntry, foundExtDone, foundExtEntry,
        extBodyStmt, wseq, WStmt.eval]
    rw [hgoalEq]; exact hc5

-- ── loopC body: `coopWindow ; uif found ; setp loopC` ───────────────────────────

/-- The `loopC` loop body (matches `bodyEncodePrefix`'s inline body when
    `endCap = inStride - 5`). -/
def loopCBodyStmt (inStride hashLog : Nat) : WStmt :=
  wseq
  [ .coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0,
    .uif "found" (foundBranchStmt inStride (inStride - 5))
      (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]),
    .setp .lt "loopC" "searchPos" (.imm (inStride - 12)) ]

/-- The loop invariant threaded through the `loopC` loop by `simSL'_measureLoop`.
    Same layout facts as `LoopCInv` but WITHOUT `searchPos ≤ inStride` — that bound
    is not preserved by a not-found step (`searchPos += 32` can pass `inStride`); the
    loop body instead recovers `searchPos < searchLim` from the loop guard. -/
def LoopCQ (inStride : Nat) (ws : WState) : Prop :=
  (ws.regs "inBase").toNat < 2 ^ 40
  ∧ (ws.regs "litAnchor").toNat ≤ (ws.regs "searchPos").toNat
  ∧ (ws.regs "loopC")
      = (if UInt64.ofNat (ws.regs "searchPos").toNat < UInt64.ofNat (inStride - 12) then 1 else 0)
  ∧ (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat
  ∧ (ws.regs "outBase").toNat + (ws.regs "op").toNat
      + 9 * (inStride - (ws.regs "litAnchor").toNat) < 2 ^ 32
  ∧ (ws.regs "outBase").toNat + (ws.regs "op").toNat
      + 9 * (inStride - (ws.regs "litAnchor").toNat) ≤ ws.gmem.size

/-- From `LoopCQ` and the loop guard (`loopC = 1`), the cursor is below the limit. -/
theorem loopCQ_guard (inStride : Nat) (ws : WState) (hstride : inStride ≤ 65536)
    (hipos : 12 ≤ inStride) (hinv : LoopCQ inStride ws) (hloopC : (ws.regs "loopC" == 1) = true) :
    (ws.regs "searchPos").toNat < inStride - 12 := by
  obtain ⟨_, _, hlc, _⟩ := hinv
  rw [beq_iff_eq, hlc] at hloopC
  by_cases h : UInt64.ofNat (ws.regs "searchPos").toNat < UInt64.ofNat (inStride - 12)
  · rw [UInt64.lt_iff_toNat_lt, toNat_ofNat_lt _ (by have := (ws.regs "searchPos").toNat_lt; omega),
      toNat_ofNat_lt _ (by omega)] at h
    exact h
  · rw [if_neg h] at hloopC; exact absurd hloopC (by decide)

/-- The full emit of one `loopC` iteration: the window collective, the found/not-found
    `uif`, and the trailing `loopC` guard recompute. -/
def loopCBodyEmit (inStride hashLog : Nat) (lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX
    lElseM lEndM lHM lXM : String) (lsicL lsicM : List SInstr) : List SInstr :=
  coopWindowEmit "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog
  ++ (uifEmit "found" lElse lEnd
        (foundBranchEmit (inStride - 5) lHE lEE lElseL lEndL lHL lXL cpH cpX
          lElseM lEndM lHM lXM lsicL lsicM)
        ([.bin .add "searchPos" "searchPos" (.imm 32)] : List SInstr)
      ++ ([.setp .lt "loopC" "searchPos" (.imm (inStride - 12))] : List SInstr))

end AlgorithmLib.LZ4WarpDSL
