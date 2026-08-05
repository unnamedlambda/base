import AlgorithmLib.SimSLAssembly

/-!
  # Carrying a per-step obligation across a simulated segment

  `Lz4Sites.RegConfined` is a property of *every* step; the simulation assembly
  speaks of segments.  `LZ4Confine.sreaches_mid` says those are compatible —
  `SReaches` is deterministic iteration, so a segment's endpoint statement pins
  down its whole trace — and `AllSteps` is the predicate that carries an
  obligation across a segment boundary.

  This file is the descent itself: for each combinator the LZ4 body is assembled
  from, a variant whose conclusion also discharges `AllSteps`.  Nothing in
  `SimSLAssembly` changes; these sit alongside.

  ## Why the obligation is a machine predicate

  `QQ : SState → Prop` takes no `WState` parameter, so the combinators stay
  generic and none of them has to mention the eval.  That works because the
  *leaf* is where the coupling is used and the leaf is trivial: at a store
  instruction the leaf lemma already holds `Couple R ss ws`, and `Couple` gives
  `ss.regs "op" l = ws.regs "op"` directly.  So the machine-side bound is one
  rewrite from the eval-side bound (`LZ4OpBound.emitLoop_head_op_le_of_final`),
  and everything between here and there is plumbing rather than mathematics.
-/

namespace AlgorithmLib.LZ4WarpDSL

open AlgorithmLib.LZ4Simt AlgorithmLib.LZ4WarpFind
open CoopCopyModel
open AlgorithmLib.LZ4Ptx (toNat_ofNat_lt u64_add_ofNat u64_sub_ofNat)

/-- An instruction whose only control-flow successor is the next pc. -/
def IsStraight : SInstr → Bool
  | .braif _ _ => false
  | .braifnot _ _ => false
  | .bra _ => false
  | .ret => false
  | _ => true

theorem sstep_pc_succ_of_straight (p : Array SInstr) (st : SState) (i : SInstr)
    (hi : p[st.pc]? = some i) (hs : IsStraight i = true) : (sstep p st).pc = st.pc + 1 := by
  have hm := sstep_pc_mem_succs p st
  simp only [succsOf, hi] at hm
  cases i
  all_goals (first
    | exact List.mem_singleton.mp hm
    | simp [IsStraight] at hs)

/-- **A branch-free segment walks its own pcs.**  `SegAt` pins each instruction and
    each one advances the pc by exactly one, so after `j` steps the machine is at
    `base + j`.  This is what makes the straight-line black boxes of the body
    (`coopExtendStep`, `coopWindow`) cost nothing in the descent: their emit lists
    are branch-free, decidably so. -/
theorem siter_pc_straight {p : Array SInstr} {emit : List SInstr} {base : Nat} {ss : SState}
    (hseg : SegAt p base emit) (hpc : ss.pc = base)
    (hstr : ∀ i ∈ emit, IsStraight i = true) :
    ∀ j, j ≤ emit.length → (siter p j ss).pc = base + j := by
  intro j
  induction j with
  | zero => intro _; simpa using hpc
  | succ k ih =>
    intro hk
    have hkl : k < emit.length := by omega
    have hpk := ih (by omega)
    have hi : p[(siter p k ss).pc]? = some emit[k] := by
      rw [hpk]; exact segAt_get hseg k _ (List.getElem?_eq_getElem hkl)
    have hstep := sstep_pc_succ_of_straight p (siter p k ss) emit[k] hi
      (hstr _ (List.getElem_mem hkl))
    rw [siter_succ, hstep, hpk]
    try omega

/-- The five program points a `uwhile` emits around its body: the head label, the
    guard branch, the back branch, the end label, and the point after it.  A
    pc-conditioned obligation is discharged at these for free when none of them
    is a guarded pc — which is the case for LZ4, whose memory sites all lie
    strictly inside a body. -/
def UwhileScaffold (base bodyLen : Nat) (q : Nat) : Prop :=
  q = base ∨ q = base + 1 ∨ q = base + 1 + 1 + bodyLen
  ∨ q = base + 1 + 1 + bodyLen + 1 ∨ q = base + 1 + 1 + bodyLen + 1 + 1

/-- **`simSL'_measureLoop`, discharging a per-step obligation as well.**

    Same statement, same hypotheses, plus: the body discharges `QQ` at every one
    of its own steps, and `QQ` holds at the loop's scaffolding points.  The
    conclusion adds `AllSteps prog QQ n ss` for the `n` the loop actually takes.

    The induction is the original's; the only new work is `allSteps_seq` over the
    same five-part decomposition the `SReaches` chain already uses, so the two
    stay in step by construction. -/
theorem simSL'_measureLoopSteps (ib : Nat) (R : List String) (cond lHead lEnd : String)
    (QQ : SState → Prop) (Q : WState → Prop) (μ : WState → Nat) (F : Nat)
    (bodyStmt : WStmt) (bodyEmit : List SInstr) (prog : Array SInstr) (base : Nat)
    (hcond : cond ∈ R)
    -- `QQ` at the loop's scaffolding points, GIVEN the coupling that actually holds
    -- there.  Unconditioned would be too strong: the obligations this is for —
    -- "at this pc there is a `ws` this state couples to" — are only true of
    -- reachable states, and the loop proof has the witness in scope at each of
    -- the five points.
    (hscaf : ∀ (st : SState) (w : WState), UwhileScaffold base bodyEmit.length st.pc →
      Couple R st w → MachInv ib st → Q w → QQ st)
    (hbody : ∀ (ss : SState) (ws : WState) (fuel : Nat),
      ss.pc = base + 1 + 1 → SegAt prog (base + 1 + 1) bodyEmit →
      LabelsResolve prog (base + 1 + 1) bodyEmit →
      Couple R ss ws → MachInv ib ss → Q ws → (ws.regs cond == 1) = true → F ≤ μ ws + fuel + 1 →
      (∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
        ss'.pc = base + 1 + 1 + bodyEmit.length ∧
        Couple R ss' (bodyStmt.eval fuel ws) ∧ MachInv ib ss' ∧ AllSteps prog QQ m ss)
      ∧ Q (bodyStmt.eval fuel ws) ∧ μ ws + 1 ≤ μ (bodyStmt.eval fuel ws))
    (hseg : SegAt prog base (uwhileEmit cond lHead lEnd bodyEmit))
    (hlr : LabelsResolve prog base (uwhileEmit cond lHead lEnd bodyEmit)) :
    ∀ (fuel : Nat) (ss : SState) (ws : WState),
      ss.pc = base → Couple R ss ws → MachInv ib ss → Q ws → F ≤ μ ws + fuel →
      WhileHalts cond bodyStmt fuel ws →
      ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
        ss'.pc = base + (uwhileEmit cond lHead lEnd bodyEmit).length ∧
        Couple R ss' (WStmt.eval fuel (.uwhile cond bodyStmt) ws) ∧ MachInv ib ss' ∧
        AllSteps prog QQ n ss := by
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
    -- `QQ` at each scaffolding state, from its pc
    have qA : QQ ss := hscaf ss ws (by rw [hpc]; exact Or.inl rfl) hc hmi hQ
    have qB : QQ (ss.setPc (base + 1)) := hscaf _ ws (Or.inr (Or.inl rfl))
      (couple_setPc hc _) (machInv_setPc ss _ hmi) hQ
    by_cases hb : (ws.regs cond == 1) = true
    · have sB0 : sstep prog (ss.setPc (base + 1)) = ss.setPc (base + 1 + 1) := by
        rw [braifnot_step prog _ cond lEnd hbrn']
        simp only [SState.setPc]
        rw [show ss.regs cond 0 = ws.regs cond from hcv, if_pos hb]
      obtain ⟨⟨nb, ss1, hrB, hpcB, hcB, hmiB, hstB⟩, hQnext, hμnext⟩ :=
        hbody (ss.setPc (base + 1 + 1)) ws fuel rfl hsegBody hlrBody
          (couple_setPc hc _) (machInv_setPc ss _ hmi) hQ hb (by omega)
      have hbra' : prog[ss1.pc]? = some (.bra lHead) := by rw [hpcB]; exact hbra
      have sBk : sstep prog ss1 = ss1.setPc base := by rw [bra_step prog ss1 lHead hbra', hLhead]
      have hHrec : WhileHalts cond bodyStmt fuel (bodyStmt.eval fuel ws) := by
        rw [WhileHalts] at hH; rw [if_pos hb] at hH; exact hH
      obtain ⟨nr, ssf, hrR, hpcR, hcR, hmiR, hstR⟩ :=
        ih (ss1.setPc base) (bodyStmt.eval fuel ws) rfl (couple_setPc hcB _)
          (machInv_setPc ss1 _ hmiB) hQnext (by omega) hHrec
      have qC : QQ ss1 := hscaf ss1 (bodyStmt.eval fuel ws) (Or.inr (Or.inr (Or.inl hpcB)))
        hcB hmiB hQnext
      have qD : QQ (ss1.setPc base) := hscaf _ (bodyStmt.eval fuel ws) (Or.inl rfl)
        (couple_setPc hcB _) (machInv_setPc ss1 _ hmiB) hQnext
      refine ⟨1 + 1 + nb + 1 + nr, ssf,
        sreaches_trans prog (1 + 1 + nb + 1) nr _ _ _
          (sreaches_trans prog (1 + 1 + nb) 1 _ _ _
            (sreaches_trans prog (1 + 1) nb _ _ _
              (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
              hrB)
            (sreaches_one_eq sBk))
          hrR, hpcR, ?_, hmiR, ?_⟩
      · have heval : WStmt.eval (fuel + 1) (.uwhile cond bodyStmt) ws
            = WStmt.eval fuel (.uwhile cond bodyStmt) (bodyStmt.eval fuel ws) := by
          simp [WStmt.eval, hb]
        rw [heval]; exact hcR
      · -- the same five-part split, at the level of the obligation
        refine allSteps_seq (n₁ := 1 + 1 + nb + 1) (n₂ := nr) ?_
          (sreaches_trans prog (1 + 1 + nb) 1 _ _ _
            (sreaches_trans prog (1 + 1) nb _ _ _
              (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
              hrB)
            (sreaches_one_eq sBk)) hstR
        refine allSteps_seq (n₁ := 1 + 1 + nb) (n₂ := 1) ?_
          (sreaches_trans prog (1 + 1) nb _ _ _
            (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
            hrB)
          (allSteps_one qC (by rw [sBk]; exact qD))
        refine allSteps_seq (n₁ := 1 + 1) (n₂ := nb) ?_
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0)) hstB
        exact allSteps_seq (n₁ := 1) (n₂ := 1)
          (allSteps_one qA (by rw [sH]; exact qB))
          (sreaches_one_eq sH)
          (allSteps_one qB (by rw [sB0]; exact hstB 0 (Nat.zero_le _)))
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
      have qE : QQ (ss.setPc (base + 1 + 1 + bodyEmit.length + 1)) :=
        hscaf _ ws (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
          (couple_setPc hc _) (machInv_setPc ss _ hmi) hQ
      have qF : QQ (ss.setPc (base + 1 + 1 + bodyEmit.length + 1 + 1)) :=
        hscaf _ ws (Or.inr (Or.inr (Or.inr (Or.inr rfl))))
          (couple_setPc hc _) (machInv_setPc ss _ hmi) hQ
      refine ⟨1 + 1 + 1, ss.setPc (base + 1 + 1 + bodyEmit.length + 1 + 1),
        sreaches_trans prog (1 + 1) 1 _ _ _
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
          (sreaches_one_eq sE), ?_, ?_, machInv_setPc ss _ hmi, ?_⟩
      · show (base + 1 + 1 + bodyEmit.length + 1 + 1)
          = base + (uwhileEmit cond lHead lEnd bodyEmit).length
        rw [uwhileEmit_length]; omega
      · have heval : WStmt.eval (fuel + 1) (.uwhile cond bodyStmt) ws = ws := by
          simp [WStmt.eval, hb]
        rw [heval]; exact couple_setPc hc _
      · refine allSteps_seq (n₁ := 1 + 1) (n₂ := 1) ?_
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
          (allSteps_one qE (by rw [sE]; exact qF))
        exact allSteps_seq (n₁ := 1) (n₂ := 1)
          (allSteps_one qA (by rw [sH]; exact qB))
          (sreaches_one_eq sH)
          (allSteps_one qB (by rw [sB0]; exact qE))

-- ── The generic layer ────────────────────────────────────────────────────────

/-- `SimSL'` plus a per-step obligation.  Same shape, one extra conjunct, so a
    `SimSLQ` can be used anywhere a `SimSL'` is expected via `SimSLQ.toSimSL'`
    and the combinators below mirror `simSL'_seq`/`simSL'_single` line for line. -/
def SimSLQ (ib : Nat) (R : List String) (QQ : Nat → Nat → SState → Prop)
    (stmt : WStmt) (emit : List SInstr) : Prop :=
  ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (ws : WState) (fuel : Nat),
    ss.pc = base → SegAt prog base emit → LabelsResolve prog base emit →
    Couple R ss ws → MachInv ib ss →
    ∃ (n : Nat) (ss' : SState),
      SReaches prog n ss ss' ∧ ss'.pc = base + emit.length ∧
      Couple R ss' (stmt.eval fuel ws) ∧ MachInv ib ss' ∧
      AllSteps prog (QQ base (base + emit.length)) n ss

theorem SimSLQ.toSimSL' {ib R QQ stmt emit} (h : SimSLQ ib R QQ stmt emit) :
    SimSL' ib R stmt emit := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi
  obtain ⟨n, ss', hr, hpc', hc', hmi', -⟩ := h prog base ss ws fuel hpc hseg hlr hc hmi
  exact ⟨n, ss', hr, hpc', hc', hmi'⟩

/-- Sequencing, with the obligation composed by `allSteps_seq`.  Note the second
    segment's start state is *determined* by the first's — `SReaches` is
    iteration — so no coupling is needed at the seam. -/
theorem simSLQ_seq (ib : Nat) (R : List String) (QQ : Nat → Nat → SState → Prop)
    (hmono : ∀ (lo mid hi : Nat) (st : SState), lo ≤ mid → mid ≤ hi →
      (QQ lo mid st → QQ lo hi st) ∧ (QQ mid hi st → QQ lo hi st))
    (a b : WStmt)
    (ea eb : List SInstr) (sa : SimSLQ ib R QQ a ea) (sb : SimSLQ ib R QQ b eb) :
    SimSLQ ib R QQ (.seq a b) (ea ++ eb) := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1, hst1⟩ :=
    sa prog base ss ws fuel hpc hseg.append_left hlr.append_left hc hmi
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2, hst2⟩ :=
    sb prog (base + ea.length) ss1 (a.eval fuel ws) fuel hpc1 hseg.append_right
      hlr.append_right hc1 hmi1
  have hw1 := allSteps_weaken
    (fun st h => (hmono base (base + ea.length) (base + ea.length + eb.length) st
      (by omega) (by omega)).1 h) hst1
  have hw2 := allSteps_weaken
    (fun st h => (hmono base (base + ea.length) (base + ea.length + eb.length) st
      (by omega) (by omega)).2 h) hst2
  rw [show base + (ea ++ eb).length = base + ea.length + eb.length from by
    rw [List.length_append]; omega]
  refine ⟨n1 + n2, ss2, sreaches_trans prog n1 n2 ss ss1 ss2 hr1 hr2, ?_, ?_, hmi2,
    allSteps_seq hw1 hr1 hw2⟩
  · exact hpc2
  · have hseval : WStmt.eval fuel (a.seq b) ws = b.eval fuel (a.eval fuel ws) := by
      simp [WStmt.eval]
    rw [hseval]; exact hc2

/-- A one-instruction leaf, mirroring `simSL'_single`.  This is where a real
    bound is discharged rather than transported: the two states are named, and
    the coupling is in hand, so an eval-level fact about `ws` becomes a machine
    fact about `ss` by `Couple.reg`. -/
theorem simSLQ_single (ib : Nat) (R : List String) (QQ : Nat → Nat → SState → Prop)
    (stmt : WStmt) (i0 : SInstr) (T : WState → WState)
    (f : ∀ (prog : Array SInstr) (ss : SState) (ws : WState),
      prog[ss.pc]? = some i0 → Couple R ss ws →
      Couple R (sstep prog ss) (T ws) ∧ (sstep prog ss).pc = ss.pc + 1)
    (heval : ∀ (ws : WState) (fuel : Nat), stmt.eval fuel ws = T ws)
    (hsafe : wtgt i0 ≠ some "lane" ∧ wtgt i0 ≠ some "inBase" ∧ wtgt i0 ≠ some "tbl")
    (hq : ∀ (prog : Array SInstr) (ss : SState) (ws : WState),
      prog[ss.pc]? = some i0 → Couple R ss ws → MachInv ib ss →
      QQ ss.pc (ss.pc + 1) ss ∧ QQ ss.pc (ss.pc + 1) (sstep prog ss)) :
    SimSLQ ib R QQ stmt [i0] := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi
  have hpci : prog[ss.pc]? = some i0 := by rw [hpc]; exact hseg.head i0 rfl
  obtain ⟨hcpl, hpc'⟩ := f prog ss ws hpci hc
  obtain ⟨q0, q1⟩ := hq prog ss ws hpci hc hmi
  rw [hpc] at q0 q1
  refine ⟨1, sstep prog ss, sreaches_one prog ss, by simp [hpc', hpc], ?_, ?_,
    allSteps_one q0 q1⟩
  · rw [heval]; exact hcpl
  · exact machInv_sstep ib prog ss hmi (fun i hi => by rw [hpci] at hi; cases hi; exact hsafe)

/-- **The `hscaf` a Couple-exporting `QQ` needs, discharged once and for all.**

    The predicate the entry conditions want is "at this program point there is an
    eval state this machine state couples to, satisfying the loop invariant".  At
    every scaffolding point the loop proof already holds exactly that witness, so
    the obligation is the identity — which is the whole reason `hscaf` had to be
    conditioned on the coupling rather than quantified over all states. -/
theorem hscaf_couple (R : List String) (Q : WState → Prop) (ib base bodyLen : Nat) :
    ∀ (st : SState) (w : WState), UwhileScaffold base bodyLen st.pc →
      Couple R st w → MachInv ib st → Q w →
      (fun s => s.pc = base → ∃ ws, Couple R s ws ∧ Q ws) st :=
  fun _ w _ hc _ hq _ => ⟨w, hc, hq⟩

/-- The pc-confinement predicate: the obligation the body has to discharge so
    the loop-head checkpoint can rule out a spurious visit to the head.  It is
    the cheapest useful `QQ` — no coupling, no eval state, just arithmetic. -/
abbrev PcIn (lo hi : Nat) (st : SState) : Prop := lo ≤ st.pc ∧ st.pc ≤ hi

/-- `PcIn` satisfies `simSLQ_seq`'s monotonicity hypothesis: a sub-range implies
    the union in either position, which is what lets the two halves of a `seq`
    be proven about their own ranges and then composed. -/
theorem pcIn_mono : ∀ (lo mid hi : Nat) (st : SState), lo ≤ mid → mid ≤ hi →
    (PcIn lo mid st → PcIn lo hi st) ∧ (PcIn mid hi st → PcIn lo hi st) :=
  fun _ _ _ _ h1 h2 =>
    ⟨fun h => ⟨h.1, by have := h.2; omega⟩, fun h => ⟨by have := h.1; omega, h.2⟩⟩

/-- A `PcIn`-confined segment never sits at a pc outside its own range — in
    particular never at the loop head that precedes it, which is exactly what
    `loopC_loop_sim_ckpt`'s `hbodyPc` asks for. -/
theorem pcIn_ne_of_lt {p : Array SInstr} {lo hi n : Nat} {ss : SState} (b : Nat)
    (h : AllSteps p (PcIn lo hi) n ss) (hb : b < lo) :
    ∀ j, j ≤ n → ¬ ((siter p j ss).pc = b) :=
  fun j hj he => by have := (h j hj).1; omega

/-- The upper-side dual of `pcIn_ne_of_lt`: a segment confined to `[lo, hi]` is
    never at a pc above `hi`.  Needed to discharge the token checkpoint on the
    segments that sit *below* `MB` in the found branch. -/
theorem pcIn_ne_of_gt {p : Array SInstr} {lo hi n : Nat} {ss : SState} (b : Nat)
    (h : AllSteps p (PcIn lo hi) n ss) (hb : hi < b) :
    ∀ j, j ≤ n → ¬ ((siter p j ss).pc = b) :=
  fun j hj he => by have := (h j hj).2; omega

/-- **A checkpoint predicate from pc-confinement, with no threading.**

    If a segment is confined to `[lo, hi]`, starts at `q`, and *nothing in that
    range can step to `q`*, then the only visit to `q` is the first one — so a
    property holding at the start holds at every visit.  The no-entry side
    condition is decidable at a concrete kernel (`succsOf` is computable), which
    is what makes this cheaper than carrying the predicate through the
    simulation: the coupling has to be exposed once, at the segment's entry,
    rather than at every combinator. -/
theorem allSteps_ckpt_of_pcIn {p : Array SInstr} {lo hi n q : Nat} {ss : SState}
    {P : SState → Prop}
    (hst : AllSteps p (PcIn lo hi) n ss) (hP : P ss)
    (hno : ∀ q', lo ≤ q' → q' ≤ hi → q ∉ succsOf p q') :
    AllSteps p (fun st => st.pc = q → P st) n ss := by
  intro j hj hq
  match j with
  | 0 => exact hP
  | k + 1 =>
    exfalso
    have hprev := hst k (by omega)
    have hmem : (siter p (k + 1) ss).pc ∈ succsOf p (siter p k ss).pc := by
      rw [siter_succ]; exact sstep_pc_mem_succs p _
    rw [hq] at hmem
    exact hno _ hprev.1 hprev.2 hmem

/-- **A confined segment cannot sit at the top of its own range before the end.**

    If `hi`'s successors are all above `hi`, then a step at `hi` leaves the range —
    so within a segment confined to `[lo, hi]` the pc equals `hi` only at the very
    last state.  The side condition is decidable at a concrete kernel. -/
theorem allSteps_ne_top {p : Array SInstr} {lo hi n : Nat} {ss : SState}
    (hst : AllSteps p (PcIn lo hi) n ss)
    (htop : ∀ q', q' ∈ succsOf p hi → hi < q') :
    ∀ j, j < n → (siter p j ss).pc ≠ hi := by
  intro j hj he
  have h1 := hst (j + 1) (by omega)
  have hmem : (siter p (j + 1) ss).pc ∈ succsOf p (siter p j ss).pc := by
    rw [siter_succ]; exact sstep_pc_mem_succs p _
  rw [he] at hmem
  have hgt := htop _ hmem
  have hle := h1.2
  omega

/-- **The seam checkpoint.**  A confined segment ending at `hi` visits `hi` exactly
    once — at its final state — so a property of that state is a property of *every*
    visit.  This is how a coupling exposed at one seam becomes an `AllSteps`
    obligation without threading anything through the simulation. -/
theorem allSteps_ckpt_end {p : Array SInstr} {lo hi n : Nat} {ss ss' : SState}
    {P : SState → Prop}
    (hst : AllSteps p (PcIn lo hi) n ss) (hr : SReaches p n ss ss')
    (htop : ∀ q', q' ∈ succsOf p hi → hi < q') (hP : P ss') :
    AllSteps p (fun st => st.pc = hi → P st) n ss := by
  intro j hj he
  rcases Nat.lt_or_ge j n with hlt | hge
  · exact absurd he (allSteps_ne_top hst htop j hlt)
  · have hjn : j = n := by omega
    have hiter : siter p j ss = ss' := by
      rw [hjn]; exact (sreaches_iff_siter p n ss ss').mp hr
    rw [hiter]; exact hP

/-- **Every single-instruction leaf is pc-confined for free.**  The two states a
    one-instruction segment visits have pcs `base` and `base + 1`, which is the
    range itself — so the bulk of the body pass costs nothing per leaf, and only
    the loops and the black-box segments need an argument of their own. -/
theorem simSLQ_single_pcIn (ib : Nat) (R : List String)
    (stmt : WStmt) (i0 : SInstr) (T : WState → WState)
    (f : ∀ (prog : Array SInstr) (ss : SState) (ws : WState),
      prog[ss.pc]? = some i0 → Couple R ss ws →
      Couple R (sstep prog ss) (T ws) ∧ (sstep prog ss).pc = ss.pc + 1)
    (heval : ∀ (ws : WState) (fuel : Nat), stmt.eval fuel ws = T ws)
    (hsafe : wtgt i0 ≠ some "lane" ∧ wtgt i0 ≠ some "inBase" ∧ wtgt i0 ≠ some "tbl") :
    SimSLQ ib R PcIn stmt [i0] :=
  simSLQ_single ib R PcIn stmt i0 T f heval hsafe
    (fun prog ss ws hpci hc _ => by
      have h := (f prog ss ws hpci hc).2
      exact ⟨by simp only [PcIn]; omega, by simp only [PcIn]; omega⟩)

/-- **A `uwhile`'s scaffolding is inside its own emit range.**  This is `hscaf`
    for the pc-confinement predicate, and it is pure arithmetic: the five
    scaffolding points span `base` to `base + bodyEmit.length + 3`, and
    `uwhileEmit` is four instructions longer than its body. -/
theorem hscaf_pcIn (R : List String) (Q : WState → Prop) (ib : Nat)
    (cond lHead lEnd : String) (bodyEmit : List SInstr) (base : Nat) :
    ∀ (st : SState) (w : WState), UwhileScaffold base bodyEmit.length st.pc →
      Couple R st w → MachInv ib st → Q w →
      PcIn base (base + (uwhileEmit cond lHead lEnd bodyEmit).length) st := by
  intro st _ hsc _ _ _
  rw [uwhileEmit_length]
  rcases hsc with e | e | e | e | e <;> exact ⟨by omega, by omega⟩

/-- `SimSLQ … PcIn` mov leaf — `simSL'_mov` with the confinement obligation
    discharged by `simSLQ_single_pcIn`.  Every straight-line leaf of the descent
    is this shape: the original proof term, unchanged, at the `Q`-carrying
    combinator. -/
theorem simSLQ_mov_pcIn (ib : Nat) (R : List String) (d : String) (a : WArg)
    (ha : ∀ n, a = .reg n → n ∈ R)
    (hd : d ≠ "lane" ∧ d ≠ "inBase" ∧ d ≠ "tbl") :
    SimSLQ ib R PcIn (.mov d a) [.mov d a.toS] :=
  simSLQ_single_pcIn ib R (.mov d a) (.mov d a.toS) (fun ws => ws.setReg d (a.eval ws))
    (fun prog ss ws hpci hc => mov_sound R prog d a ss ws hpci hc ha)
    (fun ws fuel => by simp [WStmt.eval])
    (by simp only [wtgt, ne_eq, Option.some.injEq]; exact ⟨hd.1, hd.2.1, hd.2.2⟩)

/-- `SimSLQ … PcIn` bin leaf. -/
theorem simSLQ_bin_pcIn (ib : Nat) (R : List String) (o : WOp) (d a : String) (b : WArg)
    (ha : a ∈ R) (hb : ∀ n, b = .reg n → n ∈ R)
    (hd : d ≠ "lane" ∧ d ≠ "inBase" ∧ d ≠ "tbl") :
    SimSLQ ib R PcIn (.bin o d a b) [.bin o.toS d a b.toS] :=
  simSLQ_single_pcIn ib R (.bin o d a b) (.bin o.toS d a b.toS)
    (fun ws => ws.setReg d (o.run (ws.regs a) (b.eval ws)))
    (fun prog ss ws hpci hc => bin_sound R prog o d a b ss ws hpci hc ha hb)
    (fun ws fuel => by simp [WStmt.eval])
    (by simp only [wtgt, ne_eq, Option.some.injEq]; exact ⟨hd.1, hd.2.1, hd.2.2⟩)

/-- `SimSLQ … PcIn` setp leaf — transcribed from `simSL'_setp`. -/
theorem simSLQ_setp_pcIn (ib : Nat) (R : List String) (c : SCmp) (d a : String) (b : WArg)
    (ha : a ∈ R) (hb : ∀ n, b = .reg n → n ∈ R)
    (hd : d ≠ "lane" ∧ d ≠ "inBase" ∧ d ≠ "tbl") :
    SimSLQ ib R PcIn (.setp c d a b) [.setp c d a b.toS] :=
  simSLQ_single_pcIn ib R (.setp c d a b) (.setp c d a b.toS)
    (fun ws => ws.setReg d (if c.run (ws.regs a) (b.eval ws) then 1 else 0))
    (fun prog ss ws hpci hc => setp_sound R prog c d a b ss ws hpci hc ha hb)
    (fun ws fuel => by simp [WStmt.eval])
    (by simp only [wtgt, ne_eq, Option.some.injEq]; exact ⟨hd.1, hd.2.1, hd.2.2⟩)

/-- Transcribed from `simSL'_stgB`. -/
theorem simSLQ_stgB_pcIn (ib : Nat) (R : List String) (addr s : String) (haddr : addr ∈ R) (hs : s ∈ R) :
    SimSLQ ib R PcIn (.stgB addr s) [.stg addr s] :=
  simSLQ_single_pcIn ib R (.stgB addr s) (.stg addr s)
    (fun ws => ws.stgByte (ws.regs addr) (ws.regs s))
    (fun prog ss ws hpci hc => stgU_sound R prog addr s ss ws hpci hc haddr hs)
    (fun ws fuel => by simp [WStmt.eval])
    (by simp only [wtgt, ne_eq]; exact ⟨by simp, by simp, by simp⟩)

/-- Transcribed from `simSL'_wStoreByte`. -/
theorem simSLQ_wStoreByte_pcIn (ib : Nat) (R : List String) (val : String)
    (hval : val ∈ R) (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R) :
    SimSLQ ib R PcIn (wStoreByte val)
      ([.bin .add "sbAddr" "outBase" (.reg "op")]
        ++ ([.stg "sbAddr" val] ++ [.bin .add "op" "op" (.imm 1)])) := by
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .add "sbAddr" "outBase" (.reg "op") hout (fun n h => by cases h; exact hop)
      (by decide)
  · apply simSLQ_seq ib R PcIn pcIn_mono
    · exact simSLQ_stgB_pcIn ib R "sbAddr" val hsb hval
    · exact simSLQ_bin_pcIn ib R .add "op" "op" (.imm 1) hop (fun n h => by cases h) (by decide)

/-- Transcribed from `simSL'_wEmitToken`. -/
theorem simSLQ_wEmitToken_pcIn (ib : Nat) (R : List String) (litLen tokLo : String)
    (hll : litLen ∈ R) (htl : tokLo ∈ R)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R)
    (htokHi : "tokHi" ∈ R) (htok : "tok" ∈ R) :
    SimSLQ ib R PcIn (wEmitToken litLen tokLo)
      ([.bin .min "tokHi" litLen (.imm 15)]
        ++ ([.bin .shl "tok" "tokHi" (.imm 4)]
          ++ ([.bin .bor "tok" "tok" (.reg tokLo)]
            ++ ([.bin .add "sbAddr" "outBase" (.reg "op")]
              ++ ([.stg "sbAddr" "tok"] ++ [.bin .add "op" "op" (.imm 1)]))))) := by
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .min "tokHi" litLen (.imm 15) hll (fun n h => by cases h) (by decide)
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .shl "tok" "tokHi" (.imm 4) htokHi (fun n h => by cases h) (by decide)
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .bor "tok" "tok" (.reg tokLo) htok (fun n h => by cases h; exact htl)
      (by decide)
  · exact simSLQ_wStoreByte_pcIn ib R "tok" htok hout hop hsb

/-- Transcribed from `simSL'_wEmitLSIC_body`. -/
theorem simSLQ_wEmitLSIC_body_pcIn (ib : Nat) (R : List String) (n : String) (hn : n ∈ R)
    (hnW : n ≠ "lane" ∧ n ≠ "inBase" ∧ n ≠ "tbl")
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R)
    (hc255 : "c255" ∈ R) (hlsicC : "lsicC" ∈ R) :
    SimSLQ ib R PcIn
      (wseq [ .mov "c255" (.imm 255), wStoreByte "c255", .bin .sub n n (.imm 255),
              .setp .ge "lsicC" n (.imm 255) ])
      ([.mov "c255" (SArg.imm 255)]
        ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
            ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
          ++ ([.bin .sub n n (.imm 255)] ++ [.setp .ge "lsicC" n (.imm 255)]))) := by
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_mov_pcIn ib R "c255" (.imm 255) (fun m h => by cases h) (by decide)
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_wStoreByte_pcIn ib R "c255" hc255 hout hop hsb
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .sub n n (.imm 255) hn (fun m h => by cases h) hnW
  · exact simSLQ_setp_pcIn ib R .ge "lsicC" n (.imm 255) hn (fun m h => by cases h) (by decide)

/-- Transcribed from `simSL'_foundMovs`. -/
theorem simSLQ_foundMovs_pcIn (ib : Nat) (R : List String) (endCap : Nat)
    (hecR : "ecR" ∈ R) (hec1 : "ec1" ∈ R) (hml : "ml" ∈ R) (hextC : "extC" ∈ R) :
    SimSLQ ib R PcIn
      (wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
              .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ])
      (([.mov "ecR" (SArg.imm endCap)] : List SInstr)
        ++ ([.mov "ec1" (.imm (endCap - 1))]
          ++ ([.mov "ml" (.imm 4)] ++ [.setp .ge "extC" "ml" (.imm 0)]))) := by
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_mov_pcIn ib R "ecR" (.imm endCap) (fun n h => by cases h) (by decide)
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_mov_pcIn ib R "ec1" (.imm (endCap - 1)) (fun n h => by cases h) (by decide)
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_mov_pcIn ib R "ml" (.imm 4) (fun n h => by cases h) (by decide)
  · exact simSLQ_setp_pcIn ib R .ge "extC" "ml" (.imm 0) hml (fun n h => by cases h) (by decide)

/-- Transcribed from `simSL'_foundSubs`. -/
theorem simSLQ_foundSubs_pcIn (ib : Nat) (R : List String) (litStart : String)
    (hp0 : "p0" ∈ R) (hcand0 : "cand0" ∈ R) (hls : litStart ∈ R)
    (hoff0 : "off0" ∈ R) (hlitLen : "litLen" ∈ R) :
    SimSLQ ib R PcIn
      (wseq [ .bin .sub "off0" "p0" (.reg "cand0"),
              .bin .sub "litLen" "p0" (.reg litStart) ])
      (([.bin .sub "off0" "p0" (.reg "cand0")] : List SInstr)
        ++ [.bin .sub "litLen" "p0" (.reg litStart)]) := by
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .sub "off0" "p0" (.reg "cand0") hp0 (fun n h => by cases h; exact hcand0)
      (by decide)
  · exact simSLQ_bin_pcIn ib R .sub "litLen" "p0" (.reg litStart) hp0 (fun n h => by cases h; exact hls)
      (by decide)

/-- Transcribed from `simSL'_matchPre`. -/
theorem simSLQ_matchPre_pcIn (ib : Nat) (R : List String) (litLen ml : String)
    (hll : litLen ∈ R) (hml : ml ∈ R) (hmlm : "mlm" ∈ R) (htokLo : "tokLo" ∈ R)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R)
    (htokHi : "tokHi" ∈ R) (htok : "tok" ∈ R) (hpb : "pLitBig" ∈ R) :
    SimSLQ ib R PcIn
      (wseq [ .bin .sub "mlm" ml (.imm 4), .bin .min "tokLo" "mlm" (.imm 15),
              wEmitToken litLen "tokLo", .setp .ge "pLitBig" litLen (.imm 15) ])
      (matchPreEmit litLen ml) := by
  rw [matchPreEmit]
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .sub "mlm" ml (.imm 4) hml (fun n h => by cases h) (by decide)
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .min "tokLo" "mlm" (.imm 15) hmlm (fun n h => by cases h) (by decide)
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_wEmitToken_pcIn ib R litLen "tokLo" hll htokLo hout hop hsb htokHi htok
  · exact simSLQ_setp_pcIn ib R .ge "pLitBig" litLen (.imm 15) hll (fun n h => by cases h) (by decide)

/-- Transcribed from `simSL'_matchMid`. -/
theorem simSLQ_matchMid_pcIn (ib : Nat) (R : List String) (litStart : String)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hib : "inBase" ∈ R) (hls : litStart ∈ R)
    (hcd : "cpDst" ∈ R) (hcs : "cpSrc" ∈ R) :
    SimSLQ ib R PcIn
      (wseq [ .bin .add "cpDst" "outBase" (.reg "op"),
              .bin .add "cpSrc" "inBase" (.reg litStart) ])
      (matchMidEmit litStart) := by
  rw [matchMidEmit]
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .add "cpDst" "outBase" (.reg "op") hout (fun n h => by cases h; exact hop)
      (by decide)
  · exact simSLQ_bin_pcIn ib R .add "cpSrc" "inBase" (.reg litStart) hib
      (fun n h => by cases h; exact hls) (by decide)

/-- Transcribed from `simSL'_matchOff`. -/
theorem simSLQ_matchOff_pcIn (ib : Nat) (R : List String) (litLen off : String)
    (hll : litLen ∈ R) (hoff : off ∈ R) (hmlm : "mlm" ∈ R)
    (hoffLo : "offLo" ∈ R) (hoffHi : "offHi" ∈ R)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hsb : "sbAddr" ∈ R) (hpm : "pMatBig" ∈ R) :
    SimSLQ ib R PcIn
      (wseq [ .bin .add "op" "op" (.reg litLen),
              .bin .band "offLo" off (.imm 255), wStoreByte "offLo",
              .bin .shr "offHi" off (.imm 8), .bin .band "offHi" "offHi" (.imm 255),
              wStoreByte "offHi", .setp .ge "pMatBig" "mlm" (.imm 15) ])
      (matchOffEmit litLen off) := by
  rw [matchOffEmit]
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .add "op" "op" (.reg litLen) hop (fun n h => by cases h; exact hll) (by decide)
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .band "offLo" off (.imm 255) hoff (fun n h => by cases h) (by decide)
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_wStoreByte_pcIn ib R "offLo" hoffLo hout hop hsb
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .shr "offHi" off (.imm 8) hoff (fun n h => by cases h) (by decide)
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .band "offHi" "offHi" (.imm 255) hoffHi (fun n h => by cases h) (by decide)
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_wStoreByte_pcIn ib R "offHi" hoffHi hout hop hsb
  · exact simSLQ_setp_pcIn ib R .ge "pMatBig" "mlm" (.imm 15) hmlm (fun n h => by cases h) (by decide)

/-- Transcribed from `simSL'_finalPrefix`. -/
theorem simSLQ_finalPrefix_pcIn (ib : Nat) (R : List String) (litStart litLen : String)
    (hll : litLen ∈ R) (hzero : "zero" ∈ R) (hout : "outBase" ∈ R) (hop : "op" ∈ R)
    (hsb : "sbAddr" ∈ R) (htokHi : "tokHi" ∈ R) (htok : "tok" ∈ R) (hpb : "pLitBigF" ∈ R) :
    SimSLQ ib R PcIn
      (wseq [ .mov "zero" (.imm 0), wEmitToken litLen "zero",
              .setp .ge "pLitBigF" litLen (.imm 15) ])
      (([.mov "zero" (SArg.imm 0)] : List SInstr)
        ++ (([.bin .min "tokHi" litLen (.imm 15)]
          ++ ([.bin .shl "tok" "tokHi" (.imm 4)]
            ++ ([.bin .bor "tok" "tok" (.reg "zero")]
              ++ ([.bin .add "sbAddr" "outBase" (.reg "op")]
                ++ ([.stg "sbAddr" "tok"] ++ [.bin .add "op" "op" (.imm 1)])))))
          ++ [.setp .ge "pLitBigF" litLen (.imm 15)])) := by
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_mov_pcIn ib R "zero" (.imm 0) (fun n h => by cases h) (by decide)
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_wEmitToken_pcIn ib R litLen "zero" hll hzero hout hop hsb htokHi htok
  · exact simSLQ_setp_pcIn ib R .ge "pLitBigF" litLen (.imm 15) hll (fun n h => by cases h) (by decide)

/-- Transcribed from `simSL'_finalMid`. -/
theorem simSLQ_finalMid_pcIn (ib : Nat) (R : List String) (litStart : String)
    (hout : "outBase" ∈ R) (hop : "op" ∈ R) (hib : "inBase" ∈ R) (hls : litStart ∈ R)
    (hcd : "cpDstF" ∈ R) (hcs : "cpSrcF" ∈ R) :
    SimSLQ ib R PcIn
      (wseq [ .bin .add "cpDstF" "outBase" (.reg "op"),
              .bin .add "cpSrcF" "inBase" (.reg litStart) ])
      (([.bin .add "cpDstF" "outBase" (.reg "op")] : List SInstr)
        ++ [.bin .add "cpSrcF" "inBase" (.reg litStart)]) := by
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .add "cpDstF" "outBase" (.reg "op") hout (fun n h => by cases h; exact hop)
      (by decide)
  · exact simSLQ_bin_pcIn ib R .add "cpSrcF" "inBase" (.reg litStart) hib
      (fun n h => by cases h; exact hls) (by decide)

/-- Transcribed from `simSL'_uwhile`, with the pc-confinement obligation. -/
theorem simSLQ_uwhile_pcIn (ib : Nat) (R : List String) (cond lHead lEnd : String) (body : WStmt)
    (ebody : List SInstr) (prog : Array SInstr) (base : Nat)
    (hcond : cond ∈ R) (sb : SimSLQ ib R PcIn body ebody)
    (hseg : SegAt prog base (uwhileEmit cond lHead lEnd ebody))
    (hlr : LabelsResolve prog base (uwhileEmit cond lHead lEnd ebody)) :
    ∀ (fuel : Nat) (ss : SState) (ws : WState),
      ss.pc = base → Couple R ss ws → MachInv ib ss → WhileHalts cond body fuel ws →
      ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
        ss'.pc = base + (uwhileEmit cond lHead lEnd ebody).length ∧
        Couple R ss' (WStmt.eval fuel (.uwhile cond body) ws) ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base + (uwhileEmit cond lHead lEnd ebody).length)) n ss := by
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
      obtain ⟨nb, ss1, hrB, hpcB, hcB, hmiB, hstB⟩ :=
        sb prog (base + 1 + 1) (ss.setPc (base + 1 + 1)) ws fuel rfl hsegBody hlrBody
          (couple_setPc hc _) (machInv_setPc ss _ hmi)
      have hbra' : prog[ss1.pc]? = some (.bra lHead) := by rw [hpcB]; exact hbra
      have sBk : sstep prog ss1 = ss1.setPc base := by rw [bra_step prog ss1 lHead hbra', hLhead]
      have hHrec : WhileHalts cond body fuel (body.eval fuel ws) := by
        rw [WhileHalts] at hH; rw [if_pos hb] at hH; exact hH
      obtain ⟨nr, ssf, hrR, hpcR, hcR, hmiR, hstR⟩ :=
        ih (ss1.setPc base) (body.eval fuel ws) rfl (couple_setPc hcB _)
          (machInv_setPc ss1 _ hmiB) hHrec
      refine ⟨1 + 1 + nb + 1 + nr, ssf,
        sreaches_trans prog (1 + 1 + nb + 1) nr _ _ _
          (sreaches_trans prog (1 + 1 + nb) 1 _ _ _
            (sreaches_trans prog (1 + 1) nb _ _ _
              (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
              hrB)
            (sreaches_one_eq sBk))
          hrR, hpcR, ?_, hmiR, ?_⟩
      · have heval : WStmt.eval (fuel + 1) (.uwhile cond body) ws
            = WStmt.eval fuel (.uwhile cond body) (body.eval fuel ws) := by
          simp [WStmt.eval, hb]
        rw [heval]; exact hcR
      · have hL : (uwhileEmit cond lHead lEnd ebody).length = ebody.length + 4 :=
          uwhileEmit_length ..
        have hpcB' := hpcB
        refine allSteps_seq (allSteps_seq (allSteps_seq (allSteps_seq
          (allSteps_one (by rw [PcIn, hpc]; omega) (by rw [PcIn, sH]; simp [SState.setPc]; omega))
          (sreaches_one_eq sH)
          (allSteps_one (by rw [PcIn]; simp [SState.setPc]; omega)
            (by rw [PcIn, sB0]; simp [SState.setPc]; omega)))
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
          (allSteps_weaken (fun st h =>
            ⟨by have := h.1; omega, by have := h.2; rw [hL]; omega⟩) hstB))
          (sreaches_trans prog (1 + 1) nb _ _ _
            (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0)) hrB)
          (allSteps_one (by simp only [PcIn, hpcB']; rw [hL]; omega)
            (by simp only [PcIn, sBk, SState.setPc]; omega)))
          (sreaches_trans prog (1 + 1 + nb) 1 _ _ _
            (sreaches_trans prog (1 + 1) nb _ _ _
              (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0)) hrB)
            (sreaches_one_eq sBk))
          hstR
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
          (sreaches_one_eq sE), ?_, ?_, machInv_setPc ss _ hmi, ?_⟩
      · show (base + 1 + 1 + ebody.length + 1 + 1)
          = base + (uwhileEmit cond lHead lEnd ebody).length
        rw [uwhileEmit_length]; omega
      · have heval : WStmt.eval (fuel + 1) (.uwhile cond body) ws = ws := by
          simp [WStmt.eval, hb]
        rw [heval]; exact couple_setPc hc _
      · have hL : (uwhileEmit cond lHead lEnd ebody).length = ebody.length + 4 :=
          uwhileEmit_length ..
        exact allSteps_seq (allSteps_seq
          (allSteps_one (by rw [PcIn, hpc]; omega) (by rw [PcIn, sH]; simp [SState.setPc]; omega))
          (sreaches_one_eq sH)
          (allSteps_one (by rw [PcIn]; simp [SState.setPc]; omega)
            (by rw [PcIn, sB0]; simp [SState.setPc]; rw [hL]; omega)))
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
          (allSteps_one (by rw [PcIn]; simp [SState.setPc]; rw [hL]; omega)
            (by rw [PcIn, sE]; simp [SState.setPc]; rw [hL]; omega))

/-- `BodySimInv` carrying a per-step obligation, as `SimSLQ` does for `SimSL'`.
    The extend loop's engine takes its body this way rather than as a `SimSL'`. -/
def BodySimInvQ (ib : Nat) (R : List String) (QQ : Nat → Nat → SState → Prop)
    (P : WState → Prop) (bodyStmt : WStmt) (bodyEmit : List SInstr) : Prop :=
  ∀ (prog : Array SInstr) (base : Nat) (ss : SState) (ws : WState) (fuel : Nat),
    ss.pc = base → SegAt prog base bodyEmit → LabelsResolve prog base bodyEmit →
    Couple R ss ws → MachInv ib ss → P ws →
    (∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧ ss'.pc = base + bodyEmit.length ∧
      Couple R ss' (bodyStmt.eval fuel ws) ∧ MachInv ib ss' ∧
      AllSteps prog (QQ base (base + bodyEmit.length)) m ss)
    ∧ P (bodyStmt.eval fuel ws)

/-- Transcribed from `simSL'_uwhileBodyInv`, with pc-confinement.  Closes the
    descent gate together with `simSLQ_uwhile_pcIn`. -/
theorem simSLQ_uwhileBodyInv_pcIn (ib : Nat) (R : List String) (cond lHead lEnd : String) (P : WState → Prop)
    (bodyStmt : WStmt) (bodyEmit : List SInstr) (prog : Array SInstr) (base : Nat)
    (hcond : cond ∈ R) (hbody : BodySimInvQ ib R PcIn P bodyStmt bodyEmit)
    (hseg : SegAt prog base (uwhileEmit cond lHead lEnd bodyEmit))
    (hlr : LabelsResolve prog base (uwhileEmit cond lHead lEnd bodyEmit)) :
    ∀ (fuel : Nat) (ss : SState) (ws : WState),
      ss.pc = base → Couple R ss ws → MachInv ib ss → P ws → WhileHalts cond bodyStmt fuel ws →
      ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
        ss'.pc = base + (uwhileEmit cond lHead lEnd bodyEmit).length ∧
        Couple R ss' (WStmt.eval fuel (.uwhile cond bodyStmt) ws) ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base + (uwhileEmit cond lHead lEnd bodyEmit).length)) n ss := by
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
      obtain ⟨⟨nb, ss1, hrB, hpcB, hcB, hmiB, hstB⟩, hPnext⟩ :=
        hbody prog (base + 1 + 1) (ss.setPc (base + 1 + 1)) ws fuel rfl hsegBody hlrBody
          (couple_setPc hc _) (machInv_setPc ss _ hmi) hP
      have hbra' : prog[ss1.pc]? = some (.bra lHead) := by rw [hpcB]; exact hbra
      have sBk : sstep prog ss1 = ss1.setPc base := by rw [bra_step prog ss1 lHead hbra', hLhead]
      have hHrec : WhileHalts cond bodyStmt fuel (bodyStmt.eval fuel ws) := by
        rw [WhileHalts] at hH; rw [if_pos hb] at hH; exact hH
      obtain ⟨nr, ssf, hrR, hpcR, hcR, hmiR, hstR⟩ :=
        ih (ss1.setPc base) (bodyStmt.eval fuel ws) rfl (couple_setPc hcB _)
          (machInv_setPc ss1 _ hmiB) hPnext hHrec
      refine ⟨1 + 1 + nb + 1 + nr, ssf,
        sreaches_trans prog (1 + 1 + nb + 1) nr _ _ _
          (sreaches_trans prog (1 + 1 + nb) 1 _ _ _
            (sreaches_trans prog (1 + 1) nb _ _ _
              (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
              hrB)
            (sreaches_one_eq sBk))
          hrR, hpcR, ?_, hmiR, ?_⟩
      · have heval : WStmt.eval (fuel + 1) (.uwhile cond bodyStmt) ws
            = WStmt.eval fuel (.uwhile cond bodyStmt) (bodyStmt.eval fuel ws) := by
          simp [WStmt.eval, hb]
        rw [heval]; exact hcR
      · exact allSteps_seq (allSteps_seq (allSteps_seq (allSteps_seq
          (allSteps_one (by simp only [PcIn, uwhileEmit_length, hpc]; omega)
            (by rw [sH]; simp only [PcIn, SState.setPc, uwhileEmit_length]; omega))
          (sreaches_one_eq sH)
          (allSteps_one (by simp only [PcIn, SState.setPc, uwhileEmit_length]; omega)
            (by rw [sB0]; simp only [PcIn, SState.setPc, uwhileEmit_length]; omega)))
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
          (allSteps_weaken (fun st h => ⟨by have := h.1; omega,
            by have := h.2; rw [uwhileEmit_length]; omega⟩) hstB))
          (sreaches_trans prog (1 + 1) nb _ _ _
            (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0)) hrB)
          (allSteps_one (by simp only [PcIn, uwhileEmit_length, hpcB]; omega)
            (by rw [sBk]; simp only [PcIn, SState.setPc, uwhileEmit_length]; omega)))
          (sreaches_trans prog (1 + 1 + nb) 1 _ _ _
            (sreaches_trans prog (1 + 1) nb _ _ _
              (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0)) hrB)
            (sreaches_one_eq sBk))
          hstR
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
          (sreaches_one_eq sE), ?_, ?_, machInv_setPc ss _ hmi,
        allSteps_seq (allSteps_seq
          (allSteps_one (by simp only [PcIn, uwhileEmit_length, hpc]; omega)
            (by rw [sH]; simp only [PcIn, SState.setPc, uwhileEmit_length]; omega))
          (sreaches_one_eq sH)
          (allSteps_one (by simp only [PcIn, SState.setPc, uwhileEmit_length]; omega)
            (by rw [sB0]; simp only [PcIn, SState.setPc, uwhileEmit_length]; omega)))
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq sH) (sreaches_one_eq sB0))
          (allSteps_one (by simp only [PcIn, SState.setPc, uwhileEmit_length]; omega)
            (by rw [sE]; simp only [PcIn, SState.setPc, uwhileEmit_length]; omega))⟩
      · show (base + 1 + 1 + bodyEmit.length + 1 + 1)
          = base + (uwhileEmit cond lHead lEnd bodyEmit).length
        rw [uwhileEmit_length]; omega
      · have heval : WStmt.eval (fuel + 1) (.uwhile cond bodyStmt) ws = ws := by
          simp [WStmt.eval, hb]
        rw [heval]; exact couple_setPc hc _

/-- Transcribed from `simSL'_wEmitLSIC`. -/
theorem simSLQ_wEmitLSIC_pcIn (ib : Nat) (R : List String) (n lH lX : String)
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
        Couple R ss' ((wEmitLSIC n).eval fuel ws) ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base +
          (([.setp .ge "lsicC" n (.imm 255)] : List SInstr)
            ++ (uwhileEmit "lsicC" lH lX lsicBody
              ++ (([.bin .add "sbAddr" "outBase" (.reg "op")] : List SInstr)
                ++ (([.stg "sbAddr" n] : List SInstr)
                  ++ ([.bin .add "op" "op" (.imm 1)] : List SInstr))))).length)) m ss := by
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
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1, hst1⟩ :=
    (simSLQ_setp_pcIn ib R .ge "lsicC" n (.imm 255) hn (fun m h => by cases h) (by decide))
      prog base ss ws fuel hpc hsegSetp hlr.append_left hc hmi
  -- Step 2: the LSIC loop, via `simSL'_uwhile`.
  have hsegLoop : SegAt prog (base + 1) (uwhileEmit "lsicC" lH lX lsicBody) := hsegRest.append_left
  have hlrLoop : LabelsResolve prog (base + 1) (uwhileEmit "lsicC" lH lX lsicBody) :=
    hlrRest.append_left
  have hbodySL : SimSLQ ib R PcIn
      (wseq [ .mov "c255" (.imm 255), wStoreByte "c255", .bin .sub n n (.imm 255),
              .setp .ge "lsicC" n (.imm 255) ]) lsicBody := by
    rw [hbodyDef]; exact simSLQ_wEmitLSIC_body_pcIn ib R n hn hnW hout hop hsb hc255 hlsicC
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2, hst2⟩ :=
    simSLQ_uwhile_pcIn ib R "lsicC" lH lX _ lsicBody prog (base + 1) hlsicC hbodySL hsegLoop hlrLoop
      fuel ss1 ((WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255)).eval fuel ws)
      (by rw [hpc1]; simp) (by simpa [SArg.reg] using hc1) hmi1 hHalt
  -- Step 3: the trailing `wStoreByte n`.
  have hsegStore : SegAt prog (base + 1 + (uwhileEmit "lsicC" lH lX lsicBody).length)
      ([.bin .add "sbAddr" "outBase" (.reg "op")]
        ++ ([.stg "sbAddr" n] ++ [.bin .add "op" "op" (.imm 1)])) := hsegRest.append_right
  have hlrStore : LabelsResolve prog (base + 1 + (uwhileEmit "lsicC" lH lX lsicBody).length)
      ([.bin .add "sbAddr" "outBase" (.reg "op")]
        ++ ([.stg "sbAddr" n] ++ [.bin .add "op" "op" (.imm 1)])) := hlrRest.append_right
  obtain ⟨n3, ss3, hr3, hpc3, hc3, hmi3, hst3⟩ :=
    (simSLQ_wStoreByte_pcIn ib R n hn hout hop hsb) prog
      (base + 1 + (uwhileEmit "lsicC" lH lX lsicBody).length) ss2
      ((WStmt.uwhile "lsicC"
        (wseq [ .mov "c255" (.imm 255), wStoreByte "c255", .bin .sub n n (.imm 255),
                .setp .ge "lsicC" n (.imm 255) ])).eval fuel
        ((WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255)).eval fuel ws)) fuel
      (by rw [hpc2]) hsegStore hlrStore hc2 hmi2
  refine ⟨n1 + (n2 + n3), ss3,
    sreaches_trans prog n1 (n2 + n3) _ _ _ hr1 (sreaches_trans prog n2 n3 _ _ _ hr2 hr3),
    ?_, ?_, hmi3, ?_⟩
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
  · exact allSteps_seq (allSteps_weaken (fun st h => ⟨by have := h.1; omega,
        by have := h.2
           simp only [List.length_append, List.length_cons, List.length_nil] at *
           omega⟩) hst1) hr1
      (allSteps_seq (allSteps_weaken (fun st h => ⟨by have := h.1; omega,
        by have := h.2
           simp only [List.length_append, List.length_cons, List.length_nil] at *
           omega⟩) hst2) hr2 (allSteps_weaken (fun st h => ⟨by have := h.1; omega,
        by have := h.2
           simp only [List.length_append, List.length_cons, List.length_nil] at *
           omega⟩) hst3))

/-- Transcribed from `simSL'_foundUpdates`. -/
theorem simSLQ_foundUpdates_pcIn (ib : Nat) (R : List String)
    (hp0 : "p0" ∈ R) (hml : "ml" ∈ R) (hla : "litAnchor" ∈ R) (hsp : "searchPos" ∈ R) :
    SimSLQ ib R PcIn
      (wseq [ .bin .add "litAnchor" "p0" (.reg "ml"), .mov "searchPos" (.reg "litAnchor") ])
      (([.bin .add "litAnchor" "p0" (.reg "ml")] : List SInstr)
        ++ [.mov "searchPos" (.reg "litAnchor")]) := by
  apply simSLQ_seq ib R PcIn pcIn_mono
  · exact simSLQ_bin_pcIn ib R .add "litAnchor" "p0" (.reg "ml") hp0 (fun n h => by cases h; exact hml)
      (by decide)
  · exact simSLQ_mov_pcIn ib R "searchPos" (.reg "litAnchor") (fun n h => by cases h; exact hla) (by decide)


/-- **Transcribed from `simSL'_lsicUif`, carrying pc-confinement.**

    The `uif` is not a plain `SimSL'` (it carries the LSIC fuel bound), so this is
    a full transcription rather than a combinator instance.  The two arms are
    handled separately: each is a `SReaches` chain whose `AllSteps` is the same
    chain reassembled with `allSteps_seq`, with `allSteps_one` for the branch,
    the `bra` and the two `lbl` steps. -/
theorem simSLQ_lsicUif_pcIn (ib : Nat) (R : List String)
    (pcond srcReg eReg lElse lEnd lH lX : String)
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
          ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base +
          (uifEmit pcond lElse lEnd (lsicThen srcReg eReg lH lX lsicBody) []).length)) m ss := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi hsrcG hfuel
  have hcv : ss.regs pcond 0 = ws.regs pcond := hc.reg hpcR 0
  obtain ⟨hbr, hseg1⟩ := hseg.cons
  have hbr' : prog[ss.pc]? = some (.braifnot pcond lElse) := by rw [hpc]; exact hbr
  have thenSim : ∀ ss0, ss0.pc = base + 1 → Couple R ss0 ws → MachInv ib ss0 →
      ws.regs pcond = 1 →
      ∃ (m : Nat) (ss' : SState), SReaches prog m ss0 ss' ∧
        ss'.pc = base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length ∧
        Couple R ss'
          ((wseq [ .bin .sub eReg srcReg (.imm 15), wEmitLSIC eReg ]).eval fuel ws) ∧
          MachInv ib ss' ∧
        AllSteps prog
          (PcIn (base + 1) (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length)) m ss0 := by
    intro ss0 hpc0 hc0 hmi0 hp1
    have hsegT : SegAt prog (base + 1) (lsicThen srcReg eReg lH lX lsicBody) := by
      have := hseg1.append_left (eb := [.bra lEnd, .lbl lElse] ++ ([] ++ [.lbl lEnd]))
      simpa [uifEmit, lsicThen] using this
    have hlrT : LabelsResolve prog (base + 1) (lsicThen srcReg eReg lH lX lsicBody) := by
      have := hlr.cons.append_left (eb := [.bra lEnd, .lbl lElse] ++ ([] ++ [.lbl lEnd]))
      simpa [uifEmit, lsicThen] using this
    rw [lsicThen] at hsegT hlrT
    have hsegSub : SegAt prog (base + 1) [.bin .sub eReg srcReg (.imm 15)] := hsegT.append_left
    obtain ⟨na, ssA, hrA, hpcA, hcA, hmiA, hstA⟩ :=
      (simSLQ_bin_pcIn ib R .sub eReg srcReg (.imm 15) hsrc (fun m h => by cases h) heRegW)
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
    obtain ⟨nb, ssB, hrB, hpcB, hcB, hmiB, hstB⟩ :=
      simSLQ_wEmitLSIC_pcIn ib R eReg lH lX heReg heRegW hout hop hsb hc255 hlsicC lsicBody hbodyDef
        prog (base + 1 + 1) ssA (ws.setReg eReg (ws.regs srcReg - UInt64.ofNat 15)) fuel
        (by rw [hpcA]; rfl) hsegLS hlrLS (by rw [hwsA] at hcA; exact hcA) hmiA
        (lsicLoop_halts eReg heC heO heS heL fuel
          ((WStmt.setp SCmp.ge "lsicC" eReg (WArg.imm 255)).eval fuel
            (ws.setReg eReg (ws.regs srcReg - UInt64.ofNat 15)))
          (fun hg => by rw [hentryReg]; exact hentryLsic hg)
          (by rw [hentryReg, hsubN]; omega))
    refine ⟨na + nb, ssB, sreaches_trans prog na nb _ _ _ hrA hrB, ?_, ?_, hmiB, ?_⟩
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
    · exact allSteps_seq (allSteps_weaken (fun st h => ⟨by have := h.1; omega,
        by have := h.2
           simp only [lsicThen, lsicEmit, uwhileEmit_length, List.length_append,
             List.length_cons, List.length_nil] at *
           omega⟩) hstA) hrA
        (allSteps_weaken (fun st h => ⟨by have := h.1; omega,
          by have := h.2
             simp only [lsicThen, lsicEmit, uwhileEmit_length, List.length_append,
               List.length_cons, List.length_nil] at *
             omega⟩) hstB)
  have hLen := lsicThen_length srcReg eReg lH lX lsicBody
  by_cases hb : (ws.regs pcond == 1) = true
  · have hp1 : ws.regs pcond = 1 := by rw [beq_iff_eq] at hb; exact hb
    have s0 : sstep prog ss = ss.setPc (base + 1) := by
      rw [braifnot_step prog ss pcond lElse hbr', hcv, hpc, if_pos hb]
    obtain ⟨nt, ss1, hrT, hpcT, hcT, hmiT, hstT⟩ :=
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
    have r1 : SReaches prog (1 + nt) ss ss1 :=
      sreaches_trans prog 1 nt ss (ss.setPc (base + 1)) ss1 (sreaches_one_eq s0) hrT
    have r2 : SReaches prog (1 + nt + 1) ss
        (ss1.setPc (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1 + 0)) :=
      sreaches_trans prog (1 + nt) 1 _ _ _ r1 (sreaches_one_eq s1)
    refine ⟨1 + nt + 1 + 1, _,
      sreaches_trans prog (1 + nt + 1) 1 _ _ _ r2 (sreaches_one_eq s2),
      ?_, ?_, machInv_setPc _ _ (machInv_setPc _ _ hmiT), ?_⟩
    · show (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1 + 0 + 1)
        = base + (uifEmit pcond lElse lEnd (lsicThen srcReg eReg lH lX lsicBody) []).length
      rw [uifEmit_length]; simp; omega
    · have heval : WStmt.eval fuel
          (.uif pcond (wseq [ .bin .sub eReg srcReg (.imm 15), wEmitLSIC eReg ]) .skip) ws
          = (wseq [ .bin .sub eReg srcReg (.imm 15), wEmitLSIC eReg ]).eval fuel ws := by
        simp [WStmt.eval, hb]
      rw [heval]; exact couple_setPc (couple_setPc hcT _) _
    · refine allSteps_seq (allSteps_seq (allSteps_seq ?_ (sreaches_one_eq s0) ?_) r1 ?_) r2 ?_
      · exact allSteps_one
          (by simp only [PcIn, hpc, uifEmit_length]; omega)
          (by rw [s0]; simp only [PcIn, SState.setPc, uifEmit_length]; omega)
      · exact allSteps_weaken (fun st h => ⟨by have := h.1; omega,
          by have := h.2
             simp only [uifEmit_length, lsicThen, lsicEmit, uwhileEmit_length,
               List.length_append, List.length_cons, List.length_nil] at *
             omega⟩) hstT
      · exact allSteps_one
          (by simp only [PcIn, hpcT, uifEmit_length, lsicThen, lsicEmit, uwhileEmit_length,
                List.length_append, List.length_cons, List.length_nil]; omega)
          (by rw [s1]; simp only [PcIn, SState.setPc, uifEmit_length, lsicThen, lsicEmit,
                uwhileEmit_length, List.length_append, List.length_cons, List.length_nil]; omega)
      · exact allSteps_one
          (by simp only [PcIn, SState.setPc, uifEmit_length, lsicThen, lsicEmit,
                uwhileEmit_length, List.length_append, List.length_cons, List.length_nil]; omega)
          (by rw [s2]; simp only [PcIn, SState.setPc, uifEmit_length, lsicThen, lsicEmit,
                uwhileEmit_length, List.length_append, List.length_cons, List.length_nil]; omega)
  · obtain ⟨hbra, hseg3⟩ := hseg1.append_right.cons
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
    have q2 : SReaches prog (1 + 1) ss
        (ss.setPc (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1)) :=
      sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq s0) (sreaches_one_eq s1)
    refine ⟨1 + 1 + 1, _,
      sreaches_trans prog (1 + 1) 1 _ _ _ q2 (sreaches_one_eq s2), ?_, ?_,
      machInv_setPc _ _ (machInv_setPc _ _ (machInv_setPc _ _ hmi)), ?_⟩
    · show (base + 1 + (lsicThen srcReg eReg lH lX lsicBody).length + 1 + 1 + 1)
        = base + (uifEmit pcond lElse lEnd (lsicThen srcReg eReg lH lX lsicBody) []).length
      rw [uifEmit_length]; simp; omega
    · have heval : WStmt.eval fuel
          (.uif pcond (wseq [ .bin .sub eReg srcReg (.imm 15), wEmitLSIC eReg ]) .skip) ws = ws := by
        simp [WStmt.eval, hb]
      rw [heval]; exact couple_setPc (couple_setPc (couple_setPc hc _) _) _
    · refine allSteps_seq (allSteps_seq ?_ (sreaches_one_eq s0) ?_) q2 ?_
      · exact allSteps_one
          (by simp only [PcIn, hpc, uifEmit_length]; omega)
          (by rw [s0]; simp only [PcIn, SState.setPc, uifEmit_length, lsicThen, lsicEmit,
                uwhileEmit_length, List.length_append, List.length_cons, List.length_nil]; omega)
      · exact allSteps_one
          (by simp only [PcIn, SState.setPc, uifEmit_length, lsicThen, lsicEmit,
                uwhileEmit_length, List.length_append, List.length_cons, List.length_nil]; omega)
          (by rw [s1]; simp only [PcIn, SState.setPc, uifEmit_length, lsicThen, lsicEmit,
                uwhileEmit_length, List.length_append, List.length_cons, List.length_nil]; omega)
      · exact allSteps_one
          (by simp only [PcIn, SState.setPc, uifEmit_length, lsicThen, lsicEmit,
                uwhileEmit_length, List.length_append, List.length_cons, List.length_nil]; omega)
          (by rw [s2]; simp only [PcIn, SState.setPc, uifEmit_length, lsicThen, lsicEmit,
                uwhileEmit_length, List.length_append, List.length_cons, List.length_nil]; omega)


/-- The `AllSteps` corollary: a branch-free segment is pc-confined to its own range. -/
theorem allSteps_pcIn_straight {p : Array SInstr} {emit : List SInstr} {base : Nat}
    {ss : SState} (hseg : SegAt p base emit) (hpc : ss.pc = base)
    (hstr : ∀ i ∈ emit, IsStraight i = true) (n : Nat) (hn : n ≤ emit.length) :
    AllSteps p (PcIn base (base + emit.length)) n ss := by
  intro j hj
  have := siter_pc_straight hseg hpc hstr j (by omega)
  exact ⟨by omega, by omega⟩


/-- **Transcribed from `simSL_coopExtendStep`, carrying pc-confinement.**  The
    extend step's emit is eighteen branch-free instructions, so `allSteps_pcIn_straight`
    discharges the obligation from `SegAt` alone. -/
theorem simSLQ_coopExtendStep_pcIn (R : List String) (inStride endCap : Nat)
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
          ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base + (coopExtendEmit "adv").length)) n ss := by
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
  refine ⟨18, snsteps prog 18 ss, hreach, ?_, ?_, hmi', ?_⟩
  · rw [hpc18]; rfl
  · have heval : WStmt.eval fuel (WStmt.coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap) ws
      = evalCoopExtendStep "adv" "p0" "cand0" "ml" inStride endCap ws := by
      simp only [WStmt.eval]
    rw [heval]; exact hcpl
  · exact allSteps_pcIn_straight hseg hpc (by decide) 18 (by decide)


/-- **Transcribed from `simSL_coopWindow`, carrying pc-confinement.**  Fifty-one
    branch-free instructions, so `allSteps_pcIn_straight` closes it from `SegAt`. -/
theorem simSLQ_coopWindow_pcIn (R : List String) (inStride searchLim hashLog s : Nat)
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
          ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base +
          (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride searchLim hashLog).length)) n ss := by
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
  refine ⟨51, snsteps prog 51 ss, hreach, ?_, ?_, hmi',
    allSteps_pcIn_straight hseg hpc (fun i hi => List.all_eq_true.mp (by rfl) i hi) 51
      (Nat.le_of_eq (by rfl))⟩
  · rw [hpc51]; rfl
  · have heval : WStmt.eval fuel
        (WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0) ws
      = evalCoopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0 ws := by
      simp only [WStmt.eval]
    rw [heval]; exact hcpl

/-- **Transcribed from `coopWindow_couple_found`, carrying pc-confinement.** -/
theorem coopWindow_couple_found_pcIn (R0 : List String) (inStride searchLim hashLog s p c : Nat)
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
          ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base +
          (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride searchLim hashLog).length)) n ss := by
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
  refine ⟨51, snsteps prog 51 ss, hreach, by rw [hpc51]; rfl, ?_, hmi',
    allSteps_pcIn_straight hseg hpc (fun i hi => List.all_eq_true.mp (by rfl) i hi) 51
      (Nat.le_of_eq (by rfl))⟩
  apply Couple.extend2 R0 (snsteps prog 51 ss) _ "p0" "cand0" hcpl
  · intro l; rw [hp0m l, hp0e]
  · intro l; rw [hcand0m l, hcand0e]

-- ── `simSL_coopCopy`: lift `coopCopy_couple` (with its exposed frame) ────────────


/-- **Transcribed from `CoopCopyModel.coopCopy_loop`, carrying pc-confinement.** -/
theorem coopCopy_loop_pcIn (prog : Array SInstr) (base : Nat) (lH lX dst src len : String)
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
        (∀ r : String, r ∉ coopCopyScratch → ∀ l : Fin 32, ss'.regs r l = ss.regs r l) ∧
        AllSteps prog (PcIn base (base +
          (uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)).length)) n ss := by
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
  have hLEN : (uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)).length = 14 := by
    rw [uwhileEmit_length, coopCopyBody_length]
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
        (base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1), ?_, ?_, ?_, ?_, ?_, ?_⟩
    · exact sreaches_trans prog 2 1 _ _ _
        (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq s0) (sreaches_one_eq s1)) (sreaches_one_eq s2)
    · show (base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1)
        = base + (uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)).length
      simp [uwhileEmit_length]; omega
    · show cpLoop dst0 src0 len0 0 i ss.gmem = ss.gmem
      simp [cpLoop]
    · rfl
    · intro r hr l; rfl
    · rw [hLEN]
      intro j hj
      have e1 : siter prog 1 ss = ss.setPc (ss.pc + 1) := by rw [siter_succ]; exact s0
      have e2 : siter prog 2 ss = (ss.setPc (ss.pc + 1)).setPc (sfindLabel prog lX) := by
        rw [siter_succ, e1]; exact s1
      have e3 : siter prog 3 ss
          = (((ss.setPc (ss.pc+1)).setPc (sfindLabel prog lX))).setPc
              (base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1) := by
        rw [siter_succ, e2]; exact s2
      have hj4 : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 := by omega
      rcases hj4 with rfl | rfl | rfl | rfl
      · show base ≤ ss.pc ∧ ss.pc ≤ base + 14
        rw [hpc]; omega
      · rw [e1]
        show base ≤ ss.pc + 1 ∧ ss.pc + 1 ≤ base + 14
        rw [hpc]; omega
      · rw [e2]
        show base ≤ sfindLabel prog lX ∧ sfindLabel prog lX ≤ base + 14
        rw [hLend, coopCopyBody_length]; omega
      · rw [e3]
        show base ≤ base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1
          ∧ base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1 ≤ base + 14
        rw [coopCopyBody_length]; omega
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
      obtain ⟨n, ssf, hrf, hpcf, hgf, hsf, hff, hstf⟩ := hrec
      refine ⟨1 + 1 + 10 + 1 + n, ssf, ?_, hpcf, ?_, ?_, ?_, ?_⟩
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
      · rw [hLEN] at hstf ⊢
        refine allSteps_seq (allSteps_seq (allSteps_seq
          (allSteps_seq (allSteps_one ?_ ?_) (sreaches_one_eq s0) (allSteps_one ?_ ?_))
            (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq s0) (sreaches_one_eq s1)) ?_)
          (sreaches_trans prog (1 + 1) 10 _ _ _
            (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq s0) (sreaches_one_eq s1))
            (sreaches_snsteps prog 10 ssB)) (allSteps_one ?_ ?_))
          (sreaches_trans prog (1 + 1 + 10) 1 _ _ _
            (sreaches_trans prog (1 + 1) 10 _ _ _
              (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq s0) (sreaches_one_eq s1))
              (sreaches_snsteps prog 10 ssB)) (sreaches_one_eq s2')) hstf
        · show base ≤ ss.pc ∧ ss.pc ≤ base + 14
          rw [hpc]; omega
        · rw [s0]
          show base ≤ ss.pc + 1 ∧ ss.pc + 1 ≤ base + 14
          rw [hpc]; omega
        · show base ≤ ss.pc + 1 ∧ ss.pc + 1 ≤ base + 14
          rw [hpc]; omega
        · rw [s1]
          show base ≤ ssB.pc ∧ ssB.pc ≤ base + 14
          rw [hpcbodyB]; omega
        · exact allSteps_weaken (fun st h => ⟨by have := h.1; omega,
            by have := h.2
               rw [coopCopyBody_length] at this
               omega⟩)
            (allSteps_pcIn_straight hsegBody hpcbodyB
              (fun i hi => List.all_eq_true.mp (by rfl) i hi) 10
              (Nat.le_of_eq (coopCopyBody_length dst src len).symm))
        · show base ≤ (snsteps prog 10 ssB).pc ∧ (snsteps prog 10 ssB).pc ≤ base + 14
          rw [hiterPc, hpcbodyB]; omega
        · rw [s2']
          show base ≤ base ∧ base ≤ base + 14
          omega
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
          (base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1), ?_, ?_, ?_, ?_, ?_, ?_⟩
      · exact sreaches_trans prog 2 1 _ _ _
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq s0) (sreaches_one_eq s1)) (sreaches_one_eq s2)
      · show (base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1)
          = base + (uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)).length
        simp [uwhileEmit_length]; omega
      · simp only [SState.setPc]
        rw [cpLoop, if_neg hb]
      · rfl
      · intro r hr l; rfl
      · rw [hLEN]
        intro j hj
        have e1 : siter prog 1 ss = ss.setPc (ss.pc + 1) := by rw [siter_succ]; exact s0
        have e2 : siter prog 2 ss = (ss.setPc (ss.pc + 1)).setPc (sfindLabel prog lX) := by
          rw [siter_succ, e1]; exact s1
        have e3 : siter prog 3 ss
            = (((ss.setPc (ss.pc+1)).setPc (sfindLabel prog lX))).setPc
                (base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1) := by
          rw [siter_succ, e2]; exact s2
        have hj4 : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 := by omega
        rcases hj4 with rfl | rfl | rfl | rfl
        · show base ≤ ss.pc ∧ ss.pc ≤ base + 14
          rw [hpc]; omega
        · rw [e1]
          show base ≤ ss.pc + 1 ∧ ss.pc + 1 ≤ base + 14
          rw [hpc]; omega
        · rw [e2]
          show base ≤ sfindLabel prog lX ∧ sfindLabel prog lX ≤ base + 14
          rw [hLend, coopCopyBody_length]; omega
        · rw [e3]
          show base ≤ base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1
            ∧ base + 1 + 1 + (coopCopyBody dst src len).length + 1 + 1 ≤ base + 14
          rw [coopCopyBody_length]; omega



/-- **Transcribed from `coopCopy_couple`, carrying pc-confinement.** -/
theorem coopCopy_couple_pcIn (R : List String) (dst src len lH lX : String)
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
      (∀ r : String, r ∉ coopCopyScratch → ∀ l : Fin 32, ss'.regs r l = ss.regs r l) ∧
      AllSteps prog (PcIn base (base +
        (([.mov "cpI" (.imm 0), .setp .lt "cpCont" "cpI" (.reg len)] : List SInstr)
          ++ uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)).length)) n ss := by
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
  obtain ⟨n, ss', hreach, hpcE, hgmemE, hsmemE, hframeE, hstL⟩ :=
    coopCopy_loop_pcIn prog (ss.pc + 1 + 1) lH lX dst src len
      (ws.regs dst).toNat (ws.regs src).toNat (ws.regs len).toNat
      hsegLoop hlrLoop
      (hRdisj dst hdst) (hRdisj src hsrc) (hRdisj len hlenR)
      hb1 hb2 hb3
      (ws.regs len).toNat 0 ss2 hpc2 (by omega)
      hcpI2 hcpCont2 hdst2 hsrc2 hlen2 hlane2 (by omega)
  refine ⟨n + 2, ss', ?_, ?_, ⟨?_, ?_, ?_⟩, ?_, ?_⟩
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
  · have hE : (([.mov "cpI" (.imm 0), .setp .lt "cpCont" "cpI" (.reg len)] : List SInstr)
        ++ uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)).length = 16 := by
      simp only [List.length_append, List.length_cons, List.length_nil, uwhileEmit_length,
        coopCopyBody_length]
    rw [hE]
    have q0 : PcIn ss.pc (ss.pc + 16) ss := ⟨Nat.le_refl _, by omega⟩
    have q1 : PcIn ss.pc (ss.pc + 16) ((ss.setReg "cpI" (fun _ => 0)).setPc (ss.pc + 1)) := by
      show ss.pc ≤ ss.pc + 1 ∧ ss.pc + 1 ≤ ss.pc + 16
      omega
    have q2 : PcIn ss.pc (ss.pc + 16) ss2 := by
      show ss.pc ≤ ss2.pc ∧ ss2.pc ≤ ss.pc + 16
      rw [hpc2]; omega
    have hR1 : SReaches prog 1 ss ((ss.setReg "cpI" (fun _ => 0)).setPc (ss.pc + 1)) :=
      sreaches_one_eq s0
    have hR2 : SReaches prog 1 ((ss.setReg "cpI" (fun _ => 0)).setPc (ss.pc + 1)) ss2 :=
      sreaches_one_eq (s1.trans hss2.symm)
    have hAS1 : AllSteps prog (PcIn ss.pc (ss.pc + 16)) 1 ss :=
      allSteps_one q0 (by rw [s0]; exact q1)
    have hAS2 : AllSteps prog (PcIn ss.pc (ss.pc + 16)) 1
        ((ss.setReg "cpI" (fun _ => 0)).setPc (ss.pc + 1)) :=
      allSteps_one q1 (by rw [s1.trans hss2.symm]; exact q2)
    have hAS12 : AllSteps prog (PcIn ss.pc (ss.pc + 16)) (1 + 1) ss :=
      allSteps_seq hAS1 hR1 hAS2
    have hfin : AllSteps prog (PcIn ss.pc (ss.pc + 16)) (1 + 1 + n) ss :=
      allSteps_seq (n₁ := 1 + 1) (n₂ := n) hAS12 (sreaches_trans prog 1 1 _ _ _ hR1 hR2)
        (allSteps_weaken (fun st h => ⟨by have := h.1; omega,
          by have := h.2
             rw [uwhileEmit_length, coopCopyBody_length] at this
             omega⟩) hstL)
    intro j hj
    exact hfin j (by omega)


/-- **Transcribed from `simSL_coopCopy`, carrying pc-confinement.** -/
theorem simSLQ_coopCopy_pcIn (R : List String) (dst src len lH lX : String)
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
        Couple R ss' ((WStmt.coopCopy dst src len).eval fuel ws) ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base +
          (([.mov "cpI" (.imm 0), .setp .lt "cpCont" "cpI" (.reg len)] : List SInstr)
            ++ uwhileEmit "cpCont" lH lX (coopCopyBody dst src len)).length)) n ss := by
  intro prog base ss fuel hpc hseg hlr hc hmi
  obtain ⟨n, ss', hreach, hpcE, hcpl, hframeE, hst⟩ :=
    coopCopy_couple_pcIn R dst src len lH lX prog base ss ws fuel hpc hseg hlr hc
      hdst hsrc hlenR hRdisj (fun l => hmi.1 l) hb1 hb2 hb3 hdisjW hsizeW
  refine ⟨n, ss', hreach, hpcE, hcpl, ?_, hst⟩
  refine ⟨fun l => ?_, fun l => ?_, fun l => ?_⟩
  · rw [hframeE "lane" (by simp [coopCopyScratch]) l]; exact hmi.1 l
  · rw [hframeE "inBase" (by simp [coopCopyScratch]) l]; exact hmi.2.1 l
  · rw [hframeE "tbl" (by simp [coopCopyScratch]) l]; exact hmi.2.2 l


/-- **Transcribed from `simSL'_extCBody`, carrying pc-confinement.** -/
theorem simSLQ_extCBody_pcIn (R : List String) (inStride endCap : Nat)
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
          ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base +
          (coopExtendEmit "adv"
            ++ (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
              ++ ([.setp .eq "extC" "adv" (.imm 32)] : List SInstr))).length)) m ss := by
  intro prog base ss fuel hpc hseg hlr hc hmi
  have hsegCoop : SegAt prog base (coopExtendEmit "adv") := hseg.append_left
  have hlrCoop : LabelsResolve prog base (coopExtendEmit "adv") := hlr.append_left
  -- Step 1: the coop-extend segment (18 instrs), via the bespoke wrapper.
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1, hst1⟩ :=
    simSLQ_coopExtendStep_pcIn R inStride endCap hctx hendCap hRdisj ws hboundW hbytesW
      prog base ss fuel hpc hsegCoop hlrCoop hc hmi
  have hcoopLen : (coopExtendEmit "adv").length = 18 := by decide
  -- Step 2/3: `bin add ml` then `setp eq extC` (both plain `SimSL'` leaves).
  have hsegRest : SegAt prog (base + (coopExtendEmit "adv").length)
      (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
        ++ [.setp .eq "extC" "adv" (.imm 32)]) := hseg.append_right
  have hlrRest : LabelsResolve prog (base + (coopExtendEmit "adv").length)
      (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
        ++ [.setp .eq "extC" "adv" (.imm 32)]) := hlr.append_right
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2, hst2⟩ :=
    (simSLQ_seq ib R PcIn pcIn_mono (.bin .add "ml" "ml" (.reg "adv")) (.setp .eq "extC" "adv" (.imm 32))
      _ _
      (simSLQ_bin_pcIn ib R .add "ml" "ml" (.reg "adv") hml (fun m h => by cases h; exact hadv) (by decide))
      (simSLQ_setp_pcIn ib R .eq "extC" "adv" (.imm 32) hadv (fun m h => by cases h) (by decide)))
      prog (base + (coopExtendEmit "adv").length) ss1
      (evalCoopExtendStep "adv" "p0" "cand0" "ml" inStride endCap ws) fuel
      (by rw [hpc1]) hsegRest hlrRest
      (by have : WStmt.eval fuel (WStmt.coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap) ws
            = evalCoopExtendStep "adv" "p0" "cand0" "ml" inStride endCap ws := by simp only [WStmt.eval]
          rw [← this]; exact hc1)
      hmi1
  refine ⟨n1 + n2, ss2, sreaches_trans prog n1 n2 _ _ _ hr1 hr2, ?_, ?_, hmi2, ?_⟩
  · rw [hpc2]; simp only [List.length_append, List.length_cons, List.length_nil]; omega
  · have heval : (wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
          .bin .add "ml" "ml" (.reg "adv"), .setp .eq "extC" "adv" (.imm 32) ]).eval fuel ws
        = (WStmt.seq (.bin .add "ml" "ml" (.reg "adv")) (.setp .eq "extC" "adv" (.imm 32))).eval fuel
            (evalCoopExtendStep "adv" "p0" "cand0" "ml" inStride endCap ws) := by
      simp only [wseq, WStmt.eval]
    rw [heval]; exact hc2
  · refine allSteps_seq (allSteps_weaken ?_ hst1) hr1 (allSteps_weaken ?_ hst2)
    · intro st h
      refine ⟨by have := h.1; omega, ?_⟩
      have := h.2
      simp only [List.length_append, List.length_cons, List.length_nil] at *
      omega
    · intro st h
      refine ⟨by have := h.1; omega, ?_⟩
      have := h.2
      simp only [List.length_append, List.length_cons, List.length_nil] at *
      omega



/-- **Transcribed from `simSL'_notFoundBranch`, carrying pc-confinement.** -/
theorem simSLQ_notFoundBranch_pcIn (ib : Nat) (R : List String) (hsp : "searchPos" ∈ R) :
    SimSLQ ib R PcIn (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ])
      ([.bin .add "searchPos" "searchPos" (.imm 32)] : List SInstr) := by
  have h := simSLQ_bin_pcIn ib R .add "searchPos" "searchPos" (.imm 32) hsp
    (fun n h => by cases h) (by decide)
  simpa [wseq] using h


/-- **Transcribed from `simSL'_wEmitFinalSeq`, carrying pc-confinement.** -/
theorem simSLQ_wEmitFinalSeq_pcIn (ib : Nat) (R : List String) (litStart litLen : String)
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
        Couple R ss' ((wEmitFinalSeq litStart litLen).eval fuel ws) ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base +
          (wEmitFinalSeqEmit litStart litLen lElseF lEndF lHF lXF cpH cpX lsicBody).length)) m ss := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi hgd hfl hb1 hb2 hb3 hdisj hsz
  rw [wEmitFinalSeqEmit] at hseg hlr
  -- Step 1: prefix, via `simSL'_finalPrefix`.
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1, hst1⟩ :=
    (simSLQ_finalPrefix_pcIn ib R litStart litLen hll hzero hout hop hsb htokHi htok hpb)
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
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2, hst2⟩ :=
    simSLQ_lsicUif_pcIn ib R "pLitBigF" litLen "litExtraF" lElseF lEndF lHF lXF hpb hle hll
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
  obtain ⟨n3, ss3, hr3, hpc3, hc3, hmi3, hst3⟩ :=
    (simSLQ_finalMid_pcIn ib R litStart hout hop hib hls hcd hcs)
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
  obtain ⟨n4, ss4, hr4, hpc4, hc4, hmi4, hst4⟩ :=
    simSLQ_coopCopy_pcIn R "cpDstF" "cpSrcF" litLen cpH cpX hcd hcs hll hRcp
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
  obtain ⟨n5, ss5, hr5, hpc5, hc5, hmi5, hst5⟩ :=
    (simSLQ_bin_pcIn ib R .add "op" "op" (.reg litLen) hop (fun n h => by cases h; exact hll) (by decide))
      prog _ ss4 coopSt fuel hpc4' hsegTail hlrTail hc4 hmi4
  -- assemble.
  refine ⟨n1 + (n2 + (n3 + (n4 + n5))), ss5,
    sreaches_trans prog n1 _ _ _ _ hr1
      (sreaches_trans prog n2 _ _ _ _ hr2
        (sreaches_trans prog n3 _ _ _ _ hr3
          (sreaches_trans prog n4 n5 _ _ _ hr4 hr5))), ?_, ?_, hmi5, ?_⟩
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
  · refine allSteps_seq (allSteps_weaken ?_ hst1) hr1
      (allSteps_seq (allSteps_weaken ?_ hst2) hr2
        (allSteps_seq (allSteps_weaken ?_ hst3) hr3
          (allSteps_seq (allSteps_weaken ?_ hst4) hr4 (allSteps_weaken ?_ hst5))))
    all_goals
      (intro st h
       obtain ⟨hA, hB⟩ := h
       simp only [wEmitFinalSeqEmit, finalPreEmit, finalMidEmit, finalCoopEmit,
         uifEmit, lsicThen, lsicEmit, uwhileEmit_length,
         List.length_append, List.length_cons, List.length_nil] at hA hB ⊢
       exact ⟨by omega, by omega⟩)



/-- **Transcribed from `simSL'_wEmitMatchSeq`, carrying pc-confinement.** -/
theorem simSLQ_wEmitMatchSeq_pcIn (ib : Nat) (R : List String) (litStart litLen off ml : String)
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
        Couple R ss' ((wEmitMatchSeq litStart litLen off ml).eval fuel ws) ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base +
          (wEmitMatchSeqEmit litStart litLen off ml lElseL lEndL lHL lXL cpH cpX
            lElseM lEndM lHM lXM lsicL lsicM).length)) m ss := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi hgdL hflL hb1 hb2 hb3 hdisj hsz hgdM hflM
  rw [wEmitMatchSeqEmit] at hseg hlr
  -- Step 1: matchPre.
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1, hst1⟩ :=
    (simSLQ_matchPre_pcIn ib R litLen ml hll hml hmlm htokLo hout hop hsb htokHi htok hpb)
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
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2, hst2⟩ :=
    simSLQ_lsicUif_pcIn ib R "pLitBig" litLen "litExtra" lElseL lEndL lHL lXL hpb hle hll
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
  obtain ⟨n3, ss3, hr3, hpc3, hc3, hmi3, hst3⟩ :=
    (simSLQ_matchMid_pcIn ib R litStart hout hop hib hls hcd hcs)
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
  obtain ⟨n4, ss4, hr4, hpc4, hc4, hmi4, hst4⟩ :=
    simSLQ_coopCopy_pcIn R "cpDst" "cpSrc" litLen cpH cpX hcd hcs hll hRcp
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
  obtain ⟨n5, ss5, hr5, hpc5, hc5, hmi5, hst5⟩ :=
    (simSLQ_matchOff_pcIn ib R litLen off hll hoff hmlm hoffLo hoffHi hout hop hsb hpm)
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
  obtain ⟨n6, ss6, hr6, hpc6, hc6, hmi6, hst6⟩ :=
    simSLQ_lsicUif_pcIn ib R "pMatBig" "mlm" "matExtra" lElseM lEndM lHM lXM hpm hme hmlm
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
            (sreaches_trans prog n5 n6 _ _ _ hr5 hr6)))), ?_, ?_, hmi6, ?_⟩
  · rw [hpc6, wEmitMatchSeqEmit]
    simp only [List.length_append, List.length_cons, List.length_nil, Nat.add_assoc]
  · rw [wEmitMatchSeq_eval_seq]
    rw [matchAfterMatSetp, matchCpEntry, matchAfterSetp] at hc6
    exact hc6
  · refine allSteps_seq (allSteps_weaken ?_ hst1) hr1
      (allSteps_seq (allSteps_weaken ?_ hst2) hr2
        (allSteps_seq (allSteps_weaken ?_ hst3) hr3
          (allSteps_seq (allSteps_weaken ?_ hst4) hr4
            (allSteps_seq (allSteps_weaken ?_ hst5) hr5 (allSteps_weaken ?_ hst6)))))
    all_goals
      (intro st h
       obtain ⟨hA, hB⟩ := h
       simp only [wEmitMatchSeqEmit, matchPreEmit, matchMidEmit, matchCoopEmit, matchOffEmit,
         uifEmit, lsicThen, lsicEmit, uwhileEmit_length,
         List.length_append, List.length_cons, List.length_nil] at hA hB ⊢
       exact ⟨by omega, by omega⟩)




/-- **Transcribed from `bodySimInv_extCBody` / `extLoop_sim`, carrying pc-confinement.** -/
theorem bodySimInvQ_extCBody_pcIn (ib : Nat) (R : List String) (inStride endCap : Nat)
    (hctx : ExtCtx R) (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hib40 : ib < 2 ^ 40)
    (hml : "ml" ∈ R) (hadv : "adv" ∈ R) (hextC : "extC" ∈ R)
    (hRdisj : ∀ r ∈ R, r ∉ ["idx", "pe", "pIn", "peC", "dfe", "caC", "peD", "aP", "caD", "aC",
               "bP", "bC", "pEqB", "pOk", "balOk", "mis", "revM"]) :
    BodySimInvQ ib R PcIn (extInv inStride endCap)
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
  have hsim := simSLQ_extCBody_pcIn R inStride endCap hctx hendCap hml hadv hextC hRdisj ws
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

theorem extLoop_sim_pcIn (ib : Nat) (R : List String) (inStride endCap : Nat)
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
          ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base + (uwhileEmit "extC" lHead lEnd
          (coopExtendEmit "adv" ++ (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
            ++ [.setp .eq "extC" "adv" (.imm 32)]))).length)) n ss := by
  intro fuel ss ws hpc hc hmi hInv hfuel
  exact simSLQ_uwhileBodyInv_pcIn ib R "extC" lHead lEnd (extInv inStride endCap)
    (extBodyStmt inStride endCap)
    (coopExtendEmit "adv" ++ (([.bin .add "ml" "ml" (.reg "adv")] : List SInstr)
      ++ [.setp .eq "extC" "adv" (.imm 32)]))
    prog base hextC
    (by unfold extBodyStmt
        exact bodySimInvQ_extCBody_pcIn ib R inStride endCap hctx hendCap hlen hib40 hml hadv hextC hRdisj)
    hseg hlr fuel ss ws hpc hc hmi hInv
    (extLoop_halts inStride endCap hendCap hlen fuel ws hInv hfuel)


/-- **Transcribed from `simSL'_foundBranch`, carrying pc-confinement.** -/
theorem simSLQ_foundBranch_pcIn (ib : Nat) (R : List String) (inStride endCap : Nat)
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
        Couple R ss' ((foundBranchStmt inStride endCap).eval fuel ws) ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base + (foundBranchEmit endCap lHE lEE lElseL lEndL lHL lXL
          cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length)) m ss := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi hcand hp4 hfuelExt
    hgdL hflL hb1 hb2 hb3 hdisj hsz hgdM hflM
  rw [foundBranchEmit] at hseg hlr
  -- Segment decomposition of the emit.
  have hsegMovs := hseg.append_left
  have hlrMovs := hlr.append_left
  have hsegRest := hseg.append_right
  have hlrRest := hlr.append_right
  -- Step 1: found-setup movs.
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1, hst1⟩ :=
    (simSLQ_foundMovs_pcIn ib R endCap hctx.ecR hctx.ec1 hctx.ml hextC)
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
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2, hst2⟩ :=
    extLoop_sim_pcIn ib R inStride endCap hctx hendCap hlen hib40 hctx.ml hctx.adv hextC hRdisj lHE lEE
      prog (base + 4) hsegExt hlrExt fuel ss1 (foundExtEntry endCap fuel ws)
      hpc1' hc1' hmi1 hextInv hfuelExt'
  have hc2' : Couple R ss2 (foundExtDone inStride endCap fuel ws) := hc2
  have hpc2' : ss2.pc = base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length := hpc2
  -- Step 3: the two address subs.
  have hsegAfterExt := hsegRest.append_right
  have hlrAfterExt := hlrRest.append_right
  obtain ⟨n3, ss3, hr3, hpc3, hc3, hmi3, hst3⟩ :=
    (simSLQ_foundSubs_pcIn ib R "litAnchor" hctx.p0 hctx.cand0 hla hoff0 hlitLen)
      prog _ ss2 (foundExtDone inStride endCap fuel ws) fuel hpc2'
      hsegAfterExt.append_left hlrAfterExt.append_left hc2' hmi2
  have hc3' : Couple R ss3 (foundMatchEntry inStride endCap fuel ws) := hc3
  have hpc3' : ss3.pc = base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2 := hpc3
  -- Step 4: the match-sequence emit.
  have hsegAfterSubs := hsegAfterExt.append_right
  have hlrAfterSubs := hlrAfterExt.append_right
  obtain ⟨n4, ss4, hr4, hpc4, hc4, hmi4, hst4⟩ :=
    simSLQ_wEmitMatchSeq_pcIn ib R "litAnchor" "litLen" "off0" "ml"
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
  obtain ⟨n5, ss5, hr5, hpc5, hc5, hmi5, hst5⟩ :=
    (simSLQ_foundUpdates_pcIn ib R hctx.p0 hctx.ml hla hsp)
      prog _ ss4 matchSt fuel hpc4' hsegAfterMatch hlrAfterMatch hc4 hmi4
  -- assemble.
  refine ⟨n1 + (n2 + (n3 + (n4 + n5))), ss5,
    sreaches_trans prog n1 _ _ _ _ hr1
      (sreaches_trans prog n2 _ _ _ _ hr2
        (sreaches_trans prog n3 _ _ _ _ hr3
          (sreaches_trans prog n4 n5 _ _ _ hr4 hr5))), ?_, ?_, hmi5, ?_⟩
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
  · refine allSteps_seq (allSteps_weaken ?_ hst1) hr1
      (allSteps_seq (allSteps_weaken ?_ hst2) hr2
        (allSteps_seq (allSteps_weaken ?_ hst3) hr3
          (allSteps_seq (allSteps_weaken ?_ hst4) hr4 (allSteps_weaken ?_ hst5))))
    all_goals
      (intro st h
       obtain ⟨hA, hB⟩ := h
       simp only [foundBranchEmit, foundExtBodyEmit, wEmitMatchSeqEmit, matchPreEmit,
         matchMidEmit, matchCoopEmit,
         matchOffEmit, uifEmit, lsicThen, lsicEmit, uwhileEmit_length,
         List.length_append, List.length_cons, List.length_nil] at hA hB ⊢
       exact ⟨by omega, by omega⟩)


/-- **The found branch, additionally exporting the coupling at the token-emit entry.**

    Same statement and same proof as `simSLQ_foundBranch_pcIn`, plus the checkpoint
    `st.pc = MB → Couple R st (foundMatchEntry ..)` where `MB` is the base of
    `wEmitMatchSeqEmit`.  It is a separate declaration rather than a strengthening
    so that the existing callers are untouched; the two side conditions
    (`MB`'s successors lie above it, and nothing in the match segment branches back
    to it) are hypotheses here and `decide`d at the shipped kernel.

    The three pieces: `allSteps_ckpt_end` for the run-up (which ends at `MB` with
    the coupling in hand), `allSteps_ckpt_of_pcIn` for the match segment (which
    starts there), `allSteps_off_site` for the tail (which is above it). -/
theorem simSLQ_foundBranch_ckpt (ib : Nat) (R : List String) (inStride endCap : Nat)
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
      (∀ q', q' ∈ succsOf prog (base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2) → base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2 < q') →
      (∀ q', base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2 ≤ q' →
        q' ≤ base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2 + (wEmitMatchSeqEmit "litAnchor" "litLen" "off0" "ml" lElseL lEndL lHL lXL cpH cpX
          lElseM lEndM lHM lXM lsicL lsicM).length →
        base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2 ∉ succsOf prog q') →
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
        Couple R ss' ((foundBranchStmt inStride endCap).eval fuel ws) ∧ MachInv ib ss' ∧
        AllSteps prog (PcIn base (base + (foundBranchEmit endCap lHE lEE lElseL lEndL lHL lXL
          cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length)) m ss ∧
        AllSteps prog (fun st => st.pc = (base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2) →
          Couple R st (foundMatchEntry inStride endCap fuel ws)) m ss := by
  intro prog base ss ws fuel hpc hseg hlr hc hmi hMBtop hMBno hcand hp4 hfuelExt
    hgdL hflL hb1 hb2 hb3 hdisj hsz hgdM hflM
  rw [foundBranchEmit] at hseg hlr
  -- Segment decomposition of the emit.
  have hsegMovs := hseg.append_left
  have hlrMovs := hlr.append_left
  have hsegRest := hseg.append_right
  have hlrRest := hlr.append_right
  -- Step 1: found-setup movs.
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1, hst1⟩ :=
    (simSLQ_foundMovs_pcIn ib R endCap hctx.ecR hctx.ec1 hctx.ml hextC)
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
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2, hst2⟩ :=
    extLoop_sim_pcIn ib R inStride endCap hctx hendCap hlen hib40 hctx.ml hctx.adv hextC hRdisj lHE lEE
      prog (base + 4) hsegExt hlrExt fuel ss1 (foundExtEntry endCap fuel ws)
      hpc1' hc1' hmi1 hextInv hfuelExt'
  have hc2' : Couple R ss2 (foundExtDone inStride endCap fuel ws) := hc2
  have hpc2' : ss2.pc = base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length := hpc2
  -- Step 3: the two address subs.
  have hsegAfterExt := hsegRest.append_right
  have hlrAfterExt := hlrRest.append_right
  obtain ⟨n3, ss3, hr3, hpc3, hc3, hmi3, hst3⟩ :=
    (simSLQ_foundSubs_pcIn ib R "litAnchor" hctx.p0 hctx.cand0 hla hoff0 hlitLen)
      prog _ ss2 (foundExtDone inStride endCap fuel ws) fuel hpc2'
      hsegAfterExt.append_left hlrAfterExt.append_left hc2' hmi2
  have hc3' : Couple R ss3 (foundMatchEntry inStride endCap fuel ws) := hc3
  have hpc3' : ss3.pc = base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2 := hpc3
  -- Step 4: the match-sequence emit.
  have hsegAfterSubs := hsegAfterExt.append_right
  have hlrAfterSubs := hlrAfterExt.append_right
  obtain ⟨n4, ss4, hr4, hpc4, hc4, hmi4, hst4⟩ :=
    simSLQ_wEmitMatchSeq_pcIn ib R "litAnchor" "litLen" "off0" "ml"
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
  obtain ⟨n5, ss5, hr5, hpc5, hc5, hmi5, hst5⟩ :=
    (simSLQ_foundUpdates_pcIn ib R hctx.p0 hctx.ml hla hsp)
      prog _ ss4 matchSt fuel hpc4' hsegAfterMatch hlrAfterMatch hc4 hmi4
  -- assemble.
  refine ⟨n1 + (n2 + (n3 + (n4 + n5))), ss5,
    sreaches_trans prog n1 _ _ _ _ hr1
      (sreaches_trans prog n2 _ _ _ _ hr2
        (sreaches_trans prog n3 _ _ _ _ hr3
          (sreaches_trans prog n4 n5 _ _ _ hr4 hr5))), ?_, ?_, hmi5, ?_, ?_⟩
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
  · refine allSteps_seq (allSteps_weaken ?_ hst1) hr1
      (allSteps_seq (allSteps_weaken ?_ hst2) hr2
        (allSteps_seq (allSteps_weaken ?_ hst3) hr3
          (allSteps_seq (allSteps_weaken ?_ hst4) hr4 (allSteps_weaken ?_ hst5))))
    all_goals
      (intro st h
       obtain ⟨hA, hB⟩ := h
       simp only [foundBranchEmit, foundExtBodyEmit, wEmitMatchSeqEmit, matchPreEmit,
         matchMidEmit, matchCoopEmit,
         matchOffEmit, uifEmit, lsicThen, lsicEmit, uwhileEmit_length,
         List.length_append, List.length_cons, List.length_nil] at hA hB ⊢
       exact ⟨by omega, by omega⟩)



  · have W : ∀ (n : Nat) (st0 : SState) (lo hi : Nat),
        AllSteps prog (PcIn lo hi) n st0 → base ≤ lo →
        hi ≤ (base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2) →
        AllSteps prog (PcIn base (base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2)) n st0 :=
      fun n st0 lo hi h h1 h2 => allSteps_weaken
        (fun st hh => ⟨by have := hh.1; omega, by have := hh.2; omega⟩) h
    have hrA : SReaches prog (n1 + (n2 + n3)) ss ss3 :=
      sreaches_trans prog n1 (n2 + n3) ss ss1 ss3 hr1
        (sreaches_trans prog n2 n3 ss1 ss2 ss3 hr2 hr3)
    have hASA : AllSteps prog (PcIn base (base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2)) (n1 + (n2 + n3)) ss :=
      allSteps_seq (n₁ := n1) (n₂ := n2 + n3)
        (W n1 ss _ _ hst1 (by omega)
          (by first | omega | (simp only [List.length_cons, List.length_nil]; omega))) hr1
        (allSteps_seq (n₁ := n2) (n₂ := n3)
          (W n2 ss1 _ _ hst2 (by omega)
            (by omega)) hr2
          (W n3 ss2 _ _ hst3 (by omega)
            (by first | omega | (simp only [List.length_cons, List.length_nil]; omega))))
    have hA := allSteps_ckpt_end (P := fun st => Couple R st
      (foundMatchEntry inStride endCap fuel ws)) hASA hrA hMBtop hc3'
    have hB := allSteps_ckpt_of_pcIn (P := fun st => Couple R st
      (foundMatchEntry inStride endCap fuel ws)) hst4 hc3' hMBno
    have hC : AllSteps prog (fun st => st.pc = (base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2) →
        Couple R st (foundMatchEntry inStride endCap fuel ws)) n5 ss4 :=
      allSteps_off_site (p := prog) (S := fun q => q = (base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2))
        (P := fun st => Couple R st (foundMatchEntry inStride endCap fuel ws))
        (n := n5) (ss := ss4)
        (pcIn_ne_of_lt _ hst5
          (by simp only [wEmitMatchSeqEmit, matchPreEmit, matchMidEmit, matchCoopEmit,
                matchOffEmit, uifEmit, lsicThen, lsicEmit, uwhileEmit_length,
                List.length_append, List.length_cons, List.length_nil]
              omega))
    have hfin : AllSteps prog (fun st => st.pc = (base + 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2) →
        Couple R st (foundMatchEntry inStride endCap fuel ws))
        ((n1 + (n2 + n3)) + (n4 + n5)) ss :=
      allSteps_seq (n₁ := n1 + (n2 + n3)) (n₂ := n4 + n5) hA hrA
        (allSteps_seq (n₁ := n4) (n₂ := n5) hB hr4 hC)
    intro j hj
    exact hfin j (by omega)


/-- **The output budget survives the found path.**  `foundMatchEntry` leaves
    `op`, `outBase`, `litAnchor` and `gmem` alone (`foundMatchEntry_frame` /
    `foundMatchEntry_gmem` — the found movs, the extend loop and the two subs all
    write elsewhere), so `LoopCQ`'s two budget clauses transfer verbatim to the
    token-emit entry.  This is the eval-side conjunct the `MB` checkpoint carries;
    it is what supplies `B` to `LZ4OpBound.op_le_matchAfterSetp` /
    `op_le_matchUifOut` at the stores inside the token. -/
theorem foundMatchEntry_budget (inStride endCap fuel : Nat) (ws : WState)
    (hbud32 : (ws.regs "outBase").toNat + (ws.regs "op").toNat
      + 9 * (inStride - (ws.regs "litAnchor").toNat) < 2 ^ 32)
    (hbudsz : (ws.regs "outBase").toNat + (ws.regs "op").toNat
      + 9 * (inStride - (ws.regs "litAnchor").toNat) ≤ ws.gmem.size) :
    ((foundMatchEntry inStride endCap fuel ws).regs "outBase").toNat
        + ((foundMatchEntry inStride endCap fuel ws).regs "op").toNat
        + 9 * (inStride - ((foundMatchEntry inStride endCap fuel ws).regs "litAnchor").toNat)
      < 2 ^ 32
    ∧ ((foundMatchEntry inStride endCap fuel ws).regs "outBase").toNat
        + ((foundMatchEntry inStride endCap fuel ws).regs "op").toNat
        + 9 * (inStride - ((foundMatchEntry inStride endCap fuel ws).regs "litAnchor").toNat)
      ≤ (foundMatchEntry inStride endCap fuel ws).gmem.size := by
  have hop : (foundMatchEntry inStride endCap fuel ws).regs "op" = ws.regs "op" :=
    foundMatchEntry_frame inStride endCap fuel ws "op" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have hob : (foundMatchEntry inStride endCap fuel ws).regs "outBase" = ws.regs "outBase" :=
    foundMatchEntry_frame inStride endCap fuel ws "outBase" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have hla : (foundMatchEntry inStride endCap fuel ws).regs "litAnchor" = ws.regs "litAnchor" :=
    foundMatchEntry_frame inStride endCap fuel ws "litAnchor" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have hgm : (foundMatchEntry inStride endCap fuel ws).gmem = ws.gmem :=
    foundMatchEntry_gmem inStride endCap fuel ws
  rw [hop, hob, hla, hgm]
  exact ⟨hbud32, hbudsz⟩

end AlgorithmLib.LZ4WarpDSL
