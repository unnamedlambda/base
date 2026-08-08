import AlgorithmLib.ML.BufsOf
import AlgorithmLib.ML.EmitFacts

/-!
# No kernel names a reserved register

`PTX_ADDR_SCRATCH` and the register above it are the emitter's own address
arithmetic.  A kernel that wrote either would compute its next address from a
value it had overwritten, so every emitted program is checked against them.

That check is a fold over the instruction stream, which costs one traversal per
kernel.  The emitter allocates upward from a counter, so a bound on the counter
and on the names a term carries is a bound on everything it emits — and that is
provable once rather than checked per kernel.

The two emitters that allocate registers are covered here: `emitP` for
expressions and `emitIdx` for addresses.  `emitEW` and `flatEW` sit above them
and still fold, so the guard on a whole kernel is decided rather than derived.
-/

namespace AlgorithmLib.ML

/-- The machine registers an expression reads. -/
def WFExp.machRegs : WFExp → List Nat
  | .reg r    => [r]
  | .lit _    => []
  | .add a b  => a.machRegs ++ b.machRegs
  | .mul a b  => a.machRegs ++ b.machRegs
  | .maxW a b => a.machRegs ++ b.machRegs
  | .geF a b  => a.machRegs ++ b.machRegs
  | .neg a    => a.machRegs
  | .inv a    => a.machRegs
  | .rsqrt a  => a.machRegs
  | .ex2 a    => a.machRegs
  | .exp a    => a.machRegs

theorem PReg.okB_tmp {k : Nat} (h : k < PTX_ADDR_SCRATCH) : (PReg.tmp k).okB = true := by
  simp [PReg.okB, h]

theorem PReg.okB_mach {k : Nat} (h : k < PTX_ADDR_SCRATCH) : (PReg.mach k).okB = true := by
  simp [PReg.okB, h]

/-- **Expression code names only its own registers and its temporaries.** -/
theorem emitP_regsOk : ∀ (e : WFExp) (n : Nat),
    (∀ r ∈ e.machRegs, r < PTX_ADDR_SCRATCH) → (emitP n e).2.2 ≤ PTX_ADDR_SCRATCH →
    (emitP n e).1.okB = true ∧ ∀ i ∈ (emitP n e).2.1, i.regsOkB = true := by
  intro e
  induction e with
  | reg r =>
      intro n h _
      refine ⟨?_, by simp [emitP]⟩
      simp only [emitP, PReg.okB, decide_eq_true_eq]
      exact h r (by simp [WFExp.machRegs])
  | lit v =>
      intro n _ hc
      have hn : n < PTX_ADDR_SCRATCH := by
        have : n + 1 ≤ PTX_ADDR_SCRATCH := hc
        omega
      exact ⟨by simp [emitP, PReg.okB, hn],
             by simp [emitP, PInstr.regsOkB, PReg.okB, hn]⟩
  | neg a ih =>
      intro n h hc
      simp only [emitP] at hc
      have hca : (emitP n a).2.2 < PTX_ADDR_SCRATCH := by
        have : (emitP n a).2.2 + 1 ≤ PTX_ADDR_SCRATCH := hc
        omega
      obtain ⟨o1, i1⟩ := ih n (fun r hr => h r hr) (by omega)
      refine ⟨by simp only [emitP, PReg.okB_tmp hca], ?_⟩
      intro i hi
      simp only [emitP, List.mem_append, List.mem_singleton] at hi
      rcases hi with h1 | h2
      · exact i1 i h1
      · subst h2; simp only [PInstr.regsOkB, PReg.okB_tmp hca, o1, Bool.and_self]
  | inv a ih =>
      intro n h hc
      simp only [emitP] at hc
      have hca : (emitP n a).2.2 + 1 < PTX_ADDR_SCRATCH := by
        have : (emitP n a).2.2 + 2 ≤ PTX_ADDR_SCRATCH := hc
        omega
      obtain ⟨o1, i1⟩ := ih n (fun r hr => h r hr) (by omega)
      refine ⟨by simp only [emitP]; exact PReg.okB_tmp (by omega), ?_⟩
      intro i hi
      simp only [emitP, List.mem_append, List.mem_cons, List.mem_singleton,
                 List.not_mem_nil, or_false] at hi
      rcases hi with h1 | h2 | h3
      · exact i1 i h1
      · subst h2; simp only [PInstr.regsOkB]; exact PReg.okB_tmp (by omega)
      · subst h3; simp only [PInstr.regsOkB, o1, Bool.and_true]
        exact Bool.and_eq_true _ _ ▸ ⟨PReg.okB_tmp (by omega), PReg.okB_tmp (by omega)⟩
  | rsqrt a ih =>
      intro n h hc
      simp only [emitP] at hc
      have hca : (emitP n a).2.2 + 2 < PTX_ADDR_SCRATCH := by
        have : (emitP n a).2.2 + 3 ≤ PTX_ADDR_SCRATCH := hc
        omega
      obtain ⟨o1, i1⟩ := ih n (fun r hr => h r hr) (by omega)
      refine ⟨by simp only [emitP]; exact PReg.okB_tmp (by omega), ?_⟩
      intro i hi
      simp only [emitP, List.mem_append, List.mem_cons, List.mem_singleton,
                 List.not_mem_nil, or_false] at hi
      rcases hi with h1 | h2 | h3 | h4
      · exact i1 i h1
      · subst h2; simp only [PInstr.regsOkB, o1, Bool.and_true]
        exact PReg.okB_tmp (by omega)
      · subst h3; simp only [PInstr.regsOkB]; exact PReg.okB_tmp (by omega)
      · subst h4; simp only [PInstr.regsOkB, PReg.okB_tmp (show (emitP n a).2.2 + 2 < PTX_ADDR_SCRATCH by omega), PReg.okB_tmp (show (emitP n a).2.2 + 1 < PTX_ADDR_SCRATCH by omega), PReg.okB_tmp (show (emitP n a).2.2 < PTX_ADDR_SCRATCH by omega), Bool.and_self]
  | ex2 a ih =>
      intro n h hc
      simp only [emitP] at hc
      have hca : (emitP n a).2.2 < PTX_ADDR_SCRATCH := by
        have : (emitP n a).2.2 + 1 ≤ PTX_ADDR_SCRATCH := hc
        omega
      obtain ⟨o1, i1⟩ := ih n (fun r hr => h r hr) (by omega)
      refine ⟨by simp only [emitP, PReg.okB_tmp hca], ?_⟩
      intro i hi
      simp only [emitP, List.mem_append, List.mem_singleton] at hi
      rcases hi with h1 | h2
      · exact i1 i h1
      · subst h2; simp only [PInstr.regsOkB, PReg.okB_tmp hca, o1, Bool.and_self]
  | exp a ih =>
      intro n h hc
      simp only [emitP] at hc
      have hca : (emitP n a).2.2 < PTX_ADDR_SCRATCH := by omega
      obtain ⟨o1, i1⟩ := ih n (fun r hr => h r hr) (by omega)
      refine ⟨by simp [emitP, PReg.okB_tmp hca], ?_⟩
      intro i hi
      simp only [emitP, List.mem_append, List.mem_singleton] at hi
      rcases hi with h1 | h2
      · exact i1 i h1
      · subst h2; simp only [PInstr.regsOkB, PReg.okB_tmp hca, o1, Bool.and_self]
  | add a b iha ihb =>
      intro n h hc
      simp only [emitP] at hc
      have hb := emitP_counter b (emitP n a).2.2
      have ha := emitP_counter a n
      have hcb : (emitP (emitP n a).2.2 b).2.2 < PTX_ADDR_SCRATCH := by
        have : (emitP (emitP n a).2.2 b).2.2 + 1 ≤ PTX_ADDR_SCRATCH := hc
        omega
      obtain ⟨o1, i1⟩ := iha n (fun r hr => h r (by simp [WFExp.machRegs, hr])) (by omega)
      obtain ⟨o2, i2⟩ := ihb (emitP n a).2.2 (fun r hr => h r (by simp [WFExp.machRegs, hr])) (by omega)
      refine ⟨by simp only [emitP, PReg.okB_tmp hcb], ?_⟩
      intro i hi
      simp only [emitP, List.mem_append, List.mem_singleton] at hi
      rcases hi with (h1 | h2) | h3
      · exact i1 i h1
      · exact i2 i h2
      · subst h3; simp only [PInstr.regsOkB, PReg.okB_tmp hcb, o1, o2, Bool.and_self]
  | mul a b iha ihb =>
      intro n h hc
      simp only [emitP] at hc
      have hcb : (emitP (emitP n a).2.2 b).2.2 < PTX_ADDR_SCRATCH := by
        have : (emitP (emitP n a).2.2 b).2.2 + 1 ≤ PTX_ADDR_SCRATCH := hc
        omega
      have hb := emitP_counter b (emitP n a).2.2
      obtain ⟨o1, i1⟩ := iha n (fun r hr => h r (by simp [WFExp.machRegs, hr])) (by omega)
      obtain ⟨o2, i2⟩ := ihb (emitP n a).2.2 (fun r hr => h r (by simp [WFExp.machRegs, hr])) (by omega)
      refine ⟨by simp only [emitP, PReg.okB_tmp hcb], ?_⟩
      intro i hi
      simp only [emitP, List.mem_append, List.mem_singleton] at hi
      rcases hi with (h1 | h2) | h3
      · exact i1 i h1
      · exact i2 i h2
      · subst h3; simp only [PInstr.regsOkB, PReg.okB_tmp hcb, o1, o2, Bool.and_self]
  | maxW a b iha ihb =>
      intro n h hc
      simp only [emitP] at hc
      have hcb : (emitP (emitP n a).2.2 b).2.2 < PTX_ADDR_SCRATCH := by
        have : (emitP (emitP n a).2.2 b).2.2 + 1 ≤ PTX_ADDR_SCRATCH := hc
        omega
      have hb := emitP_counter b (emitP n a).2.2
      obtain ⟨o1, i1⟩ := iha n (fun r hr => h r (by simp [WFExp.machRegs, hr])) (by omega)
      obtain ⟨o2, i2⟩ := ihb (emitP n a).2.2 (fun r hr => h r (by simp [WFExp.machRegs, hr])) (by omega)
      refine ⟨by simp only [emitP, PReg.okB_tmp hcb], ?_⟩
      intro i hi
      simp only [emitP, List.mem_append, List.mem_singleton] at hi
      rcases hi with (h1 | h2) | h3
      · exact i1 i h1
      · exact i2 i h2
      · subst h3; simp only [PInstr.regsOkB, PReg.okB_tmp hcb, o1, o2, Bool.and_self]
  | geF a b iha ihb =>
      intro n h hc
      simp only [emitP] at hc
      have hcb : (emitP (emitP n a).2.2 b).2.2 < PTX_ADDR_SCRATCH := by
        have : (emitP (emitP n a).2.2 b).2.2 + 1 ≤ PTX_ADDR_SCRATCH := hc
        omega
      have hb := emitP_counter b (emitP n a).2.2
      obtain ⟨o1, i1⟩ := iha n (fun r hr => h r (by simp [WFExp.machRegs, hr])) (by omega)
      obtain ⟨o2, i2⟩ := ihb (emitP n a).2.2 (fun r hr => h r (by simp [WFExp.machRegs, hr])) (by omega)
      refine ⟨by simp only [emitP, PReg.okB_tmp hcb], ?_⟩
      intro i hi
      simp only [emitP, List.mem_append, List.mem_singleton] at hi
      rcases hi with (h1 | h2) | h3
      · exact i1 i h1
      · exact i2 i h2
      · subst h3; simp only [PInstr.regsOkB, PReg.okB_tmp hcb, o1, o2, Bool.and_self]

/-- Integer registers an address expression names directly — a gathered index
    that an earlier load put in a register. -/
def IdxE.iregsOf : IdxE → List Nat
  | .ireg r      => [r]
  | .ldIdx _ off => off.iregsOf
  | .add a b     => a.iregsOf ++ b.iregsOf
  | .mul a b     => a.iregsOf ++ b.iregsOf
  | _            => []

/-- **Address code names only the registers it allocated, the two the prologue
    fills, the loop counter, and the buffers it was given.** -/
theorem emitIdx_regsOk (lr : Nat) (hlr : lr < PTX_ADDR_SCRATCH) :
    ∀ (e : IdxE) (n : Nat),
      (∀ r ∈ e.iregsOf, r < PTX_ADDR_SCRATCH) →
      (∀ b ∈ e.bufsOf, b < PTX_ADDR_SCRATCH) →
      (emitIdx lr n e).2.2 ≤ PTX_ADDR_SCRATCH →
      (emitIdx lr n e).1 < PTX_ADDR_SCRATCH
        ∧ ∀ i ∈ (emitIdx lr n e).2.1, i.regsOkB = true := by
  intro e
  induction e with
  | laneId => intro n _ _ _; exact ⟨by simp [emitIdx, laneIR, PTX_ADDR_SCRATCH], by simp [emitIdx]⟩
  | ctaId => intro n _ _ _; exact ⟨by simp [emitIdx, ctaIR, PTX_ADDR_SCRATCH], by simp [emitIdx]⟩
  | loopI => intro n _ _ _; exact ⟨hlr, by simp [emitIdx]⟩
  | ireg r =>
      intro n h _ _
      exact ⟨h r (by simp [IdxE.iregsOf]), by simp [emitIdx]⟩
  | lit c =>
      intro n _ _ hc
      simp only [emitIdx] at hc
      have hn : n < PTX_ADDR_SCRATCH := by omega
      exact ⟨hn, by simp [emitIdx, SI.regsOkB, iOk, hn]⟩
  | ldIdx b off ih =>
      intro n h hb hc
      simp only [emitIdx] at hc
      have hcn : (emitIdx lr n off).2.2 < PTX_ADDR_SCRATCH := by omega
      have hbb : b < PTX_ADDR_SCRATCH := hb b (by simp [IdxE.bufsOf])
      obtain ⟨o1, i1⟩ := ih n (fun r hr => h r hr)
        (fun c hc' => hb c (by simp [IdxE.bufsOf, hc'])) (by omega)
      refine ⟨by simp only [emitIdx]; exact hcn, ?_⟩
      intro i hi
      simp only [emitIdx, List.mem_append, List.mem_singleton] at hi
      rcases hi with h1 | h2
      · exact i1 i h1
      · subst h2
        simp [SI.regsOkB, iOk, hcn, hbb, o1]
  | add a b iha ihb =>
      intro n h hb hc
      simp only [emitIdx] at hc
      have hm := emitIdx_counter lr b (emitIdx lr n a).2.2
      have hcb : (emitIdx lr (emitIdx lr n a).2.2 b).2.2 < PTX_ADDR_SCRATCH := by omega
      obtain ⟨o1, i1⟩ := iha n (fun r hr => h r (by simp [IdxE.iregsOf, hr]))
        (fun c hc' => hb c (by simp [IdxE.bufsOf, hc'])) (by omega)
      obtain ⟨o2, i2⟩ := ihb (emitIdx lr n a).2.2 (fun r hr => h r (by simp [IdxE.iregsOf, hr]))
        (fun c hc' => hb c (by simp [IdxE.bufsOf, hc'])) (by omega)
      refine ⟨by simp only [emitIdx]; exact hcb, ?_⟩
      intro i hi
      simp only [emitIdx, List.mem_append, List.mem_singleton] at hi
      rcases hi with (h1 | h2) | h3
      · exact i1 i h1
      · exact i2 i h2
      · subst h3
        simp [SI.regsOkB, iOk, hcb, o1, o2]
  | mul a b iha ihb =>
      intro n h hb hc
      simp only [emitIdx] at hc
      have hm := emitIdx_counter lr b (emitIdx lr n a).2.2
      have hcb : (emitIdx lr (emitIdx lr n a).2.2 b).2.2 < PTX_ADDR_SCRATCH := by omega
      obtain ⟨o1, i1⟩ := iha n (fun r hr => h r (by simp [IdxE.iregsOf, hr]))
        (fun c hc' => hb c (by simp [IdxE.bufsOf, hc'])) (by omega)
      obtain ⟨o2, i2⟩ := ihb (emitIdx lr n a).2.2 (fun r hr => h r (by simp [IdxE.iregsOf, hr]))
        (fun c hc' => hb c (by simp [IdxE.bufsOf, hc'])) (by omega)
      refine ⟨by simp only [emitIdx]; exact hcb, ?_⟩
      intro i hi
      simp only [emitIdx, List.mem_append, List.mem_singleton] at hi
      rcases hi with (h1 | h2) | h3
      · exact i1 i h1
      · exact i2 i h2
      · subst h3
        simp [SI.regsOkB, iOk, hcb, o1, o2]

/-- **The counter never runs ahead of the instructions.**  Every temporary the
    expression emitter allocates costs an instruction, which is what lets a
    length bound stand in for a register bound. -/
theorem emitP_counter_le : ∀ (e : WFExp) (n : Nat),
    (emitP n e).2.2 ≤ n + (emitP n e).2.1.length := by
  intro e
  induction e with
  | reg r => intro n; simp [emitP]
  | lit v => intro n; simp [emitP]
  | neg a ih => intro n; have := ih n; simp [emitP]; omega
  | ex2 a ih => intro n; have := ih n; simp [emitP]; omega
  | exp a ih => intro n; have := ih n; simp [emitP]; omega
  | inv a ih => intro n; have := ih n; simp [emitP]; omega
  | rsqrt a ih => intro n; have := ih n; simp [emitP]; omega
  | add a b iha ihb =>
      intro n; have h1 := iha n; have h2 := ihb (emitP n a).2.2; simp [emitP]; omega
  | mul a b iha ihb =>
      intro n; have h1 := iha n; have h2 := ihb (emitP n a).2.2; simp [emitP]; omega
  | maxW a b iha ihb =>
      intro n; have h1 := iha n; have h2 := ihb (emitP n a).2.2; simp [emitP]; omega
  | geF a b iha ihb =>
      intro n; have h1 := iha n; have h2 := ihb (emitP n a).2.2; simp [emitP]; omega

/-- …and the same for address code. -/
theorem emitIdx_counter_le (lr : Nat) : ∀ (e : IdxE) (n : Nat),
    (emitIdx lr n e).2.2 ≤ n + (emitIdx lr n e).2.1.length := by
  intro e
  induction e with
  | laneId => intro n; simp [emitIdx]
  | ctaId => intro n; simp [emitIdx]
  | loopI => intro n; simp [emitIdx]
  | ireg r => intro n; simp [emitIdx]
  | lit c => intro n; simp [emitIdx]
  | ldIdx b off ih => intro n; have := ih n; simp [emitIdx]; omega
  | add a b iha ihb =>
      intro n; have h1 := iha n; have h2 := ihb (emitIdx lr n a).2.2; simp [emitIdx]; omega
  | mul a b iha ihb =>
      intro n; have h1 := iha n; have h2 := ihb (emitIdx lr n a).2.2; simp [emitIdx]; omega

/-- The machine registers a statement names. -/
def EWStmt.machRegsOf : EWStmt → List Nat
  | .seq a b             => a.machRegsOf ++ b.machRegsOf
  | .setR r e            => r :: e.machRegs
  | .shflXor d s _       => [d, s]
  | .loadIdx d _ _       => [d]
  | .loadV4 a b c d _ _  => [a, b, c, d]
  | .storeLane0 _ _ r    => [r]
  | .storeLane _ _ r     => [r]
  | .stSm _ r            => [r]
  | .ldSm d _            => [d]
  | .cvtIF d _           => [d]
  | .forN _ body         => body.machRegsOf
  | .forM _ _ body       => body.machRegsOf
  | _                    => []

/-- The integer registers a statement names — gathered indices an earlier load
    put in a register. -/
def EWStmt.iregsOf : EWStmt → List Nat
  | .seq a b              => a.iregsOf ++ b.iregsOf
  | .loadIdx _ _ ix       => ix.iregsOf
  | .loadV4 _ _ _ _ _ ix  => ix.iregsOf
  | .storeLane0 _ ix _    => ix.iregsOf
  | .storeLane _ ix _     => ix.iregsOf
  | .stSm ix _            => ix.iregsOf
  | .ldSm _ ix            => ix.iregsOf
  | .cvtIF _ ix           => ix.iregsOf
  | .forN _ body          => body.iregsOf
  | .forM _ _ body        => body.iregsOf
  | _                     => []

/-- Everything a statement carries by name is below the scratch register. -/
def EWStmt.NamesOk (s : EWStmt) : Prop :=
  (∀ r ∈ s.machRegsOf, r < PTX_ADDR_SCRATCH)
    ∧ (∀ r ∈ s.iregsOf, r < PTX_ADDR_SCRATCH)
    ∧ (∀ b ∈ s.bufsOf, b < PTX_ADDR_SCRATCH)

private theorem idx_tail_ok {lr n : Nat} {ix : IdxE} {tail : List SI}
    (hlr : lr < PTX_ADDR_SCRATCH)
    (hi : ∀ r ∈ ix.iregsOf, r < PTX_ADDR_SCRATCH)
    (hb : ∀ b ∈ ix.bufsOf, b < PTX_ADDR_SCRATCH)
    (hc : (emitIdx lr n ix).2.2 ≤ PTX_ADDR_SCRATCH)
    (ht : ∀ i ∈ tail, i.regsOkB = true) :
    ((emitIdx lr n ix).2.1 ++ tail).all SI.regsOkB = true := by
  obtain ⟨_, i1⟩ := emitIdx_regsOk lr hlr ix n hi hb hc
  refine List.all_eq_true.mpr (fun i hi' => ?_)
  rcases List.mem_append.mp hi' with h1 | h2
  · exact i1 i h1
  · exact ht i h2

/-- **The statement emitter names no reserved register**, for every statement:
    a bound on the instruction count is a bound on the registers, because every
    allocation costs an instruction. -/
theorem emitEW_regsOk : ∀ (s : EWStmt) (lr n : Nat),
    lr < PTX_ADDR_SCRATCH → s.NamesOk →
    n + (emitEW lr n s).length ≤ PTX_ADDR_SCRATCH →
    (emitEW lr n s).all SI.regsOkB = true := by
  intro s
  induction s with
  | skip => intro lr n _ _ _; rfl
  | barrier => intro lr n _ _ _; rfl
  | seq a b iha ihb =>
      intro lr n hlr hn hc
      obtain ⟨hm, hi, hb⟩ := hn
      simp only [emitEW, List.all_append, Bool.and_eq_true]
      simp only [emitEW, List.length_append] at hc
      refine ⟨iha lr n hlr ⟨?_, ?_, ?_⟩ (by omega), ihb lr n hlr ⟨?_, ?_, ?_⟩ (by omega)⟩
      · exact fun r hr => hm r (by simp [EWStmt.machRegsOf, hr])
      · exact fun r hr => hi r (by simp [EWStmt.iregsOf, hr])
      · exact fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])
      · exact fun r hr => hm r (by simp [EWStmt.machRegsOf, hr])
      · exact fun r hr => hi r (by simp [EWStmt.iregsOf, hr])
      · exact fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])
  | setR r e =>
      intro lr n hlr hn hc
      obtain ⟨hm, _, _⟩ := hn
      simp only [emitEW, List.length_append, List.length_map, List.length_cons,
                 List.length_nil] at hc
      have hcp : (emitP 0 e).2.2 ≤ PTX_ADDR_SCRATCH := by
        have := emitP_counter_le e 0; omega
      obtain ⟨o1, i1⟩ := emitP_regsOk e 0
        (fun x hx => hm x (by simp [EWStmt.machRegsOf, hx])) hcp
      have hr : r < PTX_ADDR_SCRATCH := hm r (by simp [EWStmt.machRegsOf])
      simp only [emitEW, List.all_append, List.all_map, List.all_cons, List.all_nil,
                 Bool.and_true, Bool.and_eq_true]
      constructor
      · exact List.all_eq_true.mpr (fun i hi' => i1 i hi')
      · simp only [SI.regsOkB, PInstr.regsOkB, PReg.okB_mach hr, o1, Bool.and_self]
  | shflXor d sr m =>
      intro lr n _ hn _
      obtain ⟨hm, _, _⟩ := hn
      simp [emitEW, SI.regsOkB, PInstr.regsOkB, PReg.okB,
            hm d (by simp [EWStmt.machRegsOf]), hm sr (by simp [EWStmt.machRegsOf])]
  | loadIdx d b ix =>
      intro lr n hlr hn hc
      obtain ⟨hm, hi, hb⟩ := hn
      simp only [emitEW, List.length_append, List.length_cons, List.length_nil] at hc
      have hcx : (emitIdx lr n ix).2.2 ≤ PTX_ADDR_SCRATCH := by
        have := emitIdx_counter_le lr ix n; omega
      obtain ⟨o1, _⟩ := emitIdx_regsOk lr hlr ix n
        (fun r hr => hi r hr) (fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])) hcx
      refine idx_tail_ok hlr (fun r hr => hi r hr)
        (fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])) hcx (fun i hi' => ?_)
      simp only [List.mem_singleton] at hi'
      subst hi'
      simp [SI.regsOkB, PReg.okB, iOk, hm d (by simp [EWStmt.machRegsOf]),
            hb b (by simp [EWStmt.bufsOf]), o1]
  | loadV4 d0 d1 d2 d3 bu ix =>
      intro lr n hlr hn hc
      obtain ⟨hm, hi, hb⟩ := hn
      simp only [emitEW, List.length_append, List.length_cons, List.length_nil] at hc
      have hcx : (emitIdx lr n ix).2.2 ≤ PTX_ADDR_SCRATCH := by
        have := emitIdx_counter_le lr ix n; omega
      obtain ⟨o1, _⟩ := emitIdx_regsOk lr hlr ix n
        (fun r hr => hi r hr) (fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])) hcx
      refine idx_tail_ok hlr (fun r hr => hi r hr)
        (fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])) hcx (fun i hi' => ?_)
      simp only [List.mem_singleton] at hi'
      subst hi'
      simp [SI.regsOkB, PReg.okB, iOk, hm d0 (by simp [EWStmt.machRegsOf]),
            hm d1 (by simp [EWStmt.machRegsOf]), hm d2 (by simp [EWStmt.machRegsOf]),
            hm d3 (by simp [EWStmt.machRegsOf]), hb bu (by simp [EWStmt.bufsOf]), o1]
  | storeLane0 bu ix r =>
      intro lr n hlr hn hc
      obtain ⟨hm, hi, hb⟩ := hn
      simp only [emitEW, List.length_append, List.length_cons, List.length_nil] at hc
      have hcnt := emitIdx_counter_le lr ix n
      have hcx : (emitIdx lr n ix).2.2 < PTX_ADDR_SCRATCH := by omega
      obtain ⟨o1, _⟩ := emitIdx_regsOk lr hlr ix n
        (fun x hx => hi x hx) (fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])) (by omega)
      refine idx_tail_ok hlr (fun x hx => hi x hx)
        (fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])) (by omega) (fun i hi' => ?_)
      simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at hi'
      rcases hi' with h1 | h2
      · subst h1
        have hl : laneIR < PTX_ADDR_SCRATCH := by simp [laneIR, PTX_ADDR_SCRATCH]
        simp [SI.regsOkB, iOk, hcx, hl]
      · subst h2
        simp [SI.regsOkB, PReg.okB, iOk, hcx, hb bu (by simp [EWStmt.bufsOf]), o1,
              hm r (by simp [EWStmt.machRegsOf])]
  | storeLane bu ix r =>
      intro lr n hlr hn hc
      obtain ⟨hm, hi, hb⟩ := hn
      simp only [emitEW, List.length_append, List.length_cons, List.length_nil] at hc
      have hcx : (emitIdx lr n ix).2.2 ≤ PTX_ADDR_SCRATCH := by
        have := emitIdx_counter_le lr ix n; omega
      obtain ⟨o1, _⟩ := emitIdx_regsOk lr hlr ix n
        (fun x hx => hi x hx) (fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])) hcx
      refine idx_tail_ok hlr (fun x hx => hi x hx)
        (fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])) hcx (fun i hi' => ?_)
      simp only [List.mem_singleton] at hi'
      subst hi'
      simp [SI.regsOkB, PReg.okB, iOk, hb bu (by simp [EWStmt.bufsOf]), o1,
            hm r (by simp [EWStmt.machRegsOf])]
  | stSm ix r =>
      intro lr n hlr hn hc
      obtain ⟨hm, hi, hb⟩ := hn
      simp only [emitEW, List.length_append, List.length_cons, List.length_nil] at hc
      have hcx : (emitIdx lr n ix).2.2 ≤ PTX_ADDR_SCRATCH := by
        have := emitIdx_counter_le lr ix n; omega
      obtain ⟨o1, _⟩ := emitIdx_regsOk lr hlr ix n
        (fun x hx => hi x hx) (fun c hc' => hb c hc') hcx
      refine idx_tail_ok hlr (fun x hx => hi x hx) (fun c hc' => hb c hc') hcx
        (fun i hi' => ?_)
      simp only [List.mem_singleton] at hi'
      subst hi'
      simp [SI.regsOkB, PReg.okB, iOk, o1, hm r (by simp [EWStmt.machRegsOf])]
  | ldSm d ix =>
      intro lr n hlr hn hc
      obtain ⟨hm, hi, hb⟩ := hn
      simp only [emitEW, List.length_append, List.length_cons, List.length_nil] at hc
      have hcx : (emitIdx lr n ix).2.2 ≤ PTX_ADDR_SCRATCH := by
        have := emitIdx_counter_le lr ix n; omega
      obtain ⟨o1, _⟩ := emitIdx_regsOk lr hlr ix n
        (fun x hx => hi x hx) (fun c hc' => hb c hc') hcx
      refine idx_tail_ok hlr (fun x hx => hi x hx) (fun c hc' => hb c hc') hcx
        (fun i hi' => ?_)
      simp only [List.mem_singleton] at hi'
      subst hi'
      simp [SI.regsOkB, PReg.okB, iOk, o1, hm d (by simp [EWStmt.machRegsOf])]
  | cvtIF d ix =>
      intro lr n hlr hn hc
      obtain ⟨hm, hi, hb⟩ := hn
      simp only [emitEW, List.length_append, List.length_cons, List.length_nil] at hc
      have hcx : (emitIdx lr n ix).2.2 ≤ PTX_ADDR_SCRATCH := by
        have := emitIdx_counter_le lr ix n; omega
      obtain ⟨o1, _⟩ := emitIdx_regsOk lr hlr ix n
        (fun x hx => hi x hx) (fun c hc' => hb c hc') hcx
      refine idx_tail_ok hlr (fun x hx => hi x hx) (fun c hc' => hb c hc') hcx
        (fun i hi' => ?_)
      simp only [List.mem_singleton] at hi'
      subst hi'
      simp [SI.regsOkB, PReg.okB, iOk, o1, hm d (by simp [EWStmt.machRegsOf])]
  | forN cnt body _ => intro lr n _ _ _; simp [emitEW, SI.regsOkB]
  | forM bu ad body _ => intro lr n _ _ _; simp [emitEW, SI.regsOkB]

/-- A statement the flattener passes straight to `emitEW`. -/
private theorem flatLeaf {s : EWStmt} {lr n base : Nat}
    (hlr : lr < PTX_ADDR_SCRATCH) (hn : s.NamesOk)
    (hc : n + flenEW lr n s ≤ PTX_ADDR_SCRATCH)
    (hleaf : flatEW lr n base s = (emitEW lr n s).map FI.si := by rfl)
    (hlen : flenEW lr n s = (emitEW lr n s).length := by rfl) :
    FlatRegsOkB (flatEW lr n base s) = true := by
  rw [hleaf]
  rw [hlen] at hc
  have := emitEW_regsOk s lr n hlr hn hc
  simp only [FlatRegsOkB, List.all_map]
  exact List.all_eq_true.mpr (fun i hi => List.all_eq_true.mp this i hi)

/-- **The flattened program names no reserved register.**

    Loops become real branches here, so the loop counter is a register the
    program writes rather than a bound the interpreter holds — which is why
    this needs the count bound and `emitEW_regsOk` does not. -/
theorem flatEW_regsOk : ∀ (s : EWStmt) (lr n base : Nat),
    lr < PTX_ADDR_SCRATCH → s.NamesOk →
    n + flenEW lr n s ≤ PTX_ADDR_SCRATCH →
    FlatRegsOkB (flatEW lr n base s) = true := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro lr n base hlr hn hc
      obtain ⟨hm, hi, hb⟩ := hn
      simp only [flenEW] at hc
      simp only [flatEW, FlatRegsOkB, List.all_append, Bool.and_eq_true]
      refine ⟨iha lr n base hlr ⟨?_, ?_, ?_⟩ (by omega),
              ihb lr n _ hlr ⟨?_, ?_, ?_⟩ (by omega)⟩
      · exact fun r hr => hm r (by simp [EWStmt.machRegsOf, hr])
      · exact fun r hr => hi r (by simp [EWStmt.iregsOf, hr])
      · exact fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])
      · exact fun r hr => hm r (by simp [EWStmt.machRegsOf, hr])
      · exact fun r hr => hi r (by simp [EWStmt.iregsOf, hr])
      · exact fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])
  | forN cnt body ih =>
      intro lr n base hlr hn hc
      obtain ⟨hm, hi, hb⟩ := hn
      simp only [flenEW] at hc
      have hnn : n < PTX_ADDR_SCRATCH := by omega
      have hbody := ih n (n + 1) (base + 3) hnn
        ⟨fun r hr => hm r hr, fun r hr => hi r hr, fun c hc' => hb c hc'⟩ (by omega)
      simp only [flatEW, FlatRegsOkB, List.all_cons, List.all_append, List.all_nil,
                 Bool.and_true, Bool.and_eq_true]
      exact ⟨by simp [FI.regsOkB, SI.regsOkB, iOk, hnn],
             by simp [FI.regsOkB, SI.regsOkB, iOk, hnn],
             by simp [FI.regsOkB, iOk, hnn],
             hbody,
             by simp [FI.regsOkB, SI.regsOkB, iOk, hnn],
             by simp [FI.regsOkB]⟩
  | forM bu ad body ih =>
      intro lr n base hlr hn hc
      obtain ⟨hm, hi, hb⟩ := hn
      simp only [flenEW] at hc
      have hnn : n < PTX_ADDR_SCRATCH := by omega
      have hn1 : n + 1 < PTX_ADDR_SCRATCH := by omega
      have hn2 : n + 2 < PTX_ADDR_SCRATCH := by omega
      have hbu : bu < PTX_ADDR_SCRATCH := hb bu (by simp [EWStmt.bufsOf])
      have hbody := ih n (n + 3) (base + 5) hnn
        ⟨fun r hr => hm r hr, fun r hr => hi r hr,
         fun c hc' => hb c (by simp [EWStmt.bufsOf, hc'])⟩ (by omega)
      simp only [flatEW, FlatRegsOkB, List.all_cons, List.all_append, List.all_nil,
                 Bool.and_true, Bool.and_eq_true]
      exact ⟨by simp [FI.regsOkB, SI.regsOkB, iOk, hn1],
             by simp [FI.regsOkB, SI.regsOkB, iOk, hn1, hn2, hbu],
             by simp [FI.regsOkB, SI.regsOkB, iOk, hnn],
             by simp [FI.regsOkB, SI.regsOkB, iOk, hnn, hn2],
             by simp [FI.regsOkB, iOk, hnn],
             hbody,
             by simp [FI.regsOkB, SI.regsOkB, iOk, hnn],
             by simp [FI.regsOkB]⟩
  | skip => intro lr n base hlr hn hc; rfl
  | barrier => intro lr n base hlr hn hc; rfl
  | setR r e => intro lr n base hlr hn hc; exact flatLeaf hlr hn hc
  | shflXor d sr m => intro lr n base hlr hn hc; exact flatLeaf hlr hn hc
  | loadIdx d b ix => intro lr n base hlr hn hc; exact flatLeaf hlr hn hc
  | loadV4 a b c d bu ix => intro lr n base hlr hn hc; exact flatLeaf hlr hn hc
  | storeLane0 bu ix r => intro lr n base hlr hn hc; exact flatLeaf hlr hn hc
  | storeLane bu ix r => intro lr n base hlr hn hc; exact flatLeaf hlr hn hc
  | stSm ix r => intro lr n base hlr hn hc; exact flatLeaf hlr hn hc
  | ldSm d ix => intro lr n base hlr hn hc; exact flatLeaf hlr hn hc
  | cvtIF d ix => intro lr n base hlr hn hc; exact flatLeaf hlr hn hc

/-- **Counting without emitting.**

    `flenEW` is defined at a leaf as the length of the emitted list, so asking
    how long a kernel is means building it.  These mirror the emitters' shape
    and are proven equal to them, which is what makes the length usable as a
    bound rather than as a second emission. -/
def IdxE.emitLen : IdxE → Nat
  | .ldIdx _ off => off.emitLen + 1
  | .add a b     => a.emitLen + b.emitLen + 1
  | .mul a b     => a.emitLen + b.emitLen + 1
  | .lit _       => 1
  | _            => 0

def WFExp.emitLen : WFExp → Nat
  | .reg _    => 0
  | .lit _    => 1
  | .add a b  => a.emitLen + b.emitLen + 1
  | .mul a b  => a.emitLen + b.emitLen + 1
  | .maxW a b => a.emitLen + b.emitLen + 1
  | .geF a b  => a.emitLen + b.emitLen + 1
  | .neg a    => a.emitLen + 1
  | .ex2 a    => a.emitLen + 1
  | .exp a    => a.emitLen + 1
  | .inv a    => a.emitLen + 2
  | .rsqrt a  => a.emitLen + 3

theorem emitIdx_len (lr : Nat) : ∀ (e : IdxE) (n : Nat),
    (emitIdx lr n e).2.1.length = e.emitLen := by
  intro e
  induction e with
  | ldIdx b off ih => intro n; simp [emitIdx, IdxE.emitLen, ih]
  | add a b iha ihb =>
      intro n; simp [emitIdx, IdxE.emitLen, iha, ihb]; omega
  | mul a b iha ihb =>
      intro n; simp [emitIdx, IdxE.emitLen, iha, ihb]; omega
  | _ => intro n; simp [emitIdx, IdxE.emitLen]

theorem emitP_len : ∀ (e : WFExp) (n : Nat),
    (emitP n e).2.1.length = e.emitLen := by
  intro e
  induction e with
  | reg r => intro n; simp [emitP, WFExp.emitLen]
  | lit v => intro n; simp [emitP, WFExp.emitLen]
  | neg a ih => intro n; simp [emitP, WFExp.emitLen, ih]
  | ex2 a ih => intro n; simp [emitP, WFExp.emitLen, ih]
  | exp a ih => intro n; simp [emitP, WFExp.emitLen, ih]
  | inv a ih => intro n; simp [emitP, WFExp.emitLen, ih]
  | rsqrt a ih => intro n; simp [emitP, WFExp.emitLen, ih]
  | add a b iha ihb => intro n; simp [emitP, WFExp.emitLen, iha, ihb]; omega
  | mul a b iha ihb => intro n; simp [emitP, WFExp.emitLen, iha, ihb]; omega
  | maxW a b iha ihb => intro n; simp [emitP, WFExp.emitLen, iha, ihb]; omega
  | geF a b iha ihb => intro n; simp [emitP, WFExp.emitLen, iha, ihb]; omega

/-- How many instructions a statement becomes.  Independent of the loop
    register and the allocation counter, which is why it takes neither. -/
def EWStmt.flenCheap : EWStmt → Nat
  | .skip                 => 0
  | .seq a b              => a.flenCheap + b.flenCheap
  | .setR _ e             => e.emitLen + 1
  | .shflXor _ _ _        => 1
  | .barrier              => 1
  | .loadIdx _ _ ix       => ix.emitLen + 1
  | .loadV4 _ _ _ _ _ ix  => ix.emitLen + 1
  | .storeLane0 _ ix _    => ix.emitLen + 2
  | .storeLane _ ix _     => ix.emitLen + 1
  | .stSm ix _            => ix.emitLen + 1
  | .ldSm _ ix            => ix.emitLen + 1
  | .cvtIF _ ix           => ix.emitLen + 1
  | .forN _ body          => body.flenCheap + 5
  | .forM _ _ body        => body.flenCheap + 7

theorem flenCheap_eq : ∀ (s : EWStmt) (lr n : Nat), flenEW lr n s = s.flenCheap := by
  intro s
  induction s with
  | seq a b iha ihb => intro lr n; simp [flenEW, EWStmt.flenCheap, iha, ihb]
  | forN cnt body ih => intro lr n; simp [flenEW, EWStmt.flenCheap, ih]
  | forM bu ad body ih => intro lr n; simp [flenEW, EWStmt.flenCheap, ih]
  | setR r e => intro lr n; simp [flenEW, emitEW, EWStmt.flenCheap, emitP_len]
  | loadIdx d b ix => intro lr n; simp [flenEW, emitEW, EWStmt.flenCheap, emitIdx_len]
  | loadV4 a b c d bu ix => intro lr n; simp [flenEW, emitEW, EWStmt.flenCheap, emitIdx_len]
  | storeLane0 bu ix r => intro lr n; simp [flenEW, emitEW, EWStmt.flenCheap, emitIdx_len]
  | storeLane bu ix r => intro lr n; simp [flenEW, emitEW, EWStmt.flenCheap, emitIdx_len]
  | stSm ix r => intro lr n; simp [flenEW, emitEW, EWStmt.flenCheap, emitIdx_len]
  | ldSm d ix => intro lr n; simp [flenEW, emitEW, EWStmt.flenCheap, emitIdx_len]
  | cvtIF d ix => intro lr n; simp [flenEW, emitEW, EWStmt.flenCheap, emitIdx_len]
  | _ => intro lr n; simp [flenEW, emitEW, EWStmt.flenCheap]

/-- The names, as a decidable check. -/
def EWStmt.namesOkB (s : EWStmt) : Bool :=
  s.machRegsOf.all (fun r => decide (r < PTX_ADDR_SCRATCH))
    && s.iregsOf.all (fun r => decide (r < PTX_ADDR_SCRATCH))
    && s.bufsOf.all (fun b => decide (b < PTX_ADDR_SCRATCH))

theorem namesOk_of_B (s : EWStmt) (h : s.namesOkB = true) : s.NamesOk := by
  simp only [EWStmt.namesOkB, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

/-- **A whole kernel names no reserved register**, from two facts about the
    statement: the names it carries, and how many instructions it becomes.

    Neither builds the instruction list, which is what the guard was doing per
    kernel. -/
theorem flatKernel_regsOk (s : EWStmt) (hn : s.namesOkB = true)
    (hc : 3 + flenEW 2 3 s ≤ PTX_ADDR_SCRATCH) :
    FlatRegsOkB (flatKernel s) = true := by
  simp only [flatKernel, FlatRegsOkB, List.all_append, List.all_map, Bool.and_eq_true]
  refine ⟨by simp [emitPrologue, FI.regsOkB, SI.regsOkB, iOk, laneIR, ctaIR,
                   PTX_ADDR_SCRATCH], ?_⟩
  exact flatEW_regsOk s 2 3 3 (by simp [PTX_ADDR_SCRATCH]) (namesOk_of_B s hn) hc

/-- The same, as one decidable check per kernel. -/
def EWStmt.kernelRegsOkB (s : EWStmt) : Bool :=
  s.namesOkB && decide (3 + s.flenCheap ≤ PTX_ADDR_SCRATCH)

theorem flatKernel_regsOk_of_B (s : EWStmt) (h : s.kernelRegsOkB = true) :
    FlatRegsOkB (flatKernel s) = true := by
  simp only [EWStmt.kernelRegsOkB, Bool.and_eq_true, decide_eq_true_eq] at h
  exact flatKernel_regsOk s h.1 (by rw [flenCheap_eq]; exact h.2)


/-! ### Renaming and expansion leave the register facts alone

    A kernel is emitted from `expandEW (op.localStmt batch)`.  Neither step
    touches a machine register or a gathered index, and `expandEW` changes the
    length only where it rewrites `exp`. -/

theorem IdxE.iregsOf_renameBuf (f : Buf → Buf) : ∀ ix : IdxE,
    (ix.renameBuf f).iregsOf = ix.iregsOf := by
  intro ix
  induction ix with
  | ldIdx b off ih => simp [IdxE.renameBuf, IdxE.iregsOf, ih]
  | add a b iha ihb => simp [IdxE.renameBuf, IdxE.iregsOf, iha, ihb]
  | mul a b iha ihb => simp [IdxE.renameBuf, IdxE.iregsOf, iha, ihb]
  | _ => rfl

theorem IdxE.emitLen_renameBuf (f : Buf → Buf) : ∀ ix : IdxE,
    (ix.renameBuf f).emitLen = ix.emitLen := by
  intro ix
  induction ix with
  | ldIdx b off ih => simp [IdxE.renameBuf, IdxE.emitLen, ih]
  | add a b iha ihb => simp [IdxE.renameBuf, IdxE.emitLen, iha, ihb]
  | mul a b iha ihb => simp [IdxE.renameBuf, IdxE.emitLen, iha, ihb]
  | _ => rfl

theorem EWStmt.machRegsOf_renameBuf (f : Buf → Buf) : ∀ s : EWStmt,
    (s.renameBuf f).machRegsOf = s.machRegsOf := by
  intro s
  induction s with
  | seq a b iha ihb => simp [EWStmt.renameBuf, EWStmt.machRegsOf, iha, ihb]
  | forN _ body ih => simp [EWStmt.renameBuf, EWStmt.machRegsOf, ih]
  | forM _ _ body ih => simp [EWStmt.renameBuf, EWStmt.machRegsOf, ih]
  | _ => rfl

theorem EWStmt.iregsOf_renameBuf (f : Buf → Buf) : ∀ s : EWStmt,
    (s.renameBuf f).iregsOf = s.iregsOf := by
  intro s
  induction s with
  | seq a b iha ihb => simp [EWStmt.renameBuf, EWStmt.iregsOf, iha, ihb]
  | forN _ body ih => simp [EWStmt.renameBuf, EWStmt.iregsOf, ih]
  | forM _ _ body ih => simp [EWStmt.renameBuf, EWStmt.iregsOf, ih]
  | _ => simp [EWStmt.renameBuf, EWStmt.iregsOf, IdxE.iregsOf_renameBuf]

theorem EWStmt.flenCheap_renameBuf (f : Buf → Buf) : ∀ s : EWStmt,
    (s.renameBuf f).flenCheap = s.flenCheap := by
  intro s
  induction s with
  | seq a b iha ihb => simp [EWStmt.renameBuf, EWStmt.flenCheap, iha, ihb]
  | forN _ body ih => simp [EWStmt.renameBuf, EWStmt.flenCheap, ih]
  | forM _ _ body ih => simp [EWStmt.renameBuf, EWStmt.flenCheap, ih]
  | _ => simp [EWStmt.renameBuf, EWStmt.flenCheap, IdxE.emitLen_renameBuf]

theorem expandExp_machRegs : ∀ e : WFExp, (expandExp e).machRegs = e.machRegs := by
  intro e
  induction e with
  | add a b iha ihb => simp [expandExp, WFExp.machRegs, iha, ihb]
  | mul a b iha ihb => simp [expandExp, WFExp.machRegs, iha, ihb]
  | maxW a b iha ihb => simp [expandExp, WFExp.machRegs, iha, ihb]
  | geF a b iha ihb => simp [expandExp, WFExp.machRegs, iha, ihb]
  | neg a ih => simp [expandExp, WFExp.machRegs, ih]
  | inv a ih => simp [expandExp, WFExp.machRegs, ih]
  | rsqrt a ih => simp [expandExp, WFExp.machRegs, ih]
  | ex2 a ih => simp [expandExp, WFExp.machRegs, ih]
  | exp a ih => simp [expandExp, WFExp.machRegs, ih]
  | _ => rfl

/-- How many `exp` nodes the rewrite will fire on. -/
def WFExp.expCount : WFExp → Nat
  | .exp a    => a.expCount + 1
  | .add a b  => a.expCount + b.expCount
  | .mul a b  => a.expCount + b.expCount
  | .maxW a b => a.expCount + b.expCount
  | .geF a b  => a.expCount + b.expCount
  | .neg a    => a.expCount
  | .inv a    => a.expCount
  | .rsqrt a  => a.expCount
  | .ex2 a    => a.expCount
  | _         => 0

/-- **Rewriting `exp` costs exactly two instructions per occurrence.**  Exact
    rather than a bound, so no kernel is rejected for headroom it does not
    need. -/
theorem expandExp_emitLen : ∀ e : WFExp,
    (expandExp e).emitLen = e.emitLen + 2 * e.expCount := by
  intro e
  induction e with
  | reg r => rfl
  | lit v => rfl
  | add a b iha ihb => simp only [expandExp, WFExp.emitLen, WFExp.expCount, iha, ihb]; omega
  | mul a b iha ihb => simp only [expandExp, WFExp.emitLen, WFExp.expCount, iha, ihb]; omega
  | maxW a b iha ihb => simp only [expandExp, WFExp.emitLen, WFExp.expCount, iha, ihb]; omega
  | geF a b iha ihb => simp only [expandExp, WFExp.emitLen, WFExp.expCount, iha, ihb]; omega
  | neg a ih => simp only [expandExp, WFExp.emitLen, WFExp.expCount, ih]; omega
  | inv a ih => simp only [expandExp, WFExp.emitLen, WFExp.expCount, ih]; omega
  | rsqrt a ih => simp only [expandExp, WFExp.emitLen, WFExp.expCount, ih]; omega
  | ex2 a ih => simp only [expandExp, WFExp.emitLen, WFExp.expCount, ih]; omega
  | exp a ih => simp only [expandExp, WFExp.emitLen, WFExp.expCount, ih]; omega

def EWStmt.expCount : EWStmt → Nat
  | .seq a b       => a.expCount + b.expCount
  | .setR _ e      => e.expCount
  | .forN _ body   => body.expCount
  | .forM _ _ body => body.expCount
  | _              => 0

theorem expandEW_machRegsOf : ∀ s : EWStmt, (expandEW s).machRegsOf = s.machRegsOf := by
  intro s
  induction s with
  | seq a b iha ihb => simp [expandEW, EWStmt.machRegsOf, iha, ihb]
  | forN _ body ih => simp [expandEW, EWStmt.machRegsOf, ih]
  | forM _ _ body ih => simp [expandEW, EWStmt.machRegsOf, ih]
  | setR r e => simp [expandEW, EWStmt.machRegsOf, expandExp_machRegs]
  | _ => rfl

theorem expandEW_iregsOf : ∀ s : EWStmt, (expandEW s).iregsOf = s.iregsOf := by
  intro s
  induction s with
  | seq a b iha ihb => simp [expandEW, EWStmt.iregsOf, iha, ihb]
  | forN _ body ih => simp [expandEW, EWStmt.iregsOf, ih]
  | forM _ _ body ih => simp [expandEW, EWStmt.iregsOf, ih]
  | _ => rfl

theorem expandEW_bufsOf : ∀ s : EWStmt, (expandEW s).bufsOf = s.bufsOf := by
  intro s
  induction s with
  | seq a b iha ihb => simp [expandEW, EWStmt.bufsOf, iha, ihb]
  | forN _ body ih => simp [expandEW, EWStmt.bufsOf, ih]
  | forM _ _ body ih => simp [expandEW, EWStmt.bufsOf, ih]
  | _ => rfl

theorem expandEW_flenCheap : ∀ s : EWStmt,
    (expandEW s).flenCheap = s.flenCheap + 2 * s.expCount := by
  intro s
  induction s with
  | seq a b iha ihb =>
      simp only [expandEW, EWStmt.flenCheap, EWStmt.expCount, iha, ihb]; omega
  | forN _ body ih =>
      simp only [expandEW, EWStmt.flenCheap, EWStmt.expCount, ih]; omega
  | forM _ _ body ih =>
      simp only [expandEW, EWStmt.flenCheap, EWStmt.expCount, ih]; omega
  | setR r e =>
      simp only [expandEW, EWStmt.flenCheap, EWStmt.expCount, expandExp_emitLen]; omega
  | _ => simp [expandEW, EWStmt.flenCheap, EWStmt.expCount]

/-! ### What a schema names in registers, and how long it is

    The same decomposition as `TOp.stmt_bufsOf`, for the two quantities the
    register guard needs. -/

theorem seqN_iregsOf (f : Nat → EWStmt) (h : ∀ s, (f s).iregsOf = []) :
    ∀ n, (seqN n f).iregsOf = [] := by
  intro n
  induction n with
  | zero => rfl
  | succ k ih => simp [seqN, EWStmt.iregsOf, ih, h k]

theorem seqN_machRegsOf_lt (N : Nat) (f : Nat → EWStmt) :
    ∀ n, (∀ s, s < n → ∀ r ∈ (f s).machRegsOf, r < N) →
    ∀ r ∈ (seqN n f).machRegsOf, r < N := by
  intro n
  induction n with
  | zero => intro _ r hr; simp [seqN, EWStmt.machRegsOf] at hr
  | succ k ih =>
      intro h r hr
      simp only [seqN, EWStmt.machRegsOf, List.mem_append] at hr
      rcases hr with h1 | h2
      · exact ih (fun s hs => h s (by omega)) r h1
      · exact h k (by omega) r h2

theorem seqN_flenCheap (f : Nat → EWStmt) (c : Nat) (h : ∀ s, (f s).flenCheap = c) :
    ∀ n, (seqN n f).flenCheap = n * c := by
  intro n
  induction n with
  | zero => simp [seqN, EWStmt.flenCheap]
  | succ k ih =>
      simp only [seqN, EWStmt.flenCheap, ih, h k]
      exact (Nat.succ_mul k c).symm

theorem BCast_ix_iregsOf (m : BCast) : m.ix.iregsOf = [] := by
  cases m <;> simp [BCast.ix, stride32, IdxE.iregsOf]

/-! ### Each schema's length and registers, in closed form

    Exact, so the operation-level budget is a real number rather than a margin.
    The batched schemas are linear in `B` because that is what unrolling the
    batch does; nothing here depends on the trip count `K`, because a counted
    loop is five instructions whatever it counts. -/

theorem dotStrided_flenCheap (bA bB out : Buf) (ixA ixB oi : IdxE) (K : Nat) :
    (dotStrided bA bB ixA ixB out oi K).flenCheap
      = ixA.emitLen + ixB.emitLen + oi.emitLen + 29 := by
  simp [dotStrided, dotStridedBody, warpRoundE, bflyRoundE, EWStmt.flenCheap,
        WFExp.emitLen, dotStepSE]
  omega

theorem maxStrided_flenCheap (b out : Buf) (ix oi : IdxE) (K : Nat) (init : Float32) :
    (maxStrided b ix out oi K init).flenCheap = ix.emitLen + oi.emitLen + 27 := by
  simp [maxStrided, maxStridedBody, warpReduceMaxE, warpMaxRoundE, EWStmt.flenCheap,
        WFExp.emitLen, maxStepSE]
  omega

theorem zipPassEW_flenCheap (bA bB out : Buf) (dA dB r : Nat) (f : WFExp)
    (ixA ixB oix : IdxE) (K : Nat) :
    (zipPassEW bA bB out dA dB r f ixA ixB oix K).flenCheap
      = ixA.emitLen + ixB.emitLen + oix.emitLen + f.emitLen + 9 := by
  simp [zipPassEW, storeBody, EWStmt.flenCheap]
  omega

theorem zip3PassEW_flenCheap (bA bB bC out : Buf) (dA dB dC r : Nat) (f : WFExp)
    (ixA ixB ixC oix : IdxE) (K : Nat) :
    (zip3PassEW bA bB bC out dA dB dC r f ixA ixB ixC oix K).flenCheap
      = ixA.emitLen + ixB.emitLen + ixC.emitLen + oix.emitLen + f.emitLen + 10 := by
  simp [zip3PassEW, storeBody, EWStmt.flenCheap]
  omega

theorem softmaxCE_flenCheap (logits bias oneHot out : Buf) (biasIx : IdxE) :
    (softmaxCE logits bias oneHot out biasIx).flenCheap = biasIx.emitLen + 57 := by
  simp [softmaxCE, smBody, smPre, smMid, smPost, seqAll, warpReduceMaxE,
        warpMaxRoundE, warpReduceSumE, warpRoundE, bflyRoundE, elemIx,
        EWStmt.flenCheap, WFExp.emitLen, IdxE.emitLen]
  omega

theorem dotBatched_flenCheap (bA bB out : Buf) (ixA : IdxE) (ixB oi : Nat → IdxE)
    (a c B K : Nat) (hB : ∀ s, (ixB s).emitLen = a) (ho : ∀ s, (oi s).emitLen = c) :
    (dotBatched bA bB ixA ixB out oi B K).flenCheap
      = ixA.emitLen + B * (a + c) + 23 * B + 6 := by
  simp only [dotBatched, batchBodyE, batchTripE, batchEpilogueE, EWStmt.flenCheap]
  rw [seqN_flenCheap _ 2 (fun s => by simp [EWStmt.flenCheap, WFExp.emitLen])]
  rw [seqN_flenCheap _ (a + 4) (fun s => by
        simp [batchStepE, EWStmt.flenCheap, WFExp.emitLen, hB s] <;> omega)]
  rw [seqN_flenCheap _ (c + 17) (fun s => by
        simp [EWStmt.flenCheap, warpReduceE, bflyRoundE, WFExp.emitLen, ho s] <;> omega)]
  simp only [Nat.mul_add, Nat.add_mul]
  omega

theorem outerBatched_flenCheap (bA bB out : Buf) (ixA ixB : Nat → IdxE) (base : IdxE)
    (a c B K : Nat) (hA : ∀ s, (ixA s).emitLen = a) (hB : ∀ s, (ixB s).emitLen = c) :
    (outerBatched bA bB out ixA ixB base B K).flenCheap
      = B * (a + c) + base.emitLen + 5 * B + 12 := by
  simp only [outerBatched, storeBody, outerBodyE, EWStmt.flenCheap]
  rw [seqN_flenCheap _ (a + c + 5) (fun s => by
        simp [outerStepE, EWStmt.flenCheap, WFExp.emitLen, dotStepSE, hA s, hB s]
          <;> omega)]
  simp only [stride32, IdxE.emitLen, WFExp.emitLen, Nat.mul_add]
  omega

/-- An exclusive bound on the machine registers an expression reads. -/
def WFExp.maxReg : WFExp → Nat
  | .reg r    => r + 1
  | .add a b  => max a.maxReg b.maxReg
  | .mul a b  => max a.maxReg b.maxReg
  | .maxW a b => max a.maxReg b.maxReg
  | .geF a b  => max a.maxReg b.maxReg
  | .neg a    => a.maxReg
  | .inv a    => a.maxReg
  | .rsqrt a  => a.maxReg
  | .ex2 a    => a.maxReg
  | .exp a    => a.maxReg
  | .lit _    => 0

theorem WFExp.machRegs_lt : ∀ (e : WFExp) (r : Nat), r ∈ e.machRegs → r < e.maxReg := by
  intro e
  induction e with
  | reg k => intro r hr; simp [WFExp.machRegs] at hr; simp [WFExp.maxReg, hr]
  | lit v => intro r hr; simp [WFExp.machRegs] at hr
  | add a b iha ihb =>
      intro r hr
      simp only [WFExp.machRegs, List.mem_append] at hr
      rcases hr with h | h
      · exact Nat.lt_of_lt_of_le (iha r h) (Nat.le_max_left _ _)
      · exact Nat.lt_of_lt_of_le (ihb r h) (Nat.le_max_right _ _)
  | mul a b iha ihb =>
      intro r hr
      simp only [WFExp.machRegs, List.mem_append] at hr
      rcases hr with h | h
      · exact Nat.lt_of_lt_of_le (iha r h) (Nat.le_max_left _ _)
      · exact Nat.lt_of_lt_of_le (ihb r h) (Nat.le_max_right _ _)
  | maxW a b iha ihb =>
      intro r hr
      simp only [WFExp.machRegs, List.mem_append] at hr
      rcases hr with h | h
      · exact Nat.lt_of_lt_of_le (iha r h) (Nat.le_max_left _ _)
      · exact Nat.lt_of_lt_of_le (ihb r h) (Nat.le_max_right _ _)
  | geF a b iha ihb =>
      intro r hr
      simp only [WFExp.machRegs, List.mem_append] at hr
      rcases hr with h | h
      · exact Nat.lt_of_lt_of_le (iha r h) (Nat.le_max_left _ _)
      · exact Nat.lt_of_lt_of_le (ihb r h) (Nat.le_max_right _ _)
  | neg a ih => intro r hr; exact ih r hr
  | inv a ih => intro r hr; exact ih r hr
  | rsqrt a ih => intro r hr; exact ih r hr
  | ex2 a ih => intro r hr; exact ih r hr
  | exp a ih => intro r hr; exact ih r hr

/-! ### The registers each schema names -/

theorem dotStrided_regs (bA bB out : Buf) (ixA ixB oi : IdxE) (K : Nat)
    (hA : ixA.iregsOf = []) (hB : ixB.iregsOf = []) (ho : oi.iregsOf = []) :
    (∀ r ∈ (dotStrided bA bB ixA ixB out oi K).machRegsOf, r < 3)
      ∧ (dotStrided bA bB ixA ixB out oi K).iregsOf = [] := by
  constructor
  · intro r hr
    simp [dotStrided, dotStridedBody, warpRoundE, bflyRoundE, EWStmt.machRegsOf,
          WFExp.machRegs, dotStepSE] at hr
    omega
  · simp [dotStrided, dotStridedBody, warpRoundE, bflyRoundE, EWStmt.iregsOf, hA, hB, ho]

theorem maxStrided_regs (b out : Buf) (ix oi : IdxE) (K : Nat) (init : Float32)
    (hix : ix.iregsOf = []) (ho : oi.iregsOf = []) :
    (∀ r ∈ (maxStrided b ix out oi K init).machRegsOf, r < 2)
      ∧ (maxStrided b ix out oi K init).iregsOf = [] := by
  constructor
  · intro r hr
    simp [maxStrided, maxStridedBody, warpReduceMaxE, warpMaxRoundE,
          EWStmt.machRegsOf, WFExp.machRegs, maxStepSE] at hr
    omega
  · simp [maxStrided, maxStridedBody, warpReduceMaxE, warpMaxRoundE, EWStmt.iregsOf,
          hix, ho]

theorem softmaxCE_regs (logits bias oneHot out : Buf) (biasIx : IdxE)
    (hb : biasIx.iregsOf = []) :
    (∀ r ∈ (softmaxCE logits bias oneHot out biasIx).machRegsOf, r < 6)
      ∧ (softmaxCE logits bias oneHot out biasIx).iregsOf = [] := by
  constructor
  · intro r hr
    simp [softmaxCE, smBody, smPre, smMid, smPost, seqAll, warpReduceMaxE,
          warpMaxRoundE, warpReduceSumE, warpRoundE, bflyRoundE,
          EWStmt.machRegsOf, WFExp.machRegs, SM_LM, SM_P, SM_OUT] at hr
    omega
  · simp [softmaxCE, smBody, smPre, smMid, smPost, seqAll, warpReduceMaxE,
          warpMaxRoundE, warpReduceSumE, warpRoundE, bflyRoundE, elemIx,
          EWStmt.iregsOf, IdxE.iregsOf, hb]

theorem dotBatched_regs (bA bB out : Buf) (ixA : IdxE) (ixB oi : Nat → IdxE)
    (B K : Nat) (hA : ixA.iregsOf = []) (hB : ∀ s, (ixB s).iregsOf = [])
    (ho : ∀ s, (oi s).iregsOf = []) :
    (∀ r ∈ (dotBatched bA bB ixA ixB out oi B K).machRegsOf, r < B + 3)
      ∧ (dotBatched bA bB ixA ixB out oi B K).iregsOf = [] := by
  constructor
  · intro q hq
    simp only [dotBatched, batchBodyE, batchTripE, batchEpilogueE, EWStmt.machRegsOf,
               List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with ((h | h | h) | h) <;>
      first
        | (simp [BRA] at h; omega)
        | (refine seqN_machRegsOf_lt _ _ _ ?_ _ h
           intro t ht r' hr'
           simp [batchStepE, warpReduceE, bflyRoundE, EWStmt.machRegsOf,
                 WFExp.machRegs, BACC, BTMP, BRA, BRB] at hr'
           omega)
  · simp only [dotBatched, batchBodyE, batchTripE, batchEpilogueE, EWStmt.iregsOf,
               List.append_eq_nil_iff]
    refine ⟨⟨?_, ?_, ?_⟩, ?_⟩ <;>
      first
        | (simpa using hA)
        | (refine seqN_iregsOf _ ?_ _
           intro t
           simp [batchStepE, warpReduceE, bflyRoundE, EWStmt.iregsOf, hB t, ho t])

theorem outerBatched_regs (bA bB out : Buf) (ixA ixB : Nat → IdxE) (base : IdxE)
    (B K : Nat) (hA : ∀ s, (ixA s).iregsOf = []) (hB : ∀ s, (ixB s).iregsOf = [])
    (hbase : base.iregsOf = []) :
    (∀ r ∈ (outerBatched bA bB out ixA ixB base B K).machRegsOf, r < 3)
      ∧ (outerBatched bA bB out ixA ixB base B K).iregsOf = [] := by
  constructor
  · intro q hq
    simp only [outerBatched, storeBody, outerBodyE, EWStmt.machRegsOf,
               List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with (h | h) | h <;>
      first
        | omega
        | (simp [WFExp.machRegs] at h; omega)
        | (refine seqN_machRegsOf_lt _ _ _ ?_ _ h
           intro t ht r' hr'
           simp [outerStepE, EWStmt.machRegsOf, WFExp.machRegs, dotStepSE] at hr'
           omega)
  · simp only [outerBatched, storeBody, outerBodyE, EWStmt.iregsOf,
               List.append_eq_nil_iff, stride32, IdxE.iregsOf]
    refine ⟨⟨trivial, ?_⟩, by simpa using hbase⟩
    refine seqN_iregsOf _ ?_ _
    intro t; simp [outerStepE, EWStmt.iregsOf, hA t, hB t]

theorem zipPassEW_machRegsOf (bA bB out : Buf) (dA dB r : Nat) (f : WFExp)
    (ixA ixB oix : IdxE) (K : Nat) :
    (zipPassEW bA bB out dA dB r f ixA ixB oix K).machRegsOf
      = [dA, dB, r] ++ f.machRegs ++ [r] := by
  simp [zipPassEW, storeBody, EWStmt.machRegsOf]

theorem zip3PassEW_machRegsOf (bA bB bC out : Buf) (dA dB dC r : Nat) (f : WFExp)
    (ixA ixB ixC oix : IdxE) (K : Nat) :
    (zip3PassEW bA bB bC out dA dB dC r f ixA ixB ixC oix K).machRegsOf
      = [dA, dB, dC, r] ++ f.machRegs ++ [r] := by
  simp [zip3PassEW, storeBody, EWStmt.machRegsOf]

theorem zipPassEW_iregsOf (bA bB out : Buf) (dA dB r : Nat) (f : WFExp)
    (ixA ixB oix : IdxE) (K : Nat)
    (hA : ixA.iregsOf = []) (hB : ixB.iregsOf = []) (ho : oix.iregsOf = []) :
    (zipPassEW bA bB out dA dB r f ixA ixB oix K).iregsOf = [] := by
  simp [zipPassEW, storeBody, EWStmt.iregsOf, hA, hB, ho]

theorem zip3PassEW_iregsOf (bA bB bC out : Buf) (dA dB dC r : Nat) (f : WFExp)
    (ixA ixB ixC oix : IdxE) (K : Nat)
    (hA : ixA.iregsOf = []) (hB : ixB.iregsOf = []) (hC : ixC.iregsOf = [])
    (ho : oix.iregsOf = []) :
    (zip3PassEW bA bB bC out dA dB dC r f ixA ixB ixC oix K).iregsOf = [] := by
  simp [zip3PassEW, storeBody, EWStmt.iregsOf, hA, hB, hC, ho]

/-! ### The map schema

    Its body is a compiled `Expr`, so the bound is an induction over the
    expression rather than a closed form in the widths. -/

/-- Instructions the value expression becomes. -/
def Expr.valLen : {Γ : Nat} → Expr Γ → Nat
  | _, .var _   => 0
  | _, .lit _   => 1
  | _, .add a b => a.valLen + b.valLen + 1
  | _, .mul a b => a.valLen + b.valLen + 1
  | _, .neg a   => a.valLen + 1
  | _, .inv a   => a.valLen + 2
  | _, .exp a   => a.valLen + 1
  | _, .rsqrt a => a.valLen + 3
  | _, .sum _ _ => 0
  | _, .letE _ b => b.valLen

/-- Instructions the compiled code becomes. -/
def Expr.codeLen : {Γ : Nat} → Expr Γ → Nat
  | _, .var _   => 0
  | _, .lit _   => 0
  | _, .add a b => a.codeLen + b.codeLen
  | _, .mul a b => a.codeLen + b.codeLen
  | _, .neg a   => a.codeLen
  | _, .inv a   => a.codeLen
  | _, .exp a   => a.codeLen
  | _, .rsqrt a => a.codeLen
  | _, .sum n f => 2 + (List.finRange n).foldl
                        (fun m j => m + (f j).codeLen + (f j).valLen + 2) 0
  | _, .letE a b => a.codeLen + a.valLen + 1 + b.codeLen

/-- An exclusive bound on the machine registers the compiled code writes and
    the value expression reads. -/
def Expr.regHi : {Γ : Nat} → Expr Γ → Nat
  | Γ, .var i   => i.val + 1
  | _, .lit _   => 0
  | _, .add a b => max a.regHi b.regHi
  | _, .mul a b => max a.regHi b.regHi
  | _, .neg a   => a.regHi
  | _, .inv a   => a.regHi
  | _, .exp a   => a.regHi
  | _, .rsqrt a => a.regHi
  | _, .sum n f => (List.finRange n).foldl (fun m j => max m (f j).regHi) 0
  | _, .letE a b => max a.regHi b.regHi

/-- The accumulating fold `sumSeq` performs, as a length. -/
theorem sumSeq_flenCheap_gen (acc : Nat) :
    ∀ (bodies : List (EWStmt × WFExp)) (init : EWStmt) (m : Nat),
      (bodies.foldl (fun s (p : EWStmt × WFExp) =>
        .seq s (.seq p.1 (.setR acc (.add (.reg acc) p.2)))) init).flenCheap
          + m
        = init.flenCheap
          + bodies.foldl (fun k (p : EWStmt × WFExp) => k + p.1.flenCheap + p.2.emitLen + 2) m := by
  intro bodies
  induction bodies with
  | nil => intro init m; rfl
  | cons x xs ih =>
      intro init m
      have := ih (.seq init (.seq x.1 (.setR acc (.add (.reg acc) x.2)))) (m + x.1.flenCheap + x.2.emitLen + 2)
      simp only [List.foldl_cons]
      simp only [EWStmt.flenCheap, WFExp.emitLen] at this ⊢
      omega

theorem sumSeq_flenCheap (acc : Nat) (bodies : List (EWStmt × WFExp)) :
    (sumSeq acc bodies).flenCheap
      = 2 + bodies.foldl (fun k (p : EWStmt × WFExp) => k + p.1.flenCheap + p.2.emitLen + 2) 0 := by
  have := sumSeq_flenCheap_gen acc bodies (.setR acc (.lit NumOps.zero)) 0
  simp only [sumSeq, EWStmt.flenCheap, WFExp.emitLen] at this ⊢
  omega

/-- **Compiled expression code has the length the expression says.** -/
theorem compileW_len : ∀ {Γ : Nat} (e : Expr Γ) (ve : Fin Γ → WFExp) (c : Nat),
    (∀ i, (ve i).emitLen = 0) →
    (compileW ve c e).1.flenCheap = e.codeLen
      ∧ (compileW ve c e).2.emitLen = e.valLen := by
  intro Γ e
  induction e with
  | var i => intro ve c h; exact ⟨rfl, h i⟩
  | lit n => intro ve c h; exact ⟨rfl, rfl⟩
  | add a b iha ihb =>
      intro ve c h
      obtain ⟨c1, v1⟩ := iha ve c h
      obtain ⟨c2, v2⟩ := ihb ve (c + slots a) h
      exact ⟨by simp [compileW, EWStmt.flenCheap, Expr.codeLen, c1, c2],
             by simp [compileW, WFExp.emitLen, Expr.valLen, v1, v2]⟩
  | mul a b iha ihb =>
      intro ve c h
      obtain ⟨c1, v1⟩ := iha ve c h
      obtain ⟨c2, v2⟩ := ihb ve (c + slots a) h
      exact ⟨by simp [compileW, EWStmt.flenCheap, Expr.codeLen, c1, c2],
             by simp [compileW, WFExp.emitLen, Expr.valLen, v1, v2]⟩
  | neg a ih =>
      intro ve c h
      obtain ⟨c1, v1⟩ := ih ve c h
      exact ⟨by simp [compileW, Expr.codeLen, c1], by simp [compileW, WFExp.emitLen, Expr.valLen, v1]⟩
  | inv a ih =>
      intro ve c h
      obtain ⟨c1, v1⟩ := ih ve c h
      exact ⟨by simp [compileW, Expr.codeLen, c1], by simp [compileW, WFExp.emitLen, Expr.valLen, v1]⟩
  | exp a ih =>
      intro ve c h
      obtain ⟨c1, v1⟩ := ih ve c h
      exact ⟨by simp [compileW, Expr.codeLen, c1], by simp [compileW, WFExp.emitLen, Expr.valLen, v1]⟩
  | rsqrt a ih =>
      intro ve c h
      obtain ⟨c1, v1⟩ := ih ve c h
      exact ⟨by simp [compileW, Expr.codeLen, c1], by simp [compileW, WFExp.emitLen, Expr.valLen, v1]⟩
  | letE a b iha ihb =>
      intro ve c h
      obtain ⟨c1, v1⟩ := iha ve c h
      obtain ⟨c2, v2⟩ := ihb (extend ve (.reg (c + slots a))) (c + slots a + 1)
        (fun i => by simp only [extend]; split <;> simp [h, WFExp.emitLen])
      refine ⟨?_, by simp [compileW, Expr.valLen, v2]⟩
      simp only [compileW, EWStmt.flenCheap, Expr.codeLen, c1, c2, v1, WFExp.emitLen]
      omega
  | sum n f ih =>
      intro ve c h
      refine ⟨?_, rfl⟩
      simp only [compileW, Expr.codeLen, sumSeq_flenCheap]
      congr 1
      have : ∀ (L : List (Fin n)) (m : Nat),
          (L.map (fun j => compileW ve (c + 1) (f j))).foldl
              (fun k (p : EWStmt × WFExp) => k + p.1.flenCheap + p.2.emitLen + 2) m
            = L.foldl (fun k j => k + (f j).codeLen + (f j).valLen + 2) m := by
        intro L
        induction L with
        | nil => intro m; rfl
        | cons x xs ihL =>
            intro m
            obtain ⟨cx, vx⟩ := ih x ve (c + 1) h
            simp only [List.map_cons, List.foldl_cons, cx, vx]
            exact ihL _
      exact this _ 0

theorem sumSeq_machRegsOf_lt (acc N : Nat) (hacc : acc < N) :
    ∀ (bodies : List (EWStmt × WFExp)) (init : EWStmt),
      (∀ r ∈ init.machRegsOf, r < N) →
      (∀ p ∈ bodies, (∀ r ∈ p.1.machRegsOf, r < N) ∧ (∀ r ∈ p.2.machRegs, r < N)) →
      ∀ r ∈ (bodies.foldl (fun s (p : EWStmt × WFExp) =>
        .seq s (.seq p.1 (.setR acc (.add (.reg acc) p.2)))) init).machRegsOf, r < N := by
  intro bodies
  induction bodies with
  | nil => intro init hi _ r hr; exact hi r hr
  | cons x xs ih =>
      intro init hi hall
      refine ih _ ?_ (fun p hp => hall p (by simp [hp]))
      intro r hr
      simp only [EWStmt.machRegsOf, List.mem_append, List.mem_cons,
                 WFExp.machRegs, List.nil_append, List.not_mem_nil, or_false] at hr
      rcases hr with h | h | h | h <;>
        first
          | exact hi r h
          | exact (hall x (by simp)).1 r h
          | omega
          | exact (hall x (by simp)).2 r h
          | (rcases h with h | h <;>
               first | omega | exact (hall x (by simp)).2 r h)

theorem compileW_iregs : ∀ {Γ : Nat} (e : Expr Γ) (ve : Fin Γ → WFExp) (c : Nat),
    (compileW ve c e).1.iregsOf = [] := by
  intro Γ e
  induction e with
  | var i => intro ve c; rfl
  | lit n => intro ve c; rfl
  | add a b iha ihb => intro ve c; simp [compileW, EWStmt.iregsOf, iha, ihb]
  | mul a b iha ihb => intro ve c; simp [compileW, EWStmt.iregsOf, iha, ihb]
  | neg a ih => intro ve c; exact ih ve c
  | inv a ih => intro ve c; exact ih ve c
  | exp a ih => intro ve c; exact ih ve c
  | rsqrt a ih => intro ve c; exact ih ve c
  | letE a b iha ihb => intro ve c; simp [compileW, EWStmt.iregsOf, iha, ihb]
  | sum n f ih =>
      intro ve c
      simp only [compileW, sumSeq]
      have key : ∀ (L : List (EWStmt × WFExp)) (init : EWStmt),
          init.iregsOf = [] → (∀ p ∈ L, p.1.iregsOf = []) →
          (L.foldl (fun s (p : EWStmt × WFExp) =>
            .seq s (.seq p.1 (.setR c (.add (.reg c) p.2)))) init).iregsOf = [] := by
        intro L
        induction L with
        | nil => intro init hi _; exact hi
        | cons x xs ihL =>
            intro init hi hall
            exact ihL _ (by simp [EWStmt.iregsOf, hi, hall x (by simp)])
              (fun p hp => hall p (by simp [hp]))
      exact key _ _ rfl (fun p hp => by
        simp only [List.mem_map] at hp
        obtain ⟨j, _, hj⟩ := hp
        rw [← hj]; exact ih j ve (c + 1))

theorem foldl_max_ge {α : Type} (g : α → Nat) :
    ∀ (L : List α) (m : Nat), m ≤ L.foldl (fun k y => max k (g y)) m := by
  intro L
  induction L with
  | nil => intro m; exact Nat.le_refl _
  | cons x xs ih => intro m; exact Nat.le_trans (Nat.le_max_left m (g x)) (ih _)

theorem le_foldl_max {α : Type} (g : α → Nat) :
    ∀ (L : List α) (m : Nat) (x : α), x ∈ L →
      g x ≤ L.foldl (fun k y => max k (g y)) m := by
  intro L
  induction L with
  | nil => intro m x hx; simp at hx
  | cons y ys ih =>
      intro m x hx
      rcases List.mem_cons.mp hx with h | h
      · subst h
        exact Nat.le_trans (Nat.le_max_right m (g x)) (foldl_max_ge g ys _)
      · exact ih _ x h

/-- **Compiled code writes no register above the expression's own allocation.** -/
theorem compileW_regs : ∀ {Γ : Nat} (e : Expr Γ) (ve : Fin Γ → WFExp) (c N : Nat),
    (∀ i, ∀ r ∈ (ve i).machRegs, r < N) → c + slots e ≤ N →
    (∀ r ∈ (compileW ve c e).1.machRegsOf, r < N)
      ∧ (∀ r ∈ (compileW ve c e).2.machRegs, r < N) := by
  intro Γ e
  induction e with
  | var i => intro ve c N h _; exact ⟨fun r hr => by simp [compileW, EWStmt.machRegsOf] at hr,
                                      fun r hr => h i r hr⟩
  | lit n => intro ve c N _ _
             exact ⟨fun r hr => by simp [compileW, EWStmt.machRegsOf] at hr,
                    fun r hr => by simp [compileW, WFExp.machRegs] at hr⟩
  | add a b iha ihb =>
      intro ve c N h hc
      simp only [slots] at hc
      obtain ⟨p1, q1⟩ := iha ve c N h (by omega)
      obtain ⟨p2, q2⟩ := ihb ve (c + slots a) N h (by omega)
      refine ⟨fun r hr => ?_, fun r hr => ?_⟩
      · simp only [compileW, EWStmt.machRegsOf, List.mem_append] at hr
        rcases hr with h' | h'
        · exact p1 r h'
        · exact p2 r h'
      · simp only [compileW, WFExp.machRegs, List.mem_append] at hr
        rcases hr with h' | h'
        · exact q1 r h'
        · exact q2 r h'
  | mul a b iha ihb =>
      intro ve c N h hc
      simp only [slots] at hc
      obtain ⟨p1, q1⟩ := iha ve c N h (by omega)
      obtain ⟨p2, q2⟩ := ihb ve (c + slots a) N h (by omega)
      refine ⟨fun r hr => ?_, fun r hr => ?_⟩
      · simp only [compileW, EWStmt.machRegsOf, List.mem_append] at hr
        rcases hr with h' | h'
        · exact p1 r h'
        · exact p2 r h'
      · simp only [compileW, WFExp.machRegs, List.mem_append] at hr
        rcases hr with h' | h'
        · exact q1 r h'
        · exact q2 r h'
  | neg a ih => intro ve c N h hc; exact ih ve c N h hc
  | inv a ih => intro ve c N h hc; exact ih ve c N h hc
  | exp a ih => intro ve c N h hc; exact ih ve c N h hc
  | rsqrt a ih => intro ve c N h hc; exact ih ve c N h hc
  | letE a b iha ihb =>
      intro ve c N h hc
      simp only [slots] at hc
      obtain ⟨p1, q1⟩ := iha ve c N h (by omega)
      have hve' : ∀ i, ∀ r ∈ ((extend ve (WFExp.reg (c + slots a))) i).machRegs, r < N := by
        intro i r hr
        simp only [extend] at hr
        split at hr
        · exact h _ r hr
        · simp [WFExp.machRegs] at hr; omega
      obtain ⟨p2, q2⟩ := ihb (extend ve (.reg (c + slots a))) (c + slots a + 1) N hve' (by omega)
      refine ⟨fun r hr => ?_, fun r hr => q2 r hr⟩
      simp only [compileW, EWStmt.machRegsOf, List.mem_append, List.mem_cons] at hr
      rcases hr with (h' | h' | h') | h'
      · exact p1 r h'
      · omega
      · exact q1 r h'
      · exact p2 r h'
  | sum n f ih =>
      intro ve c N h hc
      simp only [slots] at hc
      have hcN : c < N := by omega
      refine ⟨?_, fun r hr => by simp [compileW, WFExp.machRegs] at hr; omega⟩
      simp only [compileW, sumSeq]
      refine sumSeq_machRegsOf_lt c N hcN _ _
        (fun r hr => by simp [EWStmt.machRegsOf, WFExp.machRegs] at hr; omega) ?_
      intro p hp
      simp only [List.mem_map] at hp
      obtain ⟨j, hj, hje⟩ := hp
      have hslot : c + 1 + slots (f j) ≤ N := by
        have : slots (f j) ≤ (List.finRange n).foldl (fun m k => max m (slots (f k))) 0 :=
          le_foldl_max (fun k => slots (f k)) _ 0 j (List.mem_finRange j)
        omega
      rw [← hje]
      exact ih j ve (c + 1) N h hslot

/-- An `exp` node costs an instruction of its own, so the rewrite can at worst
    triple a length. -/
theorem WFExp.expCount_le : ∀ e : WFExp, e.expCount ≤ e.emitLen := by
  intro e
  induction e with
  | reg r => simp [WFExp.expCount, WFExp.emitLen]
  | lit v => simp [WFExp.expCount, WFExp.emitLen]
  | add a b iha ihb => simp only [WFExp.expCount, WFExp.emitLen]; omega
  | mul a b iha ihb => simp only [WFExp.expCount, WFExp.emitLen]; omega
  | maxW a b iha ihb => simp only [WFExp.expCount, WFExp.emitLen]; omega
  | geF a b iha ihb => simp only [WFExp.expCount, WFExp.emitLen]; omega
  | neg a ih => simp only [WFExp.expCount, WFExp.emitLen]; omega
  | inv a ih => simp only [WFExp.expCount, WFExp.emitLen]; omega
  | rsqrt a ih => simp only [WFExp.expCount, WFExp.emitLen]; omega
  | ex2 a ih => simp only [WFExp.expCount, WFExp.emitLen]; omega
  | exp a ih => simp only [WFExp.expCount, WFExp.emitLen]; omega

theorem EWStmt.expCount_le : ∀ s : EWStmt, s.expCount ≤ s.flenCheap := by
  intro s
  induction s with
  | seq a b iha ihb => simp only [EWStmt.expCount, EWStmt.flenCheap]; omega
  | forN _ body ih => simp only [EWStmt.expCount, EWStmt.flenCheap]; omega
  | forM _ _ body ih => simp only [EWStmt.expCount, EWStmt.flenCheap]; omega
  | setR r e => have := e.expCount_le; simp only [EWStmt.expCount, EWStmt.flenCheap]; omega
  | _ => simp [EWStmt.expCount, EWStmt.flenCheap]

/-- **What expanding costs, as a bound in the statement's own length.** -/
theorem expandEW_flenCheap_le (s : EWStmt) : (expandEW s).flenCheap ≤ 3 * s.flenCheap := by
  have h1 := expandEW_flenCheap s
  have h2 := s.expCount_le
  omega

/-! ### The load prologue -/

theorem loadSeqN_flenCheap {Γ : Nat} (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE)
    (a : Nat) (h : ∀ i, (inIx i).emitLen = a) :
    ∀ n, n ≤ Γ → (loadSeqN inB inIx n).flenCheap = n * (a + 1) := by
  intro n
  induction n with
  | zero => intro _; simp [loadSeqN, EWStmt.flenCheap]
  | succ k ih =>
      intro hk
      have hkΓ : k < Γ := by omega
      simp only [loadSeqN, EWStmt.flenCheap, dif_pos hkΓ, ih (by omega), h ⟨k, hkΓ⟩]
      exact (Nat.succ_mul k (a + 1)).symm

theorem loadSeqN_regNames {Γ : Nat} (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE)
    (hix : ∀ i, (inIx i).iregsOf = []) :
    ∀ n, (∀ r ∈ (loadSeqN inB inIx n).machRegsOf, r < max n 1)
      ∧ (loadSeqN inB inIx n).iregsOf = [] := by
  intro n
  induction n with
  | zero => exact ⟨fun r hr => by simp [loadSeqN, EWStmt.machRegsOf] at hr, rfl⟩
  | succ k ih =>
      obtain ⟨p, q⟩ := ih
      constructor
      · intro r hr
        simp only [loadSeqN, EWStmt.machRegsOf, List.mem_append] at hr
        rcases hr with h1 | h2
        · have := p r h1; omega
        · split at h2 <;> simp [EWStmt.machRegsOf] at h2 <;> omega
      · simp only [loadSeqN, EWStmt.iregsOf, List.append_eq_nil_iff]
        refine ⟨q, ?_⟩
        split <;> simp [EWStmt.iregsOf, hix]

/-! ### The map schema, assembled -/

theorem mapKernelAt_bounds {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf)
    (inIx : Fin Γ → IdxE) (out : Buf) (a : Nat)
    (hlen : ∀ i, (inIx i).emitLen = a) (hix : ∀ i, (inIx i).iregsOf = []) :
    ((mapKernelAt spec inB inIx out).ew).flenCheap
        = Γ * (a + 1) + spec.codeLen + spec.valLen + 5
      ∧ (∀ r ∈ ((mapKernelAt spec inB inIx out).ew).machRegsOf, r < Γ + slots spec + 2)
      ∧ ((mapKernelAt spec inB inIx out).ew).iregsOf = [] := by
  obtain ⟨hc, hv⟩ := compileW_len spec (fun i => .reg i.val) Γ (fun i => rfl)
  obtain ⟨pc, pv⟩ := compileW_regs spec (fun i => .reg i.val) Γ (Γ + slots spec)
    (fun i r hr => by simp [WFExp.machRegs] at hr; omega) (Nat.le_refl _)
  obtain ⟨pl, ql⟩ := loadSeqN_regNames inB inIx hix Γ
  refine ⟨?_, ?_, ?_⟩
  · simp only [mapKernelAt, compileWKernel, EWStmt.flenCheap, hc, hv, loadSeq,
               loadSeqN_flenCheap inB inIx a hlen Γ (Nat.le_refl _), elemIx,
               IdxE.emitLen, WFExp.emitLen]
    omega
  · intro r hr
    simp only [mapKernelAt, compileWKernel, EWStmt.machRegsOf, List.mem_append,
               List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with ((h | h | h) | h) <;>
      first
        | (have := pl r h; omega)
        | (have := pc r h; omega)
        | (have := pv r h; omega)
        | omega
        | (rcases h with h | h <;> first | omega | (have := pv r h; omega))
  · simp [mapKernelAt, compileWKernel, EWStmt.iregsOf, loadSeq, ql, compileW_iregs,
          elemIx, IdxE.iregsOf]

/-- How long a broadcast's address expression is. -/
def BCast.ixLen : BCast → Nat
  | .rowOf _ _  => 8
  | .scalar     => 0
  | .sharedAt _ => 5
  | .constAt _  => 1

theorem BCast_ix_emitLen (m : BCast) : m.ix.emitLen = m.ixLen := by
  cases m <;> simp [BCast.ix, BCast.ixLen, stride32, IdxE.emitLen]

/-! ### The operation-level budget

    Each schema's closed form, read off the constructor.  These are arithmetic
    in the widths the operation already carries, so checking a tape costs one
    comparison per operation rather than a traversal of the instructions it
    would become. -/

/-- An exclusive bound on the machine registers the operation's kernel names. -/
def TOp.regHi : TOp → Nat
  | .mv _ _ _ _ b _ _                => b + 3
  | .mvT _ _ _ _ b _ _               => b + 3
  | .outer _ _ _ _ _ _ _             => 3
  | .ew1 f _ _ _                     => slots f + 3
  | .ew2 f _ _ _ _                   => slots f + 4
  | .ew3 f _ _ _ _ _                 => slots f + 5
  | .ew4 f _ _ _ _ _ _               => slots f + 6
  | .upd2 f _ _ _                    => slots f + 4
  | .smce _ _ _ _ _                  => 6
  | .rowsq _ _ _ _                   => 3
  | .rowdot _ _ _ _ _ _ _            => 3
  | .rowmax _ _ _ _ _                => 2
  | .ziprow _ _ _ f _ _ _ _ _ _      => max 3 f.maxReg
  | .ziprow3 _ _ _ _ f _ _ _ _ _ _ _ => max 4 f.maxReg

/-- How many instructions the operation's kernel body becomes.  Exact — the
    batched schemas are linear in the batch because unrolling is what a batch
    is, and nothing depends on a trip count. -/
def TOp.flenOf : TOp → Nat
  | .mv _ _ _ _ b _ _                => 30 * b + 12
  | .mvT _ _ _ _ b _ _               => 30 * b + 12
  | .outer _ _ _ _ b _ _             => 12 * b + 14
  | .ew1 f _ _ _                     => f.codeLen + f.valLen + 9
  | .ew2 f _ _ _ _                   => f.codeLen + f.valLen + 13
  | .ew3 f _ _ _ _ _                 => f.codeLen + f.valLen + 17
  | .ew4 f _ _ _ _ _ _               => f.codeLen + f.valLen + 21
  | .upd2 f _ _ _                    => f.codeLen + f.valLen + 13
  | .smce _ _ _ _ _                  => 57
  | .rowsq _ _ _ _                   => 41
  | .rowdot _ _ _ mA mB _ _          => mA.ixLen + mB.ixLen + 29
  | .rowmax _ _ _ _ _                => 33
  | .ziprow _ _ _ f mA mB _ _ _ _    => mA.ixLen + mB.ixLen + f.emitLen + 17
  | .ziprow3 _ _ _ _ f mA mB mC _ _ _ _ =>
      mA.ixLen + mB.ixLen + mC.ixLen + f.emitLen + 18

/-- **The number a tape is checked against**, covering both obligations: the
    registers the body names, and the count that bounds the ones the emitter
    allocates.  The factor of three is what `expandEW` can cost. -/
def TOp.regBudget (op : TOp) : Nat := max op.regHi (3 + 3 * op.flenOf)

/-- **Every operation's kernel names registers below its bound, and gathers
    none.**  One case per constructor, each an instantiation of the schema
    lemma above it. -/
theorem TOp.stmt_regNames (batch : Nat) : ∀ op : TOp,
    (∀ r ∈ (op.stmt batch).machRegsOf, r < op.regHi)
      ∧ (op.stmt batch).iregsOf = [] := by
  intro op
  cases op with
  | mv bk w x o b i ow =>
      exact dotBatched_regs w x o (stride32 (.mul .ctaId (.lit i)))
        (fun s => stride32 (.lit (s * i))) (fun s => .add (.lit (s * ow)) .ctaId)
        b (i / 32)
        (by simp [stride32, IdxE.iregsOf])
        (fun s => by simp [stride32, IdxE.iregsOf])
        (fun s => by simp [IdxE.iregsOf])
  | mvT bk w d o b i ow =>
      exact dotBatched_regs w d o
        (.add (.mul (.add (.mul .loopI (.lit 32)) .laneId) (.lit i)) .ctaId)
        (fun s => stride32 (.lit (s * ow))) (fun s => .add (.lit (s * i)) .ctaId)
        b (ow / 32)
        (by simp [IdxE.iregsOf])
        (fun s => by simp [stride32, IdxE.iregsOf])
        (fun s => by simp [IdxE.iregsOf])
  | outer bk d x o b i ow =>
      exact outerBatched_regs d x o (fun s => .add (.lit (s * ow)) .ctaId)
        (fun s => stride32 (.lit (s * i))) (.mul .ctaId (.lit i)) b (i / 32)
        (fun s => by simp [IdxE.iregsOf])
        (fun s => by simp [stride32, IdxE.iregsOf])
        (by simp [IdxE.iregsOf])
  | rowsq x o n rows =>
      exact dotStrided_regs x x o (stride32 (.mul .ctaId (.lit n)))
        (stride32 (.mul .ctaId (.lit n))) .ctaId (n / 32)
        (by simp [stride32, IdxE.iregsOf]) (by simp [stride32, IdxE.iregsOf])
        (by simp [IdxE.iregsOf])
  | rowdot a b o mA mB n rows =>
      exact dotStrided_regs a b o mA.ix mB.ix .ctaId (n / 32)
        (BCast_ix_iregsOf mA) (BCast_ix_iregsOf mB) (by simp [IdxE.iregsOf])
  | rowmax x o n rows init =>
      exact maxStrided_regs x o (stride32 (.mul .ctaId (.lit n))) .ctaId (n / 32) init
        (by simp [stride32, IdxE.iregsOf]) (by simp [IdxE.iregsOf])
  | smce l bi oh o g =>
      exact softmaxCE_regs l bi oh o .laneId (by simp [IdxE.iregsOf])
  | ziprow a b o f mA mB n off w rows =>
      refine ⟨fun r hr => ?_, ?_⟩
      · rw [TOp.stmt, zipPassEW_machRegsOf] at hr
        show r < max 3 f.maxReg
        have h3 : (3 : Nat) ≤ max 3 f.maxReg := Nat.le_max_left _ _
        rcases List.mem_append.mp hr with h | h
        · rcases List.mem_append.mp h with h | h
          · simp only [List.mem_cons, List.not_mem_nil, or_false] at h; omega
          · exact Nat.lt_of_lt_of_le (f.machRegs_lt r h) (Nat.le_max_right _ _)
        · simp only [List.mem_cons, List.not_mem_nil, or_false] at h; omega
      · exact zipPassEW_iregsOf a b o 1 2 0 f mA.ix mB.ix
          (stride32 (.add (.mul .ctaId (.lit n)) (.lit off))) (w / 32)
          (BCast_ix_iregsOf mA) (BCast_ix_iregsOf mB)
          (by simp [stride32, IdxE.iregsOf])
  | ziprow3 a b c o f mA mB mC n off w rows =>
      refine ⟨fun r hr => ?_, ?_⟩
      · rw [TOp.stmt, zip3PassEW_machRegsOf] at hr
        show r < max 4 f.maxReg
        have h4 : (4 : Nat) ≤ max 4 f.maxReg := Nat.le_max_left _ _
        rcases List.mem_append.mp hr with h | h
        · rcases List.mem_append.mp h with h | h
          · simp only [List.mem_cons, List.not_mem_nil, or_false] at h; omega
          · exact Nat.lt_of_lt_of_le (f.machRegs_lt r h) (Nat.le_max_right _ _)
        · simp only [List.mem_cons, List.not_mem_nil, or_false] at h; omega
      · exact zip3PassEW_iregsOf a b c o 1 2 3 0 f mA.ix mB.ix mC.ix
          (stride32 (.add (.mul .ctaId (.lit n)) (.lit off))) (w / 32)
          (BCast_ix_iregsOf mA) (BCast_ix_iregsOf mB) (BCast_ix_iregsOf mC)
          (by simp [stride32, IdxE.iregsOf])
  | ew1 f a o g =>
      obtain ⟨_, p, q⟩ := mapKernelAt_bounds f (fun _ => a) (fun _ => elemIx) o 3
        (fun _ => rfl) (fun _ => rfl)
      exact ⟨fun r hr => by have := p r hr; simp only [TOp.regHi]; omega, q⟩
  | ew2 f a b o g =>
      obtain ⟨_, p, q⟩ := mapKernelAt_bounds f
        (fun j : Fin 2 => if j.val = 0 then a else b) (fun _ => elemIx) o 3
        (fun _ => rfl) (fun _ => rfl)
      exact ⟨fun r hr => by have := p r hr; simp only [TOp.regHi]; omega, q⟩
  | ew3 f a b c o g =>
      obtain ⟨_, p, q⟩ := mapKernelAt_bounds f
        (fun j : Fin 3 => if j.val = 0 then a else if j.val = 1 then b else c)
        (fun _ => elemIx) o 3 (fun _ => rfl) (fun _ => rfl)
      exact ⟨fun r hr => by have := p r hr; simp only [TOp.regHi]; omega, q⟩
  | ew4 f a b c d o g =>
      obtain ⟨_, p, q⟩ := mapKernelAt_bounds f
        (fun j : Fin 4 => if j.val = 0 then a else if j.val = 1 then b else
          if j.val = 2 then c else d)
        (fun _ => elemIx) o 3 (fun _ => rfl) (fun _ => rfl)
      exact ⟨fun r hr => by have := p r hr; simp only [TOp.regHi]; omega, q⟩
  | upd2 f a b g =>
      obtain ⟨_, p, q⟩ := mapKernelAt_bounds f
        (fun j : Fin 2 => if j.val = 0 then a else b) (fun _ => elemIx) a 3
        (fun _ => rfl) (fun _ => rfl)
      exact ⟨fun r hr => by have := p r hr; simp only [TOp.regHi]; omega, q⟩

/-- **…and becomes exactly `flenOf` instructions.** -/
theorem TOp.stmt_flen (batch : Nat) : ∀ op : TOp,
    (op.stmt batch).flenCheap = op.flenOf := by
  intro op
  cases op with
  | mv bk w x o b i ow =>
      rw [TOp.stmt, dotBatched_flenCheap w x o (stride32 (.mul .ctaId (.lit i)))
        (fun s => stride32 (.lit (s * i))) (fun s => .add (.lit (s * ow)) .ctaId)
        5 2 b (i / 32) (fun _ => rfl) (fun _ => rfl)]
      show 6 + b * (5 + 2) + 23 * b + 6 = 30 * b + 12
      omega
  | mvT bk w d o b i ow =>
      rw [TOp.stmt, dotBatched_flenCheap w d o
        (.add (.mul (.add (.mul .loopI (.lit 32)) .laneId) (.lit i)) .ctaId)
        (fun s => stride32 (.lit (s * ow))) (fun s => .add (.lit (s * i)) .ctaId)
        5 2 b (ow / 32) (fun _ => rfl) (fun _ => rfl)]
      show 6 + b * (5 + 2) + 23 * b + 6 = 30 * b + 12
      omega
  | outer bk d x o b i ow =>
      rw [TOp.stmt, outerBatched_flenCheap d x o
        (fun s => .add (.lit (s * ow)) .ctaId) (fun s => stride32 (.lit (s * i)))
        (.mul .ctaId (.lit i)) 2 5 b (i / 32) (fun _ => rfl) (fun _ => rfl)]
      show b * (2 + 5) + 2 + 5 * b + 12 = 12 * b + 14
      omega
  | rowsq x o n rows =>
      rw [TOp.stmt, dotStrided_flenCheap x x o (stride32 (.mul .ctaId (.lit n)))
        (stride32 (.mul .ctaId (.lit n))) .ctaId (n / 32)]
      rfl
  | rowdot a b o mA mB n rows =>
      rw [TOp.stmt, dotStrided_flenCheap a b o mA.ix mB.ix .ctaId (n / 32),
        BCast_ix_emitLen, BCast_ix_emitLen]
      rfl
  | rowmax x o n rows init =>
      rw [TOp.stmt, maxStrided_flenCheap x o (stride32 (.mul .ctaId (.lit n)))
        .ctaId (n / 32) init]
      rfl
  | smce l bi oh o g =>
      rw [TOp.stmt, softmaxCE_flenCheap l bi oh o .laneId]
      rfl
  | ziprow a b o f mA mB n off w rows =>
      rw [TOp.stmt, zipPassEW_flenCheap a b o 1 2 0 f mA.ix mB.ix
        (stride32 (.add (.mul .ctaId (.lit n)) (.lit off))) (w / 32),
        BCast_ix_emitLen, BCast_ix_emitLen]
      show mA.ixLen + mB.ixLen + 8 + f.emitLen + 9
              = mA.ixLen + mB.ixLen + f.emitLen + 17
      omega
  | ziprow3 a b c o f mA mB mC n off w rows =>
      rw [TOp.stmt, zip3PassEW_flenCheap a b c o 1 2 3 0 f mA.ix mB.ix mC.ix
        (stride32 (.add (.mul .ctaId (.lit n)) (.lit off))) (w / 32),
        BCast_ix_emitLen, BCast_ix_emitLen, BCast_ix_emitLen]
      show mA.ixLen + mB.ixLen + mC.ixLen + 8 + f.emitLen + 10
              = mA.ixLen + mB.ixLen + mC.ixLen + f.emitLen + 18
      omega
  | ew1 f a o g =>
      have h := (mapKernelAt_bounds f (fun _ => a) (fun _ => elemIx) o 3
        (fun _ => rfl) (fun _ => rfl)).1
      show ((mapKernelAt f (fun _ => a) (fun _ => elemIx) o).ew).flenCheap = _
      rw [h]; simp only [TOp.flenOf]; omega
  | ew2 f a b o g =>
      have h := (mapKernelAt_bounds f (fun j : Fin 2 => if j.val = 0 then a else b)
        (fun _ => elemIx) o 3 (fun _ => rfl) (fun _ => rfl)).1
      show ((mapKernelAt f _ (fun _ => elemIx) o).ew).flenCheap = _
      rw [h]; simp only [TOp.flenOf]; omega
  | ew3 f a b c o g =>
      have h := (mapKernelAt_bounds f
        (fun j : Fin 3 => if j.val = 0 then a else if j.val = 1 then b else c)
        (fun _ => elemIx) o 3 (fun _ => rfl) (fun _ => rfl)).1
      show ((mapKernelAt f _ (fun _ => elemIx) o).ew).flenCheap = _
      rw [h]; simp only [TOp.flenOf]; omega
  | ew4 f a b c d o g =>
      have h := (mapKernelAt_bounds f
        (fun j : Fin 4 => if j.val = 0 then a else if j.val = 1 then b else
          if j.val = 2 then c else d)
        (fun _ => elemIx) o 3 (fun _ => rfl) (fun _ => rfl)).1
      show ((mapKernelAt f _ (fun _ => elemIx) o).ew).flenCheap = _
      rw [h]; simp only [TOp.flenOf]; omega
  | upd2 f a b g =>
      have h := (mapKernelAt_bounds f (fun j : Fin 2 => if j.val = 0 then a else b)
        (fun _ => elemIx) a 3 (fun _ => rfl) (fun _ => rfl)).1
      show ((mapKernelAt f _ (fun _ => elemIx) a).ew).flenCheap = _
      rw [h]; simp only [TOp.flenOf]; omega

/-- **A kernel's own buffer table is always small enough to name.**

    Nothing about the model enters: the compaction lands every buffer inside a
    table of at most five entries, so this holds at any depth. -/
theorem TOp.localBufs_lt (batch : Nat) (op : TOp) :
    ∀ b ∈ (op.localStmt batch).bufsOf, b < PTX_ADDR_SCRATCH := by
  intro b hb
  simp only [TOp.localStmt, EWStmt.bufsOf_renameBuf, List.mem_map] at hb
  obtain ⟨c, hc, rfl⟩ := hb
  have h2 := compactMap_lt (op.bufs batch) c (TOp.stmt_bufsOf batch op hc)
  have h3 := bufs_le_five batch op
  exact Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le h2 h3) (by decide)

/-- **One arithmetic comparison per operation is enough.**

    The kernel an operation is emitted from names no reserved register, given
    only that its budget fits.  This is the statement that makes the guard
    scale: checking a tape builds no instruction. -/
theorem TOp.kernelRegsOk (batch : Nat) (op : TOp)
    (h : op.regBudget ≤ PTX_ADDR_SCRATCH) :
    (expandEW (op.localStmt batch)).kernelRegsOkB = true := by
  obtain ⟨hm, hi⟩ := TOp.stmt_regNames batch op
  have hf := TOp.stmt_flen batch op
  have hle1 : op.regHi ≤ PTX_ADDR_SCRATCH :=
    Nat.le_trans (Nat.le_max_left _ _) h
  have hle2 : 3 + 3 * op.flenOf ≤ PTX_ADDR_SCRATCH :=
    Nat.le_trans (Nat.le_max_right _ _) h
  simp only [EWStmt.kernelRegsOkB, EWStmt.namesOkB, Bool.and_eq_true,
             List.all_eq_true, decide_eq_true_eq]
  refine ⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩
  · intro r hr
    rw [expandEW_machRegsOf] at hr
    simp only [TOp.localStmt, EWStmt.machRegsOf_renameBuf] at hr
    exact Nat.lt_of_lt_of_le (hm r hr) hle1
  · intro r hr
    rw [expandEW_iregsOf] at hr
    simp only [TOp.localStmt, EWStmt.iregsOf_renameBuf, hi] at hr
    exact absurd hr List.not_mem_nil
  · intro c hc
    rw [expandEW_bufsOf] at hc
    exact TOp.localBufs_lt batch op c hc
  · have hl : (op.localStmt batch).flenCheap = op.flenOf := by
      simp only [TOp.localStmt, EWStmt.flenCheap_renameBuf]; exact hf
    have e1 := expandEW_flenCheap_le (op.localStmt batch)
    rw [hl] at e1
    omega

end AlgorithmLib.ML
