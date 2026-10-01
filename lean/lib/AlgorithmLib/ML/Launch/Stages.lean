module
public import AlgorithmLib.ML.Launch.Sequence
meta import AlgorithmLib.ML.Launch.Sequence
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! # None -/

namespace AlgorithmLib.ML

-- Composing stages without writing a proof
-- ---------------------------------------------------------------------------

/-!
  `Pipeline.Exclusive` unfolds to a membership statement, so a user assembling
  a pipeline writes an `rcases` chain over `List.mem_cons` whose length is the
  number of stages.  That is proof text in user code, and the amount of it
  grows with the model.

  A stage that carries its own exclusivity proof removes it.  `Sched` is the
  precedent: the library owns the proofs, the user picks constructors, and the
  composition theorem is uniform in the choice.
-/

/-- A stage bundled with the proof that its blocks do not race. -/
abbrev XStage := { S : StageSpec // S.Exclusive }

/-- **A pipeline from bundled stages** — the list bookkeeping lives here. -/
def Pipeline.ofStages (ss : List XStage) : Pipeline := ⟨ss.map Subtype.val⟩

/-- …and its exclusivity is free, at any length. -/
theorem Pipeline.ofStages_exclusive (ss : List XStage) :
    (Pipeline.ofStages ss).Exclusive := by
  intro S hS
  obtain ⟨⟨_, hT⟩, _, rfl⟩ := List.mem_map.mp hS
  exact hT

/-- …so the composition theorem needs no argument from the caller either. -/
theorem Pipeline.ofStages_runs (ss : List XStage) (st : WSt) :
    ((Pipeline.ofStages ss).run st).mem = (Pipeline.ofStages ss).denote st.mem :=
  (Pipeline.ofStages ss).run_denote (Pipeline.ofStages_exclusive ss) st

/-- An elementwise pass, bundled. -/
def mapStageX {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf)
    (grid : Nat) (h : ∀ i, inB i ≠ out) : XStage :=
  ⟨mapStage spec inB out grid h, mapStage_exclusive _ _ _ _ _⟩

/-- An in-place elementwise pass, bundled — an optimiser step is one. -/
def mapStageIPX {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf)
    (grid : Nat) : XStage :=
  ⟨mapStageIP spec inB out grid, mapStageIP_exclusive _ _ _ _⟩

/-- A row maximum, bundled. -/
def maxRowStageX (b : Buf) (ix : IdxE) (out : Buf) (K grid : Nat) (init : Float32)
    (hb : b ≠ out) : XStage :=
  ⟨maxRowStage b ix out K grid init hb, maxRowStage_exclusive _ _ _ _ _ _ hb⟩

/-- A strided reduction, bundled. -/
def reduceStageX (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (K grid : Nat)
    (h1 : bA ≠ out) (h2 : bB ≠ out) : XStage :=
  ⟨reduceStage bA bB ixA ixB out K grid h1 h2, reduceStage_exclusive _ _ _ _ _ _ _ _ _⟩

/-- The reduction with a row pass folded into it, bundled. -/
def reduce4StageX (bA bB bC bD : Buf) (ixA ixB ixC ixD : IdxE) (f : WFExp)
    (hf : f.tripleOnly = true) (out : Buf) (K grid : Nat)
    (h1 : bA ≠ out) (h2 : bB ≠ out) (h3 : bC ≠ out) (h4 : bD ≠ out) : XStage :=
  ⟨reduce4Stage bA bB bC bD ixA ixB ixC ixD f (fun x y z => f.evalTriple x y z)
     (fun st l => WFExp.evalTriple_eq f hf st l) out K grid h1 h2 h3 h4,
   reduce4Stage_exclusive _ _ _ _ _ _ _ _ _ _ _ _ _ _ h1 h2 h3 h4⟩

/-- Softmax and the cross-entropy gradient, bundled. -/
def softmaxCEStageX (logits bias oneHot out : Buf) (biasIx : IdxE) (grid : Nat)
    (h1 : logits ≠ out) (h2 : bias ≠ out) (h3 : oneHot ≠ out) : XStage :=
  ⟨softmaxCEStage logits bias oneHot out biasIx grid h1 h2 h3,
   softmaxCEStage_exclusive _ _ _ _ _ _ h1 h2 h3⟩

/-- **A row pass, bundled** — the caller gives two addressing modes, a lane
    expression over the two operand registers, and the two disjointness facts.
    Everything else is discharged here. -/
def zipRowStageX (bA bB out : Buf) (f : WFExp) (hf : f.pairOnly = true)
    (mA mB : BCast) (n off K grid : Nat) (hw : off + K * 32 ≤ n)
    (hAo : bA ≠ out) (hBo : bB ≠ out) : XStage :=
  ⟨zipRowStage bA bB out f (fun x y => f.evalPair x y) mA.ix mB.ix
     mA.ev mB.ev n off K grid hw
     (fun st l => WFExp.evalPair_eq f hf st l)
     (BCast.ix_ev mA) (BCast.ix_ev mB) hAo hBo,
   zipRowStage_exclusive _ _ _ _ _ _ _ _ _ _ _ _ _ hw _ _ _ hAo hBo⟩

/-- **A three-operand row pass, bundled.** -/
def zipRow3StageX (bA bB bC out : Buf) (f : WFExp) (hf : f.tripleOnly = true)
    (mA mB mC : BCast) (n off K grid : Nat) (hw : off + K * 32 ≤ n)
    (hAo : bA ≠ out) (hBo : bB ≠ out) (hCo : bC ≠ out) : XStage :=
  ⟨zipRow3Stage bA bB bC out f (fun x y z => f.evalTriple x y z)
     mA.ix mB.ix mC.ix mA.ev mB.ev mC.ev n off K grid hw
     (fun st l => WFExp.evalTriple_eq f hf st l)
     (BCast.ix_ev mA) (BCast.ix_ev mB) (BCast.ix_ev mC) hAo hBo hCo,
   zipRow3Stage_exclusive _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hw _ _ _ _ hAo hBo hCo⟩

/-- The four-operand row pass, bundled. -/
def zipRow4StageX (bA bB bC bD out : Buf) (f : WFExp) (hf : f.quadOnly = true)
    (mA mB mC mD : BCast) (n off K grid : Nat) (hw : off + K * 32 ≤ n)
    (hAo : bA ≠ out) (hBo : bB ≠ out) (hCo : bC ≠ out) (hDo : bD ≠ out) : XStage :=
  ⟨zipRow4Stage bA bB bC bD out f (fun x y z u => f.evalQuad x y z u)
     mA.ix mB.ix mC.ix mD.ix mA.ev mB.ev mC.ev mD.ev n off K grid hw
     (fun st l => WFExp.evalQuad_eq f hf st l)
     (BCast.ix_ev mA) (BCast.ix_ev mB) (BCast.ix_ev mC) (BCast.ix_ev mD)
     hAo hBo hCo hDo,
   zipRow4Stage_exclusive _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hw _ _ _ _ _
     hAo hBo hCo hDo⟩

/-- A batched strided reduction, bundled. -/
def dotBatchedStageX (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (out : Buf)
    (B K grid : Nat) (hg : 0 < grid) (h1 : bA ≠ out) (h2 : bB ≠ out) : XStage :=
  ⟨dotBatchedStage bA bB ixA ixB out B K grid hg h1 h2,
   dotBatchedStage_exclusive _ _ _ _ _ _ _ _ hg _ _⟩

/-- A batched outer product, bundled. -/
def outerBatchedStageX (bA bB out : Buf) (ixA ixB : Nat → IdxE) (n B K grid : Nat)
    (hn : K * 32 = n) (h1 : bA ≠ out) (h2 : bB ≠ out) : XStage :=
  ⟨outerBatchedStage bA bB out ixA ixB n B K grid hn h1 h2,
   outerBatchedStage_exclusive _ _ _ _ _ _ _ _ _ hn _ _⟩

-- ---------------------------------------------------------------------------

end AlgorithmLib.ML
