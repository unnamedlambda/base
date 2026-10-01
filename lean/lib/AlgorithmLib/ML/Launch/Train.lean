module
public import AlgorithmLib.ML.Launch.Declared
meta import AlgorithmLib.ML.Launch.Declared
public import AlgorithmLib.ML.Launch.Stages
meta import AlgorithmLib.ML.Launch.Stages
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! # None -/

namespace AlgorithmLib.ML

-- A training run: steps composed over time
-- ---------------------------------------------------------------------------

/-!
  Everything above describes **one** launch sequence from an arbitrary `WSt`.
  A training stack is that sequence repeated, with the weights carried across —
  and "the weights after `n` steps" is not a corollary of "the weights after
  one step", because each step reads what the last one wrote.

  A step is not purely device-side either: this stack computes softmax and the
  cross-entropy gradient on the host, and a model of a training step that
  omitted it would describe a different program.  So the host transformation is
  a *field*, named and carried through the composite rather than assumed away.
-/

/-- Device memory, as the pipeline layer sees it. -/
abbrev Mem := Buf → Nat → Float32

/-- **One training step**: a device pipeline, a host transformation, a device
    pipeline.  The `host` field is what makes this honest — leave it `id` for a
    step that is entirely on the device. -/
structure Step where
  fwd  : Pipeline
  host : Mem → Mem
  bwd  : Pipeline

/-- What a step computes, derived from its three parts. -/
noncomputable def Step.denote (S : Step) (m : Mem) : Mem :=
  S.bwd.denote (S.host (S.fwd.denote m))

/-- Running it: forward, then the host writes back, then backward. -/
def Step.run (S : Step) (st : WSt) : WSt :=
  let a := S.fwd.run st
  S.bwd.run { a with mem := S.host a.mem }

/-- Both halves' blocks are non-racing. -/
def Step.Exclusive (S : Step) : Prop := S.fwd.Exclusive ∧ S.bwd.Exclusive

/-- **A step computes its denotation** — the host transformation composed in,
    with no intermediate memory assumed. -/
theorem Step.run_denote (S : Step) (hex : S.Exclusive) (st : WSt) :
    (S.run st).mem = S.denote st.mem := by
  show (S.bwd.run { S.fwd.run st with mem := S.host (S.fwd.run st).mem }).mem = _
  rw [S.bwd.run_denote hex.2]
  show S.bwd.denote (S.host (S.fwd.run st).mem) = _
  rw [S.fwd.run_denote hex.1]
  rfl

/-- `n` steps of the device program. -/
def Step.iter (S : Step) : Nat → WSt → WSt
  | 0,     st => st
  | n + 1, st => S.run (S.iter n st)

/-- …and `n` applications of what one step computes. -/
noncomputable def iterMem (f : Mem → Mem) : Nat → Mem → Mem
  | 0,     m => m
  | n + 1, m => f (iterMem f n m)

/-- Iterating from an already-stepped memory is stepping the iterate — what
    turns a left fold over `n` copies of one step into `iterMem`. -/
theorem iterMem_comm (f : Mem → Mem) : ∀ (n : Nat) (m : Mem),
    iterMem f n (f m) = f (iterMem f n m) := by
  intro n
  induction n with
  | zero => intro _; rfl
  | succ k ih => intro m; show f (iterMem f k (f m)) = _; rw [ih m]; rfl

/-- **A training run computes the iterate of a step.**

    The statement that makes this a training *stack* rather than a training
    *step*: after `n` steps the memory is `n` applications of `Step.denote`,
    for every `n`, with the weights threaded through by the theorem rather than
    by an assumption about what the previous step left behind. -/
theorem Step.iter_denote (S : Step) (hex : S.Exclusive) :
    ∀ (n : Nat) (st : WSt), (S.iter n st).mem = iterMem S.denote n st.mem := by
  intro n
  induction n with
  | zero => intro _; rfl
  | succ k ih =>
      intro st
      show (S.run (S.iter k st)).mem = _
      rw [S.run_denote hex, ih st]
      rfl

/-- **Two step definitions that denote the same function agree for every run
    length.**  The swap criterion at the level of a training run: a different
    schedule, a fused optimiser, a host step moved onto the device — all are
    drop-in exactly when `Step.denote` agrees. -/
theorem Step.iter_congr (S T : Step) (hS : S.Exclusive) (hT : T.Exclusive)
    (h : S.denote = T.denote) (n : Nat) (st : WSt) :
    (S.iter n st).mem = (T.iter n st).mem := by
  rw [S.iter_denote hS, T.iter_denote hT, h]

/-- **A vendor call assumed to compute what a proven stage computes.**

    This is the shape `Law.cublasIsMatvec` takes at a call site.  The declared
    step's value *is* the proven stage's, so a plan that routes an operation
    through cuBLAS denotes exactly what the all-proven pipeline denotes — the
    difference between the two configurations is which steps are *assumed* and
    which are *proven*, not what they compute.  The assumption stays visible:
    `why` travels in the value and reaches any report. -/
noncomputable def DeclaredStep.ofStage (k : VendorKernel) (S : StageSpec) : DeclaredStep :=
  { kernel := k, outs := [S.out], step := S.step
    frame := fun m b hb => funext (fun a =>
      S.step_otherBuf m b a (by simpa using hb)) }

/-- **A recorded sequence, replayed by one call.**

    The step a `cl_cuda_graph_launch` is: its value is the captured pipeline's
    denotation, and its frame is that pipeline's outputs — derived from the
    stages, so a plan cannot claim a replay touches fewer buffers than the
    sequence it recorded.

    What is assumed is only that the driver replays what was captured.  Which
    stages were captured is proven separately, of the capturing program. -/
noncomputable def graphStep (ss : List XStage) : DeclaredStep :=
  { kernel := .cudaGraphLaunch
    outs   := ss.map (fun S => S.val.out)
    step   := fun m => (Pipeline.ofStages ss).denote m
    frame  := by
      intro m b hb
      show ((ss.map Subtype.val).foldl (fun mm S => S.step mm) m) b = m b
      have : ∀ (L : List XStage) (mm : Buf → Nat → Float32),
          (∀ S ∈ L, b ≠ S.val.out) →
          ((L.map Subtype.val).foldl (fun m' S => S.step m') mm) b = mm b := by
        intro L
        induction L with
        | nil => intro _ _; rfl
        | cons S L ih =>
            intro mm h
            show ((L.map Subtype.val).foldl (fun m' T => T.step m') (S.val.step mm)) b = mm b
            rw [ih (S.val.step mm) (fun T hT => h T (List.mem_cons_of_mem S hT))]
            funext a
            exact StageSpec.step_otherBuf S.val mm b a (h S (List.mem_cons_self ..))
      exact this ss m (fun S hS h => hb (List.mem_map.mpr ⟨S, hS, h.symm⟩)) }

/-- A plan from bundled stages and declared steps — exclusivity comes free, the
    same way `Pipeline.ofStages` gives it for an all-proven sequence. -/
noncomputable def Plan.ofSteps (ss : List (Sum XStage DeclaredStep)) : Plan :=
  ⟨ss.map (fun s => match s with | .inl S => .proven S.val | .inr d => .declared d)⟩

theorem Plan.ofSteps_exclusive (ss : List (Sum XStage DeclaredStep)) :
    (Plan.ofSteps ss).Exclusive := by
  intro S hS
  obtain ⟨x, -, hx⟩ := List.mem_map.mp hS
  cases x with
  | inl T => exact (PStep.proven.inj hx) ▸ T.property
  | inr d => exact absurd hx (by simp)

-- ---------------------------------------------------------------------------

end AlgorithmLib.ML
