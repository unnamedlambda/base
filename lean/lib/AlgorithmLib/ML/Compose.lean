import AlgorithmLib.ML.Pipeline
import AlgorithmLib.ML.Rewrite
import AlgorithmLib.ML.Sched
import AlgorithmLib.ML.Transformer

/-!
  # A launch sequence, as data

  `Pipeline.lean` proves what *one* launch leaves in memory.  Composing two of
  them was possible only by writing a bespoke theorem per pair, with the
  composite spelled out in its own conclusion — so there was no object a
  varying implementation could be checked *against*.

  The obstruction was representational.  A launch sequence existed only as the
  order in which an `IRBuilder` happened to emit calls, and an emission order is
  not a value: nothing can quantify over it, so no theorem could mention it.

  A `Pipeline` is that sequence as a list. `run` executes it; `denote` says what
  it computes, by folding each stage's `val` into the memory; `run_denote`
  proves they agree, for any number of stages. The composite is *derived* from
  the pipeline rather than restated, which is what lets two different pipelines
  be compared against one specification.
-/

namespace AlgorithmLib.ML

open Classical in
/-- **What one launch does to memory, as a function on memory.**

    Addresses the grid owns get the stage's value; everything else is
    unchanged.  The owning block is recovered from the ownership proof — under
    `Exclusive` there is at most one, so `step_val` below shows the choice is
    forced and the definition does not depend on it. -/
noncomputable def StageSpec.step (S : StageSpec) (m : Buf → Nat → Float32)
    (b : Buf) (a : Nat) : Float32 :=
  if h : b = S.out ∧ ∃ cta, cta < S.grid ∧ S.dom cta a then
    S.val m h.2.choose a
  else m b a

/-- At an owned address, `step` is the stage's value at *that* block — the
    choice inside `step` is pinned by exclusivity. -/
theorem StageSpec.step_val (S : StageSpec) (hex : S.Exclusive)
    (m : Buf → Nat → Float32) (cta a : Nat) (hlt : cta < S.grid) (hd : S.dom cta a) :
    S.step m S.out a = S.val m cta a := by
  have hex' : S.out = S.out ∧ ∃ c, c < S.grid ∧ S.dom c a := ⟨rfl, ⟨cta, hlt, hd⟩⟩
  rw [StageSpec.step, dif_pos hex']
  have hsp := hex'.2.choose_spec
  exact congrArg (fun c => S.val m c a) (hex _ cta a hsp.1 hlt hsp.2 hd)

/-- Off the output buffer, `step` changes nothing. -/
theorem StageSpec.step_otherBuf (S : StageSpec) (m : Buf → Nat → Float32)
    (b : Buf) (a : Nat) (hb : b ≠ S.out) : S.step m b a = m b a := by
  rw [StageSpec.step, dif_neg (fun h => hb h.1)]

/-- At an address no block owns, `step` changes nothing. -/
theorem StageSpec.step_otherAddr (S : StageSpec) (m : Buf → Nat → Float32) (a : Nat)
    (hno : ∀ c, c < S.grid → ¬ S.dom c a) : S.step m S.out a = m S.out a := by
  rw [StageSpec.step, dif_neg]
  rintro ⟨-, c, hc, hd⟩
  exact hno c hc hd

/-- **One launch realises its `step`.**

    The three cases are exactly the three theorems `Pipeline.lean` already had:
    owned addresses get the value, unowned ones are framed, other buffers are
    framed.  Collecting them into a single equation on memory is what makes
    composition an induction rather than a new proof each time. -/
theorem runGrid_step (S : StageSpec) (hex : S.Exclusive) (st : WSt) :
    (runGrid S.blk S.grid st).mem = S.step st.mem := by
  funext b a
  by_cases hb : b = S.out
  · subst hb
    by_cases hown : ∃ c, c < S.grid ∧ S.dom c a
    · obtain ⟨c, hc, hd⟩ := hown
      rw [runGrid_value S hex a S.grid (Nat.le_refl _) c hc hd st, S.step_val hex st.mem c a hc hd]
    · have hno : ∀ c, c < S.grid → ¬ S.dom c a := fun c hc hd => hown ⟨c, hc, hd⟩
      rw [runGrid_otherAddr S a S.grid st hno, S.step_otherAddr st.mem a hno]
  · rw [congrFun (runGrid_otherBuf S b hb S.grid st) a, S.step_otherBuf st.mem b a hb]

-- ---------------------------------------------------------------------------
-- The sequence
-- ---------------------------------------------------------------------------

/-- **A launch sequence.**  The object an implementation *is*, so that two of
    them can be checked against one specification. -/
structure Pipeline where
  stages : List StageSpec

/-- Execute it: every stage's whole grid, in order. -/
def Pipeline.run (P : Pipeline) (st : WSt) : WSt :=
  P.stages.foldl (fun s S => runGrid S.blk S.grid s) st

/-- **What it computes** — each stage's `val` folded into the memory.  Derived
    from the pipeline, not restated alongside it. -/
noncomputable def Pipeline.denote (P : Pipeline)
    (m : Buf → Nat → Float32) : Buf → Nat → Float32 :=
  P.stages.foldl (fun mm S => S.step mm) m

/-- Every stage owns its addresses exclusively. -/
def Pipeline.Exclusive (P : Pipeline) : Prop := ∀ S ∈ P.stages, S.Exclusive

/-- **The pipeline computes its denotation — at any length.**

    This is the statement that did not exist: a composite over *n* stages,
    with the composite derived from the stage list rather than written into the
    conclusion by hand.  Two pipelines with different fusion or scheduling are
    now comparable, because `denote` is a function of the pipeline and the
    specification is a fixed value on the other side of the equation. -/
theorem foldl_runGrid_step : ∀ (L : List StageSpec), (∀ T ∈ L, T.Exclusive) →
    ∀ (st : WSt),
      (L.foldl (fun s T => runGrid T.blk T.grid s) st).mem
        = L.foldl (fun mm T => T.step mm) st.mem := by
  intro L
  induction L with
  | nil => intro _ _; rfl
  | cons S rest ih =>
      intro hex st
      show (rest.foldl (fun s T => runGrid T.blk T.grid s) (runGrid S.blk S.grid st)).mem
          = rest.foldl (fun mm T => T.step mm) (S.step st.mem)
      rw [← runGrid_step S (hex S (by simp)) st]
      exact ih (fun T hT => hex T (by simp [hT])) _

theorem Pipeline.run_denote (P : Pipeline) (hex : P.Exclusive) (st : WSt) :
    (P.run st).mem = P.denote st.mem :=
  foldl_runGrid_step P.stages hex st

/-- **Two pipelines that denote the same function are interchangeable.**

    The swap criterion: an alternative schedule — more fusion, different block
    counts, a different stage decomposition — is a drop-in replacement exactly
    when its `denote` agrees, and then the memories after running them are
    equal. Nothing here assumes the two have the same number of stages. -/
theorem Pipeline.equiv_of_denote_eq (P Q : Pipeline)
    (hP : P.Exclusive) (hQ : Q.Exclusive)
    (hd : ∀ m, P.denote m = Q.denote m) (st : WSt) :
    (P.run st).mem = (Q.run st).mem := by
  rw [P.run_denote hP st, Q.run_denote hQ st, hd st.mem]

/-- Appending pipelines composes their denotations — the associativity that
    makes "fuse these two stages into one" a statement about `denote`. -/
theorem Pipeline.denote_append (P Q : Pipeline) (m : Buf → Nat → Float32) :
    (Pipeline.mk (P.stages ++ Q.stages)).denote m = Q.denote (P.denote m) := by
  show List.foldl _ m (P.stages ++ Q.stages) = _
  rw [List.foldl_append]; rfl

-- ---------------------------------------------------------------------------
-- Steps that are declared rather than proven
-- ---------------------------------------------------------------------------

/-!
  A pipeline of `StageSpec`s can only describe what has been proven.  The
  shipped inference model is not like that and is not going to be: about 99.9%
  of Qwen2's arithmetic goes through `cl_cublas_sgemv`, whose fold order NVIDIA
  does not specify, so there is no exact-`Float32` statement to make about it.

  Refusing to model it would be dishonest in the other direction — the composite
  would then describe a program nobody runs.  What follows lets a declared step
  sit *in* the sequence, carrying its assumption in the open, so the pipeline
  covers every device write and the number of unproven steps is a value you can
  read off (`Plan.declaredCount`) rather than a caveat in prose.
-/

/-- **A vendor kernel the library knows about.**

    Not a string.  A call site names a constructor, and the symbol to emit, the
    laws the call *assumes*, and the guarantees it explicitly does **not** make
    are read off that constructor by the functions below — associated once,
    here, where they can be checked against the vendor's documentation, rather
    than at each call site where they would be a comment.

    Adding a backend is one constructor and three lines.  Pairing a symbol with
    the wrong law is not expressible. -/
inductive VendorKernel where
  | cublasSgemv
  | cublasSgemvOnStream
  | cublasSgemm
  | cublasSgemmStridedBatched
  | cublasSgemmStridedBatchedOnStream
  /-- The host→device copy.  Not a contraction and not cuBLAS, but declared for
      the same reason: its source is host memory, which this model does not
      describe, so what it lands is assumed while where it can land is proven. -/
  | uploadPtr
  /-- **A captured graph, replayed.**  Not a kernel: one driver call that
      issues a whole recorded sequence.  Declared for the same reason the
      others are — what it lands is assumed, and where it can land is proven —
      except that "where" is a list of buffers rather than one. -/
  | cudaGraphLaunch
  deriving Repr, DecidableEq

/-- **The FFI symbol codegen emits.**  One per constructor, and every one of
    them is a symbol `base/src/jit.rs` registers — which is the check a free
    string could not make. -/
def VendorKernel.symbol : VendorKernel → String
  | .cublasSgemv                      => "cl_cublas_sgemv"
  | .cublasSgemvOnStream              => "cl_cublas_sgemv_on_stream"
  | .cublasSgemm                      => "cl_cublas_sgemm"
  | .cublasSgemmStridedBatched        => "cl_cublas_sgemm_strided_batched"
  | .cublasSgemmStridedBatchedOnStream =>
      "cl_cublas_sgemm_strided_batched_on_stream"
  | .uploadPtr                        => "cl_cuda_upload_ptr"
  | .cudaGraphLaunch                  => "cl_cuda_graph_launch"

/-- **Which stated propositions the call's correctness rests on.**

    A `Law` is an equation something can be rewritten by, so a kernel appears
    here only when such an equation exists for it.  `Law.cublasIsMatvec` says
    what a GEMV lands, and `cublasStep_isMatvec` is where a plan uses it.

    The batched entry points get `[]`, and that is **not** "assumed nothing" —
    it is the stronger position of assumed *without a stated equation*, which
    `lawless` below records.  Reading an empty list as exactness is the mistake
    `Plan.lawlessCount` exists to prevent. -/
def VendorKernel.assumes : VendorKernel → List Law
  | .cublasSgemv                       => [.cublasIsMatvec]
  | .cublasSgemvOnStream               => [.cublasIsMatvec]
  | .cublasSgemm                       => [.cublasIsMatvec]
  -- Batched strided mode picks its operand slices from four stride and batch
  -- arguments the launch model recovers as constants but does not interpret,
  -- and the kernel is free to contract multiply-add pairs, split the
  -- contraction, and accumulate at a different width.  No equation over
  -- `Float32` survives that, so none is stated.
  | .cublasSgemmStridedBatched         => []
  | .cublasSgemmStridedBatchedOnStream => []
  -- What the copy lands is `uploadedValue`, an opaque on the declared trust
  -- surface, rather than an equation this development states.
  | .uploadPtr                         => []
  -- A replay performs the *recorded* sequence.  Which stages those are is
  -- proven (`qwen_capture_records_the_run`); that replay performs them is the
  -- driver's contract, and no equation here states it.
  | .cudaGraphLaunch                   => []

/-- **Every primitive the enum knows.**

    `all_covers` is what lets a report be derived from this rather than from a
    list typed beside it: a constructor added without an entry here fails to
    build, so a bill computed by filtering `all` cannot quietly omit one. -/
def VendorKernel.all : List VendorKernel :=
  [.cublasSgemv, .cublasSgemvOnStream, .cublasSgemm, .cublasSgemmStridedBatched,
   .cublasSgemmStridedBatchedOnStream, .uploadPtr, .cudaGraphLaunch]

theorem VendorKernel.all_covers : ∀ k : VendorKernel, k ∈ VendorKernel.all := by
  intro k; cases k <;> decide

/-- **Whether the call is assumed with no stated equation behind it.**

    The companion to `assumes`: a kernel with an empty law list is either
    covered elsewhere (the upload, by `uploadedValue` on the trust surface) or
    covered nowhere at all.  This distinguishes the two, so a report cannot
    read silence as exactness. -/
def VendorKernel.lawless : VendorKernel → Bool
  | .cublasSgemmStridedBatched         => true
  | .cublasSgemmStridedBatchedOnStream => true
  | .cudaGraphLaunch                   => true
  | _                                  => false

/-- **What the call does *not* guarantee**, in the same breath as what it does.

    A reordering is correct relative to the laws that are declared; this names
    the ones that are deliberately absent, so "cuBLAS is fine here" is a claim
    with a stated scope rather than a shrug. -/
def VendorKernel.withholds : VendorKernel → String
  | .cudaGraphLaunch =>
      "replays the sequence recorded at capture. Which sequence that is, is " ++
      "proven of the capturing program; that the driver replays it is a " ++
      "guarantee of CUDA graphs and is assumed here, in the same standing as " ++
      "`cl_cuda_launch` running the module at the offset it is given."
  | .uploadPtr =>
      "host→device copy; the source is host memory, which this model does not " ++
      "describe. Framed, so it cannot touch any other buffer."
  | .cublasSgemmStridedBatched | .cublasSgemmStridedBatchedOnStream =>
      "batched strided mode selects operand slices by stride and batch-count " ++
      "arguments this model does not interpret, and may contract multiply-add " ++
      "pairs, split the contraction, or accumulate at another width. No " ++
      "Float32 equation is available, so none is claimed: what lands in the " ++
      "output is assumed outright, and only where it can land is proven."
  | _ =>
      "NVIDIA pins no fold order at any version, so no exact-Float32 claim is " ++
      "available: `Law.sumAssoc` is *not* granted and results may differ from " ++
      "the proven kernel's committed order."

/-- **A step whose effect is assumed rather than derived.**

    `frame` is not optional and is not an assumption: it is proven when the step
    is constructed, and it is what stops a declared step being a blank cheque.
    Without it, "assume this does whatever it does" would let the step clobber
    buffers the rest of the pipeline reasons about, and every downstream frame
    argument would silently collapse.  What is assumed is confined to *what
    lands in `out`*; *where it can land* stays proven. -/
structure DeclaredStep where
  /-- Which vendor primitive this is.  The value, not its name: the symbol to
      emit, the laws the call assumes, and the guarantees it withholds are all
      read off it, so a report cannot disagree with what was lowered. -/
  kernel : VendorKernel
  /-- Every buffer the call may write.  A list rather than one buffer because a
      primitive is not always a kernel: a graph replay issues a whole recorded
      sequence in one call, and framing it against a single output would be
      claiming it writes less than it does. -/
  outs  : List Buf
  step  : (Buf → Nat → Float32) → Buf → Nat → Float32
  frame : ∀ (m : Buf → Nat → Float32) (b : Buf), b ∉ outs → step m b = m b

/-- The primitive's symbol, e.g. `cl_cublas_sgemv`. -/
def DeclaredStep.name (d : DeclaredStep) : String := d.kernel.symbol

/-- Why it is not proven — derived from the kernel, so it reaches any report
    and cannot drift from the call that was actually lowered. -/
def DeclaredStep.why (d : DeclaredStep) : String := d.kernel.withholds

/-- The laws this step's correctness rests on. -/
def DeclaredStep.laws (d : DeclaredStep) : List Law := d.kernel.assumes

/-- A step of a plan: a proven stage, or a declared one. -/
inductive PStep where
  | proven   : StageSpec → PStep
  | declared : DeclaredStep → PStep

noncomputable def PStep.denote : PStep → (Buf → Nat → Float32) → (Buf → Nat → Float32)
  | .proven S   => S.step
  | .declared d => d.step

/-- **The whole device-write sequence** — proven and declared steps together, in
    the order the host performs them. -/
structure Plan where
  steps : List PStep

noncomputable def Plan.denote (P : Plan) (m : Buf → Nat → Float32) :
    Buf → Nat → Float32 :=
  P.steps.foldl (fun mm s => s.denote mm) m

/-- The buffers a step writes, proven or declared alike.  A stage writes one;
    a declared call writes what its primitive declares. -/
def PStep.outs : PStep → List Buf
  | .proven S   => [S.out]
  | .declared d => d.outs

/-- Off those buffers, a step changes nothing.  Both cases already carry the
    fact; this is what lets them be used interchangeably in a fold. -/
theorem PStep.denote_otherBuf (s : PStep) (m : Buf → Nat → Float32) (b : Buf)
    (hb : b ∉ s.outs) : s.denote m b = m b := by
  cases s with
  | proven S   =>
      funext a
      exact StageSpec.step_otherBuf S m b a (by simpa [PStep.outs] using hb)
  | declared d => exact d.frame m b hb

/-- **A buffer no later step writes still holds what it held.**

    The workhorse for reading a value *out* of a plan.  A plan's denotation is
    a left fold, so asking what is at one buffer at the end means knowing that
    the steps after the one that wrote it left it alone — which is exactly what
    each step's `frame` field says, and this lifts it to the sequence. -/
theorem denote_frame_list : ∀ (L : List PStep) (m : Buf → Nat → Float32) (b : Buf),
    (∀ s ∈ L, b ∉ s.outs) → (L.foldl (fun mm s => s.denote mm) m) b = m b := by
  intro L
  induction L with
  | nil => intro _ _ _; rfl
  | cons s L ih =>
      intro m b h
      show (L.foldl (fun mm t => t.denote mm) (s.denote m)) b = m b
      rw [ih (s.denote m) b (fun t ht => h t (List.mem_cons_of_mem s ht)),
          PStep.denote_otherBuf s m b (h s (List.mem_cons_self ..))]

/-- The buffers a step list writes, in order.  Stating a frame condition
    against *this* rather than against the steps themselves keeps the side
    goal free of whatever a `StageSpec` is parameterised by — for a plan built
    over an index map or a law hypothesis, the outputs still reduce to
    numerals, so the condition is `decide`-able where the step-level one is
    not. -/
def outsOf (L : List PStep) : List Buf := L.flatMap PStep.outs

/-- `denote_frame_list`, stated against `outsOf`. -/
theorem denote_frame_outs (L : List PStep) (m : Buf → Nat → Float32) (b : Buf)
    (h : ∀ o ∈ outsOf L, b ≠ o) :
    (L.foldl (fun mm s => s.denote mm) m) b = m b :=
  denote_frame_list L m b (fun s hs hmem =>
    h b (List.mem_flatMap.mpr ⟨s, hs, hmem⟩) rfl)

/-- Denotation splits along `++`, so a plan can be read one fragment at a
    time — the value counterpart of `stagesOf?_append`. -/
theorem Plan.denote_append (L₁ L₂ : List PStep) (m : Buf → Nat → Float32) :
    (Plan.mk (L₁ ++ L₂)).denote m
      = (Plan.mk L₂).denote ((Plan.mk L₁).denote m) := by
  show List.foldl _ m (L₁ ++ L₂) = _
  rw [List.foldl_append]
  rfl

/-- **How much of this plan is assumed.**  A number, not a caveat.  It goes to
    zero exactly when every step has a `StageSpec`. -/
def Plan.declaredCount (P : Plan) : Nat :=
  (P.steps.filter (fun s => match s with | .declared _ => true | _ => false)).length

/-- …and *which* primitives they are, so a report can name them. -/
def Plan.declaredNames (P : Plan) : List String :=
  P.steps.filterMap (fun s => match s with | .declared d => some d.name | _ => none)

/-- **The laws a plan rests on**, read off the steps it was lowered to rather
    than recorded beside them.

    This is the build-time answer to "which assumptions did this schedule
    make".  An all-proven plan bills nothing; every entry here is a `Law` with
    a real proposition behind it (`Law.holds`), so the list is checkable rather
    than descriptive. -/
def Plan.lawBill (P : Plan) : List Law :=
  (P.steps.filterMap (fun s => match s with
    | .declared d => some d.laws | _ => none)).flatten.eraseDups

/-- **How many of a plan's steps are assumed with no stated equation.**

    `lawBill` says which propositions a schedule rests on; this says how much
    of it rests on none.  A schedule is described honestly only by both — an
    empty bill beside a nonzero count is a *weaker* claim than an empty bill
    beside a zero one, and printing only the first would invert that. -/
def Plan.lawlessCount (P : Plan) : Nat :=
  (P.steps.filter (fun s => match s with
    | .declared d => d.kernel.lawless | _ => false)).length

/-- What the runtime actually does for a declared step. -/
abbrev Realisation := DeclaredStep → WSt → WSt

def PStep.exec (R : Realisation) : PStep → WSt → WSt
  | .proven S   => fun st => runGrid S.blk S.grid st
  | .declared d => R d

def Plan.run (R : Realisation) (P : Plan) (st : WSt) : WSt :=
  P.steps.foldl (fun s t => t.exec R s) st

/-- **The single assumption the declared steps carry.**

    One named hypothesis for the whole plan, discharged per primitive by its FFI
    contract.  For `cl_cublas_sgemv` that contract is `Law.cublasIsMatvec`,
    stated at ℝ because its `Float32` fold order is unspecified. -/
def Honours (R : Realisation) : Prop := ∀ d st, (R d st).mem = d.step st.mem

/-- Only the *proven* steps owe an exclusivity proof; a declared step's frame
    field already pins where it can write. -/
def Plan.Exclusive (P : Plan) : Prop := ∀ S, PStep.proven S ∈ P.steps → S.Exclusive

theorem foldl_PStep_exec (R : Realisation) (hR : Honours R) : ∀ (L : List PStep),
    (∀ S, PStep.proven S ∈ L → S.Exclusive) → ∀ (st : WSt),
      (L.foldl (fun s t => t.exec R s) st).mem
        = L.foldl (fun mm t => t.denote mm) st.mem := by
  intro L
  induction L with
  | nil => intro _ _; rfl
  | cons t rest ih =>
      intro hex st
      cases t with
      | proven S =>
          show (rest.foldl (fun s u => u.exec R s) (runGrid S.blk S.grid st)).mem
              = rest.foldl (fun mm u => u.denote mm) (S.step st.mem)
          rw [← runGrid_step S (hex S (by simp)) st]
          exact ih (fun T hT => hex T (List.mem_cons_of_mem _ hT)) _
      | declared d =>
          show (rest.foldl (fun s u => u.exec R s) (R d st)).mem
              = rest.foldl (fun mm u => u.denote mm) (d.step st.mem)
          rw [← hR d st]
          exact ih (fun T hT => hex T (List.mem_cons_of_mem _ hT)) _

/-- **A plan computes its denotation — proven and declared steps alike.**

    The generalisation of `Pipeline.run_denote` that covers the program actually
    shipped.  Everything proven stays proven; everything assumed is `Honours R`,
    one hypothesis, and `Plan.declaredCount` says how much of the plan rests on
    it. -/
theorem Plan.run_denote (R : Realisation) (hR : Honours R) (P : Plan)
    (hex : P.Exclusive) (st : WSt) : (P.run R st).mem = P.denote st.mem :=
  foldl_PStep_exec R hR P.steps hex st

-- ---------------------------------------------------------------------------
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
-- Lowering: the backend is a choice, and it never changes what is computed
-- ---------------------------------------------------------------------------

/-!
  A model says `y = W·x`.  Whether that becomes a proven warp kernel or a
  vendor GEMM is not part of the model — it is a property of the *lowering*,
  the same slot `Sched` occupies for schedules and batch strategy occupies for
  batching.

  What makes that safe is `lower_denote`: for **any** stage and **any** backend
  choice, the lowered step denotes exactly what the proven stage denotes.  So a
  model lowered with vendor calls and the same model lowered all-proven leave
  the same memory, and the entire difference between the two builds is which
  steps are assumed — counted by `Plan.declaredCount`, named by
  `Plan.declaredNames`, and discharged by `Honours`.

  This is proven once here rather than per model, which is the point: adding a
  model costs no theorem, and adding a *backend* costs one constructor.
-/

/-- Which implementation an operation gets.  `vendor` names a kernel the
    library knows, so the laws it assumes and the guarantees it withholds travel
    with the choice into any report. -/
inductive Backend where
  | proven : Backend
  | vendor : VendorKernel → Backend
  deriving Repr, DecidableEq

/-- The laws a lowering rests on: one entry per vendor call, none for proven
    steps.  This is the build-time answer to "which assumptions did this
    schedule make". -/
def Backend.laws : Backend → List Law
  | .proven   => []
  | .vendor k => k.assumes

/-- **The only place the backend choice enters.** -/
noncomputable def Backend.lower : Backend → XStage → PStep
  | .proven,   S => .proven S.val
  | .vendor k, S => .declared (DeclaredStep.ofStage k S.val)

/-- **The choice is denotation-preserving, for every stage.**  A vendor step is
    *defined* as computing what the stage computes; `Honours` is the assumption
    that the runtime realises it. -/
theorem Backend.lower_denote (b : Backend) (S : XStage) :
    (b.lower S).denote = S.val.step := by cases b <;> rfl

/-- A model, lowered: one backend choice per stage. -/
noncomputable def lowerAll (choices : List (Backend × XStage)) : Plan :=
  ⟨choices.map (fun c => c.1.lower c.2)⟩

/-- The all-proven pipeline the same stage list denotes. -/
def stagesOf (choices : List (Backend × XStage)) : Pipeline :=
  ⟨choices.map (fun c => c.2.val)⟩

/-- **Any lowering of a model computes what the model computes.**

    Quantified over the whole list of choices, so it covers every mixed build —
    all-proven, all-vendor, and anything between — with no per-model theorem. -/
theorem lowerAll_denote (choices : List (Backend × XStage)) (m : Buf → Nat → Float32) :
    (lowerAll choices).denote m = (stagesOf choices).denote m := by
  show (choices.map _).foldl (fun mm s => PStep.denote s mm) m
     = (choices.map _).foldl (fun mm (S : StageSpec) => S.step mm) m
  induction choices generalizing m with
  | nil => rfl
  | cons c cs ih =>
      show (cs.map _).foldl (fun mm s => PStep.denote s mm) ((c.1.lower c.2).denote m)
         = (cs.map _).foldl (fun mm (S : StageSpec) => S.step mm) (c.2.val.step m)
      rw [congrFun (Backend.lower_denote c.1 c.2) m]
      exact ih _

/-- Only proven steps owe exclusivity, and a bundled stage carries its own. -/
theorem lowerAll_exclusive (choices : List (Backend × XStage)) :
    (lowerAll choices).Exclusive := by
  intro S hS
  obtain ⟨c, -, hc⟩ := List.mem_map.mp hS
  cases hb : c.1 with
  | proven => rw [hb] at hc; exact (PStep.proven.inj hc) ▸ c.2.property
  | vendor k => rw [hb] at hc; exact absurd hc (by simp [Backend.lower])

/-- **A lowered model computes its model's denotation at run time**, for any
    realisation honouring the declared steps.

    This is the whole statement a two-configuration build needs: swap backends
    freely, and what changes is `declaredCount`, not the answer. -/
theorem lowerAll_runs (R : Realisation) (hR : Honours R)
    (choices : List (Backend × XStage)) (st : WSt) :
    (Plan.run R (lowerAll choices) st).mem = (stagesOf choices).denote st.mem := by
  rw [Plan.run_denote R hR _ (lowerAll_exclusive choices) st]
  exact lowerAll_denote choices st.mem

-- ---------------------------------------------------------------------------
-- A dense layer, addressing derived
-- ---------------------------------------------------------------------------

/-!
  Everything above is still written in buffers, index expressions and grids.
  A model should not be.  `Dense` is the smallest useful step: give it two
  widths, a batch and six buffers, and it produces the three stages a dense
  layer needs — forward, input gradient, weight gradient — with every `IdxE`,
  every trip count and every grid *derived* from the shapes.

  The three addressing patterns are the ones a hand-written model gets wrong:
  the forward row walk, the **transposed** walk for `dx` (successive outputs
  are `inW` apart, which is why no `stride32` describes it), and the outer
  product's row base.  Deriving them once is the point.
-/

/-- A dense layer `y = W·x` and its two gradients, at one batch size. -/
structure Dense where
  inW   : Nat
  outW  : Nat
  batch : Nat
  w  : Buf   -- weights, row-major `W[o·inW + i]`
  x  : Buf   -- inputs  `x[s·inW + i]`
  y  : Buf   -- outputs `y[s·outW + o]`
  dy : Buf   -- `∂L/∂y`
  dx : Buf   -- `∂L/∂x`
  dw : Buf   -- `∂L/∂W`

/-- `y[s][o] = Σᵢ W[o][i]·x[s][i]` — one warp per output unit, `batch`
    accumulators, the weight row fetched once for the whole batch. -/
def Dense.fwdStage (d : Dense) (hg : 0 < d.outW := by decide)
    (h1 : d.w ≠ d.y := by decide) (h2 : d.x ≠ d.y := by decide) : XStage :=
  dotBatchedStageX d.w d.x (stride32 (.mul .ctaId (.lit d.inW)))
    (fun s => stride32 (.lit (s * d.inW))) d.y d.batch (d.inW / 32) d.outW hg h1 h2

/-- `dx[s][i] = Σₒ dy[s][o]·W[o][i]` — the transposed walk.  Successive outputs
    are `inW` apart, so this is the one index no `stride32` describes. -/
def Dense.dxStage (d : Dense) (hg : 0 < d.inW := by decide)
    (h1 : d.w ≠ d.dx := by decide) (h2 : d.dy ≠ d.dx := by decide) : XStage :=
  dotBatchedStageX d.w d.dy
    (.add (.mul (.add (.mul .loopI (.lit 32)) .laneId) (.lit d.inW)) .ctaId)
    (fun s => stride32 (.lit (s * d.outW))) d.dx d.batch (d.outW / 32) d.inW hg h1 h2

/-- `dW[o][i] = Σₛ dy[s][o]·x[s][i]` — summed over the batch *inside* one warp,
    because two blocks accumulating into one element would be a race. -/
def Dense.dwStage (d : Dense) (hn : (d.inW / 32) * 32 = d.inW := by decide)
    (h1 : d.dy ≠ d.dw := by decide) (h2 : d.x ≠ d.dw := by decide) : XStage :=
  outerBatchedStageX d.dy d.x d.dw
    (fun s => .add (.lit (s * d.outW)) .ctaId)
    (fun s => stride32 (.lit (s * d.inW)))
    d.inW d.batch (d.inW / 32) d.outW hn h1 h2

/-- **A dense layer's forward stage computes the flat dot product.**

    The kernel folds in the committed two-level order — sequential within a
    lane, then a five-round butterfly — and the model's `Transformer.dot` is a
    flat left fold.  They are the same number only up to reassociation, and
    that reassociation is `Law.laneRegroup`, named in the statement and the only
    thing assumed.  `Expr.denote` of `Transformer.dot n a b` is the fold on the
    right, over `List.finRange` rather than `List.range`.

    This is what makes `Dense` a lowering of a *spec* rather than a kernel with
    a convenient name. -/
theorem Dense.fwd_is_flatDot (h : AllHold [Law.laneRegroup]) (d : Dense)
    (hn : d.inW % 128 = 0) (hg : 0 < d.outW) (h1 : d.w ≠ d.y) (h2 : d.x ≠ d.y)
    (m : Buf → Nat → Float32) (cta a : Nat) :
    (d.fwdStage hg h1 h2).val.val m cta a
      = (List.range d.inW).foldl
          (fun acc i => NumOps.add acc
            (NumOps.mul (m d.w (i + cta * d.inW))
                        (m d.x (i + ((a - cta) / d.outW) * d.inW))))
          (NumOps.ofNat 0) :=
  by
  -- `stride32` writes the base first, `Sched.idx` last: commuted, not defeq.
  have hA : (fun (i : Nat) (l : Lane) =>
        IdxE.eval cta i l (fun _ _ => 0) (fun _ _ => 0)
          (stride32 (.mul .ctaId (.lit d.inW))))
      = Sched.strided.idx d.inW (cta * d.inW) := by
    funext i l
    show cta * d.inW + (i * 32 + l.val) = i * 32 + l.val + cta * d.inW
    omega
  have hB : (fun (i : Nat) (l : Lane) =>
        IdxE.eval cta i l (fun _ _ => 0) (fun _ _ => 0)
          (stride32 (.lit (((a - cta) / d.outW) * d.inW))))
      = Sched.strided.idx d.inW (((a - cta) / d.outW) * d.inW) := by
    funext i l
    show ((a - cta) / d.outW) * d.inW + (i * 32 + l.val)
        = i * 32 + l.val + ((a - cta) / d.outW) * d.inW
    omega
  show bflyFold (dotStridedLane (m d.w) (m d.x)
        (fun i l => IdxE.eval cta i l (fun _ _ => 0) (fun _ _ => 0)
          (stride32 (.mul .ctaId (.lit d.inW))))
        (fun i l => IdxE.eval cta i l (fun _ _ => 0) (fun _ _ => 0)
          (stride32 (.lit (((a - cta) / d.outW) * d.inW))))
        (d.inW / 32)) ⟨0, by decide⟩ = _
  rw [hA, hB]
  exact Sched.fold_eq_flatSum h .strided (m d.w) (m d.x)
    (cta * d.inW) (((a - cta) / d.outW) * d.inW) d.inW hn

-- ---------------------------------------------------------------------------
-- A model as a graph of tensor operations
-- ---------------------------------------------------------------------------

/-!
  `Expr` is a *scalar* language, and `Frontend`'s `Vec`/`Mat` are thin wrappers
  over it — so `w * x` elaborates to `fun i => .sum inW (…)` and the fact that
  it *was* a matvec is gone.  Nothing can then decide whether to lower it to a
  proven warp kernel or to cuBLAS, because there is no longer an operation to
  lower.

  A `Node` keeps the operation.  Each one names its operands by *position in
  the graph*, and a node's output buffer **is** its own index — so the graph is
  simultaneously the program and the allocation, and there is no separate
  buffer table to keep in step with it.

  Malformed and stage-free are kept apart: `Node.stage?` returns `some none`
  for an input (nothing to launch) and `none` for an operation whose operands
  alias its output (nothing sound to launch).  Collapsing the two would let a
  bad node vanish from the plan instead of failing it.
-/

/-- A buffer number. -/
abbrev Ref := Nat

/-- One tensor operation, with the buffer it writes.

    The output is a field rather than the node's position: a backward pass
    visits buffers in an order that is not ascending — `dW2` before `dh` — so
    program order and allocation are genuinely two things. -/
inductive Node where
  /-- A buffer the graph does not compute: an input, a parameter, a label. -/
  | input  : Ref → Node
  /-- `out[s][o] = Σᵢ W[o][i]·x[s][i]`, one warp per output unit. -/
  | matvec : (w x out : Ref) → (b inW outW : Nat) → Node
  /-- `out[s][i] = Σₒ dy[s][o]·W[o][i]` — the transposed walk. -/
  | matvecT : (w dy out : Ref) → (b inW outW : Nat) → Node
  /-- `out[o][i] = Σₛ dy[s][o]·x[s][i]` — the batch sum, inside one warp. -/
  | outer  : (dy x out : Ref) → (b inW outW : Nat) → Node
  /-- An elementwise pass over `grid·32` elements. -/
  | ew     : {Γ : Nat} → Expr Γ → (Fin Γ → Ref) → Ref → Nat → Node
  /-- An elementwise pass that **reads what it writes** — an optimiser step. -/
  | ewIP   : {Γ : Nat} → Expr Γ → (Fin Γ → Ref) → Ref → Nat → Node
  /-- Softmax and the cross-entropy gradient, one warp per row. -/
  | smce   : (logits bias oneHot out : Ref) → Nat → Node
  /-- A two-operand row pass: one row per block, each operand read at its own
      broadcast mode.  This is how a statistic or a shared vector reaches an
      elementwise pass — the row index comes from the block, so no address
      arithmetic the index language cannot express is needed. -/
  | ziprow : (a b out : Ref) → (f : WFExp) → (mA mB : BCast) →
             (n off w rows : Nat) → Node
  /-- The same row pass with a third operand — a cotangent, a statistic, a
      gate.  This is the arity a nonlinear adjoint needs, and the one two
      fused elementwise steps reach. -/
  | ziprow3 : (a b c out : Ref) → (f : WFExp) → (mA mB mC : BCast) →
              (n off w rows : Nat) → Node
  /-- `out[s] = Σᵢ a[…]·b[…]` — one fold per block, each operand addressed by
      its own broadcast mode.  A row sum is this against a shared vector of
      ones; a softmax denominator is that. -/
  | rowdot : (a b out : Ref) → (mA mB : BCast) → (n rows : Nat) → Node
  /-- `out[s] = maxᵢ x[s][i]`, seeded below anything the row can hold.  The
      value a stable softmax subtracts, and the one an argmax reads. -/
  | rowmax : (x out : Ref) → (n rows : Nat) → (init : Float32) → Node
  /-- `out[s] = Σᵢ x[s][i]·x[s][i]` — one row's sum of squares per block.
      The reduction half of an RMS norm, and of any row statistic that is a
      fold of a product. -/
  | rowsq  : (x out : Ref) → (n rows : Nat) → Node

/-- The stage a node lowers to.

    `none` — malformed (an operand aliases an output that may not be read).
    `some none` — nothing to launch.
    `some (some S)` — stage `S`. -/
noncomputable def Node.stage? (batch : Nat) : Node → Option (Option XStage)
  | .input _ => some none
  | .matvec w x out b inW outW =>
      if h : 0 < outW ∧ w ≠ out ∧ x ≠ out then
        some (some (dotBatchedStageX w x (stride32 (.mul .ctaId (.lit inW)))
          (fun s => stride32 (.lit (s * inW))) out b (inW / 32) outW h.1 h.2.1 h.2.2))
      else none
  | .matvecT w dy out b inW outW =>
      if h : 0 < inW ∧ w ≠ out ∧ dy ≠ out then
        some (some (dotBatchedStageX w dy
          (.add (.mul (.add (.mul .loopI (.lit 32)) .laneId) (.lit inW)) .ctaId)
          (fun s => stride32 (.lit (s * outW))) out b (outW / 32) inW
          h.1 h.2.1 h.2.2))
      else none
  | .outer dy x out b inW outW =>
      if h : (inW / 32) * 32 = inW ∧ dy ≠ out ∧ x ≠ out then
        some (some (outerBatchedStageX dy x out
          (fun s => .add (.lit (s * outW)) .ctaId)
          (fun s => stride32 (.lit (s * inW))) inW b (inW / 32) outW
          h.1 h.2.1 h.2.2))
      else none
  | .ew spec ins out grid =>
      if h : ∀ j, ins j ≠ out then some (some (mapStageX spec ins out grid h)) else none
  | .ewIP spec ins out grid => some (some (mapStageIPX spec ins out grid))
  | .smce logits bias oneHot out grid =>
      if h : logits ≠ out ∧ bias ≠ out ∧ oneHot ≠ out then
        some (some (softmaxCEStageX logits bias oneHot out .laneId grid h.1 h.2.1 h.2.2))
      else none
  | .ziprow a b out f mA mB n off w rows =>
      if h : f.pairOnly = true ∧ (w / 32) * 32 = w ∧ off + w ≤ n
             ∧ a ≠ out ∧ b ≠ out then
        some (some (zipRowStageX a b out f h.1 mA mB n off (w / 32) rows
          (by rw [h.2.1]; exact h.2.2.1) h.2.2.2.1 h.2.2.2.2))
      else none
  | .rowdot a b out mA mB n rows =>
      if h : (n / 32) * 32 = n ∧ a ≠ out ∧ b ≠ out then
        some (some (reduceStageX a b mA.ix mB.ix out (n / 32) rows h.2.1 h.2.2))
      else none
  | .ziprow3 a b c out f mA mB mC n off w rows =>
      if h : f.tripleOnly = true ∧ (w / 32) * 32 = w ∧ off + w ≤ n
             ∧ a ≠ out ∧ b ≠ out ∧ c ≠ out then
        some (some (zipRow3StageX a b c out f h.1 mA mB mC n off (w / 32) rows
          (by rw [h.2.1]; exact h.2.2.1) h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2))
      else none
  | .rowmax x out n rows init =>
      if h : (n / 32) * 32 = n ∧ x ≠ out then
        some (some (maxRowStageX x (stride32 (.mul .ctaId (.lit n))) out
          (n / 32) rows init h.2))
      else none
  | .rowsq x out n rows =>
      if h : (n / 32) * 32 = n ∧ x ≠ out then
        some (some (reduceStageX x x (stride32 (.mul .ctaId (.lit n)))
          (stride32 (.mul .ctaId (.lit n))) out (n / 32) rows h.2 h.2))
      else none

/-- A model: operations in the order the driver performs them. -/
abbrev Net := List Node

/-- Lower the graph.  A malformed node fails the whole lowering rather than
    being skipped — collapsing "nothing to launch" into "nothing sound to
    launch" would let a bad node vanish from the plan instead of failing it. -/
noncomputable def lowerNet (batch : Nat) : Net → Option (List XStage)
  | []      => some []
  | n :: ns =>
      match n.stage? batch with
      | none          => none
      | some none     => lowerNet batch ns
      | some (some S) => (lowerNet batch ns).map (S :: ·)

/-- **The stages a model's graph lowers to**, in graph order. -/
noncomputable def Net.stages (batch : Nat) (g : Net) : Option (List XStage) :=
  lowerNet batch g

-- ---------------------------------------------------------------------------
-- Writing the graph as equations
-- ---------------------------------------------------------------------------

/-!
  A `Net` written as a list still names its own buffers.  The builder below
  removes that too: each operation *returns* the buffer it wrote, so a model is
  a sequence of bindings and the allocation is whatever the binder order
  produced.

      let z1 ← mv   w1 x  IN H
      let h  ← elem hSpec ![z1] GRIDH
      let y  ← mv   w2 h  H  C

  That is the `let h := silu (w1 * x)` surface, in the one form Lean gives for
  free: `do`-notation binds the name, and the name *is* the value, so an
  operand that does not exist is a scope error rather than a wrong number.
-/

/-- A tensor's extent.  A weight is `outW × inW`; an activation is
    `batch × width`; a per-class vector is `1 × C`.  Two numbers is all the
    shapes in this stack need, and it is enough to derive every width, trip
    count and launch grid. -/
structure Shape where
  rows : Nat
  cols : Nat
  deriving Repr, DecidableEq

/-- The builder records, per binding, the operation **and** which
    implementation it chose.  A model program leaves every choice `.proven`; a
    *schedule* is the same program with some choices replaced, and `forget`
    below is what relates the two. -/
structure BuildState where
  next   : Nat := 0
  nodes  : List (Node × Backend) := []
  shapes : List Shape := []
  /-- Cleared by any shape mismatch, so a malformed model yields `none` rather
      than a plausible graph. -/
  ok     : Bool := true

abbrev NetM := StateM BuildState

def shapeOf (r : Ref) : NetM Shape :=
  fun st => (st.shapes.getD r ⟨0, 0⟩, st)

def failBuild : NetM Unit := fun st => ((), { st with ok := false })

/-- Allocate the next buffer, at a known shape. -/
def alloc (s : Shape) : NetM Ref :=
  fun st => (st.next, { st with next := st.next + 1, shapes := st.shapes ++ [s] })

def emitAt (n : Node) (b : Backend) : NetM Unit :=
  fun st => ((), { st with nodes := st.nodes ++ [(n, b)] })

/-- `W · x`, at the given implementation.  Widths and grid are **derived**: the
    contraction is `W.cols = x.cols`, and a mismatch fails the build. -/
def mvWith (b : Backend) (w x : Ref) : NetM Ref := do
  let sw ← shapeOf w
  let sx ← shapeOf x
  if sw.cols ≠ sx.cols then failBuild
  let o ← alloc ⟨sx.rows, sw.rows⟩
  emitAt (.matvec w x o sx.rows sw.cols sw.rows) b
  pure o

/-- `Wᵀ · dy` — the input gradient. -/
def mvTWith (b : Backend) (w dy : Ref) : NetM Ref := do
  let sw ← shapeOf w
  let sd ← shapeOf dy
  if sd.cols ≠ sw.rows then failBuild
  let o ← alloc ⟨sd.rows, sw.cols⟩
  emitAt (.matvecT w dy o sd.rows sw.cols sw.rows) b
  pure o

/-- An elementwise pass.  Grid is derived from the operand's extent. -/
def elemWith {Γ : Nat} (b : Backend) (spec : Expr Γ) (ins : Fin Γ → Ref) : NetM Ref := do
  if h : 0 < Γ then
    let s ← shapeOf (ins ⟨0, h⟩)
    let o ← alloc s
    emitAt (.ew spec ins o (s.rows * s.cols / 32)) b
    pure o
  else do failBuild; pure 0

/-! The model-level names: every choice `.proven`.  A schedule uses the `With`
    forms above to say otherwise. -/
def mv      : Ref → Ref → NetM Ref := mvWith .proven
def mvT     : Ref → Ref → NetM Ref := mvTWith .proven
/-- **A schedule, with its choices erased, is a model.**  The relation between
    the two programs: same operations, same operands, same allocation — only
    the implementations differ. -/
def forget (ps : List (Node × Backend)) : Net := ps.map Prod.fst

/-- Lower a scheduled graph: each node to its stage, carrying the choice the
    schedule made.  A malformed node fails the whole lowering. -/
noncomputable def lowerPairs (batch : Nat) :
    List (Node × Backend) → Option (List (Backend × XStage))
  | []           => some []
  | (n, b) :: ps =>
      match n.stage? batch with
      | none          => none
      | some none     => lowerPairs batch ps
      | some (some S) => (lowerPairs batch ps).map ((b, S) :: ·)

/-- **The stages a scheduled graph lowers to do not depend on the schedule.**

    Erasing the choices gives the model, and the model determines the stages —
    so a schedule can only change *which implementation* runs, never *what*
    runs.  This is the structural half of the guarantee; `lowerAll_denote` is
    the semantic half. -/
theorem lowerPairs_forget (batch : Nat) :
    ∀ (ps : List (Node × Backend)),
      (lowerPairs batch ps).map (List.map Prod.snd) = lowerNet batch (forget ps) := by
  intro ps
  induction ps with
  | nil => rfl
  | cons p ps ih =>
      obtain ⟨n, b⟩ := p
      show (match n.stage? batch with
            | none => none
            | some none => lowerPairs batch ps
            | some (some S) => (lowerPairs batch ps).map ((b, S) :: ·)).map _
          = match n.stage? batch with
            | none => none
            | some none => lowerNet batch (forget ps)
            | some (some S) => (lowerNet batch (forget ps)).map (S :: ·)
      cases n.stage? batch with
      | none => rfl
      | some o =>
          cases o with
          | none => exact ih
          | some S =>
              rw [← ih]
              cases lowerPairs batch ps <;> rfl

/-- **Two programs that erase to the same model lower to the same stages.**
    The schedule-checking theorem: write the schedule as a program, and if it
    is the model with implementations chosen, nothing about *what* is computed
    can have moved. -/
theorem schedule_agrees (batch : Nat) (sched model : List (Node × Backend))
    (h : forget sched = forget model) :
    (lowerPairs batch sched).map (List.map Prod.snd)
      = (lowerPairs batch model).map (List.map Prod.snd) := by
  rw [lowerPairs_forget, lowerPairs_forget, h]

/-- **A node's decidable signature**: its kind, every buffer it names, and every
    width and grid it carries.

    `Node` cannot have `DecidableEq` — `ew` holds an `Expr` whose `sum` case
    carries a function, and its arity is implicit — so an erasure check has to
    compare something decidable.  This captures **the whole wiring**: which
    operation, which operands (`List.ofFn` unfolds the elementwise inputs), the
    output, the widths and the grid.

    What it does *not* compare is the elementwise `Expr` itself.  A schedule
    that silently substituted a different activation would pass this check, so
    a schedule is checked to have the same *structure*, not the same
    arithmetic. -/
def Node.sig : Node → Nat × List Ref × List Nat
  | .input o                     => (0, [o], [])
  | .matvec w x o b inW outW     => (1, [w, x, o], [b, inW, outW])
  | .matvecT w dy o b inW outW   => (2, [w, dy, o], [b, inW, outW])
  | .outer dy x o b inW outW     => (3, [dy, x, o], [b, inW, outW])
  | .ew (Γ := Γ) _ ins o grid    => (4, List.ofFn ins ++ [o], [Γ, grid])
  | .ewIP (Γ := Γ) _ ins o grid  => (5, List.ofFn ins ++ [o], [Γ, grid])
  | .smce l b oh o grid          => (6, [l, b, oh, o], [grid])
  | .rowsq x o n r               => (7, [x, o], [n, r])
  | .rowdot a b o mA mB n r      => (9, [a, b, o], [mA.tag, mB.tag, n, r])
  | .rowmax x o n r _            => (11, [x, o], [n, r])
  | .ziprow a b o _ mA mB n f w r => (8, [a, b, o], [mA.tag, mB.tag, n, f, w, r])
  | .ziprow3 a b c o _ mA mB mC n f w r =>
      (10, [a, b, c, o], [mA.tag, mB.tag, mC.tag, n, f, w, r])

-- ---------------------------------------------------------------------------
-- Tensor terms: shapes in the type, sharing in the term
-- ---------------------------------------------------------------------------

/-!
  A `Node` list still names its own buffers and carries its own grids.  `Ten`
  removes both: an operation's widths *are* its type indices, so `mv` accepts
  only a matching contraction and the launch grid is derived rather than
  written.  A shape mismatch is a unification failure at elaboration, naming
  both extents — there is no shape checker, no `ok` flag and no `Option`.

  ## Sharing is declared, not discovered

  Lean's own `let` substitutes, so `let h := silu (w1 * x)` used twice would
  flatten to two kernels.  `letT` is the binder that does not: it flattens its
  right-hand side once and hands the body the buffer.  This is `Frontend.letIn`
  one level up, and the reason is the same — it is what keeps a term linear in
  the depth of the network rather than exponential.

  The body takes a *variable*, not a term, which is what makes `Ten` a legal
  inductive: a body of type `Ten r c → Ten r' c'` puts `Ten` left of an arrow
  and is not strictly positive.  Parameterising by the variable representation
  `v` moves the recursive occurrence back to a positive position, and a program
  quantified over `v` cannot inspect what a variable *is* — so it cannot depend
  on the buffer numbers it will be given.

  ## Everything reduces

  `flat` is ordinary structural recursion into a first-order list.  Nothing
  here is a `StateM`, so the erasure check between a schedule and its model is
  `rfl` — which compares the elementwise `Expr` payloads too, and needs no
  `DecidableEq`.
-/

/-- The variable representation used when a program is flattened: a variable
    stands for the buffer its right-hand side was written to. -/
abbrev RefV : Nat → Nat → Type := fun _ _ => Ref

/-- A tensor expression of extent `r × c`, over variable representation `v`. -/
inductive Ten (v : Nat → Nat → Type) : Nat → Nat → Type where
  /-- A `letT`-bound intermediate. -/
  | var   : {r c : Nat} → v r c → Ten v r c
  /-- A buffer the program does not compute: an input, a parameter, a label. -/
  | inp   : {r c : Nat} → Ref → Ten v r c
  /-- `y[s][o] = Σᵢ W[o][i]·x[s][i]`. The contraction width is shared by the
      two operand types, so a mismatch does not elaborate. -/
  | mv    : {b i o : Nat} → Backend → Ten v o i → Ten v b i → Ten v b o
  /-- `dx[s][i] = Σₒ dy[s][o]·W[o][i]` — the transposed walk. -/
  | mvT   : {b i o : Nat} → Backend → Ten v o i → Ten v b o → Ten v b i
  /-- `dW[o][i] = Σₛ dy[s][o]·x[s][i]` — the batch sum. -/
  | outer : {b i o : Nat} → Backend → Ten v b o → Ten v b i → Ten v o i
  /-- A unary elementwise pass. -/
  | ew1   : {r c : Nat} → Expr 1 → Ten v r c → Ten v r c
  /-- A binary elementwise pass; both operands carry the same extent. -/
  | ew2   : {r c : Nat} → Expr 2 → Ten v r c → Ten v r c → Ten v r c
  /-- A ternary elementwise pass — a gated feed-forward (`silu(gate) * up`)
      or a fused residual-and-scale.  All three operands share an extent. -/
  | ew3   : {r c : Nat} → Expr 3 → Ten v r c → Ten v r c → Ten v r c → Ten v r c
  /-- Softmax and the cross-entropy gradient, one warp per row. -/
  | smce  : {b c : Nat} → Ten v b c → Ten v 1 c → Ten v b c → Ten v b c
  /-- `s[j] = Σᵢ x[j][i]²` — one scalar per row.  The extent drops to a single
      column, so a statistic cannot be fed to an elementwise pass that expects
      a full row: the broadcast has to be written. -/
  | rowsq : {r c : Nat} → Ten v r c → Ten v r 1
  /-- A row pass whose second operand is one scalar per row — a norm's
      statistic reaching the elements it scales. -/
  | zipS  : {r c : Nat} → WFExp → Ten v r c → Ten v r 1 → Ten v r c
  /-- A row pass whose second operand is one vector shared by every row — a
      gain, a bias, a rotation table. -/
  | zipB  : {r c : Nat} → WFExp → Ten v r c → Ten v 1 c → Ten v r c
  /-- **Rotary position embedding** on rows of width `c`, split at `c/2`:
      the cosine and sine tables are shared by every row. -/
  | rope  : {r c : Nat} → Ten v r c → Ten v 1 c → Ten v 1 c → Ten v r c
  /-- `out[s] = Σᵢ a[s][i]·b[i]` — a row fold against a shared vector.  Against
      a vector of ones it is a row sum, which is a softmax's denominator. -/
  | rowB  : {r c : Nat} → Ten v r c → Ten v 1 c → Ten v r 1
  /-- `out[s] = maxᵢ t[s][i]`, from a seed below anything the row holds. -/
  | rowMax : {r c : Nat} → Float32 → Ten v r c → Ten v r 1
  /-- A row pass over three operands: the row, one value per row, one vector
      shared by every row.  What a fused norm is. -/
  | zip3  : {r c : Nat} → WFExp → Ten v r c → Ten v r 1 → Ten v 1 c → Ten v r c
  /-- A row pass whose second operand is **one element**, at a fixed index of
      another tensor.  A router's gate for one expert is read this way. -/
  | zipC  : {r c e : Nat} → WFExp → Nat → Ten v r c → Ten v 1 e → Ten v r c
  /-- Reinterpret the extent at equal size.  Buffers are row-major and
      contiguous, so this is a *view*: it emits no operation, and the equality
      is what stops it being a reshape that would need one. -/
  | view  : {r c r' c' : Nat} → r * c = r' * c' → Ten v r c → Ten v r' c'
  /-- A name for the buffer this subterm lands in.  Emits nothing and changes
      nothing: `flat` passes straight through it, so a labelled model flattens
      to the identical tape.  It exists so a *schedule* can say where to act by
      the name the model already bound, instead of by a position in the
      compiler's output. -/
  | named : String → Ten v r c → Ten v r c
  /-- An in-place update of the first operand, then the rest of the program.
      An optimiser step is a statement rather than a value: it names no new
      buffer, it overwrites a parameter. -/
  | upd2  : {r c r' c' : Nat} → Expr 2 → Ten v r c → Ten v r c →
            Ten v r' c' → Ten v r' c'
  /-- `let x = rhs in body x` — the binder that flattens `rhs` once. -/
  | letT  : {r c r' c' : Nat} → Ten v r c → (v r c → Ten v r' c') →
            Ten v r' c'

/-- **A flattened operation.**  The same seven operations as `Ten`, with the
    buffers resolved and — unlike `Node` — every arity *definite*.  That is
    what lets a reverse pass match on an elementwise op without a dependent
    match on an implicit `Expr` arity. -/
inductive TOp where
  | mv    : Backend → (w x out : Ref) → (b inW outW : Nat) → TOp
  | mvT   : Backend → (w dy out : Ref) → (b inW outW : Nat) → TOp
  | outer : Backend → (dy x out : Ref) → (b inW outW : Nat) → TOp
  | ew1   : Expr 1 → (a out : Ref) → (grid : Nat) → TOp
  | ew2   : Expr 2 → (a b out : Ref) → (grid : Nat) → TOp
  | ew3   : Expr 3 → (a b c out : Ref) → (grid : Nat) → TOp
  | ew4   : Expr 4 → (a b c d out : Ref) → (grid : Nat) → TOp
  | smce  : (l bias oh out : Ref) → (grid : Nat) → TOp
  | rowsq  : (x out : Ref) → (n rows : Nat) → TOp
  | ziprow : (a b out : Ref) → WFExp → BCast → BCast → (n off w rows : Nat) → TOp
  | ziprow3 : (a b c out : Ref) → WFExp → BCast → BCast → BCast →
              (n off w rows : Nat) → TOp
  | rowdot : (a b out : Ref) → BCast → BCast → (n rows : Nat) → TOp
  | rowmax : (x out : Ref) → (n rows : Nat) → Float32 → TOp
  | upd2  : Expr 2 → (a b : Ref) → (grid : Nat) → TOp

/-- The graph node an operation is, with its implementation choice. -/
def TOp.node : TOp → Node × Backend
  | .mv bk w x o b i ow  => (.matvec w x o b i ow, bk)
  | .mvT bk w d o b i ow => (.matvecT w d o b i ow, bk)
  | .outer bk d x o b i ow => (.outer d x o b i ow, bk)
  | .ew1 f a o g         => (.ew f (fun _ => a) o g, .proven)
  | .ew2 f a b o g       => (.ew f (fun j : Fin 2 => if j.val = 0 then a else b) o g, .proven)
  | .ew3 f a b c o g     => (.ew f (fun j : Fin 3 => if j.val = 0 then a else if j.val = 1 then b else c) o g, .proven)
  | .ew4 f a b c d o g   => (.ew f (fun j : Fin 4 => if j.val = 0 then a else if j.val = 1 then b else if j.val = 2 then c else d) o g, .proven)
  | .smce l bi oh o g    => (.smce l bi oh o g, .proven)
  | .rowsq x o n r       => (.rowsq x o n r, .proven)
  | .rowdot a b o mA mB n r => (.rowdot a b o mA mB n r, .proven)
  | .rowmax x o n r i    => (.rowmax x o n r i, .proven)
  | .ziprow a b o f mA mB n k w r => (.ziprow a b o f mA mB n k w r, .proven)
  | .ziprow3 a b c o f mA mB mC n k w r =>
      (.ziprow3 a b c o f mA mB mC n k w r, .proven)
  | .upd2 f a b g        => (.ewIP f (fun j : Fin 2 => if j.val = 0 then a else b) a g, .proven)

/-- **The derivative of a lane expression with respect to one register.**

    `Option`-valued: `maxW` and `geF` are not differentiable, and `ex2` is the
    hardware's approximation rather than a function this stack has a derivative
    rule for.  A gradient that reached one of them is a build failure, not a
    silently wrong number. -/
def WFExp.deriv : WFExp → Nat → Option WFExp
  | .reg r',   r => some (if r' == r then .lit (NumOps.ofNat 1) else .lit (NumOps.ofNat 0))
  | .lit _,    _ => some (.lit (NumOps.ofNat 0))
  | .add a b,  r => match a.deriv r, b.deriv r with
      | some da, some db => some (.add da db)
      | _, _ => none
  | .mul a b,  r => match a.deriv r, b.deriv r with
      | some da, some db => some (.add (.mul da b) (.mul a db))
      | _, _ => none
  | .neg a,    r => (a.deriv r).map (fun da => .neg da)
  | .inv a,    r => (a.deriv r).map (fun da => .neg (.mul da (.mul (.inv a) (.inv a))))
  | .exp a,    r => (a.deriv r).map (fun da => .mul da (.exp a))
  | .rsqrt a,  r => (a.deriv r).map (fun da =>
      .neg (.mul da (.mul (.inv (.lit (NumOps.ofNat 2))) (.mul (.rsqrt a) (.inv a)))))
  | .ex2 _,    _ => none
  | .maxW _ _, _ => none
  | .geF _ _,  _ => none

/-- The adjoint pass's lane expression: the incoming gradient in register 3,
    times the forward expression's derivative in the operand's register. -/
def adjW (f : WFExp) (r : Nat) : Option WFExp :=
  (f.deriv r).map (fun d => .mul (.reg 3) d)

/-- **The scalar language, as lane code.**  Variable `i` becomes register
    `i+1`, which is the convention the row passes bind their operands to.

    `sum` and `letE` have no lane counterpart — a lane expression is evaluated
    once per element, with no reduction and no sharing — so they map to junk and
    `Expr.laneable` is what says a term avoids them. -/
def Expr.toWF : {Γ : Nat} → Expr Γ → WFExp
  | _, .var i    => .reg (i.val + 1)
  | _, .lit n    => .lit (NumOps.ofNat n)
  | _, .add a b  => .add a.toWF b.toWF
  | _, .mul a b  => .mul a.toWF b.toWF
  | _, .neg a    => .neg a.toWF
  | _, .inv a    => .inv a.toWF
  | _, .exp a    => .exp a.toWF
  | _, .rsqrt a  => .rsqrt a.toWF
  | _, .sum _ _  => .lit (NumOps.ofNat 0)
  | _, .letE _ _ => .lit (NumOps.ofNat 0)

def Expr.laneable : {Γ : Nat} → Expr Γ → Bool
  | _, .var _    => true
  | _, .lit _    => true
  | _, .add a b  => a.laneable && b.laneable
  | _, .mul a b  => a.laneable && b.laneable
  | _, .neg a    => a.laneable
  | _, .inv a    => a.laneable
  | _, .exp a    => a.laneable
  | _, .rsqrt a  => a.laneable
  | _, .sum _ _  => false
  | _, .letE _ _ => false

/-- **The lane-level derivative is the proven scalar one, translated.**

    `Expr.sderiv` is tied to the analytic derivative by `grad_hasDerivAt`; this
    says the rule a row pass differentiates by is that same rule read as lane
    code, not a second differentiator that happens to look similar.  Every
    constructor's operand order is what makes it an equality rather than a
    rearrangement. -/
theorem Expr.toWF_deriv : ∀ {Γ : Nat} (e : Expr Γ), e.laneable = true →
    ∀ (i : Fin Γ), (e.toWF).deriv (i.val + 1) = some (sderiv e i).toWF := by
  intro Γ e
  induction e with
  | var k =>
      intro _ i
      by_cases hij : i = k
      · subst hij
        simp [Expr.toWF, WFExp.deriv, sderiv]
      · have hb : ¬ (k.val + 1 = i.val + 1) := fun hc => hij (Fin.ext (by omega))
        simp only [Expr.toWF, WFExp.deriv, sderiv, if_neg hij,
          show (k.val + 1 == i.val + 1) = false from by
            simp only [beq_eq_false_iff_ne, ne_eq]; exact hb,
          if_false]
        rfl
  | lit n => intro _ _; rfl
  | add a b iha ihb =>
      intro h i
      have h' := Bool.and_eq_true .. |>.mp h
      show (match (a.toWF).deriv (i.val + 1), (b.toWF).deriv (i.val + 1) with
            | some da, some db => some (WFExp.add da db) | _, _ => none) = _
      rw [iha h'.1 i, ihb h'.2 i]
      rfl
  | mul a b iha ihb =>
      intro h i
      have h' := Bool.and_eq_true .. |>.mp h
      show (match (a.toWF).deriv (i.val + 1), (b.toWF).deriv (i.val + 1) with
            | some da, some db => some (WFExp.add (.mul da b.toWF) (.mul a.toWF db))
            | _, _ => none) = _
      rw [iha h'.1 i, ihb h'.2 i]
      rfl
  | neg a ih => intro h i; show ((a.toWF).deriv _).map _ = _; rw [ih h i]; rfl
  | inv a ih => intro h i; show ((a.toWF).deriv _).map _ = _; rw [ih h i]; rfl
  | exp a ih => intro h i; show ((a.toWF).deriv _).map _ = _; rw [ih h i]; rfl
  | rsqrt a ih => intro h i; show ((a.toWF).deriv _).map _ = _; rw [ih h i]; rfl
  | sum n f _ => intro h; exact absurd h (by simp [Expr.laneable])
  | letE a b _ _ => intro h; exact absurd h (by simp [Expr.laneable])

/-- The three combinations a rotation is built from, in the scalar language —
    so `Expr.toWF_deriv` applies to them and their adjoints are `sderiv`.

    A norm's scale is deliberately *not* written this way: its `1/n` and its
    epsilon are reciprocals, and `Expr` has only natural-number literals, so an
    image of it would emit a reciprocal instruction per element where a
    constant belongs. -/
def mulE : Expr 2 := .mul (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)
def addE : Expr 2 := .add (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)
def subE : Expr 2 := .add (.var ⟨0, by decide⟩) (.neg (.var ⟨1, by decide⟩))

def mulW : WFExp := Expr.toWF mulE
def addW : WFExp := Expr.toWF addE
def subW : WFExp := Expr.toWF subE

/-- …and they are the lane code they always were. -/
theorem laneW_unchanged :
    mulW = .mul (.reg 1) (.reg 2)
      ∧ addW = .add (.reg 1) (.reg 2)
      ∧ subW = .add (.reg 1) (.neg (.reg 2)) := ⟨rfl, rfl, rfl⟩

/-- **The kernel an operation launches**, as text-emittable code.

    `Node.stage?` cannot be run: a `StageSpec` carries a `Prop`-valued domain,
    so it is noncomputable, and the PTX has to come from somewhere that
    executes.  This is that somewhere — and `qwen_kernels_are_the_stages` is
    what stops it being a second definition of the model: it says, on the
    shipped list, that these are the very statements the proven stages carry. -/
def TOp.stmt (batch : Nat) : TOp → EWStmt
  | .mv _ w x o b inW outW =>
      dotBatched w x (stride32 (.mul .ctaId (.lit inW)))
        (fun s => stride32 (.lit (s * inW))) o
        (fun s => .add (.lit (s * outW)) .ctaId) b (inW / 32)
  | .mvT _ w d o b inW outW =>
      dotBatched w d (.add (.mul (.add (.mul .loopI (.lit 32)) .laneId) (.lit inW)) .ctaId)
        (fun s => stride32 (.lit (s * outW))) o
        (fun s => .add (.lit (s * inW)) .ctaId) b (outW / 32)
  | .outer _ d x o b inW outW =>
      outerBatched d x o (fun s => .add (.lit (s * outW)) .ctaId)
        (fun s => stride32 (.lit (s * inW))) (.mul .ctaId (.lit inW)) b (inW / 32)
  | .ew1 f a o _         => (mapKernel f (fun _ => a) o).ew
  | .ew2 f a b o _       =>
      (mapKernel f (fun j : Fin 2 => if j.val = 0 then a else b) o).ew
  | .ew3 f a b c o _     =>
      (mapKernel f (fun j : Fin 3 =>
        if j.val = 0 then a else if j.val = 1 then b else c) o).ew
  | .ew4 f a b c d o _   =>
      (mapKernel f (fun j : Fin 4 =>
        if j.val = 0 then a else if j.val = 1 then b else
          if j.val = 2 then c else d) o).ew
  | .smce l bi oh o _    => softmaxCE l bi oh o .laneId
  | .upd2 f a b _        =>
      (mapKernel f (fun j : Fin 2 => if j.val = 0 then a else b) a).ew
  | .rowsq x o n _       =>
      dotStrided x x (stride32 (.mul .ctaId (.lit n)))
        (stride32 (.mul .ctaId (.lit n))) o .ctaId (n / 32)
  | .rowdot a b o mA mB n _ => dotStrided a b mA.ix mB.ix o .ctaId (n / 32)
  | .rowmax x o n _ init =>
      maxStrided x (stride32 (.mul .ctaId (.lit n))) o .ctaId (n / 32) init
  | .ziprow a b o f mA mB n off w _ =>
      zipPassEW a b o 1 2 0 f mA.ix mB.ix
        (stride32 (.add (.mul .ctaId (.lit n)) (.lit off))) (w / 32)
  | .ziprow3 a b c o f mA mB mC n off w _ =>
      zip3PassEW a b c o 1 2 3 0 f mA.ix mB.ix mC.ix
        (stride32 (.add (.mul .ctaId (.lit n)) (.lit off))) (w / 32)

/-- **The blocks an operation launches over** — the stage's grid, on the
    runnable side.  `qwen_grids_are_the_stages` is what keeps it honest. -/
def TOp.gridOf : TOp → Nat
  | .mv _ _ _ _ _ _ outW    => outW
  | .mvT _ _ _ _ _ inW _    => inW
  | .outer _ _ _ _ _ _ outW => outW
  | .ew1 _ _ _ g          => g
  | .ew2 _ _ _ _ g        => g
  | .ew3 _ _ _ _ _ g      => g
  | .ew4 _ _ _ _ _ _ g    => g
  | .smce _ _ _ _ g       => g
  | .upd2 _ _ _ g         => g
  | .rowsq _ _ _ rows     => rows
  | .rowdot _ _ _ _ _ _ r => r
  | .rowmax _ _ _ r _     => r
  | .ziprow _ _ _ _ _ _ _ _ _ r => r
  | .ziprow3 _ _ _ _ _ _ _ _ _ _ _ r => r

/-- **The buffer an operation writes, and how many bytes it needs.**

    A host allocates from this rather than from a table written beside the
    model, so a shape that changed in the model changes the allocation. -/
def TOp.outSize (batch : Nat) : TOp → Ref × Nat
  | .mv _ _ _ o b _ outW    => (o, b * outW * 4)
  | .mvT _ _ _ o b inW _    => (o, b * inW * 4)
  | .outer _ _ _ o _ inW outW => (o, outW * inW * 4)
  | .ew1 _ _ o g          => (o, g * 32 * 4)
  | .ew2 _ _ _ o g        => (o, g * 32 * 4)
  | .ew3 _ _ _ _ o g      => (o, g * 32 * 4)
  | .ew4 _ _ _ _ _ o g    => (o, g * 32 * 4)
  | .smce _ _ _ o g       => (o, g * 32 * 4)
  | .upd2 _ a _ g         => (a, g * 32 * 4)
  | .rowsq _ o _ rows     => (o, rows * 4)
  | .rowdot _ _ o _ _ _ r => (o, r * 4)
  | .rowmax _ o _ r _     => (o, r * 4)
  | .ziprow _ _ o _ _ _ n _ _ r => (o, r * n * 4)
  | .ziprow3 _ _ _ o _ _ _ _ n _ _ r => (o, r * n * 4)

/-- Flatten to the graph, allocating computed buffers from `n` upward.

    Returns the buffer the term landed in, the next free buffer, and the
    operations in the order the driver performs them. -/
def Ten.flat : {r c : Nat} → Ten RefV r c → Ref → Ref × Ref × List TOp
  | _, _, .var r, n => (r, n, [])
  | _, _, .inp r, n => (r, n, [])
  | _, _, @Ten.mv _ b i o bk w x, n =>
      let (rw, n1, fw) := w.flat n
      let (rx, n2, fx) := x.flat n1
      (n2, n2 + 1, fw ++ fx ++ [TOp.mv bk rw rx n2 b i o])
  | _, _, @Ten.mvT _ b i o bk w d, n =>
      let (rw, n1, fw) := w.flat n
      let (rd, n2, fd) := d.flat n1
      (n2, n2 + 1, fw ++ fd ++ [TOp.mvT bk rw rd n2 b i o])
  | _, _, @Ten.outer _ b i o bk d x, n =>
      let (rd, n1, fd) := d.flat n
      let (rx, n2, fx) := x.flat n1
      (n2, n2 + 1, fd ++ fx ++ [TOp.outer bk rd rx n2 b i o])
  | r, c, .ew1 spec a, n =>
      let (ra, n1, fa) := a.flat n
      (n1, n1 + 1, fa ++ [TOp.ew1 spec ra n1 (r * c / 32)])
  | r, c, .ew2 spec a b, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      (n2, n2 + 1, fa ++ fb ++
        [TOp.ew2 spec ra rb n2 (r * c / 32)])
  | r, c, .ew3 spec a b d, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      let (rd, n3, fd) := d.flat n2
      (n3, n3 + 1, fa ++ fb ++ fd ++ [TOp.ew3 spec ra rb rd n3 (r * c / 32)])
  | r, _, @Ten.rowsq _ _ c x, n =>
      let (rx, n1, fx) := x.flat n
      (n1, n1 + 1, fx ++ [TOp.rowsq rx n1 c r])
  | r, c, @Ten.zipS _ _ _ f a b, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      (n2, n2 + 1, fa ++ fb ++ [TOp.ziprow ra rb n2 f (.rowOf c 0) .scalar c 0 c r])
  | r, c, @Ten.zipB _ _ _ f a b, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      (n2, n2 + 1, fa ++ fb ++ [TOp.ziprow ra rb n2 f (.rowOf c 0) (.sharedAt 0) c 0 c r])
  | _, _, .view _ t, n => t.flat n
  | _, _, .named _ t, n => t.flat n
  | r, c, @Ten.zipC _ _ _ _ f k a b, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      (n2, n2 + 1, fa ++ fb ++
        [TOp.ziprow ra rb n2 f (.rowOf c 0) (.constAt k) c 0 c r])
  | r, c, @Ten.zip3 _ _ _ f a b d, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      let (rd, n3, fd) := d.flat n2
      (n3, n3 + 1, fa ++ fb ++ fd ++
        [TOp.ziprow3 ra rb rd n3 f (.rowOf c 0) .scalar (.sharedAt 0) c 0 c r])
  | r, _, @Ten.rowMax _ _ c init x, n =>
      let (rx, n1, fx) := x.flat n
      (n1, n1 + 1, fx ++ [TOp.rowmax rx n1 c r init])
  | r, _, @Ten.rowB _ _ c a b, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      (n2, n2 + 1, fa ++ fb ++
        [TOp.rowdot ra rb n2 (.rowOf c 0) (.sharedAt 0) c r])
  | r, c, @Ten.rope _ _ _ x cosT sinT, n =>
      let (rx, n1, fx) := x.flat n
      let (rco, n2, fco) := cosT.flat n1
      let (rsi, n3, fsi) := sinT.flat n2
      let h := c / 2
      let t1 := n3; let t2 := n3 + 1; let t3 := n3 + 2; let t4 := n3 + 3
      let o := n3 + 4
      (o, n3 + 5, fx ++ fco ++ fsi ++
        [ TOp.ziprow rx rco t1 mulW (.rowOf c 0) (.sharedAt 0) h 0 h r
        , TOp.ziprow rx rsi t2 mulW (.rowOf c h) (.sharedAt 0) h 0 h r
        , TOp.ziprow rx rco t3 mulW (.rowOf c h) (.sharedAt h) h 0 h r
        , TOp.ziprow rx rsi t4 mulW (.rowOf c 0) (.sharedAt h) h 0 h r
        , TOp.ziprow t1 t2 o subW (.rowOf h 0) (.rowOf h 0) c 0 h r
        , TOp.ziprow t3 t4 o addW (.rowOf h 0) (.rowOf h 0) c h h r ])
  | b, _, .smce l bias oh, n =>
      let (rl, n1, fl) := l.flat n
      let (rbi, n2, fbi) := bias.flat n1
      let (roh, n3, foh) := oh.flat n2
      (n3, n3 + 1, fl ++ fbi ++ foh ++ [TOp.smce rl rbi roh n3 b])
  | _, _, @Ten.upd2 _ r c _ _ spec a b rest, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      let (rr, n3, fr) := rest.flat n2
      (rr, n3, fa ++ fb ++
        [TOp.upd2 spec ra rb (r * c / 32)] ++ fr)
  | _, _, .letT a k, n =>
      let (ra, n1, fa) := a.flat n
      let (rk, n2, fk) := (k ra).flat n1
      (rk, n2, fa ++ fk)

/-- **Which buffer each name in the model refers to.**

    Descends *every* constructor, so a name is reachable wherever it was
    written — including inside a library combinator, which is where a model
    assembled from `blockAt` and `rmsNorm` binds most of its intermediates.

    The allocation is never recomputed: each subterm's base is read off
    `Ten.flat` itself (`(w.flat n).2.1` is what `flat` hands its next operand),
    so this repeats which children exist and in what order, and nothing about
    numbering.  A name's buffer is then `(a.flat n).1` by definition — the
    buffer that subterm lands in.

    Not load-bearing: a schedule that resolves a name to the wrong buffer gets
    a site the fusion guard refuses, which costs an unfused kernel and never a
    wrong answer. -/
def Ten.labels : {r c : Nat} → Ten RefV r c → Ref → List (String × Ref)
  | _, _, .named s a, n => (s, (a.flat n).1) :: a.labels n
  | _, _, .var _, _ => []
  | _, _, .inp _, _ => []
  | _, _, .view _ t, n => t.labels n
  | _, _, @Ten.mv _ _ _ _ _ w x, n => w.labels n ++ x.labels (w.flat n).2.1
  | _, _, @Ten.mvT _ _ _ _ _ w d, n => w.labels n ++ d.labels (w.flat n).2.1
  | _, _, @Ten.outer _ _ _ _ _ d x, n => d.labels n ++ x.labels (d.flat n).2.1
  | _, _, .ew1 _ a, n => a.labels n
  | _, _, .ew2 _ a b, n => a.labels n ++ b.labels (a.flat n).2.1
  | _, _, .ew3 _ a b d, n =>
      a.labels n ++ b.labels (a.flat n).2.1
        ++ d.labels ((b.flat (a.flat n).2.1).2.1)
  | _, _, .smce l bias oh, n =>
      l.labels n ++ bias.labels (l.flat n).2.1
        ++ oh.labels ((bias.flat (l.flat n).2.1).2.1)
  | _, _, .rowsq x, n => x.labels n
  | _, _, .rowMax _ x, n => x.labels n
  | _, _, @Ten.zipS _ _ _ _ a b, n => a.labels n ++ b.labels (a.flat n).2.1
  | _, _, @Ten.zipB _ _ _ _ a b, n => a.labels n ++ b.labels (a.flat n).2.1
  | _, _, @Ten.zipC _ _ _ _ _ _ a b, n => a.labels n ++ b.labels (a.flat n).2.1
  | _, _, @Ten.rowB _ _ _ a b, n => a.labels n ++ b.labels (a.flat n).2.1
  | _, _, @Ten.rope _ _ _ x cosT sinT, n =>
      x.labels n ++ cosT.labels (x.flat n).2.1
        ++ sinT.labels ((cosT.flat (x.flat n).2.1).2.1)
  | _, _, @Ten.zip3 _ _ _ _ a b d, n =>
      a.labels n ++ b.labels (a.flat n).2.1
        ++ d.labels ((b.flat (a.flat n).2.1).2.1)
  | _, _, @Ten.upd2 _ _ _ _ _ _ a b rest, n =>
      a.labels n ++ b.labels (a.flat n).2.1
        ++ rest.labels ((b.flat (a.flat n).2.1).2.1)
  | _, _, .letT a k, n =>
      a.labels n ++ (k (a.flat n).1).labels (a.flat n).2.1

/-- The graph a term flattens to, with computed buffers allocated from
    `base` upward. -/
def Ten.graphOf {r c : Nat} (t : Ten RefV r c) (base : Ref) : List (Node × Backend) :=
  ((t.flat base).2.2).map TOp.node

-- ---------------------------------------------------------------------------
-- The surface
-- ---------------------------------------------------------------------------

/-!
  `w * x` is a contraction: the shared width lives in both operand types, so
  the operator is total on the terms it accepts and a mismatch is a
  unification failure.  The elementwise functions keep their `Transformer`
  definitions rather than re-spelling them, so a model written here and the
  shipped kernels cannot drift.

  `tlet x := e; body` is the binder.  It elaborates to `letT`, so a value used
  twice is one buffer, and it is a macro rather than a monad — there is no
  bind to reduce through and no `←`.
-/

instance {v : Nat → Nat → Type} {b i o : Nat} :
    HMul (Ten v o i) (Ten v b i) (Ten v b o) := ⟨Ten.mv Backend.proven⟩

/-- The residual add and the gated feed-forward, as scalar specs. -/
def addW2 : Expr 2 := .add (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)
def gatedW2 : Expr 2 := .mul (Transformer.silu (.var ⟨0, by decide⟩)) (.var ⟨1, by decide⟩)

/-- `silu(x) = x·σ(x)` — `Transformer.silu`, the spec the shipped kernel is
    proven against. -/
def Ten.silu {v : Nat → Nat → Type} {r c : Nat} (t : Ten v r c) : Ten v r c :=
  .ew1 (Transformer.silu (.var ⟨0, by decide⟩)) t

/-- **RMSNorm** — `out[s][i] = γ[i]·x[s][i]·rsqrt(Σⱼ x[s][j]²/n + ε)`, as three
    row passes: the sum of squares, the scale, the gain.

    `x` is mentioned twice, so bind it with `tlet` first.  A bound variable
    flattens to no operation at all, which is what keeps the statistic and the
    scaled row reading one buffer rather than two computations of it. -/
def rmsScaleW (c : Nat) (eps : Float32) : WFExp :=
  .mul (.reg 1)
    (.rsqrt (.add (.mul (.reg 2) (.lit (NumOps.inv (NumOps.ofNat c)))) (.lit eps)))

def Ten.rmsNorm {v : Nat → Nat → Type} {r c : Nat} (eps : Float32)
    (gamma : Ten v 1 c) (x : Ten v r c) (nm : String := "norm") : Ten v r c :=
  .zipB mulW (.named (nm ++ ".scale") (.zipS (rmsScaleW c eps) x x.rowsq)) gamma

/-- **The same norm as two passes instead of three** — the scale and the gain
    in one kernel.  `fuse_ziprow_den` is what says the number is the same one,
    bit for bit: `WFExp.fuseA` keeps every operation in its original order, so
    this is a schedule that needs no law. -/
def Ten.rmsNormFused {v : Nat → Nat → Type} {r c : Nat} (eps : Float32)
    (gamma : Ten v 1 c) (x : Ten v r c) (nm : String := "norm") : Ten v r c :=
  .named (nm ++ ".fused") (.zip3 (WFExp.fuseA mulW (rmsScaleW c eps)) x x.rowsq gamma)

/-- A seed for a row maximum: below any logit a model produces. -/
def softmaxFloor : Float32 := NumOps.neg (NumOps.ofNat 1000000000)

/-- **Row-wise softmax** — the row maximum, the shift, the exponential, the row
    sum against a shared vector of ones, and the scale.  Five passes, all
    proven stages.

    Subtracting the maximum is what keeps the exponential in range, and it is
    the same shift `Transformer.softmaxAt` takes as a parameter — the kernel
    now computes it rather than being handed one. -/
def Ten.softmaxRow {v : Nat → Nat → Type} {r c : Nat}
    (ones : Ten v 1 c) (t : Ten v r c) : Ten v r c :=
  .letT t (fun tb =>
    .letT (Ten.rowMax softmaxFloor (.var tb)) (fun mx =>
      .letT (.zipS (.add (.reg 1) (.neg (.reg 2))) (.var tb) (.var mx)) (fun sh =>
        .letT (.ew1 (.exp (.var ⟨0, by decide⟩)) (.var sh)) (fun e =>
          .zipS (.mul (.reg 1) (.inv (.reg 2))) (.var e)
            (Ten.rowB (.var e) ones)))))

/-- **One attention head at a decode step** — scores, softmax, weighted sum.

    Heads are rows.  At a single position the whole head is two matrix-vector
    products against the cache and a row-wise softmax between them, so it needs
    no index beyond `row × column`: `q · Kᵀ` is `mv`, `p · V` is `mvT`. -/
def Ten.attnHead {v : Nat → Nat → Type} {hd sq nh : Nat} (impl : Backend)
    (ones : Ten v 1 sq) (kc vc : Ten v sq hd) (q : Ten v nh hd) : Ten v nh hd :=
  .letT (.ew1 (.mul (.var ⟨0, by decide⟩) (.rsqrt (.lit hd))) (.mv impl kc q)) (fun sc =>
    .letT (Ten.softmaxRow ones (.var sc)) (fun p =>
      .mvT impl vc (.var p)))

/-- One expert: a gated feed-forward. -/
def Ten.expert {v : Nat → Nat → Type} {dm dff : Nat} (impl : Backend)
    (w1 w3 : Ten v dff dm) (w2 : Ten v dm dff) (x : Ten v 1 dm) : Ten v 1 dm :=
  .mv impl w2 (.ew2 gatedW2 (.mv impl w1 x) (.mv impl w3 x))

/-- Experts `0…k`, each scaled by its own gate and added. -/
def Ten.moeAcc {v : Nat → Nat → Type} {dm dff nE : Nat} (impl : Backend)
    (w : Nat → Ten v dff dm × Ten v dff dm × Ten v dm dff) (g : Ten v 1 nE)
    (x : Ten v 1 dm) : Nat → Ten v 1 dm
  | 0     => .zipC mulW 0 (Ten.expert impl (w 0).1 (w 0).2.1 (w 0).2.2 x) g
  | k + 1 => .ew2 addW2 (Ten.moeAcc impl w g x k)
               (.zipC mulW (k + 1)
                 (Ten.expert impl (w (k+1)).1 (w (k+1)).2.1 (w (k+1)).2.2 x) g)

/-- **The expert slots a launch sequence has, with the gates the host chose.**

    `moe` below evaluates every expert because one graph is one launch
    sequence.  This is the other half of the same vocabulary: `nUsed` slots,
    each reading the weights bound to it and scaled by element `j` of a gate
    row the host packed.  Which expert a slot *is* appears nowhere in the term
    — it is which buffers the bind array points slot `j` at, so choosing
    experts costs a rebinding and no new kernel.

    The gate row is an operand rather than the router's own output for the same
    reason: slot `j` reads gate `j`, and the router scored expert `e`, so
    something has to compact `nE` scores into `nUsed`.  The host does it with
    the read it already makes to rank them. -/
def Ten.moeSlots {v : Nat → Nat → Type} {dm dff nUsed : Nat} (impl : Backend)
    (w : Nat → Ten v dff dm × Ten v dff dm × Ten v dm dff)
    (g : Ten v 1 nUsed) (x : Ten v 1 dm) : Ten v 1 dm :=
  .letT x (fun xb => Ten.moeAcc impl w g (.var xb) (nUsed - 1))

/-- **The router on its own** — a projection and a row-wise softmax.

    Separate from the experts because the host runs it first, reads the row it
    writes, and only then knows which weights to bind. -/
def Ten.router {v : Nat → Nat → Type} {dm nE : Nat} (impl : Backend)
    (ones : Ten v 1 nE) (wr : Ten v nE dm) (x : Ten v 1 dm) : Ten v 1 nE :=
  Ten.softmaxRow ones (.mv impl wr x)

/-- **A mixture-of-experts feed-forward.**  The router is a projection and a
    row-wise softmax; each expert is a gated feed-forward; the gate for expert
    `e` is one element of the router row, read at a constant address.

    Every expert is evaluated.  That is the *dense* reading of the mixture — it
    computes the right answer and none of the saving, because choosing which
    experts to run is a host decision about which kernels to launch, and this
    graph describes one launch sequence. Sparsity is a rebinding, not a term. -/
def Ten.moe {v : Nat → Nat → Type} {dm dff nE : Nat} (impl : Backend)
    (ones : Ten v 1 nE) (wr : Ten v nE dm)
    (w : Nat → Ten v dff dm × Ten v dff dm × Ten v dm dff) (nUsed : Nat)
    (x : Ten v 1 dm) : Ten v 1 dm :=
  .letT x (fun xb =>
    .letT (Ten.softmaxRow ones (.mv impl wr (.var xb))) (fun g =>
      Ten.moeAcc impl w (.var g) (.var xb) nUsed))

/-- **A pre-norm transformer block**, at one decode position.

    Norm, projections, rotation, attention against the cache, the output
    projection and the residual; then norm, a gated feed-forward and the second
    residual.  Every step is a `Ten` constructor that lowers to a proven stage,
    so a block costs no new proof text — only the two `view`s that say the
    residual stream and the head layout are the same bytes.

    The cache is read, not appended to: an append is a host-side write this
    model does not describe. -/
def Ten.block {v : Nat → Nat → Type} {dm nh hd sq dff : Nat} (impl : Backend)
    (hv : 1 * dm = nh * hd) (hv' : nh * hd = 1 * dm) (eps : Float32)
    (g1 g2 : Ten v 1 dm) (wq wo : Ten v dm dm) (w1 w3 : Ten v dff dm)
    (w2 : Ten v dm dff) (cosT sinT : Ten v 1 hd) (ones : Ten v 1 sq)
    (kc vc : Ten v sq hd) (x : Ten v 1 dm) (nm : String := "block") : Ten v 1 dm :=
  .letT x (fun xb =>
    .letT (.named (nm ++ ".n1") (Ten.rmsNorm eps g1 (.var xb) (nm ++ ".attn"))) (fun n1 =>
      .letT (.named (nm ++ ".q") (.rope (.view hv (.mv impl wq (.var n1))) cosT sinT)) (fun q =>
        .letT (.named (nm ++ ".att") (.mv impl wo
                (.view hv' (Ten.attnHead impl ones kc vc (.var q))))) (fun att =>
          .letT (.named (nm ++ ".xr") (.ew2 addW2 (.var xb) (.var att))) (fun xr =>
            .letT (.named (nm ++ ".n2") (Ten.rmsNorm eps g2 (.var xr) (nm ++ ".ffn"))) (fun n2 =>
              .letT (.named (nm ++ ".ff")
                      (.ew2 gatedW2 (.mv impl w1 (.var n2)) (.mv impl w3 (.var n2))))
                (fun ff => .ew2 addW2 (.var xr) (.mv impl w2 (.var ff)))))))))

/-- **A block's shape, with the tiling condition carried rather than written.**

    A rotation needs the residual stream and the head layout to be the same
    bytes, which is an arithmetic fact about the widths — `1 * dm = nh * hd`.
    Making it a field with a tactic default means a model states its widths and
    nothing else: the condition is discharged where the widths are given, and a
    geometry that does not tile fails there with the false proposition printed,
    rather than at the use site. -/
structure BlockGeom where
  dm : Nat
  nh : Nat
  hd : Nat
  sq : Nat
  dff : Nat
  tiles : 1 * dm = nh * hd := by decide

/-- **A block at a geometry** — the same term as `Ten.block`, with the two
    proofs taken from the geometry and the implementation defaulted.

    This is the surface a model is written against: widths, weights, and the
    residual stream.  `impl` is trailing and defaulted because it is a schedule
    choice, not part of what the block computes; a model that never mentions it
    gets the proven kernels. -/
def Ten.blockAt {v : Nat → Nat → Type} (G : BlockGeom) (eps : Float32)
    (g1 g2 : Ten v 1 G.dm) (wq wo : Ten v G.dm G.dm) (w1 w3 : Ten v G.dff G.dm)
    (w2 : Ten v G.dm G.dff) (cosT sinT : Ten v 1 G.hd) (ones : Ten v 1 G.sq)
    (kc vc : Ten v G.sq G.hd) (x : Ten v 1 G.dm)
    (impl : Backend := .proven) (nm : String := "block") : Ten v 1 G.dm :=
  Ten.block impl G.tiles G.tiles.symm eps g1 g2 wq wo w1 w3 w2 cosT sinT ones kc vc x nm

/-- **The same block, scheduled with both norms fused.**

    Identical structure, two kernels shorter.  `fuse_ziprow_den` is what says it
    is the same block rather than a different one. -/
def Ten.blockFused {v : Nat → Nat → Type} {dm nh hd sq dff : Nat} (impl : Backend)
    (hv : 1 * dm = nh * hd) (hv' : nh * hd = 1 * dm) (eps : Float32)
    (g1 g2 : Ten v 1 dm) (wq wo : Ten v dm dm) (w1 w3 : Ten v dff dm)
    (w2 : Ten v dm dff) (cosT sinT : Ten v 1 hd) (ones : Ten v 1 sq)
    (kc vc : Ten v sq hd) (x : Ten v 1 dm) (nm : String := "block") : Ten v 1 dm :=
  .letT x (fun xb =>
    .letT (Ten.rmsNormFused eps g1 (.var xb) (nm ++ ".attn")) (fun n1 =>
      .letT (.rope (.view hv (.mv impl wq (.var n1))) cosT sinT) (fun q =>
        .letT (.mv impl wo
                (.view hv' (Ten.attnHead impl ones kc vc (.var q)))) (fun att =>
          .letT (.ew2 addW2 (.var xb) (.var att)) (fun xr =>
            .letT (Ten.rmsNormFused eps g2 (.var xr) (nm ++ ".ffn")) (fun n2 =>
              .letT (.ew2 gatedW2 (.mv impl w1 (.var n2)) (.mv impl w3 (.var n2)))
                (fun ff => .ew2 addW2 (.var xr) (.mv impl w2 (.var ff)))))))))

/-- An elementwise combination of two tensors of the same extent. -/
def Ten.zipWith {v : Nat → Nat → Type} {r c : Nat} (f : Expr 2)
    (a b : Ten v r c) : Ten v r c := .ew2 f a b

/-- Send one operation to a vendor implementation, leaving what it computes
    alone.  The tag a schedule adds to an otherwise unchanged equation. -/
def Ten.on {v : Nat → Nat → Type} {r c : Nat} (impl : Backend) :
    Ten v r c → Ten v r c
  | .mv _ w x    => .mv impl w x
  | .mvT _ w d   => .mvT impl w d
  | .outer _ d x => .outer impl d x
  | t            => t

/-- The forward spec of a unary elementwise pass, from an ordinary Lean
    function on expressions. -/
def ewFwd (f : Expr 1 → Expr 1) : Expr 1 := f (.var ⟨0, by decide⟩)

/-- **The backward spec of a unary elementwise pass, derived from the same
    Lean function.**  `∂L/∂in = ∂L/∂out · f'(in)`, with `f'` taken
    symbolically by `sderiv` — input 0 is the primal, input 1 the incoming
    cotangent.

    Writing the activation once and reading both passes off it is the property
    that makes training and inference one source rather than two hand-written
    graphs. -/
def ewBack (f : Expr 2 → Expr 2) : Expr 2 :=
  .mul (.var ⟨1, by decide⟩) (sderiv (f (.var ⟨0, by decide⟩)) ⟨0, by decide⟩)

/-- Weaken a unary spec into the binary context a backward pass works in. -/
def wk1 (e : Expr 1) : Expr 2 := rename (Fin.castLE (by decide)) e

/-- The backward spec of a unary elementwise op, taken from its *forward*
    spec — the term-level counterpart of `ewBack`. -/
def ewBackOf (f : Expr 1) : Expr 2 :=
  .mul (.var ⟨1, by decide⟩) (sderiv (wk1 f) ⟨0, by decide⟩)

/-- Weaken a binary spec into the ternary context its adjoints work in. -/
def wk2 (e : Expr 2) : Expr 3 := rename (Fin.castLE (by decide)) e

/-- The adjoint of a *binary* elementwise op with respect to input `i`:
    `∂L/∂inᵢ = ∂L/∂out · ∂f/∂inᵢ`.  Variables 0 and 1 are the primal inputs,
    variable 2 the incoming cotangent. -/
def ewBack2 (f : Expr 2) (i : Fin 2) : Expr 3 :=
  .mul (.var ⟨2, by decide⟩) (sderiv (wk2 f) i.castSucc)

/-- Weaken a ternary spec into the quaternary context its adjoints work in. -/
def wk3 (e : Expr 3) : Expr 4 := rename (Fin.castLE (by decide)) e

/-- The adjoint of a *ternary* elementwise op — a gated feed-forward is the
    standard case — with respect to input `i`.  Variables 0–2 are the primal
    inputs, variable 3 the incoming cotangent. -/
def ewBack3 (f : Expr 3) (i : Fin 3) : Expr 4 :=
  .mul (.var ⟨3, by decide⟩) (sderiv (wk3 f) i.castSucc)

/-- `a + b`, the cotangent accumulator. -/
def addSpec : Expr 2 := .add (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)

/-- Which buffer holds `∂L/∂b`, per forward buffer `b`. -/
abbrev CoT := List (Ref × Ref)

def CoT.get (m : CoT) (r : Ref) : Option Ref :=
  match m.find? (fun p => p.1 == r) with
  | some p => some p.2
  | none   => none

/-- **Record a cotangent contribution.**  A value used more than once — a
    residual stream is the standard case — receives one contribution per use,
    and reverse mode is only correct if they are *summed*.  The first
    contribution binds; every later one emits an add. -/
def CoT.accum (m : CoT) (r d : Ref) (fresh grid : Nat) : List TOp × CoT × Nat :=
  match m.get r with
  | none      => ([], (r, d) :: m, fresh)
  | some prev => ([.ew2 addSpec prev d fresh grid], (r, fresh) :: m, fresh + 1)

/-- **The backward operations of one forward operation.**

    A contraction contributes the weight gradient always and the input
    gradient only where one is wanted, so a model does not compute a gradient
    with respect to its own input.  `smce` emits nothing: it *is* the seed —
    the buffer it already writes is the cotangent of its logits. -/
def TOp.grad (needs : Ref → Bool) (ones : Ref) (batch fresh : Nat) (ct : CoT) :
    TOp → Option (List TOp × CoT × Nat)
  | .mv bk w x out bb inW outW =>
      match ct.get out with
      | none   => none
      | some d =>
          let (addW, ct1, f1) := CoT.accum ct w fresh (fresh + 1) (outW * inW / 32)
          if needs x then
            let (addX, ct2, f2) := CoT.accum ct1 x f1 (f1 + 1) (batch * inW / 32)
            some ([.outer bk d x fresh bb inW outW] ++ addW
                    ++ [.mvT bk w d f1 bb inW outW] ++ addX, ct2, f2)
          else
            some ([.outer bk d x fresh bb inW outW] ++ addW, ct1, f1)
  | .ew1 f a out grid =>
      match ct.get out with
      | none   => none
      | some d =>
          let (addA, ct1, f1) := CoT.accum ct a fresh (fresh + 1) grid
          some ([.ew2 (ewBackOf f) a d fresh grid] ++ addA, ct1, f1)
  | .ew2 f a b out grid =>
      match ct.get out with
      | none   => none
      | some d =>
          let (addA, ct1, f1) := CoT.accum ct a fresh (fresh + 1) grid
          let (addB, ct2, f2) := CoT.accum ct1 b f1 (f1 + 1) grid
          some ([.ew3 (ewBack2 f ⟨0, by decide⟩) a b d fresh grid] ++ addA
                  ++ [.ew3 (ewBack2 f ⟨1, by decide⟩) a b d f1 grid] ++ addB, ct2, f2)
  | .ew3 f a b c out grid =>
      match ct.get out with
      | none   => none
      | some d =>
          let (addA, ct1, f1) := CoT.accum ct a fresh (fresh + 1) grid
          let (addB, ct2, f2) := CoT.accum ct1 b f1 (f1 + 1) grid
          let (addC, ct3, f3) := CoT.accum ct2 c f2 (f2 + 1) grid
          some ([.ew4 (ewBack3 f ⟨0, by decide⟩) a b c d fresh grid] ++ addA
                  ++ [.ew4 (ewBack3 f ⟨1, by decide⟩) a b c d f1 grid] ++ addB
                  ++ [.ew4 (ewBack3 f ⟨2, by decide⟩) a b c d f2 grid] ++ addC,
                ct3, f3)
  | .smce l _ _ out _ => some ([], (l, out) :: ct, fresh)
  | .rowsq x out n rows =>
      match ct.get out with
      | none => none
      | some ds =>
          let (addX, ct1, f1) := CoT.accum ct x fresh (fresh + 1) (rows * n / 32)
          some ([.ziprow x ds fresh (.mul (.add (.reg 1) (.reg 1)) (.reg 2))
                  (.rowOf n 0) .scalar n 0 n rows] ++ addX, ct1, f1)
  | .mvT bk w d out b inW outW =>
      match ct.get out with
      | none => none
      | some g =>
          -- `dW[o][i] = Σₛ d[s][o]·g[s][i]`, `dd[s][o] = Σᵢ W[o][i]·g[s][i]`
          let (addW, ct1, f1) := CoT.accum ct w fresh (fresh + 1) (outW * inW / 32)
          if needs d then
            let (addD, ct2, f2) := CoT.accum ct1 d f1 (f1 + 1) (b * outW / 32)
            some ([.outer bk d g fresh b inW outW] ++ addW
                    ++ [.mv bk w g f1 b inW outW] ++ addD, ct2, f2)
          else
            some ([.outer bk d g fresh b inW outW] ++ addW, ct1, f1)
  | .rowdot i j out mA mB n rows =>
      match ct.get out with
      | none => none
      | some ds =>
          match mA with
          | .rowOf nA kA =>
              let opA := TOp.ziprow j ds fresh mulW mB .scalar nA kA n rows
              let (addA, ct1, f1) := CoT.accum ct i fresh (fresh + 1) (rows * nA / 32)
              if needs j then
                match mB with
                | .rowOf nB kB =>
                    let (addB, ct2, f2) := CoT.accum ct1 j f1 (f1 + 1) (rows * nB / 32)
                    some ([opA] ++ addA
                            ++ [TOp.ziprow i ds f1 mulW mA .scalar nB kB n rows]
                            ++ addB, ct2, f2)
                | _ => none
              else some ([opA] ++ addA, ct1, f1)
          | _ => none
  | .rowmax x out n rows _ =>
      match ct.get out with
      | none => none
      | some ds =>
          -- the gradient goes where the maximum was attained; `geF` is that
          -- indicator as a value, which is why no predicate is needed
          let sel : WFExp := .mul (.reg 3) (.geF (.reg 1) (.reg 2))
          let (addX, ct1, f1) := CoT.accum ct x fresh (fresh + 1) (rows * n / 32)
          some ([.ziprow3 x out ds fresh sel (.rowOf n 0) .scalar .scalar n 0 n rows]
                  ++ addX, ct1, f1)
  | .ziprow a b out f mA mB n off w rows =>
      match ct.get out with
      | none => none
      | some d =>
          match mA, adjW f 1 with
          | .rowOf nA kA, some fa =>
              let opA := TOp.ziprow3 a b d fresh fa mA mB (.rowOf n off) nA kA w rows
              let (addA, ct1, f1) := CoT.accum ct a fresh (fresh + 1) (rows * nA / 32)
              if needs b then
                match adjW f 2 with
                | none => none
                | some fb =>
                    match mB with
                    | .rowOf nB kB =>
                        let (addB, ct2, f2) := CoT.accum ct1 b f1 (f1 + 1) (rows * nB / 32)
                        some ([opA] ++ addA
                                ++ [TOp.ziprow3 a b d f1 fb mA mB
                                      (.rowOf n off) nB kB w rows]
                                ++ addB, ct2, f2)
                    | .scalar =>
                        -- one value per row: the products, then a row sum
                        let (addB, ct2, f2) :=
                          CoT.accum ct1 b (f1 + 1) (f1 + 2) ((rows + 31) / 32)
                        some ([opA] ++ addA
                                ++ [TOp.ziprow3 a b d f1 fb mA mB
                                      (.rowOf n off) w 0 w rows,
                                    TOp.rowdot f1 ones (f1 + 1)
                                      (.rowOf w 0) (.sharedAt 0) w rows]
                                ++ addB, ct2, f2)
                    | .sharedAt 0 =>
                        -- one vector shared by every row: the products, then a
                        -- column sum, which is an outer product against ones
                        let (addB, ct2, f2) :=
                          CoT.accum ct1 b (f1 + 1) (f1 + 2) (w / 32)
                        some ([opA] ++ addA
                                ++ [TOp.ziprow3 a b d f1 fb mA mB
                                      (.rowOf n off) w 0 w rows,
                                    TOp.outer Backend.proven ones f1 (f1 + 1) rows w 1]
                                ++ addB, ct2, f2)
                    | _ => none
              else some ([opA] ++ addA, ct1, f1)
          | _, _ => none
  | _ => none

def gradRev (needs : Ref → Bool) (ones : Ref) (batch : Nat) :
    List TOp → Nat → CoT → Option (List TOp × CoT × Nat)
  | [],         fresh, ct => some ([], ct, fresh)
  | op :: rest, fresh, ct =>
      match op.grad needs ones batch fresh ct with
      | none => none
      | some (bs, ct', f1) =>
          match gradRev needs ones batch rest f1 ct' with
          | none => none
          | some (rs, ct'', f2) => some (bs ++ rs, ct'', f2)

/-- **The backward pass of a forward program, derived.**

    Reverse-mode: the forward tape walked backwards, each operation
    contributing its own adjoint.  Every buffer it allocates is fresh, and the
    activation adjoints come from `sderiv` — so the backward is a function of
    the forward, not a second program written alongside it.

    An operation the reverse pass has no rule for, or one whose output has no
    cotangent, yields `none` rather than an empty contribution — a missing
    gradient is a build failure, not a silently untrained parameter. -/
def Ten.backward (needs : Ref → Bool) (ones : Ref) (batch fresh : Nat)
    (ops : List TOp) : Option (List TOp) :=
  (gradRev needs ones batch ops.reverse fresh []).map Prod.fst

/-- The same, from a cotangent already known — a slice of a tape whose loss is
    computed elsewhere, or a block differentiated on its own. -/
def Ten.backwardFrom (needs : Ref → Bool) (ones : Ref) (batch fresh : Nat)
    (seed : CoT) (ops : List TOp) : Option (List TOp) :=
  (gradRev needs ones batch ops.reverse fresh seed).map Prod.fst

/-- **Where each gradient lands.**  The reverse pass allocates as it goes, so
    which buffer holds `∂L/∂w` is a fact about the derivation, not a convention
    a host can assume. -/
def Ten.backwardCoT (needs : Ref → Bool) (ones : Ref) (batch fresh : Nat)
    (seed : CoT) (ops : List TOp) : Option CoT :=
  (gradRev needs ones batch ops.reverse fresh seed).map (fun r => r.2.1)

/-- `tlet x := e; body` — the tensor binder. -/
syntax "tlet " ident " := " term "; " ppLine term : term

macro_rules
  | `(tlet $x := $e; $b) =>
      `(Ten.letT (Ten.named $(Lean.quote x.getId.toString) $e)
          fun w => let $x := Ten.var w; $b)

-- ---------------------------------------------------------------------------
-- Depth
-- ---------------------------------------------------------------------------

/-!
  A deep model must not cost depth-many theorems.  The three lemmas below are
  what make it cost one: `flat` distributes a `letT` into an append, `forget`
  distributes over the append, and `lowerNet` distributes over it too — so the
  stages of an `n+1`-deep stack are the stages of the `n`-deep stack followed
  by one layer's, proven **once, for every `n`**.
-/

/-- Stack `n` copies of a layer, binding each layer's input so the residual
    stream is one buffer rather than a re-substituted subtree. -/
def Ten.stack {v : Nat → Nat → Type} {r c : Nat}
    (layer : Nat → Ten v r c → Ten v r c) : Nat → Ten v r c → Ten v r c
  | 0,     x => x
  | n + 1, x => .letT (Ten.stack layer n x) (fun b => layer n (.var b))

theorem forget_append (a b : List (Node × Backend)) :
    forget (a ++ b) = forget a ++ forget b := List.map_append ..

/-- Lowering distributes over concatenation: a malformed node anywhere still
    fails the whole program, and otherwise the stages concatenate. -/
theorem lowerNet_append (batch : Nat) : ∀ (a b : Net),
    lowerNet batch (a ++ b)
      = (lowerNet batch a).bind (fun sa => (lowerNet batch b).map (sa ++ ·)) := by
  intro a
  induction a with
  | nil => intro b; cases h : lowerNet batch b <;> simp [lowerNet, h]
  | cons n ns ih =>
      intro b
      simp only [List.cons_append, lowerNet, ih b]
      cases n.stage? batch with
      | none => rfl
      | some o =>
          cases o with
          | none => rfl
          | some S =>
              cases lowerNet batch ns <;> cases lowerNet batch b <;> rfl

/-- A closed program: quantified over the variable representation, so it cannot
    inspect the buffer numbers it will be handed. -/
def TenProg (r c : Nat) : Type 1 := (v : Nat → Nat → Type) → Ten v r c

/-- The graph a program flattens to, with computed buffers allocated from
    `base` upward.  `base` is the first buffer past the declared inputs. -/
def TenProg.graph (base : Ref) (p : TenProg r c) : List (Node × Backend) :=
  Ten.graphOf (p RefV) base

/-- The stages a program lowers to. -/
noncomputable def TenProg.stages (batch : Nat) (base : Ref) (p : TenProg r c) :
    Option (List XStage) :=
  lowerNet batch (forget (p.graph base))

/-- **A schedule that erases to the model lowers to the model's stages.**

    Unlike `Node.sig`, `forget` keeps the elementwise `Expr`, so this compares
    the arithmetic as well as the wiring — a schedule that substituted a
    different activation does not satisfy the hypothesis. -/
theorem TenProg.schedule_agrees (batch : Nat) (base : Ref) (sched model : TenProg r c)
    (h : forget (sched.graph base) = forget (model.graph base)) :
    sched.stages batch base = model.stages batch base := by
  simp only [TenProg.stages, h]

end AlgorithmLib.ML
