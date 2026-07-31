import AlgorithmLib.ML.Pipeline

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

/-- **A step whose effect is assumed rather than derived.**

    `frame` is not optional and is not an assumption: it is proven when the step
    is constructed, and it is what stops a declared step being a blank cheque.
    Without it, "assume this does whatever it does" would let the step clobber
    buffers the rest of the pipeline reasons about, and every downstream frame
    argument would silently collapse.  What is assumed is confined to *what
    lands in `out`*; *where it can land* stays proven. -/
structure DeclaredStep where
  /-- The primitive, e.g. `cl_cublas_sgemv`. -/
  name  : String
  /-- Why it is not proven — carried in the value so it reaches any report. -/
  why   : String
  out   : Buf
  step  : (Buf → Nat → Float32) → Buf → Nat → Float32
  frame : ∀ (m : Buf → Nat → Float32) (b : Buf), b ≠ out → step m b = m b

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

/-- The buffer a step writes, proven or declared alike. -/
def PStep.out : PStep → Buf
  | .proven S   => S.out
  | .declared d => d.out

/-- Off that buffer, a step changes nothing.  Both cases already carry the
    fact; this is what lets them be used interchangeably in a fold. -/
theorem PStep.denote_otherBuf (s : PStep) (m : Buf → Nat → Float32) (b : Buf)
    (hb : b ≠ s.out) : s.denote m b = m b := by
  cases s with
  | proven S   => funext a; exact StageSpec.step_otherBuf S m b a hb
  | declared d => exact d.frame m b hb

/-- **A buffer no later step writes still holds what it held.**

    The workhorse for reading a value *out* of a plan.  A plan's denotation is
    a left fold, so asking what is at one buffer at the end means knowing that
    the steps after the one that wrote it left it alone — which is exactly what
    each step's `frame` field says, and this lifts it to the sequence. -/
theorem denote_frame_list : ∀ (L : List PStep) (m : Buf → Nat → Float32) (b : Buf),
    (∀ s ∈ L, b ≠ s.out) → (L.foldl (fun mm s => s.denote mm) m) b = m b := by
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
def outsOf (L : List PStep) : List Buf := L.map PStep.out

/-- `denote_frame_list`, stated against `outsOf`. -/
theorem denote_frame_outs (L : List PStep) (m : Buf → Nat → Float32) (b : Buf)
    (h : ∀ o ∈ outsOf L, b ≠ o) :
    (L.foldl (fun mm s => s.denote mm) m) b = m b :=
  denote_frame_list L m b (fun s hs => h s.out (List.mem_map.mpr ⟨s, hs, rfl⟩))

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

/-- A plan of only proven steps is a pipeline, and needs no realisation at all —
    so nothing is weakened for programs that do not reach for a declared step. -/
theorem Plan.declaredCount_zero_of_allProven (P : Plan)
    (h : ∀ s ∈ P.steps, ∃ S, s = PStep.proven S) : P.declaredCount = 0 := by
  simp only [Plan.declaredCount, List.length_eq_zero_iff, List.filter_eq_nil_iff]
  intro s hs
  obtain ⟨S, hS⟩ := h s hs
  subst hS
  simp

end AlgorithmLib.ML
