module
public import AlgorithmLib.ML.Launch.Sequence
meta import AlgorithmLib.ML.Launch.Sequence
public import AlgorithmLib.ML.Kernel.Rewrite
meta import AlgorithmLib.ML.Kernel.Rewrite
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! # None -/

namespace AlgorithmLib.ML

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

end AlgorithmLib.ML
