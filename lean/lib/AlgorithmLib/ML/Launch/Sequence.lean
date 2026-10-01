module
public import AlgorithmLib.ML.Launch.Pipeline
meta import AlgorithmLib.ML.Launch.Pipeline
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # A launch sequence, as data

  `Launch/Pipeline.lean` proves what *one* launch leaves in memory.  Composing two of
  them was possible only by writing a bespoke theorem per pair, with the
  composite spelled out in its own conclusion — so there was no object a
  varying implementation could be checked *against*.

  The obstruction was representational.  A launch sequence existed only as the
  order in which a generator happened to emit calls, and an emission order is
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

    The three cases are exactly the three theorems `Launch/Pipeline.lean` already had:
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

end AlgorithmLib.ML
