module
public import AlgorithmLib.ML.Launch.Train
meta import AlgorithmLib.ML.Launch.Train
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! # None -/

namespace AlgorithmLib.ML

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

end AlgorithmLib.ML
