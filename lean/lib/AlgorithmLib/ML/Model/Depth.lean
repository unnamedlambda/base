module
public import AlgorithmLib.ML.Model.GraphBuild
meta import AlgorithmLib.ML.Model.GraphBuild
public import AlgorithmLib.ML.Model.Ten
meta import AlgorithmLib.ML.Model.Ten
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! # None -/

namespace AlgorithmLib.ML

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
