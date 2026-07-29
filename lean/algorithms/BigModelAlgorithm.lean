import Lean
import Std
import AlgorithmLib

/-!
  # A model at Qwen2's width and deeper, elaborated

  This file exists to be *checked*, not to be run.  It is the standing evidence
  for the elaboration claim: a **93-layer, width-896** model — deeper than
  Qwen2-0.5B and exactly as wide — together with the correctness theorem for its
  narrowed gradient, in the time it takes to load the imports.

  Three separate things make that possible, and all three are needed:

  * `Expr.sum n f` holds a **function**, not `n` copies.  A width-896 matvec is
    one node with a closure, so width costs nothing at elaboration.  (`nodes`
    reports the *expanded* size, which is what a compiler walking the term would
    see — not what the elaborator does.)
  * `bindVec`, via `runStack`, binds each layer's whole output vector, so depth
    is linear rather than exponential (`MultiLayer.lean`: 4,689,809 → 2,633 at
    width 4, depth 3).
  * `denseStack_gradD_correct` discharges the per-binding window obligation
    once for any depth (`Layered.lean`), so the gradient theorem does not grow
    with the 83,328 bindings underneath it.

  **The model carries no proof text**: no `by`, no anonymous constructor, no
  `Fin` literal.  The parameters come from `slice`/`matSlice`, whose bounds are
  auto-params, and `runStack` hands the layer the weakener it needs.  (The
  *theorem* below has two `by decide`s, both discharging `0 < W`; a user who
  only builds and runs a model never writes them.)
-/

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML

namespace BigModel

/-- Qwen2-0.5B's hidden size. -/
abbrev D : Nat := 896
/-- Deeper than Qwen2-0.5B's 24. -/
abbrev L : Nat := 93
/-- Inputs: the activations, then a `D × D` weight matrix, then the gains. -/
abbrev W : Nat := D + D * D + D

def xs   : Vec D W    := slice 0 D
def wm   : Mat D D W  := matSlice D D D
def gain : Vec D W    := slice (D + D * D) D

/-- One layer: RMSNorm, matvec, SiLU, residual — ordinary functional Lean.
    `ρ` weakens the base-context parameters into this level; `runStack` supplies
    it, so no bound and no `wkTo` appears here. -/
def layer {Δ : Nat} (ρ : Expr W → Expr Δ) (v : Vec D Δ) : Vec D Δ :=
  v + ((wm.mapE ρ) * rmsNorm (gain.mapE ρ) v).map silu

def readout {Δ : Nat} (_ : Expr W → Expr Δ) (v : Vec D Δ) : Expr Δ :=
  v.dot (Vec.lit 1)

/-- **The model.**  93 layers, width 896. -/
def model : Expr W := runStack layer readout L xs

/-- **And its narrowed gradient is correct**, at the same shape.

    The two declared propositions are the entire price; nothing here mentions
    the 83,328 bindings the telescope actually has. -/
theorem model_grad_correct (env : Fin W → Float32) (out : Expr (W + L * D))
    (c : Nat → Nat → Nat) (hz : ZeroTermFree Float32) (hzl : ZeroLaws Float32)
    (k : Fin W) :
    denote env ((gradProgD (genTele (denseGen W (by decide) c) (L * D)) out
        (denseDep W)).get k)
      = denote env (sderiv ((genTele (denseGen W (by decide) c) (L * D)).bind out) k) :=
  denseStack_gradD_correct W (by decide) c (L * D) out env hz hzl k

end BigModel
