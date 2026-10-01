module
public import AlgorithmLib.Core.Compiler
meta import AlgorithmLib.Core.Compiler
public import AlgorithmLib.ML.Ptx.Flat
meta import AlgorithmLib.ML.Ptx.Flat
public import AlgorithmLib.ML.Ptx.Compile
meta import AlgorithmLib.ML.Ptx.Compile
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# The kernel compilers, each with its theorem

The three translations below the model, as `Compiler` values. Each is the
existing function and the existing theorem, unchanged; what this adds is that
they can now be named, listed and chained as one kind of thing.

* `compileW` — a scalar `Expr` to warp statements and the value they leave,
  refining the expression's denotation (`compileW_sound`).
* `emitEW` — a warp statement to structured PTX, refining its elaboration
  (`emitEW_sound`).
* `flatEW` — a warp statement to flat PTX at an offset of a program, refining
  the same elaboration by a bounded run of the program counter (`flatEW_sound`).

`exprToPtx` chains the first two; its door is the second's door read on the
first's output, checked where the chain is used.
-/

namespace AlgorithmLib.ML

open AlgorithmLib

/-- `Expr` to warp statements, for the operand valuation `ve` and first free
    register `c`. -/
def compileWCompiler {Γ : Nat} (ve : Fin Γ → WFExp) (c : Nat) :
    Compiler (Expr Γ) (EWStmt × WFExp) where
  compile := compileW ve c
  Pre _ := ∀ i, (ve i).regsIn 0 c
  Refines e r := ∀ (st : WSt) (l : Lane),
    r.2.eval (runW r.1 st) l = denote (fun i => (ve i).eval st l) e
  sound e h := compileW_sound e ve c h

/-- A warp statement to structured PTX, for block `cta`, with `lr` the loop
    register and `n` the register count. -/
def emitEWCompiler (cta lr n : Nat) (h2 : 2 ≤ n) (hlr : lr < n) : Compiler EWStmt (List SI) where
  compile := emitEW lr n
  Pre s := s.ExpFree ∧ s.IdxBelow n
  Refines s code := ∀ (i : Nat) (m : MState), MInv cta i lr m →
    (SI.stepL cta code m).toWSt = (s.elabAt cta i m.ir m.imem).run m.toWSt
  sound s h i m hm := emitEW_sound cta s h.1 lr n i m h2 hlr h.2 hm

/-- A warp statement to flat PTX placed at `base`. -/
def flatEWCompiler (cta lr n base : Nat) (h2 : 2 ≤ n) (hlr : lr < n) :
    Compiler EWStmt (List FI) where
  compile := flatEW lr n base
  Pre s := s.ExpFree ∧ s.IdxBelow n ∧ s.Flat
  Refines s code := ∀ (i : Nat) (P : List FI) (m : MState), MInv cta i lr m →
    (∀ j, j < flenEW lr n s → P[base + j]? = code[j]?) →
    ∃ k m', steps cta P k (base, m) = some (base + flenEW lr n s, m')
      ∧ m'.toWSt = (s.elabAt cta i m.ir m.imem).run m.toWSt
      ∧ (∀ x, x < n → m'.ir x = m.ir x) ∧ m'.imem = m.imem
  sound s h i P m hm hP := flatEW_sound cta s h.1 lr n base i P m h2 hlr h.2.1 h.2.2 hm hP

/-- An expression all the way to structured PTX. -/
def exprToPtx {Γ : Nat} (ve : Fin Γ → WFExp) (c cta lr n : Nat) (h2 : 2 ≤ n) (hlr : lr < n) :
    Compiler (Expr Γ) (List SI × WFExp) :=
  (compileWCompiler ve c).seq (emitEWCompiler cta lr n h2 hlr).onFst

end AlgorithmLib.ML
