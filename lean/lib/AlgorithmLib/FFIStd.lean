import AlgorithmLib.FFI

/-!
# The standard callee table

`FFI.lean` says who the runtime exports; this module turns that into the tables
a body is checked and compiled against, and splices them as literals.

`stdEnv` is spliced rather than computed: a table left as a function of
`Ffi.all` makes every `decide (wf ..)` walk eighty entries and re-derive each
signature before it can look up one callee.

A function declares the whole table it was compiled against, not the part it
calls, so a generator that wants a small declaration list names the bundles it
uses and passes that table at every site — the builder and the compiler both,
since they take it separately.
-/

namespace AlgorithmLib.IR.FFI

open Lean Meta Elab Term in
unsafe def evalFnEnvUnsafe (e : Expr) : TermElabM FnEnv :=
  evalExpr FnEnv (mkConst ``FnEnv) e

instance : Inhabited (Lean.Elab.TermElabM FnEnv) := ⟨pure default⟩

@[implemented_by evalFnEnvUnsafe]
opaque evalFnEnv (e : Lean.Expr) : Lean.Elab.TermElabM FnEnv

open Lean Elab Term in
/-- Splice a table as a first-order literal, so `decide` is left with plain
    lists to walk rather than a derivation to run. -/
elab "env% " bs:term : term => do
  let e ← elabTermEnsuringType (← `(envFromFfi (Ffi.all.filter (fun f => ($bs : List Bundle).contains f.bundle))))
    (mkConst ``FnEnv)
  synthesizeSyntheticMVarsNoPostponing
  return toExpr (← evalFnEnv (← instantiateMVars e))

open Lean Elab Term in
/-- The whole table, spliced. -/
elab "stdEnv%" : term => do
  let e ← elabTermEnsuringType (← `(envFromFfi Ffi.all)) (mkConst ``FnEnv)
  synthesizeSyntheticMVarsNoPostponing
  return toExpr (← evalFnEnv (← instantiateMVars e))

/-- The callee table every body is checked and compiled against. -/
def stdEnv : FnEnv := stdEnv%

end AlgorithmLib.IR.FFI
