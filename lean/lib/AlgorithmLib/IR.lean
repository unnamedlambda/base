import AlgorithmLib.ClifData
import AlgorithmLib.Core
import AlgorithmLib.Bytes

namespace AlgorithmLib

namespace IR

/-- A function that does nothing, at a given index: one block taking the shared
    memory pointer and returning. -/
def noopAt (funcIdx : Nat) : FuncData :=
  { index := funcIdx, sigs := [], fns := [],
    blocks := [{ ref := { id := 0 },
                 params := [({ id := 0 }, ClifTy.i64)],
                 insts := [Inst.ret none] }] }

/-- The standard noop function u0:0 -/
def noopFunction : FuncData := noopAt 0

/-- The zero a `fconst` names, at each float width. Written out because `0.0`
    elaborates to whichever type the surrounding term forces, and these are the
    widths the emitters take. -/
def f32Zero : Float := 0.0
def f64Zero : Float := 0.0

/-- The callee table a body is checked and compiled against — exactly the
    `sigs` and `fns` the emitted function will declare. -/
structure FnEnv where
  sigs : List SigDecl
  fns  : List FnDecl
  deriving Inhabited, Lean.ToExpr

/-- The signature `fn` resolves to, or `none` when nothing declares it. -/
def FnEnv.sigOf (env : FnEnv) (fn : Nat) : Option SigDecl := do
  let d ← env.fns.find? (·.ref.id == fn)
  env.sigs.find? (·.ref.id == d.sig.id)

/-- The table with one more declaration, and the reference naming it.

    Ids go past the largest in use, not past the count: a table is often a
    selection from a larger one, in which case its ids are sparse and numbering
    from the count would hand a new declaration an id an existing one already
    holds. `sigOf` resolves by id and takes the first match, so that collision
    would not be an error — the call would quietly land on the wrong
    signature. -/
def FnEnv.declare (e : FnEnv) (callee : Callee) (params : List ClifTy)
    (result : Option ClifTy) : FnRef × FnEnv :=
  let sigId := e.sigs.foldl (fun m s => max m (s.ref.id + 1)) 0
  let fnId := e.fns.foldl (fun m d => max m (d.ref.id + 1)) 0
  (⟨fnId⟩,
   { sigs := e.sigs ++ [{ ref := ⟨sigId⟩, params, result }],
     fns := e.fns ++ [{ ref := ⟨fnId⟩, callee := callee, sig := ⟨sigId⟩ }] })

/-- Declare a call to another function of this same program, by `u0:N` index. -/
def FnEnv.declareLocal (e : FnEnv) (index : Nat) (params : List ClifTy)
    (result : Option ClifTy) : FnRef × FnEnv :=
  e.declare (.local index) params result

/-- Declare a symbol the JIT resolves within this program's own module — what a
    generator whose functions call each other by name needs, and the only kind
    of declaration that is not already in `Ffi`. -/
def FnEnv.declareColocated (e : FnEnv) (name : String) (params : List ClifTy)
    (result : Option ClifTy) : FnRef × FnEnv :=
  e.declare (.import name) params result

/-- Several colocated declarations of one shape, in order. -/
def FnEnv.declareColocatedAll (e : FnEnv) (names : List String) (params : List ClifTy)
    (result : Option ClifTy) : List FnRef × FnEnv :=
  names.foldl (fun (refs, e) n =>
    let (r, e) := e.declareColocated n params result
    (refs ++ [r], e)) ([], e)

end IR

end AlgorithmLib
