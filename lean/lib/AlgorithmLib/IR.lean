import AlgorithmLib.ClifData
import AlgorithmLib.Core
import AlgorithmLib.Bytes

namespace AlgorithmLib

namespace IR

/-- A function that does nothing, at a given index: one block taking the shared
    memory pointer and returning. -/
def noopAt (funcIdx : Nat) : FuncData :=
  { index := funcIdx, callees := [],
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

/-- One callee, as the compiler needs it: what to call, and what it takes.

    Only `callee` is written out. The signature is here because the checker
    reads a call's arity and types from it before anything is emitted; an
    artifact carries none, since base's table gives an import's and a local
    callee's own entry block gives its. -/
structure CalleeDecl where
  callee : Callee
  params : List ClifTy
  result : Option ClifTy
  deriving BEq, Lean.ToExpr

/-- The callee table a body is checked and compiled against.

    **Position is the reference.** A call names `callees[i]`, so the order is
    the numbering: there are no ids to allocate, none to collide, and no way to
    name a callee the table does not hold. What ships is this list with the
    signatures dropped. -/
abbrev FnEnv := List CalleeDecl

/-- What `fn` calls and what it takes, or `none` when the table is shorter. -/
def FnEnv.at? (env : FnEnv) (fn : Nat) : Option CalleeDecl := env[fn]?

/-- The table with one more callee, and the reference naming it: its position.

    A callee already in the table at the same signature keeps its place, so a
    body that calls one twice declares it once. -/
def FnEnv.use (e : FnEnv) (callee : Callee) (params : List ClifTy)
    (result : Option ClifTy) : FnRef × FnEnv :=
  let d : CalleeDecl := { callee, params, result }
  match e.idxOf? d with
  | some i => (⟨i⟩, e)
  | none   => (⟨e.length⟩, e ++ [d])

/-- Call another function of this same program, by its `u0:N` index. -/
def FnEnv.useLocal (e : FnEnv) (index : Nat) (params : List ClifTy)
    (result : Option ClifTy) : FnRef × FnEnv :=
  e.use (.local index) params result

/-- What the emitted function carries: the callees, without the signatures. -/
def FnEnv.callees (e : FnEnv) : List Callee := e.map (·.callee)

end IR

end AlgorithmLib
