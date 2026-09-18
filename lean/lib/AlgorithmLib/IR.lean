import AlgorithmLib.ClifData
import AlgorithmLib.Core
import AlgorithmLib.Bytes

namespace AlgorithmLib

namespace IR

/-- A function that does nothing, at a given index: one block taking the shared
    memory pointer and returning. -/
def noopAt (funcIdx : Nat) : FuncData :=
  { index := funcIdx,
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

/-- What a callee takes and answers.

    An `Ffi`'s is a total function of the constructor, so it needs no table.
    This exists for the other arm: a `local` call names a function of the same
    program, whose signature is its entry block's, and a body is compiled
    before the function it calls necessarily exists as a term. So the builder
    states it at the call, and the checker reads it from here.

    Never written out. The engine derives an import's from its own table and a
    local's from that function's entry block --- it has both, so shipping
    either would be a second source for one fact. -/
structure CalleeSig where
  params : List ClifTy
  result : Option ClifTy
  deriving BEq, Lean.ToExpr

/-- The signatures of this program's own functions, by `u0:N`.

    Build-time only, and it holds nothing about imports: a call names its
    callee, so there is no table of callees any more and nothing to intern,
    allocate or renumber. -/
abbrev FnEnv := List (Nat × CalleeSig)

/-- What a callee takes and answers: from the constructor for an import, from
    this table for one of the program's own. -/
def FnEnv.sigOf (e : FnEnv) : Callee → Option CalleeSig
  | .ffi f   => some { params := f.params, result := f.result }
  | .local k => (e.find? (·.1 == k)).map (·.2)

/-- The table with one local function's signature recorded. A repeat at the
    same signature changes nothing; a repeat at a different one is what
    `sigOf` will then disagree with, and the checker refuses the call. -/
def FnEnv.withLocal (e : FnEnv) (index : Nat) (params : List ClifTy)
    (result : Option ClifTy) : FnEnv :=
  let d : Nat × CalleeSig := (index, { params, result })
  if e.contains d then e else e ++ [d]

end IR

end AlgorithmLib
