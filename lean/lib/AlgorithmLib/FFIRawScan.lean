import AlgorithmLib.FFIRaw

/-!
# `FFIRaw` says what `Ffi.sig` says, and the build checks it

`FFIRaw` is mechanical — one wrapper per entry point, its arity taken from that
entry point's signature. Mechanical is only worth anything if it stays that
way, and nothing about a written-out file stops it drifting from the table it
came from: an appended constructor leaves a hole, an edited signature leaves a
wrapper that still type-checks and still calls with the wrong number of
arguments.

So this reads both and compares them. For every constructor of `IR.Ffi` it asks
whether `Sur.Raw` has a wrapper of that name, whether the wrapper takes exactly
as many binders as `Ffi.sig` lists parameters, and whether it lands in `M R` or
`M Unit` according to whether the signature has a result.

It also checks something `FFIRaw` did not motivate but nothing else states:
that every constructor is in `Ffi.all`, and that `all` lists none of them
twice. A callee's id is its position in that list, so a constructor missing
from it takes the id of one past the end and a duplicated one shadows its
first occurrence — either way the artifact calls the wrong symbol, and neither
shows up as a type error.

Any disagreement fails the build. This is the shape `ShipScan` and the trust
scanners use, and it is cheap enough — 85 declaration types, no elaboration —
to run on every build rather than on request.
-/

open Lean Meta

namespace AlgorithmLib.FFIRawScan

open AlgorithmLib.IR

/-- How many leading binders a wrapper takes, and what is left after them.

Counting the telescope rather than matching the whole type is what keeps this
independent of how a wrapper was written: `(a b : R)` and `(a : R) (b : R)`
give the same answer. -/
partial def binderCount : Expr → Nat × Expr
  | .forallE _ _ body _ => let (n, r) := binderCount body; (n + 1, r)
  | e => (0, e)

/-- The length of a `List` term, reduced far enough to see its spine. -/
partial def listLength (e : Expr) : MetaM Nat := do
  match (← whnf e).getAppFnArgs with
  | (``List.cons, #[_, _, tl]) => return (← listLength tl) + 1
  | (``List.nil, _) => return 0
  | _ => throwError "FFIRawScan: not a list literal: {e}"

/-- Whether an `Option` term is `some`. -/
def isSome (e : Expr) : MetaM Bool := do
  match (← whnf e).getAppFnArgs with
  | (``Option.some, _) => return true
  | (``Option.none, _) => return false
  | _ => throwError "FFIRawScan: not an option literal: {e}"

/-- Whether a wrapper's result type is `M R` (rather than `M Unit`). -/
def landsInR (e : Expr) : MetaM (Option Bool) := do
  let mR ← mkAppM ``HProg.Sur.M #[mkConst ``HProg.R]
  let mU ← mkAppM ``HProg.Sur.M #[mkConst ``Unit]
  if ← isDefEq e mR then return some true
  if ← isDefEq e mU then return some false
  return none

def check : MetaM Unit := do
  let env ← getEnv
  let ind ← getConstInfoInduct ``Ffi
  let mut problems : Array String := #[]

  -- `all` is what fixes every callee id, so it must name each constructor once.
  let allLen := Ffi.all.length
  if allLen != ind.ctors.length then
    problems := problems.push
      s!"`Ffi.all` lists {allLen} of {ind.ctors.length} constructors"
  if Ffi.all.eraseDups.length != allLen then
    problems := problems.push s!"`Ffi.all` lists some constructor twice"

  for ctor in ind.ctors do
    let short := ctor.getString!
    let wrapper := Name.str `AlgorithmLib.HProg.Sur.Raw short
    let some info := env.find? wrapper
      | problems := problems.push s!"{short}: no wrapper `Sur.Raw.{short}`"
        continue
    let (arity, res) := binderCount info.type
    let expected ← listLength (← mkAppM ``Ffi.params #[mkConst ctor])
    if arity != expected then
      problems := problems.push
        s!"{short}: wrapper takes {arity} argument(s), `sig` lists {expected}"
    let sigHasResult ← isSome (← mkAppM ``Ffi.result #[mkConst ctor])
    match ← landsInR res with
    | none =>
        problems := problems.push
          s!"{short}: result is neither `M R` nor `M Unit`"
    | some inR =>
        if inR != sigHasResult then
          problems := problems.push <|
            if sigHasResult then s!"{short}: `sig` has a result, wrapper is `M Unit`"
            else s!"{short}: `sig` has no result, wrapper is `M R`"

  if problems.isEmpty then
    logInfo s!"[FFIRawScan] {ind.ctors.length} entry points: \
every wrapper matches its signature, and `all` names each once"
  else
    throwError "FFIRawScan: {problems.size} disagreement(s)\n  {
      String.intercalate "\n  " problems.toList}"

end AlgorithmLib.FFIRawScan

#eval AlgorithmLib.FFIRawScan.check
