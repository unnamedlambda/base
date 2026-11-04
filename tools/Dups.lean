/-!
  One definition per concept, checked: no two theorems of the library state the
  same proposition and no two definitions have the same body, up to the names
  of their binders.

  Not a module of any package. `tools/check.sh` prepends an import of every
  module of the library and runs it with `lake env lean`; a group of two or
  more names fails it, naming the group and the modules each one lives in.
-/

open Lean Meta

namespace Dups

/-- Declarations Lean or a deriving handler makes, and the equation and
    matcher lemmas it generates: their statements repeat by construction. -/
def generated (env : Environment) (n : Name) : Bool :=
  n.isInternal || n.isImplementationDetail || isAuxRecursor env n || isNoConfusion env n ||
  ["match_", "proof_", "eq_", "_sunfold", ".inj", "sizeOf", "_cstage", "noConfusion", "_spec_"].any
    fun s => (n.toString.splitOn s).length > 1

/-- Groups of names whose keys are equal up to binder names. -/
def groups (env : Environment) (xs : Array (Name × Expr)) : Array (Array Name) := Id.run do
  let mut tbl : Std.HashMap UInt64 (Array (Name × Expr)) := {}
  for (n, e) in xs do
    tbl := tbl.insert e.hash ((tbl.getD e.hash #[]).push (n, e))
  let mut out := #[]
  for (_, ys) in tbl.toList do
    if ys.size < 2 then continue
    let mut used := Array.replicate ys.size false
    for i in [0:ys.size] do
      if used[i]! then continue
      let mut grp := #[ys[i]!.1]
      for j in [i+1:ys.size] do
        if !used[j]! && ys[i]!.2.eqv ys[j]!.2 then
          grp := grp.push ys[j]!.1
          used := used.set! j true
      if grp.size > 1 then out := out.push grp
  let _ := env
  return out

#eval show MetaM Unit from do
  let env ← getEnv
  let mods := env.header.moduleNames
  let modOf (n : Name) : Name :=
    (env.getModuleIdxFor? n).map (fun i => mods[i.toNat]!) |>.getD .anonymous
  let mut thms : Array (Name × Expr) := #[]
  let mut defs : Array (Name × Expr) := #[]
  for (n, ci) in env.constants.map₁.toList do
    unless (`AlgorithmLib).isPrefixOf (modOf n) do continue
    if generated env n then continue
    match ci with
    | .thmInfo t => if t.type.approxDepth ≥ 3 then thms := thms.push (n, t.type)
    | .defnInfo d =>
        if d.value.approxDepth ≥ 4 then defs := defs.push (n, mkApp d.type d.value)
    | _ => pure ()
  let found := (groups env thms).map (("theorems", ·)) ++ (groups env defs).map (("definitions", ·))
  for (kind, g) in found do
    IO.println s!"{kind} stated twice: {", ".intercalate (g.toList.map fun n => s!"{n} [{modOf n}]")}"
  unless found.isEmpty do
    throwError "{found.size} groups of duplicates; keep one of each where every user can import it"
  IO.println s!"{thms.size} theorems and {defs.size} definitions, each stated once"

end Dups
