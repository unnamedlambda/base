import AlgorithmLib.Gen

/-!
  # Every body an artifact carries was compiled through the checked door

  `HProg.compileFn` takes `wf env params c = true` as an auto-param, so a body
  compiled through it is well-formed or the call site is an error.
  `HProg.compileBody` takes no such obligation — `CompileSound` relates the two
  runs of an arbitrary body, so the compiler it names cannot demand one.

  This asks which door each shipped body went through, walking from the
  generator's `main`. A module that emits artifacts without reaching a `Setup`
  fails too: a scan that found nothing to look at has not looked.
-/

open Lean

namespace ShipScan

/-- The unchecked compiler, and the wrapper that carries the obligation. -/
def UNCHECKED : Name := `AlgorithmLib.HProg.compileBody
def CHECKED : Name := `AlgorithmLib.HProg.compileFn

/-- Constants reachable from a declaration's value, not expanding `compileFn`:
    its body names `compileBody`, and that is the one use under an obligation. -/
partial def closureOf (env : Environment) (seen : Std.HashSet Name) (n : Name) :
    Std.HashSet Name :=
  if seen.contains n || n == CHECKED then seen.insert n
  else
    let seen := seen.insert n
    match env.find? n with
    | none    => seen
    | some ci =>
        (match ci.value? with
         | some v => v.getUsedConstants
         | none   => #[]).foldl (fun acc c => closureOf env acc c) seen

/-- The declarations leading from `root` to `target`, nearest first, or `#[]`
    when the walk does not reach it. Breadth-first, so the path is a shortest
    one and names the declaration that should have used `compileFn`. -/
partial def pathTo (env : Environment) (root target : Name) : Array Name :=
  go (Std.HashSet.emptyWithCapacity.insert root) #[[root]]
where
  uses (n : Name) : Array Name :=
    if n == CHECKED then #[]
    else match env.find? n with
         | some ci => (match ci.value? with
                       | some v => v.getUsedConstants
                       | none   => #[])
         | none    => #[]
  go (seen : Std.HashSet Name) (front : Array (List Name)) : Array Name :=
    if front.isEmpty then #[] else
    let step := front.foldl (init := (seen, #[], none)) fun (sn, nx, hit) path =>
      match hit, path with
      | some _, _ => (sn, nx, hit)
      | none, [] => (sn, nx, none)
      | none, n :: rest =>
          (uses n).foldl (init := (sn, nx, none)) fun (sn, nx, hit) c =>
            match hit with
            | some _ => (sn, nx, hit)
            | none =>
                if c == target then (sn, nx, some (c :: n :: rest))
                else if sn.contains c then (sn, nx, none)
                else (sn.insert c, nx.push (c :: n :: rest), none)
    match step.2.2 with
    | some p => p.reverse.toArray
    | none   => go step.1 step.2.1

/-- A declaration producing a `Setup`, whatever it takes first. -/
private def yieldsSetup : Expr → Bool
  | .forallE _ _ b _ => yieldsSetup b
  | e                => e.isConstOf ``AlgorithmLib.Setup

/-- Fails the build on an artifact holding a body that was not compiled through
    `compileFn`. -/
def check (label : String) (root : Name := `main) (gatedElsewhere : Option String := none) :
    CoreM Unit := do
  let env ← getEnv
  if (env.find? root).isNone then
    throwError s!"SHIP SCAN [{label}]: no such entry point `{root}`"
  let reached := closureOf env {} root
  let setups := reached.toList.filter fun n =>
    match env.find? n with
    | some ci => yieldsSetup ci.type
    | none    => false
  if setups.isEmpty then
    throwError s!"SHIP SCAN [{label}] FAILED: `{root}` reaches no Setup, so nothing was checked"
  if reached.contains UNCHECKED then
    match gatedElsewhere with
    | some why =>
        IO.println s!"[{label}] compiles through {UNCHECKED}; gated by {why}"
        return
    | none => pure ()
    IO.println s!"UNCHECKED BODY — {label} reaches {UNCHECKED} by:"
    for n in pathTo env root UNCHECKED do IO.println s!"    {n}"
    throwError s!"SHIP SCAN [{label}] FAILED: a body it ships was compiled by \
      {UNCHECKED} rather than {CHECKED}"
  IO.println s!"[{label}] every body its {setups.length} artifact(s) carry was compiled through compileFn"

end ShipScan
