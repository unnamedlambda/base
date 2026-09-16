import AlgorithmLib.Gen

/-!
  # Every body an artifact carries was compiled through the checked door

  `Prog.compileProg` runs `wf` on the body it emitted and returns an error
  rather than a function when it fails, so a body compiled through it is
  well-formed or the generator refuses to write the artifact.
  `HProg.compileBody` takes no such obligation — `CompileSound` relates the two
  runs of an arbitrary body, so the compiler it names cannot demand one.

  This asks which door each shipped body went through, walking from the
  generator's `main`. A module that emits artifacts without reaching a `Artifact`
  fails too: a scan that found nothing to look at has not looked.
-/

open Lean

namespace ShipScan

/-- The unchecked compiler. -/
def UNCHECKED : Name := `AlgorithmLib.HProg.compileBody
/-- The door that carries the obligation: it runs `wf` on the body it emitted
    and refuses one that fails. -/
def CHECKED : List Name :=
  [`AlgorithmLib.Prog.compileProg, `AlgorithmLib.Prog.compileProgStatus]

/-- Constants reachable from a declaration's value, not expanding
    `compileProg`: its body names `compileBody`, and that is the one use under
    an obligation. -/
partial def closureOf (env : Environment) (seen : Std.HashSet Name) (n : Name) :
    Std.HashSet Name :=
  if seen.contains n || CHECKED.contains n then seen.insert n
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
    one and names the declaration that should have used `compileProg`. -/
partial def pathTo (env : Environment) (root target : Name) : Array Name :=
  go (Std.HashSet.emptyWithCapacity.insert root) #[[root]]
where
  uses (n : Name) : Array Name :=
    if CHECKED.contains n then #[]
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

/-- A declaration producing an `Artifact`, whatever it takes first --- and
    through an `Except`, because a body checked while it is emitted yields one
    only if it passed. -/
private partial def yieldsArtifact : Expr → Bool
  | .forallE _ _ b _ => yieldsArtifact b
  | e                =>
      if e.isConstOf ``AlgorithmLib.Artifact then true
      else match e.getAppFn, e.getAppArgs with
           | .const ``Except _, #[_, a] => yieldsArtifact a
           | _, _ => false

/-- Fails the build on an artifact holding a body that was not compiled through
    `compileProg`. -/
def check (label : String) (root : Name := `main) (gatedElsewhere : Option String := none) :
    CoreM Unit := do
  let env ← getEnv
  if (env.find? root).isNone then
    throwError s!"SHIP SCAN [{label}]: no such entry point `{root}`"
  let reached := closureOf env {} root
  let artifacts := reached.toList.filter fun n =>
    match env.find? n with
    | some ci => yieldsArtifact ci.type
    | none    => false
  if artifacts.isEmpty then
    throwError s!"SHIP SCAN [{label}] FAILED: `{root}` reaches no Artifact, so nothing was checked"
  if reached.contains UNCHECKED then
    match gatedElsewhere with
    | some why =>
        IO.println s!"[{label}] compiles through {UNCHECKED}; gated by {why}"
        return
    | none => pure ()
    IO.println s!"UNCHECKED BODY — {label} reaches {UNCHECKED} by:"
    for n in pathTo env root UNCHECKED do IO.println s!"    {n}"
    throwError s!"SHIP SCAN [{label}] FAILED: a body it ships was compiled by \
      {UNCHECKED} rather than one of {CHECKED}"
  IO.println s!"[{label}] every body its {artifacts.length} artifact(s) carry was compiled through compileProg"

end ShipScan
