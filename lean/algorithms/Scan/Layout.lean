import Lean

/-!
  # Every address a module names must be inside its region map

  `RegionMap.okB` asks whether the listed regions overlap — a question about the
  list. This asks whether the list is complete: every `Nat` a module declares
  under an address-shaped name must appear in the map, or be named as an
  exception.

  The map is reached through its elaborated value, so an offset the map uses via
  a helper counts as mentioned. An address that never got a name is outside this
  check; that needs the emitted stores.
-/

open Lean

namespace LayoutScan

/-- Names that read as an address: `PTX_OFF`, `IN_ID`, `bindOff`. -/
def isAddressName (n : Name) : Bool :=
  match n with
  | .str _ s => s.endsWith "_OFF" || s.endsWith "_ID" || s.endsWith "Off"
  | _        => false

/-- Declared by the module being elaborated: the environment gives an imported
    constant a module index and a local one none. -/
private def isLocal (env : Environment) (n : Name) : Bool :=
  (env.getModuleIdxFor? n).isNone

/-- Transitive constant closure of a declaration's value and type. -/
partial def closureOf (env : Environment) (seen : Std.HashSet Name) (n : Name) :
    Std.HashSet Name :=
  if seen.contains n then seen
  else
    let seen := seen.insert n
    match env.find? n with
    | none    => seen
    | some ci =>
        let cs := ci.type.getUsedConstants ++
                  (match ci.value? with | some v => v.getUsedConstants | none => #[])
        cs.foldl (fun acc c => closureOf env acc c) seen

/-- Addresses this module names that `mapDecls` do not reach. -/
def uncovered (env : Environment) (mapDecls : List Name) (exempt : List Name) :
    Array Name := Id.run do
  let mut seen : Std.HashSet Name := {}
  for m in mapDecls do
    seen := closureOf env seen m
  let mut out : Array Name := #[]
  for (n, ci) in env.constants.toList do
    if isLocal env n && isAddressName n && !n.isInternal
        && !(n ∈ exempt) && !(seen.contains n) then
      match ci with
      | .defnInfo d => if d.type.isConstOf ``Nat then out := out.push n
      | _           => pure ()
  return out.qsort Name.lt

/-- Fails the build on an address outside the map. `exempt` names, per
    declaration, the offsets that are not regions of this memory. -/
def check (label : String) (mapDecls : List Name) (exempt : List Name := []) :
    CoreM Unit := do
  let env ← getEnv
  for m in mapDecls do
    if (env.find? m).isNone then
      throwError s!"LAYOUT SCAN [{label}]: no such map `{m}`"
  let bad := uncovered env mapDecls exempt
  if bad.size != 0 then
    IO.println s!"UNCOVERED ADDRESSES ({bad.size}) — named by {label}, absent from its map:"
    for n in bad do IO.println s!"    {n}"
    throwError s!"LAYOUT SCAN [{label}] FAILED: {bad.size} address(es) outside the map"
  IO.println s!"[{label}] every address this module names is inside its region map"

end LayoutScan
