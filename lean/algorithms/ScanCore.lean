import Lean

/-!
  # The trust scanner's machinery, shared by every pipeline's scanner

  Separate from `TrustScan.lean` so a second pipeline can be scanned: each
  generator file defines its own `main`, so two generators cannot be imported
  into one module.  A thin scanner file per pipeline supplies its own `roots`
  and its own `Surface`.

  **Nothing here names a pipeline.**  Not the compressor, not the model stack —
  no `Float32`, no cuBLAS, no PTX machine.  What lives here is what a scan *is*:
  the constant closure, the hypothesis and conclusion readers, the three axioms
  of classical Lean, `native_decide`'s footprint, and the driver.  What a given
  development may rest on is a `Surface`, declared beside that development's own
  claims — `MlSurface.lean` for the model stack, `lz4Surface` in `Lz4Scan.lean`
  for the compressor.

  That split is the property worth keeping: a scan is only meaningful against
  the surface of the thing being scanned, and a shared module naming one
  development's allowances would let the other borrow them by importing it.
-/

open Lean

namespace TrustScan

/-- Transitive constant closure of a declaration's type and proof term. -/
partial def closure (env : Environment) (seen : Std.HashSet Name) (n : Name) :
    Std.HashSet Name :=
  if seen.contains n then seen
  else
    let seen := seen.insert n
    match env.find? n with
    | none => seen
    | some ci =>
        let cs := ci.type.getUsedConstants ++
                  (match ci.value? with | some v => v.getUsedConstants | none => #[])
        cs.foldl (fun acc c => closure env acc c) seen

/-- The three axioms of ordinary classical Lean.  Anything else is a finding —
    `sorryAx` above all, but also `Lean.ofReduceBool`/`Lean.trustCompiler`,
    which is `native_decide` trusting the compiler. -/
def allowedAxiom : List Name := [`propext, `Classical.choice, `Quot.sound]

/-- `native_decide`'s footprint, kept separate so it is reported rather than
    hidden: it is a *named* gap, not part of the agreed surface. -/
def nativeDecideAxioms : List Name := [`Lean.ofReduceBool, `Lean.trustCompiler]
def nativeDecideOpaque : List Name := [`Lean.reduceBool]

/-- **Hypotheses a claim still carries.**

    A scan of axioms and opaques says nothing about these, and that is the hole
    that let `SmMeta` sit undischarged: `theorem foo (h : P) : Q` is trivially
    true when `P` is unsatisfiable, and no closure walk can see it.  This
    reports the Prop-valued binders of each claim's *type*, so an assumption
    cannot be added silently. -/
private partial def hypsGo (e : Expr) (acc : Array Expr) : MetaM (Array Expr) := do
  match e with
  | .forallE nm d b bi =>
      let acc := if (← Meta.isProp d) then acc.push d else acc
      Meta.withLocalDecl nm bi d fun x => hypsGo (b.instantiate1 x) acc
  | _ => return acc

def hypsOf (n : Name) : MetaM (Array Expr) := do
  let env ← getEnv
  match env.find? n with
  | none => return #[]
  | some ci => hypsGo ci.type #[]

private partial def conclGo (e : Expr) : MetaM Expr := do
  match e with
  | .forallE nm d b bi => Meta.withLocalDecl nm bi d fun x => conclGo (b.instantiate1 x)
  | _ => return e

/-- The head constant a claim concludes with.  Pairing an obligation with a
    discharger that exists but proves something else is otherwise invisible. -/
def conclHead (n : Name) : MetaM (Option Name) := do
  let env ← getEnv
  match env.find? n with
  | none => return none
  | some ci => return (← conclGo ci.type).getAppFn.constName?

/-- **A pipeline's declared surface, as one value.**

    The four lists above are the *inference* pipeline's answers.  A second
    development — the LZ4 compressor — rests on a different base (no floats, no
    cuBLAS; a PTX machine and a memory model instead), and a scan is only
    meaningful against the surface of the thing being scanned.  Bundling them
    lets each pipeline declare its own without either being able to quietly
    borrow the other's allowances. -/
structure Surface where
  allowedOpaque      : List Name := []
  allowedHyp         : List Name := []
  derivedObligations : List (Name × Name) := []
  openObligations    : List Name := []
  /-- **Endpoints, and the exact hypotheses each may carry.**

      `derivedObligations` says a component is proven; it cannot say the
      component is *used*.  An unused layer leaves its obligation in the top
      claim's hypothesis list, so pinning that list catches it.  Adding an
      assumption, or proving a layer at one geometry and not the other, fails
      here. -/
  endpoints          : List (Name × List Name) := []
  /-- **Modules whose every declaration a root must reach.**

      The closure walk runs over elaborated proof *terms*, so a lemma used only
      through a simp set or an instance still counts as reached — which a
      textual search cannot tell.  Anything in these modules that no root reaches
      is either dead or a proof that was written and never connected, and those
      are the same finding from the build's point of view. -/
  reachedModules     : List Name := []
  /-- Declarations in `reachedModules` that are deliberately unreached. -/
  reachedExempt      : List Name := []

structure Finding where
  root       : Name
  badAxioms  : Array Name
  badOpaque  : Array Name
  usesNative : Bool

def scan (surf : Surface) (env : Environment) (root : Name) : Finding := Id.run do
  let cl := closure env {} root
  let mut badA : Array Name := #[]
  let mut badO : Array Name := #[]
  let mut nat  : Bool := false
  for n in cl.toList do
    match env.find? n with
    | some (.axiomInfo _) =>
        if n ∈ nativeDecideAxioms then nat := true
        else if !(n ∈ allowedAxiom) then badA := badA.push n
    | some (.opaqueInfo _) =>
        if n ∈ nativeDecideOpaque then nat := true
        else if !(n ∈ surf.allowedOpaque) then badO := badO.push n
    | _ => pure ()
  return { root := root, badAxioms := badA.qsort Name.lt,
           badOpaque := badO.qsort Name.lt, usesNative := nat }

/-- Auto-generated companions of a declaration, which no one references by hand.
    Recognised by name, because they are what a reachability report would
    otherwise drown in. -/
def isGenerated (env : Environment) (n : Name) : Bool :=
  n.isInternal || (env.getProjectionFnInfo? n).isSome ||
  (match n with
   | .str _ s =>
       s.startsWith "proof_" || s.startsWith "eq_" || s.startsWith "match_" ||
       s.startsWith "_" || s.endsWith "_cstage1" || s.endsWith "_cstage2" ||
       s ∈ ["rec", "recOn", "casesOn", "below", "brecOn", "ibelow",
            "binductionOn", "noConfusion", "noConfusionType", "injEq", "inj",
            "sizeOf_spec", "ctorIdx", "toCtorIdx", "ofNat_toCtorIdx", "induct",
            "fun_cases", "mk", "eq_def", "sizeOf_inst"]
   | _ => false)

/-- **The scan, as a driver.**  Takes the claim list so a pipeline's scanner is
    just its `roots` plus one call.  Throws — i.e. fails the build — on anything
    outside the declared surface, on an unknown claim name, and on an
    undeclared hypothesis. -/
def runScanWith (surf : Surface) (label : String) (roots : List Name) : CoreM Unit := do
  let env ← getEnv
  let mut bad := 0
  let mut native : Array Name := #[]
  let mut missing : Array Name := #[]
  for r in roots do
    if (env.find? r).isNone then
      missing := missing.push r
    else
      let f := scan surf env r
      if f.usesNative then native := native.push r
      if f.badAxioms.size != 0 || f.badOpaque.size != 0 then
        bad := bad + 1
        IO.println s!"UNDECLARED  {r}"
        if f.badAxioms.size != 0 then IO.println s!"   axioms: {f.badAxioms.toList}"
        if f.badOpaque.size != 0 then IO.println s!"   opaque: {f.badOpaque.toList}"
  IO.println s!"[{label}] scanned {roots.length} claims"
  IO.println s!"[{label}] native_decide reached by {native.size}: {native.toList}"
  -- hypotheses: reported per claim, so an assumption cannot be added silently
  let mut withHyps := 0
  let mut withDerived := 0
  let mut seenDerived : Array Name := #[]
  for r in roots do
    if (env.find? r).isSome then
      let hs ← Meta.MetaM.run' (hypsOf r)
      let mut bads : Array Name := #[]
      let mut obls : Array Name := #[]
      let mut derived : Array Name := #[]
      for e in hs do
        match e.getAppFn with
        | .const c _ =>
            if c ∈ surf.openObligations then obls := obls.push c
            else if surf.derivedObligations.any (fun q => q.1 == c) then derived := derived.push c
            else if !(c ∈ surf.allowedHyp) then bads := bads.push c
        | _ => pure ()
      if derived.size != 0 then
        withDerived := withDerived + 1
        for d in derived do
          if !(seenDerived.contains d) then seenDerived := seenDerived.push d
      if obls.size != 0 then
        withHyps := withHyps + 1
        IO.println s!"  OPEN OBLIGATION  {r}: {obls.toList}"
      if bads.size != 0 then
        bad := bad + 1
        IO.println s!"UNDECLARED HYPOTHESIS  {r}: {bads.toList}"
  IO.println s!"[{label}] claims resting on an open obligation: {withHyps} of {roots.length}"
  IO.println s!"[{label}] claims resting on a *derived* obligation: {withDerived} of {roots.length}"
  -- a discharge that does not exist is a failure, not a comment.  Checked only
  -- for obligations this scan actually reaches: the two pipelines share this
  -- machinery but not their environments.
  for d in seenDerived do
    match surf.derivedObligations.find? (fun q => q.1 == d) with
    | some (_, by_) =>
        if (env.find? by_).isNone then
          bad := bad + 1
          IO.println s!"MISSING DISCHARGE  {d}: {by_} is not in the environment"
        else
          -- existing is not enough: the discharger must conclude the obligation
          match ← Meta.MetaM.run' (conclHead by_) with
          | some c =>
              if c != d then
                bad := bad + 1
                IO.println s!"WRONG DISCHARGE  {d}: {by_} concludes {c}"
              else IO.println s!"[{label}]   {d} is discharged by {by_}"
          | none =>
              bad := bad + 1
              IO.println s!"WRONG DISCHARGE  {d}: {by_} has no head constant"
    | none => pure ()
  -- endpoints: the exact hypothesis set, so a proven layer cannot go unused
  for (e, allowed) in surf.endpoints do
    if (env.find? e).isNone then
      bad := bad + 1
      IO.println s!"MISSING ENDPOINT  {e}"
    else
      let hs ← Meta.MetaM.run' (hypsOf e)
      let mut extra : Array Name := #[]
      for h in hs do
        match h.getAppFn with
        | .const c _ => if !(c ∈ allowed) then extra := extra.push c
        | _ => pure ()
      if extra.size != 0 then
        bad := bad + 1
        IO.println s!"ENDPOINT ASSUMES MORE THAN DECLARED  {e}: {extra.toList}"
      else IO.println s!"[{label}] endpoint {e} rests on exactly {allowed}"
  -- reachability: a declaration no root reaches is dead, or a proof never wired in
  if !surf.reachedModules.isEmpty then
    let mut reach : Std.HashSet Name := {}
    for r in roots do reach := closure env reach r
    let mut orphan : Array Name := #[]
    for (n, ci) in env.constants.toList do
      match ci with
      | .thmInfo _ | .defnInfo _ =>
          if !(isGenerated env n) && !(reach.contains n) && !(n ∈ surf.reachedExempt) then
            match env.getModuleIdxFor? n with
            | some idx =>
                if env.header.moduleNames[idx.toNat]! ∈ surf.reachedModules then
                  orphan := orphan.push n
            | none => pure ()
      | _ => pure ()
    if orphan.size != 0 then
      bad := bad + 1
      IO.println s!"UNREACHED ({orphan.size}) — no claim uses these:"
      for n in orphan.qsort Name.lt do IO.println s!"    {n}"
    else
      IO.println s!"[{label}] every declaration in {surf.reachedModules.length} module(s) is reached by a claim"
  if missing.size != 0 then
    throwError s!"TRUST SCAN [{label}]: unknown claim(s) {missing.toList}"
  if bad != 0 then
    throwError s!"TRUST SCAN [{label}] FAILED: {bad} claim(s) outside the declared surface"
  IO.println s!"[{label}] TRUST SCAN OK — every claim inside the declared surface"

end TrustScan
