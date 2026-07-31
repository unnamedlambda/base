import Lean

/-!
  # The trust scanner's machinery, shared by every pipeline's scanner

  Split out of `TrustScan.lean` so a second pipeline can be scanned: each
  generator file defines its own `main`, so `Qwen2Algorithm` and
  `BackwardWideAlgorithm` cannot be imported into one module.  Two thin scanner
  files import this one and supply their own `roots`.

  Nothing here imports a generator, and that is the point — the machinery is not
  allowed to depend on what it is scanning.
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

/-- **The declared trust surface**, as data.

    * `Float32.*` / `float32Spec` — IEEE-754 primitives are opaque, which is
      what makes float associativity a stated law rather than a silent
      assumption.
    * `cublasSgemvResult` / `sgemmBatchedRow` — cuBLAS's results.  NVIDIA
      specifies no fold order; `Law.cublasIsMatvec` constrains the first and
      *nothing* constrains the second (`declaredLawGap = 2`).
    * `uploadedValue` — what `cl_cuda_upload_ptr` leaves in device memory.
    * `Lean.opaqueId`, `String.Internal.append` — Lean plumbing, not ours. -/
def allowedOpaque : List Name :=
  [ `float32Spec, `Float32.add, `Float32.mul, `Float32.div, `Float32.neg,
    `Float32.exp, `Float32.sqrt, `Float32.pow, `Float32.decLe,
    `Float32.ofScientific, `Float32.toBits, `Float32.sub, `Float32.lt,
    `Float32.decLt, `Float32.ofBits, `Float32.beq,
    `AlgorithmLib.ML.cublasSgemvResult, `AlgorithmLib.ML.sgemmBatchedRow,
    `Qwen2Proven.Stage.uploadedValue,
    `Lean.opaqueId, `String.Internal.append ]

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

/-- **Hypotheses a claim may carry without comment.**

    * `AllHold` — the named float laws (`Law.combinerComm` &c.).  Float
      associativity/commutativity is trust-surface by agreement, and the Law
      mechanism is what keeps it *visible in the type*.
    * `CuBlasIsMatvec` — cuBLAS's fold order, likewise.
    * `Honours` — discharged by `Qwen2Proven.Stage.idealR_honours`.
    * `MetaFaithful` — Cranelift's `ushr`/`ishl`/`isub`/`store` mean what
      `Clif.DExp.eval` says, and the meta upload delivers its bytes unchanged.
      Both halves were already on the surface (seam 14 and `uploadedValue`);
      naming them together is what let `SmMeta` be discharged rather than
      assumed, and it is deliberately the *only* place this development says
      what those four instructions compute.
    * `LT.lt`, `Eq`, `Nat.le`, `Not` — index side-conditions (`i < D`, and
      `¬ (a < …)` selecting RoPE's upper half), not assumptions about the
      world.  `Not` earned its place here by the scan rejecting the build when
      `rope_hi_is_spec` introduced it. -/
def allowedHyp : List Name :=
  [ `AlgorithmLib.ML.AllHold, `AlgorithmLib.ML.CuBlasIsMatvec,
    `AlgorithmLib.ML.Honours, `Qwen2NonVacuity.MetaFaithful,
    `LT.lt, `LE.le, `Eq, `Ne, `Nat.lt, `Nat.le, `Not ]

/-- **Obligations that were open and are now *derived* from the declared
    surface**, each paired with the theorem that derives it.  The scan checks
    that theorem exists, so this list cannot claim a discharge that is not
    there.

    A claim carrying one of these is still *conditional* — that is what a
    hypothesis is.  What changed is that the condition is no longer an
    assumption about the world: something proves it from things already on the
    surface, and the scan names what.

    * `SmMeta` — what the host owes the softmax kernel about the meta buffer
      (`TAIL = CHUNKS·32`, `CHUNKS·32 + REM ≤ SEQ`, `0 < SEQ`).  Nothing in the
      kernel checks it; a violation makes two store passes overlap silently.

      Discharged by `Qwen2NonVacuity.smMeta_of_frag`, which chains three
      theorems: `Qwen2Common.metaStageFrag_emits` (these are the instructions
      the generator emits, from any builder state), `Qwen2Common.metaFrag_stores`
      (after them the store map holds `seq`, `seq >>> 5`, `(seq >>> 5) <<< 5`
      and `seq − ((seq >>> 5) <<< 5)` at the four slots the kernel reads), and
      `smMeta_of_stores` (those four relations *are* `SmMeta`, by linear
      arithmetic).

      What it rests on is `MetaFaithful` — that Cranelift's `ushr`/`ishl`/
      `isub`/`store` mean what `Clif.DExp.eval` says and that the upload
      delivers the bytes unchanged — plus `0 < seq`, at least one token.  Both
      halves of `MetaFaithful` were already on the surface (seam 14 and
      `uploadedValue`); what is new is that they are now the *only* thing
      between the emitted program and the kernel's assumption, and that the
      four expressions are read off the instruction stream rather than off the
      source. -/
def derivedObligations : List (Name × Name) :=
  [ (`Qwen2Proven.Stage.SmMeta, `Qwen2NonVacuity.smMeta_of_frag) ]

/-- **Open obligations: assumptions that are NOT on the agreed trust surface
    and are NOT derived from it.**

    Empty, and the point of listing it is that it can stop being.  Adding to it
    is a deliberate, reviewable act; `derivedObligations` above is where things
    go when they are closed.

    **What is *not* in this list because it is not a hypothesis of any claim.**

    The end-to-end chain is now uniform: every step rests on the *same* trusted
    base, and none of it rests on a reading of the generator's source.  Where
    the layer loop used to sit, there are now theorems:

    * `Qwen2Common.tokenDriver_deviceOps` — the declared program performs
      `tokenOps`, the twenty-four repetitions coming from
      `HStmt.deviceOps_forN` rather than from a `List.replicate` someone typed;
    * `entryDriver_is_built`, `finalDriver_is_built`, `attnDriver_is_built`,
      `ffnDriver_is_built` — each declared fragment performs exactly the device
      writes a scan recovers from the CLIF the generator produced;
    * `Qwen2Common.infer_loop_is_layers` — the emitted blocks contain exactly
      one counted loop and its bound is `N_LAYERS`, read off the `icmp` and
      resolved the same way a launch's PTX slot is;
    * `Qwen2Common.infer_loop_body_calls`, `layer_fn_calls`,
      `leaf_fns_no_loops`, `Qwen2Common.final_no_loops` — the loop body
      dispatches to the layer function and nothing else, the layer function
      dispatches to the two halves, and nothing else in a decode step loops.

    What is left between those and "the program computes the token" is that
    Cranelift **executes** `jump`/`brif`/`call` as the recovered structure says.
    That is seam 14, and it is not a new assumption: `Clif.deviceOpsOf`
    describing a straight-line function's execution is already a claim of
    exactly the same kind, relied on by `attn_ops_are` and every theorem
    downstream of it.  What changed is that the loop is no longer an
    *additional* gap on top of it.

    The remaining genuinely-open item is narrower and is about the *host model*,
    not this program: `HostIR`'s emitter does not cover `ExternArg.far` or
    `.opaque`, so `flatHI_sound` — the theorem that a compiled `HStmt`
    *executes* to its `deviceOps` — cannot yet be applied to `tokenDriver`.
    `ExternArg.FarFreeB` is exactly that boundary, decidably.  Closing it means
    threading `∀ b ∈ s.farBases, b < n ∧ b ≠ ptr.id ∧ e ⟨b⟩ = unknown` through
    six emit lemmas and `flatHI_sound`; preservation is the same
    `evalPure_frame` argument that already carries `e ptr = unknown`.

    Recording it here rather than in a document because this file is what the
    build reads. -/
def openObligations : List Name := []

structure Finding where
  root       : Name
  badAxioms  : Array Name
  badOpaque  : Array Name
  usesNative : Bool

def scan (env : Environment) (root : Name) : Finding := Id.run do
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
        else if !(n ∈ allowedOpaque) then badO := badO.push n
    | _ => pure ()
  return { root := root, badAxioms := badA.qsort Name.lt,
           badOpaque := badO.qsort Name.lt, usesNative := nat }

/-- **The scan, as a driver.**  Takes the claim list so a pipeline's scanner is
    just its `roots` plus one call.  Throws — i.e. fails the build — on anything
    outside the declared surface, on an unknown claim name, and on an
    undeclared hypothesis. -/
def runScan (label : String) (roots : List Name) : CoreM Unit := do
  let env ← getEnv
  let mut bad := 0
  let mut native : Array Name := #[]
  let mut missing : Array Name := #[]
  for r in roots do
    if (env.find? r).isNone then
      missing := missing.push r
    else
      let f := scan env r
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
            if c ∈ openObligations then obls := obls.push c
            else if derivedObligations.any (fun q => q.1 == c) then derived := derived.push c
            else if !(c ∈ allowedHyp) then bads := bads.push c
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
    match derivedObligations.find? (fun q => q.1 == d) with
    | some (_, by_) =>
        if (env.find? by_).isNone then
          bad := bad + 1
          IO.println s!"MISSING DISCHARGE  {d}: {by_} is not in the environment"
        else IO.println s!"[{label}]   {d} is discharged by {by_}"
    | none => pure ()
  if missing.size != 0 then
    throwError s!"TRUST SCAN [{label}]: unknown claim(s) {missing.toList}"
  if bad != 0 then
    throwError s!"TRUST SCAN [{label}] FAILED: {bad} claim(s) outside the declared surface"
  IO.println s!"[{label}] TRUST SCAN OK — every claim inside the declared surface"

end TrustScan
