import Scan.Core
import Demo.Sat
import Demo.Sha256
import Demo.LeanEval
import Bench.CudaSaxpyPersist
import Bench.CudaVecAddPersist
/-!
  # The utility generators' scans

  These generators ship an artifact and prove little or nothing about what it
  computes, so each scan's job is to report that exactly: the claims that do
  exist, the base they rest on, and what is not stated.  They share one
  surface and one file.

  Plain rather than a module, as every scan is: it reads proof terms, which a
  module does not see.
-/

open Lean

/-!
  ## The surface the utility generators' scans run at

  These generators ship an artifact and prove almost nothing about it, so their
  scans exist to report that rather than to certify it.  The surface is
  therefore as narrow as it can be: two opaque constants, both from Lean's own
  `String`, and nothing else.

  Kept in one module because the six scanners are separate only for reasons
  that have nothing to do with the surface — each generator defines its own
  `main`, and `SatAlgorithm`/`Sha256Algorithm` additionally share
  `namespace Algorithm` — so there is no reason to state the allowance six
  times and every reason not to.
-/

namespace TrustScan

/-- **Lean's string representation, not an assumption about any algorithm.**

    Reached because a scanned root is the emitted *program*, which contains
    string literals — kernel names, PTX text, the CLIF function names.  Neither
    constant says anything about what the generator computes, and a scan that
    hid them would be reporting a smaller base than the artifact has.

    Anything else these scans reach is a real widening and must be added here,
    where the diff is reviewable. -/
def genAllowedOpaque : List Name := [`Lean.opaqueId, `String.Internal.append]

/-- The utility generators' surface: no allowed hypotheses, no obligations,
    and the two string opaques above. -/
def genSurface : Surface := { allowedOpaque := genAllowedOpaque }

/-- Their scan, at `genSurface`. -/
def runGenScan (label : String) (roots : List Name) (nativeRoster : List Name) : CoreM Unit :=
  runScanWith genSurface label roots nativeRoster

end TrustScan

/-!
  ## What the SAT artifact's claims rest on — computed, not documented

  A scanner per generator, for the reason the others give: each defines its own
  `main`, so they cannot share a module.  `SatAlgorithm` and `Sha256Algorithm`
  additionally share `namespace Algorithm`, so importing both is not possible
  even in principle.

  **This one is nearly empty, and that is the report.**  Two claims, both about
  storage.  The solver's *answer* is backed by differential testing against a
  reference, not by a theorem, and a scan over two roots does not say the
  artifact is trustworthy — it says exactly how much of it has been argued.
-/

namespace SatScan

/-- **The public claims.**

    That the regions the generator hands out do not overlap, and that they fit
    the memory it asks for.  Both by `decide`, so they are checked rather than
    asserted, and both fail the build if an offset moves onto its neighbour.

    The one deliberate alias is recorded at the definition rather than here:
    `clauseIndex_off` *is* `out_off`, because the output string is built where
    the clause index sat once solving is done.  It appears in `memMap` under one
    name, which is why `okB` accepts it. -/
def roots : List Name :=
  [ `Sat.memMap_ok
  , `Sat.memMap_within ]

/-- **What this artifact does not state.** -/
def notYetStated : List String :=
  [ "that the assignment it prints satisfies the formula it read \
     (a checker over the emitted output would be the cheap form of this)",
    "that UNSAT means no assignment exists — nothing here produces a refutation",
    "that the CDCL loop terminates",
    "and therefore: every claim about the solver's answer rests on differential \
     testing against a reference solver, not on a proof" ]

end SatScan

open TrustScan SatScan in
#eval runGenScan "sat" roots []

#eval do
  IO.println s!"[sat] roots scanned: {SatScan.roots.length}"
  for s in SatScan.notYetStated do
    IO.println s!"[sat] NOT STATED: {s}"

/-!
  ## What the SHA-256 artifact's claims rest on — computed, not documented

  A scanner per generator; this one cannot share a module with `SatScan` for a
  second reason beyond `main`, which is that both generators use
  `namespace Algorithm`.

  **This one names no theorem, and that is the report.**  What it names instead
  is the shipped program itself, so the scan reports the trusted base the
  emitted artifact actually reaches rather than the base of a theorem someone
  chose to write.
-/

namespace Sha256Scan

/-- **The shipped value, not a claim about it.**

    `clifIrSource` is the program that becomes the artifact.  Scanning it walks
    the closure of what the generator *emits*, which is the honest thing to
    report for a generator that proves nothing about its own output.

    Storage is the exception, and it is handled by construction rather than by
    a theorem: this generator allocates through `Layout.build`, which hands out
    offsets in sequence, so two fields cannot collide.  That is why there is no
    `memMap_ok` here and why its absence is not a gap — the stronger form is
    already in force.  The hand-written maps elsewhere need the theorem exactly
    because they are hand-written. -/
def roots : List Name :=
  [ `Sha256.clifIrSource ]

/-- **What this artifact does not state.** -/
def notYetStated : List String :=
  [ "that the digest it computes is SHA-256 — no reference to the standard's \
     compression function appears in any statement here",
    "that the message schedule and padding are the specification's",
    "and therefore: correctness rests entirely on differential testing against \
     a reference implementation over the test corpus" ]

end Sha256Scan

open TrustScan Sha256Scan in
#eval runGenScan "sha256" roots []

#eval do
  IO.println s!"[sha256] roots scanned: {Sha256Scan.roots.length} (the emitted program, not a theorem)"
  for s in Sha256Scan.notYetStated do
    IO.println s!"[sha256] NOT STATED: {s}"

/-!
  ## What the Lean-eval artifact's claims rest on — computed, not documented

  **One storage claim and the emitted program, and that is the report.**  What
  this generator ships is an evaluator; nothing here states what it evaluates.
-/

namespace LeanEvalScan

/-- **The public claims.**

    `memMap_ok` — the regions this generator hands out do not overlap, by
    `decide`.  There is no `memMap_within` companion: this artifact's memory is
    sized from the payload it builds rather than fixed in advance, so
    disjointness is the whole of what a static map can say.

    `clifIrSource` is the shipped program, scanned so the report is the base the
    artifact actually reaches rather than the base of the one theorem. -/
def roots : List Name :=
  [ `LeanEval.memMap_ok
  , `LeanEval.clifIrSource ]

def notYetStated : List String :=
  [ "that the evaluator computes what Lean's own evaluator computes",
    "that its parser accepts the language it claims to accept",
    "that memory sized from the payload is large enough for the payload — the \
     sizing is arithmetic in the generator, not a theorem",
    "and therefore: correctness rests on differential testing against Lean" ]

end LeanEvalScan

open TrustScan LeanEvalScan in
#eval runGenScan "leaneval" roots []

#eval do
  IO.println s!"[leaneval] roots scanned: {LeanEvalScan.roots.length}"
  for s in LeanEvalScan.notYetStated do
    IO.println s!"[leaneval] NOT STATED: {s}"

/-!
  ## What the persistent-`a·x + y` artifact's claims rest on — computed, not documented

  **This generator writes no theorem of its own, and it does not need most of
  one.**  It is four lines of `CudaPipeline.Expr`, and the library it compiles
  through carries the storage argument for every program built that way.  The
  report is therefore about where the claims live, not that they are absent.
-/

namespace CudaSaxpyPersistScan

/-- **The public claims, two of them from the library rather than here.**

    `CudaPipeline.memMap_ok` is the shared layout, proved disjoint over a
    *range* of arities rather than at one — the input-id block is the only
    region whose size depends on the program, so stating it at a single arity
    would say nothing about this one.

    `result` is what the generator actually builds.  The surface it is written
    in checks well-formedness in its own types and at generation time, so the
    scan finds no claim here resting on the compiler. -/
def roots : List Name :=
  [ `AlgorithmLib.CudaPipeline.memMap_ok
  , `CudaSaxpyPersist.result
  , `CudaSaxpyPersist.artifacts ]

def notYetStated : List String :=
  [ "that the PTX the pipeline emits computes `a·x + y` — the expression DSL is      lowered by `emitExprPTX`, and nothing states that the lowering is faithful",
    "that the three stages run in the order the runtime calls them",
    "and therefore: the numerical result rests on differential testing, while      storage and well-formedness rest on the library's proofs" ]

end CudaSaxpyPersistScan

/-- Nothing here rests on the compiler: the stages are typed terms, and
    `compileProg` checks the body it emitted. -/
def CudaSaxpyPersistScan.nativeRoster : List Name := []

open TrustScan CudaSaxpyPersistScan in
#eval runGenScan "saxpy" roots nativeRoster

#eval do
  IO.println s!"[saxpy] roots scanned: {CudaSaxpyPersistScan.roots.length} (two of them library claims)"
  for s in CudaSaxpyPersistScan.notYetStated do
    IO.println s!"[saxpy] NOT STATED: {s}"

/-!
  ## What the persistent-`x + y` artifact's claims rest on — computed, not documented

  **This generator writes no theorem of its own, and it does not need most of
  one.**  It is four lines of `CudaPipeline.Expr`, and the library it compiles
  through carries the storage argument for every program built that way.  The
  report is therefore about where the claims live, not that they are absent.
-/

namespace CudaVecAddPersistScan

/-- **The public claims, two of them from the library rather than here.**

    `CudaPipeline.memMap_ok` is the shared layout, proved disjoint over a
    *range* of arities rather than at one — the input-id block is the only
    region whose size depends on the program, so stating it at a single arity
    would say nothing about this one.

    `result` is what the generator actually builds.  The surface it is written
    in checks well-formedness in its own types and at generation time, so the
    scan finds no claim here resting on the compiler. -/
def roots : List Name :=
  [ `AlgorithmLib.CudaPipeline.memMap_ok
  , `CudaVecAddPersist.result
  , `CudaVecAddPersist.artifacts ]

def notYetStated : List String :=
  [ "that the PTX the pipeline emits computes `x + y` — the expression DSL is      lowered by `emitExprPTX`, and nothing states that the lowering is faithful",
    "that the three stages run in the order the runtime calls them",
    "and therefore: the numerical result rests on differential testing, while      storage and well-formedness rest on the library's proofs" ]

end CudaVecAddPersistScan

/-- Nothing here rests on the compiler: the stages are typed terms, and
    `compileProg` checks the body it emitted. -/
def CudaVecAddPersistScan.nativeRoster : List Name := []

open TrustScan CudaVecAddPersistScan in
#eval runGenScan "vecadd" roots nativeRoster

#eval do
  IO.println s!"[vecadd] roots scanned: {CudaVecAddPersistScan.roots.length} (two of them library claims)"
  for s in CudaVecAddPersistScan.notYetStated do
    IO.println s!"[vecadd] NOT STATED: {s}"
