import GenSurface
import CudaSaxpyPersistAlgorithm

/-!
  # What the persistent-`a·x + y` artifact's claims rest on — computed, not documented

  **This generator writes no theorem of its own, and it does not need most of
  one.**  It is four lines of `CudaPipeline.Expr`, and the library it compiles
  through carries the storage argument for every program built that way.  The
  report is therefore about where the claims live, not that they are absent.
-/

open Lean

namespace CudaSaxpyPersistScan

/-- **The public claims, two of them from the library rather than here.**

    `CudaPipeline.memMap_ok` is the shared layout, proved disjoint over a
    *range* of arities rather than at one — the input-id block is the only
    region whose size depends on the program, so stating it at a single arity
    would say nothing about this one.

    `result` is what the generator actually builds.  Scanning it covers the
    three `wf` obligations `compileTo` demands, which are discharged at
    elaboration by `native_decide` — so the scan reports this artifact as
    reaching the compiler, which it does, rather than quietly not mentioning
    it. -/
def roots : List Name :=
  [ `AlgorithmLib.CudaPipeline.memMap_ok
  , `CudaSaxpyPersist.result
  , `CudaSaxpyPersist.artifacts ]

def notYetStated : List String :=
  [ "that the PTX the pipeline emits computes `a·x + y` — the expression DSL is      lowered by `emitExprPTX`, and nothing states that the lowering is faithful",
    "that the three stages run in the order the runtime calls them",
    "and therefore: the numerical result rests on differential testing, while      storage and well-formedness rest on the library's proofs" ]

end CudaSaxpyPersistScan

/-- Claims that rest on the compiler, via `native_decide`. -/
def nativeRoster : List Name :=
  [ `CudaSaxpyPersist.result
   , `CudaSaxpyPersist.artifacts ]

open TrustScan CudaSaxpyPersistScan in
#eval runGenScan "saxpy" roots nativeRoster

#eval do
  IO.println s!"[saxpy] roots scanned: {CudaSaxpyPersistScan.roots.length} (two of them library claims)"
  for s in CudaSaxpyPersistScan.notYetStated do
    IO.println s!"[saxpy] NOT STATED: {s}"
