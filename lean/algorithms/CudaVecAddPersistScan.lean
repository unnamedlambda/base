import GenSurface
import CudaVecAddPersistAlgorithm

/-!
  # What the persistent-`x + y` artifact's claims rest on — computed, not documented

  **This generator writes no theorem of its own, and it does not need most of
  one.**  It is four lines of `CudaPipeline.Expr`, and the library it compiles
  through carries the storage argument for every program built that way.  The
  report is therefore about where the claims live, not that they are absent.
-/

open Lean

namespace CudaVecAddPersistScan

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
  , `CudaVecAddPersist.result
  , `CudaVecAddPersist.artifacts ]

def notYetStated : List String :=
  [ "that the PTX the pipeline emits computes `x + y` — the expression DSL is      lowered by `emitExprPTX`, and nothing states that the lowering is faithful",
    "that the three stages run in the order the runtime calls them",
    "and therefore: the numerical result rests on differential testing, while      storage and well-formedness rest on the library's proofs" ]

end CudaVecAddPersistScan

/-- Claims that rest on the compiler, via `native_decide`. -/
def nativeRoster : List Name :=
  [ `CudaVecAddPersist.result
   , `CudaVecAddPersist.artifacts ]

open TrustScan CudaVecAddPersistScan in
#eval runGenScan "vecadd" roots nativeRoster

#eval do
  IO.println s!"[vecadd] roots scanned: {CudaVecAddPersistScan.roots.length} (two of them library claims)"
  for s in CudaVecAddPersistScan.notYetStated do
    IO.println s!"[vecadd] NOT STATED: {s}"
