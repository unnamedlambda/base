import AlgorithmLib.FFI

/-!
# The standard callee table

Every entry point the runtime exports, declared once, in one order. A generator
names a callee through `FFI.std` and never writes a signature, so two files
cannot describe the same C symbol differently — the drift a per-file
declaration block permits is not expressible here.

The table is also what `Sur.build`, `buildChecked` and `compileFn` default
their `env` to, which is why a body carries no callee table in its text.
`compileFn` ships only the declarations a body actually calls, so a function
that calls two entry points declares two.
-/

namespace AlgorithmLib

namespace IR

namespace FFI

/-- Everything the runtime exports, in bundles. -/
structure Std where
  fileRead : FnRef
  fileWrite : FnRef
  fileReadToPtr : FnRef
  fileWriteFromPtr : FnRef
  stdinReadline : FnRef
  stdoutWrite : FnRef
  gpu : GpuSetup
  window : WindowSetup
  lmdb : LmdbSetup
  ht : HtSetup
  math : MathSetup
  thread : ThreadSetup
  cuda : CudaSetup
  cublas : CuBlasSetup
  deriving Inhabited, Lean.ToExpr

/-- The one declaration run. Order fixes every callee id in the library; it is
    an implementation detail, since a body names callees through `std`. -/
def stdWith : Std × FnEnv := envOf do
  let fileRead ← declareFileRead
  let fileWrite ← declareFileWrite
  let fileReadToPtr ← declareFileReadToPtr
  let fileWriteFromPtr ← declareFileWriteFromPtr
  let stdinReadline ← declareStdinReadline
  let stdoutWrite ← declareStdoutWrite
  let gpu ← declareGpuFFI
  let window ← declareWindowFFI
  let lmdb ← declareLmdbFFI
  let ht ← declareHtFFI
  let math ← declareMathFFI
  let thread ← declareThreadFFI
  let cuda ← declareCudaFFI
  let cublas ← declareCuBlasFFI
  pure { fileRead, fileWrite, fileReadToPtr, fileWriteFromPtr, stdinReadline, stdoutWrite,
         gpu, window, lmdb, ht, math, thread, cuda, cublas }

open Lean Meta Elab Term in
unsafe def evalStdUnsafe (e : Expr) : TermElabM Std :=
  evalExpr Std (mkConst ``Std) e

instance : Inhabited (Lean.Elab.TermElabM Std) := ⟨pure default⟩

@[implemented_by evalStdUnsafe]
opaque evalStd (e : Lean.Expr) : Lean.Elab.TermElabM Std

open Lean Elab Term in
/-- The named callees, spliced as a literal.

    `stdWith` is a `StateM` run. A body that names `std.cuda.fnLaunch` would
    otherwise carry that whole run in its term, and every `decide` touching the
    body would re-reduce all eighty-odd declarations to reach one `FnRef`. -/
elab "std%" : term => do
  let e ← elabTermEnsuringType (← `(stdWith.1)) (mkConst ``Std)
  synthesizeSyntheticMVarsNoPostponing
  return toExpr (← evalStd (← instantiateMVars e))

/-- The named callees. -/
def std : Std := std%

open Lean Meta Elab Term in
unsafe def evalFnEnvUnsafe (e : Expr) : TermElabM FnEnv :=
  evalExpr FnEnv (mkConst ``FnEnv) e

instance : Inhabited (Lean.Elab.TermElabM FnEnv) := ⟨pure default⟩

@[implemented_by evalFnEnvUnsafe]
opaque evalFnEnv (e : Lean.Expr) : Lean.Elab.TermElabM FnEnv

open Lean Elab Term in
/-- Splice the table `stdWith` builds as a first-order literal.

    `stdWith` is a `StateM` run over eighty-odd declarations. Left as such,
    `decide (wf stdEnv ..)` has to kernel-reduce that whole monadic block before
    it can look up a single callee — which does not finish. Evaluating it here
    and splicing the records it produced leaves `decide` with plain lists to
    walk, which is what it is good at. -/
elab "stdEnv%" : term => do
  let e ← elabTermEnsuringType (← `(stdWith.2)) (mkConst ``FnEnv)
  synthesizeSyntheticMVarsNoPostponing
  return toExpr (← evalFnEnv (← instantiateMVars e))

/-- The callee table every body is checked and compiled against. -/
def stdEnv : FnEnv := stdEnv%

/-- A named group of entry points, so a body can be checked against the part of
    the table it uses. -/
inductive Bundle where
  | fileIO | gpu | window | lmdb | ht | math | thread | cuda | cublas
  deriving Repr, BEq

def Bundle.refs : Bundle → List FnRef
  | .fileIO => [std.fileRead, std.fileWrite, std.fileReadToPtr, std.fileWriteFromPtr,
                std.stdinReadline, std.stdoutWrite]
  | .gpu => let g := std.gpu
            [g.fnInit, g.fnCreateBuffer, g.fnUpload, g.fnDownload, g.fnCreatePipeline,
             g.fnDispatch, g.fnCleanup, g.fnUploadPtr, g.fnDownloadPtr]
  | .window => let w := std.window
               [w.fnInit, w.fnOpen, w.fnPoll, w.fnPresentGpuBuffer, w.fnCleanup]
  | .lmdb => let l := std.lmdb
             [l.fnInit, l.fnOpen, l.fnBeginWriteTxn, l.fnPut, l.fnCommitWriteTxn,
              l.fnCursorScan, l.fnCleanup]
  | .ht => let h := std.ht
           [h.fnCreate, h.fnLookup, h.fnInsert, h.fnIncrement, h.fnCount, h.fnGetEntry,
            h.fnCleanup, h.fnInit]
  | .math => let m := std.math; [m.fnSinf, m.fnCosf, m.fnPowf]
  | .thread => let t := std.thread; [t.fnInit, t.fnSpawn, t.fnJoin, t.fnCleanup]
  | .cuda => let c := std.cuda; [c.fnInit, c.fnCreateBuffer, c.fnUpload, c.fnUploadOffset, 
      c.fnUploadAsync, c.fnUploadOffsetAsync, c.fnDownload, c.fnDownloadOffset, 
      c.fnDownloadAsync, c.fnFreeBuffer, c.fnStreamCreate, c.fnStreamSync, c.fnStreamDestroy, 
      c.fnEventCreate, c.fnEventRecord, c.fnStreamWaitEvent, c.fnEventElapsedMsBits, 
      c.fnEventDestroy, c.fnGraphBeginCapture, c.fnGraphEndCapture, c.fnGraphUpload, 
      c.fnGraphLaunch, c.fnGraphDestroy, c.fnPinnedAlloc, c.fnPinnedPtr, c.fnPinnedFree, 
      c.fnLaunch, c.fnLaunchNamed, c.fnLaunchOnStream, c.fnLaunchNamedOnStream, c.fnSync, 
      c.fnCleanup]
  | .cublas => let b := std.cublas; [b.fnSgemv, b.fnSgemvOnStream, b.fnSgemm, 
      b.fnSgemmOnStream, b.fnPtrArray, b.fnSgemmBatchedOnStream]

/-- The part of the standard table these bundles name. -/
def ofBundles (bs : List Bundle) : FnEnv :=
  let keep := bs.flatMap Bundle.refs
  let fns := stdEnv.fns.filter (fun d => keep.any (·.id == d.ref.id))
  { fns, sigs := stdEnv.sigs.filter (fun sg => fns.any (·.sig.id == sg.ref.id)) }

open Lean Elab Term in
/-- `env% [.gpu, .window]` — the named part of the standard table, spliced as a
    literal.

    Signatures still come from one place, so nothing can drift; what a body is
    checked against stays the handful of entry points it actually uses, which is
    what keeps `decide (wf ..)` cheap. Left as a filter over the whole table,
    every check would walk all ninety declarations at every call site. -/
elab "env% " bs:term : term => do
  let e ← elabTermEnsuringType (← `(ofBundles $bs)) (mkConst ``FnEnv)
  synthesizeSyntheticMVarsNoPostponing
  return toExpr (← evalFnEnv (← instantiateMVars e))

end FFI

end IR

end AlgorithmLib
