module
public import Vit.Units
meta import Vit.Units
public import Vit.Algorithm
meta import Vit.Algorithm
public import AlgorithmLib.Host.Clif
meta import AlgorithmLib.Host.Clif
public import AlgorithmLib.Host.ClifCheck
meta import AlgorithmLib.Host.ClifCheck
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The launches the emitted code makes

Recovering them reads the compiled function, so this is the one part of the
unit development that depends on the generator.
-/

namespace Vit
open AlgorithmLib AlgorithmLib.ML

/-- What one launch is, as much of it as a scan of the emitted code can see. -/
def vKeyOf (r : AlgorithmLib.Clif.LaunchRec) : VLaunchKey :=
  (r.fnName, r.kernelOff, r.nBufs, r.gridX, r.blockX)

/-- **The launches each capture makes**, recovered from the emitted instruction
    stream.  Named so that the several things decided about them share one
    walk.

    The bodies are compiled against `env`, the table the artifact ships them
    with; against a different table the same body declares different callees and
    the recovered stream would not be the shipped one.  Well-formedness is not
    re-decided here — `vBodies_wf` establishes it for every body the artifact
    carries, these two among them. -/
def vFwdLaunches : List AlgorithmLib.Clif.LaunchRec :=
  AlgorithmLib.Clif.launchesOf
    (AlgorithmLib.Prog.stateOf 1 (vCaptureDagAt 0 VFWD_N VGRAPH_DFWD_OFF))

def vStepLaunches : List AlgorithmLib.Clif.LaunchRec :=
  AlgorithmLib.Clif.launchesOf
    (AlgorithmLib.Prog.stateOf 1 (vCaptureDagAt VFWD_N VSTEP_N VGRAPH_DSTEP_OFF))

/-- **Is this recovered call a contraction in the configuration a law covers?**

    `Law.cublasGemmIsSomeReassoc` is stated of `cl_cublas_sgemm_strided_batched`
    at a batch of one: zero strides, zero operand offsets, and the leading
    dimensions the shape implies.  Outside that the operand slices come from
    arguments this model recovers as constants but does not interpret, and
    nothing is stated.

    So the configuration is *checked on the emitted arguments* rather than
    assumed from the generator: `vGemmStepOn` passing `one32` is a fact about a
    definition, and this is a fact about the program that ships.  Anything that
    is not one of the two contraction symbols passes, having no configuration
    to be wrong about. -/
def vPlainGemm (r : AlgorithmLib.Clif.LaunchRec) : Bool :=
  let at1 (strides batch offs : List Nat) : Bool :=
    strides.all (fun i => r.args.getD i .opaque == .const 0)
      && batch.all (fun i => r.args.getD i .opaque == .const 1)
      && offs.all (fun i => r.args.getD i .opaque == .const 0)
  if r.fnName == "cl_cublas_sgemm_strided_batched_on_stream" then
    at1 [8, 10, 13] [14] [16, 17, 18, 19, 20, 21]
  else if r.fnName == "cl_cublas_sgemm_strided_batched" then
    at1 [8, 10, 13] [14] [15, 16, 17, 18, 19, 20]
  else true

/-- How many of a list of recovered launches are contractions. -/
def vGemmCount (rs : List AlgorithmLib.Clif.LaunchRec) : Nat :=
  (rs.filter (fun r => r.fnName == "cl_cublas_sgemm_strided_batched_on_stream"
                    || r.fnName == "cl_cublas_sgemm_strided_batched")).length

/-! ## The model the two launch streams are recovered through

    `vFwdLaunches` and `vStepLaunches` are `launchesOf` — a fold of
    `Clif.stepPure` over the compiled body — so every claim about them rests on
    that model agreeing with the machine.  `Check.sound_entry` says it does,
    under conditions on the body's text; here they are discharged on the bodies
    that ship.

    The forward body is one block of 23582 instructions, which is where the
    width condition would be expected to fail if it were going to: this host
    computes with `f32` throughout.  It does not, because the condition ranges
    only over the operands the model turns into a base — and those are
    addresses. -/

open AlgorithmLib.IR AlgorithmLib.Clif.Check in
def vFwdBody : AlgorithmLib.IR.FuncData :=
  AlgorithmLib.Prog.stateOf 1 (vCaptureDagAt 0 VFWD_N VGRAPH_DFWD_OFF)

open AlgorithmLib.IR AlgorithmLib.Clif.Check in
def vStepBody : AlgorithmLib.IR.FuncData :=
  AlgorithmLib.Prog.stateOf 1 (vCaptureDagAt VFWD_N VSTEP_N VGRAPH_DSTEP_OFF)

open AlgorithmLib.Clif.Check in
theorem vFwd_blocks_ty_ok : TyBlocksOk TyEnv.empty vFwdBody.blocks = true := by
  native_decide

open AlgorithmLib.Clif.Check in
theorem vStep_blocks_ty_ok : TyBlocksOk TyEnv.empty vStepBody.blocks = true := by
  native_decide

open AlgorithmLib.Clif.Check in
theorem vFwd_entry_ok : EntryOk vFwdBody = true := by native_decide

open AlgorithmLib.Clif.Check in
theorem vStep_entry_ok : EntryOk vStepBody = true := by native_decide

open AlgorithmLib.IR AlgorithmLib.Clif AlgorithmLib.Clif.Check in
/-- **The model's claims about the forward capture body are sound.** -/
theorem vFwd_entry_sound {env' : AlgorithmLib.HProg.FnEnv}
    {s : AlgorithmLib.HProg.Blocks.BSt}
    {r : AlgorithmLib.HProg.Blocks.BSt × AlgorithmLib.HProg.Blocks.Next}
    {w : AlgorithmLib.HProg.Sem.World}
    (hsz : s.vals.size = (entryParams vFwdBody).length)
    (hpar : TypesAgree (entryTys vFwdBody) s.vals)
    (hr : AlgorithmLib.HProg.Blocks.runInsts env' s (entryInsts vFwdBody) = .ok r w) :
    Sound (tyRun (entryTys vFwdBody) (entryInsts vFwdBody)) r.1.vals
      (evalPure Env.empty (entryInsts vFwdBody)) :=
  sound_entry vFwd_entry_ok hsz hpar hr

open AlgorithmLib.IR AlgorithmLib.Clif.Check in
/-- …not for want of a caller: the entry parameters --- the arena base and the
    caller's two buffers with their lengths --- all declared `i64`. -/
theorem vFwd_entry_arg :
    TypesAgree (entryTys vFwdBody) #[AlgorithmLib.HProg.Sem.V.sc ClifTy.i64 0,
        AlgorithmLib.HProg.Sem.V.sc ClifTy.i64 0,
        AlgorithmLib.HProg.Sem.V.sc ClifTy.i64 0,
        AlgorithmLib.HProg.Sem.V.sc ClifTy.i64 0,
        AlgorithmLib.HProg.Sem.V.sc ClifTy.i64 0] :=
  typesAgree_of_check (by native_decide)

open AlgorithmLib.IR AlgorithmLib.Clif.Check in
/-- …and that state has the arity the entry block declares. -/
theorem vFwd_entry_size :
    (#[AlgorithmLib.HProg.Sem.V.sc ClifTy.i64 0,
        AlgorithmLib.HProg.Sem.V.sc ClifTy.i64 0,
        AlgorithmLib.HProg.Sem.V.sc ClifTy.i64 0,
        AlgorithmLib.HProg.Sem.V.sc ClifTy.i64 0,
        AlgorithmLib.HProg.Sem.V.sc ClifTy.i64 0] : Array _).size
      = (entryParams vFwdBody).length := by native_decide

end Vit

