import VitUnits
import VitAlgorithm
import AlgorithmLib.Clif

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
    (AlgorithmLib.HProg.compileBody 1
      (vCaptureDagAt 0 VFWD_N VGRAPH_DFWD_OFF) env).asState

def vStepLaunches : List AlgorithmLib.Clif.LaunchRec :=
  AlgorithmLib.Clif.launchesOf
    (AlgorithmLib.HProg.compileBody 1
      (vCaptureDagAt VFWD_N VSTEP_N VGRAPH_DSTEP_OFF) env).asState

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

end Vit

