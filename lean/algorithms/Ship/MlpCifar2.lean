import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Warp.MlpCifar
import AlgorithmLib.ML.Launch.AppTrace

/-!
# The CIFAR-10 MLP neither misuses a call nor faults, part 2

`main` brings the CUDA driver up, makes the thirteen allocations, loads the
ten kernels, writes their parameters, makes cuBLAS's handle and uploads both
weight matrices from the caller's data, in the driver's standard layout
(`Drv.App`); every other entry starts from what it leaves (`Drv.App.runTS`):
the context current, every allocation live with its pointer in its slot,
every function looked up and its parameters written, the cuBLAS handle on
the default stream. An upload asks for the bytes it reads from the caller's
data, a fetch for those it writes to the caller's output.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace MlpCifarSafe2

abbrev Run (d o : Nat) : World → Prop := Contracts.TState.holds
  (MlpCifar.app.runTS ++ [.room .data d, .room .out o])

ship_entry fetchZ1 : MlpCifar.fetchCode MlpCifar.z1B 8192 under ((Run 0 8192))

ship_entry fetchDh : MlpCifar.fetchCode MlpCifar.dhB 8192 under ((Run 0 8192))

ship_entry fetchAdj : MlpCifar.fetchCode MlpCifar.adjB 8192 under ((Run 0 8192))

ship_entry fetchDw1 : MlpCifar.fetchCode MlpCifar.dw1B 3145728 under ((Run 0 3145728))

ship_entry fetchDw2 : MlpCifar.fetchCode MlpCifar.dw2B 32768 under ((Run 0 32768))

ship_entry fetchW1 : MlpCifar.fetchCode MlpCifar.w1B 3145728 under ((Run 0 3145728))

ship_entry fetchW2 : MlpCifar.fetchCode MlpCifar.w2B 32768 under ((Run 0 32768))

ship_entry runFwd : MlpCifar.runSeq MlpCifar.fwdSteps under ((Run 0 0))

ship_entry runBwd : MlpCifar.runSeq MlpCifar.bwdSteps under ((Run 0 0))

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- Every run of the shipped forward pass that ends launches the forward
    kernels in order on the default stream, then waits. -/
theorem runFwd_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runSeq MlpCifar.fwdSteps : Prog Slot Lvl Unit)) = .ok obs w') :
    (MlpCifar.app.runCalls MlpCifar.fwdSteps).Covers w obs := by
  check_calls

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- Every run of the shipped backward pass that ends launches the backward
    kernels in order on the default stream, then waits. -/
theorem runBwd_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runSeq MlpCifar.bwdSteps : Prog Slot Lvl Unit)) = .ok obs w') :
    (MlpCifar.app.runCalls MlpCifar.bwdSteps).Covers w obs := by
  check_calls

set_option maxRecDepth 100000 in
/-- **The launches every run of the shipped forward pass makes are the proven
    forward pipeline.** -/
theorem runFwd_realises (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runSeq MlpCifar.fwdSteps : Prog Slot Lvl Unit)) = .ok obs w') :
    ∃ cs, Check.callsOf obs = (Check.callsOf w.obs).reverse ++ cs ∧
      ML.pipelineOf? MlpCifar.mlpTable none (ML.appLaunchesOf MlpCifar.app cs) = some MlpCifar.fwdPipeline := by
  obtain ⟨cs, e, hc⟩ := runFwd_calls cfg dataLen outLen w h hr
  exact ⟨cs, e, by rw [ML.launchesOf_runCalls (by decide) hc]; exact MlpCifar.fwd_realises⟩

set_option maxRecDepth 100000 in
/-- **The launches every run of the shipped backward pass makes are the proven
    backward pipeline.** -/
theorem runBwd_realises (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runSeq MlpCifar.bwdSteps : Prog Slot Lvl Unit)) = .ok obs w') :
    ∃ cs, Check.callsOf obs = (Check.callsOf w.obs).reverse ++ cs ∧
      ML.pipelineOf? MlpCifar.mlpTable none (ML.appLaunchesOf MlpCifar.app cs) = some MlpCifar.bwdPipeline := by
  obtain ⟨cs, e, hc⟩ := runBwd_calls cfg dataLen outLen w h hr
  exact ⟨cs, e, by rw [ML.launchesOf_runCalls (by decide) hc]; exact MlpCifar.bwd_realises⟩

ship_entry runSoftmax : MlpCifar.launchSlot 3 under ((Run 0 0))

ship_entry uploadBias : MlpCifar.uploadCode MlpCifar.biasB (MlpCifar.C * 4) under ((Run 128 0))

ship_entry fetchDlog : MlpCifar.fetchCode MlpCifar.dlogB (MlpCifar.B * MlpCifar.C * 4) under ((Run 0 1024))

ship_entry runFwdBlas : MlpCifar.runMixed MlpCifar.fwdBlasSteps under ((Run 0 0))

ship_entry runBwdBlas : MlpCifar.runMixed MlpCifar.bwdBlasSteps under ((Run 0 0))

end MlpCifarSafe2

#print axioms MlpCifarSafe2.fetchZ1_no_misuse
#print axioms MlpCifarSafe2.fetchDh_no_misuse
#print axioms MlpCifarSafe2.fetchAdj_no_misuse
#print axioms MlpCifarSafe2.fetchDw1_no_misuse
#print axioms MlpCifarSafe2.fetchDw2_no_misuse
#print axioms MlpCifarSafe2.fetchW1_no_misuse
#print axioms MlpCifarSafe2.fetchW2_no_misuse
#print axioms MlpCifarSafe2.runFwd_no_misuse
#print axioms MlpCifarSafe2.runBwd_no_misuse
#print axioms MlpCifarSafe2.runSoftmax_no_misuse
#print axioms MlpCifarSafe2.uploadBias_no_misuse
#print axioms MlpCifarSafe2.fetchDlog_no_misuse
#print axioms MlpCifarSafe2.runFwdBlas_no_misuse
#print axioms MlpCifarSafe2.runBwdBlas_no_misuse

#print axioms MlpCifarSafe2.runFwd_realises
#print axioms MlpCifarSafe2.runBwd_realises
