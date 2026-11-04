import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import AlgorithmLib.Check.Entry
import Warp.BackwardWide
import AlgorithmLib.ML.Launch.AppTrace

/-!
# The wide backward workload neither misuses a call nor faults

Each entry point of the wide backward workload, proven by the condition generator from
the typestate it starts in, for any lengths of data and output the caller
hands over. The program calls the CUDA driver directly, in its standard
layout (`Drv.App`). `main` starts from the driver loaded and untouched, the
kernels' PTX in the arena with the signature each declares: it brings the
context up, makes the fourteen allocations, loads the twelve kernels, writes
their parameters and uploads the host's arrays. The other entry points start
from what `main` leaves (`Drv.App.runTS`): the context current, every
allocation live with its pointer in its slot, every function looked up, every
kernel's parameters written, every device access the default stream's
(`devSeq`), and kernels that keep allocation sizes (`oracles`).

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CudaSafe
-- BackwardWide

abbrev BackwardWideLoad : World → Prop := Contracts.TState.holds
  (BackwardWide.app.loadTS ++ [.room .data BackwardWide.HOST_BYTES])

abbrev BackwardWideRun : World → Prop := Contracts.TState.holds
  (BackwardWide.app.runTS ++ [.room .out (BackwardWide.N * BackwardWide.N * 4)])

ship_entry BackwardWide_load : BackwardWide.loadFn under BackwardWideLoad

ship_entry BackwardWide_run : BackwardWide.runFn under BackwardWideRun

ship_entry BackwardWide_fetch : BackwardWide.fetchFn under BackwardWideRun

ship_entry BackwardWide_runDw : BackwardWide.runDwFn under BackwardWideRun

ship_entry BackwardWide_fetchDw : BackwardWide.fetchDwFn under BackwardWideRun

ship_entry BackwardWide_runSiluBwd : BackwardWide.runSiluBwdFn under BackwardWideRun

ship_entry BackwardWide_runT : BackwardWide.runTFn under BackwardWideRun

ship_entry BackwardWide_runQ : BackwardWide.runQFn under BackwardWideRun

ship_entry BackwardWide_runS : BackwardWide.runSFn under BackwardWideRun

ship_entry BackwardWide_runDxr : BackwardWide.runDxrFn under BackwardWideRun

ship_entry BackwardWide_fetchDxr : BackwardWide.fetchDxrFn under BackwardWideRun

ship_entry BackwardWide_runFwd : BackwardWide.runFwdFn under BackwardWideRun

ship_entry BackwardWide_runY : BackwardWide.runYFn under BackwardWideRun

ship_entry BackwardWide_runDy : BackwardWide.runDyFn under BackwardWideRun

ship_entry BackwardWide_runSgd : BackwardWide.runSgdFn under BackwardWideRun

ship_entry BackwardWide_fetchY : BackwardWide.fetchYFn under BackwardWideRun

ship_entry BackwardWide_runAdj : BackwardWide.runAdjFn under BackwardWideRun

ship_entry BackwardWide_runBwdAll : BackwardWide.runBwdAllFn under BackwardWideRun

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- **Every run of the shipped backward pass that ends launches the backward
    kernels in pipeline order on the default stream, then waits**: the calls
    are read off the shipped body by the checker, not assumed. -/
theorem BackwardWide_runBwdAll_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runBwdAllFn : Prog Slot Lvl Unit)) = .ok obs w') :
    (BackwardWide.app.runCalls BackwardWide.bwdAllSeq).Covers w obs := by
  check_calls

/-- **The launches every run of the shipped backward pass makes are the proven
    three-stage pipeline** (`bwdPipelineFull`), so what such a run computes is
    that pipeline's denotation (`bwdAll_host_computes`). -/
theorem BackwardWide_runBwdAll_realises (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : BackwardWideRun w) {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runBwdAllFn : Prog Slot Lvl Unit)) = .ok obs w') :
    ∃ cs, Check.callsOf obs = (Check.callsOf w.obs).reverse ++ cs ∧
      ML.pipelineOf? BackwardWide.bwdAllTable none (ML.appLaunchesOf BackwardWide.app cs) =
        some BackwardWide.bwdPipelineFull := by
  obtain ⟨cs, e, hc⟩ := BackwardWide_runBwdAll_calls cfg dataLen outLen w h hr
  exact ⟨cs, e, by rw [ML.launchesOf_runCalls (by decide) hc]; exact BackwardWide.bwdAll_realises⟩

end CudaSafe

#print axioms CudaSafe.BackwardWide_runBwdAll_realises
