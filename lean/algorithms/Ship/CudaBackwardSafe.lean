import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate
import Host.StaticChecks
import Warp.BackwardWide

/-!
# The wide backward workload neither misuses a call nor faults

Each entry point of the wide backward workload, proven by the condition generator from
the typestate it starts in, for any lengths of data and output the caller
hands over. `main` starts from memory alone: it brings the
context up, makes its buffers, uploads and writes the binding table. `run`
and `fetch` start from what `main` leaves: a live context whose cell holds it,
every device access the default stream's (`devSeq`), kernels and cuBLAS that
keep buffer sizes (`oracles`), and each kernel's text in the arena, within its slot: the binding table the
entries rewrite lies past every slot, so the texts outlast it. None of
them needs to know which buffers `main` made: a call handed a buffer that does
not exist answers `-1`.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CudaSafe
-- BackwardWide

abbrev BackwardWideLoad : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena BackwardWide.MEM_SIZE, .room .data (6 * BackwardWide.N * 4 + BackwardWide.N * BackwardWide.N * 4)]

abbrev BackwardWideRun : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena BackwardWide.MEM_SIZE, .room .out (BackwardWide.N * BackwardWide.N * 4),
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx,
   .cstrIn (regionBase .arena + UInt64.ofNat BackwardWide.PTX_OFF) (BackwardWide.PTX_DW_OFF - BackwardWide.PTX_OFF),
   .cstrIn (regionBase .arena + UInt64.ofNat BackwardWide.PTX_DW_OFF) (BackwardWide.PTX_SB_OFF - BackwardWide.PTX_DW_OFF),
   .cstrIn (regionBase .arena + UInt64.ofNat BackwardWide.PTX_SB_OFF) (BackwardWide.PTX_T_OFF - BackwardWide.PTX_SB_OFF),
   .cstrIn (regionBase .arena + UInt64.ofNat BackwardWide.PTX_T_OFF) (BackwardWide.PTX_Q_OFF - BackwardWide.PTX_T_OFF),
   .cstrIn (regionBase .arena + UInt64.ofNat BackwardWide.PTX_Q_OFF) (BackwardWide.PTX_S_OFF - BackwardWide.PTX_Q_OFF),
   .cstrIn (regionBase .arena + UInt64.ofNat BackwardWide.PTX_S_OFF) (BackwardWide.PTX_DXR_OFF - BackwardWide.PTX_S_OFF),
   .cstrIn (regionBase .arena + UInt64.ofNat BackwardWide.PTX_DXR_OFF) (BackwardWide.PTX_FWD_OFF - BackwardWide.PTX_DXR_OFF),
   .cstrIn (regionBase .arena + UInt64.ofNat BackwardWide.PTX_FWD_OFF) (BackwardWide.PTX_Y_OFF - BackwardWide.PTX_FWD_OFF),
   .cstrIn (regionBase .arena + UInt64.ofNat BackwardWide.PTX_Y_OFF) (BackwardWide.PTX_DY_OFF - BackwardWide.PTX_Y_OFF),
   .cstrIn (regionBase .arena + UInt64.ofNat BackwardWide.PTX_DY_OFF) (BackwardWide.PTX_SGD_OFF - BackwardWide.PTX_DY_OFF),
   .cstrIn (regionBase .arena + UInt64.ofNat BackwardWide.PTX_SGD_OFF) (BackwardWide.PTX_ADJ_OFF - BackwardWide.PTX_SGD_OFF),
   .cstrIn (regionBase .arena + UInt64.ofNat BackwardWide.PTX_ADJ_OFF) (BackwardWide.BIND_OFF - BackwardWide.PTX_ADJ_OFF)]

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

end CudaSafe
