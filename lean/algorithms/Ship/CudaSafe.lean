import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate
import Host.StaticChecks
import Warp.Silu
import Warp.SumSq
import Warp.Mlp
import Warp.Grad

/-!
# The CUDA workloads neither misuse a call nor fault

Each entry point of each warp workload, proven by the condition generator from
the typestate it starts in, for any lengths of data and output the caller
hands over. `main` starts from memory alone: it brings the
context up, makes its buffers, uploads and writes the binding table. `run`
and `fetch` start from what `main` leaves: a live context whose cell holds it,
every device access the default stream's (`devSeq`), kernels and cuBLAS that
keep buffer sizes (`oracles`), and each kernel's text in the arena. None of
them needs to know which buffers `main` made: a call handed a buffer that does
not exist answers `-1`.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CudaSafe

-- SiluWarp

abbrev SiluWarpLoad : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena SiluWarp.MEM_SIZE, .room .data (SiluWarp.N * 4)]

abbrev SiluWarpRun : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena SiluWarp.MEM_SIZE, .room .out (SiluWarp.N * 4),
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx, .cstr (regionBase .arena + UInt64.ofNat SiluWarp.PTX_OFF), .cstr (regionBase .arena + UInt64.ofNat SiluWarp.PTX_L_OFF)]

ship_entry SiluWarp_load : SiluWarp.loadFnCode under SiluWarpLoad

ship_entry SiluWarp_run : SiluWarp.runFnCode under SiluWarpRun

ship_entry SiluWarp_runLoop : SiluWarp.runLoopFnCode under SiluWarpRun

ship_entry SiluWarp_fetch : SiluWarp.fetchFnCode under SiluWarpRun

-- WarpSumSq

abbrev WarpSumSqLoad : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena WarpSumSq.MEM_SIZE, .room .data (WarpSumSq.N * 4)]

abbrev WarpSumSqRun : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena WarpSumSq.MEM_SIZE, .room .out (WarpSumSq.GRID * 4),
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx, .cstr (regionBase .arena + UInt64.ofNat WarpSumSq.PTX_OFF)]

ship_entry WarpSumSq_load : WarpSumSq.loadFnCode under WarpSumSqLoad

ship_entry WarpSumSq_run : WarpSumSq.runFnCode under WarpSumSqRun

ship_entry WarpSumSq_fetch : WarpSumSq.fetchFnCode under WarpSumSqRun

-- MlpWarp

abbrev MlpWarpLoad : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena MlpWarp.MEM_SIZE, .room .data (MlpWarp.NIN * 4)]

abbrev MlpWarpRun : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena MlpWarp.MEM_SIZE, .room .out (MlpWarp.NOUT * 4),
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx, .cstr (regionBase .arena + UInt64.ofNat MlpWarp.PTX_OFF)]

ship_entry MlpWarp_load : MlpWarp.loadFnCode under MlpWarpLoad

ship_entry MlpWarp_run : MlpWarp.runFnCode under MlpWarpRun

ship_entry MlpWarp_fetch : MlpWarp.fetchFnCode under MlpWarpRun

-- GradWarp

abbrev GradWarpLoad : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena GradWarp.MEM_SIZE, .room .data (GradWarp.NIN * 4)]

abbrev GradWarpRun : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena GradWarp.MEM_SIZE, .room .out (GradWarp.NOUT * 4),
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx, .cstr (regionBase .arena + UInt64.ofNat GradWarp.PTX_OFF), .cstr (regionBase .arena + UInt64.ofNat GradWarp.PTX_D_OFF)]

ship_entry GradWarp_load : GradWarp.loadFnCode under GradWarpLoad

ship_entry GradWarp_run : GradWarp.runFnCode under GradWarpRun

ship_entry GradWarp_runD : GradWarp.runDFnCode under GradWarpRun

ship_entry GradWarp_fetch : GradWarp.fetchFnCode under GradWarpRun

end CudaSafe
