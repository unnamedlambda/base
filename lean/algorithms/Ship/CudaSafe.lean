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

theorem SiluWarp_load_fine : Fine (emitGo (SiluWarp.loadFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem SiluWarp_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : SiluWarpLoad w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (SiluWarp.loadFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (SiluWarp.loadFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

theorem SiluWarp_run_fine : Fine (emitGo (SiluWarp.runFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem SiluWarp_run_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : SiluWarpRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (SiluWarp.runFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (SiluWarp.runFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

theorem SiluWarp_runLoop_fine : Fine (emitGo (SiluWarp.runLoopFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem SiluWarp_runLoop_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : SiluWarpRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (SiluWarp.runLoopFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (SiluWarp.runLoopFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

theorem SiluWarp_fetch_fine : Fine (emitGo (SiluWarp.fetchFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem SiluWarp_fetch_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : SiluWarpRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (SiluWarp.fetchFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (SiluWarp.fetchFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

-- WarpSumSq

abbrev WarpSumSqLoad : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena WarpSumSq.MEM_SIZE, .room .data (WarpSumSq.N * 4)]

abbrev WarpSumSqRun : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena WarpSumSq.MEM_SIZE, .room .out (WarpSumSq.GRID * 4),
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx, .cstr (regionBase .arena + UInt64.ofNat WarpSumSq.PTX_OFF)]

theorem WarpSumSq_load_fine : Fine (emitGo (WarpSumSq.loadFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem WarpSumSq_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : WarpSumSqLoad w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (WarpSumSq.loadFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (WarpSumSq.loadFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

theorem WarpSumSq_run_fine : Fine (emitGo (WarpSumSq.runFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem WarpSumSq_run_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : WarpSumSqRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (WarpSumSq.runFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (WarpSumSq.runFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

theorem WarpSumSq_fetch_fine : Fine (emitGo (WarpSumSq.fetchFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem WarpSumSq_fetch_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : WarpSumSqRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (WarpSumSq.fetchFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (WarpSumSq.fetchFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

-- MlpWarp

abbrev MlpWarpLoad : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena MlpWarp.MEM_SIZE, .room .data (MlpWarp.NIN * 4)]

abbrev MlpWarpRun : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena MlpWarp.MEM_SIZE, .room .out (MlpWarp.NOUT * 4),
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx, .cstr (regionBase .arena + UInt64.ofNat MlpWarp.PTX_OFF)]

theorem MlpWarp_load_fine : Fine (emitGo (MlpWarp.loadFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem MlpWarp_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : MlpWarpLoad w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpWarp.loadFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpWarp.loadFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

theorem MlpWarp_run_fine : Fine (emitGo (MlpWarp.runFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem MlpWarp_run_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : MlpWarpRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpWarp.runFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpWarp.runFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

theorem MlpWarp_fetch_fine : Fine (emitGo (MlpWarp.fetchFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem MlpWarp_fetch_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : MlpWarpRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpWarp.fetchFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpWarp.fetchFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

-- GradWarp

abbrev GradWarpLoad : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena GradWarp.MEM_SIZE, .room .data (GradWarp.NIN * 4)]

abbrev GradWarpRun : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena GradWarp.MEM_SIZE, .room .out (GradWarp.NOUT * 4),
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx, .cstr (regionBase .arena + UInt64.ofNat GradWarp.PTX_OFF), .cstr (regionBase .arena + UInt64.ofNat GradWarp.PTX_D_OFF)]

theorem GradWarp_load_fine : Fine (emitGo (GradWarp.loadFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem GradWarp_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : GradWarpLoad w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GradWarp.loadFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GradWarp.loadFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

theorem GradWarp_run_fine : Fine (emitGo (GradWarp.runFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem GradWarp_run_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : GradWarpRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GradWarp.runFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GradWarp.runFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

theorem GradWarp_runD_fine : Fine (emitGo (GradWarp.runDFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem GradWarp_runD_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : GradWarpRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GradWarp.runDFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GradWarp.runDFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

theorem GradWarp_fetch_fine : Fine (emitGo (GradWarp.fetchFnCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem GradWarp_fetch_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : GradWarpRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GradWarp.fetchFnCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GradWarp.fetchFnCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

end CudaSafe
