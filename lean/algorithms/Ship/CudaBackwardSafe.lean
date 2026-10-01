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

theorem BackwardWide_load_fine : Fine (emitGo (BackwardWide.loadFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideLoad w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.loadFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.loadFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideLoad (runArgs dataLen outLen) (by prog_vc BackwardWideLoad) BackwardWide_load_fine rfl h m

theorem BackwardWide_run_fine : Fine (emitGo (BackwardWide.runFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_run_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_run_fine rfl h m

theorem BackwardWide_fetch_fine : Fine (emitGo (BackwardWide.fetchFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_fetch_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.fetchFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.fetchFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_fetch_fine rfl h m

theorem BackwardWide_runDw_fine : Fine (emitGo (BackwardWide.runDwFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_runDw_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runDwFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runDwFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_runDw_fine rfl h m

theorem BackwardWide_fetchDw_fine : Fine (emitGo (BackwardWide.fetchDwFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_fetchDw_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.fetchDwFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.fetchDwFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_fetchDw_fine rfl h m

theorem BackwardWide_runSiluBwd_fine : Fine (emitGo (BackwardWide.runSiluBwdFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_runSiluBwd_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runSiluBwdFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runSiluBwdFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_runSiluBwd_fine rfl h m

theorem BackwardWide_runT_fine : Fine (emitGo (BackwardWide.runTFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_runT_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runTFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runTFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_runT_fine rfl h m

theorem BackwardWide_runQ_fine : Fine (emitGo (BackwardWide.runQFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_runQ_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runQFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runQFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_runQ_fine rfl h m

theorem BackwardWide_runS_fine : Fine (emitGo (BackwardWide.runSFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_runS_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runSFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runSFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_runS_fine rfl h m

theorem BackwardWide_runDxr_fine : Fine (emitGo (BackwardWide.runDxrFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_runDxr_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runDxrFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runDxrFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_runDxr_fine rfl h m

theorem BackwardWide_fetchDxr_fine : Fine (emitGo (BackwardWide.fetchDxrFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_fetchDxr_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.fetchDxrFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.fetchDxrFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_fetchDxr_fine rfl h m

theorem BackwardWide_runFwd_fine : Fine (emitGo (BackwardWide.runFwdFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_runFwd_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runFwdFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runFwdFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_runFwd_fine rfl h m

theorem BackwardWide_runY_fine : Fine (emitGo (BackwardWide.runYFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_runY_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runYFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runYFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_runY_fine rfl h m

theorem BackwardWide_runDy_fine : Fine (emitGo (BackwardWide.runDyFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_runDy_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runDyFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runDyFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_runDy_fine rfl h m

theorem BackwardWide_runSgd_fine : Fine (emitGo (BackwardWide.runSgdFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_runSgd_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runSgdFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runSgdFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_runSgd_fine rfl h m

theorem BackwardWide_fetchY_fine : Fine (emitGo (BackwardWide.fetchYFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_fetchY_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.fetchYFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.fetchYFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_fetchY_fine rfl h m

theorem BackwardWide_runAdj_fine : Fine (emitGo (BackwardWide.runAdjFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_runAdj_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runAdjFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runAdjFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_runAdj_fine rfl h m

theorem BackwardWide_runBwdAll_fine : Fine (emitGo (BackwardWide.runBwdAllFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem BackwardWide_runBwdAll_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BackwardWideRun w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runBwdAllFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BackwardWide.runBwdAllFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BackwardWideRun (runArgs dataLen outLen) (by prog_vc BackwardWideRun) BackwardWide_runBwdAll_fine rfl h m

end CudaSafe
