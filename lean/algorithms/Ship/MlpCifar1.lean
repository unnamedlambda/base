import AlgorithmLib.Proof.Typestate
import Warp.MlpCifar

/-!
# The CIFAR-10 MLP neither misuses a call nor faults, part 1

`main` makes the thirteen buffers, storing each one's id in the binding
table, and uploads both weight matrices from the caller's data; every other
entry starts from what it leaves: a live context, the ids where it stored them,
each buffer's size (`devBuf`), and the ten kernels' text in their slots. An
upload asks for the bytes it reads from the caller's data, a fetch for those
it writes to the caller's output.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace MlpCifarSafe

abbrev Load : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena MlpCifar.MEM_SIZE, .room .data MlpCifar.HOST_BYTES]

abbrev Run (d o : Nat) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena MlpCifar.MEM_SIZE, .room .data d, .room .out o,
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 0)) 4294967296,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 2)) 12884901890,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 4)) 21474836484,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 6)) 30064771078,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 8)) 38654705672,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 10)) 47244640266,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 12)) 12,
   .devBuf 0 98304, .devBuf 1 3145728, .devBuf 2 32768, .devBuf 3 128, .devBuf 4 1024, .devBuf 5 8192, .devBuf 6 8192, .devBuf 7 1024, .devBuf 8 1024, .devBuf 9 32768, .devBuf 10 8192, .devBuf 11 8192, .devBuf 12 3145728,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 0)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 1)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 2)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 3)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 4)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 5)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 6)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 7)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 8)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 9)) MlpCifar.SLOT]

theorem main_fine : Fine (emitGo (MlpCifar.loadCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem main_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Load w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.loadCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.loadCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry Load (runArgs dataLen outLen) (by prog_vc Load) main_fine rfl h m

theorem uploadX_fine : Fine (emitGo (MlpCifar.uploadCode MlpCifar.xB (MlpCifar.B * MlpCifar.IN * 4) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem uploadX_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 98304 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.uploadCode MlpCifar.xB (MlpCifar.B * MlpCifar.IN * 4) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.uploadCode MlpCifar.xB (MlpCifar.B * MlpCifar.IN * 4) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 98304 0) (runArgs dataLen outLen) (by prog_vc (Run 98304 0)) uploadX_fine rfl h m

theorem uploadOneHot_fine : Fine (emitGo (MlpCifar.uploadCode MlpCifar.ohB (MlpCifar.B * MlpCifar.C * 4) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem uploadOneHot_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 1024 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.uploadCode MlpCifar.ohB (MlpCifar.B * MlpCifar.C * 4) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.uploadCode MlpCifar.ohB (MlpCifar.B * MlpCifar.C * 4) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 1024 0) (runArgs dataLen outLen) (by prog_vc (Run 1024 0)) uploadOneHot_fine rfl h m

theorem fetchLogits_fine : Fine (emitGo (MlpCifar.fetchCode MlpCifar.logB (MlpCifar.B * MlpCifar.C * 4) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetchLogits_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 1024) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.logB (MlpCifar.B * MlpCifar.C * 4) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.logB (MlpCifar.B * MlpCifar.C * 4) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 1024) (runArgs dataLen outLen) (by prog_vc (Run 0 1024)) fetchLogits_fine rfl h m

theorem runFwd1_fine : Fine (emitGo (MlpCifar.launchSlot 0 MlpCifar.GRID1 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runFwd1_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 0 MlpCifar.GRID1 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 0 MlpCifar.GRID1 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runFwd1_fine rfl h m

theorem runAct_fine : Fine (emitGo (MlpCifar.launchSlot 1 MlpCifar.GRIDH : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runAct_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 1 MlpCifar.GRIDH : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 1 MlpCifar.GRIDH : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runAct_fine rfl h m

theorem runFwd2_fine : Fine (emitGo (MlpCifar.launchSlot 2 MlpCifar.GRID2 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runFwd2_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 2 MlpCifar.GRID2 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 2 MlpCifar.GRID2 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runFwd2_fine rfl h m

theorem runDw2_fine : Fine (emitGo (MlpCifar.launchSlot 4 MlpCifar.GRID2 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runDw2_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 4 MlpCifar.GRID2 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 4 MlpCifar.GRID2 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runDw2_fine rfl h m

theorem runDh_fine : Fine (emitGo (MlpCifar.launchSlot 5 MlpCifar.GRIDDH : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runDh_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 5 MlpCifar.GRIDDH : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 5 MlpCifar.GRIDDH : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runDh_fine rfl h m

theorem runAdj_fine : Fine (emitGo (MlpCifar.launchSlot 6 MlpCifar.GRIDH : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runAdj_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 6 MlpCifar.GRIDH : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 6 MlpCifar.GRIDH : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runAdj_fine rfl h m

theorem runDw1_fine : Fine (emitGo (MlpCifar.launchSlot 7 MlpCifar.GRID1 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runDw1_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 7 MlpCifar.GRID1 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 7 MlpCifar.GRID1 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runDw1_fine rfl h m

theorem runSgd1_fine : Fine (emitGo (MlpCifar.launchSlot 8 MlpCifar.GRIDW1 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runSgd1_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 8 MlpCifar.GRIDW1 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 8 MlpCifar.GRIDW1 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runSgd1_fine rfl h m

theorem runSgd2_fine : Fine (emitGo (MlpCifar.launchSlot 9 MlpCifar.GRIDW2 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runSgd2_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 9 MlpCifar.GRIDW2 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 9 MlpCifar.GRIDW2 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runSgd2_fine rfl h m

theorem fetchH_fine : Fine (emitGo (MlpCifar.fetchCode MlpCifar.hB 8192 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetchH_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 8192) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.hB 8192 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.hB 8192 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 8192) (runArgs dataLen outLen) (by prog_vc (Run 0 8192)) fetchH_fine rfl h m

end MlpCifarSafe

#print axioms MlpCifarSafe.main_no_misuse
#print axioms MlpCifarSafe.uploadX_no_misuse
#print axioms MlpCifarSafe.uploadOneHot_no_misuse
#print axioms MlpCifarSafe.fetchLogits_no_misuse
#print axioms MlpCifarSafe.runFwd1_no_misuse
#print axioms MlpCifarSafe.runAct_no_misuse
#print axioms MlpCifarSafe.runFwd2_no_misuse
#print axioms MlpCifarSafe.runDw2_no_misuse
#print axioms MlpCifarSafe.runDh_no_misuse
#print axioms MlpCifarSafe.runAdj_no_misuse
#print axioms MlpCifarSafe.runDw1_no_misuse
#print axioms MlpCifarSafe.runSgd1_no_misuse
#print axioms MlpCifarSafe.runSgd2_no_misuse
#print axioms MlpCifarSafe.fetchH_no_misuse
