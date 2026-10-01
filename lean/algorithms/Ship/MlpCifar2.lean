import AlgorithmLib.Proof.Typestate
import Warp.MlpCifar

/-!
# The CIFAR-10 MLP neither misuses a call nor faults, part 2

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

namespace MlpCifarSafe2

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

theorem fetchZ1_fine : Fine (emitGo (MlpCifar.fetchCode MlpCifar.z1B 8192 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetchZ1_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 8192) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.z1B 8192 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.z1B 8192 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 8192) (runArgs dataLen outLen) (by prog_vc (Run 0 8192)) fetchZ1_fine rfl h m

theorem fetchDh_fine : Fine (emitGo (MlpCifar.fetchCode MlpCifar.dhB 8192 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetchDh_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 8192) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.dhB 8192 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.dhB 8192 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 8192) (runArgs dataLen outLen) (by prog_vc (Run 0 8192)) fetchDh_fine rfl h m

theorem fetchAdj_fine : Fine (emitGo (MlpCifar.fetchCode MlpCifar.adjB 8192 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetchAdj_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 8192) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.adjB 8192 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.adjB 8192 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 8192) (runArgs dataLen outLen) (by prog_vc (Run 0 8192)) fetchAdj_fine rfl h m

theorem fetchDw1_fine : Fine (emitGo (MlpCifar.fetchCode MlpCifar.dw1B 3145728 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetchDw1_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 3145728) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.dw1B 3145728 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.dw1B 3145728 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 3145728) (runArgs dataLen outLen) (by prog_vc (Run 0 3145728)) fetchDw1_fine rfl h m

theorem fetchDw2_fine : Fine (emitGo (MlpCifar.fetchCode MlpCifar.dw2B 32768 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetchDw2_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 32768) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.dw2B 32768 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.dw2B 32768 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 32768) (runArgs dataLen outLen) (by prog_vc (Run 0 32768)) fetchDw2_fine rfl h m

theorem fetchW1_fine : Fine (emitGo (MlpCifar.fetchCode MlpCifar.w1B 3145728 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetchW1_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 3145728) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.w1B 3145728 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.w1B 3145728 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 3145728) (runArgs dataLen outLen) (by prog_vc (Run 0 3145728)) fetchW1_fine rfl h m

theorem fetchW2_fine : Fine (emitGo (MlpCifar.fetchCode MlpCifar.w2B 32768 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetchW2_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 32768) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.w2B 32768 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.w2B 32768 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 32768) (runArgs dataLen outLen) (by prog_vc (Run 0 32768)) fetchW2_fine rfl h m

theorem runFwd_fine : Fine (emitGo (MlpCifar.runSeq MlpCifar.fwdSteps : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runFwd_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runSeq MlpCifar.fwdSteps : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runSeq MlpCifar.fwdSteps : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runFwd_fine rfl h m

theorem runBwd_fine : Fine (emitGo (MlpCifar.runSeq MlpCifar.bwdSteps : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runBwd_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runSeq MlpCifar.bwdSteps : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runSeq MlpCifar.bwdSteps : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runBwd_fine rfl h m

theorem runSoftmax_fine : Fine (emitGo (MlpCifar.launchSlot 3 MlpCifar.B : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runSoftmax_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 3 MlpCifar.B : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.launchSlot 3 MlpCifar.B : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runSoftmax_fine rfl h m

theorem uploadBias_fine : Fine (emitGo (MlpCifar.uploadCode MlpCifar.biasB (MlpCifar.C * 4) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem uploadBias_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 128 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.uploadCode MlpCifar.biasB (MlpCifar.C * 4) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.uploadCode MlpCifar.biasB (MlpCifar.C * 4) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 128 0) (runArgs dataLen outLen) (by prog_vc (Run 128 0)) uploadBias_fine rfl h m

theorem fetchDlog_fine : Fine (emitGo (MlpCifar.fetchCode MlpCifar.dlogB (MlpCifar.B * MlpCifar.C * 4) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetchDlog_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 1024) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.dlogB (MlpCifar.B * MlpCifar.C * 4) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.fetchCode MlpCifar.dlogB (MlpCifar.B * MlpCifar.C * 4) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 1024) (runArgs dataLen outLen) (by prog_vc (Run 0 1024)) fetchDlog_fine rfl h m

theorem runFwdBlas_fine : Fine (emitGo (MlpCifar.runMixed MlpCifar.fwdBlasSteps : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runFwdBlas_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runMixed MlpCifar.fwdBlasSteps : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runMixed MlpCifar.fwdBlasSteps : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runFwdBlas_fine rfl h m

theorem runBwdBlas_fine : Fine (emitGo (MlpCifar.runMixed MlpCifar.bwdBlasSteps : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runBwdBlas_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runMixed MlpCifar.bwdBlasSteps : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runMixed MlpCifar.bwdBlasSteps : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runBwdBlas_fine rfl h m

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
