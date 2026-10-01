import AlgorithmLib.Proof.Typestate
import Host.Pilots

/-!
# The pilots never misuse a call

Each pilot artifact's entry, the histogram's in `PilotsHist` beside this, the term twin of a benchmark it is compared
against, proven by the condition generator from the typestate it starts in,
for any lengths of data and output the caller hands over. Every pilot's
entries also never fault, where the caller's data and
output hold the lengths it passes: every load and store finds the bytes it
reaches, and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace PilotsSafe

/-- The caller's data and output, of the lengths it passes. -/
abbrev Run (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .roomArg .data dl, .roomArg .out ol]

/-- The nested pilot's: the arena its artifact asks for, and the caller's data
    and output of the lengths it passes. -/
abbrev NestedStart (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .room .arena 0x100000, .roomArg .data dl, .roomArg .out ol]

abbrev RmsNormLoad (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena CudaRmsNormPersist.MEM_SIZE, .roomArg .data dl, .roomArg .out ol, .cstrIn (regionBase .arena + UInt64.ofNat CudaRmsNormPersist.PTX_SOURCE_OFF) (CudaRmsNormPersist.BIND_DESC_OFF - CudaRmsNormPersist.PTX_SOURCE_OFF)]

abbrev RmsNormRun (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena CudaRmsNormPersist.MEM_SIZE, .roomArg .data dl, .roomArg .out ol,
   .cell (regionBase .arena + UInt64.ofNat CudaRmsNormPersist.CTX_OFF) cudaCtx, .cstrIn (regionBase .arena + UInt64.ofNat CudaRmsNormPersist.PTX_SOURCE_OFF) (CudaRmsNormPersist.BIND_DESC_OFF - CudaRmsNormPersist.PTX_SOURCE_OFF)]

theorem clampSum_fine : Fine (emitGo (HProgPilots.ClampSum.code : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem clampSum_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (HProgPilots.ClampSum.code : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (HProgPilots.ClampSum.code : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Run dataLen outLen))
    clampSum_fine rfl h m

theorem nested_fine : Fine (emitGo (HProgPilots.Nested.code : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem nested_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : NestedStart dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (HProgPilots.Nested.code : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (HProgPilots.Nested.code : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (NestedStart dataLen outLen) (runArgs dataLen outLen) (by prog_vc (NestedStart dataLen outLen))
    nested_fine rfl h m

theorem rmsNorm_load_fine : Fine (emitGo (HProgPilots.RmsNorm.loadCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem rmsNorm_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : RmsNormLoad dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (HProgPilots.RmsNorm.loadCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (HProgPilots.RmsNorm.loadCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (RmsNormLoad dataLen outLen) (runArgs dataLen outLen) (by prog_vc (RmsNormLoad dataLen outLen)) rmsNorm_load_fine rfl h m

theorem rmsNorm_prep_fine : Fine (emitGo (HProgPilots.RmsNorm.prepCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem rmsNorm_prep_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : RmsNormRun dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (HProgPilots.RmsNorm.prepCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (HProgPilots.RmsNorm.prepCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (RmsNormRun dataLen outLen) (runArgs dataLen outLen) (by prog_vc (RmsNormRun dataLen outLen)) rmsNorm_prep_fine rfl h m

theorem rmsNorm_infer_fine : Fine (emitGo (HProgPilots.RmsNorm.inferCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem rmsNorm_infer_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : RmsNormRun dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (HProgPilots.RmsNorm.inferCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (HProgPilots.RmsNorm.inferCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (RmsNormRun dataLen outLen) (runArgs dataLen outLen) (by prog_vc (RmsNormRun dataLen outLen)) rmsNorm_infer_fine rfl h m

end PilotsSafe

#print axioms PilotsSafe.clampSum_no_misuse
#print axioms PilotsSafe.nested_no_misuse
#print axioms PilotsSafe.rmsNorm_load_no_misuse
#print axioms PilotsSafe.rmsNorm_prep_no_misuse
#print axioms PilotsSafe.rmsNorm_infer_no_misuse
