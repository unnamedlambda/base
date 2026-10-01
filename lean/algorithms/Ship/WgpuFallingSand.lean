import AlgorithmLib.Proof.Typestate
import Demo.FallingSand

/-!
# FallingSand's tests neither misuse a call nor fault

Each entry point, proven by the condition generator from the typestate it
starts in, for any lengths of data and output the caller hands over. A program
starts from its initial memory: the shader text it hands `gpuCreatePipeline`
is a NUL-terminated string inside the field the layout gives it (`cstrIn`),
which the calls before the pipeline leave alone. Buffer sizes need no facts:
an upload or download of the wrong size answers `-1`. The caller's data and
output hold the lengths it passes (`roomArg`); a test writes its results only
where the output has room for them.

Not faulting means every load and store finds the bytes it reaches, and every
operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuSafe

-- FallingSand: the headless tests

abbrev FallingSandTest (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.roomArg .data dl, .roomArg .out ol, .part .gpuAdapter true, .part .frozen false, .part .gpu false, .room .arena 2783568, .cstrIn (regionBase .arena + 64) 8192]

theorem FallingSand_layout : FallingSand.layoutMeta.totalSize = 2783568 ∧ FallingSand.f.stepSh.offset = 64 :=
  ⟨by decide +kernel, by decide +kernel⟩

theorem FallingSand_grainFalls_fine :
    Fine (emitGo (FallingSand.testGrainFalls : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem FallingSand_grainFalls_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : FallingSandTest dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (FallingSand.testGrainFalls : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (FallingSand.testGrainFalls : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (FallingSandTest dataLen outLen) (runArgs dataLen outLen) (by prog_vc (FallingSandTest dataLen outLen))
    FallingSand_grainFalls_fine rfl h m

theorem FallingSand_conservation_fine :
    Fine (emitGo (FallingSand.testConservation : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem FallingSand_conservation_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : FallingSandTest dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (FallingSand.testConservation : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (FallingSand.testConservation : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (FallingSandTest dataLen outLen) (runArgs dataLen outLen) (by prog_vc (FallingSandTest dataLen outLen))
    FallingSand_conservation_fine rfl h m

end WgpuSafe

#print axioms WgpuSafe.FallingSand_grainFalls_no_misuse
#print axioms WgpuSafe.FallingSand_conservation_no_misuse
