import Ship.CpuBenchStart

/-!
# The CPU benchmark's polynomial kernel neither misuses a call nor faults

Its `_clif` entry, and the CLIF fallback of its `_asm` entry, where the
caller's data and output hold the lengths it passes: four vectors a trip while
four whole ones remain, then one at a time while a whole one remains. Proven
in a file of its own, beside `CpuBench`, so the two build side by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CpuBenchSafe

theorem poly_fine : Fine (emitGo (do CpuBench.answer CpuBench.outAnswer (CpuBench.polySimd (← CpuBench.arrayWith 2 CpuBench.POLY_OFF)) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem poly_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (do CpuBench.answer CpuBench.outAnswer (CpuBench.polySimd (← CpuBench.arrayWith 2 CpuBench.POLY_OFF)) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (do CpuBench.answer CpuBench.outAnswer (CpuBench.polySimd (← CpuBench.arrayWith 2 CpuBench.POLY_OFF)) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Run dataLen outLen)) poly_fine rfl h m

end CpuBenchSafe

#print axioms CpuBenchSafe.poly_no_misuse
