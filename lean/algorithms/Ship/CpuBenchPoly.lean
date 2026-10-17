import AlgorithmLib.Check.Entry
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

ship_entry poly : (do CpuBench.answer CpuBench.outAnswer (CpuBench.polySimd (← CpuBench.arrayWith 2 CpuBench.POLY_OFF))) under Run dataLen outLen

end CpuBenchSafe

#print axioms CpuBenchSafe.poly_no_misuse
