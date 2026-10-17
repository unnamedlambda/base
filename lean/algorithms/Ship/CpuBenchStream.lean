import AlgorithmLib.Check.Entry
import Ship.CpuBenchStart

/-!
# The CPU benchmark's copy neither misuses a call nor faults

Its `_clif` entry, and the CLIF fallback of its `_asm` entry, where the
caller's data and output hold the lengths it passes: 64 bytes a trip while
64 whole bytes remain, then a u64 at a time, each stored at the same place
past the output's start; its answer reads two of the copied words, none when
nothing was copied. Proven in a file of its own, beside `CpuBench`, so the
two build side by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CpuBenchSafe

ship_entry stream : (do CpuBench.answer CpuBench.outCopy (CpuBench.copySimd (← CpuBench.array 3))) under Run dataLen outLen

end CpuBenchSafe

#print axioms CpuBenchSafe.stream_no_misuse
