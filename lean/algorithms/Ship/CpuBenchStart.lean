import AlgorithmLib.Proof.Typestate
import Bench.Cpu

/-!
# Where the CPU benchmark's entries start

The typestates its entries are proven from, shared by the files that prove
them so the files build side by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CpuBenchSafe

abbrev Any : World → Prop := Contracts.TState.holds [.part .frozen false]

/-- The arena, and the caller's data and output of the lengths it passes. -/
abbrev Run (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .room .arena 1536, .roomArg .data dl, .roomArg .out ol]

end CpuBenchSafe
