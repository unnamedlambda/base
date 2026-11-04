import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Warp.Gemv

/-!
# The warp GEMV at `wide` neither misuses a call nor faults

`main` uploads the matrix and the vector the caller hands over, back to back;
the launches, the cuBLAS baseline and `fetch` start from what it leaves, in
the driver's terms (`Drv.App.runTS`): the context current, the three
allocations live with their pointers in their slots, the six functions looked
up with their parameters written, and cuBLAS's handle on the default stream.
The sizes are the shape's,
16384 columns by 2048 rows, written out as numbers.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace GemvWarpSafeWide

abbrev sh := GemvWarp.wide

abbrev Load : World → Prop := Contracts.TState.holds
  ((GemvWarp.app sh).loadTS ++ [.room .data ((sh.m * sh.n + sh.n) * 4)])

abbrev Run : World → Prop := Contracts.TState.holds
  ((GemvWarp.app sh).runTS ++ [.room .out (sh.m * 4)])

ship_entry load : GemvWarp.loadCode sh under Load

ship_entry run : GemvWarp.runCode sh false .vec4 under Run

ship_entry runStrided : GemvWarp.runCode sh false .strided under Run

ship_entry runBlocked : GemvWarp.runCode sh false .blocked under Run

ship_entry sq : GemvWarp.runCode sh true .vec4 under Run

ship_entry sqStrided : GemvWarp.runCode sh true .strided under Run

ship_entry sqBlocked : GemvWarp.runCode sh true .blocked under Run

ship_entry fetch : GemvWarp.fetchCode sh under Run

ship_entry blas : GemvWarp.blasCode sh under Run

end GemvWarpSafeWide

#print axioms GemvWarpSafeWide.load_no_misuse
#print axioms GemvWarpSafeWide.run_no_misuse
#print axioms GemvWarpSafeWide.runStrided_no_misuse
#print axioms GemvWarpSafeWide.runBlocked_no_misuse
#print axioms GemvWarpSafeWide.sq_no_misuse
#print axioms GemvWarpSafeWide.sqStrided_no_misuse
#print axioms GemvWarpSafeWide.sqBlocked_no_misuse
#print axioms GemvWarpSafeWide.fetch_no_misuse
#print axioms GemvWarpSafeWide.blas_no_misuse
