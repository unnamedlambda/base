import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Warp.Gemv

/-!
# The warp GEMV at `mid` neither misuses a call nor faults

`main` uploads the matrix and the vector the caller hands over, back to back;
the launches, the cuBLAS baseline and `fetch` start from what it leaves, in
the driver's terms (`Drv.App.runTS`): the context current, the three
allocations live with their pointers in their slots, the six functions looked
up with their parameters written, and cuBLAS's handle on the default stream.
The sizes are the shape's,
2048 columns by 4096 rows, written out as numbers.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace GemvWarpSafeMid

abbrev sh := GemvWarp.mid

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

end GemvWarpSafeMid

#print axioms GemvWarpSafeMid.load_no_misuse
#print axioms GemvWarpSafeMid.run_no_misuse
#print axioms GemvWarpSafeMid.runStrided_no_misuse
#print axioms GemvWarpSafeMid.runBlocked_no_misuse
#print axioms GemvWarpSafeMid.sq_no_misuse
#print axioms GemvWarpSafeMid.sqStrided_no_misuse
#print axioms GemvWarpSafeMid.sqBlocked_no_misuse
#print axioms GemvWarpSafeMid.fetch_no_misuse
#print axioms GemvWarpSafeMid.blas_no_misuse
