import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate
import Warp.Gemv

/-!
# The warp GEMV at `mid` neither misuses a call nor faults

`main` uploads the matrix and the vector the caller hands over, back to back;
the launches, the cuBLAS baseline and `fetch` start from what it leaves: a live
context, the three buffers' ids where it stored them and their sizes
(`devBuf`), and the six kernels' text in the arena. The sizes are the shape's,
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
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena GemvWarp.MEM_SIZE, .room .data 33562624]

abbrev Run : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena GemvWarp.MEM_SIZE, .room .out 16384,
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx,
   .cell (regionBase .arena + UInt64.ofNat GemvWarp.A_ID) 0x100000000,
   .cell (regionBase .arena + UInt64.ofNat GemvWarp.Y_ID) 2,
   .devBuf 0 33554432, .devBuf 1 8192, .devBuf 2 16384,
   .cstr (regionBase .arena + UInt64.ofNat (GemvWarp.PTX_OFF + 0 * GemvWarp.SLOT)),
   .cstr (regionBase .arena + UInt64.ofNat (GemvWarp.PTX_OFF + 1 * GemvWarp.SLOT)),
   .cstr (regionBase .arena + UInt64.ofNat (GemvWarp.PTX_OFF + 2 * GemvWarp.SLOT)),
   .cstr (regionBase .arena + UInt64.ofNat (GemvWarp.PTX_OFF + 3 * GemvWarp.SLOT)),
   .cstr (regionBase .arena + UInt64.ofNat (GemvWarp.PTX_OFF + 4 * GemvWarp.SLOT)),
   .cstr (regionBase .arena + UInt64.ofNat (GemvWarp.PTX_OFF + 5 * GemvWarp.SLOT))]

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
