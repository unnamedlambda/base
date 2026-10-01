import AlgorithmLib.Proof.Typestate
import Warp.Gemv

/-!
# The warp GEMV at `wide` neither misuses a call nor faults

`main` uploads the matrix and the vector the caller hands over, back to back;
the launches, the cuBLAS baseline and `fetch` start from what it leaves: a live
context, the three buffers' ids where it stored them and their sizes
(`devBuf`), and the six kernels' text in the arena. The sizes are the shape's,
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
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena GemvWarp.MEM_SIZE, .room .data 134283264]

abbrev Run : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena GemvWarp.MEM_SIZE, .room .out 8192,
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx,
   .cell (regionBase .arena + UInt64.ofNat GemvWarp.A_ID) 0x100000000,
   .cell (regionBase .arena + UInt64.ofNat GemvWarp.Y_ID) 2,
   .devBuf 0 134217728, .devBuf 1 65536, .devBuf 2 8192,
   .cstr (regionBase .arena + UInt64.ofNat (GemvWarp.PTX_OFF + 0 * GemvWarp.SLOT)),
   .cstr (regionBase .arena + UInt64.ofNat (GemvWarp.PTX_OFF + 1 * GemvWarp.SLOT)),
   .cstr (regionBase .arena + UInt64.ofNat (GemvWarp.PTX_OFF + 2 * GemvWarp.SLOT)),
   .cstr (regionBase .arena + UInt64.ofNat (GemvWarp.PTX_OFF + 3 * GemvWarp.SLOT)),
   .cstr (regionBase .arena + UInt64.ofNat (GemvWarp.PTX_OFF + 4 * GemvWarp.SLOT)),
   .cstr (regionBase .arena + UInt64.ofNat (GemvWarp.PTX_OFF + 5 * GemvWarp.SLOT))]

theorem load_fine : Fine (emitGo (GemvWarp.loadCode sh : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Load w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.loadCode sh : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.loadCode sh : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry Load (runArgs dataLen outLen) (by prog_vc Load) load_fine rfl h m

theorem run_fine : Fine (emitGo (GemvWarp.runCode sh false .vec4 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem run_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.runCode sh false .vec4 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.runCode sh false .vec4 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry Run (runArgs dataLen outLen) (by prog_vc Run) run_fine rfl h m

theorem runStrided_fine : Fine (emitGo (GemvWarp.runCode sh false .strided : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runStrided_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.runCode sh false .strided : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.runCode sh false .strided : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry Run (runArgs dataLen outLen) (by prog_vc Run) runStrided_fine rfl h m

theorem runBlocked_fine : Fine (emitGo (GemvWarp.runCode sh false .blocked : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runBlocked_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.runCode sh false .blocked : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.runCode sh false .blocked : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry Run (runArgs dataLen outLen) (by prog_vc Run) runBlocked_fine rfl h m

theorem sq_fine : Fine (emitGo (GemvWarp.runCode sh true .vec4 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem sq_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.runCode sh true .vec4 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.runCode sh true .vec4 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry Run (runArgs dataLen outLen) (by prog_vc Run) sq_fine rfl h m

theorem sqStrided_fine : Fine (emitGo (GemvWarp.runCode sh true .strided : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem sqStrided_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.runCode sh true .strided : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.runCode sh true .strided : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry Run (runArgs dataLen outLen) (by prog_vc Run) sqStrided_fine rfl h m

theorem sqBlocked_fine : Fine (emitGo (GemvWarp.runCode sh true .blocked : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem sqBlocked_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.runCode sh true .blocked : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.runCode sh true .blocked : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry Run (runArgs dataLen outLen) (by prog_vc Run) sqBlocked_fine rfl h m

theorem fetch_fine : Fine (emitGo (GemvWarp.fetchCode sh : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetch_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.fetchCode sh : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.fetchCode sh : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry Run (runArgs dataLen outLen) (by prog_vc Run) fetch_fine rfl h m

theorem blas_fine : Fine (emitGo (GemvWarp.blasCode sh : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem blas_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.blasCode sh : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (GemvWarp.blasCode sh : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry Run (runArgs dataLen outLen) (by prog_vc Run) blas_fine rfl h m

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
