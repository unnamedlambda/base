import AlgorithmLib.Proof.Typestate
import Bench.Histogram1

/-!
# The histogram benchmark neither misuses a call nor faults

Its entry, where the caller's data and output hold the lengths it passes: it
copies its file names from within the data handed over, reads its file into
the arena, counts it there and writes the counts out. Proven in a file of its
own, beside the other file-to-file apps, so the two build side by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CpuAppsSafe

-- Histogram, the benchmark's

abbrev HistMain (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .room .arena HistogramBench1.MEM_SIZE, .roomArg .data dl, .roomArg .out ol]

theorem Hist_fine : Fine (emitGo (HistogramBench1.code : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Hist_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : HistMain dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (HistogramBench1.code : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (HistogramBench1.code : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (HistMain dataLen outLen) (runArgs dataLen outLen) (by prog_vc (HistMain dataLen outLen))
    Hist_fine rfl h m

end CpuAppsSafe

#print axioms CpuAppsSafe.Hist_no_misuse
