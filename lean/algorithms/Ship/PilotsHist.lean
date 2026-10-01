import AlgorithmLib.Proof.Typestate
import Host.Pilots

/-!
# The histogram pilot neither misuses a call nor faults

Its entry, where the caller's data and output hold the lengths it passes: it
copies its file names from within the data handed over, reads its file into
the arena, and counts it there. Proven in a file of its own, beside the other
pilots, so the two build side by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace PilotsSafe

/-- The histogram's: its arena, where it copies its file names and reads. -/
abbrev HistStart (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .room .arena HistogramBench1.MEM_SIZE, .roomArg .data dl, .roomArg .out ol]

theorem hist_fine : Fine (emitGo (HProgPilots.Hist.code : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem hist_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : HistStart dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (HProgPilots.Hist.code : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (HProgPilots.Hist.code : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (HistStart dataLen outLen) (runArgs dataLen outLen) (by prog_vc (HistStart dataLen outLen))
    hist_fine rfl h m

end PilotsSafe

#print axioms PilotsSafe.hist_no_misuse
