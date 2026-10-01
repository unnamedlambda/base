import AlgorithmLib.Proof.Typestate
import Demo.Sat

/-!
# The file-to-file CPU apps never misuse a call

Each app reads its input file into its own memory, computes, and writes its
answer file, for any lengths of data and output the caller hands over. The
start is memory not frozen and the arena the artifact asks for: the file
names are read from inside it, and every read and write stays in it — a read
that failed or ran past its region computes nothing, and an answer is written
only when it fits the region it is written from. The histogram's and SHA-256's are
proven in `CpuAppsHist` and `CpuAppsSha256` beside this.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CpuAppsSafe

-- Sat

abbrev SatMain : World → Prop := Contracts.TState.holds
  [.part .frozen false, .room .arena Sat.totalMemory]

theorem Sat_fine : Fine (emitGo (Sat.mainCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Sat_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : SatMain w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Sat.mainCode : Prog Slot Lvl Unit)) ≠ .misuse m :=
  safe_of_wp_entry SatMain (runArgs dataLen outLen) (by prog_vc SatMain) Sat_fine rfl h m

end CpuAppsSafe

#print axioms CpuAppsSafe.Sat_no_misuse
