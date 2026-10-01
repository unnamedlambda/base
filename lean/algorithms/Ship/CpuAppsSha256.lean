import AlgorithmLib.Proof.Typestate
import Demo.Sha256

/-!
# SHA-256 neither misuses a call nor faults

Its entry, where the caller's data and output hold the lengths it passes: it
reads its file into the arena, pads it, hashes each block and writes the
digest as hex, every load and store within the arena. Proven in a file of its
own, beside the other file-to-file apps, so the two build side by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CpuAppsSafe

/-- The arena's size as a number, for the arithmetic on addresses in it. -/
theorem sha256_size : Sha256.layoutMeta.totalSize = 4202624 := by decide +kernel

abbrev Sha256Main (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .room .arena 4202624, .roomArg .data dl, .roomArg .out ol]

theorem Sha256_fine : Fine (emitGo (Sha256.mainCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1500000 in
theorem Sha256_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.part .frozen false, .room .arena Sha256.layoutMeta.totalSize,
      .roomArg .data dataLen, .roomArg .out outLen] w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Sha256.mainCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Sha256.mainCode : Prog Slot Lvl Unit)) ≠ .fault m := by
  rw [sha256_size] at h
  exact sound_of_wp_entry (Sha256Main dataLen outLen) (runArgs dataLen outLen)
    (by prog_vc (Sha256Main dataLen outLen)) Sha256_fine rfl h m

end CpuAppsSafe

#print axioms CpuAppsSafe.Sha256_no_misuse
