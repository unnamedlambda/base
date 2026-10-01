import AlgorithmLib.Proof.Typestate
import Host.StaticChecks
import Demo.ByteCount
import Demo.ByteScrub
import Bench.ClampSum
import Demo.SelfDescribing

/-!
# Shipped artifacts never misuse a call

Each entry point of a shipped artifact, proven by the condition generator from
the typestate it starts in. Most make no foreign call at all; for them the
proof is the generator walking the body and finding none.
Every entry here also never faults, where the caller's data and output hold
the lengths it passes: ByteCount and ByteScrub read their vectors and write
theirs only when both have room, ClampSum reads
whole vectors and elements below the data's length and writes its sum only
when the output has room, SelfDescribing copies its answer out only when the
output has room, and a histogram's slot is indexed by a byte, below 256.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace ShipSafe

theorem bytecount_fine : Fine (emitGo (ByteCount.code (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- The caller's data and output, of the lengths it passes. -/
abbrev Run (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .roomArg .data dl, .roomArg .out ol]

set_option maxRecDepth 20000 in
theorem bytecount_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit ByteCount.code) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit ByteCount.code) ≠ .fault m :=
  sound_of_wp_entry (Run dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Run dataLen outLen))
    bytecount_fine rfl h m

theorem bytescrub_fine : Fine (emitGo (ByteScrub.code (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
theorem bytescrub_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit ByteScrub.code) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit ByteScrub.code) ≠ .fault m :=
  sound_of_wp_entry (Run dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Run dataLen outLen))
    bytescrub_fine rfl h m

theorem clampsum_fine : Fine (emitGo (ClampSumBench.code (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
set_option maxHeartbeats 1000000 in
theorem clampsum_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit ClampSumBench.code) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit ClampSumBench.code) ≠ .fault m :=
  sound_of_wp_entry (Run dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Run dataLen outLen)) clampsum_fine
    rfl h m

/-- SelfDescribing's arena, and the caller's data and output of the lengths
    it passes. -/
abbrev SdRun (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .room .arena 4420, .roomArg .data dl, .roomArg .out ol]

theorem selfDescribing_size : SelfDescribing.layoutMeta.totalSize = 4420 := by native_decide

theorem selfDescribing_schema_fine :
    Fine (emitGo (SelfDescribing.schemaCode (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
theorem selfDescribing_schema_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.part .frozen false, .room .arena SelfDescribing.layoutMeta.totalSize,
      .roomArg .data dataLen, .roomArg .out outLen] w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emitAns (SelfDescribing.schemaCode (V := Slot) (L := Lvl))) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emitAns (SelfDescribing.schemaCode (V := Slot) (L := Lvl))) ≠ .fault m := by
  rw [selfDescribing_size] at h
  exact sound_of_wp_entry_ans (SdRun dataLen outLen) (runArgs dataLen outLen) (by prog_vc (SdRun dataLen outLen))
    selfDescribing_schema_fine rfl h m

theorem selfDescribing_stats_fine :
    Fine (emitGo (SelfDescribing.statsCode (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
theorem selfDescribing_stats_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.part .frozen false, .room .arena SelfDescribing.layoutMeta.totalSize,
      .roomArg .data dataLen, .roomArg .out outLen] w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emitAns (SelfDescribing.statsCode (V := Slot) (L := Lvl))) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emitAns (SelfDescribing.statsCode (V := Slot) (L := Lvl))) ≠ .fault m := by
  rw [selfDescribing_size] at h
  exact sound_of_wp_entry_ans (SdRun dataLen outLen) (runArgs dataLen outLen) (by prog_vc (SdRun dataLen outLen))
    selfDescribing_stats_fine rfl h m

theorem selfDescribing_bulk_fine :
    Fine (emitGo (SelfDescribing.bulkCode (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
theorem selfDescribing_bulk_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.part .frozen false, .room .arena SelfDescribing.layoutMeta.totalSize,
      .roomArg .data dataLen, .roomArg .out outLen] w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emitAns (SelfDescribing.bulkCode (V := Slot) (L := Lvl))) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emitAns (SelfDescribing.bulkCode (V := Slot) (L := Lvl))) ≠ .fault m := by
  rw [selfDescribing_size] at h
  exact sound_of_wp_entry_ans (SdRun dataLen outLen) (runArgs dataLen outLen) (by prog_vc (SdRun dataLen outLen))
    selfDescribing_bulk_fine rfl h m

end ShipSafe

#print axioms ShipSafe.selfDescribing_stats_no_misuse
#print axioms ShipSafe.clampsum_no_misuse
