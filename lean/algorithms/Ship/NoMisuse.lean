import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate
import AlgorithmLib.Check.Entry
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

theorem bytecount_fine : Fine (emitGo (fnBody ptrParams (ByteCount.code (V := Slot) (L := Lvl))) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- The caller's data and output, of the lengths it passes. -/
abbrev Run (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.roomArg .data dl, .roomArg .out ol]

set_option maxRecDepth 20000 in
theorem bytecount_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit ByteCount.code) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit ByteCount.code) ≠ .fault m := by
  check_body

theorem bytescrub_fine : Fine (emitGo (fnBody ptrParams (ByteScrub.code (V := Slot) (L := Lvl))) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
theorem bytescrub_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit ByteScrub.code) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit ByteScrub.code) ≠ .fault m := by
  check_body

theorem clampsum_fine : Fine (emitGo (fnBody ptrParams (ClampSumBench.code (V := Slot) (L := Lvl))) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
set_option maxHeartbeats 1000000 in
theorem clampsum_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit ClampSumBench.code) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit ClampSumBench.code) ≠ .fault m := by
  check_body

/-- SelfDescribing's arena, and the caller's data and output of the lengths
    it passes. -/
abbrev SdRun (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.room .arena 4420, .roomArg .data dl, .roomArg .out ol]

theorem selfDescribing_size : SelfDescribing.layoutMeta.totalSize = 4420 := by native_decide

/-- The schema's length as a number: the compiled encoder computes it, since the
    kernel would encode the schema to check it. -/
theorem schemaBytes_length : SelfDescribing.schemaBytes.length = 186 := by native_decide

theorem selfDescribing_schema_fine :
    Fine (emitGo (fnBody ptrParams (SelfDescribing.schemaCode (V := Slot) (L := Lvl))) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
theorem selfDescribing_schema_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.room .arena SelfDescribing.layoutMeta.totalSize,
      .roomArg .data dataLen, .roomArg .out outLen] w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emitAns (SelfDescribing.schemaCode (V := Slot) (L := Lvl))) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emitAns (SelfDescribing.schemaCode (V := Slot) (L := Lvl))) ≠ .fault m := by
  rw [selfDescribing_size] at h
  unfold SelfDescribing.schemaCode
  simp only [schemaBytes_length]
  check_body

/-- Where the stats and bulk answers sit, and their pieces, as numbers. -/
theorem stats_offset : SelfDescribing.f.stats.offset = 250 := by native_decide
theorem bulk_offset : SelfDescribing.f.bulk.offset = 2364 := by native_decide
theorem statsCountAt_eq : SelfDescribing.statsCountAt = 30 := by native_decide
theorem statsSumAt_eq : SelfDescribing.statsSumAt = 43 := by native_decide
theorem statsHistAt_eq : SelfDescribing.statsHistAt = 66 := by native_decide
theorem statsSize_eq : SelfDescribing.statsSize = 2114 := by native_decide
theorem bulkSize_eq : SelfDescribing.bulkSize = 2056 := by native_decide
theorem bulkId_size : SelfDescribing.bulkId.utf8ByteSize = 8 := by native_decide

theorem selfDescribing_stats_fine :
    Fine (emitGo (fnBody ptrParams (SelfDescribing.statsCode (V := Slot) (L := Lvl))) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
theorem selfDescribing_stats_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.room .arena SelfDescribing.layoutMeta.totalSize,
      .roomArg .data dataLen, .roomArg .out outLen] w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emitAns (SelfDescribing.statsCode (V := Slot) (L := Lvl))) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emitAns (SelfDescribing.statsCode (V := Slot) (L := Lvl))) ≠ .fault m := by
  rw [selfDescribing_size] at h
  unfold SelfDescribing.statsCode
  simp only [fldAddr, stats_offset, statsCountAt_eq, statsSumAt_eq, statsHistAt_eq, statsSize_eq]
  check_body

theorem selfDescribing_bulk_fine :
    Fine (emitGo (fnBody ptrParams (SelfDescribing.bulkCode (V := Slot) (L := Lvl))) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
theorem selfDescribing_bulk_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.room .arena SelfDescribing.layoutMeta.totalSize,
      .roomArg .data dataLen, .roomArg .out outLen] w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emitAns (SelfDescribing.bulkCode (V := Slot) (L := Lvl))) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emitAns (SelfDescribing.bulkCode (V := Slot) (L := Lvl))) ≠ .fault m := by
  rw [selfDescribing_size] at h
  unfold SelfDescribing.bulkCode
  simp only [fldAddr, bulk_offset, bulkSize_eq, bulkId_size]
  check_body

end ShipSafe

#print axioms ShipSafe.selfDescribing_stats_no_misuse
#print axioms ShipSafe.clampsum_no_misuse
