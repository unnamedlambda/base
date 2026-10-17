import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate
import Host.Pilots

/-!
# The pilots never misuse a call

Each pilot artifact's entry, the histogram's in `PilotsHist` beside this, the term twin of a benchmark it is compared
against, proven by the condition generator from the typestate it starts in,
for any lengths of data and output the caller hands over. Every pilot's
entries also never fault, where the caller's data and
output hold the lengths it passes: every load and store finds the bytes it
reaches, and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace PilotsSafe

/-- The caller's data and output, of the lengths it passes. -/
abbrev Run (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .roomArg .data dl, .roomArg .out ol]

/-- The nested pilot's: the arena its artifact asks for, and the caller's data
    and output of the lengths it passes. -/
abbrev NestedStart (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .room .arena 0x100000, .roomArg .data dl, .roomArg .out ol]

abbrev RmsNormLoad (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena CudaRmsNormPersist.MEM_SIZE, .roomArg .data dl, .roomArg .out ol, .cstrIn (regionBase .arena + UInt64.ofNat CudaRmsNormPersist.PTX_SOURCE_OFF) (CudaRmsNormPersist.BIND_DESC_OFF - CudaRmsNormPersist.PTX_SOURCE_OFF)]

abbrev RmsNormRun (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena CudaRmsNormPersist.MEM_SIZE, .roomArg .data dl, .roomArg .out ol,
   .cell (regionBase .arena + UInt64.ofNat CudaRmsNormPersist.CTX_OFF) cudaCtx, .cstrIn (regionBase .arena + UInt64.ofNat CudaRmsNormPersist.PTX_SOURCE_OFF) (CudaRmsNormPersist.BIND_DESC_OFF - CudaRmsNormPersist.PTX_SOURCE_OFF)]

ship_entry clampSum : HProgPilots.ClampSum.code under Run dataLen outLen

ship_entry nested : HProgPilots.Nested.code under NestedStart dataLen outLen

ship_entry rmsNorm_load : HProgPilots.RmsNorm.loadCode under RmsNormLoad dataLen outLen

ship_entry rmsNorm_prep : HProgPilots.RmsNorm.prepCode under RmsNormRun dataLen outLen

ship_entry rmsNorm_infer : HProgPilots.RmsNorm.inferCode under RmsNormRun dataLen outLen

end PilotsSafe

#print axioms PilotsSafe.clampSum_no_misuse
#print axioms PilotsSafe.nested_no_misuse
#print axioms PilotsSafe.rmsNorm_load_no_misuse
#print axioms PilotsSafe.rmsNorm_prep_no_misuse
#print axioms PilotsSafe.rmsNorm_infer_no_misuse
