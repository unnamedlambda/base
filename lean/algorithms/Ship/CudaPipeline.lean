import AlgorithmLib.Proof.Typestate
import Bench.CudaVecAddPersist
import Bench.CudaSaxpyPersist

/-!
# The elementwise CUDA benchmarks never misuse a call, nor fault

`cuda_vecadd_persist` and `cuda_saxpy_persist` are one expression each over two
inputs, compiled by `Expr.compileTo`: their entries are the same CLIF, and
only the kernel text in memory differs. `main` starts from memory alone; `prep`
and `infer` from what `main` leaves, a live context whose cell holds it and
every device access the default stream's. `main` reads the element count
only when the caller handed over its eight bytes, and `prep` uploads only
inputs the caller handed over in full: no load or store touches memory that is
not there.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CudaPipelineSafe

abbrev Load (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena (CudaPipeline.bindDescOff + 12 + 0x100), .roomArg .data dl, .roomArg .out ol,
   .cstrIn (regionBase .arena + UInt64.ofNat CudaPipeline.ptxSourceOff) (CudaPipeline.bindDescOff - CudaPipeline.ptxSourceOff)]

abbrev Run (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena (CudaPipeline.bindDescOff + 12 + 0x100), .roomArg .data dl, .roomArg .out ol,
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx, .cstrIn (regionBase .arena + UInt64.ofNat CudaPipeline.ptxSourceOff) (CudaPipeline.bindDescOff - CudaPipeline.ptxSourceOff)]

theorem load_fine : Fine (emitGo (CudaPipeline.loadCode 2 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Load dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaPipeline.loadCode 2 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
      Sem.run cfg (runArgs dataLen outLen) w (emit (CudaPipeline.loadCode 2 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Load dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Load dataLen outLen))
    load_fine rfl h m

theorem prep_fine : Fine (emitGo (CudaPipeline.prepCode 2 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem prep_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaPipeline.prepCode 2 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
      Sem.run cfg (runArgs dataLen outLen) w (emit (CudaPipeline.prepCode 2 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Run dataLen outLen))
    prep_fine rfl h m

theorem infer_fine : Fine (emitGo (CudaPipeline.inferCode (1 : Fin 2) 256 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem infer_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaPipeline.inferCode (1 : Fin 2) 256 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
      Sem.run cfg (runArgs dataLen outLen) w (emit (CudaPipeline.inferCode (1 : Fin 2) 256 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Run dataLen outLen))
    infer_fine rfl h m

end CudaPipelineSafe

#print axioms CudaPipelineSafe.load_no_misuse
#print axioms CudaPipelineSafe.prep_no_misuse
#print axioms CudaPipelineSafe.infer_no_misuse
