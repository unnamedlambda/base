import AlgorithmLib.Check.Entry
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

ship_entry load : (CudaPipeline.loadCode 2) under Load dataLen outLen

ship_entry prep : (CudaPipeline.prepCode 2) under Run dataLen outLen

ship_entry infer : (CudaPipeline.inferCode (1 : Fin 2) 256) under Run dataLen outLen

end CudaPipelineSafe

#print axioms CudaPipelineSafe.load_no_misuse
#print axioms CudaPipelineSafe.prep_no_misuse
#print axioms CudaPipelineSafe.infer_no_misuse
