import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Bench.CudaVecAddPersist
import Bench.CudaSaxpyPersist

/-!
# The elementwise CUDA benchmarks never misuse a call, nor fault

`cuda_vecadd_persist` and `cuda_saxpy_persist` are one expression each over two
inputs, compiled by `Expr.compileTo`: their entry points are the same CLIF,
and only the kernel's PTX in memory differs. `main` starts from the driver
loaded and untouched (`Drv.App.loadTS`); `prep` and `infer` from what `main`
leaves for a length `n` (`Drv.App.runTSWith`): `n` in its cell, eight bytes
holding it and `8 * n` bytes for the two inputs. Each holds for any lengths
of data and output the caller hands over; where a copy's size is known only
at `n`'s value, what the copy needs of it is shown here.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CudaPipelineSafe

abbrev Load (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  (CudaPipeline.app.loadTS ++ [.oracles, .roomArg .data dl, .roomArg .out ol])

abbrev Run (dl ol n : UInt64) : World → Prop := Contracts.TState.holds
  (CudaPipeline.app.runTSWith [8, (n * 8).toNat] ++
    [.cell (Drv.App.at_ CudaPipeline.N_OFF) n, .roomArg .data dl, .roomArg .out ol])

ship_entry load : (CudaPipeline.loadCode 2) under Load dataLen outLen

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem prep_no_misuse (cfg : Cfg) (dataLen outLen n : UInt64) (w : World)
    (h : Run dataLen outLen n w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaPipeline.prepCode 2 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaPipeline.prepCode 2 : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body
  -- the inputs' bytes are the allocation's
  have hp : devPtrOf 4503668346847232 = some (1, 0) := by decide
  intro _
  simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp]

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem infer_no_misuse (cfg : Cfg) (dataLen outLen n : UInt64) (w : World)
    (h : Run dataLen outLen n w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaPipeline.inferCode (1 : Fin 2) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaPipeline.inferCode (1 : Fin 2) : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body
  -- the output's bytes, no more than one input's, lie in the second input
  intro hc
  simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
  have hp : devPtrOf (4503668346847232 + n * 4) = some (1, (n * 4).toNat) := by
    unfold devPtrOf
    have : (4503668346847232 + n * 4).toNat = 4503668346847232 + n.toNat * 4 := by
      simp [UInt64.toNat_add, UInt64.toNat_mul]; omega
    rw [this]
    simp only [UInt64.toNat_mul]
    split <;> simp_all <;> omega
  simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp]
  omega

end CudaPipelineSafe

#print axioms CudaPipelineSafe.load_no_misuse
#print axioms CudaPipelineSafe.prep_no_misuse
#print axioms CudaPipelineSafe.infer_no_misuse
