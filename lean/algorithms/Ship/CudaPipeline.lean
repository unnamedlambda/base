import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import AlgorithmLib.ML.Kernel.PipelineBody
import Bench.CudaVecAddPersist
import Bench.CudaSaxpyPersist

/-!
# The elementwise CUDA benchmarks compute their expressions, and never misuse a call, nor fault

`cuda_vecadd_persist` and `cuda_saxpy_persist` are one expression each over two
inputs, compiled by `ML.Pipeline.compile`: their entry points are the same CLIF,
and only the kernel's PTX in memory differs. `main` starts from the driver
loaded and untouched (`Drv.App.loadTS`); `prep` and `infer` from what `main`
leaves for a length `n` (`Drv.App.runTSWith`): `n` in its cell, the two inputs'
`8 * n` bytes and the result's `4 * n`, each with room for the blocks covering
`n`. Each holds for any lengths
of data and output the caller hands over; where a copy's size is known only
at `n`'s value, what the copy needs of it is shown here.

What each kernel stores is its expression's denotation (`vecadd_body`,
`saxpy_body`, instances of `ML.Pipeline.kernel_body`).

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CudaPipelineSafe

abbrev Load (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  ((ML.Pipeline.app 2).loadTS ++ [.oracles, .roomArg .data dl, .roomArg .out ol])

abbrev Run (dl ol n : UInt64) : World → Prop := Contracts.TState.holds
  ((ML.Pipeline.app 2).runTSWith [((n * 2 + 255) * 4).toNat, ((n + 255) * 4).toNat] ++
    [.cell (Drv.App.at_ ML.Pipeline.N_OFF) n, .roomArg .data dl, .roomArg .out ol])

ship_entry load : (ML.Pipeline.loadCode 2) under Load dataLen outLen

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem prep_no_misuse (cfg : Cfg) (dataLen outLen n : UInt64) (w : World)
    (h : Run dataLen outLen n w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (ML.Pipeline.prepCode 2 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (ML.Pipeline.prepCode 2 : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body
  -- the inputs' `8 * n` bytes fit their allocation of `4 * (2 * n + 255)`
  have hp : devPtrOf 412316860416 = some (0, 0) := by decide
  intro hc
  simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
  simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp]
  simp [UInt64.toNat_add, UInt64.toNat_mul] at *
  omega

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem infer_no_misuse (cfg : Cfg) (dataLen outLen n : UInt64) (w : World)
    (h : Run dataLen outLen n w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (ML.Pipeline.inferCode 2 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (ML.Pipeline.inferCode 2 : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body
  -- the output's bytes, no more than the result's `4 * n`, fit its allocation
  have hp : devPtrOf 481036337152 = some (1, 0) := by decide
  intro hc
  simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
  simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp]
  simp [UInt64.toNat_add, UInt64.toNat_mul] at *
  omega

-- What the kernels compute

open AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp in
/-- **`cuda_vecadd_persist`'s kernel adds**: `g` blocks leave each result word
    `j` below their lanes as the binary32 sum of the inputs' words `j`. -/
theorem vecadd_body (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (g : Nat)
    (hg : g * Pipeline.WARPS * 32 ≤ 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b ≤ 2 → 4 * (g * Pipeline.WARPS * 32) ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" 3 0 (Pipeline.kernel CudaVecAddPersist.spec).ew Pipeline.WARPS) g
      (32 * Pipeline.WARPS) (bufArgs 3) bufs bufs' hw) :
    ∀ j, j < g * Pipeline.WARPS * 32 → ({ kmem bufs with bufs := bufs' } : Mem).load (bufW 3 2 j) 4 =
      some (@denote F32 (F32.ops hw) 2
        (fun i => ⟨((kmem bufs).load (bufW 3 i.val j) 4).getD 0⟩) CudaVecAddPersist.spec).bits :=
  Pipeline.kernel_body CudaVecAddPersist.spec (by decide) hw hw32 g hg
    ⟨by decide +kernel, by decide +kernel⟩ bufs bufs' hsz hr

open AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp in
/-- **`cuda_saxpy_persist`'s kernel computes `2·x + y`**: `g` blocks leave
    each result word `j` below their lanes as the expression's binary32 value
    on the inputs' words `j`. -/
theorem saxpy_body (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (g : Nat)
    (hg : g * Pipeline.WARPS * 32 ≤ 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b ≤ 2 → 4 * (g * Pipeline.WARPS * 32) ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" 3 0 (Pipeline.kernel CudaSaxpyPersist.spec).ew Pipeline.WARPS) g
      (32 * Pipeline.WARPS) (bufArgs 3) bufs bufs' hw) :
    ∀ j, j < g * Pipeline.WARPS * 32 → ({ kmem bufs with bufs := bufs' } : Mem).load (bufW 3 2 j) 4 =
      some (@denote F32 (F32.ops hw) 2
        (fun i => ⟨((kmem bufs).load (bufW 3 i.val j) 4).getD 0⟩) CudaSaxpyPersist.spec).bits :=
  Pipeline.kernel_body CudaSaxpyPersist.spec (by decide) hw hw32 g hg
    ⟨by decide +kernel, by decide +kernel⟩ bufs bufs' hsz hr

end CudaPipelineSafe

#print axioms CudaPipelineSafe.load_no_misuse
#print axioms CudaPipelineSafe.prep_no_misuse
#print axioms CudaPipelineSafe.infer_no_misuse
#print axioms CudaPipelineSafe.vecadd_body
#print axioms CudaPipelineSafe.saxpy_body
