module
public import AlgorithmLib.ML.Kernel.Pipeline
meta import AlgorithmLib.ML.Kernel.Pipeline
public import AlgorithmLib.ML.Kernel.BodyMap
meta import AlgorithmLib.ML.Kernel.BodyMap
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# What a pipeline's kernel computes

Apart from `Pipeline`, which generators import and so may not wait on proofs.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp

namespace AlgorithmLib.ML.Pipeline

/-- **A pipeline's kernel computes its expression**: `g` blocks of `WARPS`
    warps leave each word `j` of the result buffer `n`, below the lanes they
    run, as `e`'s binary32 denotation over words `j` of the inputs as the
    launch found them. `hfine` is decided per expression. -/
theorem kernel_body {n : Nat} (e : Expr n) (hn : n ≤ MAX_INPUTS) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (g : Nat) (hg : g * WARPS * 32 ≤ 2 ^ 32)
    (hfine : Fine (emitGo (fnBody (params (n + 1)) (Body.body (n + 1) (kernel e).ew WARPS))
      ⟨(params (n + 1)).length, 0, [], [], [], none⟩).2)
    (bufs bufs' : Nat → ByteArray) (hsz : ∀ b, b ≤ n → 4 * (g * WARPS * 32) ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" (n + 1) 0 (kernel e).ew WARPS) g (32 * WARPS)
      (bufArgs (n + 1)) bufs bufs' hw) :
    ∀ j, j < g * WARPS * 32 → ({ kmem bufs with bufs := bufs' } : Mem).load (bufW (n + 1) n j) 4 =
      some (@denote F32 (F32.ops hw) n
        (fun i => ⟨((kmem bufs).load (bufW (n + 1) i.val j) 4).getD 0⟩) e).bits :=
  BodyMap.mapKernel_body e (fun i => i.val) n "main" (n + 1) WARPS g (by decide) (by decide) hw hw32
    (fun i => Nat.lt_succ_of_lt i.isLt) (Nat.lt_succ_self n) (by unfold MAX_INPUTS at hn; omega) hg hfine
    bufs bufs' (fun i => hsz _ (Nat.le_of_lt i.isLt)) (hsz n (Nat.le_refl _)) hr

end AlgorithmLib.ML.Pipeline
