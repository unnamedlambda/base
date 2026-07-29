import Lean
import Std
import AlgorithmLib

/-!
  # A proven GEMV, versus cuBLAS

  Qwen2 decode is bottlenecked on `cl_cublas_sgemv`: every token streams the
  whole weight matrix, so the kernel is memory-bound and `M = 1`.  That is
  precisely the shape where cuBLAS is *not* the last word — llama.cpp, vLLM and
  the Marlin family all beat it there.

  This kernel is the proven warp reduction applied to a matrix row: one warp per
  output element, `ld.global.v4.f32` on both operands, five-round butterfly,
  lane-0 store.  Its spec is the two-level fold the hardware performs.
-/

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML

namespace GemvWarp

/-- Qwen2's hidden size; 896 = 7 * 128, so the v4 sweep divides exactly. -/
def N : Nat := 896
def K : Nat := N / 128        -- 7 v4-iterations per warp
def M : Nat := 4864           -- rows: the FFN gate/up projection

def aB : Buf := 0             -- weight matrix, row-major
def xB : Buf := 1             -- input vector
def yB : Buf := 2             -- output vector

/-- Address of this lane's quad in row `ctaid`: `ctaid*N + i*128 + lane*4`. -/
def aIx : IdxE :=
  .add (.add (.mul .loopI (.lit 128)) (.mul .laneId (.lit 4))) (.mul .ctaId (.lit N))
/-- Same offset into the (shared) input vector. -/
def xIx : IdxE := .add (.mul .loopI (.lit 128)) (.mul .laneId (.lit 4))

/-- **The kernel is an instance of the proven schema** — no new kernel code and
    no new proof, only a choice of buffers, addressing and trip count. -/
def kernel : EWStmt := warpDotV4 aB xB aIx xIx yB .ctaId K

/-- **The shipped kernel is a point on the tuning menu.**

    `Sched.vec4` at this addressing *is* this kernel — definitionally.  That is
    the check that `Sched` enumerates the schedules the stack actually uses,
    rather than a menu invented alongside them. -/
example : kernel = Sched.vec4.realize aB xB aIx xIx yB .ctaId N := rfl

def ptx : String := emitProvenKernelN "main" 3 0 kernel

/-! ### Seam guards (`A47` G2)

    The same checks the rest of the stack carries: buffers inside the binding
    table, every branch resolving, nothing unrenderable reaching the printer,
    and the kernel writing — and not reading — its output. -/
theorem gemv_guards :
    kernel.BufBelow 3
      ∧ kernel.StageEligibleB yB = true
      ∧ FlatTargetsOkB (flatKernel (expandEW kernel)) = true
      ∧ FlatPrintableB (flatKernel (expandEW kernel)) = true := by
  refine ⟨by decide, by decide, ?_, ?_⟩ <;> native_decide

/-- **Launch geometry**: `K·32·4 = N`, i.e. the seven v4-iterations of 32 lanes
    cover all 896 elements, and the grid has one block per output row. -/
theorem gemv_geometry : K * 128 = N ∧ M = 4864 := by decide

def PTX_OFF : Nat := 0x0100
def BIND_OFF : Nat := 0x1400
def A_ID : Nat := 0x0040
def X_ID : Nat := 0x0044
def Y_ID : Nat := 0x0048
def MEM_SIZE : Nat := 0x1500

def loadFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit cuda ptr
  let ctxPtr ← cudaCtxPtr ptr
  let aBytes ← iconst64 (M * N * 4)
  let xBytes ← iconst64 (N * 4)
  let yBytes ← iconst64 (M * 4)
  let aId ← cudaCreateBuffer cuda ptr aBytes
  storeI32 aId (← absAddr ptr A_ID)
  let xId ← cudaCreateBuffer cuda ptr xBytes
  storeI32 xId (← absAddr ptr X_ID)
  let yId ← cudaCreateBuffer cuda ptr yBytes
  storeI32 yId (← absAddr ptr Y_ID)
  let _ ← call cuda.fnUpload [ctxPtr, aId, dataPtr, aBytes]
  let xSrc ← iadd dataPtr aBytes
  let _ ← call cuda.fnUpload [ctxPtr, xId, xSrc, xBytes]
  storeI32 aId (← absAddr ptr BIND_OFF)
  storeI32 xId (← absAddr ptr (BIND_OFF + 4))
  storeI32 yId (← absAddr ptr (BIND_OFF + 8))
  ret

def runFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ptxOff ← iconst64 PTX_OFF
  let nBufs ← iconst32 3
  let bindOff ← iconst64 BIND_OFF
  let one ← iconst32 1
  let warp ← iconst32 32
  let grid ← iconst32 M
  let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff grid one one warp one one
  let _ ← cudaSync cuda ptr
  ret

/-- The cuBLAS baseline on the same buffers: `y = A·x`, `A` is `M x N`. -/
def blasFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let cublas ← declareCuBlasFFI
  let ctxPtr ← cudaCtxPtr ptr
  let aId ← load32 (← absAddr ptr A_ID)
  let xId ← load32 (← absAddr ptr X_ID)
  let yId ← load32 (← absAddr ptr Y_ID)
  -- row-major A (M x N) is column-major (N x M); trans=1 gives yᵢ = Σₖ A[i,k]·xₖ
  let trans ← iconst32 1
  let mm ← iconst32 N
  let nn ← iconst32 M
  let alpha ← iconst32 0x3F800000
  let beta ← iconst32 0
  let _ ← call cublas.fnSgemv [ctxPtr, trans, mm, nn, alpha, aId, xId, beta, yId]
  let _ ← cudaSync cuda ptr
  ret

def fetchFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let yId ← load32 (← absAddr ptr Y_ID)
  let yBytes ← iconst64 (M * 4)
  let _ ← call cuda.fnDownload [ctxPtr, yId, outPtr, yBytes]
  ret

def clifIR : String :=
  noopFunction ++ "\n" ++ buildFunction 1 loadFn ++ "\n" ++ buildFunction 2 runFn
    ++ "\n" ++ buildFunction 3 fetchFn ++ "\n" ++ buildFunction 4 blasFn

theorem ptx_fits_slot : ptx.toUTF8.toList.length + 1 ≤ BIND_OFF - PTX_OFF := by
  native_decide

theorem mem_map_ok :
    AlgorithmLib.Layout.RegionMap.okB
      [⟨"ptx", PTX_OFF, BIND_OFF - PTX_OFF⟩, ⟨"bind", BIND_OFF, 8⟩] = true := by
  decide

def initialMemory : List UInt8 :=
  let p := ptx.toUTF8.toList ++ [0]
  zeros PTX_OFF ++ p ++ zeros (MEM_SIZE - PTX_OFF - p.length)

def artifacts : Array Json :=
  #[ toJsonArtifact "gemv_warp"
      { cranelift_ir := clifIR, memory_size := MEM_SIZE, initial_memory := initialMemory }
      { fn_idx := u32 1 }
      [("run", { fn_idx := u32 2 }), ("fetch", { fn_idx := u32 3 }),
       ("blas", { fn_idx := u32 4 })] ]

end GemvWarp

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir GemvWarp.artifacts

namespace GemvWarp

/-- **GEMV computes its spec** — the entire proof is one application of the
    schema theorem.  This is what a reusable schema buys: the refinement
    argument was made once, in `Schema.lean`, and every instance inherits it. -/
theorem gemv_computes_spec (cta : Nat) (st : WSt) {Γ : Nat} (env : Fin Γ → Float32)
    (ae be : Nat → Expr Γ)
    (ha : ∀ i, denote env (ae i) = st.mem aB i)
    (hb : ∀ i, denote env (be i) = st.mem xB i) :
    ((kernel.elabIn cta).run st).mem yB cta
      = denote env (warpDotV4E ae be
          (fun i l => aIx.eval cta i l) (fun i l => xIx.eval cta i l) K) :=
  warpDotV4_implements aB xB aIx xIx yB .ctaId K cta st env ae be ha hb

/-- **And the emitted PTX runs it**, from instruction 0, over real branches,
    with no hypothesis. -/
theorem gemv_ptx_computes_spec (cta : Nat) (m : MState) {Γ : Nat} (env : Fin Γ → Float32)
    (ae be : Nat → Expr Γ)
    (ha : ∀ i, denote env (ae i) = m.mem aB i)
    (hb : ∀ i, denote env (be i) = m.mem xB i) :
    ∃ k m', steps cta (flatKernel (expandEW kernel)) k (0, m)
          = some ((flatKernel (expandEW kernel)).length, m')
      ∧ m'.mem yB cta
          = denote env (warpDotV4E ae be
              (fun i l => aIx.eval cta i l) (fun i l => xIx.eval cta i l) K) := by
  obtain ⟨k, m', hs, hw⟩ :=
    flatKernel_sound_idxFree cta (expandEW kernel) (expandEW_expFree kernel)
      (expandEW_idxFree kernel (by decide))
      (expandEW_flat kernel (by decide)) m
  refine ⟨k, m', hs, ?_⟩
  have hm : m'.mem yB cta = (((expandEW kernel).elabIn cta).run m.toWSt).mem yB cta :=
    congrArg (fun st => st.mem yB cta) hw
  have hid : (expandEW kernel).elabIn cta = kernel.elabIn cta := rfl
  rw [hm, hid]
  exact gemv_computes_spec cta m.toWSt env ae be ha hb

end GemvWarp
