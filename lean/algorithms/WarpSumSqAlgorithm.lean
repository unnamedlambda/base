import Lean
import Std
import AlgorithmLib

/-!
  # A proven warp kernel, wired into a real `Artifact`

  The kernel body is `AlgorithmLib.ML.warpSumSqV4Store`, which is proven to
  compute `denote env spec` exactly (`warpSumSqV4Store_implements`).  It is
  written in the emittable mirror `EWStmt`, whose expansion into the proven
  machine language is `rfl` (`eKernel_elab` below), and lowered to PTX by
  `AlgorithmLib.ML.emitProvenKernel`, whose instruction list is proven correct
  by `flatKernel_sound`.

  This file supplies the surrounding CLIF: allocate device buffers, upload the
  caller's input, launch one warp, sync, download the single-float result.

  What is proven: the kernel computes the spec.
  What is trusted: the `EWStmt → PTX` lowering (A4/A11) and this CLIF wrapper.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.ML

namespace WarpSumSq

/-- Elements consumed: one warp, `v4` loads, `K` iterations of 4*32 floats. -/
def K : Nat := 128                 -- v4 iterations per warp
def CHUNK : Nat := K * 4 * 32      -- floats per block = 16384
def GRID : Nat := 4096             -- blocks (one warp each)
def N : Nat := GRID * CHUNK        -- 67,108,864 floats = 256 MiB

def inBuf : Buf := 0
def outBuf : Buf := 1

-- ── the kernel, in the emittable language ───────────────────────────────────

def eAcc : EWStmt :=
  .seq (.setR 0 (.lit (NumOps.ofNat 0)))
       (.forN K (.seq (.loadV4 1 2 3 4 inBuf (strideIxV4Cta CHUNK))
                      (.setR 0 (.add (.add (.add (.add (.reg 0) (.mul (.reg 1) (.reg 1)))
                                                 (.mul (.reg 2) (.reg 2)))
                                           (.mul (.reg 3) (.reg 3)))
                                     (.mul (.reg 4) (.reg 4))))))

def eRound (m : Nat) : EWStmt :=
  .seq (.shflXor 1 0 m) (.setR 0 (.add (.reg 0) (.reg 1)))

def eKernel : EWStmt :=
  .seq (.seq eAcc
             (.seq (eRound 16) (.seq (eRound 8) (.seq (eRound 4)
               (.seq (eRound 2) (eRound 1))))))
       (.storeLane0 outBuf .ctaId 0)

/-- **The emitted kernel is the proven kernel — for every block.**
    Machine-checked, by `rfl`, uniformly in the block index. -/
theorem eKernel_elab (cta : Nat) :
    eKernel.elabIn cta = warpSumSqV4Store (cta * CHUNK) inBuf outBuf cta K := rfl

def ptx : String := emitProvenKernel "main" eKernel

/-! ### Seam guards (`A47` G2) -/
theorem sumsq_guards :
    eKernel.BufBelow 2
      ∧ eKernel.StageEligibleB 1 = true
      ∧ FlatTargetsOkB (flatKernel (expandEW eKernel)) = true
      ∧ FlatPrintableB (flatKernel (expandEW eKernel)) = true := by
  refine ⟨by decide, by decide, ?_, ?_⟩ <;> native_decide

/-- Each block folds `CHUNK` elements; `GRID` blocks cover `N`. -/
theorem sumsq_geometry : GRID * CHUNK = N := by decide

-- ── memory layout ───────────────────────────────────────────────────────────

def PTX_OFF   : Nat := 0x0100
def BIND_OFF  : Nat := 0x1400
def IN_ID     : Nat := 0x0040
def OUT_ID    : Nat := 0x0044
def MEM_SIZE  : Nat := 0x1500

-- ── CLIF: allocate, upload, launch, sync, download ──────────────────────────

/-- `load`: allocate device buffers and upload once. -/
def loadFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit cuda ptr
  let ctxPtr ← cudaCtxPtr ptr
  let inBytes ← iconst64 (N * 4)
  let inId ← cudaCreateBuffer cuda ptr inBytes
  storeI32 inId (← absAddr ptr IN_ID)
  let outBytes ← iconst64 (GRID * 4)
  let outId ← cudaCreateBuffer cuda ptr outBytes
  storeI32 outId (← absAddr ptr OUT_ID)
  let _ ← call cuda.fnUpload [ctxPtr, inId, dataPtr, inBytes]
  storeI32 inId  (← absAddr ptr BIND_OFF)
  storeI32 outId (← absAddr ptr (BIND_OFF + 4))
  ret

/-- `run`: launch + sync only.  Isolates kernel time from PCIe upload. -/
def runFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ptxOff ← iconst64 PTX_OFF
  let nBufs ← iconst32 2
  let bindOff ← iconst64 BIND_OFF
  let one ← iconst32 1
  let warp ← iconst32 32
  let grid ← iconst32 GRID
  let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff grid one one warp one one
  let _ ← cudaSync cuda ptr
  ret

/-- `fetch`: download the partials. -/
def fetchFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let outId ← load32 (← absAddr ptr OUT_ID)
  let outBytes ← iconst64 (GRID * 4)
  let _ ← call cuda.fnDownload [ctxPtr, outId, outPtr, outBytes]
  ret

def clifIR : String :=
  noopFunction ++ "\n" ++ buildFunction 1 loadFn ++ "\n"
    ++ buildFunction 2 runFn ++ "\n" ++ buildFunction 3 fetchFn

theorem ptx_fits_slot : ptx.toUTF8.toList.length + 1 ≤ BIND_OFF - PTX_OFF := by
  native_decide

theorem mem_map_ok :
    AlgorithmLib.Layout.RegionMap.okB
      [⟨"ptx", PTX_OFF, BIND_OFF - PTX_OFF⟩, ⟨"bind", BIND_OFF, 8⟩] = true := by
  decide

def initialMemory : List UInt8 :=
  let ptxBytes := ptx.toUTF8.toList ++ [0]
  zeros PTX_OFF
    ++ ptxBytes
    ++ zeros (MEM_SIZE - PTX_OFF - ptxBytes.length)

def setup : Setup := {
  cranelift_ir := clifIR
  memory_size := MEM_SIZE
  initial_memory := initialMemory
}

def artifacts : Array Json :=
  #[ toJsonArtifact "warp_sumsq" setup { fn_idx := u32 1 }
       [("run", { fn_idx := u32 2 }), ("fetch", { fn_idx := u32 3 })] ]

end WarpSumSq

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir WarpSumSq.artifacts

namespace WarpSumSq

/-- **The PTX that ships computes the user's spec.**

    Read the chain right to left: `warpSumSqV4E` is an ordinary spec-language
    term; `warpSumSqV4Store_implements` says the warp kernel's *store* lands
    exactly its denotation; `eKernel_elab` says the emittable kernel expands to
    that warp kernel; and `flatKernel_sound` says the emitted instruction list —
    executed by a program counter, over real branches, from instruction 0 in an
    arbitrary machine state — performs it.

    Exact `Float` equality, for every block, **with no hypothesis at all** —
    this kernel contains no `exp`, so the one declared approximation in the
    stack does not touch it. -/
theorem ptx_computes_spec (cta : Nat) (m : MState)
    {Γ : Nat} (env : Fin Γ → Float32) (be : Nat → Expr Γ)
    (hb : ∀ i, denote env (be i) = m.mem inBuf i) :
    ∃ k m', steps cta (flatKernel (expandEW eKernel)) k (0, m)
          = some ((flatKernel (expandEW eKernel)).length, m')
      ∧ m'.mem outBuf cta = denote env (warpSumSqV4E (cta * CHUNK) be K) := by
  obtain ⟨k, m', hs, hw⟩ :=
    flatKernel_sound_idxFree cta (expandEW eKernel) (expandEW_expFree eKernel)
      (expandEW_idxFree eKernel (by decide))
      (expandEW_flat eKernel (by decide)) m
  refine ⟨k, m', hs, ?_⟩
  have hm : m'.mem outBuf cta
      = (((expandEW eKernel).elabIn cta).run m.toWSt).mem outBuf cta :=
    congrArg (fun st => st.mem outBuf cta) hw
  have hid : (expandEW eKernel).elabIn cta = eKernel.elabIn cta := rfl
  rw [hm, hid, eKernel_elab cta]
  exact warpSumSqV4Store_implements _ inBuf outBuf cta K m.toWSt env be hb

end WarpSumSq
