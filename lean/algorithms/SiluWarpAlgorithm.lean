import Lean
import Std
import AlgorithmLib

/-!
  # A user's spec, compiled to the GPU

  `Transformer.silu` is an ordinary `Expr`.  `compileW` turns it into machine
  code, `compileWKernel_correct` proves the result is the spec's denotation
  from memory, and `emitProvenKernel` lowers it to PTX (proven: `flatKernel_sound`).  Nothing here was
  hand-written as a kernel.
-/

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML

namespace SiluWarp

def GRID : Nat := 2097152
def N : Nat := GRID * 32

/-- The spec. One line. -/
def spec : Expr 1 := Transformer.silu (.var 0)

/-! **The function-first surface produces this exact term.**

    Three ways of writing the activation — by name, by its definition, and as
    raw arithmetic — all `rfl`-equal to the shipped spec.  That is the check
    that `Frontend.lean` is sugar over `Expr` and not a second, silently
    divergent model language. -/
example : spec = ofFn 1 (fun a => silu a) := rfl
example : spec = ofFn 1 (fun a => a * sigmoid a) := rfl
example : spec = ofFn 1 (fun a => a / (1 + expE (-a))) := rfl

/-- Element this lane owns: `ctaid*32 + laneid`. -/
def ix : IdxE := .add (.mul .ctaId (.lit 32)) .laneId

def kernel : EWStmt := compileWKernel (fun _ => 0) 1 (fun _ => ix) spec ix 16

def ptx : String := emitProvenKernel "main" kernel

/-! ## The same spec, `E` elements per lane

    One element per lane means 2,097,152 blocks of 32 threads, which measured
    **39% of roofline** — block scheduling, not bandwidth.  `mapLoopEW` gives
    each lane `E` elements from the *same* spec, so the block count drops by
    `E` and each thread has independent work to overlap.

    Nothing about the spec or the compiler changes; what changed is that
    `compileWKernel_correct` was generalised off block 0
    (`compileWKernel_correctAt`), so a compiled expression may sit in a loop. -/

/-- Elements per lane. -/
def E : Nat := 32

/-- Blocks, with `E` elements per lane. -/
def LGRID : Nat := N / (32 * E)

/-- The row this block owns starts at `ctaid · 32 · E`. -/
def lbase : IdxE := .mul .ctaId (.lit (32 * E))

def lkernel : EWStmt := mapLoopEW spec (fun _ => 0) 1 lbase E

def ptxLoop : String := emitProvenKernel "main" lkernel

/-! ### Seam guards (`A47` G2) — for both the one-per-lane and looped kernels -/
theorem silu_guards :
    kernel.BufBelow 2 ∧ lkernel.BufBelow 2
      ∧ kernel.StageEligibleB 1 = true ∧ lkernel.StageEligibleB 1 = true
      ∧ FlatTargetsOkB (flatKernel (expandEW kernel)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW lkernel)) = true
      ∧ FlatPrintableB (flatKernel (expandEW kernel)) = true
      ∧ FlatPrintableB (flatKernel (expandEW lkernel)) = true := by
  refine ⟨by decide, by decide, by decide, by decide, ?_, ?_, ?_, ?_⟩ <;> native_decide

/-- Both geometries cover exactly `N`: one element per lane, and `E` per lane. -/
theorem silu_geometry : GRID * 32 = N ∧ LGRID * (32 * E) = N := by decide

/-- **The looped kernel stores the same spec at every element it visits.**

    One application of `mapLoopEW_stores`; the base being lane-and-loop-free is
    a `decide`, and the input differing from the output is immediate. -/
theorem lkernel_stores (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (j0 : Nat) (l0 : Lane) (hj0 : j0 < E) :
    ((lkernel.elabAt cta 0 ir im).run st).mem 1
        ((stride32 lbase).eval cta j0 l0 ir im)
      = denote (fun _ => st.mem 0 ((stride32 lbase).eval cta j0 l0 ir im)) spec :=
  mapLoopEW_stores spec (fun _ => 0) 1 lbase (by decide) E cta
    (fun _ => (by decide : (0 : Buf) ≠ 1)) ir im st j0 l0 hj0

def PTX_OFF : Nat := 0x0100
def PTX_L_OFF : Nat := 0x1400
def BIND_OFF : Nat := 0x3400
def IN_ID : Nat := 0x0040
def OUT_ID : Nat := 0x0044
def MEM_SIZE : Nat := 0x3500

/-- **What the one-element-per-lane kernel stores.**

    `A47` G18: this demo had a conformance theorem for the *looped* kernel and
    none for the original, and **no PTX theorem for either** — so nothing said
    the emitted text performs what was proven.  Every other demo had that. -/
theorem silu_stores (st : WSt) (l0 : Lane) :
    ((kernel.elabIn 0).run st).mem 1 (ix.eval 0 0 l0)
      = denote (fun _ => st.mem 0 (ix.eval 0 0 l0)) spec :=
  compileWKernel_stores (fun _ => 0) 1 (fun _ => ix) spec ix 16 st l0
    (fun l h => by rw [elemIx_inj l l0 h])

/-- **And the emitted PTX runs it**, from instruction 0, over real branches. -/
theorem silu_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW kernel)) k (0, m)
          = some ((flatKernel (expandEW kernel)).length, m')
      ∧ m'.toWSt = ((expandEW kernel).elabIn cta).run m.toWSt :=
  flatKernel_sound_idxFree cta (expandEW kernel) (expandEW_expFree kernel)
    (expandEW_idxFree kernel
      (compileWKernel_idxFree (fun _ => 0) 1 (fun _ => ix) spec ix 16
        (fun _ => (by decide : IdxE.IregFree ix)) (by decide)))
    (expandEW_flat kernel (compileWKernel_flat _ _ _ _ _ _)) m

/-- The same for the looped kernel — the one that actually ships the bandwidth. -/
theorem silu_loop_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW lkernel)) k (0, m)
          = some ((flatKernel (expandEW lkernel)).length, m')
      ∧ m'.toWSt = ((expandEW lkernel).elabIn cta).run m.toWSt :=
  flatKernel_sound_idxFree cta (expandEW lkernel) (expandEW_expFree lkernel)
    (expandEW_idxFree lkernel (by decide))
    (expandEW_flat lkernel (by decide)) m

theorem siluMap_ok :
    AlgorithmLib.Layout.RegionMap.okB
      [⟨"ptx", PTX_OFF, PTX_L_OFF - PTX_OFF⟩,
       ⟨"ptxLoop", PTX_L_OFF, BIND_OFF - PTX_L_OFF⟩,
       ⟨"bind", BIND_OFF, 8⟩] = true := by decide

theorem siluPtx_fits :
    (ptx.toUTF8.toList.length + 1 ≤ PTX_L_OFF - PTX_OFF)
      ∧ (ptxLoop.toUTF8.toList.length + 1 ≤ BIND_OFF - PTX_L_OFF) := by native_decide

def loadFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit cuda ptr
  let ctxPtr ← cudaCtxPtr ptr
  let nBytes ← iconst64 (N * 4)
  let inId ← cudaCreateBuffer cuda ptr nBytes
  storeI32 inId (← absAddr ptr IN_ID)
  let outId ← cudaCreateBuffer cuda ptr nBytes
  storeI32 outId (← absAddr ptr OUT_ID)
  let _ ← call cuda.fnUpload [ctxPtr, inId, dataPtr, nBytes]
  storeI32 inId (← absAddr ptr BIND_OFF)
  storeI32 outId (← absAddr ptr (BIND_OFF + 4))
  ret

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

/-- The same work, `E` elements per lane: `LGRID` blocks instead of `GRID`. -/
def runLoopFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ptxOff ← iconst64 PTX_L_OFF
  let nBufs ← iconst32 2
  let bindOff ← iconst64 BIND_OFF
  let one ← iconst32 1
  let warp ← iconst32 32
  let grid ← iconst32 LGRID
  let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff grid one one warp one one
  let _ ← cudaSync cuda ptr
  ret

def fetchFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let outId ← load32 (← absAddr ptr OUT_ID)
  let nBytes ← iconst64 (N * 4)
  let _ ← call cuda.fnDownload [ctxPtr, outId, outPtr, nBytes]
  ret

def clifIR : String :=
  noopFunction ++ "\n" ++ buildFunction 1 loadFn ++ "\n"
    ++ buildFunction 2 runFn ++ "\n" ++ buildFunction 3 fetchFn ++ "\n"
    ++ buildFunction 4 runLoopFn

def initialMemory : List UInt8 :=
  let p := ptx.toUTF8.toList ++ [0]
  let q := ptxLoop.toUTF8.toList ++ [0]
  zeros PTX_OFF ++ p ++ zeros (PTX_L_OFF - PTX_OFF - p.length)
    ++ q ++ zeros (MEM_SIZE - PTX_L_OFF - q.length)

def artifacts : Array Json :=
  #[ toJsonArtifact "silu_warp"
      { cranelift_ir := clifIR, memory_size := MEM_SIZE, initial_memory := initialMemory }
      { fn_idx := u32 1 }
      [("run", { fn_idx := u32 2 }), ("fetch", { fn_idx := u32 3 }),
       ("runLoop", { fn_idx := u32 4 })] ]

end SiluWarp

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir SiluWarp.artifacts
