module
public import AlgorithmLib.Surface.Layout
meta import AlgorithmLib.Surface.Layout
public import Lean
public import AlgorithmLib.Surface.Link
meta import AlgorithmLib.Surface.Link
public import Std
public import Scan.Layout
meta import Scan.Layout
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.ML.Machine.Geometry
meta import AlgorithmLib.ML.Machine.Geometry
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
public import AlgorithmLib.Host.Driver
meta import AlgorithmLib.Host.Driver
public import AlgorithmLib.ML.Model.Frontend
meta import AlgorithmLib.ML.Model.Frontend
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

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

/-- Where the two kernels' PTX lies in the arena, past the driver's state. -/
def PTX_OFF : Nat := 0x0400
def PTX_L_OFF : Nat := 0x1700
def MEM_SIZE : Nat := 0x3700

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

/-- **The program, in the driver's standard layout**: the input and the
    output allocation, and the two kernels, both handed them in that order. -/
def app : Drv.App :=
  { bufs := [N * 4, N * 4],
    kernels := [⟨PTX_OFF, PTX_L_OFF - PTX_OFF, [0, 1], (GRID, 1, 1), (32, 1, 1)⟩,
                ⟨PTX_L_OFF, MEM_SIZE - PTX_L_OFF, [0, 1], (LGRID, 1, 1), (32, 1, 1)⟩],
    memSize := MEM_SIZE }

/-- **The whole map, not the interesting part of it.** -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"driver",  0, app.stateEnd⟩,
   ⟨"ptx",     PTX_OFF, PTX_L_OFF - PTX_OFF⟩,
   ⟨"ptxLoop", PTX_L_OFF, MEM_SIZE - PTX_L_OFF⟩]


#eval LayoutScan.check "Warp.Silu" [``memMap]
theorem siluMap_ok :
    memMap.okB = true ∧ memMap.withinB MEM_SIZE = true := by decide

theorem siluPtx_fits :
    (ptx.toUTF8.toList.length + 1 ≤ PTX_L_OFF - PTX_OFF)
      ∧ (ptxLoop.toUTF8.toList.length + 1 ≤ MEM_SIZE - PTX_L_OFF) := by native_decide

/-- The signature the entry points' typestates state of both kernels: `main`
    takes two pointers. -/
theorem silu_sigs :
    HProg.Sem.ptxParamSizes ptx "main" = some [8, 8] ∧ HProg.Sem.ptxParamSizes ptxLoop "main" = some [8, 8] := by native_decide

open AlgorithmLib.Prog


def loadFnCode : Prog V L Unit := do
  let base ← basePtr
  let dataPtr ← dataPtr
  app.setup base
  app.upload base 0 dataPtr (N * 4)

def runFnCode : Prog V L Unit := do
  let base ← basePtr
  app.run base 0
  Drv.sync

/-- The same work, `E` elements per lane: `LGRID` blocks instead of `GRID`. -/
def runLoopFnCode : Prog V L Unit := do
  let base ← basePtr
  app.run base 1
  Drv.sync

def fetchFnCode : Prog V L Unit := do
  let base ← basePtr
  app.download base (← outPtr) 1 (N * 4)


def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     Prog.entry "main" (Prog.compileProg 1 loadFnCode),
     Prog.entry "run" (Prog.compileProg 2 runFnCode),
     Prog.entry "fetch" (Prog.compileProg 3 fetchFnCode),
     Prog.entry "runLoop" (Prog.compileProg 4 runLoopFnCode)]

def initialMemory : List UInt8 := app.initialMemory [ptx, ptxLoop]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[ artifactEntry "silu_warp" { functions := clif, required_memory := MEM_SIZE, initial_memory := initialMemory } ]


end SiluWarp

def Warp.Silu.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie SiluWarp.clifIR
  emitArtifacts outDir (SiluWarp.artifacts clif)

#eval ShipScan.check "Warp.Silu" `Warp.Silu.main
