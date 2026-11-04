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
public import AlgorithmLib.ML.Ptx.Body
meta import AlgorithmLib.ML.Ptx.Body
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
public import AlgorithmLib.Host.Driver
meta import AlgorithmLib.Host.Driver
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # A proven warp kernel, wired into a real `Artifact`

  The kernel body is `AlgorithmLib.ML.warpSumSqV4Store`, which is proven to
  compute `denote env spec` exactly (`warpSumSqV4Store_implements`).  The
  shipped `eKernel` is that kernel by `rfl` (`eKernel_eq` below), and its body
  kernel (`Body.kernel`) is what ships: launching it lands the spec in each
  block's output word (`sumsq_body`).

  This file supplies the surrounding CLIF: allocate device buffers, upload the
  caller's input, launch one warp, sync, download the single-float result.

  What is proven: launching the kernel computes the spec, at the words the
  hardware computes on.
  What is trusted: the printer from a body to PTX text and this CLIF wrapper.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
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

/-- **The emitted kernel is the proven kernel**, by `rfl`: the quad base is
    `strideIxV4Cta CHUNK`, which is `v4BaseOff (cta·CHUNK)` in every block. -/
theorem eKernel_eq : eKernel = warpSumSqV4Store inBuf outBuf (strideIxV4Cta CHUNK) .ctaId K := rfl

theorem eKernel_base (cta : Nat) :
    ∀ i l, (strideIxV4Cta CHUNK).eval cta i l = v4BaseOff (cta * CHUNK) i l := fun _ _ => rfl

def ptx : String := Body.text "main" 2 0 eKernel

/-! ### Seam guards (`A47` G2) -/
theorem sumsq_guards :
    eKernel.BufBelow 2 ∧ eKernel.StageEligibleB 1 = true := ⟨by decide, by decide⟩

/-- Each block folds `CHUNK` elements; `GRID` blocks cover `N`. -/
theorem sumsq_geometry : GRID * CHUNK = N := by decide

-- ── memory layout ───────────────────────────────────────────────────────────

def PTX_OFF   : Nat := 0x0200
def MEM_SIZE  : Nat := 0x1500

-- ── CLIF: allocate, upload, launch, sync, download ──────────────────────────

open AlgorithmLib.Prog


/-- **The program, in the driver's standard layout**: the input and the
    partials' allocations, and the kernel, handed them in that order. -/
def app : Drv.App :=
  { bufs := [N * 4, GRID * 4],
    kernels := [⟨PTX_OFF, MEM_SIZE - PTX_OFF, [0, 1], (GRID, 1, 1), (32, 1, 1), []⟩],
    memSize := MEM_SIZE }

/-- `load`: allocate device buffers and upload once. -/
def loadFnCode : Prog V L Unit := do
  let base ← basePtr
  app.setup base
  app.upload base 0 (← dataPtr) (N * 4)

/-- `run`: launch + sync only.  Isolates kernel time from PCIe upload. -/
def runFnCode : Prog V L Unit := do
  let base ← basePtr
  app.run base 0
  Drv.sync

/-- `fetch`: download the partials. -/
def fetchFnCode : Prog V L Unit := do
  let base ← basePtr
  app.download base (← outPtr) 1 (GRID * 4)


def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     Prog.entry "main" (Prog.compileProg 1 loadFnCode),
     Prog.entry "run" (Prog.compileProg 2 runFnCode),
     Prog.entry "fetch" (Prog.compileProg 3 fetchFnCode)]

theorem ptx_fits_slot : ptx.toUTF8.toList.length + 1 ≤ MEM_SIZE - PTX_OFF := by
  native_decide

/-- The signature the entry points' typestates state of the kernel. -/
theorem sumsq_sig : HProg.Sem.ptxParamSizes ptx "main" = some [8, 8] := by native_decide

/-- **The whole map, not the interesting part of it.** -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"driver", 0, app.stateEnd⟩,
   ⟨"ptx",    PTX_OFF, MEM_SIZE - PTX_OFF⟩]

theorem mem_map_ok :
    memMap.okB = true ∧ memMap.withinB MEM_SIZE = true := by decide

def initialMemory : List UInt8 := app.initialMemory [ptx]

def setup (clif : List FuncData) : Artifact := {
  functions := clif
  required_memory := MEM_SIZE
  initial_memory := initialMemory
}

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[ artifactEntry "warp_sumsq" (setup clif) ]

end WarpSumSq

def Warp.SumSq.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie WarpSumSq.clifIR
  emitArtifacts outDir (WarpSumSq.artifacts clif)

namespace WarpSumSq

end WarpSumSq

#eval LayoutScan.check "warp_sumsq" [``WarpSumSq.memMap]

#eval ShipScan.check "Warp.SumSq" `Warp.SumSq.main
