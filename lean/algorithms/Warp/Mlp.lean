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
public import AlgorithmLib.ML.Ptx.Body
meta import AlgorithmLib.ML.Ptx.Body
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML

namespace MlpWarp

def D : Nat := 4                    -- model width
def L : Nat := 3                    -- layers
def GRID : Nat := 16384
def LANES : Nat := GRID * 32
def NIN : Nat := LANES * D
def NOUT : Nat := LANES

/-- One layer: RMSNorm, a matvec, SiLU, and a residual. -/
def layer {Γ : Nat} (x : Fin D → Expr Γ) : Fin D → Expr Γ :=
  let h := Transformer.rmsNorm D (fun _ => .lit 1) x
  let y := fun i => Transformer.silu (Transformer.matvec D (fun _ _ => .lit 1) h i)
  Transformer.vadd x y

/-- `n` layers, each layer's output vector **bound** so the term stays linear. -/
def stack : Nat → {Γ : Nat} → (Fin D → Expr Γ) → Expr Γ
  | 0, _, x => .sum D (fun i => x i)
  | (n + 1), Γ, x => bindVec D (layer x) (stack n (fun i => boundVar Γ i))

/-- The spec: `D` inputs, `L` layers, one scalar out. -/
def spec : Expr D := stack L (fun i => .var i)

/-- **The library combinator produces this exact term.**  `stackLayers` is the
    generic form of `stack`; that it is `rfl`-equal here is the check that the
    frontend's sharing story is the shipped one, not a second implementation. -/
example : spec = stackLayers layer (fun x => .sum D (fun i => x i)) L (fun i => .var i) :=
  rfl

/-- Lane `l` of block `c` owns inputs `[(c*32+l)*D, +D)`. -/
def inIx : Fin D → IdxE := fun i =>
  .add (.mul (.add (.mul .ctaId (.lit 32)) .laneId) (.lit D)) (.lit i.val)

/-- One output per lane. -/
def outIx : IdxE := .add (.mul .ctaId (.lit 32)) .laneId

def inBuf : Buf := 0
def outBuf : Buf := 1

def kernel : EWStmt := compileWKernel (fun _ => inBuf) outBuf inIx spec outIx (D + slots spec + 1)

def ptx : String := Body.text "main" 2 0 kernel

theorem mlp_stores (st : WSt) (l0 : Lane) :
    ((kernel.run st 0 0)).mem outBuf (outIx.eval 0 0 l0)
      = denote (fun i => st.mem inBuf ((inIx i).eval 0 0 l0)) spec :=
  compileWKernel_stores (fun _ => inBuf) outBuf inIx spec outIx _ st l0
    (fun l h => by rw [elemIx_inj l l0 h])

/-! ### Seam guards (`A47` G2) -/
theorem mlp_guards :
    kernel.BufBelow 2 ∧ kernel.StageEligibleB 1 = true := ⟨by decide, by decide⟩

/-- One lane per model instance, `GRID` blocks of 32. -/
theorem mlp_geometry : GRID * 32 = LANES ∧ NIN = LANES * D := by decide

-- ── memory layout ───────────────────────────────────────────────────────────

def PTX_OFF  : Nat := 0x0200
def MEM_SIZE : Nat := 0x30100

open AlgorithmLib.Prog


/-- **The program, in the driver's standard layout**: the inputs' and the
    outputs' allocations, and the kernel, handed them in that order. -/
def app : Drv.App :=
  { bufs := [NIN * 4, NOUT * 4],
    kernels := [⟨PTX_OFF, MEM_SIZE - PTX_OFF, [0, 1], (GRID, 1, 1), (32, 1, 1), []⟩],
    memSize := MEM_SIZE }

def loadFnCode : Prog V Lbl Unit := do
  let base ← basePtr
  app.setup base
  app.upload base 0 (← dataPtr) (NIN * 4)

def runFnCode : Prog V Lbl Unit := do
  let base ← basePtr
  app.run base 0
  Drv.sync

def fetchFnCode : Prog V Lbl Unit := do
  let base ← basePtr
  app.download base (← outPtr) 1 (NOUT * 4)


def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     Prog.entry "main" (Prog.compileProg 1 loadFnCode),
     Prog.entry "run" (Prog.compileProg 2 runFnCode),
     Prog.entry "fetch" (Prog.compileProg 3 fetchFnCode)]

theorem ptx_fits_slot : ptx.toUTF8.toList.length + 1 ≤ MEM_SIZE - PTX_OFF := by
  native_decide

/-- The signature the entry points' typestates state of the kernel. -/
theorem mlp_sig : HProg.Sem.ptxParamSizes ptx "main" = some [8, 8] := by native_decide

/-- **The whole map, not the interesting part of it.** -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"driver", 0, app.stateEnd⟩,
   ⟨"ptx",    PTX_OFF, MEM_SIZE - PTX_OFF⟩]


#eval LayoutScan.check "Warp.Mlp" [``memMap]
theorem mem_map_ok :
    memMap.okB = true ∧ memMap.withinB MEM_SIZE = true := by decide

def initialMemory : List UInt8 := app.initialMemory [ptx]

def setup (clif : List FuncData) : Artifact := {
  functions := clif
  required_memory := MEM_SIZE
  initial_memory := initialMemory
}

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[ artifactEntry "mlp_warp" (setup clif) ]

end MlpWarp

def Warp.Mlp.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie MlpWarp.clifIR
  emitArtifacts outDir (MlpWarp.artifacts clif)

namespace MlpWarp

end MlpWarp

#eval ShipScan.check "Warp.Mlp" `Warp.Mlp.main
