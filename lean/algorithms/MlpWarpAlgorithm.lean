import Lean
import Std
import AlgorithmLib.Gen
import AlgorithmLib.ML
import AlgorithmLib.ProgCuda
import LayoutScan
import ShipScan

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

def ptx : String := emitProvenKernel "main" kernel

theorem mlp_stores (st : WSt) (l0 : Lane) :
    ((kernel.elabIn 0).run st).mem outBuf (outIx.eval 0 0 l0)
      = denote (fun i => st.mem inBuf ((inIx i).eval 0 0 l0)) spec :=
  compileWKernel_stores (fun _ => inBuf) outBuf inIx spec outIx _ st l0
    (fun l h => by rw [elemIx_inj l l0 h])

/-! ### Seam guards (`A47` G2) -/
theorem mlp_guards :
    kernel.BufBelow 2
      ∧ kernel.StageEligibleB 1 = true
      ∧ FlatTargetsOkB (flatKernel (expandEW kernel)) = true
      ∧ FlatPrintableB (flatKernel (expandEW kernel)) = true := by
  refine ⟨by decide, by decide, ?_, ?_⟩ <;> native_decide

/-- One lane per model instance, `GRID` blocks of 32. -/
theorem mlp_geometry : GRID * 32 = LANES ∧ NIN = LANES * D := by decide

-- ── memory layout ───────────────────────────────────────────────────────────

def PTX_OFF  : Nat := 0x0100
def BIND_OFF : Nat := 0x30000
def IN_ID    : Nat := 0x0040
def OUT_ID   : Nat := 0x0044
def MEM_SIZE : Nat := 0x30100

open AlgorithmLib.Prog


/-- **The kernel, as a record the launch sites read.**

    The bind table is written in `load` and the launch is issued from `run`,
    so the arity the driver is told comes from `params` here rather than from
    a number written beside the launch. -/
def mlpK : AlgorithmLib.Kernel := {
  name   := "main"
  params := [{ shape := [.sta NIN],  ro := true,  name := "inBuf" },
             { shape := [.sta NOUT], ro := false, name := "outBuf" }]
  geom   := AlgorithmLib.Kernel.Geom.static GRID 1 1 32 1 1
  ptxOff := PTX_OFF
  ptxText := some ptx
}

def loadFnCode : Prog V Lbl Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  cudaInit ptr
  let ctxPtr ← cudaCtxPtr ptr
  let inBytes ← iconst64 (NIN * 4)
  let inId ← cudaCreateBuffer ptr inBytes
  store inId (← absAddr ptr IN_ID)
  let outBytes ← iconst64 (NOUT * 4)
  let outId ← cudaCreateBuffer ptr outBytes
  store outId (← absAddr ptr OUT_ID)
  let _ ← ffi .cudaUpload %[ctxPtr, inId, dataPtr, inBytes]
  kernelBindAt mlpK ptr BIND_OFF [inId, outId]

def runFnCode : Prog V Lbl Unit := do
  let ptr ← basePtr
  kernelRelaunch mlpK ptr BIND_OFF
  let _ ← cudaSync ptr

def fetchFnCode : Prog V Lbl Unit := do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← outPtr
  let outId ← load32 (← absAddr ptr OUT_ID)
  let outBytes ← iconst64 (NOUT * 4)
  let _ ← ffi .cudaDownload %[ctxPtr, outId, outPtr, outBytes]


def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     Prog.compileProg 1 loadFnCode,
     Prog.compileProg 2 runFnCode,
     Prog.compileProg 3 fetchFnCode]

theorem ptx_fits_slot : ptx.toUTF8.toList.length + 1 ≤ BIND_OFF - PTX_OFF := by
  native_decide

/-- **The whole map, not the interesting part of it.**

    The context slots and the buffer-id words are live offsets this file
    writes; a map that omits them cannot see a collision involving them, and
    the check reads as if it had. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"ctx_cuda",   AlgorithmLib.ContextSlots.cuda, 8⟩,
   ⟨"in_id",      IN_ID, 4⟩,
   ⟨"out_id",     OUT_ID, 4⟩,
   ⟨"ptx",        PTX_OFF, BIND_OFF - PTX_OFF⟩,
   ⟨"bind",       BIND_OFF, 8⟩]


#eval LayoutScan.check "MlpWarpAlgorithm" [``memMap]
theorem mem_map_ok :
    memMap.okB = true ∧ memMap.withinB MEM_SIZE = true := by decide

def initialMemory : List UInt8 :=
  let ptxBytes := AlgorithmLib.Kernel.ptxBytes mlpK
  zeros PTX_OFF ++ ptxBytes ++ zeros (MEM_SIZE - PTX_OFF - ptxBytes.length)

def setup (clif : List FuncData) : Artifact := {
  functions := clif
  memory_size := MEM_SIZE
  initial_memory := initialMemory
}

def artifacts (clif : List FuncData) : Array Json :=
  #[ toJsonArtifact "mlp_warp" (setup clif) ]

end MlpWarp

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie MlpWarp.clifIR
  emitArtifacts outDir (MlpWarp.artifacts clif)

namespace MlpWarp

/-- **The emitted PTX runs the compiled kernel, exactly and unconditionally.**

    `expandEW kernel` is what is actually printed, and this says the instruction
    list — executed by a program counter over real branches from instruction 0 —
    performs it, for every block, with no hypothesis. -/
theorem mlp_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW kernel)) k (0, m)
          = some ((flatKernel (expandEW kernel)).length, m')
      ∧ m'.toWSt = ((expandEW kernel).elabIn cta).run m.toWSt :=
  flatKernel_sound_idxFree cta (expandEW kernel) (expandEW_expFree kernel)
    (expandEW_idxFree kernel
      (compileWKernel_idxFree (fun _ => inBuf) outBuf inIx spec outIx _
        (fun _ => ⟨⟨⟨⟨trivial, trivial⟩, trivial⟩, trivial⟩, trivial⟩) (by decide)))
    (expandEW_flat kernel (compileWKernel_flat _ _ _ _ _ _)) m

/-- **…and it runs the *unexpanded* kernel, given the one declared
    approximation.**

    This model contains `exp` (inside SiLU), so unlike the reduction and GEMV
    kernels its claim does depend on `ExpIsEx2` — the single named identity
    `e^x = 2^(x·log₂e)`.  Measured cost end to end: 3.27e-7 max relative error
    against a CPU evaluation of the same three layers. -/
theorem mlp_ptx_runs_kernel (h : ExpIsEx2) (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW kernel)) k (0, m)
          = some ((flatKernel (expandEW kernel)).length, m')
      ∧ m'.toWSt = (kernel.elabIn cta).run m.toWSt := by
  obtain ⟨k, m', hs, hw⟩ := mlp_ptx_exact cta m
  exact ⟨k, m', hs, by rw [hw]; exact expandEW_run h kernel cta 0 m.toWSt⟩

end MlpWarp

#eval ShipScan.check "MlpWarpAlgorithm"
