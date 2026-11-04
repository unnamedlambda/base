module
public import Lean
public import Std
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace CudaGemvPersist

/-!
Persistent GEMV: `A` stays on the device, and each call uploads only `x`.

- `main`, data `[m: u64][n: u64][A: m·n f32, row-major]`: brings the driver
  and cuBLAS up and, for `0 < m, n < 2^31` with `4·m·n < 2^35`, keeps `m` and
  `n`, makes the allocations of `A`, `x` and `y`, and uploads `A`, where the
  data holds exactly `A` after its header.
- `prep`, data `[x: n f32]`: uploads `x`, where it fits.
- `infer`, output `[y: m f32]`: `y = A x` by cuBLAS, then downloads as many
  bytes of `y` as the output asks for, where they fit.

The program calls the CUDA driver and cuBLAS directly, in `Drv.App`'s layout;
all three allocations are sized at run time.
-/

open AlgorithmLib.Prog

/-- Where `main` keeps `m` and `n`, and where the driver's state starts. -/
def M_OFF : Nat := 0x38
def N_OFF : Nat := 0x40
def ORG : Nat := 0x100

/-- **The program in the driver's standard layout**: `A`, `x` and `y` as
    three allocations sized at run time, and cuBLAS. -/
def app : Drv.App := { bufs := [0, 0, 0], kernels := [], memSize := 0x200, blas := true, org := ORG }

def loadCode : Prog V L Unit := do
  let a := app
  let base ← basePtr
  let dp ← dataPtr
  let dl ← dataLen
  a.setup base
  Prog.when .ule (← iconst64 16) dl do
    let m ← load64 dp
    let n ← load64 (← iaddImm dp 8)
    Prog.when .ne m (← iconst64 0) do
    Prog.when .ult m (← iconst64 (2 ^ 31)) do
    Prog.when .ne n (← iconst64 0) do
    Prog.when .ult n (← iconst64 (2 ^ 31)) do
      let mnBytes ← imul (← imul m n) (← iconst64 4)
      Prog.when .ult mnBytes (← iconst64 (2 ^ 35)) do
      -- `A` is what follows the header, all of it
      let aBytes ← isub dl (← iconst64 16)
      Prog.when .eq mnBytes aBytes do
        storeI64 m (← iaddImm base M_OFF)
        storeI64 n (← iaddImm base N_OFF)
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 0), aBytes]
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 1), ← imul n (← iconst64 4)]
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 2), ← imul m (← iconst64 4)]
        let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base (a.bufSlot 0)), ← iaddImm dp 16, aBytes]

def prepCode : Prog V L Unit := do
  let base ← basePtr
  let dl ← dataLen
  let n ← load64 (← iaddImm base N_OFF)
  Prog.when .ule dl (← imul n (← iconst64 4)) do
    let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base (app.bufSlot 1)), ← dataPtr, dl]

/-- `y = A x`: `A` is row-major `m × n`, so cuBLAS reads it as the
    column-major `n × m` matrix and transposes it. -/
def inferCode : Prog V L Unit := do
  let a := app
  let base ← basePtr
  let m ← load64 (← iaddImm base M_OFF)
  let n ← load64 (← iaddImm base N_OFF)
  Prog.when .ne m (← iconst64 0) do
  Prog.when .ult m (← iconst64 (2 ^ 31)) do
  Prog.when .ne n (← iconst64 0) do
  Prog.when .ult n (← iconst64 (2 ^ 31)) do
    let m32 ← ireduce32 m
    let n32 ← ireduce32 n
    let one ← iconst32 1
    let _ ← ext (.cublas .sgemv) %[← load64 (← iaddImm base a.BLAS), one, n32, m32, ← iaddImm base a.ONE,
      ← load64 (← iaddImm base (a.bufSlot 0)), n32, ← load64 (← iaddImm base (a.bufSlot 1)), one,
      ← iaddImm base a.ZERO, ← load64 (← iaddImm base (a.bufSlot 2)), one]
    Drv.sync
    let ol ← outLen
    Prog.when .ne ol (← iconst64 0) do
      Prog.when .ule ol (← imul m (← iconst64 4)) do
        let _ ← ext (.cuda .memcpyDtoH) %[← outPtr, ← load64 (← iaddImm base (a.bufSlot 2)), ol]

def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     Prog.entry "main" (Prog.compileProg 1 loadCode),
     Prog.entry "prep" (Prog.compileProg 2 prepCode),
     Prog.entry "infer" (Prog.compileProg 3 inferCode)]

def buildSetup (clif : List FuncData) : Artifact := {
  functions := clif,
  required_memory := app.memSize,
  initial_memory := app.initialMemory []
}

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[
    artifactEntry "cuda_gemv" (buildSetup clif)
  ]

end CudaGemvPersist
