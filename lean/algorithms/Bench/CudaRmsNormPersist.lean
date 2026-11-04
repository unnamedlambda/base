module
public import Lean
public import Std
public import AlgorithmLib.Kernel.Block
meta import AlgorithmLib.Kernel.Block
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
open AlgorithmLib.Kernel

namespace CudaRmsNormPersist

/-!
Persistent RMSNorm over `N` floats: the weights stay on the device, and each
call uploads only `x`.

- `main`, data `[N: u64][w: N f32]`: brings the driver up and, for
  `0 < N < 2^30`, keeps `N`, makes the allocations and uploads `N` and `w`.
- `prep`, data `[x: N f32]`: uploads `x`, where the data holds it.
- `infer`, output `[y: N f32]`: one block of 256 threads computes
  `y = x / rms(x) · w`, then downloads as many bytes of `y` as the output
  asks for, where they fit.

The program calls the CUDA driver directly, in `Drv.App`'s layout. The
kernel reads allocation 0 as `[N: u64][x: N f32][w: N f32]` and writes `y`
to allocation 1; both are sized at run time.
-/

/-- Where `main` keeps `N`, where the driver's state starts, and where the
    kernel's PTX lies and how many bytes are kept for it. -/
def N_OFF : Nat := 0x38
def ORG : Nat := 0x100
def PTX_OFF : Nat := 0x400
def PTX_LEN : Nat := 0x2000

namespace Body
open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog AlgorithmLib.Kernel AlgorithmLib.Kernel.Block

/-- `1e-5`, the epsilon added to the mean square, as `Float32` bits. -/
def epsBits : UInt32 := 0x3727C5AC

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- **RMSNorm in one block of 256 threads**: allocation 0 is `[N][x][w]`,
    `N` read as its low word. Each thread strides `x` summing squares by fused
    multiply-adds, the block reduces the sums, and every thread computes
    `scale = 1 / sqrt(sum / N + 1e-5)`, each step exactly rounded; then
    `y[i] = x[i] · scale · w[i]`. -/
def rmsNorm : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64])
  let tid := ps.snd; let sm := (ps.drop 3).head
  let b0 := (ps.drop 4).head; let y := (ps.drop 5).head
  let n ← load32 b0
  let x ← at_ b0 8
  let w ← iadd x (← ishlImm (← ext n) 2)
  let acc ← strided tid n 256 (← f32c 0) fun i a => do
    let v ← loadF32 (← at4 x i)
    fma v v a
  let tot ← blockAll tid sm 0 8 fadd acc
  let ms ← fadd (← fdiv tot (← fcvtFromUint .f32 n)) (← f32c epsBits)
  let scale ← fdiv (← f32c 0x3F800000) (← sqrt ms)
  stridedDo tid n 256 fun i => do
    let v ← fmul (← loadF32 (← at4 x i)) scale
    store (← fmul v (← loadF32 (← at4 w i))) (← at4 y i)

end

def kernel : Kernel :=
  { name := "main", params := [.i64, .i64],
    code := emit rmsNorm (idParams ++ [.i64, .i64]), smem := 32 }

end Body

/-- The kernel's module, or empty where the printer refuses it (which
    `ptx_fits` would then not prove). -/
def ptxSource : String :=
  match Body.kernel.ptx with
  | .ok s => s
  | .error _ => ""

open AlgorithmLib.Prog

/-- **The program in the driver's standard layout**: `N`, `x` and `w` in
    allocation 0, `y` in allocation 1, and one kernel handed both. -/
def app : Drv.App :=
  { bufs := [0, 0], kernels := [⟨PTX_OFF, PTX_LEN, [0, 1], (1, 1, 1), (256, 1, 1), []⟩],
    memSize := PTX_OFF + PTX_LEN, org := ORG }

/-- Bring the driver up and load the kernel; then, where the data is `N` and
    then `N` weights, `0 < N < 2^30`: keep `N`, make the allocations, hand
    them to the kernel and upload `N` and the weights. -/
def loadCode : Prog V L Unit := do
  let a := app
  let base ← basePtr
  let dp ← dataPtr
  let dl ← dataLen
  a.setup base
  Prog.when .ule (← iconst64 8) dl do
  Prog.when .ult dl (← iconst64 (2 ^ 33)) do
    let n ← load64 dp
    Prog.when .ne n (← iconst64 0) do
    Prog.when .ult n (← iconst64 (2 ^ 30)) do
      -- the weights are what follows `N`, all of it
      let wBytes ← isub dl (← iconst64 8)
      Prog.when .eq (← imul n (← iconst64 4)) wBytes do
        storeI64 n (← iaddImm base N_OFF)
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 0), ← iadd dl wBytes]
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 1), wBytes]
        a.rebind base 0
        let buf0 ← load64 (← iaddImm base (a.bufSlot 0))
        let _ ← ext (.cuda .memcpyHtoD) %[buf0, ← iaddImm base N_OFF, ← iconst64 8]
        let _ ← ext (.cuda .memcpyHtoD) %[← iadd buf0 dl, ← iaddImm dp 8, wBytes]

/-- Upload `x` after `N`, where the data holds it. -/
def prepCode : Prog V L Unit := do
  let base ← basePtr
  let n ← load64 (← iaddImm base N_OFF)
  Prog.when .ult n (← iconst64 (2 ^ 30)) do
    let bytes ← imul n (← iconst64 4)
    Prog.when .ule bytes (← dataLen) do
      let _ ← ext (.cuda .memcpyHtoD) %[← iaddImm (← load64 (← iaddImm base (app.bufSlot 0))) 8, ← dataPtr, bytes]

/-- Run the kernel and wait for it; then download as many bytes of `y` as the
    caller's output asks for, where it asks for some and they fit. -/
def inferCode : Prog V L Unit := do
  let a := app
  let base ← basePtr
  let n ← load64 (← iaddImm base N_OFF)
  a.run base 0
  Drv.sync
  let ol ← outLen
  Prog.when .ne ol (← iconst64 0) do
  Prog.when .ult n (← iconst64 (2 ^ 30)) do
    Prog.when .ule ol (← imul n (← iconst64 4)) do
      let _ ← ext (.cuda .memcpyDtoH) %[← outPtr, ← load64 (← iaddImm base (a.bufSlot 1)), ol]

def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     Prog.entry "main" (Prog.compileProg 1 loadCode),
     Prog.entry "prep" (Prog.compileProg 2 prepCode),
     Prog.entry "infer" (Prog.compileProg 3 inferCode)]

/-- The module prints, and fits where it is placed with its terminating zero. -/
theorem ptx_fits : 0 < ptxSource.utf8ByteSize ∧ ptxSource.utf8ByteSize < PTX_LEN := by
  native_decide

def buildSetup (clif : List FuncData) : Artifact := {
  functions := clif,
  required_memory := app.memSize,
  initial_memory := app.initialMemory [ptxSource]
}

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[
    artifactEntry "cuda_rmsnorm" (buildSetup clif)
  ]

end CudaRmsNormPersist
