module
public import Lean
public import Std
public import AlgorithmLib.Vocab.PTX
meta import AlgorithmLib.Vocab.PTX
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
open AlgorithmLib.PTX

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
def PTX_LEN : Nat := 0x1000

def ptxSource : String := buildModule 36 [{ name := "main", params := ["buf0", "buf1"], body := do
  let buf0 ← ldParam "buf0"
  let buf1 ← ldParam "buf1"
  -- Read dynamic N from buf0[0]; x starts at buf0+8; w starts at buf0+8+N*4
  let nReg ← freshR;  ldGlobalU nReg buf0
  let xPtr ← freshRd; addRdI xPtr buf0 8
  let nU64 ← freshRd; cvtU64 nU64 nReg
  let nOff ← freshRd; shlRd nOff nU64 2
  let wPtr ← freshRd; addRd wPtr xPtr nOff
  let (tid, warpId, laneId) ← getWarpIds
  -- loop1: sum of squares
  let acc ← freshF; movFC acc f32_0
  let tmp ← freshF
  strideLoop tid nReg 256 "loop1" "done1" fun i => do
    let addr ← elemAddr xPtr i; ldGlobalF tmp addr; fmaRn acc tmp tmp acc
  warpReduceSum acc tmp
  lane0WriteSmem laneId warpId "skip1" fun wAddr => stSharedFD wAddr acc
  -- thread 0: sum warps, divide by N (dynamic float), add eps, rsqrt
  thread0Op tid "skip2" do
    let sBase ← smemBase
    let total ← freshF
    crossWarp8 total tmp sBase 0 addF
    let nf ← freshF; cvtF32 nf nReg
    divRn total total nf
    let eps ← freshF; movFC eps f32_eps
    addF total total eps; rsqrt total total
    stSharedF sBase 32 total
  -- loop2: normalize with dynamic w pointer
  let sBase2 ← smemBase
  let scale ← freshF; ldSharedF scale sBase2 32
  strideLoop tid nReg 256 "loop2" "done2" fun j => do
    let xAddr ← elemAddr xPtr j
    let wAddr ← elemAddr wPtr j
    let yAddr ← elemAddr buf1 j
    let xi ← freshF; ldGlobalF xi xAddr
    let wi ← freshF; ldGlobalF wi wAddr
    mulF xi xi scale; mulF xi xi wi; stGlobalF yAddr xi
  ptxRet }]

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
