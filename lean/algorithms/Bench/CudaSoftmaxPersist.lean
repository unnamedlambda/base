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

namespace CudaSoftmaxPersist

/-!
Multi-block persistent softmax over `n` floats.

Allocations, made by `main` in this order:
  0  x        — input floats (n f32)
  1  y        — output floats (n f32)
  2  meta     — [n: u32, num_blocks: u32]
  3  partials — [max0, sum0, max1, sum1, ...] (num_blocks · 8 bytes)
  4  params   — [global_max: f32, global_inv_sum: f32], written by global_reduce

Entry points:
- `main`, data `[n: u64]`: brings the driver up and, for `0 < n < 2^30`,
  keeps `n` and `num_blocks = ⌈n / 256⌉`, makes the allocations, hands them
  to the kernels and uploads `meta`.
- `prep`, data `[x: n f32]`: uploads `x`, where it fits.
- `infer`: `core` then `finalize`; `stack`: `core` 64 times, then `finalize`.
  `core` runs one kernel for `n ≤ 2048` and three otherwise; `finalize`
  waits, then downloads as many bytes of `y` as the output asks for, where
  they fit.

Four PTX entry points in one module:
  small_softmax — one block, for a short row
  block_reduce  — num_blocks blocks: each block's (local_max, local_sum)
  global_reduce — one block: (global_max, global_inv_sum) into params
  normalize     — num_blocks blocks: y[i] = ex2((x[i]-global_max)·log2e) · global_inv_sum

The program calls the CUDA driver directly, in `Drv.App`'s layout; every
allocation but `meta` and `params` is sized at run time.
-/

/-- Where `main` keeps `n`, `num_blocks` and the packed `meta` it uploads;
    where the driver's state starts; where the PTX lies and how many bytes
    are kept for it. -/
def N_OFF : Nat := 0x38
def NB_OFF : Nat := 0x40
def META_OFF : Nat := 0x48
def ORG : Nat := 0x100
def PTX_OFF : Nat := 0x800
def PTX_LEN : Nat := 0x8000

namespace Body
open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog AlgorithmLib.Kernel AlgorithmLib.Kernel.Block

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- **Softmax over one row per block**: `n`, the word at `meta`, floats a row,
    row `blockIdx` of `x` into the same row of `y`. Blocks of 256 threads,
    each striding the row three times: the maximum, the sum of
    `2^((x - max)·log₂ e)`, then each such term times the sum's reciprocal,
    the last two by the hardware's approximations. -/
def rowSoftmax : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let x := (ps.drop 4).head; let m := (ps.drop 5).head; let y := (ps.drop 6).head
  let n ← load32 m
  let off ← ishlImm (← imul (← ext cta) (← ext n)) 2
  let xr ← iadd x off
  let yr ← iadd y off
  let ninf ← f32c negInfBits
  let lMax ← strided tid n 256 ninf fun i acc => do fmax acc (← loadF32 (← at4 xr i))
  let gMax ← blockAll tid sm 0 8 fmax lMax
  let lSum ← strided tid n 256 (← f32c 0) fun i acc => do
    fadd acc (← expA (← fsub (← loadF32 (← at4 xr i)) gMax))
  let inv ← callLocal rcpApprox %[← blockAll tid sm 32 8 fadd lSum]
  stridedDo tid n 256 fun i => do
    store (← fmul (← expA (← fsub (← loadF32 (← at4 xr i)) gMax)) inv) (← at4 yr i)

/-- **Each block's softmax state**: thread `g = blockIdx·256 + threadIdx`
    starts from `(x[g], 1)`, or the empty state past `n` (`lowestBits`, `0`),
    and thread 0 writes the block's merged state as two floats at
    `partials + 8·blockIdx`. -/
def blockReduce : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let x := (ps.drop 4).head; let m := (ps.drop 5).head; let parts := (ps.drop 6).head
  let n ← load32 m
  let g ← iadd (← imul cta (← c32 256)) tid
  let inb ← icmp .ult g n
  -- past `n` the load reads element 0, which the select then drops
  let xv ← loadF32 (← at4 x (← select inb g (← c32 0)))
  let low ← f32c lowestBits
  let m0 ← select inb (← fmax xv low) low
  let s0 ← select inb (← f32c 0x3F800000) (← f32c 0)
  blockOnline tid sm 8 m0 s0 fun bm bs => do
    let pa ← iadd parts (← ishlImm (← ext cta) 3)
    store bm pa
    store bs (← at_ pa 4)

/-- **The blocks' states merged**: one block strides the `num_blocks` (the
    word at `meta + 4`) partial states, and thread 0 writes the maximum and the
    reciprocal of the sum to `params`. -/
def globalReduce : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let tid := ps.snd; let sm := (ps.drop 3).head
  let m := (ps.drop 4).head; let parts := (ps.drop 5).head; let prm := (ps.drop 6).head
  let nb ← load32 (← at_ m 4)
  let r ← wloop %[tid, ← f32c lowestBits, ← f32c 0]
    (fun cs => pure (contIfULt cs.head nb, %[cs.snd, cs.thd], ()))
    (fun cs _ => do
      let pa ← iadd parts (← ishlImm (← ext cs.head) 3)
      let (nm, ns) ← online cs.snd cs.thd (← loadF32 pa) (← loadF32 (← at_ pa 4))
      pure %[← iadd cs.head (← c32 256), nm, ns])
  blockOnline tid sm 8 r.head r.snd fun gm gs => do
    store gm prm
    store (← callLocal rcpApprox %[gs]) (← at_ prm 4)

/-- **The output**: `y[g] = 2^((x[g] - max)·log₂ e) · inv` for `g < n`, the
    maximum and `inv` from `params`, a thread an element. -/
def normalize : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd
  let x := (ps.drop 4).head; let m := (ps.drop 5).head
  let prm := (ps.drop 6).head; let y := (ps.drop 7).head
  let n ← load32 m
  let gMax ← loadF32 prm
  let inv ← loadF32 (← at_ prm 4)
  let g ← iadd (← imul cta (← c32 256)) tid
  when .ult g n do
    store (← fmul (← expA (← fsub (← loadF32 (← at4 x g)) gMax)) inv) (← at4 y g)

end

/-- Shared bytes: two reductions of eight warps' floats each. -/
def smemBytes : Nat := 64

/-- `rowSoftmax` as a kernel named `name`, over `(x, meta, y)`. -/
def rowKernel (name : String) : Kernel :=
  { name, params := [.i64, .i64, .i64],
    code := emit rowSoftmax (idParams ++ [.i64, .i64, .i64]), smem := smemBytes }

def kernels : List Kernel :=
  [rowKernel "small_softmax",
   { name := "block_reduce", params := [.i64, .i64, .i64],
     code := emit blockReduce (idParams ++ [.i64, .i64, .i64]), smem := smemBytes },
   { name := "global_reduce", params := [.i64, .i64, .i64],
     code := emit globalReduce (idParams ++ [.i64, .i64, .i64]), smem := smemBytes },
   { name := "normalize", params := [.i64, .i64, .i64, .i64],
     code := emit normalize (idParams ++ [.i64, .i64, .i64, .i64]) }]

end Body

/-- The four kernels as one module, or empty where the printer refuses one
    (which `ptx_fits` would then not prove). -/
def ptxSource : String :=
  match Kernel.module Body.kernels with
  | .ok s => s
  | .error _ => ""

-- CLIF: u0:0 noop, u0:1 load, u0:2 prep, u0:3 core, u0:4 finalize
-- Load: init CUDA, alloc 5 bufs, pack/upload meta, store n/num_blocks
open AlgorithmLib.Prog

/-- **The program in the driver's standard layout**: five allocations, made
    by `main`, and the four kernels of one module. -/
def app : Drv.App :=
  { bufs := [0, 0, 0, 0, 0],
    kernels := [⟨PTX_OFF, PTX_LEN, [0, 2, 1], (1, 1, 1), (256, 1, 1), []⟩,
                ⟨PTX_OFF, PTX_LEN, [0, 2, 3], (1, 1, 1), (256, 1, 1), []⟩,
                ⟨PTX_OFF, PTX_LEN, [2, 3, 4], (1, 1, 1), (256, 1, 1), []⟩,
                ⟨PTX_OFF, PTX_LEN, [0, 2, 4, 1], (1, 1, 1), (256, 1, 1), []⟩],
    memSize := PTX_OFF + PTX_LEN, org := ORG,
    entries := ["small_softmax", "block_reduce", "global_reduce", "normalize"] }

def loadCode : Prog V L Unit := do
  let a := app
  let base ← basePtr
  let dp ← dataPtr
  a.setup base
  Prog.when .ule (← iconst64 8) (← dataLen) do
    let n ← load64 dp
    Prog.when .ne n (← iconst64 0) do
    Prog.when .ult n (← iconst64 (2 ^ 30)) do
      let nb ← ushrImm (← iaddImm n 255) 8
      storeI64 n (← iaddImm base N_OFF)
      storeI64 nb (← iaddImm base NB_OFF)
      let xBytes ← imul n (← iconst64 4)
      let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 0), xBytes]
      let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 1), xBytes]
      let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 2), ← iconst64 8]
      let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 3), ← imul nb (← iconst64 8)]
      let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 4), ← iconst64 8]
      for j in List.range a.kernels.length do
        a.rebind base j
      storeI64 (← bor n (← ishlImm nb 32)) (← iaddImm base META_OFF)
      let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base (a.bufSlot 2)), ← iaddImm base META_OFF,
        ← iconst64 8]

/-- Upload `x`, where it fits. -/
def prepCode : Prog V L Unit := do
  let base ← basePtr
  let dl ← dataLen
  let n ← load64 (← iaddImm base N_OFF)
  Prog.when .ule dl (← imul n (← iconst64 4)) do
    let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base (app.bufSlot 0)), ← dataPtr, dl]

/-- One kernel for a short row, three for a long one. -/
def coreCode : Prog V L Unit := do
  let a := app
  let base ← basePtr
  let n ← load64 (← iaddImm base N_OFF)
  let nb ← load64 (← iaddImm base NB_OFF)
  Prog.when .ne nb (← iconst64 0) do
  Prog.when .ult nb (← iconst64 (2 ^ 31)) do
    let nb32 ← ireduce32 nb
    let _ ← ifte .ule n (← iconst64 2048)
      (thn := do a.run base 0; pure %[])
      (els := do
        a.runX base 1 nb32
        a.run base 2
        a.runX base 3 nb32
        pure %[])
    pure ()

/-- Wait; then download as many bytes of `y` as the caller's output asks
    for, where it asks for some and they fit. -/
def finalizeCode : Prog V L Unit := do
  let base ← basePtr
  let n ← load64 (← iaddImm base N_OFF)
  Drv.sync
  let ol ← outLen
  Prog.when .ne ol (← iconst64 0) do
  Prog.when .ult n (← iconst64 (2 ^ 30)) do
    Prog.when .ule ol (← imul n (← iconst64 4)) do
      let _ ← ext (.cuda .memcpyDtoH) %[← outPtr, ← load64 (← iaddImm base (app.bufSlot 1)), ol]

/-- Stack depth baked into the `stack` entry's wrapper. -/
def STACK_DEPTH : Nat := 64

def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     Prog.entry "main" (Prog.compileProg 1 loadCode),
     Prog.entry "prep" (Prog.compileProg 2 prepCode),
     Prog.compileProg 3 coreCode,
     Prog.compileProg 4 finalizeCode,
     Prog.entry "infer" (Prog.compileProg 5 (Prog.sequenceWrapper [3, 4])),
     Prog.entry "stack" (Prog.compileProg 6 (Prog.sequenceWrapper (List.replicate STACK_DEPTH 3 ++ [4])))]

/-- The module prints, and fits where it is placed with its terminating zero. -/
theorem ptx_fits : 0 < ptxSource.utf8ByteSize ∧ ptxSource.utf8ByteSize < PTX_LEN := by
  native_decide

def buildSetup (clif : List FuncData) : Artifact := {
  functions := clif,
  required_memory := app.memSize,
  initial_memory := app.initialMemory (List.replicate 4 ptxSource)
}

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[
    artifactEntry "cuda_softmax" (buildSetup clif)
  ]

end CudaSoftmaxPersist
