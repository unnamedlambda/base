module
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
open AlgorithmLib.IR
open AlgorithmLib.Prog
open AlgorithmLib.PTX

namespace AlgorithmLib


namespace CudaPipeline

instance : Inhabited (Reg k) := ⟨⟨""⟩⟩

/-!
An elementwise expression over `n` float vectors of one length, compiled to
an artifact of three entry points: `main` reads the length `N` from the
caller's first eight bytes and makes the device's allocations, `prep` uploads
the `n` inputs, which lie back to back in the caller's data, and `infer`
runs the kernel, which writes the result over input `out`, then downloads as
many of its bytes as the caller's output asks for.

The program calls the CUDA driver directly, in `Drv.App`'s layout: one
allocation of eight bytes holding `N`, and one holding the inputs back to
back, as the caller's data has them, so one copy uploads them all. Its size
is known only at run time, so `main` makes it itself. The kernel runs a fixed
grid and strides over the elements.
-/

inductive Expr : Nat → Type where
  | input : Fin n → Expr n
  | const : String → Expr n   -- PTX float literal, e.g. "0f40000000"
  | add : Expr n → Expr n → Expr n
  | mul : Expr n → Expr n → Expr n

instance : HAdd (Expr n) (Expr n) (Expr n) := ⟨.add⟩
instance : HMul (Expr n) (Expr n) (Expr n) := ⟨.mul⟩

def Expr.input0 : Expr (n + 1) := .input ⟨0, by simp⟩
def Expr.input1 : Expr (n + 2) := .input ⟨1, by simp⟩
def Expr.scalarBits (bits : String) : Expr n := .const bits
def Expr.saxpy (a x y : Expr n) : Expr n := a * x + y

/-- The kernel's grid: blocks of `blockSize` threads stride over the
    elements, `GRID` blocks at a time. -/
def GRID : Nat := 1024

/-- Where the driver's state starts, where `main` keeps `N`, and where the
    kernel's PTX lies and how many bytes are kept for it. -/
def ORG : Nat := 0x100
def N_OFF : Nat := 0x80
def PTX_OFF : Nat := 0x400
def PTX_LEN : Nat := 0x1000

/-- **The program in the driver's standard layout**: `N` in allocation 0,
    the inputs in allocation 1, of `4 * n * N` bytes, known only at run
    time, and one kernel handed both. -/
def app (blockSize : Nat := 256) : Drv.App :=
  { bufs := [8, 0], kernels := [⟨PTX_OFF, PTX_LEN, [0, 1], (GRID, 1, 1), (blockSize, 1, 1)⟩],
    memSize := PTX_OFF + PTX_LEN, org := ORG }

-- ---------------------------------------------------------------------------
-- PTX emission via the typed builder in AlgorithmLib.PTX.
-- ---------------------------------------------------------------------------

/-- Structural on `e`, so it has equation lemmas: a `partial` here would make
    the PTX lowering an opaque constant that nothing can unfold, which is a
    stronger obstacle than merely having no theorem about it. -/
def emitExprPTX {n : Nat}
    (e : Expr n) (inPtrs : Array (Reg .u64)) (off : Reg .u64) : PTX (Reg .f32) := do
  match e with
  | .input idx =>
      let addr ← freshRd
      addRd addr inPtrs[idx.val]! off
      let f ← freshF
      ldGlobalF f addr
      pure f
  | .const bits =>
      let f ← freshF
      rawLine s!"    mov.f32 {f.raw}, {bits};"
      pure f
  | .add a b =>
      let fa ← emitExprPTX a inPtrs off
      let fb ← emitExprPTX b inPtrs off
      let f ← freshF
      addF f fa fb
      pure f
  | .mul a b =>
      let fa ← emitExprPTX a inPtrs off
      let fb ← emitExprPTX b inPtrs off
      let f ← freshF
      mulF f fa fb
      pure f

/-- Element `i` of every input, for each `i` below `N` the thread meets
    striding from its global index. Input `k` starts `4 * k * N` bytes into
    the inputs' allocation. -/
def kernelBody {n : Nat} (e : Expr n) (output : Fin n) (blockSize : Nat) :
    PTX Unit := do
  let metaPtr ← ldParam "meta_ptr"
  let inBase ← ldParam "in_ptr"
  let nReg ← freshR; ldGlobalU nReg metaPtr
  let nb ← freshRd; mulWideRI nb nReg 4
  let mut inPtrs : Array (Reg .u64) := #[inBase]
  for _ in List.range (n - 1) do
    let p ← freshRd
    addRd p inPtrs.back! nb
    inPtrs := inPtrs.push p
  let cta ← freshR; movR cta ctaX
  let tid ← freshR; movR tid tidX
  let gid ← freshR; madLoRC gid cta blockSize tid
  label "LOOP"
  let p ← freshP; setpGe p gid nReg
  braIf p "DONE"
  let off ← freshRd; mulWideRI off gid 4
  let result ← emitExprPTX e inPtrs off
  let outAddr ← freshRd; addRd outAddr inPtrs[output.val]! off
  stGlobalF outAddr result
  addRI gid gid (GRID * blockSize)
  bra "LOOP"
  label "DONE"
  ptxRet

def ptxSource {n : Nat} (e : Expr n) (output : Fin n) (blockSize : Nat) : String :=
  buildModule 0 [{ name := "main", params := ["meta_ptr", "in_ptr"], body := kernelBody e output blockSize }]

-- ---------------------------------------------------------------------------
-- The three entry points
-- ---------------------------------------------------------------------------

/-- Bring the driver up; then, where the caller handed over eight bytes
    whose `N` is positive and below `2^30`: keep `N`, make both allocations,
    load the kernel, write its parameters and upload `N`. -/
def loadCode (inputs : Nat) (blockSize : Nat := 256) : Prog V L Unit := do
  let a := app blockSize
  let base ← basePtr
  let dp ← dataPtr
  let dl ← dataLen
  Drv.up base a.CTX
  Prog.when .ule (← iconst64 8) dl do
    let n ← load64 dp
    Prog.when .ne n (← iconst64 0) do
      Prog.when .ult n (← iconst64 (2 ^ 30)) do
        storeI64 n (← iaddImm base N_OFF)
        Drv.alloc base (a.bufSlot 0) 8
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 1), ← imul n (← iconst64 (4 * inputs))]
        for (j, k) in a.indexed do
          Drv.loadModule base (a.kern j k)
          Drv.getFunction base (a.kern j k)
        Drv.bind base (a.blockOff 0) (a.blockOff 0 + a.arrOff) ([0, 1].map a.bufSlot)
        Drv.upload base (a.bufSlot 0) dp 8

/-- Upload the inputs, which lie back to back from the caller's data pointer:
    only when the caller handed over all of them. -/
def prepCode (inputs : Nat) (blockSize : Nat := 256) : Prog V L Unit := do
  let a := app blockSize
  let base ← basePtr
  let n ← load64 (← iaddImm base N_OFF)
  let bytes ← imul n (← iconst64 (4 * inputs))
  Prog.when .ule bytes (← dataLen) do
    let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base (a.bufSlot 1)), ← dataPtr, bytes]

/-- Run the kernel and wait for it; then download as many bytes of input
    `output` as the caller's output asks for, where it asks for some and they
    fit in the input. -/
def inferCode {n : Nat} (output : Fin n) (blockSize : Nat := 256) : Prog V L Unit := do
  let a := app blockSize
  let base ← basePtr
  let nv ← load64 (← iaddImm base N_OFF)
  a.run base 0
  Drv.sync
  let ol ← outLen
  Prog.when .ne ol (← iconst64 0) do
    Prog.when .ult nv (← iconst64 (2 ^ 30)) do
      Prog.when .ule ol (← imul nv (← iconst64 4)) do
        let src ← iadd (← load64 (← iaddImm base (a.bufSlot 1))) (← imul nv (← iconst64 (4 * output.val)))
        let _ ← ext (.cuda .memcpyDtoH) %[← outPtr, src, ol]

-- ---------------------------------------------------------------------------
-- Compile: the PTX, the three entry points and the arena.
-- ---------------------------------------------------------------------------

/-- The artifact exporting `main`, `prep` and `infer`, for `e` written over
    input `out`. -/
def Expr.compileTo {n : Nat} (e : Expr n) (out : Nat) (h : out < n := by decide)
    (blockSize : Nat := 256) : Except String Artifact := do
  let output : Fin n := ⟨out, h⟩
  let ptx := ptxSource e output blockSize
  if ptx.utf8ByteSize + 1 > PTX_LEN then throw s!"the kernel's PTX takes {ptx.utf8ByteSize} bytes"
  let clifProg ← Prog.program
    [.ok noopFunction,
     Prog.entry "main" (Prog.compileProg 1 (loadCode n blockSize)),
     Prog.entry "prep" (Prog.compileProg 2 (prepCode n blockSize)),
     Prog.entry "infer" (Prog.compileProg 3 (inferCode output blockSize))]
  return {
    functions := clifProg
    required_memory := (app blockSize).memSize
    initial_memory := (app blockSize).initialMemory [ptx]
  }

end CudaPipeline

end AlgorithmLib
