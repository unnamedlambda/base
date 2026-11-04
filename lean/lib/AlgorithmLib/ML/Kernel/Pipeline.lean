module
public import AlgorithmLib.ML.Kernel.Library
meta import AlgorithmLib.ML.Kernel.Library
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace AlgorithmLib.ML.Pipeline

/-!
# An elementwise expression as an artifact

An `Expr n` over `n` float vectors of one length, compiled through `mapKernel`
to an artifact of three entry points: `main` reads the length `N` from the
caller's first eight bytes and makes the device's allocations, `prep` uploads
the `n` inputs, which lie back to back in the caller's data, and `infer` runs
the kernel and downloads as many bytes of its result as the caller's output
asks for.

The inputs lie in allocation 0 as the caller's data has them, so one copy
uploads them all; the result goes to allocation 1. The kernel is
`mapKernel`'s, so what it stores is the expression's denotation
(`BodyMap.mapKernel_body`). A lane computes one element; the grid is as many
blocks of `WARPS` warps as cover `N`, which reach at most `BLOCK - 1` elements
past it, so both allocations hold that many more, and the lanes past `N` read
the next input or that padding and write the result's padding.
-/

/-- Warps a block. -/
def WARPS : Nat := 8
/-- Threads a block, `2 ^ BLOCK_LOG`. -/
def BLOCK : Nat := 32 * WARPS
def BLOCK_LOG : Nat := 8

/-- Where the driver's state starts, where `main` keeps `N`, and where the
    kernel's PTX lies and how many bytes are kept for it. -/
def ORG : Nat := 0x100
def N_OFF : Nat := 0x80
def PTX_OFF : Nat := 0x400
def PTX_LEN : Nat := 0x1000

/-- The most inputs a parameter block holds the pointers of, beside the
    result's. -/
def MAX_INPUTS : Nat := 15

/-- **The program in the driver's standard layout**: the inputs' allocation
    and the result's, each sized at run time, and one kernel handed `n + 1`
    pointers: each input's, then the result's. -/
def app (n : Nat) : Drv.App :=
  { bufs := [0, 0],
    kernels := [⟨PTX_OFF, PTX_LEN, List.range (n + 1), (1, 1, 1), (BLOCK, 1, 1), []⟩],
    memSize := PTX_OFF + PTX_LEN, org := ORG }

/-- The kernel: `e` over buffers `0 … n-1`, stored to buffer `n`. -/
def kernel {n : Nat} (e : Expr n) : MapKernel n :=
  mapKernel e (fun i => i.val) n

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- Blocks covering `len` elements. -/
def blocksFor (len : V .i64) : AlgorithmLib.Prog.Prog V L (V .i64) := do
  ushrImm (← iaddImm len (BLOCK - 1)) BLOCK_LOG

-- ---------------------------------------------------------------------------
-- The three entry points
-- ---------------------------------------------------------------------------

/-- Bring the driver up; then, where the caller handed over eight bytes
    whose `N` is positive and below `2^30`: keep `N`, allocate `n·N` elements
    for the inputs and `N` for the result, each with `BLOCK - 1` more, load
    the kernel and hand it each input's pointer and the result's. -/
def loadCode (n : Nat) : AlgorithmLib.Prog.Prog V L Unit := do
  let a := app n
  let base ← basePtr
  let dp ← dataPtr
  let dl ← dataLen
  Drv.up base a.CTX
  Prog.when .ule (← iconst64 8) dl do
    let len ← load64 dp
    Prog.when .ne len (← iconst64 0) do
      Prog.when .ult len (← iconst64 (2 ^ 30)) do
        storeI64 len (← iaddImm base N_OFF)
        let inBytes ← imul (← iaddImm (← imul len (← iconst64 n)) (BLOCK - 1)) (← iconst64 4)
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 0), inBytes]
        let outBytes ← imul (← iaddImm len (BLOCK - 1)) (← iconst64 4)
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 1), outBytes]
        for (j, k) in a.indexed do
          Drv.loadModule base (a.kern j k)
          Drv.getFunction base (a.kern j k)
        let inputs ← load64 (← iaddImm base (a.bufSlot 0))
        let stride ← imul len (← iconst64 4)
        let mut ptrs := #[inputs]
        let mut next := inputs
        for _ in List.range (n - 1) do
          next ← iadd next stride
          ptrs := ptrs.push next
        let result ← load64 (← iaddImm base (a.bufSlot 1))
        Drv.bindVals base (a.blockOff 0) (a.blockOff 0 + a.arrOff) (ptrs.push result).toList

/-- Upload the inputs, which lie back to back from the caller's data pointer,
    and wait until they are on the device: only when `N` is one `main` kept and
    the caller handed over all of them. A copy from pageable memory returns once
    the bytes are staged, so without the wait the copy's tail would run into
    the next call. -/
def prepCode (n : Nat) : AlgorithmLib.Prog.Prog V L Unit := do
  let a := app n
  let base ← basePtr
  let len ← load64 (← iaddImm base N_OFF)
  Prog.when .ult len (← iconst64 (2 ^ 30)) do
    let bytes ← imul len (← iconst64 (4 * n))
    Prog.when .ule bytes (← dataLen) do
      let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base (a.bufSlot 0)), ← dataPtr, bytes]
      Drv.sync

/-- Where `N` is one `main` kept, run the kernel over the blocks covering it
    and wait for it; then download as many bytes of the result as the
    caller's output asks for, where it asks for some and they are no more
    than the result's. -/
def inferCode (n : Nat) : AlgorithmLib.Prog.Prog V L Unit := do
  let a := app n
  let base ← basePtr
  let len ← load64 (← iaddImm base N_OFF)
  Prog.when .ne len (← iconst64 0) do
    Prog.when .ult len (← iconst64 (2 ^ 30)) do
      a.runX base 0 (← ireduce32 (← blocksFor len))
      Drv.sync
      let ol ← outLen
      Prog.when .ne ol (← iconst64 0) do
        Prog.when .ule ol (← imul len (← iconst64 4)) do
          let src ← load64 (← iaddImm base (a.bufSlot 1))
          let _ ← ext (.cuda .memcpyDtoH) %[← outPtr, src, ol]

end

-- ---------------------------------------------------------------------------
-- Compile: the PTX, the three entry points and the arena.
-- ---------------------------------------------------------------------------

/-- The kernel's PTX: `kernel e`'s body over `n + 1` buffers, `WARPS` warps
    a block. -/
def ptx {n : Nat} (e : Expr n) : String :=
  Body.text "main" (n + 1) 0 (kernel e).ew WARPS

/-- The artifact exporting `main`, `prep` and `infer`, for `e`. -/
def compile {n : Nat} (e : Expr n) : Except String Artifact := do
  if n = 0 ∨ n > MAX_INPUTS then throw s!"{n} inputs; the layout holds 1 to {MAX_INPUTS}"
  let text := ptx e
  if text.utf8ByteSize + 1 > PTX_LEN then throw s!"the kernel's PTX takes {text.utf8ByteSize} bytes"
  let clifProg ← Prog.program
    [.ok noopFunction,
     Prog.entry "main" (Prog.compileProg 1 (loadCode n)),
     Prog.entry "prep" (Prog.compileProg 2 (prepCode n)),
     Prog.entry "infer" (Prog.compileProg 3 (inferCode n))]
  return {
    functions := clifProg
    required_memory := (app n).memSize
    initial_memory := (app n).initialMemory [text]
  }

end AlgorithmLib.ML.Pipeline
