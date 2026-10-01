module
public import AlgorithmLib.Core.Artifact
meta import AlgorithmLib.Core.Artifact
public import AlgorithmLib.Core.Bytes
meta import AlgorithmLib.Core.Bytes
public import AlgorithmLib.Surface.Layout
meta import AlgorithmLib.Surface.Layout
public import AlgorithmLib.Core.IR
meta import AlgorithmLib.Core.IR
public import AlgorithmLib.Surface.FFI
meta import AlgorithmLib.Surface.FFI
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Vocab.PTX
meta import AlgorithmLib.Vocab.PTX
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

-- ---------------------------------------------------------------------------
-- Expression DSL: small staged language for elementwise GPU kernels.
-- One Expr describes the per-element computation; `compileTo` produces a
-- Artifact + load/prep/infer algorithms that wire up the persistent kernel
-- pattern (alloc → upload → launch → download).
-- ---------------------------------------------------------------------------

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

-- Shared-memory layout produced by these functions:
--   0x10 ctx slot,
--   0x38 N (i64),  0x40 meta buffer id (i32),
--   0x44 + 4*i  input[i] buffer id (i32)
def ptxSourceOff : Nat := 0x0100
def bindDescOff  : Nat := 0x1400

/-- The fixed part of that layout, as regions.

    Every program this file builds shares these offsets, so a collision here is
    a collision in all of them at once. The input buffer ids start at `0x44` and
    run to `0x44 + 4n`, which is bounded by the PTX region below. -/
def memMap (n : Nat) : Layout.RegionMap :=
  [⟨"ctx_cuda",  ContextSlots.cuda, 8⟩,
   ⟨"n",          0x38, 8⟩,
   ⟨"meta_buf",   0x40, 4⟩,
   ⟨"input_bufs", 0x44, 4 * n⟩,
   ⟨"ptx",        ptxSourceOff, bindDescOff - ptxSourceOff⟩]

/-- Disjoint for every arity these pipelines are built at. Stated over a range
    rather than at one `n` because the input-id block is the only region whose
    size depends on the program, and it is the one that could grow into the
    PTX. -/
theorem memMap_ok : (List.range 16).all (fun n => Layout.RegionMap.okB (memMap n)) = true := by
  decide

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

def kernelBody {n : Nat} (e : Expr n) (output : Fin n) (blockSize : Nat) :
    PTX Unit := do
  let metaPtr ← ldParam "meta_ptr"
  let mut inPtrs : Array (Reg .u64) := #[]
  for i in List.range n do
    let p ← ldParam s!"in{i}_ptr"
    inPtrs := inPtrs.push p
  let cta ← freshR; movR cta ctaX
  let tid ← freshR; movR tid tidX
  let gid ← freshR; madLoRC gid cta blockSize tid
  let nReg ← freshR; ldGlobalU nReg metaPtr
  let p ← freshP; setpGe p gid nReg
  braIf p "DONE"
  let gid64 ← freshRd; cvtU64 gid64 gid
  let off ← freshRd; shlRd off gid64 2
  let result ← emitExprPTX e inPtrs off
  let outAddr ← freshRd; addRd outAddr inPtrs[output.val]! off
  stGlobalF outAddr result
  label "DONE"
  ptxRet

def ptxSource {n : Nat} (e : Expr n) (output : Fin n) (blockSize : Nat) : String :=
  let params := "meta_ptr" :: (List.range n).map (fun i => s!"in{i}_ptr")
  buildModule 0 [{ name := "main", params, body := kernelBody e output blockSize }]

-- ---------------------------------------------------------------------------
-- CLIF emission: each stage is a term, compiled against one callee table.
-- ---------------------------------------------------------------------------

/-- Allocate the device buffers and publish the element count. -/
def loadCode (inputs : Nat) : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  cudaInit ptr
  let n ← load64 dataPtr
  storeI64 n (← absAddr ptr 0x38)
  let nBytes ← ishlImm n 2
  let metaBytes ← iconst64 8
  let metaBuf ← cudaCreateBuffer ptr metaBytes
  storeI32 metaBuf (← absAddr ptr 0x40)
  (List.range inputs).forM fun (i : Nat) => do
    let buf ← cudaCreateBuffer ptr nBytes
    storeI32 buf (← absAddr ptr (0x44 + 4*i))
  let _ ← cudaUpload ptr metaBuf (← iconst64 0x38) metaBytes

/-- Upload the inputs, which lie back to back from the caller's data pointer. -/
def prepCode (inputs : Nat) : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let n ← load64 (← absAddr ptr 0x38)
  let nBytes ← ishlImm n 2
  let ctxPtr ← cudaCtxPtr ptr
  let _ ← (List.range inputs).foldlM (init := dataPtr) fun curSrc (i : Nat) => do
    let bufId ← load32 (← absAddr ptr (0x44 + 4*i))
    let _ ← ffi .cudaUpload %[ctxPtr, bufId, curSrc, nBytes]
    iadd curSrc nBytes

/-- Launch, synchronise, and download the output — the last only when the
    caller asked for one. -/
def inferCode {n : Nat} (output : Fin n) (blockSize : Nat) : Prog V L Unit := do
  let ptr ← basePtr
  let outPtr ← outPtr
  let outLen ← outLen
  let nElems ← load64 (← absAddr ptr 0x38)
  let blkM1 ← iaddImm nElems (blockSize - 1)
  let wg64 ← ushrImm blkM1 (Nat.log2 blockSize)
  let wg ← ireduce32 wg64
  let ptxOff ← iconst64 ptxSourceOff
  let nBufs ← iconst32 (n + 1)
  let bindOff ← iconst64 bindDescOff
  let one32 ← iconst32 1
  let blkX ← iconst32 blockSize
  let _ ← cudaLaunch ptr ptxOff nBufs bindOff wg one32 one32 blkX one32 one32
  let _ ← cudaSync ptr
  let zero64 ← iconst64 0
  Prog.when .ne outLen zero64 do
    let ctxPtr ← cudaCtxPtr ptr
    let outBufId ← load32 (← absAddr ptr (0x44 + 4*output.val))
    let _ ← ffi .cudaDownload %[ctxPtr, outBufId, outPtr, outLen]

-- ---------------------------------------------------------------------------
-- Compile: assemble PTX + CLIF + initial memory into an artifact exporting
-- the three stages as `main`, `prep` and `infer`.
-- ---------------------------------------------------------------------------

/-- The three stages are terms only once `n`, `out` and `blockSize` are given.

    That used to be a problem: the compiler wanted a `wf` proof about each of
    them, `decide` will not reduce a builder that iterates over an arity, and
    so three `native_decide` obligations travelled out to every caller. They
    are gone. The stages are typed terms, and `compileProg` checks the body it
    emitted while the generator runs. -/
def Expr.compileTo {n : Nat} (e : Expr n) (out : Nat) (h : out < n := by decide)
    (blockSize : Nat := 256) : Except String Artifact := do
  let output : Fin n := ⟨out, h⟩
  let ptxBytes := (ptxSource e output blockSize).toUTF8.toList ++ [0]
  let bindDesc := (List.range (n + 1)).foldr
    (fun i acc => uint32ToBytes (UInt32.ofNat i) ++ acc) []
  let memSize := bindDescOff + bindDesc.length + 0x100
  let initialMemory :=
    zeros ptxSourceOff
    ++ ptxBytes ++ zeros (bindDescOff - ptxSourceOff - ptxBytes.length)
    ++ bindDesc ++ zeros (memSize - bindDescOff - bindDesc.length)
  let clifProg ← Prog.program
    [.ok noopFunction,
     Prog.entry "main" (Prog.compileProg 1 (loadCode n)),
     Prog.entry "prep" (Prog.compileProg 2 (prepCode n)),
     Prog.entry "infer" (Prog.compileProg 3 (inferCode output blockSize))]
  return {
    functions := clifProg
    required_memory := memSize
    initial_memory := initialMemory
  }

end CudaPipeline

end AlgorithmLib
