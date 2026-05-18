import AlgorithmLib.Core
import AlgorithmLib.Bytes
import AlgorithmLib.Layout
import AlgorithmLib.IR
import AlgorithmLib.FFI
import AlgorithmLib.PTX

open Lean
open AlgorithmLib.IR
open AlgorithmLib.PTX

namespace AlgorithmLib

namespace CudaPipeline

private instance : Inhabited (Reg k) := ⟨⟨""⟩⟩

-- ---------------------------------------------------------------------------
-- Expression DSL: small staged language for elementwise GPU kernels.
-- One Expr describes the per-element computation; `compileTo` produces a
-- BaseConfig + load/prep/infer algorithms that wire up the persistent kernel
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

structure CompileResult where
  config : BaseConfig
  loadAlgorithm : Algorithm
  prepAlgorithm : Algorithm
  inferAlgorithm : Algorithm

-- Shared-memory layout produced by these functions:
--   0x10 ctx slot, 0x18 caller data_ptr, 0x28 caller out_ptr, 0x30 caller out_len,
--   0x38 N (i64),  0x40 meta buffer id (i32),
--   0x44 + 4*i  input[i] buffer id (i32)
private def ptxSourceOff : Nat := 0x0100
private def bindDescOff  : Nat := 0x1400

-- ---------------------------------------------------------------------------
-- PTX emission via the typed builder in AlgorithmLib.PTX.
-- ---------------------------------------------------------------------------

private partial def emitExprPTX {n : Nat}
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

private def kernelBody {n : Nat} (e : Expr n) (output : Fin n) (blockSize : Nat) :
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

private def ptxSource {n : Nat} (e : Expr n) (output : Fin n) (blockSize : Nat) : String :=
  let params := "meta_ptr" :: (List.range n).map (fun i => s!"in{i}_ptr")
  buildModule 0 [{ name := "main", params, body := kernelBody e output blockSize }]

-- ---------------------------------------------------------------------------
-- CLIF emission via the typed builder in AlgorithmLib.IR + FFI helpers.
-- ---------------------------------------------------------------------------

private def loadFn (inputs : Nat) : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit cuda ptr
  let n ← load64 dataPtr
  storeI64 n (← absAddr ptr 0x38)
  let nBytes ← ishlImm n 2
  let metaBytes ← iconst64 8
  let metaBuf ← cudaCreateBuffer cuda ptr metaBytes
  storeI32 metaBuf (← absAddr ptr 0x40)
  for i in List.range inputs do
    let buf ← cudaCreateBuffer cuda ptr nBytes
    storeI32 buf (← absAddr ptr (0x44 + 4*i))
  let _ ← cudaUpload cuda ptr metaBuf (← iconst64 0x38) metaBytes
  ret

private def prepFn (inputs : Nat) : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let n ← load64 (← absAddr ptr 0x38)
  let nBytes ← ishlImm n 2
  let ctxPtr ← cudaCtxPtr ptr
  let _ ← (List.range inputs).foldlM (init := dataPtr) fun curSrc i => do
    let bufId ← load32 (← absAddr ptr (0x44 + 4*i))
    let _ ← call cuda.fnUpload [ctxPtr, bufId, curSrc, nBytes]
    iadd curSrc nBytes
  ret

private def inferFn {n : Nat} (output : Fin n) (blockSize : Nat) : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let outPtr ← load64 (← absAddr ptr 0x28)
  let outLen ← load64 (← absAddr ptr 0x30)
  let nElems ← load64 (← absAddr ptr 0x38)
  let blkM1 ← iaddImm nElems (blockSize - 1)
  let wg64 ← ushrImm blkM1 (Nat.log2 blockSize)
  let wg ← ireduce32 wg64
  let ptxOff ← iconst64 ptxSourceOff
  let nBufs ← iconst32 (n + 1)
  let bindOff ← iconst64 bindDescOff
  let one32 ← iconst32 1
  let blkX ← iconst32 blockSize
  let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff wg one32 one32 blkX one32 one32
  let _ ← cudaSync cuda ptr
  let zero64 ← iconst64 0
  let cond ← icmp .eq outLen zero64
  let skipBlk ← declareBlock []
  let downloadBlk ← declareBlock []
  brif cond skipBlk.ref [] downloadBlk.ref []
  startBlock skipBlk
  ret
  startBlock downloadBlk
  let ctxPtr ← cudaCtxPtr ptr
  let outBufId ← load32 (← absAddr ptr (0x44 + 4*output.val))
  let _ ← call cuda.fnDownload [ctxPtr, outBufId, outPtr, outLen]
  ret

-- ---------------------------------------------------------------------------
-- Compile: assemble PTX + CLIF + initial memory into a CompileResult.
-- ---------------------------------------------------------------------------

def Expr.compileTo {n : Nat} (e : Expr n) (out : Nat) (h : out < n := by decide)
    (blockSize : Nat := 256) : CompileResult :=
  let output : Fin n := ⟨out, h⟩
  let ptxBytes := (ptxSource e output blockSize).toUTF8.toList ++ [0]
  let bindDesc := (List.range (n + 1)).foldr
    (fun i acc => uint32ToBytes (UInt32.ofNat i) ++ acc) []
  let memSize := bindDescOff + bindDesc.length + 0x100
  let initialMemory :=
    zeros ptxSourceOff
    ++ ptxBytes ++ zeros (bindDescOff - ptxSourceOff - ptxBytes.length)
    ++ bindDesc ++ zeros (memSize - bindDescOff - bindDesc.length)
  let clifIr :=
    noopFunction ++ "\n" ++
    buildFunction 1 (loadFn n) ++ "\n" ++
    buildFunction 2 (prepFn n) ++ "\n" ++
    buildFunction 3 (inferFn output blockSize)
  let mkAlg (src : UInt32) : Algorithm :=
    { fn_idx := src }
  {
    config := {
      cranelift_ir := clifIr
      memory_size := memSize
      context_offset := 0
      initial_memory := initialMemory
    }
    loadAlgorithm  := mkAlg 1
    prepAlgorithm  := mkAlg 2
    inferAlgorithm := mkAlg 3
  }

def CompileResult.toArtifacts (r : CompileResult) (name : String) : Array Json :=
  #[
    -- `load` is the entry point (called first); prep/infer go to extras.
    toJsonArtifact name r.config r.loadAlgorithm [
      ("prep",  r.prepAlgorithm),
      ("infer", r.inferAlgorithm)
    ]
  ]

end CudaPipeline

end AlgorithmLib
