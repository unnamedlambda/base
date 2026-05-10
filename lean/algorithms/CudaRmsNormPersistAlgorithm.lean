import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.PTX

namespace CudaRmsNormPersist

def PTX_SOURCE_OFF : Nat := 0x0100
def BIND_DESC_OFF  : Nat := 0x1100
def MEM_SIZE       : Nat := 0x1200
def TIMEOUT_MS     : Nat := 30000

-- App fields in shared memory (after 0x38 reserved header)
def N_OFF      : Nat := 0x38   -- i64: element count N
def BUF0_OFF   : Nat := 0x40   -- i32: buf0 id (x + weights)
def BUF1_OFF   : Nat := 0x44   -- i32: buf1 id (output)

-- buf0 = [N:u32][x:N*f32][w:N*f32], buf1 = output y
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

/-
  Load: init CUDA, read N and weights from data, alloc 2 GPU bufs,
  upload N + weights into buf0.
  Shared memory app fields: N_OFF (i64), BUF0_OFF (i32), BUF1_OFF (i32)
-/
def loadFn : IRBuilder Unit := do
  let ptr  ← entryBlock
  let cuda ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)

  cudaInit cuda ptr 0x10
  let ctxPtr ← load64 (← absAddr ptr 0x10)

  -- Read N from data[0], store at N_OFF
  let n    ← load64 dataPtr
  storeI64 n (← absAddr ptr N_OFF)

  -- buf0 size = N*4 (input x) + 8 (N header) + N*4 (weights) = 8 + N*8
  let nBytes ← ishlImm n 2
  let buf0Sz ← iaddImm (← iadd nBytes nBytes) 8
  -- buf1 size = N*4 (output)
  let buf1Sz ← ishlImm n 2

  let buf0 ← call cuda.fnCreateBuffer [ctxPtr, buf0Sz]
  let buf1 ← call cuda.fnCreateBuffer [ctxPtr, buf1Sz]
  storeI32 buf0 (← absAddr ptr BUF0_OFF)
  storeI32 buf1 (← absAddr ptr BUF1_OFF)

  -- Upload N (8 bytes) to buf0 at offset 0
  let nAddr  ← absAddr ptr N_OFF
  let _ ← call cuda.fnUploadOffset [ctxPtr, buf0, ← iconst64 0, nAddr, ← iconst64 8]

  -- Upload weights (data[1..N], N*4 bytes) to buf0 at offset 8 + N*4
  let wSrc ← iaddImm dataPtr 8
  let wOff ← iaddImm nBytes 8
  let _ ← call cuda.fnUploadOffset [ctxPtr, buf0, wOff, wSrc, nBytes]
  ret

/-
  Prep: upload input x (data_ptr, N*4 bytes) to buf0 at offset 8.
-/
def prepFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let n       ← load64 (← absAddr ptr N_OFF)
  let buf0    ← load32 (← absAddr ptr BUF0_OFF)
  let ctxPtr  ← load64 (← absAddr ptr 0x10)
  let nBytes  ← ishlImm n 2
  let _ ← call cuda.fnUploadOffset [ctxPtr, buf0, ← iconst64 8, dataPtr, nBytes]
  ret

/-
  Infer: launch kernel (1 block, 256 threads), sync, optionally download buf1 to out_ptr.
-/
def inferFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let cuda   ← declareCudaFFI
  let outPtr ← load64 (← absAddr ptr 0x28)
  let outLen ← load64 (← absAddr ptr 0x30)
  let ctxPtr ← load64 (← absAddr ptr 0x10)
  let nBufs  ← iconst32 2
  let one32  ← iconst32 1
  let blk256 ← iconst32 256

  let skipDl     ← declareBlock []
  let doDownload ← declareBlock []

  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_SOURCE_OFF) nBufs
             (← iconst64 BIND_DESC_OFF) one32 one32 one32 blk256 one32 one32
  let _ ← cudaSync cuda ptr 0x10
  brif (← icmpImm .eq outLen 0) skipDl.ref [] doDownload.ref []

  startBlock doDownload
  let buf1 ← load32 (← absAddr ptr BUF1_OFF)
  let _ ← call cuda.fnDownload [ctxPtr, buf1, outPtr, outLen]
  ret

  startBlock skipDl
  ret

def clifIR : String :=
  noopFunction ++ "\n" ++
  buildFunction 1 loadFn ++ "\n" ++
  buildFunction 2 prepFn ++ "\n" ++
  buildFunction 3 inferFn

def ptxBytes : List UInt8 := ptxSource.toUTF8.toList ++ [0]
def bindDesc : List UInt8 := [0, 0, 0, 0, 1, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros 0x0100
  let ptx := ptxBytes ++ zeros (BIND_DESC_OFF - PTX_SOURCE_OFF - ptxBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ ptx ++ bind

def actions (src : UInt32) : List Action :=
  [{ kind := .ClifCall, dst := 0, src := src, offset := 0, size := 0 }]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0,
  initial_memory := buildInitialMemory
}

def loadAlgorithm : Algorithm := { actions := actions 1, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }
def prepAlgorithm : Algorithm := { actions := actions 2, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }
def inferAlgorithm : Algorithm := { actions := actions 3, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }

def artifacts : Array Json :=
  #[
    toJsonEntry "cuda_rmsnorm_load" buildConfig loadAlgorithm,
    toJsonEntry "cuda_rmsnorm_prep" buildConfig prepAlgorithm,
    toJsonEntry "cuda_rmsnorm_infer" buildConfig inferAlgorithm,
  ]

end CudaRmsNormPersist
