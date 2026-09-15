import Lean
import Std
import AlgorithmLib.Gen
import LayoutScan

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.PTX

namespace CudaRmsNormPersist

def PTX_SOURCE_OFF : Nat := 0x0100
def BIND_DESC_OFF  : Nat := 0x1100
def MEM_SIZE       : Nat := 0x1200

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
open AlgorithmLib.Prog


/-- The CUDA context pointer lives at a fixed slot in shared memory. -/
def CTX_OFF : Nat := 0x10

/-- Load: init CUDA, read N and weights from data, alloc 2 GPU bufs, upload
    N + weights into buf0. -/
def loadCode : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr

  cudaInit ptr CTX_OFF
  let ctxPtr ← load64 (← absAddr ptr CTX_OFF)

  -- Read N from data[0], store at N_OFF
  let n    ← load64 dataPtr
  store n (← absAddr ptr N_OFF)

  -- buf0 size = N*4 (input x) + 8 (N header) + N*4 (weights) = 8 + N*8
  let nBytes ← ishlImm n 2
  let buf0Sz ← iaddImm (← iadd nBytes nBytes) 8
  -- buf1 size = N*4 (output)
  let buf1Sz ← ishlImm n 2

  let buf0 ← ffi .cudaCreateBuffer %[ctxPtr, buf0Sz]
  let buf1 ← ffi .cudaCreateBuffer %[ctxPtr, buf1Sz]
  store buf0 (← absAddr ptr BUF0_OFF)
  store buf1 (← absAddr ptr BUF1_OFF)

  -- Upload N (8 bytes) to buf0 at offset 0
  let nAddr  ← absAddr ptr N_OFF
  let _ ← ffi .cudaUploadOffset %[ctxPtr, buf0, ← iconst64 0, nAddr, ← iconst64 8]

  -- Upload weights (data[1..N], N*4 bytes) to buf0 at offset 8 + N*4
  let wSrc ← iaddImm dataPtr 8
  let wOff ← iaddImm nBytes 8
  let _ ← ffi .cudaUploadOffset %[ctxPtr, buf0, wOff, wSrc, nBytes]

/-- Prep: upload input x (data_ptr, N*4 bytes) to buf0 at offset 8. -/
def prepCode : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let n       ← load64 (← absAddr ptr N_OFF)
  let buf0    ← load32 (← absAddr ptr BUF0_OFF)
  let ctxPtr  ← load64 (← absAddr ptr CTX_OFF)
  let nBytes  ← ishlImm n 2
  let _ ← ffi .cudaUploadOffset %[ctxPtr, buf0, ← iconst64 8, dataPtr, nBytes]

/-- Infer: launch the kernel (1 block, 256 threads), sync, and download only if
    the caller asked for output.

    The download branch joins rather than returning from each arm: `Code` has no
    early return, so both arms reach one `ret`. The join block holds nothing but
    that `ret`, which costs nothing once the backend threads the jump. -/
def inferCode : Prog V L Unit := do
  let ptr ← basePtr
  let outPtr ← outPtr
  let outLen ← outLen
  let ctxPtr ← load64 (← absAddr ptr CTX_OFF)
  let nBufs  ← iconst32 2
  let one32  ← iconst32 1
  let blk256 ← iconst32 256

  let _ ← cudaLaunch ptr (← iconst64 PTX_SOURCE_OFF) nBufs
             (← iconst64 BIND_DESC_OFF) one32 one32 one32 blk256 one32 one32
  let _ ← cudaSync ptr CTX_OFF
  let _ ← ifte .eq outLen (← iconst64 0)
    (thn := pure %[])
    (els := do
      let buf1 ← load32 (← absAddr ptr BUF1_OFF)
      let _ ← ffi .cudaDownload %[ctxPtr, buf1, outPtr, outLen]
      pure %[])
  return ()


def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     Prog.compileProg 1 loadCode,
     Prog.compileProg 2 prepCode,
     Prog.compileProg 3 inferCode]

def ptxBytes : List UInt8 := ptxSource.toUTF8.toList ++ [0]
def bindDesc : List UInt8 := [0, 0, 0, 0, 1, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros 0x0100
  let ptx := ptxBytes ++ zeros (BIND_DESC_OFF - PTX_SOURCE_OFF - ptxBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ ptx ++ bind

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the runtime fills and `0x18`-`0x38` the
    input and output descriptors, so naming those is what stops an offset being
    placed where the runtime will overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"ctx_cuda",   CTX_OFF, 8⟩,
   ⟨"n",          N_OFF, 8⟩,
   ⟨"buf0",       BUF0_OFF, 4⟩,
   ⟨"buf1",       BUF1_OFF, 4⟩,
   ⟨"ptx",        PTX_SOURCE_OFF, BIND_DESC_OFF - PTX_SOURCE_OFF⟩,
   ⟨"bind",       BIND_DESC_OFF, MEM_SIZE - BIND_DESC_OFF⟩]


#eval LayoutScan.check "CudaRmsNormPersistAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def buildSetup (clif : List FuncData) : Artifact := {
  functions := clif,
  memory_size := MEM_SIZE,
  initial_memory := buildInitialMemory
}

def loadAlgorithm : UInt32 := 1
def prepAlgorithm : UInt32 := 2
def inferAlgorithm : UInt32 := 3

def artifacts (clif : List FuncData) : Array Json :=
  #[
    toJsonArtifact "cuda_rmsnorm" (buildSetup clif)
  ]


end CudaRmsNormPersist
