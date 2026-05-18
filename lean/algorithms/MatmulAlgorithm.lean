import AlgorithmLib
set_option maxRecDepth 4096
open Lean (Json toJson)
open AlgorithmLib
open AlgorithmLib.PTX

namespace Matmul

-- ===========================================================================
-- Dependent-type interface for matrix multiplication.
--
-- The Matrix type carries its shape at the type level. Incompatible shapes
-- are rejected at Lean elaboration time, before any PTX or CLIF is generated.
--
--   matmul {m k n} (A : Matrix m k) (B : Matrix k n) : Setup × Algorithm
-- ===========================================================================

structure Matrix (m n : Nat) : Type where
  mk ::

def mkMatrix (m n : Nat) : Matrix m n := Matrix.mk

-- ---------------------------------------------------------------------------
-- Deterministic PRNG: xorshift32
-- ---------------------------------------------------------------------------

def xorshift32 (s : UInt32) : UInt32 :=
  let s := s ^^^ (s <<< 13)
  let s := s ^^^ (s >>> 17)
  s ^^^ (s <<< 5)

-- f32 bit pattern in ±[0.5, 1.0):
--   sign bit from MSB, fixed biased exponent 126 (= 2^-1), 23-bit mantissa
def randF32Bits (s : UInt32) : UInt32 :=
  let sign := s &&& 0x80000000
  let mantissa := s &&& 0x007FFFFF
  let exp : UInt32 := (126 : UInt32) <<< 23
  sign ||| exp ||| mantissa

-- Tail-recursive matrix byte generator.
-- Pre-allocates a ByteArray and writes little-endian f32 bit patterns.
def genMatrixBytesBA (count : Nat) (seed : UInt32) : ByteArray :=
  let rec go : Nat → UInt32 → ByteArray → ByteArray
    | 0, _, acc => acc
    | Nat.succ n, s, acc =>
      let s' := xorshift32 s
      let bits := randF32Bits s'
      let v := bits.toNat
      let acc := acc.push (UInt8.ofNat (v &&& 0xFF))
      let acc := acc.push (UInt8.ofNat ((v >>> 8) &&& 0xFF))
      let acc := acc.push (UInt8.ofNat ((v >>> 16) &&& 0xFF))
      let acc := acc.push (UInt8.ofNat ((v >>> 24) &&& 0xFF))
      go n s' acc
  go count seed (ByteArray.mk (Array.mkEmpty (count * 4)))

def genMatrixBytes (count : Nat) (seed : UInt32) : List UInt8 :=
  (genMatrixBytesBA count seed).toList

-- ---------------------------------------------------------------------------
-- PTX kernel: tiled 16x16 GEMM with shared-memory blocking
--
-- Grid: (ceil(N/16), ceil(M/16)), block: (16, 16).
-- Each block computes a 16x16 sub-block of C.
-- Shared memory holds one 16x16 tile of A and one 16x16 tile of B.
-- ---------------------------------------------------------------------------

def ptxSource : String := buildModuleWith
  { version := "7.0", target := "sm_50", smemSize := 2048, smemAlign := 16, smemName := "tiles" }
  [{ name := "main", params := ["p_a", "p_b", "p_c", "p_par"], body := do
    let pA   ← ldParam "p_a"
    let pB   ← ldParam "p_b"
    let pC   ← ldParam "p_c"
    let pPar ← ldParam "p_par"
    -- Load M, K, N from params buffer
    let mReg ← freshR; ldGlobalU  mReg pPar
    let kReg ← freshR; ldGlobalUO kReg pPar 4
    let nReg ← freshR; ldGlobalUO nReg pPar 8
    -- Thread/block coordinates
    let tx  ← freshR; movR tx  tidX
    let ty  ← freshR; movR ty  tidY
    let bx  ← freshR; movR bx  ctaX
    let by_ ← freshR; movR by_ ctaY
    -- row = by*16 + ty,  col = bx*16 + tx
    let row ← freshR; shlR row by_ 4; addR row row ty
    let col ← freshR; shlR col bx  4; addR col col tx
    -- Shared memory: A_tile at tiles[0], B_tile at tiles[1024]
    let aTile ← smemBaseNamed "tiles"
    let bTile ← freshR; addRI bTile aTile 1024
    -- Per-thread smem offset: (ty*16 + tx) * 4
    let tOff ← freshR; shlR tOff ty 4; addR tOff tOff tx; shlR tOff tOff 2
    -- Accumulator
    let acc ← freshF; movFC acc f32_0
    -- Outer loop: step through K in tiles of 16
    let tileK ← freshR; movRC tileK 0
    label "L_OUTER"
    let p0 ← freshP; setpGe p0 tileK kReg; braIf p0 "L_WRITE"
    -- Load A_tile[ty][tx] = A[row*K + tile_k+tx], or 0.0 if out of bounds
    let p1 ← freshP; setpLt p1 row mReg
    let tkTx ← freshR; addR tkTx tileK tx
    let p2 ← freshP; setpLt p2 tkTx kReg
    let p3 ← freshP; andPred p3 p1 p2
    let f1 ← freshF
    braIfNot p3 "L_A_ZERO"
    let r14 ← freshR; mulLoR r14 row kReg; addR r14 r14 tkTx
    let rd4 ← freshRd; cvtU64 rd4 r14; shlRd rd4 rd4 2; addRd rd4 pA rd4
    ldGlobalF f1 rd4
    bra "L_A_STORE"
    label "L_A_ZERO"; movFC f1 f32_0
    label "L_A_STORE"
    let r15 ← freshR; addR r15 aTile tOff; stSharedFD r15 f1
    -- Load B_tile[ty][tx] = B[(tile_k+ty)*N + col], or 0.0 if out of bounds
    let tkTy ← freshR; addR tkTy tileK ty
    let p4 ← freshP; setpLt p4 tkTy kReg
    let p5 ← freshP; setpLt p5 col nReg
    let p6 ← freshP; andPred p6 p4 p5
    let f2 ← freshF
    braIfNot p6 "L_B_ZERO"
    let r17 ← freshR; mulLoR r17 tkTy nReg; addR r17 r17 col
    let rd5 ← freshRd; cvtU64 rd5 r17; shlRd rd5 rd5 2; addRd rd5 pB rd5
    ldGlobalF f2 rd5
    bra "L_B_STORE"
    label "L_B_ZERO"; movFC f2 f32_0
    label "L_B_STORE"
    let r18 ← freshR; addR r18 bTile tOff; stSharedFD r18 f2
    barSync
    -- Inner accumulate: acc += A_tile[ty][kk] * B_tile[kk][tx]
    let aRowBase ← freshR; shlR aRowBase ty 6; addR aRowBase aTile aRowBase
    let bColBase ← freshR; shlR bColBase tx 2; addR bColBase bTile bColBase
    let kk ← freshR; movRC kk 0
    label "L_INNER"
    let p7 ← freshP; setpGeI p7 kk 16; braIf p7 "L_INNER_END"
    let kkBytes  ← freshR; shlR kkBytes kk 2
    let aAddr    ← freshR; addR aAddr aRowBase kkBytes
    let f3 ← freshF; ldSharedFD f3 aAddr
    let kkStride ← freshR; shlR kkStride kk 6
    let bAddr    ← freshR; addR bAddr bColBase kkStride
    let f4 ← freshF; ldSharedFD f4 bAddr
    fmaRn acc f3 f4 acc
    addRI kk kk 1; bra "L_INNER"
    label "L_INNER_END"
    barSync
    addRI tileK tileK 16; bra "L_OUTER"
    -- Bounds-checked write: C[row*N + col] = acc
    label "L_WRITE"
    let pw1 ← freshP; setpGe pw1 row mReg; braIf pw1 "L_DONE"
    let pw2 ← freshP; setpGe pw2 col nReg; braIf pw2 "L_DONE"
    let r26 ← freshR; mulLoR r26 row nReg; addR r26 r26 col
    let rd6 ← freshRd; cvtU64 rd6 r26; shlRd rd6 rd6 2; addRd rd6 pC rd6
    stGlobalF rd6 acc
    label "L_DONE"
    ptxRet }]

-- ---------------------------------------------------------------------------
-- Layout constants (fixed region offsets; data region base depends on m,k,n)
-- ---------------------------------------------------------------------------

def PTX_OFF : Nat := 0x0100
def PTX_REGION : Nat := 8192
def BIND_OFF : Nat := PTX_OFF + PTX_REGION            -- 0x2100
def PARAMS_OFF : Nat := BIND_OFF + 16                  -- 0x2110
def OUTPUT_FN_OFF : Nat := PARAMS_OFF + 16             -- 0x2120
def DATA_OFF : Nat := 0x3000

-- ---------------------------------------------------------------------------
-- Payload: PTX source, bind desc, params, output filename, random A & B data
-- ---------------------------------------------------------------------------

def buildPayload (m k n : Nat) : List UInt8 :=
  let reserved := zeros PTX_OFF
  let ptxBytes := padTo (stringToBytes ptxSource) PTX_REGION
  -- bind desc: 4 buffer IDs [0=A, 1=B, 2=C, 3=params]
  let bindDesc :=
    uint32ToBytes 0 ++ uint32ToBytes 1 ++
    uint32ToBytes 2 ++ uint32ToBytes 3
  -- params [M, K, N, pad]
  let params :=
    uint32ToBytes (UInt32.ofNat m) ++
    uint32ToBytes (UInt32.ofNat k) ++
    uint32ToBytes (UInt32.ofNat n) ++
    uint32ToBytes 0
  let outFn := padTo (stringToBytes "matmul_output.bin") 256
  let preData := reserved ++ ptxBytes ++ bindDesc ++ params ++ outFn
  let prePad := zeros (DATA_OFF - preData.length)
  -- Random matrices: distinct seeds so A ≠ B
  let aData := genMatrixBytes (m * k) 0xC0FFEE01
  let bData := genMatrixBytes (k * n) 0xDEADBEEF
  -- C output region (written by CLIF after GPU download)
  let cData := zeros (m * n * 4)
  preData ++ prePad ++ aData ++ bData ++ cData

-- ---------------------------------------------------------------------------
-- CLIF orchestrator: init CUDA, create 4 buffers, upload, launch, download,
-- cleanup, write output file. No timestep loop — one kernel, one result.
-- ---------------------------------------------------------------------------

open AlgorithmLib.IR in
def clifIrSource (m k n : Nat) : String :=
  let aBytes := m * k * 4
  let bBytes := k * n * 4
  let cBytes := m * n * 4
  let aOff := DATA_OFF
  let bOff := aOff + aBytes
  let cOff := bOff + bBytes
  buildProgram do
    let fnWrite ← declareFileWrite
    let cuda ← declareCudaFFI

    let ptr ← entryBlock
    let c0 ← iconst64 0

    -- CUDA init
    cudaInit cuda ptr

    -- Create 4 buffers
    let aSz ← iconst64 aBytes
    let bSz ← iconst64 bBytes
    let cSz ← iconst64 cBytes
    let pSz ← iconst64 16
    let bufA ← cudaCreateBuffer cuda ptr aSz
    let bufB ← cudaCreateBuffer cuda ptr bSz
    let bufC ← cudaCreateBuffer cuda ptr cSz
    let bufP ← cudaCreateBuffer cuda ptr pSz
    let _ := bufA; let _ := bufB; let _ := bufC; let _ := bufP

    -- Upload A, B, params
    let aOffV ← iconst64 aOff
    let bOffV ← iconst64 bOff
    let pOffV ← iconst64 PARAMS_OFF
    let _ ← cudaUpload cuda ptr bufA aOffV aSz
    let _ ← cudaUpload cuda ptr bufB bOffV bSz
    let _ ← cudaUpload cuda ptr bufP pOffV pSz

    -- Launch: grid = (ceil(N/16), ceil(M/16), 1), block = (16, 16, 1)
    let gridX := (n + 15) / 16
    let gridY := (m + 15) / 16
    let ptxOff ← iconst64 PTX_OFF
    let nBufs ← iconst32 4
    let bindOff ← iconst64 BIND_OFF
    let gx ← iconst32 gridX
    let gy ← iconst32 gridY
    let one32 ← iconst32 1
    let blk16 ← iconst32 16
    let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff gx gy one32 blk16 blk16 one32

    -- Download C
    let cOffV ← iconst64 cOff
    let _ ← cudaDownload cuda ptr bufC cOffV cSz

    cudaCleanup cuda ptr

    -- Write output file: bytes [cOff .. cOff + cBytes)
    let fnOffV ← iconst64 OUTPUT_FN_OFF
    let _ ← call fnWrite [ptr, fnOffV, cOffV, c0, cSz]
    ret

-- ---------------------------------------------------------------------------
-- Monomorphic builder: takes concrete dims, returns (Setup, Algorithm).
-- ---------------------------------------------------------------------------

def buildMatmulConfig (m k n : Nat) : Setup × Algorithm :=
  let payload := buildPayload m k n
  let memSize := payload.length
  let cfg : Setup := {
    cranelift_ir := clifIrSource m k n,
    memory_size := memSize,
    initial_memory := payload
  }
  let alg : Algorithm := {
    fn_idx := IR.mainFnIdx
  }
  (cfg, alg)

-- ---------------------------------------------------------------------------
-- Typed matmul: the dependent-type surface.
-- Passing incompatible shapes fails at Lean elaboration time.
-- ---------------------------------------------------------------------------

def matmul {m k n : Nat} (_A : Matrix m k) (_B : Matrix k n) : Setup × Algorithm :=
  buildMatmulConfig m k n

-- ---------------------------------------------------------------------------
-- Demo: shapes fixed at Lean compile time.
-- Change M, K, N to generate a different sized matmul (type-checked).
-- ---------------------------------------------------------------------------

def M : Nat := 1024
def K : Nat := 512
def N : Nat := 256

def A : Matrix M K := mkMatrix M K
def B : Matrix K N := mkMatrix K N

def result : Setup × Algorithm := matmul A B  -- : Matrix M N (erased)

-- Uncomment to see the dependent-type check in action:
--
--   def bad : Setup × Algorithm := matmul A (mkMatrix 128 128)
--   -- error: type mismatch
--   --   mkMatrix 128 128 has type Matrix 128 128
--   --   but is expected to have type Matrix K ?n

end Matmul

def main (args : List String) : IO Unit := do
  let (cfg, alg) := Matmul.result
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[toJsonEntry "matmul_app" cfg alg]
