import AlgorithmLib
open Lean (Json toJson)
open AlgorithmLib

namespace Matmul

-- ===========================================================================
-- Dependent-type interface for matrix multiplication.
--
-- The Matrix type carries its shape at the type level. Incompatible shapes
-- are rejected at Lean elaboration time, before any PTX or CLIF is generated.
--
--   matmul {m k n} (A : Matrix m k) (B : Matrix k n) : BaseConfig × Algorithm
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

def ptxSource : String :=
  ".version 7.0\n" ++
  ".target sm_50\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 p_a,\n" ++
  "    .param .u64 p_b,\n" ++
  "    .param .u64 p_c,\n" ++
  "    .param .u64 p_par\n" ++
  ")\n" ++
  "{\n" ++
  "    .shared .align 16 .b8 tiles[2048];\n" ++  -- A_tile (1024) + B_tile (1024)
  "    .reg .u32 %r<30>;\n" ++
  "    .reg .u64 %rd<10>;\n" ++
  "    .reg .f32 %f<6>;\n" ++
  "    .reg .pred %p<8>;\n" ++
  "\n" ++
  -- Kernel params
  "    ld.param.u64 %rd0, [p_a];\n" ++
  "    ld.param.u64 %rd1, [p_b];\n" ++
  "    ld.param.u64 %rd2, [p_c];\n" ++
  "    ld.param.u64 %rd3, [p_par];\n" ++
  "    ld.global.u32 %r0, [%rd3];\n" ++       -- M
  "    ld.global.u32 %r1, [%rd3+4];\n" ++     -- K
  "    ld.global.u32 %r2, [%rd3+8];\n" ++     -- N
  "\n" ++
  -- Thread/block coords
  "    mov.u32 %r3, %tid.x;\n" ++             -- tx
  "    mov.u32 %r4, %tid.y;\n" ++             -- ty
  "    mov.u32 %r5, %ctaid.x;\n" ++           -- bx
  "    mov.u32 %r6, %ctaid.y;\n" ++           -- by
  "    shl.b32 %r7, %r6, 4;\n" ++             -- by*16
  "    add.u32 %r7, %r7, %r4;\n" ++           -- row = by*16 + ty
  "    shl.b32 %r8, %r5, 4;\n" ++             -- bx*16
  "    add.u32 %r8, %r8, %r3;\n" ++           -- col = bx*16 + tx
  "\n" ++
  -- Shared memory bases and my per-thread tile offset
  "    mov.u32 %r9, tiles;\n" ++              -- A_tile base
  "    add.u32 %r10, %r9, 1024;\n" ++         -- B_tile base
  "    shl.b32 %r11, %r4, 4;\n" ++            -- ty*16
  "    add.u32 %r11, %r11, %r3;\n" ++         -- ty*16 + tx
  "    shl.b32 %r11, %r11, 2;\n" ++           -- *4 bytes
  "\n" ++
  -- Accumulator
  "    mov.f32 %f0, 0f00000000;\n" ++
  "\n" ++
  -- Outer loop over K, step 16
  "    mov.u32 %r12, 0;\n" ++                 -- tile_k
  "$L_OUTER:\n" ++
  "    setp.ge.u32 %p0, %r12, %r1;\n" ++
  "    @%p0 bra $L_WRITE;\n" ++
  "\n" ++
  -- Load A_tile[ty][tx] = A[row * K + tile_k + tx]
  "    setp.lt.u32 %p1, %r7, %r0;\n" ++       -- row < M
  "    add.u32 %r13, %r12, %r3;\n" ++         -- tile_k + tx
  "    setp.lt.u32 %p2, %r13, %r1;\n" ++      -- tile_k + tx < K
  "    and.pred %p3, %p1, %p2;\n" ++
  "    @!%p3 bra $L_A_ZERO;\n" ++
  "    mul.lo.u32 %r14, %r7, %r1;\n" ++
  "    add.u32 %r14, %r14, %r13;\n" ++
  "    cvt.u64.u32 %rd4, %r14;\n" ++
  "    shl.b64 %rd4, %rd4, 2;\n" ++
  "    add.u64 %rd4, %rd0, %rd4;\n" ++
  "    ld.global.f32 %f1, [%rd4];\n" ++
  "    bra $L_A_STORE;\n" ++
  "$L_A_ZERO:\n" ++
  "    mov.f32 %f1, 0f00000000;\n" ++
  "$L_A_STORE:\n" ++
  "    add.u32 %r15, %r9, %r11;\n" ++         -- &A_tile[ty][tx]
  "    st.shared.f32 [%r15], %f1;\n" ++
  "\n" ++
  -- Load B_tile[ty][tx] = B[(tile_k + ty) * N + col]
  "    add.u32 %r16, %r12, %r4;\n" ++         -- tile_k + ty
  "    setp.lt.u32 %p4, %r16, %r1;\n" ++      -- tile_k + ty < K
  "    setp.lt.u32 %p5, %r8, %r2;\n" ++       -- col < N
  "    and.pred %p6, %p4, %p5;\n" ++
  "    @!%p6 bra $L_B_ZERO;\n" ++
  "    mul.lo.u32 %r17, %r16, %r2;\n" ++
  "    add.u32 %r17, %r17, %r8;\n" ++
  "    cvt.u64.u32 %rd5, %r17;\n" ++
  "    shl.b64 %rd5, %rd5, 2;\n" ++
  "    add.u64 %rd5, %rd1, %rd5;\n" ++
  "    ld.global.f32 %f2, [%rd5];\n" ++
  "    bra $L_B_STORE;\n" ++
  "$L_B_ZERO:\n" ++
  "    mov.f32 %f2, 0f00000000;\n" ++
  "$L_B_STORE:\n" ++
  "    add.u32 %r18, %r10, %r11;\n" ++
  "    st.shared.f32 [%r18], %f2;\n" ++
  "\n" ++
  "    bar.sync 0;\n" ++
  "\n" ++
  -- Inner accumulate: acc += A_tile[ty][kk] * B_tile[kk][tx]
  "    shl.b32 %r19, %r4, 6;\n" ++            -- ty * 16 * 4 = ty*64 (A row base)
  "    add.u32 %r19, %r9, %r19;\n" ++
  "    shl.b32 %r20, %r3, 2;\n" ++            -- tx * 4 (B col offset)
  "    add.u32 %r20, %r10, %r20;\n" ++
  "    mov.u32 %r21, 0;\n" ++                 -- kk
  "$L_INNER:\n" ++
  "    setp.ge.u32 %p7, %r21, 16;\n" ++
  "    @%p7 bra $L_INNER_END;\n" ++
  "    shl.b32 %r22, %r21, 2;\n" ++           -- kk * 4
  "    add.u32 %r23, %r19, %r22;\n" ++
  "    ld.shared.f32 %f3, [%r23];\n" ++       -- A_tile[ty][kk]
  "    shl.b32 %r24, %r21, 6;\n" ++           -- kk * 64
  "    add.u32 %r25, %r20, %r24;\n" ++
  "    ld.shared.f32 %f4, [%r25];\n" ++       -- B_tile[kk][tx]
  "    fma.rn.f32 %f0, %f3, %f4, %f0;\n" ++
  "    add.u32 %r21, %r21, 1;\n" ++
  "    bra $L_INNER;\n" ++
  "$L_INNER_END:\n" ++
  "    bar.sync 0;\n" ++
  "\n" ++
  "    add.u32 %r12, %r12, 16;\n" ++
  "    bra $L_OUTER;\n" ++
  "\n" ++
  -- Bounds-checked write: C[row * N + col] = acc
  "$L_WRITE:\n" ++
  "    setp.ge.u32 %p1, %r7, %r0;\n" ++
  "    @%p1 bra $L_DONE;\n" ++
  "    setp.ge.u32 %p2, %r8, %r2;\n" ++
  "    @%p2 bra $L_DONE;\n" ++
  "    mul.lo.u32 %r26, %r7, %r2;\n" ++
  "    add.u32 %r26, %r26, %r8;\n" ++
  "    cvt.u64.u32 %rd6, %r26;\n" ++
  "    shl.b64 %rd6, %rd6, 2;\n" ++
  "    add.u64 %rd6, %rd2, %rd6;\n" ++
  "    st.global.f32 [%rd6], %f0;\n" ++
  "\n" ++
  "$L_DONE:\n" ++
  "    ret;\n" ++
  "}\n"

-- ---------------------------------------------------------------------------
-- Layout constants (fixed region offsets; data region base depends on m,k,n)
-- ---------------------------------------------------------------------------

def PTX_OFF : Nat := 0x0100
def PTX_REGION : Nat := 8192
def BIND_OFF : Nat := PTX_OFF + PTX_REGION            -- 0x2100
def PARAMS_OFF : Nat := BIND_OFF + 16                  -- 0x2110
def OUTPUT_FN_OFF : Nat := PARAMS_OFF + 16             -- 0x2120
def DATA_OFF : Nat := 0x3000
def TIMEOUT_MS : Nat := 300000

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
    callVoid cuda.fnInit [ptr]

    -- Create 4 buffers
    let aSz ← iconst64 aBytes
    let bSz ← iconst64 bBytes
    let cSz ← iconst64 cBytes
    let pSz ← iconst64 16
    let bufA ← call cuda.fnCreateBuffer [ptr, aSz]
    let bufB ← call cuda.fnCreateBuffer [ptr, bSz]
    let bufC ← call cuda.fnCreateBuffer [ptr, cSz]
    let bufP ← call cuda.fnCreateBuffer [ptr, pSz]
    let _ := bufA; let _ := bufB; let _ := bufC; let _ := bufP

    -- Upload A, B, params
    let aOffV ← iconst64 aOff
    let bOffV ← iconst64 bOff
    let pOffV ← iconst64 PARAMS_OFF
    let _ ← call cuda.fnUpload [ptr, bufA, aOffV, aSz]
    let _ ← call cuda.fnUpload [ptr, bufB, bOffV, bSz]
    let _ ← call cuda.fnUpload [ptr, bufP, pOffV, pSz]

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
    let _ ← call cuda.fnLaunch
      [ptr, ptxOff, nBufs, bindOff, gx, gy, one32, blk16, blk16, one32]

    -- Download C
    let cOffV ← iconst64 cOff
    let _ ← call cuda.fnDownload [ptr, bufC, cOffV, cSz]

    callVoid cuda.fnCleanup [ptr]

    -- Write output file: bytes [cOff .. cOff + cBytes)
    let fnOffV ← iconst64 OUTPUT_FN_OFF
    let _ ← call fnWrite [ptr, fnOffV, cOffV, c0, cSz]
    ret

-- ---------------------------------------------------------------------------
-- Monomorphic builder: takes concrete dims, returns (BaseConfig, Algorithm).
-- ---------------------------------------------------------------------------

def buildMatmulConfig (m k n : Nat) : BaseConfig × Algorithm :=
  let payload := buildPayload m k n
  let memSize := payload.length
  let cfg : BaseConfig := {
    cranelift_ir := clifIrSource m k n,
    memory_size := memSize,
    context_offset := 0,
    initial_memory := payload
  }
  let alg : Algorithm := {
    actions := [IR.clifCallAction],
    cranelift_units := 0,
    timeout_ms := some TIMEOUT_MS
  }
  (cfg, alg)

-- ---------------------------------------------------------------------------
-- Typed matmul: the dependent-type surface.
-- Passing incompatible shapes fails at Lean elaboration time.
-- ---------------------------------------------------------------------------

def matmul {m k n : Nat} (_A : Matrix m k) (_B : Matrix k n) : BaseConfig × Algorithm :=
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

def result : BaseConfig × Algorithm := matmul A B  -- : Matrix M N (erased)

-- Uncomment to see the dependent-type check in action:
--
--   def bad : BaseConfig × Algorithm := matmul A (mkMatrix 128 128)
--   -- error: type mismatch
--   --   mkMatrix 128 128 has type Matrix 128 128
--   --   but is expected to have type Matrix K ?n

end Matmul

def main : IO Unit := do
  let (cfg, alg) := Matmul.result
  IO.println (Json.compress (toJsonPair cfg alg))
