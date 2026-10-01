module
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Vocab.WGSL
meta import AlgorithmLib.Vocab.WGSL
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean (Json toJson)
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.WGSL

namespace Fft

-- ---------------------------------------------------------------------------
-- GPU-accelerated FFT (Cooley-Tukey radix-2 decimation-in-time)
--
-- Input: binary file of f32 pairs (re, im) — N complex numbers, N = power of 2
-- Output: binary file of f32 pairs (re, im) — the DFT result
--
-- GPU strategy: log2(N) passes, each dispatched separately.
-- Each pass performs N/2 butterfly operations in parallel.
-- Uses ping-pong between two buffers (buf0 → buf1 → buf0 → ...).
-- Metadata buffer holds: [N: u32, stage: u32, direction: u32]
--   direction: 0 = read buf0/write buf1, 1 = read buf1/write buf0
--
-- Memory layout:
--   [0x0000..0x0100)  reserved / scratch
--   [0x0100..0x0200)  binding descriptors (4 bindings × 8 bytes)
--   [0x0200..0x2200)  WGSL shader (8KB)
--   [0x2200..0x2300)  input filename
--   [0x2300..0x2400)  output filename "fft_output.bin"
--   [0x2400..0x2440)  flags + padding
--   [0x2440..0x4440)  CLIF IR region (8KB)
--   [0x4440+)         data regions (input, buf_a, buf_b, meta)
-- ---------------------------------------------------------------------------

def maxN : Nat := 1024 * 1024  -- max 1M complex numbers
def maxDataSize : Nat := maxN * 8  -- 8 bytes per complex (2 × f32)
def metaSize : Nat := 16  -- N, stage, direction, padding (aligned to 4)

-- Offsets
def bindDesc_off : Nat := 0x100
def shader_off : Nat := 0x200
def shaderRegionSize : Nat := 8192
def inputFilename_off : Nat := shader_off + shaderRegionSize  -- 0x2200
def filenameRegionSize : Nat := 256
def outputFilename_off : Nat := inputFilename_off + filenameRegionSize  -- 0x2300
def flag_off : Nat := outputFilename_off + filenameRegionSize  -- 0x2400
def clifIr_off : Nat := flag_off + 64  -- 0x2440
def clifIrRegionSize : Nat := 8192
def inputData_off : Nat := clifIr_off + clifIrRegionSize  -- 0x4440
def bufA_off : Nat := inputData_off + maxDataSize
def bufB_off : Nat := bufA_off + maxDataSize
def meta_off : Nat := bufB_off + maxDataSize
def totalAdditionalMemory : Nat := maxDataSize * 3 + metaSize

-- ---------------------------------------------------------------------------
-- WGSL compute shader: FFT butterfly pass
--
-- Each invocation handles one butterfly.
-- N/2 invocations per dispatch, workgroup_size(64).
-- Reads from buf_a or buf_b depending on direction, writes to the other.
-- ---------------------------------------------------------------------------

def fftShader : String :=
  let bufA   : AlgorithmLib.WGSL.Expr (.arr .vec2f) := ⟨"buf_a"⟩
  let bufB   : AlgorithmLib.WGSL.Expr (.arr .vec2f) := ⟨"buf_b"⟩
  let params : AlgorithmLib.WGSL.Expr (.arr .u32)   := ⟨"params"⟩
  buildShader
    [{ binding := 0, name := "buf_a",  ty := .arr .vec2f },
     { binding := 1, name := "buf_b",  ty := .arr .vec2f },
     { binding := 2, name := "params", ty := .arr .u32, ro := true }]
    []
    [.constF "PI" "3.14159265358979323846"]
    {}
    do
      let n         ← letV (arrIdx params (litU 0))
      let stage     ← letV (arrIdx params (litU 1))
      let direction ← letV (arrIdx params (litU 2))
      let halfN     ← letV (n / litU 2)
      let tid       ← letV gidX
      ifB (tid .>= halfN) retV
      let halfBlock ← letV ((litU 1) .<< stage)
      let blockSize ← letV (halfBlock .<< litU 1)
      let blockId   ← letV (tid / halfBlock)
      let j         ← letV (tid % halfBlock)
      let iTop      ← letV (blockId * blockSize + j)
      let iBot      ← letV (iTop + halfBlock)
      let angle     ← letV (-litF "2.0" * ⟨"PI"⟩ * f32OfU j / f32OfU blockSize)
      let tw        ← letV (mkVec2f (wCos angle) (wSin angle))
      let aVal      ← varVT .vec2f
      let bVal      ← varVT .vec2f
      ifElse (direction .== litU 0)
        (do
          assign aVal (arrIdx bufA iTop)
          assign bVal (arrIdx bufA iBot))
        (do
          assign aVal (arrIdx bufB iTop)
          assign bVal (arrIdx bufB iBot))
      let tb ← letV (mkVec2f
        (v2x tw * v2x bVal - v2y tw * v2y bVal)
        (v2x tw * v2y bVal + v2y tw * v2x bVal))
      let outTop ← letV (aVal + tb)
      let outBot ← letV (aVal - tb)
      ifElse (direction .== litU 0)
        (do
          assign (arrIdx bufB iTop) outTop
          assign (arrIdx bufB iBot) outBot)
        (do
          assign (arrIdx bufA iTop) outTop
          assign (arrIdx bufA iBot) outBot)

-- ---------------------------------------------------------------------------
-- CLIF IR orchestrator
--
-- 1. Read input file → inputData region
-- 2. Bit-reverse permutation (CPU, in CLIF) → buf_a region
-- 3. GPU init, create 3 buffers (buf_a, buf_b, meta)
-- 4. Upload buf_a to GPU
-- 5. Loop log2(N) stages: update meta, upload meta, dispatch, toggle direction
-- 6. Download result from final buffer
-- 7. GPU cleanup
-- 8. Write output file
-- ---------------------------------------------------------------------------

open AlgorithmLib.Prog


abbrev fnWrite : Ffi := .fileWrite

def code : Prog V L Unit := do
  let ptr ← basePtr
  -- FFI declarations

  let c0  ← iconst64 0
  let c1  ← iconst64 1
  let c4  ← iconst64 4
  let c8  ← iconst64 8

  -- Step 1: Read input file
  let inDatOff ← iconst64 inputData_off
  let bytesRead ← readFile ptr inputFilename_off inputData_off maxDataSize

  -- A read that failed (-1), ran past the input region, or holds no complete
  -- number computes nothing: a write of size 0 would write up to a NUL.
  let maxC ← iconst64 maxDataSize
  let _ ← ifte .ugt bytesRead maxC (pure %[]) (do
   let _ ← ifte .ult bytesRead c8 (pure %[]) (do
    -- Compute N = bytes_read / 8
    let c3 ← iconst64 3
    let bigN ← ushr bytesRead c3

    -- Step 2: log2(N), the position of N's highest bit: 63 less its leading zeros.
    let lz ← clz bigN
    let log2Result ← isub (← iconst64 63) lz

    let bufAOff ← iconst64 bufA_off

    -- Bit-reversed order, gathered: bufA[j] = inputData[rev(j)], `rev`
    -- reversing j's low log2(N) bits (an involution for N a power of two).
    -- The top bits of the 64-bit reversal, shifted down by 64 - log2(N);
    -- N = 1 shifts by 64, which is 0, and reverses only 0.
    let revShift ← iadd lz c1
    forLoop bigN fun j => do
      let revIdx ← ushr (← bitrev j) revShift
      let srcAbs ← iadd ptr (← iadd inDatOff (← imul revIdx c8))
      let dstAbs ← iadd ptr (← iadd bufAOff (← imul j c8))
      storeUnaligned (← load32 srcAbs) dstAbs
      storeUnaligned (← load32 (← iadd srcAbs c4)) (← iadd dstAbs c4)

    gpuInit ptr

    -- N*8 bytes: already the multiple of 4 wgpu asks of a buffer
    let dataSz ← imul bigN c8
    let alignedSz := dataSz

    -- Create 3 buffers
    let buf0 ← gpuCreateBuffer ptr alignedSz
    let buf1 ← gpuCreateBuffer ptr alignedSz
    let metaSzC ← iconst64 metaSize
    let buf2 ← gpuCreateBuffer ptr metaSzC

    -- Write N into meta region
    let metaOffC ← iconst64 meta_off
    let metaAbs ← iadd ptr metaOffC
    let nI32 ← ireduce32 bigN
    store nI32 metaAbs

    -- Upload buf_a
    let _ ← gpuUpload ptr buf0 bufAOff alignedSz

    -- Create pipeline (3 bindings)
    let shOffC ← iconst64 shader_off
    let bdOffC ← iconst64 bindDesc_off
    let c3_i32 ← iconst32 3
    let pipeId ← gpuCreatePipeline ptr shOffC bdOffC c3_i32

    -- Compute dispatch size: ceil(N/2 / 64)
    let halfN ← ushr bigN c1
    let c63 ← iconst64 63
    let halfPad ← iadd halfN c63
    let c6 ← iconst64 6
    let wgCount ← ushr halfPad c6
    let wgCount32 ← ireduce32 wgCount
    let one32 ← iconst32 1

    -- Step 4: Stage loop — counter `stage` for log2Result iterations,
    -- accumulator `dir` toggled each iteration.
    let finalDir ← forLoopAcc log2Result c0 fun stage dir => do
      -- Write stage and direction into meta
      let metaStage ← iadd metaAbs c4
      storeUnaligned (← ireduce32 stage) metaStage
      let metaDir ← iadd metaStage c4
      storeUnaligned (← ireduce32 dir) metaDir
      -- Upload meta, dispatch, then sync via download-to-scratch
      let _ ← gpuUpload ptr buf2 metaOffC metaSzC
      let _ ← gpuDispatch ptr pipeId wgCount32 one32 one32
      let scratchOff ← iconst64 64
      let _ ← gpuDownload ptr buf2 scratchOff metaSzC
      bxor dir c1   -- next direction

    -- Step 5: Download the buffer the last stage wrote and write it out:
    -- direction 0 after the loop → the last stage wrote buf_a, else buf_b.
    let outFnOff ← iconst64 outputFilename_off
    let finish (dstOff : V .i64) (bufId : V .i32) : Prog V L Unit := do
      let _ ← gpuDownload ptr bufId dstOff alignedSz
      gpuCleanup ptr
      let _ ← ffi fnWrite %[ptr, outFnOff, dstOff, c0, dataSz]
    let bufAOffC ← iconst64 bufA_off
    let bufBOffC ← iconst64 bufB_off
    let _ ← ifte .eq finalDir c0
      (thn := do finish bufAOffC buf0; pure %[])
      (els := do finish bufBOffC buf1; pure %[])
    pure %[])
   pure %[])

def clifIrSource : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code)]

-- ---------------------------------------------------------------------------
-- Payload construction
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  let reserved := zeros 0x40
  let hdrPad := zeros (bindDesc_off - 0x40)
  -- 3 binding descriptors: [buf_id (u32), read_only (u32)] × 3
  let bindDesc :=
    uint32ToBytes 0 ++ uint32ToBytes 0 ++   -- buf0: buf_a, read_write
    uint32ToBytes 1 ++ uint32ToBytes 0 ++   -- buf1: buf_b, read_write
    uint32ToBytes 2 ++ uint32ToBytes 1       -- buf2: meta, read_only
  let bindPad := zeros (shader_off - bindDesc_off - 24)
  let shaderBytes := padTo (stringToBytes fftShader) shaderRegionSize
  let inputFnameBytes := zeros filenameRegionSize
  let outputFnameBytes := padTo (stringToBytes "fft_output.bin") filenameRegionSize
  let flagBytes := uint64ToBytes 0
  let flagPad := zeros (clifIr_off - flag_off - 8)
  let clifPad := zeros clifIrRegionSize
  reserved ++ hdrPad ++
  bindDesc ++ bindPad ++
  shaderBytes ++ inputFnameBytes ++ outputFnameBytes ++
  flagBytes ++ flagPad ++ clifPad

-- ---------------------------------------------------------------------------
-- Configuration
-- ---------------------------------------------------------------------------

def fftConfig (clif : List FuncData) : Artifact := {
  functions := clif,
  required_memory := payloads.length + totalAdditionalMemory,
  initial_memory := payloads
}

end Fft

def Demo.Fft.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie Fft.clifIrSource
  emitArtifacts outDir #[artifactEntry "fft_app" (Fft.fftConfig clif)]

#eval ShipScan.check "Demo.Fft" `Demo.Fft.main
