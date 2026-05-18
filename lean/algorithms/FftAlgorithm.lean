import AlgorithmLib
open Lean (Json toJson)
open AlgorithmLib
open AlgorithmLib.WGSL

namespace Algorithm

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
      let n         ← letV "n" (arrIdx params (litU 0))
      let stage     ← letV "stage" (arrIdx params (litU 1))
      let direction ← letV "direction" (arrIdx params (litU 2))
      let halfN     ← letV "half_n" (n / litU 2)
      let tid       ← letV "tid" gidX
      ifB (tid .>= halfN) retV
      let halfBlock ← letV "half_block" ((litU 1) .<< stage)
      let blockSize ← letV "block_size" (halfBlock .<< litU 1)
      let blockId   ← letV "block_id" (tid / halfBlock)
      let j         ← letV "j" (tid % halfBlock)
      let iTop      ← letV "i_top" (blockId * blockSize + j)
      let iBot      ← letV "i_bot" (iTop + halfBlock)
      let angle     ← letV "angle" (-litF "2.0" * ⟨"PI"⟩ * f32OfU j / f32OfU blockSize)
      let tw        ← letV "tw" (mkVec2f (wCos angle) (wSin angle))
      let aVal      ← varVT "a_val" .vec2f
      let bVal      ← varVT "b_val" .vec2f
      ifElse (direction .== litU 0)
        (do
          assign aVal (arrIdx bufA iTop)
          assign bVal (arrIdx bufA iBot))
        (do
          assign aVal (arrIdx bufB iTop)
          assign bVal (arrIdx bufB iBot))
      let tb ← letV "tb" (mkVec2f
        (v2x tw * v2x bVal - v2y tw * v2y bVal)
        (v2x tw * v2y bVal + v2y tw * v2x bVal))
      let outTop ← letV "out_top" (aVal + tb)
      let outBot ← letV "out_bot" (aVal - tb)
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

set_option maxRecDepth 2048 in
open AlgorithmLib.IR in
def clifIrSource : String := buildProgram do
  -- FFI declarations
  let fnRead ← declareFileRead
  let fnWrite ← declareFileWrite
  let gpu ← declareGpuFFI

  let ptr ← entryBlock
  let c0  ← iconst64 0
  let c1  ← iconst64 1
  let c4  ← iconst64 4
  let c8  ← iconst64 8

  -- Step 1: Read input file
  let inDatOff ← iconst64 inputData_off
  let bytesRead ← readFile ptr fnRead inputFilename_off inputData_off

  -- Compute N = bytes_read / 8
  let c3 ← iconst64 3
  let bigN ← ushr bytesRead c3

  -- Step 2: Compute log2(N) — while tmp > 1, tmp >>= 1, log2n += 1.
  let (_, log2Result) ← whileLoop2 .i64 .i64 bigN c0
    (fun tmp _ => icmp .ugt tmp c1)
    (fun tmp log2n => do
      let tmp2  ← ushr tmp c1
      let log2n2 ← iadd log2n c1
      return (tmp2, log2n2))

  let bufAOff ← iconst64 bufA_off

  -- For i in [0, bigN): compute bit-reverse(i, log2Result), then copy
  -- inputData[i] (8 bytes) to bufA[rev].
  forLoop .i64 bigN fun i => do
    -- Inner: fold log2Result bits, carrying (val, rev). bit-count is the loop counter.
    let (_, revIdx) ← forLoopAcc2 .i64 .i64 .i64 log2Result i c0 fun _ val rev => do
      let lsb      ← band val c1
      let revShift ← ishl rev c1
      let revNew   ← bor revShift lsb
      let valShift ← ushr val c1
      return (valShift, revNew)
    -- src = inputData_off + i * 8;   dst = bufA_off + rev * 8
    let srcAbs ← iadd ptr (← iadd inDatOff (← imul i c8))
    let dstAbs ← iadd ptr (← iadd bufAOff (← imul revIdx c8))
    store (← load32 srcAbs) dstAbs
    store (← load32 (← iadd srcAbs c4)) (← iadd dstAbs c4)
  gpuInit gpu ptr

  -- Align data size to multiple of 4: (N*8 + 3) & ~3
  let dataSz ← imul bigN c8
  let alignedSz ← alignUp4 dataSz

  -- Create 3 buffers
  let buf0 ← gpuCreateBuffer gpu ptr alignedSz
  let buf1 ← gpuCreateBuffer gpu ptr alignedSz
  let metaSzC ← iconst64 metaSize
  let buf2 ← gpuCreateBuffer gpu ptr metaSzC

  -- Write N into meta region
  let metaOffC ← iconst64 meta_off
  let metaAbs ← iadd ptr metaOffC
  let nI32 ← ireduce32 bigN
  store nI32 metaAbs

  -- Upload buf_a
  let _ ← gpuUpload gpu ptr buf0 bufAOff alignedSz

  -- Create pipeline (3 bindings)
  let shOffC ← iconst64 shader_off
  let bdOffC ← iconst64 bindDesc_off
  let c3_i32 ← iconst32 3
  let pipeId ← gpuCreatePipeline gpu ptr shOffC bdOffC c3_i32

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
  let finalDir ← forLoopAcc .i64 .i64 log2Result c0 fun stage dir => do
    -- Write stage and direction into meta
    let metaStage ← iadd metaAbs c4
    store (← ireduce32 stage) metaStage
    let metaDir ← iadd metaStage c4
    store (← ireduce32 dir) metaDir
    -- Upload meta, dispatch, then sync via download-to-scratch
    let _ ← gpuUpload gpu ptr buf2 metaOffC metaSzC
    let _ ← gpuDispatch gpu ptr pipeId wgCount32 one32 one32
    let scratchOff ← iconst64 64
    let _ ← gpuDownload gpu ptr buf2 scratchOff metaSzC
    bxor dir c1   -- next direction

  -- Step 5: Download result.
  -- direction==0 after loop → last was dir=1 → wrote to buf_a → download buf_a
  -- direction==1 after loop → last was dir=0 → wrote to buf_b → download buf_b
  let bufAOffC ← iconst64 bufA_off
  let bufBOffC ← iconst64 bufB_off
  let dirIsZero ← icmp .eq finalDir c0
  -- Use brif to pass both dst_offset (i64) and gpu_buf_id (i32) to final block
  let finalBlk ← declareBlock [.i64, .i32]  -- dst_off, gpu_buf_id
  brif dirIsZero finalBlk.ref [bufAOffC, buf0] finalBlk.ref [bufBOffC, buf1]

  startBlock finalBlk
  let dstOff2 := finalBlk.param 0
  let bufId := finalBlk.param 1
  let _ ← gpuDownload gpu ptr bufId dstOff2 alignedSz
  gpuCleanup gpu ptr

  -- Step 6: Write output file
  let outFnOff ← iconst64 outputFilename_off
  let _ ← call fnWrite [ptr, outFnOff, dstOff2, c0, dataSz]
  ret

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

def fftConfig : Setup := {
  cranelift_ir := clifIrSource,
  memory_size := payloads.length + totalAdditionalMemory,
  context_offset := 0,
  initial_memory := payloads
}

def fftAlgorithm : Algorithm := {
    fn_idx := IR.mainFnIdx
  }

end Algorithm

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[toJsonEntry "fft_app" Algorithm.fftConfig Algorithm.fftAlgorithm]
