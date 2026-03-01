import AlgorithmLib
open Lean (Json toJson)
open AlgorithmLib

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
  "@group(0) @binding(0)\n" ++
  "var<storage, read_write> buf_a: array<vec2<f32>>;\n" ++
  "@group(0) @binding(1)\n" ++
  "var<storage, read_write> buf_b: array<vec2<f32>>;\n" ++
  "@group(0) @binding(2)\n" ++
  "var<storage, read> params: array<u32>;\n\n" ++

  "const PI: f32 = 3.14159265358979323846;\n\n" ++

  "@compute @workgroup_size(64)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" ++
  "    let n = params[0];\n" ++
  "    let stage = params[1];\n" ++
  "    let direction = params[2];\n" ++
  "    let half_n = n / 2u;\n" ++
  "    let tid = gid.x;\n" ++
  "    if (tid >= half_n) {\n" ++
  "        return;\n" ++
  "    }\n\n" ++

  -- Compute butterfly indices for this stage
  -- block_size = 1 << (stage + 1), half_block = 1 << stage
  "    let half_block = 1u << stage;\n" ++
  "    let block_size = half_block << 1u;\n" ++
  "    let block_id = tid / half_block;\n" ++
  "    let j = tid % half_block;\n" ++
  "    let i_top = block_id * block_size + j;\n" ++
  "    let i_bot = i_top + half_block;\n\n" ++

  -- Twiddle factor: W_N^k = exp(-2*pi*i*k/block_size)
  -- k = j, N_local = block_size
  "    let angle = -2.0 * PI * f32(j) / f32(block_size);\n" ++
  "    let tw = vec2<f32>(cos(angle), sin(angle));\n\n" ++

  -- Read from source buffer
  "    var a_val: vec2<f32>;\n" ++
  "    var b_val: vec2<f32>;\n" ++
  "    if (direction == 0u) {\n" ++
  "        a_val = buf_a[i_top];\n" ++
  "        b_val = buf_a[i_bot];\n" ++
  "    } else {\n" ++
  "        a_val = buf_b[i_top];\n" ++
  "        b_val = buf_b[i_bot];\n" ++
  "    }\n\n" ++

  -- Complex multiply: tw * b_val
  "    let tb = vec2<f32>(\n" ++
  "        tw.x * b_val.x - tw.y * b_val.y,\n" ++
  "        tw.x * b_val.y + tw.y * b_val.x\n" ++
  "    );\n\n" ++

  -- Butterfly: top = a + tw*b, bot = a - tw*b
  "    let out_top = a_val + tb;\n" ++
  "    let out_bot = a_val - tb;\n\n" ++

  -- Write to destination buffer
  "    if (direction == 0u) {\n" ++
  "        buf_b[i_top] = out_top;\n" ++
  "        buf_b[i_bot] = out_bot;\n" ++
  "    } else {\n" ++
  "        buf_a[i_top] = out_top;\n" ++
  "        buf_a[i_bot] = out_bot;\n" ++
  "    }\n" ++
  "}\n"

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

  -- Step 2: Compute log2(N) — loop: shift right until ≤ 1
  let log2Hdr ← declareBlock [.i64, .i64]  -- tmp, log2n
  let log2Body ← declareBlock []
  let log2Done ← declareBlock [.i64]       -- log2n result
  jump log2Hdr.ref [bigN, c0]

  startBlock log2Hdr
  let tmp := log2Hdr.param 0
  let log2n := log2Hdr.param 1
  let cmpGt ← icmp .ugt tmp c1
  brif cmpGt log2Body.ref [] log2Done.ref [log2n]

  startBlock log2Body
  let tmp2 ← ushr tmp c1
  let log2n2 ← iadd log2n c1
  jump log2Hdr.ref [tmp2, log2n2]

  -- log2(N) computed, now bit-reverse permutation
  startBlock log2Done
  let log2Result := log2Done.param 0
  let bufAOff ← iconst64 bufA_off

  -- Outer loop: for i in [0, N)
  let outerHdr ← declareBlock [.i64]    -- i
  let outerBody ← declareBlock []
  let gpuBlk ← declareBlock []          -- after permutation
  jump outerHdr.ref [c0]

  startBlock outerHdr
  let i := outerHdr.param 0
  let iDone ← icmp .uge i bigN
  brif iDone gpuBlk.ref [] outerBody.ref []

  -- Compute bit-reverse of i with log2Result bits
  startBlock outerBody
  let revHdr ← declareBlock [.i64, .i64, .i64]  -- val, rev, bit
  let revBody ← declareBlock []
  let revDone ← declareBlock [.i64]              -- final rev
  jump revHdr.ref [i, c0, c0]

  startBlock revHdr
  let val := revHdr.param 0
  let rev := revHdr.param 1
  let bit := revHdr.param 2
  let bitDone ← icmp .uge bit log2Result
  brif bitDone revDone.ref [rev] revBody.ref []

  startBlock revBody
  let lsb ← band val c1
  let revShift ← ishl rev c1
  let revNew ← bor revShift lsb
  let valShift ← ushr val c1
  let bitNext ← iadd bit c1
  jump revHdr.ref [valShift, revNew, bitNext]

  -- Copy inputData[i] (8 bytes) to bufA[rev]
  startBlock revDone
  let revIdx := revDone.param 0
  -- src = inputData_off + i * 8
  let iMul8 ← imul i c8
  let srcOff ← iadd inDatOff iMul8
  let srcAbs ← iadd ptr srcOff
  -- dst = bufA_off + rev * 8
  let revMul8 ← imul revIdx c8
  let dstOff ← iadd bufAOff revMul8
  let dstAbs ← iadd ptr dstOff
  -- Copy 8 bytes as two i32
  let lo ← load32 srcAbs
  store lo dstAbs
  let srcHi ← iadd srcAbs c4
  let dstHi ← iadd dstAbs c4
  let hi ← load32 srcHi
  store hi dstHi
  let iNext ← iadd i c1
  jump outerHdr.ref [iNext]

  -- Step 3: GPU init
  startBlock gpuBlk
  callVoid gpu.fnInit [ptr]

  -- Align data size to multiple of 4: (N*8 + 3) & ~3
  let dataSz ← imul bigN c8
  let alignedSz ← alignUp4 dataSz

  -- Create 3 buffers
  let buf0 ← call gpu.fnCreateBuffer [ptr, alignedSz]
  let buf1 ← call gpu.fnCreateBuffer [ptr, alignedSz]
  let metaSzC ← iconst64 metaSize
  let buf2 ← call gpu.fnCreateBuffer [ptr, metaSzC]

  -- Write N into meta region
  let metaOffC ← iconst64 meta_off
  let metaAbs ← iadd ptr metaOffC
  let nI32 ← ireduce32 bigN
  store nI32 metaAbs

  -- Upload buf_a
  let _ ← call gpu.fnUpload [ptr, buf0, bufAOff, alignedSz]

  -- Create pipeline (3 bindings)
  let shOffC ← iconst64 shader_off
  let bdOffC ← iconst64 bindDesc_off
  let c3_i32 ← iconst32 3
  let pipeId ← call gpu.fnCreatePipeline [ptr, shOffC, bdOffC, c3_i32]

  -- Compute dispatch size: ceil(N/2 / 64)
  let halfN ← ushr bigN c1
  let c63 ← iconst64 63
  let halfPad ← iadd halfN c63
  let c6 ← iconst64 6
  let wgCount ← ushr halfPad c6
  let wgCount32 ← ireduce32 wgCount
  let one32 ← iconst32 1

  -- Step 4: Stage loop
  let stageHdr ← declareBlock [.i64, .i64]   -- stage, direction
  let stageBody ← declareBlock []
  let downloadBlk ← declareBlock [.i64]      -- final direction
  jump stageHdr.ref [c0, c0]

  startBlock stageHdr
  let stage := stageHdr.param 0
  let dir := stageHdr.param 1
  let stageDone ← icmp .uge stage log2Result
  brif stageDone downloadBlk.ref [dir] stageBody.ref []

  startBlock stageBody
  -- Write stage and direction into meta
  let metaStage ← iadd metaAbs c4
  let stageI32 ← ireduce32 stage
  store stageI32 metaStage
  let metaDir ← iadd metaStage c4
  let dirI32 ← ireduce32 dir
  store dirI32 metaDir
  -- Upload meta
  let _ ← call gpu.fnUpload [ptr, buf2, metaOffC, metaSzC]
  -- Dispatch
  let _ ← call gpu.fnDispatch [ptr, pipeId, wgCount32, one32, one32]
  -- Sync: download meta to scratch to force GPU completion
  let scratchOff ← iconst64 64
  let _ ← call gpu.fnDownload [ptr, buf2, scratchOff, metaSzC]
  -- Next stage, toggle direction
  let stageNext ← iadd stage c1
  let dirNext ← bxor dir c1
  jump stageHdr.ref [stageNext, dirNext]

  -- Step 5: Download result
  -- direction==0 after loop → last was dir=1 → wrote to buf_a → download buf_a
  -- direction==1 after loop → last was dir=0 → wrote to buf_b → download buf_b
  startBlock downloadBlk
  let finalDir := downloadBlk.param 0
  let bufAOffC ← iconst64 bufA_off
  let bufBOffC ← iconst64 bufB_off
  let dirIsZero ← icmp .eq finalDir c0
  -- Use brif to pass both dst_offset (i64) and gpu_buf_id (i32) to final block
  let finalBlk ← declareBlock [.i64, .i32]  -- dst_off, gpu_buf_id
  brif dirIsZero finalBlk.ref [bufAOffC, buf0] finalBlk.ref [bufBOffC, buf1]

  startBlock finalBlk
  let dstOff2 := finalBlk.param 0
  let bufId := finalBlk.param 1
  let _ ← call gpu.fnDownload [ptr, bufId, dstOff2, alignedSz]
  callVoid gpu.fnCleanup [ptr]

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

def fftConfig : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := payloads.length + totalAdditionalMemory,
  context_offset := 0
}

def fftAlgorithm : Algorithm := {
    actions := [IR.clifCallAction],
    payloads := payloads,
    cranelift_units := 0,
    timeout_ms := some 300000
  }

end Algorithm

def main : IO Unit := do
  let json := toJsonPair Algorithm.fftConfig Algorithm.fftAlgorithm
  IO.println (Json.compress json)
