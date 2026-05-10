import AlgorithmLib

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.WGSL

namespace Algorithm

-- ===========================================================================
-- Dependent-type interface for LZ4 compression.
--
-- LZ4 encodes match offsets as 2-byte (16-bit) fields. A blockSize > 65536
-- would allow matches whose offset overflows that field, producing silently
-- corrupt output that passes on small inputs but fails on large ones.
--
-- LZ4Params captures both constraints at the type level:
--   h_window : blockSize ≤ 65536   (LZ4 offset field width)
--   h_align  : blockSize % 4 = 0   (GPU u32 alignment)
--
-- buildCompressor takes an LZ4Params — invalid block sizes are rejected by
-- Lean's elaborator before any WGSL or CLIF is generated.
-- ===========================================================================

structure LZ4Params (blockSize : Nat) where
  mk ::
  h_window : blockSize ≤ 65536
  h_align  : blockSize % 4 = 0

-- ---------------------------------------------------------------------------
-- Fixed constants (independent of blockSize)
-- ---------------------------------------------------------------------------

def maxInputSize : Nat := 4 * 1024 * 1024

-- Fixed layout offsets
def bindDesc_off       : Nat := 0x100
def shader_off         : Nat := 0x200
def shaderRegionSize   : Nat := 16384
def inputFilename_off  : Nat := shader_off + shaderRegionSize   -- 0x4200
def filenameRegionSize : Nat := 256
def outputFilename_off : Nat := inputFilename_off + filenameRegionSize   -- 0x4300
def flag_off           : Nat := outputFilename_off + filenameRegionSize  -- 0x4400
def clifIr_off         : Nat := flag_off + 64                            -- 0x4440
def clifIrRegionSize   : Nat := 8192
def inputData_off      : Nat := clifIr_off + clifIrRegionSize            -- 0x6440
def outputData_off     : Nat := inputData_off + maxInputSize

-- ---------------------------------------------------------------------------
-- blockSize-derived constants
-- ---------------------------------------------------------------------------

def maxBlocks              (bs : Nat) : Nat := maxInputSize / bs
def maxCompressedBlockSize (bs : Nat) : Nat := bs + 1024
def outputBufSize          (bs : Nat) : Nat := maxBlocks bs * maxCompressedBlockSize bs
def metaSize               (bs : Nat) : Nat := maxBlocks bs * 8
def blockMeta_off          (bs : Nat) : Nat := outputData_off + outputBufSize bs
def totalAdditionalMemory  (bs : Nat) : Nat := maxInputSize + outputBufSize bs + metaSize bs

-- ---------------------------------------------------------------------------
-- WGSL compute shader: per-block LZ4 compression (parameterized by blockSize)
--
-- @workgroup_size(64), dispatch(numBlocks, 1, 1).
-- Each workgroup compresses one block independently.
-- Bindings: input (read), output (rw), block_meta (rw).
-- ---------------------------------------------------------------------------

def compressionShader (bs : Nat) : String :=
  let inputData : AlgorithmLib.WGSL.Expr (.arr .u32) := ⟨"input_data"⟩
  let outputData : AlgorithmLib.WGSL.Expr (.arr .u32) := ⟨"output_data"⟩
  let blockMeta : AlgorithmLib.WGSL.Expr (.arr .u32) := ⟨"block_meta"⟩
  let hashTable : AlgorithmLib.WGSL.Expr (.arrN .u32 256) := ⟨"hash_table"⟩
  let blockSize : AlgorithmLib.WGSL.Expr .u32 := ⟨"BLOCK_SIZE"⟩
  let maxCompBlockSize : AlgorithmLib.WGSL.Expr .u32 := ⟨"MAX_COMPRESSED_BLOCK_SIZE"⟩
  let minMatchLen : AlgorithmLib.WGSL.Expr .u32 := ⟨"MIN_MATCH_LEN"⟩
  let maxMatchLen : AlgorithmLib.WGSL.Expr .u32 := ⟨"MAX_MATCH_LEN"⟩
  let readByteE (base idx : AlgorithmLib.WGSL.Expr .u32) : AlgorithmLib.WGSL.Expr .u32 := call2 "read_byte" base idx
  let read4E (base idx : AlgorithmLib.WGSL.Expr .u32) : AlgorithmLib.WGSL.Expr .u32 := call2 "read4" base idx
  let hash4E (v : AlgorithmLib.WGSL.Expr .u32) : AlgorithmLib.WGSL.Expr .u32 := call1 "hash4" v
  buildShader
    [{ binding := 0, name := "input_data", ty := .arr .u32, ro := true },
     { binding := 1, name := "output_data", ty := .arr .u32 },
     { binding := 2, name := "block_meta", ty := .arr .u32 }]
    [("hash_table", .u32, 256)]
    [.constU "BLOCK_SIZE" bs,
     .constU "MAX_COMPRESSED_BLOCK_SIZE" (maxCompressedBlockSize bs),
     .constU "MIN_MATCH_LEN" 4,
     .constU "MAX_MATCH_LEN" 255,
     .fn {
       name := "read_byte",
       params := [{ name := "base", ty := .u32 }, { name := "idx", ty := .u32 }],
       retTy := some .u32,
       body := do
         let base : AlgorithmLib.WGSL.Expr .u32 := ⟨"base"⟩
         let idx : AlgorithmLib.WGSL.Expr .u32 := ⟨"idx"⟩
         let wordIdx ← letV "word_idx" (base + idx / litU 4)
         let shift ← letV "shift" ((idx % litU 4) * litU 8)
         retE (bandU (shrU (arrIdx inputData wordIdx) shift) (litU 0xFF))
     },
     .fn {
       name := "read4",
       params := [{ name := "base", ty := .u32 }, { name := "idx", ty := .u32 }],
       retTy := some .u32,
       body := do
         let base : AlgorithmLib.WGSL.Expr .u32 := ⟨"base"⟩
         let idx : AlgorithmLib.WGSL.Expr .u32 := ⟨"idx"⟩
         let b0 ← letV "b0" (readByteE base idx)
         let b1 ← letV "b1" (readByteE base (idx + litU 1))
         let b2 ← letV "b2" (readByteE base (idx + litU 2))
         let b3 ← letV "b3" (readByteE base (idx + litU 3))
         retE (borU (borU b0 (shlU b1 (litU 8))) (borU (shlU b2 (litU 16)) (shlU b3 (litU 24))))
     },
     .fn {
       name := "write_byte",
       params := [{ name := "base", ty := .u32 }, { name := "idx", ty := .u32 }, { name := "val", ty := .u32 }],
       retTy := none,
       body := do
         let base : AlgorithmLib.WGSL.Expr .u32 := ⟨"base"⟩
         let idx : AlgorithmLib.WGSL.Expr .u32 := ⟨"idx"⟩
         let val : AlgorithmLib.WGSL.Expr .u32 := ⟨"val"⟩
         let wordIdx ← letV "word_idx" (base + idx / litU 4)
         let shift ← letV "shift" ((idx % litU 4) * litU 8)
         let mask ← letV "mask" (bnotU (shlU (litU 0xFF) shift))
         let oldVal ← letV "old_val" (arrIdx outputData wordIdx)
         assign (arrIdx outputData wordIdx) (borU (bandU oldVal mask) (shlU (bandU val (litU 0xFF)) shift))
     },
     .fn {
       name := "hash4",
       params := [{ name := "v", ty := .u32 }],
       retTy := some .u32,
       body := do
         let v : AlgorithmLib.WGSL.Expr .u32 := ⟨"v"⟩
         retE (shrU (v * litU 2654435761) (litU 24))
     }]
    { lid := true, wid := true }
    do
      let blockId ← letV "block_id" widX
      let threadId ← letV "thread_id" lidX
      let totalInputSize ← letV "total_input_size" (arrIdx blockMeta (litU 0))
      ifB (ltE threadId (litU 4)) do
        forU "i" (threadId * litU 64) (fun i => ltE i ((threadId + litU 1) * litU 64)) (fun i => i + litU 1) fun i => do
          assign (arrIdxN hashTable i) (litU 0xFFFFFFFF)
      wBarrier
      let inputByteStart ← letV "input_byte_start" (blockId * blockSize)
      let inputWordStart ← letV "input_word_start" (inputByteStart / litU 4)
      let blockLen ← letV "block_len" (wMinU blockSize (totalInputSize - inputByteStart))
      let outputByteStart ← letV "output_byte_start" (blockId * maxCompBlockSize)
      let outputWordStart ← letV "output_word_start" (outputByteStart / litU 4)
      let stride ← letV "stride" (litU 64)
      let pos ← varV "pos" threadId
      whileB (ltE (pos + litU 3) blockLen) do
        let val ← letV "val" (read4E inputWordStart pos)
        let h ← letV "h" (hash4E val)
        assign (arrIdxN hashTable h) pos
        assign pos (pos + stride)
      wBarrier
      ifB (eqE threadId (litU 0)) do
        let ip ← varV "ip" (litU 0)
        let op ← varV "op" (litU 0)
        let anchor ← varV "anchor" (litU 0)
        let matchLimit ← letV "match_limit" (blockLen - wMinU blockLen (litU 5))
        whileB (leE (ip + litU 4) matchLimit) do
          let cur4 ← letV "cur4" (read4E inputWordStart ip)
          let h ← letV "h" (hash4E cur4)
          let refPos ← letV "ref_pos" (arrIdxN hashTable h)
          assign (arrIdxN hashTable h) ip
          let matchLen ← varV "match_len" (litU 0)
          ifB (andE (andE (neE refPos (litU 0xFFFFFFFF)) (ltE refPos ip)) (ltE (ip - refPos) (litU 65536))) do
            let ref4 ← letV "ref4" (read4E inputWordStart refPos)
            ifB (eqE ref4 cur4) do
              assign matchLen (litU 4)
              whileB (andE (ltE (ip + matchLen) matchLimit) (ltE matchLen maxMatchLen)) do
                ifB (neE (readByteE inputWordStart (refPos + matchLen)) (readByteE inputWordStart (ip + matchLen))) do
                  breakS
                assign matchLen (matchLen + litU 1)
          ifB (geE matchLen minMatchLen) do
            let literalLen ← letV "literal_len" (ip - anchor)
            let matchOffset ← letV "match_offset" (ip - refPos)
            let litToken ← letV "lit_token" (wMinU literalLen (litU 15))
            let matchToken ← letV "match_token" (wMinU (matchLen - minMatchLen) (litU 15))
            callS "write_byte" [toString outputWordStart, toString op, toString (borU (shlU litToken (litU 4)) matchToken)]
            assign op (op + litU 1)
            ifB (geE literalLen (litU 15)) do
              let rem ← varV "rem" (literalLen - litU 15)
              whileB (geE rem (litU 255)) do
                callS "write_byte" [toString outputWordStart, toString op, toString (litU 255)]
                assign op (op + litU 1)
                assign rem (rem - litU 255)
              callS "write_byte" [toString outputWordStart, toString op, toString rem]
              assign op (op + litU 1)
            forU "i" (litU 0) (fun i => ltE i literalLen) (fun i => i + litU 1) fun i => do
              callS "write_byte" [toString outputWordStart, toString op, toString (readByteE inputWordStart (anchor + i))]
              assign op (op + litU 1)
            callS "write_byte" [toString outputWordStart, toString op, toString (bandU matchOffset (litU 0xFF))]
            callS "write_byte" [toString outputWordStart, toString (op + litU 1), toString (bandU (shrU matchOffset (litU 8)) (litU 0xFF))]
            assign op (op + litU 2)
            ifB (geE (matchLen - minMatchLen) (litU 15)) do
              let rem ← varV "rem" (matchLen - minMatchLen - litU 15)
              whileB (geE rem (litU 255)) do
                callS "write_byte" [toString outputWordStart, toString op, toString (litU 255)]
                assign op (op + litU 1)
                assign rem (rem - litU 255)
              callS "write_byte" [toString outputWordStart, toString op, toString rem]
              assign op (op + litU 1)
            assign ip (ip + matchLen)
            assign anchor ip
          ifB (ltE matchLen minMatchLen) do
            assign ip (ip + litU 1)
        let finalLit ← letV "final_lit" (blockLen - anchor)
        ifB (gtE finalLit (litU 0)) do
          let litToken ← letV "lit_token" (wMinU finalLit (litU 15))
          callS "write_byte" [toString outputWordStart, toString op, toString (shlU litToken (litU 4))]
          assign op (op + litU 1)
          ifB (geE finalLit (litU 15)) do
            let rem ← varV "rem" (finalLit - litU 15)
            whileB (geE rem (litU 255)) do
              callS "write_byte" [toString outputWordStart, toString op, toString (litU 255)]
              assign op (op + litU 1)
              assign rem (rem - litU 255)
            callS "write_byte" [toString outputWordStart, toString op, toString rem]
            assign op (op + litU 1)
          forU "i" (litU 0) (fun i => ltE i finalLit) (fun i => i + litU 1) fun i => do
            callS "write_byte" [toString outputWordStart, toString op, toString (readByteE inputWordStart (anchor + i))]
            assign op (op + litU 1)
        assign (arrIdx blockMeta (litU 1 + blockId * litU 2)) op

-- ---------------------------------------------------------------------------
-- CLIF IR: file read → GPU compress → file write (parameterized by blockSize)
--
-- fn u0:0: noop (required by CraneliftUnit)
-- fn u0:1: orchestrator
--   1. cl_file_read (input file → shared memory, returns byte count)
--   2. cl_gpu_init
--   3. cl_gpu_create_buffer ×3 (input, output, metadata)
--   4. cl_gpu_upload (input data to GPU)
--   5. cl_gpu_create_pipeline (3 bindings)
--   6. cl_gpu_dispatch (numBlocks, 1, 1)
--   7. cl_gpu_download ×2 (output, metadata)
--   8. cl_gpu_cleanup
--   9. cl_file_write (compressed output — concatenated blocks with size header)
-- ---------------------------------------------------------------------------

open AlgorithmLib.IR in
def clifIrSource (bs : Nat) : String :=
  let obSz  := outputBufSize bs
  let mSz   := metaSize bs
  let mOff  := blockMeta_off bs
  let mcbSz := maxCompressedBlockSize bs
  buildProgram do
    let fnRead  ← declareFileRead
    let fnWrite ← declareFileWrite
    let gpu     ← declareGpuFFI

    let ptr ← entryBlock

    -- Step 1: Read input file
    let inData    ← iconst64 inputData_off
    let zero      ← iconst64 0
    let bytesRead ← readFile ptr fnRead inputFilename_off inputData_off

    -- Step 2: Align up to multiple of 4 (wgpu COPY_BUFFER_ALIGNMENT)
    let alignedSz ← alignUp4 bytesRead

    -- Step 3: numBlocks = ceil(bytesRead / blockSize)
    let blkSzV  ← iconst64 bs
    let c1      ← iconst64 1
    let bsm1    ← isub blkSzV c1
    let sum     ← iadd bytesRead bsm1
    let numBlocks   ← udiv sum blkSzV
    let numBlocks32 ← ireduce32 numBlocks

    -- Step 4: GPU init
    gpuInit gpu ptr

    -- Step 5: Create 3 buffers
    let buf0      ← gpuCreateBuffer gpu ptr alignedSz
    let outBufSzV ← iconst64 obSz
    let buf1      ← gpuCreateBuffer gpu ptr outBufSzV
    let metaSzV   ← iconst64 mSz
    let buf2      ← gpuCreateBuffer gpu ptr metaSzV

    -- Step 6: Write input size into metadata region
    let inputSz32 ← ireduce32 bytesRead
    let metaOffV  ← iconst64 mOff
    let metaAddr  ← iadd ptr metaOffV
    store inputSz32 metaAddr

    -- Step 7: Upload input data and metadata
    let _ ← gpuUpload gpu ptr buf0 inData alignedSz
    let _ ← gpuUpload gpu ptr buf2 metaOffV metaSzV

    -- Step 8: Create pipeline (3 bindings)
    let shOffV  ← iconst64 shader_off
    let bdOffV  ← iconst64 bindDesc_off
    let three32 ← iconst32 3
    let pipeId  ← gpuCreatePipeline gpu ptr shOffV bdOffV three32

    -- Step 9: Dispatch — numBlocks workgroups
    let one32 ← iconst32 1
    let _ ← gpuDispatch gpu ptr pipeId numBlocks32 one32 one32

    -- Step 10: Download output and metadata
    let outDataV ← iconst64 outputData_off
    let _ ← gpuDownload gpu ptr buf1 outDataV outBufSzV
    let _ ← gpuDownload gpu ptr buf2 metaOffV metaSzV

    -- Step 11: GPU cleanup
    gpuCleanup gpu ptr

    -- Step 12: Write standard LZ4 frame format
    -- Reuse inputData_off area as scratch
    let scratchAddr ← iadd ptr inData

    -- Write magic number 0x184D2204 LE
    let magic ← iconst32 0x184D2204
    store magic scratchAddr
    let c4      ← iconst64 4
    let scratchP4 ← iadd scratchAddr c4

    -- FLG=0x60, BD=0x70, HC=0x73
    let flg ← iconst64 0x60
    istore8 flg scratchP4
    let scratchP5 ← iadd scratchP4 c1
    let bd ← iconst64 0x70
    istore8 bd scratchP5
    let scratchP6 ← iadd scratchP5 c1
    let hc ← iconst64 0x73
    istore8 hc scratchP6

    -- Write 7-byte frame header to file
    let outFname ← iconst64 outputFilename_off
    let c7 ← iconst64 7
    let _ ← call fnWrite [ptr, outFname, inData, zero, c7]

    -- Loop over blocks: write [block_size:4][block_data:N] for each
    let c8        ← iconst64 8
    let maxCompBlkV ← iconst64 mcbSz

    let loopHdr  ← declareBlock [.i64, .i64]   -- block_i, file_offset
    let bi        := loopHdr.param 0
    let foff      := loopHdr.param 1
    let loopDone ← declareBlock [.i64]          -- receives final file_offset
    let loopBody ← declareBlock []
    jump loopHdr.ref [zero, c7]
    startBlock loopHdr
    let done ← icmp .uge bi numBlocks
    brif done loopDone.ref [foff] loopBody.ref []

    startBlock loopBody
    -- Read compressed size from block_meta[1 + block_i*2] = metaOff + 4 + block_i*8
    let bi8      ← imul bi c8
    let bi8p4    ← iadd bi8 c4
    let metaRel  ← iadd metaOffV bi8p4
    let metaAbsI ← iadd ptr metaRel
    let compSz32 ← load32 metaAbsI

    -- Write block_size (u32 LE) into scratch, then to file
    store compSz32 scratchAddr
    let _ ← call fnWrite [ptr, outFname, inData, foff, c4]

    -- Write block data
    let biTimesMax ← imul bi maxCompBlkV
    let blkDataRel ← iadd outDataV biTimesMax
    let compSz64   ← uextend64 compSz32
    let foffP4     ← iadd foff c4
    let _ ← call fnWrite [ptr, outFname, blkDataRel, foffP4, compSz64]

    -- Advance
    let nextFoff ← iadd foffP4 compSz64
    let nextBi   ← iadd bi c1
    jump loopHdr.ref [nextBi, nextFoff]

    -- Write 4-byte end mark (0x00000000)
    startBlock loopDone
    let finalFoff := loopDone.param 0
    let endMark ← iconst32 0
    store endMark scratchAddr
    let _ ← call fnWrite [ptr, outFname, inData, finalFoff, c4]
    ret

-- ---------------------------------------------------------------------------
-- Payload construction (parameterized by blockSize)
-- ---------------------------------------------------------------------------

def buildPayload (bs : Nat) : List UInt8 :=
  let reserved       := zeros 0x40
  let hdrPad         := zeros (bindDesc_off - 0x40)
  -- 3 binding descriptors: [buf_id (i32), read_only (i32)] × 3
  let bindDesc :=
    uint32ToBytes 0 ++ uint32ToBytes 1 ++   -- buf0: input, read_only=true
    uint32ToBytes 1 ++ uint32ToBytes 0 ++   -- buf1: output, read_only=false
    uint32ToBytes 2 ++ uint32ToBytes 0       -- buf2: metadata, read_only=false
  let bindPad        := zeros (shader_off - bindDesc_off - 24)
  let shaderBytes    := padTo (stringToBytes (compressionShader bs)) shaderRegionSize
  -- Input filename left as zeros (main.rs writes the path at runtime)
  let inputFnameBytes  := zeros filenameRegionSize
  let outputFnameBytes := padTo (stringToBytes "compress_output.lz4") filenameRegionSize
  let flagBytes      := uint64ToBytes 0
  let flagPad        := zeros (clifIr_off - flag_off - 8)
  let clifPad        := zeros clifIrRegionSize
  reserved ++ hdrPad ++
  bindDesc ++ bindPad ++
  shaderBytes ++ inputFnameBytes ++ outputFnameBytes ++
  flagBytes ++ flagPad ++ clifPad

-- ---------------------------------------------------------------------------
-- Monomorphic builder: takes a proof-carrying LZ4Params, returns config+alg.
-- The block size is fixed per instance; all layout is derived from it.
-- ---------------------------------------------------------------------------

def buildCompressor {bs : Nat} (_p : LZ4Params bs) : BaseConfig × Algorithm :=
  let payload := buildPayload bs
  let cfg : BaseConfig := {
    cranelift_ir  := clifIrSource bs,
    memory_size   := payload.length + totalAdditionalMemory bs,
    context_offset := 0,
    initial_memory := payload
  }
  let alg : Algorithm := {
    actions         := [IR.clifCallAction],
    cranelift_units := 0,
    timeout_ms      := some 120000
  }
  (cfg, alg)

-- ---------------------------------------------------------------------------
-- Demo: 16KB blocks. Both proofs close by omega.
-- Change blockSize here — Lean enforces the constraints before any code runs.
-- ---------------------------------------------------------------------------

def defaultParams : LZ4Params 16384 := ⟨by omega, by omega⟩

def result : BaseConfig × Algorithm := buildCompressor defaultParams

-- Uncomment to see the constraint in action:
--
--   def badParams : LZ4Params 131072 := ⟨by omega, by omega⟩
--   -- error: omega could not prove 131072 ≤ 65536
--
--   def misaligned : LZ4Params 100 := ⟨by omega, by omega⟩
--   -- error: omega could not prove 100 % 4 = 0

end Algorithm

def main (args : List String) : IO Unit := do
  let (cfg, alg) := Algorithm.result
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[toJsonEntry "compress_app" cfg alg]
