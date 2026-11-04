module
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.FFI
meta import AlgorithmLib.Surface.FFI
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Surface.Wg
meta import AlgorithmLib.Surface.Wg
public import AlgorithmLib.Vocab.WGSL
meta import AlgorithmLib.Vocab.WGSL
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.WGSL

namespace Compress

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

-- Fixed layout offsets: wgpu's state (`app`) from 0, the shader from 0x200
def shader_off         : Nat := 0x200
def shaderRegionSize   : Nat := 16384
def filenameRegionSize : Nat := 256
def outputFilename_off : Nat := shader_off + shaderRegionSize           -- 0x4200
def flag_off           : Nat := outputFilename_off + filenameRegionSize  -- 0x4300
def clifIr_off         : Nat := flag_off + 64                            -- 0x4340
def clifIrRegionSize   : Nat := 8192
def inputData_off      : Nat := clifIr_off + clifIrRegionSize            -- 0x6340
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
         let wordIdx ← letV (base + idx / litU 4)
         let shift ← letV ((idx % litU 4) * litU 8)
         retE (bandU (shrU (arrIdx inputData wordIdx) shift) (litU 0xFF))
     },
     .fn {
       name := "read4",
       params := [{ name := "base", ty := .u32 }, { name := "idx", ty := .u32 }],
       retTy := some .u32,
       body := do
         let base : AlgorithmLib.WGSL.Expr .u32 := ⟨"base"⟩
         let idx : AlgorithmLib.WGSL.Expr .u32 := ⟨"idx"⟩
         let b0 ← letV (readByteE base idx)
         let b1 ← letV (readByteE base (idx + litU 1))
         let b2 ← letV (readByteE base (idx + litU 2))
         let b3 ← letV (readByteE base (idx + litU 3))
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
         let wordIdx ← letV (base + idx / litU 4)
         let shift ← letV ((idx % litU 4) * litU 8)
         let mask ← letV (bnotU (shlU (litU 0xFF) shift))
         let oldVal ← letV (arrIdx outputData wordIdx)
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
      let blockId ← letV widX
      let threadId ← letV lidX
      let totalInputSize ← letV (arrIdx blockMeta (litU 0))
      ifB (ltE threadId (litU 4)) do
        forU "i" (threadId * litU 64) (fun i => ltE i ((threadId + litU 1) * litU 64)) (fun i => i + litU 1) fun i => do
          assign (arrIdxN hashTable i) (litU 0xFFFFFFFF)
      wBarrier
      let inputByteStart ← letV (blockId * blockSize)
      let inputWordStart ← letV (inputByteStart / litU 4)
      let blockLen ← letV (wMinU blockSize (totalInputSize - inputByteStart))
      let outputByteStart ← letV (blockId * maxCompBlockSize)
      let outputWordStart ← letV (outputByteStart / litU 4)
      let stride ← letV (litU 64)
      let pos ← varV threadId
      whileB (ltE (pos + litU 3) blockLen) do
        let val ← letV (read4E inputWordStart pos)
        let h ← letV (hash4E val)
        assign (arrIdxN hashTable h) pos
        assign pos (pos + stride)
      wBarrier
      ifB (eqE threadId (litU 0)) do
        let ip ← varV (litU 0)
        let op ← varV (litU 0)
        let anchor ← varV (litU 0)
        let matchLimit ← letV (blockLen - wMinU blockLen (litU 5))
        whileB (leE (ip + litU 4) matchLimit) do
          let cur4 ← letV (read4E inputWordStart ip)
          let h ← letV (hash4E cur4)
          let refPos ← letV (arrIdxN hashTable h)
          assign (arrIdxN hashTable h) ip
          let matchLen ← varV (litU 0)
          ifB (andE (andE (neE refPos (litU 0xFFFFFFFF)) (ltE refPos ip)) (ltE (ip - refPos) (litU 65536))) do
            let ref4 ← letV (read4E inputWordStart refPos)
            ifB (eqE ref4 cur4) do
              assign matchLen (litU 4)
              whileB (andE (ltE (ip + matchLen) matchLimit) (ltE matchLen maxMatchLen)) do
                ifB (neE (readByteE inputWordStart (refPos + matchLen)) (readByteE inputWordStart (ip + matchLen))) do
                  breakS
                assign matchLen (matchLen + litU 1)
          ifB (geE matchLen minMatchLen) do
            let literalLen ← letV (ip - anchor)
            let matchOffset ← letV (ip - refPos)
            let litToken ← letV (wMinU literalLen (litU 15))
            let matchToken ← letV (wMinU (matchLen - minMatchLen) (litU 15))
            callS "write_byte" [toString outputWordStart, toString op, toString (borU (shlU litToken (litU 4)) matchToken)]
            assign op (op + litU 1)
            ifB (geE literalLen (litU 15)) do
              let rem ← varV (literalLen - litU 15)
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
              let rem ← varV (matchLen - minMatchLen - litU 15)
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
        let finalLit ← letV (blockLen - anchor)
        ifB (gtE finalLit (litU 0)) do
          let litToken ← letV (wMinU finalLit (litU 15))
          callS "write_byte" [toString outputWordStart, toString op, toString (shlU litToken (litU 4))]
          assign op (op + litU 1)
          ifB (geE finalLit (litU 15)) do
            let rem ← varV (finalLit - litU 15)
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
--   1. readFile (input file → shared memory, returns byte count)
--   2. wgpu brought up: the three buffers and the pipeline (`Wg.App.setup`)
--   3. the input uploaded, one dispatch over the blocks, the output and its
--      metadata downloaded, and everything released
--   4. replaceFileAt, then writeFileAt per block (compressed output — concatenated
--      blocks with size header)
-- ---------------------------------------------------------------------------

/-- The input (up to four bytes past the largest a read takes, as the
    upload rounds up), the blocks it compresses to and their sizes, and the
    one pipeline over all three, called through wgpu directly; `shLen` is the
    shader's length in bytes, `(compressionShader bs).utf8ByteSize`. -/
def app (bs shLen : Nat) : Wg.App :=
  { bufs := [maxInputSize + 4, outputBufSize bs, metaSize bs],
    pipes := [{ shaderOff := shader_off, shaderLen := shLen,
                binds := [(0, true), (1, false), (2, false)] }],
    memSize := blockMeta_off bs + metaSize bs }

/-- The shipped shader's length, at the block size it ships with. -/
def shaderLen16384 : Nat := utf8Len% (compressionShader 16384)

/-- wgpu's state fits below the shader. -/
theorem app_fits : (app 16384 shaderLen16384).stateEnd ≤ shader_off := by decide

open AlgorithmLib.Prog in
/-- The body, for any block size.

    A function of `bs` --- which is what makes an invalid block size a type
    error --- and nothing about that costs it anything now: the term is
    ordinary Lean, and `compileProg` checks the body it emits at the size that
    ships. -/
def code (bs shLen : Nat) : Prog V L Unit :=
  let obSz  := outputBufSize bs
  let mSz   := metaSize bs
  let mOff  := blockMeta_off bs
  let mcbSz := maxCompressedBlockSize bs
  do
    let ptr ← basePtr

    -- Step 1: Read the file the caller's input names
    let inData    ← iconst64 inputData_off
    let zero      ← iconst64 0
    let bytesRead ← readInputFile ptr inputData_off maxInputSize
    -- A read that failed (-1) or ran past the input region compresses nothing.
    let _ ← ifte .ugt bytesRead (← iconst64 maxInputSize) (pure %[]) (do

      -- Step 2: Align up to multiple of 4 (wgpu COPY_BUFFER_ALIGNMENT)
      let alignedSz ← alignUp4 bytesRead

      -- Step 3: numBlocks = ceil(bytesRead / blockSize)
      let blkSzV  ← iconst64 bs
      let c1      ← iconst64 1
      let bsm1    ← isub blkSzV c1
      let sum     ← iadd bytesRead bsm1
      let numBlocks   ← udiv sum blkSzV
      let numBlocks32 ← ireduce32 numBlocks

      -- Step 4: wgpu, its buffers and the pipeline
      (app bs shLen).setup ptr

      -- Step 6: Write input size into metadata region
      let inputSz32 ← ireduce32 bytesRead
      let metaOffV  ← iconst64 mOff
      let metaAddr  ← iadd ptr metaOffV
      storeUnaligned inputSz32 metaAddr

      -- Step 7: Upload input data and metadata
      (app bs shLen).upload ptr 0 (← iadd ptr inData) alignedSz
      (app bs shLen).upload ptr 2 metaAddr (← iconst64 mSz)

      -- Step 9: Dispatch — numBlocks workgroups
      let one32 ← iconst32 1
      (app bs shLen).dispatch ptr 0 numBlocks32 one32 one32

      -- Step 10: Download output and metadata
      let outDataV ← iconst64 outputData_off
      (app bs shLen).download ptr 1 (← iadd ptr outDataV) (← iconst64 obSz)
      (app bs shLen).download ptr 2 metaAddr (← iconst64 mSz)

      -- Step 11: release what wgpu made
      (app bs shLen).cleanup ptr

      -- Step 12: Write standard LZ4 frame format
      -- Reuse inputData_off area as scratch
      let scratchAddr ← iadd ptr inData

      -- Write magic number 0x184D2204 LE
      let magic ← iconst32 0x184D2204
      storeUnaligned magic scratchAddr
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
      let outCap ← iconst64 filenameRegionSize
      let c7 ← iconst64 7
      let _ ← replaceFileAt ptr outFname outCap inData c7

      -- Loop over blocks: write [block_size:4][block_data:N] for each
      let c8        ← iconst64 8
      let maxCompBlkV ← iconst64 mcbSz

      -- For block_i in 0..numBlocks: write [block_size:4][block_data:N], the
      -- running output-file offset carried from 7 (the frame header). A size the
      -- device reports past the block's room, or of zero (a write of size 0 is
      -- a write up to a NUL), is not written.
      let blkExit ← forLoopAcc numBlocks c7 fun bi foff => do
        -- Read compressed size from block_meta[1 + block_i*2] = metaOff + 4 + block_i*8
        let bi8      ← imul bi c8
        let bi8p4    ← iadd bi8 c4
        let metaRel  ← iadd metaOffV bi8p4
        let metaAbsI ← iadd ptr metaRel
        let compSz32 ← load32 metaAbsI
        let compSz64 ← uextend64 compSz32
        let foffP4 ← iadd foff c4
        let szM1 ← isub compSz64 c1
        let _ ← ifte .uge szM1 maxCompBlkV (pure %[])
          (do
            -- Write block_size (u32 LE) into scratch, then to file
            storeUnaligned compSz32 scratchAddr
            let _ ← writeFileAt ptr outFname outCap inData foff c4
            -- Write block data
            let biTimesMax ← imul bi maxCompBlkV
            let blkDataRel ← iadd outDataV biTimesMax
            let _ ← writeFileAt ptr outFname outCap blkDataRel foffP4 compSz64
            pure %[])
        -- the offset moves past a block only when one was written
        select (← icmp .uge szM1 maxCompBlkV) foff (← iadd foffP4 compSz64)
      let finalFoff := blkExit

      -- Write 4-byte end mark (0x00000000) at the final offset
      let endMark ← iconst32 0
      storeUnaligned endMark scratchAddr
      let _ ← writeFileAt ptr outFname outCap inData finalFoff c4
      pure %[])

-- ---------------------------------------------------------------------------
-- Payload construction (parameterized by blockSize)
-- ---------------------------------------------------------------------------

def buildPayload (bs : Nat) : List UInt8 :=
  let a := app bs (compressionShader bs).utf8ByteSize
  let wgState        := zeros a.NAME ++ "main".toUTF8.toList ++ zeros (shader_off - a.NAME - 4)
  let shaderBytes    := padTo (stringToBytes (compressionShader bs)) shaderRegionSize
  let outputFnameBytes := padTo (stringToBytes "compress_output.lz4") filenameRegionSize
  let flagBytes      := uint64ToBytes 0
  let flagPad        := zeros (clifIr_off - flag_off - 8)
  let clifPad        := zeros clifIrRegionSize
  wgState ++
  shaderBytes ++ outputFnameBytes ++
  flagBytes ++ flagPad ++ clifPad

-- ---------------------------------------------------------------------------
-- Monomorphic builder: takes a proof-carrying LZ4Params, returns config+alg.
-- The block size is fixed per instance; all layout is derived from it.
-- ---------------------------------------------------------------------------


def buildCompressor {bs : Nat} (_p : LZ4Params bs) : Except String (Artifact × UInt32) := do
  let payload := buildPayload bs
  let cfg : Artifact := {
    functions := ← Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 (code bs (compressionShader bs).utf8ByteSize))],
    required_memory   := payload.length + totalAdditionalMemory bs,
    initial_memory := payload
  }
  let alg : UInt32 := IR.mainFnIdx
  return (cfg, alg)

-- ---------------------------------------------------------------------------
-- Demo: 16KB blocks. Both proofs close by omega.
-- Change blockSize here — Lean enforces the constraints before any code runs.
-- ---------------------------------------------------------------------------

def defaultParams : LZ4Params 16384 := ⟨by omega, by omega⟩

def result : Except String (Artifact × UInt32) := buildCompressor defaultParams

-- Uncomment to see the constraint in action:
--
--   def badParams : LZ4Params 131072 := ⟨by omega, by omega⟩
--   -- error: omega could not prove 131072 ≤ 65536
--
--   def misaligned : LZ4Params 100 := ⟨by omega, by omega⟩
--   -- error: omega could not prove 100 % 4 = 0

end Compress

def Demo.Compress.main (args : List String) : IO Unit := do
  let (cfg, alg) ← Prog.orDie Compress.result
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[artifactEntry "compress_app" cfg]

#eval ShipScan.check "Demo.Compress" `Demo.Compress.main
