import AlgorithmLib

open Lean (Json)
open AlgorithmLib

namespace Algorithm

-- ---------------------------------------------------------------------------
-- GPU-accelerated LZ4-style file compression
--
-- Reads input file (path from command line, written into payload by main.rs).
-- Splits into 16KB blocks, each compressed by one GPU workgroup (64 threads).
-- Phase 1: all threads cooperatively build hash table for match finding.
-- Phase 2: thread 0 does sequential greedy LZ4 encoding.
-- Writes compressed output to compress_output.lz4.
-- ---------------------------------------------------------------------------

def blockSize : Nat := 16384            -- 16KB per block
def maxInputSize : Nat := 4 * 1024 * 1024  -- 4MB max input
def maxBlocks : Nat := maxInputSize / blockSize  -- 256 blocks max
def maxCompressedBlockSize : Nat := blockSize + 1024  -- worst case overhead
def outputBufSize : Nat := maxBlocks * maxCompressedBlockSize
def metaSize : Nat := maxBlocks * 8     -- compressed size per block (u32) + padding

-- ---------------------------------------------------------------------------
-- Payload layout
-- ---------------------------------------------------------------------------

def bindDesc_off : Nat := 0x100
def shader_off : Nat := 0x200
def shaderRegionSize : Nat := 16384     -- 16KB for shader
def inputFilename_off : Nat := shader_off + shaderRegionSize  -- 0x4200
def filenameRegionSize : Nat := 256
def outputFilename_off : Nat := inputFilename_off + filenameRegionSize  -- 0x4300
def flag_off : Nat := outputFilename_off + filenameRegionSize  -- 0x4400
def clifIr_off : Nat := flag_off + 64   -- 0x4440
def clifIrRegionSize : Nat := 8192      -- 8KB for CLIF IR
def inputData_off : Nat := clifIr_off + clifIrRegionSize  -- 0x6440
def outputData_off : Nat := inputData_off + maxInputSize
def blockMeta_off : Nat := outputData_off + outputBufSize
def totalAdditionalMemory : Nat := maxInputSize + outputBufSize + metaSize

-- ---------------------------------------------------------------------------
-- WGSL compute shader: per-block LZ4 compression
--
-- @workgroup_size(64), dispatch(numBlocks, 1, 1).
-- Each workgroup compresses one 16KB block independently.
-- Bindings: input (read), output (rw), block_meta (rw).
-- ---------------------------------------------------------------------------

def compressionShader : String :=
  let blkSz := toString blockSize
  let maxCompBlkSz := toString maxCompressedBlockSize
  -- Bindings
  "@group(0) @binding(0)\n" ++
  "var<storage, read> input_data: array<u32>;\n" ++
  "@group(0) @binding(1)\n" ++
  "var<storage, read_write> output_data: array<u32>;\n" ++
  "@group(0) @binding(2)\n" ++
  "var<storage, read_write> block_meta: array<u32>;\n\n" ++

  -- Workgroup-shared hash table (256 entries)
  "var<workgroup> hash_table: array<u32, 256>;\n\n" ++

  -- Constants
  s!"const BLOCK_SIZE: u32 = {blkSz}u;\n" ++
  s!"const MAX_COMPRESSED_BLOCK_SIZE: u32 = {maxCompBlkSz}u;\n" ++
  "const MIN_MATCH_LEN: u32 = 4u;\n" ++
  "const MAX_MATCH_LEN: u32 = 255u;\n\n" ++

  -- Read byte from array<u32>
  "fn read_byte(base: u32, idx: u32) -> u32 {\n" ++
  "    let word_idx = base + idx / 4u;\n" ++
  "    let shift = (idx % 4u) * 8u;\n" ++
  "    return (input_data[word_idx] >> shift) & 0xFFu;\n" ++
  "}\n\n" ++

  -- Read 4 bytes as u32
  "fn read4(base: u32, idx: u32) -> u32 {\n" ++
  "    let b0 = read_byte(base, idx);\n" ++
  "    let b1 = read_byte(base, idx + 1u);\n" ++
  "    let b2 = read_byte(base, idx + 2u);\n" ++
  "    let b3 = read_byte(base, idx + 3u);\n" ++
  "    return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);\n" ++
  "}\n\n" ++

  -- Write byte to output array<u32>
  "fn write_byte(base: u32, idx: u32, val: u32) {\n" ++
  "    let word_idx = base + idx / 4u;\n" ++
  "    let shift = (idx % 4u) * 8u;\n" ++
  "    let mask = ~(0xFFu << shift);\n" ++
  "    let old_val = output_data[word_idx];\n" ++
  "    output_data[word_idx] = (old_val & mask) | ((val & 0xFFu) << shift);\n" ++
  "}\n\n" ++

  -- Hash function (Knuth multiplicative, 8-bit result)
  "fn hash4(v: u32) -> u32 {\n" ++
  "    return (v * 2654435761u) >> 24u;\n" ++
  "}\n\n" ++

  -- Main compute entry
  "@compute @workgroup_size(64)\n" ++
  "fn main(\n" ++
  "    @builtin(workgroup_id) wid: vec3<u32>,\n" ++
  "    @builtin(local_invocation_id) lid: vec3<u32>,\n" ++
  ") {\n" ++
  "    let block_id = wid.x;\n" ++
  "    let thread_id = lid.x;\n" ++
  "    let total_input_size = block_meta[0u];\n" ++

  -- Initialize hash table
  "    if (thread_id < 4u) {\n" ++
  "        for (var i = thread_id * 64u; i < (thread_id + 1u) * 64u; i = i + 1u) {\n" ++
  "            hash_table[i] = 0xFFFFFFFFu;\n" ++
  "        }\n" ++
  "    }\n" ++
  "    workgroupBarrier();\n\n" ++

  -- Input/output regions for this block
  "    let input_byte_start = block_id * BLOCK_SIZE;\n" ++
  "    let input_word_start = input_byte_start / 4u;\n" ++
  "    let block_len = min(BLOCK_SIZE, total_input_size - input_byte_start);\n" ++
  "    let output_byte_start = block_id * MAX_COMPRESSED_BLOCK_SIZE;\n" ++
  "    let output_word_start = output_byte_start / 4u;\n\n" ++

  -- Phase 1: Cooperative hash table population
  "    let stride = 64u;\n" ++
  "    var pos = thread_id;\n" ++
  "    while (pos + 3u < block_len) {\n" ++
  "        let val = read4(input_word_start, pos);\n" ++
  "        let h = hash4(val);\n" ++
  "        hash_table[h] = pos;\n" ++
  "        pos = pos + stride;\n" ++
  "    }\n" ++
  "    workgroupBarrier();\n\n" ++

  -- Phase 2: Thread 0 sequential greedy encoding
  "    if (thread_id == 0u) {\n" ++
  "        var ip: u32 = 0u;\n" ++
  "        var op: u32 = 0u;\n" ++
  "        var anchor: u32 = 0u;\n\n" ++

  -- LZ4 spec: last 5 bytes (LASTLITERALS) must be emitted as literals
  "        let match_limit = block_len - min(block_len, 5u);\n" ++
  "        while (ip + 4u <= match_limit) {\n" ++
  "            let cur4 = read4(input_word_start, ip);\n" ++
  "            let h = hash4(cur4);\n" ++
  "            let ref_pos = hash_table[h];\n" ++
  "            hash_table[h] = ip;\n\n" ++

  -- Check for match
  "            var match_len: u32 = 0u;\n" ++
  "            if (ref_pos != 0xFFFFFFFFu && ref_pos < ip && (ip - ref_pos) < 65536u) {\n" ++
  "                let ref4 = read4(input_word_start, ref_pos);\n" ++
  "                if (ref4 == cur4) {\n" ++
  "                    match_len = 4u;\n" ++
  "                    while (ip + match_len < match_limit && match_len < MAX_MATCH_LEN) {\n" ++
  "                        if (read_byte(input_word_start, ref_pos + match_len) != read_byte(input_word_start, ip + match_len)) {\n" ++
  "                            break;\n" ++
  "                        }\n" ++
  "                        match_len = match_len + 1u;\n" ++
  "                    }\n" ++
  "                }\n" ++
  "            }\n\n" ++

  -- Emit match or advance
  "            if (match_len >= MIN_MATCH_LEN) {\n" ++
  "                let literal_len = ip - anchor;\n" ++
  "                let match_offset = ip - ref_pos;\n" ++

  -- Token byte: upper nibble = literal len, lower nibble = match len - 4
  "                let lit_token = min(literal_len, 15u);\n" ++
  "                let match_token = min(match_len - MIN_MATCH_LEN, 15u);\n" ++
  "                write_byte(output_word_start, op, (lit_token << 4u) | match_token);\n" ++
  "                op = op + 1u;\n\n" ++

  -- Extended literal length
  "                if (literal_len >= 15u) {\n" ++
  "                    var rem = literal_len - 15u;\n" ++
  "                    while (rem >= 255u) {\n" ++
  "                        write_byte(output_word_start, op, 255u);\n" ++
  "                        op = op + 1u;\n" ++
  "                        rem = rem - 255u;\n" ++
  "                    }\n" ++
  "                    write_byte(output_word_start, op, rem);\n" ++
  "                    op = op + 1u;\n" ++
  "                }\n\n" ++

  -- Literal bytes
  "                for (var i: u32 = 0u; i < literal_len; i = i + 1u) {\n" ++
  "                    write_byte(output_word_start, op, read_byte(input_word_start, anchor + i));\n" ++
  "                    op = op + 1u;\n" ++
  "                }\n\n" ++

  -- Match offset (2 bytes LE)
  "                write_byte(output_word_start, op, match_offset & 0xFFu);\n" ++
  "                write_byte(output_word_start, op + 1u, (match_offset >> 8u) & 0xFFu);\n" ++
  "                op = op + 2u;\n\n" ++

  -- Extended match length
  "                if (match_len - MIN_MATCH_LEN >= 15u) {\n" ++
  "                    var rem = match_len - MIN_MATCH_LEN - 15u;\n" ++
  "                    while (rem >= 255u) {\n" ++
  "                        write_byte(output_word_start, op, 255u);\n" ++
  "                        op = op + 1u;\n" ++
  "                        rem = rem - 255u;\n" ++
  "                    }\n" ++
  "                    write_byte(output_word_start, op, rem);\n" ++
  "                    op = op + 1u;\n" ++
  "                }\n\n" ++

  "                ip = ip + match_len;\n" ++
  "                anchor = ip;\n" ++
  "            } else {\n" ++
  "                ip = ip + 1u;\n" ++
  "            }\n" ++
  "        }\n\n" ++

  -- Flush remaining literals (last sequence)
  "        let final_lit = block_len - anchor;\n" ++
  "        if (final_lit > 0u) {\n" ++
  "            let lit_token = min(final_lit, 15u);\n" ++
  "            write_byte(output_word_start, op, lit_token << 4u);\n" ++
  "            op = op + 1u;\n" ++
  "            if (final_lit >= 15u) {\n" ++
  "                var rem = final_lit - 15u;\n" ++
  "                while (rem >= 255u) {\n" ++
  "                    write_byte(output_word_start, op, 255u);\n" ++
  "                    op = op + 1u;\n" ++
  "                    rem = rem - 255u;\n" ++
  "                }\n" ++
  "                write_byte(output_word_start, op, rem);\n" ++
  "                op = op + 1u;\n" ++
  "            }\n" ++
  "            for (var i: u32 = 0u; i < final_lit; i = i + 1u) {\n" ++
  "                write_byte(output_word_start, op, read_byte(input_word_start, anchor + i));\n" ++
  "                op = op + 1u;\n" ++
  "            }\n" ++
  "        }\n\n" ++

  -- Store compressed size for this block (index 0 holds total_input_size)
  "        block_meta[1u + block_id * 2u] = op;\n" ++
  "    }\n" ++
  "}\n"

-- ---------------------------------------------------------------------------
-- CLIF IR: file read → GPU compress → file write
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
def clifIrSource : String := buildProgram do
  let fnRead ← declareFileRead
  let fnWrite ← declareFileWrite
  let gpu ← declareGpuFFI

  let ptr ← entryBlock

  -- Step 1: Read input file
  let inData  ← iconst64 inputData_off
  let zero    ← iconst64 0
  let bytesRead ← readFile ptr fnRead inputFilename_off inputData_off

  -- Step 2: Align up to multiple of 4 (wgpu COPY_BUFFER_ALIGNMENT)
  let alignedSz ← alignUp4 bytesRead

  -- Step 3: Compute numBlocks = (bytesRead + blockSize - 1) / blockSize
  let blkSzV ← iconst64 blockSize
  let c1 ← iconst64 1
  let bsm1 ← isub blkSzV c1
  let sum ← iadd bytesRead bsm1
  let numBlocks ← udiv sum blkSzV
  let numBlocks32 ← ireduce32 numBlocks

  -- Step 4: GPU init
  callVoid gpu.fnInit [ptr]

  -- Step 5: Create 3 buffers
  let buf0 ← call gpu.fnCreateBuffer [ptr, alignedSz]       -- input (aligned)
  let outBufSzV ← iconst64 outputBufSize
  let buf1 ← call gpu.fnCreateBuffer [ptr, outBufSzV]       -- output
  let metaSzV ← iconst64 metaSize
  let buf2 ← call gpu.fnCreateBuffer [ptr, metaSzV]         -- metadata

  -- Step 6: Write input size into metadata region
  let inputSz32 ← ireduce32 bytesRead
  let metaOffV ← iconst64 blockMeta_off
  let metaAddr ← iadd ptr metaOffV
  store inputSz32 metaAddr

  -- Step 7: Upload input data and metadata
  let _ ← call gpu.fnUpload [ptr, buf0, inData, alignedSz]
  let _ ← call gpu.fnUpload [ptr, buf2, metaOffV, metaSzV]

  -- Step 8: Create pipeline (3 bindings)
  let shOffV ← iconst64 shader_off
  let bdOffV ← iconst64 bindDesc_off
  let three32 ← iconst32 3
  let pipeId ← call gpu.fnCreatePipeline [ptr, shOffV, bdOffV, three32]

  -- Step 9: Dispatch — numBlocks workgroups
  let one32 ← iconst32 1
  let _ ← call gpu.fnDispatch [ptr, pipeId, numBlocks32, one32, one32]

  -- Step 10: Download output and metadata
  let outDataV ← iconst64 outputData_off
  let _ ← call gpu.fnDownload [ptr, buf1, outDataV, outBufSzV]
  let _ ← call gpu.fnDownload [ptr, buf2, metaOffV, metaSzV]

  -- Step 11: GPU cleanup
  callVoid gpu.fnCleanup [ptr]

  -- Step 12: Write standard LZ4 frame format
  -- Reuse inputData_off area as scratch
  let scratchAddr ← iadd ptr inData

  -- Write magic number 0x184D2204 LE
  let magic ← iconst32 0x184D2204
  store magic scratchAddr
  let c4 ← iconst64 4
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
  let c8 ← iconst64 8
  let maxCompBlkV ← iconst64 maxCompressedBlockSize

  let loopHdr ← declareBlock [.i64, .i64]   -- block_i, file_offset
  let bi := loopHdr.param 0
  let foff := loopHdr.param 1
  let loopDone ← declareBlock [.i64]        -- receives final file_offset
  let loopBody ← declareBlock []
  jump loopHdr.ref [zero, c7]
  startBlock loopHdr
  let done ← icmp .uge bi numBlocks
  brif done loopDone.ref [foff] loopBody.ref []

  startBlock loopBody
  -- Read compressed size from block_meta[1 + block_i*2] = metaOff + 4 + block_i*8
  let bi8 ← imul bi c8
  let bi8p4 ← iadd bi8 c4
  let metaRel ← iadd metaOffV bi8p4
  let metaAbsI ← iadd ptr metaRel
  let compSz32 ← load32 metaAbsI

  -- Write block_size (u32 LE) into scratch, then to file
  store compSz32 scratchAddr
  let _ ← call fnWrite [ptr, outFname, inData, foff, c4]

  -- Write block data
  let biTimesMax ← imul bi maxCompBlkV
  let blkDataRel ← iadd outDataV biTimesMax
  let compSz64 ← uextend64 compSz32
  let foffP4 ← iadd foff c4
  let _ ← call fnWrite [ptr, outFname, blkDataRel, foffP4, compSz64]

  -- Advance
  let nextFoff ← iadd foffP4 compSz64
  let nextBi ← iadd bi c1
  jump loopHdr.ref [nextBi, nextFoff]

  -- Write 4-byte end mark (0x00000000)
  startBlock loopDone
  let finalFoff := loopDone.param 0
  let endMark ← iconst32 0
  store endMark scratchAddr
  let _ ← call fnWrite [ptr, outFname, inData, finalFoff, c4]
  ret

-- ---------------------------------------------------------------------------
-- Payload construction
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  let reserved := zeros 0x40
  let hdrPad := zeros (bindDesc_off - 0x40)
  -- 3 binding descriptors: [buf_id (i32), read_only (i32)] × 3
  let bindDesc :=
    uint32ToBytes 0 ++ uint32ToBytes 1 ++   -- buf0: input, read_only=true
    uint32ToBytes 1 ++ uint32ToBytes 0 ++   -- buf1: output, read_only=false
    uint32ToBytes 2 ++ uint32ToBytes 0       -- buf2: metadata, read_only=false
  let bindPad := zeros (shader_off - bindDesc_off - 24)
  let shaderBytes := padTo (stringToBytes compressionShader) shaderRegionSize
  -- Input filename left as zeros (main.rs writes the path at runtime)
  let inputFnameBytes := zeros filenameRegionSize
  let outputFnameBytes := padTo (stringToBytes "compress_output.lz4") filenameRegionSize
  let flagBytes := uint64ToBytes 0
  let flagPad := zeros (clifIr_off - flag_off - 8)
  let clifPad := zeros clifIrRegionSize
  reserved ++ hdrPad ++
  bindDesc ++ bindPad ++
  shaderBytes ++ inputFnameBytes ++ outputFnameBytes ++
  flagBytes ++ flagPad ++ clifPad

-- ---------------------------------------------------------------------------
-- Algorithm definition
-- ---------------------------------------------------------------------------

def compressConfig : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := payloads.length + totalAdditionalMemory,
  context_offset := 0
}

def compressAlgorithm : Algorithm := {
    actions := [IR.clifCallAction],
    payloads := payloads,
    cranelift_units := 0,
    timeout_ms := some 120000
  }

end Algorithm

def main : IO Unit := do
  let json := toJsonPair Algorithm.compressConfig Algorithm.compressAlgorithm
  IO.println (Json.compress json)
