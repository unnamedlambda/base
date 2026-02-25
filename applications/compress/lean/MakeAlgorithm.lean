import Lean
import Std
import AlgorithmLib

open Lean
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

  "        while (ip + 4u <= block_len) {\n" ++
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
  "                    while (ip + match_len < block_len && match_len < MAX_MATCH_LEN) {\n" ++
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

def clifIrSource : String :=
  let inFname := toString inputFilename_off
  let outFname := toString outputFilename_off
  let inData := toString inputData_off
  let outData := toString outputData_off
  let metaOff := toString blockMeta_off
  let shOff := toString shader_off
  let bdOff := toString bindDesc_off
  let outBufSz := toString outputBufSize
  let metaSz := toString metaSize
  let blkSz := toString blockSize
  let maxCompBlk := toString maxCompressedBlockSize
  -- noop function
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n\n" ++
  -- orchestrator
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++                                   -- gpu_init, gpu_cleanup
  "    sig1 = (i64, i64) -> i32 system_v\n" ++                       -- gpu_create_buffer
  "    sig2 = (i64, i64, i64, i32) -> i32 system_v\n" ++             -- gpu_create_pipeline
  "    sig3 = (i64, i32, i64, i64) -> i32 system_v\n" ++             -- gpu_upload, gpu_download
  "    sig4 = (i64, i32, i32, i32, i32) -> i32 system_v\n" ++        -- gpu_dispatch
  "    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++        -- file_read, file_write
  "    fn0 = %cl_file_read sig5\n" ++
  "    fn1 = %cl_gpu_init sig0\n" ++
  "    fn2 = %cl_gpu_create_buffer sig1\n" ++
  "    fn3 = %cl_gpu_upload sig3\n" ++
  "    fn4 = %cl_gpu_create_pipeline sig2\n" ++
  "    fn5 = %cl_gpu_dispatch sig4\n" ++
  "    fn6 = %cl_gpu_download sig3\n" ++
  "    fn7 = %cl_gpu_cleanup sig0\n" ++
  "    fn8 = %cl_file_write sig5\n" ++
  "\n" ++
  "block0(v0: i64):\n" ++
  -- Step 1: Read input file (size=0 reads entire file, returns byte count)
  s!"    v1 = iconst.i64 {inFname}\n" ++
  s!"    v2 = iconst.i64 {inData}\n" ++
  "    v3 = iconst.i64 0\n" ++
  "    v4 = call fn0(v0, v1, v2, v3, v3)\n" ++   -- v4 = bytes read
  "\n" ++
  -- Step 2: Align v4 up to multiple of 4 (wgpu COPY_BUFFER_ALIGNMENT)
  "    v50 = iconst.i64 3\n" ++
  "    v51 = iadd v4, v50\n" ++
  "    v52 = iconst.i64 -4\n" ++          -- 0xFFFF...FFFC
  "    v53 = band v51, v52\n" ++           -- v53 = aligned input size
  "\n" ++
  -- Step 3: Compute numBlocks = (v4 + blockSize - 1) / blockSize
  s!"    v5 = iconst.i64 {blkSz}\n" ++
  "    v6 = iconst.i64 1\n" ++
  "    v7 = isub v5, v6\n" ++             -- blockSize - 1
  "    v8 = iadd v4, v7\n" ++             -- bytes_read + blockSize - 1
  "    v9 = udiv v8, v5\n" ++             -- numBlocks (i64)
  "    v10 = ireduce.i32 v9\n" ++         -- numBlocks as i32 for dispatch
  "\n" ++
  -- Step 4: GPU init
  "    call fn1(v0)\n" ++
  "\n" ++
  -- Step 5: Create 3 buffers (use aligned size for input)
  "    v11 = call fn2(v0, v53)\n" ++      -- buf0 = input (aligned)
  s!"    v12 = iconst.i64 {outBufSz}\n" ++
  "    v13 = call fn2(v0, v12)\n" ++      -- buf1 = output
  s!"    v14 = iconst.i64 {metaSz}\n" ++
  "    v15 = call fn2(v0, v14)\n" ++      -- buf2 = metadata
  "\n" ++
  -- Step 6: Write actual input size into metadata region for shader
  --   store v4 (bytes_read) as u32 at blockMeta_off in shared memory
  "    v40 = ireduce.i32 v4\n" ++        -- input size as i32
  s!"    v41 = iconst.i64 {metaOff}\n" ++
  "    v42 = iadd v0, v41\n" ++          -- absolute address of metadata
  "    store v40, v42\n" ++              -- store input size at block_meta[0]
  "\n" ++
  -- Step 7: Upload input data (aligned size)
  "    v16 = call fn3(v0, v11, v2, v53)\n" ++
  -- Upload metadata buffer (contains input size at index 0)
  "    v43 = call fn3(v0, v15, v41, v14)\n" ++
  "\n" ++
  -- Step 7: Create pipeline (3 bindings)
  s!"    v17 = iconst.i64 {shOff}\n" ++
  s!"    v18 = iconst.i64 {bdOff}\n" ++
  "    v19 = iconst.i32 3\n" ++
  "    v20 = call fn4(v0, v17, v18, v19)\n" ++
  "\n" ++
  -- Step 8: Dispatch — numBlocks workgroups in x
  "    v21 = iconst.i32 1\n" ++
  "    v22 = call fn5(v0, v20, v10, v21, v21)\n" ++
  "\n" ++
  -- Step 9: Download output and metadata
  s!"    v23 = iconst.i64 {outData}\n" ++
  "    v24 = call fn6(v0, v13, v23, v12)\n" ++
  "    v26 = call fn6(v0, v15, v41, v14)\n" ++
  "\n" ++
  -- Step 10: GPU cleanup
  "    call fn7(v0)\n" ++
  "\n" ++
  -- Step 11: Write file header: [num_blocks:u32 LE][original_size:u32 LE][block_sizes:u32 LE × N]
  -- First, build a header in shared memory at a scratch area (reuse inputData_off area,
  -- since we no longer need the input data).
  -- Header layout: 4 bytes numBlocks + 4 bytes origSize + numBlocks * 4 bytes block_sizes
  -- We'll write header + compressed block data to file using two file_write calls.
  --
  -- For simplicity: write numBlocks * maxCompressedBlockSize of output data.
  -- The metadata tells the decompressor how many bytes per block are valid.
  -- Total output = 8 + numBlocks*4 (header) + numBlocks*maxCompressedBlockSize (data)
  --
  -- Write header into input data area (reused as scratch)
  "    v60 = iadd v0, v2\n" ++            -- absolute addr of inputData_off area (scratch)
  "    store v10, v60\n" ++               -- [0..3] = numBlocks (v10 = ireduce.i32 v9)
  "    v62 = iconst.i64 4\n" ++
  "    v63 = iadd v60, v62\n" ++
  "    store v40, v63\n" ++               -- [4..7] = original_size (v40 = ireduce.i32 v4)
  "\n" ++
  -- Write per-block compressed sizes from metadata into header
  -- block_meta layout: [0]=input_size, then [1 + block_id*2]=compressed_size
  -- Read from metaOff+4 (skip index 0 = input_size), stride 8 (every other u32)
  -- Write to scratch+8, stride 4
  "    v64 = iconst.i64 8\n" ++
  "    jump block1(v3)\n" ++              -- loop counter i=0
  "\n" ++
  "block1(v70: i64):\n" ++
  "    v71 = icmp uge v70, v9\n" ++       -- i >= numBlocks?
  "    brif v71, block2, block3\n" ++
  "\n" ++
  "block3:\n" ++
  -- Read compressed size: metadata at metaOff + 4 + i*8
  "    v72 = imul v70, v64\n" ++          -- i * 8
  "    v73 = iadd v72, v62\n" ++          -- i * 8 + 4
  "    v74 = iadd v41, v73\n" ++          -- metaOff + 4 + i*8
  "    v75 = iadd v0, v74\n" ++           -- absolute addr
  "    v76 = load.i32 v75\n" ++           -- compressed size for block i
  -- Write to scratch: scratch + 8 + i*4
  "    v77 = imul v70, v62\n" ++          -- i * 4
  "    v78 = iadd v77, v64\n" ++          -- i * 4 + 8
  "    v79 = iadd v60, v78\n" ++          -- absolute addr in scratch
  "    store v76, v79\n" ++
  -- i++
  "    v80 = iadd v70, v6\n" ++           -- v6 = 1
  "    jump block1(v80)\n" ++
  "\n" ++
  "block2:\n" ++
  -- Write header: 8 + numBlocks * 4 bytes
  "    v81 = imul v9, v62\n" ++           -- numBlocks * 4
  "    v82 = iadd v81, v64\n" ++          -- header size = numBlocks*4 + 8
  s!"    v27 = iconst.i64 {outFname}\n" ++
  "    v28 = call fn8(v0, v27, v2, v3, v82)\n" ++    -- write header at file offset 0
  "\n" ++
  -- Write compressed block data after header
  s!"    v30 = iconst.i64 {maxCompBlk}\n" ++
  "    v31 = imul v9, v30\n" ++           -- total block data size
  "    v32 = call fn8(v0, v27, v23, v82, v31)\n" ++  -- write data at file offset = header size
  "\n" ++
  "    return\n" ++
  "}\n"

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
  let clifBytes := padTo (stringToBytes clifIrSource) clifIrRegionSize
  reserved ++ hdrPad ++
  bindDesc ++ bindPad ++
  shaderBytes ++ inputFnameBytes ++ outputFnameBytes ++
  flagBytes ++ flagPad ++
  clifBytes

-- ---------------------------------------------------------------------------
-- Algorithm definition
-- ---------------------------------------------------------------------------

def compressAlgorithm : Algorithm :=
  let clifCallAction : Action :=
    { kind := .ClifCall, dst := u32 0, src := u32 1, offset := u32 0, size := u32 0 }
  {
    actions := [clifCallAction],
    payloads := payloads,
    state := { cranelift_ir_offsets := [clifIr_off] },
    units := { cranelift_units := 0 },
    cranelift_assignments := [0],
    worker_threads := some 1,
    blocking_threads := some 1,
    stack_size := some (512 * 1024),
    timeout_ms := some 120000,
    thread_name_prefix := some "compress",
    additional_shared_memory := totalAdditionalMemory
  }

end Algorithm

def main : IO Unit := do
  let json := toJson Algorithm.compressAlgorithm
  IO.println (Json.compress json)
