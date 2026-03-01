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

def clifIrSource : String :=
  let inFname := toString inputFilename_off
  let outFname := toString outputFilename_off
  let inData := toString inputData_off
  let bufA := toString bufA_off
  let bufB := toString bufB_off
  let metaOff := toString meta_off
  let shOff := toString shader_off
  let bdOff := toString bindDesc_off
  let metaSzStr := toString metaSize

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

  -- Constants
  "    v1 = iconst.i64 0\n" ++
  "    v2 = iconst.i64 1\n" ++
  "    v3 = iconst.i64 4\n" ++
  "    v4 = iconst.i64 8\n" ++

  -- Step 1: Read input file
  s!"    v10 = iconst.i64 {inFname}\n" ++
  s!"    v11 = iconst.i64 {inData}\n" ++
  "    v12 = call fn0(v0, v10, v11, v1, v1)\n" ++   -- v12 = bytes read

  -- Compute N = bytes_read / 8 (each complex is 2 × f32 = 8 bytes)
  "    v13 = iconst.i64 3\n" ++
  "    v14 = ushr v12, v13\n" ++                     -- N = bytes_read >> 3

  -- Step 2: Bit-reverse permutation from inputData → bufA
  -- For each index i in [0, N), compute bit-reversed index rev(i),
  -- copy complex number from inputData[i] to bufA[rev(i)]
  -- We need log2(N) bits. Compute log2(N) first.
  -- log2(N): count trailing zeros of N (since N is power of 2)
  -- Actually easier: loop from N, shift right until 1
  "    jump block1(v14, v1)\n" ++  -- tmp = N, log2n = 0

  "\n" ++
  -- Compute log2(N)
  "block1(v20: i64, v21: i64):\n" ++  -- tmp, log2n
  "    v22 = icmp ugt v20, v2\n" ++   -- tmp > 1?
  "    brif v22, block2, block3(v21)\n" ++

  "\n" ++
  "block2:\n" ++
  "    v23 = ushr v20, v2\n" ++        -- tmp >> 1
  "    v24 = iadd v21, v2\n" ++        -- log2n + 1
  "    jump block1(v23, v24)\n" ++

  "\n" ++
  -- v30 = log2(N), now do bit-reverse permutation
  "block3(v30: i64):\n" ++
  s!"    v31 = iconst.i64 {bufA}\n" ++
  "    jump block4(v1)\n" ++           -- i = 0

  "\n" ++
  -- Bit-reverse loop: for i in [0, N)
  "block4(v40: i64):\n" ++            -- i
  "    v41 = icmp uge v40, v14\n" ++  -- i >= N?
  "    brif v41, block10, block5\n" ++

  "\n" ++
  -- Compute bit-reverse of v40 with v30 bits
  "block5:\n" ++
  "    jump block6(v40, v1, v1)\n" ++  -- val=i, rev=0, bit=0

  "\n" ++
  "block6(v50: i64, v51: i64, v52: i64):\n" ++  -- val, rev, bit
  "    v53 = icmp uge v52, v30\n" ++             -- bit >= log2n?
  "    brif v53, block7(v51), block8\n" ++

  "\n" ++
  "block8:\n" ++
  "    v54 = band v50, v2\n" ++        -- val & 1
  "    v55 = ishl v51, v2\n" ++        -- rev << 1
  "    v56 = bor v55, v54\n" ++        -- rev = (rev << 1) | (val & 1)
  "    v57 = ushr v50, v2\n" ++        -- val >> 1
  "    v58 = iadd v52, v2\n" ++        -- bit + 1
  "    jump block6(v57, v56, v58)\n" ++

  "\n" ++
  -- Copy inputData[i] (8 bytes) to bufA[rev]
  "block7(v60: i64):\n" ++            -- rev
  -- src = inputData_off + i * 8
  "    v61 = imul v40, v4\n" ++       -- i * 8 (v4 = 8)
  "    v62 = iadd v11, v61\n" ++      -- inputData_off + i*8
  "    v63 = iadd v0, v62\n" ++       -- absolute src

  -- dst = bufA_off + rev * 8
  "    v64 = imul v60, v4\n" ++       -- rev * 8
  "    v65 = iadd v31, v64\n" ++      -- bufA_off + rev*8
  "    v66 = iadd v0, v65\n" ++       -- absolute dst

  -- Copy 8 bytes (load as two i32, store as two i32)
  "    v67 = load.i32 v63\n" ++       -- first 4 bytes (real)
  "    store v67, v66\n" ++
  "    v68 = iadd v63, v3\n" ++       -- src + 4 (v3 = 4)
  "    v69 = iadd v66, v3\n" ++       -- dst + 4
  "    v70 = load.i32 v68\n" ++       -- next 4 bytes (imag)
  "    store v70, v69\n" ++

  "    v71 = iadd v40, v2\n" ++       -- i + 1
  "    jump block4(v71)\n" ++

  "\n" ++
  -- Step 3: GPU init
  "block10:\n" ++
  "    call fn1(v0)\n" ++

  -- Align data size to multiple of 4
  "    v100 = imul v14, v4\n" ++       -- N * 8 = actual data bytes
  "    v101 = iconst.i64 3\n" ++
  "    v102 = iadd v100, v101\n" ++
  "    v103 = iconst.i64 -4\n" ++
  "    v104 = band v102, v103\n" ++    -- aligned data size

  -- Create 3 buffers: buf_a (rw), buf_b (rw), meta (read)
  "    v105 = call fn2(v0, v104)\n" ++  -- buf0 = buf_a
  "    v106 = call fn2(v0, v104)\n" ++  -- buf1 = buf_b
  s!"    v107 = iconst.i64 {metaSzStr}\n" ++
  "    v108 = call fn2(v0, v107)\n" ++  -- buf2 = meta

  -- Write N into meta region
  s!"    v109 = iconst.i64 {metaOff}\n" ++
  "    v110 = iadd v0, v109\n" ++
  "    v111 = ireduce.i32 v14\n" ++    -- N as i32
  "    store v111, v110\n" ++

  -- Upload buf_a data to GPU
  "    v112 = call fn3(v0, v105, v31, v104)\n" ++

  -- Create pipeline (3 bindings)
  s!"    v113 = iconst.i64 {shOff}\n" ++
  s!"    v114 = iconst.i64 {bdOff}\n" ++
  "    v115 = iconst.i32 3\n" ++
  "    v116 = call fn4(v0, v113, v114, v115)\n" ++

  -- Compute dispatch size: N/2 threads, workgroup_size=64
  -- workgroups = ceil(N/2 / 64) = (N/2 + 63) / 64
  "    v117 = ushr v14, v2\n" ++        -- N/2
  "    v118 = iconst.i64 63\n" ++
  "    v119 = iadd v117, v118\n" ++
  "    v120 = iconst.i64 6\n" ++
  "    v121 = ushr v119, v120\n" ++     -- workgroups = (N/2 + 63) >> 6
  "    v122 = ireduce.i32 v121\n" ++    -- as i32
  "    v123 = iconst.i32 1\n" ++

  -- Step 4: Loop log2(N) stages
  "    jump block11(v1, v1)\n" ++       -- stage=0, direction=0

  "\n" ++
  -- Stage loop
  "block11(v130: i64, v131: i64):\n" ++ -- stage, direction
  "    v132 = icmp uge v130, v30\n" ++   -- stage >= log2(N)?
  "    brif v132, block13(v131), block12\n" ++

  "\n" ++
  "block12:\n" ++
  -- Write stage and direction into meta
  "    v133 = iadd v110, v3\n" ++        -- meta_off + 4 (stage field)
  "    v134 = ireduce.i32 v130\n" ++     -- stage as i32
  "    store v134, v133\n" ++
  "    v135 = iadd v133, v3\n" ++        -- meta_off + 8 (direction field)
  "    v136 = ireduce.i32 v131\n" ++     -- direction as i32
  "    store v136, v135\n" ++

  -- Upload meta
  "    v137 = call fn3(v0, v108, v109, v107)\n" ++

  -- Dispatch
  "    v138 = call fn5(v0, v116, v122, v123, v123)\n" ++

  -- Sync: download meta buffer to scratch area to force GPU completion
  -- This ensures the dispatch finishes before next meta upload
  "    v170 = iconst.i64 64\n" ++        -- scratch offset 0x40
  "    v171 = call fn6(v0, v108, v170, v107)\n" ++

  -- Next stage, toggle direction
  "    v139 = iadd v130, v2\n" ++        -- stage + 1
  "    v140 = bxor v131, v2\n" ++        -- direction ^ 1
  "    jump block11(v139, v140)\n" ++

  "\n" ++
  -- Step 5: Download result
  -- If direction == 0, result is in buf_a (last write went to buf_a)
  -- Wait, direction tracks which direction the NEXT pass would use:
  -- direction=0 means "read buf_a, write buf_b", so after toggling,
  -- if we end with direction=0, last write was to buf_a (toggle from 1 to 0 means last was dir=1 → wrote to buf_a)
  -- Actually: direction starts at 0. Stage 0: dir=0 → read A write B → toggle to 1
  -- Stage 1: dir=1 → read B write A → toggle to 0
  -- After all stages, direction tells us what the NEXT would be.
  -- If direction=0 after loop: last was dir=1 → wrote to buf_a → download buf_a
  -- If direction=1 after loop: last was dir=0 → wrote to buf_b → download buf_b
  "block13(v150: i64):\n" ++            -- final direction
  s!"    v151 = iconst.i64 {bufA}\n" ++
  s!"    v152 = iconst.i64 {bufB}\n" ++
  "    v153 = icmp eq v150, v1\n" ++    -- direction == 0? → result in buf_a
  "    brif v153, block14(v151, v105), block14(v152, v106)\n" ++

  "\n" ++
  "block14(v160: i64, v161: i32):\n" ++ -- dst_off (relative), gpu_buf_id
  -- Download from GPU to the chosen buffer region
  "    v162 = call fn6(v0, v161, v160, v104)\n" ++

  -- GPU cleanup
  "    call fn7(v0)\n" ++

  -- Step 6: Write output file
  s!"    v163 = iconst.i64 {outFname}\n" ++
  "    v164 = call fn8(v0, v163, v160, v1, v100)\n" ++

  "    return\n" ++
  "}\n"

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

def fftAlgorithm : Algorithm :=
  let clifCallAction : Action :=
    { kind := .ClifCall, dst := u32 0, src := u32 1, offset := u32 0, size := u32 0 }
  {
    actions := [clifCallAction],
    payloads := payloads,
    cranelift_units := 0,
    timeout_ms := some 300000
  }

end Algorithm

def main : IO Unit := do
  let json := toJsonPair Algorithm.fftConfig Algorithm.fftAlgorithm
  IO.println (Json.compress json)
