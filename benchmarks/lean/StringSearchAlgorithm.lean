import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace StringSearchBench

/-
  String search benchmark algorithm — Cranelift JIT version.

  Payload (via execute data arg): "input_path\0output_path\0"

  Memory layout (shared memory):
    0x0000  RESERVED        (40 bytes, runtime-managed: ctx_ptr, data/out ptrs)
    0x0100  INPUT_PATH      (256 bytes, copied from payload by CLIF)
    0x0200  OUTPUT_PATH     (256 bytes, copied from payload by CLIF)
    0x0350  OUTPUT_BUF      (64 bytes, itoa result for FileWrite)
    0x4000  INPUT_DATA      (variable, populated by FileRead)

  Single CLIF function:
    1. Copy input/output paths from payload into shared memory
    2. cl_file_read → gets bytes_read (used as file_size)
    3. SIMD 4-byte pattern match: 4 loads at offsets 0-3, AND all matches,
       popcnt bitmask to count all "that" occurrences in 16-position chunks
    4. itoa count to OUTPUT_BUF
    5. cl_file_write result

  No scalar cleanup needed: shared memory is zero-initialized past data,
  so partial reads past end produce no false positives.
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def OUTPUT_BUF      : Nat := 0x0350
def INPUT_DATA      : Nat := 0x4000
def MAX_TEXT_BYTES   : Nat := 512 * 1024 * 1024  -- 512MB max
def MEM_SIZE        : Nat := INPUT_DATA + MAX_TEXT_BYTES
def TIMEOUT_MS      : Nat := 300000

-- fn0: noop
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: String search orchestrator
def clifSearchFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "    sig1 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "\n" ++
  "    fn0 = %cl_file_read sig0\n" ++
  "    fn1 = %cl_file_write sig1\n" ++
  "\n" ++
  "block0(v0: i64):\n" ++
  "    v1 = load.i64 notrap aligned v0+0x08\n" ++    -- data_ptr (payload)
  "    v600 = iconst.i64 0\n" ++
  "    jump block1(v600)\n" ++
  "\n" ++
  -- Copy input path from payload to shared memory at INPUT_PATH_OFF (0x0100)
  "block1(v201: i64):\n" ++
  "    v202 = iadd v1, v201\n" ++
  "    v203 = uload8.i64 notrap v202\n" ++
  "    v204 = iadd_imm v0, 256\n" ++                 -- INPUT_PATH_OFF
  "    v205 = iadd v204, v201\n" ++
  "    istore8 v203, v205\n" ++
  "    v206 = icmp_imm eq v203, 0\n" ++
  "    v207 = iadd_imm v201, 1\n" ++
  "    brif v206, block2(v207), block1(v207)\n" ++
  "\n" ++
  -- Copy output path from payload to shared memory at OUTPUT_PATH_OFF (0x0200)
  "block2(v210: i64):\n" ++
  "    v211 = iconst.i64 0\n" ++
  "    jump block3(v210, v211)\n" ++
  "\n" ++
  "block3(v220: i64, v221: i64):\n" ++
  "    v222 = iadd v1, v220\n" ++
  "    v223 = uload8.i64 notrap v222\n" ++
  "    v224 = iadd_imm v0, 512\n" ++                 -- OUTPUT_PATH_OFF
  "    v225 = iadd v224, v221\n" ++
  "    istore8 v223, v225\n" ++
  "    v226 = icmp_imm eq v223, 0\n" ++
  "    v227 = iadd_imm v220, 1\n" ++
  "    v228 = iadd_imm v221, 1\n" ++
  "    brif v226, block4, block3(v227, v228)\n" ++
  "\n" ++
  -- Read text file into shared memory
  "block4:\n" ++
  "    v3 = iconst.i64 256\n" ++                     -- INPUT_PATH_OFF
  "    v4 = iconst.i64 16384\n" ++                   -- INPUT_DATA (0x4000)
  "    v5 = iconst.i64 0\n" ++
  "    v700 = iconst.i64 0\n" ++
  "    v7 = call fn0(v0, v3, v4, v5, v700)\n" ++    -- bytes_read = file_size
  -- Setup constants for SIMD search
  "    v8 = iconst.i64 0\n" ++                       -- zero
  "    v9 = iconst.i64 4\n" ++                       -- pattern length
  "    v10 = isub v7, v9\n" ++                       -- end = file_size - 4
  "    v11 = iconst.i64 1\n" ++
  "    v12 = iconst.i64 10\n" ++                     -- newline / div 10
  "    v13 = iconst.i64 48\n" ++                     -- '0'
  "    v14 = iconst.i64 848\n" ++                    -- OUTPUT_BUF (0x0350)
  "    v15 = iadd v0, v4\n" ++                       -- data_ptr = base + INPUT_DATA
  "    v16 = iconst.i8 116\n" ++                     -- 't' (0x74)
  "    v17 = splat.i8x16 v16\n" ++                   -- broadcast 't'
  "    v18 = iconst.i8 104\n" ++                     -- 'h' (0x68)
  "    v19 = splat.i8x16 v18\n" ++                   -- broadcast 'h'
  "    v20 = iconst.i8 97\n" ++                      -- 'a' (0x61)
  "    v21 = splat.i8x16 v20\n" ++                   -- broadcast 'a'
  "    v22 = iconst.i32 0\n" ++                      -- zero i32
  "    jump block5(v8, v8)\n" ++
  "\n" ++
  -- block5: SIMD loop (pos, count)
  "block5(v30: i64, v31: i64):\n" ++
  "    v32 = icmp sgt v30, v10\n" ++                 -- pos > end?
  "    brif v32, block10(v31), block6(v30, v31)\n" ++
  "\n" ++
  -- block6: SIMD 4-byte pattern match across 16 positions
  "block6(v33: i64, v34: i64):\n" ++
  "    v35 = iadd v15, v33\n" ++                     -- data_ptr + pos
  "    v36 = load.i8x16 v35\n" ++                    -- 16 bytes at offset 0
  "    v37 = load.i8x16 v35+1\n" ++                  -- 16 bytes at offset 1
  "    v38 = load.i8x16 v35+2\n" ++                  -- 16 bytes at offset 2
  "    v39 = load.i8x16 v35+3\n" ++                  -- 16 bytes at offset 3
  "    v40 = icmp eq v36, v17\n" ++                   -- byte[i] == 't'?
  "    v41 = icmp eq v37, v19\n" ++                   -- byte[i+1] == 'h'?
  "    v42 = icmp eq v38, v21\n" ++                   -- byte[i+2] == 'a'?
  "    v43 = icmp eq v39, v17\n" ++                   -- byte[i+3] == 't'?
  "    v44 = band v40, v41\n" ++                     -- match bytes 0,1
  "    v45 = band v42, v43\n" ++                     -- match bytes 2,3
  "    v46 = band v44, v45\n" ++                     -- all 4 match
  "    v47 = vhigh_bits.i32 v46\n" ++                -- extract 16-bit mask
  "    v48 = popcnt v47\n" ++                        -- count matches
  "    v49 = uextend.i64 v48\n" ++
  "    v50 = iadd v34, v49\n" ++                     -- count += matches
  "    v51 = iadd_imm v33, 16\n" ++                  -- advance 16
  "    jump block5(v51, v50)\n" ++
  "\n" ++
  -- block10: itoa count to OUTPUT_BUF, then file_write
  "block10(v60: i64):\n" ++
  "    jump block11(v60, v11)\n" ++
  "\n" ++
  -- find divisor
  "block11(v61: i64, v62: i64):\n" ++
  "    v63 = imul v62, v12\n" ++                     -- div * 10
  "    v64 = icmp ugt v63, v61\n" ++
  "    brif v64, block12(v61, v62, v14), block11(v61, v63)\n" ++
  "\n" ++
  -- write digits
  "block12(v65: i64, v66: i64, v67: i64):\n" ++
  "    v68 = udiv v65, v66\n" ++
  "    v69 = iadd v68, v13\n" ++                     -- + '0'
  "    v70 = iadd v0, v67\n" ++
  "    istore8 v69, v70\n" ++
  "    v71 = imul v68, v66\n" ++
  "    v72 = isub v65, v71\n" ++
  "    v73 = udiv v66, v12\n" ++                     -- div / 10
  "    v74 = iadd_imm v67, 1\n" ++
  "    v75 = icmp eq v73, v8\n" ++                   -- div == 0?
  "    brif v75, block13(v74), block12(v72, v73, v74)\n" ++
  "\n" ++
  -- write newline + null, then file_write
  "block13(v76: i64):\n" ++
  "    v77 = iadd v0, v76\n" ++
  "    istore8 v12, v77\n" ++                        -- '\\n'
  "    v78 = iadd_imm v76, 1\n" ++
  "    v79 = iadd v0, v78\n" ++
  "    v80 = iconst.i32 0\n" ++
  "    istore8 v80, v79\n" ++
  -- Write result file
  "    v150 = iconst.i64 512\n" ++                   -- OUTPUT_PATH_OFF
  "    v151 = iconst.i64 848\n" ++                   -- OUTPUT_BUF
  "    v152 = iconst.i64 0\n" ++
  "    v153 = iconst.i64 0\n" ++
  "    v154 = call fn1(v0, v150, v151, v152, v153)\n" ++
  "    return\n" ++
  "}\n"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifSearchFn

def controlActions : List Action :=
  [{ kind := .ClifCall, dst := 0, src := 1, offset := 0, size := 0 }]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0
}

def buildAlgorithm : Algorithm := {
  actions := controlActions,
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

end StringSearchBench

def main : IO Unit := do
  let json := toJsonPair StringSearchBench.buildConfig StringSearchBench.buildAlgorithm
  IO.println (Json.compress json)
