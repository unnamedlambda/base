import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace RegexBench

/-
  Regex benchmark algorithm — Cranelift JIT version.

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
    3. Scan words: count [a-z]+ing matches (len>=4, all lowercase, ends "ing")
    4. itoa count to OUTPUT_BUF
    5. cl_file_write result

  Word scan is scalar — checks each byte for lowercase, loads last 3 bytes
  at word end for branchless "ing" match.
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

-- fn1: Regex orchestrator
def clifRegexFn : String :=
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
  -- Setup constants
  "    v8 = iconst.i64 0\n" ++                       -- zero
  "    v9 = iconst.i64 32\n" ++                      -- space
  "    v10 = iconst.i64 97\n" ++                     -- 'a'
  "    v11 = iconst.i64 25\n" ++                     -- range for lowercase check
  "    v12 = iconst.i64 6778473\n" ++                -- "ing" as LE i24 (0x676E69)
  "    v13 = iconst.i64 4\n" ++                      -- min word length
  "    v14 = iconst.i64 10\n" ++                     -- newline / div 10
  "    v15 = iconst.i64 48\n" ++                     -- '0'
  "    v16 = iconst.i64 1\n" ++
  "    v17 = iconst.i64 848\n" ++                    -- OUTPUT_BUF (0x0350)
  "    v18 = iconst.i64 16777215\n" ++               -- 0xFFFFFF mask
  "    v19 = iadd v0, v4\n" ++                       -- data_ptr = base + INPUT_DATA
  "    jump block5(v8, v8)\n" ++                     -- (pos, count)
  "\n" ++
  -- block5: skip whitespace
  "block5(v20: i64, v21: i64):\n" ++
  "    v22 = icmp sge v20, v7\n" ++                  -- pos >= file_size?
  "    brif v22, block10(v21), block6(v20, v21)\n" ++
  "\n" ++
  "block6(v23: i64, v24: i64):\n" ++
  "    v25 = iadd v19, v23\n" ++                     -- data_ptr + pos
  "    v26 = uload8.i64 v25\n" ++
  "    v27 = icmp ule v26, v9\n" ++                  -- <= space?
  "    v28 = iadd_imm v23, 1\n" ++
  "    brif v27, block5(v28, v24), block8(v23, v24, v16, v23, v26)\n" ++
  "\n" ++
  -- block7: word loop (pos, count, all_lower, word_start)
  "block7(v40: i64, v41: i64, v42: i64, v43: i64):\n" ++
  "    v44 = iadd v19, v40\n" ++                     -- data_ptr + pos
  "    v45 = uload8.i64 v44\n" ++
  "    v46 = icmp ule v45, v9\n" ++                  -- whitespace?
  "    brif v46, block9(v40, v41, v42, v43), block8(v40, v41, v42, v43, v45)\n" ++
  "\n" ++
  -- block8: process byte — check lowercase only
  "block8(v70: i64, v71: i64, v72: i64, v73: i64, v74: i64):\n" ++
  "    v75 = isub v74, v10\n" ++                     -- byte - 'a'
  "    v76 = icmp ule v75, v11\n" ++                 -- <= 25?
  "    v77 = uextend.i64 v76\n" ++
  "    v78 = band v72, v77\n" ++                     -- all_lower &= is_lower
  "    v79 = iadd_imm v70, 1\n" ++                   -- pos++
  "    jump block7(v79, v71, v78, v73)\n" ++
  "\n" ++
  -- block9: word done — compute len, load last 3 bytes, check "ing"
  "block9(v50: i64, v51: i64, v52: i64, v53: i64):\n" ++
  "    v54 = isub v50, v53\n" ++                     -- len = ws_pos - word_start
  "    v55 = icmp sge v54, v13\n" ++                 -- len >= 4?
  "    v56 = uextend.i64 v55\n" ++
  "    v57 = band v52, v56\n" ++                     -- all_lower && len >= 4
  "    v58 = iadd_imm v50, -3\n" ++                  -- ws_pos - 3
  "    v59 = iadd v19, v58\n" ++                     -- data_ptr + (ws_pos - 3)
  "    v60 = load.i32 v59\n" ++                      -- load 4 bytes
  "    v61 = uextend.i64 v60\n" ++
  "    v62 = band v61, v18\n" ++                     -- mask to 3 bytes
  "    v63 = icmp eq v62, v12\n" ++                  -- == "ing" (LE)?
  "    v64 = uextend.i64 v63\n" ++
  "    v65 = band v57, v64\n" ++                     -- full match
  "    v66 = iadd v51, v65\n" ++                     -- count += match
  "    v67 = iadd_imm v50, 1\n" ++                   -- past whitespace
  "    jump block5(v67, v66)\n" ++
  "\n" ++
  -- block10: itoa count to OUTPUT_BUF, then file_write
  "block10(v90: i64):\n" ++
  "    jump block11(v90, v16)\n" ++
  "\n" ++
  -- find divisor
  "block11(v91: i64, v92: i64):\n" ++
  "    v93 = imul v92, v14\n" ++                     -- div * 10
  "    v94 = icmp ugt v93, v91\n" ++
  "    brif v94, block12(v91, v92, v17), block11(v91, v93)\n" ++
  "\n" ++
  -- write digits
  "block12(v95: i64, v96: i64, v97: i64):\n" ++
  "    v98 = udiv v95, v96\n" ++
  "    v99 = iadd v98, v15\n" ++                     -- + '0'
  "    v100 = iadd v0, v97\n" ++
  "    istore8 v99, v100\n" ++
  "    v101 = imul v98, v96\n" ++
  "    v102 = isub v95, v101\n" ++
  "    v103 = udiv v96, v14\n" ++                    -- div / 10
  "    v104 = iadd_imm v97, 1\n" ++
  "    v105 = icmp eq v103, v8\n" ++                 -- div == 0?
  "    brif v105, block13(v104), block12(v102, v103, v104)\n" ++
  "\n" ++
  -- write newline + null, then file_write
  "block13(v106: i64):\n" ++
  "    v107 = iadd v0, v106\n" ++
  "    istore8 v14, v107\n" ++                       -- '\\n'
  "    v108 = iadd_imm v106, 1\n" ++
  "    v109 = iadd v0, v108\n" ++
  "    v110 = iconst.i32 0\n" ++
  "    istore8 v110, v109\n" ++
  -- Write result file
  "    v150 = iconst.i64 512\n" ++                   -- OUTPUT_PATH_OFF
  "    v151 = iconst.i64 848\n" ++                   -- OUTPUT_BUF
  "    v152 = iconst.i64 0\n" ++
  "    v153 = iconst.i64 0\n" ++
  "    v154 = call fn1(v0, v150, v151, v152, v153)\n" ++
  "    return\n" ++
  "}\n"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifRegexFn

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

end RegexBench

def main : IO Unit := do
  let json := toJsonPair RegexBench.buildConfig RegexBench.buildAlgorithm
  IO.println (Json.compress json)
