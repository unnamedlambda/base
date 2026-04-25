import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace JsonBench

/-
  JSON benchmark algorithm — Cranelift JIT version.

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
    3. SIMD JSON parse: scan for "value": pattern, sum digits
    4. itoa total to OUTPUT_BUF
    5. cl_file_write result

  Uses 16-byte SIMD vectors (i8x16) for 2-byte prefix scanning ("v).
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def OUTPUT_BUF      : Nat := 0x0350
def INPUT_DATA      : Nat := 0x4000
def MAX_JSON_BYTES  : Nat := 512 * 1024 * 1024  -- 512MB max
def MEM_SIZE        : Nat := INPUT_DATA + MAX_JSON_BYTES
def TIMEOUT_MS      : Nat := 300000

-- fn0: noop
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: JSON orchestrator
-- Copies paths from payload, reads file, parses JSON, writes result
def clifJsonFn : String :=
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
  -- Read JSON file into shared memory
  "block4:\n" ++
  "    v3 = iconst.i64 256\n" ++                     -- INPUT_PATH_OFF
  "    v4 = iconst.i64 16384\n" ++                   -- INPUT_DATA (0x4000)
  "    v5 = iconst.i64 0\n" ++
  "    v700 = iconst.i64 0\n" ++
  "    v7 = call fn0(v0, v3, v4, v5, v700)\n" ++    -- bytes_read = file_size
  -- Setup constants for SIMD scan
  "    v8 = iconst.i64 0\n" ++                       -- zero i64
  "    v9 = iconst.i64 9\n" ++                       -- needle length / digit range
  "    v10 = isub v7, v9\n" ++                       -- end = file_size - 9
  "    v11 = iconst.i64 10\n" ++                     -- for * 10
  "    v12 = iconst.i64 48\n" ++                     -- '0'
  "    v13 = iadd v0, v4\n" ++                       -- data_ptr = base + INPUT_DATA
  "    v14 = iconst.i8 34\n" ++                      -- '\"' as i8
  "    v15 = splat.i8x16 v14\n" ++                   -- broadcast '\"'
  "    v16 = iconst.i8 118\n" ++                     -- 'v' as i8
  "    v17 = splat.i8x16 v16\n" ++                   -- broadcast 'v'
  "    v18 = iconst.i64 2322206377019990390\n" ++    -- \"value\": <sp> as LE i64 (bytes 1-8)
  "    v19 = iconst.i32 0\n" ++                      -- zero i32 for mask comparison
  "    jump block5(v8, v8)\n" ++
  "\n" ++
  -- block5: SIMD scan entry (pos, total)
  "block5(v20: i64, v21: i64):\n" ++
  "    v22 = icmp sgt v20, v10\n" ++                 -- pos > end?
  "    brif v22, block10(v21), block6(v20, v21)\n" ++
  "\n" ++
  -- block6: 2-byte SIMD scan for '\"v' prefix
  "block6(v23: i64, v24: i64):\n" ++
  "    v25 = iadd v13, v23\n" ++                     -- data_ptr + pos
  "    v26 = load.i8x16 v25\n" ++                    -- 16 bytes at pos
  "    v27 = load.i8x16 v25+1\n" ++                  -- 16 bytes at pos+1
  "    v28 = icmp eq v26, v15\n" ++                   -- compare with '\"'
  "    v29 = icmp eq v27, v17\n" ++                   -- compare with 'v'
  "    v30 = band v28, v29\n" ++                     -- '\"v' at lane?
  "    v31 = vhigh_bits.i32 v30\n" ++                -- extract bitmask
  "    v32 = icmp ne v31, v19\n" ++                  -- any match?
  "    v33 = iadd_imm v23, 16\n" ++                  -- next chunk
  "    brif v32, block7(v23, v24, v31), block5(v33, v24)\n" ++
  "\n" ++
  -- block7: extract first '\"v' position from mask
  "block7(v35: i64, v36: i64, v37: i32):\n" ++
  "    v38 = ctz v37\n" ++                           -- offset of first match
  "    v39 = uextend.i64 v38\n" ++
  "    v40 = iadd v35, v39\n" ++                     -- abs_pos = chunk_base + offset
  "    v41 = icmp sgt v40, v10\n" ++                 -- past end?
  "    brif v41, block10(v36), block8(v40, v36, v35, v37)\n" ++
  "\n" ++
  -- block8: verify remaining 8 bytes after '\"'
  "block8(v42: i64, v43: i64, v44: i64, v45: i32):\n" ++
  "    v46 = iadd v13, v42\n" ++                     -- data_ptr + abs_pos
  "    v47 = load.i64 v46+1\n" ++                    -- load bytes 1-8 after '\"'
  "    v48 = icmp eq v47, v18\n" ++                  -- == \"value\": <sp>?
  "    brif v48, block9(v42, v43), block30(v44, v43, v45)\n" ++
  "\n" ++
  -- block30: no full match, clear lowest bit, try next in chunk
  "block30(v49: i64, v50: i64, v51: i32):\n" ++
  "    v52 = iadd_imm v51, -1\n" ++                 -- mask - 1
  "    v53 = band v51, v52\n" ++                     -- clear lowest set bit
  "    v54 = icmp ne v53, v19\n" ++                  -- more bits?
  "    v55 = iadd_imm v49, 16\n" ++                  -- next chunk
  "    brif v54, block7(v49, v50, v53), block5(v55, v50)\n" ++
  "\n" ++
  -- block9: full match, parse digits at pos + 9
  "block9(v56: i64, v57: i64):\n" ++
  "    v58 = iadd_imm v56, 9\n" ++                  -- pos + 9 (skip needle)
  "    jump block20(v58, v57, v8)\n" ++
  "\n" ++
  -- block20: digit loop (digit_pos, total, accum)
  "block20(v59: i64, v60: i64, v61: i64):\n" ++
  "    v62 = iadd v13, v59\n" ++                     -- data_ptr + pos
  "    v63 = uload8.i64 v62\n" ++
  "    v64 = isub v63, v12\n" ++                     -- byte - '0'
  "    v65 = icmp ugt v64, v9\n" ++                  -- > 9? (unsigned range check)
  "    brif v65, block21(v59, v60, v61), block22(v59, v60, v61, v64)\n" ++
  "\n" ++
  -- block21: not a digit, total += accum, resume SIMD scan
  "block21(v66: i64, v67: i64, v68: i64):\n" ++
  "    v69 = iadd v67, v68\n" ++                     -- total += acc
  "    jump block5(v66, v69)\n" ++
  "\n" ++
  -- block22: digit, acc = acc * 10 + digit
  "block22(v70: i64, v71: i64, v72: i64, v73: i64):\n" ++
  "    v74 = imul v72, v11\n" ++                     -- acc * 10
  "    v75 = iadd v74, v73\n" ++                     -- + digit
  "    v76 = iadd_imm v70, 1\n" ++                   -- pos++
  "    jump block20(v76, v71, v75)\n" ++
  "\n" ++
  -- block10: itoa total to OUTPUT_BUF, then file_write
  "block10(v80: i64):\n" ++
  "    v81 = iconst.i64 1\n" ++
  "    v82 = iconst.i64 848\n" ++                    -- OUTPUT_BUF (0x0350)
  "    jump block11(v80, v81, v82)\n" ++
  "\n" ++
  -- find_divisor(total, div, wpos)
  "block11(v83: i64, v84: i64, v85: i64):\n" ++
  "    v86 = imul v84, v11\n" ++
  "    v87 = icmp ugt v86, v83\n" ++
  "    brif v87, block12(v83, v84, v85), block11(v83, v86, v85)\n" ++
  "\n" ++
  -- write_digit(val, div, wpos)
  "block12(v88: i64, v89: i64, v90: i64):\n" ++
  "    v91 = udiv v88, v89\n" ++
  "    v92 = iadd v91, v12\n" ++                     -- + '0'
  "    v93 = iadd v0, v90\n" ++
  "    istore8 v92, v93\n" ++
  "    v94 = imul v91, v89\n" ++
  "    v95 = isub v88, v94\n" ++
  "    v96 = udiv v89, v11\n" ++                     -- div / 10
  "    v97 = iadd_imm v90, 1\n" ++
  "    v98 = icmp eq v96, v8\n" ++                   -- div == 0?
  "    brif v98, block13(v97), block12(v95, v96, v97)\n" ++
  "\n" ++
  -- write_newline_and_null, then file_write
  "block13(v99: i64):\n" ++
  "    v100 = iadd v0, v99\n" ++
  "    v101 = iconst.i32 10\n" ++                    -- '\\n'
  "    istore8 v101, v100\n" ++
  "    v102 = iadd_imm v99, 1\n" ++
  "    v103 = iadd v0, v102\n" ++
  "    v104 = iconst.i32 0\n" ++
  "    istore8 v104, v103\n" ++
  -- Write result file
  "    v150 = iconst.i64 512\n" ++                   -- OUTPUT_PATH_OFF
  "    v151 = iconst.i64 848\n" ++                   -- OUTPUT_BUF
  "    v152 = iconst.i64 0\n" ++
  "    v153 = iconst.i64 0\n" ++
  "    v154 = call fn1(v0, v150, v151, v152, v153)\n" ++
  "    return\n" ++
  "}\n"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifJsonFn

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

end JsonBench

def main : IO Unit := do
  let json := toJsonPair JsonBench.buildConfig JsonBench.buildAlgorithm
  IO.println (Json.compress json)
