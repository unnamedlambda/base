import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace CsvBench

/-
  CSV benchmark algorithm — Cranelift JIT version.

  Payload (via execute data arg): "input_path\0output_path\0"

  Memory layout (shared memory):
    0x0000  RESERVED        (40 bytes, runtime-managed: ctx_ptr, data/out ptrs)
    0x0100  INPUT_PATH      (256 bytes, copied from payload by CLIF)
    0x0200  OUTPUT_PATH     (256 bytes, copied from payload by CLIF)
    0x0350  LEFT_VAL        (32 bytes, itoa result for FileWrite)
    0x2000  CSV_DATA        (variable, populated by FileRead)

  Single CLIF function:
    1. Copy input/output paths from payload into shared memory
    2. cl_file_read → gets bytes_read (used as end_pos)
    3. SIMD CSV parse (skip header, scan commas, parse salary digits)
    4. itoa total to LEFT_VAL
    5. cl_file_write result

  Uses 16-byte SIMD vectors (i8x16) for delimiter scanning.
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def LEFT_VAL        : Nat := 0x0350
def CSV_DATA        : Nat := 0x2000
def MAX_CSV_BYTES   : Nat := 512 * 1024 * 1024  -- 512MB max
def MEM_SIZE        : Nat := CSV_DATA + MAX_CSV_BYTES
def TIMEOUT_MS      : Nat := 300000

-- fn0: noop
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: CSV orchestrator
-- Copies paths from payload, reads file, parses CSV, writes result
def clifCsvFn : String :=
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
  -- Copy input path from payload to shared memory
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
  -- Copy output path from payload to shared memory
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
  -- Read CSV file into shared memory
  "block4:\n" ++
  "    v3 = iconst.i64 256\n" ++                     -- INPUT_PATH_OFF
  "    v4 = iconst.i64 8192\n" ++                    -- CSV_DATA
  "    v5 = iconst.i64 0\n" ++
  "    v700 = iconst.i64 0\n" ++
  "    v7 = call fn0(v0, v3, v4, v5, v700)\n" ++    -- bytes_read = end_pos
  -- Setup SIMD vectors and hoisted constants
  "    v6 = iconst.i64 0\n" ++
  "    v8 = iconst.i32 0\n" ++
  "    v300 = iconst.i8 10\n" ++
  "    v301 = splat.i8x16 v300\n" ++
  "    v302 = iconst.i8 44\n" ++
  "    v303 = splat.i8x16 v302\n" ++
  "    v401 = iconst.i64 10\n" ++
  "    v402 = iconst.i64 48\n" ++
  "    v403 = iconst.i32 5\n" ++
  "    v404 = iconst.i32 44\n" ++
  "    v405 = iconst.i32 10\n" ++
  "    v406 = iconst.i64 1\n" ++
  "    v407 = iconst.i64 848\n" ++                   -- LEFT_VAL offset
  "    jump block5(v6)\n" ++
  "\n" ++
  -- skip_header_16_check(pos)
  "block5(v10: i64):\n" ++
  "    v11 = iadd_imm v10, 16\n" ++
  "    v12 = icmp sle v11, v7\n" ++
  "    brif v12, block6(v10), block7(v10)\n" ++
  "\n" ++
  -- skip_header_16_body: SIMD newline detect
  "block6(v13: i64):\n" ++
  "    v14 = iadd v0, v13\n" ++
  "    v15 = load.i8x16 notrap v14+8192\n" ++       -- CSV_DATA
  "    v16 = icmp eq v15, v301\n" ++
  "    v17 = vhigh_bits.i32 v16\n" ++
  "    v18 = icmp ne v17, v8\n" ++
  "    v19 = iadd_imm v13, 16\n" ++
  "    brif v18, block8(v13, v17), block5(v19)\n" ++
  "\n" ++
  -- skip_header_1: byte fallback
  "block7(v20: i64):\n" ++
  "    v21 = iadd v0, v20\n" ++
  "    v22 = uload8.i32 notrap v21+8192\n" ++
  "    v23 = icmp eq v22, v405\n" ++                 -- '\n'
  "    v24 = iadd_imm v20, 1\n" ++
  "    brif v23, block9(v20), block7(v24)\n" ++
  "\n" ++
  -- skip_header_found: ctz
  "block8(v25: i64, v26: i32):\n" ++
  "    v27 = ctz v26\n" ++
  "    v28 = uextend.i64 v27\n" ++
  "    v29 = iadd v25, v28\n" ++
  "    jump block9(v29)\n" ++
  "\n" ++
  -- header_done: advance past newline
  "block9(v30: i64):\n" ++
  "    v31 = iadd_imm v30, 1\n" ++
  "    jump block10(v31, v6, v8)\n" ++
  "\n" ++
  -- comma_scan_16_check(pos, total, comma_count)
  "block10(v39: i64, v40: i64, v41: i32):\n" ++
  "    v42 = iadd_imm v39, 16\n" ++
  "    v43 = icmp sle v42, v7\n" ++
  "    brif v43, block11(v39, v40, v41), block12(v39, v40, v41)\n" ++
  "\n" ++
  -- comma_scan_16_body: SIMD comma detect
  "block11(v44: i64, v45: i64, v46: i32):\n" ++
  "    v48 = iadd v0, v44\n" ++
  "    v49 = load.i8x16 notrap v48+8192\n" ++
  "    v50 = icmp eq v49, v303\n" ++
  "    v54 = vhigh_bits.i32 v50\n" ++
  "    v55 = icmp ne v54, v8\n" ++
  "    v56 = iadd_imm v44, 16\n" ++
  "    brif v55, block13(v44, v45, v46, v54), block10(v56, v45, v46)\n" ++
  "\n" ++
  -- comma_scan_1: byte fallback
  "block12(v57: i64, v58: i64, v59: i32):\n" ++
  "    v61 = iadd v0, v57\n" ++
  "    v62 = uload8.i32 notrap v61+8192\n" ++
  "    v64 = icmp eq v62, v404\n" ++                 -- ','
  "    v65 = iadd_imm v57, 1\n" ++
  "    brif v64, block14(v57, v58, v59), block12(v65, v58, v59)\n" ++
  "\n" ++
  -- comma_found_16: ctz, count++, check ==5
  "block13(v66: i64, v67: i64, v68: i32, v69: i32):\n" ++
  "    v70 = ctz v69\n" ++
  "    v71 = uextend.i64 v70\n" ++
  "    v73 = iadd v66, v71\n" ++
  "    v74 = iadd_imm v68, 1\n" ++
  "    v76 = icmp eq v74, v403\n" ++                 -- 5 commas
  "    v77 = iadd_imm v73, 1\n" ++
  "    brif v76, block15(v77, v67, v6), block10(v77, v67, v74)\n" ++
  "\n" ++
  -- comma_found_1: count++, check ==5
  "block14(v78: i64, v79: i64, v80: i32):\n" ++
  "    v81 = iadd_imm v80, 1\n" ++
  "    v83 = icmp eq v81, v403\n" ++
  "    v84 = iadd_imm v78, 1\n" ++
  "    brif v83, block15(v84, v79, v6), block10(v84, v79, v81)\n" ++
  "\n" ++
  -- digit_loop(pos, total, accum)
  "block15(v88: i64, v89: i64, v90: i64):\n" ++
  "    v92 = iadd v0, v88\n" ++
  "    v93 = uload8.i64 notrap v92+8192\n" ++
  "    v95 = icmp eq v93, v401\n" ++                 -- '\n' (10)
  "    brif v95, block17(v88, v89, v90), block16(v88, v89, v90, v93)\n" ++
  "\n" ++
  -- is_digit: accum = accum*10 + (char - '0')
  "block16(v96: i64, v97: i64, v98: i64, v99: i64):\n" ++
  "    v101 = imul v98, v401\n" ++
  "    v103 = isub v99, v402\n" ++
  "    v104 = iadd v101, v103\n" ++
  "    v105 = iadd_imm v96, 1\n" ++
  "    jump block15(v105, v97, v104)\n" ++
  "\n" ++
  -- salary_done: total += accum, check end
  "block17(v106: i64, v107: i64, v108: i64):\n" ++
  "    v110 = iadd v107, v108\n" ++
  "    v111 = iadd_imm v106, 1\n" ++
  "    v112 = icmp sge v111, v7\n" ++
  "    brif v112, block18(v110), block10(v111, v110, v8)\n" ++
  "\n" ++
  -- itoa_start: find largest power of 10
  "block18(v113: i64):\n" ++
  "    jump block19(v113, v406)\n" ++
  "\n" ++
  -- find_divisor(total, div)
  "block19(v115: i64, v116: i64):\n" ++
  "    v118 = imul v116, v401\n" ++
  "    v119 = icmp ugt v118, v115\n" ++
  "    brif v119, block20(v115, v116), block19(v115, v118)\n" ++
  "\n" ++
  -- write_digits_start(total, div)
  "block20(v120: i64, v121: i64):\n" ++
  "    jump block21(v120, v121, v407)\n" ++
  "\n" ++
  -- write_digit(val, div, wpos)
  "block21(v123: i64, v124: i64, v125: i64):\n" ++
  "    v126 = udiv v123, v124\n" ++
  "    v128 = iadd v126, v402\n" ++
  "    v130 = iadd v0, v125\n" ++
  "    istore8 v128, v130\n" ++
  "    v131 = imul v126, v124\n" ++
  "    v132 = isub v123, v131\n" ++
  "    v134 = udiv v124, v401\n" ++
  "    v135 = iadd_imm v125, 1\n" ++
  "    v136 = icmp eq v134, v6\n" ++
  "    brif v136, block22(v135), block21(v132, v134, v135)\n" ++
  "\n" ++
  -- write_newline_and_null, then file_write
  "block22(v137: i64):\n" ++
  "    v139 = iadd v0, v137\n" ++
  "    istore8 v405, v139\n" ++                      -- '\n'
  "    v141 = iadd_imm v137, 1\n" ++
  "    v143 = iadd v0, v141\n" ++
  "    istore8 v8, v143\n" ++                        -- null
  -- Write result file
  "    v150 = iconst.i64 512\n" ++                   -- OUTPUT_PATH_OFF
  "    v151 = iconst.i64 848\n" ++                   -- LEFT_VAL
  "    v152 = iconst.i64 0\n" ++
  "    v153 = iconst.i64 0\n" ++
  "    v154 = call fn1(v0, v150, v151, v152, v153)\n" ++
  "    return\n" ++
  "}\n"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifCsvFn

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

end CsvBench

def main : IO Unit := do
  let json := toJsonPair CsvBench.buildConfig CsvBench.buildAlgorithm
  IO.println (Json.compress json)
