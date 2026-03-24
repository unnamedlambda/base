import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace WordCountBench

/-
  Word frequency counting benchmark — Cranelift JIT version.

  Payload (via execute data arg): "input_path\0output_path\0"

  Memory layout (shared memory):
    0x0000  RESERVED        (40 bytes, runtime-managed: ctx_ptr, data/out ptrs)
    0x0038  CURRENT_KEY     (8 bytes, scratch for word u64 / get_entry key)
    0x0040  NEW_VALUE       (8 bytes, scratch for count u64)
    0x0100  INPUT_PATH      (256 bytes, copied from payload by CLIF)
    0x0200  OUTPUT_PATH     (256 bytes, copied from payload by CLIF)
    0x0350  RESULT_SLOT     (8 bytes, lookup/get_entry value)
    0x4000  OUTPUT_BUF      (65536 bytes, formatted output)
    0x14000 INPUT_DATA      (variable, populated by FileRead)

  Single CLIF function:
    1. Copy input/output paths from payload into shared memory
    2. cl_file_read → gets bytes_read (used as file_size)
    3. Parse words, ht_increment for each
    4. Format phase: ht_count + ht_get_entry → word\tcount\n
    5. cl_file_write result

  Uses HT primitives: ht_create, ht_increment, ht_count, ht_get_entry.
  HT context pointer at offset 0x00 is written by runtime (context_offset := 0).
-/

def CURRENT_KEY     : Nat := 0x0038
def NEW_VALUE       : Nat := 0x0040
def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def RESULT_SLOT     : Nat := 0x0350
def OUTPUT_BUF      : Nat := 0x4000
def INPUT_DATA      : Nat := 0x14000
def MAX_TEXT_BYTES   : Nat := 512 * 1024 * 1024
def MEM_SIZE        : Nat := INPUT_DATA + MAX_TEXT_BYTES
def TIMEOUT_MS      : Nat := 300000

-- fn0: noop
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: Word count orchestrator
def clifWcFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "    sig1 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "    sig2 = (i64) -> i32 system_v\n" ++
  "    sig3 = (i64, i64, i32, i64) -> i64 system_v\n" ++
  "    sig4 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "\n" ++
  "    fn0 = %cl_file_read sig0\n" ++
  "    fn1 = %cl_file_write sig1\n" ++
  "    fn2 = colocated %ht_create sig2\n" ++
  "    fn3 = colocated %ht_increment sig3\n" ++
  "    fn4 = colocated %ht_count sig2\n" ++
  "    fn5 = colocated %ht_get_entry sig4\n" ++
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
  -- Read text file into shared memory, then setup for parse
  "block4:\n" ++
  "    v3 = iconst.i64 256\n" ++                     -- INPUT_PATH_OFF
  "    v4 = iconst.i64 81920\n" ++                   -- INPUT_DATA (0x14000)
  "    v5 = iconst.i64 0\n" ++
  "    v700 = iconst.i64 0\n" ++
  "    v7 = call fn0(v0, v3, v4, v5, v700)\n" ++    -- bytes_read = file_size
  -- Create HT
  "    v2 = load.i64 v0\n" ++                        -- HT context pointer (offset 0)
  "    v800 = call fn2(v2)\n" ++                     -- ht_create(ctx)
  -- Setup constants
  "    v8 = iconst.i64 0\n" ++                       -- zero
  "    v9 = iconst.i64 32\n" ++                      -- space
  "    v10 = iconst.i64 8\n" ++                      -- max word bytes / shift
  "    v11 = iconst.i64 1\n" ++
  "    v12 = iconst.i64 56\n" ++                     -- CURRENT_KEY (0x38)
  "    v13 = iconst.i32 8\n" ++                      -- key_len
  "    v14 = iconst.i64 0xFF\n" ++                   -- byte mask
  "    v15 = iconst.i64 9\n" ++                      -- tab
  "    v16 = iconst.i64 10\n" ++                     -- newline / divisor 10
  "    v17 = iconst.i64 48\n" ++                     -- '0'
  "    v18 = iconst.i64 16384\n" ++                  -- OUTPUT_BUF (0x4000)
  "    v19 = iconst.i64 848\n" ++                    -- RESULT_SLOT (0x0350)
  "    jump block5(v8, v2)\n" ++
  "\n" ++
  -- === PARSE PHASE ===
  -- block5: skip whitespace (pos, ctx)
  "block5(v20: i64, v21: i64):\n" ++
  "    v22 = icmp sge v20, v7\n" ++                  -- pos >= file_size?
  "    brif v22, block20(v21), block6(v20, v21)\n" ++
  "\n" ++
  -- block6: check byte
  "block6(v23: i64, v24: i64):\n" ++
  "    v25 = iadd v0, v4\n" ++
  "    v26 = iadd v25, v23\n" ++
  "    v27 = uload8.i64 v26\n" ++
  "    v28 = icmp ule v27, v9\n" ++                  -- <= space?
  "    v29 = iadd_imm v23, 1\n" ++
  "    brif v28, block5(v29, v24), block7(v23, v24, v8, v8)\n" ++
  "\n" ++
  -- block7: read word (pos, ctx, word_accum, byte_idx)
  "block7(v30: i64, v31: i64, v32: i64, v33: i64):\n" ++
  "    v34 = icmp sge v30, v7\n" ++                  -- pos >= file_size?
  "    brif v34, block9(v30, v31, v32), block8(v30, v31, v32, v33)\n" ++
  "\n" ++
  -- block8: read byte
  "block8(v35: i64, v36: i64, v37: i64, v38: i64):\n" ++
  "    v39 = iadd v0, v4\n" ++
  "    v40 = iadd v39, v35\n" ++
  "    v41 = uload8.i64 v40\n" ++
  "    v42 = icmp ule v41, v9\n" ++                  -- <= space?
  "    brif v42, block9(v35, v36, v37), block10(v35, v36, v37, v38, v41)\n" ++
  "\n" ++
  -- block9: word done → ht_increment
  "block9(v43: i64, v44: i64, v45: i64):\n" ++
  "    v46 = iadd v0, v12\n" ++                      -- CURRENT_KEY addr
  "    store.i64 v45, v46\n" ++
  "    v47 = call fn3(v44, v46, v13, v11)\n" ++      -- ht_increment(ctx, key, 8, 1)
  "    jump block5(v43, v44)\n" ++
  "\n" ++
  -- block10: accumulate byte
  "block10(v50: i64, v51: i64, v52: i64, v53: i64, v54: i64):\n" ++
  "    v55 = imul v53, v10\n" ++                     -- byte_idx * 8
  "    v56 = ishl v54, v55\n" ++
  "    v57 = bor v52, v56\n" ++
  "    v58 = iadd_imm v50, 1\n" ++
  "    v59 = iadd_imm v53, 1\n" ++
  "    v60 = icmp sge v59, v10\n" ++                 -- >= 8 bytes?
  "    brif v60, block9(v58, v51, v57), block7(v58, v51, v57, v59)\n" ++
  "\n" ++
  -- === FORMAT PHASE ===
  -- block20: get entry count from HT
  "block20(v80: i64):\n" ++
  "    v81 = call fn4(v80)\n" ++                     -- ht_count(ctx)
  "    v82 = uextend.i64 v81\n" ++
  "    jump block21(v8, v18, v80, v82)\n" ++
  "\n" ++
  -- block21: format loop (idx, out_pos, ctx, total)
  "block21(v83: i64, v84: i64, v85: i64, v86: i64):\n" ++
  "    v87 = icmp sge v83, v86\n" ++                 -- idx >= total?
  "    brif v87, block30(v84), block22(v83, v84, v85, v86)\n" ++
  "\n" ++
  -- block22: get entry, read key+val
  "block22(v88: i64, v89: i64, v90: i64, v91: i64):\n" ++
  "    v92 = ireduce.i32 v88\n" ++                   -- index as i32
  "    v93 = iadd v0, v12\n" ++                      -- key_out (CURRENT_KEY)
  "    v94 = iadd v0, v19\n" ++                      -- val_out (RESULT_SLOT)
  "    v95 = call fn5(v90, v92, v93, v94)\n" ++      -- ht_get_entry
  "    v96 = load.i64 v93\n" ++                      -- word u64
  "    v97 = load.i64 v94\n" ++                      -- count u64
  "    jump block23(v88, v89, v90, v91, v96, v97, v8)\n" ++
  "\n" ++
  -- block23: unpack word bytes (idx, out_pos, ctx, total, word, count, byte_idx)
  "block23(v100: i64, v101: i64, v102: i64, v103: i64, v104: i64, v105: i64, v106: i64):\n" ++
  "    v107 = icmp sge v106, v10\n" ++               -- >= 8?
  "    brif v107, block25(v100, v101, v102, v103, v105), block24(v100, v101, v102, v103, v104, v105, v106)\n" ++
  "\n" ++
  -- block24: extract byte
  "block24(v108: i64, v109: i64, v110: i64, v111: i64, v112: i64, v113: i64, v114: i64):\n" ++
  "    v115 = imul v114, v10\n" ++                   -- byte_idx * 8
  "    v116 = ushr v112, v115\n" ++
  "    v117 = band v116, v14\n" ++                   -- & 0xFF
  "    v118 = icmp eq v117, v8\n" ++                 -- == 0? (end of word)
  "    brif v118, block25(v108, v109, v110, v111, v113), block26(v108, v109, v110, v111, v112, v113, v114, v117)\n" ++
  "\n" ++
  -- block25: write tab + start itoa (idx, out_pos, ctx, total, count)
  "block25(v120: i64, v121: i64, v122: i64, v123: i64, v124: i64):\n" ++
  "    v125 = iadd v0, v121\n" ++
  "    istore8 v15, v125\n" ++                       -- tab
  "    v126 = iadd_imm v121, 1\n" ++
  "    jump block27(v120, v126, v122, v123, v124, v11)\n" ++
  "\n" ++
  -- block26: write byte
  "block26(v127: i64, v128: i64, v129: i64, v130: i64, v131: i64, v132: i64, v133: i64, v134: i64):\n" ++
  "    v135 = iadd v0, v128\n" ++
  "    istore8 v134, v135\n" ++
  "    v136 = iadd_imm v128, 1\n" ++
  "    v137 = iadd_imm v133, 1\n" ++
  "    jump block23(v127, v136, v129, v130, v131, v132, v137)\n" ++
  "\n" ++
  -- block27: find divisor (idx, out_pos, ctx, total, count, divisor)
  "block27(v138: i64, v139: i64, v140: i64, v141: i64, v142: i64, v143: i64):\n" ++
  "    v144 = imul v143, v16\n" ++                   -- divisor * 10
  "    v145 = icmp ugt v144, v142\n" ++              -- > count?
  "    brif v145, block28(v138, v139, v140, v141, v142, v143), block27(v138, v139, v140, v141, v142, v144)\n" ++
  "\n" ++
  -- block28: write digit (idx, out_pos, ctx, total, remainder, divisor)
  "block28(v146: i64, v147: i64, v148: i64, v149: i64, v150: i64, v151: i64):\n" ++
  "    v152 = udiv v150, v151\n" ++
  "    v153 = iadd v152, v17\n" ++                   -- + '0'
  "    v154 = iadd v0, v147\n" ++
  "    istore8 v153, v154\n" ++
  "    v155 = imul v152, v151\n" ++
  "    v156 = isub v150, v155\n" ++                  -- remainder
  "    v157 = udiv v151, v16\n" ++                   -- divisor / 10
  "    v158 = iadd_imm v147, 1\n" ++
  "    v159 = icmp eq v157, v8\n" ++                 -- done?
  "    brif v159, block29(v146, v158, v148, v149), block28(v146, v158, v148, v149, v156, v157)\n" ++
  "\n" ++
  -- block29: write newline, next entry
  "block29(v160: i64, v161: i64, v162: i64, v163: i64):\n" ++
  "    v164 = iadd v0, v161\n" ++
  "    istore8 v16, v164\n" ++                       -- newline
  "    v165 = iadd_imm v161, 1\n" ++
  "    v166 = iadd_imm v160, 1\n" ++
  "    jump block21(v166, v165, v162, v163)\n" ++
  "\n" ++
  -- block30: null-terminate, then file_write
  "block30(v167: i64):\n" ++
  "    v168 = iconst.i32 0\n" ++
  "    v169 = iadd v0, v167\n" ++
  "    istore8 v168, v169\n" ++
  -- Write result file
  "    v170 = iconst.i64 512\n" ++                   -- OUTPUT_PATH_OFF
  "    v171 = iconst.i64 16384\n" ++                 -- OUTPUT_BUF
  "    v172 = iconst.i64 0\n" ++
  "    v173 = iconst.i64 0\n" ++
  "    v174 = call fn1(v0, v170, v171, v172, v173)\n" ++
  "    return\n" ++
  "}\n"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifWcFn

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

end WordCountBench

def main : IO Unit := do
  let json := toJsonPair WordCountBench.buildConfig WordCountBench.buildAlgorithm
  IO.println (Json.compress json)
