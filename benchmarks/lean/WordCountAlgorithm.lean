import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace WordCountBench

/-
  Word frequency counting benchmark — INLINE HT version (v2).

  Single CLIF function calls Rust hash table primitives directly via
  JITBuilder::symbol(). No unique word tracking in CLIF — uses
  ht_count + ht_get_entry to iterate entries during format phase.

  Memory layout:
    0x0000  HT_CTX_PTR     (8 bytes, written by CraneliftUnit at startup)
    0x0008  FLAG_FILE      (8 bytes)
    0x0010  FLAG_CL        (8 bytes)
    0x0038  CURRENT_KEY    (8 bytes, scratch for word u64 / get_entry key)
    0x0040  NEW_VALUE      (8 bytes, scratch for count u64)
    0x0050  INPUT_FILENAME (256 bytes)
    0x0150  OUTPUT_FILENAME(256 bytes)
    0x0250  FILE_SIZE      (4 bytes, i32 LE)
    0x0300  CLIF_IR        (variable, null-terminated)
    0x3400  RESULT_SLOT    (8 bytes, lookup/get_entry value)
    0x4000  OUTPUT_BUF     (65536 bytes)
    0x14000 INPUT_DATA     (variable)
-/

def FLAG_FILE       : Nat := 0x0008
def FLAG_CL         : Nat := 0x0010
def CURRENT_KEY     : Nat := 0x0038
def NEW_VALUE       : Nat := 0x0040
def INPUT_FILENAME  : Nat := 0x0050
def OUTPUT_FILENAME : Nat := 0x0150
def FILE_SIZE       : Nat := 0x0250
def CLIF_IR_OFF     : Nat := 0x0300
def RESULT_SLOT     : Nat := 0x3400
def OUTPUT_BUF      : Nat := 0x4000
def INPUT_DATA      : Nat := 0x14000

def FILE_UNIT : UInt32 := 2
def CL_UNIT   : UInt32 := 9

def FILE_BUF_SIZE : Nat := 0x4000000
def TIMEOUT_MS    : Nat := 300000

-- HT primitives:
--   fn0 = ht_create(ctx) -> handle          sig0 = (i64) -> i32
--   fn1 = ht_increment(ctx, key, key_len, addend) -> new_val   sig1 = (i64, i64, i32, i64) -> i64
--   fn2 = ht_count(ctx) -> count            sig0 (reused)
--   fn3 = ht_get_entry(ctx, idx, key_out, val_out) -> key_len   sig2 = (i64, i32, i64, i64) -> i32
--
-- Parse phase: for each word, single ht_increment call.
-- Format phase: ht_count + loop ht_get_entry → format word\tcount\n.
def clifIR : String :=
  "function %wordcount(i64) system_v {\n" ++
  "  sig0 = (i64) -> i32 system_v\n" ++
  "  sig1 = (i64, i64, i32, i64) -> i64 system_v\n" ++
  "  sig2 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "  fn0 = colocated %ht_create sig0\n" ++
  "  fn1 = colocated %ht_increment sig1\n" ++
  "  fn2 = colocated %ht_count sig0\n" ++
  "  fn3 = colocated %ht_get_entry sig2\n" ++
  "\n" ++
  -- block0: setup constants, create HT
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 v0\n" ++               -- HT context pointer
  "  v2 = call fn0(v1)\n" ++               -- ht_create(ctx)
  "  v3 = load.i32 v0+592\n" ++            -- FILE_SIZE (0x250)
  "  v4 = sextend.i64 v3\n" ++
  "  v5 = iconst.i64 0\n" ++
  "  v6 = iconst.i64 32\n" ++              -- space
  "  v7 = iconst.i64 8\n" ++               -- max word bytes / shift
  "  v8 = iconst.i64 81920\n" ++           -- INPUT_DATA (0x14000)
  "  v9 = iconst.i64 1\n" ++
  "  v10 = iconst.i64 56\n" ++             -- CURRENT_KEY (0x38)
  "  v11 = iconst.i64 13312\n" ++          -- RESULT_SLOT (0x3400)
  "  v12 = iconst.i32 8\n" ++              -- key_len
  "  v13 = iconst.i64 0xFF\n" ++           -- byte mask
  "  v14 = iconst.i64 9\n" ++              -- tab
  "  v15 = iconst.i64 10\n" ++             -- newline / divisor 10
  "  v16 = iconst.i64 48\n" ++             -- '0'
  "  v17 = iconst.i64 16384\n" ++          -- OUTPUT_BUF (0x4000)
  "  jump block1(v5, v1)\n" ++
  "\n" ++
  -- === PARSE PHASE ===
  -- block1: skip whitespace (pos, ctx)
  "block1(v20: i64, v21: i64):\n" ++
  "  v22 = icmp sge v20, v4\n" ++          -- pos >= file_size?
  "  brif v22, block20(v21), block2(v20, v21)\n" ++
  "\n" ++
  -- block2: check byte
  "block2(v23: i64, v24: i64):\n" ++
  "  v25 = iadd v0, v8\n" ++
  "  v26 = iadd v25, v23\n" ++
  "  v27 = uload8.i64 v26\n" ++
  "  v28 = icmp ule v27, v6\n" ++          -- <= space?
  "  v29 = iadd_imm v23, 1\n" ++
  "  brif v28, block1(v29, v24), block3(v23, v24, v5, v5)\n" ++
  "\n" ++
  -- block3: read word (pos, ctx, word_accum, byte_idx)
  "block3(v30: i64, v31: i64, v32: i64, v33: i64):\n" ++
  "  v34 = icmp sge v30, v4\n" ++
  "  brif v34, block5(v30, v31, v32), block4(v30, v31, v32, v33)\n" ++
  "\n" ++
  -- block4: read byte
  "block4(v35: i64, v36: i64, v37: i64, v38: i64):\n" ++
  "  v39 = iadd v0, v8\n" ++
  "  v40 = iadd v39, v35\n" ++
  "  v41 = uload8.i64 v40\n" ++
  "  v42 = icmp ule v41, v6\n" ++
  "  brif v42, block5(v35, v36, v37), block6(v35, v36, v37, v38, v41)\n" ++
  "\n" ++
  -- block5: word done → single ht_increment call
  "block5(v43: i64, v44: i64, v45: i64):\n" ++
  "  v46 = iadd v0, v10\n" ++              -- CURRENT_KEY addr
  "  store.i64 v45, v46\n" ++
  "  v47 = call fn1(v44, v46, v12, v9)\n" ++ -- ht_increment(ctx, key, 8, 1)
  "  jump block1(v43, v44)\n" ++
  "\n" ++
  -- block6: accumulate byte
  "block6(v50: i64, v51: i64, v52: i64, v53: i64, v54: i64):\n" ++
  "  v55 = imul v53, v7\n" ++
  "  v56 = ishl v54, v55\n" ++
  "  v57 = bor v52, v56\n" ++
  "  v58 = iadd_imm v50, 1\n" ++
  "  v59 = iadd_imm v53, 1\n" ++
  "  v60 = icmp sge v59, v7\n" ++          -- >= 8 bytes?
  "  brif v60, block5(v58, v51, v57), block3(v58, v51, v57, v59)\n" ++
  "\n" ++
  -- === FORMAT PHASE ===
  -- block20: get entry count from HT
  "block20(v80: i64):\n" ++
  "  v81 = call fn2(v80)\n" ++             -- ht_count(ctx)
  "  v82 = uextend.i64 v81\n" ++
  "  jump block21(v5, v17, v80, v82)\n" ++
  "\n" ++
  -- block21: format loop (idx, out_pos, ctx, total)
  "block21(v83: i64, v84: i64, v85: i64, v86: i64):\n" ++
  "  v87 = icmp sge v83, v86\n" ++         -- idx >= total?
  "  brif v87, block30(v84), block22(v83, v84, v85, v86)\n" ++
  "\n" ++
  -- block22: get entry, read key+val
  "block22(v88: i64, v89: i64, v90: i64, v91: i64):\n" ++
  "  v92 = ireduce.i32 v88\n" ++           -- index as i32
  "  v93 = iadd v0, v10\n" ++              -- key_out (CURRENT_KEY)
  "  v94 = iadd v0, v11\n" ++              -- val_out (RESULT_SLOT)
  "  v95 = call fn3(v90, v92, v93, v94)\n" ++ -- ht_get_entry
  "  v96 = load.i64 v93\n" ++              -- word u64
  "  v97 = load.i64 v94\n" ++              -- count u64
  "  jump block23(v88, v89, v90, v91, v96, v97, v5)\n" ++
  "\n" ++
  -- block23: unpack word bytes (idx, out_pos, ctx, total, word, count, byte_idx)
  "block23(v100: i64, v101: i64, v102: i64, v103: i64, v104: i64, v105: i64, v106: i64):\n" ++
  "  v107 = icmp sge v106, v7\n" ++        -- >= 8?
  "  brif v107, block25(v100, v101, v102, v103, v105), block24(v100, v101, v102, v103, v104, v105, v106)\n" ++
  "\n" ++
  -- block24: extract byte
  "block24(v108: i64, v109: i64, v110: i64, v111: i64, v112: i64, v113: i64, v114: i64):\n" ++
  "  v115 = imul v114, v7\n" ++            -- byte_idx * 8
  "  v116 = ushr v112, v115\n" ++
  "  v117 = band v116, v13\n" ++           -- & 0xFF
  "  v118 = icmp eq v117, v5\n" ++         -- == 0? (end of word)
  "  brif v118, block25(v108, v109, v110, v111, v113), block26(v108, v109, v110, v111, v112, v113, v114, v117)\n" ++
  "\n" ++
  -- block25: write tab + start itoa (idx, out_pos, ctx, total, count)
  "block25(v120: i64, v121: i64, v122: i64, v123: i64, v124: i64):\n" ++
  "  v125 = iadd v0, v121\n" ++
  "  istore8 v14, v125\n" ++               -- tab
  "  v126 = iadd_imm v121, 1\n" ++
  "  jump block27(v120, v126, v122, v123, v124, v9)\n" ++
  "\n" ++
  -- block26: write byte
  "block26(v127: i64, v128: i64, v129: i64, v130: i64, v131: i64, v132: i64, v133: i64, v134: i64):\n" ++
  "  v135 = iadd v0, v128\n" ++
  "  istore8 v134, v135\n" ++
  "  v136 = iadd_imm v128, 1\n" ++
  "  v137 = iadd_imm v133, 1\n" ++
  "  jump block23(v127, v136, v129, v130, v131, v132, v137)\n" ++
  "\n" ++
  -- block27: find divisor (idx, out_pos, ctx, total, count, divisor)
  "block27(v138: i64, v139: i64, v140: i64, v141: i64, v142: i64, v143: i64):\n" ++
  "  v144 = imul v143, v15\n" ++           -- divisor * 10
  "  v145 = icmp ugt v144, v142\n" ++      -- > count?
  "  brif v145, block28(v138, v139, v140, v141, v142, v143), block27(v138, v139, v140, v141, v142, v144)\n" ++
  "\n" ++
  -- block28: write digit (idx, out_pos, ctx, total, remainder, divisor)
  "block28(v146: i64, v147: i64, v148: i64, v149: i64, v150: i64, v151: i64):\n" ++
  "  v152 = udiv v150, v151\n" ++
  "  v153 = iadd v152, v16\n" ++           -- + '0'
  "  v154 = iadd v0, v147\n" ++
  "  istore8 v153, v154\n" ++
  "  v155 = imul v152, v151\n" ++
  "  v156 = isub v150, v155\n" ++          -- remainder
  "  v157 = udiv v151, v15\n" ++           -- divisor / 10
  "  v158 = iadd_imm v147, 1\n" ++
  "  v159 = icmp eq v157, v5\n" ++         -- done?
  "  brif v159, block29(v146, v158, v148, v149), block28(v146, v158, v148, v149, v156, v157)\n" ++
  "\n" ++
  -- block29: write newline, next entry
  "block29(v160: i64, v161: i64, v162: i64, v163: i64):\n" ++
  "  v164 = iadd v0, v161\n" ++
  "  istore8 v15, v164\n" ++               -- newline
  "  v165 = iadd_imm v161, 1\n" ++
  "  v166 = iadd_imm v160, 1\n" ++
  "  jump block21(v166, v165, v162, v163)\n" ++
  "\n" ++
  -- block30: null-terminate, return
  "block30(v167: i64):\n" ++
  "  v168 = iconst.i32 0\n" ++
  "  v169 = iadd v0, v167\n" ++
  "  istore8 v168, v169\n" ++
  "  return\n" ++
  "}\n"

def clifIRBytes : List UInt8 :=
  clifIR.toUTF8.toList ++ [0]

def buildPayload : List UInt8 :=
  let htCtxPtr     := zeros 8                                -- 0x0000..0x0007
  let flagFile     := zeros 8                                -- 0x0008..0x000F
  let flagCl       := zeros 8                                -- 0x0010..0x0017
  let gap0         := zeros (CURRENT_KEY - 0x0018)           -- 0x0018..0x0037
  let currentKey   := zeros 8                                -- 0x0038..0x003F
  let newValue     := zeros 8                                -- 0x0040..0x0047
  let gap1         := zeros (INPUT_FILENAME - 0x0048)        -- 0x0048..0x004F
  let inputFile    := zeros 256                              -- 0x0050..0x014F
  let outputFile   := zeros 256                              -- 0x0150..0x024F
  let fileSize     := zeros 4                                -- 0x0250..0x0253
  let gap2         := zeros (CLIF_IR_OFF - 0x0254)           -- 0x0254..0x02FF
  let irBytes      := clifIRBytes                            -- 0x0300..
  let currentSize  :=
    htCtxPtr.length + flagFile.length + flagCl.length + gap0.length +
    currentKey.length + newValue.length + gap1.length +
    inputFile.length + outputFile.length + fileSize.length +
    gap2.length + irBytes.length
  let padToResult  := if RESULT_SLOT > currentSize then zeros (RESULT_SLOT - currentSize) else []
  let resultSlot   := zeros 8                                -- 0x3400..0x3407
  let padToOutput  := zeros (OUTPUT_BUF - (RESULT_SLOT + 8))
  let outputBuf    := zeros 65536                            -- 0x4000..0x13FFF
  let padToInput   := zeros (INPUT_DATA - (OUTPUT_BUF + 65536))
  htCtxPtr ++ flagFile ++ flagCl ++ gap0 ++
    currentKey ++ newValue ++ gap1 ++
    inputFile ++ outputFile ++ fileSize ++
    gap2 ++ irBytes ++ padToResult ++ resultSlot ++
    padToOutput ++ outputBuf ++ padToInput

-- Worker action base index (after 6 control actions)
def wBase : UInt32 := 6

def workerActions : List Action := [
  -- W+0: FileRead
  { kind := .FileRead,
    src := UInt32.ofNat INPUT_FILENAME,
    dst := UInt32.ofNat INPUT_DATA,
    offset := 0, size := 0 },
  -- W+1: CL fn 0 (wordcount — parse + format)
  { kind := .Fence, dst := 0, src := 0, offset := 0, size := 0 },
  -- W+2: FileWrite
  { kind := .FileWrite,
    src := UInt32.ofNat OUTPUT_BUF,
    dst := UInt32.ofNat OUTPUT_FILENAME,
    offset := 0, size := 0 }
]

def controlActions : List Action := [
  -- 0: Dispatch FileRead
  { kind := .AsyncDispatch,
    dst := FILE_UNIT,
    src := wBase,
    offset := UInt32.ofNat FLAG_FILE,
    size := 1 },
  -- 1: Wait FileRead
  { kind := .Wait,
    dst := UInt32.ofNat FLAG_FILE,
    src := 0, offset := 0, size := 0 },
  -- 2: Dispatch CL
  { kind := .AsyncDispatch,
    dst := CL_UNIT,
    src := wBase + 1,
    offset := UInt32.ofNat FLAG_CL,
    size := 1 },
  -- 3: Wait CL
  { kind := .Wait,
    dst := UInt32.ofNat FLAG_CL,
    src := 0, offset := 0, size := 0 },
  -- 4: Dispatch FileWrite
  { kind := .AsyncDispatch,
    dst := FILE_UNIT,
    src := wBase + 2,
    offset := UInt32.ofNat FLAG_FILE,
    size := 1 },
  -- 5: Wait FileWrite
  { kind := .Wait,
    dst := UInt32.ofNat FLAG_FILE,
    src := 0, offset := 0, size := 0 }
]

def buildAlgorithm : Algorithm := {
  actions := controlActions ++ workerActions,
  payloads := buildPayload,
  state := {
    file_buffer_size := FILE_BUF_SIZE,
    cranelift_ir_offsets := [CLIF_IR_OFF]
  },
  units := {
    file_units := 1,
    memory_units := 0,
    cranelift_units := 1,
  },
  memory_assignments := [],
  file_assignments := [],
  cranelift_assignments := [],
  worker_threads := some 2,
  blocking_threads := some 2,
  stack_size := none,
  timeout_ms := some TIMEOUT_MS,
  thread_name_prefix := some "wordcount-bench"
}

end WordCountBench

def main : IO Unit := do
  let json := toJson WordCountBench.buildAlgorithm
  IO.println (Json.compress json)
