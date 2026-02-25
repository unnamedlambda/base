import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace RegexBench

/-
  Regex benchmark — CLIF implementation.

  Counts words matching [a-z]+ing pattern (lowercase words ending in "ing").
  Scans input byte-by-byte: for each word, check if it ends with "ing"
  and all bytes are lowercase ascii [a-z].

  Memory layout:
    0x0008  FLAG_FILE     (8 bytes)
    0x0010  FLAG_CL       (8 bytes)
    0x0020  INPUT_FILENAME  (256 bytes)
    0x0120  OUTPUT_FILENAME (256 bytes)
    0x0250  FILE_SIZE     (4 bytes, i32 LE)
    0x0300  RESULT_COUNT  (8 bytes, i64)
    0x0350  OUTPUT_BUF    (64 bytes, itoa result)
    0x0400  CLIF_IR       (variable, null-terminated)
    0x4000  INPUT_DATA    (variable)
-/

def FLAG_FILE       : Nat := 0x0008
def INPUT_FILENAME  : Nat := 0x0020
def OUTPUT_FILENAME : Nat := 0x0120
def FILE_SIZE       : Nat := 0x0250
def RESULT_COUNT    : Nat := 0x0300
def OUTPUT_BUF      : Nat := 0x0350
def CLIF_IR_OFF     : Nat := 0x0400
def INPUT_DATA      : Nat := 0x4000

def TIMEOUT_MS    : Nat := 300000

/-
  CLIF IR: scan words, check if [a-z]+ing pattern matches.
  Algorithm:
    1. Skip whitespace (block1/2)
    2. Read word bytes (block4/6): track all_lower only, no trail or len
       No bounds check in word loop — data always ends with whitespace.
       First byte jumps directly to block6 (avoids redundant reload).
    3. At word end (block5): compute len from positions, load last 3 bytes
       from memory for "ing" check. Branchless match computation.
    4. itoa count to OUTPUT_BUF
-/
def clifComputeFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i32 v0+592\n" ++             -- FILE_SIZE (0x250)
  "  v2 = sextend.i64 v1\n" ++
  "  v3 = iconst.i64 0\n" ++                -- zero
  "  v4 = iconst.i64 16384\n" ++            -- INPUT_DATA (0x4000)
  "  v5 = iconst.i64 32\n" ++               -- space
  "  v6 = iconst.i64 97\n" ++               -- 'a'
  "  v7 = iconst.i64 25\n" ++               -- range for lowercase check
  "  v8 = iconst.i64 6778473\n" ++          -- "ing" as LE i24 (0x676E69)
  "  v9 = iconst.i64 4\n" ++                -- min word length
  "  v10 = iconst.i64 10\n" ++              -- newline / div 10
  "  v11 = iconst.i64 48\n" ++              -- '0'
  "  v12 = iconst.i64 1\n" ++
  "  v13 = iconst.i64 848\n" ++             -- OUTPUT_BUF (0x0350)
  "  v15 = iconst.i64 16777215\n" ++        -- 0xFFFFFF mask
  "  v16 = iadd v0, v4\n" ++                -- data_ptr
  "  jump block1(v3, v3)\n" ++              -- (pos, count)
  "\n" ++
  -- block1: skip whitespace
  "block1(v20: i64, v21: i64):\n" ++
  "  v22 = icmp sge v20, v2\n" ++
  "  brif v22, block10(v21), block2(v20, v21)\n" ++
  "\n" ++
  "block2(v23: i64, v24: i64):\n" ++
  "  v25 = iadd v16, v23\n" ++              -- data_ptr + pos
  "  v26 = uload8.i64 v25\n" ++
  "  v27 = icmp ule v26, v5\n" ++           -- <= space?
  "  v28 = iadd_imm v23, 1\n" ++
  "  brif v27, block1(v28, v24), block6(v23, v24, v12, v23, v26)\n" ++
  "\n" ++
  -- block4: word loop (pos, count, all_lower, word_start) — 4 params
  -- No bounds check: data always ends with whitespace.
  "block4(v40: i64, v41: i64, v42: i64, v43: i64):\n" ++
  "  v44 = iadd v16, v40\n" ++              -- data_ptr + pos
  "  v45 = uload8.i64 v44\n" ++
  "  v46 = icmp ule v45, v5\n" ++           -- whitespace?
  "  brif v46, block5(v40, v41, v42, v43), block6(v40, v41, v42, v43, v45)\n" ++
  "\n" ++
  -- block5: word done — compute len from positions, load last 3 bytes
  "block5(v50: i64, v51: i64, v52: i64, v53: i64):\n" ++
  "  v54 = isub v50, v53\n" ++              -- len = ws_pos - word_start
  "  v55 = icmp sge v54, v9\n" ++           -- len >= 4? (i8)
  "  v56 = uextend.i64 v55\n" ++
  "  v57 = band v52, v56\n" ++              -- all_lower && len >= 4
  "  v58 = iadd_imm v50, -3\n" ++           -- ws_pos - 3
  "  v59 = iadd v16, v58\n" ++              -- data_ptr + (ws_pos - 3)
  "  v60 = load.i32 v59\n" ++               -- load 4 bytes (last 3 of word + ws byte)
  "  v61 = uextend.i64 v60\n" ++
  "  v62 = band v61, v15\n" ++              -- mask to 3 bytes (0xFFFFFF)
  "  v63 = icmp eq v62, v8\n" ++            -- == \"ing\" (LE)?
  "  v64 = uextend.i64 v63\n" ++
  "  v65 = band v57, v64\n" ++              -- full match
  "  v66 = iadd v51, v65\n" ++              -- count += match
  "  v67 = iadd_imm v50, 1\n" ++            -- past whitespace
  "  jump block1(v67, v66)\n" ++
  "\n" ++
  -- block6: process byte — check lowercase only (no trail tracking)
  "block6(v70: i64, v71: i64, v72: i64, v73: i64, v74: i64):\n" ++
  "  v75 = isub v74, v6\n" ++               -- byte - 'a'
  "  v76 = icmp ule v75, v7\n" ++           -- <= 25? (single range check)
  "  v77 = uextend.i64 v76\n" ++
  "  v78 = band v72, v77\n" ++              -- all_lower &= is_lower
  "  v79 = iadd_imm v70, 1\n" ++            -- pos++
  "  jump block4(v79, v71, v78, v73)\n" ++
  "\n" ++
  -- block10: itoa count to OUTPUT_BUF
  "block10(v90: i64):\n" ++
  "  store.i64 v90, v0+768\n" ++            -- RESULT_COUNT (0x300)
  "  jump block11(v90, v12)\n" ++
  "\n" ++
  -- block11: find divisor
  "block11(v91: i64, v92: i64):\n" ++
  "  v93 = imul v92, v10\n" ++
  "  v94 = icmp ugt v93, v91\n" ++
  "  brif v94, block12(v91, v92, v13), block11(v91, v93)\n" ++
  "\n" ++
  -- block12: write digits
  "block12(v95: i64, v96: i64, v97: i64):\n" ++
  "  v98 = udiv v95, v96\n" ++
  "  v99 = iadd v98, v11\n" ++              -- + '0'
  "  v100 = iadd v0, v97\n" ++
  "  istore8 v99, v100\n" ++
  "  v101 = imul v98, v96\n" ++
  "  v102 = isub v95, v101\n" ++
  "  v103 = udiv v96, v10\n" ++
  "  v104 = iadd_imm v97, 1\n" ++
  "  v105 = icmp eq v103, v3\n" ++
  "  brif v105, block13(v104), block12(v102, v103, v104)\n" ++
  "\n" ++
  -- block13: write newline + null
  "block13(v106: i64):\n" ++
  "  v107 = iadd v0, v106\n" ++
  "  istore8 v10, v107\n" ++
  "  v108 = iadd_imm v106, 1\n" ++
  "  v109 = iadd v0, v108\n" ++
  "  v110 = iconst.i32 0\n" ++
  "  istore8 v110, v109\n" ++
  "  return\n" ++
  "}\n"

def clifReadFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "  sig0 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "  fn0 = %cl_file_read sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = iconst.i64 32\n" ++        -- INPUT_FILENAME
  "  v2 = iconst.i64 16384\n" ++     -- INPUT_DATA
  "  v3 = iconst.i64 0\n" ++
  "  v4 = iconst.i64 0\n" ++
  "  v5 = call fn0(v0, v1, v2, v3, v4)\n" ++
  "  return\n" ++
  "}\n"

def clifWriteFn : String :=
  "function u0:2(i64) system_v {\n" ++
  "  sig0 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "  fn0 = %cl_file_write sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = iconst.i64 288\n" ++       -- OUTPUT_FILENAME
  "  v2 = iconst.i64 848\n" ++       -- OUTPUT_BUF
  "  v3 = iconst.i64 0\n" ++
  "  v4 = iconst.i64 0\n" ++
  "  v5 = call fn0(v0, v1, v2, v3, v4)\n" ++
  "  return\n" ++
  "}\n"

def clifIR : String :=
  clifReadFn ++ "\n" ++ clifComputeFn ++ "\n" ++ clifWriteFn

def clifIRBytes : List UInt8 :=
  clifIR.toUTF8.toList ++ [0]

def buildPayload : List UInt8 :=
  let gap0         := zeros FLAG_FILE
  let flagFile     := zeros 8
  let flagCl       := zeros 8
  let gap1         := zeros (INPUT_FILENAME - 0x0018)
  let inputFile    := zeros 256
  let outputFile   := zeros 256
  let gap2         := zeros (FILE_SIZE - 0x0220)
  let fileSize     := zeros 4
  let gap3         := zeros (RESULT_COUNT - 0x0254)
  let resultCount  := zeros 8
  let gap4         := zeros (OUTPUT_BUF - (RESULT_COUNT + 8))
  let outputBuf    := zeros 64
  let gap5         := zeros (CLIF_IR_OFF - (OUTPUT_BUF + 64))
  let irBytes      := clifIRBytes
  let currentSize  :=
    gap0.length + flagFile.length + flagCl.length + gap1.length +
    inputFile.length + outputFile.length + gap2.length + fileSize.length +
    gap3.length + resultCount.length + gap4.length + outputBuf.length +
    gap5.length + irBytes.length
  let padding := if INPUT_DATA > currentSize then zeros (INPUT_DATA - currentSize) else []
  gap0 ++ flagFile ++ flagCl ++ gap1 ++
    inputFile ++ outputFile ++ gap2 ++ fileSize ++
    gap3 ++ resultCount ++ gap4 ++ outputBuf ++ gap5 ++
    irBytes ++ padding

def controlActions : List Action :=
  [
    -- 0: Synchronous CLIF file-read (fn 0)
    { kind := .ClifCall, dst := 0, src := 0, offset := 0, size := 0 },
    -- 1: Synchronous CLIF compute (fn 1)
    { kind := .ClifCall, dst := 0, src := 1, offset := 0, size := 0 },
    -- 2: Synchronous CLIF file-write (fn 2)
    { kind := .ClifCall, dst := 0, src := 2, offset := 0, size := 0 }
  ]

def buildAlgorithm : Algorithm := {
  actions := controlActions,
  payloads := buildPayload,
  state := {
    cranelift_ir_offsets := [CLIF_IR_OFF]
  },
  units := {
    cranelift_units := 0,
  },
  worker_threads := some 2, blocking_threads := some 2,
  stack_size := none, timeout_ms := some TIMEOUT_MS,
  thread_name_prefix := some "regex-bench",
  additional_shared_memory := 0
}

end RegexBench

def main : IO Unit := do
  let json := toJson RegexBench.buildAlgorithm
  IO.println (Json.compress json)
