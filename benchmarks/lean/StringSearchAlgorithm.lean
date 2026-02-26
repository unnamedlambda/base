import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace StringSearchBench

/-
  String search benchmark — CLIF implementation.

  Counts overlapping occurrences of "that" (4 bytes) in input text.
  Naive O(n*m) scan: compare 4 bytes at each position.

  Memory layout:
    0x0008  FLAG_FILE     (8 bytes)
    0x0010  FLAG_CL       (8 bytes)
    0x0020  INPUT_FILENAME  (256 bytes)
    0x0120  OUTPUT_FILENAME (256 bytes)
    0x0250  FILE_SIZE     (4 bytes, i32 LE)
    0x0350  OUTPUT_BUF    (64 bytes, itoa result)
    0x0400  CLIF_IR       (variable, null-terminated)
    0x4000  INPUT_DATA    (variable)
-/

def FLAG_FILE       : Nat := 0x0008
def INPUT_FILENAME  : Nat := 0x0020
def OUTPUT_FILENAME : Nat := 0x0120
def FILE_SIZE       : Nat := 0x0250
def OUTPUT_BUF      : Nat := 0x0350
def CLIF_IR_OFF     : Nat := 0x0400
def INPUT_DATA      : Nat := 0x4000

def TIMEOUT_MS    : Nat := 300000

/-
  CLIF IR: SIMD-optimized count of "that" occurrences.

  Uses i8x16 vectors to check 16 positions per iteration:
    - 4 SIMD loads at offsets 0,1,2,3 from current position
    - Compare each byte lane against the corresponding pattern byte
    - AND all 4 results for full 4-byte match
    - popcnt on the bitmask counts all matches in the chunk

  No scalar cleanup needed: 256-byte zero padding after data prevents
  false positives from partial reads past the end.
-/
def clifComputeFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i32 v0+592\n" ++             -- FILE_SIZE (0x250)
  "  v2 = sextend.i64 v1\n" ++
  "  v3 = iconst.i64 0\n" ++                -- zero
  "  v4 = iconst.i64 16384\n" ++            -- INPUT_DATA (0x4000)
  "  v5 = iconst.i64 4\n" ++                -- pattern length
  "  v6 = isub v2, v5\n" ++                 -- end = file_size - 4
  "  v7 = iconst.i64 1\n" ++
  "  v8 = iconst.i64 10\n" ++               -- newline / div 10
  "  v9 = iconst.i64 48\n" ++               -- '0'
  "  v10 = iconst.i64 848\n" ++             -- OUTPUT_BUF (0x0350)
  "  v11 = iadd v0, v4\n" ++                -- data_ptr
  "  v12 = iconst.i8 116\n" ++              -- 't' (0x74)
  "  v13 = splat.i8x16 v12\n" ++            -- broadcast 't' to 16 lanes
  "  v14 = iconst.i8 104\n" ++              -- 'h' (0x68)
  "  v15 = splat.i8x16 v14\n" ++            -- broadcast 'h'
  "  v16 = iconst.i8 97\n" ++               -- 'a' (0x61)
  "  v17 = splat.i8x16 v16\n" ++            -- broadcast 'a'
  "  v18 = iconst.i32 0\n" ++               -- zero i32
  "  jump block1(v3, v3)\n" ++
  "\n" ++
  -- block1: SIMD loop (pos, count)
  "block1(v20: i64, v21: i64):\n" ++
  "  v22 = icmp sgt v20, v6\n" ++           -- pos > end?
  "  brif v22, block10(v21), block2(v20, v21)\n" ++
  "\n" ++
  -- block2: SIMD 4-byte pattern match across 16 positions
  "block2(v23: i64, v24: i64):\n" ++
  "  v25 = iadd v11, v23\n" ++              -- data_ptr + pos
  "  v26 = load.i8x16 v25\n" ++             -- 16 bytes at offset 0
  "  v27 = load.i8x16 v25+1\n" ++           -- 16 bytes at offset 1
  "  v28 = load.i8x16 v25+2\n" ++           -- 16 bytes at offset 2
  "  v29 = load.i8x16 v25+3\n" ++           -- 16 bytes at offset 3
  "  v30 = icmp eq v26, v13\n" ++            -- byte[i] == 't'?
  "  v31 = icmp eq v27, v15\n" ++            -- byte[i+1] == 'h'?
  "  v32 = icmp eq v28, v17\n" ++            -- byte[i+2] == 'a'?
  "  v33 = icmp eq v29, v13\n" ++            -- byte[i+3] == 't'?
  "  v34 = band v30, v31\n" ++               -- match bytes 0,1
  "  v35 = band v32, v33\n" ++               -- match bytes 2,3
  "  v36 = band v34, v35\n" ++               -- all 4 bytes match
  "  v37 = vhigh_bits.i32 v36\n" ++          -- extract 16-bit mask
  "  v38 = popcnt v37\n" ++                  -- count matches
  "  v39 = uextend.i64 v38\n" ++
  "  v40 = iadd v24, v39\n" ++               -- count += matches
  "  v41 = iadd_imm v23, 16\n" ++            -- advance 16
  "  jump block1(v41, v40)\n" ++
  "\n" ++
  -- block10: itoa count to OUTPUT_BUF
  "block10(v50: i64):\n" ++
  "  jump block11(v50, v7)\n" ++
  "\n" ++
  "block11(v51: i64, v52: i64):\n" ++
  "  v53 = imul v52, v8\n" ++
  "  v54 = icmp ugt v53, v51\n" ++
  "  brif v54, block12(v51, v52, v10), block11(v51, v53)\n" ++
  "\n" ++
  "block12(v55: i64, v56: i64, v57: i64):\n" ++
  "  v58 = udiv v55, v56\n" ++
  "  v59 = iadd v58, v9\n" ++
  "  v60 = iadd v0, v57\n" ++
  "  istore8 v59, v60\n" ++
  "  v61 = imul v58, v56\n" ++
  "  v62 = isub v55, v61\n" ++
  "  v63 = udiv v56, v8\n" ++
  "  v64 = iadd_imm v57, 1\n" ++
  "  v65 = icmp eq v63, v3\n" ++
  "  brif v65, block13(v64), block12(v62, v63, v64)\n" ++
  "\n" ++
  "block13(v66: i64):\n" ++
  "  v67 = iadd v0, v66\n" ++
  "  istore8 v8, v67\n" ++
  "  v68 = iadd_imm v66, 1\n" ++
  "  v69 = iadd v0, v68\n" ++
  "  v70 = iconst.i32 0\n" ++
  "  istore8 v70, v69\n" ++
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
  let gap3         := zeros (OUTPUT_BUF - 0x0254)
  let outputBuf    := zeros 64
  let gap4         := zeros (CLIF_IR_OFF - (OUTPUT_BUF + 64))
  let currentSize  :=
    gap0.length + flagFile.length + flagCl.length + gap1.length +
    inputFile.length + outputFile.length + gap2.length + fileSize.length +
    gap3.length + outputBuf.length + gap4.length
  let padding := if INPUT_DATA > currentSize then zeros (INPUT_DATA - currentSize) else []
  gap0 ++ flagFile ++ flagCl ++ gap1 ++
    inputFile ++ outputFile ++ gap2 ++ fileSize ++
    gap3 ++ outputBuf ++ gap4 ++ padding

def controlActions : List Action :=
  [
    -- 0: Synchronous CLIF file-read (fn 0)
    { kind := .ClifCall, dst := 0, src := 0, offset := 0, size := 0 },
    -- 1: Synchronous CLIF compute (fn 1)
    { kind := .ClifCall, dst := 0, src := 1, offset := 0, size := 0 },
    -- 2: Synchronous CLIF file-write (fn 2)
    { kind := .ClifCall, dst := 0, src := 2, offset := 0, size := 0 }
  ]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := buildPayload.length,
  context_offset := 0
}

def buildAlgorithm : Algorithm := {
  actions := controlActions,
  payloads := buildPayload,
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

end StringSearchBench

def main : IO Unit := do
  let json := toJsonPair StringSearchBench.buildConfig StringSearchBench.buildAlgorithm
  IO.println (Json.compress json)
