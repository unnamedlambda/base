import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace CsvBench

/-
  CSV benchmark algorithm — Cranelift JIT version.

  The CSV parsing logic is expressed as Cranelift IR (CLIF text format),
  stored in the payload, and JIT-compiled to native code at runtime.
  This is the CPU analog of the GPU unit's WGSL shader compilation.

  Memory layout (shared between harness, file unit, and JIT code):
    0x0008  FLAG_FILE     (8 bytes)
    0x0010  FLAG_CL       (8 bytes, for cranelift dispatch)
    0x0020  INPUT_FILENAME  (256 bytes, patched by harness)
    0x0120  OUTPUT_FILENAME (256 bytes, patched by harness)
    0x02F0  END_POS       (4 bytes i32, CSV byte count, patched by harness)
    0x0350  LEFT_VAL      (32 bytes, itoa result for FileWrite)
    0x0400  CLIF_IR       (null-terminated CLIF text, ~4KB)
    0x2000  CSV_DATA      (variable, populated by FileRead)
-/

-- Memory layout offsets
def FLAG_FILE       : Nat := 0x0008
def INPUT_FILENAME  : Nat := 0x0020
def OUTPUT_FILENAME : Nat := 0x0120
def END_POS         : Nat := 0x02F0
def LEFT_VAL        : Nat := 0x0350
def CLIF_IR_OFF     : Nat := 0x0400
def CSV_DATA        : Nat := 0x2000

def TIMEOUT_MS    : Nat := 300000

/-
  Cranelift IR for CSV parsing — SIMD optimized.

  Uses 16-byte SIMD vectors (i8x16) to scan delimiters (commas, newlines):
    - splat.i8x16: broadcast delimiter byte to 16-lane vector
    - load.i8x16: load 16 bytes from memory
    - icmp eq: parallel comparison across all 16 lanes
    - vhigh_bits: extract comparison mask (1 bit per lane)
    - ctz: count trailing zeros to find first match

  Function signature: fn(memory_base_ptr: i64) -> void

  Algorithm:
  1. Load end_pos from +752 (0x02F0)
  2. Skip header: SIMD scan 16 bytes/iter for newline, byte fallback for tail
  3. For each data row:
     a. SIMD scan for commas (16 bytes/iter), count to 5th
     b. Parse salary digits byte-by-byte into accum
     c. Add accum to running total (i64)
  4. itoa: convert i64 total to ASCII at +848 (0x0350)

  Offsets: +752=END_POS, +8192=CSV_DATA, +848=LEFT_VAL
-/
def clifComputeFn : String :=
  "function u0:1(i64) system_v {\n" ++
  -- entry: load end_pos, setup SIMD vectors and hoisted constants
  "block0(v0: i64):\n" ++
  "  v1 = load.i32 v0+752\n" ++
  "  v200 = sextend.i64 v1\n" ++
  "  v6 = iconst.i64 0\n" ++
  "  v7 = iconst.i32 0\n" ++
  -- SIMD vectors for 16-byte scanning
  "  v300 = iconst.i8 10\n" ++
  "  v301 = splat.i8x16 v300\n" ++
  "  v302 = iconst.i8 44\n" ++
  "  v303 = splat.i8x16 v302\n" ++
  -- hoisted scalar constants
  "  v201 = iconst.i64 10\n" ++
  "  v202 = iconst.i64 48\n" ++
  "  v203 = iconst.i32 5\n" ++
  "  v204 = iconst.i32 44\n" ++
  "  v205 = iconst.i32 10\n" ++
  "  v206 = iconst.i64 1\n" ++
  "  v207 = iconst.i64 848\n" ++
  "  jump block1(v6)\n" ++
  "\n" ++
  -- skip_header_16_check(pos: i64): can we load 16 bytes?
  "block1(v8: i64):\n" ++
  "  v9 = iadd_imm v8, 16\n" ++
  "  v10 = icmp sle v9, v200\n" ++
  "  brif v10, block2(v8), block3(v8)\n" ++
  "\n" ++
  -- skip_header_16_body: SIMD newline detect (16 bytes/iter)
  "block2(v11: i64):\n" ++
  "  v13 = iadd v0, v11\n" ++
  "  v14 = load.i8x16 v13+8192\n" ++
  "  v15 = icmp eq v14, v301\n" ++
  "  v16 = vhigh_bits.i32 v15\n" ++
  "  v20 = icmp ne v16, v7\n" ++
  "  v21 = iadd_imm v11, 16\n" ++
  "  brif v20, block4(v11, v16), block1(v21)\n" ++
  "\n" ++
  -- skip_header_1: byte-by-byte fallback (uses hoisted v205 = 10)
  "block3(v22: i64):\n" ++
  "  v24 = iadd v0, v22\n" ++
  "  v25 = uload8.i32 v24+8192\n" ++
  "  v27 = icmp eq v25, v205\n" ++
  "  v28 = iadd_imm v22, 1\n" ++
  "  brif v27, block5(v22), block3(v28)\n" ++
  "\n" ++
  -- skip_header_found: ctz gives byte position directly (1 bit per byte)
  "block4(v29: i64, v30: i32):\n" ++
  "  v31 = ctz v30\n" ++
  "  v32 = uextend.i64 v31\n" ++
  "  v34 = iadd v29, v32\n" ++
  "  jump block5(v34)\n" ++
  "\n" ++
  -- header_done: advance past newline, jump to comma scan
  "block5(v35: i64):\n" ++
  "  v36 = iadd_imm v35, 1\n" ++
  "  jump block7(v36, v6, v7)\n" ++
  "\n" ++
  -- comma_scan_16_check(pos: i64, total, cc)
  "block7(v39: i64, v40: i64, v41: i32):\n" ++
  "  v42 = iadd_imm v39, 16\n" ++
  "  v43 = icmp sle v42, v200\n" ++
  "  brif v43, block8(v39, v40, v41), block9(v39, v40, v41)\n" ++
  "\n" ++
  -- comma_scan_16_body: SIMD comma detect (16 bytes/iter)
  "block8(v44: i64, v45: i64, v46: i32):\n" ++
  "  v48 = iadd v0, v44\n" ++
  "  v49 = load.i8x16 v48+8192\n" ++
  "  v50 = icmp eq v49, v303\n" ++
  "  v54 = vhigh_bits.i32 v50\n" ++
  "  v55 = icmp ne v54, v7\n" ++
  "  v56 = iadd_imm v44, 16\n" ++
  "  brif v55, block10(v44, v45, v46, v54), block7(v56, v45, v46)\n" ++
  "\n" ++
  -- comma_scan_1: byte-by-byte fallback (uses hoisted v204 = 44)
  "block9(v57: i64, v58: i64, v59: i32):\n" ++
  "  v61 = iadd v0, v57\n" ++
  "  v62 = uload8.i32 v61+8192\n" ++
  "  v64 = icmp eq v62, v204\n" ++
  "  v65 = iadd_imm v57, 1\n" ++
  "  brif v64, block11(v57, v58, v59), block9(v65, v58, v59)\n" ++
  "\n" ++
  -- comma_found_16: ctz gives byte position directly, count++, check ==5
  "block10(v66: i64, v67: i64, v68: i32, v69: i32):\n" ++
  "  v70 = ctz v69\n" ++
  "  v71 = uextend.i64 v70\n" ++
  "  v73 = iadd v66, v71\n" ++
  "  v74 = iadd_imm v68, 1\n" ++
  "  v76 = icmp eq v74, v203\n" ++
  "  v77 = iadd_imm v73, 1\n" ++
  "  brif v76, block13(v77, v67, v6), block7(v77, v67, v74)\n" ++
  "\n" ++
  -- comma_found_1: count++, check ==5 (uses hoisted v203 = 5)
  "block11(v78: i64, v79: i64, v80: i32):\n" ++
  "  v81 = iadd_imm v80, 1\n" ++
  "  v83 = icmp eq v81, v203\n" ++
  "  v84 = iadd_imm v78, 1\n" ++
  "  brif v83, block13(v84, v79, v6), block7(v84, v79, v81)\n" ++
  "\n" ++
  -- digit_loop(pos: i64, total: i64, accum: i64)
  "block13(v88: i64, v89: i64, v90: i64):\n" ++
  "  v92 = iadd v0, v88\n" ++
  "  v93 = uload8.i64 v92+8192\n" ++
  "  v95 = icmp eq v93, v201\n" ++
  "  brif v95, block15(v88, v89, v90), block14(v88, v89, v90, v93)\n" ++
  "\n" ++
  -- is_digit: accum = accum*10 + (char - '0'), all i64
  "block14(v96: i64, v97: i64, v98: i64, v99: i64):\n" ++
  "  v101 = imul v98, v201\n" ++
  "  v103 = isub v99, v202\n" ++
  "  v104 = iadd v101, v103\n" ++
  "  v105 = iadd_imm v96, 1\n" ++
  "  jump block13(v105, v97, v104)\n" ++
  "\n" ++
  -- salary_done: total += accum (i64), check end
  "block15(v106: i64, v107: i64, v108: i64):\n" ++
  "  v110 = iadd v107, v108\n" ++
  "  v111 = iadd_imm v106, 1\n" ++
  "  v112 = icmp sge v111, v200\n" ++
  "  brif v112, block16(v110), block7(v111, v110, v7)\n" ++
  "\n" ++
  -- itoa_start: find largest power of 10 (i64)
  "block16(v113: i64):\n" ++
  "  jump block17(v113, v206)\n" ++
  "\n" ++
  -- find_divisor(total, div)
  "block17(v115: i64, v116: i64):\n" ++
  "  v118 = imul v116, v201\n" ++
  "  v119 = icmp ugt v118, v115\n" ++
  "  brif v119, block18(v115, v116), block17(v115, v118)\n" ++
  "\n" ++
  -- write_digits_start(total, div)
  "block18(v120: i64, v121: i64):\n" ++
  "  jump block19(v120, v121, v207)\n" ++
  "\n" ++
  -- write_digit(val, div, wpos: i64)
  "block19(v123: i64, v124: i64, v125: i64):\n" ++
  "  v126 = udiv v123, v124\n" ++
  "  v128 = iadd v126, v202\n" ++
  "  v130 = iadd v0, v125\n" ++
  "  istore8 v128, v130\n" ++
  "  v131 = imul v126, v124\n" ++
  "  v132 = isub v123, v131\n" ++
  "  v134 = udiv v124, v201\n" ++
  "  v135 = iadd_imm v125, 1\n" ++
  "  v136 = icmp eq v134, v6\n" ++
  "  brif v136, block20(v135), block19(v132, v134, v135)\n" ++
  "\n" ++
  -- write_newline_and_null(wpos: i64)
  "block20(v137: i64):\n" ++
  "  v139 = iadd v0, v137\n" ++
  "  istore8 v205, v139\n" ++
  "  v141 = iadd_imm v137, 1\n" ++
  "  v143 = iadd v0, v141\n" ++
  "  istore8 v7, v143\n" ++
  "  return\n" ++
  "}\n"

def clifReadFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "  sig0 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "  fn0 = %cl_file_read sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = iconst.i64 32\n" ++        -- INPUT_FILENAME
  "  v2 = iconst.i64 8192\n" ++      -- CSV_DATA
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
  "  v2 = iconst.i64 848\n" ++       -- LEFT_VAL
  "  v3 = iconst.i64 0\n" ++
  "  v4 = iconst.i64 0\n" ++
  "  v5 = call fn0(v0, v1, v2, v3, v4)\n" ++
  "  return\n" ++
  "}\n"

def clifIR : String :=
  clifReadFn ++ "\n" ++ clifComputeFn ++ "\n" ++ clifWriteFn

-- Convert CLIF IR string to bytes for payload
def clifIRBytes : List UInt8 :=
  clifIR.toUTF8.toList ++ [0]  -- null-terminated

-- Build payload
def buildPayload : List UInt8 :=
  let gap0         := zeros FLAG_FILE            -- 0x0000..0x0007
  let flagFile     := zeros 8                    -- 0x0008..0x000F
  let flagCl       := zeros 8                    -- 0x0010..0x0017
  let gap_to_input := zeros 8                    -- 0x0018..0x001F
  let inputFile    := zeros 256                  -- 0x0020..0x011F
  let outputFile   := zeros 256                  -- 0x0120..0x021F
  let gap1         := zeros (END_POS - 0x0220)   -- 0x0220..0x02EF
  let endPos       := zeros 4                    -- 0x02F0..0x02F3
  let gap2         := zeros (LEFT_VAL - 0x02F4)  -- 0x02F4..0x034F
  let leftVal      := zeros 32                   -- 0x0350..0x036F
  let gap3         := zeros (CLIF_IR_OFF - 0x0370) -- 0x0370..0x03FF
  -- Pad to CSV_DATA start (0x2000 = 8192)
  let currentSize :=
    gap0.length + flagFile.length + flagCl.length + gap_to_input.length +
    inputFile.length + outputFile.length +
    gap1.length + endPos.length + gap2.length + leftVal.length + gap3.length
  let padding := if CSV_DATA > currentSize then zeros (CSV_DATA - currentSize) else []
  gap0 ++ flagFile ++ flagCl ++ gap_to_input ++
    inputFile ++ outputFile ++
    gap1 ++ endPos ++ gap2 ++ leftVal ++ gap3 ++ padding

-- Control actions (synchronous ClifCall)
def controlActions : List Action :=
  [
    -- 0: Synchronous CLIF file-read (fn 0)
    { kind := .ClifCall, dst := 0, src := 0, offset := 0, size := 0 },
    -- 1: Synchronous CLIF compute (fn 1)
    { kind := .ClifCall, dst := 0, src := 1, offset := 0, size := 0 },
    -- 2: Synchronous CLIF file-write (fn 2)
    { kind := .ClifCall, dst := 0, src := 2, offset := 0, size := 0 }
  ]

-- Build the full algorithm
def buildAlgorithm : Algorithm := {
  actions := controlActions,
  payloads := buildPayload,
  cranelift_ir := clifIR,
  units := {
    cranelift_units := 0,
  },
  timeout_ms := some TIMEOUT_MS,
  additional_shared_memory := 0
}

end CsvBench

def main : IO Unit := do
  let json := toJson CsvBench.buildAlgorithm
  IO.println (Json.compress json)
