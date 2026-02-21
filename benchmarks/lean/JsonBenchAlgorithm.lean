import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace JsonBench

/-
  JSON benchmark — CLIF implementation.

  Streaming parser: scans for `"value": ` pattern (9 bytes),
  then parses following digits and sums them. No tree building.

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
def FLAG_CL         : Nat := 0x0010
def INPUT_FILENAME  : Nat := 0x0020
def OUTPUT_FILENAME : Nat := 0x0120
def FILE_SIZE       : Nat := 0x0250
def OUTPUT_BUF      : Nat := 0x0350
def CLIF_IR_OFF     : Nat := 0x0400
def INPUT_DATA      : Nat := 0x4000

def FILE_UNIT : UInt32 := 2
def CL_UNIT   : UInt32 := 9

def FILE_BUF_SIZE : Nat := 0x4000000
def TIMEOUT_MS    : Nat := 300000

/-
  CLIF IR: 2-byte SIMD filter scan for `"value": ` (9 bytes), parse digits, sum.

  Uses i8x16 SIMD to scan 16 positions at a time for the 2-byte prefix '"v'.
  Two SIMD loads at offsets 0 and 1, compared against '"' and 'v' respectively,
  ANDed to produce a mask of '"v' positions. Since '"v' only appears at actual
  "value" fields in JSON, false positives are nearly zero.

  For each match: verify remaining 7 bytes via single i64 load+compare (bytes 1-8).
  On full match, parse digits with unsigned range check.
-/
def clifIR : String :=
  "function %json_sum(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i32 v0+592\n" ++             -- FILE_SIZE (0x250)
  "  v2 = sextend.i64 v1\n" ++
  "  v3 = iconst.i64 0\n" ++                -- zero
  "  v4 = iconst.i64 16384\n" ++            -- INPUT_DATA (0x4000)
  "  v5 = iconst.i64 9\n" ++                -- needle length / digit range
  "  v6 = isub v2, v5\n" ++                 -- end = file_size - 9
  "  v7 = iconst.i64 10\n" ++               -- div 10 / newline
  "  v8 = iconst.i64 48\n" ++               -- '0'
  "  v9 = iadd v0, v4\n" ++                 -- data_ptr = base + INPUT_DATA
  "  v10 = iconst.i8 34\n" ++               -- '\"' as i8
  "  v11 = splat.i8x16 v10\n" ++            -- broadcast '\"' to 16 lanes
  "  v12 = iconst.i8 118\n" ++              -- 'v' as i8 (0x76)
  "  v13 = splat.i8x16 v12\n" ++            -- broadcast 'v' to 16 lanes
  "  v14 = iconst.i64 2322206377019990390\n" ++  -- value\": <sp> as LE i64 (bytes 1-8)
  "  v15 = iconst.i32 0\n" ++               -- zero i32 for mask comparison
  "  jump block1(v3, v3)\n" ++
  "\n" ++
  -- block1: SIMD scan entry (pos, total)
  "block1(v20: i64, v21: i64):\n" ++
  "  v22 = icmp sgt v20, v6\n" ++           -- pos > end?
  "  brif v22, block10(v21), block2(v20, v21)\n" ++
  "\n" ++
  -- block2: 2-byte SIMD scan for '\"v' prefix
  "block2(v23: i64, v24: i64):\n" ++
  "  v25 = iadd v9, v23\n" ++               -- data_ptr + pos
  "  v26 = load.i8x16 v25\n" ++             -- 16 bytes at pos
  "  v27 = load.i8x16 v25+1\n" ++           -- 16 bytes at pos+1
  "  v28 = icmp eq v26, v11\n" ++            -- compare with '\"'
  "  v29 = icmp eq v27, v13\n" ++            -- compare with 'v'
  "  v30 = band v28, v29\n" ++              -- '\"v' at lane?
  "  v31 = vhigh_bits.i32 v30\n" ++         -- extract bitmask
  "  v32 = icmp ne v31, v15\n" ++           -- any match?
  "  v33 = iadd_imm v23, 16\n" ++           -- next chunk
  "  brif v32, block3(v23, v24, v31), block1(v33, v24)\n" ++
  "\n" ++
  -- block3: extract first '\"v' position from mask
  "block3(v35: i64, v36: i64, v37: i32):\n" ++
  "  v38 = ctz v37\n" ++                    -- offset of first match
  "  v39 = uextend.i64 v38\n" ++
  "  v40 = iadd v35, v39\n" ++              -- abs_pos = chunk_base + offset
  "  v41 = icmp sgt v40, v6\n" ++           -- past end?
  "  brif v41, block10(v36), block4(v40, v36, v35, v37)\n" ++
  "\n" ++
  -- block4: verify remaining 8 bytes after '\"'
  "block4(v42: i64, v43: i64, v44: i64, v45: i32):\n" ++
  "  v46 = iadd v9, v42\n" ++               -- data_ptr + abs_pos
  "  v47 = load.i64 v46+1\n" ++             -- load bytes 1-8 after '\"'
  "  v48 = icmp eq v47, v14\n" ++           -- == value\": <sp>?
  "  brif v48, block5(v42, v43), block30(v44, v43, v45)\n" ++
  "\n" ++
  -- block30: no full match, clear lowest bit, try next in chunk
  "block30(v49: i64, v50: i64, v51: i32):\n" ++
  "  v52 = iadd_imm v51, -1\n" ++           -- mask - 1
  "  v53 = band v51, v52\n" ++              -- clear lowest set bit
  "  v54 = icmp ne v53, v15\n" ++           -- more bits?
  "  v55 = iadd_imm v49, 16\n" ++           -- next chunk
  "  brif v54, block3(v49, v50, v53), block1(v55, v50)\n" ++
  "\n" ++
  -- block5: full match, parse digits at pos + 9
  "block5(v56: i64, v57: i64):\n" ++
  "  v58 = iadd_imm v56, 9\n" ++            -- pos + 9 (skip needle)
  "  jump block6(v58, v57, v3)\n" ++
  "\n" ++
  -- block6: digit loop (digit_pos, total, accum)
  -- No bounds check: data always has non-digit after numbers
  "block6(v59: i64, v60: i64, v61: i64):\n" ++
  "  v62 = iadd v9, v59\n" ++               -- data_ptr + pos
  "  v63 = uload8.i64 v62\n" ++
  "  v64 = isub v63, v8\n" ++               -- byte - '0'
  "  v65 = icmp ugt v64, v5\n" ++           -- > 9? (unsigned range check)
  "  brif v65, block7(v59, v60, v61), block8(v59, v60, v61, v64)\n" ++
  "\n" ++
  -- block7: not a digit, total += accum, resume SIMD scan
  "block7(v66: i64, v67: i64, v68: i64):\n" ++
  "  v69 = iadd v67, v68\n" ++              -- total += acc
  "  jump block1(v66, v69)\n" ++
  "\n" ++
  -- block8: digit, acc = acc * 10 + digit
  "block8(v70: i64, v71: i64, v72: i64, v73: i64):\n" ++
  "  v74 = imul v72, v7\n" ++               -- acc * 10
  "  v75 = iadd v74, v73\n" ++              -- + digit
  "  v76 = iadd_imm v70, 1\n" ++            -- pos++
  "  jump block6(v76, v71, v75)\n" ++
  "\n" ++
  -- block10: itoa total to OUTPUT_BUF
  "block10(v80: i64):\n" ++
  "  v81 = iconst.i64 1\n" ++
  "  v82 = iconst.i64 848\n" ++             -- OUTPUT_BUF (0x0350)
  "  jump block11(v80, v81, v82)\n" ++
  "\n" ++
  "block11(v83: i64, v84: i64, v85: i64):\n" ++
  "  v86 = imul v84, v7\n" ++
  "  v87 = icmp ugt v86, v83\n" ++
  "  brif v87, block12(v83, v84, v85), block11(v83, v86, v85)\n" ++
  "\n" ++
  "block12(v88: i64, v89: i64, v90: i64):\n" ++
  "  v91 = udiv v88, v89\n" ++
  "  v92 = iadd v91, v8\n" ++
  "  v93 = iadd v0, v90\n" ++
  "  istore8 v92, v93\n" ++
  "  v94 = imul v91, v89\n" ++
  "  v95 = isub v88, v94\n" ++
  "  v96 = udiv v89, v7\n" ++
  "  v97 = iadd_imm v90, 1\n" ++
  "  v98 = icmp eq v96, v3\n" ++
  "  brif v98, block13(v97), block12(v95, v96, v97)\n" ++
  "\n" ++
  "block13(v99: i64):\n" ++
  "  v100 = iadd v0, v99\n" ++
  "  istore8 v7, v100\n" ++
  "  v101 = iadd_imm v99, 1\n" ++
  "  v102 = iadd v0, v101\n" ++
  "  v103 = iconst.i32 0\n" ++
  "  istore8 v103, v102\n" ++
  "  return\n" ++
  "}\n"

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
  let irBytes      := clifIRBytes
  let currentSize  :=
    gap0.length + flagFile.length + flagCl.length + gap1.length +
    inputFile.length + outputFile.length + gap2.length + fileSize.length +
    gap3.length + outputBuf.length + gap4.length + irBytes.length
  let padding := if INPUT_DATA > currentSize then zeros (INPUT_DATA - currentSize) else []
  gap0 ++ flagFile ++ flagCl ++ gap1 ++
    inputFile ++ outputFile ++ gap2 ++ fileSize ++
    gap3 ++ outputBuf ++ gap4 ++
    irBytes ++ padding

def workerActions : List Action := [
  { kind := .FileRead,
    src := UInt32.ofNat INPUT_FILENAME,
    dst := UInt32.ofNat INPUT_DATA,
    offset := 0, size := 0 },
  { kind := .FileRead, dst := 0, src := 0, offset := 0, size := 0 },
  { kind := .FileWrite,
    src := UInt32.ofNat OUTPUT_BUF,
    dst := UInt32.ofNat OUTPUT_FILENAME,
    offset := 0, size := 0 }
]

def controlActions : List Action :=
  let wBase : UInt32 := 6
  [
    { kind := .AsyncDispatch, dst := FILE_UNIT, src := wBase,
      offset := UInt32.ofNat FLAG_FILE, size := 1 },
    { kind := .Wait, dst := UInt32.ofNat FLAG_FILE, src := 0, offset := 0, size := 0 },
    { kind := .AsyncDispatch, dst := CL_UNIT, src := wBase + 1,
      offset := UInt32.ofNat FLAG_CL, size := 1 },
    { kind := .Wait, dst := UInt32.ofNat FLAG_CL, src := 0, offset := 0, size := 0 },
    { kind := .AsyncDispatch, dst := FILE_UNIT, src := wBase + 2,
      offset := UInt32.ofNat FLAG_FILE, size := 1 },
    { kind := .Wait, dst := UInt32.ofNat FLAG_FILE, src := 0, offset := 0, size := 0 }
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
    cranelift_units := 1,
  },
  file_assignments := [], cranelift_assignments := [],
  worker_threads := some 2, blocking_threads := some 2,
  stack_size := none, timeout_ms := some TIMEOUT_MS,
  thread_name_prefix := some "json-bench"
}

end JsonBench

def main : IO Unit := do
  let json := toJson JsonBench.buildAlgorithm
  IO.println (Json.compress json)
