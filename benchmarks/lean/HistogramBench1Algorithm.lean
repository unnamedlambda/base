import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace HistogramBench1

/-
  Single-threaded histogram benchmark algorithm (w=1).

  Payload (via execute data arg): "input_path\0output_path\0"

  Memory layout:
    0x0000  RESERVED        (40 bytes, runtime-managed)
    0x0100  INPUT_PATH      (256 bytes, copied from payload by CLIF)
    0x0200  OUTPUT_PATH     (256 bytes, copied from payload by CLIF)
    0x0400  HISTOGRAM       (256 bins × 8 bytes = 2048 bytes)
    0x0C00  DATA            (up to 64MB of u32 values, read from file)
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def HIST_OFF        : Nat := 0x0400
def BINS            : Nat := 256
def HIST_BYTES      : Nat := BINS * 8    -- 2048
def DATA_OFF        : Nat := HIST_OFF + HIST_BYTES  -- 0x0C00 = 3072
def MAX_DATA_BYTES  : Nat := 64 * 1024 * 1024
def MEM_SIZE        : Nat := DATA_OFF + MAX_DATA_BYTES
def TIMEOUT_MS      : Nat := 60000

-- fn0: noop
def clifNoopFn0 : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: noop
def clifNoopFn1 : String :=
  "function u0:1(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn2: single-threaded orchestrator
-- 1. Copy input/output paths from payload to shared memory
-- 2. Read file into data region
-- 3. Zero histogram
-- 4. 4x-unrolled histogram scan with scalar tail
-- 5. Write result to file
def clifOrchestratorFn : String :=
  "function u0:2(i64) system_v {\n" ++
  "    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "    sig1 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "\n" ++
  "    fn0 = %cl_file_read sig0\n" ++
  "    fn1 = %cl_file_write sig1\n" ++
  "\n" ++
  "block0(v0: i64):\n" ++
  "    v1 = load.i64 notrap aligned v0+0x08\n" ++
  "    v200 = iconst.i64 0\n" ++
  "    jump block1(v200)\n" ++
  "\n" ++
  -- Copy input path from payload to shared memory
  "block1(v201: i64):\n" ++
  "    v202 = iadd v1, v201\n" ++
  "    v203 = uload8.i64 notrap v202\n" ++
  "    v204 = iadd_imm v0, 256\n" ++       -- INPUT_PATH_OFF
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
  "    v224 = iadd_imm v0, 512\n" ++       -- OUTPUT_PATH_OFF
  "    v225 = iadd v224, v221\n" ++
  "    istore8 v223, v225\n" ++
  "    v226 = icmp_imm eq v223, 0\n" ++
  "    v227 = iadd_imm v220, 1\n" ++
  "    v228 = iadd_imm v221, 1\n" ++
  "    brif v226, block4, block3(v227, v228)\n" ++
  "\n" ++
  -- Read file, zero histogram
  "block4:\n" ++
  "    v3 = iconst.i64 256\n" ++           -- INPUT_PATH_OFF
  "    v4 = iconst.i64 3072\n" ++          -- DATA_OFF
  "    v5 = iconst.i64 0\n" ++
  "    v6 = iconst.i64 0\n" ++
  "    v7 = call fn0(v0, v3, v4, v5, v6)\n" ++
  "    v8 = ushr_imm v7, 2\n" ++           -- n = bytes_read / 4
  "\n" ++
  "    v10 = iadd_imm v0, 1024\n" ++       -- HIST_OFF → hist_ptr
  "    v11 = iconst.i64 2048\n" ++         -- HIST_BYTES
  "    v12 = iadd v10, v11\n" ++           -- hist_end
  "    jump block5(v10)\n" ++
  "\n" ++
  -- Zero histogram (8x unrolled, 64 bytes/iter)
  "block5(v21: i64):\n" ++
  "    store.i64 notrap aligned v200, v21\n" ++
  "    store.i64 notrap aligned v200, v21+8\n" ++
  "    store.i64 notrap aligned v200, v21+16\n" ++
  "    store.i64 notrap aligned v200, v21+24\n" ++
  "    store.i64 notrap aligned v200, v21+32\n" ++
  "    store.i64 notrap aligned v200, v21+40\n" ++
  "    store.i64 notrap aligned v200, v21+48\n" ++
  "    store.i64 notrap aligned v200, v21+56\n" ++
  "    v22 = iadd_imm v21, 64\n" ++
  "    v23 = icmp ult v22, v12\n" ++
  "    brif v23, block5(v22), block6\n" ++
  "\n" ++
  -- Prepare scan pointers
  "block6:\n" ++
  "    v30 = iadd_imm v0, 3072\n" ++       -- DATA_OFF → data_ptr
  "    v31 = ishl_imm v8, 2\n" ++
  "    v32 = iadd v30, v31\n" ++           -- data_end
  "    v33 = band_imm v8, -4\n" ++
  "    v34 = ishl_imm v33, 2\n" ++
  "    v35 = iadd v30, v34\n" ++           -- data_end_4x
  "    v36 = icmp ult v30, v35\n" ++
  "    brif v36, block7(v30), block8(v30)\n" ++
  "\n" ++
  -- 4x unrolled histogram scan
  "block7(v40: i64):\n" ++
  "    v41 = uload32.i64 notrap aligned v40\n" ++
  "    v42 = ishl_imm v41, 3\n" ++
  "    v43 = iadd v10, v42\n" ++
  "    v44 = load.i64 notrap aligned v43\n" ++
  "    v45 = iadd_imm v44, 1\n" ++
  "    store.i64 notrap aligned v45, v43\n" ++
  "\n" ++
  "    v50 = uload32.i64 notrap aligned v40+4\n" ++
  "    v51 = ishl_imm v50, 3\n" ++
  "    v52 = iadd v10, v51\n" ++
  "    v53 = load.i64 notrap aligned v52\n" ++
  "    v54 = iadd_imm v53, 1\n" ++
  "    store.i64 notrap aligned v54, v52\n" ++
  "\n" ++
  "    v60 = uload32.i64 notrap aligned v40+8\n" ++
  "    v61 = ishl_imm v60, 3\n" ++
  "    v62 = iadd v10, v61\n" ++
  "    v63 = load.i64 notrap aligned v62\n" ++
  "    v64 = iadd_imm v63, 1\n" ++
  "    store.i64 notrap aligned v64, v62\n" ++
  "\n" ++
  "    v70 = uload32.i64 notrap aligned v40+12\n" ++
  "    v71 = ishl_imm v70, 3\n" ++
  "    v72 = iadd v10, v71\n" ++
  "    v73 = load.i64 notrap aligned v72\n" ++
  "    v74 = iadd_imm v73, 1\n" ++
  "    store.i64 notrap aligned v74, v72\n" ++
  "\n" ++
  "    v80 = iadd_imm v40, 16\n" ++
  "    v81 = icmp ult v80, v35\n" ++
  "    brif v81, block7(v80), block8(v35)\n" ++
  "\n" ++
  -- Scalar tail
  "block8(v90: i64):\n" ++
  "    v91 = icmp ult v90, v32\n" ++
  "    brif v91, block9(v90), block10\n" ++
  "\n" ++
  "block9(v100: i64):\n" ++
  "    v101 = uload32.i64 notrap aligned v100\n" ++
  "    v102 = ishl_imm v101, 3\n" ++
  "    v103 = iadd v10, v102\n" ++
  "    v104 = load.i64 notrap aligned v103\n" ++
  "    v105 = iadd_imm v104, 1\n" ++
  "    store.i64 notrap aligned v105, v103\n" ++
  "    v106 = iadd_imm v100, 4\n" ++
  "    v107 = icmp ult v106, v32\n" ++
  "    brif v107, block9(v106), block10\n" ++
  "\n" ++
  -- Write result
  "block10:\n" ++
  "    v95 = iconst.i64 512\n" ++          -- OUTPUT_PATH_OFF
  "    v96 = iconst.i64 1024\n" ++         -- HIST_OFF (result is the histogram itself)
  "    v97 = iconst.i64 0\n" ++
  "    v98 = iconst.i64 2048\n" ++         -- HIST_BYTES
  "    v99 = call fn1(v0, v95, v96, v97, v98)\n" ++
  "    return\n" ++
  "}\n"

def clifIR : String :=
  clifNoopFn0 ++ "\n" ++ clifNoopFn1 ++ "\n" ++ clifOrchestratorFn

def controlActions : List Action :=
  [{ kind := .ClifCall, dst := 0, src := 2, offset := 0, size := 0 }]

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

end HistogramBench1

def main : IO Unit := do
  let json := toJsonPair HistogramBench1.buildConfig HistogramBench1.buildAlgorithm
  IO.println (Json.compress json)
