import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace HistogramBench4

/-
  Multi-threaded histogram benchmark algorithm (w=4).

  Payload (via execute data arg): "input_path\0output_path\0"

  Memory layout:
    0x0000  RESERVED        (40 bytes, runtime-managed)
    0x0100  INPUT_PATH      (256 bytes, copied from payload by CLIF)
    0x0200  OUTPUT_PATH     (256 bytes, copied from payload by CLIF)
    0x0300  THREAD_CTX      (8 bytes)
    0x0400  HIST[0]         (4096 bytes, 4KB-aligned per worker)
    0x1400  HIST[1]         (4096 bytes)
    0x2400  HIST[2]         (4096 bytes)
    0x3400  HIST[3]         (4096 bytes)
    0x4400  RESULT          (2048 bytes, merged histogram)
    0x4C00  HANDLES         (32 bytes, 4 × i64)
    0x4C40  DESCRIPTORS     (192 bytes, 4 × 48-byte worker descriptors)
    0x4D00  DATA            (up to 64MB of u32 values, read from file)

  Worker descriptor (48 bytes):
    +0x00  base_ptr, +0x08 data_off, +0x10 data_start, +0x18 data_count,
    +0x20  hist_off, +0x28 bins
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def THREAD_CTX_OFF  : Nat := 0x0300
def HIST_REGION_OFF : Nat := 0x0400
def BINS            : Nat := 256
def WORKERS         : Nat := 4
def HIST_STRIDE     : Nat := 4096   -- (256*8 + 4095) & ~4095
def RESULT_OFF      : Nat := HIST_REGION_OFF + WORKERS * HIST_STRIDE  -- 0x4400 = 17408
def RESULT_SIZE     : Nat := BINS * 8  -- 2048
def HANDLES_OFF     : Nat := 19456  -- (RESULT_OFF + RESULT_SIZE + 63) & ~63
def DESCS_OFF       : Nat := 19520  -- (HANDLES_OFF + WORKERS*8 + 63) & ~63
def DESC_SIZE       : Nat := 48
def DATA_OFF        : Nat := 19712  -- (DESCS_OFF + WORKERS*DESC_SIZE + 63) & ~63
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

-- fn2: multi-threaded orchestrator
-- 1. Copy input/output paths from payload to shared memory
-- 2. Read file, compute n/chunk_size
-- 3. Spawn 4 worker threads (each runs fn3)
-- 4. Join all threads
-- 5. Merge per-worker histograms into result
-- 6. Write result to file, cleanup threads
def clifOrchestratorFn : String :=
  "function u0:2(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++
  "    sig1 = (i64, i64, i64) -> i64 system_v\n" ++
  "    sig2 = (i64, i64) -> i64 system_v\n" ++
  "    sig3 = (i64) system_v\n" ++
  "    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "\n" ++
  "    fn0 = %cl_thread_init sig0\n" ++
  "    fn1 = %cl_thread_spawn sig1\n" ++
  "    fn2 = %cl_thread_join sig2\n" ++
  "    fn3 = %cl_thread_cleanup sig3\n" ++
  "    fn4 = %cl_file_read sig5\n" ++
  "    fn5 = %cl_file_write sig4\n" ++
  "\n" ++
  "block0(v0: i64):\n" ++
  "    v1 = load.i64 notrap aligned v0+0x08\n" ++
  "    v200 = iconst.i64 0\n" ++
  "    jump block1(v200)\n" ++
  "\n" ++
  -- Copy input path
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
  -- Copy output path
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
  -- Read file, compute n/chunk, init threads
  "block4:\n" ++
  "    v3 = iconst.i64 256\n" ++           -- INPUT_PATH_OFF
  "    v4 = iconst.i64 19712\n" ++         -- DATA_OFF
  "    v5 = iconst.i64 0\n" ++
  "    v6 = iconst.i64 0\n" ++
  "    v7 = call fn4(v0, v3, v4, v5, v6)\n" ++
  "\n" ++
  "    v8 = ushr_imm v7, 2\n" ++           -- n = bytes_read / 4
  "    v9 = iconst.i64 4\n" ++             -- WORKERS
  "    v10 = iadd_imm v8, 3\n" ++          -- n + WORKERS - 1
  "    v11 = udiv v10, v9\n" ++            -- chunk_size
  "\n" ++
  "    v12 = iconst.i64 256\n" ++          -- BINS
  "    v13 = iconst.i64 4096\n" ++         -- HIST_STRIDE
  "\n" ++
  "    v14 = iadd_imm v0, 768\n" ++        -- THREAD_CTX_OFF
  "    call fn0(v14)\n" ++
  "\n" ++
  "    v16 = iconst.i64 3\n" ++            -- fn_index for worker
  "    v17 = iconst.i64 19520\n" ++        -- DESCS_OFF
  "    v18 = iconst.i64 48\n" ++           -- DESC_SIZE
  "    v19 = iconst.i64 19456\n" ++        -- HANDLES_OFF
  "    v20 = iconst.i64 19712\n" ++        -- DATA_OFF
  "    jump block5(v200, v8, v11)\n" ++
  "\n" ++
  -- Spawn workers
  "block5(v30: i64, v31: i64, v32: i64):\n" ++
  "    v33 = imul v30, v18\n" ++
  "    v34 = iadd v0, v17\n" ++
  "    v35 = iadd v34, v33\n" ++           -- desc_ptr
  "\n" ++
  "    store.i64 notrap aligned v0, v35\n" ++           -- base_ptr
  "    store.i64 notrap aligned v20, v35+0x08\n" ++     -- data_off
  "    v36 = imul v30, v32\n" ++
  "    store.i64 notrap aligned v36, v35+0x10\n" ++     -- data_start
  "    v37 = isub v31, v36\n" ++
  "    v38 = icmp ult v37, v32\n" ++
  "    v39 = select v38, v37, v32\n" ++
  "    store.i64 notrap aligned v39, v35+0x18\n" ++     -- data_count
  "    v40 = imul v30, v13\n" ++
  "    v41 = iadd_imm v40, 1024\n" ++      -- HIST_REGION_OFF
  "    store.i64 notrap aligned v41, v35+0x20\n" ++     -- hist_off
  "    store.i64 notrap aligned v12, v35+0x28\n" ++     -- bins
  "\n" ++
  "    v42 = iadd_imm v0, 768\n" ++        -- THREAD_CTX_OFF
  "    v43 = call fn1(v42, v16, v35)\n" ++
  "\n" ++
  "    v44 = ishl_imm v30, 3\n" ++
  "    v45 = iadd v0, v19\n" ++
  "    v46 = iadd v45, v44\n" ++
  "    store.i64 notrap aligned v43, v46\n" ++
  "\n" ++
  "    v47 = iadd_imm v30, 1\n" ++
  "    v48 = icmp ult v47, v9\n" ++
  "    brif v48, block5(v47, v31, v32), block6(v200)\n" ++
  "\n" ++
  -- Join workers
  "block6(v50: i64):\n" ++
  "    v51 = ishl_imm v50, 3\n" ++
  "    v52 = iadd v0, v19\n" ++
  "    v53 = iadd v52, v51\n" ++
  "    v54 = load.i64 notrap aligned v53\n" ++
  "    v55 = iadd_imm v0, 768\n" ++        -- THREAD_CTX_OFF
  "    v56 = call fn2(v55, v54)\n" ++
  "    v57 = iadd_imm v50, 1\n" ++
  "    v58 = icmp ult v57, v9\n" ++
  "    brif v58, block6(v57), block7(v200)\n" ++
  "\n" ++
  -- Merge histograms: outer loop over bins
  "block7(v60: i64):\n" ++
  "    v61 = iconst.i64 0\n" ++
  "    jump block8(v200, v61, v60)\n" ++
  "\n" ++
  -- Merge: inner loop over workers for one bin
  "block8(v70: i64, v71: i64, v72: i64):\n" ++
  "    v73 = imul v70, v13\n" ++
  "    v74 = iadd_imm v73, 1024\n" ++      -- HIST_REGION_OFF
  "    v75 = ishl_imm v72, 3\n" ++
  "    v76 = iadd v74, v75\n" ++
  "    v77 = iadd v0, v76\n" ++
  "    v78 = load.i64 notrap aligned v77\n" ++
  "    v79 = iadd v71, v78\n" ++
  "    v80 = iadd_imm v70, 1\n" ++
  "    v81 = icmp ult v80, v9\n" ++
  "    brif v81, block8(v80, v79, v72), block9(v79, v72)\n" ++
  "\n" ++
  -- Store merged bin count
  "block9(v85: i64, v86: i64):\n" ++
  "    v87 = ishl_imm v86, 3\n" ++
  "    v88 = iadd_imm v87, 17408\n" ++     -- RESULT_OFF
  "    v89 = iadd v0, v88\n" ++
  "    store.i64 notrap aligned v85, v89\n" ++
  "    v90 = iadd_imm v86, 1\n" ++
  "    v91 = icmp ult v90, v12\n" ++
  "    brif v91, block7(v90), block10\n" ++
  "\n" ++
  -- Write result and cleanup
  "block10:\n" ++
  "    v95 = iconst.i64 512\n" ++          -- OUTPUT_PATH_OFF
  "    v96 = iconst.i64 17408\n" ++        -- RESULT_OFF
  "    v97 = iconst.i64 0\n" ++
  "    v98 = iconst.i64 2048\n" ++         -- RESULT_SIZE
  "    v99 = call fn5(v0, v95, v96, v97, v98)\n" ++
  "\n" ++
  "    v100 = iadd_imm v0, 768\n" ++       -- THREAD_CTX_OFF
  "    call fn3(v100)\n" ++
  "    return\n" ++
  "}\n"

-- fn3: worker — 4x unrolled histogram scan
-- Receives pointer to 48-byte descriptor
def clifWorkerFn : String :=
  "function u0:3(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    v1 = load.i64 notrap aligned v0\n" ++           -- base_ptr
  "    v2 = load.i64 notrap aligned v0+8\n" ++         -- data_off
  "    v3 = load.i64 notrap aligned v0+16\n" ++        -- data_start
  "    v4 = load.i64 notrap aligned v0+24\n" ++        -- data_count
  "    v5 = load.i64 notrap aligned v0+32\n" ++        -- hist_off (from base)
  "    v6 = load.i64 notrap aligned v0+40\n" ++        -- bins
  "\n" ++
  "    v7 = iadd v1, v2\n" ++
  "    v8 = ishl_imm v3, 2\n" ++
  "    v9 = iadd v7, v8\n" ++              -- &data[start]
  "\n" ++
  "    v10 = iadd v1, v5\n" ++             -- hist_ptr
  "\n" ++
  "    v11 = ishl_imm v6, 3\n" ++
  "    v12 = iadd v10, v11\n" ++           -- hist_end
  "    v13 = ishl_imm v4, 2\n" ++
  "    v14 = iadd v9, v13\n" ++            -- data_end
  "\n" ++
  "    v15 = band_imm v4, -4\n" ++
  "    v16 = ishl_imm v15, 2\n" ++
  "    v17 = iadd v9, v16\n" ++            -- data_end_4x
  "\n" ++
  "    v20 = iconst.i64 0\n" ++
  "    jump block1(v10)\n" ++
  "\n" ++
  -- Zero histogram (8x unrolled)
  "block1(v21: i64):\n" ++
  "    store.i64 notrap aligned v20, v21\n" ++
  "    store.i64 notrap aligned v20, v21+8\n" ++
  "    store.i64 notrap aligned v20, v21+16\n" ++
  "    store.i64 notrap aligned v20, v21+24\n" ++
  "    store.i64 notrap aligned v20, v21+32\n" ++
  "    store.i64 notrap aligned v20, v21+40\n" ++
  "    store.i64 notrap aligned v20, v21+48\n" ++
  "    store.i64 notrap aligned v20, v21+56\n" ++
  "    v22 = iadd_imm v21, 64\n" ++
  "    v23 = icmp ult v22, v12\n" ++
  "    brif v23, block1(v22), block2(v9)\n" ++
  "\n" ++
  -- 4x unrolled scan
  "block2(v30: i64):\n" ++
  "    v31 = icmp ult v30, v17\n" ++
  "    brif v31, block3(v30), block5(v17)\n" ++
  "\n" ++
  "block3(v40: i64):\n" ++
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
  "    v81 = icmp ult v80, v17\n" ++
  "    brif v81, block3(v80), block5(v17)\n" ++
  "\n" ++
  -- Scalar tail
  "block5(v90: i64):\n" ++
  "    v91 = icmp ult v90, v14\n" ++
  "    brif v91, block6(v90), block7\n" ++
  "\n" ++
  "block6(v100: i64):\n" ++
  "    v101 = uload32.i64 notrap aligned v100\n" ++
  "    v102 = ishl_imm v101, 3\n" ++
  "    v103 = iadd v10, v102\n" ++
  "    v104 = load.i64 notrap aligned v103\n" ++
  "    v105 = iadd_imm v104, 1\n" ++
  "    store.i64 notrap aligned v105, v103\n" ++
  "    v106 = iadd_imm v100, 4\n" ++
  "    v107 = icmp ult v106, v14\n" ++
  "    brif v107, block6(v106), block7\n" ++
  "\n" ++
  "block7:\n" ++
  "    return\n" ++
  "}\n"

def clifIR : String :=
  clifNoopFn0 ++ "\n" ++ clifNoopFn1 ++ "\n" ++ clifOrchestratorFn ++ "\n" ++ clifWorkerFn

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

end HistogramBench4

def main : IO Unit := do
  let json := toJsonPair HistogramBench4.buildConfig HistogramBench4.buildAlgorithm
  IO.println (Json.compress json)
