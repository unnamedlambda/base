import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace ReductionBench

/-
  SIMD f32 sum reduction benchmark.

  Payload (via execute data arg):
    [f32 values: n floats]
    n is derived from data_len / 4

  Output (via execute_into out arg):
    [0..8)  result (f64) — sum of all elements

  Memory layout (shared memory):
    0x0000..0x0027  reserved (runtime writes ctx_ptr, data_ptr, data_len, out_ptr, out_len)

  The CLIF code reads the array directly from the data pointer (zero copy)
  and writes the result to the out pointer (zero copy).

  CLIF: 4x-unrolled SIMD sum of f32 array → f64 result.
  Main loop: 4 independent f32x4 accumulators (16 floats/iter) for ILP.
-/

def TIMEOUT_MS : Nat := 30000

-- fn0: noop
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: sum reduction — reads f32 array from data pointer,
--                       writes f64 sum to out pointer
def clifSumFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  -- Load data_ptr, data_len, out_ptr from reserved region
  "  v500 = load.i64 notrap aligned v0+0x08\n" ++  -- data_ptr
  "  v501 = load.i64 notrap aligned v0+0x10\n" ++  -- data_len
  "  v502 = load.i64 notrap aligned v0+0x18\n" ++  -- out_ptr
  -- n = data_len / 4
  "  v2 = ushr_imm v501, 2\n" ++                    -- n (element count)
  -- Compute loop bounds (byte offsets)
  "  v5 = ushr_imm v2, 4\n" ++                      -- n/16
  "  v6 = ishl_imm v5, 6\n" ++                      -- n/16 * 64 (main loop bytes)
  "  v7 = ushr_imm v2, 2\n" ++                      -- n/4
  "  v8 = ishl_imm v7, 4\n" ++                      -- n/4 * 16 (simd cleanup bytes)
  "  v9 = ishl_imm v2, 2\n" ++                      -- n*4 (total bytes)
  "  v10 = f32const 0.0\n" ++
  "  v11 = splat.f32x4 v10\n" ++
  "  v12 = iconst.i64 0\n" ++
  "  jump block1(v12, v11, v11, v11, v11)\n" ++
  "\n" ++
  -- Main loop: 4x unrolled (16 floats/iter, 4 accumulators)
  "block1(v20: i64, v21: f32x4, v22: f32x4, v23: f32x4, v24: f32x4):\n" ++
  "  v25 = icmp sge v20, v6\n" ++
  "  brif v25, block3(v20, v21, v22, v23, v24), block2(v20, v21, v22, v23, v24)\n" ++
  "\n" ++
  "block2(v30: i64, v31: f32x4, v32: f32x4, v33: f32x4, v34: f32x4):\n" ++
  "  v35 = iadd v500, v30\n" ++
  "  v36 = load.f32x4 notrap aligned v35\n" ++
  "  v37 = iadd_imm v35, 16\n" ++
  "  v38 = load.f32x4 notrap aligned v37\n" ++
  "  v39 = iadd_imm v35, 32\n" ++
  "  v40 = load.f32x4 notrap aligned v39\n" ++
  "  v41 = iadd_imm v35, 48\n" ++
  "  v42 = load.f32x4 notrap aligned v41\n" ++
  "  v43 = fadd v31, v36\n" ++
  "  v44 = fadd v32, v38\n" ++
  "  v45 = fadd v33, v40\n" ++
  "  v46 = fadd v34, v42\n" ++
  "  v47 = iadd_imm v30, 64\n" ++
  "  jump block1(v47, v43, v44, v45, v46)\n" ++
  "\n" ++
  -- Merge 4 accumulators → 1
  "block3(v50: i64, v51: f32x4, v52: f32x4, v53: f32x4, v54: f32x4):\n" ++
  "  v55 = fadd v51, v52\n" ++
  "  v56 = fadd v53, v54\n" ++
  "  v57 = fadd v55, v56\n" ++
  "  jump block4(v50, v57)\n" ++
  "\n" ++
  -- SIMD cleanup: 1 vector at a time
  "block4(v60: i64, v61: f32x4):\n" ++
  "  v62 = icmp sge v60, v8\n" ++
  "  brif v62, block6(v60, v61), block5(v60, v61)\n" ++
  "\n" ++
  "block5(v70: i64, v71: f32x4):\n" ++
  "  v72 = iadd v500, v70\n" ++
  "  v73 = load.f32x4 notrap aligned v72\n" ++
  "  v74 = fadd v71, v73\n" ++
  "  v75 = iadd_imm v70, 16\n" ++
  "  jump block4(v75, v74)\n" ++
  "\n" ++
  -- Horizontal reduce f32x4 → f64
  "block6(v80: i64, v81: f32x4):\n" ++
  "  v82 = extractlane v81, 0\n" ++
  "  v83 = extractlane v81, 1\n" ++
  "  v84 = extractlane v81, 2\n" ++
  "  v85 = extractlane v81, 3\n" ++
  "  v86 = fpromote.f64 v82\n" ++
  "  v87 = fpromote.f64 v83\n" ++
  "  v88 = fpromote.f64 v84\n" ++
  "  v89 = fpromote.f64 v85\n" ++
  "  v90 = fadd v86, v87\n" ++
  "  v91 = fadd v88, v89\n" ++
  "  v92 = fadd v90, v91\n" ++
  "  jump block7(v80, v92)\n" ++
  "\n" ++
  -- Scalar tail
  "block7(v100: i64, v101: f64):\n" ++
  "  v102 = icmp sge v100, v9\n" ++
  "  brif v102, block9(v101), block8(v100, v101)\n" ++
  "\n" ++
  "block8(v110: i64, v111: f64):\n" ++
  "  v112 = iadd v500, v110\n" ++
  "  v113 = load.f32 notrap aligned v112\n" ++
  "  v114 = fpromote.f64 v113\n" ++
  "  v115 = fadd v111, v114\n" ++
  "  v116 = iadd_imm v110, 4\n" ++
  "  jump block7(v116, v115)\n" ++
  "\n" ++
  -- Store result to out_ptr
  "block9(v120: f64):\n" ++
  "  store.f64 notrap aligned v120, v502\n" ++
  "  return\n" ++
  "}"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifSumFn

def controlActions : List Action :=
  [{ kind := .ClifCall, dst := 0, src := 1, offset := 0, size := 0 }]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := 40,      -- only reserved region needed
  context_offset := 0
}

def buildAlgorithm : Algorithm := {
  actions := controlActions,
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

end ReductionBench

def main : IO Unit := do
  let json := toJsonPair ReductionBench.buildConfig ReductionBench.buildAlgorithm
  IO.println (Json.compress json)
