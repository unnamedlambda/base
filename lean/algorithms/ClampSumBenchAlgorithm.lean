import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace ClampSumBench

/-
  Input: n f32 values
  lo = -0.5, hi = 0.5  (hard-coded)
  Result: f64 — sum of clamp(x, lo, hi) for all x

  CLIF: 4x-unrolled f32x4 clamp + accumulate, then horizontal reduce to f64.
-/

def MEM_SIZE : Nat := 40    -- reserved region only

def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

def clifClampSumFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "  v500 = load.i64 notrap aligned v0+0x08\n" ++   -- data_ptr
  "  v501 = load.i64 notrap aligned v0+0x10\n" ++   -- data_len (bytes)
  "  v502 = load.i64 notrap aligned v0+0x18\n" ++   -- out_ptr
  "  v2 = ushr_imm v501, 2\n" ++                    -- n = data_len / 4
  "  v7 = ushr_imm v2, 4\n" ++                      -- n/16 (main loop iters)
  "  v8 = ishl_imm v7, 6\n" ++                      -- n/16 * 64 (main loop byte bound)
  "  v9 = ushr_imm v2, 2\n" ++                      -- n/4
  "  v10 = ishl_imm v9, 4\n" ++                     -- n/4 * 16 (simd cleanup byte bound)
  "  v11 = ishl_imm v2, 2\n" ++                     -- n*4 (scalar tail byte bound)
  "  v600 = f32const 0x1.000000p-1\n" ++            -- hi = 0.5
  "  v601 = fneg v600\n" ++                         -- lo = -0.5
  "  v602 = splat.f32x4 v600\n" ++                  -- hi broadcast
  "  v603 = splat.f32x4 v601\n" ++                  -- lo broadcast
  "  v604 = f32const 0.0\n" ++
  "  v605 = splat.f32x4 v604\n" ++                  -- zero accumulators
  "  v606 = iconst.i64 0\n" ++
  "  jump block1(v606, v605, v605, v605, v605)\n" ++
  "\n" ++
  -- Main loop: 4x unrolled (16 floats/iter, 4 independent f32x4 accumulators)
  "block1(v20: i64, v21: f32x4, v22: f32x4, v23: f32x4, v24: f32x4):\n" ++
  "  v25 = icmp sge v20, v8\n" ++
  "  brif v25, block3(v20, v21, v22, v23, v24), block2(v20, v21, v22, v23, v24)\n" ++
  "\n" ++
  "block2(v30: i64, v31: f32x4, v32: f32x4, v33: f32x4, v34: f32x4):\n" ++
  "  v35 = iadd v500, v30\n" ++
  "  v36 = load.f32x4 notrap aligned v35\n" ++
  "  v37 = fmin v36, v602\n" ++
  "  v38 = fmax v37, v603\n" ++
  "  v39 = fadd v31, v38\n" ++
  "  v40 = iadd_imm v35, 16\n" ++
  "  v41 = load.f32x4 notrap aligned v40\n" ++
  "  v42 = fmin v41, v602\n" ++
  "  v43 = fmax v42, v603\n" ++
  "  v44 = fadd v32, v43\n" ++
  "  v45 = iadd_imm v35, 32\n" ++
  "  v46 = load.f32x4 notrap aligned v45\n" ++
  "  v47 = fmin v46, v602\n" ++
  "  v48 = fmax v47, v603\n" ++
  "  v49 = fadd v33, v48\n" ++
  "  v50 = iadd_imm v35, 48\n" ++
  "  v51 = load.f32x4 notrap aligned v50\n" ++
  "  v52 = fmin v51, v602\n" ++
  "  v53 = fmax v52, v603\n" ++
  "  v54 = fadd v34, v53\n" ++
  "  v55 = iadd_imm v30, 64\n" ++
  "  jump block1(v55, v39, v44, v49, v54)\n" ++
  "\n" ++
  -- Merge 4 f32x4 accumulators → 1
  "block3(v60: i64, v61: f32x4, v62: f32x4, v63: f32x4, v64: f32x4):\n" ++
  "  v65 = fadd v61, v62\n" ++
  "  v66 = fadd v63, v64\n" ++
  "  v67 = fadd v65, v66\n" ++
  "  jump block4(v60, v67)\n" ++
  "\n" ++
  -- SIMD cleanup: 1 vector at a time
  "block4(v70: i64, v71: f32x4):\n" ++
  "  v72 = icmp sge v70, v10\n" ++
  "  brif v72, block6(v70, v71), block5(v70, v71)\n" ++
  "\n" ++
  "block5(v80: i64, v81: f32x4):\n" ++
  "  v82 = iadd v500, v80\n" ++
  "  v83 = load.f32x4 notrap aligned v82\n" ++
  "  v84 = fmin v83, v602\n" ++
  "  v85 = fmax v84, v603\n" ++
  "  v86 = fadd v81, v85\n" ++
  "  v87 = iadd_imm v80, 16\n" ++
  "  jump block4(v87, v86)\n" ++
  "\n" ++
  -- Horizontal reduce f32x4 → f64
  "block6(v90: i64, v91: f32x4):\n" ++
  "  v92 = extractlane v91, 0\n" ++
  "  v93 = extractlane v91, 1\n" ++
  "  v94 = extractlane v91, 2\n" ++
  "  v95 = extractlane v91, 3\n" ++
  "  v96 = fpromote.f64 v92\n" ++
  "  v97 = fpromote.f64 v93\n" ++
  "  v98 = fpromote.f64 v94\n" ++
  "  v99 = fpromote.f64 v95\n" ++
  "  v100 = fadd v96, v97\n" ++
  "  v101 = fadd v98, v99\n" ++
  "  v102 = fadd v100, v101\n" ++
  "  jump block7(v90, v102)\n" ++
  "\n" ++
  -- Scalar tail (remainder after SIMD)
  "block7(v110: i64, v111: f64):\n" ++
  "  v112 = icmp sge v110, v11\n" ++
  "  brif v112, block9(v111), block8(v110, v111)\n" ++
  "\n" ++
  "block8(v120: i64, v121: f64):\n" ++
  "  v122 = iadd v500, v120\n" ++
  "  v123 = load.f32 notrap aligned v122\n" ++
  "  v124 = fmin v123, v600\n" ++
  "  v125 = fmax v124, v601\n" ++
  "  v126 = fpromote.f64 v125\n" ++
  "  v127 = fadd v121, v126\n" ++
  "  v128 = iadd_imm v120, 4\n" ++
  "  jump block7(v128, v127)\n" ++
  "\n" ++
  "block9(v130: f64):\n" ++
  "  store.f64 notrap aligned v130, v502\n" ++
  "  return\n" ++
  "}"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifClampSumFn

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
  timeout_ms := some 30000
}

end ClampSumBench

def main : IO Unit := do
  let json := toJsonPair ClampSumBench.buildConfig ClampSumBench.buildAlgorithm
  IO.println (Json.compress json)
