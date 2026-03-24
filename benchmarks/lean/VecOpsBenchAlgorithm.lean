import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace VecOpsBench

/-
  SIMD f32 vec_add benchmark.

  Payload (via execute data arg):
    [A floats: n * f32][B floats: n * f32]
    n is derived from data_len / 8 (two f32 arrays back to back)

  Output (via execute_into out arg):
    [0..8)  result (f64) — sum of all (A[i] + B[i])

  Memory layout (shared memory):
    0x0000..0x0027  reserved (runtime writes ctx_ptr, data_ptr, data_len, out_ptr, out_len)

  The CLIF code reads arrays directly from the data pointer (zero copy)
  and writes the result to the out pointer (zero copy).

  CLIF: 4x-unrolled SIMD vec_add of two f32 arrays → f64 sum.
  Main loop: 4 independent f32x4 accumulators (16 floats/iter) for ILP.
-/

def TIMEOUT_MS : Nat := 30000

-- fn0: noop
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: vec_add — reads two f32 arrays from data pointer,
--                writes f64 sum to out pointer
def clifVecAddFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  -- Load data_ptr, data_len, out_ptr from reserved region
  "  v500 = load.i64 notrap aligned v0+0x08\n" ++  -- data_ptr
  "  v501 = load.i64 notrap aligned v0+0x10\n" ++  -- data_len
  "  v502 = load.i64 notrap aligned v0+0x18\n" ++  -- out_ptr
  -- n = data_len / 8 (two f32 arrays)
  "  v2 = ushr_imm v501, 3\n" ++                    -- n
  -- A_ptr = v500 (data_ptr), B_ptr = data_ptr + n*4
  "  v5 = ishl_imm v2, 2\n" ++                      -- n*4
  "  v6 = iadd v500, v5\n" ++                        -- B_ptr
  -- Compute loop bounds
  "  v7 = ushr_imm v2, 4\n" ++                      -- n/16 (main loop iters)
  "  v8 = ishl_imm v7, 6\n" ++                      -- n/16 * 64 (main loop bytes)
  "  v9 = ushr_imm v2, 2\n" ++                      -- n/4
  "  v10 = ishl_imm v9, 4\n" ++                     -- n/4 * 16 (simd cleanup bytes)
  "  v11 = ishl_imm v2, 2\n" ++                     -- n*4 (total bytes)
  "  v12 = f32const 0.0\n" ++
  "  v13 = splat.f32x4 v12\n" ++
  "  v14 = iconst.i64 0\n" ++
  "  jump block1(v14, v13, v13, v13, v13)\n" ++
  "\n" ++
  -- Main loop: 4x unrolled (16 floats/iter, 4 accumulators)
  "block1(v20: i64, v21: f32x4, v22: f32x4, v23: f32x4, v24: f32x4):\n" ++
  "  v25 = icmp sge v20, v8\n" ++
  "  brif v25, block3(v20, v21, v22, v23, v24), block2(v20, v21, v22, v23, v24)\n" ++
  "\n" ++
  "block2(v30: i64, v31: f32x4, v32: f32x4, v33: f32x4, v34: f32x4):\n" ++
  "  v35 = iadd v500, v30\n" ++
  "  v36 = iadd v6, v30\n" ++
  "  v37 = load.f32x4 notrap aligned v35\n" ++
  "  v38 = load.f32x4 notrap aligned v36\n" ++
  "  v39 = fadd v37, v38\n" ++
  "  v40 = fadd v31, v39\n" ++
  "  v41 = iadd_imm v35, 16\n" ++
  "  v42 = iadd_imm v36, 16\n" ++
  "  v43 = load.f32x4 notrap aligned v41\n" ++
  "  v44 = load.f32x4 notrap aligned v42\n" ++
  "  v45 = fadd v43, v44\n" ++
  "  v46 = fadd v32, v45\n" ++
  "  v47 = iadd_imm v35, 32\n" ++
  "  v48 = iadd_imm v36, 32\n" ++
  "  v49 = load.f32x4 notrap aligned v47\n" ++
  "  v50 = load.f32x4 notrap aligned v48\n" ++
  "  v51 = fadd v49, v50\n" ++
  "  v52 = fadd v33, v51\n" ++
  "  v53 = iadd_imm v35, 48\n" ++
  "  v54 = iadd_imm v36, 48\n" ++
  "  v55 = load.f32x4 notrap aligned v53\n" ++
  "  v56 = load.f32x4 notrap aligned v54\n" ++
  "  v57 = fadd v55, v56\n" ++
  "  v58 = fadd v34, v57\n" ++
  "  v59 = iadd_imm v30, 64\n" ++
  "  jump block1(v59, v40, v46, v52, v58)\n" ++
  "\n" ++
  -- Merge 4 accumulators → 1
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
  "  v83 = iadd v6, v80\n" ++
  "  v84 = load.f32x4 notrap aligned v82\n" ++
  "  v85 = load.f32x4 notrap aligned v83\n" ++
  "  v86 = fadd v84, v85\n" ++
  "  v87 = fadd v81, v86\n" ++
  "  v88 = iadd_imm v80, 16\n" ++
  "  jump block4(v88, v87)\n" ++
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
  -- Scalar tail
  "block7(v110: i64, v111: f64):\n" ++
  "  v112 = icmp sge v110, v11\n" ++
  "  brif v112, block9(v111), block8(v110, v111)\n" ++
  "\n" ++
  "block8(v120: i64, v121: f64):\n" ++
  "  v122 = iadd v500, v120\n" ++
  "  v123 = iadd v6, v120\n" ++
  "  v124 = load.f32 notrap aligned v122\n" ++
  "  v125 = load.f32 notrap aligned v123\n" ++
  "  v126 = fadd v124, v125\n" ++
  "  v127 = fpromote.f64 v126\n" ++
  "  v128 = fadd v121, v127\n" ++
  "  v129 = iadd_imm v120, 4\n" ++
  "  jump block7(v129, v128)\n" ++
  "\n" ++
  -- Store result to out_ptr
  "block9(v130: f64):\n" ++
  "  store.f64 notrap aligned v130, v502\n" ++
  "  return\n" ++
  "}"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifVecAddFn

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

end VecOpsBench

def main : IO Unit := do
  let json := toJsonPair VecOpsBench.buildConfig VecOpsBench.buildAlgorithm
  IO.println (Json.compress json)
