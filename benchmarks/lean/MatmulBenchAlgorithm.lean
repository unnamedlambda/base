import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace MatmulBench

/-
  SIMD f32 matrix multiplication benchmark.

  Payload (via execute data arg):
    [0..4)   M (u32) — rows of A
    [4..8)   K (u32) — cols of A / rows of B
    [8..12)  N (u32) — cols of B
    [12..)   matrix A (M*K f32), then matrix B (K*N f32)

  Output (via execute_into out arg):
    [0..8)   result (f64) — sum of all elements of C

  Memory layout (shared memory):
    0x0000..0x0027  reserved (runtime writes ctx_ptr, data_ptr, data_len, out_ptr, out_len)

  The CLIF code reads matrices directly from the data pointer (zero copy)
  and writes the result to the out pointer (zero copy).

  CLIF: SIMD matmul with 4x-unrolled j-loop (16 cols/iter).
  Inner loop uses 4 independent f32x4 accumulators for ILP.
-/

def TIMEOUT_MS : Nat := 60000

-- fn0: noop
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: matmul — reads M,K,N and matrix data from data pointer,
--               writes f64 sum to out pointer
def clifMatmulFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  -- Load data_ptr and out_ptr from reserved region
  "  v500 = load.i64 notrap aligned v0+0x08\n" ++  -- data_ptr
  "  v501 = load.i64 notrap aligned v0+0x18\n" ++  -- out_ptr
  -- Parse M, K, N from payload header
  "  v1 = load.i32 notrap aligned v500\n" ++        -- M
  "  v2 = load.i32 notrap aligned v500+4\n" ++      -- K
  "  v3 = load.i32 notrap aligned v500+8\n" ++      -- N
  "  v4 = sextend.i64 v1\n" ++                      -- M i64
  "  v5 = sextend.i64 v2\n" ++                      -- K i64
  "  v6 = sextend.i64 v3\n" ++                      -- N i64
  -- Compute pointers: A starts at data_ptr+12, B starts at A + M*K*4
  "  v7 = iadd_imm v500, 12\n" ++                   -- A_ptr
  "  v8 = imul v4, v5\n" ++                         -- M*K
  "  v9 = ishl_imm v8, 2\n" ++                      -- M*K*4
  "  v10 = iadd v7, v9\n" ++                        -- B_ptr
  "  v13 = ishl_imm v6, 2\n" ++                     -- N*4 (row stride of B in bytes)
  "  v14 = iconst.i64 0\n" ++
  "  v15 = f64const 0.0\n" ++
  "  jump block1(v14, v15)\n" ++
  "\n" ++
  -- Outer loop: for i in 0..M
  "block1(v20: i64, v21: f64):\n" ++
  "  v22 = icmp sge v20, v4\n" ++
  "  brif v22, block9(v21), block2(v20, v21)\n" ++
  "\n" ++
  -- Middle loop setup: for p in 0..K
  "block2(v23: i64, v24: f64):\n" ++
  "  v25 = iconst.i64 0\n" ++
  "  jump block3(v23, v24, v25)\n" ++
  "\n" ++
  "block3(v30: i64, v31: f64, v32: i64):\n" ++
  "  v33 = icmp sge v32, v5\n" ++
  "  brif v33, block8(v30, v31), block4(v30, v31, v32)\n" ++
  "\n" ++
  -- Load A[i,p], compute B row pointer, init accumulators
  "block4(v34: i64, v35: f64, v36: i64):\n" ++
  "  v37 = imul v34, v5\n" ++                       -- i*K
  "  v38 = iadd v37, v36\n" ++                      -- i*K+p
  "  v39 = ishl_imm v38, 2\n" ++                    -- (i*K+p)*4
  "  v40 = iadd v7, v39\n" ++                       -- &A[i,p]
  "  v41 = load.f32 notrap aligned v40\n" ++
  "  v42 = splat.f32x4 v41\n" ++                    -- broadcast A[i,p]
  "  v43 = imul v36, v6\n" ++                       -- p*N
  "  v44 = ishl_imm v43, 2\n" ++                    -- p*N*4
  "  v45 = iadd v10, v44\n" ++                      -- &B[p,0]
  "  v46 = f32const 0.0\n" ++
  "  v47 = splat.f32x4 v46\n" ++
  "  v48 = iconst.i64 0\n" ++
  "  jump block5(v34, v35, v36, v42, v45, v48, v47, v47, v47, v47)\n" ++
  "\n" ++
  -- Inner loop: 4x unrolled j-loop (16 cols/iter)
  "block5(v50: i64, v51: f64, v52: i64, v53: f32x4, v54: i64, v55: i64, v56: f32x4, v57: f32x4, v58: f32x4, v59: f32x4):\n" ++
  "  v60 = icmp sge v55, v13\n" ++
  "  brif v60, block7(v50, v51, v52, v56, v57, v58, v59), block6(v50, v51, v52, v53, v54, v55, v56, v57, v58, v59)\n" ++
  "\n" ++
  "block6(v70: i64, v71: f64, v72: i64, v73: f32x4, v74: i64, v75: i64, v76: f32x4, v77: f32x4, v78: f32x4, v79: f32x4):\n" ++
  "  v80 = iadd v74, v75\n" ++
  "  v81 = load.f32x4 notrap aligned v80\n" ++
  "  v82 = fmul v73, v81\n" ++
  "  v83 = fadd v76, v82\n" ++
  "  v84 = iadd_imm v80, 16\n" ++
  "  v85 = load.f32x4 notrap aligned v84\n" ++
  "  v86 = fmul v73, v85\n" ++
  "  v87 = fadd v77, v86\n" ++
  "  v88 = iadd_imm v80, 32\n" ++
  "  v89 = load.f32x4 notrap aligned v88\n" ++
  "  v90 = fmul v73, v89\n" ++
  "  v91 = fadd v78, v90\n" ++
  "  v92 = iadd_imm v80, 48\n" ++
  "  v93 = load.f32x4 notrap aligned v92\n" ++
  "  v94 = fmul v73, v93\n" ++
  "  v95 = fadd v79, v94\n" ++
  "  v96 = iadd_imm v75, 64\n" ++
  "  jump block5(v70, v71, v72, v73, v74, v96, v83, v87, v91, v95)\n" ++
  "\n" ++
  -- Reduce 4 accumulators → single f64, add to total
  "block7(v100: i64, v101: f64, v102: i64, v103: f32x4, v104: f32x4, v105: f32x4, v106: f32x4):\n" ++
  "  v107 = fadd v103, v104\n" ++
  "  v108 = fadd v105, v106\n" ++
  "  v109 = fadd v107, v108\n" ++
  "  v110 = extractlane v109, 0\n" ++
  "  v111 = extractlane v109, 1\n" ++
  "  v112 = extractlane v109, 2\n" ++
  "  v113 = extractlane v109, 3\n" ++
  "  v114 = fpromote.f64 v110\n" ++
  "  v115 = fpromote.f64 v111\n" ++
  "  v116 = fpromote.f64 v112\n" ++
  "  v117 = fpromote.f64 v113\n" ++
  "  v118 = fadd v114, v115\n" ++
  "  v119 = fadd v116, v117\n" ++
  "  v120 = fadd v118, v119\n" ++
  "  v121 = fadd v101, v120\n" ++
  "  v122 = iadd_imm v102, 1\n" ++
  "  jump block3(v100, v121, v122)\n" ++
  "\n" ++
  -- Increment i
  "block8(v130: i64, v131: f64):\n" ++
  "  v132 = iadd_imm v130, 1\n" ++
  "  jump block1(v132, v131)\n" ++
  "\n" ++
  -- Store result to out_ptr
  "block9(v140: f64):\n" ++
  "  store.f64 notrap aligned v140, v501\n" ++
  "  return\n" ++
  "}"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifMatmulFn

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

end MatmulBench

def main : IO Unit := do
  let json := toJsonPair MatmulBench.buildConfig MatmulBench.buildAlgorithm
  IO.println (Json.compress json)
