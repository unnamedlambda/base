import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace RowAffineReduceBench

/-!
  Input payload:
    [rows:u64][cols:u64][X: rows*cols*f32][scale: cols*f32][bias: cols*f32]

  Output:
    [y: rows*f32] where y[i] = sum_j (X[i,j] * scale[j] + bias[j])

  This is a broadcasted row-wise affine transform plus reduction, intended as
  a more NumPy-like tensor workload than the original 1D vector kernels.
-/

def MEM_SIZE : Nat := 40
def TIMEOUT_MS : Nat := 30000

def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

def clifRowAffineReduceFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "  v500 = load.i64 notrap aligned v0+0x08\n" ++
  "  v501 = load.i64 notrap aligned v0+0x18\n" ++
  "  v1 = load.i64 notrap aligned v500\n" ++
  "  v2 = load.i64 notrap aligned v500+8\n" ++
  "  v3 = iadd_imm v500, 16\n" ++
  "  v4 = imul v1, v2\n" ++
  "  v5 = ishl_imm v4, 2\n" ++
  "  v6 = iadd v3, v5\n" ++
  "  v7 = ishl_imm v2, 2\n" ++
  "  v8 = iadd v6, v7\n" ++
  "  v9 = iconst.i64 0\n" ++
  "  jump block1(v9)\n" ++
  "\n" ++
  "block1(v10: i64):\n" ++
  "  v11 = icmp sge v10, v1\n" ++
  "  brif v11, block5, block2(v10)\n" ++
  "\n" ++
  "block2(v20: i64):\n" ++
  "  v21 = imul v20, v2\n" ++
  "  v22 = ishl_imm v21, 2\n" ++
  "  v23 = iadd v3, v22\n" ++
  "  v24 = ushr_imm v2, 4\n" ++
  "  v25 = ishl_imm v24, 6\n" ++
  "  v26 = ushr_imm v2, 2\n" ++
  "  v27 = ishl_imm v26, 4\n" ++
  "  v28 = ishl_imm v2, 2\n" ++
  "  v29 = f32const 0.0\n" ++
  "  v30 = splat.f32x4 v29\n" ++
  "  v31 = iconst.i64 0\n" ++
  "  jump block3(v20, v23, v31, v25, v27, v28, v30, v30, v30, v30)\n" ++
  "\n" ++
  "block3(v40: i64, v41: i64, v42: i64, v43: i64, v44: i64, v45: i64, v46: f32x4, v47: f32x4, v48: f32x4, v49: f32x4):\n" ++
  "  v50 = icmp sge v42, v43\n" ++
  "  brif v50, block4(v40, v41, v42, v43, v44, v45, v46, v47, v48, v49), block6(v40, v41, v42, v43, v44, v45, v46, v47, v48, v49)\n" ++
  "\n" ++
  "block6(v60: i64, v61: i64, v62: i64, v63: i64, v64: i64, v65: i64, v66: f32x4, v67: f32x4, v68: f32x4, v69: f32x4):\n" ++
  "  v70 = iadd v61, v62\n" ++
  "  v71 = iadd v6, v62\n" ++
  "  v72 = iadd v8, v62\n" ++
  "  v73 = load.f32x4 notrap aligned v70\n" ++
  "  v74 = load.f32x4 notrap aligned v71\n" ++
  "  v75 = load.f32x4 notrap aligned v72\n" ++
  "  v76 = fmul v73, v74\n" ++
  "  v77 = fadd v76, v75\n" ++
  "  v78 = fadd v66, v77\n" ++
  "  v79 = iadd_imm v70, 16\n" ++
  "  v80 = iadd_imm v71, 16\n" ++
  "  v81 = iadd_imm v72, 16\n" ++
  "  v82 = load.f32x4 notrap aligned v79\n" ++
  "  v83 = load.f32x4 notrap aligned v80\n" ++
  "  v84 = load.f32x4 notrap aligned v81\n" ++
  "  v85 = fmul v82, v83\n" ++
  "  v86 = fadd v85, v84\n" ++
  "  v87 = fadd v67, v86\n" ++
  "  v88 = iadd_imm v70, 32\n" ++
  "  v89 = iadd_imm v71, 32\n" ++
  "  v90 = iadd_imm v72, 32\n" ++
  "  v91 = load.f32x4 notrap aligned v88\n" ++
  "  v92 = load.f32x4 notrap aligned v89\n" ++
  "  v93 = load.f32x4 notrap aligned v90\n" ++
  "  v94 = fmul v91, v92\n" ++
  "  v95 = fadd v94, v93\n" ++
  "  v96 = fadd v68, v95\n" ++
  "  v97 = iadd_imm v70, 48\n" ++
  "  v98 = iadd_imm v71, 48\n" ++
  "  v99 = iadd_imm v72, 48\n" ++
  "  v100 = load.f32x4 notrap aligned v97\n" ++
  "  v101 = load.f32x4 notrap aligned v98\n" ++
  "  v102 = load.f32x4 notrap aligned v99\n" ++
  "  v103 = fmul v100, v101\n" ++
  "  v104 = fadd v103, v102\n" ++
  "  v105 = fadd v69, v104\n" ++
  "  v106 = iadd_imm v62, 64\n" ++
  "  jump block3(v60, v61, v106, v63, v64, v65, v78, v87, v96, v105)\n" ++
  "\n" ++
  "block4(v110: i64, v111: i64, v112: i64, v113: i64, v114: i64, v115: i64, v116: f32x4, v117: f32x4, v118: f32x4, v119: f32x4):\n" ++
  "  v120 = fadd v116, v117\n" ++
  "  v121 = fadd v118, v119\n" ++
  "  v122 = fadd v120, v121\n" ++
  "  jump block7(v110, v111, v112, v114, v115, v122)\n" ++
  "\n" ++
  "block7(v130: i64, v131: i64, v132: i64, v133: i64, v134: i64, v135: f32x4):\n" ++
  "  v136 = icmp sge v132, v133\n" ++
  "  brif v136, block8(v130, v131, v132, v134, v135), block9(v130, v131, v132, v133, v134, v135)\n" ++
  "\n" ++
  "block9(v140: i64, v141: i64, v142: i64, v143: i64, v144: i64, v145: f32x4):\n" ++
  "  v146 = iadd v141, v142\n" ++
  "  v147 = iadd v6, v142\n" ++
  "  v148 = iadd v8, v142\n" ++
  "  v149 = load.f32x4 notrap aligned v146\n" ++
  "  v150 = load.f32x4 notrap aligned v147\n" ++
  "  v151 = load.f32x4 notrap aligned v148\n" ++
  "  v152 = fmul v149, v150\n" ++
  "  v153 = fadd v152, v151\n" ++
  "  v154 = fadd v145, v153\n" ++
  "  v155 = iadd_imm v142, 16\n" ++
  "  jump block7(v140, v141, v155, v143, v144, v154)\n" ++
  "\n" ++
  "block8(v160: i64, v161: i64, v162: i64, v163: i64, v164: f32x4):\n" ++
  "  v165 = extractlane v164, 0\n" ++
  "  v166 = extractlane v164, 1\n" ++
  "  v167 = extractlane v164, 2\n" ++
  "  v168 = extractlane v164, 3\n" ++
  "  v169 = fadd v165, v166\n" ++
  "  v170 = fadd v167, v168\n" ++
  "  v171 = fadd v169, v170\n" ++
  "  jump block10(v160, v161, v162, v163, v171)\n" ++
  "\n" ++
  "block10(v180: i64, v181: i64, v182: i64, v183: i64, v184: f32):\n" ++
  "  v185 = icmp sge v182, v183\n" ++
  "  brif v185, block11(v180, v184), block12(v180, v181, v182, v183, v184)\n" ++
  "\n" ++
  "block12(v190: i64, v191: i64, v192: i64, v193: i64, v194: f32):\n" ++
  "  v195 = iadd v191, v192\n" ++
  "  v196 = iadd v6, v192\n" ++
  "  v197 = iadd v8, v192\n" ++
  "  v198 = load.f32 notrap aligned v195\n" ++
  "  v199 = load.f32 notrap aligned v196\n" ++
  "  v200 = load.f32 notrap aligned v197\n" ++
  "  v201 = fmul v198, v199\n" ++
  "  v202 = fadd v201, v200\n" ++
  "  v203 = fadd v194, v202\n" ++
  "  v204 = iadd_imm v192, 4\n" ++
  "  jump block10(v190, v191, v204, v193, v203)\n" ++
  "\n" ++
  "block11(v210: i64, v211: f32):\n" ++
  "  v212 = ishl_imm v210, 2\n" ++
  "  v213 = iadd v501, v212\n" ++
  "  store.f32 notrap aligned v211, v213\n" ++
  "  v214 = iadd_imm v210, 1\n" ++
  "  jump block1(v214)\n" ++
  "\n" ++
  "block5:\n" ++
  "  return\n" ++
  "}\n"

def clifIR : String := clifNoopFn ++ "\n" ++ clifRowAffineReduceFn

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
  timeout_ms := some TIMEOUT_MS
}

end RowAffineReduceBench

def main : IO Unit := do
  let json := toJsonPair RowAffineReduceBench.buildConfig RowAffineReduceBench.buildAlgorithm
  IO.println (Json.compress json)
