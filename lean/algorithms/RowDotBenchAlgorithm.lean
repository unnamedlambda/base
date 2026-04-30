import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace RowDotBench

/-!
  Input payload:
    [rows:u64][cols:u64][X: rows*cols*f32][w: cols*f32]

  Output:
    [y: rows*f32] where y[i] = dot(X[i, :], w)

  This is a simple row-wise matrix-vector multiply meant to extend the
  NumPy-style benchmark family toward shaped array workloads.
-/

def MEM_SIZE : Nat := 40
def TIMEOUT_MS : Nat := 30000

def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

def clifRowDotFn : String :=
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
  "  v8 = ushr_imm v2, 2\n" ++
  "  v9 = ishl_imm v8, 4\n" ++
  "  v10 = iconst.i64 0\n" ++
  "  jump block1(v10)\n" ++
  "\n" ++
  "block1(v20: i64):\n" ++
  "  v21 = icmp sge v20, v1\n" ++
  "  brif v21, block10, block2(v20)\n" ++
  "\n" ++
  "block2(v30: i64):\n" ++
  "  v31 = iadd_imm v30, 1\n" ++
  "  v32 = icmp slt v31, v1\n" ++
  "  brif v32, block6(v30), block3(v30)\n" ++
  "\n" ++
  "block3(v40: i64):\n" ++
  "  v41 = imul v40, v2\n" ++
  "  v42 = ishl_imm v41, 2\n" ++
  "  v43 = iadd v3, v42\n" ++
  "  v44 = f32const 0.0\n" ++
  "  v45 = splat.f32x4 v44\n" ++
  "  v46 = iconst.i64 0\n" ++
  "  jump block4(v40, v43, v46, v45)\n" ++
  "\n" ++
  "block4(v50: i64, v51: i64, v52: i64, v53: f32x4):\n" ++
  "  v54 = icmp sge v52, v9\n" ++
  "  brif v54, block5(v50, v51, v52, v53), block11(v50, v51, v52, v53)\n" ++
  "\n" ++
  "block11(v60: i64, v61: i64, v62: i64, v63: f32x4):\n" ++
  "  v64 = iadd v61, v62\n" ++
  "  v65 = iadd v6, v62\n" ++
  "  v66 = load.f32x4 notrap aligned v64\n" ++
  "  v67 = load.f32x4 notrap aligned v65\n" ++
  "  v68 = fmul v66, v67\n" ++
  "  v69 = fadd v63, v68\n" ++
  "  v70 = iadd_imm v62, 16\n" ++
  "  jump block4(v60, v61, v70, v69)\n" ++
  "\n" ++
  "block5(v80: i64, v81: i64, v82: i64, v83: f32x4):\n" ++
  "  v84 = extractlane v83, 0\n" ++
  "  v85 = extractlane v83, 1\n" ++
  "  v86 = extractlane v83, 2\n" ++
  "  v87 = extractlane v83, 3\n" ++
  "  v88 = fadd v84, v85\n" ++
  "  v89 = fadd v86, v87\n" ++
  "  v90 = fadd v88, v89\n" ++
  "  jump block12(v80, v81, v82, v90)\n" ++
  "\n" ++
  "block12(v100: i64, v101: i64, v102: i64, v103: f32):\n" ++
  "  v104 = icmp sge v102, v7\n" ++
  "  brif v104, block13(v100, v103), block14(v100, v101, v102, v103)\n" ++
  "\n" ++
  "block14(v110: i64, v111: i64, v112: i64, v113: f32):\n" ++
  "  v114 = iadd v111, v112\n" ++
  "  v115 = iadd v6, v112\n" ++
  "  v116 = load.f32 notrap aligned v114\n" ++
  "  v117 = load.f32 notrap aligned v115\n" ++
  "  v118 = fmul v116, v117\n" ++
  "  v119 = fadd v113, v118\n" ++
  "  v120 = iadd_imm v112, 4\n" ++
  "  jump block12(v110, v111, v120, v119)\n" ++
  "\n" ++
  "block13(v130: i64, v131: f32):\n" ++
  "  v132 = ishl_imm v130, 2\n" ++
  "  v133 = iadd v501, v132\n" ++
  "  store.f32 notrap aligned v131, v133\n" ++
  "  v134 = iadd_imm v130, 1\n" ++
  "  jump block1(v134)\n" ++
  "\n" ++
  "block6(v140: i64):\n" ++
  "  v141 = imul v140, v2\n" ++
  "  v142 = ishl_imm v141, 2\n" ++
  "  v143 = iadd v3, v142\n" ++
  "  v144 = iadd v143, v7\n" ++
  "  v145 = f32const 0.0\n" ++
  "  v146 = splat.f32x4 v145\n" ++
  "  v147 = iconst.i64 0\n" ++
  "  jump block7(v140, v143, v144, v147, v146, v146)\n" ++
  "\n" ++
  "block7(v150: i64, v151: i64, v152: i64, v153: i64, v154: f32x4, v155: f32x4):\n" ++
  "  v156 = icmp sge v153, v9\n" ++
  "  brif v156, block8(v150, v151, v152, v153, v154, v155), block15(v150, v151, v152, v153, v154, v155)\n" ++
  "\n" ++
  "block15(v160: i64, v161: i64, v162: i64, v163: i64, v164: f32x4, v165: f32x4):\n" ++
  "  v166 = iadd v161, v163\n" ++
  "  v167 = iadd v162, v163\n" ++
  "  v168 = iadd v6, v163\n" ++
  "  v169 = load.f32x4 notrap aligned v166\n" ++
  "  v170 = load.f32x4 notrap aligned v167\n" ++
  "  v171 = load.f32x4 notrap aligned v168\n" ++
  "  v172 = fmul v169, v171\n" ++
  "  v173 = fmul v170, v171\n" ++
  "  v174 = fadd v164, v172\n" ++
  "  v175 = fadd v165, v173\n" ++
  "  v176 = iadd_imm v163, 16\n" ++
  "  jump block7(v160, v161, v162, v176, v174, v175)\n" ++
  "\n" ++
  "block8(v180: i64, v181: i64, v182: i64, v183: i64, v184: f32x4, v185: f32x4):\n" ++
  "  v186 = extractlane v184, 0\n" ++
  "  v187 = extractlane v184, 1\n" ++
  "  v188 = extractlane v184, 2\n" ++
  "  v189 = extractlane v184, 3\n" ++
  "  v190 = fadd v186, v187\n" ++
  "  v191 = fadd v188, v189\n" ++
  "  v192 = fadd v190, v191\n" ++
  "  v193 = extractlane v185, 0\n" ++
  "  v194 = extractlane v185, 1\n" ++
  "  v195 = extractlane v185, 2\n" ++
  "  v196 = extractlane v185, 3\n" ++
  "  v197 = fadd v193, v194\n" ++
  "  v198 = fadd v195, v196\n" ++
  "  v199 = fadd v197, v198\n" ++
  "  jump block16(v180, v181, v182, v183, v192, v199)\n" ++
  "\n" ++
  "block16(v200: i64, v201: i64, v202: i64, v203: i64, v204: f32, v205: f32):\n" ++
  "  v206 = icmp sge v203, v7\n" ++
  "  brif v206, block9(v200, v204, v205), block17(v200, v201, v202, v203, v204, v205)\n" ++
  "\n" ++
  "block17(v210: i64, v211: i64, v212: i64, v213: i64, v214: f32, v215: f32):\n" ++
  "  v216 = iadd v211, v213\n" ++
  "  v217 = iadd v212, v213\n" ++
  "  v218 = iadd v6, v213\n" ++
  "  v219 = load.f32 notrap aligned v216\n" ++
  "  v220 = load.f32 notrap aligned v217\n" ++
  "  v221 = load.f32 notrap aligned v218\n" ++
  "  v222 = fmul v219, v221\n" ++
  "  v223 = fmul v220, v221\n" ++
  "  v224 = fadd v214, v222\n" ++
  "  v225 = fadd v215, v223\n" ++
  "  v226 = iadd_imm v213, 4\n" ++
  "  jump block16(v210, v211, v212, v226, v224, v225)\n" ++
  "\n" ++
  "block9(v230: i64, v231: f32, v232: f32):\n" ++
  "  v233 = ishl_imm v230, 2\n" ++
  "  v234 = iadd v501, v233\n" ++
  "  store.f32 notrap aligned v231, v234\n" ++
  "  v235 = iadd_imm v233, 4\n" ++
  "  v236 = iadd v501, v235\n" ++
  "  store.f32 notrap aligned v232, v236\n" ++
  "  v237 = iadd_imm v230, 2\n" ++
  "  jump block1(v237)\n" ++
  "\n" ++
  "block10:\n" ++
  "  return\n" ++
  "}\n"

def clifIR : String := clifNoopFn ++ "\n" ++ clifRowDotFn

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

end RowDotBench

def main : IO Unit := do
  let json := toJsonPair RowDotBench.buildConfig RowDotBench.buildAlgorithm
  IO.println (Json.compress json)
