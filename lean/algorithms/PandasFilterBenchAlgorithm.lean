import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace PandasFilterBench

def ACC_OFF  : Nat := 0x28
def N_CATS   : Nat := 16
def MEM_SIZE : Nat := ACC_OFF + N_CATS * 8   -- 168 bytes

def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

def clifFilterRevAggFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "  v500 = load.i64 notrap aligned v0+0x08\n" ++   -- data_ptr
  "  v501 = load.i64 notrap aligned v0+0x10\n" ++   -- data_len
  "  v502 = load.i64 notrap aligned v0+0x18\n" ++   -- out_ptr
  "  v503 = iadd_imm v0, 40\n" ++                   -- acc_base (0x28 = 40)
  "  v504 = iadd v500, v501\n" ++                   -- data_end
  "  v600 = iconst.i64 0\n" ++                      -- zero for init
  "  v601 = iadd_imm v503, 128\n" ++                -- acc_end
  "  v700 = f32const 0x1.900000p5\n" ++             -- price threshold = 50.0
  "  jump block1(v503)\n" ++
  "\n" ++
  -- Zero accumulators
  "block1(v10: i64):\n" ++
  "  store.i64 notrap aligned v600, v10\n" ++
  "  v11 = iadd_imm v10, 8\n" ++
  "  v12 = icmp ult v11, v601\n" ++
  "  brif v12, block1(v11), block2\n" ++
  "\n" ++
  -- Check empty
  "block2:\n" ++
  "  v20 = icmp uge v500, v504\n" ++
  "  brif v20, block5, block3(v500)\n" ++
  "\n" ++
  -- Per-row: branchless filter via integer bit-mask.
  -- fcmp → ineg → sextend → band with revenue bits → bitcast back to f64.
  -- Zero CPU branches: avoids misprediction at ~50% filter hit rate.
  "block3(v30: i64):\n" ++
  "  v31 = uload32.i64 notrap aligned v30\n" ++     -- category
  "  v32 = ishl_imm v31, 3\n" ++                    -- category * 8
  "  v33 = iadd v503, v32\n" ++                     -- &acc[category]
  "  v34 = load.f32 notrap aligned v30+4\n" ++      -- price
  "  v35 = load.f32 notrap aligned v30+8\n" ++      -- quantity
  "  v36 = fpromote.f64 v34\n" ++
  "  v37 = fpromote.f64 v35\n" ++
  "  v38 = fmul v36, v37\n" ++                      -- revenue (f64)
  "  v705 = fcmp gt v34, v700\n" ++                 -- i8: 1 (pass) or 0 (skip)
  "  v706 = ineg v705\n" ++                         -- i8: 0xFF (pass) or 0x00 (skip)
  "  v707 = sextend.i64 v706\n" ++                  -- i64: all-1s or all-0s
  "  v708 = bitcast.i64 v38\n" ++                   -- revenue bits as i64
  "  v709 = band v707, v708\n" ++                   -- mask: revenue bits or 0
  "  v710 = bitcast.f64 v709\n" ++                  -- back to f64: revenue or 0.0
  "  v39 = load.f64 notrap aligned v33\n" ++
  "  v40 = fadd v39, v710\n" ++
  "  store.f64 notrap aligned v40, v33\n" ++
  "  v41 = iadd_imm v30, 12\n" ++
  "  v42 = icmp ult v41, v504\n" ++
  "  brif v42, block3(v41), block5\n" ++
  "\n" ++
  -- Pairwise fmax reduction: 16 → 1
  "block5:\n" ++
  "  v50 = load.f64 notrap aligned v503\n" ++
  "  v51 = load.f64 notrap aligned v503+8\n" ++
  "  v52 = load.f64 notrap aligned v503+16\n" ++
  "  v53 = load.f64 notrap aligned v503+24\n" ++
  "  v54 = load.f64 notrap aligned v503+32\n" ++
  "  v55 = load.f64 notrap aligned v503+40\n" ++
  "  v56 = load.f64 notrap aligned v503+48\n" ++
  "  v57 = load.f64 notrap aligned v503+56\n" ++
  "  v58 = load.f64 notrap aligned v503+64\n" ++
  "  v59 = load.f64 notrap aligned v503+72\n" ++
  "  v60 = load.f64 notrap aligned v503+80\n" ++
  "  v61 = load.f64 notrap aligned v503+88\n" ++
  "  v62 = load.f64 notrap aligned v503+96\n" ++
  "  v63 = load.f64 notrap aligned v503+104\n" ++
  "  v64 = load.f64 notrap aligned v503+112\n" ++
  "  v65 = load.f64 notrap aligned v503+120\n" ++
  "  v70 = fmax v50, v51\n" ++
  "  v71 = fmax v52, v53\n" ++
  "  v72 = fmax v54, v55\n" ++
  "  v73 = fmax v56, v57\n" ++
  "  v74 = fmax v58, v59\n" ++
  "  v75 = fmax v60, v61\n" ++
  "  v76 = fmax v62, v63\n" ++
  "  v77 = fmax v64, v65\n" ++
  "  v78 = fmax v70, v71\n" ++
  "  v79 = fmax v72, v73\n" ++
  "  v80 = fmax v74, v75\n" ++
  "  v81 = fmax v76, v77\n" ++
  "  v82 = fmax v78, v79\n" ++
  "  v83 = fmax v80, v81\n" ++
  "  v84 = fmax v82, v83\n" ++
  "  store.f64 notrap aligned v84, v502\n" ++
  "  return\n" ++
  "}"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifFilterRevAggFn

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

end PandasFilterBench

def main : IO Unit := do
  let json := toJsonPair PandasFilterBench.buildConfig PandasFilterBench.buildAlgorithm
  IO.println (Json.compress json)
