import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace PandasFilterBench

/-
  Row layout: [category: u32][price: f32][quantity: f32] = 12 bytes, 16 categories.
  Filter: price > 50.0 (branchless via fcmp+ineg+sextend mask).
  Output: max category revenue as f64
-/

def ACC_OFF  : Nat := 0x28
def N_CATS   : Nat := 16
def MEM_SIZE : Nat := ACC_OFF + N_CATS * 8

def mainFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr  ← load64 (← absAddr ptr 0x28)
  let accBase ← absAddr ptr ACC_OFF
  let dataEnd ← iadd dataPtr dataLen
  let zero    ← iconst64 0
  let accEnd  ← iaddImm accBase 128    -- 16 * 8
  let thresh  ← fconst32 "0x1.900000p5"   -- 50.0f

  let zloop ← declareBlock [.i64]
  let chk   ← declareBlock []
  let rowL  ← declareBlock [.i64]
  let done  ← declareBlock []

  -- Zero 16 f64 accumulators
  jump zloop.ref [accBase]
  startBlock zloop
  let z := zloop.param 0
  store zero z
  let z' ← iaddImm z 8
  brif (← icmp .ult z' accEnd) zloop.ref [z'] chk.ref []

  -- Empty check
  startBlock chk
  brif (← icmp .uge dataPtr dataEnd) done.ref [] rowL.ref [dataPtr]

  -- Row loop: branchless filter via integer bit-mask
  startBlock rowL
  let row  := rowL.param 0
  let cat  ← uload32_64 row
  let acc  ← iadd accBase (← ishlImm cat 3)
  let price ← loadF32 (← iaddImm row 4)
  let qty   ← loadF32 (← iaddImm row 8)
  let rev   ← fmul (← fpromote price) (← fpromote qty)
  let mask8 ← fcmpGt price thresh     -- i8: 1 (pass) or 0 (skip)
  let neg8  ← ineg mask8              -- i8: 0xFF (pass) or 0x00 (skip)
  let mask  ← sextend64 neg8          -- i64: all-1s or all-0s
  let bits  ← bitcastI64 rev          -- revenue bits as i64
  let mrev  ← band mask bits          -- mask: revenue bits or 0
  let frev  ← bitcastF64 mrev         -- back to f64: revenue or 0.0
  let old   ← loadF64 acc
  storeF64 (← fadd old frev) acc
  let row' ← iaddImm row 12
  brif (← icmp .ult row' dataEnd) rowL.ref [row'] done.ref []

  -- Pairwise fmax reduction: 16 → 1
  startBlock done
  let v0  ← loadF64 (← iaddImm accBase 0)
  let v1  ← loadF64 (← iaddImm accBase 8)
  let v2  ← loadF64 (← iaddImm accBase 16)
  let v3  ← loadF64 (← iaddImm accBase 24)
  let v4  ← loadF64 (← iaddImm accBase 32)
  let v5  ← loadF64 (← iaddImm accBase 40)
  let v6  ← loadF64 (← iaddImm accBase 48)
  let v7  ← loadF64 (← iaddImm accBase 56)
  let v8  ← loadF64 (← iaddImm accBase 64)
  let v9  ← loadF64 (← iaddImm accBase 72)
  let v10 ← loadF64 (← iaddImm accBase 80)
  let v11 ← loadF64 (← iaddImm accBase 88)
  let v12 ← loadF64 (← iaddImm accBase 96)
  let v13 ← loadF64 (← iaddImm accBase 104)
  let v14 ← loadF64 (← iaddImm accBase 112)
  let v15 ← loadF64 (← iaddImm accBase 120)
  let m01   ← fmax v0  v1;   let m23    ← fmax v2  v3
  let m45   ← fmax v4  v5;   let m67    ← fmax v6  v7
  let m89   ← fmax v8  v9;   let m1011  ← fmax v10 v11
  let m1213 ← fmax v12 v13;  let m1415  ← fmax v14 v15
  let m0123   ← fmax m01 m23;    let m4567    ← fmax m45 m67
  let m891011 ← fmax m89 m1011;  let m12131415 ← fmax m1213 m1415
  let top ← fmax (← fmax m0123 m4567) (← fmax m891011 m12131415)
  storeF64 top outPtr
  ret

def clifIR : String := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "pandas_filter_algorithm" {
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE,
    context_offset := 0
  } {
    actions := mkCallActions 1,
    cranelift_units := 0,
    timeout_ms := some 30000
  }]

end PandasFilterBench
