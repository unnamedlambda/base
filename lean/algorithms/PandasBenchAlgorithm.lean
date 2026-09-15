import Lean
import AlgorithmLib.Gen

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

namespace PandasBench

/-
  Row layout: [category: u32][price: f32][quantity: f32] = 12 bytes, 16 categories.

  Memory layout:
    0x0000..0x0027  reserved
    0x0028..0x00A7  accumulators[16]  (16 × 8 bytes f64)

  Output: max category revenue as f64
-/

def ACC_OFF  : Nat := 0x28
def N_CATS   : Nat := 16
def MEM_SIZE : Nat := ACC_OFF + N_CATS * 8

open AlgorithmLib.Prog


def code : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let dataLen ← dataLen
  let outPtr  ← outPtr
  let accBase ← absAddr ptr ACC_OFF
  let dataEnd ← iadd dataPtr dataLen
  let zero    ← iconst64 0
  let accEnd  ← iaddImm accBase 128    -- 16 * 8

  -- Zero 16 f64 accumulators. Fixed size, so no guard.
  let _ ← dwloop %[accBase] .ult accEnd (contOnTrue := true) []
    (body := fun c => do
      let z := c.head
      storeUnaligned zero z
      let z' ← iaddImm z 8
      return (z', %[z']))

  -- Row loop: acc[category] += price * quantity
  let _ ← dwloop %[dataPtr] .ult dataEnd (contOnTrue := true) []
    (body := fun c => do
      let row := c.head
      let cat  ← uload32_64 row
      let acc  ← iadd accBase (← ishlImm cat 3)
      let price ← loadF32 (← iaddImm row 4)
      let qty   ← loadF32 (← iaddImm row 8)
      let rev   ← fmul (← fpromote price) (← fpromote qty)
      let old   ← loadF64 acc
      store (← fadd old rev) acc
      let row' ← iaddImm row 12
      return (row', %[row']))

  -- Pairwise fmax reduction: 16 → 1
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
  let v10  ← loadF64 (← iaddImm accBase 80)
  let v11  ← loadF64 (← iaddImm accBase 88)
  let v12  ← loadF64 (← iaddImm accBase 96)
  let v13  ← loadF64 (← iaddImm accBase 104)
  let v14  ← loadF64 (← iaddImm accBase 112)
  let v15  ← loadF64 (← iaddImm accBase 120)
  let m01   ← fmax v0  v1;   let m23    ← fmax v2  v3
  let m45   ← fmax v4  v5;   let m67    ← fmax v6  v7
  let m89   ← fmax v8  v9;   let m1011  ← fmax v10 v11
  let m1213 ← fmax v12 v13;  let m1415  ← fmax v14 v15
  let m0123   ← fmax m01 m23;    let m4567    ← fmax m45 m67
  let m891011 ← fmax m89 m1011;  let m12131415 ← fmax m1213 m1415
  let top ← fmax (← fmax m0123 m4567) (← fmax m891011 m12131415)
  store top outPtr


def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 code]

def artifacts (clif : List FuncData) : Array Json :=
  #[toJsonArtifact "pandas_algorithm" {
    functions := clif,
    memory_size := MEM_SIZE
  }]

end PandasBench
