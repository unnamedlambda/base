import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace RowDotBench

/-
  Row-wise dot product: y[i] = dot(X[i,:], w)
  Payload: [rows:u64][cols:u64][X: rows*cols*f32][w: cols*f32]
  Output:  [y: rows*f32]
  Dual-row path for pairs, single-row for last row. SIMD inner loop + scalar tail.
-/

def MEM_SIZE : Nat := 40

set_option maxRecDepth 4096 in
def mainFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let dataPtr← load64 (← absAddr ptr 0x18)
  let outPtr ← load64 (← absAddr ptr 0x28)
  let rows   ← load64 dataPtr
  let cols   ← load64 (← iaddImm dataPtr 8)
  let xPtr   ← iaddImm dataPtr 16
  let wPtr   ← iadd xPtr (← ishlImm (← imul rows cols) 2)
  let stride ← ishlImm cols 2            -- cols * 4 (row byte size)
  let simdEnd← ishlImm (← ushrImm cols 2) 4  -- (cols/4)*16

  -- Outer loop
  let outerL  ← declareBlock [.i64]
  let outerD  ← declareBlock [.i64]  -- dispatch: dual or single
  -- Single-row path (last row)
  let single  ← declareBlock [.i64]
  let sSimdL  ← declareBlock [.i64, .i64, .i64, .f32x4]
  let sSimdB  ← declareBlock [.i64, .i64, .i64, .f32x4]
  let sHred   ← declareBlock [.i64, .i64, .i64, .f32x4]
  let sScalar ← declareBlock [.i64, .i64, .i64, .f32]
  let sScalarB← declareBlock [.i64, .i64, .i64, .f32]
  let sStore  ← declareBlock [.i64, .f32]
  -- Dual-row path (pairs)
  let dual    ← declareBlock [.i64]
  let dSimdL  ← declareBlock [.i64, .i64, .i64, .i64, .f32x4, .f32x4]
  let dSimdB  ← declareBlock [.i64, .i64, .i64, .i64, .f32x4, .f32x4]
  let dHred   ← declareBlock [.i64, .i64, .i64, .i64, .f32x4, .f32x4]
  let dScalar ← declareBlock [.i64, .i64, .i64, .i64, .f32, .f32]
  let dScalarB← declareBlock [.i64, .i64, .i64, .i64, .f32, .f32]
  let dStore  ← declareBlock [.i64, .f32, .f32]
  let done    ← declareBlock []

  let zero   ← iconst64 0
  let zeroF  ← fconst32 f32Zero
  let zeroV  ← splat .f32x4 zeroF
  jump outerL.ref [zero]

  startBlock outerL
  let i := outerL.param 0
  brif (← icmp .sge i rows) done.ref [] outerD.ref [i]

  startBlock outerD
  let iD := outerD.param 0
  let i1 ← iaddImm iD 1
  brif (← icmp .slt i1 rows) dual.ref [iD] single.ref [iD]

  -- Single-row SIMD dot product
  startBlock single
  let i2 := single.param 0
  let rOff ← ishlImm (← imul i2 cols) 2
  let xP   ← iadd xPtr rOff
  jump sSimdL.ref [i2, xP, zero, zeroV]

  startBlock sSimdL
  let i3 := sSimdL.param 0; let xP2 := sSimdL.param 1
  let jO := sSimdL.param 2; let acc := sSimdL.param 3
  brif (← icmp .sge jO simdEnd) sHred.ref [i3, xP2, jO, acc]
                                  sSimdB.ref [i3, xP2, jO, acc]

  startBlock sSimdB
  let i4 := sSimdB.param 0; let xP3 := sSimdB.param 1
  let jO2 := sSimdB.param 2; let acc2 := sSimdB.param 3
  let xV   ← loadF32x4 (← iadd xP3 jO2)
  let wV   ← loadF32x4 (← iadd wPtr jO2)
  let acc2'← fadd acc2 (← fmul xV wV)
  jump sSimdL.ref [i4, xP3, ← iaddImm jO2 16, acc2']

  startBlock sHred
  let i5 := sHred.param 0; let xP4 := sHred.param 1
  let jO3 := sHred.param 2; let acc3 := sHred.param 3
  let e0 ← extractlane acc3 0; let e1 ← extractlane acc3 1
  let e2 ← extractlane acc3 2; let e3 ← extractlane acc3 3
  let s  ← fadd (← fadd e0 e1) (← fadd e2 e3)
  jump sScalar.ref [i5, xP4, jO3, s]

  startBlock sScalar
  let i6 := sScalar.param 0; let xP5 := sScalar.param 1
  let jO4 := sScalar.param 2; let s2 := sScalar.param 3
  brif (← icmp .sge jO4 stride) sStore.ref [i6, s2] sScalarB.ref [i6, xP5, jO4, s2]

  startBlock sScalarB
  let i7 := sScalarB.param 0; let xP6 := sScalarB.param 1
  let jO5 := sScalarB.param 2; let s3 := sScalarB.param 3
  let xS  ← loadF32 (← iadd xP6 jO5)
  let wS  ← loadF32 (← iadd wPtr jO5)
  let s3' ← fadd s3 (← fmul xS wS)
  jump sScalar.ref [i7, xP6, ← iaddImm jO5 4, s3']

  startBlock sStore
  let i8 := sStore.param 0; let s4 := sStore.param 1
  storeF32 s4 (← iadd outPtr (← ishlImm i8 2))
  jump outerL.ref [← iaddImm i8 1]

  -- Dual-row SIMD dot product (processes rows i and i+1 together)
  startBlock dual
  let i9 := dual.param 0
  let rOff2 ← ishlImm (← imul i9 cols) 2
  let xP7   ← iadd xPtr rOff2
  let xP7b  ← iadd xP7 stride   -- row i+1
  jump dSimdL.ref [i9, xP7, xP7b, zero, zeroV, zeroV]

  startBlock dSimdL
  let i10 := dSimdL.param 0; let xP8  := dSimdL.param 1
  let xP9  := dSimdL.param 2; let jO6 := dSimdL.param 3
  let ac0  := dSimdL.param 4; let ac1  := dSimdL.param 5
  brif (← icmp .sge jO6 simdEnd) dHred.ref [i10, xP8, xP9, jO6, ac0, ac1]
                                   dSimdB.ref [i10, xP8, xP9, jO6, ac0, ac1]

  startBlock dSimdB
  let i11 := dSimdB.param 0; let xP10 := dSimdB.param 1
  let xP11 := dSimdB.param 2; let jO7  := dSimdB.param 3
  let ac02 := dSimdB.param 4; let ac12 := dSimdB.param 5
  let wV2  ← loadF32x4 (← iadd wPtr jO7)
  let xV0  ← loadF32x4 (← iadd xP10 jO7)
  let xV1  ← loadF32x4 (← iadd xP11 jO7)
  let ac02'← fadd ac02 (← fmul xV0 wV2)
  let ac12'← fadd ac12 (← fmul xV1 wV2)
  jump dSimdL.ref [i11, xP10, xP11, ← iaddImm jO7 16, ac02', ac12']

  startBlock dHred
  let i12 := dHred.param 0; let xP12 := dHred.param 1
  let xP13 := dHred.param 2; let jO8  := dHred.param 3
  let ac03 := dHred.param 4; let ac13 := dHred.param 5
  let e00 ← extractlane ac03 0; let e01 ← extractlane ac03 1
  let e02 ← extractlane ac03 2; let e03 ← extractlane ac03 3
  let s0  ← fadd (← fadd e00 e01) (← fadd e02 e03)
  let e10 ← extractlane ac13 0; let e11 ← extractlane ac13 1
  let e12 ← extractlane ac13 2; let e13 ← extractlane ac13 3
  let s1  ← fadd (← fadd e10 e11) (← fadd e12 e13)
  jump dScalar.ref [i12, xP12, xP13, jO8, s0, s1]

  startBlock dScalar
  let i13 := dScalar.param 0; let xP14 := dScalar.param 1
  let xP15 := dScalar.param 2; let jO9  := dScalar.param 3
  let sf0  := dScalar.param 4; let sf1  := dScalar.param 5
  brif (← icmp .sge jO9 stride) dStore.ref [i13, sf0, sf1]
                                  dScalarB.ref [i13, xP14, xP15, jO9, sf0, sf1]

  startBlock dScalarB
  let i14  := dScalarB.param 0; let xP16 := dScalarB.param 1
  let xP17 := dScalarB.param 2; let jO10 := dScalarB.param 3
  let sf02 := dScalarB.param 4; let sf12 := dScalarB.param 5
  let wS2  ← loadF32 (← iadd wPtr jO10)
  let xS0  ← loadF32 (← iadd xP16 jO10)
  let xS1  ← loadF32 (← iadd xP17 jO10)
  let sf02'← fadd sf02 (← fmul xS0 wS2)
  let sf12'← fadd sf12 (← fmul xS1 wS2)
  jump dScalar.ref [i14, xP16, xP17, ← iaddImm jO10 4, sf02', sf12']

  startBlock dStore
  let i15 := dStore.param 0; let sf03 := dStore.param 1; let sf13 := dStore.param 2
  storeF32 sf03 (← iadd outPtr (← ishlImm i15 2))
  storeF32 sf13 (← iadd outPtr (← ishlImm (← iaddImm i15 1) 2))
  jump outerL.ref [← iaddImm i15 2]

  startBlock done
  ret

def clifIR : String := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "row_dot_algorithm" {
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE,
    context_offset := 0
  } {
    fn_idx := u32 1
  }]

end RowDotBench
