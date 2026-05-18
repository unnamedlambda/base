import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace RowAffineReduceBench

/-
  Input: [rows:u64][cols:u64][X: rows*cols*f32][scale: cols*f32][bias: cols*f32]
  Output: [y: rows*f32] where y[i] = sum_j (X[i,j] * scale[j] + bias[j])
  4x-unrolled SIMD inner loop with scalar tail.
-/

def MEM_SIZE : Nat := 40

set_option maxRecDepth 2048 in
def mainFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let dataPtr← load64 (← absAddr ptr 0x18)
  let outPtr ← load64 (← absAddr ptr 0x28)
  let rows   ← load64 dataPtr
  let cols   ← load64 (← iaddImm dataPtr 8)
  let xPtr   ← iaddImm dataPtr 16
  -- scale = xPtr + rows*cols*4, bias = scale + cols*4
  let mk     ← imul rows cols
  let scalePtr ← iadd xPtr (← ishlImm mk 2)
  let stride ← ishlImm cols 2     -- cols * 4
  let biasPtr  ← iadd scalePtr stride
  -- loop bounds
  let mainEnd ← ishlImm (← ushrImm cols 4) 6   -- (cols/16)*64
  let simdEnd ← ishlImm (← ushrImm cols 2) 4   -- (cols/4)*16
  let zero   ← fconst32 f32Zero
  let zeroV  ← splat .f32x4 zero
  let i0     ← iconst64 0

  let outerL ← declareBlock [.i64]
  let rowSetup← declareBlock [.i64]
  let mainL  ← declareBlock [.i64, .i64, .i64, .f32x4, .f32x4, .f32x4, .f32x4]
  let mainB  ← declareBlock [.i64, .i64, .i64, .f32x4, .f32x4, .f32x4, .f32x4]
  let merge  ← declareBlock [.i64, .i64, .i64, .f32x4, .f32x4, .f32x4, .f32x4]
  let simdL  ← declareBlock [.i64, .i64, .i64, .f32x4]
  let simdB  ← declareBlock [.i64, .i64, .i64, .f32x4]
  let hred   ← declareBlock [.i64, .i64, .i64, .f32x4]
  let scalarL← declareBlock [.i64, .i64, .i64, .f32]
  let scalarB← declareBlock [.i64, .i64, .i64, .f32]
  let storeR ← declareBlock [.i64, .f32]
  let done   ← declareBlock []

  jump outerL.ref [i0]

  startBlock outerL
  let i := outerL.param 0
  brif (← icmp .sge i rows) done.ref [] rowSetup.ref [i]

  startBlock rowSetup
  let i2 := rowSetup.param 0
  let iCols ← imul i2 cols
  let xPtrI ← iadd xPtr (← ishlImm iCols 2)
  let j0    ← iconst64 0
  jump mainL.ref [i2, xPtrI, j0, zeroV, zeroV, zeroV, zeroV]

  -- 4x-unrolled main SIMD loop (64 bytes per iter)
  startBlock mainL
  let i3 := mainL.param 0; let xP := mainL.param 1; let jO := mainL.param 2
  let a0 := mainL.param 3; let a1 := mainL.param 4
  let a2 := mainL.param 5; let a3 := mainL.param 6
  brif (← icmp .sge jO mainEnd) merge.ref [i3, xP, jO, a0, a1, a2, a3]
                                 mainB.ref [i3, xP, jO, a0, a1, a2, a3]

  startBlock mainB
  let i4 := mainB.param 0; let xP2 := mainB.param 1; let jO2 := mainB.param 2
  let a02 := mainB.param 3; let a12 := mainB.param 4
  let a22 := mainB.param 5; let a32 := mainB.param 6
  let xA ← iadd xP2 jO2; let sA ← iadd scalePtr jO2; let bA ← iadd biasPtr jO2
  let x0 ← loadF32x4 xA; let s0 ← loadF32x4 sA; let b0 ← loadF32x4 bA
  let a02' ← fadd a02 (← fadd (← fmul x0 s0) b0)
  let xA1 ← iaddImm xA 16; let sA1 ← iaddImm sA 16; let bA1 ← iaddImm bA 16
  let x1 ← loadF32x4 xA1; let s1 ← loadF32x4 sA1; let b1 ← loadF32x4 bA1
  let a12' ← fadd a12 (← fadd (← fmul x1 s1) b1)
  let xA2 ← iaddImm xA 32; let sA2 ← iaddImm sA 32; let bA2 ← iaddImm bA 32
  let x2 ← loadF32x4 xA2; let s2 ← loadF32x4 sA2; let b2 ← loadF32x4 bA2
  let a22' ← fadd a22 (← fadd (← fmul x2 s2) b2)
  let xA3 ← iaddImm xA 48; let sA3 ← iaddImm sA 48; let bA3 ← iaddImm bA 48
  let x3 ← loadF32x4 xA3; let s3 ← loadF32x4 sA3; let b3 ← loadF32x4 bA3
  let a32' ← fadd a32 (← fadd (← fmul x3 s3) b3)
  jump mainL.ref [i4, xP2, ← iaddImm jO2 64, a02', a12', a22', a32']

  startBlock merge
  let i5 := merge.param 0; let xP3 := merge.param 1; let jO3 := merge.param 2
  let b0' := merge.param 3; let b1' := merge.param 4
  let b2' := merge.param 5; let b3' := merge.param 6
  let acc ← fadd (← fadd b0' b1') (← fadd b2' b3')
  jump simdL.ref [i5, xP3, jO3, acc]

  startBlock simdL
  let i6 := simdL.param 0; let xP4 := simdL.param 1
  let jO4 := simdL.param 2; let acc2 := simdL.param 3
  brif (← icmp .sge jO4 simdEnd) hred.ref [i6, xP4, jO4, acc2]
                                  simdB.ref [i6, xP4, jO4, acc2]

  startBlock simdB
  let i7 := simdB.param 0; let xP5 := simdB.param 1
  let jO5 := simdB.param 2; let acc3 := simdB.param 3
  let xA4 ← iadd xP5 jO5; let sA4 ← iadd scalePtr jO5; let bA4 ← iadd biasPtr jO5
  let xV ← loadF32x4 xA4; let sV ← loadF32x4 sA4; let bV ← loadF32x4 bA4
  let acc3' ← fadd acc3 (← fadd (← fmul xV sV) bV)
  jump simdL.ref [i7, xP5, ← iaddImm jO5 16, acc3']

  startBlock hred
  let i8 := hred.param 0; let xP6 := hred.param 1
  let jO6 := hred.param 2; let acc4 := hred.param 3
  let e0 ← extractlane acc4 0; let e1 ← extractlane acc4 1
  let e2 ← extractlane acc4 2; let e3 ← extractlane acc4 3
  let sum ← fadd (← fadd e0 e1) (← fadd e2 e3)
  jump scalarL.ref [i8, xP6, jO6, sum]

  startBlock scalarL
  let i9 := scalarL.param 0; let xP7 := scalarL.param 1
  let jO7 := scalarL.param 2; let s2 := scalarL.param 3
  brif (← icmp .sge jO7 stride) storeR.ref [i9, s2]
                                 scalarB.ref [i9, xP7, jO7, s2]

  startBlock scalarB
  let i10 := scalarB.param 0; let xP8 := scalarB.param 1
  let jO8 := scalarB.param 2; let s3 := scalarB.param 3
  let xA5 ← iadd xP8 jO8; let sA5 ← iadd scalePtr jO8; let bA5 ← iadd biasPtr jO8
  let xS ← loadF32 xA5; let sS ← loadF32 sA5; let bS ← loadF32 bA5
  let s3' ← fadd s3 (← fadd (← fmul xS sS) bS)
  jump scalarL.ref [i10, xP8, ← iaddImm jO8 4, s3']

  startBlock storeR
  let i11 := storeR.param 0; let s4 := storeR.param 1
  let oAddr ← iadd outPtr (← ishlImm i11 2)
  storeF32 s4 oAddr
  jump outerL.ref [← iaddImm i11 1]

  startBlock done
  ret

def clifIR : String := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "row_affine_reduce_algorithm" {
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end RowAffineReduceBench
