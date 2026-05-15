import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace MatmulBench

/-
  SIMD f32 matrix multiplication benchmark.

  Payload: [M:u32][K:u32][N:u32][A: M*K f32][B: K*N f32]
  Output:  [result: f64] — sum of all elements of C = A*B

  CLIF: SIMD matmul with 4x-unrolled j-loop (16 cols/iter).
-/


def mainFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let outPtr  ← load64 (← absAddr ptr 0x28)
  -- Parse M, K, N
  let mVal32  ← load32 dataPtr
  let kVal32  ← load32 (← iaddImm dataPtr 4)
  let nVal32  ← load32 (← iaddImm dataPtr 8)
  let mVal    ← sextend64 mVal32
  let kVal    ← sextend64 kVal32
  let nVal    ← sextend64 nVal32
  -- A starts at dataPtr+12, B starts at A + M*K*4
  let aPtr    ← iaddImm dataPtr 12
  let mk      ← imul mVal kVal
  let bPtr    ← iadd aPtr (← ishlImm mk 2)
  let nBytes  ← ishlImm nVal 2          -- N*4 (row stride of B)
  let zero64  ← iconst64 0
  let zero    ← fconst64 f64Zero
  let zeroV   ← splat .f32x4 (← fconst32 f32Zero)

  -- Outer loop: for i in 0..M
  let outerL ← declareBlock [.i64, .f64]
  let outerB ← declareBlock [.i64, .f64]
  let innerL ← declareBlock [.i64, .f64, .i64]
  let innerB ← declareBlock [.i64, .f64, .i64]
  -- per-(i,p) j-loop
  let jLoop  ← declareBlock [.i64, .f64, .i64, .f32x4, .i64, .i64, .f32x4, .f32x4, .f32x4, .f32x4]
  let jBody  ← declareBlock [.i64, .f64, .i64, .f32x4, .i64, .i64, .f32x4, .f32x4, .f32x4, .f32x4]
  let reduce ← declareBlock [.i64, .f64, .i64, .f32x4, .f32x4, .f32x4, .f32x4]
  let nextP  ← declareBlock [.i64, .f64]
  let done   ← declareBlock [.f64]
  jump outerL.ref [zero64, zero]

  startBlock outerL
  let i := outerL.param 0; let total := outerL.param 1
  brif (← icmp .sge i mVal) done.ref [total] outerB.ref [i, total]

  startBlock outerB
  let i' := outerB.param 0; let tot' := outerB.param 1
  jump innerL.ref [i', tot', zero64]

  startBlock innerL
  let i2 := innerL.param 0; let tot2 := innerL.param 1; let p := innerL.param 2
  brif (← icmp .sge p kVal) nextP.ref [i2, tot2] innerB.ref [i2, tot2, p]

  startBlock nextP
  let iN := nextP.param 0; let tN := nextP.param 1
  jump outerL.ref [← iaddImm iN 1, tN]

  startBlock innerB
  let i3 := innerB.param 0; let tot3 := innerB.param 1; let p2 := innerB.param 2
  -- Load A[i,p]: aPtr + (i*K+p)*4
  let ikp  ← iadd (← imul i3 kVal) p2
  let aOff ← iadd aPtr (← ishlImm ikp 2)
  let aVal ← loadF32 aOff
  let aVec ← splat .f32x4 aVal
  -- B row: bPtr + p*N*4
  let bRow ← iadd bPtr (← ishlImm (← imul p2 nVal) 2)
  jump jLoop.ref [i3, tot3, p2, aVec, bRow, zero64, zeroV, zeroV, zeroV, zeroV]

  startBlock jLoop
  let i4  := jLoop.param 0; let tot4 := jLoop.param 1; let p3 := jLoop.param 2
  let aV  := jLoop.param 3; let bRow2 := jLoop.param 4; let jOff := jLoop.param 5
  let c0  := jLoop.param 6; let c1 := jLoop.param 7
  let c2  := jLoop.param 8; let c3 := jLoop.param 9
  brif (← icmp .sge jOff nBytes) reduce.ref [i4, tot4, p3, c0, c1, c2, c3]
                                  jBody.ref  [i4, tot4, p3, aV, bRow2, jOff, c0, c1, c2, c3]

  startBlock jBody
  let i5  := jBody.param 0; let tot5 := jBody.param 1; let p4 := jBody.param 2
  let aV2 := jBody.param 3; let bRow3 := jBody.param 4; let jOff2 := jBody.param 5
  let c02 := jBody.param 6; let c12 := jBody.param 7
  let c22 := jBody.param 8; let c32 := jBody.param 9
  let bBase ← iadd bRow3 jOff2
  let c0' ← fadd c02 (← fmul aV2 (← loadF32x4 bBase))
  let c1' ← fadd c12 (← fmul aV2 (← loadF32x4 (← iaddImm bBase 16)))
  let c2' ← fadd c22 (← fmul aV2 (← loadF32x4 (← iaddImm bBase 32)))
  let c3' ← fadd c32 (← fmul aV2 (← loadF32x4 (← iaddImm bBase 48)))
  jump jLoop.ref [i5, tot5, p4, aV2, bRow3, ← iaddImm jOff2 64, c0', c1', c2', c3']

  startBlock reduce
  let i6  := reduce.param 0; let tot6 := reduce.param 1; let p5 := reduce.param 2
  let r0  := reduce.param 3; let r1 := reduce.param 4
  let r2  := reduce.param 5; let r3 := reduce.param 6
  let vAcc ← fadd (← fadd r0 r1) (← fadd r2 r3)
  let rowSum ← fadd (← fadd (← fpromote (← extractlane vAcc 0))
                             (← fpromote (← extractlane vAcc 1)))
                    (← fadd (← fpromote (← extractlane vAcc 2))
                             (← fpromote (← extractlane vAcc 3)))
  let tot7 ← fadd tot6 rowSum
  jump innerL.ref [i6, tot7, ← iaddImm p5 1]

  startBlock done
  storeF64 (done.param 0) outPtr
  ret

def clifIR : String := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "matmul_algorithm" {
    cranelift_ir := clifIR,
    memory_size := 40,
    context_offset := 0
  } {
    fn_idx := u32 1
  }]

end MatmulBench
