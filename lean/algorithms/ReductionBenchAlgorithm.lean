import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace ReductionBench

/-
  SIMD f32 sum reduction benchmark.

  Payload (via execute data arg): [f32 values: n floats]
  Output (via execute_into out arg): [0..8) result (f64) — sum of all elements

  CLIF: 4x-unrolled SIMD sum of f32 array → f64 result.
-/

def TIMEOUT_MS : Nat := 30000

def mainFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr  ← load64 (← absAddr ptr 0x28)
  -- n = data_len / 4
  let n       ← ushrImm dataLen 2
  -- loop bounds (byte offsets)
  let mainEnd ← ishlImm (← ushrImm n 4) 6   -- (n/16)*64
  let simdEnd ← ishlImm (← ushrImm n 2) 4   -- (n/4)*16
  let scEnd   ← ishlImm n 2                  -- n*4
  let zero    ← fconst32 f32Zero
  let acc0    ← splat .f32x4 zero
  let i0      ← iconst64 0

  let loop1  ← declareBlock [.i64, .f32x4, .f32x4, .f32x4, .f32x4]
  let body1  ← declareBlock [.i64, .f32x4, .f32x4, .f32x4, .f32x4]
  let merge  ← declareBlock [.i64, .f32x4, .f32x4, .f32x4, .f32x4]
  jump loop1.ref [i0, acc0, acc0, acc0, acc0]

  startBlock loop1
  let i1 := loop1.param 0; let a1 := loop1.param 1; let b1 := loop1.param 2
  let c1 := loop1.param 3; let d1 := loop1.param 4
  brif (← icmp .sge i1 mainEnd) merge.ref [i1, a1, b1, c1, d1] body1.ref [i1, a1, b1, c1, d1]

  startBlock body1
  let i2 := body1.param 0; let a2 := body1.param 1; let b2 := body1.param 2
  let c2 := body1.param 3; let d2 := body1.param 4
  let off0 ← iadd dataPtr i2
  let a2' ← fadd a2 (← loadF32x4 off0)
  let b2' ← fadd b2 (← loadF32x4 (← iaddImm off0 16))
  let c2' ← fadd c2 (← loadF32x4 (← iaddImm off0 32))
  let d2' ← fadd d2 (← loadF32x4 (← iaddImm off0 48))
  jump loop1.ref [← iaddImm i2 64, a2', b2', c2', d2']

  startBlock merge
  let i3 := merge.param 0; let a3 := merge.param 1; let b3 := merge.param 2
  let c3 := merge.param 3; let d3 := merge.param 4
  let ab  ← fadd a3 b3
  let cd  ← fadd c3 d3
  let acc ← fadd ab cd

  let loop2 ← declareBlock [.i64, .f32x4]
  let body2 ← declareBlock [.i64, .f32x4]
  let hred  ← declareBlock [.i64, .f32x4]
  jump loop2.ref [i3, acc]

  startBlock loop2
  let i4 := loop2.param 0; let v4 := loop2.param 1
  brif (← icmp .sge i4 simdEnd) hred.ref [i4, v4] body2.ref [i4, v4]

  startBlock body2
  let i5 := body2.param 0; let v5 := body2.param 1
  let off ← iadd dataPtr i5
  jump loop2.ref [← iaddImm i5 16, ← fadd v5 (← loadF32x4 off)]

  startBlock hred
  let i6 := hred.param 0; let v6 := hred.param 1
  let sum64 ← fadd (← fadd (← fpromote (← extractlane v6 0))
                            (← fpromote (← extractlane v6 1)))
                   (← fadd (← fpromote (← extractlane v6 2))
                            (← fpromote (← extractlane v6 3)))

  let loop3 ← declareBlock [.i64, .f64]
  let body3 ← declareBlock [.i64, .f64]
  let done  ← declareBlock [.f64]
  jump loop3.ref [i6, sum64]

  startBlock loop3
  let i7 := loop3.param 0; let s7 := loop3.param 1
  brif (← icmp .sge i7 scEnd) done.ref [s7] body3.ref [i7, s7]

  startBlock body3
  let i8 := body3.param 0; let s8 := body3.param 1
  let aOff ← iadd dataPtr i8
  jump loop3.ref [← iaddImm i8 4, ← fadd s8 (← fpromote (← loadF32 aOff))]

  startBlock done
  storeF64 (done.param 0) outPtr
  ret

def clifIR : String := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "reduction_algorithm" {
    cranelift_ir := clifIR,
    memory_size := 40,
    context_offset := 0
  } {
    actions := mkCallActions 1,
    cranelift_units := 0,
    timeout_ms := some TIMEOUT_MS
  }]

end ReductionBench
