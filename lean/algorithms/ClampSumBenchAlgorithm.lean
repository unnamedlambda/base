import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace ClampSumBench

/-
  Input: n f32 values.  lo = -0.5, hi = 0.5  (hard-coded)
  Result: f64 — sum of clamp(x, lo, hi) for all x.

  CLIF: 4x-unrolled f32x4 clamp + accumulate, horizontal reduce to f64.
-/

def MEM_SIZE : Nat := 40

def mainFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr  ← load64 (← absAddr ptr 0x28)
  -- n = data_len / 4
  let n       ← ushrImm dataLen 2
  let mainEnd ← ishlImm (← ushrImm n 4) 6   -- (n/16)*64
  let simdEnd ← ishlImm (← ushrImm n 2) 4   -- (n/4)*16
  let scEnd   ← ishlImm n 2                  -- n*4
  -- Constants: hi=0.5, lo=-0.5
  let hi      ← fconst32 "0x1.000000p-1"
  let lo      ← fneg hi
  let hiV     ← splat .f32x4 hi
  let loV     ← splat .f32x4 lo
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
  let clamp := fun (v acc : Val) => do
    let t ← fmin v hiV; let t' ← fmax t loV; fadd acc t'
  let a2' ← clamp (← loadF32x4 off0) a2
  let b2' ← clamp (← loadF32x4 (← iaddImm off0 16)) b2
  let c2' ← clamp (← loadF32x4 (← iaddImm off0 32)) c2
  let d2' ← clamp (← loadF32x4 (← iaddImm off0 48)) d2
  jump loop1.ref [← iaddImm i2 64, a2', b2', c2', d2']

  startBlock merge
  let i3 := merge.param 0; let a3 := merge.param 1; let b3 := merge.param 2
  let c3 := merge.param 3; let d3 := merge.param 4
  let acc ← fadd (← fadd a3 b3) (← fadd c3 d3)

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
  let v  ← loadF32x4 off
  let t  ← fmin v hiV; let t' ← fmax t loV
  jump loop2.ref [← iaddImm i5 16, ← fadd v5 t']

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
  let v    ← loadF32 aOff
  let t    ← fmin v hi; let t' ← fmax t lo
  jump loop3.ref [← iaddImm i8 4, ← fadd s8 (← fpromote t')]

  startBlock done
  storeF64 (done.param 0) outPtr
  ret

def clifIR : String := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "clamp_sum_algorithm" {
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end ClampSumBench
