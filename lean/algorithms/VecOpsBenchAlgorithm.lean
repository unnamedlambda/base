import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace VecOpsBench

/-
  SIMD f32 vec_add benchmark.

  Payload (via execute data arg):
    [A floats: n * f32][B floats: n * f32]
    n is derived from data_len / 8 (two f32 arrays back to back)

  Output (via execute_into out arg):
    [0..8)  result (f64) — sum of all (A[i] + B[i])

  Memory layout (shared memory):
    0x0000..0x0027  reserved (runtime writes ctx_ptr, data_ptr, data_len, out_ptr, out_len)

  The CLIF code reads arrays directly from the data pointer (zero copy)
  and writes the result to the out pointer (zero copy).

  CLIF: 4x-unrolled SIMD vec_add of two f32 arrays → f64 sum.
  Main loop: 4 independent f32x4 accumulators (16 floats/iter) for ILP.
-/

def TIMEOUT_MS : Nat := 30000

def mainFn : IRBuilder Unit := do
  let ptr ← entryBlock
  -- Load data_ptr, data_len, out_ptr from reserved region
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr  ← load64 (← absAddr ptr 0x28)
  -- n = data_len / 8 (two f32 arrays)
  let n       ← ushrImm dataLen 3
  -- A_ptr = dataPtr, B_ptr = dataPtr + n*4
  let nBytes  ← ishlImm n 2
  let bPtr    ← iadd dataPtr nBytes
  -- Loop bounds
  let mainEnd ← ishlImm (← ushrImm n 4) 6   -- (n/16)*64 bytes for main loop
  let simdEnd ← ishlImm (← ushrImm n 2) 4   -- (n/4)*16 bytes for simd cleanup
  let scEnd   ← ishlImm n 2                   -- n*4 bytes for scalar tail
  let zero    ← fconst32 f32Zero
  let acc0    ← splat .f32x4 zero
  let i0      ← iconst64 0
  -- Main loop: 4x unrolled (16 floats/iter, 4 accumulators)
  let loop1 ← declareBlock [.i64, .f32x4, .f32x4, .f32x4, .f32x4]
  let body1 ← declareBlock [.i64, .f32x4, .f32x4, .f32x4, .f32x4]
  let merge ← declareBlock [.i64, .f32x4, .f32x4, .f32x4, .f32x4]
  jump loop1.ref [i0, acc0, acc0, acc0, acc0]
  startBlock loop1
  let i1 := loop1.param 0; let a1 := loop1.param 1; let b1 := loop1.param 2
  let c1 := loop1.param 3; let d1 := loop1.param 4
  brif (← icmp .sge i1 mainEnd) merge.ref [i1, a1, b1, c1, d1] body1.ref [i1, a1, b1, c1, d1]
  startBlock body1
  let i2 := body1.param 0; let a2 := body1.param 1; let b2 := body1.param 2
  let c2 := body1.param 3; let d2 := body1.param 4
  let aOff0 ← iadd dataPtr i2;  let bOff0 ← iadd bPtr i2
  let a2'   ← fadd a2 (← fadd (← loadF32x4 aOff0) (← loadF32x4 bOff0))
  let aOff1 ← iaddImm aOff0 16; let bOff1 ← iaddImm bOff0 16
  let b2'   ← fadd b2 (← fadd (← loadF32x4 aOff1) (← loadF32x4 bOff1))
  let aOff2 ← iaddImm aOff0 32; let bOff2 ← iaddImm bOff0 32
  let c2'   ← fadd c2 (← fadd (← loadF32x4 aOff2) (← loadF32x4 bOff2))
  let aOff3 ← iaddImm aOff0 48; let bOff3 ← iaddImm bOff0 48
  let d2'   ← fadd d2 (← fadd (← loadF32x4 aOff3) (← loadF32x4 bOff3))
  jump loop1.ref [← iaddImm i2 64, a2', b2', c2', d2']
  -- Merge 4 accumulators → 1
  startBlock merge
  let i3 := merge.param 0; let a3 := merge.param 1; let b3 := merge.param 2
  let c3 := merge.param 3; let d3 := merge.param 4
  let ab  ← fadd a3 b3
  let cd  ← fadd c3 d3
  let acc ← fadd ab cd
  -- SIMD cleanup: 1 vector at a time
  let loop2 ← declareBlock [.i64, .f32x4]
  let body2 ← declareBlock [.i64, .f32x4]
  let hred  ← declareBlock [.i64, .f32x4]
  jump loop2.ref [i3, acc]
  startBlock loop2
  let i4 := loop2.param 0; let v4 := loop2.param 1
  brif (← icmp .sge i4 simdEnd) hred.ref [i4, v4] body2.ref [i4, v4]
  startBlock body2
  let i5 := body2.param 0; let v5 := body2.param 1
  let aOff ← iadd dataPtr i5; let bOff ← iadd bPtr i5
  jump loop2.ref [← iaddImm i5 16, ← fadd v5 (← fadd (← loadF32x4 aOff) (← loadF32x4 bOff))]
  -- Horizontal reduce f32x4 → f64
  startBlock hred
  let i6 := hred.param 0; let v6 := hred.param 1
  let sum64 ← fadd (← fadd (← fpromote (← extractlane v6 0))
                            (← fpromote (← extractlane v6 1)))
                   (← fadd (← fpromote (← extractlane v6 2))
                            (← fpromote (← extractlane v6 3)))
  -- Scalar tail
  let loop3 ← declareBlock [.i64, .f64]
  let body3 ← declareBlock [.i64, .f64]
  let done  ← declareBlock [.f64]
  jump loop3.ref [i6, sum64]
  startBlock loop3
  let i7 := loop3.param 0; let s7 := loop3.param 1
  brif (← icmp .sge i7 scEnd) done.ref [s7] body3.ref [i7, s7]
  startBlock body3
  let i8 := body3.param 0; let s8 := body3.param 1
  let aOff ← iadd dataPtr i8; let bOff ← iadd bPtr i8
  jump loop3.ref [← iaddImm i8 4, ← fadd s8 (← fpromote (← fadd (← loadF32 aOff) (← loadF32 bOff)))]
  -- Store result
  startBlock done
  storeF64 (done.param 0) outPtr
  ret

def clifIR : String := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "vecops_algorithm" {
    cranelift_ir := clifIR,
    memory_size := 40,
    context_offset := 0
  } {
    actions := mkCallActions 1,
    cranelift_units := 0,
    timeout_ms := some TIMEOUT_MS
  }]

end VecOpsBench
