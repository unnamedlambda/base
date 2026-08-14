import Lean
import AlgorithmLib.Gen

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

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- Nothing here crosses the FFI. -/
def env : FnEnv := { sigs := [], fns := [] }

def code : HProg.Code := clif% do
  let dataPtr ← load64 (← absAddr basePtr 0x18)
  let dataLen ← load64 (← absAddr basePtr 0x20)
  let outPtr  ← load64 (← absAddr basePtr 0x28)
  -- n = data_len / 4
  let n       ← ushrImm dataLen 2
  let mainEnd ← ishlImm (← ushrImm n 4) 6   -- (n/16)*64
  let simdEnd ← ishlImm (← ushrImm n 2) 4   -- (n/4)*16
  let scEnd   ← ishlImm n 2                  -- n*4
  -- Constants: hi=0.5, lo=-0.5
  let hi      ← fconst32 0.5
  let lo      ← fneg hi
  let hiV     ← splat .f32x4 hi
  let loV     ← splat .f32x4 lo
  let zero    ← fconst32 f32Zero
  let acc0    ← splat .f32x4 zero
  let i0      ← iconst64 0

  -- Four vector accumulators, 64 bytes a trip.
  let m ← wloop [i0, acc0, acc0, acc0, acc0]
    (head := fun c => return (exitIfSGe (c.headD 0) mainEnd, c, ()))
    (body := fun c _ => do
      let off0 ← iadd dataPtr (c.headD 0)
      let clamp := fun (v acc : R) => do
        let t ← fmin v hiV; let t' ← fmax t loV; fadd acc t'
      let a2' ← clamp (← loadF32x4 off0) (c.getD 1 0)
      let b2' ← clamp (← loadF32x4 (← iaddImm off0 16)) (c.getD 2 0)
      let c2' ← clamp (← loadF32x4 (← iaddImm off0 32)) (c.getD 3 0)
      let d2' ← clamp (← loadF32x4 (← iaddImm off0 48)) (c.getD 4 0)
      return [← iaddImm (c.headD 0) 64, a2', b2', c2', d2'])
  let acc ← fadd (← fadd (m.getD 1 0) (m.getD 2 0))
                 (← fadd (m.getD 3 0) (m.getD 4 0))

  -- One vector at a time for the tail of the unroll.
  let h ← wloop2 (m.headD 0) acc
    (head := fun i v => return (exitIfSGe i simdEnd, [i, v], ()))
    (body := fun i v _ => do
      let off ← iadd dataPtr i
      let x  ← loadF32x4 off
      let t  ← fmin x hiV; let t' ← fmax t loV
      return [← iaddImm i 16, ← fadd v t'])
  let v6 := h.getD 1 0
  let sum64 ← fadd (← fadd (← fpromote (← extractlane v6 0))
                            (← fpromote (← extractlane v6 1)))
                   (← fadd (← fpromote (← extractlane v6 2))
                            (← fpromote (← extractlane v6 3)))

  -- And scalar for whatever is left.
  let d ← wloop2 (h.headD 0) sum64
    (head := fun i s => return (exitIfSGe i scEnd, [s], ()))
    (body := fun i s _ => do
      let aOff ← iadd dataPtr i
      let x    ← loadF32 aOff
      let t    ← fmin x hi; let t' ← fmax t lo
      return [← iaddImm i 4, ← fadd s (← fpromote t')])

  store (d.headD 0) outPtr

theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  IR.program [noopFunction, HProg.compileFn 1 code]

def artifacts : Array Json :=
  #[toJsonEntry "clamp_sum_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end ClampSumBench
