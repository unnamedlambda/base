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

open AlgorithmLib.Prog


def code : Prog V L Unit := do
  let dataPtr ← dataPtr
  let dataLen ← dataLen
  let outPtr  ← outPtr
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
  let m ← wloop %[i0, acc0, acc0, acc0, acc0]
    (head := fun c => return (exitIfSGe (c.head) mainEnd, c, ()))
    (body := fun c _ => do
      let off0 ← iadd dataPtr (c.head)
      let clamp := fun v acc => do
        let t ← fmin v hiV; let t' ← fmax t loV; fadd acc t'
      let a2' ← clamp (← loadF32x4 off0) (c.snd)
      let b2' ← clamp (← loadF32x4 (← iaddImm off0 16)) (c.thd)
      let c2' ← clamp (← loadF32x4 (← iaddImm off0 32)) (c.fth)
      let d2' ← clamp (← loadF32x4 (← iaddImm off0 48)) (c.fif)
      return %[← iaddImm (c.head) 64, a2', b2', c2', d2'])
  let acc ← fadd (← fadd (m.snd) (m.thd))
                 (← fadd (m.fth) (m.fif))

  -- One vector at a time for the tail of the unroll.
  let h ← wloop2 (m.head) acc
    (head := fun i v => return (exitIfSGe i simdEnd, %[i, v], ()))
    (body := fun i v _ => do
      let off ← iadd dataPtr i
      let x  ← loadF32x4 off
      let t  ← fmin x hiV; let t' ← fmax t loV
      return %[← iaddImm i 16, ← fadd v t'])
  let v6 := h.snd
  let sum64 ← fadd (← fadd (← fpromote (← extractlane v6 0))
                            (← fpromote (← extractlane v6 1)))
                   (← fadd (← fpromote (← extractlane v6 2))
                            (← fpromote (← extractlane v6 3)))

  -- And scalar for whatever is left.
  let d ← wloop2 (h.head) sum64
    (head := fun i s => return (exitIfSGe i scEnd, %[s], ()))
    (body := fun i s _ => do
      let aOff ← iadd dataPtr i
      let x    ← loadF32 aOff
      let t    ← fmin x hi; let t' ← fmax t lo
      return %[← iaddImm i 4, ← fadd s (← fpromote t')])

  store (d.head) outPtr


def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code)]

def artifacts (clif : List FuncData) : Array Json :=
  #[toJsonArtifact "clamp_sum_algorithm" {
    functions := clif,
    memory_size := MEM_SIZE
  }]

end ClampSumBench
