import Lean
import AlgorithmLib.Gen

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
  let zero    ← fconst32 f32Zero
  let acc0    ← splat .f32x4 zero
  let i0      ← iconst64 0

  -- Four vector accumulators, 64 bytes a trip.
  let m ← wloop %[i0, acc0, acc0, acc0, acc0]
    (head := fun c => return (exitIfSGe (c.head) mainEnd, c, ()))
    (body := fun c _ => do
      let off0 ← iadd dataPtr (c.head)
      let a2' ← fadd (c.snd) (← loadF32x4 off0)
      let b2' ← fadd (c.thd) (← loadF32x4 (← iaddImm off0 16))
      let c2' ← fadd (c.fth) (← loadF32x4 (← iaddImm off0 32))
      let d2' ← fadd (c.fif) (← loadF32x4 (← iaddImm off0 48))
      return %[← iaddImm (c.head) 64, a2', b2', c2', d2'])
  let ab  ← fadd (m.snd) (m.thd)
  let cd  ← fadd (m.fth) (m.fif)
  let acc ← fadd ab cd

  -- One vector at a time for the tail of the unroll.
  let h ← wloop2 (m.head) acc
    (head := fun i v => return (exitIfSGe i simdEnd, %[i, v], ()))
    (body := fun i v _ => do
      let off ← iadd dataPtr i
      return %[← iaddImm i 16, ← fadd v (← loadF32x4 off)])
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
      return %[← iaddImm i 4, ← fadd s (← fpromote (← loadF32 aOff))])

  store (d.head) outPtr


def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code)]

def artifacts (clif : List FuncData) : Array Json :=
  #[toJsonArtifact "reduction_algorithm" {
    functions := clif,
    memory_size := 40
  }]

end ReductionBench
