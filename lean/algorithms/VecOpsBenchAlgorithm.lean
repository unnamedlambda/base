import Lean
import AlgorithmLib.Gen

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
    0x0000..0x0027  reserved (no context slot is used; the buffers arrive as entry arguments)

  The CLIF code reads arrays directly from the data pointer (zero copy)
  and writes the result to the out pointer (zero copy).

  CLIF: 4x-unrolled SIMD vec_add of two f32 arrays → f64 sum.
  Main loop: 4 independent f32x4 accumulators (16 floats/iter) for ILP.
-/


open AlgorithmLib.Prog


def code : Prog V L Unit := do
  -- Load data_ptr, data_len, out_ptr from reserved region
  let dataPtr ← dataPtr
  let dataLen ← dataLen
  let outPtr  ← outPtr
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
  let m ← wloop %[i0, acc0, acc0, acc0, acc0]
    (head := fun c => return (exitIfSGe (c.head) mainEnd, c, ()))
    (body := fun c _ => do
      let i2 := c.head
      let aOff0 ← iadd dataPtr i2;  let bOff0 ← iadd bPtr i2
      let a2'   ← fadd (c.snd) (← fadd (← loadF32x4 aOff0) (← loadF32x4 bOff0))
      let aOff1 ← iaddImm aOff0 16; let bOff1 ← iaddImm bOff0 16
      let b2'   ← fadd (c.thd) (← fadd (← loadF32x4 aOff1) (← loadF32x4 bOff1))
      let aOff2 ← iaddImm aOff0 32; let bOff2 ← iaddImm bOff0 32
      let c2'   ← fadd (c.fth) (← fadd (← loadF32x4 aOff2) (← loadF32x4 bOff2))
      let aOff3 ← iaddImm aOff0 48; let bOff3 ← iaddImm bOff0 48
      let d2'   ← fadd (c.fif) (← fadd (← loadF32x4 aOff3) (← loadF32x4 bOff3))
      return %[← iaddImm i2 64, a2', b2', c2', d2'])

  -- Merge 4 accumulators → 1
  let ab  ← fadd (m.snd) (m.thd)
  let cd  ← fadd (m.fth) (m.fif)
  let acc ← fadd ab cd

  -- SIMD cleanup: 1 vector at a time
  let h ← wloop2 (m.head) acc
    (head := fun i v => return (exitIfSGe i simdEnd, %[i, v], ()))
    (body := fun i v _ => do
      let aOff ← iadd dataPtr i; let bOff ← iadd bPtr i
      return %[← iaddImm i 16, ← fadd v (← fadd (← loadF32x4 aOff) (← loadF32x4 bOff))])

  -- Horizontal reduce f32x4 → f64
  let v6 := h.snd
  let sum64 ← fadd (← fadd (← fpromote (← extractlane v6 0))
                            (← fpromote (← extractlane v6 1)))
                   (← fadd (← fpromote (← extractlane v6 2))
                            (← fpromote (← extractlane v6 3)))

  -- Scalar tail
  let d ← wloop2 (h.head) sum64
    (head := fun i s => return (exitIfSGe i scEnd, %[s], ()))
    (body := fun i s _ => do
      let aOff ← iadd dataPtr i; let bOff ← iadd bPtr i
      return %[← iaddImm i 4,
              ← fadd s (← fpromote (← fadd (← loadF32 aOff) (← loadF32 bOff)))])

  -- Store result
  store (d.head) outPtr


def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 code]

def artifacts (clif : List FuncData) : Array Json :=
  #[toJsonArtifact "vecops_algorithm" {
    functions := clif,
    memory_size := 40
  }]

end VecOpsBench
