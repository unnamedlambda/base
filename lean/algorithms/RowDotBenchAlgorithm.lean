import Lean
import AlgorithmLib.Gen

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

namespace RowDotBench

/-
  Row-wise dot product: y[i] = dot(X[i,:], w)
  Payload: [rows:u64][cols:u64][X: rows*cols*f32][w: cols*f32]
  Output:  [y: rows*f32]
  Dual-row path for pairs, single-row for last row. SIMD inner loop + scalar tail.
-/

def MEM_SIZE : Nat := 40

open AlgorithmLib.Prog


def code : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let outPtr ← outPtr
  let rows   ← load64 dataPtr
  let cols   ← load64 (← iaddImm dataPtr 8)
  let xPtr   ← iaddImm dataPtr 16
  let wPtr   ← iadd xPtr (← ishlImm (← imul rows cols) 2)
  let stride ← ishlImm cols 2            -- cols * 4 (row byte size)
  let simdEnd← ishlImm (← ushrImm cols 2) 4  -- (cols/4)*16

  let zero   ← iconst64 0
  let zeroF  ← fconst32 f32Zero
  let zeroV  ← splat .f32x4 zeroF

  -- Two rows a trip while a pair is left, one for the last. Each arm takes the
  -- back edge itself, with its own increment, so there is no join between the
  -- dispatch and the loop header.
  let _ ← wloopL %[zero]
    (head := fun _ c => return (exitIfSGe c.head rows, %[], ()))
    (body := fun rowLoop c _ => do
      let i := c.head
      let i1 ← iaddImm i 1
      let _ ← ifte (jTys := []) .slt i1 rows
        (thn := do
          -- Dual-row: rows i and i+1 against one weight vector
          let rOff ← ishlImm (← imul i cols) 2
          let xA   ← iadd xPtr rOff
          let xB   ← iadd xA stride
          let v ← wloop %[zero, zeroV, zeroV]
            (head := fun c => return (exitIfSGe (c.head) simdEnd, c, ()))
            (body := fun c _ => do
              let jO := c.head
              let wV ← loadF32x4 (← iadd wPtr jO)
              let x0 ← loadF32x4 (← iadd xA jO)
              let x1 ← loadF32x4 (← iadd xB jO)
              return %[← iaddImm jO 16, ← fadd (c.snd) (← fmul x0 wV),
                                       ← fadd (c.thd) (← fmul x1 wV)])
          let a0 := v.snd
          let e00 ← extractlane a0 0; let e01 ← extractlane a0 1
          let e02 ← extractlane a0 2; let e03 ← extractlane a0 3
          let s0  ← fadd (← fadd e00 e01) (← fadd e02 e03)
          let a1 := v.thd
          let e10 ← extractlane a1 0; let e11 ← extractlane a1 1
          let e12 ← extractlane a1 2; let e13 ← extractlane a1 3
          let s1  ← fadd (← fadd e10 e11) (← fadd e12 e13)
          let t ← wloop %[v.head, s0, s1]
            (head := fun c => return (exitIfSGe (c.head) stride, c.tail, ()))
            (body := fun c _ => do
              let jO := c.head
              let wS ← loadF32 (← iadd wPtr jO)
              let y0 ← loadF32 (← iadd xA jO)
              let y1 ← loadF32 (← iadd xB jO)
              return %[← iaddImm jO 4, ← fadd (c.snd) (← fmul y0 wS),
                                      ← fadd (c.thd) (← fmul y1 wS)])
          store (t.head) (← iadd outPtr (← ishlImm i 2))
          store (t.snd) (← iadd outPtr (← ishlImm (← iaddImm i 1) 2))
          continueWith rowLoop %[← iaddImm i 2])
        (els := do
          -- Single row
          let rOff ← ishlImm (← imul i cols) 2
          let xA   ← iadd xPtr rOff
          let v ← wloop %[zero, zeroV]
            (head := fun c => return (exitIfSGe (c.head) simdEnd, c, ()))
            (body := fun c _ => do
              let jO := c.head
              let xV ← loadF32x4 (← iadd xA jO)
              let wV ← loadF32x4 (← iadd wPtr jO)
              return %[← iaddImm jO 16, ← fadd (c.snd) (← fmul xV wV)])
          let a := v.snd
          let e0 ← extractlane a 0; let e1 ← extractlane a 1
          let e2 ← extractlane a 2; let e3 ← extractlane a 3
          let sm ← fadd (← fadd e0 e1) (← fadd e2 e3)
          let t ← wloop %[v.head, sm]
            (head := fun c => return (exitIfSGe (c.head) stride, c.tail, ()))
            (body := fun c _ => do
              let jO := c.head
              let xS ← loadF32 (← iadd xA jO)
              let wS ← loadF32 (← iadd wPtr jO)
              return %[← iaddImm jO 4, ← fadd (c.snd) (← fmul xS wS)])
          store (t.head) (← iadd outPtr (← ishlImm i 2))
          continueWith rowLoop %[← iaddImm i 1])
      return %[i])


def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code)]

def artifacts (clif : List FuncData) : Array Json :=
  #[toJsonArtifact "row_dot_algorithm" {
    functions := clif,
    memory_size := MEM_SIZE
  }]

end RowDotBench
