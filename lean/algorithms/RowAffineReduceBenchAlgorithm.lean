import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

namespace RowAffineReduceBench

/-
  Input: [rows:u64][cols:u64][X: rows*cols*f32][scale: cols*f32][bias: cols*f32]
  Output: [y: rows*f32] where y[i] = sum_j (X[i,j] * scale[j] + bias[j])
  4x-unrolled SIMD inner loop with scalar tail.
-/

def MEM_SIZE : Nat := 40

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- Nothing here crosses the FFI. -/
def env : FnEnv := { sigs := [], fns := [] }

def code : HProg.Code := clif% do
  let ptr := basePtr
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

  -- One row a trip. `xPtrI` does not change inside the row, so it stays in
  -- scope rather than becoming a block parameter of all three inner loops.
  let _ ← wloop1 i0
    (head := fun i => return (exitIfSGe i rows, ([] : List R), ()))
    (body := fun i _ => do
      let iCols ← imul i cols
      let xPtrI ← iadd xPtr (← ishlImm iCols 2)
      let j0    ← iconst64 0

      -- 4x-unrolled main SIMD loop (64 bytes a trip)
      let m ← wloop [j0, zeroV, zeroV, zeroV, zeroV]
        (head := fun c => return (exitIfSGe (c.headD 0) mainEnd, c, ()))
        (body := fun c _ => do
          let jO := c.headD 0
          let xA ← iadd xPtrI jO; let sA ← iadd scalePtr jO; let bA ← iadd biasPtr jO
          let x0 ← loadF32x4 xA; let s0 ← loadF32x4 sA; let b0 ← loadF32x4 bA
          let a0' ← fadd (c.getD 1 0) (← fadd (← fmul x0 s0) b0)
          let xA1 ← iaddImm xA 16; let sA1 ← iaddImm sA 16; let bA1 ← iaddImm bA 16
          let x1 ← loadF32x4 xA1; let s1 ← loadF32x4 sA1; let b1 ← loadF32x4 bA1
          let a1' ← fadd (c.getD 2 0) (← fadd (← fmul x1 s1) b1)
          let xA2 ← iaddImm xA 32; let sA2 ← iaddImm sA 32; let bA2 ← iaddImm bA 32
          let x2 ← loadF32x4 xA2; let s2 ← loadF32x4 sA2; let b2 ← loadF32x4 bA2
          let a2' ← fadd (c.getD 3 0) (← fadd (← fmul x2 s2) b2)
          let xA3 ← iaddImm xA 48; let sA3 ← iaddImm sA 48; let bA3 ← iaddImm bA 48
          let x3 ← loadF32x4 xA3; let s3 ← loadF32x4 sA3; let b3 ← loadF32x4 bA3
          let a3' ← fadd (c.getD 4 0) (← fadd (← fmul x3 s3) b3)
          return [← iaddImm jO 64, a0', a1', a2', a3'])
      let acc ← fadd (← fadd (m.getD 1 0) (m.getD 2 0))
                     (← fadd (m.getD 3 0) (m.getD 4 0))

      -- one vector a trip for the rest of the unroll
      let h ← wloop2 (m.headD 0) acc
        (head := fun jO a => return (exitIfSGe jO simdEnd, [jO, a], ()))
        (body := fun jO a _ => do
          let xA ← iadd xPtrI jO; let sA ← iadd scalePtr jO; let bA ← iadd biasPtr jO
          let xV ← loadF32x4 xA; let sV ← loadF32x4 sA; let bV ← loadF32x4 bA
          return [← iaddImm jO 16, ← fadd a (← fadd (← fmul xV sV) bV)])

      let acc4 := h.getD 1 0
      let e0 ← extractlane acc4 0; let e1 ← extractlane acc4 1
      let e2 ← extractlane acc4 2; let e3 ← extractlane acc4 3
      let sum ← fadd (← fadd e0 e1) (← fadd e2 e3)

      -- and scalar for whatever is left
      let d ← wloop2 (h.headD 0) sum
        (head := fun jO sAcc => return (exitIfSGe jO stride, [sAcc], ()))
        (body := fun jO sAcc _ => do
          let xA ← iadd xPtrI jO; let sA ← iadd scalePtr jO; let bA ← iadd biasPtr jO
          let xS ← loadF32 xA; let sS ← loadF32 sA; let bS ← loadF32 bA
          return [← iaddImm jO 4, ← fadd sAcc (← fadd (← fmul xS sS) bS)])

      let oAddr ← iadd outPtr (← ishlImm i 2)
      store (d.headD 0) oAddr
      return [← iaddImm i 1])

theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  IR.program [noopFunction, HProg.compileFn 1 code]

def artifacts : Array Json :=
  #[toJsonEntry "row_affine_reduce_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end RowAffineReduceBench
