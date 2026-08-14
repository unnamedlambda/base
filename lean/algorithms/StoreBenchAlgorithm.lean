import Lean
import AlgorithmLib.Gen

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace StoreBench

/-
  Input: n f32 values.  Output: n f32 values, each doubled.

  Every other sweep here reduces to a scalar, so the loop reads and never
  writes.  This one writes as much as it reads, which is the shape most real
  kernels have: it exercises store lowering, the addressing mode the store
  uses, and whether the loop is limited by store ports rather than arithmetic.

  Four f32x4 stores per trip -- 16 elements, 64 bytes.  The trailing
  `n % 16` elements are left alone on both sides.
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
  let n       ← ushrImm dataLen 2
  -- floor(n/16) trips of 16 elements = 64 bytes each
  let mainEnd ← ishlImm (← ushrImm n 4) 6
  let two     ← fconst32 2.0
  let twoV    ← splat .f32x4 two

  let _ ← wloop1 (← iconst64 0)
    (head := fun i => return (exitIfSGe i mainEnd, ([] : List R), ()))
    (body := fun bi _ => do
      let src ← iadd dataPtr bi
      let dst ← iadd outPtr bi
      for k in [0:4] do
        let v ← loadF32x4 (← iaddImm src (16 * k))
        storeUnaligned (← fmul v twoV) (← iaddImm dst (16 * k))
      return [← iaddImm bi 64])

theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  IR.program [noopFunction, HProg.compileFn 1 code]

def artifacts : Array Json :=
  #[toJsonEntry "store_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end StoreBench
