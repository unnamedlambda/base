import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace RegPressureBench

/-
  Input: n f32 values.  Result: f64 — their sum.

  Same shape as PlainSum but with SIXTEEN live f32x4 accumulators instead of
  four.  x86-64 has sixteen xmm registers, so together with the pointer, the
  counter and the loop bound this cannot be held in registers and the allocator
  has to spill.  That is the point: four accumulators is the easy case for any
  allocator, and it says nothing about how regalloc2 compares to LLVM's greedy
  allocator once spilling starts.

  The trailing `n % 64` elements are ignored.  The Rust mirror ignores them
  too, so the two still agree bit for bit.
-/

def MEM_SIZE : Nat := 40
def ACCS : Nat := 16

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- Nothing here crosses the FFI. -/
def env : FnEnv := { sigs := [], fns := [] }

def code : HProg.Code := clif% env HProg.ptrParams do
  let dataPtr ← load64 (← absAddr basePtr 0x18)
  let dataLen ← load64 (← absAddr basePtr 0x20)
  let outPtr  ← load64 (← absAddr basePtr 0x28)
  let n       ← ushrImm dataLen 2            -- element count
  -- floor(n/64) trips of 64 elements = 256 bytes each
  let mainEnd ← ishlImm (← ushrImm n 6) 8
  let zero    ← fconst32 f32Zero
  let acc0    ← splat .f32x4 zero
  let i0      ← iconst64 0

  let fin ← wloop (i0 :: List.replicate ACCS acc0)
    (head := fun c => return (exitIfSGe (c.headD 0) mainEnd, c, ()))
    (body := fun c _ => do
      let bi := c.headD 0
      let off ← iadd dataPtr bi
      let mut accs' : List R := []
      for k in [0:ACCS] do
        let v ← loadF32x4 (← iaddImm off (16 * k))
        accs' := accs' ++ [← fadd (c.getD (k + 1) 0) v]
      return (← iaddImm bi 256) :: accs')

  -- left fold, so the Rust mirror can reproduce the order exactly
  let mut acc := fin.getD 1 0
  for k in [1:ACCS] do
    acc ← fadd acc (fin.getD (k + 1) 0)
  let sum64 ← fadd (← fadd (← fpromote (← extractlane acc 0))
                            (← fpromote (← extractlane acc 1)))
                   (← fadd (← fpromote (← extractlane acc 2))
                            (← fpromote (← extractlane acc 3)))
  store sum64 outPtr

theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  IR.program [noopFunction, HProg.compileFn 1 env HProg.ptrParams code]

def artifacts : Array Json :=
  #[toJsonEntry "regpressure_sum_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end RegPressureBench
