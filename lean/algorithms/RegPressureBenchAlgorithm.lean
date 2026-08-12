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

def mainFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr  ← load64 (← absAddr ptr 0x28)
  let n       ← ushrImm dataLen 2            -- element count
  -- floor(n/64) trips of 64 elements = 256 bytes each
  let mainEnd ← ishlImm (← ushrImm n 6) 8
  let zero    ← fconst32 f32Zero
  let acc0    ← splat .f32x4 zero
  let i0      ← iconst64 0

  let tys   := ClifTy.i64 :: List.replicate ACCS ClifTy.f32x4
  let loop  ← declareBlock tys
  let body  ← declareBlock tys
  let fin   ← declareBlock tys
  jump loop.ref (i0 :: List.replicate ACCS acc0)

  startBlock loop
  let li := loop.param 0
  let laccs := (List.range ACCS).map (fun k => loop.param (k + 1))
  brif (← icmp .sge li mainEnd) fin.ref (li :: laccs) body.ref (li :: laccs)

  startBlock body
  let bi := body.param 0
  let off ← iadd dataPtr bi
  let mut accs' : List Val := []
  for k in [0:ACCS] do
    let v ← loadF32x4 (← iaddImm off (16 * k))
    accs' := accs' ++ [← fadd (body.param (k + 1)) v]
  jump loop.ref ((← iaddImm bi 256) :: accs')

  startBlock fin
  -- left fold, so the Rust mirror can reproduce the order exactly
  let mut acc := fin.param 1
  for k in [1:ACCS] do
    acc ← fadd acc (fin.param (k + 1))
  let sum64 ← fadd (← fadd (← fpromote (← extractlane acc 0))
                            (← fpromote (← extractlane acc 1)))
                   (← fadd (← fpromote (← extractlane acc 2))
                            (← fpromote (← extractlane acc 3)))
  storeF64 sum64 outPtr
  ret

def clifIR : Program := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "regpressure_sum_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end RegPressureBench
