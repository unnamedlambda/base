import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace SelectMaskBench

/-
  `SelectRotBench` with the masking hoisted out of the loop.

  Add and multiply only carry information upward in bit position, so the low
  24 bits of the result never depend on anything above bit 24 of the inputs.
  Masking every step is therefore redundant: let the chain wrap mod 2^64 and
  mask once at the end, and the answer is identical.

  LLVM derives this on its own -- it is demanded-bits analysis -- and it is
  worth two thirds of a cycle here, because it takes the dependency chain from
  `lea -> and -> cmov` down to `lea -> cmov`.  Cranelift does not derive it, so
  the generator has to know it.  That is the shape of the whole tradeoff:
  reachable from above, but only if someone reaches.
-/

def MEM_SIZE : Nat := 40

def mainFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr  ← load64 (← absAddr ptr 0x28)
  let n       ← ushrImm dataLen 2
  let mainEnd ← ishlImm n 2
  let i0      ← iconst64 0
  let h0      ← iconst64 1
  let one     ← iconst64 1
  let keep    ← iconst64 0xFFFFFF

  let body ← declareBlock [.i64, .i64]
  let fin  ← declareBlock [.i64]
  -- guard once, so an empty input skips the loop rather than testing inside it
  brif (← icmp .slt i0 mainEnd) body.ref [i0, h0] fin.ref [h0]

  startBlock body
  let bi := body.param 0
  let bh := body.param 1
  let x  ← uload32_64 (← iadd dataPtr bi)
  let hE ← iadd bh x
  let hO ← iadd (← ishlImm bh 1) bh
  let c  ← icmpImm .eq (← band x one) 0
  let h' ← select' c hE hO
  let i' ← iaddImm bi 4
  brif (← icmp .slt i' mainEnd) body.ref [i', h'] fin.ref [h']

  startBlock fin
  storeF64 (← fcvtFromSint .f64 (← band (fin.param 0) keep)) outPtr
  ret

def clifIR : Program := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "selectmask_algorithm" {
    clif := clifIR, memory_size := MEM_SIZE
  } { fn_idx := u32 1 }]

end SelectMaskBench
