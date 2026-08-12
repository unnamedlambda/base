import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace SelectRotBench

/-
  `SelectLeaBench` with the loop rotated.

  The other loops here are written head-tested: a header block re-checks the
  bound and the body ends with an unconditional jump back to it.  That costs a
  `jmp` plus a second branch every trip, and Cranelift rematerialises the
  invariant bound in the header because it does no LICM -- it is a backend, not
  an optimiser.  None of that sits on the dependency chain, but the loop is
  front-end bound, so instructions still cost.

  Rotating it -- guard once on entry, test at the bottom, branch straight back
  into the body -- removes the header block entirely.  It is a choice about
  what CLIF to emit, which is where every other fix in this file lives too.
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
  let hE ← band (← iadd bh x) keep
  let hO ← band (← iadd (← ishlImm bh 1) bh) keep
  let c  ← icmpImm .eq (← band x one) 0
  let h' ← select' c hE hO
  let i' ← iaddImm bi 4
  brif (← icmp .slt i' mainEnd) body.ref [i', h'] fin.ref [h']

  startBlock fin
  storeF64 (← fcvtFromSint .f64 (fin.param 0)) outPtr
  ret

def clifIR : Program := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "selectrot_algorithm" {
    clif := clifIR, memory_size := MEM_SIZE
  } { fn_idx := u32 1 }]

end SelectRotBench
