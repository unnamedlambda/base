import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace BranchyBench

/-
  Input: n f32 values, read as raw i32 words.  Result: f64.

  Every other sweep here has a straight-line loop body, so nothing has tested
  block layout or what either backend does with a branch.  This one is a
  serial state chain with a data-dependent, deliberately unpredictable branch:

      h = if x is even then (h + x) & M else (h * 3) & M

  The dependency on `h` rules out vectorisation on both sides, and the low bit
  of random float data mispredicts about half the time -- the profile of the
  parsers and decoders that stay on the CPU.

  `SelectBench` is the same computation written branchlessly.  The pair prices
  what the branch costs, the same way `clamp` and `pmin` price `fmin`.
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
  let three   ← iconst64 3
  let keep    ← iconst64 0xFFFFFF

  let loop ← declareBlock [.i64, .i64]
  let body ← declareBlock [.i64, .i64]
  let evn  ← declareBlock [.i64, .i64, .i64]
  let odd  ← declareBlock [.i64, .i64]
  let fin  ← declareBlock [.i64]
  jump loop.ref [i0, h0]

  startBlock loop
  brif (← icmp .sge (loop.param 0) mainEnd)
    fin.ref [loop.param 1] body.ref [loop.param 0, loop.param 1]

  startBlock body
  let bi := body.param 0
  let bh := body.param 1
  let x  ← uload32_64 (← iadd dataPtr bi)
  brif (← icmpImm .eq (← band x one) 0)
    evn.ref [bi, bh, x] odd.ref [bi, bh]

  startBlock evn
  jump loop.ref [← iaddImm (evn.param 0) 4,
                 ← band (← iadd (evn.param 1) (evn.param 2)) keep]

  startBlock odd
  jump loop.ref [← iaddImm (odd.param 0) 4,
                 ← band (← imul (odd.param 1) three) keep]

  startBlock fin
  storeF64 (← fcvtFromSint .f64 (fin.param 0)) outPtr
  ret

def clifIR : Program := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "branchy_algorithm" {
    clif := clifIR, memory_size := MEM_SIZE
  } { fn_idx := u32 1 }]

end BranchyBench
