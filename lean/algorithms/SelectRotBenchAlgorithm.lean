import Lean
import AlgorithmLib.Gen

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

open AlgorithmLib.Prog


def code : Prog V L Unit := do
  let dataPtr ← dataPtr
  let dataLen ← dataLen
  let outPtr  ← outPtr
  let n       ← ushrImm dataLen 2
  let mainEnd ← ishlImm n 2
  let i0      ← iconst64 0
  let h0      ← iconst64 1
  let one     ← iconst64 1
  let keep    ← iconst64 0xFFFFFF

  -- Bottom-tested: the guard runs once, so an empty input never enters and the
  -- body block branches to itself rather than through a header.
  let fin ← dwloop %[i0, h0] .slt mainEnd (contOnTrue := true) [1]
    (body := fun c => do
      let bi := c.head
      let bh := c.snd
      let x  ← uload32_64 (← iadd dataPtr bi)
      let hE ← band (← iadd bh x) keep
      let hO ← band (← iadd (← ishlImm bh 1) bh) keep
      let cnd ← icmp .eq (← band x one) (← iconst64 0)
      let h' ← select cnd hE hO
      let i' ← iaddImm bi 4
      return (i', %[i', h']))
    (guardIdx := some 0)

  store (← fcvtFromSint .f64 (fin.head)) outPtr


def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code)]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "selectrot_algorithm" {
    functions := clif, memory_size := MEM_SIZE
  }]

end SelectRotBench
