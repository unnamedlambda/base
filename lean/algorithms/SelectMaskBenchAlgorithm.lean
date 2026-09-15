import Lean
import AlgorithmLib.Gen

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

open AlgorithmLib.Prog


def code : Prog V L Unit := do
  let dataPtr ← load64 (← absAddr (← basePtr) 0x18)
  let dataLen ← load64 (← absAddr (← basePtr) 0x20)
  let outPtr  ← load64 (← absAddr (← basePtr) 0x28)
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
      let hE ← iadd bh x
      let hO ← iadd (← ishlImm bh 1) bh
      let cnd ← icmp .eq (← band x one) (← iconst64 0)
      let h' ← select cnd hE hO
      let i' ← iaddImm bi 4
      return (i', %[i', h']))
    (guardIdx := some 0)

  store (← fcvtFromSint .f64 (← band (fin.head) keep)) outPtr


def clifIR : Except String Program :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 code]

def artifacts (clif : Program) : Array Json :=
  #[toJsonEntry "selectmask_algorithm" {
    clif, memory_size := MEM_SIZE
  } { fn_idx := u32 1 }]

end SelectMaskBench
