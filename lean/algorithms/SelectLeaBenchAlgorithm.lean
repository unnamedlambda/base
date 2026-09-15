import Lean
import AlgorithmLib.Gen

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace SelectLeaBench

/-
  Input: n f32 values, read as raw i32 words.  Result: f64.

  Every other sweep here has a straight-line loop body, so nothing has tested
  block layout or what either backend does with a branch.  This one is a
  serial state chain with a data-dependent, deliberately unpredictable branch:

      h = if x is even then (h + x) & M else (h * 3) & M

  The dependency on `h` rules out vectorisation on both sides, and the low bit
  of random float data mispredicts about half the time -- the profile of the
  parsers and decoders that stay on the CPU.

  `SelectBench` with one further change: `h * 3` is emitted as `(h << 1) + h`.
  A multiply by a small constant is three cycles on x86 and a shift-add is one,
  which LLVM knows and does for free; Cranelift emits the `imul` it was asked
  for.  On a serial chain that difference is the whole loop.  Strength
  reduction is a generator-side decision here, not something to wait for.
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
  let three   ← iconst64 3
  let keep    ← iconst64 0xFFFFFF

  let fin ← wloop2 i0 h0
    (head := fun i h => return (exitIfSGe i mainEnd, %[h], ()))
    (body := fun bi bh _ => do
      let x  ← uload32_64 (← iadd dataPtr bi)
      let hE ← band (← iadd bh x) keep
      let hO ← band (← iadd (← ishlImm bh 1) bh) keep
      let c  ← icmp .eq (← band x one) (← iconst64 0)
      return %[← iaddImm bi 4, ← select c hE hO])

  store (← fcvtFromSint .f64 (fin.head)) outPtr


def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 code]

def artifacts (clif : List FuncData) : Array Json :=
  #[toJsonArtifact "selectlea_algorithm" {
    functions := clif, memory_size := MEM_SIZE
  }]

end SelectLeaBench
