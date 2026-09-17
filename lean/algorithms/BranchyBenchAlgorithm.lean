import Lean
import AlgorithmLib.Gen

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

  -- Each arm takes the back edge itself, so there is no join between the
  -- branch and the loop header — which is the shape being measured.
  let fin ← wloopL %[i0, h0]
    (head := fun _ c => return (exitIfSGe c.head mainEnd, %[c.snd], ()))
    (body := fun lbl c _ => do
      let bi := c.head; let bh := c.snd
      let x ← uload32_64 (← iadd dataPtr bi)
      let _ ← ifte (jTys := []) .eq (← band x one) (← iconst64 0)
        (thn := do continueWith lbl %[← iaddImm bi 4, ← band (← iadd bh x) keep])
        (els := do continueWith lbl %[← iaddImm bi 4, ← band (← imul bh three) keep])
      return %[bi, bh])

  store (← fcvtFromSint .f64 (fin.head)) outPtr


def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code)]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "branchy_algorithm" {
    functions := clif, memory_size := MEM_SIZE
  }]

end BranchyBench
