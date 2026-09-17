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

open AlgorithmLib.Prog


def code : Prog V L Unit := do
  let dataPtr ← dataPtr
  let dataLen ← dataLen
  let outPtr  ← outPtr
  let n       ← ushrImm dataLen 2
  -- floor(n/16) trips of 16 elements = 64 bytes each
  let mainEnd ← ishlImm (← ushrImm n 4) 6
  let two     ← fconst32 2.0
  let twoV    ← splat .f32x4 two

  let _ ← wloop1 (← iconst64 0)
    (head := fun i => return (exitIfSGe i mainEnd, %[], ()))
    (body := fun bi _ => do
      let src ← iadd dataPtr bi
      let dst ← iadd outPtr bi
      for k in [0:4] do
        let v ← loadF32x4 (← iaddImm src (16 * k))
        storeUnaligned (← fmul v twoV) (← iaddImm dst (16 * k))
      return %[← iaddImm bi 64])


def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code)]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "store_algorithm" {
    functions := clif,
    memory_size := MEM_SIZE
  }]

end StoreBench
