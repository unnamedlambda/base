import Lean
import AlgorithmLib.Gen

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace MatmulBench

/-
  SIMD f32 matrix multiplication benchmark.

  Payload: [M:u32][K:u32][N:u32][A: M*K f32][B: K*N f32]
  Output:  [result: f64] — sum of all elements of C = A*B

  CLIF: SIMD matmul with 4x-unrolled j-loop (16 cols/iter).
-/


open AlgorithmLib.Prog


def code : Prog V L Unit := do
  let dataPtr ← dataPtr
  let outPtr  ← outPtr
  -- Parse M, K, N
  let mVal32  ← load32 dataPtr
  let kVal32  ← load32 (← iaddImm dataPtr 4)
  let nVal32  ← load32 (← iaddImm dataPtr 8)
  let mVal    ← sextend64 mVal32
  let kVal    ← sextend64 kVal32
  let nVal    ← sextend64 nVal32
  -- A starts at dataPtr+12, B starts at A + M*K*4
  let aPtr    ← iaddImm dataPtr 12
  let mk      ← imul mVal kVal
  let bPtr    ← iadd aPtr (← ishlImm mk 2)
  let nBytes  ← ishlImm nVal 2          -- N*4 (row stride of B)
  let zero64  ← iconst64 0
  let zero    ← fconst64 f64Zero
  let zeroV   ← splat .f32x4 (← fconst32 f32Zero)

  -- Only what actually changes is carried: `i` is in scope inside the p-loop
  -- and `aVec`/`bRow` inside the j-loop, so they need not be block parameters.
  let outer ← wloop2 zero64 zero
    (head := fun i total => return (exitIfSGe i mVal, %[total], ()))
    (body := fun i total _ => do
      let inner ← wloop2 zero64 total
        (head := fun p tot => return (exitIfSGe p kVal, %[tot], ()))
        (body := fun p tot _ => do
          -- Load A[i,p]: aPtr + (i*K+p)*4
          let ikp  ← iadd (← imul i kVal) p
          let aOff ← iadd aPtr (← ishlImm ikp 2)
          let aVal ← loadF32 aOff
          let aVec ← splat .f32x4 aVal
          -- B row: bPtr + p*N*4
          let bRow ← iadd bPtr (← ishlImm (← imul p nVal) 2)
          let js ← wloop %[zero64, zeroV, zeroV, zeroV, zeroV]
            (head := fun c => return (exitIfSGe (c.head) nBytes, c.tail, ()))
            (body := fun c _ => do
              let bBase ← iadd bRow (c.head)
              let c0' ← fadd (c.snd) (← fmul aVec (← loadF32x4 bBase))
              let c1' ← fadd (c.thd) (← fmul aVec (← loadF32x4 (← iaddImm bBase 16)))
              let c2' ← fadd (c.fth) (← fmul aVec (← loadF32x4 (← iaddImm bBase 32)))
              let c3' ← fadd (c.fif) (← fmul aVec (← loadF32x4 (← iaddImm bBase 48)))
              return %[← iaddImm (c.head) 64, c0', c1', c2', c3'])
          let vAcc ← fadd (← fadd (js.head) (js.snd))
                          (← fadd (js.thd) (js.fth))
          let rowSum ← fadd (← fadd (← fpromote (← extractlane vAcc 0))
                                     (← fpromote (← extractlane vAcc 1)))
                            (← fadd (← fpromote (← extractlane vAcc 2))
                                     (← fpromote (← extractlane vAcc 3)))
          return %[← iaddImm p 1, ← fadd tot rowSum])
      return %[← iaddImm i 1, inner.head])

  store (outer.head) outPtr


def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code)]

def artifacts (clif : List FuncData) : Array Json :=
  #[toJsonArtifact "matmul_algorithm" {
    functions := clif,
    memory_size := 40
  }]

end MatmulBench
