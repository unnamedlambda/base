import Lean
import AlgorithmLib.Gen

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace IntSumBench

/-
  Input: n f32 values, read as raw i32 words.  Result: f64 — a mixing sum.

  Everything else here is floating point, and integer code goes down a
  different path: different lowering rules, different registers, no vector
  unit involved.  Four independent chains of `h = h * 31 + (x & 0xFFFF)` keep
  the ALU busy without a serial dependency, so this measures integer
  throughput rather than multiply latency.

  Each chain is masked back to 24 bits every step -- without that `h * 31`
  runs away and wraps, and the f64 conversion at the end would be lossy.  Held
  under 2^24 the four chains sum to under 2^26, which f64 represents exactly,
  so the comparison stays bit-exact.

  The trailing `n % 4` elements are ignored, on both sides.
-/

def MEM_SIZE : Nat := 40

open AlgorithmLib.Prog


def code : Prog V L Unit := do
  let dataPtr ← load64 (← absAddr (← basePtr) 0x18)
  let dataLen ← load64 (← absAddr (← basePtr) 0x20)
  let outPtr  ← load64 (← absAddr (← basePtr) 0x28)
  let n       ← ushrImm dataLen 2
  -- floor(n/4) trips of 4 elements = 16 bytes each
  let mainEnd ← ishlImm (← ushrImm n 2) 4
  let i0      ← iconst64 0
  let h0      ← iconst64 1
  let k31     ← iconst64 31
  let mask    ← iconst64 0xFFFF
  let keep    ← iconst64 0xFFFFFF

  -- One counter and four independent hash chains, carried together.
  let fin ← wloop (Vals.cons i0 (Vals.ofFn (n := 4) (fun _ => h0)))
    (head := fun c => return (exitIfSGe c.head mainEnd, c, ()))
    (body := fun c _ => do
      let bi := c.head
      let off ← iadd dataPtr bi
      let hs ← c.tail.uniformMapIdxM fun k h => do
        let w ← uload32_64 (← iaddImm off (4 * k))
        let x ← band w mask
        band (← iadd (← imul h k31) x) keep
      return Vals.cons (← iaddImm bi 16) hs)

  -- left fold, matching the Rust mirror
  let hs := fin.tail.uniformToList
  let tot ← match hs with
    | [] => iconst64 0
    | h :: rest => rest.foldlM (fun x y => iadd x y) h
  store (← fcvtFromSint .f64 tot) outPtr


def clifIR : Except String Program :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 code]

def artifacts (clif : Program) : Array Json :=
  #[toJsonEntry "intsum_algorithm" {
    clif,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end IntSumBench
