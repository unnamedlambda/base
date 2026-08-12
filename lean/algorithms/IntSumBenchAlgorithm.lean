import Lean
import AlgorithmLib

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

def mainFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr  ← load64 (← absAddr ptr 0x28)
  let n       ← ushrImm dataLen 2
  -- floor(n/4) trips of 4 elements = 16 bytes each
  let mainEnd ← ishlImm (← ushrImm n 2) 4
  let i0      ← iconst64 0
  let h0      ← iconst64 1
  let k31     ← iconst64 31
  let mask    ← iconst64 0xFFFF
  let keep    ← iconst64 0xFFFFFF

  let tys  := [ClifTy.i64, .i64, .i64, .i64, .i64]
  let loop ← declareBlock tys
  let body ← declareBlock tys
  let fin  ← declareBlock tys
  jump loop.ref [i0, h0, h0, h0, h0]

  startBlock loop
  let li := loop.param 0
  let ls := (List.range 4).map (fun k => loop.param (k + 1))
  brif (← icmp .sge li mainEnd) fin.ref (li :: ls) body.ref (li :: ls)

  startBlock body
  let bi := body.param 0
  let off ← iadd dataPtr bi
  let mut hs : List Val := []
  for k in [0:4] do
    let w ← uload32_64 (← iaddImm off (4 * k))
    let x ← band w mask
    hs := hs ++ [← band (← iadd (← imul (body.param (k + 1)) k31) x) keep]
  jump loop.ref ((← iaddImm bi 16) :: hs)

  startBlock fin
  -- left fold, matching the Rust mirror
  let mut tot := fin.param 1
  for k in [1:4] do
    tot ← iadd tot (fin.param (k + 1))
  storeF64 (← fcvtFromSint .f64 tot) outPtr
  ret

def clifIR : Program := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "intsum_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end IntSumBench
