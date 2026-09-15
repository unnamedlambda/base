import Lean
import AlgorithmLib.Gen

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace RegPressureBench

/-
  Input: n f32 values.  Result: f64 — their sum.

  Same shape as PlainSum but with SIXTEEN live f32x4 accumulators instead of
  four.  x86-64 has sixteen xmm registers, so together with the pointer, the
  counter and the loop bound this cannot be held in registers and the allocator
  has to spill.  That is the point: four accumulators is the easy case for any
  allocator, and it says nothing about how regalloc2 compares to LLVM's greedy
  allocator once spilling starts.

  The trailing `n % 64` elements are ignored.  The Rust mirror ignores them
  too, so the two still agree bit for bit.
-/

def MEM_SIZE : Nat := 40
def ACCS : Nat := 16

open AlgorithmLib.Prog


def code : Prog V L Unit := do
  let dataPtr ← load64 (← absAddr (← basePtr) 0x18)
  let dataLen ← load64 (← absAddr (← basePtr) 0x20)
  let outPtr  ← load64 (← absAddr (← basePtr) 0x28)
  let n       ← ushrImm dataLen 2            -- element count
  -- floor(n/64) trips of 64 elements = 256 bytes each
  let mainEnd ← ishlImm (← ushrImm n 6) 8
  let zero    ← fconst32 f32Zero
  let acc0    ← splat .f32x4 zero
  let i0      ← iconst64 0

  -- The accumulators are a *run* of one type whose width this file chose, so
  -- they are carried as one and rebuilt position by position.
  let fin ← wloop (Vals.cons i0 (Vals.ofFn (n := ACCS) (fun _ => acc0)))
    (head := fun c => return (exitIfSGe c.head mainEnd, c, ()))
    (body := fun c _ => do
      let bi := c.head
      let off ← iadd dataPtr bi
      let accs' ← c.tail.uniformMapIdxM fun k a => do
        let v ← loadF32x4 (← iaddImm off (16 * k))
        fadd a v
      return Vals.cons (← iaddImm bi 256) accs')

  -- left fold, so the Rust mirror can reproduce the order exactly
  let accs := fin.tail.uniformToList
  let acc ← match accs with
    | [] => splat .f32x4 (← fconst32 f32Zero)
    | a :: rest => rest.foldlM (fun x y => fadd x y) a
  let sum64 ← fadd (← fadd (← fpromote (← extractlane acc 0))
                            (← fpromote (← extractlane acc 1)))
                   (← fadd (← fpromote (← extractlane acc 2))
                            (← fpromote (← extractlane acc 3)))
  store sum64 outPtr


def clifIR : Except String Program :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 code]

def artifacts (clif : Program) : Array Json :=
  #[toJsonEntry "regpressure_sum_algorithm" {
    clif,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end RegPressureBench
