import Lean
import AlgorithmLib.Gen
import ShipScan

/-!
# The differential corpus

`HProgSem` claims to know what Cranelift's instructions compute. Nothing proves
that — Cranelift publishes no formal semantics — so this file checks it the only
way available: every operation, over operands chosen to separate the arms that
are easy to get wrong, evaluated *here* and executed *there*.

One straight-line body holds every case. Case `k` stores its result at
`out + 16k`, so the whole corpus is one artifact and one expected blob:
`base/tests/hprog_corpus.rs` runs the artifact through the JIT and compares.

The check is not circular. `base/src/clif_decode.rs` decides *which* Cranelift
instruction each term node emits; `HProgSem` decides what that instruction
*computes*; the machine decides who was right. The two were written from the
same documentation but not from each other, and only one of them runs on the
CPU.
-/

set_option maxRecDepth 100000

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgCorpus

/-- 16 bytes per case, so a vector result fits without overlapping the next. -/
def STRIDE : Nat := 16

private def i64min : Int := -9223372036854775808
private def i64max : Int := 9223372036854775807

/-- Operands that separate signed from unsigned, and saturation from wrap. -/
private def ints : List Int := [0, 1, -1, 3, i64min, i64max, 0x5555555555555555]

private def f32Bits (x : Float) : UInt64 := x.toFloat32.toBits.toUInt64
private def f64Bits (x : Float) : UInt64 := x.toBits

/-- The float operands worth trying: a quiet NaN, both infinities, both zeros,
    a subnormal, and something ordinary. -/
private def f32Cases : List (String × UInt64) :=
  [("nan", 0x7fc00000), ("inf", 0x7f800000), ("-inf", 0xff800000),
   ("0", 0x00000000), ("-0", 0x80000000), ("sub", 0x00000001),
   ("1.5", f32Bits 1.5), ("-2.25", f32Bits (-2.25))]

private def f64Cases : List (String × UInt64) :=
  [("nan", 0x7ff8000000000000), ("inf", 0x7ff0000000000000),
   ("-0", 0x8000000000000000), ("sub", 0x0000000000000001),
   ("1.5", f64Bits 1.5)]

/-- One case: a name and a body that leaves its result in the value it
    returns. The result *type* is the case's own --- the corpus covers every
    width --- so it is an index the structure hides. -/
structure Case (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type) where
  name : String
  {ty : ClifTy}
  run : Prog V L (V ty)

private def intBinops : List (String × (V .i64 → V .i64 → Prog V L (V .i64))) :=
  [("iadd", iadd), ("isub", isub), ("imul", imul), ("band", band),
   ("bandNot", bandNot), ("bor", bor), ("bxor", bxor)]

private def floatBinopNames : List String := ["fadd", "fsub", "fmul", "fmax", "fmin"]

/-- The operation a name stands for, at whichever width the caller asks. The
    corpus applies each at `f32` and at `f64`, so the width is a parameter
    rather than fixed by the table. -/
private def floatBinop {ty} (nm : String) (a b : V ty)
    (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Prog V L (V ty) :=
  match nm with
  | "fadd" => fadd a b h
  | "fsub" => fsub a b h
  | "fmul" => fmul a b h
  | "fmax" => fmax a b h
  | _      => fmin a b h

private def allICmp : List (String × ICmpCond) :=
  [("eq", .eq), ("ne", .ne), ("ult", .ult), ("ule", .ule), ("ugt", .ugt),
   ("uge", .uge), ("slt", .slt), ("sle", .sle), ("sgt", .sgt), ("sge", .sge)]

private def allFCmp : List (String × FloatCC) :=
  [("eq", .eq), ("ne", .ne), ("lt", .lt), ("le", .le), ("gt", .gt), ("ge", .ge)]

/-- Integer arithmetic and bitwise operations. -/
def casesInt : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  let pairs := [(1, 1), (-1, 1), (i64max, 1), (i64min, -1), (0x5555555555555555, 3)]
  for (nm, f) in intBinops do
    for (x, y) in pairs do
      cs := cs ++ [⟨s!"{nm}/{x}/{y}", do f (← iconst64 x) (← iconst64 y)⟩]
  for (x, y) in [(1, 1), (i64max, 3), (i64min, -1), (-1, i64max), (7, 2)] do
    cs := cs ++ [⟨s!"udiv/{x}/{y}", do udiv (← iconst64 x) (← iconst64 y)⟩]
  return cs

/-- Shift amounts at and past the operand width, where Cranelift masks rather
    than saturating — and the mask is the *operand's* width, not 64. -/
def casesShift : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  for x in [1, -1, i64max] do
    for sh in [0, 1, 31, 32, 63, 64, 65] do
      cs := cs ++ [⟨s!"ishl/{x}/{sh}", do ishl (← iconst64 x) (← iconst64 sh)⟩,
                   ⟨s!"ushr/{x}/{sh}", do ushr (← iconst64 x) (← iconst64 sh)⟩]
  for sh in [0, 31, 32, 33] do
    cs := cs ++ [⟨s!"ishl32/{sh}", do
                    uextend64 (← ishl (← iconst .i32 (-1)) (← iconst .i32 sh))⟩,
                 ⟨s!"ushr32/{sh}", do
                    uextend64 (← ushr (← iconst .i32 (-1)) (← iconst .i32 sh))⟩]
  return cs

/-- Unary integer operations, including `ctz` of zero and the three width
    changes, which differ only in how they treat the top bit. -/
def casesUnary : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  for x in ints do
    cs := cs ++ [⟨s!"ineg/{x}", do ineg (← iconst64 x)⟩,
                 ⟨s!"ctz/{x}", do ctz (← iconst64 x)⟩,
                 ⟨s!"popcnt/{x}", do popcnt (← iconst64 x)⟩,
                 ⟨s!"ireduce32/{x}", do uextend64 (← ireduce32 (← iconst64 x))⟩,
                 ⟨s!"sextend64/{x}", do sextend64 (← ireduce32 (← iconst64 x))⟩,
                 ⟨s!"uextend64/{x}", do uextend64 (← ireduce32 (← iconst64 x))⟩]
  return cs

/-- Every comparison condition on pairs that separate signed from unsigned. -/
def casesCmp : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  for (cn, c) in allICmp do
    for (x, y) in [(-1, 1), (1, -1), (0, 0), (i64min, i64max)] do
      cs := cs ++ [⟨s!"icmp.{cn}/{x}/{y}", do
        uextend64 (← icmp c (← iconst64 x) (← iconst64 y))⟩]
  for c in [0, 1, 2, i64max] do
    cs := cs ++ [⟨s!"select/{c}", do
      select (← iconst64 c) (← iconst64 111) (← iconst64 222)⟩]
  for (cn, c) in allFCmp do
    for (xn, x) in f32Cases do
      cs := cs ++ [⟨s!"fcmp.{cn}/{xn}/1.5", do
        uextend64 (← fcmp c (← fconst .f32 x) (← fconst .f32 (f32Bits 1.5)))⟩]
  return cs

/-- Float arithmetic at both widths. NaN appears on each side of every
    operation, because `fmax`/`fmin` are the arms most easily got wrong. -/
def casesFloat : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  let f32Pairs := [("nan", "1.5"), ("1.5", "nan"), ("inf", "-inf"),
                   ("0", "-0"), ("sub", "1.5"), ("-2.25", "1.5"), ("inf", "inf")]
  let bits32 := fun (n : String) => (f32Cases.find? (·.1 == n)).map (·.2) |>.getD 0
  let f64Pairs := [("nan", "1.5"), ("1.5", "nan"), ("inf", "-0"), ("sub", "1.5")]
  let bits64 := fun (n : String) =>
    (((("1.5", f64Bits 1.5) :: f64Cases).find? (·.1 == n)).map (·.2)).getD 0
  for nm in floatBinopNames do
    for (xn, yn) in f32Pairs do
      cs := cs ++ [⟨s!"f32.{nm}/{xn}/{yn}", do
        uextend64 (← bitcast .i32 (← floatBinop (ty := .f32) nm (← fconst .f32 (bits32 xn))
                                                   (← fconst .f32 (bits32 yn))))⟩]
    for (xn, yn) in f64Pairs do
      cs := cs ++ [⟨s!"f64.{nm}/{xn}/{yn}", do
        bitcast .i64 (← floatBinop (ty := .f64) nm (← fconst .f64 (bits64 xn)) (← fconst .f64 (bits64 yn)))⟩]
  for (xn, x) in f32Cases do
    cs := cs ++ [⟨s!"fneg/{xn}", do uextend64 (← bitcast .i32 (← fneg (← fconst .f32 x)))⟩]
    -- The *payload* of a generated NaN is target-specific, but negation's sign
    -- flip is not, so pull the sign bit out as an integer and compare it
    -- exactly. Without this the NaN-class relaxation would cover for a `fneg`
    -- that left the sign alone.
    cs := cs ++ [⟨s!"fnegSign/{xn}", do
      ushr (← uextend64 (← bitcast .i32 (← fneg (← fconst .f32 x)))) (← iconst64 31)⟩]
  return cs

/-- Conversions, where saturation and sign are the whole content. -/
def casesConv : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  for (xn, x) in f32Cases do
    cs := cs ++ [⟨s!"fpromote/{xn}", do bitcast .i64 (← fpromote (← fconst .f32 x))⟩,
                 ⟨s!"fcvtToUint32/{xn}", do uextend64 (← fcvtToUint .i32 (← fconst .f32 x))⟩,
                 ⟨s!"fcvtToUint64/{xn}", do fcvtToUint .i64 (← fconst .f32 x)⟩,
                 ⟨s!"bitcast.f32.i32/{xn}", do uextend64 (← bitcast .i32 (← fconst .f32 x))⟩]
  for x in ints do
    cs := cs ++ [⟨s!"fcvtFromSint32/{x}", do
                    uextend64 (← bitcast .i32 (← fcvtFromSint .f32 (← iconst64 x)))⟩,
                 ⟨s!"fcvtFromSint64/{x}", do
                    bitcast .i64 (← fcvtFromSint .f64 (← iconst64 x))⟩]
  return cs

/-- Vector construction, lane extraction, and the lane sign mask. -/
def casesVec : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  for (xn, x) in f32Cases do
    cs := cs ++ [⟨s!"splat.f32x4/{xn}", do splat .f32x4 (← fconst .f32 x)⟩,
                 ⟨s!"vhighBits/{xn}", do
                    uextend64 (← vhighBits (← splat .f32x4 (← fconst .f32 x)))⟩,
                 ⟨s!"extractlane/{xn}", do
                    uextend64 (← bitcast .i32
                      (← extractlane (← splat .f32x4 (← fconst .f32 x)) 2))⟩]
  return cs

/-- Control-flow shapes.

    The instruction corpus says nothing about any of this. Block parameters are
    phi nodes that Cranelift's register allocator resolves, `brif` carries
    arguments on both of its edges, a join has several predecessors, and a back
    edge re-binds a carry that shadows the header's. None of it is visible in a
    straight-line case, and it is where a term and its compiled blocks can
    actually disagree — the loop-exit numbering bug this library already had
    lived here. -/
def casesCfg : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  -- a counted loop: Σ i for i < 10
  cs := cs ++ [⟨"cfg/loop.sum10", do
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i (← iconst64 10), %[acc], ()))
      (body := fun i acc _ => return %[← iadd i (← iconst64 1), ← iadd acc i])
    pure (e.head)⟩]
  -- a loop whose test fails on entry, so the body never runs and the exit
  -- block still has to bind its parameters
  cs := cs ++ [⟨"cfg/loop.zeroTrips", do
    let e ← wloop2 (← iconst64 99) (← iconst64 7)
      (head := fun i acc => return (exitIfSGe i (← iconst64 10), %[acc], ()))
      (body := fun i acc _ => return %[← iadd i (← iconst64 1), ← iadd acc i])
    pure (e.head)⟩]
  -- a loop inside a loop, the inner one reading the outer one's carry
  cs := cs ++ [⟨"cfg/loop.nested", do
    let one ← iconst64 1
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i (← iconst64 3), %[acc], ()))
      (body := fun i acc _ => do
        let inner ← wloop2 (← iconst64 0) acc
          (head := fun j a => return (exitIfSGe j (← iconst64 4), %[a], ()))
          (body := fun j a _ => do
            let t ← iadd (← ishl i (← iconst64 2)) j
            return %[← iadd j one, ← iadd a t])
        return %[← iadd i one, inner.head])
    pure (e.head)⟩]
  -- three carries at once, so the block takes three parameters
  cs := cs ++ [⟨"cfg/loop.threeCarries", do
    let e ← wloop %[← iconst64 0, ← iconst64 1, ← iconst64 100]
      (head := fun c => return (exitIfSGe (c.head) (← iconst64 5), c.drop 1, ()))
      (body := fun c _ => return %[← iadd (c.head) (← iconst64 1),
                                  ← imul (c.snd) (← iconst64 2),
                                  ← isub (c.thd) (c.snd)])
    pure (e.snd)⟩]
  -- an f64 carried through a block parameter, which is a different register
  -- class from everything above
  cs := cs ++ [⟨"cfg/loop.f64carry", do
    let e ← wloop %[← iconst64 0, ← fconst .f64 (f64Bits 1.0)]
      (head := fun c => return (exitIfSGe (c.head) (← iconst64 4), %[c.snd], ()))
      (body := fun c _ => return %[← iadd (c.head) (← iconst64 1),
                                  ← fadd (c.snd) (← fconst .f64 (f64Bits 0.5))])
    bitcast .i64 (e.head)⟩]
  -- and an f32x4, so a vector crosses a block boundary
  cs := cs ++ [⟨"cfg/loop.vecCarry", do
    let e ← wloop %[← iconst64 0, ← splat .f32x4 (← fconst .f32 (f32Bits 1.0))]
      (head := fun c => return (exitIfSGe (c.head) (← iconst64 3), %[c.snd], ()))
      (body := fun c _ => return %[← iadd (c.head) (← iconst64 1),
                                  ← fadd (c.snd)
                                      (← splat .f32x4 (← fconst .f32 (f32Bits 2.0)))])
    uextend64 (← bitcast .i32 (← extractlane (e.head) 1))⟩]
  -- both arms of a branch, each exporting a value to the join
  for (nm, a, b) in [("then", 1, 2), ("else", 2, 1)] do
    cs := cs ++ [⟨s!"cfg/ite.{nm}", do
      let j ← ifte .slt (← iconst64 a) (← iconst64 b)
        (thn := do pure %[← iconst64 111])
        (els := do pure %[← iconst64 222])
      pure (j.head)⟩]
  -- two values across the join, and a branch nested in a branch
  cs := cs ++ [⟨"cfg/ite.twoJoins", do
    let j ← ifte .eq (← iconst64 5) (← iconst64 5)
      (thn := do pure %[← iconst64 7, ← iconst64 9])
      (els := do pure %[← iconst64 0, ← iconst64 0])
    iadd (j.head) (← imul (j.snd) (← iconst64 10))⟩]
  cs := cs ++ [⟨"cfg/ite.nested", do
    let j ← ifte .slt (← iconst64 1) (← iconst64 2)
      (thn := do
        let k ← ifte .sgt (← iconst64 3) (← iconst64 4)
          (thn := do pure %[← iconst64 10]) (els := do pure %[← iconst64 20])
        pure %[k.head])
      (els := do pure %[← iconst64 30])
    pure (j.head)⟩]
  -- a branch inside a loop body, and a loop inside a branch arm
  cs := cs ++ [⟨"cfg/ite.inLoop", do
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i (← iconst64 6), %[acc], ()))
      (body := fun i acc _ => do
        let j ← ifte .eq (← band i (← iconst64 1)) (← iconst64 0)
          (thn := do pure %[← iadd acc i]) (els := do pure %[acc])
        return %[← iadd i (← iconst64 1), j.head])
    pure (e.head)⟩]
  cs := cs ++ [⟨"cfg/loop.inIte", do
    let j ← ifte .slt (← iconst64 0) (← iconst64 1)
      (thn := do
        let e ← wloop2 (← iconst64 0) (← iconst64 0)
          (head := fun i acc => return (exitIfSGe i (← iconst64 4), %[acc], ()))
          (body := fun i acc _ => return %[← iadd i (← iconst64 1), ← iadd acc i])
        pure %[e.head])
      (els := do pure %[← iconst64 (-1)])
    pure (j.head)⟩]
  return cs

/-- The `pmin`/`pmax` pattern, appended last so adding to it cannot renumber
    any case above. -/
def casesPminmax : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  -- `pmin`/`pmax` as the generators build them: a lane-wise compare bitcast to
  -- the operand width, then a bit-for-bit select. Both orders, because the rule
  -- Cranelift matches for `pmax` reverses the compare — a mistake there computes
  -- the minimum, and nothing else in the corpus would say so.
  --
  -- Curated pairs rather than a cross product: these are the ones where `pmin`
  -- and IEEE `fmin` disagree, which is the whole reason the pattern is used.
  let pairs : List (String × UInt64 × UInt64) :=
    [("nan/1.5", 0x7fc00000, f32Bits 1.5), ("1.5/nan", f32Bits 1.5, 0x7fc00000),
     ("0/-0", 0x00000000, 0x80000000), ("-0/0", 0x80000000, 0x00000000),
     ("inf/-inf", 0x7f800000, 0xff800000), ("sub/0", 0x00000001, 0x00000000),
     ("1.5/-2.25", f32Bits 1.5, f32Bits (-2.25))]
  for (nm, x, y) in pairs do
    cs := cs ++ [
      ⟨s!"pmin/{nm}", do
         let a ← splat .f32x4 (← fconst .f32 x)
         let b ← splat .f32x4 (← fconst .f32 y)
         uextend64 (← bitcast .i32 (← extractlane
           (← bitselect (← bitcast .f32x4 (← fcmp .lt a b)) a b) 0))⟩,
      ⟨s!"pmax/{nm}", do
         let a ← splat .f32x4 (← fconst .f32 x)
         let b ← splat .f32x4 (← fconst .f32 y)
         uextend64 (← bitcast .i32 (← extractlane
           (← bitselect (← bitcast .f32x4 (← fcmp .lt b a)) a b) 0))⟩]
  return cs

/-- Leaving a loop early, appended last so adding to it cannot renumber any case
    above.

    `br` is the one construct whose compiled shape depends on what the code
    around it does — a block already closed must not be closed again, and an
    exit block reached from two places must bind its parameters at the index
    both agree on. Every case here is a different way for that to go wrong. -/
def casesBr : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  -- the plain case: leave from inside the body, carrying the accumulator
  cs := cs ++ [⟨"br/early", do
    let e ← wloop2L (← iconst64 0) (← iconst64 0)
      (head := fun _ i acc => return (exitIfSGe i (← iconst64 100), %[acc], ()))
      (body := fun lbl i acc _ => do
        when .eq i (← iconst64 5) (brk lbl %[acc])
        return %[← iadd i (← iconst64 1), ← iadd acc i])
    pure (e.head)⟩]
  -- the guard never fires, so the loop still leaves through its own test
  cs := cs ++ [⟨"br/neverTaken", do
    let e ← wloop2L (← iconst64 0) (← iconst64 0)
      (head := fun _ i acc => return (exitIfSGe i (← iconst64 6), %[acc], ()))
      (body := fun lbl i acc _ => do
        when .eq i (← iconst64 99) (brk lbl %[← iconst64 (-1)])
        return %[← iadd i (← iconst64 1), ← iadd acc i])
    pure (e.head)⟩]
  -- two values across the exit, so the exit block takes two parameters and the
  -- early path has to agree with the normal one about both
  cs := cs ++ [⟨"br/twoVals", do
    let e ← wloop2L (← iconst64 0) (← iconst64 1)
      (head := fun _ i acc => return (exitIfSGe i (← iconst64 100), %[acc, i], ()))
      (body := fun lbl i acc _ => do
        when .eq i (← iconst64 4) (brk lbl %[acc, i])
        return %[← iadd i (← iconst64 1), ← imul acc (← iconst64 3)])
    iadd (e.head) (← imul (e.snd) (← iconst64 1000))⟩]
  -- both arms leave: there is no join block, and the body has no back edge
  for (nm, i0, acc0) in [("then", 5, 0), ("else", 0, 7)] do
    cs := cs ++ [⟨s!"br/bothArms.{nm}", do
      let e ← wloop2L (← iconst64 i0) (← iconst64 acc0)
        (head := fun _ i acc => return (exitIfSGe i (← iconst64 100), %[acc], ()))
        (body := fun lbl i acc _ => do
          let _ ← ifte .sge i (← iconst64 3)
            (thn := do brk lbl %[← iconst64 777]; pure %[])
            (els := do brk lbl %[← iadd acc i]; pure %[])
          return %[i, acc])
      pure (e.head)⟩]
  -- an inner loop leaves the outer one, which is what a depth other than 0 is
  cs := cs ++ [⟨"br/depth1", do
    let e ← wloop2L (← iconst64 0) (← iconst64 0)
      (head := fun _ i acc => return (exitIfSGe i (← iconst64 10), %[acc], ()))
      (body := fun outer i acc _ => do
        let inner ← wloop2 (← iconst64 0) acc
          (head := fun j a => return (exitIfSGe j (← iconst64 10), %[a], ()))
          (body := fun j a _ => do
            when .sge a (← iconst64 20) (brk outer %[a])
            return %[← iadd j (← iconst64 1), ← iadd a (← iconst64 3)])
        return %[← iadd i (← iconst64 1), inner.head])
    pure (e.head)⟩]
  -- leaving from a point after a whole inner loop ran, so the exit block's
  -- parameters sit past every slot that loop defined
  cs := cs ++ [⟨"br/afterInner", do
    let e ← wloop2L (← iconst64 0) (← iconst64 0)
      (head := fun _ i acc => return (exitIfSGe i (← iconst64 100), %[acc], ()))
      (body := fun lbl i acc _ => do
        let inner ← wloop2L (← iconst64 0) acc
          (head := fun _ j a => return (exitIfSGe j (← iconst64 2), %[a], ()))
          (body := fun lbl j a _ => return %[← iadd j (← iconst64 1),
                                        ← iadd a (← iconst64 5)])
        when .sge i (← iconst64 3) (brk lbl %[inner.head])
        return %[← iadd i (← iconst64 1), inner.head])
    pure (e.head)⟩]
  return cs

/-- The bottom-tested loop, appended last so adding to it cannot renumber any
    case above.

    `dloop` makes its test twice from one term — on `init` before the body and
    on `cont` after it — so what these separate is the two scopes agreeing: a
    guard that reads the wrong carry, or an exit that takes the initial value
    where it should take the final one, changes only these answers. -/
def casesDLoop : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  -- the ordinary trip: Σ i for i < 10
  cs := cs ++ [⟨"dloop/sum10", do
    let lim ← iconst64 10
    let e ← dwloop %[← iconst64 0, ← iconst64 0] .slt lim (contOnTrue := true) [1]
      (body := fun c => do
        let i' ← iaddImm (c.head) 1
        return (i', %[i', ← iadd (c.snd) (c.head)]))
      (guardIdx := some 0)
    pure (e.head)⟩]
  -- the guard fails on entry, so the body never runs and the exit still binds
  cs := cs ++ [⟨"dloop/zeroTrips", do
    let lim ← iconst64 0
    let e ← dwloop %[← iconst64 0, ← iconst64 7] .slt lim (contOnTrue := true) [1]
      (body := fun c => do
        let i' ← iaddImm (c.head) 1
        return (i', %[i', ← iadd (c.snd) (c.head)]))
      (guardIdx := some 0)
    pure (e.head)⟩]
  -- exactly one trip, which is the case a top-tested loop and a bottom-tested
  -- one disagree about if the guard is wrong
  cs := cs ++ [⟨"dloop/oneTrip", do
    let lim ← iconst64 1
    let e ← dwloop %[← iconst64 0, ← iconst64 100] .slt lim (contOnTrue := true) [1]
      (body := fun c => do
        let i' ← iaddImm (c.head) 1
        return (i', %[i', ← iadd (c.snd) (← iconst64 5)]))
      (guardIdx := some 0)
    pure (e.head)⟩]
  -- both carries leave, so the exit block takes two parameters in `exitIdx`
  -- order rather than carry order
  cs := cs ++ [⟨"dloop/twoOut", do
    let lim ← iconst64 4
    let e ← dwloop %[← iconst64 0, ← iconst64 1] .slt lim (contOnTrue := true) [1, 0]
      (body := fun c => do
        let i' ← iaddImm (c.head) 1
        return (i', %[i', ← imul (c.snd) (← iconst64 3)]))
      (guardIdx := some 0)
    iadd (e.head) (← imul (e.snd) (← iconst64 1000))⟩]
  -- a bottom-tested loop inside a bottom-tested loop
  cs := cs ++ [⟨"dloop/nested", do
    let lo ← iconst64 3
    let li ← iconst64 4
    let e ← dwloop %[← iconst64 0, ← iconst64 0] .slt lo (contOnTrue := true) [1]
      (body := fun c => do
        let inner ← dwloop %[← iconst64 0, c.snd] .slt li (contOnTrue := true) [1]
          (body := fun d => do
            let j' ← iaddImm (d.head) 1
            return (j', %[j', ← iadd (d.snd) (c.head)]))
          (guardIdx := some 0)
        let i' ← iaddImm (c.head) 1
        return (i', %[i', inner.head]))
      (guardIdx := some 0)
    pure (e.head)⟩]
  -- leaving one early, so `br` and `dloop` compose
  cs := cs ++ [⟨"dloop/br", do
    let lim ← iconst64 100
    let e ← dwloopL %[← iconst64 0, ← iconst64 0] .slt lim (contOnTrue := true) [1]
      (body := fun trip c => do
        when .eq c.head (← iconst64 6) (brk trip %[c.snd])
        let i' ← iaddImm (c.head) 1
        return (i', %[i', ← iadd (c.snd) (c.head)]))
      (guardIdx := some 0)
    pure (e.head)⟩]
  return cs

/-- Taking the back edge from inside a branch, appended last.

    `continue` is the only construct that jumps *backwards* from somewhere other
    than the end of a body, so what these separate is the carries it supplies
    reaching the header: a `continue` that passed the old carries instead of the
    new ones would spin, and one that passed them in the wrong order would
    count with the accumulator. -/
def casesCont : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  -- both arms take the back edge, with different accumulator updates
  cs := cs ++ [⟨"cont/bothArms", do
    let lim ← iconst64 8
    let e ← wloop2L (← iconst64 0) (← iconst64 0)
      (head := fun _ i acc => return (exitIfSGe i lim, %[acc], ()))
      (body := fun lbl i acc _ => do
        let _ ← ifte .eq (← band i (← iconst64 1)) (← iconst64 0)
          (thn := do continueWith lbl %[← iaddImm i 1, ← iadd acc i]; pure %[])
          (els := do continueWith lbl %[← iaddImm i 1, ← iadd acc (← iconst64 100)]; pure %[])
        return %[i, acc])
    pure (e.head)⟩]
  -- one arm continues, the other falls through to the end of the body
  cs := cs ++ [⟨"cont/oneArm", do
    let lim ← iconst64 6
    let e ← wloop2L (← iconst64 0) (← iconst64 0)
      (head := fun _ i acc => return (exitIfSGe i lim, %[acc], ()))
      (body := fun lbl i acc _ => do
        let j ← ifte .eq (← band i (← iconst64 1)) (← iconst64 0)
          (thn := do continueWith lbl %[← iaddImm i 1, acc])
          (els := do pure %[← iadd acc i])
        return %[← iaddImm i 1, j.head])
    pure (e.head)⟩]
  -- an inner loop takes the outer one's back edge, which is depth 1
  cs := cs ++ [⟨"cont/depth1", do
    let lo ← iconst64 4
    let li ← iconst64 3
    let e ← wloop2L (← iconst64 0) (← iconst64 0)
      (head := fun _ i acc => return (exitIfSGe i lo, %[acc], ()))
      (body := fun outer i acc _ => do
        let inner ← wloop2 (← iconst64 0) acc
          (head := fun j a => return (exitIfSGe j li, %[a], ()))
          (body := fun j a _ => do
            when .sge a (← iconst64 9) (continueWith outer %[← iaddImm i 1, a])
            return %[← iaddImm j 1, ← iadd a (← iconst64 2)])
        return %[← iaddImm i 1, inner.head])
    pure (e.head)⟩]
  return cs

/-- Integer comparison and bitwise logic on vectors, appended last.

    `Op.check` has always accepted a vector for `band`/`bandNot`/`bor`/`bxor`,
    but `evalOp` modelled only scalars — a term could pass the checker and get
    stuck in the interpreter. `icmp` was rejected outright, which is what a
    SIMD string search needs. These cases are what pin the lane-wise answers to
    the machine rather than to a reading of the documentation. -/
def casesVecInt : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  -- a lane-wise comparison, read back through the mask `vhighBits` extracts
  for (nm, cc, x, y) in
      [("eq.hit", ICmpCond.eq, 7, 7), ("eq.miss", .eq, 7, 9),
       ("ult", .ult, 1, 200), ("slt", .slt, 1, 200),
       ("ugt", .ugt, 200, 1), ("sgt", .sgt, 200, 1),
       ("sle.eq", .sle, 5, 5), ("uge.eq", .uge, 5, 5)] do
    cs := cs ++ [⟨s!"veccmp/{nm}", do
      let a ← splat .i8x16 (← iconst .i8 x)
      let b ← splat .i8x16 (← iconst .i8 y)
      uextend64 (← vhighBits (← icmp cc a b))⟩]
  -- the bitwise operations, lane-wise, read back one lane at a time
  for nm in ["band", "bandNot", "bor", "bxor"] do
    cs := cs ++ [⟨s!"vecbits/{nm}", do
      let a ← splat .i8x16 (← iconst .i8 0xF0)
      let b ← splat .i8x16 (← iconst .i8 0x3C)
      let r ← match nm with
        | "band" => band a b | "bandNot" => bandNot a b
        | "bor" => bor a b | _ => bxor a b
      uextend64 (← vhighBits r)⟩]
  -- and a mask fed straight into `bitselect`, which is what the comparison is
  -- produced for
  cs := cs ++ [⟨"veccmp/select", do
    let a ← splat .i8x16 (← iconst .i8 3)
    let b ← splat .i8x16 (← iconst .i8 3)
    let x ← splat .i8x16 (← iconst .i8 0x11)
    let y ← splat .i8x16 (← iconst .i8 0x22)
    uextend64 (← vhighBits (← bitselect (← icmp .eq a b) x y))⟩]
  return cs

/-- Taking a bottom-tested loop's back edge from inside it, appended last.

    A `dloop` has no header — its body block branches to itself — so `cont` into
    one targets that body block. What these separate is that the carries a
    `cont` supplies reach the *next trip* rather than the exit, and that the
    loop's own back-edge test is skipped when a `cont` takes the edge instead. -/
def casesDCont : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  -- a `cont` on even trips, the ordinary back edge on odd ones
  cs := cs ++ [⟨"dcont/alternate", do
    let lim ← iconst64 10
    let e ← dwloopL %[← iconst64 0, ← iconst64 0] .slt lim (contOnTrue := true) [1]
      (body := fun trip c => do
        let i := c.head; let acc := c.snd
        let _ ← ifte .eq (← band i (← iconst64 1)) (← iconst64 0)
          (thn := do
            continueWith trip %[← iaddImm i 1, ← iadd acc (← iconst64 100)]
            pure %[])
          (els := pure %[])
        let i' ← iaddImm i 1
        return (i', %[i', ← iadd acc i]))
      (guardIdx := some 0)
    pure (e.head)⟩]
  -- a `cont` that skips the loop's own test, so the trip count is what the
  -- continue path decides
  cs := cs ++ [⟨"dcont/skipTest", do
    let lim ← iconst64 3
    let e ← dwloopL %[← iconst64 0, ← iconst64 0] .slt lim (contOnTrue := true) [1]
      (body := fun trip c => do
        let i := c.head; let acc := c.snd
        let _ ← ifte .eq i (← iconst64 0)
          (thn := do
            continueWith trip %[← iaddImm i 1, ← iadd acc (← iconst64 7)]
            pure %[])
          (els := pure %[])
        let i' ← iaddImm i 1
        return (i', %[i', ← iadd acc (← iconst64 1)]))
      (guardIdx := some 0)
    pure (e.head)⟩]
  -- `br` and `cont` in the same bottom-tested loop
  cs := cs ++ [⟨"dcont/withBr", do
    let lim ← iconst64 100
    let e ← dwloopL %[← iconst64 0, ← iconst64 0] .slt lim (contOnTrue := true) [1]
      (body := fun trip c => do
        let i := c.head; let acc := c.snd
        when .eq i (← iconst64 5) (brk trip %[acc])
        let _ ← ifte .eq (← band i (← iconst64 1)) (← iconst64 0)
          (thn := do
            continueWith trip %[← iaddImm i 1, ← iadd acc (← iconst64 10)]
            pure %[])
          (els := pure %[])
        let i' ← iaddImm i 1
        return (i', %[i', ← iadd acc (← iconst64 1)]))
      (guardIdx := some 0)
    pure (e.head)⟩]
  return cs

/-- The libm shims.

    `Float32.sin`/`cos`/`pow` are `@[extern "sinf"/"cosf"/"powf"]`, so the model
    calls the same symbols the runtime does and agreement is expected rather
    than lucky. What this catches is the part that is not by construction: that
    an `f32` argument and result survive the call boundary the JIT builds, and
    that `clif_decode` gives the signature the same shape both sides assume. -/
private def casesMath : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  for (nm, bits) in f32Cases do
    cs := cs ++ [⟨s!"sinf/{nm}", do ffi .sinf %[← fconst .f32 bits]⟩]
    cs := cs ++ [⟨s!"cosf/{nm}", do ffi .cosf %[← fconst .f32 bits]⟩]
  for (an, a) in f32Cases do
    for (bn, b) in [("2", f32Bits 2.0), ("0.5", f32Bits 0.5), ("-1", f32Bits (-1.0)),
                    ("0", f32Bits 0.0)] do
      cs := cs ++ [⟨s!"powf/{an}^{bn}", do
        ffi .powf %[← fconst .f32 a, ← fconst .f32 b]⟩]
  return cs

/-- Scratch inside the corpus arena, clear of the context slots. -/
private def htCtxSlot : Nat := 0x80
private def htKeyA : Nat := 0x90
private def htKeyB : Nat := 0x98
private def htVal : Nat := 0xA0
private def htOut : Nat := 0xB0

/-- The hash table, run as one sequence: the cases share a world, so what each
    stores is the state the ones before it left.

    Every accessor in `ht.rs` reads table `0` regardless of the handle it is
    given, and iterates a `HashMap`, so `ht_get_entry` is asked here only while
    exactly one entry exists — an index into an unordered container is not
    something the implementation promises and not something to pin. -/
private def casesHt : List (Case V L) := Id.run do
  let put : Nat → List Nat → Prog V L Unit := fun off bs => do
    for (b, i) in bs.zipIdx do
      istore8 (← iconst64 (Int.ofNat b)) (← absAddr (← basePtr) (off + i))
  let ctx : Prog V L (V .i64) := do load64 (← absAddr (← basePtr) htCtxSlot)
  let mut cs : List (Case V L) := []
  -- `init` writes the context where the slot says; the first `create` is the
  -- one that matters, since `0` is the table every accessor reads.
  cs := cs ++ [⟨"ht/create", do
    ffiVoid .htInit %[← absAddr (← basePtr) htCtxSlot]
    ffi .htCreate %[← ctx]⟩]
  -- key "ab" and an eight-byte value, then a lookup that reports its length
  cs := cs ++ [⟨"ht/lookupLen", do
    put htKeyA [0x61, 0x62]
    storeI64 (← iconst64 0x1122334455667788) (← absAddr (← basePtr) htVal)
    ffiVoid .htInsert
      %[← ctx, ← absAddr (← basePtr) htKeyA, ← iconst32 2, ← absAddr (← basePtr) htVal, ← iconst32 8]
    ffi .htLookup %[← ctx, ← absAddr (← basePtr) htKeyA, ← iconst32 2, ← absAddr (← basePtr) htOut]⟩]
  cs := cs ++ [⟨"ht/lookupValue", do load64 (← absAddr (← basePtr) htOut)⟩]
  -- exactly one entry, so this index is the only one there is
  cs := cs ++ [⟨"ht/getEntryKeyLen", do
    ffi .htGetEntry
      %[← ctx, ← iconst32 0, ← absAddr (← basePtr) htOut, ← absAddr (← basePtr) (htOut + 8)]⟩]
  cs := cs ++ [⟨"ht/count1", do ffi .htCount %[← ctx]⟩]
  -- a key nothing inserted, which is the sentinel arm
  cs := cs ++ [⟨"ht/lookupMissing", do
    put htKeyB [0x63, 0x64]
    ffi .htLookup %[← ctx, ← absAddr (← basePtr) htKeyB, ← iconst32 2, ← absAddr (← basePtr) htOut]⟩]
  -- increment creates on the first call and accumulates after, including
  -- downwards
  cs := cs ++ [⟨"ht/incrementNew", do
    ffi .htIncrement %[← ctx, ← absAddr (← basePtr) htKeyB, ← iconst32 2, ← iconst64 10]⟩]
  cs := cs ++ [⟨"ht/incrementAgain", do
    ffi .htIncrement %[← ctx, ← absAddr (← basePtr) htKeyB, ← iconst32 2, ← iconst64 5]⟩]
  cs := cs ++ [⟨"ht/incrementDown", do
    ffi .htIncrement %[← ctx, ← absAddr (← basePtr) htKeyB, ← iconst32 2, ← iconst64 (-7)]⟩]
  cs := cs ++ [⟨"ht/count2", do ffi .htCount %[← ctx]⟩]
  -- and the counter reads back as the eight bytes increment stores
  cs := cs ++ [⟨"ht/incrementStored", do
    let _ ← ffi .htLookup
      %[← ctx, ← absAddr (← basePtr) htKeyB, ← iconst32 2, ← absAddr (← basePtr) htOut]
    load64 (← absAddr (← basePtr) htOut)⟩]
  return cs

/-- Every case, named so a failure says which one. -/
def cases : List (Case V L) :=
  casesInt ++ casesShift ++ casesUnary ++ casesCmp ++ casesFloat ++ casesConv
    ++ casesVec ++ casesCfg ++ casesPminmax ++ casesBr ++ casesDLoop ++ casesCont ++ casesVecInt
    ++ casesDCont ++ casesMath ++ casesHt

/-- Every case, storing its result at its own stride in the output buffer. -/
def body : Prog V L Unit := do
  let outPtr ← outPtr
  for (c, k) in cases.zipIdx do
    let r ← c.run
    store r (← iadd outPtr (← iconst64 (STRIDE * k)))

/-- The names, in the order the body stores them. -/
def caseNames : List String := (cases (V := Prog.Slot) (L := Prog.Lvl)).map (·.name)

/-- The term the cases denote. This body is deeper than reifying it could go,
    which used to be the reason it was built rather than spliced; there is no
    splice now, and it is an ordinary term like every other. -/
def code : Code := Prog.emit body

/-- The same, refused if `wf` rejects it --- a value `main` can act on. -/
def checked : Except String Code := Prog.emitChecked body

-- No `wf` *theorem* here, deliberately. This term is a test fixture, and its
-- well-formedness is already checked twice the loud way: `checked` above, and
-- Cranelift's verifier rejecting the emitted CLIF. Proving it instead would
-- mean a `native_decide` over several thousand slots, which buys nothing and
-- costs trust surface.

def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body)]

def outBytes : Nat := caseNames.length * STRIDE

/-- The table the body needs, read off the body. -/
def env : FnEnv := (Prog.run body).2.1

/-- How a case's result is compared against the machine.

    The bit pattern of a NaN *produced by an invalid operation* is not portable:
    x86 SSE returns the "indefinite" quiet NaN with the sign bit set, AArch64
    returns it clear. Artifacts here are portable CLIF, JIT'd on whatever
    machine runs them, so a model that pinned the payload would be wrong
    somewhere. These cases are compared as "is a NaN of this width" instead,
    which still catches the failure that matters — returning a number where the
    hardware returns NaN, which is exactly what `fmax`/`fmin` got wrong. -/
inductive Compare where
  | bits
  | nanOf (width : Nat)
  deriving Repr

def compareOf (v : Sem.V) : Compare :=
  match v with
  | .sc .f32 b => if (b &&& 0x7f800000) == 0x7f800000 && (b &&& 0x007fffff) != 0
                  then .nanOf 4 else .bits
  | .sc .f64 b => if (b &&& 0x7ff0000000000000) == 0x7ff0000000000000 &&
                     (b &&& 0x000fffffffffffff) != 0 then .nanOf 8 else .bits
  | _ => .bits

/-- The five values an entry point is called with, for a corpus run: the arena,
    then the input and output buffers with their lengths. -/
def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
   .sc .i64 8, .sc .i64 (Sem.regionBase .out), .sc .i64 outBytes.toUInt64]

/-- The world every run of the corpus starts from. Nothing has to be placed in
    the arena first: a body reaches the caller's buffers through the arguments
    `entryArgVals` supplies, the way `execute_into` supplies them. -/
def startWorld : Except String Sem.Mem :=
  .ok { arena := ByteArray.mk (Array.replicate 0x100 0),
        data := ByteArray.mk (Array.replicate 8 0),
        out := ByteArray.mk (Array.replicate outBytes 0) }

/-- The corpus run through the *compiled* form, so the trace and the bytes can
    be compared against the term's.

    This is the statement `compile_sound` proves, run as well. A theorem about
    traces can be true and vacuous, or true about an order the demo never runs,
    and executing both sides is what rules that out. -/
def viaBlocks : Except String (List Sem.Obs × ByteArray) := do
  let m ← startWorld
  let f ← Prog.compileProg 1 body
  match Blocks.run env f entryArgVals { mem := m } with
  | .stuck why => .error why
  | .ok obs w => .ok (obs, w.mem.out)

/-- The corpus run through the interpreter, which is what the JIT is compared
    against. -/
def expected : Except String ByteArray := do
  let m ← startWorld
  match Sem.run { env } entryArgVals { mem := m } code with
  | .stuck why => .error why
  | .ok _ w => .ok w.mem.out

/-- The term and its compiled form agree, on both the observation trace and the
    memory they leave behind. -/
def viaTerm : Except String (List Sem.Obs × ByteArray) := do
  let m ← startWorld
  match Sem.run { env } entryArgVals { mem := m } code with
  | .stuck why => .error why
  | .ok obs w => .ok (obs, w.mem.out)

end HProgCorpus

open AlgorithmLib in
def main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  -- A case whose operands do not typecheck is a type error where it is
  -- written; that the whole body is well-formed is checked here, and fails
  -- the generator rather than reaching the JIT.
  match HProgCorpus.checked with
  | .error e => throw (IO.userError s!"the corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_corpus" {
        functions := clif, required_memory := 0x100
      }]
      let names := HProgCorpus.caseNames
      -- A case is compared as a NaN when its own stored bytes are one, at the
      -- width its name says. Everything else is compared byte for byte.
      let modes := names.zipIdx.map fun (nm, k) =>
        let word (n : Nat) : UInt64 :=
          (List.range n).foldr
            (fun i acc => (acc <<< 8) ||| (bytes.get! (k * HProgCorpus.STRIDE + i)).toUInt64)
            (0 : UInt64)
        let isF64 := nm.startsWith "f64." || nm.startsWith "fpromote"
        let v : Sem.V := if isF64 then .sc .f64 (word 8) else .sc .f32 (word 4)
        match HProgCorpus.compareOf v with
        | .bits => 0
        | .nanOf w => w
      let j := Lean.Json.mkObj [
        ("stride", Lean.toJson HProgCorpus.STRIDE),
        ("names", Lean.toJson names),
        ("modes", Lean.toJson modes),
        ("expected", Lean.toJson (bytes.toList.map (·.toNat)))]
      -- Not an artifact, so not beside them: an artifact's side data goes in
      -- a directory of its name, and this is what to compare against.
      let sideDir := System.FilePath.mk dir / "hprog_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json") j.compress
      IO.println s!"corpus: {names.length} cases, {bytes.size} expected bytes"
      -- `compile_sound`, executed rather than proved.
      match HProgCorpus.viaTerm, HProgCorpus.viaBlocks with
      | .error e, _ | _, .error e => throw (IO.userError s!"compile_sound check: {e}")
      | .ok (tObs, tOut), .ok (bObs, bOut) =>
          if tObs != bObs then
            throw (IO.userError
              s!"compile_sound: traces differ ({tObs.length} term vs {bObs.length} block)")
          else if tOut != bOut then
            throw (IO.userError "compile_sound: final memory differs")
          else
            IO.println
              s!"compile_sound (executed): term and compiled form agree \
                 — {tObs.length} observations, {tOut.size} bytes"

#eval ShipScan.check "HProgCorpus"
  (gatedElsewhere := "main, which refuses on emitChecked before emitting")
