import Lean
import AlgorithmLib.Gen

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

namespace HProgCorpus

/-- The libm shims and the hash table: the two families whose contracts in
    `HProgSem.callFfi` are transcriptions of `base/src/ffi/` rather than of
    Cranelift, and so want the same differential treatment the operations get. -/
def env : FnEnv := env% [.math, .ht]

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

open Sur in
/-- One case: a builder that leaves its result in the slot it returns. -/
abbrev Case := String × Sur.M R

open Sur in
private def intBinops : List (String × (R → R → Sur.M R)) :=
  [("iadd", iadd), ("isub", isub), ("imul", imul), ("band", band),
   ("bandNot", bandNot), ("bor", bor), ("bxor", bxor)]

open Sur in
private def floatBinops : List (String × (R → R → Sur.M R)) :=
  [("fadd", fadd), ("fsub", fsub), ("fmul", fmul), ("fmax", fmax), ("fmin", fmin)]

private def allICmp : List (String × ICmpCond) :=
  [("eq", .eq), ("ne", .ne), ("ult", .ult), ("ule", .ule), ("ugt", .ugt),
   ("uge", .uge), ("slt", .slt), ("sle", .sle), ("sgt", .sgt), ("sge", .sge)]

private def allFCmp : List (String × FloatCC) :=
  [("eq", .eq), ("ne", .ne), ("lt", .lt), ("le", .le), ("gt", .gt), ("ge", .ge)]

open Sur in
/-- Integer arithmetic and bitwise operations. -/
def casesInt : List Case := Id.run do
  let mut cs : List Case := []
  let pairs := [(1, 1), (-1, 1), (i64max, 1), (i64min, -1), (0x5555555555555555, 3)]
  for (nm, f) in intBinops do
    for (x, y) in pairs do
      cs := cs ++ [(s!"{nm}/{x}/{y}", do f (← iconst64 x) (← iconst64 y))]
  for (x, y) in [(1, 1), (i64max, 3), (i64min, -1), (-1, i64max), (7, 2)] do
    cs := cs ++ [(s!"udiv/{x}/{y}", do udiv (← iconst64 x) (← iconst64 y))]
  return cs

open Sur in
/-- Shift amounts at and past the operand width, where Cranelift masks rather
    than saturating — and the mask is the *operand's* width, not 64. -/
def casesShift : List Case := Id.run do
  let mut cs : List Case := []
  for x in [1, -1, i64max] do
    for sh in [0, 1, 31, 32, 63, 64, 65] do
      cs := cs ++ [(s!"ishl/{x}/{sh}", do ishl (← iconst64 x) (← iconst64 sh)),
                   (s!"ushr/{x}/{sh}", do ushr (← iconst64 x) (← iconst64 sh))]
  for sh in [0, 31, 32, 33] do
    cs := cs ++ [(s!"ishl32/{sh}", do
                    uextend64 (← ishl (← iconst .i32 (-1)) (← iconst .i32 sh))),
                 (s!"ushr32/{sh}", do
                    uextend64 (← ushr (← iconst .i32 (-1)) (← iconst .i32 sh)))]
  return cs

open Sur in
/-- Unary integer operations, including `ctz` of zero and the three width
    changes, which differ only in how they treat the top bit. -/
def casesUnary : List Case := Id.run do
  let mut cs : List Case := []
  for x in ints do
    cs := cs ++ [(s!"ineg/{x}", do ineg (← iconst64 x)),
                 (s!"ctz/{x}", do ctz (← iconst64 x)),
                 (s!"popcnt/{x}", do popcnt (← iconst64 x)),
                 (s!"ireduce32/{x}", do uextend64 (← ireduce32 (← iconst64 x))),
                 (s!"sextend64/{x}", do sextend64 (← ireduce32 (← iconst64 x))),
                 (s!"uextend64/{x}", do uextend64 (← ireduce32 (← iconst64 x)))]
  return cs

open Sur in
/-- Every comparison condition on pairs that separate signed from unsigned. -/
def casesCmp : List Case := Id.run do
  let mut cs : List Case := []
  for (cn, c) in allICmp do
    for (x, y) in [(-1, 1), (1, -1), (0, 0), (i64min, i64max)] do
      cs := cs ++ [(s!"icmp.{cn}/{x}/{y}", do
        uextend64 (← icmp c (← iconst64 x) (← iconst64 y)))]
  for c in [0, 1, 2, i64max] do
    cs := cs ++ [(s!"select/{c}", do
      select (← iconst64 c) (← iconst64 111) (← iconst64 222))]
  for (cn, c) in allFCmp do
    for (xn, x) in f32Cases do
      cs := cs ++ [(s!"fcmp.{cn}/{xn}/1.5", do
        uextend64 (← fcmp c (← fconst .f32 x) (← fconst .f32 (f32Bits 1.5))))]
  return cs

open Sur in
/-- Float arithmetic at both widths. NaN appears on each side of every
    operation, because `fmax`/`fmin` are the arms most easily got wrong. -/
def casesFloat : List Case := Id.run do
  let mut cs : List Case := []
  let f32Pairs := [("nan", "1.5"), ("1.5", "nan"), ("inf", "-inf"),
                   ("0", "-0"), ("sub", "1.5"), ("-2.25", "1.5"), ("inf", "inf")]
  let bits32 := fun (n : String) => (f32Cases.find? (·.1 == n)).map (·.2) |>.getD 0
  let f64Pairs := [("nan", "1.5"), ("1.5", "nan"), ("inf", "-0"), ("sub", "1.5")]
  let bits64 := fun (n : String) =>
    (((("1.5", f64Bits 1.5) :: f64Cases).find? (·.1 == n)).map (·.2)).getD 0
  for (nm, f) in floatBinops do
    for (xn, yn) in f32Pairs do
      cs := cs ++ [(s!"f32.{nm}/{xn}/{yn}", do
        uextend64 (← bitcast .i32 (← f (← fconst .f32 (bits32 xn))
                                        (← fconst .f32 (bits32 yn)))))]
    for (xn, yn) in f64Pairs do
      cs := cs ++ [(s!"f64.{nm}/{xn}/{yn}", do
        bitcast .i64 (← f (← fconst .f64 (bits64 xn)) (← fconst .f64 (bits64 yn))))]
  for (xn, x) in f32Cases do
    cs := cs ++ [(s!"fneg/{xn}", do uextend64 (← bitcast .i32 (← fneg (← fconst .f32 x))))]
    -- The *payload* of a generated NaN is target-specific, but negation's sign
    -- flip is not, so pull the sign bit out as an integer and compare it
    -- exactly. Without this the NaN-class relaxation would cover for a `fneg`
    -- that left the sign alone.
    cs := cs ++ [(s!"fnegSign/{xn}", do
      ushr (← uextend64 (← bitcast .i32 (← fneg (← fconst .f32 x)))) (← iconst64 31))]
  return cs

open Sur in
/-- Conversions, where saturation and sign are the whole content. -/
def casesConv : List Case := Id.run do
  let mut cs : List Case := []
  for (xn, x) in f32Cases do
    cs := cs ++ [(s!"fpromote/{xn}", do bitcast .i64 (← fpromote (← fconst .f32 x))),
                 (s!"fcvtToUint32/{xn}", do uextend64 (← fcvtToUint .i32 (← fconst .f32 x))),
                 (s!"fcvtToUint64/{xn}", do fcvtToUint .i64 (← fconst .f32 x)),
                 (s!"bitcast.f32.i32/{xn}", do uextend64 (← bitcast .i32 (← fconst .f32 x)))]
  for x in ints do
    cs := cs ++ [(s!"fcvtFromSint32/{x}", do
                    uextend64 (← bitcast .i32 (← fcvtFromSint .f32 (← iconst64 x)))),
                 (s!"fcvtFromSint64/{x}", do
                    bitcast .i64 (← fcvtFromSint .f64 (← iconst64 x)))]
  return cs

open Sur in
/-- Vector construction, lane extraction, and the lane sign mask. -/
def casesVec : List Case := Id.run do
  let mut cs : List Case := []
  for (xn, x) in f32Cases do
    cs := cs ++ [(s!"splat.f32x4/{xn}", do splat .f32x4 (← fconst .f32 x)),
                 (s!"vhighBits/{xn}", do
                    uextend64 (← vhighBits (← splat .f32x4 (← fconst .f32 x)))),
                 (s!"extractlane/{xn}", do
                    uextend64 (← bitcast .i32
                      (← extractlane (← splat .f32x4 (← fconst .f32 x)) 2)))]
  return cs

open Sur in
/-- Control-flow shapes.

    The instruction corpus says nothing about any of this. Block parameters are
    phi nodes that Cranelift's register allocator resolves, `brif` carries
    arguments on both of its edges, a join has several predecessors, and a back
    edge re-binds a carry that shadows the header's. None of it is visible in a
    straight-line case, and it is where a term and its compiled blocks can
    actually disagree — the loop-exit numbering bug this library already had
    lived here. -/
def casesCfg : List Case := Id.run do
  let mut cs : List Case := []
  -- a counted loop: Σ i for i < 10
  cs := cs ++ [("cfg/loop.sum10", do
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i (← iconst64 10), [acc], ()))
      (body := fun i acc _ => return [← iadd i (← iconst64 1), ← iadd acc i])
    pure (e.headD 0))]
  -- a loop whose test fails on entry, so the body never runs and the exit
  -- block still has to bind its parameters
  cs := cs ++ [("cfg/loop.zeroTrips", do
    let e ← wloop2 (← iconst64 99) (← iconst64 7)
      (head := fun i acc => return (exitIfSGe i (← iconst64 10), [acc], ()))
      (body := fun i acc _ => return [← iadd i (← iconst64 1), ← iadd acc i])
    pure (e.headD 0))]
  -- a loop inside a loop, the inner one reading the outer one's carry
  cs := cs ++ [("cfg/loop.nested", do
    let one ← iconst64 1
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i (← iconst64 3), [acc], ()))
      (body := fun i acc _ => do
        let inner ← wloop2 (← iconst64 0) acc
          (head := fun j a => return (exitIfSGe j (← iconst64 4), [a], ()))
          (body := fun j a _ => do
            let t ← iadd (← ishl i (← iconst64 2)) j
            return [← iadd j one, ← iadd a t])
        return [← iadd i one, inner.headD 0])
    pure (e.headD 0))]
  -- three carries at once, so the block takes three parameters
  cs := cs ++ [("cfg/loop.threeCarries", do
    let e ← wloop [← iconst64 0, ← iconst64 1, ← iconst64 100]
      (head := fun c => return (exitIfSGe (c.headD 0) (← iconst64 5), c.drop 1, ()))
      (body := fun c _ => return [← iadd (c.headD 0) (← iconst64 1),
                                  ← imul (c.getD 1 0) (← iconst64 2),
                                  ← isub (c.getD 2 0) (c.getD 1 0)])
    pure (e.getD 1 0))]
  -- an f64 carried through a block parameter, which is a different register
  -- class from everything above
  cs := cs ++ [("cfg/loop.f64carry", do
    let e ← wloop [← iconst64 0, ← fconst .f64 (f64Bits 1.0)]
      (head := fun c => return (exitIfSGe (c.headD 0) (← iconst64 4), [c.getD 1 0], ()))
      (body := fun c _ => return [← iadd (c.headD 0) (← iconst64 1),
                                  ← fadd (c.getD 1 0) (← fconst .f64 (f64Bits 0.5))])
    bitcast .i64 (e.headD 0))]
  -- and an f32x4, so a vector crosses a block boundary
  cs := cs ++ [("cfg/loop.vecCarry", do
    let e ← wloop [← iconst64 0, ← splat .f32x4 (← fconst .f32 (f32Bits 1.0))]
      (head := fun c => return (exitIfSGe (c.headD 0) (← iconst64 3), [c.getD 1 0], ()))
      (body := fun c _ => return [← iadd (c.headD 0) (← iconst64 1),
                                  ← fadd (c.getD 1 0)
                                      (← splat .f32x4 (← fconst .f32 (f32Bits 2.0)))])
    uextend64 (← bitcast .i32 (← extractlane (e.headD 0) 1)))]
  -- both arms of a branch, each exporting a value to the join
  for (nm, a, b) in [("then", 1, 2), ("else", 2, 1)] do
    cs := cs ++ [(s!"cfg/ite.{nm}", do
      let j ← ifte .slt (← iconst64 a) (← iconst64 b)
        (thn := do pure [← iconst64 111])
        (els := do pure [← iconst64 222])
      pure (j.headD 0))]
  -- two values across the join, and a branch nested in a branch
  cs := cs ++ [("cfg/ite.twoJoins", do
    let j ← ifte .eq (← iconst64 5) (← iconst64 5)
      (thn := do pure [← iconst64 7, ← iconst64 9])
      (els := do pure [← iconst64 0, ← iconst64 0])
    iadd (j.headD 0) (← imul (j.getD 1 0) (← iconst64 10)))]
  cs := cs ++ [("cfg/ite.nested", do
    let j ← ifte .slt (← iconst64 1) (← iconst64 2)
      (thn := do
        let k ← ifte .sgt (← iconst64 3) (← iconst64 4)
          (thn := do pure [← iconst64 10]) (els := do pure [← iconst64 20])
        pure [k.headD 0])
      (els := do pure [← iconst64 30])
    pure (j.headD 0))]
  -- a branch inside a loop body, and a loop inside a branch arm
  cs := cs ++ [("cfg/ite.inLoop", do
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i (← iconst64 6), [acc], ()))
      (body := fun i acc _ => do
        let j ← ifte .eq (← band i (← iconst64 1)) (← iconst64 0)
          (thn := do pure [← iadd acc i]) (els := do pure [acc])
        return [← iadd i (← iconst64 1), j.headD 0])
    pure (e.headD 0))]
  cs := cs ++ [("cfg/loop.inIte", do
    let j ← ifte .slt (← iconst64 0) (← iconst64 1)
      (thn := do
        let e ← wloop2 (← iconst64 0) (← iconst64 0)
          (head := fun i acc => return (exitIfSGe i (← iconst64 4), [acc], ()))
          (body := fun i acc _ => return [← iadd i (← iconst64 1), ← iadd acc i])
        pure [e.headD 0])
      (els := do pure [← iconst64 (-1)])
    pure (j.headD 0))]
  return cs

open Sur in
/-- The `pmin`/`pmax` pattern, appended last so adding to it cannot renumber
    any case above. -/
def casesPminmax : List Case := Id.run do
  let mut cs : List Case := []
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
      (s!"pmin/{nm}", do
         let a ← splat .f32x4 (← fconst .f32 x)
         let b ← splat .f32x4 (← fconst .f32 y)
         uextend64 (← bitcast .i32 (← extractlane
           (← bitselect (← bitcast .f32x4 (← fcmp .lt a b)) a b) 0))),
      (s!"pmax/{nm}", do
         let a ← splat .f32x4 (← fconst .f32 x)
         let b ← splat .f32x4 (← fconst .f32 y)
         uextend64 (← bitcast .i32 (← extractlane
           (← bitselect (← bitcast .f32x4 (← fcmp .lt b a)) a b) 0)))]
  return cs

open Sur in
/-- Leaving a loop early, appended last so adding to it cannot renumber any case
    above.

    `br` is the one construct whose compiled shape depends on what the code
    around it does — a block already closed must not be closed again, and an
    exit block reached from two places must bind its parameters at the index
    both agree on. Every case here is a different way for that to go wrong. -/
def casesBr : List Case := Id.run do
  let mut cs : List Case := []
  -- the plain case: leave from inside the body, carrying the accumulator
  cs := cs ++ [("br/early", do
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i (← iconst64 100), [acc], ()))
      (body := fun i acc _ => do
        when .eq i (← iconst64 5) (brk [acc])
        return [← iadd i (← iconst64 1), ← iadd acc i])
    pure (e.headD 0))]
  -- the guard never fires, so the loop still leaves through its own test
  cs := cs ++ [("br/neverTaken", do
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i (← iconst64 6), [acc], ()))
      (body := fun i acc _ => do
        when .eq i (← iconst64 99) (brk [← iconst64 (-1)])
        return [← iadd i (← iconst64 1), ← iadd acc i])
    pure (e.headD 0))]
  -- two values across the exit, so the exit block takes two parameters and the
  -- early path has to agree with the normal one about both
  cs := cs ++ [("br/twoVals", do
    let e ← wloop2 (← iconst64 0) (← iconst64 1)
      (head := fun i acc => return (exitIfSGe i (← iconst64 100), [acc, i], ()))
      (body := fun i acc _ => do
        when .eq i (← iconst64 4) (brk [acc, i])
        return [← iadd i (← iconst64 1), ← imul acc (← iconst64 3)])
    iadd (e.headD 0) (← imul (e.getD 1 0) (← iconst64 1000)))]
  -- both arms leave: there is no join block, and the body has no back edge
  for (nm, i0, acc0) in [("then", 5, 0), ("else", 0, 7)] do
    cs := cs ++ [(s!"br/bothArms.{nm}", do
      let e ← wloop2 (← iconst64 i0) (← iconst64 acc0)
        (head := fun i acc => return (exitIfSGe i (← iconst64 100), [acc], ()))
        (body := fun i acc _ => do
          let _ ← ifte .sge i (← iconst64 3)
            (thn := do brk [← iconst64 777]; pure [])
            (els := do brk [← iadd acc i]; pure [])
          return [i, acc])
      pure (e.headD 0))]
  -- an inner loop leaves the outer one, which is what a depth other than 0 is
  cs := cs ++ [("br/depth1", do
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i (← iconst64 10), [acc], ()))
      (body := fun i acc _ => do
        let inner ← wloop2 (← iconst64 0) acc
          (head := fun j a => return (exitIfSGe j (← iconst64 10), [a], ()))
          (body := fun j a _ => do
            when .sge a (← iconst64 20) (brkTo 1 [a])
            return [← iadd j (← iconst64 1), ← iadd a (← iconst64 3)])
        return [← iadd i (← iconst64 1), inner.headD 0])
    pure (e.headD 0))]
  -- leaving from a point after a whole inner loop ran, so the exit block's
  -- parameters sit past every slot that loop defined
  cs := cs ++ [("br/afterInner", do
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i (← iconst64 100), [acc], ()))
      (body := fun i acc _ => do
        let inner ← wloop2 (← iconst64 0) acc
          (head := fun j a => return (exitIfSGe j (← iconst64 2), [a], ()))
          (body := fun j a _ => return [← iadd j (← iconst64 1),
                                        ← iadd a (← iconst64 5)])
        when .sge i (← iconst64 3) (brk [inner.headD 0])
        return [← iadd i (← iconst64 1), inner.headD 0])
    pure (e.headD 0))]
  return cs

open Sur in
/-- The bottom-tested loop, appended last so adding to it cannot renumber any
    case above.

    `dloop` makes its test twice from one term — on `init` before the body and
    on `cont` after it — so what these separate is the two scopes agreeing: a
    guard that reads the wrong carry, or an exit that takes the initial value
    where it should take the final one, changes only these answers. -/
def casesDLoop : List Case := Id.run do
  let mut cs : List Case := []
  -- the ordinary trip: Σ i for i < 10
  cs := cs ++ [("dloop/sum10", do
    let lim ← iconst64 10
    let e ← dwloop [← iconst64 0, ← iconst64 0] .slt lim (contOnTrue := true) [1]
      (body := fun c => do
        let i' ← iaddImm (c.headD 0) 1
        return (i', [i', ← iadd (c.getD 1 0) (c.headD 0)]))
      (guardIdx := some 0)
    pure (e.headD 0))]
  -- the guard fails on entry, so the body never runs and the exit still binds
  cs := cs ++ [("dloop/zeroTrips", do
    let lim ← iconst64 0
    let e ← dwloop [← iconst64 0, ← iconst64 7] .slt lim (contOnTrue := true) [1]
      (body := fun c => do
        let i' ← iaddImm (c.headD 0) 1
        return (i', [i', ← iadd (c.getD 1 0) (c.headD 0)]))
      (guardIdx := some 0)
    pure (e.headD 0))]
  -- exactly one trip, which is the case a top-tested loop and a bottom-tested
  -- one disagree about if the guard is wrong
  cs := cs ++ [("dloop/oneTrip", do
    let lim ← iconst64 1
    let e ← dwloop [← iconst64 0, ← iconst64 100] .slt lim (contOnTrue := true) [1]
      (body := fun c => do
        let i' ← iaddImm (c.headD 0) 1
        return (i', [i', ← iadd (c.getD 1 0) (← iconst64 5)]))
      (guardIdx := some 0)
    pure (e.headD 0))]
  -- both carries leave, so the exit block takes two parameters in `exitIdx`
  -- order rather than carry order
  cs := cs ++ [("dloop/twoOut", do
    let lim ← iconst64 4
    let e ← dwloop [← iconst64 0, ← iconst64 1] .slt lim (contOnTrue := true) [1, 0]
      (body := fun c => do
        let i' ← iaddImm (c.headD 0) 1
        return (i', [i', ← imul (c.getD 1 0) (← iconst64 3)]))
      (guardIdx := some 0)
    iadd (e.headD 0) (← imul (e.getD 1 0) (← iconst64 1000)))]
  -- a bottom-tested loop inside a bottom-tested loop
  cs := cs ++ [("dloop/nested", do
    let lo ← iconst64 3
    let li ← iconst64 4
    let e ← dwloop [← iconst64 0, ← iconst64 0] .slt lo (contOnTrue := true) [1]
      (body := fun c => do
        let inner ← dwloop [← iconst64 0, c.getD 1 0] .slt li (contOnTrue := true) [1]
          (body := fun d => do
            let j' ← iaddImm (d.headD 0) 1
            return (j', [j', ← iadd (d.getD 1 0) (c.headD 0)]))
          (guardIdx := some 0)
        let i' ← iaddImm (c.headD 0) 1
        return (i', [i', inner.headD 0]))
      (guardIdx := some 0)
    pure (e.headD 0))]
  -- leaving one early, so `br` and `dloop` compose
  cs := cs ++ [("dloop/br", do
    let lim ← iconst64 100
    let e ← dwloop [← iconst64 0, ← iconst64 0] .slt lim (contOnTrue := true) [1]
      (body := fun c => do
        when .eq (c.headD 0) (← iconst64 6) (brk [c.getD 1 0])
        let i' ← iaddImm (c.headD 0) 1
        return (i', [i', ← iadd (c.getD 1 0) (c.headD 0)]))
      (guardIdx := some 0)
    pure (e.headD 0))]
  return cs

open Sur in
/-- Taking the back edge from inside a branch, appended last.

    `continue` is the only construct that jumps *backwards* from somewhere other
    than the end of a body, so what these separate is the carries it supplies
    reaching the header: a `continue` that passed the old carries instead of the
    new ones would spin, and one that passed them in the wrong order would
    count with the accumulator. -/
def casesCont : List Case := Id.run do
  let mut cs : List Case := []
  -- both arms take the back edge, with different accumulator updates
  cs := cs ++ [("cont/bothArms", do
    let lim ← iconst64 8
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i lim, [acc], ()))
      (body := fun i acc _ => do
        let _ ← ifte .eq (← band i (← iconst64 1)) (← iconst64 0)
          (thn := do continueWith [← iaddImm i 1, ← iadd acc i]; pure [])
          (els := do continueWith [← iaddImm i 1, ← iadd acc (← iconst64 100)]; pure [])
        return [i, acc])
    pure (e.headD 0))]
  -- one arm continues, the other falls through to the end of the body
  cs := cs ++ [("cont/oneArm", do
    let lim ← iconst64 6
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i lim, [acc], ()))
      (body := fun i acc _ => do
        let j ← ifte .eq (← band i (← iconst64 1)) (← iconst64 0)
          (thn := do continueWith [← iaddImm i 1, acc]; pure [])
          (els := do pure [← iadd acc i])
        return [← iaddImm i 1, j.headD 0])
    pure (e.headD 0))]
  -- an inner loop takes the outer one's back edge, which is depth 1
  cs := cs ++ [("cont/depth1", do
    let lo ← iconst64 4
    let li ← iconst64 3
    let e ← wloop2 (← iconst64 0) (← iconst64 0)
      (head := fun i acc => return (exitIfSGe i lo, [acc], ()))
      (body := fun i acc _ => do
        let inner ← wloop2 (← iconst64 0) acc
          (head := fun j a => return (exitIfSGe j li, [a], ()))
          (body := fun j a _ => do
            when .sge a (← iconst64 9) (contTo 1 [← iaddImm i 1, a])
            return [← iaddImm j 1, ← iadd a (← iconst64 2)])
        return [← iaddImm i 1, inner.headD 0])
    pure (e.headD 0))]
  return cs

open Sur in
/-- Integer comparison and bitwise logic on vectors, appended last.

    `Op.check` has always accepted a vector for `band`/`bandNot`/`bor`/`bxor`,
    but `evalOp` modelled only scalars — a term could pass the checker and get
    stuck in the interpreter. `icmp` was rejected outright, which is what a
    SIMD string search needs. These cases are what pin the lane-wise answers to
    the machine rather than to a reading of the documentation. -/
def casesVecInt : List Case := Id.run do
  let mut cs : List Case := []
  -- a lane-wise comparison, read back through the mask `vhighBits` extracts
  for (nm, cc, x, y) in
      [("eq.hit", ICmpCond.eq, 7, 7), ("eq.miss", .eq, 7, 9),
       ("ult", .ult, 1, 200), ("slt", .slt, 1, 200),
       ("ugt", .ugt, 200, 1), ("sgt", .sgt, 200, 1),
       ("sle.eq", .sle, 5, 5), ("uge.eq", .uge, 5, 5)] do
    cs := cs ++ [(s!"veccmp/{nm}", do
      let a ← splat .i8x16 (← iconst .i8 x)
      let b ← splat .i8x16 (← iconst .i8 y)
      uextend64 (← vhighBits (← icmp cc a b)))]
  -- the bitwise operations, lane-wise, read back one lane at a time
  for (nm, f) in
      [("band", (Sur.band : R → R → Sur.M R)), ("bandNot", Sur.bandNot),
       ("bor", Sur.bor), ("bxor", Sur.bxor)] do
    cs := cs ++ [(s!"vecbits/{nm}", do
      let a ← splat .i8x16 (← iconst .i8 0xF0)
      let b ← splat .i8x16 (← iconst .i8 0x3C)
      uextend64 (← vhighBits (← f a b)))]
  -- and a mask fed straight into `bitselect`, which is what the comparison is
  -- produced for
  cs := cs ++ [("veccmp/select", do
    let a ← splat .i8x16 (← iconst .i8 3)
    let b ← splat .i8x16 (← iconst .i8 3)
    let x ← splat .i8x16 (← iconst .i8 0x11)
    let y ← splat .i8x16 (← iconst .i8 0x22)
    uextend64 (← vhighBits (← bitselect (← icmp .eq a b) x y)))]
  return cs

open Sur in
/-- Taking a bottom-tested loop's back edge from inside it, appended last.

    A `dloop` has no header — its body block branches to itself — so `cont` into
    one targets that body block. What these separate is that the carries a
    `cont` supplies reach the *next trip* rather than the exit, and that the
    loop's own back-edge test is skipped when a `cont` takes the edge instead. -/
def casesDCont : List Case := Id.run do
  let mut cs : List Case := []
  -- a `cont` on even trips, the ordinary back edge on odd ones
  cs := cs ++ [("dcont/alternate", do
    let lim ← iconst64 10
    let e ← dwloop [← iconst64 0, ← iconst64 0] .slt lim (contOnTrue := true) [1]
      (body := fun c => do
        let i := c.headD 0; let acc := c.getD 1 0
        let _ ← ifte .eq (← band i (← iconst64 1)) (← iconst64 0)
          (thn := do
            continueWith [← iaddImm i 1, ← iadd acc (← iconst64 100)]
            pure [])
          (els := pure [])
        let i' ← iaddImm i 1
        return (i', [i', ← iadd acc i]))
      (guardIdx := some 0)
    pure (e.headD 0))]
  -- a `cont` that skips the loop's own test, so the trip count is what the
  -- continue path decides
  cs := cs ++ [("dcont/skipTest", do
    let lim ← iconst64 3
    let e ← dwloop [← iconst64 0, ← iconst64 0] .slt lim (contOnTrue := true) [1]
      (body := fun c => do
        let i := c.headD 0; let acc := c.getD 1 0
        let _ ← ifte .eq i (← iconst64 0)
          (thn := do
            continueWith [← iaddImm i 1, ← iadd acc (← iconst64 7)]
            pure [])
          (els := pure [])
        let i' ← iaddImm i 1
        return (i', [i', ← iadd acc (← iconst64 1)]))
      (guardIdx := some 0)
    pure (e.headD 0))]
  -- `br` and `cont` in the same bottom-tested loop
  cs := cs ++ [("dcont/withBr", do
    let lim ← iconst64 100
    let e ← dwloop [← iconst64 0, ← iconst64 0] .slt lim (contOnTrue := true) [1]
      (body := fun c => do
        let i := c.headD 0; let acc := c.getD 1 0
        when .eq i (← iconst64 5) (brk [acc])
        let _ ← ifte .eq (← band i (← iconst64 1)) (← iconst64 0)
          (thn := do
            continueWith [← iaddImm i 1, ← iadd acc (← iconst64 10)]
            pure [])
          (els := pure [])
        let i' ← iaddImm i 1
        return (i', [i', ← iadd acc (← iconst64 1)]))
      (guardIdx := some 0)
    pure (e.headD 0))]
  return cs

open Sur in
/-- The libm shims.

    `Float32.sin`/`cos`/`pow` are `@[extern "sinf"/"cosf"/"powf"]`, so the model
    calls the same symbols the runtime does and agreement is expected rather
    than lucky. What this catches is the part that is not by construction: that
    an `f32` argument and result survive the call boundary the JIT builds, and
    that `clif_decode` gives the signature the same shape both sides assume. -/
private def casesMath : List Case := Id.run do
  let mut cs : List Case := []
  for (nm, bits) in f32Cases do
    cs := cs ++ [(s!"sinf/{nm}", do call IR.Ffi.sinf.id [← fconst .f32 bits])]
    cs := cs ++ [(s!"cosf/{nm}", do call IR.Ffi.cosf.id [← fconst .f32 bits])]
  for (an, a) in f32Cases do
    for (bn, b) in [("2", f32Bits 2.0), ("0.5", f32Bits 0.5), ("-1", f32Bits (-1.0)),
                    ("0", f32Bits 0.0)] do
      cs := cs ++ [(s!"powf/{an}^{bn}", do
        call IR.Ffi.powf.id [← fconst .f32 a, ← fconst .f32 b])]
  return cs

/-- Scratch inside the corpus arena, past the `0x28` the output pointer uses. -/
private def htCtxSlot : Nat := 0x80
private def htKeyA : Nat := 0x90
private def htKeyB : Nat := 0x98
private def htVal : Nat := 0xA0
private def htOut : Nat := 0xB0

open Sur in
/-- The hash table, run as one sequence: the cases share a world, so what each
    stores is the state the ones before it left.

    Every accessor in `ht.rs` reads table `0` regardless of the handle it is
    given, and iterates a `HashMap`, so `ht_get_entry` is asked here only while
    exactly one entry exists — an index into an unordered container is not
    something the implementation promises and not something to pin. -/
private def casesHt : List Case := Id.run do
  let put : Nat → List Nat → Sur.M Unit := fun off bs => do
    for (b, i) in bs.zipIdx do
      istore8 (← iconst64 (Int.ofNat b)) (← absAddr basePtr (off + i))
  let ctx : Sur.M R := do load64 (← absAddr basePtr htCtxSlot)
  let mut cs : List Case := []
  -- `init` writes the context where the slot says; the first `create` is the
  -- one that matters, since `0` is the table every accessor reads.
  cs := cs ++ [("ht/create", do
    callVoid IR.Ffi.htInit.id [← absAddr basePtr htCtxSlot]
    call IR.Ffi.htCreate.id [← ctx])]
  -- key "ab" and an eight-byte value, then a lookup that reports its length
  cs := cs ++ [("ht/lookupLen", do
    put htKeyA [0x61, 0x62]
    storeI64 (← iconst64 0x1122334455667788) (← absAddr basePtr htVal)
    callVoid IR.Ffi.htInsert.id
      [← ctx, ← absAddr basePtr htKeyA, ← iconst32 2, ← absAddr basePtr htVal, ← iconst32 8]
    call IR.Ffi.htLookup.id [← ctx, ← absAddr basePtr htKeyA, ← iconst32 2, ← absAddr basePtr htOut])]
  cs := cs ++ [("ht/lookupValue", do load64 (← absAddr basePtr htOut))]
  -- exactly one entry, so this index is the only one there is
  cs := cs ++ [("ht/getEntryKeyLen", do
    call IR.Ffi.htGetEntry.id
      [← ctx, ← iconst32 0, ← absAddr basePtr htOut, ← absAddr basePtr (htOut + 8)])]
  cs := cs ++ [("ht/count1", do call IR.Ffi.htCount.id [← ctx])]
  -- a key nothing inserted, which is the sentinel arm
  cs := cs ++ [("ht/lookupMissing", do
    put htKeyB [0x63, 0x64]
    call IR.Ffi.htLookup.id [← ctx, ← absAddr basePtr htKeyB, ← iconst32 2, ← absAddr basePtr htOut])]
  -- increment creates on the first call and accumulates after, including
  -- downwards
  cs := cs ++ [("ht/incrementNew", do
    call IR.Ffi.htIncrement.id [← ctx, ← absAddr basePtr htKeyB, ← iconst32 2, ← iconst64 10])]
  cs := cs ++ [("ht/incrementAgain", do
    call IR.Ffi.htIncrement.id [← ctx, ← absAddr basePtr htKeyB, ← iconst32 2, ← iconst64 5])]
  cs := cs ++ [("ht/incrementDown", do
    call IR.Ffi.htIncrement.id [← ctx, ← absAddr basePtr htKeyB, ← iconst32 2, ← iconst64 (-7)])]
  cs := cs ++ [("ht/count2", do call IR.Ffi.htCount.id [← ctx])]
  -- and the counter reads back as the eight bytes increment stores
  cs := cs ++ [("ht/incrementStored", do
    let _ ← call IR.Ffi.htLookup.id
      [← ctx, ← absAddr basePtr htKeyB, ← iconst32 2, ← absAddr basePtr htOut]
    load64 (← absAddr basePtr htOut))]
  return cs

/-- Every case, named so a failure says which one. -/
def cases : List Case :=
  casesInt ++ casesShift ++ casesUnary ++ casesCmp ++ casesFloat ++ casesConv
    ++ casesVec ++ casesCfg ++ casesPminmax ++ casesBr ++ casesDLoop ++ casesCont ++ casesVecInt
    ++ casesDCont ++ casesMath ++ casesHt

open Sur in
/-- Every case, storing its result at its own stride in the output buffer. -/
def body : Sur.M Unit := do
  let outPtr ← load64 (← absAddr basePtr 0x28)
  for (c, k) in (cases.map (·.2)).zipIdx do
    let r ← c
    store r (← iadd outPtr (← iconst64 (STRIDE * k)))

/-- Built rather than spliced: `clif%` embeds the finished literal, and a body
    of this many cases nests deeper than reifying it can go. `main` runs
    `buildChecked` instead, so the surface and `wf` are both still checked —
    once, natively, rather than during elaboration. -/
def code : Code := Sur.build body env ptrParams

/-- What `clif%` would have reported, as a value `main` can refuse on. -/
def checked : Except String Code := Sur.buildChecked body env ptrParams

-- No `wf` *theorem* here, deliberately. This term is a test fixture, and its
-- well-formedness is already checked twice the loud way: `checked` above, and
-- Cranelift's verifier rejecting the emitted CLIF. Proving it instead would
-- mean a `native_decide` over several thousand slots, which buys nothing and
-- costs trust surface.

def program : Program := IR.program [noopFunction, compileBody 1 code env]

def outBytes : Nat := cases.length * STRIDE

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

/-- The world every run of the corpus starts from: an arena with the output
    pointer already placed, the way `execute_into` places it. -/
def startWorld : Except String Sem.Mem :=
  let m0 : Sem.Mem :=
    { arena := ByteArray.mk (Array.replicate 0x100 0),
      data := ByteArray.mk (Array.replicate 8 0),
      out := ByteArray.mk (Array.replicate outBytes 0) }
  match m0.store (Sem.addrOf .arena 0x28) 8 (Sem.regionBase .out) with
  | none => .error "could not place the output pointer"
  | some m => .ok m

/-- The corpus run through the *compiled* form, so the trace and the bytes can
    be compared against the term's.

    This is `compile_sound`'s statement, checked by execution rather than
    proved. Doing it in this order is deliberate: a theorem about traces can be
    true and vacuous, or true about an order the demo never runs, and executing
    both sides first is what rules that out. -/
def viaBlocks : Except String (List Sem.Obs × ByteArray) := do
  let m ← startWorld
  let f := compileBody 1 code env
  match Blocks.run env f [.sc .i64 (Sem.regionBase .arena)] { mem := m } with
  | .stuck why => .error why
  | .ok obs w => .ok (obs, w.mem.out)

/-- The corpus run through the interpreter, which is what the JIT is compared
    against. -/
def expected : Except String ByteArray := do
  let m ← startWorld
  match Sem.run { env } [.sc .i64 (Sem.regionBase .arena)] { mem := m } code with
  | .stuck why => .error why
  | .ok _ w => .ok w.mem.out

/-- The term and its compiled form agree, on both the observation trace and the
    memory they leave behind. -/
def viaTerm : Except String (List Sem.Obs × ByteArray) := do
  let m ← startWorld
  match Sem.run { env } [.sc .i64 (Sem.regionBase .arena)] { mem := m } code with
  | .stuck why => .error why
  | .ok obs w => .ok (obs, w.mem.out)

end HProgCorpus

open AlgorithmLib in
def main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  -- What `clif%` checks while splicing, checked here instead: a case whose
  -- operands do not typecheck fails the generator rather than reaching the JIT.
  match HProgCorpus.checked with
  | .error e => throw (IO.userError s!"the corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the corpus: {e}")
  | .ok bytes =>
      emitArtifacts dir #[toJsonEntry "hprog_corpus" {
        clif := HProgCorpus.program, memory_size := 0x100
      } { fn_idx := u32 1 }]
      let names := HProgCorpus.cases.map (·.1)
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
      -- Not an artifact, so not beside them: a `.json` in the output directory
      -- is one an application can embed, and this is what to compare against.
      let sideDir := System.FilePath.mk dir / "expected"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "hprog_corpus_expected.json") j.compress
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
