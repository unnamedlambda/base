module
public import Lean
public import AlgorithmLib.Surface.Link
meta import AlgorithmLib.Surface.Link
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Host.Blocks
meta import AlgorithmLib.Host.Blocks
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The differential corpus

`Host.Sem` claims to know what Cranelift's instructions compute. Nothing proves
that — Cranelift publishes no formal semantics — so this file checks it the only
way available: every operation, over operands chosen to separate the arms that
are easy to get wrong, evaluated *here* and executed *there*.

One straight-line body holds every case. Case `k` stores its result at
`out + 16k`, so the whole corpus is one artifact and one expected blob:
`base/tests/hprog_corpus.rs` runs the artifact through the JIT and compares.

The check is not circular. `base/src/clif_decode.rs` decides *which* Cranelift
instruction each term node emits; `Host.Sem` decides what that instruction
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

def i64min : Int := -9223372036854775808
def i64max : Int := 9223372036854775807

/-- Operands that separate signed from unsigned, and saturation from wrap. -/
def ints : List Int := [0, 1, -1, 3, i64min, i64max, 0x5555555555555555]

def f32Bits (x : Float) : UInt64 := x.toFloat32.toBits.toUInt64
def f64Bits (x : Float) : UInt64 := x.toBits

/-- The float operands worth trying: a quiet NaN, both infinities, both zeros,
    a subnormal, and something ordinary. -/
def f32Cases : List (String × UInt64) :=
  [("nan", 0x7fc00000), ("inf", 0x7f800000), ("-inf", 0xff800000),
   ("0", 0x00000000), ("-0", 0x80000000), ("sub", 0x00000001),
   ("1.5", f32Bits 1.5), ("-2.25", f32Bits (-2.25))]

def f64Cases : List (String × UInt64) :=
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

def intBinops : List (String × (V .i64 → V .i64 → Prog V L (V .i64))) :=
  [("iadd", iadd), ("isub", isub), ("imul", imul), ("band", band),
   ("bandNot", bandNot), ("bor", bor), ("bxor", bxor)]

def floatBinopNames : List String := ["fadd", "fsub", "fmul", "fmax", "fmin"]

/-- The operation a name stands for, at whichever width the caller asks. The
    corpus applies each at `f32` and at `f64`, so the width is a parameter
    rather than fixed by the table. -/
def floatBinop {ty} (nm : String) (a b : V ty)
    (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Prog V L (V ty) :=
  match nm with
  | "fadd" => fadd a b h
  | "fsub" => fsub a b h
  | "fmul" => fmul a b h
  | "fmax" => fmax a b h
  | _      => fmin a b h

def allICmp : List (String × ICmpCond) :=
  [("eq", .eq), ("ne", .ne), ("ult", .ult), ("ule", .ule), ("ugt", .ugt),
   ("uge", .uge), ("slt", .slt), ("sle", .sle), ("sgt", .sgt), ("sge", .sge)]

def allFCmp : List (String × FloatCC) :=
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

/-- `sinf`, `cosf` and `powf`: the model is `Libm`'s definition and the
    machine runs `Lib.Math`'s CLIF, so these check the one against the other —
    at chosen values, and over sweeps that fold every result's bits into one
    word: arbitrary bit patterns (special values, huge and tiny arguments,
    every sign), and the small angles a rotary table takes. -/
def casesMath : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  for (nm, bits) in f32Cases do
    cs := cs ++ [⟨s!"sinf/{nm}", do ffi .sinf %[← fconst .f32 bits]⟩]
    cs := cs ++ [⟨s!"cosf/{nm}", do ffi .cosf %[← fconst .f32 bits]⟩]
  for (an, a) in f32Cases do
    for (bn, b) in [("2", f32Bits 2.0), ("0.5", f32Bits 0.5), ("-1", f32Bits (-1.0)),
                    ("0", f32Bits 0.0)] do
      cs := cs ++ [⟨s!"powf/{an}^{bn}", do
        ffi .powf %[← fconst .f32 a, ← fconst .f32 b]⟩]
  let fold (acc : V .i64) (s c p : V .f32) : Prog V L (V .i64) := do
    let w (v : V .f32) (sh : Nat) : Prog V L (V .i64) := do
      ishlImm (← uextend64 (← bitcast .i32 v)) sh
    bxor (← imul acc (← iconst64 1000003)) (← bxor (← w s 0) (← bxor (← w c 32) (← w p 16)))
  cs := cs ++ [⟨"libm/sweep-sin", do
    forLoopAcc (← iconst64 3000) (← iconst64 0) fun i acc => do
      let k ← imul (← iaddImm i 1) (← iconst64 (0x9E3779B97F4A7C15 - 2 ^ 64))
      let x ← bitcast .f32 (← ireduce32 (← ushrImm k 32))
      let yb ← bor (← band (← ireduce32 k) (← iconst32 0x80ffffff)) (← iconst32 0x3c000000)
      let y ← bitcast .f32 yb
      let z ← fconst .f32 0
      fold acc (← ffi .sinf %[x]) z z⟩]
  cs := cs ++ [⟨"libm/sweep-cos", do
    forLoopAcc (← iconst64 3000) (← iconst64 0) fun i acc => do
      let k ← imul (← iaddImm i 1) (← iconst64 (0x9E3779B97F4A7C15 - 2 ^ 64))
      let x ← bitcast .f32 (← ireduce32 (← ushrImm k 32))
      let yb ← bor (← band (← ireduce32 k) (← iconst32 0x80ffffff)) (← iconst32 0x3c000000)
      let y ← bitcast .f32 yb
      let z ← fconst .f32 0
      fold acc z (← ffi .cosf %[x]) z⟩]
  cs := cs ++ [⟨"libm/sweep-pow", do
    forLoopAcc (← iconst64 3000) (← iconst64 0) fun i acc => do
      let k ← imul (← iaddImm i 1) (← iconst64 (0x9E3779B97F4A7C15 - 2 ^ 64))
      let x ← bitcast .f32 (← ireduce32 (← ushrImm k 32))
      let yb ← bor (← band (← ireduce32 k) (← iconst32 0x80ffffff)) (← iconst32 0x3c000000)
      let y ← bitcast .f32 yb
      let z ← fconst .f32 0
      fold acc z z (← ffi .powf %[x, y])⟩]
  cs := cs ++ [
    ⟨"libm/angles", do
      forLoopAcc (← iconst64 3000) (← iconst64 0) fun i acc => do
        let x ← fmul (← fcvtFromSint .f32 i) (← fconst .f32 (f32Bits 0.37))
        let e ← fmul (← fcvtFromSint .f32 i) (← fconst .f32 (f32Bits (-0.001)))
        fold acc (← ffi .sinf %[x]) (← ffi .cosf %[x]) (← ffi .powf %[← fconst .f32 (f32Bits 10000.0), e])⟩]
  return cs

/-- Scratch inside the corpus arena, clear of the context slots. -/
def htCtxSlot : Nat := 0x80
def htKeyA : Nat := 0x90
def htKeyB : Nat := 0x98
def htVal : Nat := 0xA0
def htOut : Nat := 0xB0

/-- The hash table, run as one sequence: the cases share a world, so what each
    stores is the state the ones before it left.

    Every accessor in `ht.rs` reads table `0` regardless of the handle it is
    given, and iterates a `HashMap`, so `ht_get_entry` is asked here only while
    exactly one entry exists — an index into an unordered container is not
    something the implementation promises and not something to pin. -/
def casesHt : List (Case V L) := Id.run do
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

/-- An arena slot nothing writes, read as a zero the compiler cannot see: an
    operand built as `k + zero` is not a constant, so the cases below exercise
    the instruction Cranelift emits rather than its constant folder. -/
def opaqueSlot : Nat := 0xF0
def loadsSlot : Nat := 0xE0

def zeroAt (ty : ClifTy) : Prog V L (V ty) := do
  let a ← absAddr (← basePtr) opaqueSlot
  match ty with
  | .i8 => load_i8 a | .i16 => load_i16 a | .i32 => load32 a | .i64 => load64 a
  | t => load { ty := t } a

/-- An integer operand of type `ty` the compiler cannot fold. -/
def opq (ty : ClifTy) (k : Int) (h : ty.isInt = true := by decide) : Prog V L (V ty) := do
  iadd (← iconst ty k h) (← zeroAt ty) h

/-- Float operands given by their bits, equally opaque. -/
def opqF32 (bits : UInt64) : Prog V L (V .f32) := do
  bitcast .f32 (← opq .i32 (Int.ofNat bits.toNat))
def opqF64 (bits : UInt64) : Prog V L (V .f64) := do
  bitcast .f64 (← opq .i64 (Sem.signed .i64 bits))

def minOf (t : ClifTy) : Int := -((1 : Int) <<< (t.width - 1))
def maxOf (t : ClifTy) : Int := ((1 : Int) <<< (t.width - 1)) - 1

def ibinAll : List (String × IBin) :=
  [("sdiv", .sdiv), ("urem", .urem), ("srem", .srem), ("smin", .smin), ("smax", .smax),
   ("umin", .umin), ("umax", .umax), ("umulhi", .umulhi), ("smulhi", .smulhi)]

/-- Whether a pair traps: a zero divisor, or the one quotient that overflows. -/
def traps (k : IBin) (t : ClifTy) (x y : Int) : Bool :=
  match k with
  | .sdiv => y == 0 || (x == minOf t && y == -1)
  | .urem | .srem => y == 0
  | _ => false

/-- The integer families at one width. Operands straddle the sign boundary,
    since signed and unsigned readings of the same bits are what these differ on. -/
def intFamAt (tn : String) (t : ClifTy) (ht : t.isInt = true) : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  let pairs := [(7, 2), (-7, 2), (7, -2), (-7, -2), (minOf t, -1), (maxOf t, 3),
                (minOf t, 1), (-1, -1), (0x55, 3), (-1, 0x7f)]
  for (kn, k) in ibinAll do
    for (x, y) in pairs do
      unless traps k t x y do
        cs := cs ++ [⟨s!"{kn}.{tn}/{x}/{y}", do ibin k (← opq t x ht) (← opq t y ht) ht⟩]
  for (kn, k) in [("sshr", IShift.sshr), ("rotl", .rotl), ("rotr", .rotr)] do
    for x in [1, -1, minOf t, 0x5a] do
      for sh in [0, 1, t.width - 1, t.width, t.width + 3] do
        cs := cs ++ [⟨s!"{kn}.{tn}/{x}/{sh}", do
          ishift k (← opq t x ht) (← opq .i64 (Int.ofNat sh)) (by rw [ht]; rfl)⟩]
  for (kn, k) in [("bnot", IUn.bnot), ("iabs", .iabs), ("clz", .clz),
                  ("bswap", .bswap), ("bitrev", .bitrev)] do
    if hk : k.admits t = true then
      for x in [0, 1, -1, 3, minOf t, maxOf t, 0x1234, -0x1234] do
        cs := cs ++ [⟨s!"{kn}.{tn}/{x}", do iun k (← opq t x ht) hk⟩]
  return cs

def casesIntFam : List (Case V L) :=
  intFamAt "i8" .i8 rfl ++ intFamAt "i16" .i16 rfl ++ intFamAt "i32" .i32 rfl
    ++ intFamAt "i64" .i64 rfl

def f32Extra : List (String × UInt64) :=
  [("2.5", f32Bits 2.5), ("3.5", f32Bits 3.5), ("-2.5", f32Bits (-2.5)),
   ("-0.3", f32Bits (-0.3)), ("0.5", f32Bits 0.5), ("1e10", f32Bits 1e10),
   ("-1e10", f32Bits (-1e10)), ("3e9", f32Bits 3e9), ("0.1", f32Bits 0.1)]

def f64Extra : List (String × UInt64) :=
  [("-inf", f64Bits (-1.0/0.0)), ("0", 0), ("2.5", f64Bits 2.5), ("-3.5", f64Bits (-3.5)),
   ("-0.3", f64Bits (-0.3)), ("0.1", f64Bits 0.1), ("1e300", f64Bits 1e300),
   ("1e19", f64Bits 1e19), ("-1e19", f64Bits (-1e19)), ("1.0000001", f64Bits 1.0000001),
   ("2", f64Bits 2.0), ("-2.25", f64Bits (-2.25))]

def funAll : List (String × FUn) :=
  [("sqrt", .sqrt), ("fabs", .fabs), ("ceil", .ceil), ("floor", .floor),
   ("trunc", .trunc), ("nearest", .nearest)]

/-- The float families, scalar at both widths and lane-wise on `f32x4`.

    Results are stored whole; `f64` cases carry the `f64.` prefix, which is what
    tells the comparison to read a NaN at that width. -/
def casesFloatFam : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  let f32s := f32Cases ++ f32Extra
  let f64s := f64Cases ++ f64Extra
  for (kn, k) in funAll do
    for (xn, x) in f32s do
      cs := cs ++ [⟨s!"{kn}.f32/{xn}", do fun1 k (← opqF32 x)⟩]
    for (xn, x) in f64s do
      cs := cs ++ [⟨s!"f64.{kn}/{xn}", do fun1 k (← opqF64 x)⟩]
    cs := cs ++ [⟨s!"{kn}.f32x4/-2.5", do
      fun1 k (← splat .f32x4 (← opqF32 (f32Bits (-2.5))))⟩]
  let pairs32 := [("1.5", "-2.25"), ("1.5", "0"), ("0", "0"), ("-0", "1.5"), ("inf", "inf"),
                  ("sub", "0.5"), ("0.1", "3e9"), ("nan", "-inf")]
  let bits32 := fun (n : String) => ((f32s.find? (·.1 == n)).map (·.2)).getD 0
  let bits64 := fun (n : String) => ((f64s.find? (·.1 == n)).map (·.2)).getD 0
  for (kn, k) in [("fdiv", FBin.fdiv), ("fcopysign", .fcopysign)] do
    for (xn, yn) in pairs32 do
      cs := cs ++ [⟨s!"{kn}.f32/{xn}/{yn}", do fbin k (← opqF32 (bits32 xn)) (← opqF32 (bits32 yn))⟩]
    for (xn, yn) in [("1.5", "-3.5"), ("0.1", "-0"), ("1e300", "0.1"), ("-2.25", "2")] do
      cs := cs ++ [⟨s!"f64.{kn}/{xn}/{yn}", do fbin k (← opqF64 (bits64 xn)) (← opqF64 (bits64 yn))⟩]
  -- `fma` differs from `fmul` then `fadd` exactly where the product is not
  -- representable, which `0.1 · 10 - 1` is.
  let tri64 := [("0.1", "1e19", "-1e19"), ("1.5", "2", "-3.5"), ("-0", "2", "-0"),
                ("1e300", "1e300", "-inf"), ("sub", "0.1", "0"), ("inf", "0", "1.5"),
                ("1.0000001", "1.0000001", "-2")]
  for (an, bn, cn) in tri64 do
    cs := cs ++ [⟨s!"f64.fma/{an}/{bn}/{cn}", do
      fma (← opqF64 (bits64 an)) (← opqF64 (bits64 bn)) (← opqF64 (bits64 cn))⟩]
  for (an, bn, cn) in [("0.1", "3e9", "-1e10"), ("1.5", "-2.25", "0.5"), ("sub", "0.5", "-0"),
                       ("1e10", "1e10", "-inf")] do
    cs := cs ++ [⟨s!"fma.f32/{an}/{bn}/{cn}", do
      fma (← opqF32 (bits32 an)) (← opqF32 (bits32 bn)) (← opqF32 (bits32 cn))⟩]
  cs := cs ++ [⟨"fma.f32x4/0.1/3e9/-1e10", do
    fma (← splat .f32x4 (← opqF32 (bits32 "0.1"))) (← splat .f32x4 (← opqF32 (bits32 "3e9")))
        (← splat .f32x4 (← opqF32 (bits32 "-1e10")))⟩]
  -- conversions: saturation at both ends and NaN, rounding of wide integers
  for (xn, x) in f32s do
    cs := cs ++ [⟨s!"toSint32.f32/{xn}", do fcvtToSint .i32 (← opqF32 x)⟩,
                 ⟨s!"toSint64.f32/{xn}", do fcvtToSint .i64 (← opqF32 x)⟩]
  for (xn, x) in f64s do
    cs := cs ++ [⟨s!"toSint32.f64/{xn}", do fcvtToSint .i32 (← opqF64 x)⟩,
                 ⟨s!"demote/{xn}", do fdemote (← opqF64 x)⟩]
  for x in (ints ++ [0x7fffffffffffffff, 0x20000001, 0x1000000000000801] : List Int) do
    cs := cs ++ [⟨s!"fromUint.f32/{x}", do fcvtFromUint .f32 (← opq .i64 x)⟩,
                 ⟨s!"f64.fromUint/{x}", do fcvtFromUint .f64 (← opq .i64 x)⟩,
                 ⟨s!"fromUint32.f32/{x}", do fcvtFromUint .f32 (← opq .i32 x)⟩,
                 ⟨s!"fromSint.f32/{x}", do fcvtFromSint .f32 (← opq .i64 x)⟩]
  return cs

/-- The narrow loads, over bytes whose top bits are set so zero- and
    sign-extension differ. -/
def casesLoads : List (Case V L) := Id.run do
  let mut cs : List (Case V L) := []
  cs := cs ++ [⟨"loads/put", do
    storeI64 (← opq .i64 (-0x7f7e7d7c7b7a7979)) (← absAddr (← basePtr) loadsSlot)
    iconst64 0⟩]
  for (kn, k) in [("uload16", LoadKind.uload16), ("sload16", .sload16), ("sload32", .sload32),
                  ("uload32", .uload32), ("sload8", .sload8)] do
    for off in [0, 1, 3] do
      cs := cs ++ [⟨s!"{kn}/{off}", do
        load { kind := k, ty := .i64 } (← absAddr (← basePtr) (loadsSlot + off))⟩]
  -- every width change, and the narrow stores built from them
  for (x : Int) in [-1, 0x1234, -0x1234, 0x7f, 0x80, 0x12345678] do
    cs := cs ++ [⟨s!"ireduce.i8/{x}", do ireduce .i8 (← opq .i64 x)⟩,
                 ⟨s!"ireduce.i16/{x}", do ireduce .i16 (← opq .i32 x)⟩,
                 ⟨s!"uextend.i16/{x}", do uextend .i16 (← opq .i8 x)⟩,
                 ⟨s!"sextend.i32/{x}", do sextend .i32 (← opq .i16 x)⟩,
                 ⟨s!"sextend.i64/{x}", do sextend .i64 (← opq .i8 x)⟩,
                 ⟨s!"uextend.i64/{x}", do uextend .i64 (← opq .i16 x)⟩]
  cs := cs ++ [⟨"istore16+32", do
    let p ← absAddr (← basePtr) loadsSlot
    storeI64 (← iconst64 0) p
    istore16 (← opq .i64 (-0x0123456789abcdef)) p
    istore32 (← opq .i64 0x7eadbeef) (← absAddr (← basePtr) (loadsSlot + 4))
    load64 p⟩]
  cs := cs ++ [⟨"uload16.i32/2", do
    load { kind := .uload16, ty := .i32 } (← absAddr (← basePtr) (loadsSlot + 2))⟩]
  return cs

def atomicSlot : Nat := 0xC0

/-- The atomics at every width: each read-modify-write answers the old value
    and leaves its result, which the next case reads back; a compare-exchange
    that matches and one that does not. -/
def atomicsAt (tn : String) (t : ClifTy) (ht : t.isInt = true) : List (Case V L) := Id.run do
  let p : Prog V L (V .i64) := do absAddr (← basePtr) atomicSlot
  let mut cs : List (Case V L) := []
  cs := cs ++ [⟨s!"atomic.{tn}/store", do
    atomicStore (← opq t (-0x5a5a5a5a5a5a5a5b) ht) (← p)
    atomicLoad t (← p)⟩]
  for (kn, k) in [("add", AtomicRmw.add), ("sub", .sub), ("and", .and), ("nand", .nand),
                  ("or", .or), ("xor", .xor), ("xchg", .xchg), ("umin", .umin),
                  ("umax", .umax), ("smin", .smin), ("smax", .smax)] do
    cs := cs ++ [⟨s!"atomic.{tn}/{kn}/old", do atomicRmw k (← p) (← opq t 0x3c ht)⟩,
                 ⟨s!"atomic.{tn}/{kn}/new", do atomicLoad t (← p)⟩]
  cs := cs ++ [⟨s!"atomic.{tn}/cas.miss", do
                  atomicCas (← p) (← opq t 12345 ht) (← opq t 7 ht)⟩,
               ⟨s!"atomic.{tn}/cas.hit", do
                  let old ← atomicLoad t (← p)
                  atomicCas (← p) old (← opq t (-3) ht)⟩,
               ⟨s!"atomic.{tn}/cas.after", do fence; atomicLoad t (← p)⟩]
  return cs

def casesAtomic : List (Case V L) :=
  atomicsAt "i8" .i8 rfl ++ atomicsAt "i16" .i16 rfl ++ atomicsAt "i32" .i32 rfl
    ++ atomicsAt "i64" .i64 rfl

def extSlot : Nat := 0x100

/-- The C library's memory functions, called directly: a copy, a copy between
    overlapping ranges, a fill and a length, each read back, and a heap
    allocation used and released. -/
def casesExt : List (Case V L) := Id.run do
  let at_ (off : Nat) : Prog V L (V .i64) := do absAddr (← basePtr) (extSlot + off)
  let mut cs : List (Case V L) := []
  cs := cs ++ [⟨"libc/present", do libPresent .c⟩]
  cs := cs ++ [⟨"memcpy/ret", do
    storeI64 (← opq .i64 0x0807060504030201) (← at_ 0)
    storeI64 (← opq .i64 0x100f0e0d0c0b0a09) (← at_ 8)
    let r ← ext (.c .memcpy) %[← at_ 0x20, ← at_ 0, ← iconst64 13]
    isub r (← at_ 0x20)⟩,
    ⟨"memcpy/lo", do load64 (← at_ 0x20)⟩, ⟨"memcpy/hi", do load64 (← at_ 0x28)⟩]
  cs := cs ++ [⟨"memmove/overlap", do
    let _ ← ext (.c .memmove) %[← at_ 3, ← at_ 0, ← iconst64 8]
    load64 (← at_ 0)⟩, ⟨"memmove/hi", do load64 (← at_ 8)⟩]
  cs := cs ++ [⟨"memset", do
    let _ ← ext (.c .memset) %[← at_ 0x41, ← iconst32 0x1ab, ← iconst64 5]
    load64 (← at_ 0x40)⟩]
  cs := cs ++ [⟨"strlen", do
    storeI64 (← opq .i64 0x006f6c6c6568) (← at_ 0x60)
    ext (.c .strlen) %[← at_ 0x60]⟩,
    ⟨"strlen/empty", do ext (.c .strlen) %[← at_ 0x80]⟩]
  -- Zeroed memory, written and read back, then released. Without the library
  -- the answer is `-1`, which nothing here dereferences.
  cs := cs ++ [⟨"calloc/free", do
    let p ← ext (.c .calloc) %[← opq .i64 4, ← iconst64 8]
    let r ← ifte (jTys := [.i64]) .eq p (← iconst64 (-1)) (do pure %[← iconst64 (-1)]) (do
      storeI64 (← opq .i64 0x1234) (← iaddImm p 8)
      let v ← load64 (← iaddImm p 8)
      let z ← load64 (← iaddImm p 24)
      ext (.c .free) %[p]
      pure %[← bxor v z])
    pure r.head⟩,
    ⟨"free/null", do
      ext (.c .free) %[← iconst64 0]
      iconst64 7⟩]
  return cs

/-- The performance controls a system grants without privilege: locking a
    page and unlocking it, and lowering the calling thread's priority. The
    model's system grants everything, so these check the machine does too. -/
def casesOs : List (Case V L) :=
  [⟨"os/lock", do memLock (← absAddr (← basePtr) 0x1000) (← iconst64 4096)⟩,
   ⟨"os/unlock", do memUnlock (← absAddr (← basePtr) 0x1000) (← iconst64 4096)⟩,
   ⟨"os/priority-lower", do threadPriority (← iconst32 (-1))⟩]

/-- Every case, named so a failure says which one. -/
def cases : List (Case V L) :=
  casesInt ++ casesShift ++ casesUnary ++ casesCmp ++ casesFloat ++ casesConv
    ++ casesVec ++ casesCfg ++ casesPminmax ++ casesBr ++ casesDLoop ++ casesCont ++ casesVecInt
    ++ casesDCont ++ casesMath ++ casesHt ++ casesIntFam ++ casesFloatFam ++ casesLoads ++ casesAtomic ++ casesExt
    ++ casesOs

/-- Every case, storing its result at its own stride in the output buffer. -/
def body : Prog V L Unit := do
  let outPtr ← outPtr
  for (c, k) in cases.zipIdx do
    let r ← c.run
    store r (← iadd outPtr (← iconst64 (STRIDE * k)))

/-- The names, in the order the body stores them. -/
def caseNames : List String := (cases (V := Prog.Slot) (L := Prog.Lvl)).map (·.name)

/-- The term the cases denote, an ordinary term like every other. -/
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
  .ok { arena := ByteArray.mk (Array.replicate 0x200 0),
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
  match Blocks.run Sem.noLocals f entryArgVals { mem := m } with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok obs w => .ok (obs, w.mem.out)

/-- The corpus run through the interpreter, which is what the JIT is compared
    against. -/
def expected : Except String ByteArray := do
  let m ← startWorld
  match Sem.run { env } entryArgVals { mem := m } code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

/-- The term and its compiled form agree, on both the observation trace and the
    memory they leave behind. -/
def viaTerm : Except String (List Sem.Obs × ByteArray) := do
  let m ← startWorld
  match Sem.run { env } entryArgVals { mem := m } code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok obs w => .ok (obs, w.mem.out)

-- ---------------------------------------------------------------------------
-- The same calls with every library missing
-- ---------------------------------------------------------------------------

/-- Library calls on a machine without the libraries: each function answers
    `-1` and touches nothing, and each probe answers `0`. The test points each
    call at a library file that does not exist to be that machine. The C
    library is part of every process, so it is never absent. -/
def absentCases : List (Case V L) :=
  [⟨"cuInit", do ext (.cuda .init) %[← iconst32 0]⟩,
   ⟨"cuMemAlloc", do ext (.cuda .memAlloc) %[← absAddr (← basePtr) extSlot, ← iconst64 64]⟩,
   ⟨"cuda/present", do libPresent .cuda⟩,
   -- what the allocation would have written is still zero
   ⟨"untouched", do load64 (← absAddr (← basePtr) extSlot)⟩]

def absentBody : Prog V L Unit := do
  let outPtr ← outPtr
  for (c, k) in absentCases.zipIdx do
    let r ← c.run
    store r (← iadd outPtr (← iconst64 (STRIDE * k)))

def absentNames : List String := (absentCases (V := Prog.Slot) (L := Prog.Lvl)).map (·.name)

def absentProgram : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 absentBody)]

def absentOutBytes : Nat := absentNames.length * STRIDE

/-- What the absent run leaves in the output buffer. -/
def absentExpected : Except String ByteArray := do
  let _ ← Prog.emitChecked absentBody
  let args : List Sem.V :=
    [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
     .sc .i64 8, .sc .i64 (Sem.regionBase .out), .sc .i64 absentOutBytes.toUInt64]
  let m : Sem.Mem := { arena := ByteArray.mk (Array.replicate 0x200 0),
                       data := ByteArray.mk (Array.replicate 8 0),
                       out := ByteArray.mk (Array.replicate absentOutBytes 0) }
  match Sem.run { env := (Prog.run absentBody).2.1 } args
      { mem := m, present := fun _ => false } (Prog.emit absentBody) with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

end HProgCorpus

open AlgorithmLib in
def Host.Corpus.main (args : List String) : IO Unit := do
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
        functions := clif, required_memory := 0x200
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
      match HProgCorpus.absentExpected with
      | .error e => throw (IO.userError s!"interpreting the absent corpus: {e}")
      | .ok abytes =>
          let aclif ← AlgorithmLib.Prog.orDie HProgCorpus.absentProgram
          emitArtifacts dir #[artifactEntry "hprog_absent_corpus" {
            functions := aclif, required_memory := 0x200
          }]
          let aj := Lean.Json.mkObj [
            ("stride", Lean.toJson HProgCorpus.STRIDE),
            ("names", Lean.toJson HProgCorpus.absentNames),
            ("modes", Lean.toJson (HProgCorpus.absentNames.map fun _ => 0)),
            ("expected", Lean.toJson (abytes.toList.map (·.toNat)))]
          let aDir := System.FilePath.mk dir / "hprog_absent_corpus"
          IO.FS.createDirAll aDir
          IO.FS.writeFile (aDir / "expected.json") aj.compress
          IO.println s!"absent corpus: {HProgCorpus.absentNames.length} cases"
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

#eval ShipScan.check "Host.Corpus" `Host.Corpus.main
  (gatedElsewhere := "main, which refuses on emitChecked before emitting")
