module
public import AlgorithmLib.Host.World
public import AlgorithmLib.Vocab.Libm
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `sinf`, `cosf`, `powf`, stated over IEEE doubles

What the three single-precision functions a program may call compute, as a
sequence of IEEE operations on bits: every step here is one instruction of
`Surface.LibMath`, computed as the CLIF semantics computes that instruction
(`Sem.evalOp`: an operation on doubles reads its operands' bits as doubles and
writes its result's bits), so the two agree bit for bit on every input by
construction, and the corpus checks it against the machine. The same bits on
every platform, which the C libraries' own functions do not promise.

**NaN.** A NaN argument answers itself, quietened; an infinite argument to
`sinf` or `cosf`, and a negative base to a non-integer power, answer the quiet
NaN `0x7fc00000`. The functions take and give bits, and a value in between is
the bits of a double: float arithmetic leaves a NaN's sign and payload to the
machine, machines differ, and Lean's own floats keep neither, so each function
takes the platform's NaN rule (`nan`), as the semantics does.

**Method.** Each works in double precision and rounds once to single at the
end. `sinf` and `cosf` reduce the argument by `π/2` exactly, in integer
arithmetic against 256 bits of `2/π` (Payne–Hanek), for any finite input, then
evaluate Taylor polynomials whose truncation error is below `2⁻⁶⁰` on
`[-π/4, π/4]`. `powf` handles the special cases of C11 F.10.4.4 and otherwise
takes `2^(y · log₂|x|)`, with `log₂` by the `atanh` series and `2^f` by Taylor
series, both far below a single-precision ulp. The double result is within
about `2⁻⁴⁴` relative of the exact value, so rounding it gives the correctly
rounded single except when the exact value lies that close to a midpoint.
-/

namespace AlgorithmLib.Libm

open AlgorithmLib.IR AlgorithmLib.HProg.Sem

/-- The operations, on the bits of doubles, as the CLIF semantics computes
    them on a platform whose float arithmetic leaves a NaN as `nan` says
    (`Mem.nan`). -/
abbrev D := UInt64

variable (nan : ClifTy → UInt64 → UInt64)

def dadd (a b : D) : D := nanFix nan .f64 (ofF64 (f64 a + f64 b))
def dsub (a b : D) : D := nanFix nan .f64 (ofF64 (f64 a - f64 b))
def dmul (a b : D) : D := nanFix nan .f64 (ofF64 (f64 a * f64 b))
def ddiv (a b : D) : D := nanFix nan .f64 (ofF64 (f64 a / f64 b))
def dneg (a : D) : D := a ^^^ 0x8000000000000000
def dfloor (a : D) : D := nanFix nan .f64 (funOp .floor .f64 a)
def dcmp (c : FloatCC) (a b : D) : Bool := fcmpBits c .f64 a b
/-- An `i64` converted, rounded once. -/
def dofI64 (x : UInt64) : D := ofF64 (Float.ofInt (signed .i64 x))
/-- A double converted to `i64`, saturating. -/
def dtoI64 (a : D) : UInt64 :=
  match satToSint .i64 (f64 a) with
  | .sc _ x => x
  | _ => 0
/-- A single's bits widened to a double's, and a double's narrowed to a
    single's. -/
def promote (b : UInt32) : D := nanFix nan .f64 (ofF64 (f32 b.toUInt64).toFloat)
def demote (a : D) : UInt32 := (nanFix nan .f32 (ofF32 (f64 a).toFloat32)).toUInt32

/-- A constant, by its bits. -/
def k (x : Float) : D := x.toBits

/-- `c₀ + z(c₁ + z(… + z(cₙ + z·0)))`, innermost first. -/
def horner (cs : List Float) (z : D) : D :=
  cs.foldr (fun c acc => dadd nan (k c) (dmul nan z acc)) (k 0.0)

def sinPoly (r : D) : D := let z := dmul nan r r; dadd nan r (dmul nan (dmul nan r z) (horner nan sinC z))
def cosPoly (r : D) : D := let z := dmul nan r r; dadd nan (k 1.0) (dmul nan z (horner nan cosC z))

/-- The 64 bits of `P = p₂·2¹²⁸ + p₁·2⁶⁴ + p₀` from bit `off`, for
    `1 ≤ off < 192`. -/
def bits64 (p0 p1 p2 off : UInt64) : UInt64 :=
  if off < 64 then (p0 >>> off) ||| (p1 <<< (64 - off))
  else if off == 64 then p1
  else if off < 128 then (p1 >>> (off - 64)) ||| (p2 <<< (128 - off))
  else if off == 128 then p2
  else p2 >>> (off - 128)

/-- The high half of a 64 × 64-bit product. -/
def mulhi (a b : UInt64) : UInt64 := ((a.toNat * b.toNat) >>> 64).toUInt64

/-- **Reduction by `π/2`** of a finite single `|x| ≥ 1/2`, given its bits:
    the quadrant `k mod 4` and the remainder `r`, `|r| ≤ π/4`, with
    `|x| = k·π/2 + r`.

    Write `|x| = m·2^e`, `m` 24 bits. The terms of `m·2^e·(2/π)` whose bits of
    `2/π` sit more than two places above `2^e` are multiples of four and fall
    out; the 128 bits from there on, times `m`, give the quadrant in two bits
    and the fraction in the next 64. What is left of `2/π` beyond them moves
    the fraction by less than `2⁻¹⁰⁰`. A finite single's exponent puts the
    first word of `2/π` taken at `0` or `1`. -/
def reduce (ax : UInt32) : UInt64 × D :=
  let ef := (ax >>> 23).toUInt64
  let m := ((ax &&& 0x7fffff) ||| 0x800000).toUInt64
  -- the first bit of 2/π taken, less one: max(0, e - 2) where e = ef - 150
  let s1 := if ef ≥ 152 then ef - 152 else 0
  let w0 := s1 >>> 6 == 0
  let o := s1 &&& 63
  let t0 := if w0 then twoOverPi[0]! else twoOverPi[1]!
  let t1 := if w0 then twoOverPi[1]! else twoOverPi[2]!
  let t2 := if w0 then twoOverPi[2]! else twoOverPi[3]!
  let hi := if o == 0 then t0 else (t0 <<< o) ||| (t1 >>> (64 - o))
  let lo := if o == 0 then t1 else (t1 <<< o) ||| (t2 >>> (64 - o))
  let p0 := m * lo
  let c := mulhi m lo
  let t := m * hi
  let p1 := t + c
  let p2 := mulhi m hi + (if p1 < t then 1 else 0)
  -- the integer part of m·2^e·(2/π) mod 4 starts at bit q of P
  let q : UInt64 := if ef ≥ 152 then 126 else 278 - ef
  let f := bits64 p0 p1 p2 (q - 64)
  let kq : UInt64 := (bits64 p0 p1 p2 q &&& (3 : UInt64)) + (f >>> (63 : UInt64))
  (kq &&& 3, dmul nan (dofI64 f) (k pio2Scaled))

/-- What a non-finite argument gives, on the bits: a NaN, quietened, and for
    an infinity the quiet NaN `0x7fc00000`. -/
def nanOf (b : UInt32) : UInt32 :=
  if (b &&& 0x7fffff) == 0 then 0x7fc00000 else b ||| 0x400000

/-- The four quadrants' values, by `kq`. -/
def byQuadrant (kq : UInt64) (v0 v1 v2 v3 : D) : D :=
  if kq == 0 then v0 else if kq == 1 then v1 else if kq == 2 then v2 else v3

/-- **`sinf`**, on bits. -/
def sinf (b : UInt32) : UInt32 :=
  let ax := b &&& 0x7fffffff
  if ax ≥ 0x7f800000 then nanOf b
  else if ax < 0x3f000000 then demote nan (sinPoly nan (promote nan b))
  else
    let (kq, r) := reduce nan ax
    let s := sinPoly nan r
    let c := cosPoly nan r
    let v := byQuadrant kq s c (dneg s) (dneg c)
    demote nan (if b >>> 31 != 0 then dneg v else v)

/-- **`cosf`**, on bits. -/
def cosf (b : UInt32) : UInt32 :=
  let ax := b &&& 0x7fffffff
  if ax ≥ 0x7f800000 then nanOf b
  else if ax < 0x3f000000 then demote nan (cosPoly nan (promote nan b))
  else
    let (kq, r) := reduce nan ax
    let s := sinPoly nan r
    let c := cosPoly nan r
    demote nan (byQuadrant kq c (dneg s) (dneg c) s)

/-- `log₂` of a finite, nonzero double. -/
def log2Abs (d : D) : D :=
  let e := ((d >>> 52) &&& 0x7ff) - 1023
  let m := (d &&& 0x000fffffffffffff) ||| 0x3ff0000000000000
  let big := dcmp .gt m (k sqrt2)
  let m := if big then dmul nan m (k 0.5) else m
  let e := if big then e + 1 else e
  let s := ddiv nan (dsub nan m (k 1.0)) (dadd nan m (k 1.0))
  let z := dmul nan s s
  let s2 := dadd nan s s
  let lnm := dadd nan s2 (dmul nan (dmul nan s2 z) (horner nan lnC z))
  dadd nan (dofI64 e) (dmul nan lnm (k invLn2))

/-- `2^t`, for `-152 < t < 129`. -/
def exp2 (t : D) : D :=
  let n := dfloor nan (dadd nan t (k 0.5))
  let g := dmul nan (dsub nan t n) (k ln2)
  let p := dadd nan (k 1.0) (dmul nan g (horner nan expC g))
  dmul nan p ((dtoI64 n + 1023) <<< 52)

/-- **`powf`**, on bits, with C11 F.10.4.4's special cases. -/
def powf (xb yb : UInt32) : UInt32 :=
  let ax := xb &&& 0x7fffffff
  let ay := yb &&& 0x7fffffff
  let yd := promote nan yb
  let yneg := yb >>> 31 != 0
  let xneg := xb >>> 31 != 0
  if ay == 0 then 0x3f800000
  else if xb == 0x3f800000 then 0x3f800000
  else if max ax ay > 0x7f800000 then nanOf (if ax > 0x7f800000 then xb else yb)
  else if ay == 0x7f800000 then
    if ax == 0x3f800000 then 0x3f800000
    else if (ax < 0x3f800000) != yneg then 0 else 0x7f800000
  else
    let yInt := dcmp .eq (dfloor nan yd) yd
    let half := dmul nan yd (k 0.5)
    let odd := yInt && dcmp .ne (dfloor nan half) half
    let sign : UInt32 := if odd && xneg then 0x80000000 else 0
    if ax == 0 then sign ||| (if yneg then 0x7f800000 else 0)
    else if ax == 0x7f800000 then sign ||| (if yneg then 0 else 0x7f800000)
    else if xneg && !yInt then 0x7fc00000
    else
      let t := dmul nan yd (log2Abs nan (promote nan xb))
      let v := if dcmp .ge t (k 129.0) then 0x7ff0000000000000
        else if dcmp .le t (k (-152.0)) then k 0.0 else exp2 nan t
      demote nan (if sign != 0 then dneg v else v)

end AlgorithmLib.Libm
