module
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `sinf`, `cosf`, `powf`, stated over IEEE doubles

What the three single-precision functions a program may call compute, as a
sequence of IEEE operations: every step here is one instruction of
`Surface.LibMath`, which mirrors this definition operation for operation, so
the two agree bit for bit on every input by construction and the corpus checks
it. The same bits on every platform, which the C libraries' own functions do
not promise.

**NaN.** A NaN argument answers itself, quietened; an infinite argument to
`sinf` or `cosf`, and a negative base to a non-integer power, answer the quiet
NaN `0x7fc00000`. The functions take and give bits: float arithmetic leaves a
NaN's sign and payload to the machine, machines differ, and Lean's own floats
keep neither, so floats appear here only where no NaN can.

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

/-- `2/π`, 256 bits after the binary point, most significant word first. -/
def twoOverPi : Array UInt64 :=
  #[0xa2f9836e4e441529, 0xfc2757d1f534ddc0, 0xdb6295993c439041, 0xfe5163abdebbc561]

/-- `π/2 · 2⁻⁶⁴`: a 64-bit fraction of a quadrant, to radians. -/
def pio2Scaled : Float := Float.ofBits 0x3FF921FB54442D18 * Float.ofBits 0x3BF0000000000000

/-- `n!`, as a double. -/
def factF (n : Nat) : Float := Float.ofNat ((List.range n).foldl (fun a i => a * (i + 1)) 1)

/-- `sin r = r + r·z·(s₀ + z(s₁ + …))`, `z = r²`, `sᵢ = (-1)^(i+1)/(2i+3)!`. -/
def sinC : List Float :=
  (List.range 8).map fun i => (if i % 2 == 0 then -1.0 else 1.0) / factF (2 * i + 3)

/-- `cos r = 1 + z·(c₀ + z(c₁ + …))`, `cᵢ = (-1)^(i+1)/(2i+2)!`. -/
def cosC : List Float :=
  (List.range 9).map fun i => (if i % 2 == 0 then -1.0 else 1.0) / factF (2 * i + 2)

/-- `ln m = 2s + 2s·z·(1/3 + z(1/5 + …))`, `s = (m-1)/(m+1)`, `z = s²`. -/
def lnC : List Float := (List.range 11).map fun i => 1.0 / Float.ofNat (2 * i + 3)

/-- `eᵍ = 1 + g·(1 + g(1/2! + …))`. -/
def expC : List Float := (List.range 16).map fun i => 1.0 / factF (i + 1)

def ln2 : Float := Float.ofBits 0x3FE62E42FEFA39EF
def invLn2 : Float := Float.ofBits 0x3FF71547652B82FE
def sqrt2 : Float := Float.ofBits 0x3FF6A09E667F3BCD

/-- `c₀ + z(c₁ + z(… + z(cₙ + z·0)))`, innermost first. -/
def horner (cs : List Float) (z : Float) : Float :=
  cs.foldr (fun c acc => c + z * acc) 0.0

def sinPoly (r : Float) : Float := let z := r * r; r + r * z * horner sinC z
def cosPoly (r : Float) : Float := let z := r * r; 1.0 + z * horner cosC z

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
    the fraction by less than `2⁻¹⁰⁰`. -/
def reduce (ax : UInt32) : UInt64 × Float :=
  let ef := (ax >>> 23).toUInt64
  let m := ((ax &&& 0x7fffff) ||| 0x800000).toUInt64
  -- the first bit of 2/π taken, less one: max(0, e - 2) where e = ef - 150
  let s1 := if ef ≥ 152 then ef - 152 else 0
  let w := (s1 >>> 6).toNat
  let o := s1 &&& 63
  let t0 := twoOverPi[w]!
  let t1 := twoOverPi[w + 1]!
  let t2 := twoOverPi[w + 2]!
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
  let k : UInt64 := (bits64 p0 p1 p2 q &&& (3 : UInt64)) + (f >>> (63 : UInt64))
  (k &&& 3, Int64.toFloat f.toInt64 * pio2Scaled)

/-- What a non-finite argument gives, on the bits: a NaN, quietened, and for
    an infinity the quiet NaN `0x7fc00000`. Stated on the bits because float
    arithmetic leaves a NaN's sign and payload to the machine, and machines
    differ. -/
def nanOf (b : UInt32) : UInt32 :=
  if (b &&& 0x7fffff) == 0 then 0x7fc00000 else b ||| 0x400000

/-- **`sinf`**, on bits. -/
def sinf (b : UInt32) : UInt32 :=
  let ax := b &&& 0x7fffffff
  if ax ≥ 0x7f800000 then nanOf b
  else if ax < 0x3f000000 then (sinPoly (Float32.ofBits b).toFloat).toFloat32.toBits
  else
    let (k, r) := reduce ax
    let v := if k == 0 then sinPoly r else if k == 1 then cosPoly r
      else if k == 2 then -(sinPoly r) else -(cosPoly r)
    (if b >>> 31 == 1 then -v else v).toFloat32.toBits

/-- **`cosf`**, on bits. -/
def cosf (b : UInt32) : UInt32 :=
  let ax := b &&& 0x7fffffff
  if ax ≥ 0x7f800000 then nanOf b
  else if ax < 0x3f000000 then (cosPoly (Float32.ofBits b).toFloat).toFloat32.toBits
  else
    let (k, r) := reduce ax
    (if k == 0 then cosPoly r else if k == 1 then -(sinPoly r)
      else if k == 2 then -(cosPoly r) else sinPoly r).toFloat32.toBits

/-- `log₂` of a finite, nonzero double. -/
def log2Abs (d : Float) : Float :=
  let db := d.toBits
  let e := ((db >>> 52) &&& (0x7ff : UInt64)).toInt64 - 1023
  let m := Float.ofBits ((db &&& 0x000fffffffffffff) ||| 0x3ff0000000000000)
  let big := m > sqrt2
  let m := if big then m * 0.5 else m
  let e := if big then e + 1 else e
  let s := (m - 1.0) / (m + 1.0)
  let z := s * s
  let s2 := s + s
  let lnm := s2 + s2 * z * horner lnC z
  Int64.toFloat e + lnm * invLn2

/-- `2^t`, for `-152 < t < 129`. -/
def exp2 (t : Float) : Float :=
  let n := (t + 0.5).floor
  let g := (t - n) * ln2
  let p := 1.0 + g * horner expC g
  p * Float.ofBits ((n.toInt64.toUInt64 + 1023) <<< 52)

/-- **`powf`**, on bits, with C11 F.10.4.4's special cases. -/
def powf (xb yb : UInt32) : UInt32 :=
  let ax := xb &&& 0x7fffffff
  let ay := yb &&& 0x7fffffff
  let yd := (Float32.ofBits yb).toFloat
  let yneg := yb >>> 31 == 1
  let xneg := xb >>> 31 == 1
  if ay == 0 then 0x3f800000
  else if xb == 0x3f800000 then 0x3f800000
  else if ax > 0x7f800000 then nanOf xb
  else if ay > 0x7f800000 then nanOf yb
  else if ay == 0x7f800000 then
    if ax == 0x3f800000 then 0x3f800000
    else if (ax < 0x3f800000) != yneg then 0 else 0x7f800000
  else
    let yInt := yd.floor == yd
    let half := yd * 0.5
    let odd := yInt && half.floor != half
    let sign : UInt32 := if odd && xneg then 0x80000000 else 0
    if ax == 0 then sign ||| (if yneg then 0x7f800000 else 0)
    else if ax == 0x7f800000 then sign ||| (if yneg then 0 else 0x7f800000)
    else if xneg && !yInt then 0x7fc00000
    else
      let t := yd * log2Abs (Float32.ofBits xb).toFloat
      let v := if t ≥ 129.0 then Float.ofBits 0x7ff0000000000000
        else if t ≤ -152.0 then 0.0 else exp2 t
      (if sign != 0 then -v else v).toFloat32.toBits

end AlgorithmLib.Libm
