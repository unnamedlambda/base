module
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The constants of `sinf`, `cosf`, `powf`

The bits of `2/π` and the series coefficients the three functions use, which
`Surface.LibMath` writes into its code and `Host.Libm`'s statement of them
reads. Each is a double, and is used by its bits.
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

end AlgorithmLib.Libm
