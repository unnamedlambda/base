module
public import AlgorithmLib.Host.World
meta import AlgorithmLib.Host.World
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Kernel.Hw` — what the hardware approximates

Apart from `Kernel.Run` so that what a kernel body computes (`ML.Num.Bits`)
can name the hardware without the semantics of a launch.
-/

namespace AlgorithmLib.Kernel

open AlgorithmLib.IR AlgorithmLib.HProg.Sem

/-- A GPU's NaN: `0x7fffffff` at f32, its f64 counterpart, whatever the
    operation computed. -/
def gpuNaN (t : ClifTy) (_ : UInt64) : UInt64 :=
  match t with
  | .f64 => 0x7fffffffffffffff
  | _ => 0x7fffffff

/-- `c` plus the products of bf16 values, in f64, rounded once to f32. -/
def bfDot (as bs : List UInt64) (c : UInt64) : UInt64 :=
  let v (b : UInt64) : Float := Float32.toFloat (f32 (b <<< 16))
  ofF32 ((as.zip bs).foldl (fun acc (a, b) => acc + v a * v b) (Float32.toFloat (f32 c))).toFloat32

/-- `c` plus the products of f16 values, exactly, rounded once to f16; an
    infinite or NaN operand gives `0x7fff`. -/
def hDot (as bs : List UInt64) (c : UInt64) : UInt64 :=
  let val (h : UInt64) : Int :=
    let (s, _, n, e) := Intr.f16Parts h
    (if s then -1 else 1) * ((n * 2 ^ (e + 48).toNat : Nat) : Int)
  let special (h : UInt64) : Bool := (Intr.f16Parts h).2.1 = 31
  if special c ∨ as.any special ∨ bs.any special then 0x7fff else
  -- every product and `c` a whole number of `2^-48`
  let sum : Int := (as.zip bs).foldl (fun (acc : Int) (a, b) => acc + val a * val b) (val c * 2 ^ 48)
  if sum = 0 then 0 else Intr.f16Near (sum < 0) sum.natAbs (-96)

/-- **The hardware's approximations**: the bits each approximating
    intrinsic (`ex2`, `rcp`, `lg2`, `sin`, `cos`, `sqrt` and `rsqrt`, each
    `.approx.f32`) gives on
    each argument's bits. No formula states them; a claim about a kernel that
    uses them holds for every `Hw`, or for those within a stated error.
    `Hw.rounded` is the exactly rounded functions, which the model runs where
    nothing else is said. The hardware's NaN results are `nan`'s. -/
structure Hw where
  approx : Intr → UInt64 → UInt64
  /-- The f32 bits the hardware's bf16 `mma` leaves for one sum: `c` (f32
      bits) plus the products of the bf16 values `as` and `bs` (their bits),
      in order. A tensor core does not round each step as f32 arithmetic
      would; by default the sum is taken in f64 and rounded once. -/
  mmaF : List UInt64 → List UInt64 → UInt64 → UInt64 := bfDot
  /-- The f16 bits the hardware's f16 `mma` leaves for one sum: `c` (f16
      bits) plus the products of the f16 values `as` and `bs`; by default
      exact, rounded once. -/
  mmaH : List UInt64 → List UInt64 → UInt64 → UInt64 := hDot
  /-- What the hardware's float arithmetic leaves where a result is a NaN
      (`Mem.nan`): by default a GPU's, one NaN whatever the operands. -/
  nan : ClifTy → UInt64 → UInt64 := gpuNaN

/-- A function of `Float`, rounded to `Float32`. -/
def viaFloat (f : Float → Float) (x : UInt64) : UInt64 :=
  ofF32 (Float.toFloat32 (f (Float32.toFloat (f32 x))))

def Hw.rounded : Hw where
  approx
    | .ex2Approx => viaFloat Float.exp2
    | .rcpApprox => fun x => ofF32 (1 / f32 x)
    | .lg2Approx => viaFloat Float.log2
    | .sinApprox => viaFloat Float.sin
    | .cosApprox => viaFloat Float.cos
    | .sqrtApprox => viaFloat Float.sqrt
    | .rsqrtApprox => viaFloat fun v => 1 / v.sqrt
    | _ => id

instance : Inhabited Hw := ⟨Hw.rounded⟩

end AlgorithmLib.Kernel
