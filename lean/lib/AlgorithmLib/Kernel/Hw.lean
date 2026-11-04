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

/-- **The hardware's approximations**: the bits each approximating
    intrinsic (`ex2.approx.f32`, `rcp.approx.f32`, `lg2.approx.f32`) gives on
    each argument's bits. No formula states them; a claim about a kernel that
    uses them holds for every `Hw`, or for those within a stated error.
    `Hw.rounded` is the exactly rounded functions, which the model runs where
    nothing else is said. The hardware's NaN results are `nan`'s. -/
structure Hw where
  approx : Intr → UInt64 → UInt64
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
    | _ => id

instance : Inhabited Hw := ⟨Hw.rounded⟩

end AlgorithmLib.Kernel
