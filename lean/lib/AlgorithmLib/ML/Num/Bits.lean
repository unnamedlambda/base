module
public import AlgorithmLib.ML.Num.Ops
meta import AlgorithmLib.ML.Num.Ops
public import AlgorithmLib.Kernel.Hw
meta import AlgorithmLib.Kernel.Hw
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Num.Bits` — the carrier a kernel body computes on

`F32` is an f32 value as its word, and `F32.ops hw` its operations exactly as
the host semantics (`Sem.evalOp`) computes them on f32 words on the hardware
`hw`: each arithmetic operation the `Float32` one on the decoded word,
re-encoded, a NaN result the hardware's (`hw.nan`); negation the sign
bit flipped; `inv` and `rsqrt` divisions of one, the square root `funOp`'s;
`le` the comparison `fcmp` makes; `ex2` the hardware's approximation (`hw`),
and `exp` the hardware's `ex2` of `x · log₂ e`.

A model denoted at this carrier means, by definition, what a body computing
it leaves in its words, NaNs included; nothing relates a `Float32` to its bits.
-/

namespace AlgorithmLib.ML

open AlgorithmLib.HProg.Sem AlgorithmLib.Kernel

/-- An f32 value, as its word. -/
structure F32 where
  bits : UInt64

/-- `1.0`'s word. -/
def oneBits : UInt64 := 0x3F800000

/-- `log₂ e`, the constant `exp` multiplies by before `ex2`. -/
def log2eBits : UInt64 := ofF32 1.4426950408889634

/-- A literal at `F32`: its word. -/
instance : NumLit F32 := ⟨fun v => ⟨ofF32 v⟩⟩

/-- **The operations a kernel body computes** on f32 words, the hardware's
    `ex2` being `hw`'s and a NaN result the hardware's (`hw.nan`). -/
def F32.ops (hw : Hw) : NumOps F32 where
  zero := ⟨ofF32 (Float32.ofNat 0)⟩
  one := ⟨oneBits⟩
  add a b := ⟨nanFix hw.nan .f32 (ofF32 (f32 a.bits + f32 b.bits))⟩
  mul a b := ⟨nanFix hw.nan .f32 (ofF32 (f32 a.bits * f32 b.bits))⟩
  neg a := ⟨a.bits ^^^ 0x80000000⟩
  inv a := ⟨nanFix hw.nan .f32 (ofF32 (f32 oneBits / f32 a.bits))⟩
  exp a := ⟨hw.approx .ex2Approx (nanFix hw.nan .f32 (ofF32 (f32 a.bits * f32 log2eBits)))⟩
  ex2 a := ⟨hw.approx .ex2Approx a.bits⟩
  rsqrt a := ⟨nanFix hw.nan .f32
    (ofF32 (f32 oneBits / f32 (nanFix hw.nan .f32 (funOp .sqrt .f32 a.bits) &&& widthMask .f32)))⟩
  le a b := fcmpBits .le .f32 a.bits b.bits
  ofNat n := ⟨ofF32 (Float32.ofNat n)⟩

instance litNatF32 (hw : Hw) : @LitNat F32 (F32.ops hw) _ := @LitNat.mk F32 (F32.ops hw) _ (fun _ => rfl) rfl

end AlgorithmLib.ML
