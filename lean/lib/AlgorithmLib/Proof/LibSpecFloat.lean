module
public import AlgorithmLib.Proof.LibSpec
meta import AlgorithmLib.Proof.LibSpec
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Proof.LibSpecFloat` — the floating-point steps

The weakest precondition of each floating-point construct a library function
is written with, beside `Proof.LibSpec`'s integer ones: what it needs of the
slots it reads, and the value it pushes, stated as `Sem.evalOp` computes it —
through `Float` and `Float32` for the arithmetic, on the bits for negation,
the bit casts and the constants.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace AlgorithmLib.LibSpec

section floats
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

theorem wp_fconst64 {b : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (.sc .f64 (b &&& widthMask .f64))) w) : wp cfg d J (fconst .f64 b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp]) h

theorem wp_fconst32 {b : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (.sc .f32 (b &&& widthMask .f32))) w) : wp cfg d J (fconst .f32 b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp]) h

theorem wp_fadd64 {a b : Slot .f64} {x y : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (hb : Γ[b]? = some (.sc .f64 y))
    (h : Q Γ.size (Γ.push (.sc .f64 (ofF64 (f64 x + f64 y)))) w) : wp cfg d J (fadd a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipF, Sem.get, ha, hb]) h

theorem wp_fsub64 {a b : Slot .f64} {x y : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (hb : Γ[b]? = some (.sc .f64 y))
    (h : Q Γ.size (Γ.push (.sc .f64 (ofF64 (f64 x - f64 y)))) w) : wp cfg d J (fsub a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipF, Sem.get, ha, hb]) h

theorem wp_fmul64 {a b : Slot .f64} {x y : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (hb : Γ[b]? = some (.sc .f64 y))
    (h : Q Γ.size (Γ.push (.sc .f64 (ofF64 (f64 x * f64 y)))) w) : wp cfg d J (fmul a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipF, Sem.get, ha, hb]) h

theorem wp_fneg64 {a : Slot .f64} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x))
    (h : Q Γ.size (Γ.push (.sc .f64 (x ^^^ 0x8000000000000000))) w) : wp cfg d J (fneg a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha] <;> op_fin) h

theorem wp_fpromote {a : Slot .f32} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f32 x))
    (h : Q Γ.size (Γ.push (.sc .f64 (ofF64 (f32 x).toFloat))) w) : wp cfg d J (fpromote a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha] <;> op_fin) h

theorem wp_fdemote {a : Slot .f64} {x : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x))
    (h : Q Γ.size (Γ.push (.sc .f32 (ofF32 (f64 x).toFloat32))) w) : wp cfg d J (fdemote a) Q Γ w :=
  wp_val (by simp [fdemote, fconv, Op'.erase, evalOp, Sem.get, ha]) h

theorem wp_floor64 {a : Slot .f64} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x))
    (h : Q Γ.size (Γ.push (.sc .f64 (funOp .floor .f64 x))) w) : wp cfg d J (floor a) Q Γ w :=
  wp_val (by simp [floor, fun1, Op'.erase, evalOp, mapFBits, Sem.get, ha]) h

theorem wp_fcvtFromSint64 {a : Slot .i64} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q Γ.size (Γ.push (.sc .f64 (ofF64 (Float.ofInt (signed .i64 x))))) w) :
    wp cfg d J (fcvtFromSint .f64 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha] <;> op_fin) h

theorem wp_bitcast_f64_i64 {a : Slot .f64} {x : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x))
    (h : Q Γ.size (Γ.push (.sc .i64 x)) w) : wp cfg d J (bitcast .i64 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha] <;> op_fin) h

theorem wp_bitcast_i64_f64 {a : Slot .i64} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q Γ.size (Γ.push (.sc .f64 x)) w) : wp cfg d J (bitcast .f64 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha] <;> op_fin) h

theorem wp_bitcast_f32_i32 {a : Slot .f32} {x : UInt64} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f32 x))
    (h : Q Γ.size (Γ.push (.sc .i32 x)) w) : wp cfg d J (bitcast .i32 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha] <;> op_fin) h

theorem wp_bitcast_i32_f32 {a : Slot .i32} {x : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x))
    (h : Q Γ.size (Γ.push (.sc .f32 x)) w) : wp cfg d J (bitcast .f32 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha] <;> op_fin) h

theorem wp_fcmp64 {c : FloatCC} {a b : Slot .f64} {x y : UInt64} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (hb : Γ[b]? = some (.sc .f64 y))
    (h : Q Γ.size (Γ.push (boolV (cmpF64 c (f64 x) (f64 y)))) w) : wp cfg d J (fcmp c a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, hb]) h

theorem wp_selectF64 {c : Slot .i8} {a b : Slot .f64} {p : Bool} {x y : UInt64}
    {Q : Slot .f64 → Env → World → Prop}
    (hc : Γ[c]? = some (boolV p)) (ha : Γ[a]? = some (.sc .f64 x)) (hb : Γ[b]? = some (.sc .f64 y))
    (h : Q Γ.size (Γ.push (.sc .f64 (if p then x else y))) w) : wp cfg d J (select c a b) Q Γ w :=
  wp_val (by cases p <;> simp [Op'.erase, evalOp, Sem.get, ha, hb, hc, boolV, Sem.isTrue, V.ty] <;> op_fin) h

theorem wp_umulhi64 {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (norm .i64
      (UInt64.ofNat (((x &&& widthMask .i64).toNat * (y &&& widthMask .i64).toNat) >>> 64)))) w) :
    wp cfg d J (umulhi a b) Q Γ w :=
  wp_val (by simp [umulhi, ibin, Op'.erase, evalOp, bin, ibinOp, Sem.get, ha, hb] <;> op_fin) h

end floats

end AlgorithmLib.LibSpec
