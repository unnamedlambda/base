module
public import AlgorithmLib.Proof.LibSpec
public import AlgorithmLib.Host.Libm
public import AlgorithmLib.Host.OpAgree
meta import AlgorithmLib.Proof.LibSpec
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Proof.LibSpecFloat` — the floating-point and 32-bit steps

The weakest precondition of each construct a library function on floats is
written with, beside `Proof.LibSpec`'s 64-bit integer ones. A single or a
32-bit word is held as its bits widened (`s32`), so each 32-bit step is stated
on `UInt32`; a double is held as its bits, and each step on doubles pushes the
`Host.Libm` operation it is (`Libm.dadd`, `Libm.promote`, …), which is
`Sem.evalOp`'s value by definition. A flag is `boolV`. `wp_ifteT` branches on
two slots of any integer type, and `Joins` is what an arm that leaves one
value hands on (`wp_ifte32J`, `wp_ifte8J`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace AlgorithmLib.LibSpec

variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- A single's bits as the semantics holds an `i32` or an `f32`. -/
abbrev s32 (b : UInt32) : UInt64 := b.toUInt64

theorem s32_mask (x : UInt32) : s32 x &&& 0xffffffff = s32 x := by
  apply UInt64.toNat_inj.mp; simp [UInt64.toNat_and]
  exact Nat.and_two_pow_sub_one_of_lt_two_pow (n := 32) x.toNat_lt

theorem nat_and_max64 {n : Nat} (h : n < 2 ^ 64) : n &&& 18446744073709551615 = n :=
  Nat.and_two_pow_sub_one_of_lt_two_pow (n := 64) h

theorem get_eq {Γ : Env} {i : Nat} {v v' : V} (h : Γ[i]? = some v) (e : v = v') : Γ[i]? = some v' := e ▸ h

theorem get_push_size' {Γ : Env} {v v' : V} (h : v = v') : (Γ.push v)[Γ.size]? = some v' := by
  subst h; exact get_push_size

/-! ## Reading the values back -/

theorem s32_inj {a b : UInt32} : s32 a = s32 b ↔ a = b := UInt32.toUInt64_inj
theorem s32_lt {a b : UInt32} : s32 a < s32 b ↔ a < b := by
  simp [UInt64.lt_iff_toNat_lt, UInt32.lt_iff_toNat_lt]
theorem s32_le {a b : UInt32} : s32 a ≤ s32 b ↔ a ≤ b := by
  simp [UInt64.le_iff_toNat_le, UInt32.le_iff_toNat_le]

@[simp] theorem cmp32_eq {a b : UInt32} : cmpInt .eq .i32 (s32 a) (s32 b) = (a == b) := by
  simp only [cmpInt]; rw [Bool.eq_iff_iff]; simp only [beq_iff_eq]; exact UInt32.toUInt64_inj
@[simp] theorem cmp32_ne {a b : UInt32} : cmpInt .ne .i32 (s32 a) (s32 b) = (a != b) := by
  simp only [cmpInt, bne]; congr 1; rw [Bool.eq_iff_iff]; simp only [beq_iff_eq]; exact UInt32.toUInt64_inj
@[simp] theorem cmp32_ult {a b : UInt32} : cmpInt .ult .i32 (s32 a) (s32 b) = decide (a < b) := by
  simp [cmpInt, UInt64.lt_iff_toNat_lt, UInt32.lt_iff_toNat_lt]
@[simp] theorem cmp32_ugt {a b : UInt32} : cmpInt .ugt .i32 (s32 a) (s32 b) = decide (b < a) := by
  simp [cmpInt, UInt64.lt_iff_toNat_lt, UInt32.lt_iff_toNat_lt]
@[simp] theorem cmp32_uge {a b : UInt32} : cmpInt .uge .i32 (s32 a) (s32 b) = decide (b ≤ a) := by
  simp [cmpInt, UInt64.le_iff_toNat_le, UInt32.le_iff_toNat_le]

/-- An `i64` constant's bits, in the form `simp` evaluates. -/
theorem c64_eval (k : Int) : c64 k = UInt64.ofNat (k % 18446744073709551616).toNat := rfl

/-- Adding a negative immediate subtracts it. -/
theorem add_c64_neg (x : UInt64) (n : Nat) (h : n < 2 ^ 63) : x + c64 (-(n : Int)) = x - UInt64.ofNat n := by
  rw [UInt64.sub_eq_add_neg]; congr 1
  apply UInt64.toNat_inj.mp
  rw [c64_eval, UInt64.toNat_ofNat', UInt64.toNat_neg, UInt64.toNat_ofNat']
  simp only [UInt64.size]; omega


@[simp] theorem shl_mod64 (x y : UInt64) : x <<< (y % 64) = x <<< y := by
  apply UInt64.toNat_inj.mp; simp [UInt64.toNat_shiftLeft, UInt64.toNat_mod]

theorem ext_after {Γ' Γ'' : Env} (h' : Extends Γ' Γ'') {Γ : Env} (h : Extends Γ Γ') : Extends Γ Γ'' := Extends.trans h h'

theorem wp_iconst32u {k : Int} {Q : Slot .i32 → Env → World → Prop}
    (hk : 0 ≤ k ∧ k < 2 ^ 32 := by decide)
    (h : Q Γ.size (Γ.push (.sc .i32 (s32 (UInt32.ofNat k.toNat)))) w) : wp cfg d J (iconst32 k) Q Γ w := by
  refine wp_val ?_ h
  obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le hk.1
  have hn : n < 2 ^ 32 := by omega
  have h64 : ((n : Int).emod (((1 <<< 64 : Nat)) : Int)).toNat = n := by
    show ((n : Int) % (18446744073709551616 : Int)).toNat = n
    omega
  simp only [Op'.erase, evalOp, ofInt, norm, widthMask, h64, Int.toNat_natCast]
  congr 2
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, UInt32.toNat_toUInt64, UInt64.toNat_ofNat', UInt32.toNat_ofNat',
    Nat.mod_eq_of_lt (by omega : n < 2 ^ 64), Nat.mod_eq_of_lt hn]
  exact Nat.and_two_pow_sub_one_of_lt_two_pow (n := 32) hn


/-! ## Conversions and constants -/

theorem wp_bitcastI32 {a : Slot .f32} {x : UInt32} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f32 (s32 x))) (h : Q Γ.size (Γ.push (.sc .i32 (s32 x))) w) :
    wp cfg d J (bitcast .i32 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha] <;> op_fin) h

theorem wp_bitcastF32 {a : Slot .i32} {x : UInt32} {Q : Slot .f32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 (s32 x))) (h : Q Γ.size (Γ.push (.sc .f32 (s32 x))) w) :
    wp cfg d J (bitcast .f32 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha] <;> op_fin) h

theorem wp_bitcastI64 {a : Slot .f64} {x : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (h : Q Γ.size (Γ.push (.sc .i64 x)) w) :
    wp cfg d J (bitcast .i64 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha] <;> op_fin) h

theorem wp_bitcastF64 {a : Slot .i64} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (h : Q Γ.size (Γ.push (.sc .f64 x)) w) :
    wp cfg d J (bitcast .f64 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha] <;> op_fin) h

theorem wp_uext32 {a : Slot .i32} {x : UInt32} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 (s32 x))) (h : Q Γ.size (Γ.push (.sc .i64 (s32 x))) w) :
    wp cfg d J (uextend64 a) Q Γ w :=
  wp_uextend64 ha (by rw [s32_mask]; exact h)

theorem wp_fconst64 {bits : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (.sc .f64 bits)) w) : wp cfg d J (fconst .f64 bits) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, widthMask]; exact UInt64.and_neg_one) h

theorem wp_fpromote {a : Slot .f32} {x : UInt32} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f32 (s32 x))) (h : Q Γ.size (Γ.push (.sc .f64 (Libm.promote x))) w) :
    wp cfg d J (fpromote a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, Libm.promote]) h

theorem wp_fdemote {a : Slot .f64} {x : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (h : Q Γ.size (Γ.push (.sc .f32 (s32 (Libm.demote x)))) w) :
    wp cfg d J (fdemote a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, Libm.demote, ofF32] <;> op_fin) h

theorem wp_fcvtFromSint {a : Slot .i64} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (h : Q Γ.size (Γ.push (.sc .f64 (Libm.dofI64 x))) w) :
    wp cfg d J (fcvtFromSint .f64 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, Libm.dofI64] <;> op_fin) h

theorem wp_fcvtToSint {a : Slot .f64} {x : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (h : Q Γ.size (Γ.push (.sc .i64 (Libm.dtoI64 x))) w) :
    wp cfg d J (fcvtToSint .i64 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, Libm.dtoI64, satToSint_sc] <;> op_fin) h

/-! ## Double arithmetic -/

theorem wp_fadd64 {a b : Slot .f64} {x y : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (hb : Γ[b]? = some (.sc .f64 y))
    (h : Q Γ.size (Γ.push (.sc .f64 (Libm.dadd x y))) w) : wp cfg d J (fadd a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, hb, zipF, Libm.dadd]) h

theorem wp_fsub64 {a b : Slot .f64} {x y : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (hb : Γ[b]? = some (.sc .f64 y))
    (h : Q Γ.size (Γ.push (.sc .f64 (Libm.dsub x y))) w) : wp cfg d J (fsub a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, hb, zipF, Libm.dsub]) h

theorem wp_fmul64 {a b : Slot .f64} {x y : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (hb : Γ[b]? = some (.sc .f64 y))
    (h : Q Γ.size (Γ.push (.sc .f64 (Libm.dmul x y))) w) : wp cfg d J (fmul a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, hb, zipF, Libm.dmul]) h

theorem wp_fdiv64 {a b : Slot .f64} {x y : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (hb : Γ[b]? = some (.sc .f64 y))
    (h : Q Γ.size (Γ.push (.sc .f64 (Libm.ddiv x y))) w) : wp cfg d J (fdiv a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, hb, zipF, Libm.ddiv]) h

theorem wp_fneg64 {a : Slot .f64} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (h : Q Γ.size (Γ.push (.sc .f64 (Libm.dneg x))) w) :
    wp cfg d J (fneg a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, Libm.dneg]) h

theorem wp_floor64 {a : Slot .f64} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (h : Q Γ.size (Γ.push (.sc .f64 (Libm.dfloor x))) w) :
    wp cfg d J (floor a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, mapFBits, Libm.dfloor]) h

theorem wp_fcmp64 {c : FloatCC} {a b : Slot .f64} {x y : UInt64} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x)) (hb : Γ[b]? = some (.sc .f64 y))
    (h : Q Γ.size (Γ.push (boolV (Libm.dcmp c x y))) w) : wp cfg d J (fcmp c a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, hb, Libm.dcmp]) h

theorem wp_selectF {c : Slot .i8} {a b : Slot .f64} {p : Bool} {x y : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (hc : Γ[c]? = some (boolV p)) (ha : Γ[a]? = some (.sc .f64 x)) (hb : Γ[b]? = some (.sc .f64 y))
    (h : Q Γ.size (Γ.push (.sc .f64 (if p then x else y))) w) : wp cfg d J (select c a b) Q Γ w :=
  wp_val (by cases p <;> simp [Op'.erase, evalOp, Sem.get, ha, hb, hc, boolV, Sem.isTrue, V.ty] <;> op_fin) h

/-! ## Words -/

theorem wp_band32 {a b : Slot .i32} {x y : UInt32} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 (s32 x))) (hb : Γ[b]? = some (.sc .i32 (s32 y)))
    (h : Q Γ.size (Γ.push (.sc .i32 (s32 (x &&& y)))) w) : wp cfg d J (band a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipIntBits, Sem.get, ha, hb, norm, widthMask, ← UInt32.toUInt64_and, s32_mask] <;> op_fin) h

theorem wp_bor32 {a b : Slot .i32} {x y : UInt32} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 (s32 x))) (hb : Γ[b]? = some (.sc .i32 (s32 y)))
    (h : Q Γ.size (Γ.push (.sc .i32 (s32 (x ||| y)))) w) : wp cfg d J (bor a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipIntBits, Sem.get, ha, hb, norm, widthMask, ← UInt32.toUInt64_or, s32_mask] <;> op_fin) h

theorem wp_select32 {c : Slot .i8} {a b : Slot .i32} {p : Bool} {x y : UInt32} {Q : Slot .i32 → Env → World → Prop}
    (hc : Γ[c]? = some (boolV p)) (ha : Γ[a]? = some (.sc .i32 (s32 x))) (hb : Γ[b]? = some (.sc .i32 (s32 y)))
    (h : Q Γ.size (Γ.push (.sc .i32 (s32 (if p then x else y)))) w) : wp cfg d J (select c a b) Q Γ w :=
  wp_val (by cases p <;> simp [Op'.erase, evalOp, Sem.get, ha, hb, hc, boolV, Sem.isTrue, V.ty] <;> op_fin) h

theorem wp_icmp32 {cc : ICmpCond} {a b : Slot .i32} {x y : UInt32} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 (s32 x))) (hb : Γ[b]? = some (.sc .i32 (s32 y)))
    (h : Q Γ.size (Γ.push (boolV (cmpInt cc .i32 (s32 x) (s32 y)))) w) : wp cfg d J (icmp cc a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipIntCmp, Sem.get, ha, hb] <;> op_fin) h

theorem wp_ushrImm32 {a : Slot .i32} {x : UInt32} {imm : Int} {Q : Slot .i32 → Env → World → Prop}
    (hi : 0 ≤ imm ∧ imm < 32 := by decide) (ha : Γ[a]? = some (.sc .i32 (s32 x)))
    (h : Q (Γ.push (.sc .i64 (c64 imm))).size
      ((Γ.push (.sc .i64 (c64 imm))).push (.sc .i32 (s32 (x >>> UInt32.ofNat imm.toNat)))) w) :
    wp cfg d J (ushrImm a imm) Q Γ w := by
  obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le hi.1
  have hn : n < 32 := by omega
  unfold ushrImm
  rw [wp_bind]
  apply wp_iconst64
  refine wp_val ?_ h
  simp only [Op'.erase, evalOp, shiftBin, Sem.get, get_push ha, get_push_size, norm, widthMask,
    ClifTy.width, ClifTy.isInt, Bool.and_self, if_true, Option.bind_eq_bind, Option.bind_some,
    c64_nat (by omega : n < 2 ^ 64), s32_mask, Option.some.injEq, V.sc.injEq, true_and, Int.toNat_natCast]
  apply UInt64.toNat_inj.mp
  have hx := x.toNat_lt
  have hsh : x.toNat >>> n < 2 ^ 32 := Nat.lt_of_le_of_lt (Nat.shiftRight_le _ _) hx
  simp only [UInt64.toNat_and, UInt64.toNat_shiftRight, UInt32.toNat_toUInt64, UInt64.toNat_mod,
    UInt64.toNat_ofNat', UInt32.toNat_shiftRight, UInt32.toNat_ofNat',
    Nat.mod_eq_of_lt (by omega : n < 2 ^ 64), Nat.mod_eq_of_lt (by omega : n < 2 ^ 32),
    Nat.mod_eq_of_lt (by omega : n < 64), Nat.mod_eq_of_lt hn]
  exact Nat.and_two_pow_sub_one_of_lt_two_pow (n := 32) hsh

theorem c64_toInt64 (u : UInt64) : c64 u.toInt64.toInt = u := by
  rw [c64_eval]
  apply UInt64.toNat_inj.mp
  have := u.toNat_lt
  rw [UInt64.toNat_ofNat']
  have hb : u.toBitVec.toNat = u.toNat := rfl
  simp only [Int64.toInt, UInt64.toBitVec_toInt64, BitVec.toInt_eq_toNat_cond, hb]
  split <;> omega

theorem wp_umulhi {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (Libm.mulhi x y))) w) : wp cfg d J (umulhi a b) Q Γ w := by
  refine wp_val ?_ h
  simp [Op'.erase, evalOp, bin, Sem.get, ha, hb, ibinOp, norm, widthMask, Libm.mulhi, ClifTy.width,
    nat_and_max64 x.toNat_lt, nat_and_max64 y.toNat_lt]
  exact ⟨rfl, UInt64.and_neg_one⟩

theorem wp_ushr {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x >>> y))) w) : wp cfg d J (ushr a b) Q Γ w := by
  refine wp_val ?_ h
  simp [Op'.erase, evalOp, shiftBin, Sem.get, ha, hb, norm, widthMask, ClifTy.width]
  refine ⟨rfl, ?_⟩
  rw [show (18446744073709551615 : UInt64) = -1 from rfl, UInt64.and_neg_one, UInt64.and_neg_one]
  apply UInt64.toNat_inj.mp
  simp [UInt64.toNat_shiftRight, UInt64.toNat_mod]


/-! ## Flags -/

theorem wp_iconst8f {Q : Slot .i8 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (boolV false)) w) : wp cfg d J (iconst .i8 0) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, boolV, ofInt, norm, widthMask]; rfl) h

theorem wp_iconst8t {Q : Slot .i8 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (boolV true)) w) : wp cfg d J (iconst .i8 1) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, boolV, ofInt, norm, widthMask]; rfl) h

theorem wp_bxor8 {a b : Slot .i8} {p q : Bool} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (boolV p)) (hb : Γ[b]? = some (boolV q))
    (h : Q Γ.size (Γ.push (boolV (p ^^ q))) w) : wp cfg d J (bxor a b) Q Γ w :=
  wp_val (by cases p <;> cases q <;> simp [Op'.erase, evalOp, zipIntBits, Sem.get, ha, hb, boolV, norm, widthMask] <;> decide) h

theorem wp_band8 {a b : Slot .i8} {p q : Bool} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (boolV p)) (hb : Γ[b]? = some (boolV q))
    (h : Q Γ.size (Γ.push (boolV (p && q))) w) : wp cfg d J (band a b) Q Γ w :=
  wp_val (by cases p <;> cases q <;> simp [Op'.erase, evalOp, zipIntBits, Sem.get, ha, hb, boolV, norm, widthMask] <;> decide) h

theorem wp_umax32 {a b : Slot .i32} {x y : UInt32} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 (s32 x))) (hb : Γ[b]? = some (.sc .i32 (s32 y)))
    (h : Q Γ.size (Γ.push (.sc .i32 (s32 (max x y)))) w) : wp cfg d J (umax a b) Q Γ w := by
  refine wp_val ?_ h
  show _ = some (V.sc .i32 (s32 (if x ≤ y then y else x)))
  simp only [Op'.erase, evalOp, bin, Sem.get, ha, hb, ibinOp, norm, widthMask, s32_mask,
    Option.bind_eq_bind, Option.bind_some, BEq.rfl, Bool.true_and, ClifTy.isInt, if_true,
    Option.some.injEq, V.sc.injEq, true_and]
  apply UInt64.toNat_inj.mp
  have hx := x.toNat_lt; have hy := y.toNat_lt
  by_cases hxy : x ≤ y
  · rw [if_pos hxy]
    have : x.toNat ≤ y.toNat := UInt32.le_iff_toNat_le.mp hxy
    simp only [UInt64.toNat_and, UInt64.toNat_ofNat', UInt32.toNat_toUInt64, Nat.max_eq_right this,
      Nat.mod_eq_of_lt (by omega : y.toNat < 2 ^ 64)]
    exact Nat.and_two_pow_sub_one_of_lt_two_pow (n := 32) hy
  · rw [if_neg hxy]
    have : y.toNat ≤ x.toNat := by have := UInt32.lt_iff_toNat_lt.mp (UInt32.not_le.mp hxy); omega
    simp only [UInt64.toNat_and, UInt64.toNat_ofNat', UInt32.toNat_toUInt64, Nat.max_eq_left this,
      Nat.mod_eq_of_lt (by omega : x.toNat < 2 ^ 64)]
    exact Nat.and_two_pow_sub_one_of_lt_two_pow (n := 32) hx

theorem wp_icmp8 {cc : ICmpCond} {a b : Slot .i8} {p q : Bool} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (boolV p)) (hb : Γ[b]? = some (boolV q))
    (h : Q Γ.size (Γ.push (boolV (cmpInt cc .i8 (if p then 1 else 0) (if q then 1 else 0)))) w) :
    wp cfg d J (icmp cc a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipIntCmp, Sem.get, ha, hb, boolV] <;> op_fin) h

@[simp] theorem cmp8_eq (p q : Bool) : cmpInt .eq .i8 (if p then 1 else 0) (if q then 1 else 0) = (p == q) := by
  cases p <;> cases q <;> rfl
@[simp] theorem cmp8_ne (p q : Bool) : cmpInt .ne .i8 (if p then 1 else 0) (if q then 1 else 0) = (p != q) := by
  cases p <;> cases q <;> rfl

theorem wp_ifte8 {jTys : List ClifTy} {cc : ICmpCond} {a b : Slot .i8} {p q : Bool}
    {thn els : Prog Slot Lvl (Vals Slot jTys)} {Q : Vals Slot jTys → Env → World → Prop}
    (Qj : Vals Slot jTys → Env → World → Prop)
    (ha : Γ[a]? = some (boolV p)) (hb : Γ[b]? = some (boolV q))
    (ht : cmpInt cc .i8 (if p then 1 else 0) (if q then 1 else 0) = true → wp cfg d J thn Qj (Γ.push (boolV true)) w)
    (he : cmpInt cc .i8 (if p then 1 else 0) (if q then 1 else 0) = false → ∀ ne, (Γ.push (boolV false)).size ≤ ne →
      wp cfg d J els Qj (bindAt (Γ.push (boolV false)) ne []) w)
    (hk : ∀ nJ Γ' jv vs w', Γ'.size ≤ nJ → jv.slots.mapM (fun r => Γ'[r]?) = some vs → Qj jv Γ' w' →
      Q (carriesFrom nJ jTys) (bindAt Γ' nJ vs) w') :
    wp cfg d J (ifte cc a b thn els) Q Γ w :=
  wp_ifteT Qj rfl ha hb ht he hk

theorem u32_max (x y : UInt32) : max x y = if x ≤ y then y else x := rfl

theorem u32_lt_max {a b c : UInt32} : c < max a b ↔ c < a ∨ c < b := by
  rw [u32_max]; split <;> rename_i h <;> rw [UInt32.le_iff_toNat_le] at h <;>
    simp only [UInt32.lt_iff_toNat_lt] <;> omega
theorem u32_max_le {a b c : UInt32} : max a b ≤ c ↔ a ≤ c ∧ b ≤ c := by
  rw [u32_max]; split <;> rename_i h <;> rw [UInt32.le_iff_toNat_le] at h <;>
    simp only [UInt32.le_iff_toNat_le] <;> omega

/-- What a branch whose arms each leave the value `u` of type `jt` and the
    world as it was hands on: an environment keeping `Γ`, with `u` at the join. -/
abbrev Joins (Γ : Env) (w : World) (jt : ClifTy) (u : UInt64) : Vals Slot [jt] → Env → World → Prop :=
  fun jv Γ' w' => Extends Γ Γ' ∧ Γ'[jv.head]? = some (.sc jt u) ∧ w' = w

theorem joins_k {jt : ClifTy} {u : UInt64} {Q : Vals Slot [jt] → Env → World → Prop}
    (hk : ∀ nJ G, Extends Γ G → G[nJ]? = some (.sc jt u) → Q (nJ ::ᵥ .nil) G w) :
    ∀ nJ Γ' jv vs w', Γ'.size ≤ nJ → jv.slots.mapM (fun r => Γ'[r]?) = some vs → Joins Γ w jt u jv Γ' w' →
      Q (carriesFrom nJ [jt]) (bindAt Γ' nJ vs) w' := by
  rintro nJ Γ' jv vs w' hle hvs ⟨hE, hx, rfl⟩
  obtain rfl := join1 hvs hx
  exact hk nJ _ (hE.bindAt_of hle _) (AlgorithmLib.Proof.bindAt_last Γ' nJ _ hle)

theorem wp_ifte32J {jt : ClifTy} {cc : ICmpCond} {a b : Slot .i32} {x y : UInt64} (u : UInt64)
    {thn els : Prog Slot Lvl (Vals Slot [jt])} {Q : Vals Slot [jt] → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x)) (hb : Γ[b]? = some (.sc .i32 y))
    (ht : cmpInt cc .i32 x y = true → wp cfg d J thn (Joins Γ w jt u) (Γ.push (boolV true)) w)
    (he : cmpInt cc .i32 x y = false → ∀ ne, (Γ.push (boolV false)).size ≤ ne →
      wp cfg d J els (Joins Γ w jt u) (bindAt (Γ.push (boolV false)) ne []) w)
    (hk : ∀ nJ G, Extends Γ G → G[nJ]? = some (.sc jt u) → Q (nJ ::ᵥ .nil) G w) :
    wp cfg d J (ifte cc a b thn els) Q Γ w :=
  wp_ifte32 _ ha hb ht he (joins_k hk)

theorem wp_ifte8J {jt : ClifTy} {cc : ICmpCond} {a b : Slot .i8} {p q : Bool} (u : UInt64)
    {thn els : Prog Slot Lvl (Vals Slot [jt])} {Q : Vals Slot [jt] → Env → World → Prop}
    (ha : Γ[a]? = some (boolV p)) (hb : Γ[b]? = some (boolV q))
    (ht : cmpInt cc .i8 (if p then 1 else 0) (if q then 1 else 0) = true →
      wp cfg d J thn (Joins Γ w jt u) (Γ.push (boolV true)) w)
    (he : cmpInt cc .i8 (if p then 1 else 0) (if q then 1 else 0) = false → ∀ ne, (Γ.push (boolV false)).size ≤ ne →
      wp cfg d J els (Joins Γ w jt u) (bindAt (Γ.push (boolV false)) ne []) w)
    (hk : ∀ nJ G, Extends Γ G → G[nJ]? = some (.sc jt u) → Q (nJ ::ᵥ .nil) G w) :
    wp cfg d J (ifte cc a b thn els) Q Γ w :=
  wp_ifte8 _ ha hb ht he (joins_k hk)

end AlgorithmLib.LibSpec
