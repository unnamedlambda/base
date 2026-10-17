module
public import AlgorithmLib.Host.World
meta import AlgorithmLib.Host.World
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.OpAgree` — the evaluator answers where the checker accepts

`Op.check` is the one table of which operations apply to which operand types
and what they yield; `evalOp` computes them. For an environment whose values
are shaped as their types say (`EnvShaped`) and a type environment that types
its slots as those values (`Types`):

* `evalOp_typed` — where the checker accepts an operation at type `t`, a value
  the evaluator gives has type `t` and is shaped as `t` says;
* `evalOp_defined` — where the checker accepts, the evaluator answers, unless
  the operation is one that traps (`Op.mayTrap`: a division, or a load, whose
  address may be unmapped);
* `check_of_evalOp` — where the evaluator answers, the checker accepts and at
  the answer's type, for an operation used as the checker asks of it
  (`Op.Strict`). The evaluator alone asks less in three places: a constant's
  type, a load's address and widening, and a vector's reinterpretation; a body
  that passes `wf` meets none of them.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

/-- A value shaped as its type says: a scalar of a scalar type, a vector of a
    vector type with that type's lanes. -/
def V.Shaped : V → Prop
  | .sc t _ => t.lanes = none
  | .vec t xs => ∃ l n, t.lanes = some (l, n) ∧ xs.size = n

/-- Every value in scope is shaped as its type says. -/
def EnvShaped (Γ : Env) : Prop := ∀ (r : Nat) (v : V), Γ[r]? = some v → v.Shaped

/-- `Δ` gives each slot the type of the value `Γ` holds there. -/
def Types (Δ : TyEnv) (Γ : Env) : Prop := ∀ r, Δ.get r = (Γ[r]?).map V.ty

/-- The operations whose answer depends on more than their operands' types: a
    division by zero or out of range traps, and a load reads memory. -/
def _root_.AlgorithmLib.HProg.Op.mayTrap : Op → Bool
  | .udiv _ _ | .ibin .sdiv _ _ | .ibin .urem _ _ | .ibin .srem _ _ | .load _ _ => true
  | _ => false

/-- An operation used as the checker asks, where the evaluator alone would not:
    a constant at a type of its kind, a load through an `i64` of a type its
    kind widens to, a vector reinterpreted at its own type. -/
def _root_.AlgorithmLib.HProg.Op.Strict (Γ : Env) : Op → Prop
  | .iconst ty _ => ty.isInt = true
  | .fconst ty _ => ty.isFloat = true
  | .load op a => (∃ x, Γ[a]? = some (.sc .i64 x)) ∧ op.kind.admits op.ty = true
  | .bitcast ty a => ∀ t xs, Γ[a]? = some (.vec t xs) → t = ty
  | _ => True

/-- The eight shapes a shaped value has. -/
theorem V.Shaped.cases {v : V} (h : v.Shaped) :
    (∃ x, v = .sc .i8 x) ∨ (∃ x, v = .sc .i16 x) ∨ (∃ x, v = .sc .i32 x) ∨ (∃ x, v = .sc .i64 x) ∨
    (∃ x, v = .sc .f32 x) ∨ (∃ x, v = .sc .f64 x) ∨
    (∃ xs, v = .vec .f32x4 xs ∧ xs.size = 4) ∨ (∃ xs, v = .vec .i8x16 xs ∧ xs.size = 16) := by
  cases v with
  | sc t x => cases t <;> simp_all [V.Shaped, ClifTy.lanes]
  | vec t xs =>
      obtain ⟨l, n, hl, hn⟩ := h
      cases t <;> simp_all [ClifTy.lanes]

/-- What the three claims say of one operation. -/
def Agree (m : Mem) (Δ : TyEnv) (Γ : Env) (o : Op) : Prop :=
  (∀ t, o.check Δ = some t → ∀ v, evalOp m Γ o = some v → v.ty = t ∧ v.Shaped) ∧
  (∀ t, o.check Δ = some t → o.mayTrap = false → (evalOp m Γ o).isSome = true) ∧
  (∀ v, evalOp m Γ o = some v → o.Strict Γ → o.check Δ = some v.ty)

-- Split a slot into absent or one of the eight shapes.
set_option hygiene false in
macro "slot " a:ident " with " ha:ident : tactic => `(tactic| (
  rcases $ha:ident : Γ[$a:ident]? with _ | va
  all_goals try (
    rcases V.Shaped.cases (hΓ _ _ $ha:ident) with ⟨x, rfl⟩ | ⟨x, rfl⟩ | ⟨x, rfl⟩ | ⟨x, rfl⟩ | ⟨x, rfl⟩ | ⟨x, rfl⟩ |
      ⟨xs, rfl, hxs⟩ | ⟨xs, rfl, hxs⟩)))

theorem satToSint_ty (ty : ClifTy) (x : Float) : ∃ b, satToSint ty x = .sc ty b := by
  unfold satToSint; simp only [ofInt, norm]; split <;> (try split) <;> (try split) <;> (try split) <;> exact ⟨_, rfl⟩

theorem satToUint_ty (ty : ClifTy) (x : Float) : ∃ b, satToUint ty x = .sc ty b := by
  unfold satToUint; simp only [norm]; split <;> (try split) <;> exact ⟨_, rfl⟩

/-- The bits a saturating conversion answers. -/
noncomputable def satSBits (ty : ClifTy) (x : Float) : UInt64 := Classical.choose (satToSint_ty ty x)
noncomputable def satUBits (ty : ClifTy) (x : Float) : UInt64 := Classical.choose (satToUint_ty ty x)

theorem satToSint_sc (ty : ClifTy) (x : Float) : satToSint ty x = .sc ty (satSBits ty x) :=
  Classical.choose_spec (satToSint_ty ty x)
theorem satToUint_sc (ty : ClifTy) (x : Float) : satToUint ty x = .sc ty (satUBits ty x) :=
  Classical.choose_spec (satToUint_ty ty x)

-- Unfold one operation's two readings, before its operands are split.
macro "agree_unfold" : tactic => `(tactic| (
  unfold Agree
  simp only [Op.check, evalOp, Op.mayTrap, Op.Strict, Sem.get, bin, shiftBin, un]))

-- Close one operand shape.
macro "agree_simp" : tactic => `(tactic| (
  simp [satToSint_sc, satToUint_sc, need, zipIntBits, zipIntCmp, zipBitsIf, zipBits, zipAnyBits, zipF, mapFBits, norm, ofInt, boolV, ishiftOp, iunOp,
    IUn.admits, IExt.admits, FConv.admits,
    V.ty, V.Shaped, ClifTy.isInt, ClifTy.isFloat, ClifTy.isVec, ClifTy.isFloatVec, ClifTy.lanes,
    ClifTy.width, bind, Option.bind, *]
  all_goals try decide))

variable {m : Mem} {Δ : TyEnv} {Γ : Env}

theorem agree_iconst (ty : ClifTy) (k : Int) : Agree m Δ Γ (.iconst ty k) := by
  agree_unfold; cases ty <;> agree_simp

theorem agree_fconst (ty : ClifTy) (b : UInt64) : Agree m Δ Γ (.fconst ty b) := by
  agree_unfold; cases ty <;> agree_simp

theorem agree_ineg (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a : R) : Agree m Δ Γ (.ineg a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> agree_simp

theorem agree_ctz (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a : R) : Agree m Δ Γ (.ctz a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> agree_simp

theorem agree_popcnt (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a : R) : Agree m Δ Γ (.popcnt a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> agree_simp

theorem agree_ireduce32 (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a : R) : Agree m Δ Γ (.ireduce32 a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> agree_simp

theorem agree_uextend64 (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a : R) : Agree m Δ Γ (.uextend64 a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> agree_simp

theorem agree_sextend64 (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a : R) : Agree m Δ Γ (.sextend64 a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> agree_simp

theorem agree_fneg (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a : R) : Agree m Δ Γ (.fneg a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> agree_simp

theorem agree_fpromote (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a : R) : Agree m Δ Γ (.fpromote a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> agree_simp

theorem agree_vhighBits (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a : R) : Agree m Δ Γ (.vhighBits a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> agree_simp

theorem agree_iun (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (k : IUn) (a : R) : Agree m Δ Γ (.iun k a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> cases k <;> agree_simp

theorem agree_fun1 (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (k : FUn) (a : R) : Agree m Δ Γ (.fun1 k a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> cases k <;> agree_simp

theorem agree_fcvtFromSint (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (ty : ClifTy) (a : R) : Agree m Δ Γ (.fcvtFromSint ty a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> cases ty <;> agree_simp

theorem agree_fcvtToUint (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (ty : ClifTy) (a : R) : Agree m Δ Γ (.fcvtToUint ty a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> cases ty <;> agree_simp

theorem agree_splat (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (ty : ClifTy) (a : R) : Agree m Δ Γ (.splat ty a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> cases ty <;> agree_simp

theorem agree_fconv (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (k : FConv) (ty : ClifTy) (a : R) : Agree m Δ Γ (.fconv k ty a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> cases k <;> cases ty <;> agree_simp

theorem agree_iext (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (k : IExt) (ty : ClifTy) (a : R) : Agree m Δ Γ (.iext k ty a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> cases k <;> cases ty <;> agree_simp

theorem agree_extractlane (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a : R) (l : Nat) : Agree m Δ Γ (.extractlane a l) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> agree_simp
  all_goals (
    have hn := Array.getElem?_eq_none_iff (xs := xs) (i := l)
    rcases hl : xs[l]? with _ | y <;> simp_all <;> omega)

theorem agree_iadd (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.iadd a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_isub (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.isub a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_imul (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.imul a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_udiv (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.udiv a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_ishl (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.ishl a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_ushr (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.ushr a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_band (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.band a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_bandNot (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.bandNot a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_bor (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.bor a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_bxor (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.bxor a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_fadd (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.fadd a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_fsub (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.fsub a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_fmul (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.fmul a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_fmax (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.fmax a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_fmin (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.fmin a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_icmp (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (c : ICmpCond) (a b : R) : Agree m Δ Γ (.icmp c a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_fcmp (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (c : FloatCC) (a b : R) : Agree m Δ Γ (.fcmp c a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem bin_some {a b : R} {f : ClifTy → UInt64 → UInt64 → Option V} {v : V} (h : bin Γ a b f = some v) :
    ∃ t x y, Γ[a]? = some (.sc t x) ∧ Γ[b]? = some (.sc t y) ∧ t.isInt = true ∧ f t x y = some v := by
  unfold bin at h
  rcases ha : Γ[a]? with _ | (⟨ta, x⟩ | ⟨ta, xs⟩) <;> simp [Sem.get, ha, bind, Option.bind] at h
  rcases hb : Γ[b]? with _ | (⟨tb, y⟩ | ⟨tb, ys⟩) <;> simp [hb] at h
  obtain ⟨⟨rfl, hi⟩, hf⟩ := h
  exact ⟨ta, x, y, rfl, rfl, hi, hf⟩

theorem ibinOp_sc {k : IBin} {t : ClifTy} {x y : UInt64} {v : V} (h : ibinOp k t x y = some v) :
    ∃ bits, v = .sc t bits := by
  cases k <;> simp only [ibinOp, ofInt, norm] at h <;> (try split at h) <;> (try split at h) <;>
    first | (cases h; exact ⟨_, rfl⟩) | cases h

theorem lanes_of_isInt {t : ClifTy} (h : t.isInt = true) : t.lanes = none := by
  cases t <;> first | rfl | cases h

theorem agree_ibin (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (k : IBin) (a b : R) : Agree m Δ Γ (.ibin k a b) := by
  have key : ∀ v, evalOp m Γ (.ibin k a b) = some v →
      ∃ t bits, v = .sc t bits ∧ t.isInt = true ∧ (Op.ibin k a b).check Δ = some t := by
    intro v hv
    simp only [evalOp] at hv
    obtain ⟨t, x, y, ha, hb, hi, hf⟩ := bin_some hv
    obtain ⟨bits, rfl⟩ := ibinOp_sc hf
    refine ⟨t, bits, rfl, hi, ?_⟩
    simp [Op.check, hΔ a, hΔ b, ha, hb, V.ty, need, hi, bind, Option.bind]
  refine ⟨fun t hc v hv => ?_, fun t hc hn => ?_, fun v hv _ => ?_⟩
  · obtain ⟨t', bits, rfl, hi, hc'⟩ := key v hv
    rw [hc] at hc'; cases hc'
    exact ⟨rfl, lanes_of_isInt hi⟩
  · simp only [Op.check, hΔ a, hΔ b] at hc
    rcases ha : Γ[a]? with _ | va <;> rcases hb : Γ[b]? with _ | vb <;>
      simp [ha, hb, need, bind, Option.bind] at hc
    obtain ⟨⟨hab, hi⟩, rfl⟩ := hc
    have hsa := hΓ _ _ ha; have hsb := hΓ _ _ hb
    cases va with
    | vec ta xs => obtain ⟨l, n, hl, -⟩ := hsa; rw [show (V.vec ta xs).ty = ta from rfl] at hi; rw [lanes_of_isInt hi] at hl; cases hl
    | sc ta x =>
      cases vb with
      | vec tb ys =>
          obtain ⟨l, n, hl, -⟩ := hsb
          rw [show (V.vec tb ys).ty = tb from rfl] at hab; rw [show (V.sc ta x).ty = ta from rfl] at hab hi
          rw [← hab, lanes_of_isInt hi] at hl; cases hl
      | sc tb y =>
          rw [show (V.sc ta x).ty = ta from rfl] at hab hi; rw [show (V.sc tb y).ty = tb from rfl] at hab
          subst hab
          simp only [evalOp, bin, Sem.get, ha, hb, bind, Option.bind, beq_self_eq_true, hi, Bool.and_self, if_true]
          cases k <;> first | rfl | simp [Op.mayTrap] at hn
  · obtain ⟨t', bits, rfl, hi, hc'⟩ := key v hv
    exact hc'

theorem agree_ishift (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (k : IShift) (a b : R) : Agree m Δ Γ (.ishift k a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> cases k <;> agree_simp

theorem agree_fbin (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (k : FBin) (a b : R) : Agree m Δ Γ (.fbin k a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> cases k <;> agree_simp

theorem ite_some_some (c : Prop) [Decidable c] (a b : V) :
    (if c then some a else some b) = some (if c then a else b) := by split <;> rfl

theorem ite_sc (c : Prop) [Decidable c] (t : ClifTy) (x y : UInt64) :
    (if c then V.sc t x else V.sc t y) = V.sc t (if c then x else y) := by split <;> rfl

theorem ite_vec (c : Prop) [Decidable c] (t : ClifTy) (xs ys : Array UInt64) :
    (if c then V.vec t xs else V.vec t ys) = V.vec t (if c then xs else ys) := by split <;> rfl

set_option maxHeartbeats 8000000 in
theorem agree_select (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (c a b : R) : Agree m Δ Γ (.select c a b) := by
  agree_unfold; rw [hΔ c, hΔ a, hΔ b]
  slot c with hc <;> slot a with ha <;> slot b with hb <;>
    simp [ite_some_some, ite_sc, ite_vec, apply_ite Array.size, need, V.ty, V.Shaped, ClifTy.isInt, ClifTy.lanes,
      bind, Option.bind, *]

set_option maxHeartbeats 8000000 in
theorem agree_bitselect (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (c a b : R) : Agree m Δ Γ (.bitselect c a b) := by
  agree_unfold; rw [hΔ c, hΔ a, hΔ b]
  slot c with hc <;> slot a with ha <;> slot b with hb <;> agree_simp

set_option maxHeartbeats 8000000 in
theorem agree_fma (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (c a b : R) : Agree m Δ Γ (.fma c a b) := by
  agree_unfold; rw [hΔ c, hΔ a, hΔ b]
  slot c with hc <;> slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_bitcast (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (ty : ClifTy) (a : R) : Agree m Δ Γ (.bitcast ty a) := by
  agree_unfold; rw [hΔ a]
  slot a with ha <;> cases ty <;> agree_simp

/-- A fold whose every step adds one element ends with as many more as it took
    steps. -/
theorem foldlM_size_succ {α β : Type} (g : Array α → β → Option (Array α))
    (hg : ∀ acc i acc', g acc i = some acc' → acc'.size = acc.size + 1) :
    ∀ (l : List β) (acc ls : Array α), l.foldlM g acc = some ls → ls.size = acc.size + l.length
  | [], acc, ls, h => by cases h; simp
  | i :: l, acc, ls, h => by
      simp only [List.foldlM_cons] at h
      cases hgi : g acc i with
      | none => simp [hgi] at h
      | some acc1 =>
        simp only [hgi, Option.bind_eq_bind, Option.bind_some] at h
        have h1 := foldlM_size_succ g hg l _ _ h
        have h2 := hg _ _ _ hgi
        simp only [List.length_cons]
        omega

/-- What a load of a type its kind yields answers is of that type, and shaped. -/
theorem load_typed {op : LoadOp} {a : R} {v : V} (h : evalOp m Γ (.load op a) = some v)
    (hk : op.kind.admits op.ty = true) : v.ty = op.ty ∧ v.Shaped := by
  have wide : ∀ {n : Nat}, (op.ty.isInt && decide (op.ty.width > n)) = true → op.ty.lanes = none := fun hw => by
    simp only [Bool.and_eq_true] at hw; exact lanes_of_isInt hw.1
  simp only [evalOp, Sem.get] at h
  rcases hg : Γ[a]? with _ | (⟨t, addr⟩ | ⟨t, xs⟩) <;> simp only [hg, bind, Option.bind] at h <;> try cases h
  cases hkind : op.kind <;> simp only [hkind, LoadKind.admits] at h hk
  case plain =>
    cases hl : op.ty.lanes with
    | none =>
        simp only [hl] at h
        split at h <;> cases h
        exact ⟨rfl, hl⟩
    | some p =>
        obtain ⟨lane, n⟩ := p
        simp only [hl] at h
        split at h
        · cases h
        · rename_i ls f hf
          cases h
          refine ⟨rfl, lane, n, hl, ?_⟩
          have := foldlM_size_succ _ (fun acc i acc' hs => by
            split at hs
            · cases hs
            · cases hs; simp) _ _ _ hf
          simpa using this
  all_goals
    split at h <;> cases h
    exact ⟨rfl, wide hk⟩

theorem agree_load (hΔ : Types Δ Γ) (op : LoadOp) (a : R) : Agree m Δ Γ (.load op a) := by
  refine ⟨fun t hc v hv => ?_, fun _ _ h => by simp [Op.mayTrap] at h, fun v hv hs => ?_⟩
  · simp only [Op.check, hΔ a] at hc
    rcases hg : Γ[a]? with _ | va <;> simp [hg, need, bind, Option.bind] at hc
    obtain ⟨⟨-, hk⟩, rfl⟩ := hc
    exact load_typed hv hk
  · obtain ⟨⟨x, hx⟩, hk⟩ := hs
    rw [(load_typed hv hk).1]
    simp [Op.check, hΔ a, hx, need, V.ty, hk, bind, Option.bind]

/-- **The checker and the evaluator agree on every operation.** -/
theorem agree (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (o : Op) : Agree m Δ Γ o := by
  cases o with
  | iconst ty k => exact agree_iconst ty k
  | fconst ty b => exact agree_fconst ty b
  | iadd a b => exact agree_iadd hΔ hΓ a b
  | isub a b => exact agree_isub hΔ hΓ a b
  | imul a b => exact agree_imul hΔ hΓ a b
  | udiv a b => exact agree_udiv hΔ hΓ a b
  | ineg a => exact agree_ineg hΔ hΓ a
  | ishl a b => exact agree_ishl hΔ hΓ a b
  | ushr a b => exact agree_ushr hΔ hΓ a b
  | band a b => exact agree_band hΔ hΓ a b
  | bandNot a b => exact agree_bandNot hΔ hΓ a b
  | bor a b => exact agree_bor hΔ hΓ a b
  | bxor a b => exact agree_bxor hΔ hΓ a b
  | ireduce32 a => exact agree_ireduce32 hΔ hΓ a
  | uextend64 a => exact agree_uextend64 hΔ hΓ a
  | sextend64 a => exact agree_sextend64 hΔ hΓ a
  | icmp c a b => exact agree_icmp hΔ hΓ c a b
  | select c a b => exact agree_select hΔ hΓ c a b
  | bitselect c a b => exact agree_bitselect hΔ hΓ c a b
  | ctz a => exact agree_ctz hΔ hΓ a
  | popcnt a => exact agree_popcnt hΔ hΓ a
  | fadd a b => exact agree_fadd hΔ hΓ a b
  | fsub a b => exact agree_fsub hΔ hΓ a b
  | fmul a b => exact agree_fmul hΔ hΓ a b
  | fmax a b => exact agree_fmax hΔ hΓ a b
  | fmin a b => exact agree_fmin hΔ hΓ a b
  | fneg a => exact agree_fneg hΔ hΓ a
  | fpromote a => exact agree_fpromote hΔ hΓ a
  | fcmp c a b => exact agree_fcmp hΔ hΓ c a b
  | fcvtFromSint ty a => exact agree_fcvtFromSint hΔ hΓ ty a
  | fcvtToUint ty a => exact agree_fcvtToUint hΔ hΓ ty a
  | splat ty a => exact agree_splat hΔ hΓ ty a
  | extractlane a l => exact agree_extractlane hΔ hΓ a l
  | vhighBits a => exact agree_vhighBits hΔ hΓ a
  | bitcast ty a => exact agree_bitcast hΔ hΓ ty a
  | load op a => exact agree_load hΔ op a
  | ibin k a b => exact agree_ibin hΔ hΓ k a b
  | ishift k a b => exact agree_ishift hΔ hΓ k a b
  | iun k a => exact agree_iun hΔ hΓ k a
  | fbin k a b => exact agree_fbin hΔ hΓ k a b
  | fun1 k a => exact agree_fun1 hΔ hΓ k a
  | fconv k ty a => exact agree_fconv hΔ hΓ k ty a
  | fma c a b => exact agree_fma hΔ hΓ c a b
  | iext k ty a => exact agree_iext hΔ hΓ k ty a

/-- **A value the evaluator gives is the type the checker predicted, and shaped
    as that type says.** -/
theorem evalOp_typed (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) {o : Op} {t : ClifTy} {v : V}
    (hc : o.check Δ = some t) (he : evalOp m Γ o = some v) : v.ty = t ∧ v.Shaped :=
  (agree hΔ hΓ o).1 t hc v he

/-- **Where the checker accepts, the evaluator answers**, for every operation
    that cannot trap. -/
theorem evalOp_defined (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) {o : Op} {t : ClifTy}
    (hc : o.check Δ = some t) (hn : o.mayTrap = false) : (evalOp m Γ o).isSome = true :=
  (agree hΔ hΓ o).2.1 t hc hn

/-- **Where the evaluator answers, the checker accepts**, and at the answer's
    type, for an operation used as the checker asks. -/
theorem check_of_evalOp (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) {o : Op} {v : V}
    (he : evalOp m Γ o = some v) (hs : o.Strict Γ) : o.check Δ = some v.ty :=
  (agree hΔ hΓ o).2.2 v he hs

/-- **The evaluator answers exactly where the checker accepts**, for an
    operation that cannot trap, used as the checker asks. -/
theorem evalOp_isSome_iff (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) {o : Op} (hn : o.mayTrap = false)
    (hs : o.Strict Γ) : (evalOp m Γ o).isSome = true ↔ (o.check Δ).isSome = true := by
  constructor
  · intro h
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp h
    rw [check_of_evalOp hΔ hΓ hv hs]; rfl
  · intro h
    obtain ⟨t, ht⟩ := Option.isSome_iff_exists.mp h
    exact evalOp_defined hΔ hΓ ht hn

end AlgorithmLib.HProg.Sem
