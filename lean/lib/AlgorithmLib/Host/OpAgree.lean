module
public import AlgorithmLib.Host.World
meta import AlgorithmLib.Host.World
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.OpAgree` — the semantics answers where the table types

`Op.check` is the one table of which operations apply to which operand types
and what they yield: the emitted body is checked by it, and the checker types
operations by it. `evalOp` computes them. For an environment whose values are
shaped as their types say (`EnvShaped`) and types that type its slots as those
values (`Types`):

* `evalOp_typed` — where the table types an operation at `t`, a value the
  evaluator gives has type `t` and is shaped as `t` says;
* `evalOp_defined` — where the table types it, the evaluator answers, unless
  the operation is one that traps (`Op.mayTrap`: a division, or a load, whose
  address may be unmapped);
* `check_of_evalOp` — where the evaluator answers, the table types it and at
  the answer's type, for an operation used as the table asks (`Op.Strict`).
  The evaluator alone asks less in three places: a constant's type, a load's
  address and widening, and a vector's reinterpretation; a body that passes
  `wf` meets none of them;
* `evalOp_answers` — the first two asking only of the operation's operands:
  what the checker uses (`Check.absOp`).
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
def Types (Δ : R → Option ClifTy) (Γ : Env) : Prop := ∀ r, Δ r = (Γ[r]?).map V.ty

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
def Agree (m : Mem) (Δ : R → Option ClifTy) (Γ : Env) (o : Op) : Prop :=
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

variable {m : Mem} {Δ : R → Option ClifTy} {Γ : Env}

/-- An integer operation of two operands of one integer type, total: the
    family `iadd`, `isub` and `imul` are. -/
theorem agree_intBin (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) {o : Op} {a b : R}
    (h : UInt64 → UInt64 → UInt64)
    (hc : o.check Δ = (do let ta ← Δ a; let tb ← Δ b; need (ta == tb && ta.isInt) ta))
    (he : evalOp m Γ o = bin Γ a b fun t x y => some (norm t (h x y)))
    (ht : o.mayTrap = false) (hs : o.Strict Γ = True) : Agree m Δ Γ o := by
  unfold Agree; rw [hc, he, ht, hs]; simp only [Sem.get, bin]; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

/-- A bitwise operation on two integers or two vectors of one type. -/
theorem agree_bitwise (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) {o : Op} {a b : R}
    (h : UInt64 → UInt64 → UInt64)
    (hc : o.check Δ = (do let ta ← Δ a; let tb ← Δ b; need (ta == tb && (ta.isInt || ta.isVec)) ta))
    (he : evalOp m Γ o = (do zipIntBits (← get Γ a) (← get Γ b) (fun _ x y => h x y)))
    (ht : o.mayTrap = false) (hs : o.Strict Γ = True) : Agree m Δ Γ o := by
  unfold Agree; rw [hc, he, ht, hs]; simp only [Sem.get]; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

/-- A float operation on two floats or two float vectors of one type. -/
theorem agree_floatBin (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) {o : Op} {a b : R}
    (f32op : Float32 → Float32 → Float32) (f64op : Float → Float → Float)
    (hc : o.check Δ = (do let ta ← Δ a; let tb ← Δ b; need (ta == tb && (ta.isFloat || ta.isFloatVec)) ta))
    (he : evalOp m Γ o = (do zipF m.fl (← get Γ a) (← get Γ b) f32op f64op))
    (ht : o.mayTrap = false) (hs : o.Strict Γ = True) : Agree m Δ Γ o := by
  unfold Agree; rw [hc, he, ht, hs]; simp only [Sem.get]; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

/-- A float operation on the raw bits of two floats or float vectors of one type. -/
theorem agree_floatBits (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) {o : Op} {a b : R}
    (f : ClifTy → UInt64 → UInt64 → UInt64)
    (hc : o.check Δ = (do let ta ← Δ a; let tb ← Δ b; need (ta == tb && (ta.isFloat || ta.isFloatVec)) ta))
    (he : evalOp m Γ o = (do zipBits (← get Γ a) (← get Γ b) f))
    (ht : o.mayTrap = false) (hs : o.Strict Γ = True) : Agree m Δ Γ o := by
  unfold Agree; rw [hc, he, ht, hs]; simp only [Sem.get]; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

/-- A shift: an integer, by an integer of any width. -/
theorem agree_shift (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) {o : Op} {a b : R}
    (g : ClifTy → UInt64 → UInt64 → UInt64)
    (hc : o.check Δ = (do let ta ← Δ a; let tb ← Δ b; need (ta.isInt && tb.isInt) ta))
    (he : evalOp m Γ o = shiftBin Γ a b fun t x y => some (norm t (g t x y)))
    (ht : o.mayTrap = false) (hs : o.Strict Γ = True) : Agree m Δ Γ o := by
  unfold Agree; rw [hc, he, ht, hs]; simp only [Sem.get, shiftBin]; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

/-- A unary integer operation, at the operand's type. -/
theorem agree_intUn (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) {o : Op} {a : R}
    (g : ClifTy → UInt64 → UInt64)
    (hc : o.check Δ = (do let ta ← Δ a; need ta.isInt ta))
    (he : evalOp m Γ o = un Γ a (·.isInt) fun t x => some (norm t (g t x)))
    (ht : o.mayTrap = false) (hs : o.Strict Γ = True) : Agree m Δ Γ o := by
  unfold Agree; rw [hc, he, ht, hs]; simp only [Sem.get, un]; rw [hΔ a]
  slot a with ha <;> agree_simp

theorem agree_iconst (ty : ClifTy) (k : Int) : Agree m Δ Γ (.iconst ty k) := by
  agree_unfold; cases ty <;> agree_simp

theorem agree_fconst (ty : ClifTy) (b : UInt64) : Agree m Δ Γ (.fconst ty b) := by
  agree_unfold; cases ty <;> agree_simp

theorem agree_memory : Agree m Δ Γ .memory := by
  agree_unfold; agree_simp

theorem agree_funcAddr (i : Nat) : Agree m Δ Γ (.funcAddr i) := by
  agree_unfold; agree_simp

theorem agree_ineg (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a : R) : Agree m Δ Γ (.ineg a) :=
  agree_intUn hΔ hΓ (fun _ x => 0 - x) rfl rfl rfl rfl

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

theorem agree_iadd (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.iadd a b) :=
  agree_intBin hΔ hΓ (· + ·) rfl rfl rfl rfl

theorem agree_isub (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.isub a b) :=
  agree_intBin hΔ hΓ (· - ·) rfl rfl rfl rfl

theorem agree_imul (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.imul a b) :=
  agree_intBin hΔ hΓ (· * ·) rfl rfl rfl rfl

theorem agree_udiv (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.udiv a b) := by
  agree_unfold; rw [hΔ a, hΔ b]
  slot a with ha <;> slot b with hb <;> agree_simp

theorem agree_ishl (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.ishl a b) :=
  agree_shift hΔ hΓ (fun t x y => x <<< (y % UInt64.ofNat t.width)) rfl rfl rfl rfl

theorem agree_ushr (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.ushr a b) :=
  agree_shift hΔ hΓ (fun t x y => (x &&& widthMask t) >>> (y % UInt64.ofNat t.width)) rfl rfl rfl rfl

theorem agree_band (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.band a b) :=
  agree_bitwise hΔ hΓ (· &&& ·) rfl rfl rfl rfl

theorem agree_bandNot (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.bandNot a b) :=
  agree_bitwise hΔ hΓ (fun x y => x &&& ~~~y) rfl rfl rfl rfl

theorem agree_bor (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.bor a b) :=
  agree_bitwise hΔ hΓ (· ||| ·) rfl rfl rfl rfl

theorem agree_bxor (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.bxor a b) :=
  agree_bitwise hΔ hΓ (· ^^^ ·) rfl rfl rfl rfl

theorem agree_fadd (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.fadd a b) :=
  agree_floatBin hΔ hΓ (· + ·) (· + ·) rfl rfl rfl rfl

theorem agree_fsub (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.fsub a b) :=
  agree_floatBin hΔ hΓ (· - ·) (· - ·) rfl rfl rfl rfl

theorem agree_fmul (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.fmul a b) :=
  agree_floatBin hΔ hΓ (· * ·) (· * ·) rfl rfl rfl rfl

theorem agree_fmax (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.fmax a b) :=
  agree_floatBits hΔ hΓ (fun t x y => m.fl t (pickExtreme true t x y)) rfl rfl rfl rfl

theorem agree_fmin (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (a b : R) : Agree m Δ Γ (.fmin a b) :=
  agree_floatBits hΔ hΓ (fun t x y => m.fl t (pickExtreme false t x y)) rfl rfl rfl rfl

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
theorem agree_select (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (c a b : R) :
    Agree m Δ Γ (.select c a b) := by
  unfold Agree
  simp only [Op.check, evalOp, Op.mayTrap, Op.Strict, Sem.get, hΔ c, hΔ a, hΔ b]
  rcases hcv : Γ[c]? with _ | cv <;> rcases hav : Γ[a]? with _ | av <;>
    rcases hbv : Γ[b]? with _ | bv <;> simp [need]
  have sa := hΓ _ _ hav; have sb := hΓ _ _ hbv
  rcases cv with ⟨tc, x⟩ | ⟨tc, xs⟩
  · have hty : (V.sc tc x).ty = tc := rfl
    by_cases hT : isTrue (V.sc tc x) = true <;> simp only [hT, if_true, if_false, Bool.false_eq_true] <;>
      refine ⟨?_, ?_, ?_⟩ <;> intros <;> simp_all <;> grind
  · obtain ⟨l, n, hl, -⟩ := hΓ _ _ hcv
    have hty : (V.vec tc xs).ty = tc := rfl
    cases tc <;> simp_all [ClifTy.lanes, ClifTy.isInt]

set_option maxHeartbeats 8000000 in
/-- What `zipAnyBits` answers: a value of its operands' one type, shaped as the
    first one is. -/
theorem zipAnyBits_some {u v w : V} {f : ClifTy → UInt64 → UInt64 → UInt64}
    (h : zipAnyBits u v f = some w) : u.ty = v.ty ∧ w.ty = u.ty ∧ (u.Shaped → w.Shaped) := by
  unfold zipAnyBits zipBitsIf at h
  rcases u with ⟨t, x⟩ | ⟨t, xs⟩ <;> rcases v with ⟨t', y⟩ | ⟨t', ys⟩ <;> simp at h
  · obtain ⟨rfl, rfl⟩ := h
    exact ⟨rfl, rfl, fun hs => hs⟩
  · rcases hl : t.lanes with _ | ⟨l, n⟩ <;> simp [hl] at h
    obtain ⟨⟨rfl, hsz⟩, rfl⟩ := h
    refine ⟨rfl, rfl, fun ⟨l', n', hl', hn'⟩ => ⟨l', n', hl', ?_⟩⟩
    simp [Array.size_zipWith]; omega

/-- Two shaped operands of one type: `zipAnyBits` answers. -/
theorem zipAnyBits_ok {u v : V} (f : ClifTy → UInt64 → UInt64 → UInt64)
    (hu : u.Shaped) (hv : v.Shaped) (ht : u.ty = v.ty) : ∃ w, zipAnyBits u v f = some w := by
  unfold zipAnyBits zipBitsIf
  rcases u with ⟨t, x⟩ | ⟨t, xs⟩ <;> rcases v with ⟨t', y⟩ | ⟨t', ys⟩ <;>
    simp only [V.ty, V.Shaped] at ht hu hv
  · subst ht; simp
  · subst ht; obtain ⟨l, n, hl, _⟩ := hv; simp [hl] at hu
  · subst ht; obtain ⟨l, n, hl, _⟩ := hu; simp [hl] at hv
  · subst ht
    obtain ⟨l, n, hl, hn⟩ := hu
    obtain ⟨l', n', hl', hn'⟩ := hv
    rw [hl] at hl'; cases hl'
    simp [hl, hn, hn']

theorem agree_bitselect (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (c a b : R) :
    Agree m Δ Γ (.bitselect c a b) := by
  unfold Agree
  simp only [Op.check, evalOp, Op.mayTrap, Op.Strict, Sem.get, hΔ c, hΔ a, hΔ b]
  rcases hcv : Γ[c]? with _ | cv <;> rcases hav : Γ[a]? with _ | av <;>
    rcases hbv : Γ[b]? with _ | bv <;> simp [need]
  have sc := hΓ _ _ hcv; have sa := hΓ _ _ hav; have sb := hΓ _ _ hbv
  refine ⟨?_, ?_, ?_⟩
  · intro e ht v hv
    simp only [Option.bind_eq_some_iff] at hv
    obtain ⟨mk, hm, ot, -, hw⟩ := hv
    obtain ⟨-, e2, s1⟩ := zipAnyBits_some hm
    obtain ⟨-, e4, s3⟩ := zipAnyBits_some hw
    exact ⟨by rw [e4, e2, e], s3 (s1 sc)⟩
  · intro e ht
    obtain ⟨mk, hm⟩ := zipAnyBits_ok (fun _ m x => m &&& x) sc sa e
    obtain ⟨ot, ho⟩ := zipAnyBits_ok (fun _ m y => ~~~m &&& y) sc sb (e.trans ht)
    obtain ⟨-, e2, s1⟩ := zipAnyBits_some hm
    obtain ⟨-, e4, s2⟩ := zipAnyBits_some ho
    obtain ⟨w, hw⟩ := zipAnyBits_ok (fun _ x y => x ||| y) (s1 sc) (s2 sc) (by rw [e2, e4])
    simp [hm, ho, hw]
  · intro v hv
    simp only [Option.bind_eq_some_iff] at hv
    obtain ⟨mk, hm, ot, ho, hw⟩ := hv
    obtain ⟨e1, e2, -⟩ := zipAnyBits_some hm
    obtain ⟨e1', -, -⟩ := zipAnyBits_some ho
    obtain ⟨-, e4, -⟩ := zipAnyBits_some hw
    exact ⟨⟨e1, e1.symm.trans e1'⟩, by rw [e4, e2, e1]⟩

set_option maxHeartbeats 8000000 in
theorem agree_fma (hΔ : Types Δ Γ) (hΓ : EnvShaped Γ) (c a b : R) : Agree m Δ Γ (.fma c a b) := by
  agree_unfold; rw [hΔ c, hΔ a, hΔ b]
  -- the few definitions fma's two readings use, not `agree_simp`'s whole set:
  -- 729 operand shapes make every lemma in the set count
  slot c with hc <;> slot a with ha <;> slot b with hb <;>
    simp [need, V.ty, V.Shaped, ClifTy.isFloat, ClifTy.isFloatVec, ClifTy.lanes, bind, Option.bind, *] <;>
    try decide

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
  split at h
  · cases h
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
  | memory => exact agree_memory
  | funcAddr i => exact agree_funcAddr i
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

/-- An operation other than a load reads its operands and nothing else, and a
    float operation the platform's NaN results too. -/
theorem evalOp_congr (m₁ m₂ : Mem) (Γ₁ Γ₂ : Env) (o : Op) (hl : ∀ op p, o ≠ .load op p)
    (hn : floaty o = false ∨ m₁.nan = m₂.nan)
    (h : ∀ r ∈ o.regs, Γ₁[r]? = Γ₂[r]?) : evalOp m₁ Γ₁ o = evalOp m₂ Γ₂ o := by
  have hfl : floaty o = true → m₁.fl = m₂.fl := fun hf => by
    have hm := hn.resolve_left (by simp [hf])
    funext t x; simp [Mem.fl, hm]
  cases o with
  | load op p => exact absurd rfl (hl op p)
  | iconst | fconst | memory | funcAddr => rfl
  | iadd a b | isub a b | imul a b | udiv a b | ishl a b | ushr a b | band a b | bandNot a b
  | bor a b | bxor a b | icmp _ a b | ibin _ a b | ishift _ a b =>
    have ha := h a (by simp [Op.regs]); have hb := h b (by simp [Op.regs])
    simp only [evalOp, bin, shiftBin, Sem.get, ha, hb]
  | fadd a b | fsub a b | fmul a b | fmax a b | fmin a b | fcmp _ a b | fbin _ a b =>
    have ha := h a (by simp [Op.regs]); have hb := h b (by simp [Op.regs])
    simp only [evalOp, bin, shiftBin, Sem.get, ha, hb, hfl rfl]
  | ineg a | ctz a | popcnt a | ireduce32 a | uextend64 a | sextend64 a
  | vhighBits a | splat _ a | extractlane a _ | bitcast _ a | iun _ a | iext _ _ a =>
    have ha := h a (by simp [Op.regs])
    simp only [evalOp, un, Sem.get, ha]
  | fneg a | fpromote a | fcvtFromSint _ a | fcvtToUint _ a | fun1 _ a | fconv _ _ a =>
    have ha := h a (by simp [Op.regs])
    simp only [evalOp, un, Sem.get, ha, hfl rfl]
  | select c a b | bitselect c a b =>
    have hc := h c (by simp [Op.regs]); have ha := h a (by simp [Op.regs])
    have hb := h b (by simp [Op.regs])
    simp only [evalOp, Sem.get, hc, ha, hb]
  | fma c a b =>
    have hc := h c (by simp [Op.regs]); have ha := h a (by simp [Op.regs])
    have hb := h b (by simp [Op.regs])
    simp only [evalOp, Sem.get, hc, ha, hb, hfl rfl]

/-- `Op.check` reads its operands' types and nothing else. -/
theorem check_congr {g₁ g₂ : R → Option ClifTy} (o : Op) (h : ∀ r ∈ o.regs, g₁ r = g₂ r) :
    o.check g₁ = o.check g₂ := by
  cases o with
  | iconst | fconst | memory | funcAddr => rfl
  | iadd a b | isub a b | imul a b | udiv a b | ishl a b | ushr a b | band a b | bandNot a b
  | bor a b | bxor a b | icmp _ a b | fadd a b | fsub a b | fmul a b | fmax a b | fmin a b
  | fcmp _ a b | ibin _ a b | ishift _ a b | fbin _ a b =>
    have ha := h a (by simp [Op.regs]); have hb := h b (by simp [Op.regs])
    simp only [Op.check, ha, hb]
  | ineg a | ctz a | popcnt a | ireduce32 a | uextend64 a | sextend64 a | fneg a | fpromote a
  | vhighBits a | fcvtFromSint _ a | fcvtToUint _ a | splat _ a | extractlane a _ | bitcast _ a
  | iun _ a | fun1 _ a | fconv _ _ a | iext _ _ a | load _ a =>
    have ha := h a (by simp [Op.regs])
    simp only [Op.check, ha]
  | select c a b | bitselect c a b | fma c a b =>
    have hc := h c (by simp [Op.regs]); have ha := h a (by simp [Op.regs])
    have hb := h b (by simp [Op.regs])
    simp only [Op.check, hc, ha, hb]

/-- Where `Op.check` gives a type, it found one for every operand. -/
theorem check_regs {g : R → Option ClifTy} {o : Op} {t : ClifTy} (h : o.check g = some t) :
    ∀ r ∈ o.regs, (g r).isSome = true := by
  intro r hr
  cases hg : g r with
  | some _ => rfl
  | none =>
      exfalso
      cases o <;> simp only [Op.regs, List.mem_cons, List.not_mem_nil, or_false] at hr <;>
        (try rcases hr with rfl | rfl | rfl) <;> (try subst hr) <;> simp_all [Op.check]

/-- **The semantics answers where the table types**: an operation that cannot
    trap, typed `t` by `Op.check` over types `g` that its operands' values
    have, answers a value of type `t`, shaped as `t` says. Only the operands
    are asked of: the rest of the environment may hold anything. -/
theorem evalOp_answers {m : Mem} {Γ : Env} {g : R → Option ClifTy} {o : Op} {t : ClifTy}
    (hn : o.mayTrap = false) (hc : o.check g = some t)
    (hg : ∀ r ∈ o.regs, ∀ u, g r = some u → ∃ v, Γ[r]? = some v ∧ v.ty = u ∧ v.Shaped) :
    ∃ v, evalOp m Γ o = some v ∧ v.ty = t ∧ v.Shaped := by
  classical
  let Γ' : Env := Γ.mapIdx fun i v => if i ∈ o.regs then v else .sc .i64 0
  have hreg : ∀ r ∈ o.regs, Γ'[r]? = Γ[r]? := by
    intro r hr
    simp only [Γ', Array.getElem?_mapIdx]
    cases Γ[r]? <;> simp [hr]
  have hsh : EnvShaped Γ' := by
    intro r v hv
    by_cases hr : r ∈ o.regs
    · rw [hreg r hr] at hv
      obtain ⟨u, hu⟩ := Option.isSome_iff_exists.mp (check_regs hc r hr)
      obtain ⟨v', hv', -, hs⟩ := hg r hr u hu
      rw [hv] at hv'; cases hv'; exact hs
    · simp only [Γ', Array.getElem?_mapIdx, hr, if_false] at hv
      obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.mp hv
      rfl
  have hc' : o.check (fun r => (Γ'[r]?).map V.ty) = some t := by
    rw [check_congr o (g₂ := g) (fun r hr => ?_)]
    · exact hc
    · obtain ⟨u, hu⟩ := Option.isSome_iff_exists.mp (check_regs hc r hr)
      obtain ⟨v, hv, hvt, -⟩ := hg r hr u hu
      rw [hreg r hr, hv, hu]; simp [hvt]
  have hnl : ∀ op p, o ≠ .load op p := by intro op p e; subst e; simp [Op.mayTrap] at hn
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp (evalOp_defined (m := m) (fun _ => rfl) hsh hc' hn)
  refine ⟨v, ?_, evalOp_typed (fun _ => rfl) hsh hc' hv⟩
  rw [evalOp_congr m m Γ Γ' o hnl (Or.inr rfl) (fun r hr => (hreg r hr).symm)]
  exact hv

end AlgorithmLib.HProg.Sem
