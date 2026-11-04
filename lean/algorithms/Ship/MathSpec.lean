import AlgorithmLib.Proof.LibStep
import AlgorithmLib.Surface.LibMath

/-!
# `sinf`, `cosf` and `powf` answer what `Libm` says

`Surface.LibMath` implements the three math entry points as CLIF, and
`Host.Libm` states each as a function of the argument's bits. Each is proven
once, as `LibSpec.Ans`: run on any argument, the function neither misuses a
call nor faults, answers `Libm.sinf w.mem.nan b`, `Libm.cosf w.mem.nan b` or `Libm.powf w.mem.nan x y`
widened to 64 bits, and leaves the world as it was (`sin_spec`, `cos_spec`,
`powf_spec`).

The sub-programs are proven first, each against the `Libm` function of the
same name, for any environment holding its operands: `wp_horner` by induction
over the coefficients, then `sinPoly`, `cosPoly`, `nanOf`, `bits64`,
`byQuadrant`, `reduce`, `log2Abs` and `exp2`. Their continuations are handed
an environment keeping the one they started from (`Extends`) with the answer in a
slot, so what follows reads its operands through them.

`step` takes one instruction by the rule for it (`Proof.LibSpecFloat`): the
program at the head of the goal is reduced to its first instruction and that
instruction's rules tried; a sub-program goes by its specification
(`step_ext`). `look` finds a slot's value: pushed since, kept by a
sub-program, or as a branch found it. A branch is taken by `wp_ifte32J` or
`wp_ifte8J`; at the end of an arm the value it joins is compared with the
model's under the conditions that led there (`leaf`).

`wp` is irreducible here, so a rule that does not match fails at once instead
of unfolding the semantics.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace MathSpec
open AlgorithmLib.LibSpec

section specs
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

theorem wp_f64c {x : Float} {Q : Slot .f64 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (.sc .f64 (Libm.k x))) w) : wp cfg d J (LibMath.f64c x) Q Γ w :=
  wp_fconst64 h

theorem wp_f32c {b : UInt32} {Q : Slot .f32 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (.sc .f32 (s32 b))) w) : wp cfg d J (LibMath.f32c b) Q Γ w :=
  wp_val (by simp only [Op'.erase, evalOp, widthMask, s32_mask]) h

@[simp] theorem add_c64_m64 (x : UInt64) : x + c64 (-64) = x - 64 := add_c64_neg x 64 (by decide)
@[simp] theorem add_c64_m152 (x : UInt64) : x + c64 (-152) = x - 152 := add_c64_neg x 152 (by decide)
@[simp] theorem add_c64_m1023 (x : UInt64) : x + c64 (-1023) = x - 1023 := add_c64_neg x 1023 (by decide)

theorem wp_horner {cs : List Float} {z : Slot .f64} {zz : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (hz : Γ[z]? = some (.sc .f64 zz))
    (h : ∀ Γ' r, Extends Γ Γ' → Γ'[r]? = some (.sc .f64 (Libm.horner w.mem.nan cs zz)) → Q r Γ' w) :
    wp cfg d J (LibMath.horner cs z) Q Γ w := by
  unfold LibMath.horner
  rw [wp_bind]
  refine wp_f64c ?_
  simp only [wp_bind]
  have hfold : Libm.horner w.mem.nan cs zz =
      cs.reverse.foldl (fun a c => Libm.dadd w.mem.nan (Libm.k c) (Libm.dmul w.mem.nan zz a)) (Libm.k 0.0) := by
    simp [Libm.horner, List.foldl_reverse]
  rw [hfold] at h
  have hz' := get_push (v := .sc .f64 (Libm.k 0.0)) hz
  have hk : Extends Γ (Γ.push (.sc .f64 (Libm.k 0.0))) := Extends.push_of (Extends.refl Γ) _
  have hacc : (Γ.push (.sc .f64 (Libm.k 0.0)))[Γ.size]? = some (.sc .f64 (Libm.k 0.0)) := get_push_size
  generalize Γ.push (.sc .f64 (Libm.k 0.0)) = Γ1 at hz' hk hacc ⊢
  generalize Γ.size = acc at hacc ⊢
  generalize Libm.k 0.0 = a at hacc h ⊢
  generalize cs.reverse = l at h ⊢
  induction l generalizing Γ1 acc a with
  | nil =>
    simp only [List.forIn_nil, wp_pure]
    exact h Γ1 acc hk hacc
  | cons c l ih =>
    simp only [List.forIn_cons, wp_bind]
    refine wp_f64c ?_
    refine wp_fmul64 (get_push hz') (get_push hacc) ?_
    refine wp_fadd64 (get_push get_push_size) get_push_size ?_
    simp only [wp_pure]
    exact ih _ (get_push (get_push (get_push hz'))) (Extends.push_of (Extends.push_of (Extends.push_of hk _) _) _) _ _ get_push_size
      (fun Γ' r he hr => h Γ' r he hr)

attribute [local irreducible] AlgorithmLib.Prog.wp

macro_rules
  | `(tactic| step_ext) => `(tactic| refine wp_f64c ?_)
macro_rules
  | `(tactic| step_ext) => `(tactic| refine wp_f32c ?_)
macro_rules
  | `(tactic| step_ext) => `(tactic| (refine wp_horner (by look) ?_; intro _ _ _ _))

theorem wp_sinPoly {r : Slot .f64} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (hr : Γ[r]? = some (.sc .f64 x))
    (h : ∀ Γ' s, Extends Γ Γ' → Γ'[s]? = some (.sc .f64 (Libm.sinPoly w.mem.nan x)) → Q s Γ' w) :
    wp cfg d J (LibMath.sinPoly r) Q Γ w := by
  unfold LibMath.sinPoly
  repeat step
  exact h _ _ (by extc) get_push_size

theorem wp_cosPoly {r : Slot .f64} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (hr : Γ[r]? = some (.sc .f64 x))
    (h : ∀ Γ' s, Extends Γ Γ' → Γ'[s]? = some (.sc .f64 (Libm.cosPoly w.mem.nan x)) → Q s Γ' w) :
    wp cfg d J (LibMath.cosPoly r) Q Γ w := by
  unfold LibMath.cosPoly
  repeat step
  exact h _ _ (by extc) get_push_size

theorem wp_nanOf {b : Slot .i32} {x : UInt32} {Q : Slot .f32 → Env → World → Prop}
    (hb : Γ[b]? = some (.sc .i32 (s32 x)))
    (h : ∀ Γ' s, Extends Γ Γ' → Γ'[s]? = some (.sc .f32 (s32 (Libm.nanOf x))) → Q s Γ' w) :
    wp cfg d J (LibMath.nanOf b) Q Γ w := by
  unfold LibMath.nanOf
  repeat step
  refine h _ _ (by extc) (get_push_size' ?_)
  simp only [cmp32_eq]
  simp [Libm.nanOf]

theorem wp_bits64 {p0 p1 p2 off : Slot .i64} {a0 a1 a2 o : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (h0 : Γ[p0]? = some (.sc .i64 a0)) (h1 : Γ[p1]? = some (.sc .i64 a1))
    (h2 : Γ[p2]? = some (.sc .i64 a2)) (ho : Γ[off]? = some (.sc .i64 o))
    (h : ∀ Γ' s, Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (Libm.bits64 a0 a1 a2 o)) → Q s Γ' w) :
    wp cfg d J (LibMath.bits64 p0 p1 p2 off) Q Γ w := by
  unfold LibMath.bits64 LibMath.shl LibMath.shr
  repeat step
  refine h _ _ (by extc) (get_push_size' ?_)
  simp [Libm.bits64, cmpInt, c64_eval]


macro_rules
  | `(tactic| step_ext) => `(tactic| (refine wp_bits64 (by look) (by look) (by look) (by look) ?_; intro _ _ _ _))

theorem wp_byQuadrant {k : Slot .i64} {v0 v1 v2 v3 : Slot .f64} {kq x0 x1 x2 x3 : UInt64}
    {Q : Slot .f64 → Env → World → Prop}
    (hk : Γ[k]? = some (.sc .i64 kq)) (h0 : Γ[v0]? = some (.sc .f64 x0)) (h1 : Γ[v1]? = some (.sc .f64 x1))
    (h2 : Γ[v2]? = some (.sc .f64 x2)) (h3 : Γ[v3]? = some (.sc .f64 x3))
    (h : ∀ Γ' s, Extends Γ Γ' → Γ'[s]? = some (.sc .f64 (Libm.byQuadrant kq x0 x1 x2 x3)) → Q s Γ' w) :
    wp cfg d J (LibMath.byQuadrant k v0 v1 v2 v3) Q Γ w := by
  unfold LibMath.byQuadrant
  repeat step
  refine h _ _ (by extc) (get_push_size' ?_)
  simp [Libm.byQuadrant, cmpInt, c64_eval]

set_option maxRecDepth 20000 in
set_option maxHeartbeats 1000000 in
theorem wp_reduce {ax : Slot .i32} {x : UInt32} {Q : Slot .i64 × Slot .f64 → Env → World → Prop}
    (hx : Γ[ax]? = some (.sc .i32 (s32 x)))
    (h : ∀ Γ' kk rr, Extends Γ Γ' → Γ'[kk]? = some (.sc .i64 (Libm.reduce w.mem.nan x).1) →
      Γ'[rr]? = some (.sc .f64 (Libm.reduce w.mem.nan x).2) → Q (kk, rr) Γ' w) :
    wp cfg d J (LibMath.reduce ax) Q Γ w := by
  unfold LibMath.reduce LibMath.shl LibMath.shr
  repeat step
  refine h _ _ _ (by extc) (get_eq (by look) ?_) (get_push_size' ?_)
  all_goals
    simp only [c64_toInt64, add_c64_m64, add_c64_m152]
    simp [Libm.reduce, cmpInt, c64_eval]

macro_rules
  | `(tactic| step_ext) => `(tactic| (refine wp_sinPoly (by look) ?_; intro _ _ _ _))
macro_rules
  | `(tactic| step_ext) => `(tactic| (refine wp_cosPoly (by look) ?_; intro _ _ _ _))
macro_rules
  | `(tactic| step_ext) => `(tactic| (refine wp_nanOf (by look) ?_; intro _ _ _ _))
macro_rules
  | `(tactic| step_ext) => `(tactic| (refine wp_byQuadrant (by look) (by look) (by look) (by look) (by look) ?_; intro _ _ _ _))
macro_rules
  | `(tactic| step_ext) => `(tactic| (refine wp_reduce (by look) ?_; intro _ _ _ _ _ _))

set_option maxRecDepth 20000 in
theorem wp_log2Abs {a : Slot .f64} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x))
    (h : ∀ Γ' s, Extends Γ Γ' → Γ'[s]? = some (.sc .f64 (Libm.log2Abs w.mem.nan x)) → Q s Γ' w) :
    wp cfg d J (LibMath.log2Abs a) Q Γ w := by
  unfold LibMath.log2Abs
  repeat step
  refine h _ _ (by extc) (get_push_size' ?_)
  simp only [add_c64_m1023]
  simp [Libm.log2Abs, c64_eval]

set_option maxRecDepth 20000 in
theorem wp_exp2 {a : Slot .f64} {x : UInt64} {Q : Slot .f64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f64 x))
    (h : ∀ Γ' s, Extends Γ Γ' → Γ'[s]? = some (.sc .f64 (Libm.exp2 w.mem.nan x)) → Q s Γ' w) :
    wp cfg d J (LibMath.exp2 a) Q Γ w := by
  unfold LibMath.exp2
  repeat step
  refine h _ _ (by extc) (get_push_size' ?_)
  simp [Libm.exp2, c64_eval]

macro_rules
  | `(tactic| step_ext) => `(tactic| (refine wp_log2Abs (by look) ?_; intro _ _ _ _))
macro_rules
  | `(tactic| step_ext) => `(tactic| (refine wp_exp2 (by look) ?_; intro _ _ _ _))

/-- A value's equation at a leaf of the branches, from the path to it. -/
macro "leaf" : tactic => `(tactic| (keep_paths; (try simp only [cmp32_eq, cmp32_ne, cmp32_ult, cmp32_ugt, cmp32_uge, cmp8_eq, cmp8_ne] at *); (try simp only [Libm.powf, Libm.sinf, Libm.cosf] at *); (try simp only [u32_lt_max, u32_max_le] at *); (try simp at *); (repeat' split); (all_goals try simp_all); (all_goals repeat' split); all_goals first | rfl | (exfalso; simp only [UInt32.lt_iff_toNat_lt, UInt32.le_iff_toNat_le] at *; omega) | (simp_all; done)))

theorem trig_fine (cos : Bool) : Fine (emitGo (fnBody [.f32] (LibMath.trig cos)) ⟨1, 0, [], [], [], none⟩).2 := by
  cases cos <;> exact ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
set_option maxHeartbeats 2000000 in
theorem sin_spec (cfg : Cfg) (w : World) (b : UInt32) :
    Ans cfg [.sc .f32 (s32 b)] w (LibMath.trig false) [.f32]
      (fun x w' => x = s32 (Libm.sinf w.mem.nan b) ∧ w' = w) := by
  refine ans_of_wp ?_ (trig_fine false) rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [LibMath.trig, entryParams, wp_bind, wp_params, carriesFrom, wp_ret, Bool.false_eq_true, ↓reduceIte]
  have h0 : (#[V.sc .f32 (s32 b)])[0]? = some (.sc .f32 (s32 b)) := rfl
  repeat step
  refine wp_ifte32J (s32 (Libm.sinf w.mem.nan b)) (by look) (by look) ?_ ?_ ?_
  · intro hc1; arm
    leaf
  · intro hc1 ne1 hne1
    repeat step
    refine wp_ifte32J (s32 (Libm.sinf w.mem.nan b)) (by look) (by look) ?_ ?_ ?_
    · intro hc2; arm
      leaf
    · intro hc2 ne2 hne2; arm
      leaf
    · pass
  · joined
    unfold LibMath.answer
    repeat step
    exact ⟨_, get_push_size, rfl, rfl⟩

set_option maxRecDepth 20000 in
set_option maxHeartbeats 2000000 in
theorem cos_spec (cfg : Cfg) (w : World) (b : UInt32) :
    Ans cfg [.sc .f32 (s32 b)] w (LibMath.trig true) [.f32]
      (fun x w' => x = s32 (Libm.cosf w.mem.nan b) ∧ w' = w) := by
  refine ans_of_wp ?_ (trig_fine true) rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [LibMath.trig, entryParams, wp_bind, wp_params, carriesFrom, wp_ret, ↓reduceIte]
  have h0 : (#[V.sc .f32 (s32 b)])[0]? = some (.sc .f32 (s32 b)) := rfl
  repeat step
  refine wp_ifte32J (s32 (Libm.cosf w.mem.nan b)) (by look) (by look) ?_ ?_ ?_
  · intro hc1; arm
    leaf
  · intro hc1 ne1 hne1
    repeat step
    refine wp_ifte32J (s32 (Libm.cosf w.mem.nan b)) (by look) (by look) ?_ ?_ ?_
    · intro hc2; arm
      leaf
    · intro hc2 ne2 hne2; arm
      leaf
    · pass
  · joined
    unfold LibMath.answer
    repeat step
    exact ⟨_, get_push_size, rfl, rfl⟩

theorem powf_fine : Fine (emitGo (fnBody [.f32, .f32] LibMath.powf) ⟨2, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
set_option maxHeartbeats 4000000 in
theorem powf_spec (cfg : Cfg) (w : World) (xb yb : UInt32) :
    Ans cfg [.sc .f32 (s32 xb), .sc .f32 (s32 yb)] w LibMath.powf [.f32, .f32]
      (fun r w' => r = s32 (Libm.powf w.mem.nan xb yb) ∧ w' = w) := by
  refine ans_of_wp ?_ powf_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [LibMath.powf, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .f32 (s32 xb), V.sc .f32 (s32 yb)])[0]? = some (.sc .f32 (s32 xb)) := rfl
  have h1 : (#[V.sc .f32 (s32 xb), V.sc .f32 (s32 yb)])[1]? = some (.sc .f32 (s32 yb)) := rfl
  repeat step
  refine wp_ifte32J (s32 (Libm.powf w.mem.nan xb yb)) (by look) (by look) ?_ ?_ ?_
  · intro hc1; arm
    leaf
  · intro hc1 ne1 hne1
    repeat step
    refine wp_ifte32J (s32 (Libm.powf w.mem.nan xb yb)) (by look) (by look) ?_ ?_ ?_
    · intro hc2; arm
      leaf
    · intro hc2 ne2 hne2
      repeat step
      refine wp_ifte32J (s32 (Libm.powf w.mem.nan xb yb)) (by look) (by look) ?_ ?_ ?_
      · intro hc3; arm
        leaf
      · intro hc3 ne3 hne3
        repeat step
        refine wp_ifte32J (s32 (Libm.powf w.mem.nan xb yb)) (by look) (by look) ?_ ?_ ?_
        · intro hc4
          repeat step
          refine wp_ifte32J (s32 (Libm.powf w.mem.nan xb yb)) (by look) (by look) ?_ ?_ ?_
          · intro hc5; arm
            leaf
          · intro hc5 ne5 hne5; arm
            leaf
          · pass
        · intro hc4 ne4 hne4
          repeat step
          refine wp_ifte32J (s32 (Libm.powf w.mem.nan xb yb)) (by look) (by look) ?_ ?_ ?_
          · intro hc5; arm
            leaf
          · intro hc5 ne5 hne5
            repeat step
            refine wp_ifte32J (s32 (Libm.powf w.mem.nan xb yb)) (by look) (by look) ?_ ?_ ?_
            · intro hc6; arm
              leaf
            · intro hc6 ne6 hne6
              repeat step
              refine wp_ifte8J (s32 (Libm.powf w.mem.nan xb yb)) (by look) (by look) ?_ ?_ ?_
              · intro hc7; arm
                leaf
              · intro hc7 ne7 hne7
                repeat step
                refine wp_ifte8J (if Libm.dcmp .ge (Libm.dmul w.mem.nan (Libm.promote w.mem.nan yb) (Libm.log2Abs w.mem.nan (Libm.promote w.mem.nan xb))) (Libm.k 129.0) then 0x7ff0000000000000 else (if Libm.dcmp .le (Libm.dmul w.mem.nan (Libm.promote w.mem.nan yb) (Libm.log2Abs w.mem.nan (Libm.promote w.mem.nan xb))) (Libm.k (-152.0)) then Libm.k 0.0 else Libm.exp2 w.mem.nan (Libm.dmul w.mem.nan (Libm.promote w.mem.nan yb) (Libm.log2Abs w.mem.nan (Libm.promote w.mem.nan xb))))) (by look) (by look) ?_ ?_ ?_
                · intro hc8
                  repeat step
                  refine wp_ifte8J (if Libm.dcmp .le (Libm.dmul w.mem.nan (Libm.promote w.mem.nan yb) (Libm.log2Abs w.mem.nan (Libm.promote w.mem.nan xb))) (Libm.k (-152.0)) then Libm.k 0.0 else Libm.exp2 w.mem.nan (Libm.dmul w.mem.nan (Libm.promote w.mem.nan yb) (Libm.log2Abs w.mem.nan (Libm.promote w.mem.nan xb)))) (by look) (by look) ?_ ?_ ?_
                  · intro hc9; arm
                    leaf
                  · intro hc9 ne9 hne9; arm
                    leaf
                  · intro _ _ _ _
                    rw [wp_pure]
                    refine ⟨by extc, get_eq (by simp only [Vals.head]; assumption) ?_, rfl⟩
                    keep_paths
                    simp only [cmp8_eq] at *
                    simp_all
                · intro hc8 ne8 hne8; arm
                  keep_paths
                  simp only [cmp8_eq] at *
                  simp_all
                · joined; arm
                  leaf
              · pass
            · pass
          · pass
        · pass
      · pass
    · pass
  · joined
    unfold LibMath.answer
    repeat step
    exact ⟨_, get_push_size, rfl, rfl⟩

end specs

end MathSpec
