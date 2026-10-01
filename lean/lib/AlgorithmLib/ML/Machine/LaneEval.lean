module
public import AlgorithmLib.ML.Machine.Warp
meta import AlgorithmLib.ML.Machine.Warp
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# A lane expression, read as a function of its operands

A row pass's lane code reads two, three or four operand registers. Where it
reads nothing else it *is* a function of those values (`evalPair`,
`evalTriple`, `evalQuad`), and each reading agrees with the warp machine's; the
rewrites a fusion performs (`swap12`, `fuseA`, `fuseA4`) compute the
compositions they claim.
-/

namespace AlgorithmLib.ML

/-- The lane expression reads nothing but the two operand registers. -/
def WFExp.pairOnly : WFExp → Bool
  | .reg r     => r == 1 || r == 2
  | .lit _     => true
  | .add a b   => a.pairOnly && b.pairOnly
  | .mul a b   => a.pairOnly && b.pairOnly
  | .neg a     => a.pairOnly
  | .inv a     => a.pairOnly
  | .exp a     => a.pairOnly
  | .ex2 a     => a.pairOnly
  | .rsqrt a   => a.pairOnly
  | .maxW a b  => a.pairOnly && b.pairOnly
  | .geF a b   => a.pairOnly && b.pairOnly

/-- …and then it *is* a function of the two operands, which is the form the
    stage's value field has to be stated in. -/
def WFExp.evalPair (x y : Float32) : WFExp → Float32
  | .reg r     => if r == 1 then x else if r == 2 then y else NumOps.ofNat 0
  | .lit v     => v
  | .add a b   => NumOps.add (a.evalPair x y) (b.evalPair x y)
  | .mul a b   => NumOps.mul (a.evalPair x y) (b.evalPair x y)
  | .neg a     => NumOps.neg (a.evalPair x y)
  | .inv a     => NumOps.inv (a.evalPair x y)
  | .exp a     => NumOps.exp (a.evalPair x y)
  | .ex2 a     => NumOps.ex2 (a.evalPair x y)
  | .rsqrt a   => NumOps.rsqrt (a.evalPair x y)
  | .maxW a b  => NumOps.max (a.evalPair x y) (b.evalPair x y)
  | .geF a b   => NumOps.ifGe (a.evalPair x y) (b.evalPair x y) 1.0 0.0

/-- **The two readings agree** — one induction, so a caller supplies a boolean
    rather than a proof. -/
theorem WFExp.evalPair_eq : ∀ (f : WFExp), f.pairOnly = true → ∀ (st : WSt) (l : Lane),
    f.eval st l = f.evalPair (st.regs 1 l) (st.regs 2 l) := by
  intro f
  induction f with
  | reg r =>
      intro h st l
      rcases Nat.decEq r 1 with h1 | h1
      · have h2 : r = 2 := by
          simp only [WFExp.pairOnly, Bool.or_eq_true, beq_iff_eq] at h
          rcases h with h | h
          · exact absurd h h1
          · exact h
        subst h2; rfl
      · subst h1; rfl
  | lit v => intro _ _ _; rfl
  | add a b iha ihb =>
      intro h st l
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.add _ _ = NumOps.add _ _
      rw [iha h'.1 st l, ihb h'.2 st l]
  | mul a b iha ihb =>
      intro h st l
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.mul _ _ = NumOps.mul _ _
      rw [iha h'.1 st l, ihb h'.2 st l]
  | neg a iha => intro h st l; show NumOps.neg _ = NumOps.neg _; rw [iha h st l]
  | inv a iha => intro h st l; show NumOps.inv _ = NumOps.inv _; rw [iha h st l]
  | exp a iha => intro h st l; show NumOps.exp _ = NumOps.exp _; rw [iha h st l]
  | ex2 a iha => intro h st l; show NumOps.ex2 _ = NumOps.ex2 _; rw [iha h st l]
  | rsqrt a iha => intro h st l; show NumOps.rsqrt _ = NumOps.rsqrt _; rw [iha h st l]
  | maxW a b iha ihb =>
      intro h st l
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.max _ _ = NumOps.max _ _
      rw [iha h'.1 st l, ihb h'.2 st l]
  | geF a b iha ihb =>
      intro h st l
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.ifGe _ _ _ _ = NumOps.ifGe _ _ _ _
      rw [iha h'.1 st l, ihb h'.2 st l]

/-- The lane expression reads nothing but the three operand registers. -/
def WFExp.tripleOnly : WFExp → Bool
  | .reg r     => r == 1 || r == 2 || r == 3
  | .lit _     => true
  | .add a b   => a.tripleOnly && b.tripleOnly
  | .mul a b   => a.tripleOnly && b.tripleOnly
  | .neg a     => a.tripleOnly
  | .inv a     => a.tripleOnly
  | .exp a     => a.tripleOnly
  | .ex2 a     => a.tripleOnly
  | .rsqrt a   => a.tripleOnly
  | .maxW a b  => a.tripleOnly && b.tripleOnly
  | .geF a b   => a.tripleOnly && b.tripleOnly

/-- …and is therefore a function of those three. -/
def WFExp.evalTriple (x y z : Float32) : WFExp → Float32
  | .reg r     => if r == 1 then x else if r == 2 then y else
                    if r == 3 then z else NumOps.ofNat 0
  | .lit v     => v
  | .add a b   => NumOps.add (a.evalTriple x y z) (b.evalTriple x y z)
  | .mul a b   => NumOps.mul (a.evalTriple x y z) (b.evalTriple x y z)
  | .neg a     => NumOps.neg (a.evalTriple x y z)
  | .inv a     => NumOps.inv (a.evalTriple x y z)
  | .exp a     => NumOps.exp (a.evalTriple x y z)
  | .ex2 a     => NumOps.ex2 (a.evalTriple x y z)
  | .rsqrt a   => NumOps.rsqrt (a.evalTriple x y z)
  | .maxW a b  => NumOps.max (a.evalTriple x y z) (b.evalTriple x y z)
  | .geF a b   => NumOps.ifGe (a.evalTriple x y z) (b.evalTriple x y z) 1.0 0.0

theorem WFExp.evalTriple_eq : ∀ (f : WFExp), f.tripleOnly = true →
    ∀ (st : WSt) (l : Lane),
      f.eval st l = f.evalTriple (st.regs 1 l) (st.regs 2 l) (st.regs 3 l) := by
  intro f
  induction f with
  | reg r =>
      intro h st l
      simp only [WFExp.tripleOnly, Bool.or_eq_true, beq_iff_eq] at h
      rcases h with (h | h) | h
      · subst h; rfl
      · subst h; rfl
      · subst h; rfl
  | lit v => intro _ _ _; rfl
  | add a b iha ihb =>
      intro h st l
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.add _ _ = NumOps.add _ _
      rw [iha h'.1 st l, ihb h'.2 st l]
  | mul a b iha ihb =>
      intro h st l
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.mul _ _ = NumOps.mul _ _
      rw [iha h'.1 st l, ihb h'.2 st l]
  | neg a iha => intro h st l; show NumOps.neg _ = NumOps.neg _; rw [iha h st l]
  | inv a iha => intro h st l; show NumOps.inv _ = NumOps.inv _; rw [iha h st l]
  | exp a iha => intro h st l; show NumOps.exp _ = NumOps.exp _; rw [iha h st l]
  | ex2 a iha => intro h st l; show NumOps.ex2 _ = NumOps.ex2 _; rw [iha h st l]
  | rsqrt a iha => intro h st l; show NumOps.rsqrt _ = NumOps.rsqrt _; rw [iha h st l]
  | maxW a b iha ihb =>
      intro h st l
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.max _ _ = NumOps.max _ _
      rw [iha h'.1 st l, ihb h'.2 st l]
  | geF a b iha ihb =>
      intro h st l
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.ifGe _ _ _ _ = NumOps.ifGe _ _ _ _
      rw [iha h'.1 st l, ihb h'.2 st l]

/-- A two-operand expression read as a three-operand one ignores the third. -/
theorem WFExp.evalTriple_of_pairOnly : ∀ (f : WFExp), f.pairOnly = true →
    ∀ (x y z : Float32), f.evalTriple x y z = f.evalPair x y := by
  intro f
  induction f with
  | reg r =>
      intro h x y z
      simp only [WFExp.pairOnly, Bool.or_eq_true, beq_iff_eq] at h
      rcases h with h | h <;> subst h <;> rfl
  | lit v => intro _ _ _ _; rfl
  | add a b iha ihb =>
      intro h x y z
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.add _ _ = NumOps.add _ _
      rw [iha h'.1 x y z, ihb h'.2 x y z]
  | mul a b iha ihb =>
      intro h x y z
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.mul _ _ = NumOps.mul _ _
      rw [iha h'.1 x y z, ihb h'.2 x y z]
  | neg a iha => intro h x y z; show NumOps.neg _ = NumOps.neg _; rw [iha h x y z]
  | inv a iha => intro h x y z; show NumOps.inv _ = NumOps.inv _; rw [iha h x y z]
  | exp a iha => intro h x y z; show NumOps.exp _ = NumOps.exp _; rw [iha h x y z]
  | ex2 a iha => intro h x y z; show NumOps.ex2 _ = NumOps.ex2 _; rw [iha h x y z]
  | rsqrt a iha => intro h x y z; show NumOps.rsqrt _ = NumOps.rsqrt _; rw [iha h x y z]
  | maxW a b iha ihb =>
      intro h x y z
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.max _ _ = NumOps.max _ _
      rw [iha h'.1 x y z, ihb h'.2 x y z]
  | geF a b iha ihb =>
      intro h x y z
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.ifGe _ _ _ _ = NumOps.ifGe _ _ _ _
      rw [iha h'.1 x y z, ihb h'.2 x y z]

/-- The lane expression reads nothing but the four operand registers. -/
def WFExp.quadOnly : WFExp → Bool
  | .reg r     => r == 1 || r == 2 || r == 3 || r == 4
  | .lit _     => true
  | .add a b   => a.quadOnly && b.quadOnly
  | .mul a b   => a.quadOnly && b.quadOnly
  | .neg a     => a.quadOnly
  | .inv a     => a.quadOnly
  | .exp a     => a.quadOnly
  | .ex2 a     => a.quadOnly
  | .rsqrt a   => a.quadOnly
  | .maxW a b  => a.quadOnly && b.quadOnly
  | .geF a b   => a.quadOnly && b.quadOnly

/-- …and is therefore a function of those four. -/
def WFExp.evalQuad (x y z u : Float32) : WFExp → Float32
  | .reg r     => if r == 1 then x else if r == 2 then y else
                    if r == 3 then z else if r == 4 then u else NumOps.ofNat 0
  | .lit v     => v
  | .add a b   => NumOps.add (a.evalQuad x y z u) (b.evalQuad x y z u)
  | .mul a b   => NumOps.mul (a.evalQuad x y z u) (b.evalQuad x y z u)
  | .neg a     => NumOps.neg (a.evalQuad x y z u)
  | .inv a     => NumOps.inv (a.evalQuad x y z u)
  | .exp a     => NumOps.exp (a.evalQuad x y z u)
  | .ex2 a     => NumOps.ex2 (a.evalQuad x y z u)
  | .rsqrt a   => NumOps.rsqrt (a.evalQuad x y z u)
  | .maxW a b  => NumOps.max (a.evalQuad x y z u) (b.evalQuad x y z u)
  | .geF a b   => NumOps.ifGe (a.evalQuad x y z u) (b.evalQuad x y z u) 1.0 0.0

theorem WFExp.evalQuad_eq : ∀ (f : WFExp), f.quadOnly = true →
    ∀ (st : WSt) (l : Lane),
      f.eval st l
        = f.evalQuad (st.regs 1 l) (st.regs 2 l) (st.regs 3 l) (st.regs 4 l) := by
  intro f
  induction f with
  | reg r =>
      intro h st l
      simp only [WFExp.quadOnly, Bool.or_eq_true, beq_iff_eq] at h
      rcases h with ((h | h) | h) | h <;> subst h <;> rfl
  | lit v => intro _ _ _; rfl
  | add a b iha ihb =>
      intro h st l
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.add _ _ = NumOps.add _ _
      rw [iha h'.1 st l, ihb h'.2 st l]
  | mul a b iha ihb =>
      intro h st l
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.mul _ _ = NumOps.mul _ _
      rw [iha h'.1 st l, ihb h'.2 st l]
  | neg a iha => intro h st l; show NumOps.neg _ = NumOps.neg _; rw [iha h st l]
  | inv a iha => intro h st l; show NumOps.inv _ = NumOps.inv _; rw [iha h st l]
  | exp a iha => intro h st l; show NumOps.exp _ = NumOps.exp _; rw [iha h st l]
  | ex2 a iha => intro h st l; show NumOps.ex2 _ = NumOps.ex2 _; rw [iha h st l]
  | rsqrt a iha => intro h st l; show NumOps.rsqrt _ = NumOps.rsqrt _; rw [iha h st l]
  | maxW a b iha ihb =>
      intro h st l
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.max _ _ = NumOps.max _ _
      rw [iha h'.1 st l, ihb h'.2 st l]
  | geF a b iha ihb =>
      intro h st l
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.ifGe _ _ _ _ = NumOps.ifGe _ _ _ _
      rw [iha h'.1 st l, ihb h'.2 st l]

/-- **Exchange a pair expression's two operand registers.**

    A row pass is symmetric: reading `(a, b)` through `f` is reading `(b, a)`
    through this.  It exists so a fusion that only knows how to consume the
    produced value in the *first* operand slot can still take a consumer that
    holds it in the second — commuting is cheaper than a second fusion lemma,
    and unlike one it introduces no new shape. -/
def WFExp.swap12 : WFExp → WFExp
  | .reg r     => if r == 1 then .reg 2 else if r == 2 then .reg 1 else .reg r
  | .lit v     => .lit v
  | .add a b   => .add a.swap12 b.swap12
  | .mul a b   => .mul a.swap12 b.swap12
  | .neg a     => .neg a.swap12
  | .inv a     => .inv a.swap12
  | .exp a     => .exp a.swap12
  | .ex2 a     => .ex2 a.swap12
  | .rsqrt a   => .rsqrt a.swap12
  | .maxW a b  => .maxW a.swap12 b.swap12
  | .geF a b   => .geF a.swap12 b.swap12

theorem WFExp.swap12_pairOnly : ∀ (e : WFExp), e.pairOnly = true → e.swap12.pairOnly = true := by
  intro e
  induction e with
  | reg r =>
      intro h
      simp only [WFExp.pairOnly, Bool.or_eq_true, beq_iff_eq] at h
      rcases h with h | h <;> subst h <;> rfl
  | lit v => intro _; rfl
  | add a b iha ihb =>
      intro h
      have h' := Bool.and_eq_true .. |>.mp h
      exact Bool.and_eq_true .. |>.mpr ⟨iha h'.1, ihb h'.2⟩
  | mul a b iha ihb =>
      intro h
      have h' := Bool.and_eq_true .. |>.mp h
      exact Bool.and_eq_true .. |>.mpr ⟨iha h'.1, ihb h'.2⟩
  | neg a iha => intro h; exact iha h
  | inv a iha => intro h; exact iha h
  | exp a iha => intro h; exact iha h
  | ex2 a iha => intro h; exact iha h
  | rsqrt a iha => intro h; exact iha h
  | maxW a b iha ihb =>
      intro h
      have h' := Bool.and_eq_true .. |>.mp h
      exact Bool.and_eq_true .. |>.mpr ⟨iha h'.1, ihb h'.2⟩
  | geF a b iha ihb =>
      intro h
      have h' := Bool.and_eq_true .. |>.mp h
      exact Bool.and_eq_true .. |>.mpr ⟨iha h'.1, ihb h'.2⟩

/-- **…and commuting the registers commutes the arguments.**  Nothing is
    reassociated: every operation survives in place, so the commuted pass is
    bit-identical to the original read the other way round. -/
theorem WFExp.swap12_evalPair : ∀ (e : WFExp), e.pairOnly = true →
    ∀ (x y : Float32), e.swap12.evalPair x y = e.evalPair y x := by
  intro e
  induction e with
  | reg r =>
      intro h x y
      simp only [WFExp.pairOnly, Bool.or_eq_true, beq_iff_eq] at h
      rcases h with h | h <;> subst h <;> rfl
  | lit v => intro _ _ _; rfl
  | add a b iha ihb =>
      intro h x y
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.add _ _ = NumOps.add _ _
      rw [iha h'.1 x y, ihb h'.2 x y]
  | mul a b iha ihb =>
      intro h x y
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.mul _ _ = NumOps.mul _ _
      rw [iha h'.1 x y, ihb h'.2 x y]
  | neg a iha => intro h x y; show NumOps.neg _ = NumOps.neg _; rw [iha h x y]
  | inv a iha => intro h x y; show NumOps.inv _ = NumOps.inv _; rw [iha h x y]
  | exp a iha => intro h x y; show NumOps.exp _ = NumOps.exp _; rw [iha h x y]
  | ex2 a iha => intro h x y; show NumOps.ex2 _ = NumOps.ex2 _; rw [iha h x y]
  | rsqrt a iha => intro h x y; show NumOps.rsqrt _ = NumOps.rsqrt _; rw [iha h x y]
  | maxW a b iha ihb =>
      intro h x y
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.max _ _ = NumOps.max _ _
      rw [iha h'.1 x y, ihb h'.2 x y]
  | geF a b iha ihb =>
      intro h x y
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.ifGe _ _ _ _ = NumOps.ifGe _ _ _ _
      rw [iha h'.1 x y, ihb h'.2 x y]

/-- **Fusing one row pass into the next.**  In the consumer, the produced value
    (register 1) becomes the producer's whole expression, and the consumer's
    other operand (register 2) moves to register 3 — which is the slot the
    three-operand pass binds it to.

    No arithmetic is reassociated: every operation of both passes survives in
    the same order, so the fused kernel is bit-identical to the pair. -/
def WFExp.fuseA (outer inner : WFExp) : WFExp :=
  match outer with
  | .reg r     => if r == 1 then inner else if r == 2 then .reg 3 else .reg r
  | .lit v     => .lit v
  | .add a b   => .add (a.fuseA inner) (b.fuseA inner)
  | .mul a b   => .mul (a.fuseA inner) (b.fuseA inner)
  | .neg a     => .neg (a.fuseA inner)
  | .inv a     => .inv (a.fuseA inner)
  | .exp a     => .exp (a.fuseA inner)
  | .ex2 a     => .ex2 (a.fuseA inner)
  | .rsqrt a   => .rsqrt (a.fuseA inner)
  | .maxW a b  => .maxW (a.fuseA inner) (b.fuseA inner)
  | .geF a b   => .geF (a.fuseA inner) (b.fuseA inner)

/-- **…and the fused expression computes the composition.** -/
theorem WFExp.fuseA_eval : ∀ (outer : WFExp), outer.pairOnly = true →
    ∀ (inner : WFExp) (x y z : Float32),
      (outer.fuseA inner).evalTriple x y z
        = outer.evalPair (inner.evalTriple x y z) z := by
  intro outer
  induction outer with
  | reg r =>
      intro h inner x y z
      simp only [WFExp.pairOnly, Bool.or_eq_true, beq_iff_eq] at h
      rcases h with h | h <;> subst h <;> rfl
  | lit v => intro _ _ _ _ _; rfl
  | add a b iha ihb =>
      intro h inner x y z
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.add _ _ = NumOps.add _ _
      rw [iha h'.1 inner x y z, ihb h'.2 inner x y z]
  | mul a b iha ihb =>
      intro h inner x y z
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.mul _ _ = NumOps.mul _ _
      rw [iha h'.1 inner x y z, ihb h'.2 inner x y z]
  | neg a iha => intro h i x y z; show NumOps.neg _ = NumOps.neg _; rw [iha h i x y z]
  | inv a iha => intro h i x y z; show NumOps.inv _ = NumOps.inv _; rw [iha h i x y z]
  | exp a iha => intro h i x y z; show NumOps.exp _ = NumOps.exp _; rw [iha h i x y z]
  | ex2 a iha => intro h i x y z; show NumOps.ex2 _ = NumOps.ex2 _; rw [iha h i x y z]
  | rsqrt a iha => intro h i x y z; show NumOps.rsqrt _ = NumOps.rsqrt _; rw [iha h i x y z]
  | maxW a b iha ihb =>
      intro h inner x y z
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.max _ _ = NumOps.max _ _
      rw [iha h'.1 inner x y z, ihb h'.2 inner x y z]
  | geF a b iha ihb =>
      intro h inner x y z
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.ifGe _ _ _ _ = NumOps.ifGe _ _ _ _
      rw [iha h'.1 inner x y z, ihb h'.2 inner x y z]

/-- A three-operand expression read as a four-operand one ignores the fourth. -/
theorem WFExp.evalQuad_of_tripleOnly : ∀ (f : WFExp), f.tripleOnly = true →
    ∀ (x y z u : Float32), f.evalQuad x y z u = f.evalTriple x y z := by
  intro f
  induction f with
  | reg r =>
      intro h x y z u
      simp only [WFExp.tripleOnly, Bool.or_eq_true, beq_iff_eq] at h
      rcases h with (h | h) | h <;> subst h <;> rfl
  | lit v => intro _ _ _ _ _; rfl
  | add a b iha ihb =>
      intro h x y z u
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.add _ _ = NumOps.add _ _
      rw [iha h'.1 x y z u, ihb h'.2 x y z u]
  | mul a b iha ihb =>
      intro h x y z u
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.mul _ _ = NumOps.mul _ _
      rw [iha h'.1 x y z u, ihb h'.2 x y z u]
  | neg a iha => intro h x y z u; show NumOps.neg _ = NumOps.neg _; rw [iha h x y z u]
  | inv a iha => intro h x y z u; show NumOps.inv _ = NumOps.inv _; rw [iha h x y z u]
  | exp a iha => intro h x y z u; show NumOps.exp _ = NumOps.exp _; rw [iha h x y z u]
  | ex2 a iha => intro h x y z u; show NumOps.ex2 _ = NumOps.ex2 _; rw [iha h x y z u]
  | rsqrt a iha =>
      intro h x y z u; show NumOps.rsqrt _ = NumOps.rsqrt _; rw [iha h x y z u]
  | maxW a b iha ihb =>
      intro h x y z u
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.max _ _ = NumOps.max _ _
      rw [iha h'.1 x y z u, ihb h'.2 x y z u]
  | geF a b iha ihb =>
      intro h x y z u
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.ifGe _ _ _ _ = NumOps.ifGe _ _ _ _
      rw [iha h'.1 x y z u, ihb h'.2 x y z u]

/-- **Fusing a three-operand row pass into the next one.**

    The same substitution as `fuseA` one arity up: the produced value (register
    1) becomes the producer's whole expression, and the consumer's other
    operand moves to register 4, which is where the four-operand pass binds it.
    A chain of three row passes needs exactly this. -/
def WFExp.fuseA4 (outer inner : WFExp) : WFExp :=
  match outer with
  | .reg r     => if r == 1 then inner else if r == 2 then .reg 4 else .reg r
  | .lit v     => .lit v
  | .add a b   => .add (a.fuseA4 inner) (b.fuseA4 inner)
  | .mul a b   => .mul (a.fuseA4 inner) (b.fuseA4 inner)
  | .neg a     => .neg (a.fuseA4 inner)
  | .inv a     => .inv (a.fuseA4 inner)
  | .exp a     => .exp (a.fuseA4 inner)
  | .ex2 a     => .ex2 (a.fuseA4 inner)
  | .rsqrt a   => .rsqrt (a.fuseA4 inner)
  | .maxW a b  => .maxW (a.fuseA4 inner) (b.fuseA4 inner)
  | .geF a b   => .geF (a.fuseA4 inner) (b.fuseA4 inner)

/-- **…and it too computes the composition**, with no arithmetic reassociated. -/
theorem WFExp.fuseA4_eval : ∀ (outer : WFExp), outer.pairOnly = true →
    ∀ (inner : WFExp) (x y z u : Float32),
      (outer.fuseA4 inner).evalQuad x y z u
        = outer.evalPair (inner.evalQuad x y z u) u := by
  intro outer
  induction outer with
  | reg r =>
      intro h inner x y z u
      simp only [WFExp.pairOnly, Bool.or_eq_true, beq_iff_eq] at h
      rcases h with h | h <;> subst h <;> rfl
  | lit v => intro _ _ _ _ _ _; rfl
  | add a b iha ihb =>
      intro h inner x y z u
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.add _ _ = NumOps.add _ _
      rw [iha h'.1 inner x y z u, ihb h'.2 inner x y z u]
  | mul a b iha ihb =>
      intro h inner x y z u
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.mul _ _ = NumOps.mul _ _
      rw [iha h'.1 inner x y z u, ihb h'.2 inner x y z u]
  | neg a iha =>
      intro h inner x y z u; show NumOps.neg _ = NumOps.neg _; rw [iha h inner x y z u]
  | inv a iha =>
      intro h inner x y z u; show NumOps.inv _ = NumOps.inv _; rw [iha h inner x y z u]
  | exp a iha =>
      intro h inner x y z u; show NumOps.exp _ = NumOps.exp _; rw [iha h inner x y z u]
  | ex2 a iha =>
      intro h inner x y z u; show NumOps.ex2 _ = NumOps.ex2 _; rw [iha h inner x y z u]
  | rsqrt a iha =>
      intro h inner x y z u
      show NumOps.rsqrt _ = NumOps.rsqrt _; rw [iha h inner x y z u]
  | maxW a b iha ihb =>
      intro h inner x y z u
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.max _ _ = NumOps.max _ _
      rw [iha h'.1 inner x y z u, ihb h'.2 inner x y z u]
  | geF a b iha ihb =>
      intro h inner x y z u
      have h' := Bool.and_eq_true .. |>.mp h
      show NumOps.ifGe _ _ _ _ = NumOps.ifGe _ _ _ _
      rw [iha h'.1 inner x y z u, ihb h'.2 inner x y z u]

end AlgorithmLib.ML
