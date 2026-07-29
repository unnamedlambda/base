import AlgorithmLib.ML.TapeGrad

/-!
  # Linear gradients at depth: the window obligation, discharged structurally

  `gradProgD_correct` already proves the narrowed reverse sweep equals `sderiv`.
  Its price is a hypothesis:

      hdep : ∀ i < n, ∀ q, q.val < dep i → denote (t.env env) (sderiv (t.getW i) q) = 0

  — *for every binding*.  `sderiv_notUses` reduces each one to a decidable
  syntactic check, and for the 2-binding demo in `GradWarpAlgorithm` that is a
  `decide` per binding and nobody notices.

  A 93-layer model of width 896 has **83,328 bindings**.  83,328 `decide`s is
  not a proof strategy; it is the reason "linear in depth" was still a claim
  about term *size* rather than about elaboration.

  This file removes the per-binding cost.  The observation is that a model's
  telescope is not an arbitrary list of expressions — it is *generated*.  Layer
  `ℓ`'s slot `j` is built by the same function for every `ℓ`, and it reads only
  the previous layer's slots.  So the window property should be proven **once,
  for the generator**, and lifted to all `n` bindings by an induction that does
  not care what `n` is.

  That is `WGen` (a generator carrying its window proof) and
  `genTele_windowed` (the lift).  With them, `layered_gradD_correct` gives the
  full correctness statement for a telescope of *any* depth at a cost
  independent of depth: one window proof, one `ZeroTermFree`, one `ZeroLaws`.

  ## What is and is not bought

  Bought: elaboration cost independent of depth for the *obligation*.  The
  gradient **term** is still `O(n·d)` nodes — narrow, but not free — and that is
  the honest ceiling on this route.

  Not bought: anything about `Float32` exactness.  `ZeroTermFree` and `ZeroLaws`
  are the same two declared propositions as before, named in `Rewrite`, and this
  file neither strengthens nor weakens them.
-/

namespace AlgorithmLib.ML

variable {R : Type} [NumOps R] {Γ : Nat}

-- ---------------------------------------------------------------------------
-- "Reads nothing below `m`"
-- ---------------------------------------------------------------------------

/-- No variable with index below `m` occurs in `e`.  This is the window
    property, in the form a generator can state without knowing the depth it
    will be instantiated at. -/
def NotUsesBelow {Δ : Nat} (m : Nat) (e : Expr Δ) : Prop :=
  ∀ q : Fin Δ, q.val < m → NotUses q e

/-- `liftRen` inherits the "identity below `m`" property.  The new variable is
    sent to `Δ'`, which is at or above `m`, so it never lands in the window. -/
theorem liftRen_below {Δ Δ' m : Nat} (ρ : Fin Δ → Fin Δ') (hm : m ≤ Δ)
    (hle : Δ ≤ Δ') (hρ : ∀ i : Fin Δ, (ρ i).val < m → (ρ i).val = i.val) :
    ∀ i : Fin (Δ + 1), (liftRen ρ i).val < m → (liftRen ρ i).val = i.val := by
  intro i hi
  by_cases hc : i.val < Δ
  · have hv : (liftRen ρ i).val = (ρ ⟨i.val, hc⟩).val := by
      simp only [liftRen, dif_pos hc]
    rw [hv] at hi ⊢
    exact hρ ⟨i.val, hc⟩ hi
  · exfalso
    have hv : (liftRen ρ i).val = Δ' := by simp only [liftRen, dif_neg hc]
    omega

/-- **Renaming preserves the window**, for any renaming that is the identity on
    indices it lands below `m`.

    The hypotheses are exactly what makes the `letE` case go through: `liftRen`
    sends the new variable to `Δ`, which is *not* below `m` because `m ≤ Γ ≤ Δ`,
    so the else-branch is vacuous rather than a counterexample.  `wk`, `castE`
    and `wkTo` are all instances. -/
theorem notUsesBelow_rename : ∀ {Δ : Nat} (e : Expr Δ) {Δ' : Nat} (m : Nat)
    (ρ : Fin Δ → Fin Δ') (hm : m ≤ Δ) (hle : Δ ≤ Δ')
    (hρ : ∀ i : Fin Δ, (ρ i).val < m → (ρ i).val = i.val),
    NotUsesBelow m e → NotUsesBelow m (rename ρ e) := by
  intro Δ e
  induction e with
  | var i =>
      intro Δ' m ρ hm hle hρ h q hq
      show q ≠ ρ i
      intro hc
      have hval : (ρ i).val = i.val := hρ i (hc ▸ hq)
      have hi : i.val < m := by omega
      exact absurd (Fin.ext (rfl : (⟨i.val, i.isLt⟩ : Fin _).val = i.val))
        (h ⟨i.val, i.isLt⟩ hi)
  | lit _ => intro _ _ _ _ _ _ _ _ _; trivial
  | add a b iha ihb =>
      intro Δ' m ρ hm hle hρ h q hq
      exact ⟨iha m ρ hm hle hρ (fun q' hq' => (h q' hq').1) q hq,
             ihb m ρ hm hle hρ (fun q' hq' => (h q' hq').2) q hq⟩
  | mul a b iha ihb =>
      intro Δ' m ρ hm hle hρ h q hq
      exact ⟨iha m ρ hm hle hρ (fun q' hq' => (h q' hq').1) q hq,
             ihb m ρ hm hle hρ (fun q' hq' => (h q' hq').2) q hq⟩
  | neg a ih => intro Δ' m ρ hm hle hρ h q hq; exact ih m ρ hm hle hρ h q hq
  | inv a ih => intro Δ' m ρ hm hle hρ h q hq; exact ih m ρ hm hle hρ h q hq
  | exp a ih => intro Δ' m ρ hm hle hρ h q hq; exact ih m ρ hm hle hρ h q hq
  | rsqrt a ih => intro Δ' m ρ hm hle hρ h q hq; exact ih m ρ hm hle hρ h q hq
  | sum n f ih =>
      intro Δ' m ρ hm hle hρ h q hq j
      exact ih j m ρ hm hle hρ (fun q' hq' => h q' hq' j) q hq
  | letE a b iha ihb =>
      intro Δ' m ρ hm hle hρ h q hq
      refine ⟨iha m ρ hm hle hρ (fun q' hq' => (h q' hq').1) q hq, ?_⟩
      have hb : NotUsesBelow m b := fun q' hq' => (h ⟨q'.val, by omega⟩ hq').2
      exact ihb (Δ' := Δ' + 1) m (liftRen ρ) (by omega) (by omega)
        (liftRen_below ρ hm hle hρ) hb ⟨q.val, by omega⟩ hq

/-- `wk` preserves the window (for a window inside the original context). -/
theorem notUsesBelow_wk {Δ : Nat} (m : Nat) (hm : m ≤ Δ) (e : Expr Δ)
    (h : NotUsesBelow m e) : NotUsesBelow m (wk e) :=
  notUsesBelow_rename e m (fun i => ⟨i.val, Nat.lt_succ_of_lt i.isLt⟩) hm (by omega)
    (fun _ _ => rfl) h

-- ---------------------------------------------------------------------------
-- Windowed telescopes
-- ---------------------------------------------------------------------------

/-- **A telescope whose binding `i` reads nothing below `dep i`.**

    Stated on `getW` rather than `get` because `getW` is what the reverse sweep
    consumes — which sidesteps the cast that relating the two would introduce. -/
structure TeleWindowed {n : Nat} (t : Tele Γ n) (dep : Nat → Nat) : Prop where
  /-- The window lies inside the binding's own scope. -/
  le  : ∀ i : Fin n, dep i.val ≤ Γ + i.val
  /-- …and the binding really does not reach below it. -/
  win : ∀ i : Fin n, NotUsesBelow (dep i.val) (t.getW i)

/-- **The window obligation of `gradProgD_correct`, from a syntactic window.**

    This is the bridge: `TeleWindowed` is a statement about occurrences, `hdep`
    is a statement about denotations, and `sderiv_notUses` is what connects
    them — at the cost of `ZeroLaws`, once, rather than per binding. -/
theorem TeleWindowed.hdep {n : Nat} {t : Tele Γ n} {dep : Nat → Nat}
    (hw : TeleWindowed t dep) (hzl : ZeroLaws R) (env : Fin Γ → R) :
    ∀ (i : Nat) (h : i < n) (q : Fin (Γ + n)), q.val < dep i →
      denote (t.env env) (sderiv (t.getW ⟨i, h⟩) q) = (NumOps.ofNat 0 : R) := by
  intro i h q hq
  exact sderiv_notUses hzl _ q _ (hw.win ⟨i, h⟩ q hq)

/-- **The narrowed gradient is correct for any windowed telescope.**

    The per-binding hypothesis of `gradProgD_correct` is gone; what remains is
    one window proof and the two declared propositions.  Nothing here mentions
    `n`, which is the point: the cost of using it does not grow with depth. -/
theorem windowed_gradD_correct {n : Nat} {t : Tele Γ n} {dep : Nat → Nat}
    (hw : TeleWindowed t dep) (out : Expr (Γ + n)) (env : Fin Γ → R)
    (hz : ZeroTermFree R) (hzl : ZeroLaws R) (k : Fin Γ) :
    denote env ((gradProgD t out dep).get k) = denote env (sderiv (t.bind out) k) :=
  gradProgD_correct t out env dep hz (hw.hdep hzl env) k

-- ---------------------------------------------------------------------------
-- Generated telescopes: the window proven once, for any depth
-- ---------------------------------------------------------------------------

/-- **A telescope generator carrying its window proof.**

    `e i` is slot `i`'s definition, written in the context that exists when it
    is bound.  `win` is the *one* proof a user writes: "slot `i` reads nothing
    below `dep i`".  For a layer stack that is a statement about one layer,
    discharged once — not once per layer, and not once per slot. -/
structure WGen (Γ : Nat) (dep : Nat → Nat) where
  e   : (i : Nat) → Expr (Γ + i)
  le  : ∀ i : Nat, dep i ≤ Γ + i
  win : ∀ i : Nat, NotUsesBelow (dep i) (e i)

/-- The telescope of the generator's first `n` slots. -/
def genTele {dep : Nat → Nat} (g : WGen Γ dep) : (n : Nat) → Tele Γ n
  | 0     => .nil
  | n + 1 => (genTele g n).cons (g.e n)

/-- **The lift.**  Every binding of a generated telescope inherits the
    generator's window — by induction on the depth, so the proof term does not
    grow with it.

    This is the theorem that turns "linear in depth" from a claim about term
    size into a claim about *elaboration*: instantiating it at `n = 83328`
    costs the same as at `n = 2`. -/
theorem genTele_windowed {dep : Nat → Nat} (g : WGen Γ dep) (n : Nat) :
    TeleWindowed (genTele g n) dep := by
  refine ⟨fun i => g.le i.val, ?_⟩
  induction n with
  | zero => intro i; exact absurd i.isLt (by omega)
  | succ n ih =>
      intro i
      by_cases h : i.val < n
      · have hg : (genTele g (n + 1)).getW i = wk ((genTele g n).getW ⟨i.val, h⟩) := by
          show (if h' : i.val < n then wk ((genTele g n).getW ⟨i.val, h'⟩) else wk (g.e n)) = _
          rw [dif_pos h]
        rw [hg]
        exact notUsesBelow_wk _ (Nat.le_trans (g.le i.val) (by omega)) _ (ih ⟨i.val, h⟩)
      · have hin : i.val = n := by have := i.isLt; omega
        have hg : (genTele g (n + 1)).getW i = wk (g.e n) := by
          show (if h' : i.val < n then wk ((genTele g n).getW ⟨i.val, h'⟩) else wk (g.e n)) = _
          rw [dif_neg h]
        rw [hg, hin]
        exact notUsesBelow_wk _ (g.le n) _ (g.win n)

/-- **The headline: a generated model of any depth has a correct narrowed
    gradient, at a proof cost independent of depth.**

    Everything the user supplies is depth-free: the slot generator, its window,
    and the two declared propositions.  `n` appears only as an argument. -/
theorem layered_gradD_correct {dep : Nat → Nat} (g : WGen Γ dep) (n : Nat)
    (out : Expr (Γ + n)) (env : Fin Γ → R) (hz : ZeroTermFree R) (hzl : ZeroLaws R)
    (k : Fin Γ) :
    denote env ((gradProgD (genTele g n) out dep).get k)
      = denote env (sderiv ((genTele g n).bind out) k) :=
  windowed_gradD_correct (genTele_windowed g n) out env hz hzl k

-- ---------------------------------------------------------------------------
-- The front door: a dense stack of any depth, with no proof text
-- ---------------------------------------------------------------------------

/-!
  `WGen` still asks for a window proof.  For the shape that actually occurs —
  a stack of dense layers, each reading the whole previous layer — that proof
  is written **here**, once, and a user gets a `WGen` for any width and depth by
  applying `denseGen`.  This is the same principle as `mapKernel`: the
  obligation is real, so the library discharges it rather than hiding it.
-/

/-- Row-major indexing stays in bounds.  The one arithmetic fact the matrix
    constructors need, proven once because `omega` cannot multiply. -/
theorem rowMajor_lt {m n : Nat} (i : Fin m) (j : Fin n) : i.val * n + j.val < m * n :=
  calc i.val * n + j.val < i.val * n + n := by omega
    _ = (i.val + 1) * n := by rw [Nat.succ_mul]
    _ ≤ m * n := Nat.mul_le_mul_right n i.isLt

/-- Row-major indexing is injective — the fact behind "weight `Wᵢⱼ` is one
    variable, not several".  `omega` cannot do it because it multiplies. -/
theorem rowMajor_inj {n a b c d : Nat} (hb : b < n) (hd : d < n)
    (h : a * n + b = c * n + d) : a = c ∧ b = d := by
  rcases Nat.lt_trichotomy a c with hac | hac | hac
  · exfalso
    have hle : (a + 1) * n ≤ c * n := Nat.mul_le_mul_right n hac
    rw [Nat.succ_mul] at hle
    omega
  · subst hac; exact ⟨rfl, by omega⟩
  · exfalso
    have hle : (c + 1) * n ≤ a * n := Nat.mul_le_mul_right n hac
    rw [Nat.succ_mul] at hle
    omega

/-- Where layer `ℓ`'s inputs start.  Layer `0` reads the model inputs at `0`;
    layer `ℓ` reads layer `ℓ-1`'s slots, which begin at `d + (ℓ-1)·d`. -/
def blockBase (d ℓ : Nat) : Nat := if ℓ = 0 then 0 else d + (ℓ - 1) * d

/-- The dependency window of slot `i`: the base of its own layer's inputs. -/
def denseDep (d i : Nat) : Nat := blockBase d (i / d)

/-- Every slot a dense binding reads is in scope. -/
theorem blockBase_lt (d i j : Nat) (_hd : 0 < d) (hj : j < d) :
    blockBase d (i / d) + j < d + i := by
  unfold blockBase
  by_cases h : i / d = 0
  · rw [if_pos h]; omega
  · rw [if_neg h]
    have h1 : (i / d) * d ≤ i := Nat.div_mul_le_self i d
    have h2 : 1 ≤ i / d := Nat.one_le_iff_ne_zero.mpr h
    have h3 : (i / d - 1) * d + d = (i / d) * d := by
      obtain ⟨mm, hmm⟩ : ∃ mm, i / d = mm + 1 := ⟨i / d - 1, by omega⟩
      rw [hmm, Nat.add_sub_cancel, Nat.succ_mul]
    generalize hA : (i / d - 1) * d = A at h3
    generalize hB : (i / d) * d = B at h1 h3
    omega

theorem denseDep_le (d : Nat) (hd : 0 < d) (i : Nat) : denseDep d i ≤ d + i := by
  have h := blockBase_lt d i 0 hd hd
  show blockBase d (i / d) ≤ d + i
  omega

/-- **One dense layer slot**: a weighted sum of the previous layer, bound, then
    a SiLU.

    The `letE` is not decoration — it is what stops the activation's argument
    being written twice, which is the same sharing argument `bindVec` makes for
    the forward pass.  Weights are literals here; making them variables is a
    change to `Γ`, not to the window argument. -/
def denseSlot (d : Nat) (hd : 0 < d) (c : Nat → Nat → Nat) (i : Nat) :
    Expr (d + i) :=
  .letE
    (.sum d (fun j => .mul (.lit (c i j.val))
      (.var ⟨blockBase d (i / d) + j.val, blockBase_lt d i j.val hd j.isLt⟩)))
    (.mul (.var ⟨d + i, Nat.lt_succ_self _⟩)
      (.inv (.add (.lit 1) (.exp (.neg (.var ⟨d + i, Nat.lt_succ_self _⟩))))))

/-- **The window proof — written once, for every depth and width at once.**

    This is the decl that replaces 83,328 `decide`s. -/
theorem denseSlot_win (d : Nat) (hd : 0 < d) (c : Nat → Nat → Nat) (i : Nat) :
    NotUsesBelow (denseDep d i) (denseSlot d hd c i) := by
  intro q hq
  have hqb : q.val < blockBase d (i / d) := hq
  have hqi : q.val < d + i := Nat.lt_of_lt_of_le hq (denseDep_le d hd i)
  have hlast : (⟨q.val, Nat.lt_succ_of_lt hqi⟩ : Fin (d + i + 1))
      ≠ ⟨d + i, Nat.lt_succ_self _⟩ := by
    intro hc
    have hv : q.val = d + i := congrArg Fin.val hc
    omega
  refine ⟨fun j => ⟨trivial, ?_⟩, ?_⟩
  · intro hc
    have hv : q.val = blockBase d (i / d) + j.val := congrArg Fin.val hc
    omega
  · exact ⟨hlast, trivial, hlast⟩

/-- **A dense stack generator.**  Width `d`, literal weights `c`, any depth —
    and no obligation reaches the caller. -/
def denseGen (d : Nat) (hd : 0 < d) (c : Nat → Nat → Nat) : WGen d (denseDep d) :=
  { e := denseSlot d hd c
    le := denseDep_le d hd
    win := denseSlot_win d hd c }

/-- **The 93-layer statement.**  Instantiate at `n = 93 * d` and the proof term
    is the same size as at `n = 2`: `layered_gradD_correct` is applied once, and
    nothing in it recurses over the depth at elaboration time. -/
theorem denseStack_gradD_correct (d : Nat) (hd : 0 < d) (c : Nat → Nat → Nat)
    (n : Nat) (out : Expr (d + n)) (env : Fin d → R)
    (hz : ZeroTermFree R) (hzl : ZeroLaws R) (k : Fin d) :
    denote env ((gradProgD (genTele (denseGen d hd c) n) out (denseDep d)).get k)
      = denote env (sderiv ((genTele (denseGen d hd c) n).bind out) k) :=
  layered_gradD_correct (denseGen d hd c) n out env hz hzl k

-- ---------------------------------------------------------------------------
-- Measuring the claim
-- ---------------------------------------------------------------------------

/-- Node count of a term — the quantity the depth argument is *about*.
    `sum` is counted at its expanded width, since that is what the compiler
    walks. -/
def Expr.nodes : {Δ : Nat} → Expr Δ → Nat
  | _, .var _    => 1
  | _, .lit _    => 1
  | _, .add a b  => 1 + a.nodes + b.nodes
  | _, .mul a b  => 1 + a.nodes + b.nodes
  | _, .neg a    => 1 + a.nodes
  | _, .inv a    => 1 + a.nodes
  | _, .exp a    => 1 + a.nodes
  | _, .rsqrt a  => 1 + a.nodes
  | _, .sum n f  => (List.finRange n).foldl (fun s j => s + (f j).nodes) 1
  | _, .letE a b => 1 + a.nodes + b.nodes

end AlgorithmLib.ML
