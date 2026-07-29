import AlgorithmLib.ML.Layered
import AlgorithmLib.ML.Transformer

namespace AlgorithmLib.ML

open NumOps

-- ---------------------------------------------------------------------------
-- Sums with a single surviving term
-- ---------------------------------------------------------------------------

section Sums

variable {R : Type} [NumLaws R]

/-- A fold of zeros leaves the accumulator alone. -/
theorem foldl_add_zeros {n : Nat} (v : Fin n → R) :
    ∀ (L : List (Fin n)), (∀ j ∈ L, v j = zero) → ∀ (a : R),
      L.foldl (fun acc j => add acc (v j)) a = a := by
  intro L
  induction L with
  | nil => intro _ a; rfl
  | cons x L ih =>
      intro h a
      rw [List.foldl_cons, h x (List.mem_cons_self), NumLaws.add_zero,
          ih (fun j hj => h j (List.mem_cons_of_mem x hj)) a]

/-- **A sum in which one term survives.**  The delta that appears whenever
    `sderiv` meets a variable: `∂xₖ/∂xⱼ` is `1` at `k = j` and `0` elsewhere, so
    a reduction over `k` collapses to its `j`-th term.

    Proved by induction on the width via `List.finRange_succ`, which keeps it
    independent of what nodup lemmas happen to be available. -/
theorem foldl_finRange_single : ∀ (n : Nat) (v : Fin n → R) (j0 : Fin n),
    (∀ j, j ≠ j0 → v j = zero) → ∀ (a : R),
      (List.finRange n).foldl (fun acc j => add acc (v j)) a = add a (v j0) := by
  intro n
  induction n with
  | zero => intro _ j0; exact absurd j0.isLt (by omega)
  | succ n ih =>
      intro v j0 h a
      rw [List.finRange_succ, List.foldl_cons, List.foldl_map]
      induction j0 using Fin.cases with
      | zero =>
          -- the surviving term is index 0; every `succ` term is zero
          refine foldl_add_zeros (fun j => v j.succ) (List.finRange n) (fun j _ => ?_) _
          exact h j.succ (fun hc => absurd hc (Fin.succ_ne_zero j))
      | succ k =>
          -- the surviving term is `k.succ`, so index 0 contributes nothing
          have h0 : v 0 = zero := h 0 (fun hc => absurd hc.symm (Fin.succ_ne_zero k))
          rw [h0, NumLaws.add_zero]
          exact ih (fun j => v j.succ) k
            (fun j hj => h j.succ (fun hc => hj (by
              have := congrArg Fin.val hc
              simp [Fin.succ] at this
              exact Fin.ext this))) a

/-- The same, as a statement about `Expr.sum`. -/
theorem denote_sum_single {Γ n : Nat} (env : Fin Γ → R) (g : Fin n → Expr Γ)
    (j0 : Fin n) (h : ∀ j, j ≠ j0 → denote env (g j) = zero) :
    denote env (.sum n g) = denote env (g j0) := by
  show (List.finRange n).foldl (fun acc j => add acc (denote env (g j))) zero = _
  rw [foldl_finRange_single n (fun j => denote env (g j)) j0 h zero, NumLaws.zero_add]

end Sums

-- ---------------------------------------------------------------------------
-- `NumLaws` already gives the zero facts
-- ---------------------------------------------------------------------------

/-- **`ZeroLaws` is free at `NumLaws`.**  Worth recording: the two propositions
    exist separately because `Float32` satisfies neither, but in any carrier
    with the ordinary algebra the second follows from the first, so a proof at
    this tier never has to assume both. -/
theorem zeroLaws_of_numLaws (R : Type) [NumLaws R] : ZeroLaws R where
  add_zz := by rw [NumLaws.ofNat_zero, NumLaws.add_zero]
  mul_lz := fun x => by rw [NumLaws.ofNat_zero, NumLaws.zero_mul]
  mul_rz := fun x => by rw [NumLaws.ofNat_zero, NumLaws.mul_zero]
  neg_z  := by
    rw [NumLaws.ofNat_zero]
    have h : add (zero : R) (neg zero) = zero := NumLaws.add_neg zero
    rw [NumLaws.zero_add] at h
    exact h
  sum_z  := fun n => by
    show (List.finRange n).foldl (fun acc (_ : Fin n) => add acc (ofNat 0 : R)) zero = _
    rw [foldl_add_zeros (fun _ => (ofNat 0 : R)) (List.finRange n)
          (fun _ _ => NumLaws.ofNat_zero) zero, NumLaws.ofNat_zero]

-- ---------------------------------------------------------------------------
-- The layer
-- ---------------------------------------------------------------------------

section Layer

variable {R : Type} [NumLaws R] {Γ n : Nat}

/-- `zᵢ = Σⱼ Wᵢⱼ·xⱼ`, with the inputs read as variables so `sderiv` has
    something to differentiate against. -/
def preAct (W : Fin n → Fin n → Expr Γ) (xv : Fin n → Fin Γ) (i : Fin n) : Expr Γ :=
  .sum n (fun j => .mul (W i j) (.var (xv j)))

/-- `yᵢ = act(zᵢ)`, with `zᵢ` **bound**.  The binding is what makes `sderiv`
    apply the chain rule structurally. -/
def layerOut (act : Expr (Γ + 1)) (W : Fin n → Fin n → Expr Γ) (xv : Fin n → Fin Γ)
    (i : Fin n) : Expr Γ := .letE (preAct W xv i) act

/-- `L = Σᵢ dyᵢ·yᵢ` — the layer contracted with its upstream adjoint. -/
def layerLoss (dy : Fin n → Expr Γ) (act : Expr (Γ + 1))
    (W : Fin n → Fin n → Expr Γ) (xv : Fin n → Fin Γ) : Expr Γ :=
  .sum n (fun i => .mul (dy i) (layerOut act W xv i))

/-- **The activation's derivative**, taken symbolically against the bound value.
    This is exactly the expression `siluBwdSpec` puts in a kernel. -/
def actDeriv (act : Expr (Γ + 1)) : Expr (Γ + 1) := sderiv act ⟨Γ, Nat.lt_succ_self Γ⟩

/-- **The layer's adjoint**, as the activation-backward kernel computes it:
    `dsᵢ = dyᵢ · act'(zᵢ)`. -/
def layerAdj (dy : Fin n → Expr Γ) (act : Expr (Γ + 1))
    (W : Fin n → Fin n → Expr Γ) (xv : Fin n → Fin Γ) (i : Fin n) : Expr Γ :=
  .mul (dy i) (.letE (preAct W xv i) (actDeriv act))

/-- **The input gradient**, as the transposed matvec kernel computes it:
    `dxⱼ = Σᵢ dsᵢ · Wᵢⱼ`. -/
def layerDx (dy : Fin n → Expr Γ) (act : Expr (Γ + 1))
    (W : Fin n → Fin n → Expr Γ) (xv : Fin n → Fin Γ) (j : Fin n) : Expr Γ :=
  .sum n (fun i => .mul (layerAdj dy act W xv i) (W i j))

/-- The activation reads only the value it was applied to — no model input
    reaches it except through `z`.  True of every activation in the stack. -/
def ActClosed (act : Expr (Γ + 1)) : Prop :=
  ∀ q : Fin Γ, NotUses ⟨q.val, Nat.lt_succ_of_lt q.isLt⟩ act

/-- Termwise equality lifts to a sum. -/
theorem denote_sum_congr (env : Fin Γ → R) (f g : Fin n → Expr Γ)
    (h : ∀ i, denote env (f i) = denote env (g i)) :
    denote env (.sum n f) = denote env (.sum n g) := by
  show (List.finRange n).foldl (fun acc i => add acc (denote env (f i))) zero
      = (List.finRange n).foldl (fun acc i => add acc (denote env (g i))) zero
  simp only [h]

/-- **∂zᵢ/∂xⱼ = Wᵢⱼ.**  The delta collapse, at the layer.

    `hinj` is what makes it `Wᵢⱼ` rather than a sum of several: distinct inputs
    must be distinct variables.  `hW` says the weights are not themselves
    functions of the inputs — true of a layer, and false of, say, weight
    tying, which is why it is a hypothesis rather than an assumption. -/
theorem preAct_deriv (W : Fin n → Fin n → Expr Γ) (xv : Fin n → Fin Γ)
    (hW : ∀ i k j, NotUses (xv j) (W i k)) (hinj : ∀ a b, xv a = xv b → a = b)
    (env : Fin Γ → R) (i j0 : Fin n) :
    denote env (sderiv (preAct W xv i) (xv j0)) = denote env (W i j0) := by
  have hzl : ZeroLaws R := zeroLaws_of_numLaws R
  show denote env (Expr.sum n (fun j =>
    Expr.add (.mul (sderiv (W i j) (xv j0)) (.var (xv j)))
             (.mul (W i j) (sderiv (Expr.var (xv j)) (xv j0))))) = _
  rw [denote_sum_single env _ j0 ?_]
  · -- the surviving term
    show add (mul (denote env (sderiv (W i j0) (xv j0))) _)
             (mul (denote env (W i j0))
                  (denote env (if xv j0 = xv j0 then Expr.lit 1 else Expr.lit 0))) = _
    rw [sderiv_notUses hzl _ _ env (hW i j0 j0), if_pos rfl]
    show add (mul (ofNat 0) _) (mul (denote env (W i j0)) (ofNat 1)) = _
    rw [NumLaws.ofNat_zero, NumLaws.zero_mul, NumLaws.zero_add,
        NumLaws.ofNat_one, NumLaws.mul_one]
  · -- every other term is zero
    intro j hj
    show add (mul (denote env (sderiv (W i j) (xv j0))) _)
             (mul (denote env (W i j))
                  (denote env (if xv j0 = xv j then Expr.lit 1 else Expr.lit 0))) = _
    rw [sderiv_notUses hzl _ _ env (hW i j j0),
        if_neg (fun hc => hj (hinj j j0 hc.symm))]
    show add (mul (ofNat 0) _) (mul (denote env (W i j)) (ofNat 0)) = _
    rw [NumLaws.ofNat_zero, NumLaws.zero_mul, NumLaws.mul_zero, NumLaws.add_zero]

/-- **The chain rule at the layer, against any variable.**

    Obtained structurally: `sderiv`'s `letE` rule *is* the chain rule, so
    writing the layer with `zᵢ` bound — which the sharing argument already
    required — means this file proves nothing about substitution. -/
theorem layerOut_deriv_gen (act : Expr (Γ + 1)) (W : Fin n → Fin n → Expr Γ)
    (xv : Fin n → Fin Γ) (hact : ActClosed act) (env : Fin Γ → R) (i : Fin n)
    (q : Fin Γ) :
    denote env (sderiv (layerOut act W xv i) q)
      = mul (denote (extend env (denote env (preAct W xv i))) (actDeriv act))
            (denote env (sderiv (preAct W xv i) q)) := by
  have hzl : ZeroLaws R := zeroLaws_of_numLaws R
  show denote (extend env (denote env (preAct W xv i)))
        (Expr.add (sderiv act ⟨q.val, by omega⟩)
                  (.mul (sderiv act ⟨Γ, Nat.lt_succ_self Γ⟩)
                        (wk (sderiv (preAct W xv i) q)))) = _
  show add (denote _ (sderiv act _)) (mul (denote _ (sderiv act _)) (denote _ (wk _))) = _
  rw [sderiv_notUses hzl _ _ _ (hact q), denote_wk, NumLaws.ofNat_zero,
      NumLaws.zero_add]
  rfl

/-- **∂yᵢ/∂xⱼ = act'(zᵢ)·Wᵢⱼ.** -/
theorem layerOut_deriv (act : Expr (Γ + 1)) (W : Fin n → Fin n → Expr Γ)
    (xv : Fin n → Fin Γ) (hact : ActClosed act)
    (hW : ∀ i k j, NotUses (xv j) (W i k)) (hinj : ∀ a b, xv a = xv b → a = b)
    (env : Fin Γ → R) (i j0 : Fin n) :
    denote env (sderiv (layerOut act W xv i) (xv j0))
      = mul (denote (extend env (denote env (preAct W xv i))) (actDeriv act))
            (denote env (W i j0)) := by
  rw [layerOut_deriv_gen act W xv hact env i (xv j0),
      preAct_deriv W xv hW hinj env i j0]

/-- **The composition theorem: backprop through the layer is its gradient.**

    Left: `sderiv` of the loss with respect to input `j` — the specification.
    Right: what the kernels compute — the activation backward feeding the
    transposed matvec.

    This is the statement that was missing.  Each backward kernel was proven
    against its own spec; this says the *sequence* of those specs is the
    derivative of the forward model, which is what a user training a model is
    actually relying on. -/
theorem layer_backprop_dx (dy : Fin n → Expr Γ) (act : Expr (Γ + 1))
    (W : Fin n → Fin n → Expr Γ) (xv : Fin n → Fin Γ)
    (hact : ActClosed act) (hW : ∀ i k j, NotUses (xv j) (W i k))
    (hdy : ∀ i j, NotUses (xv j) (dy i)) (hinj : ∀ a b, xv a = xv b → a = b)
    (env : Fin Γ → R) (j0 : Fin n) :
    denote env (sderiv (layerLoss dy act W xv) (xv j0))
      = denote env (layerDx dy act W xv j0) := by
  have hzl : ZeroLaws R := zeroLaws_of_numLaws R
  show denote env (Expr.sum n (fun i =>
    Expr.add (.mul (sderiv (dy i) (xv j0)) (layerOut act W xv i))
             (.mul (dy i) (sderiv (layerOut act W xv i) (xv j0))))) = _
  refine denote_sum_congr env _ _ (fun i => ?_)
  show add (mul (denote env (sderiv (dy i) (xv j0))) _)
           (mul (denote env (dy i)) (denote env (sderiv (layerOut act W xv i) (xv j0))))
        = denote env (.mul (layerAdj dy act W xv i) (W i j0))
  rw [sderiv_notUses hzl _ _ env (hdy i j0), NumLaws.ofNat_zero, NumLaws.zero_mul,
      NumLaws.zero_add, layerOut_deriv act W xv hact hW hinj env i j0]
  show mul (denote env (dy i)) (mul _ (denote env (W i j0)))
      = mul (mul (denote env (dy i)) _) (denote env (W i j0))
  rw [NumLaws.mul_assoc]
  rfl

-- ---------------------------------------------------------------------------
-- The weight gradient
-- ---------------------------------------------------------------------------

/-!
  `dx` is what flows to the previous layer; `dW` is what the optimiser actually
  updates, so a training step is not proven until both are.  The kernel is the
  outer product `dWᵢⱼ = dsᵢ·xⱼ` (`zipPassEW`), and the extra content on this side
  is that the *outer* sum collapses too: weight `Wᵢⱼ` appears in exactly one
  row's pre-activation, so `Σᵢ'` keeps only `i' = i`.
-/

/-- Weights as variables — needed to differentiate against them. -/
def varMat {Γ n : Nat} (wv : Fin n → Fin n → Fin Γ) : Fin n → Fin n → Expr Γ :=
  fun a b => .var (wv a b)

variable {R : Type} [NumLaws R] {Γ n : Nat}

/-- **∂zᵢ'/∂Wᵢⱼ = xⱼ when i' = i, and 0 otherwise.** -/
theorem preAct_deriv_w (wv : Fin n → Fin n → Fin Γ) (xv : Fin n → Fin Γ)
    (hwinj : ∀ a b c d, wv a b = wv c d → a = c ∧ b = d)
    (hdisj : ∀ a b k, wv a b ≠ xv k)
    (env : Fin Γ → R) (i' i j : Fin n) :
    denote env (sderiv (preAct (varMat wv) xv i') (wv i j))
      = if i' = i then env (xv j) else zero := by
  have hzl : ZeroLaws R := zeroLaws_of_numLaws R
  have hterm : ∀ k : Fin n,
      denote env (sderiv (Expr.mul (varMat wv i' k) (.var (xv k))) (wv i j))
        = if i' = i ∧ k = j then env (xv j) else zero := by
    intro k
    show add (mul (denote env (if wv i j = wv i' k then Expr.lit 1 else Expr.lit 0)) _)
             (mul _ (denote env (if wv i j = xv k then Expr.lit 1 else Expr.lit 0))) = _
    rw [if_neg (hdisj i j k)]
    by_cases hik : i' = i ∧ k = j
    · obtain ⟨h1, h2⟩ := hik
      subst h1; subst h2
      rw [if_pos rfl, if_pos (by simp)]
      show add (mul (ofNat 1) (env (xv k))) (mul _ (ofNat 0)) = _
      rw [NumLaws.ofNat_one, NumLaws.one_mul, NumLaws.ofNat_zero, NumLaws.mul_zero,
          NumLaws.add_zero]
    · rw [if_neg (fun hc => hik (by
            obtain ⟨ha, hb⟩ := hwinj i j i' k hc
            exact ⟨ha.symm, hb.symm⟩)), if_neg hik]
      show add (mul (ofNat 0) _) (mul _ (ofNat 0)) = _
      rw [NumLaws.ofNat_zero, NumLaws.zero_mul, NumLaws.mul_zero, NumLaws.add_zero]
  show denote env (Expr.sum n (fun k => sderiv (.mul (varMat wv i' k) (.var (xv k))) (wv i j))) = _
  by_cases hi : i' = i
  · subst hi
    rw [denote_sum_single env _ j (fun k hk => by
          rw [hterm k, if_neg (fun hc => hk hc.2)]), hterm j, if_pos ⟨rfl, rfl⟩,
        if_pos rfl]
  · show (List.finRange n).foldl (fun acc k => add acc (denote env (sderiv _ _))) zero = _
    rw [foldl_add_zeros _ (List.finRange n) (fun k _ => by
          rw [hterm k, if_neg (fun hc => hi hc.1)]) zero, if_neg hi]

/-- **The composition theorem for the weight gradient.**

    Left: `sderiv` of the loss with respect to `Wᵢⱼ`.  Right: the outer product
    the `zipPassEW` kernel computes, from the adjoint the activation-backward
    kernel produced.  Together with `layer_backprop_dx`, this is a full training
    step for the layer, proven against the same `sderiv` the forward is. -/
theorem layer_backprop_dW (dy : Fin n → Expr Γ) (act : Expr (Γ + 1))
    (wv : Fin n → Fin n → Fin Γ) (xv : Fin n → Fin Γ) (hact : ActClosed act)
    (hdy : ∀ i' i j, NotUses (wv i j) (dy i'))
    (hwinj : ∀ a b c d, wv a b = wv c d → a = c ∧ b = d)
    (hdisj : ∀ a b k, wv a b ≠ xv k)
    (env : Fin Γ → R) (i j : Fin n) :
    denote env (sderiv (layerLoss dy act (varMat wv) xv) (wv i j))
      = denote env (.mul (layerAdj dy act (varMat wv) xv i) (.var (xv j))) := by
  have hzl : ZeroLaws R := zeroLaws_of_numLaws R
  have hterm : ∀ i' : Fin n,
      denote env (sderiv (Expr.mul (dy i') (layerOut act (varMat wv) xv i')) (wv i j))
        = if i' = i then
            denote env (.mul (layerAdj dy act (varMat wv) xv i) (.var (xv j)))
          else zero := by
    intro i'
    show add (mul (denote env (sderiv (dy i') (wv i j))) _)
             (mul (denote env (dy i'))
                  (denote env (sderiv (layerOut act (varMat wv) xv i') (wv i j)))) = _
    rw [sderiv_notUses hzl _ _ env (hdy i' i j), NumLaws.ofNat_zero,
        NumLaws.zero_mul, NumLaws.zero_add,
        layerOut_deriv_gen act (varMat wv) xv hact env i' (wv i j),
        preAct_deriv_w wv xv hwinj hdisj env i' i j]
    by_cases hi : i' = i
    · subst hi
      rw [if_pos rfl, if_pos rfl]
      show mul (denote env (dy i')) (mul _ (env (xv j)))
          = mul (mul (denote env (dy i')) _) (denote env (Expr.var (xv j)))
      rw [NumLaws.mul_assoc]
      rfl
    · rw [if_neg hi, if_neg hi, NumLaws.mul_zero, NumLaws.mul_zero]
  show denote env (Expr.sum n (fun i' =>
    sderiv (.mul (dy i') (layerOut act (varMat wv) xv i')) (wv i j))) = _
  rw [denote_sum_single env _ i (fun i' hi' => by rw [hterm i', if_neg hi']),
      hterm i, if_pos rfl]

-- ---------------------------------------------------------------------------
-- A standard variable layout, with every side condition discharged
-- ---------------------------------------------------------------------------

/-!
  The composition theorems take five hypotheses about the variable layout —
  inputs distinct, weights distinct, weights disjoint from inputs, weights not
  functions of the inputs, adjoints not functions of the weights.  Each is true
  of any sane layout and none of them should be a user's problem.

  `stdVars` fixes the obvious layout — inputs, then the weight matrix
  row-major, then the adjoints — and proves all five.  A user instantiating the
  theorem supplies a width and nothing else.
-/

/-- The context a layer needs: `n` inputs, `n²` weights, `n` adjoints. -/
abbrev layerCtx (n : Nat) : Nat := n + n * n + n

/-- Input `j` is variable `j`. -/
def stdXv (n : Nat) : Fin n → Fin (layerCtx n) := fun j =>
  ⟨j.val, by have := j.isLt; show j.val < n + n * n + n; omega⟩

/-- Weight `Wᵢⱼ` is variable `n + i·n + j`. -/
def stdWv (n : Nat) : Fin n → Fin n → Fin (layerCtx n) :=
  fun i j => ⟨n + (i.val * n + j.val), by
    have := rowMajor_lt i j
    show n + (i.val * n + j.val) < n + n * n + n
    omega⟩

/-- Adjoint `dyₖ` is variable `n + n² + k`. -/
def stdDyv (n : Nat) : Fin n → Fin (layerCtx n) := fun k =>
  ⟨n + n * n + k.val, by have := k.isLt; show n + n * n + k.val < n + n * n + n; omega⟩

theorem stdXv_inj (n : Nat) : ∀ a b, stdXv n a = stdXv n b → a = b := by
  intro a b h
  have hv : (stdXv n a).val = (stdXv n b).val := congrArg Fin.val h
  exact Fin.ext hv

theorem stdWv_inj (n : Nat) :
    ∀ a b c d, stdWv n a b = stdWv n c d → a = c ∧ b = d := by
  intro a b c d h
  have hv : n + (a.val * n + b.val) = n + (c.val * n + d.val) := congrArg Fin.val h
  obtain ⟨h1, h2⟩ := rowMajor_inj b.isLt d.isLt (show a.val * n + b.val = c.val * n + d.val by omega)
  exact ⟨Fin.ext h1, Fin.ext h2⟩

theorem stdWv_disj (n : Nat) : ∀ a b k, stdWv n a b ≠ stdXv n k := by
  intro a b k hc
  have hv : n + (a.val * n + b.val) = k.val := congrArg Fin.val hc
  have := k.isLt
  omega

/-- Weights are variables that are not inputs, so they do not depend on them. -/
theorem stdW_notUses (n : Nat) :
    ∀ i k j, NotUses (stdXv n j) (varMat (stdWv n) i k) := by
  intro i k j hc
  exact stdWv_disj n i k j hc.symm

/-- Adjoints are variables that are not weights. -/
theorem stdDy_notUses (n : Nat) :
    ∀ i' i j, NotUses (stdWv n i j) (Expr.var (stdDyv n i')) := by
  intro i' i j hc
  have hv : n + (i.val * n + j.val) = n + n * n + i'.val := congrArg Fin.val hc
  have h1 : i.val * n + j.val < n * n := rowMajor_lt i j
  omega

/-- Adjoints are not inputs either. -/
theorem stdDy_notUses_x (n : Nat) :
    ∀ i' j, NotUses (stdXv n j) (Expr.var (stdDyv n i')) := by
  intro i' j hc
  have hv : j.val = n + n * n + i'.val := congrArg Fin.val hc
  have := j.isLt
  omega

-- ---------------------------------------------------------------------------
-- The side condition, discharged for the stack's activations
-- ---------------------------------------------------------------------------

/-- **`ActClosed` holds for SiLU** — so the hypothesis costs a user nothing at
    the activation the stack actually ships.  Same for sigmoid and for anything
    else built from the smooth constructors applied to the bound value. -/
theorem actClosed_silu {Γ : Nat} :
    ActClosed (Γ := Γ) (Transformer.silu (.var ⟨Γ, Nat.lt_succ_self Γ⟩)) := by
  intro q
  have hne : (⟨q.val, Nat.lt_succ_of_lt q.isLt⟩ : Fin (Γ + 1))
      ≠ ⟨Γ, Nat.lt_succ_self Γ⟩ := by
    intro hc
    have hv : q.val = Γ := congrArg Fin.val hc
    have := q.isLt
    omega
  exact ⟨hne, trivial, hne⟩

theorem actClosed_sigmoid {Γ : Nat} :
    ActClosed (Γ := Γ) (Transformer.sigmoid (.var ⟨Γ, Nat.lt_succ_self Γ⟩)) := by
  intro q
  have hne : (⟨q.val, Nat.lt_succ_of_lt q.isLt⟩ : Fin (Γ + 1))
      ≠ ⟨Γ, Nat.lt_succ_self Γ⟩ := by
    intro hc
    have hv : q.val = Γ := congrArg Fin.val hc
    have := q.isLt
    omega
  exact ⟨trivial, hne⟩

-- ---------------------------------------------------------------------------
-- The front door: a training step with nothing left to discharge
-- ---------------------------------------------------------------------------

section Std

variable {R : Type} [NumLaws R] {n : Nat}

/-- A SiLU layer at the standard layout. -/
def stdLoss (n : Nat) : Expr (layerCtx n) :=
  layerLoss (fun k => .var (stdDyv n k))
    (Transformer.silu (.var ⟨layerCtx n, Nat.lt_succ_self _⟩))
    (varMat (stdWv n)) (stdXv n)

/-- Its adjoint — what the activation-backward kernel computes. -/
def stdAdj (n : Nat) (i : Fin n) : Expr (layerCtx n) :=
  layerAdj (fun k => .var (stdDyv n k))
    (Transformer.silu (.var ⟨layerCtx n, Nat.lt_succ_self _⟩))
    (varMat (stdWv n)) (stdXv n) i

/-- **The input gradient, with no hypotheses.**

    Everything the general theorem asks for is discharged by the standard
    layout and `actClosed_silu`.  A user gets: *the transposed matvec of the
    activation backward is the derivative of the layer* — and writes nothing. -/
theorem stdLayer_dx (env : Fin (layerCtx n) → R) (j : Fin n) :
    denote env (sderiv (stdLoss n) (stdXv n j))
      = denote env (.sum n (fun i => .mul (stdAdj n i) (varMat (stdWv n) i j))) :=
  layer_backprop_dx _ _ _ _ actClosed_silu (stdW_notUses n)
    (fun i j' => stdDy_notUses_x n i j') (stdXv_inj n) env j

/-- **The weight gradient, with no hypotheses.**  The outer product of the
    adjoint with the input is `∂L/∂Wᵢⱼ`. -/
theorem stdLayer_dW (env : Fin (layerCtx n) → R) (i j : Fin n) :
    denote env (sderiv (stdLoss n) (stdWv n i j))
      = denote env (.mul (stdAdj n i) (.var (stdXv n j))) :=
  layer_backprop_dW _ _ _ _ actClosed_silu (stdDy_notUses n) (stdWv_inj n)
    (stdWv_disj n) env i j

end Std

end Layer

end AlgorithmLib.ML
