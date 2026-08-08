import AlgorithmLib.ML.Tape

/-!
  # `gradProg` is correct

  `Tape.gradProg` builds the reverse sweep as a *program*: one telescope holding
  every intermediate adjoint, and `Γ` outputs that refer to those adjoints by
  variable.  That is what turned the gradient from exponential into quadratic.
  This file proves it computes the right thing.

  The statement is against `sderiv` — the textbook symbolic derivative — exactly
  as `grad_correct` is, so the two implementations of reverse mode are held to
  the same specification.

  ## The shape of the argument

  A telescope is built with `cons` at the *end*, and `Tele.bind` pushes the last
  binding *innermost*:

      (cons t e).bind b  =  t.bind (letE e b)

  Reverse mode also starts at the end.  So an induction on the telescope walks
  the bindings in exactly the order the sweep does, and the `letE` rule of
  `sderiv`,

      d/dxₖ (let v = a in b) = ∂b/∂xₖ + (∂b/∂v)·(da/dxₖ)

  *is* the adjoint recurrence: `∂b/∂v` is the adjoint of the slot `e` binds.

  The proof is in three steps.

  1. `adjSem` — the adjoint of every slot, defined semantically by that same
     recursion.  `adjSem_spec` says the slots below `Γ` are the derivatives we
     want; it is immediate, because peeling a `cons` is peeling a `letE`.
  2. `adjSem_fold` — `adjSem` satisfies the closed-form recurrence that `adjAt`
     is the syntax of.  This is the content.
  3. `adjTele_env` / `gradProg_correct` — the telescope really does bind those
     values, so the `Γ` outputs denote the `Γ` partial derivatives.

  ## No `NumLaws`

  Nothing here needs associativity, commutativity or distributivity: the
  left-fold order `adjAt` commits to is precisely the order the induction
  produces.  So the theorem holds at **`Float32`**, on the nose, not just at ℝ.
  That is deliberate — an adjoint accumulation proven only up to reassociation
  would say nothing about the kernel that runs.
-/

namespace AlgorithmLib.ML

open NumOps

variable {R : Type} [NumOps R] {Γ : Nat}

-- ---------------------------------------------------------------------------
-- 0. Differentiation commutes with renaming
-- ---------------------------------------------------------------------------

/-! Absorbing a telescope binding into the body puts the differentiated term
    under one more binder, so the proof needs to move `sderiv` across a
    renaming.  That is true exactly when the renaming is **injective**: the
    `var` case compares indices, and a collapsing renaming would turn a `0`
    partial into a `1`.  Weakening is injective, which is all we use. -/

theorem liftRen_inj {Δ Δ' : Nat} (ρ : Fin Δ → Fin Δ')
    (hinj : ∀ a b, ρ a = ρ b → a = b) :
    ∀ a b : Fin (Δ + 1), liftRen ρ a = liftRen ρ b → a = b := by
  intro a b hab
  have hv := congrArg Fin.val hab
  have ha1 := a.isLt
  have hb1 := b.isLt
  by_cases ha : a.val < Δ <;> by_cases hb : b.val < Δ
  · simp only [liftRen, dif_pos ha, dif_pos hb] at hv
    have h2 := hinj ⟨a.val, ha⟩ ⟨b.val, hb⟩ (Fin.ext hv)
    have h3 : (⟨a.val, ha⟩ : Fin Δ).val = (⟨b.val, hb⟩ : Fin Δ).val := congrArg Fin.val h2
    exact Fin.ext h3
  · exfalso
    simp only [liftRen, dif_pos ha, dif_neg hb] at hv
    have := (ρ ⟨a.val, ha⟩).isLt
    omega
  · exfalso
    simp only [liftRen, dif_neg ha, dif_pos hb] at hv
    have := (ρ ⟨b.val, hb⟩).isLt
    omega
  · exact Fin.ext (by omega)

theorem liftRen_comp {Γ₁ Γ₂ Γ₃ : Nat} (ρ : Fin Γ₁ → Fin Γ₂) (σ : Fin Γ₂ → Fin Γ₃) :
    (fun i => liftRen σ (liftRen ρ i)) = liftRen (fun i => σ (ρ i)) := by
  funext k
  by_cases h : k.val < Γ₁
  · simp only [liftRen, dif_pos h, (ρ ⟨k.val, h⟩).isLt, dif_pos]
  · simp only [liftRen, dif_neg h, dif_neg (Nat.lt_irrefl Γ₂)]

theorem rename_comp : ∀ {Γ₁ Γ₂ Γ₃ : Nat} (ρ : Fin Γ₁ → Fin Γ₂) (σ : Fin Γ₂ → Fin Γ₃)
    (e : Expr Γ₁), rename σ (rename ρ e) = rename (fun i => σ (ρ i)) e := by
  intro Γ₁ Γ₂ Γ₃ ρ σ e
  induction e generalizing Γ₂ Γ₃ with
  | var i => rfl
  | lit n => rfl
  | add a b iha ihb => show Expr.add _ _ = Expr.add _ _; rw [iha ρ σ, ihb ρ σ]
  | mul a b iha ihb => show Expr.mul _ _ = Expr.mul _ _; rw [iha ρ σ, ihb ρ σ]
  | neg a ih   => show Expr.neg _ = Expr.neg _; rw [ih ρ σ]
  | inv a ih   => show Expr.inv _ = Expr.inv _; rw [ih ρ σ]
  | exp a ih   => show Expr.exp _ = Expr.exp _; rw [ih ρ σ]
  | rsqrt a ih => show Expr.rsqrt _ = Expr.rsqrt _; rw [ih ρ σ]
  | sum n f ih =>
      show Expr.sum n _ = Expr.sum n _
      congr 1
      funext j
      exact ih j ρ σ
  | letE a b iha ihb =>
      show Expr.letE _ _ = Expr.letE _ _
      rw [iha ρ σ, ihb (liftRen ρ) (liftRen σ), liftRen_comp ρ σ]

/-- **`sderiv` commutes with an injective renaming.**  The `letE` case is where
    it earns its keep: the binder moves, so the lifted renaming has to be pushed
    through both the body's partials and the weakened sub-derivative. -/
theorem sderiv_rename : ∀ {Δ Δ' : Nat} (ρ : Fin Δ → Fin Δ')
    (hinj : ∀ a b, ρ a = ρ b → a = b) (e : Expr Δ) (k : Fin Δ),
    sderiv (rename ρ e) (ρ k) = rename ρ (sderiv e k) := by
  intro Δ Δ' ρ hinj e
  induction e generalizing Δ' with
  | var i =>
      intro k
      show (if _ then _ else _) = rename ρ (if _ then _ else _)
      by_cases h : k = i
      · rw [h, if_pos rfl, if_pos rfl]; rfl
      · rw [if_neg h, if_neg (fun hc => h (hinj _ _ hc))]; rfl
  | lit n => intro k; rfl
  | add a b iha ihb =>
      intro k; show Expr.add _ _ = Expr.add _ _; rw [iha ρ hinj k, ihb ρ hinj k]
  | mul a b iha ihb =>
      intro k
      show Expr.add (.mul _ _) (.mul _ _) = Expr.add (.mul _ _) (.mul _ _)
      rw [iha ρ hinj k, ihb ρ hinj k]
  | neg a ih => intro k; show Expr.neg _ = Expr.neg _; rw [ih ρ hinj k]
  | inv a ih =>
      intro k
      show Expr.neg (.mul _ (.mul (.inv _) (.inv _)))
        = Expr.neg (.mul _ (.mul (.inv _) (.inv _)))
      rw [ih ρ hinj k]
  | exp a ih =>
      intro k; show Expr.mul _ (.exp _) = Expr.mul _ (.exp _); rw [ih ρ hinj k]
  | rsqrt a ih =>
      intro k
      show Expr.neg (.mul _ (.mul (.inv (.lit 2)) (.mul (.rsqrt _) (.inv _))))
        = Expr.neg (.mul _ (.mul (.inv (.lit 2)) (.mul (.rsqrt _) (.inv _))))
      rw [ih ρ hinj k]
  | sum n f ih =>
      intro k
      show Expr.sum n _ = Expr.sum n _
      congr 1
      funext j
      exact ih j ρ hinj k
  | @letE Δ₀ a b iha ihb =>
      intro k
      have hlt : liftRen ρ ⟨k.val, Nat.lt_succ_of_lt k.isLt⟩
          = ⟨(ρ k).val, Nat.lt_succ_of_lt (ρ k).isLt⟩ := by
        simp only [liftRen, dif_pos k.isLt]
      have hlast : liftRen ρ ⟨Δ₀, Nat.lt_succ_self Δ₀⟩ = ⟨Δ', Nat.lt_succ_self Δ'⟩ := by
        simp only [liftRen, dif_neg (Nat.lt_irrefl Δ₀)]
      have hwk : ∀ y : Expr Δ₀, rename (liftRen ρ) (wk y) = wk (rename ρ y) := by
        intro y
        show rename (liftRen ρ) (rename _ y) = rename _ (rename ρ y)
        rw [rename_comp, rename_comp]
        congr 1
        funext i
        simp only [liftRen, dif_pos i.isLt]
      show Expr.letE _ (.add _ (.mul _ _)) = Expr.letE _ (.add _ (.mul _ _))
      rw [← hlt, ← hlast, ihb (liftRen ρ) (liftRen_inj ρ hinj) ⟨k.val, Nat.lt_succ_of_lt k.isLt⟩,
          ihb (liftRen ρ) (liftRen_inj ρ hinj) ⟨Δ₀, Nat.lt_succ_self Δ₀⟩, iha ρ hinj k, hwk]

theorem wkMap_inj {Δ : Nat} :
    ∀ a b : Fin Δ, (⟨a.val, Nat.lt_succ_of_lt a.isLt⟩ : Fin (Δ + 1))
      = ⟨b.val, Nat.lt_succ_of_lt b.isLt⟩ → a = b := by
  intro a b h
  have h2 : (⟨a.val, Nat.lt_succ_of_lt a.isLt⟩ : Fin (Δ + 1)).val
      = (⟨b.val, Nat.lt_succ_of_lt b.isLt⟩ : Fin (Δ + 1)).val := congrArg Fin.val h
  exact Fin.ext h2

theorem sderiv_wk {Δ : Nat} (f : Expr Δ) (k : Fin Δ) :
    sderiv (wk f) ⟨k.val, Nat.lt_succ_of_lt k.isLt⟩ = wk (sderiv f k) :=
  sderiv_rename (fun i : Fin Δ => (⟨i.val, Nat.lt_succ_of_lt i.isLt⟩ : Fin (Δ + 1)))
    wkMap_inj f k

/-- Differentiating a weakened term, under the extended environment, is
    differentiating the original.  The workhorse of the telescope induction. -/
theorem denote_sderiv_wk {Δ : Nat} (f : Expr Δ) (k : Fin Δ) (env : Fin Δ → R) (v : R) :
    denote (extend env v) (sderiv (wk f) ⟨k.val, Nat.lt_succ_of_lt k.isLt⟩)
      = denote env (sderiv f k) := by
  rw [sderiv_wk, denote_wk]

-- ---------------------------------------------------------------------------
-- 1. The semantic adjoint
-- ---------------------------------------------------------------------------

/-- The adjoint of every slot of a telescope, with respect to the body `b`.

    Defined by the same recursion `Tele.bind` uses: to differentiate a telescope
    with a last binding `e`, absorb `e` into the body and recurse.  The last
    slot's adjoint is then just `∂b/∂v` — nothing is bound after it. -/
def adjSem : {n : Nat} → Tele Γ n → Expr (Γ + n) → (Fin Γ → R) → Fin (Γ + n) → R
  | 0,     .nil,      b, env, q => denote env (sderiv b q)
  | n + 1, .cons t e, b, env, q =>
      if h : q.val < Γ + n then adjSem t (.letE e b) env ⟨q.val, h⟩
      else denote ((Tele.cons t e).env env) (sderiv b q)

theorem adjSem_cons_lt {n : Nat} (t : Tele Γ n) (e : Expr (Γ + n)) (b : Expr (Γ + n + 1))
    (env : Fin Γ → R) (q : Fin (Γ + (n + 1))) (h : q.val < Γ + n) :
    adjSem (.cons t e) b env q = adjSem t (.letE e b) env ⟨q.val, h⟩ := by
  show (if h' : q.val < Γ + n then _ else _) = _
  rw [dif_pos h]

theorem adjSem_cons_last {n : Nat} (t : Tele Γ n) (e : Expr (Γ + n)) (b : Expr (Γ + n + 1))
    (env : Fin Γ → R) (q : Fin (Γ + (n + 1))) (h : ¬ q.val < Γ + n) :
    adjSem (.cons t e) b env q = denote ((Tele.cons t e).env env) (sderiv b q) := by
  show (if h' : q.val < Γ + n then _ else _) = _
  rw [dif_neg h]

/-- **The adjoints of the input slots are the derivatives.**

    Immediate by induction: peeling a `cons` off the telescope is peeling a
    `letE` off the bound expression, and `adjSem` was defined to follow it. -/
theorem adjSem_spec : ∀ {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (env : Fin Γ → R)
    (k : Fin Γ),
    adjSem t out env ⟨k.val, by omega⟩ = denote env (sderiv (t.bind out) k) := by
  intro n t
  induction t with
  | nil =>
      intro out env k
      show denote env (sderiv out ⟨k.val, by omega⟩) = denote env (sderiv out k)
      congr 1
  | cons t e ih =>
      intro out env k
      rw [adjSem_cons_lt t e out env _ (show k.val < Γ + _ by omega)]
      exact ih (.letE e out) env k

-- ---------------------------------------------------------------------------
-- 2. The adjoint recurrence
-- ---------------------------------------------------------------------------

/-! `adjAt` is a left fold over the adjoint bindings, in the order
    `j' = 0, 1, …`, which is forward slot `n-1, n-2, …`.  So the semantic
    counterpart folds over the forward indices **in reverse**, and peeling the
    first element of that list is peeling the last `cons` of the telescope —
    which is exactly the induction step.  Getting these two orders to line up is
    what lets the whole proof avoid reassociation. -/

theorem foldl_congr_mem {α β : Type} (f g : β → α → β) : ∀ (l : List α) (init : β),
    (∀ b a, a ∈ l → f b a = g b a) → l.foldl f init = l.foldl g init := by
  intro l
  induction l with
  | nil => intro init _; rfl
  | cons a l ih =>
      intro init h
      show l.foldl f (f init a) = l.foldl g (g init a)
      rw [h init a (List.Mem.head _)]
      exact ih (g init a) (fun b x hx => h b x (List.Mem.tail _ hx))

theorem foldl_const {α β : Type} (f : β → α → β) : ∀ (l : List α) (init : β),
    (∀ b a, a ∈ l → f b a = b) → l.foldl f init = init := by
  intro l
  induction l with
  | nil => intro init _; rfl
  | cons a l ih =>
      intro init h
      show l.foldl f (f init a) = init
      rw [h init a (List.Mem.head _)]
      exact ih init (fun b x hx => h b x (List.Mem.tail _ hx))

theorem revIdx_succ (n : Nat) : revIdx (n + 1) = n :: revIdx n := rfl

theorem mem_revIdx : ∀ {n i : Nat}, i ∈ revIdx n → i < n := by
  intro n
  induction n with
  | zero => intro i h; cases h
  | succ n ih =>
      intro i h
      rcases h with _ | ⟨_, h⟩
      · exact Nat.lt_succ_self n
      · exact Nat.lt_succ_of_lt (ih h)

/-- `revTake n j` is a prefix of `revIdx n`, with `revIdx (n-j)` as the rest. -/
theorem revTake_append : ∀ {n : Nat} (j : Nat), j ≤ n →
    revTake n j ++ revIdx (n - j) = revIdx n := by
  intro n j
  induction j with
  | zero => intro _; rfl
  | succ j ih =>
      intro hj
      have hrest : revIdx (n - j) = (n - 1 - j) :: revIdx (n - (j + 1)) := by
        have h1 : n - j = (n - (j + 1)) + 1 := by omega
        rw [h1, revIdx_succ]
        have h2 : n - (j + 1) = n - 1 - j := by omega
        rw [h2]
      show (revTake n j ++ [n - 1 - j]) ++ revIdx (n - (j + 1)) = revIdx n
      rw [List.append_assoc]
      show revTake n j ++ ((n - 1 - j) :: revIdx (n - (j + 1))) = revIdx n
      rw [← hrest]
      exact ih (by omega)

/-- One term of the reverse accumulation: the adjoint of binding `i` times that
    binding's partial derivative with respect to slot `q`.

    The `q.val < Γ + i` guard is not an optimisation — it is what keeps the
    theorem exact.  Binding `i` cannot mention slot `q` when `q ≥ Γ + i`, so the
    term is semantically zero; but dropping a zero term is only sound in a
    carrier where `acc + w·0 = acc`, which `Float32` is not.  Guarding instead
    of simplifying is what makes this provable on the nose. -/
def adjStep {n : Nat} (t : Tele Γ n) (b : Expr (Γ + n)) (env : Fin Γ → R)
    (q : Fin (Γ + n)) (acc : R) (i : Nat) : R :=
  if h : i < n then
    if q.val < Γ + i then
      add acc (mul (adjSem t b env ⟨Γ + i, by omega⟩)
                   (denote (t.env env) (sderiv (t.getW ⟨i, h⟩) q)))
    else acc
  else acc

theorem Tele.getW_cons_lt {n : Nat} (t : Tele Γ n) (e : Expr (Γ + n)) (i : Fin (n + 1))
    (h : i.val < n) : (Tele.cons t e).getW i = wk (t.getW ⟨i.val, h⟩) := by
  show (if h' : i.val < n then _ else _) = _
  rw [dif_pos h]

theorem Tele.getW_cons_last {n : Nat} (t : Tele Γ n) (e : Expr (Γ + n)) (i : Fin (n + 1))
    (h : ¬ i.val < n) : (Tele.cons t e).getW i = wk e := by
  show (if h' : i.val < n then _ else _) = _
  rw [dif_neg h]

/-- **The adjoint satisfies the closed-form recurrence.**

    `∂out/∂q`, then one term per later binding, accumulated left to right — the
    exact expression `adjAt` is the syntax of.

    The induction is on the telescope.  Peeling the last binding `e` turns the
    body `b` into `letE e b`, and `sderiv`'s rule for `letE`

        ∂(let v = e in b)/∂q = ∂b/∂q + (∂b/∂v)·(∂e/∂q)

    *is* the first step of the fold: `∂b/∂v` is the adjoint of the slot `e`
    binds. -/
theorem adjSem_fold : ∀ {n : Nat} (t : Tele Γ n) (b : Expr (Γ + n)) (env : Fin Γ → R)
    (q : Fin (Γ + n)),
    adjSem t b env q
      = (revIdx n).foldl (adjStep t b env q)
          (denote (t.env env) (sderiv b q)) := by
  intro n t
  induction t with
  | nil => intro b env q; rfl
  | @cons n t e ih =>
      intro b env q
      rw [revIdx_succ, List.foldl_cons]
      by_cases hq : q.val < Γ + n
      -- The slot is one the remaining telescope can reach: recurse.
      · -- The head step of the fold *is* the `letE` chain rule.
        have hwke : denote ((Tele.cons t e).env env)
              (sderiv ((Tele.cons t e).getW ⟨n, Nat.lt_succ_self n⟩) q)
            = denote (t.env env) (sderiv e ⟨q.val, hq⟩) := by
          rw [Tele.getW_cons_last t e ⟨n, Nat.lt_succ_self n⟩ (Nat.lt_irrefl n)]
          exact denote_sderiv_wk e ⟨q.val, hq⟩ (t.env env) (denote (t.env env) e)
        have hRHS : denote (t.env env) (sderiv (Expr.letE e b) ⟨q.val, hq⟩)
            = add (denote ((Tele.cons t e).env env) (sderiv b q))
                  (mul (denote ((Tele.cons t e).env env)
                          (sderiv b ⟨Γ + n, Nat.lt_succ_self (Γ + n)⟩))
                       (denote (t.env env) (sderiv e ⟨q.val, hq⟩))) := by
          show denote (extend (t.env env) (denote (t.env env) e))
              (Expr.add (sderiv b _) (.mul (sderiv b _) (wk (sderiv e ⟨q.val, hq⟩)))) = _
          rw [denote_add, denote_mul, denote_wk]
          rfl
        have hhead : adjStep (Tele.cons t e) b env q
              (denote ((Tele.cons t e).env env) (sderiv b q)) n
            = denote (t.env env) (sderiv (Expr.letE e b) ⟨q.val, hq⟩) := by
          show (if h : n < n + 1 then _ else _) = _
          rw [dif_pos (Nat.lt_succ_self n), if_pos hq, hwke,
              adjSem_cons_last t e b env ⟨Γ + n, Nat.lt_succ_self (Γ + n)⟩
                (Nat.lt_irrefl (Γ + n)), hRHS]
        rw [hhead, adjSem_cons_lt t e b env q hq, ih (Expr.letE e b) env ⟨q.val, hq⟩]
        refine foldl_congr_mem _ _ _ _ ?_
        intro acc i hi
        have hin : i < n := mem_revIdx hi
        have hw : denote ((Tele.cons t e).env env)
              (sderiv ((Tele.cons t e).getW ⟨i, Nat.lt_succ_of_lt hin⟩) q)
            = denote (t.env env) (sderiv (t.getW ⟨i, hin⟩) ⟨q.val, hq⟩) := by
          rw [Tele.getW_cons_lt t e ⟨i, Nat.lt_succ_of_lt hin⟩ hin]
          exact denote_sderiv_wk (t.getW ⟨i, hin⟩) ⟨q.val, hq⟩ (t.env env)
            (denote (t.env env) e)
        show (if h : i < n then _ else _) = (if h : i < n + 1 then _ else _)
        rw [dif_pos hin, dif_pos (Nat.lt_succ_of_lt hin)]
        by_cases hqi : q.val < Γ + i
        · rw [if_pos (show q.val < Γ + i from hqi), if_pos hqi, hw,
              adjSem_cons_lt t e b env ⟨Γ + i, by omega⟩ (show Γ + i < Γ + n by omega)]
        · rw [if_neg hqi, if_neg hqi]
      -- The slot is the one this binding introduces: nothing later touches it.
      · rw [adjSem_cons_last t e b env q hq]
        have hstep : ∀ (acc : R) (i : Nat), i ∈ revIdx n →
            adjStep (Tele.cons t e) b env q acc i = acc := by
          intro acc i hi
          have hin : i < n := mem_revIdx hi
          have hqv := q.isLt
          show (if h : i < n + 1 then _ else _) = acc
          rw [dif_pos (Nat.lt_succ_of_lt hin), if_neg (show ¬ q.val < Γ + i by omega)]
        have hhead : adjStep (Tele.cons t e) b env q
              (denote ((Tele.cons t e).env env) (sderiv b q)) n
            = denote ((Tele.cons t e).env env) (sderiv b q) := by
          show (if h : n < n + 1 then _ else _) = _
          rw [dif_pos (Nat.lt_succ_self n), if_neg hq]
        rw [hhead]
        exact (foldl_const _ _ _ hstep).symm

-- ---------------------------------------------------------------------------
-- 3. The program computes it
-- ---------------------------------------------------------------------------

/-- The first `j` terms of the reverse accumulation, semantically. -/
def adjPartial {n : Nat} (t : Tele Γ n) (b : Expr (Γ + n)) (env : Fin Γ → R)
    (q : Fin (Γ + n)) (j : Nat) : R :=
  (revTake n j).foldl (adjStep t b env q) (denote (t.env env) (sderiv b q))

/-- A telescope leaves the context it was built over alone. -/
theorem Tele.env_base {Δ : Nat} : ∀ {m : Nat} (u : Tele Δ m) (ρ : Fin Δ → R) (i : Fin Δ),
    u.env ρ ⟨i.val, by omega⟩ = ρ i := by
  intro m u
  induction u with
  | nil => intro ρ i; rfl
  | cons u e ih =>
      intro ρ i
      rw [Tele.env_cons_lt u e ρ ⟨i.val, by omega⟩]
      exact ih ρ i

/-- **The sweep saturates.**  Once the accumulation has passed binding
    `n-1-j`, no later term can touch slot `q` — binding `i` simply does not
    mention a slot bound after it.  So the `j`-step partial sum is already the
    full adjoint, which is what makes it sound to *bind* it at step `j`. -/
theorem adjSem_saturates {n : Nat} (t : Tele Γ n) (b : Expr (Γ + n)) (env : Fin Γ → R)
    (q : Fin (Γ + n)) (j : Nat) (hj : j ≤ n) (hq : ∀ i, i < n - j → Γ + i ≤ q.val) :
    adjSem t b env q = adjPartial t b env q j := by
  rw [adjSem_fold t b env q, ← revTake_append j hj, List.foldl_append]
  refine foldl_const _ _ _ ?_
  intro acc i hi
  have hin : i < n - j := mem_revIdx hi
  have hle := hq i hin
  show (if h : i < n then _ else _) = acc
  rw [dif_pos (show i < n by omega), if_neg (show ¬ q.val < Γ + i by omega)]

/-- **The syntax denotes the semantics.**  `adjAt`'s fold, read in an
    environment that already holds the earlier adjoints, is the semantic
    accumulation — variable references and all. -/
theorem denote_adjFoldE {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (env : Fin Γ → R)
    (q : Fin (Γ + n)) (J : Nat) (hJ : J ≤ n) (bigEnv : Fin ((Γ + n) + J) → R)
    (hbase : ∀ i : Fin (Γ + n), bigEnv ⟨i.val, by omega⟩ = t.env env i)
    (hadj : ∀ (j' : Nat) (hj' : j' < J), bigEnv ⟨(Γ + n) + j', by omega⟩
              = adjSem t out env ⟨Γ + (n - 1 - j'), by omega⟩) :
    ∀ (j : Nat), j ≤ J → denote bigEnv (adjFoldE t out q J hJ j) = adjPartial t out env q j := by
  have hrestrict : (fun i : Fin (Γ + n) => bigEnv ⟨i.val, by omega⟩) = t.env env :=
    funext hbase
  intro j
  induction j with
  | zero =>
      intro _
      show denote bigEnv (wkTo (by omega) (sderiv out q)) = _
      rw [denote_wkTo, hrestrict]
      rfl
  | succ j ih =>
      intro hjJ
      have hj : j < J := by omega
      have hidx : n - 1 - (n - 1 - j) = j := by omega
      have hpart : adjPartial t out env q (j + 1)
          = adjStep t out env q (adjPartial t out env q j) (n - 1 - j) := by
        show (revTake n (j + 1)).foldl _ _ = _
        simp only [revTake, List.foldl_append, List.foldl_cons, List.foldl_nil]
        rfl
      show denote bigEnv ((revTake n (j + 1)).foldl _ _) = _
      simp only [revTake, List.foldl_append, List.foldl_cons, List.foldl_nil]
      rw [dif_pos (show n - 1 - j < n ∧ n - 1 - (n - 1 - j) < J by omega)]
      rw [hpart]
      show _ = (if h : n - 1 - j < n then _ else _)
      rw [dif_pos (show n - 1 - j < n by omega)]
      by_cases hqi : q.val < Γ + (n - 1 - j)
      · rw [if_pos hqi, if_pos hqi]
        show add (denote bigEnv (adjFoldE t out q J hJ j))
              (mul (bigEnv ⟨(Γ + n) + (n - 1 - (n - 1 - j)), by omega⟩)
                   (denote bigEnv (wkTo (by omega)
                      (sderiv (t.getW ⟨n - 1 - j, show n - 1 - j < n by omega⟩) q)))) = _
        rw [ih (by omega), denote_wkTo, hrestrict]
        simp only [hidx]
        rw [hadj j hj]
      · rw [if_neg hqi, if_neg hqi]
        exact ih (by omega)

theorem adjTele_succ {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (j : Nat)
    (hj : j + 1 ≤ n) :
    adjTele t out (j + 1) hj
      = (adjTele t out j (by omega)).cons
          (adjAt t out ⟨Γ + (n - 1 - j), by omega⟩ j (by omega)) := rfl

/-- **The adjoint telescope binds the adjoints.**  Slot `(Γ+n)+j'` really does
    hold the adjoint of forward slot `Γ + (n-1-j')`. -/
theorem adjTele_env {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (env : Fin Γ → R) :
    ∀ (j : Nat) (hj : j ≤ n) (j' : Nat) (hj' : j' < j),
      (adjTele t out j hj).env (t.env env) ⟨(Γ + n) + j', by omega⟩
        = adjSem t out env ⟨Γ + (n - 1 - j'), by omega⟩ := by
  intro j
  induction j with
  | zero => intro _ j' hj'; omega
  | succ j ih =>
      intro hj j' hj'
      have hjn : j ≤ n := by omega
      rw [adjTele_succ t out j hj]
      by_cases hlt : j' < j
      · rw [show (⟨(Γ + n) + j', by omega⟩ : Fin ((Γ + n) + (j + 1)))
              = ⟨(⟨(Γ + n) + j', by omega⟩ : Fin ((Γ + n) + j)).val, by omega⟩ from rfl,
            Tele.env_cons_lt (adjTele t out j hjn) _ (t.env env)]
        exact ih hjn j' hlt
      · have hje : j' = j := by omega
        subst hje
        rw [show (⟨(Γ + n) + j', by omega⟩ : Fin ((Γ + n) + (j' + 1)))
              = ⟨(Γ + n) + j', Nat.lt_succ_self _⟩ from rfl,
            Tele.env_cons_last (adjTele t out j' hjn) _ (t.env env)]
        rw [show adjAt t out ⟨Γ + (n - 1 - j'), by omega⟩ j' (by omega)
              = adjFoldE t out ⟨Γ + (n - 1 - j'), by omega⟩ j' (by omega) j' from rfl,
            denote_adjFoldE t out env _ j' hjn _
              (Tele.env_base (adjTele t out j' hjn) (t.env env))
              (fun j'' hj'' => ih hjn j'' hj'') j' (Nat.le_refl j')]
        exact (adjSem_saturates t out env ⟨Γ + (n - 1 - j'), by omega⟩ j' hjn
          (fun i hi => show Γ + i ≤ Γ + (n - 1 - j') by omega)).symm

/-- ## `gradProg` is correct

    Output `k` of the gradient program denotes exactly `∂out/∂xₖ` as `sderiv`
    defines it — the same specification `grad_correct` is stated against.
    So the sharing-preserving representation that made the gradient quadratic
    instead of exponential computes the same numbers as the naive one.

    No `NumLaws`: this is an equality in **any** `NumOps` carrier, `Float32`
    included. -/
theorem gradProg_correct {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n))
    (env : Fin Γ → R) (k : Fin Γ) :
    denote env ((gradProg t out).get k) = denote env (sderiv (t.bind out) k) := by
  have hbase : ∀ i : Fin (Γ + n),
      (adjTele t out n (Nat.le_refl n)).env (t.env env) ⟨i.val, by omega⟩ = t.env env i :=
    Tele.env_base (adjTele t out n (Nat.le_refl n)) (t.env env)
  show denote env ((t.append (adjTele t out n (Nat.le_refl n))).bind
      (castE (by omega) (adjAt t out ⟨k.val, by omega⟩ n (Nat.le_refl n)))) = _
  rw [denote_bind, denote_castE]
  have henv : (fun i : Fin ((Γ + n) + n) =>
      (t.append (adjTele t out n (Nat.le_refl n))).env env ⟨i.val, by omega⟩)
      = (adjTele t out n (Nat.le_refl n)).env (t.env env) := by
    funext i
    rw [Tele.env_append t (adjTele t out n (Nat.le_refl n)) env ⟨i.val, by omega⟩]
  rw [henv, show adjAt t out ⟨k.val, by omega⟩ n (Nat.le_refl n)
        = adjFoldE t out ⟨k.val, by omega⟩ n (Nat.le_refl n) n from rfl,
      denote_adjFoldE t out env _ n (Nat.le_refl n) _ hbase
        (fun j' hj' => adjTele_env t out env n (Nat.le_refl n) j' hj') n (Nat.le_refl n)]
  rw [← adjSem_saturates t out env ⟨k.val, by omega⟩ n (Nat.le_refl n)
        (fun i hi => absurd hi (by omega))]
  exact adjSem_spec t out env k

-- ---------------------------------------------------------------------------
-- 4. Narrowing the sweep — and exactly what it costs
-- ---------------------------------------------------------------------------

/-! `gradProg` is quadratic because slot `q`'s adjoint sums over *every* later
    binding.  In a layered model most of those bindings never mention `q`, so
    most terms are zero and the sum should range over one block, not the whole
    program — `O(n·d)` instead of `O(n²)`, a factor of `L` (the layer count).

    **That narrowing is not free, and the reason is worth stating precisely.**
    `sderiv`'s rule for a binding,

        d/dxₖ (let v = e in b) = ∂b/∂xₖ + (∂b/∂v)·(de/dxₖ)

    emits the second term for *every* `q` in scope, whether or not `e` mentions
    it.  So the zero terms are in the **specification**, not merely in our
    implementation of it — `gradProg` is already optimal against `sderiv`.
    Dropping them is a change of spec, and it needs declaring.

    What it needs is exactly one proposition, `ZeroTermFree`, below.  It holds
    for ℝ and for `Int`.  For `Float32` it fails in exactly two places, both
    nameable:

    * `w` non-finite — `inf · 0 = NaN`, and `acc + NaN = NaN`;
    * `acc` is negative zero — `-0 + (+0) = +0`, so the sign of a zero flips.

    For finite weights the result differs from `sderiv` in *at most the sign bit
    of an exact zero*.  That is a far tighter statement than `ExpIsEx2`, which
    carries a measured 5.06e-7 relative error, and it is the honest price of
    linearity.  See `AlgorithmLib.ML.Assumptions`. -/

/-- **Declared.**  A term whose derivative is zero can be dropped from the
    accumulation.  True for ℝ and `Int`; for `Float32` true whenever `w` is
    finite and `acc` is not negative zero. -/
def ZeroTermFree (R : Type) [NumOps R] : Prop :=
  ∀ acc w : R, add acc (mul w (ofNat 0)) = acc


-- ---------------------------------------------------------------------------
-- 5. The narrowed program, and what it costs
-- ---------------------------------------------------------------------------

/-! `gradProgD` emits only the terms inside each binding's read window, so for a
    layered model it is `O(n·d)` rather than `O(n²)` — a factor of `L`, the
    layer count, which at 93 layers is the difference between elaborating and
    not.

    Its correctness carries `ZeroTermFree`, and that is not an artefact: the
    dropped terms are present in `sderiv` itself, so narrowing is a change of
    *specification*.  `gradProg`, which drops nothing, needs no such
    hypothesis — the two theorems sit side by side so the price is visible. -/

theorem denote_adjFoldED {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (env : Fin Γ → R)
    (q : Fin (Γ + n)) (dep : Nat → Nat) (J : Nat) (hJ : J ≤ n)
    (bigEnv : Fin ((Γ + n) + J) → R) (hz : ZeroTermFree R)
    (hdep : ∀ (i : Nat) (h : i < n), q.val < dep i →
      denote (t.env env) (sderiv (t.getW ⟨i, h⟩) q) = ofNat 0)
    (hbase : ∀ i : Fin (Γ + n), bigEnv ⟨i.val, by omega⟩ = t.env env i)
    (hadj : ∀ (j' : Nat) (hj' : j' < J), bigEnv ⟨(Γ + n) + j', by omega⟩
              = adjSem t out env ⟨Γ + (n - 1 - j'), by omega⟩) :
    ∀ (j : Nat), j ≤ J →
      denote bigEnv (adjFoldED t out q dep J hJ j) = adjPartial t out env q j := by
  have hrestrict : (fun i : Fin (Γ + n) => bigEnv ⟨i.val, by omega⟩) = t.env env :=
    funext hbase
  intro j
  induction j with
  | zero =>
      intro _
      show denote bigEnv (wkTo (by omega) (sderiv out q)) = _
      rw [denote_wkTo, hrestrict]
      rfl
  | succ j ih =>
      intro hjJ
      have hj : j < J := by omega
      have hidx : n - 1 - (n - 1 - j) = j := by omega
      have hpart : adjPartial t out env q (j + 1)
          = adjStep t out env q (adjPartial t out env q j) (n - 1 - j) := by
        show (revTake n (j + 1)).foldl _ _ = _
        simp only [revTake, List.foldl_append, List.foldl_cons, List.foldl_nil]
        rfl
      show denote bigEnv ((revTake n (j + 1)).foldl _ _) = _
      simp only [revTake, List.foldl_append, List.foldl_cons, List.foldl_nil]
      rw [dif_pos (show n - 1 - j < n ∧ n - 1 - (n - 1 - j) < J by omega)]
      rw [hpart]
      show _ = (if h : n - 1 - j < n then _ else _)
      rw [dif_pos (show n - 1 - j < n by omega)]
      by_cases hqi : q.val < Γ + (n - 1 - j)
      · by_cases hdq : dep (n - 1 - j) ≤ q.val
        · -- inside the window: the term is emitted, exactly as before
          rw [if_pos ⟨hdq, hqi⟩, if_pos hqi]
          show add (denote bigEnv (adjFoldED t out q dep J hJ j))
                (mul (bigEnv ⟨(Γ + n) + (n - 1 - (n - 1 - j)), by omega⟩)
                     (denote bigEnv (wkTo (by omega)
                        (sderiv (t.getW ⟨n - 1 - j, show n - 1 - j < n by omega⟩) q)))) = _
          rw [ih (by omega), denote_wkTo, hrestrict]
          simp only [hidx]
          rw [hadj j hj]
        · -- outside it: the term is dropped, and it was zero
          rw [if_neg (fun hc => hdq hc.1), if_pos hqi]
          show denote bigEnv (adjFoldED t out q dep J hJ j) = _
          rw [ih (by omega), hdep (n - 1 - j) (by omega) (by omega)]
          exact (hz (adjPartial t out env q j) _).symm
      · rw [if_neg (fun hc => hqi hc.2), if_neg hqi]
        show denote bigEnv (adjFoldED t out q dep J hJ j) = _
        exact ih (by omega)

theorem adjTeleD_succ {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (dep : Nat → Nat)
    (j : Nat) (hj : j + 1 ≤ n) :
    adjTeleD t out dep (j + 1) hj
      = (adjTeleD t out dep j (by omega)).cons
          (adjAtD t out ⟨Γ + (n - 1 - j), by omega⟩ dep j (by omega)) := rfl

theorem adjTeleD_env {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (env : Fin Γ → R)
    (dep : Nat → Nat) (hz : ZeroTermFree R)
    (hdep : ∀ (i : Nat) (h : i < n) (q : Fin (Γ + n)), q.val < dep i →
      denote (t.env env) (sderiv (t.getW ⟨i, h⟩) q) = ofNat 0) :
    ∀ (j : Nat) (hj : j ≤ n) (j' : Nat) (hj' : j' < j),
      (adjTeleD t out dep j hj).env (t.env env) ⟨(Γ + n) + j', by omega⟩
        = adjSem t out env ⟨Γ + (n - 1 - j'), by omega⟩ := by
  intro j
  induction j with
  | zero => intro _ j' hj'; omega
  | succ j ih =>
      intro hj j' hj'
      have hjn : j ≤ n := by omega
      rw [adjTeleD_succ t out dep j hj]
      by_cases hlt : j' < j
      · rw [show (⟨(Γ + n) + j', by omega⟩ : Fin ((Γ + n) + (j + 1)))
              = ⟨(⟨(Γ + n) + j', by omega⟩ : Fin ((Γ + n) + j)).val, by omega⟩ from rfl,
            Tele.env_cons_lt (adjTeleD t out dep j hjn) _ (t.env env)]
        exact ih hjn j' hlt
      · have hje : j' = j := by omega
        subst hje
        rw [show (⟨(Γ + n) + j', by omega⟩ : Fin ((Γ + n) + (j' + 1)))
              = ⟨(Γ + n) + j', Nat.lt_succ_self _⟩ from rfl,
            Tele.env_cons_last (adjTeleD t out dep j' hjn) _ (t.env env)]
        rw [show adjAtD t out ⟨Γ + (n - 1 - j'), by omega⟩ dep j' (by omega)
              = adjFoldED t out ⟨Γ + (n - 1 - j'), by omega⟩ dep j' (by omega) j' from rfl,
            denote_adjFoldED t out env _ dep j' hjn _ hz
              (fun i h hq => hdep i h _ hq)
              (Tele.env_base (adjTeleD t out dep j' hjn) (t.env env))
              (fun j'' hj'' => ih hjn j'' hj'') j' (Nat.le_refl j')]
        exact (adjSem_saturates t out env ⟨Γ + (n - 1 - j'), by omega⟩ j' hjn
          (fun i hi => show Γ + i ≤ Γ + (n - 1 - j') by omega)).symm

/-- **The narrowed gradient is correct**, under the one declared proposition and
    per-binding evidence of what each binding reads.

    Compare `gradProg_correct`, which needs neither: that is the exact price of
    linearity, and it is a `Float32` sign-of-zero and non-finite-weight
    question, not an epsilon. -/
theorem gradProgD_correct {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n))
    (env : Fin Γ → R) (dep : Nat → Nat) (hz : ZeroTermFree R)
    (hdep : ∀ (i : Nat) (h : i < n) (q : Fin (Γ + n)), q.val < dep i →
      denote (t.env env) (sderiv (t.getW ⟨i, h⟩) q) = ofNat 0)
    (k : Fin Γ) :
    denote env ((gradProgD t out dep).get k) = denote env (sderiv (t.bind out) k) := by
  have hbase : ∀ i : Fin (Γ + n),
      (adjTeleD t out dep n (Nat.le_refl n)).env (t.env env) ⟨i.val, by omega⟩ = t.env env i :=
    Tele.env_base (adjTeleD t out dep n (Nat.le_refl n)) (t.env env)
  show denote env ((t.append (adjTeleD t out dep n (Nat.le_refl n))).bind
      (castE (by omega) (adjAtD t out ⟨k.val, by omega⟩ dep n (Nat.le_refl n)))) = _
  rw [denote_bind, denote_castE]
  have henv : (fun i : Fin ((Γ + n) + n) =>
      (t.append (adjTeleD t out dep n (Nat.le_refl n))).env env ⟨i.val, by omega⟩)
      = (adjTeleD t out dep n (Nat.le_refl n)).env (t.env env) := by
    funext i
    rw [Tele.env_append t (adjTeleD t out dep n (Nat.le_refl n)) env ⟨i.val, by omega⟩]
  rw [henv, show adjAtD t out ⟨k.val, by omega⟩ dep n (Nat.le_refl n)
        = adjFoldED t out ⟨k.val, by omega⟩ dep n (Nat.le_refl n) n from rfl,
      denote_adjFoldED t out env _ dep n (Nat.le_refl n) _ hz
        (fun i h hq => hdep i h _ hq) hbase
        (fun j' hj' => adjTeleD_env t out env dep hz hdep n (Nat.le_refl n) j' hj')
        n (Nat.le_refl n)]
  rw [← adjSem_saturates t out env ⟨k.val, by omega⟩ n (Nat.le_refl n)
        (fun i hi => absurd hi (by omega))]
  exact adjSem_spec t out env k

-- ---------------------------------------------------------------------------
-- 6. Discharging the window evidence
-- ---------------------------------------------------------------------------

/-! `gradProgD_correct` needs, for each binding, that the partials below its
    window denote `ofNat 0`.  For a variable that genuinely does not occur that
    is *morally* obvious — but it is not derivable from `NumOps`, because
    `sderiv` leaves a **tree** of zeros (`add (sderiv a q) (sderiv b q)`) and
    `add (ofNat 0) (ofNat 0) = ofNat 0` is not one of the operations' laws.

    So the practical route needs a second named proposition, `ZeroLaws`, and
    with it the evidence follows from a purely **syntactic** occurrence check.
    That is the difference between a theorem that is provable in principle and
    one a user can actually discharge.

    `ZeroLaws` holds for ℝ and for `Int`.  For `Float32` the additive facts hold
    for `+0` and fail only at signed zero — the same edge `ZeroTermFree` already
    names, which is why the two travel together. -/

/-- **Declared.**  The zero facts a non-occurrence argument needs. -/
structure ZeroLaws (R : Type) [NumOps R] : Prop where
  add_zz : add (ofNat 0 : R) (ofNat 0) = ofNat 0
  mul_lz : ∀ x : R, mul (ofNat 0) x = ofNat 0
  mul_rz : ∀ x : R, mul x (ofNat 0) = ofNat 0
  neg_z  : neg (ofNat 0 : R) = ofNat 0
  sum_z  : ∀ (n : Nat), (List.finRange n).foldl
             (fun acc (_ : Fin n) => add acc (ofNat 0 : R)) zero = ofNat 0

/-- `q` does not occur in `e`. -/
def NotUses : {Δ : Nat} → Fin Δ → Expr Δ → Prop
  | _, q, .var i    => q ≠ i
  | _, _, .lit _    => True
  | _, q, .add a b  => NotUses q a ∧ NotUses q b
  | _, q, .mul a b  => NotUses q a ∧ NotUses q b
  | _, q, .neg a    => NotUses q a
  | _, q, .inv a    => NotUses q a
  | _, q, .exp a    => NotUses q a
  | _, q, .rsqrt a  => NotUses q a
  | _, q, .sum _ f  => ∀ j, NotUses q (f j)
  | Δ, q, .letE a b => NotUses q a ∧ NotUses ⟨q.val, by omega⟩ b

instance decNotUses : ∀ {Δ : Nat} (q : Fin Δ) (e : Expr Δ), Decidable (NotUses q e)
  | _, q, .var i    => inferInstanceAs (Decidable (q ≠ i))
  | _, _, .lit _    => .isTrue trivial
  | _, q, .add a b  => @instDecidableAnd _ _ (decNotUses q a) (decNotUses q b)
  | _, q, .mul a b  => @instDecidableAnd _ _ (decNotUses q a) (decNotUses q b)
  | _, q, .neg a    => decNotUses q a
  | _, q, .inv a    => decNotUses q a
  | _, q, .exp a    => decNotUses q a
  | _, q, .rsqrt a  => decNotUses q a
  | _, q, .sum n f  => @Nat.decidableForallFin n _ (fun j => decNotUses q (f j))
  | Δ, q, .letE a b =>
      @instDecidableAnd _ _ (decNotUses q a) (decNotUses ⟨q.val, by omega⟩ b)

/-- **A variable that does not occur has a zero partial.**

    This is what makes `gradProgD` usable: the window evidence reduces to a
    syntactic check on the binding, decidable and therefore dischargeable by
    `decide`, rather than a hand proof about denotations. -/
theorem sderiv_notUses (hzl : ZeroLaws R) :
    ∀ {Δ : Nat} (e : Expr Δ) (q : Fin Δ) (env : Fin Δ → R),
      NotUses q e → denote env (sderiv e q) = ofNat 0 := by
  intro Δ e
  induction e with
  | var i =>
      intro q env h
      show denote env (if _ then _ else _) = _
      rw [if_neg h]
      rfl
  | lit n => intro q env _; rfl
  | add a b iha ihb =>
      intro q env h
      show add _ _ = _
      rw [iha q env h.1, ihb q env h.2, hzl.add_zz]
  | mul a b iha ihb =>
      intro q env h
      show add (mul _ _) (mul _ _) = _
      rw [iha q env h.1, ihb q env h.2, hzl.mul_lz, hzl.mul_rz, hzl.add_zz]
  | neg a ih =>
      intro q env h
      show neg _ = _
      rw [ih q env h, hzl.neg_z]
  | inv a ih =>
      intro q env h
      show neg (mul _ _) = _
      rw [ih q env h, hzl.mul_lz, hzl.neg_z]
  | exp a ih =>
      intro q env h
      show mul _ _ = _
      rw [ih q env h, hzl.mul_lz]
  | rsqrt a ih =>
      intro q env h
      show neg (mul _ _) = _
      rw [ih q env h, hzl.mul_lz, hzl.neg_z]
  | sum n f ih =>
      intro q env h
      show (List.finRange n).foldl (fun acc j => add acc (denote env (sderiv (f j) q))) zero = _
      have hz : ∀ j, denote env (sderiv (f j) q) = ofNat 0 := fun j => ih j q env (h j)
      simp only [hz]
      exact hzl.sum_z n
  | letE a b iha ihb =>
      intro q env h
      show denote (extend env (denote env a)) (Expr.add _ (.mul _ _)) = _
      show add (denote _ (sderiv b _)) (mul (denote _ (sderiv b _)) (denote _ (wk _))) = _
      rw [ihb ⟨q.val, by omega⟩ _ h.2, denote_wk, iha q env h.1,
          hzl.mul_rz, hzl.add_zz]

end AlgorithmLib.ML
