import AlgorithmLib.ML.MultiLayer

/-!
  # Let-normal programs, and why the gradient needs them

  `MultiLayer.bindVec` binds a *vector* of values whose definitions all live in
  the same context.  That is enough for a layer stack's forward pass, where each
  layer's outputs depend only on the previous layer's.  It is **not** enough for
  a gradient, where each new binding may refer to every binding before it.

  `Tele Γ n` is that: a telescope of `n` bindings, the `i`-th an
  `Expr (Γ + i)`, so it may mention all of its predecessors.  `Tele.bind` turns
  a telescope plus a body into a single `Expr Γ` — a chain of `letE`s — and
  `denote_bind` proves the obvious thing about it.

  ## Why this is the shape the gradient must take

  Reverse mode over a shared term wants to *name* each intermediate adjoint and
  refer to it by variable.  With the current type

      grad : Expr Γ → (Fin Γ → Expr Γ)

  it cannot: the `Γ` adjoint expressions are independent terms with no way to
  share a subterm.  Working the sizes through, even a one-pass reverse rule
  gives

      grad (letE a b) k = letE a (add (g k) (mul (g last) (wk (gradA k))))

  and `g last` — the bound variable's adjoint — is copied into every one of the
  `Γ` outputs.  Summing over `k`, the total still doubles per binding.  That is
  measured: `2^d` growth per layer at `d` binders per layer.

  So the fix is not a better rule, it is a better *output type*: one program,
  with the adjoints bound and referred to by variable.  `Tele` is that type, and
  `VProg` — one telescope, many outputs — is the gradient's shape.

  Concatenating two telescopes needs a context cast (`Γ + (n+m)` is not
  definitionally `(Γ+n) + m`), so program *composition* is left for the same
  pass that builds the reverse sweep.
-/

namespace AlgorithmLib.ML

variable {R : Type} [NumOps R] {Γ : Nat}

/-- A telescope of `n` bindings over context `Γ`.  Binding `i` is an
    `Expr (Γ + i)`, so it may mention every earlier binding — which is exactly
    what `bindVec` could not express. -/
inductive Tele (Γ : Nat) : Nat → Type where
  | nil  : Tele Γ 0
  | cons : {n : Nat} → Tele Γ n → Expr (Γ + n) → Tele Γ (n + 1)

/-- Wrap a body in the telescope's `letE` chain. -/
def Tele.bind : {n : Nat} → Tele Γ n → Expr (Γ + n) → Expr Γ
  | 0,     .nil,      b => b
  | _ + 1, .cons t e, b => t.bind (.letE e b)

/-- The environment a telescope builds: each slot holds its binding's value,
    evaluated once. -/
def Tele.env : {n : Nat} → Tele Γ n → (Fin Γ → R) → (Fin (Γ + n) → R)
  | 0,     .nil,      env => env
  | _ + 1, .cons t e, env => extend (t.env env) (denote (t.env env) e)

/-- **A telescope means what it looks like it means.**

    Each binding is evaluated once, in the environment its predecessors built —
    the defining property that makes this a sharing-preserving representation
    rather than a notation for substitution. -/
theorem denote_bind : ∀ {n : Nat} (t : Tele Γ n) (b : Expr (Γ + n)) (env : Fin Γ → R),
    denote env (t.bind b) = denote (t.env env) b := by
  intro n t
  induction t with
  | nil => intro b env; rfl
  | cons t e ih =>
      intro b env
      show denote env (t.bind (.letE e b)) = _
      rw [ih (.letE e b) env]
      rfl

/-- Slot `Γ + i` of a telescope, as a variable. -/
def Tele.slot (Γ : Nat) {n : Nat} (i : Fin n) : Expr (Γ + n) :=
  .var ⟨Γ + i.val, by omega⟩

/-- Appending a binding does not disturb the earlier slots. -/
theorem Tele.env_cons_lt {n : Nat} (t : Tele Γ n) (e : Expr (Γ + n))
    (env : Fin Γ → R) (k : Fin (Γ + n)) :
    (Tele.cons t e).env env ⟨k.val, by omega⟩ = t.env env k := by
  show extend (t.env env) _ ⟨k.val, by omega⟩ = _
  rw [extend_lt _ _ ⟨k.val, by omega⟩ k.isLt]

/-- The last slot holds the last binding's value. -/
theorem Tele.env_cons_last {n : Nat} (t : Tele Γ n) (e : Expr (Γ + n))
    (env : Fin Γ → R) :
    (Tele.cons t e).env env ⟨Γ + n, by omega⟩ = denote (t.env env) e := by
  show extend (t.env env) _ ⟨Γ + n, by omega⟩ = _
  exact extend_last (t.env env) _

-- ---------------------------------------------------------------------------
-- `bindVec` is a telescope
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- Context casts and composition
-- ---------------------------------------------------------------------------

/-- Move an expression to a context of the same size.  Needed because
    `(Γ + n) + m` is not *definitionally* `Γ + (n + m)`, even though it is
    equal — the price of indexing contexts by `Nat`.  It is a `rename`, so it is
    meaning-preserving by `denote_rename`. -/
def castE {Γ' Δ : Nat} (h : Γ' = Δ) (e : Expr Γ') : Expr Δ :=
  rename (fun i => ⟨i.val, by omega⟩) e

/-- Weaken into any larger context. -/
def wkTo {Γ' Δ : Nat} (h : Γ' ≤ Δ) (e : Expr Γ') : Expr Δ :=
  rename (fun i => ⟨i.val, by omega⟩) e

@[simp] theorem denote_castE {Γ' Δ : Nat} (h : Γ' = Δ) (e : Expr Γ') (env : Fin Δ → R) :
    denote env (castE h e) = denote (fun i => env ⟨i.val, by omega⟩) e :=
  denote_rename _ e env

@[simp] theorem denote_wkTo {Γ' Δ : Nat} (h : Γ' ≤ Δ) (e : Expr Γ') (env : Fin Δ → R) :
    denote env (wkTo h e) = denote (fun i => env ⟨i.val, by omega⟩) e :=
  denote_rename _ e env

/-- Append a telescope built over the extended context.  This is what makes
    programs compose: the second block may use everything the first bound. -/
def Tele.append : {n m : Nat} → Tele Γ n → Tele (Γ + n) m → Tele Γ (n + m)
  | _, 0,     t, .nil      => t
  | _, _ + 1, t, .cons u e => (t.append u).cons (castE (by omega) e)

/-- **The environments compose.**  Running the concatenated telescope builds the
    same bindings as running one then the other. -/
theorem Tele.env_append : ∀ {n m : Nat} (t : Tele Γ n) (u : Tele (Γ + n) m)
    (env : Fin Γ → R) (k : Fin (Γ + (n + m))),
    (t.append u).env env k = u.env (t.env env) ⟨k.val, by omega⟩ := by
  intro n m
  induction m with
  | zero =>
      intro t u env k
      cases u
      show (t.env env) k = (t.env env) ⟨k.val, by omega⟩
      congr 1
  | succ m ih =>
      intro t u env k
      cases u with
      | cons u e =>
          have hEq : ∀ (i : Fin (Γ + n + m)),
              (t.append u).env env ⟨i.val, by omega⟩ = u.env (t.env env) i := by
            intro i
            rw [ih t u env ⟨i.val, by omega⟩]
          have hlast : denote ((t.append u).env env) (castE (by omega) e)
              = denote (u.env (t.env env)) e := by
            rw [denote_castE]
            congr 1
            funext i
            exact hEq i
          show extend ((t.append u).env env)
              (denote ((t.append u).env env) (castE (by omega) e)) k
            = extend (u.env (t.env env)) (denote (u.env (t.env env)) e)
                ⟨k.val, by omega⟩
          rw [hlast]
          by_cases hk : k.val < Γ + n + m
          · rw [extend_lt ((t.append u).env env) _ k (show k.val < Γ + (n + m) by omega),
                extend_lt (u.env (t.env env)) _ (⟨k.val, by omega⟩ : Fin (Γ + n + m + 1)) hk]
            exact hEq ⟨k.val, hk⟩
          · rw [extend_ge ((t.append u).env env) _ k (show ¬ (k.val < Γ + (n + m)) by omega),
                extend_ge (u.env (t.env env)) _ (⟨k.val, by omega⟩ : Fin (Γ + n + m + 1)) hk]

/-- Random access into a telescope: binding `i`, in the context it was written
    in. -/
def Tele.get : {n : Nat} → Tele Γ n → (i : Fin n) → Expr (Γ + i.val)
  | 0,     .nil,      i => absurd i.isLt (by omega)
  | n + 1, .cons t e, i =>
      if h : i.val < n then Tele.get t ⟨i.val, h⟩
      else castE (by have := i.isLt; omega) e

/-- Binding `i`, weakened into the telescope's *full* context.

    `Tele.get` returns the binding in the context it was written in, which then
    needs a cast at every use.  `getW` weakens instead — and weakens by exactly
    one `wk` per `cons`, so an induction on the telescope meets `sderiv`'s
    interaction with `wk` (`sderiv_wk`) rather than with an arbitrary cast.
    That single choice is what keeps the correctness proof free of transport. -/
def Tele.getW : {n : Nat} → Tele Γ n → (i : Fin n) → Expr (Γ + n)
  | 0,     .nil,      i => absurd i.isLt (by omega)
  | n + 1, .cons t e, i => if h : i.val < n then wk (t.getW ⟨i.val, h⟩) else wk e

-- ---------------------------------------------------------------------------
-- A program in let-normal form
-- ---------------------------------------------------------------------------

/-- A program: shared bindings plus a result.  This is what a model *is* once
    its sharing is explicit, and — per the argument above — what a gradient must
    also be. -/
structure Prog (Γ : Nat) where
  {size : Nat}
  binds : Tele Γ size
  out   : Expr (Γ + size)

/-- A program with `Γ` results — the shape a gradient needs, where every output
    shares one telescope. -/
structure VProg (Γ : Nat) (m : Nat) where
  {size : Nat}
  binds : Tele Γ size
  outs  : Fin m → Expr (Γ + size)

/-- Result `i` of a multi-output program, as an ordinary expression. -/
def VProg.get {m : Nat} (p : VProg Γ m) (i : Fin m) : Expr Γ :=
  p.binds.bind (p.outs i)

@[simp] theorem VProg.denote_get {m : Nat} (p : VProg Γ m) (i : Fin m)
    (env : Fin Γ → R) :
    denote env (p.get i) = denote (p.binds.env env) (p.outs i) :=
  denote_bind p.binds (p.outs i) env

-- ---------------------------------------------------------------------------
-- The reverse sweep
-- ---------------------------------------------------------------------------

/-- Forward-binding indices in the order the reverse sweep visits them:
    `revIdx n = [n-1, n-2, …, 0]`.

    Written as its own recursion rather than `(List.range n).reverse` so that
    *both* peels are definitional — the head of `revIdx` is the last binding
    (what the telescope induction needs) and the tail of `revTake` is the newest
    adjoint (what the adjoint-telescope induction needs).  Two definitional
    peels is the difference between a short proof and a list-lemma safari. -/
def revIdx : Nat → List Nat
  | 0     => []
  | n + 1 => n :: revIdx n

/-- The first `j` entries of `revIdx n`, i.e. `[n-1, …, n-j]`. -/
def revTake (n : Nat) : Nat → List Nat
  | 0     => []
  | j + 1 => revTake n j ++ [n - 1 - j]

/-- The adjoint of slot `q`, given that `j` adjoint bindings already exist.

    The fold is keyed by the *forward* binding index `i`; the adjoint bound for
    binding `i` sits at telescope slot `(Γ+n) + (n-1-i)`.  Both guards hold for
    every `i` the list actually contains (`revTake_mem`); they are there to keep
    the definition total. -/
def adjFoldE {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (q : Fin (Γ + n))
    (J : Nat) (_hJ : J ≤ n) (j : Nat) : Expr ((Γ + n) + J) :=
  (revTake n j).foldl
    (fun acc i =>
      if h : i < n ∧ n - 1 - i < J then
        (if q.val < Γ + i then
          .add acc
            (.mul (.var ⟨(Γ + n) + (n - 1 - i), by omega⟩)
                  (wkTo (by omega) (sderiv (t.getW ⟨i, h.1⟩) q)))
         else acc)
      else acc)
    (wkTo (by omega) (sderiv out q))

/-- The adjoint of slot `q` once all `j` adjoint bindings exist.

    `adjFoldE` separates *how far the sweep has run* (`j`) from *how big the
    context is* (`J`).  At a fixed `J` every partial sum has the same type, so
    the correctness proof can induct on the fold — with the two conflated, each
    partial sum would live in a different `Expr` and the induction would need a
    transport at every step. -/
def adjAt {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (q : Fin (Γ + n))
    (j : Nat) (hj : j ≤ n) : Expr ((Γ + n) + j) :=
  adjFoldE t out q j hj j

/-- The adjoint bindings, innermost-forward-slot last. -/
def adjTele {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) :
    (j : Nat) → j ≤ n → Tele (Γ + n) j
  | 0,     _  => .nil
  | j + 1, hj =>
      (adjTele t out j (by omega)).cons
        (adjAt t out ⟨Γ + (n - 1 - j), by omega⟩ j (by omega))

/-- **The gradient as a program.**

    One telescope — the forward bindings, then the adjoint bindings — and `Γ`
    outputs that share it.  This is the `VProg` shape the size argument said was
    necessary: the adjoints are variables, so nothing is duplicated across the
    `Γ` results.

    **Proven correct** against `sderiv` — see `gradProg_correct` in
    `AlgorithmLib.ML.TapeGrad`, which needs no `NumLaws` and so holds at
    `Float32`. -/
def gradProg {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) : VProg Γ Γ :=
  { binds := t.append (adjTele t out n (Nat.le_refl n))
    outs  := fun k =>
      castE (by omega) (adjAt t out ⟨k.val, by omega⟩ n (Nat.le_refl n)) }

-- ---------------------------------------------------------------------------
-- The narrowed sweep: linear instead of quadratic
-- ---------------------------------------------------------------------------

/-! `adjAt` sums over *every* later binding, because in general every binding is
    in scope for all its successors.  In a layered model almost all of those
    terms are zero — binding `i` simply does not mention a slot bound long
    before it.  `dep i` is the lower edge of the window binding `i` actually
    reads, and the narrowed construction below emits only the terms inside it.

    For a layered program the window is one block, so the sum ranges over `d`
    terms instead of `n`: `O(n·d)` instead of `O(n²)`, a factor of `L`.

    This is a *change of specification*, not a cleverer implementation — the
    dropped terms are in `sderiv` itself — so its correctness theorem carries
    `ZeroTermFree` (see `AlgorithmLib.ML.TapeGrad`).  `dep = fun _ => 0` is the
    un-narrowed construction, and needs no such hypothesis. -/

def adjFoldED {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (q : Fin (Γ + n))
    (dep : Nat → Nat) (J : Nat) (_hJ : J ≤ n) (j : Nat) : Expr ((Γ + n) + J) :=
  (revTake n j).foldl
    (fun acc i =>
      if h : i < n ∧ n - 1 - i < J then
        (if dep i ≤ q.val ∧ q.val < Γ + i then
          .add acc
            (.mul (.var ⟨(Γ + n) + (n - 1 - i), by omega⟩)
                  (wkTo (by omega) (sderiv (t.getW ⟨i, h.1⟩) q)))
         else acc)
      else acc)
    (wkTo (by omega) (sderiv out q))

def adjAtD {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (q : Fin (Γ + n))
    (dep : Nat → Nat) (j : Nat) (hj : j ≤ n) : Expr ((Γ + n) + j) :=
  adjFoldED t out q dep j hj j

def adjTeleD {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (dep : Nat → Nat) :
    (j : Nat) → j ≤ n → Tele (Γ + n) j
  | 0,     _  => .nil
  | j + 1, hj =>
      (adjTeleD t out dep j (by omega)).cons
        (adjAtD t out ⟨Γ + (n - 1 - j), by omega⟩ dep j (by omega))

/-- **The narrowed gradient program.**  Same shape as `gradProg`, but each
    adjoint sums only over the bindings that can actually reach it. -/
def gradProgD {n : Nat} (t : Tele Γ n) (out : Expr (Γ + n)) (dep : Nat → Nat) :
    VProg Γ Γ :=
  { binds := t.append (adjTeleD t out dep n (Nat.le_refl n))
    outs  := fun k =>
      castE (by omega) (adjAtD t out ⟨k.val, by omega⟩ dep n (Nat.le_refl n)) }

end AlgorithmLib.ML
