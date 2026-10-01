module
public import AlgorithmLib.ML.Tensor.Vocab
meta import AlgorithmLib.ML.Tensor.Vocab
public import AlgorithmLib.ML.Machine.LaneEval
meta import AlgorithmLib.ML.Machine.LaneEval
public import AlgorithmLib.ML.Kernel.Batch
meta import AlgorithmLib.ML.Kernel.Batch
public import AlgorithmLib.ML.Kernel.SoftmaxCE
meta import AlgorithmLib.ML.Kernel.SoftmaxCE
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Tensor terms and their tape

The model as it is written (`Ten`) and as it is flattened (`TOp`), with no
launch in sight: an operation says what it computes over which buffers, and
which implementation a contraction gets is a `Backend` it carries, not a
lowering it performs.
-/

namespace AlgorithmLib.ML

-- Tensor terms: shapes in the type, sharing in the term
-- ---------------------------------------------------------------------------

/-!
  A `Node` list still names its own buffers and carries its own grids.  `Ten`
  removes both: an operation's widths *are* its type indices, so `mv` accepts
  only a matching contraction and the launch grid is derived rather than
  written.  A shape mismatch is a unification failure at elaboration, naming
  both extents — there is no shape checker, no `ok` flag and no `Option`.

  ## Sharing is declared, not discovered

  Lean's own `let` substitutes, so `let h := silu (w1 * x)` used twice would
  flatten to two kernels.  `letT` is the binder that does not: it flattens its
  right-hand side once and hands the body the buffer.  This is `Frontend.letIn`
  one level up, and the reason is the same — it is what keeps a term linear in
  the depth of the network rather than exponential.

  The body takes a *variable*, not a term, which is what makes `Ten` a legal
  inductive: a body of type `Ten r c → Ten r' c'` puts `Ten` left of an arrow
  and is not strictly positive.  Parameterising by the variable representation
  `v` moves the recursive occurrence back to a positive position, and a program
  quantified over `v` cannot inspect what a variable *is* — so it cannot depend
  on the buffer numbers it will be given.

  ## Everything reduces

  `flat` is ordinary structural recursion into a first-order list.  Nothing
  here is a `StateM`, so the erasure check between a schedule and its model is
  `rfl` — which compares the elementwise `Expr` payloads too, and needs no
  `DecidableEq`.
-/

/-- The variable representation used when a program is flattened: a variable
    stands for the buffer its right-hand side was written to. -/
abbrev RefV : Nat → Nat → Type := fun _ _ => Ref

/-- A tensor expression of extent `r × c`, over variable representation `v`. -/
inductive Ten (v : Nat → Nat → Type) : Nat → Nat → Type where
  /-- A `letT`-bound intermediate. -/
  | var   : {r c : Nat} → v r c → Ten v r c
  /-- A buffer the program does not compute: an input, a parameter, a label. -/
  | inp   : {r c : Nat} → Ref → Ten v r c
  /-- `y[s][o] = Σᵢ W[o][i]·x[s][i]`. The contraction width is shared by the
      two operand types, so a mismatch does not elaborate. -/
  | mv    : {b i o : Nat} → Backend → Ten v o i → Ten v b i → Ten v b o
  /-- **A contraction of `b` rows landing in a tensor of `bAlloc`.**

      What needs it is a shape whose two sides disagree about padding: a row
      pass reduces a row of width `w` in `w/32` lane-strided trips, so an
      attention's *keys* must be a multiple of 32, while its *queries* are a
      grid and need not be.  The keys are then a taller tensor than anything
      that produces them, and this is the operation that says so.

      The rows in `[b, bAlloc)` are written by nothing.  A device buffer is
      allocated zeroed, so they read as zero — and what makes zero the right
      answer rather than merely a definite one is the mask: a padded key scores
      `-1e30`, so softmax gives it weight exactly zero and its value never
      reaches the output. -/
  | mvAt  : {b i o : Nat} → (bAlloc : Nat) → Backend → Ten v o i → Ten v b i →
            Ten v bAlloc o
  /-- `dx[s][i] = Σₒ dy[s][o]·W[o][i]` — the transposed walk. -/
  | mvT   : {b i o : Nat} → Backend → Ten v o i → Ten v b o → Ten v b i
  /-- `dW[o][i] = Σₛ dy[s][o]·x[s][i]` — the batch sum. -/
  | outer : {b i o : Nat} → Backend → Ten v b o → Ten v b i → Ten v o i
  /-- A unary elementwise pass. -/
  | ew1   : {r c : Nat} → Expr 1 → Ten v r c → Ten v r c
  /-- A binary elementwise pass; both operands carry the same extent. -/
  | ew2   : {r c : Nat} → Expr 2 → Ten v r c → Ten v r c → Ten v r c
  /-- A ternary elementwise pass — a gated feed-forward (`silu(gate) * up`)
      or a fused residual-and-scale.  All three operands share an extent. -/
  | ew3   : {r c : Nat} → Expr 3 → Ten v r c → Ten v r c → Ten v r c → Ten v r c
  /-- Softmax and the cross-entropy gradient, one warp per row. -/
  | smce  : {b c : Nat} → Ten v b c → Ten v 1 c → Ten v b c → Ten v b c
  /-- `s[j] = Σᵢ x[j][i]²` — one scalar per row.  The extent drops to a single
      column, so a statistic cannot be fed to an elementwise pass that expects
      a full row: the broadcast has to be written. -/
  | rowsq : {r c : Nat} → Ten v r c → Ten v r 1
  /-- A row pass whose second operand is one scalar per row — a norm's
      statistic reaching the elements it scales. -/
  | zipS  : {r c : Nat} → WFExp → Ten v r c → Ten v r 1 → Ten v r c
  /-- A row pass whose second operand is one vector shared by every row — a
      gain, a bias, a rotation table. -/
  | zipB  : {r c : Nat} → WFExp → Ten v r c → Ten v 1 c → Ten v r c
  /-- **Rotary position embedding** on rows of width `c`, split at `c/2`:
      the cosine and sine tables are shared by every row. -/
  | rope  : {r c : Nat} → Ten v r c → Ten v 1 c → Ten v 1 c → Ten v r c
  /-- `out[s] = Σᵢ a[s][i]·b[i]` — a row fold against a shared vector.  Against
      a vector of ones it is a row sum, which is a softmax's denominator. -/
  | rowB  : {r c : Nat} → Ten v r c → Ten v 1 c → Ten v r 1
  /-- `out[s] = maxᵢ t[s][i]`, from a seed below anything the row holds. -/
  | rowMax : {r c : Nat} → Float32 → Ten v r c → Ten v r 1
  /-- A row pass over three operands: the row, one value per row, one vector
      shared by every row.  What a fused norm is. -/
  | zip3  : {r c : Nat} → WFExp → Ten v r c → Ten v r 1 → Ten v 1 c → Ten v r c
  /-- A row pass whose second operand is **one element**, at a fixed index of
      another tensor.  A router's gate for one expert is read this way. -/
  | zipC  : {r c e : Nat} → WFExp → Nat → Ten v r c → Ten v 1 e → Ten v r c
  /-- Reinterpret the extent at equal size.  Buffers are row-major and
      contiguous, so this is a *view*: it emits no operation, and the equality
      is what stops it being a reshape that would need one. -/
  | view  : {r c r' c' : Nat} → r * c = r' * c' → Ten v r c → Ten v r' c'
  /-- **Columns `[off, off+w)` of every row.**

      Unlike `view` this is *not* free: a column window of a row-major buffer is
      strided, so it costs one pass that gathers the window into a buffer of its
      own.  What it buys is that everything downstream sees an ordinary
      contiguous tensor — in particular a contraction, which takes whole
      pointers.

      It carries the row pass's expression, so the split is not a separate copy:
      a head's share of a projection and its bias are one pass, reading the
      window and the head's own bias vector and writing a contiguous buffer.

      It exists so a projection can be computed for every head at once and split
      afterwards.  `P` per-head contractions sharing an input are one `P` times
      taller contraction, which is measurably 1.8-3.7x faster than issuing them
      separately *and* faster than batching them, and this is what expresses the
      split. -/
  | cols  : {r c w c' : Nat} → (off : Nat) → WFExp → Ten v r c → Ten v 1 c' → Ten v r w
  /-- **Three windows laid side by side** — the inverse of `cols`.

      Three per-head results become one tensor, so what was a sum of three
      contractions over three narrow matrices is one contraction over the whole
      matrix: `Σ_h Wo_h · hd_h = [Wo_0|Wo_1|Wo_2] · [hd_0;hd_1;hd_2]`.  Like
      `cols` it is a real pass — three of them, each writing its own window —
      and like `cols` it pays for itself by what it lets the contraction
      become. -/
  | cat3  : {r w c c' : Nat} → w * 3 = c → Ten v r w → Ten v r w → Ten v r w →
            Ten v 1 c' → Ten v r c
  /-- A name for the buffer this subterm lands in.  Emits nothing and changes
      nothing: `flat` passes straight through it, so a labelled model flattens
      to the identical tape.  It exists so a *schedule* can say where to act by
      the name the model already bound, instead of by a position in the
      compiler's output. -/
  | named : String → Ten v r c → Ten v r c
  /-- An in-place update of the first operand, then the rest of the program.
      An optimiser step is a statement rather than a value: it names no new
      buffer, it overwrites a parameter. -/
  | upd2  : {r c r' c' : Nat} → Expr 2 → Ten v r c → Ten v r c →
            Ten v r' c' → Ten v r' c'
  /-- `let x = rhs in body x` — the binder that flattens `rhs` once. -/
  | letT  : {r c r' c' : Nat} → Ten v r c → (v r c → Ten v r' c') →
            Ten v r' c'

/-- **A flattened operation.**  The same seven operations as `Ten`, with the
    buffers resolved and — unlike `Node` — every arity *definite*.  That is
    what lets a reverse pass match on an elementwise op without a dependent
    match on an implicit `Expr` arity. -/
inductive TOp where
  /-- `bAlloc` is how many rows the output buffer holds and `b` how many the
      contraction writes.  They differ where a consumer's extent is taller than
      the contraction's own — attention keys, padded to the multiple of 32 a row
      pass reduces over, read by queries that are not — and the rows between
      them are written by nothing. -/
  | mv    : Backend → (w x out : Ref) → (b inW outW bAlloc : Nat) → TOp
  | mvT   : Backend → (w dy out : Ref) → (b inW outW : Nat) → TOp
  | outer : Backend → (dy x out : Ref) → (b inW outW : Nat) → TOp
  | ew1   : Expr 1 → (a out : Ref) → (grid : Nat) → TOp
  | ew2   : Expr 2 → (a b out : Ref) → (grid : Nat) → TOp
  | ew3   : Expr 3 → (a b c out : Ref) → (grid : Nat) → TOp
  | ew4   : Expr 4 → (a b c d out : Ref) → (grid : Nat) → TOp
  | smce  : (l bias oh out : Ref) → (grid : Nat) → TOp
  | rowsq  : (x out : Ref) → (n rows : Nat) → TOp
  | ziprow : (a b out : Ref) → WFExp → BCast → BCast → (n off w rows : Nat) → TOp
  | ziprow3 : (a b c out : Ref) → WFExp → BCast → BCast → BCast →
              (n off w rows : Nat) → TOp
  /-- The four-operand row pass.  Only a fusion builds one: a chain of three row
      passes needs this arity, and nothing a model writes has it. -/
  | ziprow4 : (a b c d out : Ref) → WFExp → BCast → BCast → BCast → BCast →
              (n off w rows : Nat) → TOp
  /-- The reduction with a row pass folded into its left factor.  Only a fusion
      builds one; nothing a model writes has this shape. -/
  | rowdot4 : (a b c d out : Ref) → WFExp → BCast → BCast → BCast → BCast →
              (n rows : Nat) → TOp
  | rowdot : (a b out : Ref) → BCast → BCast → (n rows : Nat) → TOp
  | rowmax : (x out : Ref) → (n rows : Nat) → Float32 → TOp
  | upd2  : Expr 2 → (a b : Ref) → (grid : Nat) → TOp

/-- **The derivative of a lane expression with respect to one register.**

    `Option`-valued: `maxW` and `geF` are not differentiable, and `ex2` is the
    hardware's approximation rather than a function this stack has a derivative
    rule for.  A gradient that reached one of them is a build failure, not a
    silently wrong number. -/
def WFExp.deriv : WFExp → Nat → Option WFExp
  | .reg r',   r => some (if r' == r then .lit (NumOps.ofNat 1) else .lit (NumOps.ofNat 0))
  | .lit _,    _ => some (.lit (NumOps.ofNat 0))
  | .add a b,  r => match a.deriv r, b.deriv r with
      | some da, some db => some (.add da db)
      | _, _ => none
  | .mul a b,  r => match a.deriv r, b.deriv r with
      | some da, some db => some (.add (.mul da b) (.mul a db))
      | _, _ => none
  | .neg a,    r => (a.deriv r).map (fun da => .neg da)
  | .inv a,    r => (a.deriv r).map (fun da => .neg (.mul da (.mul (.inv a) (.inv a))))
  | .exp a,    r => (a.deriv r).map (fun da => .mul da (.exp a))
  | .rsqrt a,  r => (a.deriv r).map (fun da =>
      .neg (.mul da (.mul (.inv (.lit (NumOps.ofNat 2))) (.mul (.rsqrt a) (.inv a)))))
  | .ex2 _,    _ => none
  | .maxW _ _, _ => none
  | .geF _ _,  _ => none

/-- The adjoint pass's lane expression: the incoming gradient in register 3,
    times the forward expression's derivative in the operand's register. -/
def adjW (f : WFExp) (r : Nat) : Option WFExp :=
  (f.deriv r).map (fun d => .mul (.reg 3) d)

/-- **The scalar language, as lane code.**  Variable `i` becomes register
    `i+1`, which is the convention the row passes bind their operands to.

    `sum` and `letE` have no lane counterpart — a lane expression is evaluated
    once per element, with no reduction and no sharing — so they map to junk and
    `Expr.laneable` is what says a term avoids them. -/
def Expr.toWF : {Γ : Nat} → Expr Γ → WFExp
  | _, .var i    => .reg (i.val + 1)
  | _, .lit n    => .lit (NumOps.ofNat n)
  | _, .add a b  => .add a.toWF b.toWF
  | _, .mul a b  => .mul a.toWF b.toWF
  | _, .neg a    => .neg a.toWF
  | _, .inv a    => .inv a.toWF
  | _, .exp a    => .exp a.toWF
  | _, .rsqrt a  => .rsqrt a.toWF
  | _, .sum _ _  => .lit (NumOps.ofNat 0)
  | _, .letE _ _ => .lit (NumOps.ofNat 0)

def Expr.laneable : {Γ : Nat} → Expr Γ → Bool
  | _, .var _    => true
  | _, .lit _    => true
  | _, .add a b  => a.laneable && b.laneable
  | _, .mul a b  => a.laneable && b.laneable
  | _, .neg a    => a.laneable
  | _, .inv a    => a.laneable
  | _, .exp a    => a.laneable
  | _, .rsqrt a  => a.laneable
  | _, .sum _ _  => false
  | _, .letE _ _ => false

/-- **The lane-level derivative is the proven scalar one, translated.**

    `Expr.sderiv` is tied to the analytic derivative by `grad_hasDerivAt`; this
    says the rule a row pass differentiates by is that same rule read as lane
    code, not a second differentiator that happens to look similar.  Every
    constructor's operand order is what makes it an equality rather than a
    rearrangement. -/
theorem Expr.toWF_deriv : ∀ {Γ : Nat} (e : Expr Γ), e.laneable = true →
    ∀ (i : Fin Γ), (e.toWF).deriv (i.val + 1) = some (sderiv e i).toWF := by
  intro Γ e
  induction e with
  | var k =>
      intro _ i
      by_cases hij : i = k
      · subst hij
        simp [Expr.toWF, WFExp.deriv, sderiv]
      · have hb : ¬ (k.val + 1 = i.val + 1) := fun hc => hij (Fin.ext (by omega))
        simp only [Expr.toWF, WFExp.deriv, sderiv, if_neg hij,
          show (k.val + 1 == i.val + 1) = false from by
            simp only [beq_eq_false_iff_ne, ne_eq]; exact hb,
          if_false]
        rfl
  | lit n => intro _ _; rfl
  | add a b iha ihb =>
      intro h i
      have h' := Bool.and_eq_true .. |>.mp h
      show (match (a.toWF).deriv (i.val + 1), (b.toWF).deriv (i.val + 1) with
            | some da, some db => some (WFExp.add da db) | _, _ => none) = _
      rw [iha h'.1 i, ihb h'.2 i]
      rfl
  | mul a b iha ihb =>
      intro h i
      have h' := Bool.and_eq_true .. |>.mp h
      show (match (a.toWF).deriv (i.val + 1), (b.toWF).deriv (i.val + 1) with
            | some da, some db => some (WFExp.add (.mul da b.toWF) (.mul a.toWF db))
            | _, _ => none) = _
      rw [iha h'.1 i, ihb h'.2 i]
      rfl
  | neg a ih => intro h i; show ((a.toWF).deriv _).map _ = _; rw [ih h i]; rfl
  | inv a ih => intro h i; show ((a.toWF).deriv _).map _ = _; rw [ih h i]; rfl
  | exp a ih => intro h i; show ((a.toWF).deriv _).map _ = _; rw [ih h i]; rfl
  | rsqrt a ih => intro h i; show ((a.toWF).deriv _).map _ = _; rw [ih h i]; rfl
  | sum n f _ => intro h; exact absurd h (by simp [Expr.laneable])
  | letE a b _ _ => intro h; exact absurd h (by simp [Expr.laneable])

/-- **A laneable term's lane code computes the term.**

    `toWF_deriv` ties the two languages at the derivative; this ties them at the
    value, which is what lets one operation be restated as another carrying the
    same scalar function.  The offset is `toWF`'s own convention: variable `i` is
    register `i+1`. -/
theorem Expr.toWF_eval : ∀ {Γ : Nat} (e : Expr Γ), e.laneable = true →
    ∀ (st : WSt) (l : Lane) (env : Fin Γ → Float32),
      (∀ i : Fin Γ, st.regs (i.val + 1) l = env i) →
      (e.toWF).eval st l = denote env e := by
  intro Γ e
  induction e with
  | var i => intro _ _ _ _ he; exact he i
  | lit n => intro _ _ _ _ _; rfl
  | add a b iha ihb =>
      intro h st l env he
      simp only [Expr.laneable, Bool.and_eq_true] at h
      show NumOps.add _ _ = NumOps.add _ _
      rw [iha h.1 st l env he, ihb h.2 st l env he]
  | mul a b iha ihb =>
      intro h st l env he
      simp only [Expr.laneable, Bool.and_eq_true] at h
      show NumOps.mul _ _ = NumOps.mul _ _
      rw [iha h.1 st l env he, ihb h.2 st l env he]
  | neg a ih => intro h st l env he; show NumOps.neg _ = _; rw [ih h st l env he]; rfl
  | inv a ih => intro h st l env he; show NumOps.inv _ = _; rw [ih h st l env he]; rfl
  | exp a ih => intro h st l env he; show NumOps.exp _ = _; rw [ih h st l env he]; rfl
  | rsqrt a ih => intro h st l env he; show NumOps.rsqrt _ = _; rw [ih h st l env he]; rfl
  | sum n f _ => intro h; exact absurd h (by simp [Expr.laneable])
  | letE a b _ _ => intro h; exact absurd h (by simp [Expr.laneable])

/-- At two variables the lane code reads only the two operand registers. -/
theorem Expr.toWF_pairOnly : ∀ {Γ : Nat} (e : Expr Γ), Γ ≤ 2 → e.laneable = true →
    (e.toWF).pairOnly = true := by
  intro Γ e
  induction e with
  | var i =>
      intro hΓ _
      have hi := i.isLt
      have : i.val = 0 ∨ i.val = 1 := by omega
      rcases this with h | h <;> simp [Expr.toWF, WFExp.pairOnly, h]
  | lit n => intro _ _; rfl
  | add a b iha ihb =>
      intro hΓ h
      simp only [Expr.laneable, Bool.and_eq_true] at h
      simp [Expr.toWF, WFExp.pairOnly, iha hΓ h.1, ihb hΓ h.2]
  | mul a b iha ihb =>
      intro hΓ h
      simp only [Expr.laneable, Bool.and_eq_true] at h
      simp [Expr.toWF, WFExp.pairOnly, iha hΓ h.1, ihb hΓ h.2]
  | neg a ih => intro hΓ h; simpa [Expr.toWF, WFExp.pairOnly] using ih hΓ h
  | inv a ih => intro hΓ h; simpa [Expr.toWF, WFExp.pairOnly] using ih hΓ h
  | exp a ih => intro hΓ h; simpa [Expr.toWF, WFExp.pairOnly] using ih hΓ h
  | rsqrt a ih => intro hΓ h; simpa [Expr.toWF, WFExp.pairOnly] using ih hΓ h
  | sum n f _ => intro _ h; exact absurd h (by simp [Expr.laneable])
  | letE a b _ _ => intro _ h; exact absurd h (by simp [Expr.laneable])

/-- At three variables, the three operand registers. -/
theorem Expr.toWF_tripleOnly : ∀ {Γ : Nat} (e : Expr Γ), Γ ≤ 3 → e.laneable = true →
    (e.toWF).tripleOnly = true := by
  intro Γ e
  induction e with
  | var i =>
      intro hΓ _
      have hi := i.isLt
      have : i.val = 0 ∨ i.val = 1 ∨ i.val = 2 := by omega
      rcases this with h | h | h <;> simp [Expr.toWF, WFExp.tripleOnly, h]
  | lit n => intro _ _; rfl
  | add a b iha ihb =>
      intro hΓ h
      simp only [Expr.laneable, Bool.and_eq_true] at h
      simp [Expr.toWF, WFExp.tripleOnly, iha hΓ h.1, ihb hΓ h.2]
  | mul a b iha ihb =>
      intro hΓ h
      simp only [Expr.laneable, Bool.and_eq_true] at h
      simp [Expr.toWF, WFExp.tripleOnly, iha hΓ h.1, ihb hΓ h.2]
  | neg a ih => intro hΓ h; simpa [Expr.toWF, WFExp.tripleOnly] using ih hΓ h
  | inv a ih => intro hΓ h; simpa [Expr.toWF, WFExp.tripleOnly] using ih hΓ h
  | exp a ih => intro hΓ h; simpa [Expr.toWF, WFExp.tripleOnly] using ih hΓ h
  | rsqrt a ih => intro hΓ h; simpa [Expr.toWF, WFExp.tripleOnly] using ih hΓ h
  | sum n f _ => intro _ h; exact absurd h (by simp [Expr.laneable])
  | letE a b _ _ => intro _ h; exact absurd h (by simp [Expr.laneable])

/-- A register file holding `x` at 1, `y` at 2, `z` at 3 and nothing elsewhere —
    the state the pair and triple readings are stated against. -/
def wstOf (x y z : Float32) : WSt :=
  { regs := fun r _ => if r = 1 then x else if r = 2 then y else
                       if r = 3 then z else NumOps.ofNat 0,
    mem := fun _ _ => NumOps.ofNat 0, smem := fun _ => NumOps.ofNat 0 }

/-- **Two operands: the lane code's pair reading is the term's denotation.** -/
theorem Expr.toWF_evalPair (e : Expr 2) (h : e.laneable = true) (x y : Float32) :
    (e.toWF).evalPair x y
      = denote (fun v : Fin 2 => if v.val = 0 then x else y) e := by
  have hp := Expr.toWF_pairOnly e (by decide) h
  have hs := WFExp.evalPair_eq (e.toWF) hp (wstOf x y (NumOps.ofNat 0)) ⟨0, by decide⟩
  have hx : (wstOf x y (NumOps.ofNat 0)).regs 1 ⟨0, by decide⟩ = x := rfl
  have hy : (wstOf x y (NumOps.ofNat 0)).regs 2 ⟨0, by decide⟩ = y := rfl
  rw [hx, hy] at hs
  rw [← hs]
  refine Expr.toWF_eval e h _ _ _ ?_
  intro i
  have hi := i.isLt
  have : i.val = 0 ∨ i.val = 1 := by omega
  rcases this with hv | hv <;> simp [wstOf, hv]

/-- **Three operands: the same, one register wider.** -/
theorem Expr.toWF_evalTriple (e : Expr 3) (h : e.laneable = true) (x y z : Float32) :
    (e.toWF).evalTriple x y z
      = denote (fun v : Fin 3 => if v.val = 0 then x else if v.val = 1 then y else z) e := by
  have hp := Expr.toWF_tripleOnly e (by decide) h
  have hs := WFExp.evalTriple_eq (e.toWF) hp (wstOf x y z) ⟨0, by decide⟩
  have hx : (wstOf x y z).regs 1 ⟨0, by decide⟩ = x := rfl
  have hy : (wstOf x y z).regs 2 ⟨0, by decide⟩ = y := rfl
  have hz : (wstOf x y z).regs 3 ⟨0, by decide⟩ = z := rfl
  rw [hx, hy, hz] at hs
  rw [← hs]
  refine Expr.toWF_eval e h _ _ _ ?_
  intro i
  have hi := i.isLt
  have : i.val = 0 ∨ i.val = 1 ∨ i.val = 2 := by omega
  rcases this with hv | hv | hv <;> simp [wstOf, hv]

/-- **One operand, read through the pair reading.**  A row pass takes two
    operands; a one-operand elementwise pass supplies the same buffer twice and
    its lane code never mentions the second register. -/
theorem Expr.toWF_evalPair1 (e : Expr 1) (h : e.laneable = true) (x y : Float32) :
    (e.toWF).evalPair x y = denote (fun _ : Fin 1 => x) e := by
  have hp := Expr.toWF_pairOnly e (by decide) h
  have hs := WFExp.evalPair_eq (e.toWF) hp (wstOf x y (NumOps.ofNat 0)) ⟨0, by decide⟩
  have hx : (wstOf x y (NumOps.ofNat 0)).regs 1 ⟨0, by decide⟩ = x := rfl
  have hy : (wstOf x y (NumOps.ofNat 0)).regs 2 ⟨0, by decide⟩ = y := rfl
  rw [hx, hy] at hs
  rw [← hs]
  refine Expr.toWF_eval e h _ _ _ ?_
  intro i
  have hi := i.isLt
  have hv : i.val = 0 := by omega
  simp [wstOf, hv]

/-- **The chunk shape an elementwise pass is written at.**

    An elementwise operation covers `grid·32` addresses one lane at a time.  The
    same addresses are `rows` rows of `grid·32/rows`, and a row pass whose
    operands and destination all walk `.rowOf w 0` covers exactly those.
    `TOp.rowShape_den` says the two denote the same function, so this is a
    schedule choice: it decides which operations can share a kernel, and a kernel
    costs about a microsecond before it moves a byte.

    An operation that writes one of its own operands is left alone, and in-place
    updates have no case at all: a row pass is proven under
    `operand ≠ destination`. -/
def TOp.rowShape (rows : Nat) : TOp → Option TOp
  | .ew1 f a o g =>
      let w := g * 32 / rows
      if (w != 0) && (rows * w == g * 32) && f.laneable && (a != o) then
        some (.ziprow a a o f.toWF (.rowOf w 0) (.rowOf w 0) w 0 w rows)
      else none
  | .ew2 f a b o g =>
      let w := g * 32 / rows
      if (w != 0) && (rows * w == g * 32) && f.laneable && (a != o) && (b != o) then
        some (.ziprow a b o f.toWF (.rowOf w 0) (.rowOf w 0) w 0 w rows)
      else none
  -- A sum of squares is a dot of a row with itself.  `rowsq` hard-codes its
  -- addressing where `rowdot` takes a `BCast`, and a `BCast` is the only place a
  -- base offset can go -- so this is what lets a row statistic read a slice of a
  -- fused buffer.  Same kernel either way: both build `dotStrided`.
  | .rowsq a o n rows =>
      if a != o then
        some (.rowdot a a o (.rowOf n 0) (.rowOf n 0) n rows)
      else none
  | .ew3 f a b c o g =>
      let w := g * 32 / rows
      if (w != 0) && (rows * w == g * 32) && f.laneable
           && (a != o) && (b != o) && (c != o) then
        some (.ziprow3 a b c o f.toWF (.rowOf w 0) (.rowOf w 0) (.rowOf w 0)
                w 0 w rows)
      else none
  | _ => none

/-- Re-chunk where it applies and leave the operation alone where it does not. -/
def TOp.atRows (rows : Nat) (op : TOp) : TOp := (op.rowShape rows).getD op

/-- The three combinations a rotation is built from, in the scalar language —
    so `Expr.toWF_deriv` applies to them and their adjoints are `sderiv`.

    A norm's scale is deliberately *not* written this way: its `1/n` and its
    epsilon are reciprocals, and `Expr` has only natural-number literals, so an
    image of it would emit a reciprocal instruction per element where a
    constant belongs. -/
def mulE : Expr 2 := .mul (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)
def addE : Expr 2 := .add (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)
def subE : Expr 2 := .add (.var ⟨0, by decide⟩) (.neg (.var ⟨1, by decide⟩))

def mulW : WFExp := Expr.toWF mulE
def addW : WFExp := Expr.toWF addE
def subW : WFExp := Expr.toWF subE

/-- …and they are the lane code they always were. -/
theorem laneW_unchanged :
    mulW = .mul (.reg 1) (.reg 2)
      ∧ addW = .add (.reg 1) (.reg 2)
      ∧ subW = .add (.reg 1) (.neg (.reg 2)) := ⟨rfl, rfl, rfl⟩

/-- **The kernel an operation launches**, as text-emittable code.

    `Node.stage?` cannot be run: a `StageSpec` carries a `Prop`-valued domain,
    so it is noncomputable, and the PTX has to come from somewhere that
    executes.  This is that somewhere — and `qwen_kernels_are_the_stages` is
    what stops it being a second definition of the model: it says, on the
    shipped list, that these are the very statements the proven stages carry. -/
def TOp.stmt (batch : Nat) : TOp → EWStmt
  | .mv _ w x o b inW outW _ =>
      dotBatched w x (stride32 (.mul .ctaId (.lit inW)))
        (fun s => stride32 (.lit (s * inW))) o
        (fun s => .add (.lit (s * outW)) .ctaId) b (inW / 32)
  | .mvT _ w d o b inW outW =>
      dotBatched w d (.add (.mul (.add (.mul .loopI (.lit 32)) .laneId) (.lit inW)) .ctaId)
        (fun s => stride32 (.lit (s * outW))) o
        (fun s => .add (.lit (s * inW)) .ctaId) b (outW / 32)
  | .outer _ d x o b inW outW =>
      outerBatched d x o (fun s => .add (.lit (s * outW)) .ctaId)
        (fun s => stride32 (.lit (s * inW))) (.mul .ctaId (.lit inW)) b (inW / 32)
  | .ew1 f a o _         => (mapKernel f (fun _ => a) o).ew
  | .ew2 f a b o _       =>
      (mapKernel f (fun j : Fin 2 => if j.val = 0 then a else b) o).ew
  | .ew3 f a b c o _     =>
      (mapKernel f (fun j : Fin 3 =>
        if j.val = 0 then a else if j.val = 1 then b else c) o).ew
  | .ew4 f a b c d o _   =>
      (mapKernel f (fun j : Fin 4 =>
        if j.val = 0 then a else if j.val = 1 then b else
          if j.val = 2 then c else d) o).ew
  | .smce l bi oh o _    => softmaxCE l bi oh o .laneId
  | .upd2 f a b _        =>
      (mapKernel f (fun j : Fin 2 => if j.val = 0 then a else b) a).ew
  | .rowsq x o n _       =>
      dotStrided x x (stride32 (.mul .ctaId (.lit n)))
        (stride32 (.mul .ctaId (.lit n))) o .ctaId (n / 32)
  | .rowdot a b o mA mB n _ => dotStrided a b mA.ix mB.ix o .ctaId (n / 32)
  | .rowmax x o n _ init =>
      maxStrided x (stride32 (.mul .ctaId (.lit n))) o .ctaId (n / 32) init
  | .ziprow a b o f mA mB n off w _ =>
      zipPassEW a b o 1 2 0 f mA.ix mB.ix
        (stride32 (.add (.mul .ctaId (.lit n)) (.lit off))) (w / 32)
  | .ziprow3 a b c o f mA mB mC n off w _ =>
      zip3PassEW a b c o 1 2 3 0 f mA.ix mB.ix mC.ix
        (stride32 (.add (.mul .ctaId (.lit n)) (.lit off))) (w / 32)
  | .ziprow4 a b c d o f mA mB mC mD n off w _ =>
      zip4PassEW a b c d o 1 2 3 4 0 f mA.ix mB.ix mC.ix mD.ix
        (stride32 (.add (.mul .ctaId (.lit n)) (.lit off))) (w / 32)
  | .rowdot4 a b c d o f mA mB mC mD n _ =>
      dotStrided4 a b c d mA.ix mB.ix mC.ix mD.ix f o .ctaId (n / 32)

/-- **The blocks an operation launches over** — the stage's grid, on the
    runnable side.  `qwen_grids_are_the_stages` is what keeps it honest. -/
def TOp.gridOf : TOp → Nat
  | .mv _ _ _ _ _ _ outW _  => outW
  | .mvT _ _ _ _ _ inW _    => inW
  | .outer _ _ _ _ _ _ outW => outW
  | .ew1 _ _ _ g          => g
  | .ew2 _ _ _ _ g        => g
  | .ew3 _ _ _ _ _ g      => g
  | .ew4 _ _ _ _ _ _ g    => g
  | .smce _ _ _ _ g       => g
  | .upd2 _ _ _ g         => g
  | .rowsq _ _ _ rows     => rows
  | .rowdot _ _ _ _ _ _ r => r
  | .rowmax _ _ _ r _     => r
  | .ziprow _ _ _ _ _ _ _ _ _ r => r
  | .ziprow3 _ _ _ _ _ _ _ _ _ _ _ r => r
  | .ziprow4 _ _ _ _ _ _ _ _ _ _ _ _ _ r => r
  | .rowdot4 _ _ _ _ _ _ _ _ _ _ _ r => r

/-- **The buffer an operation writes, and how many bytes it needs.**

    A host allocates from this rather than from a table written beside the
    model, so a shape that changed in the model changes the allocation. -/
def TOp.outSize (batch : Nat) : TOp → Ref × Nat
  | .mv _ _ _ o _ _ outW bA => (o, bA * outW * 4)
  | .mvT _ _ _ o b inW _    => (o, b * inW * 4)
  | .outer _ _ _ o _ inW outW => (o, outW * inW * 4)
  | .ew1 _ _ o g          => (o, g * 32 * 4)
  | .ew2 _ _ _ o g        => (o, g * 32 * 4)
  | .ew3 _ _ _ _ o g      => (o, g * 32 * 4)
  | .ew4 _ _ _ _ _ o g    => (o, g * 32 * 4)
  | .smce _ _ _ o g       => (o, g * 32 * 4)
  | .upd2 _ a _ g         => (a, g * 32 * 4)
  | .rowsq _ o _ rows     => (o, rows * 4)
  | .rowdot _ _ o _ _ _ r => (o, r * 4)
  | .rowmax _ o _ r _     => (o, r * 4)
  | .ziprow _ _ o _ _ _ n _ _ r => (o, r * n * 4)
  | .ziprow3 _ _ _ o _ _ _ _ n _ _ r => (o, r * n * 4)
  | .ziprow4 _ _ _ _ o _ _ _ _ _ n _ _ r => (o, r * n * 4)
  | .rowdot4 _ _ _ _ o _ _ _ _ _ _ rows => (o, rows * 4)

/-- Flatten to the graph, allocating computed buffers from `n` upward.

    Returns the buffer the term landed in, the next free buffer, and the
    operations in the order the driver performs them. -/
def Ten.flat : {r c : Nat} → Ten RefV r c → Ref → Ref × Ref × List TOp
  | _, _, .var r, n => (r, n, [])
  | _, _, .inp r, n => (r, n, [])
  | _, _, @Ten.mv _ b i o bk w x, n =>
      let (rw, n1, fw) := w.flat n
      let (rx, n2, fx) := x.flat n1
      (n2, n2 + 1, fw ++ fx ++ [TOp.mv bk rw rx n2 b i o b])
  | _, _, @Ten.mvAt _ b i o bA bk w x, n =>
      let (rw, n1, fw) := w.flat n
      let (rx, n2, fx) := x.flat n1
      (n2, n2 + 1, fw ++ fx ++ [TOp.mv bk rw rx n2 b i o bA])
  | _, _, @Ten.mvT _ b i o bk w d, n =>
      let (rw, n1, fw) := w.flat n
      let (rd, n2, fd) := d.flat n1
      (n2, n2 + 1, fw ++ fd ++ [TOp.mvT bk rw rd n2 b i o])
  | _, _, @Ten.outer _ b i o bk d x, n =>
      let (rd, n1, fd) := d.flat n
      let (rx, n2, fx) := x.flat n1
      (n2, n2 + 1, fd ++ fx ++ [TOp.outer bk rd rx n2 b i o])
  | r, c, .ew1 spec a, n =>
      let (ra, n1, fa) := a.flat n
      (n1, n1 + 1, fa ++ [TOp.ew1 spec ra n1 (r * c / 32)])
  | r, c, .ew2 spec a b, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      (n2, n2 + 1, fa ++ fb ++
        [TOp.ew2 spec ra rb n2 (r * c / 32)])
  | r, c, .ew3 spec a b d, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      let (rd, n3, fd) := d.flat n2
      (n3, n3 + 1, fa ++ fb ++ fd ++ [TOp.ew3 spec ra rb rd n3 (r * c / 32)])
  | r, _, @Ten.rowsq _ _ c x, n =>
      let (rx, n1, fx) := x.flat n
      (n1, n1 + 1, fx ++ [TOp.rowsq rx n1 c r])
  | r, c, @Ten.zipS _ _ _ f a b, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      (n2, n2 + 1, fa ++ fb ++ [TOp.ziprow ra rb n2 f (.rowOf c 0) .scalar c 0 c r])
  | r, c, @Ten.zipB _ _ _ f a b, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      (n2, n2 + 1, fa ++ fb ++ [TOp.ziprow ra rb n2 f (.rowOf c 0) (.sharedAt 0) c 0 c r])
  | _, _, .view _ t, n => t.flat n
  -- The read is `.rowOf c off` — element `j` of row `s` is at `s*c + off + j` —
  -- and the write is at pitch `w`, so the window lands contiguous.  The same
  -- shape `rope` uses to split a row in half.
  | r, c, @Ten.cat3 _ _ w _ _ _ a b d z, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      let (rd, n3, fd) := d.flat n2
      let (rz, n4, fz) := z.flat n3
      -- Each pass reads its operand contiguously and writes the window
      -- `[h*w, (h+1)*w)` of one shared buffer; the three windows are disjoint,
      -- so between them they write all of it.
      (n4, n4 + 1, fa ++ fb ++ fd ++ fz ++
        [ TOp.ziprow ra rz n4 (.reg 1) (.rowOf w 0) (.sharedAt 0) c 0 w r
        , TOp.ziprow rb rz n4 (.reg 1) (.rowOf w 0) (.sharedAt 0) c w w r
        , TOp.ziprow rd rz n4 (.reg 1) (.rowOf w 0) (.sharedAt 0) c (w + w) w r ])
  | r, w, @Ten.cols _ _ c _ _ off f t z, n =>
      let (rt, n1, ft) := t.flat n
      let (rz, n2, fz) := z.flat n1
      (n2, n2 + 1, ft ++ fz ++
        [TOp.ziprow rt rz n2 f (.rowOf c off) (.sharedAt 0) w 0 w r])
  | _, _, .named _ t, n => t.flat n
  | r, c, @Ten.zipC _ _ _ _ f k a b, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      (n2, n2 + 1, fa ++ fb ++
        [TOp.ziprow ra rb n2 f (.rowOf c 0) (.constAt k) c 0 c r])
  | r, c, @Ten.zip3 _ _ _ f a b d, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      let (rd, n3, fd) := d.flat n2
      (n3, n3 + 1, fa ++ fb ++ fd ++
        [TOp.ziprow3 ra rb rd n3 f (.rowOf c 0) .scalar (.sharedAt 0) c 0 c r])
  | r, _, @Ten.rowMax _ _ c init x, n =>
      let (rx, n1, fx) := x.flat n
      (n1, n1 + 1, fx ++ [TOp.rowmax rx n1 c r init])
  | r, _, @Ten.rowB _ _ c a b, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      (n2, n2 + 1, fa ++ fb ++
        [TOp.rowdot ra rb n2 (.rowOf c 0) (.sharedAt 0) c r])
  | r, c, @Ten.rope _ _ _ x cosT sinT, n =>
      let (rx, n1, fx) := x.flat n
      let (rco, n2, fco) := cosT.flat n1
      let (rsi, n3, fsi) := sinT.flat n2
      let h := c / 2
      let t1 := n3; let t2 := n3 + 1; let t3 := n3 + 2; let t4 := n3 + 3
      let o := n3 + 4
      (o, n3 + 5, fx ++ fco ++ fsi ++
        [ TOp.ziprow rx rco t1 mulW (.rowOf c 0) (.sharedAt 0) h 0 h r
        , TOp.ziprow rx rsi t2 mulW (.rowOf c h) (.sharedAt 0) h 0 h r
        , TOp.ziprow rx rco t3 mulW (.rowOf c h) (.sharedAt h) h 0 h r
        , TOp.ziprow rx rsi t4 mulW (.rowOf c 0) (.sharedAt h) h 0 h r
        , TOp.ziprow t1 t2 o subW (.rowOf h 0) (.rowOf h 0) c 0 h r
        , TOp.ziprow t3 t4 o addW (.rowOf h 0) (.rowOf h 0) c h h r ])
  | b, _, .smce l bias oh, n =>
      let (rl, n1, fl) := l.flat n
      let (rbi, n2, fbi) := bias.flat n1
      let (roh, n3, foh) := oh.flat n2
      (n3, n3 + 1, fl ++ fbi ++ foh ++ [TOp.smce rl rbi roh n3 b])
  | _, _, @Ten.upd2 _ r c _ _ spec a b rest, n =>
      let (ra, n1, fa) := a.flat n
      let (rb, n2, fb) := b.flat n1
      let (rr, n3, fr) := rest.flat n2
      (rr, n3, fa ++ fb ++
        [TOp.upd2 spec ra rb (r * c / 32)] ++ fr)
  | _, _, .letT a k, n =>
      let (ra, n1, fa) := a.flat n
      let (rk, n2, fk) := (k ra).flat n1
      (rk, n2, fa ++ fk)

/-- **Which buffer each name in the model refers to.**

    Descends *every* constructor, so a name is reachable wherever it was
    written — including inside a library combinator, which is where a model
    assembled from `blockAt` and `rmsNorm` binds most of its intermediates.

    The allocation is never recomputed: each subterm's base is read off
    `Ten.flat` itself (`(w.flat n).2.1` is what `flat` hands its next operand),
    so this repeats which children exist and in what order, and nothing about
    numbering.  A name's buffer is then `(a.flat n).1` by definition — the
    buffer that subterm lands in.

    Not load-bearing: a schedule that resolves a name to the wrong buffer gets
    a site the fusion guard refuses, which costs an unfused kernel and never a
    wrong answer. -/
def Ten.labels : {r c : Nat} → Ten RefV r c → Ref → List (String × Ref)
  | _, _, .named s a, n => (s, (a.flat n).1) :: a.labels n
  | _, _, .var _, _ => []
  | _, _, .inp _, _ => []
  | _, _, .view _ t, n => t.labels n
  | _, _, .cols _ _ t z, n => t.labels n ++ z.labels (t.flat n).2.1
  | _, _, .cat3 _ a b d _, n =>
      a.labels n ++ b.labels (a.flat n).2.1 ++ d.labels (b.flat (a.flat n).2.1).2.1
  | _, _, @Ten.mv _ _ _ _ _ w x, n => w.labels n ++ x.labels (w.flat n).2.1
  | _, _, @Ten.mvAt _ _ _ _ _ _ w x, n => w.labels n ++ x.labels (w.flat n).2.1
  | _, _, @Ten.mvT _ _ _ _ _ w d, n => w.labels n ++ d.labels (w.flat n).2.1
  | _, _, @Ten.outer _ _ _ _ _ d x, n => d.labels n ++ x.labels (d.flat n).2.1
  | _, _, .ew1 _ a, n => a.labels n
  | _, _, .ew2 _ a b, n => a.labels n ++ b.labels (a.flat n).2.1
  | _, _, .ew3 _ a b d, n =>
      a.labels n ++ b.labels (a.flat n).2.1
        ++ d.labels ((b.flat (a.flat n).2.1).2.1)
  | _, _, .smce l bias oh, n =>
      l.labels n ++ bias.labels (l.flat n).2.1
        ++ oh.labels ((bias.flat (l.flat n).2.1).2.1)
  | _, _, .rowsq x, n => x.labels n
  | _, _, .rowMax _ x, n => x.labels n
  | _, _, @Ten.zipS _ _ _ _ a b, n => a.labels n ++ b.labels (a.flat n).2.1
  | _, _, @Ten.zipB _ _ _ _ a b, n => a.labels n ++ b.labels (a.flat n).2.1
  | _, _, @Ten.zipC _ _ _ _ _ _ a b, n => a.labels n ++ b.labels (a.flat n).2.1
  | _, _, @Ten.rowB _ _ _ a b, n => a.labels n ++ b.labels (a.flat n).2.1
  | _, _, @Ten.rope _ _ _ x cosT sinT, n =>
      x.labels n ++ cosT.labels (x.flat n).2.1
        ++ sinT.labels ((cosT.flat (x.flat n).2.1).2.1)
  | _, _, @Ten.zip3 _ _ _ _ a b d, n =>
      a.labels n ++ b.labels (a.flat n).2.1
        ++ d.labels ((b.flat (a.flat n).2.1).2.1)
  | _, _, @Ten.upd2 _ _ _ _ _ _ a b rest, n =>
      a.labels n ++ b.labels (a.flat n).2.1
        ++ rest.labels ((b.flat (a.flat n).2.1).2.1)
  | _, _, .letT a k, n =>
      a.labels n ++ (k (a.flat n).1).labels (a.flat n).2.1

end AlgorithmLib.ML
