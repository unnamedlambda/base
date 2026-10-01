module
public import AlgorithmLib.ML.Model.Depth
meta import AlgorithmLib.ML.Model.Depth
public import AlgorithmLib.ML.Tensor.Den
meta import AlgorithmLib.ML.Tensor.Den
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # The launched pipeline computes the tape's mathematics

  The missing left end of the chain.  `TenProg.stages` turns a term into
  stages and every stage is proven against its own contract — but nothing said
  the *pipeline* computes the mathematics of the *tape*.  A lowering that
  dropped an extent (the `batch` defect: every matvec launched at a global
  batch instead of the extent in its type) satisfied every stage contract and
  computed the wrong function.

  `TOp.den` is the mathematics of one operation, written directly: which
  addresses of which buffer it writes — the extents come from the op, which
  carries them from the term's *types* — and what lands there, in the
  committed fold orders (`bflyFold`, `dotStridedLane`, `denote`, `smSpec`).
  It mentions no stage, no grid, no `dom`.

  `lowerTOps_den` then closes the seam: **running the launch sequence the
  lowering produces performs exactly the tape's mathematics, op by op** — for
  every tape the lowering accepts, at every depth, by induction rather than by
  instance.  An op whose stage covers different addresses than its type
  demands, or writes different values, makes this theorem unprovable.
-/

namespace AlgorithmLib.ML

-- ---------------------------------------------------------------------------
-- One elementwise pass, re-chunked by row
-- ---------------------------------------------------------------------------

/-!
  An elementwise pass covers `g*32` addresses one lane at a time; a row pass
  covers `rows` rows of `w`.  When the two cover the same addresses they can
  share a kernel, and a kernel costs about a microsecond before it moves a byte
  -- so which chunk shape an elementwise operation is written at decides how many
  kernels a tape needs, and nothing else about it.

  The statements below are what makes that a choice rather than a rewrite to be
  trusted: a `ziprow` whose operands and destination all walk `.rowOf w 0`
  denotes the elementwise pass exactly.
-/

/-- `.rowOf w 0` at the row and column an address decomposes into is that
    address again. -/
theorem rowOf_ev_self (w a : Nat) :
    (BCast.rowOf w 0).ev (a / w) (a % w - 0) = a := by
  show a / w * w + 0 + (a % w - 0) = a
  rw [Nat.add_zero, Nat.sub_zero, Nat.mul_comm]
  exact Nat.div_add_mod a w

/-- The row guard and the element guard admit the same addresses. -/
theorem rowGuard_iff (w rows g a : Nat) (hw : 0 < w) (hg : rows * w = g * 32) :
    (a / w < rows ∧ 0 ≤ a % w ∧ a % w < 0 + w) ↔ a < g * 32 := by
  constructor
  · rintro ⟨h1, -, -⟩
    have h2 := (Nat.div_lt_iff_lt_mul hw).mp h1
    omega
  · intro h
    refine ⟨(Nat.div_lt_iff_lt_mul hw).mpr (by omega), Nat.zero_le _, ?_⟩
    have := Nat.mod_lt a hw
    omega

/-- **A two-operand elementwise pass is a row pass over the same addresses.** -/
theorem TOp.ew2_as_ziprow (f : Expr 2) (x y o : Ref) (rows w g : Nat)
    (hw : 0 < w) (hg : rows * w = g * 32) (hl : f.laneable = true)
    (m : Buf -> Nat -> Float32) :
    (TOp.ziprow x y o f.toWF (.rowOf w 0) (.rowOf w 0) w 0 w rows).den m
      = (TOp.ew2 f x y o g).den m := by
  funext b' a
  show (if b' = o ∧ a / w < rows ∧ 0 ≤ a % w ∧ a % w < 0 + w then
          (f.toWF).evalPair (m x ((BCast.rowOf w 0).ev (a / w) (a % w - 0)))
                            (m y ((BCast.rowOf w 0).ev (a / w) (a % w - 0)))
        else m b' a)
      = if b' = o ∧ a < g * 32 then
          denote (fun v : Fin 2 => if v.val = 0 then m x a else m y a) f
        else m b' a
  rw [rowOf_ev_self w a, Expr.toWF_evalPair f hl]
  by_cases hb : b' = o
  · simp only [hb, true_and]
    by_cases hc : a < g * 32
    · rw [if_pos ((rowGuard_iff w rows g a hw hg).mpr hc), if_pos hc]
    · rw [if_neg (fun k => hc ((rowGuard_iff w rows g a hw hg).mp k)), if_neg hc]
  · rw [if_neg (fun k => hb k.1), if_neg (fun k => hb k.1)]

/-- **A three-operand elementwise pass is a three-operand row pass.** -/
theorem TOp.ew3_as_ziprow3 (f : Expr 3) (x y z o : Ref) (rows w g : Nat)
    (hw : 0 < w) (hg : rows * w = g * 32) (hl : f.laneable = true)
    (m : Buf -> Nat -> Float32) :
    (TOp.ziprow3 x y z o f.toWF (.rowOf w 0) (.rowOf w 0) (.rowOf w 0)
        w 0 w rows).den m
      = (TOp.ew3 f x y z o g).den m := by
  funext b' a
  show (if b' = o ∧ a / w < rows ∧ 0 ≤ a % w ∧ a % w < 0 + w then
          (f.toWF).evalTriple (m x ((BCast.rowOf w 0).ev (a / w) (a % w - 0)))
                              (m y ((BCast.rowOf w 0).ev (a / w) (a % w - 0)))
                              (m z ((BCast.rowOf w 0).ev (a / w) (a % w - 0)))
        else m b' a)
      = if b' = o ∧ a < g * 32 then
          denote (fun v : Fin 3 =>
            if v.val = 0 then m x a else if v.val = 1 then m y a else m z a) f
        else m b' a
  rw [rowOf_ev_self w a, Expr.toWF_evalTriple f hl]
  by_cases hb : b' = o
  · simp only [hb, true_and]
    by_cases hc : a < g * 32
    · rw [if_pos ((rowGuard_iff w rows g a hw hg).mpr hc), if_pos hc]
    · rw [if_neg (fun k => hc ((rowGuard_iff w rows g a hw hg).mp k)), if_neg hc]
  · rw [if_neg (fun k => hb k.1), if_neg (fun k => hb k.1)]


/-- **A one-operand elementwise pass is a row pass reading one buffer twice.** -/
theorem TOp.ew1_as_ziprow (f : Expr 1) (x o : Ref) (rows w g : Nat)
    (hw : 0 < w) (hg : rows * w = g * 32) (hl : f.laneable = true)
    (m : Buf -> Nat -> Float32) :
    (TOp.ziprow x x o f.toWF (.rowOf w 0) (.rowOf w 0) w 0 w rows).den m
      = (TOp.ew1 f x o g).den m := by
  funext b' a
  show (if b' = o ∧ a / w < rows ∧ 0 ≤ a % w ∧ a % w < 0 + w then
          (f.toWF).evalPair (m x ((BCast.rowOf w 0).ev (a / w) (a % w - 0)))
                            (m x ((BCast.rowOf w 0).ev (a / w) (a % w - 0)))
        else m b' a)
      = if b' = o ∧ a < g * 32 then denote (fun _ : Fin 1 => m x a) f
        else m b' a
  rw [rowOf_ev_self w a, Expr.toWF_evalPair1 f hl]
  by_cases hb : b' = o
  · simp only [hb, true_and]
    by_cases hc : a < g * 32
    · rw [if_pos ((rowGuard_iff w rows g a hw hg).mpr hc), if_pos hc]
    · rw [if_neg (fun k => hc ((rowGuard_iff w rows g a hw hg).mp k)), if_neg hc]
  · rw [if_neg (fun k => hb k.1), if_neg (fun k => hb k.1)]

/-- **A row pass reading a fused buffer computes what it did before.**

    Buffer fusion lays several operations' operands end to end in one buffer.  A
    reader of the one that moved to `d` shifts its addressing by `d`
    (`BCast.shift`) and reads the fused buffer instead.  This says that is the
    only change: at every address its output covers, it lands the same value.

    The hypotheses are the fusion's side of the bargain — the fused buffer holds
    the old one's contents from `d` on, and the other operand is untouched.  The
    guard is a hypothesis rather than a disjunct: *outside* its output range the
    two do differ, since one reads the fused buffer and one does not, and a
    statement that quantified over all addresses could only be true by saying
    nothing there.
-/
theorem ziprow_shift_den
    (i i' j o : Ref) (f : WFExp) (mA mA' mB : BCast) (n off w rows d : Nat)
    (hs : mA.shift d = some mA')
    (m m' : Buf → Nat → Float32)
    (hA : ∀ a, m' i' (a + d) = m i a)
    (hB : ∀ a, m' j a = m j a)
    (a : Nat) (hr : a / n < rows) (h1 : off ≤ a % n) (h2 : a % n < off + w) :
    (TOp.ziprow i' j o f mA' mB n off w rows).den m' o a
      = (TOp.ziprow i j o f mA mB n off w rows).den m o a := by
  show (if o = o ∧ _ then _ else _) = (if o = o ∧ _ then _ else _)
  rw [if_pos ⟨rfl, hr, h1, h2⟩, if_pos ⟨rfl, hr, h1, h2⟩]
  rw [BCast.shift_ev mA d mA' hs, hB]
  exact congrArg (fun x => f.evalPair x _) (hA _)

/-- **The three-operand row pass, likewise.**  `ziprow3` is the commonest
    operation on the ViT tape (487 of 2826), so the fusion is worth little
    without it.  Each operand shifts independently: fusion may move one, two or
    all three. -/
theorem ziprow3_shift_den
    (i i' j j' k k' o : Ref) (f : WFExp) (mA mA' mB mB' mC mC' : BCast)
    (n off w rows dA dB dC : Nat)
    (hsA : mA.shift dA = some mA') (hsB : mB.shift dB = some mB')
    (hsC : mC.shift dC = some mC')
    (m m' : Buf → Nat → Float32)
    (hA : ∀ a, m' i' (a + dA) = m i a)
    (hB : ∀ a, m' j' (a + dB) = m j a)
    (hC : ∀ a, m' k' (a + dC) = m k a)
    (a : Nat) (hr : a / n < rows) (h1 : off ≤ a % n) (h2 : a % n < off + w) :
    (TOp.ziprow3 i' j' k' o f mA' mB' mC' n off w rows).den m' o a
      = (TOp.ziprow3 i j k o f mA mB mC n off w rows).den m o a := by
  show (if o = o ∧ _ then _ else _) = (if o = o ∧ _ then _ else _)
  rw [if_pos ⟨rfl, hr, h1, h2⟩, if_pos ⟨rfl, hr, h1, h2⟩]
  rw [BCast.shift_ev mA dA mA' hsA, BCast.shift_ev mB dB mB' hsB,
      BCast.shift_ev mC dC mC' hsC, hA, hB, hC]

/-- **A sum of squares is a dot of the row with itself.**  `rowDot` is *defined*
    as the butterfly over `dotStridedLane`, which is what `rowdot` denotes, and
    `.rowOf n 0` addresses `a * n + t * 32 + l` — exactly what `rowsq` walks. -/
theorem TOp.rowsq_as_rowdot (x o : Ref) (n rows : Nat) (m : Buf → Nat → Float32) :
    (TOp.rowdot x x o (.rowOf n 0) (.rowOf n 0) n rows).den m
      = (TOp.rowsq x o n rows).den m := by
  funext b' a
  by_cases hb : b' = o ∧ a < rows
  · show (if b' = o ∧ a < rows then _ else _) = (if b' = o ∧ a < rows then _ else _)
    rw [if_pos hb, if_pos hb]
    rfl
  · show (if b' = o ∧ a < rows then _ else _) = (if b' = o ∧ a < rows then _ else _)
    rw [if_neg hb, if_neg hb]

/-- **Re-chunking preserves the function an operation computes.**  Whatever
    `rowShape` accepts, it returns an operation with the same denotation, so
    choosing a chunk shape is free of semantic consequence. -/
theorem TOp.rowShape_den (rows : Nat) (op op' : TOp) (m : Buf -> Nat -> Float32)
    (h : op.rowShape rows = some op') : op'.den m = op.den m := by
  cases op
  case ew1 f x o g =>
    simp only [TOp.rowShape] at h
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq] at hc
      obtain ⟨⟨⟨hw, hg⟩, hl⟩, -⟩ := hc
      rw [Option.some.injEq] at h; subst h
      exact TOp.ew1_as_ziprow f x o rows _ g (Nat.pos_of_ne_zero hw) hg hl m
    · exact absurd h (by simp)
  case ew2 f x y o g =>
    simp only [TOp.rowShape] at h
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq] at hc
      obtain ⟨⟨⟨⟨hw, hg⟩, hl⟩, -⟩, -⟩ := hc
      rw [Option.some.injEq] at h; subst h
      exact TOp.ew2_as_ziprow f x y o rows _ g (Nat.pos_of_ne_zero hw) hg hl m
    · exact absurd h (by simp)
  case rowsq x o n rows =>
    simp only [TOp.rowShape] at h
    split at h
    · rw [Option.some.injEq] at h; subst h
      exact TOp.rowsq_as_rowdot x o n rows m
    · exact absurd h (by simp)
  case ew3 f x y z o g =>
    simp only [TOp.rowShape] at h
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq] at hc
      obtain ⟨⟨⟨⟨⟨hw, hg⟩, hl⟩, -⟩, -⟩, -⟩ := hc
      rw [Option.some.injEq] at h; subst h
      exact TOp.ew3_as_ziprow3 f x y z o rows _ g (Nat.pos_of_ne_zero hw) hg hl m
    · exact absurd h (by simp)
  all_goals simp [TOp.rowShape] at h

/-- …so the tape may be re-chunked operation by operation. -/
theorem TOp.atRows_den (rows : Nat) (op : TOp) (m : Buf -> Nat -> Float32) :
    (op.atRows rows).den m = op.den m := by
  unfold TOp.atRows
  cases hh : op.rowShape rows with
  | none => rfl
  | some op' => exact TOp.rowShape_den rows op op' m hh

/-- **Re-chunking a whole tape changes nothing it computes.**

    The list form of the theorem above: a schedule may write every elementwise
    pass as a row pass — which is what lets it share a chunk shape with the row
    passes beside it, and so share their kernel — and the memory the tape leaves
    is the same memory. -/
theorem TOp.atRows_map_den (f : TOp -> Nat) : ∀ (l : List TOp) (m : Buf -> Nat -> Float32),
    (l.map (fun op => op.atRows (f op))).foldl (fun mm o => TOp.den o mm) m
      = l.foldl (fun mm o => TOp.den o mm) m := by
  intro l
  induction l with
  | nil => intro m; rfl
  | cons o rest ih =>
      intro m
      simp only [List.map_cons, List.foldl_cons, TOp.atRows_den, ih]

-- ---------------------------------------------------------------------------
-- Each accepted operation's launch performs its mathematics
-- ---------------------------------------------------------------------------

/-- A strided-lane fold only sees its addresses, so two presentations of the
    same addresses fold identically — lane arguments included. -/
theorem dotSL_congr (A B : Nat → Float32) (fA fB fA' fB' : Nat → Lane → Nat)
    (K : Nat) (l l' : Lane)
    (hA : ∀ i, fA i l = fA' i l') (hB : ∀ i, fB i l = fB' i l') :
    dotStridedLane A B fA fB K l = dotStridedLane A B fA' fB' K l' := by
  show (List.range K).foldl _ (NumOps.ofNat 0) = (List.range K).foldl _ (NumOps.ofNat 0)
  have congr : ∀ (L : List Nat) (init : Float32),
      L.foldl (fun acc i => NumOps.add acc (NumOps.mul (A (fA i l)) (B (fB i l)))) init
        = L.foldl (fun acc i => NumOps.add acc (NumOps.mul (A (fA' i l')) (B (fB' i l')))) init := by
    intro L
    induction L with
    | nil => intro _; rfl
    | cons x xs ih =>
        intro init
        show xs.foldl _ (NumOps.add init (NumOps.mul (A (fA x l)) (B (fB x l)))) = _
        rw [hA x, hB x]
        exact ih _
  exact congr _ _

/-- The map-shaped ownership: one warp per block, one element per lane. -/
theorem own32_iff (g a : Nat) :
    (∃ cta, cta < g ∧ ∃ l : Lane, cta * 32 + l.val = a) ↔ a < g * 32 := by
  constructor
  · rintro ⟨cta, hc, l, rfl⟩
    have h1 : l.val < 32 := l.isLt
    have h2 : (cta + 1) * 32 ≤ g * 32 := Nat.mul_le_mul_right 32 hc
    rw [Nat.succ_mul] at h2
    omega
  · intro ha
    exact ⟨a / 32, by omega, laneMod a, by show a / 32 * 32 + a % 32 = a; omega⟩

/-- `K` trips of 32 lanes cover `[base, base + K·32)` exactly. -/
theorem trip_iff (K base a : Nat) :
    (∃ j, j < K ∧ ∃ l : Lane, base + (j * 32 + l.val) = a)
      ↔ base ≤ a ∧ a - base < K * 32 := by
  constructor
  · rintro ⟨j, hj, l, rfl⟩
    have h1 : l.val < 32 := l.isLt
    have h2 : (j + 1) * 32 ≤ K * 32 := Nat.mul_le_mul_right 32 hj
    rw [Nat.succ_mul] at h2
    omega
  · rintro ⟨hle, hlt⟩
    refine ⟨(a - base) / 32, (Nat.div_lt_iff_lt_mul (by decide)).mpr hlt,
      laneMod (a - base), ?_⟩
    show base + ((a - base) / 32 * 32 + (a - base) % 32) = a
    have := Nat.div_add_mod (a - base) 32
    omega

/-- Rows of width `n` tile the addresses: block `cta` owns `[cta·n, cta·n+n)`. -/
theorem row_own_iff (n g a : Nat) :
    (∃ cta, cta < g ∧ cta * n ≤ a ∧ a - cta * n < n) ↔ (a / n < g ∧ a % n < n) := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · constructor
    · rintro ⟨_, _, _, h⟩; omega
    · rintro ⟨_, h⟩; omega
  have hdm : a / n * n + a % n = a := by
    have := Nat.div_add_mod a n; rw [Nat.mul_comm] at this; omega
  have hmod : a % n < n := Nat.mod_lt _ hn
  constructor
  · rintro ⟨cta, hc, hle, hlt⟩
    have hd : a / n = cta :=
      Nat.div_eq_of_lt_le hle (by rw [Nat.succ_mul]; omega)
    exact ⟨hd ▸ hc, hmod⟩
  · rintro ⟨hc, -⟩
    exact ⟨a / n, hc, by omega, by omega⟩

/-- A contiguous run of `w` offsets, as bounds. -/
theorem exists_t_iff (w base a : Nat) :
    (∃ t, t < w ∧ base + t = a) ↔ base ≤ a ∧ a - base < w := by
  constructor
  · rintro ⟨t, ht, rfl⟩; omega
  · rintro ⟨h1, h2⟩; exact ⟨a - base, h2, by omega⟩

/-- **A stage's step, from ownership and value stated on the address alone.**

    Every stage schema in the vocabulary has this form: an address is written
    exactly when some block in the grid owns it, and what lands there is a
    function of the address (the owning block being recoverable from it). Given
    those two facts the step is a single conditional — which is the shape
    `TOp.den` is written in, so each operation reduces to supplying them. -/
theorem step_of_shape (S : StageSpec) (hex : S.Exclusive)
    (P : Nat → Prop) [DecidablePred P] (F : (Buf → Nat → Float32) → Nat → Float32)
    (hown : ∀ a, (∃ cta, cta < S.grid ∧ S.dom cta a) ↔ P a)
    (hval : ∀ m cta a, cta < S.grid → S.dom cta a → S.val m cta a = F m a)
    (m : Buf → Nat → Float32) (b' : Buf) (a : Nat) :
    S.step m b' a = if b' = S.out ∧ P a then F m a else m b' a := by
  by_cases hb : b' = S.out
  · subst hb
    by_cases ha : P a
    · obtain ⟨cta, hc, hd⟩ := (hown a).mpr ha
      rw [StageSpec.step_val S hex m cta a hc hd, hval m cta a hc hd,
        if_pos (⟨rfl, ha⟩ : S.out = S.out ∧ P a)]
    · rw [if_neg (fun hc : S.out = S.out ∧ P a => ha hc.2)]
      exact StageSpec.step_otherAddr S m a (fun c hc hd => ha ((hown a).mp ⟨c, hc, hd⟩))
  · rw [if_neg (fun hc : b' = S.out ∧ P a => hb hc.1)]
    exact StageSpec.step_otherBuf S m b' a hb

-- ---------------------------------------------------------------------------
-- Each accepted operation's launch performs its mathematics
-- ---------------------------------------------------------------------------

/-- **One accepted operation's launch performs its mathematics.**

    For every op the lowering accepts, the stage's memory action *is* `den`:
    the same buffer, the same rectangle of addresses — whose extents come from
    the op, hence from the term's types — and the same committed folds.  An op
    whose stage covered a different rectangle than its type demands makes this
    unprovable at that constructor. -/
theorem TOp.step_den (op : TOp) (batch : Nat) (X : XStage)
    (h : (op.node.1).stage? batch = some (some X)) (m : Buf → Nat → Float32) :
    X.val.step m = op.den m := by
  cases op with
  | mv bk w x o b inW outW =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (dotBatchedStageX w x (stride32 (.mul .ctaId (.lit inW)))
              (fun s => stride32 (.lit (s * inW))) o b (inW / 32) outW
              hg.1 hg.2.1 hg.2.2).property
            (fun a => a < b * outW)
            (fun m a => rowDot (m w) (m x)
              (fun t => (a % outW) * inW + t) (fun t => (a / outW) * inW + t) (inW / 32))
            (fun a => batched_own_iff b outW a hg.1) ?_ m b' a]
        · rfl
        · rintro m cta a hc ⟨s, hs, rfl⟩
          have hc' : cta < outW := hc
          simp only [dotBatchedStageX, dotBatchedStage]
          have hm : (s * outW + cta) % outW = cta := by
            rw [Nat.mul_comm s outW, Nat.mul_add_mod]
            exact Nat.mod_eq_of_lt hc'
          have hd : (s * outW + cta) / outW = s := by
            rw [Nat.mul_comm s outW, Nat.mul_add_div hg.1, Nat.div_eq_of_lt hc']
            omega
          have hsub : s * outW + cta - cta = s * outW := by omega
          have hd' : (s * outW + cta - cta) / outW = s := by
            rw [hsub, Nat.mul_div_cancel _ hg.1]
          rw [hd', hm, hd]
          rfl
  | mvT bk w d o b inW outW =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (dotBatchedStageX w d
              (.add (.mul (.add (.mul .loopI (.lit 32)) .laneId) (.lit inW)) .ctaId)
              (fun s => stride32 (.lit (s * outW))) o b (outW / 32) inW
              hg.1 hg.2.1 hg.2.2).property
            (fun a => a < b * inW)
            (fun m a => bflyFold (dotStridedLane (m w) (m d)
              (fun i l => (i * 32 + l.val) * inW + a % inW)
              (fun i l => (a / inW) * outW + (i * 32 + l.val)) (outW / 32))
              ⟨0, by decide⟩)
            (fun a => batched_own_iff b inW a hg.1) ?_ m b' a]
        · rfl
        · rintro m cta a hc ⟨s, hs, rfl⟩
          have hc' : cta < inW := hc
          simp only [dotBatchedStageX, dotBatchedStage]
          have hm : (s * inW + cta) % inW = cta := by
            rw [Nat.mul_comm s inW, Nat.mul_add_mod]
            exact Nat.mod_eq_of_lt hc'
          have hd : (s * inW + cta) / inW = s := by
            rw [Nat.mul_comm s inW, Nat.mul_add_div hg.1, Nat.div_eq_of_lt hc']
            omega
          have hsub : s * inW + cta - cta = s * inW := by omega
          have hd' : (s * inW + cta - cta) / inW = s := by
            rw [hsub, Nat.mul_div_cancel _ hg.1]
          rw [hd', hm, hd]
          rfl
  | ew1 f i o g =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (mapStageX f (fun _ => i) o g hg).property
            (fun a => a < g * 32) (fun m a => denote (fun _ => m i a) f)
            (fun a => own32_iff g a) (fun _ _ _ _ _ => rfl) m b' a]
        rfl
  | ew2 f i j o g =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (mapStageX f
              (fun v : Fin 2 => if v.val = 0 then i else j) o g hg).property
            (fun a => a < g * 32)
            (fun m a => denote (fun v : Fin 2 => if v.val = 0 then m i a else m j a) f)
            (fun a => own32_iff g a) (fun m _ a _ _ => by
              refine congrArg (fun q => denote q f) (funext fun v => ?_)
              by_cases hv : v.val = 0 <;> simp [Fin.ext_iff, hv])
            m b' a]
        rfl
  | ew3 f i j k o g =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (mapStageX f
              (fun v : Fin 3 => if v.val = 0 then i else if v.val = 1 then j else k)
              o g hg).property
            (fun a => a < g * 32)
            (fun m a => denote (fun v : Fin 3 =>
              if v.val = 0 then m i a else if v.val = 1 then m j a else m k a) f)
            (fun a => own32_iff g a) (fun m _ a _ _ => by
              refine congrArg (fun q => denote q f) (funext fun v => ?_)
              by_cases h0 : v.val = 0
              · simp [Fin.ext_iff, h0]
              · by_cases h1 : v.val = 1 <;> simp [Fin.ext_iff, h0, h1])
            m b' a]
        rfl
  | ew4 f i j k n o g =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (mapStageX f
              (fun v : Fin 4 => if v.val = 0 then i else if v.val = 1 then j else
                if v.val = 2 then k else n) o g hg).property
            (fun a => a < g * 32)
            (fun m a => denote (fun v : Fin 4 =>
              if v.val = 0 then m i a else if v.val = 1 then m j a else
                if v.val = 2 then m k a else m n a) f)
            (fun a => own32_iff g a) (fun m _ a _ _ => by
              refine congrArg (fun q => denote q f) (funext fun v => ?_)
              by_cases h0 : v.val = 0
              · simp [Fin.ext_iff, h0]
              · by_cases h1 : v.val = 1
                · simp [Fin.ext_iff, h0, h1]
                · by_cases h2 : v.val = 2 <;> simp [Fin.ext_iff, h0, h1, h2])
            m b' a]
        rfl
  | smce l bi oh o g =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (softmaxCEStageX l bi oh o .laneId g
              hg.1 hg.2.1 hg.2.2).property
            (fun a => a < g * 32)
            (fun m a => smSpec (m l) (m bi) (m oh)
              (fun ln => a / 32 * 32 + ln.val) (fun ln => ln.val) (laneMod a))
            (fun a => own32_iff g a) ?_ m b' a]
        · rfl
        · rintro m cta a _ ⟨ln, rfl⟩
          simp only [softmaxCEStageX, softmaxCEStage]
          have hl : ln.val < 32 := ln.isLt
          have hdd : (cta * 32 + ln.val) / 32 = cta := by
            rw [Nat.mul_comm cta 32, Nat.mul_add_div (by decide), Nat.div_eq_of_lt hl]
            omega
          rw [hdd]
          rfl
  | upd2 f i j g =>
      simp only [TOp.node, Node.stage?] at h
      simp only [Option.some.injEq] at h
      subst h
      funext b' a
      rw [step_of_shape _ (mapStageIPX f
            (fun v : Fin 2 => if v.val = 0 then i else j) i g).property
          (fun a => a < g * 32)
          (fun m a => denote (fun v : Fin 2 => if v.val = 0 then m i a else m j a) f)
          (fun a => own32_iff g a) (fun m _ a _ _ => by
            refine congrArg (fun q => denote q f) (funext fun v => ?_)
            by_cases hv : v.val = 0 <;> simp [Fin.ext_iff, hv])
          m b' a]
      rfl
  | rowsq x o n rows =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (reduceStageX x x (stride32 (.mul .ctaId (.lit n)))
              (stride32 (.mul .ctaId (.lit n))) o (n / 32) rows hg.2 hg.2).property
            (fun a => a < rows)
            (fun m a => rowDot (m x) (m x)
              (fun t => a * n + t) (fun t => a * n + t) (n / 32))
            (fun a => ⟨fun ⟨_, hc, hd⟩ => by subst hd; exact hc, fun ha => ⟨a, ha, rfl⟩⟩)
            (fun _ _ _ _ hd => by subst hd; simp only [reduceStageX, reduceStage]; try rfl) m b' a]
        rfl
  | rowmax x o n rows init =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (maxRowStageX x (stride32 (.mul .ctaId (.lit n))) o
              (n / 32) rows init hg.2).property
            (fun a => a < rows)
            (fun m a => bflyFoldOp (fun p q => NumOps.max p q)
              (maxStridedLane (m x)
                (fun t l => (stride32 (.mul .ctaId (.lit n))).eval a t l) (n / 32) init)
              ⟨0, by decide⟩)
            (fun a => ⟨fun ⟨_, hc, hd⟩ => by subst hd; exact hc, fun ha => ⟨a, ha, rfl⟩⟩)
            (fun _ _ _ _ hd => by
              subst hd; simp only [maxRowStageX, maxRowStage]; try rfl) m b' a]
        rfl
  | rowdot i j o mA mB n rows =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (reduceStageX i j mA.ix mB.ix o (n / 32) rows
              hg.2.1 hg.2.2).property
            (fun a => a < rows)
            (fun m a => bflyFold (dotStridedLane (m i) (m j)
              (fun t l => mA.ix.eval a t l) (fun t l => mB.ix.eval a t l) (n / 32))
              ⟨0, by decide⟩)
            (fun a => ⟨fun ⟨_, hc, hd⟩ => by subst hd; exact hc, fun ha => ⟨a, ha, rfl⟩⟩)
            (fun _ _ _ _ hd => by subst hd; simp only [reduceStageX, reduceStage]; try rfl) m b' a]
        rfl
  | rowdot4 i j k d o f mA mB mC mD n rows =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (reduce4StageX i j k d mA.ix mB.ix mC.ix mD.ix f hg.1 o
              (n / 32) rows hg.2.2.1 hg.2.2.2.1 hg.2.2.2.2.1 hg.2.2.2.2.2).property
            (fun a => a < rows)
            (fun m a => bflyFold (dotStridedLane4 (m i) (m j) (m k) (m d)
              (fun t l => mA.ix.eval a t l) (fun t l => mB.ix.eval a t l)
              (fun t l => mC.ix.eval a t l) (fun t l => mD.ix.eval a t l)
              (fun x y z => f.evalTriple x y z) (n / 32))
              ⟨0, by decide⟩)
            (fun a => ⟨fun ⟨_, hc, hd⟩ => by subst hd; exact hc, fun ha => ⟨a, ha, rfl⟩⟩)
            (fun _ _ _ _ hd => by
              subst hd; simp only [reduce4StageX, reduce4Stage]; try rfl) m b' a]
        rfl
  | outer bk d x o b inW outW =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (outerBatchedStageX d x o
              (fun s => .add (.lit (s * outW)) .ctaId)
              (fun s => stride32 (.lit (s * inW))) inW b (inW / 32) outW
              hg.1 hg.2.1 hg.2.2).property
            (fun a => a / inW < outW ∧ a % inW < inW)
            (fun m a => dotStridedLane (m d) (m x)
              (fun s _ => s * outW + a / inW) (fun s _ => s * inW + a % inW) b
              (laneMod a))
            ?_ ?_ m b' a]
        · rfl
        · intro a
          refine Iff.trans ?_ (row_own_iff inW outW a)
          constructor
          · rintro ⟨cta, hc, hd⟩
            have h2 := (trip_iff (inW / 32) (cta * inW) a).mp hd
            rw [hg.1] at h2
            exact ⟨cta, hc, h2.1, h2.2⟩
          · rintro ⟨cta, hc, h1, h2⟩
            refine ⟨cta, hc, (trip_iff (inW / 32) (cta * inW) a).mpr ?_⟩
            rw [hg.1]
            exact ⟨h1, h2⟩
        · rintro m cta a _ hd
          have h2 := (trip_iff (inW / 32) (cta * inW) a).mp hd
          rw [hg.1] at h2
          have hcta : a / inW = cta :=
            Nat.div_eq_of_lt_le h2.1 (by rw [Nat.succ_mul]; omega)
          have hdm := Nat.div_add_mod a inW
          rw [hcta, Nat.mul_comm] at hdm
          have hmod : a - cta * inW = a % inW := by omega
          simp only [outerBatchedStageX, outerBatchedStage]
          refine dotSL_congr _ _ _ _ _ _ _ _ _ ?_ ?_
          · intro s
            show s * outW + cta = s * outW + a / inW
            rw [hcta]
          · intro s
            show s * inW + ((a - cta * inW) / 32 * 32 + (a - cta * inW) % 32)
              = s * inW + a % inW
            have := Nat.div_add_mod (a - cta * inW) 32
            omega
  | ziprow3 i j k o f mA mB mC n off w rows =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (zipRow3StageX i j k o f hg.1 mA mB mC n off (w / 32) rows
              (by rw [hg.2.1]; exact hg.2.2.1) hg.2.2.2.1 hg.2.2.2.2.1
              hg.2.2.2.2.2).property
            (fun a => a / n < rows ∧ off ≤ a % n ∧ a % n < off + w)
            (fun m a => f.evalTriple (m i (mA.ev (a / n) (a % n - off)))
              (m j (mB.ev (a / n) (a % n - off))) (m k (mC.ev (a / n) (a % n - off))))
            ?_ ?_ m b' a]
        · rfl
        · intro a
          refine Iff.trans ?_ (seg_own_iff n off w rows a hg.2.2.1)
          constructor
          · rintro ⟨cta, hc, hd⟩
            have h2 := (trip_iff (w / 32) (cta * n + off) a).mp hd
            rw [hg.2.1] at h2
            exact ⟨cta, hc, (exists_t_iff w (cta * n + off) a).mpr h2⟩
          · rintro ⟨cta, hc, ht⟩
            refine ⟨cta, hc, (trip_iff (w / 32) (cta * n + off) a).mpr ?_⟩
            rw [hg.2.1]
            exact (exists_t_iff w (cta * n + off) a).mp ht
        · rintro m cta a _ hd
          have h2 := (trip_iff (w / 32) (cta * n + off) a).mp hd
          rw [hg.2.1] at h2
          have hle : cta * n ≤ a := by omega
          have hcta : a / n = cta :=
            Nat.div_eq_of_lt_le hle (by rw [Nat.succ_mul]; omega)
          have hdm := Nat.div_add_mod a n
          rw [hcta, Nat.mul_comm] at hdm
          have hmod : a - (cta * n + off) = a % n - off := by omega
          simp only [zipRow3StageX, zipRow3Stage]
          rw [hmod, hcta]
  | ziprow4 i j k d o f mA mB mC mD n off w rows =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (zipRow4StageX i j k d o f hg.1 mA mB mC mD n off (w / 32) rows
              (by rw [hg.2.1]; exact hg.2.2.1) hg.2.2.2.1 hg.2.2.2.2.1
              hg.2.2.2.2.2.1 hg.2.2.2.2.2.2).property
            (fun a => a / n < rows ∧ off ≤ a % n ∧ a % n < off + w)
            (fun m a => f.evalQuad (m i (mA.ev (a / n) (a % n - off)))
              (m j (mB.ev (a / n) (a % n - off))) (m k (mC.ev (a / n) (a % n - off)))
              (m d (mD.ev (a / n) (a % n - off))))
            ?_ ?_ m b' a]
        · rfl
        · intro a
          refine Iff.trans ?_ (seg_own_iff n off w rows a hg.2.2.1)
          constructor
          · rintro ⟨cta, hc, hd⟩
            have h2 := (trip_iff (w / 32) (cta * n + off) a).mp hd
            rw [hg.2.1] at h2
            exact ⟨cta, hc, (exists_t_iff w (cta * n + off) a).mpr h2⟩
          · rintro ⟨cta, hc, ht⟩
            refine ⟨cta, hc, (trip_iff (w / 32) (cta * n + off) a).mpr ?_⟩
            rw [hg.2.1]
            exact (exists_t_iff w (cta * n + off) a).mp ht
        · rintro m cta a _ hd
          have h2 := (trip_iff (w / 32) (cta * n + off) a).mp hd
          rw [hg.2.1] at h2
          have hle : cta * n ≤ a := by omega
          have hcta : a / n = cta :=
            Nat.div_eq_of_lt_le hle (by rw [Nat.succ_mul]; omega)
          have hdm := Nat.div_add_mod a n
          rw [hcta, Nat.mul_comm] at hdm
          have hmod : a - (cta * n + off) = a % n - off := by omega
          simp only [zipRow4StageX, zipRow4Stage]
          rw [hmod, hcta]
  | ziprow i j o f mA mB n off w rows =>
      simp only [TOp.node, Node.stage?] at h
      split at h
      case isFalse => exact absurd h (by simp)
      case isTrue hg =>
        simp only [Option.some.injEq] at h
        subst h
        funext b' a
        rw [step_of_shape _ (zipRowStageX i j o f hg.1 mA mB n off (w / 32) rows
              (by rw [hg.2.1]; exact hg.2.2.1) hg.2.2.2.1 hg.2.2.2.2).property
            (fun a => a / n < rows ∧ off ≤ a % n ∧ a % n < off + w)
            (fun m a => f.evalPair (m i (mA.ev (a / n) (a % n - off)))
              (m j (mB.ev (a / n) (a % n - off))))
            ?_ ?_ m b' a]
        · rfl
        · intro a
          refine Iff.trans ?_ (seg_own_iff n off w rows a hg.2.2.1)
          constructor
          · rintro ⟨cta, hc, hd⟩
            have h2 := (trip_iff (w / 32) (cta * n + off) a).mp hd
            rw [hg.2.1] at h2
            exact ⟨cta, hc, (exists_t_iff w (cta * n + off) a).mpr h2⟩
          · rintro ⟨cta, hc, ht⟩
            refine ⟨cta, hc, (trip_iff (w / 32) (cta * n + off) a).mpr ?_⟩
            rw [hg.2.1]
            exact (exists_t_iff w (cta * n + off) a).mp ht
        · rintro m cta a _ hd
          have h2 := (trip_iff (w / 32) (cta * n + off) a).mp hd
          rw [hg.2.1] at h2
          have hle : cta * n ≤ a := by omega
          have hcta : a / n = cta :=
            Nat.div_eq_of_lt_le hle (by rw [Nat.succ_mul]; omega)
          have hdm := Nat.div_add_mod a n
          rw [hcta, Nat.mul_comm] at hdm
          have hmod : a - (cta * n + off) = a % n - off := by omega
          simp only [zipRowStageX, zipRowStage]
          rw [hmod, hcta]


-- ---------------------------------------------------------------------------
-- …and so does the whole launch sequence
-- ---------------------------------------------------------------------------

/-- A tape operation always lowers to a launch or to nothing at all — never to
    "no launch needed".  Only `Node.input` is stage-free, and no tape operation
    is one, which is what makes the sequence's induction step unconditional. -/
theorem TOp.stage_ne_skip (op : TOp) (batch : Nat) :
    (op.node.1).stage? batch ≠ some none := by
  cases op <;> simp only [TOp.node, Node.stage?] <;>
    first
      | (split <;> simp)
      | simp

/-- **The launch sequence a tape lowers to performs the tape's mathematics.**

    Op by op, at any length: the composite of the stages' steps is the
    composite of the operations' denotations.  This is the statement that makes
    a lowering which drops an extent unprovable rather than merely untested. -/
theorem lowerTOps_den (batch : Nat) : ∀ (ops : List TOp) (ss : List XStage),
    lowerNet batch (forget (ops.map TOp.node)) = some ss →
    ∀ m : Buf → Nat → Float32,
      ss.foldl (fun mm S => S.val.step mm) m
        = ops.foldl (fun mm op => op.den mm) m := by
  intro ops
  induction ops with
  | nil =>
      intro ss h m
      simp only [List.map_nil, forget, lowerNet, Option.some.injEq] at h
      subst h
      rfl
  | cons op rest ih =>
      intro ss h m
      have hforget : forget ((op :: rest).map TOp.node)
          = op.node.1 :: forget (rest.map TOp.node) := rfl
      have hun : lowerNet batch (op.node.1 :: forget (rest.map TOp.node))
          = (match (op.node.1).stage? batch with
             | none => none
             | some none => lowerNet batch (forget (rest.map TOp.node))
             | some (some S) =>
                 (lowerNet batch (forget (rest.map TOp.node))).map (S :: ·)) := rfl
      rw [hforget, hun] at h
      cases hst : (op.node.1).stage? batch with
      | none => rw [hst] at h; exact absurd h (by simp)
      | some oS =>
          cases oS with
          | none => exact absurd hst (TOp.stage_ne_skip op batch)
          | some S =>
              rw [hst] at h
              cases hrest : lowerNet batch (forget (rest.map TOp.node)) with
              | none => rw [hrest] at h; exact absurd h (by simp)
              | some ss' =>
                  rw [hrest] at h
                  have h' : S :: ss' = ss := Option.some.inj h
                  subst h'
                  show ss'.foldl (fun mm T => T.val.step mm) (S.val.step m)
                    = rest.foldl (fun mm o => o.den mm) (op.den m)
                  rw [TOp.step_den op batch S hst m]
                  exact ih ss' hrest _

/-- **Running the pipeline a term lowers to performs the term's mathematics.**

    The left end of the chain, closed: from the emitted launch sequence back to
    the tensor term, at any depth, by induction rather than by instance. -/
theorem TenProg.run_den {r c : Nat} (batch base : Nat) (p : TenProg r c)
    (ss : List XStage) (h : p.stages batch base = some ss) (st : WSt) :
    ((Pipeline.ofStages ss).run st).mem
      = (((p RefV).flat base).2.2).foldl (fun mm op => op.den mm) st.mem := by
  rw [Pipeline.ofStages_runs ss st]
  show (ss.map Subtype.val).foldl (fun mm S => S.step mm) st.mem = _
  rw [List.foldl_map]
  exact lowerTOps_den batch _ ss h st.mem


-- ---------------------------------------------------------------------------
-- Fusion: two row passes into one, bit for bit
-- ---------------------------------------------------------------------------

/-- **A row pass feeding another is one three-operand pass.**

    On the buffer that survives, the pair and the fused kernel compute the same
    number — *bit for bit*, because `WFExp.fuseA` reassociates nothing: every
    operation of both passes survives in its original order.  So this fusion is
    a schedule move that needs no law, and the temporary simply stops being
    written.

    The hypotheses are the ones that make the temporary a temporary: nothing the
    consumer still needs may live in it. -/
theorem fuse_ziprow_den (x ss g t out : Ref) (f1 f2 : WFExp)
    (h1 : f1.pairOnly = true) (h2 : f2.pairOnly = true)
    (mA mB mC : BCast) (nP offP nC offC w rows : Nat)
    (hP : offP + w ≤ nP) (hC : offC + w ≤ nC) (hnP : 0 < nP)
    (htg : g ≠ t) (hot : out ≠ t)
    (m : Buf → Nat → Float32) (a : Nat) :
    ((TOp.ziprow t g out f2 (.rowOf nP offP) mC nC offC w rows).den
      ((TOp.ziprow x ss t f1 mA mB nP offP w rows).den m)) out a
      = (TOp.ziprow3 x ss g out (f2.fuseA f1) mA mB mC nC offC w rows).den
          m out a := by
  by_cases hg : a / nC < rows ∧ offC ≤ a % nC ∧ a % nC < offC + w
  · have hguard : (out = out ∧ a / nC < rows ∧ offC ≤ a % nC ∧ a % nC < offC + w) :=
      ⟨rfl, hg⟩
    -- the address the consumer reads its produced operand at
    have hlt : offP + (a % nC - offC) < nP := by omega
    have hdiv : (a / nC * nP + offP + (a % nC - offC)) / nP = a / nC := by
      rw [show a / nC * nP + offP + (a % nC - offC)
            = nP * (a / nC) + (offP + (a % nC - offC)) by
          rw [Nat.mul_comm]; omega,
        Nat.mul_add_div hnP, Nat.div_eq_of_lt hlt]
      omega
    have hmod : (a / nC * nP + offP + (a % nC - offC)) % nP = offP + (a % nC - offC) := by
      rw [show a / nC * nP + offP + (a % nC - offC)
            = nP * (a / nC) + (offP + (a % nC - offC)) by
          rw [Nat.mul_comm]; omega,
        Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]
    have hT : (TOp.ziprow x ss t f1 mA mB nP offP w rows).den m t
          ((BCast.rowOf nP offP).ev (a / nC) (a % nC - offC))
        = f1.evalPair (m x (mA.ev (a / nC) (a % nC - offC)))
            (m ss (mB.ev (a / nC) (a % nC - offC))) := by
      show (if t = t ∧ _ then _ else _) = _
      rw [show (BCast.rowOf nP offP).ev (a / nC) (a % nC - offC)
            = a / nC * nP + offP + (a % nC - offC) from rfl]
      rw [if_pos (⟨rfl, by rw [hdiv]; exact hg.1, by rw [hmod]; omega,
        by rw [hmod]; omega⟩ :
          t = t ∧ (a / nC * nP + offP + (a % nC - offC)) / nP < rows
            ∧ offP ≤ (a / nC * nP + offP + (a % nC - offC)) % nP
            ∧ (a / nC * nP + offP + (a % nC - offC)) % nP < offP + w)]
      rw [hdiv, hmod, show offP + (a % nC - offC) - offP = a % nC - offC from by omega]
    have hG : ∀ b : Nat,
        (TOp.ziprow x ss t f1 mA mB nP offP w rows).den m g b = m g b := by
      intro b
      show (if g = t ∧ _ then _ else _) = _
      rw [if_neg (fun hc : g = t ∧ _ => htg hc.1)]
    show (if _ then _ else _) = (if _ then _ else _)
    rw [if_pos hguard, if_pos hguard, hT, hG,
        WFExp.fuseA_eval f2 h2 f1 _ _ _, WFExp.evalTriple_of_pairOnly f1 h1]
  · show (if _ then _ else _) = (if _ then _ else _)
    rw [if_neg (fun hc : out = out ∧ _ => hg hc.2),
        if_neg (fun hc : out = out ∧ _ => hg hc.2)]
    show (if out = t ∧ _ then _ else m out a) = m out a
    rw [if_neg (fun hc : out = t ∧ _ => hot hc.1)]

/-- **The three-operand pass fused into the next one is bit-exact too.**

    The same statement as `fuse_ziprow_den` one arity up, and the same proof:
    the consumer reads the producer's output at the producer's own window, so
    the address it would have loaded is the address the producer wrote, and
    `fuseA4` substitutes the expression there with nothing reassociated.

    This is the arity a *chain* needs.  Two row passes collapse without it; a
    third has nowhere to go, which is what leaves an intermediate stored and
    reloaded inside a kernel that already runs both halves. -/
theorem fuse_ziprow3_den (x ss y g t out : Ref) (f1 f2 : WFExp)
    (h1 : f1.tripleOnly = true) (h2 : f2.pairOnly = true)
    (mA mB mC mD : BCast) (nP offP nC offC w rows : Nat)
    (hP : offP + w ≤ nP) (hC : offC + w ≤ nC) (hnP : 0 < nP)
    (htg : g ≠ t) (hot : out ≠ t)
    (m : Buf → Nat → Float32) (a : Nat) :
    ((TOp.ziprow t g out f2 (.rowOf nP offP) mD nC offC w rows).den
      ((TOp.ziprow3 x ss y t f1 mA mB mC nP offP w rows).den m)) out a
      = (TOp.ziprow4 x ss y g out (f2.fuseA4 f1) mA mB mC mD nC offC w rows).den
          m out a := by
  by_cases hg : a / nC < rows ∧ offC ≤ a % nC ∧ a % nC < offC + w
  · have hguard : (out = out ∧ a / nC < rows ∧ offC ≤ a % nC ∧ a % nC < offC + w) :=
      ⟨rfl, hg⟩
    have hlt : offP + (a % nC - offC) < nP := by omega
    have hdiv : (a / nC * nP + offP + (a % nC - offC)) / nP = a / nC := by
      rw [show a / nC * nP + offP + (a % nC - offC)
            = nP * (a / nC) + (offP + (a % nC - offC)) by
          rw [Nat.mul_comm]; omega,
        Nat.mul_add_div hnP, Nat.div_eq_of_lt hlt]
      omega
    have hmod : (a / nC * nP + offP + (a % nC - offC)) % nP = offP + (a % nC - offC) := by
      rw [show a / nC * nP + offP + (a % nC - offC)
            = nP * (a / nC) + (offP + (a % nC - offC)) by
          rw [Nat.mul_comm]; omega,
        Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]
    have hT : (TOp.ziprow3 x ss y t f1 mA mB mC nP offP w rows).den m t
          ((BCast.rowOf nP offP).ev (a / nC) (a % nC - offC))
        = f1.evalTriple (m x (mA.ev (a / nC) (a % nC - offC)))
            (m ss (mB.ev (a / nC) (a % nC - offC)))
            (m y (mC.ev (a / nC) (a % nC - offC))) := by
      show (if t = t ∧ _ then _ else _) = _
      rw [show (BCast.rowOf nP offP).ev (a / nC) (a % nC - offC)
            = a / nC * nP + offP + (a % nC - offC) from rfl]
      rw [if_pos (⟨rfl, by rw [hdiv]; exact hg.1, by rw [hmod]; omega,
        by rw [hmod]; omega⟩ :
          t = t ∧ (a / nC * nP + offP + (a % nC - offC)) / nP < rows
            ∧ offP ≤ (a / nC * nP + offP + (a % nC - offC)) % nP
            ∧ (a / nC * nP + offP + (a % nC - offC)) % nP < offP + w)]
      rw [hdiv, hmod, show offP + (a % nC - offC) - offP = a % nC - offC from by omega]
    have hG : ∀ b : Nat,
        (TOp.ziprow3 x ss y t f1 mA mB mC nP offP w rows).den m g b = m g b := by
      intro b
      show (if g = t ∧ _ then _ else _) = _
      rw [if_neg (fun hc : g = t ∧ _ => htg hc.1)]
    show (if _ then _ else _) = (if _ then _ else _)
    rw [if_pos hguard, if_pos hguard, hT, hG,
        WFExp.fuseA4_eval f2 h2 f1 _ _ _ _, WFExp.evalQuad_of_tripleOnly f1 h1]
  · show (if _ then _ else _) = (if _ then _ else _)
    rw [if_neg (fun hc : out = out ∧ _ => hg hc.2),
        if_neg (fun hc : out = out ∧ _ => hg hc.2)]
    show (if out = t ∧ _ then _ else m out a) = m out a
    rw [if_neg (fun hc : out = t ∧ _ => hot hc.1)]

/-- **A row pass fused into the reduction that consumes it is bit-exact.**

    The consumer reads the produced buffer at the producer's own window, so the
    address it would have loaded is the address the producer wrote, and the fold
    visits the same elements in the same order — only the left factor of each
    product is now computed rather than loaded.

    This is the largest family of intermediates the pair rule cannot reach: a
    row pass whose only consumer is a reduction has nowhere to go without it. -/
theorem fuse_rowdot_den (x ss y g t out : Ref) (f1 : WFExp)
    (h1 : f1.tripleOnly = true)
    (mA mB mC mD : BCast) (nP offP n rows : Nat)
    (hP : offP + n ≤ nP) (hnP : 0 < nP) (hK : (n / 32) * 32 = n)
    (htg : g ≠ t) (hot : out ≠ t) (m : Buf → Nat → Float32) (a : Nat) :
    ((TOp.rowdot t g out (.rowOf nP offP) mD n rows).den
      ((TOp.ziprow3 x ss y t f1 mA mB mC nP offP n rows).den m)) out a
      = (TOp.rowdot4 x ss y g out f1 mA mB mC mD n rows).den m out a := by
  by_cases hg : a < rows
  · have hguard : (out = out ∧ a < rows) := ⟨rfl, hg⟩
    show (if _ then _ else _) = (if _ then _ else _)
    rw [if_pos hguard, if_pos hguard]
    refine congrArg (fun z => bflyFold z ⟨0, by decide⟩) ?_
    funext l
    have hG : ((TOp.ziprow3 x ss y t f1 mA mB mC nP offP n rows).den m) g
        = m g := by
      funext b
      show (if g = t ∧ _ then _ else _) = _
      rw [if_neg (fun hc : g = t ∧ _ => htg hc.1)]
    rw [hG]
    refine dotStridedLane_of_fused _ _ _ _ _ _ _ _ _ _
      (fun u v w => f1.evalTriple u v w) (n / 32) l ?_
    intro i hi
    have hl : i * 32 + l.val < n := by
      have h1 : i + 1 ≤ n / 32 := hi
      have h2 : (i + 1) * 32 ≤ (n / 32) * 32 := Nat.mul_le_mul_right 32 h1
      rw [Nat.succ_mul] at h2
      have h3 : l.val < 32 := l.isLt
      omega
    have hix : (BCast.rowOf nP offP).ix.eval a i l (fun _ _ => 0) (fun _ _ => 0)
        = a * nP + offP + (i * 32 + l.val) := rfl
    have hlt : offP + (i * 32 + l.val) < nP := by omega
    have hdiv : (a * nP + offP + (i * 32 + l.val)) / nP = a := by
      rw [show a * nP + offP + (i * 32 + l.val)
            = nP * a + (offP + (i * 32 + l.val)) by rw [Nat.mul_comm]; omega,
        Nat.mul_add_div hnP, Nat.div_eq_of_lt hlt]
      omega
    have hmod : (a * nP + offP + (i * 32 + l.val)) % nP = offP + (i * 32 + l.val) := by
      rw [show a * nP + offP + (i * 32 + l.val)
            = nP * a + (offP + (i * 32 + l.val)) by rw [Nat.mul_comm]; omega,
        Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]
    show (TOp.ziprow3 x ss y t f1 mA mB mC nP offP n rows).den m t
        ((BCast.rowOf nP offP).ix.eval a i l (fun _ _ => 0) (fun _ _ => 0)) = _
    rw [hix]
    show (if t = t ∧ _ then _ else _) = _
    rw [if_pos (⟨rfl, by rw [hdiv]; exact hg, by rw [hmod]; omega,
      by rw [hmod]; omega⟩ :
        t = t ∧ (a * nP + offP + (i * 32 + l.val)) / nP < rows
          ∧ offP ≤ (a * nP + offP + (i * 32 + l.val)) % nP
          ∧ (a * nP + offP + (i * 32 + l.val)) % nP < offP + n)]
    rw [hdiv, hmod, show offP + (i * 32 + l.val) - offP = i * 32 + l.val from by omega]
    rw [BCast.ix_ev mA a i l, BCast.ix_ev mB a i l, BCast.ix_ev mC a i l]
  · show (if _ then _ else _) = (if _ then _ else _)
    rw [if_neg (fun hc : out = out ∧ _ => hg hc.2),
        if_neg (fun hc : out = out ∧ _ => hg hc.2)]
    show (if out = t ∧ _ then _ else m out a) = m out a
    rw [if_neg (fun hc : out = t ∧ _ => hot hc.1)]

end AlgorithmLib.ML
