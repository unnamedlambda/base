module
public import AlgorithmLib.ML.Machine.Geometry
meta import AlgorithmLib.ML.Machine.Geometry
public import AlgorithmLib.ML.Launch.Pipeline
meta import AlgorithmLib.ML.Launch.Pipeline
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # The frame obligation, proven once for every emittable kernel

  `StageSpec` asks each kernel for a `frame` field: *a block writes only its own
  output addresses*.  Until now every stage discharged it by hand, which is why
  only the kernels built by `compileWKernel` — the ones with
  `compileWKernel_otherBuf`/`_otherAddr` — could become stages at all.  A kernel
  written directly in `EWStmt`, which is most of the interesting ones (RMSNorm's
  two-phase reduction, RoPE's paired store, softmax's three passes, argmax's
  scan), had no route.

  This file proves the obligation **structurally, once**, for an arbitrary
  `EWStmt`:

  * `EWStmt.run_otherBuf` — a statement whose stores all name buffers other than
    `b` leaves `b` entirely alone.  The side condition is `EWStmt.wbufs`, a
    syntactic list, so it is `decide`-able per kernel.
  * `EWStmt.run_otherAddr` — a statement whose stores to `b` all land at
    addresses satisfying `P` leaves every non-`P` address of `b` alone.  `P` is
    the stage's `dom`, so this *is* the frame field.

  Together they give `StageSpec.frame` from two facts about the syntax, and the
  per-kernel work drops to `value` (which the `_stores` theorems already are)
  and `valOnly` (genuinely semantic, and the only place a real judgement is
  needed).

  Nothing here is specific to a model or a kernel.  The leverage is that a
  model's kernels stop needing one frame proof each.
-/

namespace AlgorithmLib.ML

section Carrier
variable {R : Type} [NumOps R] [NumLit R]

-- ---------------------------------------------------------------------------
-- Loop framing
-- ---------------------------------------------------------------------------

/-- A loop preserves whatever every one of its bodies preserves — buffer
    version, over any list of trips (`forN` runs `List.range n`, `forM` the
    range integer memory names). -/
theorem foldl_otherBuf (f : Nat → WSt R → WSt R) (b : Buf)
    (h : ∀ j (st : WSt R), (f j st).mem b = st.mem b) :
    ∀ (L : List Nat) (st : WSt R), (L.foldl (fun s' j => f j s') st).mem b = st.mem b := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons j L ih => intro st; rw [List.foldl_cons, ih, h]

/-- …and the address version, where the hypothesis need only hold for the
    trips the loop actually runs. -/
theorem foldl_otherAddr (f : Nat → WSt R → WSt R) (b : Buf) (a : Nat) :
    ∀ (L : List Nat), (∀ j ∈ L, ∀ (st : WSt R), (f j st).mem b a = st.mem b a) →
      ∀ (st : WSt R), (L.foldl (fun s' j => f j s') st).mem b a = st.mem b a := by
  intro L
  induction L with
  | nil => intro _ _; rfl
  | cons j L ih =>
      intro h st
      rw [List.foldl_cons, ih (fun j' hj' => h j' (List.mem_cons_of_mem j hj')),
        h j List.mem_cons_self]

/-- A `while` loop keeps what each trip of its body keeps. -/
theorem iwhile_inv (n : Nat) (c : IExp) (cs : List Nat) (body : EWStmt) (cta i : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt R) (P : WSt R → Prop)
    (hb : ∀ s, P s → P ((body.run s cta i ir im).withInt (ISt.keep cs st.int (body.run s cta i ir im).int)))
    (h0 : P st) : P ((EWStmt.iwhile n c cs body).run st cta i ir im) := by
  simp only [EWStmt.run]
  refine foldl_inv _ P (fun s _ hs => ?_) _ st h0
  split
  · exact hb s hs
  · exact hs

-- ---------------------------------------------------------------------------
-- Which buffers a kernel can write
-- ---------------------------------------------------------------------------

/-- **Every buffer this statement stores to**, syntactically.

    `WritesBufB` answers the question one buffer at a time, which cannot
    discharge `∀ b ≠ out`.  This lists them, so `∀ c ∈ s.wbufs, c = out` is one
    decidable check that covers every other buffer at once. -/
def EWStmt.wbufs : EWStmt → List Buf
  | .seq x y          => x.wbufs ++ y.wbufs
  | .storeLane0 c _ _ => [c]
  | .storeLane c _ _  => [c]
  | .forN _ body      => body.wbufs
  | .forM _ _ body    => body.wbufs
  | .iif _ x y        => x.wbufs ++ y.wbufs
  | .iwhile _ _ _ body => body.wbufs
  | .locals _ body    => body.wbufs
  | _                 => []

/-- The list is complete: if the predicate says a buffer is written, it is in
    the list.  Written as an agreement theorem between two independently
    defined functions so neither can silently drift from the other — adding an
    `EWStmt` constructor that stores, and updating only one of them, fails
    here rather than in a stage proof that quietly becomes unsound. -/
theorem EWStmt.mem_wbufs_of_writes (b : Buf) :
    ∀ (s : EWStmt), s.WritesBufB b = true → b ∈ s.wbufs := by
  intro s
  induction s with
  | seq x y ihx ihy =>
      intro h
      show b ∈ x.wbufs ++ y.wbufs
      rcases Bool.or_eq_true_iff.mp h with h | h
      · exact List.mem_append_left _ (ihx h)
      · exact List.mem_append_right _ (ihy h)
  | storeLane0 c _ _ => intro h; exact List.mem_singleton.mpr (of_decide_eq_true h).symm
  | storeLane c _ _  => intro h; exact List.mem_singleton.mpr (of_decide_eq_true h).symm
  | forN _ body ih   => intro h; exact ih h
  | forM _ _ body ih => intro h; exact ih h
  | iif _ x y ihx ihy =>
      intro h
      show b ∈ x.wbufs ++ y.wbufs
      rcases Bool.or_eq_true_iff.mp h with h | h
      · exact List.mem_append_left _ (ihx h)
      · exact List.mem_append_right _ (ihy h)
  | iwhile _ _ _ body ih => intro h; exact ih h
  | locals _ body ih => intro h; exact ih h
  | _ => intro h; exact absurd h (by simp [EWStmt.WritesBufB])

/-- **A kernel leaves untouched every buffer it does not name.**

    Proven by structural induction over the whole emittable language, so it
    holds for hand-written `EWStmt` kernels — the two-phase reductions, the
    paired stores — and not only for `compileWKernel` output. -/
theorem EWStmt.run_otherBuf (b : Buf) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ (s : EWStmt), (∀ c ∈ s.wbufs, c ≠ b) →
      ∀ (i : Nat) (st : WSt R), (s.run st cta i ir im).mem b = st.mem b := by
  intro s
  induction s with
  | skip => intro _ _ _; rfl
  | seq x y ihx ihy =>
      intro h i st
      show (y.run (x.run st cta i ir im) cta i ir im).mem b = _
      rw [ihy (fun c hc => h c (List.mem_append_right _ hc)) i _,
          ihx (fun c hc => h c (List.mem_append_left _ hc)) i st]
  | setR _ _ => intro _ _ _; rfl
  | shflXor _ _ _ => intro _ _ _; rfl
  | loadIdx _ _ _ => intro _ _ _; rfl
  | loadV4 _ _ _ _ _ _ => intro _ _ _; rfl
  | storeLane0 c ix r =>
      intro h i st
      show (st.store1 c (ix.eval cta i ⟨0, by decide⟩ ir im)
              (st.regs r ⟨0, by decide⟩)).mem b = _
      funext j
      show (if b = c ∧ _ then _ else _) = _
      rw [if_neg (fun hcon => h c (List.mem_singleton.mpr rfl) hcon.1.symm)]
  | storeLane c ix r =>
      intro h i st
      exact storeLane_otherBuf c (fun l => ix.eval cta i l ir im) r st b
        (fun hcon => h c (List.mem_singleton.mpr rfl) hcon.symm)
  | stSm _ _ => intro _ _ _; rfl
  | ldSm _ _ => intro _ _ _; rfl
  | barrier => intro _ _ _; rfl
  | forN n body ih =>
      intro h _ st
      exact foldl_otherBuf _ b (fun j s' => ih h j s') (List.range n) st
  | forM bu a body ih =>
      intro h _ st
      exact foldl_otherBuf _ b (fun j s' => ih h j s') (List.range (im bu a)) st
  | cvtIF _ _ => intro _ _ _; rfl
  | iif c x y ihx ihy =>
      intro h i st
      simp only [EWStmt.run]
      split
      · exact ihx (fun c hc => h c (List.mem_append_left _ hc)) i st
      · exact ihy (fun c hc => h c (List.mem_append_right _ hc)) i st
  | iwhile n c cs body ih =>
      intro h i st
      exact iwhile_inv n c cs body cta i ir im st (fun s => s.mem b = st.mem b)
        (fun s hs => (ih h i s).trans hs) rfl
  | locals rs body ih => intro h i st; exact ih h i st
  | _ => intro _ _ _; rfl

-- ---------------------------------------------------------------------------
-- Which registers a kernel can write
-- ---------------------------------------------------------------------------

/-- **Every register this statement assigns**, syntactically.

    The register analogue of `wbufs`, and needed for the same reason: a
    multi-pass kernel leaves a value in a register for a *later* pass to read
    (softmax's row maximum in `%fw5`, its reciprocal sum in `%fw3`), so
    composing the passes means knowing the intervening ones do not clobber it.
    Without this the argument is per-pass and by hand. -/
def EWStmt.wregs : EWStmt → List Nat
  | .seq x y              => x.wregs ++ y.wregs
  | .setR d _             => [d]
  | .shflXor d _ _        => [d]
  | .loadIdx d _ _        => [d]
  | .loadV4 a b c d _ _   => [a, b, c, d]
  | .ldSm d _             => [d]
  | .cvtIF d _            => [d]
  | .forN _ body          => body.wregs
  | .forM _ _ body        => body.wregs
  | .iif _ x y            => x.wregs ++ y.wregs
  | .iwhile _ _ _ body    => body.wregs
  | .locals _ body        => body.wregs
  | _                     => []

theorem foldl_otherReg (f : Nat → WSt R → WSt R) (r : Nat)
    (h : ∀ j (st : WSt R), (f j st).regs r = st.regs r) :
    ∀ (L : List Nat) (st : WSt R), (L.foldl (fun s' j => f j s') st).regs r = st.regs r := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons j L ih => intro st; rw [List.foldl_cons, ih, h]

/-- **A kernel leaves untouched every register it does not assign.** -/
theorem EWStmt.run_otherReg (r : Nat) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ (s : EWStmt), (∀ d ∈ s.wregs, d ≠ r) →
      ∀ (i : Nat) (st : WSt R), (s.run st cta i ir im).regs r = st.regs r := by
  intro s
  induction s with
  | skip => intro _ _ _; rfl
  | seq x y ihx ihy =>
      intro h i st
      show (y.run (x.run st cta i ir im) cta i ir im).regs r = _
      rw [ihy (fun d hd => h d (List.mem_append_right _ hd)) i _,
          ihx (fun d hd => h d (List.mem_append_left _ hd)) i st]
  | setR d _ =>
      intro h _ st
      exact WSt.regs_setReg_other st d r _ (Ne.symm (h d (List.mem_singleton.mpr rfl)))
  | shflXor d _ _ =>
      intro h _ st
      exact WSt.regs_setReg_other st d r _ (Ne.symm (h d (List.mem_singleton.mpr rfl)))
  | loadIdx d _ _ =>
      intro h _ st
      exact WSt.regs_setReg_other st d r _ (Ne.symm (h d (List.mem_singleton.mpr rfl)))
  | loadV4 a b c d bu ix =>
      intro h i st
      show ((((st.setReg a _).setReg b _).setReg c _).setReg d _).regs r = _
      rw [WSt.regs_setReg_other _ d r _ (Ne.symm (h d (by simp [EWStmt.wregs]))),
          WSt.regs_setReg_other _ c r _ (Ne.symm (h c (by simp [EWStmt.wregs]))),
          WSt.regs_setReg_other _ b r _ (Ne.symm (h b (by simp [EWStmt.wregs]))),
          WSt.regs_setReg_other _ a r _ (Ne.symm (h a (by simp [EWStmt.wregs])))]
  | ldSm d _ =>
      intro h _ st
      exact WSt.regs_setReg_other st d r _ (Ne.symm (h d (List.mem_singleton.mpr rfl)))
  | cvtIF d _ =>
      intro h _ st
      exact WSt.regs_setReg_other st d r _ (Ne.symm (h d (List.mem_singleton.mpr rfl)))
  | storeLane0 _ _ _ => intro _ _ _; rfl
  | storeLane b ix rr =>
      intro _ i st
      exact congrFun (storeLane_regs b (fun l => ix.eval cta i l ir im) rr st) r
  | stSm _ _ => intro _ _ _; rfl
  | barrier => intro _ _ _; rfl
  | forN n body ih =>
      intro h _ st
      exact foldl_otherReg _ r (fun j s' => ih h j s') (List.range n) st
  | forM bu a body ih =>
      intro h _ st
      exact foldl_otherReg _ r (fun j s' => ih h j s') (List.range (im bu a)) st
  | iif c x y ihx ihy =>
      intro h i st
      simp only [EWStmt.run]
      split
      · exact ihx (fun d hd => h d (List.mem_append_left _ hd)) i st
      · exact ihy (fun d hd => h d (List.mem_append_right _ hd)) i st
  | iwhile n c cs body ih =>
      intro h i st
      exact iwhile_inv n c cs body cta i ir im st (fun s => s.regs r = st.regs r)
        (fun s hs => (ih h i s).trans hs) rfl
  | locals rs body ih => intro h i st; exact ih h i st
  | _ => intro _ _ _; rfl

-- ---------------------------------------------------------------------------
-- Which addresses a kernel can write
-- ---------------------------------------------------------------------------

/-- **Every store to `b` lands inside `P`.**

    The recursion mirrors `EWStmt.run`, so a loop's obligation is stated at the
    iteration index the body runs at — which is what makes the
    predicate usable for a stage whose `dom` depends on the loop counter, such
    as RMSNorm's strided store or softmax's chunk sweep. -/
def EWStmt.StoresWithin (b : Buf) (P : Nat → Prop) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) : Nat → EWStmt → Prop
  | i, .seq x y          => StoresWithin b P cta ir im i x ∧ StoresWithin b P cta ir im i y
  | i, .storeLane0 c ix _ => c = b → P (ix.eval cta i ⟨0, by decide⟩ ir im)
  | i, .storeLane c ix _  => c = b → ∀ l, P (ix.eval cta i l ir im)
  | _, .forN n body      => ∀ j, j < n → StoresWithin b P cta ir im j body
  | _, .forM bu a body   => ∀ j, j < im bu a → StoresWithin b P cta ir im j body
  | i, .iif _ x y        => StoresWithin b P cta ir im i x ∧ StoresWithin b P cta ir im i y
  | i, .iwhile _ _ _ body => StoresWithin b P cta ir im i body
  | i, .locals _ body => StoresWithin b P cta ir im i body
  | _, _                 => True

/-- **A kernel that never writes `b` stores within anything.**

    The prefix of a two-phase kernel — the reduction, the scale computation — is
    all register traffic, and this makes its obligation vanish rather than
    unfold into a nest of `True`s.  Decidable, so the caller writes `by decide`. -/
theorem EWStmt.storesWithin_of_notWrites (b : Buf) (P : Nat → Prop) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ (s : EWStmt) (i : Nat), s.WritesBufB b = false → s.StoresWithin b P cta ir im i := by
  intro s
  induction s with
  | seq x y ihx ihy =>
      intro i h
      have h' := Bool.or_eq_false_iff.mp h
      exact ⟨ihx i h'.1, ihy i h'.2⟩
  | storeLane0 c _ _ =>
      intro _ h hc
      exact absurd (h.symm.trans (decide_eq_true hc)) (by simp)
  | storeLane c _ _  =>
      intro _ h hc
      exact absurd (h.symm.trans (decide_eq_true hc)) (by simp)
  | forN _ body ih   => intro _ h j _; exact ih j h
  | forM _ _ body ih => intro _ h j _; exact ih j h
  | iif _ x y ihx ihy =>
      intro i h
      have h' := Bool.or_eq_false_iff.mp h
      exact ⟨ihx i h'.1, ihy i h'.2⟩
  | iwhile _ _ _ body ih => intro i h; exact ih i h
  | locals _ body ih => intro i h; exact ih i h
  | _ => intro _ _; trivial

/-- **A kernel leaves untouched every address of its output buffer that it does
    not claim.**  With `P := S.dom cta` this is exactly `StageSpec.frame`'s
    second disjunct. -/
theorem EWStmt.run_otherAddr (b : Buf) (P : Nat → Prop) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (a : Nat) (ha : ¬ P a) :
    ∀ (s : EWStmt) (i : Nat) (st : WSt R), s.StoresWithin b P cta ir im i →
      (s.run st cta i ir im).mem b a = st.mem b a := by
  intro s
  induction s with
  | skip => intro _ _ _; rfl
  | seq x y ihx ihy =>
      intro i st h
      show (y.run (x.run st cta i ir im) cta i ir im).mem b a = _
      rw [ihy i _ h.2, ihx i st h.1]
  | setR _ _ => intro _ _ _; rfl
  | shflXor _ _ _ => intro _ _ _; rfl
  | loadIdx _ _ _ => intro _ _ _; rfl
  | loadV4 _ _ _ _ _ _ => intro _ _ _; rfl
  | storeLane0 c ix r =>
      intro i st h
      show (st.store1 c (ix.eval cta i ⟨0, by decide⟩ ir im)
              (st.regs r ⟨0, by decide⟩)).mem b a = _
      show (if b = c ∧ a = _ then _ else _) = _
      refine if_neg (fun hcon => ?_)
      exact ha (hcon.2 ▸ h hcon.1.symm)
  | storeLane c ix r =>
      intro i st h
      by_cases hcb : c = b
      · subst hcb
        exact storeLane_otherAddr c (fun l => ix.eval cta i l ir im) r st a
          (fun l hcon => ha (hcon ▸ h rfl l))
      · exact congrFun (storeLane_otherBuf c (fun l => ix.eval cta i l ir im) r st b
          (fun hcon => hcb hcon.symm)) a
  | stSm _ _ => intro _ _ _; rfl
  | ldSm _ _ => intro _ _ _; rfl
  | barrier => intro _ _ _; rfl
  | forN n body ih =>
      intro _ st h
      exact foldl_otherAddr _ b a (List.range n) (fun j hj s' => ih j s' (h j (List.mem_range.mp hj))) st
  | forM bu c body ih =>
      intro _ st h
      exact foldl_otherAddr _ b a (List.range (im bu c))
        (fun j hj s' => ih j s' (h j (List.mem_range.mp hj))) st
  | cvtIF _ _ => intro _ _ _; rfl
  | iif c x y ihx ihy =>
      intro i st h
      simp only [EWStmt.run]
      split
      · exact ihx i st h.1
      · exact ihy i st h.2
  | iwhile n c cs body ih =>
      intro i st h
      exact iwhile_inv n c cs body cta i ir im st (fun s => s.mem b a = st.mem b a)
        (fun s hs => (ih i s h).trans hs) rfl
  | locals rs body ih => intro i st h; exact ih i st h
  | _ => intro _ _ _; rfl

-- ---------------------------------------------------------------------------
-- The frame field, assembled
-- ---------------------------------------------------------------------------

/-- **`StageSpec.frame`, for an arbitrary emittable kernel.**

    Two hypotheses, both about syntax alone:

    * `hb` — the kernel names no buffer but `out` (`decide`-able);
    * `hd` — its stores to `out` land in `dom cta` (the one arithmetic fact,
      and the same one the stage's `value` field needs anyway).

    Every stage below is built with this rather than a bespoke induction. -/
theorem EWStmt.frame_of (s : EWStmt) (out : Buf) (dom : Nat → Nat → Prop)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (hb : ∀ c ∈ s.wbufs, c = out)
    (hd : ∀ cta, s.StoresWithin out (dom cta) cta ir im 0) :
    ∀ (cta : Nat) (st : WSt R) (b : Buf) (a : Nat), (b ≠ out ∨ ¬ dom cta a) →
      (s.run st cta 0 ir im).mem b a = st.mem b a := by
  intro cta st b a h
  rcases h with h | h
  · exact congrFun (EWStmt.run_otherBuf b cta _ _ s
      (fun c hc => by rw [hb c hc]; exact fun hcon => h hcon.symm) 0 st) a
  · by_cases hbo : b = out
    · subst hbo
      exact EWStmt.run_otherAddr b (dom cta) cta _ _ a h s 0 st (hd cta)
    · exact congrFun (EWStmt.run_otherBuf b cta _ _ s
        (fun c hc => by rw [hb c hc]; exact fun hcon => hbo hcon.symm) 0 st) a

-- ---------------------------------------------------------------------------
-- The stage, assembled
-- ---------------------------------------------------------------------------

/-- **A pipeline stage from an arbitrary hand-written kernel.**

    `mapStage`, `reduceStage` and `outerStage` each build a stage from a
    *generated* kernel and discharge all four fields internally.  That covers
    the kernels a schedule produces, and nothing else: a kernel written straight
    in `EWStmt` — every interesting one in a transformer — had no route to
    becoming a stage, because `frame` needed a bespoke induction each time.

    Here `frame` comes from `EWStmt.frame_of`, so what the author supplies is:

    * `hb`, `hd` — syntax.  `hb` is `decide`-able; `hd` is the address
      arithmetic the kernel's own store theorem already needs.
    * `hv` — where the value lands, which *is* the kernel's `_stores` theorem.
    * `hvo` — the one genuinely semantic obligation: the value must not depend
      on parts of the output buffer this block does not own.

    Two of the four fields stop being work. -/
def stageOfEW (s : EWStmt) (out : Buf) (g : Nat) (dm : Nat → Nat → Prop)
    (vl : (Buf → Nat → R) → Nat → Nat → R)
    (ir : Nat → Lane → Nat := fun _ _ => 0) (im : Buf → Nat → Nat := fun _ _ => 0)
    (hb : ∀ c ∈ s.wbufs, c = out)
    (hd : ∀ cta, s.StoresWithin out (dm cta) cta ir im 0)
    (hv : ∀ (cta : Nat) (st : WSt R) (a : Nat), dm cta a →
        (s.run st cta 0 ir im).mem out a = vl st.mem cta a)
    (hvo : ∀ (m m' : Buf → Nat → R) (cta a : Nat), dm cta a →
        (∀ b, b ≠ out → m b = m' b) → (∀ a', dm cta a' → m out a' = m' out a') →
        vl m cta a = vl m' cta a) : (StageSpec R) where
  ew      := s
  iregs   := ir
  imem    := im
  grid    := g
  out     := out
  dom     := dm
  val     := vl
  frame   := EWStmt.frame_of s out dm ir im hb hd
  value   := hv
  valOnly := hvo

@[simp] theorem stageOfEW_ew (s out g dm vl ir im hb hd hv hvo) :
    (stageOfEW (R := R) s out g dm vl ir im hb hd hv hvo).ew = s := rfl
@[simp] theorem stageOfEW_grid (s out g dm vl ir im hb hd hv hvo) :
    (stageOfEW (R := R) s out g dm vl ir im hb hd hv hvo).grid = g := rfl
@[simp] theorem stageOfEW_out (s out g dm vl ir im hb hd hv hvo) :
    (stageOfEW (R := R) s out g dm vl ir im hb hd hv hvo).out = out := rfl
@[simp] theorem stageOfEW_dom (s out g dm vl ir im hb hd hv hvo) :
    (stageOfEW (R := R) s out g dm vl ir im hb hd hv hvo).dom = dm := rfl
@[simp] theorem stageOfEW_val (s out g dm vl ir im hb hd hv hvo) :
    (stageOfEW (R := R) s out g dm vl ir im hb hd hv hvo).val = vl := rfl

/-- **A value that reads only buffers other than `out` satisfies `valOnly`.**

    The common case: an out-of-place kernel.  Supplied so the author of such a
    stage writes `readsOnly …` instead of repeating the same three-line
    argument. -/
theorem valOnly_of_indep {out : Buf}
    (vl : (Buf → Nat → R) → Nat → Nat → R)
    (h : ∀ (m m' : Buf → Nat → R) (cta a : Nat),
      (∀ b, b ≠ out → m b = m' b) → vl m cta a = vl m' cta a)
    (dm : Nat → Nat → Prop) :
    ∀ (m m' : Buf → Nat → R) (cta a : Nat), dm cta a →
      (∀ b, b ≠ out → m b = m' b) → (∀ a', dm cta a' → m out a' = m' out a') →
      vl m cta a = vl m' cta a :=
  fun m m' cta a _ hb _ => h m m' cta a hb

end Carrier

end AlgorithmLib.ML
