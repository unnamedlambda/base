import AlgorithmLib.ML.Geometry
import AlgorithmLib.ML.Pipeline

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

  Nothing here is specific to a model or a kernel.  The leverage is that the
  eleven Qwen2 kernels stop needing eleven frame proofs.
-/

namespace AlgorithmLib.ML

-- ---------------------------------------------------------------------------
-- Loop framing
-- ---------------------------------------------------------------------------

/-- A loop preserves whatever every one of its bodies preserves — buffer
    version.  Stated over the elaborated `WStmt` because that is what `run`
    consumes; the `EWStmt` cases below feed it. -/
theorem wforN_otherBuf (f : Nat → WStmt) (b : Buf)
    (h : ∀ j (st : WSt), ((f j).run st).mem b = st.mem b) :
    ∀ (n : Nat) (st : WSt), ((WStmt.forN n f).run st).mem b = st.mem b := by
  intro n
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
      intro st
      show ((List.range (n + 1)).foldl (fun s' i => (f i).run s') st).mem b = _
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      rw [h n _]
      exact ih st

/-- …and the address version, where the hypothesis need only hold for the
    iterations the loop actually runs. -/
theorem wforN_otherAddr (f : Nat → WStmt) (b : Buf) (a : Nat) :
    ∀ (n : Nat), (∀ j, j < n → ∀ (st : WSt), ((f j).run st).mem b a = st.mem b a) →
      ∀ (st : WSt), ((WStmt.forN n f).run st).mem b a = st.mem b a := by
  intro n
  induction n with
  | zero => intro _ _; rfl
  | succ n ih =>
      intro h st
      show ((List.range (n + 1)).foldl (fun s' i => (f i).run s') st).mem b a = _
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      rw [h n (by omega) _]
      exact ih (fun j hj => h j (by omega)) st

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
  | _ => intro h; exact absurd h (by simp [EWStmt.WritesBufB])

/-- **A kernel leaves untouched every buffer it does not name.**

    Proven by structural induction over the whole emittable language, so it
    holds for hand-written `EWStmt` kernels — the two-phase reductions, the
    paired stores — and not only for `compileWKernel` output. -/
theorem EWStmt.run_otherBuf (b : Buf) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ (s : EWStmt), (∀ c ∈ s.wbufs, c ≠ b) →
      ∀ (i : Nat) (st : WSt), ((s.elabAt cta i ir im).run st).mem b = st.mem b := by
  intro s
  induction s with
  | skip => intro _ _ _; rfl
  | seq x y ihx ihy =>
      intro h i st
      show ((y.elabAt cta i ir im).run ((x.elabAt cta i ir im).run st)).mem b = _
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
      exact wforN_otherBuf _ b (fun j s' => ih h j s') n st
  | forM bu a body ih =>
      intro h _ st
      exact wforN_otherBuf _ b (fun j s' => ih h j s') (im bu a) st
  | cvtIF _ _ => intro _ _ _; rfl

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
  | _                     => []

theorem wforN_otherReg (f : Nat → WStmt) (r : Nat)
    (h : ∀ j (st : WSt), ((f j).run st).regs r = st.regs r) :
    ∀ (n : Nat) (st : WSt), ((WStmt.forN n f).run st).regs r = st.regs r := by
  intro n
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
      intro st
      show ((List.range (n + 1)).foldl (fun s' i => (f i).run s') st).regs r = _
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, h n _]
      exact ih st

/-- **A kernel leaves untouched every register it does not assign.** -/
theorem EWStmt.run_otherReg (r : Nat) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ (s : EWStmt), (∀ d ∈ s.wregs, d ≠ r) →
      ∀ (i : Nat) (st : WSt), ((s.elabAt cta i ir im).run st).regs r = st.regs r := by
  intro s
  induction s with
  | skip => intro _ _ _; rfl
  | seq x y ihx ihy =>
      intro h i st
      show ((y.elabAt cta i ir im).run ((x.elabAt cta i ir im).run st)).regs r = _
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
      exact wforN_otherReg _ r (fun j s' => ih h j s') n st
  | forM bu a body ih =>
      intro h _ st
      exact wforN_otherReg _ r (fun j s' => ih h j s') (im bu a) st

-- ---------------------------------------------------------------------------
-- Which addresses a kernel can write
-- ---------------------------------------------------------------------------

/-- **Every store to `b` lands inside `P`.**

    The recursion mirrors `elabAt`, so a loop's obligation is stated at the
    iteration index the body is elaborated at — which is what makes the
    predicate usable for a stage whose `dom` depends on the loop counter, such
    as RMSNorm's strided store or softmax's chunk sweep. -/
def EWStmt.StoresWithin (b : Buf) (P : Nat → Prop) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) : Nat → EWStmt → Prop
  | i, .seq x y          => StoresWithin b P cta ir im i x ∧ StoresWithin b P cta ir im i y
  | i, .storeLane0 c ix _ => c = b → P (ix.eval cta i ⟨0, by decide⟩ ir im)
  | i, .storeLane c ix _  => c = b → ∀ l, P (ix.eval cta i l ir im)
  | _, .forN n body      => ∀ j, j < n → StoresWithin b P cta ir im j body
  | _, .forM bu a body   => ∀ j, j < im bu a → StoresWithin b P cta ir im j body
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
  | _ => intro _ _; trivial

/-- **A kernel leaves untouched every address of its output buffer that it does
    not claim.**  With `P := S.dom cta` this is exactly `StageSpec.frame`'s
    second disjunct. -/
theorem EWStmt.run_otherAddr (b : Buf) (P : Nat → Prop) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (a : Nat) (ha : ¬ P a) :
    ∀ (s : EWStmt) (i : Nat) (st : WSt), s.StoresWithin b P cta ir im i →
      ((s.elabAt cta i ir im).run st).mem b a = st.mem b a := by
  intro s
  induction s with
  | skip => intro _ _ _; rfl
  | seq x y ihx ihy =>
      intro i st h
      show ((y.elabAt cta i ir im).run ((x.elabAt cta i ir im).run st)).mem b a = _
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
      exact wforN_otherAddr _ b a n (fun j hj s' => ih j s' (h j hj)) st
  | forM bu c body ih =>
      intro _ st h
      exact wforN_otherAddr _ b a (im bu c) (fun j hj s' => ih j s' (h j hj)) st
  | cvtIF _ _ => intro _ _ _; rfl

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
    ∀ (cta : Nat) (st : WSt) (b : Buf) (a : Nat), (b ≠ out ∨ ¬ dom cta a) →
      ((s.elabAt cta 0 ir im).run st).mem b a = st.mem b a := by
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
    (vl : (Buf → Nat → Float32) → Nat → Nat → Float32)
    (ir : Nat → Lane → Nat := fun _ _ => 0) (im : Buf → Nat → Nat := fun _ _ => 0)
    (hb : ∀ c ∈ s.wbufs, c = out)
    (hd : ∀ cta, s.StoresWithin out (dm cta) cta ir im 0)
    (hv : ∀ (cta : Nat) (st : WSt) (a : Nat), dm cta a →
        ((s.elabAt cta 0 ir im).run st).mem out a = vl st.mem cta a)
    (hvo : ∀ (m m' : Buf → Nat → Float32) (cta a : Nat), dm cta a →
        (∀ b, b ≠ out → m b = m' b) → (∀ a', dm cta a' → m out a' = m' out a') →
        vl m cta a = vl m' cta a) : StageSpec where
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
    (stageOfEW s out g dm vl ir im hb hd hv hvo).ew = s := rfl
@[simp] theorem stageOfEW_grid (s out g dm vl ir im hb hd hv hvo) :
    (stageOfEW s out g dm vl ir im hb hd hv hvo).grid = g := rfl
@[simp] theorem stageOfEW_out (s out g dm vl ir im hb hd hv hvo) :
    (stageOfEW s out g dm vl ir im hb hd hv hvo).out = out := rfl
@[simp] theorem stageOfEW_dom (s out g dm vl ir im hb hd hv hvo) :
    (stageOfEW s out g dm vl ir im hb hd hv hvo).dom = dm := rfl
@[simp] theorem stageOfEW_val (s out g dm vl ir im hb hd hv hvo) :
    (stageOfEW s out g dm vl ir im hb hd hv hvo).val = vl := rfl

/-- **A value that reads only buffers other than `out` satisfies `valOnly`.**

    The common case: an out-of-place kernel.  Supplied so the author of such a
    stage writes `readsOnly …` instead of repeating the same three-line
    argument. -/
theorem valOnly_of_indep {out : Buf}
    (vl : (Buf → Nat → Float32) → Nat → Nat → Float32)
    (h : ∀ (m m' : Buf → Nat → Float32) (cta a : Nat),
      (∀ b, b ≠ out → m b = m' b) → vl m cta a = vl m' cta a)
    (dm : Nat → Nat → Prop) :
    ∀ (m m' : Buf → Nat → Float32) (cta a : Nat), dm cta a →
      (∀ b, b ≠ out → m b = m' b) → (∀ a', dm cta a' → m out a' = m' out a') →
      vl m cta a = vl m' cta a :=
  fun m m' cta a _ hb _ => h m m' cta a hb

end AlgorithmLib.ML
