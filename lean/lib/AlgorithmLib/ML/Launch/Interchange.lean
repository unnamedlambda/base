module
public import AlgorithmLib.ML.Compose
meta import AlgorithmLib.ML.Compose
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # A fused launch is the members' pipeline

  A ViT launch is not one operation.  It is `n` of them joined by `.seq` over
  one shared buffer table, launched once over one grid — which is where the
  speed comes from, and which nothing in `Pipeline.lean` could talk about: a
  `StageSpec` has exactly one output, so `runGrid`, `Exclusive` and `denote`
  are single-output by construction.

  The way out is not a multi-output stage.  It is to say that the fused launch
  *is* the members' pipeline:

      one launch of `s₁ ; … ; sₙ` over blocks `0 … g-1`
        =  `s₁` over every block, then `s₂` over every block, …

  which is a **loop interchange**, and true exactly when different blocks do not
  interfere.  Every existing theorem then applies unchanged, because the right
  side is a `Pipeline` of ordinary single-output stages.

  The interference condition is one hypothesis — that two updates from
  *different blocks* commute — and `commute_of_footprints` reduces it to three
  checkable facts: the two blocks write disjoint places, and neither block's
  value is disturbed by the other's writes.  That last pair is what
  "chunk-local" means.
-/

namespace AlgorithmLib.ML

open Classical in
/-- **What one block of a stage does to memory.**

    `StageSpec.step` is the whole grid; this is one block of it, and it is the
    granularity an interchange argument needs. -/
noncomputable def StageSpec.applyAt (S : StageSpec) (cta : Nat)
    (m : Buf → Nat → Float32) : Buf → Nat → Float32 :=
  fun b a => if b = S.out ∧ S.dom cta a then S.val m cta a else m b a

/-- Off a block's own output addresses, `applyAt` is the identity. -/
theorem StageSpec.applyAt_off (S : StageSpec) (cta : Nat) (m : Buf → Nat → Float32)
    (b : Buf) (a : Nat) (h : ¬ (b = S.out ∧ S.dom cta a)) :
    S.applyAt cta m b a = m b a := by
  rw [StageSpec.applyAt, if_neg h]

/-- The stage's contract, read as a function on memory: `frame` says where it
    does not write and `value` says what it puts where it does, and between
    them the resulting memory is determined by the entering memory alone —
    registers included in the state but not in the answer. -/
theorem StageSpec.run_applyAt (S : StageSpec) (cta : Nat) (st : WSt) :
    ((S.blk cta).run st).mem = S.applyAt cta st.mem := by
  funext b a
  by_cases h : b = S.out ∧ S.dom cta a
  · obtain ⟨hb, hd⟩ := h
    subst hb
    rw [S.valueB cta st a hd, StageSpec.applyAt, if_pos ⟨rfl, hd⟩]
  · rw [S.frameB cta st b a
          ((Classical.em (b = S.out)).elim (fun hb => Or.inr (fun hd => h ⟨hb, hd⟩)) Or.inl),
        StageSpec.applyAt, if_neg h]

-- ---------------------------------------------------------------------------
-- The fused block
-- ---------------------------------------------------------------------------

/-- **The block a fused launch runs**: the members in order, joined by `.seq`.

    The same shape as `TOp.groupStmt` — a left fold from `.skip` — because
    `EWStmt.elabAt` distributes over `.seq`, so elaborating a group's statement
    at block `cta` is exactly this. -/
def groupBlk (ss : List StageSpec) (cta : Nat) : WStmt :=
  ss.foldl (fun acc S => .seq acc (S.blk cta)) .skip

/-- What the members do to memory, in order, at one block. -/
noncomputable def blockApply (ss : List StageSpec) (cta : Nat)
    (m : Buf → Nat → Float32) : Buf → Nat → Float32 :=
  ss.foldl (fun mm S => S.applyAt cta mm) m

theorem groupBlk_run_aux (cta : Nat) : ∀ (ss : List StageSpec) (p : WStmt) (st : WSt),
    ((ss.foldl (fun acc S => .seq acc (S.blk cta)) p).run st)
      = ss.foldl (fun s S => (S.blk cta).run s) (p.run st) := by
  intro ss
  induction ss with
  | nil => intro _ _; rfl
  | cons S rest ih =>
      intro p st
      show ((rest.foldl _ (WStmt.seq p (S.blk cta))).run st) = _
      rw [ih (WStmt.seq p (S.blk cta)) st]
      rfl

theorem foldl_run_mem (cta : Nat) : ∀ (ss : List StageSpec) (st : WSt),
    (ss.foldl (fun s S => (S.blk cta).run s) st).mem = blockApply ss cta st.mem := by
  intro ss
  induction ss with
  | nil => intro _; rfl
  | cons S rest ih =>
      intro st
      show (rest.foldl _ ((S.blk cta).run st)).mem
          = rest.foldl (fun mm T => T.applyAt cta mm) (S.applyAt cta st.mem)
      rw [ih ((S.blk cta).run st), blockApply, S.run_applyAt cta st]

/-- **One block of the fused launch does what its members do, in order.** -/
theorem groupBlk_mem (ss : List StageSpec) (cta : Nat) (st : WSt) :
    ((groupBlk ss cta).run st).mem = blockApply ss cta st.mem := by
  rw [groupBlk, groupBlk_run_aux cta ss WStmt.skip st, foldl_run_mem cta ss]
  rfl

/-- A whole grid, as a fold on memory. -/
theorem runGrid_mem (blk : Nat → WStmt) (g : Nat)
    (f : Nat → (Buf → Nat → Float32) → (Buf → Nat → Float32))
    (hf : ∀ cta st, ((blk cta).run st).mem = f cta st.mem) : ∀ (st : WSt),
    (runGrid blk g st).mem = (List.range g).foldl (fun m cta => f cta m) st.mem := by
  have key : ∀ (l : List Nat) (st : WSt),
      (l.foldl (fun s cta => (blk cta).run s) st).mem
        = l.foldl (fun m cta => f cta m) st.mem := by
    intro l
    induction l with
    | nil => intro _; rfl
    | cons c t ih =>
        intro st
        show (t.foldl _ ((blk c).run st)).mem = t.foldl _ (f c st.mem)
        rw [ih ((blk c).run st), hf c st]
  intro st; exact key (List.range g) st

-- ---------------------------------------------------------------------------
-- When two blocks do not interfere
-- ---------------------------------------------------------------------------

/-- Two memory transformers that may be run in either order. -/
def Commute (f g : (Buf → Nat → Float32) → (Buf → Nat → Float32)) : Prop :=
  ∀ m, f (g m) = g (f m)

/-- **The checkable form of non-interference.**

    Block `c'` of `T` and block `c` of `U` commute when they write disjoint
    places and neither one's value is disturbed by the other's writes.  The
    second pair of conditions is what a grouping pass means by *chunk-local*:
    a block reads its own rows, so another block's writes are not among its
    inputs. -/
theorem commute_of_footprints (T U : StageSpec) (c' c : Nat)
    (hd : ∀ a, ¬ (T.out = U.out ∧ T.dom c' a ∧ U.dom c a))
    (hTU : ∀ m a, T.dom c' a → T.val (U.applyAt c m) c' a = T.val m c' a)
    (hUT : ∀ m a, U.dom c a → U.val (T.applyAt c' m) c a = U.val m c a) :
    Commute (T.applyAt c') (U.applyAt c) := by
  intro m
  funext b a
  by_cases hT : b = T.out ∧ T.dom c' a
  · have hU : ¬ (b = U.out ∧ U.dom c a) := by
      rintro ⟨hb, hu⟩
      exact hd a ⟨hT.1 ▸ hb ▸ rfl, hT.2, hu⟩
    rw [StageSpec.applyAt, if_pos hT, hTU m a hT.2]
    rw [StageSpec.applyAt, if_neg hU, StageSpec.applyAt, if_pos hT]
  · by_cases hU : b = U.out ∧ U.dom c a
    · rw [StageSpec.applyAt, if_neg hT, StageSpec.applyAt, if_pos hU]
      rw [StageSpec.applyAt, if_pos hU, hUT m a hU.2]
    · rw [StageSpec.applyAt, if_neg hT, StageSpec.applyAt, if_neg hU,
          StageSpec.applyAt, if_neg hU, StageSpec.applyAt, if_neg hT]

/-- A block commutes with a whole other block's worth of members. -/
theorem commute_blockApply (A : (Buf → Nat → Float32) → (Buf → Nat → Float32)) :
    ∀ (ss : List StageSpec) (c : Nat), (∀ U ∈ ss, Commute A (U.applyAt c)) →
      Commute A (blockApply ss c) := by
  intro ss
  induction ss with
  | nil => intro _ _ m; rfl
  | cons U rest ih =>
      intro c h m
      show A (rest.foldl (fun mm T => T.applyAt c mm) (U.applyAt c m))
          = rest.foldl (fun mm T => T.applyAt c mm) (U.applyAt c (A m))
      rw [← h U (by simp) m]
      exact ih c (fun T hT => h T (by simp [hT])) (U.applyAt c m)

-- ---------------------------------------------------------------------------
-- Where a block reads
-- ---------------------------------------------------------------------------

/-- **The addresses block `cta` computes its values from.**

    `StageSpec.valOnly` already says a stage's value ignores its own output
    buffer away from the block's own addresses.  This says the rest: which
    addresses it *does* depend on.  A stage carries no such field, because
    single-stage theorems never needed one — the interchange does, and it is
    the one thing standing between `commute_of_footprints` and a real group. -/
def StageSpec.ReadsIn (S : StageSpec) (R : Nat → Buf → Nat → Prop) : Prop :=
  ∀ (m m' : Buf → Nat → Float32) (cta a : Nat), S.dom cta a →
    (∀ b a', R cta b a' → m b a' = m' b a') → S.val m cta a = S.val m' cta a

/-- **Non-interference from read footprints.**

    The form a grouping pass can actually discharge: two blocks commute when
    they write disjoint places and neither one's *reads* meet the other's
    writes.  `commute_of_footprints` asks the same thing about values under an
    update; this asks it about addresses, which is what a syntactic
    chunk-locality check computes. -/
theorem commute_of_reads (T U : StageSpec) (RT RU : Nat → Buf → Nat → Prop)
    (hT : T.ReadsIn RT) (hU : U.ReadsIn RU) (c' c : Nat)
    (hd : ∀ a, ¬ (T.out = U.out ∧ T.dom c' a ∧ U.dom c a))
    (hTU : ∀ b a, RT c' b a → ¬ (b = U.out ∧ U.dom c a))
    (hUT : ∀ b a, RU c b a → ¬ (b = T.out ∧ T.dom c' a)) :
    Commute (T.applyAt c') (U.applyAt c) := by
  refine commute_of_footprints T U c' c hd ?_ ?_
  · intro m a hdom
    exact hT (U.applyAt c m) m c' a hdom
      (fun b a' hr => U.applyAt_off c m b a' (hTU b a' hr))
  · intro m a hdom
    exact hU (T.applyAt c' m) m c a hdom
      (fun b a' hr => T.applyAt_off c' m b a' (hUT b a' hr))

/-- **A row pass reads its two operands and nothing else.**

    Its `val` is `g (m bA (evA cta d)) (m bB (evB cta d))` with `d` the offset
    inside the block's own segment, so the footprint is read off the definition
    rather than argued for.  Whether those addresses are inside the block's
    chunk is a question about `evA`/`evB` — about the schedule — and that is
    the question a grouping pass decides. -/
theorem zipRowStage_readsIn (bA bB out : Buf) (f : WFExp)
    (g : Float32 → Float32 → Float32) (ixA ixB : IdxE) (evA evB : Nat → Nat → Nat)
    (n off K grid : Nat) (hw hf hA hB hAo hBo) :
    (zipRowStage bA bB out f g ixA ixB evA evB n off K grid hw hf hA hB hAo hBo).ReadsIn
      (fun cta b a' => (b = bA ∧ ∃ d, a' = evA cta d) ∨ (b = bB ∧ ∃ d, a' = evB cta d)) := by
  intro m m' cta a _ h
  show g _ _ = g _ _
  rw [h bA _ (Or.inl ⟨rfl, _, rfl⟩), h bB _ (Or.inr ⟨rfl, _, rfl⟩)]

/-- **A dot-product stage reads its two operands and nothing else.** -/
theorem reduceStage_readsIn (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (K grid : Nat)
    (hAo hBo) :
    (reduceStage bA bB ixA ixB out K grid hAo hBo).ReadsIn
      (fun cta b a' => (b = bA ∧ ∃ i l, a' = ixA.eval cta i l (fun _ _ => 0) (fun _ _ => 0))
                     ∨ (b = bB ∧ ∃ i l, a' = ixB.eval cta i l (fun _ _ => 0) (fun _ _ => 0))) := by
  intro m m' cta a _ h
  show bflyFold _ _ = bflyFold _ _
  congr 1
  funext l
  exact dotStridedLane_congr _ _ _ _ _ _ K l
    (fun i _ l' => h bA _ (Or.inl ⟨rfl, i, l', rfl⟩))
    (fun i _ l' => h bB _ (Or.inr ⟨rfl, i, l', rfl⟩))

/-- A three-input row pass, likewise. -/
theorem zipRow3Stage_readsIn (bA bB bC out : Buf) (f : WFExp)
    (g : Float32 → Float32 → Float32 → Float32) (ixA ixB ixC : IdxE)
    (evA evB evC : Nat → Nat → Nat) (n off K grid : Nat)
    (hw hf hA hB hC hAo hBo hCo) :
    (zipRow3Stage bA bB bC out f g ixA ixB ixC evA evB evC n off K grid
        hw hf hA hB hC hAo hBo hCo).ReadsIn
      (fun cta b a' => (b = bA ∧ ∃ d, a' = evA cta d) ∨ (b = bB ∧ ∃ d, a' = evB cta d)
                     ∨ (b = bC ∧ ∃ d, a' = evC cta d)) := by
  intro m m' cta a _ h
  show g _ _ _ = g _ _ _
  rw [h bA _ (Or.inl ⟨rfl, _, rfl⟩), h bB _ (Or.inr (Or.inl ⟨rfl, _, rfl⟩)),
      h bC _ (Or.inr (Or.inr ⟨rfl, _, rfl⟩))]

/-- A four-input row pass, likewise. -/
theorem zipRow4Stage_readsIn (bA bB bC bD out : Buf) (f : WFExp)
    (g : Float32 → Float32 → Float32 → Float32 → Float32) (ixA ixB ixC ixD : IdxE)
    (evA evB evC evD : Nat → Nat → Nat) (n off K grid : Nat)
    (hw hf hA hB hC hD hAo hBo hCo hDo) :
    (zipRow4Stage bA bB bC bD out f g ixA ixB ixC ixD evA evB evC evD n off K grid
        hw hf hA hB hC hD hAo hBo hCo hDo).ReadsIn
      (fun cta b a' => (b = bA ∧ ∃ d, a' = evA cta d) ∨ (b = bB ∧ ∃ d, a' = evB cta d)
                     ∨ (b = bC ∧ ∃ d, a' = evC cta d) ∨ (b = bD ∧ ∃ d, a' = evD cta d)) := by
  intro m m' cta a _ h
  show g _ _ _ _ = g _ _ _ _
  rw [h bA _ (Or.inl ⟨rfl, _, rfl⟩), h bB _ (Or.inr (Or.inl ⟨rfl, _, rfl⟩)),
      h bC _ (Or.inr (Or.inr (Or.inl ⟨rfl, _, rfl⟩))),
      h bD _ (Or.inr (Or.inr (Or.inr ⟨rfl, _, rfl⟩)))]

/-- **A batched contraction reads its two operands over every member's walk.**

    The footprint is over-approximated in the batch index — a `ReadsIn` set is
    indexed by the block, not by the address, and the member a block handles is
    recovered from the address.  Over-approximating is the safe direction: it
    makes the separation the caller has to prove stronger, never weaker. -/
theorem dotBatchedStage_readsIn (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE)
    (out : Buf) (B K grid : Nat) (hg hAo hBo) :
    (dotBatchedStage bA bB ixA ixB out B K grid hg hAo hBo).ReadsIn
      (fun cta bf a' =>
        (bf = bA ∧ ∃ i l, a' = ixA.eval cta i l (fun _ _ => 0) (fun _ _ => 0))
      ∨ (bf = bB ∧ ∃ s i l, a' = (ixB s).eval cta i l (fun _ _ => 0) (fun _ _ => 0))) := by
  intro m m' cta a _ h
  show bflyFold _ _ = bflyFold _ _
  congr 1
  funext l
  exact dotStridedLane_congr _ _ _ _ _ _ K l
    (fun i _ l' => h bA _ (Or.inl ⟨rfl, i, l', rfl⟩))
    (fun i _ l' => h bB _ (Or.inr ⟨rfl, _, i, l', rfl⟩))

/-- An outer product, likewise — over-approximated in the same index. -/
theorem outerBatchedStage_readsIn (bA bB out : Buf) (ixA ixB : Nat → IdxE)
    (n B K grid : Nat) (hn hAo hBo) :
    (outerBatchedStage bA bB out ixA ixB n B K grid hn hAo hBo).ReadsIn
      (fun cta bf a' =>
        (bf = bA ∧ ∃ s i l, a' = (ixA s).eval cta i l (fun _ _ => 0) (fun _ _ => 0))
      ∨ (bf = bB ∧ ∃ s i l, a' = (ixB s).eval cta i l (fun _ _ => 0) (fun _ _ => 0))) := by
  intro m m' cta a _ h
  exact dotStridedLane_congr _ _ _ _ _ _ B _
    (fun s _ l' => h bA _ (Or.inl ⟨rfl, s, _, l', rfl⟩))
    (fun s _ l' => h bB _ (Or.inr ⟨rfl, s, _, l', rfl⟩))

/-- A strided maximum reads its operand at its own walk's addresses. -/
theorem maxStridedLane_congr (mem mem' : Nat → Float32) (f : Nat → Lane → Nat)
    (K : Nat) (init : Float32) (l : Lane)
    (h : ∀ i, i < K → ∀ l', mem (f i l') = mem' (f i l')) :
    maxStridedLane mem f K init l = maxStridedLane mem' f K init l := by
  show (List.range K).foldl _ _ = (List.range K).foldl _ _
  have key : ∀ (L : List Nat), (∀ i ∈ L, i < K) → ∀ acc : Float32,
      L.foldl (fun acc i => NumOps.max acc (mem (f i l))) acc
        = L.foldl (fun acc i => NumOps.max acc (mem' (f i l))) acc := by
    intro L
    induction L with
    | nil => intro _ _; rfl
    | cons i t ih =>
        intro hmem acc
        show t.foldl _ (NumOps.max acc (mem (f i l))) = _
        rw [h i (hmem i (by simp)) l]
        exact ih (fun j hj => hmem j (by simp [hj])) _
  exact key (List.range K) (fun i hi => List.mem_range.mp hi) _

/-- A four-input strided contraction, likewise. -/
theorem dotStridedLane4_congr (memA memA' memB memB' memC memC' memD memD' : Nat → Float32)
    (fA fB fC fD : Nat → Lane → Nat) (g : Float32 → Float32 → Float32 → Float32)
    (K : Nat) (l : Lane)
    (hA : ∀ i, i < K → ∀ l', memA (fA i l') = memA' (fA i l'))
    (hB : ∀ i, i < K → ∀ l', memB (fB i l') = memB' (fB i l'))
    (hC : ∀ i, i < K → ∀ l', memC (fC i l') = memC' (fC i l'))
    (hD : ∀ i, i < K → ∀ l', memD (fD i l') = memD' (fD i l')) :
    dotStridedLane4 memA memB memC memD fA fB fC fD g K l
      = dotStridedLane4 memA' memB' memC' memD' fA fB fC fD g K l := by
  show (List.range K).foldl _ _ = (List.range K).foldl _ _
  have key : ∀ (L : List Nat), (∀ i ∈ L, i < K) → ∀ acc : Float32,
      L.foldl (fun acc i => NumOps.add acc
        (NumOps.mul (g (memA (fA i l)) (memB (fB i l)) (memC (fC i l))) (memD (fD i l)))) acc
        = L.foldl (fun acc i => NumOps.add acc
            (NumOps.mul (g (memA' (fA i l)) (memB' (fB i l)) (memC' (fC i l)))
              (memD' (fD i l)))) acc := by
    intro L
    induction L with
    | nil => intro _ _; rfl
    | cons i t ih =>
        intro hmem acc
        show t.foldl _ (NumOps.add acc (NumOps.mul (g _ _ _) _)) = _
        rw [hA i (hmem i (by simp)) l, hB i (hmem i (by simp)) l,
            hC i (hmem i (by simp)) l, hD i (hmem i (by simp)) l]
        exact ih (fun j hj => hmem j (by simp [hj])) _
  exact key (List.range K) (fun i hi => List.mem_range.mp hi) _

/-- A row maximum reads its one operand and nothing else. -/
theorem maxRowStage_readsIn (b : Buf) (ix : IdxE) (out : Buf) (K grid : Nat)
    (init : Float32) (hbo) :
    (maxRowStage b ix out K grid init hbo).ReadsIn
      (fun cta bf a' =>
        bf = b ∧ ∃ i l, a' = ix.eval cta i l (fun _ _ => 0) (fun _ _ => 0)) := by
  intro m m' cta a _ h
  refine congrFun (congrArg (bflyFoldOp (fun x c => NumOps.max x c)) ?_) _
  funext l
  exact maxStridedLane_congr _ _ _ K init l (fun i _ l' => h b _ ⟨rfl, i, l', rfl⟩)

/-- A four-input reduction, likewise. -/
theorem reduce4Stage_readsIn (bA bB bC bD : Buf) (ixA ixB ixC ixD : IdxE) (f : WFExp)
    (g : Float32 → Float32 → Float32 → Float32) (hf) (out : Buf) (K grid : Nat)
    (hAo hBo hCo hDo) :
    (reduce4Stage bA bB bC bD ixA ixB ixC ixD f g hf out K grid hAo hBo hCo hDo).ReadsIn
      (fun cta bf a' =>
        (bf = bA ∧ ∃ i l, a' = ixA.eval cta i l (fun _ _ => 0) (fun _ _ => 0))
      ∨ (bf = bB ∧ ∃ i l, a' = ixB.eval cta i l (fun _ _ => 0) (fun _ _ => 0))
      ∨ (bf = bC ∧ ∃ i l, a' = ixC.eval cta i l (fun _ _ => 0) (fun _ _ => 0))
      ∨ (bf = bD ∧ ∃ i l, a' = ixD.eval cta i l (fun _ _ => 0) (fun _ _ => 0))) := by
  intro m m' cta a _ h
  refine congrFun (congrArg bflyFold ?_) _
  funext l
  exact dotStridedLane4_congr _ _ _ _ _ _ _ _ _ _ _ _ g K l
    (fun i _ l' => h bA _ (Or.inl ⟨rfl, i, l', rfl⟩))
    (fun i _ l' => h bB _ (Or.inr (Or.inl ⟨rfl, i, l', rfl⟩)))
    (fun i _ l' => h bC _ (Or.inr (Or.inr (Or.inl ⟨rfl, i, l', rfl⟩))))
    (fun i _ l' => h bD _ (Or.inr (Or.inr (Or.inr ⟨rfl, i, l', rfl⟩))))

/-- An elementwise pass reads its inputs at the address it writes. -/
theorem mapStage_readsIn {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf)
    (grid : Nat) (hio) :
    (mapStage spec inB out grid hio).ReadsIn
      (fun _ bf a' => ∃ i, bf = inB i ∧ a' = a') := by
  intro m m' cta a _ h
  show denote _ spec = denote _ spec
  congr 1
  funext i
  exact h (inB i) a ⟨i, rfl, rfl⟩


-- ---------------------------------------------------------------------------
-- The interchange
-- ---------------------------------------------------------------------------

/-- **Moving one member out of the block loop.**

    Every block runs `A` then `G`; if `A` at one block commutes with `G` at any
    *other*, then all the `A`s can be run first.  `Nodup` is what says a block
    is not asked to commute with itself. -/
theorem foldl_pullout (A G : Nat → (Buf → Nat → Float32) → (Buf → Nat → Float32))
    (hc : ∀ c c', c ≠ c' → Commute (A c') (G c)) : ∀ (l : List Nat), l.Nodup →
    ∀ m, l.foldl (fun mm c => G c (A c mm)) m
        = l.foldl (fun mm c => G c mm) (l.foldl (fun mm c => A c mm) m) := by
  have move : ∀ (t : List Nat) (c : Nat), c ∉ t → ∀ m,
      t.foldl (fun mm c' => A c' mm) (G c m) = G c (t.foldl (fun mm c' => A c' mm) m) := by
    intro t
    induction t with
    | nil => intro _ _ _; rfl
    | cons c'' t' ih =>
        intro c hc'' m
        have hne : c'' ≠ c := fun h => hc'' (by simp [h])
        show t'.foldl _ (A c'' (G c m)) = G c (t'.foldl _ (A c'' m))
        rw [hc c c'' (fun h => hne h.symm) m]
        exact ih c (fun h => hc'' (by simp [h])) (A c'' m)
  intro l
  induction l with
  | nil => intro _ _; rfl
  | cons c t ih =>
      intro hnd m
      have hct : c ∉ t := (List.nodup_cons.mp hnd).1
      show t.foldl (fun mm c' => G c' (A c' mm)) (G c (A c m))
          = t.foldl (fun mm c' => G c' mm) (G c (t.foldl (fun mm c' => A c' mm) (A c m)))
      rw [ih (List.nodup_cons.mp hnd).2 (G c (A c m)), move t c hct (A c m)]

/-- A stage's whole grid, as the fold of its blocks. -/
theorem step_eq_foldl_applyAt (S : StageSpec) (hex : S.Exclusive)
    (m : Buf → Nat → Float32) :
    (List.range S.grid).foldl (fun mm cta => S.applyAt cta mm) m = S.step m := by
  let st : WSt := ⟨fun _ _ => NumOps.ofNat 0, m, fun _ => NumOps.ofNat 0⟩
  have h1 : (runGrid S.blk S.grid st).mem
      = (List.range S.grid).foldl (fun mm cta => S.applyAt cta mm) st.mem :=
    runGrid_mem S.blk S.grid (fun cta mm => S.applyAt cta mm)
      (fun cta s => S.run_applyAt cta s) st
  exact h1.symm.trans (runGrid_step S hex st)

/-- **A fused launch is the members' pipeline.**

    One launch of `s₁ ; … ; sₙ` over `g` blocks lands the memory that running
    each member over all `g` blocks, in order, lands — so everything stated of
    a `Pipeline` of single-output stages is available for a launch that is not
    one.  The hypothesis is the whole of what fusion needs: two updates from
    *different blocks* commute.  Same-block order is untouched, which is why
    a chain of row passes may be fused and a cross-block reduction may not. -/
theorem group_runs_as_pipeline : ∀ (ss : List StageSpec) (g : Nat),
    (∀ S ∈ ss, S.grid = g) → (∀ S ∈ ss, S.Exclusive) →
    (∀ T ∈ ss, ∀ U ∈ ss, ∀ c c' : Nat, c ≠ c' → Commute (T.applyAt c') (U.applyAt c)) →
    ∀ (st : WSt), (runGrid (groupBlk ss) g st).mem = (Pipeline.mk ss).denote st.mem := by
  intro ss
  induction ss with
  | nil =>
      intro g _ _ _ st
      rw [runGrid_mem (groupBlk []) g (fun cta m => blockApply [] cta m)
            (fun cta s => groupBlk_mem [] cta s) st]
      show (List.range g).foldl (fun m _ => m) st.mem = st.mem
      induction (List.range g) with
      | nil => rfl
      | cons _ t ih => exact ih
  | cons S rest ih =>
      intro g hg hex hc st
      rw [runGrid_mem (groupBlk (S :: rest)) g (fun cta m => blockApply (S :: rest) cta m)
            (fun cta s => groupBlk_mem (S :: rest) cta s) st]
      have hsplit : ∀ (m : Buf → Nat → Float32),
          (List.range g).foldl (fun m cta => blockApply (S :: rest) cta m) m
            = (List.range g).foldl (fun m cta => blockApply rest cta m)
                ((List.range g).foldl (fun m cta => S.applyAt cta m) m) := by
        intro m
        refine foldl_pullout (fun cta => S.applyAt cta) (fun cta => blockApply rest cta)
          ?_ (List.range g) (List.nodup_range) m
        intro c c' hne
        exact commute_blockApply (S.applyAt c') rest c
          (fun U hU => hc S (by simp) U (by simp [hU]) c c' hne)
      rw [hsplit st.mem]
      have hrest : (List.range g).foldl (fun m cta => blockApply rest cta m)
          ((List.range g).foldl (fun m cta => S.applyAt cta m) st.mem)
            = (Pipeline.mk rest).denote
                ((List.range g).foldl (fun m cta => S.applyAt cta m) st.mem) := by
        have := ih g (fun T hT => hg T (by simp [hT])) (fun T hT => hex T (by simp [hT]))
          (fun T hT U hU => hc T (by simp [hT]) U (by simp [hU]))
          ⟨fun _ _ => NumOps.ofNat 0,
           (List.range g).foldl (fun m cta => S.applyAt cta m) st.mem,
           fun _ => NumOps.ofNat 0⟩
        rw [runGrid_mem (groupBlk rest) g (fun cta m => blockApply rest cta m)
              (fun cta s => groupBlk_mem rest cta s)] at this
        exact this
      rw [hrest, ← hg S (by simp), step_eq_foldl_applyAt S (hex S (by simp)) st.mem]
      rfl

-- ---------------------------------------------------------------------------
-- The hypotheses are satisfiable
-- ---------------------------------------------------------------------------

/-! A theorem whose hypothesis cannot be met is true and says nothing, and this
    development has shipped one before.  So the commutation hypothesis is
    discharged here for the kernels fusion is actually applied to: two
    elementwise passes over one grid, including the case that makes fusion
    worth doing — the second reading the buffer the first wrote. -/

/-- **Two fused elementwise passes commute across blocks.**

    `mapStage`'s block owns `cta·32 + lane` and reads its inputs at the *same*
    address, so a block's inputs are inside its own chunk.  That is the whole
    argument, and it survives `gB i = o1`: a member reading the previous
    member's output reads only the rows its own block just wrote. -/
theorem mapStage_commute {Γ Δ : Nat} (f : Expr Γ) (g : Expr Δ)
    (fB : Fin Γ → Buf) (gB : Fin Δ → Buf) (o1 o2 : Buf) (g1 g2 : Nat)
    (h1 : ∀ i, fB i ≠ o1) (h2 : ∀ i, gB i ≠ o2)
    (c c' : Nat) (hne : c ≠ c') :
    Commute ((mapStage f fB o1 g1 h1).applyAt c') ((mapStage g gB o2 g2 h2).applyAt c) := by
  have hchunk : ∀ (x y : Nat) (a : Nat), x ≠ y →
      (∃ l : Lane, x * 32 + l.val = a) → (∃ l : Lane, y * 32 + l.val = a) → False := by
    rintro x y a hxy ⟨l, hl⟩ ⟨l', hl'⟩
    exact hxy (elemIx_blocks_disjoint x y l l' (hl.trans hl'.symm))
  refine commute_of_footprints _ _ c' c ?_ ?_ ?_
  · rintro a ⟨-, hT, hU⟩
    exact hchunk c' c a hne.symm hT hU
  · intro m a hd
    show denote (fun i => (mapStage g gB o2 g2 h2).applyAt c m (fB i) a) f
        = denote (fun i => m (fB i) a) f
    congr 1
    funext i
    exact StageSpec.applyAt_off _ c m (fB i) a (fun hc => hchunk c' c a hne.symm hd hc.2)
  · intro m a hd
    show denote (fun i => (mapStage f fB o1 g1 h1).applyAt c' m (gB i) a) g
        = denote (fun i => m (gB i) a) g
    congr 1
    funext i
    exact StageSpec.applyAt_off _ c' m (gB i) a (fun hc => hchunk c c' a hne hd hc.2)

/-- **…so the interchange holds of a real fused pair**, hypotheses and all. -/
theorem mapStages_group_runs_as_pipeline {Γ Δ : Nat} (f : Expr Γ) (g : Expr Δ)
    (fB : Fin Γ → Buf) (gB : Fin Δ → Buf) (o1 o2 : Buf) (gr : Nat)
    (h1 : ∀ i, fB i ≠ o1) (h2 : ∀ i, gB i ≠ o2) (st : WSt) :
    (runGrid (groupBlk [mapStage f fB o1 gr h1, mapStage g gB o2 gr h2]) gr st).mem
      = (Pipeline.mk [mapStage f fB o1 gr h1, mapStage g gB o2 gr h2]).denote st.mem := by
  refine group_runs_as_pipeline _ gr ?_ ?_ ?_ st
  · intro S hS; simp at hS; rcases hS with rfl | rfl <;> rfl
  · intro S hS; simp at hS
    rcases hS with rfl | rfl <;> exact mapStage_exclusive _ _ _ _ _
  · intro T hT U hU c c' hne
    simp at hT hU
    rcases hT with rfl | rfl <;> rcases hU with rfl | rfl <;>
      exact mapStage_commute _ _ _ _ _ _ _ _ _ _ c c' hne


end AlgorithmLib.ML
