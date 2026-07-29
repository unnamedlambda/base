import AlgorithmLib.ML.PtxFlat

namespace AlgorithmLib.ML

/-- Index register holding this warp's index within the block.  `0`, `1`, `2`
    are lane, block and loop counter; the warp id joins them, so block-level
    address arithmetic allocates from `4`. -/
abbrev warpIR : Nat := 3

/-- A block: `NW` warps, each with its own machine state. -/
def BState (NW : Nat) := Fin NW → MState

/-- Install a warp's post-state, and publish its memory effects to every other
    warp.  This is what keeps "all warps see one memory" a *fact about the
    semantics* rather than a side condition. -/
def BState.put {NW : Nat} (b : BState NW) (w : Fin NW) (m : MState) : BState NW :=
  fun x => if x = w then m else { (b x) with mem := m.mem, sm := m.sm }

@[simp] theorem BState.put_same {NW : Nat} (b : BState NW) (w : Fin NW) (m : MState) :
    (b.put w m) w = m := by
  show (if w = w then m else _) = m
  rw [if_pos rfl]

theorem BState.put_other {NW : Nat} (b : BState NW) (w x : Fin NW) (m : MState) (h : x ≠ w) :
    (b.put w m) x = { (b x) with mem := m.mem, sm := m.sm } := by
  show (if x = w then m else _) = _
  rw [if_neg h]

/-- Run one warp, having first told it which warp it is. -/
def runWarpB {NW : Nat} (cta : Nat) (code : List SI) (b : BState NW) (w : Fin NW) :
    BState NW :=
  b.put w (SI.stepL cta code
    { (b w) with ir := fun x => if x = warpIR then (fun _ => w.val) else (b w).ir x })

/-- Run one warp by its numeric index (out-of-range indices are a no-op, which
    keeps the fold total). -/
def runWarpN {NW : Nat} (cta : Nat) (code : List SI) (b : BState NW) (j : Nat) : BState NW :=
  if h : j < NW then runWarpB cta code b ⟨j, h⟩ else b

/-- Run every warp.  In index order — see `blockStore_perm` for why that is not
    a cheat for the schema below. -/
def runBlock {NW : Nat} (cta : Nat) (code : List SI) (b : BState NW) : BState NW :=
  (List.range NW).foldl (fun acc j => runWarpN cta code acc j) b

-- ---------------------------------------------------------------------------
-- The cross-warp reduction schema
-- ---------------------------------------------------------------------------

/-- **Phase 1.**  Each warp parks its value in *its own* shared slot, then
    barriers.  `stS` writes from every lane, so after a butterfly — which leaves
    the warp's result in all 32 lanes — every lane writes the same value and the
    slot holds it regardless of which lane won. -/
def blockStoreSI : List SI := [SI.stS warpIR (.mach 0), SI.bar]

/-- **Phase 2.**  Every warp reads all `NW` slots and folds them, left to right.
    Every warp computes the whole block's value — a broadcast reduction, which
    is what a softmax denominator or an RMS norm actually wants, and which
    avoids needing cross-warp predication. -/
def blockCombineSI (NW : Nat) : List SI :=
  (List.range NW).foldl
    (fun acc j =>
      acc ++ [SI.movIC 4 j, SI.ldS (.mach 1) 4,
              SI.fp (.addF (.mach 0) (.mach 0) (.mach 1))])
    [SI.fp (.movImm (.mach 0) (NumOps.ofNat 0))]

/-- The value warp `w` contributes: whatever it left in `%fw0`, as seen by
    lane 0. -/
def warpVal {NW : Nat} (b : BState NW) (w : Fin NW) : Float32 :=
  (b w).fw 0 ⟨0, by decide⟩

-- ---------------------------------------------------------------------------
-- Phase 1 is order-independent
-- ---------------------------------------------------------------------------

/-- `%fw0` holds the same value in every lane — what a butterfly reduction
    leaves behind, and the condition under which "which lane won the write" is
    not a question anyone has to answer. -/
def MState.Uniform0 (m : MState) : Prop := ∀ l : Lane, m.fw 0 l = m.fw 0 ⟨0, by decide⟩

theorem finRange_ne_nil : (List.finRange W) ≠ [] := by
  intro h
  have hl := congrArg List.length h
  rw [List.length_finRange] at hl
  exact absurd hl (by decide)

/-- **Writing a warp's slot leaves every other slot alone.**

    This is the disjointness that makes a fixed warp order harmless: warp `w`
    touches shared memory only at index `w`. -/
theorem blockStore_smem {NW : Nat} (cta : Nat) (b : BState NW) (v : Fin NW) (j : Nat)
    (huni : (b v).Uniform0) :
    (SI.stepL cta blockStoreSI
      { (b v) with ir := fun x =>
          if x = warpIR then (fun _ => v.val) else (b v).ir x }).sm j
      = if j = v.val then warpVal b v else (b v).sm j := by
  have hcode : SI.stepL cta blockStoreSI
      { (b v) with ir := fun x => if x = warpIR then (fun _ => v.val) else (b v).ir x }
      = SI.step cta SI.bar (SI.step cta (SI.stS warpIR (.mach 0))
          { (b v) with ir := fun x =>
              if x = warpIR then (fun _ => v.val) else (b v).ir x }) := rfl
  rw [hcode]
  show ((SI.step cta (SI.stS warpIR (.mach 0)) _).sm) j = _
  show ((List.finRange W).foldl
      (fun sm' l => fun jj => if jj = v.val then (b v).fw 0 l else sm' jj) (b v).sm) j = _
  have key : ∀ (L : List Lane), L ≠ [] → ∀ (sm0 : Nat → Float32),
      (L.foldl (fun sm' l => fun jj => if jj = v.val then (b v).fw 0 l else sm' jj) sm0) j
        = if j = v.val then warpVal b v else sm0 j := by
    intro L
    induction L with
    | nil => intro h; exact absurd rfl h
    | cons a L ih =>
        intro _ sm0
        rw [List.foldl_cons]
        cases L with
        | nil =>
            show (if j = v.val then (b v).fw 0 a else sm0 j) = _
            by_cases h : j = v.val
            · rw [if_pos h, if_pos h]; exact huni a
            · rw [if_neg h, if_neg h]
        | cons c L' =>
            rw [ih (List.cons_ne_nil c L') _]
            by_cases h : j = v.val
            · rw [if_pos h, if_pos h]
            · rw [if_neg h, if_neg h, if_neg h]
  rw [key (List.finRange W) finRange_ne_nil (b v).sm]

/-- **Writing a warp's slot leaves every other slot alone.**

    This is the disjointness that makes a fixed warp order harmless: warp `w`
    touches shared memory only at index `w`. -/
theorem blockStore_slot {NW : Nat} (cta : Nat) (b : BState NW) (w : Fin NW) (j : Nat)
    (huni : (b w).Uniform0) :
    (runWarpB cta blockStoreSI b w w).sm j
      = if j = w.val then warpVal b w else (b w).sm j := by
  show ((BState.put b w (SI.stepL cta blockStoreSI _)) w).sm j = _
  rw [BState.put_same]
  exact blockStore_smem cta b w j huni

theorem blockStore_fw {NW : Nat} (cta : Nat) (b : BState NW) (v : Fin NW) :
    (SI.stepL cta blockStoreSI
      { (b v) with ir := fun x =>
          if x = warpIR then (fun _ => v.val) else (b v).ir x }).fw = (b v).fw := rfl

/-- The store is visible to *every* warp, and still touches only slot `v`. -/
theorem runWarpB_sm {NW : Nat} (cta : Nat) (b : BState NW) (v x : Fin NW) (j : Nat)
    (huni : (b v).Uniform0) :
    (runWarpB cta blockStoreSI b v x).sm j
      = if j = v.val then warpVal b v else (b v).sm j := by
  by_cases h : x = v
  · subst h; exact blockStore_slot cta b x j huni
  · show ((BState.put b v _) x).sm j = _
    rw [BState.put_other b v x _ h]
    exact blockStore_smem cta b v j huni

/-- Registers are private: another warp's store does not touch them. -/
theorem runWarpB_fw {NW : Nat} (cta : Nat) (b : BState NW) (v x : Fin NW) (h : x ≠ v) :
    (runWarpB cta blockStoreSI b v x).fw = (b x).fw := by
  show ((BState.put b v _) x).fw = _
  rw [BState.put_other b v x _ h]

/-- Uniformity survives a store — it writes shared memory, not registers. -/
theorem runWarpB_uniform {NW : Nat} (cta : Nat) (b : BState NW) (v : Fin NW)
    (huni : ∀ u, (b u).Uniform0) : ∀ u, ((runWarpB cta blockStoreSI b v) u).Uniform0 := by
  intro u l
  by_cases hu : u = v
  · subst hu
    show ((BState.put b u _) u).fw 0 l = ((BState.put b u _) u).fw 0 _
    rw [BState.put_same, blockStore_fw]
    exact huni u l
  · show ((runWarpB cta blockStoreSI b v) u).fw 0 l = _
    rw [runWarpB_fw cta b v u hu]
    exact huni u l

/-- After any warp runs, every warp sees the same shared memory — coherence is
    maintained by the semantics, not assumed. -/
theorem runWarpB_coherent {NW : Nat} (cta : Nat) (code : List SI) (b : BState NW)
    (v x y : Fin NW) :
    (runWarpB cta code b v x).sm = (runWarpB cta code b v y).sm := by
  have h : ∀ z : Fin NW, (runWarpB cta code b v z).sm
      = (SI.stepL cta code
          { (b v) with ir := fun q =>
              if q = warpIR then (fun _ => v.val) else (b v).ir q }).sm := by
    intro z
    by_cases hz : z = v
    · subst hz
      show ((BState.put b z _) z).sm = _
      rw [BState.put_same]
    · show ((BState.put b v _) z).sm = _
      rw [BState.put_other b v z _ hz]
  rw [h x, h y]

/-- **Phase 1 is correct.**  After every warp has stored, slot `w` holds warp
    `w`'s value, and no other slot was touched.

    The induction carries both halves because they need each other: a later warp
    must not clobber an earlier slot (second half), and an earlier warp must not
    have disturbed a later warp's registers (`runWarpB_fw`). -/
theorem blockStore_fold {NW : Nat} (cta : Nat) :
    ∀ (L : List Nat) (b : BState NW), (∀ v, (b v).Uniform0) → L.Nodup →
      (∀ u y : Fin NW, (b u).sm = (b y).sm) → ∀ (x : Fin NW),
      (∀ (w : Fin NW), w.val ∈ L →
          ((L.foldl (fun acc j => runWarpN cta blockStoreSI acc j) b) x).sm w.val
            = warpVal b w)
      ∧ (∀ k, (∀ j ∈ L, k ≠ j) →
          ((L.foldl (fun acc j => runWarpN cta blockStoreSI acc j) b) x).sm k
            = (b x).sm k) := by
  intro L
  induction L with
  | nil => intro b _ _ _ x; exact ⟨fun w hw => absurd hw (by simp), fun k _ => rfl⟩
  | cons v L ih =>
      intro b huni hnd hco x
      have hnd' : L.Nodup := (List.nodup_cons.mp hnd).2
      have hvL : v ∉ L := (List.nodup_cons.mp hnd).1
      by_cases hv : v < NW
      · have hstep : runWarpN cta blockStoreSI b v = runWarpB cta blockStoreSI b ⟨v, hv⟩ := by
          show (if h : v < NW then _ else _) = _
          rw [dif_pos hv]
        have hrec := ih (runWarpN cta blockStoreSI b v)
          (by rw [hstep]; exact runWarpB_uniform cta b ⟨v, hv⟩ huni) hnd'
          (by rw [hstep]; exact fun u y => runWarpB_coherent cta blockStoreSI b ⟨v, hv⟩ u y) x
        refine ⟨?_, ?_⟩
        · intro w hw
          rw [List.foldl_cons]
          rcases List.mem_cons.mp hw with hwv | hwL
          · rw [hrec.2 w.val (fun u hu hj => hvL (by rw [hwv.symm.trans hj]; exact hu)),
                hstep, runWarpB_sm cta b ⟨v, hv⟩ x w.val (huni ⟨v, hv⟩), if_pos hwv]
            show warpVal b ⟨v, hv⟩ = warpVal b w
            congr 1
            exact Fin.ext hwv.symm
          · rw [hrec.1 w hwL, hstep]
            show warpVal (runWarpB cta blockStoreSI b ⟨v, hv⟩) w = warpVal b w
            show (runWarpB cta blockStoreSI b ⟨v, hv⟩ w).fw 0 _ = _
            rw [runWarpB_fw cta b ⟨v, hv⟩ w
              (fun h => hvL (by
                have hv2 : w.val = v := congrArg Fin.val h
                rw [← hv2]; exact hwL))]
            rfl
        · intro k hk
          rw [List.foldl_cons, hrec.2 k (fun u hu => hk u (List.mem_cons_of_mem v hu)),
              hstep, runWarpB_sm cta b ⟨v, hv⟩ x k (huni ⟨v, hv⟩),
              if_neg (hk v (List.Mem.head _))]
          exact congrFun (hco ⟨v, hv⟩ x) k
      · have hstep : runWarpN cta blockStoreSI b v = b := by
          show (if h : v < NW then _ else _) = _
          rw [dif_neg hv]
        have hrec := ih (runWarpN cta blockStoreSI b v)
          (by rw [hstep]; exact huni) hnd' (by rw [hstep]; exact hco) x
        refine ⟨?_, ?_⟩
        · intro w hw
          rw [List.foldl_cons]
          rcases List.mem_cons.mp hw with hwv | hwL
          · exact absurd (hwv ▸ w.isLt) hv
          · rw [hrec.1 w hwL, hstep]
        · intro k hk
          rw [List.foldl_cons, hrec.2 k (fun u hu => hk u (List.mem_cons_of_mem v hu)), hstep]

-- ---------------------------------------------------------------------------
-- Phase 1, over the whole block — and why the warp order does not matter
-- ---------------------------------------------------------------------------

/-- **After phase 1, slot `w` holds warp `w`'s value.** -/
theorem blockStore_sound {NW : Nat} (cta : Nat) (b : BState NW)
    (huni : ∀ v, (b v).Uniform0) (hco : ∀ u y : Fin NW, (b u).sm = (b y).sm)
    (x w : Fin NW) :
    (runBlock cta blockStoreSI b x).sm w.val = warpVal b w :=
  (blockStore_fold cta (List.range NW) b huni List.nodup_range hco x).1 w
    (List.mem_range.mpr w.isLt)

/-- **The warp order does not matter.**

    Any schedule that runs each warp exactly once — any `Nodup` list containing
    every warp index, i.e. any permutation of `range NW` — produces the same
    shared memory.  So proving the schema in index order is not an artefact of
    the model: hardware may interleave the warps however it likes and reach the
    same state.

    This is what the barrier buys, stated rather than assumed. -/
theorem blockStore_perm {NW : Nat} (cta : Nat) (b : BState NW)
    (huni : ∀ v, (b v).Uniform0) (hco : ∀ u y : Fin NW, (b u).sm = (b y).sm)
    (L : List Nat) (hnd : L.Nodup) (hall : ∀ w : Fin NW, w.val ∈ L) (x w : Fin NW) :
    ((L.foldl (fun acc j => runWarpN cta blockStoreSI acc j) b) x).sm w.val
      = (runBlock cta blockStoreSI b x).sm w.val := by
  rw [(blockStore_fold cta L b huni hnd hco x).1 w (hall w),
      blockStore_sound cta b huni hco x w]

-- ---------------------------------------------------------------------------
-- Phase 2: every warp folds the slots
-- ---------------------------------------------------------------------------

theorem stepL_foldl_append (cta : Nat) (g : Nat → List SI) :
    ∀ (L : List Nat) (init : List SI) (m : MState),
      SI.stepL cta (L.foldl (fun acc j => acc ++ g j) init) m
        = L.foldl (fun s j => SI.stepL cta (g j) s) (SI.stepL cta init m) := by
  intro L
  induction L with
  | nil => intro init m; rfl
  | cons j L ih =>
      intro init m
      rw [List.foldl_cons, ih, srunL_append, List.foldl_cons]

theorem combineStep_sm (cta j : Nat) (m0 : MState) :
    (SI.stepL cta [SI.movIC 4 j, SI.ldS (.mach 1) 4,
      SI.fp (.addF (.mach 0) (.mach 0) (.mach 1))] m0).sm = m0.sm := rfl

theorem combineStep_fw (cta j : Nat) (m0 : MState) (l : Lane) :
    (SI.stepL cta [SI.movIC 4 j, SI.ldS (.mach 1) 4,
      SI.fp (.addF (.mach 0) (.mach 0) (.mach 1))] m0).fw 0 l
      = NumOps.add (m0.fw 0 l) (m0.sm j) := rfl

/-- **Phase 2 is correct.**  Every warp ends with the left fold of the `NW`
    slots in `%fw0` — a broadcast reduction, in the order the spec commits to. -/
theorem blockCombine_spec (cta NW : Nat) (m : MState) (l : Lane) :
    (SI.stepL cta (blockCombineSI NW) m).fw 0 l
      = (List.range NW).foldl (fun acc j => NumOps.add acc (m.sm j)) (NumOps.ofNat 0) := by
  show (SI.stepL cta ((List.range NW).foldl (fun acc j => acc ++ _) _) m).fw 0 l = _
  rw [stepL_foldl_append cta]
  have key : ∀ (L : List Nat) (m0 : MState), m0.sm = m.sm →
      (L.foldl (fun s j => SI.stepL cta
          [SI.movIC 4 j, SI.ldS (.mach 1) 4,
           SI.fp (.addF (.mach 0) (.mach 0) (.mach 1))] s) m0).fw 0 l
        = L.foldl (fun acc j => NumOps.add acc (m.sm j)) (m0.fw 0 l) := by
    intro L
    induction L with
    | nil => intro m0 _; rfl
    | cons j L ih =>
        intro m0 hsm
        rw [List.foldl_cons, List.foldl_cons,
            ih _ (by rw [combineStep_sm]; exact hsm)]
        congr 1
        rw [combineStep_fw, hsm]
  rw [key (List.range NW)
        (SI.stepL cta [SI.fp (.movImm (.mach 0) (NumOps.ofNat 0))] m) rfl]
  rfl

-- ---------------------------------------------------------------------------
-- The whole schema
-- ---------------------------------------------------------------------------

/-- Warp `j`'s contribution, as a total function of a `Nat` index. -/
def slotVal {NW : Nat} (b : BState NW) (j : Nat) : Float32 :=
  if h : j < NW then warpVal b ⟨j, h⟩ else NumOps.ofNat 0

theorem foldl_congr_range {β : Type} (f g : β → Nat → β) :
    ∀ (L : List Nat) (init : β), (∀ acc j, j ∈ L → f acc j = g acc j) →
      L.foldl f init = L.foldl g init := by
  intro L
  induction L with
  | nil => intro init _; rfl
  | cons a L ih =>
      intro init h
      show L.foldl f (f init a) = L.foldl g (g init a)
      rw [h init a (List.Mem.head _)]
      exact ih (g init a) (fun acc x hx => h acc x (List.Mem.tail _ hx))

/-- **The cross-warp reduction is correct.**

    Every warp of the block ends with the left fold of all `NW` warp values in
    `%fw0` — computed through *shared memory and a barrier*, which is the one
    thing the single-warp machine could not express.

    Phase 2 is stated for an arbitrary warp `x` because it *is* per-warp: it
    reads shared memory and writes only its own registers, so every warp runs
    the identical computation and arrives at the identical answer.  That is what
    makes this a broadcast reduction — what a softmax denominator or an RMS norm
    actually needs.

    Exact `Float32` equality, in the fold order the spec commits to. -/
theorem blockReduce_sound {NW : Nat} (cta : Nat) (b : BState NW)
    (huni : ∀ v, (b v).Uniform0) (hco : ∀ u y : Fin NW, (b u).sm = (b y).sm)
    (x : Fin NW) (l : Lane) :
    (SI.stepL cta (blockCombineSI NW) ((runBlock cta blockStoreSI b) x)).fw 0 l
      = (List.range NW).foldl
          (fun acc j => NumOps.add acc (slotVal b j)) (NumOps.ofNat 0) := by
  rw [blockCombine_spec cta NW ((runBlock cta blockStoreSI b) x) l]
  refine foldl_congr_range _ _ (List.range NW) _ ?_
  intro acc j hj
  have hjn : j < NW := List.mem_range.mp hj
  show NumOps.add acc (((runBlock cta blockStoreSI b) x).sm j) = _
  rw [show j = (⟨j, hjn⟩ : Fin NW).val from rfl,
      blockStore_sound cta b huni hco x ⟨j, hjn⟩]
  show _ = NumOps.add acc (slotVal b j)
  show _ = NumOps.add acc (if h : j < NW then warpVal b ⟨j, h⟩ else NumOps.ofNat 0)
  rw [dif_pos hjn]

/-- **A block kernel.**  Whatever per-warp code `body` is — a strided sweep, a
    butterfly, a gather — the block then reduces across warps and every warp
    ends holding the block's value.

    This is the composition a real kernel has: per-warp work, barrier,
    cross-warp combine.  It is a corollary of `blockReduce_sound` rather than a
    new argument, which is the point: the cross-warp part is proven once and
    reused, exactly like `warpDotV4` at the warp level. -/
theorem blockKernel_sound {NW : Nat} (cta : Nat) (body : List SI) (b0 : BState NW)
    (huni : ∀ v, ((runBlock cta body b0) v).Uniform0)
    (hco : ∀ u y : Fin NW, ((runBlock cta body b0) u).sm = ((runBlock cta body b0) y).sm)
    (x : Fin NW) (l : Lane) :
    (SI.stepL cta (blockCombineSI NW)
        ((runBlock cta blockStoreSI (runBlock cta body b0)) x)).fw 0 l
      = (List.range NW).foldl
          (fun acc j => NumOps.add acc (slotVal (runBlock cta body b0) j))
          (NumOps.ofNat 0) :=
  blockReduce_sound cta (runBlock cta body b0) huni hco x l

/-- Coherence is established by running *any* warp, so it is a property of the
    machine rather than a hypothesis a user has to carry. -/
theorem runWarpN_coherent {NW : Nat} (cta : Nat) (code : List SI) (b : BState NW)
    (j : Nat) (hj : j < NW) (x y : Fin NW) :
    (runWarpN cta code b j x).sm = (runWarpN cta code b j y).sm := by
  have h : runWarpN cta code b j = runWarpB cta code b ⟨j, hj⟩ := by
    show (if h : j < NW then runWarpB cta code b ⟨j, h⟩ else b) = _
    rw [dif_pos hj]
  rw [h]
  exact runWarpB_coherent cta code b ⟨j, hj⟩ x y

-- ---------------------------------------------------------------------------
-- The schema is expressible in the emittable language
-- ---------------------------------------------------------------------------

/-- Phase 1, written in `EWStmt` — the language `flatKernel` compiles and
    `PtxPrint` prints.  The address is `ireg warpIR`: the warp index is *already*
    in a register, so the emitted address code is empty and the store is a
    single `st.shared.f32`. -/
def blockStoreEW : EWStmt := .seq (.stSm (.ireg warpIR) 0) .barrier

/-- **The emittable form lowers to exactly the instructions the block theorems
    are about** — by `rfl`, so there is no gap between what is proven across
    warps (`blockStore_sound`, `blockStore_perm`) and what is printed.

    Together with `flatKernel_sound`, which covers each warp's execution on the
    program-counter machine, this closes the path from a block-level schema to
    PTX. -/
theorem blockStoreEW_emits (lr n : Nat) : emitEW lr n blockStoreEW = blockStoreSI := rfl

/-- Its address mentions a register, not a gather, so it satisfies the lowering
    precondition whenever the warp-index register is below the watermark — which
    it is, `warpIR = 3` and block kernels allocate from `4`. -/
theorem blockStoreEW_idxBelow (n : Nat) (h : warpIR < n) : blockStoreEW.IdxBelow n :=
  ⟨h, trivial⟩

theorem blockStoreEW_expFree : blockStoreEW.ExpFree := ⟨trivial, trivial⟩

end AlgorithmLib.ML
