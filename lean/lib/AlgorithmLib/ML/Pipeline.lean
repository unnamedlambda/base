import AlgorithmLib.ML.Kernels
import AlgorithmLib.ML.Layered

/-!
  # Stages, grids, and pipelines

  Every kernel theorem in this stack has the shape

      given that buffer `X` holds what spec term `ae` denotes,
      after this kernel buffer `Y` holds …

  and every kernel is launched after another kernel that filled `X`.  This file
  connects the two, so that stage `n`'s hypothesis is *discharged* by stage
  `n−1`'s conclusion rather than assumed — which is what lets a user compose
  stages from the library and get the composition for free.

  Three pieces are needed.

  **The grid.**  A kernel theorem is about *one block*.  A launch runs `grid` of
  them over one memory.  `runGrid` folds the blocks; `runGrid_value` says a
  block's store survives the blocks that follow it, which needs each block to
  own its output addresses exclusively (`Exclusive`).

  **Whole buffers.**  A store theorem talks about one address.  A stage's
  contract is about a buffer, so the per-address facts are collected into
  `StageSpec.value` quantified over the address.

  **The fold.**  With stages carrying a frame (`writes only its output`) and a
  value (`and puts this there`), running them in sequence is an induction, and
  the next stage's input hypothesis falls out of the previous stage's value.

  ## What a stage is not allowed to do

  `valOnly` says a stage's output depends only on buffers *other than* its own
  output.  A kernel that reads what it writes — RoPE — is outside this
  abstraction by design: its correctness depends on intra-kernel ordering in a
  way an inter-kernel frame condition cannot express.  `EWStmt.StageEligibleB`
  decides the condition.
-/

namespace AlgorithmLib.ML

-- ---------------------------------------------------------------------------
-- Running a grid of blocks over one memory
-- ---------------------------------------------------------------------------

/-- **A launch: one kernel, `grid` blocks, one memory.**

    Two ways this differs from a real launch, and what is done about each.

    *Registers* are threaded rather than reset between blocks.  Every per-block
    theorem in this stack is stated for an **arbitrary** entry state, so nothing
    concluded here depends on what the registers held — the unfaithfulness is in
    the conservative direction.

    *Order* is ascending; the hardware interleaves arbitrarily.
    `runBlocks_perm_invariant` proves the resulting memory is the same for any
    block list with the same membership, so the ascending fold is a faithful
    stand-in. -/
def runGrid (blk : Nat → WStmt) (grid : Nat) (st : WSt) : WSt :=
  (List.range grid).foldl (fun s cta => (blk cta).run s) st

/-- **A stage**: a kernel, a grid, and the contract it honours.

    `dom cta a` — block `cta` owns address `a` of `out`.
    `val m a` — what lands at `a`, as a function of the entry memory. -/
structure StageSpec where
  ew   : EWStmt
  /-- **The integer state the launch elaborates against.**

      Four of a transformer's kernels address memory with a value read at run
      time — RoPE's token position, the KV write's slot, softmax's sequence
      length, the embedding gather's token id.  Elaboration resolves those
      through `ir`/`im`, and a stage that pinned them to zero would be a
      faithful model of the kernel *only at position zero*: every theorem would
      hold and describe the wrong launch.

      So the launch's integer state is part of the stage.  It defaults to zero,
      which is what an address-free kernel elaborates against anyway, so the
      generated stages are unaffected. -/
  iregs : Nat → Lane → Nat := fun _ _ => 0
  imem  : Buf → Nat → Nat := fun _ _ => 0
  grid : Nat
  out  : Buf
  dom  : Nat → Nat → Prop
  /-- What lands at an address, as a function of the entry memory **and the
      block that wrote it**.  The block matters: an outer product's value at
      `dW[i·n+j]` depends on row `i`, which is the block, not the address. -/
  val  : (Buf → Nat → Float32) → Nat → Nat → Float32
  /-- A block writes only its own output addresses. -/
  frame : ∀ (cta : Nat) (st : WSt) (b : Buf) (a : Nat), (b ≠ out ∨ ¬ dom cta a) →
    ((ew.elabAt cta 0 iregs imem).run st).mem b a = st.mem b a
  /-- …and puts `val` there. -/
  value : ∀ (cta : Nat) (st : WSt) (a : Nat), dom cta a →
    ((ew.elabAt cta 0 iregs imem).run st).mem out a = val st.mem cta a
  /-- **The result depends on the output buffer only where this block owns it.**

      The point is that earlier blocks of the *same* launch cannot perturb
      later ones.  Requiring independence from `out` entirely is sufficient but
      too strong: it excludes every in-place kernel, and an in-place kernel is
      not racy — `compileWKernel` loads all of its inputs before storing any of
      them, so within a block the update is sound.  What actually matters is
      that a block reads `out` only at addresses *it* owns, because `Exclusive`
      then says no other block of the launch has written them. -/
  valOnly : ∀ (m m' : Buf → Nat → Float32) (cta a : Nat), dom cta a →
    (∀ b, b ≠ out → m b = m' b) →
    (∀ a', dom cta a' → m out a' = m' out a') →
    val m cta a = val m' cta a

/-- **The block this stage runs**, elaborated against the launch's integer
    state.  Every theorem below is stated through this rather than `elabIn`, so
    a data-dependent address is modelled at the value the launch actually
    supplies. -/
def StageSpec.blk (S : StageSpec) (cta : Nat) : WStmt :=
  S.ew.elabAt cta 0 S.iregs S.imem

/-- `frame`, restated through `blk` so the rewriter can see it. -/
theorem StageSpec.frameB (S : StageSpec) (cta : Nat) (st : WSt) (b : Buf) (a : Nat)
    (h : b ≠ S.out ∨ ¬ S.dom cta a) : ((S.blk cta).run st).mem b a = st.mem b a :=
  S.frame cta st b a h

/-- …and `value`. -/
theorem StageSpec.valueB (S : StageSpec) (cta : Nat) (st : WSt) (a : Nat)
    (h : S.dom cta a) : ((S.blk cta).run st).mem S.out a = S.val st.mem cta a :=
  S.value cta st a h

/-- Distinct blocks own distinct addresses — no two blocks race. -/
def StageSpec.Exclusive (S : StageSpec) : Prop :=
  ∀ cta cta' a, cta < S.grid → cta' < S.grid → S.dom cta a → S.dom cta' a → cta = cta'

/-- **Why the bounds are part of the statement.**

    Ownership only has to be exclusive among the blocks the launch actually
    runs.  Without the bounds, a kernel whose addressing ignores `ctaId` — the
    single-block reductions, RMSNorm and softmax, which sweep their whole row
    inside one warp — could never be a stage: every hypothetical block claims
    the same addresses, so unbounded exclusivity is false, while the frame
    condition forces `dom` to cover everything the kernel writes for *every*
    `cta`.  The two demands are jointly unsatisfiable, and the kernel is
    nonetheless perfectly safe because the grid is 1.

    That safety is a fact about the *launch*, which is exactly the seam
    `Geometry.lean` is about, so the grid belongs in the statement.  Nothing is
    weakened for the kernels that do use `ctaId`: for them `dom cta` and
    `dom cta'` are disjoint outright and the bounds go unused. -/
theorem StageSpec.Exclusive.ofUnbounded {S : StageSpec}
    (h : ∀ cta cta' a, S.dom cta a → S.dom cta' a → cta = cta') : S.Exclusive :=
  fun cta cta' a _ _ h₁ h₂ => h cta cta' a h₁ h₂

/-- A grid of one block is exclusive whatever it claims. -/
theorem StageSpec.exclusive_of_grid_one {S : StageSpec} (h : S.grid = 1) :
    S.Exclusive := by
  intro cta cta' _ hc hc' _ _
  rw [h] at hc hc'
  omega

/-- **Re-running a block recomputes the same value.**

    Strictly stronger than `valOnly`: the result must ignore the output buffer
    *entirely*, not merely outside the block's own addresses.  That is what a
    block list containing the same block twice requires, and it is exactly what
    an in-place stage fails — `out[i] := out[i] + b[i]` run twice lands
    `out[i] + 2·b[i]`.

    A grid runs each block once, so `runGrid_value` does not need this.  The
    arbitrary-list and permutation results do, which is why they take it as a
    hypothesis rather than a field: an in-place kernel is a perfectly good
    stage, it just is not idempotent. -/
def StageSpec.Idempotent (S : StageSpec) : Prop :=
  ∀ (m m' : Buf → Nat → Float32) (cta a : Nat),
    (∀ b, b ≠ S.out → m b = m' b) → S.val m cta a = S.val m' cta a

/-- Memory outside the stage's output buffer is untouched by the whole grid. -/
theorem runGrid_otherBuf (S : StageSpec) (b : Buf) (hb : b ≠ S.out) :
    ∀ (n : Nat) (st : WSt), (runGrid S.blk n st).mem b = st.mem b := by
  intro n
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
      intro st
      show ((List.range (n + 1)).foldl (fun s cta => (S.blk cta).run s) st).mem b = _
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      funext a
      rw [S.frameB n _ b a (Or.inl hb)]
      exact congrFun (ih st) a

/-- Addresses that no block of the grid owns are left alone — the address-level
    companion to `runGrid_otherBuf`, and what lets an in-place stage read its
    own output buffer safely. -/
theorem runGrid_otherAddr (S : StageSpec) (a : Nat) :
    ∀ (n : Nat) (st : WSt), (∀ c, c < n → ¬ S.dom c a) →
      (runGrid S.blk n st).mem S.out a = st.mem S.out a := by
  intro n
  induction n with
  | zero => intro _ _; rfl
  | succ n ih =>
      intro st hno
      show ((List.range (n + 1)).foldl (fun s c => (S.blk c).run s) st).mem S.out a = _
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      rw [S.frameB n _ S.out a (Or.inr (hno n (by omega)))]
      exact ih st (fun c hc => hno c (by omega))

/-- **What a launch leaves in memory.**

    Block `cta` writes `val` at the addresses it owns, and — because ownership
    is exclusive, and no block reads the output buffer — that value survives
    every other block in the grid.  This is the statement a *stage* has, as
    opposed to a *block*. -/
theorem runGrid_value (S : StageSpec) (hex : S.Exclusive) (a : Nat) :
    ∀ (n : Nat), n ≤ S.grid → ∀ (cta : Nat), cta < n → S.dom cta a → ∀ (st : WSt),
      (runGrid S.blk n st).mem S.out a = S.val st.mem cta a := by
  intro n
  induction n with
  | zero => intro _ _ h; exact absurd h (by omega)
  | succ n ih =>
      intro hng cta hlt hdom st
      show ((List.range (n + 1)).foldl (fun s c => (S.blk c).run s) st).mem S.out a = _
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      by_cases hc : cta = n
      · -- the last block writes it
        subst hc
        rw [S.valueB cta _ a hdom]
        refine S.valOnly _ _ _ a hdom (fun b hb => runGrid_otherBuf S b hb _ st) ?_
        intro a' hd
        exact runGrid_otherAddr S a' cta st
          (fun c hc hdc => absurd (hex c cta a' (by omega) (by omega) hdc hd) (by omega))
      · -- an earlier block wrote it; the last block does not own it
        have hne : ¬ S.dom n a := fun hn => hc (hex cta n a (by omega) (by omega) hdom hn)
        rw [S.frameB n _ S.out a (Or.inr hne)]
        exact ih (by omega) cta (by omega) hdom st

-- ---------------------------------------------------------------------------
-- Order independence: the model is sequential, the hardware is not
-- ---------------------------------------------------------------------------

/-!
  `runGrid` folds blocks in ascending order.  A real launch interleaves them in
  whatever order the scheduler picks, so a theorem about the ascending fold
  describes the hardware **only if the result does not depend on the order**.

  With exclusive ownership and the frame condition, running the blocks in *any*
  order — any list, in fact — leaves the same memory.

  The argument is not "blocks commute" in general (they do not: two blocks
  writing one address genuinely race).  It is that under `Exclusive` the final
  contents at each address are determined by the *set* of blocks run, not their
  sequence, because at most one of them writes it and none of them reads the
  output buffer. -/

/-- Running any list of blocks: the generalisation of `runGrid` off `List.range`. -/
def runBlocks (blk : Nat → WStmt) (L : List Nat) (st : WSt) : WSt :=
  L.foldl (fun s cta => (blk cta).run s) st

/-- Memory outside the output buffer survives any block list. -/
theorem runBlocks_otherBuf (S : StageSpec) (b : Buf) (hb : b ≠ S.out) :
    ∀ (L : List Nat) (st : WSt), (runBlocks S.blk L st).mem b = st.mem b := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons c L ih =>
      intro st
      show (runBlocks S.blk L ((S.blk c).run st)).mem b = _
      rw [ih _]
      funext a
      exact S.frameB c st b a (Or.inl hb)

/-- An unowned address survives any block list. -/
theorem runBlocks_unowned (S : StageSpec) (a : Nat) :
    ∀ (L : List Nat) (st : WSt), (∀ c ∈ L, ¬ S.dom c a) →
      (runBlocks S.blk L st).mem S.out a = st.mem S.out a := by
  intro L
  induction L with
  | nil => intro _ _; rfl
  | cons c L ih =>
      intro st hno
      show (runBlocks S.blk L ((S.blk c).run st)).mem S.out a = _
      rw [ih _ (fun c' hc' => hno c' (List.mem_cons_of_mem c hc'))]
      exact S.frameB c st S.out a (Or.inr (hno c (List.mem_cons_self)))

/-- **Once the owner has run, nothing else can change the address.**

    Not just "no other block owns it" — a block list may contain the owner
    *twice*, and a second run must not change the answer.  It does not, because
    `valOnly` says the value ignores the output buffer, so re-running the owner
    recomputes the same thing. -/
theorem runBlocks_keeps (S : StageSpec) (hex : S.Exclusive) (hidem : S.Idempotent)
    (a cta : Nat) (hcta : cta < S.grid) (hdom : S.dom cta a) :
    ∀ (L : List Nat), (∀ c ∈ L, c < S.grid) → ∀ (st s : WSt),
      (∀ b, b ≠ S.out → s.mem b = st.mem b) →
      s.mem S.out a = S.val st.mem cta a →
      (runBlocks S.blk L s).mem S.out a = S.val st.mem cta a := by
  intro L
  induction L with
  | nil => intro _ _ _ _ hv; exact hv
  | cons c L ih =>
      intro hLg st s hag hv
      refine ih (fun x hx => hLg x (List.mem_cons_of_mem _ hx))
        st ((S.blk c).run s) (fun b hb => ?_) ?_
      · funext a'
        rw [S.frameB c s b a' (Or.inl hb)]
        exact congrFun (hag b hb) a'
      · by_cases hdc : S.dom c a
        · rw [S.valueB c s a hdc]
          have : c = cta := hex c cta a (hLg c (List.mem_cons_self ..)) hcta hdc hdom
          subst this
          exact hidem _ _ c a hag
        · rw [S.frameB c s S.out a (Or.inr hdc)]
          exact hv

/-- **What any block list leaves at an owned address — independent of order.**

    The value depends on the *entry* memory and the owning block, not on where
    that block sat in the list, nor on which blocks ran before or after it. -/
theorem runBlocks_value (S : StageSpec) (hex : S.Exclusive) (hidem : S.Idempotent)
    (a : Nat) :
    ∀ (L : List Nat), (∀ c ∈ L, c < S.grid) → ∀ (cta : Nat), cta ∈ L → S.dom cta a →
      ∀ (st : WSt),
      (runBlocks S.blk L st).mem S.out a = S.val st.mem cta a := by
  intro L
  induction L with
  | nil => intro _ _ h; exact absurd h (by simp)
  | cons c L ih =>
      intro hLg cta hmem hdom st
      show (runBlocks S.blk L ((S.blk c).run st)).mem S.out a = _
      by_cases hc : S.dom c a
      · have hcc : c = cta :=
          hex c cta a (hLg c (List.mem_cons_self ..)) (hLg cta hmem) hc hdom
        subst hcc
        refine runBlocks_keeps S hex hidem a c (hLg c (List.mem_cons_self ..)) hc L
          (fun x hx => hLg x (List.mem_cons_of_mem _ hx)) st _ (fun b hb => ?_) ?_
        · funext a'; exact S.frameB c st b a' (Or.inl hb)
        · exact S.valueB c st a hc
      · have hmem' : cta ∈ L := by
          rcases List.mem_cons.mp hmem with h | h
          · exact absurd (h ▸ hdom) hc
          · exact h
        rw [ih (fun x hx => hLg x (List.mem_cons_of_mem _ hx)) cta hmem' hdom _]
        refine hidem _ _ _ a (fun b hb => ?_)
        funext a'
        exact S.frameB c st b a' (Or.inl hb)

/-- **Order independence.**  Any two block lists with the same membership leave
    the same memory — so the ascending fold `runGrid` uses is a faithful stand-in
    for a launch whose blocks are scheduled arbitrarily. -/
theorem runBlocks_perm_invariant (S : StageSpec) (hex : S.Exclusive)
    (hidem : S.Idempotent)
    (L L' : List Nat) (hLg : ∀ c ∈ L, c < S.grid)
    (hmem : ∀ c, c ∈ L ↔ c ∈ L') (st : WSt) :
    (runBlocks S.blk L st).mem = (runBlocks S.blk L' st).mem := by
  have hLg' : ∀ c ∈ L', c < S.grid := fun c hc => hLg c ((hmem c).mpr hc)
  funext b a
  by_cases hb : b = S.out
  · subst hb
    by_cases hd : ∃ cta, cta ∈ L ∧ S.dom cta a
    · obtain ⟨cta, hcta, hdom⟩ := hd
      rw [runBlocks_value S hex hidem a L hLg cta hcta hdom st,
          runBlocks_value S hex hidem a L' hLg' cta ((hmem cta).mp hcta) hdom st]
    · have hL : ∀ c ∈ L, ¬ S.dom c a := fun c hc hdc => hd ⟨c, hc, hdc⟩
      have hL' : ∀ c ∈ L', ¬ S.dom c a := fun c hc hdc =>
        hd ⟨c, (hmem c).mpr hc, hdc⟩
      rw [runBlocks_unowned S a L st hL, runBlocks_unowned S a L' st hL']
  · rw [congrFun (runBlocks_otherBuf S b hb L st) a,
        congrFun (runBlocks_otherBuf S b hb L' st) a]

-- ---------------------------------------------------------------------------
-- Sequencing stages
-- ---------------------------------------------------------------------------

/-- Running a stage: its whole grid. -/
def StageSpec.run (S : StageSpec) (st : WSt) : WSt := runGrid S.blk S.grid st

/-- **A stage's contract, as a statement about the buffer.**

    This is the form the *next* stage needs: not "block `cta` stored `v` at `a`"
    but "buffer `out` holds `val` at every address the launch covers". -/
theorem StageSpec.run_value (S : StageSpec) (hex : S.Exclusive) (st : WSt)
    (a : Nat) (cta : Nat) (hlt : cta < S.grid) (hdom : S.dom cta a) :
    (S.run st).mem S.out a = S.val st.mem cta a :=
  runGrid_value S hex a S.grid (Nat.le_refl _) cta hlt hdom st

/-- …and everything else is exactly as it was. -/
theorem StageSpec.run_frame (S : StageSpec) (st : WSt) (b : Buf) (hb : b ≠ S.out) :
    (S.run st).mem b = st.mem b :=
  runGrid_otherBuf S b hb S.grid st

-- ---------------------------------------------------------------------------
-- Lifting the launch to the emitted instruction list
-- ---------------------------------------------------------------------------

/-!
  Everything above is about `elabIn cta` — the warp machine.  The PTX soundness
  theorems (`flatKernel_sound_idxFree`) are also per block.  This section lifts
  both to the grid: run the flat program once per block, threading the machine
  state, and the resulting memory is the one `runGrid` computes.
-/

/-- **One block of emitted PTX realises one block of the stage model.**

    This is `flatKernel_sound_idxFree` restated in the shape the grid fold
    needs: from any machine state, running the flat program for block `cta`
    lands in a state whose `WSt` view is exactly what `elabIn cta` computes. -/
theorem flat_block_realises (s : EWStmt) (hf : s.ExpFree) (hif : s.IdxFree)
    (hfl : s.Flat) (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel s) k (0, m) = some ((flatKernel s).length, m')
      ∧ m'.toWSt = (s.elabIn cta).run m.toWSt :=
  flatKernel_sound_idxFree cta s hf hif hfl m

/-- **`m` reaches `m'` by actually running blocks `0 … n−1`, in order.**

    Spelled out as a relation rather than left implicit, so the theorem below
    states that its witness is reached *by execution* — not merely that some
    state exists whose memory matches. -/
def gridRuns (P : List FI) : Nat → MState → MState → Prop
  | 0,     m, m' => m' = m
  | n + 1, m, m' => ∃ mid, gridRuns P n m mid ∧
      ∃ k, steps n P k (0, mid) = some (P.length, m')

/-- **The whole grid, on the emitted program.**

    There is a genuine execution — blocks `0 … n−1` of the *printed* program,
    each stepped to completion, threading the machine state — and the memory it
    ends in is exactly `runGrid`'s.  With `runBlocks_perm_invariant` the
    ascending order is not a restriction, so this covers a launch whose blocks
    the scheduler interleaves. -/
theorem flatGrid_realises (s : EWStmt) (hf : s.ExpFree) (hif : s.IdxFree)
    (hfl : s.Flat) :
    ∀ (n : Nat) (m : MState), ∃ m' : MState,
      gridRuns (flatKernel s) n m m' ∧ m'.toWSt = runGrid s.elabIn n m.toWSt := by
  intro n
  induction n with
  | zero => intro m; exact ⟨m, rfl, rfl⟩
  | succ n ih =>
      intro m
      obtain ⟨mn, hchain, hmn⟩ := ih m
      obtain ⟨k, m', hs, hw⟩ := flat_block_realises s hf hif hfl n mn
      refine ⟨m', ⟨mn, hchain, k, hs⟩, ?_⟩
      rw [hw, hmn]
      show (s.elabIn n).run
          ((List.range n).foldl (fun st cta => (s.elabIn cta).run st) m.toWSt)
        = (List.range (n + 1)).foldl (fun st cta => (s.elabIn cta).run st) m.toWSt
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

-- ---------------------------------------------------------------------------
-- The front door: a map kernel *is* a stage
-- ---------------------------------------------------------------------------

/-- `elemIx` at block `cta` addresses exactly `cta·32 + lane`. -/
theorem elemIx_val (cta : Nat) (l : Lane) :
    elemIx.eval cta 0 l (fun _ _ => 0) (fun _ _ => 0) = cta * 32 + l.val := rfl

/-- Distinct blocks own distinct elements — the exclusivity a grid needs. -/
theorem elemIx_blocks_disjoint (cta cta' : Nat) (l l' : Lane)
    (h : cta * 32 + l.val = cta' * 32 + l'.val) : cta = cta' := by
  have h1 : l.val < 32 := l.isLt
  have h2 : l'.val < 32 := l'.isLt
  omega

/-- **A map kernel, as a pipeline stage — with nothing left to prove.**

    All four contract fields are constructed here: the frame from
    `compileWKernel_otherBuf`/`_otherAddr`, the value from
    `compileWKernel_storesAt`, exclusivity from the addressing.  The one
    hypothesis left is what only the caller knows — that the kernel does not read
    the buffer it writes. -/
def mapStage {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf)
    (grid : Nat) (hio : ∀ i, inB i ≠ out) : StageSpec where
  ew   := (mapKernel spec inB out).ew
  grid := grid
  out  := out
  dom  := fun cta a => ∃ l : Lane, cta * 32 + l.val = a
  val  := fun m _ a => denote (fun i => m (inB i) a) spec
  frame := by
    intro cta st b a hb
    rcases hb with hb | hb
    · exact congrFun (compileWKernel_otherBuf inB out (fun _ => elemIx) spec elemIx
        (Γ + slots spec + 1) cta 0 _ _ st b hb) a
    · by_cases hbo : b = out
      · subst hbo
        refine compileWKernel_otherAddr inB b (fun _ => elemIx) spec elemIx
          (Γ + slots spec + 1) cta 0 _ _ st a (fun l hc => hb ⟨l, hc⟩)
      · exact congrFun (compileWKernel_otherBuf inB out (fun _ => elemIx) spec elemIx
          (Γ + slots spec + 1) cta 0 _ _ st b hbo) a
  value := by
    intro cta st a hdom
    obtain ⟨l, hl⟩ := hdom
    subst hl
    exact compileWKernel_storesAt inB out (fun _ => elemIx) spec elemIx
      (Γ + slots spec + 1) cta 0 _ _ st l
      (fun l' h => by
        have hv : cta * 32 + l'.val = cta * 32 + l.val := h
        have : l' = l := Fin.ext (by have := l.isLt; have := l'.isLt; omega)
        rw [this])
  valOnly := by
    intro m m' _ a _ h _
    congr 1
    funext i
    rw [h (inB i) (hio i)]

/-- **An in-place map stage** — `out` may itself be one of the inputs.

    `out[i] := f(out[i], b[i])` is the residual add, and it was previously
    inexpressible as a stage: `mapStage` requires `inB i ≠ out` for *every*
    input, and that requirement lives entirely in `valOnly`.  The frame and
    value obligations never needed it.

    What makes it sound is that a block reads `out` only at the address it
    writes, which is one it owns, so `Exclusive` says no other block of the
    launch has touched it.  Not idempotent — running it twice adds twice — so
    it supports `runGrid_value` but not the arbitrary-list results. -/
def mapStageIP {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf)
    (grid : Nat) : StageSpec where
  ew   := (mapKernel spec inB out).ew
  grid := grid
  out  := out
  dom  := fun cta a => ∃ l : Lane, cta * 32 + l.val = a
  val  := fun m _ a => denote (fun i => m (inB i) a) spec
  frame := by
    intro cta st b a hb
    rcases hb with hb | hb
    · exact congrFun (compileWKernel_otherBuf inB out (fun _ => elemIx) spec elemIx
        (Γ + slots spec + 1) cta 0 _ _ st b hb) a
    · by_cases hbo : b = out
      · subst hbo
        refine compileWKernel_otherAddr inB b (fun _ => elemIx) spec elemIx
          (Γ + slots spec + 1) cta 0 _ _ st a (fun l hc => hb ⟨l, hc⟩)
      · exact congrFun (compileWKernel_otherBuf inB out (fun _ => elemIx) spec elemIx
          (Γ + slots spec + 1) cta 0 _ _ st b hbo) a
  value := by
    intro cta st a hdom
    obtain ⟨l, hl⟩ := hdom
    subst hl
    exact compileWKernel_storesAt inB out (fun _ => elemIx) spec elemIx
      (Γ + slots spec + 1) cta 0 _ _ st l
      (fun l' h => by
        have hv : cta * 32 + l'.val = cta * 32 + l.val := h
        have : l' = l := Fin.ext (by have := l.isLt; have := l'.isLt; omega)
        rw [this])
  valOnly := by
    intro m m' cta a hdom h hout
    show denote (fun i => m (inB i) a) spec = denote (fun i => m' (inB i) a) spec
    refine congrArg (fun f => denote f spec) (funext fun i => ?_)
    by_cases hb : inB i = out
    · rw [hb]; exact hout a hdom
    · rw [h (inB i) hb]

/-- **The map stage is idempotent** — its value is a `denote` over the input
    buffers, none of which is `out`, so re-running a block recomputes it. -/
theorem mapStage_idempotent {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf)
    (grid : Nat) (hio : ∀ i, inB i ≠ out) :
    (mapStage spec inB out grid hio).Idempotent := by
  intro m m' _ a h
  show denote (fun i => m (inB i) a) spec = denote (fun i => m' (inB i) a) spec
  exact congrArg (fun f => denote f spec) (funext fun i => by rw [h (inB i) (hio i)])

/-- The lane that owns address `a` under `elemIx` addressing. -/
def laneMod (a : Nat) : Lane := ⟨a % 32, Nat.mod_lt _ (by decide)⟩

theorem laneMod_of {cta a : Nat} {l : Lane} (h : cta * 32 + l.val = a) : l = laneMod a := by
  have : l.val < 32 := l.isLt
  exact Fin.ext (by show l.val = a % 32; omega)

/-- **A gather stage**: a map whose *inputs* are addressed by arbitrary index
    expressions, elaborated against the launch's integer state.

    `mapStage` fixes every input index to `elemIx`, which is right for an
    element-wise kernel and wrong for the one shipped kernel that reads a token
    id at run time.  Splitting the input addressing out costs nothing — the
    output addressing is still `elemIx`, so `dom` and the exclusivity argument
    are unchanged — and it is what makes an embedding lookup a stage. -/
def gatherStage {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE)
    (out : Buf) (grid : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (hio : ∀ i, inB i ≠ out) : StageSpec where
  ew    := (mapKernelAt spec inB inIx out).ew
  iregs := ir
  imem  := im
  grid  := grid
  out   := out
  dom   := fun cta a => ∃ l : Lane, cta * 32 + l.val = a
  val   := fun m cta a =>
    denote (fun i => m (inB i) ((inIx i).eval cta 0 (laneMod a) ir im)) spec
  frame := by
    intro cta st b a hb
    rcases hb with hb | hb
    · exact congrFun (compileWKernel_otherBuf inB out inIx spec elemIx
        (Γ + slots spec + 1) cta 0 ir im st b hb) a
    · by_cases hbo : b = out
      · subst hbo
        exact compileWKernel_otherAddr inB b inIx spec elemIx
          (Γ + slots spec + 1) cta 0 ir im st a (fun l hc => hb ⟨l, hc⟩)
      · exact congrFun (compileWKernel_otherBuf inB out inIx spec elemIx
          (Γ + slots spec + 1) cta 0 ir im st b hbo) a
  value := by
    intro cta st a hdom
    obtain ⟨l, hl⟩ := hdom
    have hlm : l = laneMod a := laneMod_of hl
    have hix : elemIx.eval cta 0 l ir im = a := hl
    rw [← hix]
    show (((compileWKernel inB out inIx spec elemIx (Γ + slots spec + 1)).elabAt cta 0 ir im).run
            st).mem out (elemIx.eval cta 0 l ir im) = _
    rw [compileWKernel_storesAt inB out inIx spec elemIx (Γ + slots spec + 1) cta 0 ir im st l
      (fun l' h => by
        have hv : cta * 32 + l'.val = cta * 32 + l.val := h
        have : l' = l := Fin.ext (by have := l.isLt; have := l'.isLt; omega)
        rw [this])]
    rw [hix, hlm]
  valOnly := by
    intro m m' _ a _ h _
    refine congrArg (fun f => denote f spec) (funext fun i => ?_)
    rw [h (inB i) (hio i)]

/-- Its blocks never collide either — same `elemIx` output addressing. -/
theorem gatherStage_exclusive {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf)
    (inIx : Fin Γ → IdxE) (out : Buf) (grid : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (hio : ∀ i, inB i ≠ out) :
    (gatherStage spec inB inIx out grid ir im hio).Exclusive := by
  refine StageSpec.Exclusive.ofUnbounded ?_
  intro cta cta' a h h'
  obtain ⟨l, hl⟩ := h
  obtain ⟨l', hl'⟩ := h'
  exact elemIx_blocks_disjoint cta cta' l l' (by rw [hl, hl'])

/-- The map stage's blocks never collide. -/
theorem mapStage_exclusive {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf)
    (grid : Nat) (hio : ∀ i, inB i ≠ out) :
    (mapStage spec inB out grid hio).Exclusive := by
  refine StageSpec.Exclusive.ofUnbounded ?_
  intro cta cta' a h h'
  obtain ⟨l, hl⟩ := h
  obtain ⟨l', hl'⟩ := h'
  exact elemIx_blocks_disjoint cta cta' l l' (by rw [hl, hl'])

-- ---------------------------------------------------------------------------
-- A reduction is a stage too
-- ---------------------------------------------------------------------------

/-- A single store leaves every other address alone. -/
theorem mem_store1_other (st : WSt) (b : Buf) (i : Nat) (v : Float32) (c : Buf)
    (j : Nat) (h : ¬ (c = b ∧ j = i)) : (st.store1 b i v).mem c j = st.mem c j := by
  simp [WSt.store1, h]

/-- **The strided dot's only memory effect is its one lane-0 store.** -/
theorem dotStrided_frame (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (oi : IdxE)
    (K cta : Nat) (st : WSt) (c : Buf) (j : Nat)
    (h : ¬ (c = out ∧ j = oi.eval cta 0 ⟨0, by decide⟩)) :
    (((dotStrided bA bB ixA ixB out oi K).elabIn cta).run st).mem c j = st.mem c j := by
  show ((WStmt.storeLane0 out (oi.eval cta 0 ⟨0, by decide⟩) 0).run
          ((warpReduceSum 0 1).run
            (((dotStridedBody bA bB ixA ixB K).elabAt cta 0).run st))).mem c j = _
  rw [wrun_storeLane0, mem_store1_other _ _ _ _ _ _ h, warpReduceSum_mem,
      dotStridedBody_mem]

/-- **A strided reduction, as a pipeline stage.**

    One output per block at `%ctaid`, so ownership is `a = cta` — exclusive by
    construction, which is why this shape needs no address argument at all.
    The value is the committed two-level fold, read off `dotStrided_spec`,
    which was already stated at an arbitrary block. -/
def reduceStage (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (K grid : Nat)
    (hA : bA ≠ out) (hB : bB ≠ out) : StageSpec where
  ew   := dotStrided bA bB ixA ixB out .ctaId K
  grid := grid
  out  := out
  dom  := fun cta a => a = cta
  val  := fun m cta _ => bflyFold (dotStridedLane (m bA) (m bB)
            (fun i l => ixA.eval cta i l) (fun i l => ixB.eval cta i l) K) ⟨0, by decide⟩
  frame := by
    intro cta st b a hb
    refine dotStrided_frame bA bB ixA ixB out .ctaId K cta st b a ?_
    intro hc
    rcases hb with hb | hb
    · exact hb hc.1
    · exact hb hc.2
  value := by
    intro cta st a hdom
    subst hdom
    exact dotStrided_spec bA bB ixA ixB out .ctaId K a st
  valOnly := by
    intro m m' _ a _ h _
    rw [h bA hA, h bB hB]

/-- **The reduce stage is idempotent** — it folds two buffers, neither `out`. -/
theorem reduceStage_idempotent (bA bB : Buf) (ixA ixB : IdxE) (out : Buf)
    (K grid : Nat) (hA : bA ≠ out) (hB : bB ≠ out) :
    (reduceStage bA bB ixA ixB out K grid hA hB).Idempotent := by
  intro m m' cta a h
  show bflyFold (dotStridedLane (m bA) (m bB) _ _ K) _
     = bflyFold (dotStridedLane (m' bA) (m' bB) _ _ K) _
  rw [h bA hA, h bB hB]

theorem reduceStage_exclusive (bA bB : Buf) (ixA ixB : IdxE) (out : Buf)
    (K grid : Nat) (hA : bA ≠ out) (hB : bB ≠ out) :
    (reduceStage bA bB ixA ixB out K grid hA hB).Exclusive := by
  refine StageSpec.Exclusive.ofUnbounded ?_
  intro cta cta' a h h'
  show cta = cta'
  rw [← h, ← h']

-- ---------------------------------------------------------------------------
-- A store pass is a stage too
-- ---------------------------------------------------------------------------

/-- The two-buffer store pass touches memory only inside its output buffer, and
    only at addresses one of its `(iteration, lane)` pairs owns. -/
theorem zipPass_frame (bA bB out : Buf) (dA dB r : Nat) (f : WFExp)
    (ixA ixB oix : IdxE) (K cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (c : Buf) (a : Nat)
    (h : c ≠ out ∨ ∀ j ∈ List.range K, ∀ l : Lane, oix.eval cta j l ir im ≠ a) :
    (((zipPassEW bA bB out dA dB r f ixA ixB oix K).elabAt cta 0 ir im).run st).mem c a
      = st.mem c a := by
  have hmem : ∀ (j : Nat) (s : WSt),
      (((EWStmt.seq (.loadIdx dA bA ixA)
          (.seq (.loadIdx dB bB ixB) (.setR r f))).elabAt cta j ir im).run s).mem
        = s.mem := fun _ _ => rfl
  rcases h with hc | hno
  · exact congrFun (storeLoop_otherBuf out r oix _ cta ir im hmem c hc (List.range K) st) a
  · by_cases hco : c = out
    · subst hco
      exact storeLoop_otherAddr _ r oix _ cta ir im hmem a (List.range K) st hno
    · exact congrFun (storeLoop_otherBuf out r oix _ cta ir im hmem c hco (List.range K) st) a

/-- **The outer product, as a pipeline stage.**

    `dW[i·n + e] = adj[i] · x[e]`, one warp per row `i = ctaid`.  The output
    address determines the element as `a − cta·n`, which is why `val` takes the
    block index: the row comes from the block and the column from the address.

    Blocks are disjoint because row `cta` owns exactly `[cta·n, cta·n + K·32)`
    and `K·32 = n` — so this stage depends on the geometry guard
    (`ReduceGeom.covers`) being discharged. -/
def outerStage (bAdj bX out : Buf) (n K grid : Nat) (_hn : K * 32 = n)
    (hAo : bAdj ≠ out) (hXo : bX ≠ out) : StageSpec where
  ew   := zipPassEW bAdj bX out 1 2 0 (.mul (.reg 1) (.reg 2)) .ctaId
            (stride32 (.lit 0)) (stride32 (.mul .ctaId (.lit n))) K
  grid := grid
  out  := out
  dom  := fun cta a => ∃ j, j < K ∧ ∃ l : Lane, cta * n + (j * 32 + l.val) = a
  val  := fun m cta a => NumOps.mul (m bAdj cta) (m bX (a - cta * n))
  frame := by
    intro cta st b a hb
    refine zipPass_frame bAdj bX out 1 2 0 _ .ctaId (stride32 (.lit 0))
      (stride32 (.mul .ctaId (.lit n))) K cta _ _ st b a ?_
    rcases hb with hb | hb
    · exact Or.inl hb
    · exact Or.inr (fun j hj l hc => hb ⟨j, List.mem_range.mp hj, l, hc⟩)
  value := by
    intro cta st a hdom
    obtain ⟨j, hj, l, hl⟩ := hdom
    subst hl
    have h := zipPass_spec bAdj bX out 1 2 0 (by decide)
      (.mul (.reg 1) (.reg 2)) (.mul .ctaId (.lit n)) rfl .ctaId
      (stride32 (.lit 0)) K cta hAo hXo (fun _ _ => 0) (fun _ _ => 0) st
      (fun x y => NumOps.mul x y) (fun _ _ => rfl) j l hj
    have hsub : cta * n + (j * 32 + l.val) - cta * n = j * 32 + l.val := by omega
    have h' : (((zipPassEW bAdj bX out 1 2 0 (.mul (.reg 1) (.reg 2)) .ctaId
        (stride32 (.lit 0)) (stride32 (.mul .ctaId (.lit n))) K).elabAt cta 0
        (fun _ _ => 0) (fun _ _ => 0)).run st).mem out (cta * n + (j * 32 + l.val))
          = NumOps.mul (st.mem bAdj cta) (st.mem bX (0 + (j * 32 + l.val))) := h
    rw [Nat.zero_add] at h'
    rw [hsub]
    show ((( zipPassEW bAdj bX out 1 2 0 (.mul (.reg 1) (.reg 2)) .ctaId
        (stride32 (.lit 0)) (stride32 (.mul .ctaId (.lit n))) K).elabAt cta 0
        (fun _ _ => 0) (fun _ _ => 0)).run st).mem out _ = _
    exact h'
  valOnly := by
    intro m m' _ a _ h _
    rw [h bAdj hAo, h bX hXo]

/-- **The outer-product stage is idempotent** — both factors are read from
    buffers distinct from `out`. -/
theorem outerStage_idempotent (bAdj bX out : Buf) (n K grid : Nat) (hn : K * 32 = n)
    (hAo : bAdj ≠ out) (hXo : bX ≠ out) :
    (outerStage bAdj bX out n K grid hn hAo hXo).Idempotent := by
  intro m m' cta a h
  show NumOps.mul (m bAdj cta) (m bX (a - cta * n))
     = NumOps.mul (m' bAdj cta) (m' bX (a - cta * n))
  rw [h bAdj hAo, h bX hXo]

/-- Rows own disjoint address ranges, given that a row's trips cover it exactly. -/
theorem outerStage_exclusive (bAdj bX out : Buf) (n K grid : Nat) (hn : K * 32 = n)
    (hAo : bAdj ≠ out) (hXo : bX ≠ out) :
    (outerStage bAdj bX out n K grid hn hAo hXo).Exclusive := by
  refine StageSpec.Exclusive.ofUnbounded ?_
  intro cta cta' a h h'
  obtain ⟨j, hj, l, hl⟩ := h
  obtain ⟨j', hj', l', hl'⟩ := h'
  have h1 : l.val < 32 := l.isLt
  have h2 : l'.val < 32 := l'.isLt
  have hb : j * 32 + l.val < n := by
    have : j + 1 ≤ K := hj
    have : (j + 1) * 32 ≤ K * 32 := Nat.mul_le_mul_right 32 this
    rw [Nat.succ_mul] at this
    omega
  have hb' : j' * 32 + l'.val < n := by
    have : j' + 1 ≤ K := hj'
    have : (j' + 1) * 32 ≤ K * 32 := Nat.mul_le_mul_right 32 this
    rw [Nat.succ_mul] at this
    omega
  have heq : cta * n + (j * 32 + l.val) = cta' * n + (j' * 32 + l'.val) := by
    rw [hl, hl']
  exact (rowMajor_inj hb hb' (by
    show cta * n + (j * 32 + l.val) = cta' * n + (j' * 32 + l'.val)
    exact heq)).1

-- ---------------------------------------------------------------------------
-- The discharge
-- ---------------------------------------------------------------------------

/-- **Two stages compose, and the second's hypothesis is discharged by the
    first's conclusion.**

    Stage `A` computes `fA` from `b0` into `b1`; stage `B` computes `fB` from
    `b1` into `b2`; and the buffer `B` writes holds `fB ∘ fA` — with the
    intermediate never assumed, only *derived*.

    Everything the user supplies is disequalities between buffer numbers.  No
    proof about kernels, addresses, blocks or frames reaches them: the stage
    constructor discharged those, and this theorem discharges the link. -/
theorem two_map_stages (fA fB : Expr 1) (b0 b1 b2 : Buf)
    (h01 : b0 ≠ b1) (h12 : b1 ≠ b2) (grid : Nat) (st : WSt)
    (cta : Nat) (l : Lane) (hlt : cta < grid) :
    ((mapStage fB (fun _ => b1) b2 grid (fun _ => h12)).run
      ((mapStage fA (fun _ => b0) b1 grid (fun _ => h01)).run st)).mem b2
        (cta * 32 + l.val)
      = denote (fun _ => denote (fun _ => st.mem b0 (cta * 32 + l.val)) fA) fB := by
  have hB := (mapStage fB (fun _ => b1) b2 grid (fun _ => h12)).run_value
    (mapStage_exclusive fB (fun _ => b1) b2 grid (fun _ => h12))
    ((mapStage fA (fun _ => b0) b1 grid (fun _ => h01)).run st)
    (cta * 32 + l.val) cta hlt ⟨l, rfl⟩
  have hA := (mapStage fA (fun _ => b0) b1 grid (fun _ => h01)).run_value
    (mapStage_exclusive fA (fun _ => b0) b1 grid (fun _ => h01)) st
    (cta * 32 + l.val) cta hlt ⟨l, rfl⟩
  show ((mapStage fB (fun _ => b1) b2 grid (fun _ => h12)).run
      ((mapStage fA (fun _ => b0) b1 grid (fun _ => h01)).run st)).mem
      (mapStage fB (fun _ => b1) b2 grid (fun _ => h12)).out (cta * 32 + l.val) = _
  rw [hB]
  show denote (fun _ => ((mapStage fA (fun _ => b0) b1 grid (fun _ => h01)).run st).mem b1
        (cta * 32 + l.val)) fB = _
  show denote (fun _ => ((mapStage fA (fun _ => b0) b1 grid (fun _ => h01)).run st).mem
        (mapStage fA (fun _ => b0) b1 grid (fun _ => h01)).out (cta * 32 + l.val)) fB = _
  rw [hA]
  rfl

/-- **A map feeding a reduction — the shape of a backward pass.**

    Stage `A` computes an adjoint elementwise into `b1`; stage `B` reduces `b1`
    against a weight buffer into `b2`.  That is `ds = dy·act'(z)` followed by
    `dx = Wᵀ·ds`, and the theorem says the reduction folds *the adjoint the
    first stage actually produced* — the intermediate is derived, never assumed.

    Where the composition shows up: `dotStridedLane` is applied to
    `fun a => denote … fA`, the elementwise stage's meaning substituted pointwise
    into the reduction's fold.

    `hcov` is the scheduling obligation: every address the reduction reads must
    be one the map stage wrote.  A schedule that reduces over elements nobody
    computed is wrong, and this is where that shows up. -/
theorem map_then_reduce {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf)
    (b1 bW b2 : Buf) (ixA ixW : IdxE)
    (h01 : ∀ i, inB i ≠ b1) (h1 : b1 ≠ b2) (hW : bW ≠ b2) (hWb1 : bW ≠ b1)
    (K gridA gridB : Nat) (st : WSt) (cta : Nat) (hlt : cta < gridB)
    (hcov : ∀ (i : Nat), i < K → ∀ (l : Lane), ∃ c : Nat, c < gridA ∧ ∃ l' : Lane,
      c * 32 + l'.val = ixA.eval cta i l) :
    ((reduceStage b1 bW ixA ixW b2 K gridB h1 hW).run
      ((mapStage spec inB b1 gridA h01).run st)).mem b2 cta
      = bflyFold (dotStridedLane
          (fun a => denote (fun i => st.mem (inB i) a) spec) (st.mem bW)
          (fun i l => ixA.eval cta i l) (fun i l => ixW.eval cta i l) K)
          ⟨0, by decide⟩ := by
  have hA := (mapStage spec inB b1 gridA h01).run_value
    (mapStage_exclusive spec inB b1 gridA h01) st
  have hfrm := (mapStage spec inB b1 gridA h01).run_frame st bW hWb1
  have hB := (reduceStage b1 bW ixA ixW b2 K gridB h1 hW).run_value
    (reduceStage_exclusive b1 bW ixA ixW b2 K gridB h1 hW)
    ((mapStage spec inB b1 gridA h01).run st) cta cta hlt rfl
  show ((reduceStage b1 bW ixA ixW b2 K gridB h1 hW).run _).mem
      (reduceStage b1 bW ixA ixW b2 K gridB h1 hW).out cta = _
  rw [hB]
  show bflyFold (dotStridedLane
      (((mapStage spec inB b1 gridA h01).run st).mem b1)
      (((mapStage spec inB b1 gridA h01).run st).mem bW)
      _ _ K) _ = _
  rw [hfrm]
  congr 1
  funext l
  refine dotStridedLane_congr _ _ _ _ _ _ K l (fun i hi l' => ?_) (fun _ _ _ => rfl)
  obtain ⟨c, hc, l'', hl''⟩ := hcov i hi l'
  rw [← hl'']
  exact hA (c * 32 + l''.val) c hc ⟨l'', rfl⟩

end AlgorithmLib.ML
