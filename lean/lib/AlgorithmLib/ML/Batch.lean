import AlgorithmLib.ML.Kernels

/-!
  # Batching, without a new address constructor

  A batched matvec computes `z[s][i] = Σⱼ W[i][j]·x[s][j]` for `s < B`.  The
  obvious lowering — one block per `(s, i)` pair — needs the block index split
  into `(s, i)`, i.e. `ctaid / n` and `ctaid % n`, and `IdxE` has `add` and
  `mul` and nothing else.  That is the reading under which batching is blocked.

  It is the wrong lowering anyway.  Splitting the grid gives every `(s, i)` its
  own warp, so `W[i][·]` is read once **per sample** — the same weight traffic
  per sample as batch 1, which is the entire cost that batching exists to
  amortise.  The right lowering keeps one warp per output row and gives it `B`
  accumulators: the row of `W` is loaded once and multiplied into all `B`
  samples.  Weight traffic per sample falls by `B`, and every address stays
  affine in `(ctaid, loop, lane)` with a literal offset, because `B` is known
  when the kernel is generated.

  So batching needed no new addressing.  What it needed is here:

  * `dotBatched` — one warp, `B` accumulators, `B` outputs.  Its per-sample
    guarantee is *the same statement* `dotStrided_spec` makes, which is what
    says batching changed the schedule and not the arithmetic.
  * `outerBatched` — the gradient accumulation `dW[i][j] = Σₛ adj[s][i]·x[s][j]`.
    A grid split cannot express this at all: two blocks accumulating into one
    `dW` element is a race, so the sum over samples has to happen inside a
    warp.

  The register file is the budget: `B` accumulators plus three scratch, so
  `B ≤ 32` is comfortable on any target this stack emits for.
-/

namespace AlgorithmLib.ML

-- ---------------------------------------------------------------------------
-- Sequencing a statically-known number of copies
-- ---------------------------------------------------------------------------

/-- `f 0`, then `f 1`, …, then `f (n-1)`.

    Left-extending — `seqN (n+1) f` runs `seqN n f` and *then* `f n` — so that
    an induction on `n` peels the last copy, which is the one whose effect the
    lemmas below describe. -/
def seqN : Nat → (Nat → EWStmt) → EWStmt
  | 0,   _ => .skip
  | n+1, f => .seq (seqN n f) (f n)

/-- The same shape in the machine language. -/
def seqNW : Nat → (Nat → WStmt) → WStmt
  | 0,   _ => .skip
  | n+1, f => .seq (seqNW n f) (f n)

theorem seqN_elab (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (f : Nat → EWStmt) : ∀ n : Nat,
    (seqN n f).elabAt cta i ir im = seqNW n (fun s => (f s).elabAt cta i ir im) := by
  intro n
  induction n with
  | zero => rfl
  | succ k ih => show WStmt.seq _ _ = WStmt.seq _ _; rw [ih]

/-- **A register no copy writes survives the whole sequence.** -/
theorem seqNW_regs_frame (f : Nat → WStmt) (r : Nat) : ∀ n : Nat,
    (∀ s, s < n → ∀ st : WSt, ((f s).run st).regs r = st.regs r) →
    ∀ st : WSt, ((seqNW n f).run st).regs r = st.regs r := by
  intro n
  induction n with
  | zero => intro _ _; rfl
  | succ k ih =>
      intro h st
      show ((f k).run ((seqNW k f).run st)).regs r = _
      rw [h k (Nat.lt_succ_self k), ih (fun s hs => h s (Nat.lt_succ_of_lt hs)) st]

/-- **…and so does memory, if no copy touches it.** -/
theorem seqNW_mem_frame (f : Nat → WStmt) : ∀ n : Nat,
    (∀ s, s < n → ∀ st : WSt, ((f s).run st).mem = st.mem) →
    ∀ st : WSt, ((seqNW n f).run st).mem = st.mem := by
  intro n
  induction n with
  | zero => intro _ _; rfl
  | succ k ih =>
      intro h st
      show ((f k).run ((seqNW k f).run st)).mem = _
      rw [h k (Nat.lt_succ_self k), ih (fun s hs => h s (Nat.lt_succ_of_lt hs)) st]

-- ---------------------------------------------------------------------------
-- The register file
-- ---------------------------------------------------------------------------

/-- `%fw0` — the butterfly's shuffle temporary. -/
def BTMP : Nat := 0
/-- `%fw1` — the operand shared across the batch (a weight). -/
def BRA : Nat := 1
/-- `%fw2` — the per-sample operand (an activation). -/
def BRB : Nat := 2
/-- `%fw3` and up — sample `s`'s accumulator.  Distinct from the three scratch
    registers and from each other, which is every disjointness side condition
    the proofs below need. -/
def BACC : Nat := 3

theorem bacc_ne_tmp (s : Nat) : BACC + s ≠ BTMP := by simp [BACC, BTMP]
theorem bacc_ne_ra  (s : Nat) : BACC + s ≠ BRA  := by simp [BACC, BRA]; omega
theorem bacc_ne_rb  (s : Nat) : BACC + s ≠ BRB  := by simp [BACC, BRB]; omega
theorem bacc_inj {s s' : Nat} (h : s ≠ s') : BACC + s ≠ BACC + s' := by omega

-- ---------------------------------------------------------------------------
-- The batched reduction
-- ---------------------------------------------------------------------------

/-- One sample's accumulation step: `acc_s += a·b_s`. -/
def batchStepE (bB : Buf) (ixB : Nat → IdxE) (s : Nat) : EWStmt :=
  .seq (.loadIdx BRB bB (ixB s))
       (.setR (BACC + s) (.add (.reg (BACC + s)) (.mul (.reg BRA) (.reg BRB))))

/-- One trip of the sweep: load the shared operand **once**, then multiply it
    into all `B` accumulators.  This single line is what batching buys. -/
def batchTripE (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (B : Nat) : EWStmt :=
  .seq (.loadIdx BRA bA ixA) (seqN B (batchStepE bB ixB))

/-- Zero the accumulators, then sweep `K` trips. -/
def batchBodyE (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (B K : Nat) : EWStmt :=
  .seq (seqN B (fun s => .setR (BACC + s) (.lit (NumOps.ofNat 0))))
       (.forN K (batchTripE bA bB ixA ixB B))

/-- One butterfly round on an explicit accumulator/temporary pair. -/
def bflyRoundE (acc tmp mask : Nat) (op : WFExp) : EWStmt :=
  .seq (.shflXor tmp acc mask) (.setR acc op)

/-- The five-round sum reduction, on an explicit pair.  `warpRoundE` is this at
    the fixed pair `(0, 1)`; a batched kernel needs one per accumulator. -/
def warpReduceE (acc tmp : Nat) : EWStmt :=
  .seq (bflyRoundE acc tmp 16 (.add (.reg acc) (.reg tmp)))
    (.seq (bflyRoundE acc tmp 8 (.add (.reg acc) (.reg tmp)))
      (.seq (bflyRoundE acc tmp 4 (.add (.reg acc) (.reg tmp)))
        (.seq (bflyRoundE acc tmp 2 (.add (.reg acc) (.reg tmp)))
              (bflyRoundE acc tmp 1 (.add (.reg acc) (.reg tmp))))))

/-- Reduce and store each sample's accumulator. -/
def batchEpilogueE (out : Buf) (oi : Nat → IdxE) (B : Nat) : EWStmt :=
  seqN B (fun s =>
    .seq (warpReduceE (BACC + s) BTMP) (.storeLane0 out (oi s) (BACC + s)))

/-- **The batched strided reduction**: one warp, `B` accumulators, `B` stores. -/
def dotBatched (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (out : Buf)
    (oi : Nat → IdxE) (B K : Nat) : EWStmt :=
  .seq (batchBodyE bA bB ixA ixB B K) (batchEpilogueE out oi B)

-- ---------------------------------------------------------------------------
-- What one accumulation step does
-- ---------------------------------------------------------------------------

/-- The elaborated step, with the addressing already resolved. -/
def bstepW (bB : Buf) (fB : Nat → Lane → Nat) (s : Nat) : WStmt :=
  .seq (.loadIdx BRB bB (fB s))
       (.setR (BACC + s) (.add (.reg (BACC + s)) (.mul (.reg BRA) (.reg BRB))))

theorem batchStepE_elab (bB : Buf) (ixB : Nat → IdxE) (s cta i : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    (batchStepE bB ixB s).elabAt cta i ir im
      = bstepW bB (fun s' l => (ixB s').eval cta i l ir im) s := rfl

theorem bstepW_mem (bB : Buf) (fB : Nat → Lane → Nat) (s : Nat) (st : WSt) :
    ((bstepW bB fB s).run st).mem = st.mem := rfl

/-- A step writes `%fw2` and its own accumulator, and nothing else. -/
theorem bstepW_frame (bB : Buf) (fB : Nat → Lane → Nat) (s r : Nat)
    (h1 : r ≠ BRB) (h2 : r ≠ BACC + s) (st : WSt) :
    ((bstepW bB fB s).run st).regs r = st.regs r := by
  show ((WStmt.setR (BACC + s) _).run ((WStmt.loadIdx BRB bB (fB s)).run st)).regs r = _
  rw [wrun_setR, WSt.regs_setReg_other _ (BACC + s) r _ h2, wrun_loadIdx,
      WSt.regs_setReg_other _ BRB r _ h1]

/-- …and what it puts there. -/
theorem bstepW_acc (bB : Buf) (fB : Nat → Lane → Nat) (s : Nat) (st : WSt) :
    ((bstepW bB fB s).run st).regs (BACC + s)
      = fun l => NumOps.add (st.regs (BACC + s) l)
                   (NumOps.mul (st.regs BRA l) (st.mem bB (fB s l))) := by
  show ((WStmt.setR (BACC + s) _).run ((WStmt.loadIdx BRB bB (fB s)).run st)).regs (BACC + s) = _
  rw [wrun_setR, WSt.regs_setReg_same]
  funext l
  simp only [WFExp.eval, wrun_loadIdx, WSt.regs_setReg_same,
    WSt.regs_setReg_other _ BRB (BACC + s) _ (bacc_ne_rb s),
    WSt.regs_setReg_other _ BRB BRA _ (by decide : BRA ≠ BRB), WSt.mem_setReg]

-- ---------------------------------------------------------------------------
-- …and what the B copies of it do together
-- ---------------------------------------------------------------------------

theorem batchInner_mem (bB : Buf) (fB : Nat → Lane → Nat) (B : Nat) (st : WSt) :
    ((seqNW B (bstepW bB fB)).run st).mem = st.mem :=
  seqNW_mem_frame _ B (fun s _ st' => bstepW_mem bB fB s st') st

/-- The shared operand survives the whole batch — which is the point: it is
    loaded once and used `B` times. -/
theorem batchInner_ra (bB : Buf) (fB : Nat → Lane → Nat) (B : Nat) (st : WSt) :
    ((seqNW B (bstepW bB fB)).run st).regs BRA = st.regs BRA :=
  seqNW_regs_frame _ BRA B
    (fun s _ st' => bstepW_frame bB fB s BRA (by decide) (Ne.symm (bacc_ne_ra s)) st') st

/-- An accumulator no copy in the first `n` owns is untouched by them. -/
theorem batchInner_other (bB : Buf) (fB : Nat → Lane → Nat) (s : Nat) : ∀ n : Nat, n ≤ s →
    ∀ st : WSt, ((seqNW n (bstepW bB fB)).run st).regs (BACC + s) = st.regs (BACC + s) := by
  intro n hn st
  exact seqNW_regs_frame _ (BACC + s) n
    (fun s' hs' st' => bstepW_frame bB fB s' (BACC + s) (bacc_ne_rb s)
      (bacc_inj (by omega)) st') st

/-- **Each of the `B` accumulators takes its own product.** -/
theorem batchInner_acc (bB : Buf) (fB : Nat → Lane → Nat) : ∀ B s : Nat, s < B →
    ∀ st : WSt, ((seqNW B (bstepW bB fB)).run st).regs (BACC + s)
      = fun l => NumOps.add (st.regs (BACC + s) l)
                   (NumOps.mul (st.regs BRA l) (st.mem bB (fB s l))) := by
  intro B
  induction B with
  | zero => intro s hs; exact absurd hs (by omega)
  | succ k ih =>
      intro s hs st
      show ((bstepW bB fB k).run ((seqNW k (bstepW bB fB)).run st)).regs (BACC + s) = _
      rcases Nat.lt_or_ge s k with h | h
      · rw [bstepW_frame bB fB k (BACC + s) (bacc_ne_rb s) (bacc_inj (by omega)),
            ih s h st]
      · have hsk : s = k := by omega
        subst hsk
        rw [bstepW_acc bB fB s, batchInner_other bB fB s s (Nat.le_refl s) st,
            batchInner_ra bB fB s st, batchInner_mem bB fB s st]

-- ---------------------------------------------------------------------------
-- One trip, and the sweep
-- ---------------------------------------------------------------------------

/-- One elaborated trip: load the shared operand, accumulate into every sample. -/
def btripW (bA bB : Buf) (fA : Lane → Nat) (fB : Nat → Lane → Nat) (B : Nat) : WStmt :=
  .seq (.loadIdx BRA bA fA) (seqNW B (bstepW bB fB))

theorem batchTripE_elab (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (B cta i : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    (batchTripE bA bB ixA ixB B).elabAt cta i ir im
      = btripW bA bB (fun l => ixA.eval cta i l ir im)
          (fun s l => (ixB s).eval cta i l ir im) B := by
  show WStmt.seq _ _ = WStmt.seq _ _
  rw [seqN_elab]
  rfl

theorem btripW_mem (bA bB : Buf) (fA : Lane → Nat) (fB : Nat → Lane → Nat)
    (B : Nat) (st : WSt) : ((btripW bA bB fA fB B).run st).mem = st.mem := by
  show ((seqNW B (bstepW bB fB)).run ((WStmt.loadIdx BRA bA fA).run st)).mem = _
  rw [batchInner_mem]
  rfl

/-- **A trip adds one product to every accumulator, from the entry memory.** -/
theorem btripW_acc (bA bB : Buf) (fA : Lane → Nat) (fB : Nat → Lane → Nat)
    (B s : Nat) (hs : s < B) (st : WSt) :
    ((btripW bA bB fA fB B).run st).regs (BACC + s)
      = fun l => NumOps.add (st.regs (BACC + s) l)
                   (NumOps.mul (st.mem bA (fA l)) (st.mem bB (fB s l))) := by
  show ((seqNW B (bstepW bB fB)).run ((WStmt.loadIdx BRA bA fA).run st)).regs (BACC + s) = _
  rw [batchInner_acc bB fB B s hs]
  simp only [wrun_loadIdx, WSt.regs_setReg_same, WSt.mem_setReg,
    WSt.regs_setReg_other _ BRA (BACC + s) _ (bacc_ne_ra s)]

/-- **The sweep is the committed per-lane fold — one per sample.**

    Same shape as `dotStrided_fold`: sequential within a lane, in load order.
    A batched kernel therefore commits to the *same* arithmetic order as the
    unbatched one, which is why the two agree bit-for-bit. -/
theorem batchSweep_fold (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE)
    (B s cta : Nat) (hs : s < B) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ (L : List Nat) (st : WSt) (l : Lane),
      ((L.foldl (fun s' j => ((batchTripE bA bB ixA ixB B).elabAt cta j ir im).run s')
          st).regs (BACC + s) l)
        = L.foldl (fun acc j => NumOps.add acc
            (NumOps.mul (st.mem bA (ixA.eval cta j l ir im))
                        (st.mem bB ((ixB s).eval cta j l ir im))))
            (st.regs (BACC + s) l) := by
  intro L
  induction L with
  | nil => intro _ _; rfl
  | cons j L ih =>
      intro st l
      rw [List.foldl_cons, List.foldl_cons, ih]
      rw [batchTripE_elab bA bB ixA ixB B cta j ir im]
      rw [btripW_mem, btripW_acc bA bB _ _ B s hs st]

theorem batchInit_mem (B : Nat) (st : WSt) :
    ((seqNW B (fun s => WStmt.setR (BACC + s) (.lit (NumOps.ofNat 0)))).run st).mem
      = st.mem :=
  seqNW_mem_frame _ B (fun _ _ _ => rfl) st

theorem batchInit_acc : ∀ B s : Nat, s < B → ∀ st : WSt,
    ((seqNW B (fun s => WStmt.setR (BACC + s) (.lit (NumOps.ofNat 0)))).run st).regs (BACC + s)
      = fun _ => NumOps.ofNat 0 := by
  intro B
  induction B with
  | zero => intro s hs; exact absurd hs (by omega)
  | succ k ih =>
      intro s hs st
      show ((WStmt.setR (BACC + k) _).run ((seqNW k _).run st)).regs (BACC + s) = _
      rcases Nat.lt_or_ge s k with h | h
      · rw [wrun_setR, WSt.regs_setReg_other _ (BACC + k) (BACC + s) _ (bacc_inj (by omega)),
            ih s h st]
      · have : s = k := by omega
        subst this
        rw [wrun_setR, WSt.regs_setReg_same]
        rfl

/-- **Each accumulator holds its own sample's committed fold.** -/
theorem batchBody_spec (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE)
    (B K s cta : Nat) (hs : s < B) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) :
    (((batchBodyE bA bB ixA ixB B K).elabAt cta 0 ir im).run st).regs (BACC + s)
      = dotStridedLane (st.mem bA) (st.mem bB)
          (fun i l => ixA.eval cta i l ir im)
          (fun i l => (ixB s).eval cta i l ir im) K := by
  funext l
  show ((List.range K).foldl
      (fun s' j => ((batchTripE bA bB ixA ixB B).elabAt cta j ir im).run s')
      (((seqN B (fun s => EWStmt.setR (BACC + s) (.lit (NumOps.ofNat 0)))).elabAt
          cta 0 ir im).run st)).regs (BACC + s) l = _
  rw [seqN_elab]
  rw [batchSweep_fold bA bB ixA ixB B s cta hs ir im (List.range K) _ l]
  -- `elabAt` of a register write is that write; say so, so the init lemmas match.
  have hinit : (fun s => EWStmt.elabAt cta 0 ir im
        (EWStmt.setR (BACC + s) (WFExp.lit (NumOps.ofNat 0))))
      = (fun s => WStmt.setR (BACC + s) (WFExp.lit (NumOps.ofNat 0))) := rfl
  rw [hinit, batchInit_mem, batchInit_acc B s hs st]
  rfl

-- ---------------------------------------------------------------------------
-- The epilogue: B butterflies, B stores
-- ---------------------------------------------------------------------------

theorem mem_store1_at (st : WSt) (b : Buf) (i : Nat) (v : Float32) (j : Nat)
    (h : j ≠ i) : (st.store1 b i v).mem b j = st.mem b j := by
  simp [WSt.store1, h]

/-- The elaborated five-round reduction on an explicit pair. -/
def wredW (acc tmp : Nat) : WStmt :=
  .seq (bflyRoundOp acc tmp 16 (.add (.reg acc) (.reg tmp)))
    (.seq (bflyRoundOp acc tmp 8 (.add (.reg acc) (.reg tmp)))
      (.seq (bflyRoundOp acc tmp 4 (.add (.reg acc) (.reg tmp)))
        (.seq (bflyRoundOp acc tmp 2 (.add (.reg acc) (.reg tmp)))
              (bflyRoundOp acc tmp 1 (.add (.reg acc) (.reg tmp))))))

theorem warpReduceE_elab (acc tmp cta i : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) : (warpReduceE acc tmp).elabAt cta i ir im = wredW acc tmp := rfl

theorem wredW_mem (acc tmp : Nat) (st : WSt) : ((wredW acc tmp).run st).mem = st.mem := rfl

/-- The reduction realises the same butterfly tree the unbatched schema commits
    to — `warpReduceOp_spec` at `add`, on this accumulator. -/
theorem wredW_spec (acc tmp : Nat) (h : tmp ≠ acc) (st : WSt) :
    ((wredW acc tmp).run st).regs acc = bflyFold (st.regs acc) :=
  warpReduceOp_spec acc tmp h _ NumOps.add (fun _ _ => rfl) st

theorem wredW_frame (acc tmp r : Nat) (hr : r ≠ acc) (hr' : r ≠ tmp) (st : WSt) :
    ((wredW acc tmp).run st).regs r = st.regs r :=
  warpReduceOp_frame acc tmp _ r hr hr' st

/-- One sample's epilogue: reduce its accumulator, store it. -/
def epCopyW (out : Buf) (addr : Nat → Nat) (s : Nat) : WStmt :=
  .seq (wredW (BACC + s) BTMP) (.storeLane0 out (addr s) (BACC + s))

theorem batchEpilogueE_elab (out : Buf) (oi : Nat → IdxE) (B cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    (batchEpilogueE out oi B).elabAt cta 0 ir im
      = seqNW B (epCopyW out (fun s => (oi s).eval cta 0 ⟨0, by decide⟩ ir im)) := by
  show (seqN B _).elabAt cta 0 ir im = _
  rw [seqN_elab]
  rfl

/-- A copy writes one address of `out`, `%fw0`, and its own accumulator. -/
theorem epCopyW_mem_other (out : Buf) (addr : Nat → Nat) (s j : Nat)
    (h : j ≠ addr s) (st : WSt) :
    ((epCopyW out addr s).run st).mem out j = st.mem out j := by
  show ((WStmt.storeLane0 out (addr s) (BACC + s)).run ((wredW (BACC + s) BTMP).run st)).mem out j = _
  rw [wrun_storeLane0, mem_store1_at _ _ _ _ _ h, wredW_mem]

theorem epCopyW_regs_frame (out : Buf) (addr : Nat → Nat) (s r : Nat)
    (hr : r ≠ BACC + s) (hr' : r ≠ BTMP) (st : WSt) :
    ((epCopyW out addr s).run st).regs r = st.regs r := by
  show ((WStmt.storeLane0 out (addr s) (BACC + s)).run ((wredW (BACC + s) BTMP).run st)).regs r = _
  rw [wrun_storeLane0]
  show ((wredW (BACC + s) BTMP).run st).regs r = _
  exact wredW_frame (BACC + s) BTMP r hr hr' st

theorem epCopyW_at (out : Buf) (addr : Nat → Nat) (s : Nat) (st : WSt) :
    ((epCopyW out addr s).run st).mem out (addr s)
      = bflyFold (st.regs (BACC + s)) ⟨0, by decide⟩ := by
  show ((WStmt.storeLane0 out (addr s) (BACC + s)).run ((wredW (BACC + s) BTMP).run st)).mem
      out (addr s) = _
  rw [wrun_storeLane0, WSt.mem_store1_same,
      wredW_spec (BACC + s) BTMP (Ne.symm (bacc_ne_tmp s)) st]

/-- An accumulator no earlier copy owns survives them all. -/
theorem epSeq_regs_frame (out : Buf) (addr : Nat → Nat) (s : Nat) : ∀ n : Nat, n ≤ s →
    ∀ st : WSt, ((seqNW n (epCopyW out addr)).run st).regs (BACC + s) = st.regs (BACC + s) := by
  intro n hn st
  exact seqNW_regs_frame _ (BACC + s) n
    (fun s' hs' st' => epCopyW_regs_frame out addr s' (BACC + s)
      (bacc_inj (by omega)) (bacc_ne_tmp s) st') st

/-- **Every sample's result lands at its own address.**

    The injectivity hypothesis is the whole content of "these `B` stores do not
    collide"; without it a later sample would overwrite an earlier one and the
    kernel would be silently wrong for all but the last. -/
theorem epSeq_at (out : Buf) (addr : Nat → Nat) : ∀ B s : Nat, s < B →
    (∀ s', s' < B → s' ≠ s → addr s' ≠ addr s) → ∀ st : WSt,
      ((seqNW B (epCopyW out addr)).run st).mem out (addr s)
        = bflyFold (st.regs (BACC + s)) ⟨0, by decide⟩ := by
  intro B
  induction B with
  | zero => intro s hs; exact absurd hs (by omega)
  | succ k ih =>
      intro s hs hinj st
      show ((epCopyW out addr k).run ((seqNW k (epCopyW out addr)).run st)).mem out (addr s) = _
      rcases Nat.lt_or_ge s k with h | h
      · rw [epCopyW_mem_other out addr k (addr s)
              (Ne.symm (hinj k (Nat.lt_succ_self k) (by omega))),
            ih s h (fun s' hs' hne => hinj s' (Nat.lt_succ_of_lt hs') hne) st]
      · have : s = k := by omega
        subst this
        rw [epCopyW_at, epSeq_regs_frame out addr s s (Nat.le_refl s) st]

theorem mem_store1_ne (st : WSt) (b : Buf) (i : Nat) (v : Float32) (c : Buf) (j : Nat)
    (h : ¬ (c = b ∧ j = i)) : (st.store1 b i v).mem c j = st.mem c j := by
  simp [WSt.store1, h]

/-- A copy touches nothing but its own address in `out`. -/
theorem epCopyW_frame (out : Buf) (addr : Nat → Nat) (s : Nat) (c : Buf) (j : Nat)
    (h : ¬ (c = out ∧ j = addr s)) (st : WSt) :
    ((epCopyW out addr s).run st).mem c j = st.mem c j := by
  show ((WStmt.storeLane0 out (addr s) (BACC + s)).run
      ((wredW (BACC + s) BTMP).run st)).mem c j = _
  rw [wrun_storeLane0, mem_store1_ne _ _ _ _ _ _ h, wredW_mem]

/-- **The epilogue writes `B` addresses of `out` and nothing else.**

    The frame half of the batched reduction's contract: a stage has to promise
    not to disturb what it does not own, and what a batched reduction owns is
    exactly its `B` destinations. -/
theorem epSeq_frame (out : Buf) (addr : Nat → Nat) (c : Buf) (j : Nat) : ∀ n : Nat,
    (c ≠ out ∨ ∀ s, s < n → j ≠ addr s) → ∀ st : WSt,
      ((seqNW n (epCopyW out addr)).run st).mem c j = st.mem c j := by
  intro n
  induction n with
  | zero => intro _ _; rfl
  | succ k ih =>
      intro h st
      have h1 : ¬ (c = out ∧ j = addr k) := by
        rintro ⟨rfl, hj⟩
        rcases h with hc | hs
        · exact hc rfl
        · exact hs k (by omega) hj
      have h2 : c ≠ out ∨ ∀ s, s < k → j ≠ addr s := by
        rcases h with hc | hs
        · exact Or.inl hc
        · exact Or.inr (fun s hs' => hs s (by omega))
      show ((epCopyW out addr k).run ((seqNW k (epCopyW out addr)).run st)).mem c j = _
      rw [epCopyW_frame out addr k c j h1, ih h2 st]

-- ---------------------------------------------------------------------------
-- The schema's theorem
-- ---------------------------------------------------------------------------

theorem batchTrips_mem (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (B cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ (L : List Nat) (st : WSt),
      (L.foldl (fun s' j => ((batchTripE bA bB ixA ixB B).elabAt cta j ir im).run s')
        st).mem = st.mem := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons j L ih =>
      intro st
      rw [List.foldl_cons, ih, batchTripE_elab, btripW_mem]

/-- The sweep touches no memory: a batched reduction's only memory effect is
    its `B` stores, which is what a pipeline stage has to promise. -/
theorem batchBody_mem (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (B K cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) :
    (((batchBodyE bA bB ixA ixB B K).elabAt cta 0 ir im).run st).mem = st.mem := by
  show ((List.range K).foldl
      (fun s' j => ((batchTripE bA bB ixA ixB B).elabAt cta j ir im).run s')
      (((seqN B (fun s => EWStmt.setR (BACC + s) (.lit (NumOps.ofNat 0)))).elabAt
          cta 0 ir im).run st)).mem = _
  rw [batchTrips_mem, seqN_elab]
  have hinit : (fun s => EWStmt.elabAt cta 0 ir im
        (EWStmt.setR (BACC + s) (WFExp.lit (NumOps.ofNat 0))))
      = (fun s => WStmt.setR (BACC + s) (WFExp.lit (NumOps.ofNat 0))) := rfl
  rw [hinit, batchInit_mem]

/-- **The batched schema is correct, per sample.**

    Read the right-hand side against `dotStrided_spec`: it is the *same*
    expression — the same committed per-lane fold, the same butterfly tree.
    Batching changed which warp does the work and how often the shared operand
    is fetched; it did not change what is computed, and that is the claim.

    The one hypothesis is that the `B` destinations differ. It is not
    decorative: without it a later sample overwrites an earlier one and the
    kernel is silently wrong for every sample but the last. -/
theorem dotBatched_spec (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (out : Buf)
    (oi : Nat → IdxE) (B K s cta : Nat) (hs : s < B)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt)
    (hinj : ∀ s', s' < B → s' ≠ s →
      (oi s').eval cta 0 ⟨0, by decide⟩ ir im ≠ (oi s).eval cta 0 ⟨0, by decide⟩ ir im) :
    (((dotBatched bA bB ixA ixB out oi B K).elabAt cta 0 ir im).run st).mem out
        ((oi s).eval cta 0 ⟨0, by decide⟩ ir im)
      = bflyFold (dotStridedLane (st.mem bA) (st.mem bB)
          (fun i l => ixA.eval cta i l ir im)
          (fun i l => (ixB s).eval cta i l ir im) K) ⟨0, by decide⟩ := by
  show (((batchEpilogueE out oi B).elabAt cta 0 ir im).run
         (((batchBodyE bA bB ixA ixB B K).elabAt cta 0 ir im).run st)).mem out _ = _
  rw [batchEpilogueE_elab, epSeq_at out _ B s hs hinj,
      batchBody_spec bA bB ixA ixB B K s cta hs ir im st]

/-- **A batched reduction's only memory effect is its `B` stores.** -/
theorem dotBatched_frame (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (out : Buf)
    (oi : Nat → IdxE) (B K cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (c : Buf) (j : Nat)
    (h : c ≠ out ∨ ∀ s, s < B → j ≠ (oi s).eval cta 0 ⟨0, by decide⟩ ir im) :
    (((dotBatched bA bB ixA ixB out oi B K).elabAt cta 0 ir im).run st).mem c j
      = st.mem c j := by
  show (((batchEpilogueE out oi B).elabAt cta 0 ir im).run
         (((batchBodyE bA bB ixA ixB B K).elabAt cta 0 ir im).run st)).mem c j = _
  rw [batchEpilogueE_elab, epSeq_frame out _ c j B h, batchBody_mem]

/-- **Batching is bit-exact against the unbatched kernel.**

    Sample `s` of the batched reduction and a `dotStrided` launched for sample
    `s` alone leave the *same* `Float32` at their destinations — not equal up to
    reassociation, equal.  No law is invoked, because both sides were proven
    against the same committed fold; this is `.trans`, and that it is only a
    `.trans` is the point.  A tuning knob that changed the answer would show up
    as a missing hypothesis here. -/
theorem dotBatched_agrees (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (out out' : Buf)
    (oi : Nat → IdxE) (oi' : IdxE) (B K s cta : Nat) (hs : s < B) (st : WSt)
    (hinj : ∀ s', s' < B → s' ≠ s →
      (oi s').eval cta 0 ⟨0, by decide⟩ ≠ (oi s).eval cta 0 ⟨0, by decide⟩) :
    (((dotBatched bA bB ixA ixB out oi B K).elabIn cta).run st).mem out
        ((oi s).eval cta 0 ⟨0, by decide⟩)
      = (((dotStrided bA bB ixA (ixB s) out' oi' K).elabIn cta).run st).mem out'
          (oi'.eval cta 0 ⟨0, by decide⟩) := by
  rw [show (dotBatched bA bB ixA ixB out oi B K).elabIn cta
        = (dotBatched bA bB ixA ixB out oi B K).elabAt cta 0 (fun _ _ => 0) (fun _ _ => 0)
      from rfl,
      dotBatched_spec bA bB ixA ixB out oi B K s cta hs _ _ st hinj,
      dotStrided_spec bA bB ixA (ixB s) out' oi' K cta st]

-- ---------------------------------------------------------------------------
-- The batched outer product: a gradient accumulated over the batch
-- ---------------------------------------------------------------------------

/-!
  `dW[i][j] = Σₛ adj[s][i]·x[s][j]`.  This is the half a grid split cannot
  express at all: two blocks accumulating into one `dW` element is a race, so
  the sum over samples has to happen inside a warp.

  One warp per row `i = ctaid`, sweeping `j` strided.  Per element the warp
  runs `B` load-load-multiply-add steps and stores once, so the weight matrix
  is **written once per batch** rather than once per sample — and the optimiser
  step that follows runs once per batch too.

  The per-sample step is `stepWS`, the *same* statement `dotStrided`'s sweep is
  built from, so `dotStrided_fold` proves the accumulation with no new
  induction: the sum over samples has the identical committed order to a sum
  over elements.
-/

/-- One sample's contribution: `acc += a_s · b_s`. -/
def outerStepE (bA bB : Buf) (ixA ixB : Nat → IdxE) (s : Nat) : EWStmt :=
  .seq (.seq (.loadIdx 1 bA (ixA s)) (.loadIdx 2 bB (ixB s))) (.setR 0 dotStepSE)

/-- Zero, then accumulate over the whole batch. -/
def outerBodyE (bA bB : Buf) (ixA ixB : Nat → IdxE) (B : Nat) : EWStmt :=
  .seq (.setR 0 (.lit (NumOps.ofNat 0))) (seqN B (outerStepE bA bB ixA ixB))

/-- **The batched outer product**: one warp per row, `K` strided elements, each
    summed over the batch and stored once. -/
def outerBatched (bA bB out : Buf) (ixA ixB : Nat → IdxE) (base : IdxE) (B K : Nat) :
    EWStmt :=
  .forN K (storeBody out 0 (stride32 base) (outerBodyE bA bB ixA ixB B))

/-- Running `n` copies is folding over `0 … n-1` — the bridge that lets the
    existing sweep lemmas apply to a statically unrolled batch. -/
theorem seqNW_run : ∀ (n : Nat) (f : Nat → WStmt) (st : WSt),
    (seqNW n f).run st = (List.range n).foldl (fun s' i => (f i).run s') st := by
  intro n
  induction n with
  | zero => intro _ _; rfl
  | succ k ih =>
      intro f st
      rw [List.range_succ, List.foldl_append, ← ih]
      rfl

theorem outerStepE_elab (bA bB : Buf) (ixA ixB : Nat → IdxE) (s cta j : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    (outerStepE bA bB ixA ixB s).elabAt cta j ir im
      = stepWS bA bB (fun s' l => (ixA s').eval cta j l ir im)
                     (fun s' l => (ixB s').eval cta j l ir im) s := rfl

theorem outerBody_mem (bA bB : Buf) (ixA ixB : Nat → IdxE) (B cta j : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) :
    (((outerBodyE bA bB ixA ixB B).elabAt cta j ir im).run st).mem = st.mem := by
  show (((seqN B (outerStepE bA bB ixA ixB)).elabAt cta j ir im).run
      ((WStmt.setR 0 (.lit (NumOps.ofNat 0))).run st)).mem = _
  rw [seqN_elab, seqNW_mem_frame _ B (fun s _ st' => by rw [outerStepE_elab]; rfl)]
  rfl

/-- **The body sums the sample contributions in the committed order.** -/
theorem outerBody_val (bA bB : Buf) (ixA ixB : Nat → IdxE) (B cta j : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) (l : Lane) :
    (((outerBodyE bA bB ixA ixB B).elabAt cta j ir im).run st).regs 0 l
      = dotStridedLane (st.mem bA) (st.mem bB)
          (fun s l' => (ixA s).eval cta j l' ir im)
          (fun s l' => (ixB s).eval cta j l' ir im) B l := by
  show (((seqN B (outerStepE bA bB ixA ixB)).elabAt cta j ir im).run
      ((WStmt.setR 0 (.lit (NumOps.ofNat 0))).run st)).regs 0 l = _
  rw [seqN_elab]
  have hstep : (fun s => (outerStepE bA bB ixA ixB s).elabAt cta j ir im)
      = stepWS bA bB (fun s' l' => (ixA s').eval cta j l' ir im)
                     (fun s' l' => (ixB s').eval cta j l' ir im) := rfl
  rw [hstep, seqNW_run,
      dotStrided_fold bA bB (fun s' l' => (ixA s').eval cta j l' ir im)
        (fun s' l' => (ixB s').eval cta j l' ir im) (List.range B) _ l]
  simp only [wrun_setR, WSt.regs_setReg_same, WFExp.eval, WSt.mem_setReg, dotStridedLane]

/-- **The batched outer product stores the batch sum at every element it
    visits.**

    The three hypotheses are the three things only the caller knows: the
    destination base is lane-and-loop-free (`by decide` at any concrete base),
    and neither source is the destination.  Injectivity, the kept registers and
    the read-back are `stride32_inj` and `storeLoop_at`, discharged here.

    At `B = 1` the right-hand side is one product — which is exactly what
    `zipPass_spec` gives, so the unbatched kernel is the `B = 1` case of this
    one rather than a different kernel. -/
theorem outerBatched_spec (bA bB out : Buf) (ixA ixB : Nat → IdxE)
    (base : IdxE) (hbase : base.laneLoopFreeB = true) (B K cta : Nat)
    (hAo : bA ≠ out) (hBo : bB ≠ out)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt)
    (j0 : Nat) (l0 : Lane) (hj0 : j0 < K) :
    (((outerBatched bA bB out ixA ixB base B K).elabAt cta 0 ir im).run st).mem out
        ((stride32 base).eval cta j0 l0 ir im)
      = dotStridedLane (st.mem bA) (st.mem bB)
          (fun s l => (ixA s).eval cta j0 l ir im)
          (fun s l => (ixB s).eval cta j0 l ir im) B l0 := by
  refine storeLoop_at out 0 (stride32 base) (outerBodyE bA bB ixA ixB B) cta ir im st
    (fun j l => dotStridedLane (st.mem bA) (st.mem bB)
      (fun s l' => (ixA s).eval cta j l' ir im)
      (fun s l' => (ixB s).eval cta j l' ir im) B l)
    ((stride32 base).eval cta j0 l0 ir im)
    (dotStridedLane (st.mem bA) (st.mem bB)
      (fun s l => (ixA s).eval cta j0 l ir im)
      (fun s l => (ixB s).eval cta j0 l ir im) B l0) []
    (fun j s => outerBody_mem bA bB ixA ixB B cta j ir im s)
    (fun _ _ r' h => absurd h (by simp))
    (fun j s hinv _ l => by
      rw [outerBody_val bA bB ixA ixB B cta j ir im s l]
      exact dotStridedLane_congr _ _ _ _ _ _ B l
        (fun _ _ l' => congrFun (hinv bA hAo) _) (fun _ _ l' => congrFun (hinv bB hBo) _))
    (List.range K) st (fun _ _ => rfl) (fun _ h => absurd h (by simp)) ?_ ?_
  · intro j hj l hl
    obtain ⟨hj', hl'⟩ := stride32_inj base hbase cta j j0 l l0 ir im hl
    rw [hj', hl']
  · exact Or.inl ⟨j0, List.mem_range.mpr hj0, l0, rfl⟩

/-- **The batched outer product writes only the elements its loop visits.** -/
theorem outerBatched_frame (bA bB out : Buf) (ixA ixB : Nat → IdxE) (base : IdxE)
    (B K cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt)
    (c : Buf) (a : Nat)
    (h : c ≠ out ∨ ∀ j ∈ List.range K, ∀ l : Lane, (stride32 base).eval cta j l ir im ≠ a) :
    (((outerBatched bA bB out ixA ixB base B K).elabAt cta 0 ir im).run st).mem c a
      = st.mem c a := by
  have hmem : ∀ (j : Nat) (s : WSt),
      (((outerBodyE bA bB ixA ixB B).elabAt cta j ir im).run s).mem = s.mem :=
    fun j s => outerBody_mem bA bB ixA ixB B cta j ir im s
  rcases h with hc | hno
  · exact congrFun (storeLoop_otherBuf out 0 (stride32 base) (outerBodyE bA bB ixA ixB B)
      cta ir im hmem c hc (List.range K) st) a
  · by_cases hco : c = out
    · subst hco
      exact storeLoop_otherAddr _ 0 (stride32 base) (outerBodyE bA bB ixA ixB B)
        cta ir im hmem a (List.range K) st hno
    · exact congrFun (storeLoop_otherBuf out 0 (stride32 base) (outerBodyE bA bB ixA ixB B)
        cta ir im hmem c hco (List.range K) st) a

/-- **Distinct blocks own distinct addresses**, at the batched reduction's
    addressing `s · grid + ctaid`.

    Two blocks collide only if they agree on both the sample and the block, and
    a block index below the grid cannot be reached from a different sample —
    the samples are `grid` apart and the blocks span less than that. -/
theorem batch_block_inj (grid : Nat) : ∀ (s s' cta cta' : Nat),
    cta < grid → cta' < grid → s * grid + cta = s' * grid + cta' → cta = cta' := by
  intro s s' cta cta' hc hc' he
  rcases Nat.lt_trichotomy s s' with h | h | h
  · exfalso
    have hle : (s + 1) * grid ≤ s' * grid := Nat.mul_le_mul_right grid h
    rw [Nat.succ_mul] at hle
    omega
  · subst h; omega
  · exfalso
    have hle : (s' + 1) * grid ≤ s * grid := Nat.mul_le_mul_right grid h
    rw [Nat.succ_mul] at hle
    omega

/-- **Distinct samples land at distinct addresses**, for any per-sample stride.

    This is `dotBatched_spec`'s one hypothesis at the addressing every batched
    kernel actually emits — `s · stride + ctaid`.  It lives here rather than in
    each model file so that instantiating the schema needs no proof text. -/
theorem batch_addr_inj (m : Nat) (hm : 0 < m) (cta : Nat) :
    ∀ s s' : Nat, s' ≠ s → s' * m + cta ≠ s * m + cta := by
  intro s s' hne he
  exact hne (Nat.eq_of_mul_eq_mul_right hm (by omega))

end AlgorithmLib.ML
