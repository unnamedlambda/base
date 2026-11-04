module
public import AlgorithmLib.ML.Kernel.Library
meta import AlgorithmLib.ML.Kernel.Library
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

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

section Carrier
variable {R : Type} [NumOps R] [NumLit R]

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

section Seq
variable (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)

/-- **A register no copy writes survives the whole sequence.** -/
theorem seqN_regs_frame (f : Nat → EWStmt) (r : Nat) : ∀ n : Nat,
    (∀ s, s < n → ∀ st : WSt R, ((f s).run st cta i ir im).regs r = st.regs r) →
    ∀ st : WSt R, ((seqN n f).run st cta i ir im).regs r = st.regs r := by
  intro n
  induction n with
  | zero => intro _ _; rfl
  | succ k ih =>
      intro h st
      show ((f k).run ((seqN k f).run st cta i ir im) cta i ir im).regs r = _
      rw [h k (Nat.lt_succ_self k), ih (fun s hs => h s (Nat.lt_succ_of_lt hs)) st]

/-- **…and so does memory, if no copy touches it.** -/
theorem seqN_mem_frame (f : Nat → EWStmt) : ∀ n : Nat,
    (∀ s, s < n → ∀ st : WSt R, ((f s).run st cta i ir im).mem = st.mem) →
    ∀ st : WSt R, ((seqN n f).run st cta i ir im).mem = st.mem := by
  intro n
  induction n with
  | zero => intro _ _; rfl
  | succ k ih =>
      intro h st
      show ((f k).run ((seqN k f).run st cta i ir im) cta i ir im).mem = _
      rw [h k (Nat.lt_succ_self k), ih (fun s hs => h s (Nat.lt_succ_of_lt hs)) st]

/-- Running `n` copies is folding over `0 … n-1`. -/
theorem seqN_run : ∀ (n : Nat) (f : Nat → EWStmt) (st : WSt R),
    (seqN n f).run st cta i ir im = (List.range n).foldl (fun s' j => (f j).run s' cta i ir im) st := by
  intro n
  induction n with
  | zero => intro _ _; rfl
  | succ k ih =>
      intro f st
      rw [List.range_succ, List.foldl_append, ← ih]
      rfl

end Seq

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

/-- The five-round sum reduction, on an explicit pair.  `warpRoundE` is this at
    the fixed pair `(0, 1)`; a batched kernel needs one per accumulator. -/
def warpReduceE (acc tmp : Nat) : EWStmt :=
  .seq (bflyRoundOp acc tmp 16 (.add (.reg acc) (.reg tmp)))
    (.seq (bflyRoundOp acc tmp 8 (.add (.reg acc) (.reg tmp)))
      (.seq (bflyRoundOp acc tmp 4 (.add (.reg acc) (.reg tmp)))
        (.seq (bflyRoundOp acc tmp 2 (.add (.reg acc) (.reg tmp)))
              (bflyRoundOp acc tmp 1 (.add (.reg acc) (.reg tmp))))))

/-- One sample's epilogue: reduce its accumulator, store it. -/
def epCopyE (out : Buf) (oi : Nat → IdxE) (s : Nat) : EWStmt :=
  .seq (warpReduceE (BACC + s) BTMP) (.storeLane0 out (oi s) (BACC + s))

/-- Reduce and store each sample's accumulator. -/
def batchEpilogueE (out : Buf) (oi : Nat → IdxE) (B : Nat) : EWStmt :=
  seqN B (epCopyE out oi)

/-- **The batched strided reduction**: one warp, `B` accumulators, `B` stores. -/
def dotBatched (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (out : Buf)
    (oi : Nat → IdxE) (B K : Nat) : EWStmt :=
  .seq (batchBodyE bA bB ixA ixB B K) (batchEpilogueE out oi B)

section Batch
variable (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)

-- ---------------------------------------------------------------------------
-- What one accumulation step does
-- ---------------------------------------------------------------------------

theorem batchStep_mem (bB : Buf) (ixB : Nat → IdxE) (s : Nat) (st : WSt R) :
    ((batchStepE bB ixB s).run st cta i ir im).mem = st.mem := rfl

/-- A step writes `%fw2` and its own accumulator, and nothing else. -/
theorem batchStep_frame (bB : Buf) (ixB : Nat → IdxE) (s r : Nat)
    (h1 : r ≠ BRB) (h2 : r ≠ BACC + s) (st : WSt R) :
    ((batchStepE bB ixB s).run st cta i ir im).regs r = st.regs r := by
  simp only [batchStepE, erun_seq, erun_setR, erun_loadIdx,
    WSt.regs_setReg_other _ (BACC + s) r _ h2, WSt.regs_setReg_other _ BRB r _ h1]

/-- …and what it puts there. -/
theorem batchStep_acc (bB : Buf) (ixB : Nat → IdxE) (s : Nat) (st : WSt R) :
    ((batchStepE bB ixB s).run st cta i ir im).regs (BACC + s)
      = fun l => NumOps.add (st.regs (BACC + s) l)
                   (NumOps.mul (st.regs BRA l) (st.mem bB ((ixB s).eval cta i l ir im))) := by
  simp only [batchStepE, erun_seq, erun_setR, WSt.regs_setReg_same]
  funext l
  simp only [WFExp.eval, erun_loadIdx, WSt.regs_setReg_same,
    WSt.regs_setReg_other _ BRB (BACC + s) _ (bacc_ne_rb s),
    WSt.regs_setReg_other _ BRB BRA _ (by decide : BRA ≠ BRB), WSt.mem_setReg]

-- ---------------------------------------------------------------------------
-- …and what the B copies of it do together
-- ---------------------------------------------------------------------------

theorem batchInner_mem (bB : Buf) (ixB : Nat → IdxE) (B : Nat) (st : WSt R) :
    ((seqN B (batchStepE bB ixB)).run st cta i ir im).mem = st.mem :=
  seqN_mem_frame cta i ir im _ B (fun s _ st' => batchStep_mem cta i ir im bB ixB s st') st

/-- The shared operand survives the whole batch — which is the point: it is
    loaded once and used `B` times. -/
theorem batchInner_ra (bB : Buf) (ixB : Nat → IdxE) (B : Nat) (st : WSt R) :
    ((seqN B (batchStepE bB ixB)).run st cta i ir im).regs BRA = st.regs BRA :=
  seqN_regs_frame cta i ir im _ BRA B
    (fun s _ st' => batchStep_frame cta i ir im bB ixB s BRA (by decide) (Ne.symm (bacc_ne_ra s)) st') st

/-- An accumulator no copy in the first `n` owns is untouched by them. -/
theorem batchInner_other (bB : Buf) (ixB : Nat → IdxE) (s : Nat) : ∀ n : Nat, n ≤ s →
    ∀ st : WSt R, ((seqN n (batchStepE bB ixB)).run st cta i ir im).regs (BACC + s)
      = st.regs (BACC + s) := by
  intro n hn st
  exact seqN_regs_frame cta i ir im _ (BACC + s) n
    (fun s' hs' st' => batchStep_frame cta i ir im bB ixB s' (BACC + s) (bacc_ne_rb s)
      (bacc_inj (by omega)) st') st

/-- **Each of the `B` accumulators takes its own product.** -/
theorem batchInner_acc (bB : Buf) (ixB : Nat → IdxE) : ∀ B s : Nat, s < B →
    ∀ st : WSt R, ((seqN B (batchStepE bB ixB)).run st cta i ir im).regs (BACC + s)
      = fun l => NumOps.add (st.regs (BACC + s) l)
                   (NumOps.mul (st.regs BRA l) (st.mem bB ((ixB s).eval cta i l ir im))) := by
  intro B
  induction B with
  | zero => intro s hs; exact absurd hs (by omega)
  | succ k ih =>
      intro s hs st
      show ((batchStepE bB ixB k).run ((seqN k (batchStepE bB ixB)).run st cta i ir im)
        cta i ir im).regs (BACC + s) = _
      rcases Nat.lt_or_ge s k with h | h
      · rw [batchStep_frame cta i ir im bB ixB k (BACC + s) (bacc_ne_rb s) (bacc_inj (by omega)),
            ih s h st]
      · have hsk : s = k := by omega
        subst hsk
        rw [batchStep_acc cta i ir im bB ixB s, batchInner_other cta i ir im bB ixB s s (Nat.le_refl s) st,
            batchInner_ra cta i ir im bB ixB s st, batchInner_mem cta i ir im bB ixB s st]

-- ---------------------------------------------------------------------------
-- One trip, and the sweep
-- ---------------------------------------------------------------------------

theorem batchTrip_mem (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (B : Nat) (st : WSt R) :
    ((batchTripE bA bB ixA ixB B).run st cta i ir im).mem = st.mem := by
  show ((seqN B (batchStepE bB ixB)).run ((EWStmt.loadIdx BRA bA ixA).run st cta i ir im)
    cta i ir im).mem = _
  rw [batchInner_mem]
  rfl

/-- **A trip adds one product to every accumulator, from the entry memory.** -/
theorem batchTrip_acc (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (B s : Nat) (hs : s < B)
    (st : WSt R) :
    ((batchTripE bA bB ixA ixB B).run st cta i ir im).regs (BACC + s)
      = fun l => NumOps.add (st.regs (BACC + s) l)
                   (NumOps.mul (st.mem bA (ixA.eval cta i l ir im))
                     (st.mem bB ((ixB s).eval cta i l ir im))) := by
  show ((seqN B (batchStepE bB ixB)).run ((EWStmt.loadIdx BRA bA ixA).run st cta i ir im)
    cta i ir im).regs (BACC + s) = _
  rw [batchInner_acc cta i ir im bB ixB B s hs]
  simp only [erun_loadIdx, WSt.regs_setReg_same, WSt.mem_setReg,
    WSt.regs_setReg_other _ BRA (BACC + s) _ (bacc_ne_ra s)]

end Batch

/-- **The sweep is the committed per-lane fold — one per sample.**

    Same shape as `dotStrided_fold`: sequential within a lane, in load order.
    A batched kernel therefore commits to the *same* arithmetic order as the
    unbatched one, which is why the two agree bit-for-bit. -/
theorem batchSweep_fold (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE)
    (B s cta : Nat) (hs : s < B) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ (L : List Nat) (st : WSt R) (l : Lane),
      ((L.foldl (fun s' j => (batchTripE bA bB ixA ixB B).run s' cta j ir im) st).regs (BACC + s) l)
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
      rw [batchTrip_mem, batchTrip_acc cta j ir im bA bB ixA ixB B s hs st]

theorem batchInit_mem (B cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt R) :
    ((seqN B (fun s => EWStmt.setR (BACC + s) (.lit (NumOps.ofNat 0)))).run st cta i ir im).mem
      = st.mem :=
  seqN_mem_frame cta i ir im _ B (fun _ _ _ => rfl) st

theorem batchInit_acc [LitNat R] (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ B s : Nat, s < B → ∀ st : WSt R,
    ((seqN B (fun s => EWStmt.setR (BACC + s) (.lit (NumOps.ofNat 0)))).run st cta i ir im).regs
        (BACC + s)
      = fun _ => NumOps.ofNat 0 := by
  intro B
  induction B with
  | zero => intro s hs; exact absurd hs (by omega)
  | succ k ih =>
      intro s hs st
      show ((EWStmt.setR (BACC + k) _).run ((seqN k _).run st cta i ir im) cta i ir im).regs
        (BACC + s) = _
      rcases Nat.lt_or_ge s k with h | h
      · rw [erun_setR, WSt.regs_setReg_other _ (BACC + k) (BACC + s) _ (bacc_inj (by omega)),
            ih s h st]
      · have : s = k := by omega
        subst this
        rw [erun_setR, WSt.regs_setReg_same]
        funext l
        exact LitNat.ofNat 0

/-- **Each accumulator holds its own sample's committed fold.** -/
theorem batchBody_spec [LitNat R] (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE)
    (B K s cta : Nat) (hs : s < B) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt R) :
    ((batchBodyE bA bB ixA ixB B K).run st cta 0 ir im).regs (BACC + s)
      = dotStridedLane (st.mem bA) (st.mem bB)
          (fun i l => ixA.eval cta i l ir im)
          (fun i l => (ixB s).eval cta i l ir im) K := by
  funext l
  show ((List.range K).foldl (fun s' j => (batchTripE bA bB ixA ixB B).run s' cta j ir im)
      ((seqN B (fun s => EWStmt.setR (BACC + s) (.lit (NumOps.ofNat 0)))).run st cta 0 ir im)).regs
        (BACC + s) l = _
  rw [batchSweep_fold bA bB ixA ixB B s cta hs ir im (List.range K) _ l,
      batchInit_mem, batchInit_acc cta 0 ir im B s hs st]
  rfl

-- ---------------------------------------------------------------------------
-- The epilogue: B butterflies, B stores
-- ---------------------------------------------------------------------------

theorem mem_store1_at (st : WSt R) (b : Buf) (i : Nat) (v : R) (j : Nat)
    (h : j ≠ i) : (st.store1 b i v).mem b j = st.mem b j := by
  simp [WSt.store1, h]

theorem mem_store1_ne (st : WSt R) (b : Buf) (i : Nat) (v : R) (c : Buf) (j : Nat)
    (h : ¬ (c = b ∧ j = i)) : (st.store1 b i v).mem c j = st.mem c j := by
  simp [WSt.store1, h]

section Epilogue
variable (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)

theorem warpReduceE_mem (acc tmp : Nat) (st : WSt R) :
    ((warpReduceE acc tmp).run st cta i ir im).mem = st.mem := rfl

/-- The reduction realises the same butterfly tree the unbatched schema commits
    to — `warpReduceOp_spec` at `add`, on this accumulator. -/
theorem warpReduceE_spec (acc tmp : Nat) (h : tmp ≠ acc) (st : WSt R) :
    ((warpReduceE acc tmp).run st cta i ir im).regs acc = bflyFold (st.regs acc) :=
  warpReduceOp_spec acc tmp h _ NumOps.add (fun _ _ => rfl) st cta i ir im

theorem warpReduceE_frame (acc tmp r : Nat) (hr : r ≠ acc) (hr' : r ≠ tmp) (st : WSt R) :
    ((warpReduceE acc tmp).run st cta i ir im).regs r = st.regs r :=
  warpReduceOp_frame acc tmp _ r hr hr' st cta i ir im

/-- Where sample `s` stores: lane 0's address. -/
abbrev epAddr (oi : Nat → IdxE) (s : Nat) : Nat := (oi s).eval cta i ⟨0, by decide⟩ ir im

/-- A copy writes one address of `out`, `%fw0`, and its own accumulator. -/
theorem epCopy_mem_other (out : Buf) (oi : Nat → IdxE) (s j : Nat)
    (h : j ≠ epAddr cta i ir im oi s) (st : WSt R) :
    ((epCopyE out oi s).run st cta i ir im).mem out j = st.mem out j := by
  show (((warpReduceE (BACC + s) BTMP).run st cta i ir im).store1 out _ _).mem out j = _
  rw [mem_store1_at _ _ _ _ _ h, warpReduceE_mem]

theorem epCopy_regs_frame (out : Buf) (oi : Nat → IdxE) (s r : Nat)
    (hr : r ≠ BACC + s) (hr' : r ≠ BTMP) (st : WSt R) :
    ((epCopyE out oi s).run st cta i ir im).regs r = st.regs r := by
  show (((warpReduceE (BACC + s) BTMP).run st cta i ir im).store1 out _ _).regs r = _
  exact warpReduceE_frame cta i ir im (BACC + s) BTMP r hr hr' st

theorem epCopy_at (out : Buf) (oi : Nat → IdxE) (s : Nat) (st : WSt R) :
    ((epCopyE out oi s).run st cta i ir im).mem out (epAddr cta i ir im oi s)
      = bflyFold (st.regs (BACC + s)) ⟨0, by decide⟩ := by
  show (((warpReduceE (BACC + s) BTMP).run st cta i ir im).store1 out _
    (((warpReduceE (BACC + s) BTMP).run st cta i ir im).regs (BACC + s) ⟨0, by decide⟩)).mem
      out _ = _
  rw [WSt.mem_store1_same, warpReduceE_spec cta i ir im (BACC + s) BTMP (Ne.symm (bacc_ne_tmp s)) st]

/-- An accumulator no earlier copy owns survives them all. -/
theorem epSeq_regs_frame (out : Buf) (oi : Nat → IdxE) (s : Nat) : ∀ n : Nat, n ≤ s →
    ∀ st : WSt R, ((seqN n (epCopyE out oi)).run st cta i ir im).regs (BACC + s)
      = st.regs (BACC + s) := by
  intro n hn st
  exact seqN_regs_frame cta i ir im _ (BACC + s) n
    (fun s' hs' st' => epCopy_regs_frame cta i ir im out oi s' (BACC + s)
      (bacc_inj (by omega)) (bacc_ne_tmp s) st') st

/-- **Every sample's result lands at its own address.**

    The injectivity hypothesis is the whole content of "these `B` stores do not
    collide"; without it a later sample would overwrite an earlier one and the
    kernel would be silently wrong for all but the last. -/
theorem epSeq_at (out : Buf) (oi : Nat → IdxE) : ∀ B s : Nat, s < B →
    (∀ s', s' < B → s' ≠ s → epAddr cta i ir im oi s' ≠ epAddr cta i ir im oi s) → ∀ st : WSt R,
      ((seqN B (epCopyE out oi)).run st cta i ir im).mem out (epAddr cta i ir im oi s)
        = bflyFold (st.regs (BACC + s)) ⟨0, by decide⟩ := by
  intro B
  induction B with
  | zero => intro s hs; exact absurd hs (by omega)
  | succ k ih =>
      intro s hs hinj st
      show ((epCopyE out oi k).run ((seqN k (epCopyE out oi)).run st cta i ir im) cta i ir im).mem
        out _ = _
      rcases Nat.lt_or_ge s k with h | h
      · rw [epCopy_mem_other cta i ir im out oi k _
              (Ne.symm (hinj k (Nat.lt_succ_self k) (by omega))),
            ih s h (fun s' hs' hne => hinj s' (Nat.lt_succ_of_lt hs') hne) st]
      · have : s = k := by omega
        subst this
        rw [epCopy_at, epSeq_regs_frame cta i ir im out oi s s (Nat.le_refl s) st]

/-- A copy touches nothing but its own address in `out`. -/
theorem epCopy_frame (out : Buf) (oi : Nat → IdxE) (s : Nat) (c : Buf) (j : Nat)
    (h : ¬ (c = out ∧ j = epAddr cta i ir im oi s)) (st : WSt R) :
    ((epCopyE out oi s).run st cta i ir im).mem c j = st.mem c j := by
  show (((warpReduceE (BACC + s) BTMP).run st cta i ir im).store1 out _ _).mem c j = _
  rw [mem_store1_ne _ _ _ _ _ _ h, warpReduceE_mem]

/-- **The epilogue writes `B` addresses of `out` and nothing else.**

    The frame half of the batched reduction's contract: a stage has to promise
    not to disturb what it does not own, and what a batched reduction owns is
    exactly its `B` destinations. -/
theorem epSeq_frame (out : Buf) (oi : Nat → IdxE) (c : Buf) (j : Nat) : ∀ n : Nat,
    (c ≠ out ∨ ∀ s, s < n → j ≠ epAddr cta i ir im oi s) → ∀ st : WSt R,
      ((seqN n (epCopyE out oi)).run st cta i ir im).mem c j = st.mem c j := by
  intro n
  induction n with
  | zero => intro _ _; rfl
  | succ k ih =>
      intro h st
      have h1 : ¬ (c = out ∧ j = epAddr cta i ir im oi k) := by
        rintro ⟨rfl, hj⟩
        rcases h with hc | hs
        · exact hc rfl
        · exact hs k (by omega) hj
      have h2 : c ≠ out ∨ ∀ s, s < k → j ≠ epAddr cta i ir im oi s := by
        rcases h with hc | hs
        · exact Or.inl hc
        · exact Or.inr (fun s hs' => hs s (by omega))
      show ((epCopyE out oi k).run ((seqN k (epCopyE out oi)).run st cta i ir im) cta i ir im).mem
        c j = _
      rw [epCopy_frame cta i ir im out oi k c j h1, ih h2 st]

end Epilogue

-- ---------------------------------------------------------------------------
-- The schema's theorem
-- ---------------------------------------------------------------------------

theorem batchTrips_mem (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (B cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ (L : List Nat) (st : WSt R),
      (L.foldl (fun s' j => (batchTripE bA bB ixA ixB B).run s' cta j ir im) st).mem = st.mem := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons j L ih =>
      intro st
      rw [List.foldl_cons, ih, batchTrip_mem]

/-- The sweep touches no memory: a batched reduction's only memory effect is
    its `B` stores, which is what a pipeline stage has to promise. -/
theorem batchBody_mem (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (B K cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt R) :
    ((batchBodyE bA bB ixA ixB B K).run st cta 0 ir im).mem = st.mem := by
  show ((List.range K).foldl (fun s' j => (batchTripE bA bB ixA ixB B).run s' cta j ir im)
      ((seqN B (fun s => EWStmt.setR (BACC + s) (.lit (NumOps.ofNat 0)))).run st cta 0 ir im)).mem
    = _
  rw [batchTrips_mem, batchInit_mem]

/-- **The batched schema is correct, per sample.**

    Read the right-hand side against `dotStrided_spec`: it is the *same*
    expression — the same committed per-lane fold, the same butterfly tree.
    Batching changed which warp does the work and how often the shared operand
    is fetched; it did not change what is computed, and that is the claim.

    The one hypothesis is that the `B` destinations differ. It is not
    decorative: without it a later sample overwrites an earlier one and the
    kernel is silently wrong for every sample but the last. -/
theorem dotBatched_spec [LitNat R] (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (out : Buf)
    (oi : Nat → IdxE) (B K s cta : Nat) (hs : s < B)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt R)
    (hinj : ∀ s', s' < B → s' ≠ s →
      (oi s').eval cta 0 ⟨0, by decide⟩ ir im ≠ (oi s).eval cta 0 ⟨0, by decide⟩ ir im) :
    ((dotBatched bA bB ixA ixB out oi B K).run st cta 0 ir im).mem out
        ((oi s).eval cta 0 ⟨0, by decide⟩ ir im)
      = bflyFold (dotStridedLane (st.mem bA) (st.mem bB)
          (fun i l => ixA.eval cta i l ir im)
          (fun i l => (ixB s).eval cta i l ir im) K) ⟨0, by decide⟩ := by
  show ((seqN B (epCopyE out oi)).run ((batchBodyE bA bB ixA ixB B K).run st cta 0 ir im)
    cta 0 ir im).mem out (epAddr cta 0 ir im oi s) = _
  rw [epSeq_at cta 0 ir im out oi B s hs hinj, batchBody_spec bA bB ixA ixB B K s cta hs ir im st]

/-- **A batched reduction's only memory effect is its `B` stores.** -/
theorem dotBatched_frame (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (out : Buf)
    (oi : Nat → IdxE) (B K cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt R) (c : Buf) (j : Nat)
    (h : c ≠ out ∨ ∀ s, s < B → j ≠ (oi s).eval cta 0 ⟨0, by decide⟩ ir im) :
    ((dotBatched bA bB ixA ixB out oi B K).run st cta 0 ir im).mem c j = st.mem c j := by
  show ((seqN B (epCopyE out oi)).run ((batchBodyE bA bB ixA ixB B K).run st cta 0 ir im)
    cta 0 ir im).mem c j = _
  rw [epSeq_frame cta 0 ir im out oi c j B h, batchBody_mem]

/-- **Batching is bit-exact against the unbatched kernel.**

    Sample `s` of the batched reduction and a `dotStrided` launched for sample
    `s` alone leave the *same* `Float32` at their destinations — not equal up to
    reassociation, equal.  No law is invoked, because both sides were proven
    against the same committed fold; this is `.trans`, and that it is only a
    `.trans` is the point.  A tuning knob that changed the answer would show up
    as a missing hypothesis here. -/
theorem dotBatched_agrees [LitNat R] (bA bB : Buf) (ixA : IdxE) (ixB : Nat → IdxE) (out out' : Buf)
    (oi : Nat → IdxE) (oi' : IdxE) (B K s cta : Nat) (hs : s < B) (st : WSt R)
    (hinj : ∀ s', s' < B → s' ≠ s →
      (oi s').eval cta 0 ⟨0, by decide⟩ ≠ (oi s).eval cta 0 ⟨0, by decide⟩) :
    ((dotBatched bA bB ixA ixB out oi B K).run st cta 0).mem out
        ((oi s).eval cta 0 ⟨0, by decide⟩)
      = ((dotStrided bA bB ixA (ixB s) out' oi' K).run st cta 0).mem out'
          (oi'.eval cta 0 ⟨0, by decide⟩) := by
  rw [dotBatched_spec bA bB ixA ixB out oi B K s cta hs _ _ st hinj,
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
  built from, at sample `s`'s addresses, so the accumulation has the identical
  committed order to a sum over elements.
-/

/-- One sample's contribution: `acc += a_s · b_s`. -/
def outerStepE (bA bB : Buf) (ixA ixB : Nat → IdxE) (s : Nat) : EWStmt :=
  stepWS bA bB (ixA s) (ixB s)

/-- Zero, then accumulate over the whole batch. -/
def outerBodyE (bA bB : Buf) (ixA ixB : Nat → IdxE) (B : Nat) : EWStmt :=
  .seq (.setR 0 (.lit (NumOps.ofNat 0))) (seqN B (outerStepE bA bB ixA ixB))

/-- **The batched outer product**: one warp per row, `K` strided elements, each
    summed over the batch and stored once. -/
def outerBatched (bA bB out : Buf) (ixA ixB : Nat → IdxE) (base : IdxE) (B K : Nat) :
    EWStmt :=
  .forN K (storeBody out 0 (stride32 base) (outerBodyE bA bB ixA ixB B))

/-- The batch's fold, sample by sample at loop counter `j`: what `outerBodyE`
    accumulates into `%fw0`. -/
theorem outer_fold (bA bB : Buf) (ixA ixB : Nat → IdxE) (cta j : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) :
    ∀ (L : List Nat) (st : WSt R) (l : Lane),
      ((L.foldl (fun s' s => (outerStepE bA bB ixA ixB s).run s' cta j ir im) st).regs 0 l)
        = L.foldl (fun acc s => NumOps.add acc (NumOps.mul (st.mem bA ((ixA s).eval cta j l ir im))
            (st.mem bB ((ixB s).eval cta j l ir im)))) (st.regs 0 l) := by
  intro L
  induction L with
  | nil => intro st l; rfl
  | cons s L ih =>
      intro st l
      have hmem : ((outerStepE bA bB ixA ixB s).run st cta j ir im).mem = st.mem := rfl
      have hacc : ((outerStepE bA bB ixA ixB s).run st cta j ir im).regs 0
          = fun l => NumOps.add (st.regs 0 l) (NumOps.mul (st.mem bA ((ixA s).eval cta j l ir im))
              (st.mem bB ((ixB s).eval cta j l ir im))) := by
        simp only [outerStepE, stepWS, erun_seq, erun_setR, WSt.regs_setReg_same]
        funext l'
        simp only [dotStepSE, WFExp.eval, erun_loadIdx, WSt.mem_setReg,
          WSt.regs_setReg_other _ 2 0 _ (by decide),
          WSt.regs_setReg_other _ 1 0 _ (by decide),
          WSt.regs_setReg_other _ 2 1 _ (by decide),
          WSt.regs_setReg_same]
      show ((L.foldl _ ((outerStepE bA bB ixA ixB s).run st cta j ir im)).regs 0 l) = _
      rw [ih _ l, hmem, hacc]
      rfl

theorem outerBody_mem (bA bB : Buf) (ixA ixB : Nat → IdxE) (B cta j : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt R) :
    ((outerBodyE bA bB ixA ixB B).run st cta j ir im).mem = st.mem := by
  show ((seqN B (outerStepE bA bB ixA ixB)).run
      ((EWStmt.setR 0 (.lit (NumOps.ofNat 0))).run st cta j ir im) cta j ir im).mem = _
  rw [seqN_mem_frame cta j ir im _ B (fun s _ st' => rfl)]
  rfl

/-- **The body sums the sample contributions in the committed order.** -/
theorem outerBody_val [LitNat R] (bA bB : Buf) (ixA ixB : Nat → IdxE) (B cta j : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt R) (l : Lane) :
    ((outerBodyE bA bB ixA ixB B).run st cta j ir im).regs 0 l
      = dotStridedLane (st.mem bA) (st.mem bB)
          (fun s l' => (ixA s).eval cta j l' ir im)
          (fun s l' => (ixB s).eval cta j l' ir im) B l := by
  show ((seqN B (outerStepE bA bB ixA ixB)).run
      ((EWStmt.setR 0 (.lit (NumOps.ofNat 0))).run st cta j ir im) cta j ir im).regs 0 l = _
  rw [seqN_run, outer_fold bA bB ixA ixB cta j ir im (List.range B) _ l]
  simp only [erun_setR, WSt.regs_setReg_same, WFExp.eval, WSt.mem_setReg, dotStridedLane, LitNat.ofNat]

/-- **The batched outer product stores the batch sum at every element it
    visits.**

    The three hypotheses are the three things only the caller knows: the
    destination base is lane-and-loop-free (`by decide` at any concrete base),
    and neither source is the destination.  Injectivity, the kept registers and
    the read-back are `stride32_inj` and `storeLoop_at`, discharged here.

    At `B = 1` the right-hand side is one product — which is exactly what
    `zipPass_spec` gives, so the unbatched kernel is the `B = 1` case of this
    one rather than a different kernel. -/
theorem outerBatched_spec [LitNat R] (bA bB out : Buf) (ixA ixB : Nat → IdxE)
    (base : IdxE) (hbase : base.laneLoopFreeB = true) (B K cta : Nat)
    (hAo : bA ≠ out) (hBo : bB ≠ out)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt R)
    (j0 : Nat) (l0 : Lane) (hj0 : j0 < K) :
    ((outerBatched bA bB out ixA ixB base B K).run st cta 0 ir im).mem out
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
    (B K cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt R)
    (c : Buf) (a : Nat)
    (h : c ≠ out ∨ ∀ j ∈ List.range K, ∀ l : Lane, (stride32 base).eval cta j l ir im ≠ a) :
    ((outerBatched bA bB out ixA ixB base B K).run st cta 0 ir im).mem c a = st.mem c a := by
  have hmem : ∀ (j : Nat) (s : WSt R),
      ((outerBodyE bA bB ixA ixB B).run s cta j ir im).mem = s.mem :=
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

end Carrier

end AlgorithmLib.ML
