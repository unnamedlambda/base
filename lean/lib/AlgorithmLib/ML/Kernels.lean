import AlgorithmLib.ML.WarpCompile
import AlgorithmLib.ML.Schema
import AlgorithmLib.ML.PtxPrint

/-!
  # The zero-proof front door

  Everything below this file proves kernels; this file is where a *user* gets
  one **without writing proof text**.  The principle: every obligation the
  schemas leave open — address disjointness, combiner links, kept registers —
  is *decidable*, so the library can discharge it internally and surface only
  genuine failures.  A user whose kernel writes two different values to one
  address gets a build error naming the collision; a user whose kernel is fine
  gets a kernel bundled with its theorems and never sees an obligation at all.

  Two pieces, matching the two ways in:

  * `mapKernel` — the generated route, packaged.  Give it a spec; receive the
    kernel, its PTX, and the theorem that the output buffer holds
    `denote spec` at every lane's address.  Zero obligations: the addressing is
    fixed (`ctaid·32 + laneid`), which is injective by construction.

  * `stride32` / `IdxE.laneLoopFreeB` — the hand-tuned route's address
    obligation, made a `decide`.  Every strided reduction in the stack
    addresses `base + (loop·32 + lane)` with a lane-and-loop-free base;
    `stride32_inj` proves that shape injective **once**, for any base the
    boolean check accepts.  Softmax's `smIx` is literally `stride32 rowBaseIx`
    — its hand-written injectivity argument is subsumed.
-/

namespace AlgorithmLib.ML

-- ---------------------------------------------------------------------------
-- The canonical elementwise addressing
-- ---------------------------------------------------------------------------

/-- `ctaid·32 + laneid` — one element per lane, `⌈n/32⌉` blocks.  The
    addressing every elementwise kernel in the stack uses. -/
def elemIx : IdxE := .add (.mul .ctaId (.lit 32)) .laneId

/-- `elemIx` distinguishes lanes — proven once, used by every map kernel. -/
theorem elemIx_inj (l l0 : Lane) (h : elemIx.eval 0 0 l = elemIx.eval 0 0 l0) :
    l = l0 := by
  have hv : (0 * 32 + l.val : Nat) = 0 * 32 + l0.val := h
  exact Fin.ext (by omega)

-- ---------------------------------------------------------------------------
-- The generated route, packaged
-- ---------------------------------------------------------------------------

/-- A kernel bundled with its meaning.  The user receives both; the proofs are
    fields, not obligations. -/
structure MapKernel (Γ : Nat) where
  spec   : Expr Γ
  inB    : Fin Γ → Buf
  /-- Where each input is read.  The *output* addressing is fixed at `elemIx`
      — that is the one that must be injective — but inputs are free, which is
      what lets a kernel mix per-element reads with broadcast scalars. -/
  inIx   : Fin Γ → IdxE
  out    : Buf
  ew     : EWStmt
  /-- The output buffer, at lane `l`'s address, holds the spec's denotation
      under "variable `i` is what lane `l` sees in buffer `inB i`". -/
  stores : ∀ (st : WSt) (l : Lane),
    ((ew.elabIn 0).run st).mem out (elemIx.eval 0 0 l)
      = denote (fun i => st.mem (inB i) ((inIx i).eval 0 0 l)) spec

/-- **The front door for elementwise kernels, with per-input addressing.**

    Injectivity is an obligation on the *destination* only — two lanes writing
    one address is a race; two lanes reading one address is a broadcast.  So
    `inIx` is unconstrained and the whole theorem is still constructed here.

    This is what a reduction's *epilogue* needs: `dx_j = t_j·r − (x_j·r³/n)·S`
    reads `t` and `x` per element and `r`, `S` at a fixed slot. -/
def mapKernelAt {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE)
    (out : Buf) : MapKernel Γ :=
  { spec, inB, inIx, out
    ew := compileWKernel inB out inIx spec elemIx (Γ + slots spec + 1)
    stores := fun st l =>
      compileWKernel_stores inB out inIx spec elemIx
        (Γ + slots spec + 1) st l
        (fun l' h => by rw [elemIx_inj l' l h]) }

/-- The common case: every input read at the lane's own element. -/
def mapKernel {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf) :
    MapKernel Γ := mapKernelAt spec inB (fun _ => elemIx) out

/-- The PTX, for a kernel binding `nbuf` buffers. -/
def MapKernel.ptx {Γ : Nat} (k : MapKernel Γ) (nbuf : Nat) : String :=
  emitProvenKernelN "main" nbuf 0 k.ew

/-- **…and the emitted text runs it, from raw launch.**  Also constructed, not
    asked for: the side conditions are `decide`d, which is possible because the
    addressing is a closed term. -/
theorem mapKernelAt_ptx_exact {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf)
    (inIx : Fin Γ → IdxE) (hix : ∀ i, (inIx i).IregFree)
    (out : Buf) (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW (mapKernelAt spec inB inIx out).ew)) k (0, m)
          = some ((flatKernel (expandEW (mapKernelAt spec inB inIx out).ew)).length, m')
      ∧ m'.toWSt = ((expandEW (mapKernelAt spec inB inIx out).ew).elabIn cta).run m.toWSt :=
  flatKernel_sound_idxFree cta _ (expandEW_expFree _)
    (expandEW_idxFree _
      (compileWKernel_idxFree inB out inIx spec elemIx _ hix (by decide)))
    (expandEW_flat _ (compileWKernel_flat _ _ _ _ _ _)) m

theorem mapKernel_ptx_exact {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf)
    (out : Buf) (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW (mapKernel spec inB out).ew)) k (0, m)
          = some ((flatKernel (expandEW (mapKernel spec inB out).ew)).length, m')
      ∧ m'.toWSt = ((expandEW (mapKernel spec inB out).ew).elabIn cta).run m.toWSt :=
  mapKernelAt_ptx_exact spec inB (fun _ => elemIx)
    (fun _ => (by decide : IdxE.IregFree elemIx)) out cta m

-- ---------------------------------------------------------------------------
-- The hand-tuned route: the address obligation, made a `decide`
-- ---------------------------------------------------------------------------

/-- Syntactically free of `laneId`, `loopI` **and** `ireg` — so its value is
    the same in every lane and every iteration.  (`ireg` is excluded because
    the integer register file is per-lane; `ldIdx` is fine because `imem` has
    no lane index.) -/
def IdxE.laneLoopFreeB : IdxE → Bool
  | .laneId     => false
  | .loopI      => false
  | .ireg _     => false
  | .ctaId      => true
  | .lit _      => true
  | .ldIdx _ off => off.laneLoopFreeB
  | .add a b    => a.laneLoopFreeB && b.laneLoopFreeB
  | .mul a b    => a.laneLoopFreeB && b.laneLoopFreeB

/-- A lane-and-loop-free term evaluates the same everywhere. -/
theorem IdxE.eval_laneLoopFree (cta : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) :
    ∀ (e : IdxE), e.laneLoopFreeB = true → ∀ (j j' : Nat) (l l' : Lane),
      e.eval cta j l ir im = e.eval cta j' l' ir im := by
  intro e
  induction e with
  | laneId => intro h; exact absurd h (by simp [IdxE.laneLoopFreeB])
  | loopI  => intro h; exact absurd h (by simp [IdxE.laneLoopFreeB])
  | ireg _ => intro h; exact absurd h (by simp [IdxE.laneLoopFreeB])
  | ctaId  => intro _ _ _ _ _; rfl
  | lit _  => intro _ _ _ _ _; rfl
  | ldIdx b off ih =>
      intro h j j' l l'
      show im b (off.eval cta j l ir im) = im b (off.eval cta j' l' ir im)
      rw [ih h j j' l l']
  | add a b iha ihb =>
      intro h j j' l l'
      have hab : a.laneLoopFreeB = true ∧ b.laneLoopFreeB = true := by
        have : (a.laneLoopFreeB && b.laneLoopFreeB) = true := h
        exact ⟨by cases ha : a.laneLoopFreeB <;> simp_all,
               by cases hb : b.laneLoopFreeB <;> simp_all⟩
      show a.eval cta j l ir im + b.eval cta j l ir im = _
      rw [iha hab.1 j j' l l', ihb hab.2 j j' l l']
      rfl
  | mul a b iha ihb =>
      intro h j j' l l'
      have hab : a.laneLoopFreeB = true ∧ b.laneLoopFreeB = true := by
        have : (a.laneLoopFreeB && b.laneLoopFreeB) = true := h
        exact ⟨by cases ha : a.laneLoopFreeB <;> simp_all,
               by cases hb : b.laneLoopFreeB <;> simp_all⟩
      show a.eval cta j l ir im * b.eval cta j l ir im = _
      rw [iha hab.1 j j' l l', ihb hab.2 j j' l l']
      rfl

/-- The canonical strided address: element `loop·32 + lane` of a row starting
    at `base`.  Softmax's `smIx` is this at `base := rowBaseIx`. -/
def stride32 (base : IdxE) : IdxE :=
  .add base (.add (.mul .loopI (.lit 32)) .laneId)

/-- **The strided address is injective — for any admissible base, proven once.**

    This is the theorem behind every "distinct `(iteration, lane)` pairs hit
    distinct addresses" obligation in the stack.  The hypothesis is a boolean
    on the base's syntax, so at any concrete base it is `by decide` — which is
    what lets a smart constructor discharge it without the user seeing it. -/
theorem stride32_inj (base : IdxE) (h : base.laneLoopFreeB = true)
    (cta j j0 : Nat) (l l0 : Lane) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (he : (stride32 base).eval cta j l ir im = (stride32 base).eval cta j0 l0 ir im) :
    j = j0 ∧ l = l0 := by
  have hb := IdxE.eval_laneLoopFree cta ir im base h j j0 l l0
  have hv : (base.eval cta j l ir im + (j * 32 + l.val) : Nat)
      = base.eval cta j0 l0 ir im + (j0 * 32 + l0.val) := he
  rw [hb] at hv
  have h1 : l.val < 32 := l.isLt
  have h2 : l0.val < 32 := l0.isLt
  refine ⟨by omega, Fin.ext (by omega)⟩

-- ---------------------------------------------------------------------------
-- Packaged reductions: the dynamic-length shapes, with nothing left to prove
-- ---------------------------------------------------------------------------

/-!
  `chunkRemReduce_spec` takes ten arguments, of which the user genuinely
  chooses four (buffer, addresses, bound slots) and the rest are `rfl`,
  `decide`, or a library lemma.  These two wrappers supply the rest.

  Between them they cover every reduction in the stack: softmax's row maximum
  is `maxReduce`, its `Σexp` is the general form with `keep = [5]`, and
  argmax's two passes are `maxReduce` twice.
-/

/-- **A dynamic-length maximum**: chunk sweep, max butterfly, remainder fold.
    `%fw0` accumulates, `%fw2` takes the load, `%fw1` is the shuffle temp. -/
def maxReduceEW (b : Buf) (ixC ixR : IdxE) (bu adC adR : Nat) (init : Float32) :
    EWStmt :=
  chunkRemReduce b 2 0 (.maxW (.reg 0) (.reg 2)) ixC ixR (.lit init)
    bu adC adR warpReduceMaxE

/-- …and what it computes, with **no obligation on the caller**. -/
theorem maxReduce_spec (b : Buf) (ixC ixR : IdxE) (bu adC adR cta i : Nat)
    (init : Float32) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt)
    (l : Lane) :
    (((maxReduceEW b ixC ixR bu adC adR init).elabAt cta i ir im).run st).regs 0 l
      = chunkRemFold (st.mem b) (fun _ a x => NumOps.max a x) (fun _ => init)
          (fun j l' => ixC.eval cta j l' ir im) (fun j l' => ixR.eval cta j l' ir im)
          (im bu adC) (im bu adR) (bflyFoldOp (fun a c => NumOps.max a c)) l :=
  chunkRemReduce_spec b 2 0 (by decide) (.maxW (.reg 0) (.reg 2)) ixC ixR (.lit init)
    bu adC adR cta i warpReduceMaxE ir im
    (fun _ a x => NumOps.max a x) (bflyFoldOp (fun a c => NumOps.max a c))
    [] (fun _ h => absurd h (by simp)) st (fun _ => init) (fun _ => rfl)
    (fun _ _ _ => rfl)
    (fun st' => warpReduceMaxE_spec cta i ir im st')
    (fun st' => warpReduceMaxE_mem cta i ir im st')
    (fun _ r h => absurd h (by simp)) l

/-- **A dynamic-length sum** of the loaded values, same shape at `add`. -/
def sumReduceEW (b : Buf) (ixC ixR : IdxE) (bu adC adR : Nat) : EWStmt :=
  chunkRemReduce b 2 0 (.add (.reg 0) (.reg 2)) ixC ixR (.lit (NumOps.ofNat 0))
    bu adC adR
    (.seq (warpRoundE 16) (.seq (warpRoundE 8) (.seq (warpRoundE 4)
      (.seq (warpRoundE 2) (warpRoundE 1)))))

theorem sumReduce_spec (b : Buf) (ixC ixR : IdxE) (bu adC adR cta i : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) (l : Lane) :
    (((sumReduceEW b ixC ixR bu adC adR).elabAt cta i ir im).run st).regs 0 l
      = chunkRemFold (st.mem b) (fun _ a x => NumOps.add a x)
          (fun _ => NumOps.ofNat 0)
          (fun j l' => ixC.eval cta j l' ir im) (fun j l' => ixR.eval cta j l' ir im)
          (im bu adC) (im bu adR) (bflyFoldOp (fun a c => NumOps.add a c)) l :=
  chunkRemReduce_spec b 2 0 (by decide) (.add (.reg 0) (.reg 2)) ixC ixR
    (.lit (NumOps.ofNat 0)) bu adC adR cta i _ ir im
    (fun _ a x => NumOps.add a x) (bflyFoldOp (fun a c => NumOps.add a c))
    [] (fun _ h => absurd h (by simp)) st (fun _ => NumOps.ofNat 0) (fun _ => rfl)
    (fun _ _ _ => rfl)
    (fun st' => warpReduceSumE_spec cta i ir im st')
    (fun st' => warpReduceSumE_mem cta i ir im st')
    (fun _ r h => absurd h (by simp)) l

-- ---------------------------------------------------------------------------
-- The packaged store pass
-- ---------------------------------------------------------------------------

/-- **A strided store pass**: for each `(loop, lane)`, load one element, apply
    `f`, store the result at the same index.  The address obligation is
    `stride32_inj`, so the caller supplies only the syntax check. -/
def storePassEW (inB out : Buf) (d r : Nat) (f : WFExp) (base : IdxE) (K : Nat) :
    EWStmt :=
  .forN K (storeBody out r (stride32 base)
    (.seq (.loadIdx d inB (stride32 base)) (.setR r f)))

/-- **What a store pass leaves in memory**, at any `(loop, lane)` it visits.

    The two hypotheses left are exactly the two things only the caller knows:
    that the base address is lane-and-loop-free (`by decide` at any concrete
    base) and that the pass does not read the buffer it writes.  Everything
    else — injectivity, the `keep` bookkeeping, the read-back — is discharged
    here. -/
theorem storePass_spec (inB out : Buf) (d r : Nat) (f : WFExp)
    (base : IdxE) (hbase : base.laneLoopFreeB = true) (K cta : Nat)
    (hio : inB ≠ out) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt)
    (g : Float32 → Float32)
    (hf : ∀ (st' : WSt) (l : Lane), f.eval st' l = g (st'.regs d l))
    (j0 : Nat) (l0 : Lane) (hj0 : j0 < K) :
    (((storePassEW inB out d r f base K).elabAt cta 0 ir im).run st).mem out
        ((stride32 base).eval cta j0 l0 ir im)
      = g (st.mem inB ((stride32 base).eval cta j0 l0 ir im)) := by
  refine storeLoop_at out r (stride32 base)
    (.seq (.loadIdx d inB (stride32 base)) (.setR r f)) cta ir im st
    (fun j l => g (st.mem inB ((stride32 base).eval cta j l ir im)))
    ((stride32 base).eval cta j0 l0 ir im)
    (g (st.mem inB ((stride32 base).eval cta j0 l0 ir im))) []
    (fun _ _ => rfl) (fun _ _ r' h => absurd h (by simp))
    (fun j s hinv _ l => by
      show ((WStmt.setR r f).run
              ((WStmt.loadIdx d inB (fun l' => (stride32 base).eval cta j l' ir im)).run s)).regs r l
            = _
      rw [wrun_setR, WSt.regs_setReg_same, hf, wrun_loadIdx,
          WSt.regs_setReg_same, hinv inB hio])
    (List.range K) st (fun _ _ => rfl) (fun _ h => absurd h (by simp)) ?_ ?_
  · intro j hj l hl
    obtain ⟨hj', hl'⟩ := stride32_inj base hbase cta j j0 l l0 ir im hl
    rw [hj', hl']
  · exact Or.inl ⟨j0, List.mem_range.mpr hj0, l0, rfl⟩

-- ---------------------------------------------------------------------------
-- More than one element per lane
-- ---------------------------------------------------------------------------

/-!
  `mapKernel` gives every lane exactly one element, so a 67M-element kernel
  launches 2,097,152 blocks of 32 threads and runs at **39% of roofline** — the
  cost is block scheduling and the absence of per-thread instruction-level
  parallelism, not bandwidth.  With `E` elements per lane it reaches **67%**.

  What makes the loop possible is `compileWKernel_correctAt`: the compiled body
  is correct at *any* block and iteration, not just block 0, so a compiled
  expression can sit inside a loop.

  The addressing is `stride32 (ctaid·32·E)`, so consecutive lanes still read
  consecutive addresses — coalescing is preserved and the injectivity obligation
  is `stride32_inj` again.
-/

/-- **`E` elements per lane.**  Same spec, same compiler, `E`× fewer blocks. -/
def mapLoopEW {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf)
    (base : IdxE) (E : Nat) : EWStmt :=
  .forN E (storeBody out (Γ + slots spec + 1) (stride32 base)
    (.seq (loadSeq inB (fun _ => stride32 base))
          (.seq (compileW (fun i => .reg i.val) Γ spec).1
                (.setR (Γ + slots spec + 1)
                  (compileW (fun i => .reg i.val) Γ spec).2))))

/-- **…and it stores the spec at every element it visits.**

    The caller supplies only that the base is lane-and-loop-free (`by decide`)
    and that the kernel does not read the buffer it writes.  Everything else —
    injectivity, the `keep` bookkeeping, the read-back, and the fact that the
    compiled body is valid at iteration `j` — is discharged here. -/
theorem mapLoopEW_stores {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf)
    (base : IdxE) (hbase : base.laneLoopFreeB = true) (E cta : Nat)
    (hio : ∀ i, inB i ≠ out) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (j0 : Nat) (l0 : Lane) (hj0 : j0 < E) :
    (((mapLoopEW spec inB out base E).elabAt cta 0 ir im).run st).mem out
        ((stride32 base).eval cta j0 l0 ir im)
      = denote (fun i => st.mem (inB i) ((stride32 base).eval cta j0 l0 ir im)) spec := by
  refine storeLoop_at out (Γ + slots spec + 1) (stride32 base)
    (.seq (loadSeq inB (fun _ => stride32 base))
          (.seq (compileW (fun i => .reg i.val) Γ spec).1
                (.setR (Γ + slots spec + 1)
                  (compileW (fun i => .reg i.val) Γ spec).2)))
    cta ir im st
    (fun j l => denote (fun i => st.mem (inB i) ((stride32 base).eval cta j l ir im)) spec)
    ((stride32 base).eval cta j0 l0 ir im)
    (denote (fun i => st.mem (inB i) ((stride32 base).eval cta j0 l0 ir im)) spec) []
    (fun j s => by
      show ((WStmt.setR (Γ + slots spec + 1) (compileW (fun i => .reg i.val) Γ spec).2).run
              (((compileW (fun i => .reg i.val) Γ spec).1.elabAt cta j ir im).run
                (((loadSeq inB (fun _ => stride32 base)).elabAt cta j ir im).run s))).mem = _
      rw [wrun_setR, WSt.mem_setReg, compileW_elabAt]
      show (runW (compileW (fun i => .reg i.val) Γ spec).1
              (((loadSeq inB (fun _ => stride32 base)).elabAt cta j ir im).run s)).mem = _
      rw [compileW_mem]
      exact loadSeqN_memAt inB (fun _ => stride32 base) cta j ir im Γ s)
    (fun _ _ r' h => absurd h (by simp))
    (fun j s hinv _ l => by
      show ((WStmt.setR (Γ + slots spec + 1) (compileW (fun i => .reg i.val) Γ spec).2).run
              (((compileW (fun i => .reg i.val) Γ spec).1.elabAt cta j ir im).run
                (((loadSeq inB (fun _ => stride32 base)).elabAt cta j ir im).run s))).regs
                (Γ + slots spec + 1) l = _
      rw [wrun_setR, WSt.regs_setReg_same,
          compileWKernel_correctAt inB (fun _ => stride32 base) spec s l cta j ir im]
      congr 1
      funext i
      exact congrFun (hinv (inB i) (hio i)) _)
    (List.range E) st (fun _ _ => rfl) (fun _ h => absurd h (by simp)) ?_ ?_
  · intro j hj l hl
    obtain ⟨hj', hl'⟩ := stride32_inj base hbase cta j j0 l l0 ir im hl
    rw [hj', hl']
  · exact Or.inl ⟨j0, List.mem_range.mpr hj0, l0, rfl⟩

-- ---------------------------------------------------------------------------
-- The reduction front door
-- ---------------------------------------------------------------------------

/-!
  `mapKernel` packages the elementwise route.  This packages the *reduction*
  route: one warp per output, a strided sweep, a butterfly, a lane-0 store —
  with the spec as a field rather than an obligation.

  The spec is `dotStridedE`, the **committed** order: sequential within a lane,
  five-round butterfly across lanes.  That is deliberate and it is the honest
  choice — it makes the theorem an exact `Float32` equality with no hypothesis.

  A user who instead writes `Vec.dot` (an `Expr.sum`, i.e. a flat left fold over
  all `n`) and expects *this* kernel is asking for reassociation, which is
  `Law.sumAssoc` and is false at `Float32` (measured: 100000024 vs 100000000).
  The two are kept apart on purpose; `Rewrite.lean` is where that gap is named.
-/

/-- A reduction kernel bundled with its meaning. -/
structure ReduceKernel where
  bA  : Buf
  bB  : Buf
  ixA : IdxE
  ixB : IdxE
  out : Buf
  oi  : IdxE
  K   : Nat
  ew  : EWStmt
  /-- Lane 0 of block `cta` stores the two-level fold of `bA·bB`. -/
  reduces : ∀ (cta : Nat) (st : WSt),
    ((ew.elabIn cta).run st).mem out (oi.eval cta 0 ⟨0, by decide⟩)
      = bflyFold (dotStridedLane (st.mem bA) (st.mem bB)
          (fun i l => ixA.eval cta i l) (fun i l => ixB.eval cta i l) K)
          ⟨0, by decide⟩

/-- **The front door for strided reductions.**  Buffers, two address
    expressions, a destination and a trip count in; kernel and theorem out.

    Addressing is unconstrained on *both* operands, which is what lets the same
    schema serve a forward row walk and a backward column walk. -/
def dotKernel (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (oi : IdxE) (K : Nat) :
    ReduceKernel :=
  { bA, bB, ixA, ixB, out, oi, K
    ew := dotStrided bA bB ixA ixB out oi K
    reduces := fun cta st => dotStrided_spec bA bB ixA ixB out oi K cta st }

/-- Sum of squares — the same schema with one buffer, no new proof. -/
def sumSqKernel (b : Buf) (ix : IdxE) (out : Buf) (oi : IdxE) (K : Nat) :
    ReduceKernel := dotKernel b b ix ix out oi K

def ReduceKernel.ptx (k : ReduceKernel) (nbuf : Nat) : String :=
  emitProvenKernelN "main" nbuf 0 k.ew

-- ---------------------------------------------------------------------------
-- The two-buffer store pass
-- ---------------------------------------------------------------------------

/-!
  `storePassEW` reads one buffer.  A backward pass needs two: the weight
  gradient of a dense layer is `dW[i][j] = adj[i] · x[j]`, an **outer product**,
  which reads the adjoint and the activation at *different* addresses and
  writes a third buffer.

  The addressing that makes it loop rather than unroll is one warp per row:
  `i = ctaid`, `j = loop·32 + lane`.  Then `adj[i]` is lane-uniform, `x[j]` is
  the strided walk, and the destination is `stride32 (ctaid·n)` — so the
  injectivity obligation is `stride32_inj` again, discharged here.

  It is not only for gradients: any elementwise binary kernel with independent
  addressing (a gate product, a residual add across differently-laid-out
  buffers) is this schema.
-/

/-- **A two-buffer strided store pass**: for each `(loop, lane)`, load one
    element from each of two buffers at independent addresses, combine, store. -/
def zipPassEW (bA bB out : Buf) (dA dB r : Nat) (f : WFExp) (ixA ixB oix : IdxE)
    (K : Nat) : EWStmt :=
  .forN K (storeBody out r oix
    (.seq (.loadIdx dA bA ixA) (.seq (.loadIdx dB bB ixB) (.setR r f))))

/-- **What a two-buffer pass leaves in memory.**

    The hypotheses left to the caller are the same three only the caller knows:
    the destination base is lane-and-loop-free (`by decide` at any concrete
    base), and neither source is the destination.  Register distinctness,
    address injectivity, the `keep` bookkeeping and the read-back are discharged
    here. -/
theorem zipPass_spec (bA bB out : Buf) (dA dB r : Nat) (hAB : dA ≠ dB) (f : WFExp)
    (base : IdxE) (hbase : base.laneLoopFreeB = true) (ixA ixB : IdxE)
    (K cta : Nat) (hAo : bA ≠ out) (hBo : bB ≠ out)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt)
    (g : Float32 → Float32 → Float32)
    (hf : ∀ (st' : WSt) (l : Lane), f.eval st' l = g (st'.regs dA l) (st'.regs dB l))
    (j0 : Nat) (l0 : Lane) (hj0 : j0 < K) :
    (((zipPassEW bA bB out dA dB r f ixA ixB (stride32 base) K).elabAt cta 0 ir im).run st).mem
        out ((stride32 base).eval cta j0 l0 ir im)
      = g (st.mem bA (ixA.eval cta j0 l0 ir im))
          (st.mem bB (ixB.eval cta j0 l0 ir im)) := by
  refine storeLoop_at out r (stride32 base)
    (.seq (.loadIdx dA bA ixA) (.seq (.loadIdx dB bB ixB) (.setR r f)))
    cta ir im st
    (fun j l => g (st.mem bA (ixA.eval cta j l ir im))
                  (st.mem bB (ixB.eval cta j l ir im)))
    ((stride32 base).eval cta j0 l0 ir im)
    (g (st.mem bA (ixA.eval cta j0 l0 ir im))
       (st.mem bB (ixB.eval cta j0 l0 ir im))) []
    (fun _ _ => rfl) (fun _ _ r' h => absurd h (by simp))
    (fun j s hinv _ l => by
      show ((WStmt.setR r f).run
              ((WStmt.loadIdx dB bB (fun l' => ixB.eval cta j l' ir im)).run
                ((WStmt.loadIdx dA bA (fun l' => ixA.eval cta j l' ir im)).run s))).regs r l
            = _
      rw [wrun_setR, WSt.regs_setReg_same, hf, wrun_loadIdx, WSt.regs_setReg_same,
          WSt.regs_setReg_other _ dB dA _ hAB, wrun_loadIdx,
          WSt.regs_setReg_same, WSt.mem_setReg, hinv bA hAo, hinv bB hBo])
    (List.range K) st (fun _ _ => rfl) (fun _ h => absurd h (by simp)) ?_ ?_
  · intro j hj l hl
    obtain ⟨hj', hl'⟩ := stride32_inj base hbase cta j j0 l l0 ir im hl
    rw [hj', hl']
  · exact Or.inl ⟨j0, List.mem_range.mpr hj0, l0, rfl⟩

/-- **A three-buffer strided store pass.**  Two operands and a third value —
    a cotangent, a statistic, a gate — each at its own address.

    This is the arity a nonlinear adjoint needs (the derivative of a product
    against an incoming gradient) and the one a fused elementwise step reaches
    as soon as two passes collapse into one. -/
def zip3PassEW (bA bB bC out : Buf) (dA dB dC r : Nat) (f : WFExp)
    (ixA ixB ixC oix : IdxE) (K : Nat) : EWStmt :=
  .forN K (storeBody out r oix
    (.seq (.loadIdx dA bA ixA)
      (.seq (.loadIdx dB bB ixB) (.seq (.loadIdx dC bC ixC) (.setR r f)))))

/-- **What a three-buffer pass leaves in memory** — the two-buffer statement
    with one more operand, and the same three caller obligations. -/
theorem zip3Pass_spec (bA bB bC out : Buf) (dA dB dC r : Nat)
    (hAB : dA ≠ dB) (hAC : dA ≠ dC) (hBC : dB ≠ dC) (f : WFExp)
    (base : IdxE) (hbase : base.laneLoopFreeB = true) (ixA ixB ixC : IdxE)
    (K cta : Nat) (hAo : bA ≠ out) (hBo : bB ≠ out) (hCo : bC ≠ out)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt)
    (g : Float32 → Float32 → Float32 → Float32)
    (hf : ∀ (st' : WSt) (l : Lane),
      f.eval st' l = g (st'.regs dA l) (st'.regs dB l) (st'.regs dC l))
    (j0 : Nat) (l0 : Lane) (hj0 : j0 < K) :
    (((zip3PassEW bA bB bC out dA dB dC r f ixA ixB ixC (stride32 base) K).elabAt
        cta 0 ir im).run st).mem out ((stride32 base).eval cta j0 l0 ir im)
      = g (st.mem bA (ixA.eval cta j0 l0 ir im))
          (st.mem bB (ixB.eval cta j0 l0 ir im))
          (st.mem bC (ixC.eval cta j0 l0 ir im)) := by
  refine storeLoop_at out r (stride32 base)
    (.seq (.loadIdx dA bA ixA)
      (.seq (.loadIdx dB bB ixB) (.seq (.loadIdx dC bC ixC) (.setR r f))))
    cta ir im st
    (fun j l => g (st.mem bA (ixA.eval cta j l ir im))
                  (st.mem bB (ixB.eval cta j l ir im))
                  (st.mem bC (ixC.eval cta j l ir im)))
    ((stride32 base).eval cta j0 l0 ir im)
    (g (st.mem bA (ixA.eval cta j0 l0 ir im))
       (st.mem bB (ixB.eval cta j0 l0 ir im))
       (st.mem bC (ixC.eval cta j0 l0 ir im))) []
    (fun _ _ => rfl) (fun _ _ r' h => absurd h (by simp))
    (fun j s hinv _ l => by
      show WSt.regs ((WStmt.setR r f).run
              ((WStmt.loadIdx dC bC (fun l' => ixC.eval cta j l' ir im)).run
                ((WStmt.loadIdx dB bB (fun l' => ixB.eval cta j l' ir im)).run
                  ((WStmt.loadIdx dA bA (fun l' => ixA.eval cta j l' ir im)).run s))))
            r l = _
      rw [wrun_setR, WSt.regs_setReg_same, hf, wrun_loadIdx, WSt.regs_setReg_same,
          WSt.regs_setReg_other _ dC dB _ hBC,
          WSt.regs_setReg_other _ dC dA _ hAC, wrun_loadIdx,
          WSt.regs_setReg_same, WSt.regs_setReg_other _ dB dA _ hAB,
          wrun_loadIdx, WSt.regs_setReg_same]
      simp only [WSt.mem_setReg, hinv bA hAo, hinv bB hBo, hinv bC hCo])
    (List.range K) st (fun _ _ => rfl) (fun _ h => absurd h (by simp)) ?_ ?_
  · intro j hj l hl
    obtain ⟨hj', hl'⟩ := stride32_inj base hbase cta j j0 l l0 ir im hl
    rw [hj', hl']
  · exact Or.inl ⟨j0, List.mem_range.mpr hj0, l0, rfl⟩

end AlgorithmLib.ML
