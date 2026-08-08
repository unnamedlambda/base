import AlgorithmLib.ML.PtxM

namespace AlgorithmLib.ML

/-! Register allocation, fixed by the schema: `%fw0` accumulates, `%fw1..4` hold
    the `A` quad, `%fw5..8` the `B` quad.  `%fw1` doubles as the shuffle
    temporary, which is safe because the quads are dead by the reduction. -/

/-- One accumulation step: `acc += Σⱼ aⱼ·bⱼ`, in load order. -/
def dotStepE : WFExp :=
  .add (.add (.add (.add (.reg 0) (.mul (.reg 1) (.reg 5)))
                   (.mul (.reg 2) (.reg 6)))
             (.mul (.reg 3) (.reg 7)))
       (.mul (.reg 4) (.reg 8))

/-- The strided vectorised sweep. -/
def warpDotBody (bA bB : Buf) (ixA ixB : IdxE) (K : Nat) : EWStmt :=
  .seq (.setR 0 (.lit (NumOps.ofNat 0)))
       (.forN K (.seq (.seq (.loadV4 1 2 3 4 bA ixA) (.loadV4 5 6 7 8 bB ixB))
                      (.setR 0 dotStepE)))

/-- One butterfly round, in the emittable language. -/
def warpRoundE (m : Nat) : EWStmt :=
  .seq (.shflXor 1 0 m) (.setR 0 (.add (.reg 0) (.reg 1)))

/-- **The schema**: sweep, reduce, store. -/
def warpDotV4 (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (oi : IdxE) (K : Nat) : EWStmt :=
  .seq (.seq (warpDotBody bA bB ixA ixB K)
             (.seq (warpRoundE 16) (.seq (warpRoundE 8) (.seq (warpRoundE 4)
               (.seq (warpRoundE 2) (warpRoundE 1))))))
       (.storeLane0 out oi 0)

/-- The fold one lane performs: sequential, four elements per step, in load
    order.  Committing to this order is what makes the theorem an exact
    equality rather than a bound. -/
def dotLane (memA memB : Nat → Float32) (fA fB : Nat → Lane → Nat) (K : Nat)
    (l : Lane) : Float32 :=
  (List.range K).foldl
    (fun acc i => NumOps.add (NumOps.add (NumOps.add
        (NumOps.add acc (NumOps.mul (memA (fA i l)) (memB (fB i l))))
        (NumOps.mul (memA (fA i l + 1)) (memB (fB i l + 1))))
        (NumOps.mul (memA (fA i l + 2)) (memB (fB i l + 2))))
        (NumOps.mul (memA (fA i l + 3)) (memB (fB i l + 3))))
    (NumOps.ofNat 0)

-- ---------------------------------------------------------------------------
-- The sweep
-- ---------------------------------------------------------------------------

private def stepW (bA bB : Buf) (fA fB : Nat → Lane → Nat) (i : Nat) : WStmt :=
  .seq (.seq (.loadV4 1 2 3 4 bA (fA i)) (.loadV4 5 6 7 8 bB (fB i)))
       (.setR 0 dotStepE)

theorem warpDot_fold (bA bB : Buf) (fA fB : Nat → Lane → Nat) :
    ∀ (L : List Nat) (st : WSt) (l : Lane),
      ((L.foldl (fun s' i => (stepW bA bB fA fB i).run s') st).regs 0 l)
        = L.foldl (fun acc i => NumOps.add (NumOps.add (NumOps.add
            (NumOps.add acc (NumOps.mul (st.mem bA (fA i l)) (st.mem bB (fB i l))))
            (NumOps.mul (st.mem bA (fA i l + 1)) (st.mem bB (fB i l + 1))))
            (NumOps.mul (st.mem bA (fA i l + 2)) (st.mem bB (fB i l + 2))))
            (NumOps.mul (st.mem bA (fA i l + 3)) (st.mem bB (fB i l + 3))))
          (st.regs 0 l) := by
  intro L
  induction L with
  | nil => intro st l; rfl
  | cons i L ih =>
      intro st l
      have hmem : ((stepW bA bB fA fB i).run st).mem = st.mem := rfl
      have hacc : ((stepW bA bB fA fB i).run st).regs 0
          = fun l => NumOps.add (NumOps.add (NumOps.add
              (NumOps.add (st.regs 0 l) (NumOps.mul (st.mem bA (fA i l)) (st.mem bB (fB i l))))
              (NumOps.mul (st.mem bA (fA i l + 1)) (st.mem bB (fB i l + 1))))
              (NumOps.mul (st.mem bA (fA i l + 2)) (st.mem bB (fB i l + 2))))
              (NumOps.mul (st.mem bA (fA i l + 3)) (st.mem bB (fB i l + 3))) := by
        show ((WStmt.setR 0 dotStepE).run
                ((WStmt.seq (.loadV4 1 2 3 4 bA (fA i)) (.loadV4 5 6 7 8 bB (fB i))).run st)).regs 0
              = _
        simp only [wrun_setR, WSt.regs_setReg_same]
        funext l
        simp only [dotStepE, WFExp.eval, wrun_seq, wrun_loadV4, WSt.mem_setReg,
          WSt.regs_setReg_other _ 8 0 _ (by decide), WSt.regs_setReg_other _ 7 0 _ (by decide),
          WSt.regs_setReg_other _ 6 0 _ (by decide), WSt.regs_setReg_other _ 5 0 _ (by decide),
          WSt.regs_setReg_other _ 4 0 _ (by decide), WSt.regs_setReg_other _ 3 0 _ (by decide),
          WSt.regs_setReg_other _ 2 0 _ (by decide), WSt.regs_setReg_other _ 1 0 _ (by decide),
          WSt.regs_setReg_other _ 8 1 _ (by decide), WSt.regs_setReg_other _ 7 1 _ (by decide),
          WSt.regs_setReg_other _ 6 1 _ (by decide), WSt.regs_setReg_other _ 5 1 _ (by decide),
          WSt.regs_setReg_other _ 4 1 _ (by decide), WSt.regs_setReg_other _ 3 1 _ (by decide),
          WSt.regs_setReg_other _ 2 1 _ (by decide),
          WSt.regs_setReg_other _ 8 2 _ (by decide), WSt.regs_setReg_other _ 7 2 _ (by decide),
          WSt.regs_setReg_other _ 6 2 _ (by decide), WSt.regs_setReg_other _ 5 2 _ (by decide),
          WSt.regs_setReg_other _ 4 2 _ (by decide), WSt.regs_setReg_other _ 3 2 _ (by decide),
          WSt.regs_setReg_other _ 8 3 _ (by decide), WSt.regs_setReg_other _ 7 3 _ (by decide),
          WSt.regs_setReg_other _ 6 3 _ (by decide), WSt.regs_setReg_other _ 5 3 _ (by decide),
          WSt.regs_setReg_other _ 4 3 _ (by decide),
          WSt.regs_setReg_other _ 8 4 _ (by decide), WSt.regs_setReg_other _ 7 4 _ (by decide),
          WSt.regs_setReg_other _ 6 4 _ (by decide), WSt.regs_setReg_other _ 5 4 _ (by decide),
          WSt.regs_setReg_other _ 8 5 _ (by decide), WSt.regs_setReg_other _ 7 5 _ (by decide),
          WSt.regs_setReg_other _ 6 5 _ (by decide),
          WSt.regs_setReg_other _ 8 6 _ (by decide), WSt.regs_setReg_other _ 7 6 _ (by decide),
          WSt.regs_setReg_other _ 8 7 _ (by decide),
          WSt.regs_setReg_same]
      show ((L.foldl _ ((stepW bA bB fA fB i).run st)).regs 0 l) = _
      rw [ih _ l, hmem, hacc]
      rfl

theorem warpDotBody_spec (bA bB : Buf) (ixA ixB : IdxE) (K cta : Nat) (st : WSt) :
    (((warpDotBody bA bB ixA ixB K).elabAt cta 0).run st).regs 0
      = dotLane (st.mem bA) (st.mem bB)
          (fun i l => ixA.eval cta i l) (fun i l => ixB.eval cta i l) K := by
  funext l
  show ((List.range K).foldl _ ((WStmt.setR 0 (.lit (NumOps.ofNat 0))).run st)).regs 0 l = _
  have hstep : (fun j => EWStmt.elabAt cta j (fun _ _ => 0) (fun _ _ => 0)
        (((EWStmt.loadV4 1 2 3 4 bA ixA).seq (EWStmt.loadV4 5 6 7 8 bB ixB)).seq
          (EWStmt.setR 0 dotStepE)))
      = stepW bA bB (fun i l => ixA.eval cta i l) (fun i l => ixB.eval cta i l) := rfl
  rw [hstep, warpDot_fold bA bB (fun i l => ixA.eval cta i l) (fun i l => ixB.eval cta i l)
        (List.range K) _ l]
  simp only [wrun_setR, WSt.regs_setReg_same, WFExp.eval, dotLane]
  rfl

/-- **The schema is correct.**  Proven once, for every buffer pair, every
    addressing scheme, every trip count and every block. -/
theorem warpDotV4_spec (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (oi : IdxE)
    (K cta : Nat) (st : WSt) :
    (((warpDotV4 bA bB ixA ixB out oi K).elabIn cta).run st).mem out
        (oi.eval cta 0 ⟨0, by decide⟩)
      = bflyFold (dotLane (st.mem bA) (st.mem bB)
          (fun i l => ixA.eval cta i l) (fun i l => ixB.eval cta i l) K) ⟨0, by decide⟩ := by
  show (((warpReduceSum 0 1).run
          (((warpDotBody bA bB ixA ixB K).elabAt cta 0).run st)).store1 out _
        (((warpReduceSum 0 1).run
          (((warpDotBody bA bB ixA ixB K).elabAt cta 0).run st)).regs 0 ⟨0, by decide⟩)).mem out _ = _
  rw [WSt.mem_store1_same, warpReduceSum_spec 0 1 (by decide), warpDotBody_spec]

-- ---------------------------------------------------------------------------
-- The spec-language bridge
-- ---------------------------------------------------------------------------

variable {Γ : Nat}

/-- The per-lane fold, as a spec-language term. -/
def dotLaneE (ae be : Nat → Expr Γ) (fA fB : Nat → Lane → Nat) (K : Nat) (l : Lane) :
    Expr Γ :=
  (List.range K).foldl
    (fun acc i => .add (.add (.add
        (.add acc (.mul (ae (fA i l)) (be (fB i l))))
        (.mul (ae (fA i l + 1)) (be (fB i l + 1))))
        (.mul (ae (fA i l + 2)) (be (fB i l + 2))))
        (.mul (ae (fA i l + 3)) (be (fB i l + 3))))
    (.lit 0)

/-- The whole two-level reduction as one `Expr`. -/
def warpDotV4E (ae be : Nat → Expr Γ) (fA fB : Nat → Lane → Nat) (K : Nat) : Expr Γ :=
  bflyStepE 1 (bflyStepE 2 (bflyStepE 4 (bflyStepE 8 (bflyStepE 16
    (dotLaneE ae be fA fB K))))) ⟨0, by decide⟩

theorem denote_dotLaneE (env : Fin Γ → Float32) (ae be : Nat → Expr Γ)
    (fA fB : Nat → Lane → Nat) (K : Nat) (memA memB : Nat → Float32)
    (ha : ∀ i, denote env (ae i) = memA i) (hb : ∀ i, denote env (be i) = memB i) (l : Lane) :
    denote env (dotLaneE ae be fA fB K l) = dotLane memA memB fA fB K l := by
  show denote env ((List.range K).foldl _ (.lit 0)) = (List.range K).foldl _ (NumOps.ofNat 0)
  have key : ∀ (L : List Nat) (acc : Expr Γ),
      denote env (L.foldl (fun acc i => Expr.add (.add (.add
          (.add acc (.mul (ae (fA i l)) (be (fB i l))))
          (.mul (ae (fA i l + 1)) (be (fB i l + 1))))
          (.mul (ae (fA i l + 2)) (be (fB i l + 2))))
          (.mul (ae (fA i l + 3)) (be (fB i l + 3)))) acc)
        = L.foldl (fun acc i => NumOps.add (NumOps.add (NumOps.add
            (NumOps.add acc (NumOps.mul (memA (fA i l)) (memB (fB i l))))
            (NumOps.mul (memA (fA i l + 1)) (memB (fB i l + 1))))
            (NumOps.mul (memA (fA i l + 2)) (memB (fB i l + 2))))
            (NumOps.mul (memA (fA i l + 3)) (memB (fB i l + 3)))) (denote env acc) := by
    intro L
    induction L with
    | nil => intro acc; rfl
    | cons i L ih =>
        intro acc
        rw [List.foldl_cons, List.foldl_cons, ih]
        simp only [denote_add, denote_mul, ha, hb]
  rw [key (List.range K) (.lit 0)]
  rfl

/-- **The schema computes a spec-language term.**

    `ha`/`hb` are the layout links: buffer slot `i` holds what spec term `ae i`
    (resp. `be i`) denotes.  Given those, the kernel's store lands exactly
    `denote env spec` — for any instantiation of the schema, with no further
    proof. -/
theorem warpDotV4_implements (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (oi : IdxE)
    (K cta : Nat) (st : WSt) (env : Fin Γ → Float32) (ae be : Nat → Expr Γ)
    (ha : ∀ i, denote env (ae i) = st.mem bA i)
    (hb : ∀ i, denote env (be i) = st.mem bB i) :
    (((warpDotV4 bA bB ixA ixB out oi K).elabIn cta).run st).mem out
        (oi.eval cta 0 ⟨0, by decide⟩)
      = denote env (warpDotV4E ae be
          (fun i l => ixA.eval cta i l) (fun i l => ixB.eval cta i l) K) := by
  rw [warpDotV4_spec bA bB ixA ixB out oi K cta st]
  show _ = denote env (bflyStepE 1 (bflyStepE 2 (bflyStepE 4 (bflyStepE 8
    (bflyStepE 16 (dotLaneE ae be _ _ K))))) ⟨0, by decide⟩)
  simp only [bflyStepE, denote_add, bflyFold, bflyStep,
             denote_dotLaneE env ae be _ _ K (st.mem bA) (st.mem bB) ha hb]

/-- Sum of squares is the schema with both operands equal — no new proof.

    (The *shipped* sum-of-squares kernel keeps its own single-load variant: as a
    dot product with itself it would issue two `ld.global.v4` per quad and halve
    the achieved bandwidth.  The schema still covers it, which is the point.) -/
theorem warpDotV4_sumsq (b : Buf) (ix : IdxE) (out : Buf) (oi : IdxE)
    (K cta : Nat) (st : WSt) :
    (((warpDotV4 b b ix ix out oi K).elabIn cta).run st).mem out
        (oi.eval cta 0 ⟨0, by decide⟩)
      = bflyFold (dotLane (st.mem b) (st.mem b)
          (fun i l => ix.eval cta i l) (fun i l => ix.eval cta i l) K) ⟨0, by decide⟩ :=
  warpDotV4_spec b b ix ix out oi K cta st

-- ---------------------------------------------------------------------------
-- The strided dot: the same reduction, with arbitrary addressing on both sides
-- ---------------------------------------------------------------------------

/-!
  `warpDotV4` issues `ld.global.v4.f32`, which requires the four elements a lane
  reads to be **consecutive**.  For a forward matvec they are — a lane walks
  along a row.  For the *backward* matvec they are not: `∂L/∂xⱼ = Σᵢ adjᵢ·Wᵢⱼ`
  walks down a *column*, so consecutive `i` are `N` floats apart.

  That is the whole reason a backward pass would otherwise have to be compiled
  by unrolling — and unrolling is what makes width 896 impossible (measured:
  the compiled route grows as ≈ `G^2.4`, so 896 is ~10⁹ instructions).

  `dotStrided` is the same two-level reduction with scalar loads and an
  independent `IdxE` per operand, so a transposed walk is just a different
  address expression and the trip count stays a *parameter*.  One warp per
  output, `K = N/32` iterations, **independent of `N` in code size**.

  It is strictly more general than `warpDotV4` and strictly slower (one
  `ld.global.f32` per element instead of a quad), which is the honest trade: the
  forward keeps the vectorised schema, the backward gets the strided one.
-/

/-- One accumulation step of the strided dot: `acc += a·b`. -/
def dotStepSE : WFExp := .add (.reg 0) (.mul (.reg 1) (.reg 2))

def stepWS (bA bB : Buf) (fA fB : Nat → Lane → Nat) (i : Nat) : WStmt :=
  .seq (.seq (.loadIdx 1 bA (fA i)) (.loadIdx 2 bB (fB i))) (.setR 0 dotStepSE)

/-- The strided sweep: zero the accumulator, then `K` scalar-load steps. -/
def dotStridedBody (bA bB : Buf) (ixA ixB : IdxE) (K : Nat) : EWStmt :=
  .seq (.setR 0 (.lit (NumOps.ofNat 0)))
       (.forN K (.seq (.seq (.loadIdx 1 bA ixA) (.loadIdx 2 bB ixB))
                      (.setR 0 dotStepSE)))

/-- **The strided schema**: sweep, butterfly, lane-0 store. -/
def dotStrided (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (oi : IdxE) (K : Nat) :
    EWStmt :=
  .seq (.seq (dotStridedBody bA bB ixA ixB K)
             (.seq (warpRoundE 16) (.seq (warpRoundE 8) (.seq (warpRoundE 4)
               (.seq (warpRoundE 2) (warpRoundE 1))))))
       (.storeLane0 out oi 0)

/-- The fold one lane performs — sequential, one element per step, in load
    order.  Committing to it is what makes the theorem exact at `Float32`. -/
def dotStridedLane (memA memB : Nat → Float32) (fA fB : Nat → Lane → Nat) (K : Nat)
    (l : Lane) : Float32 :=
  (List.range K).foldl
    (fun acc i => NumOps.add acc (NumOps.mul (memA (fA i l)) (memB (fB i l))))
    (NumOps.ofNat 0)

theorem dotStrided_fold (bA bB : Buf) (fA fB : Nat → Lane → Nat) :
    ∀ (L : List Nat) (st : WSt) (l : Lane),
      ((L.foldl (fun s' i => (stepWS bA bB fA fB i).run s') st).regs 0 l)
        = L.foldl (fun acc i =>
            NumOps.add acc (NumOps.mul (st.mem bA (fA i l)) (st.mem bB (fB i l))))
          (st.regs 0 l) := by
  intro L
  induction L with
  | nil => intro st l; rfl
  | cons i L ih =>
      intro st l
      have hmem : ((stepWS bA bB fA fB i).run st).mem = st.mem := rfl
      have hacc : ((stepWS bA bB fA fB i).run st).regs 0
          = fun l => NumOps.add (st.regs 0 l)
              (NumOps.mul (st.mem bA (fA i l)) (st.mem bB (fB i l))) := by
        show ((WStmt.setR 0 dotStepSE).run
                ((WStmt.seq (.loadIdx 1 bA (fA i)) (.loadIdx 2 bB (fB i))).run st)).regs 0 = _
        simp only [wrun_setR, WSt.regs_setReg_same]
        funext l'
        simp only [dotStepSE, WFExp.eval, wrun_seq, wrun_loadIdx, WSt.mem_setReg,
          WSt.regs_setReg_other _ 2 0 _ (by decide),
          WSt.regs_setReg_other _ 1 0 _ (by decide),
          WSt.regs_setReg_other _ 2 1 _ (by decide),
          WSt.regs_setReg_same]
      show ((L.foldl _ ((stepWS bA bB fA fB i).run st)).regs 0 l) = _
      rw [ih _ l, hmem, hacc]
      rfl

/-- The strided sweep touches no memory — it only loads and accumulates.  The
    pipeline layer needs this to conclude that a reduction's sole memory effect
    is its final lane-0 store. -/
theorem dotStridedBody_mem (bA bB : Buf) (ixA ixB : IdxE) (K cta : Nat) (st : WSt) :
    (((dotStridedBody bA bB ixA ixB K).elabAt cta 0).run st).mem = st.mem := by
  show ((WStmt.forN K (fun j => (stepWS bA bB (fun i l => ixA.eval cta i l)
          (fun i l => ixB.eval cta i l) j))).run
        ((WStmt.setR 0 (.lit (NumOps.ofNat 0))).run st)).mem = _
  have key : ∀ (L : List Nat) (s : WSt),
      (L.foldl (fun s' i => (stepWS bA bB (fun i l => ixA.eval cta i l)
        (fun i l => ixB.eval cta i l) i).run s') s).mem = s.mem := by
    intro L
    induction L with
    | nil => intro _; rfl
    | cons i L ih =>
        intro s
        rw [List.foldl_cons, ih]
        rfl
  exact key (List.range K) _

theorem dotStridedBody_spec (bA bB : Buf) (ixA ixB : IdxE) (K cta : Nat) (st : WSt) :
    (((dotStridedBody bA bB ixA ixB K).elabAt cta 0).run st).regs 0
      = dotStridedLane (st.mem bA) (st.mem bB)
          (fun i l => ixA.eval cta i l) (fun i l => ixB.eval cta i l) K := by
  funext l
  show ((List.range K).foldl _ ((WStmt.setR 0 (.lit (NumOps.ofNat 0))).run st)).regs 0 l = _
  have hstep : (fun j => EWStmt.elabAt cta j (fun _ _ => 0) (fun _ _ => 0)
        (((EWStmt.loadIdx 1 bA ixA).seq (EWStmt.loadIdx 2 bB ixB)).seq
          (EWStmt.setR 0 dotStepSE)))
      = stepWS bA bB (fun i l => ixA.eval cta i l) (fun i l => ixB.eval cta i l) := rfl
  rw [hstep, dotStrided_fold bA bB (fun i l => ixA.eval cta i l)
        (fun i l => ixB.eval cta i l) (List.range K) _ l]
  simp only [wrun_setR, WSt.regs_setReg_same, WFExp.eval, dotStridedLane]
  rfl

/-- The fold depends on memory only at the addresses it reads.  What a pipeline
    needs: an upstream stage need only agree where the reduction looks. -/
theorem dotStridedLane_congr (memA memA' memB memB' : Nat → Float32)
    (fA fB : Nat → Lane → Nat) (K : Nat) (l : Lane)
    (hA : ∀ i, i < K → ∀ l', memA (fA i l') = memA' (fA i l'))
    (hB : ∀ i, i < K → ∀ l', memB (fB i l') = memB' (fB i l')) :
    dotStridedLane memA memB fA fB K l = dotStridedLane memA' memB' fA fB K l := by
  show (List.range K).foldl _ (NumOps.ofNat 0) = (List.range K).foldl _ (NumOps.ofNat 0)
  have key : ∀ (L : List Nat), (∀ i ∈ L, i < K) → ∀ (a : Float32),
      L.foldl (fun acc i => NumOps.add acc (NumOps.mul (memA (fA i l)) (memB (fB i l)))) a
        = L.foldl (fun acc i =>
            NumOps.add acc (NumOps.mul (memA' (fA i l)) (memB' (fB i l)))) a := by
    intro L
    induction L with
    | nil => intro _ _; rfl
    | cons i L ih =>
        intro hm a
        rw [List.foldl_cons, List.foldl_cons,
            hA i (hm i (List.mem_cons_self)) l, hB i (hm i (List.mem_cons_self)) l,
            ih (fun i' hi' => hm i' (List.mem_cons_of_mem i hi'))]
  exact key (List.range K) (fun i hi => List.mem_range.mp hi) _

/-- **The strided schema is correct** — every buffer pair, every pair of address
    expressions, every trip count, every block. -/
theorem dotStrided_spec (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (oi : IdxE)
    (K cta : Nat) (st : WSt) :
    (((dotStrided bA bB ixA ixB out oi K).elabIn cta).run st).mem out
        (oi.eval cta 0 ⟨0, by decide⟩)
      = bflyFold (dotStridedLane (st.mem bA) (st.mem bB)
          (fun i l => ixA.eval cta i l) (fun i l => ixB.eval cta i l) K)
          ⟨0, by decide⟩ := by
  show (((warpReduceSum 0 1).run
          (((dotStridedBody bA bB ixA ixB K).elabAt cta 0).run st)).store1 out _
        (((warpReduceSum 0 1).run
          (((dotStridedBody bA bB ixA ixB K).elabAt cta 0).run st)).regs 0 ⟨0, by decide⟩)).mem out _ = _
  rw [WSt.mem_store1_same, warpReduceSum_spec 0 1 (by decide), dotStridedBody_spec]

/-- The maximum's step: one load, one `max.f32` into the accumulator. -/
def maxStepSE : WFExp := .maxW (.reg 0) (.reg 1)

def stepMaxWS (b : Buf) (f : Nat → Lane → Nat) (i : Nat) : WStmt :=
  .seq (.loadIdx 1 b (f i)) (.setR 0 maxStepSE)

/-- The strided maximum sweep: seed the accumulator, then `K` load-and-max
    steps.  The seed is a parameter because a row maximum over a bounded
    quantity has a known floor and a softmax should not have to invent one. -/
def maxStridedBody (b : Buf) (ix : IdxE) (K : Nat) (init : Float32) : EWStmt :=
  .seq (.setR 0 (.lit init))
       (.forN K (.seq (.loadIdx 1 b ix) (.setR 0 maxStepSE)))

/-- The fold one lane performs — sequential, one element per step, in load
    order, exactly as the sum's is. -/
def maxStridedLane (mem : Nat → Float32) (f : Nat → Lane → Nat) (K : Nat)
    (init : Float32) (l : Lane) : Float32 :=
  (List.range K).foldl (fun acc i => NumOps.max acc (mem (f i l))) init

theorem maxStrided_fold (b : Buf) (f : Nat → Lane → Nat) :
    ∀ (L : List Nat) (st : WSt) (l : Lane),
      ((L.foldl (fun s' i => (stepMaxWS b f i).run s') st).regs 0 l)
        = L.foldl (fun acc i => NumOps.max acc (st.mem b (f i l))) (st.regs 0 l) := by
  intro L
  induction L with
  | nil => intro st l; rfl
  | cons i L ih =>
      intro st l
      have hmem : ((stepMaxWS b f i).run st).mem = st.mem := rfl
      have hacc : ((stepMaxWS b f i).run st).regs 0
          = fun l => NumOps.max (st.regs 0 l) (st.mem b (f i l)) := by
        show ((WStmt.setR 0 maxStepSE).run ((WStmt.loadIdx 1 b (f i)).run st)).regs 0 = _
        simp only [wrun_setR, WSt.regs_setReg_same]
        funext l'
        simp only [maxStepSE, WFExp.eval, wrun_loadIdx, WSt.mem_setReg,
          WSt.regs_setReg_other _ 1 0 _ (by decide), WSt.regs_setReg_same]
      show ((L.foldl _ ((stepMaxWS b f i).run st)).regs 0 l) = _
      rw [ih _ l, hmem, hacc]
      rfl

theorem maxStridedBody_spec (b : Buf) (ix : IdxE) (K cta : Nat) (init : Float32)
    (st : WSt) :
    (((maxStridedBody b ix K init).elabAt cta 0).run st).regs 0
      = maxStridedLane (st.mem b) (fun i l => ix.eval cta i l) K init := by
  funext l
  show ((List.range K).foldl _ ((WStmt.setR 0 (.lit init)).run st)).regs 0 l = _
  have hstep : (fun j => EWStmt.elabAt cta j (fun _ _ => 0) (fun _ _ => 0)
        ((EWStmt.loadIdx 1 b ix).seq (EWStmt.setR 0 maxStepSE)))
      = stepMaxWS b (fun i l => ix.eval cta i l) := rfl
  rw [hstep, maxStrided_fold b (fun i l => ix.eval cta i l) (List.range K) _ l]
  simp only [wrun_setR, WSt.regs_setReg_same, WFExp.eval, maxStridedLane]
  rfl

/-- The strided per-lane fold, as a spec-language term. -/
def dotStridedLaneE (ae be : Nat → Expr Γ) (fA fB : Nat → Lane → Nat) (K : Nat)
    (l : Lane) : Expr Γ :=
  (List.range K).foldl
    (fun acc i => .add acc (.mul (ae (fA i l)) (be (fB i l)))) (.lit 0)

/-- The whole two-level strided reduction as one `Expr`. -/
def dotStridedE (ae be : Nat → Expr Γ) (fA fB : Nat → Lane → Nat) (K : Nat) :
    Expr Γ :=
  bflyStepE 1 (bflyStepE 2 (bflyStepE 4 (bflyStepE 8 (bflyStepE 16
    (dotStridedLaneE ae be fA fB K))))) ⟨0, by decide⟩

theorem denote_dotStridedLaneE (env : Fin Γ → Float32) (ae be : Nat → Expr Γ)
    (fA fB : Nat → Lane → Nat) (K : Nat) (memA memB : Nat → Float32)
    (ha : ∀ i, denote env (ae i) = memA i) (hb : ∀ i, denote env (be i) = memB i)
    (l : Lane) :
    denote env (dotStridedLaneE ae be fA fB K l) = dotStridedLane memA memB fA fB K l := by
  show denote env ((List.range K).foldl _ (.lit 0)) = (List.range K).foldl _ (NumOps.ofNat 0)
  have key : ∀ (L : List Nat) (acc : Expr Γ),
      denote env (L.foldl (fun acc i =>
          Expr.add acc (.mul (ae (fA i l)) (be (fB i l)))) acc)
        = L.foldl (fun acc i =>
            NumOps.add acc (NumOps.mul (memA (fA i l)) (memB (fB i l)))) (denote env acc) := by
    intro L
    induction L with
    | nil => intro acc; rfl
    | cons i L ih =>
        intro acc
        rw [List.foldl_cons, List.foldl_cons, ih]
        simp only [denote_add, denote_mul, ha, hb]
  rw [key (List.range K) (.lit 0)]
  rfl

/-- **The strided schema computes a spec-language term.**

    Same shape as `warpDotV4_implements`, and the same zero-obligation story:
    given the two layout links, the store lands exactly `denote env spec`.  This
    is what the backward matvec stands on. -/
theorem dotStrided_implements (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (oi : IdxE)
    (K cta : Nat) (st : WSt) (env : Fin Γ → Float32) (ae be : Nat → Expr Γ)
    (ha : ∀ i, denote env (ae i) = st.mem bA i)
    (hb : ∀ i, denote env (be i) = st.mem bB i) :
    (((dotStrided bA bB ixA ixB out oi K).elabIn cta).run st).mem out
        (oi.eval cta 0 ⟨0, by decide⟩)
      = denote env (dotStridedE ae be
          (fun i l => ixA.eval cta i l) (fun i l => ixB.eval cta i l) K) := by
  rw [dotStrided_spec bA bB ixA ixB out oi K cta st]
  show _ = denote env (bflyStepE 1 (bflyStepE 2 (bflyStepE 4 (bflyStepE 8
    (bflyStepE 16 (dotStridedLaneE ae be _ _ K))))) ⟨0, by decide⟩)
  simp only [bflyStepE, denote_add, bflyFold, bflyStep,
             denote_dotStridedLaneE env ae be _ _ K (st.mem bA) (st.mem bB) ha hb]

-- ---------------------------------------------------------------------------
-- The other butterfly: max
-- ---------------------------------------------------------------------------

/-! A softmax stabilises by subtracting the row maximum, so it needs the same
    five-round shuffle network the sum uses — with `max` in place of `add`.

    `WFExp.maxW` makes that expressible: it is `max.f32`, one instruction, and
    `NumOps.max` is *defined* from `le` rather than assumed, so the spec side
    needs no new axiom either. -/

-- ---------------------------------------------------------------------------
-- The generic sweep: one load, one accumulate
-- ---------------------------------------------------------------------------

/-!
  `warpDot_fold` above proves the *vectorised dot* sweep, and `warpAccSq_fold`
  in `Warp.lean` proves the *sum-of-squares* sweep.  They are the same argument
  twice, specialised to a fixed combining expression.

  Every remaining reduction in this stack — RMSNorm's `Σx²`, softmax's row
  maximum, softmax's `Σexp(x−m)`, and argmax's two passes — is that argument a
  third, fourth, fifth and sixth time.  So it is worth doing once, parametric in
  the combining expression.

  The one subtlety is that a combining expression may read registers *besides*
  the accumulator and the loaded element — softmax's sum pass reads the row
  maximum in `%fw5`.  Those registers are loop-invariant, and `SweepFrameOn` says
  so: the sweep disturbs only `acc` and `d`, so anything else still holds its
  entry value and `g` may depend on it.
-/

/-- The body every reduction sweep shares: load one element into `%fw{d}`, then
    fold it into `%fw{acc}` with `f`. -/
def sweepBody (b : Buf) (d acc : Nat) (f : WFExp) (ix : IdxE) : EWStmt :=
  .seq (.loadIdx d b ix) (.setR acc f)

/-- What a sweep's combining expression is allowed to depend on: memory, and
    the registers named in `keep`.

    `keep` is explicit rather than "everything but `acc` and `d`" because passes
    get **chained**, and a butterfly between two sweeps clobbers its shuffle
    temporary.  Softmax's sum pass reads the row maximum in `%fw5` and must not
    care that `%fw1` was destroyed on the way; naming `keep = [5]` says exactly
    that, where a blanket frame would be false. -/
def SweepFrameOn (keep : List Nat) (st0 st : WSt) : Prop :=
  st.mem = st0.mem ∧ ∀ r ∈ keep, st.regs r = st0.regs r

theorem sweepFrame_refl (keep : List Nat) (st : WSt) : SweepFrameOn keep st st :=
  ⟨rfl, fun _ _ => rfl⟩

/-- **The generic sweep leaves a left fold in the accumulator.**

    `g` is the semantic combining function; `hf` links it to the syntactic `f`,
    and is discharged per instance by `simp [WFExp.eval]` — for a concrete `f`
    it is a one-liner.  The fold is *sequential in load order*, which is what
    makes every instance an exact `Float32` equality rather than a bound. -/
theorem sweep_fold (b : Buf) (d acc : Nat) (hne : d ≠ acc) (f : WFExp)
    (ix : Nat → Lane → Nat) (g : Lane → Float32 → Float32 → Float32)
    (keep : List Nat) (hkeep : ∀ r ∈ keep, r ≠ acc ∧ r ≠ d) (st0 : WSt)
    (hf : ∀ st, SweepFrameOn keep st0 st → ∀ l,
            f.eval st l = g l (st.regs acc l) (st.regs d l)) :
    ∀ (L : List Nat) (st : WSt), SweepFrameOn keep st0 st → ∀ (l : Lane),
      ((L.foldl (fun s i =>
          (WStmt.seq (.loadIdx d b (ix i)) (.setR acc f)).run s) st).regs acc l)
        = L.foldl (fun a i => g l a (st0.mem b (ix i l))) (st.regs acc l) := by
  intro L
  induction L with
  | nil => intro st _ l; rfl
  | cons i L ih =>
      intro st hfr l
      -- after the load, before the accumulate
      have hmem1 : (st.setReg d (fun l => st.mem b (ix i l))).mem = st.mem := rfl
      have hfr1 : SweepFrameOn keep st0 (st.setReg d (fun l => st.mem b (ix i l))) := by
        refine ⟨by rw [hmem1]; exact hfr.1, ?_⟩
        intro r hr
        rw [WSt.regs_setReg_other _ d r _ (hkeep r hr).2]
        exact hfr.2 r hr
      -- the whole body's effect on the accumulator
      have hacc : ((WStmt.seq (.loadIdx d b (ix i)) (.setR acc f)).run st).regs acc
          = fun l => g l (st.regs acc l) (st0.mem b (ix i l)) := by
        simp only [wrun_seq, wrun_loadIdx, wrun_setR, WSt.regs_setReg_same]
        funext l'
        rw [hf _ hfr1 l', WSt.regs_setReg_other _ d acc _ (Ne.symm hne),
            WSt.regs_setReg_same, hfr.1]
      have hbody : SweepFrameOn keep st0
          ((WStmt.seq (.loadIdx d b (ix i)) (.setR acc f)).run st) := by
        refine ⟨?_, ?_⟩
        · show ((WStmt.setR acc f).run (st.setReg d _)).mem = _
          rw [wrun_setR, WSt.mem_setReg, hmem1]; exact hfr.1
        · intro r hr
          show ((WStmt.setR acc f).run (st.setReg d _)).regs r = _
          rw [wrun_setR, WSt.regs_setReg_other _ acc r _ (hkeep r hr).1,
              WSt.regs_setReg_other _ d r _ (hkeep r hr).2]
          exact hfr.2 r hr
      show ((L.foldl _ ((WStmt.seq _ _).run st)).regs acc l) = _
      rw [ih _ hbody l, hacc]
      rfl

/-- **What a sweep leaves alone.**  Memory is untouched, and so is every
    register but the accumulator and the load target.

    This is what lets passes be *chained*: softmax's sum pass reads the row
    maximum in `%fw5`, and this is the theorem that says the max is still there.
    Without it each pass would have to re-establish the previous pass's result
    by hand, which is exactly the bookkeeping schemas exist to remove. -/
theorem sweep_frame (b : Buf) (d acc : Nat) (f : WFExp) (fx : Nat → Lane → Nat)
    (keep : List Nat) (hkeep : ∀ r ∈ keep, r ≠ acc ∧ r ≠ d) :
    ∀ (L : List Nat) (st : WSt),
      SweepFrameOn keep st (L.foldl (fun s i =>
        (WStmt.seq (.loadIdx d b (fx i)) (.setR acc f)).run s) st) := by
  intro L
  induction L with
  | nil => intro st; exact sweepFrame_refl keep st
  | cons i L ih =>
      intro st
      have hstep : SweepFrameOn keep st
          ((WStmt.seq (.loadIdx d b (fx i)) (.setR acc f)).run st) := by
        refine ⟨rfl, ?_⟩
        intro r hr
        show ((WStmt.setR acc f).run (st.setReg d _)).regs r = _
        rw [wrun_setR, WSt.regs_setReg_other _ acc r _ (hkeep r hr).1,
            WSt.regs_setReg_other _ d r _ (hkeep r hr).2]
      obtain ⟨hm, hr⟩ := ih ((WStmt.seq (.loadIdx d b (fx i)) (.setR acc f)).run st)
      show SweepFrameOn keep st (L.foldl _ ((WStmt.seq _ _).run st))
      exact ⟨by rw [hm]; exact hstep.1, fun r hr' => by
        rw [hr r hr']; exact hstep.2 r hr'⟩

/-- The frame, for a sweep written as an elaborated loop. -/
theorem sweepLoop_frame (b : Buf) (d acc : Nat) (f : WFExp) (ix : IdxE) (K cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (keep : List Nat) (hkeep : ∀ r ∈ keep, r ≠ acc ∧ r ≠ d) (st : WSt) :
    SweepFrameOn keep st
      ((WStmt.forN K (fun j => (sweepBody b d acc f ix).elabAt cta j ir im)).run st) :=
  sweep_frame b d acc f (fun j l => ix.eval cta j l ir im) keep hkeep (List.range K) st

/-- The sweep under an elaborated loop, for **any** trip count and elaboration
    environment.  Both `EWStmt` loop forms are this lemma: `forN` supplies the
    trip count as a literal and `forM` reads it from integer memory, but they
    elaborate to the same `WStmt.forN`, so neither needs its own proof. -/
theorem sweepLoop_spec (b : Buf) (d acc : Nat) (hne : d ≠ acc) (f : WFExp) (ix : IdxE)
    (g : Lane → Float32 → Float32 → Float32) (K cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (keep : List Nat) (hkeep : ∀ r ∈ keep, r ≠ acc ∧ r ≠ d) (st : WSt)
    (hf : ∀ st', SweepFrameOn keep st st' → ∀ l,
            f.eval st' l = g l (st'.regs acc l) (st'.regs d l)) (l : Lane) :
    ((WStmt.forN K (fun j => (sweepBody b d acc f ix).elabAt cta j ir im)).run st).regs acc l
      = (List.range K).foldl
          (fun a j => g l a (st.mem b (ix.eval cta j l ir im))) (st.regs acc l) :=
  sweep_fold b d acc hne f (fun j l => ix.eval cta j l ir im) g keep hkeep st hf
    (List.range K) st (sweepFrame_refl keep st) l

/-- The static-trip-count sweep, as kernels write it. -/
theorem sweepN_spec (b : Buf) (d acc : Nat) (hne : d ≠ acc) (f : WFExp) (ix : IdxE)
    (g : Lane → Float32 → Float32 → Float32) (K cta : Nat)
    (keep : List Nat) (hkeep : ∀ r ∈ keep, r ≠ acc ∧ r ≠ d) (st : WSt)
    (hf : ∀ st', SweepFrameOn keep st st' → ∀ l,
            f.eval st' l = g l (st'.regs acc l) (st'.regs d l)) (l : Lane) :
    (((EWStmt.forN K (sweepBody b d acc f ix)).elabAt cta 0).run st).regs acc l
      = (List.range K).foldl
          (fun a i => g l a (st.mem b (ix.eval cta i l))) (st.regs acc l) :=
  sweepLoop_spec b d acc hne f ix g K cta _ _ keep hkeep st hf l

/-- The **dynamic**-trip-count sweep: the bound is read from integer memory, so
    one statement covers every sequence length at once. -/
theorem sweepM_spec (b : Buf) (d acc : Nat) (hne : d ≠ acc) (f : WFExp) (ix : IdxE)
    (g : Lane → Float32 → Float32 → Float32) (bu ad cta i : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (keep : List Nat) (hkeep : ∀ r ∈ keep, r ≠ acc ∧ r ≠ d) (st : WSt)
    (hf : ∀ st', SweepFrameOn keep st st' → ∀ l,
            f.eval st' l = g l (st'.regs acc l) (st'.regs d l)) (l : Lane) :
    (((EWStmt.forM bu ad (sweepBody b d acc f ix)).elabAt cta i ir im).run st).regs acc l
      = (List.range (im bu ad)).foldl
          (fun a j => g l a (st.mem b (ix.eval cta j l ir im))) (st.regs acc l) :=
  sweepLoop_spec b d acc hne f ix g (im bu ad) cta ir im keep hkeep st hf l

/-- The semantic fold a sweep performs, as a named function. -/
def sweepFold (mem : Nat → Float32) (fx : Nat → Lane → Nat)
    (g : Lane → Float32 → Float32 → Float32) (init : Float32) (K : Nat) (l : Lane) :
    Float32 :=
  (List.range K).foldl (fun a i => g l a (mem (fx i l))) init

/-- …and the same fold as a spec-language term. -/
def sweepFoldE (ae : Nat → Expr Γ) (fx : Nat → Lane → Nat)
    (ge : Lane → Expr Γ → Expr Γ → Expr Γ) (init : Expr Γ) (K : Nat) (l : Lane) :
    Expr Γ :=
  (List.range K).foldl (fun a i => ge l a (ae (fx i l))) init

/-- **The spec-language fold denotes the machine fold.**  `ha` is the layout
    link (buffer slot `i` holds what `ae i` denotes); `hg` links the syntactic
    combiner to the semantic one.  Both are one-liners at each instance. -/
theorem denote_sweepFoldE (env : Fin Γ → Float32) (ae : Nat → Expr Γ)
    (fx : Nat → Lane → Nat) (ge : Lane → Expr Γ → Expr Γ → Expr Γ)
    (g : Lane → Float32 → Float32 → Float32) (mem : Nat → Float32)
    (ha : ∀ i, denote env (ae i) = mem i)
    (hg : ∀ (l : Lane) (a x : Expr Γ),
            denote env (ge l a x) = g l (denote env a) (denote env x))
    (init : Expr Γ) (K : Nat) (l : Lane) :
    denote env (sweepFoldE ae fx ge init K l)
      = sweepFold mem fx g (denote env init) K l := by
  show denote env ((List.range K).foldl _ init) = (List.range K).foldl _ (denote env init)
  have key : ∀ (L : List Nat) (acc : Expr Γ),
      denote env (L.foldl (fun a i => ge l a (ae (fx i l))) acc)
        = L.foldl (fun a i => g l a (mem (fx i l))) (denote env acc) := by
    intro L
    induction L with
    | nil => intro acc; rfl
    | cons i L ih =>
        intro acc
        rw [List.foldl_cons, List.foldl_cons, ih, hg l acc (ae (fx i l)), ha]
  exact key (List.range K) init

-- ---------------------------------------------------------------------------
-- The generic butterfly: one shuffle network, any combining operation
-- ---------------------------------------------------------------------------

/-!
  `warpRound`/`bflyStep` in `Warp.lean` are hardcoded to `add`.  A softmax needs
  the same five-round network under `max`, and an argmax under `min` — so the
  network is generalised here over the combining expression, once.

  The shuffle is what it always was; only the fold changes.  That means the
  *tree shape* is proven once and every reduction operation inherits it.
-/

/-- One butterfly round under an arbitrary combining expression. -/
def bflyRoundOp (acc tmp mask : Nat) (op : WFExp) : WStmt :=
  .seq (.shflXor tmp acc mask) (.setR acc op)

/-- The semantic step: combine each lane's value with its partner's. -/
def bflyStepOp (g : Float32 → Float32 → Float32) (m : Nat) (v : Lane → Float32) :
    Lane → Float32 :=
  fun l => g (v l) (v (xorLane l m))

/-- The five-round network, semantically. -/
def bflyFoldOp (g : Float32 → Float32 → Float32) (v : Lane → Float32) : Lane → Float32 :=
  bflyStepOp g 1 (bflyStepOp g 2 (bflyStepOp g 4 (bflyStepOp g 8 (bflyStepOp g 16 v))))

theorem bflyRoundOp_spec (acc tmp mask : Nat) (h : tmp ≠ acc) (op : WFExp)
    (g : Float32 → Float32 → Float32)
    (hop : ∀ (st : WSt) (l : Lane), op.eval st l = g (st.regs acc l) (st.regs tmp l))
    (st : WSt) :
    ((bflyRoundOp acc tmp mask op).run st).regs acc = bflyStepOp g mask (st.regs acc) := by
  simp only [bflyRoundOp, wrun_seq, wrun_setR, wrun_shflXor, WSt.regs_setReg_same]
  funext l
  rw [hop, WSt.regs_setReg_other _ tmp acc _ (Ne.symm h), WSt.regs_setReg_same]
  rfl

/-- **The generic warp reduction realises the butterfly tree, for any operation.**

    `warpReduceSum_spec` is this at `add`; `warpReduceMaxE_spec` below is it at
    `max`.  The tree is proven once. -/
theorem warpReduceOp_spec (acc tmp : Nat) (h : tmp ≠ acc) (op : WFExp)
    (g : Float32 → Float32 → Float32)
    (hop : ∀ (st : WSt) (l : Lane), op.eval st l = g (st.regs acc l) (st.regs tmp l))
    (st : WSt) :
    ((WStmt.seq (bflyRoundOp acc tmp 16 op)
       (.seq (bflyRoundOp acc tmp 8 op)
         (.seq (bflyRoundOp acc tmp 4 op)
           (.seq (bflyRoundOp acc tmp 2 op) (bflyRoundOp acc tmp 1 op))))).run st).regs acc
      = bflyFoldOp g (st.regs acc) := by
  simp only [wrun_seq]
  rw [bflyRoundOp_spec acc tmp 1 h op g hop, bflyRoundOp_spec acc tmp 2 h op g hop,
      bflyRoundOp_spec acc tmp 4 h op g hop, bflyRoundOp_spec acc tmp 8 h op g hop,
      bflyRoundOp_spec acc tmp 16 h op g hop]
  rfl

theorem sweepFrame_trans {keep : List Nat} {a b c : WSt}
    (h1 : SweepFrameOn keep a b) (h2 : SweepFrameOn keep b c) : SweepFrameOn keep a c :=
  ⟨by rw [h2.1, h1.1], fun r hr => by rw [h2.2 r hr, h1.2 r hr]⟩

-- ---------------------------------------------------------------------------
-- The chunk/remainder reduction schema
-- ---------------------------------------------------------------------------

/-!
  A warp-level reduction over a **dynamic** length, where the tail cannot be
  masked because this instruction set has no float select.  Four steps:

  1. initialise the accumulator;
  2. sweep the full 32-wide chunks, one distinct element per lane;
  3. reduce across lanes;
  4. fold in the `< 32` remainder — every lane reading the *same* element, so
     the accumulator stays warp-equal without a second reduction and every
     element is counted exactly once.

  Softmax uses this twice (row maximum, then `Σexp`), and an argmax will use it
  twice more.  Proving it once here is what makes those instantiations rather
  than four more bespoke proofs — and it is the shape that was wrong, unproven,
  in the softmax draft that shipped a 32× index overrun.
-/

/-- The schema. -/
def chunkRemReduce (b : Buf) (d acc : Nat) (f : WFExp) (ixC ixR : IdxE)
    (init : WFExp) (bu adC adR : Nat) (bfly : EWStmt) : EWStmt :=
  .seq (.seq (.setR acc init) (.forM bu adC (sweepBody b d acc f ixC)))
       (.seq bfly (.forM bu adR (sweepBody b d acc f ixR)))

/-- What it computes, in the order it computes it. -/
def chunkRemFold (mem : Nat → Float32) (g : Lane → Float32 → Float32 → Float32)
    (initV : Lane → Float32) (fxC fxR : Nat → Lane → Nat) (KC KR : Nat)
    (bflyG : (Lane → Float32) → Lane → Float32) (l : Lane) : Float32 :=
  (List.range KR).foldl (fun a j => g l a (mem (fxR j l)))
    (bflyG (fun l' =>
      (List.range KC).foldl (fun a j => g l' a (mem (fxC j l'))) (initV l')) l)

/-- **The chunk/remainder reduction is correct**, for every trip count the
    integer memory names — so one statement covers every sequence length.

    The butterfly is a parameter, described by its three effects, so the same
    theorem serves the `add` network and the `max` network. -/
theorem chunkRemReduce_spec (b : Buf) (d acc : Nat) (hne : d ≠ acc) (f : WFExp)
    (ixC ixR : IdxE) (init : WFExp) (bu adC adR cta i : Nat) (bfly : EWStmt)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (g : Lane → Float32 → Float32 → Float32)
    (bflyG : (Lane → Float32) → Lane → Float32)
    (keep : List Nat) (hkeep : ∀ r ∈ keep, r ≠ acc ∧ r ≠ d) (st : WSt)
    (initV : Lane → Float32) (hinit : ∀ l, init.eval st l = initV l)
    (hf : ∀ st', SweepFrameOn keep st st' → ∀ l,
            f.eval st' l = g l (st'.regs acc l) (st'.regs d l))
    (hbflyR : ∀ st', ((bfly.elabAt cta i ir im).run st').regs acc = bflyG (st'.regs acc))
    (hbflyM : ∀ st', ((bfly.elabAt cta i ir im).run st').mem = st'.mem)
    (hbflyK : ∀ st', ∀ r ∈ keep, ((bfly.elabAt cta i ir im).run st').regs r = st'.regs r)
    (l : Lane) :
    (((chunkRemReduce b d acc f ixC ixR init bu adC adR bfly).elabAt cta i ir im).run st).regs acc l
      = chunkRemFold (st.mem b) g initV
          (fun j l' => ixC.eval cta j l' ir im) (fun j l' => ixR.eval cta j l' ir im)
          (im bu adC) (im bu adR) bflyG l := by
  show ((WStmt.forN (im bu adR) (fun j => (sweepBody b d acc f ixR).elabAt cta j ir im)).run
          ((bfly.elabAt cta i ir im).run
            ((WStmt.forN (im bu adC)
              (fun j => (sweepBody b d acc f ixC).elabAt cta j ir im)).run
                ((WStmt.setR acc init).run st)))).regs acc l = _
  -- after the initialiser
  have hI : SweepFrameOn keep st ((WStmt.setR acc init).run st) :=
    ⟨rfl, fun r hr => WSt.regs_setReg_other _ acc r _ (hkeep r hr).1⟩
  -- after the chunk sweep
  have hC : SweepFrameOn keep st
      ((WStmt.forN (im bu adC)
        (fun j => (sweepBody b d acc f ixC).elabAt cta j ir im)).run
          ((WStmt.setR acc init).run st)) :=
    sweepFrame_trans hI (sweepLoop_frame b d acc f ixC (im bu adC) cta ir im keep hkeep _)
  have hCval : ∀ l', ((WStmt.forN (im bu adC)
        (fun j => (sweepBody b d acc f ixC).elabAt cta j ir im)).run
          ((WStmt.setR acc init).run st)).regs acc l'
      = (List.range (im bu adC)).foldl
          (fun a j => g l' a (st.mem b (ixC.eval cta j l' ir im))) (initV l') := by
    intro l'
    rw [sweepLoop_spec b d acc hne f ixC g (im bu adC) cta ir im keep hkeep _
          (fun st' hfr => hf st' (sweepFrame_trans hI hfr)) l']
    simp only [wrun_setR, WSt.mem_setReg, WSt.regs_setReg_same]
    rw [hinit l']
  -- after the butterfly
  have hB : SweepFrameOn keep st
      ((bfly.elabAt cta i ir im).run ((WStmt.forN (im bu adC)
        (fun j => (sweepBody b d acc f ixC).elabAt cta j ir im)).run
          ((WStmt.setR acc init).run st))) :=
    sweepFrame_trans hC ⟨hbflyM _, fun r hr => hbflyK _ r hr⟩
  -- the remainder sweep
  rw [sweepLoop_spec b d acc hne f ixR g (im bu adR) cta ir im keep hkeep _
        (fun st' hfr => hf st' (sweepFrame_trans hB hfr)) l,
      hbflyR, hB.1]
  show (List.range (im bu adR)).foldl _ (bflyG _ l) = _
  congr 2
  funext l'
  exact hCval l'

/-- A butterfly round writes only the accumulator and its shuffle temporary. -/
theorem bflyRoundOp_frame (acc tmp mask : Nat) (op : WFExp) (r : Nat)
    (hr : r ≠ acc) (hr' : r ≠ tmp) (st : WSt) :
    ((bflyRoundOp acc tmp mask op).run st).regs r = st.regs r := by
  simp only [bflyRoundOp, wrun_seq, wrun_setR, wrun_shflXor,
             WSt.regs_setReg_other _ acc r _ hr, WSt.regs_setReg_other _ tmp r _ hr']

/-- …and so does the whole five-round network.  This is what lets a value
    computed *before* a reduction survive it — softmax's row maximum sits in
    `%fw5` across the sum butterfly precisely because of this. -/
theorem warpReduceOp_frame (acc tmp : Nat) (op : WFExp) (r : Nat)
    (hr : r ≠ acc) (hr' : r ≠ tmp) (st : WSt) :
    ((WStmt.seq (bflyRoundOp acc tmp 16 op)
       (.seq (bflyRoundOp acc tmp 8 op)
         (.seq (bflyRoundOp acc tmp 4 op)
           (.seq (bflyRoundOp acc tmp 2 op) (bflyRoundOp acc tmp 1 op))))).run st).regs r
      = st.regs r := by
  simp only [wrun_seq]
  rw [bflyRoundOp_frame acc tmp 1 op r hr hr', bflyRoundOp_frame acc tmp 2 op r hr hr',
      bflyRoundOp_frame acc tmp 4 op r hr hr', bflyRoundOp_frame acc tmp 8 op r hr hr',
      bflyRoundOp_frame acc tmp 16 op r hr hr']

/-- The map-and-store pass body: compute into `%fw{r}`, then every lane stores. -/
def storeBody (out : Buf) (r : Nat) (oix : IdxE) (compute : EWStmt) : EWStmt :=
  .seq compute (.storeLane out oix r)

/-- **A store loop writes only its output buffer.**

    The frame half of a store pass's contract — needed by the pipeline layer,
    where a stage must promise not to disturb anything it does not own. -/
theorem storeLoop_otherBuf (out : Buf) (r : Nat) (oix : IdxE) (compute : EWStmt)
    (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (hcompMem : ∀ (j : Nat) (s : WSt), ((compute.elabAt cta j ir im).run s).mem = s.mem)
    (c : Buf) (hc : c ≠ out) :
    ∀ (L : List Nat) (s : WSt),
      (L.foldl (fun s' j => ((storeBody out r oix compute).elabAt cta j ir im).run s') s).mem c
        = s.mem c := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons j L ih =>
      intro s
      rw [List.foldl_cons, ih]
      show ((WStmt.storeLane out (fun l => oix.eval cta j l ir im) r).run
              ((compute.elabAt cta j ir im).run s)).mem c = _
      rw [storeLane_otherBuf out _ r _ c hc, hcompMem j s]

/-- **…and only at addresses some `(iteration, lane)` owns.** -/
theorem storeLoop_otherAddr (out : Buf) (r : Nat) (oix : IdxE) (compute : EWStmt)
    (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (hcompMem : ∀ (j : Nat) (s : WSt), ((compute.elabAt cta j ir im).run s).mem = s.mem)
    (a : Nat) :
    ∀ (L : List Nat) (s : WSt), (∀ j ∈ L, ∀ l : Lane, oix.eval cta j l ir im ≠ a) →
      (L.foldl (fun s' j => ((storeBody out r oix compute).elabAt cta j ir im).run s') s).mem
        out a = s.mem out a := by
  intro L
  induction L with
  | nil => intro _ _; rfl
  | cons j L ih =>
      intro s hno
      rw [List.foldl_cons, ih _ (fun j' hj' => hno j' (List.mem_cons_of_mem j hj'))]
      show ((WStmt.storeLane out (fun l => oix.eval cta j l ir im) r).run
              ((compute.elabAt cta j ir im).run s)).mem out a = _
      rw [storeLane_otherAddr out _ r _ a (hno j (List.mem_cons_self)),
          congrFun (hcompMem j s) out]

/-- **What a store pass leaves in memory.**

    Stated as an invariant rather than with an injectivity side condition: an
    address holds `w` at the end if every `(iteration, lane)` writing it writes
    `w`, and either one of them does or it already held `w`.  Injectivity of the
    address map implies the agreement hypothesis, so instances that have it
    discharge this immediately — but kernels that write an address twice with
    the same value (softmax's remainder lanes do exactly that) are covered too,
    which an injectivity-based statement would wrongly exclude.

    `hcompVal` is where "the computation does not read what this pass writes"
    lives: it may depend on any buffer *other* than `out`. -/
theorem storeLoop_at (out : Buf) (r : Nat) (oix : IdxE) (compute : EWStmt)
    (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt)
    (V : Nat → Lane → Float32) (A : Nat) (w : Float32) (keep : List Nat)
    (hcompMem : ∀ (j : Nat) (s : WSt), ((compute.elabAt cta j ir im).run s).mem = s.mem)
    (hcompKeep : ∀ (j : Nat) (s : WSt), ∀ r' ∈ keep,
                  ((compute.elabAt cta j ir im).run s).regs r' = s.regs r')
    (hcompVal : ∀ (j : Nat) (s : WSt), (∀ c, c ≠ out → s.mem c = st.mem c) →
                  (∀ r' ∈ keep, s.regs r' = st.regs r') →
                  ∀ l, ((compute.elabAt cta j ir im).run s).regs r l = V j l) :
    ∀ (L : List Nat) (s : WSt), (∀ c, c ≠ out → s.mem c = st.mem c) →
      (∀ r' ∈ keep, s.regs r' = st.regs r') →
      (∀ j ∈ L, ∀ l : Lane, oix.eval cta j l ir im = A → V j l = w) →
      ((∃ j ∈ L, ∃ l : Lane, oix.eval cta j l ir im = A) ∨ s.mem out A = w) →
      (L.foldl (fun s' j => ((storeBody out r oix compute).elabAt cta j ir im).run s') s).mem
          out A = w := by
  intro L
  induction L with
  | nil =>
      intro s _ _ _ hd
      rcases hd with ⟨j, hj, _⟩ | h
      · exact absurd hj (by simp)
      · exact h
  | cons j L ih =>
      intro s hinv hkeep hall hd
      -- the state after this iteration
      have hc := hcompVal j s hinv hkeep
      have hcm := hcompMem j s
      have hstep : ((storeBody out r oix compute).elabAt cta j ir im).run s
          = (WStmt.storeLane out (fun l => oix.eval cta j l ir im) r).run
              ((compute.elabAt cta j ir im).run s) := rfl
      -- other buffers are still the initial ones
      have hinv' : ∀ c, c ≠ out →
          (((storeBody out r oix compute).elabAt cta j ir im).run s).mem c = st.mem c := by
        intro c hcne
        rw [hstep, storeLane_otherBuf out _ r _ c hcne, hcm]
        exact hinv c hcne
      have hkeep' : ∀ r' ∈ keep,
          (((storeBody out r oix compute).elabAt cta j ir im).run s).regs r' = st.regs r' := by
        intro r' hr'
        rw [hstep]
        show ((WStmt.storeLane out _ r).run _).regs r' = _
        rw [storeLane_regs, hcompKeep j s r' hr']
        exact hkeep r' hr'
      show (L.foldl _ (((storeBody out r oix compute).elabAt cta j ir im).run s)).mem out A = _
      refine ih _ hinv' hkeep' (fun j' hj' l hl => hall j' (by simp [hj']) l hl) ?_
      by_cases hlater : ∃ j' ∈ L, ∃ l : Lane, oix.eval cta j' l ir im = A
      · exact Or.inl hlater
      -- nothing later writes A, so this iteration must settle it
      refine Or.inr ?_
      by_cases hnow : ∃ l : Lane, oix.eval cta j l ir im = A
      · obtain ⟨l0, hl0⟩ := hnow
        rw [hstep]
        have : ((WStmt.storeLane out (fun l => oix.eval cta j l ir im) r).run
                  ((compute.elabAt cta j ir im).run s)).mem out
                    (oix.eval cta j l0 ir im)
              = ((compute.elabAt cta j ir im).run s).regs r l0 :=
          storeLane_at out (fun l => oix.eval cta j l ir im) r _ l0
            (fun l h => by
            have hb : oix.eval cta j l ir im = oix.eval cta j l0 ir im := h
            rw [hc l, hc l0, hall j (by simp) l (by rw [hb, hl0]),
                hall j (by simp) l0 hl0])
        rw [hl0] at this
        rw [this, hc l0]
        exact hall j (by simp) l0 hl0
      · -- this iteration does not write A either, so the incoming value stands
        rw [hstep, storeLane_otherAddr out (fun l => oix.eval cta j l ir im) r _ A
              (fun l hcon => hnow ⟨l, hcon⟩), hcm]
        rcases hd with ⟨j', hj', l', hl'⟩ | h
        · rcases List.mem_cons.mp hj' with hEq | hIn
          · exact absurd ⟨l', by rw [← hEq]; exact hl'⟩ hnow
          · exact absurd ⟨j', hIn, l', hl'⟩ hlater
        · exact h

/-- One butterfly round under `max`. -/
def warpMaxRoundE (m : Nat) : EWStmt :=
  .seq (.shflXor 1 0 m) (.setR 0 (.maxW (.reg 0) (.reg 1)))

/-- The five-round maximum: after it, `%fw0` holds the warp's maximum in
    **every** lane — the property that let RMSNorm drop its shared memory, and
    that will do the same for softmax. -/
def warpReduceMaxE : EWStmt :=
  .seq (warpMaxRoundE 16) (.seq (warpMaxRoundE 8) (.seq (warpMaxRoundE 4)
    (.seq (warpMaxRoundE 2) (warpMaxRoundE 1))))

/-- Inside the lowerable fragment, so it composes with everything else without a
    new precondition. -/
theorem warpReduceMaxE_expFree : warpReduceMaxE.ExpFree :=
  ⟨⟨trivial, trivial, trivial⟩, ⟨trivial, trivial, trivial⟩,
   ⟨trivial, trivial, trivial⟩, ⟨trivial, trivial, trivial⟩,
   trivial, trivial, trivial⟩

theorem warpReduceMaxE_idxFree : warpReduceMaxE.IdxFree := by decide

/-- **The max reduction realises the same butterfly tree.**

    An instance of `warpReduceOp_spec`, not a second proof — which is the point
    of generalising the network.  After it, `%fw0` holds the warp maximum in
    *every* lane, which is what lets softmax drop the shared memory and the
    barriers the hand-written kernel needed. -/
theorem warpReduceMaxE_spec (cta i : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (st : WSt) :
    (((warpReduceMaxE.elabAt cta i ir im)).run st).regs 0
      = bflyFoldOp (fun a b => NumOps.max a b) (st.regs 0) :=
  warpReduceOp_spec 0 1 (by decide) (.maxW (.reg 0) (.reg 1))
    (fun a b => NumOps.max a b) (fun _ _ => rfl) st

/-- A butterfly touches registers only — memory is untouched, which is what
    lets a reduction sit between two sweeps without disturbing what they read. -/
theorem warpReduceMaxE_mem (cta i : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (st : WSt) :
    ((warpReduceMaxE.elabAt cta i ir im).run st).mem = st.mem := rfl

theorem maxStridedBody_mem (b : Buf) (ix : IdxE) (K cta : Nat) (init : Float32)
    (st : WSt) :
    (((maxStridedBody b ix K init).elabAt cta 0).run st).mem = st.mem := by
  show ((WStmt.forN K (fun j => (stepMaxWS b (fun i l => ix.eval cta i l) j))).run
        ((WStmt.setR 0 (.lit init)).run st)).mem = _
  have key : ∀ (L : List Nat) (s : WSt),
      (L.foldl (fun s' i => (stepMaxWS b (fun i l => ix.eval cta i l) i).run s') s).mem
        = s.mem := by
    intro L
    induction L with
    | nil => intro _; rfl
    | cons i L ih =>
        intro s
        show (L.foldl _ ((stepMaxWS b _ i).run s)).mem = _
        rw [ih _]
        rfl
  exact key (List.range K) _

/-- **The strided maximum**: sweep, max butterfly, lane-0 store — the sum's
    schema at `max`, and the reduction a stable softmax needs. -/
def maxStrided (b : Buf) (ix : IdxE) (out : Buf) (oi : IdxE) (K : Nat)
    (init : Float32) : EWStmt :=
  .seq (.seq (maxStridedBody b ix K init) warpReduceMaxE) (.storeLane0 out oi 0)

theorem maxStrided_spec (b : Buf) (ix : IdxE) (out : Buf) (oi : IdxE)
    (K cta : Nat) (init : Float32) (st : WSt) :
    (((maxStrided b ix out oi K init).elabIn cta).run st).mem out
        (oi.eval cta 0 ⟨0, by decide⟩)
      = bflyFoldOp (fun a c => NumOps.max a c)
          (maxStridedLane (st.mem b) (fun i l => ix.eval cta i l) K init)
          ⟨0, by decide⟩ := by
  show (((warpReduceMaxE.elabAt cta 0 (fun _ _ => 0) (fun _ _ => 0)).run
          (((maxStridedBody b ix K init).elabAt cta 0).run st)).store1 out _
        (((warpReduceMaxE.elabAt cta 0 (fun _ _ => 0) (fun _ _ => 0)).run
          (((maxStridedBody b ix K init).elabAt cta 0).run st)).regs 0
            ⟨0, by decide⟩)).mem out _ = _
  rw [WSt.mem_store1_same, warpReduceMaxE_spec, maxStridedBody_spec]

/-- The sum butterfly is the same instance at `add` — stated so both reductions
    visibly stand on one theorem. -/
theorem warpReduceSumE_spec (cta i : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (st : WSt) :
    ((((EWStmt.seq (warpRoundE 16) (.seq (warpRoundE 8) (.seq (warpRoundE 4)
        (.seq (warpRoundE 2) (warpRoundE 1))))).elabAt cta i ir im)).run st).regs 0
      = bflyFoldOp (fun a b => NumOps.add a b) (st.regs 0) :=
  warpReduceOp_spec 0 1 (by decide) (.add (.reg 0) (.reg 1))
    (fun a b => NumOps.add a b) (fun _ _ => rfl) st

theorem warpReduceSumE_frame (cta i : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (r : Nat) (hr : r ≠ 0) (hr' : r ≠ 1) (st : WSt) :
    ((((EWStmt.seq (warpRoundE 16) (.seq (warpRoundE 8) (.seq (warpRoundE 4)
        (.seq (warpRoundE 2) (warpRoundE 1))))).elabAt cta i ir im)).run st).regs r
      = st.regs r :=
  warpReduceOp_frame 0 1 (.add (.reg 0) (.reg 1)) r hr hr' st

theorem warpReduceSumE_mem (cta i : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (st : WSt) :
    ((((EWStmt.seq (warpRoundE 16) (.seq (warpRoundE 8) (.seq (warpRoundE 4)
        (.seq (warpRoundE 2) (warpRoundE 1))))).elabAt cta i ir im)).run st).mem
      = st.mem := rfl

end AlgorithmLib.ML
