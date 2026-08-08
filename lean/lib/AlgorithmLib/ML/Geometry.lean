import AlgorithmLib.ML.Kernels

/-!
  # Seam guards: geometry and buffer binding

  Two links between a proven kernel and a running one are settled *outside*
  every theorem.

  **Launch geometry.**  A reduction kernel's theorem is per block: it stores the
  fold over `t ∈ [0, K)`, `l ∈ [0, 32)`.  Whether that covers the intended `n`
  elements — whether `K·32 = n` — and whether the grid has one block per output
  are facts about the *launch*, chosen elsewhere.  Get either wrong and every
  theorem still holds while the kernel reduces the wrong set.

  **Buffer binding.**  `emitProvenKernelN` declares `nbuf` parameters loaded into
  `%rd0 … %rd(nbuf−1)`, and address formation uses `%rd{b}` for buffer `b`.  A
  kernel naming `b ≥ nbuf` emits a reference to an undeclared register.
  `EWStmt.IdxBelow` does not cover this: it bounds integer **register** indices,
  not buffers.

  Neither is a theorem about the machine.  Both are **decidable checks on closed
  terms** — `by decide` at any concrete kernel — and they fail the build rather
  than the run.
-/

namespace AlgorithmLib.ML

-- ---------------------------------------------------------------------------
-- Buffer indices are within the binding table
-- ---------------------------------------------------------------------------

/-- Every buffer an address expression loads from is below `n`. -/
def IdxE.BufBelow (n : Nat) : IdxE → Prop
  | .laneId      => True
  | .loopI       => True
  | .ireg _      => True
  | .ctaId       => True
  | .lit _       => True
  | .ldIdx b off => b < n ∧ off.BufBelow n
  | .add a b     => a.BufBelow n ∧ b.BufBelow n
  | .mul a b     => a.BufBelow n ∧ b.BufBelow n

instance IdxE.decBufBelow (n : Nat) : ∀ e : IdxE, Decidable (e.BufBelow n)
  | .laneId  => .isTrue trivial
  | .loopI   => .isTrue trivial
  | .ireg _  => .isTrue trivial
  | .ctaId   => .isTrue trivial
  | .lit _   => .isTrue trivial
  | .ldIdx b off => @instDecidableAnd _ _ (Nat.decLt b n) (decBufBelow n off)
  | .add a b => @instDecidableAnd _ _ (decBufBelow n a) (decBufBelow n b)
  | .mul a b => @instDecidableAnd _ _ (decBufBelow n a) (decBufBelow n b)

/-- **Every buffer an address expression names**, as a list rather than a
    proposition — so a bound can be *derived* from the buffers an operation
    declares instead of decided at each one. -/
def IdxE.bufsOf : IdxE → List Buf
  | .ldIdx b off => b :: off.bufsOf
  | .add a b     => a.bufsOf ++ b.bufsOf
  | .mul a b     => a.bufsOf ++ b.bufsOf
  | _            => []

/-- **Every buffer the kernel touches is inside the binding table.**

    The check that makes the emitted PTX well-formed: buffer `b` becomes
    `%rd{b}`, and `emitProvenKernelN` declares exactly `%rd0 … %rd(nbuf−1)`. -/
def EWStmt.BufBelow (n : Nat) : EWStmt → Prop
  | .skip                 => True
  | .seq a b              => a.BufBelow n ∧ b.BufBelow n
  | .setR _ _             => True
  | .shflXor _ _ _        => True
  | .barrier              => True
  | .loadIdx _ b ix       => b < n ∧ ix.BufBelow n
  | .loadV4 _ _ _ _ b ix  => b < n ∧ ix.BufBelow n
  | .storeLane0 b ix _    => b < n ∧ ix.BufBelow n
  | .storeLane b ix _     => b < n ∧ ix.BufBelow n
  | .stSm ix _            => ix.BufBelow n
  | .ldSm _ ix            => ix.BufBelow n
  | .forN _ body          => body.BufBelow n
  | .forM bu _ body       => bu < n ∧ body.BufBelow n
  | .cvtIF _ ix           => ix.BufBelow n

/-- Every buffer a kernel names. -/
def EWStmt.bufsOf : EWStmt → List Buf
  | .skip                 => []
  | .seq a b              => a.bufsOf ++ b.bufsOf
  | .setR _ _             => []
  | .shflXor _ _ _        => []
  | .barrier              => []
  | .loadIdx _ b ix       => b :: ix.bufsOf
  | .loadV4 _ _ _ _ b ix  => b :: ix.bufsOf
  | .storeLane0 b ix _    => b :: ix.bufsOf
  | .storeLane b ix _     => b :: ix.bufsOf
  | .stSm ix _            => ix.bufsOf
  | .ldSm _ ix            => ix.bufsOf
  | .forN _ body          => body.bufsOf
  | .forM bu _ body       => bu :: body.bufsOf
  | .cvtIF _ ix           => ix.bufsOf

/-- **A bound on the names is a bound on the statement.**  Proven once, for
    every statement: what a kernel needs checked is which buffers it mentions,
    and `bufsOf` answers that without traversing it again per model. -/
theorem IdxE.bufBelow_of_bufsOf (n : Nat) : ∀ ix : IdxE,
    (∀ b ∈ ix.bufsOf, b < n) → ix.BufBelow n := by
  intro ix
  induction ix with
  | ldIdx b off ih =>
      intro h
      exact ⟨h b (by simp [IdxE.bufsOf]), ih (fun c hc => h c (by simp [IdxE.bufsOf, hc]))⟩
  | add a b iha ihb =>
      intro h
      exact ⟨iha (fun c hc => h c (by simp [IdxE.bufsOf, hc])),
             ihb (fun c hc => h c (by simp [IdxE.bufsOf, hc]))⟩
  | mul a b iha ihb =>
      intro h
      exact ⟨iha (fun c hc => h c (by simp [IdxE.bufsOf, hc])),
             ihb (fun c hc => h c (by simp [IdxE.bufsOf, hc]))⟩
  | _ => intro _; trivial

theorem EWStmt.bufBelow_of_bufsOf (n : Nat) : ∀ s : EWStmt,
    (∀ b ∈ s.bufsOf, b < n) → s.BufBelow n := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro h
      exact ⟨iha (fun c hc => h c (by simp [EWStmt.bufsOf, hc])),
             ihb (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))⟩
  | loadIdx d b ix =>
      intro h
      exact ⟨h b (by simp [EWStmt.bufsOf]),
             IdxE.bufBelow_of_bufsOf n ix (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))⟩
  | loadV4 a b c d bu ix =>
      intro h
      exact ⟨h bu (by simp [EWStmt.bufsOf]),
             IdxE.bufBelow_of_bufsOf n ix (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))⟩
  | storeLane0 b ix r =>
      intro h
      exact ⟨h b (by simp [EWStmt.bufsOf]),
             IdxE.bufBelow_of_bufsOf n ix (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))⟩
  | storeLane b ix r =>
      intro h
      exact ⟨h b (by simp [EWStmt.bufsOf]),
             IdxE.bufBelow_of_bufsOf n ix (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))⟩
  | stSm ix r => intro h; exact IdxE.bufBelow_of_bufsOf n ix h
  | ldSm d ix => intro h; exact IdxE.bufBelow_of_bufsOf n ix h
  | forN k body ih => intro h; exact ih h
  | forM bu a body ih =>
      intro h
      exact ⟨h bu (by simp [EWStmt.bufsOf]), ih (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))⟩
  | cvtIF d ix => intro h; exact IdxE.bufBelow_of_bufsOf n ix h
  | _ => intro _; trivial

instance EWStmt.decBufBelow (n : Nat) : ∀ s : EWStmt, Decidable (s.BufBelow n)
  | .skip => .isTrue trivial
  | .seq a b => @instDecidableAnd _ _ (decBufBelow n a) (decBufBelow n b)
  | .setR _ _ => .isTrue trivial
  | .shflXor _ _ _ => .isTrue trivial
  | .barrier => .isTrue trivial
  | .loadIdx _ b ix => @instDecidableAnd _ _ (Nat.decLt b n) (IdxE.decBufBelow n ix)
  | .loadV4 _ _ _ _ b ix => @instDecidableAnd _ _ (Nat.decLt b n) (IdxE.decBufBelow n ix)
  | .storeLane0 b ix _ => @instDecidableAnd _ _ (Nat.decLt b n) (IdxE.decBufBelow n ix)
  | .storeLane b ix _ => @instDecidableAnd _ _ (Nat.decLt b n) (IdxE.decBufBelow n ix)
  | .stSm ix _ => IdxE.decBufBelow n ix
  | .ldSm _ ix => IdxE.decBufBelow n ix
  | .forN _ body => decBufBelow n body
  | .forM bu _ body => @instDecidableAnd _ _ (Nat.decLt bu n) (decBufBelow n body)
  | .cvtIF _ ix => IdxE.decBufBelow n ix

-- ---------------------------------------------------------------------------
-- Launch geometry
-- ---------------------------------------------------------------------------

/-- **The geometry of a warp reduction**, with the coverage obligation as a
    field rather than a comment.

    `outs` blocks, one warp each, every warp folding `trips · 32` elements.  The
    theorems are per-block and say nothing about either number; this record is
    where they are related, and `covers` is what makes "the kernel reduces `n`
    elements" true rather than hoped. -/
structure ReduceGeom where
  /-- Elements each output reduces over. -/
  n     : Nat
  /-- Outputs, i.e. blocks in the grid. -/
  outs  : Nat
  /-- Loop trips per warp. -/
  trips : Nat
  /-- The 32 lanes × `trips` iterations cover exactly `n` — no element dropped,
      none counted twice. -/
  covers : trips * 32 = n := by decide

/-- **The geometry of an elementwise pass.**  `grid` blocks × 32 lanes ×
    `elems` per lane must be exactly the element count. -/
structure MapGeom where
  /-- Elements to process. -/
  n     : Nat
  /-- Blocks in the grid. -/
  grid  : Nat
  /-- Elements per lane. -/
  elems : Nat
  covers : grid * (32 * elems) = n := by decide

/-- A one-element-per-lane pass — the `mapKernel` shape. -/
def MapGeom.simple (n grid : Nat) (h : grid * 32 = n := by decide) : MapGeom :=
  { n, grid, elems := 1, covers := by simpa using h }

-- ---------------------------------------------------------------------------
-- Kernels that cannot be pipeline stages
-- ---------------------------------------------------------------------------

/-- Does this address expression read buffer `b`? -/
def IdxE.ReadsBufB (b : Buf) : IdxE → Bool
  | .laneId       => false
  | .loopI        => false
  | .ireg _       => false
  | .ctaId        => false
  | .lit _        => false
  | .ldIdx c off  => decide (c = b) || off.ReadsBufB b
  | .add x y      => x.ReadsBufB b || y.ReadsBufB b
  | .mul x y      => x.ReadsBufB b || y.ReadsBufB b

/-- **Does this kernel load from buffer `b`?**

    `StageSpec.valOnly` requires a stage's output to be independent of its own
    output buffer's prior contents, so a kernel that *reads what it writes*
    cannot be a stage.  RoPE is the example in this stack: it rotates a buffer in
    place, and its correctness depends on intra-kernel ordering in a way an
    inter-kernel frame condition cannot express.

    Making the exclusion decidable means such a kernel can be *shown* not to be a
    stage, and an attempt to use one fails with a clear reason. -/
def EWStmt.ReadsBufB (b : Buf) : EWStmt → Bool
  | .skip                 => false
  | .seq x y              => x.ReadsBufB b || y.ReadsBufB b
  | .setR _ _             => false
  | .shflXor _ _ _        => false
  | .barrier              => false
  | .loadIdx _ c ix       => decide (c = b) || ix.ReadsBufB b
  | .loadV4 _ _ _ _ c ix  => decide (c = b) || ix.ReadsBufB b
  | .storeLane0 _ ix _    => ix.ReadsBufB b
  | .storeLane _ ix _     => ix.ReadsBufB b
  | .stSm ix _            => ix.ReadsBufB b
  | .ldSm _ ix            => ix.ReadsBufB b
  | .forN _ body          => body.ReadsBufB b
  | .forM bu _ body       => decide (bu = b) || body.ReadsBufB b
  | .cvtIF _ ix           => ix.ReadsBufB b

/-- Does this kernel *store* to buffer `b`? -/
def EWStmt.WritesBufB (b : Buf) : EWStmt → Bool
  | .skip                 => false
  | .seq x y              => x.WritesBufB b || y.WritesBufB b
  | .setR _ _             => false
  | .shflXor _ _ _        => false
  | .barrier              => false
  | .loadIdx _ _ _        => false
  | .loadV4 _ _ _ _ _ _   => false
  | .storeLane0 c _ _     => decide (c = b)
  | .storeLane c _ _      => decide (c = b)
  | .stSm _ _             => false
  | .ldSm _ _             => false
  | .forN _ body          => body.WritesBufB b
  | .forM _ _ body        => body.WritesBufB b
  | .cvtIF _ _            => false

/-- **The condition a stage must satisfy**: it writes `out`, and does not read
    it.

    A stage must at minimum write the buffer it claims as its output; that is
    what this checks.  It is *necessary*, not sufficient — `StageSpec.valOnly`
    is the semantic obligation and is discharged per stage.  For the stronger,
    fully syntactic condition see `IdempotentEligibleB` below. -/
def EWStmt.StageEligibleB (out : Buf) (s : EWStmt) : Bool :=
  s.WritesBufB out

/-- **Syntactically idempotent**: writes `out` and never reads it.

    This is the precondition for `StageSpec.Idempotent`, which is what
    re-running a block requires — and therefore what the arbitrary-list and
    permutation results need.  A grid runs each block once and does *not* need
    it, so an in-place kernel is still a stage; `runGrid_value` covers it.

    Both halves matter.  Checking only "does not read `out`" passes vacuously
    for any buffer the kernel never touches, so a guard written against the
    wrong buffer number would report success.  Requiring `WritesBufB` turns a
    wrong buffer number into a build failure. -/
def EWStmt.IdempotentEligibleB (out : Buf) (s : EWStmt) : Bool :=
  s.WritesBufB out && !(s.ReadsBufB out)

end AlgorithmLib.ML
