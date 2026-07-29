import AlgorithmLib.ML.Warp
import AlgorithmLib.PTX

/-!
  # Lowering a proven warp kernel to PTX

  `Warp.lean` proves kernels against a 32-lane machine model.  Nothing there
  emits anything — which is why, until now, the ML stack produced theorems and
  no executable code.

  This file defines `EWStmt`, the *emittable* mirror of `WStmt`: loops carry a
  syntactic body and addresses are `IdxE` terms rather than Lean closures, so a
  kernel written here can both be proven (via `elabAt`, which expands it into
  the proven machine language) and compiled.

  The lowering to PTX is not in this file: see `PtxM`/`PtxFlat`/`PtxPrint`.
-/

namespace AlgorithmLib.ML
open AlgorithmLib.PTX

-- ---------------------------------------------------------------------------
-- An emittable mirror of `WStmt`
-- ---------------------------------------------------------------------------

/-! `WStmt.forN` takes a Lean function `Nat → WStmt`, and `loadIdx` a function
    `Lane → Nat`.  Both are convenient for proving and impossible to emit — PTX
    cannot evaluate a Lean closure.

    `EWStmt` is the emittable mirror: loops carry a *syntactic* body and
    addresses are `IdxE` terms.  `elab` expands it into the proven `WStmt`, so
    a kernel is written once in `EWStmt`, proven via `elab`, and emitted
    directly.  Nothing already proven changes. -/
inductive EWStmt where
  | skip       : EWStmt
  | seq        : EWStmt → EWStmt → EWStmt
  | setR       : Nat → WFExp → EWStmt
  | shflXor    : Nat → Nat → Nat → EWStmt
  | loadIdx    : Nat → Buf → IdxE → EWStmt
  | loadV4     : Nat → Nat → Nat → Nat → Buf → IdxE → EWStmt
  | storeLane0 : Buf → IdxE → Nat → EWStmt
  | storeLane  : Buf → IdxE → Nat → EWStmt
  /-- `st.shared.f32` — stage a register into on-chip memory.  With `barrier`,
      this is what makes a *block* kernel writable in the emittable language
      rather than only in the machine language. -/
  | stSm       : IdxE → Nat → EWStmt
  /-- `ld.shared.f32` — read a slot back, possibly one another warp wrote. -/
  | ldSm       : Nat → IdxE → EWStmt
  /-- `bar.sync` — a no-op within a warp, and the synchronisation point a block
      reduction is proven against (`Block.blockStore_perm`). -/
  | barrier    : EWStmt
  | forN       : Nat → EWStmt → EWStmt
  /-- A loop whose trip count is read from integer memory at a fixed address —
      lane-uniform by construction.  This is what a sequence-length-dependent
      kernel needs, and `forN` covers every static shape. -/
  | forM       : Buf → Nat → EWStmt → EWStmt
  /-- `cvt.rn.f32.u32` — put an *index* where a float expression can see it.
      The only bridge between `IdxE` and `WFExp`, and the reason an argmax is
      expressible: its arithmetic mixes an element's value with the element's
      own index. -/
  | cvtIF      : Nat → IdxE → EWStmt

/-- Expand an emittable statement into the proven machine language, given the
    enclosing loop counter. -/
def EWStmt.elabAt (cta i : Nat) (ir : Nat → Lane → Nat := fun _ _ => 0)
    (im : Buf → Nat → Nat := fun _ _ => 0) : EWStmt → WStmt
  | .skip                 => .skip
  | .seq a b              => .seq (a.elabAt cta i ir im) (b.elabAt cta i ir im)
  | .setR r e             => .setR r e
  | .shflXor d s m        => .shflXor d s m
  | .loadIdx d b ix       => .loadIdx d b (fun l => ix.eval cta i l ir im)
  | .loadV4 a b c d bu ix => .loadV4 a b c d bu (fun l => ix.eval cta i l ir im)
  | .storeLane0 b ix r    => .storeLane0 b (ix.eval cta i ⟨0, by decide⟩ ir im) r
  | .storeLane b ix r     => .storeLane b (fun l => ix.eval cta i l ir im) r
  | .stSm ix r            => .stSmem (fun l => ix.eval cta i l ir im) r
  | .ldSm d ix            => .ldSmem d (fun l => ix.eval cta i l ir im)
  | .barrier              => .barrier
  | .forN n body          => .forN n (fun j => body.elabAt cta j ir im)
  | .forM bu a body       => .forN (im bu a) (fun j => body.elabAt cta j ir im)
  | .cvtIF d ix           => .setLaneF d (fun l => ix.eval cta i l ir im)

/-- Top-level expansion for block `cta`.  The machine model describes **one
    warp in one block**, so the block id is a parameter of elaboration and the
    proof holds per block. -/
def EWStmt.elabIn (cta : Nat) (s : EWStmt) : WStmt := s.elabAt cta 0

/-- Single-block expansion. -/
def EWStmt.expand (s : EWStmt) : WStmt := s.elabAt 0 0

/-! ## Lowering

    An `EWStmt` reaches PTX through three proven passes:

    * `PtxM.lean`     — `emitEW`, proven against the warp machine
    * `PtxFlat.lean`  — `flatEW`, proven against a program-counter machine with
                        real branches
    * `PtxPrint.lean` — `emitProvenKernel`, the text encoding

    The three shipped kernels are emitted by `emitProvenKernel`. -/

/-- Addresses only: no data-dependent index anywhere.  `ExpFree` implies this;
    it is separate so that `expandEW`, which rewrites `exp` but never touches an
    address, can state exactly what it preserves. -/
def EWStmt.IdxFree : EWStmt → Prop
  | .skip => True
  | .seq a b => a.IdxFree ∧ b.IdxFree
  | .setR _ _ => True
  | .shflXor _ _ _ => True
  | .loadIdx _ _ ix => ix.IregFree
  | .loadV4 _ _ _ _ _ ix => ix.IregFree
  | .storeLane0 _ ix _ => ix.IregFree
  | .storeLane _ ix _ => ix.IregFree
  | .stSm ix _ => ix.IregFree
  | .ldSm _ ix => ix.IregFree
  | .barrier => True
  | .forN _ body => body.IdxFree
  | .forM _ _ _ => False
  | .cvtIF _ ix => ix.IregFree

instance EWStmt.decIdxFree : ∀ s : EWStmt, Decidable s.IdxFree
  | .skip => .isTrue trivial
  | .seq a b => @instDecidableAnd _ _ (decIdxFree a) (decIdxFree b)
  | .setR _ _ => .isTrue trivial
  | .shflXor _ _ _ => .isTrue trivial
  | .loadIdx _ _ ix => IdxE.decIregFree ix
  | .loadV4 _ _ _ _ _ ix => IdxE.decIregFree ix
  | .storeLane0 _ ix _ => IdxE.decIregFree ix
  | .storeLane _ ix _ => IdxE.decIregFree ix
  | .stSm ix _ => IdxE.decIregFree ix
  | .ldSm _ ix => IdxE.decIregFree ix
  | .barrier => .isTrue trivial
  | .forN _ body => decIdxFree body
  | .forM _ _ _ => .isFalse (fun h => h)
  | .cvtIF _ ix => IdxE.decIregFree ix

/-- **On the address-free fragment the register file is irrelevant.**

    This is what keeps the existing lowering theorems literally unchanged after
    `elabAt` gained an integer register file: every kernel written before
    `IdxE.ireg` is `IdxFree`, so its elaboration does not depend on `ir` at all.
    It is also the bridge the `ireg` lowering will need, since the interesting
    case is exactly where this lemma stops applying. -/
theorem EWStmt.elabAt_idxFree (cta : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) :
    ∀ (s : EWStmt) (i : Nat), s.IdxFree → s.elabAt cta i ir im = s.elabAt cta i := by
  intro s
  induction s with
  | skip => intro _ _; rfl
  | seq a b iha ihb =>
      intro i h; show WStmt.seq _ _ = WStmt.seq _ _; rw [iha i h.1, ihb i h.2]
  | setR _ _ => intro _ _; rfl
  | shflXor _ _ _ => intro _ _; rfl
  | loadIdx d b ix =>
      intro i h
      show WStmt.loadIdx _ _ _ = WStmt.loadIdx _ _ _
      exact congrArg _ (funext fun l => IdxE.eval_iregFree cta i l ir im ix h)
  | loadV4 a b c d bu ix =>
      intro i h
      show WStmt.loadV4 _ _ _ _ _ _ = WStmt.loadV4 _ _ _ _ _ _
      exact congrArg _ (funext fun l => IdxE.eval_iregFree cta i l ir im ix h)
  | storeLane0 b ix r =>
      intro i h
      show WStmt.storeLane0 _ _ _ = WStmt.storeLane0 _ _ _
      rw [IdxE.eval_iregFree cta i ⟨0, by decide⟩ ir im ix h]
  | storeLane b ix r =>
      intro i h
      show WStmt.storeLane _ _ _ = WStmt.storeLane _ _ _
      exact congrArg (fun f => WStmt.storeLane b f r)
        (funext fun l => IdxE.eval_iregFree cta i l ir im ix h)
  | stSm ix r =>
      intro i h
      show WStmt.stSmem _ _ = WStmt.stSmem _ _
      exact congrArg (fun f => WStmt.stSmem f r)
        (funext fun l => IdxE.eval_iregFree cta i l ir im ix h)
  | ldSm d ix =>
      intro i h
      show WStmt.ldSmem _ _ = WStmt.ldSmem _ _
      exact congrArg _ (funext fun l => IdxE.eval_iregFree cta i l ir im ix h)
  | barrier => intro _ _; rfl
  | forN n body ih =>
      intro i h
      show WStmt.forN _ _ = WStmt.forN _ _
      exact congrArg _ (funext fun j => ih j h)
  | forM _ _ _ _ => intro _ h; exact (h : False).elim
  | cvtIF d ix =>
      intro i h
      show WStmt.setLaneF _ _ = WStmt.setLaneF _ _
      exact congrArg _ (funext fun l => IdxE.eval_iregFree cta i l ir im ix h)

/-- Every data-dependent index the statement mentions was loaded into a register
    below `n` — the emitter's allocation watermark.  This is what makes an
    `ireg` address *survive* the address code emitted around it. -/
def EWStmt.IdxBelow (n : Nat) : EWStmt → Prop
  | .skip => True
  | .seq a b => a.IdxBelow n ∧ b.IdxBelow n
  | .setR _ _ => True
  | .shflXor _ _ _ => True
  | .loadIdx _ _ ix => ix.regsBelow n
  | .loadV4 _ _ _ _ _ ix => ix.regsBelow n
  | .storeLane0 _ ix _ => ix.regsBelow n
  | .storeLane _ ix _ => ix.regsBelow n
  | .stSm ix _ => ix.regsBelow n
  | .ldSm _ ix => ix.regsBelow n
  | .barrier => True
  | .forN _ body => body.IdxBelow n
  | .forM _ _ body => body.IdxBelow n
  | .cvtIF _ ix => ix.regsBelow n

instance EWStmt.decIdxBelow (n : Nat) : ∀ s : EWStmt, Decidable (s.IdxBelow n)
  | .skip => .isTrue trivial
  | .seq a b => @instDecidableAnd _ _ (decIdxBelow n a) (decIdxBelow n b)
  | .setR _ _ => .isTrue trivial
  | .shflXor _ _ _ => .isTrue trivial
  | .loadIdx _ _ ix => IdxE.decRegsBelow n ix
  | .loadV4 _ _ _ _ _ ix => IdxE.decRegsBelow n ix
  | .storeLane0 _ ix _ => IdxE.decRegsBelow n ix
  | .storeLane _ ix _ => IdxE.decRegsBelow n ix
  | .stSm ix _ => IdxE.decRegsBelow n ix
  | .ldSm _ ix => IdxE.decRegsBelow n ix
  | .barrier => .isTrue trivial
  | .forN _ body => decIdxBelow n body
  | .forM _ _ body => decIdxBelow n body
  | .cvtIF _ ix => IdxE.decRegsBelow n ix

/-- Statements the **branch** lowering covers.

    Both loop forms are covered.  `forM`'s memory-bounded shape is seven
    instructions rather than five — `mov` the address, `ld.global.u32` the
    bound, then the familiar init/test/exit/body/increment/back-edge — and
    `flatEW_sound` proves that shape at the shifted offsets.  The predicate
    is a `Decidable` gate on the *leaves* being emittable, not a restriction
    on loops. -/
def EWStmt.Flat : EWStmt → Prop
  | .skip => True
  | .seq a b => a.Flat ∧ b.Flat
  | .setR _ _ => True
  | .shflXor _ _ _ => True
  | .loadIdx _ _ _ => True
  | .loadV4 _ _ _ _ _ _ => True
  | .storeLane0 _ _ _ => True
  | .storeLane _ _ _ => True
  | .stSm _ _ => True
  | .ldSm _ _ => True
  | .barrier => True
  | .forN _ body => body.Flat
  | .forM _ _ body => body.Flat
  | .cvtIF _ _ => True

instance EWStmt.decFlat : ∀ s : EWStmt, Decidable s.Flat
  | .skip => .isTrue trivial
  | .seq a b => @instDecidableAnd _ _ (decFlat a) (decFlat b)
  | .setR _ _ => .isTrue trivial
  | .shflXor _ _ _ => .isTrue trivial
  | .loadIdx _ _ _ => .isTrue trivial
  | .loadV4 _ _ _ _ _ _ => .isTrue trivial
  | .storeLane0 _ _ _ => .isTrue trivial
  | .storeLane _ _ _ => .isTrue trivial
  | .stSm _ _ => .isTrue trivial
  | .ldSm _ _ => .isTrue trivial
  | .barrier => .isTrue trivial
  | .forN _ body => decFlat body
  | .forM _ _ body => decFlat body
  | .cvtIF _ _ => .isTrue trivial

theorem IdxE.regsBelow_of_iregFree (n : Nat) : ∀ e : IdxE, e.IregFree → e.regsBelow n := by
  intro e
  induction e with
  | laneId => intro _; trivial
  | loopI  => intro _; trivial
  | ctaId  => intro _; trivial
  | lit _  => intro _; trivial
  | ireg r => intro h; exact h.elim
  | ldIdx _ _ _ => intro h; exact h.elim
  | add a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩
  | mul a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩

theorem EWStmt.idxBelow_of_idxFree (n : Nat) : ∀ s : EWStmt, s.IdxFree → s.IdxBelow n := by
  intro s
  induction s with
  | skip => intro _; trivial
  | seq a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩
  | setR _ _ => intro _; trivial
  | shflXor _ _ _ => intro _; trivial
  | loadIdx _ _ ix => intro h; exact IdxE.regsBelow_of_iregFree n ix h
  | loadV4 _ _ _ _ _ ix => intro h; exact IdxE.regsBelow_of_iregFree n ix h
  | storeLane0 _ ix _ => intro h; exact IdxE.regsBelow_of_iregFree n ix h
  | storeLane _ ix _ => intro h; exact IdxE.regsBelow_of_iregFree n ix h
  | stSm ix _ => intro h; exact IdxE.regsBelow_of_iregFree n ix h
  | ldSm _ ix => intro h; exact IdxE.regsBelow_of_iregFree n ix h
  | barrier => intro _; trivial
  | forN _ body ih => intro h; exact ih h
  | forM _ _ _ _ => intro h; exact (h : False).elim
  | cvtIF _ ix => intro h; exact IdxE.regsBelow_of_iregFree n ix h

theorem EWStmt.idxBelow_mono {n n' : Nat} (h : n ≤ n') :
    ∀ s : EWStmt, s.IdxBelow n → s.IdxBelow n' := by
  intro s
  induction s with
  | skip => intro _; trivial
  | seq a b iha ihb => intro hs; exact ⟨iha hs.1, ihb hs.2⟩
  | setR _ _ => intro _; trivial
  | shflXor _ _ _ => intro _; trivial
  | loadIdx _ _ ix => intro hs; exact IdxE.regsBelow_mono h ix hs
  | loadV4 _ _ _ _ _ ix => intro hs; exact IdxE.regsBelow_mono h ix hs
  | storeLane0 _ ix _ => intro hs; exact IdxE.regsBelow_mono h ix hs
  | storeLane _ ix _ => intro hs; exact IdxE.regsBelow_mono h ix hs
  | stSm ix _ => intro hs; exact IdxE.regsBelow_mono h ix hs
  | ldSm _ ix => intro hs; exact IdxE.regsBelow_mono h ix hs
  | barrier => intro _; trivial
  | forN _ body ih => intro hs; exact ih hs
  | forM _ _ body ih => intro hs; exact ih hs
  | cvtIF _ ix => intro hs; exact IdxE.regsBelow_mono h ix hs

/-- **Elaboration only reads index registers below the watermark.**  The spec-side
    frame property, and the reason a `seq` can be elaborated once against the
    *initial* register file even though the first half may write registers. -/
theorem EWStmt.elabAt_frame (cta n : Nat) (ir ir' : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (hir : ∀ x, x < n → ir' x = ir x) :
    ∀ (s : EWStmt) (i : Nat), s.IdxBelow n → s.elabAt cta i ir' im = s.elabAt cta i ir im := by
  intro s
  induction s with
  | skip => intro _ _; rfl
  | seq a b iha ihb =>
      intro i h; show WStmt.seq _ _ = WStmt.seq _ _; rw [iha i h.1, ihb i h.2]
  | setR _ _ => intro _ _; rfl
  | shflXor _ _ _ => intro _ _; rfl
  | loadIdx d b ix =>
      intro i h
      show WStmt.loadIdx _ _ _ = WStmt.loadIdx _ _ _
      exact congrArg _ (funext fun l => IdxE.eval_frame cta i l n ir ir' im hir ix h)
  | loadV4 a b c d bu ix =>
      intro i h
      show WStmt.loadV4 _ _ _ _ _ _ = WStmt.loadV4 _ _ _ _ _ _
      exact congrArg _ (funext fun l => IdxE.eval_frame cta i l n ir ir' im hir ix h)
  | storeLane0 b ix r =>
      intro i h
      show WStmt.storeLane0 _ _ _ = WStmt.storeLane0 _ _ _
      rw [IdxE.eval_frame cta i ⟨0, by decide⟩ n ir ir' im hir ix h]
  | storeLane b ix r =>
      intro i h
      show WStmt.storeLane _ _ _ = WStmt.storeLane _ _ _
      exact congrArg (fun f => WStmt.storeLane b f r)
        (funext fun l => IdxE.eval_frame cta i l n ir ir' im hir ix h)
  | stSm ix r =>
      intro i h
      show WStmt.stSmem _ _ = WStmt.stSmem _ _
      exact congrArg (fun f => WStmt.stSmem f r)
        (funext fun l => IdxE.eval_frame cta i l n ir ir' im hir ix h)
  | ldSm d ix =>
      intro i h
      show WStmt.ldSmem _ _ = WStmt.ldSmem _ _
      exact congrArg _ (funext fun l => IdxE.eval_frame cta i l n ir ir' im hir ix h)
  | barrier => intro _ _; rfl
  | forN n body ih =>
      intro i h
      show WStmt.forN _ _ = WStmt.forN _ _
      exact congrArg _ (funext fun j => ih j h)
  | forM bu a body ih =>
      intro i h
      show WStmt.forN _ _ = WStmt.forN _ _
      exact congrArg _ (funext fun j => ih j h)
  | cvtIF d ix =>
      intro i h
      show WStmt.setLaneF _ _ = WStmt.setLaneF _ _
      exact congrArg _ (funext fun l => IdxE.eval_frame cta i l n ir ir' im hir ix h)

end AlgorithmLib.ML
