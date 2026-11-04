module
public import AlgorithmLib.ML.Machine.Warp
meta import AlgorithmLib.ML.Machine.Warp
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # What an emitted statement's addresses may read

  The two fragments of `EWStmt` the PTX lowering (`ML.Ptx.Body`) is stated
  over: `IdxFree`, no data-dependent index at all, and `IdxBelow n`, every
  data-dependent index read from an integer register below the emitter's
  allocation watermark `n`.
-/

namespace AlgorithmLib.ML

/-- Addresses only: no data-dependent index anywhere. -/
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
  | .iif _ t e => t.IdxFree ∧ e.IdxFree
  | .iwhile _ _ _ body => body.IdxFree
  | .locals _ body => body.IdxFree
  | _ => True

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
  | .iif _ t e => @instDecidableAnd _ _ (decIdxFree t) (decIdxFree e)
  | .iwhile _ _ _ body => decIdxFree body
  | .locals _ body => decIdxFree body
  | .setI .. | .ldB .. | .stB .. | .ldH .. | .stH .. | .ballot .. | .shflIdx .. => .isTrue trivial

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
  | .iif _ t e => t.IdxBelow n ∧ e.IdxBelow n
  | .iwhile _ _ _ body => body.IdxBelow n
  | .locals _ body => body.IdxBelow n
  | _ => True

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
  | .iif _ t e => @instDecidableAnd _ _ (decIdxBelow n t) (decIdxBelow n e)
  | .iwhile _ _ _ body => decIdxBelow n body
  | .locals _ body => decIdxBelow n body
  | .setI .. | .ldB .. | .stB .. | .ldH .. | .stH .. | .ballot .. | .shflIdx .. => .isTrue trivial

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
  | iif _ t e iht ihe => intro h; exact ⟨iht h.1, ihe h.2⟩
  | iwhile _ _ _ body ih => intro h; exact ih h
  | locals _ body ih => intro h; exact ih h
  | _ => intro _; trivial

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
  | iif _ t e iht ihe => intro hs; exact ⟨iht hs.1, ihe hs.2⟩
  | iwhile _ _ _ body ih => intro hs; exact ih hs
  | locals _ body ih => intro hs; exact ih hs
  | _ => intro _; trivial

end AlgorithmLib.ML
