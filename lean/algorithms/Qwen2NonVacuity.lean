import Qwen2Common

/-!
  # The inference claims, at concrete values

  `0 sorry` does not imply non-vacuous, and neither does a clean axiom scan: a
  theorem `∀ h : P, Q` is true for free when `P` is unsatisfiable, and no
  closure walk over axioms and opaques can see it.  This project has been bitten
  by exactly that before — a headline chaining theorem whose hypothesis nothing
  could satisfy.

  Thirteen of the seventeen public inference claims rest on `SmMeta`, an
  obligation about the meta buffer.  If `SmMeta` were unsatisfiable all thirteen
  would be vacuously true.  `smMetaW` below exhibits a witness, and the
  instantiations after it apply the claims at that witness — so each is known to
  have content, not merely to typecheck.

  What is deliberately *not* discharged here is `AllHold [Law.combinerComm]`:
  that is a float law, on the agreed trust surface, and the Law mechanism exists
  precisely so it stays visible in the type rather than being proven away.
-/

open AlgorithmLib AlgorithmLib.ML Qwen2Proven.Stage Qwen2Common

set_option maxRecDepth 8000

namespace Qwen2NonVacuity

/-- A meta buffer for `seqLen = 64`: `CHUNKS = 2`, `TAIL = 64`, `REM = 0`. -/
def gimW : Buf → Nat → Nat := fun b a =>
  if b = B_META then
    (if a = Qwen2Proven.SEQ_SLOT then 64
     else if a = Qwen2Proven.CHUNKS_SLOT then 2
     else if a = Qwen2Proven.TAIL_SLOT then 64
     else if a = Qwen2Proven.REM_SLOT then 0 else 0)
  else 0

/-- The softmax bind really does name the meta buffer — so `gimW` is describing
    the slot the kernel reads, not some other one. -/
example : bSoft 1 = B_META := by decide

/-- **`SmMeta` is satisfiable.**  Without this, every claim carrying it could be
    true for the wrong reason. -/
theorem smMetaW : SmMeta (fun b => gimW (bSoft b)) where
  tail := by decide
  row  := by decide
  pos  := by decide

/-- …and it is not satisfiable *by accident*: a meta buffer whose tail
    disagrees with `CHUNKS·32` fails, which is the configuration the kernel's
    two store passes silently mis-handle. -/
def gimBad : Buf → Nat → Nat := fun b a =>
  if b = B_META then
    (if a = Qwen2Proven.SEQ_SLOT then 64
     else if a = Qwen2Proven.CHUNKS_SLOT then 2
     else if a = Qwen2Proven.TAIL_SLOT then 63
     else if a = Qwen2Proven.REM_SLOT then 0 else 0)
  else 0

example : ¬ (gimBad (bSoft 1) Qwen2Proven.TAIL_SLOT = gimBad (bSoft 1) Qwen2Proven.CHUNKS_SLOT * 32) := by decide

/-- **`SmMeta` follows from the four formulas the host is meant to write.**

    This shrinks the obligation.  `SmMeta` states three relations with no
    account of *why* they should hold; what the host actually intends is to
    publish `seqLen` and its decomposition into 32-wide chunks.  Given that
    intent — slots holding `seq`, `seq / 32`, `(seq / 32) * 32`, `seq % 32` —
    all three relations are arithmetic, and the only genuine precondition left
    is `0 < seq`, i.e. at least one token.

    So what remains undischarged is no longer "three unexplained invariants"
    but "the host's meta-writing code computes these four expressions", which is
    a statement about ~4 CLIF instructions and can be checked against the
    generator. -/
theorem smMeta_of_seqLen (im : Buf → Nat → Nat) (seq : Nat) (hpos : 0 < seq)
    (hs : im 1 Qwen2Proven.SEQ_SLOT = seq)
    (hc : im 1 Qwen2Proven.CHUNKS_SLOT = seq / 32)
    (ht : im 1 Qwen2Proven.TAIL_SLOT = (seq / 32) * 32)
    (hr : im 1 Qwen2Proven.REM_SLOT = seq % 32) : SmMeta im where
  tail := by rw [ht, hc]
  row  := by rw [hc, hr, hs]; omega
  pos  := by rw [hs]; exact hpos

-- ---------------------------------------------------------------------------
-- `SmMeta`, discharged down to one named seam
-- ---------------------------------------------------------------------------

open AlgorithmLib.Clif in
/-- **What Cranelift and the upload are trusted to do**, in one definition.

    The meta buffer the softmax kernel reads holds, at each slot, the *value of
    the expression the host stored there*.  `rho` is the valuation of the
    runtime roots — what `pos` actually was at this decode step.

    Two already-agreed seams, stated together because together they are exactly
    what stands between `Clif`'s store map and the kernel's `im`:

    * Cranelift's `ushr`/`ishl`/`isub`/`store` mean what `DExp.eval` says they
      mean.  That is seam 14, and this is the *only* place this development
      says what those four instructions compute.
    * `cl_cuda_upload_ptr` delivers the twenty-four bytes unchanged — the
      `uploadedValue` opaque, already on the surface.

    Everything else about `SmMeta` is now a theorem. -/
def MetaFaithful (im : Buf → Nat → Nat) (mm : StoreMap) (root : Nat)
    (base : Int) (rho : Nat → Int) : Prop :=
  ∀ (slot : Nat) (v : SymVal) (d : DExp),
    mm.get? root (base + 4 * (slot : Int)) = some v → v.toD? = some d →
    ((im 1 slot : Nat) : Int) = d.eval rho

open AlgorithmLib.Clif in
/-- **`SmMeta` holds of any store map with the fragment's four expressions in
    it**, given that the buffer faithfully holds their values.

    Read what is left: `hf` (the two trusted seams) and `hpos` (at least one
    token). The three relations themselves are arithmetic — and *linear*
    arithmetic at that, once `seq / 32` is named, which is why `omega` closes
    them.  Nothing here assumes anything about the meta buffer that the host
    was not proven to write. -/
theorem smMeta_of_stores
    (im : Buf → Nat → Nat) (mm : StoreMap) (root : Nat) (rho : Nat → Int)
    (S : DExp) (vSeq : SymVal)
    (hf : MetaFaithful im mm root (Qwen2Common.META_STAGE_OFF : Int) rho)
    (h2 : mm.get? root ((Qwen2Common.META_STAGE_OFF : Int) + 8) = some vSeq)
    (hv2 : vSeq.toD? = some S)
    (h3 : mm.get? root ((Qwen2Common.META_STAGE_OFF : Int) + 12)
            = some (.derived (.shr S 5)))
    (h4 : mm.get? root ((Qwen2Common.META_STAGE_OFF : Int) + 16)
            = some (.derived (.shl (.shr S 5) 5)))
    (h5 : mm.get? root ((Qwen2Common.META_STAGE_OFF : Int) + 20)
            = some (.derived (.sub S (.shl (.shr S 5) 5))))
    (hpos : 0 < S.eval rho) : SmMeta im := by
  have h32 : (2 : Int) ^ 5 = 32 := by decide
  have e2 := hf Qwen2Proven.SEQ_SLOT vSeq S (by simpa using h2) hv2
  have e3 := hf Qwen2Proven.CHUNKS_SLOT _ _ (by simpa using h3) rfl
  have e4 := hf Qwen2Proven.TAIL_SLOT _ _ (by simpa using h4) rfl
  have e5 := hf Qwen2Proven.REM_SLOT _ _ (by simpa using h5) rfl
  simp only [DExp.eval, h32] at e2 e3 e4 e5
  -- name `seq` and `seq / 32`, after which every relation is linear
  obtain ⟨sq, hsq⟩ : ∃ x, DExp.eval rho S = x := ⟨_, rfl⟩
  obtain ⟨ch, hch⟩ : ∃ x, sq / 32 = x := ⟨_, rfl⟩
  rw [hsq] at e2 hpos
  rw [hsq, hch] at e3 e4 e5
  exact { tail := by omega, row := by omega, pos := by omega }

open AlgorithmLib.Clif in
/-- **`SmMeta`, for the fragment the generator actually emits.**

    The chain, end to end:

    * `Qwen2Common.metaStageFrag_emits` — these twenty-five instructions are
      what the generator emits, from any builder state;
    * `Qwen2Common.metaFrag_stores` — after them the store map holds `seq`,
      `seq >>> 5`, `(seq >>> 5) <<< 5` and `seq − ((seq >>> 5) <<< 5)` at the
      four slots the softmax kernel reads;
    * `smMeta_of_stores` — those four relations give `SmMeta`.

    What is assumed: `MetaFaithful` (Cranelift's four instructions and the
    upload, both already trust-surface), `0 < seq` (at least one token), and the
    compiler's own allocation convention.  `SmMeta` itself is no longer
    assumed. -/
theorem smMeta_of_frag
    (ptr dataPtr pos32 seqLen64 : AlgorithmLib.IR.Val)
    (e : Env) (m : StoreMap) (n : Nat) (S : DExp)
    (hptr : ptr.id < n) (hdp : dataPtr.id < n) (hpo : pos32.id < n)
    (hseq : seqLen64.id < n) (he : e ptr = SymVal.unknown)
    (hS : (e seqLen64).toD? = some S) (hnc : ∀ k : Int, e seqLen64 ≠ SymVal.const k)
    (im : Buf → Nat → Nat) (rho : Nat → Int)
    (hf : MetaFaithful im
            (bevalPure ⟨e, m⟩ (Qwen2Common.metaFragInsts ptr dataPtr pos32 seqLen64 n)).mem
            ptr.id (Qwen2Common.META_STAGE_OFF : Int) rho)
    (hpos : 0 < S.eval rho) : SmMeta im := by
  obtain ⟨q2, q3, q4, q5⟩ :=
    Qwen2Common.metaFrag_stores ptr dataPtr pos32 seqLen64 e m n S hptr hdp hpo hseq he hS hnc
  exact smMeta_of_stores im _ ptr.id rho S (e seqLen64) hf q2 hS q3 q4 q5 hpos

/-- The witness above is an instance of it, at `seq = 64`. -/
example : SmMeta (fun b => gimW (bSoft b)) :=
  smMeta_of_seqLen _ 64 (by decide) (by decide) (by decide) (by decide) (by decide)

-- ---------------------------------------------------------------------------
-- The claims, applied
-- ---------------------------------------------------------------------------

/-- The layer's twenty-two device writes resolve to `layerPlan` — at a concrete
    index map with a *proven* `SmMeta`, so the hypotheses are jointly
    satisfiable and this is not an empty implication. -/
theorem layer_realises_concrete (h : AllHold [Law.combinerComm]) :
    planOf? (layerKernels gimW h smMetaW) layerDeclared (attnOps ++ ffnOps)
      = some (layerPlan gimW h smMetaW) :=
  layer_ops_realise_plan gimW h smMetaW

/-- The gap is two, concretely. -/
theorem lawGap_concrete (h : AllHold [Law.combinerComm]) :
    (layerPlan gimW h smMetaW).declaredLawGap = 2 :=
  layer_declaredLawGap gimW h smMetaW

/-- A whole token's device writes resolve to `tokenPlan`. -/
theorem token_realises_concrete (h : AllHold [Law.combinerComm]) :
    planOf? (tokenKernels gimW h smMetaW) tokenDeclared tokenOps
      = some (tokenPlan gimW h smMetaW) :=
  token_ops_realise_plan gimW h smMetaW

/-- …and running it computes its denotation, with `Honours` discharged by
    `idealR_honours` rather than assumed. -/
theorem token_computes_concrete (h : AllHold [Law.combinerComm]) (st : WSt) :
    ((tokenPlan gimW h smMetaW).run idealR st).mem
      = (tokenPlan gimW h smMetaW).denote st.mem :=
  token_computes gimW h smMetaW idealR idealR_honours st

-- ── The value readouts, at concrete indices ────────────────────────────────

/-- At the first row. -/
theorem ffn_down_at_0 (hl : CuBlasIsMatvec) (m : Buf → Nat → Float32) :
    ffnPlan.denote m B_AO 0
      = (List.finRange D_FF).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (m B_WD (0 * D_FF + j.val)) (ffnMem3 m B_ACT j.val)))
          (NumOps.ofNat 0) :=
  ffn_down_is_matvec hl m 0 (by decide)

/-- …and at the last, where an off-by-one would hide. -/
theorem ffn_down_at_last (hl : CuBlasIsMatvec) (m : Buf → Nat → Float32) :
    ffnPlan.denote m B_AO (D - 1)
      = (List.finRange D_FF).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (m B_WD ((D - 1) * D_FF + j.val)) (ffnMem3 m B_ACT j.val)))
          (NumOps.ofNat 0) :=
  ffn_down_is_matvec hl m (D - 1) (by decide)

/-- The output projection, likewise, at both ends. -/
theorem attn_o_at_0 (hl : CuBlasIsMatvec) (h : AllHold [Law.combinerComm])
    (m : Buf → Nat → Float32) :
    (attnPlan gimW h smMetaW).denote m B_XN 0
      = (List.finRange D).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (m B_WO (0 * D + j.val))
              (attnMem gimW h smMetaW 14 m B_AO j.val)))
          (NumOps.ofNat 0) :=
  attn_o_is_matvec gimW hl h smMetaW m 0 (by decide)

theorem attn_o_at_last (hl : CuBlasIsMatvec) (h : AllHold [Law.combinerComm])
    (m : Buf → Nat → Float32) :
    (attnPlan gimW h smMetaW).denote m B_XN (D - 1)
      = (List.finRange D).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (m B_WO ((D - 1) * D + j.val))
              (attnMem gimW h smMetaW 14 m B_AO j.val)))
          (NumOps.ofNat 0) :=
  attn_o_is_matvec gimW hl h smMetaW m (D - 1) (by decide)

end Qwen2NonVacuity
