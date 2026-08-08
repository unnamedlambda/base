import Qwen2Common

/-!
  # The shipped FFN, against the model definition

  The chain so far proves *the program performs plan P* and *P's steps compute
  these values*.  What it did not say is that those values are the model's:
  `Transformer` appeared exactly once in the whole Qwen2 stack, and
  `layerPlan.denote` was never mentioned.  So "the program computes the layer"
  was assembled by reading, not by theorem.

  This file closes that for the feed-forward half.  Each proven stage's `val`
  is, by construction, the `denote` of an `Expr` — and those `Expr`s are the
  ones in `Transformer.lean` (`siluGateSpec` is
  `.mul (Transformer.silu (.var 0)) (.var 1)`; `addSpec` is `.add v0 v1`).
  What was missing was the *composition*: tying the plan's memory at a given
  address to the stage's `val` at the block that owns it.  That is `step_val`,
  and it is what the theorems below supply.

  The vendor GEMVs remain what they were — `cublasIsMatvec` and, for the two
  batched contractions, nothing.  This is about the proven kernels.
-/

open AlgorithmLib AlgorithmLib.ML Qwen2Proven.Stage Qwen2Common
set_option maxRecDepth 8000

namespace Qwen2Spec
/-- Memory after RMSNorm and both projection GEMVs. -/
noncomputable def ffnMem2 (m : Buf → Nat → Float32) : Buf → Nat → Float32 :=
  (cublasStep B_WU B_XN B_UP D_FF D).step
    ((cublasStep B_WG B_XN B_GATE D_FF D).step (ffnNormStage.step m))

/-- **The silu stage's value is its spec's denotation** — and that spec is
    `.mul (Transformer.silu (.var 0)) (.var 1)`, written from the model. -/
theorem ffn_silu_val (m : Buf → Nat → Float32) (cta a : Nat) :
    ffnSiluStage.val m cta a
      = denote (fun i => m (bindFfnSilu (Qwen2Proven.twoIn i)) a) Qwen2Proven.siluGateSpec := rfl

/-- …and the gate/up buffers it reads are the ones the plan binds. -/
example : bindFfnSilu (Qwen2Proven.twoIn ⟨0, by decide⟩) = B_GATE := by decide
example : bindFfnSilu (Qwen2Proven.twoIn ⟨1, by decide⟩) = B_UP := by decide

/-- **The activation the shipped FFN leaves in `B_ACT` is the model's.**

    Not "a silu-shaped thing": `denote … siluGateSpec` where `siluGateSpec` is
    `.mul (Transformer.silu (.var 0)) (.var 1)`, read off `Transformer.lean`.
    The route is `step_val` — the plan's memory at an owned address is the
    stage's `val` at the block that owns it. -/
theorem ffn_act_is_spec (m : Buf → Nat → Float32) (j : Nat) (hj : j < D_FF) :
    ffnMem3 m B_ACT j
      = denote (fun i => ffnMem2 m (bindFfnSilu (Qwen2Proven.twoIn i)) j)
          Qwen2Proven.siluGateSpec := by
  have hff : D_FF = 4864 := rfl
  have hw : AlgorithmLib.ML.W = 32 := rfl
  have hdom : ffnSiluStage.dom (j / 32) j :=
    ⟨⟨j % 32, by omega⟩, by show j / 32 * 32 + j % 32 = j; omega⟩
  have hlt : j / 32 < ffnSiluStage.grid := by
    show j / 32 < D_FF / 32
    omega
  show ffnSiluStage.step (ffnMem2 m) B_ACT j = _
  rw [show (B_ACT : Buf) = ffnSiluStage.out from rfl,
      StageSpec.step_val ffnSiluStage ffnSiluStage_exclusive _ (j / 32) j hlt hdom]
  exact ffn_silu_val _ _ _

/-- Memory just before the residual add: everything the FFN computed. -/
noncomputable def ffnMem5 (m : Buf → Nat → Float32) : Buf → Nat → Float32 :=
  (cublasStep B_WD B_ACT B_AO D D_FF).step (ffnMem3 m)

/-- The add stage's value is its spec's denotation — `addSpec = .add v0 v1`. -/
theorem ffn_add_val (m : Buf → Nat → Float32) (cta a : Nat) :
    ffnAddStage.val m cta a
      = denote (fun i => m (bindFfnAdd (Qwen2Proven.twoIn i)) a) Qwen2Proven.addSpec := rfl

example : bindFfnAdd (Qwen2Proven.twoIn ⟨0, by decide⟩) = B_X := by decide
example : bindFfnAdd (Qwen2Proven.twoIn ⟨1, by decide⟩) = B_AO := by decide

/-- **The FFN's output is the residual plus what it computed** — stated as the
    spec's `denote`, so the `+` is the model's, not a re-derivation. -/
theorem ffn_out_is_spec (m : Buf → Nat → Float32) (i : Nat) (hi : i < D) :
    ffnPlan.denote m B_X i
      = denote (fun k => ffnMem5 m (bindFfnAdd (Qwen2Proven.twoIn k)) i)
          Qwen2Proven.addSpec := by
  have hdd : D = 896 := rfl
  have hw : AlgorithmLib.ML.W = 32 := rfl
  have hdom : ffnAddStage.dom (i / 32) i :=
    ⟨⟨i % 32, by omega⟩, by show i / 32 * 32 + i % 32 = i; omega⟩
  have hlt : i / 32 < ffnAddStage.grid := by
    show i / 32 < D / 32
    omega
  show ffnAddStage.step (ffnMem5 m) B_X i = _
  rw [show (B_X : Buf) = ffnAddStage.out from rfl,
      StageSpec.step_val ffnAddStage ffnAddStage_exclusive _ (i / 32) i hlt hdom]
  exact ffn_add_val _ _ _


-- ---------------------------------------------------------------------------
-- Attention's four add stages
-- ---------------------------------------------------------------------------

/-! The three bias adds and the residual add are all `addStage`, so each one's
    `val` is `denote … addSpec` for free; what has to be supplied per stage is
    the composition — which block owns the address, and that the plan's memory
    there is that block's value.  Same recipe as `ffn_out_is_spec`. -/

section AttnAdds
variable (gim : Buf → Nat → Nat)

theorem attn_biasQ_val (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32) (cta a : Nat) :
    (aBiasQStage gim).val m cta a
      = denote (fun k => m (bBiasQ (Qwen2Proven.twoIn k)) a) Qwen2Proven.addSpec := rfl

example : bBiasQ (Qwen2Proven.twoIn ⟨0, by decide⟩) = B_Q := by decide
example : bBiasK (Qwen2Proven.twoIn ⟨0, by decide⟩) = B_K := by decide
example : bBiasV (Qwen2Proven.twoIn ⟨0, by decide⟩) = B_V := by decide
example : bAadd (Qwen2Proven.twoIn ⟨0, by decide⟩) = B_X := by decide
example : bAadd (Qwen2Proven.twoIn ⟨1, by decide⟩) = B_XN := by decide

/-- **The biased query is the model's add.** -/
theorem attn_biasQ_is_spec (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32)
    (i : Nat) (hi : i < D) :
    attnMem gim h hm 5 m B_Q i
      = denote (fun k => attnMem gim h hm 4 m (bBiasQ (Qwen2Proven.twoIn k)) i)
          Qwen2Proven.addSpec := by
  have hdd : D = 896 := rfl
  have hw : AlgorithmLib.ML.W = 32 := rfl
  have hdom : (aBiasQStage gim).dom (i / 32) i :=
    ⟨⟨i % 32, by omega⟩, by show i / 32 * 32 + i % 32 = i; omega⟩
  have hlt : i / 32 < (aBiasQStage gim).grid := by
    show i / 32 < D / 32
    omega
  show (aBiasQStage gim).step (attnMem gim h hm 4 m) B_Q i = _
  rw [show (B_Q : Buf) = (aBiasQStage gim).out from rfl,
      StageSpec.step_val (aBiasQStage gim) (StageSpec.rename_exclusive _ _ _ _ _ (addStage_exclusive _)) _ (i / 32) i hlt hdom]
  exact attn_biasQ_val gim h hm _ _ _

/-- **…and the residual add that ends the attention half.** -/
theorem attn_add_is_spec (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32)
    (i : Nat) (hi : i < D) :
    (attnPlan gim h hm).denote m B_X i
      = denote (fun k => attnMem gim h hm 15 m (bAadd (Qwen2Proven.twoIn k)) i)
          Qwen2Proven.addSpec := by
  have hdd : D = 896 := rfl
  have hw : AlgorithmLib.ML.W = 32 := rfl
  have hdom : (aAddStage gim).dom (i / 32) i :=
    ⟨⟨i % 32, by omega⟩, by show i / 32 * 32 + i % 32 = i; omega⟩
  have hlt : i / 32 < (aAddStage gim).grid := by
    show i / 32 < D / 32
    omega
  show (aAddStage gim).step (attnMem gim h hm 15 m) B_X i = _
  rw [show (B_X : Buf) = (aAddStage gim).out from rfl,
      StageSpec.step_val (aAddStage gim) (StageSpec.rename_exclusive _ _ _ _ _ (addStage_exclusive _)) _ (i / 32) i hlt hdom]
  rfl

end AttnAdds

-- ---------------------------------------------------------------------------
-- RMSNorm, under its two named laws
-- ---------------------------------------------------------------------------

/-! RMSNorm is the one proven stage whose `val` is not a spec's `denote`: it is
    `stageOfEW` with a hand-written `rmsVal`, because the kernel computes the
    sum of squares as a *two-level* fold — a per-lane strided sweep, then a
    butterfly across the warp.  That is what the hardware does, and stating it
    that way keeps the stage exact.

    Joining it to the model's flat `Σ x²` is exactly what the law registry is
    for: `Law.combinerComm` to move off lane 0, `Law.laneRegroup` to flatten
    the two levels.  Both are already on the trusted surface, and both stay
    visible in the type below — which is the point. -/

/-- The kernel's per-lane sweep *is* a strided dot of `x` with itself. -/
theorem rmsLaneFold_eq_dot (mem : Nat → Float32) (cta : Nat) :
    Qwen2Proven.rmsLaneFold mem cta
      = dotStridedLane mem mem (fun i l => i * 32 + l.val) (fun i l => i * 32 + l.val)
          (D / 32) := rfl

/-- **RMSNorm's value is the model's**, under the two named laws. -/
theorem rms_val_is_spec (hc : AllHold [Law.combinerComm])
    (hs : AllHold [Law.laneRegroup])
    (mem : Buf → Nat → Float32) (cta a : Nat) :
    rmsVal mem cta a
      = NumOps.mul (NumOps.mul (mem 0 a) (mem 1 a))
          (NumOps.rsqrt (NumOps.add
            (NumOps.mul
              ((List.range D).foldl
                (fun acc i => NumOps.add acc (NumOps.mul (mem 0 i) (mem 0 i)))
                (NumOps.ofNat 0))
              (NumOps.inv Qwen2Proven.dFloat))
            Qwen2Proven.rmsEps)) := by
  simp only [rmsVal, rmsScaleVal]
  rw [rmsLaneFold_eq_dot,
      show (bflyFold : (Lane → Float32) → Lane → Float32)
          = bflyFoldOp (fun x y => NumOps.add x y) from rfl,
      bfly_lane_uniform_add hc _ (laneOf a) ⟨0, by decide⟩,
      show bflyFoldOp (fun x y => NumOps.add x y)
          = (bflyFold : (Lane → Float32) → Lane → Float32) from rfl,
      strided_eq_flatSum hs (mem 0) (mem 0) (D / 32),
      show D / 32 * 32 = D from rfl]

-- ---------------------------------------------------------------------------
-- RoPE
-- ---------------------------------------------------------------------------

/-! `Transformer.rope`'s two branches are `x lo * cos - x hi * sin` and
    `x lo * sin + x hi * cos`, with `hi = half + lo`.  The kernel's `ropeLoVal`
    and `ropeHiVal` have exactly those shapes — `a - b` *is* `.add a (.neg b)` —
    and `ropeHiIx = ropeLoIx + Qwen2Proven.HALF` is the `hi = half + lo` of the model.  What
    the theorems below add is the index arithmetic: which addresses those are. -/

theorem rope_lo_is_spec (im : Buf → Nat → Nat) (mem : Buf → Nat → Float32)
    (cta a : Nat) (h : a < cta * Qwen2Proven.HEAD_DIM + Qwen2Proven.HALF) :
    ropeVal im mem cta a
      = NumOps.add
          (NumOps.mul
            (mem 0 (cta * Qwen2Proven.HEAD_DIM + (ropeLane cta a).val))
            (mem 2 (Qwen2Proven.MAX_SEQ * Qwen2Proven.HALF
                      + (im 1 1 * Qwen2Proven.HALF + (ropeLane cta a).val))))
          (NumOps.neg (NumOps.mul
            (mem 0 (cta * Qwen2Proven.HEAD_DIM + (ropeLane cta a).val
                      + Qwen2Proven.HALF))
            (mem 2 (im 1 1 * Qwen2Proven.HALF + (ropeLane cta a).val)))) := by
  simp only [ropeVal, if_pos h]
  rfl

theorem rope_hi_is_spec (im : Buf → Nat → Nat) (mem : Buf → Nat → Float32)
    (cta a : Nat) (h : ¬ (a < cta * Qwen2Proven.HEAD_DIM + Qwen2Proven.HALF)) :
    ropeVal im mem cta a
      = NumOps.add
          (NumOps.mul
            (mem 0 (cta * Qwen2Proven.HEAD_DIM + (ropeLane cta a).val))
            (mem 2 (im 1 1 * Qwen2Proven.HALF + (ropeLane cta a).val)))
          (NumOps.mul
            (mem 0 (cta * Qwen2Proven.HEAD_DIM + (ropeLane cta a).val
                      + Qwen2Proven.HALF))
            (mem 2 (Qwen2Proven.MAX_SEQ * Qwen2Proven.HALF
                      + (im 1 1 * Qwen2Proven.HALF + (ropeLane cta a).val)))) := by
  simp only [ropeVal, if_neg h]
  rfl

-- ---------------------------------------------------------------------------
-- The KV write
-- ---------------------------------------------------------------------------

/-! The model treats the cache as an *input* (`K V : Fin seq → Fin hd → Expr Γ`),
    so the KV write has no arithmetic spec to match — its correspondence is a
    layout fact: the slot it fills holds head `cta`'s element, gathered from the
    source at the position `im` supplies.  Stating it keeps the cache honest:
    if the destination arithmetic drifted, `attnHead`'s `K t i` would no longer
    be what the write put there. -/
theorem kv_is_gather (im : Buf → Nat → Nat) (mem : Buf → Nat → Float32) (cta a : Nat) :
    kvVal im mem cta a
      = mem 0 (cta * Qwen2Proven.HEAD_DIM + kvElemOf im cta a) := rfl

-- ---------------------------------------------------------------------------
-- The feed-forward half, as one equation
-- ---------------------------------------------------------------------------

/-- Rewrite under a `List.foldl` when the step functions agree pointwise. -/
theorem foldl_congr_fin {n : Nat} (F G : Float32 → Fin n → Float32)
    (h : ∀ acc j, F acc j = G acc j) (init : Float32) (l : List (Fin n)) :
    l.foldl F init = l.foldl G init := by
  induction l generalizing init with
  | nil => rfl
  | cons a as ih => simp only [List.foldl_cons, h, ih]

/-- `B_X` is untouched until the residual add. -/
theorem ffnMem5_resid (m : Buf → Nat → Float32) : ffnMem5 m B_X = m B_X := by
  show (cublasStep B_WD B_ACT B_AO D D_FF).step (ffnMem3 m) B_X = _
  rw [(cublasStep B_WD B_ACT B_AO D D_FF).frame (ffnMem3 m) B_X (by decide)]
  show ffnSiluStage.step _ B_X = _
  funext a
  rw [StageSpec.step_otherBuf ffnSiluStage _ B_X a (by decide),
      congrFun ((cublasStep B_WU B_XN B_UP D_FF D).frame _ B_X (by decide)) a,
      congrFun ((cublasStep B_WG B_XN B_GATE D_FF D).frame _ B_X (by decide)) a]
  exact StageSpec.step_otherBuf ffnNormStage m B_X a (by decide)

/-- Carry a buffer across a segment of `ffnPlan`, cut from the plan itself so
    no `StageSpec` is ever compared — the same trick as `attnMem_carry`. -/
theorem ffnMem_carry (j k : Nat) (hjk : j ≤ k) (m : Buf → Nat → Float32) (b : Buf)
    (hb : ∀ o ∈ outsOf ((ffnPlan.steps.take k).drop j), b ≠ o) :
    (Plan.mk (ffnPlan.steps.take k)).denote m b
      = (Plan.mk (ffnPlan.steps.take j)).denote m b := by
  have hmin : ffnPlan.steps.take j = (ffnPlan.steps.take k).take j := by
    rw [List.take_take, Nat.min_eq_left hjk]
  have hsplit : ffnPlan.steps.take k
      = ffnPlan.steps.take j ++ (ffnPlan.steps.take k).drop j := by
    rw [hmin]; exact (List.take_append_drop j _).symm
  show (Plan.mk (ffnPlan.steps.take k)).denote m b = _
  rw [hsplit, Plan.denote_append]
  exact denote_frame_outs _ _ b hb

theorem ffn_seg_5_6 : outsOf ((ffnPlan.steps.take 6).drop 5) = [B_X] := rfl

theorem ffnMem_full (m : Buf → Nat → Float32) :
    (Plan.mk (ffnPlan.steps.take 6)).denote m = ffnPlan.denote m := rfl

/-- The residual add is the last write, so `B_AO` survives it. -/
theorem ffn_ao_at_end (m : Buf → Nat → Float32) :
    ffnPlan.denote m B_AO = ffnMem5 m B_AO := by
  rw [← ffnMem_full m]
  exact ffnMem_carry 5 6 (by omega) m B_AO (by rw [ffn_seg_5_6]; decide)

set_option maxHeartbeats 1000000 in
/-- **The whole feed-forward half, in the model's terms.**

    Residual plus `W_down · (silu(gate) ⊙ up)`, where the activation is
    `denote … siluGateSpec` — the `Expr` written from `Transformer.silu`.  The
    only assumption beyond the agreed surface is `CuBlasIsMatvec`, which is what
    the three vendor GEMVs rest on and is itself trust-surface. -/
theorem ffn_half_is_spec (hl : CuBlasIsMatvec) (m : Buf → Nat → Float32)
    (i : Nat) (hi : i < D) :
    ffnPlan.denote m B_X i
      = NumOps.add (m B_X i)
          ((List.finRange D_FF).foldl
            (fun acc j => NumOps.add acc
              (NumOps.mul (m B_WD (i * D_FF + j.val))
                (denote (fun k => ffnMem2 m (bindFfnSilu (Qwen2Proven.twoIn k)) j.val)
                   Qwen2Proven.siluGateSpec)))
            (NumOps.ofNat 0)) := by
  rw [ffn_out_is_spec m i hi]
  show NumOps.add (ffnMem5 m B_X i) (ffnMem5 m B_AO i) = _
  rw [ffnMem5_resid, ← ffn_ao_at_end, ffn_down_is_matvec hl m i hi]
  exact congrArg (NumOps.add (m B_X i))
    (foldl_congr_fin _ _ (fun acc j => by rw [ffn_act_is_spec m j.val j.isLt]) _ _)

-- ---------------------------------------------------------------------------
-- The embedding gather, and where the model stops
-- ---------------------------------------------------------------------------

/-- **The embedding lookup is a row copy**, and its spec says exactly that:
    `embedSpec = .var 0`, the identity on the gathered element.  So the content
    is entirely in the index — which row of the table, chosen by the token id
    the launch's integer memory supplies. -/
theorem embed_is_gather (im : Buf → Nat → Nat) (grid : Nat)
    (m : Buf → Nat → Float32) (cta a : Nat) :
    (Qwen2Proven.Stage.embedStage im grid).val m cta a
      = m 0 ((Qwen2Proven.embedIx).eval cta 0 (laneMod a) (fun _ _ => 0) im) := rfl

/-! **Where the model stops.**

    `argmaxStage` has no counterpart in `Transformer.lean`, and should not: the
    model defines a *layer*, and the argmax is the sampling tail — picking a
    token is a decoding policy, not part of the function the layer computes.
    Its value is `argmaxVal`, a max-reduction over the vocabulary, and that is
    the specification; there is nothing above it to check it against.  Recording
    this so a future audit does not read the absence as an omission.  (A `rfl`
    restating `argmaxVal` was tried and is worthless anyway — it is a tautology,
    and elaborating it overflows the stack.) -/
-- ---------------------------------------------------------------------------
-- The attention half, as far as the laws reach
-- ---------------------------------------------------------------------------

/-! The attention composite assembles the same way as `ffn_half_is_spec`, with
    one difference that is not about effort: **softmax**.  The kernel computes
    the numerically-stable max-subtracted form,
    `exp(zₐ − max) · inv(Σ exp(zⱼ − max))`, while `Transformer.softmax` is
    `exp(zᵢ) · inv(Σ exp(zⱼ))` with no max.  Those agree in exact arithmetic and
    **not** in `Float32`, so joining them needs a law the registry does not have
    — the five are `expIsEx2`, `sumAssoc`, `stridedRegroup`, `cublasIsMatvec`,
    `combinerComm`.  Adding a sixth widens the trusted surface, which is a
    decision, not a derivation.

    So the theorem below goes as far as the existing laws reach: the output
    projection's input is the attention output buffer, and everything from there
    to the residual is the model's.  What it does *not* claim is that
    `B_PR` holds `Transformer.softmax` of the scores. -/

set_option maxHeartbeats 1000000 in
/-- **The attention half's residual output, in the model's terms.**

    `x + Wo · attnOut`, with the add being `addSpec`'s denotation and the
    projection `CuBlasIsMatvec`.  The two batched contractions and softmax sit
    upstream in `attnMem … 14`, and are exactly what `declaredLawGap = 2` and
    the softmax note above account for. -/
theorem attn_half_resid_is_spec (gim : Buf → Nat → Nat) (hl : CuBlasIsMatvec)
    (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b)))
    (m : Buf → Nat → Float32) (i : Nat) (hi : i < D) :
    (attnPlan gim h hm).denote m B_X i
      = NumOps.add (attnMem gim h hm 15 m B_X i)
          ((List.finRange D).foldl
            (fun acc j => NumOps.add acc
              (NumOps.mul (m B_WO (i * D + j.val))
                (attnMem gim h hm 14 m B_AO j.val)))
            (NumOps.ofNat 0)) := by
  have hseg : outsOf (((attnPlan gim h hm).steps.take 16).drop 15) = [B_X] := rfl
  have hxn : attnMem gim h hm 15 m B_XN = (attnPlan gim h hm).denote m B_XN := by
    rw [← attnMem_full gim h hm m]
    exact (attnMem_carry gim h hm 15 16 (by omega) m B_XN
            (by rw [hseg]; decide)).symm
  rw [attn_add_is_spec gim h hm m i hi]
  show NumOps.add (attnMem gim h hm 15 m B_X i) (attnMem gim h hm 15 m B_XN i) = _
  rw [congrFun hxn i]
  exact congrArg (NumOps.add (attnMem gim h hm 15 m B_X i))
    (attn_o_is_matvec gim hl h hm m i hi)

-- ---------------------------------------------------------------------------
-- Softmax
-- ---------------------------------------------------------------------------

/-! The kernel computes `exp(zₐ − max) · inv(Σⱼ exp(zⱼ − max))`.  With
    `Transformer.softmaxAt` — softmax about a supplied shift, which is what real
    implementations and hence the published Qwen2 weights were evaluated
    against — that is the model's shape exactly, at `shift := smxMax`.  **No new
    law is needed for the shift.**

    What is left is flattening `Qwen2Proven.smSumFold`, a chunk+remainder fold with runtime
    bounds, to the model's flat `Expr.sum`.  That is the *same* fold-regrouping
    family the project already licenses (`sumAssoc`, `stridedRegroup`) at a
    different shape — a generalisation of an accepted law, not a new class of
    assumption. -/

/-- **The softmax value has the model's shape**, at the kernel's own shift.
    Both sides are `mul (exp (add z (neg s))) (inv …)`; only the summation
    differs, and that difference is the fold-regrouping law. -/
theorem softmax_val_shape (im : Buf → Nat → Nat) (mem : Buf → Nat → Float32)
    (cta a : Nat) :
    smxVal im mem cta a
      = NumOps.mul
          (NumOps.exp (NumOps.add (mem 0 a) (NumOps.neg (smxMax im mem cta))))
          (smxInv im mem cta) := rfl

/-- …and the denominator is `Σ exp(z − max)` in the kernel's committed fold
    order — the thing a regrouping law would flatten. -/
theorem softmax_inv_is_sum (im : Buf → Nat → Nat) (mem : Buf → Nat → Float32)
    (cta : Nat) :
    smxInv im mem cta
      = NumOps.inv (Qwen2Proven.smSumFold (mem 0) (fun _ => smxMax im mem cta) cta
          (fun _ _ => 0) im ⟨0, by decide⟩) := rfl

-- ---------------------------------------------------------------------------
-- The whole layer
-- ---------------------------------------------------------------------------

/-- The residual stream is untouched until attention's last step writes it. -/
theorem attn_x_untouched (gim : Buf → Nat → Nat) (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32) :
    attnMem gim h hm 15 m B_X = m B_X := by
  funext a
  exact attnMem_frame gim h hm 15 m B_X
    (by rw [show outsOf ((attnPlan gim h hm).steps.take 15)
              = [B_XN, B_Q, B_K, B_V, B_Q, B_K, B_V, B_Q, B_K, B_KC, B_VC,
                 B_SC, B_PR, B_AO] ++ [B_XN] from rfl]
        decide) a

/-- Weights are read-only: nothing attention writes is `W_down`. -/
theorem attn_wd_untouched (gim : Buf → Nat → Nat) (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32) :
    (attnPlan gim h hm).denote m B_WD = m B_WD := by
  funext a
  rw [← attnMem_full gim h hm m]
  exact attnMem_frame gim h hm 16 m B_WD
    (by rw [show outsOf ((attnPlan gim h hm).steps.take 16)
              = [B_XN, B_Q, B_K, B_V, B_Q, B_K, B_V, B_Q, B_K, B_KC, B_VC,
                 B_SC, B_PR, B_AO, B_XN] ++ [B_X] from rfl]
        decide) a

set_option maxHeartbeats 1000000 in
/-- **A whole decoder layer, in the model's terms.**

    `x + W_o·attnOut + W_down·(silu(gate) ⊙ up)` — the double-residual shape of
    `Transformer.layer`, assembled from the two halves rather than read off the
    generator.  Both `W_o` and `W_down` are the *caller's* weight buffers, not
    plan memory: attention writes neither, and that is `attn_wd_untouched`.

    What is left inside plan memory is `attnMem … 14 m B_AO`, the output of the
    two batched contractions (scores `Q·Kᵀ`, then `probs·V`).  Those are the
    `declaredLawGap = 2`: NVIDIA's batched `sgemm` has no fold-order
    specification, so no equation for them exists to state.  Everything else in
    the layer is either a proven kernel or `CuBlasIsMatvec`. -/
theorem layer_is_spec (gim : Buf → Nat → Nat) (hl : CuBlasIsMatvec)
    (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b)))
    (m : Buf → Nat → Float32) (i : Nat) (hi : i < D) :
    (layerPlan gim h hm).denote m B_X i
      = NumOps.add
          (NumOps.add (m B_X i)
            ((List.finRange D).foldl
              (fun acc j => NumOps.add acc
                (NumOps.mul (m B_WO (i * D + j.val))
                  (attnMem gim h hm 14 m B_AO j.val)))
              (NumOps.ofNat 0)))
          ((List.finRange D_FF).foldl
            (fun acc j => NumOps.add acc
              (NumOps.mul (m B_WD (i * D_FF + j.val))
                (denote (fun k =>
                    ffnMem2 ((attnPlan gim h hm).denote m)
                      (bindFfnSilu (Qwen2Proven.twoIn k)) j.val)
                  Qwen2Proven.siluGateSpec)))
            (NumOps.ofNat 0)) := by
  show (Plan.mk ((attnPlan gim h hm).steps ++ ffnPlan.steps)).denote m B_X i = _
  rw [Plan.denote_append]
  show ffnPlan.denote ((attnPlan gim h hm).denote m) B_X i = _
  rw [ffn_half_is_spec hl ((attnPlan gim h hm).denote m) i hi,
      attn_wd_untouched gim h hm m,
      attn_half_resid_is_spec gim hl h hm m i hi,
      attn_x_untouched gim h hm m]

-- ---------------------------------------------------------------------------
-- Two fold-shape lemmas, proved once
-- ---------------------------------------------------------------------------

/-- `range (m+n)` folds as `range m` and then `range n` shifted by `m`.  Pure
    list arithmetic; no law. -/
theorem foldl_range_add {α : Type} (g : α → Nat → α) (m : Nat) :
    ∀ (n : Nat) (a : α),
      (List.range (m + n)).foldl g a
        = (List.range n).foldl (fun x j => g x (m + j)) ((List.range m).foldl g a) := by
  intro n
  induction n with
  | zero => intro a; rfl
  | succ n ih =>
    intro a
    simp only [show m + (n + 1) = (m + n) + 1 from rfl, List.range_succ, List.foldl_append,
      List.foldl_cons, List.foldl_nil, ih a]

/-- `denote`'s `.sum` folds over `finRange`; the kernel folds over `range`.
    Same walk, same order — this is the translation, and it is a theorem. -/
theorem foldl_finRange_range {α : Type} (g : α → Nat → α) :
    ∀ (n : Nat) (a : α),
      (List.finRange n).foldl (fun x (j : Fin n) => g x j.val) a
        = (List.range n).foldl g a := by
  intro n
  induction n with
  | zero => intro a; rfl
  | succ n ih =>
    intro a
    rw [List.finRange_succ_last, List.foldl_append, List.foldl_map, List.range_succ,
      List.foldl_append]
    show g ((List.finRange n).foldl (fun x (j : Fin n) => g x j.val) a) n = _
    rw [ih a]
    rfl

-- ---------------------------------------------------------------------------
-- Softmax's denominator, flattened
-- ---------------------------------------------------------------------------

/-- **The kernel's `Σ exp(z − max)` is the flat sum over the row.**

    Two levels come apart here.  The chunk sweep is lane-partitioned and closed
    by a butterfly — that is `Law.laneRegroup`, at `f i = exp(zᵢ − max)` and
    at this row's base rather than at zero, which is exactly why the law is
    stated for an arbitrary summand and base.  The remainder sweep is a plain
    sequential fold on top, and needs no law at all: `SmMeta.tail` says the
    remainder starts where the chunks stopped, so the two together walk
    `0 … CHUNKS·32 + REM` once each, in order. -/
theorem softmax_inv_is_flat_sum (h : AllHold [Law.laneRegroup])
    (im : Buf → Nat → Nat) (hm : SmMeta im) (mem : Buf → Nat → Float32) (cta : Nat) :
    smxInv im mem cta
      = NumOps.inv
          ((List.range (im 1 Qwen2Proven.CHUNKS_SLOT * 32 + im 1 Qwen2Proven.REM_SLOT)).foldl
            (fun acc i => NumOps.add acc (NumOps.exp (NumOps.add
              (mem 0 (cta * im 1 Qwen2Proven.SEQ_SLOT + i))
              (NumOps.neg (smxMax im mem cta))))) NumOps.zero) := by
  refine congrArg NumOps.inv ?_
  show (List.range (im 1 Qwen2Proven.REM_SLOT)).foldl
      (fun a j => NumOps.add a (NumOps.exp (NumOps.add
        (mem 0 (cta * im 1 Qwen2Proven.SEQ_SLOT + (im 1 Qwen2Proven.TAIL_SLOT + j)))
        (NumOps.neg (smxMax im mem cta)))))
      (bflyFoldOp (fun a b => NumOps.add a b)
        (fun l => (List.range (im 1 Qwen2Proven.CHUNKS_SLOT)).foldl
          (fun acc j => NumOps.add acc (NumOps.exp (NumOps.add
            (mem 0 (cta * im 1 Qwen2Proven.SEQ_SLOT + (j * 32 + l.val)))
            (NumOps.neg (smxMax im mem cta))))) NumOps.zero)
        ⟨0, by decide⟩) = _
  rw [strided_regroup_at h
        (fun i => NumOps.exp (NumOps.add (mem 0 i) (NumOps.neg (smxMax im mem cta))))
        (cta * im 1 Qwen2Proven.SEQ_SLOT) (im 1 Qwen2Proven.CHUNKS_SLOT),
      hm.tail,
      foldl_range_add
        (fun acc i => NumOps.add acc (NumOps.exp (NumOps.add
          (mem 0 (cta * im 1 Qwen2Proven.SEQ_SLOT + i))
          (NumOps.neg (smxMax im mem cta)))))
        (im 1 Qwen2Proven.CHUNKS_SLOT * 32) (im 1 Qwen2Proven.REM_SLOT)]

-- ---------------------------------------------------------------------------
-- …and softmax is the model's softmax
-- ---------------------------------------------------------------------------

/-- The attention row as an expression context: variable `0` is the shift,
    variable `j+1` is score `j` of the row block `cta` owns. -/
def rowEnv (im : Buf → Nat → Nat) (mem : Buf → Nat → Float32) (cta n : Nat) :
    Fin (n + 1) → Float32 :=
  fun k => match k.val with
    | 0          => smxMax im mem cta
    | Nat.succ j => mem 0 (cta * im 1 Qwen2Proven.SEQ_SLOT + j)

/-- Score `j` of the row, as an expression. -/
def rowScore (n : Nat) (j : Fin n) : Expr (n + 1) := .var ⟨j.val + 1, by omega⟩

/-- The shift, as an expression. -/
def rowShift (n : Nat) : Expr (n + 1) := .var ⟨0, by omega⟩

/-- **The softmax kernel computes the model's softmax.**

    `Transformer.softmaxAt n shift z i` — `e^{zᵢ−s} / Σⱼ e^{zⱼ−s}` — denoted at
    the row, with the shift instantiated to the row maximum the kernel itself
    computed.  This is the model as written, not a restatement of the kernel:
    the reference implementation these weights were trained and evaluated
    against subtracts a shift too (`torch.nn.functional.softmax`), so the
    shifted form is the definition and no law pays for it.

    The one law is `Law.laneRegroup`, for the denominator's two-level fold —
    the same law RMSNorm already uses, at a different summand. -/
theorem softmax_is_spec (h : AllHold [Law.laneRegroup])
    (im : Buf → Nat → Nat) (hm : SmMeta im) (mem : Buf → Nat → Float32) (cta : Nat)
    (i : Fin (im 1 Qwen2Proven.CHUNKS_SLOT * 32 + im 1 Qwen2Proven.REM_SLOT)) :
    smxVal im mem cta (cta * im 1 Qwen2Proven.SEQ_SLOT + i.val)
      = denote (rowEnv im mem cta _)
          (Transformer.softmaxAt _ (rowShift _) (rowScore _) i) := by
  show NumOps.mul
      (NumOps.exp (NumOps.add (mem 0 (cta * im 1 Qwen2Proven.SEQ_SLOT + i.val))
        (NumOps.neg (smxMax im mem cta))))
      (smxInv im mem cta) = _
  rw [softmax_inv_is_flat_sum h im hm mem cta,
      ← foldl_finRange_range
        (fun acc j => NumOps.add acc (NumOps.exp (NumOps.add
          (mem 0 (cta * im 1 Qwen2Proven.SEQ_SLOT + j))
          (NumOps.neg (smxMax im mem cta)))))
        (im 1 Qwen2Proven.CHUNKS_SLOT * 32 + im 1 Qwen2Proven.REM_SLOT)]
  rfl

end Qwen2Spec
