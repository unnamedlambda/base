import Qwen2Algorithm
import Qwen2Spec
import Qwen2NonVacuity

open AlgorithmLib AlgorithmLib.ML AlgorithmLib.Clif
open Qwen2Common Qwen2Proven Qwen2Proven.Stage Qwen2Spec

/-! # The top of the chain

  Every other theorem in this development proves one link.  This module states
  the conclusion those links exist for, about the program that actually ships.
-/

/-- **What the emitted CLIF does to the residual stream is the transformer
    layer.**

    Read left to right:

    1. `planOf? … (deviceOpsOf …)` takes the two shipped CLIF functions,
       recovers the device operations they perform — `Clif.launchesOf` and
       `Clif.bindsOf` reading the emitted instruction stream, not a declaration
       — and resolves them against the kernel table.  That resolution succeeds
       and yields `layerPlan`, stated as `= some (layerPlan …)` rather than
       `∃ Pl`, so no other plan can satisfy it.
    2. Running that plan under any realisation `R` that `Honours` the vendor
       calls leaves `B_X i` equal to
       `x + W_o·attnOut + W_down·(silu(gate) ⊙ up)`, with `Transformer.layer`
       pinned to this shape by the `rfl` guards in `ML/Transformer.lean`.

    The hypotheses are the whole assumption set: `hl`, `h` and `hm` are named
    laws, `hR` the vendor contract.  Below it sits `Clif.lean`'s instruction
    semantics and `ML/Ptx.lean`'s emitter; outside it sit ptxas and the
    driver. -/
theorem shipped_layer_is_transformer (gim : Buf → Nat → Nat)
    (hl : CuBlasIsMatvec)
    (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b)))
    (R : Realisation) (hR : Honours R) (st : WSt)
    (i : Nat) (hi : i < Qwen2Common.D) :
    planOf? (layerKernels gim h hm) layerDeclared
        (deviceOpsOf ROOT (Qwen2.inferLayerAttnFn.run {}).2
          ++ deviceOpsOf ROOT (Qwen2.inferLayerFfnFn.run {}).2)
      = some (layerPlan gim h hm)
    ∧ ((layerPlan gim h hm).run R st).mem B_X i
        = NumOps.add
            (NumOps.add (st.mem B_X i)
              ((List.finRange Qwen2Common.D).foldl
                (fun acc j => NumOps.add acc
                  (NumOps.mul (st.mem B_WO (i * Qwen2Common.D + j.val))
                    (attnMem gim h hm 14 st.mem B_AO j.val)))
                (NumOps.ofNat 0)))
            ((List.finRange Qwen2Common.D_FF).foldl
              (fun acc j => NumOps.add acc
                (NumOps.mul (st.mem B_WD (i * Qwen2Common.D_FF + j.val))
                  (denote (fun k =>
                      ffnMem2 ((attnPlan gim h hm).denote st.mem)
                        (bindFfnSilu (Qwen2Proven.twoIn k)) j.val)
                    Qwen2Proven.siluGateSpec)))
              (NumOps.ofNat 0)) :=
  ⟨layer_program_realises_plan gim h hm,
   by rw [layer_computes gim h hm R hR st]
      exact Qwen2Spec.layer_is_spec gim hl h hm st.mem i hi⟩

/-- Apply `f` to `x` `n` times.  Written here rather than reused because the
    recursion has to unfold on the left, matching how `List.replicate` grows. -/
def iterN {α : Type} (f : α → α) : Nat → α → α
  | 0,     x => x
  | n + 1, x => iterN f n (f x)

/-- A plan repeated `n` times denotes `n` iterations of its denotation. -/
theorem denote_replicate (S : List PStep) :
    ∀ (n : Nat) (m : Buf → Nat → Float32),
      (Plan.mk ((List.replicate n S).flatten)).denote m
        = iterN (fun mm => (Plan.mk S).denote mm) n m := by
  intro n
  induction n with
  | zero => intro m; rfl
  | succ k ih =>
      intro m
      show (Plan.mk (S ++ (List.replicate k S).flatten)).denote m = _
      rw [Plan.denote_append, ih]
      rfl

/-- **A token is twenty-four layers.**  The structural half of the token-level
    statement: `tokenPlan`'s denotation is the embedding, then the layer
    denotation iterated `N_LAYERS` times, then the sampling tail.  Combined
    with `Qwen2Spec.layer_is_spec`, which identifies one iteration with the
    transformer layer, this carries the spec across the whole token. -/
theorem token_is_layers (gim : Buf → Nat → Nat)
    (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b)))
    (m : Buf → Nat → Float32) :
    (tokenPlan gim h hm).denote m
      = (finalPlan gim).denote
          (iterN (fun mm => (layerPlan gim h hm).denote mm)
            Qwen2Common.N_LAYERS ((entryPlan gim).denote m)) := by
  show (Plan.mk ((entryPlan gim).steps
        ++ ((List.replicate Qwen2Common.N_LAYERS (layerPlan gim h hm).steps).flatten
            ++ (finalPlan gim).steps))).denote m = _
  rw [Plan.denote_append, Plan.denote_append, denote_replicate]

/-- **The whole token, end to end, at concrete values.**

    `shipped_layer_is_transformer` states one layer; this states a token, and
    every part of it is concrete rather than universally quantified: `gimW` is
    the shipped index map, `idealR` a realisation proven to `Honour` the vendor
    calls by `idealR_honours`, and `smMetaW` the meta discharged from what the
    generator emits.  So the statement cannot be satisfied vacuously.

    1. the device operations recovered from the emitted CLIF — all 533 of them,
       with the 24x loop bound read out of the emitted `icmp` — resolve to
       `tokenPlan`;
    2. running it leaves memory equal to the embedding, then the layer
       denotation iterated `N_LAYERS` times, then the sampling tail.

    `Qwen2Spec.layer_is_spec` identifies one iteration with
    `x + W_o*attnOut + W_down*(silu(gate) (*) up)`, so what the shipped program
    computes for a token is the transformer, twenty-four times over. -/
theorem shipped_token_is_layers (h : AllHold [Law.combinerComm]) (st : WSt) :
    planOf? (tokenKernels Qwen2NonVacuity.gimW h Qwen2NonVacuity.smMetaW)
        tokenDeclared tokenOps
      = some (tokenPlan Qwen2NonVacuity.gimW h Qwen2NonVacuity.smMetaW)
    ∧ ((tokenPlan Qwen2NonVacuity.gimW h Qwen2NonVacuity.smMetaW).run
          idealR st).mem
        = (finalPlan Qwen2NonVacuity.gimW).denote
            (iterN (fun mm =>
                (layerPlan Qwen2NonVacuity.gimW h Qwen2NonVacuity.smMetaW).denote mm)
              Qwen2Common.N_LAYERS
              ((entryPlan Qwen2NonVacuity.gimW).denote st.mem)) :=
  ⟨Qwen2NonVacuity.token_realises_concrete h,
   by rw [Qwen2NonVacuity.token_computes_concrete h st]
      exact token_is_layers _ h _ st.mem⟩
