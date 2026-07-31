import Qwen2Algorithm
import Qwen2Spec

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
