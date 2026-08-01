import ScanCore
import Qwen2Algorithm
import Qwen2NonVacuity
import Qwen2Spec
import Qwen2Top

/-!
  # What the inference claims actually rest on — computed, not documented

  The A46 ledger *states* the trusted base.  This file *computes* it: for each
  public end-to-end claim it walks the transitive constant closure of the proof
  term and reports every `axiom` and every `opaque` reachable from it.

  Why this exists: a ledger is a claim about the code, and this session found
  two ways such a claim goes stale without anyone noticing — a file that had not
  compiled in three refactors (so its theorems were not being checked at all,
  while `0 sorry` still reported green), and, earlier, a headline theorem that
  was vacuous on an unsatisfiable hypothesis.  Neither is visible to an axiom
  count of the *theorems you remembered to look at*.  This looks at all of them
  and fails the build on anything outside the declared surface.

  **To widen the trusted base you must edit `ScanCore`'s
  `allowedOpaque`/`allowedAxiom`**, which is a reviewable diff, rather than a
  sentence in a document.

  `BackwardScan.lean` is the same scan over the training pipeline; the two are
  separate files because each generator defines its own `main`.
-/

open Lean

namespace TrustScan

/-- **The public claims.**  Adding a claim here subjects it to the scan; the
    point is that this list is the thing reviewers argue about. -/
def roots : List Name :=
  [ `Qwen2Common.token_computes
  , `Qwen2Common.token_ops_realise_plan
  , `Qwen2Common.every_staged_kernel_planned
  , `Qwen2Common.layer_ops_realise_plan
  , `Qwen2Common.layer_declaredLawGap
  , `Qwen2Common.ffn_gate_is_matvec
  , `Qwen2Common.ffn_up_is_matvec
  , `Qwen2Common.ffn_down_is_matvec
  , `Qwen2Common.attn_q_is_matvec
  , `Qwen2Common.attn_k_is_matvec
  , `Qwen2Common.attn_v_is_matvec
  , `Qwen2Common.attn_o_is_matvec
  , `Qwen2Common.attn_q_path
  , `Qwen2Common.attn_k_path
  , `Qwen2Common.attn_v_path
  , `layer_program_realises_plan
  , `layer_program_computes
  -- the single statement the rest exists for
  , `shipped_layer_is_transformer
  , `token_is_layers
  , `shipped_token_is_layers
  -- the concrete instantiations: these are what make the rest non-vacuous
  , `Qwen2NonVacuity.smMetaW
  , `Qwen2NonVacuity.layer_realises_concrete
  , `Qwen2NonVacuity.token_realises_concrete
  , `Qwen2NonVacuity.token_computes_concrete
  , `Qwen2NonVacuity.ffn_down_at_0
  , `Qwen2NonVacuity.ffn_down_at_last
  , `Qwen2NonVacuity.attn_o_at_0
  , `Qwen2NonVacuity.attn_o_at_last
  -- the spec-correspondence link: these say the values are the *model's*
  , `Qwen2Spec.ffn_act_is_spec
  , `Qwen2Spec.ffn_out_is_spec
  , `Qwen2Spec.ffn_silu_val
  , `Qwen2Spec.ffn_add_val
  , `Qwen2Spec.attn_biasQ_is_spec
  , `Qwen2Spec.attn_add_is_spec
  , `Qwen2Spec.rms_val_is_spec
  , `Qwen2Spec.rmsLaneFold_eq_dot
  , `Qwen2Spec.rope_lo_is_spec
  , `Qwen2Spec.rope_hi_is_spec
  , `Qwen2Spec.kv_is_gather
  -- the composite: a whole half of a layer, in the model's terms
  , `Qwen2Spec.ffn_half_is_spec
  , `Qwen2Spec.attn_half_resid_is_spec
  , `Qwen2Spec.embed_is_gather
  , `Qwen2Spec.softmax_val_shape
  , `Qwen2Spec.softmax_inv_is_sum
  , `Qwen2Spec.softmax_inv_is_flat_sum
  , `Qwen2Spec.softmax_is_spec
  , `Qwen2Spec.layer_is_spec
  -- the host's side of `SmMeta`: the four formulas it publishes, as emitted
  , `Qwen2Common.metaStageFrag_emits
  , `Qwen2Common.metaFrag_slots
  , `Qwen2NonVacuity.smMeta_of_seqLen
  , `Qwen2Common.metaFrag_mem
  , `Qwen2Common.metaFrag_stores
  , `Qwen2NonVacuity.smMeta_of_stores
  , `Qwen2NonVacuity.smMeta_of_frag
  -- the decode step as a *declared program*: the 24× loop is a node, not a list
  , `Qwen2Common.attnDriver_deviceOps
  , `Qwen2Common.ffnDriver_deviceOps
  , `Qwen2Common.layerDriver_deviceOps
  , `Qwen2Common.loopDriver_deviceOps
  , `Qwen2Common.tokenDriver_deviceOps
  , `Qwen2Common.tokenDriver_count
  -- …and the declared program is the built one: same device writes per half,
  -- and the control flow between the halves recovered from the emitted blocks
  , `Qwen2Common.entryDriver_is_built
  , `Qwen2Common.finalDriver_is_built
  , `attnDriver_is_built
  , `ffnDriver_is_built
  , `Qwen2Common.infer_loop_is_layers
  , `Qwen2Common.infer_loop_body_calls
  , `Qwen2Common.final_no_loops
  , `layer_fn_calls
  , `leaf_fns_no_loops
  ]

end TrustScan

open TrustScan in
#eval runScan "inference" roots
