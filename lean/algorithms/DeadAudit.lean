import MlpScan

/-!
  Decl-level reachability audit: which declarations in the ML library and the
  CIFAR generator are reached by nothing.

  File-level import closure is not enough — a module can be imported with zero
  of its declarations used.  This walks `getUsedConstants` over each constant's
  type *and* value, transitively, from the scanner's roots plus the ledger
  anchors, and diffs against every declaration in the audited modules.

  Not built as part of any target; run it with `lake env lean DeadAudit.lean`.
-/

open Lean

namespace DeadAudit

/-- The ledger anchors: `example := @X` produces no declaration, so the names
    Assumptions.lean pins are invisible to a closure walk unless named here. -/
def anchorNames : List String := ["compileW_sound", "compileWKernel_correct", "compileW_frame", "emitIdx_sound", "emitEW_sound", "emitEW_frame", "flatEW_sound", "flatKernel_sound", "grad_correct", "gradProg_correct", "gradProgD_correct", "notUsesBelow_rename", "genTele_windowed", "windowed_gradD_correct", "denseSlot_win", "denseStack_gradD_correct", "dotStrided_spec", "dotStrided_implements", "ofFn", "kernelOfFn", "mapKernel_ofFn_ptx_exact", "slice", "matSlice", "stackLayers", "zipPass_spec", "mapKernelAt", "dotKernel", "Sched.realize_spec", "elabAt_addrFree", "compileWKernel_correctAt", "mapLoopEW_stores", "foldl_finRange_single", "zeroLaws_of_numLaws", "layer_backprop_dx", "layer_backprop_dW", "stdLayer_dx", "stdLayer_dW", "actClosed_silu", "sweep_fold", "sweepLoop_spec", "sweepN_spec", "sweepM_spec", "sweep_frame", "bflyRoundOp_spec", "warpReduceOp_spec", "chunkRemReduce_spec", "denote_sweepFoldE", "storeFold_at", "storeLane_at", "storeLoop_at", "compileWKernel_stores", "storeLane_two_first", "storeLane_two_second", "storeLane_regs", "wrun_setLaneF", "Law.holds", "Law.all_covers", "VendorKernel.all_covers", "bfly_eq_laneSum", "gradProgD_correct", "blockReduce_sound", "warpDotV4_implements", "warpSumSqV4Store_implements", "blockReduce_sound", "blockStore_perm", "strideCover", "ExpIsEx2", "ZeroTermFree", "ZeroLaws", "CuBlasIsMatvec", "CuBlasIsSomeReassoc", "LaneRegroup", "CombinerComm", "cublasSgemvResult", "AlgorithmLib.ML.blockKernel_sound", "AlgorithmLib.ML.runWarpN_coherent", "AlgorithmLib.ML.blockStoreEW_emits", "AlgorithmLib.ML.blockStoreEW_idxBelow", "AlgorithmLib.ML.blockStoreEW_expFree", "AlgorithmLib.ML.maxReduce_spec", "AlgorithmLib.ML.sumReduce_spec", "AlgorithmLib.ML.storePass_spec", "AlgorithmLib.ML.emitKernelSI_sound", "AlgorithmLib.ML.programText_endLabel", "AlgorithmLib.ML.elemIx_val", "AlgorithmLib.ML.outerStage_exclusive", "AlgorithmLib.ML.warpDotV4_sumsq", "AlgorithmLib.ML.warpReduceMaxE_expFree", "AlgorithmLib.ML.warpReduceMaxE_idxFree", "AlgorithmLib.ML.strided_eq_flatSum", "AlgorithmLib.ML.grad_correct_int", "AlgorithmLib.ML.qdot_fits", "AlgorithmLib.ML.maxSafeN_fits", "AlgorithmLib.ML.fp4_roundtrip", "AlgorithmLib.ML.actClosed_sigmoid", "AlgorithmLib.ML.KVCache.attnMix_congr", "AlgorithmLib.ML.schedAgree_refl", "AlgorithmLib.ML.Vec.dot_eq", "AlgorithmLib.ML.Vec.add_eq", "AlgorithmLib.ML.Vec.hadamard_eq", "AlgorithmLib.ML.matVec_eq", "AlgorithmLib.ML.rmsNorm_eq", "AlgorithmLib.ML.softmax_eq", "runGrid_step", "Pipeline.run_denote", "Pipeline.equiv_of_denote_eq", "Pipeline.denote_append", "StageSpec.Idempotent", "StageSpec.step_val", "runGrid_otherAddr", "mapStageIP", "mapStage_idempotent", "reduceStage_idempotent", "outerStage_idempotent"]

def resolve (env : Environment) (s : String) : Option Name :=
  let cands := [s, "AlgorithmLib.ML." ++ s, "AlgorithmLib." ++ s]
  cands.findSome? (fun c =>
    let n := c.toName
    if env.constants.contains n then some n else none)

partial def close (env : Environment) (seen : Std.HashSet Name) (n : Name) :
    Std.HashSet Name :=
  if seen.contains n then seen
  else
    let seen := seen.insert n
    match env.constants.find? n with
    | none => seen
    | some ci =>
      let used := ci.type.getUsedConstants
        ++ (match ci.value? with | some v => v.getUsedConstants | none => #[])
      used.foldl (fun acc m => close env acc m) seen

/-- Lean's own generated declarations, recognised the same way `ScanCore` does,
    plus constructors and structure fields.  A constructor consumed only by
    pattern matching is reached through `casesOn`/`match_`, which are internal
    and so are not walked — reporting it would be a false positive, and a whole
    inductive's worth of them would drown the real findings. -/
def isGenerated (env : Environment) (n : Name) : Bool :=
  TrustScan.isGenerated env n
    || (match env.constants.find? n with
        | some (.ctorInfo _) => true
        | _ => false)
    || (match n with
        | .str _ s => s ∈ ["elim", "ctorElim", "ctorElimType"]
        | _ => false)
    || (n.toString.splitOn ".brecOn.").length > 1

def auditedModule (m : Name) : Bool :=
  (`AlgorithmLib.ML).isPrefixOf m || m == `MlpCifarAlgorithm

def moduleOf (env : Environment) (n : Name) : Option Name :=
  (env.getModuleIdxFor? n).map (fun i => env.header.moduleNames[i.toNat]!)

def run : CoreM Unit := do
  let env ← getEnv
  -- The scan's roots are claims.  `main` is the other root: everything the
  -- generator emits is reached from it and from nothing a theorem mentions.
  let emitRoots := ([`main] ++
    (env.constants.toList.map Prod.fst).filter (fun n =>
      match n with
      | .str _ s => s == "artifacts" && (moduleOf env n).map auditedModule == some true
      | _ => false)).filter (fun n => env.constants.contains n)
  let roots := MlpScan.roots ++ (anchorNames.filterMap (resolve env)) ++ emitRoots
  IO.println s!"[audit] emission roots: {emitRoots}"
  let missing := anchorNames.filter (fun s => (resolve env s).isNone)
  IO.println s!"[audit] roots: {roots.length}  (anchors unresolved: {missing.length})"
  for m in missing do IO.println s!"[audit]   UNRESOLVED ANCHOR {m}"
  let reached := roots.foldl (fun acc r => close env acc r) (∅ : Std.HashSet Name)
  IO.println s!"[audit] reachable constants: {reached.size}"
  -- every declaration the audited modules define
  let mut byMod : Std.HashMap Name (Array Name) := ∅
  let mut total := 0
  for (n, _) in env.constants.toList do
    match moduleOf env n with
    | some m =>
        if auditedModule m && !isGenerated env n && !reached.contains n then
          total := total + 1
          byMod := byMod.insert m ((byMod.getD m #[]).push n)
    | none => pure ()
  IO.println s!"[audit] UNREACHED: {total}"
  for (m, ns) in byMod.toList.toArray.qsort (fun a b => a.1.toString < b.1.toString) do
    IO.println s!"\n=== {m}  ({ns.size})"
    for n in ns.qsort (fun a b => a.toString < b.toString) do
      let kind := match env.constants.find? n with
        | some (.thmInfo _) => "thm"
        | some (.axiomInfo _) => "AXIOM"
        | some (.opaqueInfo _) => "opaque"
        | some (.inductInfo _) => "type"
        | _ => "def"
      IO.println s!"  {kind}  {n}"

end DeadAudit

#eval DeadAudit.run
