import AlgorithmLib.ML.PtxPrint
import AlgorithmLib.ML.Block
import AlgorithmLib.ML.Schema
import AlgorithmLib.ML.TapeGrad
import AlgorithmLib.ML.Layered
import AlgorithmLib.ML.KVCache
import AlgorithmLib.ML.Rewrite
import AlgorithmLib.ML.Backprop
import AlgorithmLib.ML.Pipeline
import AlgorithmLib.ML.Geometry
import AlgorithmLib.Layout
import AlgorithmLib.ML.Frontend
import AlgorithmLib.ML.Sched
import AlgorithmLib.ML.QuantMX

namespace AlgorithmLib.ML.Assumptions

section LedgerAnchors

-- the lowering, end to end
example := @compileW_sound
example := @compileWKernel_correct
example := @compileW_frame
example := @emitIdx_sound
example := @emitEW_sound
example := @emitEW_frame
example := @flatEW_sound
example := @flatKernel_sound

-- differentiation
example := @grad_correct
example := @gradProg_correct
example := @gradProgD_correct

-- A39: the window obligation, discharged once for any depth
example := @notUsesBelow_rename
example := @genTele_windowed
example := @windowed_gradD_correct
example := @denseSlot_win
example := @denseStack_gradD_correct

-- A39/A40: the width-looped backward
example := @dotStrided_spec
example := @dotStrided_implements

-- A41: the function-first surface
example := @ofFn
example := @kernelOfFn
example := @mapKernel_ofFn_ptx_exact
example := @slice
example := @matSlice
example := @stackLayers

-- A42/A43/A44: the backward pass, the menu, elements per lane
example := @zipPass_spec
example := @mapKernelAt
example := @dotKernel
example := @Sched.realize_spec
example := @elabAt_addrFree
example := @compileWKernel_correctAt
example := @mapLoopEW_stores

-- A45: backprop through a layer
example := @foldl_finRange_single
example := @zeroLaws_of_numLaws
example := @layer_backprop_dx
example := @layer_backprop_dW
example := @stdLayer_dx
example := @stdLayer_dW
example := @actClosed_silu

-- (seam-guard and pipeline anchors live in `AlgorithmLib.ML.Audit`)

-- schemas: where spec conformance comes from
example := @sweep_fold
example := @sweepLoop_spec
example := @sweepN_spec
example := @sweepM_spec
example := @sweep_frame
example := @bflyRoundOp_spec
example := @warpReduceOp_spec
example := @chunkRemReduce_spec
example := @denote_sweepFoldE
example := @storeFold_at
example := @storeLane_at
example := @storeLoop_at
example := @compileWKernel_stores
example := @storeLane_two_first
example := @storeLane_two_second
example := @storeLane_regs
example := @wrun_setLaneF
example := @Law.holds
example := @bfly_eq_laneSum
example := @gradProgD_correct
example := @blockReduce_sound
example := @warpDotV4_implements
example := @warpSumSqV4Store_implements
example := @blockReduce_sound
example := @blockStore_perm
example := @strideCover

-- the declared propositions, so a rename cannot orphan its ledger entry
example := @ExpIsEx2
example := @ZeroTermFree
example := @ZeroLaws

-- results stated about shipped artifacts, reachable only from here
example := @AlgorithmLib.ML.blockKernel_sound
example := @AlgorithmLib.ML.runWarpN_coherent
example := @AlgorithmLib.ML.blockStoreEW_emits
example := @AlgorithmLib.ML.blockStoreEW_idxBelow
example := @AlgorithmLib.ML.blockStoreEW_expFree
example := @AlgorithmLib.ML.maxReduce_spec
example := @AlgorithmLib.ML.sumReduce_spec
example := @AlgorithmLib.ML.storePass_spec
example := @AlgorithmLib.ML.emitKernelSI_sound
example := @AlgorithmLib.ML.programText_endLabel
example := @AlgorithmLib.ML.elemIx_val
example := @AlgorithmLib.ML.outerStage_exclusive
example := @AlgorithmLib.ML.warpDotV4_sumsq
example := @AlgorithmLib.ML.warpReduceMaxE_expFree
example := @AlgorithmLib.ML.warpReduceMaxE_idxFree
example := @AlgorithmLib.ML.strided_eq_flatSum
example := @AlgorithmLib.ML.grad_correct_int
example := @AlgorithmLib.ML.qdot_fits
example := @AlgorithmLib.ML.maxSafeN_fits
example := @AlgorithmLib.ML.fp4_roundtrip
example := @AlgorithmLib.ML.actClosed_sigmoid
example := @AlgorithmLib.ML.KVCache.attnMix_congr
example := @AlgorithmLib.ML.schedAgree_refl

-- the functional surface is definitionally the spec
example := @AlgorithmLib.ML.Vec.dot_eq
example := @AlgorithmLib.ML.Vec.add_eq
example := @AlgorithmLib.ML.Vec.hadamard_eq
example := @AlgorithmLib.ML.matVec_eq
example := @AlgorithmLib.ML.rmsNorm_eq
example := @AlgorithmLib.ML.softmax_eq

end LedgerAnchors

end AlgorithmLib.ML.Assumptions
