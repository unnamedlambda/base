import AlgorithmLib.ML.PtxPrint
import AlgorithmLib.ML.Block
import AlgorithmLib.ML.Schema
import AlgorithmLib.ML.TapeGrad
import AlgorithmLib.ML.Layered
import AlgorithmLib.ML.KVCache
import AlgorithmLib.ML.Rewrite
import AlgorithmLib.ML.Backprop
import AlgorithmLib.ML.Pipeline
import AlgorithmLib.ML.Compose
import AlgorithmLib.ML.Geometry
import AlgorithmLib.Layout
import AlgorithmLib.ML.Frontend
import AlgorithmLib.ML.Sched
import AlgorithmLib.ML.QuantMX

/-!
  # What the ML stack proves, and what it rests on

  Every theorem named below is anchored in `LedgerAnchors` at the foot of this
  file, so renaming or deleting one breaks the build.  A ledger that can claim
  more than the code contains is worse than none.

  ## Scope

  Proven: a model written as an `Expr` compiles to an `EWStmt`, elaborates into
  the warp machine, lowers through `PtxM`/`PtxFlat`/`PtxPrint`, and the emitted
  PTX runs the spec — from raw launch, at exact `Float32` equality, under the
  declared laws below.  This covers **each kernel this library emits**.

  Composition is proven at the `Pipeline` level: `run_denote` derives an n-ary
  composite from the stage list rather than restating it, and
  `equiv_of_denote_eq` is the criterion for swapping one schedule for another.
  What is *not* covered is the host program that orders the launches.

  ## Trusted base

  These carry no theorem.  Everything above rests on them.

  | # | trusted | extent |
  |---|---|---|
  | 1 | PTX opcode text encoding | `PtxPrint` renders ~20 opcodes; no theorem ties the emitted text to `PInstr` semantics.  Everything *structural* — registers, addresses, branches, loops — is proven, and `qwen2_emit_ok` discharges branch resolution and printability.  Same position CompCert's assembly printer occupies. |
  | 2 | `ptxas` and the GPU | instruction semantics, and that `ptxas` does not contract `mul`+`add` into `fma`.  The second is not assumed blindly: `.rn` variants are emitted to prevent it, after a bit-exact test caught it. |
  | 3 | cuBLAS `sgemv` | ~99.9% of the shipped model's arithmetic, carried as the **named** assumption `Law.cublasIsMatvec`, stated as a left-to-right `Float32` fold, which is the only form expressible without an ℝ semantics for `Float32` — and therefore stronger than NVIDIA guarantees, since no fold order is pinned at any version.  It appears in the type of anything depending on it. |
  | 4 | the CLIF host program | `IR.lean` carries no theorems.  `Clif.lean` models the launch structure — `scanBlock_length` proves the extraction neither invents nor misses a launch, for every block and environment — but the builder itself is a `StateM` action, so facts about a *specific* generator are checked at generation time rather than proven. |
  | 5 | the ℝ bridge | `instNumOpsReal` maps DSL operations to Mathlib operations by name.  `instNumLawsReal` is *proven* on top of it, and `sderiv_hasDerivAt` proves `sderiv` is Mathlib's derivative — but only on the `Regular` fragment, which excludes `letE`. |
  | 6 | `Float32` | native operations with no IEEE model.  `NumLaws` is inhabited at `ℝ` and `Int` only, so no theorem here reasons about float rounding. |

  ## Declared laws

  Named propositions that are *false* at `Float32` and are therefore never
  applied silently: each appears in the type of any theorem depending on it.

  `Law.all` is the registry and `Law.all_covers` holds it to every constructor,
  so this table is prose beside a list the build checks — one row per `Law`:

  | law | content |
  |---|---|
  | `ExpIsEx2` | `e^x = 2^(x·log₂e)` — PTX has no exact `exp`.  Measured 5.06e-7 on silu. |
  | `SumAssoc` | butterfly reduction = sequential fold.  Measured: 100000024 vs 100000000. |
  | `LaneRegroup` | any lane-partitioned fold read as the flat one, given a proof that the walk covers each element once.  Two schedules of one reduction differ by 14%.  `StridedRegroup` is its instance at the interleaved walk. |
  | `CuBlasIsMatvec` | a vendor GEMV equals a left-to-right `Float32` fold of the matvec.  The only form expressible without an ℝ semantics for `Float32`, and therefore *stronger* than NVIDIA guarantees, whose fold order is unspecified. |
  | `CuBlasIsSomeReassoc` | a vendor GEMV sums the right products, each once, in *some* association — what NVIDIA actually promises, and all of the leeway expressible over `Float32`.  `cublasIsMatvec_strengthens` proves the row above refines it. |
  | `CombinerComm` | `Float32` add and max commute, so a butterfly is lane-uniform.  True at IEEE-754 for non-NaN inputs. |

  Beyond the registry, and therefore never carried in a `Law` list:
  `ZeroTermFree` (dropping zero terms from a gradient sum) and `ZeroLaws`,
  which `Layered.lean` takes as hypotheses of the depth-independent window
  argument.

  ## Non-vacuity

  A theorem with an unsatisfiable hypothesis typechecks.  `NonVacuity.lean`
  instantiates the headline theorems at concrete values and guards them with
  deliberately-broken variants, because `0 sorry` does not imply non-vacuous.
-/

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
example := @Law.all_covers
example := @VendorKernel.all_covers
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
example := @CuBlasIsMatvec
example := @CuBlasIsSomeReassoc
example := @LaneRegroup
example := @CombinerComm
example := @cublasSgemvResult

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

-- composition: the launch sequence as a value
example := @runGrid_step
example := @Pipeline.run_denote
example := @Pipeline.equiv_of_denote_eq
example := @Pipeline.denote_append
example := @StageSpec.Idempotent
example := @StageSpec.step_val
example := @runGrid_otherAddr
example := @mapStageIP
example := @mapStage_idempotent
example := @reduceStage_idempotent
example := @outerStage_idempotent

end LedgerAnchors

end AlgorithmLib.ML.Assumptions
