module
public import Lz4.Confine64
meta import Lz4.Confine64
public import Lz4.Host
meta import Lz4.Host
public import Lz4.NonVacuity
meta import Lz4.NonVacuity
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # The composed claim

  Applies each proven layer to the next: `RegConfined` → `KernelConfined` →
  `RaceFree` → `LaunchAgreesPerWarp`/`LaunchFrame` → `LaunchesTo` → decode.

  `run_correct` assumes `LayoutOK` and derives the rest, the schedule included.
  `Lz4Assumptions.Anchors` holds the result at both geometries; `Lz4Scan`'s
  `endpoints` holds its exact hypothesis set.
-/

namespace Lz4Whole

open Lz4Ship
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4WarpDSL
open Lz4Interleave

/-- `KernelConfined` at every memory the run passes through, 32 KiB.  Arbitrary
    `g`, because launches 2…20 start from what the previous one left. -/
theorem confined32 (inPtr outPtr : Nat) (smemB : List UInt8) :
    ∀ g, LayoutOK 15 inPtr outPtr g → KernelConfined 15 inPtr outPtr g smemB := by
  intro g hlay
  obtain ⟨h1, h2, h3, h4, h5⟩ := confineHyps_of_layoutOK 15 inPtr outPtr g hlay
  exact Lz4Sites.kernelConfined_shipped inPtr outPtr g smemB h1 h2 h3 h4 h5

theorem confined64 (inPtr outPtr : Nat) (smemB : List UInt8) :
    ∀ g, LayoutOK 16 inPtr outPtr g → KernelConfined 16 inPtr outPtr g smemB := by
  intro g hlay
  obtain ⟨h1, h2, h3, h4, h5⟩ := confineHyps_of_layoutOK 16 inPtr outPtr g hlay
  exact Lz4Sites.kernelConfined_shipped64 inPtr outPtr g smemB h1 h2 h3 h4 h5

/-- **The run, end to end.**  A final memory in which every block decodes to its
    slice of the input as the host uploaded it — not to what an intermediate
    launch left.  `LaunchesTo` is a conclusion, `LayoutOK` the only hypothesis. -/
theorem run_correct (b : Nat) (ht : TailOOB b) (inPtr outPtr : Nat)
    (smemB : List UInt8) (n : Nat)
    (hcorrect : ShippedCorrect b)
    (hconf : ∀ g, LayoutOK b inPtr outPtr g → KernelConfined b inPtr outPtr g smemB)
    (gm : Array UInt8) (hlay : LayoutOK b inPtr outPtr gm) :
    ∃ gfinal,
      Lz4Launches.LaunchesTo b inPtr outPtr smemB (n + 1) gm gfinal ∧
      ∀ w, w < (WP.mk b).numBlk →
        ∃ k, 0 < k ∧ k ≤ (WP.mk b).lenOff ∧
          AlgorithmLib.readU32LE gfinal
            (outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff) = k ∧
          AlgorithmLib.LZ4Imp.decompress
            ((List.range k).map (fun i => gfinal.getD
              (outPtr + w * (WP.mk b).outStride + i) 0))
            (WP.mk b).inStride
            = some (gmemInpAt gm (inPtr + w * (WP.mk b).inStride) (WP.mk b).inStride) := by
  obtain ⟨gfinal, hruns, -⟩ :=
    launchesTo_of_layout b ht inPtr outPtr smemB hcorrect hconf (n + 1) gm hlay
  exact ⟨gfinal, hruns,
    Lz4Launches.launches_correct b inPtr outPtr gm gfinal smemB n hcorrect hlay hruns⟩

/-- The shipped 32 KiB artifact, over its twenty launches. -/
theorem shipped32_run_correct (inPtr outPtr : Nat) (smemB : List UInt8)
    (gm : Array UInt8) (hlay : LayoutOK 15 inPtr outPtr gm) :
    ∃ gfinal,
      Lz4Launches.LaunchesTo 15 inPtr outPtr smemB rLaunches gm gfinal ∧
      ∀ w, w < (WP.mk 15).numBlk →
        ∃ k, 0 < k ∧ k ≤ (WP.mk 15).lenOff ∧
          AlgorithmLib.readU32LE gfinal
            (outPtr + w * (WP.mk 15).outStride + (WP.mk 15).lenOff) = k ∧
          AlgorithmLib.LZ4Imp.decompress
            ((List.range k).map (fun i => gfinal.getD
              (outPtr + w * (WP.mk 15).outStride + i) 0))
            (WP.mk 15).inStride
            = some (gmemInpAt gm (inPtr + w * (WP.mk 15).inStride) (WP.mk 15).inStride) :=
  run_correct 15 tail32 inPtr outPtr smemB 19 shipped32_correct
    (confined32 inPtr outPtr smemB) gm hlay

/-- The same for the 64 KiB artifact. -/
theorem shipped64_run_correct (inPtr outPtr : Nat) (smemB : List UInt8)
    (gm : Array UInt8) (hlay : LayoutOK 16 inPtr outPtr gm) :
    ∃ gfinal,
      Lz4Launches.LaunchesTo 16 inPtr outPtr smemB rLaunches gm gfinal ∧
      ∀ w, w < (WP.mk 16).numBlk →
        ∃ k, 0 < k ∧ k ≤ (WP.mk 16).lenOff ∧
          AlgorithmLib.readU32LE gfinal
            (outPtr + w * (WP.mk 16).outStride + (WP.mk 16).lenOff) = k ∧
          AlgorithmLib.LZ4Imp.decompress
            ((List.range k).map (fun i => gfinal.getD
              (outPtr + w * (WP.mk 16).outStride + i) 0))
            (WP.mk 16).inStride
            = some (gmemInpAt gm (inPtr + w * (WP.mk 16).inStride) (WP.mk 16).inStride) :=
  run_correct 16 tail64 inPtr outPtr smemB 19 shipped64_correct
    (confined64 inPtr outPtr smemB) gm hlay

/-- **Not vacuous.**  The composed claim at concrete addresses and a concrete
    memory, with no hypothesis left.  An unsatisfiable contract still typechecks;
    this rules that out for the composition, not only for `LayoutOK`. -/
theorem run_correct_witness :
    ∃ gfinal,
      Lz4Launches.LaunchesTo 15 0 209715232 [] rLaunches
        (Lz4NonVacuity.zeroMem 434522144) gfinal ∧
      ∀ w, w < (WP.mk 15).numBlk →
        ∃ k, 0 < k ∧ k ≤ (WP.mk 15).lenOff ∧
          AlgorithmLib.readU32LE gfinal
            (209715232 + w * (WP.mk 15).outStride + (WP.mk 15).lenOff) = k ∧
          AlgorithmLib.LZ4Imp.decompress
            ((List.range k).map (fun i => gfinal.getD
              (209715232 + w * (WP.mk 15).outStride + i) 0))
            (WP.mk 15).inStride
            = some (gmemInpAt (Lz4NonVacuity.zeroMem 434522144)
                (0 + w * (WP.mk 15).inStride) (WP.mk 15).inStride) :=
  shipped32_run_correct 0 209715232 [] _ Lz4NonVacuity.layoutOK_witness32

end Lz4Whole
