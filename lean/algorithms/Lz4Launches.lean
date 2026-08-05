import Lz4NonVacuity

/-!
  # Repeated launches over the same buffers

  Every theorem about the compressor describes ONE launch.  The artifact runs
  `rLaunches = 20` of them over the same two device buffers, so launches 2…20
  start from an output buffer the previous launch already wrote, and read an
  input buffer that has been sitting under 19 prior launches.  That the result is
  still correct is a theorem below rather than an argument in prose.

  It composes for two reasons, and both are theorems below rather than
  assertions: a launch changes nothing outside the union of the warps' output
  ranges (`LaunchFrame`, the whole-launch lift of the per-warp frame clause
  already inside `ShippedCorrect`), and `LayoutOK` puts the entire input buffer
  before the first of those ranges — so the input a warp reads on launch 20 is
  the input it read on launch 1.

  `ShippedCorrect` already quantifies over an *arbitrary* initial global memory,
  which is what makes a dirty output buffer a non-issue; that generality was
  established when the clear-loop proof stopped depending on zeros.
-/

namespace Lz4Launches

open Algorithm
open AlgorithmLib.LZ4WarpDSL

/-- **A launch's frame.**  Bytes outside every warp's output range come through a
    launch unchanged, and the launch does not resize device memory.

    This is the whole-launch counterpart of the per-warp frame clause in
    `ShippedCorrect`, and it sits in the same place: `LaunchAgreesPerWarp` says
    what the concurrent launch does *inside* each warp's range, this says it does
    nothing outside all of them.  Both are discharged together by an interleaving
    argument; neither is a new kind of assumption. -/
def LaunchFrame (b : Nat) (outPtr : Nat) (gm gfinal : Array UInt8) : Prop :=
  gfinal.size = gm.size ∧
  ∀ j, (∀ w, w < (WP.mk b).numBlk →
          j < outPtr + w * (WP.mk b).outStride ∨
          outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4 ≤ j) →
    gfinal.getD j 0 = gm.getD j 0

/-- `n` launches over the same buffers, each from the memory the last one left. -/
def LaunchesTo (b inPtr outPtr : Nat) (smemB : List UInt8) :
    Nat → Array UInt8 → Array UInt8 → Prop
  | 0, g, g' => g = g'
  | n + 1, g, g' =>
      ∃ gmid, LaunchAgreesPerWarp b inPtr outPtr g smemB gmid ∧
        LaunchFrame b outPtr g gmid ∧ LaunchesTo b inPtr outPtr smemB n gmid g'

/-- **The input survives a launch.**  Every byte of the input buffer lies before
    the first output range, by `LayoutOK`'s whole-buffer clause, so the frame
    leaves it alone. -/
theorem input_preserved (b inPtr outPtr : Nat) (gm gmid : Array UInt8)
    (hlayout : LayoutOK b inPtr outPtr gm)
    (hframe : LaunchFrame b outPtr gm gmid) :
    ∀ w, w < (WP.mk b).numBlk →
      gmemInpAt gmid (inPtr + w * (WP.mk b).inStride) (WP.mk b).inStride
        = gmemInpAt gm (inPtr + w * (WP.mk b).inStride) (WP.mk b).inStride := by
  intro w hw
  obtain ⟨hglob, _⟩ := hlayout
  obtain ⟨_, hout⟩ := hframe
  unfold gmemInpAt
  apply List.map_congr_left
  intro i hi
  rw [List.mem_range] at hi
  -- the byte sits inside warp `w`'s input slice, hence before the whole output
  have hlast : inPtr + w * (WP.mk b).inStride + i < outPtr := by
    have h1 : (w + 1) * (WP.mk b).inStride ≤ (WP.mk b).numBlk * (WP.mk b).inStride :=
      Nat.mul_le_mul_right _ hw
    have h2 : (w + 1) * (WP.mk b).inStride
        = w * (WP.mk b).inStride + (WP.mk b).inStride := Nat.succ_mul w _
    omega
  exact hout _ (fun w' _ => Or.inl (by have := Nat.zero_le (w' * (WP.mk b).outStride); omega))

/-- `LayoutOK` only constrains addresses and the size of memory, and a launch
    changes neither. -/
theorem layoutOK_preserved (b inPtr outPtr : Nat) (gm gmid : Array UInt8)
    (hlayout : LayoutOK b inPtr outPtr gm)
    (hframe : LaunchFrame b outPtr gm gmid) :
    LayoutOK b inPtr outPtr gmid := by
  obtain ⟨hglob, hper⟩ := hlayout
  obtain ⟨hsize, _⟩ := hframe
  refine ⟨hglob, ?_⟩
  intro w hw
  obtain ⟨l1, l2, l3, l5, l6⟩ := hper w hw
  exact ⟨l1, l2, by omega, l5, by omega⟩

/-- **Every block of the last launch decodes to the ORIGINAL input.**

    Stated for `n + 1` launches, so it says something about a run that happened.
    The input slices on the right-hand side are read out of `gm`, the memory as
    the host uploaded it — not out of whatever the 19th launch left behind. -/
theorem launches_correct (b : Nat) (inPtr outPtr : Nat) (gm gfinal : Array UInt8)
    (smemB : List UInt8) (n : Nat)
    (hcorrect : ShippedCorrect b)
    (hlayout : LayoutOK b inPtr outPtr gm)
    (hruns : LaunchesTo b inPtr outPtr smemB (n + 1) gm gfinal) :
    ∀ w, w < (WP.mk b).numBlk →
      ∃ k, 0 < k ∧ k ≤ (WP.mk b).lenOff ∧
        AlgorithmLib.readU32LE gfinal
          (outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff) = k ∧
        AlgorithmLib.LZ4Imp.decompress
          ((List.range k).map (fun i => gfinal.getD
            (outPtr + w * (WP.mk b).outStride + i) 0))
          (WP.mk b).inStride
          = some (gmemInpAt gm (inPtr + w * (WP.mk b).inStride) (WP.mk b).inStride) := by
  induction n generalizing gm with
  | zero =>
      obtain ⟨gmid, hagree, _hframe, hrest⟩ := hruns
      have h : gmid = gfinal := hrest
      rw [← h]
      exact launch_correct b inPtr outPtr gm smemB gmid hcorrect hlayout hagree
  | succ m ih =>
      obtain ⟨gmid, _hagree, hframe, hrest⟩ := hruns
      intro w hw
      obtain ⟨k, hk0, hkle, hlen, hdec⟩ :=
        ih gmid (layoutOK_preserved b inPtr outPtr gm gmid hlayout hframe) hrest w hw
      exact ⟨k, hk0, hkle, hlen,
        by rw [hdec, input_preserved b inPtr outPtr gm gmid hlayout hframe w hw]⟩

/-- The shipped 32 KiB artifact, over its `rLaunches` repetitions.

    This is the *given the launches happened* form — it takes `LaunchesTo`.  The
    endpoint that constructs it is `Lz4Whole.shipped32_run_correct`, which assumes
    only `LayoutOK`. -/
theorem shipped32_launches_correct (inPtr outPtr : Nat) (gm gfinal : Array UInt8)
    (smemB : List UInt8)
    (hlayout : LayoutOK 15 inPtr outPtr gm)
    (hruns : LaunchesTo 15 inPtr outPtr smemB rLaunches gm gfinal) :
    ∀ w, w < (WP.mk 15).numBlk →
      ∃ k, 0 < k ∧ k ≤ (WP.mk 15).lenOff ∧
        AlgorithmLib.readU32LE gfinal
          (outPtr + w * (WP.mk 15).outStride + (WP.mk 15).lenOff) = k ∧
        AlgorithmLib.LZ4Imp.decompress
          ((List.range k).map (fun i => gfinal.getD
            (outPtr + w * (WP.mk 15).outStride + i) 0))
          (WP.mk 15).inStride
          = some (gmemInpAt gm (inPtr + w * (WP.mk 15).inStride) (WP.mk 15).inStride) :=
  launches_correct 15 inPtr outPtr gm gfinal smemB 19 shipped32_correct hlayout hruns

theorem shipped64_launches_correct (inPtr outPtr : Nat) (gm gfinal : Array UInt8)
    (smemB : List UInt8)
    (hlayout : LayoutOK 16 inPtr outPtr gm)
    (hruns : LaunchesTo 16 inPtr outPtr smemB rLaunches gm gfinal) :
    ∀ w, w < (WP.mk 16).numBlk →
      ∃ k, 0 < k ∧ k ≤ (WP.mk 16).lenOff ∧
        AlgorithmLib.readU32LE gfinal
          (outPtr + w * (WP.mk 16).outStride + (WP.mk 16).lenOff) = k ∧
        AlgorithmLib.LZ4Imp.decompress
          ((List.range k).map (fun i => gfinal.getD
            (outPtr + w * (WP.mk 16).outStride + i) 0))
          (WP.mk 16).inStride
          = some (gmemInpAt gm (inPtr + w * (WP.mk 16).inStride) (WP.mk 16).inStride) :=
  launches_correct 16 inPtr outPtr gm gfinal smemB 19 shipped64_correct hlayout hruns

/-- info: 'Lz4Launches.launches_correct' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms launches_correct

end Lz4Launches
