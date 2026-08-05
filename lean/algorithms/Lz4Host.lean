import Lz4CompAlgorithm
import AlgorithmLib.Clif

set_option maxRecDepth 8192

/-!
  # What the emitted host program actually does

  `warpClif` printed a CLIF program and no theorem said anything about it.  The
  values could not drift — `wBlockDim` *is* `LZ4Simt.modelBlockDim` by definition
  and `WP.gridX` *is* `numBlk` — but nothing checked that those values reach a
  launch, that the input is uploaded first, or that only one kernel is launched.

  `Clif.launchesOf` reads the device operations back out of the blocks the
  generator emitted, so these theorems are about the program that ships rather
  than about the source that produced it.

  ## Why `native_decide`

  These four are the only claims in the LZ4 development that reach the Lean
  compiler, and `Lz4Scan` reports them as such on every build.  Both cheaper
  routes were measured and neither works:

  * `decide` — kernel reduction has to evaluate the `LaunchRec`s, whose
    `bindOff` field is `WP.bindOff`, which serializes the entire PTX kernel
    through unoptimised term rewriting.  It exhausted 15 GB of RAM.
  * `rfl` with the binding offset held opaque (`warpBuilderAt` takes it as a
    parameter for exactly this reason) — still out of memory.  Reducing the
    `IRBuilder` state monad itself is the cost, not the serializer.

  So this is the same trade the inference pipeline makes for
  `Qwen2Common.infer_loop_is_layers` and `final_no_loops`.  What it buys is a
  statement about the emitted program; what it costs is the compiler, named.
-/

namespace Lz4Host

open Algorithm
open AlgorithmLib
open AlgorithmLib.Clif

/-- **The device operations the shipped 32 KiB host program performs**: upload
    the input, then launch — nothing else, and in that order. -/
theorem host_ops32 :
    (launchesOf ((warpBuilder (WP.mk 15)).run {}).2).map
        (fun r => (r.fnName, r.gridX, r.blockX, r.kernelOff, r.nBufs))
      = [("cl_cuda_upload_ptr_offset", none, none, none, none),
         ("cl_cuda_launch", some 6400, some 32, some 256, some 2)] := by
  native_decide

theorem host_ops64 :
    (launchesOf ((warpBuilder (WP.mk 16)).run {}).2).map
        (fun r => (r.fnName, r.gridX, r.blockX, r.kernelOff, r.nBufs))
      = [("cl_cuda_upload_ptr_offset", none, none, none, none),
         ("cl_cuda_launch", some 3200, some 32, some 256, some 2)] := by
  native_decide

/-- **The launch geometry is the geometry the correctness proof assumes.**  The
    grid is exactly `numBlk` — the bound `ShippedCorrect` quantifies warps under,
    so no warp with `w ≥ numBlk` is ever started — and the block is exactly
    `modelBlockDim`, which `LZ4Simt.initRegs_ntid` proves is the `%ntid.x` the
    machine model hands the prologue.  Stated against the definitions, not
    against literals, so a change to either side breaks this. -/
theorem host_grid_is_numBlk32 :
    (launchesOf ((warpBuilder (WP.mk 15)).run {}).2).map (fun r => (r.gridX, r.blockX))
      = [(none, none),
         (some ((WP.mk 15).numBlk : Int), some (AlgorithmLib.LZ4Simt.modelBlockDim : Int))] := by
  native_decide

theorem host_grid_is_numBlk64 :
    (launchesOf ((warpBuilder (WP.mk 16)).run {}).2).map (fun r => (r.gridX, r.blockX))
      = [(none, none),
         (some ((WP.mk 16).numBlk : Int), some (AlgorithmLib.LZ4Simt.modelBlockDim : Int))] := by
  native_decide

-- ── One allocation, and both kernel parameters bound to it ────────────────────

/-- **The whole device-call sequence of the shipped host program.**  One
    `cl_cuda_create_buffer`, and only one — the input and the output live in a
    single allocation, which is what makes `LayoutOK`'s placement clause an
    equation the program establishes rather than a hypothesis about how two
    independent `cudaMalloc` results happen to land. -/
theorem host_calls32 :
    callsOf ((warpBuilder (WP.mk 15)).run {}).2
      = ["cl_cuda_init", "cl_cuda_create_buffer", "cl_cuda_upload_ptr_offset",
         "cl_cuda_launch", "cl_cuda_download_ptr_offset", "cl_cuda_cleanup"] := by
  native_decide

theorem host_calls64 :
    callsOf ((warpBuilder (WP.mk 16)).run {}).2
      = ["cl_cuda_init", "cl_cuda_create_buffer", "cl_cuda_upload_ptr_offset",
         "cl_cuda_launch", "cl_cuda_download_ptr_offset", "cl_cuda_cleanup"] := by
  native_decide

/-- Exactly one allocation is made, at both shipped geometries. -/
theorem host_single_allocation :
    ((callsOf ((warpBuilder (WP.mk 15)).run {}).2).filter
        (· == "cl_cuda_create_buffer")).length = 1 ∧
    ((callsOf ((warpBuilder (WP.mk 16)).run {}).2).filter
        (· == "cl_cuda_create_buffer")).length = 1 := by
  rw [host_calls32, host_calls64]; exact ⟨rfl, rfl⟩

/-- **Both bind slots name buffer 0.**  The binding table is the last eight bytes
    of the payload image, and they are two little-endian zeros — so the kernel's
    `in_ptr` and `out_ptr` parameters receive the *same* device pointer, and the
    output base can only be what the prologue computes from it.

    Read off `warpPayloadDSL` itself, which is the image the artifact embeds, so
    this cannot disagree with what is uploaded. -/
theorem bind_table_same_buffer (w : WP) :
    ∃ pre : List UInt8, pre.length = w.bindOff ∧
      warpPayloadDSL w = pre ++ (uint32ToBytes 0 ++ uint32ToBytes 0) := by
  refine ⟨zeros rPTX_OFF ++ (w.ptxBytes ++ zeros (w.bindOff - rPTX_OFF - w.ptxLen)), ?_,
    by simp only [warpPayloadDSL, List.append_assoc]⟩
  have halign : rPTX_OFF + w.ptxLen ≤ w.bindOff := by
    simp only [WP.bindOff, rPTX_OFF]; omega
  have hdef : w.ptxLen = w.ptxBytes.length := rfl
  simp only [List.length_append, zeros, List.length_replicate]
  omega

/-- **The twenty repetitions are recovered from the emitted program.**

    `Lz4Launches.launches_correct` proves the repetitions compose; this is what
    ties the number *twenty* to the program that ships.

    Recognising the loop takes `loopHdrAccOf?`/`loopBodyAccOkB`: a `forLoopAcc`
    threads an accumulator, so its header has two parameters, and a scan that
    requires exactly one sees no loop at all.  The trip count must resolve to a
    compile-time constant, so it is read out of the program rather than assumed.

    Stated against `rLaunches`, so changing the generator's repetition count
    breaks this. -/
theorem host_loop_is_rLaunches32 :
    loopsOf ((warpBuilder (WP.mk 15)).run {}).2 = [⟨1, 2, 3, rLaunches⟩] := by
  native_decide

theorem host_loop_is_rLaunches64 :
    loopsOf ((warpBuilder (WP.mk 16)).run {}).2 = [⟨1, 2, 3, rLaunches⟩] := by
  native_decide

/-- The launch sits inside that loop's body block, so the twenty trips are
    twenty launches rather than twenty iterations of something else. -/
theorem host_launch_in_loop_body32 :
    (blockInsts? ((warpBuilder (WP.mk 15)).run {}).2 2).map
        (callsIn ((warpBuilder (WP.mk 15)).run {}).2.fns)
      = some ["cl_cuda_launch"] := by
  native_decide

theorem host_launch_in_loop_body64 :
    (blockInsts? ((warpBuilder (WP.mk 16)).run {}).2 2).map
        (callsIn ((warpBuilder (WP.mk 16)).run {}).2.fns)
      = some ["cl_cuda_launch"] := by
  native_decide

-- ── The shape, as one value ──────────────────────────────────────────────────

/-- **Everything the emitted host program's device side is known to do**, bundled
    so it can be anchored and so the two geometries stay in step: a field proven
    at 15 and missing at 16 fails to build.

    Nothing below the host uses these — the runtime semantics that would let a
    kernel theorem consume them is trusted base row 5.  What they do is fix the
    program's shape, so a generator change that moved a launch out of the loop,
    added an allocation, or altered the grid stops the build. -/
structure HostShape (b : Nat) : Prop where
  calls : callsOf ((warpBuilder (WP.mk b)).run {}).2
      = ["cl_cuda_init", "cl_cuda_create_buffer", "cl_cuda_upload_ptr_offset",
         "cl_cuda_launch", "cl_cuda_download_ptr_offset", "cl_cuda_cleanup"]
  oneAlloc : ((callsOf ((warpBuilder (WP.mk b)).run {}).2).filter
      (· == "cl_cuda_create_buffer")).length = 1
  geom : (launchesOf ((warpBuilder (WP.mk b)).run {}).2).map (fun r => (r.gridX, r.blockX))
      = [(none, none),
         (some ((WP.mk b).numBlk : Int), some (AlgorithmLib.LZ4Simt.modelBlockDim : Int))]
  loop : loopsOf ((warpBuilder (WP.mk b)).run {}).2 = [⟨1, 2, 3, rLaunches⟩]
  launchInBody : (blockInsts? ((warpBuilder (WP.mk b)).run {}).2 2).map
      (callsIn ((warpBuilder (WP.mk b)).run {}).2.fns) = some ["cl_cuda_launch"]
  bindTable : ∃ pre : List UInt8, pre.length = (WP.mk b).bindOff ∧
      warpPayloadDSL (WP.mk b) = pre ++ (uint32ToBytes 0 ++ uint32ToBytes 0)

theorem hostShape32 : HostShape 15 :=
  ⟨host_calls32, host_single_allocation.1, host_grid_is_numBlk32,
   host_loop_is_rLaunches32, host_launch_in_loop_body32, bind_table_same_buffer _⟩

theorem hostShape64 : HostShape 16 :=
  ⟨host_calls64, host_single_allocation.2, host_grid_is_numBlk64,
   host_loop_is_rLaunches64, host_launch_in_loop_body64, bind_table_same_buffer _⟩

end Lz4Host
