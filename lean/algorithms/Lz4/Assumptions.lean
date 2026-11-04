module
public import Lz4.CodeShip
meta import Lz4.CodeShip
public import Lz4.Host
meta import Lz4.Host
public import Lz4.HostLaunch
meta import Lz4.HostLaunch
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # What the LZ4 compressor proves, and what it rests on

  `Scan.Lz4` computes this surface from the proof terms and fails the build on
  anything outside it. This file is the prose; that file is the check. Every
  theorem named here is anchored in `Anchors` below, so renaming or deleting one
  breaks elaboration.

  ## Scope

  Proven, at both shipped geometries (32 KiB and 64 KiB blocks): the artifact's
  kernel (`Lz4Ship.WP.code`) compiles to PTX, and a launch of it on device
  memory whose buffer 0 holds the input blocks then room for their outputs
  runs, on the Code and on the PTX it compiles to, to the same device memory.
  Each block's output then holds its encoded length `n ≤ lenOff` at `lenOff`,
  and `LZ4Imp.decompress` of its first `n` bytes is exactly the block's input
  (`Lz4Code.shipped_roundtrips`). The launch leaves the inputs as they were and
  buffer 0 as large, so it runs again: after the `rLaunches` launches the
  artifact makes, every block still decodes to its input as it was before the
  first (`Lz4Code.shipped_launches`). The only hypothesis beyond the platform's
  is that buffer 0 is large enough.

  The emitted host program carries the caller's bytes through
  (`Lz4CompSafe.warp_run`, in `Ship.Lz4Run`; the scan pins its hypotheses).
  Every run of it that ends, from what `load` leaves, on a device honest
  about the kernel it loaded, leaves allocation 0 decoding block by block to
  what the launches found there: what it held before the run, with bytes of
  the caller's data, as the caller handed it over, written over its start.
  Where it downloads, the caller's output holds at the end those outputs
  byte for byte: the checker shows no host store touches the caller's data or
  the output's blocks (`Lz4CompSafe.keptRanges`).

  The proven object is the executed object: `Lz4Ship.warpKernelDSLStr` is the
  printed module of `WP.code`'s entry, the very entry `shipped_roundtrips`
  runs, and the artifact embeds those bytes (`warpPayloadDSL`).

  ## Trusted base

  These carry no theorem. Everything above rests on them.

  | # | trusted | extent |
  |---|---|---|
  | 1 | `PtxRun ⊨ real PTX` | The PTX subset's semantics, a launch its blocks in order and each block its threads with collectives answered by the warp's laws. The position CompCert's ISA model occupies. `HwOk` states the GPU's float rules, which this kernel never reaches. |
  | 2 | `ptxas` and the GPU | Instruction semantics below PTX. |
  | 3 | the PTX printer | `PtxIr.Module.text`, a syntax printer from the typed PTX the theorem runs. |
  | 4 | `LZ4Imp.decompress ⊨ the LZ4 format` | The spec is a Lean decoder, not the format document. Validated empirically: the bench decodes every emitted block with `lz4_flex` and compares byte for byte over 209.7 MB. |
  | 5 | the CUDA driver | That it answers `cuMemcpyHtoD`, `cuLaunchKernel` and `cuMemcpyDtoH` as the model does (`cudaDrv`), the launch on a device honest about it (`Honest` for `lz4Machine`, the denotation the PTX model's own launch realizes, `lz4Machine_realized`). Under that, one launch is `Lz4Code.lz4_drv_launch` and the whole program `Lz4CompSafe.warp_run`. |
  | 6 | the hardware runs the blocks as some order of them does | Blocks write disjoint outputs and read only inputs no block writes (`launch_roundtrips`'s layout), so every order leaves the same memory. |

  ## The launch count and the grid

  The host program launches the kernel `rLaunches` times on a grid of `numBlk`
  blocks of 32 threads (`Lz4CompSafe.warp15_calls`, `warp16_calls`), with both
  parameters naming allocation 0 (`Lz4Host.params_same_buffer`).

  ## Known gaps that are not hypotheses

  * **`dataLen` is unenforced.** The host uploads whatever the caller passes.
    The theorem holds of whatever buffer 0 then holds.
-/

namespace Lz4Assumptions

open Lz4Ship Lz4Code AlgorithmLib AlgorithmLib.Kernel

/-- **The ledger's anchors.** Each field is the type of a theorem named in the
    prose above, at both geometries. Renaming or deleting one fails to
    elaborate, so the ledger cannot outlive the code it describes. -/
structure Anchors where
  /-- The whole claim, at both shipped geometries. -/
  roundtrips : ∀ (b : Nat), b = 15 ∨ b = 16 → ∀ (hw : Hw), PtxSound.HwOk hw → ∀ (outA : UInt64)
    (bufs : Nat → ByteArray), (WP.mk b).bytes ≤ (bufs 0).size →
    RoundTrips (WP.mk b).code (WP.mk b).numBlk (WP.mk b).inStride (WP.mk b).outStride (WP.mk b).lenOff hw outA bufs
  /-- The artifact's repeated launches, at both shipped geometries. -/
  launches : ∀ (b : Nat), b = 15 ∨ b = 16 → ∀ (hw : Hw), PtxSound.HwOk hw → ∀ (outA : UInt64)
    (bufs : Nat → ByteArray), (WP.mk b).bytes ≤ (bufs 0).size →
    ∃ e, (WP.mk b).code.shipped = .ok e ∧ ∃ bN, LaunchesRun (WP.mk b).code e (WP.mk b).numBlk
      [.sc .i64 base0, .sc .i64 outA] hw rLaunches bufs bN ∧
      ∀ c, c < (WP.mk b).numBlk → ∃ m, m ≤ (WP.mk b).lenOff ∧
        u32At (bN 0) ((WP.mk b).numBlk * (WP.mk b).inStride + LZ4WarpDSL.copySlack + c * (WP.mk b).outStride +
          (WP.mk b).lenOff) = m ∧
        LZ4Imp.decompress ((List.range m).map fun i =>
            (bN 0).get! ((WP.mk b).numBlk * (WP.mk b).inStride + LZ4WarpDSL.copySlack + c * (WP.mk b).outStride + i))
            (WP.mk b).inStride
          = some ((List.range (WP.mk b).inStride).map fun j => (bufs 0).get! (c * (WP.mk b).inStride + j))
  /-- The payload image fits the arena. -/
  payloadFits : ∀ w : WP, (warpPayloadDSL w).length ≤ w.memSize
  /-- Both of the kernel's parameters name allocation 0. -/
  sameBuffer : ∀ w : WP, (lzApp w).kernels.map (·.args) = [[0, 0]]

def anchors : Anchors :=
  { roundtrips := shipped_roundtrips
    launches := fun b hb hw hhw outA bufs h => shipped_launches b hb hw hhw outA bufs h rLaunches (by decide)
    payloadFits := payload_fits
    sameBuffer := Lz4Host.params_same_buffer }

end Lz4Assumptions
