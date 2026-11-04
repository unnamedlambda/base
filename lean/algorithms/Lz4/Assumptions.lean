module
public import Lz4.Whole
meta import Lz4.Whole
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # What the LZ4 compressor proves, and what it rests on

  `Lz4Scan.lean` computes this surface from the proof terms and fails the build on
  anything outside it.  This file is the prose; that file is the check.  Every
  theorem named here is anchored in `Anchors` below, so renaming or deleting one
  breaks elaboration.

  ## Scope

  Proven, at both shipped geometries (`blkLog` 15 and 16): for every warp of the
  launch, from an *arbitrary* global memory, shared memory and buffer address
  satisfying `LayoutOK`, the machine reaches `pc = 272`, the `u32` at the length
  slot holds some `k ≤ lenOff`, and `LZ4Imp.decompress` of the first `k` output
  bytes returns exactly that warp's input slice.  `launch_correct` lifts this to
  the FINAL device memory, so every block decodes out of the memory the host
  downloads rather than each warp in isolation.

  `Lz4Whole.shipped32_run_correct` / `shipped64_run_correct` compose that with
  every layer under it — confinement, race-freedom, the interleaving theorem, and
  the twenty repetitions — into one statement whose only hypothesis is
  `LayoutOK`.  The schedule is not assumed: `schedComplete_exists` constructs a
  long-enough one for each launch, and `launch_agrees` holds for all of them.

  The proven object is the executed object by construction: `WP.kernel w` is one
  definition, `serializeKernel` prints it, `warpPayloadDSL` embeds those bytes,
  and the artifact's `initial_memory` is that payload.

  ## Trusted base

  These carry no theorem.  Everything above rests on them.

  | # | trusted | extent |
  |---|---|---|
  | 1 | `sstep ⊨ real PTX` | The 32-lane machine in `LZ4Simt`/`LZ4SimtRSim` is how per-warp PTX semantics enter.  The position CompCert's ISA model occupies. |
  | 2 | `ptxas` and the GPU | Instruction semantics below PTX.  Fatter than CompCert's assembler, because `ptxas` optimises. |
  | 3 | `serializeKernel` | An unverified printer, `Array SInstr → String`.  Structure is proven up to it; the text it emits is not tied back to `SInstr` semantics. |
  | 4 | `LZ4Imp.decompress ⊨ the LZ4 format` | The spec is a Lean decoder, not the format document.  Validated empirically — `compress_roundtrip.rs` decodes every emitted block with `lz4_flex` and compares byte-for-byte over 209.7 MB. |
  | 5 | the CUDA driver | That `cuMemcpyHtoD` puts the caller's bytes in the allocation and `cuMemcpyDtoH` reads back its output region.  Nothing ties the theorem's `gm` to the host's `data`. |
  | 6 | the hardware realises *some* schedule | PTX's documented DRF-SC guarantee.  Race-freedom itself is proven (`schedule_completes`). |

  ## Open obligations

  One, and it is host-side.  `Lz4Scan` reports every claim carrying it, and the
  `endpoints` list there fails the build if the composed claims ever assume
  anything else.

  * `LayoutOK` — the buffer-placement contract.  Satisfiable
    (`Lz4NonVacuity.layoutOK_witness`), so the claims below it are not vacuous.
    Its placement clause is an equation, `outPtr = inPtr + totIn`, both halves of
    which are facts about the shipped program: the kernel computes its output base
    from its input base (`prologueInstrs` index 1), and the host makes ONE
    allocation handed to both parameters (`Lz4Host.params_same_buffer`, the
    calls in `Lz4CompSafe.warp15_calls`).  `Lz4NonVacuity.layoutOK_of_alloc` derives every
    per-warp clause from that one equation.  What remains is that
    `cuMemAlloc n` yields `n` contiguous addressable bytes — row 5.

  The chain below it is proven throughout, at both geometries, and — the part a
  name-based check cannot see — each link is *applied* to the next, ending at the
  `Anchors` fields `whole32`/`whole64`:

      WholeRun  ←  LaunchesTo  ←  LaunchAgreesPerWarp / LaunchFrame
                ←  RaceFree  ←  KernelConfined  ←  RegConfined / CursorAtSites

  `LaunchesTo` appears in `Lz4Scan`'s open list because the *generic* lemmas
  (`launches_correct`) take it — they are stated about any run.  At the shipped
  geometries it is a conclusion, built by `Lz4Interleave.launchesTo_of_layout`.

  ## The launch count and the grid

  `shipped32_run_correct` is stated at `rLaunches` launches and `numBlk` warps,
  the generator's own constants.  That the host program launches exactly that
  many times on exactly that grid is `Lz4CompSafe.warp15_calls` and
  `warp16_calls` (`Ship.Lz4Comp`): every run of the emitted body that ends
  makes the calls `Lz4CompSafe.warpCalls` admits, a pattern written with the
  same `rLaunches` and `WP.gridX`, so a generator change that altered either
  breaks it.  It is proven of runs, by the checker; what it rests on beyond the
  three ordinary axioms is the binding table's offset, a number the compiler
  computes because the kernel cannot serialize the PTX text to check it.

  ## Known gaps that are not hypotheses

  * **`dataLen` is unenforced.**  The host uploads whatever the caller passes.
    The theorems hold about whatever `gm` then contains; that it contains the
    corpus is the caller's side of the contract.
-/

namespace Lz4Assumptions

open Lz4Ship
open AlgorithmLib.LZ4WarpDSL

/-- **The whole claim for one geometry, written out.**

    `Anchors` holds this at `b = 15` and `b = 16`, so the ledger cannot be
    satisfied by a theorem proven at one geometry and missing at the other — the
    gap that a name-based check cannot see.  Written out rather than referring to
    the theorem's own statement, so weakening the conclusion breaks it too. -/
def WholeRun (b : Nat) : Prop :=
  ∀ (inPtr outPtr : Nat) (smemB : List UInt8) (gm : Array UInt8),
    LayoutOK b inPtr outPtr gm →
    ∃ gfinal,
      Lz4Launches.LaunchesTo b inPtr outPtr smemB rLaunches gm gfinal ∧
      ∀ w, w < (WP.mk b).numBlk →
        ∃ k, 0 < k ∧ k ≤ (WP.mk b).lenOff ∧
          AlgorithmLib.readU32LE gfinal
            (outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff) = k ∧
          AlgorithmLib.LZ4Imp.decompress
            ((List.range k).map (fun i => gfinal.getD
              (outPtr + w * (WP.mk b).outStride + i) 0))
            (WP.mk b).inStride
            = some (gmemInpAt gm (inPtr + w * (WP.mk b).inStride) (WP.mk b).inStride)

/-- **The ledger's anchors.**  Each field is the *type* of a theorem named in the
    prose above.  Renaming or deleting one of them fails to elaborate, so the
    ledger cannot outlive the code it describes. -/
structure Anchors where
  /-- The per-warp claim, at both shipped geometries. -/
  shipped32 : ShippedCorrect 15
  shipped64 : ShippedCorrect 16
  /-- **The composition**, at both shipped geometries: the buffer contract in,
      the twenty launches and every block's decode out.  A layer that is proven
      but never applied leaves its obligation in this field's hypothesis list,
      which is what makes the field fail rather than merely read well. -/
  whole32 : WholeRun 15
  whole64 : WholeRun 16
  /-- Distinct warps write disjoint ranges — the proven half of DRF. -/
  disjoint :
    ∀ (b outPtr w w' : Nat), w ≠ w' → ∀ j : Nat,
      outPtr + w * (WP.mk b).outStride ≤ j →
      j < outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4 →
      j < outPtr + w' * (WP.mk b).outStride ∨
        outPtr + w' * (WP.mk b).outStride + (WP.mk b).lenOff + 4 ≤ j
  /-- The payload image fits the arena. -/
  payloadFits : ∀ w : WP, (warpPayloadDSL w).length ≤ w.memSize
  /-- Both of the kernel's parameters name allocation 0. -/
  sameBuffer : ∀ w : WP, (lzApp w).kernels.map (·.args) = [[0, 0]]

def anchors : Anchors :=
  { shipped32   := shipped32_correct
    shipped64   := shipped64_correct
    whole32     := fun i o s g h => Lz4Whole.shipped32_run_correct i o s g h
    whole64     := fun i o s g h => Lz4Whole.shipped64_run_correct i o s g h
    disjoint    := warp_regions_disjoint
    payloadFits := payload_fits
    sameBuffer  := Lz4Host.params_same_buffer }

end Lz4Assumptions
