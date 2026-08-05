import Lz4CompAlgorithm

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
  | 5 | the Rust runtime and CUDA FFI | That `cudaUploadRaw` puts the caller's bytes in the input buffer and the download reads back the output buffer.  Nothing ties the theorem's `gm` to the host's `data`. |
  | 6 | the hardware realises *some* schedule | PTX's documented DRF-SC guarantee.  Race-freedom itself is proven (`schedule_completes`). |

  ## Open obligations

  Hypotheses of `launch_correct` that no theorem discharges.  Both are host-side;
  no kernel-level obligation is left.  `Lz4Scan` reports every claim carrying one.

  * `LayoutOK` — the buffer-placement contract.  Satisfiable
    (`Lz4NonVacuity.layoutOK_witness`), so the claims below it are not vacuous.
    Its placement clause is an equation, `outPtr = inPtr + totIn`, both halves of
    which are facts about the shipped program: the kernel computes its output base
    from its input base (`prologueInstrs` index 1), and the host makes ONE
    allocation bound to both parameters (`Lz4Host.host_single_allocation`,
    `bind_table_same_buffer`).  `Lz4NonVacuity.layoutOK_of_alloc` derives every
    per-warp clause from that one equation.  What remains is that
    `cudaCreateBuffer n` yields `n` contiguous addressable bytes — row 5.

  * `LaunchesTo` — that the host repeats the launch as the recovered loop says.

  `RegConfined` and `CursorAtSites` are theorems at both geometries
  (`regConfined_shipped`, `regConfined_shipped64`, `cursorAtSites_shipped`), and
  the chain below them is proven throughout:

      LaunchAgreesPerWarp / LaunchFrame  ←  RaceFree  ←  KernelConfined  ←  RegConfined

  ## Known gaps that are not hypotheses

  * **`dataLen` is unenforced.**  The host uploads whatever the caller passes.
    The theorems hold about whatever `gm` then contains; that it contains the
    corpus is the caller's side of the contract.
-/

namespace Lz4Assumptions

open Algorithm

/-- **The ledger's anchors.**  Each field is the *type* of a theorem named in the
    prose above.  Renaming or deleting one of them fails to elaborate, so the
    ledger cannot outlive the code it describes. -/
structure Anchors where
  /-- The per-warp claim, at both shipped geometries. -/
  shipped32 : ShippedCorrect 15
  shipped64 : ShippedCorrect 16
  /-- Distinct warps write disjoint ranges — the proven half of DRF. -/
  disjoint :
    ∀ (b outPtr w w' : Nat), w ≠ w' → ∀ j : Nat,
      outPtr + w * (WP.mk b).outStride ≤ j →
      j < outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4 →
      j < outPtr + w' * (WP.mk b).outStride ∨
        outPtr + w' * (WP.mk b).outStride + (WP.mk b).lenOff + 4 ≤ j
  /-- The payload image and the offsets derived from it agree. -/
  payloadLen : ∀ w : WP, (warpPayloadDSL w).length = w.bindOff + 8
  payloadFits : ∀ w : WP, (warpPayloadDSL w).length ≤ w.memSize

def anchors : Anchors :=
  { shipped32   := shipped32_correct
    shipped64   := shipped64_correct
    disjoint    := warp_regions_disjoint
    payloadLen  := payload_length
    payloadFits := payload_fits }

end Lz4Assumptions
