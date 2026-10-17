module
public import AlgorithmLib.Host.Clif
meta import AlgorithmLib.Host.Clif
public import Lz4.Comp
meta import Lz4.Comp
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # What the emitted host program is known to do

  The calls the host program makes are `Lz4CompSafe.warp15_calls` and
  `warp16_calls` (`Ship.Lz4Comp`): every run of the emitted body that ends
  starts the device, makes one allocation, uploads the input into it, launches
  the kernel `rLaunches` times on a grid of `numBlk` blocks of `modelBlockDim`
  threads, copies the blocks back where the output has room, and stops the
  device.  That is a statement about runs, proven by the checker
  (`AlgorithmLib.Check`), and the pattern `Lz4CompSafe.warpCalls` is written
  with the generator's own `rLaunches`, `WP.gridX` and `wBlockDim`, so a change
  to any of them breaks it.

  What that pattern does not say is here: both kernel parameters are one
  buffer, read off the payload image.
-/

namespace Lz4Host

open Lz4Ship
open AlgorithmLib
open AlgorithmLib.Clif

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

end Lz4Host
