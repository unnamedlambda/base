import AlgorithmLib.Proof.Typestate
import Lz4.Comp

/-!
# The LZ4 compressor neither misuses a call nor faults

Both shipped geometries, 32 KiB and 64 KiB blocks, for any lengths of data and
output the caller hands over: the input is uploaded into one allocation (a
length past it is refused), the kernel launched on it, and the blocks copied
back only into an output with room for all of them.

The binding table sits past the kernel's text, so its offset is the text's
length: the compiled one computes it, since the kernel would serialize the
kernel to check it.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace Lz4CompSafe

theorem bindOff_15 : (Lz4Ship.WP.mk 15).bindOff = 10720 := by native_decide
theorem bindOff_16 : (Lz4Ship.WP.mk 16).bindOff = 10720 := by native_decide
theorem memSize_15 : (Lz4Ship.WP.mk 15).memSize = 10848 := by native_decide
theorem memSize_16 : (Lz4Ship.WP.mk 16).memSize = 10848 := by native_decide

abbrev Start (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena 10848, .roomArg .data dl, .roomArg .out ol,
   .cstrIn (regionBase .arena + UInt64.ofNat Lz4Ship.rPTX_OFF) (10720 - Lz4Ship.rPTX_OFF)]

theorem warp15_fine : Fine (emitGo (Lz4Ship.warpCodeAt (Lz4Ship.WP.mk 15) 10720 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem warp15_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena (Lz4Ship.WP.mk 15).memSize,
      .roomArg .data dataLen, .roomArg .out outLen,
      .cstrIn (regionBase .arena + UInt64.ofNat Lz4Ship.rPTX_OFF) ((Lz4Ship.WP.mk 15).bindOff - Lz4Ship.rPTX_OFF)] w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk 15) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk 15) : Prog Slot Lvl Unit)) ≠ .fault m := by
  rw [memSize_15, bindOff_15] at h
  rw [show (Lz4Ship.warpCode (Lz4Ship.WP.mk 15) : Prog Slot Lvl Unit) = (Lz4Ship.warpCodeAt (Lz4Ship.WP.mk 15) 10720 : Prog Slot Lvl Unit) by
    rw [Lz4Ship.warpCode, bindOff_15]]
  exact sound_of_wp_entry (Start dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Start dataLen outLen))
    warp15_fine rfl h m

theorem warp16_fine : Fine (emitGo (Lz4Ship.warpCodeAt (Lz4Ship.WP.mk 16) 10720 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem warp16_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena (Lz4Ship.WP.mk 16).memSize,
      .roomArg .data dataLen, .roomArg .out outLen,
      .cstrIn (regionBase .arena + UInt64.ofNat Lz4Ship.rPTX_OFF) ((Lz4Ship.WP.mk 16).bindOff - Lz4Ship.rPTX_OFF)] w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk 16) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk 16) : Prog Slot Lvl Unit)) ≠ .fault m := by
  rw [memSize_16, bindOff_16] at h
  rw [show (Lz4Ship.warpCode (Lz4Ship.WP.mk 16) : Prog Slot Lvl Unit) = (Lz4Ship.warpCodeAt (Lz4Ship.WP.mk 16) 10720 : Prog Slot Lvl Unit) by
    rw [Lz4Ship.warpCode, bindOff_16]]
  exact sound_of_wp_entry (Start dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Start dataLen outLen))
    warp16_fine rfl h m

end Lz4CompSafe

#print axioms Lz4CompSafe.warp15_no_misuse
#print axioms Lz4CompSafe.warp16_no_misuse
