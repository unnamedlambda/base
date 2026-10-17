import AlgorithmLib.Check.Entry
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
  check_body

open AlgorithmLib.Check in
/-- **The calls the host program makes**, in order, at geometry `g` with the
    binding table at `bo`: start the device; make one buffer, for the input and
    the output; upload the data to its start; launch the kernel at `rPTX_OFF`
    on that one buffer bound twice, `rLaunches` times, each on a grid of
    `g.gridX` blocks of `wBlockDim` threads; copy the blocks back from
    `g.outOff`, where the output has room for them; stop the device. An
    argument written `none` is one the program computes as it runs: the
    buffer `cudaCreateBuffer` answered, and the data's length. -/
def warpCalls (g : Lz4Ship.WP) (bo : Nat) : Tr :=
  .seq
    (.lit [⟨.ffi .cudaInit, [some (regionBase .arena + 16)]⟩,
      ⟨.ffi .cudaCreateBuffer, [some cudaCtx, some (UInt64.ofNat (g.outOff + g.outAlloc))]⟩,
      ⟨.ffi .cudaUploadOffset, [some cudaCtx, none, some 0, some (regionBase .data), none]⟩])
    (.seq
      (.rep Lz4Ship.rLaunches (.lit [⟨.ffi .cudaLaunch,
        [some cudaCtx, some (regionBase .arena + UInt64.ofNat Lz4Ship.rPTX_OFF), some 2,
         some (regionBase .arena + UInt64.ofNat bo), some (UInt64.ofNat g.gridX), some 1, some 1,
         some (UInt64.ofNat Lz4Ship.wBlockDim), some 1, some 1]⟩]))
      (.seq
        (.alt
          (.lit [⟨.ffi .cudaDownloadOffset,
            [some cudaCtx, none, some (UInt64.ofNat g.outOff), some (regionBase .out),
             some (UInt64.ofNat g.totOut)]⟩])
          (.lit []))
        (.lit [⟨.ffi .cudaCleanup, [some (regionBase .arena + 16)]⟩])))

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- **Every run of the 32 KiB host program that ends makes the calls
    `warpCalls` admits**, after whatever was observed before it: one
    allocation, one upload before any launch, exactly `rLaunches` launches,
    each on a grid of `numBlk` blocks of `modelBlockDim` threads, and nothing
    after the device is stopped. A statement about runs of the emitted body, not a reading of
    its text. -/
theorem warp15_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena (Lz4Ship.WP.mk 15).memSize,
      .roomArg .data dataLen, .roomArg .out outLen,
      .cstrIn (regionBase .arena + UInt64.ofNat Lz4Ship.rPTX_OFF) ((Lz4Ship.WP.mk 15).bindOff - Lz4Ship.rPTX_OFF)] w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk 15) : Prog Slot Lvl Unit))
      = .ok obs w') :
    (warpCalls (Lz4Ship.WP.mk 15) (Lz4Ship.WP.mk 15).bindOff).Covers w obs := by
  rw [memSize_15, bindOff_15] at h
  rw [show (Lz4Ship.warpCode (Lz4Ship.WP.mk 15) : Prog Slot Lvl Unit) = (Lz4Ship.warpCodeAt (Lz4Ship.WP.mk 15) 10720 : Prog Slot Lvl Unit) by
    rw [Lz4Ship.warpCode, bindOff_15]] at hr
  rw [bindOff_15]
  check_calls

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
  check_body

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- The same of the 64 KiB host program. -/
theorem warp16_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena (Lz4Ship.WP.mk 16).memSize,
      .roomArg .data dataLen, .roomArg .out outLen,
      .cstrIn (regionBase .arena + UInt64.ofNat Lz4Ship.rPTX_OFF) ((Lz4Ship.WP.mk 16).bindOff - Lz4Ship.rPTX_OFF)] w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk 16) : Prog Slot Lvl Unit))
      = .ok obs w') :
    (warpCalls (Lz4Ship.WP.mk 16) (Lz4Ship.WP.mk 16).bindOff).Covers w obs := by
  rw [memSize_16, bindOff_16] at h
  rw [show (Lz4Ship.warpCode (Lz4Ship.WP.mk 16) : Prog Slot Lvl Unit) = (Lz4Ship.warpCodeAt (Lz4Ship.WP.mk 16) 10720 : Prog Slot Lvl Unit) by
    rw [Lz4Ship.warpCode, bindOff_16]] at hr
  rw [bindOff_16]
  check_calls

end Lz4CompSafe

#print axioms Lz4CompSafe.warp15_no_misuse
#print axioms Lz4CompSafe.warp16_no_misuse
#print axioms Lz4CompSafe.warp15_calls
#print axioms Lz4CompSafe.warp16_calls
