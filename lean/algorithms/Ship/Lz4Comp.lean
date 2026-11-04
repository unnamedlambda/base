import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Lz4.Comp

/-!
# The LZ4 compressor neither misuses a call nor faults

Both shipped geometries, 32 KiB and 64 KiB blocks. `load` brings the CUDA
driver up and makes the compressor's one allocation, loads the kernel and
writes its parameters, in the driver's standard layout (`Drv.App`); `main`
starts from what it leaves (`Drv.App.runTS`) and, for any lengths of data and
output the caller hands over, uploads the input into the allocation (a length
past the input's room is refused), launches the kernel on it, and copies the
blocks back only into an output with room for all of them.

The layout's state sits past the kernel's text, so its offset is the text's
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

theorem load15_eq : (Lz4Ship.loadCode (Lz4Ship.WP.mk 15) : Prog Slot Lvl Unit) =
    Lz4Ship.loadCodeAt (Lz4Ship.WP.mk 15) 10720 := by rw [Lz4Ship.loadCode, bindOff_15]
theorem load16_eq : (Lz4Ship.loadCode (Lz4Ship.WP.mk 16) : Prog Slot Lvl Unit) =
    Lz4Ship.loadCodeAt (Lz4Ship.WP.mk 16) 10720 := by rw [Lz4Ship.loadCode, bindOff_16]
theorem warp15_eq : (Lz4Ship.warpCode (Lz4Ship.WP.mk 15) : Prog Slot Lvl Unit) =
    Lz4Ship.warpCodeAt (Lz4Ship.WP.mk 15) 10720 := by rw [Lz4Ship.warpCode, bindOff_15]
theorem warp16_eq : (Lz4Ship.warpCode (Lz4Ship.WP.mk 16) : Prog Slot Lvl Unit) =
    Lz4Ship.warpCodeAt (Lz4Ship.WP.mk 16) 10720 := by rw [Lz4Ship.warpCode, bindOff_16]

/-- What `load` starts from: the driver loaded, nothing made, the kernel's
    text and its entry point's name in the arena. -/
abbrev Load (b : Nat) : World → Prop := Contracts.TState.holds (Lz4Ship.lzAppAt (Lz4Ship.WP.mk b) 10720).loadTS

/-- What `main` starts from: what `load` leaves, and the caller's data and
    output of any lengths. -/
abbrev Run (b : Nat) (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  ((Lz4Ship.lzAppAt (Lz4Ship.WP.mk b) 10720).runTS ++ [.roomArg .data dl, .roomArg .out ol])

ship_entry load15 : Lz4Ship.loadCodeAt (Lz4Ship.WP.mk 15) 10720 under (Load 15)
ship_entry load16 : Lz4Ship.loadCodeAt (Lz4Ship.WP.mk 16) 10720 under (Load 16)

theorem warp15_fine : Fine (emitGo (fnBody ptrParams (Lz4Ship.warpCodeAt (Lz4Ship.WP.mk 15) 10720 : Prog Slot Lvl Unit)) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem warp15_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 15 dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk 15) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk 15) : Prog Slot Lvl Unit)) ≠ .fault m := by
  rw [warp15_eq]
  check_body

open AlgorithmLib.Check in
/-- **The calls one compression makes**, in order, at geometry `g` with the
    layout at `org`: the caller's data copied to the allocation's start, of a
    length within the input's room (or no copy); the kernel launched
    `rLaunches` times on the default stream, each on a grid of `g.gridX`
    blocks of `wBlockDim` threads, with both its parameters the allocation's
    pointer; and the blocks copied back from `g.outOff` into the caller's
    output (or no copy, where it has no room for them). -/
def warpCalls (g : Lz4Ship.WP) (org : Nat) : Tr :=
  .seq
    (.alt (.lit [⟨.ext (.cuda .memcpyHtoD), [.is (devPtr 0), .is (regionBase .data), .within 0 g.outOff]⟩])
      (.lit []))
    (.seq
      (.rep Lz4Ship.rLaunches (.lit [⟨.ext (.cuda .launchKernel),
        [.is (drvHandle kFunc 0), .is (UInt64.ofNat g.gridX), .is 1, .is 1, .is (UInt64.ofNat Lz4Ship.wBlockDim),
         .is 1, .is 1, .is 0, .is 0, .is (Drv.App.at_ ((Lz4Ship.lzAppAt g org).paramsOff 0)), .is 0,
         .is (devPtr 0), .is (devPtr 0)]⟩]))
      (.alt (.lit [⟨.ext (.cuda .memcpyDtoH),
          [.is (regionBase .out), .is (devPtr 0 + UInt64.ofNat g.outOff), .is (UInt64.ofNat g.totOut)]⟩])
        (.lit [])))

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- **Every run of the 32 KiB compression that ends makes the calls
    `warpCalls` admits**, after whatever was observed before it. A statement
    about runs of the emitted body, not a reading of its text. -/
theorem warp15_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 15 dataLen outLen w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk 15) : Prog Slot Lvl Unit))
      = .ok obs w') :
    (warpCalls (Lz4Ship.WP.mk 15) 10720).Covers w obs := by
  rw [warp15_eq] at hr
  check_calls

theorem warp16_fine : Fine (emitGo (fnBody ptrParams (Lz4Ship.warpCodeAt (Lz4Ship.WP.mk 16) 10720 : Prog Slot Lvl Unit)) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem warp16_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 16 dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk 16) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk 16) : Prog Slot Lvl Unit)) ≠ .fault m := by
  rw [warp16_eq]
  check_body

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- The same of the 64 KiB compression. -/
theorem warp16_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 16 dataLen outLen w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk 16) : Prog Slot Lvl Unit))
      = .ok obs w') :
    (warpCalls (Lz4Ship.WP.mk 16) 10720).Covers w obs := by
  rw [warp16_eq] at hr
  check_calls

end Lz4CompSafe

#print axioms Lz4CompSafe.load15_no_misuse
#print axioms Lz4CompSafe.warp15_no_misuse
#print axioms Lz4CompSafe.warp15_calls
#print axioms Lz4CompSafe.warp16_calls
