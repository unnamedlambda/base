import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Vit.Algorithm
import Ship.VitState

/-!
# The ViT's main, seed and fetch neither misuse a call nor fault

`main` brings the driver and cuBLAS up, makes every buffer, loads every kernel
and writes its parameters, uploads the inputs, and makes the pooled streams
with a cuBLAS handle each and the events: 2580 buffers and 806 kernels, a body
the checker reads by compiled evaluation (`ship_entry`); what it leaves is
`VitReach`'s. `seed` copies `dL/dlogits` from the caller's data into the buffer the backward
starts from and waits for it to land, and `fetch` copies the logits into the caller's output, each
through the device pointer `main` kept in that buffer's slot of the standard
layout (`Drv.App`): the allocation live, of the size the copy moves.

The layout's state sits past the kernels' text, so its offsets are computed
from that text's length: they are given as literals, checked compiled, since
the kernel would serialize the text to check them.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace VitSafe

theorem bytes : Vit.SQ * Vit.NC * 4 = 102400 := by native_decide
theorem outIx : Vit.VOUT = 976 := by native_decide
theorem seedIx : Vit.VSEED = 977 := by native_decide

/-- What `main` leaves that a copy of buffer `i` through slot `s` reads: a
    live context, no capture open, the allocation of 102400 bytes with its
    pointer in its slot, and the caller's data and output. -/
abbrev Run (i s d o : Nat) : World → Prop := Contracts.TState.holds
  [.part (.has .cuda) true, .part .capturing false, .part .idle true, .part .cuda true,
   .devSeq, .oracles, .room .arena 931252, .room .data d, .room .out o,
   .alloc i 102400, .cell (Drv.App.at_ s) (devPtr i)]

def fetchCode : Prog Slot Lvl Unit := Vit.vFetchFn Vit.VOUT (Vit.SQ * Vit.NC * 4)
theorem fetch_eq : fetchCode = (do
    let base ← basePtr
    let out ← outPtr
    let _ ← ext (.cuda .memcpyDtoH) %[out, ← load64 (← iaddImm base (Vit.vApp.bufSlot Vit.VOUT)), ← iconst64 102400]) := by
  unfold fetchCode Vit.vFetchFn Drv.App.download Drv.download; rw [bytes]; rfl

theorem fetch_fine : Fine (emitGo (fnBody ptrParams fetchCode) ⟨4, 0, [], [], [], none⟩).2 := by
  rw [fetch_eq]; exact ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetch_sound (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 976 (Vit.vApp.bufSlot Vit.VOUT) 0 102400) w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit fetchCode) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit fetchCode) ≠ .fault m := by
  have hf := fetch_fine
  rw [fetch_eq] at hf ⊢
  check_body

def seedCode : Prog Slot Lvl Unit := Vit.vSeedFn
theorem seed_eq : seedCode = (do
    let base ← basePtr
    let src ← dataPtr
    let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base (Vit.vApp.bufSlot Vit.VSEED)), src, ← iconst64 102400]
    let _ ← ext (.cuda .ctxSynchronize) %[]) := by
  unfold seedCode Vit.vSeedFn Drv.App.upload Drv.upload Drv.sync; rw [bytes]; rfl

theorem seed_fine : Fine (emitGo (fnBody ptrParams seedCode) ⟨4, 0, [], [], [], none⟩).2 := by
  rw [seed_eq]; exact ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem seed_sound (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 977 (Vit.vApp.bufSlot Vit.VSEED) 102400 0) w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit seedCode) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit seedCode) ≠ .fault m := by
  have hf := seed_fine
  rw [seed_eq] at hf ⊢
  check_body

ship_entry main : Vit.vLoadFn under VitState.Load

end VitSafe

#print axioms VitSafe.main_no_misuse
#print axioms VitSafe.fetch_sound
#print axioms VitSafe.seed_sound
