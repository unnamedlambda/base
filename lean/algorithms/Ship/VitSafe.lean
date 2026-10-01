import AlgorithmLib.Proof.Typestate
import Vit.Algorithm

/-!
# The ViT's seed and fetches neither misuse a call nor fault

`seed` uploads `dL/dlogits` from the caller's data into the buffer the backward
starts from, `fetch` downloads the logits into the caller's output, and
`fetchAny` downloads the buffer the caller names, as many bytes as it asks for,
once it has checked the name is a slot of the binding table and the output has
room. Each reads
the buffer's id from the binding table `main` filled; which id it finds does not
matter here, since a buffer that is not there, or not that size, is an error the
call returns.

The binding table sits past the kernels' text, so its offsets are computed from
that text's length: they are given as literals, checked compiled, since the
kernel would serialize the text to check them.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace VitSafe

theorem bindOut : Vit.vBindOff Vit.VOUT = 763968 := by native_decide
theorem bindSeed : Vit.vBindOff Vit.VSEED = 763972 := by native_decide

/-- What `main` leaves that these two read: a live context, no capture open, and
    the caller's data and output. -/
abbrev Run (d o : Nat) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena 775428, .room .data d, .room .out o,
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx]

def fetchCode : Prog Slot Lvl Unit := Vit.vFetchFn Vit.VOUT (Vit.SQ * Vit.NC * 4)
theorem fetch_eq : fetchCode = (do
    let ptr ← basePtr
    let ctxPtr ← cudaCtxPtr ptr
    let outPtr ← outPtr
    let id ← load32 (← absAddr ptr 763968)
    let bytes ← iconst64 102400
    let _ ← ffi .cudaDownload %[ctxPtr, id, outPtr, bytes]) := by
  unfold fetchCode Vit.vFetchFn; rw [bindOut]; rfl

theorem fetch_fine : Fine (emitGo fetchCode ⟨5, 0, [], [], [], none⟩).2 := by
  rw [fetch_eq]; exact ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetch_sound (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 102400) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit fetchCode) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit fetchCode) ≠ .fault m := by
  have hf := fetch_fine
  rw [fetch_eq] at hf ⊢
  exact sound_of_wp_entry (Run 0 102400) (runArgs dataLen outLen) (by prog_vc (Run 0 102400)) hf rfl h m

def seedCode : Prog Slot Lvl Unit := Vit.vSeedFn
theorem seed_eq : seedCode = (do
    let ptr ← basePtr
    let ctxPtr ← cudaCtxPtr ptr
    let dataPtr ← dataPtr
    let id ← load32 (← absAddr ptr 763972)
    let bytes ← iconst64 102400
    let _ ← ffi .cudaUpload %[ctxPtr, id, dataPtr, bytes]) := by
  unfold seedCode Vit.vSeedFn; rw [bindSeed]; rfl

theorem seed_fine : Fine (emitGo seedCode ⟨5, 0, [], [], [], none⟩).2 := by
  rw [seed_eq]; exact ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem seed_sound (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 102400 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit seedCode) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit seedCode) ≠ .fault m := by
  have hf := seed_fine
  rw [seed_eq] at hf ⊢
  exact sound_of_wp_entry (Run 102400 0) (runArgs dataLen outLen) (by prog_vc (Run 102400 0)) hf rfl h m

theorem bindBase : Vit.VBIND_OFF = 760064 := by native_decide
theorem nbuf : Vit.VNBUF = 2580 := by native_decide

/-- What `main` leaves that `fetchAny` reads, and the lengths the caller
    handed over. -/
abbrev Any (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena 775428, .roomArg .data dl, .roomArg .out ol,
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx]

def anyCode : Prog Slot Lvl Unit := Vit.vFetchAnyFn
theorem any_eq : anyCode = (do
    let ptr ← basePtr
    let ctxPtr ← cudaCtxPtr ptr
    let dataPtr ← dataPtr
    let outPtr ← outPtr
    let dataLen ← dataLen
    let outLen ← outLen
    Prog.when .ule (← iconst64 8) dataLen do
      let idx ← uextend64 (← load32 dataPtr)
      let nb ← uextend64 (← load32 (← iaddImm dataPtr 4))
      Prog.when .ult idx (← iconst64 2580) do
        Prog.when .ule nb outLen do
          let base ← absAddr ptr 760064
          let off ← ishlImm idx 2
          let id ← load32 (← iadd base off)
          let _ ← ffi .cudaDownload %[ctxPtr, id, outPtr, nb]) := by
  unfold anyCode Vit.vFetchAnyFn; rw [bindBase, nbuf]; rfl

theorem any_fine : Fine (emitGo anyCode ⟨5, 0, [], [], [], none⟩).2 := by
  rw [any_eq]; exact ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem any_sound (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Any dataLen outLen) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit anyCode) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit anyCode) ≠ .fault m := by
  have hf := any_fine
  rw [any_eq] at hf ⊢
  exact sound_of_wp_entry (Any dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Any dataLen outLen)) hf rfl h m

end VitSafe

#print axioms VitSafe.fetch_sound
#print axioms VitSafe.seed_sound
#print axioms VitSafe.any_sound
