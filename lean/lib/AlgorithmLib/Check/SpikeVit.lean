module
public import AlgorithmLib.Check.Absint
public import AlgorithmLib.Surface.ProgFFI
@[expose] public section

/-!
# The ViT's seed and fetch, checked

The bodies `Ship.VitSafe` proves with `prog_vc`, written out as `VitSafe`'s
`fetch_eq` and `seed_eq` state them, and proven here by `chk` alone: the only
proof per body is that the kernel evaluates the checker to `true`.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog

namespace AlgorithmLib.Check.SpikeVit

open AlgorithmLib.HProg.Contracts (TState)

/-- What `main` leaves that `fetch` and `seed` read. -/
def run (d o : Nat) : TState :=
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena 775428, .room .data d, .room .out o,
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx]

def fetchCode : Prog Slot Lvl Unit := do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← outPtr
  let id ← load32 (← absAddr ptr 763968)
  let bytes ← iconst64 102400
  let _ ← ffi .cudaDownload %[ctxPtr, id, outPtr, bytes]

def seedCode : Prog Slot Lvl Unit := do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let dataPtr ← dataPtr
  let id ← load32 (← absAddr ptr 763972)
  let bytes ← iconst64 102400
  let _ ← ffi .cudaUpload %[ctxPtr, id, dataPtr, bytes]

theorem fetch_fine : Fine (emitGo fetchCode ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

theorem fetch_sound (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (run 0 102400).holds w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit fetchCode) ≠ .misuse m ∧
      Sem.run cfg (runArgs dataLen outLen) w (emit fetchCode) ≠ .fault m :=
  sound_of_chk (run 0 102400) dataLen outLen (by decide +kernel) fetch_fine h m

theorem seed_fine : Fine (emitGo seedCode ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

theorem seed_sound (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (run 102400 0).holds w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit seedCode) ≠ .misuse m ∧
      Sem.run cfg (runArgs dataLen outLen) w (emit seedCode) ≠ .fault m :=
  sound_of_chk (run 102400 0) dataLen outLen (by decide +kernel) seed_fine h m

/-- The checker refuses what it should: `fetch` from a typestate that does not
    give the output room. -/
example : chk (entryOf (run 0 100)) fetchCode (fun _ _ => true) = false := by decide +kernel

end AlgorithmLib.Check.SpikeVit
