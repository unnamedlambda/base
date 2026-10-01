module
public import Host.CudaCorpus
meta import Host.CudaCorpus
public import AlgorithmLib.Host.Static
meta import AlgorithmLib.Host.Static
public import AlgorithmLib.Surface.Handles
meta import AlgorithmLib.Surface.Handles
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# Misuse ruled out before the program runs

`Host.Static`'s checker on four bodies that drive the device through streams.
Each accepted one gets a theorem that no run of it misuses a foreign call, on
any input; each refused one is checked to really misuse on some input, so the
refusal is not the checker being cautious.

* `twoStreams true`: two streams write one buffer, ordered by an event.
  Accepted.
* `twoStreams false`: the same without the event. Refused: the second launch
  races the first.
* `onInput true`: a byte of input decides whether a stream launches before the
  download; the stream is synchronised either way. Accepted for every input.
* `onInput false`: the same without the synchronisation. Refused: on an input
  whose byte is nonzero the download races the launch, and the checker, which
  does not know the byte, follows both arms.
-/

open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgStaticChecks

open HProgCudaCorpus (SRC KERNEL MEM image)

/-- The binding slot the launches read. -/
def BA : Nat := 0x700

/-- Launch the add-one kernel over the buffer bound at `BA`, on `s`. -/
def launchOn (ptr : V .i64) (c : Ctx V .cuda) (s : Strm V) : Prog V L (V .i32) := do
  let one ← iconst32 1
  c.launchOn (← iadd ptr (← iconst64 KERNEL)) one (← iadd ptr (← iconst64 BA)) one one one
    (← iconst32 64) one one s

/-- Two streams write one buffer, the second after the first when `ordered`. -/
def twoStreams (ordered : Bool) : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  cudaInit ptr
  let c ← cudaCtx ptr
  let s1 ← c.streamCreate
  let s2 ← c.streamCreate
  let bX ← cudaCreateBuffer ptr (← iconst64 64)
  let _ ← cudaUpload ptr bX (← iconst64 SRC) (← iconst64 64)
  storeI32 bX (← iadd ptr (← iconst64 BA))
  let _ ← launchOn ptr c s1
  if ordered then
    let e ← c.eventCreate
    let _ ← c.eventRecord e s1
    let _ ← c.streamWaitEvent s2 e
  let _ ← launchOn ptr c s2
  let _ ← c.streamSync s2
  let _ ← ffi .cudaDownload %[c.ptr, bX, out, ← iconst64 64]
  cudaCleanup ptr

/-- A byte of input decides whether a stream launches before the download; with
    `synced` the stream is waited for before it. -/
def onInput (synced : Bool) : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  cudaInit ptr
  let c ← cudaCtx ptr
  let s1 ← c.streamCreate
  let bX ← cudaCreateBuffer ptr (← iconst64 64)
  let _ ← cudaUpload ptr bX (← iconst64 SRC) (← iconst64 64)
  storeI32 bX (← iadd ptr (← iconst64 BA))
  let flag ← uload8_64 (← dataPtr)
  when .ne flag (← iconst64 0) do
    let _ ← launchOn ptr c s1
    if synced then
      let _ ← c.streamSync s1
  let _ ← ffi .cudaDownload %[c.ptr, bX, out, ← iconst64 64]
  cudaCleanup ptr

-- ---------------------------------------------------------------------------
-- The world every run starts in, up to data
-- ---------------------------------------------------------------------------

/-- The canonical world: the arena holds the image, the input is eight bytes
    and the output sixty-four, the device is fresh. Nothing about the input's
    bytes is assumed. -/
def canon : Sem.World :=
  { mem := { arena := ⟨image.toArray⟩, data := ByteArray.mk (Array.replicate 8 0),
             out := ByteArray.mk (Array.replicate 64 0) } }

/-- What every run shares with the canonical world: the image. -/
def imageKnown : Static.Known := [(.arena, 0, image.length)]

def args : List Sem.V :=
  [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data), .sc .i64 8,
   .sc .i64 (Sem.regionBase .out), .sc .i64 64]

/-- The checker at the entry, on a body. -/
def check (body : Body) : Bool :=
  Static.effectsOk 1000 (Static.entry args canon imageKnown) (Prog.emit body)

theorem twoStreams_ordered_ok : check (twoStreams true) = true := by native_decide
theorem twoStreams_unordered_refused : check (twoStreams false) = false := by native_decide
theorem onInput_synced_ok : check (onInput true) = true := by native_decide
theorem onInput_unsynced_refused : check (onInput false) = false := by native_decide

/-- **No run of the ordered program misuses a foreign call**, in any world that
    agrees with the canonical one up to data. -/
theorem twoStreams_no_misuse {w : Sem.World} (hw : Static.Same imageKnown w canon) (cfg : Sem.Cfg)
    (m : String) : Sem.run cfg args w (Prog.emit (twoStreams true)) ≠ .misuse m :=
  Static.run_progress twoStreams_ordered_ok hw cfg m

/-- **No input makes the synchronised program misuse a foreign call.** -/
theorem onInput_no_misuse {w : Sem.World} (hw : Static.Same imageKnown w canon) (cfg : Sem.Cfg)
    (m : String) : Sem.run cfg args w (Prog.emit (onInput true)) ≠ .misuse m :=
  Static.run_progress onInput_synced_ok hw cfg m

-- The refusals are real: each refused body misuses on some input.

def isMisuse : Sem.Outcome (List Sem.Obs) → Bool
  | .misuse _ => true
  | _ => false

def runOn (body : Body) (input : UInt8) : Sem.Outcome (List Sem.Obs) :=
  Sem.run { env := (Prog.run body).2.1 } args
    { canon with mem := { canon.mem with data := ByteArray.mk (Array.replicate 8 input) } }
    (Prog.emit body)

#guard isMisuse (runOn (twoStreams false) 0)
#guard isMisuse (runOn (onInput false) 1)
#guard !isMisuse (runOn (onInput false) 0)
#guard !isMisuse (runOn (twoStreams true) 0)
#guard !isMisuse (runOn (onInput true) 1)

end HProgStaticChecks
