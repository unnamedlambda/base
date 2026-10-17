module
public import Host.CudaCorpus
meta import Host.CudaCorpus
public import AlgorithmLib.Check.Entry
meta import AlgorithmLib.Check.Entry
public import AlgorithmLib.Surface.Handles
meta import AlgorithmLib.Surface.Handles
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# Misuse ruled out before the program runs

The checker (`Check`) on four bodies that drive the device through streams,
from any world in the `Device0` typestate. Each accepted one gets a theorem
that no run of it misuses a foreign call, on any input; each refused one is
checked to really misuse on some input, so the refusal is not the checker
being cautious.

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
-- The world every run starts in
-- ---------------------------------------------------------------------------

/-- The canonical world: the arena holds the image, the input is eight bytes
    and the output sixty-four, the device is fresh. -/
def canon : Sem.World :=
  { mem := { arena := ⟨image.toArray⟩, data := ByteArray.mk (Array.replicate 8 0),
             out := ByteArray.mk (Array.replicate 64 0) } }

def args : List Sem.V :=
  [.sc .i64 (Sem.regionBase .data), .sc .i64 8,
   .sc .i64 (Sem.regionBase .out), .sc .i64 64]

/-- Where the bodies may start: a device to open, memory not frozen, kernels
    that keep their buffers' sizes, the arena holding the image's slots and
    the kernel's text, and room in the output for the download. Nothing about
    the input's bytes is asked. -/
def device0 : Contracts.TState :=
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena MEM, .room .out 64,
    .cstrIn (Sem.regionBase .arena + KERNEL.toUInt64) (0x300 - KERNEL)]

abbrev Device0 : Sem.World → Prop := device0.holds

/-- Whether the checker accepts a body from `device0`. -/
def check (body : Body) : Bool :=
  match Check.certOrRefusal true (Check.entryKnown device0 args) (Prog.emit body) with
  | .ok _ => true
  | .error _ => false

#guard check (twoStreams true)
#guard !check (twoStreams false)
#guard check (onInput true)
#guard !check (onInput false)

/-- **No run of the ordered program misuses a foreign call**, from any world
    in `Device0`. -/
theorem twoStreams_no_misuse (cfg : Sem.Cfg) (w : Sem.World) (h : Device0 w) (m : String) :
    Sem.run cfg args w (Prog.emit (twoStreams true)) ≠ .misuse m := by
  check_body

/-- **No input makes the synchronised program misuse a foreign call.** -/
theorem onInput_no_misuse (cfg : Sem.Cfg) (w : Sem.World) (h : Device0 w) (m : String) :
    Sem.run cfg args w (Prog.emit (onInput true)) ≠ .misuse m := by
  check_body

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
