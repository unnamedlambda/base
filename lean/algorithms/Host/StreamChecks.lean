module
public import Host.DriverCorpus
meta import Host.DriverCorpus
public import AlgorithmLib.Check.Entry
meta import AlgorithmLib.Check.Entry
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# What the checker refuses, and what it accepts, about streams

Each body calls the CUDA driver directly: it brings the driver up, makes one
allocation, copies sixty-four bytes from the arena into it, loads the add-one
kernel (`HProgDriverCorpus.addOnePtx`), and launches it on created streams over
the allocation before copying it back to the caller's output.

* `twoStreams true`: two streams write the allocation, ordered by an event.
  Accepted.
* `twoStreams false`: the same without the event. Refused: the second launch
  races the first.
* `onInput true`: a byte of input decides whether a stream launches before the
  copy back; the stream is synchronised either way. Accepted for every input.
* `onInput false`: the same without the synchronisation. Refused: on an input
  whose byte is nonzero the copy races the launch, and the checker, which does
  not know the byte, follows both arms.
* `byInput true`: a stream launches as many times as the first input byte
  says, then is waited for. The checker carries the stream's readiness round
  the loop: at the head the host knows every access, or the stream does, and
  either lets the next launch on the stream go. Accepted; without the wait,
  refused.

Each refusal is real: the refused body misuses the driver on some input, run
by the interpreter from the corpus's world.
-/

open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgStreamChecks

open HProgDriverCorpus (S_CTX S_A S_MOD S_F S_STREAM S_C2 S_E1 PARAMS SRC NAME_MAIN PTX1 PTX2 MEM image oracle)

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- Where the launch's parameter cell is, and the array of one pointer to it. -/
def CELL : Nat := PARAMS
def ARR : Nat := PARAMS + 8

/-- Bring the driver up, make the allocation and fill it, load the kernel, and
    write its one parameter. -/
def prelude (base : V .i64) : Prog V L Unit := do
  let z32 ← iconst32 0
  let _ ← ext (.cuda .init) %[z32]
  let _ ← ext (.cuda .primaryCtxRetain) %[← iaddImm base S_CTX, z32]
  let _ ← ext (.cuda .ctxSetCurrent) %[← load64 (← iaddImm base S_CTX)]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base S_A, ← iconst64 64]
  let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base S_A), ← iaddImm base SRC, ← iconst64 64]
  let _ ← ext (.cuda .moduleLoadData) %[← iaddImm base S_MOD, ← iaddImm base PTX1]
  let _ ← ext (.cuda .moduleGetFunction) %[← iaddImm base S_F, ← load64 (← iaddImm base S_MOD),
    ← iaddImm base NAME_MAIN]
  storeI64 (← load64 (← iaddImm base S_A)) (← iaddImm base CELL)
  storeI64 (← iaddImm base CELL) (← iaddImm base ARR)

/-- Make a non-blocking stream, its handle at `slot`. -/
def stream (base : V .i64) (slot : Nat) : Prog V L Unit := do
  let _ ← ext (.cuda .streamCreate) %[← iaddImm base slot, ← iconst32 1]

/-- Launch the add-one kernel over the allocation, on the stream at `slot`. -/
def launchOn (base : V .i64) (slot : Nat) : Prog V L Unit := do
  let one ← iconst32 1
  let _ ← ext (.cuda .launchKernel) %[← load64 (← iaddImm base S_F), one, one, one, ← iconst32 64, one, one,
    ← iconst32 0, ← load64 (← iaddImm base slot), ← iaddImm base ARR, ← iconst64 0]

/-- Copy the allocation back to the caller's output. -/
def back (base out : V .i64) : Prog V L Unit := do
  let _ ← ext (.cuda .memcpyDtoH) %[out, ← load64 (← iaddImm base S_A), ← iconst64 64]

/-- Two streams write the allocation, the second after the first when
    `ordered`. -/
def twoStreams (ordered : Bool) : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  prelude base
  stream base S_STREAM
  stream base S_C2
  launchOn base S_STREAM
  if ordered then
    let _ ← ext (.cuda .eventCreate) %[← iaddImm base S_E1, ← iconst32 0]
    let _ ← ext (.cuda .eventRecord) %[← load64 (← iaddImm base S_E1), ← load64 (← iaddImm base S_STREAM)]
    let _ ← ext (.cuda .streamWaitEvent) %[← load64 (← iaddImm base S_C2), ← load64 (← iaddImm base S_E1),
      ← iconst32 0]
  launchOn base S_C2
  let _ ← ext (.cuda .streamSynchronize) %[← load64 (← iaddImm base S_C2)]
  back base out

/-- A byte of input decides whether a stream launches before the copy back;
    with `synced` the stream is waited for before it. -/
def onInput (synced : Bool) : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  prelude base
  stream base S_STREAM
  let flag ← uload8_64 (← dataPtr)
  when .ne flag (← iconst64 0) do
    launchOn base S_STREAM
  if synced then
    let _ ← ext (.cuda .streamSynchronize) %[← load64 (← iaddImm base S_STREAM)]
  back base out

/-- Launch on a stream as many times as the first input byte says; with
    `synced`, wait for the stream before copying back. -/
def byInput (synced : Bool) : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  prelude base
  stream base S_STREAM
  let n ← uload8_64 (← dataPtr)
  forLoop n fun _ => launchOn base S_STREAM
  if synced then
    let _ ← ext (.cuda .streamSynchronize) %[← load64 (← iaddImm base S_STREAM)]
  back base out

-- ---------------------------------------------------------------------------
-- The world every run starts in
-- ---------------------------------------------------------------------------

def args : List Sem.V :=
  [.sc .i64 (Sem.regionBase .data), .sc .i64 8,
   .sc .i64 (Sem.regionBase .out), .sc .i64 64]

/-- Where the bodies may start: the driver loaded and nothing made, memory not
    frozen, kernels that keep their buffers' sizes, the arena holding the
    kernel's text and its entry point's name, and room in the output for the
    copy back. Nothing about the input's bytes is asked. -/
def device0 : Contracts.TState :=
  [.part (.has .cuda) true, .part .capturing false, .part .frozen false, .part .idle true, .devSeq, .oracles,
   .retained 0, .nbufs 0, .nmods 0, .nfuncs 0, .streams [], .events 0,
   .room .arena MEM, .room .out 64,
   .text (Sem.regionBase .arena + UInt64.ofNat NAME_MAIN) (some "main") 5 [],
   .text (Sem.regionBase .arena + UInt64.ofNat PTX1) none (PTX2 - PTX1) [("main", [8])]]

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
#guard check (byInput true)
#guard !check (byInput false)

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- **No run of the ordered program misuses a call**, from any world in
    `Device0`. -/
theorem twoStreams_no_misuse (cfg : Sem.Cfg) (w : Sem.World) (h : Device0 w) (m : String) :
    Sem.run cfg args w (Prog.emit (twoStreams true)) ≠ .misuse m := by
  check_body

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- **No input makes the synchronised program misuse a call.** -/
theorem onInput_no_misuse (cfg : Sem.Cfg) (w : Sem.World) (h : Device0 w) (m : String) :
    Sem.run cfg args w (Prog.emit (onInput true)) ≠ .misuse m := by
  check_body

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- **No run misuses a call, however many times the input says to launch.** -/
theorem byInput_no_misuse (cfg : Sem.Cfg) (w : Sem.World) (h : Device0 w) (m : String) :
    Sem.run cfg args w (Prog.emit (byInput true)) ≠ .misuse m := by
  check_body

-- The refusals are real: each refused body misuses the driver on some input.

def isMisuse : Sem.Outcome (List Sem.Obs) → Bool
  | .misuse _ => true
  | _ => false

def runOn (body : Body) (input : UInt8) : Sem.Outcome (List Sem.Obs) :=
  Sem.run { env := (Prog.run body).2.1 } args
    { mem := { arena := ⟨image.toArray⟩, data := ByteArray.mk (Array.replicate 8 input),
               out := ByteArray.mk (Array.replicate 64 0) },
      kernel := oracle }
    (Prog.emit body)

#guard isMisuse (runOn (twoStreams false) 0)
#guard !isMisuse (runOn (twoStreams true) 0)
#guard isMisuse (runOn (onInput false) 1)
#guard !isMisuse (runOn (onInput false) 0)
#guard !isMisuse (runOn (onInput true) 1)
#guard isMisuse (runOn (byInput false) 1)
#guard isMisuse (runOn (byInput false) 200)
#guard !isMisuse (runOn (byInput false) 0)
#guard !isMisuse (runOn (byInput true) 200)

end HProgStreamChecks
