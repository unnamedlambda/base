module
public import Host.DriverCorpus
meta import Host.DriverCorpus
public import AlgorithmLib.Host.DevSpec
meta import AlgorithmLib.Host.DevSpec
public import AlgorithmLib.Proof.Typestate
meta import AlgorithmLib.Proof.Typestate
public import AlgorithmLib.Check.Entry
meta import AlgorithmLib.Check.Entry
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

/-!
# A generator proven at the generator

`fillTable` and `sumTrips` exercise the generator's reach: a call
whose answer is bound, a loop carrying an accumulator, and a loop that stores
values nothing knows, whose invariant is its entry typestate without cells.
-/

@[expose] public section

open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace HProgProgHoareChecks

open AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.DevSpec

/-- The entry arguments every body here is run with: eight bytes of input
    and sixty-four of output. -/
def args : List V :=
  [.sc .i64 (regionBase .data), .sc .i64 8, .sc .i64 (regionBase .out), .sc .i64 64]

/-- The world the runs start in: the driver corpus's arena, fresh devices. -/
def canon : World :=
  { mem := { arena := ⟨HProgDriverCorpus.image.toArray⟩, data := ByteArray.mk (Array.replicate 8 0),
             out := ByteArray.mk (Array.replicate 64 0) } }

def isMisuse : Outcome (List Obs) → Bool
  | .misuse _ => true
  | _ => false

/-- A table filled in a loop: the context read once from its slot, a key and a
    value stored to scratch each trip, and one insertion per trip. Every call
    is read by the library function's statement: the slot holds null or a
    table all the way round the loop, and the context is what the slot holds. -/
def fillTable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type} : Prog V L Unit := do
  let base ← basePtr
  let n ← load64 (← dataPtr)
  let slot ← iaddImm base 0
  callLibVoid .htInit %[slot]
  let c ← load64 slot
  let _ ← callLib .htCreate %[c]
  let key ← iaddImm base 0x40
  let val ← iaddImm base 0x80
  let eight ← iconst32 8
  forLoop n fun i => do
    storeI64 i key
    storeI64 i val
    callLibVoid .htInsert %[c, key, eight, val, eight]
  callLibVoid .htCleanup %[slot]

/-- A sum over as many trips as the input says, into the output: a loop that
    carries an accumulator. -/
def sumTrips {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type} : Prog V L Unit := do
  let base ← basePtr
  let n ← load64 (← dataPtr)
  let key ← iaddImm base 0x40
  let sum ← forLoopAcc n (← iconst64 0) fun i acc => do
    storeI64 i key
    iadd acc i
  storeI64 sum (← outPtr)

/-- Where they start: no worker running and no copy in flight, the arena
    holding the slot and the scratch, the input its count, the output its
    answer. -/
abbrev Scratch : World → Prop :=
  Contracts.TState.holds [.part .idle true, .room .arena 0xC0, .room .data 8,
    .room .out 8]

theorem fillTable_fine : Fine (emitGo (fnBody ptrParams (fillTable (V := Slot) (L := Lvl))) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

theorem sumTrips_fine : Fine (emitGo (fnBody ptrParams (sumTrips (V := Slot) (L := Lvl))) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 20000 in
/-- **Filling a table, in a program linked with `Lib.Ht`, never misuses a
    call**, however many entries the input asks for: its calls to the hash
    table are the library's functions, read by their statements. -/
theorem fillTable_no_misuse (cfg : Cfg) [Check.Linked true cfg] (w : World) (h : Scratch w) (m : String) :
    run cfg args w (emit fillTable) ≠ .misuse m := by
  check_body

set_option maxRecDepth 20000 in
/-- **Summing never misuses a call** — it makes none, and stores only where
    its rooms say. -/
theorem sumTrips_no_misuse (cfg : Cfg) (w : World) (h : Scratch w) (m : String) :
    run cfg args w (emit sumTrips) ≠ .misuse m := by
  check_body

/-- Open the first USB device and, where it opened, claim its first
    interface and, where that was granted, read the device descriptor into
    the output through it and release it; then close the device. -/
def usbSession {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type} : Prog V L Unit := do
  let d ← ext (.usb .open) %[← iconst32 0]
  when .ne d (← iconst64 0) do
    let i ← ext (.usb .claim) %[d, ← iconst32 0]
    when .ne i (← iconst64 0) do
      let _ ← ext (.usb .control) %[i, ← iconst32 0x80, ← iconst32 6, ← iconst32 0x100, ← iconst32 0,
        ← outPtr, ← iconst32 18, ← iconst32 1000]
      ext (.usb .release) %[i]
    ext (.usb .close) %[d]

/-- Where it may start: room in the output for the descriptor. Nothing about USB is asked: the device may not be there, or
    may refuse to open. -/
abbrev OutRoom : World → Prop :=
  Contracts.TState.holds [.room .out 18]

theorem usbSession_fine : Fine (emitGo (fnBody ptrParams (usbSession (V := Slot) (L := Lvl))) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- **A USB session never misuses the library**, whatever devices the
    machine has: each call is handed a device or interface it holds, because
    each one opened or claimed was checked not null before it was used, and
    the descriptor lands where there is room. -/
theorem usbSession_no_misuse (cfg : Cfg) (w : World) (h : OutRoom w) (m : String) :
    run cfg args w (emit usbSession) ≠ .misuse m := by
  check_body

-- The check is not idle: without it, a machine with no device to open hands
-- the claim a null device, which is misuse; with it, nothing is claimed.
def usbUnchecked {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type} : Prog V L Unit := do
  let d ← ext (.usb .open) %[← iconst32 0]
  let _ ← ext (.usb .claim) %[d, ← iconst32 0]
  pure ()

def usbRun (checked : Bool) : Outcome (List Obs) :=
  if checked then run { env := (Prog.run (usbSession (V := Slot) (L := Lvl))).2.1 } args canon (emit usbSession)
  else run { env := (Prog.run (usbUnchecked (V := Slot) (L := Lvl))).2.1 } args canon (emit usbUnchecked)

#guard isMisuse (usbRun false)
#guard !isMisuse (usbRun true)

/-- Allocate a block and, where it was allocated, fill the output's first
    eight bytes, copy them to the next eight, and free the block. -/
def heapSession {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type} : Prog V L Unit := do
  let p ← ext (.c .calloc) %[← iconst64 4, ← iconst64 8]
  when .ne p (← iconst64 0) do
    let out ← outPtr
    let _ ← ext (.c .memset) %[out, ← iconst32 0xab, ← iconst64 8]
    let _ ← ext (.c .memcpy) %[← iadd out (← iconst64 8), out, ← iconst64 8]
    let _ ← ext (.c .free) %[p]
    pure ()

/-- Room in the output for the sixteen bytes. -/
abbrev OutRoom16 : World → Prop :=
  Contracts.TState.holds [.room .out 16]

theorem heapSession_fine : Fine (emitGo (fnBody ptrParams (heapSession (V := Slot) (L := Lvl))) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- **A heap session never misuses the C library**: the block freed is one
    allocated and checked not null, the fill and the copy land in room, and
    the copy's ranges are apart. -/
theorem heapSession_no_misuse (cfg : Cfg) (w : World) (h : OutRoom16 w) (m : String) :
    run cfg args w (emit heapSession) ≠ .misuse m := by
  check_body

-- Freeing the block twice is misuse: the second free is handed a block no
-- longer allocated.
def heapTwice {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type} : Prog V L Unit := do
  let p ← ext (.c .calloc) %[← iconst64 4, ← iconst64 8]
  let _ ← ext (.c .free) %[p]
  let _ ← ext (.c .free) %[p]
  pure ()

def heapRun (twice : Bool) : Outcome (List Obs) :=
  if twice then run { env := (Prog.run (heapTwice (V := Slot) (L := Lvl))).2.1 } args canon (emit heapTwice)
  else run { env := (Prog.run (heapSession (V := Slot) (L := Lvl))).2.1 } args canon (emit heapSession)

#guard isMisuse (heapRun true)
#guard !isMisuse (heapRun false)

end HProgProgHoareChecks
