module
public import Host.StaticChecks
meta import Host.StaticChecks
public import AlgorithmLib.Host.DevSpec
meta import AlgorithmLib.Host.DevSpec
public import AlgorithmLib.Proof.Typestate
meta import AlgorithmLib.Proof.Typestate
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

/-!
# A generator proven at the generator

`syncLoop` synchronises the live context's default stream as many times as the
first input byte says. `syncLoop_no_misuse` says no run of `emit syncLoop`
misuses a foreign call, for any input, from any world in the `Live` typestate.
Its proof is `prog_vc Live`: the typestate is the loop's invariant and the
call's contract, and the weakest precondition does the rest. The one check on
the emission is `syncLoop_fine`, decided on the finished code.

`syncLoop_pt` is the same fact by the construct rules, one per construct, with
the loop's invariant and the argument slots written out.

`ctxLoop` is the shape generators have: it opens the context into its slot of
shared memory, reads the handle back from the slot before each call, and closes
it. `ctxLoop_no_misuse` asks only that memory is not frozen and the arena holds
the slots; the typestate learns the slot's value from `cudaInit`, carries it
through the loop, and the loads read it.

`fillTable` and `sumTrips` exercise the rest of the generator's reach: a call
whose answer is bound, a loop carrying an accumulator, and a loop that stores
values nothing knows, whose invariant is its entry typestate without cells.
-/

@[expose] public section

open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace HProgProgHoareChecks

/-- Synchronise the default stream of the live context as many times as the
    first input byte says. -/
def syncLoop {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type} : Prog V L Unit := do
  let n ← uload8_64 (← dataPtr)
  let z ← iconst64 0x6000000000
  let sid ← iconst32 (-1)
  forLoop n fun _ => ffiVoid .cudaStreamSync %[z, sid]

open AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.DevSpec

abbrev vCtx : V := .sc .i64 0x6000000000
abbrev vSid : V := ofInt .i32 (-1)

theorem sync_call (cfg : Cfg) (w : World) (h : Live w) :
    ∃ r w', callOf cfg.locals (.ffi .cudaStreamSync) [vCtx, vSid]
      (obsCall w (.ffi .cudaStreamSync) [vCtx, vSid]) = some (r, w') ∧ Live w' :=
  (by prog_keeps : Keeps cfg Live .cudaStreamSync [vCtx, vSid]) w h

theorem push_get {Γ : Env} {i : Nat} {v x : V} (h : Γ[i]? = some x) : (Γ.push v)[i]? = some x := by
  have hi : i < Γ.size := by
    rcases Nat.lt_or_ge i Γ.size with hi | hi
    · exact hi
    · rw [Array.getElem?_eq_none hi] at h; cases h
  rw [Array.getElem?_push_lt hi, ← h, Array.getElem?_eq_getElem hi]

theorem push_get_size {Γ : Env} {v : V} : (Γ.push v)[Γ.size]? = some v := by simp

theorem bindAt_get {Γ : Env} {n i : Nat} {vs : List V} {x : V} (hn : Γ.size ≤ n) (h : Γ[i]? = some x) :
    (bindAt Γ n vs)[i]? = some x := by
  have hi : i < Γ.size := by
    rcases Nat.lt_or_ge i Γ.size with hi | hi
    · exact hi
    · rw [Array.getElem?_eq_none hi] at h; cases h
  rw [bindAt_lt (by omega) hi, ← h]

/-- The two values every call reads, where the loop was entered. -/
def Args (z sid : Nat) (Γ : Env) : Prop := Γ[z]? = some vCtx ∧ Γ[sid]? = some vSid

theorem Args.push {z sid : Nat} {Γ : Env} (h : Args z sid Γ) (v : V) : Args z sid (Γ.push v) :=
  ⟨push_get h.1, push_get h.2⟩

theorem Args.bindAt {z sid n : Nat} {Γ : Env} (h : Args z sid Γ) (hn : Γ.size ≤ n) (vs : List V) :
    Args z sid (bindAt Γ n vs) :=
  ⟨bindAt_get hn h.1, bindAt_get hn h.2⟩

/-- **The generator, proven at the generator**: from any world where the device
    is not capturing and memory is not frozen, every call it makes answers. -/
theorem syncLoop_pt (cfg : Cfg) :
    PT cfg 0 { ok := fun _ _ => True } (fun _ w => Live w) (syncLoop (V := Slot) (L := Lvl))
      (fun _ _ _ => True) := by
  unfold syncLoop
  refine PT.bind (Q := fun _ _ w => Live w) ?_ fun p => ?_
  · unfold dataPtr; exact PT.params (PT.ret fun _ _ h => h)
  refine PT.bind (Q := fun _ _ w => Live w) ?_ fun n => ?_
  · refine PT.op fun r => PT.ret ?_
    rintro _ _ ⟨_, _, rfl, _, h, _⟩; exact h
  refine PT.bind (Q := fun z Γ w => Γ[z]? = some vCtx ∧ Live w) ?_ fun z => ?_
  · refine PT.op fun r => PT.ret ?_
    rintro _ _ ⟨Γ0, v, rfl, hr, h, hv⟩
    simp only [Op'.erase, evalOp, Option.some.injEq] at hv; subst hv; subst hr
    exact ⟨push_get_size.trans (by rfl), h⟩
  refine PT.bind (Q := fun sid Γ w => Args z sid Γ ∧ Live w) ?_ fun sid => ?_
  · refine PT.op fun r => PT.ret ?_
    rintro _ _ ⟨Γ0, v, rfl, hr, ⟨hz, h⟩, hv⟩
    simp only [Op'.erase, evalOp, Option.some.injEq] at hv; subst hv; subst hr
    exact ⟨⟨push_get hz, push_get_size.trans (by rfl)⟩, h⟩
  unfold forLoop wloop1 wloop wloopL
  refine PT.bind (Q := fun i0 Γ w => Args z sid Γ ∧ Live w) ?_ fun i0 => ?_
  · refine PT.op fun r => PT.ret ?_
    rintro _ _ ⟨Γ0, v, rfl, _, ⟨ha, h⟩, _⟩
    exact ⟨ha.push v, h⟩
  refine PT.bind (Q := fun _ _ w => Live w) ?_ fun _ => PT.ret fun _ _ _ => trivial
  refine PT.loop (fun Γ0 _ w => Args z sid Γ0 ∧ Live w)
    (fun Γ0 cs _ Γ1 w => Γ1 = bindAt Γ0 Γ0.size cs ∧ Args z sid Γ0 ∧ Live w)
    (fun _ _ _ w => Live w) ?_ ?_ ?_ ?_ ?_
  · rintro Γ w cs ⟨ha, h⟩ _; exact ⟨ha, h⟩
  · intro n0 Γ0 cs
    refine PT.ret ?_
    rintro Γh wh ⟨hs, rfl, hI⟩
    exact ⟨by rw [hs], hI⟩
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨_, _, h⟩ _ _ _; exact h
  · -- One trip, by its weakest precondition: no assertion between statements.
    rintro nb Γ0 cs ⟨cnd, ex, u⟩
    refine PT.conseq (wp_sound cfg _ _ _ _) ?_ (fun _ _ _ h => h)
    rintro _ w ⟨Γ1, t, f, hs1, rfl, ⟨rfl, ha, h⟩, _, _⟩
    have hA : Args z sid (bindAt (Array.push (bindAt Γ0 Γ0.size cs) (V.sc t f)) nb cs) := by
      refine ((ha.bindAt (Nat.le_refl _) cs).push _).bindAt ?_ cs
      rw [Array.size_push]; omega
    simp only [ffiVoid, ffi, iaddImm, iconst64, iconst, iadd, op, wp_bind, wp_call, wp_op, wp_ret,
      Vals.slots, List.mapM_cons, List.mapM_nil, hA.1, hA.2]
    intro vs hvs
    simp at hvs; subst hvs
    obtain ⟨r, w', hc, h'⟩ := sync_call cfg w h
    exact ⟨r, w', hc, by simp; exact fun _ _ _ _ _ _ _ _ => ⟨ha, h'⟩⟩
  · intro nE
    exact PT.ret fun _ _ ⟨_, _, _, _, _, h⟩ => h

/-- The one check on the finished emission: nothing failed, and the code fits
    the fuel the model counts slots with. -/
theorem syncLoop_fine : Fine (emitGo (syncLoop (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨rfl, by decide⟩

/-- **What ships never misuses a call**: for every input, however many trips the
    first byte asks for, from any world where the device is not capturing. No
    part of the program is run to show it, and nothing about it is written but
    its typestate. -/
theorem syncLoop_no_misuse (cfg : Cfg) (args : List V) (w : World) (hargs : args.length = 5)
    (h : Live w) (m : String) : run cfg args w (emit syncLoop) ≠ .misuse m :=
  safe_of_wp Live (by prog_vc Live) syncLoop_fine hargs h m

/-- The same, from the construct rules. -/
theorem syncLoop_no_misuse_by_rules (cfg : Cfg) (args : List V) (w : World) (hargs : args.length = 5)
    (h : Live w) (m : String) : run cfg args w (emit syncLoop) ≠ .misuse m :=
  run_safe (emit_triple (params := ptrParams) (syncLoop_pt cfg) syncLoop_fine
    (fun _ _ ⟨e1, e2⟩ => by subst e1 e2; exact ⟨by simp [hargs, ptrParams], h⟩)) m

-- The precondition is not idle: a device capturing on the default stream
-- refuses the first synchronisation, and an input that asks for no trips makes
-- none.
open HProgStaticChecks (canon args) in
def runIn (capturing : Bool) (input : UInt8) : Outcome (List Obs) :=
  run { env := (Prog.run (syncLoop (V := Slot) (L := Lvl))).2.1 } args
    { canon with
      mem := { canon.mem with data := ByteArray.mk (Array.replicate 8 input) }
      dev := { canon.dev with
        live := true
        capture := if capturing then some { origin := defaultParty, joined := [], race := {}, ops := #[] }
          else none } }
    (emit syncLoop)

def isMisuse : Outcome (List Obs) → Bool
  | .misuse _ => true
  | _ => false

#guard isMisuse (runIn true 1)
#guard isMisuse (runIn true 200)
#guard !isMisuse (runIn true 0)
#guard !isMisuse (runIn false 200)

/-- Open the CUDA context into its slot of shared memory, synchronise its
    default stream as many times as the first input byte says, reading the
    context back from the slot each time as every generator does, and close
    it. -/
def ctxLoop {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type} : Prog V L Unit := do
  let base ← basePtr
  cudaInit base
  let n ← uload8_64 (← dataPtr)
  let sid ← iconst32 (-1)
  forLoop n fun _ => do
    let ctx ← cudaCtxPtr base
    ffiVoid .cudaStreamSync %[ctx, sid]
  cudaCleanup base

/-- Where `ctxLoop` may start: host memory is not frozen, and the arena holds
    the context slots. Nothing about the device is asked: the program opens
    the context itself. -/
abbrev Arena : World → Prop :=
  Contracts.TState.holds [.part .cudaDevice true, .part .frozen false, .room .arena 64]

theorem ctxLoop_fine : Fine (emitGo (ctxLoop (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨rfl, by decide⟩

open HProgStaticChecks (args) in
/-- **A generator that opens, uses and closes a context never misuses a
    call**, for any input, from any world whose arena holds the context
    slots. The context each call is handed is the one `cudaInit` wrote, read
    back from its slot: the typestate carries the slot's value through the
    loop, and nothing but the typestate is written. -/
theorem ctxLoop_no_misuse (cfg : Cfg) (w : World) (h : Arena w) (m : String) :
    run cfg args w (emit ctxLoop) ≠ .misuse m :=
  safe_of_wp_entry Arena args (by prog_vc Arena) ctxLoop_fine rfl h m

/-- A table filled in a loop: the context read once from its slot, a key and a
    value stored to scratch each trip, and one insertion per trip. Every call
    answers, which the condition generator reaches through a bound answer
    (`htCreate`'s), a store of values it does not know, and a loop whose
    invariant is the entry typestate less its cells. -/
def fillTable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type} : Prog V L Unit := do
  let base ← basePtr
  let n ← load64 (← dataPtr)
  let slot ← iaddImm base 0
  ffiVoid .htInit %[slot]
  let c ← load64 slot
  let _ ← ffi .htCreate %[c]
  let key ← iaddImm base 0x40
  let val ← iaddImm base 0x80
  let eight ← iconst32 8
  forLoop n fun i => do
    storeI64 i key
    storeI64 i val
    ffiVoid .htInsert %[c, key, eight, val, eight]
  ffiVoid .htCleanup %[slot]

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

/-- Where they start: memory not frozen, the arena holding the slot and the
    scratch, the input its count, the output its answer. -/
abbrev Scratch : World → Prop :=
  Contracts.TState.holds [.part .frozen false, .room .arena 0xC0, .room .data 8, .room .out 8]

theorem fillTable_fine : Fine (emitGo (fillTable (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨rfl, by decide⟩

theorem sumTrips_fine : Fine (emitGo (sumTrips (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨rfl, by decide⟩

open HProgStaticChecks (args) in
set_option maxRecDepth 20000 in
/-- **Filling a table never misuses a call**, however many entries the input
    asks for. -/
theorem fillTable_no_misuse (cfg : Cfg) (w : World) (h : Scratch w) (m : String) :
    run cfg args w (emit fillTable) ≠ .misuse m :=
  safe_of_wp_entry Scratch args (by prog_vc Scratch) fillTable_fine rfl h m

open HProgStaticChecks (args) in
set_option maxRecDepth 20000 in
/-- **Summing never misuses a call** — it makes none, and stores only where
    its rooms say. -/
theorem sumTrips_no_misuse (cfg : Cfg) (w : World) (h : Scratch w) (m : String) :
    run cfg args w (emit sumTrips) ≠ .misuse m :=
  safe_of_wp_entry Scratch args (by prog_vc Scratch) sumTrips_fine rfl h m

/-- Open the first USB device and, where it opened, claim its first
    interface, read its device descriptor into the output, release the
    interface and close the device. -/
def usbSession {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type} : Prog V L Unit := do
  let d ← ext (.usb .open) %[← iconst32 0]
  when .ne d (← iconst64 0) do
    let iface ← iconst32 0
    let _ ← ext (.usb .claim) %[d, iface]
    let _ ← ext (.usb .control) %[d, ← iconst32 0x80, ← iconst32 6, ← iconst32 0x100, ← iconst32 0,
      ← outPtr, ← iconst32 18, ← iconst32 1000]
    let _ ← ext (.usb .release) %[d, iface]
    let _ ← ext (.usb .close) %[d]
    pure ()

/-- Where it may start: memory not frozen, and room in the output for the
    descriptor. Nothing about USB is asked: the device may not be there, or
    may refuse to open. -/
abbrev OutRoom : World → Prop :=
  Contracts.TState.holds [.part .frozen false, .room .out 18]

theorem usbSession_fine : Fine (emitGo (usbSession (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨rfl, by decide⟩

open HProgStaticChecks (args) in
/-- **A USB session never misuses the library**, whatever devices the
    machine has: each call is handed a device it holds, because the one it
    opened was checked not null before it was used, and the descriptor lands
    where there is room. -/
theorem usbSession_no_misuse (cfg : Cfg) (w : World) (h : OutRoom w) (m : String) :
    run cfg args w (emit usbSession) ≠ .misuse m :=
  safe_of_wp_entry OutRoom args (by prog_vc OutRoom) usbSession_fine rfl h m

-- The check is not idle: without it, a machine with no device to open hands
-- the claim a null device, which is misuse; with it, nothing is claimed.
def usbUnchecked {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type} : Prog V L Unit := do
  let d ← ext (.usb .open) %[← iconst32 0]
  let _ ← ext (.usb .claim) %[d, ← iconst32 0]
  pure ()

open HProgStaticChecks (canon args) in
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
  Contracts.TState.holds [.part .frozen false, .room .out 16]

theorem heapSession_fine : Fine (emitGo (heapSession (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨rfl, by decide⟩

open HProgStaticChecks (args) in
/-- **A heap session never misuses the C library**: the block freed is one
    allocated and checked not null, the fill and the copy land in room, and
    the copy's ranges are apart. -/
theorem heapSession_no_misuse (cfg : Cfg) (w : World) (h : OutRoom16 w) (m : String) :
    run cfg args w (emit heapSession) ≠ .misuse m :=
  safe_of_wp_entry OutRoom16 args (by prog_vc OutRoom16) heapSession_fine rfl h m

-- Freeing the block twice is misuse: the second free is handed a block no
-- longer allocated.
def heapTwice {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type} : Prog V L Unit := do
  let p ← ext (.c .calloc) %[← iconst64 4, ← iconst64 8]
  let _ ← ext (.c .free) %[p]
  let _ ← ext (.c .free) %[p]
  pure ()

open HProgStaticChecks (canon args) in
def heapRun (twice : Bool) : Outcome (List Obs) :=
  if twice then run { env := (Prog.run (heapTwice (V := Slot) (L := Lvl))).2.1 } args canon (emit heapTwice)
  else run { env := (Prog.run (heapSession (V := Slot) (L := Lvl))).2.1 } args canon (emit heapSession)

#guard isMisuse (heapRun true)
#guard !isMisuse (heapRun false)

end HProgProgHoareChecks
