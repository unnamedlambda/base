module
public import AlgorithmLib.Host.Contracts
meta import AlgorithmLib.Host.Contracts
public import AlgorithmLib.Host.DevTrack
meta import AlgorithmLib.Host.DevTrack
public import AlgorithmLib.Host.DevSeq
meta import AlgorithmLib.Host.DevSeq
public import AlgorithmLib.Host.DevBufs
meta import AlgorithmLib.Host.DevBufs
public import AlgorithmLib.Host.DevReady
meta import AlgorithmLib.Host.DevReady
public import AlgorithmLib.Host.HtRep
meta import AlgorithmLib.Host.HtRep
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Lifecycle` — the entry points' contracts

`callBits` is the one reading of the Rust entry points. What a frontend may
rely on about an entry point is defined and proven of that reading here;
`Host.Contract` gathers it, with the same of the C libraries, into one row per
callee (`Callee.contract`, `contract_sound`):

* `spec` --- the ABI, the memory the call may write, and what it computes
  (`Ffi.spec`); `callFfi_respects_frame` is the proof for the memory;
* `pre` --- when it answers (`Contracts.Pre`, `pre_safe`);
* `moves` --- which *lifecycle parts* it may change: whether the CUDA context is
  live or capturing, whether memory is frozen for a worker, and whether the
  LMDB, window, wgpu and thread contexts are live (`callBits_parts`);
* `after` --- what a call with given argument bits leaves of a *typestate*
  (`TState`), a list of facts of three kinds: a lifecycle part's value, that a
  region is at least so large (a *room*), and what an eight-byte *cell* of
  memory holds (`after_sound`). A call keeps the parts it does not move, every
  room outside pinned memory, and every cell its memory frame cannot reach
  (`keeps`). A lifecycle call adds what it makes: its part, and the handle it
  writes to its slot, so `cudaInit` at a slot leaves the slot holding the
  context. A cleanup whose slot the typestate knows is resolved by it: a slot
  holding zero is left alone, one holding the handle ends the lifecycle. What
  depends on state no fact names, such as a capture beginning or a window
  opening on a display, is left unknown.

The same facts give the memory preconditions: a room gives the slots, reads
and writes inside it (`roomAt`, `slot8_of_state`, `readable_of_state`,
`writable_of_state`), and a cell gives the value a load reads. A program's own
store moves a typestate too (`afterStore`).

A typestate is therefore carried through every call by computing `after`, with
nothing proved about the call. `callOf_ffi` connects the table to the calls
programs make: outside a worker's lifetime every entry point but `threadSpawn`
is `callBits` on its argument bits.

What every call keeps is one structure, `Keeps`, proven of each entry point by
following its contract to every answer it gives (`walk`) and closing each field
there (`keeps_close`); `derive_keeps` states and proves it for every
constructor of `Ffi`, so a new entry point adds no proof here.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec
  AlgorithmLib.Device

namespace AlgorithmLib.HProg.Contracts

theorem store_frozen {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m') :
    m'.frozen = m.frozen := by
  unfold Mem.store at h
  obtain ⟨⟨r, off⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  · cases h; unfold Mem.setRegion; split <;> rfl

theorem copyIn_frozen {m m' : Mem} {a : UInt64} {src : ByteArray} (h : copyIn m a src = some m') :
    m'.frozen = m.frozen := (copyIn_same h).frozen.symm

theorem store_busy {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m') :
    m'.busy = m.busy := by
  unfold Mem.store at h
  obtain ⟨⟨r, off⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  · cases h; unfold Mem.setRegion; split <;> rfl

theorem copyIn_busy {m m' : Mem} {a : UInt64} {src : ByteArray} (h : copyIn m a src = some m') :
    m'.busy = m.busy := (copyIn_same h).busy.symm

theorem put_live (d : Dev) (id : Nat) (b : Option ByteArray) : (d.put id b).live = d.live := rfl

theorem putAll_live : ∀ (ws : List (Nat × ByteArray)) (d : Dev),
    (ws.foldl (fun d (x : Nat × ByteArray) => d.put x.1 (some x.2)) d).live = d.live
  | [], _ => rfl
  | p :: ws, d => putAll_live ws (d.put p.1 (some p.2))

theorem runLaunch_live {d d' : Dev} {k : Launch → List ByteArray → List ByteArray} {l : Launch}
    {ids : List Nat} (h : d.runLaunch k l ids = some d') : d'.live = d.live := by
  unfold Dev.runLaunch at h
  obtain ⟨ins, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  · cases h; exact putAll_live _ d

theorem runVendor_live {d d' : Dev} {v : VendorCall → List ByteArray → ByteArray} {c : VendorCall}
    {ins : List Nat} {out : Nat} (h : d.runVendor v c ins out = some d') : d'.live = d.live := by
  unfold Dev.runVendor at h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  · cases h; rfl

theorem devOp_live {d d' : Dev} {w : World} {p : Nat} {op : DevOp} {rs ws : List Nat}
    (h : d.devOp w p op rs ws = some d') : d'.live = d.live := by
  have run_live : ∀ {d'}, Dev.devOp.run d w p op rs ws = some d' → d'.live = d.live := by
    intro d' h
    unfold Dev.devOp.run at h
    obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    split at h
    · have := runLaunch_live h; exact this
    · have := runVendor_live h; exact this
  unfold Dev.devOp at h
  split at h
  · split at h
    · obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
      cases h; rfl
    · exact run_live h
  · exact run_live h

theorem syncWrite_eq {d d' : Dev} {id : Nat} {b : ByteArray} (h : d.syncWrite id b = some d') :
    d'.live = d.live ∧ d'.capture = d.capture := by
  unfold Dev.syncWrite at h
  obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
  cases h; exact ⟨rfl, rfl⟩

theorem syncRead_eq {d d' : Dev} {id : Nat} (h : d.syncRead id = some d') :
    d'.live = d.live ∧ d'.capture = d.capture := by
  unfold Dev.syncRead at h
  obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
  cases h; exact ⟨rfl, rfl⟩

theorem devOp_capturing {d d' : Dev} {w : World} {p : Nat} {op : DevOp} {rs ws : List Nat}
    (h : d.devOp w p op rs ws = some d') : d'.capture.isSome = d.capture.isSome := by
  unfold Dev.devOp at h
  have run_cap : ∀ {d'}, Dev.devOp.run d w p op rs ws = some d' → d'.capture = d.capture := by
    intro d' h
    unfold Dev.devOp.run at h
    obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    split at h
    · have := (runLaunch_meta h).2.1; exact this
    · have := (runVendor_meta h).2.1; exact this
  split at h
  · rename_i c hc
    split at h
    · obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
      cases h; simp [hc]
    · rw [run_cap h]
  · rw [run_cap h]

theorem devOnly_eq {w : World} {k : Option (Option V × Dev)} {r : Option V} {w' : World}
    (h : devOnly w k = some (r, w')) :
    ∃ d, k = some (r, d) ∧ w' = { w with dev := d, mem := w.mem.retire (d.race.clock hostParty) } := by
  unfold devOnly at h
  obtain ⟨⟨r0, d⟩, hkd, he⟩ := Option.map_eq_some_iff.mp h
  simp only [Prod.mk.injEq] at he
  obtain ⟨rfl, rfl⟩ := he
  exact ⟨d, hkd, rfl⟩

theorem pump_life (w : World) : w.pump.mem = w.mem ∧ w.pump.dev = w.dev ∧ w.pump.win.live = w.win.live ∧
    w.pump.lmdb = w.lmdb ∧ w.pump.gpu = w.gpu ∧ w.pump.thread = w.thread ∧ w.pump.display = w.display ∧
    w.pump.cudaDevice = w.cudaDevice ∧ w.pump.gpuAdapter = w.gpuAdapter := by
  unfold World.pump; split <;> exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem foldlM_ops_live (w : World) : ∀ (ops : List DevOp) (d d' : Dev),
    ops.foldlM (fun d op => match op with
      | .launch l ids => d.runLaunch w.kernel l ids
      | .vendor c ins out => d.runVendor w.vendor c ins out) d = some d' →
    d'.live = d.live ∧ d'.capture = d.capture
  | [], d, d', h => by simp at h; subst h; exact ⟨rfl, rfl⟩
  | op :: ops, d, d', h => by
      simp only [List.foldlM_cons] at h
      obtain ⟨d1, h1, h2⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨e1, e2⟩ := foldlM_ops_live w ops d1 d' h2
      have : d1.live = d.live ∧ d1.capture = d.capture := by
        cases op with
        | launch l ids => exact ⟨runLaunch_live h1, (runLaunch_meta h1).2.1⟩
        | vendor c ins out => exact ⟨runVendor_live h1, (runVendor_meta h1).2.1⟩
      exact ⟨e1.trans this.1, e2.trans this.2⟩

theorem graphOps_life {w : World} {ops : Array DevOp} {d d' : Dev}
    (h : ops.foldlM (fun d op => match op with
      | .launch l ids => d.runLaunch w.kernel l ids
      | .vendor c ins out => d.runVendor w.vendor c ins out) d = some d') :
    d'.live = d.live ∧ d'.capture = d.capture := by
  rw [← Array.foldlM_toList] at h
  exact foldlM_ops_live w _ d d' h

theorem runOne_live {sh : Dispatch → List ByteArray → List ByteArray} {g g' : Wgpu} {d : Nat × List UInt64}
    (h : g.runOne sh d = some g') : g'.live = g.live := by
  unfold Wgpu.runOne at h
  split at h
  · cases h; rfl
  · split at h <;> (cases h; rfl)

theorem runAll_live {sh : Dispatch → List ByteArray → List ByteArray} :
    ∀ (ds : List (Nat × List UInt64)) (g g' : Wgpu), ds.foldlM (Wgpu.runOne sh) g = some g' → g'.live = g.live
  | [], g, g', h => by simp at h; subst h; rfl
  | d :: ds, g, g', h => by
      simp only [List.foldlM_cons] at h
      obtain ⟨g1, h1, h2⟩ := Option.bind_eq_some_iff.mp h
      exact (runAll_live ds g1 g' h2).trans (runOne_live h1)

theorem flush_live {sh : Dispatch → List ByteArray → List ByteArray} {g g' : Wgpu}
    (h : Wgpu.flush sh g = some g') : g'.live = g.live := by
  unfold Wgpu.flush at h
  obtain ⟨g1, h1, h⟩ := Option.bind_eq_some_iff.mp h
  cases h; exact runAll_live _ g g1 h1

theorem fold_life {α : Type} {g : Dev → α → Option Dev}
    (hg : ∀ d x d', g d x = some d' → d'.live = d.live ∧ d'.capture.isSome = d.capture.isSome) :
    ∀ (xs : List α) (d d' : Dev), xs.foldlM g d = some d' →
      d'.live = d.live ∧ d'.capture.isSome = d.capture.isSome
  | [], d, d', h => by simp at h; subst h; exact ⟨rfl, rfl⟩
  | x :: xs, d, d', h => by
      simp only [List.foldlM_cons] at h
      obtain ⟨d1, h1, h2⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨e1, e2⟩ := fold_life hg xs d1 d' h2
      obtain ⟨f1, f2⟩ := hg d x d1 h1
      exact ⟨e1.trans f1, e2.trans f2⟩

/-- **The lifecycle parts of a world** a typestate is built from: whether the
    CUDA context is live and capturing, whether memory is frozen for a worker,
    and whether the LMDB, window, wgpu and thread contexts are live. -/
inductive Part where
  | cuda | capturing | frozen | lmdb | win | gpu | thread
  /-- The LMDB context has no environment open. -/
  | lmdbEmpty
  /-- There is a display to open windows on: an oracle no call changes. -/
  | display
  /-- There is a CUDA device: an oracle no call changes. -/
  | cudaDevice
  /-- wgpu finds an adapter: an oracle no call changes. -/
  | gpuAdapter
  /-- No asynchronous copy is in flight over page-locked memory. -/
  | idle
  deriving DecidableEq, Repr

def Part.get : Part → World → Bool
  | .cuda, w => w.dev.live
  | .capturing, w => w.dev.capture.isSome
  | .frozen, w => w.mem.frozen
  | .lmdb, w => w.lmdb.live
  | .win, w => w.win.live
  | .gpu, w => w.gpu.live
  | .thread, w => w.thread.live
  | .lmdbEmpty, w => w.lmdb.envs.isEmpty
  | .display, w => w.display
  | .cudaDevice, w => w.cudaDevice
  | .gpuAdapter, w => w.gpuAdapter
  | .idle, w => w.mem.busy.isEmpty

/-- **The kinds of handle a library hands a program**, which a typestate
    may say it holds: an open window, serial port or USB device. -/
inductive Held where
  | window | port | usbDevice | usbInterface | heap
  deriving DecidableEq, Repr

/-- Whether `a` is a handle of kind `k` the program holds open; for the heap,
    the start of a live allocation. -/
def Held.live : Held → World → UInt64 → Bool
  | .window, w, a => (w.wlWin? a).isSome
  | .port, w, a => (w.serPort? a).isSome
  | .usbDevice, w, a => (w.usbDev? a).isSome
  | .usbInterface, w, a => (w.usbIface? a).isSome
  | .heap, w, a => (w.heapAt? a).isSome

/-- **The parts each entry point may move.** Bringing a context up or down,
    beginning or ending a capture, and joining or dropping a worker; every other
    call keeps every part (`callBits_parts`). -/
def movesLife : IR.Ffi → List Part
  | .cudaInit | .cudaCleanup => [.cuda, .capturing]
  | .cudaGraphBeginCapture | .cudaGraphEndCapture => [.capturing]
  | .lmdbInit | .lmdbCleanup => [.lmdb, .lmdbEmpty]
  | .lmdbOpen | .lmdbBeginWriteTxn | .lmdbPut | .lmdbCommitWriteTxn | .lmdbCursorScan => [.lmdbEmpty]
  | .windowInit | .windowCleanup => [.win]
  | .gpuInit | .gpuCleanup => [.gpu]
  | .threadInit => [.thread]
  | .threadCleanup => [.thread, .frozen]
  | .threadJoin => [.frozen]
  | _ => []

/-- **The parts each entry point may move**: the lifecycle parts `movesLife`
    names, and whether an asynchronous copy is in flight, which no entry point
    is shown to keep. -/
def moves (f : IR.Ffi) : List Part := .idle :: movesLife f

/-- A call to `f` from `w` to `w'` kept every part it does not move. -/
def KeepsParts (f : IR.Ffi) (w w' : World) : Prop := ∀ p, p ∉ moves f → p.get w' = p.get w

/-- Follow a contract to each of its answers. -/
macro "walk" h:ident : tactic => `(tactic| (
  try simp only [bind, Option.bind, cudaFail, cudaI64, lmdbI32, gpuUploadAt] at $h:ident
  try unfold memInfoOf at $h:ident
  try unfold cudaLaunchOn at $h:ident
  try unfold sgemvOn at $h:ident
  try unfold sgemmOn at $h:ident
  try unfold ffiCudaUploadAsyncAt at $h:ident
  try unfold ffiCudaDownloadAsyncAt at $h:ident
  try unfold asyncCopy at $h:ident
  try simp only [bind, Option.bind] at $h:ident
  repeat' (first
    | (cases $h:ident; done)
    | (injection $h:ident with h1; injection h1 with h2 h3; subst h3; try subst h2)
    | (rw [Option.bind_eq_some_iff] at $h:ident; obtain ⟨_, _, $h:ident⟩ := $h:ident)
    | split at $h:ident
    | (dsimp only at $h:ident))))

/-- Close one part at one answer of a contract. -/
macro "part_close" : tactic => `(tactic| (
  intro p hp
  try (have := graphOps_life ‹Array.foldlM _ _ _ = some _›)
  try (have := fold_life (fun _ _ _ h => ⟨devOp_live h, devOp_capturing h⟩) _ _ _ ‹List.foldlM _ _ _ = some _›)
  cases p
  all_goals first
    | rfl
    | (exfalso; exact hp (by decide))
    | grind [Part.get, devFail, devOk, devI64, Dev.endCapture, Mem.retire, Mem.forParty, put_live,
        putAll_live, → runLaunch_live, → runVendor_live, → devOp_live, → devOp_capturing, → syncWrite_eq,
        → syncRead_eq, → runLaunch_meta, → runVendor_meta, → flush_live, → store_frozen, → copyIn_frozen,
        pump_life]))

/-- Close one answer of a contract: the kernel and cuBLAS oracles as they were. -/
macro "oracle_close" : tactic => `(tactic| first
  | exact ⟨rfl, rfl⟩
  | (unfold World.pump; split <;> exact ⟨rfl, rfl⟩))

-- ---------------------------------------------------------------------------
-- What the calls leave of memory's layout
-- ---------------------------------------------------------------------------

theorem foldl_set_size (off : Nat) (x : Nat → UInt8) : ∀ (l : List Nat) (bs : ByteArray),
    (l.foldl (fun (b : ByteArray) i => b.set! (off + i) (x i)) bs).size = bs.size
  | [], _ => rfl
  | i :: l, bs => by
      simp only [List.foldl_cons]
      rw [foldl_set_size off x l, ByteArray.size_set!_eq]

theorem store_sizes {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m')
    (r : Region) : (m'.region r).size = (m.region r).size := by
  unfold Mem.store at h
  obtain ⟨⟨r0, off⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  · cases h
    by_cases e : r0 = r
    · subst e
      rw [Mem.region_setRegion_self]
      exact foldl_set_size off (fun i => ((v >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8) _ _
    · rw [Mem.region_setRegion_ne _ _ _ _ e]

theorem copyIn_sizes {m m' : Mem} {a : UInt64} {src : ByteArray} (h : copyIn m a src = some m')
    (r : Region) : (m'.region r).size = (m.region r).size := ((copyIn_same h).size r).symm

/-- Every region's size. -/
def _root_.AlgorithmLib.HProg.Sem.Mem.sizes (m : Mem) (r : Region) : Nat := (m.region r).size

theorem store_sizes_eq {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m') :
    m'.sizes = m.sizes := funext (store_sizes h)

theorem copyIn_sizes_eq {m m' : Mem} {a : UInt64} {src : ByteArray} (h : copyIn m a src = some m') :
    m'.sizes = m.sizes := funext (copyIn_sizes h)

theorem retire_sizes (m : Mem) (c : Clock) : (m.retire c).sizes = m.sizes :=
  funext fun r => by cases r <;> rfl

theorem forParty_sizes (m : Mem) (p : Nat) : (m.forParty p).sizes = m.sizes :=
  funext fun r => by cases r <;> rfl

theorem sizes_with_frozen (m : Mem) (b : Bool) : ({ m with frozen := b } : Mem).sizes = m.sizes :=
  funext fun r => by cases r <;> rfl

theorem sizes_with_live (m : Mem) (L : List (Nat × Nat)) : ({ m with pinnedLive := L } : Mem).sizes = m.sizes :=
  funext fun r => by cases r <;> rfl

theorem sizes_with_busy (m : Mem) (B : List Busy) : ({ m with busy := B } : Mem).sizes = m.sizes :=
  funext fun r => by cases r <;> rfl

/-- A call to `f` from `w` to `w'` kept the size of every region but the pinned
    one, which pinned allocation grows. -/
def KeepsRooms (w w' : World) : Prop := ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r

open Lean Elab Tactic Meta in
/-- Every memory step a path took, as an equation between region sizes. -/
elab "sizes_facts" : tactic => withMainContext do
  for h in (← getLCtx) do
    if h.isImplementationDetail then continue
    let T ← instantiateMVars h.type
    let some (_, lhs, rhs) := T.eq? | continue
    unless rhs.isAppOf ``Option.some do continue
    let lem := if lhs.isAppOf ``Mem.store then some ``store_sizes_eq
      else if lhs.isAppOf ``copyIn then some ``copyIn_sizes_eq else none
    let some lem := lem | continue
    let pf ← mkAppM lem #[h.toExpr]
    let g ← getMainGoal
    let (_, g') ← g.note `hsz pf
    replaceMainGoal [g']

/-- Close one region at one answer of a contract. -/
macro "room_close" : tactic => `(tactic| (
  intro r hr
  sizes_facts
  cases r
  all_goals first
    | rfl
    | exact absurd rfl hr
    | (simp only [*, retire_sizes, forParty_sizes, sizes_with_frozen, sizes_with_live, sizes_with_busy,
        (pump_life _).1]; done)
    | grind [pump_life]))

/-- What an asynchronous copy answers: the device and memory its continuation
    makes, with the copy recorded among memory's in-flight copies. -/
theorem asyncCopy_eq {w : World} {p : Nat} {a : UInt64} {n : Nat} {wr : Bool} {rs ws : List Nat}
    {k : Race → Option (Dev × Mem)} {r : Option V} {w' : World}
    (h : asyncCopy w p a n wr rs ws k = some (r, w')) :
    ∃ race d m B, k race = some (d, m) ∧ w' = { w with dev := d, mem := { m with busy := B } } := by
  unfold asyncCopy at h
  split at h
  · cases h
  · obtain ⟨race, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨⟨d, m⟩, hk, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    exact ⟨race, d, m, _, hk, h.2.symm⟩

set_option maxHeartbeats 1000000 in
theorem rooms_cudaDownloadAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownloadAsync bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaDownloadAsync bits w = some (r, w') at h
  unfold ffiCudaDownloadAsync ffiCudaDownloadAsyncAt at h
  split at h
  · simp only [cudaFail] at h
    split at h
    · injection h with h1; injection h1 with h2 h3; subst h3; intro _ _; rfl
    · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · injection h with h1; injection h1 with h2 h3; subst h3; intro _ _; rfl
      · split at h
        · injection h with h1; injection h1 with h2 h3; subst h3; intro _ _; rfl
        · split at h
          · injection h with h1; injection h1 with h2 h3; subst h3; intro _ _; rfl
          · split at h
            · injection h with h1; injection h1 with h2 h3; subst h3; intro _ _; rfl
            · obtain ⟨race, d, m, B, hk, rfl⟩ := asyncCopy_eq h
              obtain ⟨m0, hm0, hk⟩ := Option.bind_eq_some_iff.mp hk
              simp only [Option.some.injEq, Prod.mk.injEq] at hk
              obtain ⟨-, rfl⟩ := hk
              intro reg _
              show ({ m0 with busy := B } : Mem).sizes reg = w.mem.sizes reg
              rw [sizes_with_busy, copyIn_sizes_eq hm0, forParty_sizes]
  · cases h

/-- A step that leaves page-locked memory the size it was. -/
def KeepsPinned (w w' : World) : Prop := w'.mem.sizes .pinned = w.mem.sizes .pinned

/-- Close the size of page-locked memory at one answer of a contract. -/
macro "pinned_close" : tactic => `(tactic| (
  unfold KeepsPinned
  sizes_facts
  first
    | rfl
    | (simp only [*, retire_sizes, forParty_sizes, sizes_with_frozen, sizes_with_live, sizes_with_busy,
        (pump_life _).1]; done)
    | grind [pump_life]))

set_option maxHeartbeats 1000000 in
theorem pinned_cudaDownloadAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownloadAsync bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaDownloadAsync bits w = some (r, w') at h
  unfold ffiCudaDownloadAsync ffiCudaDownloadAsyncAt at h
  split at h
  · simp only [cudaFail] at h
    split at h
    · injection h with h1; injection h1 with h2 h3; subst h3; rfl
    · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · injection h with h1; injection h1 with h2 h3; subst h3; rfl
      · split at h
        · injection h with h1; injection h1 with h2 h3; subst h3; rfl
        · split at h
          · injection h with h1; injection h1 with h2 h3; subst h3; rfl
          · split at h
            · injection h with h1; injection h1 with h2 h3; subst h3; rfl
            · obtain ⟨race, d, m, B, hk, rfl⟩ := asyncCopy_eq h
              obtain ⟨m0, hm0, hk⟩ := Option.bind_eq_some_iff.mp hk
              simp only [Option.some.injEq, Prod.mk.injEq] at hk
              obtain ⟨-, rfl⟩ := hk
              show ({ m0 with busy := B } : Mem).sizes .pinned = w.mem.sizes .pinned
              rw [sizes_with_busy, copyIn_sizes_eq hm0, forParty_sizes]
  · cases h

/-- **What a call to `f` keeps of the world**, and what it answers: the
    observations made so far, the parts it does not move, the kernel and cuBLAS
    oracles, the size of every region but the pinned one, the pinned one's too
    unless the call allocates there, and a scalar of the type `f`'s signature
    gives. -/
structure Keeps (f : IR.Ffi) (r : Option V) (w w' : World) : Prop where
  obs : w'.obs = w.obs
  parts : KeepsParts f w w'
  oracles : w'.kernel = w.kernel ∧ w'.vendor = w.vendor
  rooms : KeepsRooms w w'
  pinned : f ≠ .cudaPinnedAlloc → KeepsPinned w w'
  scalar : ∀ v, r = some v → ∃ x, v = .sc (f.result.getD .i64) x

/-- Close the answer at one answer of a contract: a scalar of the result type. -/
macro "scalar_close" : tactic => `(tactic| (
  intro v hv
  cases hv <;> exact ⟨_, rfl⟩))

/-- Close one answer of a contract: the observations as they were. -/
macro "obs_close" : tactic => `(tactic| first
  | rfl
  | (unfold World.pump; split <;> rfl))

/-- Close `Keeps` at one answer of a contract. -/
macro "keeps_close" : tactic => `(tactic| (
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · obs_close
  · part_close
  · oracle_close
  · room_close
  · first
      | (intro hf; exact absurd rfl hf)
      | (intro _; pinned_close)
  · scalar_close))

/-- One entry's `Keeps`: its contract followed to every answer. -/
macro "keeps_entry " h:ident : tactic => `(tactic| (
  dsimp only [callBits] at $h:ident
  first
    | (cases $h:ident; done)
    | (unfold_head $h
       first
         | (obtain ⟨d, hk, rfl⟩ := devOnly_eq $h; clear $h; walk hk; all_goals keeps_close)
         | (walk $h; all_goals keeps_close))))

set_option maxHeartbeats 4000000 in
theorem keeps_cudaDownloadAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownloadAsync bits w = some (r, w')) : Keeps .cudaDownloadAsync r w w' := by
  refine ⟨?_, ?_, ?_, rooms_cudaDownloadAsync h, fun _ => pinned_cudaDownloadAsync h, ?_⟩
  all_goals
    change ffiCudaDownloadAsync bits w = some (r, w') at h
    unfold ffiCudaDownloadAsync at h
  · walk h; all_goals obs_close
  · walk h; all_goals part_close
  · walk h; all_goals oracle_close
  · walk h; all_goals scalar_close

open Lean Elab Command in
/-- `keeps_<entry>` for every entry point that has none, and `callBits_keeps`
    over them all. -/
elab "derive_keeps" : command => do
  let some (.inductInfo iv) := (← getEnv).find? ``IR.Ffi | throwError "derive_keeps: no Ffi"
  let ns ← getCurrNamespace
  let mut alts := #[]
  for c in iv.ctors do
    let short := Name.mkSimple s!"keeps_{c.componentsRev.head!}"
    let thm := mkIdent short
    let ctor := mkIdent c
    unless (← getEnv).contains (ns ++ short) do
      elabCommand (← `(command|
        set_option maxHeartbeats 4000000 in
        theorem $thm {bits : List UInt64} {w : World} {r : Option V} {w' : World}
            (h : callBits $ctor bits w = some (r, w')) : Keeps $ctor r w w' := by
          keeps_entry h))
    alts := alts.push (← `(Lean.Parser.Term.matchAltExpr| | $ctor, h => $thm h))
  elabCommand (← `(command|
    theorem $(mkIdent `callBits_keeps) (f : IR.Ffi) {bits : List UInt64} {w : World} {r : Option V} {w' : World}
        (h : callBits f bits w = some (r, w')) : Keeps f r w w' :=
      match f, h with $alts:matchAlt*))

derive_keeps

/-- **No call adds to or drops from the observations**: a call is observed
    where it is made, before it runs. -/
theorem callBits_obs (f : IR.Ffi) {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) : w'.obs = w.obs := (callBits_keeps f h).obs

/-- **Every call keeps every part it does not move.** -/
theorem callBits_parts (f : IR.Ffi) {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) : KeepsParts f w w' := (callBits_keeps f h).parts

/-- **No call changes what a kernel or cuBLAS computes.** -/
theorem callBits_oracles (f : IR.Ffi) {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor :=
  (callBits_keeps f h).oracles

/-- **Every call keeps the size of every region but the pinned one.** -/
theorem callBits_rooms (f : IR.Ffi) {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) : KeepsRooms w w' := (callBits_keeps f h).rooms

/-- **Every call but the one that allocates it leaves page-locked memory the
    size it was.** -/
theorem callBits_pinned (f : IR.Ffi) {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (hf : f ≠ .cudaPinnedAlloc) (h : callBits f bits w = some (r, w')) : KeepsPinned w w' :=
  (callBits_keeps f h).pinned hf

/-- **An entry point's answer is a scalar of the type its signature gives.** -/
theorem callBits_scalar (f : IR.Ffi) {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits f bits w = some (some v, w')) : ∃ x, v = .sc (f.result.getD .i64) x :=
  (callBits_keeps f h).scalar v rfl


-- ---------------------------------------------------------------------------
-- What the movers leave
-- ---------------------------------------------------------------------------

theorem sets_cudaInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaInit bits w = some (r, w')) :
    w'.dev.live = w.cudaDevice ∧ w'.dev.capture.isSome = false := by
  change ffiCudaInit bits w = some (r, w') at h
  unfold ffiCudaInit at h
  walk h
  all_goals first | exact ⟨rfl, rfl⟩ | simp_all | rfl | grind [→ store_frozen]

theorem sets_cudaCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaCleanup bits w = some (r, w')) : w'.dev.live = false ∧ w'.dev.capture.isSome = false := by
  change ffiCudaCleanup bits w = some (r, w') at h
  unfold ffiCudaCleanup at h
  walk h
  all_goals first | exact ⟨rfl, rfl⟩ | rfl | grind [→ store_frozen]

theorem sets_lmdbInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbInit bits w = some (r, w')) : w'.lmdb.live = true := by
  change ffiLmdbInit bits w = some (r, w') at h
  unfold ffiLmdbInit at h
  walk h
  all_goals first | exact ⟨rfl, rfl⟩ | rfl | grind [→ store_frozen]

theorem empties_lmdbInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbInit bits w = some (r, w')) : w'.lmdb.envs.isEmpty = true := by
  change ffiLmdbInit bits w = some (r, w') at h
  unfold ffiLmdbInit at h
  walk h
  all_goals first | rfl | grind [→ store_frozen]

theorem sets_gpuInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuInit bits w = some (r, w')) : w'.gpu.live = w.gpuAdapter := by
  change ffiGpuInit bits w = some (r, w') at h
  unfold ffiGpuInit at h
  walk h
  all_goals first | exact ⟨rfl, rfl⟩ | simp_all | rfl | grind [→ store_frozen]

theorem sets_gpuCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCleanup bits w = some (r, w')) : w'.gpu.live = false := by
  change ffiGpuCleanup bits w = some (r, w') at h
  unfold ffiGpuCleanup at h
  walk h
  all_goals first | exact ⟨rfl, rfl⟩ | rfl | grind [→ store_frozen]

theorem sets_threadInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadInit bits w = some (r, w')) : w'.thread.live = true := by
  change ffiThreadInit bits w = some (r, w') at h
  unfold ffiThreadInit at h
  walk h
  all_goals first | exact ⟨rfl, rfl⟩ | rfl | grind [→ store_frozen]

theorem sets_threadCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadCleanup bits w = some (r, w')) : w'.mem.frozen = false := by
  change ffiThreadCleanup bits w = some (r, w') at h
  unfold ffiThreadCleanup at h
  walk h
  all_goals first | exact ⟨rfl, rfl⟩ | rfl | grind [→ store_frozen]

theorem sets_threadJoin {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadJoin bits w = some (r, w')) (hz : w.mem.frozen = false) : w'.mem.frozen = false := by
  change ffiThreadJoin bits w = some (r, w') at h
  unfold ffiThreadJoin at h
  walk h
  all_goals first | rfl | exact hz

-- ---------------------------------------------------------------------------
-- Typestates: parts, rooms and cells
-- ---------------------------------------------------------------------------
/-- The value `n` bytes of region bytes `bs` from `off` make, little-endian. -/
def bytesAt (bs : ByteArray) (off n : Nat) : UInt64 :=
  (List.range n).foldr (fun i acc => (acc <<< 8) ||| (bs.get! (off + i)).toUInt64) 0

theorem load_some {m : Mem} {a : UInt64} {n : Nat} {v : UInt64} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (h : m.load a n = some v) :
    off + n ≤ (m.region r).size ∧ m.readable r off n = true ∧ v = bytesAt (m.region r) off n := by
  unfold Mem.load at h
  simp only [hd, Option.bind_eq_bind, Option.bind_some] at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_eq_eq_not, Bool.not_true, not_or,
      Bool.not_eq_false] at hc
    injection h with h
    exact ⟨by omega, by simpa using hc.2, h.symm⟩

theorem load_of {m : Mem} {a : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hs : off + n ≤ (m.region r).size)
    (hr : m.readable r off n = true) : m.load a n = some (bytesAt (m.region r) off n) := by
  unfold Mem.load
  simp only [hd, Option.bind_eq_bind, Option.bind_some]
  rw [if_neg (by simp [hr]; omega)]
  rfl

theorem readable_off_pinned {m : Mem} {r : Region} (hr : r ≠ .pinned) (off n : Nat) :
    m.readable r off n = !m.frozen := by
  cases r <;> simp_all (config := { decide := true }) [Mem.readable]

theorem foldr_bytes_congr {bs bs' : ByteArray} {off : Nat} : ∀ (l : List Nat) (init : UInt64),
    (∀ i ∈ l, bs'.get! (off + i) = bs.get! (off + i)) →
    l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| (bs'.get! (off + i)).toUInt64) init =
      l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| (bs.get! (off + i)).toUInt64) init
  | [], _, _ => rfl
  | i :: l, init, h => by
      simp only [List.foldr_cons, foldr_bytes_congr l init (fun j hj => h j (List.mem_cons_of_mem _ hj)),
        h i List.mem_cons_self]

theorem bytesAt_congr {bs bs' : ByteArray} {off n : Nat}
    (h : ∀ i, i < n → bs'.get! (off + i) = bs.get! (off + i)) : bytesAt bs' off n = bytesAt bs off n :=
  foldr_bytes_congr _ _ (fun i hi => h i (List.mem_range.mp hi))

/-- **A load survives a step that changes none of its bytes**, keeps its
    region's size, and leaves memory unfrozen, outside pinned memory. -/
theorem load_kept {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned)
    (hbytes : ∀ i, i < n → decodeAddr (a + UInt64.ofNat i) = some (r, off + i))
    (h : m.load a n = some v) (hz : m'.frozen = false) (hsz : m'.sizes r = m.sizes r)
    (hu : ∀ i, i < n → Unchanged m m' (a + UInt64.ofNat i)) : m'.load a n = some v := by
  obtain ⟨hs, hr, rfl⟩ := load_some hd h
  have hs' : off + n ≤ (m'.region r).size := by
    have : (m'.region r).size = (m.region r).size := hsz
    omega
  rw [load_of hd hs' (by rw [readable_off_pinned hp, hz]; rfl)]
  congr 1
  apply bytesAt_congr
  intro i hi
  have hz0 : m.frozen = false := by
    rw [readable_off_pinned hp] at hr; simpa using hr
  have h1 := load_of (m := m) (n := 1) (hbytes i hi) (by omega) (by rw [readable_off_pinned hp, hz0]; rfl)
  have h1' := load_of (m := m') (n := 1) (hbytes i hi) (by omega) (by rw [readable_off_pinned hp, hz]; rfl)
  have e := hu i hi _ _ h1 h1'
  simp [bytesAt] at e
  exact (UInt8.toUInt64_inj.mp e).symm

/-- Bytes a step leaves unchanged, and where it keeps the region's size, read
    the same. -/
theorem bytes_kept {m m' : Mem} {a : UInt64} {k : Nat} {r : Region} {off : Nat}
    (hp : r ≠ .pinned) (hbytes : ∀ i, i < k → decodeAddr (a + UInt64.ofNat i) = some (r, off + i))
    (hs : off + k ≤ (m.region r).size) (hz : m.frozen = false) (hz' : m'.frozen = false)
    (hsz : m'.sizes r = m.sizes r) (hu : ∀ i, i < k → Unchanged m m' (a + UInt64.ofNat i)) :
    ∀ i, i < k → (m'.region r).get! (off + i) = (m.region r).get! (off + i) := by
  intro i hi
  have hs' : (m'.region r).size = (m.region r).size := hsz
  have h1 := load_of (m := m) (n := 1) (hbytes i hi) (by omega) (by rw [readable_off_pinned hp, hz]; rfl)
  have h1' := load_of (m := m') (n := 1) (hbytes i hi) (by omega) (by rw [readable_off_pinned hp, hz']; rfl)
  have e := hu i hi _ _ h1 h1'
  simp [bytesAt] at e
  exact (UInt8.toUInt64_inj.mp e).symm

/-- **A C string reads the same** where its bytes, terminator included, and its
    region's size are the same. -/
theorem strSpan_congr {m m' : Mem} {a : UInt64} {r : Region} {off k : Nat}
    (hsp : Sem.strSpan m a = some (r, off, k)) (hs : (m'.region r).size = (m.region r).size)
    (hb : ∀ i, i < k → (m'.region r).get! (off + i) = (m.region r).get! (off + i)) :
    Sem.strSpan m' a = some (r, off, k) ∧ readCStr m' a = readCStr m a := by
  unfold Sem.strSpan at hsp ⊢
  unfold readCStr
  cases hd : decodeAddr a with
  | none => rw [hd] at hsp; cases hsp
  | some p =>
    obtain ⟨r0, off0⟩ := p
    simp only [hd, bind, Option.bind] at hsp ⊢
    cases hf : (List.range ((m.region r0).size - off0)).find? (fun i => (m.region r0).get! (off0 + i) == 0) with
    | none => rw [hf] at hsp; cases hsp
    | some n =>
      rw [hf] at hsp
      simp only [Option.some.injEq, Prod.mk.injEq] at hsp
      obtain ⟨rfl, rfl, rfl⟩ := hsp
      have hf' : (List.range ((m'.region r0).size - off0)).find? (fun i => (m'.region r0).get! (off0 + i) == 0)
          = some n := by
        rw [hs]
        rw [List.find?_range_eq_some] at hf ⊢
        obtain ⟨h1, h2, h3⟩ := hf
        refine ⟨by rw [hb n (by omega)]; exact h1, h2, fun j hj => ?_⟩
        rw [hb j (by omega)]; exact h3 j hj
      rw [hf']
      refine ⟨rfl, ?_⟩
      simp only
      congr 2
      apply List.map_congr_left
      intro i hi
      exact hb i (by simp at hi; omega)

theorem strSpan_bound {m : Mem} {a : UInt64} {r : Region} {off k : Nat}
    (h : Sem.strSpan m a = some (r, off, k)) : off + k ≤ (m.region r).size := by
  unfold Sem.strSpan at h
  cases hd : decodeAddr a with
  | none => rw [hd] at h; cases h
  | some p =>
    obtain ⟨r0, off0⟩ := p
    simp only [hd, bind, Option.bind] at h
    cases hf : (List.range ((m.region r0).size - off0)).find? (fun i => (m.region r0).get! (off0 + i) == 0) with
    | none => rw [hf] at h; cases h
    | some n =>
      rw [hf] at h
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      have h2 := (List.find?_range_eq_some.mp hf).2.1
      simp only [List.mem_range] at h2
      omega

theorem strSpan_decode {m : Mem} {a : UInt64} {r : Region} {off k : Nat}
    (h : Sem.strSpan m a = some (r, off, k)) : decodeAddr a = some (r, off) := by
  unfold Sem.strSpan at h
  cases hd : decodeAddr a with
  | none => rw [hd] at h; cases h
  | some p =>
    obtain ⟨r0, off0⟩ := p
    simp only [hd, bind, Option.bind] at h
    split at h
    · cases h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, -⟩ := h
      rfl

theorem ByteArray.get!_set!_same (b : ByteArray) (i : Nat) (v : UInt8) (h : i < b.size) :
    (b.set! i v).get! i = v := by
  cases b with | mk bs =>
  simp [ByteArray.set!, ByteArray.get!, Array.set!_eq_setIfInBounds, getElem!_def]
  simp [ByteArray.size] at h
  simp [h]

/-- The byte a store of `v` puts at `off + j`. -/
def byteOfV (v : UInt64) (j : Nat) : UInt8 := ((v >>> (8 * UInt64.ofNat j)) &&& 0xff).toUInt8

theorem foldl_set!_at (off : Nat) (v : UInt64) : ∀ (n : Nat) (b : ByteArray) (j : Nat), j < n →
    off + n ≤ b.size →
    ((List.range n).foldl (fun (b : ByteArray) i => b.set! (off + i) (byteOfV v i)) b).get! (off + j)
      = byteOfV v j
  | 0, _, _, hj, _ => absurd hj (Nat.not_lt_zero _)
  | n + 1, b, j, hj, hs => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      have hsz : ((List.range n).foldl (fun (b : ByteArray) i => b.set! (off + i) (byteOfV v i)) b).size
          = b.size := (foldl_set!_other off (off + n) v (List.range n) b (by
        intro i hi; have := List.mem_range.mp hi; omega)).2
      by_cases e : j = n
      · subst e
        exact ByteArray.get!_set!_same _ _ _ (by rw [hsz]; omega)
      · rw [ByteArray.get!_set!_of_ne _ _ _ _ (by omega)]
        exact foldl_set!_at off v n b j (by omega) (by omega)

theorem foldr_fun_congr (f g : Nat → UInt64) : ∀ (l : List Nat) (init : UInt64), (∀ i ∈ l, f i = g i) →
    l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| f i) init =
      l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| g i) init
  | [], _, _ => rfl
  | i :: l, init, h => by
      simp only [List.foldr_cons, foldr_fun_congr f g l init (fun j hj => h j (List.mem_cons_of_mem _ hj)),
        h i List.mem_cons_self]

theorem shl8_or (acc : UInt64) (b : UInt8) :
    ((acc <<< 8) ||| b.toUInt64).toNat = acc.toNat % 2 ^ 56 * 256 + b.toNat := by
  rw [UInt64.toNat_or, UInt64.toNat_shiftLeft]
  have hb : b.toUInt64.toNat = b.toNat := by simp
  rw [hb]
  have e : acc.toNat <<< ((8 : UInt64).toNat % 64) % 2 ^ 64 = (acc.toNat % 2 ^ 56) <<< 8 := by
    simp only [Nat.shiftLeft_eq]
    have : (8 : UInt64).toNat % 64 = 8 := by decide
    rw [this]; omega
  rw [e, ← Nat.shiftLeft_add_eq_or_of_lt (UInt8.toNat_lt b), Nat.shiftLeft_eq]

theorem byte_toNat (v : UInt64) (j : Nat) (hj : j < 8) :
    (byteOfV v j).toNat = v.toNat / 2 ^ (8 * j) % 256 := by
  unfold byteOfV
  simp only [UInt64.toNat_toUInt8, UInt64.toNat_and, UInt64.toNat_shiftRight]
  have h1 : (8 * UInt64.ofNat j).toNat % 64 = 8 * j := by
    rw [UInt64.toNat_mul, UInt64.toNat_ofNat']; simp; omega
  rw [h1, Nat.shiftRight_eq_div_pow, show (0xff : UInt64).toNat = 2 ^ 8 - 1 from rfl,
    Nat.and_two_pow_sub_one_eq_mod]
  omega

/-- Eight bytes of a value, little-endian, make the value. -/
theorem reassemble8 (v : UInt64) :
    (List.range 8).foldr (fun i (acc : UInt64) => (acc <<< 8) ||| (byteOfV v i).toUInt64) 0 = v := by
  apply UInt64.toNat_inj.mp
  simp only [List.range, List.range.loop, List.foldr, shl8_or, UInt64.toNat_zero]
  have hv := UInt64.toNat_lt_size v
  simp only [byte_toNat v 0 (by decide), byte_toNat v 1 (by decide), byte_toNat v 2 (by decide),
    byte_toNat v 3 (by decide), byte_toNat v 4 (by decide), byte_toNat v 5 (by decide),
    byte_toNat v 6 (by decide), byte_toNat v 7 (by decide)]
  simp only [UInt64.size] at hv
  omega

/-- **An eight-byte store reads back**, outside pinned memory. -/
theorem load_store8 {m m' : Mem} {a v : UInt64} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned) (h : m.store a 8 v = some m') :
    m'.load a 8 = some v := by
  have hz : m'.frozen = false := by
    have e := (Mem.store_meta h).2.2
    unfold Mem.store at h
    simp only [hd, Option.bind_eq_bind, Option.bind_some] at h
    split at h
    · cases h
    · rename_i hc
      simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_eq_eq_not, Bool.not_true, not_or,
        Bool.not_eq_false] at hc
      have := hc.2
      cases r <;> simp_all [Mem.reachable]
  have hsz : (m'.region r).size = (m.region r).size := store_sizes h r
  unfold Mem.store at h
  simp only [hd, Option.bind_eq_bind, Option.bind_some] at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hc
    injection h with h
    subst h
    rw [load_of hd (by rw [hsz]; omega) (by rw [readable_off_pinned hp, hz]; rfl)]
    congr 1
    rw [Mem.region_setRegion_self]
    unfold bytesAt
    refine (foldr_fun_congr _ (fun i => (byteOfV v i).toUInt64) _ _ fun j hj => ?_).trans (reassemble8 v)
    exact congrArg UInt8.toUInt64 (foldl_set!_at off v 8 _ j (List.mem_range.mp hj) (by omega))


-- ---------------------------------------------------------------------------
-- Typestates
-- ---------------------------------------------------------------------------

/-- **A fact a typestate names**: a lifecycle part's value, that a region is at
    least so large, what an eight-byte cell of memory holds, or that a handle
    a library handed out is held open. -/
inductive Fact where
  | part (p : Part) (b : Bool)
  | room (r : Region) (n : Nat)
  | cell (a : UInt64) (v : UInt64)
  | held (k : Held) (a : UInt64)
  /-- Null, or a handle of kind `k` held open: what an open leaves before the
      program has looked at its answer. -/
  | opened (k : Held) (a : UInt64)
  /-- The slot at `a` holds null or a hash table of `Lib.Ht` whose values are
      at most `n` bytes, made at most `k` times. -/
  | htAt (a : UInt64) (n k : Nat)
  /-- Memory at `a` holds a NUL-terminated string. -/
  | cstr (a : UInt64)
  /-- The device's tracker is one the race proof covers, and every access it
      records is the default stream's: every buffer is ready for it. -/
  | devSeq
  /-- Kernels and cuBLAS keep the sizes of the buffers they write. -/
  | oracles
  /-- Memory at `a` holds a NUL-terminated string within `n` bytes: what a
      step that leaves those bytes alone keeps. -/
  | cstrIn (a : UInt64) (n : Nat)
  /-- Region `r` holds at least `x` bytes, `x` a value the program is handed,
      such as the data's length. -/
  | roomArg (r : Region) (x : UInt64)
  /-- Device buffer `a` has been made, with `n` bytes while it lives: what
      every call keeps but those that start the device afresh. -/
  | devBuf (a : UInt64) (n : UInt64)
  /-- Page-locked host memory spans at most `n` bytes: what every call keeps
      but the one that allocates it. -/
  | pinnedUsed (n : Nat)
  /-- The device's streams by id, each live or destroyed. -/
  | streams (live : List Bool)
  /-- The device has made `n` events. -/
  | events (n : Nat)
  /-- Event `e` exists. -/
  | evLive (e : Nat)
  /-- Party `p` knows every access the device's tracker records: every buffer
      is ready for it. -/
  | readyAll (p : Nat)
  /-- Event `e` is recorded, outside any capture, at a clock past every access
      the tracker records: a stream that waits on it knows them all. -/
  | evAll (e : Nat)
  deriving Repr, DecidableEq


def Fact.holds : Fact → World → Prop
  | .part p b, w => p.get w = b
  | .room r n, w => n ≤ w.mem.sizes r
  | .cell a v, w => w.mem.load a 8 = some v
  | .held k a, w => k.live w a = true
  | .opened k a, w => a ≠ 0 → k.live w a = true
  | .htAt a n k, w => HtSpec.HtObj w a n k
  | .cstr a, w => CStr w.mem a
  | .devSeq, w => Device.TrackOk w.dev ∧ Device.DefaultOnly w.dev.race
  | .oracles, w => Sem.KeepsSize w.kernel ∧ VendorKeeps w.vendor
  | .cstrIn a n, w => w.mem.frozen = false ∧ CStr w.mem a ∧ a.toNat + n < 2 ^ 64 ∧
      ∃ r off k, r ≠ .pinned ∧ off + n < 2 ^ 36 ∧ Sem.strSpan w.mem a = some (r, off, k) ∧ k ≤ n
  | .roomArg r x, w => x.toNat ≤ w.mem.sizes r ∧ x.toNat ≤ 2 ^ 36
  | .devBuf a n, w => Device.BufIs (asI32 a) n.toNat w.dev
  | .pinnedUsed n, w => w.mem.sizes .pinned ≤ n
  | .streams l, w => w.dev.streams.toList = l
  | .events n, w => w.dev.events.size = n
  | .evLive e, w => (w.dev.event? e).isSome = true
  | .readyAll p, w => Device.KnowsAll w.dev.race p
  | .evAll e, w => ∃ clk, w.dev.event? e = some (some (clk, false)) ∧ Device.Covers w.dev.race clk

/-- A fact about the device's streams, events and tracker. -/
def Fact.isDev : Fact → Bool
  | .streams _ | .events _ | .evLive _ | .readyAll _ | .evAll _ => true
  | _ => false

/-- **A device fact survives a step** that leaves what it reads: the streams,
    the number of events, the events it names, and the tracker's accesses, its
    clocks only growing. -/
theorem Fact.holds_step {x : Fact} (hx : x.isDev = true) {w w' : World} (hw : x.holds w)
    (hs : (∃ l, x = .streams l) → w'.dev.streams = w.dev.streams)
    (hn : (∃ n, x = .events n) → w'.dev.events.size = w.dev.events.size)
    (he : ∀ e : Nat, (x = .evLive e ∨ x = .evAll e) → w'.dev.event? e = w.dev.event? e)
    (hl : w'.dev.race.lastW = w.dev.race.lastW) (hr : w'.dev.race.reads = w.dev.race.reads)
    (hg : DevSpec.Grows w.dev.race w'.dev.race) : x.holds w' := by
  cases x with
  | streams l => show _ = _; rw [hs ⟨l, rfl⟩]; exact hw
  | events n => show _ = _; rw [hn ⟨n, rfl⟩]; exact hw
  | evLive e => show _ = _; rw [he e (.inl rfl)]; exact hw
  | readyAll p => exact Device.KnowsAll.grow hw hl hr hg
  | evAll e =>
      obtain ⟨clk, hc, hcv⟩ := hw
      exact ⟨clk, (he e (.inr rfl)).trans hc, Device.Covers.grow hcv hl hr hg⟩
  | _ => simp [Fact.isDev] at hx

/-- A device fact survives a step that leaves the device as it was. -/
theorem Fact.holds_dev {x : Fact} (hx : x.isDev = true) {w w' : World} (h : w'.dev = w.dev) (hw : x.holds w) :
    x.holds w' :=
  Fact.holds_step hx hw (fun _ => by rw [h]) (fun _ => by rw [h]) (fun _ _ => by rw [h])
    (by rw [h]) (by rw [h]) (by rw [h]; exact DevSpec.Grows.refl _)

/-- **A typestate**: facts about the world, all of which hold. -/
abbrev TState := List Fact

def TState.holds (S : TState) (w : World) : Prop := ∀ x ∈ S, x.holds w

theorem TState.holds_cons {x : Fact} {S : TState} {w : World} (h : x.holds w) (hS : S.holds w) :
    TState.holds (x :: S) w := by
  intro q hq
  rcases List.mem_cons.mp hq with rfl | hq
  · exact h
  · exact hS q hq

/-- Whether an eight-byte cell at `a` lies in one region other than the pinned
    one, every byte of it decoding there. -/
def cellOk (a : UInt64) : Bool :=
  match decodeAddr a with
  | some (r, off) => decide (r ≠ .pinned) &&
      (List.range 8).all fun i => decide (decodeAddr (a + UInt64.ofNat i) = some (r, off + i))
  | none => false

theorem cellOk_spec {a : UInt64} (h : cellOk a = true) :
    ∃ r off, decodeAddr a = some (r, off) ∧ r ≠ .pinned ∧
      ∀ i, i < 8 → decodeAddr (a + UInt64.ofNat i) = some (r, off + i) := by
  unfold cellOk at h
  split at h
  · rename_i r off hd
    simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at h
    exact ⟨r, off, hd, h.1, h.2⟩
  · cases h

/-- Whether a call with these argument bits may write the byte at `x`, over
    every answer it may give. -/
def mayWrite (f : IR.Ffi) (bits : List UInt64) (x : UInt64) : Bool := (frame f).mayWrite bits x

theorem mayWrite_sound {f : IR.Ffi} {bits : List UInt64} {x : UInt64} (h : mayWrite f bits x = false)
    (ret : Option UInt64) : ¬ (frame f).allows bits ret x :=
  Frame.mayWrite_sound h ret

theorem mapM_asBits_sc : ∀ (bits : List UInt64), (bits.map (V.sc .i64)).mapM asBits = some bits
  | [] => rfl
  | b :: bits => by simp [List.mapM_cons, asBits, mapM_asBits_sc bits]

theorem frozen_of_load {m : Mem} {a : UInt64} {n : Nat} {v : UInt64} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned) (h : m.load a n = some v) : m.frozen = false := by
  have := (load_some hd h).2.1
  rw [readable_off_pinned hp] at this
  simpa using this

/-- **No call freezes memory**: only spawning a worker does. -/
theorem callBits_unfrozen {f : IR.Ffi} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) (hz : w.mem.frozen = false) : w'.mem.frozen = false := by
  by_cases hm : Part.frozen ∈ moves f
  · cases f <;> first | exact absurd hm (by decide) | skip
    · exact sets_threadJoin h hz
    · exact sets_threadCleanup h
  · have := callBits_parts f h .frozen hm
    simp only [Part.get] at this
    rw [this]; exact hz

/-- A byte outside a call's frame survives it. -/
theorem unchanged_of_mayWrite {f : IR.Ffi} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) {x : UInt64} (hx : mayWrite f bits x = false) :
    Unchanged w.mem w'.mem x := by
  have hcall : callFfi f (bits.map (V.sc .i64)) w = some (r, w') := by
    unfold callFfi; rw [mapM_asBits_sc]; exact h
  exact callFfi_respects_frame f _ w r w' hcall bits (mapM_asBits_sc bits) _ (mayWrite_sound hx _)

/-- **A cell outside a call's frame survives it.** -/
theorem cell_kept {f : IR.Ffi} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    {a v : UInt64} (h : callBits f bits w = some (r, w')) (hok : cellOk a = true)
    (hno : ((List.range 8).all fun i => !mayWrite f bits (a + UInt64.ofNat i)) = true)
    (hc : w.mem.load a 8 = some v) : w'.mem.load a 8 = some v := by
  obtain ⟨rg, off, hd, hp, hb⟩ := cellOk_spec hok
  have hz := callBits_unfrozen h (frozen_of_load hd hp hc)
  refine load_kept hd hp hb hc hz (callBits_rooms f h rg hp) fun i hi => ?_
  have hm : mayWrite f bits (a + UInt64.ofNat i) = false := by
    have := List.all_eq_true.mp hno i (List.mem_range.mpr hi)
    simpa using this
  exact unchanged_of_mayWrite h hm

theorem load_setLive {m : Mem} {a : UInt64} {r : Region} {off : Nat} (hd : decodeAddr a = some (r, off))
    (hp : r ≠ .pinned) (L : List (Nat × Nat)) (n : Nat) :
    ({ m with pinnedLive := L } : Mem).load a n = m.load a n := by
  unfold Mem.load
  simp only [hd, Option.bind_eq_bind, Option.bind_some]
  cases r <;> first | exact absurd rfl hp | rfl

theorem load_setFrozen {m : Mem} (hz : m.frozen = false) (a : UInt64) (n : Nat) :
    ({ m with frozen := false } : Mem).load a n = m.load a n := by
  cases m; simp_all

/-- A mover that stores `K` at the slot it is handed leaves the slot holding
    `K`. -/
theorem cell_of_store {m m' : Mem} {slot K : UInt64} (hok : cellOk slot = true)
    (h : m.store slot 8 K = some m') : m'.load slot 8 = some K := by
  obtain ⟨rg, off, hd, hp, -⟩ := cellOk_spec hok
  exact load_store8 hd hp h

/-- What a value found after `set` is: one found before, or the one set. -/
theorem Ht.get_set {h : Ht} {k v k' v' : ByteArray} (hg : (h.set k v).get k' = some v') :
    h.get k' = some v' ∨ v' = v := by
  unfold Ht.set at hg
  split at hg
  · unfold Ht.get at hg ⊢
    rw [List.find?_map] at hg
    have hpf : ((fun x : ByteArray × ByteArray => x.1.toList == k'.toList) ∘
        (fun e => if (e.1.toList == k.toList) = true then (e.1, v) else e)) =
        (fun x => x.1.toList == k'.toList) := by
      funext e; simp only [Function.comp_apply]; split <;> rfl
    rw [hpf] at hg
    cases hf : h.entries.find? (fun x => x.1.toList == k'.toList) with
    | none => rw [hf] at hg; cases hg
    | some e =>
        rw [hf] at hg
        simp only [Option.map_some, Option.some.injEq] at hg
        split at hg
        · right; simpa using hg.symm
        · left; simpa using hg
  · unfold Ht.get at hg ⊢
    rw [List.find?_append] at hg
    cases hf : h.entries.find? (·.1.toList == k'.toList) with
    | some e => left; rw [hf] at hg; simpa [hf] using hg
    | none =>
        right
        rw [hf] at hg
        simp only [Option.none_or, List.find?_cons, List.find?_nil] at hg
        split at hg
        · simpa using hg.symm
        · cases hg

/-- Whether `x` and `a` lie in two different regions. -/
def otherRegion (x a : UInt64) : Bool :=
  match decodeAddr x, decodeAddr a with
  | some (r, _), some (r', _) => r != r'
  | _, _ => false

theorem region_setRegion_other {m : Mem} {r r' : Region} (b : ByteArray) (h : r ≠ r') :
    (m.setRegion r' b).region r = m.region r := by
  cases r <;> cases r' <;> first | rfl | exact absurd rfl h

theorem store_region_other {m m' : Mem} {x : UInt64} {n : Nat} {v : UInt64} {r r' : Region} {off : Nat}
    (h : m.store x n v = some m') (hd : decodeAddr x = some (r', off)) (hr : r ≠ r') :
    m'.region r = m.region r := by
  unfold Mem.store at h
  rw [hd] at h
  try simp only [Option.bind_eq_bind, Option.bind_some, bind, Option.bind] at h
  split at h
  · cases h
  · cases h; exact region_setRegion_other _ hr

/-- **A C string survives a store into another region.** -/
theorem cstr_other {m m' : Mem} {x a : UInt64} {n : Nat} {v : UInt64} (ho : otherRegion x a = true)
    (h : m.store x n v = some m') (hc : CStr m a) : CStr m' a := by
  unfold otherRegion at ho
  cases hx : decodeAddr x with
  | none => rw [hx] at ho; cases ho
  | some p =>
    obtain ⟨r', o'⟩ := p
    cases ha : decodeAddr a with
    | none => rw [hx, ha] at ho; cases ho
    | some q =>
      obtain ⟨r, o⟩ := q
      rw [hx, ha] at ho
      have hr : r ≠ r' := by intro e; subst e; revert ho; cases r <;> exact fun h => by simp at h; cases h
      unfold CStr readCStr at hc ⊢
      rw [ha] at hc ⊢
      simp only [Option.bind_eq_bind, Option.bind_some] at hc ⊢
      rw [store_region_other h hx hr]
      exact hc

/-- The LMDB slot's initialiser writes memory only at its slot. -/
theorem slotInit_mem {f : IR.Ffi} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (hf : f = .lmdbInit) (h : callBits f bits w = some (r, w')) :
    ∃ slot, bits = [slot] ∧ ∃ v, w.mem.store slot 8 v = some w'.mem := by
  subst hf
  change ffiLmdbInit bits w = some (r, w') at h
  unfold ffiLmdbInit at h
  split at h
  · split at h
    · cases h
    · simp only [Option.bind_eq_bind] at h
      obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
      cases h; exact ⟨_, rfl, _, hm⟩
  · cases h

/-- A byte past a range that does not wrap is outside it. -/
theorem within_apart {b a : UInt64} {len n i : Nat} (hi : i < n) (ha : a.toNat + n ≤ 2 ^ 64)
    (h : b.toNat + len ≤ a.toNat ∨ (a.toNat + n ≤ b.toNat ∧ b.toNat + len ≤ 2 ^ 64)) :
    within b len (a + UInt64.ofNat i) = false := by
  simp only [within, decide_eq_false_iff_not, Nat.not_lt]
  rw [UInt64.toNat_sub, UInt64.toNat_add, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega : i < 2 ^ 64),
    Nat.mod_eq_of_lt (by omega : a.toNat + i < 2 ^ 64)]
  have := b.toNat_lt
  omega

/-- Whether `[b, b + len)` and the `n` bytes from `a` lie apart, neither
    wrapping. -/
def apart (b : UInt64) (len : Nat) (a : UInt64) (n : Nat) : Bool :=
  decide (a.toNat + n ≤ 2 ^ 64 ∧ (b.toNat + len ≤ a.toNat ∨ (a.toNat + n ≤ b.toNat ∧ b.toNat + len ≤ 2 ^ 64)))

/-- Whether a frame may write any of the `n` bytes from `a`: never for a frame
    that writes nothing, by the ranges for a frame of one range (records
    included), byte by byte otherwise. -/
def _root_.AlgorithmLib.HProg.Frame.touches (fr : Frame) (args : List UInt64) (a : UInt64) (n : Nat) :
    Bool :=
  match fr with
  | .none => false
  | .at d l => !apart (args.getD d 0) (args.getD l 0).toNat a n
  | .atOff b o l => !apart (args.getD b 0 + args.getD o 0) (args.getD l 0).toNat a n
  | .fixed d m => decide (m > 2 ^ 64) || !apart (args.getD d 0) m a n
  | .atRecords d c m => decide (m * (asI32 (args.getD c 0)).toNat > 2 ^ 64) ||
      !apart (args.getD d 0) (m * (asI32 (args.getD c 0)).toNat) a n
  | _ => (List.range n).any fun i => fr.mayWrite args (a + UInt64.ofNat i)

theorem Frame.touches_false {fr : Frame} {args : List UInt64} {a : UInt64} {n : Nat}
    (h : fr.touches args a n = false) : ∀ i, i < n → fr.mayWrite args (a + UInt64.ofNat i) = false := by
  intro i hi
  unfold Frame.touches at h
  split at h
  · rfl
  · simp only [apart, Bool.not_eq_false', decide_eq_true_eq] at h
    exact within_apart hi h.1 h.2
  · simp only [apart, Bool.not_eq_false', decide_eq_true_eq] at h
    exact within_apart hi h.1 h.2
  · simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not, apart, Bool.not_eq_false',
      decide_eq_true_eq] at h
    simp only [Frame.mayWrite, Bool.or_eq_false_iff, decide_eq_false_iff_not]
    exact ⟨h.1, within_apart hi h.2.1 h.2.2⟩
  · simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not, apart, Bool.not_eq_false',
      decide_eq_true_eq] at h
    simp only [Frame.mayWrite, Bool.or_eq_false_iff, decide_eq_false_iff_not]
    exact ⟨h.1, within_apart hi h.2.1 h.2.2⟩
  · simp only [List.any_eq_false, List.mem_range, Bool.not_eq_true] at h
    exact h i hi

/-- **A C string within `n` bytes is kept** by a step that keeps memory
    unfrozen, keeps every region's size and leaves those bytes alone. -/
theorem cstrIn_kept {w w' : World} {a : UInt64} {n : Nat} (hx : (Fact.cstrIn a n).holds w)
    (hz' : w'.mem.frozen = false) (hsz : ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r)
    (hu : ∀ i, i < n → Unchanged w.mem w'.mem (a + UInt64.ofNat i)) : (Fact.cstrIn a n).holds w' := by
  obtain ⟨hz, hc, hwrap, r, off, k, hp, hspan, hsp, hk⟩ := hx
  have hd := strSpan_decode hsp
  have hbytes : ∀ i, i < k → decodeAddr (a + UInt64.ofNat i) = some (r, off + i) := fun i hi =>
    Sem.decodeAddr_add hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  have hb := bytes_kept hp hbytes (strSpan_bound hsp) hz hz' (hsz r hp) (fun i hi => hu i (by omega))
  obtain ⟨hsp', hrd⟩ := strSpan_congr hsp (hsz r hp) hb
  refine ⟨hz', ?_, hwrap, r, off, k, hp, hspan, hsp', hk⟩
  unfold CStr; rw [hrd]; exact hc

/-- Whether a call with these argument bits keeps a fact, whatever it answers. -/
def keeps (f : IR.Ffi) (bits : List UInt64) : Fact → Bool
  | .part p _ => !(moves f).contains p
  | .room r _ => r != .pinned
  | .cell a _ => cellOk a && (List.range 8).all fun i => !mayWrite f bits (a + UInt64.ofNat i)
  | .held _ _ | .opened _ _ => false
  | .cstr a => f == .lmdbInit && otherRegion (bits.headD 0) a
  | .htAt _ _ _ => false
  | .devSeq => Device.devUntouched.contains f || Device.seqKeepers.contains f
  | .oracles => true
  | .cstrIn a n => !(frame f).touches bits a n
  | .roomArg r _ => r != .pinned
  | .devBuf _ _ => Device.devUntouched.contains f || Device.bufKeepers.contains f
  | .pinnedUsed _ => f != .cudaPinnedAlloc
  | .streams _ | .events _ | .evLive _ | .readyAll _ | .evAll _ => Device.devUntouched.contains f

theorem keeps_sound {f : IR.Ffi} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) {x : Fact} (hk : keeps f bits x = true) (hx : x.holds w) :
    x.holds w' := by
  cases x with
  | part p b =>
      have hn : p ∉ moves f := fun hin => by simp_all [keeps]
      exact (callBits_parts f h p hn).trans hx
  | room rg n =>
      have hp : rg ≠ .pinned := by intro e; subst e; exact absurd hk (by simp only [keeps]; decide)
      show n ≤ w'.mem.sizes rg
      rw [callBits_rooms f h rg hp]; exact hx
  | cell a v =>
      simp only [keeps, Bool.and_eq_true] at hk
      exact cell_kept h hk.1 hk.2 hx
  | held _ _ | opened _ _ => simp [keeps] at hk
  | cstr a =>
      simp only [keeps, Bool.and_eq_true, beq_iff_eq] at hk
      obtain ⟨slot, rfl, _, hm⟩ := slotInit_mem hk.1 h
      exact cstr_other hk.2 hm hx
  | devSeq =>
      obtain ⟨ht, hd⟩ := hx
      refine ⟨Device.callBits_track f h ht, ?_⟩
      simp only [keeps, Bool.or_eq_true, List.contains_iff_mem] at hk
      rcases hk with hu | hq
      · rw [Device.callBits_dev_same f hu h]; exact hd
      · exact Device.callBits_seq f hq h hd
  | oracles =>
      obtain ⟨e1, e2⟩ := callBits_oracles f h
      show Sem.KeepsSize w'.kernel ∧ VendorKeeps w'.vendor
      rw [e1, e2]; exact hx
  | cstrIn a n =>
      refine cstrIn_kept hx (callBits_unfrozen h hx.1) (fun rg hp => callBits_rooms f h rg hp) fun i hi => ?_
      refine unchanged_of_mayWrite h ?_
      simp only [keeps, Bool.not_eq_true'] at hk
      exact Frame.touches_false hk i hi
  | roomArg rg x =>
      have hp : rg ≠ .pinned := by intro e; subst e; exact absurd hk (by simp only [keeps]; decide)
      show x.toNat ≤ w'.mem.sizes rg ∧ _
      rw [callBits_rooms f h rg hp]; exact hx
  | devBuf a n =>
      simp only [keeps, Bool.or_eq_true, List.contains_iff_mem] at hk
      rcases hk with hu | hq
      · show Device.BufIs _ _ w'.dev; rw [Device.callBits_dev_same f hu h]; exact hx
      · exact Device.callBits_bufs f hq h hx
  | pinnedUsed n =>
      have hf : f ≠ .cudaPinnedAlloc := by intro e; subst e; simp [keeps] at hk
      show w'.mem.sizes .pinned ≤ n
      rw [callBits_pinned f hf h]; exact hx
  | streams _ | events _ | evLive _ | readyAll _ | evAll _ =>
      simp only [keeps, List.contains_iff_mem] at hk
      exact Fact.holds_dev rfl (Device.callBits_dev_same f hk h) hx
  | htAt _ _ _ => simp [keeps] at hk

theorem kept_sound {f : IR.Ffi} {bits : List UInt64} {S : TState} {w : World} {r : Option V} {w' : World}
    (hS : S.holds w) (h : callBits f bits w = some (r, w')) : TState.holds (S.filter (keeps f bits)) w' := by
  intro x hx
  obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
  exact keeps_sound h hk (hS x hm)

/-- The cell a mover leaves at its slot, when the slot is a cell. -/
def slotCell (bits : List UInt64) (v : UInt64) (S : TState) : TState :=
  if cellOk (bits.headD 0) then .cell (bits.headD 0) v :: S else S

theorem slotCell_holds {bits : List UInt64} {v : UInt64} {S : TState} {w' : World}
    (hc : ∀ slot, bits = [slot] → cellOk slot = true → w'.mem.load slot 8 = some v)
    (hb : ∃ slot, bits = [slot]) (hS : S.holds w') : (slotCell bits v S).holds w' := by
  obtain ⟨slot, rfl⟩ := hb
  unfold slotCell
  split
  · exact TState.holds_cons (hc slot rfl (by assumption)) hS
  · exact hS

/-- **What a call leaves of a typestate, but the device's streams and
    events**: the facts it keeps, and what a lifecycle call makes: its parts,
    and the handle it writes to its slot. A cleanup whose slot the typestate
    knows is resolved by it. -/
def afterLife (f : IR.Ffi) (bits : List UInt64) (S : TState) : TState :=
  let k := S.filter (keeps f bits)
  let slot := bits.headD 0
  match f with
  | .cudaInit =>
      if S.contains (.part .cudaDevice true) then
        .devSeq :: .part .cuda true :: .part .capturing false :: slotCell bits cudaCtx k
      else if S.contains (.part .cudaDevice false) then
        .devSeq :: .part .cuda false :: .part .capturing false :: slotCell bits 0 k
      else .devSeq :: .part .capturing false :: k
  | .cudaCleanup => .devSeq :: .part .cuda false :: .part .capturing false :: slotCell bits 0 k
  | .lmdbInit => .part .lmdb true :: .part .lmdbEmpty true :: slotCell bits lmdbCtx k
  | .gpuInit =>
      if S.contains (.part .gpuAdapter true) then .part .gpu true :: slotCell bits gpuCtx k
      else if S.contains (.part .gpuAdapter false) then .part .gpu false :: slotCell bits 0 k
      else k
  | .windowInit =>
      if S.contains (.part .display true) then .part .win true :: slotCell bits winCtx k
      else if S.contains (.part .display false) then slotCell bits 0 k
      else k
  | .gpuCleanup => .part .gpu false :: slotCell bits 0 k
  | .threadInit => .part .thread true :: slotCell bits threadCtx k
  | .lmdbCleanup =>
      if S.contains (.cell slot 0) then S
      else if S.contains (.cell slot lmdbCtx) then .part .lmdb false :: slotCell bits 0 k else k
  | .windowCleanup =>
      if S.contains (.cell slot 0) then S
      else if S.contains (.cell slot winCtx) then .part .win false :: slotCell bits 0 k else k
  | .threadCleanup =>
      .part .frozen false ::
        if S.contains (.cell slot threadCtx) && cellOk slot then .part .thread false :: .cell slot 0 :: k
        else k
  | .threadJoin => if S.contains (.part .frozen false) then .part .frozen false :: k else k
  | .cudaPinnedAlloc =>
      S.filterMap (fun | .pinnedUsed n => some (.pinnedUsed (n + 63 + (bits.getD 1 0).toNat)) | _ => none) ++ k
  | _ => k

/-- A one-slot entry answers only on one argument. -/
theorem one_slot {bits : List UInt64} {α : Type} {g : UInt64 → Option α} {x : α}
    (h : (match bits with | [slot] => g slot | _ => none) = some x) : ∃ slot, bits = [slot] := by
  match bits, h with
  | [slot], _ => exact ⟨slot, rfl⟩

theorem cell_storeAt {w w' : World} {m' : Mem} {slot K : UInt64} (hok : cellOk slot = true)
    (hm : w.mem.store slot 8 K = some m') (hw : w'.mem = m') : w'.mem.load slot 8 = some K := by
  rw [hw]; exact cell_of_store hok hm

theorem holds_of_mem {S : TState} {w : World} {x : Fact} (hS : S.holds w) (h : S.contains x = true) :
    x.holds w := hS x (by simpa using h)

theorem slotCell_one {slot v : UInt64} {S : TState} {w' : World}
    (hc : cellOk slot = true → w'.mem.load slot 8 = some v) (hS : S.holds w') :
    (slotCell [slot] v S).holds w' := by
  unfold slotCell
  split
  · exact TState.holds_cons (hc (by assumption)) hS
  · exact hS

/-- What a mover that stores `K` at its one slot leaves there. -/
def StoresAt (K : UInt64) (bits : List UInt64) (w' : World) : Prop :=
  ∃ slot, bits = [slot] ∧ (cellOk slot = true → w'.mem.load slot 8 = some K)

/-- What `cudaInit` stores at its slot: the context with a device, null
    without one. -/
theorem stores_cudaInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaInit bits w = some (r, w')) :
    StoresAt (if w.cudaDevice then cudaCtx else 0) bits w' := by
  change ffiCudaInit bits w = some (r, w') at h
  unfold ffiCudaInit at h
  split at h
  · rename_i slot
    refine ⟨slot, rfl, fun hok => ?_⟩
    split at h <;> rename_i hd <;> simp only [hd, if_true, if_false, Bool.false_eq_true] <;>
      obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h <;> cases h <;> exact cell_of_store hok hm
  · cases h

theorem stores_cudaCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaCleanup bits w = some (r, w')) : StoresAt 0 bits w' := by
  change ffiCudaCleanup bits w = some (r, w') at h
  unfold ffiCudaCleanup at h
  split at h
  · rename_i slot
    refine ⟨slot, rfl, fun hok => ?_⟩
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    cases h
    obtain ⟨rg, off, hd, hp, -⟩ := cellOk_spec hok
    show ({ m with pinnedLive := [] } : Mem).load slot 8 = some 0
    rw [load_setLive hd hp]
    exact cell_of_store hok hm
  · cases h

theorem stores_lmdbInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbInit bits w = some (r, w')) : StoresAt lmdbCtx bits w' := by
  change ffiLmdbInit bits w = some (r, w') at h
  unfold ffiLmdbInit at h
  split at h
  · rename_i slot
    refine ⟨slot, rfl, fun hok => ?_⟩
    split at h
    · cases h
    · obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
      cases h
      exact cell_of_store hok hm
  · cases h

/-- What `windowInit` leaves: with a display, the window's context at its slot
    and the window live; without one, null there. -/
theorem windowInit_display {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowInit bits w = some (r, w')) :
    (w.display = true → StoresAt winCtx bits w' ∧ w'.win.live = true) ∧
    (w.display = false → StoresAt 0 bits w') := by
  change ffiWindowInit bits w = some (r, w') at h
  unfold ffiWindowInit at h
  split at h
  · rename_i slot
    split at h
    · cases h
    · split at h
      · rename_i hd
        obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
        cases h
        exact ⟨fun _ => ⟨⟨slot, rfl, fun hok => cell_of_store hok hm⟩, rfl⟩, fun h0 => by simp_all⟩
      · rename_i hd
        obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
        cases h
        exact ⟨fun h1 => absurd h1 hd, fun _ => ⟨slot, rfl, fun hok => cell_of_store hok hm⟩⟩
  · cases h

/-- What `gpuInit` stores at its slot: the context with an adapter, null
    without one. -/
theorem stores_gpuInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuInit bits w = some (r, w')) :
    StoresAt (if w.gpuAdapter then gpuCtx else 0) bits w' := by
  change ffiGpuInit bits w = some (r, w') at h
  unfold ffiGpuInit at h
  split at h
  · rename_i slot
    refine ⟨slot, rfl, fun hok => ?_⟩
    split at h <;> rename_i hd <;> simp only [hd, if_true, if_false, Bool.false_eq_true] <;>
      obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h <;> cases h <;> exact cell_of_store hok hm
  · cases h

theorem stores_gpuCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCleanup bits w = some (r, w')) : StoresAt 0 bits w' := by
  change ffiGpuCleanup bits w = some (r, w') at h
  unfold ffiGpuCleanup at h
  split at h
  · rename_i slot
    refine ⟨slot, rfl, fun hok => ?_⟩
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    cases h
    exact cell_of_store hok hm
  · cases h

theorem stores_threadInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadInit bits w = some (r, w')) : StoresAt threadCtx bits w' := by
  change ffiThreadInit bits w = some (r, w') at h
  unfold ffiThreadInit at h
  split at h
  · rename_i slot
    refine ⟨slot, rfl, fun hok => ?_⟩
    split at h
    · cases h
    · obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
      cases h
      exact cell_of_store hok hm
  · cases h

/-- A cleanup handed a slot holding zero changes nothing. -/
theorem lmdbCleanup_zero {slot : UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCleanup [slot] w = some (r, w')) (hc : w.mem.load slot 8 = some 0) : w' = w := by
  change ffiLmdbCleanup [slot] w = some (r, w') at h
  simp only [ffiLmdbCleanup, hc, Option.bind_eq_bind, Option.bind_some, BEq.rfl, if_true,
    Option.some.injEq, Prod.mk.injEq] at h
  exact h.2.symm

/-- A cleanup handed a slot holding its handle ends the lifecycle and zeroes the slot. -/
theorem lmdbCleanup_ctx {slot : UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCleanup [slot] w = some (r, w')) (hc : w.mem.load slot 8 = some lmdbCtx) :
    w'.lmdb.live = false ∧ (cellOk slot = true → w'.mem.load slot 8 = some 0) := by
  change ffiLmdbCleanup [slot] w = some (r, w') at h
  simp only [ffiLmdbCleanup, hc, Option.bind_eq_bind, Option.bind_some] at h
  have h0 : (lmdbCtx == 0) = false := by decide
  rw [h0, if_neg (by decide)] at h
  split at h
  · obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    cases h
    exact ⟨rfl, fun hok => cell_of_store hok hm⟩
  · cases h

theorem windowCleanup_zero {slot : UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowCleanup [slot] w = some (r, w')) (hc : w.mem.load slot 8 = some 0) : w' = w := by
  change ffiWindowCleanup [slot] w = some (r, w') at h
  simp only [ffiWindowCleanup, hc, Option.bind_eq_bind, Option.bind_some, BEq.rfl, if_true,
    Option.some.injEq, Prod.mk.injEq] at h
  exact h.2.symm

theorem windowCleanup_ctx {slot : UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowCleanup [slot] w = some (r, w')) (hc : w.mem.load slot 8 = some winCtx) :
    w'.win.live = false ∧ (cellOk slot = true → w'.mem.load slot 8 = some 0) := by
  change ffiWindowCleanup [slot] w = some (r, w') at h
  simp only [ffiWindowCleanup, hc, Option.bind_eq_bind, Option.bind_some] at h
  have h0 : (winCtx == 0) = false := by decide
  rw [h0, if_neg (by decide)] at h
  split at h
  · obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    cases h
    exact ⟨rfl, fun hok => cell_of_store hok hm⟩
  · cases h

theorem threadCleanup_ctx {slot : UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadCleanup [slot] w = some (r, w')) (hok : cellOk slot = true)
    (hc : w.mem.load slot 8 = some threadCtx) :
    w'.thread.live = false ∧ w'.mem.load slot 8 = some 0 := by
  obtain ⟨rg, off, hd, hp, -⟩ := cellOk_spec hok
  have hz := frozen_of_load hd hp hc
  change ffiThreadCleanup [slot] w = some (r, w') at h
  simp only [ffiThreadCleanup, load_setFrozen hz, hc, Option.bind_eq_bind, Option.bind_some] at h
  have h0 : (threadCtx == 0) = false := by decide
  rw [h0, if_neg (by decide)] at h
  split at h
  · obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    cases h
    exact ⟨rfl, cell_of_store hok hm⟩
  · cases h

theorem TState.holds_cell {S : TState} {w : World} {a v : UInt64} (hS : S.holds w)
    (h : S.contains (.cell a v) = true) : w.mem.load a 8 = some v := holds_of_mem hS h

theorem TState.holds_cell_mem {S : TState} {w : World} {a v : UInt64} (hS : S.holds w)
    (h : Fact.cell a v ∈ S) : w.mem.load a 8 = some v := hS _ h

/-- `exactly b n` has `n` bytes. -/
theorem exactly_size' (b : ByteArray) (n : Nat) : (exactly b n).size = n := by
  rw [exactly, ByteArray.size_append, ByteArray.size_extract]
  have : (ByteArray.mk (Array.replicate (n - b.size) 0)).size = n - b.size := Array.size_replicate ..
  rw [this]; omega

/-- **An allocation of page-locked memory grows it by at most its size**, and
    the alignment before it. -/
theorem pinnedAlloc_grows {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedAlloc bits w = some (r, w')) :
    w'.mem.sizes .pinned ≤ w.mem.sizes .pinned + 63 + (bits.getD 1 0).toNat := by
  change ffiCudaPinnedAlloc bits w = some (r, w') at h
  unfold ffiCudaPinnedAlloc at h
  split at h
  · rename_i ctx size
    simp only [List.getD_cons_succ, List.getD_cons_zero]
    split at h
    · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; omega
    · obtain ⟨ok, -, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; omega
      · dsimp only at h
        split at h
        · cases h
        · simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨-, rfl⟩ := h
          show (w.mem.pinned ++ (ByteArray.mk (Array.replicate _ 0) ++ _)).size ≤ w.mem.pinned.size + 63 + size.toNat
          rw [ByteArray.size_append, ByteArray.size_append, exactly_size']
          have : (ByteArray.mk (Array.replicate ((64 - w.mem.pinned.size % 64) % 64) (0 : UInt8))).size =
              (64 - w.mem.pinned.size % 64) % 64 := Array.size_replicate ..
          rw [this]
          have := Nat.mod_lt (64 - w.mem.pinned.size % 64) (by decide : 64 > 0)
          omega
  · cases h

/-- **What `afterLife` says of a typestate is true after the call.** -/
theorem afterLife_sound {f : IR.Ffi} {bits : List UInt64} {S : TState} {w : World} {r : Option V}
    {w' : World} (hS : S.holds w) (h : callBits f bits w = some (r, w')) : (afterLife f bits S).holds w' := by
  have hk := kept_sound hS h
  cases f
  case cudaInit =>
    have hq : Fact.holds .devSeq w' := ⟨Device.track_cudaInit h, Device.seq_cudaInit h⟩
    have hs := sets_cudaInit h
    have hc0 := stores_cudaInit h
    have hcap : Fact.holds (.part .capturing false) w' := hs.2
    change TState.holds (if S.contains (.part .cudaDevice true) = true then
        .devSeq :: .part .cuda true :: .part .capturing false :: slotCell bits cudaCtx (S.filter (keeps .cudaInit bits))
      else if S.contains (.part .cudaDevice false) = true then
        .devSeq :: .part .cuda false :: .part .capturing false :: slotCell bits 0 (S.filter (keeps .cudaInit bits))
      else .devSeq :: .part .capturing false :: S.filter (keeps .cudaInit bits)) w'
    split
    · rename_i h0
      have hd : w.cudaDevice = true := holds_of_mem hS h0
      obtain ⟨slot, rfl, hc⟩ := hc0
      rw [hd, if_pos rfl] at hc
      exact TState.holds_cons hq (TState.holds_cons (show w'.dev.live = true by rw [hs.1, hd])
        (TState.holds_cons hcap (slotCell_one hc hk)))
    · split
      · rename_i _ h0
        have hd : w.cudaDevice = false := holds_of_mem hS h0
        obtain ⟨slot, rfl, hc⟩ := hc0
        rw [hd] at hc
        exact TState.holds_cons hq (TState.holds_cons (show w'.dev.live = false by rw [hs.1, hd])
          (TState.holds_cons hcap (slotCell_one hc hk)))
      · exact TState.holds_cons hq (TState.holds_cons hcap hk)
  case cudaCleanup =>
    have hq : Fact.holds .devSeq w' := ⟨Device.track_cudaCleanup h, Device.seq_cudaCleanup h⟩
    obtain ⟨slot, rfl, hc⟩ := stores_cudaCleanup h
    exact TState.holds_cons hq
      (TState.holds_cons (sets_cudaCleanup h).1 (TState.holds_cons (sets_cudaCleanup h).2 (slotCell_one hc hk)))
  case cudaPinnedAlloc =>
    intro x hx
    rcases List.mem_append.mp hx with hx | hx
    · obtain ⟨y, hy, hyx⟩ := List.mem_filterMap.mp hx
      cases y with
      | pinnedUsed n =>
          simp only [Option.some.injEq] at hyx; subst hyx
          have hn : w.mem.sizes .pinned ≤ n := hS _ hy
          show w'.mem.sizes .pinned ≤ n + 63 + (bits.getD 1 0).toNat
          have := pinnedAlloc_grows h
          omega
      | _ => simp at hyx
    · exact hk x hx
  case lmdbInit =>
    obtain ⟨slot, rfl, hc⟩ := stores_lmdbInit h
    exact TState.holds_cons (sets_lmdbInit h) (TState.holds_cons (empties_lmdbInit h) (slotCell_one hc hk))
  case gpuInit =>
    have hs := sets_gpuInit h
    have hc0 := stores_gpuInit h
    change TState.holds (if S.contains (.part .gpuAdapter true) = true then
        .part .gpu true :: slotCell bits gpuCtx (S.filter (keeps .gpuInit bits))
      else if S.contains (.part .gpuAdapter false) = true then
        .part .gpu false :: slotCell bits 0 (S.filter (keeps .gpuInit bits))
      else S.filter (keeps .gpuInit bits)) w'
    split
    · rename_i h0
      have hd : w.gpuAdapter = true := holds_of_mem hS h0
      obtain ⟨slot, rfl, hc⟩ := hc0
      rw [hd, if_pos rfl] at hc
      exact TState.holds_cons (show w'.gpu.live = true by rw [hs, hd]) (slotCell_one hc hk)
    · split
      · rename_i _ h0
        have hd : w.gpuAdapter = false := holds_of_mem hS h0
        obtain ⟨slot, rfl, hc⟩ := hc0
        rw [hd] at hc
        exact TState.holds_cons (show w'.gpu.live = false by rw [hs, hd]) (slotCell_one hc hk)
      · exact hk
  case gpuCleanup =>
    obtain ⟨slot, rfl, hc⟩ := stores_gpuCleanup h
    exact TState.holds_cons (sets_gpuCleanup h) (slotCell_one hc hk)
  case windowInit =>
    have hi := windowInit_display h
    change TState.holds (if S.contains (.part .display true) = true then
        .part .win true :: slotCell bits winCtx (S.filter (keeps .windowInit bits))
      else if S.contains (.part .display false) = true then slotCell bits 0 (S.filter (keeps .windowInit bits))
      else S.filter (keeps .windowInit bits)) w'
    split
    · rename_i h0
      obtain ⟨⟨slot, rfl, hc⟩, hl⟩ := hi.1 (holds_of_mem hS h0)
      exact TState.holds_cons hl (slotCell_one hc hk)
    · split
      · rename_i _ h0
        obtain ⟨slot, rfl, hc⟩ := hi.2 (holds_of_mem hS h0)
        exact slotCell_one hc hk
      · exact hk
  case threadInit =>
    obtain ⟨slot, rfl, hc⟩ := stores_threadInit h
    exact TState.holds_cons (sets_threadInit h) (slotCell_one hc hk)
  case lmdbCleanup =>
    obtain ⟨slot, rfl⟩ : ∃ slot, bits = [slot] := by
      change ffiLmdbCleanup bits w = some (r, w') at h
      unfold ffiLmdbCleanup at h
      split at h
      · exact ⟨_, rfl⟩
      · cases h
    change TState.holds (if S.contains (.cell slot 0) = true then S
      else if S.contains (.cell slot lmdbCtx) = true then
        .part .lmdb false :: slotCell [slot] 0 (S.filter (keeps .lmdbCleanup [slot]))
      else S.filter (keeps .lmdbCleanup [slot])) w'
    split
    · rename_i h0; rw [lmdbCleanup_zero h (TState.holds_cell hS h0)]; exact hS
    · split
      · rename_i hc
        have := lmdbCleanup_ctx h (TState.holds_cell hS hc)
        exact TState.holds_cons this.1 (slotCell_one this.2 hk)
      · exact hk
  case windowCleanup =>
    obtain ⟨slot, rfl⟩ : ∃ slot, bits = [slot] := by
      change ffiWindowCleanup bits w = some (r, w') at h
      unfold ffiWindowCleanup at h
      split at h
      · exact ⟨_, rfl⟩
      · cases h
    change TState.holds (if S.contains (.cell slot 0) = true then S
      else if S.contains (.cell slot winCtx) = true then
        .part .win false :: slotCell [slot] 0 (S.filter (keeps .windowCleanup [slot]))
      else S.filter (keeps .windowCleanup [slot])) w'
    split
    · rename_i h0; rw [windowCleanup_zero h (TState.holds_cell hS h0)]; exact hS
    · split
      · rename_i hc
        have := windowCleanup_ctx h (TState.holds_cell hS hc)
        exact TState.holds_cons this.1 (slotCell_one this.2 hk)
      · exact hk
  case threadCleanup =>
    obtain ⟨slot, rfl⟩ : ∃ slot, bits = [slot] := by
      change ffiThreadCleanup bits w = some (r, w') at h
      unfold ffiThreadCleanup at h
      split at h
      · exact ⟨_, rfl⟩
      · cases h
    change TState.holds (.part .frozen false ::
      if (S.contains (.cell slot threadCtx) && cellOk slot) = true then
        .part .thread false :: .cell slot 0 :: S.filter (keeps .threadCleanup [slot])
      else S.filter (keeps .threadCleanup [slot])) w'
    refine TState.holds_cons (sets_threadCleanup h) ?_
    split
    · rename_i hc
      rw [Bool.and_eq_true] at hc
      have := threadCleanup_ctx h hc.2 (TState.holds_cell hS hc.1)
      exact TState.holds_cons this.1 (TState.holds_cons this.2 hk)
    · exact hk
  case threadJoin =>
    change TState.holds (if S.contains (.part .frozen false) = true then
      .part .frozen false :: S.filter (keeps .threadJoin bits) else S.filter (keeps .threadJoin bits)) w'
    split
    · rename_i hz
      exact TState.holds_cons (sets_threadJoin h (holds_of_mem hS hz)) hk
    · exact hk
  all_goals exact hk

-- ---------------------------------------------------------------------------
-- What a room gives
-- ---------------------------------------------------------------------------

theorem TState.holds_app {S T : TState} {w : World} (hS : S.holds w) (hT : T.holds w) :
    TState.holds (S ++ T) w :=
  fun x hx => (List.mem_append.mp hx).elim (hS x) (hT x)

theorem TState.holds_if {c : Bool} {T : TState} {w : World} (h : c = true → T.holds w) :
    TState.holds (if c then T else []) w := by
  split
  · exact h ‹_›
  · exact fun _ e => nomatch e

theorem TState.holds_one {x : Fact} {w : World} (h : x.holds w) : TState.holds [x] w :=
  TState.holds_cons h fun _ e => nomatch e

/-- A fact about the device's streams and events alone. -/
def Fact.isMeta : Fact → Bool
  | .streams _ | .events _ | .evLive _ => true
  | _ => false

/-- A fact about streams and events survives a step that leaves them. -/
theorem Fact.holds_meta {x : Fact} (hx : x.isMeta = true) {w w' : World} (hw : x.holds w)
    (hs : w'.dev.streams = w.dev.streams) (he : w'.dev.events = w.dev.events) : x.holds w' := by
  cases x with
  | streams l => show _ = _; rw [hs]; exact hw
  | events n => show _ = _; rw [he]; exact hw
  | evLive e => show (Dev.event? _ _).isSome = _; simp only [Dev.event?, he] at hw ⊢; exact hw
  | _ => simp [Fact.isMeta] at hx

/-- The streams a typestate names. -/
def streamsOf (S : TState) : Option (List Bool) := S.findSome? fun | .streams l => some l | _ => none

theorem streamsOf_holds {S : TState} {w : World} {l : List Bool} (hS : S.holds w) (h : streamsOf S = some l) :
    w.dev.streams.toList = l := by
  obtain ⟨x, hx, hf⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp only [reduceCtorEq, Option.some.injEq] at hf
  subst hf; exact hS _ hx

/-- How many events a typestate says the device has made. -/
def eventsOf (S : TState) : Option Nat := S.findSome? fun | .events n => some n | _ => none

theorem eventsOf_holds {S : TState} {w : World} {n : Nat} (hS : S.holds w) (h : eventsOf S = some n) :
    w.dev.events.size = n := by
  obtain ⟨x, hx, hf⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp only [reduceCtorEq, Option.some.injEq] at hf
  subst hf; exact hS _ hx

/-- The party a stream id names, by the streams `l`. -/
def partyOf (l : List Bool) (sid : UInt64) : Option Nat :=
  if asI32 sid < 0 then some defaultParty
  else if l.getD (asI32 sid).toNat false then some (streamParty (asI32 sid).toNat) else none

theorem partyOf_eq {d : Dev} {l : List Bool} (h : d.streams.toList = l) (sid : UInt64) :
    d.party? (asI32 sid) = partyOf l sid := by
  have hg : ∀ i, d.streams.getD i false = l.getD i false := fun i => by
    rw [← h, Array.getD_eq_getD_getElem?, List.getD_eq_getElem?_getD, Array.getElem?_toList]
  simp only [Dev.party?, partyOf, hg]

/-- Whether the typestate says `p` knows every access: by `p`'s fact, by the
    host's, or, for the default stream, by `devSeq`. -/
def readyIn (S : TState) (p : Nat) : Bool :=
  S.contains (.readyAll p) || S.contains (.readyAll hostParty) || (p == defaultParty && S.contains .devSeq)

theorem readyIn_sound {S : TState} {w : World} {p : Nat} (hS : S.holds w) (h : readyIn S p = true) :
    Device.KnowsAll w.dev.race p := by
  simp only [readyIn, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, List.contains_iff_mem] at h
  rcases h with (h | h) | ⟨rfl, h⟩
  · exact hS _ h
  · exact Device.KnowsAll.of_host (hS _ h) p
  · obtain ⟨ht, hd⟩ := (hS _ h : Device.TrackOk _ ∧ _)
    obtain ⟨evs, hc, -⟩ := ht.outer
    exact Device.KnowsAll.of_seq hc.ownBelow hd

/-- An event id a fact names: a nonnegative `i32`. -/
def evId (eid : UInt64) : Option Nat := if 0 ≤ asI32 eid then some (asI32 eid).toNat else none

theorem evId_spec {eid : UInt64} {e : Nat} (h : evId eid = some e) : ((e : Int) = asI32 eid) := by
  unfold evId at h
  split at h
  · cases h; omega
  · cases h

theorem event?_nat (d : Dev) (e : Nat) : d.event? e = (d.events[e]?).join := by
  unfold Dev.event?; rw [if_neg (by omega)]; simp

theorem event?_lt {d : Dev} {e : Nat} (h : (d.event? e).isSome = true) : e < d.events.size := by
  rw [event?_nat] at h
  rcases Nat.lt_or_ge e d.events.size with hl | hl
  · exact hl
  · rw [Array.getElem?_eq_none hl] at h; cases h

/-- **The device facts a call leaves**: of streams, events, and who knows
    every access, what each device call makes, keeps, or passes on; nothing
    for every other call, whose `keeps` decides. -/
def devAfter (f : IR.Ffi) (bits : List UInt64) (S : TState) : TState :=
  match f, bits with
  | .cudaInit, _ | .cudaCleanup, _ => [.streams [], .events 0, .readyAll hostParty]
  | .cudaStreamCreate, [c] =>
      match streamsOf S with
      | some l =>
          if c == cudaCtx && S.contains (.part .cuda true) then
            .streams (l ++ [true]) :: S.filter (fun x => x.isDev && !(x matches .streams _)) ++
              (if readyIn S defaultParty then [.readyAll (streamParty l.length)] else [])
          else []
      | none => []
  | .cudaEventCreate, [c] =>
      match eventsOf S with
      | some n =>
          if c == cudaCtx && S.contains (.part .cuda true) then
            .events (n + 1) :: .evLive n :: S.filter (fun x => x.isDev && !(x matches .events _))
          else []
      | none => []
  | .cudaEventRecord, [c, eid, sid] =>
      match streamsOf S, evId eid with
      | some l, some e =>
          match partyOf l sid with
          | some p =>
              if c == cudaCtx && S.contains (.part .cuda true) && S.contains (.part .capturing false) && S.contains (.evLive e) then
                .evLive e :: S.filter (fun x => x.isDev && x != .evAll e && x != .evLive e) ++
                  (if readyIn S p then [.evAll e] else [])
              else []
          | none => []
      | _, _ => []
  | .cudaStreamWaitEvent, [c, sid, eid] =>
      match streamsOf S, evId eid with
      | some l, some e =>
          match partyOf l sid with
          | some p =>
              if c == cudaCtx && S.contains (.part .cuda true) && S.contains (.part .capturing false) && S.contains (.evAll e) then .readyAll p :: S.filter Fact.isDev
              else []
          | none => []
      | _, _ => []
  | .cudaStreamSync, [c, sid] =>
      match streamsOf S with
      | some l =>
          match partyOf l sid with
          | some p =>
              if c == cudaCtx && S.contains (.part .cuda true) && S.contains (.part .capturing false) then
                S.filter Fact.isDev ++ (if readyIn S p then [.readyAll hostParty] else [])
              else []
          | none => []
      | none => []
  | .cudaLaunch, _ | .cudaLaunchNamed, _ =>
      if S.contains (.part .capturing false) then
        S.filter Fact.isMeta ++ (if readyIn S defaultParty then [.readyAll defaultParty] else [])
      else []
  | .cudaLaunchOnStream, [_, _, _, _, _, _, _, _, _, _, sid] =>
      match streamsOf S with
      | some l =>
          match partyOf l sid with
          | some p => if S.contains (.part .capturing false) then S.filter Fact.isMeta ++ (if readyIn S p then [.readyAll p] else []) else []
          | none => []
      | none => []
  | .cudaUpload, _ | .cudaUploadOffset, _ | .cudaDownload, _ | .cudaDownloadOffset, _ =>
      S.filter Fact.isMeta ++ (if S.contains (.readyAll hostParty) then [.readyAll hostParty] else []) ++
        (if readyIn S defaultParty then [.readyAll defaultParty] else [])
  | .cudaCreateBuffer, _ => S.filter Fact.isDev
  | _, _ => []

/-- What a launch on `p` leaves of the device facts. -/
theorem devAfter_launch {S : TState} {w w' : World} {p : Nat} (hS : S.holds w)
    (hl : Device.LaunchStep p w.dev w'.dev) :
    TState.holds (S.filter Fact.isMeta ++ (if readyIn S p then [.readyAll p] else [])) w' := by
  rcases hl with he | ⟨ns, hop, hs, hev⟩
  · refine TState.holds_app (fun x hx => ?_) (TState.holds_if fun hr => TState.holds_one ?_)
    · obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
      exact Fact.holds_meta hk (hS x hm) (by rw [he]) (by rw [he])
    · show Device.KnowsAll _ _; rw [he]; exact readyIn_sound hS hr
  · refine TState.holds_app (fun x hx => ?_) (TState.holds_if fun hr => TState.holds_one ?_)
    · obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
      exact Fact.holds_meta hk (hS x hm) hs hev
    · exact (readyIn_sound hS hr).op_self hop

/-- What a synchronous copy leaves of the device facts. -/
theorem devAfter_copy {S : TState} {w w' : World} (hS : S.holds w) (hc : Device.CopyStep w.dev w'.dev) :
    TState.holds (S.filter Fact.isMeta ++ (if S.contains (.readyAll hostParty) then [.readyAll hostParty] else []) ++
      (if readyIn S defaultParty then [.readyAll defaultParty] else [])) w' := by
  rcases hc with he | ⟨rs, ws, r1, hop, hr, hs, hev⟩
  · refine TState.holds_app (TState.holds_app (fun x hx => ?_) (TState.holds_if fun h => TState.holds_one ?_))
      (TState.holds_if fun h => TState.holds_one ?_)
    · obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
      exact Fact.holds_meta hk (hS x hm) (by rw [he]) (by rw [he])
    · show Device.KnowsAll _ _; rw [he]; exact hS _ (List.contains_iff_mem.mp h)
    · show Device.KnowsAll _ _; rw [he]; exact readyIn_sound hS h
  · refine TState.holds_app (TState.holds_app (fun x hx => ?_) (TState.holds_if fun h => TState.holds_one ?_))
      (TState.holds_if fun h => TState.holds_one ?_)
    · obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
      exact Fact.holds_meta hk (hS x hm) hs hev
    · show Device.KnowsAll _ _
      rw [hr]
      exact (((hS _ (List.contains_iff_mem.mp h) : Device.KnowsAll _ _).of_host defaultParty).op_self hop).sync
    · show Device.KnowsAll _ _
      rw [hr]
      exact ((readyIn_sound hS h).op_self hop).grow rfl rfl (DevSpec.sync_grows _ _)

/-- **The device facts a call leaves are true after it.** -/
theorem devAfter_sound {f : IR.Ffi} {bits : List UInt64} {S : TState} {w : World} {r : Option V}
    {w' : World} (hS : S.holds w) (h : callBits f bits w = some (r, w')) : (devAfter f bits S).holds w' := by
  have hlive : S.contains (.part .cuda true) = true → w.dev.live = true := fun hc => holds_of_mem hS hc
  have hcap : S.contains (.part .capturing false) = true → w.dev.capture = none := fun hc => by
    have h1 : (Fact.part Part.capturing false).holds w := holds_of_mem hS hc
    change w.dev.capture.isSome = false at h1
    cases hcc : w.dev.capture with
    | none => rfl
    | some _ => rw [hcc] at h1; cases h1
  have fresh : ∀ {d : Dev}, d.streams = #[] → d.events = #[] → d.race = {} →
      ∀ {w'' : World}, w''.dev = d → TState.holds [.streams [], .events 0, .readyAll hostParty] w'' :=
    fun hs he hr _ hd => TState.holds_cons (by show _ = _; rw [hd, hs]; try rfl)
      (TState.holds_cons (by show _ = _; rw [hd, he]; try rfl)
        (TState.holds_one (by show Device.KnowsAll _ _; rw [hd, hr]; exact Device.KnowsAll.empty _)))
  unfold devAfter
  split
  · obtain ⟨h1, h2, h3⟩ := Device.init_dev (show ffiCudaInit _ w = _ from h)
    exact fresh h1 h2 h3 rfl
  · obtain ⟨h1, h2, h3⟩ := Device.cleanup_dev (show ffiCudaCleanup _ w = _ from h)
    exact fresh h1 h2 h3 rfl
  · -- a stream created
    rename_i c
    split
    · rename_i l hl
      split
      · rename_i hc
        simp only [Bool.and_eq_true, beq_iff_eq] at hc
        obtain ⟨rfl, hc⟩ := hc
        obtain ⟨-, hs, he, hr⟩ := Device.streamCreate_dev (show ffiCudaStreamCreate _ w = _ from h) (hlive hc)
        have hl' := streamsOf_holds hS hl
        have hacc : w'.dev.race.lastW = w.dev.race.lastW ∧ w'.dev.race.reads = w.dev.race.reads := by
          rw [hr]; exact ⟨rfl, rfl⟩
        have hg : DevSpec.Grows w.dev.race w'.dev.race := by rw [hr]; exact DevSpec.joinInto_grows _ _ _
        have hev : ∀ e : Nat, w'.dev.event? e = w.dev.event? e := fun e => by simp only [Dev.event?, he]
        refine TState.holds_cons (by show _ = _; rw [hs, Array.toList_push, hl']) ?_
        refine TState.holds_app (fun x hx => ?_) (TState.holds_if fun hrd => TState.holds_one ?_)
        · obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
          simp only [Bool.and_eq_true, Bool.not_eq_true'] at hk
          refine Fact.holds_step hk.1 (hS x hm) (fun ⟨l, e⟩ => by subst e; simp at hk)
            (fun ⟨n, e⟩ => by rw [he]) (fun e _ => hev e) hacc.1 hacc.2 hg
        · show Device.KnowsAll _ _
          rw [hr, ← hl', Array.length_toList]
          exact (readyIn_sound hS hrd).learn _ (DevSpec.ClockLe.join_left _ _)
      · exact fun _ e => nomatch e
    · exact fun _ e => nomatch e
  · -- an event created
    rename_i c
    split
    · rename_i n hn
      split
      · rename_i hc
        simp only [Bool.and_eq_true, beq_iff_eq] at hc
        obtain ⟨rfl, hc⟩ := hc
        obtain ⟨-, hs, he, hr⟩ := Device.eventCreate_dev (show ffiCudaEventCreate _ w = _ from h) (hlive hc)
        have hn' := eventsOf_holds hS hn
        refine TState.holds_cons (by show _ = _; rw [he, Array.size_push, hn'])
          (TState.holds_cons ?_ (fun x hx => ?_))
        · show (Dev.event? _ _).isSome = true
          rw [event?_nat, he, ← hn', Array.getElem?_push_size]; rfl
        · obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
          simp only [Bool.and_eq_true, Bool.not_eq_true'] at hk
          have hx0 := hS x hm
          refine Fact.holds_step hk.1 hx0 (fun _ => hs) (fun ⟨n, e⟩ => by subst e; simp at hk)
            (fun e hxe => ?_) (by rw [hr]) (by rw [hr]) (by rw [hr]; exact DevSpec.Grows.refl _)
          have hlt : e < w.dev.events.size := by
            rcases hxe with rfl | rfl
            · exact event?_lt hx0
            · obtain ⟨clk, hc, -⟩ := hx0; exact event?_lt (by rw [hc]; rfl)
          rw [event?_nat, event?_nat, he, Array.getElem?_push_lt hlt, Array.getElem?_eq_getElem hlt]
      · exact fun _ e => nomatch e
    · exact fun _ e => nomatch e
  · -- an event recorded
    rename_i c eid sid
    split
    · rename_i l e hl he
      split
      · rename_i p hp
        split
        · rename_i hc
          simp only [Bool.and_eq_true, beq_iff_eq] at hc
          obtain ⟨⟨⟨rfl, hc1⟩, hc2⟩, hc3⟩ := hc
          have hl' := streamsOf_holds hS hl
          have hp' : w.dev.party? (asI32 sid) = some p := by rw [partyOf_eq hl']; exact hp
          have hei := evId_spec he
          have hlv : (w.dev.event? e).isSome = true := holds_of_mem hS hc3
          rcases Device.eventRecord_dev (show ffiCudaEventRecord _ w = _ from h) (hlive hc1) (hcap hc2) hp'
            with hnone | ⟨hs, hev, hr⟩
          · rw [← hei] at hnone; rw [hnone] at hlv; cases hlv
          · have hlt := event?_lt hlv
            have hidx : (asI32 eid).toNat = e := by omega
            rw [hidx] at hev
            have hset : ∀ e' : Nat, w'.dev.event? e' =
                if e' = e then some (some ((w.dev.race.issue p).2, false)) else w.dev.event? e' := fun e' => by
              rw [event?_nat, event?_nat, hev]
              split
              · next he' => subst he'; simp [Array.set!, Array.getElem?_setIfInBounds_self_of_lt hlt]
              · next he' => simp [Array.set!, Array.getElem?_setIfInBounds_ne (Ne.symm he')]
            refine TState.holds_cons ?_ (TState.holds_app (fun x hx => ?_) (TState.holds_if fun hrd => TState.holds_one ?_))
            · show (Dev.event? _ _).isSome = true
              rw [hset, if_pos rfl]; rfl
            · obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
              simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hk
              obtain ⟨⟨hk1, hk2⟩, hk3⟩ := hk
              refine Fact.holds_step hk1 (hS x hm) (fun _ => hs) (fun _ => by rw [hev]; simp [Array.set!])
                (fun e' hxe => ?_) (by rw [hr]; rfl) (by rw [hr]; rfl)
                (by rw [hr]; exact DevSpec.setClock_issued_grows _ _)
              have hne : e' ≠ e := by rintro rfl; rcases hxe with rfl | rfl <;> simp_all
              rw [hset, if_neg hne]
            · refine ⟨(w.dev.race.issue p).2, by rw [hset, if_pos rfl], ?_⟩
              rw [hr]; exact Device.Covers.issue (readyIn_sound hS hrd)
        · exact fun _ e => nomatch e
      · exact fun _ e => nomatch e
    · exact fun _ e => nomatch e
  · -- a stream waits on an event
    rename_i c sid eid
    split
    · rename_i l e hl he
      split
      · rename_i p hp
        split
        · rename_i hc
          simp only [Bool.and_eq_true, beq_iff_eq] at hc
          obtain ⟨⟨⟨rfl, hc1⟩, hc2⟩, hc3⟩ := hc
          have hl' := streamsOf_holds hS hl
          have hp' : w.dev.party? (asI32 sid) = some p := by rw [partyOf_eq hl']; exact hp
          have hei := evId_spec he
          obtain ⟨clk, hclk, hcov⟩ := (holds_of_mem hS hc3 : Fact.holds (.evAll e) w)
          rw [hei] at hclk
          rcases Device.waitEvent_dev (show ffiCudaStreamWaitEvent _ w = _ from h) (hlive hc1) (hcap hc2) hp'
            with (hn | hn) | ⟨clk', hc', hs, hev, hr⟩
          · rw [hclk] at hn; cases hn
          · rw [hclk] at hn; cases hn
          · rw [hclk] at hc'
            simp only [Option.some.injEq, Prod.mk.injEq, and_true] at hc'
            subst hc'
            refine TState.holds_cons ?_ (fun x hx => ?_)
            · show Device.KnowsAll _ _; rw [hr]; exact hcov.wait p
            · obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
              refine Fact.holds_step hk (hS x hm) (fun _ => hs) (fun _ => by rw [hev])
                (fun e' _ => by rw [event?_nat, event?_nat, hev]) (by rw [hr]; rfl) (by rw [hr]; rfl)
                (by rw [hr]; exact (DevSpec.setClock_issued_grows _ _).trans (DevSpec.joinInto_grows _ _ _))
        · exact fun _ e => nomatch e
      · exact fun _ e => nomatch e
    · exact fun _ e => nomatch e
  · -- the host waits for a stream
    rename_i c sid
    split
    · rename_i l hl
      split
      · rename_i p hp
        split
        · rename_i hc
          simp only [Bool.and_eq_true, beq_iff_eq] at hc
          obtain ⟨⟨rfl, hc1⟩, hc2⟩ := hc
          have hl' := streamsOf_holds hS hl
          have hp' : w.dev.party? (asI32 sid) = some p := by rw [partyOf_eq hl']; exact hp
          obtain ⟨hs, hev, hr⟩ := Device.streamSync_dev (show ffiCudaStreamSync _ w = _ from h) (hlive hc1) (hcap hc2) hp'
          refine TState.holds_app (fun x hx => ?_) (TState.holds_if fun hrd => TState.holds_one ?_)
          · obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
            refine Fact.holds_step hk (hS x hm) (fun _ => hs) (fun _ => by rw [hev])
              (fun e' _ => by rw [event?_nat, event?_nat, hev]) (by rw [hr]; rfl) (by rw [hr]; rfl)
              (by rw [hr]; exact DevSpec.sync_grows _ _)
          · show Device.KnowsAll _ _; rw [hr]; exact (readyIn_sound hS hrd).sync
        · exact fun _ e => nomatch e
      · exact fun _ e => nomatch e
    · exact fun _ e => nomatch e
  · split
    · rename_i hc; exact devAfter_launch hS (Device.launch_dev (show ffiCudaLaunch _ w = _ from h) (hcap hc))
    · exact fun _ e => nomatch e
  · split
    · rename_i hc
      exact devAfter_launch hS (Device.launchNamed_dev (show ffiCudaLaunchNamed _ w = _ from h) (hcap hc))
    · exact fun _ e => nomatch e
  · -- a launch on a stream
    rename_i sid
    split
    · rename_i l hl
      split
      · rename_i p hp
        split
        · rename_i hc
          have hp' : w.dev.party? (asI32 sid) = some p := by rw [partyOf_eq (streamsOf_holds hS hl)]; exact hp
          exact devAfter_launch hS (Device.launchOnStream_dev (show ffiCudaLaunchOnStream _ w = _ from h) (hcap hc) hp')
        · exact fun _ e => nomatch e
      · exact fun _ e => nomatch e
    · exact fun _ e => nomatch e
  · exact devAfter_copy hS (Device.upload_dev (show ffiCudaUpload _ w = _ from h))
  · exact devAfter_copy hS (Device.uploadOffset_dev (show ffiCudaUploadOffset _ w = _ from h))
  · exact devAfter_copy hS (Device.download_dev (show ffiCudaDownload _ w = _ from h))
  · exact devAfter_copy hS (Device.downloadOffset_dev (show ffiCudaDownloadOffset _ w = _ from h))
  · obtain ⟨hs, hev, hl, hr, hg⟩ := Device.createBuffer_dev (show ffiCudaCreateBuffer _ w = _ from h)
    intro x hx
    obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
    exact Fact.holds_step hk (hS x hm) (fun _ => hs) (fun _ => by rw [hev])
      (fun e' _ => by rw [event?_nat, event?_nat, hev]) hl hr hg
  · exact fun _ e => nomatch e

/-- **What a call leaves of a typestate**: what `afterLife` says, and what
    `devAfter` says of the device's streams and events. -/
def after (f : IR.Ffi) (bits : List UInt64) (S : TState) : TState := afterLife f bits S ++ devAfter f bits S

/-- The device calls whose device facts after them read none of their
    arguments. -/
def devArgFree : IR.Ffi → Bool
  | .cudaInit | .cudaCleanup | .cudaLaunch | .cudaLaunchNamed | .cudaUpload | .cudaUploadOffset
  | .cudaDownload | .cudaDownloadOffset | .cudaCreateBuffer => true
  | _ => false

theorem devAfter_free {f : IR.Ffi} (h : devArgFree f = true) (bits bits' : List UInt64) (S : TState) :
    devAfter f bits S = devAfter f bits' S := by
  cases f <;> simp only [devArgFree, Bool.false_eq_true] at h <;> simp only [devAfter]


/-- **What a call leaves of a typestate is true after it.** -/
theorem after_sound {f : IR.Ffi} {bits : List UInt64} {S : TState} {w : World} {r : Option V}
    {w' : World} (hS : S.holds w) (h : callBits f bits w = some (r, w')) : (after f bits S).holds w' :=
  TState.holds_app (afterLife_sound hS h) (devAfter_sound hS h)

theorem reachable_of {m : Mem} {r : Region} (hp : r ≠ .pinned) (hz : m.frozen = false) (off n : Nat) :
    m.reachable r off n = true := by
  cases r <;> first | exact absurd rfl hp | (unfold Mem.reachable; rw [hz]; rfl)

theorem store_isSome {m : Mem} {a : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned) (hs : off + n ≤ m.sizes r)
    (hz : m.frozen = false) (v : UInt64) : (m.store a n v).isSome = true := by
  unfold Mem.store
  simp only [hd, Option.bind_eq_bind, Option.bind_some]
  rw [if_neg]
  · rfl
  · simp only [reachable_of hp hz, Bool.not_true, Bool.or_false, decide_eq_true_eq]
    unfold Mem.sizes at hs; omega

theorem load_isSome {m : Mem} {a : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned) (hs : off + n ≤ m.sizes r)
    (hz : m.frozen = false) : (m.load a n).isSome = true := by
  rw [load_of hd hs (by rw [readable_off_pinned hp, hz]; rfl)]; rfl

theorem copyOut_isSome {m : Mem} {a : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned) (hs : off + n ≤ m.sizes r)
    (hz : m.frozen = false) (hn : off + n ≤ 2 ^ 36) : (copyOut m a n).isSome = true := by
  unfold copyOut
  suffices ∀ (l : List Nat) (acc : ByteArray), (∀ i ∈ l, i < n) →
      (l.foldlM (fun (acc : ByteArray) i => do
        let b ← m.load (a + UInt64.ofNat i) 1
        pure (acc.push b.toUInt8)) acc).isSome = true from
    this _ _ (fun i hi => List.mem_range.mp hi)
  intro l
  induction l with
  | nil => intro _ _; rfl
  | cons i l ih =>
      intro acc hl
      have hi := hl i List.mem_cons_self
      have hd' := Sem.decodeAddr_add hd (i := i) (by rw [Sem.regionSpan_toNat]; omega)
      obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp (load_isSome (n := 1) hd' hp (by omega) hz)
      simp only [List.foldlM_cons, hb]
      exact ih _ (fun j hj => hl j (List.mem_cons_of_mem _ hj))

theorem copyIn_isSome {m : Mem} {a : UInt64} {r : Region} {off : Nat} (src : ByteArray)
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned) (hs : off + src.size ≤ m.sizes r)
    (hz : m.frozen = false) (hn : off + src.size ≤ 2 ^ 36) : (copyIn m a src).isSome = true := by
  unfold copyIn
  suffices ∀ (l : List Nat) (mm : Mem), (∀ i ∈ l, i < src.size) → mm.frozen = false →
      mm.sizes r = m.sizes r →
      (l.foldlM (fun (mm : Mem) i => mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64) mm).isSome = true from
    this _ _ (fun i hi => List.mem_range.mp hi) hz rfl
  intro l
  induction l with
  | nil => intro _ _ _ _; rfl
  | cons i l ih =>
      intro mm hl hz' hsz
      have hi := hl i List.mem_cons_self
      have hd' := Sem.decodeAddr_add hd (i := i) (by rw [Sem.regionSpan_toNat]; omega)
      obtain ⟨m1, hm1⟩ := Option.isSome_iff_exists.mp
        (store_isSome (n := 1) hd' hp (by rw [hsz]; omega) hz' (src.get! i).toUInt64)
      simp only [List.foldlM_cons, hm1, Option.bind_eq_bind, Option.bind_some]
      exact ih m1 (fun j hj => hl j (List.mem_cons_of_mem _ hj)) ((store_frozen hm1).trans hz')
        ((congrFun (store_sizes_eq hm1) r).trans hsz)

/-- **The CUDA context is live**, where the typestate says so. -/
theorem ctxOk_of_state {S : TState} {w : World} (hS : S.holds w)
    (h : S.contains (.part .cuda true) = true) : DevSpec.CtxOk w cudaCtx :=
  .inr ⟨rfl, holds_of_mem hS h⟩

/-- **Nothing is being captured**, where the typestate says so. -/
theorem capture_of_state {S : TState} {w : World} (hS : S.holds w)
    (h : S.contains (.part .capturing false) = true) : w.dev.capture = none := by
  have := holds_of_mem hS h
  simpa [Fact.holds, Part.get, Option.isSome_eq_false_iff, Option.isNone_iff_eq_none] using this

/-- **The vendor keeps its promises**, where the typestate names its oracles. -/
theorem vendor_of_state {S : TState} {w : World} (hS : S.holds w)
    (h : S.contains .oracles = true) : VendorKeeps w.vendor :=
  (holds_of_mem hS h).2

/-- Whether a typestate gives `n` bytes at `a`: they lie in one region other
    than the pinned one, which it knows is large enough, and memory is not
    frozen. -/
def roomAt (S : TState) (a : UInt64) (n : Nat) : Bool :=
  match decodeAddr a with
  | some (r, off) => decide (r ≠ .pinned) && decide (off + n ≤ 2 ^ 36) &&
      S.contains (.part .frozen false) &&
      S.any fun | .room r' k => decide (r' = r) && decide (off + n ≤ k) | _ => false
  | none => false

/-- `n` bytes at `a` lie inside one region outside pinned memory, within the
    span an address names, with memory not frozen: a load or a store of them
    finds them, and so does one of any part of them. -/
def Fits (m : Mem) (a : UInt64) (n : Nat) : Prop :=
  ∃ r off, decodeAddr a = some (r, off) ∧ r ≠ .pinned ∧ off + n ≤ m.sizes r ∧ m.frozen = false ∧
    off + n ≤ 2 ^ 36

theorem Fits.load {m : Mem} {a : UInt64} {n : Nat} (h : Fits m a n) : ∃ x, m.load a n = some x := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := h
  exact Option.isSome_iff_exists.mp (load_isSome hd hp hs hz)

theorem Fits.store {m : Mem} {a : UInt64} {n : Nat} (h : Fits m a n) (v : UInt64) :
    ∃ m', m.store a n v = some m' := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := h
  exact Option.isSome_iff_exists.mp (store_isSome hd hp hs hz v)

theorem Fits.mono {m : Mem} {a : UInt64} {n n' : Nat} (hn : n ≤ n') (h : Fits m a n') : Fits m a n := by
  obtain ⟨r, off, hd, hp, hs, hz, hsp⟩ := h
  exact ⟨r, off, hd, hp, by omega, hz, by omega⟩

/-- Part of bytes that fit fits. -/
theorem Fits.part {m : Mem} {a : UInt64} {n i k : Nat} (h : Fits m a n) (hik : i + k ≤ n) (hk : 0 < k) :
    Fits m (a + UInt64.ofNat i) k := by
  obtain ⟨r, off, hd, hp, hs, hz, hsp⟩ := h
  have hd' := Sem.decodeAddr_add (i := i) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  exact ⟨r, off + i, hd', hp, by omega, hz, by omega⟩

/-- A fit asks only a region's size and whether memory is frozen. -/
theorem Fits.congr {m m' : Mem} {a : UInt64} {n : Nat} (hs : m'.sizes = m.sizes) (hz : m'.frozen = m.frozen)
    (h : Fits m a n) : Fits m' a n := by
  obtain ⟨r, off, hd, hp, hsz, hf, hsp⟩ := h
  exact ⟨r, off, hd, hp, by rw [hs]; exact hsz, by rw [hz]; exact hf, hsp⟩

theorem roomAt_spec {S : TState} {w : World} {a : UInt64} {n : Nat} (hS : S.holds w)
    (h : roomAt S a n = true) : Fits w.mem a n := by
  unfold roomAt at h
  split at h
  · rename_i r off hd
    simp only [Bool.and_eq_true, decide_eq_true_eq, List.any_eq_true] at h
    obtain ⟨⟨⟨hp, hn⟩, hz⟩, x, hx, hr⟩ := h
    have hz' : w.mem.frozen = false := holds_of_mem hS hz
    refine ⟨r, off, hd, hp, ?_, hz', hn⟩
    cases x with
    | room r' k =>
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hr
        obtain ⟨rfl, hk⟩ := hr
        exact Nat.le_trans hk (hS _ hx)
    | part _ _ => cases hr
    | cell _ _ => cases hr
    | held _ _ | opened _ _ | htAt _ _ _ | cstr _ | devSeq | oracles | cstrIn _ _ | roomArg _ _ | devBuf _ _
    | pinnedUsed _ | streams _ | events _ | evLive _ | readyAll _ | evAll _ => cases hr
  · cases h

theorem slot8_of_state {S : TState} {w : World} {a : UInt64} (hS : S.holds w) (h : roomAt S a 8 = true) :
    Slot8 w.mem a := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := roomAt_spec hS h
  exact store_isSome hd hp hs hz

theorem store_of_state {S : TState} {w : World} {a : UInt64} {n : Nat} (hS : S.holds w)
    (h : roomAt S a n = true) (v : UInt64) : (w.mem.store a n v).isSome = true := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := roomAt_spec hS h
  exact store_isSome hd hp hs hz v

theorem load_of_state {S : TState} {w : World} {a : UInt64} {n : Nat} (hS : S.holds w)
    (h : roomAt S a n = true) : (w.mem.load a n).isSome = true := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := roomAt_spec hS h
  exact load_isSome hd hp hs hz

theorem load8_of_state {S : TState} {w : World} {a : UInt64} (hS : S.holds w) (h : roomAt S a 8 = true) :
    Load8 w.mem a := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := roomAt_spec hS h
  exact load_isSome hd hp hs hz

theorem load8_of_cell {S : TState} {w : World} {a v : UInt64} (hS : S.holds w)
    (h : S.contains (.cell a v) = true) : Load8 w.mem a := by
  unfold Load8; rw [TState.holds_cell hS h]; rfl

theorem Fits.readable {m : Mem} {a : UInt64} {n : Nat} (h : Fits m a n) : Readable m a n := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := h
  exact copyOut_isSome hd hp hs hz hn

theorem Fits.writable {m : Mem} {a : UInt64} {n : Nat} (h : Fits m a n) : Writable m a n := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := h
  intro src hsrc
  subst hsrc
  exact copyIn_isSome src hd hp hs hz hn

theorem readable_of_state {S : TState} {w : World} {a : UInt64} {n : Nat} (hS : S.holds w)
    (h : roomAt S a n = true) : Readable w.mem a n :=
  (roomAt_spec hS h).readable

theorem writable_of_state {S : TState} {w : World} {a : UInt64} {n : Nat} (hS : S.holds w)
    (h : roomAt S a n = true) : Writable w.mem a n :=
  (roomAt_spec hS h).writable

theorem roomAt_mono {S : TState} {a : UInt64} {n k : Nat} (h : roomAt S a n = true) (hk : k ≤ n) :
    roomAt S a k = true := by
  unfold roomAt at h ⊢
  split at h
  · simp only [Bool.and_eq_true, decide_eq_true_eq, List.any_eq_true] at h ⊢
    obtain ⟨⟨⟨hp, hn⟩, hz⟩, x, hx, hr⟩ := h
    refine ⟨⟨⟨hp, by omega⟩, hz⟩, x, hx, ?_⟩
    cases x <;> simp only [Bool.and_eq_true, decide_eq_true_eq] at hr ⊢
    all_goals first | exact hr | exact ⟨hr.1, by omega⟩
  · cases h

/-- **A poll's events fit** where the typestate gives room for as many records
    as the program asks for: fewer may arrive, never more. -/
theorem pollRoom_of_state {S : TState} {w : World} {p m : UInt64} (hS : S.holds w)
    (h : roomAt S p (32 * (asI32 m).toNat) = true) :
    Writable w.pump.mem p
      (winEventBytes (w.pump.win.pending.take (min w.pump.win.pending.length (asI32 m).toNat))).size := by
  rw [(pump_life w).1, winEventBytes_size]
  exact writable_of_state hS (roomAt_mono h (by simp only [List.length_take]; omega))

/-- The most room the typestate names past `a`, in `a`'s region. -/
def roomLeft (S : TState) (a : UInt64) : Nat :=
  match decodeAddr a with
  | some (r, off) =>
      (S.foldl (fun m x => match x with | .room r' k => if r' = r then max m k else m | _ => m) 0) - off
  | none => 0

/-- Room for `K` bytes is room for any fewer. -/
theorem readable_upto_of_state {S : TState} {w : World} {a : UInt64} {n K : Nat} (hS : S.holds w)
    (h : roomAt S a K = true) (hn : n ≤ K) : Readable w.mem a n :=
  readable_of_state hS (roomAt_mono h hn)

/-- Room for any number of bytes up to `n`. -/
theorem writable_upto_imp_of_state {S : TState} {w : World} {r : UInt64} {n : Nat} {P : Prop} (hS : S.holds w)
    (h : roomAt S r n = true) : P → ∀ k, k ≤ n → Writable w.mem r k :=
  fun _ k hk => writable_of_state hS (roomAt_mono h hk)

theorem cstr_of_state {S : TState} {w : World} {a : UInt64} (hS : S.holds w) (h : S.contains (.cstr a) = true) :
    CStr w.mem a :=
  holds_of_mem hS h

theorem cstr_imp_of_state {S : TState} {w : World} {a : UInt64} {P : Prop} (hS : S.holds w)
    (h : S.contains (.cstr a) = true) : P → CStr w.mem a :=
  fun _ => holds_of_mem hS h

/-- **A read into memory has room**: room for as many bytes as it asks for. -/
theorem readTo_of_state {S : TState} {w : World} {dst fileOff size : UInt64} {p : String → Prop}
    (hS : S.holds w) (h : roomAt S dst size.toNat = true) :
    ∀ path c, p path → w.fs.get path = some c → Writable w.mem dst (readCountPtr c fileOff size) :=
  fun _ _ _ _ => writable_of_state hS (roomAt_mono h (Nat.min_le_left _ _))

-- ---------------------------------------------------------------------------
-- What a program's own store leaves
-- ---------------------------------------------------------------------------

/-- Whether a store of `n` bytes at `a` keeps a fact. -/
def storeKeeps (a : UInt64) (n : Nat) : Fact → Bool
  | .part _ _ | .room _ _ | .held _ _ | .opened _ _ | .devSeq | .oracles | .roomArg _ _
  | .devBuf _ _ | .pinnedUsed _ | .streams _ | .events _ | .evLive _ | .readyAll _ | .evAll _ => true
  | .cell c _ => cellOk c && (List.range 8).all fun i => !within a n (c + UInt64.ofNat i)
  | .cstr _ => false
  | .cstrIn c k => decide (a.toNat + n ≤ c.toNat ∨ (c.toNat + k ≤ a.toNat ∧ a.toNat + n ≤ 2 ^ 64))
  | .htAt c _ _ => decide (a.toNat < (regionBase .pinned).toNat) && cellOk c &&
      (List.range 8).all fun i => !within a n (c + UInt64.ofNat i)

def Fact.isCell : Fact → Bool
  | .cell _ _ | .cstr _ | .cstrIn _ _ | .htAt _ _ _ => true
  | _ => false

/-- **What a store leaves of a typestate**: every part and room; the cells it
    does not overlap, when its address is known; and the cell it writes, when
    it writes a known eight-byte value to a cell. -/
def afterStore (a v : Option UInt64) (n : Nat) (S : TState) : TState :=
  match a with
  | none => S.filter (!·.isCell)
  | some a =>
      let k := S.filter (storeKeeps a n)
      match v with
      | some v => if n == 8 && cellOk a then .cell a v :: k else k
      | none => k

/-- A step that keeps every part, every region's size and every handle
    held keeps every fact but the cells. -/
theorem nonCell_kept {S : TState} {w w' : World} (hS : S.holds w) (hp : ∀ p : Part, p.get w' = p.get w)
    (hh : ∀ (k : Held) (a : UInt64), k.live w' a = k.live w a)
    (hdv : w'.dev = w.dev ∧ w'.kernel = w.kernel ∧ w'.vendor = w.vendor)
    (hsz : w'.mem.sizes = w.mem.sizes) : TState.holds (S.filter (!·.isCell)) w' := by
  intro x hx
  obtain ⟨hm, hc⟩ := List.mem_filter.mp hx
  have := hS x hm
  cases x with
  | part p b => exact (hp p).trans this
  | room r n => show n ≤ w'.mem.sizes r; rw [hsz]; exact this
  | cell _ _ => cases hc
  | held k a => show k.live w' a = true; rw [hh k a]; exact this
  | opened k a => show a ≠ 0 → k.live w' a = true; rw [hh k a]; exact this
  | cstr _ => cases hc
  | htAt _ _ _ => cases hc
  | devSeq => show Device.TrackOk w'.dev ∧ _; rw [hdv.1]; exact this
  | oracles => show Sem.KeepsSize w'.kernel ∧ _; rw [hdv.2.1, hdv.2.2]; exact this
  | cstrIn _ _ => cases hc
  | roomArg r x => show x.toNat ≤ w'.mem.sizes r ∧ _; rw [hsz]; exact this
  | devBuf _ _ => show Device.BufIs _ _ w'.dev; rw [hdv.1]; exact this
  | pinnedUsed n => show w'.mem.sizes .pinned ≤ n; rw [hsz]; exact this
  | streams _ | events _ | evLive _ | readyAll _ | evAll _ => exact Fact.holds_dev rfl hdv.1 this

/-- A store that cannot reach a cell's eight bytes leaves what the cell
    holds. -/
theorem store_cell_kept {w w' : World} {a b c v : UInt64} {n : Nat} (hz : w'.mem.frozen = w.mem.frozen)
    (hsz : w'.mem.sizes = w.mem.sizes) (h : w.mem.store a n b = some w'.mem) (hn : n ≤ 2 ^ 64)
    (hc : cellOk c = true) (hk : ∀ i, i < 8 → within a n (c + UInt64.ofNat i) = false)
    (hx : w.mem.load c 8 = some v) : w'.mem.load c 8 = some v := by
  obtain ⟨rg, off, hd, hpin, hb⟩ := cellOk_spec hc
  have hz' : w'.mem.frozen = false := by rw [hz]; exact frozen_of_load hd hpin hx
  refine load_kept hd hpin hb hx hz' (congrFun hsz rg) fun i hi => Unchanged.of_eq ?_
  refine Mem.store_other h _ fun j hj e => ?_
  have := within_of hn ⟨j, hj, e⟩
  rw [hk i hi] at this; cases this

theorem storeKeeps_sound {S : TState} {w w' : World} {a b : UInt64} {n : Nat} (hS : S.holds w)
    (hp : ∀ p : Part, p.get w' = p.get w) (hh : ∀ (k : Held) (a : UInt64), k.live w' a = k.live w a)
    (hhp : w'.heap = w.heap) (hdv : w'.dev = w.dev ∧ w'.kernel = w.kernel ∧ w'.vendor = w.vendor)
    (hsz : w'.mem.sizes = w.mem.sizes)
    (h : w.mem.store a n b = some w'.mem) (hn : n ≤ 2 ^ 64) : TState.holds (S.filter (storeKeeps a n)) w' := by
  intro x hx
  obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
  have hx0 := hS x hm
  cases x with
  | part p b => exact (hp p).trans hx0
  | room r k => show k ≤ w'.mem.sizes r; rw [hsz]; exact hx0
  | held k a => show k.live w' a = true; rw [hh k a]; exact hx0
  | opened k a => show a ≠ 0 → k.live w' a = true; rw [hh k a]; exact hx0
  | cstr _ => simp [storeKeeps] at hk
  | devSeq => show Device.TrackOk w'.dev ∧ _; rw [hdv.1]; exact hx0
  | roomArg r x => show x.toNat ≤ w'.mem.sizes r ∧ _; rw [hsz]; exact hx0
  | devBuf _ _ => show Device.BufIs _ _ w'.dev; rw [hdv.1]; exact hx0
  | pinnedUsed n => show w'.mem.sizes .pinned ≤ n; rw [hsz]; exact hx0
  | streams _ | events _ | evLive _ | readyAll _ | evAll _ => exact Fact.holds_dev rfl hdv.1 hx0
  | oracles => show Sem.KeepsSize w'.kernel ∧ _; rw [hdv.2.1, hdv.2.2]; exact hx0
  | cstrIn c k =>
      simp only [storeKeeps, decide_eq_true_eq] at hk
      have hz' : w'.mem.frozen = false := by
        have := hp .frozen; simp only [Part.get] at this; rw [this]; exact hx0.1
      have hw := hx0.2.2.1
      refine cstrIn_kept hx0 hz' (fun rg _ => congrFun hsz rg) fun i hi => Unchanged.of_eq ?_
      refine Mem.store_other h _ fun j hj e => ?_
      have e2 := congrArg UInt64.toNat e
      rw [UInt64.toNat_add, UInt64.toNat_add, UInt64.toNat_ofNat', UInt64.toNat_ofNat',
        Nat.mod_eq_of_lt (by omega : i < 2 ^ 64), Nat.mod_eq_of_lt (by omega : j < 2 ^ 64),
        Nat.mod_eq_of_lt (by omega : c.toNat + i < 2 ^ 64)] at e2
      rcases hk with h1 | ⟨h1, h2⟩
      · rw [Nat.mod_eq_of_lt (by omega : a.toNat + j < 2 ^ 64)] at e2; omega
      · rw [Nat.mod_eq_of_lt (by omega : a.toNat + j < 2 ^ 64)] at e2; omega
  | cell c v =>
      simp only [storeKeeps, Bool.and_eq_true, List.all_eq_true, List.mem_range, Bool.not_eq_true'] at hk
      exact store_cell_kept (hp .frozen) hsz h hn hk.1 hk.2 hx0
  | htAt c m m' =>
      simp only [storeKeeps, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range,
        Bool.not_eq_true'] at hk
      obtain ⟨⟨ha, hc⟩, hmiss⟩ := hk
      obtain ⟨r, off, hd, -⟩ := Sem.store_frame h
      have hr : r ≠ .pinned := by
        rintro rfl
        obtain ⟨rfl, ho⟩ := Sem.decodeAddr_some hd
        rw [Sem.regionSpan_toNat] at ho
        have hb : (regionBase .pinned).toNat = 0x4000000000 := rfl
        rw [UInt64.toNat_add, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega : off < 2 ^ 64), hb,
          Nat.mod_eq_of_lt (by omega)] at ha
        omega
      have hl : w'.mem.load c 8 = w.mem.load c 8 := by
        obtain ⟨s, -, -, -, -, hl, -⟩ : HtSpec.HtObj w c m m' := hx0
        rw [hl]; exact store_cell_kept (hp .frozen) hsz h hn hc hmiss hl
      exact HtSpec.HtObj.stored hx0 h hhp hd hr hl

/-- A store of `n` bytes at `a` leaves a cell eight bytes it cannot reach,
    stated on the addresses as numbers: what a store at an address known only
    by its bounds keeps. -/
theorem storeKeeps_cell_of {a c v : UInt64} {n : Nat} (hc : cellOk c = true)
    (hd : a.toNat + n ≤ c.toNat ∨ c.toNat + 8 ≤ a.toNat) (hw : a.toNat + n ≤ 2 ^ 64)
    (hcw : c.toNat + 8 ≤ 2 ^ 64) : storeKeeps a n (.cell c v) = true := by
  simp only [storeKeeps, hc, Bool.true_and, List.all_eq_true, List.mem_range, Bool.not_eq_true']
  intro i hi
  simp only [within, decide_eq_false_iff_not, Nat.not_lt]
  have hci : (c + UInt64.ofNat i).toNat = c.toNat + i := by
    rw [UInt64.toNat_add, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega : i < 2 ^ 64),
      Nat.mod_eq_of_lt (by omega)]
  rw [UInt64.toNat_sub, hci]
  have ha := a.toNat_lt
  rcases hd with h | h
  · rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]; omega
  · rw [Nat.mod_eq_of_lt (by omega)]; omega

theorem storeKeeps_cstrIn_of {a c : UInt64} {n k : Nat}
    (hd : a.toNat + n ≤ c.toNat ∨ (c.toNat + k ≤ a.toNat ∧ a.toNat + n ≤ 2 ^ 64)) :
    storeKeeps a n (.cstrIn c k) = true := by
  simp only [storeKeeps, decide_eq_true_eq]; exact hd

/-- What a store of `m` bytes keeps, a narrower one keeps. -/
theorem storeKeeps_mono {a : UInt64} {m n : Nat} {x : Fact} (h : storeKeeps a m x = true) (hn : n ≤ m) :
    storeKeeps a n x = true := by
  cases x with
  | cell c v =>
      simp only [storeKeeps, Bool.and_eq_true, List.all_eq_true, List.mem_range, Bool.not_eq_true',
        within, decide_eq_false_iff_not, Nat.not_lt] at h ⊢
      exact ⟨h.1, fun i hi => Nat.le_trans hn (h.2 i hi)⟩
  | cstrIn c k =>
      simp only [storeKeeps, decide_eq_true_eq] at h ⊢
      omega
  | cstr _ => simp [storeKeeps] at h
  | htAt c k k' =>
      simp only [storeKeeps, Bool.and_eq_true, List.all_eq_true, List.mem_range, Bool.not_eq_true',
        within, decide_eq_false_iff_not, Nat.not_lt] at h ⊢
      exact ⟨h.1, fun i hi => Nat.le_trans hn (h.2 i hi)⟩
  | _ => rfl

theorem afterStore_sound {S : TState} {w w' : World} {ao vo : Option UInt64} {a b : UInt64} {n : Nat}
    (hS : S.holds w) (hp : ∀ p : Part, p.get w' = p.get w) (hh : ∀ (k : Held) (a : UInt64), k.live w' a = k.live w a)
    (hhp : w'.heap = w.heap)
    (hdv : w'.dev = w.dev ∧ w'.kernel = w.kernel ∧ w'.vendor = w.vendor)
    (hsz : w'.mem.sizes = w.mem.sizes)
    (h : ao.isSome → w.mem.store a n b = some w'.mem) (ha : ∀ x, ao = some x → x = a)
    (hv : ∀ x, vo = some x → x = b) (hn : n ≤ 2 ^ 64) : (afterStore ao vo n S).holds w' := by
  cases ao with
  | none => exact nonCell_kept hS hp hh hdv hsz
  | some a' =>
      obtain rfl := ha a' rfl
      have hst := h rfl
      have hk := storeKeeps_sound hS hp hh hhp hdv hsz hst hn
      cases vo with
      | none => exact hk
      | some v' =>
          obtain rfl := hv v' rfl
          show TState.holds (if (n == 8 && cellOk a') = true then _ else _) w'
          split
          · rename_i hc
            simp only [Bool.and_eq_true, beq_iff_eq] at hc
            obtain ⟨rfl, hok⟩ := hc
            exact TState.holds_cons (cell_of_store hok hst) hk
          · exact hk

theorem foldl_stores_sizes {α : Type} (f : α → Mem → Option Mem) (hf : ∀ x m m', f x m = some m' →
    m'.sizes = m.sizes ∧ m'.frozen = m.frozen ∧ m'.busy = m.busy) :
    ∀ (l : List α) (m m' : Mem), l.foldlM (fun mm x => f x mm) m = some m' →
      m'.sizes = m.sizes ∧ m'.frozen = m.frozen ∧ m'.busy = m.busy
  | [], m, m', h => by cases h; exact ⟨rfl, rfl, rfl⟩
  | x :: l, m, m', h => by
      simp only [List.foldlM_cons, Option.bind_eq_bind] at h
      obtain ⟨m1, h1, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨e1, f1, b1⟩ := hf x m m1 h1
      obtain ⟨e2, f2, b2⟩ := foldl_stores_sizes f hf l m1 m' h
      exact ⟨e2.trans e1, f2.trans f1, b2.trans b1⟩

set_option maxHeartbeats 2000000 in
/-- Every entry point is found by its own C name. -/
theorem ofCname_cname (f : IR.Ffi) : IR.Ffi.ofCname f.cname = some f := by
  cases f <;> decide

/-- **A call is its contract.** Outside a worker's lifetime, a call to any entry
    point but `threadSpawn` is `callBits` on its argument bits. -/
theorem callOf_ffi {f : IR.Ffi} (hf : f.engine = true) (lc : Locals) (vs : List V) {w : World}
    (hz : w.mem.frozen = false) :
    callOf lc (.ffi f) vs w = (vs.mapM asBits).bind (fun bits => callBits f bits w) := by
  have hf' : (f == .threadSpawn) = false ∧ f.inLib = false := by
    simpa [IR.Ffi.engine, bne, Bool.and_eq_true] using hf
  have hs := hf'.1
  have hl := hf'.2
  simp only [callOf, hz, hs, hl, Bool.false_and, Bool.false_eq_true, if_false, callImport, ofCname_cname,
    bind, Option.bind, callFfi]

-- ---------------------------------------------------------------------------
-- The device, from the typestate
-- ---------------------------------------------------------------------------

/-- **Every buffer is ready for `p`** where the typestate says `p` knows every
    access. -/
theorem ready_of_state {S : TState} {w : World} {p : Nat} (hS : S.holds w) (h : readyIn S p = true)
    (b : Nat) (write : Bool) : Ready w.dev.race p b write :=
  (readyIn_sound hS h).ready b write

/-- Every access the default stream's own, it knows them all. -/
theorem readyIn_seq {S : TState} (h : S.contains .devSeq = true) : readyIn S defaultParty = true := by
  simp only [readyIn, h, beq_self_eq_true, Bool.and_self, Bool.or_true]

theorem keepsSize_of_state {S : TState} {w : World} (hS : S.holds w) (h : S.contains .oracles = true) :
    Sem.KeepsSize w.kernel := (holds_of_mem (x := .oracles) hS h).1

theorem vendorKeeps_of_state {S : TState} {w : World} (hS : S.holds w) (h : S.contains .oracles = true) :
    VendorKeeps w.vendor := (holds_of_mem (x := .oracles) hS h).2

-- ---------------------------------------------------------------------------
-- Half of a cell
-- ---------------------------------------------------------------------------

theorem Mem.readable_sub {m : Mem} {r : Region} {off n off' n' : Nat} (h : m.readable r off n = true)
    (h1 : off ≤ off') (h2 : off' + n' ≤ off + n) : m.readable r off' n' = true := by
  unfold Mem.readable at h ⊢
  simp only [Bool.and_eq_true, Bool.or_eq_true] at h ⊢
  obtain ⟨hz, hp⟩ := h
  refine ⟨hz, ?_⟩
  rcases hp with hp | ⟨hl, hb⟩
  · exact .inl hp
  · refine .inr ⟨?_, ?_⟩
    · obtain ⟨x, hx, hx'⟩ := List.any_eq_true.mp hl
      refine List.any_eq_true.mpr ⟨x, hx, ?_⟩
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hx' ⊢
      omega
    · refine List.all_eq_true.mpr fun b hb' => ?_
      have := List.all_eq_true.mp hb b hb'
      simp only [Busy.clear, Bool.or_eq_true, Bool.not_eq_true', decide_eq_true_eq] at this ⊢
      rcases this with hw | hc | hc
      · exact .inl hw
      · exact .inr (.inl (by omega))
      · exact .inr (.inr (by omega))

theorem byte_toNat_lt (x : UInt8) : x.toUInt64.toNat < 256 := by
  simp only [UInt8.toNat_toUInt64]; exact x.toNat_lt

theorem load_step_toNat (x : UInt64) (b : UInt8) :
    ((x <<< 8) ||| b.toUInt64).toNat = 256 * (x.toNat % 2 ^ 56) + b.toUInt64.toNat := by
  have hb := byte_toNat_lt b
  rw [UInt64.toNat_or, UInt64.toNat_shiftLeft]
  simp only [UInt64.reduceToNat, Nat.reduceMod, Nat.shiftLeft_eq]
  have : x.toNat * 2 ^ 8 % 2 ^ 64 = 2 ^ 8 * (x.toNat % 2 ^ 56) := by omega
  rw [this, ← Nat.two_pow_add_eq_or_of_lt (by simpa using hb)]

theorem Mem.load_lo4 {m : Mem} {a v : UInt64} (h : m.load a 8 = some v) : m.load a 4 = some (v &&& 0xFFFFFFFF) := by
  unfold Mem.load at h ⊢
  cases hd : decodeAddr a with
  | none => simp [hd] at h
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [hd] at h ⊢
    dsimp only [bind, Option.bind] at h ⊢
    by_cases hc : (decide (off + 8 > (m.region r).size) || !m.readable r off 8) = true
    · rw [if_pos hc] at h; cases h
    rw [if_neg hc] at h
    simp only [Option.some.injEq] at h
    simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true', not_or, Bool.not_eq_false] at hc
    rw [if_neg (by
      intro hc'
      simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at hc'
      rcases hc' with h1 | h1
      · omega
      · rw [Mem.readable_sub hc.2 (Nat.le_refl _) (by omega)] at h1; cases h1)]
    subst h
    congr 1
    apply UInt64.toNat_inj.mp
    simp only [show List.range 8 = [0, 1, 2, 3, 4, 5, 6, 7] from rfl, show List.range 4 = [0, 1, 2, 3] from rfl,
      List.foldr_cons, List.foldr_nil, load_step_toNat, UInt64.toNat_and, UInt64.reduceToNat]
    rw [show (4294967295 : Nat) = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]
    have := byte_toNat_lt ((m.region r).get! (off + 0)); have := byte_toNat_lt ((m.region r).get! (off + 1))
    have := byte_toNat_lt ((m.region r).get! (off + 2)); have := byte_toNat_lt ((m.region r).get! (off + 3))
    have := byte_toNat_lt ((m.region r).get! (off + 4)); have := byte_toNat_lt ((m.region r).get! (off + 5))
    have := byte_toNat_lt ((m.region r).get! (off + 6)); have := byte_toNat_lt ((m.region r).get! (off + 7))
    simp only [UInt64.toNat_zero, Nat.zero_mod, Nat.mul_zero, Nat.zero_add] at *
    omega

theorem Mem.load_hi4 {m : Mem} {a v : UInt64} {r : Region} {off : Nat} (h : m.load a 8 = some v)
    (hd : decodeAddr a = some (r, off)) (hi : off + 4 < regionSpan.toNat) :
    m.load (a + 4) 4 = some (v >>> 32) := by
  have hd4 : decodeAddr (a + 4) = some (r, off + 4) := Sem.decodeAddr_add (i := 4) hd hi
  unfold Mem.load at h ⊢
  rw [hd] at h; rw [hd4]
  dsimp only [bind, Option.bind] at h ⊢
  by_cases hc : (decide (off + 8 > (m.region r).size) || !m.readable r off 8) = true
  · rw [if_pos hc] at h; cases h
  rw [if_neg hc] at h
  simp only [Option.some.injEq] at h
  simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true', not_or, Bool.not_eq_false] at hc
  rw [if_neg (by
    intro hc'
    simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at hc'
    rcases hc' with h1 | h1
    · omega
    · rw [Mem.readable_sub hc.2 (by omega) (by omega)] at h1; cases h1)]
  subst h
  congr 1
  apply UInt64.toNat_inj.mp
  simp only [show List.range 8 = [0, 1, 2, 3, 4, 5, 6, 7] from rfl, show List.range 4 = [0, 1, 2, 3] from rfl,
    List.foldr_cons, List.foldr_nil, load_step_toNat, UInt64.toNat_shiftRight, UInt64.reduceToNat, Nat.reduceMod,
    Nat.shiftRight_eq_div_pow, Nat.add_assoc, Nat.reduceAdd]
  have := byte_toNat_lt ((m.region r).get! (off + 0)); have := byte_toNat_lt ((m.region r).get! (off + 1))
  have := byte_toNat_lt ((m.region r).get! (off + 2)); have := byte_toNat_lt ((m.region r).get! (off + 3))
  have := byte_toNat_lt ((m.region r).get! (off + 4)); have := byte_toNat_lt ((m.region r).get! (off + 5))
  have := byte_toNat_lt ((m.region r).get! (off + 6)); have := byte_toNat_lt ((m.region r).get! (off + 7))
  simp only [UInt64.toNat_zero, Nat.zero_mod, Nat.mul_zero, Nat.zero_add, Nat.add_zero] at *
  omega

/-- The four bytes at a cell, or four past it: the low or the high half of its
    value. -/
theorem TState.holds_cell_half {S : TState} {w : World} {a v : UInt64} (hS : S.holds w)
    (h : S.contains (.cell a v) = true) :
    w.mem.load a 4 = some (v &&& 0xFFFFFFFF) := Mem.load_lo4 (holds_of_mem hS h)

theorem TState.holds_cell_hi {S : TState} {w : World} {a v : UInt64} {r : Region} {off : Nat}
    (hS : S.holds w) (h : S.contains (.cell a v) = true) (hd : decodeAddr a = some (r, off))
    (hi : off + 4 < regionSpan.toNat) : w.mem.load (a + 4) 4 = some (v >>> 32) :=
  Mem.load_hi4 (holds_of_mem hS h) hd hi

theorem opReady_of_state {S : TState} {w : World} (hS : S.holds w) (h : S.contains .devSeq = true)
    (rs ws : List Nat) : OpReady w.dev.race defaultParty rs ws :=
  ⟨fun b _ => (ready_of_state hS (readyIn_seq h) b true).1,
    fun b _ => (ready_of_state hS (readyIn_seq h) b true).2 rfl⟩

theorem devBuf_size {S : TState} {w : World} {a n : UInt64} (hS : S.holds w) (hm : Fact.devBuf a n ∈ S)
    {b : ByteArray} (hb : w.dev.get? (asI32 a) = some b) : b.size = n.toNat :=
  (hS _ hm).2.2 b hb

/-- **A matrix-vector product fits buffers whose sizes the typestate knows**,
    and finds them ready on the default stream. -/
theorem sgemvOk_of_state {S : TState} {w : World} {trans m n a x y na nx ny : UInt64}
    (hS : S.holds w) (hd : S.contains .devSeq = true)
    (ha : Fact.devBuf a na ∈ S) (hx : Fact.devBuf x nx ∈ S) (hy : Fact.devBuf y ny ∈ S)
    (hm : 0 < asI32 m) (hn : 0 < asI32 n)
    (hA : 4 * ((asI32 m).toNat * (asI32 n).toNat) ≤ na.toNat)
    (hX : 4 * (if asI32 trans != 0 then (asI32 m).toNat else (asI32 n).toNat) ≤ nx.toNat)
    (hY : 4 * (if asI32 trans != 0 then (asI32 n).toNat else (asI32 m).toNat) ≤ ny.toNat) :
    SgemvOk w defaultParty trans m n a x y := by
  intro A X Y hA' hX' hY'
  refine ⟨hm, hn, ?_, opReady_of_state hS hd _ _⟩
  simp only [sgemvFits, Bool.and_eq_true, decide_eq_true_eq]
  rw [devBuf_size hS ha hA', devBuf_size hS hx hX', devBuf_size hS hy hY']
  exact ⟨⟨hA, hX⟩, hY⟩

/-- **A strided product fits buffers whose sizes the typestate knows**, and
    finds them ready on the default stream. -/
theorem gemmOk_of_state {S : TState} {w : World} {fits : ByteArray → ByteArray → ByteArray → Bool}
    {a b c na nb nc : UInt64} (hS : S.holds w) (hd : S.contains .devSeq = true)
    (ha : Fact.devBuf a na ∈ S) (hb : Fact.devBuf b nb ∈ S) (hc : Fact.devBuf c nc ∈ S)
    (hf : ∀ A B C : ByteArray, A.size = na.toNat → B.size = nb.toNat → C.size = nc.toNat → fits A B C = true) :
    GemmOk fits w defaultParty a b c := fun A B C hA hB hC =>
  ⟨hf A B C (devBuf_size hS ha hA) (devBuf_size hS hb hB) (devBuf_size hS hc hC), opReady_of_state hS hd _ _⟩

/-- A value below `2 ^ 63` reads back as itself. -/
theorem asI64_low {x : UInt64} (h : x.toNat < 2 ^ 63) : asI64 x = x.toNat := by
  have hlt : ¬ (0x8000000000000000 : UInt64) ≤ x &&& 0xffffffffffffffff := by
    rw [UInt64.le_iff_toNat_le, UInt64.toNat_and]
    simp only [UInt64.reduceToNat]
    rw [show (18446744073709551615 : Nat) = 2 ^ 64 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
      Nat.mod_eq_of_lt x.toNat_lt]; omega
  have hu : (x &&& 0xffffffffffffffff).toNat = x.toNat := by
    rw [UInt64.toNat_and]; simp only [UInt64.reduceToNat]
    rw [show (18446744073709551615 : Nat) = 2 ^ 64 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
      Nat.mod_eq_of_lt x.toNat_lt]
  simp only [asI64, signed, widthMask, ClifTy.width, ge_iff_le, Nat.le_refl, if_true, hlt, if_false, hu]

/-- A 64-bit value written out as its structure reads as its number. -/
theorem toNat_mk_lit (n : Nat) (h : n < 2 ^ 64) :
    ({ toBitVec := { toFin := ⟨n, h⟩ } } : UInt64).toNat = n := rfl

/-- A value below `2 ^ 31`, cut to 32 bits, reads back as itself. -/
theorem asI32_low {x : UInt64} (h : x.toNat < 2 ^ 31) : asI32 (x &&& widthMask .i32) = x.toNat := by
  have hand : x.toNat &&& 4294967295 = x.toNat := by
    rw [show (4294967295 : Nat) = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]; omega
  have hu : x &&& 4294967295 &&& 4294967295 = x := by
    apply UInt64.toNat_inj.mp; simp only [UInt64.toNat_and, UInt64.reduceToNat, hand]
  have hlt : ¬ (1 <<< UInt64.ofNat 31 : UInt64) ≤ x := by
    rw [show (1 <<< UInt64.ofNat 31 : UInt64) = 2147483648 from by decide, UInt64.le_iff_toNat_le]
    simp only [UInt64.reduceToNat]; omega
  simp only [asI32, signed, widthMask, ClifTy.width, hu]
  simp only [ge_iff_le, Nat.reduceLeDiff, if_false, Nat.reduceSub, hlt]

/-- A value below `2 ^ 31` reads back as itself as an `i32`. -/
theorem asI32_small {x : UInt64} (h : x.toNat < 2 ^ 31) : asI32 x = x.toNat := by
  have hm : x &&& widthMask .i32 = x := by
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_and]; simp only [widthMask, UInt64.reduceToNat]
    rw [show (4294967295 : Nat) = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]; omega
  have := asI32_low h
  rwa [hm] at this

/-- A product of two values below `2 ^ 31`, in bytes of four, does not wrap. -/
theorem mul_shl2 {m n : UInt64} (hm : m.toNat < 2 ^ 31) (hn : n.toNat < 2 ^ 31) :
    ((m * n) <<< 2).toNat = 4 * (m.toNat * n.toNat) := by
  have : m.toNat * n.toNat < 2 ^ 62 :=
    Nat.lt_of_lt_of_le (Nat.mul_lt_mul'' hm hn) (by decide)
  simp only [UInt64.toNat_shiftLeft, UInt64.toNat_mul, UInt64.reduceToNat, Nat.reduceMod, Nat.shiftLeft_eq]
  rw [Nat.mod_eq_of_lt (by omega : m.toNat * n.toNat < 2 ^ 64)]
  omega

theorem shl2_low {m : UInt64} (hm : m.toNat < 2 ^ 31) : (m <<< 2).toNat = 4 * m.toNat := by
  simp only [UInt64.toNat_shiftLeft, UInt64.reduceToNat, Nat.reduceMod, Nat.shiftLeft_eq]
  omega

theorem mapM_isSome_of {α β : Type} {f : α → Option β} :
    ∀ {l : List α}, (∀ x ∈ l, (f x).isSome = true) → (l.mapM f).isSome = true
  | [], _ => rfl
  | x :: l, h => by
      obtain ⟨y, hy⟩ := Option.isSome_iff_exists.mp (h x List.mem_cons_self)
      obtain ⟨ys, hys⟩ := Option.isSome_iff_exists.mp (mapM_isSome_of (l := l) fun z hz => h z (List.mem_cons_of_mem _ hz))
      simp [List.mapM_cons, hy, hys]

/-- Bindings in room can be read. -/
theorem readIds_of_state {S : TState} {w : World} {p : UInt64} {n : Nat} (hS : S.holds w)
    (h : roomAt S p (4 * n) = true) : (readIds w.mem p n).isSome = true := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := roomAt_spec hS h
  have hl : ((List.range n).mapM fun i => w.mem.load (p + UInt64.ofNat (4 * i)) 4).isSome = true := by
    refine mapM_isSome_of fun i hi => ?_
    have hi := List.mem_range.mp hi
    have hd' := Sem.decodeAddr_add (i := 4 * i) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
    exact load_isSome hd' hp (by omega) hz
  obtain ⟨raw, hraw⟩ := Option.isSome_iff_exists.mp hl
  unfold readIds
  rw [hraw]; rfl

/-- A pipeline's bindings can be read where the typestate gives their bytes
    room: each is two four-byte words. -/
theorem readBinds_of_state {S : TState} {w : World} {p : UInt64} {n : Nat} (hS : S.holds w)
    (h : roomAt S p (8 * n) = true) : (readBinds w.mem p n).isSome = true := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := roomAt_spec hS h
  unfold readBinds
  refine mapM_isSome_of fun i hi => ?_
  have hi := List.mem_range.mp hi
  have hd1 := Sem.decodeAddr_add (i := 8 * i) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  have hd2 := Sem.decodeAddr_add (i := 8 * i + 4) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp (load_isSome (n := 4) hd1 hp (by omega) hz)
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp (load_isSome (n := 4) hd2 hp (by omega) hz)
  rw [ha, hb]; rfl

/-- **A launch's bindings are ready** where they can be read and every access
    is the default stream's. -/
theorem bindsReady_of_state {S : TState} {w : World} {nBufs bindPtr : UInt64} (hS : S.holds w)
    (hq : S.contains .devSeq = true) (h : roomAt S bindPtr (4 * (asI32 nBufs).toNat) = true) :
    BindsReady w defaultParty nBufs bindPtr := by
  obtain ⟨ids, hids⟩ := Option.isSome_iff_exists.mp (readIds_of_state hS h)
  exact ⟨ids, hids, fun id _ _ => ready_of_state hS (readyIn_seq hq) _ _⟩

/-- A C string the typestate names within some bytes is there. -/
theorem cstrIn_of_state {S : TState} {w : World} {a : UInt64} (hS : S.holds w)
    (h : (S.any fun | .cstrIn a' _ => a' == a | _ => false) = true) : CStr w.mem a := by
  obtain ⟨x, hx, hr⟩ := List.any_eq_true.mp h
  cases x
  all_goals first | cases hr | skip
  rename_i a' n
  simp only [beq_iff_eq] at hr; subst hr
  exact (hS _ hx).2.1

/-- **Memory a call reads or writes, by a length the program was handed**: its
    region holds that many bytes, and the access lies within them. -/
theorem copyOut_of_roomArg {S : TState} {w : World} {a x : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hS : S.holds w) (hm : Fact.roomArg r x ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned) (hb : off + n ≤ x.toNat) :
    (copyOut w.mem a n).isSome = true := by
  have hx := hS _ hm
  exact copyOut_isSome hd hp (by have := hx.1; omega) (holds_of_mem hS hz) (by have := hx.2; omega)

theorem readable_of_roomArg {S : TState} {w : World} {a x : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hS : S.holds w) (hm : Fact.roomArg r x ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned) (hb : off + n ≤ x.toNat) :
    Readable w.mem a n :=
  copyOut_of_roomArg hS hm hz hd hp hb

theorem writable_of_roomArg {S : TState} {w : World} {a x : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hS : S.holds w) (hm : Fact.roomArg r x ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned) (hb : off + n ≤ x.toNat) :
    Writable w.mem a n := by
  have hx := hS _ hm
  intro src hsrc
  subst hsrc
  exact copyIn_isSome src hd hp (by have := hx.1; omega) (holds_of_mem hS hz) (by have := hx.2; omega)

/-- The same, at an address a value past a known one. -/
theorem readable_of_roomArg_add {S : TState} {w : World} {b x X : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hS : S.holds w) (hm : Fact.roomArg r X ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hd : decodeAddr b = some (r, off)) (hp : r ≠ .pinned) (hb : off + x.toNat + n ≤ X.toNat) :
    Readable w.mem (b + x) n := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · show (copyOut w.mem (b + x) 0).isSome = true
    rfl
  · have hx := hS _ hm
    have hd' : decodeAddr (b + x) = some (r, off + x.toNat) := by
      have := Sem.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; have := hx.2; omega)
      rwa [UInt64.ofNat_toNat] at this
    exact readable_of_roomArg hS hm hz hd' hp (by omega)

theorem writable_of_roomArg_add {S : TState} {w : World} {b x X : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hS : S.holds w) (hm : Fact.roomArg r X ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hd : decodeAddr b = some (r, off)) (hp : r ≠ .pinned) (hb : off + x.toNat + n ≤ X.toNat) :
    Writable w.mem (b + x) n := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · intro src hsrc
    show (copyIn w.mem (b + x) src).isSome = true
    unfold copyIn; rw [hsrc]; rfl
  · have hx := hS _ hm
    have hd' : decodeAddr (b + x) = some (r, off + x.toNat) := by
      have := Sem.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; have := hx.2; omega)
      rwa [UInt64.ofNat_toNat] at this
    exact writable_of_roomArg hS hm hz hd' hp (by omega)

end AlgorithmLib.HProg.Contracts

