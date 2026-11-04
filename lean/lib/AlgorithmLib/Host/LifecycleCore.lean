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


open Lean Elab Command in
/-- `keeps_<entry>` for every entry point whose name starts with `pfx` and has
    none. -/
elab "derive_keeps_only " pfx:str : command => do
  let some (.inductInfo iv) := (← getEnv).find? ``IR.Ffi | throwError "derive_keeps_only: no Ffi"
  let ns ← getCurrNamespace
  for c in iv.ctors do
    let name := c.componentsRev.head!.toString
    unless name.startsWith pfx.getString do continue
    let short := Name.mkSimple s!"keeps_{name}"
    let thm := mkIdent short
    let ctor := mkIdent c
    unless (← getEnv).contains (ns ++ short) do
      elabCommand (← `(command|
        set_option maxHeartbeats 4000000 in
        theorem $thm {bits : List UInt64} {w : World} {r : Option V} {w' : World}
            (h : callBits $ctor bits w = some (r, w')) : Keeps $ctor r w w' := by
          keeps_entry h))

end AlgorithmLib.HProg.Contracts
