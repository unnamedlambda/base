module
public import AlgorithmLib.Host.Contracts
meta import AlgorithmLib.Host.Contracts
public import AlgorithmLib.Host.DevTrack
meta import AlgorithmLib.Host.DevTrack
public import AlgorithmLib.Host.DevSeq
meta import AlgorithmLib.Host.DevSeq
public import AlgorithmLib.Host.DevBufs
meta import AlgorithmLib.Host.DevBufs
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Lifecycle` — the contract table

`callBits` is the one reading of the Rust entry points. Everything a frontend
may rely on about an entry point is gathered here into one value per entry,
`Ffi.contract`, and proven of that reading once, `contract_sound`:

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
  opening on a display, is left unknown;
* `blind` --- for the entries that cannot tell data apart, what they read to
  decide what to do and what stays known after them; `Host.Static` checks
  programs by it, and `Contract.Sound.blind` is the congruence it rests on.

The same facts give the memory preconditions: a room gives the slots, reads
and writes inside it (`roomAt`, `slot8_of_state`, `readable_of_state`,
`writable_of_state`), and a cell gives the value a load reads. A program's own
store moves a typestate too (`afterStore`).

A typestate is therefore carried through every call by computing `after`, with
nothing proved about the call. `callOf_ffi` connects the table to the calls
programs make: outside a worker's lifetime every entry point but `threadSpawn`
is `callBits` on its argument bits.

The per-entry proofs follow each contract to every answer it gives (`walk`) and
close each part there (`part_close`).
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

/-- **The kinds of handle a library hands a program**, which a typestate
    may say it holds: an open window, serial port or USB device. -/
inductive Held where
  | window | port | usbDevice | heap
  deriving DecidableEq, Repr

/-- Whether `a` is a handle of kind `k` the program holds open; for the heap,
    the start of a live allocation. -/
def Held.live : Held → World → UInt64 → Bool
  | .window, w, a => (w.wlWin? a).isSome
  | .port, w, a => (w.serPort? a).isSome
  | .usbDevice, w, a => (w.usbDev? a).isSome
  | .heap, w, a => (w.heapAt? a).isSome

/-- **The parts each entry point may move.** Bringing a context up or down,
    beginning or ending a capture, and joining or dropping a worker; every other
    call keeps every part (`callBits_parts`). -/
def moves : IR.Ffi → List Part
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
    | (injection $h:ident with h1; injection h1 with h2 h3; subst h3)
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

set_option maxHeartbeats 1000000 in
theorem parts_fileRead {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileRead bits w = some (r, w')) : KeepsParts .fileRead w w' := by
  change ffiFileRead bits w = some (r, w') at h
  unfold ffiFileRead at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_fileWrite {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileWrite bits w = some (r, w')) : KeepsParts .fileWrite w w' := by
  change ffiFileWrite bits w = some (r, w') at h
  unfold ffiFileWrite at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_fileReadToPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileReadToPtr bits w = some (r, w')) : KeepsParts .fileReadToPtr w w' := by
  change ffiFileReadToPtr bits w = some (r, w') at h
  unfold ffiFileReadToPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_fileWriteFromPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileWriteFromPtr bits w = some (r, w')) : KeepsParts .fileWriteFromPtr w w' := by
  change ffiFileWriteFromPtr bits w = some (r, w') at h
  unfold ffiFileWriteFromPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_stdinReadline {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .stdinReadline bits w = some (r, w')) : KeepsParts .stdinReadline w w' := by
  change ffiStdinReadline bits w = some (r, w') at h
  unfold ffiStdinReadline at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_stdoutWrite {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .stdoutWrite bits w = some (r, w')) : KeepsParts .stdoutWrite w w' := by
  change ffiStdoutWrite bits w = some (r, w') at h
  unfold ffiStdoutWrite at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_sinf {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .sinf bits w = some (r, w')) : KeepsParts .sinf w w' := by
  change ffiSinf bits w = some (r, w') at h
  unfold ffiSinf at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cosf {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cosf bits w = some (r, w')) : KeepsParts .cosf w w' := by
  change ffiCosf bits w = some (r, w') at h
  unfold ffiCosf at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_powf {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .powf bits w = some (r, w')) : KeepsParts .powf w w' := by
  change ffiPowf bits w = some (r, w') at h
  unfold ffiPowf at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_htInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htInit bits w = some (r, w')) : KeepsParts .htInit w w' := by
  change ffiHtInit bits w = some (r, w') at h
  unfold ffiHtInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_htCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCleanup bits w = some (r, w')) : KeepsParts .htCleanup w w' := by
  change ffiHtCleanup bits w = some (r, w') at h
  unfold ffiHtCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_htCreate {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCreate bits w = some (r, w')) : KeepsParts .htCreate w w' := by
  change ffiHtCreate bits w = some (r, w') at h
  unfold ffiHtCreate at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_htCount {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCount bits w = some (r, w')) : KeepsParts .htCount w w' := by
  change ffiHtCount bits w = some (r, w') at h
  unfold ffiHtCount at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_htLookup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htLookup bits w = some (r, w')) : KeepsParts .htLookup w w' := by
  change ffiHtLookup bits w = some (r, w') at h
  unfold ffiHtLookup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_htInsert {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htInsert bits w = some (r, w')) : KeepsParts .htInsert w w' := by
  change ffiHtInsert bits w = some (r, w') at h
  unfold ffiHtInsert at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_htIncrement {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htIncrement bits w = some (r, w')) : KeepsParts .htIncrement w w' := by
  change ffiHtIncrement bits w = some (r, w') at h
  unfold ffiHtIncrement at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_htGetEntry {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htGetEntry bits w = some (r, w')) : KeepsParts .htGetEntry w w' := by
  change ffiHtGetEntry bits w = some (r, w') at h
  unfold ffiHtGetEntry at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaInit bits w = some (r, w')) : KeepsParts .cudaInit w w' := by
  change ffiCudaInit bits w = some (r, w') at h
  unfold ffiCudaInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaCleanup bits w = some (r, w')) : KeepsParts .cudaCleanup w w' := by
  change ffiCudaCleanup bits w = some (r, w') at h
  unfold ffiCudaCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaCreateBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaCreateBuffer bits w = some (r, w')) : KeepsParts .cudaCreateBuffer w w' := by
  change ffiCudaCreateBuffer bits w = some (r, w') at h
  unfold ffiCudaCreateBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaUpload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUpload bits w = some (r, w')) : KeepsParts .cudaUpload w w' := by
  change ffiCudaUpload bits w = some (r, w') at h
  unfold ffiCudaUpload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaUploadOffset {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUploadOffset bits w = some (r, w')) : KeepsParts .cudaUploadOffset w w' := by
  change ffiCudaUploadOffset bits w = some (r, w') at h
  unfold ffiCudaUploadOffset at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaDownload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownload bits w = some (r, w')) : KeepsParts .cudaDownload w w' := by
  change ffiCudaDownload bits w = some (r, w') at h
  unfold ffiCudaDownload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaDownloadOffset {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownloadOffset bits w = some (r, w')) : KeepsParts .cudaDownloadOffset w w' := by
  change ffiCudaDownloadOffset bits w = some (r, w') at h
  unfold ffiCudaDownloadOffset at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaFreeBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaFreeBuffer bits w = some (r, w')) : KeepsParts .cudaFreeBuffer w w' := by
  change ffiCudaFreeBuffer bits w = some (r, w') at h
  unfold ffiCudaFreeBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaSync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaSync bits w = some (r, w')) : KeepsParts .cudaSync w w' := by
  change ffiCudaSync bits w = some (r, w') at h
  unfold ffiCudaSync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaLaunch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunch bits w = some (r, w')) : KeepsParts .cudaLaunch w w' := by
  change ffiCudaLaunch bits w = some (r, w') at h
  unfold ffiCudaLaunch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cublasSgemv {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemv bits w = some (r, w')) : KeepsParts .cublasSgemv w w' := by
  change ffiCublasSgemv bits w = some (r, w') at h
  unfold ffiCublasSgemv at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cublasSgemm {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemm bits w = some (r, w')) : KeepsParts .cublasSgemm w w' := by
  change ffiCublasSgemm bits w = some (r, w') at h
  unfold ffiCublasSgemm at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cublasGemmExBf16 {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasGemmExBf16 bits w = some (r, w')) : KeepsParts .cublasGemmExBf16 w w' := by
  change ffiCublasGemmExBf16 bits w = some (r, w') at h
  unfold ffiCublasGemmExBf16 at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cublasSgemvOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemvOnStream bits w = some (r, w')) : KeepsParts .cublasSgemvOnStream w w' := by
  change ffiCublasSgemvOnStream bits w = some (r, w') at h
  unfold ffiCublasSgemvOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cublasGemmStridedBatchedExBf16 {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasGemmStridedBatchedExBf16 bits w = some (r, w')) : KeepsParts .cublasGemmStridedBatchedExBf16 w w' := by
  change ffiCublasGemmStridedBatchedExBf16 bits w = some (r, w') at h
  unfold ffiCublasGemmStridedBatchedExBf16 at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cublasPtrArray {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasPtrArray bits w = some (r, w')) : KeepsParts .cublasPtrArray w w' := by
  change ffiCublasPtrArray bits w = some (r, w') at h
  unfold ffiCublasPtrArray at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cublasSgemmBatchedOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemmBatchedOnStream bits w = some (r, w')) : KeepsParts .cublasSgemmBatchedOnStream w w' := by
  change ffiCublasSgemmBatchedOnStream bits w = some (r, w') at h
  unfold ffiCublasSgemmBatchedOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaLaunchNamedOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunchNamedOnStream bits w = some (r, w')) : KeepsParts .cudaLaunchNamedOnStream w w' := by
  change ffiCudaLaunchNamedOnStream bits w = some (r, w') at h
  unfold ffiCudaLaunchNamedOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaEventElapsedMsBits {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventElapsedMsBits bits w = some (r, w')) : KeepsParts .cudaEventElapsedMsBits w w' := by
  change ffiCudaEventElapsedMsBits bits w = some (r, w') at h
  unfold ffiCudaEventElapsedMsBits at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaUploadAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUploadAsync bits w = some (r, w')) : KeepsParts .cudaUploadAsync w w' := by
  change ffiCudaUploadAsync bits w = some (r, w') at h
  unfold ffiCudaUploadAsync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaUploadOffsetAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUploadOffsetAsync bits w = some (r, w')) : KeepsParts .cudaUploadOffsetAsync w w' := by
  change ffiCudaUploadOffsetAsync bits w = some (r, w') at h
  unfold ffiCudaUploadOffsetAsync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaDownloadAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownloadAsync bits w = some (r, w')) : KeepsParts .cudaDownloadAsync w w' := by
  change ffiCudaDownloadAsync bits w = some (r, w') at h
  unfold ffiCudaDownloadAsync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cublasSgemmOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemmOnStream bits w = some (r, w')) : KeepsParts .cublasSgemmOnStream w w' := by
  change ffiCublasSgemmOnStream bits w = some (r, w') at h
  unfold ffiCublasSgemmOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaLaunchOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunchOnStream bits w = some (r, w')) : KeepsParts .cudaLaunchOnStream w w' := by
  change ffiCudaLaunchOnStream bits w = some (r, w') at h
  unfold ffiCudaLaunchOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaStreamCreate {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamCreate bits w = some (r, w')) : KeepsParts .cudaStreamCreate w w' := by
  change ffiCudaStreamCreate bits w = some (r, w') at h
  unfold ffiCudaStreamCreate at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaStreamSync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamSync bits w = some (r, w')) : KeepsParts .cudaStreamSync w w' := by
  change ffiCudaStreamSync bits w = some (r, w') at h
  unfold ffiCudaStreamSync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaStreamDestroy {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamDestroy bits w = some (r, w')) : KeepsParts .cudaStreamDestroy w w' := by
  change ffiCudaStreamDestroy bits w = some (r, w') at h
  unfold ffiCudaStreamDestroy at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaEventCreate {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventCreate bits w = some (r, w')) : KeepsParts .cudaEventCreate w w' := by
  change ffiCudaEventCreate bits w = some (r, w') at h
  unfold ffiCudaEventCreate at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaEventRecord {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventRecord bits w = some (r, w')) : KeepsParts .cudaEventRecord w w' := by
  change ffiCudaEventRecord bits w = some (r, w') at h
  unfold ffiCudaEventRecord at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaStreamWaitEvent {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamWaitEvent bits w = some (r, w')) : KeepsParts .cudaStreamWaitEvent w w' := by
  change ffiCudaStreamWaitEvent bits w = some (r, w') at h
  unfold ffiCudaStreamWaitEvent at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaEventDestroy {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventDestroy bits w = some (r, w')) : KeepsParts .cudaEventDestroy w w' := by
  change ffiCudaEventDestroy bits w = some (r, w') at h
  unfold ffiCudaEventDestroy at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaGraphBeginCapture {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphBeginCapture bits w = some (r, w')) : KeepsParts .cudaGraphBeginCapture w w' := by
  change ffiCudaGraphBeginCapture bits w = some (r, w') at h
  unfold ffiCudaGraphBeginCapture at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaGraphEndCapture {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphEndCapture bits w = some (r, w')) : KeepsParts .cudaGraphEndCapture w w' := by
  change ffiCudaGraphEndCapture bits w = some (r, w') at h
  unfold ffiCudaGraphEndCapture at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaGraphUpload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphUpload bits w = some (r, w')) : KeepsParts .cudaGraphUpload w w' := by
  change ffiCudaGraphUpload bits w = some (r, w') at h
  unfold ffiCudaGraphUpload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaGraphLaunch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphLaunch bits w = some (r, w')) : KeepsParts .cudaGraphLaunch w w' := by
  change ffiCudaGraphLaunch bits w = some (r, w') at h
  unfold ffiCudaGraphLaunch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaGraphDestroy {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphDestroy bits w = some (r, w')) : KeepsParts .cudaGraphDestroy w w' := by
  change ffiCudaGraphDestroy bits w = some (r, w') at h
  unfold ffiCudaGraphDestroy at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaLaunchNamed {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunchNamed bits w = some (r, w')) : KeepsParts .cudaLaunchNamed w w' := by
  change ffiCudaLaunchNamed bits w = some (r, w') at h
  unfold ffiCudaLaunchNamed at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaPinnedAlloc {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedAlloc bits w = some (r, w')) : KeepsParts .cudaPinnedAlloc w w' := by
  change ffiCudaPinnedAlloc bits w = some (r, w') at h
  unfold ffiCudaPinnedAlloc at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaPinnedPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedPtr bits w = some (r, w')) : KeepsParts .cudaPinnedPtr w w' := by
  change ffiCudaPinnedPtr bits w = some (r, w') at h
  unfold ffiCudaPinnedPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaPinnedPtrAt {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedPtrAt bits w = some (r, w')) : KeepsParts .cudaPinnedPtrAt w w' := by
  change ffiCudaPinnedPtrAt bits w = some (r, w') at h
  unfold ffiCudaPinnedPtrAt at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaPinnedFree {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedFree bits w = some (r, w')) : KeepsParts .cudaPinnedFree w w' := by
  change ffiCudaPinnedFree bits w = some (r, w') at h
  unfold ffiCudaPinnedFree at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaMemInfoFree {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaMemInfoFree bits w = some (r, w')) : KeepsParts .cudaMemInfoFree w w' := by
  change ffiCudaMemInfoFree bits w = some (r, w') at h
  unfold ffiCudaMemInfoFree at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cudaMemInfoTotal {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaMemInfoTotal bits w = some (r, w')) : KeepsParts .cudaMemInfoTotal w w' := by
  change ffiCudaMemInfoTotal bits w = some (r, w') at h
  unfold ffiCudaMemInfoTotal at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_gpuInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuInit bits w = some (r, w')) : KeepsParts .gpuInit w w' := by
  change ffiGpuInit bits w = some (r, w') at h
  unfold ffiGpuInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_gpuCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCleanup bits w = some (r, w')) : KeepsParts .gpuCleanup w w' := by
  change ffiGpuCleanup bits w = some (r, w') at h
  unfold ffiGpuCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_gpuCreateBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCreateBuffer bits w = some (r, w')) : KeepsParts .gpuCreateBuffer w w' := by
  change ffiGpuCreateBuffer bits w = some (r, w') at h
  unfold ffiGpuCreateBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_gpuCreatePipeline {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCreatePipeline bits w = some (r, w')) : KeepsParts .gpuCreatePipeline w w' := by
  change ffiGpuCreatePipeline bits w = some (r, w') at h
  unfold ffiGpuCreatePipeline at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_gpuUpload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuUpload bits w = some (r, w')) : KeepsParts .gpuUpload w w' := by
  change ffiGpuUpload bits w = some (r, w') at h
  unfold ffiGpuUpload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_gpuUploadPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuUploadPtr bits w = some (r, w')) : KeepsParts .gpuUploadPtr w w' := by
  change ffiGpuUploadPtr bits w = some (r, w') at h
  unfold ffiGpuUploadPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_gpuDispatch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuDispatch bits w = some (r, w')) : KeepsParts .gpuDispatch w w' := by
  change ffiGpuDispatch bits w = some (r, w') at h
  unfold ffiGpuDispatch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_gpuDownload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuDownload bits w = some (r, w')) : KeepsParts .gpuDownload w w' := by
  change ffiGpuDownload bits w = some (r, w') at h
  unfold ffiGpuDownload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_gpuDownloadPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuDownloadPtr bits w = some (r, w')) : KeepsParts .gpuDownloadPtr w w' := by
  change ffiGpuDownloadPtr bits w = some (r, w') at h
  unfold ffiGpuDownloadPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_lmdbInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbInit bits w = some (r, w')) : KeepsParts .lmdbInit w w' := by
  change ffiLmdbInit bits w = some (r, w') at h
  unfold ffiLmdbInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_lmdbCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCleanup bits w = some (r, w')) : KeepsParts .lmdbCleanup w w' := by
  change ffiLmdbCleanup bits w = some (r, w') at h
  unfold ffiLmdbCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_lmdbOpen {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbOpen bits w = some (r, w')) : KeepsParts .lmdbOpen w w' := by
  change ffiLmdbOpen bits w = some (r, w') at h
  unfold ffiLmdbOpen at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_lmdbBeginWriteTxn {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbBeginWriteTxn bits w = some (r, w')) : KeepsParts .lmdbBeginWriteTxn w w' := by
  change ffiLmdbBeginWriteTxn bits w = some (r, w') at h
  unfold ffiLmdbBeginWriteTxn at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_lmdbPut {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbPut bits w = some (r, w')) : KeepsParts .lmdbPut w w' := by
  change ffiLmdbPut bits w = some (r, w') at h
  unfold ffiLmdbPut at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_lmdbCommitWriteTxn {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCommitWriteTxn bits w = some (r, w')) : KeepsParts .lmdbCommitWriteTxn w w' := by
  change ffiLmdbCommitWriteTxn bits w = some (r, w') at h
  unfold ffiLmdbCommitWriteTxn at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_lmdbCursorScan {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCursorScan bits w = some (r, w')) : KeepsParts .lmdbCursorScan w w' := by
  change ffiLmdbCursorScan bits w = some (r, w') at h
  unfold ffiLmdbCursorScan at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_windowInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowInit bits w = some (r, w')) : KeepsParts .windowInit w w' := by
  change ffiWindowInit bits w = some (r, w') at h
  unfold ffiWindowInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_windowCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowCleanup bits w = some (r, w')) : KeepsParts .windowCleanup w w' := by
  change ffiWindowCleanup bits w = some (r, w') at h
  unfold ffiWindowCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_windowOpen {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowOpen bits w = some (r, w')) : KeepsParts .windowOpen w w' := by
  change ffiWindowOpen bits w = some (r, w') at h
  unfold ffiWindowOpen at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_windowPoll {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowPoll bits w = some (r, w')) : KeepsParts .windowPoll w w' := by
  change ffiWindowPoll bits w = some (r, w') at h
  unfold ffiWindowPoll at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_windowPresentGpuBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowPresentGpuBuffer bits w = some (r, w')) : KeepsParts .windowPresentGpuBuffer w w' := by
  change ffiWindowPresentGpuBuffer bits w = some (r, w') at h
  unfold ffiWindowPresentGpuBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_nativeLoad {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .nativeLoad bits w = some (r, w')) : KeepsParts .nativeLoad w w' := by
  change ffiNativeLoad bits w = some (r, w') at h
  unfold ffiNativeLoad at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_threadInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadInit bits w = some (r, w')) : KeepsParts .threadInit w w' := by
  change ffiThreadInit bits w = some (r, w') at h
  unfold ffiThreadInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_threadCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadCleanup bits w = some (r, w')) : KeepsParts .threadCleanup w w' := by
  change ffiThreadCleanup bits w = some (r, w') at h
  unfold ffiThreadCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_threadJoin {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadJoin bits w = some (r, w')) : KeepsParts .threadJoin w w' := by
  change ffiThreadJoin bits w = some (r, w') at h
  unfold ffiThreadJoin at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_nativeFree {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .nativeFree bits w = some (r, w')) : KeepsParts .nativeFree w w' := by
  change ffiNativeFree bits w = some (r, w') at h
  unfold ffiNativeFree at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_nativeArch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .nativeArch bits w = some (r, w')) : KeepsParts .nativeArch w w' := by
  change ffiNativeArch bits w = some (r, w') at h
  unfold ffiNativeArch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_cpuHas {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cpuHas bits w = some (r, w')) : KeepsParts .cpuHas w w' := by
  change ffiCpuHas bits w = some (r, w') at h
  unfold ffiCpuHas at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_fileCreateDirAll {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileCreateDirAll bits w = some (r, w')) : KeepsParts .fileCreateDirAll w w' := by
  change ffiFileCreateDirAll bits w = some (r, w') at h
  unfold ffiFileCreateDirAll at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_memLock {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .memLock bits w = some (r, w')) : KeepsParts .memLock w w' := by
  change ffiGrant "lock" 2 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_memUnlock {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .memUnlock bits w = some (r, w')) : KeepsParts .memUnlock w w' := by
  change ffiGrant "unlock" 2 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_memAdviseHuge {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .memAdviseHuge bits w = some (r, w')) : KeepsParts .memAdviseHuge w w' := by
  change ffiGrant "hugepages" 2 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

set_option maxHeartbeats 1000000 in
theorem parts_threadPriority {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadPriority bits w = some (r, w')) : KeepsParts .threadPriority w w' := by
  change ffiGrant "priority" 1 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals part_close)
    | (walk h; all_goals part_close)

/-- **Every call keeps every part it does not move.** -/
theorem callBits_parts (f : IR.Ffi) {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) : KeepsParts f w w' := by
  cases f with
  | fileRead => exact parts_fileRead h
  | fileWrite => exact parts_fileWrite h
  | fileReadToPtr => exact parts_fileReadToPtr h
  | fileWriteFromPtr => exact parts_fileWriteFromPtr h
  | stdinReadline => exact parts_stdinReadline h
  | stdoutWrite => exact parts_stdoutWrite h
  | sinf => exact parts_sinf h
  | cosf => exact parts_cosf h
  | powf => exact parts_powf h
  | htInit => exact parts_htInit h
  | htCleanup => exact parts_htCleanup h
  | htCreate => exact parts_htCreate h
  | htCount => exact parts_htCount h
  | htLookup => exact parts_htLookup h
  | htInsert => exact parts_htInsert h
  | htIncrement => exact parts_htIncrement h
  | htGetEntry => exact parts_htGetEntry h
  | cudaInit => exact parts_cudaInit h
  | cudaCleanup => exact parts_cudaCleanup h
  | cudaCreateBuffer => exact parts_cudaCreateBuffer h
  | cudaUpload => exact parts_cudaUpload h
  | cudaUploadOffset => exact parts_cudaUploadOffset h
  | cudaDownload => exact parts_cudaDownload h
  | cudaDownloadOffset => exact parts_cudaDownloadOffset h
  | cudaFreeBuffer => exact parts_cudaFreeBuffer h
  | cudaSync => exact parts_cudaSync h
  | cudaLaunch => exact parts_cudaLaunch h
  | cublasSgemv => exact parts_cublasSgemv h
  | cublasSgemm => exact parts_cublasSgemm h
  | cublasGemmExBf16 => exact parts_cublasGemmExBf16 h
  | cublasSgemvOnStream => exact parts_cublasSgemvOnStream h
  | cublasGemmStridedBatchedExBf16 => exact parts_cublasGemmStridedBatchedExBf16 h
  | cublasPtrArray => exact parts_cublasPtrArray h
  | cublasSgemmBatchedOnStream => exact parts_cublasSgemmBatchedOnStream h
  | cudaLaunchNamedOnStream => exact parts_cudaLaunchNamedOnStream h
  | cudaEventElapsedMsBits => exact parts_cudaEventElapsedMsBits h
  | cudaUploadAsync => exact parts_cudaUploadAsync h
  | cudaUploadOffsetAsync => exact parts_cudaUploadOffsetAsync h
  | cudaDownloadAsync => exact parts_cudaDownloadAsync h
  | cublasSgemmOnStream => exact parts_cublasSgemmOnStream h
  | cudaLaunchOnStream => exact parts_cudaLaunchOnStream h
  | cudaStreamCreate => exact parts_cudaStreamCreate h
  | cudaStreamSync => exact parts_cudaStreamSync h
  | cudaStreamDestroy => exact parts_cudaStreamDestroy h
  | cudaEventCreate => exact parts_cudaEventCreate h
  | cudaEventRecord => exact parts_cudaEventRecord h
  | cudaStreamWaitEvent => exact parts_cudaStreamWaitEvent h
  | cudaEventDestroy => exact parts_cudaEventDestroy h
  | cudaGraphBeginCapture => exact parts_cudaGraphBeginCapture h
  | cudaGraphEndCapture => exact parts_cudaGraphEndCapture h
  | cudaGraphUpload => exact parts_cudaGraphUpload h
  | cudaGraphLaunch => exact parts_cudaGraphLaunch h
  | cudaGraphDestroy => exact parts_cudaGraphDestroy h
  | cudaLaunchNamed => exact parts_cudaLaunchNamed h
  | cudaPinnedAlloc => exact parts_cudaPinnedAlloc h
  | cudaPinnedPtr => exact parts_cudaPinnedPtr h
  | cudaPinnedPtrAt => exact parts_cudaPinnedPtrAt h
  | cudaPinnedFree => exact parts_cudaPinnedFree h
  | cudaMemInfoFree => exact parts_cudaMemInfoFree h
  | cudaMemInfoTotal => exact parts_cudaMemInfoTotal h
  | gpuInit => exact parts_gpuInit h
  | gpuCleanup => exact parts_gpuCleanup h
  | gpuCreateBuffer => exact parts_gpuCreateBuffer h
  | gpuCreatePipeline => exact parts_gpuCreatePipeline h
  | gpuUpload => exact parts_gpuUpload h
  | gpuUploadPtr => exact parts_gpuUploadPtr h
  | gpuDispatch => exact parts_gpuDispatch h
  | gpuDownload => exact parts_gpuDownload h
  | gpuDownloadPtr => exact parts_gpuDownloadPtr h
  | lmdbInit => exact parts_lmdbInit h
  | lmdbCleanup => exact parts_lmdbCleanup h
  | lmdbOpen => exact parts_lmdbOpen h
  | lmdbBeginWriteTxn => exact parts_lmdbBeginWriteTxn h
  | lmdbPut => exact parts_lmdbPut h
  | lmdbCommitWriteTxn => exact parts_lmdbCommitWriteTxn h
  | lmdbCursorScan => exact parts_lmdbCursorScan h
  | windowInit => exact parts_windowInit h
  | windowCleanup => exact parts_windowCleanup h
  | windowOpen => exact parts_windowOpen h
  | windowPoll => exact parts_windowPoll h
  | windowPresentGpuBuffer => exact parts_windowPresentGpuBuffer h
  | nativeLoad => exact parts_nativeLoad h
  | threadInit => exact parts_threadInit h
  | threadCleanup => exact parts_threadCleanup h
  | threadJoin => exact parts_threadJoin h
  | nativeFree => exact parts_nativeFree h
  | nativeArch => exact parts_nativeArch h
  | cpuHas => exact parts_cpuHas h
  | fileCreateDirAll => exact parts_fileCreateDirAll h
  | threadStart | threadFinish => change (none : Option (Option V × World)) = some (r, w') at h; cases h
  | memLock => exact parts_memLock h
  | memUnlock => exact parts_memUnlock h
  | memAdviseHuge => exact parts_memAdviseHuge h
  | threadPriority => exact parts_threadPriority h
  | threadSpawn => change (none : Option (Option V × World)) = some (r, w') at h; cases h

/-- Close one answer of a contract: the kernel and cuBLAS oracles as they were. -/
macro "oracle_close" : tactic => `(tactic| first
  | exact ⟨rfl, rfl⟩
  | (unfold World.pump; split <;> exact ⟨rfl, rfl⟩))

set_option maxHeartbeats 1000000 in
theorem oracles_fileRead {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileRead bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiFileRead bits w = some (r, w') at h
  unfold ffiFileRead at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_fileWrite {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileWrite bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiFileWrite bits w = some (r, w') at h
  unfold ffiFileWrite at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_fileReadToPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileReadToPtr bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiFileReadToPtr bits w = some (r, w') at h
  unfold ffiFileReadToPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_fileWriteFromPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileWriteFromPtr bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiFileWriteFromPtr bits w = some (r, w') at h
  unfold ffiFileWriteFromPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_stdinReadline {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .stdinReadline bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiStdinReadline bits w = some (r, w') at h
  unfold ffiStdinReadline at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_stdoutWrite {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .stdoutWrite bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiStdoutWrite bits w = some (r, w') at h
  unfold ffiStdoutWrite at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_sinf {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .sinf bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiSinf bits w = some (r, w') at h
  unfold ffiSinf at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cosf {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cosf bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCosf bits w = some (r, w') at h
  unfold ffiCosf at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_powf {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .powf bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiPowf bits w = some (r, w') at h
  unfold ffiPowf at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_htInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htInit bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiHtInit bits w = some (r, w') at h
  unfold ffiHtInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_htCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCleanup bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiHtCleanup bits w = some (r, w') at h
  unfold ffiHtCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_htCreate {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCreate bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiHtCreate bits w = some (r, w') at h
  unfold ffiHtCreate at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_htCount {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCount bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiHtCount bits w = some (r, w') at h
  unfold ffiHtCount at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_htLookup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htLookup bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiHtLookup bits w = some (r, w') at h
  unfold ffiHtLookup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_htInsert {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htInsert bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiHtInsert bits w = some (r, w') at h
  unfold ffiHtInsert at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_htIncrement {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htIncrement bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiHtIncrement bits w = some (r, w') at h
  unfold ffiHtIncrement at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_htGetEntry {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htGetEntry bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiHtGetEntry bits w = some (r, w') at h
  unfold ffiHtGetEntry at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaInit bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaInit bits w = some (r, w') at h
  unfold ffiCudaInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaCleanup bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaCleanup bits w = some (r, w') at h
  unfold ffiCudaCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaCreateBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaCreateBuffer bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaCreateBuffer bits w = some (r, w') at h
  unfold ffiCudaCreateBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaUpload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUpload bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaUpload bits w = some (r, w') at h
  unfold ffiCudaUpload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaUploadOffset {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUploadOffset bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaUploadOffset bits w = some (r, w') at h
  unfold ffiCudaUploadOffset at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaDownload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownload bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaDownload bits w = some (r, w') at h
  unfold ffiCudaDownload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaDownloadOffset {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownloadOffset bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaDownloadOffset bits w = some (r, w') at h
  unfold ffiCudaDownloadOffset at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaFreeBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaFreeBuffer bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaFreeBuffer bits w = some (r, w') at h
  unfold ffiCudaFreeBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaSync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaSync bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaSync bits w = some (r, w') at h
  unfold ffiCudaSync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaLaunch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunch bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaLaunch bits w = some (r, w') at h
  unfold ffiCudaLaunch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cublasSgemv {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemv bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCublasSgemv bits w = some (r, w') at h
  unfold ffiCublasSgemv at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cublasSgemm {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemm bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCublasSgemm bits w = some (r, w') at h
  unfold ffiCublasSgemm at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cublasGemmExBf16 {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasGemmExBf16 bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCublasGemmExBf16 bits w = some (r, w') at h
  unfold ffiCublasGemmExBf16 at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cublasSgemvOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemvOnStream bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCublasSgemvOnStream bits w = some (r, w') at h
  unfold ffiCublasSgemvOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cublasGemmStridedBatchedExBf16 {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasGemmStridedBatchedExBf16 bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCublasGemmStridedBatchedExBf16 bits w = some (r, w') at h
  unfold ffiCublasGemmStridedBatchedExBf16 at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cublasPtrArray {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasPtrArray bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCublasPtrArray bits w = some (r, w') at h
  unfold ffiCublasPtrArray at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cublasSgemmBatchedOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemmBatchedOnStream bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCublasSgemmBatchedOnStream bits w = some (r, w') at h
  unfold ffiCublasSgemmBatchedOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaLaunchNamedOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunchNamedOnStream bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaLaunchNamedOnStream bits w = some (r, w') at h
  unfold ffiCudaLaunchNamedOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaEventElapsedMsBits {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventElapsedMsBits bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaEventElapsedMsBits bits w = some (r, w') at h
  unfold ffiCudaEventElapsedMsBits at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaUploadAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUploadAsync bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaUploadAsync bits w = some (r, w') at h
  unfold ffiCudaUploadAsync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaUploadOffsetAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUploadOffsetAsync bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaUploadOffsetAsync bits w = some (r, w') at h
  unfold ffiCudaUploadOffsetAsync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaDownloadAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownloadAsync bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaDownloadAsync bits w = some (r, w') at h
  unfold ffiCudaDownloadAsync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cublasSgemmOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemmOnStream bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCublasSgemmOnStream bits w = some (r, w') at h
  unfold ffiCublasSgemmOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaLaunchOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunchOnStream bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaLaunchOnStream bits w = some (r, w') at h
  unfold ffiCudaLaunchOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaStreamCreate {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamCreate bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaStreamCreate bits w = some (r, w') at h
  unfold ffiCudaStreamCreate at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaStreamSync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamSync bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaStreamSync bits w = some (r, w') at h
  unfold ffiCudaStreamSync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaStreamDestroy {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamDestroy bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaStreamDestroy bits w = some (r, w') at h
  unfold ffiCudaStreamDestroy at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaEventCreate {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventCreate bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaEventCreate bits w = some (r, w') at h
  unfold ffiCudaEventCreate at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaEventRecord {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventRecord bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaEventRecord bits w = some (r, w') at h
  unfold ffiCudaEventRecord at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaStreamWaitEvent {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamWaitEvent bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaStreamWaitEvent bits w = some (r, w') at h
  unfold ffiCudaStreamWaitEvent at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaEventDestroy {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventDestroy bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaEventDestroy bits w = some (r, w') at h
  unfold ffiCudaEventDestroy at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaGraphBeginCapture {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphBeginCapture bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaGraphBeginCapture bits w = some (r, w') at h
  unfold ffiCudaGraphBeginCapture at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaGraphEndCapture {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphEndCapture bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaGraphEndCapture bits w = some (r, w') at h
  unfold ffiCudaGraphEndCapture at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaGraphUpload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphUpload bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaGraphUpload bits w = some (r, w') at h
  unfold ffiCudaGraphUpload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaGraphLaunch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphLaunch bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaGraphLaunch bits w = some (r, w') at h
  unfold ffiCudaGraphLaunch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaGraphDestroy {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphDestroy bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaGraphDestroy bits w = some (r, w') at h
  unfold ffiCudaGraphDestroy at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaLaunchNamed {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunchNamed bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaLaunchNamed bits w = some (r, w') at h
  unfold ffiCudaLaunchNamed at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaPinnedAlloc {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedAlloc bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaPinnedAlloc bits w = some (r, w') at h
  unfold ffiCudaPinnedAlloc at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaPinnedPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedPtr bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaPinnedPtr bits w = some (r, w') at h
  unfold ffiCudaPinnedPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaPinnedPtrAt {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedPtrAt bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaPinnedPtrAt bits w = some (r, w') at h
  unfold ffiCudaPinnedPtrAt at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaPinnedFree {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedFree bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaPinnedFree bits w = some (r, w') at h
  unfold ffiCudaPinnedFree at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaMemInfoFree {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaMemInfoFree bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaMemInfoFree bits w = some (r, w') at h
  unfold ffiCudaMemInfoFree at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cudaMemInfoTotal {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaMemInfoTotal bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCudaMemInfoTotal bits w = some (r, w') at h
  unfold ffiCudaMemInfoTotal at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_gpuInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuInit bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGpuInit bits w = some (r, w') at h
  unfold ffiGpuInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_gpuCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCleanup bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGpuCleanup bits w = some (r, w') at h
  unfold ffiGpuCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_gpuCreateBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCreateBuffer bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGpuCreateBuffer bits w = some (r, w') at h
  unfold ffiGpuCreateBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_gpuCreatePipeline {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCreatePipeline bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGpuCreatePipeline bits w = some (r, w') at h
  unfold ffiGpuCreatePipeline at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_gpuUpload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuUpload bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGpuUpload bits w = some (r, w') at h
  unfold ffiGpuUpload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_gpuUploadPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuUploadPtr bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGpuUploadPtr bits w = some (r, w') at h
  unfold ffiGpuUploadPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_gpuDispatch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuDispatch bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGpuDispatch bits w = some (r, w') at h
  unfold ffiGpuDispatch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_gpuDownload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuDownload bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGpuDownload bits w = some (r, w') at h
  unfold ffiGpuDownload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_gpuDownloadPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuDownloadPtr bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGpuDownloadPtr bits w = some (r, w') at h
  unfold ffiGpuDownloadPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_lmdbInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbInit bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiLmdbInit bits w = some (r, w') at h
  unfold ffiLmdbInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_lmdbCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCleanup bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiLmdbCleanup bits w = some (r, w') at h
  unfold ffiLmdbCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_lmdbOpen {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbOpen bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiLmdbOpen bits w = some (r, w') at h
  unfold ffiLmdbOpen at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_lmdbBeginWriteTxn {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbBeginWriteTxn bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiLmdbBeginWriteTxn bits w = some (r, w') at h
  unfold ffiLmdbBeginWriteTxn at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_lmdbPut {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbPut bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiLmdbPut bits w = some (r, w') at h
  unfold ffiLmdbPut at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_lmdbCommitWriteTxn {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCommitWriteTxn bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiLmdbCommitWriteTxn bits w = some (r, w') at h
  unfold ffiLmdbCommitWriteTxn at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_lmdbCursorScan {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCursorScan bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiLmdbCursorScan bits w = some (r, w') at h
  unfold ffiLmdbCursorScan at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_windowInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowInit bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiWindowInit bits w = some (r, w') at h
  unfold ffiWindowInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_windowCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowCleanup bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiWindowCleanup bits w = some (r, w') at h
  unfold ffiWindowCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_windowOpen {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowOpen bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiWindowOpen bits w = some (r, w') at h
  unfold ffiWindowOpen at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_windowPoll {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowPoll bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiWindowPoll bits w = some (r, w') at h
  unfold ffiWindowPoll at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_windowPresentGpuBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowPresentGpuBuffer bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiWindowPresentGpuBuffer bits w = some (r, w') at h
  unfold ffiWindowPresentGpuBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_nativeLoad {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .nativeLoad bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiNativeLoad bits w = some (r, w') at h
  unfold ffiNativeLoad at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_threadInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadInit bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiThreadInit bits w = some (r, w') at h
  unfold ffiThreadInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_threadCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadCleanup bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiThreadCleanup bits w = some (r, w') at h
  unfold ffiThreadCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_threadJoin {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadJoin bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiThreadJoin bits w = some (r, w') at h
  unfold ffiThreadJoin at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_nativeFree {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .nativeFree bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiNativeFree bits w = some (r, w') at h
  unfold ffiNativeFree at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_nativeArch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .nativeArch bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiNativeArch bits w = some (r, w') at h
  unfold ffiNativeArch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_cpuHas {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cpuHas bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiCpuHas bits w = some (r, w') at h
  unfold ffiCpuHas at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_fileCreateDirAll {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileCreateDirAll bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiFileCreateDirAll bits w = some (r, w') at h
  unfold ffiFileCreateDirAll at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_memLock {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .memLock bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGrant "lock" 2 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_memUnlock {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .memUnlock bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGrant "unlock" 2 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_memAdviseHuge {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .memAdviseHuge bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGrant "hugepages" 2 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

set_option maxHeartbeats 1000000 in
theorem oracles_threadPriority {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadPriority bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  change ffiGrant "priority" 1 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals oracle_close)
    | (walk h; all_goals oracle_close)

/-- **No call changes what a kernel or cuBLAS computes.** -/
theorem callBits_oracles (f : IR.Ffi) {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) : w'.kernel = w.kernel ∧ w'.vendor = w.vendor := by
  cases f with
  | fileRead => exact oracles_fileRead h
  | fileWrite => exact oracles_fileWrite h
  | fileReadToPtr => exact oracles_fileReadToPtr h
  | fileWriteFromPtr => exact oracles_fileWriteFromPtr h
  | stdinReadline => exact oracles_stdinReadline h
  | stdoutWrite => exact oracles_stdoutWrite h
  | sinf => exact oracles_sinf h
  | cosf => exact oracles_cosf h
  | powf => exact oracles_powf h
  | htInit => exact oracles_htInit h
  | htCleanup => exact oracles_htCleanup h
  | htCreate => exact oracles_htCreate h
  | htCount => exact oracles_htCount h
  | htLookup => exact oracles_htLookup h
  | htInsert => exact oracles_htInsert h
  | htIncrement => exact oracles_htIncrement h
  | htGetEntry => exact oracles_htGetEntry h
  | cudaInit => exact oracles_cudaInit h
  | cudaCleanup => exact oracles_cudaCleanup h
  | cudaCreateBuffer => exact oracles_cudaCreateBuffer h
  | cudaUpload => exact oracles_cudaUpload h
  | cudaUploadOffset => exact oracles_cudaUploadOffset h
  | cudaDownload => exact oracles_cudaDownload h
  | cudaDownloadOffset => exact oracles_cudaDownloadOffset h
  | cudaFreeBuffer => exact oracles_cudaFreeBuffer h
  | cudaSync => exact oracles_cudaSync h
  | cudaLaunch => exact oracles_cudaLaunch h
  | cublasSgemv => exact oracles_cublasSgemv h
  | cublasSgemm => exact oracles_cublasSgemm h
  | cublasGemmExBf16 => exact oracles_cublasGemmExBf16 h
  | cublasSgemvOnStream => exact oracles_cublasSgemvOnStream h
  | cublasGemmStridedBatchedExBf16 => exact oracles_cublasGemmStridedBatchedExBf16 h
  | cublasPtrArray => exact oracles_cublasPtrArray h
  | cublasSgemmBatchedOnStream => exact oracles_cublasSgemmBatchedOnStream h
  | cudaLaunchNamedOnStream => exact oracles_cudaLaunchNamedOnStream h
  | cudaEventElapsedMsBits => exact oracles_cudaEventElapsedMsBits h
  | cudaUploadAsync => exact oracles_cudaUploadAsync h
  | cudaUploadOffsetAsync => exact oracles_cudaUploadOffsetAsync h
  | cudaDownloadAsync => exact oracles_cudaDownloadAsync h
  | cublasSgemmOnStream => exact oracles_cublasSgemmOnStream h
  | cudaLaunchOnStream => exact oracles_cudaLaunchOnStream h
  | cudaStreamCreate => exact oracles_cudaStreamCreate h
  | cudaStreamSync => exact oracles_cudaStreamSync h
  | cudaStreamDestroy => exact oracles_cudaStreamDestroy h
  | cudaEventCreate => exact oracles_cudaEventCreate h
  | cudaEventRecord => exact oracles_cudaEventRecord h
  | cudaStreamWaitEvent => exact oracles_cudaStreamWaitEvent h
  | cudaEventDestroy => exact oracles_cudaEventDestroy h
  | cudaGraphBeginCapture => exact oracles_cudaGraphBeginCapture h
  | cudaGraphEndCapture => exact oracles_cudaGraphEndCapture h
  | cudaGraphUpload => exact oracles_cudaGraphUpload h
  | cudaGraphLaunch => exact oracles_cudaGraphLaunch h
  | cudaGraphDestroy => exact oracles_cudaGraphDestroy h
  | cudaLaunchNamed => exact oracles_cudaLaunchNamed h
  | cudaPinnedAlloc => exact oracles_cudaPinnedAlloc h
  | cudaPinnedPtr => exact oracles_cudaPinnedPtr h
  | cudaPinnedPtrAt => exact oracles_cudaPinnedPtrAt h
  | cudaPinnedFree => exact oracles_cudaPinnedFree h
  | cudaMemInfoFree => exact oracles_cudaMemInfoFree h
  | cudaMemInfoTotal => exact oracles_cudaMemInfoTotal h
  | gpuInit => exact oracles_gpuInit h
  | gpuCleanup => exact oracles_gpuCleanup h
  | gpuCreateBuffer => exact oracles_gpuCreateBuffer h
  | gpuCreatePipeline => exact oracles_gpuCreatePipeline h
  | gpuUpload => exact oracles_gpuUpload h
  | gpuUploadPtr => exact oracles_gpuUploadPtr h
  | gpuDispatch => exact oracles_gpuDispatch h
  | gpuDownload => exact oracles_gpuDownload h
  | gpuDownloadPtr => exact oracles_gpuDownloadPtr h
  | lmdbInit => exact oracles_lmdbInit h
  | lmdbCleanup => exact oracles_lmdbCleanup h
  | lmdbOpen => exact oracles_lmdbOpen h
  | lmdbBeginWriteTxn => exact oracles_lmdbBeginWriteTxn h
  | lmdbPut => exact oracles_lmdbPut h
  | lmdbCommitWriteTxn => exact oracles_lmdbCommitWriteTxn h
  | lmdbCursorScan => exact oracles_lmdbCursorScan h
  | windowInit => exact oracles_windowInit h
  | windowCleanup => exact oracles_windowCleanup h
  | windowOpen => exact oracles_windowOpen h
  | windowPoll => exact oracles_windowPoll h
  | windowPresentGpuBuffer => exact oracles_windowPresentGpuBuffer h
  | nativeLoad => exact oracles_nativeLoad h
  | threadInit => exact oracles_threadInit h
  | threadCleanup => exact oracles_threadCleanup h
  | threadJoin => exact oracles_threadJoin h
  | nativeFree => exact oracles_nativeFree h
  | nativeArch => exact oracles_nativeArch h
  | cpuHas => exact oracles_cpuHas h
  | fileCreateDirAll => exact oracles_fileCreateDirAll h
  | memLock => exact oracles_memLock h
  | memUnlock => exact oracles_memUnlock h
  | memAdviseHuge => exact oracles_memAdviseHuge h
  | threadPriority => exact oracles_threadPriority h
  | threadStart | threadFinish => change (none : Option (Option V × World)) = some (r, w') at h; cases h
  | threadSpawn => change (none : Option (Option V × World)) = some (r, w') at h; cases h

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

set_option maxHeartbeats 1000000 in
theorem rooms_fileRead {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileRead bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiFileRead bits w = some (r, w') at h
  unfold ffiFileRead at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_fileWrite {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileWrite bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiFileWrite bits w = some (r, w') at h
  unfold ffiFileWrite at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_fileReadToPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileReadToPtr bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiFileReadToPtr bits w = some (r, w') at h
  unfold ffiFileReadToPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_fileWriteFromPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileWriteFromPtr bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiFileWriteFromPtr bits w = some (r, w') at h
  unfold ffiFileWriteFromPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_stdinReadline {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .stdinReadline bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiStdinReadline bits w = some (r, w') at h
  unfold ffiStdinReadline at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_stdoutWrite {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .stdoutWrite bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiStdoutWrite bits w = some (r, w') at h
  unfold ffiStdoutWrite at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_sinf {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .sinf bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiSinf bits w = some (r, w') at h
  unfold ffiSinf at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cosf {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cosf bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCosf bits w = some (r, w') at h
  unfold ffiCosf at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_powf {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .powf bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiPowf bits w = some (r, w') at h
  unfold ffiPowf at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_htInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htInit bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiHtInit bits w = some (r, w') at h
  unfold ffiHtInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_htCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCleanup bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiHtCleanup bits w = some (r, w') at h
  unfold ffiHtCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_htCreate {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCreate bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiHtCreate bits w = some (r, w') at h
  unfold ffiHtCreate at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_htCount {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCount bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiHtCount bits w = some (r, w') at h
  unfold ffiHtCount at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_htLookup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htLookup bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiHtLookup bits w = some (r, w') at h
  unfold ffiHtLookup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_htInsert {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htInsert bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiHtInsert bits w = some (r, w') at h
  unfold ffiHtInsert at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_htIncrement {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htIncrement bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiHtIncrement bits w = some (r, w') at h
  unfold ffiHtIncrement at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_htGetEntry {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htGetEntry bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiHtGetEntry bits w = some (r, w') at h
  unfold ffiHtGetEntry at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaInit bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaInit bits w = some (r, w') at h
  unfold ffiCudaInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaCleanup bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaCleanup bits w = some (r, w') at h
  unfold ffiCudaCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaCreateBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaCreateBuffer bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaCreateBuffer bits w = some (r, w') at h
  unfold ffiCudaCreateBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaUpload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUpload bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaUpload bits w = some (r, w') at h
  unfold ffiCudaUpload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaUploadOffset {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUploadOffset bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaUploadOffset bits w = some (r, w') at h
  unfold ffiCudaUploadOffset at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaDownload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownload bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaDownload bits w = some (r, w') at h
  unfold ffiCudaDownload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaDownloadOffset {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownloadOffset bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaDownloadOffset bits w = some (r, w') at h
  unfold ffiCudaDownloadOffset at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaFreeBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaFreeBuffer bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaFreeBuffer bits w = some (r, w') at h
  unfold ffiCudaFreeBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaSync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaSync bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaSync bits w = some (r, w') at h
  unfold ffiCudaSync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaLaunch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunch bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaLaunch bits w = some (r, w') at h
  unfold ffiCudaLaunch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cublasSgemv {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemv bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCublasSgemv bits w = some (r, w') at h
  unfold ffiCublasSgemv at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cublasSgemm {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemm bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCublasSgemm bits w = some (r, w') at h
  unfold ffiCublasSgemm at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cublasGemmExBf16 {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasGemmExBf16 bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCublasGemmExBf16 bits w = some (r, w') at h
  unfold ffiCublasGemmExBf16 at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cublasSgemvOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemvOnStream bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCublasSgemvOnStream bits w = some (r, w') at h
  unfold ffiCublasSgemvOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cublasGemmStridedBatchedExBf16 {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasGemmStridedBatchedExBf16 bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCublasGemmStridedBatchedExBf16 bits w = some (r, w') at h
  unfold ffiCublasGemmStridedBatchedExBf16 at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cublasPtrArray {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasPtrArray bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCublasPtrArray bits w = some (r, w') at h
  unfold ffiCublasPtrArray at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cublasSgemmBatchedOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemmBatchedOnStream bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCublasSgemmBatchedOnStream bits w = some (r, w') at h
  unfold ffiCublasSgemmBatchedOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaLaunchNamedOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunchNamedOnStream bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaLaunchNamedOnStream bits w = some (r, w') at h
  unfold ffiCudaLaunchNamedOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaEventElapsedMsBits {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventElapsedMsBits bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaEventElapsedMsBits bits w = some (r, w') at h
  unfold ffiCudaEventElapsedMsBits at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaUploadAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUploadAsync bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaUploadAsync bits w = some (r, w') at h
  unfold ffiCudaUploadAsync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaUploadOffsetAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUploadOffsetAsync bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaUploadOffsetAsync bits w = some (r, w') at h
  unfold ffiCudaUploadOffsetAsync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
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

set_option maxHeartbeats 1000000 in
theorem rooms_cublasSgemmOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemmOnStream bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCublasSgemmOnStream bits w = some (r, w') at h
  unfold ffiCublasSgemmOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaLaunchOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunchOnStream bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaLaunchOnStream bits w = some (r, w') at h
  unfold ffiCudaLaunchOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaStreamCreate {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamCreate bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaStreamCreate bits w = some (r, w') at h
  unfold ffiCudaStreamCreate at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaStreamSync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamSync bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaStreamSync bits w = some (r, w') at h
  unfold ffiCudaStreamSync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaStreamDestroy {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamDestroy bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaStreamDestroy bits w = some (r, w') at h
  unfold ffiCudaStreamDestroy at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaEventCreate {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventCreate bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaEventCreate bits w = some (r, w') at h
  unfold ffiCudaEventCreate at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaEventRecord {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventRecord bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaEventRecord bits w = some (r, w') at h
  unfold ffiCudaEventRecord at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaStreamWaitEvent {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamWaitEvent bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaStreamWaitEvent bits w = some (r, w') at h
  unfold ffiCudaStreamWaitEvent at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaEventDestroy {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventDestroy bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaEventDestroy bits w = some (r, w') at h
  unfold ffiCudaEventDestroy at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaGraphBeginCapture {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphBeginCapture bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaGraphBeginCapture bits w = some (r, w') at h
  unfold ffiCudaGraphBeginCapture at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaGraphEndCapture {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphEndCapture bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaGraphEndCapture bits w = some (r, w') at h
  unfold ffiCudaGraphEndCapture at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaGraphUpload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphUpload bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaGraphUpload bits w = some (r, w') at h
  unfold ffiCudaGraphUpload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaGraphLaunch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphLaunch bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaGraphLaunch bits w = some (r, w') at h
  unfold ffiCudaGraphLaunch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaGraphDestroy {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphDestroy bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaGraphDestroy bits w = some (r, w') at h
  unfold ffiCudaGraphDestroy at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaLaunchNamed {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunchNamed bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaLaunchNamed bits w = some (r, w') at h
  unfold ffiCudaLaunchNamed at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaPinnedAlloc {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedAlloc bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaPinnedAlloc bits w = some (r, w') at h
  unfold ffiCudaPinnedAlloc at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaPinnedPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedPtr bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaPinnedPtr bits w = some (r, w') at h
  unfold ffiCudaPinnedPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaPinnedPtrAt {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedPtrAt bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaPinnedPtrAt bits w = some (r, w') at h
  unfold ffiCudaPinnedPtrAt at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaPinnedFree {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedFree bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaPinnedFree bits w = some (r, w') at h
  unfold ffiCudaPinnedFree at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaMemInfoFree {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaMemInfoFree bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaMemInfoFree bits w = some (r, w') at h
  unfold ffiCudaMemInfoFree at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cudaMemInfoTotal {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaMemInfoTotal bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCudaMemInfoTotal bits w = some (r, w') at h
  unfold ffiCudaMemInfoTotal at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_gpuInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuInit bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGpuInit bits w = some (r, w') at h
  unfold ffiGpuInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_gpuCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCleanup bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGpuCleanup bits w = some (r, w') at h
  unfold ffiGpuCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_gpuCreateBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCreateBuffer bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGpuCreateBuffer bits w = some (r, w') at h
  unfold ffiGpuCreateBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_gpuCreatePipeline {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCreatePipeline bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGpuCreatePipeline bits w = some (r, w') at h
  unfold ffiGpuCreatePipeline at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_gpuUpload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuUpload bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGpuUpload bits w = some (r, w') at h
  unfold ffiGpuUpload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_gpuUploadPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuUploadPtr bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGpuUploadPtr bits w = some (r, w') at h
  unfold ffiGpuUploadPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_gpuDispatch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuDispatch bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGpuDispatch bits w = some (r, w') at h
  unfold ffiGpuDispatch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_gpuDownload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuDownload bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGpuDownload bits w = some (r, w') at h
  unfold ffiGpuDownload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_gpuDownloadPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuDownloadPtr bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGpuDownloadPtr bits w = some (r, w') at h
  unfold ffiGpuDownloadPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_lmdbInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbInit bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiLmdbInit bits w = some (r, w') at h
  unfold ffiLmdbInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_lmdbCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCleanup bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiLmdbCleanup bits w = some (r, w') at h
  unfold ffiLmdbCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_lmdbOpen {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbOpen bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiLmdbOpen bits w = some (r, w') at h
  unfold ffiLmdbOpen at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_lmdbBeginWriteTxn {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbBeginWriteTxn bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiLmdbBeginWriteTxn bits w = some (r, w') at h
  unfold ffiLmdbBeginWriteTxn at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_lmdbPut {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbPut bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiLmdbPut bits w = some (r, w') at h
  unfold ffiLmdbPut at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_lmdbCommitWriteTxn {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCommitWriteTxn bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiLmdbCommitWriteTxn bits w = some (r, w') at h
  unfold ffiLmdbCommitWriteTxn at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_lmdbCursorScan {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCursorScan bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiLmdbCursorScan bits w = some (r, w') at h
  unfold ffiLmdbCursorScan at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_windowInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowInit bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiWindowInit bits w = some (r, w') at h
  unfold ffiWindowInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_windowCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowCleanup bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiWindowCleanup bits w = some (r, w') at h
  unfold ffiWindowCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_windowOpen {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowOpen bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiWindowOpen bits w = some (r, w') at h
  unfold ffiWindowOpen at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_windowPoll {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowPoll bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiWindowPoll bits w = some (r, w') at h
  unfold ffiWindowPoll at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_windowPresentGpuBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowPresentGpuBuffer bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiWindowPresentGpuBuffer bits w = some (r, w') at h
  unfold ffiWindowPresentGpuBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_nativeLoad {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .nativeLoad bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiNativeLoad bits w = some (r, w') at h
  unfold ffiNativeLoad at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_threadInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadInit bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiThreadInit bits w = some (r, w') at h
  unfold ffiThreadInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_threadCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadCleanup bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiThreadCleanup bits w = some (r, w') at h
  unfold ffiThreadCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_threadJoin {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadJoin bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiThreadJoin bits w = some (r, w') at h
  unfold ffiThreadJoin at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_nativeFree {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .nativeFree bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiNativeFree bits w = some (r, w') at h
  unfold ffiNativeFree at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_nativeArch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .nativeArch bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiNativeArch bits w = some (r, w') at h
  unfold ffiNativeArch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_cpuHas {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cpuHas bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiCpuHas bits w = some (r, w') at h
  unfold ffiCpuHas at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_fileCreateDirAll {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileCreateDirAll bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiFileCreateDirAll bits w = some (r, w') at h
  unfold ffiFileCreateDirAll at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_memLock {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .memLock bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGrant "lock" 2 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_memUnlock {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .memUnlock bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGrant "unlock" 2 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_memAdviseHuge {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .memAdviseHuge bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGrant "hugepages" 2 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
theorem rooms_threadPriority {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadPriority bits w = some (r, w')) : KeepsRooms w w' := by
  change ffiGrant "priority" 1 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals room_close)
    | (walk h; all_goals room_close)
set_option maxHeartbeats 1000000 in
set_option maxHeartbeats 1000000 in
set_option maxHeartbeats 1000000 in
set_option maxHeartbeats 1000000 in
/-- **Every call keeps the size of every region but the pinned one.** -/
theorem callBits_rooms (f : IR.Ffi) {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) : KeepsRooms w w' := by
  cases f with
  | fileRead => exact rooms_fileRead h
  | fileWrite => exact rooms_fileWrite h
  | fileReadToPtr => exact rooms_fileReadToPtr h
  | fileWriteFromPtr => exact rooms_fileWriteFromPtr h
  | stdinReadline => exact rooms_stdinReadline h
  | stdoutWrite => exact rooms_stdoutWrite h
  | sinf => exact rooms_sinf h
  | cosf => exact rooms_cosf h
  | powf => exact rooms_powf h
  | htInit => exact rooms_htInit h
  | htCleanup => exact rooms_htCleanup h
  | htCreate => exact rooms_htCreate h
  | htCount => exact rooms_htCount h
  | htLookup => exact rooms_htLookup h
  | htInsert => exact rooms_htInsert h
  | htIncrement => exact rooms_htIncrement h
  | htGetEntry => exact rooms_htGetEntry h
  | cudaInit => exact rooms_cudaInit h
  | cudaCleanup => exact rooms_cudaCleanup h
  | cudaCreateBuffer => exact rooms_cudaCreateBuffer h
  | cudaUpload => exact rooms_cudaUpload h
  | cudaUploadOffset => exact rooms_cudaUploadOffset h
  | cudaDownload => exact rooms_cudaDownload h
  | cudaDownloadOffset => exact rooms_cudaDownloadOffset h
  | cudaFreeBuffer => exact rooms_cudaFreeBuffer h
  | cudaSync => exact rooms_cudaSync h
  | cudaLaunch => exact rooms_cudaLaunch h
  | cublasSgemv => exact rooms_cublasSgemv h
  | cublasSgemm => exact rooms_cublasSgemm h
  | cublasGemmExBf16 => exact rooms_cublasGemmExBf16 h
  | cublasSgemvOnStream => exact rooms_cublasSgemvOnStream h
  | cublasGemmStridedBatchedExBf16 => exact rooms_cublasGemmStridedBatchedExBf16 h
  | cublasPtrArray => exact rooms_cublasPtrArray h
  | cublasSgemmBatchedOnStream => exact rooms_cublasSgemmBatchedOnStream h
  | cudaLaunchNamedOnStream => exact rooms_cudaLaunchNamedOnStream h
  | cudaEventElapsedMsBits => exact rooms_cudaEventElapsedMsBits h
  | cudaUploadAsync => exact rooms_cudaUploadAsync h
  | cudaUploadOffsetAsync => exact rooms_cudaUploadOffsetAsync h
  | cudaDownloadAsync => exact rooms_cudaDownloadAsync h
  | cublasSgemmOnStream => exact rooms_cublasSgemmOnStream h
  | cudaLaunchOnStream => exact rooms_cudaLaunchOnStream h
  | cudaStreamCreate => exact rooms_cudaStreamCreate h
  | cudaStreamSync => exact rooms_cudaStreamSync h
  | cudaStreamDestroy => exact rooms_cudaStreamDestroy h
  | cudaEventCreate => exact rooms_cudaEventCreate h
  | cudaEventRecord => exact rooms_cudaEventRecord h
  | cudaStreamWaitEvent => exact rooms_cudaStreamWaitEvent h
  | cudaEventDestroy => exact rooms_cudaEventDestroy h
  | cudaGraphBeginCapture => exact rooms_cudaGraphBeginCapture h
  | cudaGraphEndCapture => exact rooms_cudaGraphEndCapture h
  | cudaGraphUpload => exact rooms_cudaGraphUpload h
  | cudaGraphLaunch => exact rooms_cudaGraphLaunch h
  | cudaGraphDestroy => exact rooms_cudaGraphDestroy h
  | cudaLaunchNamed => exact rooms_cudaLaunchNamed h
  | cudaPinnedAlloc => exact rooms_cudaPinnedAlloc h
  | cudaPinnedPtr => exact rooms_cudaPinnedPtr h
  | cudaPinnedPtrAt => exact rooms_cudaPinnedPtrAt h
  | cudaPinnedFree => exact rooms_cudaPinnedFree h
  | cudaMemInfoFree => exact rooms_cudaMemInfoFree h
  | cudaMemInfoTotal => exact rooms_cudaMemInfoTotal h
  | gpuInit => exact rooms_gpuInit h
  | gpuCleanup => exact rooms_gpuCleanup h
  | gpuCreateBuffer => exact rooms_gpuCreateBuffer h
  | gpuCreatePipeline => exact rooms_gpuCreatePipeline h
  | gpuUpload => exact rooms_gpuUpload h
  | gpuUploadPtr => exact rooms_gpuUploadPtr h
  | gpuDispatch => exact rooms_gpuDispatch h
  | gpuDownload => exact rooms_gpuDownload h
  | gpuDownloadPtr => exact rooms_gpuDownloadPtr h
  | lmdbInit => exact rooms_lmdbInit h
  | lmdbCleanup => exact rooms_lmdbCleanup h
  | lmdbOpen => exact rooms_lmdbOpen h
  | lmdbBeginWriteTxn => exact rooms_lmdbBeginWriteTxn h
  | lmdbPut => exact rooms_lmdbPut h
  | lmdbCommitWriteTxn => exact rooms_lmdbCommitWriteTxn h
  | lmdbCursorScan => exact rooms_lmdbCursorScan h
  | windowInit => exact rooms_windowInit h
  | windowCleanup => exact rooms_windowCleanup h
  | windowOpen => exact rooms_windowOpen h
  | windowPoll => exact rooms_windowPoll h
  | windowPresentGpuBuffer => exact rooms_windowPresentGpuBuffer h
  | nativeLoad => exact rooms_nativeLoad h
  | threadInit => exact rooms_threadInit h
  | threadCleanup => exact rooms_threadCleanup h
  | threadJoin => exact rooms_threadJoin h
  | nativeFree => exact rooms_nativeFree h
  | nativeArch => exact rooms_nativeArch h
  | cpuHas => exact rooms_cpuHas h
  | fileCreateDirAll => exact rooms_fileCreateDirAll h
  | threadStart | threadFinish => change (none : Option (Option V × World)) = some (r, w') at h; cases h
  | memLock => exact rooms_memLock h
  | memUnlock => exact rooms_memUnlock h
  | memAdviseHuge => exact rooms_memAdviseHuge h
  | threadPriority => exact rooms_threadPriority h
  | threadSpawn => change (none : Option (Option V × World)) = some (r, w') at h; cases h

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
theorem pinned_fileRead {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileRead bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiFileRead bits w = some (r, w') at h
  unfold ffiFileRead at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_fileWrite {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileWrite bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiFileWrite bits w = some (r, w') at h
  unfold ffiFileWrite at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_fileReadToPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileReadToPtr bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiFileReadToPtr bits w = some (r, w') at h
  unfold ffiFileReadToPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_fileWriteFromPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileWriteFromPtr bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiFileWriteFromPtr bits w = some (r, w') at h
  unfold ffiFileWriteFromPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_stdinReadline {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .stdinReadline bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiStdinReadline bits w = some (r, w') at h
  unfold ffiStdinReadline at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_stdoutWrite {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .stdoutWrite bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiStdoutWrite bits w = some (r, w') at h
  unfold ffiStdoutWrite at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_sinf {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .sinf bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiSinf bits w = some (r, w') at h
  unfold ffiSinf at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cosf {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cosf bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCosf bits w = some (r, w') at h
  unfold ffiCosf at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_powf {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .powf bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiPowf bits w = some (r, w') at h
  unfold ffiPowf at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_htInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htInit bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiHtInit bits w = some (r, w') at h
  unfold ffiHtInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_htCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCleanup bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiHtCleanup bits w = some (r, w') at h
  unfold ffiHtCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_htCreate {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCreate bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiHtCreate bits w = some (r, w') at h
  unfold ffiHtCreate at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_htCount {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCount bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiHtCount bits w = some (r, w') at h
  unfold ffiHtCount at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_htLookup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htLookup bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiHtLookup bits w = some (r, w') at h
  unfold ffiHtLookup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_htInsert {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htInsert bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiHtInsert bits w = some (r, w') at h
  unfold ffiHtInsert at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_htIncrement {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htIncrement bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiHtIncrement bits w = some (r, w') at h
  unfold ffiHtIncrement at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_htGetEntry {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htGetEntry bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiHtGetEntry bits w = some (r, w') at h
  unfold ffiHtGetEntry at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaInit bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaInit bits w = some (r, w') at h
  unfold ffiCudaInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaCleanup bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaCleanup bits w = some (r, w') at h
  unfold ffiCudaCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaCreateBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaCreateBuffer bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaCreateBuffer bits w = some (r, w') at h
  unfold ffiCudaCreateBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaUpload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUpload bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaUpload bits w = some (r, w') at h
  unfold ffiCudaUpload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaUploadOffset {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUploadOffset bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaUploadOffset bits w = some (r, w') at h
  unfold ffiCudaUploadOffset at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaDownload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownload bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaDownload bits w = some (r, w') at h
  unfold ffiCudaDownload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaDownloadOffset {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaDownloadOffset bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaDownloadOffset bits w = some (r, w') at h
  unfold ffiCudaDownloadOffset at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaFreeBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaFreeBuffer bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaFreeBuffer bits w = some (r, w') at h
  unfold ffiCudaFreeBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaSync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaSync bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaSync bits w = some (r, w') at h
  unfold ffiCudaSync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaLaunch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunch bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaLaunch bits w = some (r, w') at h
  unfold ffiCudaLaunch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cublasSgemv {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemv bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCublasSgemv bits w = some (r, w') at h
  unfold ffiCublasSgemv at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cublasSgemm {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemm bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCublasSgemm bits w = some (r, w') at h
  unfold ffiCublasSgemm at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cublasGemmExBf16 {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasGemmExBf16 bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCublasGemmExBf16 bits w = some (r, w') at h
  unfold ffiCublasGemmExBf16 at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cublasSgemvOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemvOnStream bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCublasSgemvOnStream bits w = some (r, w') at h
  unfold ffiCublasSgemvOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cublasGemmStridedBatchedExBf16 {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasGemmStridedBatchedExBf16 bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCublasGemmStridedBatchedExBf16 bits w = some (r, w') at h
  unfold ffiCublasGemmStridedBatchedExBf16 at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cublasPtrArray {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasPtrArray bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCublasPtrArray bits w = some (r, w') at h
  unfold ffiCublasPtrArray at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cublasSgemmBatchedOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemmBatchedOnStream bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCublasSgemmBatchedOnStream bits w = some (r, w') at h
  unfold ffiCublasSgemmBatchedOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaLaunchNamedOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunchNamedOnStream bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaLaunchNamedOnStream bits w = some (r, w') at h
  unfold ffiCudaLaunchNamedOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaEventElapsedMsBits {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventElapsedMsBits bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaEventElapsedMsBits bits w = some (r, w') at h
  unfold ffiCudaEventElapsedMsBits at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaUploadAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUploadAsync bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaUploadAsync bits w = some (r, w') at h
  unfold ffiCudaUploadAsync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaUploadOffsetAsync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaUploadOffsetAsync bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaUploadOffsetAsync bits w = some (r, w') at h
  unfold ffiCudaUploadOffsetAsync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

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


set_option maxHeartbeats 1000000 in
theorem pinned_cublasSgemmOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cublasSgemmOnStream bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCublasSgemmOnStream bits w = some (r, w') at h
  unfold ffiCublasSgemmOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaLaunchOnStream {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunchOnStream bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaLaunchOnStream bits w = some (r, w') at h
  unfold ffiCudaLaunchOnStream at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaStreamCreate {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamCreate bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaStreamCreate bits w = some (r, w') at h
  unfold ffiCudaStreamCreate at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaStreamSync {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamSync bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaStreamSync bits w = some (r, w') at h
  unfold ffiCudaStreamSync at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaStreamDestroy {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamDestroy bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaStreamDestroy bits w = some (r, w') at h
  unfold ffiCudaStreamDestroy at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaEventCreate {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventCreate bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaEventCreate bits w = some (r, w') at h
  unfold ffiCudaEventCreate at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaEventRecord {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventRecord bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaEventRecord bits w = some (r, w') at h
  unfold ffiCudaEventRecord at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaStreamWaitEvent {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaStreamWaitEvent bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaStreamWaitEvent bits w = some (r, w') at h
  unfold ffiCudaStreamWaitEvent at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaEventDestroy {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaEventDestroy bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaEventDestroy bits w = some (r, w') at h
  unfold ffiCudaEventDestroy at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaGraphBeginCapture {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphBeginCapture bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaGraphBeginCapture bits w = some (r, w') at h
  unfold ffiCudaGraphBeginCapture at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaGraphEndCapture {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphEndCapture bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaGraphEndCapture bits w = some (r, w') at h
  unfold ffiCudaGraphEndCapture at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaGraphUpload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphUpload bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaGraphUpload bits w = some (r, w') at h
  unfold ffiCudaGraphUpload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaGraphLaunch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphLaunch bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaGraphLaunch bits w = some (r, w') at h
  unfold ffiCudaGraphLaunch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaGraphDestroy {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaGraphDestroy bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaGraphDestroy bits w = some (r, w') at h
  unfold ffiCudaGraphDestroy at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaLaunchNamed {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaLaunchNamed bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaLaunchNamed bits w = some (r, w') at h
  unfold ffiCudaLaunchNamed at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaPinnedPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedPtr bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaPinnedPtr bits w = some (r, w') at h
  unfold ffiCudaPinnedPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaPinnedPtrAt {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedPtrAt bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaPinnedPtrAt bits w = some (r, w') at h
  unfold ffiCudaPinnedPtrAt at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaPinnedFree {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaPinnedFree bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaPinnedFree bits w = some (r, w') at h
  unfold ffiCudaPinnedFree at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaMemInfoFree {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaMemInfoFree bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaMemInfoFree bits w = some (r, w') at h
  unfold ffiCudaMemInfoFree at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cudaMemInfoTotal {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cudaMemInfoTotal bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCudaMemInfoTotal bits w = some (r, w') at h
  unfold ffiCudaMemInfoTotal at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_gpuInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuInit bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGpuInit bits w = some (r, w') at h
  unfold ffiGpuInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_gpuCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCleanup bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGpuCleanup bits w = some (r, w') at h
  unfold ffiGpuCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_gpuCreateBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCreateBuffer bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGpuCreateBuffer bits w = some (r, w') at h
  unfold ffiGpuCreateBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_gpuCreatePipeline {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuCreatePipeline bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGpuCreatePipeline bits w = some (r, w') at h
  unfold ffiGpuCreatePipeline at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_gpuUpload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuUpload bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGpuUpload bits w = some (r, w') at h
  unfold ffiGpuUpload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_gpuUploadPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuUploadPtr bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGpuUploadPtr bits w = some (r, w') at h
  unfold ffiGpuUploadPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_gpuDispatch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuDispatch bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGpuDispatch bits w = some (r, w') at h
  unfold ffiGpuDispatch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_gpuDownload {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuDownload bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGpuDownload bits w = some (r, w') at h
  unfold ffiGpuDownload at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_gpuDownloadPtr {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .gpuDownloadPtr bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGpuDownloadPtr bits w = some (r, w') at h
  unfold ffiGpuDownloadPtr at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_lmdbInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbInit bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiLmdbInit bits w = some (r, w') at h
  unfold ffiLmdbInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_lmdbCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCleanup bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiLmdbCleanup bits w = some (r, w') at h
  unfold ffiLmdbCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_lmdbOpen {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbOpen bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiLmdbOpen bits w = some (r, w') at h
  unfold ffiLmdbOpen at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_lmdbBeginWriteTxn {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbBeginWriteTxn bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiLmdbBeginWriteTxn bits w = some (r, w') at h
  unfold ffiLmdbBeginWriteTxn at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_lmdbPut {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbPut bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiLmdbPut bits w = some (r, w') at h
  unfold ffiLmdbPut at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_lmdbCommitWriteTxn {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCommitWriteTxn bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiLmdbCommitWriteTxn bits w = some (r, w') at h
  unfold ffiLmdbCommitWriteTxn at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_lmdbCursorScan {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .lmdbCursorScan bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiLmdbCursorScan bits w = some (r, w') at h
  unfold ffiLmdbCursorScan at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_windowInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowInit bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiWindowInit bits w = some (r, w') at h
  unfold ffiWindowInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_windowCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowCleanup bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiWindowCleanup bits w = some (r, w') at h
  unfold ffiWindowCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_windowOpen {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowOpen bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiWindowOpen bits w = some (r, w') at h
  unfold ffiWindowOpen at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_windowPoll {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowPoll bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiWindowPoll bits w = some (r, w') at h
  unfold ffiWindowPoll at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_windowPresentGpuBuffer {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .windowPresentGpuBuffer bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiWindowPresentGpuBuffer bits w = some (r, w') at h
  unfold ffiWindowPresentGpuBuffer at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_nativeLoad {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .nativeLoad bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiNativeLoad bits w = some (r, w') at h
  unfold ffiNativeLoad at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_threadInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadInit bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiThreadInit bits w = some (r, w') at h
  unfold ffiThreadInit at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_threadCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadCleanup bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiThreadCleanup bits w = some (r, w') at h
  unfold ffiThreadCleanup at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_threadJoin {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadJoin bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiThreadJoin bits w = some (r, w') at h
  unfold ffiThreadJoin at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_nativeFree {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .nativeFree bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiNativeFree bits w = some (r, w') at h
  unfold ffiNativeFree at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_nativeArch {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .nativeArch bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiNativeArch bits w = some (r, w') at h
  unfold ffiNativeArch at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_cpuHas {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .cpuHas bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiCpuHas bits w = some (r, w') at h
  unfold ffiCpuHas at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_fileCreateDirAll {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileCreateDirAll bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiFileCreateDirAll bits w = some (r, w') at h
  unfold ffiFileCreateDirAll at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_memLock {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .memLock bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGrant "lock" 2 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_memUnlock {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .memUnlock bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGrant "unlock" 2 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_memAdviseHuge {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .memAdviseHuge bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGrant "hugepages" 2 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

set_option maxHeartbeats 1000000 in
theorem pinned_threadPriority {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .threadPriority bits w = some (r, w')) : KeepsPinned w w' := by
  change ffiGrant "priority" 1 bits w = some (r, w') at h
  unfold ffiGrant at h
  first
    | (obtain ⟨d, hk, rfl⟩ := devOnly_eq h; clear h; walk hk; all_goals pinned_close)
    | (walk h; all_goals pinned_close)

/-- **Every call but the one that allocates it leaves page-locked memory the
    size it was.** -/
theorem callBits_pinned (f : IR.Ffi) {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (hf : f ≠ .cudaPinnedAlloc) (h : callBits f bits w = some (r, w')) : KeepsPinned w w' := by
  cases f with
  | fileRead => exact pinned_fileRead h
  | fileWrite => exact pinned_fileWrite h
  | fileReadToPtr => exact pinned_fileReadToPtr h
  | fileWriteFromPtr => exact pinned_fileWriteFromPtr h
  | stdinReadline => exact pinned_stdinReadline h
  | stdoutWrite => exact pinned_stdoutWrite h
  | sinf => exact pinned_sinf h
  | cosf => exact pinned_cosf h
  | powf => exact pinned_powf h
  | htInit => exact pinned_htInit h
  | htCleanup => exact pinned_htCleanup h
  | htCreate => exact pinned_htCreate h
  | htCount => exact pinned_htCount h
  | htLookup => exact pinned_htLookup h
  | htInsert => exact pinned_htInsert h
  | htIncrement => exact pinned_htIncrement h
  | htGetEntry => exact pinned_htGetEntry h
  | cudaInit => exact pinned_cudaInit h
  | cudaCleanup => exact pinned_cudaCleanup h
  | cudaCreateBuffer => exact pinned_cudaCreateBuffer h
  | cudaUpload => exact pinned_cudaUpload h
  | cudaUploadOffset => exact pinned_cudaUploadOffset h
  | cudaDownload => exact pinned_cudaDownload h
  | cudaDownloadOffset => exact pinned_cudaDownloadOffset h
  | cudaFreeBuffer => exact pinned_cudaFreeBuffer h
  | cudaSync => exact pinned_cudaSync h
  | cudaLaunch => exact pinned_cudaLaunch h
  | cublasSgemv => exact pinned_cublasSgemv h
  | cublasSgemm => exact pinned_cublasSgemm h
  | cublasGemmExBf16 => exact pinned_cublasGemmExBf16 h
  | cublasSgemvOnStream => exact pinned_cublasSgemvOnStream h
  | cublasGemmStridedBatchedExBf16 => exact pinned_cublasGemmStridedBatchedExBf16 h
  | cublasPtrArray => exact pinned_cublasPtrArray h
  | cublasSgemmBatchedOnStream => exact pinned_cublasSgemmBatchedOnStream h
  | cudaLaunchNamedOnStream => exact pinned_cudaLaunchNamedOnStream h
  | cudaEventElapsedMsBits => exact pinned_cudaEventElapsedMsBits h
  | cudaUploadAsync => exact pinned_cudaUploadAsync h
  | cudaUploadOffsetAsync => exact pinned_cudaUploadOffsetAsync h
  | cudaDownloadAsync => exact pinned_cudaDownloadAsync h
  | cublasSgemmOnStream => exact pinned_cublasSgemmOnStream h
  | cudaLaunchOnStream => exact pinned_cudaLaunchOnStream h
  | cudaStreamCreate => exact pinned_cudaStreamCreate h
  | cudaStreamSync => exact pinned_cudaStreamSync h
  | cudaStreamDestroy => exact pinned_cudaStreamDestroy h
  | cudaEventCreate => exact pinned_cudaEventCreate h
  | cudaEventRecord => exact pinned_cudaEventRecord h
  | cudaStreamWaitEvent => exact pinned_cudaStreamWaitEvent h
  | cudaEventDestroy => exact pinned_cudaEventDestroy h
  | cudaGraphBeginCapture => exact pinned_cudaGraphBeginCapture h
  | cudaGraphEndCapture => exact pinned_cudaGraphEndCapture h
  | cudaGraphUpload => exact pinned_cudaGraphUpload h
  | cudaGraphLaunch => exact pinned_cudaGraphLaunch h
  | cudaGraphDestroy => exact pinned_cudaGraphDestroy h
  | cudaLaunchNamed => exact pinned_cudaLaunchNamed h
  | cudaPinnedAlloc => exact absurd rfl hf
  | cudaPinnedPtr => exact pinned_cudaPinnedPtr h
  | cudaPinnedPtrAt => exact pinned_cudaPinnedPtrAt h
  | cudaPinnedFree => exact pinned_cudaPinnedFree h
  | cudaMemInfoFree => exact pinned_cudaMemInfoFree h
  | cudaMemInfoTotal => exact pinned_cudaMemInfoTotal h
  | gpuInit => exact pinned_gpuInit h
  | gpuCleanup => exact pinned_gpuCleanup h
  | gpuCreateBuffer => exact pinned_gpuCreateBuffer h
  | gpuCreatePipeline => exact pinned_gpuCreatePipeline h
  | gpuUpload => exact pinned_gpuUpload h
  | gpuUploadPtr => exact pinned_gpuUploadPtr h
  | gpuDispatch => exact pinned_gpuDispatch h
  | gpuDownload => exact pinned_gpuDownload h
  | gpuDownloadPtr => exact pinned_gpuDownloadPtr h
  | lmdbInit => exact pinned_lmdbInit h
  | lmdbCleanup => exact pinned_lmdbCleanup h
  | lmdbOpen => exact pinned_lmdbOpen h
  | lmdbBeginWriteTxn => exact pinned_lmdbBeginWriteTxn h
  | lmdbPut => exact pinned_lmdbPut h
  | lmdbCommitWriteTxn => exact pinned_lmdbCommitWriteTxn h
  | lmdbCursorScan => exact pinned_lmdbCursorScan h
  | windowInit => exact pinned_windowInit h
  | windowCleanup => exact pinned_windowCleanup h
  | windowOpen => exact pinned_windowOpen h
  | windowPoll => exact pinned_windowPoll h
  | windowPresentGpuBuffer => exact pinned_windowPresentGpuBuffer h
  | nativeLoad => exact pinned_nativeLoad h
  | threadInit => exact pinned_threadInit h
  | threadCleanup => exact pinned_threadCleanup h
  | threadJoin => exact pinned_threadJoin h
  | nativeFree => exact pinned_nativeFree h
  | nativeArch => exact pinned_nativeArch h
  | cpuHas => exact pinned_cpuHas h
  | fileCreateDirAll => exact pinned_fileCreateDirAll h
  | memLock => exact pinned_memLock h
  | memUnlock => exact pinned_memUnlock h
  | memAdviseHuge => exact pinned_memAdviseHuge h
  | threadPriority => exact pinned_threadPriority h
  | threadStart | threadFinish => change (none : Option (Option V × World)) = some (r, w') at h; cases h
  | threadSpawn => change (none : Option (Option V × World)) = some (r, w') at h; cases h


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
    (hsp : Static.strSpan m a = some (r, off, k)) (hs : (m'.region r).size = (m.region r).size)
    (hb : ∀ i, i < k → (m'.region r).get! (off + i) = (m.region r).get! (off + i)) :
    Static.strSpan m' a = some (r, off, k) ∧ readCStr m' a = readCStr m a := by
  unfold Static.strSpan at hsp ⊢
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
    (h : Static.strSpan m a = some (r, off, k)) : off + k ≤ (m.region r).size := by
  unfold Static.strSpan at h
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
    (h : Static.strSpan m a = some (r, off, k)) : decodeAddr a = some (r, off) := by
  unfold Static.strSpan at h
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
  /-- Every value the hash table holds is at most `n` bytes. -/
  | htVals (n : Nat)
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
  deriving Repr

/-- Equality of facts, written out: the derived instance does not compute under
    `Meta.reduce` on cells, and the condition generator computes typestates. -/
def Fact.decEq : (x y : Fact) → Decidable (x = y)
  | .part p b, .part p' b' =>
      if h : p = p' ∧ b = b' then isTrue (by rw [h.1, h.2]) else isFalse (fun e => by cases e; exact h ⟨rfl, rfl⟩)
  | .room r n, .room r' n' =>
      if h : r = r' ∧ n = n' then isTrue (by rw [h.1, h.2]) else isFalse (fun e => by cases e; exact h ⟨rfl, rfl⟩)
  | .cell a v, .cell a' v' =>
      if h : a = a' ∧ v = v' then isTrue (by rw [h.1, h.2]) else isFalse (fun e => by cases e; exact h ⟨rfl, rfl⟩)
  | .held k a, .held k' a' =>
      if h : k = k' ∧ a = a' then isTrue (by rw [h.1, h.2]) else isFalse (fun e => by cases e; exact h ⟨rfl, rfl⟩)
  | .opened k a, .opened k' a' =>
      if h : k = k' ∧ a = a' then isTrue (by rw [h.1, h.2]) else isFalse (fun e => by cases e; exact h ⟨rfl, rfl⟩)
  | .htVals n, .htVals n' =>
      if h : n = n' then isTrue (by rw [h]) else isFalse (fun e => by cases e; exact h rfl)
  | .cstr a, .cstr a' =>
      if h : a = a' then isTrue (by rw [h]) else isFalse (fun e => by cases e; exact h rfl)
  | .part .., .room .. | .part .., .cell .. | .part .., .held .. | .part .., .opened ..
  | .part .., .htVals .. | .part .., .cstr .. | .room .., .part .. | .room .., .cell ..
  | .room .., .held .. | .room .., .opened .. | .room .., .htVals .. | .room .., .cstr ..
  | .cell .., .part .. | .cell .., .room .. | .cell .., .held .. | .cell .., .opened ..
  | .cell .., .htVals .. | .cell .., .cstr .. | .held .., .part .. | .held .., .room ..
  | .held .., .cell .. | .held .., .opened .. | .held .., .htVals .. | .held .., .cstr ..
  | .opened .., .part .. | .opened .., .room .. | .opened .., .cell .. | .opened .., .held ..
  | .opened .., .htVals .. | .opened .., .cstr .. | .htVals .., .part .. | .htVals .., .room ..
  | .htVals .., .cell .. | .htVals .., .held .. | .htVals .., .opened .. | .htVals .., .cstr ..
  | .cstr .., .part .. | .cstr .., .room .. | .cstr .., .cell .. | .cstr .., .held ..
  | .cstr .., .opened .. | .cstr .., .htVals ..
  | .part .., .devSeq | .part .., .oracles | .devSeq, .part .. | .oracles, .part ..
  | .room .., .devSeq | .room .., .oracles | .devSeq, .room .. | .oracles, .room ..
  | .cell .., .devSeq | .cell .., .oracles | .devSeq, .cell .. | .oracles, .cell ..
  | .held .., .devSeq | .held .., .oracles | .devSeq, .held .. | .oracles, .held ..
  | .opened .., .devSeq | .opened .., .oracles | .devSeq, .opened .. | .oracles, .opened ..
  | .htVals .., .devSeq | .htVals .., .oracles | .devSeq, .htVals .. | .oracles, .htVals ..
  | .cstr .., .devSeq | .cstr .., .oracles | .devSeq, .cstr .. | .oracles, .cstr ..
  | .devSeq, .oracles | .oracles, .devSeq
  | .part .., .cstrIn .. | .cstrIn .., .part .. | .room .., .cstrIn .. | .cstrIn .., .room ..
  | .cell .., .cstrIn .. | .cstrIn .., .cell .. | .held .., .cstrIn .. | .cstrIn .., .held ..
  | .opened .., .cstrIn .. | .cstrIn .., .opened .. | .htVals .., .cstrIn .. | .cstrIn .., .htVals ..
  | .cstr .., .cstrIn .. | .cstrIn .., .cstr .. | .devSeq, .cstrIn .. | .cstrIn .., .devSeq
  | .oracles, .cstrIn .. | .cstrIn .., .oracles
  | .part .., .roomArg .. | .roomArg .., .part .. | .room .., .roomArg .. | .roomArg .., .room ..
  | .cell .., .roomArg .. | .roomArg .., .cell .. | .held .., .roomArg .. | .roomArg .., .held ..
  | .opened .., .roomArg .. | .roomArg .., .opened .. | .htVals .., .roomArg .. | .roomArg .., .htVals ..
  | .cstr .., .roomArg .. | .roomArg .., .cstr .. | .cstrIn .., .roomArg .. | .roomArg .., .cstrIn ..
  | .devSeq, .roomArg .. | .roomArg .., .devSeq | .oracles, .roomArg .. | .roomArg .., .oracles =>
      isFalse (fun e => by cases e)
  | .part .., .devBuf .. | .devBuf .., .part .. | .room .., .devBuf .. | .devBuf .., .room ..
  | .cell .., .devBuf .. | .devBuf .., .cell .. | .held .., .devBuf .. | .devBuf .., .held ..
  | .opened .., .devBuf .. | .devBuf .., .opened .. | .htVals .., .devBuf .. | .devBuf .., .htVals ..
  | .cstr .., .devBuf .. | .devBuf .., .cstr .. | .devSeq, .devBuf .. | .devBuf .., .devSeq
  | .oracles, .devBuf .. | .devBuf .., .oracles | .cstrIn .., .devBuf .. | .devBuf .., .cstrIn ..
  | .roomArg .., .devBuf .. | .devBuf .., .roomArg .. =>
      isFalse (fun e => by cases e)
  | .part .., .pinnedUsed .. | .pinnedUsed .., .part .. | .room .., .pinnedUsed .. | .pinnedUsed .., .room ..
  | .cell .., .pinnedUsed .. | .pinnedUsed .., .cell .. | .held .., .pinnedUsed .. | .pinnedUsed .., .held ..
  | .opened .., .pinnedUsed .. | .pinnedUsed .., .opened .. | .htVals .., .pinnedUsed .. | .pinnedUsed .., .htVals ..
  | .cstr .., .pinnedUsed .. | .pinnedUsed .., .cstr .. | .devSeq, .pinnedUsed .. | .pinnedUsed .., .devSeq
  | .oracles, .pinnedUsed .. | .pinnedUsed .., .oracles | .cstrIn .., .pinnedUsed .. | .pinnedUsed .., .cstrIn ..
  | .roomArg .., .pinnedUsed .. | .pinnedUsed .., .roomArg .. | .devBuf .., .pinnedUsed .. | .pinnedUsed .., .devBuf .. =>
      isFalse (fun e => by cases e)
  | .pinnedUsed n, .pinnedUsed n' =>
      if h : n = n' then isTrue (by rw [h]) else isFalse (fun e => by cases e; exact h rfl)
  | .devSeq, .devSeq => isTrue rfl
  | .oracles, .oracles => isTrue rfl
  | .cstrIn a n, .cstrIn a' n' =>
      if h : a = a' ∧ n = n' then isTrue (by rw [h.1, h.2]) else isFalse (fun e => by cases e; exact h ⟨rfl, rfl⟩)
  | .roomArg r x, .roomArg r' x' =>
      if h : r = r' ∧ x = x' then isTrue (by rw [h.1, h.2]) else isFalse (fun e => by cases e; exact h ⟨rfl, rfl⟩)
  | .devBuf a n, .devBuf a' n' =>
      if h : a = a' ∧ n = n' then isTrue (by rw [h.1, h.2]) else isFalse (fun e => by cases e; exact h ⟨rfl, rfl⟩)

instance : DecidableEq Fact := Fact.decEq

def Fact.holds : Fact → World → Prop
  | .part p b, w => p.get w = b
  | .room r n, w => n ≤ w.mem.sizes r
  | .cell a v, w => w.mem.load a 8 = some v
  | .held k a, w => k.live w a = true
  | .opened k a, w => a ≠ 0 → k.live w a = true
  | .htVals n, w => ∀ k v, w.ht.get k = some v → v.size ≤ n
  | .cstr a, w => CStr w.mem a
  | .devSeq, w => Device.TrackOk w.dev ∧ Device.DefaultOnly w.dev.race
  | .oracles, w => Static.KeepsSize w.kernel ∧ VendorKeeps w.vendor
  | .cstrIn a n, w => w.mem.frozen = false ∧ CStr w.mem a ∧ a.toNat + n < 2 ^ 64 ∧
      ∃ r off k, r ≠ .pinned ∧ off + n < 2 ^ 36 ∧ Static.strSpan w.mem a = some (r, off, k) ∧ k ≤ n
  | .roomArg r x, w => x.toNat ≤ w.mem.sizes r ∧ x.toNat ≤ 2 ^ 36
  | .devBuf a n, w => Device.BufIs (asI32 a) n.toNat w.dev
  | .pinnedUsed n, w => w.mem.sizes .pinned ≤ n

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

/-- **A cell outside a call's frame survives it.** -/
theorem cell_kept {f : IR.Ffi} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    {a v : UInt64} (h : callBits f bits w = some (r, w')) (hf : f ≠ .fileRead) (hok : cellOk a = true)
    (hno : ((List.range 8).all fun i => !mayWrite f bits (a + UInt64.ofNat i)) = true)
    (hc : w.mem.load a 8 = some v) : w'.mem.load a 8 = some v := by
  obtain ⟨rg, off, hd, hp, hb⟩ := cellOk_spec hok
  have hz := callBits_unfrozen h (frozen_of_load hd hp hc)
  refine load_kept hd hp hb hc hz (callBits_rooms f h rg hp) fun i hi => ?_
  have hcall : callFfi f (bits.map (V.sc .i64)) w = some (r, w') := by
    unfold callFfi; rw [mapM_asBits_sc]; exact h
  have hm : mayWrite f bits (a + UInt64.ofNat i) = false := by
    have := List.all_eq_true.mp hno i (List.mem_range.mpr hi)
    simpa using this
  exact callFfi_respects_frame f _ w r w' hcall bits (mapM_asBits_sc bits) (fun e => absurd e hf) _
    (mayWrite_sound hm _)

/-- What a read of a nonzero size may write: that many bytes at its
    destination. A read of size `0` reads the whole file, which bounds
    nothing. -/
def readFrame : Frame := .atOff 0 2 4

theorem fileRead_unchanged {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .fileRead bits w = some (r, w')) (hs : bits.getD 4 0 ≠ 0) {x : UInt64}
    (hx : readFrame.mayWrite bits x = false) : Unchanged w.mem w'.mem x := by
  change ffiFileRead bits w = some (r, w') at h
  unfold ffiFileRead at h
  match bits, h, hs, hx with
  | [base, pathOff, dstOff, fileOff, size], h, hs, hx =>
    simp only [List.getD_cons_succ, List.getD_cons_zero] at hs
    simp only [readFrame, Frame.mayWrite, List.getD_cons_succ, List.getD_cons_zero, within,
      decide_eq_false_iff_not, Nat.not_lt] at hx
    simp only [Option.bind_eq_bind] at h
    cases hp : readPath w.mem (base + pathOff) with
    | none => rw [hp] at h; cases h
    | some path =>
      rw [hp, Option.bind_some] at h
      split at h
      · cases h; exact Unchanged.refl _ _
      · simp only [hs, beq_iff_eq, if_false] at h
        cases hc : copyIn w.mem (base + dstOff) _ with
        | none => rw [hc] at h; cases h
        | some m =>
          rw [hc] at h
          simp only [Option.bind_some, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨_, rfl⟩ := h
          refine Unchanged.of_eq (copyIn_other hc x fun i hi e => ?_)
          simp only [List.size_toByteArray, List.length_map, List.length_range] at hi
          subst e
          rw [UInt64.add_comm, UInt64.add_sub_cancel, UInt64.toNat_ofNat'] at hx
          have : i % 2 ^ 64 ≤ i := Nat.mod_le _ _
          have := size.toNat_lt
          omega

/-- Whether a call with these argument bits may write the byte at `x`: by
    its frame, and a read of a nonzero size by the bytes it asks for. -/
def writesAt (f : IR.Ffi) (bits : List UInt64) (x : UInt64) : Bool :=
  if f == .fileRead then bits.getD 4 0 == 0 || readFrame.mayWrite bits x else mayWrite f bits x

theorem unchanged_of_writesAt {f : IR.Ffi} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) {x : UInt64} (hx : writesAt f bits x = false) :
    Unchanged w.mem w'.mem x := by
  unfold writesAt at hx
  by_cases hf : f = .fileRead
  · subst hf
    simp only [beq_self_eq_true, if_true, Bool.or_eq_false_iff, beq_eq_false_iff_ne, ne_eq] at hx
    exact fileRead_unchanged h hx.1 hx.2
  · have hb : (f == .fileRead) = false := by simpa using hf
    rw [hb, if_neg (by decide)] at hx
    have hcall : callFfi f (bits.map (V.sc .i64)) w = some (r, w') := by
      unfold callFfi; rw [mapM_asBits_sc]; exact h
    exact callFfi_respects_frame f _ w r w' hcall bits (mapM_asBits_sc bits) (fun e => absurd e hf) _
      (mayWrite_sound hx _)

/-- `cell_kept`, for any call, a read by the bytes it asks for. -/
theorem cell_kept_at {f : IR.Ffi} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    {a v : UInt64} (h : callBits f bits w = some (r, w')) (hok : cellOk a = true)
    (hno : ((List.range 8).all fun i => !writesAt f bits (a + UInt64.ofNat i)) = true)
    (hc : w.mem.load a 8 = some v) : w'.mem.load a 8 = some v := by
  obtain ⟨rg, off, hd, hp, hb⟩ := cellOk_spec hok
  have hz := callBits_unfrozen h (frozen_of_load hd hp hc)
  refine load_kept hd hp hb hc hz (callBits_rooms f h rg hp) fun i hi => ?_
  have hm : writesAt f bits (a + UInt64.ofNat i) = false := by
    have := List.all_eq_true.mp hno i (List.mem_range.mpr hi)
    simpa using this
  exact unchanged_of_writesAt h hm

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

/-- The calls that keep a bound on the hash table's values: one that reads
    it, one that makes a table, and an insert, whose value is its length. -/
theorem ht_kept {f : IR.Ffi} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w'))
    (hf : f = .htLookup ∨ f = .htCount ∨ f = .htCreate ∨ f = .htInsert) :
    ∀ k v, w'.ht.get k = some v → w.ht.get k = some v ∨ (f = .htInsert ∧ v.size = (bits.getD 4 0).toNat) := by
  intro k v hv
  rcases hf with rfl | rfl | rfl | rfl
  · change ffiHtLookup bits w = some (r, w') at h
    unfold ffiHtLookup at h
    split at h
    · simp only [Option.bind_eq_bind] at h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · cases h; exact .inl hv
      · split at h
        · cases h; exact .inl hv
        · obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
          cases h; exact .inl hv
    · cases h
  · change ffiHtCount bits w = some (r, w') at h
    unfold ffiHtCount at h
    split at h
    · split at h <;> (cases h; exact .inl hv)
    · cases h
  · change ffiHtCreate bits w = some (r, w') at h
    unfold ffiHtCreate at h
    split at h
    · split at h <;> (cases h; exact .inl hv)
    · cases h
  · change ffiHtInsert bits w = some (r, w') at h
    unfold ffiHtInsert at h
    split at h
    · rename_i ctx kp kl vp vl
      simp only [Option.bind_eq_bind] at h
      obtain ⟨key, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨val, hval, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · cases h; exact .inl hv
      · cases h
        rcases Ht.get_set hv with h1 | rfl
        · exact .inl h1
        · exact .inr ⟨rfl, by simp [Static.copyOut_size hval]⟩
    · cases h

theorem htInit_empty {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htInit bits w = some (r, w')) : Fact.holds (.htVals 0) w' := by
  change ffiHtInit bits w = some (r, w') at h
  unfold ffiHtInit at h
  split at h
  · simp only [Option.bind_eq_bind] at h
    obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    cases h
    intro k v hv; simp [Ht.get] at hv
  · cases h

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

/-- The slot initialisers write memory only at their slot. -/
theorem slotInit_mem {f : IR.Ffi} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (hf : f = .lmdbInit ∨ f = .htInit) (h : callBits f bits w = some (r, w')) :
    ∃ slot, bits = [slot] ∧ ∃ v, w.mem.store slot 8 v = some w'.mem := by
  rcases hf with rfl | rfl
  · change ffiLmdbInit bits w = some (r, w') at h
    unfold ffiLmdbInit at h
    split at h
    · split at h
      · cases h
      · simp only [Option.bind_eq_bind] at h
        obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
        cases h; exact ⟨_, rfl, _, hm⟩
    · cases h
  · change ffiHtInit bits w = some (r, w') at h
    unfold ffiHtInit at h
    split at h
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
    Static.decodeAddr_add hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  have hb := bytes_kept hp hbytes (strSpan_bound hsp) hz hz' (hsz r hp) (fun i hi => hu i (by omega))
  obtain ⟨hsp', hrd⟩ := strSpan_congr hsp (hsz r hp) hb
  refine ⟨hz', ?_, hwrap, r, off, k, hp, hspan, hsp', hk⟩
  unfold CStr; rw [hrd]; exact hc

/-- Whether a call with these argument bits keeps a fact, whatever it answers. -/
def keeps (f : IR.Ffi) (bits : List UInt64) : Fact → Bool
  | .part p _ => !(moves f).contains p
  | .room r _ => r != .pinned
  | .cell a _ => cellOk a && (List.range 8).all fun i => !writesAt f bits (a + UInt64.ofNat i)
  | .held _ _ | .opened _ _ => false
  | .cstr a => (f == .lmdbInit || f == .htInit) && otherRegion (bits.headD 0) a
  | .htVals n => f == .htLookup || f == .htCount || f == .htCreate ||
      (f == .htInsert && decide ((bits.getD 4 0).toNat ≤ n))
  | .devSeq => Device.devUntouched.contains f || Device.seqKeepers.contains f
  | .oracles => true
  | .cstrIn a n => if f == .fileRead then bits.getD 4 0 != 0 && !readFrame.touches bits a n
      else !(frame f).touches bits a n
  | .roomArg r _ => r != .pinned
  | .devBuf _ _ => Device.devUntouched.contains f || Device.bufKeepers.contains f
  | .pinnedUsed _ => f != .cudaPinnedAlloc

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
      exact cell_kept_at h hk.1 hk.2 hx
  | held _ _ | opened _ _ => simp [keeps] at hk
  | cstr a =>
      simp only [keeps, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at hk
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
      show Static.KeepsSize w'.kernel ∧ VendorKeeps w'.vendor
      rw [e1, e2]; exact hx
  | cstrIn a n =>
      refine cstrIn_kept hx (callBits_unfrozen h hx.1) (fun rg hp => callBits_rooms f h rg hp) fun i hi => ?_
      refine unchanged_of_writesAt h ?_
      unfold writesAt
      unfold keeps at hk
      by_cases hf : f = .fileRead
      · subst hf
        simp only [beq_self_eq_true, if_true, Bool.and_eq_true, bne_iff_ne, ne_eq, Bool.not_eq_true'] at hk
        simp only [beq_self_eq_true, if_true, Bool.or_eq_false_iff, beq_eq_false_iff_ne, ne_eq]
        exact ⟨hk.1, Frame.touches_false hk.2 i hi⟩
      · have hb : (f == .fileRead) = false := by simpa using hf
        simp only [hb, if_false, Bool.false_eq_true, Bool.not_eq_true'] at hk ⊢
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
  | htVals n =>
      intro k v hv
      simp only [keeps, Bool.or_eq_true, beq_iff_eq, Bool.and_eq_true, decide_eq_true_eq] at hk
      have hf : f = .htLookup ∨ f = .htCount ∨ f = .htCreate ∨ f = .htInsert := by
        rcases hk with ((h1 | h1) | h1) | h1
        · exact .inl h1
        · exact .inr (.inl h1)
        · exact .inr (.inr (.inl h1))
        · exact .inr (.inr (.inr h1.1))
      rcases ht_kept h hf k v hv with h1 | ⟨rfl, h2⟩
      · exact hx k v h1
      · rcases hk with ((h1 | h1) | h1) | ⟨-, h3⟩
        all_goals first | cases h1 | omega

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

def Fact.isHtVals : Fact → Bool
  | .htVals _ => true
  | _ => false

/-- A bound on the hash table's values, widened to cover `b`. -/
def Fact.widened (b : Nat) : Fact → Option Fact
  | .htVals n => some (.htVals (max n b))
  | _ => none

/-- The same, keeping every other fact. -/
def Fact.widen (b : Nat) : Fact → Fact
  | .htVals n => .htVals (max n b)
  | x => x

theorem TState.holds_widen {S : TState} {w : World} (b : Nat) (hS : S.holds w) :
    TState.holds (S.map (Fact.widen b)) w := by
  intro x hx
  obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hx
  have := hS y hy
  cases y with
  | htVals n => intro k v hv; exact Nat.le_trans (this k v hv) (Nat.le_max_left _ _)
  | _ => exact this

/-- **What a call leaves of a typestate**: the facts it keeps, and what a
    lifecycle call makes: its parts, and the handle it writes to its slot. A
    cleanup whose slot the typestate knows is resolved by it. -/
def after (f : IR.Ffi) (bits : List UInt64) (S : TState) : TState :=
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
  | .htInit => .htVals 0 :: slotCell bits htCtx k
  | .htInsert => k.filter (!·.isHtVals) ++ S.filterMap (Fact.widened (bits.getD 4 0).toNat)
  | .htCleanup => slotCell bits 0 k
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

theorem stores_htInit {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htInit bits w = some (r, w')) : StoresAt htCtx bits w' := by
  change ffiHtInit bits w = some (r, w') at h
  unfold ffiHtInit at h
  split at h
  · rename_i slot
    refine ⟨slot, rfl, fun hok => ?_⟩
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    cases h
    exact cell_of_store hok hm
  · cases h

theorem stores_htCleanup {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits .htCleanup bits w = some (r, w')) : StoresAt 0 bits w' := by
  change ffiHtCleanup bits w = some (r, w') at h
  unfold ffiHtCleanup at h
  split at h
  · rename_i slot
    refine ⟨slot, rfl, fun hok => ?_⟩
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
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

/-- **What a call leaves of a typestate is true after it.** -/
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

theorem after_sound {f : IR.Ffi} {bits : List UInt64} {S : TState} {w : World} {r : Option V}
    {w' : World} (hS : S.holds w) (h : callBits f bits w = some (r, w')) : (after f bits S).holds w' := by
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
  case htInit =>
    obtain ⟨slot, rfl, hc⟩ := stores_htInit h
    exact TState.holds_cons (htInit_empty h) (slotCell_one hc hk)
  case htCleanup =>
    obtain ⟨slot, rfl, hc⟩ := stores_htCleanup h
    exact slotCell_one hc hk
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
  case htInsert =>
    change TState.holds ((S.filter (keeps .htInsert bits)).filter (!·.isHtVals) ++
      S.filterMap (Fact.widened (bits.getD 4 0).toNat)) w'
    intro x hx
    rcases List.mem_append.mp hx with hx | hx
    · exact hk x (List.mem_filter.mp hx).1
    · obtain ⟨y, hy, hw⟩ := List.mem_filterMap.mp hx
      cases y with
      | htVals n =>
          cases hw
          intro k v hv
          rcases ht_kept h (.inr (.inr (.inr rfl))) k v hv with h1 | ⟨-, h2⟩
          · exact Nat.le_trans (hS _ hy k v h1) (Nat.le_max_left _ _)
          · rw [h2]; exact Nat.le_max_right _ _
      | _ => cases hw
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
      have hd' := Static.decodeAddr_add hd (i := i) (by rw [Static.regionSpan_toNat]; omega)
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
      have hd' := Static.decodeAddr_add hd (i := i) (by rw [Static.regionSpan_toNat]; omega)
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

theorem roomAt_spec {S : TState} {w : World} {a : UInt64} {n : Nat} (hS : S.holds w)
    (h : roomAt S a n = true) : ∃ r off, decodeAddr a = some (r, off) ∧ r ≠ .pinned ∧
      off + n ≤ w.mem.sizes r ∧ w.mem.frozen = false ∧ off + n ≤ 2 ^ 36 := by
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
    | held _ _ | opened _ _ | htVals _ | cstr _ | devSeq | oracles | cstrIn _ _ | roomArg _ _ | devBuf _ _
    | pinnedUsed _ => cases hr
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

theorem readable_of_state {S : TState} {w : World} {a : UInt64} {n : Nat} (hS : S.holds w)
    (h : roomAt S a n = true) : Readable w.mem a n := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := roomAt_spec hS h
  exact copyOut_isSome hd hp hs hz hn

theorem writable_of_state {S : TState} {w : World} {a : UInt64} {n : Nat} (hS : S.holds w)
    (h : roomAt S a n = true) : Writable w.mem a n := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := roomAt_spec hS h
  intro src hsrc
  subst hsrc
  exact copyIn_isSome src hd hp hs hz hn

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

/-- Room at `r` for any value the hash table holds. -/
def htRoom (S : TState) (r : UInt64) : Bool :=
  S.any fun | .htVals n => roomAt S r n | _ => false

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

/-- **A lookup's result has room**: a bound on the table's values, and room
    for that many bytes. -/
theorem htLookup_of_state {S : TState} {w : World} {r keyPtr : UInt64} {keyLen : Nat} (hS : S.holds w)
    (h : htRoom S r = true) :
    ∀ key val, readBytes w.mem keyPtr keyLen = some key → w.ht.get key = some val → Writable w.mem r val.size := by
  intro key val _ hv
  obtain ⟨x, hx, hr⟩ := List.any_eq_true.mp h
  cases x with
  | htVals n => exact writable_of_state hS (roomAt_mono hr (hS _ hx key val hv))
  | _ => cases hr

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

/-- **A path the file shims can scan**, from the typestate: a C string it
    names, or room for the `pathMax` bytes a scan may read. -/
theorem pathOk_of_state {S : TState} {w : World} {a : UInt64} (hS : S.holds w)
    (h : (S.contains (.cstr a) || roomAt S a pathMax) = true) : PathOk w.mem a := by
  rcases Bool.or_eq_true_iff.mp h with h | h
  · exact pathOk_of_cstr (cstr_of_state hS h)
  · obtain ⟨r, off, hd, _, hs, _, _⟩ := roomAt_spec hS h
    exact pathOk_of_room hd hs

theorem pathOk_imp_of_state {S : TState} {w : World} {a : UInt64} {P : Prop} (hS : S.holds w)
    (h : (S.contains (.cstr a) || roomAt S a pathMax) = true) : P → PathOk w.mem a :=
  fun _ => pathOk_of_state hS h

/-- **A read into shared memory has room**: it asks for a length, not the whole
    file, and there is room for that length. -/
theorem fileRead_room_of_state {S : TState} {w : World} {dst fileOff size : UInt64} {p : String → Prop}
    (hS : S.holds w) (h : (size != 0 && roomAt S dst size.toNat) = true) :
    ∀ path c, p path → w.fs.get path = some c → Writable w.mem dst (readCount c fileOff size) := by
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at h
  intro _ c _ _
  refine writable_of_state hS (roomAt_mono h.2 ?_)
  unfold readCount
  simp only [beq_iff_eq, h.1, if_false]
  exact Nat.min_le_left _ _

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
  | .part _ _ | .room _ _ | .held _ _ | .opened _ _ | .htVals _ | .devSeq | .oracles | .roomArg _ _
  | .devBuf _ _ | .pinnedUsed _ => true
  | .cell c _ => cellOk c && (List.range 8).all fun i => !within a n (c + UInt64.ofNat i)
  | .cstr _ => false
  | .cstrIn c k => decide (a.toNat + n ≤ c.toNat ∨ (c.toNat + k ≤ a.toNat ∧ a.toNat + n ≤ 2 ^ 64))

def Fact.isCell : Fact → Bool
  | .cell _ _ | .cstr _ | .cstrIn _ _ => true
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
    (hht : w'.ht = w.ht) (hdv : w'.dev = w.dev ∧ w'.kernel = w.kernel ∧ w'.vendor = w.vendor)
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
  | htVals n => show ∀ k v, w'.ht.get k = some v → v.size ≤ n; rw [hht]; exact this
  | cstr _ => cases hc
  | devSeq => show Device.TrackOk w'.dev ∧ _; rw [hdv.1]; exact this
  | oracles => show Static.KeepsSize w'.kernel ∧ _; rw [hdv.2.1, hdv.2.2]; exact this
  | cstrIn _ _ => cases hc
  | roomArg r x => show x.toNat ≤ w'.mem.sizes r ∧ _; rw [hsz]; exact this
  | devBuf _ _ => show Device.BufIs _ _ w'.dev; rw [hdv.1]; exact this
  | pinnedUsed n => show w'.mem.sizes .pinned ≤ n; rw [hsz]; exact this

theorem storeKeeps_sound {S : TState} {w w' : World} {a b : UInt64} {n : Nat} (hS : S.holds w)
    (hp : ∀ p : Part, p.get w' = p.get w) (hh : ∀ (k : Held) (a : UInt64), k.live w' a = k.live w a)
    (hht : w'.ht = w.ht) (hdv : w'.dev = w.dev ∧ w'.kernel = w.kernel ∧ w'.vendor = w.vendor)
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
  | htVals n => show ∀ k v, w'.ht.get k = some v → v.size ≤ n; rw [hht]; exact hx0
  | cstr _ => simp [storeKeeps] at hk
  | devSeq => show Device.TrackOk w'.dev ∧ _; rw [hdv.1]; exact hx0
  | roomArg r x => show x.toNat ≤ w'.mem.sizes r ∧ _; rw [hsz]; exact hx0
  | devBuf _ _ => show Device.BufIs _ _ w'.dev; rw [hdv.1]; exact hx0
  | pinnedUsed n => show w'.mem.sizes .pinned ≤ n; rw [hsz]; exact hx0
  | oracles => show Static.KeepsSize w'.kernel ∧ _; rw [hdv.2.1, hdv.2.2]; exact hx0
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
      obtain ⟨rg, off, hd, hpin, hb⟩ := cellOk_spec hk.1
      have hz' : w'.mem.frozen = false := by
        have := hp .frozen; simp only [Part.get] at this; rw [this]; exact frozen_of_load hd hpin hx0
      refine load_kept hd hpin hb hx0 hz' (congrFun hsz rg) fun i hi => Unchanged.of_eq ?_
      refine Mem.store_other h _ fun j hj e => ?_
      have := within_of hn ⟨j, hj, e⟩
      rw [hk.2 i hi] at this; cases this

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
  | _ => rfl

theorem afterStore_sound {S : TState} {w w' : World} {ao vo : Option UInt64} {a b : UInt64} {n : Nat}
    (hS : S.holds w) (hp : ∀ p : Part, p.get w' = p.get w) (hh : ∀ (k : Held) (a : UInt64), k.live w' a = k.live w a)
    (hht : w'.ht = w.ht) (hdv : w'.dev = w.dev ∧ w'.kernel = w.kernel ∧ w'.vendor = w.vendor)
    (hsz : w'.mem.sizes = w.mem.sizes)
    (h : ao.isSome → w.mem.store a n b = some w'.mem) (ha : ∀ x, ao = some x → x = a)
    (hv : ∀ x, vo = some x → x = b) (hn : n ≤ 2 ^ 64) : (afterStore ao vo n S).holds w' := by
  cases ao with
  | none => exact nonCell_kept hS hp hh hht hdv hsz
  | some a' =>
      obtain rfl := ha a' rfl
      have hst := h rfl
      have hk := storeKeeps_sound hS hp hh hht hdv hsz hst hn
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
    m'.sizes = m.sizes ∧ m'.frozen = m.frozen) :
    ∀ (l : List α) (m m' : Mem), l.foldlM (fun mm x => f x mm) m = some m' →
      m'.sizes = m.sizes ∧ m'.frozen = m.frozen
  | [], m, m', h => by cases h; exact ⟨rfl, rfl⟩
  | x :: l, m, m', h => by
      simp only [List.foldlM_cons, Option.bind_eq_bind] at h
      obtain ⟨m1, h1, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨e1, f1⟩ := hf x m m1 h1
      obtain ⟨e2, f2⟩ := foldl_stores_sizes f hf l m1 m' h
      exact ⟨e2.trans e1, f2.trans f1⟩

set_option maxHeartbeats 2000000 in
/-- Every entry point is found by its own C name. -/
theorem ofCname_cname (f : IR.Ffi) : IR.Ffi.ofCname f.cname = some f := by
  cases f <;> decide

/-- **A call is its contract.** Outside a worker's lifetime, a call to any entry
    point but `threadSpawn` is `callBits` on its argument bits. -/
theorem callOf_ffi {f : IR.Ffi} (hf : f ≠ .threadSpawn) (lc : Locals) (vs : List V) {w : World}
    (hz : w.mem.frozen = false) :
    callOf lc (.ffi f) vs w = (vs.mapM asBits).bind (fun bits => callBits f bits w) := by
  have hs : (f == .threadSpawn) = false := by simpa using hf
  simp only [callOf, hz, hs, Bool.false_and, Bool.false_eq_true, if_false, callImport, ofCname_cname,
    bind, Option.bind, callFfi]

/-- **What a call blind to data reads**: the memory it must know to decide what
    to do (`need`), and what stays known after it (`post`). A call with such a
    description answers alike on every pair of worlds that agree on everything
    but data and on the bytes it needs; `Host.Static` checks programs by it. -/
structure Blind where
  need : List UInt64 → Static.Known → Mem → Bool
  post : List UInt64 → Static.Known → Static.Known

/-- The entries blind to data, and what they read. -/
def blind (f : IR.Ffi) : Option Blind :=
  if Static.supported f then some ⟨Static.need f, Static.post f⟩ else none

theorem blind_some {f : IR.Ffi} {b : Blind} (h : blind f = some b) :
    Static.supported f = true ∧ b = ⟨Static.need f, Static.post f⟩ := by
  unfold blind at h
  split at h
  · rename_i hs; cases h; exact ⟨hs, rfl⟩
  · cases h

/-- **Everything known about an entry point**, in one value: its ABI, the
    memory it may write and what it computes (`Ffi.spec`), when it answers,
    which lifecycle parts it may move, what it leaves of a typestate, and what
    it reads when it cannot tell data apart. -/
structure Contract where
  spec : Spec
  pre : List UInt64 → World → Prop
  moves : List Part
  /-- What a call with these argument bits leaves of a typestate. -/
  after : List UInt64 → TState → TState
  /-- What it reads, when it cannot tell data apart. -/
  blind : Option Blind

/-- **The contract table**, read once from `callBits`. -/
def _root_.AlgorithmLib.IR.Ffi.contract (f : IR.Ffi) : Contract where
  spec := f.spec
  pre := Pre f
  moves := moves f
  after := after f
  blind := blind f

/-- What it means for a table entry to be true of `callBits`. -/
structure Contract.Sound (f : IR.Ffi) : Prop where
  /-- Under its precondition the call answers: it is not misuse. -/
  answers : ∀ bits w, f.contract.pre bits w → (callBits f bits w).isSome = true
  /-- An answer keeps every part the entry does not move. -/
  keeps : ∀ bits w r w', callBits f bits w = some (r, w') →
    ∀ p, p ∉ f.contract.moves → p.get w' = p.get w
  /-- An answer lands in what the entry says it leaves of any typestate. -/
  leaves : ∀ S bits w r w', S.holds w → callBits f bits w = some (r, w') → (f.contract.after bits S).holds w'
  /-- An answer changes no byte outside the entry's memory frame; for
      `fileRead`, whose frame is its answer, of files shorter than `2^64` bytes. -/
  writes : ∀ args w ret w', callFfi f args w = some (ret, w') → ∀ bits, args.mapM asBits = some bits →
    (f = .fileRead → ∀ p c, w.fs.get p = some c → c.size < 2 ^ 64) →
    ∀ a, ¬ f.contract.spec.frame.allows bits (ret.bind asBits) a → Unchanged w.mem w'.mem a
  /-- An entry blind to data answers alike on worlds that agree up to data and
      on what it needs, and leaves them agreeing on what it says stays known. -/
  blind : ∀ b, f.contract.blind = some b → ∀ K w₁ w₂ bits, Static.Same K w₁ w₂ →
    b.need bits K w₂.mem = true → Static.ResSame (b.post bits K) (callBits f bits w₁) (callBits f bits w₂)

/-- **The table is true of the model**, entry by entry. -/
theorem contract_sound (f : IR.Ffi) : Contract.Sound f where
  answers := pre_safe f
  keeps := fun _ _ _ _ h => callBits_parts f h
  leaves := fun _ _ _ _ _ hS h => after_sound hS h
  writes := fun args w ret w' h bits hb hfs a hout => callFfi_respects_frame f args w ret w' h bits hb hfs a hout
  blind := fun b hb _ _ _ bits hs hn => by
    obtain ⟨hsup, rfl⟩ := blind_some hb
    exact Static.callBits_cong hs f hsup bits hn

-- ---------------------------------------------------------------------------
-- The device, from the typestate
-- ---------------------------------------------------------------------------

/-- **Every buffer is ready for the default stream** where every access is its
    own. -/
theorem ready_of_state {S : TState} {w : World} (hS : S.holds w) (h : S.contains .devSeq = true)
    (b : Nat) (write : Bool) : Ready w.dev.race defaultParty b write := by
  obtain ⟨⟨⟨evs, hr, -⟩, -⟩, hd⟩ := holds_of_mem (x := .devSeq) hS h
  exact Device.ready_default hr.ownBelow hd b write

theorem keepsSize_of_state {S : TState} {w : World} (hS : S.holds w) (h : S.contains .oracles = true) :
    Static.KeepsSize w.kernel := (holds_of_mem (x := .oracles) hS h).1

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
  have hd4 : decodeAddr (a + 4) = some (r, off + 4) := Static.decodeAddr_add (i := 4) hd hi
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
  ⟨fun b _ => (ready_of_state hS h b true).1, fun b _ => (ready_of_state hS h b true).2 rfl⟩

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
    have hd' := Static.decodeAddr_add (i := 4 * i) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
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
  have hd1 := Static.decodeAddr_add (i := 8 * i) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  have hd2 := Static.decodeAddr_add (i := 8 * i + 4) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp (load_isSome (n := 4) hd1 hp (by omega) hz)
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp (load_isSome (n := 4) hd2 hp (by omega) hz)
  rw [ha, hb]; rfl

/-- **A launch's bindings are ready** where they can be read and every access
    is the default stream's. -/
theorem bindsReady_of_state {S : TState} {w : World} {nBufs bindPtr : UInt64} (hS : S.holds w)
    (hq : S.contains .devSeq = true) (h : roomAt S bindPtr (4 * (asI32 nBufs).toNat) = true) :
    BindsReady w defaultParty nBufs bindPtr := by
  obtain ⟨ids, hids⟩ := Option.isSome_iff_exists.mp (readIds_of_state hS h)
  exact ⟨ids, hids, fun id _ _ => ready_of_state hS hq _ _⟩

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
      have := Static.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; have := hx.2; omega)
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
      have := Static.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; have := hx.2; omega)
      rwa [UInt64.ofNat_toNat] at this
    exact writable_of_roomArg hS hm hz hd' hp (by omega)

end AlgorithmLib.HProg.Contracts

