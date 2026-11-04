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
# `Host.LifecycleCore` — the parts of a world a typestate names

The lifecycle parts (`Part`): whether the CUDA context is live or capturing,
whether an asynchronous copy is in flight, and the
oracles no call changes (a display, a CUDA device, a wgpu adapter, a library
loaded). The kinds of handle a program may hold open (`Held`). And the facts
about the device's and memory's steps that the keeping proofs of the C
libraries' rows (`Host.ExtLife`) read: what keeps a context live, what keeps
the size of a region.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec
  AlgorithmLib.Device

namespace AlgorithmLib.HProg.Contracts

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
    CUDA context is live and capturing. -/
inductive Part where
  | cuda | capturing
  /-- There is a display to open windows on: an oracle no call changes. -/
  | display
  /-- There is a CUDA device: an oracle no call changes. -/
  | cudaDevice
  /-- wgpu finds an adapter: an oracle no call changes. -/
  | gpuAdapter
  /-- No asynchronous copy is in flight over page-locked memory. -/
  | idle
  /-- Library `l` loaded: an oracle no call changes, and what a theorem states
      of the platform it needs. -/
  | has (l : Lib)
  deriving DecidableEq, Repr

def Part.get : Part → World → Bool
  | .cuda, w => w.dev.live
  | .capturing, w => w.dev.capture.isSome
  | .display, w => w.display
  | .cudaDevice, w => w.cudaDevice
  | .gpuAdapter, w => w.gpuAdapter
  | .idle, w => w.mem.busy.isEmpty
  | .has l, w => w.has l

/-- **The kinds of handle a library hands a program**, which a typestate
    may say it holds: an open window, serial port or USB device. -/
inductive Held where
  | window | port | usbDevice | usbInterface | heap
  /-- The one live wgpu object of a kind made afresh each time round a loop
      (`WgState.isOne`), or any value once the device is lost. -/
  | wg (k : IR.WgObj)
  deriving DecidableEq, Repr

/-- Whether `a` is a handle of kind `k` the program holds open; for the heap,
    the start of a live allocation. -/
def Held.live : Held → World → UInt64 → Bool
  | .window, w, a => (w.wlWin? a).isSome
  | .port, w, a => (w.serPort? a).isSome
  | .usbDevice, w, a => (w.usbDev? a).isSome
  | .usbInterface, w, a => (w.usbIface? a).isSome
  | .heap, w, a => (w.heapAt? a).isSome
  | .wg k, w, a => w.wg.lost || match wgIndex a with
    | some i => w.wg.isOne k i
    | none => false

-- ---------------------------------------------------------------------------
-- What the calls leave of memory's layout
-- ---------------------------------------------------------------------------

theorem copyIn_sizes {m m' : Mem} {a : UInt64} {src : ByteArray} (h : copyIn m a src = some m')
    (r : Region) : (m'.region r).size = (m.region r).size := ((copyIn_same h).size r).symm

theorem copyIn_sizes_eq {m m' : Mem} {a : UInt64} {src : ByteArray} (h : copyIn m a src = some m') :
    m'.sizes = m.sizes := funext (copyIn_sizes h)

theorem retire_sizes (m : Mem) (c : Clock) : (m.retire c).sizes = m.sizes :=
  funext fun r => by cases r <;> rfl

theorem forParty_sizes (m : Mem) (p : Nat) : (m.forParty p).sizes = m.sizes :=
  funext fun r => by cases r <;> rfl

theorem sizes_with_live (m : Mem) (L : List (Nat × Nat)) : ({ m with pinnedLive := L } : Mem).sizes = m.sizes :=
  funext fun r => by cases r <;> rfl

theorem sizes_with_busy (m : Mem) (B : List Busy) : ({ m with busy := B } : Mem).sizes = m.sizes :=
  funext fun r => by cases r <;> rfl

/-- A step from `w` to `w'` kept the size of every region but the pinned
    one, which pinned allocation grows. -/
def KeepsRooms (w w' : World) : Prop := ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r

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

end AlgorithmLib.HProg.Contracts
