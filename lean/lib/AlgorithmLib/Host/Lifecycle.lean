module
public import AlgorithmLib.Host.LifecycleCore
meta import AlgorithmLib.Host.LifecycleCore
public import AlgorithmLib.Host.KeepsCuda
meta import AlgorithmLib.Host.KeepsCuda
public import AlgorithmLib.Host.KeepsBlas
meta import AlgorithmLib.Host.KeepsBlas
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! `Host.Lifecycle`: what each entry point keeps (`callBits_keeps`), and the lifecycle facts that follow. -/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec

namespace AlgorithmLib.HProg.Contracts

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

/-- **Every call leaves page-locked memory the size it was.** -/
theorem callBits_pinned (f : IR.Ffi) {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) : KeepsPinned w w' :=
  (callBits_keeps f h).pinned

/-- **An entry point's answer is a scalar of the type its signature gives.** -/
theorem callBits_scalar (f : IR.Ffi) {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits f bits w = some (some v, w')) : ∃ x, v = .sc (f.result.getD .i64) x :=
  (callBits_keeps f h).scalar v rfl

-- ---------------------------------------------------------------------------
-- What the movers leave
-- ---------------------------------------------------------------------------

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
  /-- The driver is initialised: `cuInit` has answered. -/
  | drvInit
  /-- The primary context is retained `n` times. -/
  | retained (n : Nat)
  /-- The device has made `n` allocations, live or freed: the next one is
      `devPtr n`. -/
  | nbufs (n : Nat)
  /-- Allocation `id` is live and holds `n` bytes: it starts at `devPtr id`. -/
  | alloc (id n : Nat)
  /-- Memory at `a` holds a C string within `n` bytes, `s` where it is
      known: what a step that leaves those bytes alone keeps. Where the string
      is a module's PTX, `sigs` names entry points it declares with their
      parameters' sizes, so a lookup reads them here and never the text. -/
  | text (a : UInt64) (s : Option String) (n : Nat) (sigs : List (String × List Nat))
  /-- The driver has loaded `n` modules, live or unloaded: the next is
      `drvHandle kModule n`. -/
  | nmods (n : Nat)
  /-- The driver has looked up `n` functions: the next is `drvHandle kFunc n`. -/
  | nfuncs (n : Nat)
  /-- Module `i` is loaded, and declares each of `sigs`'s entry points with
      parameters of those sizes. -/
  | mod (i : Nat) (sigs : List (String × List Nat))
  /-- Function `i` is an entry point of a loaded module whose declaration
      takes parameters of these sizes. -/
  | fn (i : Nat) (sizes : List Nat)
  /-- cuBLAS has made `n` handles: the next is `drvHandle kBlas n`. -/
  | nblas (n : Nat)
  /-- cuBLAS handle `i` is live and runs its routines on stream `s`. -/
  | blasOn (i : Nat) (s : UInt64)
  /-- The driver has captured `n` graphs, live or destroyed: the next is
      `drvHandle kGraph n`. -/
  | ngraphs (n : Nat)
  /-- Graph `i` is live, and every allocation its nodes bind is live. -/
  | graph (i : Nat)
  /-- The driver has instantiated `n` graphs, live or destroyed: the next is
      `drvHandle kExec n`. -/
  | nexecs (n : Nat)
  /-- Instantiated graph `i` is live, and every allocation its nodes bind is
      live. -/
  | exec (i : Nat)
  /-- A capture is in progress, begun on party `q`, joined by the streams
      `j`, with the ordering among what it records `r`; each event of `evs`
      is recorded inside it at its clock, and every allocation its nodes
      bind is live. -/
  | cap (q : Nat) (j : List Nat) (r : Race) (evs : List (Nat × Clock))
  /-- wgpu has made `n` objects, live or released: the next is `wgHandle n`. -/
  | nwg (n : Nat)
  /-- wgpu object `i` is live, of shape `s`, or the device is lost. -/
  | wgo (i : Nat) (s : WgShape)
  /-- The program holds no live wgpu object of kind `k` (`none`), or at most
      one, of shape `s`; or the device is lost. -/
  | wg1 (k : IR.WgObj) (s : Option WgShape)
  /-- The `n` bytes at `a` are UTF-8, and where `shader` is set, a WGSL source
      the device accepts: what a step that leaves those bytes alone keeps. -/
  | wgText (a : UInt64) (n : Nat) (shader : Bool)
  deriving Repr, DecidableEq

/-- A kernel launch whose allocations are live, or a vendor routine whose
    inputs are live and whose output is its last input. -/
abbrev OpLive (d : Dev) (op : DevOp) : Prop :=
  (∃ l ids, op = .launch l ids ∧ ∀ j ∈ ids, ((d.bufs[j]?).join).isSome = true) ∨
  (∃ c ins out, op = .vendor c ins out ∧ ins.getLast? = some out ∧
    ∀ j ∈ ins, ((d.bufs[j]?).join).isSome = true)

/-- Every operation of `ops` is one whose allocations are live (`OpLive`). -/
def OpsLive (d : Dev) (ops : Array DevOp) : Prop := ∀ op ∈ ops.toList, OpLive d op

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
  | .drvInit, w => w.dev.drvInit = true
  | .retained n, w => w.dev.retained = n
  | .nbufs n, w => w.dev.bufs.size = n
  | .alloc id n, w => ∃ b, (w.dev.bufs[id]?).join = some b ∧ b.size = n
  | .text a s n sigs, w => (w.mem.frozen = false ∧ CStr w.mem a ∧ a.toNat + n < 2 ^ 64 ∧
      ∃ r off k, r ≠ .pinned ∧ off + n < 2 ^ 36 ∧ Sem.strSpan w.mem a = some (r, off, k) ∧ k ≤ n) ∧
      ∃ t, readCStrAt w.mem a = some t ∧ (∀ x, s = some x → x = t) ∧ ∀ p ∈ sigs, ptxParamSizes t p.1 = some p.2
  | .nmods n, w => w.dev.modules.size = n
  | .nfuncs n, w => w.dev.funcs.size = n
  | .mod i sigs, w => ∃ s, w.dev.modules[i]? = some (some s) ∧ ∀ p ∈ sigs, ptxParamSizes s p.1 = some p.2
  | .fn i sizes, w => ∃ mi e s, w.dev.funcs[i]? = some (mi, e) ∧ w.dev.modules[mi]? = some (some s) ∧
      ptxParamSizes s e = some sizes
  | .nblas n, w => w.dev.blas.size = n
  | .blasOn i s, w => w.dev.blas[i]? = some (some s)
  | .ngraphs n, w => w.dev.graphs.size = n
  | .graph i, w => ∃ ops, w.dev.graphs[i]? = some (some ops) ∧ OpsLive w.dev ops
  | .nexecs n, w => w.dev.execs.size = n
  | .exec i, w => ∃ ops, w.dev.execs[i]? = some (some ops) ∧ OpsLive w.dev ops
  | .cap q j r evs, w => ∃ c, w.dev.capture = some c ∧ c.origin = q ∧ c.joined = j ∧ c.race = r ∧
      OpsLive w.dev c.ops ∧ ∀ e ∈ evs, w.dev.event? e.1 = some (some (e.2, true))
  | .nwg n, w => w.wg.objs.size = n
  | .wgo i s, w => w.wg.lost = true ∨ ∃ d, w.wg.objs[i]? = some (true, d) ∧ w.wg.Fits d s
  | .wg1 k none, w => w.wg.lost = true ∨ w.wg.NoneOf k
  | .wg1 k (some s), w => w.wg.lost = true ∨ ∀ (i : Nat) (d : WgData), w.wg.objs[i]? = some (true, d) →
      d.kind = k → w.wg.isOne k i = true ∧ w.wg.Fits d s
  | .wgText a n sh, w => w.mem.frozen = false ∧ a.toNat + n < 2 ^ 64 ∧
      (∃ r off, decodeAddr a = some (r, off) ∧ r ≠ .pinned ∧ off + n ≤ w.mem.sizes r ∧ off + n ≤ 2 ^ 36) ∧
      ∃ t, Sem.wgText w.mem a (UInt64.ofNat n) = some t ∧ (sh = true → w.wgslOk t = true)

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

/-- A fact about the driver's own state: what `cudaAfter` states of a driver
    call, and what any step that leaves the device alone keeps. -/
def Fact.isDrv : Fact → Bool
  | .drvInit | .retained _ | .nbufs _ | .alloc _ _ | .nmods _ | .nfuncs _ | .mod _ _ | .fn _ _ | .nblas _ | .blasOn _ _ | .ngraphs _ | .graph _ | .nexecs _ | .exec _ | .cap _ _ _ _ => true
  | _ => false

/-- A driver fact survives a step that leaves the device as it was. -/
theorem Fact.holds_drv {x : Fact} (hx : x.isDrv = true) {w w' : World} (h : w'.dev = w.dev) (hw : x.holds w) :
    x.holds w' := by
  cases x <;> simp only [Fact.isDrv, Bool.false_eq_true] at hx
  all_goals simp only [Fact.holds] at hw ⊢; rw [h]; exact hw

/-- A fact about wgpu's objects: what any step that leaves them alone keeps. -/
def Fact.isWg : Fact → Bool
  | .nwg _ | .wgo _ _ | .wg1 _ _ => true
  | _ => false

/-- A wgpu fact survives a step that leaves wgpu's objects as they were. -/
theorem Fact.holds_wg {x : Fact} (hx : x.isWg = true) {w w' : World} (h : w'.wg = w.wg) (hw : x.holds w) :
    x.holds w' := by
  cases x with
  | nwg n => show _ = _; rw [h]; exact hw
  | wgo i s => simp only [Fact.holds] at hw ⊢; rw [h]; exact hw
  | wg1 k s => cases s <;> (simp only [Fact.holds] at hw ⊢; rw [h]; exact hw)
  | _ => simp [Fact.isWg] at hx

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

/-- A C string's bytes left alone leave what it reads. -/
theorem text_kept {w w' : World} {a : UInt64} {s : Option String} {n : Nat} {sg : List (String × List Nat)}
    (hx : (Fact.text a s n sg).holds w)
    (hz' : w'.mem.frozen = false) (hsz : ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r)
    (hu : ∀ i, i < n → Unchanged w.mem w'.mem (a + UInt64.ofNat i)) : (Fact.text a s n sg).holds w' := by
  obtain ⟨⟨hz, hc, hwrap, r, off, k, hp, hspan, hsp, hk⟩, t, hs, hst, hsg⟩ := hx
  have hd := strSpan_decode hsp
  have hbytes : ∀ i, i < k → decodeAddr (a + UInt64.ofNat i) = some (r, off + i) := fun i hi =>
    Sem.decodeAddr_add hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  have hb := bytes_kept hp hbytes (strSpan_bound hsp) hz hz' (hsz r hp) (fun i hi => hu i (by omega))
  obtain ⟨hsp', hrd⟩ := strSpan_congr hsp (hsz r hp) hb
  refine ⟨⟨hz', ?_, hwrap, r, off, k, hp, hspan, hsp', hk⟩, t, ?_, hst, hsg⟩
  · unfold CStr; rw [hrd]; exact hc
  · show readCStr w'.mem a = some t; rw [hrd]; exact hs

/-- Whether a call with these argument bits keeps a fact, whatever it answers. -/
def keeps (f : IR.Ffi) (bits : List UInt64) : Fact → Bool
  | .part p _ => !(moves f).contains p
  | .room r _ => r != .pinned
  | .cell a _ => cellOk a && (List.range 8).all fun i => !mayWrite f bits (a + UInt64.ofNat i)
  | .held _ _ | .opened _ _ => false
  | .cstr a => f == .lmdbInit && otherRegion (bits.headD 0) a
  | .htAt _ _ _ => false
  | .devSeq => Device.devUntouched.contains f
  | .oracles => true
  | .cstrIn a n => !(frame f).touches bits a n
  | .roomArg r _ => r != .pinned
  | .devBuf _ _ => Device.devUntouched.contains f
  | .pinnedUsed _ => true
  | .streams _ | .events _ | .evLive _ | .readyAll _ | .evAll _ => Device.devUntouched.contains f
  | .drvInit | .retained _ | .nbufs _ | .alloc _ _ | .nmods _ | .nfuncs _ | .mod _ _ | .fn _ _ | .nblas _ | .blasOn _ _ | .ngraphs _ | .graph _ | .nexecs _ | .exec _ | .cap _ _ _ _ | .text _ _ _ _ => false
  | .nwg _ | .wgo _ _ | .wg1 _ _ | .wgText _ _ _ => false

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
      simp only [keeps, List.contains_iff_mem] at hk
      rw [Device.callBits_dev_same f hk h]; exact hd
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
      simp only [keeps, List.contains_iff_mem] at hk
      show Device.BufIs _ _ w'.dev; rw [Device.callBits_dev_same f hk h]; exact hx
  | pinnedUsed n =>
      show w'.mem.sizes .pinned ≤ n
      rw [callBits_pinned f h]; exact hx
  | streams _ | events _ | evLive _ | readyAll _ | evAll _ =>
      simp only [keeps, List.contains_iff_mem] at hk
      exact Fact.holds_dev rfl (Device.callBits_dev_same f hk h) hx
  | htAt _ _ _ => simp [keeps] at hk
  | drvInit | retained _ | nbufs _ | alloc _ _ | nmods _ | nfuncs _ | mod _ _ | fn _ _ | nblas _ | blasOn _ _ | ngraphs _ | graph _ | nexecs _ | exec _ | cap _ _ _ _ | text _ _ _ _
  | nwg _ | wgo _ _ | wg1 _ _ | wgText _ _ _ => simp [keeps] at hk

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
  | .lmdbInit => .part .lmdb true :: .part .lmdbEmpty true :: slotCell bits lmdbCtx k
  | .threadInit => .part .thread true :: slotCell bits threadCtx k
  | .lmdbCleanup =>
      if S.contains (.cell slot 0) then S
      else if S.contains (.cell slot lmdbCtx) then .part .lmdb false :: slotCell bits 0 k else k
  | .threadCleanup =>
      .part .frozen false ::
        if S.contains (.cell slot threadCtx) && cellOk slot then .part .thread false :: .cell slot 0 :: k
        else k
  | .threadJoin => if S.contains (.part .frozen false) then .part .frozen false :: k else k
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

/-- **What `afterLife` says of a typestate is true after the call.** -/
theorem afterLife_sound {f : IR.Ffi} {bits : List UInt64} {S : TState} {w : World} {r : Option V}
    {w' : World} (hS : S.holds w) (h : callBits f bits w = some (r, w')) : (afterLife f bits S).holds w' := by
  have hk := kept_sound hS h
  cases f
  case lmdbInit =>
    obtain ⟨slot, rfl, hc⟩ := stores_lmdbInit h
    exact TState.holds_cons (sets_lmdbInit h) (TState.holds_cons (empties_lmdbInit h) (slotCell_one hc hk))
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

/-- **What a call leaves of a typestate**: what `afterLife` says. -/
def after (f : IR.Ffi) (bits : List UInt64) (S : TState) : TState := afterLife f bits S

/-- **What a call leaves of a typestate is true after it.** -/
theorem after_sound {f : IR.Ffi} {bits : List UInt64} {S : TState} {w : World} {r : Option V}
    {w' : World} (hS : S.holds w) (h : callBits f bits w = some (r, w')) : (after f bits S).holds w' :=
  afterLife_sound hS h

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

/-- Memory whose bytes from `a` load as they did copies out the same. -/
theorem copyOut_congr {m m' : Mem} {a : UInt64} {n : Nat}
    (h : ∀ i, i < n → m'.load (a + UInt64.ofNat i) 1 = m.load (a + UInt64.ofNat i) 1) :
    copyOut m' a n = copyOut m a n := by
  unfold copyOut
  suffices ∀ (l : List Nat) (acc : ByteArray), (∀ i ∈ l, i < n) →
      l.foldlM (fun (acc : ByteArray) i => do
        let b ← m'.load (a + UInt64.ofNat i) 1
        pure (acc.push b.toUInt8)) acc =
      l.foldlM (fun (acc : ByteArray) i => do
        let b ← m.load (a + UInt64.ofNat i) 1
        pure (acc.push b.toUInt8)) acc from
    this _ _ (fun i hi => List.mem_range.mp hi)
  intro l
  induction l with
  | nil => intro _ _; rfl
  | cons i l ih =>
      intro acc hl
      simp only [List.foldlM_cons, h i (hl i List.mem_cons_self)]
      cases m.load (a + UInt64.ofNat i) 1 with
      | none => rfl
      | some b => exact ih _ (fun j hj => hl j (List.mem_cons_of_mem _ hj))

theorem wgText_kept {w w' : World} {a : UInt64} {n : Nat} {sh : Bool}
    (hx : (Fact.wgText a n sh).holds w)
    (hz' : w'.mem.frozen = false) (hsz : ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r)
    (hok : w'.wgslOk = w.wgslOk)
    (hu : ∀ i, i < n → Unchanged w.mem w'.mem (a + UInt64.ofNat i)) : (Fact.wgText a n sh).holds w' := by
  obtain ⟨hz, hw, ⟨r, off, hd, hp, hs, hsp⟩, t, ht, hok'⟩ := hx
  refine ⟨hz', hw, ⟨r, off, hd, hp, by rw [hsz r hp]; exact hs, hsp⟩, t, ?_, by rw [hok]; exact hok'⟩
  have hn : (UInt64.ofNat n).toNat = n := by rw [UInt64.toNat_ofNat']; exact Nat.mod_eq_of_lt (by omega)
  have hc : copyOut w'.mem a n = copyOut w.mem a n := copyOut_congr fun i hi => by
    have hd' := Sem.decodeAddr_add hd (i := i) (by rw [Sem.regionSpan_toNat]; omega)
    obtain ⟨x, hx⟩ := Option.isSome_iff_exists.mp (load_isSome (n := 1) hd' hp (by omega) hz)
    obtain ⟨y, hy⟩ := Option.isSome_iff_exists.mp
      (load_isSome (n := 1) hd' hp (by rw [hsz r hp]; omega) hz')
    rw [hx, hy, hu i hi x y hx hy]
  unfold Sem.wgText readBytes at ht ⊢
  rw [hn] at ht ⊢
  rw [hc]; exact ht

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
    | pinnedUsed _ | streams _ | events _ | evLive _ | readyAll _ | evAll _ | drvInit | retained _ | nbufs _ | alloc _ _ | nmods _ | nfuncs _ | mod _ _ | fn _ _ | nblas _ | blasOn _ _ | ngraphs _ | graph _ | nexecs _ | exec _ | cap _ _ _ _ | text _ _ _ _
    | nwg _ | wgo _ _ | wg1 _ _ | wgText _ _ _ => cases hr
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
  | .devBuf _ _ | .pinnedUsed _ | .streams _ | .events _ | .evLive _ | .readyAll _ | .evAll _
  | .drvInit | .retained _ | .nbufs _ | .alloc _ _ | .nmods _ | .nfuncs _ | .mod _ _ | .fn _ _ | .nblas _ | .blasOn _ _ | .ngraphs _ | .graph _ | .nexecs _ | .exec _ | .cap _ _ _ _ => true
  | .nwg _ | .wgo _ _ | .wg1 _ _ => true
  | .text c _ k _ | .wgText c k _ => decide (a.toNat + n ≤ c.toNat ∨ (c.toNat + k ≤ a.toNat ∧ a.toNat + n ≤ 2 ^ 64))
  | .cell c _ => cellOk c && (List.range 8).all fun i => !within a n (c + UInt64.ofNat i)
  | .cstr _ => false
  | .cstrIn c k => decide (a.toNat + n ≤ c.toNat ∨ (c.toNat + k ≤ a.toNat ∧ a.toNat + n ≤ 2 ^ 64))
  | .htAt c _ _ => decide (a.toNat < (regionBase .pinned).toNat) && cellOk c &&
      (List.range 8).all fun i => !within a n (c + UInt64.ofNat i)

def Fact.isCell : Fact → Bool
  | .cell _ _ | .cstr _ | .cstrIn _ _ | .htAt _ _ _ | .text _ _ _ _ | .wgText _ _ _ => true
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
    (hdv : w'.dev = w.dev ∧ w'.kernel = w.kernel ∧ w'.vendor = w.vendor ∧ w'.wg = w.wg ∧ w'.wgslOk = w.wgslOk)
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
  | oracles => show Sem.KeepsSize w'.kernel ∧ _; rw [hdv.2.1, hdv.2.2.1]; exact this
  | cstrIn _ _ => cases hc
  | text _ _ _ _ => cases hc
  | roomArg r x => show x.toNat ≤ w'.mem.sizes r ∧ _; rw [hsz]; exact this
  | devBuf _ _ => show Device.BufIs _ _ w'.dev; rw [hdv.1]; exact this
  | pinnedUsed n => show w'.mem.sizes .pinned ≤ n; rw [hsz]; exact this
  | streams _ | events _ | evLive _ | readyAll _ | evAll _ => exact Fact.holds_dev rfl hdv.1 this
  | drvInit | retained _ | nbufs _ | alloc _ _ | nmods _ | nfuncs _ | mod _ _ | fn _ _ | nblas _ | blasOn _ _ | ngraphs _ | graph _ | nexecs _ | exec _ | cap _ _ _ _ => exact Fact.holds_drv rfl hdv.1 this
  | nwg _ | wgo _ _ | wg1 _ _ => exact Fact.holds_wg rfl hdv.2.2.2.1 this
  | wgText _ _ _ => cases hc

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
    (hhp : w'.heap = w.heap) (hdv : w'.dev = w.dev ∧ w'.kernel = w.kernel ∧ w'.vendor = w.vendor ∧ w'.wg = w.wg ∧ w'.wgslOk = w.wgslOk)
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
  | drvInit | retained _ | nbufs _ | alloc _ _ | nmods _ | nfuncs _ | mod _ _ | fn _ _ | nblas _ | blasOn _ _ | ngraphs _ | graph _ | nexecs _ | exec _ | cap _ _ _ _ => exact Fact.holds_drv rfl hdv.1 hx0
  | nwg _ | wgo _ _ | wg1 _ _ => exact Fact.holds_wg rfl hdv.2.2.2.1 hx0
  | wgText c k sh =>
      simp only [storeKeeps, decide_eq_true_eq] at hk
      have hz' : w'.mem.frozen = false := by
        have := hp .frozen; simp only [Part.get] at this; rw [this]; exact hx0.1
      refine wgText_kept hx0 hz' (fun rg _ => congrFun hsz rg) hdv.2.2.2.2 fun i hi => Unchanged.of_eq ?_
      refine Mem.store_other h _ fun j hj e => ?_
      have hw : c.toNat + k < 2 ^ 64 := hx0.2.1
      have e2 := congrArg UInt64.toNat e
      rw [UInt64.toNat_add, UInt64.toNat_add, UInt64.toNat_ofNat', UInt64.toNat_ofNat',
        Nat.mod_eq_of_lt (by omega : i < 2 ^ 64), Nat.mod_eq_of_lt (by omega : j < 2 ^ 64),
        Nat.mod_eq_of_lt (by omega : c.toNat + i < 2 ^ 64)] at e2
      rcases hk with h1 | ⟨h1, h2⟩
      · rw [Nat.mod_eq_of_lt (by omega : a.toNat + j < 2 ^ 64)] at e2; omega
      · rw [Nat.mod_eq_of_lt (by omega : a.toNat + j < 2 ^ 64)] at e2; omega
  | oracles => show Sem.KeepsSize w'.kernel ∧ _; rw [hdv.2.1, hdv.2.2.1]; exact hx0
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
  | text c t k _ =>
      simp only [storeKeeps, decide_eq_true_eq] at hk
      have hz' : w'.mem.frozen = false := by
        have := hp .frozen; simp only [Part.get] at this; rw [this]; exact hx0.1.1
      have hw := hx0.1.2.2.1
      refine text_kept hx0 hz' (fun rg _ => congrFun hsz rg) fun i hi => Unchanged.of_eq ?_
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
  | text c _ k _ | wgText c k _ =>
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
    (hdv : w'.dev = w.dev ∧ w'.kernel = w.kernel ∧ w'.vendor = w.vendor ∧ w'.wg = w.wg ∧ w'.wgslOk = w.wgslOk)
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

