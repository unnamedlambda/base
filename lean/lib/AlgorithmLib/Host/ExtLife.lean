module
public import AlgorithmLib.Host.ExtContracts
meta import AlgorithmLib.Host.ExtContracts
public import AlgorithmLib.Host.Lifecycle
meta import AlgorithmLib.Host.Lifecycle
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Typestates across the engine's own libraries

What a call of the C, window, CPU, serial or USB library leaves of a typestate
(`extAfter`), proven of the libraries' reading (`extAfter_sound`). These
libraries change only their own handles and the memory in a call's frame, so
a call keeps every lifecycle part and every room, and every cell its frame
cannot reach (`extKeeps`). It keeps each handle the typestate holds, except
the one it closes. An open that answers other than null leaves its answer
held, so the fact comes from the answer and not from the arguments: a
program learns it holds a window only where it has checked the handle.

A call of another library keeps nothing here.
-/

namespace AlgorithmLib.HProg.Contracts

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

/-- The engine's own libraries, whose calls change only their own handles and
    the memory in their frame: every engine function but a thread's start and
    join, which change which memory is the program's. -/
def _root_.AlgorithmLib.IR.Ext.aside : Ext → Bool
  | .present _ | .c _ | .window _ | .cpu _ | .serial _ | .usb _ | .fs _ | .net _ => true
  | .base f => !f.thread
  | _ => false

/-- The kind of handle a call opens. -/
def _root_.AlgorithmLib.IR.Ext.opens : Ext → Option Held
  | .window .open => some .window
  | .serial .open => some .port
  | .usb .open => some .usbDevice
  | .usb .claim => some .usbInterface
  | .c .calloc => some .heap
  | .fs .open => some .file
  | .fs .dirOpen => some .dir
  | .net .tcpListen => some .listener
  | .net .tcpConnect | .net .tcpAccept => some .stream
  | .net .udpBind => some .udp
  | _ => none

/-- The kind of handle a call closes: the one it is handed first. -/
def _root_.AlgorithmLib.IR.Ext.closes : Ext → Option Held
  | .window .close => some .window
  | .serial .close => some .port
  | .usb .close => some .usbDevice
  | .usb .release => some .usbInterface
  | .c .free => some .heap
  | .fs .close => some .file
  | .fs .dirClose => some .dir
  | .net .tcpListenerClose => some .listener
  | .net .tcpClose => some .stream
  | .net .udpClose => some .udp
  | _ => none

set_option hygiene false in
/-- Takes apart a decoder's answer `h : decoder … = some c` down to each way
    it answers, closing the ways that cannot give `c`. -/
macro "decode_inv" : tactic => `(tactic| (
  repeat' (first
    | (split at h)
    | (dsimp only at h)
    | (rw [Option.bind_eq_bind, Option.bind_eq_some_iff] at h; obtain ⟨_, -, h⟩ := h)
    | (rw [Option.bind_eq_some_iff] at h; obtain ⟨_, -, h⟩ := h)
    | (rw [Option.map_eq_some_iff] at h; obtain ⟨_, -, h⟩ := h)
    | (injection h with h))
  all_goals first | cases h | skip))

-- ---------------------------------------------------------------------------
-- Handles held
-- ---------------------------------------------------------------------------

/-- A handle is held when some open slot has its address. -/
theorem find_live_iff {α : Type} (arr : Array α) (live : α → Bool) (addr : Nat → UInt64) (a : UInt64) :
    ((List.range arr.size).find? fun k => addr k == a && (arr[k]?.map live).getD false).isSome = true ↔
      ∃ j, addr j = a ∧ arr[j]?.map live = some true := by
  rw [List.find?_isSome]
  constructor
  · rintro ⟨j, -, h⟩
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    refine ⟨j, h.1, ?_⟩
    have h2 := h.2
    revert h2
    cases arr[j]?.map live with
    | none => intro h2; cases h2
    | some b => intro h2; simp only [Option.getD_some] at h2; rw [h2]
  · rintro ⟨j, hj, h⟩
    refine ⟨j, ?_, by simp [hj, h]⟩
    rw [List.mem_range]
    rcases Nat.lt_or_ge j arr.size with hl | hl
    · exact hl
    · rw [Array.getElem?_eq_none hl] at h; cases h

theorem window_live_iff {w : World} {a : UInt64} :
    Held.window.live w a = true ↔ ∃ j, wlWinAddr j = a ∧ w.wl.windows[j]?.map (·.live) = some true :=
  find_live_iff _ _ _ _

theorem port_live_iff {w : World} {a : UInt64} :
    Held.port.live w a = true ↔ ∃ j, serPortAddr j = a ∧ w.ser[j]?.map (·.live) = some true :=
  find_live_iff _ _ _ _

theorem usbDevice_live_iff {w : World} {a : UInt64} :
    Held.usbDevice.live w a = true ↔ ∃ j, usbDevAddr j = a ∧ w.usb[j]?.map (·.live) = some true :=
  find_live_iff _ _ _ _

theorem usbInterface_live_iff {w : World} {a : UInt64} :
    Held.usbInterface.live w a = true ↔ ∃ j, usbIfaceAddr j = a ∧ w.usbIfaces[j]?.map (·.live) = some true :=
  find_live_iff _ _ _ _

theorem file_live_iff {w : World} {a : UInt64} :
    Held.file.live w a = true ↔ ∃ j, fsFileAddr j = a ∧ w.openFiles[j]?.map (·.live) = some true :=
  find_live_iff _ _ _ _

theorem dir_live_iff {w : World} {a : UInt64} :
    Held.dir.live w a = true ↔ ∃ j, fsDirAddr j = a ∧ w.openDirs[j]?.map (·.live) = some true :=
  find_live_iff _ _ _ _

/-- A socket of kind `kind` the program holds. -/
def sockHeld (kind : SockKind) (s : Sock) : Bool := s.live && s.held && s.kind == kind

theorem sock_live_iff {w : World} {kind : SockKind} {a : UInt64} :
    (w.sock? kind a).isSome = true ↔ ∃ j, netSockAddr j = a ∧ w.socks[j]?.map (sockHeld kind) = some true :=
  find_live_iff _ _ _ _

/-- The kinds of handle the file and network libraries hand out. -/
def _root_.AlgorithmLib.HProg.Contracts.Held.fsnet : Held → Bool
  | .file | .dir | .listener | .stream | .udp => true
  | _ => false

/-- A call that leaves the files and sockets alone keeps every such handle. -/
theorem fsnet_keep {w w' : World} (hf : w'.openFiles = w.openFiles) (hd : w'.openDirs = w.openDirs)
    (hs : w'.socks = w.socks) {k : Held} (hk : k.fsnet = true) {a : UInt64} (hl : k.live w a = true) :
    k.live w' a = true := by
  cases k <;> simp only [Held.fsnet, reduceCtorEq] at hk <;>
    simp only [Held.live, World.fsFile?, World.fsDir?, World.sock?, hf, hd, hs] at hl ⊢ <;> exact hl

/-- The slot a handle names has the handle's address. -/
theorem find_addr {α : Type} {arr : Array α} {live : α → Bool} {addr : Nat → UInt64} {a : UInt64} {j : Nat}
    (h : (List.range arr.size).find? (fun k => addr k == a && (arr[k]?.map live).getD false) = some j) :
    addr j = a := by
  have := List.find?_some h
  simp only [Bool.and_eq_true, beq_iff_eq] at this
  exact this.1

theorem map_modify_live {α : Type} (live : α → Bool) (arr : Array α) (k j : Nat) (g : α → α)
    (hg : ∀ x, live (g x) = live x) : (arr.modify k g)[j]?.map live = arr[j]?.map live := by
  rw [Array.getElem?_modify]
  split
  · cases arr[j]? with
    | none => rfl
    | some x => simp [hg]
  · rfl

theorem map_push_live {α : Type} (live : α → Bool) (arr : Array α) (x : α) {j : Nat}
    (h : arr[j]?.map live = some true) : (arr.push x)[j]?.map live = some true := by
  rw [Array.getElem?_push]
  split
  · rename_i hj; subst hj; simp at h
  · exact h

-- ---------------------------------------------------------------------------
-- Memory rewritten in place
-- ---------------------------------------------------------------------------

/-- `m'` is `m` with at most its bytes changed and pinned memory grown:
    every other region as large. -/
def Rewritten (m m' : Mem) : Prop :=
  (∀ r, r ≠ .pinned → m'.sizes r = m.sizes r) ∧ m.sizes .pinned ≤ m'.sizes .pinned ∧ m'.busy = m.busy ∧
    m'.lent = m.lent

theorem Rewritten.refl (m : Mem) : Rewritten m m := ⟨fun _ _ => rfl, Nat.le_refl _, rfl, rfl⟩

theorem Rewritten.of_eq {m m' : Mem} (hs : m'.sizes = m.sizes)
    (hb : m'.busy = m.busy) (hl : m'.lent = m.lent) : Rewritten m m' :=
  ⟨fun r _ => by rw [hs], by rw [hs]; exact Nat.le_refl _, hb, hl⟩

theorem Rewritten.copyIn (m : Mem) (a : UInt64) (src : ByteArray) :
    Rewritten m ((Sem.copyIn m a src).getD m) := by
  cases h : Sem.copyIn m a src with
  | none => exact Rewritten.refl m
  | some m' => exact Rewritten.of_eq (copyIn_sizes_eq h) (copyIn_busy h) (copyIn_lent h)

theorem Blk.live {w : World} {x : UInt64} {n : Nat} (h : Blk w x n) : Held.live .heap w x = true := by
  obtain ⟨off, m, hat, -⟩ := h
  simp [Held.live, hat]

/-- **What a call of the engine's own libraries leaves**: the observations
    made so far, every lifecycle part, memory rewritten in place, each handle
    held but the one it closes, and the handle an open answers. -/
structure LibStep (e : Ext) (bits : List UInt64) (w : World) (r : Option V) (w' : World) : Prop where
  obs : w'.obs = w.obs
  dev : w'.dev = w.dev
  display : w'.display = w.display
  cudaDevice : w'.cudaDevice = w.cudaDevice
  gpuAdapter : w'.gpuAdapter = w.gpuAdapter
  kernel : w'.kernel = w.kernel
  vendor : w'.vendor = w.vendor
  present : w'.present = w.present
  wg : w'.wg = w.wg
  wgslOk : w'.wgslOk = w.wgslOk
  threads : w'.threads = w.threads ∧ w'.lend = w.lend ∧ w'.grants = w.grants
  mem : Rewritten w.mem w'.mem
  held : ∀ k a, e.closes ≠ some k → k.live w a = true → k.live w' a = true
  opens : ∀ k x, e.opens = some k → r.bind asBits = some x → x ≠ 0 → k.live w' x = true
  /-- A function of the C library that frees nothing keeps every heap block. -/
  blk : (∃ f, e = .c f) → e.closes ≠ some .heap → ∀ x n, Blk w x n → Blk w' x n

theorem answer_zero {r : Option V} {x : UInt64} (hr : r = some (.sc .i64 0)) (hx : r.bind asBits = some x)
    (h0 : x ≠ 0) : False := by
  subst hr; simp [asBits] at hx; exact h0 hx.symm

-- ---------------------------------------------------------------------------
-- The window library
-- ---------------------------------------------------------------------------

theorem wlPump_live (w : World) (j : Nat) :
    w.wlPump.wl.windows[j]?.map (·.live) = w.wl.windows[j]?.map (·.live) := by
  unfold World.wlPump
  split
  · rfl
  split
  · rfl
  · rename_i batch rest hin
    clear hin
    dsimp only
    generalize w.wl.windows = ws
    induction batch generalizing ws with
    | nil => rfl
    | cons b bs ih =>
        obtain ⟨k, r⟩ := b
        simp only [List.foldl_cons]
        rw [ih, map_modify_live]
        intro x; split <;> rfl

theorem wlPump_shape (w : World) : ∃ wl inp, w.wlPump = { w with wl := wl, wlInput := inp } := by
  unfold World.wlPump
  split
  · exact ⟨w.wl, w.wlInput, rfl⟩
  split
  · exact ⟨w.wl, w.wlInput, rfl⟩
  · exact ⟨_, _, rfl⟩

theorem wlEffect_shape (c : WlCall) (w : World) :
    ∃ wl inp, (wlEffect c w).2 = { w with wl := wl, wlInput := inp } := by
  cases c with
  | init =>
      simp only [wlEffect]
      split
      · exact ⟨w.wl, w.wlInput, rfl⟩
      split
      · exact ⟨_, w.wlInput, rfl⟩
      · exact ⟨w.wl, w.wlInput, rfl⟩
  | «open» => exact ⟨_, w.wlInput, rfl⟩
  | close => exact ⟨_, w.wlInput, rfl⟩
  | pump => exact wlPump_shape w
  | poll k =>
      simp only [wlEffect]
      split
      · exact ⟨_, w.wlInput, rfl⟩
      · exact ⟨w.wl, w.wlInput, rfl⟩
  | answer => exact ⟨w.wl, w.wlInput, rfl⟩

theorem wlEffect_live {c : WlCall} {w : World} {j : Nat} (hc : ∀ k, c = .close k → k ≠ j)
    (h : w.wl.windows[j]?.map (·.live) = some true) :
    (wlEffect c w).2.wl.windows[j]?.map (·.live) = some true := by
  cases c with
  | init =>
      simp only [wlEffect]
      split
      · exact h
      split <;> exact h
  | «open» => exact map_push_live _ _ _ h
  | close k =>
      simp only [wlEffect, Array.getElem?_modify]
      rw [if_neg (hc k rfl)]; exact h
  | pump => simp only [wlEffect]; rw [wlPump_live]; exact h
  | poll k =>
      simp only [wlEffect]
      split
      · rw [map_modify_live _ _ _ _ _ (by intro; rfl)]; exact h
      · exact h
  | answer => exact h

theorem wlDecode_close {f : WindowFn} {bits : List UInt64} {w : World} {j : Nat}
    (h : wlDecode f bits w = some (.close j)) : f = .close ∧ ∃ p, bits = [p] ∧ w.wlWin? p = some j := by
  cases f
  case close =>
    simp only [wlDecode] at h
    unfold wlDClose at h
    split at h
    · obtain ⟨k, hk, e⟩ := Option.map_eq_some_iff.mp h
      cases e
      exact ⟨rfl, _, rfl, hk⟩
    · cases h
  all_goals
    simp only [wlDecode] at h
    try unfold wlDOpen at h
    try unfold wlDPoll at h
    try unfold wlDPixels at h
    decode_inv

theorem wlDecode_open {bits : List UInt64} {w : World} {c : WlCall} (h : wlDecode .open bits w = some c) :
    c = .answer (.sc .i64 0) ∨ ∃ wd ht, c = .open wd ht := by
  simp only [wlDecode] at h
  unfold wlDOpen at h
  decode_inv
  all_goals first | exact .inl rfl | exact .inr ⟨_, rfl⟩ | exact .inr ⟨_, _, rfl⟩

theorem wlCall_of {f : WindowFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : wlCall f bits w = some (r, w')) :
    ∃ c, wlDecode f bits w = some c ∧ r = (wlEffect c w).1 ∧
      ∃ M, w' = { (wlEffect c w).2 with mem := M } ∧ Rewritten w.mem M := by
  simp only [wlCall] at h
  obtain ⟨c, hc, h⟩ := Option.map_eq_some_iff.mp h
  refine ⟨c, hc, ?_⟩
  split at h
  all_goals first
    | (simp only [Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h
       refine ⟨rfl, _, rfl, ?_⟩
       first
         | exact Rewritten.copyIn _ _ _
         | (split <;> first | exact Rewritten.copyIn _ _ _ | exact Rewritten.refl _))
    | (have e1 : r = (wlEffect c w).1 := by rw [h]
       have e2 : w' = (wlEffect c w).2 := by rw [h]
       refine ⟨e1, w.mem, ?_, Rewritten.refl _⟩
       rw [e2, ← wlEffect_mem c w])

/-- **A window call but an open answers a value of the type it returns.** -/
theorem wl_answer {f : WindowFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : wlCall f bits w = some (r, w')) (hf : f ≠ .open) :
    ∀ v, r = some v → ∃ x, v = .sc ((Ext.window f).sig.2.getD .i64) x := by
  obtain ⟨c, hc, rfl, -⟩ := wlCall_of h
  intro v hv
  cases f with
  | «open» => exact absurd rfl hf
  | init =>
      simp only [wlDecode] at hc; split at hc
      · cases hc; simp only [wlEffect] at hv; split at hv
        · cases hv; exact ⟨_, rfl⟩
        · split at hv <;> (cases hv; exact ⟨_, rfl⟩)
      · cases hc
  | close =>
      simp only [wlDecode, wlDClose] at hc; split at hc
      · obtain ⟨k, -, rfl⟩ := Option.map_eq_some_iff.mp hc; cases hv
      · cases hc
  | pump => simp only [wlDecode] at hc; split at hc
            · cases hc; cases hv
            · cases hc
  | poll =>
      simp only [wlDecode, wlDPoll] at hc; split at hc
      · simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hc
        obtain ⟨k, -, -, -, hc⟩ := hc
        cases hc
        simp only [wlEffect] at hv; split at hv <;> (cases hv; exact ⟨_, rfl⟩)
      · cases hc
  | pixels =>
      simp only [wlDecode, wlDPixels] at hc; split at hc
      · simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hc
        obtain ⟨k, -, hc⟩ := hc
        cases hc
        cases hv; exact ⟨_, rfl⟩
      · cases hc

theorem wl_step {f : WindowFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : wlCall f bits w = some (r, w')) : LibStep (.window f) bits w r w' := by
  obtain ⟨c, hc, hr, M, rfl, hM⟩ := wlCall_of h
  obtain ⟨wl, inp, he⟩ := wlEffect_shape c w
  have hwl : ((wlEffect c w).2).wl = wl := by rw [he]
  refine ⟨by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he]; exact ⟨rfl, rfl, rfl⟩, hM, ?_, ?_, fun ⟨_, h⟩ => nomatch h⟩
  · intro k a hk hl
    cases k with
    | window =>
        obtain ⟨j, hj, hlj⟩ := window_live_iff.mp hl
        refine window_live_iff.mpr ⟨j, hj, ?_⟩
        show (wlEffect c w).2.wl.windows[j]?.map (·.live) = some true
        refine wlEffect_live ?_ hlj
        rintro k rfl rfl
        obtain ⟨rfl, p, rfl, hp⟩ := wlDecode_close hc
        exact hk rfl
    | port =>
        have : ((wlEffect c w).2).ser = w.ser := by rw [he]
        simp only [Held.live, World.serPort?] at hl ⊢; rw [this]; exact hl
    | usbDevice =>
        have : ((wlEffect c w).2).usb = w.usb := by rw [he]
        simp only [Held.live, World.usbDev?] at hl ⊢; rw [this]; exact hl
    | usbInterface =>
        have : ((wlEffect c w).2).usbIfaces = w.usbIfaces := by rw [he]
        simp only [Held.live, World.usbIface?] at hl ⊢; rw [this]; exact hl
    | heap =>
        have : ((wlEffect c w).2).heap = w.heap := by rw [he]
        simp only [Held.live, World.heapAt?] at hl ⊢; rw [this]; exact hl
    | wg _ =>
        have : ((wlEffect c w).2).wg = w.wg := by rw [he]
        simp only [Held.live] at hl ⊢; rw [this]; exact hl
    | _ => exact fsnet_keep (by rw [he]) (by rw [he]) (by rw [he]) rfl hl
  · intro k x hk hx h0
    cases f <;> (try cases hk)
    rcases wlDecode_open hc with rfl | ⟨wd, ht, rfl⟩
    · exact (answer_zero hr hx h0).elim
    · simp only [wlEffect] at hr
      subst hr
      simp only [Option.bind_some, asBits, Option.some.injEq] at hx
      subst hx
      refine window_live_iff.mpr ⟨w.wl.windows.size, rfl, ?_⟩
      simp [wlEffect]

-- ---------------------------------------------------------------------------
-- The serial library
-- ---------------------------------------------------------------------------

theorem serEffect_shape (c : SerCall) (w : World) : ∃ ser, (serEffect c w).2 = { w with ser := ser } := by
  cases c with
  | «open» p =>
      simp only [serEffect]
      split
      · exact ⟨_, rfl⟩
      · exact ⟨w.ser, rfl⟩
  | _ => exact ⟨_, rfl⟩

theorem serEffect_live {c : SerCall} {w : World} {j : Nat} (hc : ∀ k, c = .close k → k ≠ j)
    (h : w.ser[j]?.map (·.live) = some true) :
    (serEffect c w).2.ser[j]?.map (·.live) = some true := by
  cases c with
  | name => exact h
  | «open» p =>
      simp only [serEffect]
      split
      · exact map_push_live _ _ _ h
      · exact h
  | close k =>
      simp only [serEffect, Array.getElem?_modify]
      rw [if_neg (hc k rfl)]; exact h
  | read k => simp only [serEffect]; rw [map_modify_live _ _ _ _ _ (by intro; rfl)]; exact h
  | write k => simp only [serEffect]; rw [map_modify_live _ _ _ _ _ (by intro; rfl)]; exact h
  | answer => exact h

theorem serDecode_close {f : SerialFn} {bits : List UInt64} {w : World} {j : Nat}
    (h : serDecode f bits w = some (.close j)) : f = .close ∧ ∃ p, bits = [p] ∧ w.serPort? p = some j := by
  cases f
  case close =>
    simp only [serDecode] at h
    unfold serDClose at h
    split at h
    · obtain ⟨k, hk, e⟩ := Option.map_eq_some_iff.mp h
      cases e
      exact ⟨rfl, _, rfl, hk⟩
    · cases h
  all_goals
    simp only [serDecode] at h
    try unfold serDName at h
    try unfold serDOpen at h
    try unfold serDRead at h
    try unfold serDWrite at h
    try unfold serDPending at h
    decode_inv

theorem serDecode_open {bits : List UInt64} {w : World} {c : SerCall} (h : serDecode .open bits w = some c) :
    c = .answer (.sc .i64 0) ∨ ∃ p, c = .open p := by
  simp only [serDecode] at h
  unfold serDOpen at h
  decode_inv
  all_goals first | exact .inl rfl | exact .inr ⟨_, rfl⟩ | exact .inr ⟨_, _, rfl⟩

theorem serialCall_of {f : SerialFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : serialCall f bits w = some (r, w')) :
    ∃ c, serDecode f bits w = some c ∧ r = (serEffect c w).1 ∧
      ∃ M, w' = { (serEffect c w).2 with mem := M } ∧ Rewritten w.mem M := by
  simp only [serialCall] at h
  obtain ⟨c, hc, h⟩ := Option.map_eq_some_iff.mp h
  refine ⟨c, hc, ?_⟩
  split at h
  all_goals first
    | (simp only [Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h
       refine ⟨rfl, _, rfl, ?_⟩
       first
         | exact Rewritten.copyIn _ _ _
         | (split <;> first | exact Rewritten.copyIn _ _ _ | exact Rewritten.refl _))
    | (have e1 : r = (serEffect c w).1 := by rw [h]
       have e2 : w' = (serEffect c w).2 := by rw [h]
       refine ⟨e1, w.mem, ?_, Rewritten.refl _⟩
       rw [e2, ← serEffect_mem c w])

theorem serial_step {f : SerialFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : serialCall f bits w = some (r, w')) : LibStep (.serial f) bits w r w' := by
  obtain ⟨c, hc, hr, M, rfl, hM⟩ := serialCall_of h
  obtain ⟨ser, he⟩ := serEffect_shape c w
  refine ⟨by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he]; exact ⟨rfl, rfl, rfl⟩, hM, ?_, ?_, fun ⟨_, h⟩ => nomatch h⟩
  · intro k a hk hl
    cases k with
    | port =>
        obtain ⟨j, hj, hlj⟩ := port_live_iff.mp hl
        refine port_live_iff.mpr ⟨j, hj, ?_⟩
        show (serEffect c w).2.ser[j]?.map (·.live) = some true
        refine serEffect_live ?_ hlj
        rintro k rfl rfl
        obtain ⟨rfl, p, rfl, hp⟩ := serDecode_close hc
        exact hk rfl
    | window =>
        have : ((serEffect c w).2).wl = w.wl := by rw [he]
        simp only [Held.live, World.wlWin?] at hl ⊢; rw [this]; exact hl
    | usbDevice =>
        have : ((serEffect c w).2).usb = w.usb := by rw [he]
        simp only [Held.live, World.usbDev?] at hl ⊢; rw [this]; exact hl
    | usbInterface =>
        have : ((serEffect c w).2).usbIfaces = w.usbIfaces := by rw [he]
        simp only [Held.live, World.usbIface?] at hl ⊢; rw [this]; exact hl
    | heap =>
        have : ((serEffect c w).2).heap = w.heap := by rw [he]
        simp only [Held.live, World.heapAt?] at hl ⊢; rw [this]; exact hl
    | wg _ =>
        have : ((serEffect c w).2).wg = w.wg := by rw [he]
        simp only [Held.live] at hl ⊢; rw [this]; exact hl
    | _ => exact fsnet_keep (by rw [he]) (by rw [he]) (by rw [he]) rfl hl
  · intro k x hk hx h0
    cases f <;> (try cases hk)
    rcases serDecode_open hc with rfl | ⟨p, rfl⟩
    · exact (answer_zero hr hx h0).elim
    · simp only [serEffect] at hr
      split at hr
      · rename_i input hs
        subst hr
        simp only [Option.bind_some, asBits, Option.some.injEq] at hx
        subst hx
        refine port_live_iff.mpr ⟨w.ser.size, rfl, ?_⟩
        show ((serEffect (.open p) w).2.ser[w.ser.size]?).map (·.live) = some true
        unfold serEffect
        dsimp only
        split
        · simp
        · rename_i hs'; rw [hs] at hs'; cases hs'
      · exact (answer_zero hr hx h0).elim

-- ---------------------------------------------------------------------------
-- The USB library
-- ---------------------------------------------------------------------------

theorem usbEffect_shape (c : UsbCall) (w : World) :
    ∃ usb ifs, (usbEffect c w).2 = { w with usb := usb, usbIfaces := ifs } := by
  cases c <;> exact ⟨_, _, rfl⟩

theorem usbEffect_live {c : UsbCall} {w : World} {j : Nat} (hc : ∀ k, c = .close k → k ≠ j)
    (h : w.usb[j]?.map (·.live) = some true) :
    (usbEffect c w).2.usb[j]?.map (·.live) = some true := by
  cases c with
  | «open» => exact map_push_live _ _ _ h
  | close k =>
      simp only [usbEffect, Array.getElem?_modify]
      rw [if_neg (hc k rfl)]; exact h
  | claim k => simp only [usbEffect]; rw [map_modify_live _ _ _ _ _ (by intro; rfl)]; exact h
  | release j =>
      simp only [usbEffect]
      split
      · rw [map_modify_live _ _ _ _ _ (by intro; rfl)]; exact h
      · exact h
  | transfer => exact h
  | answer => exact h

theorem usbEffect_iface_live {c : UsbCall} {w : World} {j : Nat} (hc : ∀ k, c = .release k → k ≠ j)
    (h : w.usbIfaces[j]?.map (·.live) = some true) :
    (usbEffect c w).2.usbIfaces[j]?.map (·.live) = some true := by
  cases c with
  | claim => exact map_push_live _ _ _ h
  | release k =>
      simp only [usbEffect, Array.getElem?_modify]
      rw [if_neg (hc k rfl)]; exact h
  | «open» => exact h
  | close => exact h
  | transfer => exact h
  | answer => exact h

theorem usbDecode_close {f : UsbFn} {bits : List UInt64} {w : World} {j : Nat}
    (h : usbDecode f bits w = some (.close j)) : f = .close ∧ ∃ p, bits = [p] ∧ w.usbDev? p = some j := by
  cases f
  case close =>
    simp only [usbDecode] at h
    unfold usbDClose at h
    split at h
    · obtain ⟨k, hk, e⟩ := Option.map_eq_some_iff.mp h
      cases e
      exact ⟨rfl, _, rfl, hk⟩
    · cases h
  all_goals
    simp only [usbDecode] at h
    try unfold usbDInfo at h
    try unfold usbDOpen at h
    try unfold usbDClaim at h
    try unfold usbDRelease at h
    try unfold usbDControl at h
    try unfold usbDData at h
    decode_inv

theorem usbDecode_release {f : UsbFn} {bits : List UInt64} {w : World} {j : Nat}
    (h : usbDecode f bits w = some (.release j)) : f = .release := by
  cases f
  case release => rfl
  all_goals
    simp only [usbDecode] at h
    try unfold usbDInfo at h
    try unfold usbDOpen at h
    try unfold usbDClose at h
    try unfold usbDClaim at h
    try unfold usbDControl at h
    try unfold usbDData at h
    decode_inv

theorem usbDecode_claim {bits : List UInt64} {w : World} {c : UsbCall} (h : usbDecode .claim bits w = some c) :
    c = .answer (.sc .i64 0) ∨ ∃ k n, c = .claim k n := by
  simp only [usbDecode] at h
  unfold usbDClaim at h
  decode_inv
  all_goals first | exact .inl rfl | exact .inr ⟨_, _, rfl⟩

theorem usbDecode_open {bits : List UInt64} {w : World} {c : UsbCall} (h : usbDecode .open bits w = some c) :
    c = .answer (.sc .i64 0) ∨ ∃ i, c = .open i := by
  simp only [usbDecode] at h
  unfold usbDOpen at h
  decode_inv
  all_goals first | exact .inl rfl | exact .inr ⟨_, rfl⟩ | exact .inr ⟨_, _, rfl⟩

theorem usbCall_of {f : UsbFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : usbCall f bits w = some (r, w')) :
    ∃ c, usbDecode f bits w = some c ∧ r = (usbEffect c w).1 ∧
      ∃ M, w' = { (usbEffect c w).2 with mem := M } ∧ Rewritten w.mem M := by
  simp only [usbCall] at h
  obtain ⟨c, hc, h⟩ := Option.map_eq_some_iff.mp h
  refine ⟨c, hc, ?_⟩
  split at h
  all_goals first
    | (simp only [Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h
       refine ⟨rfl, _, rfl, ?_⟩
       first
         | exact Rewritten.copyIn _ _ _
         | (split <;> first | exact Rewritten.copyIn _ _ _ | exact Rewritten.refl _))
    | (have e1 : r = (usbEffect c w).1 := by rw [h]
       have e2 : w' = (usbEffect c w).2 := by rw [h]
       refine ⟨e1, w.mem, ?_, Rewritten.refl _⟩
       rw [e2, ← usbEffect_mem c w])

theorem usb_step {f : UsbFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : usbCall f bits w = some (r, w')) : LibStep (.usb f) bits w r w' := by
  obtain ⟨c, hc, hr, M, rfl, hM⟩ := usbCall_of h
  obtain ⟨usb, ifs, he⟩ := usbEffect_shape c w
  refine ⟨by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he]; exact ⟨rfl, rfl, rfl⟩, hM, ?_, ?_, fun ⟨_, h⟩ => nomatch h⟩
  · intro k a hk hl
    cases k with
    | usbInterface =>
        obtain ⟨j, hj, hlj⟩ := usbInterface_live_iff.mp hl
        refine usbInterface_live_iff.mpr ⟨j, hj, ?_⟩
        show (usbEffect c w).2.usbIfaces[j]?.map (·.live) = some true
        refine usbEffect_iface_live ?_ hlj
        rintro k rfl rfl
        exact hk (by rw [usbDecode_release hc]; rfl)
    | usbDevice =>
        obtain ⟨j, hj, hlj⟩ := usbDevice_live_iff.mp hl
        refine usbDevice_live_iff.mpr ⟨j, hj, ?_⟩
        show (usbEffect c w).2.usb[j]?.map (·.live) = some true
        refine usbEffect_live ?_ hlj
        rintro k rfl rfl
        obtain ⟨rfl, p, rfl, hp⟩ := usbDecode_close hc
        exact hk rfl
    | window =>
        have : ((usbEffect c w).2).wl = w.wl := by rw [he]
        simp only [Held.live, World.wlWin?] at hl ⊢; rw [this]; exact hl
    | port =>
        have : ((usbEffect c w).2).ser = w.ser := by rw [he]
        simp only [Held.live, World.serPort?] at hl ⊢; rw [this]; exact hl
    | heap =>
        have : ((usbEffect c w).2).heap = w.heap := by rw [he]
        simp only [Held.live, World.heapAt?] at hl ⊢; rw [this]; exact hl
    | wg _ =>
        have : ((usbEffect c w).2).wg = w.wg := by rw [he]
        simp only [Held.live] at hl ⊢; rw [this]; exact hl
    | _ => exact fsnet_keep (by rw [he]) (by rw [he]) (by rw [he]) rfl hl
  · intro k x hk hx h0
    cases f <;> (try cases hk)
    · rcases usbDecode_open hc with rfl | ⟨i, rfl⟩
      · exact (answer_zero hr hx h0).elim
      · simp only [usbEffect] at hr
        subst hr
        simp only [Option.bind_some, asBits, Option.some.injEq] at hx
        subst hx
        refine usbDevice_live_iff.mpr ⟨w.usb.size, rfl, ?_⟩
        simp [usbEffect]
    · rcases usbDecode_claim hc with rfl | ⟨k, n, rfl⟩
      · exact (answer_zero hr hx h0).elim
      · simp only [usbEffect] at hr
        subst hr
        simp only [Option.bind_some, asBits, Option.some.injEq] at hx
        subst hx
        refine usbInterface_live_iff.mpr ⟨w.usbIfaces.size, rfl, ?_⟩
        simp [usbEffect]

-- ---------------------------------------------------------------------------
-- The file library
-- ---------------------------------------------------------------------------

theorem fsEffect_shape (c : FsCall) (w : World) :
    ∃ fs fl dl, (fsEffect c w).2 = { w with fs := fs, openFiles := fl, openDirs := dl } := by
  cases c <;> simp only [fsEffect]
  all_goals (repeat' split) <;> exact ⟨_, _, _, rfl⟩

theorem fsEffect_file_live {c : FsCall} {w : World} {j : Nat} (hc : ∀ k, c = .close k → k ≠ j)
    (h : w.openFiles[j]?.map (·.live) = some true) :
    (fsEffect c w).2.openFiles[j]?.map (·.live) = some true := by
  cases c with
  | close k =>
      simp only [fsEffect, Array.getElem?_modify]
      rw [if_neg (hc k rfl)]; exact h
  | _ =>
      simp only [fsEffect]
      (repeat' split) <;> first | exact h | exact map_push_live _ _ _ h

theorem fsEffect_dir_live {c : FsCall} {w : World} {j : Nat} (hc : ∀ k, c = .dirClose k → k ≠ j)
    (h : w.openDirs[j]?.map (·.live) = some true) :
    (fsEffect c w).2.openDirs[j]?.map (·.live) = some true := by
  cases c with
  | dirClose k =>
      simp only [fsEffect, Array.getElem?_modify]
      rw [if_neg (hc k rfl)]; exact h
  | dirNext k =>
      simp only [fsEffect]
      split
      · rw [map_modify_live _ _ _ _ _ (by intro; rfl)]; exact h
      · exact h
  | _ =>
      simp only [fsEffect]
      (repeat' split) <;> first | exact h | exact map_push_live _ _ _ h

theorem fsDecode_close {f : FsFn} {bits : List UInt64} {w : World} {j : Nat}
    (h : fsDecode f bits w = some (.close j)) : f = .close := by
  cases f <;> simp only [fsDecode] at h
  case close => rfl
  all_goals
    try unfold fsDOpen at h
    try unfold fsDRead at h
    try unfold fsDWrite at h
    try unfold fsDSize at h
    try unfold fsDSetLen at h
    try unfold fsDSync at h
    try unfold fsDPathCall at h
    try unfold fsDRename at h
    try unfold fsDDirOpen at h
    try unfold fsDDirNext at h
    try unfold fsDDirClose at h
    try unfold fsPath at h
    decode_inv

theorem fsDecode_dirClose {f : FsFn} {bits : List UInt64} {w : World} {j : Nat}
    (h : fsDecode f bits w = some (.dirClose j)) : f = .dirClose := by
  cases f <;> simp only [fsDecode] at h
  case dirClose => rfl
  all_goals
    try unfold fsDOpen at h
    try unfold fsDClose at h
    try unfold fsDRead at h
    try unfold fsDWrite at h
    try unfold fsDSize at h
    try unfold fsDSetLen at h
    try unfold fsDSync at h
    try unfold fsDPathCall at h
    try unfold fsDRename at h
    try unfold fsDDirOpen at h
    try unfold fsDDirNext at h
    try unfold fsPath at h
    decode_inv

theorem fsCall_of {f : FsFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : fsCall f bits w = some (r, w')) :
    ∃ c, fsDecode f bits w = some c ∧ r = (fsEffect c w).1 ∧
      ∃ M, w' = { (fsEffect c w).2 with mem := M } ∧ Rewritten w.mem M := by
  simp only [fsCall] at h
  obtain ⟨c, hc, h⟩ := Option.map_eq_some_iff.mp h
  refine ⟨c, hc, ?_⟩
  split at h
  all_goals first
    | (simp only [Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h
       refine ⟨rfl, _, rfl, ?_⟩
       exact Rewritten.copyIn _ _ _)
    | (have e1 : r = (fsEffect c w).1 := by rw [h]
       have e2 : w' = (fsEffect c w).2 := by rw [h]
       refine ⟨e1, w.mem, ?_, Rewritten.refl _⟩
       rw [e2, ← fsEffect_mem c w])

/-- An open's answer: null, or the handle it pushed, held. -/
theorem fsEffect_open_live {c : FsCall} {w : World} {x : UInt64}
    (hr : (fsEffect c w).1.bind asBits = some x) (h0 : x ≠ 0)
    (hc : c = .answer (.sc .i64 0) ∨ ∃ p r wr cr t, c = .open p r wr cr t) :
    Held.file.live (fsEffect c w).2 x = true := by
  rcases hc with rfl | ⟨p, r, wr, cr, t, rfl⟩
  · simp [fsEffect, asBits] at hr; exact absurd hr.symm h0
  · simp only [fsEffect] at hr ⊢
    repeat' split at hr
    all_goals first
      | (simp [asBits] at hr; exact absurd hr.symm h0)
      | (simp only [Option.bind_some, asBits, Option.some.injEq] at hr
         subst hr
         refine file_live_iff.mpr ⟨w.openFiles.size, rfl, ?_⟩
         simp_all)

theorem fsDecode_open {bits : List UInt64} {w : World} {c : FsCall} (h : fsDecode .open bits w = some c) :
    c = .answer (.sc .i64 0) ∨ ∃ p r wr cr t, c = .open p r wr cr t := by
  simp only [fsDecode] at h
  unfold fsDOpen at h
  split at h
  · unfold fsPath at h
    rw [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
    obtain ⟨_, -, h⟩ := h
    repeat' split at h
    all_goals (injection h with h; subst h)
    all_goals first | exact .inl rfl | exact .inr ⟨_, _, _, _, _, rfl⟩
  · cases h

theorem fsDecode_dirOpen {bits : List UInt64} {w : World} {c : FsCall} (h : fsDecode .dirOpen bits w = some c) :
    c = .answer (.sc .i64 0) ∨ ∃ p, c = .dirOpen p := by
  simp only [fsDecode] at h
  unfold fsDDirOpen at h
  split at h
  · unfold fsPath at h
    rw [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
    obtain ⟨_, -, h⟩ := h
    repeat' split at h
    all_goals (injection h with h; subst h)
    all_goals first | exact .inl rfl | exact .inr ⟨_, rfl⟩
  · cases h

theorem fs_step {f : FsFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : fsCall f bits w = some (r, w')) : LibStep (.fs f) bits w r w' := by
  obtain ⟨c, hc, hr, M, rfl, hM⟩ := fsCall_of h
  obtain ⟨fs, fl, dl, he⟩ := fsEffect_shape c w
  refine ⟨by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he]; exact ⟨rfl, rfl, rfl⟩, hM, ?_, ?_, fun ⟨_, h⟩ => nomatch h⟩
  · intro k a hk hl
    cases k with
    | file =>
        obtain ⟨j, hj, hlj⟩ := file_live_iff.mp hl
        refine file_live_iff.mpr ⟨j, hj, ?_⟩
        show (fsEffect c w).2.openFiles[j]?.map (·.live) = some true
        refine fsEffect_file_live ?_ hlj
        rintro k rfl rfl
        exact hk (by rw [fsDecode_close hc]; rfl)
    | dir =>
        obtain ⟨j, hj, hlj⟩ := dir_live_iff.mp hl
        refine dir_live_iff.mpr ⟨j, hj, ?_⟩
        show (fsEffect c w).2.openDirs[j]?.map (·.live) = some true
        refine fsEffect_dir_live ?_ hlj
        rintro k rfl rfl
        exact hk (by rw [fsDecode_dirClose hc]; rfl)
    | window =>
        have : ((fsEffect c w).2).wl = w.wl := by rw [he]
        simp only [Held.live, World.wlWin?] at hl ⊢; rw [this]; exact hl
    | port =>
        have : ((fsEffect c w).2).ser = w.ser := by rw [he]
        simp only [Held.live, World.serPort?] at hl ⊢; rw [this]; exact hl
    | usbDevice =>
        have : ((fsEffect c w).2).usb = w.usb := by rw [he]
        simp only [Held.live, World.usbDev?] at hl ⊢; rw [this]; exact hl
    | usbInterface =>
        have : ((fsEffect c w).2).usbIfaces = w.usbIfaces := by rw [he]
        simp only [Held.live, World.usbIface?] at hl ⊢; rw [this]; exact hl
    | heap =>
        have : ((fsEffect c w).2).heap = w.heap := by rw [he]
        simp only [Held.live, World.heapAt?] at hl ⊢; rw [this]; exact hl
    | wg _ =>
        have : ((fsEffect c w).2).wg = w.wg := by rw [he]
        simp only [Held.live] at hl ⊢; rw [this]; exact hl
    | listener =>
        have : ((fsEffect c w).2).socks = w.socks := by rw [he]
        simp only [Held.live, World.sock?] at hl ⊢; rw [this]; exact hl
    | stream =>
        have : ((fsEffect c w).2).socks = w.socks := by rw [he]
        simp only [Held.live, World.sock?] at hl ⊢; rw [this]; exact hl
    | udp =>
        have : ((fsEffect c w).2).socks = w.socks := by rw [he]
        simp only [Held.live, World.sock?] at hl ⊢; rw [this]; exact hl
  · intro k x hk hx h0
    cases f <;> (try cases hk)
    · subst hr
      show Held.file.live { (fsEffect c w).2 with mem := M } x = true
      have := fsEffect_open_live hx h0 (fsDecode_open hc)
      simpa [Held.live, World.fsFile?] using this
    · subst hr
      rcases fsDecode_dirOpen hc with rfl | ⟨p, rfl⟩
      · simp [fsEffect, asBits] at hx; exact absurd hx.symm h0
      · simp only [fsEffect] at hx ⊢
        split at hx
        · simp only [Option.bind_some, asBits, Option.some.injEq] at hx
          subst hx
          refine dir_live_iff.mpr ⟨w.openDirs.size, rfl, ?_⟩
          simp_all
        · simp [asBits] at hx; exact absurd hx.symm h0

-- ---------------------------------------------------------------------------
-- The network library
-- ---------------------------------------------------------------------------

theorem map_modify_mono {α : Type} (live : α → Bool) (arr : Array α) (k j : Nat) (g : α → α)
    (hg : ∀ x, live x = true → live (g x) = true) (h : arr[j]?.map live = some true) :
    (arr.modify k g)[j]?.map live = some true := by
  rw [Array.getElem?_modify]
  split
  · cases e : arr[j]? with
    | none => rw [e] at h; cases h
    | some x => rw [e] at h; simp only [Option.map_some, Option.some.injEq] at h ⊢; exact hg x h
  · exact h

theorem netEffect_shape (c : NetCall) (w : World) : ∃ ss, (netEffect c w).2 = { w with socks := ss } := by
  cases c <;> simp only [netEffect]
  all_goals (repeat' split) <;> exact ⟨_, rfl⟩

/-- A socket keeps being held through every call but its close. -/
theorem netEffect_live {c : NetCall} {w : World} {kind : SockKind} {j : Nat} (hc : ∀ k, c = .close k → k ≠ j)
    (h : w.socks[j]?.map (sockHeld kind) = some true) :
    (netEffect c w).2.socks[j]?.map (sockHeld kind) = some true := by
  cases c with
  | close k =>
      simp only [netEffect]
      have hk := hc k rfl
      split
      · refine map_modify_mono _ _ _ _ _ (fun x hx => by simpa [sockHeld] using hx) ?_
        rw [Array.getElem?_modify, if_neg hk]; exact h
      · rw [Array.getElem?_modify, if_neg hk]; exact h
  | _ =>
      simp only [netEffect]
      (repeat' split) <;>
        (repeat' first
          | exact h
          | apply map_push_live
          | refine map_modify_mono _ _ _ _ _ (fun x hx => by
              simp only [sockHeld, Bool.and_eq_true] at hx ⊢; (try split) <;> simp_all) ?_)

/-- The held kind a socket kind is. -/
def _root_.AlgorithmLib.HProg.Sem.SockKind.held : SockKind → Held
  | .listener => .listener
  | .stream => .stream
  | .udp => .udp

theorem sock_find {w : World} {kind : SockKind} {a : UInt64} {k : Nat} (h : w.sock? kind a = some k) :
    netSockAddr k = a ∧ w.socks[k]?.map (sockHeld kind) = some true := by
  unfold World.sock? at h
  have := List.find?_some h
  simp only [Bool.and_eq_true, beq_iff_eq] at this
  refine ⟨this.1, ?_⟩
  cases e : w.socks[k]? with
  | none => rw [e] at this; simp at this
  | some s => rw [e] at this; simp only [Option.map_some, Option.getD_some] at this; simp [sockHeld, this.2]

theorem netDecode_close {f : NetFn} {bits : List UInt64} {w : World} {j : Nat}
    (h : netDecode f bits w = some (.close j)) :
    ∃ kind p, (Ext.net f).closes = some kind.held ∧ w.sock? kind p = some j := by
  cases f <;> simp only [netDecode] at h
  case tcpClose =>
    unfold netDHeld at h; split at h
    · obtain ⟨k, hk, e⟩ := Option.map_eq_some_iff.mp h; cases e; exact ⟨.stream, _, rfl, hk⟩
    · cases h
  case tcpListenerClose =>
    unfold netDHeld at h; split at h
    · obtain ⟨k, hk, e⟩ := Option.map_eq_some_iff.mp h; cases e; exact ⟨.listener, _, rfl, hk⟩
    · cases h
  case udpClose =>
    unfold netDHeld at h; split at h
    · obtain ⟨k, hk, e⟩ := Option.map_eq_some_iff.mp h; cases e; exact ⟨.udp, _, rfl, hk⟩
    · cases h
  all_goals
    try unfold netDResolve at h
    try unfold netDTimed at h
    try unfold netDListen at h
    try unfold netDLocal at h
    try unfold netDAccept at h
    try unfold netDSend at h
    try unfold netDRecv at h
    try unfold netDHeld at h
    try unfold netDUdpSend at h
    try unfold netDUdpRecv at h
    decode_inv

theorem netCall_of {f : NetFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : netCall f bits w = some (r, w')) :
    ∃ c, netDecode f bits w = some c ∧ r = (netEffect c w).1 ∧
      ∃ M, w' = { (netEffect c w).2 with mem := M } ∧ Rewritten w.mem M := by
  simp only [netCall] at h
  obtain ⟨c, hc, h⟩ := Option.map_eq_some_iff.mp h
  refine ⟨c, hc, ?_⟩
  split at h
  all_goals first
    | (simp only [Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h
       refine ⟨rfl, _, rfl, ?_⟩
       exact Rewritten.copyIn _ _ _)
    | (have e1 : r = (netEffect c w).1 := by rw [h]
       have e2 : w' = (netEffect c w).2 := by rw [h]
       refine ⟨e1, w.mem, ?_, Rewritten.refl _⟩
       rw [e2, ← netEffect_mem c w])

/-- What an open of the network library answers, other than null, is a socket
    of the kind it opens, held. -/
theorem net_opens {f : NetFn} {bits : List UInt64} {w : World} {c : NetCall} {k : Held} {x : UInt64}
    (h : netDecode f bits w = some c) (hk : (Ext.net f).opens = some k)
    (hx : (netEffect c w).1.bind asBits = some x) (h0 : x ≠ 0) : k.live (netEffect c w).2 x = true := by
  have zero : ∀ {v : Option V}, v = some (.sc .i64 0) → v.bind asBits = some x → False := by
    intro v hv hb; subst hv; simp [asBits] at hb; exact h0 hb.symm
  cases f <;> simp only [Ext.opens, reduceCtorEq, Option.some.injEq] at hk <;> subst hk <;>
    simp only [netDecode] at h
  all_goals
    try unfold netDTimed at h
    try unfold netDListen at h
    try unfold netDAccept at h
    decode_inv
  all_goals first
    | exact (zero rfl hx).elim
    | (simp only [netEffect] at hx ⊢
       repeat' split at hx
       all_goals first
         | exact (zero rfl hx).elim
         | (simp only [Option.bind_some, asBits, Option.some.injEq, sockAns] at hx
            subst hx
            simp only [Held.live]
            refine sock_live_iff.mpr ⟨_, rfl, ?_⟩
            simp_all [sockHeld, sockAns, Array.getElem?_modify, Array.getElem?_push] <;>
              (try (split <;> simp_all))))

theorem net_step {f : NetFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : netCall f bits w = some (r, w')) : LibStep (.net f) bits w r w' := by
  obtain ⟨c, hc, hr, M, rfl, hM⟩ := netCall_of h
  obtain ⟨ss, he⟩ := netEffect_shape c w
  have keepSock : ∀ (kind : SockKind) a, (Ext.net f).closes ≠ some kind.held →
      (w.sock? kind a).isSome = true → ((netEffect c w).2.sock? kind a).isSome = true := by
    intro kind a hk hl
    obtain ⟨j, hj, hlj⟩ := sock_live_iff.mp hl
    refine sock_live_iff.mpr ⟨j, hj, netEffect_live ?_ hlj⟩
    rintro k rfl rfl
    obtain ⟨kind', p, hcl, hp⟩ := netDecode_close hc
    obtain ⟨-, hk'⟩ := sock_find hp
    have : kind' = kind := by
      cases e : w.socks[k]? with
      | none => rw [e] at hlj; cases hlj
      | some s =>
          rw [e] at hlj hk'
          simp only [Option.map_some, Option.some.injEq, sockHeld, Bool.and_eq_true, beq_iff_eq] at hlj hk'
          exact hk'.2.symm.trans hlj.2
    subst this
    exact hk hcl
  refine ⟨by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he], by rw [he]; exact ⟨rfl, rfl, rfl⟩, hM, ?_, ?_, fun ⟨_, h⟩ => nomatch h⟩
  · intro k a hk hl
    cases k with
    | listener => exact keepSock .listener a hk hl
    | stream => exact keepSock .stream a hk hl
    | udp => exact keepSock .udp a hk hl
    | window =>
        have : ((netEffect c w).2).wl = w.wl := by rw [he]
        simp only [Held.live, World.wlWin?] at hl ⊢; rw [this]; exact hl
    | port =>
        have : ((netEffect c w).2).ser = w.ser := by rw [he]
        simp only [Held.live, World.serPort?] at hl ⊢; rw [this]; exact hl
    | usbDevice =>
        have : ((netEffect c w).2).usb = w.usb := by rw [he]
        simp only [Held.live, World.usbDev?] at hl ⊢; rw [this]; exact hl
    | usbInterface =>
        have : ((netEffect c w).2).usbIfaces = w.usbIfaces := by rw [he]
        simp only [Held.live, World.usbIface?] at hl ⊢; rw [this]; exact hl
    | heap =>
        have : ((netEffect c w).2).heap = w.heap := by rw [he]
        simp only [Held.live, World.heapAt?] at hl ⊢; rw [this]; exact hl
    | wg _ =>
        have : ((netEffect c w).2).wg = w.wg := by rw [he]
        simp only [Held.live] at hl ⊢; rw [this]; exact hl
    | file =>
        have h1 : ((netEffect c w).2).openFiles = w.openFiles := by rw [he]
        simp only [Held.live, World.fsFile?] at hl ⊢; rw [h1]; exact hl
    | dir =>
        have h2 : ((netEffect c w).2).openDirs = w.openDirs := by rw [he]
        simp only [Held.live, World.fsDir?] at hl ⊢; rw [h2]; exact hl
  · intro k x hk hx h0
    subst hr
    have := net_opens hc hk hx h0
    cases k <;> simpa [Held.live, World.sock?, World.fsFile?, World.fsDir?, World.wlWin?, World.serPort?,
      World.usbDev?, World.usbIface?, World.heapAt?] using this

-- ---------------------------------------------------------------------------
-- The CPU library
-- ---------------------------------------------------------------------------

theorem cpu_step {f : CpuFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : cpuCall f bits w = some (r, w')) : LibStep (.cpu f) bits w r w' := by
  simp only [cpuCall] at h
  split at h
  · cases h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ⟨rfl, rfl, rfl⟩, Rewritten.refl _, fun _ _ _ hl => hl, ?_,
      fun ⟨_, h⟩ => nomatch h⟩
    intro k x hk; simp [Ext.opens] at hk

theorem base_step {f : BaseFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (hf : f.thread = false) (h : baseCall f bits w = some (r, w')) : LibStep (.base f) bits w r w' := by
  obtain ⟨hw, hmem, -⟩ := ExtContracts.baseCall_same hf h
  have hr : Rewritten w.mem w'.mem := by
    rcases hmem with hm | ⟨-, _, _, hc⟩
    · rw [hm]; exact Rewritten.refl _
    · exact Rewritten.of_eq (copyIn_sizes_eq hc) (copyIn_busy hc) (copyIn_lent hc)
  generalize w'.native = n at hw
  generalize w'.stdin = si at hw
  generalize w'.stdout = so at hw
  generalize w'.stderr = se at hw
  generalize w'.mem = m at hw hr
  subst hw
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ⟨rfl, rfl, rfl⟩, hr,
    fun k _ _ hl => by cases k <;> exact hl,
    fun k x hk => by simp [Ext.opens] at hk, fun ⟨_, h⟩ => nomatch h⟩

-- ---------------------------------------------------------------------------
-- The C library
-- ---------------------------------------------------------------------------

/-- A call that only rewrites memory and opens nothing. -/
theorem LibStep.of_mem {e : Ext} {bits : List UInt64} {w : World} {r : Option V} {m : Mem}
    (hm : Rewritten w.mem m) (ho : e.opens = none) (hl : m.pinnedLive = w.mem.pinnedLive) :
    LibStep e bits w r { w with mem := m } :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ⟨rfl, rfl, rfl⟩, hm, fun k _ _ hl => by cases k <;> exact hl,
    fun k _ hk => (by rw [ho] at hk; cases hk), fun _ _ _ _ h => h.same rfl hm.2.1 hl⟩

theorem find_cons_isSome {α : Type} (p : α → Bool) (x : α) (l : List α) (h : (l.find? p).isSome = true) :
    ((x :: l).find? p).isSome = true := by
  rw [List.find?_cons]; split
  · rfl
  · exact h

theorem cCall_step {f : CFn} {bits bits' : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : cCall f bits' w = some (r, w')) : LibStep (.c f) bits w r w' := by
  unfold cCall at h
  split at h
  -- memcpy
  · dsimp only at h
    split at h
    · cases h
    obtain ⟨bs, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    cases h
    exact LibStep.of_mem (Rewritten.of_eq (copyIn_sizes_eq hm) (copyIn_busy hm) (copyIn_lent hm)) rfl
      (DevSpec.copyIn_same hm).live.symm
  -- memmove
  · obtain ⟨bs, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    cases h
    exact LibStep.of_mem (Rewritten.of_eq (copyIn_sizes_eq hm) (copyIn_busy hm) (copyIn_lent hm)) rfl
      (DevSpec.copyIn_same hm).live.symm
  -- memset
  · obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    cases h
    exact LibStep.of_mem (Rewritten.of_eq (copyIn_sizes_eq hm) (copyIn_busy hm) (copyIn_lent hm)) rfl
      (DevSpec.copyIn_same hm).live.symm
  -- calloc
  · dsimp only at h
    split at h
    · cases h
    split at h
    · cases h
      refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ⟨rfl, rfl, rfl⟩, Rewritten.refl _, fun k _ _ hl => hl, ?_,
        fun _ _ _ _ h => h⟩
      intro k x _ hx h0
      exact (answer_zero rfl hx h0).elim
    · rename_i hz hroom
      cases h
      refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ⟨rfl, rfl, rfl⟩, ⟨fun r hr => by cases r <;> first | rfl | exact absurd rfl hr, ?_, rfl, rfl⟩,
        fun k a _ hl => ?_, ?_, fun _ _ _ _ h => h.alloc _⟩
      · simp only [Mem.sizes, Mem.region, ByteArray.size_append]; omega
      · cases k
        all_goals first | exact hl | skip
        simp only [Held.live, World.heapAt?] at hl ⊢
        split at hl
        · exact find_cons_isSome _ _ _ hl
        · exact hl
      · intro k x hk hx h0
        simp only [Ext.opens, Option.some.injEq] at hk; subst hk
        simp only [Option.bind_some, asBits, Option.some.injEq] at hx; subst hx
        simp only [beq_iff_eq] at hz
        have hoff : w.mem.pinnedNext < 2 ^ 36 := by
          simp only [regionSpan, UInt64.reduceToNat] at hroom; omega
        simp [Held.live, World.heapAt?, decodeAddr_pinned hoff]
  -- free
  · split at h
    · cases h
      exact LibStep.of_mem (Rewritten.refl _) rfl rfl
    · obtain ⟨⟨off, n⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
      cases h
      refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ⟨rfl, rfl, rfl⟩, Rewritten.of_eq (funext fun r => by cases r <;> rfl) rfl rfl,
        fun k a hk hl => ?_, ?_, fun _ hc => absurd rfl hc⟩
      · cases k
        all_goals first | exact hl | exact absurd rfl hk
      · intro k x hk; simp [Ext.opens] at hk
  · cases h

theorem c_step {f : CFn} {vs : List V} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : extCall (.c f) vs w = some (r, w')) : LibStep (.c f) bits w r w' := by
  simp only [extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true, Bool.false_eq_true,
    if_false] at h
  obtain ⟨bits', -, h⟩ := Option.bind_eq_some_iff.mp h
  exact cCall_step h

-- ---------------------------------------------------------------------------
-- The typestate
-- ---------------------------------------------------------------------------

/-- Asking whether a library is there changes nothing and answers a flag. -/
theorem present_same {l : Lib} {vs : List V} {w : World} {r : Option V} {w' : World}
    (h : extCall (.present l) vs w = some (r, w')) : w' = w ∧ ∃ x, r = some (.sc .i32 x) := by
  unfold extCall at h
  split at h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h; exact ⟨h.2.symm, _, h.1.symm⟩
  · split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h; exact ⟨h.2.symm, _, h.1.symm⟩
    all_goals first
      | (cases h; done)
      | simp_all

theorem lib_step {e : Ext} {vs : List V} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (ha : e.aside = true) (h : extCall e vs w = some (r, w')) (hb : vs.mapM asBits = some bits) :
    LibStep e bits w r w' := by
  cases e <;> simp only [Ext.aside, reduceCtorEq] at ha
  case c f => exact c_step h
  case present l =>
    obtain ⟨rfl, -⟩ := present_same h
    exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ⟨rfl, rfl, rfl⟩, Rewritten.refl _, fun _ _ _ hl => hl,
      fun k x hk => by simp [Ext.opens] at hk, fun ⟨_, h⟩ => nomatch h⟩
  all_goals
    simp only [extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true, Bool.false_eq_true,
      if_false, hb, Option.bind_eq_bind, Option.bind_some] at h
  · exact wl_step h
  · exact cpu_step h
  · exact serial_step h
  · exact usb_step h
  · exact fs_step h
  · exact net_step h
  · exact base_step (by simpa using ha) h

/-- Whether a call with these argument bits keeps a fact, whatever it
    answers. A close keeps no handle of the kind it closes: which one it
    closes is a value the typestate may not know. -/
def extKeeps (e : Ext) (bits : List UInt64) : Fact → Bool
  | .part _ _ | .room _ _ => e.aside
  | .cell a _ => e.aside && cellOk a && (List.range 8).all fun i => !e.frame.mayWrite bits (a + UInt64.ofNat i)
  | .held k _ | .opened k _ => e.aside && !decide (e.closes = some k)
  | .devSeq | .oracles => e.aside
  | .cstrIn a n => e.aside && !(e.frame.touches bits a n)
  | .roomArg _ _ | .devBuf _ _ => e.aside
  | .streams _ | .events _ | .evLive _ | .readyAll _ | .evAll _ => e.aside
  | .drvInit | .retained _ | .nbufs _ | .alloc _ _ | .nmods _ | .nfuncs _ | .mod _ _ | .fn _ _ _ | .modText _ _ | .fnText _ _ _ | .nblas _ | .blasOn _ _ | .ngraphs _ | .graph _ | .nexecs _ | .exec _ | .cap _ _ _ _ => e.aside
  | .nwg _ | .wgo _ _ | .wg1 _ _ => e.aside
  | .threads _ _ _ _ _ | .owns _ _ => e.aside
  | .wgText a n _ => e.aside && !(e.frame.touches bits a n)
  | .text _ _ _ _ => false
  | .pinnedUsed _ => false
  | .htAt _ _ _ | .cstr _ => false

/-- **What a library call leaves of a typestate**, from its argument bits and
    its answer: the facts it keeps, and for an open, that its answer is null
    or a handle held. -/
def extAfter (e : Ext) (bits : List UInt64) (ans : Option UInt64) (S : TState) : TState :=
  let kept := S.filter (extKeeps e bits)
  match e.opens, ans with
  | some k, some x => .opened k x :: kept
  | _, _ => kept

theorem extKeeps_aside {e : Ext} {bits : List UInt64} {x : Fact} (h : extKeeps e bits x = true) :
    e.aside = true := by
  cases x <;> simp_all [extKeeps]

theorem extKeeps_sound {e : Ext} {vs : List V} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : extCall e vs w = some (r, w')) (hb : vs.mapM asBits = some bits) (st : LibStep e bits w r w')
    {x : Fact} (hk : extKeeps e bits x = true) (hx : x.holds w) : x.holds w' := by
  cases x with
  | part p b =>
      show p.get w' = b
      rw [← hx]
      cases p
      · show w'.dev.live = w.dev.live; rw [st.dev]
      · show w'.dev.capture.isSome = w.dev.capture.isSome; rw [st.dev]
      · show w'.display = w.display; rw [st.display]
      · show w'.cudaDevice = w.cudaDevice; rw [st.cudaDevice]
      · show w'.gpuAdapter = w.gpuAdapter; rw [st.gpuAdapter]
      · show w'.mem.busy.isEmpty = w.mem.busy.isEmpty; rw [st.mem.2.2.1]
      · show w'.has _ = w.has _; simp only [World.has, st.present]
  | room rg n =>
      refine ⟨?_, fun ha => by rw [st.mem.2.2.2]; exact hx.2 ha⟩
      by_cases hp : rg = .pinned
      · subst hp; exact Nat.le_trans hx.1 st.mem.2.1
      · rw [st.mem.1 rg hp]; exact hx.1
  | cell a v =>
      simp only [extKeeps, Bool.and_eq_true] at hk
      obtain ⟨⟨-, hok⟩, hno⟩ := hk
      obtain ⟨rg, off, hd, hp, hbytes⟩ := cellOk_spec hok
      refine load_kept hd hp hbytes hx (st.mem.1 _ hp) st.mem.2.2.2 fun i hi => ?_
      have hm : e.frame.mayWrite bits (a + UInt64.ofNat i) = false := by
        have := List.all_eq_true.mp hno i (List.mem_range.mpr hi)
        simpa using this
      exact ExtContracts.ext_respects_frame e vs w r w' h bits hb _ (Frame.mayWrite_sound hm _)
  | held k a =>
      simp only [extKeeps, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at hk
      exact st.held k a hk.2 hx
  | opened k a =>
      simp only [extKeeps, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at hk
      exact fun h0 => st.held k a hk.2 (hx h0)
  | devSeq =>
      show Device.TrackOk w'.dev ∧ Device.DefaultOnly w'.dev.race
      rw [st.dev]; exact hx
  | oracles =>
      show Sem.KeepsSize w'.kernel ∧ VendorKeeps w'.vendor
      rw [st.kernel, st.vendor]; exact hx
  | devBuf _ _ =>
      show Device.BufIs _ _ w'.dev
      rw [st.dev]; exact hx
  | pinnedUsed _ => simp [extKeeps] at hk
  | streams _ | events _ | evLive _ | readyAll _ | evAll _ => exact Fact.holds_dev rfl st.dev hx
  | drvInit | retained _ | nbufs _ | alloc _ _ | nmods _ | nfuncs _ | mod _ _ | fn _ _ _ | modText _ _ | fnText _ _ _ | nblas _ | blasOn _ _ | ngraphs _ | graph _ | nexecs _ | exec _ | cap _ _ _ _ => exact Fact.holds_drv rfl st.dev hx
  | text _ _ _ _ => simp [extKeeps] at hk
  | nwg _ | wgo _ _ | wg1 _ _ => exact Fact.holds_wg rfl st.wg hx
  | wgText a n sh =>
      simp only [extKeeps, Bool.and_eq_true, Bool.not_eq_true'] at hk
      refine wgText_kept hx (fun rg hp => st.mem.1 rg hp) st.wgslOk st.mem.2.2.2 fun i hi => ?_
      exact ExtContracts.ext_respects_frame e vs w r w' h bits hb _
        (Frame.mayWrite_sound (Frame.touches_false hk.2 i hi) _)
  | roomArg rg x =>
      refine ⟨?_, hx.2.1, hx.2.2⟩
      by_cases hp : rg = .pinned
      · subst hp; exact Nat.le_trans hx.1 st.mem.2.1
      · rw [st.mem.1 rg hp]; exact hx.1
  | cstrIn a n =>
      simp only [extKeeps, Bool.and_eq_true, Bool.not_eq_true'] at hk
      refine cstrIn_kept hx (fun rg hp => st.mem.1 rg hp) st.mem.2.2.2 fun i hi => ?_
      exact ExtContracts.ext_respects_frame e vs w r w' h bits hb _
        (Frame.mayWrite_sound (Frame.touches_false hk.2 i hi) _)
  | htAt _ _ _ | cstr _ => simp [extKeeps] at hk
  | threads _ _ _ _ _ => exact threads_kept hx st.threads (st.mem.1 .arena (by decide)) st.mem.2.2.2
  | owns _ _ => exact owns_kept hx (st.mem.1 .arena (by decide)) st.mem.2.2.2

/-- **The typestate follows a library call**: from a typestate that holds,
    a call of the engine's own libraries lands in what `extAfter` computes. -/
theorem extAfter_sound {e : Ext} {vs : List V} {bits : List UInt64} {S : TState} {w : World}
    {r : Option V} {w' : World} (hS : S.holds w) (hb : vs.mapM asBits = some bits)
    (h : extCall e vs w = some (r, w')) : (extAfter e bits (r.bind asBits) S).holds w' := by
  have hkept : TState.holds (S.filter (extKeeps e bits)) w' := by
    intro x hx
    obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
    exact extKeeps_sound h hb (lib_step (extKeeps_aside hk) h hb) hk (hS x hm)
  unfold extAfter
  split
  · rename_i k x hk hx
    have ha : e.aside = true := by cases e <;> simp_all [Ext.opens, Ext.aside]
    exact TState.holds_cons (fun h0 => (lib_step ha h hb).opens k x hk hx h0) hkept
  · exact hkept

/-- A handle opened and checked not null is held. -/
theorem held_of_opened {S : TState} {w : World} {k : Held} {x : UInt64} (hS : S.holds w)
    (hm : Fact.opened k x ∈ S) (hx : x ≠ 0) : TState.holds (.held k x :: S) w :=
  TState.holds_cons ((hS _ hm) hx) hS

-- ---------------------------------------------------------------------------
-- Answers
-- ---------------------------------------------------------------------------

theorem wlDecode_answer {f : WindowFn} {bits : List UInt64} {w : World} {v : V}
    (h : wlDecode f bits w = some (.answer v)) : ∃ t x, v = .sc t x := by
  cases f <;> simp only [wlDecode] at h
  all_goals
    try unfold wlDOpen at h
    try unfold wlDClose at h
    try unfold wlDPoll at h
    try unfold wlDPixels at h
    decode_inv
  all_goals exact ⟨_, _, rfl⟩

theorem serDecode_answer {f : SerialFn} {bits : List UInt64} {w : World} {v : V}
    (h : serDecode f bits w = some (.answer v)) : ∃ t x, v = .sc t x := by
  cases f <;> simp only [serDecode] at h
  all_goals
    try unfold serDName at h
    try unfold serDOpen at h
    try unfold serDClose at h
    try unfold serDRead at h
    try unfold serDWrite at h
    try unfold serDPending at h
    decode_inv
  all_goals exact ⟨_, _, rfl⟩

theorem fsDecode_answer {f : FsFn} {bits : List UInt64} {w : World} {v : V}
    (h : fsDecode f bits w = some (.answer v)) : ∃ t x, v = .sc t x := by
  cases f <;> simp only [fsDecode] at h
  all_goals
    try unfold fsDOpen at h
    try unfold fsDClose at h
    try unfold fsDRead at h
    try unfold fsDWrite at h
    try unfold fsDSize at h
    try unfold fsDSetLen at h
    try unfold fsDSync at h
    try unfold fsDPathCall at h
    try unfold fsDRename at h
    try unfold fsDDirOpen at h
    try unfold fsDDirNext at h
    try unfold fsDDirClose at h
    try unfold fsPath at h
    try unfold fsNeg at h
    decode_inv
  all_goals exact ⟨_, _, rfl⟩

theorem netDecode_answer {f : NetFn} {bits : List UInt64} {w : World} {v : V}
    (h : netDecode f bits w = some (.answer v)) : ∃ t x, v = .sc t x := by
  cases f <;> simp only [netDecode] at h
  all_goals
    try unfold netDResolve at h
    try unfold netDTimed at h
    try unfold netDListen at h
    try unfold netDLocal at h
    try unfold netDAccept at h
    try unfold netDSend at h
    try unfold netDRecv at h
    try unfold netDHeld at h
    try unfold netDUdpSend at h
    try unfold netDUdpRecv at h
    try unfold netNeg at h
    try unfold netNull at h
    decode_inv
  all_goals exact ⟨_, _, rfl⟩

theorem usbDecode_answer {f : UsbFn} {bits : List UInt64} {w : World} {v : V}
    (h : usbDecode f bits w = some (.answer v)) : ∃ t x, v = .sc t x := by
  cases f <;> simp only [usbDecode] at h
  all_goals
    try unfold usbDInfo at h
    try unfold usbDOpen at h
    try unfold usbDClose at h
    try unfold usbDClaim at h
    try unfold usbDRelease at h
    try unfold usbDControl at h
    try unfold usbDData at h
    decode_inv
  all_goals first | exact ⟨_, _, rfl⟩ | exact ⟨_, _, rfl⟩

/-- Every answer of the C library is a scalar. -/
theorem cCall_scalar {f : CFn} {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : cCall f bits w = some (some v, w')) : ∃ t x, v = .sc t x := by
  unfold cCall at h
  split at h
  · dsimp only at h
    split at h
    · cases h
    obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    cases h; exact ⟨_, _, rfl⟩
  · obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    cases h; exact ⟨_, _, rfl⟩
  · obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    cases h; exact ⟨_, _, rfl⟩
  · dsimp only at h
    split at h
    · cases h
    split at h
    · cases h; exact ⟨_, _, rfl⟩
    · cases h; exact ⟨_, _, rfl⟩
  · split at h
    · cases h
    · obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      cases h
  · cases h

/-- Every answer of the engine's own libraries is a scalar. -/
theorem lib_scalar {e : Ext} {vs : List V} {bits : List UInt64} {w : World} {v : V} {w' : World}
    (ha : e.aside = true) (h : extCall e vs w = some (some v, w')) (hb : vs.mapM asBits = some bits) :
    ∃ t x, v = .sc t x := by
  cases e <;> simp only [Ext.aside, reduceCtorEq] at ha
  case c f =>
    simp only [extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true, Bool.false_eq_true,
      if_false] at h
    obtain ⟨bits', -, h⟩ := Option.bind_eq_some_iff.mp h
    exact cCall_scalar h
  case present l =>
    obtain ⟨-, x, hx⟩ := present_same h
    cases hx; exact ⟨_, _, rfl⟩
  all_goals
    simp only [extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true, Bool.false_eq_true,
      if_false, hb, Option.bind_eq_bind, Option.bind_some] at h
  · obtain ⟨c, hc, hr, -⟩ := wlCall_of h
    cases c <;> simp only [wlEffect, wlTrue, wlFalse] at hr
    all_goals (repeat' split at hr) <;> first
      | (cases hr <;> exact ⟨_, _, rfl⟩)
      | (cases hr <;> exact wlDecode_answer hc)
  · simp only [cpuCall] at h
    split at h
    · cases h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h
      exact ⟨_, _, rfl⟩
  · obtain ⟨c, hc, hr, -⟩ := serialCall_of h
    cases c <;> simp only [serEffect] at hr
    all_goals (repeat' split at hr) <;> first
      | (cases hr <;> exact ⟨_, _, rfl⟩)
      | (cases hr <;> exact serDecode_answer hc)
  · obtain ⟨c, hc, hr, -⟩ := usbCall_of h
    cases c <;> simp only [usbEffect] at hr
    all_goals (repeat' split at hr) <;> first
      | (cases hr <;> exact ⟨_, _, rfl⟩)
      | (cases hr <;> exact usbDecode_answer hc)
  · obtain ⟨c, hc, hr, -⟩ := fsCall_of h
    cases c <;> simp only [fsEffect] at hr
    all_goals (repeat' split at hr) <;> first
      | (cases hr <;> exact ⟨_, _, rfl⟩)
      | (cases hr <;> exact fsDecode_answer hc)
  · obtain ⟨c, hc, hr, -⟩ := netCall_of h
    cases c <;> simp only [netEffect] at hr
    all_goals (repeat' split at hr) <;> first
      | (cases hr <;> exact ⟨_, _, rfl⟩)
      | (cases hr <;> exact netDecode_answer hc)
  · obtain ⟨x, hx⟩ := (ExtContracts.baseCall_same (by simpa using ha) h).2.2 _ rfl
    exact ⟨_, x, hx⟩

/-- **An open answers a 64-bit word**: the handle it opened, or null. -/
theorem lib_open_i64 {e : Ext} {k : Held} (hk : e.opens = some k) {vs : List V} {bits : List UInt64}
    {w : World} {v : V} {w' : World} (h : extCall e vs w = some (some v, w')) (hb : vs.mapM asBits = some bits) :
    ∃ x, v = .sc .i64 x := by
  cases e <;> simp only [Ext.opens, reduceCtorEq] at hk
  case c f =>
    cases f <;> simp only [reduceCtorEq] at hk
    simp only [extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true, Bool.false_eq_true,
      if_false] at h
    obtain ⟨bits', -, h⟩ := Option.bind_eq_some_iff.mp h
    rcases bits' with _ | ⟨n, _ | ⟨sz, _ | ⟨_, _⟩⟩⟩ <;> simp only [cCall, reduceCtorEq] at h
    repeat' split at h
    all_goals simp only [Option.some.injEq, Prod.mk.injEq, reduceCtorEq] at h
    all_goals exact ⟨_, h.1.symm⟩
  all_goals rename_i f
  all_goals cases f <;> simp only [reduceCtorEq] at hk
  all_goals
    simp only [extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true, Bool.false_eq_true,
      if_false, hb, Option.bind_eq_bind, Option.bind_some] at h
  · obtain ⟨c, hc, hr, -⟩ := wlCall_of h
    simp only [wlDecode, wlDOpen] at hc
    repeat' split at hc
    all_goals first
      | (cases hc; done)
      | (simp only [Option.some.injEq] at hc; subst hc; simp only [wlEffect, Option.some.injEq] at hr; exact ⟨_, hr⟩)
      | (obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
         repeat' split at hc
         all_goals (simp only [Option.some.injEq] at hc; subst hc; simp only [wlEffect, Option.some.injEq] at hr
                    exact ⟨_, hr⟩))
  · obtain ⟨c, hc, hr, -⟩ := serialCall_of h
    simp only [serDecode, serDOpen] at hc
    repeat' split at hc
    all_goals first
      | (cases hc; done)
      | (obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
         repeat' split at hc
         all_goals (simp only [Option.some.injEq] at hc; subst hc; simp only [serEffect] at hr
                    repeat' split at hr
                    all_goals (simp only [Option.some.injEq] at hr; exact ⟨_, hr⟩)))
  · obtain ⟨c, hc, hr, -⟩ := usbCall_of h
    simp only [usbDecode, usbDOpen] at hc
    repeat' split at hc
    all_goals first
      | (cases hc; done)
      | (simp only [Option.some.injEq] at hc; subst hc; simp only [usbEffect, Option.some.injEq] at hr; exact ⟨_, hr⟩)
  · obtain ⟨c, hc, hr, -⟩ := usbCall_of h
    rcases usbDecode_claim hc with rfl | ⟨_, _, rfl⟩ <;>
      (simp only [usbEffect, Option.some.injEq] at hr; exact ⟨_, hr⟩)
  all_goals first
    | (obtain ⟨c, hc, hr, -⟩ := fsCall_of h
       first
         | (rcases fsDecode_open hc with rfl | ⟨_, _, _, _, _, rfl⟩)
         | (rcases fsDecode_dirOpen hc with rfl | ⟨_, rfl⟩)
       all_goals (simp only [fsEffect] at hr; (repeat' split at hr) <;>
         (simp only [Option.some.injEq] at hr; exact ⟨_, hr⟩)))
    | (obtain ⟨c, hc, hr, -⟩ := netCall_of h
       clear h
       have h := hc
       simp only [netDecode] at h
       try unfold netDTimed at h
       try unfold netDListen at h
       try unfold netDAccept at h
       decode_inv
       all_goals (simp only [netEffect, netNull, sockAns] at hr; (repeat' split at hr) <;>
         (simp only [Option.some.injEq] at hr; exact ⟨_, hr⟩)))

/-- A call of the engine's own libraries reads only its arguments' bits. -/
theorem extCall_bits {e : Ext} (ha : e.aside = true) {cs cs' : List V} {bits : List UInt64} (w : World)
    (h1 : cs.mapM asBits = some bits) (h2 : cs'.mapM asBits = some bits) :
    extCall e cs w = extCall e cs' w := by
  cases e <;> simp only [Ext.aside, reduceCtorEq] at ha
  case present l =>
    cases cs with
    | nil =>
        cases cs' with
        | nil => rfl
        | cons c cs' =>
            simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h1
            subst h1
            simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
              Option.some.injEq] at h2
            obtain ⟨_, _, _, _, e⟩ := h2
            cases e
    | cons c cs =>
        cases cs' with
        | nil =>
            simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h2
            subst h2
            simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
              Option.some.injEq] at h1
            obtain ⟨_, _, _, _, e⟩ := h1
            cases e
        | cons c' cs' => simp only [extCall]
  all_goals
    simp only [extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true, Bool.false_eq_true,
      if_false, h1, h2, Option.bind_eq_bind, Option.bind_some]

-- ---------------------------------------------------------------------------
-- What a call needs
-- ---------------------------------------------------------------------------

/-- A fact that names no handle: what `roomAt` reads. -/
def Fact.plain : Fact → Bool
  | .held _ _ | .opened _ _ => false
  | _ => true

/-- Room found among some of a typestate's facts is room in all of them. -/
theorem roomAt_sub {S S' : TState} (hsub : ∀ x ∈ S', x ∈ S) {a : UInt64} {n : Nat}
    (h : roomAt S' a n = true) : roomAt S a n = true := by
  unfold roomAt at h ⊢
  split at h
  · simp only [Bool.and_eq_true, decide_eq_true_eq, List.any_eq_true, List.contains_iff_mem] at h ⊢
    obtain ⟨⟨hp, hn⟩, x, hx, hr⟩ := h
    exact ⟨⟨hp, hn⟩, x, hsub _ hx, hr⟩
  · cases h

/-- What an engine function needs: its arguments, and for a wait or a wake
    four bytes at an address a multiple of four. -/
def baseNeeds : BaseFn → List UInt64 → Option (List (Held × UInt64) × List (UInt64 × Nat))
  | .clockNow, [] | .clockWall, [] => some ([], [])
  | .memLock, [_, _] | .memUnlock, [_, _] | .memAdviseHuge, [_, _] => some ([], [])
  | .threadPriority, [_] => some ([], [])
  | .memWait, [a, _, _] | .memWake, [a, _] => if a % 4 = 0 then some ([], [(a, 4)]) else none
  | .nativeArch, [] => some ([], [])
  | .nativeLoad, [src, len] => some ([], [(src, (asI64 len).toNat)])
  | .cpuHas, [p, c] | .stdoutWrite, [p, c] | .stderrWrite, [p, c] => some ([], [(p, c.toNat)])
  | .stdinRead, [d, c] | .random, [d, c] => some ([], [(d, c.toNat)])
  | _, _ => none

theorem baseNeeds_length {f : BaseFn} {bits : List UInt64} {n : List (Held × UInt64) × List (UInt64 × Nat)}
    (h : baseNeeds f bits = some n) : bits.length = (Ext.base f).sig.1.length := by
  unfold baseNeeds at h
  split at h <;> first | rfl | cases h

theorem baseNeeds_pre {f : BaseFn} {bits : List UInt64} {w : World} {hs : List (Held × UInt64)}
    {rs : List (UInt64 × Nat)} (hn : baseNeeds f bits = some (hs, rs)) (room : ∀ q ∈ rs, Fits w.mem q.1 q.2) :
    ExtContracts.basePre f bits w := by
  unfold baseNeeds at hn
  split at hn <;> (try cases hn) <;> (try trivial)
  all_goals first
    | exact fun _ => (room _ List.mem_cons_self).readable
    | exact fun _ => (room _ List.mem_cons_self).writable
    | exact (room _ List.mem_cons_self).readable
    | exact fun _ _ _ _ => (Fits.mono (Nat.min_le_right _ _) (room _ List.mem_cons_self)).writable
    | exact fun _ => ⟨(room _ List.mem_cons_self).readable,
        (room _ (List.mem_cons_of_mem _ List.mem_cons_self)).readable⟩
    | exact ⟨fun _ => (room _ List.mem_cons_self).readable, fun _ _ _ _ =>
        (Fits.mono (Nat.min_le_left _ _) (room _ (List.mem_cons_of_mem _ List.mem_cons_self))).writable⟩
    | (split at hn
       · rename_i ha; cases hn
         obtain ⟨r, off, hd, hp, hs, -, hl⟩ := room _ List.mem_cons_self
         exact ⟨ha, load_isSome hd hp hs hl⟩
       · cases hn)

/-- What a file library call needs: each handle it is handed, held, and room
    for each path and buffer. -/
def fsNeeds : FsFn → List UInt64 → Option (List (Held × UInt64) × List (UInt64 × Nat))
  | .open, [p, c, _] => some ([], [(p, c.toNat)])
  | .close, [h] => some ([(.file, h)], [])
  | .read, [h, buf, len, _] => some ([(.file, h)], [(buf, len.toNat)])
  | .write, [h, buf, len, _] => some ([(.file, h)], [(buf, len.toNat)])
  | .size, [h] => some ([(.file, h)], [])
  | .setLen, [h, _] => some ([(.file, h)], [])
  | .sync, [h] => some ([(.file, h)], [])
  | .createDir, [p, c] => some ([], [(p, c.toNat)])
  | .removeFile, [p, c] => some ([], [(p, c.toNat)])
  | .removeDir, [p, c] => some ([], [(p, c.toNat)])
  | .rename, [a, ac, b, bc] => some ([], [(a, ac.toNat), (b, bc.toNat)])
  | .dirOpen, [p, c] => some ([], [(p, c.toNat)])
  | .dirNext, [d, out, cap] => some ([(.dir, d)], [(out, cap.toNat)])
  | .dirClose, [d] => some ([(.dir, d)], [])
  | _, _ => none

theorem fsNeeds_length {f : FsFn} {bits : List UInt64} {n : List (Held × UInt64) × List (UInt64 × Nat)}
    (h : fsNeeds f bits = some n) : bits.length = (Ext.fs f).sig.1.length := by
  unfold fsNeeds at h
  split at h <;> first | rfl | cases h

/-- What a network library call needs: each socket it is handed, held, and
    room for each name, address record and buffer. -/
def netNeeds : NetFn → List UInt64 → Option (List (Held × UInt64) × List (UInt64 × Nat))
  | .resolve, [n, c, out, oc] => some ([], [(n, c.toNat), (out, oc.toNat)])
  | .tcpConnect, [a, _] => some ([], [(a, addrBytes)])
  | .tcpListen, [a] => some ([], [(a, addrBytes)])
  | .tcpLocal, [h, out] => some ([(.listener, h)], [(out, addrBytes)])
  | .tcpAccept, [l, _] => some ([(.listener, l)], [])
  | .tcpSend, [s, buf, len] => some ([(.stream, s)], [(buf, len.toNat)])
  | .tcpRecv, [s, buf, len] => some ([(.stream, s)], [(buf, len.toNat)])
  | .tcpShutdown, [s] => some ([(.stream, s)], [])
  | .tcpClose, [s] => some ([(.stream, s)], [])
  | .tcpListenerClose, [l] => some ([(.listener, l)], [])
  | .udpBind, [a, _] => some ([], [(a, addrBytes)])
  | .udpLocal, [u, out] => some ([(.udp, u)], [(out, addrBytes)])
  | .udpSend, [u, buf, len, a] => some ([(.udp, u)], [(buf, len.toNat), (a, addrBytes)])
  | .udpRecv, [u, buf, len] => some ([(.udp, u)], [(buf, len.toNat)])
  | .udpClose, [u] => some ([(.udp, u)], [])
  | _, _ => none

theorem netNeeds_length {f : NetFn} {bits : List UInt64} {n : List (Held × UInt64) × List (UInt64 × Nat)}
    (h : netNeeds f bits = some n) : bits.length = (Ext.net f).sig.1.length := by
  unfold netNeeds at h
  split at h <;> first | rfl | cases h

/-- The C strings a call reads: a window's title. -/
def extStrs : Ext → List UInt64 → List UInt64
  | .window .open, [t, _, _] => [t]
  | _, _ => []

/-- **What a call of the engine's own libraries needs of a typestate**: the
    handles it is handed, held, and room for each buffer it reads or writes,
    by address and length, with the C strings `extStrs` names; `none` for a
    call the typestate cannot vouch for. -/
def extNeeds : Ext → List UInt64 → Option (List (Held × UInt64) × List (UInt64 × Nat))
  | .present _, [] => some ([], [])
  | .c .memcpy, [d, s, n] =>
      if n.toNat = 0 ∨ s.toNat + n.toNat ≤ d.toNat ∨ d.toNat + n.toNat ≤ s.toNat
      then some ([], [(s, n.toNat), (d, n.toNat)]) else none
  | .c .memmove, [d, s, n] => some ([], [(s, n.toNat), (d, n.toNat)])
  | .c .memset, [d, _, n] => some ([], [(d, n.toNat)])
  | .c .calloc, [n, sz] => if 0 < n.toNat * sz.toNat then some ([], []) else none
  | .c .free, [p] => some ([(.heap, p)], [])
  | .window .init, [] | .window .pump, [] => some ([], [])
  | .window .open, [_, _, _] => some ([], [])
  | .window .close, [p] | .window .pixels, [p] => some ([(.window, p)], [])
  | .window .poll, [p, out] => some ([(.window, p)], [(out, 32)])
  | .cpu .count, [] | .cpu .unpin, [] => some ([], [])
  | .cpu .core, [_] | .cpu .package, [_] | .cpu .pin, [_] => some ([], [])
  | .base f, bits => baseNeeds f bits
  | .fs f, bits => fsNeeds f bits
  | .net f, bits => netNeeds f bits
  | .serial .count, [] => some ([], [])
  | .serial .name, [_, out, cap] => some ([], [(out, lenOf cap)])
  | .serial .close, [p] | .serial .pending, [p] => some ([(.port, p)], [])
  | .serial .read, [p, buf, len] | .serial .write, [p, buf, len] => some ([(.port, p)], [(buf, lenOf len)])
  | .usb .count, [] | .usb .info, [_, _] | .usb .open, [_] => some ([], [])
  | .usb .close, [d] | .usb .claim, [d, _] => some ([(.usbDevice, d)], [])
  | .usb .release, [i] => some ([(.usbInterface, i)], [])
  | .usb .control, [i, _, _, _, _, data, len, _] => some ([(.usbInterface, i)], [(data, len16 len)])
  | .usb .bulk, [i, _, data, len, _] | .usb .interrupt, [i, _, data, len, _] =>
      some ([(.usbInterface, i)], [(data, lenOf len)])
  | _, _ => none

theorem extNeeds_length {e : Ext} {bits : List UInt64} {n : List (Held × UInt64) × List (UInt64 × Nat)}
    (h : extNeeds e bits = some n) : bits.length = e.sig.1.length := by
  unfold extNeeds at h
  split at h <;> first | rfl | cases h | exact baseNeeds_length h | exact fsNeeds_length h | exact netNeeds_length h

theorem copyIn_zeros {w : World} {a : UInt64} {n : Nat} (h : Fits w.mem a n) :
    ∃ m, Sem.copyIn w.mem a (ByteArray.mk (Array.replicate n 0)) = some m :=
  Option.isSome_iff_exists.mp (h.writable (ByteArray.mk (Array.replicate n 0)) (Array.size_replicate ..))

theorem readBytes_some {w : World} {a : UInt64} {n : Nat} (h : Fits w.mem a n) :
    ∃ bs, readBytes w.mem a n = some bs :=
  Option.isSome_iff_exists.mp h.readable

theorem live_some {α : Type} {o : Option α} (h : o.isSome = true) : ∃ x, o = some x :=
  Option.isSome_iff_exists.mp h

/-- A held interface names a slot of the interfaces claimed. -/
theorem usbIface_get {w : World} {a : UInt64} {j : Nat} (h : w.usbIface? a = some j) :
    ∃ f, w.usbIfaces[j]? = some f := by
  unfold World.usbIface? at h
  have := List.find?_some h
  simp only [Bool.and_eq_true] at this
  cases e : w.usbIfaces[j]? with
  | none => rw [e] at this; simp at this
  | some f => exact ⟨f, rfl⟩

/-- A length the C side reads as an `int64_t`, not negative, is its own. -/
theorem lenOf_eq {x : UInt64} (h : negI64 x = false) : lenOf x = x.toNat := by
  unfold negI64 at h; unfold lenOf
  have h' : ¬ (x ≥ 0x8000000000000000) := by simpa using h
  rw [if_pos]
  rw [ge_iff_le, UInt64.le_iff_toNat_le] at h'
  rw [UInt64.lt_iff_toNat_lt]; omega

theorem negI64_false {b c : Bool} (h : ¬ ((b || c) = true)) : b = false := by
  cases b <;> simp_all

theorem fsNeeds_pre {f : FsFn} {bits : List UInt64} {w : World} {hs : List (Held × UInt64)}
    {rs : List (UInt64 × Nat)} (hn : fsNeeds f bits = some (hs, rs))
    (live : ∀ p ∈ hs, p.1.live w p.2 = true) (room : ∀ q ∈ rs, Fits w.mem q.1 q.2) :
    ExtContracts.fsPre f bits w := by
  have field : ∀ {p c : UInt64}, Fits w.mem p c.toNat → ∃ x, readField w.mem p c = some x :=
    fun h => live_some (readField_isSome h.readable)
  unfold fsNeeds at hn
  split at hn <;> (try cases hn)
  -- open
  · obtain ⟨x, hx⟩ := field (room _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [fsDecode, fsDOpen, fsPath, hx, Option.bind_eq_bind, Option.bind_some]
    repeat' split
    all_goals rfl
  -- close
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    exact ⟨rfl, by simp [fsDecode, fsDClose, hk]⟩
  -- read
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [fsDecode, fsDRead, hk, Option.bind_eq_bind, Option.bind_some]
    split
    · rfl
    · rename_i hh bb ll oo hneg
      have hr := room _ List.mem_cons_self
      obtain ⟨m, hm⟩ := copyIn_zeros
        (Fits.mono (n := ((w.fsContents k).extract oo.toNat (oo.toNat + ll.toNat)).size)
          (by rw [ByteArray.size_extract]; omega) hr)
      simp only [hm, Option.bind_some, Option.isSome_some]
  -- write
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [fsDecode, fsDWrite, hk, Option.bind_eq_bind, Option.bind_some]
    split
    · rfl
    · rename_i hneg
      have hr := room _ List.mem_cons_self
      obtain ⟨bs, hbs⟩ := readBytes_some hr
      simp [hbs]
  -- size
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    exact ⟨rfl, by simp [fsDecode, fsDSize, hk]⟩
  -- setLen
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [fsDecode, fsDSetLen, hk, Option.bind_eq_bind, Option.bind_some]
    split <;> rfl
  -- sync
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    exact ⟨rfl, by simp [fsDecode, fsDSync, hk]⟩
  -- createDir, removeFile, removeDir
  all_goals try
    (obtain ⟨x, hx⟩ := field (room _ List.mem_cons_self)
     refine ⟨rfl, ?_⟩
     simp only [fsDecode, fsDPathCall, fsDDirOpen, fsPath, hx, Option.bind_eq_bind, Option.bind_some]
     repeat' split
     all_goals rfl
     done)
  -- rename
  · obtain ⟨x, hx⟩ := field (room _ List.mem_cons_self)
    obtain ⟨y, hy⟩ := field (room _ (List.mem_cons_of_mem _ List.mem_cons_self))
    refine ⟨rfl, ?_⟩
    simp only [fsDecode, fsDRename, hx, hy, Option.bind_eq_bind, Option.bind_some]
    repeat' split
    all_goals rfl
  -- dirNext
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [fsDecode, fsDDirNext, hk, Option.bind_eq_bind, Option.bind_some]
    split
    · rename_i name _ _
      obtain ⟨m, hm⟩ := copyIn_zeros
        (Fits.mono (n := min name.utf8ByteSize (lenOf _))
          (Nat.le_trans (Nat.min_le_right _ _) (lenOf_le _)) (room _ List.mem_cons_self))
      simp [hm]
    · rfl
  -- dirClose
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    exact ⟨rfl, by simp [fsDecode, fsDDirClose, hk]⟩

theorem netNeeds_pre {f : NetFn} {bits : List UInt64} {w : World} {hs : List (Held × UInt64)}
    {rs : List (UInt64 × Nat)} (hn : netNeeds f bits = some (hs, rs))
    (live : ∀ p ∈ hs, p.1.live w p.2 = true) (room : ∀ q ∈ rs, Fits w.mem q.1 q.2) :
    ExtContracts.netPre f bits w := by
  have field : ∀ {p c : UInt64}, Fits w.mem p c.toNat → ∃ x, readField w.mem p c = some x :=
    fun h => live_some (readField_isSome h.readable)
  have addr : ∀ {a : UInt64}, Fits w.mem a addrBytes → ∃ x, addrAt w a = some x := fun h => by
    obtain ⟨bs, hbs⟩ := readBytes_some h
    exact ⟨NetAddr.ofBytes bs, by simp [addrAt, hbs]⟩
  unfold netNeeds at hn
  split at hn <;> (try cases hn)
  -- resolve
  · obtain ⟨x, hx⟩ := field (room _ List.mem_cons_self)
    have hr := room _ (List.mem_cons_of_mem _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [netDecode, netDResolve, hx, Option.bind_eq_bind, Option.bind_some]
    repeat' split
    all_goals try rfl
    rename_i as _
    obtain ⟨m, hm⟩ := copyIn_zeros (Fits.mono (n := addrBytes * min as.length (lenOf _ / addrBytes))
      (Nat.le_trans (Nat.le_trans (Nat.mul_le_mul_left _ (Nat.min_le_right _ _)) (Nat.mul_div_le _ _))
        (lenOf_le _)) hr)
    simp [hm]
  -- connect
  · obtain ⟨x, hx⟩ := addr (room _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [netDecode, netDTimed, hx, Option.bind_eq_bind, Option.bind_some]
    repeat' split
    all_goals rfl
  -- listen
  · obtain ⟨x, hx⟩ := addr (room _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [netDecode, netDListen, hx, Option.bind_eq_bind, Option.bind_some]
    split <;> rfl
  -- the listener's address
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    dsimp only at hk
    obtain ⟨m, hm⟩ := copyIn_zeros (room _ List.mem_cons_self)
    exact ⟨rfl, by simp [netDecode, netDLocal, hk, hm]⟩
  -- accept
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    dsimp only at hk
    refine ⟨rfl, ?_⟩
    simp only [netDecode, netDAccept, hk, Option.bind_eq_bind, Option.bind_some]
    split <;> rfl
  -- send
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    dsimp only at hk
    refine ⟨rfl, ?_⟩
    simp only [netDecode, netDSend, hk, Option.bind_eq_bind, Option.bind_some]
    split
    · rfl
    · rename_i hneg
      have hr := room _ List.mem_cons_self
      obtain ⟨bs, hbs⟩ := readBytes_some hr
      simp [hbs]
  -- receive
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    dsimp only at hk
    refine ⟨rfl, ?_⟩
    simp only [netDecode, netDRecv, hk, Option.bind_eq_bind, Option.bind_some]
    split
    · rfl
    · rename_i hneg
      split
      · rfl
      · have hr := room _ List.mem_cons_self
        obtain ⟨m, hm⟩ := copyIn_zeros hr
        simp [hm]
  -- shutdown, close, the listener's close
  all_goals try
    (obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
     dsimp only at hk
     refine ⟨rfl, ?_⟩
     simp [netDecode, netDHeld, hk]
     done)
  -- bind
  · obtain ⟨x, hx⟩ := addr (room _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [netDecode, netDTimed, hx, Option.bind_eq_bind, Option.bind_some]
    repeat' split
    all_goals rfl
  -- the UDP socket's address
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    dsimp only at hk
    obtain ⟨m, hm⟩ := copyIn_zeros (room _ List.mem_cons_self)
    exact ⟨rfl, by simp [netDecode, netDLocal, hk, hm]⟩
  -- UDP send
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    dsimp only at hk
    obtain ⟨x, hx⟩ := addr (room _ (List.mem_cons_of_mem _ List.mem_cons_self))
    refine ⟨rfl, ?_⟩
    simp only [netDecode, netDUdpSend, hk, hx, Option.bind_eq_bind, Option.bind_some]
    split
    · split
      · rfl
      · rename_i hneg
        have hr := room _ List.mem_cons_self
        obtain ⟨bs, hbs⟩ := readBytes_some hr
        simp [hbs]
    · rfl
  -- UDP receive
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    dsimp only at hk
    refine ⟨rfl, ?_⟩
    simp only [netDecode, netDUdpRecv, hk, Option.bind_eq_bind, Option.bind_some]
    split
    · rfl
    · rename_i hneg
      have hr := room _ List.mem_cons_self
      obtain ⟨m, hm⟩ := copyIn_zeros hr
      simp [hm]

/-- **What a call needs is enough**: the handles `extNeeds` names held, and
    the bytes it names in memory, give the call's precondition. -/
theorem extNeeds_pre {e : Ext} {bits : List UInt64} {w : World}
    {hs : List (Held × UInt64)} {rs : List (UInt64 × Nat)}
    (hn : extNeeds e bits = some (hs, rs))
    (live : ∀ p ∈ hs, p.1.live w p.2 = true) (room : ∀ q ∈ rs, Fits w.mem q.1 q.2)
    (strs : ∀ a ∈ extStrs e bits, CStr w.mem a) :
    ExtContracts.ExtPre e bits w := by
  intro _
  unfold extNeeds at hn
  split at hn <;> (try cases hn)
  -- whether a library is there
  · trivial
  -- C: memcpy, with its ranges apart
  · split at hn
    · rename_i hdis
      cases hn
      exact ⟨Fits.readable (room _ List.mem_cons_self),
        Fits.writable (room _ (List.mem_cons_of_mem _ List.mem_cons_self)), hdis⟩
    · cases hn
  -- C: memmove, memset
  · exact ⟨Fits.readable (room _ List.mem_cons_self),
      Fits.writable (room _ (List.mem_cons_of_mem _ List.mem_cons_self))⟩
  · exact Fits.writable (room _ List.mem_cons_self)
  -- C: calloc, of a size that is not zero
  · split at hn
    · rename_i hz; cases hn; exact hz
    · cases hn
  -- C: free, of a block held
  · exact .inr (live _ List.mem_cons_self)
  -- window: init, pump
  · exact ⟨rfl, rfl⟩
  · exact ⟨rfl, rfl⟩
  -- window: open, its title a C string
  · obtain ⟨c, hc⟩ := Option.isSome_iff_exists.mp (strs _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [wlDecode, wlDOpen]
    split
    · rfl
    · simp only [hc, Option.bind_eq_bind, Option.bind_some]
      split <;> rfl
  -- window: close, pixels
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    exact ⟨rfl, by simp [wlDecode, wlDClose, hk]⟩
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    exact ⟨rfl, by simp [wlDecode, wlDPixels, hk]⟩
  -- window: poll
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    obtain ⟨m, hm⟩ := copyIn_zeros (room _ List.mem_cons_self)
    exact ⟨rfl, by simp [wlDecode, wlDPoll, hk, hm]⟩
  -- cpu
  all_goals try exact rfl
  -- the engine's own
  · exact baseNeeds_pre hn room
  · exact fsNeeds_pre hn live room
  · exact netNeeds_pre hn live room
  -- serial: count
  · exact ⟨rfl, rfl⟩
  -- serial: name
  · refine ⟨rfl, ?_⟩
    simp only [serDecode, serDName]
    split
    · rfl
    · rename_i name _
      obtain ⟨m, hm⟩ := copyIn_zeros
        (Fits.mono (n := min name.utf8ByteSize _) (Nat.min_le_right _ _) (room _ List.mem_cons_self))
      simp [hm]
  -- serial: close, pending
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    exact ⟨rfl, by simp [serDecode, serDClose, hk]⟩
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    exact ⟨rfl, by simp [serDecode, serDPending, hk]⟩
  -- serial: read, write
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    obtain ⟨m, hm⟩ := copyIn_zeros (room _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [serDecode, serDRead, hk, Option.bind_eq_bind, Option.bind_some]
    split
    · rfl
    · simp [hm]
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    obtain ⟨bs, hbs⟩ := readBytes_some (room _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [serDecode, serDWrite, hk, Option.bind_eq_bind, Option.bind_some]
    split
    · rfl
    · simp [hbs]
  -- usb: count, info, open
  · exact ⟨rfl, rfl⟩
  · exact ⟨rfl, rfl⟩
  · refine ⟨rfl, ?_⟩
    simp only [usbDecode, usbDOpen]
    split
    · split <;> rfl
    · rfl
  -- usb: close, claim, release
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    exact ⟨rfl, by simp [usbDecode, usbDClose, hk]⟩
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [usbDecode, usbDClaim, hk, Option.bind_eq_bind, Option.bind_some]
    repeat' split
    all_goals rfl
  · obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    exact ⟨rfl, by simp [usbDecode, usbDRelease, hk]⟩
  -- usb: control, bulk, interrupt
  all_goals
    obtain ⟨k, hk⟩ := live_some (live _ List.mem_cons_self)
    obtain ⟨f, hf⟩ := usbIface_get hk
    obtain ⟨m, hm⟩ := copyIn_zeros (room _ List.mem_cons_self)
    obtain ⟨bs, hbs⟩ := readBytes_some (room _ List.mem_cons_self)
    refine ⟨rfl, ?_⟩
    simp only [usbDecode, usbDControl, usbDData, World.usbIfaceOf?, hk, hf, Option.bind_eq_bind, Option.bind_some]
    repeat' split
    all_goals simp_all

/-- The same, from a typestate holding the handles and giving the room. -/
theorem extNeeds_sound {e : Ext} {bits : List UInt64} {S : TState} {w : World}
    {hs : List (Held × UInt64)} {rs : List (UInt64 × Nat)}
    (hn : extNeeds e bits = some (hs, rs)) (hS : S.holds w)
    (hh : ∀ p ∈ hs, Fact.held p.1 p.2 ∈ S) (hr : rs.all (fun q => roomAt S q.1 q.2) = true)
    (strs : ∀ a ∈ extStrs e bits, CStr w.mem a) :
    ExtContracts.ExtPre e bits w :=
  extNeeds_pre hn (fun p hp => hS _ (hh p hp)) (fun q hq => roomAt_spec hS (List.all_eq_true.mp hr q hq)) strs

/-- A library call the typestate follows keeps every lifecycle part. -/
theorem LibStep.parts {e : Ext} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (st : LibStep e bits w r w') (p : Part) : p.get w' = p.get w := by
  cases p <;> simp only [Part.get, st.dev, st.display, st.cudaDevice,
    st.gpuAdapter, st.mem.2.2, World.has, st.present]

end AlgorithmLib.HProg.Contracts
