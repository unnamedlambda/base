module
public import AlgorithmLib.Check.Domain
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.Calls` — what a C library call needs, decided

A call's precondition (`ExtContracts.ExtPre`) is a proposition about the
world; the checker needs a decision. `preExt` decides it from what the checker
knows of the arguments and the typestate, by *atoms* each proven once: bytes
in memory the typestate gives room for (`memN`, `memB`), a C string
(`cstrB`), a lifecycle part (`partB`), a handle held (`handB`), and the
device's own facts. **`preExt_sound`**: where `preExt` accepts, `ExtPre` holds
of the bits the arguments carry.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Prog

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState readyIn streamsOf eventsOf partyOf evId roomAt roomAt_spec holds_of_mem
  Part)

-- ---------------------------------------------------------------------------
-- Arguments
-- ---------------------------------------------------------------------------

/-- Each abstract value describes the value in its place. -/
def DescAll (Γ : Env) (w : World) : List AV → List V → Prop
  | [], [] => True
  | a :: as, v :: vs => a.Desc Γ w v ∧ DescAll Γ w as vs
  | _, _ => False

/-- Each abstract value describes a scalar with the bits in its place. -/
def ArgBits (Γ : Env) (w : World) : List AV → List UInt64 → Prop
  | [], [] => True
  | a :: as, x :: xs => (∃ t, a.Desc Γ w (.sc t x)) ∧ ArgBits Γ w as xs
  | _, _ => False

theorem ArgBits.cons_inv {Γ : Env} {w : World} {a : AV} {as : List AV} {bits : List UInt64}
    (h : ArgBits Γ w (a :: as) bits) :
    ∃ x xs, bits = x :: xs ∧ (∃ t, a.Desc Γ w (.sc t x)) ∧ ArgBits Γ w as xs := by
  match bits, h with
  | x :: xs, ⟨hx, hxs⟩ => exact ⟨x, xs, rfl, hx, hxs⟩

theorem ArgBits.nil_inv {Γ : Env} {w : World} {bits : List UInt64} (h : ArgBits Γ w [] bits) : bits = [] := by
  match bits, h with
  | [], _ => rfl

theorem argBits_of {Γ : Env} {w : World} :
    ∀ {as : List AV} {vs : List V} {bits : List UInt64}, DescAll Γ w as vs →
      vs.mapM asBits = some bits → ArgBits Γ w as bits
  | [], [], bits, _, hb => by simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at hb; subst hb; trivial
  | [], _ :: _, _, h, _ => h.elim
  | _ :: _, [], _, h, _ => h.elim
  | a :: as, v :: vs, bits, ⟨hd, hds⟩, hb => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at hb
      obtain ⟨x, hx, xs, hxs, rfl⟩ := hb
      cases v with
      | sc t y => simp only [asBits, Option.some.injEq] at hx; subst hx; exact ⟨⟨t, hd⟩, argBits_of hds hxs⟩
      | vec _ _ => cases hx

theorem bits_exist {Γ : Env} {w : World} :
    ∀ {as : List AV} {vs : List V}, DescAll Γ w as vs → as.all (fun a => a.scTy.isSome) = true →
      ∃ bits, vs.mapM asBits = some bits
  | [], [], _, _ => ⟨[], rfl⟩
  | [], _ :: _, h, _ => h.elim
  | _ :: _, [], h, _ => h.elim
  | a :: as, v :: vs, ⟨hd, hds⟩, hs => by
      simp only [List.all_cons, Bool.and_eq_true] at hs
      obtain ⟨t, ht⟩ := Option.isSome_iff_exists.mp hs.1
      obtain ⟨x, rfl⟩ := AV.scTy_desc hd ht
      obtain ⟨bits, hb⟩ := bits_exist hds hs.2
      exact ⟨x :: bits, by simp [List.mapM_cons, asBits, hb]⟩

theorem known_bits {Γ : Env} {w : World} {a : AV} {x k : UInt64} (hd : ∃ t, a.Desc Γ w (.sc t x))
    (h : a.bits = some k) : x = k := by
  obtain ⟨t, hd⟩ := hd
  obtain ⟨t', e⟩ := AV.bits_desc hd h
  cases e; rfl

-- ---------------------------------------------------------------------------
-- Atoms
-- ---------------------------------------------------------------------------

theorem memN_readable {Γ : Env} {w : World} {S : TState} {p : AV} {x : UInt64} {k : Nat} (hS : S.holds w)
    (hd : ∃ t, p.Desc Γ w (.sc t x)) (h : memN S p k = true) : Readable w.mem x k := by
  obtain ⟨r, off, hdec, hp, hs, hn⟩ := memN_spec hS hd h
  exact Contracts.copyOut_isSome hdec hp hs hn

theorem memN_writable {Γ : Env} {w : World} {S : TState} {p : AV} {x : UInt64} {k : Nat} (hS : S.holds w)
    (hd : ∃ t, p.Desc Γ w (.sc t x)) (h : memN S p k = true) : Writable w.mem x k := by
  obtain ⟨r, off, hdec, hp, hs, hn⟩ := memN_spec hS hd h
  intro src hsrc; subst hsrc
  exact Contracts.copyIn_isSome src hdec hp hs hn

theorem roomIn_mono {S : TState} {r : Region} {k n : Nat} (h : roomIn S r k = true) (hn : n ≤ k) :
    roomIn S r n = true := by
  unfold roomIn at h ⊢
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.any_eq_true] at h ⊢
  obtain ⟨hp, x, hx, hk⟩ := h
  refine ⟨hp, x, hx, ?_⟩
  cases x <;> simp_all
  omega

theorem memN_mono {S : TState} {p : AV} {k n : Nat} (h : memN S p k = true) (hn : n ≤ k) : memN S p n = true := by
  unfold memN at h ⊢
  split at h
  · exact Contracts.roomAt_mono h hn
  · rename_i r m
    simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true] at h ⊢
    obtain ⟨⟨hs, hlt⟩, hr⟩ := h
    refine ⟨⟨by omega, hlt⟩, ?_⟩
    rcases hr with hr | hr
    · exact .inl (roomIn_mono hr (by omega))
    · refine .inr ?_
      split at hr
      · rename_i r' d hfit
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hr ⊢
        exact ⟨⟨hr.1.1, by omega⟩, hr.2⟩
      · cases hr
  · cases h

theorem memN_slot8 {Γ : Env} {w : World} {S : TState} {p : AV} {x : UInt64} (hS : S.holds w)
    (hd : ∃ t, p.Desc Γ w (.sc t x)) (h : memN S p 8 = true) : Contracts.Slot8 w.mem x := by
  obtain ⟨r, off, hdec, hp, hs, -⟩ := memN_spec hS hd h
  exact fun v => Contracts.store_isSome hdec hp hs v

/-- A length bounded by the region an address points into, no further than
    the bound leaves room for. -/
def memLen (S : TState) (p n : AV) : Bool :=
  match p.asPtr, n.num with
  | .ptr r m, some nn =>
      match nn.fit with
      | some (r', s, d) => decide (r = r') && decide (s = 1) && decide (m.hi ≤ d) &&
          decide (m.hi + nn.hi ≤ span) && decide (m.hi < span)
      | none => false
  | _, _ => false

/-- Bytes `[p, p + n)` with `n` an argument: known, a whole caller length at
    the start of its region, or no larger than a number that fits. -/
def memB (S : TState) (p n : AV) : Bool :=
  match n.bits with
  | some k => memN S p k.toNat
  | none => memLen S p n || (match n.num with | some nn => memN S p nn.hi | none => false)

theorem memLen_spec {Γ : Env} {w : World} {S : TState} {p n : AV} {x k : UInt64} (hS : S.holds w)
    (hp : ∃ t, p.Desc Γ w (.sc t x)) (hn : ∃ t, n.Desc Γ w (.sc t k)) (h : memLen S p n = true) :
    ∃ r off, decodeAddr x = some (r, off) ∧ r ≠ .pinned ∧ off + k.toNat ≤ w.mem.sizes r ∧
      off + k.toNat ≤ 2 ^ 36 := by
  obtain ⟨t0, hp0⟩ := hp
  have hp : ∃ t, p.asPtr.Desc Γ w (.sc t x) := ⟨t0, AV.asPtr_desc hp0⟩
  unfold memLen at h
  split at h
  · rename_i r m nn hpe hne
    rw [hpe] at hp
    split at h
    · rename_i r' s d hf
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h
      obtain ⟨⟨⟨⟨rfl, rfl⟩, hd⟩, hs⟩, h3⟩ := h
      obtain ⟨t, off, e, ⟨h1, h2, -⟩, -⟩ := hp
      simp only [V.sc.injEq] at e
      obtain ⟨-, rfl⟩ := e
      obtain ⟨t', hn⟩ := hn
      obtain ⟨y, e', hy⟩ := AV.num_desc hn hne
      simp only [V.sc.injEq] at e'
      obtain ⟨-, rfl⟩ := e'
      obtain ⟨-, hy2, -, hfit⟩ := hy
      rw [hf] at hfit
      obtain ⟨hpin, hle⟩ := hfit
      exact ⟨r, off, ptr_decode h2 h3, hpin, by omega, by unfold span at hs; omega⟩
    · cases h
  · cases h

theorem memB_spec {Γ : Env} {w : World} {S : TState} {p n : AV} {x k : UInt64} (hS : S.holds w)
    (hp : ∃ t, p.Desc Γ w (.sc t x)) (hn : ∃ t, n.Desc Γ w (.sc t k)) (h : memB S p n = true) :
    ∃ r off, decodeAddr x = some (r, off) ∧ r ≠ .pinned ∧ off + k.toNat ≤ w.mem.sizes r ∧
      off + k.toNat ≤ 2 ^ 36 := by
  unfold memB at h
  split at h
  · rename_i k' hk
    rw [known_bits hn hk]
    exact memN_spec hS hp h
  · rcases Bool.or_eq_true_iff.mp h with h | h
    · exact memLen_spec hS hp hn h
    · split at h
      · rename_i nn hnn
        obtain ⟨t, hd⟩ := hn
        obtain ⟨y, e, hy⟩ := AV.num_desc hd hnn
        cases e
        obtain ⟨r, off, hdec, hpin, hs, h36⟩ := memN_spec hS hp h
        exact ⟨r, off, hdec, hpin, by have := hy.2.1; omega, by have := hy.2.1; omega⟩
      · cases h

/-- Bytes `[p, p + n)`: by `memB`, or by `rb`, which answers for the
    arguments at `i` and `j`. -/
def memBR (S : TState) (rb : Nat → Nat → Bool) (i j : Nat) (p n : AV) : Bool := memB S p n || rb i j

/-- What `rb` answers is true of the arguments' bits. -/
def RbSound (w : World) (bits : List UInt64) (rb : Nat → Nat → Bool) : Prop :=
  ∀ i j, rb i j = true → Fits w.mem (bits.getD i 0) (bits.getD j 0).toNat

theorem memBR_fits {Γ : Env} {w : World} {S : TState} {rb : Nat → Nat → Bool} {i j : Nat} {p n : AV}
    {x k : UInt64} (hS : S.holds w) (hp : ∃ t, p.Desc Γ w (.sc t x)) (hn : ∃ t, n.Desc Γ w (.sc t k))
    (h : memBR S rb i j p n = true) (hrb : rb i j = true → Fits w.mem x k.toNat) : Fits w.mem x k.toNat := by
  rcases Bool.or_eq_true_iff.mp h with h | h
  · exact memB_spec hS hp hn h
  · exact hrb h

theorem memB_readable {Γ : Env} {w : World} {S : TState} {p n : AV} {x k : UInt64} (hS : S.holds w)
    (hp : ∃ t, p.Desc Γ w (.sc t x)) (hn : ∃ t, n.Desc Γ w (.sc t k)) (h : memB S p n = true) :
    Readable w.mem x k.toNat := by
  obtain ⟨r, off, hdec, hpin, hs, h36⟩ := memB_spec hS hp hn h
  exact Contracts.copyOut_isSome hdec hpin hs h36

theorem memB_writable {Γ : Env} {w : World} {S : TState} {p n : AV} {x k : UInt64} (hS : S.holds w)
    (hp : ∃ t, p.Desc Γ w (.sc t x)) (hn : ∃ t, n.Desc Γ w (.sc t k)) (h : memB S p n = true) :
    Writable w.mem x k.toNat := by
  obtain ⟨r, off, hdec, hpin, hs, h36⟩ := memB_spec hS hp hn h
  intro src hsrc
  exact Contracts.copyIn_isSome src hdec hpin (by rw [hsrc]; exact hs) (by rw [hsrc]; exact h36)

/-- A C string the typestate names at a known address. -/
def cstrB (S : TState) (a : AV) : Bool :=
  match a.bits with
  | some x => S.contains (.cstr x) || S.any fun | .cstrIn a' _ => a' == x | _ => false
  | none => false

theorem cstrB_sound {Γ : Env} {w : World} {S : TState} {a : AV} {x : UInt64} (hS : S.holds w)
    (hd : ∃ t, a.Desc Γ w (.sc t x)) (h : cstrB S a = true) : Contracts.CStr w.mem x := by
  unfold cstrB at h
  split at h
  · rename_i y hy
    rw [known_bits hd hy]
    rcases Bool.or_eq_true_iff.mp h with h | h
    · exact Contracts.cstr_of_state hS h
    · exact Contracts.cstrIn_of_state hS h
  · cases h

theorem partB_sound {w : World} {S : TState} {p : Part} {b : Bool} (hS : S.holds w)
    (h : S.contains (.part p b) = true) : p.get w = b := holds_of_mem hS h

-- ---------------------------------------------------------------------------
-- What a call leaves
-- ---------------------------------------------------------------------------

/-- The argument positions a frame reads. -/
def frPos : Frame → List Nat
  | .none | .anywhere => []
  | .at d l => [d, l]
  | .atOff b o l => [b, o, l]
  | .fixed d _ => [d]
  | .fixed2 a b _ => [a, b]
  | .atRecords d c _ => [d, c]
  | .dataDependent d => [d]
  | .dataDependent2 a b => [a, b]

theorem mayWrite_congr {fr : Frame} {args args' : List UInt64}
    (h : ∀ k ∈ frPos fr, args.getD k 0 = args'.getD k 0) (x : UInt64) :
    fr.mayWrite args x = fr.mayWrite args' x := by
  cases fr <;> simp_all [Frame.mayWrite, frPos]

theorem touches_congr {fr : Frame} {args args' : List UInt64}
    (h : ∀ k ∈ frPos fr, args.getD k 0 = args'.getD k 0) (a : UInt64) (n : Nat) :
    fr.touches args a n = fr.touches args' a n := by
  cases fr <;> simp_all [Frame.touches, frPos, mayWrite_congr h]

theorem argBits_getD {Γ : Env} {w : World} : ∀ {as : List AV} {bits : List UInt64} {k : Nat} {y : UInt64},
    ArgBits Γ w as bits → (as.getD k .any).bits = some y → bits.getD k 0 = y
  | [], [], k, y, _, h => by simp [AV.bits] at h
  | [], _ :: _, _, _, h, _ => h.elim
  | _ :: _, [], _, _, h, _ => h.elim
  | a :: as, x :: xs, 0, y, ⟨hd, _⟩, h => by simpa using known_bits hd h
  | a :: as, x :: xs, k + 1, y, ⟨_, hds⟩, h => by simpa using argBits_getD hds h

/-- At the positions whose arguments are known, the bits they have are the
    bits read off their descriptions. -/
theorem knownAt_getD {Γ : Env} {w : World} {as : List AV} {bits : List UInt64} {ps : List Nat}
    (hb : ArgBits Γ w as bits) (hall : ps.all (fun k => (as.getD k .any).bits.isSome) = true) :
    ∀ k ∈ ps, bits.getD k 0 = (as.map (·.bits.getD 0)).getD k 0 := fun k hk' => by
  obtain ⟨y, hy⟩ := Option.isSome_iff_exists.mp (List.all_eq_true.mp hall k hk')
  rw [argBits_getD hb hy]
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  rw [List.getD_eq_getElem?_getD] at hy
  cases e : as[k]? with
  | none => rw [e] at hy; cases hy
  | some a => rw [e] at hy; simp only [Option.getD_some] at hy; simp [hy]

-- ---------------------------------------------------------------------------
-- Entry points
-- ---------------------------------------------------------------------------

/-- `gemmStridedFits`, by the three buffers' sizes. -/
def gemmStridedFitsN (ea eb ec : Nat) (ta tb m n k sa sb sc batch oa ob oc la lb lc : UInt64)
    (A B C : Nat) : Bool :=
  let M := (asI32 m).toNat
  let N := (asI32 n).toNat
  let K := (asI32 k).toNat
  let ra := if asI32 ta != 0 then K else M
  let ca := if asI32 ta != 0 then M else K
  let rb := if asI32 tb != 0 then N else K
  let cb := if asI32 tb != 0 then K else N
  let lda := if asI32 la != 0 then (asI32 la).toNat else ra
  let ldb := if asI32 lb != 0 then (asI32 lb).toNat else rb
  let ldc := if asI32 lc != 0 then (asI32 lc).toNat else M
  let last := (asI32 batch).toNat - 1
  let fits (e off stride span : Nat) (size : Nat) : Bool :=
    decide (e * (off + last * stride + span) ≤ size)
  decide (ra ≤ lda) && decide (rb ≤ ldb) && decide (M ≤ ldc)
    && fits ea (asI64 oa).toNat (asI64 sa).toNat (colMajorSpan ra ca lda) A
    && fits eb (asI64 ob).toNat (asI64 sb).toNat (colMajorSpan rb cb ldb) B
    && fits ec (asI64 oc).toNat (asI64 sc).toNat (colMajorSpan M N ldc) C

theorem gemmStridedFits_eq (ea eb ec : Nat) (ta tb m n k sa sb sc batch oa ob oc la lb lc : UInt64)
    (A B C : ByteArray) : gemmStridedFits ea eb ec ta tb m n k sa sb sc batch oa ob oc la lb lc A B C =
      gemmStridedFitsN ea eb ec ta tb m n k sa sb sc batch oa ob oc la lb lc A.size B.size C.size := rfl

/-- The bounds the typestate gives on the table at slot `a`: on its values,
    and on how many times it was made. -/
def htAtOf (a : UInt64) : TState → Option (Nat × Nat)
  | [] => none
  | x :: S => match x with
    | .htAt c n k => if c = a then some (n, k) else htAtOf a S
    | _ => htAtOf a S

theorem htAtOf_mem {a : UInt64} : ∀ {S : TState} {n k : Nat}, htAtOf a S = some (n, k) → Fact.htAt a n k ∈ S
  | [], _, _, h => by cases h
  | x :: S, n, k, h => by
      unfold htAtOf at h
      split at h
      · rename_i c m j
        split at h
        · rename_i e; subst e; cases h; exact List.mem_cons_self
        · exact List.mem_cons_of_mem _ (htAtOf_mem h)
      · exact List.mem_cons_of_mem _ (htAtOf_mem h)

/-- A handle of kind `k` held, by its description. -/
def handB (k : Contracts.Held) : AV → Bool
  | .hand k' true => k' == k
  | .blk _ true => k == .heap
  | _ => false

/-- A wait's word: four bytes of the program's memory, at an address the
    checker knows to be a multiple of four. -/
def wordB (S : TState) (p : AV) : Bool :=
  memN S p 4 && (match p with | .ptr .arena n => n.al % 4 == 0 | _ => false)

/-- What an engine function needs, decided: its arguments, and a wait's or
    a wake's word. -/
def preBase (S : TState) (rb : Nat → Nat → Bool) : BaseFn → List AV → Bool
  | .clockNow, [] => true
  | .memLock, [_, _] | .memUnlock, [_, _] | .memAdviseHuge, [_, _] => true
  | .threadPriority, [_] => true
  | .memWait, [p, _, _] | .memWake, [p, _] => wordB S p
  | .nativeArch, [] => true
  | .nativeLoad, [s, n] => match n.bits with
      | some k => memN S s (asI64 k).toNat
      | none => false
  | .cpuHas, [p, c] | .fileCreateDirAll, [p, c] | .fileSize, [p, c] | .fileSetLen, [p, c, _]
  | .stdoutWrite, [p, c] | .stdinReadline, [p, c] => memBR S rb 0 1 p c
  | .fileReadToPtr, [pp, pc, s, _, n] | .fileWriteFromPtr, [pp, pc, s, _, n] =>
      memBR S rb 0 1 pp pc && memBR S rb 2 4 s n
  | _, _ => false

/-- **What a call of the engine's own libraries needs, decided** by the
    arguments it is handed, as `extNeeds` asks it: each handle held, and room
    for each buffer it reads or writes, by a length known. -/
def preExt (S : TState) (rb : Nat → Nat → Bool := fun _ _ => false) : Ext → List AV → Bool
  | .present _, [] => true
  | .c .memcpy, [d, s, n] =>
      match d.bits, s.bits, n.bits with
      | some d', some s', some n' =>
          decide (n'.toNat = 0 ∨ s'.toNat + n'.toNat ≤ d'.toNat ∨ d'.toNat + n'.toNat ≤ s'.toNat) &&
            memN S s n'.toNat && memN S d n'.toNat
      | _, _, _ => false
  | .c .memmove, [d, s, n] => match n.bits with
      | some n' => memN S s n'.toNat && memN S d n'.toNat
      | none => false
  | .c .memset, [d, _, n] => match n.bits with
      | some n' => memN S d n'.toNat
      | none => false
  | .c .calloc, [n, sz] => match n.bits, sz.bits with
      | some a, some b => decide (0 < a.toNat * b.toNat)
      | _, _ => false
  | .c .free, [p] => handB .heap p
  | .window .init, [] | .window .pump, [] => true
  | .window .open, [t, _, _] => cstrB S t
  | .window .close, [p] | .window .pixels, [p] => handB .window p
  | .window .poll, [p, out] => handB .window p && memN S out 32
  | .cpu .count, [] | .cpu .unpin, [] => true
  | .cpu .core, [_] | .cpu .package, [_] | .cpu .pin, [_] => true
  | .base f, as => preBase S rb f as
  | .serial .count, [] => true
  | .serial .name, [_, out, cap] => match cap.bits with
      | some c => memN S out (lenOf c)
      | none => false
  | .serial .close, [p] | .serial .pending, [p] => handB .port p
  | .serial .read, [p, buf, len] | .serial .write, [p, buf, len] =>
      handB .port p && match len.bits with
        | some l => memN S buf (lenOf l)
        | none => false
  | .usb .count, [] | .usb .info, [_, _] | .usb .open, [_] => true
  | .usb .close, [d] | .usb .claim, [d, _] => handB .usbDevice d
  | .usb .release, [i] => handB .usbInterface i
  | .usb .control, [i, _, _, _, _, data, len, _] =>
      handB .usbInterface i && match len.bits with
        | some l => memN S data (len16 l)
        | none => false
  | .usb .bulk, [i, _, data, len, _] | .usb .interrupt, [i, _, data, len, _] =>
      handB .usbInterface i && match len.bits with
        | some l => memN S data (lenOf l)
        | none => false
  | _, _ => false

/-- Whether a library call with these arguments keeps a fact: as `extKeeps`
    says, where a cell or a string reads the arguments' bits only when they
    are known. -/
def extKeepsA (e : Ext) (as : List AV) (f : Fact) : Bool :=
  match f with
  | .cell .. | .cstrIn .. | .wgText .. =>
      (frPos e.frame).all (fun k => (as.getD k .any).bits.isSome) &&
        Contracts.extKeeps e (as.map (·.bits.getD 0)) f
  | _ => Contracts.extKeeps e [] f

/-- Whether a library call keeps a fact reads only the arguments its frame
    names. -/
theorem extKeeps_congr {e : Ext} {bits bits' : List UInt64} (f : Fact)
    (h : ∀ k ∈ frPos e.frame, bits.getD k 0 = bits'.getD k 0) :
    Contracts.extKeeps e bits f = Contracts.extKeeps e bits' f := by
  cases f <;> first | rfl | simp only [Contracts.extKeeps, mayWrite_congr h, touches_congr h]

/-- `gemmBf16Fits` on buffer sizes. -/
def gemmBf16FitsN (ta tb m n k oa ob oc la lb lc : UInt64) (A B C : Nat) : Bool :=
  let M := (asI32 m).toNat
  let N := (asI32 n).toNat
  let K := (asI32 k).toNat
  let ra := if asI32 ta != 0 then K else M
  let ca := if asI32 ta != 0 then M else K
  let rb := if asI32 tb != 0 then N else K
  let cb := if asI32 tb != 0 then K else N
  let lda := if asI32 la != 0 then (asI32 la).toNat else ra
  let ldb := if asI32 lb != 0 then (asI32 lb).toNat else rb
  let ldc := if asI32 lc != 0 then (asI32 lc).toNat else M
  decide (ra ≤ lda) && decide (rb ≤ ldb) && decide (M ≤ ldc)
    && decide (2 * ((asI64 oa).toNat + colMajorSpan ra ca lda) ≤ A)
    && decide (2 * ((asI64 ob).toNat + colMajorSpan rb cb ldb) ≤ B)
    && decide (4 * ((asI64 oc).toNat + colMajorSpan M N ldc) ≤ C)

theorem gemmBf16Fits_eq (ta tb m n k oa ob oc la lb lc : UInt64) (A B C : ByteArray) :
    gemmBf16Fits ta tb m n k oa ob oc la lb lc A B C =
      gemmBf16FitsN ta tb m n k oa ob oc la lb lc A.size B.size C.size := rfl


end AlgorithmLib.Check
