module
public import AlgorithmLib.Check.Domain
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.Calls` — what a foreign call needs, decided

A call's precondition (`Contracts.Pre`) is a proposition about the world; the
checker needs a decision. `preB` decides it from what the checker knows of the
arguments and the typestate, by *atoms* each proven once: a context handle
that is null or live (`ctxB`), bytes in memory the typestate gives room for
(`memN`, `memB`), a C string or a path (`cstrB`, `pathB`), a lifecycle part
(`partB`), and the device's own facts. **`preB_sound`**: where `preB` accepts,
`Pre` holds of the bits the arguments carry.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Prog

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState after after_sound keeps keeps_sound devArgFree devAfter devAfter_free devAfter_sound readyIn streamsOf eventsOf partyOf evId roomAt roomAt_spec holds_of_mem
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

/-- A handle that is null, or the live one of a context. -/
def ctxB (S : TState) (p : Part) (H : UInt64) (a : AV) : Bool :=
  match a.bits with
  | some c => c == 0 || (c == H && S.contains (.part p true))
  | none => false

theorem ctxB_sound {Γ : Env} {w : World} {S : TState} {p : Part} {H x : UInt64} {a : AV} (hS : S.holds w)
    (hd : ∃ t, a.Desc Γ w (.sc t x)) (h : ctxB S p H a = true) : x = 0 ∨ (x = H ∧ p.get w = true) := by
  unfold ctxB at h
  split at h
  · rename_i c hc
    rw [known_bits hd hc]
    simp only [Bool.or_eq_true, beq_iff_eq, Bool.and_eq_true] at h
    rcases h with h | ⟨h1, h2⟩
    · exact .inl h
    · exact .inr ⟨h1, holds_of_mem hS h2⟩
  · cases h

theorem memN_readable {Γ : Env} {w : World} {S : TState} {p : AV} {x : UInt64} {k : Nat} (hS : S.holds w)
    (hd : ∃ t, p.Desc Γ w (.sc t x)) (h : memN S p k = true) : Readable w.mem x k := by
  obtain ⟨r, off, hdec, hp, hs, hz, hn⟩ := memN_spec hS hd h
  exact Contracts.copyOut_isSome hdec hp hs hz hn

theorem memN_writable {Γ : Env} {w : World} {S : TState} {p : AV} {x : UInt64} {k : Nat} (hS : S.holds w)
    (hd : ∃ t, p.Desc Γ w (.sc t x)) (h : memN S p k = true) : Writable w.mem x k := by
  obtain ⟨r, off, hdec, hp, hs, hz, hn⟩ := memN_spec hS hd h
  intro src hsrc; subst hsrc
  exact Contracts.copyIn_isSome src hdec hp hs hz hn

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
        exact ⟨⟨⟨hr.1.1.1, by omega⟩, hr.1.2⟩, hr.2⟩
      · cases hr
  · cases h

theorem memN_slot8 {Γ : Env} {w : World} {S : TState} {p : AV} {x : UInt64} (hS : S.holds w)
    (hd : ∃ t, p.Desc Γ w (.sc t x)) (h : memN S p 8 = true) : Contracts.Slot8 w.mem x := by
  obtain ⟨r, off, hdec, hp, hs, hz, -⟩ := memN_spec hS hd h
  exact fun v => Contracts.store_isSome hdec hp hs hz v

/-- A length bounded by the region an address points into, no further than
    the bound leaves room for. -/
def memLen (S : TState) (p n : AV) : Bool :=
  match p.asPtr, n.num with
  | .ptr r m, some nn =>
      match nn.fit with
      | some (r', s, d) => decide (r = r') && decide (s = 1) && decide (m.hi ≤ d) &&
          decide (m.hi + nn.hi ≤ span) && decide (m.hi < span) && S.contains (.part .frozen false)
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
      w.mem.frozen = false ∧ off + k.toNat ≤ 2 ^ 36 := by
  obtain ⟨t0, hp0⟩ := hp
  have hp : ∃ t, p.asPtr.Desc Γ w (.sc t x) := ⟨t0, AV.asPtr_desc hp0⟩
  unfold memLen at h
  split at h
  · rename_i r m nn hpe hne
    rw [hpe] at hp
    split at h
    · rename_i r' s d hf
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h
      obtain ⟨⟨⟨⟨⟨rfl, rfl⟩, hd⟩, hs⟩, h3⟩, hz⟩ := h
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
      exact ⟨r, off, ptr_decode h2 h3, hpin, by omega, holds_of_mem hS hz, by unfold span at hs; omega⟩
    · cases h
  · cases h

theorem memB_spec {Γ : Env} {w : World} {S : TState} {p n : AV} {x k : UInt64} (hS : S.holds w)
    (hp : ∃ t, p.Desc Γ w (.sc t x)) (hn : ∃ t, n.Desc Γ w (.sc t k)) (h : memB S p n = true) :
    ∃ r off, decodeAddr x = some (r, off) ∧ r ≠ .pinned ∧ off + k.toNat ≤ w.mem.sizes r ∧
      w.mem.frozen = false ∧ off + k.toNat ≤ 2 ^ 36 := by
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
        obtain ⟨r, off, hdec, hpin, hs, hz, h36⟩ := memN_spec hS hp h
        exact ⟨r, off, hdec, hpin, by have := hy.2.1; omega, hz, by have := hy.2.1; omega⟩
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
  obtain ⟨r, off, hdec, hpin, hs, hz, h36⟩ := memB_spec hS hp hn h
  exact Contracts.copyOut_isSome hdec hpin hs hz h36

theorem memB_writable {Γ : Env} {w : World} {S : TState} {p n : AV} {x k : UInt64} (hS : S.holds w)
    (hp : ∃ t, p.Desc Γ w (.sc t x)) (hn : ∃ t, n.Desc Γ w (.sc t k)) (h : memB S p n = true) :
    Writable w.mem x k.toNat := by
  obtain ⟨r, off, hdec, hpin, hs, hz, h36⟩ := memB_spec hS hp hn h
  intro src hsrc
  exact Contracts.copyIn_isSome src hdec hpin (by rw [hsrc]; exact hs) hz (by rw [hsrc]; exact h36)

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

/-- A path the file shims can scan: a C string, or room for `pathMax` bytes. -/
def pathB (S : TState) (a : AV) : Bool := cstrB S a || memN S a pathMax

theorem pathB_sound {Γ : Env} {w : World} {S : TState} {a : AV} {x : UInt64} (hS : S.holds w)
    (hd : ∃ t, a.Desc Γ w (.sc t x)) (h : pathB S a = true) : Contracts.PathOk w.mem x := by
  unfold pathB at h
  rcases Bool.or_eq_true_iff.mp h with h | h
  · exact Contracts.pathOk_of_cstr (cstrB_sound hS hd h)
  · obtain ⟨r, off, hdec, -, hs, -, -⟩ := memN_spec hS hd h
    exact Contracts.pathOk_of_room hdec hs

theorem partB_sound {w : World} {S : TState} {p : Part} {b : Bool} (hS : S.holds w)
    (h : S.contains (.part p b) = true) : p.get w = b := holds_of_mem hS h

-- ---------------------------------------------------------------------------
-- What a call leaves
-- ---------------------------------------------------------------------------

/-- The bits of every argument, where all are known. -/
def allBits : List AV → Option (List UInt64)
  | [] => some []
  | a :: as => do let b ← a.bits; let bs ← allBits as; pure (b :: bs)

/-- The argument positions a frame reads. -/
def frPos : Frame → List Nat
  | .none | .anywhere => []
  | .at d l => [d, l]
  | .atOff b o l => [b, o, l]
  | .atOffRet b o => [b, o]
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

/-- The argument positions a call's keeping of fact `x` reads: its frame's for
    the bytes it may write, the slot for a C string `init` writes, the count for
    the values a table holds. -/
def fpos (f : Ffi) : Fact → List Nat
  | .cell .. | .cstrIn .. => if f = .fileRead then 4 :: frPos Contracts.readFrame else frPos (frame f)
  | .cstr _ => [0]
  | .htVals _ => [4]
  | _ => []

theorem keeps_congr {f : Ffi} {bits bits' : List UInt64} (x : Fact)
    (h : ∀ k ∈ fpos f x, bits.getD k 0 = bits'.getD k 0) : keeps f bits x = keeps f bits' x := by
  have hgd : ∀ (l : List UInt64), l.headD 0 = l.getD 0 0 := fun l => by cases l <;> rfl
  cases x with
  | cell a v =>
      by_cases hf : f = .fileRead
      · subst hf
        simp only [fpos, if_true, List.mem_cons] at h
        have h4 := h 4 (.inl rfl)
        have hfr : ∀ k ∈ frPos Contracts.readFrame, bits.getD k 0 = bits'.getD k 0 := fun k hk => h k (.inr hk)
        simp only [keeps, Contracts.writesAt, beq_self_eq_true, if_true, h4, mayWrite_congr hfr]
      · have hfr : ∀ k ∈ frPos (frame f), bits.getD k 0 = bits'.getD k 0 := fun k hk => h k (by
          simp only [fpos, if_neg hf]; exact hk)
        have hb : (f == .fileRead) = false := by simpa using hf
        simp only [keeps, Contracts.writesAt, hb, Bool.false_eq_true, if_false, Contracts.mayWrite,
          mayWrite_congr hfr]
  | cstrIn a n =>
      by_cases hf : f = .fileRead
      · subst hf
        simp only [fpos, if_true, List.mem_cons] at h
        have h4 := h 4 (.inl rfl)
        have hfr : ∀ k ∈ frPos Contracts.readFrame, bits.getD k 0 = bits'.getD k 0 := fun k hk => h k (.inr hk)
        simp only [keeps, beq_self_eq_true, if_true, h4, touches_congr hfr]
      · have hfr : ∀ k ∈ frPos (frame f), bits.getD k 0 = bits'.getD k 0 := fun k hk => h k (by
          simp only [fpos, if_neg hf]; exact hk)
        have hb : (f == .fileRead) = false := by simpa using hf
        simp only [keeps, hb, Bool.false_eq_true, if_false, touches_congr hfr]
  | cstr a =>
      have h0 := h 0 (by simp [fpos])
      simp only [keeps, hgd, h0]
  | htVals n =>
      have h4 := h 4 (by simp [fpos])
      simp only [keeps, h4]
  | _ => rfl

/-- Whether a call keeps a fact, decided by the arguments it is handed: where
    its keeping reads only arguments known, as `keeps` says. -/
def keepA (f : Ffi) (as : List AV) (x : Fact) : Bool :=
  (fpos f x).all (fun k => (as.getD k .any).bits.isSome) && keeps f (as.map (·.bits.getD 0)) x

/-- What a call leaves of a typestate: the table's `after` where every argument
    is known; otherwise each fact `keepA` keeps. -/
def postA (f : Ffi) (as : List AV) (S : TState) : TState :=
  match allBits as with
  | some bits => after f bits S
  | none => S.filter (keepA f as) ++ (if devArgFree f then devAfter f [] S else [])

theorem allBits_eq {Γ : Env} {w : World} : ∀ {as : List AV} {bits bs : List UInt64},
    ArgBits Γ w as bits → allBits as = some bs → bits = bs
  | [], [], bs, _, h => by simp only [allBits, Option.some.injEq] at h; exact h
  | [], _ :: _, _, h, _ => h.elim
  | _ :: _, [], _, h, _ => h.elim
  | a :: as, x :: xs, bs, ⟨hd, hds⟩, h => by
      simp only [allBits, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨b, hb, bs', hbs', rfl⟩ := h
      rw [known_bits hd hb, allBits_eq hds hbs']

theorem argBits_getD {Γ : Env} {w : World} : ∀ {as : List AV} {bits : List UInt64} {k : Nat} {y : UInt64},
    ArgBits Γ w as bits → (as.getD k .any).bits = some y → bits.getD k 0 = y
  | [], [], k, y, _, h => by simp [AV.bits] at h
  | [], _ :: _, _, _, h, _ => h.elim
  | _ :: _, [], _, _, h, _ => h.elim
  | a :: as, x :: xs, 0, y, ⟨hd, _⟩, h => by simpa using known_bits hd h
  | a :: as, x :: xs, k + 1, y, ⟨_, hds⟩, h => by simpa using argBits_getD hds h

/-- **A fact `keepA` keeps** holds after the call. -/
theorem keepA_sound {Γ : Env} {f : Ffi} {as : List AV} {bits : List UInt64} {w0 w w' : World}
    {r : Option V} (hb : ArgBits Γ w0 as bits) (h : callBits f bits w = some (r, w')) {x : Fact}
    (hk : keepA f as x = true) (hx : x.holds w) : x.holds w' := by
  unfold keepA at hk
  simp only [Bool.and_eq_true] at hk
  obtain ⟨hall, hk⟩ := hk
  refine keeps_sound h ?_ hx
  rw [keeps_congr (bits' := as.map (·.bits.getD 0)) x (fun k hk' => ?_)]
  · exact hk
  · obtain ⟨y, hy⟩ := Option.isSome_iff_exists.mp (List.all_eq_true.mp hall k hk')
    rw [argBits_getD hb hy]
    rw [List.getD_eq_getElem?_getD, List.getElem?_map]
    rw [List.getD_eq_getElem?_getD] at hy
    cases e : as[k]? with
    | none => rw [e] at hy; cases hy
    | some a => rw [e] at hy; simp only [Option.getD_some] at hy; simp [hy]

theorem postA_sound {Γ : Env} {f : Ffi} {as : List AV} {S : TState} {bits : List UInt64} {w w' : World}
    {r : Option V} (hb : ArgBits Γ w as bits) (hS : S.holds w)
    (h : callBits f bits w = some (r, w')) : (postA f as S).holds w' := by
  unfold postA
  split
  · rename_i bs hbs; rw [← allBits_eq hb hbs]; exact after_sound hS h
  · refine Contracts.TState.holds_app (fun x hx => ?_) (Contracts.TState.holds_if fun hf => ?_)
    · obtain ⟨hx, hk⟩ := List.mem_filter.mp hx
      exact keepA_sound hb h hk (hS x hx)
    · rw [devAfter_free hf [] bits]; exact devAfter_sound hS h

-- ---------------------------------------------------------------------------
-- Entry points
-- ---------------------------------------------------------------------------

/-- The size of the device buffer the typestate names by id. -/
def devBufOf (S : TState) (a : UInt64) : Option UInt64 :=
  S.findSome? fun | .devBuf a' n => if a' == a then some n else none | _ => none

theorem devBufOf_mem {S : TState} {a n : UInt64} (h : devBufOf S a = some n) : Fact.devBuf a n ∈ S := by
  unfold devBufOf at h
  obtain ⟨x, hx, hf⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp only [reduceCtorEq] at hf
  rename_i a' n'
  split at hf
  · rename_i he; simp only [beq_iff_eq] at he; subst he; cases hf; exact hx
  · cases hf

/-- A slot holding null, or a live context's handle, by the cell the typestate
    names there. -/
def slotOk (S : TState) (p : Part) (H : UInt64) (s : AV) : Bool :=
  match s.bits with
  | some a => match cellAt S a with
    | some v => v == 0 || (v == H && S.contains (.part p true))
    | none => false
  | none => false

theorem slotOk_sound {Γ : Env} {w : World} {S : TState} {p : Part} {H x : UInt64} {s : AV} (hS : S.holds w)
    (hd : ∃ t, s.Desc Γ w (.sc t x)) (h : slotOk S p H s = true) :
    ∀ c, w.mem.load x 8 = some c → c = 0 ∨ (c = H ∧ p.get w = true) := by
  unfold slotOk at h
  split at h
  · rename_i a ha
    rw [known_bits hd ha]
    split at h
    · rename_i v hv
      intro c hc
      have hl : w.mem.load a 8 = some v := hS _ (cellAt_mem hv)
      rw [hl] at hc; cases hc
      simp only [Bool.or_eq_true, beq_iff_eq, Bool.and_eq_true] at h
      rcases h with h | ⟨h1, h2⟩
      · exact .inl h
      · exact .inr ⟨h1, holds_of_mem hS h2⟩
    · cases h
  · cases h

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

/-- The CUDA context handle. -/
def cudaB (S : TState) (c : AV) : Bool := ctxB S .cuda cudaCtx c

/-- What one entry point needs, decided, by the arguments it is handed. -/
def preCudaInit (S : TState) (as : List AV) : Bool :=
  match as with
  | [s] => memN S s 8
  | _ => false

def preCudaCleanup (S : TState) (as : List AV) : Bool :=
  match as with
  | [s] => memN S s 8
  | _ => false

def preCudaCreateBuffer (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _] => cudaB S c
  | _ => false

def preCudaSync (S : TState) (as : List AV) : Bool :=
  match as with
  | [c] => cudaB S c
  | _ => false

def preCudaUpload (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [c, _, d, n] => cudaB S c && memBR S rb 2 3 d n && readyIn S defaultParty
  | _ => false

def preCudaDownload (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [c, _, d, n] => cudaB S c && memBR S rb 2 3 d n && readyIn S defaultParty
  | _ => false

def preCudaUploadOffset (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [c, _, _, d, n] => cudaB S c && memBR S rb 3 4 d n && readyIn S defaultParty
  | _ => false

def preCudaDownloadOffset (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [c, _, _, d, n] => cudaB S c && memBR S rb 3 4 d n && readyIn S defaultParty
  | _ => false

def preCudaLaunch (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, kp, nb, bp, _, _, _, _, _, _] =>
      cudaB S c && S.contains .oracles && S.contains (.part .capturing false) && cstrB S kp &&
        readyIn S defaultParty &&
        (match nb.bits with
         | some n => memN S bp (4 * (asI32 n).toNat)
         | none => false)
  | _ => false

def preCudaLaunchNamed (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, kp, np, nb, bp, _, _, _, _, _, _] =>
      cudaB S c && S.contains .oracles && S.contains (.part .capturing false) && cstrB S kp &&
        cstrB S np && readyIn S defaultParty &&
        (match nb.bits with
         | some n => memN S bp (4 * (asI32 n).toNat)
         | none => false)
  | _ => false

/-- Whether the party a stream id names knows every access, by the streams
    the typestate names; a stream id naming no party asks nothing. -/
def streamReady (S : TState) (sid : AV) : Bool :=
  match sid.bits, streamsOf S with
  | some s, some l => match partyOf l s with
    | some p => readyIn S p
    | none => true
  | _, _ => false

def preCudaLaunchOnStream (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, kp, nb, bp, _, _, _, _, _, _, sid] =>
      cudaB S c && S.contains .oracles && S.contains (.part .capturing false) && cstrB S kp &&
        streamReady S sid &&
        (match nb.bits with
         | some n => memN S bp (4 * (asI32 n).toNat)
         | none => false)
  | _ => false

/-- A wait on an event every access before whose record its stream knew. -/
def preCudaStreamWaitEvent (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _, e] =>
      cudaB S c && S.contains (.part .capturing false) &&
        (match e.bits with
         | some eid => match evId eid with
           | some n => S.contains (.evAll n)
           | none => false
         | none => false)
  | _ => false

def preGpuInit (S : TState) (as : List AV) : Bool :=
  match as with
  | [s] => memN S s 8
  | _ => false

def preGpuCleanup (S : TState) (as : List AV) : Bool :=
  match as with
  | [s] => memN S s 8
  | _ => false

def preGpuCreateBuffer (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _] => ctxB S .gpu gpuCtx c
  | _ => false

def preGpuDispatch (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _, _, _, _] => ctxB S .gpu gpuCtx c
  | _ => false

def preGpuCreatePipeline (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, sp, bp, n] =>
      ctxB S .gpu gpuCtx c && cstrB S sp &&
        (match n.bits with
         | some k => memN S bp (8 * (asI32 k).toNat)
         | none => false)
  | _ => false

def preGpuUpload (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [c, _, s, n] => ctxB S .gpu gpuCtx c && memBR S rb 2 3 s n
  | _ => false

def preGpuDownload (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [c, _, d, n] => ctxB S .gpu gpuCtx c && memBR S rb 2 3 d n
  | _ => false

def preWindowInit (S : TState) (as : List AV) : Bool :=
  match as with
  | [s] => S.contains (.part .win false) && memN S s 8
  | _ => false

def preWindowOpen (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _, _, tp, tl, bp, bl] =>
      ctxB S .win winCtx c &&
        (match tl.bits, bl.bits with
         | some a, some b => memN S tp (asI64 a).toNat && memN S bp (asI64 b).toNat
         | _, _ => false)
  | _ => false

def preWindowPoll (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, ep, me] =>
      ctxB S .win winCtx c &&
        (match me.bits with
         | some m => memN S ep (32 * (asI32 m).toNat)
         | none => false)
  | _ => false

def preWindowPresentGpuBuffer (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, g, _] => ctxB S .win winCtx c && ctxB S .gpu gpuCtx g
  | _ => false

def preFileRead (S : TState) (as : List AV) : Bool :=
  match as with
  | [b, po, dO, _, n] =>
      match b.bits, po.bits, dO.bits, n.bits with
      | some base, some p, some d, some k =>
          pathB S (.known (.sc .i64 (base + p))) && k != 0 && memN S (.known (.sc .i64 (base + d))) k.toNat
      | _, _, _, _ => false
  | _ => false

/-- The addresses `base + s` for `s` a number `sn` describes: inside the region
    `base` starts. -/
def addrAt (base : UInt64) (sn : Num) : AV :=
  match [Region.arena, .data, .out].find? (regionBase · = base) with
  | some r => if sn.hi ≤ span then .ptr r ⟨.i64, sn.lo, sn.hi, 1, none⟩ else .any
  | none => .any

theorem addrAt_desc {Γ : Env} {w : World} {base s : UInt64} {sn : Num} (hs : sn.Holds w s.toNat) :
    (addrAt base sn).Desc Γ w (.sc .i64 (base + s)) := by
  unfold addrAt
  split
  · rename_i r hr
    have hb : regionBase r = base := by simpa using List.find?_some hr
    split
    · rename_i hsp
      refine ⟨s.toNat, by rw [hb, UInt64.ofNat_toNat], ⟨hs.1, hs.2.1, Nat.one_dvd _, trivial⟩, hsp⟩
    · trivial
  · trivial

def preFileWrite (S : TState) (as : List AV) : Bool :=
  match as with
  | [b, po, so, _, n] =>
      match b.bits, po.bits with
      | some base, some p =>
          pathB S (.known (.sc .i64 (base + p))) &&
            (match so.bits, n.bits with
             | some s, some k =>
                 if k = 0 then cstrB S (.known (.sc .i64 (base + s)))
                 else memN S (.known (.sc .i64 (base + s))) k.toNat
             | _, _ =>
                 match so.num, n.num with
                 | some sn, some nn => decide (1 ≤ nn.lo) && memN S (addrAt base sn) nn.hi
                 | _, _ => false)
      | _, _ => false
  | _ => false

def preFileReadToPtr (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [pp, d, _, n] => pathB S pp && memBR S rb 1 3 d n
  | _ => false

def preNativeArch (S : TState) (as : List AV) : Bool :=
  match as with
  | [] => true
  | _ => false

def preNativeLoad (S : TState) (as : List AV) : Bool :=
  match as with
  | [s, n] => match n.bits with
      | some k => memN S s (asI64 k).toNat
      | none => false
  | _ => false

def preCpuHas (S : TState) (as : List AV) : Bool :=
  match as with
  | [n] => cstrB S n
  | _ => false

def preHtInit (S : TState) (as : List AV) : Bool :=
  match as with
  | [s] => memN S s 8
  | _ => false

def preHtCleanup (S : TState) (as : List AV) : Bool :=
  match as with
  | [s] => memN S s 8
  | _ => false

def preLmdbInit (S : TState) (as : List AV) : Bool :=
  match as with
  | [s] => S.contains (.part .lmdb false) && memN S s 8
  | _ => false

def preLmdbBeginWriteTxn (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _] => ctxB S .lmdb lmdbCtx c
  | _ => false

def preLmdbCommitWriteTxn (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _] => ctxB S .lmdb lmdbCtx c
  | _ => false

def preHtCreate (S : TState) (as : List AV) : Bool :=
  match as with
  | [_] => true
  | _ => false

def preWindowCleanup (S : TState) (as : List AV) : Bool :=
  match as with
  | [s] => memN S s 8 && slotOk S .win winCtx s
  | _ => false

def preLmdbCleanup (S : TState) (as : List AV) : Bool :=
  match as with
  | [s] => memN S s 8 && slotOk S .lmdb lmdbCtx s
  | _ => false

def preLmdbOpen (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, pp, _] => ctxB S .lmdb lmdbCtx c && cstrB S pp && S.contains (.part .lmdbEmpty true)
  | _ => false

def preCublasSgemv (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, tr, m, n, _, a, x, _, y] =>
      cudaB S c && S.contains (.part .capturing false) && S.contains .oracles && S.contains .devSeq &&
        (match tr.bits, m.bits, n.bits, a.bits, x.bits, y.bits with
         | some tr, some m, some n, some a, some x, some y =>
             match devBufOf S a, devBufOf S x, devBufOf S y with
             | some na, some nx, some ny =>
                 decide (0 < asI32 m) && decide (0 < asI32 n) &&
                   decide (4 * ((asI32 m).toNat * (asI32 n).toNat) ≤ na.toNat) &&
                   decide (4 * (if asI32 tr != 0 then (asI32 m).toNat else (asI32 n).toNat) ≤ nx.toNat) &&
                   decide (4 * (if asI32 tr != 0 then (asI32 n).toNat else (asI32 m).toNat) ≤ ny.toNat)
             | _, _, _ => false
         | _, _, _, _, _, _ => false)
  | _ => false

def preCublasSgemm (S : TState) (as : List AV) : Bool :=
  match as, allBits as with
  | [c, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
    some [_, ta, tb, m, n, k, _, a, sa, b, sb, _, cc, sc, batch, oa, ob, oc, la, lb, lc] =>
      cudaB S c && S.contains (.part .capturing false) && S.contains .oracles && S.contains .devSeq &&
        (match devBufOf S a, devBufOf S b, devBufOf S cc with
         | some na, some nb, some nc =>
             gemmStridedFitsN 4 4 4 ta tb m n k sa sb sc batch oa ob oc la lb lc na.toNat nb.toNat nc.toNat
         | _, _, _ => false)
  | _, _ => false

def preLmdbPut (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _, kp, kl, vp, vl] =>
      ctxB S .lmdb lmdbCtx c &&
        (match kl.bits, vl.bits with
         | some a, some b => memN S kp (asI32 a).toNat && memN S vp (asI32 b).toNat
         | _, _ => false)
  | _ => false

def preHtInsert (S : TState) (as : List AV) : Bool :=
  match as with
  | [_, kp, kl, vp, vl] =>
      match kl.bits, vl.bits with
      | some a, some b => memN S kp a.toNat && memN S vp b.toNat
      | _, _ => false
  | _ => false

/-- The bound the typestate gives on the hash table's values. -/
def htValsOf : TState → Option Nat
  | [] => none
  | .htVals n :: _ => some n
  | _ :: S => htValsOf S

theorem htValsOf_mem : ∀ {S : TState} {n : Nat}, htValsOf S = some n → Fact.htVals n ∈ S
  | [], _, h => by cases h
  | .htVals m :: _, _, h => by cases h; exact List.mem_cons_self
  | .part .. :: S, _, h | .room .. :: S, _, h | .cell .. :: S, _, h | .held .. :: S, _, h
  | .opened .. :: S, _, h | .cstr .. :: S, _, h | .devSeq :: S, _, h | .oracles :: S, _, h
  | .cstrIn .. :: S, _, h | .roomArg .. :: S, _, h | .devBuf .. :: S, _, h
  | .pinnedUsed .. :: S, _, h | .streams .. :: S, _, h | .events .. :: S, _, h | .evLive .. :: S, _, h
  | .readyAll .. :: S, _, h | .evAll .. :: S, _, h => List.mem_cons_of_mem _ (htValsOf_mem h)

/-- A lookup into room for any value the table holds. -/
def preHtLookup (S : TState) (as : List AV) : Bool :=
  match as with
  | [_, kp, kl, rp] =>
      match kl.bits, htValsOf S with
      | some a, some n => memN S kp a.toNat && memN S rp n
      | _, _ => false
  | _ => false

/-- A scan from the first key only (a key of no bytes), into room for `cap`
    bytes. -/
def preLmdbCursorScan (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _, _, kl, _, rp, cap] =>
      ctxB S .lmdb lmdbCtx c &&
        (match kl.bits, cap.bits with
         | some k, some n => decide (asI32 k ≤ 0) && memN S rp n.toNat
         | _, _ => false)
  | _ => false

/-- A handle of kind `k` held, by its description. -/
def handB (k : Contracts.Held) : AV → Bool
  | .hand k' true => k' == k
  | .blk _ true => k == .heap
  | _ => false

/-- **What a call of the engine's own libraries needs, decided** by the
    arguments it is handed, as `extNeeds` asks it: each handle held, and room
    for each buffer it reads or writes, by a length known. -/
def preExt (S : TState) : Ext → List AV → Bool
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
  | .window .close, [p] | .window .pixels, [p] => handB .window p
  | .window .poll, [p, out] => handB .window p && memN S out 32
  | .cpu .count, [] | .cpu .unpin, [] => true
  | .cpu .core, [_] | .cpu .package, [_] | .cpu .pin, [_] => true
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
  | .usb .close, [d] | .usb .claim, [d, _] | .usb .release, [d, _] => handB .usbDevice d
  | .usb .control, [d, _, _, _, _, data, len, _] =>
      handB .usbDevice d && match len.bits with
        | some l => memN S data (len16 l)
        | none => false
  | .usb .bulk, [d, _, _, data, len, _] | .usb .interrupt, [d, _, _, data, len, _] =>
      handB .usbDevice d && match len.bits with
        | some l => memN S data (lenOf l)
        | none => false
  | _, _ => false

/-- Whether a library call with these arguments keeps a fact: as `extKeeps`
    says, where a cell or a string reads the arguments' bits only when they
    are known. -/
def extKeepsA (e : Ext) (as : List AV) (f : Fact) : Bool :=
  match f with
  | .cell .. | .cstrIn .. =>
      match as.mapM AV.bits with
      | some bits => Contracts.extKeeps e bits f
      | none => false
  | _ => Contracts.extKeeps e [] f

def preCudaLaunchNamedOnStream (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, kp, np, nb, bp, _, _, _, _, _, _, sid] =>
      cudaB S c && S.contains .oracles && S.contains (.part .capturing false) && cstrB S kp &&
        cstrB S np && streamReady S sid &&
        (match nb.bits with
         | some n => memN S bp (4 * (asI32 n).toNat)
         | none => false)
  | _ => false

def preCudaUploadAsync (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [c, _, s, n, sid] =>
      cudaB S c && S.contains (.part .capturing false) && memBR S rb 2 3 s n && streamReady S sid
  | _ => false

def preCudaUploadOffsetAsync (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [c, _, _, s, n, sid] =>
      cudaB S c && S.contains (.part .capturing false) && memBR S rb 3 4 s n && streamReady S sid
  | _ => false

def preCudaDownloadAsync (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [c, _, d, n, sid] =>
      cudaB S c && S.contains (.part .capturing false) && memBR S rb 2 3 d n && streamReady S sid
  | _ => false

def preCublasSgemvOnStream (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, tr, m, n, _, a, x, _, y, sid] =>
      cudaB S c && S.contains (.part .capturing false) && S.contains .oracles && streamReady S sid &&
        (match tr.bits, m.bits, n.bits, a.bits, x.bits, y.bits with
         | some tr, some m, some n, some a, some x, some y =>
             match devBufOf S a, devBufOf S x, devBufOf S y with
             | some na, some nx, some ny =>
                 decide (0 < asI32 m) && decide (0 < asI32 n) &&
                   decide (4 * ((asI32 m).toNat * (asI32 n).toNat) ≤ na.toNat) &&
                   decide (4 * (if asI32 tr != 0 then (asI32 m).toNat else (asI32 n).toNat) ≤ nx.toNat) &&
                   decide (4 * (if asI32 tr != 0 then (asI32 n).toNat else (asI32 m).toNat) ≤ ny.toNat)
             | _, _, _ => false
         | _, _, _, _, _, _ => false)
  | _ => false

def preCublasSgemmOnStream (S : TState) (as : List AV) : Bool :=
  match as, allBits as with
  | [c, _, _, _, _, _, _, _, _, _, _, _, _, _, _, sid, _, _, _, _, _, _],
    some [_, ta, tb, m, n, k, _, a, sa, b, sb, _, cc, sc, batch, _, oa, ob, oc, la, lb, lc] =>
      cudaB S c && S.contains (.part .capturing false) && S.contains .oracles && streamReady S sid &&
        (match devBufOf S a, devBufOf S b, devBufOf S cc with
         | some na, some nb, some nc =>
             gemmStridedFitsN 4 4 4 ta tb m n k sa sb sc batch oa ob oc la lb lc na.toNat nb.toNat nc.toNat
         | _, _, _ => false)
  | _, _ => false

def preCublasGemmStridedBatchedExBf16 (S : TState) (as : List AV) : Bool :=
  match as, allBits as with
  | [c, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
    some [_, ta, tb, m, n, k, _, a, sa, b, sb, _, cc, sc, batch, oa, ob, oc, la, lb, lc] =>
      cudaB S c && S.contains (.part .capturing false) && S.contains .oracles && S.contains .devSeq &&
        (match devBufOf S a, devBufOf S b, devBufOf S cc with
         | some na, some nb, some nc =>
             gemmStridedFitsN 2 2 4 ta tb m n k sa sb sc batch oa ob oc la lb lc na.toNat nb.toNat nc.toNat
         | _, _, _ => false)
  | _, _ => false

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

def preCublasGemmExBf16 (S : TState) (as : List AV) : Bool :=
  match as, allBits as with
  | [c, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
    some [_, ta, tb, m, n, k, _, a, b, _, cc, oa, ob, oc, la, lb, lc] =>
      cudaB S c && S.contains (.part .capturing false) && S.contains .oracles && S.contains .devSeq &&
        (match devBufOf S a, devBufOf S b, devBufOf S cc with
         | some na, some nb, some nc =>
             gemmBf16FitsN ta tb m n k oa ob oc la lb lc na.toNat nb.toNat nc.toNat
         | _, _, _ => false)
  | _, _ => false

def preStdinReadline (S : TState) (as : List AV) : Bool :=
  match as with
  | [b, dO, n] =>
      match b.bits, dO.bits, n.bits with
      | some base, some d, some k => k != 0 && memN S (.known (.sc .i64 (base + d))) k.toNat
      | _, _, _ => false
  | _ => false

def preThreadCleanup (S : TState) (as : List AV) : Bool :=
  match as with
  | [s] => memN S s 8 && slotOk S .thread threadCtx s
  | _ => false

def preFileCreateDirAll (S : TState) (as : List AV) : Bool :=
  match as with
  | [p] => pathB S p
  | _ => false

def preFileWriteFromPtr (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [pp, s, _, n] => pathB S pp && memBR S rb 1 3 s n
  | _ => false

def preStdoutWrite (S : TState) (as : List AV) : Bool :=
  match as with
  | [b, so, n] =>
      match b.bits, so.bits, n.bits with
      | some base, some s, some k => memN S (.known (.sc .i64 (base + s))) k.toNat
      | _, _, _ => false
  | _ => false

def preCudaFreeBuffer (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _] => cudaB S c && readyIn S defaultParty
  | _ => false

def preCublasPtrArray (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _, _, _, _] => cudaB S c && readyIn S defaultParty
  | _ => false

/-- An allocation that fits what is left of page-locked memory, by the bound
    the typestate has on what is used. -/
def preCudaPinnedAlloc (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, n] =>
      cudaB S c &&
        (match n.bits with
         | some k => S.any fun
             | .pinnedUsed u => decide (u + 63 + k.toNat ≤ regionSpan.toNat)
             | _ => false
         | none => false)
  | _ => false

def preGpuUploadPtr (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [c, _, s, n] => ctxB S .gpu gpuCtx c && memBR S rb 2 3 s n
  | _ => false

def preGpuDownloadPtr (S : TState) (rb : Nat → Nat → Bool) (as : List AV) : Bool :=
  match as with
  | [c, _, _, d, n] => ctxB S .gpu gpuCtx c && memBR S rb 3 4 d n
  | _ => false

def preThreadInit (S : TState) (as : List AV) : Bool :=
  match as with
  | [s] => S.contains (.part .thread false) && memN S s 8
  | _ => false

def preThreadJoin (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _] => ctxB S .thread threadCtx c
  | _ => false

/-- A call whose contract asks nothing but its arguments' number. -/
def preArity (n : Nat) (as : List AV) : Bool := as.length == n

/-- A call whose contract asks a live CUDA context first, among `n`
    arguments, and where `cap`, the device not capturing. -/
def preCtxN (S : TState) (n : Nat) (cap : Bool) (as : List AV) : Bool :=
  match as with
  | c :: rest => rest.length + 1 == n && cudaB S c && (!cap || S.contains (.part .capturing false))
  | [] => false

/-- **What a call needs, decided.** One case per entry point the checker
    decides; every other entry point is refused. -/
def preB (f : Ffi) (as : List AV) (S : TState) (rb : Nat → Nat → Bool := fun _ _ => false) : Bool :=
  match f with
  | .cosf | .sinf | .nativeFree | .htCount | .threadPriority => preArity 1 as
  | .powf | .memAdviseHuge | .memLock | .memUnlock => preArity 2 as
  | .cudaEventCreate | .cudaMemInfoFree | .cudaMemInfoTotal | .cudaStreamCreate => preCtxN S 1 false as
  | .cudaEventDestroy | .cudaGraphDestroy | .cudaPinnedFree | .cudaPinnedPtr => preCtxN S 2 false as
  | .cudaGraphUpload => preCtxN S 3 false as
  | .cudaPinnedPtrAt => preCtxN S 4 false as
  | .cudaGraphBeginCapture | .cudaGraphEndCapture | .cudaStreamDestroy | .cudaStreamSync => preCtxN S 2 true as
  | .cudaEventRecord => preCtxN S 3 true as
  | .cudaInit => preCudaInit S as
  | .cudaCleanup => preCudaCleanup S as
  | .cudaCreateBuffer => preCudaCreateBuffer S as
  | .cudaSync => preCudaSync S as
  | .cudaUpload => preCudaUpload S rb as
  | .cudaDownload => preCudaDownload S rb as
  | .cudaUploadOffset => preCudaUploadOffset S rb as
  | .cudaDownloadOffset => preCudaDownloadOffset S rb as
  | .cudaLaunch => preCudaLaunch S as
  | .cudaLaunchNamed => preCudaLaunchNamed S as
  | .cudaLaunchOnStream => preCudaLaunchOnStream S as
  | .cudaStreamWaitEvent => preCudaStreamWaitEvent S as
  | .gpuInit => preGpuInit S as
  | .gpuCleanup => preGpuCleanup S as
  | .gpuCreateBuffer => preGpuCreateBuffer S as
  | .gpuDispatch => preGpuDispatch S as
  | .gpuCreatePipeline => preGpuCreatePipeline S as
  | .gpuUpload => preGpuUpload S rb as
  | .gpuDownload => preGpuDownload S rb as
  | .windowInit => preWindowInit S as
  | .windowOpen => preWindowOpen S as
  | .windowPoll => preWindowPoll S as
  | .windowPresentGpuBuffer => preWindowPresentGpuBuffer S as
  | .fileRead => preFileRead S as
  | .fileWrite => preFileWrite S as
  | .fileReadToPtr => preFileReadToPtr S rb as
  | .nativeArch => preNativeArch S as
  | .nativeLoad => preNativeLoad S as
  | .cpuHas => preCpuHas S as
  | .htInit => preHtInit S as
  | .htCleanup => preHtCleanup S as
  | .lmdbInit => preLmdbInit S as
  | .lmdbBeginWriteTxn => preLmdbBeginWriteTxn S as
  | .lmdbCommitWriteTxn => preLmdbCommitWriteTxn S as
  | .htCreate => preHtCreate S as
  | .windowCleanup => preWindowCleanup S as
  | .lmdbCleanup => preLmdbCleanup S as
  | .lmdbOpen => preLmdbOpen S as
  | .cublasSgemv => preCublasSgemv S as
  | .cublasSgemm => preCublasSgemm S as
  | .lmdbPut => preLmdbPut S as
  | .htInsert => preHtInsert S as
  | .htLookup => preHtLookup S as
  | .lmdbCursorScan => preLmdbCursorScan S as
  | .fileCreateDirAll => preFileCreateDirAll S as
  | .fileWriteFromPtr => preFileWriteFromPtr S rb as
  | .stdoutWrite => preStdoutWrite S as
  | .cudaFreeBuffer => preCudaFreeBuffer S as
  | .cublasPtrArray => preCublasPtrArray S as
  | .cudaPinnedAlloc => preCudaPinnedAlloc S as
  | .gpuUploadPtr => preGpuUploadPtr S rb as
  | .gpuDownloadPtr => preGpuDownloadPtr S rb as
  | .threadInit => preThreadInit S as
  | .threadJoin => preThreadJoin S as
  | .stdinReadline => preStdinReadline S as
  | .threadCleanup => preThreadCleanup S as
  | .cudaLaunchNamedOnStream => preCudaLaunchNamedOnStream S as
  | .cudaUploadAsync => preCudaUploadAsync S rb as
  | .cudaUploadOffsetAsync => preCudaUploadOffsetAsync S rb as
  | .cudaDownloadAsync => preCudaDownloadAsync S rb as
  | .cublasSgemvOnStream => preCublasSgemvOnStream S as
  | .cublasSgemmOnStream => preCublasSgemmOnStream S as
  | .cublasGemmStridedBatchedExBf16 => preCublasGemmStridedBatchedExBf16 S as
  | .cublasGemmExBf16 => preCublasGemmExBf16 S as
  | _ => false

/-- **What a device call answers, by the facts**: a stream's or an event's
    creation, in a live CUDA context whose streams or events the typestate
    counts, answers the next index. -/
def devAnswer (f : Ffi) (as : List AV) (S : TState) : Option Nat :=
  match f with
  | .cudaStreamCreate => match as with
    | [c] => if c.bits == some cudaCtx && S.contains (.part .cuda true) then (streamsOf S).map List.length else none
    | _ => none
  | .cudaEventCreate => match as with
    | [c] => if c.bits == some cudaCtx && S.contains (.part .cuda true) then eventsOf S else none
    | _ => none
  | _ => none

end AlgorithmLib.Check
