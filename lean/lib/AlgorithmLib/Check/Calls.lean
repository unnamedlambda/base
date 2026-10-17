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

open AlgorithmLib.HProg.Contracts (Fact TState after after_sound keeps keeps_sound roomAt roomAt_spec holds_of_mem
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

/-- What a call leaves of a typestate: the table's `after` where every argument
    is known; otherwise each fact whose keeping reads only arguments it knows,
    where it is kept. -/
def postA (f : Ffi) (as : List AV) (S : TState) : TState :=
  match allBits as with
  | some bits => after f bits S
  | none => S.filter fun x =>
      (fpos f x).all (fun k => (as.getD k .any).bits.isSome) && keeps f (as.map (·.bits.getD 0)) x

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

theorem postA_sound {Γ : Env} {f : Ffi} {as : List AV} {S : TState} {bits : List UInt64} {w w' : World}
    {r : Option V} (hb : ArgBits Γ w as bits) (hS : S.holds w)
    (h : callBits f bits w = some (r, w')) : (postA f as S).holds w' := by
  unfold postA
  split
  · rename_i bs hbs; rw [← allBits_eq hb hbs]; exact after_sound hS h
  · intro x hx
    obtain ⟨hx, hk⟩ := List.mem_filter.mp hx
    simp only [Bool.and_eq_true] at hk
    obtain ⟨hall, hk⟩ := hk
    refine keeps_sound h ?_ (hS x hx)
    rw [keeps_congr (bits' := as.map (·.bits.getD 0)) x (fun k hk' => ?_)]
    · exact hk
    · obtain ⟨y, hy⟩ := Option.isSome_iff_exists.mp (List.all_eq_true.mp hall k hk')
      rw [argBits_getD hb hy]
      rw [List.getD_eq_getElem?_getD, List.getElem?_map]
      rw [List.getD_eq_getElem?_getD] at hy
      cases e : as[k]? with
      | none => rw [e] at hy; cases hy
      | some a => rw [e] at hy; simp only [Option.getD_some] at hy; simp [hy]

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

def preCudaUpload (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _, d, n] => cudaB S c && memB S d n && S.contains .devSeq
  | _ => false

def preCudaDownload (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _, d, n] => cudaB S c && memB S d n && S.contains .devSeq
  | _ => false

def preCudaUploadOffset (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _, _, d, n] => cudaB S c && memB S d n && S.contains .devSeq
  | _ => false

def preCudaDownloadOffset (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _, _, d, n] => cudaB S c && memB S d n && S.contains .devSeq
  | _ => false

def preCudaLaunch (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, kp, nb, bp, _, _, _, _, _, _] =>
      cudaB S c && S.contains .oracles && S.contains (.part .capturing false) && cstrB S kp &&
        S.contains .devSeq &&
        (match nb.bits with
         | some n => memN S bp (4 * (asI32 n).toNat)
         | none => false)
  | _ => false

def preCudaLaunchNamed (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, kp, np, nb, bp, _, _, _, _, _, _] =>
      cudaB S c && S.contains .oracles && S.contains (.part .capturing false) && cstrB S kp &&
        cstrB S np && S.contains .devSeq &&
        (match nb.bits with
         | some n => memN S bp (4 * (asI32 n).toNat)
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

def preGpuUpload (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _, s, n] => ctxB S .gpu gpuCtx c && memB S s n
  | _ => false

def preGpuDownload (S : TState) (as : List AV) : Bool :=
  match as with
  | [c, _, d, n] => ctxB S .gpu gpuCtx c && memB S d n
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

def preFileWrite (S : TState) (as : List AV) : Bool :=
  match as with
  | [b, po, so, _, n] =>
      match b.bits, po.bits, so.bits, n.bits with
      | some base, some p, some s, some k =>
          pathB S (.known (.sc .i64 (base + p))) &&
            (if k = 0 then cstrB S (.known (.sc .i64 (base + s)))
             else memN S (.known (.sc .i64 (base + s))) k.toNat)
      | _, _, _, _ => false
  | _ => false

def preFileReadToPtr (S : TState) (as : List AV) : Bool :=
  match as with
  | [pp, d, _, n] => pathB S pp && memB S d n
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
  | .pinnedUsed .. :: S, _, h => List.mem_cons_of_mem _ (htValsOf_mem h)

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

/-- **What a call needs, decided.** One case per entry point a shipped body
    calls; every other entry point is refused. -/
def preB (f : Ffi) (as : List AV) (S : TState) : Bool :=
  match f with
  | .cudaInit => preCudaInit S as
  | .cudaCleanup => preCudaCleanup S as
  | .cudaCreateBuffer => preCudaCreateBuffer S as
  | .cudaSync => preCudaSync S as
  | .cudaUpload => preCudaUpload S as
  | .cudaDownload => preCudaDownload S as
  | .cudaUploadOffset => preCudaUploadOffset S as
  | .cudaDownloadOffset => preCudaDownloadOffset S as
  | .cudaLaunch => preCudaLaunch S as
  | .cudaLaunchNamed => preCudaLaunchNamed S as
  | .gpuInit => preGpuInit S as
  | .gpuCleanup => preGpuCleanup S as
  | .gpuCreateBuffer => preGpuCreateBuffer S as
  | .gpuDispatch => preGpuDispatch S as
  | .gpuCreatePipeline => preGpuCreatePipeline S as
  | .gpuUpload => preGpuUpload S as
  | .gpuDownload => preGpuDownload S as
  | .windowInit => preWindowInit S as
  | .windowOpen => preWindowOpen S as
  | .windowPoll => preWindowPoll S as
  | .windowPresentGpuBuffer => preWindowPresentGpuBuffer S as
  | .fileRead => preFileRead S as
  | .fileWrite => preFileWrite S as
  | .fileReadToPtr => preFileReadToPtr S as
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
  | _ => false

end AlgorithmLib.Check
