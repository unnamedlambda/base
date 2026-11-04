module
public import AlgorithmLib.Check.Calls
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.PreSound` — what `preB` decides is the precondition

**`preB_sound`**: where `preB` accepts a call on arguments the checker
describes, `Contracts.Pre` holds of the bits they carry. One lemma per entry
point, each from the atoms' own lemmas.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Prog

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState holds_of_mem Part readyIn streamsOf streamsOf_holds partyOf partyOf_eq evId evId_spec)

theorem ArgBits.cons_inv {Γ : Env} {w : World} {a : AV} {as : List AV} {bits : List UInt64}
    (h : ArgBits Γ w (a :: as) bits) :
    ∃ x xs, bits = x :: xs ∧ (∃ t, a.Desc Γ w (.sc t x)) ∧ ArgBits Γ w as xs := by
  match bits, h with
  | x :: xs, ⟨hx, hxs⟩ => exact ⟨x, xs, rfl, hx, hxs⟩

theorem ArgBits.nil_inv {Γ : Env} {w : World} {bits : List UInt64} (h : ArgBits Γ w [] bits) : bits = [] := by
  match bits, h with
  | [], _ => rfl

/-- Take apart the bits of a literal argument list, naming each argument's
    description. -/
syntax "argbits " ident " [" ident,* "]" : tactic
set_option hygiene false in
macro_rules
  | `(tactic| argbits $hb []) => `(tactic| (replace $hb := ArgBits.nil_inv $hb; subst $hb))
  | `(tactic| argbits $hb [$h]) =>
      `(tactic| (replace $hb := ArgBits.cons_inv $hb; obtain ⟨_, _, e, $h, $hb⟩ := $hb; subst e; argbits $hb []))
  | `(tactic| argbits $hb [$h, $hs,*]) =>
      `(tactic| (replace $hb := ArgBits.cons_inv $hb; obtain ⟨_, _, e, $h, $hb⟩ := $hb; subst e; argbits $hb [$hs,*]))

-- ---------------------------------------------------------------------------
-- Atoms, in the precondition's vocabulary
-- ---------------------------------------------------------------------------

variable {Γ : Env} {w : World} {S : TState} {rb : Nat → Nat → Bool}

theorem fits_readable {x : UInt64} {k j : Nat} (hf : Fits w.mem x k) (hj : j ≤ k) : Readable w.mem x j := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  exact Contracts.copyOut_isSome hd hp (by omega) hz (by omega)

theorem fits_writable {x : UInt64} {k j : Nat} (hf : Fits w.mem x k) (hj : j ≤ k) : Writable w.mem x j := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  intro src hsrc
  exact Contracts.copyIn_isSome src hd hp (by omega) hz (by omega)

theorem fits_slot8 {x : UInt64} (hf : Fits w.mem x 8) : Contracts.Slot8 w.mem x := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := hf
  exact fun v => Contracts.store_isSome hd hp hs hz v

theorem fits_load8 {x : UInt64} (hf : Fits w.mem x 8) : Contracts.Load8 w.mem x := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := hf
  exact Contracts.load_isSome hd hp hs hz

theorem fits_load4 {x : UInt64} (hf : Fits w.mem x 4) : (w.mem.load x 4).isSome = true := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := hf
  exact Contracts.load_isSome hd hp hs hz

/-- An offset into the program's memory that is a multiple of four leaves its
    address one: the region starts at a multiple of four, and no offset the
    checker admits wraps. -/
theorem arena_addr_mod4 {off : Nat} (h4 : 4 ∣ off) (hspan : off ≤ span) :
    (regionBase .arena + UInt64.ofNat off) % 4 = 0 := by
  obtain ⟨k, rfl⟩ := h4
  apply UInt64.toNat_inj.mp
  simp only [regionBase, UInt64.toNat_mod, UInt64.toNat_add, UInt64.toNat_ofNat']
  simp

theorem readIds_of_fits {p : UInt64} {n : Nat} (hf : Fits w.mem p (4 * n)) : (readIds w.mem p n).isSome = true := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  have hl : ((List.range n).mapM fun i => w.mem.load (p + UInt64.ofNat (4 * i)) 4).isSome = true := by
    refine Contracts.mapM_isSome_of fun i hi => ?_
    have hi := List.mem_range.mp hi
    have hd' := Sem.decodeAddr_add (i := 4 * i) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
    exact Contracts.load_isSome hd' hp (by omega) hz
  obtain ⟨raw, hraw⟩ := Option.isSome_iff_exists.mp hl
  unfold readIds
  rw [hraw]; rfl

theorem bindsReady_of_fits {nBufs bindPtr : UInt64} {p : Nat} (hS : S.holds w) (hq : readyIn S p = true)
    (hf : Fits w.mem bindPtr (4 * (asI32 nBufs).toNat)) : BindsReady w p nBufs bindPtr := by
  obtain ⟨ids, hids⟩ := Option.isSome_iff_exists.mp (readIds_of_fits hf)
  exact ⟨ids, hids, fun id _ _ => Contracts.ready_of_state hS hq _ _⟩

theorem memN_fits' {p : AV} {x : UInt64} {k : Nat} (hS : S.holds w) (hd : ∃ t, p.Desc Γ w (.sc t x))
    (h : memN S p k = true) : Fits w.mem x k := memN_spec hS hd h

theorem memB_fits {p n : AV} {x k : UInt64} (hS : S.holds w) (hp : ∃ t, p.Desc Γ w (.sc t x))
    (hn : ∃ t, n.Desc Γ w (.sc t k)) (h : memB S p n = true) : Fits w.mem x k.toNat :=
  memB_spec hS hp hn h

-- ---------------------------------------------------------------------------
-- Entry points
-- ---------------------------------------------------------------------------

section entries

variable {bits : List UInt64}

theorem streamReady_sound {sid : AV} {x : UInt64} (hS : S.holds w) (hd : ∃ t, sid.Desc Γ w (.sc t x))
    (h : streamReady S sid = true) : ∀ p, w.dev.party? (asI32 x) = some p → readyIn S p = true := by
  unfold streamReady at h
  split at h
  · rename_i s l hs hl
    have hx := known_bits hd hs
    subst hx
    intro p hp
    rw [partyOf_eq (streamsOf_holds hS hl)] at hp
    split at h
    · rename_i p' hp'
      rw [hp'] at hp; cases hp; exact h
    · rename_i hp'
      rw [hp'] at hp; cases hp
  · cases h

theorem pre_lmdbCleanup {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .lmdbCleanup [s] S rb = true) : Contracts.Pre .lmdbCleanup bits w := by
  dsimp only [preB, preLmdbCleanup] at h; simp only [Bool.and_eq_true] at h
  argbits hb [hs]
  have hf := memN_fits' hS hs h.1
  exact ⟨fits_load8 hf, slotOk_sound hS hs h.2, fits_slot8 hf⟩

theorem pre_lmdbInit {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .lmdbInit [s] S rb = true) : Contracts.Pre .lmdbInit bits w := by
  dsimp only [preB, preLmdbInit] at h; simp only [Bool.and_eq_true] at h
  argbits hb [hs]
  exact ⟨partB_sound hS h.1, fits_slot8 (memN_fits' hS hs h.2)⟩

theorem pre_lmdbOpen {c pp x : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, pp, x] bits)
    (h : preB .lmdbOpen [c, pp, x] S rb = true) : Contracts.Pre .lmdbOpen bits w := by
  dsimp only [preB, preLmdbOpen] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hp⟩, he⟩ := h
  argbits hb [hc, hpp, hx]
  have hemp : w.lmdb.envs.isEmpty = true := partB_sound hS he
  refine ⟨ctxB_sound hS hc hc', fun _ => cstrB_sound hS hpp hp, fun _ _ => ?_⟩
  have he' : w.lmdb.envs = #[] := by simpa using hemp
  rw [he']; simp

theorem pre_lmdbBeginWriteTxn {c x : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, x] bits)
    (h : preB .lmdbBeginWriteTxn [c, x] S rb = true) : Contracts.Pre .lmdbBeginWriteTxn bits w := by
  dsimp only [preB, preLmdbBeginWriteTxn] at h
  argbits hb [hc, hx]
  exact ctxB_sound hS hc h

theorem pre_lmdbCommitWriteTxn {c x : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, x] bits)
    (h : preB .lmdbCommitWriteTxn [c, x] S rb = true) : Contracts.Pre .lmdbCommitWriteTxn bits w := by
  dsimp only [preB, preLmdbCommitWriteTxn] at h
  argbits hb [hc, hx]
  exact ctxB_sound hS hc h

theorem pre_lmdbPut {c x kp kl vp vl : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, x, kp, kl, vp, vl] bits)
    (h : preB .lmdbPut [c, x, kp, kl, vp, vl] S rb = true) : Contracts.Pre .lmdbPut bits w := by
  dsimp only [preB, preLmdbPut] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hc', hm⟩ := h
  argbits hb [hc, hx, hkp, hkl, hvp, hvl]
  split at hm
  · rename_i a b ha hb'
    have e1 := known_bits hkl ha
    have e2 := known_bits hvl hb'
    subst e1 e2
    simp only [Bool.and_eq_true] at hm
    exact ⟨ctxB_sound hS hc hc', fun _ =>
      ⟨fits_readable (memN_fits' hS hkp hm.1) (Nat.le_refl _), fits_readable (memN_fits' hS hvp hm.2) (Nat.le_refl _)⟩⟩
  · cases hm

theorem pre_lmdbCursorScan {c hd kp kl me rp cap : AV} (hS : S.holds w)
    (hb : ArgBits Γ w [c, hd, kp, kl, me, rp, cap] bits)
    (h : preB .lmdbCursorScan [c, hd, kp, kl, me, rp, cap] S rb = true) : Contracts.Pre .lmdbCursorScan bits w := by
  dsimp only [preB, preLmdbCursorScan] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hc', hm⟩ := h
  argbits hb [hc, hhd, hkp, hkl, hme, hrp, hcap]
  split at hm
  · rename_i k n hk hn
    have e1 := known_bits hkl hk
    have e2 := known_bits hcap hn
    subst e1 e2
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hm
    exact ⟨ctxB_sound hS hc hc', fun _ j hj => fits_writable (memN_fits' hS hrp hm.2) hj,
      fun _ hpos => absurd hpos (by omega)⟩
  · cases hm

theorem pre_lmdbCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbCleanup as S rb = true) : Contracts.Pre .lmdbCleanup bits w := by
  dsimp only [preB] at h
  unfold preLmdbCleanup at h
  split at h
  · exact pre_lmdbCleanup (rb := rb) hS hb h
  · cases h

theorem pre_lmdbInit' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbInit as S rb = true) : Contracts.Pre .lmdbInit bits w := by
  dsimp only [preB] at h
  unfold preLmdbInit at h
  split at h
  · exact pre_lmdbInit (rb := rb) hS hb h
  · cases h

theorem pre_lmdbOpen' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbOpen as S rb = true) : Contracts.Pre .lmdbOpen bits w := by
  dsimp only [preB] at h
  unfold preLmdbOpen at h
  split at h
  · exact pre_lmdbOpen (rb := rb) hS hb h
  · cases h

theorem pre_lmdbBeginWriteTxn' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbBeginWriteTxn as S rb = true) : Contracts.Pre .lmdbBeginWriteTxn bits w := by
  dsimp only [preB] at h
  unfold preLmdbBeginWriteTxn at h
  split at h
  · exact pre_lmdbBeginWriteTxn (rb := rb) hS hb h
  · cases h

theorem pre_lmdbCommitWriteTxn' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbCommitWriteTxn as S rb = true) : Contracts.Pre .lmdbCommitWriteTxn bits w := by
  dsimp only [preB] at h
  unfold preLmdbCommitWriteTxn at h
  split at h
  · exact pre_lmdbCommitWriteTxn (rb := rb) hS hb h
  · cases h

theorem pre_lmdbPut' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbPut as S rb = true) : Contracts.Pre .lmdbPut bits w := by
  dsimp only [preB] at h
  unfold preLmdbPut at h
  split at h
  · exact pre_lmdbPut (rb := rb) hS hb h
  · cases h

theorem pre_lmdbCursorScan' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbCursorScan as S rb = true) : Contracts.Pre .lmdbCursorScan bits w := by
  dsimp only [preB] at h
  unfold preLmdbCursorScan at h
  split at h
  · exact pre_lmdbCursorScan (rb := rb) hS hb h
  · cases h

end entries

/-- **What `preB` accepts, `Pre` holds of**: where the checker describes the
    arguments and the typestate, a call `preB` accepts has its precondition. -/
theorem handB_sound {a : AV} {x : UInt64} {k : Contracts.Held} (hd : ∃ t, a.Desc Γ w (.sc t x))
    (h : handB k a = true) : k.live w x = true := by
  unfold handB at h
  split at h
  · rename_i k'
    simp only [beq_iff_eq] at h
    subst h
    obtain ⟨t, x', e, hl⟩ := hd
    cases e
    rcases hl with ⟨h1, -⟩ | hl
    · cases h1
    · exact hl
  · simp only [beq_iff_eq] at h
    subst h
    obtain ⟨t, x', e, hl⟩ := hd
    cases e
    rcases hl with ⟨h1, -⟩ | hl
    · cases h1
    · exact hl.live
  · cases h

/-- The checker's word for a wait is one the model reads: an address in the
    program's memory at an offset a multiple of four, with four bytes there. -/
private theorem wordB_sound {p : AV} {x : UInt64} (hS : S.holds w) (hd : ∃ t, p.Desc Γ w (.sc t x))
    (h : wordB S p = true) : x % 4 = 0 ∧ Fits w.mem x 4 := by
  unfold wordB at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨hm, hal⟩ := h
  refine ⟨?_, memN_fits' hS hd hm⟩
  split at hal
  · obtain ⟨t0, hd⟩ := hd
    obtain ⟨off, hv, ⟨-, hhi, hdvd, -⟩, hspan⟩ := hd
    cases hv
    have h4 : 4 ∣ off := Nat.dvd_trans (Nat.dvd_of_mod_eq_zero (by simpa using hal)) hdvd
    exact arena_addr_mod4 h4 (Nat.le_trans hhi hspan)
  · cases hal

/-- **What `preBase` decides is what the engine's functions ask.** -/
private theorem preBase_sound (hS : S.holds w) {f : BaseFn} {as : List AV} {bits : List UInt64}
    (hb : ArgBits Γ w as bits) (h : preBase S rb f as = true) (hrb : RbSound w bits rb) :
    ∃ rs, Contracts.baseNeeds f bits = some ([], rs) ∧ ∀ q ∈ rs, Fits w.mem q.1 q.2 := by
  unfold preBase at h
  split at h
  · argbits hb []; exact ⟨[], rfl, by simp⟩
  · argbits hb [h1, h2]; exact ⟨[], rfl, by simp⟩
  · argbits hb [h1, h2]; exact ⟨[], rfl, by simp⟩
  · argbits hb [h1, h2]; exact ⟨[], rfl, by simp⟩
  · argbits hb [h1]; exact ⟨[], rfl, by simp⟩
  · argbits hb [h1, h2, h3]
    obtain ⟨ha, hf⟩ := wordB_sound hS h1 h
    exact ⟨[(_, 4)], by simp [Contracts.baseNeeds, ha] <;> rfl, fun q hq => by
      simp only [List.mem_singleton] at hq; subst hq; exact hf⟩
  · argbits hb [h1, h2]
    obtain ⟨ha, hf⟩ := wordB_sound hS h1 h
    exact ⟨[(_, 4)], by simp [Contracts.baseNeeds, ha] <;> rfl, fun q hq => by
      simp only [List.mem_singleton] at hq; subst hq; exact hf⟩
  · argbits hb []; exact ⟨[], rfl, by simp⟩
  · split at h
    · rename_i k hk
      argbits hb [h1, h2]
      have := known_bits h2 hk; subst this
      exact ⟨[(_, _)], rfl, fun q hq => by
        simp only [List.mem_singleton] at hq; subst hq; exact memN_fits' hS h1 h⟩
    · cases h
  all_goals first
    | (argbits hb [h1, h2]
       exact ⟨[(_, _)], rfl, fun q hq => by
         simp only [List.mem_singleton] at hq; subst hq; exact memBR_fits hS h1 h2 h (hrb 0 1)⟩)
    | (argbits hb [h1, h2, h3]
       exact ⟨[(_, _)], rfl, fun q hq => by
         simp only [List.mem_singleton] at hq; subst hq; exact memBR_fits hS h1 h2 h (hrb 0 1)⟩)
    | (simp only [Bool.and_eq_true] at h
       argbits hb [h1, h2, h3, h4, h5]
       exact ⟨[(_, _), (_, _)], rfl, fun q hq => by
         simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
         rcases hq with rfl | rfl
         · exact memBR_fits hS h1 h2 h.1 (hrb 0 1)
         · exact memBR_fits hS h3 h5 h.2 (hrb 2 4)⟩)
    | cases h

/-- Room for `n` bytes the arguments name, and a handle the arguments name,
    as the library contracts ask them. -/
private theorem ext_room {p : AV} {x : UInt64} {k : Nat} (hS : S.holds w) (hd : ∃ t, p.Desc Γ w (.sc t x))
    (h : memN S p k = true) : ∀ q ∈ [(x, k)], Fits w.mem q.1 q.2 := by
  intro q hq; simp only [List.mem_singleton] at hq; subst hq; exact memN_fits hS hd h

private theorem ext_hand {a : AV} {x : UInt64} {k : Contracts.Held} (hd : ∃ t, a.Desc Γ w (.sc t x))
    (h : handB k a = true) : ∀ p ∈ [(k, x)], p.1.live w p.2 = true := by
  intro q hq; simp only [List.mem_singleton] at hq; subst hq; exact handB_sound hd h

/-- **What `preExt` decides is what the library contracts ask**: the handles
    and the room `extNeeds` names for the bits handed over. -/
theorem preExt_sound (hS : S.holds w) {e : IR.Ext} {as : List AV} {bits : List UInt64} (hb : ArgBits Γ w as bits)
    (h : preExt S rb e as = true) (hrb : RbSound w bits rb) :
    ∃ hs rs, Contracts.extNeeds e bits = some (hs, rs) ∧ (∀ p ∈ hs, p.1.live w p.2 = true) ∧
      (∀ q ∈ rs, Fits w.mem q.1 q.2) := by
  have none_ : ∀ p ∈ ([] : List (Contracts.Held × UInt64)), p.1.live w p.2 = true := by simp
  have noroom : ∀ q ∈ ([] : List (UInt64 × Nat)), Fits w.mem q.1 q.2 := by simp
  unfold preExt at h
  split at h <;> (try (cases h; done))
  -- whether a library is there
  · have := ArgBits.nil_inv hb
    subst this
    exact ⟨[], [], rfl, none_, noroom⟩
  -- C: memcpy
  · rename_i d s n
    split at h
    · rename_i d' s' n' e1 e2 e3
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h
      obtain ⟨⟨hc, hs⟩, hd⟩ := h
      argbits hb [h1, h2, h3]
      have := known_bits h1 e1; subst this
      have := known_bits h2 e2; subst this
      have := known_bits h3 e3; subst this
      refine ⟨[], _, by simp only [Contracts.extNeeds]; rw [if_pos hc], none_, fun q hq => ?_⟩
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
      rcases hq with rfl | rfl
      · exact memN_fits hS h2 hs
      · exact memN_fits hS h1 hd
    · cases h
  -- C: memmove
  · rename_i d s n
    split at h
    · rename_i n' e3
      simp only [Bool.and_eq_true] at h
      argbits hb [h1, h2, h3]
      have := known_bits h3 e3; subst this
      refine ⟨[], _, rfl, none_, fun q hq => ?_⟩
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
      rcases hq with rfl | rfl
      · exact memN_fits hS h2 h.1
      · exact memN_fits hS h1 h.2
    · cases h
  -- C: memset
  · rename_i d c n
    split at h
    · rename_i n' e3
      argbits hb [h1, h2, h3]
      have := known_bits h3 e3; subst this
      exact ⟨[], _, rfl, none_, ext_room hS h1 h⟩
    · cases h
  -- C: calloc
  · rename_i n sz
    split at h
    · rename_i a b e1 e2
      simp only [decide_eq_true_eq] at h
      argbits hb [h1, h2]
      have := known_bits h1 e1; subst this
      have := known_bits h2 e2; subst this
      exact ⟨[], [], by simp [Contracts.extNeeds, h], none_, noroom⟩
    · cases h
  -- C: free
  · rename_i p
    argbits hb [h1]
    exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  -- window: init, pump
  · argbits hb []; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb []; exact ⟨[], [], rfl, none_, noroom⟩
  -- window: open
  · argbits hb [h1, h2, h3]; exact ⟨[], [], rfl, none_, noroom⟩
  -- window: close, pixels
  · argbits hb [h1]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  · argbits hb [h1]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  -- window: poll
  · simp only [Bool.and_eq_true] at h
    argbits hb [h1, h2]
    exact ⟨_, _, rfl, ext_hand h1 h.1, ext_room hS h2 h.2⟩
  -- cpu
  · argbits hb []; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb []; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb [h1]; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb [h1]; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb [h1]; exact ⟨[], [], rfl, none_, noroom⟩
  -- the engine's own
  · obtain ⟨rs, hn, hr⟩ := preBase_sound hS hb h hrb
    exact ⟨[], rs, hn, none_, hr⟩
  -- serial: count
  · argbits hb []; exact ⟨[], [], rfl, none_, noroom⟩
  -- serial: name
  · split at h
    · rename_i c e3
      argbits hb [h1, h2, h3]
      have := known_bits h3 e3; subst this
      exact ⟨[], _, rfl, none_, ext_room hS h2 h⟩
    · cases h
  -- serial: close, pending
  · argbits hb [h1]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  · argbits hb [h1]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  -- serial: read, write
  · simp only [Bool.and_eq_true] at h
    obtain ⟨hp, h⟩ := h
    split at h
    · rename_i l e3
      argbits hb [h1, h2, h3]
      have := known_bits h3 e3; subst this
      exact ⟨_, _, rfl, ext_hand h1 hp, ext_room hS h2 h⟩
    · cases h
  · simp only [Bool.and_eq_true] at h
    obtain ⟨hp, h⟩ := h
    split at h
    · rename_i l e3
      argbits hb [h1, h2, h3]
      have := known_bits h3 e3; subst this
      exact ⟨_, _, rfl, ext_hand h1 hp, ext_room hS h2 h⟩
    · cases h
  -- usb: count, info, open
  · argbits hb []; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb [h1, h2]; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb [h1]; exact ⟨[], [], rfl, none_, noroom⟩
  -- usb: close, claim, release
  · argbits hb [h1]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  · argbits hb [h1, h2]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  · argbits hb [h1]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  -- usb: control
  · simp only [Bool.and_eq_true] at h
    obtain ⟨hp, h⟩ := h
    split at h
    · rename_i l e7
      argbits hb [h1, h2, h3, h4, h5, h6, h7, h8]
      have := known_bits h7 e7; subst this
      exact ⟨_, _, rfl, ext_hand h1 hp, ext_room hS h6 h⟩
    · cases h
  -- usb: bulk, interrupt
  all_goals
    simp only [Bool.and_eq_true] at h
    obtain ⟨hp, h⟩ := h
    split at h
    · rename_i l e4
      argbits hb [h1, h2, h3, h4, h5]
      have := known_bits h4 e4; subst this
      exact ⟨_, _, rfl, ext_hand h1 hp, ext_room hS h3 h⟩
    · cases h

theorem pre_threadInit' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .threadInit as S rb = true) : Contracts.Pre .threadInit bits w := by
  dsimp only [preB] at h
  unfold preThreadInit at h
  split at h
  · simp only [Bool.and_eq_true] at h
    argbits hb [hs]
    exact ⟨partB_sound hS h.1, fits_slot8 (memN_fits' hS hs h.2)⟩
  · cases h

theorem pre_threadJoin' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .threadJoin as S rb = true) : Contracts.Pre .threadJoin bits w := by
  dsimp only [preB] at h
  unfold preThreadJoin at h
  split at h
  · argbits hb [hc, hd]
    exact ctxB_sound hS hc h
  · cases h

theorem fits_forParty {m : Mem} {x : UInt64} {k : Nat} (p : Nat) (hf : Fits m x k) : Fits (m.forParty p) x k := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  exact ⟨r, off, hd, hp, by rw [Contracts.forParty_sizes]; exact hs, hz, hn⟩

theorem fits_readable_mem {m : Mem} {x : UInt64} {k j : Nat} (hf : Fits m x k) (hj : j ≤ k) : Readable m x j := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  exact Contracts.copyOut_isSome hd hp (by omega) hz (by omega)

theorem fits_writable_mem {m : Mem} {x : UInt64} {k j : Nat} (hf : Fits m x k) (hj : j ≤ k) : Writable m x j := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  intro src hsrc
  exact Contracts.copyIn_isSome src hd hp (by omega) hz (by omega)

/-- A party that knows every access may run any operation. -/
theorem opReady_of_ready {p : Nat} (hS : S.holds w) (hq : readyIn S p = true) (rs ws : List Nat) :
    Contracts.OpReady w.dev.race p rs ws :=
  ⟨fun b _ => (Contracts.ready_of_state hS hq b true).1, fun b _ => (Contracts.ready_of_state hS hq b true).2 rfl⟩

theorem sgemvOk_of_ready {p : Nat} {trans m n a x y na nx ny : UInt64} (hS : S.holds w)
    (hq : readyIn S p = true)
    (ha : Fact.devBuf a na ∈ S) (hx : Fact.devBuf x nx ∈ S) (hy : Fact.devBuf y ny ∈ S)
    (hm : 0 < asI32 m) (hn : 0 < asI32 n)
    (hA : 4 * ((asI32 m).toNat * (asI32 n).toNat) ≤ na.toNat)
    (hX : 4 * (if asI32 trans != 0 then (asI32 m).toNat else (asI32 n).toNat) ≤ nx.toNat)
    (hY : 4 * (if asI32 trans != 0 then (asI32 n).toNat else (asI32 m).toNat) ≤ ny.toNat) :
    Contracts.SgemvOk w p trans m n a x y := by
  intro A X Y hA' hX' hY'
  refine ⟨hm, hn, ?_, opReady_of_ready hS hq _ _⟩
  simp only [sgemvFits, Bool.and_eq_true, decide_eq_true_eq]
  rw [Contracts.devBuf_size hS ha hA', Contracts.devBuf_size hS hx hX', Contracts.devBuf_size hS hy hY']
  exact ⟨⟨hA, hX⟩, hY⟩

theorem gemmOk_of_ready {p : Nat} {fits : ByteArray → ByteArray → ByteArray → Bool} {a b c na nb nc : UInt64}
    (hS : S.holds w) (hq : readyIn S p = true)
    (ha : Fact.devBuf a na ∈ S) (hb : Fact.devBuf b nb ∈ S) (hc : Fact.devBuf c nc ∈ S)
    (hf : ∀ A B C : ByteArray, A.size = na.toNat → B.size = nb.toNat → C.size = nc.toNat → fits A B C = true) :
    Contracts.GemmOk fits w p a b c := fun A B C hA hB hC =>
  ⟨hf A B C (Contracts.devBuf_size hS ha hA) (Contracts.devBuf_size hS hb hB) (Contracts.devBuf_size hS hc hC),
    opReady_of_ready hS hq _ _⟩

theorem pre_threadCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .threadCleanup as S rb = true) : Contracts.Pre .threadCleanup bits w := by
  dsimp only [preB] at h
  unfold preThreadCleanup at h
  split at h
  · simp only [Bool.and_eq_true] at h
    argbits hb [hs]
    have hf := memN_fits' hS hs h.1
    have hz : w.mem.frozen = false := by obtain ⟨_, _, _, _, _, hz, _⟩ := hf; exact hz
    have e : ({ w.mem with frozen := false } : Mem) = w.mem := by rw [← hz]
    show Contracts.ThreadSlotOk w _
    unfold Contracts.ThreadSlotOk
    rw [e]
    exact ⟨slotOk_sound hS hs h.2, fits_load8 hf, fits_slot8 hf⟩
  · cases h

theorem ArgBits.length_eq {Γ : Env} {w : World} :
    ∀ {as : List AV} {bits : List UInt64}, ArgBits Γ w as bits → bits.length = as.length
  | [], [], _ => rfl
  | _ :: _, _ :: _, h => by simp [ArgBits.length_eq h.2]
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim

theorem preArity_len {n : Nat} {as : List AV} (h : preArity n as = true) : as.length = n := by
  simpa [preArity] using h

theorem preCtxN_sound (hS : S.holds w) {as : List AV} (hb : ArgBits Γ w as bits) {n : Nat} {cap : Bool}
    (h : preCtxN S n cap as = true) :
    ∃ x rest, bits = x :: rest ∧ rest.length + 1 = n ∧ (x = 0 ∨ (x = cudaCtx ∧ Part.cuda.get w = true)) ∧
      (cap = true → w.dev.capture = none) := by
  unfold preCtxN at h
  split at h
  · rename_i c rest
    cases bits with
    | nil => exact hb.elim
    | cons x xs =>
        obtain ⟨hc, hr⟩ := hb
        simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, Bool.not_eq_true'] at h
        obtain ⟨⟨hl, hcb⟩, hcap⟩ := h
        refine ⟨x, xs, rfl, by rw [ArgBits.length_eq hr]; exact hl, ctxB_sound hS hc hcb, fun hc' => ?_⟩
        rcases hcap with e | e
        · rw [hc'] at e; cases e
        · exact Contracts.capture_of_state hS e
  · cases h

theorem preB_sound {f : Ffi} {as : List AV} {bits : List UInt64} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB f as S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre f bits w := by
  cases f
  case lmdbCleanup => exact pre_lmdbCleanup' hS hb h
  case lmdbInit => exact pre_lmdbInit' hS hb h
  case lmdbOpen => exact pre_lmdbOpen' hS hb h
  case lmdbBeginWriteTxn => exact pre_lmdbBeginWriteTxn' hS hb h
  case lmdbCommitWriteTxn => exact pre_lmdbCommitWriteTxn' hS hb h
  case lmdbPut => exact pre_lmdbPut' hS hb h
  case lmdbCursorScan => exact pre_lmdbCursorScan' hS hb h
  case threadInit => exact pre_threadInit' hS hb h
  case threadJoin => exact pre_threadJoin' hS hb h
  case threadCleanup => exact pre_threadCleanup' hS hb h
  case cosf =>
    have hl := (ArgBits.length_eq hb).trans (preArity_len (by dsimp only [preB] at h; exact h))
    match bits, hl with | [_], _ => trivial
  case sinf =>
    have hl := (ArgBits.length_eq hb).trans (preArity_len (by dsimp only [preB] at h; exact h))
    match bits, hl with | [_], _ => trivial
  case powf =>
    have hl := (ArgBits.length_eq hb).trans (preArity_len (by dsimp only [preB] at h; exact h))
    match bits, hl with | [_, _], _ => trivial
  all_goals (dsimp only [preB] at h; cases h)


/-- **The C strings a call reads are there**, where the checker accepts it. -/
theorem preExt_strs (hS : S.holds w) {e : IR.Ext} {as : List AV} {bits : List UInt64} (hb : ArgBits Γ w as bits)
    (h : preExt S rb e as = true) : ∀ a ∈ Contracts.extStrs e bits, Contracts.CStr w.mem a := by
  intro a ha
  cases e with
  | window f =>
      cases f with
      | «open» =>
          match as, bits, hb with
          | [t, _, _], [x, _, _], hb =>
              simp only [Contracts.extStrs, List.mem_singleton] at ha
              subst ha
              simp only [preExt] at h
              exact cstrB_sound hS hb.1 h
          | [], _, hb => simp [preExt] at h
          | [_], _, hb => simp [preExt] at h
          | [_, _], _, hb => simp [preExt] at h
          | _ :: _ :: _ :: _ :: _, _, hb => simp [preExt] at h
          | [_, _, _], [], hb => exact hb.elim
          | [_, _, _], [_], hb => exact hb.2.elim
          | [_, _, _], [_, _], hb => exact hb.2.2.elim
          | [_, _, _], _ :: _ :: _ :: _ :: _, hb => exact hb.2.2.2.elim
      | _ => simp [Contracts.extStrs] at ha
  | _ => simp [Contracts.extStrs] at ha

end AlgorithmLib.Check
