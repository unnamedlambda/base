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
  obtain ⟨r, off, hd, hp, hs, hn, hl⟩ := hf
  exact Contracts.copyOut_isSome hd hp (by omega) (by omega) (hl.sub (Nat.le_refl _) (by omega))

theorem fits_writable {x : UInt64} {k j : Nat} (hf : Fits w.mem x k) (hj : j ≤ k) : Writable w.mem x j := by
  obtain ⟨r, off, hd, hp, hs, hn, hl⟩ := hf
  intro src hsrc
  exact Contracts.copyIn_isSome src hd hp (by omega) (by omega) (by rw [hsrc]; exact hl.sub (Nat.le_refl _) (by omega))

theorem fits_slot8 {x : UInt64} (hf : Fits w.mem x 8) : Contracts.Slot8 w.mem x := by
  obtain ⟨r, off, hd, hp, hs, -, hl⟩ := hf
  exact fun v => Contracts.store_isSome hd hp hs hl v

theorem fits_load8 {x : UInt64} (hf : Fits w.mem x 8) : Contracts.Load8 w.mem x := by
  obtain ⟨r, off, hd, hp, hs, -, hl⟩ := hf
  exact Contracts.load_isSome hd hp hs hl

theorem fits_load4 {x : UInt64} (hf : Fits w.mem x 4) : (w.mem.load x 4).isSome = true := by
  obtain ⟨r, off, hd, hp, hs, -, hl⟩ := hf
  exact Contracts.load_isSome hd hp hs hl

/-- An offset into the program's memory that is a multiple of four leaves its
    address one: the region starts at a multiple of four, and no offset the
    checker admits wraps. -/
theorem arena_addr_mod4 {off : Nat} (h4 : 4 ∣ off) (hspan : off ≤ span) :
    (regionBase .arena + UInt64.ofNat off) % 4 = 0 := by
  obtain ⟨k, rfl⟩ := h4
  apply UInt64.toNat_inj.mp
  simp only [regionBase, Region.idx, UInt64.toNat_mod, UInt64.toNat_add, UInt64.toNat_ofNat']
  simp

theorem readIds_of_fits {p : UInt64} {n : Nat} (hf : Fits w.mem p (4 * n)) : (readIds w.mem p n).isSome = true := by
  obtain ⟨r, off, hd, hp, hs, hn, hl0⟩ := hf
  have hl : ((List.range n).mapM fun i => w.mem.load (p + UInt64.ofNat (4 * i)) 4).isSome = true := by
    refine Contracts.mapM_isSome_of fun i hi => ?_
    have hi := List.mem_range.mp hi
    have hd' := Sem.decodeAddr_add (i := 4 * i) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
    exact Contracts.load_isSome hd' hp (by omega) (hl0.sub (by omega) (by omega))
  obtain ⟨raw, hraw⟩ := Option.isSome_iff_exists.mp hl
  unfold readIds
  rw [hraw]; rfl

theorem bindsReady_of_fits {nBufs bindPtr : UInt64} {p : Nat} (hS : S.holds w) (hq : readyIn S p = true)
    (hf : Fits w.mem bindPtr (4 * (asI32 nBufs).toNat)) : BindsReady w p nBufs bindPtr := by
  obtain ⟨ids, hids⟩ := Option.isSome_iff_exists.mp (readIds_of_fits hf)
  exact ⟨ids, hids, fun id _ _ => Contracts.ready_of_state hS hq _ _⟩

theorem memB_fits {p n : AV} {x k : UInt64} (hS : S.holds w) (hp : ∃ t, p.Desc Γ w (.sc t x))
    (hn : ∃ t, n.Desc Γ w (.sc t k)) (h : memB S p n = true) : Fits w.mem x k.toNat :=
  memB_spec hS hp hn h

-- ---------------------------------------------------------------------------
-- Entry points
-- ---------------------------------------------------------------------------

section entries

variable {bits : List UInt64}

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
  refine ⟨?_, memN_fits hS hd hm⟩
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
        simp only [List.mem_singleton] at hq; subst hq; exact memN_fits hS h1 h⟩
    · cases h
  all_goals first
    | (argbits hb [h1, h2]
       exact ⟨[(_, _)], rfl, fun q hq => by
         simp only [List.mem_singleton] at hq; subst hq; exact memBR_fits hS h1 h2 h (hrb 0 1)⟩)
    | (argbits hb [h1, h2, h3]
       exact ⟨[(_, _)], rfl, fun q hq => by
         simp only [List.mem_singleton] at hq; subst hq; exact memBR_fits hS h1 h2 h (hrb 0 1)⟩)
    | cases h

/-- Room for `n` bytes the arguments name, and a handle the arguments name,
    as the library contracts ask them. -/
private theorem ext_room {p : AV} {x : UInt64} {k : Nat} (hS : S.holds w) (hd : ∃ t, p.Desc Γ w (.sc t x))
    (h : memN S p k = true) : ∀ q ∈ [(x, k)], Fits w.mem q.1 q.2 := by
  intro q hq; simp only [List.mem_singleton] at hq; subst hq; exact memN_fits hS hd h

private theorem ext_hand {a : AV} {x : UInt64} {k : Contracts.Held} (hd : ∃ t, a.Desc Γ w (.sc t x))
    (h : handB k a = true) : ∀ p ∈ [(k, x)], p.1.live w p.2 = true := by
  intro q hq; simp only [List.mem_singleton] at hq; subst hq; exact handB_sound hd h

/-- **What `preFs` decides is what the file library asks.** -/
private theorem preFs_sound (hS : S.holds w) {f : FsFn} {as : List AV} {bits : List UInt64}
    (hb : ArgBits Γ w as bits) (h : preFs S rb f as = true) (hrb : RbSound w bits rb) :
    ∃ hs rs, Contracts.fsNeeds f bits = some (hs, rs) ∧ (∀ p ∈ hs, p.1.live w p.2 = true) ∧
      (∀ q ∈ rs, Fits w.mem q.1 q.2) := by
  unfold preFs at h
  split at h
  -- open
  · argbits hb [h1, h2, h3]
    exact ⟨[], _, rfl, by simp, fun q hq => by
      simp only [List.mem_singleton] at hq; subst hq; exact memBR_fits hS h1 h2 h (hrb 0 1)⟩
  -- close
  · argbits hb [h1]
    exact ⟨_, [], rfl, fun p hp => by
      simp only [List.mem_singleton] at hp; subst hp; exact handB_sound h1 h, by simp⟩
  -- read, write
  all_goals first
    | (simp only [Bool.and_eq_true] at h
       argbits hb [h1, h2, h3, h4]
       exact ⟨_, _, rfl, fun p hp => by
         simp only [List.mem_singleton] at hp; subst hp; exact handB_sound h1 h.1, fun q hq => by
         simp only [List.mem_singleton] at hq; subst hq; exact (memBR_fits hS h2 h3 h.2 (hrb 1 2))⟩)
    | (argbits hb [h1]
       exact ⟨_, [], rfl, fun p hp => by
         simp only [List.mem_singleton] at hp; subst hp; exact handB_sound h1 h, by simp⟩)
    | (argbits hb [h1, h2]
       first
         | exact ⟨_, [], rfl, fun p hp => by
             simp only [List.mem_singleton] at hp; subst hp; exact handB_sound h1 h, by simp⟩
         | exact ⟨[], _, rfl, by simp, fun q hq => by
             simp only [List.mem_singleton] at hq; subst hq; exact memBR_fits hS h1 h2 h (hrb 0 1)⟩)
    | (simp only [Bool.and_eq_true] at h
       argbits hb [h1, h2, h3, h4]
       exact ⟨[], _, rfl, by simp, fun q hq => by
         simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
         rcases hq with rfl | rfl
         · exact memBR_fits hS h1 h2 h.1 (hrb 0 1)
         · exact memBR_fits hS h3 h4 h.2 (hrb 2 3)⟩)
    | (simp only [Bool.and_eq_true] at h
       argbits hb [h1, h2, h3]
       exact ⟨_, _, rfl, fun p hp => by
         simp only [List.mem_singleton] at hp; subst hp; exact handB_sound h1 h.1, fun q hq => by
         simp only [List.mem_singleton] at hq; subst hq; exact (memBR_fits hS h2 h3 h.2 (hrb 1 2))⟩)
    | cases h

/-- **What `preNet` decides is what the network library asks.** -/
private theorem preNet_sound (hS : S.holds w) {f : NetFn} {as : List AV} {bits : List UInt64}
    (hb : ArgBits Γ w as bits) (h : preNet S rb f as = true) (hrb : RbSound w bits rb) :
    ∃ hs rs, Contracts.netNeeds f bits = some (hs, rs) ∧ (∀ p ∈ hs, p.1.live w p.2 = true) ∧
      (∀ q ∈ rs, Fits w.mem q.1 q.2) := by
  have one : ∀ {k : Contracts.Held} {a : AV} {x : UInt64}, (∃ t, a.Desc Γ w (.sc t x)) → handB k a = true →
      ∀ p ∈ [(k, x)], p.1.live w p.2 = true := fun hd hh p hp => by
    simp only [List.mem_singleton] at hp; subst hp; exact handB_sound hd hh
  unfold preNet at h
  split at h
  -- resolve
  · simp only [Bool.and_eq_true] at h
    argbits hb [h1, h2, h3, h4]
    exact ⟨[], _, rfl, by simp, fun q hq => by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
      rcases hq with rfl | rfl
      · exact memBR_fits hS h1 h2 h.1 (hrb 0 1)
      · exact (memBR_fits hS h3 h4 h.2 (hrb 2 3))⟩
  -- connect
  · argbits hb [h1, h2]; exact ⟨[], _, rfl, by simp, ext_room hS h1 h⟩
  -- listen
  · argbits hb [h1]; exact ⟨[], _, rfl, by simp, ext_room hS h1 h⟩
  -- the listener's address
  · simp only [Bool.and_eq_true] at h
    argbits hb [h1, h2]; exact ⟨_, _, rfl, one h1 h.1, ext_room hS h2 h.2⟩
  -- accept
  · argbits hb [h1, h2]; exact ⟨_, [], rfl, one h1 h, by simp⟩
  -- send, receive
  · simp only [Bool.and_eq_true] at h
    argbits hb [h1, h2, h3]
    exact ⟨_, _, rfl, one h1 h.1, fun q hq => by
      simp only [List.mem_singleton] at hq; subst hq; exact (memBR_fits hS h2 h3 h.2 (hrb 1 2))⟩
  · simp only [Bool.and_eq_true] at h
    argbits hb [h1, h2, h3]
    exact ⟨_, _, rfl, one h1 h.1, fun q hq => by
      simp only [List.mem_singleton] at hq; subst hq; exact (memBR_fits hS h2 h3 h.2 (hrb 1 2))⟩
  -- shutdown, close, the listener's close
  · argbits hb [h1]; exact ⟨_, [], rfl, one h1 h, by simp⟩
  · argbits hb [h1]; exact ⟨_, [], rfl, one h1 h, by simp⟩
  · argbits hb [h1]; exact ⟨_, [], rfl, one h1 h, by simp⟩
  -- bind
  · argbits hb [h1, h2]; exact ⟨[], _, rfl, by simp, ext_room hS h1 h⟩
  -- the UDP socket's address
  · simp only [Bool.and_eq_true] at h
    argbits hb [h1, h2]; exact ⟨_, _, rfl, one h1 h.1, ext_room hS h2 h.2⟩
  -- UDP send
  · simp only [Bool.and_eq_true] at h
    argbits hb [h1, h2, h3, h4]
    exact ⟨_, _, rfl, one h1 h.1.1, fun q hq => by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
      rcases hq with rfl | rfl
      · exact (memBR_fits hS h2 h3 h.1.2 (hrb 1 2))
      · exact memN_fits hS h4 h.2⟩
  -- UDP receive
  · simp only [Bool.and_eq_true] at h
    argbits hb [h1, h2, h3]
    exact ⟨_, _, rfl, one h1 h.1, fun q hq => by
      simp only [List.mem_singleton] at hq; subst hq; exact (memBR_fits hS h2 h3 h.2 (hrb 1 2))⟩
  -- close
  · argbits hb [h1]; exact ⟨_, [], rfl, one h1 h, by simp⟩
  · cases h

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
  · exact preFs_sound hS hb h hrb
  · exact preNet_sound hS hb h hrb
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

theorem fits_forParty {m : Mem} {x : UInt64} {k : Nat} (p : Nat) (hf : Fits m x k) : Fits (m.forParty p) x k := by
  obtain ⟨r, off, hd, hp, hs, hn⟩ := hf
  exact ⟨r, off, hd, hp, by rw [Contracts.forParty_sizes]; exact hs, hn⟩

theorem fits_readable_mem {m : Mem} {x : UInt64} {k j : Nat} (hf : Fits m x k) (hj : j ≤ k) : Readable m x j := by
  obtain ⟨r, off, hd, hp, hs, hn, hl⟩ := hf
  exact Contracts.copyOut_isSome hd hp (by omega) (by omega) (hl.sub (Nat.le_refl _) (by omega))

theorem fits_writable_mem {m : Mem} {x : UInt64} {k j : Nat} (hf : Fits m x k) (hj : j ≤ k) : Writable m x j := by
  obtain ⟨r, off, hd, hp, hs, hn, hl⟩ := hf
  intro src hsrc
  exact Contracts.copyIn_isSome src hd hp (by omega) (by omega) (by rw [hsrc]; exact hl.sub (Nat.le_refl _) (by omega))

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

theorem ArgBits.length_eq {Γ : Env} {w : World} :
    ∀ {as : List AV} {bits : List UInt64}, ArgBits Γ w as bits → bits.length = as.length
  | [], [], _ => rfl
  | _ :: _, _ :: _, h => by simp [ArgBits.length_eq h.2]
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim


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
