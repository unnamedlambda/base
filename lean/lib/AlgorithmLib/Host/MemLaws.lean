module
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.MemLaws` — what stores, copies and addresses do to memory

Two memories with the same layout (`MemSame`) answer alike whether a load,
store or copy succeeds. A store or a copy changes only the bytes from its
address (`store_frame`, `copyIn_frame`); a region's base plus an offset inside
its span decodes to both (`decodeAddr_base`, `decodeAddr_some`,
`decodeAddr_add`). `KeepsSize` is what a kernel oracle that keeps its
buffers' lengths satisfies, and `strSpan` where a string ends.
-/

namespace AlgorithmLib.HProg.Sem


theorem Mem.readable_sub {m : Mem} {r : Region} {off n off' n' : Nat} (h : m.readable r off n = true)
    (h1 : off ≤ off') (h2 : off' + n' ≤ off + n) : m.readable r off' n' = true := by
  unfold Mem.readable at h ⊢
  simp only [Bool.and_eq_true, Bool.or_eq_true] at h ⊢
  rcases h with hp | ⟨hl, hb⟩
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

open AlgorithmLib.IR
open AlgorithmLib.HProg

/-- Two memories that differ at most in their bytes. -/
structure MemSame (m₁ m₂ : Mem) : Prop where
  size : ∀ r, (m₁.region r).size = (m₂.region r).size
  live : m₁.pinnedLive = m₂.pinnedLive
  busy : m₁.busy = m₂.busy

/-- A kernel oracle that hands back buffers of the lengths it was given. -/
def KeepsSize (k : Launch → List ByteArray → List ByteArray) : Prop :=
  ∀ l bs, (k l bs).map ByteArray.size = bs.map ByteArray.size

theorem MemSame.refl (m : Mem) : MemSame m m := ⟨fun _ => rfl, rfl, rfl⟩

theorem MemSame.symm {m₁ m₂ : Mem} (h : MemSame m₁ m₂) : MemSame m₂ m₁ :=
  ⟨fun r => (h.size r).symm, h.live.symm, h.busy.symm⟩

theorem MemSame.trans {m₁ m₂ m₃ : Mem} (h₁ : MemSame m₁ m₂) (h₂ : MemSame m₂ m₃) : MemSame m₁ m₃ :=
  ⟨fun r => (h₁.size r).trans (h₂.size r), h₁.live.trans h₂.live, h₁.busy.trans h₂.busy⟩

theorem MemSame.retire {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (c : Clock) :
    MemSame (m₁.retire c) (m₂.retire c) :=
  ⟨fun r => by have := h.size r; cases r <;> exact this, h.live, by simp only [Mem.retire, h.busy]⟩

theorem MemSame.readable {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (r : Region) (off n : Nat) :
    m₁.readable r off n = m₂.readable r off n := by
  simp only [Mem.readable, h.live, h.busy]

theorem MemSame.reachable {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (r : Region) (off n : Nat) :
    m₁.reachable r off n = m₂.reachable r off n := by
  simp only [Mem.reachable, h.live, h.busy]

theorem MemSame.load_isSome {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (a : UInt64) (n : Nat) :
    (m₁.load a n).isSome = (m₂.load a n).isSome := by
  unfold Mem.load
  cases decodeAddr a with
  | none => rfl
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [bind, Option.bind, h.size r, h.readable r off n]
    split <;> rfl

theorem foldlM_isSome {α β γ : Type} (f : α → γ → Option α) (g : β → γ → Option β)
    (hfg : ∀ a b x, (f a x).isSome = (g b x).isSome) :
    ∀ (l : List γ) (a : α) (b : β), (l.foldlM f a).isSome = (l.foldlM g b).isSome
  | [], _, _ => rfl
  | x :: xs, a, b => by
    simp only [List.foldlM_cons]
    have := hfg a b x
    cases hf : f a x <;> cases hg : g b x <;> rw [hf, hg] at this <;> simp at this ⊢
    exact foldlM_isSome f g hfg xs _ _

theorem MemSame.copyOut_isSome {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (a : UInt64) (n : Nat) :
    (copyOut m₁ a n).isSome = (copyOut m₂ a n).isSome := by
  unfold copyOut
  apply foldlM_isSome
  intro acc acc' i
  have := MemSame.load_isSome h (a + UInt64.ofNat i) 1
  cases h1 : m₁.load (a + UInt64.ofNat i) 1 <;> cases h2 : m₂.load (a + UInt64.ofNat i) 1 <;>
    rw [h1, h2] at this <;> simp_all

theorem set!_size (b : ByteArray) (i : Nat) (v : UInt8) : (b.set! i v).size = b.size := by
  cases b; simp [ByteArray.set!, ByteArray.size, Array.set!_eq_setIfInBounds]

theorem get!_set!_ne (b : ByteArray) (i j : Nat) (v : UInt8) (h : i ≠ j) :
    (b.set! i v).get! j = b.get! j := by
  cases b with | mk bs =>
  simp [ByteArray.set!, ByteArray.get!, Array.set!_eq_setIfInBounds, getElem!_def, h]

theorem get!_set!_same (b : ByteArray) (i : Nat) (v : UInt8) (h : i < b.size) :
    (b.set! i v).get! i = v := by
  cases b with | mk bs =>
  simp only [ByteArray.size] at h
  simp [ByteArray.set!, ByteArray.get!, Array.set!_eq_setIfInBounds, h]

/-- The bytes a run of `set!` from `off` leaves: the written ones inside the
    range, the old ones outside it. -/
theorem get!_setRun (bs : ByteArray) (off : Nat) (f : Nat → UInt8) :
    ∀ (n j : Nat), off + n ≤ bs.size →
      ((List.range n).foldl (fun (b : ByteArray) i => b.set! (off + i) (f i)) bs).get! j
        = if off ≤ j ∧ j < off + n then f (j - off) else bs.get! j := by
  intro n
  induction n with
  | zero => intro j _; simp only [List.range_zero, List.foldl_nil]; rw [if_neg (by omega)]
  | succ n ih =>
    intro j hn
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    by_cases hj : j = off + n
    · subst hj
      have hsz : ((List.range n).foldl (fun (b : ByteArray) i => b.set! (off + i) (f i)) bs).size
          = bs.size := by
        clear ih
        induction n with
        | zero => rfl
        | succ n ih2 =>
          rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
            set!_size, ih2 (by omega)]
      rw [get!_set!_same _ _ _ (by rw [hsz]; omega)]
      simp
    · rw [get!_set!_ne _ _ _ _ (Ne.symm hj), ih j (by omega)]
      by_cases h1 : off ≤ j ∧ j < off + n
      · rw [if_pos h1, if_pos ⟨h1.1, by omega⟩]
      · rw [if_neg h1, if_neg (by omega)]

theorem size_setRun (bs : ByteArray) (off : Nat) (f : Nat → UInt8) :
    ∀ n, ((List.range n).foldl (fun (b : ByteArray) i => b.set! (off + i) (f i)) bs).size = bs.size
  | 0 => rfl
  | n + 1 => by
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
      set!_size, size_setRun bs off f n]

theorem region_setRegion_self (m : Mem) (r : Region) (b : ByteArray) : (m.setRegion r b).region r = b := by
  cases r <;> rfl

theorem region_setRegion_ne (m : Mem) {r r' : Region} (b : ByteArray) (h : r' ≠ r) :
    (m.setRegion r b).region r' = m.region r' := by
  cases r <;> cases r' <;> first | rfl | exact absurd rfl h

/-- What a store does to memory, region by region. -/
theorem store_some {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m') :
    ∃ r off, decodeAddr a = some (r, off) ∧ off + n ≤ (m.region r).size ∧
      m' = m.setRegion r ((List.range n).foldl (fun (b : ByteArray) i =>
        b.set! (off + i) (((v >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8)) (m.region r)) := by
  unfold Mem.store at h
  cases hd : decodeAddr a with
  | none => rw [hd] at h; cases h
  | some p =>
    obtain ⟨r, off⟩ := p
    rw [hd] at h
    simp only [bind, Option.bind] at h
    split at h
    · cases h
    · rename_i hc
      injection h with h
      refine ⟨r, off, rfl, ?_, h.symm⟩
      simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at hc
      omega

theorem MemSame.store_isSome {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (a : UInt64) (n : Nat) (v₁ v₂ : UInt64) :
    (m₁.store a n v₁).isSome = (m₂.store a n v₂).isSome := by
  unfold Mem.store
  cases decodeAddr a with
  | none => rfl
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [bind, Option.bind, h.size r, h.reachable r off n]
    split <;> rfl

theorem MemSame.setRegion {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (r : Region) {b₁ b₂ : ByteArray}
    (hb : b₁.size = b₂.size) : MemSame (m₁.setRegion r b₁) (m₂.setRegion r b₂) := by
  refine ⟨fun r' => ?_, ?_, ?_⟩
  · by_cases hr : r' = r
    · subst hr; rw [region_setRegion_self, region_setRegion_self, hb]
    · rw [region_setRegion_ne _ _ hr, region_setRegion_ne _ _ hr, h.size r']
  all_goals cases r <;> first | exact h.live | exact h.busy

/-- `m'` is `m` with at most `n` bytes from `off` in region `r` changed. -/
structure StoreFrame (m m' : Mem) (r : Region) (off n : Nat) : Prop where
  same : MemSame m m'
  other : ∀ r', r' ≠ r → m'.region r' = m.region r'
  outside : ∀ j, j < off ∨ off + n ≤ j → (m'.region r).get! j = (m.region r).get! j

theorem StoreFrame.trans {m m' m'' : Mem} {r : Region} {off n n' : Nat}
    (h₁ : StoreFrame m m' r off n) (h₂ : StoreFrame m' m'' r off n') (hle : n ≤ n') :
    StoreFrame m m'' r off n' :=
  ⟨h₁.same.trans h₂.same, fun r' hr => (h₂.other r' hr).trans (h₁.other r' hr),
    fun j hj => (h₂.outside j hj).trans (h₁.outside j (by omega))⟩

theorem StoreFrame.refl (m : Mem) (r : Region) (off n : Nat) : StoreFrame m m r off n :=
  ⟨MemSame.refl m, fun _ _ => rfl, fun _ _ => rfl⟩

theorem StoreFrame.widen {m m' : Mem} {r : Region} {off n off' n' : Nat}
    (h : StoreFrame m m' r off n) (h1 : off' ≤ off) (h2 : off + n ≤ off' + n') :
    StoreFrame m m' r off' n' :=
  ⟨h.same, h.other, fun j hj => h.outside j (by omega)⟩

theorem store_frame {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m') :
    ∃ r off, decodeAddr a = some (r, off) ∧ StoreFrame m m' r off n := by
  obtain ⟨r, off, hd, hn, rfl⟩ := store_some h
  refine ⟨r, off, hd, ⟨?_, fun r' hr => region_setRegion_ne _ _ hr, fun j hj => ?_⟩⟩
  · have := (MemSame.refl m).setRegion r (b₁ := m.region r)
      (b₂ := (List.range n).foldl (fun (b : ByteArray) i =>
        b.set! (off + i) (((v >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8)) (m.region r))
      (by rw [size_setRun])
    have e : m.setRegion r (m.region r) = m := by cases r <;> rfl
    rwa [e] at this
  · rw [region_setRegion_self, get!_setRun _ _ _ _ _ hn, if_neg (by omega)]

theorem regionSpan_toNat : regionSpan.toNat = 2 ^ 36 := by decide

theorem regionBase_toNat (r : Region) : (regionBase r).toNat + regionSpan.toNat ≤ 2 ^ 39 := by
  cases r <;> decide

theorem addrCond (b a : UInt64) :
    (decide (a ≥ b) && decide (a - b < regionSpan))
      = decide (b.toNat ≤ a.toNat ∧ a.toNat < b.toNat + 2 ^ 36) := by
  rw [Bool.eq_iff_iff, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq, decide_eq_true_eq]
  simp only [ge_iff_le, UInt64.le_iff_toNat_le, UInt64.lt_iff_toNat_lt, regionSpan_toNat]
  constructor
  · rintro ⟨h1, h2⟩
    rw [UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr h1)] at h2; omega
  · rintro ⟨h1, h2⟩
    refine ⟨h1, ?_⟩
    rw [UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr h1)]; omega

theorem decodeAddr_base (r : Region) (k : Nat) (hk : k < regionSpan.toNat) :
    decodeAddr (regionBase r + UInt64.ofNat k) = some (r, k) := by
  rw [regionSpan_toNat] at hk
  have hb := regionBase_toNat r
  rw [regionSpan_toNat] at hb
  have hx : (regionBase r + UInt64.ofNat k).toNat = (regionBase r).toNat + k := by
    rw [UInt64.toNat_add, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (a := k) (by omega),
      Nat.mod_eq_of_lt (by omega)]
  have hsub : ((regionBase r + UInt64.ofNat k) - regionBase r).toNat = k := by
    rw [UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr (by omega)), hx]; omega
  unfold decodeAddr
  generalize regionBase r + UInt64.ofNat k = a at hx hsub
  simp only [List.findSome?, addrCond]
  have e1 : (regionBase .arena).toNat = 2 ^ 36 := by decide
  have e2 : (regionBase .data).toNat = 2 * 2 ^ 36 := by decide
  have e3 : (regionBase .out).toNat = 3 * 2 ^ 36 := by decide
  have e4 : (regionBase .pinned).toNat = 4 * 2 ^ 36 := by decide
  cases r <;> rw [e1, e2, e3, e4] <;> simp only [e1, e2, e3, e4] at hx <;>
    simp (disch := omega) only [decide_eq_true, decide_eq_false, if_true, if_false,
      Bool.false_eq_true, hsub]

theorem decodeAddr_some {a : UInt64} {r : Region} {off : Nat} (h : decodeAddr a = some (r, off)) :
    a = regionBase r + UInt64.ofNat off ∧ off < regionSpan.toNat := by
  unfold decodeAddr at h
  simp only [List.findSome?, addrCond] at h
  have key : ∀ b : UInt64, decide (b.toNat ≤ a.toNat ∧ a.toNat < b.toNat + 2 ^ 36) = true →
      (a - b).toNat = off → a = b + UInt64.ofNat off ∧ off < regionSpan.toNat := by
    intro b hc ho
    simp only [decide_eq_true_eq] at hc
    rw [UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr hc.1)] at ho
    refine ⟨?_, by rw [regionSpan_toNat]; omega⟩
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_add, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (a := off) (by omega)]
    have := a.toNat_lt
    rw [Nat.mod_eq_of_lt (by omega)]; omega
  by_cases c1 : decide ((regionBase .arena).toNat ≤ a.toNat ∧ a.toNat < (regionBase .arena).toNat + 2 ^ 36) = true
  · rw [if_pos c1] at h; simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, h2⟩ := h; exact key _ c1 h2
  rw [if_neg c1] at h
  by_cases c2 : decide ((regionBase .data).toNat ≤ a.toNat ∧ a.toNat < (regionBase .data).toNat + 2 ^ 36) = true
  · rw [if_pos c2] at h; simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, h2⟩ := h; exact key _ c2 h2
  rw [if_neg c2] at h
  by_cases c3 : decide ((regionBase .out).toNat ≤ a.toNat ∧ a.toNat < (regionBase .out).toNat + 2 ^ 36) = true
  · rw [if_pos c3] at h; simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, h2⟩ := h; exact key _ c3 h2
  rw [if_neg c3] at h
  by_cases c4 : decide ((regionBase .pinned).toNat ≤ a.toNat ∧ a.toNat < (regionBase .pinned).toNat + 2 ^ 36) = true
  · rw [if_pos c4] at h; simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, h2⟩ := h; exact key _ c4 h2
  rw [if_neg c4] at h
  cases h

theorem decodeAddr_add {a : UInt64} {r : Region} {off i : Nat} (h : decodeAddr a = some (r, off))
    (hi : off + i < regionSpan.toNat) : decodeAddr (a + UInt64.ofNat i) = some (r, off + i) := by
  obtain ⟨rfl, _⟩ := decodeAddr_some h
  rw [UInt64.add_assoc, ← UInt64.ofNat_add]
  exact decodeAddr_base r _ hi

theorem foldlM_rel {α β γ : Type} (R : α → β → Prop) (f : α → γ → Option α) (g : β → γ → Option β)
    (hfg : ∀ a b x b', R a b → g b x = some b' → ∃ a', f a x = some a' ∧ R a' b') :
    ∀ (l : List γ) (a : α) (b : β) (b' : β), R a b → l.foldlM g b = some b' →
      ∃ a', l.foldlM f a = some a' ∧ R a' b'
  | [], a, b, b', hr, h => by cases h; exact ⟨a, rfl, hr⟩
  | x :: xs, a, b, b', hr, h => by
    simp only [List.foldlM_cons] at h ⊢
    cases hg : g b x with
    | none => rw [hg] at h; cases h
    | some b1 =>
      rw [hg] at h
      obtain ⟨a1, hf, hr1⟩ := hfg a b x b1 hr hg
      rw [hf]
      exact foldlM_rel R f g hfg xs a1 b1 b' hr1 h

/-- A copy into memory writes the bytes from its address and nothing else. -/
theorem copyIn_frame {m m' : Mem} {a : UInt64} {src : ByteArray} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hn : off + src.size ≤ regionSpan.toNat)
    (h : copyIn m a src = some m') : StoreFrame m m' r off src.size := by
  unfold copyIn at h
  suffices ∀ n, n ≤ src.size → ∀ m', (List.range n).foldlM
      (fun mm i => mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64) m = some m' →
      StoreFrame m m' r off n from this _ (Nat.le_refl _) m' h
  intro n
  induction n with
  | zero => intro _ m' h; cases h; exact StoreFrame.refl m r off 0
  | succ n ih =>
    intro hn' m' h
    rw [List.range_succ, List.foldlM_append] at h
    cases h1 : (List.range n).foldlM
        (fun mm i => mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64) m with
    | none => rw [h1] at h; cases h
    | some m1 =>
      rw [h1] at h
      simp only [Option.bind_eq_bind, Option.bind_some, List.foldlM_cons, List.foldlM_nil] at h
      have hf1 := ih (by omega) m1 h1
      cases h2 : m1.store (a + UInt64.ofNat n) 1 (src.get! n).toUInt64 with
      | none => rw [h2] at h; cases h
      | some m2 =>
        rw [h2] at h; cases h
        obtain ⟨r2, off2, hd2, hf2⟩ := store_frame h2
        rw [decodeAddr_add hd (by omega)] at hd2
        injection hd2 with hd2; injection hd2 with e1 e2; subst e1; subst e2
        exact hf1.trans (hf2.widen (by omega) (by omega)) (by omega)

theorem MemSame.copyIn_isSome {m₁ m₂ m₂' : Mem} (h : MemSame m₁ m₂) {a : UInt64} {s₁ s₂ : ByteArray}
    (hsz : s₁.size = s₂.size) (h2 : copyIn m₂ a s₂ = some m₂') :
    ∃ m₁', copyIn m₁ a s₁ = some m₁' ∧ MemSame m₁' m₂' := by
  unfold copyIn at h2 ⊢
  rw [hsz]
  refine foldlM_rel MemSame _ _ ?_ _ m₁ m₂ m₂' h h2
  intro x y i y' hxy hy
  have hs := MemSame.store_isSome hxy (a + UInt64.ofNat i) 1 (s₁.get! i).toUInt64 (s₂.get! i).toUInt64
  rw [hy] at hs
  cases hx : x.store (a + UInt64.ofNat i) 1 (s₁.get! i).toUInt64 with
  | none => rw [hx] at hs; cases hs
  | some x' =>
    refine ⟨x', rfl, ?_⟩
    obtain ⟨_, _, _, hfx⟩ := store_frame hx
    obtain ⟨_, _, _, hfy⟩ := store_frame hy
    exact hfx.same.symm.trans (hxy.trans hfy.same)

theorem copyOut_size {m : Mem} {a : UInt64} {n : Nat} {b : ByteArray} (h : copyOut m a n = some b) :
    b.size = n := by
  unfold copyOut at h
  suffices ∀ (l : List Nat) (acc b : ByteArray), l.foldlM (fun (acc : ByteArray) i => do
      let x ← m.load (a + UInt64.ofNat i) 1
      pure (acc.push x.toUInt8)) acc = some b → b.size = acc.size + l.length by
    have := this _ _ _ h; simpa using this
  intro l
  induction l with
  | nil => intro acc b h; cases h; simp
  | cons x xs ih =>
    intro acc b h
    simp only [List.foldlM_cons] at h
    cases hl : m.load (a + UInt64.ofNat x) 1 with
    | none => simp [hl] at h
    | some v =>
      simp only [hl, Option.bind_eq_bind, Option.bind_some, Option.pure_def] at h
      have := ih _ _ h
      have hp : (acc.push v.toUInt8).size = acc.size + 1 := by
        cases acc; simp [ByteArray.push, ByteArray.size]
      simp only [List.length_cons] at this ⊢
      omega

theorem MemSame.copyOut_same {m₁ m₂ : Mem} (h : MemSame m₁ m₂) {a : UInt64} {n : Nat} {b₂ : ByteArray}
    (h2 : copyOut m₂ a n = some b₂) : ∃ b₁, copyOut m₁ a n = some b₁ ∧ b₁.size = b₂.size := by
  have hs := MemSame.copyOut_isSome h a n
  rw [h2] at hs
  cases h1 : copyOut m₁ a n with
  | none => rw [h1] at hs; cases hs
  | some b₁ => exact ⟨b₁, rfl, by rw [copyOut_size h1, copyOut_size h2]⟩

/-- Where the string at `a` ends, in the region it is in: the bytes up to and
    including its terminating zero. -/
def strSpan (m : Mem) (a : UInt64) : Option (Region × Nat × Nat) := do
  let (r, off) ← decodeAddr a
  let bs := m.region r
  let n ← (List.range (bs.size - off)).find? (fun i => bs.get! (off + i) == 0)
  some (r, off, n + 1)

theorem mapM_congr_mem {α β : Type} (l : List α) (f g : α → Option β) (h : ∀ x ∈ l, f x = g x) :
    l.mapM f = l.mapM g := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.mapM_cons]
    rw [h x (List.mem_cons_self ..), ih fun y hy => h y (List.mem_cons_of_mem x hy)]

theorem foldr_congr_mem {α β : Type} (l : List α) (f g : α → β → β) (b : β)
    (h : ∀ x ∈ l, ∀ acc, f x acc = g x acc) : l.foldr f b = l.foldr g b := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldr_cons]
    rw [ih fun y hy => h y (List.mem_cons_of_mem x hy), h x (List.mem_cons_self ..)]

/-- Every region's size. -/
def Mem.sizes (m : Mem) (r : Region) : Nat := (m.region r).size

end AlgorithmLib.HProg.Sem

namespace AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem

-- ---------------------------------------------------------------------------
-- What a write leaves alone
-- ---------------------------------------------------------------------------

/-- An address a region decodes is that region's base plus the offset. -/
theorem decodeAddr_eq {a : UInt64} {r : Region} {off : Nat} (h : decodeAddr a = some (r, off)) :
    a = regionBase r + UInt64.ofNat off := by
  unfold decodeAddr at h
  simp only [List.findSome?] at h
  by_cases h1 : (decide (a ≥ regionBase Region.arena) &&
      decide (a - regionBase Region.arena < regionSpan)) = true
  · rw [if_pos h1] at h; injection h with h; injection h with e1 e2; subst e1; subst e2
    simp only [UInt64.ofNat_toNat]; rw [UInt64.add_comm, UInt64.sub_add_cancel]
  · rw [if_neg h1] at h
    by_cases h2 : (decide (a ≥ regionBase Region.data) &&
        decide (a - regionBase Region.data < regionSpan)) = true
    · rw [if_pos h2] at h; injection h with h; injection h with e1 e2; subst e1; subst e2
      simp only [UInt64.ofNat_toNat]; rw [UInt64.add_comm, UInt64.sub_add_cancel]
    · rw [if_neg h2] at h
      by_cases h3 : (decide (a ≥ regionBase Region.out) &&
          decide (a - regionBase Region.out < regionSpan)) = true
      · rw [if_pos h3] at h; injection h with h; injection h with e1 e2; subst e1; subst e2
        simp only [UInt64.ofNat_toNat]; rw [UInt64.add_comm, UInt64.sub_add_cancel]
      · rw [if_neg h3] at h
        by_cases h4 : (decide (a ≥ regionBase Region.pinned) &&
            decide (a - regionBase Region.pinned < regionSpan)) = true
        · rw [if_pos h4] at h; injection h with h; injection h with e1 e2; subst e1; subst e2
          simp only [UInt64.ofNat_toNat]; rw [UInt64.add_comm, UInt64.sub_add_cancel]
        · rw [if_neg h4] at h; cases h

theorem Mem.region_setRegion_ne (m : Mem) (r r' : Region) (b : ByteArray) (h : r ≠ r') :
    (m.setRegion r b).region r' = m.region r' := by
  cases r <;> cases r' <;> first | rfl | exact absurd rfl h

/-- Replacing a region's bytes leaves which bytes are reachable alone. -/
theorem _root_.AlgorithmLib.HProg.Sem.Mem.reachable_setRegion (m : Mem) (r : Region) (b : ByteArray) :
    (m.setRegion r b).reachable = m.reachable := by
  cases r <;> rfl

theorem _root_.AlgorithmLib.HProg.Sem.Mem.readable_setRegion (m : Mem) (r : Region) (b : ByteArray) :
    (m.setRegion r b).readable = m.readable := by
  cases r <;> rfl

/-- A store leaves the live allocations and the copies in flight alone. -/
theorem _root_.AlgorithmLib.HProg.Sem.Mem.store_meta {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64}
    (h : m.store a n v = some m') :
    m'.pinnedLive = m.pinnedLive ∧ m'.busy = m.busy := by
  unfold Mem.store at h
  cases hd : decodeAddr a with
  | none => simp [hd] at h
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [hd, Option.bind_eq_bind, Option.bind_some] at h
    split at h
    · cases h
    · injection h with h; subst h
      cases r <;> exact ⟨rfl, rfl⟩

/-- Liveness and growth of pinned memory. -/
theorem _root_.ByteArray.get!_append_left' (b c : ByteArray) (i : Nat) (h : i < b.size) :
    (b ++ c).get! i = b.get! i := by
  rw [show (b ++ c).get! i = (b ++ c).data[i]! from rfl, ByteArray.data_append,
    show b.get! i = b.data[i]! from rfl]
  have h' : i < b.data.size := h
  simp [getElem!_def, Array.getElem?_append_left h', Array.getElem?_eq_getElem h']

/-- A copy into host memory leaves them alone too. -/
theorem copyIn_meta {m m' : Mem} {a : UInt64} {src : ByteArray} (h : copyIn m a src = some m') :
    m'.pinnedLive = m.pinnedLive ∧ m'.busy = m.busy := by
  unfold copyIn at h
  have key : ∀ (is : List Nat) (mm mm' : Mem),
      is.foldlM (fun mm i => mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64) mm = some mm' →
      mm'.pinnedLive = mm.pinnedLive ∧ mm'.busy = mm.busy := by
    intro is
    induction is with
    | nil => intro mm mm' h; simp at h; subst h; exact ⟨rfl, rfl⟩
    | cons i is ih =>
      intro mm mm' h
      simp only [List.foldlM_cons, Option.bind_eq_bind] at h
      cases hs : mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64 with
      | none => simp [hs] at h
      | some m1 =>
        simp only [hs, Option.bind_some] at h
        obtain ⟨h1, h2⟩ := ih m1 mm' h
        obtain ⟨h3, h4⟩ := Mem.store_meta hs
        exact ⟨h1.trans h3, h2.trans h4⟩
  exact key _ m m' h

/-- Fewer copies in flight, the same allocations: whatever could be read still
    can be. -/
theorem Mem.readable_busy_sub {m m' : Mem} (hlive : m'.pinnedLive = m.pinnedLive)
    (hsub : ∀ b ∈ m.busy, b ∈ m'.busy) (r : Region) (off n : Nat)
    (h : m'.readable r off n = true) : m.readable r off n = true := by
  unfold Mem.readable at h ⊢
  rw [hlive] at h
  simp only [Bool.and_eq_true, Bool.not_eq_true', Bool.or_eq_true, List.all_eq_true] at h ⊢
  rcases h with h | ⟨h2, h3⟩
  · exact Or.inl h
  · exact Or.inr ⟨h2, fun b hb => h3 b (hsub b hb)⟩

/-- Setting the bytes at `off + i` for `i ∈ is` leaves every other byte and the
    size alone. -/
theorem foldl_set!_other (off j : Nat) (v : UInt64) :
    ∀ (is : List Nat) (b : ByteArray), (∀ i ∈ is, off + i ≠ j) →
      (is.foldl (fun (b : ByteArray) i =>
          b.set! (off + i) (((v >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8)) b).get! j = b.get! j
      ∧ (is.foldl (fun (b : ByteArray) i =>
          b.set! (off + i) (((v >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8)) b).size = b.size
  | [], _, _ => ⟨rfl, rfl⟩
  | i :: is, b, h => by
    simp only [List.foldl_cons]
    obtain ⟨h1, h2⟩ := foldl_set!_other off j v is _ (fun k hk => h k (List.mem_cons_of_mem _ hk))
    refine ⟨h1.trans ?_, h2.trans ?_⟩
    · exact Sem.get!_set!_ne _ _ _ _ (h i (List.mem_cons_self ..))
    · exact Sem.set!_size _ _ _

/-- **A store changes no byte outside the bytes it stores.** -/
theorem Mem.store_other {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64}
    (h : m.store a n v = some m') (a' : UInt64) (hout : ∀ i < n, a' ≠ a + UInt64.ofNat i) :
    m'.load a' 1 = m.load a' 1 := by
  unfold Mem.store at h
  cases hd : decodeAddr a with
  | none => simp [hd] at h
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [hd, Option.bind_eq_bind, Option.bind_some] at h
    split at h
    · cases h
    · injection h with h; subst h
      unfold Mem.load
      cases hd' : decodeAddr a' with
      | none => rfl
      | some q =>
        obtain ⟨r', off'⟩ := q
        simp only [Option.bind_eq_bind, Option.bind_some]
        by_cases hr : r = r'
        · subst hr
          have hne : ∀ i ∈ List.range n, off + i ≠ off' + 0 := by
            intro i hi e
            have hi := List.mem_range.mp hi
            apply hout i hi
            have e' : off' = off + i := by omega
            rw [decodeAddr_eq hd', decodeAddr_eq hd, e', UInt64.ofNat_add, UInt64.add_assoc]
          obtain ⟨hg, hs⟩ := foldl_set!_other off (off' + 0) v (List.range n) (m.region r) hne
          rw [Sem.region_setRegion_self, hs, Mem.readable_setRegion]
          simp only [show List.range 1 = [0] from rfl, List.foldr_cons, List.foldr_nil]
          rw [hg]
        · simp [Mem.region_setRegion_ne _ _ _ _ hr, Mem.readable_setRegion]

/-- Copying `src` to `a` changes no byte outside `[a, a + src.size)`. -/
theorem copyIn_other {m m' : Mem} {a : UInt64} {src : ByteArray}
    (h : copyIn m a src = some m') (a' : UInt64)
    (hout : ∀ i < src.size, a' ≠ a + UInt64.ofNat i) :
    m'.load a' 1 = m.load a' 1 := by
  unfold copyIn at h
  have key : ∀ (is : List Nat) (mm mm' : Mem), (∀ i ∈ is, i < src.size) →
      is.foldlM (fun mm i => mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64) mm = some mm' →
      mm'.load a' 1 = mm.load a' 1 := by
    intro is
    induction is with
    | nil => intro mm mm' _ h; simp at h; subst h; rfl
    | cons i is ih =>
      intro mm mm' hlt h
      simp only [List.foldlM_cons, Option.bind_eq_bind] at h
      cases hs : mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64 with
      | none => simp [hs] at h
      | some m1 =>
        simp only [hs, Option.bind_some] at h
        rw [ih m1 mm' (fun k hk => hlt k (List.mem_cons_of_mem _ hk)) h]
        apply Mem.store_other hs
        intro k hk e
        have hk0 : k = 0 := by omega
        subst hk0
        exact hout i (hlt i (List.mem_cons_self ..)) (by simpa using e)
  exact key _ m m' (fun i hi => List.mem_range.mp hi) h

theorem _root_.ByteArray.size_append' (b c : ByteArray) : (b ++ c).size = b.size + c.size := by
  simp

theorem Mem.region_withLive (m : Mem) (L : List (Nat × Nat)) (r : Region) :
    ({ m with pinnedLive := L } : Mem).region r = m.region r := by cases r <;> rfl

/-- A load that succeeds in `m'` succeeds in `m` with the same value, when the
    two hold the same bytes and every range `m'` lets the host touch, `m` does. -/
theorem Mem.load_congr {m m' : Mem} (hreg : ∀ r, m'.region r = m.region r)
    (hreach : ∀ r off n, m'.readable r off n = true → m.readable r off n = true)
    {a : UInt64} {n : Nat} {y : UInt64} (h : m'.load a n = some y) : m.load a n = some y := by
  unfold Mem.load at h ⊢
  cases hd : decodeAddr a with
  | none => simp [hd] at h
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [hd, Option.bind_eq_bind, Option.bind_some, hreg] at h ⊢
    split at h
    · cases h
    · rename_i hc
      rw [if_neg]
      · exact h
      · simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true', not_or] at hc ⊢
        refine ⟨hc.1, ?_⟩
        cases h' : m'.readable r off n
        · exact absurd h' hc.2
        · simp [hreach r off n h']

theorem Mem.readable_withLive {m : Mem} {L : List (Nat × Nat)} (hL : ∀ r ∈ L, r ∈ m.pinnedLive)
    (r : Region) (off n : Nat) (h : ({ m with pinnedLive := L } : Mem).readable r off n = true) :
    m.readable r off n = true := by
  unfold Mem.readable at h ⊢
  simp only [Bool.and_eq_true, Bool.not_eq_true', Bool.or_eq_true, List.any_eq_true] at h ⊢
  rcases h with h | ⟨⟨x, hx, h2⟩, h3⟩
  · exact Or.inl h
  · exact Or.inr ⟨⟨x, hL x hx, h2⟩, h3⟩

theorem Mem.load_shrink {m : Mem} {L : List (Nat × Nat)} (hL : ∀ r ∈ L, r ∈ m.pinnedLive)
    {a : UInt64} {n : Nat} {y : UInt64}
    (h : ({ m with pinnedLive := L } : Mem).load a n = some y) : m.load a n = some y :=
  Mem.load_congr (Mem.region_withLive m L) (Mem.readable_withLive hL) h

/-- Two memories holding the same bytes agree on every load that succeeds in
    both: which ranges may be touched decides whether a load succeeds, never
    what it reads. -/
theorem Mem.load_agree {m m' : Mem} (hreg : ∀ r, m'.region r = m.region r)
    {a : UInt64} {n : Nat} {x y : UInt64} (hx : m.load a n = some x) (hy : m'.load a n = some y) :
    x = y := by
  unfold Mem.load at hx hy
  cases hd : decodeAddr a with
  | none => simp [hd] at hx
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [hd, Option.bind_eq_bind, Option.bind_some, hreg] at hx hy
    split at hx <;> split at hy <;> simp_all

theorem Mem.region_withBusy (m : Mem) (B : List Busy) (r : Region) :
    ({ m with busy := B } : Mem).region r = m.region r := by cases r <;> rfl

theorem _root_.List.foldr_congr_mem {α β : Type} (f g : α → β → β) (b : β) :
    ∀ (l : List α), (∀ a ∈ l, ∀ x, f a x = g a x) → l.foldr f b = l.foldr g b
  | [], _ => rfl
  | a :: l, h => by
    simp only [List.foldr_cons]
    rw [List.foldr_congr_mem f g b l (fun a' ha' => h a' (List.mem_cons_of_mem _ ha')),
      h a (List.mem_cons_self ..)]

/-- A fresh allocation leaves readable whatever was. -/
theorem Mem.readable_grow (m : Mem) (extra : ByteArray) (x : Nat × Nat) (r : Region) (off n : Nat)
    (h : m.readable r off n = true) :
    ({ m with pinned := m.pinned ++ extra, pinnedLive := x :: m.pinnedLive } : Mem).readable r off n
      = true := by
  unfold Mem.readable at h ⊢
  simp only [Bool.and_eq_true, Bool.not_eq_true', Bool.or_eq_true, List.any_eq_true,
    List.any_cons] at h ⊢
  rcases h with h | ⟨⟨y, hy, h2⟩, h3⟩
  · exact Or.inl h
  · exact Or.inr ⟨Or.inr ⟨y, hy, h2⟩, h3⟩

theorem Mem.load_grow {m : Mem} (extra : ByteArray) (x : Nat × Nat) {a : UInt64} {n : Nat}
    {v : UInt64} (h : m.load a n = some v) :
    ({ m with pinned := m.pinned ++ extra, pinnedLive := x :: m.pinnedLive } : Mem).load a n
      = some v := by
  unfold Mem.load at h ⊢
  cases hd : decodeAddr a with
  | none => simp [hd] at h
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [hd, Option.bind_eq_bind, Option.bind_some] at h ⊢
    split at h
    · cases h
    · rename_i hc
      have hc0 := hc
      simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true', not_or] at hc
      cases r
      case pinned =>
        have hsz : off + n ≤ m.pinned.size := by simp [Mem.region] at hc; omega
        rw [if_neg]
        · rw [← h]
          congr 1
          apply List.foldr_congr_mem
          intro i hi acc
          simp only [Mem.region]
          rw [ByteArray.get!_append_left' _ _ _ (by simp at hi; omega)]
        · have hr : m.readable .pinned off n = true := by
            revert hc0; cases m.readable .pinned off n <;> simp
          have hg := Mem.readable_grow m extra x .pinned off n hr
          simp only [hg, Bool.not_true, Bool.or_false, decide_eq_true_eq, Mem.region,
            ByteArray.size_append]
          omega
      all_goals (rw [if_neg]; · exact h
                 simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true', not_or] at hc ⊢
                 exact ⟨hc.1, by simp [Mem.readable_grow m extra x _ off n (by simpa using hc.2)]⟩)

/-- **No byte changes value**: a byte readable at `a` before and after holds
    the same value. Allocation may make new bytes readable and freeing may make
    old ones unreadable; neither is a write, and a frame bounds writes. -/
def Unchanged (m m' : Mem) (a : UInt64) : Prop :=
  ∀ x y, m.load a 1 = some x → m'.load a 1 = some y → x = y

theorem Unchanged.of_eq {m m' : Mem} {a : UInt64} (h : m'.load a 1 = m.load a 1) :
    Unchanged m m' a := by
  intro x y hx hy; rw [h, hx] at hy; injection hy

theorem Unchanged.refl (m : Mem) (a : UInt64) : Unchanged m m a := Unchanged.of_eq rfl

/-- Memories holding the same bytes: no byte changes value between them. -/
theorem Unchanged.of_region {m m' : Mem} (hreg : ∀ r, m'.region r = m.region r) (a : UInt64) :
    Unchanged m m' a := fun _ _ hx hy => Mem.load_agree hreg hx hy

theorem Unchanged.trans' {m m' m'' : Mem} {a : UInt64} (h1 : m'.load a 1 = m.load a 1)
    (h2 : ∀ r, m''.region r = m'.region r) : Unchanged m m'' a := by
  intro x y hx hy
  -- `m''` may refuse where `m'` does not, but not read differently
  cases h' : m'.load a 1 with
  | none =>
      -- then `m` refused too, contradicting `hx`
      rw [h1] at h'; rw [h'] at hx; cases hx
  | some z =>
      have := Mem.load_agree h2 h' hy
      rw [h1, hx] at h'; cases h'; exact this

end AlgorithmLib.HProg
