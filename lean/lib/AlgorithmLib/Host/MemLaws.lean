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

open AlgorithmLib.IR
open AlgorithmLib.HProg

/-- Two memories that differ at most in their bytes. -/
structure MemSame (m₁ m₂ : Mem) : Prop where
  size : ∀ r, (m₁.region r).size = (m₂.region r).size
  live : m₁.pinnedLive = m₂.pinnedLive
  busy : m₁.busy = m₂.busy
  frozen : m₁.frozen = m₂.frozen

/-- A kernel oracle that hands back buffers of the lengths it was given. -/
def KeepsSize (k : Launch → List ByteArray → List ByteArray) : Prop :=
  ∀ l bs, (k l bs).map ByteArray.size = bs.map ByteArray.size

theorem MemSame.refl (m : Mem) : MemSame m m := ⟨fun _ => rfl, rfl, rfl, rfl⟩

theorem MemSame.symm {m₁ m₂ : Mem} (h : MemSame m₁ m₂) : MemSame m₂ m₁ :=
  ⟨fun r => (h.size r).symm, h.live.symm, h.busy.symm, h.frozen.symm⟩

theorem MemSame.trans {m₁ m₂ m₃ : Mem} (h₁ : MemSame m₁ m₂) (h₂ : MemSame m₂ m₃) : MemSame m₁ m₃ :=
  ⟨fun r => (h₁.size r).trans (h₂.size r), h₁.live.trans h₂.live, h₁.busy.trans h₂.busy,
    h₁.frozen.trans h₂.frozen⟩

theorem MemSame.retire {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (c : Clock) :
    MemSame (m₁.retire c) (m₂.retire c) :=
  ⟨fun r => by have := h.size r; cases r <;> exact this, h.live, by simp only [Mem.retire, h.busy], h.frozen⟩

theorem MemSame.readable {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (r : Region) (off n : Nat) :
    m₁.readable r off n = m₂.readable r off n := by
  simp only [Mem.readable, h.frozen, h.live, h.busy]

theorem MemSame.reachable {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (r : Region) (off n : Nat) :
    m₁.reachable r off n = m₂.reachable r off n := by
  simp only [Mem.reachable, h.frozen, h.live, h.busy]

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
  refine ⟨fun r' => ?_, ?_, ?_, ?_⟩
  · by_cases hr : r' = r
    · subst hr; rw [region_setRegion_self, region_setRegion_self, hb]
    · rw [region_setRegion_ne _ _ hr, region_setRegion_ne _ _ hr, h.size r']
  all_goals cases r <;> first | exact h.live | exact h.busy | exact h.frozen

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

end AlgorithmLib.HProg.Sem
