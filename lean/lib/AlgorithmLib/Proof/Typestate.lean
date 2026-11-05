module
public import AlgorithmLib.Proof.ProgHoare
public import AlgorithmLib.Proof.Rules
public import AlgorithmLib.Host.DevSpec
public import AlgorithmLib.Host.Lifecycle
public import AlgorithmLib.Host.ExtLife
public import AlgorithmLib.Host.Summary
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Proof.Typestate` — what the typestate and the semantics say

The facts the checker (`Check`) and the library proofs (`LibSpec`) read a body
by: what the standard typestates (`TState`: lifecycle parts, rooms, cells)
promise about a world and how a store or a call moves them (`store_parts`,
`afterStore_forget`, `TState.holds_obsCall`), what an operation computes on
known operands (`evalOp_*`), what a load from a known cell reads, and the
arguments a host hands an entry point (`runArgs`). `Live` is the device
calls' typestate.
-/

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare
open AlgorithmLib.HProg.DevSpec

namespace AlgorithmLib.HProg.Sem

/-- Each slot holds a scalar of its type, in order. -/
def ArgsTy (Γ : Env) : List R → List ClifTy → Prop
  | [], [] => True
  | r :: rs, t :: ts => (∃ x, Γ[r]? = some (.sc t x)) ∧ ArgsTy Γ rs ts
  | _, _ => False

def isF (t : ClifTy) : Bool := t == .f32 || t == .f64

/-- Two operands of one integer type: that type. -/
def sameInt : List ClifTy → Option ClifTy
  | [t, u] => if t == u && t.isInt then some t else none
  | _ => none

theorem sameInt_eq {ts : List ClifTy} {T : ClifTy} (h : sameInt ts = some T) :
    ts = [T, T] ∧ T.isInt = true := by
  unfold sameInt at h; split at h
  · simp only [Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨rfl, h2⟩, rfl⟩ := h; exact ⟨rfl, h2⟩
  · cases h

theorem args2 {Γ : Env} {a b : R} {t u : ClifTy} (h : ArgsTy Γ [a, b] [t, u]) :
    (∃ x, Γ[a]? = some (.sc t x)) ∧ ∃ y, Γ[b]? = some (.sc u y) := ⟨h.1, h.2.1⟩

theorem bin_sound {Γ : Env} {a b : R} {ts : List ClifTy} {T : ClifTy}
    {f : ClifTy → UInt64 → UInt64 → Option V} (hargs : ArgsTy Γ [a, b] ts) (h : sameInt ts = some T)
    (hf : ∀ x y, ∃ z, f T x y = some (.sc T z)) : ∃ z, bin Γ a b f = some (.sc T z) := by
  obtain ⟨rfl, hi⟩ := sameInt_eq h
  obtain ⟨⟨x, hx⟩, ⟨y, hy⟩⟩ := args2 hargs
  simpa [bin, get, hx, hy, hi] using hf x y

end AlgorithmLib.HProg.Sem

namespace AlgorithmLib.Prog

/-- Binding past everything an environment extends keeps the extension. -/
theorem bindAt_append {Γ δ : Env} {n : Nat} (vs : List V) (h : Γ.size ≤ n) :
    bindAt (Γ ++ δ) n vs = Γ ++ bindAt δ (n - Γ.size) vs := by
  apply Array.toList_inj.mp
  simp only [bindAt, Array.take, Array.toList_append, Array.toList_extract, Array.toList_replicate,
    Array.size_append, List.append_assoc, List.extract_eq_drop_take, List.drop_zero, List.take_append,
    Array.length_toList, Nat.sub_zero]
  rw [List.take_of_length_le (by simp; omega)]
  have : n - (Γ.size + δ.size) = n - Γ.size - δ.size := by omega
  rw [this]

/-- The memory's address binds the arena's base: nothing about the world is
    asked. -/
theorem wp_memory {cfg : Cfg} {d : Nat} {J : Post} {α : Type}
    (c : Slot .i64 → Prog Slot Lvl α) (Q : α → Env → World → Prop) :
    wp cfg d J (.op .memory c) Q =
      fun Γ w => wp cfg d J (c Γ.size) Q (Γ.push (.sc .i64 (regionBase .arena))) w := by
  funext Γ w
  simp only [wp_op, Op'.erase, evalOp, Option.some.injEq, forall_eq', reduceCtorEq, false_implies, true_and]

open Contracts (Part Fact TState afterStore afterStore_sound)

theorem Part.get_obsCall (p : Part) (w : World) (c : Callee) (vs : List V) :
    p.get (obsCall w c vs) = p.get w := by
  cases p <;> rfl

theorem TState.holds_obsCall {S : TState} {w : World} {c : Callee} {vs : List V} (h : S.holds w) :
    S.holds (obsCall w c vs) := by
  intro x hx
  have := h x hx
  cases x with
  | part p b => exact (Part.get_obsCall p w c vs).trans this
  | room _ _ => exact this
  | cell _ _ => exact this
  | held k _ => cases k <;> exact this
  | opened k _ => cases k <;> exact this
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem this rfl rfl
  | cstr _ => exact this
  | devSeq => exact this
  | oracles => exact this
  | cstrIn _ _ => exact this
  | roomArg _ _ => exact this
  | devBuf _ _ => exact this
  | pinnedUsed _ => exact this
  | streams _ | events _ | evLive _ | readyAll _ | evAll _ => exact this
  | drvInit | retained _ | nbufs _ | alloc _ _ | nmods _ | nfuncs _ | mod _ _ | fn _ _ _ | modText _ _ | fnText _ _ _ | nblas _ | blasOn _ _ | ngraphs _ | graph _ | nexecs _ | exec _ | cap _ _ _ _ | text _ _ _ _ => exact this
  | nwg _ | wgo _ _ => exact this
  | wg1 _ s => cases s <;> exact this
  | wgText _ _ _ => exact this
  | threads _ _ _ _ _ => exact this
  | owns _ _ => exact this

theorem ofInt_nat_val {n : Nat} (h : n < 2 ^ 64) :
    UInt64.ofNat ((n : Int).emod (1 <<< 64)).toNat = UInt64.ofNat n := by
  have e : (n : Int).emod ((1 <<< 64 : Nat) : Int) = n := by
    rw [Nat.one_shiftLeft]
    exact Int.emod_eq_of_lt (by omega) (by omega)
  rw [e, Int.toNat_natCast]

/-- An answer written as a count below the width, read back as its bits. -/
theorem ofInt_toNat {ty t : ClifTy} {n k : Nat} {x : UInt64} (h : ofInt ty (n : Int) = .sc t x)
    (hm : (widthMask ty).toNat = 2 ^ k - 1) (hk : k ≤ 64) (hn : n < 2 ^ k) : x.toNat = n := by
  have h64 : n < 2 ^ 64 := Nat.lt_of_lt_of_le hn (Nat.pow_le_pow_right (by decide) hk)
  simp only [ofInt, norm, ofInt_nat_val h64, V.sc.injEq] at h
  obtain ⟨-, rfl⟩ := h
  have e2 : (UInt64.ofNat n).toNat = n := by simp; omega
  rw [UInt64.toNat_and, hm, e2, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hn]

theorem ofInt_neg_one_i64 {t : ClifTy} {x : UInt64} (h : ofInt .i64 (-1) = .sc t x) :
    x = 0xFFFFFFFFFFFFFFFF := by
  have e : ofInt .i64 (-1) = .sc .i64 0xFFFFFFFFFFFFFFFF := by rfl
  rw [e, V.sc.injEq] at h; exact h.2.symm

theorem ofInt_neg_one_i32 {t : ClifTy} {x : UInt64} (h : ofInt .i32 (-1) = .sc t x) : x = 0xFFFFFFFF := by
  have e : ofInt .i32 (-1) = .sc .i32 0xFFFFFFFF := by rfl
  rw [e, V.sc.injEq] at h; exact h.2.symm

/-- **A file read answers a 64-bit count: `-1`, or at most the bytes it was
    asked for.** -/
theorem fsRead_answer {h buf len off : UInt64} {w : World} {v : V} {w' : World}
    (hc : fsCall .read [h, buf, len, off] w = some (some v, w')) :
    ∃ x, v = .sc .i64 x ∧ (x.toNat = 18446744073709551615 ∨ (0 < len.toNat → x.toNat ≤ len.toNat)) := by
  simp only [fsCall] at hc
  obtain ⟨c, hd, he⟩ := Option.map_eq_some_iff.mp hc
  simp only [Prod.mk.injEq] at he
  obtain ⟨he, -⟩ := he
  simp only [fsDecode, fsDRead] at hd
  rw [Option.bind_eq_bind, Option.bind_eq_some_iff] at hd
  obtain ⟨k, -, hd⟩ := hd
  split at hd
  · simp only [Option.some.injEq] at hd; subst hd
    simp only [fsNeg, fsEffect, Option.some.injEq] at he
    subst he
    exact ⟨0xFFFFFFFFFFFFFFFF, rfl, .inl rfl⟩
  · rw [Option.bind_eq_bind, Option.bind_eq_some_iff] at hd
    obtain ⟨_, -, hd⟩ := hd
    simp only [Option.some.injEq] at hd; subst hd
    simp only [fsEffect, Option.some.injEq] at he
    subst he
    have hs : ((w.fsContents k).extract off.toNat (off.toNat + len.toNat)).size ≤ len.toNat := by
      rw [ByteArray.size_extract]; omega
    have := len.toNat_lt
    obtain ⟨x, hx⟩ : ∃ x, ofInt .i64 (((w.fsContents k).extract off.toNat (off.toNat + len.toNat)).size : Int)
        = .sc .i64 x := ⟨_, rfl⟩
    refine ⟨x, hx, .inr fun _ => ?_⟩
    rw [ofInt_toNat hx (k := 64) (by decide) (by decide) (by omega)]
    exact hs

/-- A typestate asks less once some of its facts are dropped. -/
theorem TState.holds_filter {S : TState} {w : World} (p : Fact → Bool) (h : S.holds w) :
    TState.holds (S.filter p) w := fun x hx => h x (List.mem_filter.mp hx).1

/-- Whether a typestate names every fact another does. -/
def TState.sub (S S' : TState) : Bool := S.all (S'.contains ·)

/-- A typestate asks less than one that names every fact it names. -/
theorem TState.holds_sub {S S' : TState} (h : TState.sub S S' = true) {w : World}
    (hS : S'.holds w) : S.holds w := by
  intro x hx
  exact hS x (by have := List.all_eq_true.mp h x hx; simpa using this)

/-- Each fact of a list is one of `S'`'s. -/
def TState.AllIn (S' : TState) : List Fact → Prop
  | [] => True
  | x :: xs => x ∈ S' ∧ TState.AllIn S' xs

/-- A typestate asks less than one whose facts include each of its own, found
    by place where a fact names a value the kernel cannot compare. -/
theorem TState.holds_allIn {S S' : TState} (h : TState.AllIn S' S) {w : World}
    (hS : S'.holds w) : S.holds w := by
  intro x hx
  induction S with
  | nil => cases hx
  | cons y ys ih =>
      rcases List.mem_cons.mp hx with rfl | hx
      · exact hS _ h.1
      · exact ih h.2 hx

/-- The typestate ordinary device calls need: the context is live, and no
    stream is being captured. -/
abbrev Live : World → Prop :=
  TState.holds [.part .cuda true, .part .capturing false]

theorem tyBytes_le (ty : ClifTy) : tyBytes ty ≤ 2 ^ 64 := by cases ty <;> decide

export AlgorithmLib.HProg.Contracts (Fits)

/-- Bytes within a length the program was handed. -/
theorem fits_of_roomArg {S : TState} {w : World} {a x : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hS : S.holds w) (hm : Contracts.Fact.roomArg r x ∈ S)
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned) (hb : off + n ≤ x.toNat) :
    Fits w.mem a n :=
  ⟨r, off, hd, hp, by have := (hS _ hm).1; omega,
    by have := (hS _ hm).2.1; omega, Contracts.roomArg_lentOk (hS _ hm)⟩

/-- Bytes at a value past an address, within a length the program was handed. -/
theorem fits_of_roomArg_add {S : TState} {w : World} {b x X : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hS : S.holds w) (hm : Contracts.Fact.roomArg r X ∈ S)
    (hd : decodeAddr b = some (r, off)) (hp : r ≠ .pinned) (hb : off + x.toNat + n ≤ X.toNat)
    (hn : 0 < n) : Fits w.mem (b + x) n := by
  have hx := hS _ hm
  have hd' : decodeAddr (b + x) = some (r, off + x.toNat) := by
    have := Sem.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; have := hx.2.1; omega)
    rwa [UInt64.ofNat_toNat] at this
  exact fits_of_roomArg hS hm hd' hp (by omega)

/-- Bytes at a value past an address, within a room the typestate names. -/
theorem fits_of_room_add {S : TState} {w : World} {b x : UInt64} {n N : Nat} {r : Region} {off : Nat}
    (hS : S.holds w) (hm : Contracts.Fact.room r N ∈ S)
    (hd : decodeAddr b = some (r, off)) (hp : r ≠ .pinned) (hb : off + x.toNat + n ≤ N)
    (hN : N ≤ 2 ^ 36) (hn : 0 < n) : Fits w.mem (b + x) n := by
  have hx : N ≤ w.mem.sizes r := (hS _ hm).1
  have hd' : decodeAddr (b + x) = some (r, off + x.toNat) := by
    have := Sem.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
    rwa [UInt64.ofNat_toNat] at this
  exact ⟨r, off + x.toNat, hd', hp, by omega, by omega, Contracts.room_lentOk (hS _ hm)⟩

/-- A place a literal past a sum: the same place as the literal added first. -/
theorem fits_regroup {m : Mem} {b x c : UInt64} {n : Nat} (h : Fits m (b + c + x) n) :
    Fits m (b + x + c) n := by
  rwa [UInt64.add_assoc, UInt64.add_comm x c, ← UInt64.add_assoc]

theorem store_parts {w : World} {m : Mem} {a : UInt64} {n : Nat} {b : UInt64}
    (hz : m.busy = w.mem.busy)
    (p : Part) : p.get { obsStore w a n b with mem := m } = p.get w := by
  cases p <;> first | rfl | (show m.busy.isEmpty = w.mem.busy.isEmpty; rw [hz])

theorem afterStore_forget {S : TState} {w : World} {a v : UInt64} {n : Nat}
    (h : (afterStore (some a) (some v) n S).holds w) : (afterStore (some a) none n S).holds w := by
  unfold afterStore at h ⊢
  dsimp only at h ⊢
  split at h
  · exact fun x hx => h x (List.mem_cons_of_mem _ hx)
  · exact h

theorem all_cons_true {β : Type} {f : β → Bool} {a : β} {l : List β} (h1 : f a = true)
    (h2 : l.all f = true) : (a :: l).all f = true := by
  simp only [List.all_cons, h1, h2, Bool.and_self]

/-- Lanes stored one after another keep every region's size and the copies in flight. -/
theorem foldlM_lane_stores_inv {addr : UInt64} {lw : Nat} : ∀ (l : List (UInt64 × Nat)) (m0 m : Mem),
    l.foldlM (fun mm (p : UInt64 × Nat) => mm.store (addr + UInt64.ofNat (p.2 * lw)) lw p.1) m0 = some m →
    m.sizes = m0.sizes ∧ m.busy = m0.busy
  | [], m0, m, h => by simp at h; subst h; exact ⟨rfl, rfl⟩
  | p :: l, m0, m, h => by
      simp only [List.foldlM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨m1, h1, h2⟩ := h
      obtain ⟨hs, hb⟩ := foldlM_lane_stores_inv l m1 m h2
      exact ⟨hs.trans (Contracts.store_sizes_eq h1),
        hb.trans (Contracts.store_busy h1)⟩

/-- A store to any address keeps every fact but the cells. -/
theorem afterStore_nonCell {S : TState} {w : World} {a : UInt64} {n : Nat}
    (h : (afterStore (some a) none n S).holds w) : TState.holds (S.filter (!·.isCell)) w := by
  intro f hf
  obtain ⟨hm, hc⟩ := List.mem_filter.mp hf
  apply h
  refine List.mem_filter.mpr ⟨hm, ?_⟩
  cases f <;> simp_all [Contracts.storeKeeps, Contracts.Fact.isCell]

theorem evalOp_iadd {m : Mem} {Γ : Env} {a b : R} {t : ClifTy} {x y : UInt64}
    (ha : Γ[a]? = some (.sc t x)) (hb : Γ[b]? = some (.sc t y)) (ht : t.isInt = true) :
    evalOp m Γ (.iadd a b) = some (norm t (x + y)) := by
  show bin Γ a b _ = _
  simp only [bin, Sem.get, ha, hb]
  have : (t == t) = true := by cases t <;> rfl
  simp [ht, this]

/-- A 64-bit addition over values known symbolically: their sum, as a term. -/
theorem evalOp_iadd64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.iadd a b) = some (.sc .i64 (x + y)) := by
  rw [evalOp_iadd ha hb rfl]
  simp only [norm, widthMask]
  congr 2
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and]
  have := (x + y).toNat_lt
  show (x + y).toNat &&& (2^64 - 1) = (x + y).toNat
  rw [Nat.and_two_pow_sub_one_eq_mod]
  omega

/-- The lesser of two 64-bit values is at most each. -/
theorem umin_bounds (x y : UInt64) :
    (UInt64.ofNat (min x.toNat y.toNat)).toNat ≤ x.toNat ∧ (UInt64.ofNat (min x.toNat y.toNat)).toNat ≤ y.toNat := by
  have hx := x.toNat_lt
  have hy := y.toNat_lt
  rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  omega

/-- The lesser of two 64-bit values known symbolically, as a term. -/
theorem evalOp_umin64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.ibin .umin a b) = some (.sc .i64 (UInt64.ofNat (min x.toNat y.toNat))) := by
  show bin Γ a b _ = _
  simp only [bin, Sem.get, ha, hb]
  simp [ibinOp, ClifTy.isInt, norm64]
  have hw : (widthMask ClifTy.i64).toNat = 2 ^ 64 - 1 := rfl
  rw [hw, Nat.and_two_pow_sub_one_eq_mod, Nat.and_two_pow_sub_one_eq_mod,
    Nat.mod_eq_of_lt x.toNat_lt, Nat.mod_eq_of_lt y.toNat_lt]

/-- The leading zeros of a 64-bit value, as `clz` counts them. -/
def clz64 (x : UInt64) : UInt64 :=
  UInt64.ofNat ((bitsDown 64 x.toNat).takeWhile (· == false)).length

/-- A 64-bit value's bits in reverse order, as `bitrev` lays them. -/
def bitrev64 (x : UInt64) : UInt64 :=
  UInt64.ofNat ((List.range 64).foldl
    (fun acc i => if (x &&& widthMask .i64).toNat.testBit i then acc ||| (1 <<< (64 - 1 - i)) else acc) 0)

theorem toNat_and_widthMask64 (x : UInt64) : x.toNat &&& (widthMask .i64).toNat = x.toNat := by
  rw [← UInt64.toNat_and, and_wide64]

theorem evalOp_clz64 {m : Mem} {Γ : Env} {a : R} {x : UInt64} (ha : Γ[a]? = some (.sc .i64 x)) :
    evalOp m Γ (.iun .clz a) = some (.sc .i64 (clz64 x)) := by
  show un Γ a _ _ = _
  simp only [un, Sem.get, ha]
  simp [iunOp, ClifTy.isInt, ClifTy.width, toNat_and_widthMask64, norm64, clz64]

theorem evalOp_bitrev64 {m : Mem} {Γ : Env} {a : R} {x : UInt64} (ha : Γ[a]? = some (.sc .i64 x)) :
    evalOp m Γ (.iun .bitrev a) = some (.sc .i64 (bitrev64 x)) := by
  show un Γ a _ _ = _
  simp only [un, Sem.get, ha]
  simp [iunOp, ClifTy.isInt, ClifTy.width, norm64, bitrev64]

theorem bitsDown_getElem (n k : Nat) (hk : k < (bitsDown 64 n).length) :
    (bitsDown 64 n)[k] = n.testBit (63 - k) := by
  simp [bitsDown] at hk ⊢

/-- With `c` leading zeros a value is below `2^(64-c)`, and at least
    `2^(63-c)` where it has a one. -/
theorem clz_bounds (n : Nat) (hn : n < 2^64) :
    n < 2^(64 - ((bitsDown 64 n).takeWhile (· == false)).length) ∧
    (((bitsDown 64 n).takeWhile (· == false)).length < 64 →
      2^(63 - ((bitsDown 64 n).takeWhile (· == false)).length) ≤ n) := by
  generalize hT : (bitsDown 64 n).takeWhile (· == false) = T
  generalize hD : (bitsDown 64 n).dropWhile (· == false) = D
  have hsplit : T ++ D = bitsDown 64 n := by rw [← hT, ← hD]; exact List.takeWhile_append_dropWhile
  have hlen : (bitsDown 64 n).length = 64 := by simp [bitsDown]
  have hlen' : T.length + D.length = 64 := by rw [← List.length_append, hsplit, hlen]
  have hall : ∀ x ∈ T, (x == false) = true := by
    have := List.all_takeWhile (p := (· == false)) (l := bitsDown 64 n)
    rw [hT, List.all_eq_true] at this; exact this
  refine ⟨?_, ?_⟩
  · apply Nat.lt_pow_two_of_testBit
    intro i hi
    by_cases h64 : i < 64
    · have hk : 63 - i < T.length := by omega
      have e1 : (bitsDown 64 n)[63 - i]'(by omega) = T[63 - i] := by
        simp only [← hsplit]; rw [List.getElem_append_left hk]
      have h1 := hall _ (List.getElem_mem hk)
      rw [← e1, bitsDown_getElem] at h1
      have e : 63 - (63 - i) = i := by omega
      rw [e] at h1; simpa using h1
    · exact Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hn (Nat.pow_le_pow_right (by decide) (by omega)))
  · intro hc
    have hne : D ≠ [] := by intro h; rw [h] at hlen'; simp at hlen'; omega
    have hh := List.head_dropWhile_not (· == false) (l := bitsDown 64 n) (by rw [hD]; exact hne)
    have e1 : (bitsDown 64 n)[T.length]'(by omega)
        = D[0]'(by cases D with | nil => exact absurd rfl hne | cons => simp) := by
      simp only [← hsplit]; rw [List.getElem_append_right (by omega)]; simp
    have e2 : ((bitsDown 64 n).dropWhile (· == false)).head (by rw [hD]; exact hne)
        = D[0]'(by cases D with | nil => exact absurd rfl hne | cons => simp) := by
      subst hD; simp [List.head_eq_getElem]
    rw [e2, ← e1, bitsDown_getElem] at hh
    have : n.testBit (63 - T.length) = true := by simpa using hh
    exact Nat.ge_two_pow_of_testBit this

theorem bitrev64_zero : bitrev64 0 = 0 := by decide

theorem clz64_toNat (n : UInt64) :
    (clz64 n).toNat = ((bitsDown 64 n.toNat).takeWhile (· == false)).length := by
  have h := (List.takeWhile_prefix (· == false) (l := bitsDown 64 n.toNat)).length_le
  have h2 : (bitsDown 64 n.toNat).length = 64 := by simp [bitsDown]
  simp only [clz64, UInt64.toNat_ofNat']
  apply Nat.mod_eq_of_lt; omega

/-- A bit-reversed index, shifted down by one past the leading zeros of a bound
    above the index, stays below the bound: it is the index's low bits
    reversed, as many as the bound's highest bit is high. Stated as the
    arithmetic reads it, where the `+ 1` may be a computed one. -/
theorem bitrev_index_lt (j n c : UInt64) (hc : c = 1) :
    n.toNat ≤ j.toNat ∨ (bitrev64 j >>> ((clz64 n + c) % 64)).toNat < n.toNat := by
  subst hc
  rcases Nat.lt_or_ge j.toNat n.toNat with hj | hj
  · right
    have hb := clz_bounds n.toNat n.toNat_lt
    have hz := clz64_toNat n
    generalize ((bitsDown 64 n.toNat).takeWhile (· == false)).length = cv at hb hz
    obtain ⟨hlt, hge⟩ := hb
    have hc1 : cv < 64 := by
      apply Nat.lt_of_not_le; intro h
      have : 64 - cv = 0 := by omega
      rw [this] at hlt; omega
    by_cases h63 : cv = 63
    · subst h63
      have hn : n.toNat < 2 := by simpa using hlt
      have hj0 : j = 0 := by apply UInt64.toNat_inj.mp; simp; omega
      subst hj0; rw [bitrev64_zero]; simp; omega
    · have hs : ((clz64 n + 1) % 64).toNat = cv + 1 := by
        simp [UInt64.toNat_add, UInt64.toNat_mod, hz]; omega
      rw [UInt64.toNat_shiftRight, hs, Nat.mod_eq_of_lt (show cv + 1 < 64 by omega)]
      have hr := (bitrev64 j).toNat_lt
      have h2 : (bitrev64 j).toNat >>> (cv + 1) < 2^(63 - cv) := by
        rw [Nat.shiftRight_eq_div_pow]
        apply Nat.div_lt_of_lt_mul
        rw [← Nat.pow_add]; have : cv + 1 + (63 - cv) = 64 := by omega
        rw [this]; exact hr
      have := hge hc1; omega
  · left; exact hj

theorem evalOp_ireduce32_64 {m : Mem} {Γ : Env} {a : R} {x : UInt64} (ha : Γ[a]? = some (.sc .i64 x)) :
    evalOp m Γ (.ireduce32 a) = some (.sc .i32 (x &&& widthMask .i32)) := by
  show un Γ a _ _ = _
  simp only [un, Sem.get, ha]
  rfl

theorem evalOp_isub64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.isub a b) = some (.sc .i64 (x - y)) := by
  show bin Γ a b _ = _
  simp only [bin, Sem.get, ha, hb]
  have e1 : (ClifTy.i64 == ClifTy.i64) = true := rfl
  have e2 : ClifTy.i64.isInt = true := rfl
  simp [e1, e2, norm64]

theorem evalOp_imul64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.imul a b) = some (.sc .i64 (x * y)) := by
  show bin Γ a b _ = _
  simp only [bin, Sem.get, ha, hb]
  have e1 : (ClifTy.i64 == ClifTy.i64) = true := rfl
  have e2 : ClifTy.i64.isInt = true := rfl
  simp [e1, e2, norm64]

theorem evalOp_udiv64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) (hy : y ≠ 0) :
    evalOp m Γ (.udiv a b) = some (.sc .i64 (x / y)) := by
  show bin Γ a b _ = _
  simp only [bin, Sem.get, ha, hb]
  have e1 : (ClifTy.i64 == ClifTy.i64) = true := rfl
  have e2 : ClifTy.i64.isInt = true := rfl
  simp [e1, e2, norm64, hy]

theorem evalOp_select64 {m : Mem} {Γ : Env} {c a b : R} {tc : ClifTy} {z x y : UInt64}
    (hc : Γ[c]? = some (.sc tc z)) (htc : tc.isInt = true)
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.select c a b) = some (.sc .i64 (if Sem.isTrue (.sc tc z) then x else y)) := by
  simp only [evalOp, Sem.get, hc, ha, hb, Option.bind_eq_bind, Option.bind_some, htc]
  split
  · split <;> simp_all
  · rename_i h; exact absurd h (by simp only [V.ty]; decide)

theorem evalOp_icmp64 {m : Mem} {Γ : Env} {c : ICmpCond} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.icmp c a b) = some (.sc .i8 (if cmpInt c .i64 x y then 1 else 0)) := by
  simp only [evalOp, Sem.get, ha, hb, Option.bind_eq_bind, Option.bind_some, zipIntCmp, boolV]
  rfl

theorem evalOp_bandNot64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.bandNot a b) = some (.sc .i64 (x &&& ~~~y)) := by
  simp [evalOp, Sem.get, ha, hb, zipIntBits, norm64, ClifTy.isInt]

theorem evalOp_ishl64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64} {tb : ClifTy}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc tb y)) (htb : tb.isInt = true) :
    evalOp m Γ (.ishl a b) = some (.sc .i64 (x <<< (y % 64))) := by
  show shiftBin Γ a b _ = _
  simp only [shiftBin, Sem.get, ha, hb]
  have e2 : ClifTy.i64.isInt = true := rfl
  simp [htb, e2, norm64, ClifTy.width]

theorem evalOp_ushr64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64} {tb : ClifTy}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc tb y)) (htb : tb.isInt = true) :
    evalOp m Γ (.ushr a b) = some (.sc .i64 (x >>> (y % 64))) := by
  show shiftBin Γ a b _ = _
  simp only [shiftBin, Sem.get, ha, hb]
  have e2 : ClifTy.i64.isInt = true := rfl
  have hm : x &&& (18446744073709551615 : UInt64) = x := by
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_and]
    have := x.toNat_lt
    show x.toNat &&& (2^64 - 1) = x.toNat
    rw [Nat.and_two_pow_sub_one_eq_mod]
    omega
  simp [htb, e2, norm64, ClifTy.width, widthMask, hm]

theorem evalOp_band64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.band a b) = some (.sc .i64 (x &&& y)) := by
  simp [evalOp, Sem.get, ha, hb, zipIntBits, norm64, ClifTy.isInt]

theorem evalOp_bor64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.bor a b) = some (.sc .i64 (x ||| y)) := by
  simp [evalOp, Sem.get, ha, hb, zipIntBits, norm64, ClifTy.isInt]

theorem evalOp_bxor64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.bxor a b) = some (.sc .i64 (x ^^^ y)) := by
  simp [evalOp, Sem.get, ha, hb, zipIntBits, norm64, ClifTy.isInt]

/-- An integer operation that asks its operands be of one type: when one is a
    64-bit value, so is the other, or the operation is undefined. -/
theorem left_i64 {m : Mem} {Γ : Env} {a b : R} {t : ClifTy} {x y : UInt64} {v : V} (o : Op)
    (ho : o = .iadd a b ∨ o = .isub a b ∨ o = .imul a b ∨ o = .band a b ∨ o = .bor a b ∨ o = .bxor a b)
    (ha : Γ[a]? = some (.sc t x)) (hb : Γ[b]? = some (.sc .i64 y)) (h : evalOp m Γ o = some v) :
    t = .i64 := by
  rcases ho with rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [evalOp, bin, zipIntBits, Sem.get, ha, hb, Option.bind_eq_bind, Option.bind_some] at h <;>
    (split at h <;> (try cases h) <;> (cases t <;> first | rfl | (rename_i hc; exact absurd hc (by decide))))

/-- `left_i64`, the value of a type not known on the right. -/
theorem right_i64 {m : Mem} {Γ : Env} {a b : R} {t : ClifTy} {x y : UInt64} {v : V} (o : Op)
    (ho : o = .iadd a b ∨ o = .isub a b ∨ o = .imul a b ∨ o = .band a b ∨ o = .bor a b ∨ o = .bxor a b)
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc t y)) (h : evalOp m Γ o = some v) :
    t = .i64 := by
  rcases ho with rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [evalOp, bin, zipIntBits, Sem.get, ha, hb, Option.bind_eq_bind, Option.bind_some] at h <;>
    (split at h <;> (try cases h) <;> (cases t <;> first | rfl | (rename_i hc; exact absurd hc (by decide))))

theorem evalOp_load {m : Mem} {Γ : Env} {a : R} {op : LoadOp} {t : ClifTy} {x v : UInt64}
    (hk : op.kind = .plain) (hl0 : op.ty.lanes = none)
    (ha : Γ[a]? = some (.sc t x)) (hl : m.load x (tyBytes op.ty) = some v) :
    evalOp m Γ (.load op a) = some (.sc op.ty v) := by
  unfold evalOp
  simp only [Sem.get, ha]
  simp [hk, hl0, hl]

/-- The arguments a host hands an entry point: the data's and the output's
    bases and lengths as the caller gives them. -/
def runArgs (dataLen outLen : UInt64) : List V :=
  [.sc .i64 (regionBase .data), .sc .i64 dataLen, .sc .i64 (regionBase .out), .sc .i64 outLen]

-- ---------------------------------------------------------------------------
-- What a slot holds
-- ---------------------------------------------------------------------------

section vars
variable {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop} {Γ : Env} {w : World}

/-- The operations whose value, when they have one, is a scalar. -/
def _root_.AlgorithmLib.HProg.Op.scalarResult : Op → Bool
  | .iconst .. | .iadd .. | .isub .. | .imul .. | .udiv .. | .ishl .. | .ushr .. | .ineg .. | .ctz ..
  | .popcnt .. | .ireduce32 .. | .uextend64 .. | .sextend64 .. | .ibin .. | .ishift .. | .iun .. => true
  | .load op _ => op.ty.lanes.isNone
  | _ => false

theorem evalOp_scalar {m : Mem} {Γ : Env} {o : Op} {v : V} (hs : o.scalarResult = true)
    (h : evalOp m Γ o = some v) : ∃ t x, v = .sc t x := by
  cases o <;> simp only [Op.scalarResult] at hs <;> (try cases hs)
  all_goals simp only [evalOp, bin, shiftBin, un] at h
  all_goals decode_inv
  all_goals first
    | exact ⟨_, _, rfl⟩
    | (simp only [ibinOp, iunOp] at h; split at h <;> decode_inv <;> exact ⟨_, _, rfl⟩)
    | (simp only [ishiftOp]; split <;> exact ⟨_, _, rfl⟩)
    | (simp_all)

/-- A scalar load answers a value of the type it loads. -/
theorem evalOp_load_ty {m : Mem} {Γ : Env} {op : LoadOp} {a : R} {t : ClifTy} {x : UInt64}
    (hl : op.ty.lanes = none) (h : evalOp m Γ (.load op a) = some (.sc t x)) : t = op.ty := by
  simp only [evalOp] at h
  revert h
  generalize Sem.get Γ a = g
  rcases g with _ | ⟨_, addr⟩ | _ <;> simp only [Option.bind, bind, reduceCtorEq, false_imp_iff]
  simp only [hl, Option.isSome_none, Bool.and_false, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
  cases op.kind <;> simp only [hl] <;> split <;> simp [norm, ofInt] <;> intros <;> simp_all

/-- How many bytes a load reads: a vector's lanes for a vector. -/
def loadBytes (op : LoadOp) : Nat :=
  match op.kind with
  | .uload8 | .sload8 => 1
  | .uload16 | .sload16 => 2
  | .uload32 | .sload32 => 4
  | .plain => match op.ty.lanes with
    | some (l, n) => n * tyBytes l
    | none => tyBytes op.ty

/-- A scalar load from an address whose bytes are there answers. -/
theorem evalOp_load_some {m : Mem} {Γ : Env} {op : LoadOp} {a : R} {t : ClifTy} {x : UInt64}
    (hl : op.ty.lanes = none) (ha : Γ[a]? = some (.sc t x)) (h : ∃ b, m.load x (loadBytes op) = some b) :
    evalOp m Γ (.load op a) ≠ none := by
  obtain ⟨b, hb⟩ := h
  intro hn
  simp only [evalOp, Sem.get, ha] at hn
  unfold loadBytes at hb
  cases hk : op.kind <;> simp_all

/-- A byte read is below 256. -/
theorem load1_lt {m : Mem} {a b : UInt64} (h : m.load a 1 = some b) : b.toNat < 256 := by
  unfold Mem.load at h
  rcases hd : decodeAddr a with _ | ⟨r, off⟩ <;> simp only [hd, bind, Option.bind, reduceCtorEq] at h
  split at h
  · cases h
  · cases h
    simp only [List.range_one, List.foldr_cons, List.foldr_nil, UInt64.zero_shiftLeft, UInt64.zero_or,
      UInt8.toNat_toUInt64]
    exact UInt8.toNat_lt _

/-- So is a byte load's answer, zero-extended. -/
theorem evalOp_uload8_lt {m : Mem} {Γ : Env} {op : LoadOp} {a : R} {t : ClifTy} {x : UInt64}
    (hk : op.kind = .uload8) (h : evalOp m Γ (.load op a) = some (.sc t x)) : x.toNat < 256 := by
  simp only [evalOp] at h
  revert h
  generalize Sem.get Γ a = g
  rcases g with _ | ⟨_, addr⟩ | _ <;> simp only [Option.bind, bind, reduceCtorEq, false_imp_iff, hk]
  split
  · simp
  cases hl : m.load addr 1 with
  | none => simp
  | some b =>
    simp only [norm, Option.some.injEq, V.sc.injEq]
    rintro ⟨-, rfl⟩
    exact Nat.lt_of_le_of_lt (by rw [UInt64.toNat_and]; exact Nat.and_le_left) (load1_lt hl)

/-- What a library function's call answers, when it answers, is a scalar of the
    type its signature gives (`libAnswer`). -/
theorem callOf_lib_scalar {f : LibFn} {lc : Locals} {vs : List V} {w : World} {v : V} {w' : World}
    (h : callOf lc (.lib f) vs w = some (some v, w')) : ∃ x, v = .sc (f.result.getD .i64) x := by
  simp only [callOf] at h
  cases hl : lc f.libSlot (vs.map normV) w with
  | error _ => simp [hl, Except.toOption] at h
  | ok p =>
      obtain ⟨r, w1⟩ := p
      simp only [hl, Except.toOption, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨h1, -⟩ := h
      unfold libAnswer at h1
      split at h1
      · cases hr : f.result with
        | none => simp [hr] at h1
        | some t =>
            simp only [hr, Option.map_some, Option.some.injEq] at h1
            subst h1; exact ⟨_, rfl⟩
      · cases h1

end vars

-- ---------------------------------------------------------------------------
-- The engine's own libraries, and branches on what they answer
-- ---------------------------------------------------------------------------

section ext
variable {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop} {Γ : Env} {w : World}

open Contracts (Held extNeeds extAfter extKeeps roomAt) in
/-- An unsigned comparison that held: the first operand was below the second. -/
theorem icmp_ult_true {m : Mem} {Γ : Env} {a b : R} {x y : UInt64} {t : ClifTy} {f : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : evalOp m Γ (.icmp .ult a b) = some (.sc t f)) (hf : (f != 0) = true) : x < y := by
  simp only [evalOp, Sem.get, ha, hb, Option.bind_eq_bind, Option.bind_some, zipIntCmp] at h
  simp only [show (ClifTy.i64 == ClifTy.i64 && ClifTy.i64.isInt) = true from rfl, if_true, Option.some.injEq,
    boolV, cmpInt, V.sc.injEq] at h
  obtain ⟨-, rfl⟩ := h
  split at hf
  · simpa using ‹decide (x < y) = true›
  · simp at hf

/-- A comparison of two 64-bit values answers. -/
theorem icmp_i64_some {m : Mem} {Γ : Env} {a b : R} {x y : UInt64} (c : ICmpCond)
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.icmp c a b) ≠ none := by
  simp [evalOp, Sem.get, ha, hb, zipIntCmp]
  decide

/-- A value of type `t`: a scalar of it, or a vector of it with its lanes. -/
def TyV (t : ClifTy) (v : V) : Prop :=
  match t.lanes with
  | none => ∃ x, v = .sc t x
  | some (_, n) => ∃ xs, v = .vec t xs ∧ xs.size = n

theorem TyV.sc {t : ClifTy} {x : UInt64} (hl : t.lanes = none) : TyV t (.sc t x) := by
  unfold TyV; rw [hl]; exact ⟨x, rfl⟩

theorem TyV.vec {t lane : ClifTy} {n : Nat} {xs : Array UInt64} (hl : t.lanes = some (lane, n))
    (hn : xs.size = n) : TyV t (.vec t xs) := by
  unfold TyV; rw [hl]; exact ⟨xs, rfl, hn⟩

section VecTy

/-- Lanes loaded one after another: as many as asked, when every one is there. -/
theorem foldlM_lanes_size {m : Mem} {addr : UInt64} {w : Nat} : ∀ (l : List Nat) (acc r : Array UInt64),
    l.foldlM (fun acc i => (m.load (addr + UInt64.ofNat (i * w)) w).bind fun b => pure (acc.push b)) acc = some r →
    r.size = acc.size + l.length
  | [], acc, r, h => by simp only [List.foldlM_nil, Option.pure_def, Option.some.injEq] at h; subst h; simp
  | i :: l, acc, r, h => by
      simp only [List.foldlM_cons, Option.bind_eq_bind, Option.pure_def] at h
      cases hb : m.load (addr + UInt64.ofNat (i * w)) w with
      | none => rw [hb] at h; simp only [Option.bind_none, reduceCtorEq] at h
      | some b =>
          rw [hb] at h; simp only [Option.bind_some] at h
          have := foldlM_lanes_size l _ _ h
          simp only [Array.size_push, List.length_cons] at this ⊢; omega

theorem foldlM_lanes_some {m : Mem} {addr : UInt64} {w : Nat} (P : Nat → Prop)
    (hf : ∀ i, P i → ∃ b, m.load (addr + UInt64.ofNat (i * w)) w = some b) :
    ∀ (l : List Nat) (acc : Array UInt64), (∀ i ∈ l, P i) →
    ∃ r, l.foldlM (fun acc i => (m.load (addr + UInt64.ofNat (i * w)) w).bind fun b => pure (acc.push b)) acc = some r
  | [], acc, _ => ⟨acc, rfl⟩
  | i :: l, acc, hl => by
      obtain ⟨b, hb⟩ := hf i (hl i (by simp))
      simp only [List.foldlM_cons, Option.bind_eq_bind, Option.pure_def, hb, Option.bind_some]
      exact foldlM_lanes_some P hf l _ fun j hj => hl j (by simp [hj])

/-- A vector load answers a vector of its type, and answers where its bytes fit. -/
theorem evalOp_vload {m : Mem} {Γ : Env} {op : LoadOp} {a : R} {lane : ClifTy} {n : Nat} {t' : ClifTy}
    {x : UInt64} (hal : op.aligned = false) (hk : op.kind = .plain) (hl : op.ty.lanes = some (lane, n)) (ha : Γ[a]? = some (.sc t' x)) :
    (∀ v, evalOp m Γ (.load op a) = some v → TyV op.ty v) ∧
    (Fits m x (n * tyBytes lane) → evalOp m Γ (.load op a) ≠ none) := by
  have hw : 0 < tyBytes lane := by
    cases h : op.ty <;> rw [h] at hl <;> simp [ClifTy.lanes] at hl <;> obtain ⟨rfl, rfl⟩ := hl <;> decide
  simp only [evalOp, Sem.get, ha, hal, hk, hl, Option.bind_eq_bind, Option.bind_some, Bool.false_and,
    Bool.false_eq_true, if_false]
  refine ⟨fun v hv => ?_, fun hf => ?_⟩
  · obtain ⟨r, hr, hv⟩ := Option.bind_eq_some_iff.mp hv
    cases hv
    exact TyV.vec hl (by simpa using foldlM_lanes_size _ _ _ hr)
  · obtain ⟨r, hr⟩ := foldlM_lanes_some (m := m) (addr := x) (w := tyBytes lane) (fun i => i < n)
      (fun i hi => (hf.part (i := i * tyBytes lane) (k := tyBytes lane)
        (by have := Nat.mul_le_mul_right (tyBytes lane) (Nat.succ_le_of_lt hi); rw [Nat.succ_mul] at this; omega)
        hw).load)
      (List.range n) #[] (fun i hi => by simpa using hi)
    rw [hr]; simp

/-- A scalar answer, typed: of its constructor. -/
theorem tyV_sc_ex {e : Option V} {T : ClifTy} (hl : T.lanes = none) (h : ∃ v, e = some v ∧ TyV T v) :
    ∃ x, e = some (.sc T x) := by
  obtain ⟨v, hv, ht⟩ := h
  unfold TyV at ht; rw [hl] at ht
  obtain ⟨x, rfl⟩ := ht
  exact ⟨x, hv⟩

end VecTy

section LoopN

/-- A comparison that answers a scalar compared two integers of one type, and
    answers whether they compare. -/
theorem icmp_sc_inv {m : Mem} {Γ : Env} {cc : ICmpCond} {a b : R} {t : ClifTy} {f : UInt64}
    (h : evalOp m Γ (.icmp cc a b) = some (.sc t f)) :
    ∃ ty x y, Γ[a]? = some (.sc ty x) ∧ Γ[b]? = some (.sc ty y) ∧ (f != 0) = cmpInt cc ty x y := by
  simp only [evalOp, Sem.get, Option.bind_eq_bind] at h
  cases ha : Γ[a]? with
  | none => rw [ha] at h; simp at h
  | some va =>
  cases hb : Γ[b]? with
  | none => rw [ha, hb] at h; simp at h
  | some vb =>
  rw [ha, hb] at h
  simp only [Option.bind_some] at h
  cases va with
  | vec _ _ => cases vb <;> simp [zipIntCmp] at h <;> split at h <;> simp_all
  | sc ta x =>
  cases vb with
  | vec _ _ => simp [zipIntCmp] at h
  | sc tb y =>
  simp only [zipIntCmp] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨rfl, -⟩ := hc
    simp only [Option.some.injEq, boolV, V.sc.injEq] at h
    obtain ⟨-, rfl⟩ := h
    exact ⟨ta, x, y, rfl, rfl, by cases cmpInt cc ta x y <;> rfl⟩
  · cases h

theorem mapM_get {Γ : Env} : ∀ {rs : List Nat} {vs : List V}, rs.mapM (fun r => Γ[r]?) = some vs →
    ∀ i : Nat, vs[i]? = (rs[i]?).bind (fun r => Γ[r]?)
  | [], vs, h, i => by simp at h; subst h; simp
  | r :: rs, vs, h, i => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨v, hv, vs', hvs', rfl⟩ := h
      cases i with
      | zero => simp [hv]
      | succ i => simpa using mapM_get hvs' i

end LoopN

end ext

/-- A length the caller handed over is at most a region's span. -/
theorem roomArg_toNat_le {S : Contracts.TState} {w : World} {r : Region} {x : UInt64} (hS : S.holds w)
    (hm : Contracts.Fact.roomArg r x ∈ S) : x.toNat ≤ 2 ^ 36 := (hS _ hm).2.1

/-- A 64-bit value below `2^63` is itself, signed. -/
theorem signed_i64_of_lt {x : UInt64} (h : x.toNat < 2 ^ 63) : signed .i64 x = x.toNat := by
  have hm : x &&& widthMask .i64 = x := by
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_and]
    show x.toNat &&& (2 ^ 64 - 1) = x.toNat
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt (UInt64.toNat_lt x)]
  simp only [signed, ClifTy.width, hm]
  rw [if_pos (by decide), if_neg]
  intro hge
  have : (0x8000000000000000 : UInt64).toNat ≤ x.toNat := UInt64.le_iff_toNat_le.mp hge
  simp at this; omega

/-- Bytes at an address past a region's base, within a length the program was
    handed for it. -/
theorem fits_of_roomArg_sub {S : Contracts.TState} {w : World} {a X : UInt64} {n : Nat} {r : Region}
    (hS : S.holds w) (hm : Contracts.Fact.roomArg r X ∈ S)
    (hp : r ≠ .pinned) (hle : (regionBase r).toNat ≤ a.toNat)
    (hb : a.toNat - (regionBase r).toNat + n ≤ X.toNat) (hn : 0 < n) : Fits w.mem a n := by
  have hx := hS _ hm
  have hk : a.toNat - (regionBase r).toNat < regionSpan.toNat := by
    simp only [regionSpan, UInt64.reduceToNat]; have := hx.2.1; omega
  have hd := Sem.decodeAddr_base r _ hk
  have hsub : (a - regionBase r).toNat = a.toNat - (regionBase r).toNat :=
    UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr hle)
  rw [← hsub, UInt64.ofNat_toNat, UInt64.add_comm, UInt64.sub_add_cancel] at hd
  rw [hsub] at hd
  exact ⟨r, _, hd, hp, by have := hx.1; omega, by have := hx.2.1; omega, Contracts.roomArg_lentOk hx⟩

/-- Bytes at an address past a region's base, within a room the typestate
    names. -/
theorem fits_of_room_sub {S : Contracts.TState} {w : World} {a : UInt64} {n N : Nat} {r : Region}
    (hS : S.holds w) (hm : Contracts.Fact.room r N ∈ S)
    (hp : r ≠ .pinned) (hle : (regionBase r).toNat ≤ a.toNat)
    (hb : a.toNat - (regionBase r).toNat + n ≤ N) (hN : N ≤ 2 ^ 36) (hn : 0 < n) : Fits w.mem a n := by
  have hx : N ≤ w.mem.sizes r := (hS _ hm).1
  have hk : a.toNat - (regionBase r).toNat < regionSpan.toNat := by
    simp only [regionSpan, UInt64.reduceToNat]; omega
  have hd := Sem.decodeAddr_base r _ hk
  have hsub : (a - regionBase r).toNat = a.toNat - (regionBase r).toNat :=
    UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr hle)
  rw [← hsub, UInt64.ofNat_toNat, UInt64.add_comm, UInt64.sub_add_cancel] at hd
  rw [hsub] at hd
  exact ⟨r, _, hd, hp, by omega, by omega, Contracts.room_lentOk (hS _ hm)⟩

/-- A mask clearing the low `k` bits rounds down to a multiple of `2^k`. -/
theorem toNat_and_mask {x : UInt64} {k : Nat} (hk : k < 64) :
    (x &&& UInt64.ofNat (2 ^ 64 - 2 ^ k)).toNat = x.toNat / 2 ^ k * 2 ^ k := by
  rw [UInt64.toNat_and, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by have := Nat.one_le_two_pow (n := k); omega)]
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_and, ← Nat.shiftLeft_eq, ← Nat.shiftRight_eq_div_pow, Nat.testBit_shiftLeft,
    Nat.testBit_shiftRight]
  have hx := UInt64.toNat_lt x
  by_cases hi : k ≤ i
  · have hsub : (2 ^ 64 - 2 ^ k).testBit i = decide (i < 64) := by
      rw [show 2 ^ 64 - 2 ^ k = 2 ^ k * (2 ^ (64 - k) - 1) by
        rw [Nat.mul_sub, Nat.mul_one, ← Nat.pow_add, Nat.add_sub_cancel' (by omega)]]
      rw [Nat.mul_comm, ← Nat.shiftLeft_eq, Nat.testBit_shiftLeft, Nat.testBit_two_pow_sub_one]
      by_cases h64 : i < 64 <;> simp [hi, h64] <;> omega
    rw [hsub, Nat.add_sub_cancel' hi]
    by_cases h64 : i < 64
    · simp [hi, h64]
    · simp only [h64, decide_false, Bool.and_false, hi, decide_true, Bool.true_and]
      exact (Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hx (Nat.pow_le_pow_right (by decide) (by omega)))).symm
  · have : (2 ^ 64 - 2 ^ k).testBit i = false := by
      rw [show 2 ^ 64 - 2 ^ k = 2 ^ k * (2 ^ (64 - k) - 1) by
        rw [Nat.mul_sub, Nat.mul_one, ← Nat.pow_add, Nat.add_sub_cancel' (by omega)]]
      rw [Nat.mul_comm, ← Nat.shiftLeft_eq, Nat.testBit_shiftLeft]
      simp [hi]
    simp [this, hi]

/-- Lanes that each fit store. -/
theorem foldlM_lane_stores_some {addr : UInt64} {lw : Nat} : ∀ (l : List (UInt64 × Nat)) (m0 : Mem),
    (∀ p ∈ l, Fits m0 (addr + UInt64.ofNat (p.2 * lw)) lw) →
    ∃ m, l.foldlM (fun mm (p : UInt64 × Nat) => mm.store (addr + UInt64.ofNat (p.2 * lw)) lw p.1) m0 = some m
  | [], m0, _ => ⟨m0, rfl⟩
  | p :: l, m0, h => by
      obtain ⟨m1, h1⟩ := (h p (by simp)).store p.1
      simp only [List.foldlM_cons, h1, Option.bind_eq_bind, Option.bind_some]
      exact foldlM_lane_stores_some l m1 fun q hq =>
        (h q (by simp [hq])).congr (Contracts.store_sizes_eq h1) (Mem.store_lent h1)

/-- A mask that clears the low `k` bits, given by its value. -/
theorem toNat_and_lowMask {x m : UInt64} (k : Nat) (hk : k < 64) (hm : m.toNat = 2 ^ 64 - 2 ^ k) :
    (x &&& m).toNat = x.toNat / 2 ^ k * 2 ^ k := by
  have : m = UInt64.ofNat (2 ^ 64 - 2 ^ k) := by
    apply UInt64.toNat_inj.mp
    rw [hm, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by have := Nat.one_le_two_pow (n := k); omega)]
  subst this; exact toNat_and_mask hk

theorem toNat_and_le_right (a b : UInt64) : (a &&& b).toNat ≤ b.toNat := by
  rw [UInt64.toNat_and]; exact Nat.and_le_right

/-- Room at an address a value past a known one: the region's room covers
    the offset and the length. -/
theorem roomAt_add {S : Contracts.TState} {b x : UInt64} {n K : Nat} {r : Region} {off : Nat}
    (hm : Contracts.Fact.room r K ∈ S)
    (hd : decodeAddr b = some (r, off)) (hp : r ≠ .pinned) (hb : off + x.toNat + n ≤ K)
    (hs : off + x.toNat + n < 2 ^ 36) : Contracts.roomAt S (b + x) n = true := by
  have hd' : decodeAddr (b + x) = some (r, off + x.toNat) := by
    have := Sem.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
    rwa [UInt64.ofNat_toNat] at this
  unfold Contracts.roomAt
  rw [hd']
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.any_eq_true]
  refine ⟨⟨hp, by omega⟩, _, hm, ?_⟩
  simp only [Bool.and_eq_true, decide_eq_true_eq, true_and]
  omega

/-- **Room for page-locked memory**, from a bound on what it spans. -/
theorem pinnedRoom_of_state {S : Contracts.TState} {w : World} {n : Nat} {size : UInt64} (hS : S.holds w)
    (hm : Contracts.Fact.pinnedUsed n ∈ S) (hb : n + 63 + size.toNat ≤ regionSpan.toNat) :
    Contracts.PinnedRoom w size := by
  have hn : w.mem.sizes .pinned ≤ n := hS _ hm
  unfold Contracts.PinnedRoom
  have := Nat.mod_lt (64 - w.mem.pinned.size % 64) (by decide : 64 > 0)
  change w.mem.pinned.size ≤ n at hn
  omega

/-- `readable_upto_of_state`, for a write. -/
theorem writable_upto_of_state {S : Contracts.TState} {w : World} {a : UInt64} {n K : Nat} (hS : S.holds w)
    (h : Contracts.roomAt S a K = true) (hn : n ≤ K) : Writable w.mem a n :=
  Contracts.writable_of_state hS (Contracts.roomAt_mono h hn)

end AlgorithmLib.Prog
