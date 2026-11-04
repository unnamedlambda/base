module
public import AlgorithmLib.Host.MemLaws
meta import AlgorithmLib.Host.MemLaws
public import AlgorithmLib.Host.Driver
meta import AlgorithmLib.Host.Driver
@[expose] public section

/-!
# The heap, as a program's statements read it

What a statement says of memory the C library's heap holds: a live block
(`Blk`, `Whole`), a field of one (`fld`), bytes that can be read (`Bytes`,
`leN`), a world a block's bytes can be read and written in (`Calm`), runs of
pinned memory apart from each other (`Before`, `Apart`), and a world that is
another with one run of bytes rewritten (`Wrote`), as a store leaves it
(`Wrote.ofStore`). What else holds of them is `Host.ExtLife`'s and
`Proof.LibSpec`'s.
-/

namespace AlgorithmLib.HProg.Contracts

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

/-- **A live heap allocation.** `x` is the start of an allocation of at least
    `n` bytes the C library's heap holds: inside pinned memory, and among its
    live allocations. -/
def Blk (w : World) (x : UInt64) (n : Nat) : Prop :=
  ∃ off m, w.heapAt? x = some (off, m) ∧ n ≤ m ∧ 0 < m ∧ off + m ≤ w.mem.pinned.size ∧ off + m ≤ 2 ^ 36 ∧
    (off, m) ∈ w.mem.pinnedLive

-- ---------------------------------------------------------------------------
-- Where a block is, and what reads, writes and allocation do to blocks
-- ---------------------------------------------------------------------------

theorem inSpan_iff (a b : UInt64) :
    (decide (a ≥ b) && decide (a - b < regionSpan)) = true ↔ b.toNat ≤ a.toNat ∧ a.toNat - b.toNat < 2 ^ 36 := by
  simp only [Bool.and_eq_true, decide_eq_true_eq, ge_iff_le, UInt64.le_iff_toNat_le]
  constructor
  · rintro ⟨h1, h2⟩
    rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr h1)] at h2
    exact ⟨h1, h2⟩
  · rintro ⟨h1, h2⟩
    refine ⟨h1, ?_⟩
    rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr h1)]
    exact h2
theorem decodeAddr_pinned {off : Nat} (h : off < 2 ^ 36) :
    decodeAddr (addrOf .pinned off) = some (.pinned, off) := by
  have hx : (addrOf .pinned off).toNat = 0x4000000000 + off := by
    simp only [addrOf, regionBase, UInt64.toNat_add, UInt64.toNat_ofNat', UInt64.reduceToNat]; omega
  unfold decodeAddr
  simp only [List.findSome?]
  rw [if_neg (by rw [inSpan_iff, hx]; simp [regionBase]; omega),
      if_neg (by rw [inSpan_iff, hx]; simp [regionBase]; omega),
      if_neg (by rw [inSpan_iff, hx]; simp [regionBase]; omega),
      if_pos (by rw [inSpan_iff, hx]; simp [regionBase]; omega)]
  have : (addrOf .pinned off - regionBase .pinned).toNat = off := by
    rw [UInt64.toNat_sub_of_le _ _ (by rw [UInt64.le_iff_toNat_le, hx]; simp [regionBase])]
    simp [hx, regionBase]
  simpa using this

/-- Where a heap allocation's start decodes to. -/
theorem heapAt_decode {w : World} {x : UInt64} {off m : Nat} (h : w.heapAt? x = some (off, m)) :
    decodeAddr x = some (.pinned, off) ∧ w.heap.find? (·.1 == off) = some (off, m) := by
  unfold World.heapAt? at h
  split at h
  · rename_i o hd
    have := List.find?_some h
    simp only [beq_iff_eq] at this
    subst this
    exact ⟨hd, h⟩
  · cases h

theorem Blk.mono {w : World} {x : UInt64} {n k : Nat} (h : Blk w x n) (hk : k ≤ n) : Blk w x k := by
  obtain ⟨off, m, hat, hn, rest⟩ := h
  exact ⟨off, m, hat, Nat.le_trans hk hn, rest⟩

/-- A step that leaves the heap's allocations as they were, pinned memory no
    smaller and its live allocations as they were keeps a block. -/
theorem Blk.same {w w' : World} {x : UInt64} {n : Nat} (h : Blk w x n) (hh : w'.heap = w.heap)
    (hs : w.mem.pinned.size ≤ w'.mem.pinned.size) (hl : w'.mem.pinnedLive = w.mem.pinnedLive) : Blk w' x n := by
  obtain ⟨off, m, hat, hn, hm, hsz, h36, hlv⟩ := h
  refine ⟨off, m, ?_, hn, hm, Nat.le_trans hsz hs, h36, by rw [hl]; exact hlv⟩
  unfold World.heapAt? at hat ⊢
  rw [hh]; exact hat

/-- An allocation leaves every block there was. -/
theorem Blk.alloc {w : World} {x : UInt64} {n : Nat} (h : Blk w x n) (bytes : Nat) :
    Blk { w with
      mem := { w.mem with
        pinned := w.mem.pinned ++ ByteArray.mk (Array.replicate (w.mem.pinnedNext - w.mem.pinned.size + bytes) 0)
        pinnedLive := (w.mem.pinnedNext, bytes) :: w.mem.pinnedLive }
      heap := (w.mem.pinnedNext, bytes) :: w.heap } x n := by
  obtain ⟨off, m, hat, hn, hm, hsz, h36, hlv⟩ := h
  obtain ⟨hd, hf⟩ := heapAt_decode hat
  have hne : (w.mem.pinnedNext == off) = false := by
    have : w.mem.pinned.size ≤ w.mem.pinnedNext := by unfold Mem.pinnedNext; omega
    simp only [beq_eq_false_iff_ne, ne_eq]; omega
  refine ⟨off, m, ?_, hn, hm, ?_, h36, List.mem_cons_of_mem _ hlv⟩
  · unfold World.heapAt?
    rw [hd]
    show ((w.mem.pinnedNext, bytes) :: w.heap).find? (·.1 == off) = some (off, m)
    rw [List.find?_cons, hne]; exact hf
  · show off + m ≤ (w.mem.pinned ++ ByteArray.mk _).size
    rw [ByteArray.size_append]; omega

/-- **What `calloc` answers**: null, or a block of the bytes it was asked for. -/
theorem calloc_blk {n sz x : UInt64} {w w' : World}
    (h : cCall .calloc [n, sz] w = some (some (.sc .i64 x), w')) (hx : x ≠ 0) :
    Blk w' x (n.toNat * sz.toNat) := by
  simp only [cCall] at h
  split at h
  · cases h
  split at h
  · simp only [Option.some.injEq, Prod.mk.injEq, V.sc.injEq, true_and] at h
    exact absurd h.1.symm hx
  · rename_i hz hroom
    simp only [Option.some.injEq, Prod.mk.injEq, V.sc.injEq, true_and] at h
    obtain ⟨rfl, rfl⟩ := h
    have hpos : 0 < n.toNat * sz.toNat := by
      rcases Nat.eq_zero_or_pos (n.toNat * sz.toNat) with e | e
      · simp [e] at hz
      · exact e
    have hroom' : w.mem.pinnedNext + n.toNat * sz.toNat ≤ 2 ^ 36 := by
      simp only [regionSpan, UInt64.reduceToNat] at hroom; omega
    have hge : w.mem.pinned.size ≤ w.mem.pinnedNext := by unfold Mem.pinnedNext; omega
    refine ⟨w.mem.pinnedNext, n.toNat * sz.toNat, ?_, Nat.le_refl _, hpos, ?_, hroom', List.mem_cons_self⟩
    · unfold World.heapAt?
      rw [decodeAddr_pinned (by omega)]
      show ((w.mem.pinnedNext, n.toNat * sz.toNat) :: w.heap).find? (·.1 == w.mem.pinnedNext) = some _
      rw [List.find?_cons]; simp
    · show _ ≤ (w.mem.pinned ++ ByteArray.mk _).size
      rw [ByteArray.size_append]
      show _ ≤ w.mem.pinned.size + (Array.replicate _ (0 : UInt8)).size
      rw [Array.size_replicate]; omega

/-- **What a call of `calloc` answers**, by the bits it is handed. -/
theorem calloc_answer {vs : List V} {n sz x : UInt64} {w w' : World} (hb : vs.mapM asBits = some [n, sz])
    (h : extCall (.c .calloc) vs w = some (some (.sc .i64 x), w')) (hx : x ≠ 0) :
    Blk w' x (n.toNat * sz.toNat) := by
  simp only [extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true, Bool.false_eq_true,
    if_false, hb, Option.bind_eq_bind, Option.bind_some] at h
  exact calloc_blk h hx

/-- Bytes inside a live block, with no copy in flight and no worker running,
    can be read. -/
theorem Blk.load {w : World} {x : UInt64} {n : Nat} (h : Blk w x n) {j k : Nat} (hjk : j + k ≤ n) (hk : 0 < k)
    (hb : w.mem.busy = []) : (w.mem.load (x + UInt64.ofNat j) k).isSome = true := by
  obtain ⟨off, m, hat, hn, hm, hsz, h36, hlv⟩ := h
  obtain ⟨hd, -⟩ := heapAt_decode hat
  have hd' := Sem.decodeAddr_add (i := j) hd (by rw [Sem.regionSpan_toNat]; omega)
  unfold Mem.load
  rw [hd']
  simp only [Option.bind_eq_bind, Option.bind_some, Mem.region, Mem.readable, hb]
  have hany : (w.mem.pinnedLive.any fun x => decide (x.1 ≤ off + j) && decide (off + j + k ≤ x.1 + x.2)) = true :=
    List.any_eq_true.mpr ⟨(off, m), hlv, by simp; omega⟩
  rw [if_neg (by simp [hany]; omega)]
  rfl

/-- …and written. -/
theorem Blk.store {w : World} {x : UInt64} {n : Nat} (h : Blk w x n) {j k : Nat} (hjk : j + k ≤ n) (hk : 0 < k)
    (hb : w.mem.busy = []) (v : UInt64) :
    (w.mem.store (x + UInt64.ofNat j) k v).isSome = true := by
  obtain ⟨off, m, hat, hn, hm, hsz, h36, hlv⟩ := h
  obtain ⟨hd, -⟩ := heapAt_decode hat
  have hd' := Sem.decodeAddr_add (i := j) hd (by rw [Sem.regionSpan_toNat]; omega)
  unfold Mem.store
  rw [hd']
  simp only [Option.bind_eq_bind, Option.bind_some, Mem.region, Mem.reachable, hb]
  have hany : (w.mem.pinnedLive.any fun x => decide (x.1 ≤ off + j) && decide (off + j + k ≤ x.1 + x.2)) = true :=
    List.any_eq_true.mpr ⟨(off, m), hlv, by simp; omega⟩
  rw [if_neg (by simp [hany]; omega)]
  rfl

theorem store_live {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m') :
    m'.pinnedLive = m.pinnedLive := by
  unfold Mem.store at h
  obtain ⟨⟨r, off⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  · cases h; unfold Mem.setRegion; split <;> rfl

theorem foldl_stores_live {α : Type} (f : α → Mem → Option Mem)
    (hf : ∀ x m m', f x m = some m' → m'.pinnedLive = m.pinnedLive) :
    ∀ (l : List α) (m m' : Mem), l.foldlM (fun mm x => f x mm) m = some m' → m'.pinnedLive = m.pinnedLive
  | [], m, m', h => by cases h; rfl
  | x :: l, m, m', h => by
      simp only [List.foldlM_cons, Option.bind_eq_bind] at h
      obtain ⟨m1, h1, h⟩ := Option.bind_eq_some_iff.mp h
      exact (foldl_stores_live f hf l m1 m' h).trans (hf x m m1 h1)

/-- A program's own stores keep every block. -/
theorem Blk.stored {w : World} {m : Mem} {x : UInt64} {n : Nat} (h : Blk w x n) {a c : UInt64} {k : Nat}
    (hsz : m.sizes = w.mem.sizes) (hl : m.pinnedLive = w.mem.pinnedLive) :
    Blk { obsStore w a k c with mem := m } x n :=
  h.same rfl (Nat.le_of_eq (congrFun hsz .pinned).symm) hl

/-- The value `n` bytes of region bytes `bs` from `off` make, little-endian. -/
def bytesAt (bs : ByteArray) (off n : Nat) : UInt64 :=
  (List.range n).foldr (fun i acc => (acc <<< 8) ||| (bs.get! (off + i)).toUInt64) 0

theorem load_some {m : Mem} {a : UInt64} {n : Nat} {v : UInt64} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (h : m.load a n = some v) :
    off + n ≤ (m.region r).size ∧ m.readable r off n = true ∧ v = bytesAt (m.region r) off n := by
  unfold Mem.load at h
  simp only [hd, Option.bind_eq_bind, Option.bind_some] at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_eq_eq_not, Bool.not_true, not_or,
      Bool.not_eq_false] at hc
    injection h with h
    exact ⟨by omega, by simpa using hc.2, h.symm⟩

theorem load_of {m : Mem} {a : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hs : off + n ≤ (m.region r).size)
    (hr : m.readable r off n = true) : m.load a n = some (bytesAt (m.region r) off n) := by
  unfold Mem.load
  simp only [hd, Option.bind_eq_bind, Option.bind_some]
  rw [if_neg (by simp [hr]; omega)]
  rfl

theorem readable_off_pinned {m : Mem} {r : Region} (hr : r ≠ .pinned) (off n : Nat) :
    m.readable r off n = true := by
  cases r <;> simp_all (config := { decide := true }) [Mem.readable]

theorem foldr_bytes_congr {bs bs' : ByteArray} {off : Nat} : ∀ (l : List Nat) (init : UInt64),
    (∀ i ∈ l, bs'.get! (off + i) = bs.get! (off + i)) →
    l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| (bs'.get! (off + i)).toUInt64) init =
      l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| (bs.get! (off + i)).toUInt64) init
  | [], _, _ => rfl
  | i :: l, init, h => by
      simp only [List.foldr_cons, foldr_bytes_congr l init (fun j hj => h j (List.mem_cons_of_mem _ hj)),
        h i List.mem_cons_self]

/-- The byte a store of `v` puts at `off + j`. -/
def byteOfV (v : UInt64) (j : Nat) : UInt8 := ((v >>> (8 * UInt64.ofNat j)) &&& 0xff).toUInt8

theorem foldr_fun_congr (f g : Nat → UInt64) : ∀ (l : List Nat) (init : UInt64), (∀ i ∈ l, f i = g i) →
    l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| f i) init =
      l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| g i) init
  | [], _, _ => rfl
  | i :: l, init, h => by
      simp only [List.foldr_cons, foldr_fun_congr f g l init (fun j hj => h j (List.mem_cons_of_mem _ hj)),
        h i List.mem_cons_self]

theorem shl8_or (acc : UInt64) (b : UInt8) :
    ((acc <<< 8) ||| b.toUInt64).toNat = acc.toNat % 2 ^ 56 * 256 + b.toNat := by
  rw [UInt64.toNat_or, UInt64.toNat_shiftLeft]
  have hb : b.toUInt64.toNat = b.toNat := by simp
  rw [hb]
  have e : acc.toNat <<< ((8 : UInt64).toNat % 64) % 2 ^ 64 = (acc.toNat % 2 ^ 56) <<< 8 := by
    simp only [Nat.shiftLeft_eq]
    have : (8 : UInt64).toNat % 64 = 8 := by decide
    rw [this]; omega
  rw [e, ← Nat.shiftLeft_add_eq_or_of_lt (UInt8.toNat_lt b), Nat.shiftLeft_eq]

theorem foldl_set_size (off : Nat) (x : Nat → UInt8) : ∀ (l : List Nat) (bs : ByteArray),
    (l.foldl (fun (b : ByteArray) i => b.set! (off + i) (x i)) bs).size = bs.size
  | [], _ => rfl
  | i :: l, bs => by
      simp only [List.foldl_cons]
      rw [foldl_set_size off x l, Sem.set!_size]

theorem store_sizes {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m')
    (r : Region) : (m'.region r).size = (m.region r).size := by
  unfold Mem.store at h
  obtain ⟨⟨r0, off⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  · cases h
    by_cases e : r0 = r
    · subst e
      rw [Sem.region_setRegion_self]
      exact foldl_set_size off (fun i => ((v >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8) _ _
    · rw [Mem.region_setRegion_ne _ _ _ _ e]

theorem store_sizes_eq {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m') :
    m'.sizes = m.sizes := funext (store_sizes h)

-- ---------------------------------------------------------------------------
-- Bytes in memory: loads, strings and eight-byte stores
-- ---------------------------------------------------------------------------
theorem bytesAt_congr {bs bs' : ByteArray} {off n : Nat}
    (h : ∀ i, i < n → bs'.get! (off + i) = bs.get! (off + i)) : bytesAt bs' off n = bytesAt bs off n :=
  foldr_bytes_congr _ _ (fun i hi => h i (List.mem_range.mp hi))

/-- **A load survives a step that changes none of its bytes** and keeps its
    region's size, outside pinned memory. -/
theorem load_kept {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned)
    (hbytes : ∀ i, i < n → decodeAddr (a + UInt64.ofNat i) = some (r, off + i))
    (h : m.load a n = some v) (hsz : m'.sizes r = m.sizes r)
    (hu : ∀ i, i < n → Unchanged m m' (a + UInt64.ofNat i)) : m'.load a n = some v := by
  obtain ⟨hs, hr, rfl⟩ := load_some hd h
  have hs' : off + n ≤ (m'.region r).size := by
    have : (m'.region r).size = (m.region r).size := hsz
    omega
  rw [load_of hd hs' (readable_off_pinned hp _ _)]
  congr 1
  apply bytesAt_congr
  intro i hi
  have h1 := load_of (m := m) (n := 1) (hbytes i hi) (by omega) (readable_off_pinned hp _ _)
  have h1' := load_of (m := m') (n := 1) (hbytes i hi) (by omega) (readable_off_pinned hp _ _)
  have e := hu i hi _ _ h1 h1'
  simp [bytesAt] at e
  exact (UInt8.toUInt64_inj.mp e).symm

/-- Bytes a step leaves unchanged, and where it keeps the region's size, read
    the same. -/
theorem bytes_kept {m m' : Mem} {a : UInt64} {k : Nat} {r : Region} {off : Nat}
    (hp : r ≠ .pinned) (hbytes : ∀ i, i < k → decodeAddr (a + UInt64.ofNat i) = some (r, off + i))
    (hs : off + k ≤ (m.region r).size)
    (hsz : m'.sizes r = m.sizes r) (hu : ∀ i, i < k → Unchanged m m' (a + UInt64.ofNat i)) :
    ∀ i, i < k → (m'.region r).get! (off + i) = (m.region r).get! (off + i) := by
  intro i hi
  have hs' : (m'.region r).size = (m.region r).size := hsz
  have h1 := load_of (m := m) (n := 1) (hbytes i hi) (by omega) (readable_off_pinned hp _ _)
  have h1' := load_of (m := m') (n := 1) (hbytes i hi) (by omega) (readable_off_pinned hp _ _)
  have e := hu i hi _ _ h1 h1'
  simp [bytesAt] at e
  exact (UInt8.toUInt64_inj.mp e).symm

/-- **A C string reads the same** where its bytes, terminator included, and its
    region's size are the same. -/
theorem strSpan_congr {m m' : Mem} {a : UInt64} {r : Region} {off k : Nat}
    (hsp : Sem.strSpan m a = some (r, off, k)) (hs : (m'.region r).size = (m.region r).size)
    (hb : ∀ i, i < k → (m'.region r).get! (off + i) = (m.region r).get! (off + i)) :
    Sem.strSpan m' a = some (r, off, k) ∧ readCStr m' a = readCStr m a := by
  unfold Sem.strSpan at hsp ⊢
  unfold readCStr
  cases hd : decodeAddr a with
  | none => rw [hd] at hsp; cases hsp
  | some p =>
    obtain ⟨r0, off0⟩ := p
    simp only [hd, bind, Option.bind] at hsp ⊢
    cases hf : (List.range ((m.region r0).size - off0)).find? (fun i => (m.region r0).get! (off0 + i) == 0) with
    | none => rw [hf] at hsp; cases hsp
    | some n =>
      rw [hf] at hsp
      simp only [Option.some.injEq, Prod.mk.injEq] at hsp
      obtain ⟨rfl, rfl, rfl⟩ := hsp
      have hf' : (List.range ((m'.region r0).size - off0)).find? (fun i => (m'.region r0).get! (off0 + i) == 0)
          = some n := by
        rw [hs]
        rw [List.find?_range_eq_some] at hf ⊢
        obtain ⟨h1, h2, h3⟩ := hf
        refine ⟨by rw [hb n (by omega)]; exact h1, h2, fun j hj => ?_⟩
        rw [hb j (by omega)]; exact h3 j hj
      rw [hf']
      refine ⟨rfl, ?_⟩
      simp only
      congr 2
      apply List.map_congr_left
      intro i hi
      exact hb i (by simp at hi; omega)

theorem strSpan_bound {m : Mem} {a : UInt64} {r : Region} {off k : Nat}
    (h : Sem.strSpan m a = some (r, off, k)) : off + k ≤ (m.region r).size := by
  unfold Sem.strSpan at h
  cases hd : decodeAddr a with
  | none => rw [hd] at h; cases h
  | some p =>
    obtain ⟨r0, off0⟩ := p
    simp only [hd, bind, Option.bind] at h
    cases hf : (List.range ((m.region r0).size - off0)).find? (fun i => (m.region r0).get! (off0 + i) == 0) with
    | none => rw [hf] at h; cases h
    | some n =>
      rw [hf] at h
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      have h2 := (List.find?_range_eq_some.mp hf).2.1
      simp only [List.mem_range] at h2
      omega

theorem strSpan_decode {m : Mem} {a : UInt64} {r : Region} {off k : Nat}
    (h : Sem.strSpan m a = some (r, off, k)) : decodeAddr a = some (r, off) := by
  unfold Sem.strSpan at h
  cases hd : decodeAddr a with
  | none => rw [hd] at h; cases h
  | some p =>
    obtain ⟨r0, off0⟩ := p
    simp only [hd, bind, Option.bind] at h
    split at h
    · cases h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, -⟩ := h
      rfl

theorem foldl_set!_at (off : Nat) (v : UInt64) : ∀ (n : Nat) (b : ByteArray) (j : Nat), j < n →
    off + n ≤ b.size →
    ((List.range n).foldl (fun (b : ByteArray) i => b.set! (off + i) (byteOfV v i)) b).get! (off + j)
      = byteOfV v j
  | 0, _, _, hj, _ => absurd hj (Nat.not_lt_zero _)
  | n + 1, b, j, hj, hs => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      have hsz : ((List.range n).foldl (fun (b : ByteArray) i => b.set! (off + i) (byteOfV v i)) b).size
          = b.size := (foldl_set!_other off (off + n) v (List.range n) b (by
        intro i hi; have := List.mem_range.mp hi; omega)).2
      by_cases e : j = n
      · subst e
        exact Sem.get!_set!_same _ _ _ (by rw [hsz]; omega)
      · rw [Sem.get!_set!_ne _ _ _ _ (by omega)]
        exact foldl_set!_at off v n b j (by omega) (by omega)

theorem byte_toNat (v : UInt64) (j : Nat) (hj : j < 8) :
    (byteOfV v j).toNat = v.toNat / 2 ^ (8 * j) % 256 := by
  unfold byteOfV
  simp only [UInt64.toNat_toUInt8, UInt64.toNat_and, UInt64.toNat_shiftRight]
  have h1 : (8 * UInt64.ofNat j).toNat % 64 = 8 * j := by
    rw [UInt64.toNat_mul, UInt64.toNat_ofNat']; simp; omega
  rw [h1, Nat.shiftRight_eq_div_pow, show (0xff : UInt64).toNat = 2 ^ 8 - 1 from rfl,
    Nat.and_two_pow_sub_one_eq_mod]
  omega

/-- Eight bytes of a value, little-endian, make the value. -/
theorem reassemble8 (v : UInt64) :
    (List.range 8).foldr (fun i (acc : UInt64) => (acc <<< 8) ||| (byteOfV v i).toUInt64) 0 = v := by
  apply UInt64.toNat_inj.mp
  simp only [List.range, List.range.loop, List.foldr, shl8_or, UInt64.toNat_zero]
  have hv := UInt64.toNat_lt_size v
  simp only [byte_toNat v 0 (by decide), byte_toNat v 1 (by decide), byte_toNat v 2 (by decide),
    byte_toNat v 3 (by decide), byte_toNat v 4 (by decide), byte_toNat v 5 (by decide),
    byte_toNat v 6 (by decide), byte_toNat v 7 (by decide)]
  simp only [UInt64.size] at hv
  omega

/-- **An eight-byte store reads back**, outside pinned memory. -/
theorem load_store8 {m m' : Mem} {a v : UInt64} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned) (h : m.store a 8 v = some m') :
    m'.load a 8 = some v := by
  have hsz : (m'.region r).size = (m.region r).size := store_sizes h r
  unfold Mem.store at h
  simp only [hd, Option.bind_eq_bind, Option.bind_some] at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hc
    injection h with h
    subst h
    rw [load_of hd (by rw [hsz]; omega) (readable_off_pinned hp _ _)]
    congr 1
    rw [Sem.region_setRegion_self]
    unfold bytesAt
    refine (foldr_fun_congr _ (fun i => (byteOfV v i).toUInt64) _ _ fun j hj => ?_).trans (reassemble8 v)
    exact congrArg UInt8.toUInt64 (foldl_set!_at off v 8 _ j (List.mem_range.mp hj) (by omega))

end AlgorithmLib.HProg.Contracts

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Contracts

namespace AlgorithmLib.LibSpec

/-- The eight bytes at `j` in the block `s` starts. -/
def fld (w : World) (s : UInt64) (j : Nat) : Option UInt64 := w.mem.load (s + UInt64.ofNat j) 8

/-- A world with no copy in flight: a heap block's bytes can be read and
    written. -/
def Calm (w : World) : Prop := w.mem.busy = []

/-- **The bytes `bs` can be read at `p`**: none, wherever `p` is, or a run
    inside one region. -/
def Bytes (w : World) (p : UInt64) (bs : List UInt8) : Prop :=
  bs = [] ∨ ∃ r off, decodeAddr p = some (r, off) ∧ off + bs.length ≤ (w.mem.region r).size ∧ off + bs.length ≤ 2 ^ 36 ∧
    w.mem.readable r off bs.length = true ∧ ∀ i, i < bs.length → (w.mem.region r).get! (off + i) = bs.getD i 0

/-- `k` bytes of `bs` from `j`, little-endian. -/
def leN (bs : List UInt8) (j k : Nat) : UInt64 :=
  (List.range k).foldr (fun i acc => (acc <<< 8) ||| (bs.getD (j + i) 0).toUInt64) 0

/-- The bits of a 64-bit constant. -/
def c64 (k : Int) : UInt64 := UInt64.ofNat (k.emod (1 <<< 64)).toNat

/-- **The heap block at `x` is exactly `n` bytes.** -/
def Whole (w : World) (x : UInt64) (n : Nat) : Prop := Blk w x n ∧ ∃ off, w.heapAt? x = some (off, n)

/-- **`k` bytes at `y` end before `x` starts**, both in pinned memory. -/
def Before (y : UInt64) (k : Nat) (x : UInt64) : Prop :=
  ∃ oy ox, decodeAddr y = some (.pinned, oy) ∧ decodeAddr x = some (.pinned, ox) ∧ oy + k ≤ ox

/-- **`w'` is `w` with at most the `n` bytes at `d` rewritten**: the same
    heap, the same sizes and liveness, and every byte outside that run. -/
structure Wrote (w w' : World) (d : UInt64) (n : Nat) : Prop where
  same : MemSame w.mem w'.mem
  heap : w'.heap = w.heap
  get : ∀ r' j, (0 < n → ∀ r off, decodeAddr d = some (r, off) → r' ≠ r ∨ j < off ∨ off + n ≤ j) →
    (w'.mem.region r').get! j = (w.mem.region r').get! j

/-- A store rewrites at most the bytes it writes. -/
theorem Wrote.ofStore {w w' : World} {p x : UInt64} {n : Nat} (hs : w.mem.store p n x = some w'.mem)
    (hh : w'.heap = w.heap) : Wrote w w' p n := by
  obtain ⟨r, off, hd, hF⟩ := store_frame hs
  refine ⟨hF.same, hh, fun r' j hc => ?_⟩
  by_cases e : r' = r
  · subst e
    rcases Nat.eq_zero_or_pos n with hn | hn
    · exact hF.outside j (by omega)
    · rcases hc hn r' off hd with hc | hc | hc
      · exact absurd rfl hc
      · exact hF.outside j (Or.inl hc)
      · exact hF.outside j (Or.inr hc)
  · exact congrArg (·.get! j) (hF.other r' e)

/-- **Two runs in pinned memory are apart**: one ends before the other
    starts. -/
def Apart (x : UInt64) (n : Nat) (y : UInt64) (k : Nat) : Prop := Before x n y ∨ Before y k x

/-- The big-endian number a run of bytes makes. -/
def be : List UInt8 → Nat
  | [] => 0
  | b :: bs => b.toNat * 256 ^ bs.length + be bs

end AlgorithmLib.LibSpec
