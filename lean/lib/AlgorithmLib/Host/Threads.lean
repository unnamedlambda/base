module
public import AlgorithmLib.Host.Heap
@[expose] public section

/-!
# `Host.Threads` — what a thread sees, and why no two parties race

A thread the program starts (`ffiThreadSpawn`) runs the program's function
`u0:fn` on one word and owns the arena bytes its start lends it until it is
joined (`ffiThreadJoin`). This file states what a thread runs in and what it
and the program can change while it runs.

* `Thread.view`: the world a thread runs in holds its span's bytes, at the
  addresses the program knows them by, and nothing else it may read or write.
* `ThreadHonest`: what the machine answers at a join (`World.threadOut`) is
  what the function leaves in its span and returns, run in that view.
* No race with the program: every load and store the program makes in the
  arena is clear of every span a running thread owns (`Mem.load_clear`,
  `Mem.store_clear`), so a store leaves every owned byte as it was
  (`Mem.store_lent_bytes`).
* No race between threads: the spans running threads own lie apart
  (`LentApart`), which a start and a join keep.
* A thread stays in its span: its view is confined to it (`Confined`), and
  every load and store in confined memory reads or writes inside the span and
  keeps it confined (`Confined.load`, `Confined.store`).
* A join changes the arena only in the thread's span, to what the thread
  left there, and answers what it returned (`ffiThreadJoin_honest`).
-/

open AlgorithmLib.IR AlgorithmLib.HProg

namespace AlgorithmLib.HProg.Sem

-- ---------------------------------------------------------------------------
-- What a thread runs in
-- ---------------------------------------------------------------------------

/-- **The world a thread runs in**: an arena holding the span's bytes at
    their offsets, every byte before them lent away, and no other resource. -/
def Thread.view (t : Thread) (bs : ByteArray) : World :=
  { mem := { arena := ByteArray.mk (Array.replicate t.off 0) ++ bs, data := .empty, out := .empty,
             lent := [(0, t.off)] } }

/-- **The thread side, named once.** What the machine answers at a thread's
    join, on the bytes of its span, is what its function leaves in the span
    and returns, run on its word in the thread's view of those bytes. -/
def ThreadHonest (L : Locals) (out : Thread → ByteArray → ByteArray × UInt64) : Prop :=
  ∀ t bs r w', L t.fn [.sc .i64 t.arg] (t.view bs) = .ok (some (.sc .i64 r), w') →
    out t bs = (w'.mem.arena.extract t.off (t.off + t.len), r)

/-- What the program's functions `L` answer at a join: the function run in
    the thread's view, where it returns a word; the bytes as they were and
    `0` where it does not. -/
def Thread.runOut (L : Locals) (t : Thread) (bs : ByteArray) : ByteArray × UInt64 :=
  match L t.fn [.sc .i64 t.arg] (t.view bs) with
  | .ok (some (.sc .i64 r), w') => (w'.mem.arena.extract t.off (t.off + t.len), r)
  | _ => (bs, 0)

theorem Thread.runOut_honest (L : Locals) : ThreadHonest L (Thread.runOut L) := by
  intro t bs r w' h
  simp only [Thread.runOut, h]

-- ---------------------------------------------------------------------------
-- No race with the program
-- ---------------------------------------------------------------------------

/-- A load the program makes in the arena is clear of every span a running
    thread owns. -/
theorem Mem.load_clear {m : Mem} {a : UInt64} {n : Nat} {v : UInt64} {off : Nat}
    (h : m.load a n = some v) (hd : decodeAddr a = some (.arena, off)) : m.lentClear off n = true := by
  obtain ⟨-, hr, -⟩ := Contracts.load_some hd h
  simpa [Mem.LentOk] using Mem.lentOk_of_readable hr

/-- A store the program makes in the arena is clear of every span a running
    thread owns. -/
theorem Mem.store_clear {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} {off : Nat}
    (h : m.store a n v = some m') (hd : decodeAddr a = some (.arena, off)) : m.lentClear off n = true := by
  unfold Mem.store at h
  simp only [hd, Option.bind_eq_bind, Option.bind_some] at h
  split at h
  · cases h
  · rename_i hc
    have hr : m.reachable .arena off n = true := by
      cases hr : m.reachable .arena off n
      · simp [hr] at hc
      · rfl
    simpa [Mem.LentOk] using Mem.lentOk_of_reachable hr

/-- **A store leaves every byte a running thread owns.** -/
theorem Mem.store_lent_bytes {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m')
    {o l : Nat} (hol : (o, l) ∈ m.lent) {j : Nat} (hj : o ≤ j) (hj' : j < o + l) :
    m'.arena.get! j = m.arena.get! j := by
  obtain ⟨r, off, hd, -, he⟩ := store_some h
  cases r with
  | arena =>
      have hc := Mem.store_clear h hd
      unfold Mem.lentClear at hc
      have hx := List.all_eq_true.mp hc _ hol
      simp only [Bool.or_eq_true, decide_eq_true_eq] at hx
      rw [he]
      exact (foldl_set!_other off j v _ _ fun i hi => by have := List.mem_range.mp hi; omega).1
  | _ => rw [he]; rfl

-- ---------------------------------------------------------------------------
-- A thread stays in its span
-- ---------------------------------------------------------------------------

/-- Memory confined to the arena's bytes `[off, off + len)`: no other region
    holds a byte, the arena ends with the span, and the bytes before it are
    lent away. -/
def Confined (m : Mem) (off len : Nat) : Prop :=
  m.arena.size ≤ off + len ∧ (0, off) ∈ m.lent ∧ ∀ r, r ≠ .arena → m.sizes r = 0

/-- **A thread's view is confined to its span.** -/
theorem Thread.view_confined (t : Thread) (bs : ByteArray) (hb : bs.size = t.len) :
    Confined (t.view bs).mem t.off t.len := by
  refine ⟨?_, List.mem_singleton_self _, fun r hr => ?_⟩
  · show (ByteArray.mk (Array.replicate t.off 0) ++ bs).size ≤ _
    rw [ByteArray.size_append, hb]
    show (Array.replicate t.off (0 : UInt8)).size + t.len ≤ _
    rw [Array.size_replicate]; exact Nat.le_refl _
  · cases r <;> first | exact absurd rfl hr | rfl

/-- In confined memory, a load of any bytes reads inside the span. -/
theorem Confined.load {m : Mem} {off len : Nat} (hc : Confined m off len) {a : UInt64} {n : Nat} {v : UInt64}
    (h : m.load a n = some v) (hn : 0 < n) :
    ∃ o, decodeAddr a = some (.arena, o) ∧ off ≤ o ∧ o + n ≤ off + len := by
  cases hd : decodeAddr a with
  | none => simp [Mem.load, hd] at h
  | some p =>
    obtain ⟨r, o⟩ := p
    obtain ⟨hs, -, -⟩ := Contracts.load_some hd h
    by_cases hr : r = .arena
    · subst hr
      have hcl := Mem.load_clear h hd
      unfold Mem.lentClear at hcl
      have hx := List.all_eq_true.mp hcl _ hc.2.1
      simp only [Bool.or_eq_true, decide_eq_true_eq] at hx
      exact ⟨o, rfl, by omega, by have := hc.1; exact Nat.le_trans hs this⟩
    · have := hc.2.2 r hr
      unfold Mem.sizes at this
      omega

/-- In confined memory, a store of any bytes writes inside the span, and
    leaves the memory confined. -/
theorem Confined.store {m m' : Mem} {off len : Nat} (hc : Confined m off len) {a : UInt64} {n : Nat}
    {v : UInt64} (h : m.store a n v = some m') (hn : 0 < n) :
    (∃ o, decodeAddr a = some (.arena, o) ∧ off ≤ o ∧ o + n ≤ off + len) ∧ Confined m' off len := by
  have hsz := Contracts.store_sizes_eq h
  have hl := Mem.store_lent h
  refine ⟨?_, ?_, by rw [hl]; exact hc.2.1, fun r hr => by rw [hsz]; exact hc.2.2 r hr⟩
  · obtain ⟨r, o, hd, hs, -⟩ := store_some h
    by_cases hr : r = .arena
    · subst hr
      have hcl := Mem.store_clear h hd
      unfold Mem.lentClear at hcl
      have hx := List.all_eq_true.mp hcl _ hc.2.1
      simp only [Bool.or_eq_true, decide_eq_true_eq] at hx
      exact ⟨o, hd, by omega, Nat.le_trans hs hc.1⟩
    · have := hc.2.2 r hr
      unfold Mem.sizes at this
      omega
  · have : m'.sizes .arena = m.sizes .arena := by rw [hsz]
    exact Nat.le_trans (Nat.le_of_eq this) hc.1

-- ---------------------------------------------------------------------------
-- No race between threads
-- ---------------------------------------------------------------------------

/-- The spans running threads own lie apart. -/
def LentApart (m : Mem) : Prop :=
  m.lent.Pairwise fun a b => a.1 + a.2 ≤ b.1 ∨ b.1 + b.2 ≤ a.1

/-- A thread's start keeps the spans apart: the one it lends is clear of
    every other. -/
theorem ffiThreadSpawn_apart {bits : List UInt64} {w w' : World} {r : Option V}
    (h : ffiThreadSpawn bits w = some (r, w')) (ha : LentApart w.mem) : LentApart w'.mem := by
  unfold ffiThreadSpawn at h
  split at h
  · split at h
    · cases h
    · split at h
      · cases h
      · rename_i off len _
        split at h
        · cases h
        · rename_i hc
          have hcl : w.mem.lentClear off len = true := by
            cases e : w.mem.lentClear off len
            · simp [e] at hc
            · rfl
          split at h
          · cases h; exact ha
          · cases h
            refine List.Pairwise.cons (fun b hb => ?_) ha
            have hx := List.all_eq_true.mp hcl b hb
            simp only [Bool.or_eq_true, decide_eq_true_eq] at hx
            exact hx
  · cases h

-- ---------------------------------------------------------------------------
-- What a join changes
-- ---------------------------------------------------------------------------

theorem get!_of_data {a b : ByteArray} {i j : Nat} (h : a.data[i]? = b.data[j]?) : a.get! i = b.get! j := by
  show a.data[i]! = b.data[j]!
  rw [getElem!_def, getElem!_def, h]

theorem fitBytes_size (n : Nat) (bs : ByteArray) : (fitBytes n bs).size = n := by
  unfold fitBytes
  simp only [ByteArray.size_append, ByteArray.size_extract]
  show _ + (Array.replicate _ _).size = _
  rw [Array.size_replicate]; omega

theorem fitBytes_get {n : Nat} {bs : ByteArray} {j : Nat} (hn : j < n) (hb : j < bs.size) :
    (fitBytes n bs).get! j = bs.get! j := by
  apply get!_of_data
  unfold fitBytes
  have hd : bs.data.size = bs.size := rfl
  have h1 : (bs.extract 0 n).data.size = min n bs.data.size - 0 := by
    rw [ByteArray.data_extract, Array.size_extract]
  rw [ByteArray.data_append, Array.getElem?_append_left (by rw [h1]; omega), ByteArray.data_extract,
    Array.getElem?_extract, if_pos (by omega)]
  simp

/-- Copying `l` bytes over `[o, o + l)` of an array they fit in keeps its size. -/
theorem copySlice_size {src dest : ByteArray} {o l : Nat} (hs : src.size = l) (hd : o + l ≤ dest.size) :
    (src.copySlice 0 dest o l).size = dest.size := by
  rw [ByteArray.copySlice_eq_append]
  simp only [ByteArray.size_append, ByteArray.size_extract]
  have h1 : src.data.size = l := hs
  have h2 : dest.data.size = dest.size := rfl
  rw [h2]; omega

/-- ... and every byte outside it. -/
theorem copySlice_get_out {src dest : ByteArray} {o l i : Nat} (hs : src.size = l) (hd : o + l ≤ dest.size)
    (hi : i < o ∨ o + l ≤ i) : (src.copySlice 0 dest o l).get! i = dest.get! i := by
  apply get!_of_data
  simp only [ByteArray.copySlice]
  have h1 : src.data.size = l := hs
  have h2 : dest.data.size = dest.size := rfl
  have hm : min l (src.data.size - 0) = l := by omega
  rw [hm]
  rcases hi with hi | hi
  · rw [Array.getElem?_append_left (by simp only [Array.size_append, Array.size_extract]; omega),
      Array.getElem?_append_left (by simp only [Array.size_extract]; omega), Array.getElem?_extract]
    rw [if_pos (by omega)]; simp
  · rw [Array.getElem?_append_right (by simp only [Array.size_append, Array.size_extract]; omega),
      Array.getElem?_extract]
    simp only [Array.size_append, Array.size_extract]
    split
    · congr 1; omega
    · rw [Array.getElem?_eq_none (by omega)]

/-- ... and puts the source's bytes inside it. -/
theorem copySlice_get_in {src dest : ByteArray} {o l j : Nat} (hs : src.size = l) (hd : o + l ≤ dest.size)
    (hj : j < l) : (src.copySlice 0 dest o l).get! (o + j) = src.get! j := by
  apply get!_of_data
  simp only [ByteArray.copySlice]
  have h1 : src.data.size = l := hs
  have h2 : dest.data.size = dest.size := rfl
  rw [Array.getElem?_append_left (by simp only [Array.size_append, Array.size_extract]; omega),
    Array.getElem?_append_right (by simp only [Array.size_extract]; omega), Array.getElem?_extract]
  simp only [Array.size_extract]
  rw [if_pos (by omega)]
  congr 1; omega

/-- A byte of a span cut from an array is the array's byte there. -/
theorem extract_get {a : ByteArray} {o l j : Nat} (hd : o + l ≤ a.size) (hj : j < l) :
    (a.extract o (o + l)).get! j = a.get! (o + j) := by
  apply get!_of_data
  rw [ByteArray.data_extract, Array.getElem?_extract, if_pos (by
    have : a.data.size = a.size := rfl
    omega)]

/-- A join keeps the spans apart. -/
theorem ffiThreadJoin_apart {bits : List UInt64} {w w' : World} {r : Option V}
    (h : ffiThreadJoin bits w = some (r, w')) (ha : LentApart w.mem) : LentApart w'.mem := by
  unfold ffiThreadJoin at h
  split at h
  · simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
    obtain ⟨n, -, t, -, h⟩ := h
    cases h
    exact ha.sublist List.erase_sublist
  · cases h

/-- **A join, on an honest machine**: of a running thread whose function,
    run in its view of its span, returns `x` and leaves `wc`, the join answers
    `x`, keeps the arena's size and every byte outside the span, and puts in
    the span what the function left there. -/
theorem ffiThreadJoin_honest {L : Locals} {w w' wc : World} {h x : UInt64} {r : Option V} {n : Nat}
    {t : Thread} (hon : ThreadHonest L w.threadOut) (hn : threadOf h = some n)
    (ht : w.threads[n]? = some (some t)) (hfit : t.off + t.len ≤ w.mem.arena.size)
    (hrun : L t.fn [.sc .i64 t.arg] (t.view (w.mem.arena.extract t.off (t.off + t.len))) =
      .ok (some (.sc .i64 x), wc))
    (hwc : t.off + t.len ≤ wc.mem.arena.size)
    (h : ffiThreadJoin [h] w = some (r, w')) :
    r = some (.sc .i64 x) ∧ w'.mem.arena.size = w.mem.arena.size ∧
      (∀ i, i < t.off ∨ t.off + t.len ≤ i → w'.mem.arena.get! i = w.mem.arena.get! i) ∧
      ∀ j, j < t.len → w'.mem.arena.get! (t.off + j) = wc.mem.arena.get! (t.off + j) := by
  have ho := hon _ _ _ _ hrun
  simp only [ffiThreadJoin, hn, Option.bind_eq_bind, Option.bind_some, ht, Option.join_some, ho,
    Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  have hfs := fitBytes_size t.len (wc.mem.arena.extract t.off (t.off + t.len))
  refine ⟨rfl, copySlice_size hfs hfit, fun i hi => copySlice_get_out hfs hfit hi, fun j hj => ?_⟩
  show ((fitBytes t.len _).copySlice 0 w.mem.arena t.off t.len).get! (t.off + j) = _
  rw [copySlice_get_in hfs hfit hj, fitBytes_get hj (by simp only [ByteArray.size_extract]; omega),
    extract_get hwc hj]

end AlgorithmLib.HProg.Sem
