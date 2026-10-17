module
public import AlgorithmLib.Host.Heap
meta import AlgorithmLib.Host.Heap
public import AlgorithmLib.Surface.LibHt
meta import AlgorithmLib.Surface.LibHt
@[expose] public section

/-!
# The hash table a context holds

What it is for memory to hold a table of `Lib.Ht`: the context block's header
(`Rep`), its three arrays and every entry at its place (`Lay`, `Tab`), the
index over the entries' keys (`Idx`, by the hash `H` and the slots a probe
visits, `path`), a program's slot holding such a context (`HtAt`), and one
holding null or a table of small values, as it stood before writes outside
the heap (`HtObj`, `OutSteps`). The
statements of the library's functions are over these (`Ship.HtSpec`), and so
is what the checker keeps of a table between a program's calls.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Contracts

namespace HtSpec
open AlgorithmLib.LibSpec AlgorithmLib.LibHt

/-- **The table a context block holds**, as far as its header says: the block
    is live, its bytes can be read and written, and its counts are the
    model's. -/
structure Rep (w : World) (s : UInt64) (T : Ht) : Prop where
  blk : Blk w s SIZE
  calm : Calm w
  created : fld w s CREATED = some (UInt64.ofNat T.handles.length)
  count : fld w s COUNT = some (UInt64.ofNat T.entries.length)
  handles : T.handles = List.range T.handles.length
  small : T.handles.length < 2 ^ 64 ∧ T.entries.length < 2 ^ 64

/-- Where a context's arrays are and how much of each is in use, and where
    in the arena each entry's key and value start. -/
structure Lay where
  e : UInt64
  ecap : Nat
  ix : UInt64
  icap : Nat
  ar : UInt64
  used : Nat
  acap : Nat
  offs : List (Nat × Nat)

/-- **Entry `i` of the entry array is `kv`, its key at `ko` and its value at
    `vo` in the arena**: its four fields say so and how long they are, and the
    arena holds their bytes there. -/
def EntO (w : World) (L : Lay) (i : Nat) (kv : ByteArray × ByteArray) (ko vo : Nat) : Prop :=
  fld w L.e (i * ENTRY) = some (UInt64.ofNat ko) ∧
    fld w L.e (i * ENTRY + 8) = some (UInt64.ofNat kv.1.size) ∧
    fld w L.e (i * ENTRY + 16) = some (UInt64.ofNat vo) ∧
    fld w L.e (i * ENTRY + 24) = some (UInt64.ofNat kv.2.size) ∧
    ko + kv.1.size ≤ L.used ∧ vo + kv.2.size ≤ L.used ∧
    Bytes w (L.ar + UInt64.ofNat ko) kv.1.data.toList ∧ Bytes w (L.ar + UInt64.ofNat vo) kv.2.data.toList

/-- **Entry `i` of the entry array is `kv`**, somewhere in the arena. -/
def EntAt (w : World) (L : Lay) (i : Nat) (kv : ByteArray × ByteArray) : Prop :=
  ∃ ko vo, EntO w L i kv ko vo

/-- **No value's run in the arena meets another run**: each entry's value is
    apart from every entry's key and from every other entry's value, so a
    value can be overwritten where it is. -/
def Sep (T : Ht) (offs : List (Nat × Nat)) : Prop :=
  ∀ i j (hi : i < T.entries.length) (hj : j < T.entries.length) (oi : i < offs.length) (oj : j < offs.length),
    ((offs[i]).2 + (T.entries[i]).2.size ≤ (offs[j]).1 ∨ (offs[j]).1 + (T.entries[j]).1.size ≤ (offs[i]).2) ∧
    (i ≠ j → (offs[i]).2 + (T.entries[i]).2.size ≤ (offs[j]).2 ∨
      (offs[j]).2 + (T.entries[j]).2.size ≤ (offs[i]).2)

/-- **The table a context holds, with its entries**: the header, the three
    arrays as blocks of their capacities, and every entry of the model's table
    at its place in the entry array. -/
structure Tab (w : World) (s : UInt64) (T : Ht) (L : Lay) : Prop where
  rep : Rep w s T
  fe : fld w s ENTRIES = some L.e
  fecap : fld w s ECAP = some (UInt64.ofNat L.ecap)
  fix : fld w s INDEX = some L.ix
  ficap : fld w s ICAP = some (UInt64.ofNat L.icap)
  far : fld w s ARENA = some L.ar
  fused : fld w s AUSED = some (UInt64.ofNat L.used)
  facap : fld w s ACAP = some (UInt64.ofNat L.acap)
  we : Whole w L.e (L.ecap * ENTRY)
  wi : Whole w L.ix (L.icap * 8)
  wa : Whole w L.ar L.acap
  room : T.entries.length ≤ L.ecap ∧ L.used ≤ L.acap
  pos : 0 < L.ecap ∧ 0 < L.icap ∧ 0 < L.acap
  apart : Apart s SIZE L.e (L.ecap * ENTRY) ∧ Apart s SIZE L.ix (L.icap * 8) ∧ Apart s SIZE L.ar L.acap ∧
    Apart L.e (L.ecap * ENTRY) L.ix (L.icap * 8) ∧ Apart L.e (L.ecap * ENTRY) L.ar L.acap ∧
    Apart L.ix (L.icap * 8) L.ar L.acap
  olen : L.offs.length = T.entries.length
  ento : ∀ i (hi : i < T.entries.length) (ho : i < L.offs.length), EntO w L i T.entries[i] (L.offs[i]).1 (L.offs[i]).2
  sep : Sep T L.offs

/-- The hash after `i` words of `bs`. -/
def hashW (bs : List UInt8) : Nat → UInt64
  | 0 => UInt64.ofNat bs.length * c64 MIX
  | i + 1 => (hashW bs i ^^^ leN bs (8 * i) 8) * c64 MIX

/-- The tail word after `j` bytes from `base`. -/
def hashT (bs : List UInt8) (base : Nat) : Nat → UInt64
  | 0 => c64 0
  | j + 1 => hashT bs base j ||| (leN bs (base + j) 1 <<< ((UInt64.ofNat j <<< (c64 3 % 64)) % 64))

/-- **The hash of a key**, as `LibHt.hash` computes it. -/
def H (bs : List UInt8) : UInt64 :=
  let h := (hashW bs (bs.length / 8) ^^^ hashT bs (8 * (bs.length / 8)) (bs.length % 8)) * c64 MIX
  let h := h ^^^ (h >>> (c64 32 % 64))
  let h := h * c64 FIN
  h ^^^ (h >>> (c64 29 % 64))

/-- What an index slot holds for entry `p` whose key hashes to `h`: the
    hash's high half above, the entry's position plus one below. -/
def slotVal (h : UInt64) (p : Nat) : UInt64 :=
  ((h >>> (c64 32 % 64)) <<< (c64 32 % 64)) ||| UInt64.ofNat (p + 1)

/-- The index's mask, for a capacity. -/
def maskOf (icap : Nat) : UInt64 := UInt64.ofNat icap + c64 (-1)

/-- The slots a probe for hash `h` visits, in order. -/
def path (icap : Nat) (h : UInt64) : Nat → UInt64
  | 0 => h &&& maskOf icap
  | t + 1 => (path icap h t + c64 1) &&& maskOf icap

/-- The keys of a table's entries, in order. -/
def keysOf (T : Ht) : List (List UInt8) := T.entries.map (·.1.data.toList)

/-- **The index covers the keys `ks`**: they are distinct, every slot is
    empty or holds one of them, and each is reached from its hash's first
    slot before any empty slot. -/
structure Idx (w : World) (L : Lay) (ks : List (List UInt8)) : Prop where
  cap : 0 < L.icap ∧ L.icap ≤ 2 ^ 33
  small : ks.length < 2 ^ 32
  nodup : ks.Nodup
  slots : ∀ j, j < L.icap → ∃ v, fld w L.ix (8 * j) = some v ∧
    (v = 0 ∨ ∃ p, ∃ hp : p < ks.length, v = slotVal (H ks[p]) p)
  chain : ∀ p (hp : p < ks.length), ∃ t, fld w L.ix (8 * (path L.icap (H ks[p]) t).toNat) =
      some (slotVal (H ks[p]) p) ∧
    ∀ t', t' < t → fld w L.ix (8 * (path L.icap (H ks[p]) t').toNat) ≠ some 0

/-- **A program's slot holds the context `s`, which holds the table `T`**:
    the slot is outside the heap and holds `s`, and `s` is a context holding
    `T` with its index. What a program that uses the table keeps from call to
    call. -/
def HtAt (w : World) (slot s : UInt64) (T : Ht) : Prop :=
  ∃ r off L, decodeAddr slot = some (r, off) ∧ r ≠ .pinned ∧ w.mem.load slot 8 = some s ∧ Tab w s T L ∧
    Idx w L (keysOf T)

/-- **`w'` is `w` after writes outside the heap**: any number of runs of
    bytes rewritten, none in pinned memory. -/
inductive OutSteps : World → World → Prop
  | refl (w : World) : OutSteps w w
  | step {w w' w'' : World} {p : UInt64} {n : Nat} {r : Region} {off : Nat} : OutSteps w w' → Wrote w' w'' p n →
      decodeAddr p = some (r, off) → r ≠ .pinned → OutSteps w w''

/-- **A program's slot holds null or a table whose values are at most `n`
    bytes, made at most `k` times**: the slot is outside the heap, and what it
    holds is null, or a context that held such a table with its index when
    the heap was last written. What the checker keeps of a table between a
    program's calls. -/
def HtObj (w : World) (slot : UInt64) (n k : Nat) : Prop :=
  ∃ s r off, decodeAddr slot = some (r, off) ∧ r ≠ .pinned ∧ w.mem.load slot 8 = some s ∧
    (s = 0 ∨ ∃ w0 T L, OutSteps w0 w ∧ Tab w0 s T L ∧ Idx w0 L (keysOf T) ∧
      (∀ kv ∈ T.entries, kv.2.size ≤ n) ∧ T.handles.length ≤ k)

/-- A world with the same memory and heap is one after no write. -/
theorem OutSteps.of_mem {w0 w w' : World} (h : OutSteps w0 w) (hm : w'.mem = w.mem) (hh : w'.heap = w.heap) :
    OutSteps w0 w' :=
  .step (p := regionBase .arena) (n := 0) h ⟨by rw [hm]; exact MemSame.refl _, hh, fun _ _ _ => by rw [hm]⟩
    (show decodeAddr (regionBase .arena) = some (.arena, 0) by decide) (by decide)

/-- What changes neither memory nor the heap leaves a table at its slot. -/
theorem HtObj.of_mem {w w' : World} {slot : UInt64} {n k : Nat} (h : HtObj w slot n k) (hm : w'.mem = w.mem)
    (hh : w'.heap = w.heap) : HtObj w' slot n k := by
  obtain ⟨s, r, off, hd, hr, hl, hT⟩ := h
  exact ⟨s, r, off, hd, hr, by rw [hm]; exact hl,
    hT.imp id fun ⟨w0, T, L, ho, rest⟩ => ⟨w0, T, L, ho.of_mem hm hh, rest⟩⟩

/-- **A store outside the heap that leaves a slot's eight bytes leaves the
    table at the slot.** -/
theorem HtObj.stored {w w' : World} {slot a x : UInt64} {n c k : Nat} {r : Region} {off : Nat}
    (h : HtObj w slot n c)
    (hs : w.mem.store a k x = some w'.mem) (hh : w'.heap = w.heap) (hd : decodeAddr a = some (r, off))
    (hr : r ≠ .pinned) (hl : w'.mem.load slot 8 = w.mem.load slot 8) : HtObj w' slot n c := by
  obtain ⟨s, r0, off0, hd0, hr0, hl0, hT⟩ := h
  exact ⟨s, r0, off0, hd0, hr0, hl.trans hl0,
    hT.imp id fun ⟨w0, T, L, ho, rest⟩ => ⟨w0, T, L, .step ho (Wrote.ofStore hs hh) hd hr, rest⟩⟩

/-- Larger bounds hold of a table. -/
theorem HtObj.mono {w : World} {slot : UInt64} {n m k j : Nat} (h : HtObj w slot n k) (hm : n ≤ m) (hk : k ≤ j) :
    HtObj w slot m j := by
  obtain ⟨s, r, off, hd, hr, hl, hT⟩ := h
  exact ⟨s, r, off, hd, hr, hl,
    hT.imp id fun ⟨w0, T, L, ho, hT, hI, hb, hc⟩ =>
      ⟨w0, T, L, ho, hT, hI, fun kv hkv => Nat.le_trans (hb kv hkv) hm, Nat.le_trans hc hk⟩⟩

end HtSpec
