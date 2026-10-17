import AlgorithmLib.Proof.LibSpec
import AlgorithmLib.Host.HtRep
import AlgorithmLib.Surface.Link

/-!
# The hash table's functions do what the model says

`Surface.LibHt` implements the hash-table entry points as CLIF over the C
heap. The model is a table `Ht` that is no memory at all: its entries in the
order they were first inserted and the handles made, with `Ht.get` and
`Ht.set`. `Rep w s T` relates the two: the context block at `s` in world `w` holds the
table `T`. Each function is then proven once, as `LibSpec.Ans`: handed a
context holding `T`, it neither misuses a call nor faults, answers what the
model answers for `T`, and leaves a context holding the table the model
leaves (`create_spec`, `count_spec`); handed no context it answers the
model's refusal and changes nothing (`create_null`, `count_null`).

`Rep` states the block's header: the handle count and the entry count.
`Owned` states the three array fields: each holds null or a block of its own,
which `htCleanup` needs to free them and the context (`cleanup_spec`), and
which a table's context has (`Tab.owned`).
`wp_withTable` is the guard every accessor starts with.

`Tab w s T L` is the whole of it but the index: the header, the three arrays
as blocks at the places and capacities `L` names, and each entry of `T` at its
place in the entry array with its key's and value's bytes in the arena where
`L.offs` says (`EntO`; `Tab.ent` forgets where), the four blocks apart from
one another, and no value's run in the arena meeting another run (`Sep`), so
that a value can be overwritten where it is. A
write clear of those blocks (`Clear`) leaves it (`Tab.wrote`). `htGetEntry`,
handed room for an entry's key and value clear of the table, answers the
key's length and leaves the memory the model's two copies leave
(`getEntry_spec`).

`init_spec` is where a context comes from: `htInit` leaves in its slot a
context holding the empty table with its arrays and an index of empty slots,
or null where one of the four could not be allocated.

`H` is the hash of a key as `LibHt.hash` computes it (`wp_hash`), and
`LibHt.bytesEq` answers whether two runs of one length are the same bytes
(`wp_bytesEq`); neither changes anything.

The index is open-addressed: `path` is the slots a probe for a hash visits,
`slotVal` what a slot holds for an entry, and `Probe` what a probe needs of a
table: every slot empty or holding an entry, and every entry reached from its
hash's first slot before any empty one. Over such a table `LibHt.find`
answers a slot and whether the key is there (`Found`, `wp_find`): the entry
with that key, or the empty slot the key would go in, the key being in none.

`Idx w L ks` is the index covering the distinct keys `ks`; a table with its
index is one a probe can read (`Tab.probe`), and a write clear of the index
leaves it (`Idx.wrote`). `htLookup` of a key that can be read, handed room
for its value clear of the table, answers the value's length and leaves the
memory the model's copy leaves, or answers the model's refusal
(`lookup_spec`), with `get_none` and `get_some` saying which value the model
finds.

Keys and values live in the arena, which `LibHt.append` grows by moving it to
a fresh block. `Appended w w' s L L' bs` is what that leaves: a layout that
differs in the arena alone, what the arena held at the same places, `bs` after
it, and everything off the old arena and the header's arena fields kept
(`AppP`). `wp_append`: the bytes are appended and their place answered, or
`-1` is and nothing but the heap has changed. A table whose arena was
appended to is the table at the new layout, and its index the index
(`Tab.appended`, `Idx.appended`).

The entry array and the index grow the same way, each by `LibSpec.moved`:
`Tab.entGrown` is the entry array moved to a fresh block with its entries
copied, `Tab.reindexed` the index moved to a fresh block of empty slots. A
new entry is an entry in place past the last, then counted (`Tab.push`), and
its key written into the empty slot its probe ends at (`Idx.insert`). A write
off the context block, the entries in use and the arena in use leaves the
table (`Tab.wroteOff`).

`Kept w w' s L L'` is what every change to the table leaves of the rest of
the world: every run clear of the table kept, every block that is not the
table's there, and each block of the table one it had or one past the memory
there was, so that what was clear of the table still is (`Kept.clear`). A
write inside one of the table's blocks, a block of it moved and an append
each leave that (`Kept.wrote`, `Moved.kept`, `Appended.kept`), and it
composes (`Kept.trans`). `wp_grow`: `LibHt.grow` rebuilds the index at twice
its size by probing for each key in turn over the index of the keys before it
(`Tab.probeTake`), or leaves it where the new one cannot be allocated; the
table and its index either way.

`wp_insertAt`: `LibHt.insertAt` of a key no entry has, at the empty slot its
probe ended at (`Spot`), handed the key's and the value's bytes off the entry
array, the arena and the header fields that name them (`Src`), leaves the
table with the new entry (`pushed`) and an index that covers its key, or the
table as it was where the entry array or the arena could not be grown
(`InsQ`); the rest of the world is `Kept` either way.

`wp_set`: `LibHt.set` of a key and a value that can be read, clear of the
table, leaves the table the model's `Ht.set` leaves, or the table as it was
(`SetQ`). The model sets a key no entry has by adding an entry (`set_absent`)
and one an entry has by replacing that entry's value (`set_present`,
`setVal`), which the library does where the value is when the new one is as
long (`Tab.setInPlace`, by `Sep`) and at the arena's end otherwise
(`Tab.setMoved`). `insert_spec` is `htInsert`: what `set` leaves, or nothing
changed before a handle has been made (`InsertQ`); `insert_null` is the call
handed no context.

`increment_spec` is `htIncrement`: where no entry has the key, the addend
answered and a new entry holding its eight bytes, inserted from the header's
scratch field (`Tab.scratch`); where one has, with a value of at least eight
bytes, the sum answered and stored over the value's first eight bytes
(`bumped`, `LibSpec.stored_bytes`). `increment_null` is the call handed no
context.

`lookup_null` and `getEntry_null` are those calls handed no context: the
answer for no table, and nothing changed.

The `_calls` theorems say the same of a linked program's calls: where the
program's function table has one of these functions, a call to it at any call
depth neither misuses a call nor faults and answers and leaves what the
function's statement says (`LibSpec.Ans.calls`, `LibSpec.Does.calls`).

`HtAt w slot s T` is what a program that uses the table keeps from call to
call: its slot, outside the heap, holds the address `s` of a context that
holds `T` with its index. A write outside the heap that leaves the slot
leaves it (`HtAt.wrote`), as does one clear of the table that misses the slot
(`HtAt.wroteClear`, `Misses`), and each entry point's statement is a step on
it: `HtAt.of_init`, `HtAt.of_create` by `create_tab`, `HtAt.of_set`,
`HtAt.of_insert`, `HtAt.of_increment`, and, the table unchanged,
`HtAt.of_lookup` and `HtAt.of_getEntry`, whose copies out are writes
(`LookQ`, `GetQ`). `HtAt.rep` and `HtAt.owned` are what `htCount` and
`htCleanup` ask of the slot.

The `_at` theorems are a linked program's calls with the table at its slot
and its buffers outside the heap (`Out`): each leaves at the slot the table
the entry point's model leaves (`incOf` for `htIncrement`), or, where an
allocation failed, the table as it was. Noting a call leaves the table
(`HtAt.obs`). Each says too what the program keeps of its own memory
(`LibSpec.Outside`): everything outside the heap, but for the slot `htInit`
and `htCleanup` write and the runs `htLookup` and `htGetEntry` copy to.

Every statement here is of partial correctness: a function that returns has
done what its statement says, and no step of it misuses a call or faults; that
a probe ends is not shown.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace HtSpec
open AlgorithmLib.LibSpec AlgorithmLib.LibHt

/-- **A context block and its three arrays**: the block is live, and its
    array fields each hold null or a block of its own. -/
structure Owned (w : World) (s e i a : UInt64) : Prop where
  blk : Blk w s SIZE
  entries : fld w s ENTRIES = some e
  index : fld w s INDEX = some i
  arena : fld w s ARENA = some a
  own : Own w e ∧ Own w i ∧ Own w a
  apart : e ≠ s ∧ i ≠ s ∧ a ≠ s ∧ (i = 0 ∨ i ≠ e) ∧ (a = 0 ∨ a ≠ e) ∧ (a = 0 ∨ a ≠ i)

/-- A store into the block clear of its three array fields leaves them. -/
theorem Owned.setFld {w : World} {s e i a x : UInt64} {j : Nat} {m : Mem} (h : Owned w s e i a) (hj : j + 8 ≤ SIZE)
    (hE : j + 8 ≤ ENTRIES ∨ ENTRIES + 8 ≤ j) (hI : j + 8 ≤ INDEX ∨ INDEX + 8 ≤ j)
    (hA : j + 8 ≤ ARENA ∨ ARENA + 8 ≤ j) (hm : w.mem.store (s + UInt64.ofNat j) 8 x = some m) :
    Owned (stored w (s + UInt64.ofNat j) x m) s e i a :=
  ⟨Blk.keep h.blk hm, (fld_kept h.blk (by decide) hj hE.symm hm).trans h.entries,
    (fld_kept h.blk (by decide) hj hI.symm hm).trans h.index, (fld_kept h.blk (by decide) hj hA.symm hm).trans h.arena,
    ⟨h.own.1.stored hm, h.own.2.1.stored hm, h.own.2.2.stored hm⟩, h.apart⟩

theorem create_fine : Fine (emitGo create ⟨1, 0, [], [], [], none⟩).2 := ⟨rfl, by decide⟩

/-- **`htCreate` on no context** answers `0xFFFFFFFF` and changes nothing. -/
theorem create_null (cfg : Cfg) (w : World) :
    Ans cfg [.sc .i64 0] w create [.i64] (fun x w' => x = 0xFFFFFFFF ∧ w' = w) := by
  refine ans_of_wp ?_ create_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [create, entryParams, wp_bind, wp_params, carriesFrom, wp_ret, wp_pure]
  have h0 : (#[V.sc .i64 0] : Env)[0]? = some (.sc .i64 0) := rfl
  apply wp_iconst64
  refine wp_ifte (AnsJ fun x w' => x = 0xFFFFFFFF ∧ w' = w)
    (x := 0) (y := c64 0) (by slot) (by slot) ?_ ?_ ?_
  · intro _
    wp_seq
    apply wp_iconst64
    exact ⟨_, get_push_size, by decide, rfl⟩
  · intro h
    exact absurd h (by decide)
  · join_ans

/-- **`htCreate`** answers the number of handles made so far and counts one
    more, as the model does, and leaves the context's arrays. -/
theorem create_spec (cfg : Cfg) (w : World) (s : UInt64) (T : Ht) (h : Rep w s T)
    (hn : T.handles.length + 1 < 2 ^ 64) :
    Ans cfg [.sc .i64 s] w create [.i64] (fun x w' => x = UInt64.ofNat T.handles.length ∧
      Rep w' s { T with handles := T.handles ++ [T.handles.length] } ∧
      ∀ e i a, Owned w s e i a → Owned w' s e i a) := by
  refine ans_of_wp ?_ create_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [create, entryParams, wp_bind, wp_params, carriesFrom, wp_ret, wp_pure]
  have h0 : (#[V.sc .i64 s] : Env)[0]? = some (.sc .i64 s) := rfl
  apply wp_iconst64
  refine wp_ifte (AnsJ fun x w' => x = UInt64.ofNat T.handles.length ∧
      Rep w' s { T with handles := T.handles ++ [T.handles.length] } ∧
      ∀ e i a, Owned w s e i a → Owned w' s e i a)
    (x := s) (y := c64 0) (by slot) (by slot) ?_ ?_ (by join_ans)
  · intro he
    exact absurd (by simpa [cmpInt, show c64 0 = 0 from by decide] using he) (Blk.ne_zero h.blk)
  · intro _ ne hne
    wp_seq
    refine wp_at (x := s) (by slot) ?_
    refine wp_load64 (by slot) h.created ?_
    refine wp_iaddImm (by slot) ?_
    obtain ⟨m, hm⟩ := Blk.fld_store h.blk h.calm (j := CREATED) (by decide)
      (UInt64.ofNat T.handles.length + c64 1)
    refine wp_storeI64 (by slot) (by slot) hm ?_
    refine ⟨_, by slot, rfl, ?_, fun e i a ho =>
      ho.setFld (j := CREATED) (by decide) (by decide) (by decide) (by decide) hm⟩
    exact {
      blk := Blk.keep h.blk hm
      calm := h.calm.stored hm
      created := by
        rw [fld_stored hm]
        simp [show c64 1 = 1 from by decide, UInt64.ofNat_add]
      count := (fld_kept h.blk (by decide) (by decide) (by decide) hm).trans h.count
      handles := by
        show T.handles ++ [T.handles.length] = List.range (T.handles ++ [T.handles.length]).length
        rw [List.length_append, List.length_singleton, List.range_succ, ← h.handles]
      small := ⟨by simpa using hn, h.small.2⟩ }

/-- The model's table exists once a handle has been made. -/
theorem Rep.hasTable {w : World} {s : UInt64} {T : Ht} (h : Rep w s T) :
    T.hasTable = decide (T.handles.length ≠ 0) := by
  unfold Ht.hasTable
  rw [h.handles]
  cases hl : T.handles.length with
  | zero => simp
  | succ n => simp [List.mem_range]

/-- **`withTable`**: the default where there is no context or no table yet,
    the continuation where the table has been made. -/
theorem wp_withTable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World} {s : Slot .i64} {x c : UInt64}
    {dflt k : Prog Slot Lvl (Slot .i64)} {R : UInt64 → World → Prop}
    (hs : Γ[s]? = some (.sc .i64 x)) (hc : x ≠ 0 → fld w x CREATED = some c)
    (hd : x = 0 ∨ c = 0 → ∀ Γ', Ext Γ Γ' → wp cfg d J dflt (AnsQ R) Γ' w)
    (hk : x ≠ 0 → c ≠ 0 → ∀ Γ', Ext Γ Γ' → wp cfg d J k (AnsQ R) Γ' w) :
    wp cfg d J (withTable s dflt k) (AnsQ R) Γ w := by
  have z0 : c64 0 = 0 := by decide
  unfold withTable
  wp_seq
  apply wp_iconst64
  refine wp_ifte (AnsJ R) (x := x) (y := c64 0) (by slot) (by slot) ?_ ?_ (by join_ans)
  · intro he
    wp_seq
    exact hd (Or.inl (by simpa [cmpInt, z0] using he)) _ (by ext_env)
  · intro he ne hne
    have hx : x ≠ 0 := by simpa [cmpInt, z0] using he
    wp_seq
    refine wp_at (by slot) ?_
    refine wp_load64 (by slot) (hc hx) ?_
    refine wp_ifte (AnsJ R) (x := c) (y := c64 0) (by slot) (by slot) ?_ ?_ (by join_ans)
    · intro he'
      wp_seq
      exact hd (Or.inr (by simpa [cmpInt, z0] using he')) _ (by ext_env)
    · intro he' ne' hne'
      wp_seq
      exact hk hx (by simpa [cmpInt, z0] using he') _ (by ext_env)

theorem count_fine : Fine (emitGo count ⟨1, 0, [], [], [], none⟩).2 := ⟨rfl, by decide⟩

/-- **`htCount` on no context** answers `0` and changes nothing. -/
theorem count_null (cfg : Cfg) (w : World) :
    Ans cfg [.sc .i64 0] w count [.i64] (fun x w' => x = 0 ∧ w' = w) := by
  refine ans_of_wp ?_ count_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [count, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 0] : Env)[0]? = some (.sc .i64 0) := rfl
  refine wp_at (by slot) ?_
  refine wp_withTable (x := 0) (c := 0) (by slot) (fun h => absurd rfl h) ?_ (fun h => absurd rfl h)
  intro _ Γ' hext
  apply wp_iconst64
  exact ⟨_, get_push_size, by decide, rfl⟩

/-- **`htCount`** answers how many entries the table holds, or `0` before a
    handle has been made, and changes nothing: as the model does. -/
theorem count_spec (cfg : Cfg) (w : World) (s : UInt64) (T : Ht) (h : Rep w s T) :
    Ans cfg [.sc .i64 s] w count [.i64] (fun x w' =>
      x = (if T.hasTable then UInt64.ofNat T.entries.length else 0) ∧ w' = w) := by
  refine ans_of_wp ?_ count_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [count, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 s] : Env)[0]? = some (.sc .i64 s) := rfl
  refine wp_at (by slot) ?_
  refine wp_withTable (c := UInt64.ofNat T.handles.length) (by slot) (fun _ => h.created) ?_ ?_
  · intro hz Γ' hext
    have hl : T.handles.length = 0 := by
      rcases hz with hz | hz
      · exact absurd hz (Blk.ne_zero h.blk)
      · have := congrArg UInt64.toNat hz
        rwa [UInt64.toNat_ofNat', Nat.mod_eq_of_lt h.small.1] at this
    apply wp_iconst64
    exact ⟨_, get_push_size, by simp [h.hasTable, hl]; decide, rfl⟩
  · intro _ hc Γ' hext
    have hl : T.handles.length ≠ 0 := fun e => hc (by rw [e]; rfl)
    refine wp_load64 (by slot) h.count ?_
    exact ⟨_, get_push_size, by simp [h.hasTable, hl], rfl⟩

theorem Rep.grows {w w' : World} {s : UInt64} {T : Ht} (h : Rep w s T) (g : Grows w w') : Rep w' s T :=
  ⟨g.blk _ _ h.blk, g.calm h.calm, g.fld h.created, g.fld h.count, h.handles, h.small⟩

/-- A store into the block past its two counts leaves the table it holds. -/
theorem Rep.setFld {w : World} {s x : UInt64} {T : Ht} {j : Nat} {m : Mem} (h : Rep w s T) (hj : j + 8 ≤ SIZE)
    (h16 : 16 ≤ j) (hm : w.mem.store (s + UInt64.ofNat j) 8 x = some m) :
    Rep (stored w (s + UInt64.ofNat j) x m) s T :=
  ⟨Blk.keep h.blk hm, h.calm.stored hm,
    (fld_kept h.blk (by decide) hj (Or.inl (by simp only [CREATED]; omega)) hm).trans h.created,
    (fld_kept h.blk (by decide) hj (Or.inl (by simp only [COUNT]; omega)) hm).trans h.count, h.handles, h.small⟩

/-- A store outside pinned memory leaves the table. -/
theorem Rep.storeOut {w : World} {s p x : UInt64} {T : Ht} {m : Mem} {r : Region} {off : Nat} (h : Rep w s T)
    (hd : decodeAddr p = some (r, off)) (hr : r ≠ .pinned) (hm : w.mem.store p 8 x = some m) :
    Rep (stored w p x m) s T := by
  obtain ⟨o, -, hdec⟩ := Blk.decode h.blk
  exact ⟨Blk.keep h.blk hm, h.calm.stored hm,
    (load_store_clear hm hd (hdec CREATED (by decide)) (Or.inl (Ne.symm hr))).trans h.created,
    (load_store_clear hm hd (hdec COUNT (by decide)) (Or.inl (Ne.symm hr))).trans h.count, h.handles, h.small⟩

/-- A fresh block of the context's size holds the empty table. -/
theorem Rep.fresh {w : World} {s : UInt64} (hc : Calm w) (hb : Blk w s SIZE)
    (hz : ∀ j, j + 8 ≤ SIZE → fld w s j = some 0) : Rep w s {} :=
  ⟨hb, hc, hz CREATED (by decide), hz COUNT (by decide), rfl, by decide⟩

theorem init_fine : Fine (emitGo init ⟨1, 0, [], [], [], none⟩).2 := ⟨rfl, by decide⟩

/-- A store outside pinned memory leaves a context and its arrays. -/
theorem Owned.storeOut {w : World} {s e i a p x : UInt64} {m : Mem} {r : Region} {off : Nat} (h : Owned w s e i a)
    (hd : decodeAddr p = some (r, off)) (hr : r ≠ .pinned) (hm : w.mem.store p 8 x = some m) :
    Owned (stored w p x m) s e i a := by
  obtain ⟨o, -, hdec⟩ := Blk.decode h.blk
  have k : ∀ j v, j + 8 ≤ SIZE → fld w s j = some v → fld (stored w p x m) s j = some v := fun j v hj hv =>
    (load_store_clear hm hd (hdec j (by omega)) (Or.inl (Ne.symm hr))).trans hv
  exact ⟨Blk.keep h.blk hm, k _ _ (by decide) h.entries, k _ _ (by decide) h.index, k _ _ (by decide) h.arena,
    ⟨h.own.1.stored hm, h.own.2.1.stored hm, h.own.2.2.stored hm⟩, h.apart⟩

/-- No worker runs in `w'` and region `r` is as large as in `w`: a slot of `r`
    with room in `w` can be written in `w'`. -/
def Open (w : World) (r : Region) (w' : World) : Prop :=
  w'.mem.frozen = false ∧ (w'.mem.region r).size = (w.mem.region r).size

theorem Open.freed {w w' w'' : World} {r : Region} {x : UInt64} (h : Open w r w') (f : Freed w' w'' x)
    (hr : r ≠ .pinned) : Open w r w'' :=
  ⟨f.frozen.trans h.1, (f.size r hr).trans h.2⟩

theorem open_of_stays {w w' : World} {r : Region} (h : Stays w w') (hc : Calm w) (hr : r ≠ .pinned) : Open w r w' :=
  ⟨(h.calm hc).1, h.size r hr⟩

theorem cleanup_fine : Fine (emitGo cleanup ⟨1, 0, [], [], [], none⟩).2 := ⟨rfl, by decide⟩

/-- **`htCleanup`** frees the context its slot holds, with its arrays, and
    leaves null in the slot, and outside the heap changes the slot alone. -/
theorem cleanup_spec (cfg : Cfg) (w : World) (slot s : UInt64) (r : Region) (off : Nat) (hc : Calm w)
    (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned) (hl : w.mem.load slot 8 = some s)
    (hs : s = 0 ∨ ∃ e i a, Owned w s e i a) :
    Does cfg [.sc .i64 slot] w cleanup [.i64] (fun w' =>
      w'.mem.load slot 8 = some 0 ∧ Outside w w' (Misses slot)) := by
  refine does_of_wp ?_ cleanup_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [cleanup, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 slot] : Env)[0]? = some (.sc .i64 slot) := rfl
  have hroom := (load_some hd hl).1
  refine wp_load64 (by slot) hl ?_
  apply wp_iconst64
  refine wp_when (fun w' => w'.mem.frozen = false ∧ (w'.mem.region r).size = (w.mem.region r).size ∧
      Outside w w' fun _ _ => True)
    (x := s) (y := c64 0) (by slot) (by slot) ?_ (fun _ => ⟨hc.1, rfl, Outside.refl _ _⟩) ?_
  · intro hne
    obtain ⟨e, i, a, ho⟩ := hs.resolve_left (by simpa [cmpInt, show c64 0 = 0 from by decide] using hne)
    obtain ⟨hes, his, has, hie, hae, hai⟩ := ho.apart
    obtain ⟨oe, oi, oa⟩ := ho.own
    wp_seq
    -- the entries
    refine wp_at (by slot) ?_
    refine wp_load64 (by slot) ho.entries ?_
    refine wp_free (n := 0) (by slot) (oe.imp id fun ⟨n, hb⟩ => hb.mono (Nat.zero_le _)) ?_
    intro w1 f1
    have b1 := f1.blk _ _ hes.symm ho.blk
    have hi1 := f1.fld s SIZE INDEX i hes.symm ho.blk (by decide) ho.index
    have ha1 := f1.fld s SIZE ARENA a hes.symm ho.blk (by decide) ho.arena
    have oi1 := oi.freed f1 hie
    have oa1 := oa.freed f1 hae
    -- the index
    refine wp_at (by slot) ?_
    refine wp_load64 (by slot) hi1 ?_
    refine wp_free (n := 0) (by slot) (oi1.imp id fun ⟨n, hb⟩ => hb.mono (Nat.zero_le _)) ?_
    intro w2 f2
    have b2 := f2.blk _ _ his.symm b1
    have ha2 := f2.fld s SIZE ARENA a his.symm b1 (by decide) ha1
    have oa2 := oa1.freed f2 hai
    -- the arena
    refine wp_at (by slot) ?_
    refine wp_load64 (by slot) ha2 ?_
    refine wp_free (n := 0) (by slot) (oa2.imp id fun ⟨n, hb⟩ => hb.mono (Nat.zero_le _)) ?_
    intro w3 f3
    have b3 := f3.blk _ _ has.symm b2
    -- the context
    refine wp_free (by slot) (Or.inr b3) ?_
    intro w4 f4
    exact ⟨by ext_env, by rw [f4.frozen, f3.frozen, f2.frozen, f1.frozen]; exact hc.1,
      by rw [f4.size r hr, f3.size r hr, f2.size r hr, f1.size r hr],
      (((f1.outside _).trans (f2.outside _)).trans (f3.outside _)).trans (f4.outside _)⟩
  · intro Γ' w' hext ⟨hz, hsz, hO⟩
    obtain ⟨m, hm⟩ := store_room (x := c64 0) hd hr (by rw [hsz]; exact hroom) hz
    have hW := wrote_stored hm
    refine wp_storeI64 (by slot) (by slot) hm ?_
    refine ⟨?_, (hO.mono fun _ _ _ => trivial).trans hW.outsideBut⟩
    rw [show c64 0 = 0 from by decide] at hm
    exact load_store8 hm

theorem Tab.ent {w : World} {s : UInt64} {T : Ht} {L : Lay} (h : Tab w s T L) (i : Nat) (hi : i < T.entries.length) :
    EntAt w L i T.entries[i] :=
  ⟨_, _, h.ento i hi (h.olen ▸ hi)⟩

theorem Tab.be {w : World} {s : UInt64} {T : Ht} {L : Lay} (h : Tab w s T L) : Blk w L.e (L.ecap * ENTRY) := h.we.blk
theorem Tab.bi {w : World} {s : UInt64} {T : Ht} {L : Lay} (h : Tab w s T L) : Blk w L.ix (L.icap * 8) := h.wi.blk
theorem Tab.ba {w : World} {s : UInt64} {T : Ht} {L : Lay} (h : Tab w s T L) : Blk w L.ar L.acap := h.wa.blk

/-- The context block and its arrays, with their sizes. -/
def blocks (s : UInt64) (L : Lay) : List (UInt64 × Nat) :=
  [(s, SIZE), (L.e, L.ecap * ENTRY), (L.ix, L.icap * 8), (L.ar, L.acap)]

/-- **`n` bytes at `p` are clear of the table**: outside pinned memory, or
    apart from the context block and each of its arrays. -/
def Clear (s : UInt64) (L : Lay) (p : UInt64) (n : Nat) : Prop :=
  ∀ y k, (y, k) ∈ blocks s L → ∀ r off oy, decodeAddr p = some (r, off) → decodeAddr y = some (.pinned, oy) →
    r ≠ .pinned ∨ off + n ≤ oy ∨ oy + k ≤ off

theorem size_toList (b : ByteArray) : b.data.toList.length = b.size := Array.length_toList

/-- A write that leaves the header's fields, off the entries in use and the
    arena in use, leaves the table. -/
theorem Tab.wroteCore {w w' : World} {s p : UInt64} {T : Ht} {L : Lay} {n : Nat} (h : Tab w s T L)
    (hw : Wrote w w' p n)
    (hf : ∀ j, j + 8 ≤ SIZE → (j + 8 ≤ SCRATCH ∨ SCRATCH + 8 ≤ j) → fld w' s j = fld w s j)
    (ce : Off L.e (T.entries.length * ENTRY) p n) (ca : Off L.ar L.used p n) : Tab w' s T L := by
  have hr := h.room
  have be : Blk w L.e (T.entries.length * ENTRY) := Blk.mono h.be (Nat.mul_le_mul_right _ hr.1)
  have ba : Blk w L.ar L.used := Blk.mono h.ba hr.2
  refine {
    rep := ⟨hw.stays.blk _ _ h.rep.blk, hw.stays.calm h.rep.calm, (hf _ (by decide) (by decide)).trans h.rep.created,
      (hf _ (by decide) (by decide)).trans h.rep.count, h.rep.handles, h.rep.small⟩
    fe := (hf _ (by decide) (by decide)).trans h.fe
    fecap := (hf _ (by decide) (by decide)).trans h.fecap
    fix := (hf _ (by decide) (by decide)).trans h.fix
    ficap := (hf _ (by decide) (by decide)).trans h.ficap
    far := (hf _ (by decide) (by decide)).trans h.far
    fused := (hf _ (by decide) (by decide)).trans h.fused
    facap := (hf _ (by decide) (by decide)).trans h.facap
    we := hw.whole h.we
    wi := hw.whole h.wi
    wa := hw.whole h.wa
    room := h.room
    pos := h.pos
    apart := h.apart
    olen := h.olen
    sep := h.sep
    ento := fun i hi ho => ?_ }
  obtain ⟨h0, h8, h16, h24, hk, hv, bk, bv⟩ := h.ento i hi ho
  have he : ∀ j, j + 8 ≤ ENTRY → fld w' L.e (i * ENTRY + j) = fld w L.e (i * ENTRY + j) := fun j hj =>
    hw.fld_blk be (by simp only [ENTRY] at hj ⊢; omega) ce
  exact ⟨(he 0 (by decide)).trans h0, (he 8 (by decide)).trans h8, (he 16 (by decide)).trans h16,
    (he 24 (by decide)).trans h24, hk, hv,
    hw.bytes_blk ba (by rw [size_toList]; omega) bk ca, hw.bytes_blk ba (by rw [size_toList]; omega) bv ca⟩

/-- A write off the context block, the entries in use and the arena in use
    leaves the table. -/
theorem Tab.wroteOff {w w' : World} {s p : UInt64} {T : Ht} {L : Lay} {n : Nat} (h : Tab w s T L)
    (hw : Wrote w w' p n) (cs : Off s SIZE p n) (ce : Off L.e (T.entries.length * ENTRY) p n)
    (ca : Off L.ar L.used p n) : Tab w' s T L :=
  h.wroteCore hw (fun _ hj _ => hw.fld_blk h.rep.blk hj cs) ce ca

/-- A store into the header's scratch field leaves the table. -/
theorem Tab.scratch {w : World} {s x : UInt64} {T : Ht} {L : Lay} {m : Mem} (h : Tab w s T L)
    (hm : w.mem.store (s + UInt64.ofNat SCRATCH) 8 x = some m) :
    Tab (stored w (s + UInt64.ofNat SCRATCH) x m) s T L :=
  h.wroteCore (wrote_stored hm) (fun j hj hd => fld_kept h.rep.blk hj (by decide) hd hm)
    (Off.prefix (h.apart.1.clear h.rep.blk (by decide)) (Nat.mul_le_mul_right _ h.room.1))
    (Off.prefix (h.apart.2.2.1.clear h.rep.blk (by decide)) h.room.2)

/-- A write clear of the table leaves it. -/
theorem Tab.wrote {w w' : World} {s p : UInt64} {T : Ht} {L : Lay} {n : Nat} (h : Tab w s T L)
    (hw : Wrote w w' p n) (hc : Clear s L p n) : Tab w' s T L :=
  h.wroteOff hw (hc s SIZE (by simp [blocks]))
    (Off.prefix (hc L.e (L.ecap * ENTRY) (by simp [blocks])) (Nat.mul_le_mul_right _ h.room.1))
    (Off.prefix (hc L.ar L.acap (by simp [blocks])) h.room.2)

theorem getEntry_fine : Fine (emitGo getEntry ⟨4, 0, [], [], [], none⟩).2 := ⟨rfl, by decide⟩

/-- What `htGetEntry` answers and leaves: `-1` and the world where there is
    no table or no such entry; else the key's length, and memory with the key
    copied to `kout` and then the value to `vout`, two writes of those runs,
    the table as it was. -/
def GetQ (w : World) (s : UInt64) (T : Ht) (L : Lay) (index kout vout : UInt64) (x : UInt64) (w' : World) : Prop :=
  if T.hasTable then
    match T.entries[index.toNat]? with
    | none => x = c64 (-1) ∧ w' = w
    | some (k, v) => x = UInt64.ofNat k.size ∧ Tab w' s T L ∧
        (∃ m1, copyIn w.mem kout k = some m1 ∧ copyIn m1 vout v = some w'.mem) ∧
        ∃ w1, Wrote w w1 kout k.size ∧ Wrote w1 w' vout v.size
  else x = c64 (-1) ∧ w' = w

/-- **`htGetEntry`**, handed room for the entry's key and value clear of the
    table, answers what the model answers and leaves the memory the model
    leaves. -/
theorem getEntry_spec (cfg : Cfg) (w : World) (s : UInt64) (T : Ht) (L : Lay) (h : Tab w s T L)
    (index kout vout : UInt64) (hi : index.toNat < 2 ^ 32)
    (hroom : ∀ k v, T.entries[index.toNat]? = some (k, v) →
      Room w kout k.size ∧ Room w vout v.size ∧ Clear s L kout k.size ∧ Clear s L vout v.size) :
    Ans cfg [.sc .i64 s, .sc .i32 index, .sc .i64 kout, .sc .i64 vout] w getEntry [.i64, .i32, .i64, .i64]
      (GetQ w s T L index kout vout) := by
  refine ans_of_wp ?_ getEntry_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [getEntry, entryOf, keyAt, valAt, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 s, .sc .i32 index, .sc .i64 kout, .sc .i64 vout] : Env)[0]? = some (.sc .i64 s) := rfl
  have h1 : (#[V.sc .i64 s, .sc .i32 index, .sc .i64 kout, .sc .i64 vout] : Env)[0 + 1]? = some (.sc .i32 index) := rfl
  have h2 : (#[V.sc .i64 s, .sc .i32 index, .sc .i64 kout, .sc .i64 vout] : Env)[0 + 1 + 1]? = some (.sc .i64 kout) := rfl
  have h3 : (#[V.sc .i64 s, .sc .i32 index, .sc .i64 kout, .sc .i64 vout] : Env)[0 + 1 + 1 + 1]? = some (.sc .i64 vout) := rfl
  refine wp_withTable (c := UInt64.ofNat T.handles.length) (by slot) (fun _ => h.rep.created) ?_ ?_
  · intro hz Γ' hext
    have hl : T.handles.length = 0 := by
      rcases hz with hz | hz
      · exact absurd hz (Blk.ne_zero h.rep.blk)
      · have := congrArg UInt64.toNat hz
        rwa [UInt64.toNat_ofNat', Nat.mod_eq_of_lt h.rep.small.1] at this
    apply wp_iconst64
    exact ⟨_, get_push_size, by simp [GetQ, h.rep.hasTable, hl]⟩
  · intro _ hc Γ' hext
    have hl : T.handles.length ≠ 0 := fun e => hc (by rw [e]; rfl)
    have hidx : index &&& 0xFFFFFFFF = index := by
      apply UInt64.toNat_inj.mp
      rw [UInt64.toNat_and, show (0xFFFFFFFF : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
        Nat.mod_eq_of_lt hi]
    have hcnt : (UInt64.ofNat T.entries.length).toNat = T.entries.length := by
      rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt h.rep.small.2]
    wp_seq
    refine wp_uextend64 (x := index) (by slot) ?_
    rw [hidx]
    refine wp_at (x := s) (by slot) ?_
    refine wp_load64 (by slot) h.rep.count ?_
    refine wp_failIf (x := index) (y := UInt64.ofNat T.entries.length) (by slot) (by slot) ?_ ?_
    · intro hge
      have hge' : T.entries.length ≤ index.toNat := by
        have := UInt64.le_iff_toNat_le.mp (by simpa [cmpInt] using hge)
        rwa [hcnt] at this
      simp [GetQ, h.rep.hasTable, hl, List.getElem?_eq_none hge']
    · intro hlt Γ2 hext2
      have hlt' : index.toNat < T.entries.length := by
        have := UInt64.lt_iff_toNat_lt.mp (by simpa [cmpInt] using hlt)
        rwa [hcnt] at this
      have hent := h.ent _ hlt'
      generalize hkv : T.entries[index.toNat] = kv at hent
      obtain ⟨k, v⟩ := kv
      obtain ⟨ko, vo, f0, f8, f16, f24, hko, hvo, bk, bv⟩ := hent
      have hget : T.entries[index.toNat]? = some (k, v) := by rw [List.getElem?_eq_getElem hlt', hkv]
      obtain ⟨rk, rv, ck, cv⟩ := hroom k v hget
      have hr := h.room
      have hko2 : ko + k.size ≤ L.used := hko
      have hvo2 : vo + v.size ≤ L.used := hvo
      have hacap : L.acap ≤ 2 ^ 36 := by
        obtain ⟨o, m, -, hn, -, -, h36, -⟩ := h.ba
        omega
      have hks : (UInt64.ofNat k.size).toNat = k.size := by
        rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
      have hvs : (UInt64.ofNat v.size).toNat = v.size := by
        rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
      have eA : L.e + (index + c64 1 + c64 (-1)) * c64 (ENTRY : Nat) = L.e + UInt64.ofNat (index.toNat * ENTRY) := by
        rw [UInt64.add_assoc, show c64 1 + c64 (-1) = 0 from by decide, UInt64.add_zero,
          show c64 (ENTRY : Nat) = UInt64.ofNat ENTRY from by decide, UInt64.ofNat_mul, UInt64.ofNat_toNat]
      have e8 : ∀ (j : Nat) (z : Int), c64 z = UInt64.ofNat j → L.e + UInt64.ofNat (index.toNat * ENTRY) + c64 z =
          L.e + UInt64.ofNat (index.toNat * ENTRY + j) := fun j z hz => by
        rw [UInt64.add_assoc, hz, ← UInt64.ofNat_add]
      wp_seq
      refine wp_iaddImm (x := index) (by slot) ?_
      refine wp_at (x := s) (by slot) ?_
      refine wp_load64 (by slot) h.fe ?_
      refine wp_iaddImm (x := index + c64 1) (by slot) ?_
      apply wp_iconst64
      refine wp_imul (by slot) (by slot) ?_
      refine wp_iadd (by slot) (by slot) ?_
      rw [eA]
      refine wp_iaddImm (x := L.e + UInt64.ofNat (index.toNat * ENTRY)) (by slot) ?_
      refine wp_load64 (by slot) (by rw [e8 8 8 (by decide)]; exact f8) ?_
      refine wp_at (x := s) (by slot) ?_
      refine wp_load64 (by slot) h.far ?_
      refine wp_load64 (by slot) f0 ?_
      refine wp_iadd (by slot) (by slot) ?_
      refine wp_memcpy (bs := k.data.toList) (by slot) (by slot) (by slot) bk (by rw [size_toList, hks])
        (by rw [hks]; exact rk)
        (by rw [hks]; exact apart_blk rk h.ba (by omega) (ck _ _ (by simp [blocks]))) ?_
      intro w1 hw1 hb1 hc1
      rw [hks] at hw1
      have t1 := h.wrote hw1 ck
      have hent1 := t1.ent _ hlt'
      rw [hkv] at hent1
      obtain ⟨ko', vo', g0, g8, g16, g24, hko', hvo', bk', bv'⟩ := hent1
      have hvo3 : vo' + v.size ≤ L.used := hvo'
      refine wp_at (x := s) (by slot) ?_
      refine wp_load64 (by slot) t1.far ?_
      refine wp_iaddImm (x := L.e + UInt64.ofNat (index.toNat * ENTRY)) (by slot) ?_
      refine wp_load64 (by slot) (by rw [e8 16 16 (by decide)]; exact g16) ?_
      refine wp_iadd (by slot) (by slot) ?_
      refine wp_iaddImm (x := L.e + UInt64.ofNat (index.toNat * ENTRY)) (by slot) ?_
      refine wp_load64 (by slot) (by rw [e8 24 24 (by decide)]; exact g24) ?_
      refine wp_memcpy (bs := v.data.toList) (by slot) (by slot) (by slot) bv' (by rw [size_toList, hvs])
        (by rw [hvs]; exact hw1.room rv)
        (by rw [hvs]; exact apart_blk (hw1.room rv) t1.ba (by omega) (cv _ _ (by simp [blocks]))) ?_
      intro w2 hw2 hb2 hc2
      rw [hvs] at hw2
      refine ⟨_, by slot, ?_⟩
      have ek : ∀ b : ByteArray, ByteArray.mk b.data.toList.toArray = b := fun b => by simp
      rw [ek] at hc1 hc2
      unfold GetQ
      rw [if_pos (by simp [h.rep.hasTable, hl]), hget]
      exact ⟨rfl, t1.wrote hw2 cv, ⟨_, hc1, hc2⟩, w1, hw1, hw2⟩

section fns
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- **`hash`** of bytes that can be read answers their hash and changes
    nothing. -/
theorem wp_hash {p n : Slot .i64} {a : UInt64} {bs : List UInt8} {Q : Slot .i64 → Env → World → Prop}
    (hp : Γ[p]? = some (.sc .i64 a)) (hn : Γ[n]? = some (.sc .i64 (UInt64.ofNat bs.length)))
    (hb : Bytes w a bs)
    (hk : ∀ Γ' r, Ext Γ Γ' → Γ'[r]? = some (.sc .i64 (H bs)) → Q r Γ' w) :
    wp cfg d J (LibHt.hash p n) Q Γ w := by
  have hlen := hb.len_le
  have hwn : (UInt64.ofNat (bs.length / 8)).toNat = bs.length / 8 := by
    rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  have htn : (UInt64.ofNat (bs.length % 8)).toNat = bs.length % 8 := by
    rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  unfold LibHt.hash
  wp_seq
  apply wp_iconst64
  refine wp_ushrImm (x := UInt64.ofNat bs.length) (by slot) ?_
  rw [shr3 (by omega)]
  refine wp_imul (x := UInt64.ofNat bs.length) (y := c64 MIX) (by slot) (by slot) ?_
  refine wp_forLoopAcc (a0 := UInt64.ofNat bs.length * c64 MIX) (fun i h w' => w = w' ∧ h = hashW bs i) (by slot)
    (by slot) ⟨rfl, rfl⟩ ?_ ?_
  · intro J' Γ' wi i acc si sa hE hi hinv hsi hsa
    obtain ⟨hw, ha⟩ := hinv
    subst hw; subst ha
    rw [hwn] at hi
    wp_seq
    refine wp_ishlImm hsi ?_
    rw [shl3 (by omega)]
    refine wp_iadd (x := a) (by slot) (by slot) ?_
    refine wp_load64 (by slot) (hb.load (j := 8 * i) (k := 8) (by omega) (by decide)) ?_
    refine wp_bxor (by slot) (by slot) ?_
    refine wp_imul (y := c64 MIX) (by slot) (by slot) ?_
    exact ⟨by ext_env, _, get_push_size, rfl, rfl⟩
  · intro Γ' w' r acc hE hr hinv
    obtain ⟨hw, ha⟩ := hinv
    subst hw; subst ha
    rw [hwn] at hr
    refine wp_ishlImm (x := UInt64.ofNat (bs.length / 8)) (by slot) ?_
    rw [shl3 (by omega)]
    refine wp_isub (x := UInt64.ofNat bs.length) (by slot) (by slot) ?_
    rw [ofNat_sub_le (by omega) (by omega), show bs.length - 8 * (bs.length / 8) = bs.length % 8 from by omega]
    apply wp_iconst64
    refine wp_forLoopAcc (a0 := c64 0) (fun j t w' => w = w' ∧ t = hashT bs (8 * (bs.length / 8)) j) (by slot)
      (by slot) ⟨rfl, rfl⟩ ?_ ?_
    · intro J' Γ2 wi j t sj st hE2 hj hinv hsj hst
      obtain ⟨hw, ht⟩ := hinv
      subst hw; subst ht
      rw [htn] at hj
      wp_seq
      refine wp_iadd (x := UInt64.ofNat (8 * (bs.length / 8))) (y := UInt64.ofNat j) (by slot) (by slot) ?_
      rw [← UInt64.ofNat_add]
      refine wp_iadd (x := a) (by slot) (by slot) ?_
      refine wp_uload8 (by slot) (hb.load (j := 8 * (bs.length / 8) + j) (k := 1) (by omega) (by decide)) ?_
      refine wp_ishlImm (x := UInt64.ofNat j) (by slot) ?_
      refine wp_ishl (by slot) (by slot) ?_
      refine wp_bor (by slot) (by slot) ?_
      exact ⟨by ext_env, _, get_push_size, rfl, rfl⟩
    · intro Γ2 w2 r2 t hE2 hr2 hinv
      obtain ⟨hw, ht⟩ := hinv
      subst hw; subst ht
      rw [htn] at hr2
      refine wp_bxor (by slot) (by slot) ?_
      refine wp_imul (y := c64 MIX) (by slot) (by slot) ?_
      refine wp_ushrImm (by slot) ?_
      refine wp_bxor (by slot) (by slot) ?_
      apply wp_iconst64
      refine wp_imul (by slot) (by slot) ?_
      refine wp_ushrImm (by slot) ?_
      refine wp_bxor (by slot) (by slot) ?_
      have e2 : Ext Γ Γ2 := (Ext.trans (by ext_env) hE).trans (Ext.trans (by ext_env) hE2)
      exact hk _ _ (by ext_env) get_push_size

/-- What differs in the first `i` words of `as` and `bs`, or-ed together. -/
def diffW (as bs : List UInt8) : Nat → UInt64
  | 0 => c64 0
  | i + 1 => diffW as bs i ||| (leN as (8 * i) 8 ^^^ leN bs (8 * i) 8)

/-- …and in the `j` bytes from `base`, on top of `d0`. -/
def diffT (as bs : List UInt8) (base : Nat) (d0 : UInt64) : Nat → UInt64
  | 0 => d0
  | j + 1 => diffT as bs base d0 j ||| (leN as (base + j) 1 ^^^ leN bs (base + j) 1)

theorem diffW_zero (as bs : List UInt8) : ∀ i, diffW as bs i = 0 ↔ ∀ t, t < 8 * i → as.getD t 0 = bs.getD t 0
  | 0 => by simp [diffW]; decide
  | i + 1 => by
    rw [diffW, UInt64.or_eq_zero_iff, UInt64.xor_eq_zero_iff, diffW_zero as bs i, leN_eq_iff (by decide)]
    constructor
    · rintro ⟨h1, h2⟩ t ht
      by_cases c : t < 8 * i
      · exact h1 t c
      · have := h2 (t - 8 * i) (by omega)
        rwa [show 8 * i + (t - 8 * i) = t from by omega] at this
    · intro h
      exact ⟨fun t ht => h t (by omega), fun t ht => h _ (by omega)⟩

theorem diffT_zero (as bs : List UInt8) (base : Nat) (d0 : UInt64) :
    ∀ j, diffT as bs base d0 j = 0 ↔ d0 = 0 ∧ ∀ t, t < j → as.getD (base + t) 0 = bs.getD (base + t) 0
  | 0 => by simp [diffT]
  | j + 1 => by
    rw [diffT, UInt64.or_eq_zero_iff, UInt64.xor_eq_zero_iff, diffT_zero as bs base d0 j, leN_eq_iff (by decide)]
    constructor
    · rintro ⟨⟨h0, h1⟩, h2⟩
      refine ⟨h0, fun t ht => ?_⟩
      by_cases c : t < j
      · exact h1 t c
      · have := h2 0 (by decide)
        rwa [Nat.add_zero, show j = t from by omega] at this
    · rintro ⟨h0, h⟩
      exact ⟨⟨h0, fun t ht => h t (by omega)⟩, fun t ht => by
        rw [show t = 0 from by omega, Nat.add_zero]; exact h j (by omega)⟩

/-- **`bytesEq`** of two runs of one length answers whether they are the
    same bytes, and changes nothing. -/
theorem wp_bytesEq {a b n : Slot .i64} {pa pb nv : UInt64} {as bs : List UInt8}
    {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 pa)) (hb : Γ[b]? = some (.sc .i64 pb)) (hn : Γ[n]? = some (.sc .i64 nv))
    (hnv : nv = UInt64.ofNat as.length)
    (hA : Bytes w pa as) (hB : Bytes w pb bs) (hl : bs.length = as.length)
    (hk : ∀ Γ' r, Ext Γ Γ' → Γ'[r]? = some (.sc .i64 (if as = bs then 1 else 0)) → Q r Γ' w) :
    wp cfg d J (bytesEq a b n) Q Γ w := by
  subst hnv
  have hlen := hA.len_le
  have hwn : (UInt64.ofNat (as.length / 8)).toNat = as.length / 8 := by
    rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  have htn : (UInt64.ofNat (as.length % 8)).toNat = as.length % 8 := by
    rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  unfold bytesEq
  wp_seq
  apply wp_iconst64
  refine wp_ushrImm (x := UInt64.ofNat as.length) (by slot) ?_
  rw [shr3 (by omega)]
  refine wp_forLoopAcc (a0 := c64 0) (fun i h w' => w = w' ∧ h = diffW as bs i) (by slot) (by slot) ⟨rfl, rfl⟩ ?_ ?_
  · intro J' Γ' wi i acc si sa hE hi hinv hsi hsa
    obtain ⟨hw, hacc⟩ := hinv
    subst hw; subst hacc
    rw [hwn] at hi
    wp_seq
    refine wp_ishlImm hsi ?_
    rw [shl3 (by omega)]
    refine wp_iadd (x := pa) (by slot) (by slot) ?_
    refine wp_load64 (by slot) (hA.load (j := 8 * i) (k := 8) (by omega) (by decide)) ?_
    refine wp_iadd (x := pb) (by slot) (by slot) ?_
    refine wp_load64 (by slot) (hB.load (j := 8 * i) (k := 8) (by omega) (by decide)) ?_
    refine wp_bxor (by slot) (by slot) ?_
    refine wp_bor (by slot) (by slot) ?_
    exact ⟨by ext_env, _, get_push_size, rfl, rfl⟩
  · intro Γ' w' r acc hE hr hinv
    obtain ⟨hw, hacc⟩ := hinv
    subst hw; subst hacc
    rw [hwn] at hr
    refine wp_ishlImm (x := UInt64.ofNat (as.length / 8)) (by slot) ?_
    rw [shl3 (by omega)]
    refine wp_isub (x := UInt64.ofNat as.length) (by slot) (by slot) ?_
    rw [ofNat_sub_le (by omega) (by omega), show as.length - 8 * (as.length / 8) = as.length % 8 from by omega]
    refine wp_forLoopAcc (a0 := diffW as bs (as.length / 8))
      (fun j t w' => w = w' ∧ t = diffT as bs (8 * (as.length / 8)) (diffW as bs (as.length / 8)) j) (by slot)
      (by slot) ⟨rfl, rfl⟩ ?_ ?_
    · intro J' Γ2 wi j t sj st hE2 hj hinv hsj hst
      obtain ⟨hw, ht⟩ := hinv
      subst hw; subst ht
      rw [htn] at hj
      wp_seq
      refine wp_iadd (x := UInt64.ofNat (8 * (as.length / 8))) (y := UInt64.ofNat j) (by slot) (by slot) ?_
      rw [← UInt64.ofNat_add]
      refine wp_iadd (x := pa) (by slot) (by slot) ?_
      refine wp_uload8 (by slot) (hA.load (j := 8 * (as.length / 8) + j) (k := 1) (by omega) (by decide)) ?_
      refine wp_iadd (x := pb) (by slot) (by slot) ?_
      refine wp_uload8 (by slot) (hB.load (j := 8 * (as.length / 8) + j) (k := 1) (by omega) (by decide)) ?_
      refine wp_bxor (by slot) (by slot) ?_
      refine wp_bor (by slot) (by slot) ?_
      exact ⟨by ext_env, _, get_push_size, rfl, rfl⟩
    · intro Γ2 w2 r2 t hE2 hr2 hinv
      obtain ⟨hw, ht⟩ := hinv
      subst hw; subst ht
      rw [htn] at hr2
      refine wp_icmp (y := c64 0) (by slot) (by slot) ?_
      refine wp_uextend8 (by slot) ?_
      have e2 : Ext Γ Γ2 := (Ext.trans (by ext_env) hE).trans (Ext.trans (by ext_env) hE2)
      refine hk _ _ (by ext_env) ?_
      have key : (diffT as bs (8 * (as.length / 8)) (diffW as bs (as.length / 8)) (as.length % 8) = 0) ↔ as = bs := by
        rw [diffT_zero, diffW_zero, list_eq_iff_getD hl]
        constructor
        · rintro ⟨h1, h2⟩ t ht
          by_cases c : t < 8 * (as.length / 8)
          · exact h1 t c
          · have := h2 (t - 8 * (as.length / 8)) (by omega)
            rwa [show 8 * (as.length / 8) + (t - 8 * (as.length / 8)) = t from by omega] at this
        · intro h
          exact ⟨fun t ht => h t (by omega), fun t ht => h _ (by omega)⟩
      rw [get_push_size]
      congr 2
      by_cases c : as = bs
      · have hz := key.mpr c
        rw [if_pos c]
        simp [cmpInt, hz, show c64 0 = 0 from by decide]
      · have hz := mt key.mp c
        rw [if_neg c]
        simp [cmpInt, hz, show c64 0 = 0 from by decide]

end fns

theorem slotVal_toNat (h : UInt64) {p : Nat} (hp : p + 1 < 2 ^ 32) :
    (slotVal h p).toNat = h.toNat / 2 ^ 32 * 2 ^ 32 + (p + 1) := by
  have hh := h.toNat_lt
  unfold slotVal
  rw [UInt64.toNat_or, UInt64.toNat_shiftLeft, UInt64.toNat_shiftRight,
    show (c64 32 % 64).toNat % 64 = 32 from by decide, Nat.shiftRight_eq_div_pow, UInt64.toNat_ofNat',
    Nat.mod_eq_of_lt (show p + 1 < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by rw [Nat.shiftLeft_eq]; omega),
    ← Nat.shiftLeft_add_eq_or_of_lt hp, Nat.shiftLeft_eq]

theorem slotVal_lo (h : UInt64) {p : Nat} (hp : p + 1 < 2 ^ 32) :
    slotVal h p &&& c64 0xffffffff = UInt64.ofNat (p + 1) := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, slotVal_toNat h hp, show (c64 0xffffffff).toNat = 2 ^ 32 - 1 from by decide,
    Nat.and_two_pow_sub_one_eq_mod, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (show p + 1 < 2 ^ 64 by omega)]
  omega

theorem slotVal_hi (h : UInt64) {p : Nat} (hp : p + 1 < 2 ^ 32) :
    slotVal h p >>> (c64 32 % 64) = h >>> (c64 32 % 64) := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_shiftRight, UInt64.toNat_shiftRight, slotVal_toNat h hp,
    show (c64 32 % 64).toNat % 64 = 32 from by decide, Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]
  omega

theorem slotVal_ne_zero (h : UInt64) {p : Nat} (hp : p + 1 < 2 ^ 32) : slotVal h p ≠ 0 := by
  intro e
  have := congrArg UInt64.toNat e
  rw [slotVal_toNat h hp] at this
  simp at this

theorem maskOf_toNat {icap : Nat} (h0 : 0 < icap) (h1 : icap < 2 ^ 64) : (maskOf icap).toNat = icap - 1 := by
  unfold maskOf
  rw [UInt64.toNat_add, UInt64.toNat_ofNat', Nat.mod_eq_of_lt h1, show (c64 (-1)).toNat = 2 ^ 64 - 1 from by decide]
  omega

theorem and_mask_lt (x : UInt64) {icap : Nat} (h0 : 0 < icap) (h1 : icap < 2 ^ 64) :
    (x &&& maskOf icap).toNat < icap := by
  rw [UInt64.toNat_and, maskOf_toNat h0 h1]
  have := @Nat.and_le_right x.toNat (icap - 1)
  omega

theorem path_lt {icap : Nat} (h : UInt64) (t : Nat) (h0 : 0 < icap) (h1 : icap < 2 ^ 64) :
    (path icap h t).toNat < icap := by
  cases t <;> exact and_mask_lt _ h0 h1

/-- **What a probe needs of a table**: the header fields it reads, the keys
    `ks` of the entries the index covers at their places, every slot empty or
    holding one of them, and each of them reached from its hash's first slot
    before any empty slot. -/
structure Probe (w : World) (s : UInt64) (L : Lay) (ks : List (List UInt8)) : Prop where
  ficap : fld w s ICAP = some (UInt64.ofNat L.icap)
  fix : fld w s INDEX = some L.ix
  fe : fld w s ENTRIES = some L.e
  far : fld w s ARENA = some L.ar
  cap : 0 < L.icap ∧ L.icap ≤ 2 ^ 33
  small : ks.length < 2 ^ 32
  key : ∀ p (hp : p < ks.length), ∃ ko, fld w L.e (p * ENTRY) = some (UInt64.ofNat ko) ∧
    fld w L.e (p * ENTRY + 8) = some (UInt64.ofNat ks[p].length) ∧ Bytes w (L.ar + UInt64.ofNat ko) ks[p]
  slots : ∀ j, j < L.icap → ∃ v, fld w L.ix (8 * j) = some v ∧
    (v = 0 ∨ ∃ p, ∃ hp : p < ks.length, v = slotVal (H ks[p]) p)
  chain : ∀ p (hp : p < ks.length), ∃ t, fld w L.ix (8 * (path L.icap (H ks[p]) t).toNat) =
      some (slotVal (H ks[p]) p) ∧
    ∀ t', t' < t → fld w L.ix (8 * (path L.icap (H ks[p]) t').toNat) ≠ some 0

/-- **What a probe for the key `kb` answers**: the address of a slot of the
    index, and `0` where that slot is the empty one the key would go in, the
    key being none of `ks`, or the position plus one of the entry with that
    key. -/
def Found (w : World) (L : Lay) (ks : List (List UInt8)) (kb : List UInt8) (slotA pos : UInt64) : Prop :=
  ∃ i : UInt64, i.toNat < L.icap ∧ slotA = L.ix + UInt64.ofNat (8 * i.toNat) ∧
    ((pos = 0 ∧ fld w L.ix (8 * i.toNat) = some 0 ∧ kb ∉ ks ∧
        ∃ t, i = path L.icap (H kb) t ∧ ∀ t', t' < t → fld w L.ix (8 * (path L.icap (H kb) t').toNat) ≠ some 0) ∨
      ∃ p, ∃ hp : p < ks.length, pos = UInt64.ofNat (p + 1) ∧ ks[p] = kb)

/-- A slot holds the entry with the key `kb`. -/
def Match (ks : List (List UInt8)) (kb : List UInt8) (v : UInt64) : Prop :=
  ∃ p, ∃ hp : p < ks.length, v = slotVal (H ks[p]) p ∧ ks[p] = kb

theorem slotVal_inj {h h' : UInt64} {p p' : Nat} (hp : p + 1 < 2 ^ 32) (hp' : p' + 1 < 2 ^ 32)
    (e : slotVal h p = slotVal h' p') : p = p' := by
  have := congrArg (· &&& c64 0xffffffff) e
  simp only [slotVal_lo _ hp, slotVal_lo _ hp'] at this
  have := congrArg UInt64.toNat this
  rw [UInt64.toNat_ofNat', UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)] at this
  omega

/-- The loop's test, on the flag the head computed. -/
theorem evalStop {m : Mem} {Γ1 : Env} {sst sz : Slot .i8} {stop : Bool} (h1 : Γ1[sst]? = some (boolV stop))
    (h2 : Γ1[sz]? = some (ofInt .i8 0)) : evalOp m Γ1 (.icmp .ne sst sz) = some (boolV stop) := by
  cases stop <;> simp [evalOp, Sem.get, h1, h2, zipIntCmp, boolV] <;> first | done | rfl | decide

section fns
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

set_option maxHeartbeats 1600000 in
/-- **`find`** over a table a probe can read answers a slot of the index and
    whether the key is there (`Found`), and changes nothing. -/
theorem wp_find {s key len h : Slot .i64} {sv kp : UInt64} {L : Lay} {ks : List (List UInt8)} {kb : List UInt8}
    {Q : Slot .i64 × Slot .i64 → Env → World → Prop}
    (hs : Γ[s]? = some (.sc .i64 sv)) (hkey : Γ[key]? = some (.sc .i64 kp))
    (hlen : Γ[len]? = some (.sc .i64 (UInt64.ofNat kb.length))) (hh : Γ[h]? = some (.sc .i64 (H kb)))
    (hP : Probe w sv L ks) (hB : Bytes w kp kb)
    (hk : ∀ Γ' (ra rp : Slot .i64) slotA pos, Ext Γ Γ' → Γ'[ra]? = some (.sc .i64 slotA) →
      Γ'[rp]? = some (.sc .i64 pos) → Found w L ks kb slotA pos → Q (ra, rp) Γ' w) :
    wp cfg d J (find s key len h) Q Γ w := by
  have hc0 := hP.cap.1
  have hc1 : L.icap < 2 ^ 64 := by have := hP.cap.2; omega
  have hsm := hP.small
  have z0 : c64 0 = 0 := by decide
  unfold find keyAt
  wp_seq
  refine wp_at (x := sv) (by slot) ?_
  refine wp_load64 (by slot) hP.ficap ?_
  refine wp_iaddImm (x := UInt64.ofNat L.icap) (by slot) ?_
  refine wp_at (x := sv) (by slot) ?_
  refine wp_load64 (by slot) hP.fix ?_
  refine wp_at (x := sv) (by slot) ?_
  refine wp_load64 (by slot) hP.fe ?_
  refine wp_ushrImm (x := H kb) (by slot) ?_
  apply wp_iconst64
  apply wp_iconst64
  refine wp_band (x := H kb) (y := UInt64.ofNat L.icap + c64 (-1)) (by slot) (by slot) ?_
  refine wp_wloop1 (x0 := H kb &&& (UInt64.ofNat L.icap + c64 (-1)))
    (fun i w' => w = w' ∧ ∃ t, i = path L.icap (H kb) t ∧ ∀ t', t' < t →
      fld w L.ix (8 * (path L.icap (H kb) t').toNat) ≠ some 0 ∧
      ∀ p, ∀ hp : p < ks.length, fld w L.ix (8 * (path L.icap (H kb) t').toNat) = some (slotVal (H (ks[p]'hp)) p) →
        (ks[p]'hp) ≠ kb)
    (fun i a Γ1 w1 => w = w1 ∧ (∃ t, i = path L.icap (H kb) t ∧ ∀ t', t' < t →
      fld w L.ix (8 * (path L.icap (H kb) t').toNat) ≠ some 0 ∧
      ∀ p, ∀ hp : p < ks.length, fld w L.ix (8 * (path L.icap (H kb) t').toNat) = some (slotVal (H (ks[p]'hp)) p) →
        (ks[p]'hp) ≠ kb) ∧
      ∃ (sst sz : Slot .i8) (ssl ssel : Slot .i64) (v : UInt64) (stop : Bool) (sel : UInt64),
        a = (exitIf .ne sst sz, %[ssl, ssel], ()) ∧ Γ1[sst]? = some (boolV stop) ∧ Γ1[sz]? = some (ofInt .i8 0) ∧
        Γ1[ssl]? = some (.sc .i64 (L.ix + UInt64.ofNat (8 * i.toNat))) ∧ Γ1[ssel]? = some (.sc .i64 sel) ∧
        fld w L.ix (8 * i.toNat) = some v ∧
        (stop = true → Found w L ks kb (L.ix + UInt64.ofNat (8 * i.toNat)) sel) ∧
        (stop = false → v ≠ 0 ∧ ∀ p, ∀ hp : p < ks.length, v = slotVal (H (ks[p]'hp)) p → (ks[p]'hp) ≠ kb))
    (fun vs w' => w = w' ∧ ∃ a p, vs = [.sc .i64 a, .sc .i64 p] ∧ Found w L ks kb a p)
    (by slot) ⟨rfl, 0, rfl, fun t' ht' => absurd ht' (Nat.not_lt_zero _)⟩ ?_ ?_ ?_ ?_ ?_
  · -- the head
    intro J' Γ' wi i si hE hinv hsi
    obtain ⟨hw, t, hi, hprev⟩ := hinv
    subst hw
    have hilt : i.toNat < L.icap := by rw [hi]; exact path_lt _ _ hc0 hc1
    obtain ⟨v, hv, hvcase⟩ := hP.slots i.toNat hilt
    wp_seq
    refine wp_ishlImm hsi ?_
    rw [show i <<< (c64 3 % 64) = UInt64.ofNat (8 * i.toNat) from by
      rw [← shl3 (by have := hP.cap.2; omega), UInt64.ofNat_toNat]]
    refine wp_iadd (x := L.ix) (by slot) (by slot) ?_
    refine wp_load64 (by slot) hv ?_
    refine wp_band (x := v) (y := c64 0xffffffff) (by slot) (by slot) ?_
    refine wp_ushrImm (x := v) (by slot) ?_
    refine wp_ifteV (fun x w' => w = w' ∧ (x ≠ 0 ↔ Match ks kb v)) (x := v >>> (c64 32 % 64))
      (y := H kb >>> (c64 32 % 64)) (by slot) (by slot) ?_ ?_ ?_
    · intro hne
      wp_seq
      refine ⟨by ext_env, c64 0, by slot, by first | rfl | trivial, ?_⟩
      rw [z0]
      refine ⟨fun h => absurd rfl h, ?_⟩
      rintro ⟨p, hp, rfl, hpe⟩
      exfalso
      rw [slotVal_hi (H (ks[p]'hp)) (p := p) (by omega), hpe] at hne
      simp only [cmpInt, bne_self_eq_false] at hne
      exact Bool.noConfusion hne
    · intro heq ne1 hne1
      wp_seq
      refine wp_ifteV (fun x w' => w = w' ∧ (x ≠ 0 ↔ Match ks kb v)) (x := v) (y := c64 0) (by slot) (by slot)
        ?_ ?_ ?_
      · intro hv0
        wp_seq
        refine ⟨by ext_env, c64 0, by slot, by first | rfl | trivial, ?_⟩
        rw [z0]
        refine ⟨fun h => absurd rfl h, ?_⟩
        rintro ⟨p, hp, rfl, hpe⟩
        exfalso
        exact slotVal_ne_zero (H (ks[p]'hp)) (p := p) (by omega) (by simpa [cmpInt, z0] using hv0)
      · intro hvne ne2 hne2
        have hv0 : v ≠ 0 := by simpa [cmpInt, z0] using hvne
        obtain ⟨p, hp, hvp⟩ := hvcase.resolve_left hv0
        have hp32 : p + 1 < 2 ^ 32 := by omega
        obtain ⟨ko, f0, f8, bk⟩ := hP.key p hp
        have eA : L.e + ((v &&& c64 0xffffffff) + c64 (-1)) * c64 (ENTRY : Nat) =
            L.e + UInt64.ofNat (p * ENTRY) := by
          rw [hvp, slotVal_lo _ hp32, UInt64.ofNat_add, UInt64.add_assoc,
            show UInt64.ofNat 1 + c64 (-1) = 0 from by decide, UInt64.add_zero,
            show c64 (ENTRY : Nat) = UInt64.ofNat ENTRY from by decide, ← UInt64.ofNat_mul]
        have e8 : L.e + UInt64.ofNat (p * ENTRY) + c64 8 = L.e + UInt64.ofNat (p * ENTRY + 8) := by
          rw [UInt64.add_assoc, show c64 8 = UInt64.ofNat 8 from by decide, ← UInt64.ofNat_add]
        wp_seq
        refine wp_iaddImm (x := v &&& c64 0xffffffff) (by slot) ?_
        apply wp_iconst64
        refine wp_imul (by slot) (by slot) ?_
        refine wp_iadd (x := L.e) (by slot) (by slot) ?_
        rw [eA]
        refine wp_iaddImm (x := L.e + UInt64.ofNat (p * ENTRY)) (by slot) ?_
        refine wp_load64 (by slot) (by rw [e8]; exact f8) ?_
        refine wp_ifteV (fun x w' => w = w' ∧ (x ≠ 0 ↔ Match ks kb v)) (x := UInt64.ofNat (ks[p]'hp).length)
          (y := UInt64.ofNat kb.length) (by slot) (by slot) ?_ ?_ ?_
        · intro hlne
          wp_seq
          refine ⟨by ext_env, c64 0, by slot, by first | rfl | trivial, ?_⟩
          rw [z0]
          refine ⟨fun h => absurd rfl h, ?_⟩
          rintro ⟨p', hp', hv', hpe⟩
          exfalso
          have := slotVal_inj hp32 (by omega) (hvp.symm.trans hv')
          subst this
          have e : (ks[p]'hp).length = kb.length := by rw [hpe]
          have hf : cmpInt .ne .i64 (UInt64.ofNat kb.length) (UInt64.ofNat kb.length) = false :=
            bne_self_eq_false _
          rw [e, hf] at hlne
          cases hlne
        · intro hleq ne3 hne3
          have hll : (ks[p]'hp).length = kb.length := by
            have hb1 := bk.len_le
            have hb2 := hB.len_le
            have h1 : UInt64.ofNat (ks[p]'hp).length = UInt64.ofNat kb.length := by simpa [cmpInt] using hleq
            have h2 := congrArg UInt64.toNat h1
            rwa [UInt64.toNat_ofNat', UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega),
              Nat.mod_eq_of_lt (by omega)] at h2
          wp_seq
          refine wp_at (x := sv) (by slot) ?_
          refine wp_load64 (by slot) hP.far ?_
          refine wp_load64 (by slot) f0 ?_
          refine wp_iadd (by slot) (by slot) ?_
          refine wp_bytesEq (as := (ks[p]'hp)) (bs := kb) (by slot) (by slot)
            (by slot) (by rw [hll]) bk hB hll.symm ?_
          intro Γ4 r hE4 hr
          refine ⟨Ext.trans (by ext_env) hE4, _, hr, by first | rfl | trivial, ?_⟩
          by_cases c : (ks[p]'hp) = kb
          · rw [if_pos c]; exact ⟨fun _ => ⟨p, hp, hvp, c⟩, fun _ => by decide⟩
          · rw [if_neg c]
            refine ⟨fun h => absurd rfl h, ?_⟩
            rintro ⟨p', hp', hv', hpe⟩
            have := slotVal_inj hp32 (by omega) (hvp.symm.trans hv')
            subst this
            exact absurd hpe c
        · intro Γ5 w5 x nJ hE5 hx hr
          exact ⟨Ext.trans (by ext_env) hE5, x, hx, hr⟩
      · intro Γ5 w5 x nJ hE5 hx hr
        exact ⟨Ext.trans (by ext_env) hE5, x, hx, hr⟩
    · intro Γ6 w6 fnd nJ hE6 hfnd hr
      obtain ⟨hw, hiff⟩ := hr
      subst hw
      simp only [carriesFrom, Vals.head]
      refine wp_icmp (cc := .eq) (x := v) (y := c64 0) (by slot) (by slot) ?_
      refine wp_icmp (cc := .ne) (x := fnd) (y := c64 0) (by slot) (by slot) ?_
      refine wp_bor8 (by slot) (by slot) ?_
      apply wp_iconst8
      refine wp_icmp (cc := .eq) (x := v) (y := c64 0) (by slot) (by slot) ?_
      refine wp_select (x := c64 0) (y := v &&& c64 0xffffffff) (by slot) (by slot) (by slot) ?_
      refine ⟨Ext.trans (Ext.trans (by ext_env) hE6) (by ext_env), rfl, ⟨t, hi, hprev⟩, _, _, _, _, v, _, _, rfl,
        by slot, by slot, by slot, by slot, hv, ?_, ?_⟩
      · intro hst
        refine ⟨i, hilt, rfl, ?_⟩
        by_cases hv0 : v = 0
        · subst hv0
          refine Or.inl ⟨by simp [cmpInt, z0], hv, ?_, t, hi, fun t' ht' => (hprev t' ht').1⟩
          intro hmem
          obtain ⟨p, hp, hpe⟩ := List.getElem_of_mem hmem
          obtain ⟨t2, ht2, hbef⟩ := hP.chain p hp
          have hH : H (ks[p]'hp) = H kb := by rw [hpe]
          rcases Nat.lt_trichotomy t2 t with h | h | h
          · exact (hprev t2 h).2 p hp (by rw [← hH]; exact ht2) hpe
          · subst h
            rw [hH, ← hi, hv] at ht2
            exact slotVal_ne_zero _ (by omega) (Option.some.inj ht2).symm
          · rw [hH] at hbef
            exact hbef t h (by rw [← hi]; exact hv)
        · have hf : fnd ≠ 0 := by simpa [cmpInt, z0, hv0] using hst
          obtain ⟨p, hp, hvp, hpe⟩ := hiff.mp hf
          refine Or.inr ⟨p, hp, ?_, hpe⟩
          simp only [cmpInt, z0, beq_iff_eq, hv0, if_false]
          rw [hvp, slotVal_lo _ (by omega)]
      · intro hst
        have h2 : v ≠ 0 ∧ fnd = 0 := by simpa [cmpInt, z0] using hst
        exact ⟨h2.1, fun p hp hvp hpe => (hiff.mpr ⟨p, hp, hvp, hpe⟩) h2.2⟩
  · -- leaving
    intro x a Γ1 w1 t f vs hd hev hxx hvs
    obtain ⟨hw, -, sst, sz, ssl, ssel, v, stop, sel, rfl, h1, h2, h3, h4, hv, hT, hF⟩ := hd
    subst hw
    have he := evalStop (m := w.mem) h1 h2
    simp only [exitIf] at hev hxx hvs
    rw [he] at hev
    cases stop with
    | false =>
      exfalso
      simp [boolV] at hev
      obtain ⟨rfl, rfl⟩ := hev
      simp at hxx
    | true =>
      simp only [Vals.slots, List.mapM_cons, List.mapM_nil, get_push h3, get_push h4, Option.pure_def,
        Option.bind_eq_bind, Option.bind_some, Option.some.injEq] at hvs
      exact ⟨rfl, _, _, hvs.symm, hT rfl⟩
  · -- going round
    intro J' x a Γ1 wb t f Γ2 si hd hE1 hev hxx hE2 hsi
    obtain ⟨hw, ⟨t0, hi, hprev⟩, sst, sz, ssl, ssel, v, stop, sel, rfl, h1, h2, h3, h4, hv, hT, hF⟩ := hd
    subst hw
    have he := evalStop (m := w.mem) h1 h2
    simp only [exitIf] at hev hxx
    rw [he] at hev
    have hst : stop = false := by
      cases stop with
      | false => rfl
      | true =>
        exfalso
        simp [boolV] at hev
        obtain ⟨rfl, rfl⟩ := hev
        simp at hxx
    obtain ⟨hv0, hnm⟩ := hF hst
    wp_seq
    refine wp_iaddImm hsi ?_
    refine wp_band (x := x + c64 1) (y := UInt64.ofNat L.icap + c64 (-1)) (by slot) (by slot) ?_
    refine ⟨_, by slot, rfl, t0 + 1, by rw [hi]; rfl, fun t' ht' => ?_⟩
    rcases Nat.lt_or_ge t' t0 with h | h
    · exact hprev t' h
    · have ht : t' = t0 := by omega
      subst ht
      rw [← hi]
      exact ⟨by rw [hv]; simpa using hv0, fun p hp hf => hnm p hp (Option.some.inj (hv.symm.trans hf))⟩
  · intro x a Γ1 w1 hd
    obtain ⟨hw, -, sst, sz, ssl, ssel, v, stop, sel, rfl, h1, h2, h3, h4, hv, hT, hF⟩ := hd
    subst hw
    simp only [exitIf]
    rw [evalStop (m := w.mem) h1 h2]
    exact nofun
  · intro Γb vs w' nE hEb hle hX
    obtain ⟨hw, a, p, rfl, hF⟩ := hX
    subst hw
    simp only [carriesFrom, Vals.head, Vals.snd, Vals.tail]
    exact hk _ _ _ _ _ (Ext.trans (by ext_env) (hEb.bindAt' hle _)) (bindAt_head _ _ _ _)
      (bindAt_second _ _ _ _ _) hF

end fns

theorem toList_loop (b : ByteArray) : ∀ (n i : Nat) (r : List UInt8), b.size - i = n → i ≤ b.size →
    ByteArray.toList.loop b i r = r.reverse ++ b.data.toList.drop i
  | 0, i, r, hn, hi => by
    rw [ByteArray.toList.loop, if_neg (by omega), List.drop_of_length_le (by
      rw [Array.length_toList]; have : b.size = b.data.size := rfl; omega)]
    simp
  | n + 1, i, r, hn, hi => by
    have hlt : i < b.size := by omega
    rw [ByteArray.toList.loop, if_pos hlt, toList_loop b n (i + 1) _ (by omega) (by omega)]
    have hd : b.data.toList.drop i = b.get! i :: b.data.toList.drop (i + 1) := by
      rw [List.drop_eq_getElem_cons (by rw [Array.length_toList]; exact hlt)]
      congr 1
      simp [ByteArray.get!, hlt]
    rw [hd]
    simp

/-- A byte array's list is its data's. -/
theorem toList_data (b : ByteArray) : b.toList = b.data.toList := by
  unfold ByteArray.toList
  rw [toList_loop b b.size 0 [] rfl (Nat.zero_le _)]
  simp

/-- A table with its index is one a probe can read. -/
theorem Tab.probe {w : World} {s : UInt64} {T : Ht} {L : Lay} (h : Tab w s T L) (hI : Idx w L (keysOf T)) :
    Probe w s L (keysOf T) := by
  refine ⟨h.ficap, h.fix, h.fe, h.far, hI.cap, hI.small, fun p hp => ?_, hI.slots, hI.chain⟩
  have hp' : p < T.entries.length := by simpa [keysOf] using hp
  obtain ⟨ko, vo, f0, f8, -, -, -, -, bk, -⟩ := h.ent p hp'
  refine ⟨ko, f0, ?_, ?_⟩
  · simpa [keysOf, size_toList] using f8
  · simpa [keysOf] using bk

/-- A write clear of the index leaves it. -/
theorem Idx.wrote {w w' : World} {L : Lay} {ks : List (List UInt8)} {p : UInt64} {n : Nat} (hI : Idx w L ks)
    (hb : Blk w L.ix (L.icap * 8)) (hw : Wrote w w' p n)
    (hc : ∀ r off oy, decodeAddr p = some (r, off) → decodeAddr L.ix = some (.pinned, oy) →
      r ≠ .pinned ∨ off + n ≤ oy ∨ oy + L.icap * 8 ≤ off) : Idx w' L ks := by
  have hc1 : L.icap < 2 ^ 64 := by have := hI.cap.2; omega
  have hf : ∀ j, j < L.icap → fld w' L.ix (8 * j) = fld w L.ix (8 * j) := fun j hj =>
    hw.fld_blk hb (by omega) hc
  refine ⟨hI.cap, hI.small, hI.nodup, fun j hj => ?_, fun q hq => ?_⟩
  · rw [hf j hj]; exact hI.slots j hj
  · obtain ⟨t, ht, hbef⟩ := hI.chain q hq
    refine ⟨t, by rw [hf _ (path_lt _ _ hI.cap.1 hc1)]; exact ht, fun t' ht' => ?_⟩
    rw [hf _ (path_lt _ _ hI.cap.1 hc1)]
    exact hbef t' ht'

theorem lookup_fine : Fine (emitGo lookup ⟨4, 0, [], [], [], none⟩).2 := ⟨by decide +kernel, by decide +kernel⟩

/-- What `htLookup` answers and leaves: `0xFFFFFFFF` and the world where
    there is no table or no entry with the key; else the value's length, and
    memory with the value copied to `out`, a write of that run, the table as
    it was. -/
def LookQ (w : World) (s : UInt64) (T : Ht) (L : Lay) (kb : List UInt8) (out : UInt64) (x : UInt64)
    (w' : World) : Prop :=
  if T.hasTable then
    (kb ∉ keysOf T ∧ x = c64 0xFFFFFFFF ∧ w' = w) ∨
    ∃ p, ∃ hp : p < T.entries.length, (T.entries[p]'hp).1.data.toList = kb ∧
      x = UInt64.ofNat (T.entries[p]'hp).2.size ∧ copyIn w.mem out (T.entries[p]'hp).2 = some w'.mem ∧
      Tab w' s T L ∧ Idx w' L (keysOf T) ∧ Wrote w w' out (T.entries[p]'hp).2.size
  else x = c64 0xFFFFFFFF ∧ w' = w

set_option maxHeartbeats 1600000 in
/-- **`htLookup`** of a key that can be read, handed room for its value clear
    of the table, answers what the model answers and leaves the memory the
    model leaves. -/
theorem lookup_spec (cfg : Cfg) (w : World) (s : UInt64) (T : Ht) (L : Lay) (h : Tab w s T L)
    (hI : Idx w L (keysOf T)) (key klen out : UInt64) (kb : List UInt8) (hB : Bytes w key kb)
    (hkl : klen = UInt64.ofNat kb.length) (hk32 : kb.length < 2 ^ 32)
    (hroom : ∀ p (hp : p < T.entries.length), (T.entries[p]'hp).1.data.toList = kb →
      Room w out (T.entries[p]'hp).2.size ∧ Clear s L out (T.entries[p]'hp).2.size) :
    Ans cfg [.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 out] w lookup [.i64, .i64, .i32, .i64]
      (LookQ w s T L kb out) := by
  refine ans_of_wp ?_ lookup_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [lookup, entryOf, valAt, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 out] : Env)[0]? = some (.sc .i64 s) := rfl
  have h1 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 out] : Env)[0 + 1]? = some (.sc .i64 key) := rfl
  have h2 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 out] : Env)[0 + 1 + 1]? =
    some (.sc .i32 klen) := rfl
  have h3 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 out] : Env)[0 + 1 + 1 + 1]? =
    some (.sc .i64 out) := rfl
  have z0 : c64 0 = 0 := by decide
  refine wp_withTable (c := UInt64.ofNat T.handles.length) (by slot) (fun _ => h.rep.created) ?_ ?_
  · intro hz Γ' hext
    have hl : T.handles.length = 0 := by
      rcases hz with hz | hz
      · exact absurd hz (Blk.ne_zero h.rep.blk)
      · have := congrArg UInt64.toNat hz
        rwa [UInt64.toNat_ofNat', Nat.mod_eq_of_lt h.rep.small.1] at this
    apply wp_iconst64
    exact ⟨_, get_push_size, by simp [LookQ, h.rep.hasTable, hl]⟩
  · intro _ hc Γ' hext
    have hl : T.handles.length ≠ 0 := fun e => hc (by rw [e]; rfl)
    have hT : T.hasTable = true := by simp [h.rep.hasTable, hl]
    have hmask : klen &&& 0xFFFFFFFF = UInt64.ofNat kb.length := by
      rw [hkl]
      apply UInt64.toNat_inj.mp
      rw [UInt64.toNat_and, show (0xFFFFFFFF : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
        UInt64.toNat_ofNat']
      omega
    wp_seq
    refine wp_uextend64 (x := klen) (by slot) ?_
    rw [hmask]
    refine wp_hash (bs := kb) (by slot) (by slot) hB ?_
    intro Γ1 r hE1 hr
    refine wp_find (kb := kb) (L := L) (ks := keysOf T) (by slot) (by slot) (by slot) hr (h.probe hI) hB ?_
    intro Γ2 ra rp slotA pos hE2 hra hrp hF
    apply wp_iconst64
    refine wp_ifte (AnsJ (LookQ w s T L kb out)) (x := pos) (y := c64 0) (by slot) (by slot) ?_ ?_
      (by join_ans)
    · intro hp0
      have hp0' : pos = 0 := by simpa [cmpInt, z0] using hp0
      wp_seq
      apply wp_iconst64
      refine ⟨_, get_push_size, ?_⟩
      unfold LookQ
      rw [if_pos hT]
      obtain ⟨i, -, -, hF⟩ := hF
      rcases hF with ⟨-, -, hnot, -⟩ | ⟨p, hp, hpos, -⟩
      · exact Or.inl ⟨hnot, rfl, rfl⟩
      · exfalso
        have hsm := hI.small
        have := congrArg UInt64.toNat (hpos.symm.trans hp0')
        rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)] at this
        simp at this
    · intro hpne ne hne
      have hpne' : pos ≠ 0 := by simpa [cmpInt, z0] using hpne
      obtain ⟨i, -, -, hF⟩ := hF
      rcases hF with ⟨hpos, -⟩ | ⟨p, hp, hpos, hkey⟩
      · exact absurd hpos hpne'
      have hp' : p < T.entries.length := by simpa [keysOf] using hp
      have hkey' : (T.entries[p]'hp').1.data.toList = kb := by simpa [keysOf] using hkey
      obtain ⟨rv, cv⟩ := hroom p hp' hkey'
      obtain ⟨ko, vo, f0, f8, f16, f24, hko, hvo, bk, bv⟩ := h.ent p hp'
      have hr := h.room
      have hacap : L.acap ≤ 2 ^ 36 := by
        obtain ⟨o, m, -, hn, -, -, h36, -⟩ := h.ba
        omega
      have hvs : (UInt64.ofNat (T.entries[p]'hp').2.size).toNat = (T.entries[p]'hp').2.size := by
        rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
      have eA : L.e + (pos + c64 (-1)) * c64 (ENTRY : Nat) = L.e + UInt64.ofNat (p * ENTRY) := by
        rw [hpos, UInt64.ofNat_add, UInt64.add_assoc, show UInt64.ofNat 1 + c64 (-1) = 0 from by decide,
          UInt64.add_zero, show c64 (ENTRY : Nat) = UInt64.ofNat ENTRY from by decide, ← UInt64.ofNat_mul]
      have e8 : ∀ (j : Nat) (z : Int), c64 z = UInt64.ofNat j → L.e + UInt64.ofNat (p * ENTRY) + c64 z =
          L.e + UInt64.ofNat (p * ENTRY + j) := fun j z hz => by
        rw [UInt64.add_assoc, hz, ← UInt64.ofNat_add]
      wp_seq
      refine wp_at (x := s) (by slot) ?_
      refine wp_load64 (by slot) h.fe ?_
      refine wp_iaddImm (x := pos) (by slot) ?_
      apply wp_iconst64
      refine wp_imul (by slot) (by slot) ?_
      refine wp_iadd (x := L.e) (by slot) (by slot) ?_
      rw [eA]
      refine wp_iaddImm (x := L.e + UInt64.ofNat (p * ENTRY)) (by slot) ?_
      refine wp_load64 (by slot) (by rw [e8 24 24 (by decide)]; exact f24) ?_
      refine wp_at (x := s) (by slot) ?_
      refine wp_load64 (by slot) h.far ?_
      refine wp_iaddImm (x := L.e + UInt64.ofNat (p * ENTRY)) (by slot) ?_
      refine wp_load64 (by slot) (by rw [e8 16 16 (by decide)]; exact f16) ?_
      refine wp_iadd (by slot) (by slot) ?_
      refine wp_memcpy (bs := (T.entries[p]'hp').2.data.toList) (by slot) (by slot) (by slot) bv
        (by rw [size_toList, hvs]) (by rw [hvs]; exact rv)
        (by rw [hvs]; exact apart_blk rv h.ba (by omega) (cv _ _ (by simp [blocks]))) ?_
      intro w1 hw1 hb1 hc1
      rw [hvs] at hw1
      refine ⟨_, by slot, ?_⟩
      have ek : ∀ b : ByteArray, ByteArray.mk b.data.toList.toArray = b := fun b => by simp
      rw [ek] at hc1
      unfold LookQ
      rw [if_pos hT]
      exact Or.inr ⟨p, hp', hkey', rfl, hc1, h.wrote hw1 cv, hI.wrote h.bi hw1 (cv _ _ (by simp [blocks])), hw1⟩

/-- The model finds no value for a key none of the entries has. -/
theorem get_none {T : Ht} {kb : List UInt8} (h : kb ∉ keysOf T) : T.get (ByteArray.mk kb.toArray) = none := by
  unfold Ht.get
  rw [List.find?_eq_none.mpr fun e he => by
    simp only [toList_data, beq_iff_eq]
    exact fun c => h (List.mem_map.mpr ⟨e, he, by simpa using c⟩)]
  rfl

theorem find_key {kb : List UInt8} : ∀ (l : List (ByteArray × ByteArray)) (p : Nat) (hp : p < l.length),
    (l.map (·.1.data.toList)).Nodup → (l[p]'hp).1.data.toList = kb →
    l.find? (fun e => e.1.toList == (ByteArray.mk kb.toArray).toList) = some (l[p]'hp)
  | e :: l, 0, _, _, hk => by
    have hm : (e.1.toList == (ByteArray.mk kb.toArray).toList) = true := by simpa [toList_data] using hk
    rw [List.find?_cons, hm]
    rfl
  | e :: l, p + 1, hp, hn, hk => by
    have hn' := List.nodup_cons.mp hn
    have hmem : kb ∈ l.map (·.1.data.toList) := by
      rw [← hk]
      exact List.mem_map.mpr ⟨l[p]'(by simpa using hp), List.getElem_mem _, rfl⟩
    have hne : e.1.data.toList ≠ kb := fun c => hn'.1 (by
      show e.1.data.toList ∈ _
      rw [c]; exact hmem)
    have hm : (e.1.toList == (ByteArray.mk kb.toArray).toList) = false := by simpa [toList_data] using hne
    rw [List.find?_cons, hm]
    exact find_key l p (by simpa using hp) hn'.2 hk

/-- The model finds the value of the entry with the key, the keys being
    distinct. -/
theorem get_some {T : Ht} {kb : List UInt8} {p : Nat} (hp : p < T.entries.length) (hn : (keysOf T).Nodup)
    (hk : (T.entries[p]'hp).1.data.toList = kb) : T.get (ByteArray.mk kb.toArray) = some (T.entries[p]'hp).2 := by
  unfold Ht.get
  rw [find_key T.entries p hp hn hk]
  rfl

/-- A table's context and arrays are a context with its arrays. -/
theorem Tab.owned {w : World} {s : UInt64} {T : Ht} {L : Lay} (h : Tab w s T L) : Owned w s L.e L.ix L.ar := by
  obtain ⟨a1, a2, a3, a4, a5, a6⟩ := h.apart
  obtain ⟨p1, p2, p3⟩ := h.pos
  have q1 : 0 < L.ecap * ENTRY := Nat.mul_pos p1 (by decide)
  have q2 : 0 < L.icap * 8 := Nat.mul_pos p2 (by decide)
  exact ⟨h.rep.blk, h.fe, h.fix, h.far, ⟨Or.inr ⟨_, h.be⟩, Or.inr ⟨_, h.bi⟩, Or.inr ⟨_, h.ba⟩⟩,
    (a1.ne (by decide) q1).symm, (a2.ne (by decide) q2).symm, (a3.ne (by decide) p3).symm,
    Or.inr (a4.ne q1 q2).symm, Or.inr (a5.ne q1 p3).symm, Or.inr (a6.ne q2 p3).symm⟩

/-- What each arm of `htInit` joins with: the slot's address still in slot 0,
    and null or a context holding the empty table with its index. -/
abbrev InitT (w : World) (r : Region) (slot : UInt64) : Vals Slot [.i64] → Env → World → Prop :=
  fun jv Γ' w' => Γ'[0]? = some (.sc .i64 slot) ∧ ∃ x, Γ'[jv.head]? = some (.sc .i64 x) ∧ Open w r w' ∧
    (Outside w w' fun _ _ => True) ∧
    (x = 0 ∨ ∃ L, Tab w' x {} L ∧ Idx w' L [])

theorem initT_join {w : World} {r : Region} {slot : UInt64} {nJ : Nat} {Γ' : Env} {jv : Vals Slot [.i64]}
    {vs : List V} {w' : World} (hle : Γ'.size ≤ nJ) (hvs : jv.slots.mapM (fun r => Γ'[r]?) = some vs)
    (h : InitT w r slot jv Γ' w') : InitT w r slot (carriesFrom nJ [.i64]) (bindAt Γ' nJ vs) w' := by
  obtain ⟨hslot, x, hx, rest⟩ := h
  have hj := join1 hvs hx
  subst hj
  exact ⟨get_bindAt hle hslot, x, AlgorithmLib.Proof.bindAt_last _ _ _ hle, rest⟩

set_option maxHeartbeats 1600000 in
/-- **`htInit`** leaves in its slot a context holding the empty table, with
    its arrays and its index, or null where the context or an array could not
    be allocated, and outside the heap changes the slot alone. -/
theorem init_spec (cfg : Cfg) (w : World) (slot : UInt64) (r : Region) (off : Nat) (hc : Calm w)
    (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned) (hroom : off + 8 ≤ (w.mem.region r).size) :
    Does cfg [.sc .i64 slot] w init [.i64] (fun w' =>
      ∃ s, w'.mem.load slot 8 = some s ∧ (s = 0 ∨ ∃ L, Tab w' s {} L ∧ Idx w' L []) ∧
        Outside w w' (Misses slot)) := by
  refine does_of_wp ?_ init_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [init, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 slot] : Env)[0]? = some (.sc .i64 slot) := rfl
  have z0 : c64 0 = 0 := by decide
  apply wp_iconst64
  apply wp_iconst64
  refine wp_calloc (by slot) (by decide) ?_
  intro s w1 g1 hs
  refine wp_ifte (InitT w r slot) (x := s) (y := c64 0) (by slot) (by slot) ?_ ?_ ?_
  · intro _
    wp_seq
    refine ⟨?_, c64 0, ?_, open_of_stays g1.stays hc hr, g1.outside _, Or.inl z0⟩ <;> slot
  · intro hne1 ne1 hn1
    have fs := hs.resolve_left (by simpa [cmpInt, z0] using hne1)
    have hrep : Rep w1 s {} := Rep.fresh (g1.calm hc) fs.blk (fs.zero hc)
    have hU : fld w1 s AUSED = some 0 := fs.zero hc AUSED (by decide)
    have hst : Stays w w1 := g1.stays
    have ho : Outside w w1 fun _ _ => True := g1.outside _
    wp_seq
    apply wp_iconst64
    refine wp_calloc (by slot) (by decide) ?_
    intro e w2 g2 he
    have hes := fresh_ne he hrep.blk
    have hbs1 := hrep.blk
    replace hrep := hrep.grows g2
    replace hU := g2.fld hU
    replace hst := hst.trans g2.stays
    replace ho := ho.trans (g2.outside _)
    refine wp_ifte (InitT w r slot) (x := e) (y := c64 0) (by slot) (by slot) ?_ ?_
      (fun nJ Γ' jv vs w' hle hvs hq => initT_join hle hvs hq)
    · intro _
      wp_seq
      refine wp_free (by slot) (Or.inr hrep.blk) ?_
      intro v1 f1
      refine ⟨?_, c64 0, ?_, (open_of_stays hst hc hr).freed f1 hr, ho.trans (f1.outside _), Or.inl z0⟩ <;> slot
    · intro hne2 ne2 hn2
      have fe := he.resolve_left (by simpa [cmpInt, z0] using hne2)
      have be : Blk w2 e (64 * ENTRY) := fe.blk
      have we : Whole w2 e (64 * ENTRY) := fe.whole
      have a_se : Apart s SIZE e (64 * ENTRY) := Or.inl (fe.before hbs1)
      have oe : Own w2 e := Or.inr ⟨_, be⟩
      wp_seq
      apply wp_iconst64
      refine wp_calloc (by slot) (by decide) ?_
      intro ix w3 g3 hi
      have his := fresh_ne hi hrep.blk
      have hie := fresh_apart hi oe
      have hbs2 := hrep.blk
      have hbe2 := be
      have hcalm2 := hrep.calm
      replace be := g3.blk _ _ be
      replace we := g3.whole _ _ we
      replace oe := oe.grows g3
      replace hrep := hrep.grows g3
      replace hU := g3.fld hU
      replace hst := hst.trans g3.stays
      replace ho := ho.trans (g3.outside _)
      refine wp_ifte (InitT w r slot) (x := ix) (y := c64 0) (by slot) (by slot) ?_ ?_
        (fun nJ Γ' jv vs w' hle hvs hq => initT_join hle hvs hq)
      · intro _
        wp_seq
        refine wp_free (by slot) (Or.inr be) ?_
        intro v1 f1
        refine wp_free (by slot) (Or.inr (f1.blk _ _ hes.symm hrep.blk)) ?_
        intro v2 f2
        refine ⟨?_, c64 0, ?_, ((open_of_stays hst hc hr).freed f1 hr).freed f2 hr,
          (ho.trans (f1.outside _)).trans (f2.outside _), Or.inl z0⟩ <;> slot
      · intro hne3 ne3 hn3
        have fi := hi.resolve_left (by simpa [cmpInt, z0] using hne3)
        have bi : Blk w3 ix (128 * 8) := fi.blk
        have wi : Whole w3 ix (128 * 8) := fi.whole
        have zi : ∀ j, j + 8 ≤ 128 * 8 → fld w3 ix j = some 0 := fi.zero hcalm2
        have a_si : Apart s SIZE ix (128 * 8) := Or.inl (fi.before hbs2)
        have a_ei : Apart e (64 * ENTRY) ix (128 * 8) := Or.inl (fi.before hbe2)
        have oi : Own w3 ix := Or.inr ⟨_, bi⟩
        have hie' : ix ≠ e := hie.resolve_left (Blk.ne_zero bi)
        wp_seq
        apply wp_iconst64
        refine wp_calloc (by slot) (by decide) ?_
        intro ar w4 g4 ha
        have has := fresh_ne ha hrep.blk
        have hbs3 := hrep.blk
        have hbe3 := be
        have hbi3 := bi
        replace be := g4.blk _ _ be
        replace bi := g4.blk _ _ bi
        replace we := g4.whole _ _ we
        replace wi := g4.whole _ _ wi
        replace zi : ∀ j, j + 8 ≤ 128 * 8 → fld w4 ix j = some 0 := fun j hj => g4.fld (zi j hj)
        replace hrep := hrep.grows g4
        replace hU := g4.fld hU
        replace hst := hst.trans g4.stays
        replace ho := ho.trans (g4.outside _)
        refine wp_ifte (InitT w r slot) (x := ar) (y := c64 0) (by slot) (by slot) ?_ ?_
          (fun nJ Γ' jv vs w' hle hvs hq => initT_join hle hvs hq)
        · intro _
          wp_seq
          refine wp_free (by slot) (Or.inr bi) ?_
          intro v1 f1
          refine wp_free (by slot) (Or.inr (f1.blk _ _ hie'.symm be)) ?_
          intro v2 f2
          refine wp_free (by slot) (Or.inr (f2.blk _ _ hes.symm (f1.blk _ _ his.symm hrep.blk))) ?_
          intro v3 f3
          refine ⟨?_, c64 0, ?_, (((open_of_stays hst hc hr).freed f1 hr).freed f2 hr).freed f3 hr,
            ((ho.trans (f1.outside _)).trans (f2.outside _)).trans (f3.outside _), Or.inl z0⟩ <;> slot
        · intro hne4 ne4 hn4
          have fa := ha.resolve_left (by simpa [cmpInt, z0] using hne4)
          have ba : Blk w4 ar 4096 := fa.blk
          have wa : Whole w4 ar 4096 := fa.whole
          have a_sa : Apart s SIZE ar 4096 := Or.inl (fa.before hbs3)
          have a_ea : Apart e (64 * ENTRY) ar 4096 := Or.inl (fa.before hbe3)
          have a_ia : Apart ix (128 * 8) ar 4096 := Or.inl (fa.before hbi3)
          wp_seq
          refine wp_at (by slot) ?_
          refine wp_setFld (j := ENTRIES) (by slot) (by slot) hrep.blk hrep.calm (by decide) ?_
          intro m1 hm1
          have hE := fld_stored hm1
          replace hU := (fld_kept hrep.blk (i := AUSED) (by decide) (by decide) (by decide) hm1).trans hU
          replace zi := fun j hj => (fld_other hrep.blk bi a_si (by decide) hj hm1).trans (zi j hj)
          replace bi := Blk.keep bi hm1
          replace we := (wrote_stored hm1).whole we
          replace wi := (wrote_stored hm1).whole wi
          replace wa := (wrote_stored hm1).whole wa
          replace hst := hst.trans (stays_stored hm1)
          replace ho := ho.trans (Outside.setFld hrep.blk (by decide) hm1 _)
          replace hrep := hrep.setFld (by decide) (by decide) hm1
          apply wp_iconst64
          refine wp_at (by slot) ?_
          refine wp_setFld (j := ECAP) (by slot) (by slot) hrep.blk hrep.calm (by decide) ?_
          intro m2 hm2
          have hEC := fld_stored hm2
          replace hE := (fld_kept hrep.blk (i := ENTRIES) (by decide) (by decide) (by decide) hm2).trans hE
          replace hU := (fld_kept hrep.blk (i := AUSED) (by decide) (by decide) (by decide) hm2).trans hU
          replace zi := fun j hj => (fld_other hrep.blk bi a_si (by decide) hj hm2).trans (zi j hj)
          replace bi := Blk.keep bi hm2
          replace we := (wrote_stored hm2).whole we
          replace wi := (wrote_stored hm2).whole wi
          replace wa := (wrote_stored hm2).whole wa
          replace hst := hst.trans (stays_stored hm2)
          replace ho := ho.trans (Outside.setFld hrep.blk (by decide) hm2 _)
          replace hrep := hrep.setFld (by decide) (by decide) hm2
          refine wp_at (by slot) ?_
          refine wp_setFld (j := INDEX) (by slot) (by slot) hrep.blk hrep.calm (by decide) ?_
          intro m3 hm3
          have hI := fld_stored hm3
          replace hE := (fld_kept hrep.blk (i := ENTRIES) (by decide) (by decide) (by decide) hm3).trans hE
          replace hEC := (fld_kept hrep.blk (i := ECAP) (by decide) (by decide) (by decide) hm3).trans hEC
          replace hU := (fld_kept hrep.blk (i := AUSED) (by decide) (by decide) (by decide) hm3).trans hU
          replace zi := fun j hj => (fld_other hrep.blk bi a_si (by decide) hj hm3).trans (zi j hj)
          replace bi := Blk.keep bi hm3
          replace we := (wrote_stored hm3).whole we
          replace wi := (wrote_stored hm3).whole wi
          replace wa := (wrote_stored hm3).whole wa
          replace hst := hst.trans (stays_stored hm3)
          replace ho := ho.trans (Outside.setFld hrep.blk (by decide) hm3 _)
          replace hrep := hrep.setFld (by decide) (by decide) hm3
          apply wp_iconst64
          refine wp_at (by slot) ?_
          refine wp_setFld (j := ICAP) (by slot) (by slot) hrep.blk hrep.calm (by decide) ?_
          intro m4 hm4
          have hIC := fld_stored hm4
          replace hE := (fld_kept hrep.blk (i := ENTRIES) (by decide) (by decide) (by decide) hm4).trans hE
          replace hEC := (fld_kept hrep.blk (i := ECAP) (by decide) (by decide) (by decide) hm4).trans hEC
          replace hI := (fld_kept hrep.blk (i := INDEX) (by decide) (by decide) (by decide) hm4).trans hI
          replace hU := (fld_kept hrep.blk (i := AUSED) (by decide) (by decide) (by decide) hm4).trans hU
          replace zi := fun j hj => (fld_other hrep.blk bi a_si (by decide) hj hm4).trans (zi j hj)
          replace bi := Blk.keep bi hm4
          replace we := (wrote_stored hm4).whole we
          replace wi := (wrote_stored hm4).whole wi
          replace wa := (wrote_stored hm4).whole wa
          replace hst := hst.trans (stays_stored hm4)
          replace ho := ho.trans (Outside.setFld hrep.blk (by decide) hm4 _)
          replace hrep := hrep.setFld (by decide) (by decide) hm4
          refine wp_at (by slot) ?_
          refine wp_setFld (j := ARENA) (by slot) (by slot) hrep.blk hrep.calm (by decide) ?_
          intro m5 hm5
          have hA := fld_stored hm5
          replace hE := (fld_kept hrep.blk (i := ENTRIES) (by decide) (by decide) (by decide) hm5).trans hE
          replace hEC := (fld_kept hrep.blk (i := ECAP) (by decide) (by decide) (by decide) hm5).trans hEC
          replace hI := (fld_kept hrep.blk (i := INDEX) (by decide) (by decide) (by decide) hm5).trans hI
          replace hIC := (fld_kept hrep.blk (i := ICAP) (by decide) (by decide) (by decide) hm5).trans hIC
          replace hU := (fld_kept hrep.blk (i := AUSED) (by decide) (by decide) (by decide) hm5).trans hU
          replace zi := fun j hj => (fld_other hrep.blk bi a_si (by decide) hj hm5).trans (zi j hj)
          replace bi := Blk.keep bi hm5
          replace we := (wrote_stored hm5).whole we
          replace wi := (wrote_stored hm5).whole wi
          replace wa := (wrote_stored hm5).whole wa
          replace hst := hst.trans (stays_stored hm5)
          replace ho := ho.trans (Outside.setFld hrep.blk (by decide) hm5 _)
          replace hrep := hrep.setFld (by decide) (by decide) hm5
          apply wp_iconst64
          refine wp_at (by slot) ?_
          refine wp_setFld (j := ACAP) (by slot) (by slot) hrep.blk hrep.calm (by decide) ?_
          intro m6 hm6
          have hAC := fld_stored hm6
          replace hE := (fld_kept hrep.blk (i := ENTRIES) (by decide) (by decide) (by decide) hm6).trans hE
          replace hEC := (fld_kept hrep.blk (i := ECAP) (by decide) (by decide) (by decide) hm6).trans hEC
          replace hI := (fld_kept hrep.blk (i := INDEX) (by decide) (by decide) (by decide) hm6).trans hI
          replace hIC := (fld_kept hrep.blk (i := ICAP) (by decide) (by decide) (by decide) hm6).trans hIC
          replace hA := (fld_kept hrep.blk (i := ARENA) (by decide) (by decide) (by decide) hm6).trans hA
          replace hU := (fld_kept hrep.blk (i := AUSED) (by decide) (by decide) (by decide) hm6).trans hU
          replace zi := fun j hj => (fld_other hrep.blk bi a_si (by decide) hj hm6).trans (zi j hj)
          replace bi := Blk.keep bi hm6
          replace we := (wrote_stored hm6).whole we
          replace wi := (wrote_stored hm6).whole wi
          replace wa := (wrote_stored hm6).whole wa
          replace hst := hst.trans (stays_stored hm6)
          replace ho := ho.trans (Outside.setFld hrep.blk (by decide) hm6 _)
          replace hrep := hrep.setFld (by decide) (by decide) hm6
          refine ⟨?_, s, ?_, open_of_stays hst hc hr, ho, Or.inr ⟨⟨e, 64, ix, 128, ar, 0, 4096, []⟩, ?_, ?_⟩⟩
          · slot
          · slot
          · exact {
              rep := hrep
              fe := hE
              fecap := hEC
              fix := hI
              ficap := hIC
              far := hA
              fused := hU
              facap := hAC
              we := we
              wi := wi
              wa := wa
              room := ⟨Nat.zero_le _, Nat.zero_le _⟩
              pos := ⟨show 0 < 64 from by decide, show 0 < 128 from by decide, show 0 < 4096 from by decide⟩
              apart := ⟨a_se, a_si, a_sa, a_ei, a_ea, a_ia⟩
              olen := rfl
              sep := fun i _ hi => absurd hi (Nat.not_lt_zero i)
              ento := fun i hi => absurd hi (Nat.not_lt_zero i) }
          · exact ⟨⟨show 0 < 128 from by decide, show 128 ≤ 2 ^ 33 from by decide⟩, by decide, List.nodup_nil,
              fun j hj => ⟨0, zi (8 * j) (by have : j < 128 := hj; omega), Or.inl rfl⟩,
              fun p hp => absurd hp (Nat.not_lt_zero p)⟩
  · intro nJ Γ' jv vs w' hle hvs ⟨hslot, x, hx, hop, hO, hq⟩
    have hj := join1 hvs hx
    subst hj
    obtain ⟨m, hm⟩ := store_room (x := x) hd hr (by rw [hop.2]; exact hroom) hop.1
    refine wp_storeI64 (AlgorithmLib.Proof.bindAt_last _ _ _ hle) (get_bindAt hle hslot) hm ?_
    refine ⟨x, load_store8 hm, hq.imp id fun ⟨L, ht, hi⟩ => ⟨L, ?_, ?_⟩,
      (hO.mono fun _ _ _ => trivial).trans (wrote_stored hm).outsideBut⟩
    · exact ht.wrote (wrote_stored hm) fun y k _ r' off' oy hd' _ => Or.inl (by rw [hd] at hd'; cases hd'; exact hr)
    · exact hi.wrote ht.bi (wrote_stored hm) fun r' off' oy hd' _ => Or.inl (by rw [hd] at hd'; cases hd'; exact hr)

/-- `htInit` neither misuses a call nor faults. -/
theorem init_sound (cfg : Cfg) (w : World) (slot : UInt64) (r : Region) (off : Nat) (hc : Calm w)
    (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned) (hroom : off + 8 ≤ (w.mem.region r).size)
    (m : String) :
    Sem.run cfg [.sc .i64 slot] w (emit init [.i64]) ≠ .misuse m ∧
    Sem.run cfg [.sc .i64 slot] w (emit init [.i64]) ≠ .fault m :=
  (init_spec cfg w slot r off hc hd hr hroom).sound m

-- ---------------------------------------------------------------------------
-- The arena
-- ---------------------------------------------------------------------------

/-- What a change to the arena protects: a run inside the memory there was,
    off the arena and off the header's three arena fields. -/
def AppP (w : World) (s : UInt64) (L : Lay) : UInt64 → Nat → Prop := MovP w s ARENA 24 L.ar L.acap

/-- **`w'` is `w` with `bs` appended to the arena**, the arena perhaps moved
    to a fresh block: the layout `L'` differs from `L` in the arena alone,
    what the arena held it holds at the same places, `bs` follows, and what is
    off the old arena and the header's arena fields is kept. -/
structure Appended (w w' : World) (s : UInt64) (L L' : Lay) (bs : List UInt8) : Prop where
  e : L'.e = L.e
  ecap : L'.ecap = L.ecap
  ix : L'.ix = L.ix
  icap : L'.icap = L.icap
  used : L'.used = L.used + bs.length
  offs : L'.offs = L.offs
  keeps : Keeps w w' (AppP w s L)
  calm : Calm w → Calm w'
  blk : ∀ y k, y ≠ L.ar → Blk w y k → Blk w' y k
  whole : ∀ y k, y ≠ L.ar → Whole w y k → Whole w' y k
  size : ∀ r, r ≠ .pinned → (w'.mem.region r).size = (w.mem.region r).size
  far : fld w' s ARENA = some L'.ar
  fused : fld w' s AUSED = some (UInt64.ofNat L'.used)
  facap : fld w' s ACAP = some (UInt64.ofNat L'.acap)
  wa : Whole w' L'.ar L'.acap
  le : L'.used ≤ L'.acap
  apos : 0 < L'.acap
  pre : ∀ j cs, j + cs.length ≤ L.used → Bytes w (L.ar + UInt64.ofNat j) cs → Bytes w' (L'.ar + UInt64.ofNat j) cs
  new : Bytes w' (L'.ar + UInt64.ofNat L.used) bs
  which : (L'.ar = L.ar ∧ L'.acap = L.acap) ∨ Past w L'.ar
  off : ∀ q n, InW w q n → Off L.ar L.acap q n → Off L'.ar L'.acap q n
  inw : ∀ q n, InW w q n → InW w' q n
  psize : (w.mem.region .pinned).size ≤ (w'.mem.region .pinned).size

/-- More heap appends nothing. -/
theorem Tab.grown {w w' : World} {s : UInt64} {T : Ht} {L : Lay} (h : Tab w s T L) (g : Grows w w') :
    Appended w w' s L L [] :=
  { e := rfl, ecap := rfl, ix := rfl, icap := rfl, used := rfl, offs := rfl, keeps := g.keeps _, calm := g.calm
    blk := fun y k _ hb => g.blk y k hb, whole := fun y k _ hb => g.whole y k hb, size := g.size
    far := g.fld h.far, fused := g.fld h.fused, facap := g.fld h.facap, wa := g.whole _ _ h.wa, le := h.room.2
    apos := h.pos.2.2, pre := fun _ _ _ hb => g.bytes _ _ hb, new := Or.inl rfl, which := Or.inl ⟨rfl, rfl⟩
    off := fun _ _ _ ho => ho, inw := fun _ _ hi => InW.grows g hi, psize := g.psize }

/-- The arena moved to a fresh block: its bytes copied, the old block freed,
    the header's fields set. -/
theorem Tab.regrown {w w1 w2 w3 : World} {s na : UInt64} {T : Ht} {L : Lay} {N : Nat} {cs0 : List UInt8}
    {m4 m5 : Mem} (h : Tab w s T L) (g1 : Grows w w1) (f : Fresh w w1 na N) (hN : L.used ≤ N) (hN0 : 0 < N)
    (hcs : Bytes w1 L.ar cs0) (hl : cs0.length = L.used) (W : Wrote w1 w2 na L.used) (B : Bytes w2 na cs0)
    (F : Freed w2 w3 L.ar) (hm4 : w3.mem.store (s + UInt64.ofNat ARENA) 8 na = some m4)
    (hm5 : (stored w3 (s + UInt64.ofNat ARENA) na m4).mem.store (s + UInt64.ofNat ACAP) 8 (UInt64.ofNat N) =
      some m5) :
    Appended w (stored (stored w3 (s + UInt64.ofNat ARENA) na m4) (s + UInt64.ofNat ACAP) (UInt64.ofNat N) m5) s L
      { L with ar := na, acap := N } [] := by
  have hsa : s ≠ L.ar := h.apart.2.2.1.ne (by decide) h.pos.2.2
  obtain ⟨M, wx, f1, f2, hfs, hby, -⟩ := moved (a := ARENA) (c := 24) h.rep.blk h.wa hsa g1 f hN W F
    ⟨Nat.le_refl _, by decide⟩ ⟨by decide, by decide⟩ (Or.inl (by decide)) (by decide) hm4 hm5 rfl
  refine {
    e := rfl
    ecap := rfl
    ix := rfl
    icap := rfl
    used := rfl
    offs := rfl
    keeps := M.keeps
    calm := M.calm
    blk := M.blk
    whole := M.whole
    size := M.size
    far := f1
    fused := hfs AUSED _ (by decide) (Or.inr (by decide)) (Or.inl (by decide)) h.fused
    facap := f2
    wa := wx
    le := hN
    apos := hN0
    pre := ?_
    new := Or.inl rfl
    which := Or.inr f.past
    off := fun q n hi _ => Past.off f.past hi _
    inw := M.inw
    psize := M.psize }
  intro j cs hj hb
  by_cases hc0 : cs = []
  · exact Or.inl hc0
  have hb1 := g1.bytes _ _ hb
  have hlt : ((cs0.drop j).take cs.length).length = cs.length := by simp; omega
  have e : cs = (cs0.drop j).take cs.length := hb1.unique (hcs.sub (j := j) (k := cs.length) (by omega)) hlt
  have hB : Bytes w2 (na + UInt64.ofNat j) cs := by
    rw [e]; exact B.sub (j := j) (k := cs.length) (by omega)
  exact hby j cs (by omega) hB

/-- Bytes copied to the arena's end, and the header's count of them set. -/
theorem Appended.fill {w w1 w2 : World} {s : UInt64} {T : Ht} {L L1 : Lay} {bs : List UInt8} {m : Mem}
    (h : Tab w s T L) (A : Appended w w1 s L L1 []) (hroom : L.used + bs.length ≤ L1.acap)
    (W : Wrote w1 w2 (L1.ar + UInt64.ofNat L.used) bs.length) (B : Bytes w2 (L1.ar + UInt64.ofNat L.used) bs)
    (hm : w2.mem.store (s + UInt64.ofNat AUSED) 8 (UInt64.ofNat (L.used + bs.length)) = some m) :
    Appended w (stored w2 (s + UInt64.ofNat AUSED) (UInt64.ofNat (L.used + bs.length)) m) s L
      { L1 with used := L.used + bs.length } bs := by
  have hsa : s ≠ L.ar := h.apart.2.2.1.ne (by decide) h.pos.2.2
  have a_sa : Apart s SIZE L1.ar L1.acap := by
    rcases A.which with ⟨e1, e2⟩ | hb
    · rw [e1, e2]; exact h.apart.2.2.1
    · exact Or.inl (hb.before h.rep.blk)
  have bs1 : Blk w1 s SIZE := A.blk _ _ hsa h.rep.blk
  have bs2 : Blk w2 s SIZE := W.stays.blk _ _ bs1
  have ba1 : Blk w1 L1.ar L1.acap := A.wa.blk
  have ba2 : Blk w2 L1.ar L1.acap := W.stays.blk _ _ ba1
  have W3 := wrote_stored hm
  have c_as : ∀ r off oy, decodeAddr (L1.ar + UInt64.ofNat L.used) = some (r, off) →
      decodeAddr s = some (.pinned, oy) → r ≠ .pinned ∨ off + bs.length ≤ oy ∨ oy + SIZE ≤ off :=
    a_sa.symm.clear ba1 hroom
  have c_sa : ∀ r off oy, decodeAddr (s + UInt64.ofNat AUSED) = some (r, off) →
      decodeAddr L1.ar = some (.pinned, oy) → r ≠ .pinned ∨ off + 8 ≤ oy ∨ oy + L1.acap ≤ off :=
    a_sa.clear bs2 (by decide)
  refine {
    e := A.e
    ecap := A.ecap
    ix := A.ix
    icap := A.icap
    used := rfl
    offs := A.offs
    keeps := ?_
    calm := ?_
    blk := ?_
    whole := ?_
    size := ?_
    far := ?_
    fused := fld_stored hm
    facap := ?_
    wa := W3.whole (W.whole A.wa)
    le := hroom
    apos := A.apos
    pre := ?_
    new := W3.bytes_blk ba2 hroom B c_sa
    which := A.which
    off := A.off
    inw := fun _ _ hi => InW.wrote W3 (InW.wrote W (A.inw _ _ hi))
    psize := Nat.le_trans A.psize (Nat.le_of_eq ((W.same.size .pinned).trans (W3.same.size .pinned))) }
  · exact (A.keeps.trans (W.keeps _ fun q n hn hp => Off.wrote ba1 hroom (A.off q n hp.1 hp.2.1))).trans
      (W3.keeps _ fun q n hn hp => Off.sub bs2 hp.2.2 (by decide) (by decide) (by decide) (by decide))
  · exact fun c => (W.stays.calm (A.calm c)).stored hm
  · exact fun y k hy hb => Blk.keep (W.stays.blk _ _ (A.blk y k hy hb)) hm
  · exact fun y k hy hb => W3.whole (W.whole (A.whole y k hy hb))
  · exact fun r hr => (W3.stays.size r hr).trans ((W.stays.size r hr).trans (A.size r hr))
  · exact (fld_kept bs2 (by decide) (by decide) (by decide) hm).trans ((W.fld_blk bs1 (by decide) c_as).trans A.far)
  · exact (fld_kept bs2 (by decide) (by decide) (by decide) hm).trans
      ((W.fld_blk bs1 (by decide) c_as).trans A.facap)
  · intro j cs hj hb
    by_cases hc0 : cs = []
    · exact Or.inl hc0
    have hcl := List.length_pos_iff.mpr hc0
    have h1 := A.pre j cs hj hb
    refine W3.bytes_blk ba2 (by omega) (W.bytes h1 fun hn r off r' off' hd hq => ?_) c_sa
    obtain ⟨o, -, hdj⟩ := Blk.decode ba1
    rw [hdj L.used (by omega)] at hd
    rw [hdj j (by omega)] at hq
    cases hd
    cases hq
    exact Or.inr (Or.inl (by omega))

/-- A table whose arena was appended to is the table, at the new layout. -/
theorem Tab.appended {w w' : World} {s : UInt64} {T : Ht} {L L' : Lay} {bs : List UInt8} (h : Tab w s T L)
    (A : Appended w w' s L L' bs) : Tab w' s T L' := by
  have hsa : s ≠ L.ar := h.apart.2.2.1.ne (by decide) h.pos.2.2
  have hr := h.room
  have hpos := h.pos
  have hE : 0 < L.ecap * ENTRY := Nat.mul_pos hpos.1 (by decide)
  have hI : 0 < L.icap * 8 := by omega
  have hea : L.e ≠ L.ar := h.apart.2.2.2.2.1.ne hE hpos.2.2
  have hia : L.ix ≠ L.ar := h.apart.2.2.2.2.2.ne hI hpos.2.2
  have hfs : ∀ j v, j + 8 ≤ ARENA → fld w s j = some v → fld w' s j = some v := fun j v hj hv =>
    A.keeps.load _ 8 v (by decide) ⟨Blk.inW h.rep.blk (Nat.le_trans hj (by decide)),
      h.apart.2.2.1.clear h.rep.blk (Nat.le_trans hj (by decide)),
      off_same h.rep.blk (by decide) (Nat.le_trans hj (by decide)) (Or.inl hj)⟩ hv
  have hfe : ∀ j v, j + 8 ≤ L.ecap * ENTRY → fld w L.e j = some v → fld w' L.e j = some v := fun j v hj hv =>
    A.keeps.load _ 8 v (by decide) ⟨Blk.inW h.be hj, h.apart.2.2.2.2.1.clear h.be hj,
      Off.part h.rep.blk (h.apart.1.symm.clear h.be hj) (by decide) (by decide)⟩ hv
  have hap : ∀ y k, Blk w y k → Apart y k L.ar L.acap → Apart y k L'.ar L'.acap := fun y k hb ha => by
    rcases A.which with ⟨e1, e2⟩ | hb'
    · rw [e1, e2]; exact ha
    · exact Or.inl (hb'.before hb)
  obtain ⟨e', ecap', ix', icap', ar', used', acap', offs'⟩ := L'
  have h1 : e' = L.e := A.e
  have h2 : ecap' = L.ecap := A.ecap
  have h3 : ix' = L.ix := A.ix
  have h4 : icap' = L.icap := A.icap
  have h5 : used' = L.used + bs.length := A.used
  have h6 : offs' = L.offs := A.offs
  subst h1; subst h2; subst h3; subst h4; subst h5; subst h6
  refine {
    rep := ⟨A.blk _ _ hsa h.rep.blk, A.calm h.rep.calm, hfs _ _ (by decide) h.rep.created,
      hfs _ _ (by decide) h.rep.count, h.rep.handles, h.rep.small⟩
    fe := hfs _ _ (by decide) h.fe
    fecap := hfs _ _ (by decide) h.fecap
    fix := hfs _ _ (by decide) h.fix
    ficap := hfs _ _ (by decide) h.ficap
    far := A.far
    fused := A.fused
    facap := A.facap
    we := A.whole _ _ hea h.we
    wi := A.whole _ _ hia h.wi
    wa := A.wa
    room := ⟨hr.1, A.le⟩
    pos := ⟨hpos.1, hpos.2.1, A.apos⟩
    apart := ⟨h.apart.1, h.apart.2.1, hap _ _ h.rep.blk h.apart.2.2.1, h.apart.2.2.2.1,
      hap _ _ h.be h.apart.2.2.2.2.1, hap _ _ h.bi h.apart.2.2.2.2.2⟩
    olen := h.olen
    sep := h.sep
    ento := fun i hi ho => ?_ }
  obtain ⟨h0, h8, h16, h24, hk, hv, bk, bv⟩ := h.ento i hi ho
  have hi32 : ∀ c, c + 8 ≤ 32 → i * ENTRY + c + 8 ≤ L.ecap * ENTRY := by
    intro c hc
    simp only [ENTRY]
    omega
  exact ⟨hfe _ _ (hi32 0 (by decide)) h0, hfe _ _ (hi32 8 (by decide)) h8, hfe _ _ (hi32 16 (by decide)) h16,
    hfe _ _ (hi32 24 (by decide)) h24, Nat.le_trans hk (Nat.le_add_right _ _), Nat.le_trans hv (Nat.le_add_right _ _),
    A.pre _ _ (by rw [size_toList]; exact hk) bk, A.pre _ _ (by rw [size_toList]; exact hv) bv⟩

/-- …and its index is the index. -/
theorem Idx.appended {w w' : World} {s : UInt64} {T : Ht} {L L' : Lay} {bs : List UInt8} {ks : List (List UInt8)}
    (hI : Idx w L ks) (h : Tab w s T L) (A : Appended w w' s L L' bs) : Idx w' L' ks := by
  have hc1 : L.icap < 2 ^ 64 := by have := hI.cap.2; omega
  have hfi : ∀ j v, j < L.icap → fld w L.ix (8 * j) = some v → fld w' L.ix (8 * j) = some v := fun j v hj hv =>
    A.keeps.load _ 8 v (by decide) ⟨Blk.inW h.bi (by omega), h.apart.2.2.2.2.2.clear h.bi (by omega),
      Off.part h.rep.blk (h.apart.2.1.symm.clear h.bi (by omega)) (by decide) (by decide)⟩ hv
  obtain ⟨e', ecap', ix', icap', ar', used', acap', offs'⟩ := L'
  have h3 : ix' = L.ix := A.ix
  have h4 : icap' = L.icap := A.icap
  subst h3; subst h4
  refine ⟨hI.cap, hI.small, hI.nodup, fun j hj => ?_, fun q hq => ?_⟩
  · obtain ⟨v, hv, hc⟩ := hI.slots j hj
    exact ⟨v, hfi j v hj hv, hc⟩
  · obtain ⟨t, ht, hbef⟩ := hI.chain q hq
    refine ⟨t, hfi _ _ (path_lt _ _ hI.cap.1 hc1) ht, fun t' ht' e => ?_⟩
    obtain ⟨v, hv, -⟩ := hI.slots _ (path_lt (H (ks[q]'hq)) t' hI.cap.1 hc1)
    rw [hfi _ _ (path_lt _ _ hI.cap.1 hc1) hv] at e
    exact hbef t' ht' (hv.trans e)

/-- What `append` answers and leaves: `-1` and more heap at most, or where the
    bytes went, with the arena they were appended to. -/
def AppQ (w : World) (s : UInt64) (L : Lay) (bs : List UInt8) (x : UInt64) (w' : World) : Prop :=
  (x = c64 (-1) ∧ Grows w w') ∨ (x = UInt64.ofNat L.used ∧ ∃ L', Appended w w' s L L' bs)

/-- What making room leaves: no room and more heap at most, or an arena with
    room, nothing appended yet. -/
def AppM (w : World) (s : UInt64) (L : Lay) (n : Nat) (x : UInt64) (w' : World) : Prop :=
  (x = 0 ∧ Grows w w') ∨ (x = c64 1 ∧ ∃ L1, Appended w w' s L L1 [] ∧ L.used + n ≤ L1.acap)

section fns
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- **`append`** of bytes that can be read and, if there are any, are off the
    arena and the header's arena fields: they are appended and their place answered, or `-1` is and
    nothing but the heap has changed. -/
theorem wp_append {s p n : Slot .i64} {sv pv : UInt64} {T : Ht} {L : Lay} {bs : List UInt8}
    {Q : Slot .i64 → Env → World → Prop}
    (hs : Γ[s]? = some (.sc .i64 sv)) (hp : Γ[p]? = some (.sc .i64 pv))
    (hn : Γ[n]? = some (.sc .i64 (UInt64.ofNat bs.length)))
    (h : Tab w sv T L) (hb : Bytes w pv bs) (ho : bs ≠ [] → Off L.ar L.acap pv bs.length)
    (hf : bs ≠ [] → Off (sv + UInt64.ofNat ARENA) 24 pv bs.length)
    (hk : ∀ Γ' w' r x, Ext Γ Γ' → Γ'[r]? = some (.sc .i64 x) → AppQ w sv L bs x w' → Q r Γ' w') :
    wp cfg d J (LibHt.append s p n) Q Γ w := by
  have hlen := hb.len_le
  have hacap := Blk.le36 h.ba
  have hr := h.room
  have hpos := h.pos
  have z0 : c64 0 = 0 := by decide
  have hu : (UInt64.ofNat L.used).toNat = L.used := by rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  have hl : (UInt64.ofNat bs.length).toNat = bs.length := by rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  have hsa : sv ≠ L.ar := h.apart.2.2.1.ne (by decide) h.pos.2.2
  unfold LibHt.append
  wp_seq
  refine wp_at (x := sv) (by slot) ?_
  refine wp_at (x := sv) (by slot) ?_
  refine wp_load64 (by slot) h.fused ?_
  refine wp_iadd (x := UInt64.ofNat L.used) (y := UInt64.ofNat bs.length) (by slot) (by slot) ?_
  rw [← UInt64.ofNat_add]
  apply wp_iconst64
  refine wp_load64 (by slot) h.facap ?_
  refine wp_ifteV (AppM w sv L bs.length) (x := UInt64.ofNat (L.used + bs.length)) (y := UInt64.ofNat L.acap)
    (by slot) (by slot) ?_ ?_ ?_
  · intro hle
    have hle' : L.used + bs.length ≤ L.acap :=
      (ofNat_le_iff (by omega) (by omega)).mp (by simpa [cmpInt] using hle)
    wp_seq
    apply wp_iconst64
    exact ⟨by ext_env, _, by slot, Or.inr ⟨rfl, L, h.grown (Grows.refl w), hle'⟩⟩
  · intro hgt ne hne
    have hgt' : ¬ (L.used + bs.length ≤ L.acap) := fun c =>
      absurd ((ofNat_le_iff (by omega) (by omega)).mpr c) (by simpa [cmpInt] using hgt)
    have hN : (UInt64.ofNat (max (2 * L.acap) (L.used + bs.length))).toNat =
        max (2 * L.acap) (L.used + bs.length) := by
      rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
    wp_seq
    refine wp_load64 (by slot) h.facap ?_
    refine wp_ishlImm (x := UInt64.ofNat L.acap) (by slot) ?_
    rw [shl1 (by omega)]
    refine wp_umax (x := UInt64.ofNat (2 * L.acap)) (y := UInt64.ofNat (L.used + bs.length)) (by slot) (by slot) ?_
    rw [umax_ofNat (by omega) (by omega)]
    refine wp_calloc (by slot) (fun e => by
      have := congrArg UInt64.toNat e
      rw [hN] at this
      simp at this
      omega) ?_
    intro x w1 g1 hx
    rw [hN] at hx
    refine wp_ifteV (AppM w sv L bs.length) (x := x) (y := c64 0) (by slot) (by slot) ?_ ?_ ?_
    · intro _
      wp_seq
      exact ⟨by ext_env, _, by slot, Or.inl ⟨z0, g1⟩⟩
    · intro hnz ne2 hne2
      have f := hx.resolve_left (by simpa [cmpInt, z0] using hnz)
      have c1 := g1.calm h.rep.calm
      obtain ⟨cs0, hcl, hcs⟩ := (Blk.room0 (g1.blk _ _ h.ba) c1 hr.2).bytes
      have a_na : Apart x (max (2 * L.acap) (L.used + bs.length)) L.ar L.acap := Or.inr (f.before h.ba)
      wp_seq
      refine wp_at (x := sv) (by slot) ?_
      refine wp_load64 (by slot) (g1.fld h.far) ?_
      refine wp_memcpy (bs := cs0) (by slot) (by slot) (by slot) hcs (by rw [hu]; exact hcl)
        (by rw [hu]; exact Blk.room0 f.blk c1 (by omega)) (by rw [hu]; exact a_na.num (by omega) hr.2) ?_
      intro w2 W2 B2 _
      rw [hu] at W2
      refine wp_free (by slot) (Or.inr (W2.stays.blk _ _ (g1.blk _ _ h.ba))) ?_
      intro w3 F
      have bs3 : Blk w3 sv SIZE := F.blk _ _ hsa (W2.stays.blk _ _ (g1.blk _ _ h.rep.blk))
      have c3 : Calm w3 := F.calm (W2.stays.calm c1)
      obtain ⟨m4, hm4⟩ := Blk.fld_store bs3 c3 (j := ARENA) (by decide) x
      refine wp_at (x := sv) (by slot) ?_
      refine wp_storeI64 (by slot) (by slot) hm4 ?_
      obtain ⟨m5, hm5⟩ := Blk.fld_store (Blk.keep bs3 hm4) (c3.stored hm4) (j := ACAP) (by decide)
        (UInt64.ofNat (max (2 * L.acap) (L.used + bs.length)))
      refine wp_storeI64 (by slot) (by slot) hm5 ?_
      apply wp_iconst64
      exact ⟨by ext_env, _, by slot, Or.inr ⟨rfl, _,
        h.regrown g1 f (by omega) (by omega) hcs hcl W2 B2 F hm4 hm5, Nat.le_max_right _ _⟩⟩
    · intro Γ5 w5 v nJ hE5 hv hr5
      exact ⟨Ext.trans (by ext_env) hE5, v, hv, hr5⟩
  · intro Γ6 w6 v nJ hE6 hv hr6
    simp only [carriesFrom, Vals.head]
    rcases hr6 with ⟨rfl, g⟩ | ⟨rfl, L1, A, hroom⟩
    · refine wp_ifteV (AppQ w sv L bs) (x := 0) (y := c64 0) (by slot) (by slot) ?_ ?_ ?_
      · intro _
        wp_seq
        apply wp_iconst64
        exact ⟨by ext_env, _, by slot, Or.inl ⟨rfl, g⟩⟩
      · intro hne
        exact absurd hne (by simp [cmpInt, z0])
      · intro Γ7 w7 y nJ7 hE7 hy hr7
        simp only [carriesFrom]
        exact hk _ _ _ _ (Ext.trans (Ext.trans (by ext_env) hE6) hE7) hy hr7
    · have hb1 : Bytes w6 pv bs := by
        by_cases e : bs = []
        · exact Or.inl e
        · exact A.keeps.bytes pv bs e ⟨hb.inW e, ho e, hf e⟩ hb
      have hap : (L1.ar + UInt64.ofNat L.used).toNat + bs.length ≤ pv.toNat ∨
          pv.toNat + bs.length ≤ (L1.ar + UInt64.ofNat L.used).toNat := by
        by_cases e : bs = []
        · subst e; simp; omega
        · exact apart_bytes hb1 A.wa.blk hroom (A.off _ _ (hb.inW e) (ho e))
      have c6 := A.calm h.rep.calm
      refine wp_ifteV (AppQ w sv L bs) (x := c64 1) (y := c64 0) (by slot) (by slot) ?_ ?_ ?_
      · intro he
        exact absurd he (by decide)
      · intro _ ne3 hne3
        wp_seq
        refine wp_at (x := sv) (by slot) ?_
        refine wp_load64 (by slot) A.far ?_
        refine wp_iadd (x := L1.ar) (y := UInt64.ofNat L.used) (by slot) (by slot) ?_
        refine wp_memcpy (bs := bs) (by slot) (by slot) (by slot) hb1 (by rw [hl])
          (by rw [hl]; exact Blk.room A.wa.blk c6 hroom) (by rw [hl]; exact hap) ?_
        intro w7 W7 B7 _
        rw [hl] at W7
        obtain ⟨m, hm⟩ := Blk.fld_store (W7.stays.blk _ _ (A.blk _ _ hsa h.rep.blk)) (W7.stays.calm c6)
          (j := AUSED) (by decide) (UInt64.ofNat (L.used + bs.length))
        refine wp_storeI64 (by slot) (by slot) hm ?_
        exact ⟨by ext_env, _, by slot, Or.inr ⟨rfl, _, A.fill h hroom W7 B7 hm⟩⟩
      · intro Γ7 w7 y nJ7 hE7 hy hr7
        simp only [carriesFrom]
        exact hk _ _ _ _ (Ext.trans (Ext.trans (by ext_env) hE6) hE7) hy hr7

end fns

-- ---------------------------------------------------------------------------
-- A new entry
-- ---------------------------------------------------------------------------

/-- A key no entry has, written into the empty slot its probe ends at, is one
    the index covers. -/
theorem Idx.insert {w w' : World} {L : Lay} {ks : List (List UInt8)} {kb : List UInt8} {i : UInt64} {t : Nat}
    (hI : Idx w L ks) (hnot : kb ∉ ks) (hsm : ks.length + 1 < 2 ^ 32)
    (hpath : i = path L.icap (H kb) t)
    (hbef : ∀ t', t' < t → fld w L.ix (8 * (path L.icap (H kb) t').toNat) ≠ some 0)
    (hempty : fld w L.ix (8 * i.toNat) = some 0)
    (hnew : fld w' L.ix (8 * i.toNat) = some (slotVal (H kb) ks.length))
    (hoth : ∀ j, j < L.icap → j ≠ i.toNat → fld w' L.ix (8 * j) = fld w L.ix (8 * j)) :
    Idx w' L (ks ++ [kb]) := by
  have hc0 := hI.cap.1
  have hc1 : L.icap < 2 ^ 64 := by have := hI.cap.2; omega
  have hlt : ∀ (h : UInt64) (t : Nat), (path L.icap h t).toNat < L.icap := fun h t => path_lt h t hc0 hc1
  have hne : ∀ j, fld w L.ix (8 * j) ≠ some 0 → j ≠ i.toNat := fun j hj e => hj (by rw [e]; exact hempty)
  refine ⟨hI.cap, by simpa using hsm, ?_, fun j hj => ?_, fun p hp => ?_⟩
  · simp only [List.nodup_append, hI.nodup, List.mem_singleton, true_and]
    refine ⟨by simp, ?_⟩
    rintro a ha b rfl rfl
    exact hnot ha
  · by_cases e : j = i.toNat
    · subst e
      exact ⟨_, hnew, Or.inr ⟨ks.length, by simp, by simp⟩⟩
    · rw [hoth j hj e]
      obtain ⟨v, hv, hc⟩ := hI.slots j hj
      refine ⟨v, hv, hc.imp id ?_⟩
      rintro ⟨p, hp, rfl⟩
      exact ⟨p, by simp; omega, by rw [List.getElem_append_left hp]⟩
  · by_cases c : p < ks.length
    · have ek : (ks ++ [kb])[p]'hp = ks[p]'c := List.getElem_append_left c
      rw [ek]
      obtain ⟨t0, ht0, hb0⟩ := hI.chain p c
      have hsv := slotVal_ne_zero (H (ks[p]'c)) (p := p) (by omega)
      refine ⟨t0, ?_, fun t' ht' => ?_⟩
      · rw [hoth _ (hlt _ _) (hne _ (by rw [ht0]; exact fun e => hsv (Option.some.inj e)))]
        exact ht0
      · by_cases e : (path L.icap (H (ks[p]'c)) t').toNat = i.toNat
        · rw [e, hnew]
          exact fun e' => slotVal_ne_zero (H kb) (p := ks.length) (by omega) (Option.some.inj e')
        · rw [hoth _ (hlt _ _) e]
          exact hb0 t' ht'
    · have hp' : p = ks.length := by simp at hp; omega
      subst hp'
      have ek : (ks ++ [kb])[ks.length]'hp = kb := by simp
      rw [ek]
      refine ⟨t, by rw [← hpath]; exact hnew, fun t' ht' => ?_⟩
      rw [hoth _ (hlt _ _) (hne _ (hbef t' ht'))]
      exact hbef t' ht'

/-- A table with one more entry is as separate, where the new key starts past
    every run there is and the new value past the new key. -/
theorem Sep.push {T : Ht} {offs : List (Nat × Nat)} {kv : ByteArray × ByteArray} {ko vo : Nat} (hs : Sep T offs)
    (hol : offs.length = T.entries.length)
    (hB : ∀ i (hi : i < T.entries.length) (ho : i < offs.length),
      (offs[i]).1 + (T.entries[i]).1.size ≤ ko ∧ (offs[i]).2 + (T.entries[i]).2.size ≤ ko)
    (hkv : ko + kv.1.size ≤ vo) : Sep { T with entries := T.entries ++ [kv] } (offs ++ [(ko, vo)]) := by
  have gE : ∀ i (c : i < T.entries.length) (hi : i < (T.entries ++ [kv]).length),
      (T.entries ++ [kv])[i]'hi = T.entries[i]'c := fun i c hi => List.getElem_append_left c
  have gO : ∀ i (c : i < offs.length) (ho : i < (offs ++ [(ko, vo)]).length),
      (offs ++ [(ko, vo)])[i]'ho = offs[i]'c := fun i c ho => List.getElem_append_left c
  have gEn : ∀ i (hi : i < (T.entries ++ [kv]).length), i = T.entries.length → (T.entries ++ [kv])[i]'hi = kv := by
    intro i hi e
    subst e
    simp
  have gOn : ∀ i (ho : i < (offs ++ [(ko, vo)]).length), i = T.entries.length →
      (offs ++ [(ko, vo)])[i]'ho = (ko, vo) := by
    intro i ho e
    have e' : i = offs.length := by omega
    subst e'
    simp
  intro i j hi hj oi oj
  have hi' : i < (T.entries ++ [kv]).length := hi
  have hj' : j < (T.entries ++ [kv]).length := hj
  have hi2 : i < T.entries.length + 1 := by simpa using hi'
  have hj2 : j < T.entries.length + 1 := by simpa using hj'
  show (((offs ++ [(ko, vo)])[i]'oi).2 + ((T.entries ++ [kv])[i]'hi').2.size ≤ ((offs ++ [(ko, vo)])[j]'oj).1 ∨
      ((offs ++ [(ko, vo)])[j]'oj).1 + ((T.entries ++ [kv])[j]'hj').1.size ≤ ((offs ++ [(ko, vo)])[i]'oi).2) ∧
    (i ≠ j → ((offs ++ [(ko, vo)])[i]'oi).2 + ((T.entries ++ [kv])[i]'hi').2.size ≤ ((offs ++ [(ko, vo)])[j]'oj).2 ∨
      ((offs ++ [(ko, vo)])[j]'oj).2 + ((T.entries ++ [kv])[j]'hj').2.size ≤ ((offs ++ [(ko, vo)])[i]'oi).2)
  by_cases ci : i < T.entries.length <;> by_cases cj : j < T.entries.length
  · rw [gE i ci, gO i (by omega), gE j cj, gO j (by omega)]
    exact hs i j ci cj _ _
  · rw [gE i ci, gO i (by omega), gEn j hj' (by omega), gOn j oj (by omega)]
    obtain ⟨a, b⟩ := hB i ci (by omega)
    exact ⟨Or.inl b, fun _ => Or.inl (by dsimp only; omega)⟩
  · rw [gE j cj, gO j (by omega), gEn i hi' (by omega), gOn i oi (by omega)]
    obtain ⟨a, b⟩ := hB j cj (by omega)
    exact ⟨Or.inr (by dsimp only; omega), fun _ => Or.inr (by dsimp only; omega)⟩
  · rw [gEn j hj' (by omega), gOn j oj (by omega), gEn i hi' (by omega), gOn i oi (by omega)]
    exact ⟨Or.inr hkv, fun hne => absurd (by omega) hne⟩

/-- An entry in place past the last, counted, is the table's next entry,
    where its key starts past every run there is and its value past its key. -/
theorem Tab.push {w : World} {s : UInt64} {T : Ht} {L : Lay} {kv : ByteArray × ByteArray} {ko vo : Nat} {m : Mem}
    (h : Tab w s T L) (hlt : T.entries.length < L.ecap) (hent : EntO w L T.entries.length kv ko vo)
    (hB : ∀ i (hi : i < T.entries.length) (ho : i < L.offs.length),
      (L.offs[i]).1 + (T.entries[i]).1.size ≤ ko ∧ (L.offs[i]).2 + (T.entries[i]).2.size ≤ ko)
    (hkv : ko + kv.1.size ≤ vo)
    (hm : w.mem.store (s + UInt64.ofNat COUNT) 8 (UInt64.ofNat (T.entries.length + 1)) = some m) :
    Tab (stored w (s + UInt64.ofNat COUNT) (UInt64.ofNat (T.entries.length + 1)) m) s
      { T with entries := T.entries ++ [kv] } { L with offs := L.offs ++ [(ko, vo)] } := by
  have W := wrote_stored hm
  have hr := h.room
  have h36 := Blk.le36 h.be
  have hol := h.olen
  have ce := h.apart.1.clear h.rep.blk (j := COUNT) (n := 8) (by decide)
  have ca := h.apart.2.2.1.clear h.rep.blk (j := COUNT) (n := 8) (by decide)
  have hf : ∀ j, j + 8 ≤ SIZE → (j + 8 ≤ COUNT ∨ COUNT + 8 ≤ j) →
      fld (stored w (s + UInt64.ofNat COUNT) (UInt64.ofNat (T.entries.length + 1)) m) s j = fld w s j :=
    fun j hj hc => fld_kept h.rep.blk hj (by decide) hc hm
  have hE : ∀ i kv' a b, i < L.ecap → EntO w L i kv' a b →
      EntO (stored w (s + UInt64.ofNat COUNT) (UInt64.ofNat (T.entries.length + 1)) m) L i kv' a b := by
    rintro i kv' a b hi ⟨h0, h8, h16, h24, hk, hv, bk, bv⟩
    have he : ∀ j, j + 8 ≤ ENTRY → fld (stored w (s + UInt64.ofNat COUNT) (UInt64.ofNat (T.entries.length + 1)) m)
        L.e (i * ENTRY + j) = fld w L.e (i * ENTRY + j) := fun j hj =>
      W.fld_blk h.be (by simp only [ENTRY] at hj ⊢; omega) ce
    exact ⟨(he 0 (by decide)).trans h0, (he 8 (by decide)).trans h8, (he 16 (by decide)).trans h16,
      (he 24 (by decide)).trans h24, hk, hv,
      W.bytes_blk h.ba (by rw [size_toList]; omega) bk ca, W.bytes_blk h.ba (by rw [size_toList]; omega) bv ca⟩
  have hlen : (T.entries ++ [kv]).length = T.entries.length + 1 := by simp
  refine {
    rep := ⟨Blk.keep h.rep.blk hm, h.rep.calm.stored hm, (hf _ (by decide) (Or.inl (by decide))).trans h.rep.created,
      ?_, h.rep.handles, h.rep.small.1, ?_⟩
    fe := (hf _ (by decide) (Or.inr (by decide))).trans h.fe
    fecap := (hf _ (by decide) (Or.inr (by decide))).trans h.fecap
    fix := (hf _ (by decide) (Or.inr (by decide))).trans h.fix
    ficap := (hf _ (by decide) (Or.inr (by decide))).trans h.ficap
    far := (hf _ (by decide) (Or.inr (by decide))).trans h.far
    fused := (hf _ (by decide) (Or.inr (by decide))).trans h.fused
    facap := (hf _ (by decide) (Or.inr (by decide))).trans h.facap
    we := W.whole h.we
    wi := W.whole h.wi
    wa := W.whole h.wa
    room := ⟨by show (T.entries ++ [kv]).length ≤ L.ecap; omega, hr.2⟩
    pos := h.pos
    apart := h.apart
    olen := by show (L.offs ++ [(ko, vo)]).length = (T.entries ++ [kv]).length; simp [hol]
    sep := h.sep.push hol hB hkv
    ento := fun i hi ho => ?_ }
  · show fld _ s COUNT = some (UInt64.ofNat (T.entries ++ [kv]).length)
    rw [hlen]
    exact fld_stored hm
  · show (T.entries ++ [kv]).length < 2 ^ 64
    simp only [ENTRY] at h36
    omega
  · have hi' : i < (T.entries ++ [kv]).length := hi
    have ho' : i < (L.offs ++ [(ko, vo)]).length := ho
    show EntO _ L i ((T.entries ++ [kv])[i]'hi') ((L.offs ++ [(ko, vo)])[i]'ho').1 ((L.offs ++ [(ko, vo)])[i]'ho').2
    by_cases c : i < T.entries.length
    · rw [List.getElem_append_left c, List.getElem_append_left (show i < L.offs.length by omega)]
      exact hE i _ _ _ (by omega) (h.ento i c (by omega))
    · have hi2 : i = L.offs.length := by rw [hlen] at hi'; omega
      subst hi2
      have e1 : (T.entries ++ [kv])[L.offs.length]'hi' = kv := by simp [hol]
      have e2 : (L.offs ++ [(ko, vo)])[L.offs.length]'ho' = (ko, vo) := by simp
      rw [e1, e2, hol]
      exact hE _ _ _ _ hlt hent

/-- Loads of the index a step keeps make it the index, at a layout with the
    same index. -/
theorem Idx.keeps {w w' : World} {L L' : Lay} {ks : List (List UInt8)} {P : UInt64 → Nat → Prop} (hI : Idx w L ks)
    (K : Keeps w w' P) (hP : ∀ j, j < L.icap → P (L.ix + UInt64.ofNat (8 * j)) 8) (hix : L'.ix = L.ix)
    (hic : L'.icap = L.icap) : Idx w' L' ks := by
  have hc1 : L.icap < 2 ^ 64 := by have := hI.cap.2; omega
  have hfi : ∀ j v, j < L.icap → fld w L.ix (8 * j) = some v → fld w' L.ix (8 * j) = some v := fun j v hj hv =>
    K.load _ 8 v (by decide) (hP j hj) hv
  obtain ⟨e', ecap', ix', icap', ar', used', acap', offs'⟩ := L'
  have h3 : ix' = L.ix := hix
  have h4 : icap' = L.icap := hic
  subst h3; subst h4
  refine ⟨hI.cap, hI.small, hI.nodup, fun j hj => ?_, fun q hq => ?_⟩
  · obtain ⟨v, hv, hc⟩ := hI.slots j hj
    exact ⟨v, hfi j v hj hv, hc⟩
  · obtain ⟨t, ht, hbef⟩ := hI.chain q hq
    refine ⟨t, hfi _ _ (path_lt _ _ hI.cap.1 hc1) ht, fun t' ht' e => ?_⟩
    obtain ⟨v, hv, -⟩ := hI.slots _ (path_lt (H (ks[q]'hq)) t' hI.cap.1 hc1)
    rw [hfi _ _ (path_lt _ _ hI.cap.1 hc1) hv] at e
    exact hbef t' ht' (hv.trans e)

-- ---------------------------------------------------------------------------
-- The entry array and the index moved
-- ---------------------------------------------------------------------------

/-- The entry array moved to a fresh block: the entries in use copied, the
    old block freed, the header's fields set. -/
theorem Tab.entGrown {w w1 w2 w3 w5 : World} {s ne : UInt64} {T : Ht} {L : Lay} {ks : List (List UInt8)}
    {cs0 : List UInt8} {m4 m5 : Mem} {N : Nat} (h : Tab w s T L) (hI : Idx w L ks) (g1 : Grows w w1)
    (f : Fresh w w1 ne (N * ENTRY)) (hN : T.entries.length ≤ N) (hN0 : 0 < N) (hcs : Bytes w1 L.e cs0)
    (hl : cs0.length = T.entries.length * ENTRY) (W : Wrote w1 w2 ne (T.entries.length * ENTRY))
    (B : Bytes w2 ne cs0) (F : Freed w2 w3 L.e) (hm4 : w3.mem.store (s + UInt64.ofNat ENTRIES) 8 ne = some m4)
    (hm5 : (stored w3 (s + UInt64.ofNat ENTRIES) ne m4).mem.store (s + UInt64.ofNat ECAP) 8 (UInt64.ofNat N) =
      some m5)
    (hw5 : w5 = stored (stored w3 (s + UInt64.ofNat ENTRIES) ne m4) (s + UInt64.ofNat ECAP) (UInt64.ofNat N) m5) :
    Tab w5 s T { L with e := ne, ecap := N } ∧ Idx w5 { L with e := ne, ecap := N } ks ∧
      Moved w w5 s ECAP 16 L.e (L.ecap * ENTRY) := by
  have hpos := h.pos
  have hr := h.room
  have hE : 0 < L.ecap * ENTRY := Nat.mul_pos hpos.1 (by decide)
  have hIx : 0 < L.icap * 8 := by omega
  have hse : s ≠ L.e := h.apart.1.ne (by decide) hE
  have hie : L.ix ≠ L.e := (h.apart.2.2.2.1.ne hE hIx).symm
  have hae : L.ar ≠ L.e := (h.apart.2.2.2.2.1.ne hE hpos.2.2).symm
  have hkN : T.entries.length * ENTRY ≤ N * ENTRY := Nat.mul_le_mul_right _ hN
  obtain ⟨M, wx, f1, f2, hfs, -, hfc⟩ := moved (a := ECAP) (c := 16) h.rep.blk h.we hse g1 f hkN W F
    ⟨by decide, by decide⟩ ⟨Nat.le_refl _, by decide⟩ (Or.inr (by decide)) (by decide) hm4 hm5 hw5
  have pa : ∀ j n, j + n ≤ L.acap → MovP w s ECAP 16 L.e (L.ecap * ENTRY) (L.ar + UInt64.ofNat j) n :=
    fun j n hj => movP_blk h.rep.blk h.ba h.apart.2.2.2.2.1.symm h.apart.2.2.1.symm (by decide) (by decide) hj
  have pi : ∀ j, j < L.icap → MovP w s ECAP 16 L.e (L.ecap * ENTRY) (L.ix + UInt64.ofNat (8 * j)) 8 :=
    fun j hj => movP_blk h.rep.blk h.bi h.apart.2.2.2.1.symm h.apart.2.1.symm (by decide) (by decide) (by omega)
  have hfe : ∀ j v, j + 8 ≤ T.entries.length * ENTRY → fld w L.e j = some v → fld w5 ne j = some v := by
    intro j v hj hv
    have h1' : fld w1 L.e j = some (leN cs0 j 8) := hcs.load (by omega) (by decide)
    have ev : v = leN cs0 j 8 := Option.some.inj ((g1.fld hv).symm.trans h1')
    have h2 : fld w2 ne j = some (leN cs0 j 8) := B.load (by omega) (by decide)
    rw [ev]
    exact hfc j _ (by omega) h2
  have hby : ∀ j cs, j + cs.length ≤ L.acap → Bytes w (L.ar + UInt64.ofNat j) cs →
      Bytes w5 (L.ar + UInt64.ofNat j) cs := by
    intro j cs hj hb
    by_cases c : cs = []
    · exact Or.inl c
    · exact M.keeps.bytes _ _ c (pa j _ hj) hb
  refine ⟨{
    rep := ⟨M.blk _ _ hse h.rep.blk, M.calm h.rep.calm,
      hfs _ _ (by decide) (Or.inl (by decide)) (Or.inl (by decide)) h.rep.created,
      hfs _ _ (by decide) (Or.inl (by decide)) (Or.inl (by decide)) h.rep.count, h.rep.handles, h.rep.small⟩
    fe := f1
    fecap := f2
    fix := hfs _ _ (by decide) (Or.inr (by decide)) (Or.inr (by decide)) h.fix
    ficap := hfs _ _ (by decide) (Or.inr (by decide)) (Or.inr (by decide)) h.ficap
    far := hfs _ _ (by decide) (Or.inr (by decide)) (Or.inr (by decide)) h.far
    fused := hfs _ _ (by decide) (Or.inr (by decide)) (Or.inr (by decide)) h.fused
    facap := hfs _ _ (by decide) (Or.inr (by decide)) (Or.inr (by decide)) h.facap
    we := wx
    wi := M.whole _ _ hie h.wi
    wa := M.whole _ _ hae h.wa
    room := ⟨hN, hr.2⟩
    pos := ⟨hN0, hpos.2.1, hpos.2.2⟩
    apart := ⟨Or.inl (f.before h.rep.blk), h.apart.2.1, h.apart.2.2.1, Or.inr (f.before h.bi),
      Or.inr (f.before h.ba), h.apart.2.2.2.2.2⟩
    olen := h.olen
    sep := h.sep
    ento := fun i hi ho => ?_ }, Idx.keeps hI M.keeps pi rfl rfl, M⟩
  obtain ⟨h0, h8, h16, h24, hk, hv, bk, bv⟩ := h.ento i hi ho
  have hi32 : ∀ c, c + 8 ≤ 32 → i * ENTRY + c + 8 ≤ T.entries.length * ENTRY := by
    intro c hc
    simp only [ENTRY]
    omega
  exact ⟨hfe _ _ (hi32 0 (by decide)) h0, hfe _ _ (hi32 8 (by decide)) h8, hfe _ _ (hi32 16 (by decide)) h16,
    hfe _ _ (hi32 24 (by decide)) h24, hk, hv, hby _ _ (by rw [size_toList]; exact Nat.le_trans hk hr.2) bk,
    hby _ _ (by rw [size_toList]; exact Nat.le_trans hv hr.2) bv⟩

/-- The index moved to a fresh block of empty slots: the old block freed, the
    header's fields set. -/
theorem Tab.reindexed {w w1 w3 w5 : World} {s nx : UInt64} {T : Ht} {L : Lay} {m4 m5 : Mem} {N : Nat}
    (h : Tab w s T L) (g1 : Grows w w1) (f : Fresh w w1 nx (N * 8)) (hN0 : 0 < N) (F : Freed w1 w3 L.ix)
    (hm4 : w3.mem.store (s + UInt64.ofNat INDEX) 8 nx = some m4)
    (hm5 : (stored w3 (s + UInt64.ofNat INDEX) nx m4).mem.store (s + UInt64.ofNat ICAP) 8 (UInt64.ofNat N) =
      some m5)
    (hw5 : w5 = stored (stored w3 (s + UInt64.ofNat INDEX) nx m4) (s + UInt64.ofNat ICAP) (UInt64.ofNat N) m5) :
    Tab w5 s T { L with ix := nx, icap := N } ∧ Idx w5 { L with ix := nx, icap := N } [] ∧
      Moved w w5 s ICAP 16 L.ix (L.icap * 8) := by
  have hpos := h.pos
  have hr := h.room
  have hE : 0 < L.ecap * ENTRY := Nat.mul_pos hpos.1 (by decide)
  have hIx : 0 < L.icap * 8 := by omega
  have hsi : s ≠ L.ix := h.apart.2.1.ne (by decide) hIx
  have hei : L.e ≠ L.ix := h.apart.2.2.2.1.ne hE hIx
  have hai : L.ar ≠ L.ix := (h.apart.2.2.2.2.2.ne hIx hpos.2.2).symm
  have hN36 := Blk.le36 f.blk
  obtain ⟨M, wx, f1, f2, hfs, -, hfc⟩ := moved (a := ICAP) (c := 16) (k := 0) h.rep.blk h.wi hsi g1 f
    (Nat.zero_le _) (Wrote.refl w1 nx 0) F ⟨by decide, by decide⟩ ⟨Nat.le_refl _, by decide⟩ (Or.inr (by decide))
    (by decide) hm4 hm5 hw5
  have pa : ∀ j n, j + n ≤ L.acap → MovP w s ICAP 16 L.ix (L.icap * 8) (L.ar + UInt64.ofNat j) n :=
    fun j n hj => movP_blk h.rep.blk h.ba h.apart.2.2.2.2.2.symm h.apart.2.2.1.symm (by decide) (by decide) hj
  have pe : ∀ j, j + 8 ≤ L.ecap * ENTRY → MovP w s ICAP 16 L.ix (L.icap * 8) (L.e + UInt64.ofNat j) 8 :=
    fun j hj => movP_blk h.rep.blk h.be h.apart.2.2.2.1 h.apart.1.symm (by decide) (by decide) hj
  have hby : ∀ j cs, j + cs.length ≤ L.acap → Bytes w (L.ar + UInt64.ofNat j) cs →
      Bytes w5 (L.ar + UInt64.ofNat j) cs := by
    intro j cs hj hb
    by_cases c : cs = []
    · exact Or.inl c
    · exact M.keeps.bytes _ _ c (pa j _ hj) hb
  refine ⟨{
    rep := ⟨M.blk _ _ hsi h.rep.blk, M.calm h.rep.calm,
      hfs _ _ (by decide) (Or.inl (by decide)) (Or.inl (by decide)) h.rep.created,
      hfs _ _ (by decide) (Or.inl (by decide)) (Or.inl (by decide)) h.rep.count, h.rep.handles, h.rep.small⟩
    fe := hfs _ _ (by decide) (Or.inl (by decide)) (Or.inl (by decide)) h.fe
    fecap := hfs _ _ (by decide) (Or.inl (by decide)) (Or.inl (by decide)) h.fecap
    fix := f1
    ficap := f2
    far := hfs _ _ (by decide) (Or.inr (by decide)) (Or.inr (by decide)) h.far
    fused := hfs _ _ (by decide) (Or.inr (by decide)) (Or.inr (by decide)) h.fused
    facap := hfs _ _ (by decide) (Or.inr (by decide)) (Or.inr (by decide)) h.facap
    we := M.whole _ _ hei h.we
    wi := wx
    wa := M.whole _ _ hai h.wa
    room := hr
    pos := ⟨hpos.1, hN0, hpos.2.2⟩
    apart := ⟨h.apart.1, Or.inl (f.before h.rep.blk), h.apart.2.2.1, Or.inl (f.before h.be),
      h.apart.2.2.2.2.1, Or.inr (f.before h.ba)⟩
    olen := h.olen
    sep := h.sep
    ento := fun i hi ho => ?_ }, ⟨⟨hN0, show N ≤ 2 ^ 33 by omega⟩, by simp, List.nodup_nil, fun j hj => ?_, fun p hp => ?_⟩, M⟩
  · obtain ⟨h0, h8, h16, h24, hk, hv, bk, bv⟩ := h.ento i hi ho
    have hi32 : ∀ c, c + 8 ≤ 32 → i * ENTRY + c + 8 ≤ L.ecap * ENTRY := by
      intro c hc
      simp only [ENTRY]
      omega
    have hfe : ∀ j v, j + 8 ≤ L.ecap * ENTRY → fld w L.e j = some v → fld w5 L.e j = some v := fun j v hj hv =>
      M.keeps.load _ 8 v (by decide) (pe j hj) hv
    exact ⟨hfe _ _ (hi32 0 (by decide)) h0, hfe _ _ (hi32 8 (by decide)) h8,
      hfe _ _ (hi32 16 (by decide)) h16, hfe _ _ (hi32 24 (by decide)) h24, hk, hv,
      hby _ _ (by rw [size_toList]; exact Nat.le_trans hk hr.2) bk, hby _ _ (by rw [size_toList]; exact Nat.le_trans hv hr.2) bv⟩
  · have hj' : j < N := hj
    exact ⟨0, hfc (8 * j) 0 (by omega) (f.zero h.rep.calm (8 * j) (by omega)), Or.inl rfl⟩
  · exact absurd hp (by simp)

-- ---------------------------------------------------------------------------
-- What a change to the table keeps
-- ---------------------------------------------------------------------------

/-- What a change to the table protects: a run inside the memory there was,
    clear of the table. -/
def TabP (w : World) (s : UInt64) (L : Lay) (q : UInt64) (n : Nat) : Prop := InW w q n ∧ Clear s L q n

/-- **`w'` is `w` with the table changed and nothing else**: every run clear
    of the table is kept, every block that is not the table's is there, memory
    has not shrunk, and each of the table's blocks is one it had or starts
    past the memory there was. -/
structure Kept (w w' : World) (s : UInt64) (L L' : Lay) : Prop where
  keeps : Keeps w w' (TabP w s L)
  blk : ∀ y k, (∀ b ∈ blocks s L, y ≠ b.1) → Blk w y k → Blk w' y k
  whole : ∀ y k, (∀ b ∈ blocks s L, y ≠ b.1) → Whole w y k → Whole w' y k
  size : ∀ r, r ≠ .pinned → (w'.mem.region r).size = (w.mem.region r).size
  inw : ∀ q n, InW w q n → InW w' q n
  psize : (w.mem.region .pinned).size ≤ (w'.mem.region .pinned).size
  fresh : ∀ b ∈ blocks s L', b ∈ blocks s L ∨ Past w b.1

theorem Kept.refl (w : World) (s : UInt64) (L : Lay) : Kept w w s L L :=
  ⟨Keeps.refl _ _, fun _ _ _ h => h, fun _ _ _ h => h, fun _ _ => rfl, fun _ _ h => h, Nat.le_refl _,
    fun _ h => Or.inl h⟩

/-- What was clear of the table is clear of the table that is left. -/
theorem Kept.clear {w w' : World} {s : UInt64} {L L' : Lay} (K : Kept w w' s L L') {q : UInt64} {n : Nat}
    (hi : InW w q n) (hc : Clear s L q n) : Clear s L' q n :=
  fun y k hb => (K.fresh (y, k) hb).elim (hc y k) fun hp => Past.off hp hi k

theorem Kept.trans {w w' w'' : World} {s : UInt64} {L L' L'' : Lay} (K1 : Kept w w' s L L')
    (K2 : Kept w' w'' s L' L'') : Kept w w'' s L L'' := by
  have hne : ∀ y k, (∀ b ∈ blocks s L, y ≠ b.1) → Blk w y k → ∀ b ∈ blocks s L', y ≠ b.1 :=
    fun y k hy hb b hb' => (K1.fresh b hb').elim (hy b) fun hp => hp.ne hb
  refine ⟨K1.keeps.next K2.keeps fun q n _ hp hi' => ⟨hi', K1.clear hp.1 hp.2⟩,
    fun y k hy hb => K2.blk y k (hne y k hy hb) (K1.blk y k hy hb),
    fun y k hy hb => K2.whole y k (hne y k hy hb.blk) (K1.whole y k hy hb),
    fun r hr => (K2.size r hr).trans (K1.size r hr), fun q n hi => K2.inw q n (K1.inw q n hi),
    Nat.le_trans K1.psize K2.psize, fun b hb => ?_⟩
  rcases K2.fresh b hb with h | h
  · exact K1.fresh b h
  · exact Or.inr (h.mono K1.psize)

/-- A write inside one of the table's blocks. -/
theorem Kept.wrote {w w' : World} {s y : UInt64} {L : Lay} {k j m : Nat} (hb : (y, k) ∈ blocks s L)
    (hy : Blk w y k) (W : Wrote w w' (y + UInt64.ofNat j) m) (hj : j + m ≤ k) : Kept w w' s L L :=
  ⟨W.keeps _ fun _ _ _ hp => Off.wrote hy hj (hp.2 y k hb), fun _ _ _ h => W.stays.blk _ _ h,
    fun _ _ _ h => W.whole h, W.stays.size, fun _ _ hi => InW.wrote W hi, Nat.le_of_eq (W.same.size .pinned),
    fun _ h => Or.inl h⟩

/-- One of the table's blocks moved. -/
theorem Moved.kept {w w' : World} {s old : UInt64} {L L' : Lay} {a c oc : Nat} (hs : Blk w s SIZE)
    (M : Moved w w' s a c old oc) (hold : (old, oc) ∈ blocks s L) (hac : a + c ≤ SIZE) (ha : a < SIZE)
    (hfr : ∀ b ∈ blocks s L', b ∈ blocks s L ∨ Past w b.1) : Kept w w' s L L' :=
  ⟨M.keeps.mono fun q n hp => ⟨hp.1, hp.2 old oc hold, Off.part hs (hp.2 s SIZE (by simp [blocks])) hac ha⟩,
    fun y k hy hb => M.blk y k (hy _ hold) hb, fun y k hy hb => M.whole y k (hy _ hold) hb, M.size, M.inw,
    M.psize, hfr⟩

/-- The arena appended to. -/
theorem Appended.kept {w w' : World} {s : UInt64} {T : Ht} {L L' : Lay} {bs : List UInt8} (h : Tab w s T L)
    (A : Appended w w' s L L' bs) : Kept w w' s L L' := by
  have har : (L.ar, L.acap) ∈ blocks s L := by simp [blocks]
  refine ⟨A.keeps.mono fun q n hp => ⟨hp.1, hp.2 _ _ har,
      Off.part h.rep.blk (hp.2 s SIZE (by simp [blocks])) (by decide) (by decide)⟩,
    fun y k hy hb => A.blk y k (hy _ har) hb, fun y k hy hb => A.whole y k (hy _ har) hb, A.size, A.inw, A.psize,
    fun b hb => ?_⟩
  simp only [blocks, List.mem_cons, List.not_mem_nil, or_false, A.e, A.ecap, A.ix, A.icap] at hb ⊢
  rcases hb with rfl | rfl | rfl | rfl
  · exact Or.inl (Or.inl rfl)
  · exact Or.inl (Or.inr (Or.inl rfl))
  · exact Or.inl (Or.inr (Or.inr (Or.inl rfl)))
  · rcases A.which with ⟨e1, e2⟩ | hp
    · exact Or.inl (Or.inr (Or.inr (Or.inr (by rw [e1, e2]))))
    · exact Or.inr hp

/-- The frame names the table's blocks alone. -/
theorem Kept.lay {w w' : World} {s : UInt64} {L L' L'' : Lay} (K : Kept w w' s L L')
    (hb : blocks s L'' = blocks s L') : Kept w w' s L L'' :=
  ⟨K.keeps, K.blk, K.whole, K.size, K.inw, K.psize, fun b hb' => K.fresh b (hb ▸ hb')⟩

/-- The index is the index at every layout with the same index block. -/
theorem Idx.lay {w : World} {L L' : Lay} {ks : List (List UInt8)} (hI : Idx w L ks) (hix : L'.ix = L.ix)
    (hic : L'.icap = L.icap) : Idx w L' ks :=
  Idx.keeps hI (Keeps.refl w fun _ _ => True) (fun _ _ => trivial) hix hic

/-- A table whose index covers its first `i` keys is one a probe can read. -/
theorem Tab.probeTake {w : World} {s : UInt64} {T : Ht} {L : Lay} {i : Nat} (h : Tab w s T L)
    (hI : Idx w L ((keysOf T).take i)) : Probe w s L ((keysOf T).take i) := by
  refine ⟨h.ficap, h.fix, h.fe, h.far, hI.cap, hI.small, fun p hp => ?_, hI.slots, hI.chain⟩
  have hp' : p < T.entries.length := by
    have : p < min i (keysOf T).length := by simpa using hp
    have hl : (keysOf T).length = T.entries.length := by simp [keysOf]
    omega
  obtain ⟨ko, vo, f0, f8, -, -, -, -, bk, -⟩ := h.ent p hp'
  refine ⟨ko, f0, ?_, ?_⟩
  · simpa [keysOf, size_toList] using f8
  · simpa [keysOf] using bk

section fns
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- What `grow` leaves: the table and its index, at some layout. -/
def GrowQ (w : World) (s : UInt64) (T : Ht) (L : Lay) (w' : World) : Prop :=
  ∃ L', Tab w' s T L' ∧ Idx w' L' (keysOf T) ∧ Kept w w' s L L'

set_option maxHeartbeats 1600000 in
/-- **`grow`** rebuilds the index at twice its size, or leaves it where the
    new one cannot be allocated: the table and its index, either way. -/
theorem wp_grow {s : Slot .i64} {sv : UInt64} {T : Ht} {L : Lay} {Q : Unit → Env → World → Prop}
    (hs : Γ[s]? = some (.sc .i64 sv)) (h : Tab w sv T L) (hI : Idx w L (keysOf T))
    (hk : ∀ Γ' w', Ext Γ Γ' → GrowQ w sv T L w' → Q () Γ' w') :
    wp cfg d J (LibHt.grow s) Q Γ w := by
  have hpos := h.pos
  have hr := h.room
  have hi36 := Blk.le36 h.bi
  have he36 := Blk.le36 h.be
  have hlen : (keysOf T).length = T.entries.length := by simp [keysOf]
  have z0 : c64 0 = 0 := by decide
  have hN : (UInt64.ofNat (8 * (2 * L.icap))).toNat = 2 * L.icap * 8 := by
    rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]; omega
  have hcnt : (UInt64.ofNat T.entries.length).toNat = T.entries.length := by
    rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt h.rep.small.2]
  unfold LibHt.grow keyAt
  wp_seq
  refine wp_at (x := sv) (by slot) ?_
  refine wp_load64 (by slot) h.ficap ?_
  refine wp_ishlImm (x := UInt64.ofNat L.icap) (by slot) ?_
  rw [shl1 (by omega)]
  refine wp_ishlImm (x := UInt64.ofNat (2 * L.icap)) (by slot) ?_
  rw [shl3 (by omega)]
  refine wp_calloc (by slot) (fun e => by
    have := congrArg UInt64.toNat e
    rw [hN] at this
    simp at this
    omega) ?_
  intro x w1 g1 hx
  rw [hN] at hx
  apply wp_iconst64
  refine wp_when (GrowQ w sv T L) (x := x) (y := c64 0) (by slot) (by slot) ?_ ?_ ?_
  · intro hnz
    have f := hx.resolve_left (by simpa [cmpInt, z0] using hnz)
    have c1 := g1.calm h.rep.calm
    have hsi : sv ≠ L.ix := h.apart.2.1.ne (by decide) (by omega)
    wp_seq
    refine wp_at (x := sv) (by slot) ?_
    refine wp_load64 (by slot) (g1.fld h.fix) ?_
    refine wp_free (by slot) (Or.inr (g1.blk _ _ h.bi)) ?_
    intro w3 F
    have bs3 : Blk w3 sv SIZE := F.blk _ _ hsi (g1.blk _ _ h.rep.blk)
    have c3 : Calm w3 := F.calm c1
    obtain ⟨m4, hm4⟩ := Blk.fld_store bs3 c3 (j := INDEX) (by decide) x
    refine wp_at (x := sv) (by slot) ?_
    refine wp_storeI64 (by slot) (by slot) hm4 ?_
    obtain ⟨m5, hm5⟩ := Blk.fld_store (Blk.keep bs3 hm4) (c3.stored hm4) (j := ICAP) (by decide)
      (UInt64.ofNat (2 * L.icap))
    refine wp_storeI64 (by slot) (by slot) hm5 ?_
    obtain ⟨hT5, hI5, M⟩ := h.reindexed g1 f (by omega) F hm4 hm5 rfl
    have K5 : Kept w (stored (stored w3 (sv + UInt64.ofNat INDEX) x m4) (sv + UInt64.ofNat ICAP)
        (UInt64.ofNat (2 * L.icap)) m5) sv L { L with ix := x, icap := 2 * L.icap } := by
      refine Moved.kept h.rep.blk M (by simp [blocks]) (by decide) (by decide) fun b hb => ?_
      simp only [blocks, List.mem_cons, List.not_mem_nil, or_false] at hb ⊢
      rcases hb with rfl | rfl | rfl | rfl
      · exact Or.inl (Or.inl rfl)
      · exact Or.inl (Or.inr (Or.inl rfl))
      · exact Or.inr f.past
      · exact Or.inl (Or.inr (Or.inr (Or.inr rfl)))
    generalize stored (stored w3 (sv + UInt64.ofNat INDEX) x m4) (sv + UInt64.ofNat ICAP)
      (UInt64.ofNat (2 * L.icap)) m5 = w5 at hT5 hI5 K5 M ⊢
    generalize ({ L with ix := x, icap := 2 * L.icap } : Lay) = L' at hT5 hI5 K5 ⊢
    refine wp_at (x := sv) (by slot) ?_
    refine wp_load64 (by slot) hT5.fe ?_
    refine wp_at (x := sv) (by slot) ?_
    refine wp_load64 (by slot) hT5.rep.count ?_
    refine wp_forLoop (fun i wi => Tab wi sv T L' ∧ Idx wi L' ((keysOf T).take i) ∧ Kept w wi sv L L') (by slot)
      ⟨hT5, by simpa using hI5, K5⟩ ?_ ?_
    · intro J' Γ1 wi i si hE1 hi hinv hsi'
      obtain ⟨hT, hIi, hK⟩ := hinv
      rw [hcnt] at hi
      have hki : i < (keysOf T).length := by omega
      obtain ⟨ko, vo, f0, f8, -, -, hko, -, bk, -⟩ := hT.ent i hi
      have hkb : (keysOf T)[i]'hki = (T.entries[i]'hi).1.data.toList := by simp [keysOf]
      have hnot : (T.entries[i]'hi).1.data.toList ∉ (keysOf T).take i := by
        rw [← hkb]; exact not_mem_take_of_nodup hI.nodup hki
      have hbi := hT.bi
      have hi36' := Blk.le36 hbi
      have ht : ((keysOf T).take i).length = i := by simp; omega
      have eA : L'.e + UInt64.ofNat i * c64 (ENTRY : Nat) = L'.e + UInt64.ofNat (i * ENTRY) := by
        rw [show c64 (ENTRY : Nat) = UInt64.ofNat ENTRY from by decide, ← UInt64.ofNat_mul]
      have e8 : L'.e + UInt64.ofNat (i * ENTRY) + c64 8 = L'.e + UInt64.ofNat (i * ENTRY + 8) := by
        rw [UInt64.add_assoc, show c64 8 = UInt64.ofNat 8 from by decide, ← UInt64.ofNat_add]
      wp_seq
      apply wp_iconst64
      refine wp_imul (x := UInt64.ofNat i) (by slot) (by slot) ?_
      refine wp_iadd (x := L'.e) (by slot) (by slot) ?_
      rw [eA]
      refine wp_iaddImm (x := L'.e + UInt64.ofNat (i * ENTRY)) (by slot) ?_
      refine wp_load64 (by slot) (by rw [e8]; exact f8) ?_
      refine wp_at (x := sv) (by slot) ?_
      refine wp_load64 (by slot) hT.far ?_
      refine wp_load64 (by slot) f0 ?_
      refine wp_iadd (x := L'.ar) (y := UInt64.ofNat ko) (by slot) (by slot) ?_
      refine wp_hash (bs := (T.entries[i]'hi).1.data.toList) (by slot) (by rw [size_toList]; slot) bk ?_
      intro Γ2 r hE2 hr2
      refine wp_find (kb := (T.entries[i]'hi).1.data.toList) (L := L') (ks := (keysOf T).take i) (by slot) (by slot)
        (by rw [size_toList]; slot) hr2 (hT.probeTake hIi) bk ?_
      intro Γ3 ra rp slotA pos hE3 hra hrp hF
      obtain ⟨ii, hii, hsl, hF⟩ := hF
      have hF' : fld wi L'.ix (8 * ii.toNat) = some 0 ∧ ∃ t,
          ii = path L'.icap (H (T.entries[i]'hi).1.data.toList) t ∧ ∀ t', t' < t →
            fld wi L'.ix (8 * (path L'.icap (H (T.entries[i]'hi).1.data.toList) t').toNat) ≠ some 0 := by
        rcases hF with ⟨-, hemp, -, t, hpath, hbef⟩ | ⟨p, hp, -, hkey⟩
        · exact ⟨hemp, t, hpath, hbef⟩
        · exact absurd (hkey ▸ List.getElem_mem hp) hnot
      obtain ⟨hemp, t, hpath, hbef⟩ := hF'
      subst hsl
      have hv : ((H (T.entries[i]'hi).1.data.toList >>> (c64 32 % 64)) <<< (c64 32 % 64)) |||
          (UInt64.ofNat i + c64 1) = slotVal (H (T.entries[i]'hi).1.data.toList) i := by
        unfold slotVal
        rw [show c64 1 = UInt64.ofNat 1 from by decide, ← UInt64.ofNat_add]
      obtain ⟨m, hm⟩ := Blk.fld_store hbi hT.rep.calm (j := 8 * ii.toNat) (by omega)
        (slotVal (H (T.entries[i]'hi).1.data.toList) i)
      refine wp_ushrImm (x := H (T.entries[i]'hi).1.data.toList) (by slot) ?_
      refine wp_ishlImm (by slot) ?_
      refine wp_iaddImm (x := UInt64.ofNat i) (by slot) ?_
      refine wp_bor (by slot) (by slot) ?_
      rw [hv]
      refine wp_storeI64 (by slot) (by slot) hm ?_
      have W := wrote_stored hm
      have hj : 8 * ii.toNat + 8 ≤ L'.icap * 8 := by omega
      refine ⟨Ext.trans (Ext.trans (Ext.trans (by ext_env) hE2) hE3) (by ext_env), ?_, ?_, ?_⟩
      · exact hT.wroteOff W (hT.apart.2.1.symm.clear hbi hj)
          (Off.prefix (hT.apart.2.2.2.1.symm.clear hbi hj) (Nat.mul_le_mul_right _ hT.room.1))
          (Off.prefix (hT.apart.2.2.2.2.2.clear hbi hj) hT.room.2)
      · rw [← List.take_append_getElem hki, hkb]
        have hsm : i + 1 < 2 ^ 32 := by
          have h1 := hT.room.1
          have h2 := Blk.le36 hT.be
          simp only [ENTRY] at h2
          omega
        refine Idx.insert hIi hnot (by rw [ht]; exact hsm) hpath hbef hemp ?_ ?_
        · rw [ht]
          exact load_store8 hm
        · intro j hj' hne
          exact fld_kept hbi (i := 8 * j) (j := 8 * ii.toNat) (by omega) (by omega) (by omega) hm
      · exact hK.trans (Kept.wrote (y := L'.ix) (k := L'.icap * 8) (by simp [blocks]) hbi W hj)
    · intro Γ9 w9 hE9 hinv
      obtain ⟨hT, hIi, hK⟩ := hinv
      rw [hcnt, ← hlen, List.take_length] at hIi
      exact ⟨Ext.trans (by ext_env) hE9, L', hT, hIi, hK⟩
  · intro _
    have A := h.grown g1
    exact ⟨L, h.appended A, hI.appended h A, A.kept h⟩
  · intro Γ' w' hE hR
    exact hk _ _ (Ext.trans (by ext_env) hE) hR

end fns

-- ---------------------------------------------------------------------------
-- Inserting
-- ---------------------------------------------------------------------------

/-- **The index covers `ks`, and slot `i` is the empty one the probe for `kb`
    ends at**, after `t` steps past slots that are not empty. -/
structure Spot (w : World) (L : Lay) (ks : List (List UInt8)) (kb : List UInt8) (i : UInt64) (t : Nat) : Prop where
  idx : Idx w L ks
  lt : i.toNat < L.icap
  emp : fld w L.ix (8 * i.toNat) = some 0
  on : i = path L.icap (H kb) t
  bef : ∀ t', t' < t → fld w L.ix (8 * (path L.icap (H kb) t').toNat) ≠ some 0

theorem Spot.keeps {w w' : World} {L L' : Lay} {ks : List (List UInt8)} {kb : List UInt8} {i : UInt64} {t : Nat}
    {P : UInt64 → Nat → Prop} (S : Spot w L ks kb i t) (K : Keeps w w' P)
    (hP : ∀ j, j < L.icap → P (L.ix + UInt64.ofNat (8 * j)) 8) (hix : L'.ix = L.ix) (hic : L'.icap = L.icap) :
    Spot w' L' ks kb i t := by
  have hI := S.idx
  have hc1 : L.icap < 2 ^ 64 := by have := hI.cap.2; omega
  have hfi : ∀ j v, j < L.icap → fld w L.ix (8 * j) = some v → fld w' L.ix (8 * j) = some v := fun j v hj hv =>
    K.load _ 8 v (by decide) (hP j hj) hv
  have hI' : Idx w' L' ks := Idx.keeps hI K hP hix hic
  obtain ⟨e', ecap', ix', icap', ar', used', acap', offs'⟩ := L'
  have h3 : ix' = L.ix := hix
  have h4 : icap' = L.icap := hic
  subst h3; subst h4
  refine ⟨hI', S.lt, hfi _ _ S.lt S.emp, S.on, fun t' ht' e => ?_⟩
  obtain ⟨v, hv, -⟩ := hI.slots _ (path_lt (H kb) t' hI.cap.1 hc1)
  rw [hfi _ _ (path_lt _ _ hI.cap.1 hc1) hv] at e
  exact S.bef t' ht' (hv.trans e)

theorem Spot.wrote {w w' : World} {L : Lay} {ks : List (List UInt8)} {kb : List UInt8} {i : UInt64} {t : Nat}
    {p : UInt64} {n : Nat} (S : Spot w L ks kb i t) (hb : Blk w L.ix (L.icap * 8)) (W : Wrote w w' p n)
    (hc : Off L.ix (L.icap * 8) p n) : Spot w' L ks kb i t := by
  have hc1 : L.icap < 2 ^ 64 := by have := S.idx.cap.2; omega
  have hf : ∀ j, j < L.icap → fld w' L.ix (8 * j) = fld w L.ix (8 * j) := fun j hj => W.fld_blk hb (by omega) hc
  refine ⟨S.idx.wrote hb W hc, S.lt, (hf _ S.lt).trans S.emp, S.on, fun t' ht' => ?_⟩
  rw [hf _ (path_lt _ _ S.idx.cap.1 hc1)]
  exact S.bef t' ht'

/-- One field of the entry past the last, stored: the table and the probe's
    slot are as they were, the field holds what was stored, and the entry's
    other fields and the arena hold what they held. -/
theorem Tab.entStore {w : World} {s : UInt64} {T : Ht} {L : Lay} {ks : List (List UInt8)} {kb : List UInt8}
    {i : UInt64} {t c : Nat} {v : UInt64} {m : Mem} (h : Tab w s T L) (S : Spot w L ks kb i t)
    (hlt : T.entries.length < L.ecap) (hc : c + 8 ≤ 32)
    (hm : w.mem.store (L.e + UInt64.ofNat (T.entries.length * ENTRY + c)) 8 v = some m) :
    Tab (stored w (L.e + UInt64.ofNat (T.entries.length * ENTRY + c)) v m) s T L ∧
      Spot (stored w (L.e + UInt64.ofNat (T.entries.length * ENTRY + c)) v m) L ks kb i t ∧
      Kept w (stored w (L.e + UInt64.ofNat (T.entries.length * ENTRY + c)) v m) s L L ∧
      fld (stored w (L.e + UInt64.ofNat (T.entries.length * ENTRY + c)) v m) L.e (T.entries.length * ENTRY + c) =
        some v ∧
      (∀ c' v', c' + 8 ≤ 32 → (c' + 8 ≤ c ∨ c + 8 ≤ c') → fld w L.e (T.entries.length * ENTRY + c') = some v' →
        fld (stored w (L.e + UInt64.ofNat (T.entries.length * ENTRY + c)) v m) L.e (T.entries.length * ENTRY + c') =
          some v') ∧
      (∀ j cs, j + cs.length ≤ L.acap → Bytes w (L.ar + UInt64.ofNat j) cs →
        Bytes (stored w (L.e + UInt64.ofNat (T.entries.length * ENTRY + c)) v m) (L.ar + UInt64.ofNat j) cs) := by
  have W := wrote_stored hm
  have hjc : ∀ c, c + 8 ≤ 32 → T.entries.length * ENTRY + c + 8 ≤ L.ecap * ENTRY := by
    intro c hc
    simp only [ENTRY]
    omega
  have hj := hjc c hc
  refine ⟨h.wroteOff W (h.apart.1.symm.clear h.be hj) (off_prefix h.be hj (Nat.le_add_right _ _))
      (Off.prefix (h.apart.2.2.2.2.1.clear h.be hj) h.room.2),
    S.wrote h.bi W (h.apart.2.2.2.1.clear h.be hj),
    Kept.wrote (y := L.e) (k := L.ecap * ENTRY) (by simp [blocks]) h.be W hj, fld_stored hm,
    fun c' v' hc' hd hv => (fld_kept h.be (hjc c' hc') hj (by omega) hm).trans hv,
    fun j cs hj' hb => W.bytes_blk h.ba hj' hb (h.apart.2.2.2.2.1.clear h.be hj)⟩

/-- The table with one more entry. -/
def pushed (T : Ht) (kb vb : List UInt8) : Ht :=
  { T with entries := T.entries ++ [(ByteArray.mk kb.toArray, ByteArray.mk vb.toArray)] }

theorem keysOf_pushed (T : Ht) (kb vb : List UInt8) : keysOf (pushed T kb vb) = keysOf T ++ [kb] := by
  simp [keysOf, pushed]

/-- What bytes handed to `insertAt` are off: the entry array, the arena, and
    the header fields that name them. -/
def Src (s : UInt64) (L : Lay) (q : UInt64) (n : Nat) : Prop :=
  Off L.e (L.ecap * ENTRY) q n ∧ Off L.ar L.acap q n ∧ Off (s + UInt64.ofNat ECAP) 16 q n ∧
    Off (s + UInt64.ofNat ARENA) 24 q n

/-- **What `insertAt` leaves**: the table with the new entry and its index,
    or the table as it was where something could not be allocated. -/
def InsQ (w : World) (s : UInt64) (T : Ht) (L : Lay) (kb vb : List UInt8) (w' : World) : Prop :=
  ∃ L', Kept w w' s L L' ∧ ((Tab w' s (pushed T kb vb) L' ∧ Idx w' L' (keysOf T ++ [kb])) ∨
    (Tab w' s T L' ∧ Idx w' L' (keysOf T)))

theorem InsQ.grows {w w' : World} {s : UInt64} {T : Ht} {L : Lay} {kb vb : List UInt8} (h : Tab w s T L)
    (hI : Idx w L (keysOf T)) (g : Grows w w') : InsQ w s T L kb vb w' :=
  ⟨L, (h.grown g).kept h, Or.inr ⟨h.appended (h.grown g), hI.appended h (h.grown g)⟩⟩

/-- What making room for an entry leaves. -/
def EntM (w : World) (s : UInt64) (T : Ht) (L : Lay) (kb vb : List UInt8) (i : UInt64) (t : Nat) (kp vp : UInt64)
    (x : UInt64) (w' : World) : Prop :=
  (x = 0 ∧ Grows w w') ∨ (x = c64 1 ∧ ∃ L1, Tab w' s T L1 ∧ Spot w' L1 (keysOf T) kb i t ∧ Kept w w' s L L1 ∧
    T.entries.length < L1.ecap ∧ L1.ix = L.ix ∧ L1.ar = L.ar ∧ L1.acap = L.acap ∧ Bytes w' kp kb ∧ Bytes w' vp vb)

section fns
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

set_option maxHeartbeats 6400000 in
/-- **`insertAt`** of a key no entry has, at the empty slot its probe ended
    at: the table has the new entry and the index covers its key, or the
    table is as it was. -/
theorem wp_insertAt {s slot key klen val vlen hh : Slot .i64} {sv slotA kp vp : UInt64} {T : Ht} {L : Lay}
    {kb vb : List UInt8} {i : UInt64} {t : Nat} {Q : Unit → Env → World → Prop}
    (hs : Γ[s]? = some (.sc .i64 sv)) (hslot : Γ[slot]? = some (.sc .i64 slotA))
    (hkey : Γ[key]? = some (.sc .i64 kp)) (hklen : Γ[klen]? = some (.sc .i64 (UInt64.ofNat kb.length)))
    (hval : Γ[val]? = some (.sc .i64 vp)) (hvlen : Γ[vlen]? = some (.sc .i64 (UInt64.ofNat vb.length)))
    (hhh : Γ[hh]? = some (.sc .i64 (H kb)))
    (h : Tab w sv T L) (S : Spot w L (keysOf T) kb i t) (hnot : kb ∉ keysOf T)
    (hsl : slotA = L.ix + UInt64.ofNat (8 * i.toNat))
    (hkb : Bytes w kp kb) (hvb : Bytes w vp vb) (sk : kb ≠ [] → Src sv L kp kb.length)
    (svv : vb ≠ [] → Src sv L vp vb.length)
    (hk : ∀ Γ' w', Ext Γ Γ' → InsQ w sv T L kb vb w' → Q () Γ' w') :
    wp cfg d J (LibHt.insertAt s slot key klen val vlen hh) Q Γ w := by
  have hpos := h.pos
  have hr := h.room
  have he36 := Blk.le36 h.be
  simp only [ENTRY] at he36
  have hlen : (keysOf T).length = T.entries.length := by simp [keysOf]
  have z0 : c64 0 = 0 := by decide
  have hcnt : (UInt64.ofNat T.entries.length).toNat = T.entries.length := by
    rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt h.rep.small.2]
  have hecap : (UInt64.ofNat L.ecap).toNat = L.ecap := by rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  have hEW : c64 (ENTRY : Nat) = UInt64.ofNat ENTRY := by decide
  unfold LibHt.insertAt
  wp_seq
  refine wp_at (x := sv) (by slot) ?_
  refine wp_at (x := sv) (by slot) ?_
  refine wp_load64 (by slot) h.rep.count ?_
  apply wp_iconst64
  apply wp_iconst64
  apply wp_iconst64
  refine wp_load64 (by slot) h.fecap ?_
  refine wp_ifteV (EntM w sv T L kb vb i t kp vp) (x := UInt64.ofNat T.entries.length) (y := UInt64.ofNat L.ecap)
    (by slot) (by slot) ?_ ?_ ?_
  · intro hlt
    have hlt' : T.entries.length < L.ecap := by
      have := UInt64.lt_iff_toNat_lt.mp (by simpa [cmpInt] using hlt)
      rwa [hcnt, hecap] at this
    wp_seq
    apply wp_iconst64
    exact ⟨by ext_env, _, by slot, Or.inr ⟨rfl, L, h, S, Kept.refl _ _ _, hlt', rfl, rfl, rfl, hkb, hvb⟩⟩
  · intro hge ne hne
    have hN : (UInt64.ofNat (2 * L.ecap) * c64 (ENTRY : Nat)).toNat = 2 * L.ecap * ENTRY := by
      rw [hEW, ← UInt64.ofNat_mul, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by simp only [ENTRY]; omega)]
    have hK : (UInt64.ofNat T.entries.length * c64 (ENTRY : Nat)).toNat = T.entries.length * ENTRY := by
      rw [hEW, ← UInt64.ofNat_mul, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by simp only [ENTRY]; omega)]
    wp_seq
    refine wp_load64 (by slot) h.fecap ?_
    refine wp_ishlImm (x := UInt64.ofNat L.ecap) (by slot) ?_
    rw [shl1 (by omega)]
    refine wp_imul (x := UInt64.ofNat (2 * L.ecap)) (y := c64 (ENTRY : Nat)) (by slot) (by slot) ?_
    refine wp_calloc (by slot) (fun e => by
      have := congrArg UInt64.toNat e
      rw [hN] at this
      simp [ENTRY] at this
      omega) ?_
    intro x w1 g1 hx
    rw [hN] at hx
    refine wp_ifteV (EntM w sv T L kb vb i t kp vp) (x := x) (y := c64 0) (by slot) (by slot) ?_ ?_ ?_
    · intro _
      wp_seq
      exact ⟨by ext_env, _, by slot, Or.inl ⟨z0, g1⟩⟩
    · intro hnz ne2 hne2
      have f := hx.resolve_left (by simpa [cmpInt, z0] using hnz)
      have c1 := g1.calm h.rep.calm
      have hE : 0 < L.ecap * ENTRY := Nat.mul_pos hpos.1 (by decide)
      have hse : sv ≠ L.e := h.apart.1.ne (by decide) hE
      have hkE : T.entries.length * ENTRY ≤ L.ecap * ENTRY := Nat.mul_le_mul_right _ hr.1
      obtain ⟨cs0, hcl, hcs⟩ := (Blk.room0 (g1.blk _ _ h.be) c1 hkE).bytes
      have a_ne : Apart x (2 * L.ecap * ENTRY) L.e (L.ecap * ENTRY) := Or.inr (f.before h.be)
      wp_seq
      refine wp_at (x := sv) (by slot) ?_
      refine wp_load64 (by slot) (g1.fld h.fe) ?_
      refine wp_imul (x := UInt64.ofNat T.entries.length) (y := c64 (ENTRY : Nat)) (by slot) (by slot) ?_
      refine wp_memcpy (bs := cs0) (by slot) (by slot) (by slot) hcs (by rw [hK]; exact hcl)
        (by rw [hK]; exact Blk.room0 f.blk c1 (by simp only [ENTRY]; omega))
        (by rw [hK]; exact a_ne.num (by simp only [ENTRY]; omega) hkE) ?_
      intro w2 W2 B2 _
      rw [hK] at W2
      refine wp_free (by slot) (Or.inr (W2.stays.blk _ _ (g1.blk _ _ h.be))) ?_
      intro w3 F
      have bs3 : Blk w3 sv SIZE := F.blk _ _ hse (W2.stays.blk _ _ (g1.blk _ _ h.rep.blk))
      have c3 : Calm w3 := F.calm (W2.stays.calm c1)
      obtain ⟨m4, hm4⟩ := Blk.fld_store bs3 c3 (j := ENTRIES) (by decide) x
      refine wp_at (x := sv) (by slot) ?_
      refine wp_storeI64 (by slot) (by slot) hm4 ?_
      obtain ⟨m5, hm5⟩ := Blk.fld_store (Blk.keep bs3 hm4) (c3.stored hm4) (j := ECAP) (by decide)
        (UInt64.ofNat (2 * L.ecap))
      refine wp_storeI64 (by slot) (by slot) hm5 ?_
      apply wp_iconst64
      obtain ⟨hT5, hI5, M⟩ := h.entGrown S.idx g1 f (N := 2 * L.ecap) (by omega) (by omega) hcs hcl W2 B2 F hm4
        hm5 rfl
      refine ⟨by ext_env, _, by slot, Or.inr ⟨rfl, _, hT5, ?_, ?_, ?_, rfl, rfl, rfl, ?_, ?_⟩⟩
      · exact S.keeps M.keeps (fun j hj => movP_blk h.rep.blk h.bi h.apart.2.2.2.1.symm h.apart.2.1.symm
          (by decide) (by decide) (by omega)) rfl rfl
      · refine Moved.kept h.rep.blk M (by simp [blocks]) (by decide) (by decide) fun b hb => ?_
        simp only [blocks, List.mem_cons, List.not_mem_nil, or_false] at hb ⊢
        rcases hb with rfl | rfl | rfl | rfl
        · exact Or.inl (Or.inl rfl)
        · exact Or.inr f.past
        · exact Or.inl (Or.inr (Or.inr (Or.inl rfl)))
        · exact Or.inl (Or.inr (Or.inr (Or.inr rfl)))
      · show T.entries.length < 2 * L.ecap
        omega
      · by_cases e : kb = []
        · exact Or.inl e
        · exact M.keeps.bytes _ _ e ⟨hkb.inW e, (sk e).1, (sk e).2.2.1⟩ hkb
      · by_cases e : vb = []
        · exact Or.inl e
        · exact M.keeps.bytes _ _ e ⟨hvb.inW e, (svv e).1, (svv e).2.2.1⟩ hvb
    · intro Γ5 w5 v nJ hE5 hv hr5
      exact ⟨Ext.trans (by ext_env) hE5, v, hv, hr5⟩
  · intro Γ6 w6 v nJ hE6 hv hr6
    simp only [carriesFrom, Vals.head]
    rcases hr6 with ⟨rfl, g⟩ | ⟨rfl, L1, hT1, S1, K1, hlt1, hix1, har1, hac1, hkb1, hvb1⟩
    · refine wp_when (InsQ w sv T L kb vb) (x := 0) (y := c64 0) (by slot) (by slot) ?_ ?_ ?_
      · intro hne
        exact absurd hne (by simp [cmpInt, z0])
      · intro _
        exact InsQ.grows h S.idx g
      · intro Γ' w' hE hR
        exact hk _ _ (Ext.trans (Ext.trans (by ext_env) hE6) hE) hR
    · refine wp_when (InsQ w sv T L kb vb) (x := c64 1) (y := c64 0) (by slot) (by slot) ?_ ?_ ?_
      · intro _
        have ok1 : kb ≠ [] → Off L1.ar L1.acap kp kb.length := fun e => by rw [har1, hac1]; exact (sk e).2.1
        have hu1 := Blk.le36 hT1.ba
        have hr1 := hT1.room
        wp_seq
        refine wp_append (bs := kb) (by slot) (by slot) (by slot) hT1 hkb1 ok1 (fun e => (sk e).2.2.2) ?_
        intro Γ7 w7 r7 x7 hE7 hr7 hA
        rcases hA with ⟨rfl, g7⟩ | ⟨rfl, L2, A1⟩
        · refine wp_when (InsQ w sv T L kb vb) (x := c64 (-1)) (y := c64 (-1)) (by slot) (by slot) ?_ ?_ ?_
          · intro hne
            exact absurd hne (by simp [cmpInt])
          · intro _
            exact ⟨L1, K1.trans ((hT1.grown g7).kept hT1), Or.inr ⟨hT1.appended (hT1.grown g7),
              S1.idx.appended hT1 (hT1.grown g7)⟩⟩
          · intro Γ' w' hE hR
            exact ⟨Ext.trans (Ext.trans (by ext_env) hE7) hE, hR⟩
        · have hT2 := hT1.appended A1
          have K2 := K1.trans (A1.kept hT1)
          have S2 : Spot w7 L2 (keysOf T) kb i t := S1.keeps A1.keeps (fun j hj => movP_blk hT1.rep.blk hT1.bi
            hT1.apart.2.2.2.2.2 hT1.apart.2.1.symm (by decide) (by decide) (by omega)) A1.ix A1.icap
          have hvb2 : Bytes w7 vp vb := by
            by_cases e : vb = []
            · exact Or.inl e
            · exact A1.keeps.bytes _ _ e ⟨hvb1.inW e, by rw [har1, hac1]; exact (svv e).2.1, (svv e).2.2.2⟩ hvb1
          have ok2 : vb ≠ [] → Off L2.ar L2.acap vp vb.length := fun e =>
            A1.off _ _ (hvb1.inW e) (by rw [har1, hac1]; exact (svv e).2.1)
          have hu2 := Blk.le36 hT2.ba
          have hr2 := hT2.room
          refine wp_when (InsQ w sv T L kb vb) (x := UInt64.ofNat L1.used) (y := c64 (-1)) (by slot) (by slot)
            ?_ ?_ ?_
          · intro _
            wp_seq
            refine wp_append (bs := vb) (by slot) (by slot) (by slot) hT2 hvb2 ok2 (fun e => (svv e).2.2.2) ?_
            intro Γ8 w8 r8 x8 hE8 hr8 hA2
            rcases hA2 with ⟨rfl, g8⟩ | ⟨rfl, L3, A2⟩
            · refine wp_when (InsQ w sv T L kb vb) (x := c64 (-1)) (y := c64 (-1)) (by slot) (by slot) ?_ ?_ ?_
              · intro hne
                exact absurd hne (by simp [cmpInt])
              · intro _
                exact ⟨L2, K2.trans ((hT2.grown g8).kept hT2), Or.inr ⟨hT2.appended (hT2.grown g8),
                  S2.idx.appended hT2 (hT2.grown g8)⟩⟩
              · intro Γ' w' hE hR
                exact ⟨Ext.trans (Ext.trans (by ext_env) hE8) hE, hR⟩
            · have hT3 := hT2.appended A2
              have K3 := K2.trans (A2.kept hT2)
              have S3 : Spot w8 L3 (keysOf T) kb i t := S2.keeps A2.keeps (fun j hj => movP_blk hT2.rep.blk hT2.bi
                hT2.apart.2.2.2.2.2 hT2.apart.2.1.symm (by decide) (by decide) (by omega)) A2.ix A2.icap
              refine wp_when (InsQ w sv T L kb vb) (x := UInt64.ofNat L2.used) (y := c64 (-1)) (by slot) (by slot)
                ?_ ?_ ?_
              · intro _
                have hlt3 : T.entries.length < L3.ecap := by rw [A2.ecap, A1.ecap]; exact hlt1
                have he3 := Blk.le36 hT3.be
                simp only [ENTRY] at he3
                have hsl3 : slotA = L3.ix + UInt64.ofNat (8 * i.toNat) := by rw [A2.ix, A1.ix, hix1]; exact hsl
                have eA : L3.e + UInt64.ofNat T.entries.length * c64 (ENTRY : Nat) =
                    L3.e + UInt64.ofNat (T.entries.length * ENTRY) := by rw [hEW, ← UInt64.ofNat_mul]
                have e8 : ∀ (j : Nat) (z : Int), c64 z = UInt64.ofNat j →
                    L3.e + UInt64.ofNat (T.entries.length * ENTRY) + c64 z =
                      L3.e + UInt64.ofNat (T.entries.length * ENTRY + j) := fun j z hz => by
                  rw [UInt64.add_assoc, hz, ← UInt64.ofNat_add]
                have hjc : ∀ c, c + 8 ≤ 32 → T.entries.length * ENTRY + c + 8 ≤ L3.ecap * ENTRY := by
                  intro c hc
                  simp only [ENTRY]
                  omega
                have en : UInt64.ofNat T.entries.length + c64 1 = UInt64.ofNat (T.entries.length + 1) := by
                  rw [show c64 1 = UInt64.ofNat 1 from by decide, ← UInt64.ofNat_add]
                wp_seq
                refine wp_at (x := sv) (by slot) ?_
                refine wp_load64 (by slot) hT3.fe ?_
                refine wp_imul (x := UInt64.ofNat T.entries.length) (y := c64 (ENTRY : Nat)) (by slot) (by slot) ?_
                refine wp_iadd (x := L3.e) (by slot) (by slot) ?_
                rw [eA]
                obtain ⟨ma, hma⟩ := Blk.fld_store hT3.be hT3.rep.calm (j := T.entries.length * ENTRY)
                  (hjc 0 (by decide)) (UInt64.ofNat L1.used)
                refine wp_storeI64 (by slot) (by slot) hma ?_
                obtain ⟨hTa, Sa, Ka, fa, -, ba⟩ := hT3.entStore S3 hlt3 (c := 0) (by decide) hma
                refine wp_iaddImm (x := L3.e + UInt64.ofNat (T.entries.length * ENTRY)) (by slot) ?_
                rw [e8 8 8 (by decide)]
                obtain ⟨mb, hmb⟩ := Blk.fld_store hTa.be hTa.rep.calm (j := T.entries.length * ENTRY + 8)
                  (hjc 8 (by decide)) (UInt64.ofNat kb.length)
                refine wp_storeI64 (by slot) (by slot) hmb ?_
                obtain ⟨hTb, Sb, Kb, fb, ob, bb⟩ := hTa.entStore Sa hlt3 (c := 8) (by decide) hmb
                refine wp_iaddImm (x := L3.e + UInt64.ofNat (T.entries.length * ENTRY)) (by slot) ?_
                rw [e8 16 16 (by decide)]
                obtain ⟨mc, hmc⟩ := Blk.fld_store hTb.be hTb.rep.calm (j := T.entries.length * ENTRY + 16)
                  (hjc 16 (by decide)) (UInt64.ofNat L2.used)
                refine wp_storeI64 (by slot) (by slot) hmc ?_
                obtain ⟨hTc, Sc, Kc, fc, oc, bc⟩ := hTb.entStore Sb hlt3 (c := 16) (by decide) hmc
                refine wp_iaddImm (x := L3.e + UInt64.ofNat (T.entries.length * ENTRY)) (by slot) ?_
                rw [e8 24 24 (by decide)]
                obtain ⟨md, hmd⟩ := Blk.fld_store hTc.be hTc.rep.calm (j := T.entries.length * ENTRY + 24)
                  (hjc 24 (by decide)) (UInt64.ofNat vb.length)
                refine wp_storeI64 (by slot) (by slot) hmd ?_
                obtain ⟨hTd, Sd, Kd, fd, od, bd⟩ := hTc.entStore Sc hlt3 (c := 24) (by decide) hmd
                refine wp_iaddImm (x := UInt64.ofNat T.entries.length) (by slot) ?_
                rw [en]
                refine wp_ushrImm (x := H kb) (by slot) ?_
                refine wp_ishlImm (by slot) ?_
                refine wp_bor (by slot) (by slot) ?_
                rw [show ((H kb >>> (c64 32 % 64)) <<< (c64 32 % 64)) ||| UInt64.ofNat (T.entries.length + 1) =
                  slotVal (H kb) T.entries.length from rfl]
                have hbi := hTd.bi
                have hji : 8 * i.toNat + 8 ≤ L3.icap * 8 := by have := Sd.lt; omega
                obtain ⟨me, hme⟩ := Blk.fld_store hbi hTd.rep.calm (j := 8 * i.toNat) hji
                  (slotVal (H kb) T.entries.length)
                subst hsl3
                refine wp_storeI64 (by slot) (by slot) hme ?_
                have We := wrote_stored hme
                have hTe := hTd.wroteOff We (hTd.apart.2.1.symm.clear hbi hji)
                  (Off.prefix (hTd.apart.2.2.2.1.symm.clear hbi hji) (Nat.mul_le_mul_right _ hTd.room.1))
                  (Off.prefix (hTd.apart.2.2.2.2.2.clear hbi hji) hTd.room.2)
                have Ie : Idx _ L3 (keysOf T ++ [kb]) := Idx.insert Sd.idx hnot (by rw [hlen]; omega) Sd.on Sd.bef
                  Sd.emp (by rw [hlen]; exact load_store8 hme) (fun j hj' hne =>
                    fld_kept hbi (i := 8 * j) (j := 8 * i.toNat) (by omega) (by omega) (by omega) hme)
                have Ke := Kept.wrote (s := sv) (L := L3) (y := L3.ix) (k := L3.icap * 8) (by simp [blocks]) hbi We
                  hji
                have cei := hTd.apart.2.2.2.1.symm.clear hbi hji
                have cai := hTd.apart.2.2.2.2.2.clear hbi hji
                have hr3 := hT3.room
                have hk1 : L1.used + kb.length ≤ L3.acap := by
                  have h1 := A2.used
                  have h2 := A1.used
                  have h3 := hr3.2
                  omega
                have hk2 : L2.used + vb.length ≤ L3.acap := by
                  have h1 := A2.used
                  have h3 := hr3.2
                  omega
                have bk : Bytes _ (L3.ar + UInt64.ofNat L1.used) kb := We.bytes_blk hTd.ba hk1
                  (bd _ _ hk1 (bc _ _ hk1 (bb _ _ hk1 (ba _ _ hk1
                    (A2.pre L1.used kb (by rw [A1.used]; omega) A1.new))))) cai
                have bv : Bytes _ (L3.ar + UInt64.ofNat L2.used) vb := We.bytes_blk hTd.ba hk2
                  (bd _ _ hk2 (bc _ _ hk2 (bb _ _ hk2 (ba _ _ hk2 A2.new)))) cai
                have hent : EntO _ L3 T.entries.length (ByteArray.mk kb.toArray, ByteArray.mk vb.toArray) L1.used
                    L2.used :=
                  ⟨(We.fld_blk hTd.be (hjc 0 (by decide)) cei).trans
                      (od 0 _ (by decide) (Or.inl (by decide)) (oc 0 _ (by decide) (Or.inl (by decide))
                        (ob 0 _ (by decide) (Or.inl (by decide)) fa))),
                    (We.fld_blk hTd.be (hjc 8 (by decide)) cei).trans
                      (od 8 _ (by decide) (Or.inl (by decide)) (oc 8 _ (by decide) (Or.inl (by decide)) fb)),
                    (We.fld_blk hTd.be (hjc 16 (by decide)) cei).trans (od 16 _ (by decide) (Or.inl (by decide)) fc),
                    (We.fld_blk hTd.be (hjc 24 (by decide)) cei).trans fd,
                    by show L1.used + kb.length ≤ L3.used; rw [A2.used, A1.used]; omega,
                    by show L2.used + vb.length ≤ L3.used; rw [A2.used]; omega,
                    bk, bv⟩
                obtain ⟨mf, hmf⟩ := Blk.fld_store hTe.rep.blk hTe.rep.calm (j := COUNT) (by decide)
                  (UInt64.ofNat (T.entries.length + 1))
                refine wp_storeI64 (by slot) (by slot) hmf ?_
                have Wf := wrote_stored hmf
                have hB : ∀ i (hi : i < T.entries.length) (ho : i < L3.offs.length),
                    (L3.offs[i]).1 + (T.entries[i]).1.size ≤ L1.used ∧
                      (L3.offs[i]).2 + (T.entries[i]).2.size ≤ L1.used := by
                  rw [A2.offs, A1.offs]
                  exact fun i hi ho => ⟨(hT1.ento i hi ho).2.2.2.2.1, (hT1.ento i hi ho).2.2.2.2.2.1⟩
                have hTf : Tab _ sv (pushed T kb vb) { L3 with offs := L3.offs ++ [(L1.used, L2.used)] } :=
                  hTe.push hlt3 hent hB (by show L1.used + kb.length ≤ L2.used; rw [A1.used]; omega) hmf
                have If : Idx _ { L3 with offs := L3.offs ++ [(L1.used, L2.used)] } (keysOf T ++ [kb]) :=
                  (Ie.wrote hTe.bi Wf (hTe.apart.2.1.clear hTe.rep.blk (by decide))).lay rfl rfl
                have Kf := Kept.wrote (s := sv) (L := L3) (y := sv) (k := SIZE) (by simp [blocks]) hTe.rep.blk Wf
                  (by decide)
                have Kall : Kept w _ sv L { L3 with offs := L3.offs ++ [(L1.used, L2.used)] } :=
                  (K3.trans (Ka.trans (Kb.trans (Kc.trans (Kd.trans (Ke.trans Kf)))))).lay rfl
                refine wp_ishlImm (x := UInt64.ofNat (T.entries.length + 1)) (by slot) ?_
                refine wp_at (x := sv) (by slot) ?_
                refine wp_load64 (by slot) hTf.ficap ?_
                refine wp_when (InsQ w sv T L kb vb) (by slot) (by slot) ?_ ?_ ?_
                · intro _
                  refine wp_grow (T := pushed T kb vb) (by slot) hTf (by rw [keysOf_pushed]; exact If) ?_
                  rintro Γ' w' hE ⟨L4, hT4, hI4, K4⟩
                  exact ⟨Ext.trans (by ext_env) hE, L4, Kall.trans K4, Or.inl ⟨hT4, by
                    rw [← keysOf_pushed T kb vb]; exact hI4⟩⟩
                · intro _
                  exact ⟨_, Kall, Or.inl ⟨hTf, If⟩⟩
                · intro Γ' w' hE hR
                  exact ⟨Ext.trans (by ext_env) hE, hR⟩
              · intro he
                exact absurd (by simpa [cmpInt] using he) (ofNat_ne_neg1 (n := L2.used) (by omega))
              · intro Γ' w' hE hR
                exact ⟨Ext.trans (Ext.trans (by ext_env) hE8) hE, hR⟩
          · intro he
            exact absurd (by simpa [cmpInt] using he) (ofNat_ne_neg1 (n := L1.used) (by omega))
          · intro Γ' w' hE hR
            exact ⟨Ext.trans (Ext.trans (by ext_env) hE7) hE, hR⟩
      · intro he
        exact absurd he (by decide)
      · intro Γ' w' hE hR
        exact hk _ _ (Ext.trans (Ext.trans (by ext_env) hE6) hE) hR

end fns

-- ---------------------------------------------------------------------------
-- Setting a key
-- ---------------------------------------------------------------------------

/-- The table with entry `p`'s value replaced. -/
def setVal (T : Ht) (p : Nat) (vb : List UInt8) : Ht :=
  { T with entries := T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray) }

theorem keysOf_setVal (T : Ht) (p : Nat) (vb : List UInt8) : keysOf (setVal T p vb) = keysOf T := by
  unfold keysOf setVal
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_map, List.getElem_modify]
  split <;> rfl

/-- The model sets a key no entry has by adding an entry. -/
theorem set_absent {T : Ht} {kb vb : List UInt8} (hnot : kb ∉ keysOf T) :
    T.set (ByteArray.mk kb.toArray) (ByteArray.mk vb.toArray) = pushed T kb vb := by
  unfold Ht.set
  rw [get_none hnot]
  rfl

/-- The model sets a key an entry has by replacing that entry's value, the
    keys being distinct. -/
theorem set_present {T : Ht} {kb vb : List UInt8} {p : Nat} (hp : p < T.entries.length) (hn : (keysOf T).Nodup)
    (hk : (T.entries[p]'hp).1.data.toList = kb) :
    T.set (ByteArray.mk kb.toArray) (ByteArray.mk vb.toArray) = setVal T p vb := by
  unfold Ht.set setVal
  rw [get_some hp hn hk, if_pos (show (some (T.entries[p]'hp).2).isSome = true from rfl)]
  congr 1
  apply List.ext_getElem (by simp)
  intro q h1 h2
  have hq : q < T.entries.length := by simpa using h1
  simp only [List.getElem_map, List.getElem_modify]
  by_cases e : p = q
  · subst e
    rw [if_pos rfl, if_pos (by simp [toList_data, hk])]
  · have hne := nodup_getElem_ne hn (p := p) (q := q) (by simpa [keysOf] using hp) (by simpa [keysOf] using hq) e
    simp only [keysOf, List.getElem_map] at hne
    rw [if_neg e, if_neg]
    simp only [toList_data, beq_iff_eq]
    intro c
    exact hne (hk.trans (by simpa using c.symm))

/-- A value replaced by one as long leaves the runs as separate. -/
theorem Sep.sameSize {T : Ht} {offs : List (Nat × Nat)} {p : Nat} {vb : List UInt8} (hs : Sep T offs)
    (hp : p < T.entries.length) (hsz : vb.length = (T.entries[p]).2.size) : Sep (setVal T p vb) offs := by
  have hlen : (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length = T.entries.length :=
    List.length_modify _ _ _
  have e1 : ∀ i (hi : i < T.entries.length) hi',
      ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[i]'hi').1 = (T.entries[i]'hi).1 := by
    intro i hi hi'
    rw [List.getElem_modify]
    split <;> rfl
  have e2 : ∀ i (hi : i < T.entries.length) hi',
      ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[i]'hi').2.size = (T.entries[i]'hi).2.size := by
    intro i hi hi'
    rw [List.getElem_modify]
    split
    · rename_i e
      subst e
      exact hsz
    · rfl
  intro i j hi hj oi oj
  have hi0 : i < T.entries.length := by rw [← hlen]; exact hi
  have hj0 : j < T.entries.length := by rw [← hlen]; exact hj
  have hi' : i < (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length := hi
  have hj' : j < (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length := hj
  show ((offs[i]).2 + ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[i]'hi').2.size ≤ (offs[j]).1 ∨
      (offs[j]).1 + ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[j]'hj').1.size ≤ (offs[i]).2) ∧
    (i ≠ j → (offs[i]).2 + ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[i]'hi').2.size ≤ (offs[j]).2 ∨
      (offs[j]).2 + ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[j]'hj').2.size ≤ (offs[i]).2)
  rw [e1 j hj0 hj', e2 i hi0 hi', e2 j hj0 hj']
  exact hs i j hi0 hj0 oi oj

/-- A value moved past every run there is leaves the runs separate. -/
theorem Sep.moved {T : Ht} {offs : List (Nat × Nat)} {p vo : Nat} {vb : List UInt8} (hs : Sep T offs)
    (ho : p < offs.length)
    (hU : ∀ i (hi : i < T.entries.length) (oi : i < offs.length),
      (offs[i]).1 + (T.entries[i]).1.size ≤ vo ∧ (offs[i]).2 + (T.entries[i]).2.size ≤ vo) :
    Sep (setVal T p vb) (offs.set p ((offs[p]'ho).1, vo)) := by
  have hlen : (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length = T.entries.length :=
    List.length_modify _ _ _
  have e1 : ∀ i (hi : i < T.entries.length) hi',
      ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[i]'hi').1 = (T.entries[i]'hi).1 := by
    intro i hi hi'
    rw [List.getElem_modify]
    split <;> rfl
  have e2 : ∀ i (hi : i < T.entries.length) hi', p ≠ i →
      ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[i]'hi').2 = (T.entries[i]'hi).2 := by
    intro i hi hi' hne
    rw [List.getElem_modify, if_neg hne]
  have o1 : ∀ i (oi : i < offs.length) (oi' : i < (offs.set p ((offs[p]'ho).1, vo)).length), p ≠ i → (offs.set p ((offs[p]'ho).1, vo))[i]'oi' = offs[i]'oi :=
    fun i oi oi' hne => List.getElem_set_ne hne _
  have o2 : ∀ (oi' : p < (offs.set p ((offs[p]'ho).1, vo)).length), (offs.set p ((offs[p]'ho).1, vo))[p]'oi' = ((offs[p]'ho).1, vo) := fun oi' => List.getElem_set_self _
  intro i j hi hj oi oj
  have hi0 : i < T.entries.length := by rw [← hlen]; exact hi
  have hj0 : j < T.entries.length := by rw [← hlen]; exact hj
  have oi0 : i < offs.length := by simpa using oi
  have oj0 : j < offs.length := by simpa using oj
  have hi' : i < (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length := hi
  have hj' : j < (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length := hj
  show (((offs.set p ((offs[p]'ho).1, vo))[i]'oi).2 +
        ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[i]'hi').2.size ≤
          ((offs.set p ((offs[p]'ho).1, vo))[j]'oj).1 ∨
      ((offs.set p ((offs[p]'ho).1, vo))[j]'oj).1 +
        ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[j]'hj').1.size ≤
          ((offs.set p ((offs[p]'ho).1, vo))[i]'oi).2) ∧
    (i ≠ j → ((offs.set p ((offs[p]'ho).1, vo))[i]'oi).2 +
        ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[i]'hi').2.size ≤
          ((offs.set p ((offs[p]'ho).1, vo))[j]'oj).2 ∨
      ((offs.set p ((offs[p]'ho).1, vo))[j]'oj).2 +
        ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[j]'hj').2.size ≤
          ((offs.set p ((offs[p]'ho).1, vo))[i]'oi).2)
  rw [e1 j hj0 hj']
  by_cases ci : p = i <;> by_cases cj : p = j
  · subst ci
    subst cj
    rw [o2]
    exact ⟨Or.inr (hU p hj0 oj0).1, fun hne => absurd rfl hne⟩
  · subst ci
    rw [o2, o1 j oj0 oj cj, e2 j hj0 hj' cj]
    exact ⟨Or.inr (hU j hj0 oj0).1, fun _ => Or.inr (hU j hj0 oj0).2⟩
  · subst cj
    rw [o2, o1 i oi0 oi ci, e2 i hi0 hi' ci]
    exact ⟨(hs i p hi0 hj0 oi0 oj0).1, fun _ => Or.inl (hU i hi0 oi0).2⟩
  · rw [o1 i oi0 oi ci, o1 j oj0 oj cj, e2 i hi0 hi' ci, e2 j hj0 hj' cj]
    exact hs i j hi0 hj0 oi0 oj0

/-- A value overwritten where it is, by one as long: the table with that
    value. -/
theorem Tab.setInPlace {w w' : World} {s : UInt64} {T : Ht} {L : Lay} {p n : Nat} {vb : List UInt8}
    (h : Tab w s T L) (hp : p < T.entries.length) (ho : p < L.offs.length)
    (hsz : vb.length = (T.entries[p]).2.size) (hn : n ≤ vb.length)
    (W : Wrote w w' (L.ar + UInt64.ofNat (L.offs[p]).2) n) (B : Bytes w' (L.ar + UInt64.ofNat (L.offs[p]).2) vb) :
    Tab w' s (setVal T p vb) L := by
  have hr := h.room
  have hvp := (h.ento p hp ho).2.2.2.2.2.1
  have hj : (L.offs[p]).2 + n ≤ L.acap := by omega
  have cs := h.apart.2.2.1.symm.clear h.ba hj
  have ce := h.apart.2.2.2.2.1.symm.clear h.ba hj
  have hf : ∀ j, j + 8 ≤ SIZE → fld w' s j = fld w s j := fun j hj => W.fld_blk h.rep.blk hj cs
  have hlen : (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length = T.entries.length :=
    List.length_modify _ _ _
  have hroom : (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length ≤ L.ecap := by
    rw [hlen]; exact hr.1
  have holen : L.offs.length = (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length := by
    rw [hlen]; exact h.olen
  refine {
    rep := ⟨W.stays.blk _ _ h.rep.blk, W.stays.calm h.rep.calm, (hf _ (by decide)).trans h.rep.created, ?_,
      h.rep.handles, h.rep.small.1, ?_⟩
    fe := (hf _ (by decide)).trans h.fe
    fecap := (hf _ (by decide)).trans h.fecap
    fix := (hf _ (by decide)).trans h.fix
    ficap := (hf _ (by decide)).trans h.ficap
    far := (hf _ (by decide)).trans h.far
    fused := (hf _ (by decide)).trans h.fused
    facap := (hf _ (by decide)).trans h.facap
    we := W.whole h.we
    wi := W.whole h.wi
    wa := W.whole h.wa
    room := ⟨hroom, hr.2⟩
    pos := h.pos
    apart := h.apart
    olen := holen
    sep := h.sep.sameSize hp hsz
    ento := fun i hi ho' => ?_ }
  · show fld w' s COUNT = some (UInt64.ofNat (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length)
    rw [hlen]
    exact (hf _ (by decide)).trans h.rep.count
  · show (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length < 2 ^ 64
    rw [hlen]
    exact h.rep.small.2
  · have hi0 : i < T.entries.length := by rw [← hlen]; exact hi
    have hi' : i < (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length := hi
    obtain ⟨h0, h8, h16, h24, hk, hv, bk, bv⟩ := h.ento i hi0 ho'
    have he : ∀ j, j + 8 ≤ ENTRY → fld w' L.e (i * ENTRY + j) = fld w L.e (i * ENTRY + j) := fun j hj' =>
      W.fld_blk h.be (by simp only [ENTRY] at hj' ⊢; omega) ce
    obtain ⟨s1, s2⟩ := h.sep p i hp hi0 ho ho'
    have bk' : Bytes w' (L.ar + UInt64.ofNat (L.offs[i]).1) (T.entries[i]).1.data.toList :=
      W.bytes_in h.ba bk hj (by rw [size_toList]; omega) (by
        rw [size_toList]
        rcases s1 with c | c
        · exact Or.inl (by omega)
        · exact Or.inr (by omega))
    show EntO w' L i ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[i]'hi') (L.offs[i]).1 (L.offs[i]).2
    rw [List.getElem_modify]
    by_cases e : p = i
    · subst e
      rw [if_pos rfl]
      exact ⟨(he 0 (by decide)).trans h0, (he 8 (by decide)).trans h8, (he 16 (by decide)).trans h16,
        by show fld w' L.e (p * ENTRY + 24) = some (UInt64.ofNat vb.length)
           rw [hsz]; exact (he 24 (by decide)).trans h24, hk,
        by show (L.offs[p]).2 + vb.length ≤ L.used
           rw [hsz]; exact hv, bk', B⟩
    · rw [if_neg e]
      exact ⟨(he 0 (by decide)).trans h0, (he 8 (by decide)).trans h8, (he 16 (by decide)).trans h16,
        (he 24 (by decide)).trans h24, hk, hv, bk', W.bytes_in h.ba bv hj (by rw [size_toList]; omega) (by
          rw [size_toList]
          rcases s2 e with c | c
          · exact Or.inl (by omega)
          · exact Or.inr (by omega))⟩

/-- A value appended to the arena and named by its entry: the table with
    that value. -/
theorem Tab.setMoved {w : World} {s : UInt64} {T : Ht} {L : Lay} {p vo : Nat} {vb : List UInt8} {m1 m2 : Mem}
    (h : Tab w s T L) (hp : p < T.entries.length) (ho : p < L.offs.length)
    (hU : ∀ i (hi : i < T.entries.length) (oi : i < L.offs.length),
      (L.offs[i]).1 + (T.entries[i]).1.size ≤ vo ∧ (L.offs[i]).2 + (T.entries[i]).2.size ≤ vo)
    (hvo : vo + vb.length ≤ L.used) (B : Bytes w (L.ar + UInt64.ofNat vo) vb)
    (hm1 : w.mem.store (L.e + UInt64.ofNat (p * ENTRY + 16)) 8 (UInt64.ofNat vo) = some m1)
    (hm2 : (stored w (L.e + UInt64.ofNat (p * ENTRY + 16)) (UInt64.ofNat vo) m1).mem.store
      (L.e + UInt64.ofNat (p * ENTRY + 24)) 8 (UInt64.ofNat vb.length) = some m2) :
    Tab (stored (stored w (L.e + UInt64.ofNat (p * ENTRY + 16)) (UInt64.ofNat vo) m1)
        (L.e + UInt64.ofNat (p * ENTRY + 24)) (UInt64.ofNat vb.length) m2) s (setVal T p vb)
      { L with offs := L.offs.set p ((L.offs[p]).1, vo) } := by
  have hr := h.room
  have W1 := wrote_stored hm1
  have W2 := wrote_stored hm2
  have hjc : ∀ i c, i < T.entries.length → c + 8 ≤ 32 → i * ENTRY + c + 8 ≤ L.ecap * ENTRY := by
    intro i c hi hc
    simp only [ENTRY]
    omega
  have hj16 := hjc p 16 hp (by decide)
  have hj24 := hjc p 24 hp (by decide)
  have be1 := Blk.keep h.be hm1
  have hf : ∀ j, j + 8 ≤ SIZE → fld (stored (stored w (L.e + UInt64.ofNat (p * ENTRY + 16)) (UInt64.ofNat vo) m1)
      (L.e + UInt64.ofNat (p * ENTRY + 24)) (UInt64.ofNat vb.length) m2) s j = fld w s j := fun j hj =>
    (W2.fld_blk (Blk.keep h.rep.blk hm1) hj (h.apart.1.symm.clear be1 hj24)).trans
      (W1.fld_blk h.rep.blk hj (h.apart.1.symm.clear h.be hj16))
  have hby : ∀ j cs, j + cs.length ≤ L.acap → Bytes w (L.ar + UInt64.ofNat j) cs →
      Bytes (stored (stored w (L.e + UInt64.ofNat (p * ENTRY + 16)) (UInt64.ofNat vo) m1)
        (L.e + UInt64.ofNat (p * ENTRY + 24)) (UInt64.ofNat vb.length) m2) (L.ar + UInt64.ofNat j) cs :=
    fun j cs hj hb => W2.bytes_blk (Blk.keep h.ba hm1) hj
      (W1.bytes_blk h.ba hj hb (h.apart.2.2.2.2.1.clear h.be hj16)) (h.apart.2.2.2.2.1.clear be1 hj24)
  have hfe : ∀ j, j + 8 ≤ L.ecap * ENTRY → (j + 8 ≤ p * ENTRY + 16 ∨ p * ENTRY + 32 ≤ j) →
      fld (stored (stored w (L.e + UInt64.ofNat (p * ENTRY + 16)) (UInt64.ofNat vo) m1)
        (L.e + UInt64.ofNat (p * ENTRY + 24)) (UInt64.ofNat vb.length) m2) L.e j = fld w L.e j := fun j hj hd =>
    (fld_kept be1 hj hj24 (by omega) hm2).trans (fld_kept h.be hj hj16 (by omega) hm1)
  have hlen : (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length = T.entries.length :=
    List.length_modify _ _ _
  have hroom : (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length ≤ L.ecap := by
    rw [hlen]; exact hr.1
  have holen : (L.offs.set p ((L.offs[p]).1, vo)).length =
      (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length := by
    rw [hlen, List.length_set]; exact h.olen
  refine {
    rep := ⟨Blk.keep (Blk.keep h.rep.blk hm1) hm2, (h.rep.calm.stored hm1).stored hm2,
      (hf _ (by decide)).trans h.rep.created, ?_, h.rep.handles, h.rep.small.1, ?_⟩
    fe := (hf _ (by decide)).trans h.fe
    fecap := (hf _ (by decide)).trans h.fecap
    fix := (hf _ (by decide)).trans h.fix
    ficap := (hf _ (by decide)).trans h.ficap
    far := (hf _ (by decide)).trans h.far
    fused := (hf _ (by decide)).trans h.fused
    facap := (hf _ (by decide)).trans h.facap
    we := W2.whole (W1.whole h.we)
    wi := W2.whole (W1.whole h.wi)
    wa := W2.whole (W1.whole h.wa)
    room := ⟨hroom, hr.2⟩
    pos := h.pos
    apart := h.apart
    olen := holen
    sep := h.sep.moved ho hU
    ento := fun i hi ho' => ?_ }
  · show fld _ s COUNT = some (UInt64.ofNat (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length)
    rw [hlen]
    exact (hf _ (by decide)).trans h.rep.count
  · show (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length < 2 ^ 64
    rw [hlen]
    exact h.rep.small.2
  · have hi0 : i < T.entries.length := by rw [← hlen]; exact hi
    have hi' : i < (T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray)).length := hi
    have ho0 : i < L.offs.length := by rw [h.olen]; exact hi0
    have ho2 : i < (L.offs.set p ((L.offs[p]).1, vo)).length := ho'
    obtain ⟨h0, h8, h16, h24, hk, hv, bk, bv⟩ := h.ento i hi0 ho0
    show EntO _ L i ((T.entries.modify p fun e => (e.1, ByteArray.mk vb.toArray))[i]'hi')
      ((L.offs.set p ((L.offs[p]).1, vo))[i]'ho2).1 ((L.offs.set p ((L.offs[p]).1, vo))[i]'ho2).2
    rw [List.getElem_modify]
    by_cases e : p = i
    · subst e
      rw [if_pos rfl, List.getElem_set_self]
      exact ⟨(hfe _ (hjc p 0 hp (by decide)) (Or.inl (by omega))).trans h0,
        (hfe _ (hjc p 8 hp (by decide)) (Or.inl (by omega))).trans h8,
        (fld_kept be1 hj16 hj24 (by omega) hm2).trans (fld_stored hm1), fld_stored hm2, hk, hvo,
        hby _ _ (by rw [size_toList]; exact Nat.le_trans hk hr.2) bk, hby _ _ (Nat.le_trans hvo hr.2) B⟩
    · rw [if_neg e, List.getElem_set_ne e]
      have hd : ∀ c, c + 8 ≤ 32 → i * ENTRY + c + 8 ≤ p * ENTRY + 16 ∨ p * ENTRY + 32 ≤ i * ENTRY + c := by
        intro c hc
        simp only [ENTRY]
        omega
      exact ⟨(hfe _ (hjc i 0 hi0 (by decide)) (hd 0 (by decide))).trans h0,
        (hfe _ (hjc i 8 hi0 (by decide)) (hd 8 (by decide))).trans h8,
        (hfe _ (hjc i 16 hi0 (by decide)) (hd 16 (by decide))).trans h16,
        (hfe _ (hjc i 24 hi0 (by decide)) (hd 24 (by decide))).trans h24, hk, hv,
        hby _ _ (by rw [size_toList]; exact Nat.le_trans hk hr.2) bk,
        hby _ _ (by rw [size_toList]; exact Nat.le_trans hv hr.2) bv⟩

/-- What is clear of the table is off what `insertAt` asks its bytes be off. -/
theorem Clear.src {w : World} {s q : UInt64} {T : Ht} {L : Lay} {n : Nat} (h : Tab w s T L) (c : Clear s L q n) :
    Src s L q n :=
  ⟨c L.e _ (by simp [blocks]), c L.ar _ (by simp [blocks]),
    Off.part h.rep.blk (c s SIZE (by simp [blocks])) (by decide) (by decide),
    Off.part h.rep.blk (c s SIZE (by simp [blocks])) (by decide) (by decide)⟩

/-- **What `set` leaves**: the table the model's `set` leaves and its index,
    or the table as it was where something could not be allocated. -/
def SetQ (w : World) (s : UInt64) (T : Ht) (L : Lay) (kb vb : List UInt8) (w' : World) : Prop :=
  ∃ L', Kept w w' s L L' ∧
    ((Tab w' s (T.set (ByteArray.mk kb.toArray) (ByteArray.mk vb.toArray)) L' ∧
        Idx w' L' (keysOf (T.set (ByteArray.mk kb.toArray) (ByteArray.mk vb.toArray)))) ∨
      (Tab w' s T L' ∧ Idx w' L' (keysOf T)))

section fns
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

set_option maxHeartbeats 3200000 in
/-- **`set`** of a key and a value that can be read, clear of the table:
    the table is the one the model's `set` leaves, or as it was. -/
theorem wp_set {s key klen val vlen : Slot .i64} {sv kp vp : UInt64} {T : Ht} {L : Lay} {kb vb : List UInt8}
    {Q : Unit → Env → World → Prop}
    (hs : Γ[s]? = some (.sc .i64 sv)) (hkey : Γ[key]? = some (.sc .i64 kp))
    (hklen : Γ[klen]? = some (.sc .i64 (UInt64.ofNat kb.length))) (hval : Γ[val]? = some (.sc .i64 vp))
    (hvlen : Γ[vlen]? = some (.sc .i64 (UInt64.ofNat vb.length)))
    (h : Tab w sv T L) (hI : Idx w L (keysOf T)) (hkb : Bytes w kp kb) (hvb : Bytes w vp vb)
    (ck : kb ≠ [] → Clear sv L kp kb.length) (cv : vb ≠ [] → Clear sv L vp vb.length)
    (hk : ∀ Γ' w', Ext Γ Γ' → SetQ w sv T L kb vb w' → Q () Γ' w') :
    wp cfg d J (LibHt.set s key klen val vlen) Q Γ w := by
  have z0 : c64 0 = 0 := by decide
  have hr := h.room
  have ha36 := Blk.le36 h.ba
  have hvl := hvb.len_le
  have hl : (UInt64.ofNat vb.length).toNat = vb.length := by rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  have hEW : c64 (ENTRY : Nat) = UInt64.ofNat ENTRY := by decide
  unfold LibHt.set entryOf valAt
  wp_seq
  refine wp_hash (bs := kb) (by slot) (by slot) hkb ?_
  intro Γ1 r hE1 hr1
  refine wp_find (kb := kb) (L := L) (ks := keysOf T) (by slot) (by slot) (by slot) hr1 (h.probe hI) hkb ?_
  intro Γ2 ra rp slotA pos hE2 hra hrp hF
  apply wp_iconst64
  obtain ⟨i, hi, hsl, hF⟩ := hF
  refine wp_ifte (fun _ Γ' w' => Ext Γ Γ' ∧ SetQ w sv T L kb vb w') (x := pos) (y := c64 0) (by slot) (by slot)
    ?_ ?_ ?_
  · intro hp0
    have hp0' : pos = 0 := by simpa [cmpInt, z0] using hp0
    have hF' : fld w L.ix (8 * i.toNat) = some 0 ∧ kb ∉ keysOf T ∧ ∃ t, i = path L.icap (H kb) t ∧
        ∀ t', t' < t → fld w L.ix (8 * (path L.icap (H kb) t').toNat) ≠ some 0 := by
      rcases hF with ⟨-, hemp, hnot, t, hpath, hbef⟩ | ⟨p, hp, hpos, -⟩
      · exact ⟨hemp, hnot, t, hpath, hbef⟩
      · exfalso
        have hsm := hI.small
        have := congrArg UInt64.toNat (hpos.symm.trans hp0')
        rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)] at this
        simp at this
    obtain ⟨hemp, hnot, t, hpath, hbef⟩ := hF'
    wp_seq
    refine wp_insertAt (kb := kb) (vb := vb) (by slot) (by slot) (by slot) (by slot) (by slot) (by slot) (by slot) h
      ⟨hI, hi, hemp, hpath, hbef⟩ hnot hsl hkb hvb (fun e => (ck e).src h) (fun e => (cv e).src h) ?_
    rintro Γ' w' hE ⟨L', K, hR⟩
    refine ⟨Ext.trans (Ext.trans (Ext.trans hE1 hE2) (by ext_env)) hE, L', K, hR.imp ?_ id⟩
    rintro ⟨hT, hIx⟩
    rw [set_absent hnot, keysOf_pushed]
    exact ⟨hT, hIx⟩
  · intro hpne ne hne
    have hpne' : pos ≠ 0 := by simpa [cmpInt, z0] using hpne
    rcases hF with ⟨hpos, -⟩ | ⟨p, hp, hpos, hkey⟩
    · exact absurd hpos hpne'
    subst hpos
    have hp' : p < T.entries.length := by simpa [keysOf] using hp
    have hkey' : (T.entries[p]'hp').1.data.toList = kb := by simpa [keysOf] using hkey
    have ho : p < L.offs.length := by rw [h.olen]; exact hp'
    have hset := set_present (vb := vb) hp' hI.nodup hkey'
    obtain ⟨f0, f8, f16, f24, hko, hvo, bk, bv⟩ := h.ento p hp' ho
    have eA : ∀ e0 : UInt64, e0 + (UInt64.ofNat (p + 1) + c64 (-1)) * c64 (ENTRY : Nat) =
        e0 + UInt64.ofNat (p * ENTRY) := fun e0 => by
      rw [UInt64.ofNat_add, UInt64.add_assoc, show UInt64.ofNat 1 + c64 (-1) = 0 from by decide,
        UInt64.add_zero, hEW, ← UInt64.ofNat_mul]
    have e8 : ∀ (e0 : UInt64) (j : Nat) (z : Int), c64 z = UInt64.ofNat j →
        e0 + UInt64.ofNat (p * ENTRY) + c64 z = e0 + UInt64.ofNat (p * ENTRY + j) := fun e0 j z hz => by
      rw [UInt64.add_assoc, hz, ← UInt64.ofNat_add]
    wp_seq
    refine wp_at (x := sv) (by slot) ?_
    refine wp_load64 (by slot) h.fe ?_
    refine wp_iaddImm (x := UInt64.ofNat (p + 1)) (by slot) ?_
    apply wp_iconst64
    refine wp_imul (by slot) (by slot) ?_
    refine wp_iadd (x := L.e) (by slot) (by slot) ?_
    rw [eA]
    refine wp_iaddImm (x := L.e + UInt64.ofNat (p * ENTRY)) (by slot) ?_
    refine wp_load64 (by slot) (by rw [e8 _ 24 24 (by decide)]; exact f24) ?_
    refine wp_ifte (fun _ Γ' w' => Ext Γ Γ' ∧ SetQ w sv T L kb vb w')
      (x := UInt64.ofNat (T.entries[p]'hp').2.size) (y := UInt64.ofNat vb.length) (by slot) (by slot) ?_ ?_ ?_
    · intro heq
      have hsz : vb.length = (T.entries[p]'hp').2.size := by
        have h1 : UInt64.ofNat (T.entries[p]'hp').2.size = UInt64.ofNat vb.length := by simpa [cmpInt] using heq
        have := congrArg UInt64.toNat h1
        rw [UInt64.toNat_ofNat', UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
          at this
        exact this.symm
      have hj : (L.offs[p]).2 + vb.length ≤ L.acap := by omega
      have hap : (L.ar + UInt64.ofNat (L.offs[p]).2).toNat + vb.length ≤ vp.toNat ∨
          vp.toNat + vb.length ≤ (L.ar + UInt64.ofNat (L.offs[p]).2).toNat := by
        by_cases e : vb = []
        · subst e; simp; omega
        · exact apart_bytes hvb h.ba hj (cv e L.ar L.acap (by simp [blocks]))
      wp_seq
      refine wp_at (x := sv) (by slot) ?_
      refine wp_load64 (by slot) h.far ?_
      refine wp_iaddImm (x := L.e + UInt64.ofNat (p * ENTRY)) (by slot) ?_
      refine wp_load64 (by slot) (by rw [e8 _ 16 16 (by decide)]; exact f16) ?_
      refine wp_iadd (x := L.ar) (by slot) (by slot) ?_
      refine wp_memcpy (bs := vb) (by slot) (by slot) (by slot) hvb (by rw [hl])
        (by rw [hl]; exact Blk.room h.ba h.rep.calm hj) (by rw [hl]; exact hap) ?_
      intro w1 W1 B1 _
      rw [hl] at W1
      refine ⟨Ext.trans (Ext.trans hE1 hE2) (by ext_env), L, Kept.wrote (y := L.ar) (k := L.acap) (by simp [blocks]) h.ba W1 hj,
        Or.inl ?_⟩
      rw [hset, keysOf_setVal]
      exact ⟨h.setInPlace hp' ho hsz (Nat.le_refl _) W1 B1,
        hI.wrote h.bi W1 (h.apart.2.2.2.2.2.symm.clear h.ba hj)⟩
    · intro hneq ne2 hne2
      wp_seq
      refine wp_append (bs := vb) (by slot) (by slot) (by slot) h hvb (fun e => cv e L.ar L.acap (by simp [blocks]))
        (fun e => Off.part h.rep.blk (cv e sv SIZE (by simp [blocks])) (by decide) (by decide)) ?_
      intro Γ3 w3 r3 x3 hE3 hr3 hA
      apply wp_iconst64
      rcases hA with ⟨rfl, g⟩ | ⟨rfl, L1, A⟩
      · refine wp_when (SetQ w sv T L kb vb) (x := c64 (-1)) (y := c64 (-1)) (by slot) (by slot) ?_ ?_ ?_
        · intro hne'
          exact absurd hne' (by simp [cmpInt])
        · intro _
          exact ⟨L, (h.grown g).kept h, Or.inr ⟨h.appended (h.grown g), hI.appended h (h.grown g)⟩⟩
        · intro Γ' w' hE hR
          exact ⟨Ext.trans (Ext.trans (Ext.trans (Ext.trans hE1 hE2) (by ext_env)) hE3) (Ext.trans (by ext_env) hE), hR⟩
      · have hT1 := h.appended A
        have K1 := A.kept h
        have I1 := hI.appended h A
        have ho1 : p < L1.offs.length := by rw [A.offs]; exact ho
        have hU : ∀ i (hi : i < T.entries.length) (oi : i < L1.offs.length),
            (L1.offs[i]).1 + (T.entries[i]).1.size ≤ L.used ∧ (L1.offs[i]).2 + (T.entries[i]).2.size ≤ L.used := by
          rw [A.offs]
          exact fun i hi oi => ⟨(h.ento i hi oi).2.2.2.2.1, (h.ento i hi oi).2.2.2.2.2.1⟩
        have he1 := Blk.le36 hT1.be
        simp only [ENTRY] at he1
        have hjc : ∀ c, c + 8 ≤ 32 → p * ENTRY + c + 8 ≤ L1.ecap * ENTRY := by
          intro c hc
          have := hT1.room.1
          simp only [ENTRY]
          omega
        refine wp_when (SetQ w sv T L kb vb) (x := UInt64.ofNat L.used) (y := c64 (-1)) (by slot) (by slot)
          ?_ ?_ ?_
        · intro _
          wp_seq
          refine wp_at (x := sv) (by slot) ?_
          refine wp_load64 (by slot) hT1.fe ?_
          refine wp_iaddImm (x := UInt64.ofNat (p + 1)) (by slot) ?_
          apply wp_iconst64
          refine wp_imul (by slot) (by slot) ?_
          refine wp_iadd (x := L1.e) (by slot) (by slot) ?_
          rw [eA]
          refine wp_iaddImm (x := L1.e + UInt64.ofNat (p * ENTRY)) (by slot) ?_
          rw [e8 _ 16 16 (by decide)]
          obtain ⟨m1, hm1⟩ := Blk.fld_store hT1.be hT1.rep.calm (j := p * ENTRY + 16) (hjc 16 (by decide))
            (UInt64.ofNat L.used)
          refine wp_storeI64 (by slot) (by slot) hm1 ?_
          refine wp_iaddImm (x := L1.e + UInt64.ofNat (p * ENTRY)) (by slot) ?_
          rw [e8 _ 24 24 (by decide)]
          obtain ⟨m2, hm2⟩ := Blk.fld_store (Blk.keep hT1.be hm1) (hT1.rep.calm.stored hm1)
            (j := p * ENTRY + 24) (hjc 24 (by decide)) (UInt64.ofNat vb.length)
          refine wp_storeI64 (by slot) (by slot) hm2 ?_
          have W1 := wrote_stored hm1
          have W2 := wrote_stored hm2
          have Ka := Kept.wrote (s := sv) (L := L1) (y := L1.e) (k := L1.ecap * ENTRY) (by simp [blocks]) hT1.be W1
            (hjc 16 (by decide))
          have Kb := Kept.wrote (s := sv) (L := L1) (y := L1.e) (k := L1.ecap * ENTRY) (by simp [blocks])
            (Blk.keep hT1.be hm1) W2 (hjc 24 (by decide))
          have Ia := I1.wrote hT1.bi W1 (hT1.apart.2.2.2.1.clear hT1.be (hjc 16 (by decide)))
          have Ib := Ia.wrote (Blk.keep hT1.bi hm1) W2
            (hT1.apart.2.2.2.1.clear (Blk.keep hT1.be hm1) (hjc 24 (by decide)))
          refine ⟨Ext.trans (by ext_env) (Ext.refl _), { L1 with offs := L1.offs.set p ((L1.offs[p]).1, L.used) },
            ((K1.trans Ka).trans Kb).lay rfl, Or.inl ?_⟩
          rw [hset, keysOf_setVal]
          exact ⟨hT1.setMoved hp' ho1 hU (by rw [A.used]; omega) A.new hm1 hm2, Ib.lay rfl rfl⟩
        · intro he
          exact absurd (by simpa [cmpInt] using he) (ofNat_ne_neg1 (n := L.used) (by omega))
        · intro Γ' w' hE hR
          exact ⟨Ext.trans (Ext.trans (Ext.trans (Ext.trans hE1 hE2) (by ext_env)) hE3) (Ext.trans (by ext_env) hE), hR⟩
    · rintro nJ Γ' jv vs w' hle hvs ⟨hE, hR⟩
      exact ⟨hE.bindAt hle _, hR⟩
  · rintro nJ Γ' jv vs w' hle hvs ⟨hE, hR⟩
    exact hk _ _ (hE.bindAt hle _) hR

end fns

-- ---------------------------------------------------------------------------
-- `htInsert`
-- ---------------------------------------------------------------------------

theorem insert_fine : Fine (emitGo LibHt.insert ⟨5, 0, [], [], [], none⟩).2 := ⟨by decide +kernel, by decide +kernel⟩

/-- What `htInsert` leaves, for a context holding `T`: what `set` leaves, or
    the world as it was before a handle has been made. -/
def InsertQ (w : World) (s : UInt64) (T : Ht) (L : Lay) (kb vb : List UInt8) (w' : World) : Prop :=
  if T.hasTable then SetQ w s T L kb vb w' else w' = w

/-- **`htInsert`** of a key and a value that can be read, clear of the table,
    leaves the table the model's `set` leaves, or the table as it was where
    something could not be allocated; before a handle has been made it changes
    nothing. -/
theorem insert_spec (cfg : Cfg) (w : World) (s : UInt64) (T : Ht) (L : Lay) (h : Tab w s T L)
    (hI : Idx w L (keysOf T)) (key klen val vlen : UInt64) (kb vb : List UInt8) (hkb : Bytes w key kb)
    (hvb : Bytes w val vb) (hkl : klen = UInt64.ofNat kb.length) (hk32 : kb.length < 2 ^ 32)
    (hvl : vlen = UInt64.ofNat vb.length) (hv32 : vb.length < 2 ^ 32)
    (ck : kb ≠ [] → Clear s L key kb.length) (cv : vb ≠ [] → Clear s L val vb.length) :
    Does cfg [.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 val, .sc .i32 vlen] w LibHt.insert
      [.i64, .i64, .i32, .i64, .i32] (InsertQ w s T L kb vb) := by
  refine does_of_wp ?_ insert_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [LibHt.insert, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 val, .sc .i32 vlen] : Env)[0]? =
    some (.sc .i64 s) := rfl
  have h1 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 val, .sc .i32 vlen] : Env)[0 + 1]? =
    some (.sc .i64 key) := rfl
  have h2 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 val, .sc .i32 vlen] : Env)[0 + 1 + 1]? =
    some (.sc .i32 klen) := rfl
  have h3 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 val, .sc .i32 vlen] : Env)[0 + 1 + 1 + 1]? =
    some (.sc .i64 val) := rfl
  have h4 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 val, .sc .i32 vlen] : Env)[0 + 1 + 1 + 1 + 1]? =
    some (.sc .i32 vlen) := rfl
  have mask : ∀ n, n < 2 ^ 32 → UInt64.ofNat n &&& 0xFFFFFFFF = UInt64.ofNat n := by
    intro n hn
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_and, show (0xFFFFFFFF : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
      UInt64.toNat_ofNat']
    omega
  refine wp_mono cfg _ 0 _ (AnsQ fun _ w' => InsertQ w s T L kb vb w') _ (fun a Γ' w' hq => ?_) _ _ ?_
  · obtain ⟨_, _, hq⟩ := hq
    exact hq
  refine wp_withTable (c := UInt64.ofNat T.handles.length) (by slot) (fun _ => h.rep.created) ?_ ?_
  · intro hz Γ' hext
    have hl : T.handles.length = 0 := by
      rcases hz with hz | hz
      · exact absurd hz (Blk.ne_zero h.rep.blk)
      · have := congrArg UInt64.toNat hz
        rwa [UInt64.toNat_ofNat', Nat.mod_eq_of_lt h.rep.small.1] at this
    apply wp_iconst64
    exact ⟨_, get_push_size, by simp [InsertQ, h.rep.hasTable, hl]⟩
  · intro _ hc Γ' hext
    have hl : T.handles.length ≠ 0 := fun e => hc (by rw [e]; rfl)
    have hT : T.hasTable = true := by simp [h.rep.hasTable, hl]
    wp_seq
    refine wp_uextend64 (x := klen) (by slot) ?_
    rw [hkl, mask _ hk32]
    refine wp_uextend64 (x := vlen) (by slot) ?_
    rw [hvl, mask _ hv32]
    refine wp_set (kb := kb) (vb := vb) (by slot) (by slot) (by slot) (by slot) (by slot) h hI hkb hvb ck cv ?_
    intro Γ2 w2 hE hR
    apply wp_iconst64
    refine ⟨_, get_push_size, ?_⟩
    show InsertQ w s T L kb vb w2
    unfold InsertQ
    rw [if_pos hT]
    exact hR

/-- **`htInsert`** handed no context changes nothing. -/
theorem insert_null (cfg : Cfg) (w : World) (key klen val vlen : UInt64) :
    Does cfg [.sc .i64 0, .sc .i64 key, .sc .i32 klen, .sc .i64 val, .sc .i32 vlen] w LibHt.insert
      [.i64, .i64, .i32, .i64, .i32] (fun w' => w' = w) := by
  refine does_of_wp ?_ insert_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [LibHt.insert, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 0, .sc .i64 key, .sc .i32 klen, .sc .i64 val, .sc .i32 vlen] : Env)[0]? =
    some (.sc .i64 0) := rfl
  refine wp_mono cfg _ 0 _ (AnsQ fun _ w' => w' = w) _ (fun a Γ' w' hq => ?_) _ _ ?_
  · obtain ⟨_, _, hq⟩ := hq
    exact hq
  refine wp_withTable (x := 0) (c := 0) (by slot) (fun h => absurd rfl h) ?_ (fun h => absurd rfl h)
  intro _ Γ' hext
  apply wp_iconst64
  exact ⟨_, get_push_size, rfl⟩

-- ---------------------------------------------------------------------------
-- `htIncrement`
-- ---------------------------------------------------------------------------

theorem increment_fine : Fine (emitGo increment ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- The value the model leaves for a counter with `addend` added: its first
    eight bytes the sum, the rest as they were. -/
def bumped (ex : ByteArray) (addend : UInt64) : ByteArray :=
  le64 (ofLe64 ex + addend) ++ ((List.range (ex.size - 8)).map (fun i => ex.get! (8 + i))).toByteArray

theorem bumped_eq (ex : ByteArray) (addend : UInt64) (h8 : 8 ≤ ex.size) :
    bumped ex addend = ByteArray.mk (le8 (ofLe64 ex + addend) ++ ex.data.toList.drop 8).toArray := by
  have ht : (List.range (ex.size - 8)).map (fun i => ex.get! (8 + i)) = ex.data.toList.drop 8 := by
    apply List.ext_getElem
    · rw [List.length_map, List.length_range, List.length_drop, Array.length_toList]
      rfl
    · intro i h1 h2
      have hi : 8 + i < ex.data.size := by
        have : i < ex.size - 8 := by simpa using h1
        have : ex.size = ex.data.size := rfl
        omega
      simp only [List.getElem_map, List.getElem_range, List.getElem_drop, Array.getElem_toList, ByteArray.get!]
      exact getElem!_pos ex.data (8 + i) hi
  unfold bumped
  rw [le64_eq, ht]
  apply ByteArray.ext
  rw [ByteArray.data_append, List.data_toByteArray]
  simp

/-- **What `htIncrement` answers and leaves**, for a context holding `T`: the
    addend and a new entry holding it where no entry has the key, the sum and
    the entry's value with the sum in its first eight bytes where one has; the
    addend and nothing changed before a handle has been made. -/
def IncQ (w : World) (s : UInt64) (T : Ht) (L : Lay) (kb : List UInt8) (addend x : UInt64) (w' : World) : Prop :=
  if T.hasTable then
    match T.get (ByteArray.mk kb.toArray) with
    | none => x = addend ∧ SetQ w s T L kb (le8 addend) w'
    | some ex => x = ofLe64 ex + addend ∧ ∃ L', Kept w w' s L L' ∧
        Tab w' s (T.set (ByteArray.mk kb.toArray) (bumped ex addend)) L' ∧
        Idx w' L' (keysOf (T.set (ByteArray.mk kb.toArray) (bumped ex addend)))
  else x = addend ∧ w' = w

set_option maxHeartbeats 3200000 in
/-- **`htIncrement`** of a key that can be read, clear of the table, whose
    value if it has one is at least eight bytes: the model's answer and the
    model's table, or, where a new entry could not be allocated, the table as
    it was. -/
theorem increment_spec (cfg : Cfg) (w : World) (s : UInt64) (T : Ht) (L : Lay) (h : Tab w s T L)
    (hI : Idx w L (keysOf T)) (key klen addend : UInt64) (kb : List UInt8) (hkb : Bytes w key kb)
    (hkl : klen = UInt64.ofNat kb.length) (hk32 : kb.length < 2 ^ 32) (ck : kb ≠ [] → Clear s L key kb.length)
    (h8 : ∀ ex, T.get (ByteArray.mk kb.toArray) = some ex → 8 ≤ ex.size) :
    Ans cfg [.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 addend] w increment [.i64, .i64, .i32, .i64]
      (IncQ w s T L kb addend) := by
  refine ans_of_wp ?_ increment_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [increment, entryOf, valAt, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 addend] : Env)[0]? = some (.sc .i64 s) := rfl
  have h1 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 addend] : Env)[0 + 1]? =
    some (.sc .i64 key) := rfl
  have h2 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 addend] : Env)[0 + 1 + 1]? =
    some (.sc .i32 klen) := rfl
  have h3 : (#[V.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 addend] : Env)[0 + 1 + 1 + 1]? =
    some (.sc .i64 addend) := rfl
  have z0 : c64 0 = 0 := by decide
  have hr := h.room
  have ha36 := Blk.le36 h.ba
  have hEW : c64 (ENTRY : Nat) = UInt64.ofNat ENTRY := by decide
  refine wp_withTable (c := UInt64.ofNat T.handles.length) (by slot) (fun _ => h.rep.created) ?_ ?_
  · intro hz Γ' hext
    have hl : T.handles.length = 0 := by
      rcases hz with hz | hz
      · exact absurd hz (Blk.ne_zero h.rep.blk)
      · have := congrArg UInt64.toNat hz
        rwa [UInt64.toNat_ofNat', Nat.mod_eq_of_lt h.rep.small.1] at this
    wp_seq
    exact ⟨addend, by slot, by simp [IncQ, h.rep.hasTable, hl]⟩
  · intro _ hc Γ' hext
    have hl : T.handles.length ≠ 0 := fun e => hc (by rw [e]; rfl)
    have hT : T.hasTable = true := by simp [h.rep.hasTable, hl]
    have hmask : klen &&& 0xFFFFFFFF = UInt64.ofNat kb.length := by
      rw [hkl]
      apply UInt64.toNat_inj.mp
      rw [UInt64.toNat_and, show (0xFFFFFFFF : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
        UInt64.toNat_ofNat']
      omega
    wp_seq
    refine wp_uextend64 (x := klen) (by slot) ?_
    rw [hmask]
    refine wp_hash (bs := kb) (by slot) (by slot) hkb ?_
    intro Γ1 r hE1 hr1
    refine wp_find (kb := kb) (L := L) (ks := keysOf T) (by slot) (by slot) (by slot) hr1 (h.probe hI) hkb ?_
    intro Γ2 ra rp slotA pos hE2 hra hrp hF
    apply wp_iconst64
    obtain ⟨i, hi, hsl, hF⟩ := hF
    refine wp_ifte (AnsJ (IncQ w s T L kb addend)) (x := pos) (y := c64 0) (by slot) (by slot) ?_ ?_
      (by join_ans)
    · intro hp0
      have hp0' : pos = 0 := by simpa [cmpInt, z0] using hp0
      have hF' : fld w L.ix (8 * i.toNat) = some 0 ∧ kb ∉ keysOf T ∧ ∃ t, i = path L.icap (H kb) t ∧
          ∀ t', t' < t → fld w L.ix (8 * (path L.icap (H kb) t').toNat) ≠ some 0 := by
        rcases hF with ⟨-, hemp, hnot, t, hpath, hbef⟩ | ⟨p, hp, hpos, -⟩
        · exact ⟨hemp, hnot, t, hpath, hbef⟩
        · exfalso
          have hsm := hI.small
          have := congrArg UInt64.toNat (hpos.symm.trans hp0')
          rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)] at this
          simp at this
      obtain ⟨hemp, hnot, t, hpath, hbef⟩ := hF'
      obtain ⟨m0, hm0⟩ := Blk.fld_store h.rep.blk h.rep.calm (j := SCRATCH) (by decide) addend
      have W0 := wrote_stored hm0
      have hT0 := h.scratch hm0
      have S0 : Spot (stored w (s + UInt64.ofNat SCRATCH) addend m0) L (keysOf T) kb i t :=
        (⟨hI, hi, hemp, hpath, hbef⟩ : Spot w L (keysOf T) kb i t).wrote h.bi W0
          (h.apart.2.1.clear h.rep.blk (by decide))
      have K0 := Kept.wrote (s := s) (L := L) (y := s) (k := SIZE) (by simp [blocks]) h.rep.blk W0 (by decide)
      obtain ⟨cs, hcl, hcs⟩ := (Blk.room h.rep.blk h.rep.calm (j := SCRATCH) (k := 8) (by decide)).bytes
      have hb0 : Bytes (stored w (s + UInt64.ofNat SCRATCH) addend m0) (s + UInt64.ofNat SCRATCH) (le8 addend) := by
        have := stored_bytes hcs (by omega) hm0
        rwa [List.drop_of_length_le (by omega), List.append_nil] at this
      have hkb0 : Bytes (stored w (s + UInt64.ofNat SCRATCH) addend m0) key kb := by
        by_cases e : kb = []
        · exact Or.inl e
        · exact W0.bytes hkb fun _ => Off.wrote h.rep.blk (by decide) (ck e s SIZE (by simp [blocks]))
      wp_seq
      refine wp_at (x := s) (by slot) ?_
      refine wp_storeI64 (by slot) (by slot) hm0 ?_
      apply wp_iconst64
      rw [show c64 8 = UInt64.ofNat 8 from by decide]
      refine wp_insertAt (kb := kb) (vb := le8 addend) (by slot) (by slot) (by slot) (by slot) (by slot)
        (by rw [le8_length]; slot) (by slot) hT0 S0 hnot hsl hkb0 hb0 (fun e => (ck e).src h) (fun _ => by
          rw [le8_length]
          exact ⟨h.apart.1.clear h.rep.blk (by decide), h.apart.2.2.1.clear h.rep.blk (by decide),
            off_same h.rep.blk (by decide) (by decide) (Or.inr (by decide)),
            off_same h.rep.blk (by decide) (by decide) (Or.inl (by decide))⟩) ?_
      rintro Γ3 w3 hE3 ⟨L', K, hR⟩
      refine ⟨addend, by slot, ?_⟩
      show IncQ w s T L kb addend addend w3
      simp only [IncQ, hT, if_true, get_none hnot]
      refine ⟨trivial, L', K0.trans K, hR.imp ?_ id⟩
      rintro ⟨hTa, hIa⟩
      rw [set_absent hnot, keysOf_pushed]
      exact ⟨hTa, hIa⟩
    · intro hpne ne hne
      have hpne' : pos ≠ 0 := by simpa [cmpInt, z0] using hpne
      rcases hF with ⟨hpos, -⟩ | ⟨p, hp, hpos, hkey⟩
      · exact absurd hpos hpne'
      subst hpos
      have hp' : p < T.entries.length := by simpa [keysOf] using hp
      have hkey' : (T.entries[p]'hp').1.data.toList = kb := by simpa [keysOf] using hkey
      have ho : p < L.offs.length := by rw [h.olen]; exact hp'
      have hget := get_some hp' hI.nodup hkey'
      have hex := h8 _ hget
      obtain ⟨f0, f8, f16, f24, hko, hvo, bk, bv⟩ := h.ento p hp' ho
      have hlen8 : 8 ≤ (T.entries[p]'hp').2.data.toList.length := by rw [size_toList]; exact hex
      have hj : (L.offs[p]).2 + 8 ≤ L.acap := by omega
      have hld : w.mem.load (L.ar + UInt64.ofNat (L.offs[p]).2) 8 =
          some (leN (T.entries[p]'hp').2.data.toList 0 8) := by
        have := bv.load (j := 0) (k := 8) (by omega) (by decide)
        rwa [add_ofNat_zero] at this
      have eA : ∀ e0 : UInt64, e0 + (UInt64.ofNat (p + 1) + c64 (-1)) * c64 (ENTRY : Nat) =
          e0 + UInt64.ofNat (p * ENTRY) := fun e0 => by
        rw [UInt64.ofNat_add, UInt64.add_assoc, show UInt64.ofNat 1 + c64 (-1) = 0 from by decide,
          UInt64.add_zero, hEW, ← UInt64.ofNat_mul]
      have e16 : L.e + UInt64.ofNat (p * ENTRY) + c64 16 = L.e + UInt64.ofNat (p * ENTRY + 16) := by
        rw [UInt64.add_assoc, show c64 16 = UInt64.ofNat 16 from by decide, ← UInt64.ofNat_add]
      obtain ⟨m1, hm1⟩ := Blk.fld_store h.ba h.rep.calm (j := (L.offs[p]).2) hj
        (leN (T.entries[p]'hp').2.data.toList 0 8 + addend)
      have W1 := wrote_stored hm1
      have B1 := stored_bytes bv hlen8 hm1
      wp_seq
      refine wp_at (x := s) (by slot) ?_
      refine wp_load64 (by slot) h.fe ?_
      refine wp_iaddImm (x := UInt64.ofNat (p + 1)) (by slot) ?_
      apply wp_iconst64
      refine wp_imul (by slot) (by slot) ?_
      refine wp_iadd (x := L.e) (by slot) (by slot) ?_
      rw [eA]
      refine wp_at (x := s) (by slot) ?_
      refine wp_load64 (by slot) h.far ?_
      refine wp_iaddImm (x := L.e + UInt64.ofNat (p * ENTRY)) (by slot) ?_
      refine wp_load64 (by slot) (by rw [e16]; exact f16) ?_
      refine wp_iadd (x := L.ar) (by slot) (by slot) ?_
      refine wp_load64 (by slot) hld ?_
      refine wp_iadd (by slot) (by slot) ?_
      refine wp_storeI64 (by slot) (by slot) hm1 ?_
      refine ⟨_, by slot, ?_⟩
      show IncQ w s T L kb addend _ _
      simp only [IncQ, hT, if_true, hget]
      have hset : T.set (ByteArray.mk kb.toArray) (bumped (T.entries[p]'hp').2 addend) =
          setVal T p (le8 (leN (T.entries[p]'hp').2.data.toList 0 8 + addend) ++
            (T.entries[p]'hp').2.data.toList.drop 8) := by
        rw [bumped_eq _ _ hex, ofLe64_data]
        exact set_present hp' hI.nodup hkey'
      refine ⟨by rw [ofLe64_data], L, Kept.wrote (y := L.ar) (k := L.acap) (by simp [blocks]) h.ba W1 hj, ?_⟩
      rw [hset, keysOf_setVal]
      exact ⟨h.setInPlace hp' ho (by simp [le8_length]; omega) (by simp [le8_length]) W1 B1,
        hI.wrote h.bi W1 (h.apart.2.2.2.2.2.symm.clear h.ba hj)⟩

/-- **`htIncrement`** handed no context answers the addend and changes
    nothing. -/
theorem increment_null (cfg : Cfg) (w : World) (key klen addend : UInt64) :
    Ans cfg [.sc .i64 0, .sc .i64 key, .sc .i32 klen, .sc .i64 addend] w increment [.i64, .i64, .i32, .i64]
      (fun x w' => x = addend ∧ w' = w) := by
  refine ans_of_wp ?_ increment_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [increment, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 0, .sc .i64 key, .sc .i32 klen, .sc .i64 addend] : Env)[0]? = some (.sc .i64 0) := rfl
  have h3 : (#[V.sc .i64 0, .sc .i64 key, .sc .i32 klen, .sc .i64 addend] : Env)[0 + 1 + 1 + 1]? =
    some (.sc .i64 addend) := rfl
  refine wp_withTable (x := 0) (c := 0) (by slot) (fun h => absurd rfl h) ?_ (fun h => absurd rfl h)
  intro _ Γ' hext
  wp_seq
  exact ⟨addend, by slot, rfl, rfl⟩

-- ---------------------------------------------------------------------------
-- The table's functions, called from a program
-- ---------------------------------------------------------------------------

section calls
variable {fns : Nat → Option Fn} {env : FnEnv} {steps i : Nat}

/-- **A linked program's calls to the table's functions** are as the
    statements above say, at every call depth. -/
theorem init_calls (hf : fns i = some (implFn [.i64] (.void init))) (w : World) (slot : UInt64) (r : Region)
    (off : Nat) (hc : Calm w) (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned)
    (hroom : off + 8 ≤ (w.mem.region r).size) (k : Nat) :
    CallsD (termLocals fns env steps k) i [.sc .i64 slot] w (fun w' =>
      ∃ s, w'.mem.load slot 8 = some s ∧ (s = 0 ∨ ∃ L, Tab w' s {} L ∧ Idx w' L []) ∧
        Outside w w' (Misses slot)) :=
  Does.calls hf (fun cfg => init_spec cfg w slot r off hc hd hr hroom) k

theorem count_calls (hf : fns i = some (implFn [.i64] (.status count))) (w : World) (s : UInt64) (T : Ht)
    (h : Rep w s T) (k : Nat) :
    CallsA (termLocals fns env steps k) i [.sc .i64 s] w (fun x w' =>
      x = (if T.hasTable then UInt64.ofNat T.entries.length else 0) ∧ w' = w) :=
  Ans.calls hf (fun cfg => count_spec cfg w s T h) k

theorem getEntry_calls (hf : fns i = some (implFn [.i64, .i32, .i64, .i64] (.status getEntry))) (w : World)
    (s : UInt64) (T : Ht) (L : Lay) (h : Tab w s T L) (index kout vout : UInt64) (hix : index.toNat < 2 ^ 32)
    (hroom : ∀ k v, T.entries[index.toNat]? = some (k, v) →
      Room w kout k.size ∧ Room w vout v.size ∧ Clear s L kout k.size ∧ Clear s L vout v.size) (k : Nat) :
    CallsA (termLocals fns env steps k) i [.sc .i64 s, .sc .i32 index, .sc .i64 kout, .sc .i64 vout] w
      (GetQ w s T L index kout vout) :=
  Ans.calls hf (fun cfg => getEntry_spec cfg w s T L h index kout vout hix hroom) k

theorem create_calls (hf : fns i = some (implFn [.i64] (.status create))) (w : World) (s : UInt64) (T : Ht)
    (h : Rep w s T) (hn : T.handles.length + 1 < 2 ^ 64) (k : Nat) :
    CallsA (termLocals fns env steps k) i [.sc .i64 s] w (fun x w' => x = UInt64.ofNat T.handles.length ∧
      Rep w' s { T with handles := T.handles ++ [T.handles.length] } ∧
      ∀ e i a, Owned w s e i a → Owned w' s e i a) :=
  Ans.calls hf (fun cfg => create_spec cfg w s T h hn) k

theorem cleanup_calls (hf : fns i = some (implFn [.i64] (.void cleanup))) (w : World) (slot s : UInt64)
    (r : Region) (off : Nat) (hc : Calm w) (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned)
    (hl : w.mem.load slot 8 = some s) (hs : s = 0 ∨ ∃ e i a, Owned w s e i a) (k : Nat) :
    CallsD (termLocals fns env steps k) i [.sc .i64 slot] w (fun w' =>
      w'.mem.load slot 8 = some 0 ∧ Outside w w' (Misses slot)) :=
  Does.calls hf (fun cfg => cleanup_spec cfg w slot s r off hc hd hr hl hs) k

theorem lookup_calls (hf : fns i = some (implFn [.i64, .i64, .i32, .i64] (.status lookup))) (w : World)
    (s : UInt64) (T : Ht) (L : Lay) (h : Tab w s T L) (hI : Idx w L (keysOf T)) (key klen out : UInt64)
    (kb : List UInt8) (hB : Bytes w key kb) (hkl : klen = UInt64.ofNat kb.length) (hk32 : kb.length < 2 ^ 32)
    (hroom : ∀ p (hp : p < T.entries.length), (T.entries[p]'hp).1.data.toList = kb →
      Room w out (T.entries[p]'hp).2.size ∧ Clear s L out (T.entries[p]'hp).2.size) (k : Nat) :
    CallsA (termLocals fns env steps k) i [.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 out] w
      (LookQ w s T L kb out) :=
  Ans.calls hf (fun cfg => lookup_spec cfg w s T L h hI key klen out kb hB hkl hk32 hroom) k

theorem insert_calls (hf : fns i = some (implFn [.i64, .i64, .i32, .i64, .i32] (.void LibHt.insert))) (w : World)
    (s : UInt64) (T : Ht) (L : Lay) (h : Tab w s T L) (hI : Idx w L (keysOf T)) (key klen val vlen : UInt64)
    (kb vb : List UInt8) (hkb : Bytes w key kb) (hvb : Bytes w val vb) (hkl : klen = UInt64.ofNat kb.length)
    (hk32 : kb.length < 2 ^ 32) (hvl : vlen = UInt64.ofNat vb.length) (hv32 : vb.length < 2 ^ 32)
    (ck : kb ≠ [] → Clear s L key kb.length) (cv : vb ≠ [] → Clear s L val vb.length) (k : Nat) :
    CallsD (termLocals fns env steps k) i
      [.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 val, .sc .i32 vlen] w (InsertQ w s T L kb vb) :=
  Does.calls hf (fun cfg => insert_spec cfg w s T L h hI key klen val vlen kb vb hkb hvb hkl hk32 hvl hv32 ck cv) k

theorem increment_calls (hf : fns i = some (implFn [.i64, .i64, .i32, .i64] (.status increment))) (w : World)
    (s : UInt64) (T : Ht) (L : Lay) (h : Tab w s T L) (hI : Idx w L (keysOf T)) (key klen addend : UInt64)
    (kb : List UInt8) (hkb : Bytes w key kb) (hkl : klen = UInt64.ofNat kb.length) (hk32 : kb.length < 2 ^ 32)
    (ck : kb ≠ [] → Clear s L key kb.length)
    (h8 : ∀ ex, T.get (ByteArray.mk kb.toArray) = some ex → 8 ≤ ex.size) (k : Nat) :
    CallsA (termLocals fns env steps k) i [.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 addend] w
      (IncQ w s T L kb addend) :=
  Ans.calls hf (fun cfg => increment_spec cfg w s T L h hI key klen addend kb hkb hkl hk32 ck h8) k

end calls

-- ---------------------------------------------------------------------------
-- The table at a program's slot
-- ---------------------------------------------------------------------------

/-- A handle made: the table with one more handle, its entries as they were. -/
theorem Tab.created {w : World} {s x : UInt64} {T : Ht} {L : Lay} {hs : List Nat} {m : Mem} (h : Tab w s T L)
    (hm : w.mem.store (s + UInt64.ofNat CREATED) 8 x = some m)
    (hrep : Rep (stored w (s + UInt64.ofNat CREATED) x m) s { T with handles := hs }) :
    Tab (stored w (s + UInt64.ofNat CREATED) x m) s { T with handles := hs } L := by
  have W := wrote_stored hm
  have hr := h.room
  have ce := h.apart.1.clear h.rep.blk (j := CREATED) (n := 8) (by decide)
  have ca := h.apart.2.2.1.clear h.rep.blk (j := CREATED) (n := 8) (by decide)
  have hf : ∀ j, j + 8 ≤ SIZE → CREATED + 8 ≤ j →
      fld (stored w (s + UInt64.ofNat CREATED) x m) s j = fld w s j :=
    fun j hj hd => fld_kept h.rep.blk hj (by decide) (Or.inr hd) hm
  refine {
    rep := hrep
    fe := (hf _ (by decide) (by decide)).trans h.fe
    fecap := (hf _ (by decide) (by decide)).trans h.fecap
    fix := (hf _ (by decide) (by decide)).trans h.fix
    ficap := (hf _ (by decide) (by decide)).trans h.ficap
    far := (hf _ (by decide) (by decide)).trans h.far
    fused := (hf _ (by decide) (by decide)).trans h.fused
    facap := (hf _ (by decide) (by decide)).trans h.facap
    we := W.whole h.we
    wi := W.whole h.wi
    wa := W.whole h.wa
    room := hr
    pos := h.pos
    apart := h.apart
    olen := h.olen
    sep := h.sep
    ento := fun i hi ho => ?_ }
  obtain ⟨h0, h8, h16, h24, hk, hv, bk, bv⟩ := h.ento i hi ho
  have hi' : i < L.ecap := Nat.lt_of_lt_of_le hi hr.1
  have he : ∀ j, j + 8 ≤ ENTRY → fld (stored w (s + UInt64.ofNat CREATED) x m) L.e (i * ENTRY + j) =
      fld w L.e (i * ENTRY + j) := fun j hj => W.fld_blk h.be (by simp only [ENTRY] at hj ⊢; omega) ce
  exact ⟨(he 0 (by decide)).trans h0, (he 8 (by decide)).trans h8, (he 16 (by decide)).trans h16,
    (he 24 (by decide)).trans h24, hk, hv,
    W.bytes_blk h.ba (by rw [size_toList]; exact Nat.le_trans hk hr.2) bk ca,
    W.bytes_blk h.ba (by rw [size_toList]; exact Nat.le_trans hv hr.2) bv ca⟩

/-- **`htCreate`** over a whole table: the handle answered, the table with it
    and its index, and nothing else changed. -/
theorem create_tab (cfg : Cfg) (w : World) (s : UInt64) (T : Ht) (L : Lay) (h : Tab w s T L)
    (hI : Idx w L (keysOf T)) (hn : T.handles.length + 1 < 2 ^ 64) :
    Ans cfg [.sc .i64 s] w create [.i64] (fun x w' => x = UInt64.ofNat T.handles.length ∧
      Tab w' s { T with handles := T.handles ++ [T.handles.length] } L ∧ Idx w' L (keysOf T) ∧
      Kept w w' s L L) := by
  refine ans_of_wp ?_ create_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [create, entryParams, wp_bind, wp_params, carriesFrom, wp_ret, wp_pure]
  have h0 : (#[V.sc .i64 s] : Env)[0]? = some (.sc .i64 s) := rfl
  apply wp_iconst64
  refine wp_ifte (AnsJ fun x w' => x = UInt64.ofNat T.handles.length ∧
      Tab w' s { T with handles := T.handles ++ [T.handles.length] } L ∧ Idx w' L (keysOf T) ∧ Kept w w' s L L)
    (x := s) (y := c64 0) (by slot) (by slot) ?_ ?_ (by join_ans)
  · intro he
    exact absurd (by simpa [cmpInt, show c64 0 = 0 from by decide] using he) (Blk.ne_zero h.rep.blk)
  · intro _ ne hne
    wp_seq
    refine wp_at (x := s) (by slot) ?_
    refine wp_load64 (by slot) h.rep.created ?_
    refine wp_iaddImm (by slot) ?_
    obtain ⟨m, hm⟩ := Blk.fld_store h.rep.blk h.rep.calm (j := CREATED) (by decide)
      (UInt64.ofNat T.handles.length + c64 1)
    refine wp_storeI64 (by slot) (by slot) hm ?_
    have W := wrote_stored hm
    refine ⟨_, by slot, rfl, h.created hm ?_, hI.wrote h.bi W (h.apart.2.1.clear h.rep.blk (by decide)),
      Kept.wrote (y := s) (k := SIZE) (by simp [blocks]) h.rep.blk W (by decide)⟩
    exact {
      blk := Blk.keep h.rep.blk hm
      calm := h.rep.calm.stored hm
      created := by
        rw [fld_stored hm]
        simp [show c64 1 = 1 from by decide, UInt64.ofNat_add]
      count := (fld_kept h.rep.blk (by decide) (by decide) (by decide) hm).trans h.rep.count
      handles := by
        show T.handles ++ [T.handles.length] = List.range (T.handles ++ [T.handles.length]).length
        rw [List.length_append, List.length_singleton, List.range_succ, ← h.rep.handles]
      small := ⟨by simpa using hn, h.rep.small.2⟩ }

/-- **`htLookup`** handed no context answers `0xFFFFFFFF` and changes
    nothing. -/
theorem lookup_null (cfg : Cfg) (w : World) (key klen out : UInt64) :
    Ans cfg [.sc .i64 0, .sc .i64 key, .sc .i32 klen, .sc .i64 out] w lookup [.i64, .i64, .i32, .i64]
      (fun x w' => x = c64 0xFFFFFFFF ∧ w' = w) := by
  refine ans_of_wp ?_ lookup_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [lookup, entryOf, valAt, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 0, .sc .i64 key, .sc .i32 klen, .sc .i64 out] : Env)[0]? = some (.sc .i64 0) := rfl
  refine wp_withTable (x := 0) (c := 0) (by slot) (fun h => absurd rfl h) ?_ (fun h => absurd rfl h)
  intro _ Γ' hext
  apply wp_iconst64
  exact ⟨_, get_push_size, rfl, rfl⟩

/-- **`htGetEntry`** handed no context answers `-1` and changes nothing. -/
theorem getEntry_null (cfg : Cfg) (w : World) (index kout vout : UInt64) :
    Ans cfg [.sc .i64 0, .sc .i32 index, .sc .i64 kout, .sc .i64 vout] w getEntry [.i64, .i32, .i64, .i64]
      (fun x w' => x = c64 (-1) ∧ w' = w) := by
  refine ans_of_wp ?_ getEntry_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [getEntry, entryOf, keyAt, valAt, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 0, .sc .i32 index, .sc .i64 kout, .sc .i64 vout] : Env)[0]? = some (.sc .i64 0) := rfl
  refine wp_withTable (x := 0) (c := 0) (by slot) (fun h => absurd rfl h) ?_ (fun h => absurd rfl h)
  intro _ Γ' hext
  apply wp_iconst64
  exact ⟨_, get_push_size, rfl, rfl⟩

theorem HtAt.load {w : World} {slot s : UInt64} {T : Ht} (h : HtAt w slot s T) : w.mem.load slot 8 = some s := by
  obtain ⟨_, _, _, _, _, hl, _, _⟩ := h
  exact hl

/-- A run outside the heap is clear of every table. -/
theorem clear_out {s p : UInt64} {L : Lay} {n : Nat} {r : Region} {off : Nat} (hd : decodeAddr p = some (r, off))
    (hr : r ≠ .pinned) : Clear s L p n := fun _ _ _ r' off' _ hp _ => by
  rw [hd] at hp
  cases hp
  exact Or.inl hr

/-- What a change to the table keeps, it keeps of the slot. -/
theorem HtAt.kept {w w' : World} {slot s : UInt64} {T' : Ht} {L L' : Lay} {r : Region} {off : Nat}
    (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned) (hl : w.mem.load slot 8 = some s)
    (K : Kept w w' s L L') (hT : Tab w' s T' L') (hI : Idx w' L' (keysOf T')) : HtAt w' slot s T' :=
  ⟨r, off, L', hd, hr, K.keeps.load slot 8 s (by decide) ⟨load_inW hl, clear_out hd hr⟩ hl, hT, hI⟩

/-- A write outside the heap that leaves the slot leaves the table at it. -/
theorem HtAt.wrote {w w' : World} {slot s p : UInt64} {T : Ht} {n : Nat} {r : Region} {off : Nat}
    (h : HtAt w slot s T) (W : Wrote w w' p n) (hd : decodeAddr p = some (r, off)) (hr : r ≠ .pinned)
    (hl : w'.mem.load slot 8 = w.mem.load slot 8) : HtAt w' slot s T := by
  obtain ⟨r0, off0, L, hd0, hr0, hl0, hT, hI⟩ := h
  exact ⟨r0, off0, L, hd0, hr0, hl.trans hl0, hT.wrote W (clear_out hd hr),
    hI.wrote hT.bi W (clear_out (s := s) (L := L) hd hr L.ix (L.icap * 8) (by simp [blocks]))⟩

/-- `htInit` leaves in its slot null, or a context holding the empty table. -/
theorem HtAt.of_init {w w' : World} {slot : UInt64} {r : Region} {off : Nat}
    (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned)
    (h : ∃ s, w'.mem.load slot 8 = some s ∧ (s = 0 ∨ ∃ L, Tab w' s {} L ∧ Idx w' L []) ∧
      Outside w w' (Misses slot)) :
    ∃ s, w'.mem.load slot 8 = some s ∧ (s = 0 ∨ HtAt w' slot s {}) := by
  obtain ⟨s, hl, h, -⟩ := h
  exact ⟨s, hl, h.imp id fun ⟨L, hT, hI⟩ => ⟨r, off, L, hd, hr, hl, hT, hI⟩⟩

/-- `set` leaves at the slot the table the model's `set` leaves, or the table
    as it was. -/
theorem HtAt.of_set {w w' : World} {slot s : UInt64} {T : Ht} {L : Lay} {kb vb : List UInt8} {r : Region}
    {off : Nat} (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned) (hl : w.mem.load slot 8 = some s)
    (h : SetQ w s T L kb vb w') :
    HtAt w' slot s (T.set (ByteArray.mk kb.toArray) (ByteArray.mk vb.toArray)) ∨ HtAt w' slot s T := by
  obtain ⟨L', K, ⟨hT, hI⟩ | ⟨hT, hI⟩⟩ := h
  · exact Or.inl (HtAt.kept hd hr hl K hT hI)
  · exact Or.inr (HtAt.kept hd hr hl K hT hI)

/-- `htInsert` leaves at the slot the table the model leaves, or the table as
    it was. -/
theorem HtAt.of_insert {w w' : World} {slot s : UInt64} {T : Ht} {L : Lay} {kb vb : List UInt8} {r : Region}
    {off : Nat} (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned) (hl : w.mem.load slot 8 = some s)
    (hT : Tab w s T L) (hI : Idx w L (keysOf T)) (h : InsertQ w s T L kb vb w') :
    HtAt w' slot s (if T.hasTable then T.set (ByteArray.mk kb.toArray) (ByteArray.mk vb.toArray) else T) ∨
      HtAt w' slot s T := by
  unfold InsertQ at h
  by_cases c : T.hasTable = true
  · rw [if_pos c] at h ⊢
    exact HtAt.of_set hd hr hl h
  · rw [if_neg c] at h
    subst h
    exact Or.inr ⟨r, off, L, hd, hr, hl, hT, hI⟩

/-- `htCreate` leaves at the slot the table with one more handle. -/
theorem HtAt.of_create {w w' : World} {slot s : UInt64} {T : Ht} {L : Lay} {r : Region} {off : Nat}
    (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned) (hl : w.mem.load slot 8 = some s)
    (h : Tab w' s { T with handles := T.handles ++ [T.handles.length] } L ∧ Idx w' L (keysOf T) ∧ Kept w w' s L L) :
    HtAt w' slot s { T with handles := T.handles ++ [T.handles.length] } :=
  HtAt.kept hd hr hl h.2.2 h.1 h.2.1

/-- A write clear of the table that misses the slot leaves the table at it. -/
theorem HtAt.wroteClear {w w' : World} {slot s p : UInt64} {T : Ht} {L : Lay} {n : Nat} {r : Region} {off : Nat}
    (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned) (hl : w.mem.load slot 8 = some s)
    (hT : Tab w s T L) (hI : Idx w L (keysOf T)) (W : Wrote w w' p n) (hc : Clear s L p n)
    (hm : Misses slot p n) : HtAt w' slot s T :=
  ⟨r, off, L, hd, hr, (W.load hd fun hn r' off' hd' => hm hn r off r' off' hd hd').trans hl, hT.wrote W hc,
    hI.wrote hT.bi W (hc L.ix (L.icap * 8) (by simp [blocks]))⟩

/-- `htIncrement` leaves at the slot the table the model leaves, or the table
    as it was. -/
theorem HtAt.of_increment {w w' : World} {slot s x addend : UInt64} {T : Ht} {L : Lay} {kb : List UInt8}
    {r : Region} {off : Nat} (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned)
    (hl : w.mem.load slot 8 = some s) (hT : Tab w s T L) (hI : Idx w L (keysOf T))
    (h : IncQ w s T L kb addend x w') :
    HtAt w' slot s (if T.hasTable then
        match T.get (ByteArray.mk kb.toArray) with
        | none => T.set (ByteArray.mk kb.toArray) (ByteArray.mk (le8 addend).toArray)
        | some ex => T.set (ByteArray.mk kb.toArray) (bumped ex addend)
      else T) ∨ HtAt w' slot s T := by
  unfold IncQ at h
  by_cases c : T.hasTable = true
  · rw [if_pos c] at h ⊢
    cases hg : T.get (ByteArray.mk kb.toArray) with
    | none =>
      rw [hg] at h
      exact HtAt.of_set hd hr hl h.2
    | some ex =>
      rw [hg] at h
      obtain ⟨-, L', K, hT', hI'⟩ := h
      exact Or.inl (HtAt.kept hd hr hl K hT' hI')
  · rw [if_neg c] at h
    obtain ⟨-, rfl⟩ := h
    exact Or.inr ⟨r, off, L, hd, hr, hl, hT, hI⟩

/-- `htLookup`, its value copied clear of the table and missing the slot,
    leaves the table at the slot. -/
theorem HtAt.of_lookup {w w' : World} {slot s out x : UInt64} {T : Ht} {L : Lay} {kb : List UInt8}
    {r : Region} {off : Nat} (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned)
    (hl : w.mem.load slot 8 = some s) (hT : Tab w s T L) (hI : Idx w L (keysOf T))
    (hc : ∀ p (hp : p < T.entries.length), (T.entries[p]'hp).1.data.toList = kb →
      Clear s L out (T.entries[p]'hp).2.size ∧ Misses slot out (T.entries[p]'hp).2.size)
    (h : LookQ w s T L kb out x w') : HtAt w' slot s T := by
  have same : HtAt w slot s T := ⟨r, off, L, hd, hr, hl, hT, hI⟩
  unfold LookQ at h
  by_cases c : T.hasTable = true
  · rw [if_pos c] at h
    rcases h with ⟨-, -, rfl⟩ | ⟨p, hp, hk, -, -, -, -, W⟩
    · exact same
    · exact HtAt.wroteClear hd hr hl hT hI W (hc p hp hk).1 (hc p hp hk).2
  · rw [if_neg c] at h
    obtain ⟨-, rfl⟩ := h
    exact same

/-- `htGetEntry`, its key and value copied clear of the table and missing the
    slot, leaves the table at the slot. -/
theorem HtAt.of_getEntry {w w' : World} {slot s index kout vout x : UInt64} {T : Ht} {L : Lay}
    {r : Region} {off : Nat} (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned)
    (hl : w.mem.load slot 8 = some s) (hT : Tab w s T L) (hI : Idx w L (keysOf T))
    (hc : ∀ k v, T.entries[index.toNat]? = some (k, v) →
      Clear s L kout k.size ∧ Clear s L vout v.size ∧ Misses slot kout k.size ∧ Misses slot vout v.size)
    (h : GetQ w s T L index kout vout x w') : HtAt w' slot s T := by
  have same : HtAt w slot s T := ⟨r, off, L, hd, hr, hl, hT, hI⟩
  unfold GetQ at h
  by_cases c : T.hasTable = true
  · rw [if_pos c] at h
    cases hg : T.entries[index.toNat]? with
    | none =>
      rw [hg] at h
      obtain ⟨-, rfl⟩ := h
      exact same
    | some kv =>
      obtain ⟨k, v⟩ := kv
      rw [hg] at h
      obtain ⟨-, -, -, w1, W1, W2⟩ := h
      obtain ⟨ck, cv, mk, mv⟩ := hc k v hg
      have hl1 : w1.mem.load slot 8 = some s :=
        (W1.load hd fun hn r' off' hd' => mk hn r off r' off' hd hd').trans hl
      exact HtAt.wroteClear hd hr hl1 (hT.wrote W1 ck)
        (hI.wrote hT.bi W1 (ck L.ix (L.icap * 8) (by simp [blocks]))) W2 cv mv
  · rw [if_neg c] at h
    obtain ⟨-, rfl⟩ := h
    exact same

/-- The context at the slot holds the table, for `htCount`. -/
theorem HtAt.rep {w : World} {slot s : UInt64} {T : Ht} (h : HtAt w slot s T) : Rep w s T := by
  obtain ⟨_, _, _, _, _, _, hT, _⟩ := h
  exact hT.rep

/-- The context at the slot is one `htCleanup` frees. -/
theorem HtAt.owned {w : World} {slot s : UInt64} {T : Ht} (h : HtAt w slot s T) : ∃ e i a, Owned w s e i a := by
  obtain ⟨_, _, _, _, _, _, hT, _⟩ := h
  exact ⟨_, _, _, hT.owned⟩

/-- What a change to the table keeps, a program keeps of its own memory. -/
theorem Kept.outside {w w' : World} {s : UInt64} {L L' : Lay} (K : Kept w w' s L L') (hc : Calm w') :
    Outside w w' fun _ _ => True :=
  ⟨K.keeps.mono fun _ _ hp => ⟨hp.1, fun _ _ _ r off _ hq _ => Or.inl (hp.2.1 r off hq)⟩, fun _ => hc, K.size⟩

theorem SetQ.outside {w w' : World} {s : UInt64} {T : Ht} {L : Lay} {kb vb : List UInt8}
    (h : SetQ w s T L kb vb w') : Outside w w' fun _ _ => True := by
  obtain ⟨L', K, ⟨hT, -⟩ | ⟨hT, -⟩⟩ := h <;> exact K.outside hT.rep.calm

theorem InsertQ.outside {w w' : World} {s : UInt64} {T : Ht} {L : Lay} {kb vb : List UInt8}
    (h : InsertQ w s T L kb vb w') : Outside w w' fun _ _ => True := by
  unfold InsertQ at h
  by_cases c : T.hasTable = true
  · rw [if_pos c] at h
    exact SetQ.outside h
  · rw [if_neg c] at h
    subst h
    exact Outside.refl _ _

theorem IncQ.outside {w w' : World} {s x addend : UInt64} {T : Ht} {L : Lay} {kb : List UInt8}
    (h : IncQ w s T L kb addend x w') : Outside w w' fun _ _ => True := by
  unfold IncQ at h
  by_cases c : T.hasTable = true
  · rw [if_pos c] at h
    cases hg : T.get (ByteArray.mk kb.toArray) with
    | none => rw [hg] at h; exact SetQ.outside h.2
    | some ex =>
      rw [hg] at h
      obtain ⟨-, L', K, hT', -⟩ := h
      exact K.outside hT'.rep.calm
  · rw [if_neg c] at h
    obtain ⟨-, rfl⟩ := h
    exact Outside.refl _ _

theorem _root_.AlgorithmLib.LibSpec.Out.clear {p : UInt64} (h : Out p) (s : UInt64) (L : Lay) (n : Nat) :
    Clear s L p n :=
  fun _ _ _ r off _ hp _ => Or.inl (h r off hp)

/-- Noting a call leaves the table at the slot. -/
theorem HtAt.obs {w : World} {slot s : UInt64} {T : Ht} (h : HtAt w slot s T) (c : Callee) (vs : List V) :
    HtAt (obsCall w c vs) slot s T := by
  obtain ⟨r, off, L, hd, hr, hl, hT, hI⟩ := h
  exact HtAt.wrote ⟨r, off, L, hd, hr, hl, hT, hI⟩ (wrote_obs w c vs slot) hd hr rfl

/-- The model's answer and table for `htIncrement`. -/
def incOf (T : Ht) (kb : List UInt8) (addend : UInt64) : UInt64 × Ht :=
  if T.hasTable then
    match T.get (ByteArray.mk kb.toArray) with
    | none => (addend, T.set (ByteArray.mk kb.toArray) (ByteArray.mk (le8 addend).toArray))
    | some ex => (ofLe64 ex + addend, T.set (ByteArray.mk kb.toArray) (bumped ex addend))
  else (addend, T)

section at_slot
variable {fns : Nat → Option Fn} {env : FnEnv} {steps i : Nat}

/-- **A linked program's calls to the table's functions, the table at its
    slot**: each leaves at the slot the table the entry point's model leaves,
    or, where an allocation failed, the table as it was, and outside the heap
    changes nothing but the slot or the runs it copies to. -/
theorem create_at (hf : fns i = some (implFn [.i64] (.status create))) {w : World} {slot s : UInt64} {T : Ht}
    (h : HtAt w slot s T) (hn : T.handles.length + 1 < 2 ^ 64) (k : Nat) :
    CallsA (termLocals fns env steps k) i [.sc .i64 s] w (fun x w' => x = UInt64.ofNat T.handles.length ∧
      HtAt w' slot s { T with handles := T.handles ++ [T.handles.length] } ∧ Outside w w' fun _ _ => True) := by
  obtain ⟨r, off, L, hd, hr, hl, hT, hI⟩ := h
  exact (Ans.calls hf (fun cfg => create_tab cfg w s T L hT hI hn) k).mono
    fun x w' ⟨hx, hq⟩ => ⟨hx, HtAt.of_create hd hr hl hq, hq.2.2.outside hq.1.rep.calm⟩

theorem count_at (hf : fns i = some (implFn [.i64] (.status count))) {w : World} {slot s : UInt64} {T : Ht}
    (h : HtAt w slot s T) (k : Nat) :
    CallsA (termLocals fns env steps k) i [.sc .i64 s] w (fun x w' =>
      x = (if T.hasTable then UInt64.ofNat T.entries.length else 0) ∧ w' = w) :=
  count_calls hf w s T h.rep k

theorem insert_at (hf : fns i = some (implFn [.i64, .i64, .i32, .i64, .i32] (.void LibHt.insert))) {w : World}
    {slot s : UInt64} {T : Ht} (h : HtAt w slot s T) (key klen val vlen : UInt64)
    (kb vb : List UInt8) (hkb : Bytes w key kb) (hvb : Bytes w val vb) (hkl : klen = UInt64.ofNat kb.length)
    (hk32 : kb.length < 2 ^ 32) (hvl : vlen = UInt64.ofNat vb.length) (hv32 : vb.length < 2 ^ 32)
    (ok : Out key) (ov : Out val) (k : Nat) :
    CallsD (termLocals fns env steps k) i
      [.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 val, .sc .i32 vlen] w (fun w' =>
        (HtAt w' slot s (if T.hasTable then T.set (ByteArray.mk kb.toArray) (ByteArray.mk vb.toArray) else T) ∨
          HtAt w' slot s T) ∧ Outside w w' fun _ _ => True) := by
  obtain ⟨r, off, L, hd, hr, hl, hT, hI⟩ := h
  exact (insert_calls hf w s T L hT hI key klen val vlen kb vb hkb hvb hkl hk32 hvl hv32
    (fun _ => ok.clear _ _ _) (fun _ => ov.clear _ _ _) k).mono fun w' hq =>
      ⟨HtAt.of_insert hd hr hl hT hI hq, hq.outside⟩

theorem increment_at (hf : fns i = some (implFn [.i64, .i64, .i32, .i64] (.status increment))) {w : World}
    {slot s : UInt64} {T : Ht} (h : HtAt w slot s T) (key klen addend : UInt64)
    (kb : List UInt8) (hkb : Bytes w key kb) (hkl : klen = UInt64.ofNat kb.length) (hk32 : kb.length < 2 ^ 32)
    (ok : Out key) (h8 : ∀ ex, T.get (ByteArray.mk kb.toArray) = some ex → 8 ≤ ex.size) (k : Nat) :
    CallsA (termLocals fns env steps k) i [.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 addend] w
      (fun x w' => x = (incOf T kb addend).1 ∧ (HtAt w' slot s (incOf T kb addend).2 ∨ HtAt w' slot s T) ∧
        Outside w w' fun _ _ => True) := by
  obtain ⟨r, off, L, hd, hr, hl, hT, hI⟩ := h
  refine (increment_calls hf w s T L hT hI key klen addend kb hkb hkl hk32 (fun _ => ok.clear _ _ _) h8 k).mono
    fun x w' hq => ?_
  have hs := HtAt.of_increment hd hr hl hT hI hq
  have hO := hq.outside
  unfold IncQ at hq
  unfold incOf
  by_cases c : T.hasTable = true
  · rw [if_pos c] at hq hs ⊢
    cases hg : T.get (ByteArray.mk kb.toArray) with
    | none => rw [hg] at hq hs; exact ⟨hq.1, hs, hO⟩
    | some ex => rw [hg] at hq hs; exact ⟨hq.1, hs, hO⟩
  · rw [if_neg c] at hq hs ⊢
    exact ⟨hq.1, hs, hO⟩

theorem lookup_at (hf : fns i = some (implFn [.i64, .i64, .i32, .i64] (.status lookup))) {w : World}
    {slot s : UInt64} {T : Ht} (h : HtAt w slot s T) (key klen out : UInt64)
    (kb : List UInt8) (hB : Bytes w key kb) (hkl : klen = UInt64.ofNat kb.length) (hk32 : kb.length < 2 ^ 32)
    (oo : Out out)
    (hroom : ∀ p (hp : p < T.entries.length), (T.entries[p]'hp).1.data.toList = kb →
      Room w out (T.entries[p]'hp).2.size ∧ Misses slot out (T.entries[p]'hp).2.size) (k : Nat) :
    CallsA (termLocals fns env steps k) i [.sc .i64 s, .sc .i64 key, .sc .i32 klen, .sc .i64 out] w
      (fun x w' => HtAt w' slot s T ∧ ∃ L, LookQ w s T L kb out x w') := by
  obtain ⟨r, off, L, hd, hr, hl, hT, hI⟩ := h
  exact (lookup_calls hf w s T L hT hI key klen out kb hB hkl hk32
    (fun p hp hk => ⟨(hroom p hp hk).1, oo.clear _ _ _⟩) k).mono fun x w' hq =>
      ⟨HtAt.of_lookup hd hr hl hT hI (fun p hp hk => ⟨oo.clear _ _ _, (hroom p hp hk).2⟩) hq, L, hq⟩

theorem getEntry_at (hf : fns i = some (implFn [.i64, .i32, .i64, .i64] (.status getEntry))) {w : World}
    {slot s : UInt64} {T : Ht} (h : HtAt w slot s T) (index kout vout : UInt64)
    (hix : index.toNat < 2 ^ 32) (ok : Out kout) (ov : Out vout)
    (hroom : ∀ k v, T.entries[index.toNat]? = some (k, v) →
      Room w kout k.size ∧ Room w vout v.size ∧ Misses slot kout k.size ∧ Misses slot vout v.size) (k : Nat) :
    CallsA (termLocals fns env steps k) i [.sc .i64 s, .sc .i32 index, .sc .i64 kout, .sc .i64 vout] w
      (fun x w' => HtAt w' slot s T ∧ ∃ L, GetQ w s T L index kout vout x w') := by
  obtain ⟨r, off, L, hd, hr, hl, hT, hI⟩ := h
  exact (getEntry_calls hf w s T L hT index kout vout hix
    (fun k v hg => ⟨(hroom k v hg).1, (hroom k v hg).2.1, ok.clear _ _ _, ov.clear _ _ _⟩) k).mono fun x w' hq =>
      ⟨HtAt.of_getEntry hd hr hl hT hI
        (fun k v hg => ⟨ok.clear _ _ _, ov.clear _ _ _, (hroom k v hg).2.2.1, (hroom k v hg).2.2.2⟩) hq, L, hq⟩

theorem init_at (hf : fns i = some (implFn [.i64] (.void init))) (w : World) (slot : UInt64) (r : Region)
    (off : Nat) (hc : Calm w) (hd : decodeAddr slot = some (r, off)) (hr : r ≠ .pinned)
    (hroom : off + 8 ≤ (w.mem.region r).size) (k : Nat) :
    CallsD (termLocals fns env steps k) i [.sc .i64 slot] w (fun w' =>
      (∃ s, w'.mem.load slot 8 = some s ∧ (s = 0 ∨ HtAt w' slot s {})) ∧ Outside w w' (Misses slot)) :=
  (init_calls hf w slot r off hc hd hr hroom k).mono fun _ hq => ⟨HtAt.of_init hd hr hq, hq.choose_spec.2.2⟩

theorem cleanup_at (hf : fns i = some (implFn [.i64] (.void cleanup))) {w : World} {slot s : UInt64} {T : Ht}
    (h : HtAt w slot s T) (k : Nat) :
    CallsD (termLocals fns env steps k) i [.sc .i64 slot] w (fun w' =>
      w'.mem.load slot 8 = some 0 ∧ Outside w w' (Misses slot)) := by
  obtain ⟨e, i', a, ho⟩ := h.owned
  obtain ⟨r, off, L, hd, hr, hl, hT, hI⟩ := h
  exact cleanup_calls hf w slot s r off hT.rep.calm hd hr hl (Or.inr ⟨e, i', a, ho⟩) k

end at_slot

end HtSpec

#print axioms HtSpec.init_spec
#print axioms HtSpec.cleanup_spec
#print axioms HtSpec.create_null
#print axioms HtSpec.create_spec
#print axioms HtSpec.count_null
#print axioms HtSpec.count_spec
#print axioms HtSpec.getEntry_spec
#print axioms HtSpec.wp_hash
#print axioms HtSpec.wp_bytesEq
#print axioms HtSpec.wp_find
#print axioms HtSpec.lookup_spec
#print axioms HtSpec.wp_append
#print axioms HtSpec.wp_grow
#print axioms HtSpec.wp_insertAt
#print axioms HtSpec.wp_set
#print axioms HtSpec.insert_spec
#print axioms HtSpec.increment_spec
#print axioms HtSpec.insert_calls
#print axioms HtSpec.create_tab
#print axioms HtSpec.HtAt.of_insert
#print axioms HtSpec.HtAt.of_getEntry
#print axioms HtSpec.getEntry_at
#print axioms HtSpec.increment_at
