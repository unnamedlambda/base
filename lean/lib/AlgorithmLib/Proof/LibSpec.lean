module
public import AlgorithmLib.Proof.Loops
meta import AlgorithmLib.Proof.Loops
public import AlgorithmLib.Proof.ProgHoare
meta import AlgorithmLib.Proof.ProgHoare
public import AlgorithmLib.Host.Heap
meta import AlgorithmLib.Host.Heap
public import AlgorithmLib.Surface.LibBase
meta import AlgorithmLib.Surface.LibBase
public import AlgorithmLib.Host.Summary
meta import AlgorithmLib.Host.Summary
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Proof.LibSpec` — what a library function does

A program's calls to the engine's entry points are linked to functions written
in CLIF over the C heap (`Surface.Lib*`). What such a function does is stated
as `Ans`: run on its arguments from a world, its body neither misuses a call
nor faults, and where it ends it answers a value and leaves a world the
statement relates to the ones it started from. One statement is the function's
safety and its contract.

A statement is proven at the generator, by its weakest precondition
(`Proof.ProgHoare`), one step at a time:

* `wp_iconst64`, `wp_iadd`, `wp_iaddImm`, `wp_at`, `wp_load64`, `wp_storeI64`,
  `wp_setFld`, `wp_calloc`, `wp_free`, `wp_ifte` and `wp_when`, and the
  arithmetic ones (`wp_isub`, `wp_imul`, `wp_band`, `wp_bor`, `wp_bxor`,
  `wp_ishlImm`, `wp_ushrImm`, `wp_umax`, `wp_icmp`, `wp_select`, …) each say what one construct needs and leaves, with the slots
  it reads named by hypotheses `Γ[a]? = some v`;
* `slot` proves such a hypothesis for an environment built by pushes and
  joins, and `ext_env` that such an environment holds what an earlier one
  held (`Ext`);
* `wp_forLoop` and `wp_forLoopAcc` are a counted loop, by an invariant of the
  trips made, the accumulator and the world;
* `wp_wloop1` is a loop of one carry left where its head says, by an
  invariant of the carry and the world, and `wp_ifteV` a branch whose arms
  answer one value, with what follows it;
* `wp_seq` sequences, `join_ans` closes the join of a branch whose arms carry
  the answer, and `ans_of_wp` turns the weakest precondition at the entry
  into the statement.

A function that answers nothing is stated as `Does` (`does_of_wp`), and
either statement gives that the run neither misuses nor faults (`Ans.sound`,
`Does.sound`).

The heap is read through `fld`, the eight bytes at an offset in a block: a
field written reads back (`fld_stored`) and leaves the block's other fields
(`fld_kept`), from `load_store8` and `load_store_clear`, which hold of every
region. `wp_calloc` is an allocation: null, or a fresh block of zeros past
all the pinned memory there was (`Fresh`) in a world with more heap, where
every block, load, run of bytes and room is as it was (`Grows`); `Stays` is what both more heap and a store leave. `wp_free` frees
null or a block, and leaves every other block with its fields (`Freed`).
`Own` is null or a block, which is what a field that points at an array
holds; what `calloc` answers is one (`fresh_own`), apart from every block
there was (`fresh_ne`, `fresh_apart`) and starting past each (`Before`,
`Fresh.before`).

A run of bytes that can be read is `Bytes`: a load inside it reads them
(`Bytes.load`). `wp_memcpy` copies one into room (`Room`, which a run inside
a block has: `Blk.room`), apart from it, and leaves a world where only those bytes are rewritten (`Wrote`):
loads and runs clear of them are what they were (`Wrote.load`,
`Wrote.bytes`; `Wrote.fld_blk` and `Wrote.bytes_blk` for a block's), and a
store is such a write (`wrote_stored`).

Two blocks are `Apart` where one ends before the other starts; a store or a
copy into one leaves the fields and runs of the other (`fld_other`,
`bytes_other`, `Apart.clear`).

`Whole` is a heap block of exactly its size and `Off` a run off a block.
`Keeps w w' P` says `w'` has the runs, room and loads of `w` that `P`
protects; more heap keeps everything (`Grows.keeps`), a write what is clear of
it (`Wrote.keeps`, with `Off.wrote`, `Fresh.wrote` and `Off.sub` for a write
inside a block, a fresh block or some fields), and a free what is off the
whole block freed (`Freed.keeps`); steps that protect different things compose
through what is still inside memory (`Keeps.next`).

`moved` is a block regrown: more heap with a fresh block, a write inside it,
the old block freed, and the two fields that name it and its size set. What
that leaves is `Moved`: every run off the old block and off those fields
kept, every other block there. `Past w x` says `x` starts past the memory `w`
had, so that everything inside that memory is off it (`Past.off`).

`le8 x` is the eight bytes of a word, which make the word (`leN_le8`) and are
the model's `le64` (`le64_eq`); an eight-byte store over bytes that can be
read leaves them, then the rest (`stored_bytes`).

A library function is one of a linked program's functions (`implFn`), and a
call to it there is as its statement says at every call depth (`CallsA`,
`CallsD`; `Ans.calls`, `Does.calls`). A program that makes such a call goes
on from each answer and world the statement allows (`wp_callA`, `wp_callD`),
the call noted first, which writes nothing (`wrote_obs`).

A call to a library outside the program that answers a result code goes on
from each answer the model gives (`wp_extRc`); `wp_status` turns the code into
a library function's `0` or `-1`, and `wp_ifte32`, `wp_failIf32` branch on
32-bit slots.

`Outside w w' P` is what a program keeps of its own memory over a call into a
library: `w'` is `w` outside the heap, where `P` holds. More heap, a free and
a write into the heap leave everything outside it (`Grows.outside`,
`Freed.outside`, `Wrote.outside`); a slot written leaves what misses it
(`Wrote.outsideBut`, `Misses`).

Up to eight bytes are told apart by the little-endian word they make
(`leN_eq_iff`), which is what comparing two runs a word at a time needs.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace AlgorithmLib.LibSpec

-- ---------------------------------------------------------------------------
-- Memory
-- ---------------------------------------------------------------------------

theorem readable_of_reachable {m : Mem} {r : Region} {off n : Nat} (h : m.reachable r off n = true) :
    m.readable r off n = true := by
  simp only [Mem.reachable, Mem.readable, Bool.and_eq_true, Bool.or_eq_true, List.all_eq_true] at h ⊢
  refine ⟨h.1, h.2.imp id fun ⟨h1, h2⟩ => ⟨h1, fun b hb => ?_⟩⟩
  simp [h2 b hb]

theorem readable_setRegion (m : Mem) (r r' : Region) (b : ByteArray) (off n : Nat) :
    (m.setRegion r b).readable r' off n = m.readable r' off n := by
  cases r <;> rfl

/-- **An eight-byte store reads back.** -/
theorem load_store8 {m m' : Mem} {a v : UInt64} (h : m.store a 8 v = some m') : m'.load a 8 = some v := by
  have hsz := fun r => store_sizes h r
  unfold Mem.store at h
  cases hd : decodeAddr a with
  | none => simp [hd] at h
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [hd, Option.bind_eq_bind, Option.bind_some] at h
    split at h
    · cases h
    · rename_i hc
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, Bool.not_eq_true', Bool.not_eq_false] at hc
      injection h with h
      subst h
      rw [load_of hd (by rw [hsz]; omega)
        (by rw [readable_setRegion]; exact readable_of_reachable (by simpa using hc.2))]
      congr 1
      rw [Sem.region_setRegion_self]
      unfold bytesAt
      refine (foldr_fun_congr _ (fun i => (byteOfV v i).toUInt64) _ _ fun j hj => ?_).trans (reassemble8 v)
      exact congrArg UInt8.toUInt64 (foldl_set!_at off v 8 _ j (List.mem_range.mp hj) (by omega))

/-- **A store leaves the bytes it does not cover**: a load of bytes in another
    region, or clear of the stored ones, reads what it read. -/
theorem load_store_clear {m m' : Mem} {a b : UInt64} {n k : Nat} {v : UInt64} {r r' : Region} {off off' : Nat}
    (h : m.store a n v = some m') (hd : decodeAddr a = some (r, off)) (hd' : decodeAddr b = some (r', off'))
    (hc : r' ≠ r ∨ off' + k ≤ off ∨ off + n ≤ off') : m'.load b k = m.load b k := by
  obtain ⟨r0, off0, hd0, hF⟩ := store_frame h
  rw [hd] at hd0
  cases hd0
  unfold Mem.load
  simp only [hd', Option.bind_eq_bind, Option.bind_some, ← hF.same.size r', ← hF.same.readable r' off' k]
  split
  · rfl
  · congr 1
    by_cases e : r' = r
    · subst e
      exact foldr_bytes_congr _ _ fun i hi => hF.outside _ (by
        have := List.mem_range.mp hi
        rcases hc with hc | hc | hc
        · exact absurd rfl hc
        · omega
        · omega)
    · rw [hF.other r' e]

theorem Blk.decode {w : World} {s : UInt64} {n : Nat} (h : Blk w s n) :
    ∃ off, decodeAddr s = some (.pinned, off) ∧ ∀ j, j < n → decodeAddr (s + UInt64.ofNat j) = some (.pinned, off + j) := by
  obtain ⟨off, m, hat, hn, hm, hsz, h36, hlv⟩ := h
  obtain ⟨hd, -⟩ := heapAt_decode hat
  exact ⟨off, hd, fun j hj => Sem.decodeAddr_add hd (by rw [Sem.regionSpan_toNat]; omega)⟩

theorem Blk.ne_zero {w : World} {s : UInt64} {n : Nat} (h : Blk w s n) : s ≠ 0 := by
  obtain ⟨off, hd, -⟩ := Blk.decode h
  rintro rfl
  rw [show decodeAddr 0 = none from by decide] at hd
  cases hd

/-- The world a store leaves. -/
abbrev stored (w : World) (p : UInt64) (x : UInt64) (m : Mem) : World := { obsStore w p 8 x with mem := m }

theorem Calm.stored {w : World} {p x : UInt64} {m : Mem} (h : Calm w) (hs : w.mem.store p 8 x = some m) :
    Calm (stored w p x m) := by
  obtain ⟨_, h2, h3⟩ := Mem.store_meta hs
  exact ⟨h3.trans h.1, h2.trans h.2⟩

theorem Blk.keep {w : World} {s p x : UInt64} {n : Nat} {m : Mem} (h : Blk w s n)
    (hs : w.mem.store p 8 x = some m) : Blk (stored w p x m) s n :=
  Blk.stored h (store_sizes_eq hs) (Mem.store_meta hs).1

/-- A field of a block can be written. -/
theorem Blk.fld_store {w : World} {s : UInt64} {n j : Nat} (h : Blk w s n) (hc : Calm w) (hj : j + 8 ≤ n)
    (x : UInt64) : ∃ m, w.mem.store (s + UInt64.ofNat j) 8 x = some m :=
  Option.isSome_iff_exists.mp (Blk.store h hj (by decide) hc.1 hc.2 x)

/-- A field written reads back. -/
theorem fld_stored {w : World} {s x : UInt64} {j : Nat} {m : Mem}
    (hs : w.mem.store (s + UInt64.ofNat j) 8 x = some m) : fld (stored w (s + UInt64.ofNat j) x m) s j = some x :=
  load_store8 hs

/-- A field written leaves the block's other fields. -/
theorem fld_kept {w : World} {s x : UInt64} {n i j : Nat} {m : Mem} (h : Blk w s n) (hi : i + 8 ≤ n) (hj : j + 8 ≤ n)
    (hij : i + 8 ≤ j ∨ j + 8 ≤ i) (hs : w.mem.store (s + UInt64.ofNat j) 8 x = some m) :
    fld (stored w (s + UInt64.ofNat j) x m) s i = fld w s i := by
  obtain ⟨off, -, hd⟩ := Blk.decode h
  exact load_store_clear hs (hd j (by omega)) (hd i (by omega)) (Or.inr (by omega))

-- ---------------------------------------------------------------------------
-- Runs of bytes
-- ---------------------------------------------------------------------------

theorem readable_sub {m : Mem} {r : Region} {off n off' k : Nat} (h : m.readable r off n = true)
    (h1 : off ≤ off') (h2 : off' + k ≤ off + n) : m.readable r off' k = true := by
  simp only [Mem.readable, Bool.and_eq_true, Bool.or_eq_true, List.any_eq_true, List.all_eq_true,
    decide_eq_true_eq, Busy.clear, Bool.not_eq_true'] at h ⊢
  refine ⟨h.1, h.2.imp id fun ⟨⟨a, ha, hc⟩, hb⟩ => ⟨⟨a, ha, by omega⟩, fun b hb' => ?_⟩⟩
  rcases hb b hb' with h | h | h
  · exact Or.inl h
  · exact Or.inr (Or.inl (by omega))
  · exact Or.inr (Or.inr (by omega))

theorem reachable_sub {m : Mem} {r : Region} {off n off' k : Nat} (h : m.reachable r off n = true)
    (h1 : off ≤ off') (h2 : off' + k ≤ off + n) : m.reachable r off' k = true := by
  simp only [Mem.reachable, Bool.and_eq_true, Bool.or_eq_true, List.any_eq_true, List.all_eq_true,
    decide_eq_true_eq, Busy.clear] at h ⊢
  refine ⟨h.1, h.2.imp id fun ⟨⟨a, ha, hc⟩, hb⟩ => ⟨⟨a, ha, by omega⟩, fun b hb' => ?_⟩⟩
  rcases hb b hb' with h | h
  · exact Or.inl (by omega)
  · exact Or.inr (by omega)

/-- A load inside a run of bytes reads them. -/
theorem Bytes.load {w : World} {p : UInt64} {bs : List UInt8} (h : Bytes w p bs) {j k : Nat}
    (hjk : j + k ≤ bs.length) (hk : 0 < k) : w.mem.load (p + UInt64.ofNat j) k = some (leN bs j k) := by
  rcases h with rfl | ⟨r, off, hd, hs, h36, hr, hb⟩
  · simp only [List.length_nil] at hjk; omega
  have hd' := Sem.decodeAddr_add (i := j) hd (by rw [Sem.regionSpan_toNat]; omega)
  rw [load_of hd' (by omega) (readable_sub hr (by omega) (by omega))]
  congr 1
  unfold bytesAt leN
  exact foldr_fun_congr _ _ _ _ fun i hi => by
    have := List.mem_range.mp hi
    rw [Nat.add_assoc, hb (j + i) (by omega)]

/-- **There is room for `n` bytes at `p`**: none, wherever `p` is, or a run
    inside one region that can be reached. -/
def Room (w : World) (p : UInt64) (n : Nat) : Prop :=
  n = 0 ∨ ∃ r off, decodeAddr p = some (r, off) ∧ off + n ≤ (w.mem.region r).size ∧ off + n ≤ 2 ^ 36 ∧
    w.mem.reachable r off n = true

-- ---------------------------------------------------------------------------
-- Slots
-- ---------------------------------------------------------------------------

theorem get_push {Γ : Env} {i : Nat} {v x : V} (h : Γ[i]? = some x) : (Γ.push v)[i]? = some x := by
  have hi : i < Γ.size := by
    rcases Nat.lt_or_ge i Γ.size with hi | hi
    · exact hi
    · rw [Array.getElem?_eq_none hi] at h; cases h
  rw [Array.getElem?_push_lt hi, ← h, Array.getElem?_eq_getElem hi]

theorem get_push_size {Γ : Env} {v : V} : (Γ.push v)[Γ.size]? = some v := by simp

theorem get_bindAt {Γ : Env} {n i : Nat} {vs : List V} {x : V} (hn : Γ.size ≤ n) (h : Γ[i]? = some x) :
    (bindAt Γ n vs)[i]? = some x := by
  have hi : i < Γ.size := by
    rcases Nat.lt_or_ge i Γ.size with hi | hi
    · exact hi
    · rw [Array.getElem?_eq_none hi] at h; cases h
  rw [bindAt_lt (by omega) hi, ← h]

/-- `Γ'` holds every slot `Γ` holds: what a body's later environments are to
    the one a step was entered with. -/
def Ext (Γ Γ' : Env) : Prop := ∀ (i : Nat) (v : V), Γ[i]? = some v → Γ'[i]? = some v

theorem Ext.refl (Γ : Env) : Ext Γ Γ := fun _ _ h => h

theorem Ext.get {Γ Γ' : Env} (h : Ext Γ Γ') {i : Nat} {v : V} (hi : Γ[i]? = some v) : Γ'[i]? = some v := h i v hi

theorem Ext.push {Γ Γ' : Env} (h : Ext Γ Γ') (v : V) : Ext Γ (Γ'.push v) := fun i x hi => get_push (h i x hi)

theorem Ext.bindAt {Γ Γ' : Env} (h : Ext Γ Γ') {n : Nat} (hn : Γ'.size ≤ n) (vs : List V) :
    Ext Γ (bindAt Γ' n vs) := fun i x hi => get_bindAt hn (h i x hi)

syntax "slot_walk" : tactic
macro_rules
  | `(tactic| slot_walk) => `(tactic| first
      | with_reducible assumption
      | with_reducible exact get_push_size
      | (with_reducible apply get_push; slot_walk)
      | (with_reducible refine get_bindAt (by with_reducible assumption) ?_; slot_walk)
      | (with_reducible refine Ext.get (by with_reducible assumption) ?_; slot_walk))

/-- An environment built by pushes and joins holds what the one it was built
    on holds. -/
syntax "ext_env" : tactic
macro_rules
  | `(tactic| ext_env) => `(tactic| first
      | with_reducible assumption
      | with_reducible exact Ext.refl _
      | (with_reducible refine Ext.push ?_ _; ext_env)
      | (with_reducible refine Ext.bindAt ?_ (by with_reducible assumption) _; ext_env))

/-- Read a slot of an environment built by pushes and joins: the slot's name
    is the size of the environment it was pushed onto, so the walk compares
    names as written. -/
macro "slot" : tactic => `(tactic| ((try simp only [Vals.head, Vals.tail, Vals.snd, Vals.thd, Vals.fth,
  Vals.fif, Nat.zero_add, Nat.reduceAdd]); slot_walk))

-- ---------------------------------------------------------------------------
-- Values
-- ---------------------------------------------------------------------------

theorem norm64 (z : UInt64) : norm .i64 z = .sc .i64 z := by
  simp only [norm]
  congr 1
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, show (widthMask .i64).toNat = 2 ^ 64 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
    Nat.mod_eq_of_lt z.toNat_lt]

theorem ofInt64 (k : Int) : ofInt .i64 k = .sc .i64 (c64 k) := norm64 _

theorem c64_nat {k : Nat} (hk : k < 2 ^ 64) : c64 (k : Int) = UInt64.ofNat k := by
  unfold c64
  congr 1
  show ((k : Int) % ((18446744073709551616 : Nat) : Int)).toNat = k
  omega

-- ---------------------------------------------------------------------------
-- One step of a body
-- ---------------------------------------------------------------------------

section steps
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

theorem wp_iconst64 {k : Int} {Q : Slot .i64 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (.sc .i64 (c64 k))) w) : wp cfg d J (iconst64 k) Q Γ w := by
  simp only [iconst64, iconst, op, wp_op, wp_ret, Op'.erase, evalOp, ofInt64, Option.some.injEq, forall_eq',
    reduceCtorEq, false_implies, true_and]
  exact h

theorem wp_iadd {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x + y))) w) : wp cfg d J (iadd a b) Q Γ w := by
  simp [iadd, op, wp_op, wp_ret, Op'.erase, evalOp, bin, Sem.get, ha, hb, norm64]
  exact ⟨fun h => absurd h (by decide), fun _ => h⟩

theorem wp_iaddImm {a : Slot .i64} {x : UInt64} {imm : Int} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q (Γ.push (.sc .i64 (c64 imm))).size ((Γ.push (.sc .i64 (c64 imm))).push (.sc .i64 (x + c64 imm))) w) :
    wp cfg d J (iaddImm a imm) Q Γ w := by
  unfold iaddImm
  rw [wp_bind]
  exact wp_iconst64 (wp_iadd (get_push ha) get_push_size h)

/-- The address of the field at `off`. -/
theorem wp_at {a : Slot .i64} {x : UInt64} {off : Nat} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q (Γ.push (.sc .i64 (UInt64.ofNat off))).size
      ((Γ.push (.sc .i64 (UInt64.ofNat off))).push (.sc .i64 (x + UInt64.ofNat off))) w)
    (hoff : off < 2 ^ 64 := by decide) :
    wp cfg d J (Lib.at_ a off) Q Γ w := by
  unfold Lib.at_
  refine wp_iaddImm ha ?_
  rw [c64_nat hoff]
  exact h

theorem wp_load64 {a : Slot .i64} {p v : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 p)) (hl : w.mem.load p 8 = some v)
    (h : Q Γ.size (Γ.push (.sc .i64 v)) w) : wp cfg d J (load64 a) Q Γ w := by
  simp [load64, load, op, wp_op, wp_ret, Op'.erase, evalOp, Sem.get, ha, hl, tyBytes, ClifTy.lanes, ClifTy.width]
  exact h

theorem wp_storeI64 {v a : Slot .i64} {x p : UInt64} {m : Mem} {Q : Unit → Env → World → Prop}
    (hv : Γ[v]? = some (.sc .i64 x)) (ha : Γ[a]? = some (.sc .i64 p)) (hs : w.mem.store p 8 x = some m)
    (h : Q () Γ (stored w p x m)) : wp cfg d J (storeI64 v a) Q Γ w := by
  simp [storeI64, storeW, wp_store, wp_ret, runStmt, Sem.get, hv, ha, hs, tyBytes, ClifTy.width]
  exact h

/-- A branch on two 64-bit slots, both arms ending in `Qj`. -/
theorem wp_ifte {jTys : List ClifTy} {cc : ICmpCond} {a b : Slot .i64} {x y : UInt64}
    {thn els : Prog Slot Lvl (Vals Slot jTys)} {Q : Vals Slot jTys → Env → World → Prop}
    (Qj : Vals Slot jTys → Env → World → Prop)
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (ht : cmpInt cc .i64 x y = true → wp cfg d J thn Qj (Γ.push (boolV true)) w)
    (he : cmpInt cc .i64 x y = false → ∀ ne, (Γ.push (boolV false)).size ≤ ne →
      wp cfg d J els Qj (bindAt (Γ.push (boolV false)) ne []) w)
    (hk : ∀ nJ Γ' jv vs w', Γ'.size ≤ nJ → jv.slots.mapM (fun r => Γ'[r]?) = some vs → Qj jv Γ' w' →
      Q (carriesFrom nJ jTys) (bindAt Γ' nJ vs) w') :
    wp cfg d J (ifte cc a b thn els) Q Γ w := by
  simp only [ifte, wp_ite, wp_ret]
  have hev : evalOp w.mem Γ (.icmp cc a b) = some (boolV (cmpInt cc .i64 x y)) := by
    simp [evalOp, Sem.get, ha, hb, zipIntCmp]
    rfl
  refine ⟨Qj, Qj, ⟨?_, ?_, ?_⟩, fun nJ Γ' jv vs w' hle hvs hq => hk nJ Γ' jv vs w' hle hvs (hq.elim id id)⟩
  · intro t f hc hf
    rw [hev] at hc
    have hb' : V.sc t f = boolV (cmpInt cc .i64 x y) := (Option.some.inj hc).symm
    cases hcmp : cmpInt cc .i64 x y
    · rw [hcmp] at hb'; simp [boolV] at hb'; obtain ⟨rfl, rfl⟩ := hb'; simp at hf
    · rw [hb', hcmp]; exact ht hcmp
  · intro ne t f hne hc hf
    rw [hev] at hc
    have hb' : V.sc t f = boolV (cmpInt cc .i64 x y) := (Option.some.inj hc).symm
    cases hcmp : cmpInt cc .i64 x y
    · rw [hb', hcmp]; exact he hcmp ne (by simpa using hne)
    · rw [hcmp] at hb'; simp [boolV] at hb'; obtain ⟨rfl, rfl⟩ := hb'; simp at hf
  · rw [hev]; nofun

/-- The one value a join of one 64-bit slot carries. -/
theorem join1 {Γ' : Env} {jv : Vals Slot [.i64]} {vs : List V} {x : V}
    (hvs : jv.slots.mapM (fun r => Γ'[r]?) = some vs) (hx : Γ'[jv.head]? = some x) : vs = [x] := by
  cases jv with
  | cons v t =>
    cases t
    simp only [Vals.slots, Vals.head] at hvs hx
    rw [List.mapM_cons, hx] at hvs
    simp at hvs
    exact hvs.symm

end steps

/-- Sequence the statements of the body at the head of the goal. -/
macro "wp_seq" : tactic => `(tactic| simp only [wp_bind, wp_pure, wp_ret])

-- ---------------------------------------------------------------------------
-- A body that answers
-- ---------------------------------------------------------------------------

/-- The answer slot holds a 64-bit value `R` holds of, with the world left. -/
abbrev AnsQ (R : UInt64 → World → Prop) : Slot .i64 → Env → World → Prop :=
  fun a Γ w' => ∃ x, Γ[a]? = some (.sc .i64 x) ∧ R x w'

/-- `AnsQ`, of the one value a branch joins. -/
abbrev AnsJ (R : UInt64 → World → Prop) : Vals Slot [.i64] → Env → World → Prop :=
  fun jv Γ w' => AnsQ R jv.head Γ w'

/-- **What a library function that answers does**: run on `args` from `w`, its
    body neither misuses a call nor faults, and where it ends it answers a
    64-bit `x` and leaves a world `w'` with `Q x w'`. -/
def Ans (cfg : Cfg) (args : List V) (w : World) (p : StatusBody) (params : List ClifTy)
    (Q : UInt64 → World → Prop) : Prop :=
  Triple cfg (At args.toArray w) (runAns p params).2.1
    { ok := fun Γ w' => ∃ a x, (runAns p params).1 = some a ∧ Γ[a]? = some (.sc .i64 x) ∧ Q x w'
      faultOk := false }

theorem ans_of_wp {cfg : Cfg} {p : StatusBody} {params : List ClifTy} {args : List V} {w : World}
    {Q : UInt64 → World → Prop}
    (hwp : wp cfg 0 { ok := fun _ _ => True, faultOk := false } (fnBody params p) (AnsQ Q) args.toArray w)
    (hf : Fine (emitGo (fnBody params p) ⟨params.length, 0, [], [], [], none⟩).2) (hargs : args.length = params.length) :
    Ans cfg args w p params Q := by
  have h1 := (PT.conseq (P' := fun Γ' w' => Γ' = args.toArray ∧ w' = w) (wp_sound cfg 0 _ (fnBody params p) _)
    (fun Γ' w' h => by obtain ⟨rfl, rfl⟩ := h; exact hwp) (fun _ _ _ h => h))
    ⟨params.length, 0, [], [], [], none⟩ (At args.toArray w) rfl hf
    (nil_rule fun Γ' w' h => ⟨by rw [h.1]; simp [hargs], h⟩)
  unfold Ans
  show Triple cfg _ (emitGo (fnBody params p) ⟨params.length, 0, [], [], [], none⟩).2.flush.pieces.reverse _
  rw [St.out_eq_flush]
  exact Hoare.conseq h1 (fun _ _ h => h) (fun Γ w' ⟨a, ha, _, x, hx, hq⟩ => ⟨a, x, ha, hx, hq⟩)
    (fun _ _ _ _ h => h) (fun _ _ _ h => h)

/-- Close the join of a branch whose arms carry the answer. -/
macro "join_ans" : tactic => `(tactic|
  (intro nJ Γ' jv vs w' hle hvs hq
   obtain ⟨x, hx, hq⟩ := hq
   have hj := join1 hvs hx
   subst hj
   exact ⟨x, AlgorithmLib.Proof.bindAt_last _ _ _ hle, hq⟩))

-- ---------------------------------------------------------------------------
-- Allocation
-- ---------------------------------------------------------------------------

theorem Whole.blk {w : World} {x : UInt64} {n : Nat} (h : Whole w x n) : Blk w x n := h.1

/-- **`n` bytes at `p` are off the `k` bytes at `y`**: outside pinned memory,
    or apart from them. -/
def Off (y : UInt64) (k : Nat) (p : UInt64) (n : Nat) : Prop :=
  ∀ r off oy, decodeAddr p = some (r, off) → decodeAddr y = some (.pinned, oy) →
    r ≠ .pinned ∨ off + n ≤ oy ∨ oy + k ≤ off

/-- **`w'` is `w` with more heap**: what was calm stays calm, every block
    there was is there, every load answers what it answered, and the regions
    a program is handed are as large. What an allocation leaves, and what a
    call's being observed leaves. -/
structure Grows (w w' : World) : Prop where
  calm : Calm w → Calm w'
  blk : ∀ x n, Blk w x n → Blk w' x n
  load : ∀ a k v, w.mem.load a k = some v → w'.mem.load a k = some v
  size : ∀ r, r ≠ .pinned → (w'.mem.region r).size = (w.mem.region r).size
  bytes : ∀ p bs, Bytes w p bs → Bytes w' p bs
  room : ∀ p n, Room w p n → Room w' p n
  whole : ∀ x n, Whole w x n → Whole w' x n
  psize : (w.mem.region .pinned).size ≤ (w'.mem.region .pinned).size

/-- **What `w` has of calm, blocks and room, `w'` has**: what a store leaves,
    and what more heap leaves. -/
structure Stays (w w' : World) : Prop where
  calm : Calm w → Calm w'
  blk : ∀ x n, Blk w x n → Blk w' x n
  size : ∀ r, r ≠ .pinned → (w'.mem.region r).size = (w.mem.region r).size

theorem Stays.refl (w : World) : Stays w w := ⟨id, fun _ _ h => h, fun _ _ => rfl⟩

theorem Stays.trans {w w' w'' : World} (h : Stays w w') (h' : Stays w' w'') : Stays w w'' :=
  ⟨fun c => h'.calm (h.calm c), fun x n b => h'.blk x n (h.blk x n b), fun r hr => (h'.size r hr).trans (h.size r hr)⟩

theorem Grows.stays {w w' : World} (h : Grows w w') : Stays w w' := ⟨h.calm, h.blk, h.size⟩

theorem stays_stored {w : World} {p x : UInt64} {m : Mem} (hs : w.mem.store p 8 x = some m) :
    Stays w (stored w p x m) :=
  ⟨fun c => c.stored hs, fun _ _ b => Blk.keep b hs, fun r _ => store_sizes hs r⟩

/-- Eight bytes with room, outside pinned memory, can be written where no
    worker is running. -/
theorem store_room {m : Mem} {a x : UInt64} {r : Region} {off : Nat} (hd : decodeAddr a = some (r, off))
    (hr : r ≠ .pinned) (hs : off + 8 ≤ (m.region r).size) (hz : m.frozen = false) : ∃ m', m.store a 8 x = some m' := by
  unfold Mem.store
  simp only [hd, Option.bind_eq_bind, Option.bind_some]
  rw [if_neg (by
    have : m.reachable r off 8 = true := by
      simp only [Mem.reachable, hz, Bool.not_false, Bool.true_and, Bool.or_eq_true]
      exact Or.inl (by cases r <;> first | rfl | exact absurd rfl hr)
    simp [this]; omega)]
  exact ⟨_, rfl⟩

theorem Grows.refl (w : World) : Grows w w :=
  ⟨id, fun _ _ h => h, fun _ _ _ h => h, fun _ _ => rfl, fun _ _ h => h, fun _ _ h => h, fun _ _ h => h,
    Nat.le_refl _⟩

theorem Grows.trans {w w' w'' : World} (h : Grows w w') (h' : Grows w' w'') : Grows w w'' :=
  ⟨fun c => h'.calm (h.calm c), fun x n b => h'.blk x n (h.blk x n b), fun a k v l => h'.load a k v (h.load a k v l),
    fun r hr => (h'.size r hr).trans (h.size r hr), fun p bs b => h'.bytes p bs (h.bytes p bs b),
    fun p n b => h'.room p n (h.room p n b), fun x n b => h'.whole x n (h.whole x n b),
    Nat.le_trans h.psize h'.psize⟩

theorem Grows.fld {w w' : World} (h : Grows w w') {s v : UInt64} {j : Nat} (hf : fld w s j = some v) :
    fld w' s j = some v := h.load _ _ _ hf

/-- The world an allocation of `b` bytes leaves. -/
def alloc (w : World) (b : Nat) : World :=
  { w with
    mem := { w.mem with
      pinned := w.mem.pinned ++ ByteArray.mk (Array.replicate (w.mem.pinnedNext - w.mem.pinned.size + b) 0)
      pinnedLive := (w.mem.pinnedNext, b) :: w.mem.pinnedLive }
    heap := (w.mem.pinnedNext, b) :: w.heap }

theorem pinnedNext_ge (m : Mem) : m.pinned.size ≤ m.pinnedNext := by unfold Mem.pinnedNext; omega

theorem alloc_size (w : World) (b : Nat) : (alloc w b).mem.pinned.size = w.mem.pinnedNext + b := by
  show (w.mem.pinned ++ ByteArray.mk _).size = _
  rw [ByteArray.size_append]
  show w.mem.pinned.size + (Array.replicate _ (0 : UInt8)).size = _
  rw [Array.size_replicate]
  have := pinnedNext_ge w.mem
  omega

theorem alloc_grows (w : World) (b : Nat) : Grows w (alloc w b) := by
  have hsz : w.mem.pinned.size ≤ (alloc w b).mem.pinned.size := by
    rw [alloc_size]; have := pinnedNext_ge w.mem; omega
  refine ⟨fun c => c, fun x n h => Blk.alloc h b, fun a k v h => ?_,
    fun r hr => by cases r <;> first | rfl | exact absurd rfl hr, fun p bs h => ?_, fun p n h => ?_,
    fun x n h => ?_, hsz⟩
  rotate_left
  · rcases h with rfl | ⟨r, off, hd, hs, h36, hr, hg⟩
    · exact Or.inl rfl
    · by_cases e : r = .pinned
      · subst e
        refine Or.inr ⟨.pinned, off, hd, Nat.le_trans hs hsz, h36, ?_, fun i hi => ?_⟩
        · simp only [Mem.readable, Bool.and_eq_true, Bool.or_eq_true] at hr ⊢
          refine ⟨hr.1, hr.2.imp id fun ⟨h1, h2⟩ => ⟨?_, h2⟩⟩
          show ((w.mem.pinnedNext, b) :: w.mem.pinnedLive).any _ = true
          rw [List.any_cons, h1, Bool.or_true]
        · rw [← hg i hi]
          exact ByteArray.get!_append_left' _ _ _ (by have : off + bs.length ≤ w.mem.pinned.size := hs; omega)
      · have hreg : (alloc w b).mem.region r = w.mem.region r := by cases r <;> first | rfl | exact absurd rfl e
        exact Or.inr ⟨r, off, hd, by rw [hreg]; exact hs, h36,
          by rw [readable_off_pinned e] at hr ⊢; exact hr, fun i hi => by rw [hreg]; exact hg i hi⟩
  · rcases h with h0 | ⟨r, off, hd, hs, h36, hr⟩
    · exact Or.inl h0
    · by_cases e : r = .pinned
      · subst e
        refine Or.inr ⟨.pinned, off, hd, Nat.le_trans hs hsz, h36, ?_⟩
        simp only [Mem.reachable, Bool.and_eq_true, Bool.or_eq_true] at hr ⊢
        refine ⟨hr.1, hr.2.imp id fun ⟨h1, h2⟩ => ⟨?_, h2⟩⟩
        show ((w.mem.pinnedNext, b) :: w.mem.pinnedLive).any _ = true
        rw [List.any_cons, h1, Bool.or_true]
      · have hreg : (alloc w b).mem.region r = w.mem.region r := by cases r <;> first | rfl | exact absurd rfl e
        refine Or.inr ⟨r, off, hd, by rw [hreg]; exact hs, h36, ?_⟩
        cases r <;> first | exact hr | exact absurd rfl e
  · obtain ⟨off, m, hat, hn, hm, hsz', h36, hlv⟩ := h.1
    obtain ⟨off', hw⟩ := h.2
    rw [hat] at hw
    cases hw
    obtain ⟨hd, hf⟩ := heapAt_decode hat
    have hne : (w.mem.pinnedNext == off) = false := by
      have := pinnedNext_ge w.mem
      simp only [beq_eq_false_iff_ne, ne_eq]; omega
    refine ⟨Blk.alloc h.1 b, off, ?_⟩
    unfold World.heapAt?
    rw [hd]
    show ((w.mem.pinnedNext, b) :: w.heap).find? (·.1 == off) = some (off, n)
    rw [List.find?_cons, hne]; exact hf
  cases hd : decodeAddr a with
  | none => simp [Mem.load, hd] at h
  | some p =>
    obtain ⟨r, off⟩ := p
    obtain ⟨hs, hr, rfl⟩ := load_some hd h
    by_cases e : r = .pinned
    · subst e
      rw [load_of hd (by rw [show (alloc w b).mem.region .pinned = (alloc w b).mem.pinned from rfl, alloc_size]
                         have := pinnedNext_ge w.mem
                         have : off + k ≤ w.mem.pinned.size := hs
                         omega)
        (by
          simp only [Mem.readable, Bool.and_eq_true, Bool.or_eq_true] at hr ⊢
          refine ⟨hr.1, hr.2.imp id fun ⟨h1, h2⟩ => ⟨?_, h2⟩⟩
          show ((w.mem.pinnedNext, b) :: w.mem.pinnedLive).any _ = true
          rw [List.any_cons, h1, Bool.or_true])]
      congr 1
      exact foldr_bytes_congr _ _ fun i hi => ByteArray.get!_append_left' _ _ _ (by
        have := List.mem_range.mp hi
        have : off + k ≤ w.mem.pinned.size := hs
        omega)
    · have hreg : (alloc w b).mem.region r = w.mem.region r := by cases r <;> first | rfl | exact absurd rfl e
      rw [load_of hd (by rw [hreg]; exact hs)
        (by rw [readable_off_pinned e] at hr ⊢; exact hr), hreg]

theorem get!_zeros (b : ByteArray) (n i : Nat) (hi : b.size ≤ i) :
    (b ++ ByteArray.mk (Array.replicate n 0)).get! i = 0 := by
  rw [show (b ++ ByteArray.mk (Array.replicate n 0)).get! i = (b ++ ByteArray.mk (Array.replicate n 0)).data[i]! from rfl,
    ByteArray.data_append]
  have h' : b.data.size ≤ i := hi
  simp only [getElem!_def, Array.getElem?_append_right h', Array.getElem?_replicate]
  by_cases hlt : i - b.data.size < n
  · rw [if_pos hlt]
  · rw [if_neg hlt]; rfl

/-- **A fresh block is zero.** -/
theorem alloc_zero {w : World} {b j : Nat} (hc : Calm w) (hj : j + 8 ≤ b) (hroom : w.mem.pinnedNext + b ≤ 2 ^ 36) :
    fld (alloc w b) (addrOf .pinned w.mem.pinnedNext) j = some 0 := by
  have hd := Sem.decodeAddr_add (i := j) (decodeAddr_pinned (off := w.mem.pinnedNext) (by omega))
    (by rw [Sem.regionSpan_toNat]; omega)
  unfold fld
  rw [load_of hd (by rw [show (alloc w b).mem.region .pinned = (alloc w b).mem.pinned from rfl, alloc_size]; omega)
    (by
      simp only [Mem.readable, Bool.and_eq_true, Bool.or_eq_true]
      refine ⟨by simpa using hc.1, Or.inr ⟨?_, ?_⟩⟩
      · show ((w.mem.pinnedNext, b) :: w.mem.pinnedLive).any _ = true
        rw [List.any_cons]
        simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq]
        exact Or.inl ⟨by omega, by omega⟩
      · show (alloc w b).mem.busy.all _ = true
        rw [show (alloc w b).mem.busy = w.mem.busy from rfl, hc.2]; rfl)]
  congr 1
  unfold bytesAt
  refine (foldr_fun_congr _ (fun _ => 0) _ _ fun i _ => ?_).trans (by decide)
  rw [show (alloc w b).mem.region .pinned = w.mem.pinned ++ ByteArray.mk (Array.replicate _ 0) from rfl,
    get!_zeros _ _ _ (by have := pinnedNext_ge w.mem; omega)]
  rfl

/-- A fresh block starts where no block there was starts. -/
theorem alloc_fresh {w : World} {y : UInt64} {k : Nat} (h : Blk w y k) (hroom : w.mem.pinnedNext < 2 ^ 36) :
    y ≠ addrOf .pinned w.mem.pinnedNext := by
  obtain ⟨off, m, hat, hn, hm, hsz, h36, hlv⟩ := h
  obtain ⟨hd, -⟩ := heapAt_decode hat
  rintro rfl
  rw [decodeAddr_pinned hroom] at hd
  have := pinnedNext_ge w.mem
  have : w.mem.pinnedNext = off := by simpa using hd
  omega

/-- A fresh block starts past every block there was. -/
theorem alloc_before {w : World} {y : UInt64} {k : Nat} (h : Blk w y k) (hroom : w.mem.pinnedNext < 2 ^ 36) :
    Before y k (addrOf .pinned w.mem.pinnedNext) := by
  obtain ⟨off, m, hat, hn, hm, hsz, h36, hlv⟩ := h
  obtain ⟨hd, -⟩ := heapAt_decode hat
  have := pinnedNext_ge w.mem
  exact ⟨off, _, hd, decodeAddr_pinned hroom, by omega⟩

/-- **`x` is a block of `n` bytes that `w'` has and `w` did not**: it starts
    past all of `w`'s pinned memory, and is zero. -/
structure Fresh (w w' : World) (x : UInt64) (n : Nat) : Prop where
  blk : Blk w' x n
  past : ∃ ox, decodeAddr x = some (.pinned, ox) ∧ (w.mem.region .pinned).size ≤ ox
  zero : Calm w → ∀ j, j + 8 ≤ n → fld w' x j = some 0
  whole : Whole w' x n

theorem alloc_heapAt (w : World) (b : Nat) (hroom : w.mem.pinnedNext < 2 ^ 36) :
    (alloc w b).heapAt? (addrOf .pinned w.mem.pinnedNext) = some (w.mem.pinnedNext, b) := by
  unfold World.heapAt?
  rw [decodeAddr_pinned hroom]
  show ((w.mem.pinnedNext, b) :: w.heap).find? (·.1 == w.mem.pinnedNext) = some _
  rw [List.find?_cons]; simp

/-- A fresh block starts past every block there was. -/
theorem Fresh.before {w w' : World} {x y : UInt64} {n k : Nat} (f : Fresh w w' x n) (h : Blk w y k) :
    Before y k x := by
  obtain ⟨ox, hdx, hpast⟩ := f.past
  obtain ⟨off, m, hat, hn, hm, hsz, h36, hlv⟩ := h
  obtain ⟨hd, -⟩ := heapAt_decode hat
  have : off + m ≤ (w.mem.region .pinned).size := hsz
  exact ⟨off, ox, hd, hdx, by omega⟩

theorem Fresh.ne {w w' : World} {x y : UInt64} {n k : Nat} (f : Fresh w w' x n) (h : Blk w y k) : y ≠ x := by
  obtain ⟨ox, hdx, hpast⟩ := f.past
  obtain ⟨off, m, hat, hn, hm, hsz, h36, hlv⟩ := h
  obtain ⟨hd, -⟩ := heapAt_decode hat
  have : off + m ≤ (w.mem.region .pinned).size := hsz
  rintro rfl
  rw [hd] at hdx
  cases hdx
  omega

/-- **What `calloc` does**, asked for `n` bytes, not none: it answers null and
    leaves the world, or the start of a fresh block in a world with more heap. -/
theorem calloc_cases {n sz : UInt64} (w : World) (hn : n ≠ 0) (hsz : sz = 1) :
    extCall (.c .calloc) [.sc .i64 n, .sc .i64 sz] w = some (some (.sc .i64 0), w) ∨
    (w.mem.pinnedNext + n.toNat ≤ 2 ^ 36 ∧
      extCall (.c .calloc) [.sc .i64 n, .sc .i64 sz] w =
        some (some (.sc .i64 (addrOf .pinned w.mem.pinnedNext)), alloc w n.toNat)) := by
  subst hsz
  have hpos : n.toNat ≠ 0 := fun e => hn (UInt64.toNat_inj.mp e)
  simp only [extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true, Bool.false_eq_true, if_false,
    List.mapM_cons, List.mapM_nil, asBits, Option.pure_def, Option.bind_eq_bind, Option.bind_some, cCall,
    UInt64.toNat_one, Nat.mul_one, beq_iff_eq, hpos]
  by_cases hroom : w.mem.pinnedNext + n.toNat > regionSpan.toNat
  · left; rw [if_pos hroom]
  · right
    rw [if_neg hroom]
    exact ⟨by rw [Sem.regionSpan_toNat] at hroom; omega, rfl⟩

theorem obs_grows (w : World) (c : Callee) (vs : List V) : Grows w (obsCall w c vs) :=
  ⟨id, fun _ _ h => h, fun _ _ _ h => h, fun _ _ => rfl, fun _ _ h => h, fun _ _ h => h, fun _ _ h => h,
    Nat.le_refl _⟩

section steps
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- **`calloc`** of a number of bytes, not none: null, or a zeroed block that
    starts past every block there was, in a world with more heap. -/
theorem wp_calloc {a : Slot .i64} {n : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 n)) (hn : n ≠ 0)
    (h : ∀ x w', Grows w w' → (x = 0 ∨ Fresh w w' x n.toNat) →
      Q (Γ.push (.sc .i64 (c64 1))).size ((Γ.push (.sc .i64 (c64 1))).push (.sc .i64 x)) w') :
    wp cfg d J (Lib.calloc a) Q Γ w := by
  unfold Lib.calloc
  rw [wp_bind]
  apply wp_iconst64
  simp only [Prog.ext, Prog.callLocal, wp_callLocal, wp_ret]
  intro vs hvs
  have hvs' : vs = [.sc .i64 n, .sc .i64 (c64 1)] := by
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, get_push ha, get_push_size, Option.pure_def,
      Option.bind_eq_bind, Option.bind_some, Option.some.injEq] at hvs
    exact hvs.symm
  subst hvs'
  have hw0 := obs_grows w (.ext (.c .calloc)) [.sc .i64 n, .sc .i64 (c64 1)]
  simp only [callOf]
  rcases calloc_cases (sz := c64 1) (obsCall w (.ext (.c .calloc)) [.sc .i64 n, .sc .i64 (c64 1)]) hn (by decide)
    with hc | ⟨hroom, hc⟩
  · rw [hc]
    refine ⟨nofun, ?_⟩
    rintro x w' hxw v hv
    cases hxw
    cases hv
    exact h 0 _ hw0 (Or.inl rfl)
  · rw [hc]
    refine ⟨nofun, ?_⟩
    rintro x w' hxw v hv
    cases hxw
    cases hv
    have hpos : n.toNat ≠ 0 := fun e => hn (UInt64.toNat_inj.mp e)
    have hblk : Blk (alloc (obsCall w (.ext (.c .calloc)) [.sc .i64 n, .sc .i64 (c64 1)]) n.toNat)
        (addrOf .pinned (obsCall w (.ext (.c .calloc)) [.sc .i64 n, .sc .i64 (c64 1)]).mem.pinnedNext) n.toNat := by
      have hne : addrOf .pinned (obsCall w (.ext (.c .calloc)) [.sc .i64 n, .sc .i64 (c64 1)]).mem.pinnedNext ≠ 0 := by
        intro e
        have := decodeAddr_pinned
          (off := (obsCall w (.ext (.c .calloc)) [.sc .i64 n, .sc .i64 (c64 1)]).mem.pinnedNext) (by omega)
        rw [e, show decodeAddr 0 = none from by decide] at this
        cases this
      have := calloc_answer (vs := [.sc .i64 n, .sc .i64 (c64 1)]) rfl hc hne
      rw [show (c64 1).toNat = 1 from by decide, Nat.mul_one] at this
      exact this
    exact h _ _ (hw0.trans (alloc_grows _ _)) (Or.inr ⟨hblk,
      ⟨_, decodeAddr_pinned (off := (obsCall w (.ext (.c .calloc)) [.sc .i64 n, .sc .i64 (c64 1)]).mem.pinnedNext)
        (by omega), pinnedNext_ge w.mem⟩,
      fun hcalm j hj => alloc_zero hcalm hj hroom,
      hblk, _, alloc_heapAt (obsCall w (.ext (.c .calloc)) [.sc .i64 n, .sc .i64 (c64 1)]) n.toNat (by omega)⟩)

/-- A field of a block, written. -/
theorem wp_setFld {v a : Slot .i64} {x s : UInt64} {j n : Nat} {Q : Unit → Env → World → Prop}
    (hv : Γ[v]? = some (.sc .i64 x)) (ha : Γ[a]? = some (.sc .i64 (s + UInt64.ofNat j)))
    (hb : Blk w s n) (hc : Calm w) (hj : j + 8 ≤ n)
    (h : ∀ m, w.mem.store (s + UInt64.ofNat j) 8 x = some m → Q () Γ (stored w (s + UInt64.ofNat j) x m)) :
    wp cfg d J (storeI64 v a) Q Γ w := by
  obtain ⟨m, hm⟩ := Blk.fld_store hb hc hj x
  exact wp_storeI64 hv ha hm (h m hm)

end steps

/-- **What a library function that answers nothing does**: run on `args` from
    `w`, its body neither misuses a call nor faults, and where it ends it
    leaves a world `Q` holds of. -/
def Does (cfg : Cfg) (args : List V) (w : World) (p : Body) (params : List ClifTy) (Q : World → Prop) : Prop :=
  Triple cfg (At args.toArray w) (emit p params) { ok := fun _ w' => Q w', faultOk := false }

theorem does_of_wp {cfg : Cfg} {p : Body} {params : List ClifTy} {args : List V} {w : World} {Q : World → Prop}
    (hwp : wp cfg 0 { ok := fun _ _ => True, faultOk := false } (fnBody params p) (fun _ _ w' => Q w')
      args.toArray w)
    (hf : Fine (emitGo (fnBody params p) ⟨params.length, 0, [], [], [], none⟩).2)
    (hargs : args.length = params.length) :
    Does cfg args w p params Q := by
  have h1 := emit_triple (cfg := cfg) (J := { ok := fun _ _ => True, faultOk := false })
    (P := fun Γ' w' => Γ' = args.toArray ∧ w' = w) (Q := fun _ _ w' => Q w') (params := params)
    (P0 := At args.toArray w)
    (PT.conseq (wp_sound cfg 0 _ (fnBody params p) _) (fun Γ' w' h => by obtain ⟨rfl, rfl⟩ := h; exact hwp)
      (fun _ _ _ h => h)) hf
    (fun Γ' w' h => ⟨by rw [h.1]; simp [hargs], h⟩)
  exact Hoare.conseq h1 (fun _ _ h => h) (fun _ _ h => h) (fun _ _ _ _ h => h) (fun _ _ _ h => h)

-- ---------------------------------------------------------------------------
-- Freeing
-- ---------------------------------------------------------------------------

/-- The world freeing the allocation `(off, n)` leaves. -/
def freed (w : World) (off n : Nat) : World :=
  { w with mem := { w.mem with pinnedLive := w.mem.pinnedLive.erase (off, n) }, heap := w.heap.erase (off, n) }

/-- **`p` is not an address of the heap.** -/
def Out (p : UInt64) : Prop := ∀ r off, decodeAddr p = some (r, off) → r ≠ .pinned

/-- **`w'` is `w` with the block at `x` freed**, or as it was: what was calm
    stays calm, every other block is there with its fields, and memory
    outside the heap is as it was. -/
structure Freed (w w' : World) (x : UInt64) : Prop where
  calm : Calm w → Calm w'
  blk : ∀ y n, y ≠ x → Blk w y n → Blk w' y n
  fld : ∀ y n j v, y ≠ x → Blk w y n → j + 8 ≤ n → fld w y j = some v → fld w' y j = some v
  out : ∀ a k r off, decodeAddr a = some (r, off) → r ≠ .pinned → w'.mem.load a k = w.mem.load a k
  size : ∀ r, r ≠ .pinned → (w'.mem.region r).size = (w.mem.region r).size
  frozen : w'.mem.frozen = w.mem.frozen
  whole : ∀ y n, y ≠ x → Whole w y n → Whole w' y n
  bytes : ∀ n, Whole w x n → ∀ p bs, Bytes w p bs → Off x n p bs.length → Bytes w' p bs
  room : ∀ n, Whole w x n → ∀ p k, Room w p k → Off x n p k → Room w' p k
  load : ∀ n, Whole w x n → ∀ a k v, 0 < k → Off x n a k → w.mem.load a k = some v → w'.mem.load a k = some v
  psize : (w'.mem.region .pinned).size = (w.mem.region .pinned).size
  outBytes : ∀ p bs, Out p → Bytes w p bs → Bytes w' p bs
  outRoom : ∀ p k, Out p → Room w p k → Room w' p k

theorem Freed.refl (w : World) (x : UInt64) : Freed w w x :=
  ⟨id, fun _ _ _ h => h, fun _ _ _ _ _ _ _ h => h, fun _ _ _ _ _ _ => rfl, fun _ _ => rfl, rfl,
    fun _ _ _ h => h, fun _ _ _ _ h _ => h, fun _ _ _ _ h _ => h, fun _ _ _ _ _ _ _ h => h, rfl,
    fun _ _ _ h => h, fun _ _ _ h => h⟩

theorem find?_erase_ne {l : List (Nat × Nat)} {a : Nat × Nat} {k : Nat} (h : a.1 ≠ k) :
    (l.erase a).find? (·.1 == k) = l.find? (·.1 == k) := by
  induction l with
  | nil => rfl
  | cons b l ih =>
    rw [List.erase_cons]
    by_cases e : b = a
    · subst e
      simp only [beq_self_eq_true, if_true]
      rw [List.find?_cons, show (b.1 == k) = false from by simpa using h]
    · rw [if_neg (by simpa using e), List.find?_cons, List.find?_cons, ih]

/-- Two heap allocations that start at different addresses start at different
    offsets. -/
theorem heapAt_off_ne {w : World} {x y : UInt64} {ox nx oy ny : Nat} (hx : w.heapAt? x = some (ox, nx))
    (hy : w.heapAt? y = some (oy, ny)) (hne : y ≠ x) : oy ≠ ox := by
  obtain ⟨hdx, -⟩ := heapAt_decode hx
  obtain ⟨hdy, -⟩ := heapAt_decode hy
  rintro rfl
  exact hne ((Sem.decodeAddr_some hdy).1.trans (Sem.decodeAddr_some hdx).1.symm)

theorem freed_spec {w : World} {x : UInt64} {off n : Nat} (hx : w.heapAt? x = some (off, n)) :
    Freed w (freed w off n) x := by
  have hkeep : ∀ (o len : Nat), 0 < len → (o + len ≤ off ∨ off + n ≤ o) →
      (w.mem.pinnedLive.any fun x => decide (x.1 ≤ o) && decide (o + len ≤ x.1 + x.2)) = true →
      ((w.mem.pinnedLive.erase (off, n)).any fun x => decide (x.1 ≤ o) && decide (o + len ≤ x.1 + x.2)) = true := by
    intro o len hl hcl ha
    obtain ⟨a, ha, hb⟩ := List.any_eq_true.mp ha
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
    refine List.any_eq_true.mpr ⟨a, (List.mem_erase_of_ne ?_).mpr ha, by
      simp only [Bool.and_eq_true, decide_eq_true_eq]; exact hb⟩
    rintro rfl
    simp only at hb
    omega
  have hxd := (heapAt_decode hx).1
  have hreg : ∀ r, (freed w off n).mem.region r = w.mem.region r := fun r => by cases r <;> rfl
  have hob : ∀ p bs, Out p → Bytes w p bs → Bytes (freed w off n) p bs := by
    intro p bs ho hb
    rcases hb with rfl | ⟨r, o, hd, hs, h36, hr, hg⟩
    · exact Or.inl rfl
    have e := ho r o hd
    exact Or.inr ⟨r, o, hd, by rw [hreg]; exact hs, h36, by rw [readable_off_pinned e] at hr ⊢; exact hr,
      fun i hi => by rw [hreg]; exact hg i hi⟩
  have hor : ∀ p k, Out p → Room w p k → Room (freed w off n) p k := by
    intro p k ho hrm
    rcases hrm with h0 | ⟨r, o, hd, hs, h36, hr⟩
    · exact Or.inl h0
    have e := ho r o hd
    exact Or.inr ⟨r, o, hd, by rw [hreg]; exact hs, h36, by cases r <;> first | exact hr | exact absurd rfl e⟩
  refine ⟨fun c => c, ?_, ?_, ?_, fun r hr => by cases r <;> first | rfl | exact absurd rfl hr, rfl, ?_, ?_, ?_, ?_,
    rfl, hob, hor⟩
  rotate_left 3
  · rintro y k hne ⟨hb, oy, hy⟩
    have ho := heapAt_off_ne hx hy hne
    obtain ⟨hd, hf⟩ := heapAt_decode hy
    obtain ⟨oy', m, hat, hk, hm, hsz, h36, hlv⟩ := hb
    rw [hy] at hat
    cases hat
    have hat' : (freed w off n).heapAt? y = some (oy, k) := by
      unfold World.heapAt?
      rw [hd]
      show (w.heap.erase (off, n)).find? (·.1 == oy) = some (oy, k)
      rw [find?_erase_ne (by exact fun e => ho e.symm)]; exact hf
    exact ⟨⟨oy, k, hat', hk, hm, hsz, h36,
      (List.mem_erase_of_ne (by intro e; exact ho (congrArg Prod.fst e))).mpr hlv⟩, oy, hat'⟩
  · rintro n' ⟨-, o', hw⟩ p bs hb ho
    rw [hx] at hw
    cases hw
    by_cases hl : bs.length = 0
    · exact Or.inl (List.length_eq_zero_iff.mp hl)
    rcases hb with rfl | ⟨r, o, hd, hs, h36, hr, hg⟩
    · exact Or.inl rfl
    refine Or.inr ⟨r, o, hd, by rw [show (freed w off n).mem.region r = w.mem.region r from by cases r <;> rfl]; exact hs,
      h36, ?_, fun i hi => by
        rw [show (freed w off n).mem.region r = w.mem.region r from by cases r <;> rfl]; exact hg i hi⟩
    by_cases e : r = .pinned
    · subst e
      have hcl := (ho .pinned o off hd hxd).resolve_left (fun c => c rfl)
      simp only [Mem.readable, Bool.and_eq_true, Bool.or_eq_true] at hr ⊢
      refine ⟨hr.1, hr.2.imp id fun ⟨h1, h2⟩ => ⟨?_, h2⟩⟩
      exact hkeep o bs.length (by omega) hcl h1
    · rw [readable_off_pinned e] at hr ⊢; exact hr
  · rintro n' ⟨-, o', hw⟩ p k hrm ho
    rw [hx] at hw
    cases hw
    rcases hrm with h0 | ⟨r, o, hd, hs, h36, hr⟩
    · exact Or.inl h0
    by_cases hk0 : k = 0
    · exact Or.inl hk0
    refine Or.inr ⟨r, o, hd, by rw [show (freed w off n).mem.region r = w.mem.region r from by cases r <;> rfl]; exact hs,
      h36, ?_⟩
    by_cases e : r = .pinned
    · subst e
      have hcl := (ho .pinned o off hd hxd).resolve_left (fun c => c rfl)
      simp only [Mem.reachable, Bool.and_eq_true, Bool.or_eq_true] at hr ⊢
      refine ⟨hr.1, hr.2.imp id fun ⟨h1, h2⟩ => ⟨?_, h2⟩⟩
      exact hkeep o k (by omega) hcl h1
    · cases r <;> first | exact hr | exact absurd rfl e
  · rintro n' ⟨-, o', hw⟩ a k v hk ho hl
    rw [hx] at hw
    cases hw
    cases hd : decodeAddr a with
    | none => simp [Mem.load, hd] at hl
    | some q =>
      obtain ⟨r, o⟩ := q
      obtain ⟨hs, hr, rfl⟩ := load_some hd hl
      have hreg : (freed w off n).mem.region r = w.mem.region r := by cases r <;> rfl
      rw [load_of (m := (freed w off n).mem) hd (by rw [hreg]; exact hs) (by
        by_cases e : r = .pinned
        · subst e
          have hcl := (ho .pinned o off hd hxd).resolve_left (fun c => c rfl)
          simp only [Mem.readable, Bool.and_eq_true, Bool.or_eq_true] at hr ⊢
          refine ⟨hr.1, hr.2.imp id fun ⟨h1, h2⟩ => ⟨?_, h2⟩⟩
          exact hkeep o k hk hcl h1
        · rw [readable_off_pinned e] at hr ⊢; exact hr), hreg]
  · rintro y k hne ⟨oy, m, hat, hk, hm, hsz, h36, hlv⟩
    have ho := heapAt_off_ne hx hat hne
    obtain ⟨hd, hf⟩ := heapAt_decode hat
    refine ⟨oy, m, ?_, hk, hm, hsz, h36, ?_⟩
    · unfold World.heapAt?
      rw [hd]
      show (w.heap.erase (off, n)).find? (·.1 == oy) = some (oy, m)
      rw [find?_erase_ne (by exact fun e => ho e.symm)]; exact hf
    · show (oy, m) ∈ w.mem.pinnedLive.erase (off, n)
      exact (List.mem_erase_of_ne (by intro e; exact ho (congrArg Prod.fst e))).mpr hlv
  · rintro y k j v hne ⟨oy, m, hat, hk, hm, hsz, h36, hlv⟩ hj hv
    have ho := heapAt_off_ne hx hat hne
    obtain ⟨hd, -⟩ := heapAt_decode hat
    have hdj := Sem.decodeAddr_add (i := j) hd (by rw [Sem.regionSpan_toNat]; omega)
    obtain ⟨hs, hr, rfl⟩ := load_some hdj hv
    unfold LibSpec.fld
    rw [load_of (m := (freed w off n).mem) hdj (show _ ≤ ((freed w off n).mem.region .pinned).size from hs) (by
      simp only [Mem.readable, Bool.and_eq_true, Bool.or_eq_true] at hr ⊢
      refine ⟨hr.1, hr.2.imp id fun ⟨_, h2⟩ => ⟨?_, h2⟩⟩
      show (w.mem.pinnedLive.erase (off, n)).any _ = true
      rw [List.any_eq_true]
      exact ⟨(oy, m), (List.mem_erase_of_ne (by intro e; exact ho (congrArg Prod.fst e))).mpr hlv,
        by simp only [Bool.and_eq_true, decide_eq_true_eq]; omega⟩)]
    rfl
  · intro a k r o hd hr
    unfold Mem.load
    simp only [hd, Option.bind_eq_bind, Option.bind_some]
    rw [show (freed w off n).mem.region r = w.mem.region r from by cases r <;> rfl,
      readable_off_pinned hr, readable_off_pinned hr]
    rfl

section steps
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- **`free`** of null or of a block. -/
theorem wp_free {a : Slot .i64} {x : UInt64} {n : Nat} {Q : Unit → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hx : x = 0 ∨ Blk w x n) (h : ∀ w', Freed w w' x → Q () Γ w') :
    wp cfg d J (Lib.free a) Q Γ w := by
  unfold Lib.free
  simp only [Prog.ext, Prog.callLocal, wp_callLocal, wp_ret]
  intro vs hvs
  have hvs' : vs = [.sc .i64 x] := by
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, ha, Option.pure_def, Option.bind_eq_bind,
      Option.bind_some, Option.some.injEq] at hvs
    exact hvs.symm
  subst hvs'
  simp only [callOf, extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true, Bool.false_eq_true,
    if_false, List.mapM_cons, List.mapM_nil, asBits, Option.pure_def, Option.bind_eq_bind, Option.bind_some, cCall]
  rcases hx with rfl | hb
  · refine ⟨nofun, ?_⟩
    rintro r w' hrw
    cases hrw
    exact h _ ⟨id, fun _ _ _ h => h, fun _ _ _ _ _ _ _ h => h, fun _ _ _ _ _ _ => rfl, fun _ _ => rfl, rfl,
      fun _ _ _ h => h, fun _ _ _ _ h _ => h, fun _ _ _ _ h _ => h, fun _ _ _ _ _ _ _ h => h, rfl,
      fun _ _ _ h => h, fun _ _ _ h => h⟩
  · have hx0 : (x == 0) = false := by simpa using Blk.ne_zero hb
    obtain ⟨off, m, hat, -⟩ := hb
    have hat' : (obsCall w (.ext (.c .free)) [.sc .i64 x]).heapAt? x = some (off, m) := hat
    simp only [hx0, hat', Bool.false_eq_true, if_false, Option.bind_some]
    refine ⟨nofun, ?_⟩
    rintro r w' hrw
    cases hrw
    have f := freed_spec hat'
    exact h _ ⟨f.calm, f.blk, f.fld, f.out, f.size, f.frozen, f.whole, f.bytes, f.room, f.load, f.psize,
      f.outBytes, f.outRoom⟩

/-- A branch taken for effect: `R` holds of the world either way. -/
theorem wp_when {cc : ICmpCond} {a b : Slot .i64} {x y : UInt64} {thn : Prog Slot Lvl Unit}
    {Q : Unit → Env → World → Prop} (R : World → Prop)
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (ht : cmpInt cc .i64 x y = true →
      wp cfg d J thn (fun _ Γ' w' => Ext Γ Γ' ∧ R w') (Γ.push (boolV true)) w)
    (he : cmpInt cc .i64 x y = false → R w)
    (hk : ∀ Γ' w', Ext Γ Γ' → R w' → Q () Γ' w') :
    wp cfg d J (when cc a b thn) Q Γ w := by
  unfold Prog.when
  rw [wp_bind]
  refine wp_ifte (fun _ Γ' w' => Ext Γ Γ' ∧ R w') ha hb ?_ ?_ ?_
  · intro h
    wp_seq
    exact ht h
  · intro h ne hne
    wp_seq
    exact ⟨by ext_env, he h⟩
  · intro nJ Γ' jv vs w' hle hvs ⟨hext, hr⟩
    wp_seq
    exact hk _ _ (hext.bindAt hle _) hr

end steps

/-- Null, or the start of a block. -/
def Own (w : World) (p : UInt64) : Prop := p = 0 ∨ ∃ n, Blk w p n

theorem Own.freed {w w' : World} {p x : UInt64} (h : Own w p) (f : Freed w w' x) (hp : p = 0 ∨ p ≠ x) : Own w' p := by
  rcases h with rfl | ⟨n, hb⟩
  · exact Or.inl rfl
  · exact Or.inr ⟨n, f.blk _ _ (hp.resolve_left (Blk.ne_zero hb)) hb⟩

theorem Own.grows {w w' : World} {p : UInt64} (h : Own w p) (g : Grows w w') : Own w' p :=
  h.imp id fun ⟨n, hb⟩ => ⟨n, g.blk _ _ hb⟩

theorem Own.stored {w : World} {p q x : UInt64} {m : Mem} (h : Own w p) (hm : w.mem.store q 8 x = some m) :
    Own (stored w q x m) p :=
  h.imp id fun ⟨n, hb⟩ => ⟨n, Blk.keep hb hm⟩

/-- What `calloc` answers is null or a block. -/
theorem fresh_own {w w' : World} {x : UInt64} {n : Nat} (hx : x = 0 ∨ Fresh w w' x n) : Own w' x :=
  hx.imp id fun h => ⟨n, h.blk⟩

/-- What `calloc` answers is not a block there was. -/
theorem fresh_ne {w w' : World} {x s : UInt64} {n k : Nat} (hx : x = 0 ∨ Fresh w w' x n) (hs : Blk w s k) :
    x ≠ s := by
  rcases hx with rfl | f
  · exact (Blk.ne_zero hs).symm
  · exact (f.ne hs).symm

/-- …and is null or not what was null or a block. -/
theorem fresh_apart {w w' : World} {x p : UInt64} {n : Nat} (hx : x = 0 ∨ Fresh w w' x n) (hp : Own w p) :
    x = 0 ∨ x ≠ p := by
  rcases hx with rfl | f
  · exact Or.inl rfl
  · rcases hp with rfl | ⟨k, hk⟩
    · exact Or.inr (Blk.ne_zero f.blk)
    · exact Or.inr (f.ne hk).symm

/-- Close what evaluating an operation leaves: its side conditions. -/
macro "op_fin" : tactic => `(tactic| first
  | done
  | rfl
  | decide
  | (constructor <;> first | rfl | decide)
  | (intro h; exact absurd h (by decide)))

section ops
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- An operation that evaluates. -/
theorem wp_val {ty : ClifTy} {o : Op' Slot ty} {v : V} {Q : Slot ty → Env → World → Prop}
    (he : evalOp w.mem Γ o.erase = some v) (h : Q Γ.size (Γ.push v) w) : wp cfg d J (op o) Q Γ w := by
  simp only [op, wp_op, wp_ret, he]
  exact ⟨nofun, fun v' hv => by cases hv; exact h⟩

theorem wp_isub {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x - y))) w) : wp cfg d J (isub a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, bin, Sem.get, ha, hb, norm64] <;> op_fin) h

theorem wp_imul {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x * y))) w) : wp cfg d J (imul a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, bin, Sem.get, ha, hb, norm64] <;> op_fin) h

theorem wp_band {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x &&& y))) w) : wp cfg d J (band a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipIntBits, Sem.get, ha, hb, norm64] <;> op_fin) h

theorem wp_bor {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x ||| y))) w) : wp cfg d J (bor a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipIntBits, Sem.get, ha, hb, norm64] <;> op_fin) h

theorem wp_bxor {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x ^^^ y))) w) : wp cfg d J (bxor a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipIntBits, Sem.get, ha, hb, norm64] <;> op_fin) h

theorem and_wide64 (z : UInt64) : z &&& widthMask .i64 = z := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, show (widthMask .i64).toNat = 2 ^ 64 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
    Nat.mod_eq_of_lt z.toNat_lt]

theorem wp_ishlImm {a : Slot .i64} {x : UInt64} {imm : Int} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q (Γ.push (.sc .i64 (c64 imm))).size
      ((Γ.push (.sc .i64 (c64 imm))).push (.sc .i64 (x <<< (c64 imm % 64)))) w) :
    wp cfg d J (ishlImm a imm) Q Γ w := by
  unfold ishlImm
  rw [wp_bind]
  apply wp_iconst64
  exact wp_val (by simp [Op'.erase, evalOp, shiftBin, Sem.get, get_push ha, norm64] <;> op_fin) h

theorem wp_ushrImm {a : Slot .i64} {x : UInt64} {imm : Int} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q (Γ.push (.sc .i64 (c64 imm))).size
      ((Γ.push (.sc .i64 (c64 imm))).push (.sc .i64 (x >>> (c64 imm % 64)))) w) :
    wp cfg d J (ushrImm a imm) Q Γ w := by
  unfold ushrImm
  rw [wp_bind]
  apply wp_iconst64
  exact wp_val (by simp [Op'.erase, evalOp, shiftBin, Sem.get, get_push ha, norm64, and_wide64] <;> op_fin) h

theorem wp_umax {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (if x ≤ y then y else x))) w) : wp cfg d J (umax a b) Q Γ w := by
  refine wp_val (o := .ibin .umax a b (by decide)) (v := .sc .i64 (if x ≤ y then y else x)) ?_ h
  simp only [Op'.erase, evalOp, bin, Sem.get, ha, hb, ibinOp]
  simp only [Option.bind_eq_bind, Option.bind_some, BEq.rfl, Bool.true_and]
  rw [if_pos (by rfl), norm64, and_wide64, and_wide64]
  congr 2
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by have := x.toNat_lt; have := y.toNat_lt; omega)]
  split
  · rename_i hle; exact Nat.max_eq_right (UInt64.le_iff_toNat_le.mp hle)
  · rename_i hle; exact Nat.max_eq_left (by have := (not_congr UInt64.le_iff_toNat_le).mp hle; omega)

/-- A comparison, as a value. -/
theorem wp_icmp {cc : ICmpCond} {a b : Slot .i64} {x y : UInt64} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (boolV (cmpInt cc .i64 x y))) w) : wp cfg d J (icmp cc a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipIntCmp, Sem.get, ha, hb] <;> op_fin) h

theorem wp_bor8 {a b : Slot .i8} {p q : Bool} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (boolV p)) (hb : Γ[b]? = some (boolV q))
    (h : Q Γ.size (Γ.push (boolV (p || q))) w) : wp cfg d J (bor a b) Q Γ w :=
  wp_val (by cases p <;> cases q <;> simp [Op'.erase, evalOp, zipIntBits, Sem.get, ha, hb, boolV] <;> op_fin) h

theorem wp_select {c : Slot .i8} {a b : Slot .i64} {p : Bool} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (hc : Γ[c]? = some (boolV p)) (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (if p then x else y))) w) : wp cfg d J (select c a b) Q Γ w :=
  wp_val (by cases p <;> simp [Op'.erase, evalOp, Sem.get, ha, hb, hc, boolV, Sem.isTrue, V.ty] <;> op_fin) h

theorem wp_uextend64 {a : Slot .i32} {x : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x))
    (h : Q Γ.size (Γ.push (.sc .i64 (x &&& 0xFFFFFFFF))) w) : wp cfg d J (uextend64 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, un, Sem.get, ha] <;> op_fin) h

theorem wp_iconst8 {k : Int} {Q : Slot .i8 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (ofInt .i8 k)) w) : wp cfg d J (iconst .i8 k) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp] <;> op_fin) h

theorem wp_iconst32 {k : Int} {Q : Slot .i32 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (ofInt .i32 k)) w) : wp cfg d J (iconst32 k) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp] <;> op_fin) h

theorem wp_uload8 {a : Slot .i64} {p b : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 p)) (hl : w.mem.load p 1 = some b)
    (h : Q Γ.size (Γ.push (.sc .i64 b)) w) : wp cfg d J (uload8_64 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, hl, norm64] <;> op_fin) h

end ops

-- ---------------------------------------------------------------------------
-- Loops
-- ---------------------------------------------------------------------------

/-- The first value a join binds, whatever the environment's size. -/
theorem bindAt_head (Γ : Env) (n : Nat) (x : V) (xs : List V) : (bindAt Γ n (x :: xs))[n]? = some x := by
  rw [bindAt_get?, if_neg (Nat.lt_irrefl n)]; simp

theorem bindAt_second (Γ : Env) (n : Nat) (x y : V) (xs : List V) :
    (bindAt Γ n (x :: y :: xs))[n + 1]? = some y := by
  rw [bindAt_get?, if_neg (by omega)]; simp

/-- A join at or past where `Γ` ends keeps what `Γ` holds. -/
theorem Ext.bindAt' {Γ Γ' : Env} (h : Ext Γ Γ') {n : Nat} (hn : Γ.size ≤ n) (vs : List V) :
    Ext Γ (Sem.bindAt Γ' n vs) := by
  intro i v hi
  have hlt : i < Γ.size := by
    rcases Nat.lt_or_ge i Γ.size with hlt | hge
    · exact hlt
    · rw [Array.getElem?_eq_none hge] at hi; cases hi
  have hi' := h i v hi
  have hlt' : i < Γ'.size := by
    rcases Nat.lt_or_ge i Γ'.size with hlt | hge
    · exact hlt
    · rw [Array.getElem?_eq_none hge] at hi'; cases hi'
  rw [bindAt_lt (by omega) hlt']; exact hi'

theorem Ext.trans {Γ Γ' Γ'' : Env} (h : Ext Γ Γ') (h' : Ext Γ' Γ'') : Ext Γ Γ'' := fun i v hi => h' i v (h i v hi)

theorem Ext.size {Γ Γ' : Env} (h : Ext Γ Γ') : Γ.size ≤ Γ'.size := by
  rcases Nat.lt_or_ge Γ'.size Γ.size with hlt | hge
  · have := h Γ'.size Γ[Γ'.size] (Array.getElem?_eq_getElem hlt)
    rw [Array.getElem?_eq_none (Nat.le_refl _)] at this; cases this
  · exact hge

theorem ofNat_lt_iff {i : Nat} {n : UInt64} (hi : i ≤ n.toNat) : UInt64.ofNat i < n ↔ i < n.toNat := by
  rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by have := n.toNat_lt; omega)]

section loops
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- **A counted loop threading an accumulator**: `Inv i a` of the trips made,
    the accumulator and the world, kept by a trip. -/
theorem wp_forLoopAcc {limit acc0 : Slot .i64} {n a0 : UInt64}
    {body : Slot .i64 → Slot .i64 → Prog Slot Lvl (Slot .i64)} {Q : Slot .i64 → Env → World → Prop}
    (Inv : Nat → UInt64 → World → Prop)
    (hl : Γ[limit]? = some (.sc .i64 n)) (ha : Γ[acc0]? = some (.sc .i64 a0)) (h0 : Inv 0 a0 w)
    (hb : ∀ (J' : Post) (Γ' : Env) (wi : World) (i : Nat) (a : UInt64) (si sa : Slot .i64), Ext Γ Γ' →
      i < n.toNat → Inv i a wi → Γ'[si]? = some (.sc .i64 (UInt64.ofNat i)) → Γ'[sa]? = some (.sc .i64 a) →
      wp cfg (d + 1) J' (body si sa)
        (fun r Γ'' w'' => Ext Γ' Γ'' ∧ ∃ a', Γ''[r]? = some (.sc .i64 a') ∧ Inv (i + 1) a' w'') Γ' wi)
    (hk : ∀ Γ' w' r a, Ext Γ Γ' → Γ'[r]? = some (.sc .i64 a) → Inv n.toNat a w' → Q r Γ' w') :
    wp cfg d J (forLoopAcc limit acc0 body) Q Γ w := by
  unfold forLoopAcc wloop2 wloop wloopL
  rw [wp_bind]
  apply wp_iconst64
  have hl' : (Γ.push (.sc .i64 (c64 0)))[limit]? = some (.sc .i64 n) := get_push hl
  have ha' : (Γ.push (.sc .i64 (c64 0)))[acc0]? = some (.sc .i64 a0) := get_push ha
  have hc' : (Γ.push (.sc .i64 (c64 0)))[Γ.size]? = some (.sc .i64 (UInt64.ofNat 0)) := get_push_size
  have hext : Ext Γ (Γ.push (.sc .i64 (c64 0))) := (Ext.refl Γ).push _
  generalize Γ.push (.sc .i64 (c64 0)) = Γc at hl' ha' hc' hext ⊢
  rw [wp_bind, wp_loop]
  refine ⟨fun Γ0 cs w' => Γ0 = Γc ∧ ∃ i a, i ≤ n.toNat ∧ cs = [.sc .i64 (UInt64.ofNat i), .sc .i64 a] ∧ Inv i a w',
    fun Γ0 cs a Γ1 w1 => Γ0 = Γc ∧ Γ1 = bindAt Γc Γc.size cs ∧
      a = (contIfULt (ty := .i64) Γc.size limit, %[Γc.size + 1], ()) ∧
      ∃ i acc, i ≤ n.toNat ∧ cs = [.sc .i64 (UInt64.ofNat i), .sc .i64 acc] ∧ Inv i acc w1,
    fun Γ0 Γb vs w' => Γ0 = Γc ∧ Ext Γc Γb ∧ ∃ a, vs = [.sc .i64 a] ∧ Inv n.toNat a w', ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro cs hcs
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, hc', ha', Option.pure_def, Option.bind_eq_bind,
      Option.bind_some, Option.some.injEq] at hcs
    exact ⟨rfl, 0, a0, Nat.zero_le _, hcs.symm, h0⟩
  · rintro n0 Γ0 cs wh hn0 ⟨rfl, hI⟩
    subst hn0
    simp only [wp_pure]
    exact ⟨by first | rfl | trivial, by first | rfl | trivial, by first | rfl | trivial, hI⟩
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨rfl, rfl, rfl, i, acc, hi, rfl, hinv⟩ hev hx hvs
    have hev' : evalOp w1.mem (bindAt Γ0 Γ0.size [.sc .i64 (UInt64.ofNat i), .sc .i64 acc])
        (.icmp .ult Γ0.size limit) = some (boolV (cmpInt .ult .i64 (UInt64.ofNat i) n)) := by
      simp [evalOp, Sem.get, bindAt_head, get_bindAt (Nat.le_refl _) hl', zipIntCmp]
      rfl
    simp only [contIfULt, contIf] at hev hx hvs
    rw [hev'] at hev
    have hnlt : ¬ i < n.toNat := by
      intro hlt
      have : cmpInt .ult .i64 (UInt64.ofNat i) n = true := by simpa [cmpInt] using (ofNat_lt_iff hi).mpr hlt
      rw [this] at hev
      simp [boolV] at hev
      obtain ⟨rfl, rfl⟩ := hev
      simp at hx
    have hin : i = n.toNat := by omega
    subst hin
    refine ⟨rfl, ((Ext.refl _).bindAt (Nat.le_refl _) _).push _, acc, ?_, hinv⟩
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, get_push (bindAt_second _ _ _ _ _), Option.pure_def,
      Option.bind_eq_bind, Option.bind_some, Option.some.injEq] at hvs
    exact hvs.symm
  · rintro nb Γ0 cs a Γ1 t f wb hnb ⟨rfl, rfl, rfl, i, acc, hi, rfl, hinv⟩ hev hx
    have hev' : evalOp wb.mem (bindAt Γ0 Γ0.size [.sc .i64 (UInt64.ofNat i), .sc .i64 acc])
        (.icmp .ult Γ0.size limit) = some (boolV (cmpInt .ult .i64 (UInt64.ofNat i) n)) := by
      simp [evalOp, Sem.get, bindAt_head, get_bindAt (Nat.le_refl _) hl', zipIntCmp]
      rfl
    simp only [contIfULt, contIf] at hev hx
    rw [hev'] at hev
    have hlt : i < n.toNat := by
      rcases Nat.lt_or_ge i n.toNat with hlt | hge
      · exact hlt
      · have : cmpInt .ult .i64 (UInt64.ofNat i) n = false := by
          simp only [cmpInt]
          exact decide_eq_false fun h => absurd ((ofNat_lt_iff hi).mp h) (by omega)
        rw [this] at hev
        simp [boolV] at hev
        obtain ⟨rfl, rfl⟩ := hev
        simp at hx
    simp only [wp_bind, wp_pure, carriesFrom, Vals.head, Vals.snd, Vals.tail]
    have hsz : ((bindAt Γ0 Γ0.size [V.sc .i64 (UInt64.ofNat i), .sc .i64 acc]).push (.sc t f)).size ≤ nb := by
      rw [Array.size_push]; omega
    have hE : Ext Γ (bindAt ((bindAt Γ0 Γ0.size [V.sc .i64 (UInt64.ofNat i), .sc .i64 acc]).push (.sc t f)) nb
        [.sc .i64 (UInt64.ofNat i), .sc .i64 acc]) :=
      (((hext.bindAt (Nat.le_refl _) _).push _).bindAt hsz _)
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hb _ _ wb i acc nb (nb + 1) hE hlt hinv (bindAt_head _ _ _ _)
      (bindAt_second _ _ _ _ _))
    rintro r Γ'' w'' ⟨hE2, a', hr, hinv'⟩
    refine wp_iaddImm (hE2.get (bindAt_head _ _ _ _)) ?_
    intro nx hnx
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, get_push_size, get_push (get_push hr), Option.pure_def,
      Option.bind_eq_bind, Option.bind_some, Option.some.injEq] at hnx
    refine ⟨by first | rfl | trivial, i + 1, a', hlt, ?_, hinv'⟩
    rw [← hnx, UInt64.ofNat_add, show c64 1 = 1 from by decide]
    rfl
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, hEb, a, rfl, hinv⟩
    simp only [wp_pure, wp_ret, carriesFrom, Vals.head]
    exact hk _ _ _ _ ((hext.trans hEb).bindAt' (Nat.le_trans hext.size hle) _) (bindAt_head _ _ _ _) hinv
  · rintro Γ0 cs a Γ1 w1 ⟨rfl, rfl, rfl, i, acc, hi, rfl, hinv⟩ hev
    have hev' : evalOp w1.mem (bindAt Γ0 Γ0.size [.sc .i64 (UInt64.ofNat i), .sc .i64 acc])
        (.icmp .ult Γ0.size limit) = some (boolV (cmpInt .ult .i64 (UInt64.ofNat i) n)) := by
      simp [evalOp, Sem.get, bindAt_head, get_bindAt (Nat.le_refl _) hl', zipIntCmp]
      rfl
    simp only [contIfULt, contIf] at hev
    rw [hev'] at hev
    cases hev

/-- **A counted loop**: `Inv i` of the trips made and the world, kept by a trip. -/
theorem wp_forLoop {limit : Slot .i64} {n : UInt64} {body : Slot .i64 → Prog Slot Lvl Unit}
    {Q : Unit → Env → World → Prop} (Inv : Nat → World → Prop)
    (hl : Γ[limit]? = some (.sc .i64 n)) (h0 : Inv 0 w)
    (hb : ∀ (J' : Post) (Γ' : Env) (wi : World) (i : Nat) (si : Slot .i64), Ext Γ Γ' → i < n.toNat → Inv i wi →
      Γ'[si]? = some (.sc .i64 (UInt64.ofNat i)) →
      wp cfg (d + 1) J' (body si) (fun _ Γ'' w'' => Ext Γ' Γ'' ∧ Inv (i + 1) w'') Γ' wi)
    (hk : ∀ Γ' w', Ext Γ Γ' → Inv n.toNat w' → Q () Γ' w') :
    wp cfg d J (forLoop limit body) Q Γ w := by
  unfold forLoop wloop1 wloop wloopL
  rw [wp_bind]
  apply wp_iconst64
  have hl' : (Γ.push (.sc .i64 (c64 0)))[limit]? = some (.sc .i64 n) := get_push hl
  have hc' : (Γ.push (.sc .i64 (c64 0)))[Γ.size]? = some (.sc .i64 (UInt64.ofNat 0)) := get_push_size
  have hext : Ext Γ (Γ.push (.sc .i64 (c64 0))) := (Ext.refl Γ).push _
  generalize Γ.push (.sc .i64 (c64 0)) = Γc at hl' hc' hext ⊢
  rw [wp_bind, wp_loop]
  refine ⟨fun Γ0 cs w' => Γ0 = Γc ∧ ∃ i, i ≤ n.toNat ∧ cs = [.sc .i64 (UInt64.ofNat i)] ∧ Inv i w',
    fun Γ0 cs a Γ1 w1 => Γ0 = Γc ∧ Γ1 = bindAt Γc Γc.size cs ∧
      a = (contIfULt (ty := .i64) Γc.size limit, %[], ()) ∧
      ∃ i, i ≤ n.toNat ∧ cs = [.sc .i64 (UInt64.ofNat i)] ∧ Inv i w1,
    fun Γ0 Γb vs w' => Γ0 = Γc ∧ Ext Γc Γb ∧ vs = [] ∧ Inv n.toNat w', ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro cs hcs
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, hc', Option.pure_def, Option.bind_eq_bind,
      Option.bind_some, Option.some.injEq] at hcs
    exact ⟨rfl, 0, Nat.zero_le _, hcs.symm, h0⟩
  · rintro n0 Γ0 cs wh hn0 ⟨rfl, hI⟩
    subst hn0
    simp only [wp_pure]
    exact ⟨by first | rfl | trivial, by first | rfl | trivial, by first | rfl | trivial, hI⟩
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨rfl, rfl, rfl, i, hi, rfl, hinv⟩ hev hx hvs
    have hev' : evalOp w1.mem (bindAt Γ0 Γ0.size [.sc .i64 (UInt64.ofNat i)])
        (.icmp .ult Γ0.size limit) = some (boolV (cmpInt .ult .i64 (UInt64.ofNat i) n)) := by
      simp [evalOp, Sem.get, bindAt_head, get_bindAt (Nat.le_refl _) hl', zipIntCmp]
      rfl
    simp only [contIfULt, contIf] at hev hx hvs
    rw [hev'] at hev
    have hnlt : ¬ i < n.toNat := by
      intro hlt
      have : cmpInt .ult .i64 (UInt64.ofNat i) n = true := by simpa [cmpInt] using (ofNat_lt_iff hi).mpr hlt
      rw [this] at hev
      simp [boolV] at hev
      obtain ⟨rfl, rfl⟩ := hev
      simp at hx
    have hin : i = n.toNat := by omega
    subst hin
    refine ⟨rfl, ((Ext.refl _).bindAt (Nat.le_refl _) _).push _, ?_, hinv⟩
    simp only [Vals.slots, List.mapM_nil, Option.pure_def, Option.some.injEq] at hvs
    exact hvs.symm
  · rintro nb Γ0 cs a Γ1 t f wb hnb ⟨rfl, rfl, rfl, i, hi, rfl, hinv⟩ hev hx
    have hev' : evalOp wb.mem (bindAt Γ0 Γ0.size [.sc .i64 (UInt64.ofNat i)])
        (.icmp .ult Γ0.size limit) = some (boolV (cmpInt .ult .i64 (UInt64.ofNat i) n)) := by
      simp [evalOp, Sem.get, bindAt_head, get_bindAt (Nat.le_refl _) hl', zipIntCmp]
      rfl
    simp only [contIfULt, contIf] at hev hx
    rw [hev'] at hev
    have hlt : i < n.toNat := by
      rcases Nat.lt_or_ge i n.toNat with hlt | hge
      · exact hlt
      · have : cmpInt .ult .i64 (UInt64.ofNat i) n = false := by
          simp only [cmpInt]
          exact decide_eq_false fun h => absurd ((ofNat_lt_iff hi).mp h) (by omega)
        rw [this] at hev
        simp [boolV] at hev
        obtain ⟨rfl, rfl⟩ := hev
        simp at hx
    simp only [wp_bind, wp_pure, carriesFrom, Vals.head]
    have hsz : ((bindAt Γ0 Γ0.size [V.sc .i64 (UInt64.ofNat i)]).push (.sc t f)).size ≤ nb := by
      rw [Array.size_push]; omega
    have hE : Ext Γ (bindAt ((bindAt Γ0 Γ0.size [V.sc .i64 (UInt64.ofNat i)]).push (.sc t f)) nb
        [.sc .i64 (UInt64.ofNat i)]) :=
      (((hext.bindAt (Nat.le_refl _) _).push _).bindAt hsz _)
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hb _ _ wb i nb hE hlt hinv (bindAt_head _ _ _ _))
    rintro r Γ'' w'' ⟨hE2, hinv'⟩
    refine wp_iaddImm (hE2.get (bindAt_head _ _ _ _)) ?_
    intro nx hnx
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, get_push_size, Option.pure_def,
      Option.bind_eq_bind, Option.bind_some, Option.some.injEq] at hnx
    refine ⟨by first | rfl | trivial, i + 1, hlt, ?_, hinv'⟩
    rw [← hnx, UInt64.ofNat_add, show c64 1 = 1 from by decide]
    rfl
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, hEb, rfl, hinv⟩
    simp only [wp_pure, wp_ret]
    exact hk _ _ ((hext.trans hEb).bindAt' (Nat.le_trans hext.size hle) _) hinv
  · rintro Γ0 cs a Γ1 w1 ⟨rfl, rfl, rfl, i, hi, rfl, hinv⟩ hev
    have hev' : evalOp w1.mem (bindAt Γ0 Γ0.size [.sc .i64 (UInt64.ofNat i)])
        (.icmp .ult Γ0.size limit) = some (boolV (cmpInt .ult .i64 (UInt64.ofNat i) n)) := by
      simp [evalOp, Sem.get, bindAt_head, get_bindAt (Nat.le_refl _) hl', zipIntCmp]
      rfl
    simp only [contIfULt, contIf] at hev
    rw [hev'] at hev
    cases hev

end loops

section joins
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- A branch on two 64-bit slots whose arms each answer one 64-bit value
    `R` holds of, and what follows it. -/
theorem wp_ifteV {cc : ICmpCond} {a b : Slot .i64} {x y : UInt64}
    {thn els : Prog Slot Lvl (Vals Slot [.i64])} {Q : Vals Slot [.i64] → Env → World → Prop}
    (R : UInt64 → World → Prop)
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (ht : cmpInt cc .i64 x y = true →
      wp cfg d J thn (fun jv Γ' w' => Ext Γ Γ' ∧ ∃ v, Γ'[jv.head]? = some (.sc .i64 v) ∧ R v w')
        (Γ.push (boolV true)) w)
    (he : cmpInt cc .i64 x y = false → ∀ ne, (Γ.push (boolV false)).size ≤ ne →
      wp cfg d J els (fun jv Γ' w' => Ext Γ Γ' ∧ ∃ v, Γ'[jv.head]? = some (.sc .i64 v) ∧ R v w')
        (bindAt (Γ.push (boolV false)) ne []) w)
    (hk : ∀ Γ' w' v (nJ : Nat), Ext Γ Γ' → Γ'[nJ]? = some (.sc .i64 v) → R v w' →
      Q (carriesFrom nJ [.i64]) Γ' w') :
    wp cfg d J (ifte cc a b thn els) Q Γ w := by
  refine wp_ifte (fun jv Γ' w' => Ext Γ Γ' ∧ ∃ v, Γ'[jv.head]? = some (.sc .i64 v) ∧ R v w') ha hb ht he ?_
  rintro nJ Γ' jv vs w' hle hvs ⟨hE, v, hv, hr⟩
  have hj := join1 hvs hv
  subst hj
  exact hk _ _ v nJ (hE.bindAt hle _) (bindAt_head _ _ _ _) hr

/-- **A loop of one 64-bit carry**, left where its head says: `Inv` of the
    carry and the world at each trip's start, `Hd` what the head leaves with
    its condition and exit values, `X` of the exit values. -/
theorem wp_wloop1 {exitTys : List ClifTy} {β : Type} {init : Slot .i64} {x0 : UInt64}
    {head : Slot .i64 → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × β)}
    {body : Slot .i64 → β → Prog Slot Lvl (Vals Slot [.i64])}
    {Q : Vals Slot exitTys → Env → World → Prop}
    (Inv : UInt64 → World → Prop)
    (Hd : UInt64 → Cond Slot × Vals Slot exitTys × β → Env → World → Prop)
    (X : List V → World → Prop)
    (hi : Γ[init]? = some (.sc .i64 x0)) (h0 : Inv x0 w)
    (hh : ∀ (J' : Post) (Γ' : Env) (wi : World) (x : UInt64) (si : Slot .i64), Ext Γ Γ' → Inv x wi →
      Γ'[si]? = some (.sc .i64 x) →
      wp cfg (d + 1) J' (head si) (fun a Γ1 w1 => Ext Γ' Γ1 ∧ Hd x a Γ1 w1) Γ' wi)
    (hx : ∀ x a Γ1 w1 t f vs, Hd x a Γ1 w1 → evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) →
      ((f != 0) == a.1.exitOnTrue) = true →
      a.2.1.slots.mapM (fun r => (Γ1.push (.sc t f))[r]?) = some vs → X vs w1)
    (hb : ∀ (J' : Post) x a Γ1 wb t f Γ2 (si : Slot .i64), Hd x a Γ1 wb → Ext Γ Γ1 →
      evalOp wb.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) → ((f != 0) == a.1.exitOnTrue) = false →
      Ext Γ1 Γ2 → Γ2[si]? = some (.sc .i64 x) →
      wp cfg (d + 1) J' (body si a.2.2)
        (fun next Γ3 w3 => ∃ x', Γ3[next.head]? = some (.sc .i64 x') ∧ Inv x' w3) Γ2 wb)
    (hf : ∀ x a Γ1 w1, Hd x a Γ1 w1 → evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) ≠ none)
    (hk : ∀ Γb vs w' nE, Ext Γ Γb → Γ.size ≤ nE → X vs w' → Q (carriesFrom nE exitTys) (bindAt Γb nE vs) w') :
    wp cfg d J (wloop1 init head body) Q Γ w := by
  unfold wloop1 wloop wloopL
  rw [wp_loop]
  refine ⟨fun Γ0 cs w' => Γ0 = Γ ∧ ∃ x, cs = [.sc .i64 x] ∧ Inv x w',
    fun Γ0 cs a Γ1 w1 => Γ0 = Γ ∧ Ext Γ Γ1 ∧ ∃ x, cs = [.sc .i64 x] ∧ Hd x a Γ1 w1,
    fun Γ0 Γb vs w' => Γ0 = Γ ∧ Ext Γ Γb ∧ X vs w', ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro cs hcs
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, hi, Option.pure_def, Option.bind_eq_bind,
      Option.bind_some, Option.some.injEq] at hcs
    exact ⟨rfl, x0, hcs.symm, h0⟩
  · rintro n0 Γ0 cs wh hn0 ⟨rfl, x, rfl, hinv⟩
    subst hn0
    simp only [carriesFrom, Vals.head]
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hh _ _ wh x Γ0.size ((Ext.refl _).bindAt (Nat.le_refl _) _) hinv
      (bindAt_head _ _ _ _))
    rintro a Γ1 w1 ⟨hE, hd⟩
    exact ⟨by first | rfl | trivial, Ext.trans ((Ext.refl _).bindAt (Nat.le_refl _) _) hE, x,
      by first | rfl | trivial, hd⟩
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨rfl, hE, x, rfl, hd⟩ hev hxx hvs
    exact ⟨rfl, hE.push _, hx x a Γ1 w1 t f vs hd hev hxx hvs⟩
  · rintro nb Γ0 cs a Γ1 t f wb hnb ⟨rfl, hE, x, rfl, hd⟩ hev hxx
    subst hnb
    simp only [carriesFrom, Vals.head]
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hb _ x a Γ1 wb t f _ (Γ1.size + 1) hd hE hev hxx
      (((Ext.refl _).push _).bindAt (by rw [Array.size_push]; omega) _) (bindAt_head _ _ _ _))
    rintro next Γ3 w3 ⟨x', hx', hinv⟩ nx hnx
    have hj := join1 hnx hx'
    subst hj
    exact ⟨by first | rfl | trivial, x', by first | rfl | trivial, hinv⟩
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, hE, hX⟩
    simp only [wp_ret]
    exact hk _ _ _ _ hE hle hX
  · rintro Γ0 cs a Γ1 w1 ⟨rfl, hE, x, rfl, hd⟩ hev
    exact absurd hev (hf x a Γ1 w1 hd)

end joins

-- ---------------------------------------------------------------------------
-- Copies
-- ---------------------------------------------------------------------------

theorem byte_back (b : UInt8) : ((b.toUInt64 >>> (8 * UInt64.ofNat 0)) &&& 0xff).toUInt8 = b := by
  apply UInt8.toNat_inj.mp
  have := b.toNat_lt
  simp
  rw [show (255 : Nat) = 2 ^ 8 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt this]

theorem copyOut_bytes {w : World} {p : UInt64} {bs : List UInt8} (h : Bytes w p bs) :
    copyOut w.mem p bs.length = some (ByteArray.mk bs.toArray) := by
  unfold copyOut
  suffices ∀ k, k ≤ bs.length → (List.range k).foldlM (fun (acc : ByteArray) i => do
      let b ← w.mem.load (p + UInt64.ofNat i) 1
      pure (acc.push b.toUInt8)) ByteArray.empty = some (ByteArray.mk (bs.take k).toArray) by
    have := this bs.length (Nat.le_refl _)
    rwa [List.take_length] at this
  intro k
  induction k with
  | zero => intro _; rfl
  | succ k ih =>
    intro hk
    rw [List.range_succ, List.foldlM_append, ih (by omega)]
    simp only [Option.bind_eq_bind, Option.bind_some, List.foldlM_cons, List.foldlM_nil,
      h.load (j := k) (k := 1) (by omega) (by decide)]
    have hb : (leN bs k 1).toUInt8 = bs[k]'(by omega) := by
      have hg : bs[k]?.getD 0 = bs[k]'(by omega) := by
        rw [List.getElem?_eq_getElem (by omega)]; rfl
      simp [leN, List.range, List.range.loop, hg]
    rw [hb, List.take_succ_eq_append_getElem (by omega)]
    simp [ByteArray.push]

/-- Bytes with room that can be reached can be written. -/
theorem store_ok {m : Mem} {a v : UInt64} {n : Nat} {r : Region} {off : Nat} (hd : decodeAddr a = some (r, off))
    (hs : off + n ≤ (m.region r).size) (hr : m.reachable r off n = true) : ∃ m', m.store a n v = some m' := by
  unfold Mem.store
  simp only [hd, Option.bind_eq_bind, Option.bind_some]
  rw [if_neg (by simp [hr]; omega)]
  exact ⟨_, rfl⟩

theorem get!_ofList (bs : List UInt8) (i : Nat) : (ByteArray.mk bs.toArray).get! i = bs.getD i 0 := by
  simp [ByteArray.get!, List.getD_eq_getElem?_getD]
  rfl

/-- **A copy into memory with room that can be reached** happens, and leaves
    the bytes copied there and every other byte as it was. -/
theorem copyIn_spec {m : Mem} {a : UInt64} {r : Region} {off : Nat} {bs : List UInt8}
    (hd : decodeAddr a = some (r, off)) (hs : off + bs.length ≤ (m.region r).size) (h36 : off + bs.length ≤ 2 ^ 36)
    (hr : m.reachable r off bs.length = true) :
    ∃ m', copyIn m a (ByteArray.mk bs.toArray) = some m' ∧ StoreFrame m m' r off bs.length ∧
      ∀ i, i < bs.length → (m'.region r).get! (off + i) = bs.getD i 0 := by
  unfold copyIn
  rw [show (ByteArray.mk bs.toArray).size = bs.length from by simp [ByteArray.size]]
  suffices ∀ k, k ≤ bs.length → ∃ m', (List.range k).foldlM
      (fun mm i => mm.store (a + UInt64.ofNat i) 1 ((ByteArray.mk bs.toArray).get! i).toUInt64) m = some m' ∧
      StoreFrame m m' r off k ∧ ∀ i, i < k → (m'.region r).get! (off + i) = bs.getD i 0 from
    this _ (Nat.le_refl _)
  intro k
  induction k with
  | zero => exact fun _ => ⟨m, rfl, StoreFrame.refl m r off 0, fun i hi => absurd hi (Nat.not_lt_zero i)⟩
  | succ k ih =>
    intro hk
    obtain ⟨m1, h1, hF, hb⟩ := ih (by omega)
    have hdk := Sem.decodeAddr_add (i := k) hd (by rw [Sem.regionSpan_toNat]; omega)
    obtain ⟨m2, h2⟩ := store_ok (m := m1) (v := ((ByteArray.mk bs.toArray).get! k).toUInt64) (n := 1) hdk
      (by rw [← hF.same.size r]; omega)
      (by rw [← hF.same.reachable]; exact reachable_sub hr (by omega) (by omega))
    refine ⟨m2, ?_, ?_, ?_⟩
    · rw [List.range_succ, List.foldlM_append, h1]
      simp only [Option.bind_eq_bind, Option.bind_some, List.foldlM_cons, List.foldlM_nil, h2]
      rfl
    · obtain ⟨r2, off2, hd2, hf2⟩ := store_frame h2
      rw [hdk] at hd2; cases hd2
      exact hF.trans (hf2.widen (by omega) (by omega)) (by omega)
    · intro i hi
      obtain ⟨r2, off2, hd2, hn2, rfl⟩ := store_some h2
      rw [hdk] at hd2; cases hd2
      rw [region_setRegion_self, get!_setRun _ _ _ _ _ hn2]
      by_cases e : i = k
      · subst e; rw [if_pos (by omega), Nat.sub_self, byte_back, get!_ofList]
      · rw [if_neg (by omega)]; exact hb i (by omega)

theorem Wrote.stays {w w' : World} {d : UInt64} {n : Nat} (h : Wrote w w' d n) : Stays w w' :=
  ⟨fun c => ⟨h.same.frozen.symm.trans c.1, h.same.busy.symm.trans c.2⟩,
    fun _ _ b => b.same h.heap (Nat.le_of_eq (h.same.size .pinned)) h.same.live.symm,
    fun r _ => (h.same.size r).symm⟩

theorem Wrote.whole {w w' : World} {d : UInt64} {n : Nat} (h : Wrote w w' d n) {x : UInt64} {k : Nat}
    (hx : Whole w x k) : Whole w' x k := by
  obtain ⟨off, hw⟩ := hx.2
  refine ⟨h.stays.blk _ _ hx.1, off, ?_⟩
  unfold World.heapAt? at hw ⊢
  rw [h.heap]; exact hw

/-- A load clear of the bytes written reads what it read. -/
theorem Wrote.load {w w' : World} {d : UInt64} {n : Nat} (h : Wrote w w' d n) {b : UInt64} {k : Nat} {r' : Region}
    {off' : Nat} (hd' : decodeAddr b = some (r', off'))
    (hc : 0 < n → ∀ r off, decodeAddr d = some (r, off) → r' ≠ r ∨ off' + k ≤ off ∨ off + n ≤ off') :
    w'.mem.load b k = w.mem.load b k := by
  unfold Mem.load
  simp only [hd', Option.bind_eq_bind, Option.bind_some, ← h.same.size r', ← h.same.readable r' off' k]
  split
  · rfl
  · congr 1
    exact foldr_bytes_congr _ _ fun i hi => h.get _ _ fun hn r off hd => by
      have := List.mem_range.mp hi
      rcases hc hn r off hd with hc | hc | hc
      · exact Or.inl hc
      · exact Or.inr (Or.inl (by omega))
      · exact Or.inr (Or.inr (by omega))

/-- Bytes clear of the ones written are the bytes they were. -/
theorem Wrote.bytes {w w' : World} {d : UInt64} {n : Nat} (h : Wrote w w' d n) {p : UInt64} {bs : List UInt8}
    (hb : Bytes w p bs)
    (hc : 0 < n → ∀ r off r' off', decodeAddr d = some (r, off) → decodeAddr p = some (r', off') →
      r' ≠ r ∨ off' + bs.length ≤ off ∨ off + n ≤ off') : Bytes w' p bs := by
  rcases hb with rfl | ⟨r', off', hd', hs, h36, hr, hg⟩
  · exact Or.inl rfl
  refine Or.inr ⟨r', off', hd', by rw [← h.same.size r']; exact hs, h36, by rw [← h.same.readable]; exact hr,
    fun i hi => ?_⟩
  rw [h.get _ _ fun hn r off hd => by
    rcases hc hn r off r' off' hd hd' with hc | hc | hc
    · exact Or.inl hc
    · exact Or.inr (Or.inl (by omega))
    · exact Or.inr (Or.inr (by omega))]
  exact hg i hi

/-- A store is a write of its eight bytes. -/
theorem wrote_stored {w : World} {p x : UInt64} {m : Mem} (hs : w.mem.store p 8 x = some m) :
    Wrote w (stored w p x m) p 8 :=
  Wrote.ofStore (w' := stored w p x m) hs rfl

theorem Wrote.room {w w' : World} {d : UInt64} {n : Nat} (h : Wrote w w' d n) {p : UInt64} {k : Nat}
    (hr : Room w p k) : Room w' p k :=
  hr.imp id fun ⟨r, off, hd, hs, h36, hc⟩ =>
    ⟨r, off, hd, by rw [← h.same.size r]; exact hs, h36, by rw [← h.same.reachable]; exact hc⟩

/-- **A copy into room** happens, and what it leaves is a write of that run
    with the bytes there. -/
theorem copyIn_room {w : World} {p : UInt64} {bs : List UInt8} (h : Room w p bs.length) :
    ∃ m', copyIn w.mem p (ByteArray.mk bs.toArray) = some m' ∧
      ∀ w', w'.mem = m' → w'.heap = w.heap → Wrote w w' p bs.length ∧ Bytes w' p bs := by
  rcases h with h0 | ⟨r, off, hd, hs, h36, hr⟩
  · have hb : bs = [] := List.length_eq_zero_iff.mp h0
    subst hb
    exact ⟨w.mem, rfl, fun w' hm hh => ⟨⟨hm ▸ MemSame.refl _, hh, fun r' j _ => by rw [hm]⟩, Or.inl rfl⟩⟩
  · obtain ⟨m', hci, hF, hg⟩ := copyIn_spec hd hs h36 hr
    refine ⟨m', hci, fun w' hm hh => ?_⟩
    subst hm
    refine ⟨⟨hF.same, hh, fun r' j hc => ?_⟩, Or.inr ⟨r, off, hd, ?_, h36, ?_, hg⟩⟩
    · by_cases e : r' = r
      · subst e
        by_cases hk : 0 < bs.length
        · rcases hc hk r' off hd with hc | hc | hc
          · exact absurd rfl hc
          · exact hF.outside j (Or.inl hc)
          · exact hF.outside j (Or.inr (by omega))
        · exact hF.outside j (by omega)
      · exact congrArg (·.get! j) (hF.other r' e)
    · rw [← hF.same.size r]; exact hs
    · rw [← hF.same.readable]; exact readable_of_reachable hr

/-- The address a decoded address is. -/
theorem toNat_of_decode {a : UInt64} {r : Region} {off : Nat} (h : decodeAddr a = some (r, off)) :
    a.toNat = (regionBase r).toNat + off := by
  obtain ⟨rfl, ho⟩ := Sem.decodeAddr_some h
  rw [Sem.regionSpan_toNat] at ho
  rw [UInt64.toNat_add, UInt64.toNat_ofNat']
  cases r <;> simp [regionBase] <;> omega

/-- Runs in different regions, or apart in one, are apart as numbers. -/
theorem apart_of_decode {a b : UInt64} {r r' : Region} {off off' n : Nat} (ha : decodeAddr a = some (r, off))
    (hb : decodeAddr b = some (r', off')) (hn : off + n ≤ 2 ^ 36) (hn' : off' + n ≤ 2 ^ 36)
    (hc : r ≠ r' ∨ off + n ≤ off' ∨ off' + n ≤ off) : a.toNat + n ≤ b.toNat ∨ b.toNat + n ≤ a.toNat := by
  rw [toNat_of_decode ha, toNat_of_decode hb]
  cases r <;> cases r' <;> simp [regionBase] at hc ⊢ <;> omega

/-- A run inside a block has room. -/
theorem Blk.room {w : World} {x : UInt64} {n : Nat} (h : Blk w x n) (hc : Calm w) {j k : Nat} (hjk : j + k ≤ n) :
    Room w (x + UInt64.ofNat j) k := by
  by_cases hk : k = 0
  · exact Or.inl hk
  · obtain ⟨off, m, hat, hn, hm, hsz, h36, hlv⟩ := h
    obtain ⟨hd, -⟩ := heapAt_decode hat
    refine Or.inr ⟨.pinned, off + j, Sem.decodeAddr_add hd (by rw [Sem.regionSpan_toNat]; omega),
      by show off + j + k ≤ w.mem.pinned.size; omega, by omega, ?_⟩
    have hany : (w.mem.pinnedLive.any fun x => decide (x.1 ≤ off + j) && decide (off + j + k ≤ x.1 + x.2)) = true :=
      List.any_eq_true.mpr ⟨(off, m), hlv, by simp; omega⟩
    simp [Mem.reachable, hc.1, hc.2, hany]

/-- A field of a block clear of the bytes written is what it was. -/
theorem Wrote.fld_blk {w w' : World} {p : UInt64} {n : Nat} (h : Wrote w w' p n) {y : UInt64} {k j : Nat}
    (hb : Blk w y k) (hj : j + 8 ≤ k)
    (hc : ∀ r off oy, decodeAddr p = some (r, off) → decodeAddr y = some (.pinned, oy) →
      r ≠ .pinned ∨ off + n ≤ oy ∨ oy + k ≤ off) : fld w' y j = fld w y j := by
  obtain ⟨oy, hd, hdj⟩ := Blk.decode hb
  exact h.load (hdj j (by omega)) fun _ r off hdp => by
    rcases hc r off oy hdp hd with c | c | c
    · exact Or.inl (Ne.symm c)
    · exact Or.inr (Or.inr (by omega))
    · exact Or.inr (Or.inl (by omega))

/-- A run inside a block clear of the bytes written is what it was. -/
theorem Wrote.bytes_blk {w w' : World} {p : UInt64} {n : Nat} (h : Wrote w w' p n) {y : UInt64} {k j : Nat}
    {bs : List UInt8} (hb : Blk w y k) (hj : j + bs.length ≤ k) (hbs : Bytes w (y + UInt64.ofNat j) bs)
    (hc : ∀ r off oy, decodeAddr p = some (r, off) → decodeAddr y = some (.pinned, oy) →
      r ≠ .pinned ∨ off + n ≤ oy ∨ oy + k ≤ off) : Bytes w' (y + UInt64.ofNat j) bs := by
  by_cases hl : bs.length = 0
  · exact Or.inl (List.length_eq_zero_iff.mp hl)
  · obtain ⟨oy, hd, hdj⟩ := Blk.decode hb
    refine h.bytes hbs fun _ r off r' off' hdp hdy => ?_
    rw [hdj j (by omega)] at hdy
    cases hdy
    rcases hc r off oy hdp hd with c | c | c
    · exact Or.inl (Ne.symm c)
    · exact Or.inr (Or.inr (by omega))
    · exact Or.inr (Or.inl (by omega))

/-- Room clear of a block is apart, as numbers, from every run of its length
    inside the block. -/
theorem apart_blk {w : World} {p y : UInt64} {n k j : Nat} (hr : Room w p n) (hb : Blk w y k) (hj : j + n ≤ k)
    (hc : ∀ r off oy, decodeAddr p = some (r, off) → decodeAddr y = some (.pinned, oy) →
      r ≠ .pinned ∨ off + n ≤ oy ∨ oy + k ≤ off) :
    p.toNat + n ≤ (y + UInt64.ofNat j).toNat ∨ (y + UInt64.ofNat j).toNat + n ≤ p.toNat := by
  rcases hr with h0 | ⟨r, off, hd, hs, h36, -⟩
  · omega
  · obtain ⟨oy, m, hat, hn, hm, hsz, h36', hlv⟩ := hb
    obtain ⟨hdy, -⟩ := heapAt_decode hat
    by_cases hn0 : n = 0
    · omega
    · refine apart_of_decode hd (Sem.decodeAddr_add (i := j) hdy (by rw [Sem.regionSpan_toNat]; omega)) h36
        (by omega) ?_
      rcases hc r off oy hd hdy with c | c | c
      · exact Or.inl c
      · exact Or.inr (Or.inl (by omega))
      · exact Or.inr (Or.inr (by omega))

theorem Apart.symm {x y : UInt64} {n k : Nat} (h : Apart x n y k) : Apart y k x n := Or.symm h

theorem Apart.ne {x y : UInt64} {n k : Nat} (h : Apart x n y k) (hn : 0 < n) (hk : 0 < k) : x ≠ y := by
  rintro rfl
  rcases h with ⟨a, b, ha, hb, hab⟩ | ⟨a, b, ha, hb, hab⟩ <;>
  · rw [ha] at hb; cases hb; omega

/-- Where an address inside or at the end of a block decodes to, if it does. -/
theorem decode_in_blk {d : UInt64} {n j : Nat} {od : Nat} (hd : decodeAddr d = some (.pinned, od))
    (h36 : od + n ≤ 2 ^ 36) (hj : j ≤ n) {r : Region} {off : Nat}
    (h : decodeAddr (d + UInt64.ofNat j) = some (r, off)) : r = .pinned ∧ off = od + j := by
  by_cases hlt : od + j < 2 ^ 36
  · rw [Sem.decodeAddr_add hd (by rw [Sem.regionSpan_toNat]; exact hlt)] at h
    cases h; exact ⟨rfl, rfl⟩
  · exfalso
    have h1 := toNat_of_decode h
    have h2 := toNat_of_decode hd
    obtain ⟨-, ho⟩ := Sem.decodeAddr_some h
    rw [Sem.regionSpan_toNat] at ho
    rw [UInt64.toNat_add, UInt64.toNat_ofNat'] at h1
    cases r <;> simp [regionBase] at h1 h2 <;> omega

/-- A run inside one block is clear of a block apart from it. -/
theorem Apart.clear {w : World} {d y : UInt64} {n' k j n : Nat} (hd : Blk w d n') (ha : Apart d n' y k)
    (hj : j + n ≤ n') : ∀ r off oy, decodeAddr (d + UInt64.ofNat j) = some (r, off) →
      decodeAddr y = some (.pinned, oy) → r ≠ .pinned ∨ off + n ≤ oy ∨ oy + k ≤ off := by
  intro r off oy h1 h2
  obtain ⟨od, m, hat, hn, hm, hsz, h36, hlv⟩ := hd
  obtain ⟨hdd, -⟩ := heapAt_decode hat
  obtain ⟨-, rfl⟩ := decode_in_blk hdd (n := n') (by omega) (by omega) h1
  rcases ha with ⟨a, b, ha, hb, hab⟩ | ⟨a, b, ha, hb, hab⟩
  · rw [hdd] at ha; cases ha
    rw [h2] at hb; cases hb
    exact Or.inr (Or.inl (by omega))
  · rw [h2] at ha; cases ha
    rw [hdd] at hb; cases hb
    exact Or.inr (Or.inr (by omega))

/-- A field written in one block leaves the fields of a block apart from it. -/
theorem fld_other {w : World} {s y x : UInt64} {n k j i : Nat} {m : Mem} (hs : Blk w s n) (hy : Blk w y k)
    (ha : Apart s n y k) (hj : j + 8 ≤ n) (hi : i + 8 ≤ k) (hm : w.mem.store (s + UInt64.ofNat j) 8 x = some m) :
    fld (stored w (s + UInt64.ofNat j) x m) y i = fld w y i :=
  (wrote_stored hm).fld_blk hy hi (ha.clear hs hj)

/-- …and its runs of bytes. -/
theorem bytes_other {w : World} {s y x : UInt64} {n k j i : Nat} {m : Mem} {bs : List UInt8} (hs : Blk w s n)
    (hy : Blk w y k) (ha : Apart s n y k) (hj : j + 8 ≤ n) (hi : i + bs.length ≤ k)
    (hb : Bytes w (y + UInt64.ofNat i) bs) (hm : w.mem.store (s + UInt64.ofNat j) 8 x = some m) :
    Bytes (stored w (s + UInt64.ofNat j) x m) (y + UInt64.ofNat i) bs :=
  (wrote_stored hm).bytes_blk hy hi hb (ha.clear hs hj)

section copies
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- **`memcpy`** of the bytes at the source into room, apart from them: memory
    is what the copy leaves, with the bytes at the destination and nothing
    else rewritten, and the answer is the destination. -/
theorem wp_memcpy {a s n : Slot .i64} {pd ps k : UInt64} {bs : List UInt8} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 pd)) (hs : Γ[s]? = some (.sc .i64 ps)) (hn : Γ[n]? = some (.sc .i64 k))
    (hsrc : Bytes w ps bs) (hlen : bs.length = k.toNat) (hdst : Room w pd k.toNat)
    (hapart : pd.toNat + k.toNat ≤ ps.toNat ∨ ps.toNat + k.toNat ≤ pd.toNat)
    (h : ∀ w', Wrote w w' pd k.toNat → Bytes w' pd bs → copyIn w.mem pd (ByteArray.mk bs.toArray) = some w'.mem →
      Q Γ.size (Γ.push (.sc .i64 pd)) w') :
    wp cfg d J (Lib.memcpy a s n) Q Γ w := by
  unfold Lib.memcpy
  simp only [Prog.ext, Prog.callLocal, wp_callLocal, wp_ret]
  intro vs hvs
  have hvs' : vs = [.sc .i64 pd, .sc .i64 ps, .sc .i64 k] := by
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, ha, hs, hn, Option.pure_def, Option.bind_eq_bind,
      Option.bind_some, Option.some.injEq] at hvs
    exact hvs.symm
  subst hvs'
  simp only [callOf, extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true, Bool.false_eq_true,
    if_false, List.mapM_cons, List.mapM_nil, asBits, Option.pure_def, Option.bind_eq_bind, Option.bind_some, cCall]
  have hco : copyOut w.mem ps k.toNat = some (ByteArray.mk bs.toArray) := hlen ▸ copyOut_bytes hsrc
  obtain ⟨m', hci, hW⟩ := copyIn_room (w := w) (p := pd) (bs := bs) (by rw [hlen]; exact hdst)
  rw [if_neg (by simp; omega)]
  erw [hco]
  simp only [Option.bind_some]
  erw [hci]
  simp only [Option.bind_some]
  refine ⟨nofun, ?_⟩
  rintro x w' hxw v hv
  cases hxw
  cases hv
  obtain ⟨hw, hb⟩ := hW { obsCall w (.ext (.c .memcpy)) [.sc .i64 pd, .sc .i64 ps, .sc .i64 k] with mem := m' } rfl rfl
  exact h _ (by rw [← hlen]; exact hw) hb hci

/-- `-1` where the comparison holds, what `k` answers where it does not. -/
theorem wp_failIf {cc : ICmpCond} {a b : Slot .i64} {x y : UInt64} {k : Prog Slot Lvl (Slot .i64)}
    {R : UInt64 → World → Prop} (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (ht : cmpInt cc .i64 x y = true → R (c64 (-1)) w)
    (he : cmpInt cc .i64 x y = false → ∀ Γ', Ext Γ Γ' → wp cfg d J k (AnsQ R) Γ' w) :
    wp cfg d J (Lib.failIf cc a b k) (AnsQ R) Γ w := by
  unfold Lib.failIf
  wp_seq
  refine wp_ifte (AnsJ R) ha hb ?_ ?_ (by join_ans)
  · intro h
    wp_seq
    apply wp_iconst64
    exact ⟨_, get_push_size, ht h⟩
  · intro h ne hne
    wp_seq
    exact he h _ (by ext_env)

end copies

-- ---------------------------------------------------------------------------
-- Words of bytes
-- ---------------------------------------------------------------------------

section words
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

theorem wp_ishl {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x <<< (y % 64)))) w) : wp cfg d J (ishl a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, shiftBin, Sem.get, ha, hb, norm64] <;> op_fin) h

theorem wp_uextend8 {a : Slot .i8} {p : Bool} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (boolV p))
    (h : Q Γ.size (Γ.push (.sc .i64 (if p then 1 else 0))) w) : wp cfg d J (uextend64 a) Q Γ w :=
  wp_val (by cases p <;> simp [Op'.erase, evalOp, un, Sem.get, ha, boolV] <;> op_fin) h

end words

/-- `k` bytes given by `f`, little-endian. -/
def leV (f : Nat → UInt8) (k : Nat) : UInt64 :=
  (List.range k).foldr (fun i acc => (acc <<< 8) ||| (f i).toUInt64) 0

theorem leV_succ (f : Nat → UInt8) (k : Nat) :
    leV f (k + 1) = (leV (fun i => f (i + 1)) k <<< 8) ||| (f 0).toUInt64 := by
  unfold leV
  rw [List.range_succ_eq_map, List.foldr_cons, List.foldr_map]

theorem leV_lt : ∀ (k : Nat) (f : Nat → UInt8), (leV f k).toNat < 256 ^ k
  | 0, f => by simp [leV]
  | k + 1, f => by
    rw [leV_succ, shl8_or, Nat.pow_succ]
    have := leV_lt k (fun i => f (i + 1))
    have := (f 0).toNat_lt
    omega

/-- Up to eight bytes are told apart by the word they make. -/
theorem leV_inj : ∀ (k : Nat) (f g : Nat → UInt8), k ≤ 8 → (leV f k = leV g k ↔ ∀ t, t < k → f t = g t)
  | 0, f, g, _ => by simp [leV]
  | k + 1, f, g, hk => by
    have ih := leV_inj k (fun i => f (i + 1)) (fun i => g (i + 1)) (by omega)
    have hp : (256 : Nat) ^ k ≤ 2 ^ 56 :=
      Nat.le_trans (Nat.pow_le_pow_right (by decide) (show k ≤ 7 by omega)) (by decide)
    have lf := leV_lt k (fun i => f (i + 1))
    have lg := leV_lt k (fun i => g (i + 1))
    rw [leV_succ, leV_succ]
    constructor
    · intro h
      have hn := congrArg UInt64.toNat h
      rw [shl8_or, shl8_or] at hn
      have bf := (f 0).toNat_lt
      have bg := (g 0).toNat_lt
      have h1 : (leV (fun i => f (i + 1)) k).toNat = (leV (fun i => g (i + 1)) k).toNat := by omega
      have h2 : (f 0).toNat = (g 0).toNat := by omega
      intro t ht
      cases t with
      | zero => exact UInt8.toNat_inj.mp h2
      | succ t => exact (ih.mp (UInt64.toNat_inj.mp h1)) t (by omega)
    · intro h
      rw [ih.mpr (fun t ht => h (t + 1) (by omega)), h 0 (by omega)]

theorem leN_eq_iff {as bs : List UInt8} {j k : Nat} (hk : k ≤ 8) :
    leN as j k = leN bs j k ↔ ∀ t, t < k → as.getD (j + t) 0 = bs.getD (j + t) 0 :=
  leV_inj k (fun i => as.getD (j + i) 0) (fun i => bs.getD (j + i) 0) hk

theorem list_eq_iff_getD {as bs : List UInt8} (hl : bs.length = as.length) :
    as = bs ↔ ∀ t, t < as.length → as.getD t 0 = bs.getD t 0 := by
  constructor
  · rintro rfl t _; rfl
  · intro h
    refine List.ext_getElem hl.symm fun i h1 h2 => ?_
    have := h i h1
    simpa [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h1, List.getElem?_eq_getElem h2] using this

/-- A shift right by three is a division by eight. -/
theorem shr3 {n : Nat} (hn : n < 2 ^ 64) : UInt64.ofNat n >>> (c64 3 % 64) = UInt64.ofNat (n / 8) := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_shiftRight, show (c64 3 % 64).toNat % 64 = 3 from by decide, Nat.shiftRight_eq_div_pow,
    UInt64.toNat_ofNat', UInt64.toNat_ofNat', Nat.mod_eq_of_lt hn, Nat.mod_eq_of_lt (by omega)]

/-- A shift left by three is eight times. -/
theorem shl3 {n : Nat} (hn : n < 2 ^ 60) : UInt64.ofNat n <<< (c64 3 % 64) = UInt64.ofNat (8 * n) := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_shiftLeft, show (c64 3 % 64).toNat % 64 = 3 from by decide, Nat.shiftLeft_eq,
    UInt64.toNat_ofNat', UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  omega

theorem ofNat_sub_le {a b : Nat} (ha : a < 2 ^ 64) (hb : b ≤ a) :
    UInt64.ofNat a - UInt64.ofNat b = UInt64.ofNat (a - b) := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr (by
    rw [UInt64.toNat_ofNat', UInt64.toNat_ofNat', Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt (by omega)]; exact hb)),
    UInt64.toNat_ofNat', UInt64.toNat_ofNat', UInt64.toNat_ofNat', Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt (by omega),
    Nat.mod_eq_of_lt (by omega)]

theorem Bytes.len_le {w : World} {p : UInt64} {bs : List UInt8} (h : Bytes w p bs) : bs.length ≤ 2 ^ 36 := by
  rcases h with rfl | ⟨r, off, -, -, h36, -⟩
  · simp
  · omega



/-- A function with a statement neither misuses a call nor faults. -/
theorem Ans.sound {cfg : Cfg} {args : List V} {w : World} {p : StatusBody} {params : List ClifTy}
    {Q : UInt64 → World → Prop} (h : Ans cfg args w p params Q) (m : String) :
    Sem.run cfg args w (runAns p params).2.1 ≠ .misuse m ∧ Sem.run cfg args w (runAns p params).2.1 ≠ .fault m :=
  run_sound h rfl m

theorem Does.sound {cfg : Cfg} {args : List V} {w : World} {p : Body} {params : List ClifTy}
    {Q : World → Prop} (h : Does cfg args w p params Q) (m : String) :
    Sem.run cfg args w (emit p params) ≠ .misuse m ∧ Sem.run cfg args w (emit p params) ≠ .fault m :=
  run_sound h rfl m

-- ---------------------------------------------------------------------------
-- What a step keeps
-- ---------------------------------------------------------------------------

/-- A run lies inside `w`'s memory, where it decodes. -/
def InW (w : World) (q : UInt64) (n : Nat) : Prop :=
  ∀ r off, decodeAddr q = some (r, off) → off + n ≤ (w.mem.region r).size

/-- What is inside memory stays inside it: memory does not shrink. -/
theorem InW.grows {w w' : World} {q : UInt64} {n : Nat} (g : Grows w w') (h : InW w q n) : InW w' q n := by
  intro r off hd
  have := h r off hd
  by_cases e : r = .pinned
  · subst e; exact Nat.le_trans this g.psize
  · rw [g.size r e]; exact this

theorem InW.freed {w w' : World} {q x : UInt64} {n : Nat} (f : Freed w w' x) (h : InW w q n) : InW w' q n := by
  intro r off hd
  have := h r off hd
  by_cases e : r = .pinned
  · subst e; rw [f.psize]; exact this
  · rw [f.size r e]; exact this

theorem InW.wrote {w w' : World} {q d : UInt64} {n m : Nat} (W : Wrote w w' d m) (h : InW w q n) : InW w' q n := by
  intro r off hd
  rw [← W.same.size r]
  exact h r off hd

/-- **`w'` keeps what `P` protects of `w`**: its runs of bytes, its room and
    its loads. -/
structure Keeps (w w' : World) (P : UInt64 → Nat → Prop) : Prop where
  bytes : ∀ q cs, cs ≠ [] → P q cs.length → Bytes w q cs → Bytes w' q cs
  room : ∀ q k, 0 < k → P q k → Room w q k → Room w' q k
  load : ∀ q k v, 0 < k → P q k → w.mem.load q k = some v → w'.mem.load q k = some v

theorem Keeps.refl (w : World) (P : UInt64 → Nat → Prop) : Keeps w w P :=
  ⟨fun _ _ _ _ h => h, fun _ _ _ _ h => h, fun _ _ _ _ _ h => h⟩

theorem Keeps.trans {w w' w'' : World} {P : UInt64 → Nat → Prop} (h : Keeps w w' P) (h' : Keeps w' w'' P) :
    Keeps w w'' P :=
  ⟨fun q cs hn hp hb => h'.bytes q cs hn hp (h.bytes q cs hn hp hb),
    fun q k hk hp hr => h'.room q k hk hp (h.room q k hk hp hr),
    fun q k v hk hp hl => h'.load q k v hk hp (h.load q k v hk hp hl)⟩

theorem Keeps.mono {w w' : World} {P P' : UInt64 → Nat → Prop} (h : Keeps w w' P) (hP : ∀ q n, P' q n → P q n) :
    Keeps w w' P' :=
  ⟨fun q cs hn hp hb => h.bytes q cs hn (hP _ _ hp) hb, fun q k hk hp hr => h.room q k hk (hP _ _ hp) hr,
    fun q k v hk hp hl => h.load q k v hk (hP _ _ hp) hl⟩

theorem Grows.keeps {w w' : World} (g : Grows w w') (P : UInt64 → Nat → Prop) : Keeps w w' P :=
  ⟨fun _ _ _ _ hb => g.bytes _ _ hb, fun _ _ _ _ hr => g.room _ _ hr, fun _ _ _ _ _ hl => g.load _ _ _ hl⟩

/-- A write keeps what is clear of the bytes written. -/
theorem Wrote.keeps {w w' : World} {d : UInt64} {m : Nat} (h : Wrote w w' d m) (P : UInt64 → Nat → Prop)
    (hP : ∀ q n, 0 < n → P q n → ∀ r off r' off', decodeAddr d = some (r, off) → decodeAddr q = some (r', off') →
      r' ≠ r ∨ off' + n ≤ off ∨ off + m ≤ off') : Keeps w w' P := by
  refine ⟨fun q cs hne hp hb => h.bytes hb fun _ r off r' off' hd hq =>
      hP q cs.length (List.length_pos_iff.mpr hne) hp r off r' off' hd hq,
    fun q k _ _ hr => h.room hr, fun q k v hk hp hl => ?_⟩
  cases hd : decodeAddr q with
  | none => simp [Mem.load, hd] at hl
  | some a =>
    obtain ⟨r', off'⟩ := a
    exact (h.load hd fun _ r off hdd => hP q k hk hp r off r' off' hdd hd).trans hl

/-- A free keeps what is off the whole block freed. -/
theorem Freed.keeps {w w' : World} {x : UInt64} {n : Nat} (f : Freed w w' x) (hw : Whole w x n)
    (P : UInt64 → Nat → Prop) (hP : ∀ q k, P q k → Off x n q k) : Keeps w w' P :=
  ⟨fun q cs _ hp hb => f.bytes n hw q cs hb (hP _ _ hp), fun q k _ hp hr => f.room n hw q k hr (hP _ _ hp),
    fun q k v hk hp hl => f.load n hw q k v hk (hP _ _ hp) hl⟩

/-- A run off a block is clear of a write inside the block. -/
theorem Off.wrote {w : World} {y q : UInt64} {k j m n : Nat} (hy : Blk w y k) (hj : j + m ≤ k) (ho : Off y k q n) :
    ∀ r off r' off', decodeAddr (y + UInt64.ofNat j) = some (r, off) → decodeAddr q = some (r', off') →
      r' ≠ r ∨ off' + n ≤ off ∨ off + m ≤ off' := by
  intro r off r' off' hd hq
  obtain ⟨oy, m', hat, hn, hm, hsz, h36, hlv⟩ := hy
  obtain ⟨hdd, -⟩ := heapAt_decode hat
  obtain ⟨rfl, rfl⟩ := decode_in_blk hdd (n := k) (by omega) (by omega) hd
  rcases ho r' off' oy hq hdd with c | c | c
  · exact Or.inl c
  · exact Or.inr (Or.inl (by omega))
  · exact Or.inr (Or.inr (by omega))

/-- A run inside the memory there was is clear of a write inside a fresh block. -/
theorem Fresh.wrote {w0 w1 : World} {x q : UInt64} {nx j m n : Nat} (f : Fresh w0 w1 x nx) (hj : j + m ≤ nx)
    (hi : InW w0 q n) : ∀ r off r' off', decodeAddr (x + UInt64.ofNat j) = some (r, off) →
      decodeAddr q = some (r', off') → r' ≠ r ∨ off' + n ≤ off ∨ off + m ≤ off' := by
  intro r off r' off' hd hq
  obtain ⟨ox, hdx, hpast⟩ := f.past
  obtain ⟨ox', m', hat, hn, hm, hsz, h36, hlv⟩ := f.blk
  obtain ⟨hdd, -⟩ := heapAt_decode hat
  rw [hdx] at hdd
  cases hdd
  obtain ⟨rfl, rfl⟩ := decode_in_blk hdx (n := nx) (by omega) (by omega) hd
  have := hi r' off' hq
  by_cases e : r' = .pinned
  · subst e
    exact Or.inr (Or.inl (by omega))
  · exact Or.inl e

/-- A run off some fields of a block is clear of a write inside them. -/
theorem Off.sub {w : World} {s q : UInt64} {n a c j m len : Nat} (hs : Blk w s n) (h : Off (s + UInt64.ofNat a) c q len)
    (h1 : a ≤ j) (h2 : j + m ≤ a + c) (h3 : a + c ≤ n) (ha : a < n) :
    ∀ r off r' off', decodeAddr (s + UInt64.ofNat j) = some (r, off) → decodeAddr q = some (r', off') →
      r' ≠ r ∨ off' + len ≤ off ∨ off + m ≤ off' := by
  intro r off r' off' hd hq
  obtain ⟨os, hds, hdj⟩ := Blk.decode hs
  obtain ⟨os', m', hat, hn, hm, hsz, h36, hlv⟩ := hs
  obtain ⟨hdd, -⟩ := heapAt_decode hat
  rw [hds] at hdd
  cases hdd
  obtain ⟨rfl, rfl⟩ := decode_in_blk hds (n := n) (by omega) (by omega) hd
  rcases h r' off' (os + a) hq (hdj a ha) with c | c | c
  · exact Or.inl c
  · exact Or.inr (Or.inl (by omega))
  · exact Or.inr (Or.inr (by omega))

/-- Fields of one block are off other fields of it. -/
theorem off_same {w : World} {s : UInt64} {n a c j m : Nat} (hs : Blk w s n) (ha : a < n) (hj : j + m ≤ n)
    (h : j + m ≤ a ∨ a + c ≤ j) : Off (s + UInt64.ofNat a) c (s + UInt64.ofNat j) m := by
  intro r off oy hd hy
  obtain ⟨os, hds, hdj⟩ := Blk.decode hs
  obtain ⟨os', m', hat, hn, hm, hsz, h36, hlv⟩ := hs
  obtain ⟨hdd, -⟩ := heapAt_decode hat
  rw [hds] at hdd
  cases hdd
  rw [hdj a ha] at hy
  cases hy
  obtain ⟨rfl, rfl⟩ := decode_in_blk hds (n := n) (by omega) (by omega) hd
  exact Or.inr (by omega)

theorem Blk.inW {w : World} {y : UInt64} {k j m : Nat} (hy : Blk w y k) (hj : j + m ≤ k) :
    InW w (y + UInt64.ofNat j) m := by
  intro r off hd
  obtain ⟨oy, m', hat, hn, hm, hsz, h36, hlv⟩ := hy
  obtain ⟨hdd, -⟩ := heapAt_decode hat
  obtain ⟨rfl, rfl⟩ := decode_in_blk hdd (n := k) (by omega) (by omega) hd
  show oy + j + m ≤ w.mem.pinned.size
  omega

theorem Bytes.inW {w : World} {q : UInt64} {cs : List UInt8} (hb : Bytes w q cs) (hne : cs ≠ []) :
    InW w q cs.length := by
  rcases hb with rfl | ⟨r, off, hd, hs, -⟩
  · exact absurd rfl hne
  · intro r' off' hd'
    rw [hd] at hd'; cases hd'; exact hs

theorem Room.inW {w : World} {q : UInt64} {k : Nat} (hr : Room w q k) (hk : 0 < k) : InW w q k := by
  rcases hr with h0 | ⟨r, off, hd, hs, -⟩
  · omega
  · intro r' off' hd'
    rw [hd] at hd'; cases hd'; exact hs

theorem load_inW {w : World} {q v : UInt64} {k : Nat} (hl : w.mem.load q k = some v) : InW w q k := by
  intro r off hd
  exact (load_some hd hl).1

/-- Room is bytes that can be read. -/
theorem Room.bytes {w : World} {p : UInt64} {k : Nat} (h : Room w p k) : ∃ bs, bs.length = k ∧ Bytes w p bs := by
  rcases h with h0 | ⟨r, off, hd, hs, h36, hr⟩
  · exact ⟨[], h0.symm, Or.inl rfl⟩
  · refine ⟨(List.range k).map fun i => (w.mem.region r).get! (off + i), by simp, Or.inr ⟨r, off, hd, ?_, ?_, ?_, ?_⟩⟩
    · simpa using hs
    · simpa using h36
    · simpa using readable_of_reachable hr
    · intro i hi
      have hi' : i < k := by simpa using hi
      simp [List.getD_eq_getElem?_getD, hi']

/-- A run inside a run of bytes. -/
theorem Bytes.sub {w : World} {p : UInt64} {bs : List UInt8} (h : Bytes w p bs) {j k : Nat}
    (hjk : j + k ≤ bs.length) : Bytes w (p + UInt64.ofNat j) ((bs.drop j).take k) := by
  by_cases hk : k = 0
  · subst hk; exact Or.inl (by simp)
  rcases h with rfl | ⟨r, off, hd, hs, h36, hr, hg⟩
  · simp at hjk; omega
  have hlen : ((bs.drop j).take k).length = k := by simp; omega
  refine Or.inr ⟨r, off + j, Sem.decodeAddr_add hd (by rw [Sem.regionSpan_toNat]; omega), by rw [hlen]; omega,
    by rw [hlen]; omega, by rw [hlen]; exact readable_sub hr (by omega) (by omega), fun i hi => ?_⟩
  rw [hlen] at hi
  rw [Nat.add_assoc, hg (j + i) (by omega)]
  simp [List.getD_eq_getElem?_getD, hi]

/-- Two runs of one length at one address are the same bytes. -/
theorem Bytes.unique {w : World} {p : UInt64} {as bs : List UInt8} (h1 : Bytes w p as) (h2 : Bytes w p bs)
    (hl : bs.length = as.length) : as = bs := by
  rcases h1 with rfl | ⟨r, off, hd, hs, h36, hr, hg⟩
  · exact (List.length_eq_zero_iff.mp (by simpa using hl)).symm
  rcases h2 with rfl | ⟨r', off', hd', hs', h36', hr', hg'⟩
  · exact List.length_eq_zero_iff.mp (by simpa using hl.symm)
  rw [hd] at hd'
  cases hd'
  exact (list_eq_iff_getD hl).mpr fun t ht => (hg t ht).symm.trans (hg' t (by omega))

/-- Runs from the starts of two blocks apart are apart as numbers. -/
theorem Apart.num {x y : UInt64} {n k m : Nat} (ha : Apart x n y k) (hm : m ≤ n) (hm' : m ≤ k) :
    x.toNat + m ≤ y.toNat ∨ y.toNat + m ≤ x.toNat := by
  rcases ha with ⟨a, b, ha, hb, hab⟩ | ⟨a, b, ha, hb, hab⟩
  · left; rw [toNat_of_decode ha, toNat_of_decode hb]; omega
  · right; rw [toNat_of_decode ha, toNat_of_decode hb]; omega

/-- Bytes off a block are apart, as numbers, from every run of their length
    inside the block. -/
theorem apart_bytes {w : World} {p y : UInt64} {bs : List UInt8} {k j : Nat} (hb : Bytes w p bs) (hy : Blk w y k)
    (hj : j + bs.length ≤ k) (hc : Off y k p bs.length) :
    (y + UInt64.ofNat j).toNat + bs.length ≤ p.toNat ∨ p.toNat + bs.length ≤ (y + UInt64.ofNat j).toNat := by
  rcases hb with rfl | ⟨r, off, hd, hs, h36, -⟩
  · simp; omega
  · obtain ⟨oy, m, hat, hn, hm, hsz, h36', hlv⟩ := hy
    obtain ⟨hdy, -⟩ := heapAt_decode hat
    by_cases hn0 : bs.length = 0
    · omega
    · have := apart_of_decode hd (Sem.decodeAddr_add (i := j) hdy (by rw [Sem.regionSpan_toNat]; omega)) h36
        (by omega) (by
          rcases hc r off oy hd hdy with c | c | c
          · exact Or.inl c
          · exact Or.inr (Or.inl (by omega))
          · exact Or.inr (Or.inr (by omega)))
      omega


theorem add_ofNat_zero (x : UInt64) : x + UInt64.ofNat 0 = x := by simp

theorem Blk.le36 {w : World} {x : UInt64} {n : Nat} (h : Blk w x n) : n ≤ 2 ^ 36 := by
  obtain ⟨o, m, -, hn, -, -, h36, -⟩ := h
  omega

/-- The start of a block has room. -/
theorem Blk.room0 {w : World} {x : UInt64} {n k : Nat} (h : Blk w x n) (hc : Calm w) (hk : k ≤ n) : Room w x k := by
  have := Blk.room h hc (j := 0) (k := k) (by omega)
  rwa [add_ofNat_zero] at this

/-- A shift left by one is twice. -/
theorem shl1 {n : Nat} (hn : n < 2 ^ 63) : UInt64.ofNat n <<< (c64 1 % 64) = UInt64.ofNat (2 * n) := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_shiftLeft, show (c64 1 % 64).toNat % 64 = 1 from by decide, Nat.shiftLeft_eq,
    UInt64.toNat_ofNat', UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  omega

theorem ofNat_le_iff {a b : Nat} (ha : a < 2 ^ 64) (hb : b < 2 ^ 64) : UInt64.ofNat a ≤ UInt64.ofNat b ↔ a ≤ b := by
  rw [UInt64.le_iff_toNat_le, UInt64.toNat_ofNat', UInt64.toNat_ofNat', Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb]

theorem umax_ofNat {a b : Nat} (ha : a < 2 ^ 64) (hb : b < 2 ^ 64) :
    (if UInt64.ofNat a ≤ UInt64.ofNat b then UInt64.ofNat b else UInt64.ofNat a) = UInt64.ofNat (max a b) := by
  by_cases h : a ≤ b
  · rw [if_pos ((ofNat_le_iff ha hb).mpr h), Nat.max_eq_right h]
  · rw [if_neg (fun c => h ((ofNat_le_iff ha hb).mp c)), Nat.max_eq_left (by omega)]

/-- What one step keeps the next keeps, where what the first protects and is
    still inside memory the second protects. -/
theorem Keeps.next {w w' w'' : World} {P P' : UInt64 → Nat → Prop} (h : Keeps w w' P) (h' : Keeps w' w'' P')
    (hP : ∀ q n, 0 < n → P q n → InW w' q n → P' q n) : Keeps w w'' P :=
  ⟨fun q cs hn hp hb => h'.bytes q cs hn (hP _ _ (List.length_pos_iff.mpr hn) hp ((h.bytes q cs hn hp hb).inW hn))
      (h.bytes q cs hn hp hb),
    fun q k hk hp hr => h'.room q k hk (hP _ _ hk hp ((h.room q k hk hp hr).inW hk)) (h.room q k hk hp hr),
    fun q k v hk hp hl => h'.load q k v hk (hP _ _ hk hp (load_inW (h.load q k v hk hp hl))) (h.load q k v hk hp hl)⟩

/-- A run off a block is off every part of it. -/
theorem Off.part {w : World} {s q : UInt64} {n a c len : Nat} (hs : Blk w s n) (h : Off s n q len)
    (hac : a + c ≤ n) (ha : a < n) : Off (s + UInt64.ofNat a) c q len := by
  intro r off oy hq hy
  obtain ⟨os, hds, hdj⟩ := Blk.decode hs
  rw [hdj a ha] at hy
  cases hy
  rcases h r off os hq hds with c | c | c
  · exact Or.inl c
  · exact Or.inr (Or.inl (by omega))
  · exact Or.inr (Or.inr (by omega))

-- ---------------------------------------------------------------------------
-- A block moved to a fresh one
-- ---------------------------------------------------------------------------

/-- **`x` starts past all of `w`'s pinned memory**: where a block `w` does not
    have starts. -/
def Past (w : World) (x : UInt64) : Prop :=
  ∃ ox, decodeAddr x = some (.pinned, ox) ∧ (w.mem.region .pinned).size ≤ ox

theorem Past.before {w : World} {x y : UInt64} {k : Nat} (hp : Past w x) (h : Blk w y k) : Before y k x := by
  obtain ⟨ox, hdx, hpast⟩ := hp
  obtain ⟨off, m, hat, hn, hm, hsz, h36, hlv⟩ := h
  obtain ⟨hd, -⟩ := heapAt_decode hat
  have : off + m ≤ (w.mem.region .pinned).size := hsz
  exact ⟨off, ox, hd, hdx, by omega⟩

theorem Past.ne {w : World} {x y : UInt64} {k : Nat} (hp : Past w x) (h : Blk w y k) : y ≠ x := by
  obtain ⟨ox, hdx, hpast⟩ := hp
  obtain ⟨off, m, hat, hn, hm, hsz, h36, hlv⟩ := h
  obtain ⟨hd, -⟩ := heapAt_decode hat
  have : off + m ≤ (w.mem.region .pinned).size := hsz
  rintro rfl
  rw [hd] at hdx
  cases hdx
  omega

/-- What is inside the memory there was is off every run that starts past it. -/
theorem Past.off {w : World} {x q : UInt64} {n : Nat} (hp : Past w x) (hi : InW w q n) (m : Nat) : Off x m q n := by
  intro r off oy hq hy
  obtain ⟨ox, hdx, hpast⟩ := hp
  rw [hdx] at hy
  cases hy
  by_cases e : r = .pinned
  · subst e
    have := hi _ _ hq
    exact Or.inr (Or.inl (by omega))
  · exact Or.inl e

theorem Past.mono {w w' : World} {x : UInt64} (hp : Past w' x)
    (hle : (w.mem.region .pinned).size ≤ (w'.mem.region .pinned).size) : Past w x := by
  obtain ⟨ox, hdx, hpast⟩ := hp
  exact ⟨ox, hdx, Nat.le_trans hle hpast⟩

theorem Wrote.refl (w : World) (d : UInt64) (n : Nat) : Wrote w w d n := ⟨MemSame.refl _, rfl, fun _ _ _ => rfl⟩

/-- A run off a block is off its start. -/
theorem Off.prefix {y q : UInt64} {k k' n : Nat} (h : Off y k q n) (hk : k' ≤ k) : Off y k' q n := by
  intro r off oy hq hy
  rcases h r off oy hq hy with c | c | c
  · exact Or.inl c
  · exact Or.inr (Or.inl c)
  · exact Or.inr (Or.inr (by omega))

/-- A run inside a block past its first `c` bytes is off them. -/
theorem off_prefix {w : World} {s : UInt64} {n c j m : Nat} (hs : Blk w s n) (hj : j + m ≤ n) (hc : c ≤ j) :
    Off s c (s + UInt64.ofNat j) m := by
  intro r off oy hd hy
  obtain ⟨os, hds, -⟩ := Blk.decode hs
  obtain ⟨os', m', hat, hn, hm, hsz, h36, hlv⟩ := hs
  obtain ⟨hdd, -⟩ := heapAt_decode hat
  rw [hds] at hdd hy
  cases hdd
  cases hy
  obtain ⟨rfl, rfl⟩ := decode_in_blk hds (n := n) (by omega) (by omega) hd
  exact Or.inr (Or.inr (by omega))

/-- What moving a block protects: a run inside the memory there was, off the
    old block and off the `c` bytes at `a` in the block that points to it. -/
def MovP (w : World) (s : UInt64) (a c : Nat) (old : UInt64) (oc : Nat) (q : UInt64) (n : Nat) : Prop :=
  InW w q n ∧ Off old oc q n ∧ Off (s + UInt64.ofNat a) c q n

/-- **`w'` is `w` with the block at `old` moved**: what is off that block and
    off the fields that name it is kept, every other block is there, and
    memory has not shrunk. -/
structure Moved (w w' : World) (s : UInt64) (a c : Nat) (old : UInt64) (oc : Nat) : Prop where
  keeps : Keeps w w' (MovP w s a c old oc)
  calm : Calm w → Calm w'
  blk : ∀ y k, y ≠ old → Blk w y k → Blk w' y k
  whole : ∀ y k, y ≠ old → Whole w y k → Whole w' y k
  size : ∀ r, r ≠ .pinned → (w'.mem.region r).size = (w.mem.region r).size
  inw : ∀ q n, InW w q n → InW w' q n
  psize : (w.mem.region .pinned).size ≤ (w'.mem.region .pinned).size

/-- **A block moved to a fresh one**: more heap with a fresh block `x`, a
    write inside `x`, the old block freed, and two fields of the block `s`
    set, the first to `x`. What is left is `Moved`; `x` is whole and holds
    what the write left in it; the two fields hold what was stored, and the
    other fields of `s` what they held. -/
theorem moved {w w1 w2 w3 w5 : World} {s old x v2 : UInt64} {n a c a1 a2 oc N k : Nat} {m4 m5 : Mem}
    (hs : Blk w s n) (wo : Whole w old oc) (hso : s ≠ old) (g1 : Grows w w1) (f : Fresh w w1 x N) (hk : k ≤ N)
    (W : Wrote w1 w2 x k) (F : Freed w2 w3 old) (h1 : a ≤ a1 ∧ a1 + 8 ≤ a + c) (h2 : a ≤ a2 ∧ a2 + 8 ≤ a + c)
    (h12 : a1 + 8 ≤ a2 ∨ a2 + 8 ≤ a1) (hac : a + c ≤ n)
    (hm4 : w3.mem.store (s + UInt64.ofNat a1) 8 x = some m4)
    (hm5 : (stored w3 (s + UInt64.ofNat a1) x m4).mem.store (s + UInt64.ofNat a2) 8 v2 = some m5)
    (hw5 : w5 = stored (stored w3 (s + UInt64.ofNat a1) x m4) (s + UInt64.ofNat a2) v2 m5) :
    Moved w w5 s a c old oc ∧ Whole w5 x N ∧ fld w5 s a1 = some x ∧ fld w5 s a2 = some v2 ∧
      (∀ j v, j + 8 ≤ n → (j + 8 ≤ a1 ∨ a1 + 8 ≤ j) → (j + 8 ≤ a2 ∨ a2 + 8 ≤ j) → fld w s j = some v →
        fld w5 s j = some v) ∧
      (∀ j cs, j + cs.length ≤ N → Bytes w2 (x + UInt64.ofNat j) cs → Bytes w5 (x + UInt64.ofNat j) cs) ∧
      (∀ j v, j + 8 ≤ N → fld w2 x j = some v → fld w5 x j = some v) := by
  subst hw5
  have a_xs : Apart x N s n := Or.inr (f.before hs)
  have a_xo : Apart x N old oc := Or.inr (f.before wo.blk)
  have bs1 : Blk w1 s n := g1.blk _ _ hs
  have bs2 : Blk w2 s n := W.stays.blk _ _ bs1
  have bs3 : Blk w3 s n := F.blk _ _ hso bs2
  have bs4 := Blk.keep bs3 hm4
  have bn1 : Blk w1 x N := f.blk
  have bn2 : Blk w2 x N := W.stays.blk _ _ bn1
  have hxo : x ≠ old := (f.ne wo.blk).symm
  have bn3 : Blk w3 x N := F.blk _ _ hxo bn2
  have bn4 := Blk.keep bn3 hm4
  have wo2 : Whole w2 old oc := W.whole (g1.whole _ _ wo)
  have W4 := wrote_stored hm4
  have W5 := wrote_stored hm5
  have c_xs : ∀ r off oy, decodeAddr x = some (r, off) → decodeAddr s = some (.pinned, oy) →
      r ≠ .pinned ∨ off + k ≤ oy ∨ oy + n ≤ off := by
    have := a_xs.clear bn1 (j := 0) (n := k) (by omega)
    rwa [add_ofNat_zero] at this
  have c1 : ∀ {ww : World}, Blk ww s n → ∀ r off oy, decodeAddr (s + UInt64.ofNat a1) = some (r, off) →
      decodeAddr x = some (.pinned, oy) → r ≠ .pinned ∨ off + 8 ≤ oy ∨ oy + N ≤ off :=
    fun hb => a_xs.symm.clear hb (by omega)
  have c2 : ∀ {ww : World}, Blk ww s n → ∀ r off oy, decodeAddr (s + UInt64.ofNat a2) = some (r, off) →
      decodeAddr x = some (.pinned, oy) → r ≠ .pinned ∨ off + 8 ≤ oy ∨ oy + N ≤ off :=
    fun hb => a_xs.symm.clear hb (by omega)
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, W5.whole (W4.whole (F.whole _ _ hxo (W.whole f.whole))), ?_, fld_stored hm5,
    ?_, ?_, ?_⟩
  · refine ((((g1.keeps _).trans (W.keeps _ fun q n hn hp => ?_)).trans (F.keeps wo2 _ fun q k hp => hp.2.1)).trans
      (W4.keeps _ fun q n hn hp => Off.sub bs3 hp.2.2 h1.1 h1.2 hac (by omega))).trans
      (W5.keeps _ fun q n hn hp => Off.sub bs4 hp.2.2 h2.1 h2.2 hac (by omega))
    have := f.wrote (j := 0) (m := k) (n := n) (by omega) hp.1
    rwa [add_ofNat_zero] at this
  · exact fun c => ((F.calm (W.stays.calm (g1.calm c))).stored hm4).stored hm5
  · exact fun y k hy hb => Blk.keep (Blk.keep (F.blk _ _ hy (W.stays.blk _ _ (g1.blk _ _ hb))) hm4) hm5
  · exact fun y k hy hb => W5.whole (W4.whole (F.whole _ _ hy (W.whole (g1.whole _ _ hb))))
  · exact fun r hr => (W5.stays.size r hr).trans ((W4.stays.size r hr).trans ((F.size r hr).trans
      ((W.stays.size r hr).trans (g1.size r hr))))
  · exact fun _ _ hi => InW.wrote W5 (InW.wrote W4 (InW.freed F (InW.wrote W (InW.grows g1 hi))))
  · rw [← W5.same.size, ← W4.same.size, F.psize, ← W.same.size]
    exact g1.psize
  · exact (fld_kept bs4 (by omega) (by omega) h12 hm5).trans (fld_stored hm4)
  · intro j v hj hj1 hj2 hv
    have h2' : fld w2 s j = some v := (W.fld_blk bs1 hj c_xs).trans (g1.fld hv)
    have h3 := F.fld s n j _ hso bs2 hj h2'
    exact (fld_kept bs4 hj (by omega) hj2 hm5).trans ((fld_kept bs3 hj (by omega) hj1 hm4).trans h3)
  · intro j cs hj hB
    have h3 : Bytes w3 (x + UInt64.ofNat j) cs := F.bytes _ wo2 _ _ hB (a_xo.clear bn2 hj)
    exact W5.bytes_blk bn4 hj (W4.bytes_blk bn3 hj h3 (c1 bs3)) (c2 bs4)
  · intro j v hj hv
    have h3 := F.fld x N j v hxo bn2 hj hv
    exact (W5.fld_blk bn4 hj (c2 bs4)).trans ((W4.fld_blk bn3 hj (c1 bs3)).trans h3)

/-- A run inside a block apart from the moved one and from the block that
    names it is one a move protects. -/
theorem movP_blk {w : World} {s y old : UInt64} {n0 k oc a c j n : Nat} (hs : Blk w s n0) (hy : Blk w y k)
    (hyo : Apart y k old oc) (hys : Apart y k s n0) (hac : a + c ≤ n0) (ha : a < n0) (hj : j + n ≤ k) :
    MovP w s a c old oc (y + UInt64.ofNat j) n :=
  ⟨Blk.inW hy hj, hyo.clear hy hj, Off.part hs (hys.clear hy hj) hac ha⟩

/-- In a list of distinct elements, none is among those before it. -/
theorem not_mem_take_of_nodup {α : Type} {l : List α} (hn : l.Nodup) {i : Nat} (hi : i < l.length) :
    l[i] ∉ l.take i := by
  have h1 : (l.take (i + 1)).Nodup := hn.sublist (List.take_sublist _ _)
  rw [← List.take_append_getElem hi, List.nodup_append] at h1
  intro hm
  exact h1.2.2 _ hm _ (List.mem_singleton.mpr rfl) rfl

theorem ofNat_ne_neg1 {n : Nat} (hn : n < 2 ^ 64 - 1) : UInt64.ofNat n ≠ c64 (-1) := by
  intro e
  have := congrArg UInt64.toNat e
  rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), show (c64 (-1)).toNat = 2 ^ 64 - 1 from by decide] at this
  omega

/-- Distinct elements at distinct places. -/
theorem nodup_getElem_ne {α : Type} {l : List α} (hn : l.Nodup) {p q : Nat} (hp : p < l.length) (hq : q < l.length)
    (hne : p ≠ q) : l[p] ≠ l[q] := by
  have key : ∀ a b (ha : a < l.length) (hb : b < l.length), a < b → l[a] ≠ l[b] := by
    intro a b ha hb hab e
    apply not_mem_take_of_nodup hn hb
    rw [← e]
    have h1 : a < (l.take b).length := by simp; omega
    have h2 : (l.take b)[a]'h1 = l[a] := by simp
    rw [← h2]
    exact List.getElem_mem _
  rcases Nat.lt_or_gt_of_ne hne with h | h
  · exact key p q hp hq h
  · exact fun e => key q p hq hp h e.symm

/-- A run inside a block, apart from a write inside it, is what it was. -/
theorem Wrote.bytes_in {w w' : World} {y : UInt64} {k a n b : Nat} {cs : List UInt8}
    (W : Wrote w w' (y + UInt64.ofNat a) n) (hb : Blk w y k) (hbs : Bytes w (y + UInt64.ofNat b) cs)
    (ha : a + n ≤ k) (hb' : b + cs.length ≤ k) (hd : a + n ≤ b ∨ b + cs.length ≤ a) :
    Bytes w' (y + UInt64.ofNat b) cs := by
  by_cases hc : cs = []
  · exact Or.inl hc
  have hl := List.length_pos_iff.mpr hc
  refine W.bytes hbs fun hn r off r' off' hd' hq => ?_
  obtain ⟨o, -, hdj⟩ := Blk.decode hb
  rw [hdj a (by omega)] at hd'
  rw [hdj b (by omega)] at hq
  cases hd'
  cases hq
  rcases hd with h | h
  · exact Or.inr (Or.inr (by omega))
  · exact Or.inr (Or.inl (by omega))

-- ---------------------------------------------------------------------------
-- The bytes of a word
-- ---------------------------------------------------------------------------

/-- The eight bytes of a 64-bit value, least first. -/
def le8 (x : UInt64) : List UInt8 := (List.range 8).map (byteOfV x)

theorem le8_length (x : UInt64) : (le8 x).length = 8 := by simp [le8]

theorem le8_getD (x : UInt64) {i : Nat} (hi : i < 8) : (le8 x).getD i 0 = byteOfV x i := by
  simp [le8, List.getD_eq_getElem?_getD, hi]

/-- The eight bytes of a value make the value. -/
theorem leN_le8 (x : UInt64) : leN (le8 x) 0 8 = x := by
  unfold leN
  refine (foldr_fun_congr _ (fun i => (byteOfV x i).toUInt64) _ _ fun i hi => ?_).trans (reassemble8 x)
  rw [Nat.zero_add, le8_getD x (List.mem_range.mp hi)]

/-- The model's eight bytes of a value are these. -/
theorem le64_eq (x : UInt64) : le64 x = ByteArray.mk (le8 x).toArray := by
  have h := List.data_toByteArray (l := le8 x)
  calc le64 x = (le8 x).toByteArray := rfl
    _ = ByteArray.mk (le8 x).toByteArray.data := rfl
    _ = ByteArray.mk (le8 x).toArray := by rw [h]

/-- The value the model reads from eight bytes is the word they make. -/
theorem ofLe64_ofList (bs : List UInt8) : ofLe64 (ByteArray.mk bs.toArray) = leN bs 0 8 := by
  unfold ofLe64 leN
  exact foldr_fun_congr _ _ _ _ fun i _ => by rw [get!_ofList, Nat.zero_add]

theorem ofLe64_data (b : ByteArray) : ofLe64 b = leN b.data.toList 0 8 := by
  have := ofLe64_ofList b.data.toList
  rwa [show ByteArray.mk b.data.toList.toArray = b from by cases b; simp] at this

/-- **An eight-byte store over bytes that can be read** leaves the value's
    bytes, then the rest. -/
theorem stored_bytes {w : World} {p x : UInt64} {m : Mem} {bs : List UInt8} (hb : Bytes w p bs)
    (h8 : 8 ≤ bs.length) (hs : w.mem.store p 8 x = some m) : Bytes (stored w p x m) p (le8 x ++ bs.drop 8) := by
  have W := wrote_stored hs
  rcases hb with rfl | ⟨r, off, hd, hsz, h36, hr, hg⟩
  · simp at h8
  have hl : (le8 x ++ bs.drop 8).length = bs.length := by
    simp [le8_length]
    omega
  obtain ⟨-, -, hv⟩ := load_some (m := (stored w p x m).mem) hd (load_store8 hs)
  have hbytes : ∀ i, i < 8 → ((stored w p x m).mem.region r).get! (off + i) = byteOfV x i :=
    (leV_inj 8 (fun i => ((stored w p x m).mem.region r).get! (off + i)) (byteOfV x) (Nat.le_refl _)).mp
      (hv.symm.trans (reassemble8 x).symm)
  refine Or.inr ⟨r, off, hd, by rw [hl, ← W.same.size r]; exact hsz, by rw [hl]; exact h36,
    by rw [hl, ← W.same.readable]; exact hr, fun i hi => ?_⟩
  rw [hl] at hi
  by_cases c : i < 8
  · rw [hbytes i c, ← le8_getD x c]
    simp [List.getD_eq_getElem?_getD, List.getElem?_append_left, le8_length, c]
  · rw [W.get r (off + i) (fun _ r0 off0 hd0 => by
      rw [hd] at hd0
      cases hd0
      exact Or.inr (Or.inr (by omega))), hg i hi]
    simp [List.getD_eq_getElem?_getD, List.getElem?_append_right, le8_length, Nat.le_of_not_lt c]

-- ---------------------------------------------------------------------------
-- A library function, called
-- ---------------------------------------------------------------------------

/-- A library function as one of a program's functions. -/
def implFn (params : List ClifTy) : Lib.Impl → Fn
  | .void b => { params, code := emit b params, status := none }
  | .status b => { params, code := (runAns b params).2.1, status := (runAns b params).1 }

/-- The library functions `impl` gives, each at the place its entry point
    names (`Ffi.libSlot`): the functions a program that calls the entry points
    is run with. -/
def libFnsOf (impl : Ffi → Option Lib.Impl) (i : Nat) : Option Fn :=
  (Ffi.all.find? fun f => f.inLib && f.libSlot == i).bind fun f => (impl f).map (implFn f.params)

/-- **A call to function `i` on `args` from `w` answers as `Q` says**: it
    neither misuses a call nor faults, and where it returns, its answer and
    the world it leaves are `Q`'s. -/
def CallsA (L : Sem.Locals) (i : Nat) (args : List V) (w : World) (Q : UInt64 → World → Prop) : Prop :=
  match L i args w with
  | .error f => FailOk (f.out : Outcome Env) false
  | .ok (x, w') => ∃ v, x = some (.sc .i64 v) ∧ Q v w'

/-- **A call to function `i` on `args` from `w` does as `Q` says.** -/
def CallsD (L : Sem.Locals) (i : Nat) (args : List V) (w : World) (Q : World → Prop) : Prop :=
  match L i args w with
  | .error f => FailOk (f.out : Outcome Env) false
  | .ok (_, w') => Q w'

/-- A function with a statement, in a program's table, is called as the
    statement says at every call depth. -/
theorem Ans.calls {fns : Nat → Option Fn} {env : FnEnv} {steps i : Nat} {p : StatusBody} {params : List ClifTy}
    {args : List V} {w : World} {Q : UInt64 → World → Prop} (hf : fns i = some (implFn params (.status p)))
    (h : ∀ cfg, Ans cfg args w p params Q) : ∀ k, CallsA (termLocals fns env steps k) i args w Q
  | 0 => by simp [CallsA, termLocals, Sem.noLocals, Sem.Fail.out, FailOk]
  | k + 1 => by
    unfold CallsA
    simp only [termLocals, hf]
    unfold Sem.callFn
    by_cases hlen : (implFn params (.status p)).params.length ≠ args.length
    · simp [hlen, Sem.Fail.out, FailOk]
    · have hs := h { env, steps, locals := termLocals fns env steps k } steps args.toArray w ⟨rfl, rfl⟩
      rw [if_neg hlen]
      simp only [implFn] at hs ⊢
      cases hr : runCode steps { env, steps, locals := termLocals fns env steps k } args.toArray w
          (runAns p params).2.1 with
      | ok Γ1 w1 =>
        rw [hr] at hs
        obtain ⟨a, x, ha, hx, hq⟩ := hs
        refine ⟨x, ?_, hq⟩
        rw [ha]
        exact hx
      | _ =>
        rw [hr] at hs
        simp_all [CodeRes.Sat, Sem.Fail.out, FailOk]

theorem Does.calls {fns : Nat → Option Fn} {env : FnEnv} {steps i : Nat} {p : Body} {params : List ClifTy}
    {args : List V} {w : World} {Q : World → Prop} (hf : fns i = some (implFn params (.void p)))
    (h : ∀ cfg, Does cfg args w p params Q) : ∀ k, CallsD (termLocals fns env steps k) i args w Q
  | 0 => by simp [CallsD, termLocals, Sem.noLocals, Sem.Fail.out, FailOk]
  | k + 1 => by
    unfold CallsD
    simp only [termLocals, hf]
    unfold Sem.callFn
    by_cases hlen : (implFn params (.void p)).params.length ≠ args.length
    · simp [hlen, Sem.Fail.out, FailOk]
    · have hs := h { env, steps, locals := termLocals fns env steps k } steps args.toArray w ⟨rfl, rfl⟩
      rw [if_neg hlen]
      simp only [implFn] at hs ⊢
      cases hr : runCode steps { env, steps, locals := termLocals fns env steps k } args.toArray w
          (emit p params) <;> rw [hr] at hs <;> simp_all [CodeRes.Sat, Sem.Fail.out, FailOk]

-- ---------------------------------------------------------------------------
-- Calls out of the program, and their result codes
-- ---------------------------------------------------------------------------

section ext
variable {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Γ : Env} {w : World}

/-- A branch on two 32-bit slots, both arms ending in `Qj`. -/
theorem wp_ifte32 {jTys : List ClifTy} {cc : ICmpCond} {a b : Slot .i32} {x y : UInt64}
    {thn els : Prog Slot Lvl (Vals Slot jTys)} {Q : Vals Slot jTys → Env → World → Prop}
    (Qj : Vals Slot jTys → Env → World → Prop)
    (ha : Γ[a]? = some (.sc .i32 x)) (hb : Γ[b]? = some (.sc .i32 y))
    (ht : cmpInt cc .i32 x y = true → wp cfg d J thn Qj (Γ.push (boolV true)) w)
    (he : cmpInt cc .i32 x y = false → ∀ ne, (Γ.push (boolV false)).size ≤ ne →
      wp cfg d J els Qj (bindAt (Γ.push (boolV false)) ne []) w)
    (hk : ∀ nJ Γ' jv vs w', Γ'.size ≤ nJ → jv.slots.mapM (fun r => Γ'[r]?) = some vs → Qj jv Γ' w' →
      Q (carriesFrom nJ jTys) (bindAt Γ' nJ vs) w') :
    wp cfg d J (ifte cc a b thn els) Q Γ w := by
  simp only [ifte, wp_ite, wp_ret]
  have hev : evalOp w.mem Γ (.icmp cc a b) = some (boolV (cmpInt cc .i32 x y)) := by
    simp [evalOp, Sem.get, ha, hb, zipIntCmp]
    rfl
  refine ⟨Qj, Qj, ⟨?_, ?_, ?_⟩, fun nJ Γ' jv vs w' hle hvs hq => hk nJ Γ' jv vs w' hle hvs (hq.elim id id)⟩
  · intro t f hc hf
    rw [hev] at hc
    have hb' : V.sc t f = boolV (cmpInt cc .i32 x y) := (Option.some.inj hc).symm
    cases hcmp : cmpInt cc .i32 x y
    · rw [hcmp] at hb'; simp [boolV] at hb'; obtain ⟨rfl, rfl⟩ := hb'; simp at hf
    · rw [hb', hcmp]; exact ht hcmp
  · intro ne t f hne hc hf
    rw [hev] at hc
    have hb' : V.sc t f = boolV (cmpInt cc .i32 x y) := (Option.some.inj hc).symm
    cases hcmp : cmpInt cc .i32 x y
    · rw [hb', hcmp]; exact he hcmp ne (by simpa using hne)
    · rw [hcmp] at hb'; simp [boolV] at hb'; obtain ⟨rfl, rfl⟩ := hb'; simp at hf
  · rw [hev]; nofun

/-- `-1` where the comparison of two 32-bit slots holds, what `k` answers
    where it does not. -/
theorem wp_failIf32 {cc : ICmpCond} {a b : Slot .i32} {x y : UInt64} {k : Prog Slot Lvl (Slot .i64)}
    {R : UInt64 → World → Prop} (ha : Γ[a]? = some (.sc .i32 x)) (hb : Γ[b]? = some (.sc .i32 y))
    (ht : cmpInt cc .i32 x y = true → R (c64 (-1)) w)
    (he : cmpInt cc .i32 x y = false → ∀ Γ', Ext Γ Γ' → wp cfg d J k (AnsQ R) Γ' w) :
    wp cfg d J (Lib.failIf cc a b k) (AnsQ R) Γ w := by
  unfold Lib.failIf
  wp_seq
  refine wp_ifte32 (AnsJ R) ha hb ?_ ?_ (by join_ans)
  · intro h
    wp_seq
    apply wp_iconst64
    exact ⟨_, get_push_size, ht h⟩
  · intro h ne hne
    wp_seq
    exact he h _ (by ext_env)

/-- `0` after a result code of `0`, `-1` after any other. -/
theorem wp_status {rc : Slot .i32} {x : UInt64} {R : UInt64 → World → Prop}
    (ha : Γ[rc]? = some (.sc .i32 x)) (h : R (if x = 0 then 0 else c64 (-1)) w) :
    wp cfg d J (Lib.status rc) (AnsQ R) Γ w := by
  unfold Lib.status Lib.onOk
  wp_seq
  apply wp_iconst32
  refine wp_failIf32 (x := x) (y := 0) (get_push ha) get_push_size ?_ ?_
  · intro hc
    have hx : x ≠ 0 := by simpa [cmpInt] using hc
    rwa [if_neg hx] at h
  · intro hc Γ' hext
    have hx : x = 0 := by simpa [cmpInt] using hc
    apply wp_iconst64
    rw [if_pos hx] at h
    exact ⟨_, get_push_size, h⟩

/-- **A call to a library outside the program** that answers a result code:
    where the model answers the call, the program goes on from each answer
    and world it gives. -/
theorem wp_extRc {ps : List ClifTy} {e : AlgorithmLib.IR.Ext} {args : Vals Slot ps} {k : ResV Slot (some .i32) → Prog Slot Lvl α}
    {Q : α → Env → World → Prop} {cs : List V}
    (hcs : args.slots.mapM (fun r => Γ[r]?) = some cs)
    (hdef : (extCall e cs (obsCall w (.ext e) cs)).isSome = true)
    (hQ : ∀ v w', extCall e cs (obsCall w (.ext e) cs) = some (some v, w') →
      wp cfg d J (k Γ.size) Q (Γ.push v) w') :
    wp cfg d J (.callLocal (res := some .i32) ⟨.ext e⟩ args k) Q Γ w := by
  rw [wp_callLocal]
  intro vs hvs
  rw [hcs] at hvs; cases hvs
  simp only [callOf]
  refine ⟨fun hn => ?_, fun x w' hx => ?_⟩
  · rw [hn] at hdef; cases hdef
  · simp only [Option.isSome_some, if_true]
    rintro v rfl
    exact hQ v w' hx

end ext

-- ---------------------------------------------------------------------------
-- Only the heap changed
-- ---------------------------------------------------------------------------

/-- **A write of `n` bytes at `p` misses the eight at `slot`**. -/
def Misses (slot p : UInt64) (n : Nat) : Prop :=
  0 < n → ∀ r off r' off', decodeAddr slot = some (r, off) → decodeAddr p = some (r', off') →
    r ≠ r' ∨ off + 8 ≤ off' ∨ off' + n ≤ off

/-- **`w'` is `w` outside the heap, where `P` holds**: every run of the
    memory there was that is not of the heap and that `P` holds of is kept,
    what was calm is calm, and memory outside the heap is the size it was.
    What a program keeps of its own memory over a call into a library. -/
structure Outside (w w' : World) (P : UInt64 → Nat → Prop) : Prop where
  keeps : Keeps w w' (fun q n => InW w q n ∧ Out q ∧ P q n)
  calm : Calm w → Calm w'
  size : ∀ r, r ≠ .pinned → (w'.mem.region r).size = (w.mem.region r).size

theorem Outside.refl (w : World) (P : UInt64 → Nat → Prop) : Outside w w P := ⟨Keeps.refl _ _, id, fun _ _ => rfl⟩

theorem Outside.trans {w w' w'' : World} {P : UInt64 → Nat → Prop} (h : Outside w w' P) (h' : Outside w' w'' P) :
    Outside w w'' P :=
  ⟨h.keeps.next h'.keeps fun _ _ _ hp hi' => ⟨hi', hp.2.1, hp.2.2⟩, fun c => h'.calm (h.calm c),
    fun r hr => (h'.size r hr).trans (h.size r hr)⟩

theorem Outside.mono {w w' : World} {P P' : UInt64 → Nat → Prop} (h : Outside w w' P)
    (hP : ∀ q n, P' q n → P q n) : Outside w w' P' :=
  ⟨h.keeps.mono fun q n hp => ⟨hp.1, hp.2.1, hP q n hp.2.2⟩, h.calm, h.size⟩

/-- More heap changes only the heap. -/
theorem Grows.outside {w w' : World} (g : Grows w w') (P : UInt64 → Nat → Prop) : Outside w w' P :=
  ⟨g.keeps _, g.calm, g.size⟩

/-- A free changes only the heap. -/
theorem Freed.outside {w w' : World} {x : UInt64} (f : Freed w w' x) (P : UInt64 → Nat → Prop) : Outside w w' P := by
  refine ⟨⟨fun q cs _ hp hb => f.outBytes q cs hp.2.1 hb, fun q k _ hp hr => f.outRoom q k hp.2.1 hr,
    fun q k v _ hp hl => ?_⟩, f.calm, f.size⟩
  cases hd : decodeAddr q with
  | none => simp [Mem.load, hd] at hl
  | some a =>
    obtain ⟨r, off⟩ := a
    exact (f.out q k r off hd (hp.2.1 r off hd)).trans hl

/-- A write into the heap changes only the heap. -/
theorem Wrote.outside {w w' : World} {d : UInt64} {n : Nat} (W : Wrote w w' d n)
    (hd : ∀ r off, decodeAddr d = some (r, off) → r = .pinned) (P : UInt64 → Nat → Prop) : Outside w w' P :=
  ⟨W.keeps _ fun _ _ _ hp r off r' off' hdd hq => Or.inl fun e => hp.2.1 r' off' hq (e.trans (hd r off hdd)),
    W.stays.calm, W.stays.size⟩

/-- A field of a block set changes only the heap. -/
theorem Outside.setFld {w : World} {s x : UInt64} {n j : Nat} {m : Mem} (hb : Blk w s n) (hj : j < n)
    (hm : w.mem.store (s + UInt64.ofNat j) 8 x = some m) (P : UInt64 → Nat → Prop) :
    Outside w (stored w (s + UInt64.ofNat j) x m) P := by
  obtain ⟨o, -, hdec⟩ := Blk.decode hb
  exact (wrote_stored hm).outside (fun r off h => by rw [hdec j hj] at h; cases h; rfl) P

/-- An address decoded outside the heap is not of the heap. -/
theorem out_of {p : UInt64} {r : Region} {off : Nat} (hd : decodeAddr p = some (r, off)) (hr : r ≠ .pinned) :
    Out p := fun r' off' h => by
  rw [hd] at h
  cases h
  exact hr

theorem misses_of {slot p : UInt64} {n : Nat} {r r' : Region} {off off' : Nat}
    (hs : decodeAddr slot = some (r, off)) (hp : decodeAddr p = some (r', off'))
    (h : r ≠ r' ∨ off + 8 ≤ off' ∨ off' + n ≤ off) : Misses slot p n := fun _ r1 o1 r2 o2 h1 h2 => by
  rw [hs] at h1
  rw [hp] at h2
  cases h1
  cases h2
  exact h

/-- Bytes with room outside the heap can be written where no worker is
    running. -/
theorem room_out {w : World} {p : UInt64} {n : Nat} {r : Region} {off : Nat} (hd : decodeAddr p = some (r, off))
    (hr : r ≠ .pinned) (hs : off + n ≤ (w.mem.region r).size) (h36 : off + n ≤ 2 ^ 36)
    (hz : w.mem.frozen = false) : Room w p n :=
  Or.inr ⟨r, off, hd, hs, h36, by
    simp only [Mem.reachable, hz, Bool.not_false, Bool.true_and, Bool.or_eq_true]
    exact Or.inl (by cases r <;> first | rfl | exact absurd rfl hr)⟩

/-- A slot written changes, outside the heap, at most the slot. -/
theorem Wrote.outsideBut {w w' : World} {slot : UInt64} (W : Wrote w w' slot 8) : Outside w w' (Misses slot) :=
  ⟨W.keeps _ fun q n hn hp r off r' off' hd hq => by
      rcases hp.2.2 hn r off r' off' hd hq with h | h | h
      · exact Or.inl (Ne.symm h)
      · exact Or.inr (Or.inr h)
      · exact Or.inr (Or.inl h),
    W.stays.calm, W.stays.size⟩

theorem CallsA.mono {L : Sem.Locals} {i : Nat} {args : List V} {w : World} {Q Q' : UInt64 → World → Prop}
    (h : CallsA L i args w Q) (hq : ∀ x w', Q x w' → Q' x w') : CallsA L i args w Q' := by
  unfold CallsA at h ⊢
  cases hr : L i args w with
  | error f => rw [hr] at h; exact h
  | ok p =>
    rw [hr] at h
    obtain ⟨v, hv, hQ⟩ := h
    exact ⟨v, hv, hq _ _ hQ⟩

theorem CallsD.mono {L : Sem.Locals} {i : Nat} {args : List V} {w : World} {Q Q' : World → Prop}
    (h : CallsD L i args w Q) (hq : ∀ w', Q w' → Q' w') : CallsD L i args w Q' := by
  unfold CallsD at h ⊢
  cases hr : L i args w with
  | error f => rw [hr] at h; exact h
  | ok p => rw [hr] at h; exact hq _ h

/-- Noting a call writes nothing. -/
theorem wrote_obs (w : World) (c : Callee) (vs : List V) (p : UInt64) : Wrote w (obsCall w c vs) p 0 :=
  ⟨MemSame.refl _, rfl, fun _ _ _ => rfl⟩

theorem mapM_slots_nil {Γ : Env} : ([] : List R).mapM (fun r => Γ[r]?) = some [] := rfl

theorem mapM_slots_cons {Γ : Env} {r : R} {rs : List R} {v : V} {vs : List V}
    (h : Γ[r]? = some v) (t : rs.mapM (fun r => Γ[r]?) = some vs) :
    (r :: rs).mapM (fun r => Γ[r]?) = some (v :: vs) := by
  simp only [List.mapM_cons, h, t, Option.bind_eq_bind, Option.bind_some, Option.pure_def]

/-- The values a call's argument slots hold, read slot by slot: each by `slot`,
    falling back to `rfl` where a value is written differently from the one
    the environment holds. Unifying the whole list at once unfolds the
    environment through every slot, seconds per argument. -/
macro "slots" : tactic => `(tactic| (simp only [Vals.slots]; repeat (first
  | ((with_reducible refine mapM_slots_cons ?_ ?_); (first | slot | rfl))
  | with_reducible exact mapM_slots_nil)))

section fromProgram
variable {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop} {Γ : Env} {w : World}

/-- **A call to a function that answers**, from a program: where the call is
    as `Q'` says from the world with the call noted, the program goes on from
    each answer and world `Q'` allows. -/
theorem wp_callA {ps : List ClifTy} {i : Nat} {args : Vals Slot ps} {k : ResV Slot (some .i64) → Prog Slot Lvl α}
    {cs : List V} {Q' : UInt64 → World → Prop} (hfo : J.faultOk = false)
    (hcs : args.slots.mapM (fun r => Γ[r]?) = some cs)
    (hc : CallsA cfg.locals i cs (obsCall w (.local i) cs) Q')
    (hQ : ∀ x w', Q' x w' → wp cfg d J (k Γ.size) Q (Γ.push (.sc .i64 x)) w') :
    wp cfg d J (.callLocal (res := some .i64) ⟨.local i⟩ args k) Q Γ w := by
  rw [wp_callLocal]
  intro vs hvs
  rw [hcs] at hvs; cases hvs
  simp only [callOf, failOf]
  unfold CallsA at hc
  revert hc
  cases cfg.locals i cs (obsCall w (.local i) cs) with
  | error f =>
      intro h
      simp only [Except.toOption]
      rw [hfo]
      exact ⟨fun _ => h, fun _ _ hc => by cases hc⟩
  | ok p =>
      obtain ⟨x, w'⟩ := p
      intro h
      obtain ⟨v, rfl, hv⟩ := h
      simp only [Except.toOption, reduceCtorEq, false_implies, Option.some.injEq, Prod.mk.injEq, true_and]
      rintro x' w'' ⟨rfl, rfl⟩
      simp only [Option.isSome_some, if_true]
      rintro v' ⟨rfl⟩
      exact hQ _ _ hv

/-- **A call to a function that answers nothing**, from a program. -/
theorem wp_callD {ps : List ClifTy} {i : Nat} {args : Vals Slot ps} {k : ResV Slot none → Prog Slot Lvl α}
    {cs : List V} {Q' : World → Prop} (hfo : J.faultOk = false)
    (hcs : args.slots.mapM (fun r => Γ[r]?) = some cs)
    (hc : CallsD cfg.locals i cs (obsCall w (.local i) cs) Q')
    (hQ : ∀ w', Q' w' → wp cfg d J (k ()) Q Γ w') :
    wp cfg d J (.callLocal (res := none) ⟨.local i⟩ args k) Q Γ w := by
  rw [wp_callLocal]
  intro vs hvs
  rw [hcs] at hvs; cases hvs
  simp only [callOf, failOf]
  unfold CallsD at hc
  revert hc
  cases cfg.locals i cs (obsCall w (.local i) cs) with
  | error f =>
      intro h
      simp only [Except.toOption]
      rw [hfo]
      exact ⟨fun _ => h, fun _ _ hc => by cases hc⟩
  | ok p =>
      obtain ⟨x, w'⟩ := p
      intro h
      simp only [Except.toOption, reduceCtorEq, false_implies, Option.some.injEq, Prod.mk.injEq, true_and]
      rintro x' w'' ⟨rfl, rfl⟩
      simp only [Option.isSome_none, Bool.false_eq_true, if_false]
      exact hQ _ h

end fromProgram

end AlgorithmLib.LibSpec
