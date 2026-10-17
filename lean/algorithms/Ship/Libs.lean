import AlgorithmLib.Check.Entry
import AlgorithmLib.Surface.Link

/-!
# The libraries' own functions neither misuse a call nor fault

A program's calls to the engine's entry points are linked to functions written
in CLIF over the C library and the vendor libraries (`Surface.Lib*`). Here a
library function is checked as a body of its own: entered with arguments its
typestate describes, it makes no call the model refuses and touches no byte
that is not there.

The functions that set a context up are each handed the address of eight
writable bytes of the arena, where they leave the context they allocate, or
null. A function handed a context takes null or a live heap block of the
context's size. No copy may be in flight (`.part .idle true`), since the
heap's blocks are read and written.

`htInit` frees what it allocated where a later allocation fails, which the
checker does not follow; it is `HtSpec.init_sound` (`Ship.HtSpec`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.Check

namespace LibSafe

/-- What the functions that set a context up ask of the world: no worker
    running and no copy in flight. -/
abbrev Quiet : World → Prop := Contracts.TState.holds [.part .frozen false, .part .idle true]

set_option maxRecDepth 200000 in
/-- **`threadInit`** allocates its context, writes the first handle into it,
    and leaves its address in the slot. -/
theorem threadInit_no_misuse (cfg : Cfg) (off : Nat) (w : World) (h : Quiet w)
    (hroom : off + 8 ≤ w.mem.sizes .arena) (hspan : off + 8 ≤ span) (m : String) :
    Sem.run cfg [.sc .i64 (regionBase .arena + UInt64.ofNat off)] w (emit LibThread.init [.i64]) ≠ .misuse m ∧
    Sem.run cfg [.sc .i64 (regionBase .arena + UInt64.ofNat off)] w (emit LibThread.init [.i64]) ≠ .fault m := by
  check_lib [AV.room .arena 8]
  exact ⟨AV.room_desc (by decide) (by decide) hroom hspan, trivial⟩

set_option maxRecDepth 200000 in
/-- **`lmdbInit`** allocates its context and leaves its address in the slot. -/
theorem lmdbInit_no_misuse (cfg : Cfg) (off : Nat) (w : World) (h : Quiet w)
    (hroom : off + 8 ≤ w.mem.sizes .arena) (hspan : off + 8 ≤ span) (m : String) :
    Sem.run cfg [.sc .i64 (regionBase .arena + UInt64.ofNat off)] w (emit LibLmdb.init [.i64]) ≠ .misuse m ∧
    Sem.run cfg [.sc .i64 (regionBase .arena + UInt64.ofNat off)] w (emit LibLmdb.init [.i64]) ≠ .fault m := by
  check_lib [AV.room .arena 8]
  exact ⟨AV.room_desc (by decide) (by decide) hroom hspan, trivial⟩

/-- The body of a library function that answers a status. -/
abbrev status (b : StatusBody) (params : List ClifTy) : Code := (runAns b params).2.1

/-- What the functions handed a context ask of it: null, or a live heap block
    of the context's size. -/
theorem ctx_desc {Γ : Env} {w : World} {s : UInt64} {n : Nat} (hs : s = 0 ∨ Contracts.Blk w s n) :
    (AV.blk n false).Desc Γ w (.sc .i64 s) :=
  ⟨s, rfl, hs.imp (fun h => ⟨rfl, h⟩) id⟩

set_option maxRecDepth 200000 in
/-- **`htCreate`**, handed the table's context or null, counts one more
    handle in it. -/
theorem htCreate_no_misuse (cfg : Cfg) (s : UInt64) (w : World) (h : Quiet w)
    (hs : s = 0 ∨ Contracts.Blk w s LibHt.SIZE) (m : String) :
    Sem.run cfg [.sc .i64 s] w (status LibHt.create [.i64]) ≠ .misuse m ∧
    Sem.run cfg [.sc .i64 s] w (status LibHt.create [.i64]) ≠ .fault m := by
  check_lib [AV.blk LibHt.SIZE false]
  exact ⟨ctx_desc hs, trivial⟩

set_option maxRecDepth 200000 in
/-- **`htCount`**, handed the table's context or null, reads how many
    entries it holds. -/
theorem htCount_no_misuse (cfg : Cfg) (s : UInt64) (w : World) (h : Quiet w)
    (hs : s = 0 ∨ Contracts.Blk w s LibHt.SIZE) (m : String) :
    Sem.run cfg [.sc .i64 s] w (status LibHt.count [.i64]) ≠ .misuse m ∧
    Sem.run cfg [.sc .i64 s] w (status LibHt.count [.i64]) ≠ .fault m := by
  check_lib [AV.blk LibHt.SIZE false]
  exact ⟨ctx_desc hs, trivial⟩

end LibSafe

#print axioms LibSafe.threadInit_no_misuse
#print axioms LibSafe.lmdbInit_no_misuse
#print axioms LibSafe.htCreate_no_misuse
#print axioms LibSafe.htCount_no_misuse
