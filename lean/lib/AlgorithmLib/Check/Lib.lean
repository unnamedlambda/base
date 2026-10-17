module
public import AlgorithmLib.Check.Trace
public import AlgorithmLib.Host.Hoare
public import AlgorithmLib.Host.Lifecycle
@[expose] public section

/-!
# `Check.Lib` — what the checker takes of the libraries a program is linked with

An entry point a library implements is a call to one of the program's own
functions (`Ffi.inLib`, `Sem.callOf`). The checker reads such a call by the
library function's statement, and its theorems hold of a program linked with
functions that meet those statements: `Linked lb cfg` says so of `cfg`'s
functions where `lb` says the checker read library calls, and nothing where it
did not.

`LibCall L f vs w Q`: the function `L` has at `f`'s place, called on `vs` from
`w`, neither misuses a call nor faults, and where it returns leaves a world
`Q` holds of. `LibStep w w' lo hi` is what such a call leaves of the rest: every
region's size outside pinned memory, no worker running and no copy in flight,
every cell outside pinned memory clear of the bytes from `lo` up to `hi`, and
the calls observed before it.

`HtOk L` is the hash table's functions, each from a slot that holds null or a
table (`HtSpec.HtObj`): `htInit` leaves an empty one at its slot, `htCleanup`
null, and `htCreate`, `htInsert` and `htLookup`, handed what the slot holds,
leave the table there, `htCreate` made once more and `htInsert` with values
as long as the one it stored.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (cellOk)

/-- **A call to the library function at `f`'s place does as `Q` says**: it
    neither misuses a call nor faults, and where it returns, the world it
    leaves is `Q`'s. -/
def LibCall (L : Locals) (f : Ffi) (vs : List V) (w : World) (Q : World → Prop) : Prop :=
  match L f.libSlot vs w with
  | .error e => FailOk (e.out : Outcome Env) false
  | .ok (_, w') => Q w'

/-- **What a call into a library leaves of the rest of the world**, having
    written, outside the heap, at most the bytes from `lo` up to `hi`. -/
structure LibStep (w w' : World) (lo hi : Nat) : Prop where
  rooms : ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r
  frozen : w'.mem.frozen = false
  idle : w'.mem.busy.isEmpty = true
  cell : ∀ c v, cellOk c = true → hi ≤ c.toNat ∨ c.toNat + 8 ≤ lo → w.mem.load c 8 = some v →
    w'.mem.load c 8 = some v
  calls : ∃ cs : List (Callee × List V), callsOf w'.obs = cs.reverse ++ callsOf w.obs

/-- A step that wrote inside a run wrote inside any run around it. -/
theorem LibStep.mono {w w' : World} {lo hi lo' hi' : Nat} (h : LibStep w w' lo hi) (hlo : lo' ≤ lo)
    (hhi : hi ≤ hi') : LibStep w w' lo' hi' :=
  ⟨h.rooms, h.frozen, h.idle, fun c v hc hm => h.cell c v hc (by omega), h.calls⟩

/-- **Room for `k` bytes at `p`, outside pinned memory.** -/
def NPRoom (w : World) (p : UInt64) (k : Nat) : Prop :=
  ∃ r off, decodeAddr p = some (r, off) ∧ r ≠ .pinned ∧ off + k ≤ w.mem.sizes r ∧ off + k ≤ 2 ^ 36

/-- **The hash table's functions meet their statements.** -/
structure HtOk (L : Locals) : Prop where
  init : ∀ (w : World) (x : UInt64), w.mem.frozen = false → w.mem.busy.isEmpty = true → cellOk x = true →
    NPRoom w x 8 →
    LibCall L .htInit [.sc .i64 x] w fun w' => HtSpec.HtObj w' x 0 0 ∧ LibStep w w' x.toNat (x.toNat + 8)
  cleanup : ∀ (w : World) (x : UInt64) (n k : Nat), w.mem.frozen = false → w.mem.busy.isEmpty = true →
    HtSpec.HtObj w x n k →
    LibCall L .htCleanup [.sc .i64 x] w fun w' =>
      w'.mem.load x 8 = some 0 ∧ LibStep w w' x.toNat (x.toNat + 8)
  create : ∀ (w : World) (x s : UInt64) (n k : Nat), w.mem.frozen = false → w.mem.busy.isEmpty = true →
    HtSpec.HtObj w x n k → w.mem.load x 8 = some s → k + 1 < 2 ^ 64 →
    LibCall L .htCreate [.sc .i64 s] w fun w' => HtSpec.HtObj w' x n (k + 1) ∧ LibStep w w' 0 0
  insert : ∀ (w : World) (x s kp kl vp vl : UInt64) (n k : Nat), w.mem.frozen = false →
    w.mem.busy.isEmpty = true → HtSpec.HtObj w x n k → w.mem.load x 8 = some s → kl.toNat < 2 ^ 32 →
    vl.toNat < 2 ^ 32 → NPRoom w kp kl.toNat → NPRoom w vp vl.toNat →
    LibCall L .htInsert [.sc .i64 s, .sc .i64 kp, .sc .i32 kl, .sc .i64 vp, .sc .i32 vl] w fun w' =>
      HtSpec.HtObj w' x (max n vl.toNat) k ∧ LibStep w w' 0 0
  lookup : ∀ (w : World) (x s kp kl rp : UInt64) (n k : Nat), w.mem.frozen = false →
    w.mem.busy.isEmpty = true → HtSpec.HtObj w x n k → w.mem.load x 8 = some s → kl.toNat < 2 ^ 32 →
    NPRoom w kp kl.toNat → NPRoom w rp n → rp.toNat + n ≤ x.toNat ∨ x.toNat + 8 ≤ rp.toNat →
    LibCall L .htLookup [.sc .i64 s, .sc .i64 kp, .sc .i32 kl, .sc .i64 rp] w fun w' =>
      HtSpec.HtObj w' x n k ∧ LibStep w w' rp.toNat (rp.toNat + n)

/-- **`cfg`'s functions are the libraries'**, where `lb` says the checker
    read library calls by the libraries' statements. -/
class Linked (lb : Bool) (cfg : Cfg) : Prop where
  ht : lb = true → HtOk cfg.locals

instance (cfg : Cfg) : Linked false cfg := ⟨fun h => nomatch h⟩

end AlgorithmLib.Check
