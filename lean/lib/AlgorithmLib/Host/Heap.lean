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

end AlgorithmLib.HProg.Contracts

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Contracts

namespace AlgorithmLib.LibSpec

/-- The eight bytes at `j` in the block `s` starts. -/
def fld (w : World) (s : UInt64) (j : Nat) : Option UInt64 := w.mem.load (s + UInt64.ofNat j) 8

/-- A world with no worker running and no copy in flight: a heap block's
    bytes can be read and written. -/
def Calm (w : World) : Prop := w.mem.frozen = false ∧ w.mem.busy = []

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

end AlgorithmLib.LibSpec
