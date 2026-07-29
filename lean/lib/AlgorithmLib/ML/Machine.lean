import AlgorithmLib.ML.Expr

/-!
  # Buffers and the stride decomposition

  What survives of the original sequential machine.  `Stmt`/`St`/`FExp` and
  their compiler are **gone**: they had an emitter-free existence — proofs about
  code that could never run — and the warp machine (`Warp.lean`) plus the proven
  lowering superseded them entirely.

  Kept here: the buffer identifier every layer shares, and the stride-cover
  lemmas, which state the one fact the thread decomposition rests on — that a
  stride-`s` sweep partitions `[0, n)` exactly.

  **Element-addressed buffers**, not bytes: `mem : Buf → Nat → Float32`.  The
  f32 byte layout is not modelled.
-/

namespace AlgorithmLib.ML

/-- Buffer identifier. -/
abbrev Buf := Nat

-- ---------------------------------------------------------------------------
-- The stride decomposition
-- ---------------------------------------------------------------------------

/-- Indices visited by thread `tid` in a `stride`-strided sweep of `[0, n)`. -/
def strideIdx (tid stride n : Nat) : List Nat :=
  (List.range n).filter (fun i => i % stride = tid % stride)

/-- **Every element is touched exactly once.**  The `strideLoop` emitted by
    `AlgorithmLib.PTX.strideLoop` partitions `[0, n)` across the block: index
    `i` belongs to thread `i % stride` and to no other.  This is the whole
    correctness content of the thread decomposition, and it is why the schema
    theorems below may reason with a single counted loop over `[0, n)`. -/
theorem strideCover (stride n i : Nat) (hi : i < n) (tid : Nat) :
    i ∈ strideIdx tid stride n ↔ i % stride = tid % stride := by
  simp [strideIdx, List.mem_filter, List.mem_range, hi]

end AlgorithmLib.ML
