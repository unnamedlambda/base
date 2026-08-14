import VitModel
import AlgorithmLib.ML.RegBound
import AlgorithmLib.ML.Interchange
open AlgorithmLib AlgorithmLib.ML

/-!
  # The shipped ViT tape, described

  Definitions only — what a group is, what a launch key is, what makes grouping
  sound.  The two modules that decide things about them (`VitGuards`,
  `VitDag`) import this one and each other's build does not wait on it, which is
  what keeps a `native_decide` off the critical path of the other.
-/

namespace Vit
open AlgorithmLib.IR

/-- **The groups an emitted kernel comes from.**

    A group whose operation is a contraction prints no kernel — cuBLAS performs
    it — so it has no statement to guard and its `flenOf` (thirty instructions
    per row of the batch) would not fit a budget it never has to meet. -/
def vProvenUnits : List (List Nat) :=
  vUnits.filter (fun u => ((vUnitOps u).head? >>= vGemmOf).isNone)

open AlgorithmLib.ML in
/-- The group kernel is the emitted one, by definition rather than by
    inspection: `vTextOf` prints `vUnitStmt`, and this is that statement. -/
theorem vit_unit_is_group (u : List Nat) :
    vUnitStmt u = AlgorithmLib.ML.TOp.groupStmt SQ (vUnitBufs u) (vUnitOps u) := rfl

open AlgorithmLib.ML in
/-- Every member's buffers are in the group's table — because the table *is*
    the members' buffers, so this costs nothing per group. -/
theorem vit_unit_covers (u : List Nat) :
    ∀ op ∈ vUnitOps u, ∀ c ∈ (AlgorithmLib.ML.TOp.stmt SQ op).bufsOf, c ∈ vUnitBufs u := by
  intro op hop c hc
  exact (AlgorithmLib.ML.mem_eraseDups _ _).mpr
    (List.mem_flatMap.mpr ⟨op, hop, AlgorithmLib.ML.TOp.stmt_bufsOf SQ op hc⟩)

open AlgorithmLib.ML in
/-- **Whether one group's register budget fits.**

    One comparison, on widths the operations already carry: the group's largest
    `regHi` and the sum of its members' instruction counts.  Checking a tape
    this way builds no instruction, which is what makes it scale. -/
def vBudgetOk (u : List Nat) : Bool :=
  decide (AlgorithmLib.ML.TOp.groupBudget (vUnitOps u)
            ≤ AlgorithmLib.ML.PTX_ADDR_SCRATCH)

/-- **The groups the arithmetic budget does not cover**, and why they exist.

    A bias gradient is `Σ_s d[s][i]` — an `outer` of width one against the ones
    vector — and `outer` sums over the batch with the row index a Lean function
    rather than an `IdxE`, so it is *unrolled*: two hundred rows become 2414
    instructions.  The budget bounds register numbers by instruction count, so
    these fail it by a factor of seven while every other emitted kernel comes in
    at 1011 of 1020.

    They are checked on their instructions instead.  Naming them is the point:
    the guard is not weakened to fit them, and the list is a measurement of how
    far the cheap check reaches. -/
def vBigUnits : List (List Nat) := vProvenUnits.filter (fun u => !vBudgetOk u)

/-- **The primitive every contraction is issued through.**

    The value rather than its name, so the laws it assumes and the guarantees it
    withholds are read off what was lowered instead of recorded beside it. -/
def vDeclaredKernel : AlgorithmLib.ML.VendorKernel :=
  .cublasSgemmAt1OnStream

/-- Launches whose effect is assumed, and launches whose effect is a kernel this
    development emits. -/
def vDeclaredLaunches : Nat := (vFused.filter (fun op => (vGemmOf op).isSome)).length
def vProvenLaunches : Nat := vProvenUnits.length

/-- The bound the model's scores and row sums stay inside.  A score is a dot
    product of 64 normed values scaled by `1/8`; the row sum is at least one and
    at most the key count. -/
def VBOUND : Float32 := 1000000.0

-- ---------------------------------------------------------------------------
-- The law bill
-- ---------------------------------------------------------------------------

/-- What one launch is, as much of it as a scan of the emitted code can see:
    which FFI symbol, from which PTX slot, with how many buffers, over what
    grid and block. -/
abbrev VLaunchKey := String × Option Int × Option Int × Option Int × Option Int

/-- **The launches a range of the tape should make**, derived from the tape
    rather than transcribed from the emitted code. -/
def vIssueKeys (lo hi : Nat) (onStream : Bool) : List VLaunchKey :=
  ((List.range VNUNIT).zip vUnits).flatMap (fun (k, u) =>
    if lo ≤ vUnitLo u && vUnitHi u < hi then
      match (vUnitOps u).head? >>= vGemmOf with
      | some _ =>
          if onStream then
            [("cl_cublas_sgemm_strided_batched_on_stream", none, none, none, none)]
          else
            (vUnitOps u).filterMap (fun op => (vGemmOf op).map (fun _ =>
              ("cl_cublas_sgemm_strided_batched", none, none, none, none)))
      | none =>
          let w := vWarpsOf (vUnitGrid u)
          [(if onStream then "cl_cuda_launch_on_stream" else "cl_cuda_launch",
            some (Int.ofNat (vSlotOff (vSlotIx k))),
            some (Int.ofNat (vUnitBufs u).length),
            some (Int.ofNat (vUnitGrid u / w)),
            some (Int.ofNat (32 * w)))]
    else [])

def vWarmKey : VLaunchKey :=
  ("cl_cublas_sgemm_strided_batched", none, none, none, none)

/-- The whole launch sequence of a DAG capture: the eager warm-up pass that
    makes every module resident, then the same launches on the pooled streams
    between `beginCapture` and `endCapture`. -/
def vCaptureKeys (lo hi : Nat) : List VLaunchKey :=
  vWarmKey :: (vIssueKeys lo hi false ++ vIssueKeys lo hi true)

-- ---------------------------------------------------------------------------
-- What makes a group of operations one kernel
-- ---------------------------------------------------------------------------

/-! A launch here is a group, and grouping is only sound under conditions the
    grouping pass arranges but never stated.  Take any of them away and one
    kernel is not the members run in order: two blocks would write the same
    address, or a block would read a row another block owns and has not
    necessarily reached.

    All four are decidable over the shipped tape, and all four are checked. -/

/-- Every member of a group runs at the grid the group is launched over. -/
def vGroupGrids : Bool :=
  vUnits.all (fun u => (vUnitOps u).all (fun op =>
    AlgorithmLib.ML.TOp.gridOf op == vUnitGrid u))

/-- **The window an operation writes**: buffer, row pitch, column offset,
    width.  A row pass writes a *window* of its output — three of them are how a
    concatenation lands — so the buffer alone is not what has to be disjoint.
    Anything else is treated as writing all of its buffer. -/
def vWriteWin (op : AlgorithmLib.ML.TOp) : Ref × Nat × Nat × Nat :=
  match op with
  | .ziprow _ _ o _ _ _ n k w _              => (o, n, k, w)
  | .ziprow3 _ _ _ o _ _ _ _ n k w _         => (o, n, k, w)
  | .ziprow4 _ _ _ _ o _ _ _ _ _ n k w _     => (o, n, k, w)
  | _ => ((AlgorithmLib.ML.TOp.outSize SQ op).1, 1, 0, 1)

/-- Two windows that cannot touch: different buffers, or the same pitch and
    non-overlapping columns. -/
def vWinDisjoint (a b : Ref × Nat × Nat × Nat) : Bool :=
  a.1 != b.1
    || (a.2.1 == b.2.1
        && (decide (a.2.2.1 + a.2.2.2 ≤ b.2.2.1) || decide (b.2.2.1 + b.2.2.2 ≤ a.2.2.1)))

/-- No two members of a group write the same place. -/
def vGroupWritesDisjoint : Bool :=
  vUnits.all (fun u =>
    let ws := (vUnitOps u).map vWriteWin
    (List.range ws.length).all (fun i =>
      (List.range ws.length).all (fun j =>
        i == j || vWinDisjoint (ws.getD i (0,0,0,0)) (ws.getD j (0,0,0,0)))))

/-- Every member of a group keeps to its own chunk: the rows a block owns are
    the only rows it reads and the only rows it writes.

    The condition belongs to *grouping*, so it is asked of groups with more
    than one member: a lone operation launched as its own kernel has nothing to
    interleave with.  Two kinds of operation are always alone for exactly that
    reason — a contraction, which is cuBLAS, and the unrolled `outer` a bias
    gradient is, which sums over the whole batch rather than one chunk. -/
def vGroupChunkLocal : Bool :=
  vProvenUnits.all (fun u =>
    decide ((vUnitOps u).length ≤ 1) || (vUnitOps u).all vChunkLocal)

/-- **No member's *shared* read hits an earlier member's write.**

    Reading a group-mate's output is not the hazard — it is the point: a member
    reads its own chunk of what the member before it wrote, in the same block,
    in program order, and that is what makes one kernel out of a chain of row
    passes.  `vChunkLocal` is what confines it to that chunk.

    The hazard is a read *across* rows — a gain, a rotation table, the padding
    mask, anything a `BCast` marks `sharedAt` or `constAt`.  If an earlier
    member of the same group wrote that buffer, a block would read a row another
    block owns and has not necessarily reached.  So it is these reads, and only
    these, that must miss the group's earlier writes. -/
def vGroupSharedReadsOk : Bool :=
  vProvenUnits.all (fun u =>
    let ops := vUnitOps u
    (List.range ops.length).all (fun i =>
      (vSharedReads (ops.getD i (.rowsq 0 0 0 0))).all (fun r =>
        ((List.range i).map (fun j =>
          (AlgorithmLib.ML.TOp.outSize SQ (ops.getD j (.rowsq 0 0 0 0))).1)).all
            (fun w => w != r))))

