import AlgorithmLib.ML.Math.Weave
import AlgorithmLib.ML.Launch.Pipeline

/-!
  # The kernels' broadcasts are reindexings

  `BCast` is how a shipped kernel addresses an operand: a row of a matrix, one
  value per row, a vector shared by every row, or a single constant address.
  Those four modes were chosen because they are what `IdxE` can express — the
  index language is affine in `(ctaId, loopI, laneId)` with no division — and
  not because of any algebra.

  This module shows they *are* an algebra: each mode is the action of an affine
  `Reindex` on the point `[row, column]`, read at row-major addresses.  So the
  broadcasting a kernel performs is the paper's reindexing morphism, and the
  laws proven in `Weave` are laws about the addresses these kernels touch.

  The target shape's row extent never appears — a row-major address is
  `row * cols + col` — so the statement is for every row extent, and only the
  column extent is pinned per mode.
-/

namespace AlgorithmLib.ML.Broadcast

/-- Row-major addressing of a rank-two shape. -/
theorem flatten_pair (R C x y : Nat) : flatten [R, C] [x, y] = x * C + y := by
  simp [flatten, Ix.size]

/-- The affine map a broadcast mode performs on `[row, column]`.

    Every mode is `Λ · [row, col] + v` with entries in `{0,1}`: the matrix
    selects which of the two coordinates survive, and the offset carries the
    mode's constant. -/
def bcastReindex {P Q : Ix} : BCast → Reindex P Q
  | .rowOf _ k  => { lin := [[1, 0], [0, 1]], off := [0, k] }
  | .scalar     => { lin := [[1, 0], [0, 0]], off := [0, 0] }
  | .sharedAt k => { lin := [[0, 0], [0, 1]], off := [0, k] }
  | .constAt k  => { lin := [[0, 0], [0, 0]], off := [0, k] }

/-- The column extent of the array each mode reads into. -/
def bcastCols (w : Nat) : BCast → Nat
  | .rowOf s _  => s
  | .scalar     => 1
  | .sharedAt _ => w
  | .constAt _  => w

/-- **A broadcast mode is a reindexing.**

    Sending `[row, col]` through the mode's affine map and reading the
    row-major address gives exactly the address the kernel reads.  This is the
    statement that makes the algebra's `pull` the same operation as a shipped
    kernel's operand addressing. -/
theorem bcast_is_reindex {P Q : Ix} (mode : BCast) (w R cta o : Nat) :
    flatten [R, bcastCols w mode] ((bcastReindex (P := P) (Q := Q) mode).apply [cta, o])
      = mode.ev cta o := by
  cases mode <;>
    simp [bcastReindex, bcastCols, Reindex.apply, matVec, dotN, vadd, flatten_pair,
      BCast.ev, Nat.add_comm, Nat.add_assoc]

/-- The data really is a rank-two-to-rank-two affine map: the matrix has one
    row per target axis and one column per source axis. -/
theorem bcastReindex_wf {P Q : Ix} (mode : BCast)
    (hP : P.length = 2) (hQ : Q.length = 2) :
    (bcastReindex (P := P) (Q := Q) mode).WF := by
  cases mode <;>
    exact ⟨by simp [bcastReindex, hQ], by simp [bcastReindex, hQ], by
      intro r hr
      simp only [bcastReindex, List.mem_cons, List.not_mem_nil, or_false] at hr
      rcases hr with h | h <;> simp [h, hP]⟩

end AlgorithmLib.ML.Broadcast
