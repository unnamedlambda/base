import Lean
import Std
import AlgorithmLib.ML.Model.Frontend
import AlgorithmLib.ML.Machine.Geometry
import AlgorithmLib.ML.Launch.HostBridge
import AlgorithmLib.Surface.Layout
import AlgorithmLib.ML.Math.Backprop
import AlgorithmLib.ML.Math.Quant
import AlgorithmLib.ML.Kernel.BodyCheck
open AlgorithmLib AlgorithmLib.ML

namespace NonVacuity

/-- A one-input identity spec, for instantiating the map-shaped theorems. -/
def idE : Expr 1 := .var ⟨0, by decide⟩

/-- Two chained map stages, concrete buffers and grid. -/
example (st : WSt) (l : Lane) :
    ((mapStage idE (fun _ => (1 : Buf)) 2 4 (fun _ => (by decide : (1:Buf) ≠ 2))).run
      ((mapStage idE (fun _ => (0 : Buf)) 1 4 (fun _ => (by decide : (0:Buf) ≠ 1))).run st)).mem 2
        (3 * 32 + l.val)
      = denote (fun _ => denote (fun _ => st.mem 0 (3 * 32 + l.val)) idE) idE :=
  two_map_stages idE idE 0 1 2 (by decide) (by decide) 4 st 3 l (by decide)

/-- A map feeding a reduction — the backward-pass shape — with the coverage
    obligation actually discharged. -/
example (st : WSt) :
    ((reduceStage 1 3 (stride32 (.lit 0)) (stride32 (.lit 0)) 2 4 8
        (by decide) (by decide)).run
      ((mapStage idE (fun _ => (0 : Buf)) 1 4 (fun _ => (by decide : (0:Buf) ≠ 1))).run st)).mem 2 5
      = bflyFold (dotStridedLane
          (fun a => denote (fun _ => st.mem 0 a) idE) (st.mem 3)
          (fun i l => (stride32 (.lit 0)).eval 5 i l)
          (fun i l => (stride32 (.lit 0)).eval 5 i l) 4) ⟨0, by decide⟩ :=
  map_then_reduce idE (fun _ => (0 : Buf)) 1 3 2 _ _
    (fun _ => (by decide : (0:Buf) ≠ 1)) (by decide) (by decide) (by decide)
    4 4 8 st 5 (by decide)
    (fun i hi l => ⟨i, hi, l, by show i * 32 + l.val = 0 + (i * 32 + l.val); omega⟩)

/-- Block-order independence, at concrete permuted lists. -/
example (st : WSt) :
    (runBlocks (mapStage (R := Float32) idE (fun _ => (0 : Buf)) 1 4
        (fun _ => (by decide : (0:Buf) ≠ 1))).blk [0,1,2] st).mem
      = (runBlocks (mapStage (R := Float32) idE (fun _ => (0 : Buf)) 1 4
        (fun _ => (by decide : (0:Buf) ≠ 1))).blk [2,0,1] st).mem :=
  runBlocks_perm_invariant _ (mapStage_exclusive _ _ _ _ _)
    (mapStage_idempotent _ _ _ _ _) _ _
    (by intro c h; show c < 4; simp at h; omega)
    (by intro c; constructor <;> (intro h; simp at h ⊢; omega)) st

example (env : Fin (layerCtx 4) → Int) (j : Fin 4) :
    denote env (sderiv (stdLoss 4) (stdXv 4 j))
      = denote env (.sum 4 (fun i => .mul (stdAdj 4 i) (varMat (stdWv 4) i j))) :=
  stdLayer_dx env j

example (env : Fin (layerCtx 4) → Int) (i j : Fin 4) :
    denote env (sderiv (stdLoss 4) (stdWv 4 i j))
      = denote env (.mul (stdAdj 4 i) (.var (stdXv 4 j))) :=
  stdLayer_dW env i j

/-- A kernel that stores to buffer 1 and reads buffer 0. -/
def okKernel : EWStmt := .seq (.loadIdx 0 0 elemIx) (.storeLane 1 elemIx 0)

/-- The same, but reading what it writes — the RoPE shape. -/
def inPlaceKernel : EWStmt := .seq (.loadIdx 0 1 elemIx) (.storeLane 1 elemIx 0)

/-- A kernel naming a buffer past a 2-entry binding table. -/
def outOfTableKernel : EWStmt := .seq (.loadIdx 0 7 elemIx) (.storeLane 1 elemIx 0)

example : okKernel.StageEligibleB 1 = true := by decide
/-- Rejects a kernel that reads its own output. -/
example : inPlaceKernel.StageEligibleB 1 = true := by decide
-- …but it is not idempotent, which is the property re-running needs.
example : inPlaceKernel.IdempotentEligibleB 1 = false := by decide
example : okKernel.IdempotentEligibleB 1 = true := by decide
/-- **Rejects a buffer the kernel never writes** — the G13 failure. -/
example : okKernel.StageEligibleB 5 = false := by decide
/-- …and one it only reads. -/
example : okKernel.StageEligibleB 0 = false := by decide

example : okKernel.BufBelow 2 := by decide
/-- Rejects a kernel naming a buffer outside the table. -/
example : ¬ outOfTableKernel.BufBelow 2 := by decide

/-- A kernel every lane of which stores the same word. -/
def racyKernel : EWStmt := .seq (.loadIdx 0 0 elemIx) (.storeLane 1 (.lit 0) 0)

example : BodyCheck.launchOk 2 4 1 (fun _ => 1024) okKernel = true := by native_decide
/-- Rejects a launch whose lanes store one word. -/
example : BodyCheck.launchOk 2 4 1 (fun _ => 1024) racyKernel = false := by native_decide
/-- Rejects a launch whose last block stores past its buffer. -/
example : BodyCheck.launchOk 2 64 1 (fun _ => 1024) okKernel = false := by native_decide

/-- A packed layout, and one with a gap the weaker `okB` would accept. -/
example : AlgorithmLib.Layout.RegionMap.packedB 0
    [⟨"a", 0, 4⟩, ⟨"b", 4, 8⟩] = true := by decide
/-- Rejects a gap. -/
example : AlgorithmLib.Layout.RegionMap.packedB 0
    [⟨"a", 0, 4⟩, ⟨"b", 8, 8⟩] = false := by decide
/-- Rejects a reordering — which `okB` accepts, since the regions stay
    disjoint.  This is why the packed check exists. -/
example : AlgorithmLib.Layout.RegionMap.okB
    [⟨"b", 4, 8⟩, ⟨"a", 0, 4⟩] = true := by decide
example : AlgorithmLib.Layout.RegionMap.packedB 0
    [⟨"b", 4, 8⟩, ⟨"a", 0, 4⟩] = false := by decide

example (st : WSt) (l0 : Lane) : ∃ _s0 : WSt, True := ⟨st, trivial⟩

end NonVacuity
