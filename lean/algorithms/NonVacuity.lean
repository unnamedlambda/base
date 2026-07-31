import Lean
import Std
import AlgorithmLib
import BackwardWideAlgorithm

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
    (runBlocks (mapStage idE (fun _ => (0 : Buf)) 1 4
        (fun _ => (by decide : (0:Buf) ≠ 1))).blk [0,1,2] st).mem
      = (runBlocks (mapStage idE (fun _ => (0 : Buf)) 1 4
        (fun _ => (by decide : (0:Buf) ≠ 1))).blk [2,0,1] st).mem :=
  runBlocks_perm_invariant _ (mapStage_exclusive _ _ _ _ _)
    (mapStage_idempotent _ _ _ _ _) _ _
    (by intro c h; show c < 4; simp at h; omega)
    (by intro c; constructor <;> (intro h; simp at h ⊢; omega)) st

/-- At the first block. -/
example (st : WSt) : (BackwardWide.dxStage.run (BackwardWide.sbStage.run st)).mem
    BackwardWide.dxB 0
      = bflyFold (dotStridedLane
          (fun a => denote (fun i => st.mem (BackwardWide.siluBwdIn i) a)
            BackwardWide.siluBwdSpec)
          (st.mem BackwardWide.wB)
          (fun i l => BackwardWide.adjIx.eval 0 i l)
          (fun i l => BackwardWide.wIx.eval 0 i l) BackwardWide.K) ⟨0, by decide⟩ :=
  BackwardWide.bwd_chain st 0 (by decide)

/-- And at the last, where an off-by-one vacuity would hide. -/
example (st : WSt) : (BackwardWide.dxStage.run (BackwardWide.sbStage.run st)).mem
    BackwardWide.dxB 895
      = bflyFold (dotStridedLane
          (fun a => denote (fun i => st.mem (BackwardWide.siluBwdIn i) a)
            BackwardWide.siluBwdSpec)
          (st.mem BackwardWide.wB)
          (fun i l => BackwardWide.adjIx.eval 895 i l)
          (fun i l => BackwardWide.wIx.eval 895 i l) BackwardWide.K) ⟨0, by decide⟩ :=
  BackwardWide.bwd_chain st 895 (by decide)

example (env : Fin (layerCtx 4) → Int) (j : Fin 4) :
    denote env (sderiv (stdLoss 4) (stdXv 4 j))
      = denote env (.sum 4 (fun i => .mul (stdAdj 4 i) (varMat (stdWv 4) i j))) :=
  stdLayer_dx env j

example (env : Fin (layerCtx 4) → Int) (i j : Fin 4) :
    denote env (sderiv (stdLoss 4) (stdWv 4 i j))
      = denote env (.mul (stdAdj 4 i) (.var (stdXv 4 j))) :=
  stdLayer_dW env i j

/-- A real execution chain exists and lands where `runGrid` says. -/
example (m : MState) : ∃ m' : MState,
    gridRuns (flatKernel (expandEW BackwardWide.siluBwd.ew)) 28 m m'
      ∧ m'.toWSt = runGrid (expandEW BackwardWide.siluBwd.ew).elabIn 28 m.toWSt :=
  flatGrid_realises _ (expandEW_expFree _)
    (expandEW_idxFree _ (compileWKernel_idxFree _ _ _ _ _ _
      (fun _ => (by decide : IdxE.IregFree elemIx)) (by decide)))
    (expandEW_flat _ (compileWKernel_flat _ _ _ _ _ _)) 28 m

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

/-- A flat program whose branch target runs past the end. -/
def badTargets : List FI := [.si (.movIC 0 0), .jmp 99]
/-- …and one carrying an instruction the printer cannot render. -/
def badPrintable : List FI := [.si (.loop 0 (.lit 0) [])]

example : FlatTargetsOkB [FI.si (.movIC 0 0), FI.jmp 1] = true := by decide
/-- Rejects an out-of-range branch. -/
example : FlatTargetsOkB badTargets = false := by decide
example : FlatPrintableB [FI.si (.movIC 0 0)] = true := by decide
/-- Rejects an unrenderable instruction. -/
example : FlatPrintableB badPrintable = false := by decide

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
