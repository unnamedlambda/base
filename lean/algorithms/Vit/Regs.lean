module
public import Vit.Units
meta import Vit.Units
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open AlgorithmLib AlgorithmLib.ML

/-!
  # No emitted kernel names a register the emitter reserved

  Its own module because of what it costs: the arithmetic budget covers every
  kernel but the 254 unrolled bias gradients, and those have to be checked on
  their instruction lists — 27 s of building `flatKernel` for kernels of 2414
  instructions each.  Beside `VitGuards` rather than inside it.
-/

namespace Vit
open AlgorithmLib.IR

open AlgorithmLib.ML in
/-- The two facts this module decides: that a group's table stays below the
    address scratch, and that the kernels no arithmetic bound reaches name no
    reserved register either. -/
theorem vit_regs_checked :
    vProvenUnits.all (fun u =>
      decide ((vUnitBufs u).length ≤ AlgorithmLib.ML.PTX_ADDR_SCRATCH)) = true
      ∧ vBigUnits.all (fun u => AlgorithmLib.ML.FlatRegsOkB
          (AlgorithmLib.ML.flatKernel (AlgorithmLib.ML.expandEW (vUnitStmt u)))) = true := by
  native_decide

open AlgorithmLib.ML in
/-- **Seam guard: no group's table can reach the address scratch.**  A kernel's
    parameter list is the group's table, and the widest group here binds
    seventeen. -/
theorem vit_group_tables :
    vProvenUnits.all (fun u =>
      decide ((vUnitBufs u).length ≤ AlgorithmLib.ML.PTX_ADDR_SCRATCH)) = true :=
  vit_regs_checked.1

open AlgorithmLib.ML in
/-- **Seam guard: no unrolled kernel names a reserved register.**  Decided on
    the instruction list, because no arithmetic bound covers these. -/
theorem vit_regs_ok_big :
    vBigUnits.all (fun u => AlgorithmLib.ML.FlatRegsOkB
      (AlgorithmLib.ML.flatKernel (AlgorithmLib.ML.expandEW (vUnitStmt u)))) = true :=
  vit_regs_checked.2

open AlgorithmLib.ML in
/-- **Seam guard: no emitted kernel names a register the emitter reserved.**

    The hazard is real in this development: a `%rd60`/`%rd61` collision in this
    printer made two of ninety kernels fail to load with a sticky illegal
    address, and it was found by running them, not by a proof.

    Two routes to the same conclusion — the arithmetic budget where it reaches,
    the instruction list where it does not — so the cheap check covers what it
    covers and nothing is assumed about the rest. -/
theorem vit_regs_ok :
    vProvenUnits.all (fun u => AlgorithmLib.ML.FlatRegsOkB
      (AlgorithmLib.ML.flatKernel (AlgorithmLib.ML.expandEW (vUnitStmt u)))) = true :=
  List.all_eq_true.mpr (fun u hu => by
    by_cases h : vBudgetOk u = true
    · refine AlgorithmLib.ML.flatKernel_regsOk_of_B _ ?_
      rw [vit_unit_is_group u]
      exact AlgorithmLib.ML.TOp.groupRegsOk SQ (vUnitBufs u) (vUnitOps u)
        (of_decide_eq_true (List.all_eq_true.mp vit_group_tables u hu))
        (vit_unit_covers u)
        (of_decide_eq_true h)
    · exact List.all_eq_true.mp vit_regs_ok_big u
        (List.mem_filter.mpr ⟨hu, by simp [h]⟩))

open AlgorithmLib.ML in
/-- **Seam guard: every branch target in every emitted kernel exists.** -/
theorem vit_targets :
    vProvenUnits.all (fun u => AlgorithmLib.ML.FlatTargetsOkB
      (AlgorithmLib.ML.flatKernel (AlgorithmLib.ML.expandEW (vUnitStmt u)))) = true :=
  List.all_eq_true.mpr (fun u _ => AlgorithmLib.ML.flatKernel_targets _)

open AlgorithmLib.ML in
/-- **Seam guard: every emitted kernel is printable** — no instruction the
    emitter would have to guess at. -/
theorem vit_printable :
    vProvenUnits.all (fun u => AlgorithmLib.ML.FlatPrintableB
      (AlgorithmLib.ML.flatKernel (AlgorithmLib.ML.expandEW (vUnitStmt u)))) = true :=
  List.all_eq_true.mpr (fun u _ => AlgorithmLib.ML.flatKernel_printable _)

open AlgorithmLib.ML in
/-- **Seam guard: every emitted kernel's buffers are inside its own parameter
    list.**  `emitProvenKernelN` declares `(vUnitBufs u).length` parameters; a
    statement naming a higher slot would print a load from a register the
    kernel never declared. -/
theorem vit_bufs_bound :
    vProvenUnits.all (fun u =>
      decide ((vUnitStmt u).BufBelow (vUnitBufs u).length)) = true :=
  List.all_eq_true.mpr (fun u _ => decide_eq_true (by
    refine AlgorithmLib.ML.EWStmt.bufBelow_of_bufsOf _ _ ?_
    intro b hb
    rw [vit_unit_is_group u, AlgorithmLib.ML.TOp.groupStmt,
        AlgorithmLib.ML.groupStmt_bufsOf] at hb
    simp only [AlgorithmLib.ML.EWStmt.bufsOf, List.nil_append, List.mem_flatMap,
               List.mem_map] at hb
    obtain ⟨op, hop, c, hc, rfl⟩ := hb
    exact AlgorithmLib.ML.compactMap_lt _ c (vit_unit_covers u op hop c hc)))

/-- **…and the register guard has something to reject.**  A check that passes on
    everything says nothing; this is the instruction it exists for, writing the
    register the emitter's address arithmetic uses as scratch. -/
theorem vit_regs_ok_rejects_the_scratch :
    AlgorithmLib.ML.FlatRegsOkB
      [AlgorithmLib.ML.FI.si (.movIC AlgorithmLib.ML.PTX_ADDR_SCRATCH 0)] = false := by
  decide


end Vit
