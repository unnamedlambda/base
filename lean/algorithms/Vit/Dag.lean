module
public import Vit.Units
meta import Vit.Units
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open AlgorithmLib AlgorithmLib.ML

/-!
  # The groups the launches are made of, checked

  Its own module so that it runs beside `VitGuards` rather than after it.
-/

namespace Vit
open AlgorithmLib.IR

/-- **Every group is one kernel's worth**: each group's operations share a
    grid, read only their own chunk, write apart from one another, and read
    what another member writes only where that is allowed. -/
theorem vit_groups_sound :
    vGroupGrids = true ∧ vGroupChunkLocal = true
      ∧ vGroupWritesDisjoint = true ∧ vGroupSharedReadsOk = true := by native_decide

end Vit
