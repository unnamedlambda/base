module
public import AlgorithmLib.Host.Clif
meta import AlgorithmLib.Host.Clif
public import Lz4.Comp
meta import Lz4.Comp
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # What the emitted host program is known to do

  The calls the compression makes are `Lz4CompSafe.warp15_calls` and
  `warp16_calls` (`Ship.Lz4Comp`): every run of the emitted body that ends
  uploads the input into the allocation `load` made, launches the kernel
  `rLaunches` times on a grid of `numBlk` blocks of `modelBlockDim` threads,
  and copies the blocks back where the output has room. That is a statement
  about runs, proven by the checker (`AlgorithmLib.Check`), and the pattern
  `Lz4CompSafe.warpCalls` is written with the generator's own `rLaunches`,
  `WP.gridX` and `wBlockDim`, so a change to any of them breaks it.

  What that pattern does not say is here: both kernel parameters are one
  allocation.
-/

namespace Lz4Host

open Lz4Ship
open AlgorithmLib

/-- **Both parameters name allocation 0.** The layout writes each kernel's
    parameter cells from the slots of the allocations it is handed, so the
    kernel's `in_ptr` and `out_ptr` receive the *same* device pointer, and the
    output base can only be what the prologue computes from it. -/
theorem params_same_buffer (w : WP) : (lzApp w).kernels.map (·.args) = [[0, 0]] := rfl

end Lz4Host
