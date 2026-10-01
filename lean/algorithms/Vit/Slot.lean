import Vit.Units
open AlgorithmLib AlgorithmLib.ML

/-!
  # The slot each launch names holds that group's own kernel

  Its own module because of what it costs: deciding it renders every group's
  kernel text, which is the same work the generator does to emit them — 39 s.
-/

namespace Vit
open AlgorithmLib.IR

-- ---------------------------------------------------------------------------
-- The slot a launch reads holds the kernel that group emits
-- ---------------------------------------------------------------------------

/-- **Seam guard: the slot each launch names holds that group's own kernel.**

    `vit_capture_records_the_forward` recovers, from the emitted instruction
    stream, that launch `k` reads PTX at `vSlotOff (vSlotIx k)`.  That is an
    offset into a table; this is what the table holds there.

    The two are separated by a dedup — `vSlotMap` files groups under a
    `Std.HashMap` keyed on `(nbufs, warps, EWStmt)`, which is what makes twelve
    identical blocks twelve references to one kernel.  A key collision would
    point a launch at another group's kernel while every other guard in this
    file still passed, since each of them is about a unit rather than about the
    correspondence between units and slots.  This closes that by comparing the
    rendered texts, so the dedup's key does not have to be trusted to be
    injective — only its result checked.

    Stated over every launch, contractions included: those render no kernel and
    both sides are the empty string. -/
theorem vit_slot_holds_the_kernel :
    ((List.range VNUNIT).zip vUnits).all
      (fun (k, u) => vPtx.getD (vSlotIx k) "" == vTextOf u) = true := by
  native_decide


end Vit
