import GenSurface
import SatAlgorithm

/-!
  # What the SAT artifact's claims rest on — computed, not documented

  A scanner per generator, for the reason the others give: each defines its own
  `main`, so they cannot share a module.  `SatAlgorithm` and `Sha256Algorithm`
  additionally share `namespace Algorithm`, so importing both is not possible
  even in principle.

  **This one is nearly empty, and that is the report.**  Two claims, both about
  storage.  The solver's *answer* is backed by differential testing against a
  reference, not by a theorem, and a scan over two roots does not say the
  artifact is trustworthy — it says exactly how much of it has been argued.
-/

open Lean

namespace SatScan

/-- **The public claims.**

    That the regions the generator hands out do not overlap, and that they fit
    the memory it asks for.  Both by `decide`, so they are checked rather than
    asserted, and both fail the build if an offset moves onto its neighbour.

    The one deliberate alias is recorded at the definition rather than here:
    `clauseIndex_off` *is* `out_off`, because the output string is built where
    the clause index sat once solving is done.  It appears in `memMap` under one
    name, which is why `okB` accepts it. -/
def roots : List Name :=
  [ `Algorithm.memMap_ok
  , `Algorithm.memMap_within ]

/-- **What this artifact does not state.** -/
def notYetStated : List String :=
  [ "that the assignment it prints satisfies the formula it read \
     (a checker over the emitted output would be the cheap form of this)",
    "that UNSAT means no assignment exists — nothing here produces a refutation",
    "that the CDCL loop terminates",
    "and therefore: every claim about the solver's answer rests on differential \
     testing against a reference solver, not on a proof" ]

end SatScan

open TrustScan SatScan in
#eval runGenScan "sat" roots

#eval do
  IO.println s!"[sat] roots scanned: {SatScan.roots.length}"
  for s in SatScan.notYetStated do
    IO.println s!"[sat] NOT STATED: {s}"
