import ScanCore

/-!
  # The surface the utility generators' scans run at

  These generators ship an artifact and prove almost nothing about it, so their
  scans exist to report that rather than to certify it.  The surface is
  therefore as narrow as it can be: two opaque constants, both from Lean's own
  `String`, and nothing else.

  Kept in one module because the six scanners are separate only for reasons
  that have nothing to do with the surface — each generator defines its own
  `main`, and `SatAlgorithm`/`Sha256Algorithm` additionally share
  `namespace Algorithm` — so there is no reason to state the allowance six
  times and every reason not to.
-/

open Lean

namespace TrustScan

/-- **Lean's string representation, not an assumption about any algorithm.**

    Reached because a scanned root is the emitted *program*, which contains
    string literals — kernel names, PTX text, the CLIF function names.  Neither
    constant says anything about what the generator computes, and a scan that
    hid them would be reporting a smaller base than the artifact has.

    Anything else these scans reach is a real widening and must be added here,
    where the diff is reviewable. -/
def genAllowedOpaque : List Name := [`Lean.opaqueId, `String.Internal.append]

/-- The utility generators' surface: no allowed hypotheses, no obligations,
    and the two string opaques above. -/
def genSurface : Surface := { allowedOpaque := genAllowedOpaque }

/-- Their scan, at `genSurface`. -/
def runGenScan (label : String) (roots : List Name) (nativeRoster : List Name) : CoreM Unit :=
  runScanWith genSurface label roots nativeRoster

end TrustScan
