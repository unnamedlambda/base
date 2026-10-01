import Scan.GenSurface
import Demo.Sha256

/-!
  # What the SHA-256 artifact's claims rest on — computed, not documented

  A scanner per generator; this one cannot share a module with `SatScan` for a
  second reason beyond `main`, which is that both generators use
  `namespace Algorithm`.

  **This one names no theorem, and that is the report.**  What it names instead
  is the shipped program itself, so the scan reports the trusted base the
  emitted artifact actually reaches rather than the base of a theorem someone
  chose to write.
-/

open Lean

namespace Sha256Scan

/-- **The shipped value, not a claim about it.**

    `clifIrSource` is the program that becomes the artifact.  Scanning it walks
    the closure of what the generator *emits*, which is the honest thing to
    report for a generator that proves nothing about its own output.

    Storage is the exception, and it is handled by construction rather than by
    a theorem: this generator allocates through `Layout.build`, which hands out
    offsets in sequence, so two fields cannot collide.  That is why there is no
    `memMap_ok` here and why its absence is not a gap — the stronger form is
    already in force.  The hand-written maps elsewhere need the theorem exactly
    because they are hand-written. -/
def roots : List Name :=
  [ `Algorithm.clifIrSource ]

/-- **What this artifact does not state.** -/
def notYetStated : List String :=
  [ "that the digest it computes is SHA-256 — no reference to the standard's \
     compression function appears in any statement here",
    "that the message schedule and padding are the specification's",
    "and therefore: correctness rests entirely on differential testing against \
     a reference implementation over the test corpus" ]

end Sha256Scan

open TrustScan Sha256Scan in
#eval runGenScan "sha256" roots []

#eval do
  IO.println s!"[sha256] roots scanned: {Sha256Scan.roots.length} (the emitted program, not a theorem)"
  for s in Sha256Scan.notYetStated do
    IO.println s!"[sha256] NOT STATED: {s}"
