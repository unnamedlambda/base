import GenSurface
import LeanEvalAlgorithm

/-!
  # What the Lean-eval artifact's claims rest on — computed, not documented

  **One storage claim and the emitted program, and that is the report.**  What
  this generator ships is an evaluator; nothing here states what it evaluates.
-/

open Lean

namespace LeanEvalScan

/-- **The public claims.**

    `memMap_ok` — the regions this generator hands out do not overlap, by
    `decide`.  There is no `memMap_within` companion: this artifact's memory is
    sized from the payload it builds rather than fixed in advance, so
    disjointness is the whole of what a static map can say.

    `clifIrSource` is the shipped program, scanned so the report is the base the
    artifact actually reaches rather than the base of the one theorem. -/
def roots : List Name :=
  [ `LeanEval.memMap_ok
  , `LeanEval.clifIrSource ]

def notYetStated : List String :=
  [ "that the evaluator computes what Lean's own evaluator computes",
    "that its parser accepts the language it claims to accept",
    "that memory sized from the payload is large enough for the payload — the \
     sizing is arithmetic in the generator, not a theorem",
    "and therefore: correctness rests on differential testing against Lean" ]

end LeanEvalScan

open TrustScan LeanEvalScan in
#eval runGenScan "leaneval" roots

#eval do
  IO.println s!"[leaneval] roots scanned: {LeanEvalScan.roots.length}"
  for s in LeanEvalScan.notYetStated do
    IO.println s!"[leaneval] NOT STATED: {s}"
