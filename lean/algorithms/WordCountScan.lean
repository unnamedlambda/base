import GenSurface
import WordCountAlgorithm

/-!
  # What the word-count artifact's claims rest on — computed, not documented

  **Two storage claims and the emitted program, and that is the report.**
-/

open Lean

namespace WordCountScan

/-- **The public claims.**  That the regions do not overlap and that they fit
    the memory the generator asks for, both by `decide`, plus the shipped
    program so the report covers what the artifact reaches. -/
def roots : List Name :=
  [ `WordCountBench.memMap_ok
  , `WordCountBench.memMap_within
  , `WordCountBench.artifacts ]

def notYetStated : List String :=
  [ "that the count it produces is the number of words in the input, for any \
     definition of word — none is stated here",
    "that its notion of a separator matches the reference's",
    "and therefore: correctness rests on differential testing over the corpus" ]

end WordCountScan

open TrustScan WordCountScan in
#eval runGenScan "wordcount" roots

#eval do
  IO.println s!"[wordcount] roots scanned: {WordCountScan.roots.length}"
  for s in WordCountScan.notYetStated do
    IO.println s!"[wordcount] NOT STATED: {s}"
