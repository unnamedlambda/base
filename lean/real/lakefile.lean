import Lake
open Lake DSL

-- The ℝ layer lives in its own package so that `AlgorithmLib` and every
-- algorithm build stay Mathlib-free (and fast).  Only this package pays for
-- Mathlib, and only proofs live here — nothing here is ever executed or
-- emitted into an Artifact.
require algorithmLib from "../lib"
require mathlib from git "https://github.com/leanprover-community/mathlib4" @ "v4.26.0"

package algorithmLibReal where
  srcDir := "."

@[default_target]
lean_lib AlgorithmLibReal
