import Lake
open Lake DSL

require algorithmLib from "../../lean"

package csv where
  srcDir := "lean"

@[default_target]
lean_exe generate where
  root := `MakeAlgorithm
  supportInterpreter := true
