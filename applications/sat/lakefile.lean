import Lake
open Lake DSL

require algorithmLib from "../../lean"

package sat where
  srcDir := "lean"

@[default_target]
lean_exe generate where
  root := `MakeAlgorithm
  supportInterpreter := true
