import Lake
open Lake DSL

require algorithmLib from "../../lean"

package raytrace where
  srcDir := "lean"

@[default_target]
lean_exe generate where
  root := `MakeAlgorithm
  supportInterpreter := true
