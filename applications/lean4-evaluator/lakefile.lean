import Lake
open Lake DSL

require algorithmLib from "../../lean"

package lean4evaluator where
  srcDir := "lean"

@[default_target]
lean_exe generate where
  root := `LeanEval
  supportInterpreter := true
