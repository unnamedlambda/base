import Lake
open Lake DSL

package algorithms where
  srcDir := "lean"

@[default_target]
lean_exe generate where
  root := `MakeAlgorithm
  supportInterpreter := true
