import Lake
open Lake DSL

package leaneval where
  srcDir := "lean"

@[default_target]
lean_exe generate where
  root := `LeanEval
  supportInterpreter := true
