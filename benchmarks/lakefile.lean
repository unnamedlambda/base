import Lake
open Lake DSL

require algorithmLib from "../lean"

package benchAlgorithms where
  srcDir := "lean"

lean_exe generate_csv where
  root := `CsvBenchAlgorithm
  supportInterpreter := true
