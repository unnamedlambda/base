import Lake
open Lake DSL

require algorithmLib from "../lean"

package benchAlgorithms where
  srcDir := "lean"

lean_exe generate_csv where
  root := `CsvBenchAlgorithm
  supportInterpreter := true

lean_exe generate_regex where
  root := `RegexBenchAlgorithm
  supportInterpreter := true

lean_exe generate_json where
  root := `JsonBenchAlgorithm
  supportInterpreter := true

lean_exe generate_string_search where
  root := `StringSearchAlgorithm
  supportInterpreter := true

lean_exe generate_wordcount where
  root := `WordCountAlgorithm
  supportInterpreter := true
