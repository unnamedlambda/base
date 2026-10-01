module
public import Tokenizer.Test
/-! The executable that writes `Tokenizer.Test`'s artifacts. -/

public def main (args : List String) : IO Unit := Tokenizer.Test.main args
