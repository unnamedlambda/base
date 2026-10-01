module
public import Bench.Python
/-! The executable that writes `Bench.Python`'s artifacts. -/

public def main (args : List String) : IO Unit := Bench.Python.main args
