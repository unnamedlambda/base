module
public import Bench.System
/-! The executable that writes `Bench.System`'s artifacts. -/

public def main (args : List String) : IO Unit := Bench.System.main args
