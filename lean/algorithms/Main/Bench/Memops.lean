module
public import Bench.Memops
/-! The executable that writes `Bench.Memops`'s artifacts. -/

public def main (args : List String) : IO Unit := Bench.Memops.main args
