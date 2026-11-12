module
public import Bench.Hash
/-! The executable that writes `Bench.Hash`'s artifact. -/

public def main (args : List String) : IO Unit := Bench.Hash.main args
