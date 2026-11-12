module
public import Bench.Proto
/-! The executable that writes `Bench.Proto`'s artifact for a schema. -/

public def main (args : List String) : IO Unit := Bench.Proto.main args
