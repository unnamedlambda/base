module
public import Bench.Json
/-! The executable that writes `Bench.Json`'s artifact. -/

public def main (args : List String) : IO Unit := Bench.Json.main args
