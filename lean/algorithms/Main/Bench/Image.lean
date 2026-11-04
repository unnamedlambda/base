module
public import Bench.Image
/-! The executable that writes `Bench.Image`'s artifacts. -/

public def main (args : List String) : IO Unit := Bench.Image.main args
