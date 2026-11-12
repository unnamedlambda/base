module
public import Bench.Analytics
/-! The executable that writes `Bench.Analytics`'s artifact. -/

public def main (args : List String) : IO Unit := Bench.Analytics.main args
