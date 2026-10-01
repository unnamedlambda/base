module
public import Lz4.Comp
/-! The executable that writes `Lz4.Comp`'s artifacts. -/

public def main (args : List String) : IO Unit := Lz4.Comp.main args
