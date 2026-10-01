module
public import Demo.Sat
/-! The executable that writes `Demo.Sat`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.Sat.main args
