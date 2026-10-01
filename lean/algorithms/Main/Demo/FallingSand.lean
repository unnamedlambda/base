module
public import Demo.FallingSand
/-! The executable that writes `Demo.FallingSand`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.FallingSand.main args
