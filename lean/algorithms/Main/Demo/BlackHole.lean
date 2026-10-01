module
public import Demo.BlackHole
/-! The executable that writes `Demo.BlackHole`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.BlackHole.main args
