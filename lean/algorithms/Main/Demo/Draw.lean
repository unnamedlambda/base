module
public import Demo.Draw
/-! The executable that writes `Demo.Draw`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.Draw.main args
