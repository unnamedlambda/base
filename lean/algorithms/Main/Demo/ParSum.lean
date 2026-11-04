module
public import Demo.ParSum
/-! The executable that writes `Demo.ParSum`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.ParSum.main args
