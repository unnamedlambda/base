module
public import Demo.SharedSum
/-! The executable that writes `Demo.SharedSum`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.SharedSum.main args
